"""T023: REER search wrapper module implementation.

Implements DSPy integration wrapper for REER search functionality,
providing structured search signatures and composable modules for
content discovery and context enrichment.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import timezone, datetime
from enum import Enum
import logging
import time
from typing import Any
import random
import re

try:
    from core.candidate_scorer import PerplexityCalculator
except Exception:
    PerplexityCalculator = None  # type: ignore

try:
    import dspy
    from dspy import ChainOfThought, InputField, Module, OutputField, Predict, Signature

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

    # Create fallback classes to prevent import errors
    class MockSignature:
        pass

    class MockModule:
        def __init__(self):
            pass

    def MockField(desc=None):
        return None

    Signature = MockSignature
    InputField = MockField
    OutputField = MockField
    Module = MockModule
    Predict = None
    ChainOfThought = None

try:
    from ..core.exceptions import ScoringError, ValidationError
    from ..core.trace_store import REERTraceStore, TraceRecord
except ImportError:
    # Fallback for standalone usage
    try:
        from core.exceptions import ScoringError, ValidationError
        from core.trace_store import REERTraceStore, TraceRecord
    except ImportError:
        # Create mock classes if imports fail
        class ValidationError(Exception):
            pass

        class ScoringError(Exception):
            pass

        class REERTraceStore:
            async def initialize(self):
                pass

            async def add_trace_event(self, event):
                pass

        class TraceRecord:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy options."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    TRENDING = "trending"
    COMPETITIVE = "competitive"
    HYBRID = "hybrid"


@dataclass
class SearchContext:
    """Context for REER search operations."""

    query: str
    platform: str = "twitter"
    strategy: SearchStrategy = SearchStrategy.HYBRID
    max_results: int = 10
    time_filter: str | None = None  # "day", "week", "month"
    language: str = "en"
    include_metrics: bool = True
    include_sentiment: bool = True
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Individual search result."""

    result_id: str
    content: str
    source: str
    platform: str
    author: str | None = None
    timestamp: datetime | None = None
    engagement_metrics: dict[str, Any] = field(default_factory=dict)
    sentiment_score: float | None = None
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class REERSearchResult:
    """Complete REER search result."""

    search_id: str
    query: str
    strategy: SearchStrategy
    results: list[SearchResult]
    total_found: int
    search_time: float
    context_summary: str = ""
    trends_identified: list[str] = field(default_factory=list)
    search_metadata: dict[str, Any] = field(default_factory=dict)


# --- REER Thinking Utilities ---


def breakdown_thinking(thinking: str) -> list[str]:
    """Split a thinking plan into short segments.

    Heuristic: split by newlines or sentence ends, keep 1–2 sentences per segment.
    """
    # Normalize newlines
    text = thinking.strip()
    if not text:
        return []
    # Prefer newline bullets
    parts = [p.strip(" -•\t") for p in text.splitlines() if p.strip()]
    if len(parts) >= 2:
        return parts
    # Fallback to sentence split
    sents = re.split(r"(?<=[.!?])\s+", text)
    segs: list[str] = []
    buf: list[str] = []
    for s in sents:
        buf.append(s)
        if len(" ".join(buf)) > 80 or len(buf) >= 2:
            segs.append(" ".join(buf).strip())
            buf = []
    if buf:
        segs.append(" ".join(buf).strip())
    return [s for s in segs if s]


_REPHRASE_DICT = {
    "show": ["share", "outline", "demonstrate", "reveal", "present"],
    "explain": ["break down", "clarify", "describe", "elaborate", "illustrate"],
    "start": ["begin", "kick off", "initiate", "launch", "commence"],
    "simple": ["straightforward", "concise", "basic", "clear", "direct"],
    "example": ["sample", "case", "instance", "illustration", "demonstration"],
    "value": ["benefit", "worth", "importance", "significance", "advantage"],
    "summarize": ["outline", "recap", "synthesize", "condense", "abstract"],
    "core": ["main", "central", "key", "primary", "essential"],
    "end": ["finish", "conclude", "complete", "close", "wrap up"],
    "hook": ["opener", "lead", "intro", "grabber", "attention-getter"],
    "improve": ["enhance", "optimize", "refine", "boost", "strengthen"],
    "identify": ["find", "discover", "recognize", "detect", "pinpoint"],
    "apply": ["use", "implement", "employ", "utilize", "leverage"],
    "measure": ["track", "assess", "evaluate", "quantify", "gauge"],
    "focus": ["concentrate", "center", "target", "emphasize", "prioritize"],
    "build": ["create", "develop", "construct", "establish", "form"],
    "need": ["require", "must have", "should have", "demand", "necessitate"],
    "work": ["function", "operate", "perform", "execute", "run"],
    "make": ["create", "produce", "generate", "build", "develop"],
    "use": ["utilize", "employ", "apply", "leverage", "implement"],
}


class _RephraseSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Rephrase a line to keep intent but improve brevity and clarity."""

    line = InputField(desc="Line to rephrase")
    style = InputField(desc="Tone: concise, professional")
    rephrase1 = OutputField(desc="Rephrased option 1 (<= 80 chars)")
    rephrase2 = OutputField(desc="Rephrased option 2 (<= 80 chars)")


def propose_segment_variants(segment: str, k: int = 2) -> list[str]:
    """Produce k rephrasings using MLX directly."""

    # Use MLX directly for variant generation - this works well!
    try:
        from .mlx_variants import propose_segment_variants_mlx

        return propose_segment_variants_mlx(segment, k)
    except ImportError as e:
        logger.error(f"MLX variants module not available: {e}")
        return []


@dataclass
class REERConfig:
    stop_thresh: float = 0.25
    max_steps: int = 5
    num_expansion: int = 2
    ppl_model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"


class REERRefinementProcessor:
    """Iterative local search over thinking steps (z) minimizing log-PPL(y|x,z)."""

    def __init__(self, config: REERConfig):
        self.config = config
        self.ppl = (
            PerplexityCalculator(model_name=config.ppl_model_id)
            if PerplexityCalculator
            else None
        )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self.ppl is None:
            raise ValidationError("PerplexityCalculator unavailable")
        await self.ppl.initialize()
        self._initialized = True

    async def compute_log_ppl(self, x: str, z: str, y: str) -> float:
        await self.initialize()
        return await self.ppl.calculate_conditional_log_perplexity(x, z, y)

    async def refine(self, x: str, z: str, y: str) -> dict[str, Any]:
        """Run REER refinement returning details and best thinking plan.

        Returns dict with keys: z_init, z_best, ppl_init, ppl_best, steps_taken, candidates
        """
        await self.initialize()
        segments = breakdown_thinking(z)
        z_current = z
        ppl_init = await self.compute_log_ppl(x, z_current, y)
        ppl_current = ppl_init

        # Log initial perplexity
        logger.info(
            f"Initial log-PPL: {ppl_init:.3f}, stop threshold: {self.config.stop_thresh}"
        )

        # Don't stop immediately - at least try to refine once
        # Note: Lower perplexity is better

        steps_taken = 0
        all_candidates: list[dict[str, Any]] = []
        for idx in range(min(self.config.max_steps, len(segments))):
            base_segments = breakdown_thinking(z_current)
            if not base_segments:
                break
            target_idx = idx % len(base_segments)
            seg = base_segments[target_idx]
            proposals = propose_segment_variants(seg, self.config.num_expansion)
            logger.debug(
                f"Step {idx}: Generated {len(proposals)} variants for segment {target_idx}"
            )
            best_local = {"ppl": float("inf"), "z": z_current}
            local_cands = []
            for p in proposals:
                new_segments = base_segments[:]
                new_segments[target_idx] = p
                z_new = "\n".join(new_segments)
                ppl_new = await self.compute_log_ppl(x, z_new, y)
                local_cands.append(
                    {"segment_idx": target_idx, "candidate": p, "ppl": ppl_new}
                )
                if ppl_new < best_local["ppl"]:
                    best_local = {"ppl": ppl_new, "z": z_new}
            all_candidates.extend(local_cands)
            # Accept if improved
            if best_local["ppl"] < ppl_current:
                improvement = ppl_current - best_local["ppl"]
                logger.info(
                    f"Step {idx}: Improved! PPL: {ppl_current:.3f} -> {best_local['ppl']:.3f} (delta: {improvement:.3f})"
                )
                z_current = best_local["z"]
                ppl_current = best_local["ppl"]
                steps_taken += 1
                if ppl_current < self.config.stop_thresh:
                    logger.info(
                        f"Reached stop threshold {self.config.stop_thresh}, stopping refinement"
                    )
                    break
            else:
                # No improvement, but continue trying other segments
                logger.debug(
                    f"Step {idx}: No improvement (best: {best_local['ppl']:.3f} vs current: {ppl_current:.3f})"
                )
                pass

        return {
            "z_init": z,
            "z_best": z_current,
            "ppl_init": ppl_init,
            "ppl_best": ppl_current,
            "steps_taken": steps_taken,
            "candidates": all_candidates,
        }

    success: bool = True
    error_message: str | None = None


# DSPy Signatures for REER search operations
class QueryEnhancementSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Enhance search query for better REER search results."""

    original_query = InputField(desc="Original search query")
    platform = InputField(desc="Target platform for search")
    search_intent = InputField(
        desc="Intent of the search (trending, competitive, etc.)"
    )
    context = InputField(desc="Additional context or requirements")

    enhanced_query = OutputField(desc="Improved search query with relevant keywords")
    search_terms = OutputField(desc="List of additional search terms and synonyms")
    hashtags = OutputField(desc="Relevant hashtags to include in search")


class ContentAnalysisSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Analyze search results for insights and patterns."""

    search_results = InputField(desc="Raw search results content")
    query_context = InputField(desc="Original search query and context")
    analysis_focus = InputField(desc="Specific aspects to analyze")

    content_summary = OutputField(desc="Summary of key themes and insights")
    trending_patterns = OutputField(desc="Identified trending patterns and topics")
    engagement_insights = OutputField(desc="Insights about high-engagement content")
    recommendations = OutputField(desc="Recommendations for content creation")


class TrendIdentificationSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Identify trends from search results."""

    content_batch = InputField(desc="Batch of content to analyze for trends")
    platform_context = InputField(desc="Platform-specific context")
    time_window = InputField(desc="Time window for trend analysis")

    trending_topics = OutputField(desc="List of trending topics identified")
    emerging_themes = OutputField(desc="Emerging themes and patterns")
    viral_elements = OutputField(desc="Elements that contribute to viral content")


class QueryEnhancementModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for enhancing search queries."""

    def __init__(self, use_reasoning: bool = True):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        super().__init__()

        if use_reasoning:
            self.enhancer = ChainOfThought(QueryEnhancementSignature)
        else:
            self.enhancer = Predict(QueryEnhancementSignature)

    def forward(self, **kwargs):
        """Enhance search query."""
        return self.enhancer(**kwargs)


class ContentAnalysisModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for analyzing search results."""

    def __init__(self, use_reasoning: bool = True):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        super().__init__()

        if use_reasoning:
            self.analyzer = ChainOfThought(ContentAnalysisSignature)
            self.trend_identifier = ChainOfThought(TrendIdentificationSignature)
        else:
            self.analyzer = Predict(ContentAnalysisSignature)
            self.trend_identifier = Predict(TrendIdentificationSignature)

    def forward(self, **kwargs):
        """Analyze search results."""
        return self.analyzer(**kwargs)

    def identify_trends(self, **kwargs):
        """Identify trends from content."""
        return self.trend_identifier(**kwargs)


class MockREERSearchEngine:
    """Mock REER search engine for demonstration.

    In production, this would integrate with actual REER search backend.
    """

    def __init__(self):
        """Initialize mock search engine."""
        self.search_history: list[dict[str, Any]] = []

        # Mock trending topics by platform
        self.trending_data = {
            "twitter": [
                "AI automation",
                "productivity tips",
                "remote work",
                "machine learning",
                "startup growth",
                "tech trends",
            ],
            "linkedin": [
                "professional development",
                "leadership skills",
                "industry insights",
                "career growth",
                "business strategy",
                "innovation",
            ],
            "instagram": [
                "lifestyle content",
                "visual storytelling",
                "brand aesthetics",
                "creator economy",
                "social impact",
                "behind the scenes",
            ],
        }

        # Mock content templates
        self.content_templates = {
            "AI automation": [
                "Just automated my entire workflow with AI - saved 10 hours this week! Here's how: {details}",
                "AI is changing everything. Here are 5 automation tools every professional should know about: {list}",
                "The future of work is here. AI automation isn't replacing jobs, it's making us more efficient. Thoughts? {cta}",
            ],
            "productivity tips": [
                "Productivity hack that changed my life: {tip}. What's your favorite productivity trick? {cta}",
                "5 productivity apps I can't live without: {list}. Which ones do you use? {cta}",
                "Time management is life management. Here's my framework for maximizing daily productivity: {framework}",
            ],
            "remote work": [
                "Remote work isn't just about working from home - it's about working smarter. Key lessons: {lessons}",
                "Building culture in remote teams: {strategies}. What works for your team? {cta}",
                "The remote work revolution is here to stay. How to thrive in distributed teams: {tips}",
            ],
        }

    async def search(self, context: SearchContext) -> list[SearchResult]:
        """Perform mock search operation.

        Args:
            context: Search context and parameters

        Returns:
            List of mock search results
        """
        await asyncio.sleep(0.1)  # Simulate search latency

        results = []
        platform_trending = self.trending_data.get(context.platform, [])

        # Generate mock results based on query
        for i in range(min(context.max_results, 5)):
            # Select relevant trending topic
            relevant_topic = None
            for topic in platform_trending:
                if any(word in context.query.lower() for word in topic.lower().split()):
                    relevant_topic = topic
                    break

            if not relevant_topic:
                relevant_topic = platform_trending[i % len(platform_trending)]

            # Generate mock content
            templates = self.content_templates.get(
                relevant_topic,
                [
                    "Great insights on {topic}! Here's what I learned: {details}",
                    "Interesting perspective on {topic}. Thoughts? {cta}",
                    "Breaking: New developments in {topic} space. Details: {info}",
                ],
            )

            template = templates[i % len(templates)]
            content = template.format(
                topic=context.query,
                details="[detailed explanation]",
                list="[numbered list]",
                tips="[actionable tips]",
                cta="Let me know your thoughts!",
                framework="[structured approach]",
                lessons="[key learnings]",
                strategies="[proven strategies]",
                info="[relevant information]",
            )

            result = SearchResult(
                result_id=f"result_{i}_{int(time.time())}",
                content=content,
                source=f"mock_source_{i}",
                platform=context.platform,
                author=f"user_{i}",
                timestamp=datetime.now(timezone.utc),
                engagement_metrics={
                    "likes": (i + 1) * 50,
                    "shares": (i + 1) * 10,
                    "comments": (i + 1) * 5,
                    "engagement_rate": 0.05 + (i * 0.01),
                },
                sentiment_score=0.7 + (i * 0.05),
                relevance_score=0.9 - (i * 0.1),
                metadata={
                    "topic": relevant_topic,
                    "search_strategy": context.strategy.value,
                    "mock_result": True,
                },
            )

            results.append(result)

        # Record search
        self.search_history.append(
            {
                "query": context.query,
                "platform": context.platform,
                "strategy": context.strategy.value,
                "results_count": len(results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return results


class REERSearchModule:
    """REER search wrapper module with DSPy integration.

    Provides structured search functionality with query enhancement,
    result analysis, and trend identification using DSPy modules.
    """

    def __init__(self, search_engine: MockREERSearchEngine | None = None):
        """Initialize REER search module.

        Args:
            search_engine: Optional search engine (uses mock if not provided)
        """
        self.search_engine = search_engine or MockREERSearchEngine()

        # Initialize DSPy modules if available
        if DSPY_AVAILABLE:
            self.query_enhancer = QueryEnhancementModule(use_reasoning=True)
            self.content_analyzer = ContentAnalysisModule(use_reasoning=True)
        else:
            self.query_enhancer = None
            self.content_analyzer = None

        # Search state
        self._initialized = False
        self.search_cache: dict[str, REERSearchResult] = {}
        self.trace_store = REERTraceStore()

        logger.info("Initialized REER search module")

    async def initialize(self) -> None:
        """Initialize the search module."""
        if self._initialized:
            return

        logger.info("Initializing REER search module")

        try:
            # Initialize trace store
            await self.trace_store.initialize()

            self._initialized = True
            logger.info("REER search module initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize REER search module: {e}")
            raise ValidationError(f"REER search initialization failed: {e}")

    async def search(
        self,
        query: str,
        platform: str = "twitter",
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        depth: int = 3,
        **kwargs,
    ) -> REERSearchResult:
        """Perform enhanced REER search with DSPy integration.

        Args:
            query: Search query
            platform: Target platform
            strategy: Search strategy
            depth: Search depth (1-5)
            **kwargs: Additional search parameters

        Returns:
            Enhanced search results
        """
        search_id = f"search_{int(time.time())}_{hash(query) % 10000}"
        start_time = time.time()

        logger.info(f"Starting REER search: {search_id} for query '{query}'")

        try:
            await self.initialize()

            # Create search context
            context = SearchContext(
                query=query,
                platform=platform,
                strategy=strategy,
                max_results=depth * 5,  # More results for deeper search
                **kwargs,
            )

            # Step 1: Enhance query using DSPy (if available)
            enhanced_query = query
            search_terms = []
            hashtags = []

            if self.query_enhancer:
                logger.debug("Enhancing search query with DSPy")
                enhancement_result = self.query_enhancer(
                    original_query=query,
                    platform=platform,
                    search_intent=strategy.value,
                    context=f"Deep search with depth {depth}",
                )

                enhanced_query = enhancement_result.enhanced_query
                search_terms = (
                    enhancement_result.search_terms.split(",")
                    if enhancement_result.search_terms
                    else []
                )
                hashtags = (
                    enhancement_result.hashtags.split(",")
                    if enhancement_result.hashtags
                    else []
                )

            # Step 2: Perform search
            logger.debug(f"Performing search with enhanced query: {enhanced_query}")

            # Update context with enhanced query
            enhanced_context = SearchContext(
                query=enhanced_query,
                platform=platform,
                strategy=strategy,
                max_results=context.max_results,
                **kwargs,
            )

            search_results = await self.search_engine.search(enhanced_context)

            # Step 3: Analyze results using DSPy (if available)
            context_summary = ""
            trends_identified = []

            if self.content_analyzer and search_results:
                logger.debug("Analyzing search results with DSPy")

                # Prepare content for analysis
                results_content = "\n".join(
                    [
                        f"Result {i + 1}: {result.content[:200]}..."
                        for i, result in enumerate(search_results[:5])
                    ]
                )

                # Analyze content
                analysis_result = self.content_analyzer(
                    search_results=results_content,
                    query_context=f"Query: {query}, Platform: {platform}",
                    analysis_focus="trends, engagement patterns, content themes",
                )

                context_summary = analysis_result.content_summary

                # Identify trends
                trend_result = self.content_analyzer.identify_trends(
                    content_batch=results_content,
                    platform_context=platform,
                    time_window="recent",
                )

                trends_identified = (
                    trend_result.trending_topics.split(",")
                    if trend_result.trending_topics
                    else []
                )

            # Step 4: Create final result
            search_time = time.time() - start_time

            reer_result = REERSearchResult(
                search_id=search_id,
                query=query,
                strategy=strategy,
                results=search_results,
                total_found=len(search_results),
                search_time=search_time,
                context_summary=context_summary,
                trends_identified=[t.strip() for t in trends_identified if t.strip()],
                search_metadata={
                    "enhanced_query": enhanced_query,
                    "search_terms": search_terms,
                    "hashtags": hashtags,
                    "depth": depth,
                    "platform": platform,
                    "dspy_enhanced": bool(self.query_enhancer),
                    "dspy_analyzed": bool(self.content_analyzer),
                },
                success=True,
            )

            # Cache result
            self.search_cache[search_id] = reer_result

            # Record trace event (simplified for DSPy context)
            # Seed params must match schema (topic, style, length, thread_size)
            seed_params = {
                "topic": query[:80],
                "style": "informational",
                "length": max(1, len(query)),
                "thread_size": 1,
            }

            trace_data = {
                "id": f"search_{search_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_post_id": search_id,
                "seed_params": seed_params,
                "score": (
                    len(search_results) / context.max_results
                    if context.max_results > 0
                    else 0.0
                ),
                "metrics": {
                    "results_count": len(search_results),
                    "search_time": search_time,
                    "trends_count": len(trends_identified),
                },
                "strategy_features": trends_identified[:5],  # Limit features
                "provider": "dspy::reer_search",
                "metadata": {"search_id": search_id, "depth": depth},
            }

            try:
                # Append dict; underlying store validates against TraceRecord
                await self.trace_store.append_trace(trace_data)
            except Exception as trace_error:
                # Don't fail search if tracing fails
                logger.warning(f"Failed to record trace: {trace_error}")

            logger.info(
                f"REER search completed: {search_id} found {len(search_results)} results "
                f"in {search_time:.2f}s"
            )

            return reer_result

        except Exception as e:
            search_time = time.time() - start_time
            logger.exception(f"REER search failed: {search_id}: {e}")

            return REERSearchResult(
                search_id=search_id,
                query=query,
                strategy=strategy,
                results=[],
                total_found=0,
                search_time=search_time,
                success=False,
                error_message=str(e),
            )

    async def search_trending(
        self, platform: str = "twitter", time_window: str = "day", max_results: int = 10
    ) -> REERSearchResult:
        """Search for trending content on platform.

        Args:
            platform: Target platform
            time_window: Time window for trends
            max_results: Maximum results to return

        Returns:
            Trending content search results
        """
        return await self.search(
            query="trending topics",
            platform=platform,
            strategy=SearchStrategy.TRENDING,
            depth=2,
            time_filter=time_window,
            max_results=max_results,
        )

    async def search_competitive(
        self,
        competitor_query: str,
        platform: str = "twitter",
        analysis_focus: str = "engagement",
    ) -> REERSearchResult:
        """Search for competitive content analysis.

        Args:
            competitor_query: Query about competitors or competitive content
            platform: Target platform
            analysis_focus: Focus of competitive analysis

        Returns:
            Competitive content search results
        """
        return await self.search(
            query=competitor_query,
            platform=platform,
            strategy=SearchStrategy.COMPETITIVE,
            depth=3,
            include_metrics=True,
            filters={"analysis_focus": analysis_focus},
        )

    async def search_semantic(
        self, concept: str, platform: str = "twitter", semantic_depth: int = 3
    ) -> REERSearchResult:
        """Perform semantic search for concept-related content.

        Args:
            concept: Concept or topic for semantic search
            platform: Target platform
            semantic_depth: Depth of semantic analysis

        Returns:
            Semantic search results
        """
        return await self.search(
            query=concept,
            platform=platform,
            strategy=SearchStrategy.SEMANTIC,
            depth=semantic_depth,
            include_sentiment=True,
        )

    async def get_search_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent search history.

        Args:
            limit: Maximum number of searches to return

        Returns:
            List of recent searches
        """
        if hasattr(self.search_engine, "search_history"):
            return self.search_engine.search_history[-limit:]
        return []

    async def get_cached_result(self, search_id: str) -> REERSearchResult | None:
        """Get cached search result.

        Args:
            search_id: Search ID to retrieve

        Returns:
            Cached search result if available
        """
        return self.search_cache.get(search_id)

    async def clear_cache(self) -> None:
        """Clear search cache."""
        self.search_cache.clear()
        logger.info("REER search cache cleared")

    async def analyze_search_patterns(
        self, time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Analyze search patterns and trends.

        Args:
            time_window_hours: Time window for analysis

        Returns:
            Search pattern analysis
        """
        try:
            # Get recent search history
            history = await self.get_search_history(limit=50)

            if not history:
                return {"error": "No search history available"}

            # Analyze patterns
            platforms = {}
            strategies = {}
            queries = []

            for search in history:
                platform = search.get("platform", "unknown")
                strategy = search.get("strategy", "unknown")
                query = search.get("query", "")

                platforms[platform] = platforms.get(platform, 0) + 1
                strategies[strategy] = strategies.get(strategy, 0) + 1
                queries.append(query)

            # Identify common terms
            all_words = []
            for query in queries:
                all_words.extend(query.lower().split())

            word_counts = {}
            for word in all_words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1

            common_terms = sorted(
                word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "total_searches": len(history),
                "platform_distribution": platforms,
                "strategy_distribution": strategies,
                "common_terms": dict(common_terms),
                "unique_queries": len(set(queries)),
                "analysis_window_hours": time_window_hours,
            }

        except Exception as e:
            logger.exception(f"Search pattern analysis failed: {e}")
            return {"error": str(e)}

    def is_available(self) -> bool:
        """Check if REER search module is available."""
        return self._initialized

    async def get_module_status(self) -> dict[str, Any]:
        """Get module status and configuration.

        Returns:
            Module status information
        """
        return {
            "initialized": self._initialized,
            "dspy_available": DSPY_AVAILABLE,
            "query_enhancer_enabled": bool(self.query_enhancer),
            "content_analyzer_enabled": bool(self.content_analyzer),
            "cache_size": len(self.search_cache),
            "search_engine_type": type(self.search_engine).__name__,
            "supported_strategies": [s.value for s in SearchStrategy],
            "supported_platforms": ["twitter", "linkedin", "instagram"],
        }
