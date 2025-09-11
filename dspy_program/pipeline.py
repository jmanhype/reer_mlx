"""T022: Main DSPy pipeline orchestrator implementation.

Implements the main DSPy pipeline for end-to-end content generation, optimization,
and performance evaluation. Integrates REER search, content scoring, and GEPA trainer
for comprehensive social media content pipeline.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

try:
    import dspy
    from dspy import ChainOfThought, InputField, Module, OutputField, Predict, Signature
    from dspy.teleprompt import MIPRO, BootstrapFewShot

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
    BootstrapFewShot = None
    MIPRO = None

try:
    from ..core.candidate_scorer import (
        ContentCandidate,
        REERCandidateScorer,
        ScoringMetrics,
    )
    from ..core.exceptions import OptimizationError, ScoringError, ValidationError

    # GEPA moved to DSPy; use runner
    from .gepa_runner import run_gepa
    from ..plugins.dspy_lm import DSPyConfig, DSPyLanguageModelAdapter
except ImportError:
    # Fallback for standalone usage
    try:
        from core.candidate_scorer import (
            ContentCandidate,
            REERCandidateScorer,
            ScoringMetrics,
        )
        from core.exceptions import OptimizationError, ScoringError, ValidationError
        from dspy_program.gepa_runner import run_gepa
        from plugins.dspy_lm import DSPyConfig, DSPyLanguageModelAdapter
    except ImportError:
        # Create mock classes if imports fail
        class ValidationError(Exception):
            pass

        class OptimizationError(Exception):
            pass

        class ScoringError(Exception):
            pass

        class ScoringMetrics:
            def __init__(self):
                self.overall_score = 0.5
                self.engagement_score = 0.5
                self.quality_score = 0.5
                self.viral_potential = 0.5
                self.brand_alignment = 0.5

        class ContentCandidate:
            def __init__(self, candidate_id, text, metadata=None):
                self.candidate_id = candidate_id
                self.text = text
                self.metadata = metadata or {}

        class REERCandidateScorer:
            async def initialize(self):
                pass

            async def score_candidate(self, candidate):
                return ScoringMetrics()

        run_gepa = None  # GEPA not available in fallback

        class DSPyConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class DSPyLanguageModelAdapter:
            def __init__(self, config):
                self.config = config
                self.lm = None

            async def initialize(self):
                pass

            async def get_model_info(self):
                return {"mock": True}


try:
    from .evaluator import KPIEvaluator, PerformanceMetrics
    from .reer_module import REERSearchModule, REERSearchResult
except ImportError:
    # Will be set to None if imports fail
    REERSearchModule = None
    REERSearchResult = None
    KPIEvaluator = None
    PerformanceMetrics = None


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for DSPy pipeline."""

    # Model configuration
    dspy_config: DSPyConfig

    # Pipeline parameters
    max_iterations: int = 5
    quality_threshold: float = 0.8
    diversity_requirement: float = 0.6
    batch_size: int = 10

    # Optimization settings (DSPy GEPA)
    use_optimization: bool = True
    gepa_auto: str = "light"  # light|medium|heavy
    gepa_reflection_model: str = "gpt-4o"
    gepa_use_perplexity: bool = False

    # Search configuration
    enable_reer_search: bool = True
    search_depth: int = 3

    # Evaluation settings
    enable_kpi_evaluation: bool = True
    evaluation_metrics: list[str] = field(
        default_factory=lambda: [
            "engagement_rate",
            "quality_score",
            "viral_potential",
            "brand_alignment",
        ]
    )

    # Output settings
    save_intermediate_results: bool = True
    output_directory: Path | None = None


@dataclass
class ContentRequest:
    """Request for content generation."""

    request_id: str
    topic: str
    platform: str = "twitter"  # twitter, linkedin, instagram
    audience: str = "general"
    style: str = "professional"
    requirements: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    optimization_target: dict[str, Any] | None = None


@dataclass
class ContentResult:
    """Result of content generation pipeline."""

    request_id: str
    content: str
    metadata: dict[str, Any]
    scores: ScoringMetrics
    search_results: REERSearchResult | None = None
    optimization_result: Any | None = None
    performance_metrics: PerformanceMetrics | None = None
    generation_time: float = 0.0
    success: bool = True
    error_message: str | None = None


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""

    pipeline_id: str
    results: list[ContentResult]
    summary_metrics: dict[str, Any]
    total_time: float
    success_rate: float
    best_content: ContentResult | None = None


# DSPy Signatures
class ContentGenerationSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Generate social media content based on requirements."""

    topic = InputField(desc="Content topic or theme")
    platform = InputField(desc="Target social media platform")
    audience = InputField(desc="Target audience description")
    style = InputField(desc="Content style and tone")
    requirements = InputField(desc="Specific content requirements")

    content = OutputField(desc="Generated social media content")
    reasoning = OutputField(desc="Explanation of content choices")


class ContentRefinementSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Refine content based on feedback and scores."""

    original_content = InputField(desc="Original generated content")
    feedback = InputField(desc="Feedback and improvement suggestions")
    target_metrics = InputField(desc="Target performance metrics")
    platform_constraints = InputField(desc="Platform-specific constraints")

    refined_content = OutputField(desc="Improved content version")
    improvements = OutputField(desc="List of improvements made")


class ContentGeneratorModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for content generation."""

    def __init__(self, use_reasoning: bool = True):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        super().__init__()

        if use_reasoning:
            self.generator = ChainOfThought(ContentGenerationSignature)
            self.refiner = ChainOfThought(ContentRefinementSignature)
        else:
            self.generator = Predict(ContentGenerationSignature)
            self.refiner = Predict(ContentRefinementSignature)

    def forward(self, **kwargs):
        """Generate content using DSPy."""
        return self.generator(**kwargs)

    def refine(self, **kwargs):
        """Refine content using DSPy."""
        return self.refiner(**kwargs)


class REERDSPyPipeline:
    """Main DSPy pipeline orchestrator for content generation and optimization.

    Integrates DSPy modules, REER search, content scoring, and GEPA optimization
    for end-to-end social media content generation and optimization.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for pipeline functionality")

        self.config = config
        self.pipeline_id = f"pipeline_{int(datetime.now().timestamp())}"

        # Initialize components
        self.dspy_adapter = DSPyLanguageModelAdapter(config.dspy_config)
        self.content_generator = ContentGeneratorModule(use_reasoning=True)

        # Initialize REER search module
        if config.enable_reer_search:
            self.reer_search = REERSearchModule()
        else:
            self.reer_search = None

        # Initialize scorer
        self.scorer = REERCandidateScorer()

        # GEPA is available via DSPy runner; flag only
        self.gepa_enabled = bool(config.use_optimization)

        # Initialize KPI evaluator
        if config.enable_kpi_evaluation:
            self.kpi_evaluator = KPIEvaluator(metrics=config.evaluation_metrics)
        else:
            self.kpi_evaluator = None

        # Pipeline state
        self._initialized = False
        self.execution_history: list[dict[str, Any]] = []

        logger.info(f"Initialized DSPy pipeline {self.pipeline_id}")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("Initializing DSPy pipeline components")

        try:
            # Initialize DSPy adapter
            await self.dspy_adapter.initialize()

            # Configure DSPy settings
            dspy.settings.configure(lm=self.dspy_adapter.lm)

            # Initialize scorer
            await self.scorer.initialize()

            # Initialize REER search if enabled
            if self.reer_search:
                await self.reer_search.initialize()

            # Initialize KPI evaluator if enabled
            if self.kpi_evaluator:
                await self.kpi_evaluator.initialize()

            self._initialized = True
            logger.info("DSPy pipeline initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize pipeline: {e}")
            raise ValidationError(f"Pipeline initialization failed: {e}")

    async def generate_content(self, request: ContentRequest) -> ContentResult:
        """Generate content for a single request.

        Args:
            request: Content generation request

        Returns:
            Content generation result
        """
        start_time = time.time()
        logger.info(f"Generating content for request {request.request_id}")

        try:
            await self.initialize()

            # Step 1: REER search for context (if enabled)
            search_results = None
            search_context = ""

            if self.reer_search:
                logger.debug("Performing REER search for context")
                search_results = await self.reer_search.search(
                    query=request.topic,
                    platform=request.platform,
                    depth=self.config.search_depth,
                )
                search_context = self._format_search_context(search_results)

            # Step 2: Generate initial content
            logger.debug("Generating initial content with DSPy")
            generation_result = self.content_generator(
                topic=request.topic,
                platform=request.platform,
                audience=request.audience,
                style=request.style,
                requirements=self._format_requirements(request, search_context),
            )

            initial_content = generation_result.content

            # Step 3: Score initial content
            logger.debug("Scoring generated content")
            candidate = ContentCandidate(
                candidate_id=f"{request.request_id}_initial",
                text=initial_content,
                metadata={
                    "topic": request.topic,
                    "platform": request.platform,
                    "audience": request.audience,
                    "style": request.style,
                    "search_context": bool(search_context),
                },
            )

            scores = await self.scorer.score_candidate(candidate)

            # Step 4: Iterative refinement
            refined_content = initial_content
            refined_scores = scores

            for iteration in range(self.config.max_iterations):
                if refined_scores.overall_score >= self.config.quality_threshold:
                    logger.debug(f"Quality threshold reached at iteration {iteration}")
                    break

                logger.debug(f"Refining content - iteration {iteration + 1}")

                # Generate feedback based on scores
                feedback = self._generate_feedback(refined_scores, request)

                # Refine content
                refinement_result = self.content_generator.refine(
                    original_content=refined_content,
                    feedback=feedback,
                    target_metrics=self._format_target_metrics(request),
                    platform_constraints=self._format_platform_constraints(
                        request.platform
                    ),
                )

                refined_content = refinement_result.refined_content

                # Score refined content
                refined_candidate = ContentCandidate(
                    candidate_id=f"{request.request_id}_refined_{iteration}",
                    text=refined_content,
                    metadata=candidate.metadata,
                )

                refined_scores = await self.scorer.score_candidate(refined_candidate)

                # Check for improvement
                if refined_scores.overall_score <= scores.overall_score:
                    logger.debug("No improvement, stopping refinement")
                    break

                scores = refined_scores

            # Step 5: KPI evaluation (if enabled)
            performance_metrics = None
            if self.kpi_evaluator:
                logger.debug("Evaluating KPI performance")
                performance_metrics = await self.kpi_evaluator.evaluate(
                    content=refined_content,
                    metadata={
                        "platform": request.platform,
                        "topic": request.topic,
                        "audience": request.audience,
                    },
                )

            # Create result
            generation_time = time.time() - start_time

            result = ContentResult(
                request_id=request.request_id,
                content=refined_content,
                metadata={
                    "topic": request.topic,
                    "platform": request.platform,
                    "audience": request.audience,
                    "style": request.style,
                    "iterations": iteration + 1,
                    "reasoning": getattr(generation_result, "reasoning", ""),
                    "search_enabled": bool(search_results),
                },
                scores=scores,
                search_results=search_results,
                performance_metrics=performance_metrics,
                generation_time=generation_time,
                success=True,
            )

            logger.info(
                f"Content generated successfully for {request.request_id} "
                f"in {generation_time:.2f}s with score {scores.overall_score:.3f}"
            )

            return result

        except Exception as e:
            generation_time = time.time() - start_time
            logger.exception(f"Content generation failed for {request.request_id}: {e}")

            return ContentResult(
                request_id=request.request_id,
                content="",
                metadata={},
                scores=ScoringMetrics(),
                generation_time=generation_time,
                success=False,
                error_message=str(e),
            )

    async def generate_batch(self, requests: list[ContentRequest]) -> PipelineResult:
        """Generate content for multiple requests.

        Args:
            requests: List of content generation requests

        Returns:
            Pipeline execution result
        """
        start_time = time.time()
        logger.info(f"Starting batch generation for {len(requests)} requests")

        results = []
        success_count = 0

        # Process requests in batches
        for i in range(0, len(requests), self.config.batch_size):
            batch = requests[i : i + self.config.batch_size]
            logger.debug(f"Processing batch {i // self.config.batch_size + 1}")

            # Generate content for batch
            batch_tasks = [self.generate_content(request) for request in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch generation error: {result}")
                    continue

                results.append(result)
                if result.success:
                    success_count += 1

                # Save intermediate results if configured
                if (
                    self.config.save_intermediate_results
                    and self.config.output_directory
                ):
                    await self._save_intermediate_result(result)

        # Calculate summary metrics
        total_time = time.time() - start_time
        success_rate = success_count / len(requests) if requests else 0.0

        summary_metrics = self._calculate_summary_metrics(results)

        # Find best content
        best_content = None
        if results:
            best_content = max(
                [r for r in results if r.success],
                key=lambda r: r.scores.overall_score,
                default=None,
            )

        pipeline_result = PipelineResult(
            pipeline_id=self.pipeline_id,
            results=results,
            summary_metrics=summary_metrics,
            total_time=total_time,
            success_rate=success_rate,
            best_content=best_content,
        )

        logger.info(
            f"Batch generation completed: {success_count}/{len(requests)} successful "
            f"in {total_time:.2f}s (success rate: {success_rate:.1%})"
        )

        return pipeline_result

    async def optimize_with_gepa(
        self,
        optimization_target: dict[str, Any],
        content_requests: list[ContentRequest] | None = None,
    ) -> Any:
        """Optimize content generation using DSPy GEPA.

        Args:
            optimization_target: Optimization target parameters
            content_requests: Optional requests for context

        Returns:
            GEPA optimization result
        """
        if not self.gepa_enabled:
            raise ValidationError("GEPA optimization not enabled")

        logger.info("Starting GEPA optimization")

        try:
            await self.initialize()

            # Build train tasks from content requests or target
            if content_requests:
                train = [
                    {"topic": r.topic, "audience": r.audience} for r in content_requests
                ]
            else:
                train = [
                    {
                        "topic": optimization_target.get("topic", ""),
                        "audience": optimization_target.get("audience", "general"),
                    }
                ]

            # Choose gen model from adapter config if available
            gen_model = None
            if hasattr(self.config.dspy_config, "model"):
                gen_model = getattr(self.config.dspy_config, "model")

            optimized_program = run_gepa(
                train,
                val_tasks=None,
                gen_model=gen_model or "mlx-community/Llama-3.2-3B-Instruct-4bit",
                reflection_model=self.config.gepa_reflection_model,
                auto=self.config.gepa_auto,
                track_stats=True,
                use_cot=True,
                use_perplexity=self.config.gepa_use_perplexity,
            )

            logger.info("GEPA optimization (DSPy) completed")

            return optimized_program

        except Exception as e:
            logger.exception(f"GEPA optimization failed: {e}")
            raise OptimizationError(f"GEPA optimization failed: {e}")

    def _format_search_context(self, search_results: REERSearchResult) -> str:
        """Format REER search results as context."""
        if not search_results or not search_results.results:
            return ""

        context_parts = ["Relevant context from search:"]

        for i, result in enumerate(search_results.results[:3], 1):  # Top 3 results
            context_parts.append(f"{i}. {result.content[:200]}...")

        return "\n".join(context_parts)

    def _format_requirements(
        self, request: ContentRequest, search_context: str = ""
    ) -> str:
        """Format content requirements for DSPy."""
        requirements = [
            f"Platform: {request.platform}",
            f"Audience: {request.audience}",
            f"Style: {request.style}",
        ]

        # Add specific requirements
        for key, value in request.requirements.items():
            requirements.append(f"{key}: {value}")

        # Add constraints
        if request.constraints:
            requirements.append("Constraints:")
            for key, value in request.constraints.items():
                requirements.append(f"- {key}: {value}")

        # Add search context
        if search_context:
            requirements.append(f"\n{search_context}")

        return "\n".join(requirements)

    def _format_target_metrics(self, request: ContentRequest) -> str:
        """Format target metrics for refinement."""
        targets = [
            f"Quality score: >= {self.config.quality_threshold}",
            "High engagement potential",
            "Platform-appropriate format",
            "Audience-relevant content",
        ]

        if request.optimization_target:
            for key, value in request.optimization_target.items():
                targets.append(f"{key}: {value}")

        return "\n".join(targets)

    def _format_platform_constraints(self, platform: str) -> str:
        """Format platform-specific constraints."""
        constraints = {
            "twitter": [
                "Character limit: 280 characters",
                "Use relevant hashtags (2-3 max)",
                "Engaging and concise",
                "Include call-to-action if appropriate",
            ],
            "linkedin": [
                "Professional tone",
                "Longer form content allowed",
                "Industry-relevant hashtags",
                "Professional call-to-action",
            ],
            "instagram": [
                "Visual-first content",
                "Engaging caption",
                "Multiple hashtags allowed",
                "Storytelling approach",
            ],
        }

        return "\n".join(constraints.get(platform, []))

    def _generate_feedback(
        self, scores: ScoringMetrics, request: ContentRequest
    ) -> str:
        """Generate improvement feedback based on scores."""
        feedback_parts = ["Improvement suggestions:"]

        if scores.engagement_score < 0.7:
            feedback_parts.append(
                "- Increase engagement with questions or calls-to-action"
            )

        if scores.quality_score < 0.7:
            feedback_parts.append("- Improve content quality and clarity")

        if scores.viral_potential < 0.6:
            feedback_parts.append("- Add elements that increase shareability")

        if scores.brand_alignment < 0.7:
            feedback_parts.append("- Better align with brand voice and values")

        if scores.fluency_score < 0.8:
            feedback_parts.append("- Improve language fluency and readability")

        # Platform-specific feedback
        if request.platform == "twitter" and len(scores.text_length or "") > 280:
            feedback_parts.append("- Reduce length to fit Twitter character limit")

        return "\n".join(feedback_parts)

    def _calculate_summary_metrics(
        self, results: list[ContentResult]
    ) -> dict[str, Any]:
        """Calculate summary metrics from results."""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                "total_requests": len(results),
                "successful_generations": 0,
                "success_rate": 0.0,
            }

        # Calculate averages
        avg_scores = {
            "avg_overall_score": sum(r.scores.overall_score for r in successful_results)
            / len(successful_results),
            "avg_engagement_score": sum(
                r.scores.engagement_score for r in successful_results
            )
            / len(successful_results),
            "avg_quality_score": sum(r.scores.quality_score for r in successful_results)
            / len(successful_results),
            "avg_viral_potential": sum(
                r.scores.viral_potential for r in successful_results
            )
            / len(successful_results),
            "avg_generation_time": sum(r.generation_time for r in successful_results)
            / len(successful_results),
        }

        # Find best scores
        best_scores = {
            "best_overall_score": max(
                r.scores.overall_score for r in successful_results
            ),
            "best_engagement_score": max(
                r.scores.engagement_score for r in successful_results
            ),
            "best_quality_score": max(
                r.scores.quality_score for r in successful_results
            ),
            "best_viral_potential": max(
                r.scores.viral_potential for r in successful_results
            ),
        }

        return {
            "total_requests": len(results),
            "successful_generations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            **avg_scores,
            **best_scores,
        }

    async def _save_intermediate_result(self, result: ContentResult) -> None:
        """Save intermediate result to file."""
        if not self.config.output_directory:
            return

        try:
            output_dir = self.config.output_directory
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"content_{result.request_id}_{int(datetime.now().timestamp())}.json"
            )
            filepath = output_dir / filename

            # Convert to serializable format
            data = {
                "request_id": result.request_id,
                "content": result.content,
                "metadata": result.metadata,
                "scores": {
                    "overall_score": result.scores.overall_score,
                    "engagement_score": result.scores.engagement_score,
                    "quality_score": result.scores.quality_score,
                    "viral_potential": result.scores.viral_potential,
                    "brand_alignment": result.scores.brand_alignment,
                },
                "generation_time": result.generation_time,
                "success": result.success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save intermediate result: {e}")

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status and configuration.

        Returns:
            Pipeline status information
        """
        return {
            "pipeline_id": self.pipeline_id,
            "initialized": self._initialized,
            "config": {
                "max_iterations": self.config.max_iterations,
                "quality_threshold": self.config.quality_threshold,
                "batch_size": self.config.batch_size,
                "use_optimization": self.config.use_optimization,
                "enable_reer_search": self.config.enable_reer_search,
                "enable_kpi_evaluation": self.config.enable_kpi_evaluation,
            },
            "components": {
                "dspy_adapter": (
                    await self.dspy_adapter.get_model_info()
                    if self.dspy_adapter
                    else None
                ),
                "reer_search": bool(self.reer_search),
                "gepa_enabled": bool(self.gepa_enabled),
                "kpi_evaluator": bool(self.kpi_evaluator),
            },
            "execution_history_count": len(self.execution_history),
        }


class PipelineFactory:
    """Factory for creating DSPy pipelines with different configurations."""

    @staticmethod
    def create_basic_pipeline(
        provider: str = "openai", model: str = "gpt-3.5-turbo", **kwargs
    ) -> REERDSPyPipeline:
        """Create a basic content generation pipeline.

        Args:
            provider: LM provider
            model: Model name
            **kwargs: Additional configuration

        Returns:
            Configured pipeline
        """
        dspy_config = DSPyConfig(
            provider=provider, model=model, reasoning_mode=True, **kwargs
        )

        pipeline_config = PipelineConfig(
            dspy_config=dspy_config,
            enable_reer_search=False,
            use_optimization=False,
            enable_kpi_evaluation=False,
        )

        return REERDSPyPipeline(pipeline_config)

    @staticmethod
    def create_full_pipeline(
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        output_directory: Path | None = None,
        **kwargs,
    ) -> REERDSPyPipeline:
        """Create a full-featured pipeline with all components.

        Args:
            provider: LM provider
            model: Model name
            output_directory: Optional output directory
            **kwargs: Additional configuration

        Returns:
            Fully configured pipeline
        """
        dspy_config = DSPyConfig(
            provider=provider, model=model, reasoning_mode=True, **kwargs
        )

        pipeline_config = PipelineConfig(
            dspy_config=dspy_config,
            enable_reer_search=True,
            use_optimization=True,
            enable_kpi_evaluation=True,
            save_intermediate_results=True,
            output_directory=output_directory,
        )

        return REERDSPyPipeline(pipeline_config)

    @staticmethod
    def create_optimization_pipeline(
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        **kwargs,
    ) -> REERDSPyPipeline:
        """Create a pipeline optimized for GEPA training.

        Args:
            provider: LM provider
            model: Model name
            optimization_config: GEPA optimization configuration
            **kwargs: Additional configuration

        Returns:
            Optimization-focused pipeline
        """
        dspy_config = DSPyConfig(
            provider=provider, model=model, reasoning_mode=True, **kwargs
        )

        pipeline_config = PipelineConfig(
            dspy_config=dspy_config,
            enable_reer_search=True,
            use_optimization=True,
            enable_kpi_evaluation=True,
            max_iterations=3,
            quality_threshold=0.9,
        )

        return REERDSPyPipeline(pipeline_config)
