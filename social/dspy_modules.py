"""T026: Social-specific DSPy modules for content generation

Implements DSPy signatures and modules specifically designed for social media
content ideation, composition, and optimization using the REER framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any

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


logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of social media content."""

    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"
    ENTERTAINING = "entertaining"
    NEWS = "news"
    COMMUNITY = "community"
    BEHIND_SCENES = "behind_scenes"


class Platform(Enum):
    """Supported social media platforms."""

    X = "x"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"


@dataclass
class ContentBrief:
    """Content generation brief."""

    topic: str
    platform: Platform
    content_type: ContentType
    target_audience: str
    key_message: str
    tone: str = "professional"
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    character_limit: int | None = None
    include_cta: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


if DSPY_AVAILABLE:

    class IdeateSignature(Signature):
        """DSPy signature for social media content ideation."""

        topic = InputField(desc="Main topic or subject for content ideation")
        platform = InputField(
            desc="Target social media platform (x, linkedin, facebook, instagram, tiktok)"
        )
        content_type = InputField(
            desc="Type of content (promotional, educational, entertaining, news, community, behind_scenes)"
        )
        target_audience = InputField(desc="Target audience demographics and interests")
        tone = InputField(
            desc="Desired tone of voice (professional, casual, humorous, inspirational, etc.)"
        )
        context = InputField(
            desc="Additional context, trends, or specific requirements"
        )

        content_ideas = OutputField(
            desc="List of 3-5 creative content ideas with brief descriptions"
        )
        hook_suggestions = OutputField(
            desc="Attention-grabbing opening lines or hooks for each idea"
        )
        hashtag_recommendations = OutputField(
            desc="Relevant hashtags for discoverability"
        )
        engagement_tactics = OutputField(
            desc="Specific tactics to encourage audience engagement"
        )
        timing_suggestions = OutputField(desc="Optimal posting timing recommendations")

    class ComposeSignature(Signature):
        """DSPy signature for social media content composition."""

        content_idea = InputField(
            desc="Selected content idea to develop into full post"
        )
        platform = InputField(
            desc="Target social media platform with specific formatting requirements"
        )
        character_limit = InputField(
            desc="Character limit for the platform (e.g., 280 for X, 3000 for LinkedIn)"
        )
        key_message = InputField(
            desc="Core message or value proposition to communicate"
        )
        target_audience = InputField(
            desc="Specific audience to tailor language and approach"
        )
        tone = InputField(desc="Tone of voice to maintain throughout the content")
        include_cta = InputField(
            desc="Whether to include a call-to-action (true/false)"
        )
        hashtags = InputField(desc="Specific hashtags to include or hashtag strategy")
        mentions = InputField(desc="Accounts to mention or mention strategy")

        post_content = OutputField(desc="Complete, ready-to-publish social media post")
        alternative_versions = OutputField(
            desc="2-3 alternative versions with different approaches"
        )
        optimal_hashtags = OutputField(
            desc="Final hashtag selection optimized for reach and engagement"
        )
        engagement_prediction = OutputField(
            desc="Predicted engagement level and reasoning"
        )
        improvement_suggestions = OutputField(
            desc="Suggestions for further optimization"
        )

    class OptimizeSignature(Signature):
        """DSPy signature for content optimization based on performance data."""

        original_content = InputField(desc="Original social media post content")
        performance_metrics = InputField(
            desc="Engagement metrics (likes, shares, comments, reach, etc.)"
        )
        audience_feedback = InputField(
            desc="Comments, reactions, and audience responses"
        )
        platform = InputField(desc="Platform where content was published")
        goal = InputField(
            desc="Optimization goal (increase engagement, improve reach, drive clicks, etc.)"
        )

        optimization_analysis = OutputField(
            desc="Analysis of what worked and what didn't"
        )
        improved_content = OutputField(desc="Optimized version of the content")
        strategy_adjustments = OutputField(
            desc="Recommended adjustments to content strategy"
        )
        future_recommendations = OutputField(
            desc="Recommendations for future content in similar topics"
        )

    class TrendAnalysisSignature(Signature):
        """DSPy signature for social media trend analysis."""

        trending_topics = InputField(
            desc="Current trending topics, hashtags, or themes"
        )
        platform = InputField(desc="Platform where trends are observed")
        industry = InputField(desc="Industry or niche context")
        brand_voice = InputField(desc="Brand voice and messaging guidelines")
        content_history = InputField(desc="Previous content performance and themes")

        trend_relevance = OutputField(
            desc="Analysis of which trends are relevant to the brand"
        )
        content_opportunities = OutputField(
            desc="Specific content opportunities based on trends"
        )
        risk_assessment = OutputField(
            desc="Potential risks or considerations for trend participation"
        )
        timing_strategy = OutputField(
            desc="Optimal timing strategy for trend-based content"
        )

    class SocialContentIdeator(Module):
        """DSPy module for social media content ideation."""

        def __init__(self):
            super().__init__()
            self.ideate = ChainOfThought(IdeateSignature)

        def forward(self, content_brief: ContentBrief) -> dict[str, Any]:
            """
            Generate content ideas based on the brief.

            Args:
                content_brief: Content generation brief

            Returns:
                Dictionary with content ideas and recommendations
            """
            try:
                context_str = (
                    f"Additional context: {content_brief.metadata}"
                    if content_brief.metadata
                    else ""
                )

                result = self.ideate(
                    topic=content_brief.topic,
                    platform=content_brief.platform.value,
                    content_type=content_brief.content_type.value,
                    target_audience=content_brief.target_audience,
                    tone=content_brief.tone,
                    context=context_str,
                )

                return {
                    "content_ideas": result.content_ideas,
                    "hook_suggestions": result.hook_suggestions,
                    "hashtag_recommendations": result.hashtag_recommendations,
                    "engagement_tactics": result.engagement_tactics,
                    "timing_suggestions": result.timing_suggestions,
                    "brief": content_brief,
                }

            except Exception as e:
                logger.exception(f"Content ideation failed: {e}")
                return {"error": str(e), "content_ideas": [], "brief": content_brief}

    class SocialContentComposer(Module):
        """DSPy module for social media content composition."""

        def __init__(self):
            super().__init__()
            self.compose = ChainOfThought(ComposeSignature)

        def forward(
            self, content_idea: str, content_brief: ContentBrief
        ) -> dict[str, Any]:
            """
            Compose complete social media content.

            Args:
                content_idea: Selected content idea to develop
                content_brief: Content generation brief

            Returns:
                Dictionary with composed content and metadata
            """
            try:
                # Platform-specific character limits
                char_limits = {
                    Platform.X: 280,
                    Platform.LINKEDIN: 3000,
                    Platform.FACEBOOK: 63206,
                    Platform.INSTAGRAM: 2200,
                    Platform.TIKTOK: 2200,
                }

                character_limit = content_brief.character_limit or char_limits.get(
                    content_brief.platform, 280
                )

                result = self.compose(
                    content_idea=content_idea,
                    platform=content_brief.platform.value,
                    character_limit=str(character_limit),
                    key_message=content_brief.key_message,
                    target_audience=content_brief.target_audience,
                    tone=content_brief.tone,
                    include_cta=str(content_brief.include_cta),
                    hashtags=(
                        ", ".join(content_brief.hashtags)
                        if content_brief.hashtags
                        else "None specified"
                    ),
                    mentions=(
                        ", ".join(content_brief.mentions)
                        if content_brief.mentions
                        else "None specified"
                    ),
                )

                return {
                    "post_content": result.post_content,
                    "alternative_versions": result.alternative_versions,
                    "optimal_hashtags": result.optimal_hashtags,
                    "engagement_prediction": result.engagement_prediction,
                    "improvement_suggestions": result.improvement_suggestions,
                    "character_count": len(result.post_content),
                    "brief": content_brief,
                    "content_idea": content_idea,
                }

            except Exception as e:
                logger.exception(f"Content composition failed: {e}")
                return {"error": str(e), "post_content": "", "brief": content_brief}

    class SocialContentOptimizer(Module):
        """DSPy module for content optimization based on performance."""

        def __init__(self):
            super().__init__()
            self.optimize = ChainOfThought(OptimizeSignature)

        def forward(
            self,
            content: str,
            metrics: dict[str, Any],
            platform: Platform,
            goal: str = "increase engagement",
        ) -> dict[str, Any]:
            """
            Optimize content based on performance data.

            Args:
                content: Original content to optimize
                metrics: Performance metrics data
                platform: Social media platform
                goal: Optimization goal

            Returns:
                Dictionary with optimization analysis and recommendations
            """
            try:
                # Format metrics for prompt
                metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])

                # Extract audience feedback if available
                feedback_str = metrics.get("comments", "No feedback available")
                if isinstance(feedback_str, list):
                    feedback_str = "; ".join(feedback_str[:5])  # Top 5 comments

                result = self.optimize(
                    original_content=content,
                    performance_metrics=metrics_str,
                    audience_feedback=str(feedback_str),
                    platform=platform.value,
                    goal=goal,
                )

                return {
                    "optimization_analysis": result.optimization_analysis,
                    "improved_content": result.improved_content,
                    "strategy_adjustments": result.strategy_adjustments,
                    "future_recommendations": result.future_recommendations,
                    "original_content": content,
                    "metrics": metrics,
                    "platform": platform.value,
                }

            except Exception as e:
                logger.exception(f"Content optimization failed: {e}")
                return {"error": str(e), "original_content": content}

    class SocialTrendAnalyzer(Module):
        """DSPy module for social media trend analysis."""

        def __init__(self):
            super().__init__()
            self.analyze = ChainOfThought(TrendAnalysisSignature)

        def forward(
            self,
            trending_topics: list[str],
            platform: Platform,
            industry: str,
            brand_voice: str,
            content_history: str = "",
        ) -> dict[str, Any]:
            """
            Analyze trends for content opportunities.

            Args:
                trending_topics: List of current trending topics
                platform: Social media platform
                industry: Brand industry or niche
                brand_voice: Brand voice guidelines
                content_history: Previous content performance summary

            Returns:
                Dictionary with trend analysis and recommendations
            """
            try:
                trends_str = ", ".join(trending_topics)

                result = self.analyze(
                    trending_topics=trends_str,
                    platform=platform.value,
                    industry=industry,
                    brand_voice=brand_voice,
                    content_history=content_history,
                )

                return {
                    "trend_relevance": result.trend_relevance,
                    "content_opportunities": result.content_opportunities,
                    "risk_assessment": result.risk_assessment,
                    "timing_strategy": result.timing_strategy,
                    "analyzed_trends": trending_topics,
                    "platform": platform.value,
                    "industry": industry,
                }

            except Exception as e:
                logger.exception(f"Trend analysis failed: {e}")
                return {"error": str(e), "analyzed_trends": trending_topics}

    class SocialContentPipeline(Module):
        """Complete social media content pipeline combining all modules."""

        def __init__(self):
            super().__init__()
            self.ideator = SocialContentIdeator()
            self.composer = SocialContentComposer()
            self.optimizer = SocialContentOptimizer()
            self.trend_analyzer = SocialTrendAnalyzer()

        def generate_content(
            self,
            content_brief: ContentBrief,
            trending_topics: list[str] | None = None,
        ) -> dict[str, Any]:
            """
            Complete content generation pipeline.

            Args:
                content_brief: Content generation brief
                trending_topics: Optional trending topics to consider

            Returns:
                Complete content generation results
            """
            results = {"timestamp": datetime.now().isoformat(), "brief": content_brief}

            try:
                # Step 1: Trend analysis (if trends provided)
                if trending_topics:
                    trend_analysis = self.trend_analyzer(
                        trending_topics=trending_topics,
                        platform=content_brief.platform,
                        industry=content_brief.metadata.get("industry", "general"),
                        brand_voice=content_brief.tone,
                        content_history=content_brief.metadata.get(
                            "content_history", ""
                        ),
                    )
                    results["trend_analysis"] = trend_analysis

                # Step 2: Content ideation
                ideation_results = self.ideator(content_brief)
                results["ideation"] = ideation_results

                if "error" in ideation_results:
                    return results

                # Step 3: Content composition for top idea
                content_ideas = ideation_results.get("content_ideas", [])
                if content_ideas:
                    # Use first idea or extract from string
                    first_idea = (
                        content_ideas[0]
                        if isinstance(content_ideas, list)
                        else str(content_ideas).split("\n")[0]
                    )

                    composition_results = self.composer(first_idea, content_brief)
                    results["composition"] = composition_results

                return results

            except Exception as e:
                logger.exception(f"Content pipeline failed: {e}")
                results["error"] = str(e)
                return results

        def optimize_existing_content(
            self,
            content: str,
            metrics: dict[str, Any],
            platform: Platform,
            goal: str = "increase engagement",
        ) -> dict[str, Any]:
            """
            Optimize existing content based on performance.

            Args:
                content: Original content
                metrics: Performance metrics
                platform: Social media platform
                goal: Optimization goal

            Returns:
                Optimization results
            """
            return self.optimizer(content, metrics, platform, goal)

else:
    # Fallback implementations when DSPy is not available
    class SocialContentIdeator:
        def __init__(self):
            logger.warning("DSPy not available, using fallback implementation")

        def forward(self, content_brief: ContentBrief) -> dict[str, Any]:
            return {
                "error": "DSPy not available",
                "content_ideas": [
                    "Basic content idea based on topic: " + content_brief.topic
                ],
                "brief": content_brief,
            }

    class SocialContentComposer:
        def __init__(self):
            logger.warning("DSPy not available, using fallback implementation")

        def forward(
            self, content_idea: str, content_brief: ContentBrief
        ) -> dict[str, Any]:
            return {
                "error": "DSPy not available",
                "post_content": f"Basic post about {content_brief.topic}",
                "brief": content_brief,
            }

    class SocialContentOptimizer:
        def __init__(self):
            logger.warning("DSPy not available, using fallback implementation")

        def forward(
            self,
            content: str,
            metrics: dict[str, Any],
            platform: Platform,
            goal: str = "increase engagement",
        ) -> dict[str, Any]:
            return {"error": "DSPy not available", "original_content": content}

    class SocialTrendAnalyzer:
        def __init__(self):
            logger.warning("DSPy not available, using fallback implementation")

        def forward(
            self,
            trending_topics: list[str],
            platform: Platform,
            industry: str,
            brand_voice: str,
            content_history: str = "",
        ) -> dict[str, Any]:
            return {"error": "DSPy not available", "analyzed_trends": trending_topics}

    class SocialContentPipeline:
        def __init__(self):
            logger.warning("DSPy not available, using fallback implementation")
            self.ideator = SocialContentIdeator()
            self.composer = SocialContentComposer()
            self.optimizer = SocialContentOptimizer()
            self.trend_analyzer = SocialTrendAnalyzer()

        def generate_content(
            self,
            content_brief: ContentBrief,
            trending_topics: list[str] | None = None,
        ) -> dict[str, Any]:
            return {"error": "DSPy not available", "brief": content_brief}

        def optimize_existing_content(
            self,
            content: str,
            metrics: dict[str, Any],
            platform: Platform,
            goal: str = "increase engagement",
        ) -> dict[str, Any]:
            return {"error": "DSPy not available", "original_content": content}
