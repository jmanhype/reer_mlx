"""T024: KPI evaluator for performance metrics implementation.

Implements comprehensive KPI evaluation system for social media content
performance, including engagement metrics, quality assessment, and
business impact measurement with DSPy integration.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
import statistics
import time
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

try:
    from ..core.candidate_scorer import ScoringMetrics
    from ..core.exceptions import ScoringError, ValidationError
except ImportError:
    # Fallback for standalone usage
    try:
        from core.candidate_scorer import ScoringMetrics
        from core.exceptions import ScoringError, ValidationError
    except ImportError:
        # Create mock classes if imports fail
        class ValidationError(Exception):
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
                self.fluency_score = 0.5
                self.coherence_score = 0.5
                self.relevance_score = 0.5
                self.perplexity = 5.0
                self.text_length = None


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    ENGAGEMENT = "engagement"
    QUALITY = "quality"
    REACH = "reach"
    CONVERSION = "conversion"
    BRAND = "brand"
    VIRAL = "viral"
    SENTIMENT = "sentiment"
    BUSINESS = "business"


class MetricLevel(Enum):
    """Metric evaluation levels."""

    CONTENT = "content"  # Individual content piece
    CAMPAIGN = "campaign"  # Collection of content
    PLATFORM = "platform"  # Platform-specific performance
    OVERALL = "overall"  # Overall performance


@dataclass
class MetricDefinition:
    """Definition of a performance metric."""

    name: str
    metric_type: MetricType
    description: str
    calculation_method: str
    weight: float = 1.0
    target_value: float | None = None
    benchmark_value: float | None = None
    higher_is_better: bool = True
    unit: str = "score"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of a metric evaluation."""

    metric_name: str
    value: float
    normalized_value: float  # 0.0 to 1.0
    target_achieved: bool
    benchmark_comparison: float | None = None
    confidence: float = 1.0
    calculation_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Complete performance metrics result."""

    evaluation_id: str
    content_id: str
    platform: str
    metric_results: list[MetricResult]
    overall_score: float
    grade: str  # A, B, C, D, F
    evaluation_time: float
    recommendations: list[str] = field(default_factory=list)
    insights: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BenchmarkData:
    """Benchmark data for comparison."""

    platform: str
    content_type: str
    audience_segment: str
    metrics: dict[str, float]
    sample_size: int
    time_period: str
    last_updated: datetime


# DSPy Signatures for KPI evaluation
class MetricAnalysisSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Analyze content for specific performance metrics."""

    content = InputField(desc="Content to analyze")
    platform = InputField(desc="Platform where content will be published")
    metric_type = InputField(desc="Type of metric to evaluate")
    context = InputField(desc="Additional context about content and audience")

    metric_score = OutputField(desc="Numerical score for the metric (0-10)")
    reasoning = OutputField(desc="Explanation of the score")
    improvement_suggestions = OutputField(desc="Specific suggestions for improvement")


class PerformancePredictionSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Predict performance metrics for content."""

    content = InputField(desc="Content to evaluate")
    platform = InputField(desc="Target platform")
    audience_data = InputField(desc="Target audience information")
    historical_data = InputField(desc="Historical performance data for reference")

    engagement_prediction = OutputField(desc="Predicted engagement rate (0-100)")
    reach_prediction = OutputField(desc="Predicted reach score (0-10)")
    viral_potential = OutputField(desc="Viral potential score (0-10)")
    quality_assessment = OutputField(desc="Content quality score (0-10)")


class BenchmarkAnalysisSignature(dspy.Signature if DSPY_AVAILABLE else object):
    """Compare performance against benchmarks."""

    performance_data = InputField(desc="Current performance metrics")
    benchmark_data = InputField(desc="Benchmark data for comparison")
    industry_context = InputField(desc="Industry and competitive context")

    benchmark_comparison = OutputField(desc="Comparison against benchmarks")
    competitive_position = OutputField(desc="Position relative to competition")
    improvement_areas = OutputField(desc="Areas needing improvement")


class MetricAnalysisModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for metric analysis."""

    def __init__(self, use_reasoning: bool = True):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        super().__init__()

        if use_reasoning:
            self.metric_analyzer = ChainOfThought(MetricAnalysisSignature)
            self.performance_predictor = ChainOfThought(PerformancePredictionSignature)
            self.benchmark_analyzer = ChainOfThought(BenchmarkAnalysisSignature)
        else:
            self.metric_analyzer = Predict(MetricAnalysisSignature)
            self.performance_predictor = Predict(PerformancePredictionSignature)
            self.benchmark_analyzer = Predict(BenchmarkAnalysisSignature)

    def forward(self, **kwargs):
        """Analyze metrics for content."""
        return self.metric_analyzer(**kwargs)

    def predict_performance(self, **kwargs):
        """Predict performance metrics."""
        return self.performance_predictor(**kwargs)

    def analyze_benchmarks(self, **kwargs):
        """Analyze against benchmarks."""
        return self.benchmark_analyzer(**kwargs)


class KPICalculator:
    """Calculator for various KPI metrics."""

    @staticmethod
    def calculate_engagement_rate(
        likes: int, comments: int, shares: int, impressions: int
    ) -> float:
        """Calculate engagement rate."""
        if impressions <= 0:
            return 0.0

        total_engagement = likes + comments + shares
        return (total_engagement / impressions) * 100

    @staticmethod
    def calculate_viral_coefficient(
        shares: int, original_reach: int, average_follower_count: int = 100
    ) -> float:
        """Calculate viral coefficient."""
        if original_reach <= 0:
            return 0.0

        viral_reach = shares * average_follower_count
        return viral_reach / original_reach

    @staticmethod
    def calculate_quality_score(
        readability_score: float,
        relevance_score: float,
        completeness_score: float,
        originality_score: float,
    ) -> float:
        """Calculate overall quality score."""
        weights = {
            "readability": 0.25,
            "relevance": 0.35,
            "completeness": 0.25,
            "originality": 0.15,
        }

        weighted_score = (
            readability_score * weights["readability"]
            + relevance_score * weights["relevance"]
            + completeness_score * weights["completeness"]
            + originality_score * weights["originality"]
        )

        return min(10.0, max(0.0, weighted_score))

    @staticmethod
    def calculate_brand_alignment(
        tone_match: float, message_consistency: float, value_alignment: float
    ) -> float:
        """Calculate brand alignment score."""
        return (tone_match + message_consistency + value_alignment) / 3

    @staticmethod
    def calculate_reach_efficiency(
        actual_reach: int, potential_reach: int, cost: float = 0.0
    ) -> float:
        """Calculate reach efficiency."""
        if potential_reach <= 0:
            return 0.0

        reach_percentage = (actual_reach / potential_reach) * 100

        if cost > 0:
            # Cost per reach point
            efficiency = reach_percentage / cost
        else:
            efficiency = reach_percentage

        return min(100.0, max(0.0, efficiency))


class BenchmarkManager:
    """Manager for benchmark data and comparisons."""

    def __init__(self):
        """Initialize benchmark manager."""
        self.benchmarks: dict[str, BenchmarkData] = {}
        self._load_default_benchmarks()

    def _load_default_benchmarks(self) -> None:
        """Load default benchmark data."""
        # Default industry benchmarks (example data)
        default_benchmarks = {
            "twitter_general": BenchmarkData(
                platform="twitter",
                content_type="general",
                audience_segment="general",
                metrics={
                    "engagement_rate": 1.5,  # 1.5% average
                    "viral_coefficient": 0.1,
                    "quality_score": 6.5,
                    "brand_alignment": 7.0,
                    "reach_efficiency": 25.0,
                },
                sample_size=10000,
                time_period="2024",
                last_updated=datetime.now(UTC),
            ),
            "linkedin_professional": BenchmarkData(
                platform="linkedin",
                content_type="professional",
                audience_segment="business",
                metrics={
                    "engagement_rate": 2.1,
                    "viral_coefficient": 0.05,
                    "quality_score": 7.5,
                    "brand_alignment": 8.0,
                    "reach_efficiency": 30.0,
                },
                sample_size=5000,
                time_period="2024",
                last_updated=datetime.now(UTC),
            ),
            "instagram_visual": BenchmarkData(
                platform="instagram",
                content_type="visual",
                audience_segment="general",
                metrics={
                    "engagement_rate": 3.2,
                    "viral_coefficient": 0.15,
                    "quality_score": 7.0,
                    "brand_alignment": 7.5,
                    "reach_efficiency": 35.0,
                },
                sample_size=8000,
                time_period="2024",
                last_updated=datetime.now(UTC),
            ),
        }

        self.benchmarks = default_benchmarks

    def get_benchmark(
        self,
        platform: str,
        content_type: str = "general",
        audience_segment: str = "general",
    ) -> BenchmarkData | None:
        """Get benchmark data for specific criteria."""
        key = f"{platform}_{content_type}"
        return self.benchmarks.get(key)

    def add_benchmark(self, benchmark: BenchmarkData) -> None:
        """Add new benchmark data."""
        key = f"{benchmark.platform}_{benchmark.content_type}"
        self.benchmarks[key] = benchmark

    def compare_to_benchmark(
        self, metrics: dict[str, float], benchmark: BenchmarkData
    ) -> dict[str, float]:
        """Compare metrics to benchmark."""
        comparison = {}

        for metric_name, value in metrics.items():
            if metric_name in benchmark.metrics:
                benchmark_value = benchmark.metrics[metric_name]

                if benchmark_value > 0:
                    # Calculate percentage difference
                    comparison[metric_name] = (
                        (value - benchmark_value) / benchmark_value
                    ) * 100
                else:
                    comparison[metric_name] = 0.0

        return comparison


class KPIEvaluator:
    """Comprehensive KPI evaluator for social media content performance.

    Evaluates content against multiple performance metrics including engagement,
    quality, reach, and business impact with DSPy-enhanced analysis.
    """

    def __init__(
        self, metrics: list[str] | None = None, use_dspy_analysis: bool = True
    ):
        """Initialize KPI evaluator.

        Args:
            metrics: List of metrics to evaluate
            use_dspy_analysis: Whether to use DSPy for enhanced analysis
        """
        self.metrics = metrics or [
            "engagement_rate",
            "quality_score",
            "viral_potential",
            "brand_alignment",
            "reach_efficiency",
        ]

        self.use_dspy_analysis = use_dspy_analysis and DSPY_AVAILABLE

        # Initialize components
        self.calculator = KPICalculator()
        self.benchmark_manager = BenchmarkManager()

        # Initialize DSPy modules if available
        if self.use_dspy_analysis:
            self.analysis_module = MetricAnalysisModule(use_reasoning=True)
        else:
            self.analysis_module = None

        # Metric definitions
        self.metric_definitions = self._create_metric_definitions()

        # Evaluation state
        self._initialized = False
        self.evaluation_history: list[PerformanceMetrics] = []

        logger.info(f"Initialized KPI evaluator with {len(self.metrics)} metrics")

    def _create_metric_definitions(self) -> dict[str, MetricDefinition]:
        """Create standard metric definitions."""
        return {
            "engagement_rate": MetricDefinition(
                name="engagement_rate",
                metric_type=MetricType.ENGAGEMENT,
                description="Rate of user engagement with content",
                calculation_method="(likes + comments + shares) / impressions * 100",
                weight=1.0,
                target_value=2.5,
                unit="percentage",
                higher_is_better=True,
            ),
            "quality_score": MetricDefinition(
                name="quality_score",
                metric_type=MetricType.QUALITY,
                description="Overall content quality assessment",
                calculation_method="weighted_average(readability, relevance, completeness, originality)",
                weight=1.0,
                target_value=7.5,
                unit="score",
                higher_is_better=True,
            ),
            "viral_potential": MetricDefinition(
                name="viral_potential",
                metric_type=MetricType.VIRAL,
                description="Potential for content to go viral",
                calculation_method="viral_coefficient + shareability_factors",
                weight=0.8,
                target_value=0.15,
                unit="coefficient",
                higher_is_better=True,
            ),
            "brand_alignment": MetricDefinition(
                name="brand_alignment",
                metric_type=MetricType.BRAND,
                description="Alignment with brand voice and values",
                calculation_method="average(tone_match, message_consistency, value_alignment)",
                weight=1.0,
                target_value=8.0,
                unit="score",
                higher_is_better=True,
            ),
            "reach_efficiency": MetricDefinition(
                name="reach_efficiency",
                metric_type=MetricType.REACH,
                description="Efficiency of content reach",
                calculation_method="(actual_reach / potential_reach) * 100",
                weight=0.9,
                target_value=30.0,
                unit="percentage",
                higher_is_better=True,
            ),
            "sentiment_score": MetricDefinition(
                name="sentiment_score",
                metric_type=MetricType.SENTIMENT,
                description="Sentiment analysis of content and responses",
                calculation_method="sentiment_analysis(content + responses)",
                weight=0.7,
                target_value=0.7,
                unit="score",
                higher_is_better=True,
            ),
            "conversion_rate": MetricDefinition(
                name="conversion_rate",
                metric_type=MetricType.CONVERSION,
                description="Rate of desired actions taken",
                calculation_method="conversions / total_interactions * 100",
                weight=1.2,
                target_value=5.0,
                unit="percentage",
                higher_is_better=True,
            ),
        }

    async def initialize(self) -> None:
        """Initialize the KPI evaluator."""
        if self._initialized:
            return

        logger.info("Initializing KPI evaluator")

        try:
            # Validate metric definitions
            for metric_name in self.metrics:
                if metric_name not in self.metric_definitions:
                    logger.warning(f"Metric '{metric_name}' not defined, using default")

            self._initialized = True
            logger.info("KPI evaluator initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize KPI evaluator: {e}")
            raise ValidationError(f"KPI evaluator initialization failed: {e}")

    async def evaluate(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
        benchmark_comparison: bool = True,
    ) -> PerformanceMetrics:
        """Evaluate content performance across all configured metrics.

        Args:
            content: Content text to evaluate
            metadata: Content metadata (platform, audience, etc.)
            scoring_metrics: Optional pre-computed scoring metrics
            benchmark_comparison: Whether to compare against benchmarks

        Returns:
            Complete performance metrics
        """
        evaluation_id = f"eval_{int(time.time())}_{hash(content) % 10000}"
        start_time = time.time()

        logger.info(f"Starting KPI evaluation: {evaluation_id}")

        try:
            await self.initialize()

            platform = metadata.get("platform", "twitter")
            content_id = metadata.get("content_id", evaluation_id)

            # Step 1: Calculate base metrics
            metric_results = []

            for metric_name in self.metrics:
                if metric_name in self.metric_definitions:
                    result = await self._evaluate_metric(
                        metric_name, content, metadata, scoring_metrics
                    )
                    metric_results.append(result)

            # Step 2: Enhanced analysis with DSPy (if available)
            if self.analysis_module:
                await self._enhance_with_dspy_analysis(
                    content, metadata, metric_results
                )

            # Step 3: Calculate overall score
            overall_score = self._calculate_overall_score(metric_results)
            grade = self._calculate_grade(overall_score)

            # Step 4: Generate recommendations
            recommendations = await self._generate_recommendations(
                metric_results, content, metadata
            )

            # Step 5: Benchmark comparison (if enabled)
            insights = {}
            if benchmark_comparison:
                insights = await self._perform_benchmark_comparison(
                    metric_results, platform, metadata
                )

            # Create performance metrics result
            evaluation_time = time.time() - start_time

            performance_metrics = PerformanceMetrics(
                evaluation_id=evaluation_id,
                content_id=content_id,
                platform=platform,
                metric_results=metric_results,
                overall_score=overall_score,
                grade=grade,
                evaluation_time=evaluation_time,
                recommendations=recommendations,
                insights=insights,
            )

            # Store in history
            self.evaluation_history.append(performance_metrics)

            logger.info(
                f"KPI evaluation completed: {evaluation_id} "
                f"score={overall_score:.2f} grade={grade} "
                f"time={evaluation_time:.2f}s"
            )

            return performance_metrics

        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.exception(f"KPI evaluation failed: {evaluation_id}: {e}")

            return PerformanceMetrics(
                evaluation_id=evaluation_id,
                content_id=metadata.get("content_id", evaluation_id),
                platform=metadata.get("platform", "unknown"),
                metric_results=[],
                overall_score=0.0,
                grade="F",
                evaluation_time=evaluation_time,
                recommendations=[f"Evaluation failed: {str(e)}"],
            )

    async def _evaluate_metric(
        self,
        metric_name: str,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> MetricResult:
        """Evaluate a specific metric."""
        definition = self.metric_definitions[metric_name]

        # Calculate metric value based on type
        if metric_name == "engagement_rate":
            value = await self._calculate_engagement_rate(
                content, metadata, scoring_metrics
            )
        elif metric_name == "quality_score":
            value = await self._calculate_quality_score(
                content, metadata, scoring_metrics
            )
        elif metric_name == "viral_potential":
            value = await self._calculate_viral_potential(
                content, metadata, scoring_metrics
            )
        elif metric_name == "brand_alignment":
            value = await self._calculate_brand_alignment(
                content, metadata, scoring_metrics
            )
        elif metric_name == "reach_efficiency":
            value = await self._calculate_reach_efficiency(
                content, metadata, scoring_metrics
            )
        elif metric_name == "sentiment_score":
            value = await self._calculate_sentiment_score(
                content, metadata, scoring_metrics
            )
        elif metric_name == "conversion_rate":
            value = await self._calculate_conversion_rate(
                content, metadata, scoring_metrics
            )
        else:
            # Default calculation
            value = 5.0  # Neutral score

        # Normalize value (0.0 to 1.0)
        if definition.target_value:
            normalized_value = min(1.0, value / definition.target_value)
        else:
            normalized_value = min(1.0, value / 10.0)  # Assume 10 is max

        # Check target achievement
        target_achieved = False
        if definition.target_value:
            if definition.higher_is_better:
                target_achieved = value >= definition.target_value
            else:
                target_achieved = value <= definition.target_value

        return MetricResult(
            metric_name=metric_name,
            value=value,
            normalized_value=normalized_value,
            target_achieved=target_achieved,
            confidence=0.8,  # Default confidence
            calculation_details={
                "method": definition.calculation_method,
                "target_value": definition.target_value,
                "metadata_used": list(metadata.keys()),
            },
        )

    async def _calculate_engagement_rate(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate engagement rate metric."""
        if scoring_metrics:
            return scoring_metrics.engagement_score * 10  # Convert to 0-10 scale

        # Heuristic calculation based on content analysis
        content_length = len(content)
        has_question = "?" in content
        has_cta = any(
            cta in content.lower() for cta in ["comment", "share", "like", "thoughts"]
        )
        has_hashtags = "#" in content

        base_score = 2.0  # Base engagement rate

        # Adjust based on content features
        if has_question:
            base_score += 0.5
        if has_cta:
            base_score += 0.4
        if has_hashtags:
            base_score += 0.3
        if 50 <= content_length <= 200:
            base_score += 0.2  # Optimal length

        return min(10.0, base_score)

    async def _calculate_quality_score(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate quality score metric."""
        if scoring_metrics:
            return scoring_metrics.quality_score * 10

        # Basic quality assessment
        readability = 7.0  # Default readability
        relevance = 8.0  # Assume relevant
        completeness = 7.5  # Default completeness
        originality = 6.0  # Default originality

        # Adjust based on content analysis
        if len(content) < 20:
            completeness -= 2.0
        if len(content.split()) < 5:
            readability -= 1.0

        return self.calculator.calculate_quality_score(
            readability, relevance, completeness, originality
        )

    async def _calculate_viral_potential(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate viral potential metric."""
        if scoring_metrics:
            return scoring_metrics.viral_potential * 10

        # Viral potential factors
        viral_score = 0.0

        # Content factors
        if any(
            trend in content.lower() for trend in ["breaking", "exclusive", "shocking"]
        ):
            viral_score += 2.0
        if "!" in content:
            viral_score += 0.5
        if any(emoji in content for emoji in ["🔥", "💯", "⚡", "🚀"]):
            viral_score += 1.0
        if "#" in content:
            viral_score += 0.5

        # Timing and platform factors
        platform = metadata.get("platform", "twitter")
        if platform == "twitter":
            viral_score += 0.5  # Twitter has higher viral potential

        return min(10.0, viral_score)

    async def _calculate_brand_alignment(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate brand alignment metric."""
        if scoring_metrics:
            return scoring_metrics.brand_alignment * 10

        # Default brand alignment calculation
        tone_match = 7.5  # Default tone match
        message_consistency = 8.0  # Default consistency
        value_alignment = 7.0  # Default value alignment

        return self.calculator.calculate_brand_alignment(
            tone_match, message_consistency, value_alignment
        )

    async def _calculate_reach_efficiency(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate reach efficiency metric."""
        # Estimate reach efficiency based on content characteristics
        efficiency = 25.0  # Base efficiency

        # Hashtag usage improves reach
        hashtag_count = content.count("#")
        if hashtag_count > 0:
            efficiency += min(5.0, hashtag_count * 1.0)

        # Optimal length improves reach
        if 50 <= len(content) <= 200:
            efficiency += 3.0

        # Platform-specific adjustments
        platform = metadata.get("platform", "twitter")
        if platform == "linkedin":
            efficiency += 2.0  # Professional content tends to reach better

        return min(100.0, efficiency)

    async def _calculate_sentiment_score(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate sentiment score metric."""
        # Basic sentiment analysis
        positive_words = ["great", "awesome", "excellent", "amazing", "love", "best"]
        negative_words = ["bad", "awful", "terrible", "hate", "worst", "horrible"]

        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        # Calculate sentiment score (0.0 to 1.0)
        if positive_count + negative_count == 0:
            return 0.5  # Neutral

        sentiment = (positive_count - negative_count) / (
            positive_count + negative_count
        )
        return max(0.0, min(1.0, (sentiment + 1) / 2))

    async def _calculate_conversion_rate(
        self,
        content: str,
        metadata: dict[str, Any],
        scoring_metrics: ScoringMetrics | None = None,
    ) -> float:
        """Calculate conversion rate metric."""
        # Estimate conversion potential
        conversion_score = 1.0  # Base conversion rate

        # CTA presence
        cta_words = ["click", "visit", "download", "sign up", "register", "buy", "shop"]
        if any(cta in content.lower() for cta in cta_words):
            conversion_score += 2.0

        # Links or mentions
        if "http" in content or "@" in content:
            conversion_score += 1.0

        # Urgency indicators
        urgency_words = ["now", "today", "limited", "urgent", "hurry"]
        if any(word in content.lower() for word in urgency_words):
            conversion_score += 0.5

        return min(10.0, conversion_score)

    async def _enhance_with_dspy_analysis(
        self, content: str, metadata: dict[str, Any], metric_results: list[MetricResult]
    ) -> None:
        """Enhance metric results with DSPy analysis."""
        if not self.analysis_module:
            return

        try:
            platform = metadata.get("platform", "twitter")

            # Get DSPy analysis for key metrics
            for result in metric_results:
                if result.metric_name in [
                    "engagement_rate",
                    "quality_score",
                    "viral_potential",
                ]:
                    dspy_result = self.analysis_module(
                        content=content,
                        platform=platform,
                        metric_type=result.metric_name,
                        context=json.dumps(metadata),
                    )

                    # Parse DSPy score and update confidence
                    try:
                        dspy_score = float(dspy_result.metric_score)
                        # Blend with calculated score
                        blended_score = (result.value + dspy_score) / 2
                        result.value = blended_score
                        result.confidence = 0.9  # Higher confidence with DSPy

                        # Add DSPy insights to calculation details
                        result.calculation_details.update(
                            {
                                "dspy_score": dspy_score,
                                "dspy_reasoning": dspy_result.reasoning,
                                "dspy_suggestions": dspy_result.improvement_suggestions,
                            }
                        )

                    except (ValueError, AttributeError):
                        # Keep original score if DSPy parsing fails
                        pass

        except Exception as e:
            logger.warning(f"DSPy enhancement failed: {e}")

    def _calculate_overall_score(self, metric_results: list[MetricResult]) -> float:
        """Calculate weighted overall score."""
        if not metric_results:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for result in metric_results:
            definition = self.metric_definitions.get(result.metric_name)
            weight = definition.weight if definition else 1.0

            weighted_sum += result.normalized_value * weight
            total_weight += weight

        return (weighted_sum / total_weight) * 10 if total_weight > 0 else 0.0

    def _calculate_grade(self, overall_score: float) -> str:
        """Calculate letter grade from overall score."""
        if overall_score >= 9.0:
            return "A"
        if overall_score >= 8.0:
            return "B"
        if overall_score >= 7.0:
            return "C"
        if overall_score >= 6.0:
            return "D"
        return "F"

    async def _generate_recommendations(
        self, metric_results: list[MetricResult], content: str, metadata: dict[str, Any]
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        for result in metric_results:
            if not result.target_achieved:
                metric_name = result.metric_name

                if metric_name == "engagement_rate":
                    recommendations.append(
                        "Add questions or calls-to-action to increase engagement"
                    )
                elif metric_name == "quality_score":
                    recommendations.append("Improve content clarity and completeness")
                elif metric_name == "viral_potential":
                    recommendations.append("Add trending elements or emotional hooks")
                elif metric_name == "brand_alignment":
                    recommendations.append(
                        "Ensure content matches brand voice and values"
                    )
                elif metric_name == "reach_efficiency":
                    recommendations.append("Optimize hashtags and posting timing")

                # Add DSPy suggestions if available
                if "dspy_suggestions" in result.calculation_details:
                    suggestions = result.calculation_details["dspy_suggestions"]
                    if suggestions:
                        recommendations.append(f"DSPy suggests: {suggestions}")

        # Add general recommendations
        if len(content) < 50:
            recommendations.append("Consider adding more detail to the content")
        if "#" not in content:
            recommendations.append("Add relevant hashtags to improve discoverability")

        return recommendations[:5]  # Limit to top 5 recommendations

    async def _perform_benchmark_comparison(
        self,
        metric_results: list[MetricResult],
        platform: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform benchmark comparison analysis."""
        insights = {}

        # Get benchmark data
        benchmark = self.benchmark_manager.get_benchmark(platform)

        if benchmark:
            # Create metrics dict for comparison
            metrics = {result.metric_name: result.value for result in metric_results}

            # Compare to benchmark
            comparison = self.benchmark_manager.compare_to_benchmark(metrics, benchmark)

            insights["benchmark_comparison"] = comparison
            insights["benchmark_platform"] = platform
            insights["benchmark_sample_size"] = benchmark.sample_size

            # Calculate overall benchmark performance
            positive_comparisons = sum(1 for diff in comparison.values() if diff > 0)
            insights["above_benchmark_metrics"] = positive_comparisons
            insights["total_compared_metrics"] = len(comparison)
            insights["benchmark_performance"] = (
                positive_comparisons / len(comparison) * 100 if comparison else 0
            )

        return insights

    async def get_evaluation_summary(
        self, time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Get summary of recent evaluations.

        Args:
            time_window_hours: Time window for summary

        Returns:
            Evaluation summary statistics
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=time_window_hours)
        recent_evaluations = [
            eval_result
            for eval_result in self.evaluation_history
            if eval_result.timestamp >= cutoff_time
        ]

        if not recent_evaluations:
            return {"message": "No recent evaluations found"}

        # Calculate statistics
        overall_scores = [
            eval_result.overall_score for eval_result in recent_evaluations
        ]
        grades = [eval_result.grade for eval_result in recent_evaluations]

        return {
            "total_evaluations": len(recent_evaluations),
            "time_window_hours": time_window_hours,
            "average_score": statistics.mean(overall_scores),
            "median_score": statistics.median(overall_scores),
            "best_score": max(overall_scores),
            "worst_score": min(overall_scores),
            "grade_distribution": {grade: grades.count(grade) for grade in set(grades)},
            "metrics_evaluated": list(self.metrics),
            "improvement_rate": self._calculate_improvement_rate(recent_evaluations),
        }

    def _calculate_improvement_rate(
        self, evaluations: list[PerformanceMetrics]
    ) -> float:
        """Calculate improvement rate over time."""
        if len(evaluations) < 2:
            return 0.0

        # Sort by timestamp
        sorted_evals = sorted(evaluations, key=lambda x: x.timestamp)

        first_half = sorted_evals[: len(sorted_evals) // 2]
        second_half = sorted_evals[len(sorted_evals) // 2 :]

        first_avg = statistics.mean([e.overall_score for e in first_half])
        second_avg = statistics.mean([e.overall_score for e in second_half])

        if first_avg > 0:
            return ((second_avg - first_avg) / first_avg) * 100
        return 0.0

    def is_available(self) -> bool:
        """Check if KPI evaluator is available."""
        return self._initialized

    async def get_evaluator_status(self) -> dict[str, Any]:
        """Get evaluator status and configuration.

        Returns:
            Evaluator status information
        """
        return {
            "initialized": self._initialized,
            "dspy_available": DSPY_AVAILABLE,
            "dspy_analysis_enabled": self.use_dspy_analysis,
            "metrics_configured": self.metrics,
            "total_evaluations": len(self.evaluation_history),
            "metric_definitions_count": len(self.metric_definitions),
            "benchmark_data_available": len(self.benchmark_manager.benchmarks),
            "supported_platforms": ["twitter", "linkedin", "instagram"],
        }
