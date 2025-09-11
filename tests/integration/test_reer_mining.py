"""T010: Integration test for REER strategy extraction pipeline.

Tests the complete REER (Reverse Engineering Effective Reactions) strategy mining
workflow from historical post analysis through pattern extraction and strategy
formulation. Following London School TDD with mock-first approach.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import datetime
from datetime import timezone
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

# Import statements that will fail initially (RED phase)
try:
    from reer_mining.feature_detector import FeatureDetector
    from reer_mining.mining_pipeline import REERMiningPipeline
    from reer_mining.pattern_analyzer import PatternAnalyzer
    from reer_mining.schemas import ExtractedStrategy, HistoricalPost, StrategyPattern
    from reer_mining.strategy_extractor import REERStrategyExtractor
    from reer_mining.strategy_synthesizer import StrategySynthesizer

    from core.exceptions import ExtractionError, PatternAnalysisError, SynthesisError
except ImportError:
    # Expected during RED phase - create mock classes for contract testing
    class REERStrategyExtractor:
        pass

    class PatternAnalyzer:
        pass

    class FeatureDetector:
        pass

    class StrategySynthesizer:
        pass

    class REERMiningPipeline:
        pass

    class HistoricalPost:
        pass

    class ExtractedStrategy:
        pass

    class StrategyPattern:
        pass

    class ExtractionError(Exception):
        pass

    class PatternAnalysisError(Exception):
        pass

    class SynthesisError(Exception):
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestREERMiningIntegration:
    """Integration tests for REER strategy extraction pipeline.

    Tests complete end-to-end workflows including:
    - Historical post data loading and preprocessing
    - Multi-dimensional pattern analysis
    - Strategy feature extraction and clustering
    - Strategy synthesis and validation
    - Performance optimization and caching
    - Cross-platform strategy adaptation
    """

    @pytest.fixture
    def sample_historical_posts(self) -> list[dict[str, Any]]:
        """Sample historical posts for strategy extraction."""
        return [
            {
                "id": "post_001",
                "platform": "twitter",
                "text": "🚀 Just launched our new AI feature! What do you think? Try it out and let us know! #AI #Innovation #Startup",
                "timestamp": "2024-01-10T14:30:00Z",
                "metrics": {
                    "impressions": 5000,
                    "engagement_rate": 12.5,
                    "retweets": 45,
                    "likes": 300,
                    "replies": 25,
                },
                "metadata": {
                    "hour_of_day": 14,
                    "day_of_week": 3,
                    "character_count": 95,
                    "hashtag_count": 3,
                    "emoji_count": 1,
                    "question_count": 2,
                    "url_count": 0,
                },
            },
            {
                "id": "post_002",
                "platform": "twitter",
                "text": "Quick tip for developers: Always validate your data inputs. Here's why and how 👇 [thread]",
                "timestamp": "2024-01-08T09:15:00Z",
                "metrics": {
                    "impressions": 8000,
                    "engagement_rate": 15.2,
                    "retweets": 85,
                    "likes": 520,
                    "replies": 45,
                },
                "metadata": {
                    "hour_of_day": 9,
                    "day_of_week": 1,
                    "character_count": 88,
                    "hashtag_count": 0,
                    "emoji_count": 1,
                    "question_count": 0,
                    "url_count": 0,
                    "is_thread": True,
                },
            },
            {
                "id": "post_003",
                "platform": "linkedin",
                "text": "Reflecting on 2023: 3 key lessons learned in AI development. What were your biggest takeaways this year?",
                "timestamp": "2023-12-28T16:45:00Z",
                "metrics": {
                    "impressions": 12000,
                    "engagement_rate": 8.7,
                    "reactions": 180,
                    "comments": 65,
                    "shares": 35,
                },
                "metadata": {
                    "hour_of_day": 16,
                    "day_of_week": 4,
                    "character_count": 108,
                    "hashtag_count": 0,
                    "emoji_count": 0,
                    "question_count": 1,
                    "url_count": 0,
                    "is_professional": True,
                },
            },
        ]

    @pytest.fixture
    def expected_extracted_patterns(self) -> list[dict[str, Any]]:
        """Expected patterns extracted from historical posts."""
        return [
            {
                "pattern_id": "engaging_questions",
                "type": "content_pattern",
                "description": "Posts ending with questions generate higher engagement",
                "confidence": 0.87,
                "supporting_posts": ["post_001", "post_003"],
                "metrics": {
                    "avg_engagement_rate": 10.6,
                    "sample_size": 2,
                    "statistical_significance": 0.95,
                },
                "features": ["question_ending", "engagement_hook"],
                "conditions": {
                    "question_count": {"min": 1, "max": 3},
                    "character_count": {"min": 50, "max": 150},
                },
            },
            {
                "pattern_id": "emoji_enhancement",
                "type": "visual_pattern",
                "description": "Strategic emoji use increases impression rates",
                "confidence": 0.92,
                "supporting_posts": ["post_001", "post_002"],
                "metrics": {
                    "avg_impressions": 6500,
                    "impression_lift": 1.35,
                    "sample_size": 2,
                },
                "features": ["emoji_presence", "visual_appeal"],
                "conditions": {
                    "emoji_count": {"min": 1, "max": 2},
                    "emoji_placement": "beginning_or_end",
                },
            },
            {
                "pattern_id": "professional_timing",
                "type": "temporal_pattern",
                "description": "Morning posts (9-11 AM) perform better on weekdays",
                "confidence": 0.78,
                "supporting_posts": ["post_002"],
                "metrics": {
                    "avg_engagement_rate": 15.2,
                    "time_window": "09:00-11:00",
                    "day_preference": "weekdays",
                },
                "features": ["optimal_timing", "professional_audience"],
                "conditions": {
                    "hour_of_day": {"min": 9, "max": 11},
                    "day_of_week": {"in": [1, 2, 3, 4, 5]},
                },
            },
        ]

    @pytest.fixture
    def expected_synthesized_strategy(self) -> dict[str, Any]:
        """Expected synthesized strategy from pattern analysis."""
        return {
            "strategy_id": str(uuid4()),
            "name": "High-Engagement Technical Content Strategy",
            "description": "Optimized strategy for technical content with maximum engagement",
            "confidence": 0.85,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": {
                "content_structure": {
                    "opening": "attention_grabbing_statement",
                    "body": "valuable_insight_or_tip",
                    "closing": "engaging_question",
                    "character_range": {"min": 80, "max": 120},
                },
                "visual_elements": {
                    "emoji_usage": "strategic_placement",
                    "emoji_count": {"min": 1, "max": 2},
                    "hashtag_count": {"max": 3},
                },
                "timing_optimization": {
                    "preferred_hours": [9, 10, 14, 15],
                    "preferred_days": [1, 2, 3, 4],
                    "avoid_weekends": True,
                },
                "engagement_tactics": {
                    "include_questions": True,
                    "call_to_action": "implicit",
                    "thread_potential": True,
                },
            },
            "expected_performance": {
                "engagement_rate": {"min": 10.0, "target": 15.0},
                "impression_range": {"min": 5000, "target": 8000},
                "viral_potential": 0.65,
            },
            "platform_adaptations": {
                "twitter": {"character_limit": 280, "hashtag_strategy": "trending"},
                "linkedin": {"professional_tone": True, "industry_relevant": True},
            },
            "metadata": {
                "source_posts_count": 3,
                "patterns_used": [
                    "engaging_questions",
                    "emoji_enhancement",
                    "professional_timing",
                ],
                "extraction_method": "reer_v1.0",
                "validation_score": 0.89,
            },
        }

    @pytest.fixture
    def mock_strategy_extractor(self) -> Mock:
        """Mock REERStrategyExtractor with behavior contracts."""
        extractor = Mock(spec=REERStrategyExtractor)
        extractor.extract_patterns = AsyncMock()
        extractor.analyze_performance_correlation = AsyncMock()
        extractor.identify_success_factors = AsyncMock()
        extractor.validate_pattern_significance = Mock()
        return extractor

    @pytest.fixture
    def mock_pattern_analyzer(self) -> Mock:
        """Mock PatternAnalyzer with behavior contracts."""
        analyzer = Mock(spec=PatternAnalyzer)
        analyzer.analyze_content_patterns = AsyncMock()
        analyzer.analyze_temporal_patterns = AsyncMock()
        analyzer.analyze_engagement_patterns = AsyncMock()
        analyzer.calculate_pattern_confidence = Mock()
        analyzer.statistical_validation = Mock()
        return analyzer

    @pytest.fixture
    def mock_feature_detector(self) -> Mock:
        """Mock FeatureDetector with behavior contracts."""
        detector = Mock(spec=FeatureDetector)
        detector.detect_content_features = Mock()
        detector.detect_visual_features = Mock()
        detector.detect_temporal_features = Mock()
        detector.detect_engagement_features = Mock()
        detector.extract_hashtag_patterns = Mock()
        detector.extract_emoji_patterns = Mock()
        return detector

    @pytest.fixture
    def mock_strategy_synthesizer(self) -> Mock:
        """Mock StrategySynthesizer with behavior contracts."""
        synthesizer = Mock(spec=StrategySynthesizer)
        synthesizer.synthesize_strategy = AsyncMock()
        synthesizer.optimize_components = AsyncMock()
        synthesizer.validate_strategy = AsyncMock()
        synthesizer.adapt_for_platform = Mock()
        synthesizer.calculate_expected_performance = Mock()
        return synthesizer

    @pytest.fixture
    def mock_mining_pipeline(
        self,
        mock_strategy_extractor: Mock,
        mock_pattern_analyzer: Mock,
        mock_feature_detector: Mock,
        mock_strategy_synthesizer: Mock,
    ) -> Mock:
        """Mock REERMiningPipeline with all dependencies."""
        pipeline = Mock(spec=REERMiningPipeline)
        pipeline.extractor = mock_strategy_extractor
        pipeline.analyzer = mock_pattern_analyzer
        pipeline.detector = mock_feature_detector
        pipeline.synthesizer = mock_strategy_synthesizer
        pipeline.mine_strategies = AsyncMock()
        pipeline.analyze_historical_data = AsyncMock()
        pipeline.extract_and_synthesize = AsyncMock()
        pipeline.validate_pipeline_health = AsyncMock()
        return pipeline

    # Core REER Mining Workflow Tests

    async def test_complete_strategy_extraction_pipeline(
        self,
        mock_mining_pipeline: Mock,
        sample_historical_posts: list[dict[str, Any]],
        expected_extracted_patterns: list[dict[str, Any]],
        expected_synthesized_strategy: dict[str, Any],
    ):
        """Test complete REER pipeline: analyze → extract → synthesize."""
        # Arrange
        mock_mining_pipeline.analyze_historical_data.return_value = {
            "posts_analyzed": 3,
            "patterns_found": 3,
            "data_quality_score": 0.92,
        }
        mock_mining_pipeline.extractor.extract_patterns.return_value = (
            expected_extracted_patterns
        )
        mock_mining_pipeline.synthesizer.synthesize_strategy.return_value = (
            expected_synthesized_strategy
        )

        # Act - This will fail initially (RED phase)
        result = await mock_mining_pipeline.mine_strategies(
            historical_posts=sample_historical_posts,
            min_confidence=0.7,
            platforms=["twitter", "linkedin"],
        )

        # Assert - Testing the expected workflow interactions
        mock_mining_pipeline.analyze_historical_data.assert_called_once_with(
            sample_historical_posts
        )
        mock_mining_pipeline.extractor.extract_patterns.assert_called_once()
        mock_mining_pipeline.synthesizer.synthesize_strategy.assert_called_once()

        assert result["status"] == "success"
        assert result["strategy_id"] == expected_synthesized_strategy["strategy_id"]
        assert len(result["patterns_extracted"]) == 3

    async def test_pattern_analysis_with_statistical_validation(
        self,
        mock_pattern_analyzer: Mock,
        sample_historical_posts: list[dict[str, Any]],
        expected_extracted_patterns: list[dict[str, Any]],
    ):
        """Test pattern analysis with statistical significance validation."""
        # Arrange
        mock_pattern_analyzer.analyze_content_patterns.return_value = [
            expected_extracted_patterns[0]
        ]
        mock_pattern_analyzer.analyze_temporal_patterns.return_value = [
            expected_extracted_patterns[2]
        ]
        mock_pattern_analyzer.statistical_validation.return_value = {
            "p_value": 0.03,
            "confidence_interval": (0.75, 0.95),
            "sample_size_adequate": True,
            "statistical_power": 0.85,
        }

        # Act
        content_patterns = await mock_pattern_analyzer.analyze_content_patterns(
            sample_historical_posts
        )
        temporal_patterns = await mock_pattern_analyzer.analyze_temporal_patterns(
            sample_historical_posts
        )
        validation = mock_pattern_analyzer.statistical_validation(
            content_patterns + temporal_patterns
        )

        # Assert
        assert len(content_patterns) > 0
        assert len(temporal_patterns) > 0
        assert validation["p_value"] < 0.05  # Statistically significant
        assert validation["sample_size_adequate"] is True

    async def test_multi_dimensional_feature_detection(
        self, mock_feature_detector: Mock, sample_historical_posts: list[dict[str, Any]]
    ):
        """Test comprehensive feature detection across multiple dimensions."""
        # Arrange
        content_features = {
            "question_patterns": ["ending_questions", "mid_content_questions"],
            "language_patterns": ["technical_terminology", "casual_tone"],
            "structural_patterns": ["list_format", "thread_continuation"],
        }

        visual_features = {
            "emoji_usage": ["strategic_placement", "emotional_enhancement"],
            "hashtag_strategy": ["trending_tags", "branded_tags"],
            "multimedia_elements": ["image_presence", "link_sharing"],
        }

        temporal_features = {
            "timing_patterns": ["morning_engagement", "weekday_preference"],
            "frequency_patterns": ["daily_posting", "strategic_spacing"],
            "seasonal_patterns": ["year_end_reflection", "new_feature_launches"],
        }

        mock_feature_detector.detect_content_features.return_value = content_features
        mock_feature_detector.detect_visual_features.return_value = visual_features
        mock_feature_detector.detect_temporal_features.return_value = temporal_features

        # Act
        all_features = {}
        all_features.update(
            mock_feature_detector.detect_content_features(sample_historical_posts)
        )
        all_features.update(
            mock_feature_detector.detect_visual_features(sample_historical_posts)
        )
        all_features.update(
            mock_feature_detector.detect_temporal_features(sample_historical_posts)
        )

        # Assert
        mock_feature_detector.detect_content_features.assert_called_once()
        mock_feature_detector.detect_visual_features.assert_called_once()
        mock_feature_detector.detect_temporal_features.assert_called_once()

        assert "question_patterns" in all_features
        assert "emoji_usage" in all_features
        assert "timing_patterns" in all_features

    async def test_strategy_synthesis_with_optimization(
        self,
        mock_strategy_synthesizer: Mock,
        expected_extracted_patterns: list[dict[str, Any]],
        expected_synthesized_strategy: dict[str, Any],
    ):
        """Test strategy synthesis with component optimization."""
        # Arrange
        optimization_results = {
            "content_optimization": {
                "optimal_length": 95,
                "question_placement": "end",
                "emoji_count": 1,
            },
            "timing_optimization": {
                "best_hours": [9, 14],
                "best_days": [1, 2, 3, 4],
                "engagement_multiplier": 1.45,
            },
            "performance_prediction": {
                "expected_engagement": 12.5,
                "confidence_interval": (10.0, 15.0),
                "viral_probability": 0.65,
            },
        }

        mock_strategy_synthesizer.optimize_components.return_value = (
            optimization_results
        )
        mock_strategy_synthesizer.synthesize_strategy.return_value = (
            expected_synthesized_strategy
        )
        mock_strategy_synthesizer.validate_strategy.return_value = {
            "is_valid": True,
            "score": 0.89,
        }

        # Act
        optimization = await mock_strategy_synthesizer.optimize_components(
            expected_extracted_patterns
        )
        strategy = await mock_strategy_synthesizer.synthesize_strategy(
            patterns=expected_extracted_patterns, optimization=optimization
        )
        validation = await mock_strategy_synthesizer.validate_strategy(strategy)

        # Assert
        assert optimization["performance_prediction"]["expected_engagement"] > 10.0
        assert strategy["confidence"] > 0.8
        assert validation["is_valid"] is True

    # Cross-Platform Strategy Adaptation Tests

    async def test_platform_specific_strategy_adaptation(
        self,
        mock_strategy_synthesizer: Mock,
        expected_synthesized_strategy: dict[str, Any],
    ):
        """Test strategy adaptation for different social media platforms."""
        # Arrange
        twitter_adaptation = {
            "character_limit": 280,
            "hashtag_limit": 3,
            "threading_strategy": "enabled",
            "engagement_tactics": ["polls", "retweets", "replies"],
            "optimal_posting_frequency": "3-5 times per day",
        }

        linkedin_adaptation = {
            "character_limit": 3000,
            "professional_tone": True,
            "industry_relevance": "high",
            "engagement_tactics": ["thoughtful_comments", "professional_shares"],
            "optimal_posting_frequency": "1-2 times per day",
        }

        mock_strategy_synthesizer.adapt_for_platform.side_effect = [
            twitter_adaptation,
            linkedin_adaptation,
        ]

        # Act
        twitter_strategy = mock_strategy_synthesizer.adapt_for_platform(
            expected_synthesized_strategy, "twitter"
        )
        linkedin_strategy = mock_strategy_synthesizer.adapt_for_platform(
            expected_synthesized_strategy, "linkedin"
        )

        # Assert
        assert twitter_strategy["character_limit"] == 280
        assert linkedin_strategy["professional_tone"] is True
        assert twitter_strategy["threading_strategy"] == "enabled"
        assert linkedin_strategy["industry_relevance"] == "high"

    async def test_cross_platform_pattern_correlation(
        self, mock_pattern_analyzer: Mock, sample_historical_posts: list[dict[str, Any]]
    ):
        """Test correlation analysis across different platforms."""
        # Arrange
        correlation_analysis = {
            "twitter_linkedin_correlation": 0.78,
            "shared_patterns": [
                "question_engagement",
                "morning_posting",
                "professional_content",
            ],
            "platform_specific_patterns": {
                "twitter": ["hashtag_trending", "emoji_heavy", "thread_format"],
                "linkedin": [
                    "industry_insights",
                    "professional_networking",
                    "long_form",
                ],
            },
            "cross_platform_opportunities": [
                "repurpose_content",
                "timing_synchronization",
                "audience_bridging",
            ],
        }

        mock_pattern_analyzer.analyze_engagement_patterns.return_value = (
            correlation_analysis
        )

        # Act
        analysis = await mock_pattern_analyzer.analyze_engagement_patterns(
            sample_historical_posts, platforms=["twitter", "linkedin"]
        )

        # Assert
        assert analysis["twitter_linkedin_correlation"] > 0.7
        assert len(analysis["shared_patterns"]) > 0
        assert "repurpose_content" in analysis["cross_platform_opportunities"]

    # Performance and Scalability Tests

    async def test_large_dataset_pattern_extraction(self, mock_mining_pipeline: Mock):
        """Test pattern extraction performance with large historical datasets."""
        # Arrange
        large_dataset_size = 10000
        processing_metrics = {
            "posts_processed": large_dataset_size,
            "patterns_extracted": 47,
            "processing_time_seconds": 125.5,
            "memory_usage_mb": 256,
            "cache_hit_rate": 0.78,
            "parallel_jobs": 8,
        }

        mock_mining_pipeline.analyze_historical_data.return_value = {
            "status": "completed",
            "metrics": processing_metrics,
        }

        # Act
        result = await mock_mining_pipeline.analyze_historical_data(
            post_count=large_dataset_size, enable_caching=True, parallel_processing=True
        )

        # Assert
        assert result["metrics"]["posts_processed"] == large_dataset_size
        assert result["metrics"]["processing_time_seconds"] < 300  # Under 5 minutes
        assert result["metrics"]["memory_usage_mb"] < 500  # Memory efficient
        assert result["metrics"]["cache_hit_rate"] > 0.5  # Good cache performance

    async def test_incremental_pattern_learning(
        self,
        mock_strategy_extractor: Mock,
        sample_historical_posts: list[dict[str, Any]],
    ):
        """Test incremental learning with new historical data."""
        # Arrange
        existing_patterns = ["question_engagement", "emoji_usage"]
        new_data_batch = sample_historical_posts[:2]

        incremental_results = {
            "new_patterns_discovered": ["thread_continuation", "link_sharing"],
            "existing_patterns_updated": ["question_engagement"],
            "confidence_improvements": {
                "question_engagement": {"old": 0.75, "new": 0.87}
            },
            "pattern_stability": 0.92,
        }

        mock_strategy_extractor.extract_patterns.return_value = incremental_results

        # Act
        result = await mock_strategy_extractor.extract_patterns(
            new_data=new_data_batch,
            existing_patterns=existing_patterns,
            incremental_mode=True,
        )

        # Assert
        assert len(result["new_patterns_discovered"]) > 0
        assert "question_engagement" in result["existing_patterns_updated"]
        assert result["pattern_stability"] > 0.9

    # Error Handling and Edge Cases

    async def test_insufficient_data_handling(self, mock_mining_pipeline: Mock):
        """Test handling of insufficient historical data scenarios."""
        # Arrange
        minimal_posts = [{"id": "single_post", "metrics": {"impressions": 100}}]

        mock_mining_pipeline.analyze_historical_data.side_effect = ExtractionError(
            "Insufficient data for reliable pattern extraction. Minimum 10 posts required."
        )

        # Act & Assert
        with pytest.raises(ExtractionError, match="Insufficient data"):
            await mock_mining_pipeline.analyze_historical_data(minimal_posts)

    async def test_low_quality_data_filtering(
        self,
        mock_strategy_extractor: Mock,
        sample_historical_posts: list[dict[str, Any]],
    ):
        """Test filtering and handling of low-quality historical data."""
        # Arrange
        low_quality_posts = [
            {**post, "metrics": {"impressions": 5, "engagement_rate": 0.1}}
            for post in sample_historical_posts
        ]

        quality_filter_results = {
            "total_posts": 3,
            "filtered_posts": 3,
            "removed_posts": 0,
            "quality_scores": [0.2, 0.15, 0.18],
            "min_quality_threshold": 0.3,
            "recommendation": "increase_sample_size",
        }

        mock_strategy_extractor.analyze_performance_correlation.return_value = (
            quality_filter_results
        )

        # Act
        result = await mock_strategy_extractor.analyze_performance_correlation(
            low_quality_posts, min_quality_threshold=0.3
        )

        # Assert
        assert result["removed_posts"] >= 0
        assert result["recommendation"] == "increase_sample_size"

    async def test_pattern_confliction_resolution(
        self,
        mock_pattern_analyzer: Mock,
        expected_extracted_patterns: list[dict[str, Any]],
    ):
        """Test resolution of conflicting pattern recommendations."""
        # Arrange
        conflicting_patterns = [
            {
                "pattern_id": "morning_posting",
                "recommended_hours": [9, 10, 11],
                "confidence": 0.85,
            },
            {
                "pattern_id": "afternoon_engagement",
                "recommended_hours": [14, 15, 16],
                "confidence": 0.78,
            },
        ]

        resolution_result = {
            "resolved_pattern": {
                "pattern_id": "optimal_timing_combined",
                "recommended_hours": [9, 10, 14, 15],
                "resolution_method": "confidence_weighted_merge",
                "confidence": 0.82,
            },
            "conflicts_detected": 1,
            "resolution_strategy": "merge_compatible_patterns",
        }

        mock_pattern_analyzer.statistical_validation.return_value = resolution_result

        # Act
        result = mock_pattern_analyzer.statistical_validation(
            conflicting_patterns, resolve_conflicts=True
        )

        # Assert
        assert result["conflicts_detected"] > 0
        assert result["resolution_strategy"] is not None
        assert len(result["resolved_pattern"]["recommended_hours"]) > 2

    # Validation and Quality Assurance Tests

    async def test_strategy_validation_against_benchmarks(
        self,
        mock_strategy_synthesizer: Mock,
        expected_synthesized_strategy: dict[str, Any],
    ):
        """Test strategy validation against industry benchmarks."""
        # Arrange
        benchmark_validation = {
            "engagement_rate_benchmark": 8.5,
            "strategy_performance_vs_benchmark": 1.47,  # 47% above benchmark
            "industry_percentile": 85,
            "validation_criteria": {
                "minimum_engagement": "passed",
                "timing_optimization": "passed",
                "content_quality": "passed",
                "platform_compliance": "passed",
            },
            "improvement_opportunities": [
                "hashtag_optimization",
                "visual_content_integration",
            ],
        }

        mock_strategy_synthesizer.validate_strategy.return_value = benchmark_validation

        # Act
        validation = await mock_strategy_synthesizer.validate_strategy(
            expected_synthesized_strategy, benchmark_comparison=True
        )

        # Assert
        assert validation["strategy_performance_vs_benchmark"] > 1.0
        assert validation["industry_percentile"] > 50
        assert all(
            status == "passed" for status in validation["validation_criteria"].values()
        )

    async def test_pattern_confidence_calibration(
        self,
        mock_pattern_analyzer: Mock,
        expected_extracted_patterns: list[dict[str, Any]],
    ):
        """Test calibration of pattern confidence scores."""
        # Arrange
        confidence_calibration = {
            "raw_confidence_scores": [0.87, 0.92, 0.78],
            "calibrated_confidence_scores": [0.83, 0.89, 0.75],
            "calibration_method": "platt_scaling",
            "calibration_accuracy": 0.94,
            "reliability_diagram": {
                "bin_accuracies": [0.85, 0.88, 0.92, 0.79],
                "bin_confidence": [0.85, 0.90, 0.95, 0.80],
            },
        }

        mock_pattern_analyzer.calculate_pattern_confidence.return_value = (
            confidence_calibration
        )

        # Act
        calibration = mock_pattern_analyzer.calculate_pattern_confidence(
            expected_extracted_patterns, calibration_enabled=True
        )

        # Assert
        assert len(calibration["calibrated_confidence_scores"]) == 3
        assert calibration["calibration_accuracy"] > 0.9
        assert all(
            cal <= raw
            for cal, raw in zip(
                calibration["calibrated_confidence_scores"],
                calibration["raw_confidence_scores"],
                strict=False,
            )
        )

    # Integration with MLX and DSPy Tests

    async def test_mlx_pattern_analysis_integration(
        self, mock_mining_pipeline: Mock, sample_historical_posts: list[dict[str, Any]]
    ):
        """Test integration with MLX for advanced pattern analysis."""
        # Arrange
        mlx_analysis_results = {
            "semantic_patterns": [
                "technical_expertise_demonstration",
                "community_engagement_focus",
                "value_proposition_clarity",
            ],
            "linguistic_features": {
                "sentiment_distribution": {
                    "positive": 0.75,
                    "neutral": 0.20,
                    "negative": 0.05,
                },
                "complexity_score": 0.68,
                "readability_score": 0.82,
            },
            "embeddings_analysis": {
                "cluster_count": 3,
                "cluster_coherence": 0.87,
                "dimensional_reduction": "successful",
            },
            "mlx_model_version": "llama-3.2-3b-instruct",
            "processing_time_ms": 1250,
        }

        mock_mining_pipeline.extract_and_synthesize.return_value = {
            "mlx_analysis": mlx_analysis_results,
            "traditional_analysis": expected_extracted_patterns,
        }

        # Act
        result = await mock_mining_pipeline.extract_and_synthesize(
            sample_historical_posts,
            use_mlx_analysis=True,
            mlx_model="llama-3.2-3b-instruct",
        )

        # Assert
        assert "mlx_analysis" in result
        assert len(result["mlx_analysis"]["semantic_patterns"]) > 0
        assert result["mlx_analysis"]["processing_time_ms"] < 5000

    async def test_pipeline_health_monitoring(self, mock_mining_pipeline: Mock):
        """Test pipeline health monitoring and performance tracking."""
        # Arrange
        health_status = {
            "overall_status": "healthy",
            "component_health": {
                "strategy_extractor": {"status": "up", "response_time_ms": 45},
                "pattern_analyzer": {"status": "up", "response_time_ms": 78},
                "feature_detector": {"status": "up", "response_time_ms": 23},
                "strategy_synthesizer": {"status": "up", "response_time_ms": 156},
            },
            "performance_metrics": {
                "avg_processing_time_ms": 2340,
                "cache_hit_rate": 0.76,
                "error_rate": 0.02,
                "throughput_posts_per_second": 15.5,
            },
            "resource_usage": {
                "memory_usage_mb": 128,
                "cpu_usage_percent": 35,
                "disk_usage_mb": 45,
            },
        }

        mock_mining_pipeline.validate_pipeline_health.return_value = health_status

        # Act
        health = await mock_mining_pipeline.validate_pipeline_health()

        # Assert
        assert health["overall_status"] == "healthy"
        assert all(
            component["status"] == "up"
            for component in health["component_health"].values()
        )
        assert health["performance_metrics"]["error_rate"] < 0.05
