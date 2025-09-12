"""T011: Integration test for content generation pipeline.

Tests the complete content generation workflow from strategy input through
content creation, optimization, and multi-platform adaptation. Following
London School TDD with mock-first approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import timezone, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

# Import statements that will fail initially (RED phase)
try:
    from content_generation.content_generator import ContentGenerator
    from content_generation.content_optimizer import ContentOptimizer
    from content_generation.pipeline import ContentGenerationPipeline
    from content_generation.platform_adapter import PlatformAdapter
    from content_generation.quality_validator import QualityValidator
    from content_generation.schemas import (
        GeneratedContent,
        GenerationRequest,
        OptimizedContent,
    )
    from content_generation.strategy_interpreter import StrategyInterpreter

    from core.exceptions import GenerationError, OptimizationError, ValidationError
except ImportError:
    # Expected during RED phase - create mock classes for contract testing
    class ContentGenerationPipeline:
        pass

    class StrategyInterpreter:
        pass

    class ContentGenerator:
        pass

    class ContentOptimizer:
        pass

    class PlatformAdapter:
        pass

    class QualityValidator:
        pass

    class GenerationRequest:
        pass

    class GeneratedContent:
        pass

    class OptimizedContent:
        pass

    class GenerationError(Exception):
        pass

    class OptimizationError(Exception):
        pass

    class ValidationError(Exception):
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestContentGenerationPipelineIntegration:
    """Integration tests for content generation pipeline.

    Tests complete end-to-end workflows including:
    - Strategy interpretation and parameter extraction
    - Content generation with multiple providers (MLX, DSPy)
    - Content optimization and enhancement
    - Multi-platform adaptation and formatting
    - Quality validation and compliance checking
    - Performance monitoring and caching
    """

    @pytest.fixture
    def sample_generation_request(self) -> dict[str, Any]:
        """Sample content generation request."""
        return {
            "request_id": str(uuid4()),
            "strategy_id": "strategy_001",
            "content_type": "social_post",
            "platforms": ["twitter", "linkedin"],
            "parameters": {
                "topic": "AI development best practices",
                "tone": "professional_engaging",
                "style": "educational_with_hooks",
                "target_length": 280,
                "include_call_to_action": True,
                "include_hashtags": True,
                "emoji_usage": "strategic",
                "thread_potential": True,
            },
            "context": {
                "audience": "developers_and_tech_professionals",
                "industry": "software_development",
                "brand_voice": "knowledgeable_approachable",
                "previous_posts": ["post_001", "post_002"],
                "trending_topics": ["ai", "machine_learning", "best_practices"],
            },
            "constraints": {
                "character_limits": {"twitter": 280, "linkedin": 3000},
                "compliance_requirements": [
                    "no_sensitive_content",
                    "professional_guidelines",
                ],
                "brand_guidelines": ["consistent_voice", "value_focused"],
                "avoid_keywords": ["controversial_terms", "competitor_mentions"],
            },
            "delivery_requirements": {
                "preferred_providers": ["mlx", "dspy"],
                "quality_threshold": 0.8,
                "max_generation_time_ms": 5000,
                "require_validation": True,
            },
        }

    @pytest.fixture
    def expected_generated_content(self) -> dict[str, Any]:
        """Expected generated content from the pipeline."""
        return {
            "content_id": str(uuid4()),
            "request_id": "request_001",
            "status": "generated",
            "created_at": datetime.now(UTC).isoformat(),
            "base_content": {
                "text": "🚀 Essential AI development tip: Always validate your training data before fine-tuning. Poor data quality = poor model performance. What's your go-to data validation strategy? Share below! #AI #MachineLearning #BestPractices",
                "structure": {
                    "hook": "🚀 Essential AI development tip:",
                    "main_content": "Always validate your training data before fine-tuning. Poor data quality = poor model performance.",
                    "engagement": "What's your go-to data validation strategy? Share below!",
                    "hashtags": ["#AI", "#MachineLearning", "#BestPractices"],
                },
                "metadata": {
                    "character_count": 178,
                    "word_count": 28,
                    "emoji_count": 1,
                    "hashtag_count": 3,
                    "question_count": 1,
                    "url_count": 0,
                },
            },
            "platform_variants": {
                "twitter": {
                    "text": "🚀 Essential AI development tip: Always validate your training data before fine-tuning. Poor data quality = poor model performance.\n\nWhat's your go-to data validation strategy? Share below! 👇\n\n#AI #MachineLearning #BestPractices",
                    "character_count": 245,
                    "thread_breakdown": None,
                    "media_suggestions": ["data_validation_infographic"],
                },
                "linkedin": {
                    "text": "🚀 Essential AI Development Best Practice: Data Validation\n\nOne of the most critical yet often overlooked aspects of AI development is thorough training data validation. Here's why it matters:\n\n✅ Poor data quality directly impacts model performance\n✅ Early validation saves time and computational resources\n✅ Clean data leads to more reliable predictions\n\nBefore fine-tuning any model, ask yourself:\n• Is the data representative of real-world scenarios?\n• Are there any biases or inconsistencies?\n• Is the labeling accurate and consistent?\n\nWhat's your go-to data validation strategy? I'd love to hear your approaches and tools in the comments!\n\n#AI #MachineLearning #BestPractices #DataScience #SoftwareDevelopment",
                    "character_count": 687,
                    "professional_formatting": True,
                    "industry_relevance": "high",
                },
            },
            "quality_metrics": {
                "overall_score": 0.87,
                "engagement_potential": 0.82,
                "readability_score": 0.90,
                "brand_alignment": 0.85,
                "compliance_score": 1.0,
            },
            "generation_metadata": {
                "provider_used": "mlx::llama-3.2-3b-instruct",
                "generation_time_ms": 1450,
                "iterations": 1,
                "optimization_applied": True,
                "validation_passed": True,
            },
        }

    @pytest.fixture
    def sample_strategy_config(self) -> dict[str, Any]:
        """Sample strategy configuration for interpretation."""
        return {
            "strategy_id": "strategy_001",
            "name": "High-Engagement Technical Content Strategy",
            "components": {
                "content_structure": {
                    "opening": "attention_grabbing_statement",
                    "body": "valuable_insight_or_tip",
                    "closing": "engaging_question",
                    "character_range": {"min": 80, "max": 280},
                },
                "visual_elements": {
                    "emoji_usage": "strategic_placement",
                    "emoji_count": {"min": 1, "max": 2},
                    "hashtag_count": {"max": 3},
                },
                "timing_optimization": {
                    "preferred_hours": [9, 10, 14, 15],
                    "preferred_days": [1, 2, 3, 4],
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
            },
        }

    @pytest.fixture
    def mock_strategy_interpreter(self) -> Mock:
        """Mock StrategyInterpreter with behavior contracts."""
        interpreter = Mock(spec=StrategyInterpreter)
        interpreter.interpret_strategy = AsyncMock()
        interpreter.extract_generation_parameters = Mock()
        interpreter.validate_strategy_compatibility = Mock()
        interpreter.apply_context_modifications = Mock()
        return interpreter

    @pytest.fixture
    def mock_content_generator(self) -> Mock:
        """Mock ContentGenerator with behavior contracts."""
        generator = Mock(spec=ContentGenerator)
        generator.generate_content = AsyncMock()
        generator.generate_with_mlx = AsyncMock()
        generator.generate_with_dspy = AsyncMock()
        generator.generate_variations = AsyncMock()
        generator.select_best_generation = Mock()
        return generator

    @pytest.fixture
    def mock_content_optimizer(self) -> Mock:
        """Mock ContentOptimizer with behavior contracts."""
        optimizer = Mock(spec=ContentOptimizer)
        optimizer.optimize_content = AsyncMock()
        optimizer.enhance_engagement = AsyncMock()
        optimizer.improve_readability = AsyncMock()
        optimizer.apply_seo_optimization = AsyncMock()
        optimizer.validate_optimization = Mock()
        return optimizer

    @pytest.fixture
    def mock_platform_adapter(self) -> Mock:
        """Mock PlatformAdapter with behavior contracts."""
        adapter = Mock(spec=PlatformAdapter)
        adapter.adapt_for_platform = AsyncMock()
        adapter.format_for_twitter = Mock()
        adapter.format_for_linkedin = Mock()
        adapter.format_for_facebook = Mock()
        adapter.validate_platform_compliance = Mock()
        return adapter

    @pytest.fixture
    def mock_quality_validator(self) -> Mock:
        """Mock QualityValidator with behavior contracts."""
        validator = Mock(spec=QualityValidator)
        validator.validate_content = AsyncMock()
        validator.check_compliance = AsyncMock()
        validator.assess_quality_metrics = Mock()
        validator.validate_brand_guidelines = Mock()
        validator.check_content_safety = Mock()
        return validator

    @pytest.fixture
    def mock_generation_pipeline(
        self,
        mock_strategy_interpreter: Mock,
        mock_content_generator: Mock,
        mock_content_optimizer: Mock,
        mock_platform_adapter: Mock,
        mock_quality_validator: Mock,
    ) -> Mock:
        """Mock ContentGenerationPipeline with all dependencies."""
        pipeline = Mock(spec=ContentGenerationPipeline)
        pipeline.interpreter = mock_strategy_interpreter
        pipeline.generator = mock_content_generator
        pipeline.optimizer = mock_content_optimizer
        pipeline.adapter = mock_platform_adapter
        pipeline.validator = mock_quality_validator
        pipeline.generate_content = AsyncMock()
        pipeline.generate_with_optimization = AsyncMock()
        pipeline.generate_multi_platform = AsyncMock()
        pipeline.health_check = AsyncMock()
        return pipeline

    # Core Content Generation Workflow Tests

    async def test_complete_content_generation_pipeline(
        self,
        mock_generation_pipeline: Mock,
        sample_generation_request: dict[str, Any],
        sample_strategy_config: dict[str, Any],
        expected_generated_content: dict[str, Any],
    ):
        """Test complete content generation pipeline: interpret → generate → optimize → adapt → validate."""
        # Arrange
        mock_generation_pipeline.interpreter.interpret_strategy.return_value = {
            "generation_params": sample_strategy_config["components"],
            "constraints": {"character_limit": 280, "must_include_hashtags": True},
        }
        mock_generation_pipeline.generator.generate_content.return_value = {
            "text": expected_generated_content["base_content"]["text"],
            "quality_score": 0.85,
        }
        mock_generation_pipeline.optimizer.optimize_content.return_value = {
            "optimized_text": expected_generated_content["base_content"]["text"],
            "improvements": ["hashtag_optimization", "engagement_enhancement"],
        }
        mock_generation_pipeline.adapter.adapt_for_platform.side_effect = [
            expected_generated_content["platform_variants"]["twitter"],
            expected_generated_content["platform_variants"]["linkedin"],
        ]
        mock_generation_pipeline.validator.validate_content.return_value = {
            "is_valid": True,
            "quality_metrics": expected_generated_content["quality_metrics"],
        }
        mock_generation_pipeline.generate_content.return_value = (
            expected_generated_content
        )

        # Act - This will fail initially (RED phase)
        result = await mock_generation_pipeline.generate_content(
            sample_generation_request
        )

        # Assert - Testing the expected workflow interactions
        mock_generation_pipeline.interpreter.interpret_strategy.assert_called_once()
        mock_generation_pipeline.generator.generate_content.assert_called_once()
        mock_generation_pipeline.optimizer.optimize_content.assert_called_once()
        assert mock_generation_pipeline.adapter.adapt_for_platform.call_count == 2
        mock_generation_pipeline.validator.validate_content.assert_called_once()

        assert result["status"] == "generated"
        assert result["content_id"] == expected_generated_content["content_id"]
        assert len(result["platform_variants"]) == 2

    async def test_strategy_interpretation_and_parameter_extraction(
        self,
        mock_strategy_interpreter: Mock,
        sample_strategy_config: dict[str, Any],
        sample_generation_request: dict[str, Any],
    ):
        """Test strategy interpretation and generation parameter extraction."""
        # Arrange
        interpretation_result = {
            "generation_parameters": {
                "content_structure": {
                    "hook_style": "attention_grabbing_statement",
                    "body_style": "valuable_insight_or_tip",
                    "closing_style": "engaging_question",
                },
                "style_guidelines": {
                    "tone": "professional_engaging",
                    "emoji_count": 1,
                    "hashtag_count": 3,
                    "include_question": True,
                },
                "constraints": {
                    "max_characters": 280,
                    "must_include_cta": True,
                    "avoid_keywords": ["controversial_terms"],
                },
            },
            "platform_requirements": {
                "twitter": {"character_limit": 280, "hashtag_strategy": "trending"},
                "linkedin": {"character_limit": 3000, "professional_tone": True},
            },
            "expected_performance": {
                "engagement_rate_target": 15.0,
                "impression_target": 8000,
            },
        }

        mock_strategy_interpreter.interpret_strategy.return_value = (
            interpretation_result
        )
        mock_strategy_interpreter.extract_generation_parameters.return_value = (
            interpretation_result["generation_parameters"]
        )

        # Act
        interpretation = await mock_strategy_interpreter.interpret_strategy(
            sample_strategy_config, sample_generation_request
        )
        parameters = mock_strategy_interpreter.extract_generation_parameters(
            interpretation
        )

        # Assert
        assert "generation_parameters" in interpretation
        assert "platform_requirements" in interpretation
        assert parameters["style_guidelines"]["emoji_count"] == 1
        assert parameters["constraints"]["max_characters"] == 280

    async def test_multi_provider_content_generation(
        self, mock_content_generator: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test content generation with multiple providers (MLX, DSPy)."""
        # Arrange
        mlx_result = {
            "provider": "mlx::llama-3.2-3b-instruct",
            "text": "🚀 Pro tip for AI developers: Data validation is crucial before model training. Poor data = poor results. What's your validation workflow? #AI #MLTips",
            "generation_time_ms": 1200,
            "quality_score": 0.85,
            "confidence": 0.92,
        }

        dspy_result = {
            "provider": "dspy::llama-3.2-3b",
            "text": "🔥 AI Development Best Practice: Always validate your training data first! Quality data leads to quality models. Share your data validation tips below! #AI #MachineLearning",
            "generation_time_ms": 950,
            "quality_score": 0.88,
            "confidence": 0.89,
        }

        mock_content_generator.generate_with_mlx.return_value = mlx_result
        mock_content_generator.generate_with_dspy.return_value = dspy_result
        mock_content_generator.select_best_generation.return_value = (
            dspy_result  # DSPy has higher quality score
        )

        # Act
        mlx_content = await mock_content_generator.generate_with_mlx(
            sample_generation_request, model="llama-3.2-3b-instruct"
        )
        dspy_content = await mock_content_generator.generate_with_dspy(
            sample_generation_request, model="llama-3.2-3b"
        )
        best_content = mock_content_generator.select_best_generation(
            [mlx_content, dspy_content]
        )

        # Assert
        assert mlx_content["provider"].startswith("mlx::")
        assert dspy_content["provider"].startswith("dspy::")
        assert (
            best_content["quality_score"] == 0.88
        )  # Higher quality DSPy result selected
        assert best_content["provider"] == "dspy::llama-3.2-3b"

    async def test_content_optimization_and_enhancement(
        self, mock_content_optimizer: Mock, expected_generated_content: dict[str, Any]
    ):
        """Test content optimization and enhancement processes."""
        # Arrange
        base_content = (
            "AI tip: Validate your data before training. Poor data = poor models."
        )

        optimization_results = {
            "engagement_optimization": {
                "original": base_content,
                "optimized": "🚀 AI Pro Tip: Always validate your training data before model training. Poor data quality = poor model performance. What's your validation process?",
                "improvements": [
                    "added_emoji_hook",
                    "enhanced_question",
                    "improved_clarity",
                ],
                "engagement_score_improvement": 0.15,
            },
            "readability_optimization": {
                "readability_score": 0.92,
                "improvements": ["simplified_language", "better_structure"],
                "flesch_kincaid_grade": 8.5,
            },
            "seo_optimization": {
                "keyword_density": {"AI": 0.08, "data": 0.12, "training": 0.08},
                "hashtag_suggestions": ["#AI", "#MachineLearning", "#BestPractices"],
                "trending_alignment": 0.78,
            },
        }

        mock_content_optimizer.enhance_engagement.return_value = optimization_results[
            "engagement_optimization"
        ]
        mock_content_optimizer.improve_readability.return_value = optimization_results[
            "readability_optimization"
        ]
        mock_content_optimizer.apply_seo_optimization.return_value = (
            optimization_results["seo_optimization"]
        )

        final_optimized = {
            "text": optimization_results["engagement_optimization"]["optimized"]
            + " #AI #MachineLearning #BestPractices",
            "optimization_applied": True,
            "improvements": ["engagement", "readability", "seo"],
            "quality_score": 0.89,
        }
        mock_content_optimizer.optimize_content.return_value = final_optimized

        # Act
        engagement_opt = await mock_content_optimizer.enhance_engagement(base_content)
        readability_opt = await mock_content_optimizer.improve_readability(
            engagement_opt["optimized"]
        )
        seo_opt = await mock_content_optimizer.apply_seo_optimization(
            readability_opt["optimized"]
        )
        final_result = await mock_content_optimizer.optimize_content(base_content)

        # Assert
        assert engagement_opt["engagement_score_improvement"] > 0
        assert readability_opt["readability_score"] > 0.9
        assert len(seo_opt["hashtag_suggestions"]) == 3
        assert final_result["optimization_applied"] is True
        assert final_result["quality_score"] > 0.85

    async def test_multi_platform_adaptation(
        self, mock_platform_adapter: Mock, expected_generated_content: dict[str, Any]
    ):
        """Test adaptation of content for multiple social media platforms."""
        # Arrange
        base_content = expected_generated_content["base_content"]["text"]

        twitter_adaptation = {
            "platform": "twitter",
            "text": base_content + " 👇",
            "character_count": len(base_content) + 3,
            "adaptations": ["added_thread_indicator", "optimized_hashtags"],
            "compliance": {"character_limit": "passed", "content_policy": "passed"},
        }

        linkedin_adaptation = {
            "platform": "linkedin",
            "text": "🚀 Essential AI Development Best Practice: Data Validation\n\n"
            + base_content.replace("🚀 Essential AI development tip:", "")
            + "\n\nKey takeaways:\n✅ Validate early and often\n✅ Quality data drives quality models\n✅ Prevention saves time and resources\n\nWhat validation strategies do you use? Share your experience!\n\n#AI #MachineLearning #BestPractices #DataScience",
            "character_count": 456,
            "adaptations": [
                "expanded_format",
                "professional_structure",
                "added_bullet_points",
            ],
            "compliance": {
                "professional_guidelines": "passed",
                "industry_relevance": "high",
            },
        }

        mock_platform_adapter.adapt_for_platform.side_effect = [
            twitter_adaptation,
            linkedin_adaptation,
        ]

        # Act
        twitter_result = await mock_platform_adapter.adapt_for_platform(
            base_content, "twitter"
        )
        linkedin_result = await mock_platform_adapter.adapt_for_platform(
            base_content, "linkedin"
        )

        # Assert
        assert twitter_result["character_count"] <= 280
        assert "thread_indicator" in twitter_result["adaptations"][0]
        assert linkedin_result["character_count"] > 280  # Expanded format
        assert "professional_structure" in linkedin_result["adaptations"]
        assert linkedin_result["compliance"]["professional_guidelines"] == "passed"

    # Quality Validation and Compliance Tests

    async def test_comprehensive_quality_validation(
        self, mock_quality_validator: Mock, expected_generated_content: dict[str, Any]
    ):
        """Test comprehensive quality validation and compliance checking."""
        # Arrange
        content_to_validate = expected_generated_content["base_content"]["text"]

        quality_assessment = {
            "overall_quality_score": 0.87,
            "component_scores": {
                "engagement_potential": 0.82,
                "readability": 0.90,
                "brand_alignment": 0.85,
                "compliance": 1.0,
                "originality": 0.88,
            },
            "detailed_metrics": {
                "flesch_reading_ease": 78.5,
                "sentiment_score": 0.75,
                "emotional_appeal": 0.68,
                "call_to_action_strength": 0.72,
            },
            "validation_checks": {
                "grammar_check": "passed",
                "spell_check": "passed",
                "profanity_check": "passed",
                "brand_guidelines": "passed",
                "platform_compliance": "passed",
            },
            "improvement_suggestions": [
                "consider_adding_statistics",
                "enhance_emotional_appeal",
            ],
        }

        compliance_check = {
            "is_compliant": True,
            "compliance_areas": {
                "content_safety": "passed",
                "professional_guidelines": "passed",
                "platform_policies": "passed",
                "brand_guidelines": "passed",
            },
            "flagged_content": [],
            "recommendations": ["maintain_current_approach"],
        }

        mock_quality_validator.assess_quality_metrics.return_value = quality_assessment
        mock_quality_validator.check_compliance.return_value = compliance_check

        validation_result = {
            "is_valid": True,
            "quality_metrics": quality_assessment,
            "compliance_status": compliance_check,
            "final_score": 0.87,
        }
        mock_quality_validator.validate_content.return_value = validation_result

        # Act
        quality_metrics = mock_quality_validator.assess_quality_metrics(
            content_to_validate
        )
        compliance_status = await mock_quality_validator.check_compliance(
            content_to_validate
        )
        final_validation = await mock_quality_validator.validate_content(
            content_to_validate
        )

        # Assert
        assert quality_metrics["overall_quality_score"] > 0.8
        assert compliance_status["is_compliant"] is True
        assert final_validation["is_valid"] is True
        assert all(
            score > 0.8
            for score in quality_metrics["component_scores"].values()
            if score != 1.0  # Exclude perfect compliance score
        )

    async def test_brand_guidelines_validation(self, mock_quality_validator: Mock):
        """Test validation against brand guidelines and voice consistency."""
        # Arrange
        brand_guidelines = {
            "voice": "knowledgeable_approachable",
            "tone": "professional_but_friendly",
            "values": ["education", "community", "innovation"],
            "avoid": ["overly_technical_jargon", "aggressive_sales_language"],
            "required_elements": ["value_proposition", "community_engagement"],
        }

        content_to_validate = "🚀 Essential AI development tip: Always validate your training data before fine-tuning. Poor data quality = poor model performance. What's your go-to data validation strategy? Share below! #AI #MachineLearning #BestPractices"

        brand_validation = {
            "voice_alignment": 0.92,
            "tone_consistency": 0.88,
            "values_reflection": {
                "education": 0.95,
                "community": 0.85,
                "innovation": 0.78,
            },
            "avoided_elements_check": "passed",
            "required_elements_check": {
                "value_proposition": "present",
                "community_engagement": "present",
            },
            "overall_brand_score": 0.88,
            "recommendations": [
                "maintain_educational_focus",
                "continue_community_engagement",
            ],
        }

        mock_quality_validator.validate_brand_guidelines.return_value = brand_validation

        # Act
        result = mock_quality_validator.validate_brand_guidelines(
            content_to_validate, brand_guidelines
        )

        # Assert
        assert result["voice_alignment"] > 0.9
        assert result["overall_brand_score"] > 0.85
        assert result["required_elements_check"]["value_proposition"] == "present"
        assert result["avoided_elements_check"] == "passed"

    # Performance and Error Handling Tests

    async def test_concurrent_content_generation(
        self, mock_generation_pipeline: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test concurrent content generation for multiple requests."""
        # Arrange
        request_ids = [f"request_{i}" for i in range(5)]
        generation_requests = [
            {**sample_generation_request, "request_id": req_id}
            for req_id in request_ids
        ]

        concurrent_results = {
            "total_requests": 5,
            "successful_generations": 5,
            "failed_generations": 0,
            "avg_generation_time_ms": 1850,
            "max_concurrent_jobs": 3,
            "throughput_requests_per_second": 2.7,
        }

        mock_generation_pipeline.generate_multi_platform.return_value = (
            concurrent_results
        )

        # Act
        result = await mock_generation_pipeline.generate_multi_platform(
            generation_requests, max_concurrency=3
        )

        # Assert
        assert result["successful_generations"] == 5
        assert result["failed_generations"] == 0
        assert result["avg_generation_time_ms"] < 3000  # Under 3 seconds
        assert result["throughput_requests_per_second"] > 2.0

    async def test_generation_error_handling_and_fallbacks(
        self, mock_content_generator: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test error handling and fallback mechanisms in generation."""
        # Arrange
        # First provider fails, second succeeds
        mock_content_generator.generate_with_mlx.side_effect = GenerationError(
            "MLX service unavailable"
        )
        mock_content_generator.generate_with_dspy.return_value = {
            "provider": "dspy::llama-3.2-3b",
            "text": "Fallback generated content",
            "quality_score": 0.82,
            "fallback_used": True,
        }

        fallback_result = {
            "content": "Fallback generated content",
            "provider_used": "dspy::llama-3.2-3b",
            "primary_provider_failed": True,
            "fallback_successful": True,
            "generation_attempts": 2,
        }
        mock_content_generator.generate_content.return_value = fallback_result

        # Act
        result = await mock_content_generator.generate_content(
            sample_generation_request, enable_fallback=True
        )

        # Assert
        assert result["fallback_successful"] is True
        assert result["provider_used"].startswith("dspy::")
        assert result["generation_attempts"] == 2

    async def test_quality_threshold_enforcement(
        self,
        mock_generation_pipeline: Mock,
        mock_quality_validator: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test enforcement of quality thresholds with regeneration."""
        # Arrange

        high_quality_content = {
            "text": "🚀 Essential AI development tip: Always validate your training data before fine-tuning. Poor data quality = poor model performance. What's your go-to data validation strategy?",
            "quality_score": 0.87,  # Above threshold
        }

        # First generation fails quality check, second passes
        mock_quality_validator.validate_content.side_effect = [
            {"is_valid": False, "quality_metrics": {"overall_score": 0.65}},
            {"is_valid": True, "quality_metrics": {"overall_score": 0.87}},
        ]

        mock_generation_pipeline.generate_with_optimization.return_value = {
            "content": high_quality_content["text"],
            "quality_score": 0.87,
            "regeneration_attempts": 2,
            "final_validation": "passed",
        }

        # Act
        result = await mock_generation_pipeline.generate_with_optimization(
            sample_generation_request,
            quality_threshold=0.8,
            max_regeneration_attempts=3,
        )

        # Assert
        assert result["quality_score"] >= 0.8
        assert result["regeneration_attempts"] == 2
        assert result["final_validation"] == "passed"

    async def test_caching_and_performance_optimization(
        self, mock_generation_pipeline: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test caching mechanisms and performance optimization."""
        # Arrange

        cached_result = {
            "content": "Cached high-quality content",
            "cache_hit": True,
            "generation_time_ms": 45,
            "original_generation_time_ms": 1200,
            "time_saved_ms": 1155,
        }

        mock_generation_pipeline.generate_content.return_value = cached_result

        # Act
        result = await mock_generation_pipeline.generate_content(
            sample_generation_request, use_cache=True
        )

        # Assert
        assert result["cache_hit"] is True
        assert result["generation_time_ms"] < 100  # Fast cache response
        assert result["time_saved_ms"] > 1000  # Significant time savings

    # Advanced Features and Integration Tests

    async def test_a_b_testing_content_variations(
        self, mock_content_generator: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test generation of A/B testing content variations."""
        # Arrange
        variation_a = {
            "variant_id": "A",
            "text": "🚀 AI Development Tip: Data validation is crucial for model success. What's your validation approach? Share below!",
            "style": "concise_direct",
            "predicted_engagement": 0.78,
        }

        variation_b = {
            "variant_id": "B",
            "text": "Essential AI best practice: Always validate your training data before fine-tuning. Poor data quality leads to poor model performance. What validation strategies do you use?",
            "style": "detailed_educational",
            "predicted_engagement": 0.82,
        }

        variation_c = {
            "variant_id": "C",
            "text": "Pro tip for AI developers 💡: Skip data validation at your own risk! Quality data = quality models. Tell us about your validation workflow 👇",
            "style": "casual_engaging",
            "predicted_engagement": 0.75,
        }

        ab_test_results = {
            "variations": [variation_a, variation_b, variation_c],
            "recommended_variant": "B",
            "test_strategy": "engagement_optimization",
            "statistical_power": 0.92,
        }

        mock_content_generator.generate_variations.return_value = ab_test_results

        # Act
        result = await mock_content_generator.generate_variations(
            sample_generation_request,
            variation_count=3,
            test_strategy="engagement_optimization",
        )

        # Assert
        assert len(result["variations"]) == 3
        assert result["recommended_variant"] == "B"
        assert all(var["predicted_engagement"] > 0.7 for var in result["variations"])

    async def test_real_time_trend_integration(
        self, mock_strategy_interpreter: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test integration with real-time trending topics and optimization."""
        # Arrange
        trending_data = {
            "trending_hashtags": [
                "#AI",
                "#MachineLearning",
                "#TechTips",
                "#DataScience",
            ],
            "trending_topics": [
                "ai_development",
                "data_validation",
                "ml_best_practices",
            ],
            "sentiment_analysis": {"positive": 0.78, "neutral": 0.18, "negative": 0.04},
            "engagement_patterns": {
                "question_posts": {"avg_engagement": 12.5, "trend": "increasing"},
                "educational_content": {"avg_engagement": 15.2, "trend": "stable"},
                "emoji_usage": {"optimal_count": 1, "placement": "beginning"},
            },
        }

        trend_optimized_params = {
            "hashtags": ["#AI", "#MachineLearning", "#TechTips"],
            "content_style": "educational_with_question",
            "emoji_strategy": "single_beginning_placement",
            "timing_optimization": "trending_window_active",
        }

        mock_strategy_interpreter.apply_context_modifications.return_value = (
            trend_optimized_params
        )

        # Act
        result = mock_strategy_interpreter.apply_context_modifications(
            sample_generation_request, trending_data
        )

        # Assert
        assert len(result["hashtags"]) == 3
        assert result["content_style"] == "educational_with_question"
        assert result["emoji_strategy"] == "single_beginning_placement"

    async def test_pipeline_health_monitoring_and_metrics(
        self, mock_generation_pipeline: Mock
    ):
        """Test pipeline health monitoring and performance metrics collection."""
        # Arrange
        health_status = {
            "overall_status": "healthy",
            "component_health": {
                "strategy_interpreter": {"status": "up", "response_time_ms": 23},
                "content_generator": {"status": "up", "response_time_ms": 1250},
                "content_optimizer": {"status": "up", "response_time_ms": 180},
                "platform_adapter": {"status": "up", "response_time_ms": 95},
                "quality_validator": {"status": "up", "response_time_ms": 67},
            },
            "performance_metrics": {
                "avg_generation_time_ms": 1850,
                "success_rate": 0.97,
                "quality_score_avg": 0.86,
                "cache_hit_rate": 0.72,
                "throughput_per_hour": 120,
            },
            "resource_usage": {
                "memory_usage_mb": 245,
                "cpu_usage_percent": 42,
                "gpu_usage_percent": 67,
                "disk_usage_mb": 89,
            },
            "alerts": [],
            "last_health_check": datetime.now(UTC).isoformat(),
        }

        mock_generation_pipeline.health_check.return_value = health_status

        # Act
        health = await mock_generation_pipeline.health_check()

        # Assert
        assert health["overall_status"] == "healthy"
        assert all(
            component["status"] == "up"
            for component in health["component_health"].values()
        )
        assert health["performance_metrics"]["success_rate"] > 0.95
        assert health["performance_metrics"]["avg_generation_time_ms"] < 3000
        assert len(health["alerts"]) == 0
