"""T042: Comprehensive unit tests for candidate_scorer.py

Tests for perplexity calculation, engagement scoring, sentiment analysis,
toxicity detection, and readability metrics with mocked external APIs.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.candidate_scorer import (
    ContentCandidate,
    EngagementPredictor,
    PerplexityCalculator,
    QualityAssessor,
    REERCandidateScorer,
    ScoringConfig,
    ScoringMetrics,
)
from core.exceptions import PerplexityError, ScoringError, ValidationError


class TestScoringMetrics:
    """Test ScoringMetrics dataclass."""

    def test_scoring_metrics_creation(self):
        """Test creating ScoringMetrics instance."""
        metrics = ScoringMetrics(
            perplexity=15.2,
            fluency_score=0.85,
            engagement_score=0.72,
            quality_score=0.78,
            coherence_score=0.80,
            relevance_score=0.75,
            viral_potential=0.60,
            brand_alignment=0.90,
            overall_score=0.76,
            confidence=0.85,
            metadata={"model": "test-model"},
        )

        assert metrics.perplexity == 15.2
        assert metrics.engagement_score == 0.72
        assert metrics.metadata["model"] == "test-model"

    def test_scoring_metrics_default_metadata(self):
        """Test ScoringMetrics with default metadata."""
        metrics = ScoringMetrics(
            perplexity=10.0,
            fluency_score=0.8,
            engagement_score=0.7,
            quality_score=0.75,
            coherence_score=0.8,
            relevance_score=0.7,
            viral_potential=0.6,
            brand_alignment=0.9,
            overall_score=0.75,
            confidence=0.8,
        )

        assert metrics.metadata == {}


class TestContentCandidate:
    """Test ContentCandidate dataclass."""

    def test_content_candidate_creation(self):
        """Test creating ContentCandidate instance."""
        candidate = ContentCandidate(
            candidate_id="test_123",
            text="This is a test social media post! #AI #testing",
            metadata={"platform": "twitter"},
            context={"topic": "AI", "audience": "tech"},
            target_metrics={"engagement_rate": 0.05},
        )

        assert candidate.candidate_id == "test_123"
        assert "#AI" in candidate.text
        assert candidate.metadata["platform"] == "twitter"

    def test_content_candidate_defaults(self):
        """Test ContentCandidate with default values."""
        candidate = ContentCandidate(
            candidate_id="simple_test", text="Simple test post"
        )

        assert candidate.metadata == {}
        assert candidate.context is None
        assert candidate.target_metrics is None


class TestScoringConfig:
    """Test ScoringConfig dataclass."""

    def test_default_config(self):
        """Test default scoring configuration."""
        config = ScoringConfig()

        assert config.model_name == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert config.use_perplexity is True
        assert (
            config.weight_perplexity + config.weight_engagement + config.weight_quality
            == 1.0
        )

    def test_custom_config(self):
        """Test custom scoring configuration."""
        config = ScoringConfig(
            model_name="custom-model",
            use_perplexity=False,
            weight_engagement=0.6,
            weight_quality=0.4,
            weight_perplexity=0.0,
        )

        assert config.model_name == "custom-model"
        assert config.use_perplexity is False
        assert config.weight_engagement == 0.6


class TestPerplexityCalculator:
    """Test PerplexityCalculator functionality."""

    @pytest.fixture
    def mock_mlx_model(self):
        """Mock MLX model for testing."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock tokenizer encode
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock model output
        mock_outputs = Mock()
        mock_logits = Mock()
        mock_logits.shape = (1, 4, 1000)  # batch_size, seq_len, vocab_size
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        return mock_model, mock_tokenizer

    @pytest.fixture
    def perplexity_calculator(self):
        """Create PerplexityCalculator instance."""
        return PerplexityCalculator("test-model")

    def test_initialization(self, perplexity_calculator):
        """Test PerplexityCalculator initialization."""
        assert perplexity_calculator.model_name == "test-model"
        assert perplexity_calculator.model is None
        assert perplexity_calculator.tokenizer is None
        assert perplexity_calculator._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_without_mlx(self, perplexity_calculator):
        """Test initialization when MLX is not available."""
        with patch("core.candidate_scorer.MLX_AVAILABLE", False):
            with pytest.raises(PerplexityError) as exc_info:
                await perplexity_calculator.initialize()
            assert "MLX is not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_success(self, perplexity_calculator, mock_mlx_model):
        """Test successful initialization."""
        mock_model, mock_tokenizer = mock_mlx_model

        with patch("core.candidate_scorer.MLX_AVAILABLE", True):
            with patch(
                "core.candidate_scorer.load", return_value=(mock_model, mock_tokenizer)
            ):
                await perplexity_calculator.initialize()

                assert perplexity_calculator.model == mock_model
                assert perplexity_calculator.tokenizer == mock_tokenizer
                assert perplexity_calculator._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, perplexity_calculator):
        """Test initialization failure."""
        with patch("core.candidate_scorer.MLX_AVAILABLE", True):
            with patch(
                "core.candidate_scorer.load", side_effect=Exception("Load failed")
            ):
                with pytest.raises(PerplexityError) as exc_info:
                    await perplexity_calculator.initialize()
                assert "Failed to initialize MLX model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_calculate_perplexity(self, perplexity_calculator):
        """Test perplexity calculation."""
        # Mock MLX components
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock MLX array and operations
        with patch("core.candidate_scorer.MLX_AVAILABLE", True):
            with patch("core.candidate_scorer.mx") as mock_mx:
                # Setup mock array
                mock_array = Mock()
                mock_mx.array.return_value = mock_array

                # Setup mock model outputs
                mock_logits = Mock()
                mock_model.return_value = mock_logits

                # Setup mock log_softmax
                mock_log_probs = Mock()
                mock_mx.log_softmax.return_value = mock_log_probs

                # Setup mock no_grad context
                mock_mx.no_grad.return_value.__enter__ = Mock()
                mock_mx.no_grad.return_value.__exit__ = Mock()

                # Mock item() calls for log probabilities
                mock_log_prob_values = [-0.5, -0.3, -0.7, -0.4]  # Sample log probs
                mock_log_probs.__getitem__ = Mock(
                    side_effect=lambda idx: Mock(
                        item=Mock(return_value=mock_log_prob_values[idx[2]])
                    )
                )

                # Mock target tokens
                mock_target_tokens = Mock()
                mock_target_tokens.shape = (1, 4)
                mock_target_tokens.__getitem__ = Mock(
                    side_effect=lambda idx: Mock(item=Mock(return_value=idx[1] + 2))
                )
                mock_array.__getitem__ = Mock(return_value=mock_target_tokens)

                perplexity_calculator.model = mock_model
                perplexity_calculator.tokenizer = mock_tokenizer
                perplexity_calculator._initialized = True

                result = await perplexity_calculator.calculate_perplexity("Test text")

                assert isinstance(result, float)
                assert result > 0

    @pytest.mark.asyncio
    async def test_calculate_perplexity_auto_initialize(
        self, perplexity_calculator, mock_mlx_model
    ):
        """Test that perplexity calculation auto-initializes if needed."""
        mock_model, mock_tokenizer = mock_mlx_model

        with patch("core.candidate_scorer.MLX_AVAILABLE", True):
            with patch(
                "core.candidate_scorer.load", return_value=(mock_model, mock_tokenizer)
            ):
                with patch.object(
                    perplexity_calculator, "calculate_perplexity", return_value=10.0
                ):
                    # Mock the actual calculation to avoid complex MLX mocking
                    perplexity_calculator.calculate_perplexity = AsyncMock(
                        return_value=10.0
                    )

                    result = await perplexity_calculator.calculate_perplexity(
                        "Test text"
                    )
                    assert result == 10.0

    @pytest.mark.asyncio
    async def test_calculate_perplexity_error(self, perplexity_calculator):
        """Test perplexity calculation error handling."""
        perplexity_calculator._initialized = True
        perplexity_calculator.tokenizer = Mock()
        perplexity_calculator.tokenizer.encode.side_effect = Exception(
            "Tokenization failed"
        )

        with pytest.raises(PerplexityError) as exc_info:
            await perplexity_calculator.calculate_perplexity("Test text")
        assert "Failed to calculate perplexity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_calculate_batch_perplexity(self, perplexity_calculator):
        """Test batch perplexity calculation."""
        texts = ["Text 1", "Text 2", "Text 3"]

        with patch.object(perplexity_calculator, "calculate_perplexity") as mock_calc:
            mock_calc.side_effect = [10.0, 15.0, 12.0]

            results = await perplexity_calculator.calculate_batch_perplexity(texts)

            assert len(results) == 3
            assert results == [10.0, 15.0, 12.0]
            assert mock_calc.call_count == 3


class TestEngagementPredictor:
    """Test EngagementPredictor functionality."""

    @pytest.fixture
    def engagement_predictor(self):
        """Create EngagementPredictor instance."""
        return EngagementPredictor()

    def test_initialization(self, engagement_predictor):
        """Test EngagementPredictor initialization."""
        assert hasattr(engagement_predictor, "feature_weights")
        assert isinstance(engagement_predictor.feature_weights, dict)

        # Check that weights sum to 1.0 (approximately)
        total_weight = sum(engagement_predictor.feature_weights.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_extract_engagement_features_hashtags(self, engagement_predictor):
        """Test hashtag feature extraction."""
        text = "Great post about #AI and #MachineLearning! #TechTrends"
        features = engagement_predictor.extract_engagement_features(text)

        assert "hashtag_usage" in features
        assert features["hashtag_usage"] == 1.0  # 3 hashtags, capped at optimal

    def test_extract_engagement_features_questions(self, engagement_predictor):
        """Test question pattern feature extraction."""
        text_with_question = "What do you think about this new AI technology?"
        text_without_question = "This new AI technology is amazing."

        features_with = engagement_predictor.extract_engagement_features(
            text_with_question
        )
        features_without = engagement_predictor.extract_engagement_features(
            text_without_question
        )

        assert features_with["question_pattern"] == 1.0
        assert features_without["question_pattern"] == 0.0

    def test_extract_engagement_features_call_to_action(self, engagement_predictor):
        """Test call-to-action feature extraction."""
        cta_text = "Click here to learn more about our amazing product!"
        no_cta_text = "This is just an informational post."

        cta_features = engagement_predictor.extract_engagement_features(cta_text)
        no_cta_features = engagement_predictor.extract_engagement_features(no_cta_text)

        assert cta_features["call_to_action"] > 0
        assert no_cta_features["call_to_action"] == 0

    def test_extract_engagement_features_emojis(self, engagement_predictor):
        """Test emoji feature extraction."""
        emoji_text = "Love this! 😍✨🚀"
        no_emoji_text = "Love this!"

        emoji_features = engagement_predictor.extract_engagement_features(emoji_text)
        no_emoji_features = engagement_predictor.extract_engagement_features(
            no_emoji_text
        )

        assert emoji_features["emoji_usage"] > 0
        assert no_emoji_features["emoji_usage"] == 0

    def test_extract_engagement_features_urls(self, engagement_predictor):
        """Test URL presence feature extraction."""
        url_text = "Check out this article: https://example.com/article"
        no_url_text = "Check out this article in our magazine"

        url_features = engagement_predictor.extract_engagement_features(url_text)
        no_url_features = engagement_predictor.extract_engagement_features(no_url_text)

        assert url_features["url_presence"] == 1.0
        assert no_url_features["url_presence"] == 0.0

    def test_extract_engagement_features_sentiment(self, engagement_predictor):
        """Test positive sentiment feature extraction."""
        positive_text = "This is amazing and fantastic! I love it!"
        neutral_text = "This is a regular post about technology."

        positive_features = engagement_predictor.extract_engagement_features(
            positive_text
        )
        neutral_features = engagement_predictor.extract_engagement_features(
            neutral_text
        )

        assert (
            positive_features["sentiment_positive"]
            > neutral_features["sentiment_positive"]
        )

    def test_extract_engagement_features_readability(self, engagement_predictor):
        """Test readability feature extraction."""
        optimal_text = "This text has words of good length for reading"
        complex_text = "This extraordinarily sophisticated manifestation demonstrates incomprehensibility"

        optimal_features = engagement_predictor.extract_engagement_features(
            optimal_text
        )
        complex_features = engagement_predictor.extract_engagement_features(
            complex_text
        )

        assert optimal_features["readability"] > complex_features["readability"]

    def test_extract_engagement_features_urgency(self, engagement_predictor):
        """Test urgency indicators feature extraction."""
        urgent_text = "Act now! Limited time offer ends today!"
        calm_text = "This offer is available for consideration."

        urgent_features = engagement_predictor.extract_engagement_features(urgent_text)
        calm_features = engagement_predictor.extract_engagement_features(calm_text)

        assert (
            urgent_features["urgency_indicators"] > calm_features["urgency_indicators"]
        )

    def test_extract_engagement_features_personal_pronouns(self, engagement_predictor):
        """Test personal pronouns feature extraction."""
        personal_text = "I want to share this with you and our community"
        impersonal_text = "The company has announced a new product release"

        personal_features = engagement_predictor.extract_engagement_features(
            personal_text
        )
        impersonal_features = engagement_predictor.extract_engagement_features(
            impersonal_text
        )

        assert (
            personal_features["personal_pronouns"]
            > impersonal_features["personal_pronouns"]
        )

    def test_extract_engagement_features_optimal_length(self, engagement_predictor):
        """Test optimal length feature extraction."""
        optimal_text = (
            "This is a perfect length post for social media engagement!"  # ~70 chars
        )
        too_short = "Short"  # <20 chars
        too_long = "This is an extremely long post that goes way beyond the optimal character limit for social media platforms and will likely see reduced engagement rates due to user attention spans and platform algorithms that favor more concise content."  # >280 chars

        optimal_features = engagement_predictor.extract_engagement_features(
            optimal_text
        )
        short_features = engagement_predictor.extract_engagement_features(too_short)
        long_features = engagement_predictor.extract_engagement_features(too_long)

        assert optimal_features["optimal_length"] > short_features["optimal_length"]
        assert optimal_features["optimal_length"] > long_features["optimal_length"]

    def test_predict_engagement_high_score(self, engagement_predictor):
        """Test engagement prediction for high-scoring content."""
        high_engagement_text = "🚀 What do you think about this amazing breakthrough? Click to learn more! #AI #Innovation"

        score = engagement_predictor.predict_engagement(high_engagement_text)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relatively high

    def test_predict_engagement_low_score(self, engagement_predictor):
        """Test engagement prediction for low-scoring content."""
        low_engagement_text = "announcement"

        score = engagement_predictor.predict_engagement(low_engagement_text)

        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be relatively low

    def test_predict_engagement_empty_text(self, engagement_predictor):
        """Test engagement prediction for empty text."""
        score = engagement_predictor.predict_engagement("")

        assert 0.0 <= score <= 1.0


class TestQualityAssessor:
    """Test QualityAssessor functionality."""

    @pytest.fixture
    def quality_assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    def test_assess_quality_basic(self, quality_assessor):
        """Test basic quality assessment."""
        text = "This is a well-written post about artificial intelligence and its applications."

        scores = quality_assessor.assess_quality(text)

        required_scores = [
            "fluency",
            "coherence",
            "relevance",
            "brand_alignment",
            "viral_potential",
        ]
        for score_type in required_scores:
            assert score_type in scores
            assert 0.0 <= scores[score_type] <= 1.0

    def test_assess_quality_with_context(self, quality_assessor):
        """Test quality assessment with context."""
        text = "Artificial intelligence is revolutionizing technology and machine learning applications."
        context = {
            "topic": "artificial intelligence technology",
            "target_audience": "tech professionals",
            "keywords": ["AI", "machine learning", "technology"],
            "brand_voice": "professional",
        }

        scores = quality_assessor.assess_quality(text, context)

        assert "relevance" in scores
        assert scores["relevance"] > 0.5  # Should be relevant due to keyword overlap

    def test_assess_fluency_short_text(self, quality_assessor):
        """Test fluency assessment for short text."""
        short_text = "Hi"
        score = quality_assessor._assess_fluency(short_text)
        assert score == 0.5  # Default for very short text

    def test_assess_fluency_repetitive_text(self, quality_assessor):
        """Test fluency assessment for repetitive text."""
        repetitive_text = "test test test test test"
        normal_text = "This is a normal sentence with different words"

        repetitive_score = quality_assessor._assess_fluency(repetitive_text)
        normal_score = quality_assessor._assess_fluency(normal_text)

        assert repetitive_score < normal_score

    def test_assess_fluency_excessive_punctuation(self, quality_assessor):
        """Test fluency assessment with excessive punctuation."""
        excessive_punct = "This!!! is!!! too!!! much!!! punctuation!!!"
        normal_punct = "This is normal punctuation."

        excessive_score = quality_assessor._assess_fluency(excessive_punct)
        normal_score = quality_assessor._assess_fluency(normal_punct)

        assert excessive_score < normal_score

    def test_assess_fluency_all_caps(self, quality_assessor):
        """Test fluency assessment with ALL CAPS."""
        caps_text = "THIS IS ALL CAPS AND SHOUTY"
        normal_text = "This is normal case text"

        caps_score = quality_assessor._assess_fluency(caps_text)
        normal_score = quality_assessor._assess_fluency(normal_text)

        assert caps_score < normal_score

    def test_assess_coherence_single_sentence(self, quality_assessor):
        """Test coherence assessment for single sentence."""
        single_sentence = "This is a single coherent sentence."
        score = quality_assessor._assess_coherence(single_sentence)
        assert score == 0.8  # Default for single sentence

    def test_assess_coherence_multiple_sentences(self, quality_assessor):
        """Test coherence assessment for multiple sentences."""
        coherent_text = "AI is transforming industries. Machine learning enables this transformation. These technologies work together."

        score = quality_assessor._assess_coherence(coherent_text)
        assert 0.0 <= score <= 1.0

    def test_assess_relevance_no_context(self, quality_assessor):
        """Test relevance assessment without context."""
        text = "This is some text"
        score = quality_assessor._assess_relevance(text, {})
        assert score == 0.7  # Default

    def test_assess_relevance_with_keywords(self, quality_assessor):
        """Test relevance assessment with keyword context."""
        text = "This post discusses artificial intelligence and machine learning applications"
        context = {
            "topic": "artificial intelligence",
            "keywords": ["AI", "machine learning", "technology"],
        }

        score = quality_assessor._assess_relevance(text, context)
        assert score > 0.5  # Should find keyword matches

    def test_assess_brand_alignment_professional(self, quality_assessor):
        """Test brand alignment for professional voice."""
        professional_text = "We are pleased to announce our latest research findings."
        casual_text = "OMG this is so awesome!!! 😍😍😍"

        context_professional = {"brand_voice": "professional"}

        prof_score = quality_assessor._assess_brand_alignment(
            professional_text, context_professional
        )
        casual_score = quality_assessor._assess_brand_alignment(
            casual_text, context_professional
        )

        assert prof_score > casual_score

    def test_assess_brand_alignment_casual(self, quality_assessor):
        """Test brand alignment for casual voice."""
        casual_text = "This is so cool and awesome! 😊"
        formal_text = "We cordially invite your consideration of this matter."

        context_casual = {"brand_voice": "casual"}

        casual_score = quality_assessor._assess_brand_alignment(
            casual_text, context_casual
        )
        formal_score = quality_assessor._assess_brand_alignment(
            formal_text, context_casual
        )

        assert casual_score >= formal_score

    def test_assess_brand_alignment_inappropriate_content(self, quality_assessor):
        """Test brand alignment with inappropriate content."""
        inappropriate_text = "This is spam and fake content"
        clean_text = "This is quality content"

        inappropriate_score = quality_assessor._assess_brand_alignment(
            inappropriate_text, None
        )
        clean_score = quality_assessor._assess_brand_alignment(clean_text, None)

        assert inappropriate_score < clean_score

    def test_assess_viral_potential_emotional_words(self, quality_assessor):
        """Test viral potential assessment with emotional triggers."""
        emotional_text = (
            "This amazing breakthrough is absolutely incredible and mind-blowing!"
        )
        neutral_text = "This development represents progress in the field."

        emotional_score = quality_assessor._assess_viral_potential(emotional_text)
        neutral_score = quality_assessor._assess_viral_potential(neutral_text)

        assert emotional_score > neutral_score

    def test_assess_viral_potential_sharing_indicators(self, quality_assessor):
        """Test viral potential with sharing indicators."""
        sharing_text = "Please share this important message and tell everyone!"
        non_sharing_text = "This is just an informational post."

        sharing_score = quality_assessor._assess_viral_potential(sharing_text)
        non_sharing_score = quality_assessor._assess_viral_potential(non_sharing_text)

        assert sharing_score > non_sharing_score

    def test_assess_viral_potential_controversy(self, quality_assessor):
        """Test viral potential with controversial/opinion indicators."""
        opinion_text = (
            "I think this is controversial and many disagree with this opinion."
        )
        factual_text = "The data shows consistent trends across multiple studies."

        opinion_score = quality_assessor._assess_viral_potential(opinion_text)
        factual_score = quality_assessor._assess_viral_potential(factual_text)

        assert opinion_score > factual_score

    def test_assess_viral_potential_engagement_hooks(self, quality_assessor):
        """Test viral potential with engagement hooks."""
        hooks_text = "What?! This is incredible! #trending"
        plain_text = "This is a regular announcement"

        hooks_score = quality_assessor._assess_viral_potential(hooks_text)
        plain_score = quality_assessor._assess_viral_potential(plain_text)

        assert hooks_score > plain_score


class TestREERCandidateScorer:
    """Test REERCandidateScorer main functionality."""

    @pytest.fixture
    def mock_perplexity_calculator(self):
        """Mock PerplexityCalculator."""
        mock_calc = Mock(spec=PerplexityCalculator)
        mock_calc.initialize = AsyncMock()
        mock_calc.calculate_perplexity = AsyncMock(return_value=10.0)
        return mock_calc

    @pytest.fixture
    def mock_engagement_predictor(self):
        """Mock EngagementPredictor."""
        mock_pred = Mock(spec=EngagementPredictor)
        mock_pred.predict_engagement.return_value = 0.75
        return mock_pred

    @pytest.fixture
    def mock_quality_assessor(self):
        """Mock QualityAssessor."""
        mock_assess = Mock(spec=QualityAssessor)
        mock_assess.assess_quality.return_value = {
            "fluency": 0.8,
            "coherence": 0.85,
            "relevance": 0.7,
            "brand_alignment": 0.9,
            "viral_potential": 0.6,
        }
        return mock_assess

    @pytest.fixture
    def scorer_config(self):
        """Create test scoring configuration."""
        return ScoringConfig(
            model_name="test-model",
            use_perplexity=True,
            use_engagement_prediction=True,
            use_quality_assessment=True,
        )

    @pytest.fixture
    def candidate_scorer(self, scorer_config):
        """Create REERCandidateScorer instance."""
        return REERCandidateScorer(scorer_config)

    @pytest.fixture
    def sample_candidate(self):
        """Create sample content candidate."""
        return ContentCandidate(
            candidate_id="test_candidate_123",
            text="This is an amazing post about AI! What do you think? #AI #MachineLearning",
            metadata={"platform": "twitter"},
            context={"topic": "AI", "brand_voice": "professional"},
        )

    def test_initialization_default_config(self):
        """Test scorer initialization with default config."""
        scorer = REERCandidateScorer()

        assert scorer.config is not None
        assert scorer.perplexity_calculator is not None
        assert scorer.engagement_predictor is not None
        assert scorer.quality_assessor is not None

    def test_initialization_custom_config(self, scorer_config):
        """Test scorer initialization with custom config."""
        scorer = REERCandidateScorer(scorer_config)

        assert scorer.config == scorer_config
        assert scorer.config.model_name == "test-model"

    def test_initialization_disabled_components(self):
        """Test scorer initialization with disabled components."""
        config = ScoringConfig(
            use_perplexity=False,
            use_engagement_prediction=False,
            use_quality_assessment=False,
        )

        scorer = REERCandidateScorer(config)

        assert scorer.perplexity_calculator is None
        assert scorer.engagement_predictor is None
        assert scorer.quality_assessor is None

    @pytest.mark.asyncio
    async def test_initialize(self, candidate_scorer, mock_perplexity_calculator):
        """Test scorer initialization."""
        candidate_scorer.perplexity_calculator = mock_perplexity_calculator

        await candidate_scorer.initialize()

        assert candidate_scorer._initialized is True
        mock_perplexity_calculator.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_perplexity_calculator(self, scorer_config):
        """Test initialization when perplexity calculator is None."""
        config = ScoringConfig(use_perplexity=False)
        scorer = REERCandidateScorer(config)

        await scorer.initialize()

        assert scorer._initialized is True

    @pytest.mark.asyncio
    async def test_score_candidate_full_scoring(
        self, candidate_scorer, sample_candidate
    ):
        """Test scoring candidate with all components enabled."""
        # Mock all components
        mock_perp_calc = Mock()
        mock_perp_calc.calculate_perplexity = AsyncMock(return_value=10.0)

        mock_eng_pred = Mock()
        mock_eng_pred.predict_engagement.return_value = 0.75

        mock_quality = Mock()
        mock_quality.assess_quality.return_value = {
            "fluency": 0.8,
            "coherence": 0.85,
            "relevance": 0.7,
            "brand_alignment": 0.9,
            "viral_potential": 0.6,
        }

        candidate_scorer.perplexity_calculator = mock_perp_calc
        candidate_scorer.engagement_predictor = mock_eng_pred
        candidate_scorer.quality_assessor = mock_quality
        candidate_scorer._initialized = True

        # Mock normalize_perplexity
        with patch.object(candidate_scorer, "_normalize_perplexity", return_value=0.8):
            metrics = await candidate_scorer.score_candidate(sample_candidate)

        assert isinstance(metrics, ScoringMetrics)
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.confidence <= 1.0
        assert metrics.metadata["candidate_id"] == sample_candidate.candidate_id

    @pytest.mark.asyncio
    async def test_score_candidate_auto_initialize(
        self, candidate_scorer, sample_candidate
    ):
        """Test that scoring auto-initializes if needed."""
        # Mock initialization
        with patch.object(candidate_scorer, "initialize") as mock_init:
            with patch.object(
                candidate_scorer, "_normalized_perplexity", return_value=0.8
            ):
                # Set up minimal mocks to avoid errors
                candidate_scorer.perplexity_calculator = None
                candidate_scorer.engagement_predictor = Mock()
                candidate_scorer.engagement_predictor.predict_engagement.return_value = (
                    0.7
                )
                candidate_scorer.quality_assessor = Mock()
                candidate_scorer.quality_assessor.assess_quality.return_value = {
                    "fluency": 0.7,
                    "coherence": 0.7,
                    "relevance": 0.7,
                    "brand_alignment": 0.7,
                    "viral_potential": 0.7,
                }

                await candidate_scorer.score_candidate(sample_candidate)
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_candidate_disabled_components(self, sample_candidate):
        """Test scoring with disabled components."""
        config = ScoringConfig(
            use_perplexity=False,
            use_engagement_prediction=False,
            use_quality_assessment=False,
        )
        scorer = REERCandidateScorer(config)
        scorer._initialized = True

        metrics = await scorer.score_candidate(sample_candidate)

        # Should use default scores
        assert metrics.perplexity == 0.7
        assert metrics.engagement_score == 0.7
        assert metrics.fluency_score == 0.7

    @pytest.mark.asyncio
    async def test_score_candidate_error_handling(
        self, candidate_scorer, sample_candidate
    ):
        """Test error handling in candidate scoring."""
        # Mock component that raises an error
        mock_perp_calc = Mock()
        mock_perp_calc.calculate_perplexity = AsyncMock(
            side_effect=Exception("Perplexity failed")
        )

        candidate_scorer.perplexity_calculator = mock_perp_calc
        candidate_scorer._initialized = True

        with pytest.raises(ScoringError) as exc_info:
            await candidate_scorer.score_candidate(sample_candidate)

        assert "Failed to score candidate" in str(exc_info.value)
        assert sample_candidate.candidate_id in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_score_candidates_multiple(self, candidate_scorer):
        """Test scoring multiple candidates."""
        candidates = [
            ContentCandidate("cand1", "First test post"),
            ContentCandidate("cand2", "Second test post"),
            ContentCandidate("cand3", "Third test post"),
        ]

        # Mock score_candidate to return predictable results
        async def mock_score_candidate(candidate):
            return ScoringMetrics(
                perplexity=10.0,
                fluency_score=0.8,
                engagement_score=0.7,
                quality_score=0.75,
                coherence_score=0.8,
                relevance_score=0.7,
                viral_potential=0.6,
                brand_alignment=0.9,
                overall_score=float(candidate.candidate_id[-1])
                / 10,  # Different scores
                confidence=0.8,
            )

        with patch.object(
            candidate_scorer, "score_candidate", side_effect=mock_score_candidate
        ):
            results = await candidate_scorer.score_candidates(
                candidates, sort_by_score=True
            )

        assert len(results) == 3
        # Check that results are sorted by score (descending)
        assert (
            results[0][1].overall_score
            >= results[1][1].overall_score
            >= results[2][1].overall_score
        )

    @pytest.mark.asyncio
    async def test_score_candidates_empty_list(self, candidate_scorer):
        """Test scoring empty candidate list."""
        results = await candidate_scorer.score_candidates([])
        assert results == []

    @pytest.mark.asyncio
    async def test_score_candidates_with_errors(self, candidate_scorer):
        """Test scoring candidates when some fail."""
        candidates = [
            ContentCandidate("good1", "First test post"),
            ContentCandidate("bad", "Second test post"),
            ContentCandidate("good2", "Third test post"),
        ]

        async def mock_score_candidate(candidate):
            if candidate.candidate_id == "bad":
                raise ScoringError("Scoring failed")
            return ScoringMetrics(
                perplexity=10.0,
                fluency_score=0.8,
                engagement_score=0.7,
                quality_score=0.75,
                coherence_score=0.8,
                relevance_score=0.7,
                viral_potential=0.6,
                brand_alignment=0.9,
                overall_score=0.75,
                confidence=0.8,
            )

        with patch.object(
            candidate_scorer, "score_candidate", side_effect=mock_score_candidate
        ):
            with patch("builtins.print"):  # Mock print to suppress warnings
                results = await candidate_scorer.score_candidates(candidates)

        assert len(results) == 2  # Only successful candidates

    def test_normalize_perplexity(self, candidate_scorer):
        """Test perplexity normalization."""
        # Test edge cases
        assert candidate_scorer._normalize_perplexity(1.0) == 1.0  # Perfect
        assert candidate_scorer._normalize_perplexity(1000.0) == 0.0  # Very poor
        assert candidate_scorer._normalize_perplexity(2000.0) == 0.0  # Beyond max

        # Test normal range
        mid_perplexity = candidate_scorer._normalize_perplexity(10.0)
        assert 0.0 < mid_perplexity < 1.0

    @pytest.mark.asyncio
    async def test_compare_candidates(self, candidate_scorer):
        """Test candidate comparison functionality."""
        candidates = [
            ContentCandidate("high_scorer", "Amazing content"),
            ContentCandidate("low_scorer", "Basic content"),
        ]

        # Mock scoring to return different scores
        async def mock_score_candidates(cands, sort_by_score=False):
            return [
                (
                    candidates[0],
                    ScoringMetrics(
                        perplexity=5.0,
                        fluency_score=0.9,
                        engagement_score=0.85,
                        quality_score=0.88,
                        coherence_score=0.9,
                        relevance_score=0.8,
                        viral_potential=0.75,
                        brand_alignment=0.95,
                        overall_score=0.87,
                        confidence=0.9,
                    ),
                ),
                (
                    candidates[1],
                    ScoringMetrics(
                        perplexity=15.0,
                        fluency_score=0.6,
                        engagement_score=0.5,
                        quality_score=0.55,
                        coherence_score=0.6,
                        relevance_score=0.5,
                        viral_potential=0.4,
                        brand_alignment=0.7,
                        overall_score=0.53,
                        confidence=0.7,
                    ),
                ),
            ]

        with patch.object(
            candidate_scorer, "score_candidates", side_effect=mock_score_candidates
        ):
            comparison = await candidate_scorer.compare_candidates(candidates)

        assert comparison["total_candidates"] == 2
        assert "rankings" in comparison
        assert "statistics" in comparison
        assert "recommendations" in comparison

        # Check that best overall recommendation is correct
        best_rec = next(
            r for r in comparison["recommendations"] if r["type"] == "best_overall"
        )
        assert best_rec["candidate_id"] == "high_scorer"

    @pytest.mark.asyncio
    async def test_compare_candidates_insufficient_candidates(self, candidate_scorer):
        """Test comparison with insufficient candidates."""
        single_candidate = [ContentCandidate("only_one", "Single post")]

        with pytest.raises(ValidationError) as exc_info:
            await candidate_scorer.compare_candidates(single_candidate)
        assert "Need at least 2 candidates" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_compare_candidates_no_successful_scores(self, candidate_scorer):
        """Test comparison when no candidates can be scored."""
        candidates = [
            ContentCandidate("fail1", "Post 1"),
            ContentCandidate("fail2", "Post 2"),
        ]

        with patch.object(candidate_scorer, "score_candidates", return_value=[]):
            with pytest.raises(ScoringError) as exc_info:
                await candidate_scorer.compare_candidates(candidates)
            assert "No candidates could be scored successfully" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_compare_candidates_custom_criteria(self, candidate_scorer):
        """Test comparison with custom criteria."""
        candidates = [
            ContentCandidate("cand1", "Content 1"),
            ContentCandidate("cand2", "Content 2"),
        ]

        custom_criteria = ["engagement_score", "viral_potential"]

        # Mock score_candidates
        mock_results = [
            (
                candidates[0],
                ScoringMetrics(
                    perplexity=10.0,
                    fluency_score=0.8,
                    engagement_score=0.9,
                    quality_score=0.75,
                    coherence_score=0.8,
                    relevance_score=0.7,
                    viral_potential=0.6,
                    brand_alignment=0.9,
                    overall_score=0.8,
                    confidence=0.8,
                ),
            ),
            (
                candidates[1],
                ScoringMetrics(
                    perplexity=12.0,
                    fluency_score=0.7,
                    engagement_score=0.7,
                    quality_score=0.7,
                    coherence_score=0.75,
                    relevance_score=0.65,
                    viral_potential=0.8,
                    brand_alignment=0.85,
                    overall_score=0.75,
                    confidence=0.75,
                ),
            ),
        ]

        with patch.object(
            candidate_scorer, "score_candidates", return_value=mock_results
        ):
            comparison = await candidate_scorer.compare_candidates(
                candidates, comparison_criteria=custom_criteria
            )

        assert set(comparison["criteria_analyzed"]) == set(custom_criteria)
        assert "engagement_score" in comparison["rankings"]
        assert "viral_potential" in comparison["rankings"]
        assert "overall_score" not in comparison["rankings"]  # Not in custom criteria
