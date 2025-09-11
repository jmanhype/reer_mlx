"""T016: Candidate scorer with perplexity calculation implementation.

Provides comprehensive scoring for social media content candidates using
perplexity calculation, engagement prediction, and multi-dimensional
quality assessment. Integrates with MLX models for efficient evaluation.
"""

import asyncio
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from .exceptions import ScoringError, PerplexityError, ValidationError


@dataclass
class ScoringMetrics:
    """Container for various scoring metrics."""

    perplexity: float
    fluency_score: float
    engagement_score: float
    quality_score: float
    coherence_score: float
    relevance_score: float
    viral_potential: float
    brand_alignment: float
    overall_score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentCandidate:
    """Represents a content candidate for scoring."""

    candidate_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    target_metrics: Optional[Dict[str, float]] = None


@dataclass
class ScoringConfig:
    """Configuration for scoring behavior."""

    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    max_sequence_length: int = 512
    use_perplexity: bool = True
    use_engagement_prediction: bool = True
    use_quality_assessment: bool = True
    weight_perplexity: float = 0.3
    weight_engagement: float = 0.4
    weight_quality: float = 0.3
    temperature: float = 0.7
    batch_size: int = 4


class PerplexityCalculator:
    """Calculates perplexity using MLX language models."""

    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        """Initialize perplexity calculator.

        Args:
            model_name: MLX model name for perplexity calculation
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MLX model and tokenizer."""
        if not MLX_AVAILABLE:
            raise PerplexityError("MLX is not available. Please install mlx-lm.")

        try:
            # Load model and tokenizer
            self.model, self.tokenizer = load(self.model_name)
            self._initialized = True
        except Exception as e:
            raise PerplexityError(
                f"Failed to initialize MLX model {self.model_name}: {str(e)}",
                original_error=e,
            )

    async def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text.

        Args:
            text: Input text to calculate perplexity for

        Returns:
            Perplexity value (lower is better)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Convert to MLX array
            input_ids = mx.array(tokens)[None, :]  # Add batch dimension

            # Get model outputs
            with mx.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Calculate log probabilities
            log_probs = mx.log_softmax(logits, axis=-1)

            # Get probabilities for actual tokens (shifted by 1)
            target_tokens = input_ids[:, 1:]  # Remove first token
            log_probs = log_probs[:, :-1, :]  # Remove last prediction

            # Gather log probabilities for actual tokens
            batch_size, seq_len = target_tokens.shape
            token_log_probs = []

            for i in range(seq_len):
                token_id = target_tokens[0, i].item()
                token_log_prob = log_probs[0, i, token_id].item()
                token_log_probs.append(token_log_prob)

            # Calculate perplexity
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)
            perplexity = math.exp(-avg_log_prob)

            return float(perplexity)

        except Exception as e:
            raise PerplexityError(
                f"Failed to calculate perplexity: {str(e)}",
                details={"text_length": len(text), "model": self.model_name},
                original_error=e,
            )

    async def calculate_batch_perplexity(self, texts: List[str]) -> List[float]:
        """Calculate perplexity for multiple texts.

        Args:
            texts: List of texts to calculate perplexity for

        Returns:
            List of perplexity values
        """
        perplexities = []

        for text in texts:
            perplexity = await self.calculate_perplexity(text)
            perplexities.append(perplexity)

        return perplexities


class EngagementPredictor:
    """Predicts engagement metrics based on content features."""

    def __init__(self):
        """Initialize engagement predictor."""
        self.feature_weights = {
            # Content structure weights
            "hashtag_usage": 0.15,
            "question_pattern": 0.12,
            "call_to_action": 0.18,
            "emoji_usage": 0.08,
            "url_presence": 0.05,
            # Linguistic features
            "sentiment_positive": 0.10,
            "readability": 0.08,
            "urgency_indicators": 0.07,
            "personal_pronouns": 0.05,
            # Technical features
            "optimal_length": 0.12,
            "keyword_density": 0.00,  # Intentionally 0 to reach 1.0
        }

        # Ensure weights sum to 1.0
        total_weight = sum(self.feature_weights.values())
        if total_weight != 1.0:
            # Normalize weights
            for key in self.feature_weights:
                self.feature_weights[key] /= total_weight

    def extract_engagement_features(self, text: str) -> Dict[str, float]:
        """Extract features that correlate with engagement."""
        features = {}

        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)

        # Content structure features
        features["hashtag_usage"] = min(
            text.count("#") / 3, 1.0
        )  # Optimal around 2-3 hashtags
        features["question_pattern"] = 1.0 if "?" in text else 0.0

        # Call to action detection
        cta_patterns = [
            r"\b(click|try|check|visit|download|learn|sign up|get|buy|shop)\b",
            r"\b(see more|read more|link in bio|swipe up)\b",
        ]
        cta_score = sum(
            1 for pattern in cta_patterns if re.search(pattern, text.lower())
        )
        features["call_to_action"] = min(cta_score / 2, 1.0)

        # Emoji usage (Unicode characters above ASCII range)
        emoji_count = sum(1 for char in text if ord(char) > 127)
        features["emoji_usage"] = min(emoji_count / 3, 1.0)

        # URL presence
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        features["url_presence"] = 1.0 if re.search(url_pattern, text) else 0.0

        # Sentiment indicators (simple positive words)
        positive_words = {
            "amazing",
            "awesome",
            "great",
            "excellent",
            "fantastic",
            "wonderful",
            "love",
            "excited",
            "happy",
            "thrilled",
            "perfect",
            "incredible",
            "best",
            "brilliant",
            "outstanding",
            "superb",
            "remarkable",
        }
        text_words = set(text.lower().split())
        positive_count = len(text_words.intersection(positive_words))
        features["sentiment_positive"] = min(positive_count / 3, 1.0)

        # Readability (simple measure)
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        readability = 1.0 - min(
            abs(avg_word_length - 5) / 5, 1.0
        )  # Optimal around 5 chars
        features["readability"] = readability

        # Urgency indicators
        urgency_words = {
            "now",
            "today",
            "hurry",
            "quick",
            "fast",
            "urgent",
            "limited",
            "soon",
        }
        urgency_count = len(text_words.intersection(urgency_words))
        features["urgency_indicators"] = min(urgency_count / 2, 1.0)

        # Personal pronouns
        personal_pronouns = {"i", "we", "you", "me", "us", "my", "our", "your"}
        pronoun_count = len(text_words.intersection(personal_pronouns))
        features["personal_pronouns"] = (
            min(pronoun_count / word_count, 0.3) / 0.3
        )  # Normalize to 1.0

        # Optimal length (for Twitter-like platforms)
        if 50 <= char_count <= 200:
            features["optimal_length"] = 1.0
        elif 20 <= char_count < 50 or 200 < char_count <= 280:
            features["optimal_length"] = 0.8
        elif char_count < 20 or char_count > 280:
            features["optimal_length"] = 0.3
        else:
            features["optimal_length"] = 0.5

        return features

    def predict_engagement(self, text: str) -> float:
        """Predict engagement score for given text.

        Args:
            text: Input text to predict engagement for

        Returns:
            Engagement score between 0.0 and 1.0
        """
        features = self.extract_engagement_features(text)

        # Calculate weighted score
        engagement_score = sum(
            features.get(feature, 0.0) * weight
            for feature, weight in self.feature_weights.items()
        )

        return min(max(engagement_score, 0.0), 1.0)


class QualityAssessor:
    """Assesses content quality across multiple dimensions."""

    def assess_quality(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Assess quality across multiple dimensions.

        Args:
            text: Input text to assess
            context: Optional context for assessment

        Returns:
            Dictionary of quality scores
        """
        scores = {}

        # Fluency assessment
        scores["fluency"] = self._assess_fluency(text)

        # Coherence assessment
        scores["coherence"] = self._assess_coherence(text)

        # Relevance assessment (if context provided)
        if context:
            scores["relevance"] = self._assess_relevance(text, context)
        else:
            scores["relevance"] = 0.7  # Default moderate relevance

        # Brand alignment assessment
        scores["brand_alignment"] = self._assess_brand_alignment(text, context)

        # Viral potential assessment
        scores["viral_potential"] = self._assess_viral_potential(text)

        return scores

    def _assess_fluency(self, text: str) -> float:
        """Assess text fluency."""
        # Simple fluency metrics
        words = text.split()

        if len(words) < 3:
            return 0.5

        # Check for repeated words
        unique_words = set(words)
        repetition_penalty = len(unique_words) / len(words)

        # Check for basic grammar patterns
        grammar_score = 1.0

        # Penalize excessive punctuation
        punct_count = sum(1 for char in text if char in "!?.,;:")
        punct_ratio = punct_count / len(text)
        if punct_ratio > 0.1:
            grammar_score *= 0.8

        # Penalize ALL CAPS
        upper_ratio = sum(1 for char in text if char.isupper()) / len(text)
        if upper_ratio > 0.3:
            grammar_score *= 0.7

        fluency = (repetition_penalty + grammar_score) / 2
        return min(max(fluency, 0.0), 1.0)

    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return 0.8  # Single sentence is generally coherent

        # Simple coherence metrics
        coherence_score = 1.0

        # Check sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            length_std = np.std(lengths)
            # Moderate variation is good
            if length_std < 2 or length_std > 10:
                coherence_score *= 0.9

        # Check for topic consistency (simple keyword overlap)
        all_words = set()
        sentence_words = []
        for sentence in sentences:
            words = set(sentence.lower().split())
            sentence_words.append(words)
            all_words.update(words)

        # Calculate overlap between sentences
        if len(sentence_words) > 1:
            overlaps = []
            for i in range(len(sentence_words) - 1):
                overlap = len(sentence_words[i].intersection(sentence_words[i + 1]))
                overlaps.append(overlap / max(len(sentence_words[i]), 1))

            if overlaps:
                avg_overlap = np.mean(overlaps)
                # Some overlap is good for coherence
                if avg_overlap < 0.1 or avg_overlap > 0.8:
                    coherence_score *= 0.8

        return min(max(coherence_score, 0.0), 1.0)

    def _assess_relevance(self, text: str, context: Dict[str, Any]) -> float:
        """Assess relevance to provided context."""
        if not context:
            return 0.7

        # Extract context keywords
        context_keywords = set()
        if "topic" in context:
            context_keywords.update(context["topic"].lower().split())
        if "target_audience" in context:
            context_keywords.update(context["target_audience"].lower().split())
        if "keywords" in context and isinstance(context["keywords"], list):
            for keyword in context["keywords"]:
                context_keywords.update(keyword.lower().split())

        if not context_keywords:
            return 0.7

        # Calculate keyword overlap
        text_words = set(text.lower().split())
        overlap = len(text_words.intersection(context_keywords))
        relevance = min(overlap / len(context_keywords), 1.0)

        # Boost if at least some relevance found
        if relevance > 0:
            relevance = max(relevance, 0.5)

        return relevance

    def _assess_brand_alignment(
        self, text: str, context: Optional[Dict[str, Any]]
    ) -> float:
        """Assess brand alignment."""
        # Default brand alignment based on professionalism
        professionalism_score = 1.0

        # Check for inappropriate content (simple detection)
        inappropriate_words = {"spam", "scam", "fake", "clickbait", "hate"}
        text_words = set(text.lower().split())

        if text_words.intersection(inappropriate_words):
            professionalism_score *= 0.3

        # Check tone based on context
        if context and "brand_voice" in context:
            brand_voice = context["brand_voice"].lower()

            if "professional" in brand_voice:
                # Penalize excessive emojis or informal language
                emoji_count = sum(1 for char in text if ord(char) > 127)
                if emoji_count > 2:
                    professionalism_score *= 0.8

                informal_words = {"lol", "omg", "wtf", "awesome", "cool"}
                if text_words.intersection(informal_words):
                    professionalism_score *= 0.9

            elif "casual" in brand_voice:
                # Reward some informality
                informal_indicators = emoji_count + len(
                    text_words.intersection({"awesome", "cool", "great"})
                )
                if informal_indicators > 0:
                    professionalism_score = min(professionalism_score * 1.1, 1.0)

        return min(max(professionalism_score, 0.0), 1.0)

    def _assess_viral_potential(self, text: str) -> float:
        """Assess viral potential of content."""
        viral_score = 0.0

        # Emotional triggers
        emotional_words = {
            "amazing",
            "shocking",
            "unbelievable",
            "incredible",
            "insane",
            "mind-blowing",
            "secret",
            "revealed",
            "exposed",
            "breakthrough",
        }
        text_words = set(text.lower().split())
        emotional_count = len(text_words.intersection(emotional_words))
        viral_score += min(emotional_count * 0.2, 0.4)

        # Social sharing indicators
        sharing_words = {"share", "tag", "retweet", "tell", "spread", "viral"}
        sharing_count = len(text_words.intersection(sharing_words))
        viral_score += min(sharing_count * 0.15, 0.3)

        # Controversy/Opinion indicators
        opinion_words = {
            "think",
            "believe",
            "opinion",
            "agree",
            "disagree",
            "controversial",
        }
        opinion_count = len(text_words.intersection(opinion_words))
        viral_score += min(opinion_count * 0.1, 0.2)

        # Engagement hooks
        hooks = ["?", "!", "#"]
        hook_count = sum(text.count(hook) for hook in hooks)
        viral_score += min(hook_count * 0.05, 0.1)

        return min(viral_score, 1.0)


class REERCandidateScorer:
    """Main candidate scorer with perplexity calculation and quality assessment.

    Provides comprehensive scoring for social media content candidates using
    multiple evaluation dimensions including perplexity, engagement prediction,
    and quality assessment.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize candidate scorer.

        Args:
            config: Optional scoring configuration
        """
        self.config = config or ScoringConfig()
        self.perplexity_calculator = (
            PerplexityCalculator(self.config.model_name)
            if self.config.use_perplexity
            else None
        )
        self.engagement_predictor = (
            EngagementPredictor() if self.config.use_engagement_prediction else None
        )
        self.quality_assessor = (
            QualityAssessor() if self.config.use_quality_assessment else None
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all scoring components."""
        if self.perplexity_calculator and not self._initialized:
            await self.perplexity_calculator.initialize()
        self._initialized = True

    async def score_candidate(
        self, candidate: ContentCandidate, normalize_scores: bool = True
    ) -> ScoringMetrics:
        """Score a single content candidate.

        Args:
            candidate: Content candidate to score
            normalize_scores: Whether to normalize scores to 0-1 range

        Returns:
            ScoringMetrics with all calculated scores
        """
        if not self._initialized:
            await self.initialize()

        try:
            scores = {}

            # Calculate perplexity
            if self.perplexity_calculator:
                perplexity = await self.perplexity_calculator.calculate_perplexity(
                    candidate.text
                )
                # Normalize perplexity (lower is better, so invert)
                scores["perplexity"] = (
                    self._normalize_perplexity(perplexity)
                    if normalize_scores
                    else perplexity
                )
            else:
                scores["perplexity"] = 0.7  # Default moderate score

            # Predict engagement
            if self.engagement_predictor:
                scores["engagement"] = self.engagement_predictor.predict_engagement(
                    candidate.text
                )
            else:
                scores["engagement"] = 0.7

            # Assess quality
            quality_scores = {}
            if self.quality_assessor:
                quality_scores = self.quality_assessor.assess_quality(
                    candidate.text, candidate.context
                )
            else:
                quality_scores = {
                    "fluency": 0.7,
                    "coherence": 0.7,
                    "relevance": 0.7,
                    "brand_alignment": 0.7,
                    "viral_potential": 0.7,
                }

            # Extract individual quality scores
            scores.update(quality_scores)

            # Calculate overall quality score
            quality_weights = {
                "fluency": 0.3,
                "coherence": 0.3,
                "relevance": 0.2,
                "brand_alignment": 0.2,
            }

            quality_score = sum(
                quality_scores.get(dimension, 0.7) * weight
                for dimension, weight in quality_weights.items()
            )
            scores["quality"] = quality_score

            # Calculate overall score using configured weights
            overall_score = (
                scores["perplexity"] * self.config.weight_perplexity
                + scores["engagement"] * self.config.weight_engagement
                + scores["quality"] * self.config.weight_quality
            )

            # Calculate confidence based on score consistency
            score_values = [
                scores["perplexity"],
                scores["engagement"],
                scores["quality"],
            ]
            confidence = 1.0 - np.std(
                score_values
            )  # Higher consistency = higher confidence
            confidence = max(min(confidence, 1.0), 0.0)

            return ScoringMetrics(
                perplexity=scores["perplexity"],
                fluency_score=quality_scores.get("fluency", 0.7),
                engagement_score=scores["engagement"],
                quality_score=scores["quality"],
                coherence_score=quality_scores.get("coherence", 0.7),
                relevance_score=quality_scores.get("relevance", 0.7),
                viral_potential=quality_scores.get("viral_potential", 0.7),
                brand_alignment=quality_scores.get("brand_alignment", 0.7),
                overall_score=overall_score,
                confidence=confidence,
                metadata={
                    "candidate_id": candidate.candidate_id,
                    "text_length": len(candidate.text),
                    "scoring_model": self.config.model_name,
                    "weights": {
                        "perplexity": self.config.weight_perplexity,
                        "engagement": self.config.weight_engagement,
                        "quality": self.config.weight_quality,
                    },
                },
            )

        except Exception as e:
            raise ScoringError(
                f"Failed to score candidate {candidate.candidate_id}: {str(e)}",
                details={
                    "candidate_id": candidate.candidate_id,
                    "text_length": len(candidate.text),
                },
                original_error=e,
            )

    async def score_candidates(
        self, candidates: List[ContentCandidate], sort_by_score: bool = True
    ) -> List[Tuple[ContentCandidate, ScoringMetrics]]:
        """Score multiple content candidates.

        Args:
            candidates: List of candidates to score
            sort_by_score: Whether to sort results by overall score (descending)

        Returns:
            List of (candidate, metrics) tuples
        """
        if not candidates:
            return []

        results = []

        # Score candidates individually
        for candidate in candidates:
            try:
                metrics = await self.score_candidate(candidate)
                results.append((candidate, metrics))
            except ScoringError as e:
                # Log error but continue with other candidates
                print(
                    f"Warning: Failed to score candidate {candidate.candidate_id}: {e}"
                )
                continue

        # Sort by overall score if requested
        if sort_by_score:
            results.sort(key=lambda x: x[1].overall_score, reverse=True)

        return results

    def _normalize_perplexity(self, perplexity: float) -> float:
        """Normalize perplexity to 0-1 scale (lower perplexity = higher score)."""
        # Typical perplexity values range from 1 (perfect) to 1000+ (very poor)
        # We'll use a logarithmic normalization
        if perplexity <= 1.0:
            return 1.0
        elif perplexity >= 1000.0:
            return 0.0
        else:
            # Log scale normalization
            log_perplexity = math.log(perplexity)
            log_max = math.log(1000.0)
            normalized = 1.0 - (log_perplexity / log_max)
            return max(min(normalized, 1.0), 0.0)

    async def compare_candidates(
        self, candidates: List[ContentCandidate], comparison_criteria: List[str] = None
    ) -> Dict[str, Any]:
        """Compare candidates across multiple criteria.

        Args:
            candidates: List of candidates to compare
            comparison_criteria: List of criteria to compare (default: all)

        Returns:
            Comparison analysis with rankings and insights
        """
        if len(candidates) < 2:
            raise ValidationError("Need at least 2 candidates for comparison")

        if comparison_criteria is None:
            comparison_criteria = [
                "overall_score",
                "engagement_score",
                "quality_score",
                "fluency_score",
                "viral_potential",
                "brand_alignment",
            ]

        # Score all candidates
        scored_candidates = await self.score_candidates(candidates, sort_by_score=False)

        if not scored_candidates:
            raise ScoringError("No candidates could be scored successfully")

        # Build comparison analysis
        comparison = {
            "total_candidates": len(scored_candidates),
            "criteria_analyzed": comparison_criteria,
            "rankings": {},
            "statistics": {},
            "recommendations": [],
        }

        # Rank candidates by each criterion
        for criterion in comparison_criteria:
            ranked = sorted(
                scored_candidates,
                key=lambda x: getattr(x[1], criterion, 0.0),
                reverse=True,
            )

            comparison["rankings"][criterion] = [
                {
                    "rank": i + 1,
                    "candidate_id": candidate.candidate_id,
                    "score": getattr(metrics, criterion, 0.0),
                    "text_preview": (
                        candidate.text[:100] + "..."
                        if len(candidate.text) > 100
                        else candidate.text
                    ),
                }
                for i, (candidate, metrics) in enumerate(ranked)
            ]

        # Calculate statistics
        for criterion in comparison_criteria:
            scores = [
                getattr(metrics, criterion, 0.0) for _, metrics in scored_candidates
            ]
            comparison["statistics"][criterion] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "range": float(np.max(scores) - np.min(scores)),
            }

        # Generate recommendations
        best_overall = max(scored_candidates, key=lambda x: x[1].overall_score)
        comparison["recommendations"].append(
            {
                "type": "best_overall",
                "candidate_id": best_overall[0].candidate_id,
                "reason": "Highest overall score",
                "score": best_overall[1].overall_score,
            }
        )

        # Find candidate with best engagement potential
        best_engagement = max(scored_candidates, key=lambda x: x[1].engagement_score)
        if best_engagement[0].candidate_id != best_overall[0].candidate_id:
            comparison["recommendations"].append(
                {
                    "type": "best_engagement",
                    "candidate_id": best_engagement[0].candidate_id,
                    "reason": "Highest engagement potential",
                    "score": best_engagement[1].engagement_score,
                }
            )

        return comparison
