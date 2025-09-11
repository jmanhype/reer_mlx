"""T020: Scoring heuristics module implementation.

Provides comprehensive heuristics for evaluating content quality, engagement
potential, brand alignment, and other metrics for social media content.
"""

import re
import math
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import Counter

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from core.exceptions import ScoringError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class HeuristicWeights:
    """Weights for different heuristic components."""

    length_score: float = 0.15
    readability_score: float = 0.20
    engagement_score: float = 0.25
    sentiment_score: float = 0.15
    hashtag_score: float = 0.10
    mention_score: float = 0.05
    emoji_score: float = 0.10


@dataclass
class ContentMetrics:
    """Container for various content metrics."""

    # Basic metrics
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int

    # Engagement metrics
    hashtag_count: int
    mention_count: int
    emoji_count: int
    url_count: int

    # Quality metrics
    readability_score: float
    sentiment_score: float
    engagement_potential: float

    # Advanced metrics
    keyword_density: Dict[str, float] = field(default_factory=dict)
    pos_tag_distribution: Dict[str, int] = field(default_factory=dict)
    named_entities: List[str] = field(default_factory=list)


class ReadabilityCalculator:
    """Calculate various readability metrics."""

    @staticmethod
    def flesch_reading_ease(text: str) -> float:
        """Calculate Flesch Reading Ease score.

        Args:
            text: Input text

        Returns:
            Flesch Reading Ease score (0-100, higher is easier)
        """
        sentences = ReadabilityCalculator._count_sentences(text)
        words = ReadabilityCalculator._count_words(text)
        syllables = ReadabilityCalculator._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

        return max(0.0, min(100.0, score))

    @staticmethod
    def flesch_kincaid_grade(text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level.

        Args:
            text: Input text

        Returns:
            Grade level (lower is easier)
        """
        sentences = ReadabilityCalculator._count_sentences(text)
        words = ReadabilityCalculator._count_words(text)
        syllables = ReadabilityCalculator._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

        return max(0.0, grade)

    @staticmethod
    def automated_readability_index(text: str) -> float:
        """Calculate Automated Readability Index (ARI).

        Args:
            text: Input text

        Returns:
            ARI score
        """
        sentences = ReadabilityCalculator._count_sentences(text)
        words = ReadabilityCalculator._count_words(text)
        chars = len(re.sub(r"\s+", "", text))

        if sentences == 0 or words == 0:
            return 0.0

        ari = 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43

        return max(0.0, ari)

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in text."""
        # Simple sentence boundary detection
        sentences = re.split(r"[.!?]+", text.strip())
        return len([s for s in sentences if s.strip()])

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text."""
        words = re.findall(r"\b\w+\b", text.lower())
        return len(words)

    @staticmethod
    def _count_syllables(text: str) -> int:
        """Count syllables in text (approximation)."""
        words = re.findall(r"\b\w+\b", text.lower())
        syllable_count = 0

        for word in words:
            syllables = ReadabilityCalculator._syllables_in_word(word)
            syllable_count += syllables

        return max(1, syllable_count)

    @staticmethod
    def _syllables_in_word(word: str) -> int:
        """Estimate syllables in a single word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and count > 1:
            count -= 1

        return max(1, count)


class SentimentAnalyzer:
    """Simple sentiment analysis using lexicon-based approach."""

    # Basic sentiment lexicon (simplified)
    POSITIVE_WORDS = {
        "amazing",
        "awesome",
        "brilliant",
        "excellent",
        "fantastic",
        "great",
        "incredible",
        "love",
        "perfect",
        "wonderful",
        "best",
        "good",
        "happy",
        "excited",
        "thrilled",
        "delighted",
        "pleased",
        "satisfied",
        "joy",
        "beautiful",
        "stunning",
        "impressive",
        "outstanding",
        "remarkable",
    }

    NEGATIVE_WORDS = {
        "awful",
        "terrible",
        "horrible",
        "bad",
        "worst",
        "hate",
        "disgusting",
        "disappointing",
        "frustrating",
        "annoying",
        "boring",
        "ugly",
        "sad",
        "angry",
        "furious",
        "upset",
        "worried",
        "concerned",
        "confused",
        "difficult",
        "hard",
        "challenging",
        "problem",
        "issue",
        "trouble",
    }

    INTENSIFIERS = {
        "very": 1.5,
        "really": 1.3,
        "extremely": 1.8,
        "incredibly": 1.7,
        "absolutely": 1.6,
        "totally": 1.4,
        "completely": 1.5,
        "quite": 1.2,
        "rather": 1.1,
        "somewhat": 0.8,
        "slightly": 0.7,
        "barely": 0.5,
    }

    @classmethod
    def analyze_sentiment(cls, text: str) -> float:
        """Analyze sentiment of text.

        Args:
            text: Input text

        Returns:
            Sentiment score (-1.0 to 1.0, positive is good)
        """
        words = re.findall(r"\b\w+\b", text.lower())

        if not words:
            return 0.0

        score = 0.0
        word_count = 0
        intensifier = 1.0

        for word in words:
            # Check for intensifiers
            if word in cls.INTENSIFIERS:
                intensifier = cls.INTENSIFIERS[word]
                continue

            # Score positive words
            if word in cls.POSITIVE_WORDS:
                score += 1.0 * intensifier
                word_count += 1

            # Score negative words
            elif word in cls.NEGATIVE_WORDS:
                score -= 1.0 * intensifier
                word_count += 1

            # Reset intensifier
            intensifier = 1.0

        if word_count == 0:
            return 0.0

        # Normalize by word count
        normalized_score = score / word_count

        # Bound between -1 and 1
        return max(-1.0, min(1.0, normalized_score))


class EngagementPredictor:
    """Predict engagement potential based on content features."""

    # Platform-specific optimal lengths (characters)
    OPTIMAL_LENGTHS = {
        "twitter": (120, 280),
        "instagram": (150, 300),
        "facebook": (100, 250),
        "linkedin": (200, 600),
        "general": (100, 300),
    }

    # Engagement boosting patterns
    ENGAGEMENT_PATTERNS = {
        r"\b(how to|tips?|guide|tutorial)\b": 1.2,  # Instructional content
        r"\b(free|save|discount|offer)\b": 1.1,  # Value propositions
        r"\?": 1.15,  # Questions
        r"!": 1.05,  # Exclamations
        r"\b(you|your)\b": 1.1,  # Direct address
        r"\b(new|latest|breaking)\b": 1.1,  # Novelty
        r"\b(amazing|incredible|awesome)\b": 1.05,  # Superlatives
    }

    @classmethod
    def predict_engagement(
        cls,
        text: str,
        platform: str = "general",
        metrics: Optional[ContentMetrics] = None,
    ) -> float:
        """Predict engagement potential.

        Args:
            text: Content text
            platform: Target platform
            metrics: Pre-calculated metrics

        Returns:
            Engagement score (0.0 to 1.0)
        """
        if metrics is None:
            metrics = ContentAnalyzer.analyze_content(text)

        score = 0.5  # Base score

        # Length optimization
        length_score = cls._calculate_length_score(text, platform)
        score += (length_score - 0.5) * 0.2

        # Hashtag optimization
        hashtag_score = cls._calculate_hashtag_score(metrics.hashtag_count)
        score += (hashtag_score - 0.5) * 0.15

        # Emoji boost
        emoji_score = cls._calculate_emoji_score(metrics.emoji_count, len(text))
        score += (emoji_score - 0.5) * 0.1

        # Pattern matching
        pattern_score = cls._calculate_pattern_score(text)
        score += (pattern_score - 0.5) * 0.15

        # Question boost
        if "?" in text:
            score += 0.1

        # Call-to-action detection
        cta_patterns = [
            r"\b(click|tap|swipe|follow|subscribe|share|comment|like)\b",
            r"\b(check out|learn more|read more|sign up)\b",
            r"\b(download|get|try|start)\b",
        ]

        for pattern in cta_patterns:
            if re.search(pattern, text.lower()):
                score += 0.05
                break

        # Readability bonus
        if metrics.readability_score > 60:  # Good readability
            score += 0.1

        # Sentiment bonus for positive content
        if metrics.sentiment_score > 0.2:
            score += 0.05

        return max(0.0, min(1.0, score))

    @classmethod
    def _calculate_length_score(cls, text: str, platform: str) -> float:
        """Calculate length optimization score."""
        length = len(text)
        optimal_min, optimal_max = cls.OPTIMAL_LENGTHS.get(
            platform, cls.OPTIMAL_LENGTHS["general"]
        )

        if optimal_min <= length <= optimal_max:
            return 1.0
        elif length < optimal_min:
            return length / optimal_min
        else:
            # Penalty for being too long
            penalty = (length - optimal_max) / optimal_max
            return max(0.0, 1.0 - penalty)

    @classmethod
    def _calculate_hashtag_score(cls, hashtag_count: int) -> float:
        """Calculate hashtag optimization score."""
        if hashtag_count == 0:
            return 0.3
        elif 1 <= hashtag_count <= 3:
            return 1.0
        elif 4 <= hashtag_count <= 5:
            return 0.8
        elif 6 <= hashtag_count <= 10:
            return 0.6
        else:
            return 0.2  # Too many hashtags

    @classmethod
    def _calculate_emoji_score(cls, emoji_count: int, text_length: int) -> float:
        """Calculate emoji usage score."""
        if text_length == 0:
            return 0.5

        emoji_ratio = emoji_count / max(1, text_length / 100)  # Per 100 chars

        if 0.5 <= emoji_ratio <= 2.0:
            return 1.0
        elif emoji_ratio < 0.5:
            return 0.5 + emoji_ratio
        else:
            return max(0.2, 1.0 - (emoji_ratio - 2.0) / 5.0)

    @classmethod
    def _calculate_pattern_score(cls, text: str) -> float:
        """Calculate score based on engagement patterns."""
        score = 0.5

        for pattern, boost in cls.ENGAGEMENT_PATTERNS.items():
            if re.search(pattern, text.lower()):
                score *= boost

        return min(1.0, score)


class ContentAnalyzer:
    """Main content analysis class."""

    @staticmethod
    def analyze_content(text: str) -> ContentMetrics:
        """Perform comprehensive content analysis.

        Args:
            text: Content text to analyze

        Returns:
            ContentMetrics object with all calculated metrics
        """
        # Basic metrics
        word_count = len(re.findall(r"\b\w+\b", text))
        character_count = len(text)
        sentence_count = max(1, ReadabilityCalculator._count_sentences(text))
        paragraph_count = len([p for p in text.split("\n\n") if p.strip()])

        # Engagement metrics
        hashtag_count = len(re.findall(r"#\w+", text))
        mention_count = len(re.findall(r"@\w+", text))
        emoji_count = len(
            re.findall(
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f018-\U0001f270]",
                text,
            )
        )
        url_count = len(re.findall(r"https?://\S+", text))

        # Quality metrics
        readability_score = ReadabilityCalculator.flesch_reading_ease(text)
        sentiment_score = SentimentAnalyzer.analyze_sentiment(text)

        # Create metrics object first
        metrics = ContentMetrics(
            word_count=word_count,
            character_count=character_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            hashtag_count=hashtag_count,
            mention_count=mention_count,
            emoji_count=emoji_count,
            url_count=url_count,
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            engagement_potential=0.0,  # Will be calculated below
        )

        # Calculate engagement potential using the metrics
        metrics.engagement_potential = EngagementPredictor.predict_engagement(
            text, metrics=metrics
        )

        # Advanced metrics
        metrics.keyword_density = ContentAnalyzer._calculate_keyword_density(text)

        return metrics

    @staticmethod
    def _calculate_keyword_density(text: str, top_n: int = 10) -> Dict[str, float]:
        """Calculate keyword density for top words.

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            Dictionary of keyword densities
        """
        # Common stop words to exclude
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }

        words = re.findall(r"\b\w+\b", text.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

        if not filtered_words:
            return {}

        word_counts = Counter(filtered_words)
        total_words = len(filtered_words)

        # Calculate density for top words
        density = {}
        for word, count in word_counts.most_common(top_n):
            density[word] = count / total_words

        return density


class HeuristicScorer:
    """Main scoring class using various heuristics."""

    def __init__(self, weights: Optional[HeuristicWeights] = None):
        """Initialize scorer with custom weights.

        Args:
            weights: Custom weights for scoring components
        """
        self.weights = weights or HeuristicWeights()

    def score_content(
        self,
        text: str,
        platform: str = "general",
        target_audience: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Score content using multiple heuristics.

        Args:
            text: Content text
            platform: Target platform
            target_audience: Target audience description

        Returns:
            Tuple of (overall_score, component_scores)
        """
        try:
            # Analyze content
            metrics = ContentAnalyzer.analyze_content(text)

            # Calculate component scores
            scores = {}

            # Length score
            scores["length"] = self._score_length(text, platform)

            # Readability score
            scores["readability"] = self._score_readability(metrics.readability_score)

            # Engagement score
            scores["engagement"] = metrics.engagement_potential

            # Sentiment score
            scores["sentiment"] = self._score_sentiment(metrics.sentiment_score)

            # Hashtag score
            scores["hashtag"] = self._score_hashtags(metrics.hashtag_count, platform)

            # Mention score
            scores["mention"] = self._score_mentions(metrics.mention_count)

            # Emoji score
            scores["emoji"] = self._score_emojis(metrics.emoji_count, len(text))

            # Calculate weighted overall score
            overall_score = (
                scores["length"] * self.weights.length_score
                + scores["readability"] * self.weights.readability_score
                + scores["engagement"] * self.weights.engagement_score
                + scores["sentiment"] * self.weights.sentiment_score
                + scores["hashtag"] * self.weights.hashtag_score
                + scores["mention"] * self.weights.mention_score
                + scores["emoji"] * self.weights.emoji_score
            )

            return overall_score, scores

        except Exception as e:
            logger.error(f"Content scoring failed: {e}")
            raise ScoringError(f"Scoring failed: {e}")

    def _score_length(self, text: str, platform: str) -> float:
        """Score content length for platform."""
        return EngagementPredictor._calculate_length_score(text, platform)

    def _score_readability(self, readability_score: float) -> float:
        """Score readability (0-100 scale to 0-1)."""
        return min(1.0, readability_score / 100.0)

    def _score_sentiment(self, sentiment_score: float) -> float:
        """Score sentiment (-1 to 1 scale to 0-1)."""
        # Convert to 0-1 scale, with neutral being 0.5
        return (sentiment_score + 1.0) / 2.0

    def _score_hashtags(self, hashtag_count: int, platform: str) -> float:
        """Score hashtag usage."""
        return EngagementPredictor._calculate_hashtag_score(hashtag_count)

    def _score_mentions(self, mention_count: int) -> float:
        """Score mention usage."""
        if mention_count == 0:
            return 0.5
        elif mention_count <= 2:
            return 1.0
        elif mention_count <= 5:
            return 0.8
        else:
            return 0.3  # Too many mentions

    def _score_emojis(self, emoji_count: int, text_length: int) -> float:
        """Score emoji usage."""
        return EngagementPredictor._calculate_emoji_score(emoji_count, text_length)


# Platform-specific scoring profiles
PLATFORM_WEIGHTS = {
    "twitter": HeuristicWeights(
        length_score=0.20,
        readability_score=0.15,
        engagement_score=0.30,
        sentiment_score=0.10,
        hashtag_score=0.15,
        mention_score=0.05,
        emoji_score=0.05,
    ),
    "instagram": HeuristicWeights(
        length_score=0.10,
        readability_score=0.15,
        engagement_score=0.25,
        sentiment_score=0.15,
        hashtag_score=0.20,
        mention_score=0.05,
        emoji_score=0.10,
    ),
    "linkedin": HeuristicWeights(
        length_score=0.15,
        readability_score=0.30,
        engagement_score=0.20,
        sentiment_score=0.15,
        hashtag_score=0.05,
        mention_score=0.10,
        emoji_score=0.05,
    ),
    "facebook": HeuristicWeights(
        length_score=0.15,
        readability_score=0.20,
        engagement_score=0.25,
        sentiment_score=0.15,
        hashtag_score=0.10,
        mention_score=0.05,
        emoji_score=0.10,
    ),
}


def create_platform_scorer(platform: str) -> HeuristicScorer:
    """Create a scorer optimized for a specific platform.

    Args:
        platform: Platform name (twitter, instagram, linkedin, facebook)

    Returns:
        Configured HeuristicScorer
    """
    weights = PLATFORM_WEIGHTS.get(platform.lower(), HeuristicWeights())
    return HeuristicScorer(weights)


async def score_content_async(
    text: str, platform: str = "general", scorer: Optional[HeuristicScorer] = None
) -> Tuple[float, Dict[str, float]]:
    """Async wrapper for content scoring.

    Args:
        text: Content text
        platform: Target platform
        scorer: Optional custom scorer

    Returns:
        Tuple of (overall_score, component_scores)
    """
    if scorer is None:
        scorer = create_platform_scorer(platform)

    # Run scoring in thread pool for CPU-intensive operations
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, scorer.score_content, text, platform)
