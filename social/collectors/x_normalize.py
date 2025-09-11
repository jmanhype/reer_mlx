"""T025: X (Twitter) Analytics Data Normalizer

Normalizes X (Twitter) analytics data into a standardized format
for consistent processing across the REER × DSPy × MLX pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class XMetricType(Enum):
    """Types of X (Twitter) metrics."""

    ENGAGEMENT = "engagement"
    REACH = "reach"
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    RETWEETS = "retweets"
    LIKES = "likes"
    REPLIES = "replies"
    QUOTES = "quotes"
    FOLLOWS = "follows"
    PROFILE_VISITS = "profile_visits"
    VIDEO_VIEWS = "video_views"


@dataclass
class NormalizedMetric:
    """Standardized social media metric."""

    metric_type: XMetricType
    value: int | float
    timestamp: datetime
    post_id: str | None = None
    account_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedPost:
    """Standardized social media post data."""

    post_id: str
    content: str
    author_id: str
    timestamp: datetime
    platform: str = "x"
    metrics: list[NormalizedMetric] = field(default_factory=list)
    engagement_rate: float | None = None
    reach: int | None = None
    impressions: int | None = None
    url: str | None = None
    media_urls: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class XAnalyticsNormalizer:
    """Normalizes X (Twitter) analytics data into standardized format."""

    def __init__(self):
        """Initialize the normalizer."""
        self.logger = logging.getLogger(__name__)

    def normalize_tweet_data(self, raw_data: dict[str, Any]) -> NormalizedPost:
        """
        Normalize raw tweet data from X API v2.

        Args:
            raw_data: Raw tweet data from X API

        Returns:
            NormalizedPost with standardized data
        """
        try:
            # Extract basic tweet information
            tweet_data = raw_data.get("data", raw_data)

            post_id = tweet_data.get("id", "")
            content = tweet_data.get("text", "")
            author_id = tweet_data.get("author_id", "")

            # Parse timestamp
            created_at = tweet_data.get("created_at")
            if isinstance(created_at, str):
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)

            # Extract metrics
            public_metrics = tweet_data.get("public_metrics", {})
            metrics = self._extract_metrics(
                public_metrics, post_id, author_id, timestamp
            )

            # Calculate engagement rate if we have the data
            engagement_rate = self._calculate_engagement_rate(public_metrics)

            # Extract entities
            entities = tweet_data.get("entities", {})
            hashtags = [tag["tag"] for tag in entities.get("hashtags", [])]
            mentions = [mention["username"] for mention in entities.get("mentions", [])]

            # Extract media URLs
            media_urls = []
            if "attachments" in tweet_data:
                media_keys = tweet_data["attachments"].get("media_keys", [])
                # Would need to resolve media_keys to URLs via includes data
                media_urls = media_keys  # Placeholder

            # Build post URL
            url = (
                f"https://twitter.com/{author_id}/status/{post_id}"
                if post_id and author_id
                else None
            )

            normalized_post = NormalizedPost(
                post_id=post_id,
                content=content,
                author_id=author_id,
                timestamp=timestamp,
                platform="x",
                metrics=metrics,
                engagement_rate=engagement_rate,
                reach=public_metrics.get("impression_count"),
                impressions=public_metrics.get("impression_count"),
                url=url,
                media_urls=media_urls,
                hashtags=hashtags,
                mentions=mentions,
                metadata={
                    "raw_data": raw_data,
                    "conversation_id": tweet_data.get("conversation_id"),
                    "in_reply_to_user_id": tweet_data.get("in_reply_to_user_id"),
                    "referenced_tweets": tweet_data.get("referenced_tweets", []),
                    "lang": tweet_data.get("lang"),
                    "possibly_sensitive": tweet_data.get("possibly_sensitive", False),
                },
            )

            self.logger.debug(f"Normalized tweet {post_id} with {len(metrics)} metrics")
            return normalized_post

        except Exception as e:
            self.logger.exception(f"Failed to normalize tweet data: {e}")
            raise ValueError(f"Invalid tweet data format: {e}")

    def normalize_analytics_batch(
        self, analytics_data: list[dict[str, Any]]
    ) -> list[NormalizedPost]:
        """
        Normalize a batch of analytics data.

        Args:
            analytics_data: List of raw analytics data from X API

        Returns:
            List of normalized posts
        """
        normalized_posts = []

        for data in analytics_data:
            try:
                normalized_post = self.normalize_tweet_data(data)
                normalized_posts.append(normalized_post)
            except Exception as e:
                self.logger.warning(f"Skipping invalid analytics data: {e}")
                continue

        self.logger.info(
            f"Normalized {len(normalized_posts)} posts from {len(analytics_data)} raw entries"
        )
        return normalized_posts

    def _extract_metrics(
        self,
        public_metrics: dict[str, Any],
        post_id: str,
        author_id: str,
        timestamp: datetime,
    ) -> list[NormalizedMetric]:
        """Extract and normalize individual metrics."""
        metrics = []

        metric_mapping = {
            "retweet_count": XMetricType.RETWEETS,
            "like_count": XMetricType.LIKES,
            "reply_count": XMetricType.REPLIES,
            "quote_count": XMetricType.QUOTES,
            "impression_count": XMetricType.IMPRESSIONS,
        }

        for api_field, metric_type in metric_mapping.items():
            if api_field in public_metrics:
                value = public_metrics[api_field]
                if isinstance(value, int | float) and value >= 0:
                    metric = NormalizedMetric(
                        metric_type=metric_type,
                        value=value,
                        timestamp=timestamp,
                        post_id=post_id,
                        account_id=author_id,
                    )
                    metrics.append(metric)

        return metrics

    def _calculate_engagement_rate(
        self, public_metrics: dict[str, Any]
    ) -> float | None:
        """Calculate engagement rate from public metrics."""
        try:
            impressions = public_metrics.get("impression_count", 0)
            if impressions <= 0:
                return None

            engagements = (
                public_metrics.get("retweet_count", 0)
                + public_metrics.get("like_count", 0)
                + public_metrics.get("reply_count", 0)
                + public_metrics.get("quote_count", 0)
            )

            return (engagements / impressions) * 100 if impressions > 0 else 0.0

        except (TypeError, ZeroDivisionError):
            return None

    def normalize_user_metrics(
        self, user_data: dict[str, Any]
    ) -> list[NormalizedMetric]:
        """
        Normalize user-level metrics from X API.

        Args:
            user_data: Raw user data from X API

        Returns:
            List of normalized user metrics
        """
        metrics = []
        timestamp = datetime.now(timezone.utc)

        public_metrics = user_data.get("public_metrics", {})
        user_id = user_data.get("id", "")

        user_metric_mapping = {
            "followers_count": XMetricType.FOLLOWS,
            "following_count": "following",  # Custom handling
            "tweet_count": "tweets",  # Custom handling
            "listed_count": "lists",  # Custom handling
        }

        for api_field, metric_type in user_metric_mapping.items():
            if api_field in public_metrics:
                value = public_metrics[api_field]
                if isinstance(value, int | float) and value >= 0:
                    # Handle standard metrics
                    if isinstance(metric_type, XMetricType):
                        metric = NormalizedMetric(
                            metric_type=metric_type,
                            value=value,
                            timestamp=timestamp,
                            account_id=user_id,
                            metadata={"metric_source": "user_profile"},
                        )
                        metrics.append(metric)
                    else:
                        # Handle custom metrics in metadata
                        metric = NormalizedMetric(
                            metric_type=XMetricType.ENGAGEMENT,  # Generic type
                            value=value,
                            timestamp=timestamp,
                            account_id=user_id,
                            metadata={
                                "custom_metric_type": metric_type,
                                "metric_source": "user_profile",
                            },
                        )
                        metrics.append(metric)

        return metrics

    def extract_trending_data(self, trends_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and normalize trending topics data.

        Args:
            trends_data: Raw trends data from X API

        Returns:
            Normalized trending data
        """
        try:
            trends = trends_data.get("trends", [])
            normalized_trends = []

            for trend in trends:
                normalized_trend = {
                    "name": trend.get("name", ""),
                    "query": trend.get("query", ""),
                    "tweet_volume": trend.get("tweet_volume"),
                    "url": trend.get("url", ""),
                    "promoted_content": trend.get("promoted_content"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                normalized_trends.append(normalized_trend)

            return {
                "trends": normalized_trends,
                "location": trends_data.get("locations", [{}])[0],
                "as_of": trends_data.get("as_of", ""),
                "created_at": trends_data.get("created_at", ""),
                "normalized_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.exception(f"Failed to normalize trends data: {e}")
            return {"trends": [], "error": str(e)}
