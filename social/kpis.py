"""T027: Social Media KPI Metrics Calculator

Calculates key performance indicators for social media content including
engagement rates, reach metrics, virality scores, and platform-specific KPIs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class KPICategory(Enum):
    """Categories of social media KPIs."""

    ENGAGEMENT = "engagement"
    REACH = "reach"
    VIRALITY = "virality"
    CONVERSION = "conversion"
    GROWTH = "growth"
    CONTENT_QUALITY = "content_quality"


class Platform(Enum):
    """Supported social media platforms for KPI calculation."""

    X = "x"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"


@dataclass
class PostMetrics:
    """Raw metrics for a social media post."""

    post_id: str
    platform: Platform
    timestamp: datetime
    impressions: int = 0
    reach: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    clicks: int = 0
    video_views: int = 0
    video_completion_rate: float = 0.0
    followers_gained: int = 0
    profile_visits: int = 0
    link_clicks: int = 0
    hashtag_clicks: int = 0
    mentions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIResult:
    """Calculated KPI result."""

    name: str
    category: KPICategory
    value: float | int
    unit: str
    platform: Platform
    calculation_method: str
    benchmark: float | None = None
    performance_level: str | None = None  # poor, average, good, excellent
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIDashboard:
    """Complete KPI dashboard for social media performance."""

    account_id: str
    platform: Platform
    period_start: datetime
    period_end: datetime
    total_posts: int
    kpis: list[KPIResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class SocialKPICalculator:
    """Calculator for social media key performance indicators."""

    def __init__(self):
        """Initialize the KPI calculator."""
        self.logger = logging.getLogger(__name__)

        # Platform-specific benchmarks (industry averages)
        self.benchmarks = {
            Platform.X: {
                "engagement_rate": 0.045,  # 4.5%
                "click_through_rate": 0.023,  # 2.3%
                "retweet_rate": 0.015,  # 1.5%
            },
            Platform.LINKEDIN: {
                "engagement_rate": 0.054,  # 5.4%
                "click_through_rate": 0.026,  # 2.6%
                "share_rate": 0.012,  # 1.2%
            },
            Platform.FACEBOOK: {
                "engagement_rate": 0.064,  # 6.4%
                "click_through_rate": 0.9,  # 0.9%
                "share_rate": 0.27,  # 0.27%
            },
            Platform.INSTAGRAM: {
                "engagement_rate": 0.83,  # 0.83%
                "save_rate": 0.2,  # 0.2%
                "story_completion_rate": 0.7,  # 70%
            },
            Platform.TIKTOK: {
                "engagement_rate": 4.25,  # 4.25%
                "completion_rate": 0.16,  # 16%
                "share_rate": 0.046,  # 4.6%
            },
        }

    def calculate_engagement_rate(self, metrics: PostMetrics) -> KPIResult:
        """
        Calculate engagement rate for a post.

        Engagement Rate = (Likes + Comments + Shares) / Impressions * 100
        """
        try:
            total_engagements = metrics.likes + metrics.comments + metrics.shares

            # Add platform-specific engagements
            if metrics.platform == Platform.INSTAGRAM:
                total_engagements += metrics.saves
            elif metrics.platform == Platform.TIKTOK:
                total_engagements += metrics.video_views // 100  # Normalize video views

            if metrics.impressions > 0:
                rate = (total_engagements / metrics.impressions) * 100
            else:
                rate = 0.0

            benchmark = self.benchmarks.get(metrics.platform, {}).get(
                "engagement_rate", 0.0
            )
            performance = self._assess_performance(rate, benchmark)

            return KPIResult(
                name="Engagement Rate",
                category=KPICategory.ENGAGEMENT,
                value=round(rate, 3),
                unit="%",
                platform=metrics.platform,
                calculation_method="(Likes + Comments + Shares) / Impressions * 100",
                benchmark=benchmark,
                performance_level=performance,
                metadata={
                    "total_engagements": total_engagements,
                    "impressions": metrics.impressions,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate engagement rate: {e}")
            return self._create_error_kpi(
                "Engagement Rate", KPICategory.ENGAGEMENT, metrics.platform, str(e)
            )

    def calculate_reach_rate(
        self, metrics: PostMetrics, total_followers: int
    ) -> KPIResult:
        """
        Calculate reach rate as percentage of followers reached.

        Reach Rate = Reach / Total Followers * 100
        """
        try:
            if total_followers > 0 and metrics.reach > 0:
                rate = (metrics.reach / total_followers) * 100
            else:
                rate = 0.0

            return KPIResult(
                name="Reach Rate",
                category=KPICategory.REACH,
                value=round(rate, 2),
                unit="%",
                platform=metrics.platform,
                calculation_method="Reach / Total Followers * 100",
                metadata={
                    "reach": metrics.reach,
                    "total_followers": total_followers,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate reach rate: {e}")
            return self._create_error_kpi(
                "Reach Rate", KPICategory.REACH, metrics.platform, str(e)
            )

    def calculate_virality_score(
        self, metrics: PostMetrics, account_avg_shares: float | None = None
    ) -> KPIResult:
        """
        Calculate virality score based on sharing behavior.

        Virality Score = (Shares / Impressions) * 1000
        Higher score indicates more viral content.
        """
        try:
            if metrics.impressions > 0:
                virality_score = (metrics.shares / metrics.impressions) * 1000
            else:
                virality_score = 0.0

            # Assess virality level
            if virality_score >= 10:
                virality_level = "viral"
            elif virality_score >= 5:
                virality_level = "high"
            elif virality_score >= 2:
                virality_level = "moderate"
            elif virality_score >= 0.5:
                virality_level = "low"
            else:
                virality_level = "minimal"

            return KPIResult(
                name="Virality Score",
                category=KPICategory.VIRALITY,
                value=round(virality_score, 3),
                unit="per 1000 impressions",
                platform=metrics.platform,
                calculation_method="(Shares / Impressions) * 1000",
                performance_level=virality_level,
                metadata={
                    "shares": metrics.shares,
                    "impressions": metrics.impressions,
                    "virality_level": virality_level,
                    "account_avg_shares": account_avg_shares,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate virality score: {e}")
            return self._create_error_kpi(
                "Virality Score", KPICategory.VIRALITY, metrics.platform, str(e)
            )

    def calculate_click_through_rate(self, metrics: PostMetrics) -> KPIResult:
        """
        Calculate click-through rate for links in posts.

        CTR = Link Clicks / Impressions * 100
        """
        try:
            total_clicks = metrics.link_clicks + metrics.clicks

            if metrics.impressions > 0:
                ctr = (total_clicks / metrics.impressions) * 100
            else:
                ctr = 0.0

            benchmark = self.benchmarks.get(metrics.platform, {}).get(
                "click_through_rate", 0.0
            )
            performance = self._assess_performance(ctr, benchmark)

            return KPIResult(
                name="Click-Through Rate",
                category=KPICategory.CONVERSION,
                value=round(ctr, 3),
                unit="%",
                platform=metrics.platform,
                calculation_method="Link Clicks / Impressions * 100",
                benchmark=benchmark,
                performance_level=performance,
                metadata={
                    "total_clicks": total_clicks,
                    "link_clicks": metrics.link_clicks,
                    "general_clicks": metrics.clicks,
                    "impressions": metrics.impressions,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate CTR: {e}")
            return self._create_error_kpi(
                "Click-Through Rate", KPICategory.CONVERSION, metrics.platform, str(e)
            )

    def calculate_video_completion_rate(self, metrics: PostMetrics) -> KPIResult | None:
        """
        Calculate video completion rate for video content.

        Completion Rate = Video Completions / Video Views * 100
        """
        if metrics.video_views == 0:
            return None

        try:
            # Use provided completion rate or estimate from engagement
            if metrics.video_completion_rate > 0:
                completion_rate = metrics.video_completion_rate * 100
            else:
                # Estimate based on engagement (simplified)
                engagement_ratio = (
                    metrics.likes + metrics.comments
                ) / metrics.video_views
                completion_rate = min(engagement_ratio * 100, 100)

            benchmark = self.benchmarks.get(metrics.platform, {}).get(
                "completion_rate", 0.0
            )
            performance = self._assess_performance(completion_rate / 100, benchmark)

            return KPIResult(
                name="Video Completion Rate",
                category=KPICategory.CONTENT_QUALITY,
                value=round(completion_rate, 2),
                unit="%",
                platform=metrics.platform,
                calculation_method="Video Completions / Video Views * 100",
                benchmark=benchmark * 100 if benchmark else None,
                performance_level=performance,
                metadata={
                    "video_views": metrics.video_views,
                    "estimated": metrics.video_completion_rate == 0,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate video completion rate: {e}")
            return self._create_error_kpi(
                "Video Completion Rate",
                KPICategory.CONTENT_QUALITY,
                metrics.platform,
                str(e),
            )

    def calculate_save_rate(self, metrics: PostMetrics) -> KPIResult | None:
        """
        Calculate save rate (primarily for Instagram and LinkedIn).

        Save Rate = Saves / Impressions * 100
        """
        if metrics.saves == 0 or metrics.platform not in [
            Platform.INSTAGRAM,
            Platform.LINKEDIN,
        ]:
            return None

        try:
            if metrics.impressions > 0:
                save_rate = (metrics.saves / metrics.impressions) * 100
            else:
                save_rate = 0.0

            benchmark = self.benchmarks.get(metrics.platform, {}).get("save_rate", 0.0)
            performance = self._assess_performance(save_rate, benchmark)

            return KPIResult(
                name="Save Rate",
                category=KPICategory.CONTENT_QUALITY,
                value=round(save_rate, 3),
                unit="%",
                platform=metrics.platform,
                calculation_method="Saves / Impressions * 100",
                benchmark=benchmark,
                performance_level=performance,
                metadata={
                    "saves": metrics.saves,
                    "impressions": metrics.impressions,
                    "post_id": metrics.post_id,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate save rate: {e}")
            return self._create_error_kpi(
                "Save Rate", KPICategory.CONTENT_QUALITY, metrics.platform, str(e)
            )

    def calculate_growth_rate(
        self, current_followers: int, previous_followers: int, days: int
    ) -> KPIResult:
        """
        Calculate follower growth rate.

        Growth Rate = ((Current - Previous) / Previous) / Days * 100
        """
        try:
            if previous_followers > 0 and days > 0:
                growth_rate = (
                    ((current_followers - previous_followers) / previous_followers)
                    / days
                    * 100
                )
            else:
                growth_rate = 0.0

            return KPIResult(
                name="Daily Growth Rate",
                category=KPICategory.GROWTH,
                value=round(growth_rate, 4),
                unit="% per day",
                platform=Platform.X,  # Platform agnostic
                calculation_method="((Current - Previous) / Previous) / Days * 100",
                metadata={
                    "current_followers": current_followers,
                    "previous_followers": previous_followers,
                    "days": days,
                    "net_growth": current_followers - previous_followers,
                },
            )

        except Exception as e:
            self.logger.exception(f"Failed to calculate growth rate: {e}")
            return self._create_error_kpi(
                "Daily Growth Rate", KPICategory.GROWTH, Platform.X, str(e)
            )

    def calculate_all_kpis(
        self,
        metrics: PostMetrics,
        total_followers: int | None = None,
        account_avg_shares: float | None = None,
    ) -> list[KPIResult]:
        """
        Calculate all applicable KPIs for a post.

        Args:
            metrics: Post metrics data
            total_followers: Total follower count for reach calculations
            account_avg_shares: Account average shares for comparison

        Returns:
            List of calculated KPI results
        """
        kpis = []

        # Core KPIs (always calculated)
        kpis.append(self.calculate_engagement_rate(metrics))
        kpis.append(self.calculate_virality_score(metrics, account_avg_shares))
        kpis.append(self.calculate_click_through_rate(metrics))

        # Conditional KPIs
        if total_followers:
            kpis.append(self.calculate_reach_rate(metrics, total_followers))

        video_completion_kpi = self.calculate_video_completion_rate(metrics)
        if video_completion_kpi:
            kpis.append(video_completion_kpi)

        save_rate_kpi = self.calculate_save_rate(metrics)
        if save_rate_kpi:
            kpis.append(save_rate_kpi)

        # Filter out None values and sort by category
        kpis = [kpi for kpi in kpis if kpi is not None]
        kpis.sort(key=lambda x: x.category.value)

        return kpis

    def create_dashboard(
        self,
        account_id: str,
        platform: Platform,
        post_metrics: list[PostMetrics],
        period_start: datetime,
        period_end: datetime,
        total_followers: int | None = None,
    ) -> KPIDashboard:
        """
        Create a comprehensive KPI dashboard for an account.

        Args:
            account_id: Social media account identifier
            platform: Social media platform
            post_metrics: List of post metrics for the period
            period_start: Start of analysis period
            period_end: End of analysis period
            total_followers: Current total followers

        Returns:
            Complete KPI dashboard
        """
        all_kpis = []

        # Calculate KPIs for each post
        for post_metric in post_metrics:
            post_kpis = self.calculate_all_kpis(post_metric, total_followers)
            all_kpis.extend(post_kpis)

        # Calculate aggregate metrics
        summary = self._calculate_summary_metrics(post_metrics, all_kpis)
        recommendations = self._generate_recommendations(summary, platform)

        return KPIDashboard(
            account_id=account_id,
            platform=platform,
            period_start=period_start,
            period_end=period_end,
            total_posts=len(post_metrics),
            kpis=all_kpis,
            summary=summary,
            recommendations=recommendations,
        )

    def _assess_performance(self, value: float, benchmark: float | None) -> str:
        """Assess performance level against benchmark."""
        if benchmark is None:
            return "unknown"

        if value >= benchmark * 1.5:
            return "excellent"
        if value >= benchmark * 1.1:
            return "good"
        if value >= benchmark * 0.9:
            return "average"
        return "poor"

    def _calculate_summary_metrics(
        self, post_metrics: list[PostMetrics], all_kpis: list[KPIResult]
    ) -> dict[str, Any]:
        """Calculate summary metrics across all posts."""
        if not post_metrics:
            return {}

        total_impressions = sum(p.impressions for p in post_metrics)
        total_engagements = sum(p.likes + p.comments + p.shares for p in post_metrics)
        total_clicks = sum(p.clicks + p.link_clicks for p in post_metrics)

        # Average KPIs by category
        kpi_averages = {}
        for category in KPICategory:
            category_kpis = [kpi for kpi in all_kpis if kpi.category == category]
            if category_kpis:
                avg_value = sum(kpi.value for kpi in category_kpis) / len(category_kpis)
                kpi_averages[category.value] = round(avg_value, 3)

        return {
            "total_posts": len(post_metrics),
            "total_impressions": total_impressions,
            "total_engagements": total_engagements,
            "total_clicks": total_clicks,
            "avg_impressions_per_post": round(total_impressions / len(post_metrics), 0),
            "avg_engagements_per_post": round(total_engagements / len(post_metrics), 1),
            "overall_engagement_rate": (
                round((total_engagements / total_impressions * 100), 3)
                if total_impressions > 0
                else 0
            ),
            "overall_click_rate": (
                round((total_clicks / total_impressions * 100), 3)
                if total_impressions > 0
                else 0
            ),
            "kpi_averages": kpi_averages,
        }

    def _generate_recommendations(
        self, summary: dict[str, Any], platform: Platform
    ) -> list[str]:
        """Generate actionable recommendations based on KPI performance."""
        recommendations = []

        kpi_averages = summary.get("kpi_averages", {})
        benchmarks = self.benchmarks.get(platform, {})

        # Engagement recommendations
        avg_engagement = kpi_averages.get("engagement", 0)
        engagement_benchmark = benchmarks.get("engagement_rate", 0) * 100

        if avg_engagement < engagement_benchmark * 0.8:
            recommendations.append(
                "Consider improving content quality and posting at optimal times to boost engagement"
            )
            recommendations.append(
                "Experiment with more interactive content formats (polls, questions, videos)"
            )

        # Click-through rate recommendations
        avg_ctr = kpi_averages.get("conversion", 0)
        ctr_benchmark = benchmarks.get("click_through_rate", 0) * 100

        if avg_ctr < ctr_benchmark * 0.8:
            recommendations.append(
                "Optimize call-to-action wording and placement to improve click-through rates"
            )
            recommendations.append(
                "Test different link preview formats and compelling copy"
            )

        # Platform-specific recommendations
        if platform == Platform.TIKTOK:
            recommendations.append(
                "Focus on video completion rates by creating compelling hooks in the first 3 seconds"
            )
        elif platform == Platform.INSTAGRAM:
            recommendations.append(
                "Encourage saves by creating valuable, reference-worthy content"
            )
        elif platform == Platform.LINKEDIN:
            recommendations.append(
                "Share industry insights and professional tips to increase engagement"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _create_error_kpi(
        self, name: str, category: KPICategory, platform: Platform, error: str
    ) -> KPIResult:
        """Create an error KPI result."""
        return KPIResult(
            name=name,
            category=category,
            value=0,
            unit="error",
            platform=platform,
            calculation_method="error",
            metadata={"error": error},
        )
