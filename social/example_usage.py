"""Example usage of the social media modules for REER × DSPy × MLX.

Demonstrates the integration of X analytics normalization, DSPy content generation,
and KPI calculation for social media management.
"""

from datetime import UTC, datetime, timedelta
import logging

from .collectors.x_normalize import XAnalyticsNormalizer
from .dspy_modules import (
    ContentBrief,
    ContentType,
    Platform,
    SocialContentPipeline,
)
from .kpis import PostMetrics, SocialKPICalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_x_analytics_normalization():
    """Example of normalizing X (Twitter) analytics data."""
    logger.info("=== X Analytics Normalization Example ===")

    # Sample raw tweet data from X API v2
    raw_tweet_data = {
        "data": {
            "id": "1234567890",
            "text": "Excited to share our latest AI research findings! 🚀 #AI #MachineLearning #Research",
            "author_id": "user123",
            "created_at": "2024-01-15T10:30:00.000Z",
            "public_metrics": {
                "retweet_count": 45,
                "like_count": 234,
                "reply_count": 18,
                "quote_count": 12,
                "impression_count": 15420,
            },
            "entities": {
                "hashtags": [
                    {"start": 62, "end": 65, "tag": "AI"},
                    {"start": 66, "end": 82, "tag": "MachineLearning"},
                    {"start": 83, "end": 92, "tag": "Research"},
                ],
                "mentions": [],
            },
            "conversation_id": "1234567890",
            "lang": "en",
        }
    }

    # Initialize normalizer
    normalizer = XAnalyticsNormalizer()

    # Normalize the tweet data
    normalized_post = normalizer.normalize_tweet_data(raw_tweet_data)

    logger.info(f"Normalized post ID: {normalized_post.post_id}")
    logger.info(f"Content: {normalized_post.content}")
    logger.info(f"Engagement rate: {normalized_post.engagement_rate:.2f}%")
    logger.info(f"Hashtags: {normalized_post.hashtags}")
    logger.info(f"Metrics count: {len(normalized_post.metrics)}")

    return normalized_post


def example_content_generation():
    """Example of DSPy-powered content generation."""
    logger.info("\n=== DSPy Content Generation Example ===")

    # Create content brief
    content_brief = ContentBrief(
        topic="AI and Machine Learning trends in 2024",
        platform=Platform.X,
        content_type=ContentType.EDUCATIONAL,
        target_audience="Tech professionals and AI enthusiasts",
        key_message="AI is transforming industries faster than ever",
        tone="professional yet accessible",
        hashtags=["#AI", "#MachineLearning", "#Tech2024"],
        include_cta=True,
        metadata={"industry": "technology", "posting_time": "morning"},
    )

    # Initialize content pipeline
    pipeline = SocialContentPipeline()

    # Generate content (Note: This requires DSPy to be properly configured)
    try:
        results = pipeline.generate_content(
            content_brief=content_brief,
            trending_topics=["#AI", "#MachineLearning", "#TechTrends"],
        )

        logger.info("Content generation results:")
        if "ideation" in results:
            ideation = results["ideation"]
            logger.info(f"Content ideas: {ideation.get('content_ideas', 'N/A')}")

        if "composition" in results:
            composition = results["composition"]
            logger.info(f"Generated post: {composition.get('post_content', 'N/A')}")
            logger.info(f"Character count: {composition.get('character_count', 0)}")

    except Exception as e:
        logger.warning(f"Content generation failed (DSPy may not be configured): {e}")

        # Fallback example
        logger.info("Using fallback content example:")
        example_post = "🚀 The AI revolution of 2024 is here! From generative models to autonomous systems, machine learning is reshaping every industry. What trends are you most excited about? #AI #MachineLearning #Tech2024"
        logger.info(f"Example post: {example_post}")
        logger.info(f"Character count: {len(example_post)}")

    return content_brief


def example_kpi_calculation():
    """Example of KPI calculation for social media metrics."""
    logger.info("\n=== KPI Calculation Example ===")

    # Create sample post metrics
    post_metrics = PostMetrics(
        post_id="1234567890",
        platform=Platform.X,
        timestamp=datetime.now(UTC),
        impressions=15420,
        reach=12340,
        likes=234,
        comments=18,
        shares=45,  # retweets
        clicks=89,
        link_clicks=34,
        followers_gained=5,
        profile_visits=67,
    )

    # Initialize KPI calculator
    calculator = SocialKPICalculator()

    # Calculate all KPIs
    total_followers = 10000  # Example follower count
    kpis = calculator.calculate_all_kpis(post_metrics, total_followers)

    logger.info("Calculated KPIs:")
    for kpi in kpis:
        performance_text = (
            f" ({kpi.performance_level})" if kpi.performance_level else ""
        )
        logger.info(f"  {kpi.name}: {kpi.value} {kpi.unit}{performance_text}")

    # Create dashboard for multiple posts
    post_metrics_list = [post_metrics]  # In real usage, this would be multiple posts

    dashboard = calculator.create_dashboard(
        account_id="@example_account",
        platform=Platform.X,
        post_metrics=post_metrics_list,
        period_start=datetime.now(UTC) - timedelta(days=7),
        period_end=datetime.now(UTC),
        total_followers=total_followers,
    )

    logger.info("\nDashboard Summary:")
    logger.info(f"  Total posts analyzed: {dashboard.total_posts}")
    logger.info(
        f"  Overall engagement rate: {dashboard.summary.get('overall_engagement_rate', 0):.2f}%"
    )
    logger.info(
        f"  Average impressions per post: {dashboard.summary.get('avg_impressions_per_post', 0):,.0f}"
    )

    if dashboard.recommendations:
        logger.info(f"  Top recommendation: {dashboard.recommendations[0]}")

    return dashboard


def example_integrated_workflow():
    """Example of integrated workflow combining all modules."""
    logger.info("\n=== Integrated Workflow Example ===")

    # Step 1: Normalize incoming analytics data
    logger.info("Step 1: Normalizing analytics data...")
    normalized_post = example_x_analytics_normalization()

    # Step 2: Generate new content based on performance insights
    logger.info("\nStep 2: Generating optimized content...")

    # Extract insights from normalized data
    if normalized_post.engagement_rate and normalized_post.engagement_rate > 5.0:
        content_type = (
            ContentType.EDUCATIONAL
        )  # High engagement suggests educational content works
        tone = "professional"
    else:
        content_type = ContentType.ENTERTAINING  # Try more engaging content
        tone = "casual"

    # Create content brief based on insights
    optimized_brief = ContentBrief(
        topic="AI developments inspired by community feedback",
        platform=Platform.X,
        content_type=content_type,
        target_audience="Tech professionals",
        key_message="Building on successful content themes",
        tone=tone,
        hashtags=normalized_post.hashtags,  # Reuse successful hashtags
        include_cta=True,
        metadata={
            "based_on_post": normalized_post.post_id,
            "previous_engagement": normalized_post.engagement_rate,
        },
    )

    logger.info(f"  Content type selected: {content_type.value}")
    logger.info(f"  Tone selected: {tone}")
    logger.info(f"  Reusing hashtags: {optimized_brief.hashtags}")

    # Step 3: Calculate performance predictions
    logger.info("\nStep 3: Calculating performance predictions...")
    dashboard = example_kpi_calculation()

    # Extract insights for future content
    avg_engagement = dashboard.summary.get("overall_engagement_rate", 0)
    logger.info(f"  Historical average engagement: {avg_engagement:.2f}%")

    if avg_engagement > 3.0:
        logger.info("  Prediction: Strong engagement expected for similar content")
    else:
        logger.info("  Prediction: Consider adjusting content strategy")

    # Step 4: Provide recommendations
    logger.info("\nStep 4: Strategic recommendations...")
    for i, recommendation in enumerate(dashboard.recommendations[:3], 1):
        logger.info(f"  {i}. {recommendation}")

    return {
        "normalized_data": normalized_post,
        "content_brief": optimized_brief,
        "kpi_dashboard": dashboard,
    }


def main():
    """Run all examples."""
    logger.info("Running REER × DSPy × MLX Social Media Module Examples")
    logger.info("=" * 60)

    try:
        # Run individual examples
        example_x_analytics_normalization()
        example_content_generation()
        example_kpi_calculation()

        # Run integrated workflow
        integrated_results = example_integrated_workflow()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("\nKey capabilities demonstrated:")
        logger.info("✓ X (Twitter) analytics data normalization")
        logger.info("✓ DSPy-powered content generation")
        logger.info("✓ Comprehensive KPI calculation")
        logger.info("✓ Integrated workflow optimization")

        return integrated_results

    except Exception as e:
        logger.exception(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run examples
    results = main()
