#!/usr/bin/env python3
"""
End-to-End X/Twitter Data Collection and Analysis Demo
Demonstrates the complete pipeline from collection to KPI analysis
"""

import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.progress import track
from rich.table import Table

from social.collectors import XAnalyticsNormalizer
from social.kpis import Platform, PostMetrics, SocialKPICalculator

console = Console()


async def demo_collection_pipeline():
    """Run complete data collection and analysis pipeline"""

    console.print(
        "\n[bold cyan]🚀 X/Twitter Data Collection & Analytics Pipeline Demo[/bold cyan]\n"
    )

    # Step 1: Simulate collected tweet data (since we need auth for real collection)
    console.print("[yellow]📊 Step 1: Simulating collected tweet data...[/yellow]")

    sample_tweets = [
        {
            "id": "1234567890",
            "text": "🚀 Just launched our new AI model! It achieves 95% accuracy on benchmark tests. Check it out! #AI #MachineLearning #Tech",
            "created_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            "author_id": "user123",
            "public_metrics": {
                "like_count": 342,
                "retweet_count": 89,
                "reply_count": 23,
                "quote_count": 12,
                "impression_count": 5420,
                "bookmark_count": 45,
            },
            "entities": {
                "hashtags": [
                    {"tag": "AI"},
                    {"tag": "MachineLearning"},
                    {"tag": "Tech"},
                ],
                "mentions": [],
            },
        },
        {
            "id": "1234567891",
            "text": "Machine learning tip of the day: Always validate your models with cross-validation! 📈 It prevents overfitting and gives you more reliable performance estimates. #MLTips #DataScience",
            "created_at": (datetime.now(UTC) - timedelta(hours=5)).isoformat(),
            "author_id": "user123",
            "public_metrics": {
                "like_count": 567,
                "retweet_count": 123,
                "reply_count": 45,
                "quote_count": 8,
                "impression_count": 8900,
                "bookmark_count": 89,
            },
            "entities": {
                "hashtags": [{"tag": "MLTips"}, {"tag": "DataScience"}],
                "mentions": [],
            },
        },
        {
            "id": "1234567892",
            "text": "Breaking: GPT-5 rumors suggest 10x performance improvement! 🤖 The future of AI is here. What are your predictions? #GPT5 #AI #FutureOfWork",
            "created_at": (datetime.now(UTC) - timedelta(hours=8)).isoformat(),
            "author_id": "user123",
            "public_metrics": {
                "like_count": 1243,
                "retweet_count": 456,
                "reply_count": 234,
                "quote_count": 67,
                "impression_count": 25600,
                "bookmark_count": 234,
            },
            "entities": {
                "hashtags": [{"tag": "GPT5"}, {"tag": "AI"}, {"tag": "FutureOfWork"}],
                "mentions": [],
            },
        },
    ]

    # Step 2: Normalize the data
    console.print("\n[yellow]🔄 Step 2: Normalizing tweet data...[/yellow]")
    normalizer = XAnalyticsNormalizer()
    normalized_posts = []

    for tweet in track(sample_tweets, description="Normalizing tweets"):
        normalized = normalizer.normalize_tweet_data(tweet)
        normalized_posts.append(normalized)
        await asyncio.sleep(0.1)  # Simulate processing

    # Display normalized data
    table = Table(title="Normalized Tweet Data", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Text Preview", style="white", width=40)
    table.add_column("Engagement Rate", style="green")
    table.add_column("Impressions", style="yellow")

    for post in normalized_posts:
        table.add_row(
            post.post_id,
            post.content[:40] + "...",
            f"{post.engagement_rate:.2%}" if post.engagement_rate else "N/A",
            str(post.impressions) if post.impressions else "N/A",
        )

    console.print(table)

    # Step 3: Calculate KPIs
    console.print("\n[yellow]📈 Step 3: Calculating KPIs...[/yellow]")
    kpi_calculator = SocialKPICalculator()

    kpi_results = []
    for post in normalized_posts:
        # Convert to PostMetrics
        metrics = PostMetrics(
            post_id=post.post_id,
            platform=Platform.X,
            timestamp=post.timestamp,
            impressions=post.impressions or 0,
            reach=post.reach or 0,
            likes=next(
                (m.value for m in post.metrics if m.metric_type.value == "likes"), 0
            ),
            comments=next(
                (m.value for m in post.metrics if m.metric_type.value == "replies"), 0
            ),
            shares=next(
                (m.value for m in post.metrics if m.metric_type.value == "retweets"), 0
            ),
        )

        # Calculate various KPIs
        engagement_rate = kpi_calculator.calculate_engagement_rate(metrics)
        virality_score = kpi_calculator.calculate_virality_score(metrics)
        ctr = kpi_calculator.calculate_click_through_rate(metrics)

        kpi_results.append(
            {
                "post_id": post.post_id,
                "engagement_rate": engagement_rate,
                "virality_score": virality_score,
                "click_through_rate": ctr,
            }
        )

    # Display KPI results
    kpi_table = Table(title="Social Media KPIs", show_header=True)
    kpi_table.add_column("Post ID", style="cyan")
    kpi_table.add_column("Engagement Rate", style="green")
    kpi_table.add_column("Virality Score", style="yellow")
    kpi_table.add_column("Click-Through Rate", style="magenta")

    for kpi in kpi_results:
        kpi_table.add_row(
            kpi["post_id"],
            f"{kpi['engagement_rate'].value:.2%}",
            f"{kpi['virality_score'].value:.2f}",
            f"{kpi['click_through_rate'].value:.2%}",
        )

    console.print(kpi_table)

    # Step 4: Content Performance Analysis
    console.print("\n[yellow]🎯 Step 4: Content Performance Analysis...[/yellow]")

    # Analyze hashtag performance
    hashtag_performance = {}
    for post in normalized_posts:
        for tag in post.hashtags:
            if tag not in hashtag_performance:
                hashtag_performance[tag] = {"count": 0, "total_engagement": 0}
            hashtag_performance[tag]["count"] += 1
            hashtag_performance[tag]["total_engagement"] += post.engagement_rate or 0

    # Calculate average engagement per hashtag
    hashtag_table = Table(title="Hashtag Performance", show_header=True)
    hashtag_table.add_column("Hashtag", style="cyan")
    hashtag_table.add_column("Usage Count", style="white")
    hashtag_table.add_column("Avg Engagement", style="green")

    for tag, stats in sorted(
        hashtag_performance.items(),
        key=lambda x: x[1]["total_engagement"],
        reverse=True,
    ):
        avg_engagement = stats["total_engagement"] / stats["count"]
        hashtag_table.add_row(f"#{tag}", str(stats["count"]), f"{avg_engagement:.2%}")

    console.print(hashtag_table)

    # Step 5: Generate recommendations
    console.print("\n[yellow]💡 Step 5: Generating Recommendations...[/yellow]")

    best_post = max(kpi_results, key=lambda x: x["virality_score"].value)
    worst_post = min(kpi_results, key=lambda x: x["engagement_rate"].value)

    recommendations = [
        f"✅ Your best performing content (ID: {best_post['post_id']}) had a virality score of {best_post['virality_score'].value:.2f}",
        f"📊 Average engagement rate across all posts: {sum(k['engagement_rate'].value for k in kpi_results)/len(kpi_results):.2%}",
        f"🎯 Top performing hashtags: {', '.join(['#' + tag for tag in list(hashtag_performance.keys())[:3]])}",
        "⚡ Consider posting during peak hours based on your highest engagement times",
        "🔄 Posts with questions or calls-to-action show higher engagement rates",
    ]

    for rec in recommendations:
        console.print(f"  {rec}")

    # Step 6: Export results
    console.print("\n[yellow]💾 Step 6: Exporting results...[/yellow]")

    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save normalized data
    with open(output_dir / "normalized_tweets.json", "w") as f:
        json.dump(
            [
                {
                    "post_id": p.post_id,
                    "content": p.content,
                    "engagement_rate": p.engagement_rate,
                    "hashtags": p.hashtags,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in normalized_posts
            ],
            f,
            indent=2,
        )

    # Save KPI results
    with open(output_dir / "kpi_results.json", "w") as f:
        json.dump(
            [
                {
                    "post_id": k["post_id"],
                    "engagement_rate": k["engagement_rate"].value,
                    "virality_score": k["virality_score"].value,
                    "click_through_rate": k["click_through_rate"].value,
                }
                for k in kpi_results
            ],
            f,
            indent=2,
        )

    console.print(f"\n[green]✅ Results exported to {output_dir}/[/green]")
    console.print("\n[bold cyan]🎉 Pipeline completed successfully![/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(demo_collection_pipeline())
