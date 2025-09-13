#!/usr/bin/env python3
"""
Twitter/X Data Collection Script for REER Analysis.
Uses twscrape to collect tweets for mining successful content strategies.
"""

import asyncio
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Optional

try:
    from twscrape import API, AccountsPool
except ImportError:
    print("Error: twscrape not installed. Run: pip install twscrape")
    exit(1)


async def setup_accounts(session_db: str = "accounts.db"):
    """Setup twscrape with existing accounts database."""
    pool = AccountsPool(session_db)
    api = API(pool)

    # Check if we have active accounts
    accounts = await pool.get_all()
    if not accounts:
        print("No accounts found. Please run setup_twscrape.py first")
        return None

    active = [acc for acc in accounts if acc.active]
    if not active:
        print(f"No active accounts. Found {len(accounts)} inactive accounts.")
        print("Attempting to relogin...")
        await pool.login_all()

    return api


async def search_tweets(
    api: API, query: str, limit: int = 100, min_likes: int = 10, days_back: int = 7
) -> list:
    """Search for tweets matching query."""
    tweets = []

    # Build search query with filters
    search_query = query
    if min_likes > 0:
        search_query += f" min_faves:{min_likes}"

    # Add date range
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    search_query += f" since:{since}"

    print(f"Searching: {search_query}")

    # Collect tweets
    async for tweet in api.search(search_query, limit=limit):
        tweet_data = {
            "id": tweet.id,
            "text": tweet.rawContent,
            "user": tweet.user.username,
            "created_at": tweet.date.isoformat(),
            "likes": tweet.likeCount,
            "retweets": tweet.retweetCount,
            "replies": tweet.replyCount,
            "views": tweet.viewCount,
            "url": tweet.url,
            "is_reply": tweet.inReplyToTweetId is not None,
            "lang": tweet.lang,
        }
        tweets.append(tweet_data)

        if len(tweets) % 10 == 0:
            print(f"  Collected {len(tweets)} tweets...")

    return tweets


async def user_timeline(
    api: API, username: str, limit: int = 50, include_replies: bool = False
) -> list:
    """Get tweets from a specific user."""
    tweets = []

    print(f"Getting timeline for @{username}")

    async for tweet in api.user_by_login(username, limit=limit):
        # Skip replies if not wanted
        if not include_replies and tweet.inReplyToTweetId:
            continue

        tweet_data = {
            "id": tweet.id,
            "text": tweet.rawContent,
            "user": username,
            "created_at": tweet.date.isoformat(),
            "likes": tweet.likeCount,
            "retweets": tweet.retweetCount,
            "replies": tweet.replyCount,
            "views": tweet.viewCount,
            "url": tweet.url,
            "is_reply": tweet.inReplyToTweetId is not None,
            "lang": tweet.lang,
        }
        tweets.append(tweet_data)

    return tweets


def save_tweets(tweets: list, output_path: str, format: str = "json"):
    """Save tweets to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(tweets, f, indent=2, default=str)
        print(f"Saved {len(tweets)} tweets to {output_path}")

    elif format == "csv":
        if tweets:
            keys = tweets[0].keys()
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(tweets)
            print(f"Saved {len(tweets)} tweets to {output_path}")

    return output_path


def analyze_for_reer(tweets: list) -> dict:
    """Analyze tweets for REER-compatible patterns."""
    analysis = {
        "total_tweets": len(tweets),
        "avg_engagement": 0,
        "top_performers": [],
        "content_patterns": {},
        "optimal_length": 0,
    }

    if not tweets:
        return analysis

    # Calculate engagement metrics
    total_engagement = 0
    for tweet in tweets:
        engagement = tweet["likes"] + tweet["retweets"] * 2 + tweet["replies"]
        tweet["engagement_score"] = engagement
        total_engagement += engagement

    analysis["avg_engagement"] = total_engagement / len(tweets)

    # Find top performers
    sorted_tweets = sorted(tweets, key=lambda x: x["engagement_score"], reverse=True)
    analysis["top_performers"] = sorted_tweets[:10]

    # Analyze content patterns
    lengths = [len(tweet["text"]) for tweet in tweets]
    analysis["optimal_length"] = sum(lengths) / len(lengths)

    # Identify successful patterns
    high_performers = [
        t for t in tweets if t["engagement_score"] > analysis["avg_engagement"] * 2
    ]
    if high_performers:
        print(
            f"\nFound {len(high_performers)} high-performing tweets for REER analysis"
        )

    return analysis


async def main():
    parser = argparse.ArgumentParser(
        description="Collect Twitter/X data for REER analysis"
    )
    parser.add_argument("mode", choices=["search", "user"], help="Collection mode")
    parser.add_argument("-q", "--query", help="Search query")
    parser.add_argument("-u", "--username", help="Username for timeline")
    parser.add_argument("-l", "--limit", type=int, default=100, help="Number of tweets")
    parser.add_argument(
        "--min-likes", type=int, default=10, help="Minimum likes filter"
    )
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back")
    parser.add_argument(
        "--include-replies", action="store_true", help="Include replies"
    )
    parser.add_argument(
        "-o", "--output", default="data/twitter_data.json", help="Output file"
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Output format"
    )
    parser.add_argument("--session-db", default="accounts.db", help="Session database")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze for REER patterns"
    )

    args = parser.parse_args()

    # Setup API
    api = await setup_accounts(args.session_db)
    if not api:
        return

    # Collect tweets
    tweets = []
    if args.mode == "search":
        if not args.query:
            print("Error: --query required for search mode")
            return
        tweets = await search_tweets(
            api, args.query, args.limit, args.min_likes, args.days_back
        )
    elif args.mode == "user":
        if not args.username:
            print("Error: --username required for user mode")
            return
        tweets = await user_timeline(
            api, args.username, args.limit, args.include_replies
        )

    if tweets:
        # Save tweets
        output_path = save_tweets(tweets, args.output, args.format)

        # Analyze if requested
        if args.analyze:
            print("\nAnalyzing tweets for REER patterns...")
            analysis = analyze_for_reer(tweets)

            print(f"Total tweets: {analysis['total_tweets']}")
            print(f"Average engagement: {analysis['avg_engagement']:.1f}")
            print(f"Optimal length: {analysis['optimal_length']:.0f} chars")

            # Save analysis
            analysis_path = output_path.with_suffix(".analysis.json")
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Saved analysis to {analysis_path}")

            # Create REER-ready format
            reer_data = {
                "context": f"Twitter content analysis from {args.query or args.username}",
                "candidates": [
                    {
                        "text": t["text"],
                        "score": t["engagement_score"],
                        "metadata": {
                            "likes": t["likes"],
                            "retweets": t["retweets"],
                            "url": t["url"],
                        },
                    }
                    for t in analysis["top_performers"]
                ],
            }

            reer_path = output_path.with_suffix(".reer.json")
            with open(reer_path, "w") as f:
                json.dump(reer_data, f, indent=2)
            print(f"Created REER-ready data at {reer_path}")
    else:
        print("No tweets collected")


if __name__ == "__main__":
    asyncio.run(main())
