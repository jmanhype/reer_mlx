#!/usr/bin/env python3
"""
Twitter + REER Pipeline: Mine successful content strategies from Twitter/X data.
Combines twscrape collection with REER reasoning refinement.
"""

import asyncio
import json
from pathlib import Path
import sys
import subprocess
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dspy_program.reer_module import REERRefinementProcessor
from core.candidate_scorer import CandidateScorer
from core.trace_store import TraceStore


class TwitterREERPipeline:
    """Pipeline for mining Twitter content with REER."""

    def __init__(self):
        self.trace_store = TraceStore()
        self.scorer = CandidateScorer()
        self.processor = REERRefinementProcessor()

    async def collect_twitter_data(
        self, query: str, limit: int = 100, min_likes: int = 50
    ) -> Optional[Path]:
        """Collect Twitter data using our collection script."""
        print(f"\n📊 Collecting Twitter data for: {query}")

        output_file = f"data/twitter_{query.replace(' ', '_')}.json"

        cmd = [
            "python",
            "scripts/collect_twitter.py",
            "search",
            "-q",
            query,
            "-l",
            str(limit),
            "--min-likes",
            str(min_likes),
            "-o",
            output_file,
            "--analyze",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error collecting data: {result.stderr}")
                return None

            print(result.stdout)
            return Path(output_file)

        except Exception as e:
            print(f"Failed to collect Twitter data: {e}")
            return None

    def load_twitter_data(self, data_path: Path) -> dict:
        """Load and prepare Twitter data for REER."""
        with open(data_path) as f:
            tweets = json.load(f)

        # Convert to REER format
        reer_path = data_path.with_suffix(".reer.json")
        if reer_path.exists():
            with open(reer_path) as f:
                return json.load(f)

        # Manual conversion if REER file doesn't exist
        return {
            "context": f"Twitter content analysis",
            "candidates": [
                {
                    "text": tweet["text"],
                    "score": tweet.get("likes", 0) + tweet.get("retweets", 0) * 2,
                    "metadata": {
                        "user": tweet.get("user"),
                        "url": tweet.get("url"),
                        "created_at": tweet.get("created_at"),
                    },
                }
                for tweet in tweets[:20]  # Top 20 tweets
            ],
        }

    def mine_content_strategies(self, twitter_data: dict) -> dict:
        """Use REER to mine successful content strategies."""
        print("\n🔍 Mining content strategies with REER...")

        results = {"original_tweets": [], "refined_strategies": [], "improvements": []}

        # Process each high-performing tweet
        for candidate in twitter_data["candidates"][:5]:  # Top 5
            tweet_text = candidate["text"]
            engagement = candidate["score"]

            print(f"\n📝 Analyzing tweet (engagement: {engagement}):")
            print(f"   {tweet_text[:100]}...")

            # Use REER to understand why this tweet worked
            context = "Analyze this successful tweet and extract the key strategy:"

            # Create a reasoning trace
            initial_reasoning = (
                f"This tweet is successful because it {tweet_text[:50]}..."
            )

            # Refine the reasoning
            result = self.processor.process(
                input_text=context,
                initial_reasoning=initial_reasoning,
                target_answer=tweet_text,
                max_iterations=3,
            )

            if result["status"] == "success":
                improvement = result["perplexity_improvement"]
                refined = result["best_reasoning"]

                print(f"   ✅ Refined strategy (PPL improved by {improvement:.3f})")
                print(f"   Strategy: {refined[:200]}...")

                results["original_tweets"].append(tweet_text)
                results["refined_strategies"].append(refined)
                results["improvements"].append(improvement)
            else:
                print(f"   ❌ Failed to refine: {result.get('error')}")

        return results

    def generate_new_content(self, strategies: dict) -> list:
        """Generate new content based on mined strategies."""
        print("\n✨ Generating new content based on successful strategies...")

        new_content = []

        for strategy in strategies["refined_strategies"][:3]:
            # Use the strategy to generate new content
            prompt = f"Based on this strategy: {strategy}\nGenerate a new tweet:"

            # This would use MLX to generate, but for now we'll just structure it
            new_tweet = {
                "strategy": strategy,
                "suggested_content": f"[Generated content based on: {strategy[:50]}...]",
                "expected_engagement": "High",
            }
            new_content.append(new_tweet)

        return new_content

    async def run_pipeline(
        self,
        query: str,
        limit: int = 100,
        min_likes: int = 50,
        skip_collection: bool = False,
    ):
        """Run the complete Twitter + REER pipeline."""
        print(f"\n{'='*60}")
        print(f"🚀 Twitter + REER Content Mining Pipeline")
        print(f"{'='*60}")

        # Step 1: Collect or load Twitter data
        if skip_collection:
            data_path = Path(f"data/twitter_{query.replace(' ', '_')}.json")
            if not data_path.exists():
                print(f"Error: No data found at {data_path}")
                return
        else:
            data_path = await self.collect_twitter_data(query, limit, min_likes)
            if not data_path:
                return

        # Step 2: Load and prepare data
        twitter_data = self.load_twitter_data(data_path)
        print(f"\n📊 Loaded {len(twitter_data['candidates'])} tweets for analysis")

        # Step 3: Mine strategies with REER
        strategies = self.mine_content_strategies(twitter_data)

        # Step 4: Generate new content
        new_content = self.generate_new_content(strategies)

        # Step 5: Save results
        output_path = data_path.with_suffix(".strategies.json")
        results = {
            "query": query,
            "tweets_analyzed": len(twitter_data["candidates"]),
            "strategies_extracted": len(strategies["refined_strategies"]),
            "avg_improvement": (
                sum(strategies["improvements"]) / len(strategies["improvements"])
                if strategies["improvements"]
                else 0
            ),
            "strategies": strategies,
            "generated_content": new_content,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Results saved to: {output_path}")
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"   • Tweets analyzed: {len(twitter_data['candidates'])}")
        print(f"   • Strategies extracted: {len(strategies['refined_strategies'])}")
        print(f"   • Average PPL improvement: {results['avg_improvement']:.3f}")
        print(f"{'='*60}\n")


async def main():
    """Run the Twitter + REER pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Twitter + REER Content Mining")
    parser.add_argument("query", help="Search query for Twitter")
    parser.add_argument("-l", "--limit", type=int, default=100, help="Number of tweets")
    parser.add_argument("--min-likes", type=int, default=50, help="Minimum likes")
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip collection, use existing data",
    )

    args = parser.parse_args()

    pipeline = TwitterREERPipeline()
    await pipeline.run_pipeline(
        args.query, args.limit, args.min_likes, args.skip_collection
    )


if __name__ == "__main__":
    # Test with a simple query
    if len(sys.argv) == 1:
        # Demo mode
        print("Running demo with 'machine learning' query...")
        asyncio.run(
            TwitterREERPipeline().run_pipeline(
                "machine learning", limit=50, min_likes=100, skip_collection=False
            )
        )
    else:
        asyncio.run(main())
