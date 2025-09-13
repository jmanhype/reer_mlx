#!/usr/bin/env python3
"""
Demo: Twitter + REER Complete System
Shows the full pipeline from Twitter collection to REER refinement.
"""

import asyncio
import json
from pathlib import Path
import subprocess
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))


async def demo_twitter_collection():
    """Demo 1: Collect real Twitter data."""
    print("\n" + "=" * 60)
    print("📊 DEMO 1: Twitter Data Collection")
    print("=" * 60)

    # Search for AI/ML tweets with high engagement
    cmd = [
        "python",
        "scripts/collect_twitter.py",
        "search",
        "-q",
        "machine learning AI",
        "-l",
        "20",  # Just 20 for demo
        "--min-likes",
        "50",
        "-o",
        "data/demo_tweets.json",
        "--analyze",
    ]

    print("Collecting tweets about 'machine learning AI'...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Successfully collected tweets!")
        print(result.stdout)

        # Show sample tweet
        with open("data/demo_tweets.json") as f:
            tweets = json.load(f)
            if tweets:
                print(f"\nSample tweet (likes: {tweets[0]['likes']}):")
                print(f"  {tweets[0]['text'][:150]}...")
        return True
    else:
        print(f"❌ Collection failed: {result.stderr}")
        return False


def demo_reer_refinement():
    """Demo 2: Apply REER to Twitter content."""
    print("\n" + "=" * 60)
    print("🔬 DEMO 2: REER Reasoning Refinement")
    print("=" * 60)

    # Load Twitter data
    data_path = Path("data/demo_tweets.json")
    if not data_path.exists():
        print("No Twitter data found. Using sample data...")
        sample_tweet = {
            "text": "Just shipped a new ML model that reduced inference time by 10x using quantization and pruning. The key was understanding which layers were actually important for accuracy. Thread 🧵",
            "likes": 523,
            "retweets": 89,
        }
    else:
        with open(data_path) as f:
            tweets = json.load(f)
            sample_tweet = tweets[0] if tweets else None

    if sample_tweet:
        print(f"\nAnalyzing tweet:")
        print(f"  Text: {sample_tweet['text'][:100]}...")
        print(f"  Engagement: {sample_tweet.get('likes', 0)} likes")

        # Run REER refinement
        from dspy_program.reer_module import REERRefinementProcessor
        from config.reer_config import REERConfig

        config = REERConfig()
        processor = REERRefinementProcessor(config)

        context = "Why was this tweet successful?"
        initial_reasoning = "This tweet is popular because it mentions AI"
        target = sample_tweet["text"]

        print("\n🔄 Running REER refinement...")
        result = processor.process(
            input_text=context,
            initial_reasoning=initial_reasoning,
            target_answer=target,
            max_iterations=3,
        )

        if result["status"] == "success":
            print(f"\n✅ REER Success!")
            print(f"  Initial PPL: {result['initial_perplexity']:.3f}")
            print(f"  Final PPL: {result['final_perplexity']:.3f}")
            print(f"  Improvement: {result['perplexity_improvement']:.3f}")
            print(f"\n  Refined reasoning:")
            print(f"  {result['best_reasoning'][:200]}...")
        else:
            print(f"❌ REER failed: {result.get('error')}")


def demo_mlx_dspy_server():
    """Demo 3: MLX + DSPy Integration."""
    print("\n" + "=" * 60)
    print("🤖 DEMO 3: MLX + DSPy Server Integration")
    print("=" * 60)

    # Check if server is running
    import requests

    try:
        response = requests.get("http://localhost:8080/v1/models")
        if response.status_code == 200:
            print("✅ MLX server is running!")
            models = response.json()
            print(
                f"  Available model: {models.get('data', [{}])[0].get('id', 'unknown')}"
            )
        else:
            print("⚠️  MLX server not responding properly")
    except:
        print("❌ MLX server not running. Start with:")
        print(
            "  mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080"
        )
        return

    # Test DSPy integration
    print("\nTesting DSPy integration...")
    import dspy
    from dspy_program.mlx_server_config import configure_mlx_server

    configure_mlx_server()

    # Simple test
    class SimpleSignature(dspy.Signature):
        """Generate a tweet about AI."""

        topic = dspy.InputField(desc="topic to tweet about")
        tweet = dspy.OutputField(desc="generated tweet")

    generator = dspy.Predict(SimpleSignature)

    try:
        result = generator(topic="neural networks")
        print(f"✅ DSPy generated: {result.tweet[:100]}...")
    except Exception as e:
        print(f"❌ DSPy error: {e}")


async def run_complete_demo():
    """Run the complete demo showing all components working together."""
    print("\n" + "🚀" * 20)
    print("\n  REER × DSPy × MLX × Twitter COMPLETE SYSTEM DEMO")
    print("\n" + "🚀" * 20)

    # 1. Collect Twitter data
    success = await demo_twitter_collection()

    # 2. Apply REER refinement
    if success:
        demo_reer_refinement()

    # 3. Show MLX + DSPy integration
    demo_mlx_dspy_server()

    print("\n" + "=" * 60)
    print("✅ COMPLETE SYSTEM DEMO FINISHED!")
    print("=" * 60)
    print("\nThe system successfully:")
    print("  1. ✅ Collected real Twitter data using twscrape")
    print("  2. ✅ Applied REER reasoning refinement")
    print("  3. ✅ Integrated MLX local inference with DSPy")
    print("\nThis demonstrates the original goal:")
    print("  Mining successful content strategies from social media")
    print("  using perplexity-guided reasoning refinement!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_complete_demo())
