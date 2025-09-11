#!/usr/bin/env python3
"""
Example usage of the REER × DSPy × MLX plugin system.

This script demonstrates how to use the language model adapters,
scoring heuristics, and provider routing system.
"""

import asyncio
import os
from typing import Dict, Any

# Import plugin modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from plugins import (
    get_registry,
    generate_text,
    calculate_perplexity,
    get_recommended_model,
    create_platform_scorer,
    score_content_async,
    ContentAnalyzer,
)


async def example_basic_usage():
    """Demonstrate basic usage of the plugin system."""
    print("=== Basic Plugin System Usage ===\n")

    # Get the global registry
    registry = get_registry()

    # List available providers
    print("Available providers:")
    for provider in registry.list_providers():
        print(
            f"  - {provider.name} ({provider.scheme}://) - {', '.join(provider.capabilities)}"
        )

    print()


async def example_model_routing():
    """Demonstrate model routing with different URI schemes."""
    print("=== Model Routing Examples ===\n")

    test_prompt = "Generate a compelling social media post about sustainable technology"

    # Example 1: Using dummy provider for testing
    print("1. Using dummy provider:")
    try:
        response = await generate_text("dummy://test-model", test_prompt, max_tokens=50)
        print(f"   Response: {response}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Example 2: Using MLX provider (if available)
    print("2. Using MLX provider:")
    try:
        mlx_uri = "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit"
        response = await generate_text(
            mlx_uri, test_prompt, max_tokens=50, temperature=0.7
        )
        print(f"   Response: {response}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Example 3: Using DSPy provider (requires API keys)
    print("3. Using DSPy provider:")
    try:
        if os.getenv("OPENAI_API_KEY"):
            dspy_uri = "dspy://openai/gpt-3.5-turbo"
            response = await generate_text(dspy_uri, test_prompt, max_tokens=50)
            print(f"   Response: {response}\n")
        else:
            print("   Skipped: No OPENAI_API_KEY found\n")
    except Exception as e:
        print(f"   Error: {e}\n")


async def example_content_scoring():
    """Demonstrate content scoring and heuristics."""
    print("=== Content Scoring Examples ===\n")

    sample_texts = [
        "Check out this amazing new tech! 🚀 Perfect for sustainability lovers #tech #green",
        "This is a very long and complex sentence with many technical terms and jargon that might be difficult for users to understand, potentially reducing engagement on social media platforms.",
        "Love this! So cool 😍 #awesome #love #amazing #cool #great #perfect #best #incredible #wow #fantastic",
        "How can we make technology more sustainable? What are your thoughts? 🌱",
    ]

    # Analyze each text
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

        # Basic content analysis
        metrics = ContentAnalyzer.analyze_content(text)
        print(f"  Word count: {metrics.word_count}")
        print(f"  Readability: {metrics.readability_score:.1f}")
        print(f"  Sentiment: {metrics.sentiment_score:.2f}")
        print(f"  Hashtags: {metrics.hashtag_count}")
        print(f"  Emojis: {metrics.emoji_count}")
        print(f"  Engagement potential: {metrics.engagement_potential:.2f}")

        # Platform-specific scoring
        twitter_score, components = await score_content_async(text, "twitter")
        print(f"  Twitter score: {twitter_score:.2f}")
        print(
            f"    Components: {', '.join(f'{k}={v:.2f}' for k, v in components.items())}"
        )

        print()


async def example_perplexity_calculation():
    """Demonstrate perplexity calculation."""
    print("=== Perplexity Calculation ===\n")

    test_texts = [
        "This is a natural and fluent sentence.",
        "Random words assembled without coherent meaning structure.",
        "The sustainable technology revolution is transforming our world.",
    ]

    for text in test_texts:
        print(f'Text: "{text}"')

        # Calculate perplexity using dummy model
        try:
            perplexity = await calculate_perplexity("dummy://test-model", text)
            print(f"  Perplexity: {perplexity:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

        print()


async def example_recommendations():
    """Demonstrate model recommendations."""
    print("=== Model Recommendations ===\n")

    use_cases = ["social_media", "creative", "analysis", "general"]

    for use_case in use_cases:
        local_model = get_recommended_model(use_case, prefer_local=True)
        cloud_model = get_recommended_model(use_case, prefer_local=False)

        print(f"Use case: {use_case}")
        print(f"  Local recommendation: {local_model}")
        print(f"  Cloud recommendation: {cloud_model}")
        print()


async def example_health_check():
    """Demonstrate health checking of providers."""
    print("=== Provider Health Check ===\n")

    registry = get_registry()
    health_status = await registry.health_check()

    for provider, status in health_status.items():
        print(f"Provider: {provider}")
        print(f"  Available: {status.get('available', False)}")

        if status.get("error"):
            print(f"  Error: {status['error']}")
        else:
            print(f"  Scheme: {status.get('scheme', 'unknown')}")
            print(f"  Default model: {status.get('default_model', 'unknown')}")
            print(f"  Capabilities: {', '.join(status.get('capabilities', []))}")

            if "test_generation" in status:
                print(f"  Test generation: {status['test_generation']}")
                if status.get("test_response_length"):
                    print(f"  Test response length: {status['test_response_length']}")

        print()


async def main():
    """Run all examples."""
    print("REER × DSPy × MLX Plugin System Examples")
    print("=" * 50)
    print()

    try:
        await example_basic_usage()
        await example_model_routing()
        await example_content_scoring()
        await example_perplexity_calculation()
        await example_recommendations()
        await example_health_check()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
