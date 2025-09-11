#!/usr/bin/env python3
"""
Example usage of the REER × DSPy × MLX plugin system.

This script demonstrates how to use the language model adapters,
scoring heuristics, and provider routing system.
"""

import asyncio
import os

# Import plugin modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import contextlib

from plugins import (
    ContentAnalyzer,
    calculate_perplexity,
    generate_text,
    get_recommended_model,
    get_registry,
    score_content_async,
)


async def example_basic_usage():
    """Demonstrate basic usage of the plugin system."""

    # Get the global registry
    registry = get_registry()

    # List available providers
    for _provider in registry.list_providers():
        pass


async def example_model_routing():
    """Demonstrate model routing with different URI schemes."""

    test_prompt = "Generate a compelling social media post about sustainable technology"

    # Example 1: Using dummy provider for testing
    with contextlib.suppress(Exception):
        await generate_text("dummy://test-model", test_prompt, max_tokens=50)

    # Example 2: Using MLX provider (if available)
    try:
        mlx_uri = "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit"
        await generate_text(mlx_uri, test_prompt, max_tokens=50, temperature=0.7)
    except Exception:
        pass

    # Example 3: Using DSPy provider (requires API keys)
    try:
        if os.getenv("OPENAI_API_KEY"):
            dspy_uri = "dspy://openai/gpt-3.5-turbo"
            await generate_text(dspy_uri, test_prompt, max_tokens=50)
        else:
            pass
    except Exception:
        pass


async def example_content_scoring():
    """Demonstrate content scoring and heuristics."""

    sample_texts = [
        "Check out this amazing new tech! 🚀 Perfect for sustainability lovers #tech #green",
        "This is a very long and complex sentence with many technical terms and jargon that might be difficult for users to understand, potentially reducing engagement on social media platforms.",
        "Love this! So cool 😍 #awesome #love #amazing #cool #great #perfect #best #incredible #wow #fantastic",
        "How can we make technology more sustainable? What are your thoughts? 🌱",
    ]

    # Analyze each text
    for _i, text in enumerate(sample_texts, 1):

        # Basic content analysis
        ContentAnalyzer.analyze_content(text)

        # Platform-specific scoring
        twitter_score, components = await score_content_async(text, "twitter")


async def example_perplexity_calculation():
    """Demonstrate perplexity calculation."""

    test_texts = [
        "This is a natural and fluent sentence.",
        "Random words assembled without coherent meaning structure.",
        "The sustainable technology revolution is transforming our world.",
    ]

    for text in test_texts:

        # Calculate perplexity using dummy model
        with contextlib.suppress(Exception):
            await calculate_perplexity("dummy://test-model", text)


async def example_recommendations():
    """Demonstrate model recommendations."""

    use_cases = ["social_media", "creative", "analysis", "general"]

    for use_case in use_cases:
        get_recommended_model(use_case, prefer_local=True)
        get_recommended_model(use_case, prefer_local=False)


async def example_health_check():
    """Demonstrate health checking of providers."""

    registry = get_registry()
    health_status = await registry.health_check()

    for _provider, status in health_status.items():

        if status.get("error"):
            pass
        else:

            if "test_generation" in status:
                if status.get("test_response_length"):
                    pass


async def main():
    """Run all examples."""

    try:
        await example_basic_usage()
        await example_model_routing()
        await example_content_scoring()
        await example_perplexity_calculation()
        await example_recommendations()
        await example_health_check()

    except Exception:
        pass


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
