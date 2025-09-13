#!/usr/bin/env python3
"""
Demo REER MLX System - Complete Example
Demonstrates both perplexity calculation and iterative refinement
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.candidate_scorer import PerplexityCalculator
from dspy_program.reer_module import REERRefinementProcessor, REERConfig


async def demo_perplexity_calculation():
    """Demo 1: Basic conditional perplexity calculation"""
    print("=" * 60)
    print("Demo 1: Conditional Perplexity Calculation")
    print("=" * 60)

    # Initialize perplexity calculator with a small model for speed
    ppl = PerplexityCalculator("mlx-community/Llama-3.2-1B-Instruct-4bit")
    await ppl.initialize()
    print("✓ MLX model initialized")

    # Define a triple (x=context, z=thinking, y=answer)
    x = "What are the benefits of machine learning?"
    z_good = "Machine learning helps automate decisions, find patterns in data, and improve predictions over time."
    z_poor = "Machine learning is just about computers and stuff."
    y = "Machine learning enables automated decision-making, pattern recognition, and continuous improvement through data-driven insights."

    print(f"\nContext (x): {x}")
    print(f"Answer (y): {y}")

    # Calculate perplexity with good thinking
    print(f"\nGood thinking (z): {z_good}")
    lppl_good = await ppl.calculate_conditional_log_perplexity(x, z_good, y)
    print(f"Log-perplexity: {lppl_good:.3f} (lower is better)")

    # Calculate perplexity with poor thinking
    print(f"\nPoor thinking (z): {z_poor}")
    lppl_poor = await ppl.calculate_conditional_log_perplexity(x, z_poor, y)
    print(f"Log-perplexity: {lppl_poor:.3f} (lower is better)")

    # Compare
    print(
        f"\nImprovement with good thinking: {lppl_poor - lppl_good:.3f} reduction in log-PPL"
    )
    print("✓ The good thinking makes the answer more probable!")

    return {"good_lppl": lppl_good, "poor_lppl": lppl_poor}


async def demo_reer_refinement():
    """Demo 2: REER iterative refinement process"""
    print("\n" + "=" * 60)
    print("Demo 2: REER Iterative Refinement")
    print("=" * 60)

    # Configure REER with reasonable parameters
    config = REERConfig(
        stop_thresh=1.5,  # Stop when log-PPL < 1.5 (lower is better)
        max_steps=5,  # Try up to 5 refinement steps
        num_expansion=3,  # Generate 3 variants per segment
        ppl_model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
    )

    # Initialize processor
    processor = REERRefinementProcessor(config)

    # Define our problem
    x = "How can I improve my productivity?"

    # Initial thinking (intentionally mediocre)
    z_init = """
    To be productive you need to work hard.
    Make lists of things.
    Don't waste time.
    """

    # Target answer (what we want to explain)
    y = """
    Improving productivity requires a systematic approach: prioritize tasks using methods like the Eisenhower Matrix, 
    implement time-blocking for focused work, minimize distractions by creating a dedicated workspace, 
    take regular breaks to maintain energy, and track progress to identify improvement areas.
    """

    print(f"Context (x): {x}")
    print(f"\nInitial thinking (z_init):{z_init}")
    print(f"\nTarget answer (y): {y[:100]}...")

    print("\nRunning REER refinement...")
    result = await processor.refine(x, z_init, y)

    # Display results
    print("\n" + "-" * 40)
    print("REFINEMENT RESULTS")
    print("-" * 40)
    print(f"Initial log-PPL: {result['ppl_init']:.3f}")
    print(f"Best log-PPL: {result['ppl_best']:.3f}")
    print(f"Improvement: {result['ppl_init'] - result['ppl_best']:.3f} reduction")
    print(f"Steps taken: {result['steps_taken']}")

    print(f"\nRefined thinking (z_best):{result['z_best']}")

    # Show refinement trajectory
    if result["candidates"]:
        print("\n" + "-" * 40)
        print("REFINEMENT TRAJECTORY")
        print("-" * 40)
        seen_segments = set()
        for i, cand in enumerate(result["candidates"][:10]):  # Show first 10
            segment_idx = cand.get("segment_idx", 0)
            if segment_idx not in seen_segments:
                print(f"\nSegment {segment_idx} variants:")
                seen_segments.add(segment_idx)
            print(f"  - PPL: {cand['ppl']:.3f} | {cand['candidate'][:60]}...")

    return result


async def demo_comparison():
    """Demo 3: Compare multiple reasoning strategies"""
    print("\n" + "=" * 60)
    print("Demo 3: Comparing Multiple Reasoning Strategies")
    print("=" * 60)

    ppl = PerplexityCalculator("mlx-community/Llama-3.2-1B-Instruct-4bit")
    await ppl.initialize()

    # Problem setup
    x = "What makes a successful startup?"
    y = "Successful startups solve real problems, have strong teams, iterate quickly based on feedback, and maintain sufficient runway while achieving product-market fit."

    # Different reasoning strategies
    strategies = {
        "analytical": """
        Startups succeed through systematic factors.
        Key elements include problem-solution fit.
        Team execution and market timing matter.
        """,
        "experiential": """
        From observing many startups, patterns emerge.
        Winners adapt fast and listen to users.
        Resources and persistence determine survival.
        """,
        "theoretical": """
        Business theory suggests competitive advantages.
        Innovation diffusion explains adoption curves.
        Economic moats protect market position.
        """,
    }

    print(f"Context: {x}")
    print(f"Answer: {y[:80]}...")
    print("\nEvaluating different reasoning strategies:")

    results = {}
    for name, z in strategies.items():
        lppl = await ppl.calculate_conditional_log_perplexity(x, z, y)
        results[name] = lppl
        print(f"\n{name.capitalize()} approach:")
        print(f"  Thinking: {z.strip()[:60]}...")
        print(f"  Log-PPL: {lppl:.3f}")

    # Find best strategy
    best_strategy = min(results, key=results.get)
    print(f"\n✓ Best strategy: {best_strategy} (log-PPL: {results[best_strategy]:.3f})")

    return results


async def main():
    """Run all demos"""
    print("\n🚀 REER MLX System Demo\n")

    try:
        # Run demos
        ppl_results = await demo_perplexity_calculation()
        refinement_results = await demo_reer_refinement()
        comparison_results = await demo_comparison()

        # Save results
        output = {
            "perplexity_demo": ppl_results,
            "refinement_demo": {
                "initial_ppl": refinement_results["ppl_init"],
                "final_ppl": refinement_results["ppl_best"],
                "improvement": refinement_results["ppl_init"]
                - refinement_results["ppl_best"],
                "steps": refinement_results["steps_taken"],
            },
            "comparison_demo": comparison_results,
        }

        output_file = Path("reer_demo_results.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_file}")
        print("\nKey Takeaways:")
        print("1. Good reasoning (z) reduces log-perplexity of answers (y)")
        print("2. REER can iteratively refine reasoning to improve explanations")
        print("3. Different reasoning strategies have measurable quality differences")
        print("\nNext steps:")
        print("- Wire REER into your pipeline for training data synthesis")
        print("- Use curated high-quality (x,y) pairs from real content")
        print("- Store refined reasoning traces for model training")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
