#!/usr/bin/env python3
"""
Test REER with MLX backend - minimal example
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig
from tools.ppl_eval import select_ppl_evaluator


def main():
    print("=" * 60)
    print("Testing REER with MLX Backend")
    print("=" * 60)

    # Test data
    x = "What is AI?"
    y = "AI is artificial intelligence."

    print(f"\nInput (x): {x}")
    print(f"Output (y): {y}")

    # Light config for testing
    config = TrajectorySearchConfig(
        max_iters=2,
        max_candidates_per_segment=2,
        patience=1,
        target_ppl=1.5,
        segment_window=1,
    )

    print("\nInitializing MLX evaluator...")
    try:
        # Use a smaller model for faster testing
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"

        # Create evaluator
        ppl_evaluator = select_ppl_evaluator("mlx", model_name)

        print(f"✓ MLX evaluator initialized with {model_name}")

        # Create search instance (ppl_fn first, then config)
        search = TrajectorySearch(ppl_evaluator, config)

        print("\nRunning trajectory search...")
        result = search.search(x, y)

        print("\n✅ Search complete!")

        # Get search history
        history = result.get("search_history", [])
        if history:
            orig_ppl = history[0].get("ppl", "N/A")
            final_ppl = result.get("ppl_final", "N/A")
            print(
                f"Original PPL: {orig_ppl:.2f}"
                if isinstance(orig_ppl, (int, float))
                else f"Original PPL: {orig_ppl}"
            )
            print(
                f"Optimized PPL: {final_ppl:.2f}"
                if isinstance(final_ppl, (int, float))
                else f"Optimized PPL: {final_ppl}"
            )

            # Calculate improvement
            if isinstance(orig_ppl, (int, float)) and isinstance(
                final_ppl, (int, float)
            ):
                improvement = (orig_ppl - final_ppl) / orig_ppl * 100
                print(f"Improvement: {improvement:.1f}%")

        # Get reasoning trajectory
        z_segments = result.get("z_segments", [])
        if z_segments:
            print(f"\nOptimized reasoning trajectory ({len(z_segments)} steps):")
            for i, seg in enumerate(z_segments, 1):
                print(f"  {i}. {seg}")
        else:
            print("\nNo reasoning trajectory found.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
