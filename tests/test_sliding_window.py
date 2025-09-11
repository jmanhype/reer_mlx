#!/usr/bin/env python3
"""Test script for MLX sliding window functionality."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.ppl_eval import make_mlx_ppl_evaluator


def test_sliding_window():
    """Test MLX evaluator with long contexts requiring sliding windows."""

    print("Testing MLX Sliding Window Support")
    print("=" * 50)

    # Create a long context that exceeds typical window size
    x = "The quick brown fox jumps over the lazy dog. " * 50  # ~350 tokens
    z = (
        "This is additional context that provides more information. " * 100
    )  # ~700 tokens
    y = "The final answer is 42."  # ~6 tokens

    print(f"Context sizes (approx):")
    print(f"  x: ~{len(x.split())} words")
    print(f"  z: ~{len(z.split())} words")
    print(f"  y: ~{len(y.split())} words")
    print(f"  Total: ~{len((x + z + y).split())} words\n")

    try:
        # Test with different window sizes
        for window_size in [512, 1024, 2048]:
            print(f"Testing window_size={window_size}:")

            # Initialize evaluator with specific window size
            model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            ppl_fn = make_mlx_ppl_evaluator(
                model_name, window_size=window_size, stride=window_size // 2
            )

            # Compute PPL
            ppl = ppl_fn(x, y, z)
            print(f"  PPL(y|x,z) = {ppl:.3f}")

            # Verify it handles edge cases
            ppl_empty = ppl_fn("", y, "")
            print(f"  PPL(y|empty) = {ppl_empty:.3f}")

            ppl_no_y = ppl_fn(x, "", z)
            print(f"  PPL(empty|x,z) = {ppl_no_y:.3f} (should be 10.0)")

            print()

    except Exception as e:
        print(f"Error during testing: {e}")
        return False

    print("✓ All tests passed!")
    return True


def test_very_long_context():
    """Test with extremely long context to stress sliding window."""

    print("\nTesting Very Long Context")
    print("=" * 50)

    # Create an extremely long context
    x = " ".join(
        [f"Sentence {i} provides context." for i in range(500)]
    )  # ~2000 tokens
    z = " ".join([f"Additional info {i}." for i in range(500)])  # ~1500 tokens
    y = "The conclusion based on all the above context."  # ~10 tokens

    print(f"Very long context sizes (approx):")
    print(f"  x: ~{len(x.split())} words")
    print(f"  z: ~{len(z.split())} words")
    print(f"  y: ~{len(y.split())} words")
    print(f"  Total: ~{len((x + z + y).split())} words\n")

    try:
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"

        # Test with small window to force multiple sliding windows
        ppl_fn = make_mlx_ppl_evaluator(
            model_name, window_size=512, stride=256  # Small window  # 50% overlap
        )

        ppl = ppl_fn(x, y, z)
        print(f"PPL with 512 token window: {ppl:.3f}")

        # Compare with larger window
        ppl_fn_large = make_mlx_ppl_evaluator(model_name, window_size=2048, stride=1024)

        ppl_large = ppl_fn_large(x, y, z)
        print(f"PPL with 2048 token window: {ppl_large:.3f}")

        # Results should be similar but not identical due to windowing
        print(f"\nDifference: {abs(ppl - ppl_large):.3f}")
        print("(Small differences expected due to windowing strategy)")

    except Exception as e:
        print(f"Error during very long context test: {e}")
        return False

    print("\n✓ Very long context test passed!")
    return True


if __name__ == "__main__":
    success = test_sliding_window()
    if success:
        test_very_long_context()
