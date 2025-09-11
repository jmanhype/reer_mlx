#!/usr/bin/env python3
"""Test script for Together backend validation and error handling."""

from pathlib import Path
import sys
from unittest.mock import Mock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.ppl_eval import make_together_dspy_ppl_evaluator


def test_together_without_loglikelihood():
    """Test that Together backend fails fast without loglikelihood support."""

    print("Testing Together Backend Validation")
    print("=" * 50)

    # Mock dspy.LM without loglikelihood
    import dspy

    original_lm = dspy.LM if hasattr(dspy, "LM") else None

    try:
        # Create mock LM without loglikelihood
        mock_lm = Mock()
        # Remove loglikelihood attribute if it exists
        if hasattr(mock_lm, "loglikelihood"):
            delattr(mock_lm, "loglikelihood")

        # Replace dspy.LM temporarily
        dspy.LM = lambda model: mock_lm

        print("Testing model without loglikelihood support...")
        try:
            ppl_fn = make_together_dspy_ppl_evaluator("meta-llama/Llama-2-7b-hf")
            print("  ❌ Should have raised RuntimeError!")
            return False
        except RuntimeError as e:
            print(f"  ✓ Correctly raised error:\n    {str(e).split(chr(10))[0]}...")
            assert "does not expose loglikelihood method" in str(e)
            assert "Switch to MLX backend" in str(e)
            print("  ✓ Error message includes helpful suggestions")

    finally:
        # Restore original
        if original_lm:
            dspy.LM = original_lm

    return True


def test_together_with_loglikelihood():
    """Test that Together backend works with loglikelihood support."""

    print("\nTesting Together Backend with Loglikelihood")
    print("=" * 50)

    import dspy

    original_lm = dspy.LM if hasattr(dspy, "LM") else None

    try:
        # Create mock LM with loglikelihood
        mock_lm = Mock()

        # Add loglikelihood method that returns expected format
        def mock_loglikelihood(context, continuation):
            # Simulate realistic response
            return {"loglikelihood": -5.0}

        mock_lm.loglikelihood = mock_loglikelihood

        # Replace dspy.LM temporarily
        dspy.LM = lambda model: mock_lm

        print("Testing model with loglikelihood support...")
        try:
            ppl_fn = make_together_dspy_ppl_evaluator("meta-llama/Llama-2-7b-hf")
            print("  ✓ Evaluator created successfully")

            # Test PPL computation
            ppl = ppl_fn("Context", "Output", "Additional")
            print(f"  ✓ PPL computed: {ppl:.3f}")

            # Test empty y case
            ppl_empty = ppl_fn("Context", "", "Additional")
            assert ppl_empty == 10.0
            print("  ✓ Empty y returns 10.0")

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False

    finally:
        # Restore original
        if original_lm:
            dspy.LM = original_lm

    return True


def test_unexpected_response_format():
    """Test handling of unexpected response formats from loglikelihood."""

    print("\nTesting Unexpected Response Format Handling")
    print("=" * 50)

    import dspy

    original_lm = dspy.LM if hasattr(dspy, "LM") else None

    try:
        # Test various response formats
        test_cases = [
            ("dict format", {"loglikelihood": -3.0}, True),
            ("float format", -3.0, True),
            ("int format", -3, True),
            ("wrong dict key", {"perplexity": 5.0}, False),
            ("string format", "-3.0", False),
            ("None", None, False),
        ]

        for name, response, should_work in test_cases:
            mock_lm = Mock()
            mock_lm.loglikelihood = lambda context, continuation: response
            dspy.LM = lambda model: mock_lm

            print(f"  Testing {name}: {response}")

            try:
                ppl_fn = make_together_dspy_ppl_evaluator("test-model")
                result = ppl_fn("x", "y", "z")

                if should_work:
                    print(f"    ✓ Worked as expected, PPL = {result:.3f}")
                else:
                    print(f"    ❌ Should have failed but got: {result}")
                    return False

            except RuntimeError as e:
                if not should_work:
                    print(
                        f"    ✓ Failed as expected: {str(e).split(chr(10))[0][:50]}..."
                    )
                else:
                    print(f"    ❌ Should have worked but failed: {e}")
                    return False

    finally:
        if original_lm:
            dspy.LM = original_lm

    return True


def test_runtime_attribute_loss():
    """Test handling when loglikelihood disappears at runtime."""

    print("\nTesting Runtime Attribute Loss")
    print("=" * 50)

    import dspy

    original_lm = dspy.LM if hasattr(dspy, "LM") else None

    try:
        # Create mock that loses loglikelihood after init
        mock_lm = Mock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call succeeds
                return {"loglikelihood": -3.0}
            # Subsequent calls fail
            raise AttributeError("loglikelihood")

        mock_lm.loglikelihood = Mock(side_effect=side_effect)
        dspy.LM = lambda model: mock_lm

        print("Testing attribute loss after initialization...")
        ppl_fn = make_together_dspy_ppl_evaluator("test-model")
        print("  ✓ Initialization succeeded")

        # First call should work
        result1 = ppl_fn("x", "y", "z")
        print(f"  ✓ First call worked: PPL = {result1:.3f}")

        # Second call should handle the error gracefully
        try:
            result2 = ppl_fn("x", "y", "z")
            print(f"  ❌ Should have raised error but got: {result2}")
            return False
        except RuntimeError as e:
            print(
                f"  ✓ Correctly handled runtime loss: {str(e).split(chr(10))[0][:50]}..."
            )
            assert "lost loglikelihood capability" in str(e)

    finally:
        if original_lm:
            dspy.LM = original_lm

    return True


if __name__ == "__main__":
    all_passed = True

    tests = [
        test_together_without_loglikelihood,
        test_together_with_loglikelihood,
        test_unexpected_response_format,
        test_runtime_attribute_loss,
    ]

    for test in tests:
        if not test():
            all_passed = False
            print(f"\n❌ Test {test.__name__} failed!")

    if all_passed:
        print("\n" + "=" * 50)
        print("✅ All Together backend validation tests passed!")
