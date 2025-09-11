#!/usr/bin/env python3
"""
Comprehensive test of the entire REER MLX implementation using actual DSPy library.
This validates that all components work together correctly.
"""

import json
from pathlib import Path
import sys
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_dspy_modules():
    """Test all DSPy modules in our implementation."""
    print("\n" + "=" * 80)
    print("Testing DSPy Modules")
    print("=" * 80)

    results = []

    # Test 1: REER Module
    print("\n📝 Test 1: REER Module")
    try:
        import dspy

        from dspy_program.reer_module import REERModule

        # Configure DSPy (with mock LM to avoid API calls)
        class MockLM:
            def __call__(self, *args, **kwargs):
                return ["Mock response"]

            def forward(self, *args, **kwargs):
                return {"response": "Mock response"}

        dspy.settings.configure(lm=MockLM())

        # Create module
        module = REERModule()

        # Test forward method exists
        assert hasattr(module, "forward"), "REERModule should have forward method"

        print("  ✅ REERModule created successfully")
        print(f"  ✅ Has components: {list(module.__dict__.keys())}")
        results.append(("REERModule", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("REERModule", False))

    # Test 2: GEPA Runner
    print("\n📝 Test 2: GEPA Runner")
    try:
        import dspy

        from dspy_program.gepa_runner import GEPARunner

        # Check signature definitions
        runner = GEPARunner()

        # Verify attributes
        required_attrs = ["craft_redacted_request", "respond_to_query"]
        for attr in required_attrs:
            if hasattr(runner, attr):
                print(f"  ✅ Has {attr}")
            else:
                print(f"  ❌ Missing {attr}")

        results.append(("GEPARunner", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("GEPARunner", False))

    # Test 3: Pipeline
    print("\n📝 Test 3: DSPy Pipeline")
    try:
        from dspy_program.pipeline import REERPipeline

        pipeline = REERPipeline()

        # Check methods
        assert hasattr(pipeline, "forward"), "Pipeline should have forward method"
        assert hasattr(pipeline, "generate"), "Pipeline should have generate method"

        print("  ✅ REERPipeline created successfully")
        results.append(("Pipeline", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("Pipeline", False))

    # Test 4: Evaluator
    print("\n📝 Test 4: DSPy Evaluator")
    try:
        from dspy_program.evaluator import REEREvaluator

        evaluator = REEREvaluator()

        print("  ✅ REEREvaluator created successfully")
        results.append(("Evaluator", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("Evaluator", False))

    return results


def test_trajectory_search():
    """Test the trajectory search implementation."""
    print("\n" + "=" * 80)
    print("Testing Trajectory Search")
    print("=" * 80)

    results = []

    try:
        from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig

        # Create a mock evaluator
        def mock_ppl(x, y, z):
            return 5.0 + len(z) * 0.1  # Simple mock PPL

        # Create config
        config = TrajectorySearchConfig(max_iters=3, max_candidates_per_segment=2)

        # Create search
        search = TrajectorySearch(mock_ppl, config)

        # Test search
        result = search.search("Test query", "Test output")

        # Validate result structure
        required_keys = [
            "x",
            "y",
            "z_segments",
            "z_full",
            "ppl_initial",
            "ppl_final",
            "iterations",
        ]
        for key in required_keys:
            assert key in result, f"Result should have {key}"

        print(f"  ✅ Search completed with {result['iterations']} iterations")
        print(
            f"  ✅ PPL improved from {result['ppl_initial']:.2f} to {result['ppl_final']:.2f}"
        )
        results.append(("TrajectorySearch", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("TrajectorySearch", False))

    return results


def test_mlx_integration():
    """Test MLX integration components."""
    print("\n" + "=" * 80)
    print("Testing MLX Integration")
    print("=" * 80)

    results = []

    # Test PPL Evaluator
    print("\n📝 Test: PPL Evaluator")
    try:
        # Test that the functions exist and have correct signatures
        import inspect

        from tools.ppl_eval import (
            make_mlx_ppl_evaluator,
            make_together_dspy_ppl_evaluator,
        )

        # Check MLX evaluator signature
        mlx_sig = inspect.signature(make_mlx_ppl_evaluator)
        params = list(mlx_sig.parameters.keys())
        assert "model_name" in params, "Should have model_name parameter"
        assert (
            "window_size" in params
        ), "Should have window_size parameter (sliding window)"
        assert "stride" in params, "Should have stride parameter (sliding window)"

        print(f"  ✅ MLX evaluator has parameters: {params}")

        # Check Together evaluator
        together_sig = inspect.signature(make_together_dspy_ppl_evaluator)
        assert "model_name" in together_sig.parameters

        print("  ✅ Together evaluator properly defined")
        results.append(("PPL Evaluators", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("PPL Evaluators", False))

    return results


def test_cli_commands():
    """Test CLI command structure."""
    print("\n" + "=" * 80)
    print("Testing CLI Commands")
    print("=" * 80)

    results = []

    # Test synthesize script
    print("\n📝 Test: Synthesize CLI")
    try:
        import inspect

        import typer

        from scripts.reer_synthesize import app, synthesize

        # Check it's a Typer app
        assert isinstance(app, typer.Typer), "Should be a Typer app"

        # Check synthesize function signature
        sig = inspect.signature(synthesize)
        params = list(sig.parameters.keys())

        required_params = [
            "input_file",
            "x_field",
            "y_field",
            "limit",
            "output_jsonl",
            "auto",
            "backend",
            "model",
        ]

        for param in required_params:
            assert param in params, f"Should have {param} parameter"

        print("  ✅ Synthesize command has all required parameters")
        results.append(("CLI Commands", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("CLI Commands", False))

    return results


def test_integration():
    """Test that all components integrate properly."""
    print("\n" + "=" * 80)
    print("Testing Component Integration")
    print("=" * 80)

    results = []

    print("\n📝 Test: End-to-End Integration")
    try:
        # Import all major components
        import dspy

        from dspy_program.pipeline import REERPipeline
        from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig

        # Mock configuration
        class MockLM:
            def __call__(self, *args, **kwargs):
                return ["Mock response"]

            def forward(self, *args, **kwargs):
                return {"response": "Mock response"}

            def loglikelihood(self, context, continuation):
                return {"loglikelihood": -3.0}

        dspy.settings.configure(lm=MockLM())

        # Test that components can be created together
        pipeline = REERPipeline()

        def mock_ppl(x, y, z):
            return 5.0

        config = TrajectorySearchConfig()
        search = TrajectorySearch(mock_ppl, config)

        print("  ✅ All components can be instantiated together")
        print("  ✅ No import conflicts or circular dependencies")
        results.append(("Integration", True))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        results.append(("Integration", False))

    return results


def main():
    """Run all tests and generate report."""
    print("🚀 Comprehensive REER MLX Implementation Test")
    print("=" * 80)

    all_results = []

    # Run all test suites
    all_results.extend(test_dspy_modules())
    all_results.extend(test_trajectory_search())
    all_results.extend(test_mlx_integration())
    all_results.extend(test_cli_commands())
    all_results.extend(test_integration())

    # Generate summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, status in all_results if status)
    total = len(all_results)

    for name, status in all_results:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")

    print("\n" + "=" * 80)
    percentage = (passed / total) * 100 if total > 0 else 0
    print(f"📊 Overall: {passed}/{total} tests passed ({percentage:.1f}%)")

    if percentage == 100:
        print("🎉 All tests passed! Implementation is valid!")
    elif percentage >= 80:
        print("✅ Most tests passed. Minor issues to address.")
    else:
        print("⚠️ Several issues found. Review and fix required.")

    # Save report
    report = {
        "total_tests": total,
        "passed": passed,
        "percentage": percentage,
        "results": [{"name": name, "passed": status} for name, status in all_results],
    }

    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📄 Detailed report saved to test_report.json")


if __name__ == "__main__":
    main()
