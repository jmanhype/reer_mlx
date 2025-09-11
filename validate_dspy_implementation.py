#!/usr/bin/env python3
"""
Validate DSPy implementation by inspecting the actual DSPy library source code.
This script examines DSPy's actual API and validates our usage patterns.
"""

import inspect
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def inspect_dspy_library():
    """Inspect the actual DSPy library to understand its structure."""
    try:
        import dspy

        print("=" * 80)
        print("DSPy Library Inspection Report")
        print("=" * 80)

        # Get DSPy version and location
        print(
            f"\n📦 DSPy Version: {dspy.__version__ if hasattr(dspy, '__version__') else 'Unknown'}"
        )
        print(f"📁 Location: {dspy.__file__}")

        # Inspect main DSPy components
        print("\n🔍 Main DSPy Components:")
        print("-" * 40)

        main_components = []
        for name in dir(dspy):
            if not name.startswith("_"):
                obj = getattr(dspy, name)
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    main_components.append(
                        (name, type(obj).__name__, str(obj.__module__))
                    )

        for name, obj_type, module in sorted(main_components):
            print(f"  • {name:30} [{obj_type:10}] from {module}")

        # Inspect DSPy Signatures
        print("\n📝 DSPy Signature System:")
        print("-" * 40)

        if hasattr(dspy, "Signature"):
            sig_class = dspy.Signature
            print(f"  Base Class: {sig_class}")
            print(f"  MRO: {[cls.__name__ for cls in sig_class.__mro__]}")

            # Check for InputField and OutputField
            if hasattr(dspy, "InputField"):
                print(f"  ✓ InputField available: {dspy.InputField}")
            if hasattr(dspy, "OutputField"):
                print(f"  ✓ OutputField available: {dspy.OutputField}")

        # Inspect DSPy Modules
        print("\n🧩 DSPy Module Types:")
        print("-" * 40)

        module_types = [
            "Module",
            "Predict",
            "ChainOfThought",
            "ProgramOfThought",
            "ReAct",
        ]
        for module_name in module_types:
            if hasattr(dspy, module_name):
                module_class = getattr(dspy, module_name)
                print(f"  • {module_name}:")
                print(f"    - Type: {type(module_class)}")

                # Get init signature if it's a class
                if inspect.isclass(module_class):
                    try:
                        init_sig = inspect.signature(module_class.__init__)
                        params = [p for p in init_sig.parameters.keys() if p != "self"]
                        print(f"    - __init__ params: {params}")
                    except:
                        pass

                    # Check for forward method
                    if hasattr(module_class, "forward"):
                        print(f"    - Has forward() method: ✓")

        # Inspect DSPy Optimizers
        print("\n🎯 DSPy Optimizers:")
        print("-" * 40)

        # Check for teleprompt/optimizers
        if hasattr(dspy, "teleprompt"):
            teleprompt = dspy.teleprompt
            for name in dir(teleprompt):
                if not name.startswith("_"):
                    obj = getattr(teleprompt, name)
                    if inspect.isclass(obj):
                        print(f"  • {name}")

        # Direct optimizer checks
        optimizer_names = [
            "GEPA",
            "MIPROv2",
            "BootstrapFewShot",
            "BootstrapFinetune",
            "COPRO",
        ]
        for opt_name in optimizer_names:
            if hasattr(dspy, opt_name):
                print(f"  • {opt_name} - Available at dspy.{opt_name}")

        # Inspect DSPy LM support
        print("\n🤖 DSPy Language Model Support:")
        print("-" * 40)

        if hasattr(dspy, "LM"):
            lm_class = dspy.LM
            print(f"  LM Class: {lm_class}")

            # Check LM methods
            lm_methods = [m for m in dir(lm_class) if not m.startswith("_")]
            print(f"  Available methods: {', '.join(lm_methods[:10])}")

        # Check for important functions
        print("\n🔧 DSPy Utility Functions:")
        print("-" * 40)

        utilities = ["configure", "settings", "Example", "Prediction", "Evaluate"]
        for util in utilities:
            if hasattr(dspy, util):
                obj = getattr(dspy, util)
                print(f"  • {util}: {type(obj).__name__}")

        return True

    except ImportError as e:
        print(f"❌ Error importing DSPy: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inspecting DSPy: {e}")
        return False


def validate_our_dspy_usage():
    """Validate our DSPy implementation against the actual library."""
    print("\n" + "=" * 80)
    print("Validating Our DSPy Implementation")
    print("=" * 80)

    import dspy

    # Track validation results
    validations = []

    # Test 1: Signature creation
    print("\n✅ Test 1: DSPy Signature Creation")
    try:

        class TestSignature(dspy.Signature):
            """Test signature."""

            query = dspy.InputField()
            response = dspy.OutputField()

        validations.append(
            ("Signature with class", True, "✓ Can create Signature classes")
        )

        # Test string signature
        test_sig = dspy.Predict("question -> answer")
        validations.append(("String signature", True, "✓ Can create string signatures"))
    except Exception as e:
        validations.append(("Signature creation", False, f"✗ Error: {e}"))

    # Test 2: Module creation
    print("\n✅ Test 2: DSPy Module Creation")
    try:

        class TestModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")
                self.cot = dspy.ChainOfThought("question -> answer")

            def forward(self, question):
                return self.predict(question=question)

        module = TestModule()
        validations.append(("Module creation", True, "✓ Can create DSPy modules"))
    except Exception as e:
        validations.append(("Module creation", False, f"✗ Error: {e}"))

    # Test 3: Check our imports
    print("\n✅ Test 3: Checking Our Import Patterns")

    # Read our DSPy files
    dspy_files = list(Path(project_root / "dspy_program").glob("*.py"))

    for file_path in dspy_files:
        if file_path.name == "__init__.py":
            continue

        print(f"\n  Checking: {file_path.name}")
        with open(file_path, "r") as f:
            content = f.read()

        # Check imports
        issues = []

        # Common patterns to check
        patterns = {
            "dspy.Signature": "Signature base class",
            "dspy.Module": "Module base class",
            "dspy.InputField": "Input field declaration",
            "dspy.OutputField": "Output field declaration",
            "dspy.Predict": "Basic prediction module",
            "dspy.ChainOfThought": "CoT module",
            "dspy.settings.configure": "Settings configuration",
            "dspy.LM": "Language model class",
        }

        for pattern, description in patterns.items():
            if pattern in content:
                # Verify this actually exists in DSPy
                parts = pattern.split(".")
                try:
                    obj = dspy
                    for part in parts[1:]:  # Skip 'dspy'
                        obj = getattr(obj, part)
                    print(f"    ✓ {pattern} - Valid")
                except AttributeError:
                    issues.append(f"    ✗ {pattern} - Not found in DSPy!")
                    print(f"    ✗ {pattern} - NOT FOUND in actual DSPy!")

        if issues:
            validations.append((f"{file_path.name} imports", False, "\n".join(issues)))
        else:
            validations.append(
                (f"{file_path.name} imports", True, "✓ All imports valid")
            )

    # Test 4: GEPA Optimizer
    print("\n✅ Test 4: GEPA Optimizer Validation")
    try:
        if hasattr(dspy, "GEPA"):
            gepa_sig = inspect.signature(dspy.GEPA.__init__)
            params = [p for p in gepa_sig.parameters.keys() if p != "self"]
            print(f"  GEPA __init__ parameters: {params}")
            validations.append(
                (
                    "GEPA optimizer",
                    True,
                    f"✓ GEPA available with params: {params[:5]}...",
                )
            )
        else:
            validations.append(
                ("GEPA optimizer", False, "✗ GEPA not found in dspy namespace")
            )
    except Exception as e:
        validations.append(("GEPA optimizer", False, f"✗ Error: {e}"))

    # Test 5: Validate actual usage
    print("\n✅ Test 5: Testing Actual DSPy Patterns")

    # Test configuration
    try:
        # Note: We don't actually configure to avoid API calls
        print("  Testing dspy.settings.configure pattern...")
        if hasattr(dspy.settings, "configure"):
            sig = inspect.signature(dspy.settings.configure)
            print(f"    configure() params: {list(sig.parameters.keys())}")
            validations.append(
                ("settings.configure", True, "✓ Configuration method available")
            )
    except Exception as e:
        validations.append(("settings.configure", False, f"✗ Error: {e}"))

    # Print validation summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)

    passed = sum(1 for _, status, _ in validations if status)
    total = len(validations)

    for name, status, message in validations:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}: {message}")

    print(f"\n📊 Results: {passed}/{total} validations passed ({passed*100//total}%)")

    return passed == total


def check_dspy_lm_compatibility():
    """Check DSPy's LM interface for loglikelihood support."""
    print("\n" + "=" * 80)
    print("DSPy LM Interface Analysis")
    print("=" * 80)

    try:
        import dspy

        if hasattr(dspy, "LM"):
            lm_class = dspy.LM

            print("\n🔍 DSPy.LM Methods:")
            methods = [m for m in dir(lm_class) if not m.startswith("_")]
            for method in sorted(methods):
                method_obj = getattr(lm_class, method)
                if callable(method_obj):
                    try:
                        sig = inspect.signature(method_obj)
                        print(f"  • {method}{sig}")
                    except:
                        print(f"  • {method}()")

            # Check specifically for loglikelihood
            if hasattr(lm_class, "loglikelihood"):
                print("\n✅ loglikelihood method found!")
                sig = inspect.signature(lm_class.loglikelihood)
                print(f"  Signature: {sig}")
            else:
                print("\n⚠️ No loglikelihood method in dspy.LM")
                print("  Our Together backend validation is correct!")

    except Exception as e:
        print(f"Error analyzing LM: {e}")


def analyze_our_gepa_usage():
    """Analyze our GEPA runner implementation."""
    print("\n" + "=" * 80)
    print("GEPA Runner Analysis")
    print("=" * 80)

    gepa_file = project_root / "dspy_program" / "gepa_runner.py"

    if gepa_file.exists():
        with open(gepa_file, "r") as f:
            content = f.read()

        # Extract GEPA usage
        if "GEPA(" in content:
            print("✅ GEPA initialization found")

            # Check for required parameters
            required_params = ["metric", "num_threads", "track_stats"]
            for param in required_params:
                if f"{param}=" in content:
                    print(f"  ✓ {param} parameter set")
                else:
                    print(f"  ⚠️ {param} parameter might be missing")

        # Check compile method usage
        if ".compile(" in content:
            print("\n✅ GEPA compile() method used")
            if "student=" in content and "trainset=" in content:
                print("  ✓ Required compile parameters present")


if __name__ == "__main__":
    # Run all validations
    print("🚀 Starting DSPy Implementation Validation\n")

    # 1. Inspect DSPy library
    if inspect_dspy_library():
        # 2. Validate our usage
        validate_our_dspy_usage()

        # 3. Check LM compatibility
        check_dspy_lm_compatibility()

        # 4. Analyze GEPA usage
        analyze_our_gepa_usage()

    print("\n✨ Validation Complete!")
