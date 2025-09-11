#!/usr/bin/env python3
"""
Validate DSPy implementation by inspecting the actual DSPy library source code.
This script examines DSPy's actual API and validates our usage patterns.
"""

import inspect
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def inspect_dspy_library():  # noqa: PLR0912
    """Inspect the actual DSPy library to understand its structure."""
    try:
        import dspy

        logger.info("=" * 80)
        logger.info("DSPy Library Inspection Report")
        logger.info("=" * 80)

        # Get DSPy version and location
        logger.info(
            f"\n📦 DSPy Version: {dspy.__version__ if hasattr(dspy, '__version__') else 'Unknown'}"
        )
        logger.info(f"📁 Location: {dspy.__file__}")

        # Inspect main DSPy components
        logger.info("\n🔍 Main DSPy Components:")
        logger.info("-" * 40)

        main_components = []
        for name in dir(dspy):
            if not name.startswith("_"):
                obj = getattr(dspy, name)
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    main_components.append(
                        (name, type(obj).__name__, str(obj.__module__))
                    )

        for name, obj_type, module in sorted(main_components):
            logger.info(f"  • {name:30} [{obj_type:10}] from {module}")

        # Inspect DSPy Signatures
        logger.info("\n📝 DSPy Signature System:")
        logger.info("-" * 40)

        if hasattr(dspy, "Signature"):
            sig_class = dspy.Signature
            logger.info(f"  Base Class: {sig_class}")
            logger.info(f"  MRO: {[cls.__name__ for cls in sig_class.__mro__]}")

            # Check for InputField and OutputField
            if hasattr(dspy, "InputField"):
                logger.info(f"  ✓ InputField available: {dspy.InputField}")
            if hasattr(dspy, "OutputField"):
                logger.info(f"  ✓ OutputField available: {dspy.OutputField}")

        # Inspect DSPy Modules
        logger.info("\n🧩 DSPy Module Types:")
        logger.info("-" * 40)

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
                logger.info(f"  • {module_name}:")
                logger.info(f"    - Type: {type(module_class)}")

                # Get init signature if it's a class
                if inspect.isclass(module_class):
                    try:
                        init_sig = inspect.signature(module_class.__init__)
                        params = [p for p in init_sig.parameters if p != "self"]
                        logger.info(f"    - __init__ params: {params}")
                    except Exception as ex:
                        logger.debug(f"Unable to inspect __init__: {ex}")

                    # Check for forward method
                    if hasattr(module_class, "forward"):
                        logger.info("    - Has forward() method: ✓")

        # Inspect DSPy Optimizers
        logger.info("\n🎯 DSPy Optimizers:")
        logger.info("-" * 40)

        # Check for teleprompt/optimizers
        if hasattr(dspy, "teleprompt"):
            teleprompt = dspy.teleprompt
            for name in dir(teleprompt):
                if not name.startswith("_"):
                    obj = getattr(teleprompt, name)
                    if inspect.isclass(obj):
                        logger.info(f"  • {name}")

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
                logger.info(f"  • {opt_name} - Available at dspy.{opt_name}")

        # Inspect DSPy LM support
        logger.info("\n🤖 DSPy Language Model Support:")
        logger.info("-" * 40)

        if hasattr(dspy, "LM"):
            lm_class = dspy.LM
            logger.info(f"  LM Class: {lm_class}")

            # Check LM methods
            lm_methods = [m for m in dir(lm_class) if not m.startswith("_")]
            logger.info(f"  Available methods: {', '.join(lm_methods[:10])}")

        # Check for important functions
        logger.info("\n🔧 DSPy Utility Functions:")
        logger.info("-" * 40)

        utilities = ["configure", "settings", "Example", "Prediction", "Evaluate"]
        for util in utilities:
            if hasattr(dspy, util):
                obj = getattr(dspy, util)
                logger.info(f"  • {util}: {type(obj).__name__}")
    except ImportError:
        logger.exception("❌ Error importing DSPy")
        return False
    except Exception:
        logger.exception("❌ Error inspecting DSPy")
        return False
    else:
        return True


def validate_our_dspy_usage():  # noqa: PLR0912
    """Validate our DSPy implementation against the actual library."""
    logger.info("\n" + "=" * 80)
    logger.info("Validating Our DSPy Implementation")
    logger.info("=" * 80)

    import dspy

    # Track validation results
    validations = []

    # Test 1: Signature creation
    logger.info("\n✅ Test 1: DSPy Signature Creation")
    try:

        class TestSignature(dspy.Signature):
            """Test signature."""

            query = dspy.InputField()
            response = dspy.OutputField()

        validations.append(
            ("Signature with class", True, "✓ Can create Signature classes")
        )

        # Test string signature
        _ = dspy.Predict("question -> answer")
        validations.append(("String signature", True, "✓ Can create string signatures"))
    except Exception as e:
        validations.append(("Signature creation", False, f"✗ Error: {e}"))

    # Test 2: Module creation
    logger.info("\n✅ Test 2: DSPy Module Creation")
    try:

        class TestModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")
                self.cot = dspy.ChainOfThought("question -> answer")

            def forward(self, question):
                return self.predict(question=question)

        _ = TestModule()
        validations.append(("Module creation", True, "✓ Can create DSPy modules"))
    except Exception as e:
        validations.append(("Module creation", False, f"✗ Error: {e}"))

    # Test 3: Check our imports
    logger.info("\n✅ Test 3: Checking Our Import Patterns")

    # Read our DSPy files
    dspy_files = list(Path(project_root / "dspy_program").glob("*.py"))

    for file_path in dspy_files:
        if file_path.name == "__init__.py":
            continue

        logger.info(f"\n  Checking: {file_path.name}")
        with open(file_path) as f:
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

        for pattern, _description in patterns.items():
            if pattern in content:
                # Verify this actually exists in DSPy
                parts = pattern.split(".")
                try:
                    obj = dspy
                    for part in parts[1:]:  # Skip 'dspy'
                        obj = getattr(obj, part)
                    logger.info(f"    ✓ {pattern} - Valid")
                except AttributeError:
                    issues.append(f"    ✗ {pattern} - Not found in DSPy!")
                    logger.warning(f"    ✗ {pattern} - NOT FOUND in actual DSPy!")

        if issues:
            validations.append((f"{file_path.name} imports", False, "\n".join(issues)))
        else:
            validations.append(
                (f"{file_path.name} imports", True, "✓ All imports valid")
            )

    # Test 4: GEPA Optimizer
    logger.info("\n✅ Test 4: GEPA Optimizer Validation")
    try:
        if hasattr(dspy, "GEPA"):
            gepa_sig = inspect.signature(dspy.GEPA.__init__)
            params = [p for p in gepa_sig.parameters if p != "self"]
            logger.info(f"  GEPA __init__ parameters: {params}")
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
    logger.info("\n✅ Test 5: Testing Actual DSPy Patterns")

    # Test configuration
    try:
        # Note: We don't actually configure to avoid API calls
        logger.info("  Testing dspy.settings.configure pattern...")
        if hasattr(dspy.settings, "configure"):
            sig = inspect.signature(dspy.settings.configure)
            logger.info(f"    configure() params: {list(sig.parameters.keys())}")
            validations.append(
                ("settings.configure", True, "✓ Configuration method available")
            )
    except Exception as e:
        validations.append(("settings.configure", False, f"✗ Error: {e}"))

    # Print validation summary
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)

    passed = sum(1 for _, status, _ in validations if status)
    total = len(validations)

    for name, status, message in validations:
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {name}: {message}")

    logger.info(
        f"\n📊 Results: {passed}/{total} validations passed ({passed * 100 // total}%)"
    )

    return passed == total


def check_dspy_lm_compatibility():
    """Check DSPy's LM interface for loglikelihood support."""
    logger.info("\n" + "=" * 80)
    logger.info("DSPy LM Interface Analysis")
    logger.info("=" * 80)

    try:
        import dspy

        if hasattr(dspy, "LM"):
            lm_class = dspy.LM

            logger.info("\n🔍 DSPy.LM Methods:")
            methods = [m for m in dir(lm_class) if not m.startswith("_")]
            for method in sorted(methods):
                method_obj = getattr(lm_class, method)
                if callable(method_obj):
                    try:
                        sig = inspect.signature(method_obj)
                        logger.info(f"  • {method}{sig}")
                    except Exception:
                        logger.info(f"  • {method}()")

            # Check specifically for loglikelihood
            if hasattr(lm_class, "loglikelihood"):
                logger.info("\n✅ loglikelihood method found!")
                sig = inspect.signature(lm_class.loglikelihood)
                logger.info(f"  Signature: {sig}")
            else:
                logger.warning("\n⚠️ No loglikelihood method in dspy.LM")
                logger.info("  Our Together backend validation is correct!")

    except Exception:
        logger.exception("Error analyzing LM")


def analyze_our_gepa_usage():
    """Analyze our GEPA runner implementation."""
    logger.info("\n" + "=" * 80)
    logger.info("GEPA Runner Analysis")
    logger.info("=" * 80)

    gepa_file = project_root / "dspy_program" / "gepa_runner.py"

    if gepa_file.exists():
        with open(gepa_file) as f:
            content = f.read()

        # Extract GEPA usage
        if "GEPA(" in content:
            logger.info("✅ GEPA initialization found")

            # Check for required parameters
            required_params = ["metric", "num_threads", "track_stats"]
            for param in required_params:
                if f"{param}=" in content:
                    logger.info(f"  ✓ {param} parameter set")
                else:
                    logger.warning(f"  ⚠️ {param} parameter might be missing")

        # Check compile method usage
        if ".compile(" in content:
            logger.info("\n✅ GEPA compile() method used")
            if "student=" in content and "trainset=" in content:
                logger.info("  ✓ Required compile parameters present")


if __name__ == "__main__":
    # Run all validations
    logger.info("🚀 Starting DSPy Implementation Validation\n")

    # 1. Inspect DSPy library
    if inspect_dspy_library():
        # 2. Validate our usage
        validate_our_dspy_usage()

        # 3. Check LM compatibility
        check_dspy_lm_compatibility()

        # 4. Analyze GEPA usage
        analyze_our_gepa_usage()

    logger.info("\n✨ Validation Complete!")
