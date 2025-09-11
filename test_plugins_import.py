#!/usr/bin/env python3
"""
Quick test to verify plugin imports work correctly.
"""


def test_plugin_imports():
    """Test that all plugin modules can be imported."""

    print("Testing plugin imports...")

    try:
        # Test MLX adapter import
        from plugins.mlx_lm import MLXLanguageModelAdapter, BaseLMAdapter

        print("✅ MLX adapter imported successfully")

        # Test DSPy adapter import
        from plugins.dspy_lm import DSPyLanguageModelAdapter, DSPyConfig

        print("✅ DSPy adapter imported successfully")

        # Test heuristics import
        from plugins.heuristics import HeuristicScorer, ContentAnalyzer

        print("✅ Heuristics module imported successfully")

        # Test registry import
        from plugins.lm_registry import LanguageModelRegistry, get_registry

        print("✅ LM registry imported successfully")

        # Test main plugins module import
        from plugins import (
            get_registry,
            generate_text,
            HeuristicScorer,
            MLXLanguageModelAdapter,
        )

        print("✅ Main plugins module imported successfully")

        # Test registry creation
        registry = get_registry()
        providers = registry.list_providers()
        print(f"✅ Registry created with {len(providers)} providers:")
        for provider in providers:
            print(f"   - {provider.name} ({provider.scheme}://)")

        print("\n🎉 All imports successful! Plugin system is ready.")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_plugin_imports()
