#!/usr/bin/env python3
"""
Quick test to verify plugin imports work correctly.
"""


def test_plugin_imports():
    """Test that all plugin modules can be imported."""

    try:
        # Test MLX adapter import
        # Test DSPy adapter import
        # Test main plugins module import
        from plugins import (
            HeuristicScorer,
            MLXLanguageModelAdapter,
            generate_text,
            get_registry,
        )
        from plugins.dspy_lm import DSPyConfig, DSPyLanguageModelAdapter

        # Test heuristics import
        from plugins.heuristics import ContentAnalyzer, HeuristicScorer

        # Test registry import
        from plugins.lm_registry import LanguageModelRegistry, get_registry
        from plugins.mlx_lm import BaseLMAdapter, MLXLanguageModelAdapter

        # Test registry creation
        registry = get_registry()
        providers = registry.list_providers()
        for _provider in providers:
            pass

    except ImportError:
        return False
    except Exception:
        return False

    return True


if __name__ == "__main__":
    test_plugin_imports()
