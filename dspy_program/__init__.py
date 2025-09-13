"""REER × DSPy × MLX - Core DSPy Integration

This module implements REER (Reverse-Engineered Reasoning) with DSPy and MLX integration
for perplexity-guided reasoning optimization.

Core Components:
    reer_module: REER refinement processor and search
    gepa_runner: GEPA optimization with DSPy
    mlx_variants: Direct MLX variant generation
    mlx_server_config: MLX server configuration for DSPy
"""

from .reer_module import (
    REERConfig,
    REERRefinementProcessor,
    REERSearchModule,
    REERSearchResult,
    SearchStrategy,
)

from .gepa_runner import (
    run_gepa,
    build_examples_from_traces,
)

__all__ = [
    # REER core
    "REERConfig",
    "REERRefinementProcessor",
    "REERSearchModule",
    "REERSearchResult",
    "SearchStrategy",
    # GEPA optimization
    "run_gepa",
    "build_examples_from_traces",
]
