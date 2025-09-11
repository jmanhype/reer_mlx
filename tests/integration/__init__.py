"""Integration tests package for REER × DSPy × MLX Social Posting Pack.

This package contains comprehensive integration tests that verify end-to-end
workflows and component interactions. All tests follow TDD principles and
MUST fail initially (RED phase) since implementations don't exist yet.

Test Modules:
- test_data_collection: T009 - X analytics import → normalization pipeline
- test_reer_mining: T010 - REER strategy extraction workflow
- test_pipeline: T011 - Content generation pipeline integration
- test_provider_switching: T012 - Provider switching scenarios (mlx::/dspy::)
- test_gepa_tuning: T013 - GEPA optimization flow integration
"""

__all__ = [
    "test_data_collection",
    "test_reer_mining",
    "test_pipeline",
    "test_provider_switching",
    "test_gepa_tuning",
]
