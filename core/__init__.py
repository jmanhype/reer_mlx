"""REER × DSPy × MLX Social Posting Pack - Core Module"""

# Core modules (sorted imports)
from .candidate_scorer import ContentCandidate, REERCandidateScorer, ScoringMetrics
from .exceptions import (
    ConvergenceError,
    ExtractionError,
    FitnessError,
    ImportError,
    NormalizationError,
    OptimizationError,
    PatternAnalysisError,
    PerplexityError,
    REERBaseException,
    ScoringError,
    StorageError,
    StrategyError,
    SynthesisError,
    TraceStoreError,
    TrainingError,
    TrajectoryError,
    ValidationError,
)
from .trace_store import REERTraceStore, TraceRecord
from .trajectory_synthesizer import (
    REERTrajectorySynthesizer,
    StrategyPattern,
    StrategySynthesis,
)

__version__ = "0.1.0"
__author__ = "REER Team"
__description__ = "Core functionality for social posting with DSPy and MLX"

__all__ = [
    # Exceptions
    "REERBaseException",
    "StorageError",
    "TraceStoreError",
    "ImportError",
    "NormalizationError",
    "ValidationError",
    "ExtractionError",
    "PatternAnalysisError",
    "SynthesisError",
    "OptimizationError",
    "ConvergenceError",
    "FitnessError",
    "TrainingError",
    "ScoringError",
    "PerplexityError",
    "TrajectoryError",
    "StrategyError",
    # Trace Store
    "REERTraceStore",
    "TraceRecord",
    # Trajectory Synthesizer
    "REERTrajectorySynthesizer",
    "StrategySynthesis",
    "StrategyPattern",
    # Candidate Scorer
    "REERCandidateScorer",
    "ContentCandidate",
    "ScoringMetrics",
]


# GEPA trainer compatibility shim (deprecated)
class REERGEPATrainer:  # pragma: no cover - import-time shim
    def __init__(self, *_, **__):
        raise ImportError(
            "REERGEPATrainer has been removed. Use dspy_program.gepa_runner.run_gepa instead."
        )


# Backwards-compat names removed: Individual, Population, OptimizationResult
