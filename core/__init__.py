"""REER × DSPy × MLX Social Posting Pack - Core Module"""

__version__ = "0.1.0"
__author__ = "REER Team"
__description__ = "Core functionality for social posting with DSPy and MLX"

# Core modules
from .exceptions import *
from .trace_store import REERTraceStore, TraceRecord
from .trajectory_synthesizer import (
    REERTrajectorySynthesizer,
    StrategySynthesis,
    StrategyPattern,
)
from .candidate_scorer import REERCandidateScorer, ContentCandidate, ScoringMetrics
from .trainer import REERGEPATrainer, OptimizationResult, Individual, Population

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
    # GEPA Trainer
    "REERGEPATrainer",
    "OptimizationResult",
    "Individual",
    "Population",
]
