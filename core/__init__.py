"""REER × DSPy × MLX Social Posting Pack - Core Module"""

__version__ = "0.1.0"
__author__ = "REER Team"
__description__ = "Core functionality for social posting with DSPy and MLX"

# Core modules
from .candidate_scorer import ContentCandidate, REERCandidateScorer, ScoringMetrics
from .exceptions import *
from .trace_store import REERTraceStore, TraceRecord
from .trainer import Individual, OptimizationResult, Population, REERGEPATrainer
from .trajectory_synthesizer import (
    REERTrajectorySynthesizer,
    StrategyPattern,
    StrategySynthesis,
)

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
