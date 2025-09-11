"""Core exceptions for REER × DSPy × MLX Social Posting Pack.

Custom exceptions for domain-specific error handling with type safety
and detailed error context for debugging and monitoring.
"""

from typing import Any, Dict, Optional


class REERBaseException(Exception):
    """Base exception for all REER-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Storage and persistence errors
class StorageError(REERBaseException):
    """Error in storage operations (JSONL, database, etc.)."""

    pass


class TraceStoreError(StorageError):
    """Error in trace store operations."""

    pass


# Data processing errors
class ImportError(REERBaseException):
    """Error during data import operations."""

    pass


class NormalizationError(REERBaseException):
    """Error during data normalization."""

    pass


class ValidationError(REERBaseException):
    """Error during data validation."""

    pass


# Mining and extraction errors
class ExtractionError(REERBaseException):
    """Error during strategy extraction."""

    pass


class PatternAnalysisError(REERBaseException):
    """Error during pattern analysis."""

    pass


class SynthesisError(REERBaseException):
    """Error during strategy synthesis."""

    pass


# Optimization and training errors
class OptimizationError(REERBaseException):
    """Error during GEPA optimization."""

    pass


class ConvergenceError(OptimizationError):
    """Error related to convergence in optimization."""

    pass


class FitnessError(OptimizationError):
    """Error during fitness evaluation."""

    pass


class TrainingError(REERBaseException):
    """Error during model training."""

    pass


# Scoring and evaluation errors
class ScoringError(REERBaseException):
    """Error during candidate scoring."""

    pass


class PerplexityError(ScoringError):
    """Error during perplexity calculation."""

    pass


# Trajectory and synthesis errors
class TrajectoryError(REERBaseException):
    """Error during trajectory synthesis."""

    pass


class StrategyError(REERBaseException):
    """Error in strategy operations."""

    pass
