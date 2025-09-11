"""REER × DSPy × MLX Social Posting Pack - DSPy Program Module

This module implements the core DSPy pipeline orchestration for social media content
generation and optimization, integrating REER search, content scoring, and GEPA training.

Modules:
    pipeline: Main DSPy pipeline orchestrator (T022)
    reer_module: REER search wrapper with DSPy integration (T023)
    evaluator: KPI evaluator for performance metrics (T024)
"""

from .evaluator import (
    BenchmarkData,
    BenchmarkManager,
    KPICalculator,
    KPIEvaluator,
    MetricDefinition,
    MetricLevel,
    MetricResult,
    MetricType,
    PerformanceMetrics,
)
from .pipeline import (
    ContentGenerationSignature,
    ContentGeneratorModule,
    ContentRefinementSignature,
    ContentRequest,
    ContentResult,
    PipelineConfig,
    PipelineFactory,
    PipelineResult,
    REERDSPyPipeline,
)
from .reer_module import (
    ContentAnalysisModule,
    MockREERSearchEngine,
    QueryEnhancementModule,
    REERSearchModule,
    REERSearchResult,
    SearchContext,
    SearchResult,
    SearchStrategy,
)

__all__ = [
    # Pipeline components
    "REERDSPyPipeline",
    "PipelineConfig",
    "ContentRequest",
    "ContentResult",
    "PipelineResult",
    "PipelineFactory",
    "ContentGeneratorModule",
    "ContentGenerationSignature",
    "ContentRefinementSignature",
    # REER search components
    "REERSearchModule",
    "REERSearchResult",
    "SearchResult",
    "SearchContext",
    "SearchStrategy",
    "QueryEnhancementModule",
    "ContentAnalysisModule",
    "MockREERSearchEngine",
    # KPI evaluation components
    "KPIEvaluator",
    "PerformanceMetrics",
    "MetricResult",
    "MetricDefinition",
    "MetricType",
    "MetricLevel",
    "BenchmarkData",
    "BenchmarkManager",
    "KPICalculator",
]
