"""REER × DSPy × MLX Social Posting Pack - DSPy Program Module

This module implements the core DSPy pipeline orchestration for social media content
generation and optimization, integrating REER search, content scoring, and GEPA training.

Modules:
    pipeline: Main DSPy pipeline orchestrator (T022)
    reer_module: REER search wrapper with DSPy integration (T023)
    evaluator: KPI evaluator for performance metrics (T024)
"""

from .pipeline import (
    REERDSPyPipeline,
    PipelineConfig,
    ContentRequest,
    ContentResult,
    PipelineResult,
    PipelineFactory,
    ContentGeneratorModule,
    ContentGenerationSignature,
    ContentRefinementSignature,
)

from .reer_module import (
    REERSearchModule,
    REERSearchResult,
    SearchResult,
    SearchContext,
    SearchStrategy,
    QueryEnhancementModule,
    ContentAnalysisModule,
    MockREERSearchEngine,
)

from .evaluator import (
    KPIEvaluator,
    PerformanceMetrics,
    MetricResult,
    MetricDefinition,
    MetricType,
    MetricLevel,
    BenchmarkData,
    BenchmarkManager,
    KPICalculator,
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
