"""REER × DSPy × MLX Social Posting Pack - Social Module"""

from .collectors import XAnalyticsNormalizer
from .dspy_modules import (
    SocialContentIdeator,
    SocialContentComposer,
    SocialContentOptimizer,
    SocialTrendAnalyzer,
    SocialContentPipeline,
    ContentBrief,
    ContentType,
    Platform,
)
from .kpis import SocialKPICalculator, PostMetrics, KPIResult, KPIDashboard, KPICategory

__all__ = [
    # Analytics and data normalization
    "XAnalyticsNormalizer",
    # DSPy content generation modules
    "SocialContentIdeator",
    "SocialContentComposer",
    "SocialContentOptimizer",
    "SocialTrendAnalyzer",
    "SocialContentPipeline",
    # Content generation data structures
    "ContentBrief",
    "ContentType",
    "Platform",
    # KPI calculation and metrics
    "SocialKPICalculator",
    "PostMetrics",
    "KPIResult",
    "KPIDashboard",
    "KPICategory",
]
