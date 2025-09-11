"""REER × DSPy × MLX Social Posting Pack - Plugins Module"""

# Language Model Adapters
from .mlx_lm import (
    MLXLanguageModelAdapter,
    MLXModelFactory,
    MLXModelConfig,
    MLXGenerationConfig,
    BaseLMAdapter,
    load_llama_mlx,
    load_mistral_mlx,
)

from .dspy_lm import (
    DSPyLanguageModelAdapter,
    DSPyModelFactory,
    DSPyConfig,
    PromptTemplate,
    SOCIAL_MEDIA_TEMPLATES,
    create_social_media_adapter,
)

# Scoring and Heuristics
from .heuristics import (
    HeuristicScorer,
    HeuristicWeights,
    ContentMetrics,
    ContentAnalyzer,
    ReadabilityCalculator,
    SentimentAnalyzer,
    EngagementPredictor,
    PLATFORM_WEIGHTS,
    create_platform_scorer,
    score_content_async,
)

# Language Model Registry
from .lm_registry import (
    LanguageModelRegistry,
    ProviderConfig,
    ModelReference,
    DummyLanguageModelAdapter,
    get_registry,
    generate_text,
    calculate_perplexity,
    get_recommended_model,
    validate_model_uri,
)

__all__ = [
    # MLX Adapter
    "MLXLanguageModelAdapter",
    "MLXModelFactory",
    "MLXModelConfig",
    "MLXGenerationConfig",
    "BaseLMAdapter",
    "load_llama_mlx",
    "load_mistral_mlx",
    # DSPy Adapter
    "DSPyLanguageModelAdapter",
    "DSPyModelFactory",
    "DSPyConfig",
    "PromptTemplate",
    "SOCIAL_MEDIA_TEMPLATES",
    "create_social_media_adapter",
    # Heuristics
    "HeuristicScorer",
    "HeuristicWeights",
    "ContentMetrics",
    "ContentAnalyzer",
    "ReadabilityCalculator",
    "SentimentAnalyzer",
    "EngagementPredictor",
    "PLATFORM_WEIGHTS",
    "create_platform_scorer",
    "score_content_async",
    # Registry
    "LanguageModelRegistry",
    "ProviderConfig",
    "ModelReference",
    "DummyLanguageModelAdapter",
    "get_registry",
    "generate_text",
    "calculate_perplexity",
    "get_recommended_model",
    "validate_model_uri",
]
