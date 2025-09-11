"""REER × DSPy × MLX Social Posting Pack - Plugins Module"""

# Language Model Adapters
from .dspy_lm import (
    SOCIAL_MEDIA_TEMPLATES,
    DSPyConfig,
    DSPyLanguageModelAdapter,
    DSPyModelFactory,
    PromptTemplate,
    create_social_media_adapter,
)

# Scoring and Heuristics
from .heuristics import (
    PLATFORM_WEIGHTS,
    ContentAnalyzer,
    ContentMetrics,
    EngagementPredictor,
    HeuristicScorer,
    HeuristicWeights,
    ReadabilityCalculator,
    SentimentAnalyzer,
    create_platform_scorer,
    score_content_async,
)

# Language Model Registry
from .lm_registry import (
    DummyLanguageModelAdapter,
    LanguageModelRegistry,
    ModelReference,
    ProviderConfig,
    calculate_perplexity,
    generate_text,
    get_recommended_model,
    get_registry,
    validate_model_uri,
)
from .mlx_lm import (
    BaseLMAdapter,
    MLXGenerationConfig,
    MLXLanguageModelAdapter,
    MLXModelConfig,
    MLXModelFactory,
    load_llama_mlx,
    load_mistral_mlx,
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
