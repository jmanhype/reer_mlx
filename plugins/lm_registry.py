"""T021: LM registry for provider routing implementation.

Provides centralized registry for language model providers with routing
based on URI schemes (mlx::/dspy::/dummy::) and automatic adapter selection.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import logging
import re
from typing import Any
from urllib.parse import urlparse

from core.exceptions import ValidationError

from .dspy_lm import DSPyLanguageModelAdapter, DSPyModelFactory
from .mlx_lm import BaseLMAdapter, MLXLanguageModelAdapter, MLXModelFactory

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a language model provider."""

    name: str
    scheme: str  # URI scheme (e.g., 'mlx', 'dspy', 'dummy')
    adapter_class: type[BaseLMAdapter]
    factory_class: type | None = None
    default_model: str | None = None
    supported_models: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    priority: int = 100  # Lower number = higher priority


@dataclass
class ModelReference:
    """Reference to a specific model with provider information."""

    uri: str  # Full URI (e.g., "mlx://llama-3.2-3b-instruct")
    provider: str  # Provider name
    model_path: str  # Model path/name
    parameters: dict[str, Any] = field(default_factory=dict)


class DummyLanguageModelAdapter(BaseLMAdapter):
    """Dummy adapter for testing and fallback purposes."""

    def __init__(self, model_name: str = "dummy-model", **kwargs):
        """Initialize dummy adapter.

        Args:
            model_name: Name of the dummy model
            **kwargs: Additional parameters (ignored)
        """
        self.model_name = model_name
        self.parameters = kwargs
        self._responses = [
            "This is a dummy response from the {model} model.",
            "Here's another sample output for testing purposes.",
            "Dummy content generation for social media posting.",
            "Test response with high engagement potential!",
            "Sample text for development and testing scenarios.",
        ]
        self._response_index = 0

        logger.info(f"Initialized dummy adapter: {model_name}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """Generate dummy text.

        Args:
            prompt: Input prompt (used for context)
            max_tokens: Maximum tokens (affects response length)
            temperature: Temperature (affects response selection)
            **kwargs: Additional parameters

        Returns:
            Dummy generated text
        """
        # Simulate async operation
        await asyncio.sleep(0.1)

        # Select response based on prompt and temperature
        base_response = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        # Format with model name
        response = base_response.format(model=self.model_name)

        # Add prompt context
        context = prompt[:50] + "..." if len(prompt) > 50 else prompt

        full_response = f"Based on '{context}': {response}"

        # Simulate max_tokens constraint
        if (
            max_tokens and len(full_response) > max_tokens * 4
        ):  # Rough char to token ratio
            full_response = full_response[: max_tokens * 4] + "..."

        return full_response

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate streaming dummy text.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Temperature
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)

        # Split response into chunks for streaming simulation
        words = response.split()
        chunk_size = max(1, len(words) // 5)

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield chunk + " "

    async def get_perplexity(self, text: str) -> float:
        """Return dummy perplexity.

        Args:
            text: Text to analyze

        Returns:
            Dummy perplexity score
        """
        # Simulate perplexity based on text length and complexity
        words = len(text.split())

        if words < 10:
            return 15.0  # Short text, higher perplexity
        if words < 50:
            return 8.0  # Medium text, good perplexity
        return 12.0  # Long text, slightly higher perplexity

    def is_available(self) -> bool:
        """Dummy adapter is always available."""
        return True


class LanguageModelRegistry:
    """Central registry for language model providers and routing."""

    def __init__(self):
        """Initialize the registry."""
        self._providers: dict[str, ProviderConfig] = {}
        self._adapters: dict[str, BaseLMAdapter] = {}  # Cached adapters
        self._default_provider: str | None = None

        # Register built-in providers
        self._register_builtin_providers()

    def _register_builtin_providers(self) -> None:
        """Register built-in language model providers."""
        # MLX Provider
        mlx_config = ProviderConfig(
            name="mlx",
            scheme="mlx",
            adapter_class=MLXLanguageModelAdapter,
            factory_class=MLXModelFactory,
            default_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            supported_models=[
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            ],
            capabilities=[
                "local_inference",
                "apple_silicon",
                "perplexity",
                "streaming",
            ],
            priority=10,
        )
        self.register_provider(mlx_config)

        # DSPy Provider
        dspy_config = ProviderConfig(
            name="dspy",
            scheme="dspy",
            adapter_class=DSPyLanguageModelAdapter,
            factory_class=DSPyModelFactory,
            default_model="openai/gpt-3.5-turbo",
            supported_models=[
                "openai/gpt-3.5-turbo",
                "openai/gpt-4",
                "openai/gpt-4-turbo",
                "anthropic/claude-3-sonnet-20240229",
                "anthropic/claude-3-haiku-20240307",
                "together/meta-llama/Llama-3-8b-chat-hf",
                "together/mistralai/Mistral-7B-Instruct-v0.3",
            ],
            capabilities=[
                "cloud_api",
                "structured_prompting",
                "optimization",
                "reasoning",
            ],
            priority=20,
        )
        self.register_provider(dspy_config)

        # Dummy Provider
        dummy_config = ProviderConfig(
            name="dummy",
            scheme="dummy",
            adapter_class=DummyLanguageModelAdapter,
            default_model="dummy-model",
            supported_models=["dummy-model", "test-model", "fallback-model"],
            capabilities=["testing", "fallback", "development"],
            priority=1000,  # Lowest priority
        )
        self.register_provider(dummy_config)

        # Set default provider
        self._default_provider = "mlx"

    def register_provider(self, config: ProviderConfig) -> None:
        """Register a new language model provider.

        Args:
            config: Provider configuration
        """
        self._providers[config.name] = config
        logger.info(f"Registered provider: {config.name} (scheme: {config.scheme})")

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get provider configuration by name.

        Args:
            name: Provider name

        Returns:
            Provider configuration or None
        """
        return self._providers.get(name)

    def list_providers(self) -> list[ProviderConfig]:
        """List all registered providers.

        Returns:
            List of provider configurations
        """
        return list(self._providers.values())

    def parse_model_uri(self, uri: str) -> ModelReference:
        """Parse a model URI into components.

        Args:
            uri: Model URI (e.g., "mlx://llama-3.2-3b-instruct", "dspy://openai/gpt-4")

        Returns:
            ModelReference object

        Raises:
            ValidationError: If URI format is invalid
        """
        # Handle scheme-less URIs
        if "://" not in uri:
            # Default to the default provider
            provider_name = self._default_provider or "dummy"
            model_path = uri
            parameters = {}
        else:
            # Parse full URI
            try:
                parsed = urlparse(uri)
                scheme = parsed.scheme

                # Find provider by scheme
                provider_name = None
                for name, config in self._providers.items():
                    if config.scheme == scheme:
                        provider_name = name
                        break

                if not provider_name:
                    raise ValidationError(f"Unknown provider scheme: {scheme}")

                # Extract model path and parameters
                model_path = parsed.netloc + parsed.path
                if model_path.startswith("/"):
                    model_path = model_path[1:]

                # Parse query parameters
                parameters = {}
                if parsed.query:
                    for param in parsed.query.split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            # Try to convert to appropriate type
                            try:
                                parameters[key] = float(value)
                            except ValueError:
                                if value.lower() in ("true", "false"):
                                    parameters[key] = value.lower() == "true"
                                else:
                                    parameters[key] = value

            except Exception as e:
                raise ValidationError(f"Invalid model URI '{uri}': {e}")

        return ModelReference(
            uri=uri,
            provider=provider_name,
            model_path=model_path,
            parameters=parameters,
        )

    async def get_adapter(self, uri: str, **kwargs) -> BaseLMAdapter:
        """Get or create a language model adapter for the given URI.

        Args:
            uri: Model URI
            **kwargs: Additional parameters for adapter creation

        Returns:
            Language model adapter instance

        Raises:
            ValidationError: If provider or model is not supported
        """
        # Check if adapter is already cached
        cache_key = f"{uri}_{hash(frozenset(kwargs.items()))}"
        if cache_key in self._adapters:
            return self._adapters[cache_key]

        # Parse URI
        model_ref = self.parse_model_uri(uri)

        # Get provider config
        provider_config = self.get_provider(model_ref.provider)
        if not provider_config:
            raise ValidationError(f"Provider '{model_ref.provider}' not found")

        # Merge parameters
        all_params = {**model_ref.parameters, **kwargs}

        try:
            # Create adapter instance
            if provider_config.name == "mlx":
                adapter = await self._create_mlx_adapter(model_ref, all_params)
            elif provider_config.name == "dspy":
                adapter = await self._create_dspy_adapter(model_ref, all_params)
            elif provider_config.name == "dummy":
                adapter = DummyLanguageModelAdapter(
                    model_name=model_ref.model_path, **all_params
                )
            # Use factory if available
            elif provider_config.factory_class:
                adapter = provider_config.factory_class.create_adapter(
                    model_ref.model_path, **all_params
                )
            else:
                # Direct instantiation
                adapter = provider_config.adapter_class(
                    model_ref.model_path, **all_params
                )

            # Cache the adapter
            self._adapters[cache_key] = adapter

            logger.info(f"Created adapter for {uri}")
            return adapter

        except Exception as e:
            logger.exception(f"Failed to create adapter for {uri}: {e}")
            raise ValidationError(f"Adapter creation failed: {e}")

    async def _create_mlx_adapter(
        self, model_ref: ModelReference, params: dict[str, Any]
    ) -> MLXLanguageModelAdapter:
        """Create MLX adapter with proper configuration."""
        from .mlx_lm import MLXGenerationConfig, MLXModelConfig

        # Split parameters into model and generation configs
        model_params = {
            k: v
            for k, v in params.items()
            if k
            in [
                "tokenizer_config",
                "trust_remote_code",
                "revision",
                "dtype",
                "max_context_length",
            ]
        }

        gen_params = {
            k: v
            for k, v in params.items()
            if k in ["max_tokens", "temperature", "top_p", "repetition_penalty", "seed"]
        }

        model_config = MLXModelConfig(model_path=model_ref.model_path, **model_params)

        generation_config = MLXGenerationConfig(**gen_params) if gen_params else None

        adapter = MLXLanguageModelAdapter(model_config, generation_config)
        await adapter.load_model()  # Pre-load the model

        return adapter

    async def _create_dspy_adapter(
        self, model_ref: ModelReference, params: dict[str, Any]
    ) -> DSPyLanguageModelAdapter:
        """Create DSPy adapter with proper configuration."""
        from .dspy_lm import DSPyConfig

        # Parse model path for provider and model
        if "/" in model_ref.model_path:
            provider, model = model_ref.model_path.split("/", 1)
        else:
            # Default to OpenAI if no provider specified
            provider = "openai"
            model = model_ref.model_path

        config = DSPyConfig(provider=provider, model=model, **params)

        adapter = DSPyLanguageModelAdapter(config)
        await adapter.initialize()

        return adapter

    async def route_generate(self, uri: str, prompt: str, **kwargs) -> str:
        """Route generation request to appropriate provider.

        Args:
            uri: Model URI
            prompt: Text prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """
        adapter = await self.get_adapter(uri, **kwargs)
        return await adapter.generate(prompt, **kwargs)

    async def route_perplexity(self, uri: str, text: str, **kwargs) -> float:
        """Route perplexity calculation to appropriate provider.

        Args:
            uri: Model URI
            text: Text to analyze
            **kwargs: Additional parameters

        Returns:
            Perplexity score
        """
        adapter = await self.get_adapter(uri, **kwargs)
        return await adapter.get_perplexity(text)

    def get_recommended_model(
        self, use_case: str = "general", prefer_local: bool = True
    ) -> str:
        """Get recommended model URI for a use case.

        Args:
            use_case: Use case (general, social_media, creative, analysis)
            prefer_local: Prefer local models over cloud APIs

        Returns:
            Recommended model URI
        """
        # Define use case preferences
        use_case_models = {
            "social_media": {
                "local": "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit",
                "cloud": "dspy://openai/gpt-3.5-turbo",
            },
            "creative": {
                "local": "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit",
                "cloud": "dspy://anthropic/claude-3-sonnet-20240229",
            },
            "analysis": {
                "local": "mlx://mlx-community/Qwen2.5-7B-Instruct-4bit",
                "cloud": "dspy://openai/gpt-4",
            },
            "general": {
                "local": "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit",
                "cloud": "dspy://openai/gpt-3.5-turbo",
            },
        }

        models = use_case_models.get(use_case, use_case_models["general"])

        if prefer_local:
            return models["local"]
        return models["cloud"]

    def set_default_provider(self, provider_name: str) -> None:
        """Set the default provider for scheme-less URIs.

        Args:
            provider_name: Name of the provider to set as default

        Raises:
            ValidationError: If provider doesn't exist
        """
        if provider_name not in self._providers:
            raise ValidationError(f"Provider '{provider_name}' not found")

        self._default_provider = provider_name
        logger.info(f"Set default provider to: {provider_name}")

    def clear_cache(self) -> None:
        """Clear all cached adapters."""
        self._adapters.clear()
        logger.info("Cleared adapter cache")

    async def health_check(self) -> dict[str, dict[str, Any]]:
        """Check health of all providers.

        Returns:
            Health status for each provider
        """
        health_status = {}

        for name, config in self._providers.items():
            try:
                # Create a test adapter
                test_uri = f"{config.scheme}://{config.default_model}"
                adapter = await self.get_adapter(test_uri)

                is_available = adapter.is_available()

                health_status[name] = {
                    "available": is_available,
                    "scheme": config.scheme,
                    "default_model": config.default_model,
                    "capabilities": config.capabilities,
                    "error": None,
                }

                if is_available:
                    # Test generation
                    try:
                        test_response = await adapter.generate(
                            "Test prompt", max_tokens=10
                        )
                        health_status[name]["test_generation"] = True
                        health_status[name]["test_response_length"] = len(test_response)
                    except Exception as e:
                        health_status[name]["test_generation"] = False
                        health_status[name]["generation_error"] = str(e)

            except Exception as e:
                health_status[name] = {"available": False, "error": str(e)}

        return health_status


# Global registry instance
_registry = None


def get_registry() -> LanguageModelRegistry:
    """Get the global language model registry instance.

    Returns:
        Global registry instance
    """
    global _registry
    if _registry is None:
        _registry = LanguageModelRegistry()
    return _registry


# Convenience functions
async def generate_text(uri: str, prompt: str, **kwargs) -> str:
    """Generate text using the registry.

    Args:
        uri: Model URI
        prompt: Text prompt
        **kwargs: Generation parameters

    Returns:
        Generated text
    """
    registry = get_registry()
    return await registry.route_generate(uri, prompt, **kwargs)


async def calculate_perplexity(uri: str, text: str, **kwargs) -> float:
    """Calculate perplexity using the registry.

    Args:
        uri: Model URI
        text: Text to analyze
        **kwargs: Additional parameters

    Returns:
        Perplexity score
    """
    registry = get_registry()
    return await registry.route_perplexity(uri, text, **kwargs)


def get_recommended_model(use_case: str = "general", prefer_local: bool = True) -> str:
    """Get recommended model for use case.

    Args:
        use_case: Use case
        prefer_local: Prefer local models

    Returns:
        Model URI
    """
    registry = get_registry()
    return registry.get_recommended_model(use_case, prefer_local)


# URI pattern validation
URI_PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):\/\/([^?#]+)(\?[^#]*)?(#.*)?$")


def validate_model_uri(uri: str) -> bool:
    """Validate model URI format.

    Args:
        uri: URI to validate

    Returns:
        True if valid, False otherwise
    """
    if "://" not in uri:
        return True  # Scheme-less URIs are valid

    return URI_PATTERN.match(uri) is not None
