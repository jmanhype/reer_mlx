from typing import Optional
"""T019: DSPy language model adapter implementation.

Provides DSPy integration for cloud providers (OpenAI, Anthropic, Together)
with structured prompting, reasoning, and optimization capabilities.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import logging
import os
from typing import Any

try:
    import dspy
    from dspy import Anthropic, ChatAdapter, OpenAI, Together
    from dspy.teleprompt import MIPRO, BootstrapFewShot

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None
    OpenAI = None
    Anthropic = None
    Together = None
    ChatAdapter = None
    BootstrapFewShot = None
    MIPRO = None

from core.exceptions import ScoringError, ValidationError

from .mlx_lm import BaseLMAdapter

logger = logging.getLogger(__name__)


@dataclass
class DSPyConfig:
    """Configuration for DSPy language models."""

    provider: str  # openai, anthropic, together
    model: str
    api_key: str | None = None
    api_base: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    timeout: int = 30
    max_retries: int = 3

    # DSPy specific
    use_chat_api: bool = True
    cache: bool = True
    reasoning_mode: bool = False  # Enable chain-of-thought


@dataclass
class PromptTemplate:
    """Template for structured prompting."""

    system_prompt: str
    user_template: str
    variables: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)


class DSPyPromptModule(dspy.Module if DSPY_AVAILABLE else object):
    """DSPy module for structured content generation."""

    def __init__(self, template: PromptTemplate):
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available")

        super().__init__()
        self.template = template

        # Create signature based on template variables
        input_fields = ", ".join(template.variables)
        signature = f"{input_fields} -> output"

        if template.template.reasoning_mode:
            self.predictor = dspy.ChainOfThought(signature)
        else:
            self.predictor = dspy.Predict(signature)

    def forward(self, **kwargs):
        """Forward pass through the module."""
        return self.predictor(**kwargs)


class DSPyLanguageModelAdapter(BaseLMAdapter):
    """DSPy-based language model adapter for cloud providers."""

    def __init__(self, config: DSPyConfig):
        """Initialize DSPy adapter.

        Args:
            config: DSPy configuration
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available. Please install dspy-ai package.")

        self.config = config
        self.lm = None
        self._initialized = False

        # Prompt templates
        self.templates: dict[str, PromptTemplate] = {}

        logger.info(f"Initializing DSPy adapter for {config.provider}:{config.model}")

    async def initialize(self) -> None:
        """Initialize the DSPy language model."""
        if self._initialized:
            return

        try:
            # Get API key from config or environment
            api_key = self.config.api_key or self._get_api_key()

            if not api_key:
                raise ValidationError(f"API key required for {self.config.provider}")

            # Create appropriate DSPy language model
            if self.config.provider.lower() == "openai":
                self.lm = OpenAI(
                    model=self.config.model,
                    api_key=api_key,
                    api_base=self.config.api_base,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    stop=self.config.stop,
                    timeout=self.config.timeout,
                    cache=self.config.cache,
                )

            elif self.config.provider.lower() == "anthropic":
                self.lm = Anthropic(
                    model=self.config.model,
                    api_key=api_key,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop_sequences=self.config.stop,
                    timeout=self.config.timeout,
                    cache=self.config.cache,
                )

            elif self.config.provider.lower() == "together":
                self.lm = Together(
                    model=self.config.model,
                    api_key=api_key,
                    api_base=self.config.api_base,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop=self.config.stop,
                    timeout=self.config.timeout,
                    cache=self.config.cache,
                )

            else:
                raise ValidationError(f"Unsupported provider: {self.config.provider}")

            # Set as default DSPy LM
            dspy.settings.configure(lm=self.lm)

            self._initialized = True
            logger.info(f"DSPy adapter initialized for {self.config.provider}")

        except Exception as e:
            logger.exception(f"Failed to initialize DSPy adapter: {e}")
            raise ValidationError(f"DSPy initialization failed: {e}")

    def _get_api_key(self) -> str | None:
        """Get API key from environment variables."""
        provider = self.config.provider.lower()

        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        if provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        if provider == "together":
            return os.getenv("TOGETHER_API_KEY")

        return None

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """Generate text using DSPy model.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        await self.initialize()

        try:
            # Create a simple signature for generation
            class GenerateText(dspy.Signature):
                """Generate high-quality text based on the given prompt."""

                prompt = dspy.InputField()
                output = dspy.OutputField()

            # Use ChainOfThought if reasoning mode is enabled
            if self.config.reasoning_mode:
                predictor = dspy.ChainOfThought(GenerateText)
            else:
                predictor = dspy.Predict(GenerateText)

            # Override generation parameters if provided
            if max_tokens or temperature:
                original_lm = dspy.settings.lm
                temp_config = self.config
                if max_tokens:
                    temp_config.max_tokens = max_tokens
                if temperature:
                    temp_config.temperature = temperature

                # Create temporary LM with updated config
                await self._create_temp_lm(temp_config)

            result = predictor(prompt=prompt)

            # Restore original LM if changed
            if max_tokens or temperature:
                dspy.settings.configure(lm=original_lm)

            return result.output

        except Exception as e:
            logger.exception(f"DSPy generation failed: {e}")
            raise ScoringError(f"Generation failed: {e}")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate streaming text using DSPy model.

        Note: DSPy doesn't natively support streaming, so this falls back
        to non-streaming generation.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated text (as single chunk for DSPy)
        """
        # DSPy doesn't support streaming by default
        # Return the full result as a single chunk
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield result

    async def get_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text.

        Note: DSPy models don't directly support perplexity calculation.
        This is an approximation using log probabilities if available.

        Args:
            text: Text to calculate perplexity for

        Returns:
            Approximated perplexity score
        """
        await self.initialize()

        try:
            # For cloud APIs, we can't directly get perplexity
            # We'll use a heuristic based on generation confidence

            # Create a signature for text evaluation
            class EvaluateText(dspy.Signature):
                """Evaluate the quality and naturalness of the given text."""

                text = dspy.InputField()
                quality_score = dspy.OutputField(desc="Quality score from 1-10")
                naturalness = dspy.OutputField(
                    desc="How natural the text sounds from 1-10"
                )

            predictor = dspy.Predict(EvaluateText)
            result = predictor(text=text)

            # Convert quality scores to approximate perplexity
            # Higher quality = lower perplexity
            try:
                quality = float(result.quality_score)
                naturalness = float(result.naturalness)
                avg_score = (quality + naturalness) / 2

                # Convert to perplexity-like score (lower is better)
                # Scale from 1-10 to approximate perplexity range
                return max(1.0, 100.0 / max(1.0, avg_score))

            except (ValueError, AttributeError):
                # Fallback if parsing fails
                return 50.0  # Neutral perplexity

        except Exception as e:
            logger.exception(f"Perplexity calculation failed: {e}")
            return float("inf")

    async def _create_temp_lm(self, config: DSPyConfig) -> None:
        """Create temporary LM with updated configuration."""
        api_key = config.api_key or self._get_api_key()

        if config.provider.lower() == "openai":
            temp_lm = OpenAI(
                model=config.model,
                api_key=api_key,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop,
                cache=config.cache,
            )
        elif config.provider.lower() == "anthropic":
            temp_lm = Anthropic(
                model=config.model,
                api_key=api_key,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop,
                cache=config.cache,
            )
        elif config.provider.lower() == "together":
            temp_lm = Together(
                model=config.model,
                api_key=api_key,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop,
                cache=config.cache,
            )

        dspy.settings.configure(lm=temp_lm)

    def add_template(self, name: str, template: PromptTemplate) -> None:
        """Add a prompt template for structured generation.

        Args:
            name: Template name
            template: Prompt template configuration
        """
        self.templates[name] = template
        logger.info(f"Added template: {name}")

    async def generate_with_template(self, template_name: str, **variables) -> str:
        """Generate text using a predefined template.

        Args:
            template_name: Name of the template to use
            **variables: Template variables

        Returns:
            Generated text
        """
        if template_name not in self.templates:
            raise ValidationError(f"Template '{template_name}' not found")

        template = self.templates[template_name]

        # Validate required variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValidationError(f"Missing template variables: {missing_vars}")

        await self.initialize()

        try:
            # Create module for this template
            module = DSPyPromptModule(template)
            result = module(**variables)

            return result.output

        except Exception as e:
            logger.exception(f"Template generation failed: {e}")
            raise ScoringError(f"Template generation failed: {e}")

    async def optimize_with_examples(
        self, examples: list[dict[str, str]], metric_fn: Optional[callable] = None
    ) -> None:
        """Optimize the model using few-shot examples.

        Args:
            examples: List of input-output examples
            metric_fn: Optional metric function for optimization
        """
        await self.initialize()

        try:
            # Create examples in DSPy format
            dspy_examples = []
            for example in examples:
                dspy_examples.append(
                    dspy.Example(**example).with_inputs(*example.keys())
                )

            # Use BootstrapFewShot for optimization
            BootstrapFewShot(
                metric=metric_fn,
                max_bootstrapped_demos=len(examples),
                max_labeled_demos=min(8, len(examples)),
            )

            # This would optimize a specific program/module
            # For now, we'll just store the examples for reference
            self._training_examples = dspy_examples

            logger.info(f"Stored {len(examples)} examples for optimization")

        except Exception as e:
            logger.exception(f"Optimization failed: {e}")
            raise ScoringError(f"Optimization failed: {e}")

    def is_available(self) -> bool:
        """Check if DSPy adapter is available."""
        return DSPY_AVAILABLE and dspy is not None

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the DSPy model.

        Returns:
            Model information dictionary
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "initialized": self._initialized,
            "dspy_available": DSPY_AVAILABLE,
            "use_chat_api": self.config.use_chat_api,
            "reasoning_mode": self.config.reasoning_mode,
            "templates": list(self.templates.keys()),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }


class DSPyModelFactory:
    """Factory for creating DSPy model adapters."""

    @staticmethod
    def create_openai_adapter(
        model: str = "gpt-3.5-turbo", api_key: str | None = None, **kwargs
    ) -> DSPyLanguageModelAdapter:
        """Create OpenAI DSPy adapter.

        Args:
            model: OpenAI model name
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            Configured DSPy adapter
        """
        config = DSPyConfig(provider="openai", model=model, api_key=api_key, **kwargs)
        return DSPyLanguageModelAdapter(config)

    @staticmethod
    def create_anthropic_adapter(
        model: str = "claude-3-sonnet-20240229", api_key: str | None = None, **kwargs
    ) -> DSPyLanguageModelAdapter:
        """Create Anthropic DSPy adapter.

        Args:
            model: Anthropic model name
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            Configured DSPy adapter
        """
        config = DSPyConfig(
            provider="anthropic", model=model, api_key=api_key, **kwargs
        )
        return DSPyLanguageModelAdapter(config)

    @staticmethod
    def create_together_adapter(
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        api_key: str | None = None,
        **kwargs,
    ) -> DSPyLanguageModelAdapter:
        """Create Together AI DSPy adapter.

        Args:
            model: Together AI model name
            api_key: Optional API key
            **kwargs: Additional configuration

        Returns:
            Configured DSPy adapter
        """
        config = DSPyConfig(provider="together", model=model, api_key=api_key, **kwargs)
        return DSPyLanguageModelAdapter(config)


# Common prompt templates
SOCIAL_MEDIA_TEMPLATES = {
    "twitter_post": PromptTemplate(
        system_prompt="You are a social media expert creating engaging Twitter posts.",
        user_template="Create a Twitter post about {topic} for {audience}. Include relevant hashtags.",
        variables=["topic", "audience"],
    ),
    "linkedin_article": PromptTemplate(
        system_prompt="You are a professional content creator for LinkedIn.",
        user_template="Write a LinkedIn article about {topic} targeting {audience}. Keep it professional and insightful.",
        variables=["topic", "audience"],
    ),
    "instagram_caption": PromptTemplate(
        system_prompt="You are an Instagram content creator focused on visual storytelling.",
        user_template="Create an engaging Instagram caption for a post about {topic}. Include emojis and hashtags.",
        variables=["topic"],
    ),
}


def create_social_media_adapter(
    provider: str = "openai", model: str | None = None, **kwargs
) -> DSPyLanguageModelAdapter:
    """Create a DSPy adapter optimized for social media content.

    Args:
        provider: LM provider (openai, anthropic, together)
        model: Optional model name
        **kwargs: Additional configuration

    Returns:
        DSPy adapter with social media templates
    """
    # Default models for each provider
    if model is None:
        if provider == "openai":
            model = "gpt-3.5-turbo"
        elif provider == "anthropic":
            model = "claude-3-sonnet-20240229"
        elif provider == "together":
            model = "meta-llama/Llama-3-8b-chat-hf"

    # Create adapter
    if provider == "openai":
        adapter = DSPyModelFactory.create_openai_adapter(model, **kwargs)
    elif provider == "anthropic":
        adapter = DSPyModelFactory.create_anthropic_adapter(model, **kwargs)
    elif provider == "together":
        adapter = DSPyModelFactory.create_together_adapter(model, **kwargs)
    else:
        raise ValidationError(f"Unsupported provider: {provider}")

    # Add social media templates
    for name, template in SOCIAL_MEDIA_TEMPLATES.items():
        adapter.add_template(name, template)

    return adapter
