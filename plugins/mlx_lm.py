"""T018: MLX language model adapter implementation.

Provides MLX integration for Apple Silicon optimization with efficient
inference and memory management. Supports local model loading and
generation with Apple's MLX framework.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Iterator
from collections.abc import AsyncIterator
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.utils import load as load_utils

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    load = None
    generate = None
    stream_generate = None
    load_utils = None

from core.exceptions import ValidationError, ScoringError


logger = logging.getLogger(__name__)


@dataclass
class MLXGenerationConfig:
    """Configuration for MLX model generation."""

    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    repetition_context_size: int = 20
    seed: Optional[int] = None
    stop_tokens: List[str] = field(default_factory=list)


@dataclass
class MLXModelConfig:
    """Configuration for MLX model loading."""

    model_path: str
    tokenizer_config: Optional[Dict[str, Any]] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    dtype: str = "float16"  # float16, bfloat16, float32
    max_context_length: int = 4096


class BaseLMAdapter(ABC):
    """Base class for language model adapters."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Generate text stream from prompt."""
        pass

    @abstractmethod
    async def get_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if adapter is available and ready."""
        pass


class MLXLanguageModelAdapter(BaseLMAdapter):
    """MLX-based language model adapter for Apple Silicon optimization."""

    def __init__(
        self,
        model_config: MLXModelConfig,
        generation_config: Optional[MLXGenerationConfig] = None,
    ):
        """Initialize MLX adapter.

        Args:
            model_config: Configuration for model loading
            generation_config: Configuration for text generation
        """
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Please install mlx and mlx-lm packages."
            )

        self.model_config = model_config
        self.generation_config = generation_config or MLXGenerationConfig()
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        logger.info(f"Initializing MLX adapter for model: {model_config.model_path}")

    async def load_model(self) -> None:
        """Load the MLX model and tokenizer."""
        if self._model_loaded:
            return

        try:
            logger.info(f"Loading MLX model from {self.model_config.model_path}")

            # Load model and tokenizer using mlx-lm
            self.model, self.tokenizer = load(
                self.model_config.model_path,
                tokenizer_config=self.model_config.tokenizer_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            self._model_loaded = True
            logger.info("MLX model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise ValidationError(f"Model loading failed: {e}")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using MLX model.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        await self.load_model()

        # Use provided parameters or defaults from config
        gen_config = {
            "max_tokens": max_tokens or self.generation_config.max_tokens,
            "temp": temperature or self.generation_config.temperature,
            "top_p": top_p or self.generation_config.top_p,
            "repetition_penalty": kwargs.get(
                "repetition_penalty", self.generation_config.repetition_penalty
            ),
            "repetition_context_size": kwargs.get(
                "repetition_context_size",
                self.generation_config.repetition_context_size,
            ),
        }

        if self.generation_config.seed is not None:
            gen_config["seed"] = self.generation_config.seed

        try:
            # Generate text using mlx-lm
            response = generate(
                model=self.model, tokenizer=self.tokenizer, prompt=prompt, **gen_config
            )

            # Extract generated text (remove prompt)
            if response.startswith(prompt):
                generated_text = response[len(prompt) :].strip()
            else:
                generated_text = response.strip()

            return generated_text

        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            raise ScoringError(f"Generation failed: {e}")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate streaming text using MLX model.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        await self.load_model()

        gen_config = {
            "max_tokens": max_tokens or self.generation_config.max_tokens,
            "temp": temperature or self.generation_config.temperature,
            "top_p": kwargs.get("top_p", self.generation_config.top_p),
        }

        try:
            # Use streaming generation if available
            if stream_generate is not None:
                for chunk in stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    **gen_config,
                ):
                    yield chunk
            else:
                # Fallback to non-streaming
                result = await self.generate(prompt, **gen_config)
                yield result

        except Exception as e:
            logger.error(f"MLX streaming generation failed: {e}")
            raise ScoringError(f"Streaming generation failed: {e}")

    async def get_perplexity(self, text: str) -> float:
        """Calculate perplexity for given text using MLX model.

        Args:
            text: Text to calculate perplexity for

        Returns:
            Perplexity score
        """
        await self.load_model()

        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Convert to MLX array
            input_ids = mx.array(tokens).reshape(1, -1)

            # Get model logits
            with mx.stream():
                logits = self.model(input_ids)

            # Calculate perplexity
            # Shift targets for next token prediction
            if len(tokens) < 2:
                return float("inf")

            targets = mx.array(tokens[1:]).reshape(1, -1)
            logits = logits[:, :-1, :]  # Remove last position

            # Calculate log probabilities
            log_probs = mx.log_softmax(logits, axis=-1)

            # Gather probabilities for target tokens
            target_log_probs = mx.take_along_axis(
                log_probs, targets.reshape(1, -1, 1), axis=-1
            ).squeeze(-1)

            # Calculate mean negative log likelihood
            nll = -mx.mean(target_log_probs)

            # Convert to perplexity
            perplexity = mx.exp(nll).item()

            return float(perplexity)

        except Exception as e:
            logger.error(f"Perplexity calculation failed: {e}")
            return float("inf")

    async def calculate_token_probabilities(
        self, prompt: str, continuation: str
    ) -> List[float]:
        """Calculate token-level probabilities for a continuation.

        Args:
            prompt: Context prompt
            continuation: Text to calculate probabilities for

        Returns:
            List of token probabilities
        """
        await self.load_model()

        try:
            full_text = prompt + continuation
            tokens = self.tokenizer.encode(full_text)
            prompt_tokens = self.tokenizer.encode(prompt)

            input_ids = mx.array(tokens).reshape(1, -1)

            # Get model predictions
            with mx.stream():
                logits = self.model(input_ids)

            # Get probabilities for continuation tokens
            continuation_start = len(prompt_tokens)
            continuation_logits = logits[0, continuation_start - 1 : -1, :]
            continuation_tokens = tokens[continuation_start:]

            # Calculate softmax probabilities
            probs = mx.softmax(continuation_logits, axis=-1)

            # Extract probabilities for actual tokens
            token_probs = []
            for i, token_id in enumerate(continuation_tokens):
                prob = probs[i, token_id].item()
                token_probs.append(float(prob))

            return token_probs

        except Exception as e:
            logger.error(f"Token probability calculation failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if MLX adapter is available."""
        return MLX_AVAILABLE and mx is not None

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        await self.load_model()

        info = {
            "model_path": self.model_config.model_path,
            "loaded": self._model_loaded,
            "dtype": self.model_config.dtype,
            "max_context_length": self.model_config.max_context_length,
            "mlx_available": MLX_AVAILABLE,
        }

        if self.tokenizer is not None:
            try:
                info.update(
                    {
                        "vocab_size": (
                            len(self.tokenizer.get_vocab())
                            if hasattr(self.tokenizer, "get_vocab")
                            else "unknown"
                        ),
                        "tokenizer_type": type(self.tokenizer).__name__,
                    }
                )
            except Exception:
                pass

        return info

    async def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model_loaded:
            self.model = None
            self.tokenizer = None
            self._model_loaded = False

            # Force garbage collection on MLX arrays
            if mx is not None:
                mx.eval([])  # Ensure all pending operations complete

            logger.info("MLX model unloaded")


class MLXModelFactory:
    """Factory for creating MLX model adapters."""

    @staticmethod
    def create_adapter(model_path: str, **kwargs) -> MLXLanguageModelAdapter:
        """Create an MLX adapter instance.

        Args:
            model_path: Path to the model
            **kwargs: Additional configuration parameters

        Returns:
            Configured MLX adapter
        """
        model_config = MLXModelConfig(
            model_path=model_path,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "tokenizer_config",
                    "trust_remote_code",
                    "revision",
                    "dtype",
                    "max_context_length",
                ]
            },
        )

        generation_config = MLXGenerationConfig(
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in ["max_tokens", "temperature", "top_p", "repetition_penalty", "seed"]
            }
        )

        return MLXLanguageModelAdapter(model_config, generation_config)

    @staticmethod
    def create_from_huggingface(
        model_name: str, cache_dir: Optional[str] = None, **kwargs
    ) -> MLXLanguageModelAdapter:
        """Create adapter for HuggingFace model.

        Args:
            model_name: HuggingFace model name
            cache_dir: Optional cache directory
            **kwargs: Additional configuration

        Returns:
            MLX adapter for HuggingFace model
        """
        # For HuggingFace models, we use the model name directly
        # mlx-lm handles the downloading and conversion
        return MLXModelFactory.create_adapter(model_path=model_name, **kwargs)


# Convenience functions for common models
async def load_llama_mlx(
    model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit", **kwargs
) -> MLXLanguageModelAdapter:
    """Load a Llama model with MLX.

    Args:
        model_path: Path or HuggingFace model name
        **kwargs: Additional configuration

    Returns:
        Configured MLX adapter
    """
    adapter = MLXModelFactory.create_from_huggingface(model_path, **kwargs)
    await adapter.load_model()
    return adapter


async def load_mistral_mlx(
    model_path: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit", **kwargs
) -> MLXLanguageModelAdapter:
    """Load a Mistral model with MLX.

    Args:
        model_path: Path or HuggingFace model name
        **kwargs: Additional configuration

    Returns:
        Configured MLX adapter
    """
    adapter = MLXModelFactory.create_from_huggingface(model_path, **kwargs)
    await adapter.load_model()
    return adapter
