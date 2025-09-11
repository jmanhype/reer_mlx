"""T018: MLX language model adapter implementation.

Provides MLX integration for Apple Silicon optimization with efficient
inference and memory management. Supports local model loading and
generation with Apple's MLX framework.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import threading
import time
from typing import Any

try:
    from mlx import nn
    import mlx.core as mx
    from mlx_lm import generate, load, stream_generate
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

import contextlib

from core.exceptions import ScoringError, ValidationError
from tools.memory_profiler import LazyModelLoader, check_memory_limit, memory_profile

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for MLX inference."""

    inference_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    batch_size: int = 1
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0


class ModelCache:
    """Thread-safe model cache for MLX adapters."""

    def __init__(self, max_size: int = 3):
        self._cache: dict[str, tuple[Any, Any]] = {}  # model_path -> (model, tokenizer)
        self._access_times: dict[str, float] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, model_path: str) -> tuple[Any, Any] | None:
        """Get cached model and tokenizer."""
        with self._lock:
            if model_path in self._cache:
                self._access_times[model_path] = time.time()
                return self._cache[model_path]
            return None

    def set(self, model_path: str, model: Any, tokenizer: Any) -> None:
        """Cache model and tokenizer."""
        with self._lock:
            # Remove oldest entry if cache is full
            if len(self._cache) >= self._max_size and model_path not in self._cache:
                oldest_path = min(
                    self._access_times.keys(), key=lambda k: self._access_times[k]
                )
                del self._cache[oldest_path]
                del self._access_times[oldest_path]

            self._cache[model_path] = (model, tokenizer)
            self._access_times[model_path] = time.time()

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class TokenizerCache:
    """Cache for pre-computed tokenizations."""

    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, list[int]] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    @lru_cache(maxsize=1000)
    def _tokenize_cached(self, text: str, tokenizer_id: str) -> tuple[int, ...]:
        """LRU cached tokenization helper."""
        # This is a placeholder - actual implementation would use the tokenizer
        return ()

    def get_or_tokenize(self, text: str, tokenizer, tokenizer_id: str) -> list[int]:
        """Get cached tokens or tokenize and cache."""
        cache_key = f"{tokenizer_id}:{hash(text)}"

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Tokenize
        tokens = tokenizer.encode(text)

        with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[cache_key] = tokens

        return tokens


class PerformanceBenchmark:
    """Benchmark and profiling utilities for MLX inference."""

    @staticmethod
    def time_inference(func):
        """Decorator to time inference operations."""

        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()

            inference_time_ms = (end_time - start_time) * 1000
            logger.info(f"Inference completed in {inference_time_ms:.2f}ms")

            return result

        return wrapper

    @staticmethod
    async def benchmark_model(
        adapter: "MLXLanguageModelAdapter",
        test_prompts: list[str] = None,
        num_runs: int = 10,
    ) -> dict[str, float]:
        """Benchmark model performance with multiple prompts."""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about technology.",
            ]

        total_times = []
        total_tokens = 0

        for _ in range(num_runs):
            for prompt in test_prompts:
                start_time = time.perf_counter()
                result = await adapter.generate(prompt, max_tokens=50)
                end_time = time.perf_counter()

                inference_time = end_time - start_time
                total_times.append(inference_time)
                total_tokens += len(result.split())

        avg_time_ms = (sum(total_times) / len(total_times)) * 1000
        tokens_per_second = total_tokens / sum(total_times)

        return {
            "avg_inference_time_ms": avg_time_ms,
            "tokens_per_second": tokens_per_second,
            "total_runs": len(total_times),
            "min_time_ms": min(total_times) * 1000,
            "max_time_ms": max(total_times) * 1000,
        }


# Global caches for performance optimization
_model_cache = ModelCache()
_tokenizer_cache = TokenizerCache()


@dataclass
class MLXGenerationConfig:
    """Configuration for MLX model generation."""

    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    repetition_context_size: int = 20
    seed: int | None = None
    stop_tokens: list[str] = field(default_factory=list)
    # Performance optimization settings
    enable_batch_processing: bool = True
    batch_size: int = 8
    enable_kv_cache: bool = True
    enable_flash_attention: bool = True


@dataclass
class MLXModelConfig:
    """Configuration for MLX model loading."""

    model_path: str
    tokenizer_config: dict[str, Any] | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    dtype: str = "float16"  # float16, bfloat16, float32
    max_context_length: int = 4096
    # Quantization and optimization settings
    quantization: str | None = None  # "int8", "int4", None
    enable_model_caching: bool = True
    warmup_prompts: list[str] = field(
        default_factory=lambda: ["Hello, world!", "What is AI?"]
    )
    precompute_tokenization: bool = True


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
        generation_config: MLXGenerationConfig | None = None,
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

        # Use lazy loading for the model to save memory
        self._lazy_model = LazyModelLoader(self._load_model_internal)
        self.tokenizer = None
        self._model_loaded = False
        self._warmed_up = False
        self._tokenizer_id = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._metrics = PerformanceMetrics()

        logger.info(f"Initializing MLX adapter for model: {model_config.model_path}")

    def _load_model_internal(self) -> tuple[Any, Any]:
        """Internal method for loading MLX model."""
        logger.info(f"Loading MLX model from {self.model_config.model_path}")

        # Check cache first if enabled
        if self.model_config.enable_model_caching:
            cached = _model_cache.get(self.model_config.model_path)
            if cached is not None:
                logger.info("MLX model loaded from cache")
                return cached

        # Prepare loading arguments
        load_kwargs = {
            "tokenizer_config": self.model_config.tokenizer_config,
            "trust_remote_code": self.model_config.trust_remote_code,
        }

        # Add quantization if specified
        if self.model_config.quantization:
            load_kwargs["quantization"] = self.model_config.quantization
            logger.info(f"Loading with {self.model_config.quantization} quantization")

        # Load model and tokenizer using mlx-lm
        model, tokenizer = load(self.model_config.model_path, **load_kwargs)

        # Apply additional optimizations
        if hasattr(model, "eval"):
            model.eval()

        # Cache the model if enabled
        if self.model_config.enable_model_caching:
            _model_cache.set(self.model_config.model_path, model, tokenizer)

        logger.info("MLX model loaded successfully")
        return model, tokenizer

    @memory_profile(operation_name="load_mlx_model")
    async def load_model(self) -> None:
        """Load the MLX model and tokenizer with caching and optimization."""
        if self._model_loaded:
            return

        try:
            check_memory_limit()

            # Trigger lazy loading
            self.model = self._lazy_model
            _, self.tokenizer = self._load_model_internal()

            self._model_loaded = True
            self._tokenizer_id = f"{type(self.tokenizer).__name__}_{id(self.tokenizer)}"

            # Perform warmup
            await self._warmup_model()

        except Exception as e:
            logger.exception(f"Failed to load MLX model: {e}")
            raise ValidationError(f"Model loading failed: {e}")

    async def _warmup_model(self) -> None:
        """Warm up the model with sample prompts for optimal performance."""
        if self._warmed_up or not self.model_config.warmup_prompts:
            return

        logger.info("Warming up model for optimal performance...")
        warmup_start = time.perf_counter()

        try:
            for prompt in self.model_config.warmup_prompts:
                # Run a quick inference to warm up the model
                _ = await self._generate_internal(
                    prompt, max_tokens=10, log_performance=False
                )

            self._warmed_up = True
            warmup_time = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Model warmup completed in {warmup_time:.2f}ms")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}, continuing anyway")

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> str:
        """Generate text using MLX model with performance optimizations.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        return await self._generate_internal(
            prompt, max_tokens, temperature, top_p, log_performance=True, **kwargs
        )

    @PerformanceBenchmark.time_inference
    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        log_performance: bool = True,
        **kwargs,
    ) -> str:
        """Internal generation method with performance tracking."""
        await self.load_model()

        start_time = time.perf_counter()

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
            # Optimize tokenization with caching
            if self.model_config.precompute_tokenization and self._tokenizer_id:
                tokens = _tokenizer_cache.get_or_tokenize(
                    prompt, self.tokenizer, self._tokenizer_id
                )
                len(tokens)
            else:
                len(self.tokenizer.encode(prompt))

            # Generate text using mlx-lm with MLX stream context for memory efficiency
            with mx.stream():
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    **gen_config,
                )

            # Extract generated text (remove prompt)
            if response.startswith(prompt):
                generated_text = response[len(prompt) :].strip()
            else:
                generated_text = response.strip()

            # Update performance metrics
            if log_performance:
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000
                generated_tokens = len(generated_text.split())

                self._metrics.inference_time_ms = inference_time_ms
                self._metrics.tokens_per_second = (
                    generated_tokens / (inference_time_ms / 1000)
                    if inference_time_ms > 0
                    else 0
                )

                if inference_time_ms < 50:
                    logger.info(
                        f"🚀 Fast inference achieved: {inference_time_ms:.2f}ms"
                    )
                elif inference_time_ms > 100:
                    logger.warning(
                        f"⚠️ Slow inference detected: {inference_time_ms:.2f}ms"
                    )

            return generated_text

        except Exception as e:
            logger.exception(f"MLX generation failed: {e}")
            raise ScoringError(f"Generation failed: {e}")

    async def generate_batch(
        self, prompts: list[str], max_tokens: int | None = None, **kwargs
    ) -> list[str]:
        """Generate text for multiple prompts in batches for improved throughput.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        if not self.generation_config.enable_batch_processing:
            # Fallback to sequential processing
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, max_tokens, **kwargs)
                results.append(result)
            return results

        await self.load_model()

        batch_size = self.generation_config.batch_size
        results = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_start_time = time.perf_counter()

            # Use ThreadPoolExecutor for parallel processing within batch
            batch_tasks = []
            for prompt in batch:
                task = asyncio.create_task(
                    self._generate_internal(
                        prompt, max_tokens, log_performance=False, **kwargs
                    )
                )
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions and collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch generation error: {result}")
                    results.append("")  # Empty string for failed generations
                else:
                    results.append(result)

            batch_time = (time.perf_counter() - batch_start_time) * 1000
            logger.info(f"Batch of {len(batch)} processed in {batch_time:.2f}ms")

        return results

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
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
            logger.exception(f"MLX streaming generation failed: {e}")
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
            logger.exception(f"Perplexity calculation failed: {e}")
            return float("inf")

    async def calculate_token_probabilities(
        self, prompt: str, continuation: str
    ) -> list[float]:
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
            logger.exception(f"Token probability calculation failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if MLX adapter is available."""
        return MLX_AVAILABLE and mx is not None

    async def get_model_info(self) -> dict[str, Any]:
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
            with contextlib.suppress(Exception):
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

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            Current performance metrics
        """
        return self._metrics

    async def benchmark_performance(self, num_runs: int = 10) -> dict[str, float]:
        """Run performance benchmark on the model.

        Args:
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        return await PerformanceBenchmark.benchmark_model(self, num_runs=num_runs)


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
                    "quantization",
                    "enable_model_caching",
                    "warmup_prompts",
                    "precompute_tokenization",
                ]
            },
        )

        generation_config = MLXGenerationConfig(
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "repetition_penalty",
                    "seed",
                    "enable_batch_processing",
                    "batch_size",
                    "enable_kv_cache",
                    "enable_flash_attention",
                ]
            }
        )

        return MLXLanguageModelAdapter(model_config, generation_config)

    @staticmethod
    def create_from_huggingface(
        model_name: str, cache_dir: str | None = None, **kwargs
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
