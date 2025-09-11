"""T036-T040: Integration module for REER × DSPy × MLX components.

This module provides centralized integration services connecting:
- TraceStore with REER mining operations
- LM registry with provider routing
- Rate limiting with exponential backoff
- Structured logging across all components
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from ..plugins.lm_registry import get_registry
from .exceptions import (
    ExtractionError,
    ScoringError,
    TraceStoreError,
)
from .trace_store import REERTraceStore
from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig
from tools.ppl_eval import select_ppl_evaluator

# ============================================================================
# Rate Limiting and Backoff
# ============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    exponential_backoff_base: float = 2.0
    max_backoff_delay: float = 300.0  # 5 minutes
    jitter: bool = True


class RateLimiter:
    """Rate limiter with exponential backoff and jitter."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: list[float] = []
        self.consecutive_failures = 0
        self._lock = asyncio.Lock()

        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.RateLimiter")

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            await self._wait_for_capacity()
            await self._apply_backoff()

            # Record request time
            now = time.time()
            self.request_times.append(now)

            # Clean old request times (older than 1 hour)
            cutoff = now - 3600
            self.request_times = [t for t in self.request_times if t > cutoff]

    async def _wait_for_capacity(self) -> None:
        """Wait until we have capacity for another request."""
        now = time.time()

        # Check minute-based rate limit
        minute_cutoff = now - 60
        recent_requests = [t for t in self.request_times if t > minute_cutoff]

        if len(recent_requests) >= self.config.max_requests_per_minute:
            wait_time = 60 - (now - recent_requests[0])
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # Check hour-based rate limit
        hour_cutoff = now - 3600
        hour_requests = [t for t in self.request_times if t > hour_cutoff]

        if len(hour_requests) >= self.config.max_requests_per_hour:
            wait_time = 3600 - (now - hour_requests[0])
            if wait_time > 0:
                self.logger.warning(
                    f"Hourly rate limit reached, waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)

    async def _apply_backoff(self) -> None:
        """Apply exponential backoff if there have been consecutive failures."""
        if self.consecutive_failures > 0:
            delay = min(
                self.config.exponential_backoff_base**self.consecutive_failures,
                self.config.max_backoff_delay,
            )

            # Add jitter
            if self.config.jitter:
                import random

                delay *= 0.5 + 0.5 * random.random()

            self.logger.info(
                f"Applying backoff delay {delay:.2f}s "
                f"(failures: {self.consecutive_failures})"
            )
            await asyncio.sleep(delay)

    def record_success(self) -> None:
        """Record a successful request to reset backoff."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request to increase backoff."""
        self.consecutive_failures += 1
        self.logger.warning(
            f"Request failure recorded (consecutive: {self.consecutive_failures})"
        )


# ============================================================================
# Structured Logging Configuration
# ============================================================================


@dataclass
class LoggingConfig:
    """Configuration for structured logging."""

    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    include_trace_id: bool = True
    include_performance_metrics: bool = True
    log_file: Path | None = None
    max_log_size_mb: int = 100
    backup_count: int = 5


class StructuredLogger:
    """Structured logger with trace correlation and performance metrics."""

    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure the logger with structured output."""
        self.logger.setLevel(getattr(logging, self.config.level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        if self.config.format == "json":
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_text_formatter()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if self.config.log_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=timezone.utc
                    ).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                # Add trace ID if available
                if hasattr(record, "trace_id"):
                    log_entry["trace_id"] = record.trace_id

                # Add performance metrics if available
                if hasattr(record, "duration_ms"):
                    log_entry["duration_ms"] = record.duration_ms

                # Add any extra fields
                if hasattr(record, "extra_fields"):
                    log_entry.update(record.extra_fields)

                return json.dumps(log_entry)

        return JSONFormatter()

    def _create_text_formatter(self) -> logging.Formatter:
        """Create text formatter for human-readable logging."""
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d"

        if self.config.include_trace_id:
            format_str += " | trace_id=%(trace_id)s"

        format_str += " - %(message)s"

        return logging.Formatter(format_str)

    def log_with_context(
        self,
        level: str,
        message: str,
        trace_id: str | None = None,
        duration_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Log with additional context."""
        extra = {}

        if trace_id and self.config.include_trace_id:
            extra["trace_id"] = trace_id

        if duration_ms and self.config.include_performance_metrics:
            extra["duration_ms"] = duration_ms

        if kwargs:
            extra["extra_fields"] = kwargs

        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, message, extra=extra)


# ============================================================================
# Integrated REER Mining Service
# ============================================================================


@dataclass
class REERMiningConfig:
    """Configuration for integrated REER mining."""

    trace_store_path: Path
    schema_path: Path | None = None
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    default_provider_uri: str = "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit"
    backup_enabled: bool = True
    validate_traces: bool = True


class IntegratedREERMiner:
    """Integrated REER mining service with TraceStore, LM registry, and rate limiting."""

    def __init__(self, config: REERMiningConfig):
        self.config = config

        # Initialize components
        self.trace_store = REERTraceStore(
            file_path=config.trace_store_path,
            schema_path=config.schema_path,
            backup_enabled=config.backup_enabled,
            validate_on_write=config.validate_traces,
        )

        self.lm_registry = get_registry()
        self.rate_limiter = RateLimiter(config.rate_limit_config)

        # Setup structured logging
        self.logger = StructuredLogger(
            f"{__name__}.IntegratedREERMiner", config.logging_config
        )

        # Performance tracking
        self._operation_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_storage_operations": 0,
            "rate_limit_hits": 0,
        }

    async def synthesize_trajectory(
        self,
        x: str,
        y: str,
        output_jsonl: Path,
        *,
        auto: str = "light",
        backend: str = "mlx",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run local-search REER synthesis for a single (x,y) and append to JSONL.

        Args:
            x: Query/prompt text
            y: Reference answer/output text
            output_jsonl: Path to output JSONL file
            auto: Budget preset (light|medium|heavy)
            mlx_model: Optional MLX model name for real PPL; proxy if None

        Returns: Result dict with z segments and ppl stats
        """
        if auto == "light":
            cfg = TrajectorySearchConfig(max_iters=6, max_candidates_per_segment=3)
        elif auto == "medium":
            cfg = TrajectorySearchConfig(max_iters=10, max_candidates_per_segment=4)
        else:
            cfg = TrajectorySearchConfig(max_iters=14, max_candidates_per_segment=5)

        if not model:
            raise ValueError("model must be provided for backend 'mlx' or 'together'")
        ppl = select_ppl_evaluator(backend, model)
        search = TrajectorySearch(ppl, cfg)
        result = search.search(x, y)

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    async def extract_and_store(
        self,
        source_post_id: str,
        content: str,
        seed_params: dict[str, Any],
        provider_uri: str | None = None,
        trace_id: str | None = None,
    ) -> str:
        """Extract strategy from content and store in TraceStore.

        Args:
            source_post_id: ID of the source post
            content: Content to extract strategy from
            seed_params: Parameters that generated the strategy
            provider_uri: LM provider URI (uses default if None)
            trace_id: Optional trace ID for correlation

        Returns:
            Trace ID of stored result

        Raises:
            ExtractionError: If strategy extraction fails
            TraceStoreError: If storage fails
        """
        # Generate trace ID if not provided
        if not trace_id:
            trace_id = str(uuid4())

        start_time = time.time()

        try:
            self.logger.log_with_context(
                "INFO",
                f"Starting extraction for post {source_post_id}",
                trace_id=trace_id,
                source_post_id=source_post_id,
            )

            # Use default provider if none specified
            if not provider_uri:
                provider_uri = self.config.default_provider_uri

            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Extract strategy using LM registry
            strategy_features, confidence = await self._extract_strategy(
                content, provider_uri, trace_id
            )

            # Calculate mock performance score
            score = await self._calculate_performance_score(
                content, strategy_features, provider_uri, trace_id
            )

            # Create trace record
            trace_data = {
                "id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_post_id": source_post_id,
                "seed_params": seed_params,
                "score": score,
                "metrics": {
                    "impressions": 1000,  # Mock data
                    "engagement_rate": score * 100,
                    "retweets": int(score * 50),
                    "likes": int(score * 200),
                },
                "strategy_features": strategy_features,
                "provider": provider_uri,
                "metadata": {
                    "extraction_method": "llm_analysis",
                    "confidence": confidence,
                },
            }

            # Store in TraceStore
            stored_trace_id = await self.trace_store.append_trace(trace_data)

            # Record success
            self.rate_limiter.record_success()
            self._operation_stats["successful_extractions"] += 1

            # Log completion
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "INFO",
                "Successfully extracted and stored strategy",
                trace_id=trace_id,
                duration_ms=duration_ms,
                score=score,
                features_count=len(strategy_features),
            )

            return stored_trace_id

        except Exception as e:
            # Record failure
            self.rate_limiter.record_failure()
            self._operation_stats["failed_extractions"] += 1

            # Log error
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "ERROR",
                f"Extraction failed: {e}",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
            )

            if isinstance(e, TraceStoreError | ExtractionError):
                raise
            raise ExtractionError(
                f"Extraction failed for post {source_post_id}: {e}",
                details={"source_post_id": source_post_id, "trace_id": trace_id},
                original_error=e,
            )

        finally:
            self._operation_stats["total_extractions"] += 1

    async def _extract_strategy(
        self, content: str, provider_uri: str, trace_id: str
    ) -> tuple[list[str], float]:
        """Extract strategy features from content using LM provider."""

        # Create extraction prompt
        prompt = f"""
        Analyze the following social media content and extract key strategy features:

        Content: {content}

        Identify the main strategic elements such as:
        - Content type (educational, entertaining, promotional, etc.)
        - Tone (casual, professional, humorous, etc.)
        - Engagement tactics (questions, calls-to-action, etc.)
        - Target audience indicators
        - Timing/trending elements

        Respond with a JSON list of strategy features and a confidence score (0.0-1.0).
        """

        try:
            # Generate using LM registry
            await self.lm_registry.route_generate(
                provider_uri, prompt, max_tokens=200, temperature=0.3
            )

            # Parse response (mock parsing for now)
            strategy_features = [
                "conversational_tone",
                "question_engagement",
                "educational_content",
                "trend_awareness",
            ]

            confidence = 0.85  # Mock confidence

            return strategy_features, confidence

        except Exception as e:
            raise ExtractionError(
                f"Strategy extraction failed: {e}",
                details={"provider_uri": provider_uri, "trace_id": trace_id},
                original_error=e,
            )

    async def _calculate_performance_score(
        self,
        content: str,
        strategy_features: list[str],
        provider_uri: str,
        trace_id: str,
    ) -> float:
        """Calculate performance score for the content."""

        try:
            # Calculate perplexity
            perplexity = await self.lm_registry.route_perplexity(provider_uri, content)

            # Mock scoring logic
            base_score = 1.0 / (1.0 + perplexity / 10.0)
            feature_bonus = len(strategy_features) * 0.05

            return min(1.0, base_score + feature_bonus)

        except Exception as e:
            raise ScoringError(
                f"Performance scoring failed: {e}",
                details={"provider_uri": provider_uri, "trace_id": trace_id},
                original_error=e,
            )

    async def query_traces(
        self,
        provider: str | None = None,
        min_score: float | None = None,
        strategy_features: list[str] | None = None,
        limit: int | None = None,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query traces from the TraceStore with optional filters."""

        start_time = time.time()

        try:
            self.logger.log_with_context(
                "INFO",
                "Querying traces from TraceStore",
                trace_id=trace_id,
                filters={
                    "provider": provider,
                    "min_score": min_score,
                    "strategy_features": strategy_features,
                    "limit": limit,
                },
            )

            traces = await self.trace_store.query_traces(
                provider=provider,
                min_score=min_score,
                strategy_features=strategy_features,
                limit=limit,
            )

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "INFO",
                f"Retrieved {len(traces)} traces",
                trace_id=trace_id,
                duration_ms=duration_ms,
                result_count=len(traces),
            )

            return traces

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "ERROR",
                f"Trace query failed: {e}",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
            )
            raise

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the mining service."""

        # Calculate success rate
        total_extractions = self._operation_stats["total_extractions"]
        successful_extractions = self._operation_stats["successful_extractions"]

        success_rate = (
            successful_extractions / total_extractions if total_extractions > 0 else 0.0
        )

        # Get trace store stats
        store_stats = await self.trace_store.validate_store()

        # Get LM registry health
        registry_health = await self.lm_registry.health_check()

        return {
            "extraction_stats": self._operation_stats,
            "success_rate": success_rate,
            "trace_store_stats": store_stats,
            "lm_registry_health": registry_health,
            "rate_limiter": {
                "consecutive_failures": self.rate_limiter.consecutive_failures,
                "recent_requests": len(self.rate_limiter.request_times),
            },
        }

    @asynccontextmanager
    async def trace_context(self, operation_name: str):
        """Context manager for operation tracing."""
        trace_id = str(uuid4())
        start_time = time.time()

        self.logger.log_with_context(
            "INFO", f"Starting operation: {operation_name}", trace_id=trace_id
        )

        try:
            yield trace_id
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "ERROR",
                f"Operation {operation_name} failed: {e}",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
            )
            raise
        else:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "INFO",
                f"Operation {operation_name} completed successfully",
                trace_id=trace_id,
                duration_ms=duration_ms,
            )


# ============================================================================
# Factory Functions
# ============================================================================


def create_mining_service(
    trace_store_path: str | Path, **kwargs
) -> IntegratedREERMiner:
    """Factory function to create a configured mining service.

    Args:
        trace_store_path: Path to trace store file
        **kwargs: Additional configuration options

    Returns:
        Configured IntegratedREERMiner instance
    """
    config = REERMiningConfig(trace_store_path=Path(trace_store_path), **kwargs)

    return IntegratedREERMiner(config)


def setup_logging(
    level: str = "INFO", format: str = "json", log_file: Path | None = None
) -> LoggingConfig:
    """Setup global logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Log format ("json" or "text")
        log_file: Optional log file path

    Returns:
        LoggingConfig instance
    """
    return LoggingConfig(level=level, format=format, log_file=log_file)
