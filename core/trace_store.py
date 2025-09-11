"""T014: REER trace store (append-only JSONL) implementation.

Provides append-only storage for REER traces in JSONL format with
atomic writes, concurrent access safety, and efficient querying.
Supports validation against JSON schema and maintains data integrity.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
import fcntl
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiofiles
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, Field, validator

from tools.memory_profiler import (
    StreamingJSONLReader,
    check_memory_limit,
    memory_profile,
)

from .exceptions import TraceStoreError, ValidationError

logger = logging.getLogger(__name__)

# Constants
MAX_SEED_LENGTH = 10_000
MAX_THREAD_SIZE = 25
ENGAGEMENT_RATE_MAX = 100.0
PROCESS_MEM_MB_HIGH = 500


class TraceRecord(BaseModel):
    """Pydantic model for REER trace records with validation."""

    id: str = Field(..., description="Unique trace identifier (UUID v4)")
    timestamp: str = Field(..., description="When trace was created (ISO 8601)")
    source_post_id: str = Field(..., description="Original post identifier")
    seed_params: dict[str, Any] = Field(
        ..., description="Parameters that generated this strategy"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Performance score (0.0-1.0)")
    metrics: dict[str, Any] = Field(..., description="Performance metrics")
    strategy_features: list[str] = Field(
        ..., min_items=1, description="Extracted strategy patterns"
    )
    provider: str = Field(
        ..., pattern=r"^(mlx|dspy)::.+$", description="LM provider used for extraction"
    )
    metadata: dict[str, Any] = Field(..., description="Additional context")

    @validator("id")
    @classmethod
    def validate_uuid(cls, v):
        """Validate UUID format."""
        try:
            UUID(v, version=4)
        except ValueError:
            raise ValueError("id must be a valid UUID v4") from None
        return v

    @validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("timestamp must be ISO 8601 format") from None
        return v

    @validator("seed_params")
    @classmethod
    def validate_seed_params(cls, v):
        """Validate seed_params structure."""
        required_fields = {"topic", "style", "length", "thread_size"}
        if not all(field in v for field in required_fields):
            raise ValueError(f"seed_params must contain: {required_fields}")

        # Validate constraints
        if not isinstance(v.get("length"), int) or not (
            1 <= v["length"] <= MAX_SEED_LENGTH
        ):
            raise ValueError("seed_params.length must be integer 1-10000")
        if not isinstance(v.get("thread_size"), int) or not (
            1 <= v["thread_size"] <= MAX_THREAD_SIZE
        ):
            raise ValueError("seed_params.thread_size must be integer 1-25")

        return v

    @validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        """Validate metrics structure."""
        required_fields = {"impressions", "engagement_rate", "retweets", "likes"}
        if not all(field in v for field in required_fields):
            raise ValueError(f"metrics must contain: {required_fields}")

        # Validate constraints
        for field in ["impressions", "retweets", "likes"]:
            if not isinstance(v.get(field), int) or v[field] < 0:
                raise ValueError(f"metrics.{field} must be non-negative integer")

        engagement_rate = v.get("engagement_rate")
        if not isinstance(engagement_rate, int | float) or not (
            0.0 <= engagement_rate <= ENGAGEMENT_RATE_MAX
        ):
            raise ValueError("metrics.engagement_rate must be number 0.0-100.0")

        return v

    @validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        required_fields = {"extraction_method", "confidence"}
        if not all(field in v for field in required_fields):
            raise ValueError(f"metadata must contain: {required_fields}")

        confidence = v.get("confidence")
        if not isinstance(confidence, int | float) or not (0.0 <= confidence <= 1.0):
            raise ValueError("metadata.confidence must be number 0.0-1.0")

        return v


class REERTraceStore:
    """Append-only JSONL storage for REER traces.

    Features:
    - Atomic writes with file locking
    - Schema validation on write
    - Efficient append operations
    - Query by various criteria
    - Concurrent access safety
    - Automatic backup creation
    """

    def __init__(
        self,
        file_path: Path,
        schema_path: Path | None = None,
        backup_enabled: bool = True,
        validate_on_write: bool = True,
    ):
        """Initialize trace store.

        Args:
            file_path: Path to JSONL trace file
            schema_path: Path to JSON schema for validation
            backup_enabled: Whether to create backup files
            validate_on_write: Whether to validate traces on write
        """
        self.file_path = Path(file_path)
        self.schema_path = Path(schema_path) if schema_path else None
        self.backup_enabled = backup_enabled
        self.validate_on_write = validate_on_write
        self._lock = asyncio.Lock()
        self._schema_cache: dict[str, Any] | None = None

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def _load_schema(self) -> dict[str, Any] | None:
        """Load JSON schema for validation."""
        if not self.schema_path or not self.schema_path.exists():
            return None

        if self._schema_cache is None:
            async with aiofiles.open(self.schema_path) as f:
                content = await f.read()
                self._schema_cache = json.loads(content)

        return self._schema_cache

    def _validate_trace(self, trace: dict[str, Any]) -> None:
        """Validate trace against Pydantic model and JSON schema.

        Args:
            trace: Trace dictionary to validate

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Pydantic validation
            TraceRecord(**trace)
        except Exception as e:
            raise ValidationError(
                f"Pydantic validation failed: {str(e)}",
                details={"trace_id": trace.get("id"), "error": str(e)},
            ) from e

    async def _validate_trace_async(self, trace: dict[str, Any]) -> None:
        """Async validation including JSON schema if available."""
        # Pydantic validation (synchronous)
        self._validate_trace(trace)

        # JSON schema validation (if schema available)
        if self.validate_on_write:
            schema = await self._load_schema()
            if schema:
                try:
                    validate(trace, schema)
                except JSONSchemaValidationError as e:
                    raise ValidationError(
                        f"JSON schema validation failed: {e.message}",
                        details={
                            "trace_id": trace.get("id"),
                            "schema_path": str(e.schema_path),
                            "instance_path": str(e.absolute_path),
                        },
                        original_error=e,
                    ) from e

    @asynccontextmanager
    async def _file_lock(self, mode: str = "a"):
        """Async context manager for file locking."""
        async with self._lock:
            try:
                # Check memory limits before file operations
                check_memory_limit()

                # Use synchronous file operations for locking
                # aiofiles doesn't support fcntl operations
                with open(self.file_path, mode) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    yield f
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                raise TraceStoreError(
                    f"File lock error: {str(e)}",
                    details={"file_path": str(self.file_path), "mode": mode},
                    original_error=e,
                ) from e

    @memory_profile(operation_name="append_trace")
    async def append_trace(self, trace: dict[str, Any]) -> str:
        """Append a single trace to the store.

        Args:
            trace: Trace dictionary to append

        Returns:
            str: Trace ID

        Raises:
            ValidationError: If trace validation fails
            TraceStoreError: If storage operation fails
        """
        # Validate trace
        await self._validate_trace_async(trace)

        # Ensure trace has ID and timestamp
        if "id" not in trace:
            trace["id"] = str(uuid4())
        if "timestamp" not in trace:
            trace["timestamp"] = datetime.now(UTC).isoformat()

        try:
            async with self._file_lock("a") as f:
                # Write JSONL line
                json_line = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
                f.write(json_line + "\n")
                f.flush()

        except Exception as e:
            raise TraceStoreError(
                f"Failed to append trace: {str(e)}",
                details={"trace_id": trace.get("id")},
                original_error=e,
            ) from e
        else:
            return trace["id"]

    @memory_profile(operation_name="append_traces")
    async def append_traces(self, traces: list[dict[str, Any]]) -> list[str]:
        """Append multiple traces atomically with memory optimization.

        Args:
            traces: List of trace dictionaries

        Returns:
            List[str]: List of trace IDs

        Raises:
            ValidationError: If any trace validation fails
            TraceStoreError: If storage operation fails
        """

        # Use generator for memory-efficient validation
        async def validate_traces_generator():
            for i, trace in enumerate(traces):
                try:
                    await self._validate_trace_async(trace)
                    yield trace, i
                except ValidationError as e:
                    e.details["trace_index"] = i
                    raise

        # Process traces using generator to reduce memory footprint
        trace_ids = []
        validated_traces = []

        async for trace, _index in validate_traces_generator():
            # Ensure trace has ID and timestamp
            if "id" not in trace:
                trace["id"] = str(uuid4())
            if "timestamp" not in trace:
                trace["timestamp"] = datetime.now(UTC).isoformat()

            trace_ids.append(trace["id"])
            validated_traces.append(trace)

            # Check memory usage every 50 traces
            if len(validated_traces) % 50 == 0:
                check_memory_limit()

        try:
            async with self._file_lock("a") as f:
                # Use adaptive chunk sizing based on available memory
                base_chunk_size = 50
                from tools.memory_profiler import get_memory_tracker

                current_memory = get_memory_tracker().get_current_snapshot()

                # Reduce chunk size if memory usage is high
                if current_memory.process_memory_mb > PROCESS_MEM_MB_HIGH:
                    chunk_size = base_chunk_size // 2
                else:
                    chunk_size = base_chunk_size

                # Process traces in chunks to limit memory usage
                for i in range(0, len(validated_traces), chunk_size):
                    chunk = validated_traces[i : i + chunk_size]

                    # Use generator to process chunk
                    json_lines = (
                        json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                        for trace in chunk
                    )

                    # Write chunk
                    for line in json_lines:
                        f.write(line)

                    f.flush()
                    check_memory_limit()

                    # Clear processed chunk from memory
                    del chunk

        except Exception as e:
            raise TraceStoreError(
                f"Failed to append traces: {str(e)}",
                details={"trace_count": len(traces)},
                original_error=e,
            ) from e
        else:
            return trace_ids

    async def get_trace_by_id(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by its ID.

        Args:
            trace_id: The trace ID to find

        Returns:
            Dict or None if not found
        """
        async for trace in self.iter_traces():
            if trace.get("id") == trace_id:
                return trace
        return None

    async def query_traces(
        self,
        provider: str | None = None,
        source_post_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        strategy_features: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query traces with filtering criteria.

        Args:
            provider: Filter by provider pattern
            source_post_id: Filter by source post ID
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            strategy_features: Required strategy features (all must be present)
            since: Filter traces after this timestamp
            until: Filter traces before this timestamp
            limit: Maximum number of results

        Returns:
            List of matching traces
        """
        results = []
        count = 0

        async for trace in self.iter_traces():
            # Apply filters
            if provider and not trace.get("provider", "").startswith(provider):
                continue

            if source_post_id and trace.get("source_post_id") != source_post_id:
                continue

            if min_score is not None and trace.get("score", 0) < min_score:
                continue

            if max_score is not None and trace.get("score", 1) > max_score:
                continue

            if strategy_features:
                trace_features = set(trace.get("strategy_features", []))
                if not all(feature in trace_features for feature in strategy_features):
                    continue

            # Timestamp filtering
            if since or until:
                try:
                    trace_time = datetime.fromisoformat(
                        trace.get("timestamp", "").replace("Z", "+00:00")
                    )
                    if since and trace_time < since:
                        continue
                    if until and trace_time > until:
                        continue
                except ValueError:
                    continue  # Skip traces with invalid timestamps

            results.append(trace)
            count += 1

            if limit and count >= limit:
                break

        return results

    async def iter_traces(self) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over all traces in the store."""
        if not self.file_path.exists():
            return

        try:
            # Use memory-efficient streaming reader
            reader = StreamingJSONLReader(self.file_path)
            async for trace in reader.aiter():
                yield trace
        except Exception as e:
            raise TraceStoreError(
                f"Failed to read traces: {str(e)}",
                details={"file_path": str(self.file_path)},
                original_error=e,
            ) from e

    def iter_traces_sync(self) -> Iterator[dict[str, Any]]:
        """Synchronous iterator over all traces in the store."""
        if not self.file_path.exists():
            return

        try:
            # Use memory-efficient streaming reader
            reader = StreamingJSONLReader(self.file_path)
            yield from reader
        except Exception as e:
            raise TraceStoreError(
                f"Failed to read traces: {str(e)}",
                details={"file_path": str(self.file_path)},
                original_error=e,
            ) from e

    async def count_traces(self) -> int:
        """Count total number of traces in the store."""
        count = 0
        async for _ in self.iter_traces():
            count += 1
        return count

    async def get_latest_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recently added traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of traces ordered by timestamp (newest first)
        """
        traces = []
        async for trace in self.iter_traces():
            traces.append(trace)

        # Sort by timestamp (newest first)
        with suppress(ValueError, TypeError):
            traces.sort(
                key=lambda t: datetime.fromisoformat(
                    t.get("timestamp", "").replace("Z", "+00:00")
                ),
                reverse=True,
            )

        return traces[:limit]

    async def backup(self, backup_path: Path | None = None) -> Path:
        """Create a backup of the trace store.

        Args:
            backup_path: Custom backup path, or auto-generate if None

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.file_path.with_suffix(f".{timestamp}.backup.jsonl")

        try:
            if self.file_path.exists():
                async with (
                    aiofiles.open(self.file_path) as src,
                    aiofiles.open(backup_path, "w") as dst,
                ):
                    async for line in src:
                        await dst.write(line)

        except Exception as e:
            raise TraceStoreError(
                f"Failed to create backup: {str(e)}",
                details={"backup_path": str(backup_path)},
                original_error=e,
            ) from e
        else:
            return backup_path

    async def validate_store(self) -> dict[str, Any]:
        """Validate the entire trace store with memory optimization.

        Returns:
            Dict with validation results and statistics
        """
        stats = {
            "total_traces": 0,
            "valid_traces": 0,
            "invalid_traces": 0,
            "validation_errors": [],
            "file_exists": self.file_path.exists(),
            "file_size_bytes": 0,
        }

        if not self.file_path.exists():
            return stats

        stats["file_size_bytes"] = self.file_path.stat().st_size

        try:
            # Use batched validation to reduce memory usage
            batch_size = 100
            current_batch = []

            async for trace in self.iter_traces():
                current_batch.append(trace)
                stats["total_traces"] += 1

                # Process batch when it reaches the size limit
                if len(current_batch) >= batch_size:
                    await self._validate_batch(current_batch, stats)
                    current_batch.clear()
                    check_memory_limit()

            # Process remaining traces
            if current_batch:
                await self._validate_batch(current_batch, stats)

        except Exception as e:
            raise TraceStoreError(
                f"Failed to validate store: {str(e)}", original_error=e
            ) from e

        return stats

    async def _validate_batch(
        self, traces: list[dict[str, Any]], stats: dict[str, Any]
    ) -> None:
        """Validate a batch of traces."""
        for trace in traces:
            try:
                await self._validate_trace_async(trace)
                stats["valid_traces"] += 1
            except ValidationError as e:
                stats["invalid_traces"] += 1
                stats["validation_errors"].append(
                    {"trace_id": trace.get("id", "unknown"), "error": str(e)}
                )

    def optimize_memory_usage(self) -> dict[str, Any]:
        """Optimize memory usage for the trace store.

        Returns:
            Dict with optimization results
        """
        from tools.memory_profiler import MemoryOptimizer, get_memory_tracker

        initial_memory = get_memory_tracker().get_current_snapshot()

        # Clear schema cache
        self._schema_cache = None

        # Force garbage collection
        collected = MemoryOptimizer.force_garbage_collection()

        # Clean up large objects
        cleaned = MemoryOptimizer.cleanup_large_objects()

        final_memory = get_memory_tracker().get_current_snapshot()

        return {
            "initial_memory_mb": initial_memory.process_memory_mb,
            "final_memory_mb": final_memory.process_memory_mb,
            "memory_freed_mb": initial_memory.process_memory_mb
            - final_memory.process_memory_mb,
            "objects_collected": sum(collected.values()),
            "objects_cleaned": cleaned,
            "optimization_timestamp": datetime.now(UTC).isoformat(),
        }

    async def stream_traces_filtered(
        self,
        filter_func: Callable[[dict[str, Any]], bool],
        chunk_size: int = 1000,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Stream traces in chunks with filtering to optimize memory usage.

        Args:
            filter_func: Function to filter traces
            chunk_size: Number of traces per chunk

        Yields:
            Lists of filtered traces
        """
        chunk = []

        async for trace in self.iter_traces():
            try:
                if filter_func(trace):
                    chunk.append(trace)

                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                        check_memory_limit()

            except Exception as e:
                logger.warning(
                    f"Error filtering trace {trace.get('id', 'unknown')}: {e}"
                )
                continue

        # Yield remaining traces
        if chunk:
            yield chunk

    async def get_memory_efficient_stats(self) -> dict[str, Any]:
        """Get store statistics with minimal memory usage."""
        stats = {
            "total_traces": 0,
            "file_size_bytes": 0,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "providers": set(),
            "memory_usage_mb": 0,
        }

        if not self.file_path.exists():
            return stats

        stats["file_size_bytes"] = self.file_path.stat().st_size

        # Get initial memory reading
        from tools.memory_profiler import get_memory_tracker

        initial_memory = get_memory_tracker().get_current_snapshot()

        # Stream traces with minimal memory footprint
        batch_size = 50
        current_batch_count = 0

        async for trace in self.iter_traces():
            stats["total_traces"] += 1
            current_batch_count += 1

            # Track timestamp range
            timestamp = trace.get("timestamp")
            if timestamp:
                if (
                    not stats["earliest_timestamp"]
                    or timestamp < stats["earliest_timestamp"]
                ):
                    stats["earliest_timestamp"] = timestamp
                if (
                    not stats["latest_timestamp"]
                    or timestamp > stats["latest_timestamp"]
                ):
                    stats["latest_timestamp"] = timestamp

            # Track providers
            provider = trace.get("provider")
            if provider:
                stats["providers"].add(provider)

            # Check memory every batch
            if current_batch_count >= batch_size:
                check_memory_limit()
                current_batch_count = 0

        # Convert sets to lists for JSON serialization
        stats["providers"] = list(stats["providers"])

        # Calculate memory usage
        final_memory = get_memory_tracker().get_current_snapshot()
        stats["memory_usage_mb"] = (
            final_memory.process_memory_mb - initial_memory.process_memory_mb
        )

        return stats

    @memory_profile(operation_name="lazy_query_traces")
    async def lazy_query_traces(
        self,
        provider: str | None = None,
        source_post_id: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        strategy_features: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        yield_size: int = 100,
    ) -> AsyncIterator[dict[str, Any]]:
        """Lazy query traces with memory-efficient streaming.

        Args:
            (same as query_traces)
            yield_size: Number of results to yield at once

        Yields:
            Individual matching traces
        """
        count = 0
        batch = []

        async for trace in self.iter_traces():
            # Apply filters (same logic as query_traces)
            if provider and not trace.get("provider", "").startswith(provider):
                continue

            if source_post_id and trace.get("source_post_id") != source_post_id:
                continue

            if min_score is not None and trace.get("score", 0) < min_score:
                continue

            if max_score is not None and trace.get("score", 1) > max_score:
                continue

            if strategy_features:
                trace_features = set(trace.get("strategy_features", []))
                if not all(feature in trace_features for feature in strategy_features):
                    continue

            # Timestamp filtering
            if since or until:
                try:
                    trace_time = datetime.fromisoformat(
                        trace.get("timestamp", "").replace("Z", "+00:00")
                    )
                    if since and trace_time < since:
                        continue
                    if until and trace_time > until:
                        continue
                except ValueError:
                    continue

            batch.append(trace)
            count += 1

            # Yield in batches to control memory
            if len(batch) >= yield_size:
                for result in batch:
                    yield result
                batch.clear()
                check_memory_limit()

            if limit and count >= limit:
                break

        # Yield remaining results
        for result in batch:
            yield result
