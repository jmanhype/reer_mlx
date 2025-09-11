"""T014: REER trace store (append-only JSONL) implementation.

Provides append-only storage for REER traces in JSONL format with
atomic writes, concurrent access safety, and efficient querying.
Supports validation against JSON schema and maintains data integrity.
"""

import json
import asyncio
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, AsyncIterator
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
import aiofiles
from pydantic import BaseModel, Field, validator
from jsonschema import validate, ValidationError as JSONSchemaValidationError

from .exceptions import TraceStoreError, ValidationError


class TraceRecord(BaseModel):
    """Pydantic model for REER trace records with validation."""

    id: str = Field(..., description="Unique trace identifier (UUID v4)")
    timestamp: str = Field(..., description="When trace was created (ISO 8601)")
    source_post_id: str = Field(..., description="Original post identifier")
    seed_params: Dict[str, Any] = Field(
        ..., description="Parameters that generated this strategy"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Performance score (0.0-1.0)")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    strategy_features: List[str] = Field(
        ..., min_items=1, description="Extracted strategy patterns"
    )
    provider: str = Field(
        ..., pattern=r"^(mlx|dspy)::.+$", description="LM provider used for extraction"
    )
    metadata: Dict[str, Any] = Field(..., description="Additional context")

    @validator("id")
    def validate_uuid(cls, v):
        """Validate UUID format."""
        try:
            UUID(v, version=4)
        except ValueError:
            raise ValueError("id must be a valid UUID v4")
        return v

    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate ISO 8601 timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("timestamp must be ISO 8601 format")
        return v

    @validator("seed_params")
    def validate_seed_params(cls, v):
        """Validate seed_params structure."""
        required_fields = {"topic", "style", "length", "thread_size"}
        if not all(field in v for field in required_fields):
            raise ValueError(f"seed_params must contain: {required_fields}")

        # Validate constraints
        if not isinstance(v.get("length"), int) or not (1 <= v["length"] <= 10000):
            raise ValueError("seed_params.length must be integer 1-10000")
        if not isinstance(v.get("thread_size"), int) or not (
            1 <= v["thread_size"] <= 25
        ):
            raise ValueError("seed_params.thread_size must be integer 1-25")

        return v

    @validator("metrics")
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
        if not isinstance(engagement_rate, (int, float)) or not (
            0.0 <= engagement_rate <= 100.0
        ):
            raise ValueError("metrics.engagement_rate must be number 0.0-100.0")

        return v

    @validator("metadata")
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        required_fields = {"extraction_method", "confidence"}
        if not all(field in v for field in required_fields):
            raise ValueError(f"metadata must contain: {required_fields}")

        confidence = v.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
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
        schema_path: Optional[Path] = None,
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
        self._schema_cache: Optional[Dict[str, Any]] = None

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load JSON schema for validation."""
        if not self.schema_path or not self.schema_path.exists():
            return None

        if self._schema_cache is None:
            async with aiofiles.open(self.schema_path, "r") as f:
                content = await f.read()
                self._schema_cache = json.loads(content)

        return self._schema_cache

    def _validate_trace(self, trace: Dict[str, Any]) -> None:
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
            )

    async def _validate_trace_async(self, trace: Dict[str, Any]) -> None:
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
                    )

    @asynccontextmanager
    async def _file_lock(self, mode: str = "a"):
        """Async context manager for file locking."""
        async with self._lock:
            try:
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
                )

    async def append_trace(self, trace: Dict[str, Any]) -> str:
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
            trace["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            async with self._file_lock("a") as f:
                # Write JSONL line
                json_line = json.dumps(trace, ensure_ascii=False)
                f.write(json_line + "\n")
                f.flush()

            return trace["id"]

        except Exception as e:
            raise TraceStoreError(
                f"Failed to append trace: {str(e)}",
                details={"trace_id": trace.get("id")},
                original_error=e,
            )

    async def append_traces(self, traces: List[Dict[str, Any]]) -> List[str]:
        """Append multiple traces atomically.

        Args:
            traces: List of trace dictionaries

        Returns:
            List[str]: List of trace IDs

        Raises:
            ValidationError: If any trace validation fails
            TraceStoreError: If storage operation fails
        """
        # Validate all traces first
        for i, trace in enumerate(traces):
            try:
                await self._validate_trace_async(trace)
            except ValidationError as e:
                e.details["trace_index"] = i
                raise

        # Ensure all traces have IDs and timestamps
        trace_ids = []
        for trace in traces:
            if "id" not in trace:
                trace["id"] = str(uuid4())
            if "timestamp" not in trace:
                trace["timestamp"] = datetime.now(timezone.utc).isoformat()
            trace_ids.append(trace["id"])

        try:
            async with self._file_lock("a") as f:
                for trace in traces:
                    json_line = json.dumps(trace, ensure_ascii=False)
                    f.write(json_line + "\n")
                f.flush()

            return trace_ids

        except Exception as e:
            raise TraceStoreError(
                f"Failed to append traces: {str(e)}",
                details={"trace_count": len(traces)},
                original_error=e,
            )

    async def get_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
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
        provider: Optional[str] = None,
        source_post_id: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        strategy_features: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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

    async def iter_traces(self) -> AsyncIterator[Dict[str, Any]]:
        """Async iterator over all traces in the store."""
        if not self.file_path.exists():
            return

        try:
            async with aiofiles.open(self.file_path, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue
        except Exception as e:
            raise TraceStoreError(
                f"Failed to read traces: {str(e)}",
                details={"file_path": str(self.file_path)},
                original_error=e,
            )

    def iter_traces_sync(self) -> Iterator[Dict[str, Any]]:
        """Synchronous iterator over all traces in the store."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise TraceStoreError(
                f"Failed to read traces: {str(e)}",
                details={"file_path": str(self.file_path)},
                original_error=e,
            )

    async def count_traces(self) -> int:
        """Count total number of traces in the store."""
        count = 0
        async for _ in self.iter_traces():
            count += 1
        return count

    async def get_latest_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
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
        try:
            traces.sort(
                key=lambda t: datetime.fromisoformat(
                    t.get("timestamp", "").replace("Z", "+00:00")
                ),
                reverse=True,
            )
        except (ValueError, TypeError):
            # If timestamp parsing fails, return in file order
            pass

        return traces[:limit]

    async def backup(self, backup_path: Optional[Path] = None) -> Path:
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
                async with aiofiles.open(self.file_path, "r") as src:
                    async with aiofiles.open(backup_path, "w") as dst:
                        async for line in src:
                            await dst.write(line)

            return backup_path

        except Exception as e:
            raise TraceStoreError(
                f"Failed to create backup: {str(e)}",
                details={"backup_path": str(backup_path)},
                original_error=e,
            )

    async def validate_store(self) -> Dict[str, Any]:
        """Validate the entire trace store.

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
            async for trace in self.iter_traces():
                stats["total_traces"] += 1

                try:
                    await self._validate_trace_async(trace)
                    stats["valid_traces"] += 1
                except ValidationError as e:
                    stats["invalid_traces"] += 1
                    stats["validation_errors"].append(
                        {"trace_id": trace.get("id", "unknown"), "error": str(e)}
                    )

        except Exception as e:
            raise TraceStoreError(
                f"Failed to validate store: {str(e)}", original_error=e
            )

        return stats
