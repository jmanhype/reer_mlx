"""Memory profiling tools and optimization utilities.

Provides comprehensive memory tracking, profiling decorators, leak detection,
and resource monitoring utilities for the REER MLX system.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import functools
import gc
import json
import logging
from pathlib import Path
import signal
import sys
import threading
import traceback
from typing import Any, TypeVar
import weakref

import numpy as np
import psutil

# Type variables for generic decorators
F = TypeVar("F", bound=Callable)

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a point-in-time memory snapshot."""

    timestamp: datetime
    process_memory_mb: float
    system_memory_mb: float
    system_memory_percent: float
    gc_counts: dict[int, int]
    object_counts: dict[str, int]
    stack_trace: str | None = None
    operation: str | None = None
    custom_metrics: dict[str, Any] = None

    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class MemoryProfileResult:
    """Results from memory profiling operation."""

    operation_name: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    peak_memory_mb: float
    memory_delta_mb: float
    duration_seconds: float
    gc_collections: dict[int, int]
    potential_leaks: list[dict[str, Any]]
    recommendations: list[str]


class MemoryTracker:
    """Tracks memory usage and detects potential issues."""

    def __init__(self):
        self.snapshots: list[MemorySnapshot] = []
        self.tracked_objects: set[weakref.ref] = set()
        self.operation_stack: list[str] = []
        self.baseline_memory: float | None = None
        self.memory_threshold_mb: float = 1000.0  # Alert threshold
        self.leak_detection_enabled: bool = True

    def get_current_snapshot(self, operation: str | None = None) -> MemorySnapshot:
        """Get current memory state snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()

        # Get system memory info
        system_memory = psutil.virtual_memory()

        # Get garbage collector counts
        gc_counts = {i: gc.get_count()[i] for i in range(len(gc.get_count()))}

        # Count objects by type
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        # Get stack trace if requested
        stack_trace = None
        if operation:
            stack_trace = "".join(traceback.format_stack())

        return MemorySnapshot(
            timestamp=datetime.now(timezone.utc),
            process_memory_mb=memory_info.rss / 1024 / 1024,
            system_memory_mb=system_memory.total / 1024 / 1024,
            system_memory_percent=system_memory.percent,
            gc_counts=gc_counts,
            object_counts=object_counts,
            stack_trace=stack_trace,
            operation=operation,
        )

    def start_tracking(self, operation: str) -> None:
        """Start tracking memory for an operation."""
        self.operation_stack.append(operation)
        snapshot = self.get_current_snapshot(operation)
        self.snapshots.append(snapshot)

        if self.baseline_memory is None:
            self.baseline_memory = snapshot.process_memory_mb

    def stop_tracking(self, operation: str) -> MemoryProfileResult:
        """Stop tracking and return profiling results."""
        if not self.operation_stack or self.operation_stack[-1] != operation:
            logger.warning(
                f"Memory tracking mismatch: expected {operation}, got {self.operation_stack}"
            )

        if self.operation_stack:
            self.operation_stack.pop()

        end_snapshot = self.get_current_snapshot(operation)
        self.snapshots.append(end_snapshot)

        # Find corresponding start snapshot
        start_snapshot = None
        for snapshot in reversed(self.snapshots[:-1]):
            if snapshot.operation == operation:
                start_snapshot = snapshot
                break

        if not start_snapshot:
            logger.error(f"Could not find start snapshot for operation: {operation}")
            start_snapshot = self.snapshots[0] if self.snapshots else end_snapshot

        # Calculate metrics
        memory_delta = end_snapshot.process_memory_mb - start_snapshot.process_memory_mb
        duration = (end_snapshot.timestamp - start_snapshot.timestamp).total_seconds()

        # Find peak memory usage
        peak_memory = max(
            snapshot.process_memory_mb
            for snapshot in self.snapshots
            if start_snapshot.timestamp <= snapshot.timestamp <= end_snapshot.timestamp
        )

        # Calculate GC collections during operation
        gc_collections = {}
        for generation in start_snapshot.gc_counts:
            gc_collections[generation] = (
                end_snapshot.gc_counts[generation]
                - start_snapshot.gc_counts[generation]
            )

        # Detect potential leaks
        potential_leaks = self._detect_potential_leaks(start_snapshot, end_snapshot)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            start_snapshot, end_snapshot, memory_delta, gc_collections
        )

        return MemoryProfileResult(
            operation_name=operation,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_memory_mb=peak_memory,
            memory_delta_mb=memory_delta,
            duration_seconds=duration,
            gc_collections=gc_collections,
            potential_leaks=potential_leaks,
            recommendations=recommendations,
        )

    def _detect_potential_leaks(
        self, start_snapshot: MemorySnapshot, end_snapshot: MemorySnapshot
    ) -> list[dict[str, Any]]:
        """Detect potential memory leaks by comparing object counts."""
        leaks = []

        for obj_type, end_count in end_snapshot.object_counts.items():
            start_count = start_snapshot.object_counts.get(obj_type, 0)
            count_delta = end_count - start_count

            # Flag significant increases in object counts
            if count_delta > 100 and count_delta > start_count * 0.5:
                leaks.append(
                    {
                        "object_type": obj_type,
                        "count_increase": count_delta,
                        "start_count": start_count,
                        "end_count": end_count,
                        "growth_ratio": count_delta / max(start_count, 1),
                    }
                )

        return sorted(leaks, key=lambda x: x["count_increase"], reverse=True)[:10]

    def _generate_recommendations(
        self,
        start_snapshot: MemorySnapshot,
        end_snapshot: MemorySnapshot,
        memory_delta: float,
        gc_collections: dict[int, int],
    ) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        # High memory usage
        if memory_delta > 100:
            recommendations.append(
                f"High memory increase detected (+{memory_delta:.1f}MB). Consider optimizing data structures or implementing lazy loading."
            )

        # Excessive GC activity
        if any(count > 50 for count in gc_collections.values()):
            recommendations.append(
                "High garbage collection activity detected. Consider object pooling or reducing object allocations."
            )

        # Memory threshold exceeded
        if end_snapshot.process_memory_mb > self.memory_threshold_mb:
            recommendations.append(
                f"Memory usage ({end_snapshot.process_memory_mb:.1f}MB) exceeded threshold ({self.memory_threshold_mb}MB)."
            )

        # System memory pressure
        if end_snapshot.system_memory_percent > 85:
            recommendations.append(
                f"System memory usage high ({end_snapshot.system_memory_percent:.1f}%). Consider reducing memory footprint."
            )

        return recommendations

    def reset(self) -> None:
        """Reset tracking state."""
        self.snapshots.clear()
        self.tracked_objects.clear()
        self.operation_stack.clear()
        self.baseline_memory = None

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory statistics."""
        current = self.get_current_snapshot()
        return {
            "current_memory_mb": current.process_memory_mb,
            "baseline_memory_mb": self.baseline_memory or current.process_memory_mb,
            "memory_delta_mb": current.process_memory_mb - (self.baseline_memory or 0),
            "system_memory_percent": current.system_memory_percent,
            "total_snapshots": len(self.snapshots),
            "active_operations": len(self.operation_stack),
        }


# Global memory tracker instance
_memory_tracker = MemoryTracker()


def memory_profile(operation_name: str | None = None, log_results: bool = True):
    """Decorator to profile memory usage of a function.

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        log_results: Whether to log profiling results
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            _memory_tracker.start_tracking(op_name)
            try:
                return func(*args, **kwargs)
            finally:
                profile_result = _memory_tracker.stop_tracking(op_name)
                if log_results:
                    _log_profile_result(profile_result)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            _memory_tracker.start_tracking(op_name)
            try:
                return await func(*args, **kwargs)
            finally:
                profile_result = _memory_tracker.stop_tracking(op_name)
                if log_results:
                    _log_profile_result(profile_result)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def memory_context(operation_name: str, log_results: bool = True):
    """Context manager for memory profiling."""
    _memory_tracker.start_tracking(operation_name)
    try:
        yield _memory_tracker
    finally:
        profile_result = _memory_tracker.stop_tracking(operation_name)
        if log_results:
            _log_profile_result(profile_result)


@asynccontextmanager
async def async_memory_context(operation_name: str, log_results: bool = True):
    """Async context manager for memory profiling."""
    _memory_tracker.start_tracking(operation_name)
    try:
        yield _memory_tracker
    finally:
        profile_result = _memory_tracker.stop_tracking(operation_name)
        if log_results:
            _log_profile_result(profile_result)


def _log_profile_result(result: MemoryProfileResult) -> None:
    """Log memory profiling results."""
    logger.info(
        f"Memory Profile - {result.operation_name}: "
        f"Δ{result.memory_delta_mb:+.1f}MB "
        f"(peak: {result.peak_memory_mb:.1f}MB, "
        f"duration: {result.duration_seconds:.2f}s)"
    )

    if result.potential_leaks:
        logger.warning(
            f"Potential leaks detected: {len(result.potential_leaks)} object types"
        )

    for recommendation in result.recommendations:
        logger.info(f"Recommendation: {recommendation}")


class MemoryOptimizer:
    """Memory optimization utilities."""

    @staticmethod
    def optimize_numpy_arrays(*arrays: np.ndarray) -> None:
        """Optimize numpy arrays for memory efficiency."""
        for array in arrays:
            if array.dtype == np.float64:
                # Consider using float32 if precision allows
                logger.debug(
                    f"Array using float64, consider float32 for {array.nbytes} bytes"
                )

    @staticmethod
    def cleanup_large_objects(size_threshold_mb: float = 10.0) -> int:
        """Clean up large objects from memory."""
        cleaned = 0
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj) / 1024 / 1024
                if size > size_threshold_mb:
                    # Check if object is a large data structure that can be cleared
                    if isinstance(obj, list | dict | set) and hasattr(obj, "clear"):
                        obj.clear()
                        cleaned += 1
            except (TypeError, AttributeError):
                continue

        gc.collect()
        return cleaned

    @staticmethod
    def force_garbage_collection() -> dict[int, int]:
        """Force garbage collection and return collection counts."""
        collected = {}
        for generation in range(gc.get_count().__len__()):
            collected[generation] = gc.collect(generation)
        return collected


class StreamingJSONLReader:
    """Memory-efficient JSONL reader with streaming support."""

    def __init__(self, file_path: Path, chunk_size: int = 8192):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self._buffer = ""

    def __iter__(self):
        """Iterate over JSONL records without loading entire file."""
        if not self.file_path.exists():
            return

        with open(self.file_path, encoding="utf-8") as f:
            self._buffer = ""

            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                self._buffer += chunk

                # Process complete lines
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()

                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON line: {line[:100]}...")
                            continue

            # Process remaining buffer
            if self._buffer.strip():
                try:
                    yield json.loads(self._buffer.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in buffer: {self._buffer[:100]}...")

    async def aiter(self):
        """Async iterator over JSONL records."""
        import aiofiles

        if not self.file_path.exists():
            return

        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            self._buffer = ""

            while True:
                chunk = await f.read(self.chunk_size)
                if not chunk:
                    break

                self._buffer += chunk

                # Process complete lines
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()

                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON line: {line[:100]}...")
                            continue

                # Yield control to event loop
                await asyncio.sleep(0)

            # Process remaining buffer
            if self._buffer.strip():
                try:
                    yield json.loads(self._buffer.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in buffer: {self._buffer[:100]}...")


class LazyModelLoader:
    """Lazy loading wrapper for large ML models."""

    def __init__(self, loader_func: Callable, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._model = None
        self._loaded = False

    def __getattr__(self, name):
        if not self._loaded:
            self._load_model()
        return getattr(self._model, name)

    def _load_model(self):
        """Load the model on first access."""
        if not self._loaded:
            logger.info("Lazy loading model...")
            self._model = self.loader_func(*self.args, **self.kwargs)
            self._loaded = True

    def unload(self):
        """Unload the model to free memory."""
        if self._loaded:
            logger.info("Unloading model...")
            self._model = None
            self._loaded = False
            gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded


def save_memory_report(file_path: Path, tracker: MemoryTracker | None = None) -> None:
    """Save detailed memory report to file."""
    tracker = tracker or _memory_tracker

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory_stats": tracker.get_memory_stats(),
        "snapshots": [],
    }

    for snapshot in tracker.snapshots:
        report["snapshots"].append(
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "process_memory_mb": snapshot.process_memory_mb,
                "system_memory_percent": snapshot.system_memory_percent,
                "operation": snapshot.operation,
                "gc_counts": snapshot.gc_counts,
                "top_objects": dict(
                    sorted(
                        snapshot.object_counts.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                ),
            }
        )

    with open(file_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Memory report saved to {file_path}")


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    return _memory_tracker


# Memory limit utilities
class MemoryLimitExceeded(Exception):
    """Raised when memory usage exceeds configured limits."""

    pass


def set_memory_limit(limit_mb: float) -> None:
    """Set global memory usage limit."""
    _memory_tracker.memory_threshold_mb = limit_mb


def check_memory_limit() -> None:
    """Check if current memory usage exceeds limit."""
    current = _memory_tracker.get_current_snapshot()
    if current.process_memory_mb > _memory_tracker.memory_threshold_mb:
        raise MemoryLimitExceeded(
            f"Memory usage ({current.process_memory_mb:.1f}MB) "
            f"exceeds limit ({_memory_tracker.memory_threshold_mb}MB)"
        )


class MemoryMonitor:
    """Continuous memory monitoring with alerts and automatic cleanup."""

    def __init__(
        self,
        alert_threshold_mb: float = 1000.0,
        critical_threshold_mb: float = 2000.0,
        cleanup_threshold_mb: float = 1500.0,
        monitoring_interval: float = 30.0,
    ):
        self.alert_threshold_mb = alert_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.monitoring_interval = monitoring_interval
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable[[MemorySnapshot], None]] = []

    def add_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add callback function to be called on memory events."""
        self._callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="MemoryMonitor"
        )
        self._monitor_thread.start()
        logger.info(
            f"Memory monitoring started (interval: {self.monitoring_interval}s)"
        )

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                snapshot = _memory_tracker.get_current_snapshot("monitor")
                self._process_snapshot(snapshot)

                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.exception(f"Memory callback error: {e}")

            except Exception as e:
                logger.exception(f"Memory monitoring error: {e}")

    def _process_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Process memory snapshot and trigger actions."""
        memory_mb = snapshot.process_memory_mb

        if memory_mb > self.critical_threshold_mb:
            logger.critical(
                f"CRITICAL: Memory usage {memory_mb:.1f}MB exceeds critical threshold {self.critical_threshold_mb}MB"
            )
            self._trigger_emergency_cleanup()

        elif memory_mb > self.cleanup_threshold_mb:
            logger.warning(
                f"Memory usage {memory_mb:.1f}MB exceeds cleanup threshold {self.cleanup_threshold_mb}MB"
            )
            self._trigger_cleanup()

        elif memory_mb > self.alert_threshold_mb:
            logger.warning(
                f"Memory usage {memory_mb:.1f}MB exceeds alert threshold {self.alert_threshold_mb}MB"
            )

    def _trigger_cleanup(self) -> None:
        """Trigger normal memory cleanup."""
        logger.info("Triggering memory cleanup...")
        try:
            cleaned = MemoryOptimizer.cleanup_large_objects()
            collected = MemoryOptimizer.force_garbage_collection()
            logger.info(
                f"Cleanup complete: {cleaned} objects cleaned, {sum(collected.values())} objects collected"
            )
        except Exception as e:
            logger.exception(f"Memory cleanup failed: {e}")

    def _trigger_emergency_cleanup(self) -> None:
        """Trigger aggressive memory cleanup."""
        logger.critical("Triggering emergency memory cleanup...")
        try:
            # More aggressive cleanup
            cleaned = MemoryOptimizer.cleanup_large_objects(size_threshold_mb=1.0)
            collected = MemoryOptimizer.force_garbage_collection()

            # Clear various caches
            if hasattr(functools, "_CacheInfo"):
                # Clear functools.lru_cache caches
                for obj in gc.get_objects():
                    if hasattr(obj, "cache_clear"):
                        with suppress(Exception):
                            obj.cache_clear()

            logger.critical(
                f"Emergency cleanup complete: {cleaned} objects cleaned, {sum(collected.values())} objects collected"
            )
        except Exception as e:
            logger.exception(f"Emergency cleanup failed: {e}")


class MemoryProfiler:
    """Advanced memory profiling with model-specific optimizations."""

    def __init__(self):
        self.model_memory_usage: dict[str, float] = {}
        self.operation_profiles: dict[str, list[MemoryProfileResult]] = {}
        self.optimization_history: list[dict[str, Any]] = []

    @memory_profile(operation_name="model_loading")
    def profile_model_loading(self, model_name: str, loader_func: Callable) -> Any:
        """Profile memory usage during model loading."""
        logger.info(f"Profiling model loading: {model_name}")

        # Record pre-loading memory
        pre_memory = _memory_tracker.get_current_snapshot(f"pre_load_{model_name}")

        try:
            # Load model
            model = loader_func()

            # Record post-loading memory
            post_memory = _memory_tracker.get_current_snapshot(
                f"post_load_{model_name}"
            )

            # Calculate model memory usage
            model_memory = post_memory.process_memory_mb - pre_memory.process_memory_mb
            self.model_memory_usage[model_name] = model_memory

            logger.info(
                f"Model {model_name} loaded, memory usage: {model_memory:.1f}MB"
            )
            return model

        except Exception as e:
            logger.exception(f"Model loading failed for {model_name}: {e}")
            raise

    def get_model_memory_recommendations(self) -> list[str]:
        """Get model-specific memory optimization recommendations."""
        recommendations = []

        total_model_memory = sum(self.model_memory_usage.values())
        if total_model_memory > 500:  # 500MB threshold
            recommendations.append(
                f"High model memory usage ({total_model_memory:.1f}MB). "
                "Consider model quantization or using smaller variants."
            )

        # Find memory-heavy models
        heavy_models = [
            (name, memory)
            for name, memory in self.model_memory_usage.items()
            if memory > 200  # 200MB per model
        ]

        if heavy_models:
            heavy_models.sort(key=lambda x: x[1], reverse=True)
            for name, memory in heavy_models[:3]:  # Top 3 heavy models
                recommendations.append(
                    f"Model '{name}' uses {memory:.1f}MB. "
                    f"Consider lazy loading or model unloading when not in use."
                )

        return recommendations

    def analyze_operation_patterns(self, operation_name: str) -> dict[str, Any]:
        """Analyze memory patterns for a specific operation."""
        if operation_name not in self.operation_profiles:
            return {"error": f"No profiles found for operation: {operation_name}"}

        profiles = self.operation_profiles[operation_name]
        if not profiles:
            return {"error": f"No profile data for operation: {operation_name}"}

        # Calculate statistics
        memory_deltas = [p.memory_delta_mb for p in profiles]
        peak_memories = [p.peak_memory_mb for p in profiles]
        durations = [p.duration_seconds for p in profiles]

        return {
            "operation": operation_name,
            "profile_count": len(profiles),
            "memory_delta": {
                "min": min(memory_deltas),
                "max": max(memory_deltas),
                "avg": sum(memory_deltas) / len(memory_deltas),
            },
            "peak_memory": {
                "min": min(peak_memories),
                "max": max(peak_memories),
                "avg": sum(peak_memories) / len(peak_memories),
            },
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
            },
            "leak_patterns": self._analyze_leak_patterns(profiles),
            "recommendations": self._generate_operation_recommendations(profiles),
        }

    def _analyze_leak_patterns(
        self, profiles: list[MemoryProfileResult]
    ) -> dict[str, Any]:
        """Analyze memory leak patterns across profiles."""
        leak_counts = {}
        persistent_leaks = []

        for profile in profiles:
            for leak in profile.potential_leaks:
                obj_type = leak["object_type"]
                leak_counts[obj_type] = leak_counts.get(obj_type, 0) + 1

        # Find objects that consistently leak
        profile_count = len(profiles)
        for obj_type, count in leak_counts.items():
            if count > profile_count * 0.7:  # Present in >70% of profiles
                persistent_leaks.append(
                    {
                        "object_type": obj_type,
                        "occurrence_rate": count / profile_count,
                        "total_occurrences": count,
                    }
                )

        return {
            "total_leak_types": len(leak_counts),
            "persistent_leaks": persistent_leaks,
            "most_common_leaks": sorted(
                leak_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def _generate_operation_recommendations(
        self, profiles: list[MemoryProfileResult]
    ) -> list[str]:
        """Generate recommendations based on operation analysis."""
        recommendations = []

        avg_delta = sum(p.memory_delta_mb for p in profiles) / len(profiles)
        max_delta = max(p.memory_delta_mb for p in profiles)

        if avg_delta > 50:
            recommendations.append(
                f"Operation shows consistent memory growth (avg: {avg_delta:.1f}MB). "
                "Consider implementing memory cleanup or optimizing data structures."
            )

        if max_delta > 200:
            recommendations.append(
                f"Operation can consume large amounts of memory (max: {max_delta:.1f}MB). "
                "Consider implementing streaming or chunking for large datasets."
            )

        # Check for GC pressure
        avg_gc = sum(sum(p.gc_collections.values()) for p in profiles) / len(profiles)
        if avg_gc > 20:
            recommendations.append(
                f"High garbage collection activity detected (avg: {avg_gc:.1f} collections). "
                "Consider object pooling or reducing temporary object creation."
            )

        return recommendations


class MemoryResourceManager:
    """Resource manager for automatic memory cleanup and optimization."""

    def __init__(self):
        self.managed_resources: dict[str, Any] = {}
        self.cleanup_callbacks: dict[str, Callable] = {}
        self.resource_limits: dict[str, float] = {}

    def register_resource(
        self,
        resource_id: str,
        resource: Any,
        cleanup_callback: Callable | None = None,
        memory_limit_mb: float | None = None,
    ) -> None:
        """Register a resource for managed cleanup."""
        self.managed_resources[resource_id] = resource

        if cleanup_callback:
            self.cleanup_callbacks[resource_id] = cleanup_callback

        if memory_limit_mb:
            self.resource_limits[resource_id] = memory_limit_mb

    def cleanup_resource(self, resource_id: str) -> bool:
        """Clean up a specific resource."""
        if resource_id not in self.managed_resources:
            return False

        try:
            # Call custom cleanup if available
            if resource_id in self.cleanup_callbacks:
                self.cleanup_callbacks[resource_id]()

            # Remove from managed resources
            resource = self.managed_resources.pop(resource_id)

            # Generic cleanup for common types
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "clear"):
                resource.clear()

            # Clean up callbacks and limits
            self.cleanup_callbacks.pop(resource_id, None)
            self.resource_limits.pop(resource_id, None)

            logger.info(f"Resource {resource_id} cleaned up successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to cleanup resource {resource_id}: {e}")
            return False

    def cleanup_all_resources(self) -> int:
        """Clean up all managed resources."""
        cleaned = 0
        for resource_id in list(self.managed_resources.keys()):
            if self.cleanup_resource(resource_id):
                cleaned += 1
        return cleaned

    def check_resource_limits(self) -> dict[str, bool]:
        """Check if any resources exceed their memory limits."""
        violations = {}
        current_snapshot = _memory_tracker.get_current_snapshot()

        for resource_id, limit_mb in self.resource_limits.items():
            # This is a simplified check - in practice, you'd want more
            # sophisticated resource-specific memory tracking
            if current_snapshot.process_memory_mb > limit_mb:
                violations[resource_id] = True
                logger.warning(
                    f"Resource {resource_id} may exceed memory limit ({limit_mb}MB)"
                )

        return violations


# Global instances
_memory_monitor = MemoryMonitor()
_memory_profiler = MemoryProfiler()
_resource_manager = MemoryResourceManager()


def start_memory_monitoring(**kwargs) -> None:
    """Start global memory monitoring."""
    _memory_monitor.start_monitoring()


def stop_memory_monitoring() -> None:
    """Stop global memory monitoring."""
    _memory_monitor.stop_monitoring()


def get_memory_profiler() -> MemoryProfiler:
    """Get the global memory profiler instance."""
    return _memory_profiler


def get_resource_manager() -> MemoryResourceManager:
    """Get the global resource manager instance."""
    return _resource_manager


def setup_memory_limits(
    alert_mb: float = 1000.0,
    critical_mb: float = 2000.0,
    cleanup_mb: float = 1500.0,
) -> None:
    """Setup memory monitoring limits."""
    _memory_monitor.alert_threshold_mb = alert_mb
    _memory_monitor.critical_threshold_mb = critical_mb
    _memory_monitor.cleanup_threshold_mb = cleanup_mb
    set_memory_limit(critical_mb)


# Signal handler for memory emergencies
def _emergency_memory_handler(signum, frame):
    """Emergency signal handler for memory issues."""
    logger.critical("Emergency memory cleanup triggered by signal")
    _memory_monitor._trigger_emergency_cleanup()


def setup_emergency_memory_handler() -> None:
    """Setup signal handler for memory emergencies."""
    try:
        signal.signal(signal.SIGUSR1, _emergency_memory_handler)
        logger.info("Emergency memory handler setup complete (send SIGUSR1 to trigger)")
    except Exception as e:
        logger.warning(f"Could not setup emergency memory handler: {e}")


# Export key components
__all__ = [
    "MemorySnapshot",
    "MemoryProfileResult",
    "MemoryTracker",
    "MemoryOptimizer",
    "MemoryMonitor",
    "MemoryProfiler",
    "MemoryResourceManager",
    "StreamingJSONLReader",
    "LazyModelLoader",
    "memory_profile",
    "memory_context",
    "async_memory_context",
    "get_memory_tracker",
    "get_memory_profiler",
    "get_resource_manager",
    "save_memory_report",
    "set_memory_limit",
    "check_memory_limit",
    "start_memory_monitoring",
    "stop_memory_monitoring",
    "setup_memory_limits",
    "setup_emergency_memory_handler",
    "MemoryLimitExceeded",
]
