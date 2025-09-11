"""
Unit tests for memory profiling tools.
"""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from tools.memory_profiler import (
    LazyModelLoader,
    MemoryOptimizer,
    MemoryProfiler,
    MemorySnapshot,
    MemoryTracker,
    StreamingJSONLReader,
    get_memory_tracker,
    memory_context,
    memory_profile,
)


class TestMemoryTracker(unittest.TestCase):
    """Test memory tracking functionality."""

    def setUp(self):
        self.tracker = MemoryTracker()

    def test_get_current_snapshot(self):
        """Test memory snapshot creation."""
        snapshot = self.tracker.get_current_snapshot("test_op")

        assert isinstance(snapshot, MemorySnapshot)
        assert isinstance(snapshot.process_memory_mb, float)
        assert isinstance(snapshot.system_memory_mb, float)
        assert isinstance(snapshot.system_memory_percent, float)
        assert isinstance(snapshot.gc_counts, dict)
        assert isinstance(snapshot.object_counts, dict)
        assert snapshot.operation == "test_op"

    def test_memory_tracking(self):
        """Test start/stop tracking."""
        operation = "test_operation"

        self.tracker.start_tracking(operation)
        assert len(self.tracker.operation_stack) == 1
        assert self.tracker.operation_stack[0] == operation

        result = self.tracker.stop_tracking(operation)
        assert len(self.tracker.operation_stack) == 0
        assert result.operation_name == operation
        assert isinstance(result.memory_delta_mb, float)

    def test_get_memory_stats(self):
        """Test memory statistics."""
        stats = self.tracker.get_memory_stats()

        required_keys = [
            "current_memory_mb",
            "baseline_memory_mb",
            "memory_delta_mb",
            "system_memory_percent",
            "total_snapshots",
            "active_operations",
        ]

        for key in required_keys:
            assert key in stats


class TestMemoryProfileDecorator(unittest.TestCase):
    """Test memory profiling decorator."""

    def test_sync_function_profiling(self):
        """Test profiling synchronous functions."""

        @memory_profile(operation_name="test_sync", log_results=False)
        def test_function(size=1000):
            return list(range(size))

        result = test_function(size=5000)
        assert len(result) == 5000

    def test_async_function_profiling(self):
        """Test profiling asynchronous functions."""

        @memory_profile(operation_name="test_async", log_results=False)
        async def test_async_function(size=1000):
            await asyncio.sleep(0.001)
            return list(range(size))

        async def run_test():
            result = await test_async_function(size=3000)
            assert len(result) == 3000

        asyncio.run(run_test())

    def test_memory_context_manager(self):
        """Test memory context manager."""
        with memory_context("test_context", log_results=False):
            data = [i**2 for i in range(10000)]
            assert len(data) == 10000


class TestStreamingJSONLReader(unittest.TestCase):
    """Test streaming JSONL reader."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )

        # Write test data
        test_data = [
            {"id": i, "value": f"test_{i}", "number": i * 2} for i in range(100)
        ]

        for item in test_data:
            self.temp_file.write(json.dumps(item) + "\n")

        self.temp_file.close()
        self.test_data = test_data

    def tearDown(self):
        Path(self.temp_file.name).unlink(missing_ok=True)

    def test_streaming_read(self):
        """Test streaming JSONL reading."""
        reader = StreamingJSONLReader(Path(self.temp_file.name), chunk_size=512)

        read_data = list(reader)

        assert len(read_data) == len(self.test_data)

        for original, read in zip(self.test_data, read_data, strict=False):
            assert original == read

    def test_async_streaming_read(self):
        """Test async streaming JSONL reading."""

        async def run_test():
            reader = StreamingJSONLReader(Path(self.temp_file.name), chunk_size=512)

            read_data = []
            async for item in reader.aiter():
                read_data.append(item)

            assert len(read_data) == len(self.test_data)

            for original, read in zip(self.test_data, read_data, strict=False):
                assert original == read

        asyncio.run(run_test())


class TestLazyModelLoader(unittest.TestCase):
    """Test lazy model loading."""

    def setUp(self):
        self.load_called = False

        def mock_loader():
            self.load_called = True
            return Mock(predict=Mock(return_value=np.array([1, 2, 3])))

        self.mock_loader = mock_loader

    def test_lazy_loading(self):
        """Test that model is loaded lazily."""
        lazy_model = LazyModelLoader(self.mock_loader)

        # Model should not be loaded yet
        assert not lazy_model.is_loaded
        assert not self.load_called

        # Access model attribute - should trigger loading
        result = lazy_model.predict(np.array([1, 2, 3]))

        # Model should now be loaded
        assert lazy_model.is_loaded
        assert self.load_called
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_model_unloading(self):
        """Test model unloading."""
        lazy_model = LazyModelLoader(self.mock_loader)

        # Load model
        lazy_model.predict(np.array([1, 2, 3]))
        assert lazy_model.is_loaded

        # Unload model
        lazy_model.unload()
        assert not lazy_model.is_loaded


class TestMemoryOptimizer(unittest.TestCase):
    """Test memory optimization utilities."""

    def test_force_garbage_collection(self):
        """Test garbage collection."""
        result = MemoryOptimizer.force_garbage_collection()

        assert isinstance(result, dict)
        assert len(result) > 0

        for generation, count in result.items():
            assert isinstance(generation, int)
            assert isinstance(count, int)

    def test_cleanup_large_objects(self):
        """Test large object cleanup."""
        # Create some large objects
        list(range(100000))
        {f"key_{i}": f"value_{i}" for i in range(50000)}

        # Run cleanup
        cleaned = MemoryOptimizer.cleanup_large_objects(size_threshold_mb=0.001)

        assert isinstance(cleaned, int)
        # Should be >= 0 (some objects might be cleaned up)
        assert cleaned >= 0

    def test_numpy_array_optimization(self):
        """Test numpy array optimization."""
        # Create float64 array
        test_array = np.random.random((100, 100)).astype(np.float64)

        # This should not raise an error
        MemoryOptimizer.optimize_numpy_arrays(test_array)


class TestMemoryProfiler(unittest.TestCase):
    """Test advanced memory profiler."""

    def setUp(self):
        self.profiler = MemoryProfiler()

    def test_profile_model_loading(self):
        """Test model loading profiling."""

        def mock_model_loader():
            return Mock(predict=Mock(return_value="predictions"))

        model = self.profiler.profile_model_loading("test_model", mock_model_loader)

        assert model is not None
        assert "test_model" in self.profiler.model_memory_usage
        assert isinstance(self.profiler.model_memory_usage["test_model"], float)

    def test_model_memory_recommendations(self):
        """Test model memory recommendations."""
        # Add some mock model memory usage
        self.profiler.model_memory_usage = {
            "large_model": 600.0,  # Should trigger recommendation
            "small_model": 50.0,  # Should not trigger recommendation
        }

        recommendations = self.profiler.get_model_memory_recommendations()

        assert isinstance(recommendations, list)
        # Should have at least one recommendation for the large model
        assert len(recommendations) > 0
        assert any("high model memory usage" in rec.lower() for rec in recommendations)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_memory_tracking_integration(self):
        """Test complete memory tracking workflow."""
        tracker = get_memory_tracker()
        initial_stats = tracker.get_memory_stats()

        @memory_profile(operation_name="integration_test", log_results=False)
        def memory_intensive_function():
            # Simulate memory usage
            data = [i**2 for i in range(100000)]
            return len(data)

        result = memory_intensive_function()
        assert result == 100000

        final_stats = tracker.get_memory_stats()

        # Should have at least one more snapshot
        assert final_stats["total_snapshots"] > initial_stats["total_snapshots"]


if __name__ == "__main__":
    unittest.main()
