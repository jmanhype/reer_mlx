#!/usr/bin/env python3
"""
Example usage of REER MLX memory profiling tools.

This example demonstrates:
1. Basic memory profiling with decorators
2. Advanced memory monitoring
3. Model-specific profiling
4. Memory optimization techniques
5. Resource management
6. Memory leak detection
"""

import asyncio
import logging
from pathlib import Path
import random

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import memory profiling tools
from ..core.trace_store import REERTraceStore
from tools.memory_profiler import (
    LazyModelLoader,
    MemoryOptimizer,
    StreamingJSONLReader,
    async_memory_context,
    get_memory_profiler,
    get_memory_tracker,
    get_resource_manager,
    memory_context,
    memory_profile,
    save_memory_report,
    setup_emergency_memory_handler,
    setup_memory_limits,
    start_memory_monitoring,
    stop_memory_monitoring,
)


# Example 1: Basic memory profiling with decorators
@memory_profile(operation_name="data_processing")
def process_large_dataset(size: int = 10000) -> list[float]:
    """Example function that processes a large dataset."""
    logger.info(f"Processing dataset of size {size}")

    # Simulate memory-intensive operations
    data = [random.random() for _ in range(size)]

    # Some processing
    result = [x * 2 + 1 for x in data if x > 0.5]

    # More processing with numpy
    numpy_data = np.array(result)
    processed = np.sqrt(numpy_data) * np.log(numpy_data + 1)

    return processed.tolist()


@memory_profile(operation_name="async_data_loading")
async def load_data_async(num_batches: int = 5, batch_size: int = 1000) -> list[dict]:
    """Example async function for data loading."""
    logger.info(f"Loading {num_batches} batches of {batch_size} items each")

    all_data = []
    for i in range(num_batches):
        # Simulate async data loading
        await asyncio.sleep(0.1)

        batch = [
            {
                "id": j,
                "value": random.random(),
                "category": f"category_{j % 10}",
                "metadata": {"batch": i, "index": j},
            }
            for j in range(batch_size)
        ]
        all_data.extend(batch)

    return all_data


# Example 2: Memory monitoring with context managers
def demonstrate_memory_contexts():
    """Demonstrate memory profiling context managers."""
    logger.info("=== Memory Context Demonstration ===")

    # Synchronous context
    with memory_context("sync_operation"):
        data = [i**2 for i in range(100000)]
        processed = sum(x for x in data if x % 2 == 0)
        logger.info(f"Sync operation result: {processed}")


async def demonstrate_async_memory_contexts():
    """Demonstrate async memory profiling context managers."""
    logger.info("=== Async Memory Context Demonstration ===")

    async with async_memory_context("async_operation"):
        data = await load_data_async(num_batches=3, batch_size=500)
        processed = len([item for item in data if item["value"] > 0.5])
        logger.info(f"Async operation result: {processed}")


# Example 3: Model-specific profiling
class DummyModel:
    """Dummy ML model for demonstration."""

    def __init__(self, size: int = 1000):
        self.weights = np.random.random((size, size))
        self.bias = np.random.random(size)
        logger.info(f"Model initialized with {size}x{size} weights")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return np.dot(input_data, self.weights) + self.bias


def create_dummy_model(size: int = 1000) -> DummyModel:
    """Factory function for model creation."""
    return DummyModel(size)


def demonstrate_model_profiling():
    """Demonstrate model-specific memory profiling."""
    logger.info("=== Model Profiling Demonstration ===")

    profiler = get_memory_profiler()

    # Profile model loading
    model = profiler.profile_model_loading(
        "dummy_model_large", lambda: create_dummy_model(2000)
    )

    # Use the model
    input_data = np.random.random((10, 2000))
    predictions = model.predict(input_data)
    logger.info(f"Model predictions shape: {predictions.shape}")

    # Get model memory recommendations
    recommendations = profiler.get_model_memory_recommendations()
    logger.info("Model memory recommendations:")
    for rec in recommendations:
        logger.info(f"  - {rec}")


# Example 4: Lazy model loading
def demonstrate_lazy_loading():
    """Demonstrate lazy model loading."""
    logger.info("=== Lazy Loading Demonstration ===")

    # Create lazy model loader
    lazy_model = LazyModelLoader(create_dummy_model, size=1500)

    logger.info(f"Model loaded: {lazy_model.is_loaded}")

    # Model will be loaded on first access
    input_data = np.random.random((5, 1500))
    predictions = lazy_model.predict(input_data)

    logger.info(f"Model loaded: {lazy_model.is_loaded}")
    logger.info(f"Predictions shape: {predictions.shape}")

    # Unload model to free memory
    lazy_model.unload()
    logger.info(f"Model loaded after unload: {lazy_model.is_loaded}")


# Example 5: Memory optimization
def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques."""
    logger.info("=== Memory Optimization Demonstration ===")

    # Create some large data structures
    list(range(100000))
    {f"key_{i}": f"value_{i}" for i in range(50000)}
    set(range(75000))

    # Get initial memory stats
    initial_stats = get_memory_tracker().get_memory_stats()
    logger.info(f"Initial memory: {initial_stats['current_memory_mb']:.1f}MB")

    # Optimize numpy arrays
    large_array = np.random.random((1000, 1000)).astype(np.float64)
    MemoryOptimizer.optimize_numpy_arrays(large_array)

    # Clean up large objects
    cleaned = MemoryOptimizer.cleanup_large_objects(size_threshold_mb=1.0)
    logger.info(f"Cleaned up {cleaned} large objects")

    # Force garbage collection
    collected = MemoryOptimizer.force_garbage_collection()
    logger.info(f"Garbage collection results: {collected}")

    # Get final memory stats
    final_stats = get_memory_tracker().get_memory_stats()
    logger.info(f"Final memory: {final_stats['current_memory_mb']:.1f}MB")
    logger.info(f"Memory delta: {final_stats['memory_delta_mb']:+.1f}MB")


# Example 6: Resource management
def demonstrate_resource_management():
    """Demonstrate automatic resource management."""
    logger.info("=== Resource Management Demonstration ===")

    resource_manager = get_resource_manager()

    # Create some resources
    large_data = list(range(100000))
    model = DummyModel(size=500)
    file_handle = open(__file__)

    # Register resources
    resource_manager.register_resource(
        "large_data",
        large_data,
        cleanup_callback=lambda: large_data.clear(),
        memory_limit_mb=50.0,
    )

    resource_manager.register_resource("ml_model", model, memory_limit_mb=100.0)

    resource_manager.register_resource(
        "file_handle",
        file_handle,
        cleanup_callback=lambda: file_handle.close(),
        memory_limit_mb=1.0,
    )

    logger.info(f"Registered {len(resource_manager.managed_resources)} resources")

    # Check resource limits
    violations = resource_manager.check_resource_limits()
    if violations:
        logger.warning(f"Resource limit violations: {violations}")

    # Clean up all resources
    cleaned = resource_manager.cleanup_all_resources()
    logger.info(f"Cleaned up {cleaned} resources")


# Example 7: Trace store memory optimization
async def demonstrate_trace_store_optimization():
    """Demonstrate memory-optimized trace store operations."""
    logger.info("=== Trace Store Memory Optimization Demonstration ===")

    # Create temporary trace store
    temp_path = Path("/tmp/test_traces.jsonl")
    store = REERTraceStore(temp_path)

    # Generate sample traces
    sample_traces = []
    for i in range(1000):
        trace = {
            "source_post_id": f"post_{i}",
            "seed_params": {
                "topic": f"topic_{i % 10}",
                "style": "casual",
                "length": random.randint(100, 500),
                "thread_size": random.randint(1, 10),
            },
            "score": random.random(),
            "metrics": {
                "impressions": random.randint(100, 10000),
                "engagement_rate": random.uniform(0, 100),
                "retweets": random.randint(0, 500),
                "likes": random.randint(0, 1000),
            },
            "strategy_features": [f"feature_{j}" for j in random.sample(range(20), 3)],
            "provider": f"mlx::model_{i % 5}",
            "metadata": {
                "extraction_method": "automated",
                "confidence": random.random(),
            },
        }
        sample_traces.append(trace)

    # Append traces with memory profiling
    logger.info("Appending traces with memory optimization...")
    trace_ids = await store.append_traces(sample_traces)
    logger.info(f"Added {len(trace_ids)} traces")

    # Get memory-efficient stats
    stats = await store.get_memory_efficient_stats()
    logger.info(f"Store stats: {stats}")

    # Use lazy querying
    logger.info("Performing lazy query...")
    count = 0
    async for trace in store.lazy_query_traces(min_score=0.5, limit=100):
        count += 1
    logger.info(f"Lazy query returned {count} traces")

    # Optimize memory usage
    optimization_result = store.optimize_memory_usage()
    logger.info(f"Memory optimization: {optimization_result}")

    # Clean up
    temp_path.unlink(missing_ok=True)


# Example 8: Streaming JSONL processing
def demonstrate_streaming_jsonl():
    """Demonstrate memory-efficient JSONL processing."""
    logger.info("=== Streaming JSONL Demonstration ===")

    # Create temporary JSONL file
    temp_file = Path("/tmp/test_streaming.jsonl")

    # Write test data
    with open(temp_file, "w") as f:
        for i in range(10000):
            data = {
                "id": i,
                "value": random.random(),
                "category": f"cat_{i % 20}",
                "large_field": "x" * (random.randint(100, 1000)),
            }
            f.write(f"{json.dumps(data)}\n")

    logger.info(f"Created test file with {temp_file.stat().st_size} bytes")

    # Process with streaming reader
    reader = StreamingJSONLReader(temp_file, chunk_size=4096)

    total_processed = 0
    category_counts = {}

    for record in reader:
        total_processed += 1
        category = record.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

        # Process in batches to show memory efficiency
        if total_processed % 1000 == 0:
            current_memory = get_memory_tracker().get_current_snapshot()
            logger.info(
                f"Processed {total_processed} records, memory: {current_memory.process_memory_mb:.1f}MB"
            )

    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Categories found: {len(category_counts)}")

    # Clean up
    temp_file.unlink(missing_ok=True)


# Main demonstration function
async def main():
    """Main demonstration function."""
    logger.info("Starting REER MLX Memory Profiling Demonstration")

    # Setup memory monitoring
    setup_memory_limits(alert_mb=500.0, critical_mb=1000.0, cleanup_mb=750.0)
    setup_emergency_memory_handler()

    try:
        # Start memory monitoring in background
        start_memory_monitoring()

        # Run demonstrations
        logger.info("\n" + "=" * 60)
        process_large_dataset(size=50000)

        logger.info("\n" + "=" * 60)
        await load_data_async(num_batches=10, batch_size=2000)

        logger.info("\n" + "=" * 60)
        demonstrate_memory_contexts()

        logger.info("\n" + "=" * 60)
        await demonstrate_async_memory_contexts()

        logger.info("\n" + "=" * 60)
        demonstrate_model_profiling()

        logger.info("\n" + "=" * 60)
        demonstrate_lazy_loading()

        logger.info("\n" + "=" * 60)
        demonstrate_memory_optimization()

        logger.info("\n" + "=" * 60)
        demonstrate_resource_management()

        logger.info("\n" + "=" * 60)
        await demonstrate_trace_store_optimization()

        logger.info("\n" + "=" * 60)
        demonstrate_streaming_jsonl()

        # Save memory report
        report_path = Path("/tmp/memory_report.json")
        save_memory_report(report_path)
        logger.info(f"Memory report saved to {report_path}")

        # Get final memory statistics
        final_stats = get_memory_tracker().get_memory_stats()
        logger.info("\nFinal Memory Statistics:")
        logger.info(f"  Current Memory: {final_stats['current_memory_mb']:.1f}MB")
        logger.info(f"  Baseline Memory: {final_stats['baseline_memory_mb']:.1f}MB")
        logger.info(f"  Memory Delta: {final_stats['memory_delta_mb']:+.1f}MB")
        logger.info(f"  Total Snapshots: {final_stats['total_snapshots']}")
        logger.info(f"  Active Operations: {final_stats['active_operations']}")

    finally:
        # Stop memory monitoring
        stop_memory_monitoring()

    logger.info("Memory profiling demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
