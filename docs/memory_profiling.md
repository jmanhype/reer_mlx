# REER MLX Memory Profiling Tools

This document describes the comprehensive memory profiling and optimization tools available in the REER MLX system.

## Overview

The memory profiling system provides:

- **Real-time memory tracking** with automatic alerts
- **Decorators and context managers** for profiling specific operations
- **Model-specific profiling** for ML model memory usage
- **Memory leak detection** and optimization recommendations
- **Streaming data processing** with minimal memory footprint
- **Resource management** with automatic cleanup
- **Emergency memory handling** for critical situations

## Quick Start

### Basic Usage

```python
from tools.memory_profiler import memory_profile, memory_context

# Decorate functions for automatic profiling
@memory_profile(operation_name="data_processing")
def process_data(data):
    return [x * 2 for x in data]

# Use context managers for code blocks
with memory_context("critical_operation"):
    result = expensive_computation()
```

### Setup Memory Monitoring

```python
from tools.memory_profiler import setup_memory_limits, start_memory_monitoring

# Configure thresholds
setup_memory_limits(
    alert_mb=1000.0,      # Warning threshold
    critical_mb=2000.0,   # Critical threshold
    cleanup_mb=1500.0     # Automatic cleanup threshold
)

# Start background monitoring
start_memory_monitoring()
```

## Core Components

### 1. MemoryTracker

Tracks memory usage and detects potential issues.

```python
from tools.memory_profiler import get_memory_tracker

tracker = get_memory_tracker()
stats = tracker.get_memory_stats()
print(f"Current memory: {stats['current_memory_mb']:.1f}MB")
```

### 2. MemoryProfiler

Advanced profiling with model-specific optimizations.

```python
from tools.memory_profiler import get_memory_profiler

profiler = get_memory_profiler()

# Profile model loading
model = profiler.profile_model_loading("my_model", load_model_func)

# Get recommendations
recommendations = profiler.get_model_memory_recommendations()
```

### 3. MemoryMonitor

Continuous monitoring with automatic alerts and cleanup.

```python
from tools.memory_profiler import MemoryMonitor

monitor = MemoryMonitor(
    alert_threshold_mb=1000.0,
    critical_threshold_mb=2000.0,
    cleanup_threshold_mb=1500.0,
    monitoring_interval=30.0  # seconds
)

monitor.start_monitoring()
```

### 4. MemoryOptimizer

Utilities for memory optimization.

```python
from tools.memory_profiler import MemoryOptimizer

# Clean up large objects
cleaned = MemoryOptimizer.cleanup_large_objects(size_threshold_mb=10.0)

# Force garbage collection
collected = MemoryOptimizer.force_garbage_collection()

# Optimize numpy arrays
MemoryOptimizer.optimize_numpy_arrays(my_array)
```

### 5. LazyModelLoader

Lazy loading for ML models to reduce memory footprint.

```python
from tools.memory_profiler import LazyModelLoader

# Create lazy loader
lazy_model = LazyModelLoader(load_model_func, model_path="path/to/model")

# Model is loaded on first access
predictions = lazy_model.predict(input_data)

# Unload to free memory
lazy_model.unload()
```

### 6. StreamingJSONLReader

Memory-efficient JSONL processing.

```python
from tools.memory_profiler import StreamingJSONLReader

reader = StreamingJSONLReader(file_path, chunk_size=8192)

for record in reader:
    process_record(record)  # Process one record at a time
```

### 7. MemoryResourceManager

Automatic resource management and cleanup.

```python
from tools.memory_profiler import get_resource_manager

manager = get_resource_manager()

# Register resource with cleanup callback
manager.register_resource(
    "my_resource", resource_object,
    cleanup_callback=cleanup_func,
    memory_limit_mb=100.0
)

# Clean up all resources
cleaned = manager.cleanup_all_resources()
```

## Trace Store Memory Optimizations

The trace store includes several memory-optimized methods:

### Lazy Querying

```python
from core.trace_store import REERTraceStore

store = REERTraceStore(path)

# Stream results instead of loading all into memory
async for trace in store.lazy_query_traces(min_score=0.5, limit=1000):
    process_trace(trace)
```

### Memory-Efficient Statistics

```python
# Get stats without loading all traces into memory
stats = await store.get_memory_efficient_stats()
print(f"Total traces: {stats['total_traces']}")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

### Streaming Filtered Processing

```python
# Process traces in chunks with filtering
async for chunk in store.stream_traces_filtered(
    filter_func=lambda t: t.get("score", 0) > 0.7,
    chunk_size=1000
):
    process_chunk(chunk)
```

### Memory Optimization

```python
# Optimize trace store memory usage
result = store.optimize_memory_usage()
print(f"Freed {result['memory_freed_mb']:.1f}MB")
```

## Advanced Features

### Emergency Memory Handling

```python
from tools.memory_profiler import setup_emergency_memory_handler

# Setup signal handler for memory emergencies
setup_emergency_memory_handler()

# Trigger emergency cleanup (send SIGUSR1 to process)
# or programmatically:
from tools.memory_profiler import _memory_monitor
_memory_monitor._trigger_emergency_cleanup()
```

### Memory Leak Detection

The system automatically detects potential memory leaks:

```python
# Memory profiling results include leak detection
@memory_profile()
def my_function():
    # Function that might leak memory
    pass

# Check profiling results for potential leaks
tracker = get_memory_tracker()
for snapshot in tracker.snapshots:
    if snapshot.operation:
        print(f"Operation: {snapshot.operation}")
```

### Custom Memory Callbacks

```python
def memory_callback(snapshot):
    if snapshot.process_memory_mb > 1500:
        logger.warning(f"High memory usage: {snapshot.process_memory_mb:.1f}MB")

monitor = MemoryMonitor()
monitor.add_callback(memory_callback)
monitor.start_monitoring()
```

## Configuration

### Environment Variables

- `REER_MEMORY_ALERT_MB`: Alert threshold in MB (default: 1000)
- `REER_MEMORY_CRITICAL_MB`: Critical threshold in MB (default: 2000)
- `REER_MEMORY_CLEANUP_MB`: Cleanup threshold in MB (default: 1500)
- `REER_MEMORY_MONITORING_INTERVAL`: Monitoring interval in seconds (default: 30)

### Programmatic Configuration

```python
from tools.memory_profiler import setup_memory_limits, set_memory_limit

# Setup all limits at once
setup_memory_limits(alert_mb=800.0, critical_mb=1600.0, cleanup_mb=1200.0)

# Or set individual limits
set_memory_limit(1500.0)  # Sets critical threshold
```

## Best Practices

### 1. Use Decorators for Functions

```python
@memory_profile(operation_name="data_processing")
def process_large_dataset(data):
    # Your code here
    pass
```

### 2. Use Context Managers for Code Blocks

```python
with memory_context("critical_section"):
    # Memory-intensive code
    pass
```

### 3. Monitor Long-Running Applications

```python
# At application startup
setup_memory_limits()
start_memory_monitoring()
setup_emergency_memory_handler()

# At application shutdown
stop_memory_monitoring()
```

### 4. Use Lazy Loading for Large Models

```python
# Instead of loading immediately
model = LazyModelLoader(load_model_func, model_path)

# Model loads only when needed
predictions = model.predict(data)
```

### 5. Stream Large Datasets

```python
# Instead of loading all data
reader = StreamingJSONLReader(large_file)
for record in reader:
    process_record(record)
```

### 6. Clean Up Resources Explicitly

```python
# Register important resources
get_resource_manager().register_resource(
    "model", model, cleanup_callback=model.cleanup
)

# Clean up when done
get_resource_manager().cleanup_all_resources()
```

## Performance Considerations

- Memory monitoring has minimal overhead (< 1% CPU)
- Profiling decorators add ~0.1ms per function call
- Streaming readers use constant memory regardless of file size
- Garbage collection is triggered strategically to minimize impact

## Troubleshooting

### High Memory Usage Warnings

1. Check for memory leaks in profiling results
2. Use lazy loading for large objects
3. Implement streaming for large datasets
4. Clean up unused resources explicitly

### Memory Limit Exceeded Errors

1. Increase memory limits if appropriate
2. Optimize data structures
3. Use chunking/batching for large operations
4. Enable automatic cleanup

### Performance Issues

1. Reduce monitoring frequency
2. Use sampling for high-frequency operations
3. Optimize garbage collection strategy
4. Consider memory mapping for large files

## Examples

See `/examples/memory_profiling_example.py` for comprehensive usage examples.

## API Reference

For detailed API documentation, see the docstrings in:
- `/tools/memory_profiler.py` - Core profiling tools
- `/core/trace_store.py` - Memory-optimized trace storage

## Contributing

When adding new memory-intensive operations:

1. Add `@memory_profile()` decorators
2. Use streaming when processing large datasets
3. Register resources with the resource manager
4. Test with memory limits enabled
5. Add appropriate error handling for memory limits