"""
Memory profiling configuration for REER MLX system.
"""

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class MemoryConfig:
    """Configuration for memory profiling and optimization."""

    # Memory thresholds in MB
    alert_threshold_mb: float = 1000.0
    critical_threshold_mb: float = 2000.0
    cleanup_threshold_mb: float = 1500.0

    # Monitoring settings
    monitoring_interval_seconds: float = 30.0
    monitoring_enabled: bool = True

    # Profiling settings
    profiling_enabled: bool = True
    log_profiling_results: bool = True
    detailed_profiling: bool = False

    # Optimization settings
    auto_cleanup_enabled: bool = True
    cleanup_large_objects_threshold_mb: float = 10.0
    force_gc_on_cleanup: bool = True

    # Model loading settings
    lazy_loading_enabled: bool = True
    model_unload_after_use: bool = False
    model_memory_limit_mb: float = 500.0

    # Streaming settings
    default_chunk_size: int = 8192
    streaming_enabled: bool = True
    adaptive_chunk_sizing: bool = True

    # Emergency settings
    emergency_handler_enabled: bool = True
    emergency_cleanup_aggressive: bool = True

    # Reporting settings
    save_memory_reports: bool = True
    report_directory: str = "/tmp/reer_memory_reports"
    report_retention_days: int = 7

    # Advanced settings
    leak_detection_enabled: bool = True
    gc_stats_collection: bool = True
    object_tracking_enabled: bool = True

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create configuration from environment variables."""
        return cls(
            alert_threshold_mb=float(os.getenv("REER_MEMORY_ALERT_MB", 1000.0)),
            critical_threshold_mb=float(os.getenv("REER_MEMORY_CRITICAL_MB", 2000.0)),
            cleanup_threshold_mb=float(os.getenv("REER_MEMORY_CLEANUP_MB", 1500.0)),
            monitoring_interval_seconds=float(
                os.getenv("REER_MEMORY_MONITORING_INTERVAL", 30.0)
            ),
            monitoring_enabled=os.getenv(
                "REER_MEMORY_MONITORING_ENABLED", "true"
            ).lower()
            == "true",
            profiling_enabled=os.getenv("REER_MEMORY_PROFILING_ENABLED", "true").lower()
            == "true",
            log_profiling_results=os.getenv("REER_MEMORY_LOG_RESULTS", "true").lower()
            == "true",
            detailed_profiling=os.getenv(
                "REER_MEMORY_DETAILED_PROFILING", "false"
            ).lower()
            == "true",
            auto_cleanup_enabled=os.getenv("REER_MEMORY_AUTO_CLEANUP", "true").lower()
            == "true",
            cleanup_large_objects_threshold_mb=float(
                os.getenv("REER_MEMORY_CLEANUP_THRESHOLD_MB", 10.0)
            ),
            force_gc_on_cleanup=os.getenv("REER_MEMORY_FORCE_GC", "true").lower()
            == "true",
            lazy_loading_enabled=os.getenv("REER_MEMORY_LAZY_LOADING", "true").lower()
            == "true",
            model_unload_after_use=os.getenv(
                "REER_MEMORY_MODEL_UNLOAD", "false"
            ).lower()
            == "true",
            model_memory_limit_mb=float(os.getenv("REER_MEMORY_MODEL_LIMIT_MB", 500.0)),
            default_chunk_size=int(os.getenv("REER_MEMORY_CHUNK_SIZE", 8192)),
            streaming_enabled=os.getenv("REER_MEMORY_STREAMING", "true").lower()
            == "true",
            adaptive_chunk_sizing=os.getenv(
                "REER_MEMORY_ADAPTIVE_CHUNKS", "true"
            ).lower()
            == "true",
            emergency_handler_enabled=os.getenv(
                "REER_MEMORY_EMERGENCY_HANDLER", "true"
            ).lower()
            == "true",
            emergency_cleanup_aggressive=os.getenv(
                "REER_MEMORY_EMERGENCY_AGGRESSIVE", "true"
            ).lower()
            == "true",
            save_memory_reports=os.getenv("REER_MEMORY_SAVE_REPORTS", "true").lower()
            == "true",
            report_directory=os.getenv(
                "REER_MEMORY_REPORT_DIR", "/tmp/reer_memory_reports"
            ),
            report_retention_days=int(
                os.getenv("REER_MEMORY_REPORT_RETENTION_DAYS", 7)
            ),
            leak_detection_enabled=os.getenv(
                "REER_MEMORY_LEAK_DETECTION", "true"
            ).lower()
            == "true",
            gc_stats_collection=os.getenv("REER_MEMORY_GC_STATS", "true").lower()
            == "true",
            object_tracking_enabled=os.getenv(
                "REER_MEMORY_OBJECT_TRACKING", "true"
            ).lower()
            == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "alert_threshold_mb": self.alert_threshold_mb,
            "critical_threshold_mb": self.critical_threshold_mb,
            "cleanup_threshold_mb": self.cleanup_threshold_mb,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "monitoring_enabled": self.monitoring_enabled,
            "profiling_enabled": self.profiling_enabled,
            "log_profiling_results": self.log_profiling_results,
            "detailed_profiling": self.detailed_profiling,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "cleanup_large_objects_threshold_mb": self.cleanup_large_objects_threshold_mb,
            "force_gc_on_cleanup": self.force_gc_on_cleanup,
            "lazy_loading_enabled": self.lazy_loading_enabled,
            "model_unload_after_use": self.model_unload_after_use,
            "model_memory_limit_mb": self.model_memory_limit_mb,
            "default_chunk_size": self.default_chunk_size,
            "streaming_enabled": self.streaming_enabled,
            "adaptive_chunk_sizing": self.adaptive_chunk_sizing,
            "emergency_handler_enabled": self.emergency_handler_enabled,
            "emergency_cleanup_aggressive": self.emergency_cleanup_aggressive,
            "save_memory_reports": self.save_memory_reports,
            "report_directory": self.report_directory,
            "report_retention_days": self.report_retention_days,
            "leak_detection_enabled": self.leak_detection_enabled,
            "gc_stats_collection": self.gc_stats_collection,
            "object_tracking_enabled": self.object_tracking_enabled,
        }

    def validate(self) -> None:
        """Validate configuration values."""
        if self.alert_threshold_mb <= 0:
            raise ValueError("Alert threshold must be positive")

        if self.critical_threshold_mb <= self.alert_threshold_mb:
            raise ValueError("Critical threshold must be greater than alert threshold")

        if self.cleanup_threshold_mb <= self.alert_threshold_mb:
            raise ValueError("Cleanup threshold must be greater than alert threshold")

        if self.monitoring_interval_seconds <= 0:
            raise ValueError("Monitoring interval must be positive")

        if self.cleanup_large_objects_threshold_mb <= 0:
            raise ValueError("Cleanup threshold must be positive")

        if self.model_memory_limit_mb <= 0:
            raise ValueError("Model memory limit must be positive")

        if self.default_chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if self.report_retention_days < 0:
            raise ValueError("Report retention days cannot be negative")


# Default configuration instance
default_config = MemoryConfig()

# Environment-based configuration instance
env_config = MemoryConfig.from_env()


def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration."""
    try:
        env_config.validate()
        return env_config
    except ValueError as e:
        import logging

        logging.warning(
            f"Invalid memory configuration from environment: {e}. Using defaults."
        )
        default_config.validate()
        return default_config


def apply_memory_config(config: MemoryConfig = None) -> None:
    """Apply memory configuration to the profiling system."""
    if config is None:
        config = get_memory_config()

    from tools.memory_profiler import (
        get_memory_tracker,
        setup_emergency_memory_handler,
        setup_memory_limits,
        start_memory_monitoring,
    )

    # Apply memory limits
    setup_memory_limits(
        alert_mb=config.alert_threshold_mb,
        critical_mb=config.critical_threshold_mb,
        cleanup_mb=config.cleanup_threshold_mb,
    )

    # Configure tracker settings
    tracker = get_memory_tracker()
    tracker.memory_threshold_mb = config.critical_threshold_mb
    tracker.leak_detection_enabled = config.leak_detection_enabled

    # Setup emergency handler if enabled
    if config.emergency_handler_enabled:
        setup_emergency_memory_handler()

    # Start monitoring if enabled
    if config.monitoring_enabled:
        start_memory_monitoring()


# Configuration presets for different environments
class MemoryConfigPresets:
    """Predefined memory configuration presets."""

    @staticmethod
    def development() -> MemoryConfig:
        """Configuration for development environment."""
        return MemoryConfig(
            alert_threshold_mb=500.0,
            critical_threshold_mb=1000.0,
            cleanup_threshold_mb=750.0,
            monitoring_interval_seconds=10.0,
            detailed_profiling=True,
            log_profiling_results=True,
            auto_cleanup_enabled=True,
            save_memory_reports=True,
        )

    @staticmethod
    def testing() -> MemoryConfig:
        """Configuration for testing environment."""
        return MemoryConfig(
            alert_threshold_mb=200.0,
            critical_threshold_mb=400.0,
            cleanup_threshold_mb=300.0,
            monitoring_interval_seconds=5.0,
            detailed_profiling=False,
            log_profiling_results=False,
            auto_cleanup_enabled=True,
            save_memory_reports=False,
            emergency_handler_enabled=False,
        )

    @staticmethod
    def production() -> MemoryConfig:
        """Configuration for production environment."""
        return MemoryConfig(
            alert_threshold_mb=2000.0,
            critical_threshold_mb=4000.0,
            cleanup_threshold_mb=3000.0,
            monitoring_interval_seconds=60.0,
            detailed_profiling=False,
            log_profiling_results=False,
            auto_cleanup_enabled=True,
            save_memory_reports=True,
            emergency_handler_enabled=True,
            emergency_cleanup_aggressive=True,
        )

    @staticmethod
    def minimal() -> MemoryConfig:
        """Minimal configuration with profiling disabled."""
        return MemoryConfig(
            alert_threshold_mb=5000.0,  # Very high threshold
            critical_threshold_mb=10000.0,
            cleanup_threshold_mb=7500.0,
            monitoring_enabled=False,
            profiling_enabled=False,
            log_profiling_results=False,
            auto_cleanup_enabled=False,
            save_memory_reports=False,
            leak_detection_enabled=False,
            emergency_handler_enabled=False,
        )
