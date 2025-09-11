"""T040: Global logging configuration for REER × DSPy × MLX.

Provides centralized logging configuration with:
- Structured JSON and text output
- Performance monitoring
- Trace correlation
- Component-specific loggers
- File rotation and management
"""

from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone
import json
import logging
import logging.config
from pathlib import Path
from typing import Any

# Ensure config directory exists
config_dir = Path(__file__).parent
config_dir.mkdir(exist_ok=True)


@dataclass
class LoggingConfig:
    """Global logging configuration."""

    # Logging levels
    root_level: str = "INFO"
    module_levels: dict[str, str] = field(default_factory=dict)

    # Output configuration
    console_enabled: bool = True
    file_enabled: bool = True
    json_format: bool = True

    # File configuration
    log_dir: Path = field(default_factory=lambda: Path.home() / ".reer" / "logs")
    main_log_file: str = "reer.log"
    error_log_file: str = "error.log"
    performance_log_file: str = "performance.log"

    # Rotation settings
    max_file_size: str = "10MB"
    backup_count: int = 5

    # Performance tracking
    performance_logging: bool = True
    trace_correlation: bool = True

    # Component-specific settings
    component_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log records."""

    def filter(self, record):
        # Only pass records that have performance metrics
        return hasattr(record, "duration_ms") or hasattr(record, "performance_metrics")


class TraceFilter(logging.Filter):
    """Filter to add trace information to log records."""

    def filter(self, record):
        # Add trace ID if not present
        if not hasattr(record, "trace_id"):
            record.trace_id = getattr(record, "_trace_id", "no-trace")

        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Add trace information
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        # Add performance metrics
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, "performance_metrics"):
            log_entry["performance"] = record.performance_metrics

        # Add component information
        if hasattr(record, "component"):
            log_entry["component"] = record.component

        # Add custom fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with trace and performance info."""

    def __init__(self, include_trace=True, include_performance=True):
        self.include_trace = include_trace
        self.include_performance = include_performance

        # Build format string
        format_parts = [
            "%(asctime)s",
            "%(levelname)-8s",
            "%(name)s:%(funcName)s:%(lineno)d",
        ]

        if include_trace:
            format_parts.append("trace=%(trace_id)s")

        if include_performance:
            format_parts.append("duration=%(duration_ms)sms")

        format_parts.append("%(message)s")

        format_string = " | ".join(format_parts)

        super().__init__(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Add default values for optional fields
        if not hasattr(record, "trace_id"):
            record.trace_id = "no-trace"

        if not hasattr(record, "duration_ms"):
            record.duration_ms = "-"

        return super().format(record)


def create_logging_config(config: LoggingConfig) -> dict[str, Any]:
    """Create logging configuration dictionary."""

    # Ensure log directory exists
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "performance": {"()": PerformanceFilter},
            "trace": {"()": TraceFilter},
        },
        "formatters": {
            "json": {"()": JSONFormatter},
            "text": {
                "()": TextFormatter,
                "include_trace": config.trace_correlation,
                "include_performance": config.performance_logging,
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {"null": {"class": "logging.NullHandler"}},
        "loggers": {
            # Root logger
            "": {"level": config.root_level, "handlers": []}
        },
    }

    # Console handler
    if config.console_enabled:
        console_formatter = "json" if config.json_format else "text"
        logging_config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": console_formatter,
            "filters": ["trace"],
        }
        logging_config["loggers"][""]["handlers"].append("console")

    # File handlers
    if config.file_enabled:
        # Main log file
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(config.log_dir / config.main_log_file),
            "maxBytes": _parse_size(config.max_file_size),
            "backupCount": config.backup_count,
            "formatter": "json" if config.json_format else "text",
            "filters": ["trace"],
        }
        logging_config["loggers"][""]["handlers"].append("file")

        # Error log file
        logging_config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(config.log_dir / config.error_log_file),
            "maxBytes": _parse_size(config.max_file_size),
            "backupCount": config.backup_count,
            "formatter": "json" if config.json_format else "text",
            "level": "ERROR",
            "filters": ["trace"],
        }
        logging_config["loggers"][""]["handlers"].append("error_file")

        # Performance log file
        if config.performance_logging:
            logging_config["handlers"]["performance_file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(config.log_dir / config.performance_log_file),
                "maxBytes": _parse_size(config.max_file_size),
                "backupCount": config.backup_count,
                "formatter": "json",
                "filters": ["performance", "trace"],
            }

    # Module-specific loggers
    for module, level in config.module_levels.items():
        logging_config["loggers"][module] = {
            "level": level,
            "handlers": [],
            "propagate": True,
        }

    # Component-specific configurations
    for component, comp_config in config.component_configs.items():
        if component not in logging_config["loggers"]:
            logging_config["loggers"][component] = {
                "level": comp_config.get("level", config.root_level),
                "handlers": [],
                "propagate": comp_config.get("propagate", True),
            }

    return logging_config


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes."""
    size_str = size_str.upper()

    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    if size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    if size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    return int(size_str)


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Setup global logging configuration.

    Args:
        config: Logging configuration (uses default if None)
    """
    if config is None:
        config = LoggingConfig()

    # Create logging configuration
    logging_config = create_logging_config(config)

    # Apply configuration
    logging.config.dictConfig(logging_config)

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            "extra_fields": {
                "config_file": str(config.log_dir / "logging.json"),
                "log_level": config.root_level,
                "json_format": config.json_format,
            }
        },
    )


def get_component_logger(
    component_name: str, level: str | None = None, trace_id: str | None = None
) -> logging.Logger:
    """Get a logger for a specific component.

    Args:
        component_name: Name of the component
        level: Logging level override
        trace_id: Optional trace ID for correlation

    Returns:
        Configured logger
    """
    logger = logging.getLogger(component_name)

    if level:
        logger.setLevel(getattr(logging, level.upper()))

    # Add trace ID if provided
    if trace_id:
        logger = LoggerAdapter(logger, {"trace_id": trace_id})

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context information."""

    def process(self, msg, kwargs):
        # Add extra context to all log records
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"].update(self.extra)

        return msg, kwargs


class PerformanceLogger:
    """Performance logging utility."""

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"Operation completed: {self.operation_name}",
                extra={
                    "duration_ms": duration_ms,
                    "performance_metrics": {
                        "operation": self.operation_name,
                        "success": True,
                    },
                },
            )
        else:
            self.logger.error(
                f"Operation failed: {self.operation_name}",
                extra={
                    "duration_ms": duration_ms,
                    "performance_metrics": {
                        "operation": self.operation_name,
                        "success": False,
                        "error_type": exc_type.__name__ if exc_type else None,
                    },
                },
                exc_info=True,
            )


# ============================================================================
# Pre-configured Logging Setups
# ============================================================================


def setup_development_logging() -> None:
    """Setup logging for development environment."""
    config = LoggingConfig(
        root_level="DEBUG",
        console_enabled=True,
        file_enabled=True,
        json_format=False,  # Human-readable for development
        module_levels={
            "urllib3": "WARNING",
            "requests": "WARNING",
            "asyncio": "WARNING",
        },
    )
    setup_logging(config)


def setup_production_logging() -> None:
    """Setup logging for production environment."""
    config = LoggingConfig(
        root_level="INFO",
        console_enabled=True,
        file_enabled=True,
        json_format=True,  # Structured for production
        module_levels={"urllib3": "ERROR", "requests": "ERROR", "asyncio": "ERROR"},
        component_configs={
            "core": {"level": "INFO"},
            "plugins": {"level": "INFO"},
            "scripts": {"level": "INFO"},
        },
    )
    setup_logging(config)


def setup_testing_logging() -> None:
    """Setup logging for testing environment."""
    config = LoggingConfig(
        root_level="WARNING",
        console_enabled=False,
        file_enabled=False,
        json_format=False,
    )
    setup_logging(config)


# ============================================================================
# Convenience Functions
# ============================================================================


def log_performance(operation_name: str, logger: logging.Logger | None = None):
    """Decorator for performance logging.

    Args:
        operation_name: Name of the operation
        logger: Logger to use (creates one if None)

    Returns:
        Decorator function
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(logger, operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


async def async_log_performance(
    operation_name: str, logger: logging.Logger | None = None
):
    """Async context manager for performance logging."""
    if logger is None:
        logger = logging.getLogger(__name__)

    return PerformanceLogger(logger, operation_name)


def create_trace_logger(
    base_logger: logging.Logger, trace_id: str
) -> logging.LoggerAdapter:
    """Create a logger with trace ID correlation.

    Args:
        base_logger: Base logger
        trace_id: Trace ID for correlation

    Returns:
        Logger adapter with trace ID
    """
    return LoggerAdapter(base_logger, {"trace_id": trace_id})


# ============================================================================
# Module-specific Logger Setup
# ============================================================================


def setup_module_loggers() -> None:
    """Setup loggers for all REER modules."""

    # Core modules
    core_loggers = [
        "core.trace_store",
        "core.trajectory_synthesizer",
        "core.candidate_scorer",
        "core.trainer",
        "core.integration",
    ]

    # Plugin modules
    plugin_loggers = [
        "plugins.mlx_lm",
        "plugins.dspy_lm",
        "plugins.dspy_pipeline",
        "plugins.lm_registry",
    ]

    # Script modules
    script_loggers = ["scripts.cli_mlx", "scripts.social_reer", "scripts.cli_common"]

    # Set appropriate levels
    for logger_name in core_loggers + plugin_loggers + script_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)


# Initialize logging on import
if __name__ != "__main__":
    # Setup default logging configuration
    try:
        setup_production_logging()
        setup_module_loggers()
    except Exception:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        )
