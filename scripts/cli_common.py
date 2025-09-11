"""T037: Common CLI utilities for REER × DSPy × MLX integration.

Provides shared functionality for all CLI scripts including:
- LM registry integration
- Structured logging setup
- Rate limiting configuration
- Common error handling
- Performance monitoring
"""

import asyncio
from collections.abc import Callable
from functools import wraps
from pathlib import Path
import sys
import time
from typing import Any, TypeVar

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.exceptions import REERBaseException
from core.integration import (
    IntegratedREERMiner,
    LoggingConfig,
    RateLimitConfig,
    REERMiningConfig,
)
from plugins.lm_registry import (
    get_recommended_model,
    get_registry,
)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global console for rich output
console = Console()

# Global configuration
_cli_config = {
    "logging_level": "INFO",
    "default_provider": "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit",
    "trace_store_path": Path.home() / ".reer" / "traces.jsonl",
    "log_file": Path.home() / ".reer" / "logs" / "cli.log",
    "rate_limit_enabled": True,
}


# ============================================================================
# Configuration Management
# ============================================================================


def get_cli_config() -> dict[str, Any]:
    """Get current CLI configuration."""
    return _cli_config.copy()


def set_cli_config(**kwargs) -> None:
    """Update CLI configuration."""
    _cli_config.update(kwargs)


def ensure_reer_directory() -> Path:
    """Ensure REER directory exists and return path."""
    reer_dir = Path.home() / ".reer"
    reer_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (reer_dir / "logs").mkdir(exist_ok=True)
    (reer_dir / "cache").mkdir(exist_ok=True)
    (reer_dir / "models").mkdir(exist_ok=True)

    return reer_dir


# ============================================================================
# LM Registry Integration
# ============================================================================


class CLIModelManager:
    """Model manager for CLI applications."""

    def __init__(self):
        self.registry = get_registry()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the model manager."""
        if self._initialized:
            return

        console.print("[cyan]Initializing language model registry...[/cyan]")

        try:
            # Check health of all providers
            health_status = await self.registry.health_check()

            # Display provider status
            self._display_provider_status(health_status)

            self._initialized = True
            console.print("[green]✓ Language model registry initialized[/green]")

        except Exception as e:
            console.print(f"[red]Failed to initialize LM registry: {e}[/red]")
            raise

    def _display_provider_status(
        self, health_status: dict[str, dict[str, Any]]
    ) -> None:
        """Display provider health status."""

        status_table = Table(title="Language Model Providers")
        status_table.add_column("Provider", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Default Model", style="yellow")
        status_table.add_column("Capabilities", style="blue")

        for provider, status in health_status.items():
            status_icon = "✅" if status.get("available", False) else "❌"
            capabilities = ", ".join(status.get("capabilities", []))

            status_table.add_row(
                provider, status_icon, status.get("default_model", "N/A"), capabilities
            )

        console.print(status_table)

    async def get_adapter(self, uri: str | None = None, **kwargs):
        """Get language model adapter."""
        if not self._initialized:
            await self.initialize()

        if not uri:
            uri = _cli_config["default_provider"]

        return await self.registry.get_adapter(uri, **kwargs)

    async def generate_text(self, prompt: str, uri: str | None = None, **kwargs) -> str:
        """Generate text using the registry."""
        if not uri:
            uri = _cli_config["default_provider"]

        return await self.registry.route_generate(uri, prompt, **kwargs)

    def get_recommended_model(self, use_case: str = "general") -> str:
        """Get recommended model for use case."""
        return get_recommended_model(use_case, prefer_local=True)

    def list_providers(self) -> None:
        """List all available providers."""
        providers = self.registry.list_providers()

        providers_table = Table(title="Available Providers")
        providers_table.add_column("Name", style="cyan")
        providers_table.add_column("Scheme", style="green")
        providers_table.add_column("Priority", style="yellow")
        providers_table.add_column("Models", style="blue")

        for provider in sorted(providers, key=lambda p: p.priority):
            model_count = len(provider.supported_models)
            providers_table.add_row(
                provider.name,
                provider.scheme,
                str(provider.priority),
                f"{model_count} models",
            )

        console.print(providers_table)


# Global model manager instance
_model_manager = None


def get_model_manager() -> CLIModelManager:
    """Get global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = CLIModelManager()
    return _model_manager


# ============================================================================
# Mining Service Integration
# ============================================================================


class CLIMiningService:
    """Mining service for CLI applications."""

    def __init__(self):
        self._service: IntegratedREERMiner | None = None
        self._config: REERMiningConfig | None = None

    async def initialize(
        self, trace_store_path: Path | None = None, rate_limit_enabled: bool = True
    ) -> None:
        """Initialize the mining service."""

        if not trace_store_path:
            trace_store_path = _cli_config["trace_store_path"]

        # Ensure directory exists
        trace_store_path.parent.mkdir(parents=True, exist_ok=True)

        # Create configuration
        rate_config = (
            RateLimitConfig()
            if rate_limit_enabled
            else RateLimitConfig(
                max_requests_per_minute=10000,  # Effectively unlimited
                max_requests_per_hour=100000,
            )
        )

        logging_config = LoggingConfig(
            level=_cli_config["logging_level"], log_file=_cli_config["log_file"]
        )

        self._config = REERMiningConfig(
            trace_store_path=trace_store_path,
            rate_limit_config=rate_config,
            logging_config=logging_config,
            default_provider_uri=_cli_config["default_provider"],
        )

        # Create service
        self._service = IntegratedREERMiner(self._config)

        console.print("[green]✓ Mining service initialized[/green]")

    async def extract_and_store(self, *args, **kwargs) -> str:
        """Extract and store strategy (delegated to service)."""
        if not self._service:
            await self.initialize()

        return await self._service.extract_and_store(*args, **kwargs)

    async def query_traces(self, *args, **kwargs) -> list:
        """Query traces (delegated to service)."""
        if not self._service:
            await self.initialize()

        return await self._service.query_traces(*args, **kwargs)

    async def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self._service:
            await self.initialize()

        return await self._service.get_performance_stats()


# Global mining service instance
_mining_service = None


def get_mining_service() -> CLIMiningService:
    """Get global mining service instance."""
    global _mining_service
    if _mining_service is None:
        _mining_service = CLIMiningService()
    return _mining_service


# ============================================================================
# Decorators and Context Managers
# ============================================================================


def with_lm_registry(func: F) -> F:
    """Decorator to initialize LM registry for CLI commands."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        model_manager = get_model_manager()
        await model_manager.initialize()

        # Add model_manager to kwargs if not present
        if "model_manager" not in kwargs:
            kwargs["model_manager"] = model_manager

        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Run the async wrapper
        return asyncio.run(async_wrapper(*args, **kwargs))

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def with_mining_service(func: F) -> F:
    """Decorator to initialize mining service for CLI commands."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        mining_service = get_mining_service()
        await mining_service.initialize()

        # Add mining_service to kwargs if not present
        if "mining_service" not in kwargs:
            kwargs["mining_service"] = mining_service

        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        return asyncio.run(async_wrapper(*args, **kwargs))

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def with_error_handling(func: F) -> F:
    """Decorator to add comprehensive error handling to CLI commands."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except REERBaseException as e:
            console.print(f"[red]REER Error:[/red] {e.message}")
            if e.details:
                console.print(f"[yellow]Details:[/yellow] {e.details}")
            logger.error(f"REER error in {func.__name__}: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise typer.Exit(1)

    return wrapper


def performance_monitor(operation_name: str):
    """Decorator to monitor performance of CLI operations."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                console.print(
                    f"[green]✓ {operation_name} completed in {duration:.2f}s[/green]"
                )
                return result

            except Exception:
                duration = time.time() - start_time
                console.print(
                    f"[red]✗ {operation_name} failed after {duration:.2f}s[/red]"
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                console.print(
                    f"[green]✓ {operation_name} completed in {duration:.2f}s[/green]"
                )
                return result

            except Exception:
                duration = time.time() - start_time
                console.print(
                    f"[red]✗ {operation_name} failed after {duration:.2f}s[/red]"
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================================
# Common CLI Utilities
# ============================================================================


def setup_cli_logging(level: str = "INFO", enable_file_logging: bool = True) -> None:
    """Setup CLI logging configuration."""

    # Update global config
    set_cli_config(logging_level=level)

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
    )

    if enable_file_logging:
        log_file = _cli_config["log_file"]
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days",
        )


def display_model_info(model_uri: str) -> None:
    """Display information about a model URI."""

    try:
        registry = get_registry()
        model_ref = registry.parse_model_uri(model_uri)

        info_table = Table(title=f"Model Information: {model_uri}")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Provider", model_ref.provider)
        info_table.add_row("Model Path", model_ref.model_path)
        info_table.add_row("URI", model_ref.uri)

        if model_ref.parameters:
            for key, value in model_ref.parameters.items():
                info_table.add_row(f"Parameter: {key}", str(value))

        console.print(info_table)

    except Exception as e:
        console.print(f"[red]Failed to parse model URI:[/red] {e}")


def display_trace_stats(traces: list) -> None:
    """Display statistics for a list of traces."""

    if not traces:
        console.print("[yellow]No traces found[/yellow]")
        return

    # Calculate statistics
    scores = [trace.get("score", 0) for trace in traces]
    providers = [trace.get("provider", "unknown") for trace in traces]

    stats_table = Table(title="Trace Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Traces", str(len(traces)))
    stats_table.add_row("Average Score", f"{sum(scores) / len(scores):.3f}")
    stats_table.add_row("Min Score", f"{min(scores):.3f}")
    stats_table.add_row("Max Score", f"{max(scores):.3f}")

    # Provider distribution
    from collections import Counter

    provider_counts = Counter(providers)
    most_common = provider_counts.most_common(1)[0]
    stats_table.add_row(
        "Most Used Provider", f"{most_common[0]} ({most_common[1]} traces)"
    )

    console.print(stats_table)


def show_progress(description: str, total: int | None = None):
    """Create a progress context manager."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


# ============================================================================
# Common CLI Options
# ============================================================================


def provider_option(default: str | None = None):
    """Common provider URI option for CLI commands."""
    return typer.Option(
        default or _cli_config["default_provider"],
        "--provider",
        "-p",
        help="Language model provider URI (e.g., mlx://model-name, dspy://provider/model)",
    )


def trace_store_option(default: Path | None = None):
    """Common trace store path option for CLI commands."""
    return typer.Option(
        default or _cli_config["trace_store_path"],
        "--trace-store",
        "-t",
        help="Path to trace store file",
    )


def logging_level_option():
    """Common logging level option for CLI commands."""
    return typer.Option(
        "INFO", "--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


def rate_limit_option():
    """Common rate limiting option for CLI commands."""
    return typer.Option(
        True, "--rate-limit/--no-rate-limit", help="Enable rate limiting for API calls"
    )


# ============================================================================
# Initialization Functions
# ============================================================================


def init_cli_environment(
    logging_level: str = "INFO",
    provider_uri: str | None = None,
    enable_rate_limiting: bool = True,
) -> None:
    """Initialize the CLI environment with all necessary components."""

    # Setup logging
    setup_cli_logging(logging_level)

    # Update configuration
    config_updates = {
        "logging_level": logging_level,
        "rate_limit_enabled": enable_rate_limiting,
    }

    if provider_uri:
        config_updates["default_provider"] = provider_uri

    set_cli_config(**config_updates)

    # Ensure REER directory exists
    ensure_reer_directory()

    logger.info("CLI environment initialized")


async def verify_integrations() -> bool:
    """Verify that all integrations are working properly."""

    console.print("[cyan]Verifying system integrations...[/cyan]")

    try:
        # Test LM registry
        model_manager = get_model_manager()
        await model_manager.initialize()

        # Test mining service
        mining_service = get_mining_service()
        await mining_service.initialize()

        # Test basic operations
        _cli_config["default_provider"]

        console.print("[green]✓ All integrations verified successfully[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Integration verification failed: {e}[/red]")
        return False
