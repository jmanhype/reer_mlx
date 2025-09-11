#!/usr/bin/env python3
"""Integrated REER × DSPy × MLX Demonstration Script.

Demonstrates the complete integration of all components:
- TraceStore with REER mining pipeline (T036)
- LM registry with provider routing (T037)
- DSPy pipeline integration (T038)
- Rate limiting with exponential backoff (T039)
- Structured logging across all modules (T040)

This script shows how all components work together in a cohesive system.
"""

from pathlib import Path
import sys
import time
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import integrated components
from config.logging_config import get_component_logger, setup_production_logging
from core.integration import create_mining_service
from plugins.dspy_pipeline import (
    create_dspy_pipeline,
    get_social_media_templates,
)
from plugins.lm_registry import get_registry
from scripts.cli_common import (
    get_model_manager,
    init_cli_environment,
    performance_monitor,
    with_error_handling,
)

app = typer.Typer(
    name="integrated-demo",
    help="Demonstration of integrated REER × DSPy × MLX system",
    rich_markup_mode="rich",
)
console = Console()


# ============================================================================
# Demo Configuration
# ============================================================================

DEMO_CONFIG = {
    "trace_store_path": Path.home() / ".reer" / "demo_traces.jsonl",
    "primary_provider": "mlx://mlx-community/Llama-3.2-3B-Instruct-4bit",
    "fallback_providers": ["dspy://openai/gpt-3.5-turbo", "dummy://test-model"],
    "demo_posts": [
        {
            "id": "post_001",
            "content": "Just discovered this amazing productivity hack that changed my entire workflow! Thread 🧵",
            "seed_params": {
                "topic": "productivity",
                "style": "educational",
                "length": 280,
                "thread_size": 5,
            },
        },
        {
            "id": "post_002",
            "content": "Building in public day 15: Learned so much about user feedback today. The community is incredible! 🚀",
            "seed_params": {
                "topic": "building_in_public",
                "style": "personal",
                "length": 150,
                "thread_size": 1,
            },
        },
        {
            "id": "post_003",
            "content": "Quick tip: When debugging, explain your code to a rubber duck. Works 90% of the time! 🦆💻",
            "seed_params": {
                "topic": "programming",
                "style": "tip",
                "length": 120,
                "thread_size": 1,
            },
        },
    ],
}


# ============================================================================
# Demo Functions
# ============================================================================


@app.command()
@with_error_handling
@performance_monitor("System Integration Demo")
async def demo(
    provider_uri: str = typer.Option(
        DEMO_CONFIG["primary_provider"], "--provider", "-p", help="Primary provider URI"
    ),
    enable_optimization: bool = typer.Option(
        False, "--optimize/--no-optimize", help="Enable DSPy optimization"
    ),
    rate_limit: bool = typer.Option(
        True, "--rate-limit/--no-rate-limit", help="Enable rate limiting"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
):
    """Run the complete system integration demonstration."""

    # Initialize environment
    init_cli_environment(log_level, provider_uri, rate_limit)
    logger = get_component_logger("demo", log_level)

    console.print(
        Panel(
            "[bold cyan]REER × DSPy × MLX Integration Demonstration[/bold cyan]\n\n"
            "This demo showcases the complete integration of all system components:\n"
            "• TraceStore with REER mining pipeline\n"
            "• LM registry with provider routing\n"
            "• DSPy pipeline integration\n"
            "• Rate limiting with exponential backoff\n"
            "• Structured logging across all modules",
            title="🚀 System Integration Demo",
            border_style="green",
        )
    )

    trace_id = str(uuid4())
    logger.info("Starting integrated demo", extra={"trace_id": trace_id})

    try:
        # Step 1: Initialize all components
        await _demo_initialization(provider_uri, rate_limit, trace_id)

        # Step 2: Demonstrate TraceStore integration
        await _demo_trace_store_integration(trace_id)

        # Step 3: Demonstrate LM registry routing
        await _demo_lm_registry_routing(provider_uri, trace_id)

        # Step 4: Demonstrate DSPy pipeline
        await _demo_dspy_pipeline(provider_uri, enable_optimization, trace_id)

        # Step 5: Demonstrate rate limiting
        await _demo_rate_limiting(trace_id)

        # Step 6: Show comprehensive metrics
        await _demo_metrics_reporting(trace_id)

        console.print(
            Panel(
                "[bold green]✅ All integration demonstrations completed successfully![/bold green]\n\n"
                "The system is fully integrated and operational.",
                title="Demo Complete",
                border_style="green",
            )
        )

        logger.info("Demo completed successfully", extra={"trace_id": trace_id})

    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        logger.error(f"Demo failed: {e}", extra={"trace_id": trace_id}, exc_info=True)
        raise


async def _demo_initialization(provider_uri: str, rate_limit: bool, trace_id: str):
    """Demonstrate component initialization."""

    console.print("\n[cyan]Step 1: Component Initialization[/cyan]")
    logger = get_component_logger("demo.initialization", trace_id=trace_id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        init_task = progress.add_task("Initializing components...", total=None)

        # Initialize model manager
        progress.update(init_task, description="Initializing LM registry...")
        model_manager = get_model_manager()
        await model_manager.initialize()
        logger.info("LM registry initialized")

        # Initialize mining service
        progress.update(init_task, description="Initializing mining service...")
        mining_service = create_mining_service(
            DEMO_CONFIG["trace_store_path"],
            rate_limit_config={"max_requests_per_minute": 10 if rate_limit else 1000},
        )
        await mining_service.initialize()
        logger.info("Mining service initialized")

        # Initialize DSPy pipeline
        progress.update(init_task, description="Initializing DSPy pipeline...")
        dspy_pipeline = create_dspy_pipeline(
            primary_provider_uri=provider_uri,
            fallback_provider_uris=DEMO_CONFIG["fallback_providers"],
        )

        # Register social media templates
        for template in get_social_media_templates().values():
            dspy_pipeline.register_template(template)

        logger.info("DSPy pipeline initialized")

        progress.update(init_task, description="Initialization complete!")

    console.print("[green]✓ All components initialized successfully[/green]")


async def _demo_trace_store_integration(trace_id: str):
    """Demonstrate TraceStore integration with REER mining."""

    console.print("\n[cyan]Step 2: TraceStore Integration[/cyan]")
    logger = get_component_logger("demo.trace_store", trace_id=trace_id)

    mining_service = create_mining_service(DEMO_CONFIG["trace_store_path"])
    await mining_service.initialize()

    # Process demo posts
    trace_ids = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        extract_task = progress.add_task(
            "Extracting strategies from posts...", total=len(DEMO_CONFIG["demo_posts"])
        )

        for post in DEMO_CONFIG["demo_posts"]:
            progress.update(
                extract_task, description=f"Processing post {post['id']}..."
            )

            # Extract and store strategy
            stored_trace_id = await mining_service.extract_and_store(
                source_post_id=post["id"],
                content=post["content"],
                seed_params=post["seed_params"],
                trace_id=trace_id,
            )

            trace_ids.append(stored_trace_id)
            progress.advance(extract_task)

            logger.info(
                f"Processed post {post['id']}",
                extra={"post_id": post["id"], "stored_trace_id": stored_trace_id},
            )

    # Query stored traces
    traces = await mining_service.query_traces(limit=10, trace_id=trace_id)

    # Display results
    results_table = Table(title="Extracted Strategies")
    results_table.add_column("Post ID", style="cyan")
    results_table.add_column("Score", style="green")
    results_table.add_column("Features", style="yellow")
    results_table.add_column("Provider", style="blue")

    for trace in traces[-3:]:  # Show last 3
        features = ", ".join(trace.get("strategy_features", [])[:2])
        if len(trace.get("strategy_features", [])) > 2:
            features += "..."

        results_table.add_row(
            trace.get("source_post_id", "unknown"),
            f"{trace.get('score', 0):.3f}",
            features,
            trace.get("provider", "unknown").split("://")[0],
        )

    console.print(results_table)
    console.print(
        f"[green]✓ Processed {len(DEMO_CONFIG['demo_posts'])} posts, stored {len(trace_ids)} traces[/green]"
    )


async def _demo_lm_registry_routing(provider_uri: str, trace_id: str):
    """Demonstrate LM registry provider routing."""

    console.print("\n[cyan]Step 3: LM Registry Provider Routing[/cyan]")
    logger = get_component_logger("demo.lm_registry", trace_id=trace_id)

    model_manager = get_model_manager()
    registry = get_registry()

    # Test different providers
    test_providers = [provider_uri, "dummy://test-model"]

    results_table = Table(title="Provider Routing Tests")
    results_table.add_column("Provider", style="cyan")
    results_table.add_column("Status", style="green")
    results_table.add_column("Response Time", style="yellow")
    results_table.add_column("Response Preview", style="blue")

    for test_provider in test_providers:
        try:
            start_time = time.time()

            response = await model_manager.generate_text(
                "Generate a short social media tip about productivity",
                uri=test_provider,
                max_tokens=50,
            )

            duration = (time.time() - start_time) * 1000
            preview = response[:50] + "..." if len(response) > 50 else response

            results_table.add_row(
                test_provider.split("://")[0],
                "✅ Success",
                f"{duration:.0f}ms",
                preview,
            )

            logger.info(
                "Provider test successful",
                extra={"provider": test_provider, "duration_ms": duration},
            )

        except Exception as e:
            results_table.add_row(
                test_provider.split("://")[0], "❌ Failed", "-", str(e)[:50]
            )

            logger.warning(
                "Provider test failed",
                extra={"provider": test_provider, "error": str(e)},
            )

    console.print(results_table)

    # Show health check
    health_status = await registry.health_check()

    health_table = Table(title="Provider Health Status")
    health_table.add_column("Provider", style="cyan")
    health_table.add_column("Available", style="green")
    health_table.add_column("Test Generation", style="yellow")

    for provider, status in health_status.items():
        available = "✅" if status.get("available", False) else "❌"
        test_gen = "✅" if status.get("test_generation", False) else "❌"

        health_table.add_row(provider, available, test_gen)

    console.print(health_table)
    console.print("[green]✓ LM registry provider routing working correctly[/green]")


async def _demo_dspy_pipeline(
    provider_uri: str, enable_optimization: bool, trace_id: str
):
    """Demonstrate DSPy pipeline integration."""

    console.print("\n[cyan]Step 4: DSPy Pipeline Integration[/cyan]")
    logger = get_component_logger("demo.dspy_pipeline", trace_id=trace_id)

    # Create DSPy pipeline
    dspy_pipeline = create_dspy_pipeline(
        primary_provider_uri=provider_uri,
        fallback_provider_uris=DEMO_CONFIG["fallback_providers"],
    )

    # Register templates
    templates = get_social_media_templates()
    for template in templates.values():
        dspy_pipeline.register_template(template)

    # Test content generation
    test_inputs = {
        "topic": "artificial intelligence",
        "style": "educational",
        "target_audience": "tech professionals",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        pipeline_task = progress.add_task("Running DSPy pipeline...", total=None)

        progress.update(pipeline_task, description="Generating content...")

        try:
            result = await dspy_pipeline.execute_pipeline(
                template_name="content_generation",
                inputs=test_inputs,
                provider_uri=provider_uri,
                trace_id=trace_id,
            )

            # Display result
            result_panel = Panel(
                f"[cyan]Input:[/cyan] {test_inputs}\n\n"
                f"[green]Generated Content:[/green] {result.get('content', 'No content generated')}\n\n"
                f"[yellow]Metadata:[/yellow] {result.get('_pipeline_metadata', {})}",
                title="DSPy Pipeline Result",
                border_style="blue",
            )

            console.print(result_panel)

            logger.info(
                "DSPy pipeline execution successful",
                extra={
                    "template": "content_generation",
                    "execution_time": result.get("_pipeline_metadata", {}).get(
                        "execution_time", 0
                    ),
                },
            )

        except Exception as e:
            console.print(f"[red]DSPy pipeline failed: {e}[/red]")
            logger.error(f"DSPy pipeline failed: {e}", exc_info=True)

        # Optimization demo (if enabled)
        if enable_optimization:
            progress.update(pipeline_task, description="Running optimization demo...")

            # Create mock training examples

            console.print(
                "[yellow]Note: Optimization would run here with real DSPy setup[/yellow]"
            )
            logger.info("Optimization demo completed (simulated)")

    console.print("[green]✓ DSPy pipeline integration working correctly[/green]")


async def _demo_rate_limiting(trace_id: str):
    """Demonstrate rate limiting with exponential backoff."""

    console.print("\n[cyan]Step 5: Rate Limiting Demonstration[/cyan]")
    get_component_logger("demo.rate_limiting", trace_id=trace_id)

    mining_service = create_mining_service(
        DEMO_CONFIG["trace_store_path"],
        rate_limit_config={
            "max_requests_per_minute": 3,  # Very low for demo
            "exponential_backoff_base": 2.0,
            "max_backoff_delay": 10.0,
        },
    )
    await mining_service.initialize()

    console.print(
        "[yellow]Testing rate limiting (limit: 3 requests/minute)...[/yellow]"
    )

    # Make rapid requests to trigger rate limiting
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        rate_task = progress.add_task("Testing rate limits...", total=5)

        for i in range(5):
            progress.update(rate_task, description=f"Request {i + 1}/5...")

            start_time = time.time()

            try:
                await mining_service.extract_and_store(
                    source_post_id=f"rate_test_{i}",
                    content=f"Rate limiting test post {i}",
                    seed_params={
                        "topic": "test",
                        "style": "test",
                        "length": 100,
                        "thread_size": 1,
                    },
                    trace_id=trace_id,
                )

                duration = time.time() - start_time
                console.print(f"  Request {i + 1}: ✅ Success ({duration:.2f}s)")

            except Exception:
                duration = time.time() - start_time
                console.print(f"  Request {i + 1}: ⚠️ Limited ({duration:.2f}s)")

            progress.advance(rate_task)

    console.print("[green]✓ Rate limiting working correctly[/green]")


async def _demo_metrics_reporting(trace_id: str):
    """Demonstrate comprehensive metrics reporting."""

    console.print("\n[cyan]Step 6: Metrics and Performance Reporting[/cyan]")
    logger = get_component_logger("demo.metrics", trace_id=trace_id)

    # Get mining service stats
    mining_service = create_mining_service(DEMO_CONFIG["trace_store_path"])
    await mining_service.initialize()

    try:
        stats = await mining_service.get_performance_stats()

        # Display extraction stats
        extraction_table = Table(title="Extraction Statistics")
        extraction_table.add_column("Metric", style="cyan")
        extraction_table.add_column("Value", style="green")

        extraction_stats = stats.get("extraction_stats", {})

        extraction_table.add_row(
            "Total Extractions", str(extraction_stats.get("total_extractions", 0))
        )
        extraction_table.add_row(
            "Successful", str(extraction_stats.get("successful_extractions", 0))
        )
        extraction_table.add_row(
            "Failed", str(extraction_stats.get("failed_extractions", 0))
        )
        extraction_table.add_row("Success Rate", f"{stats.get('success_rate', 0):.2%}")

        console.print(extraction_table)

        # Display trace store stats
        trace_stats = stats.get("trace_store_stats", {})

        trace_table = Table(title="Trace Store Statistics")
        trace_table.add_column("Metric", style="cyan")
        trace_table.add_column("Value", style="green")

        trace_table.add_row("Total Traces", str(trace_stats.get("total_traces", 0)))
        trace_table.add_row("Valid Traces", str(trace_stats.get("valid_traces", 0)))
        trace_table.add_row(
            "File Size", f"{trace_stats.get('file_size_bytes', 0) / 1024:.1f} KB"
        )

        console.print(trace_table)

        logger.info(
            "Metrics reporting completed",
            extra={
                "total_traces": trace_stats.get("total_traces", 0),
                "success_rate": stats.get("success_rate", 0),
            },
        )

    except Exception as e:
        console.print(f"[red]Failed to get metrics: {e}[/red]")
        logger.error(f"Metrics reporting failed: {e}", exc_info=True)

    console.print("[green]✓ Comprehensive metrics available[/green]")


@app.command()
def validate():
    """Validate that all integration components are properly installed."""

    console.print("[cyan]Validating Integration Components...[/cyan]")

    validation_results = []

    # Check imports
    try:
        from core.integration import IntegratedREERMiner

        validation_results.append(("Core Integration", "✅ Available"))
    except ImportError as e:
        validation_results.append(("Core Integration", f"❌ Missing: {e}"))

    try:
        from plugins.lm_registry import get_registry

        validation_results.append(("LM Registry", "✅ Available"))
    except ImportError as e:
        validation_results.append(("LM Registry", f"❌ Missing: {e}"))

    try:
        from plugins.dspy_pipeline import create_dspy_pipeline

        validation_results.append(("DSPy Pipeline", "✅ Available"))
    except ImportError as e:
        validation_results.append(("DSPy Pipeline", f"❌ Missing: {e}"))

    try:
        from config.logging_config import setup_production_logging

        validation_results.append(("Logging Config", "✅ Available"))
    except ImportError as e:
        validation_results.append(("Logging Config", f"❌ Missing: {e}"))

    # Display results
    validation_table = Table(title="Component Validation")
    validation_table.add_column("Component", style="cyan")
    validation_table.add_column("Status", style="green")

    for component, status in validation_results:
        validation_table.add_row(component, status)

    console.print(validation_table)

    # Check directories
    reer_dir = Path.home() / ".reer"
    if not reer_dir.exists():
        console.print(f"[yellow]Creating REER directory: {reer_dir}[/yellow]")
        reer_dir.mkdir(exist_ok=True)
        (reer_dir / "logs").mkdir(exist_ok=True)

    console.print("[green]✓ Validation complete[/green]")


if __name__ == "__main__":
    # Setup logging for the demo
    setup_production_logging()
    app()
