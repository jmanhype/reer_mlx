#!/usr/bin/env python3
"""
REER × DSPy × MLX - Social Pipeline CLI

Command-line interface for executing complete social media content generation pipelines.
Orchestrates data collection, REER mining, GEPA tuning, and content generation.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.tree import Tree
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


app = typer.Typer(
    name="social-run",
    help="Social media pipeline execution CLI",
    rich_markup_mode="rich",
)
console = Console()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


@app.command()
def pipeline(
    config_file: Path = typer.Argument(..., help="Pipeline configuration file"),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Output directory for generated content",
    ),
    stages: list[str] | None = typer.Option(
        None,
        "--stage",
        "-s",
        help="Specific stages to run (collect, mine, tune, generate)",
    ),
    platforms: list[str] | None = typer.Option(
        None, "--platform", "-p", help="Target platforms (can be used multiple times)"
    ),
    content_count: int = typer.Option(
        10, "--content-count", "-c", help="Number of content pieces to generate"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be executed without running"
    ),
    skip_cache: bool = typer.Option(
        False, "--skip-cache", help="Skip cached results and rerun all stages"
    ),
    parallel_stages: bool = typer.Option(
        False, "--parallel", help="Run compatible stages in parallel"
    ),
    save_intermediates: bool = typer.Option(
        True,
        "--save-intermediates/--no-intermediates",
        help="Save intermediate results from each stage",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Execute complete social media content generation pipeline.

    Examples:

        # Run full pipeline
        social-run pipeline config.json

        # Run specific stages
        social-run pipeline config.json --stage collect --stage mine

        # Generate content for specific platforms
        social-run pipeline config.json --platform x --platform instagram --content-count 20

        # Dry run to see execution plan
        social-run pipeline config.json --dry-run
    """

    # Validate config file
    if not config_file.exists():
        console.print(f"[red]Configuration file not found:[/red] {config_file}")
        raise typer.Exit(1)

    # Load configuration
    try:
        with open(config_file) as f:
            config = json.load(f)
    except Exception as e:
        console.print(f"[red]Failed to load configuration:[/red] {e}")
        raise typer.Exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine stages to run
    all_stages = ["collect", "mine", "tune", "generate"]
    run_stages = stages if stages else all_stages

    # Show pipeline configuration
    _display_pipeline_config(config, run_stages, platforms, content_count, output_dir)

    if dry_run:
        console.print(
            "[yellow]DRY RUN MODE - Pipeline execution plan shown above[/yellow]"
        )
        return

    try:
        # Execute pipeline
        results = asyncio.run(
            _execute_pipeline(
                config,
                run_stages,
                output_dir,
                platforms,
                content_count,
                skip_cache,
                parallel_stages,
                save_intermediates,
                verbose,
            )
        )

        # Display results
        _display_pipeline_results(results)

        console.print("[green]✓ Pipeline execution completed successfully![/green]")
        console.print(f"[cyan]Results saved to:[/cyan] {output_dir}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        console.print(f"[red]Error during pipeline execution:[/red] {e}")
        raise typer.Exit(1)


async def _execute_pipeline(
    config: dict[str, Any],
    stages: list[str],
    output_dir: Path,
    platforms: list[str] | None,
    content_count: int,
    skip_cache: bool,
    parallel_stages: bool,
    save_intermediates: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Execute the social media pipeline."""

    results = {
        "stages_completed": [],
        "stage_results": {},
        "total_time": 0.0,
        "content_generated": 0,
        "errors": [],
    }

    start_time = time.time()

    # Create pipeline progress table
    pipeline_table = Table(title="Pipeline Execution Progress")
    pipeline_table.add_column("Stage", style="cyan")
    pipeline_table.add_column("Status", style="green")
    pipeline_table.add_column("Duration", style="yellow")
    pipeline_table.add_column("Output", style="magenta")

    with Live(pipeline_table, refresh_per_second=1, console=console):

        # Stage 1: Data Collection
        if "collect" in stages:
            stage_start = time.time()
            pipeline_table.add_row("Data Collection", "🔄 Running", "", "")

            try:
                collect_result = await _run_collection_stage(
                    config, output_dir, platforms, skip_cache, verbose
                )

                stage_duration = time.time() - stage_start
                results["stage_results"]["collect"] = collect_result
                results["stages_completed"].append("collect")

                # Update table
                pipeline_table.rows[-1] = (
                    "Data Collection",
                    "✅ Complete",
                    f"{stage_duration:.1f}s",
                    f"{collect_result.get('records_collected', 0)} records",
                )

            except Exception as e:
                stage_duration = time.time() - stage_start
                results["errors"].append(f"Collection stage failed: {e}")
                pipeline_table.rows[-1] = (
                    "Data Collection",
                    "❌ Failed",
                    f"{stage_duration:.1f}s",
                    str(e),
                )

        # Stage 2: REER Mining
        if "mine" in stages:
            stage_start = time.time()
            pipeline_table.add_row("REER Mining", "🔄 Running", "", "")

            try:
                mine_result = await _run_mining_stage(
                    config, output_dir, skip_cache, verbose
                )

                stage_duration = time.time() - stage_start
                results["stage_results"]["mine"] = mine_result
                results["stages_completed"].append("mine")

                pipeline_table.rows[-1] = (
                    "REER Mining",
                    "✅ Complete",
                    f"{stage_duration:.1f}s",
                    f"{mine_result.get('patterns_extracted', 0)} patterns",
                )

            except Exception as e:
                stage_duration = time.time() - stage_start
                results["errors"].append(f"Mining stage failed: {e}")
                pipeline_table.rows[-1] = (
                    "REER Mining",
                    "❌ Failed",
                    f"{stage_duration:.1f}s",
                    str(e),
                )

        # Stage 3: GEPA Tuning
        if "tune" in stages:
            stage_start = time.time()
            pipeline_table.add_row("GEPA Tuning", "🔄 Running", "", "")

            try:
                tune_result = await _run_tuning_stage(
                    config, output_dir, skip_cache, verbose
                )

                stage_duration = time.time() - stage_start
                results["stage_results"]["tune"] = tune_result
                results["stages_completed"].append("tune")

                pipeline_table.rows[-1] = (
                    "GEPA Tuning",
                    "✅ Complete",
                    f"{stage_duration:.1f}s",
                    f"Fitness: {tune_result.get('best_fitness', 0):.3f}",
                )

            except Exception as e:
                stage_duration = time.time() - stage_start
                results["errors"].append(f"Tuning stage failed: {e}")
                pipeline_table.rows[-1] = (
                    "GEPA Tuning",
                    "❌ Failed",
                    f"{stage_duration:.1f}s",
                    str(e),
                )

        # Stage 4: Content Generation
        if "generate" in stages:
            stage_start = time.time()
            pipeline_table.add_row("Content Generation", "🔄 Running", "", "")

            try:
                generate_result = await _run_generation_stage(
                    config, output_dir, platforms, content_count, verbose
                )

                stage_duration = time.time() - stage_start
                results["stage_results"]["generate"] = generate_result
                results["stages_completed"].append("generate")
                results["content_generated"] = generate_result.get("content_count", 0)

                pipeline_table.rows[-1] = (
                    "Content Generation",
                    "✅ Complete",
                    f"{stage_duration:.1f}s",
                    f"{generate_result.get('content_count', 0)} posts",
                )

            except Exception as e:
                stage_duration = time.time() - stage_start
                results["errors"].append(f"Generation stage failed: {e}")
                pipeline_table.rows[-1] = (
                    "Content Generation",
                    "❌ Failed",
                    f"{stage_duration:.1f}s",
                    str(e),
                )

    results["total_time"] = time.time() - start_time

    # Save pipeline results
    if save_intermediates:
        await _save_pipeline_results(results, output_dir)

    return results


async def _run_collection_stage(
    config: dict[str, Any],
    output_dir: Path,
    platforms: list[str] | None,
    skip_cache: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run data collection stage."""

    # Mock data collection
    await asyncio.sleep(2)  # Simulate collection time

    # Use configured platforms or provided ones
    target_platforms = platforms or config.get("platforms", ["x"])

    result = {
        "records_collected": 150 * len(target_platforms),
        "platforms": target_platforms,
        "data_files": [f"{platform}_data.json" for platform in target_platforms],
    }

    # Save mock data files
    for platform in target_platforms:
        data_file = output_dir / f"{platform}_data.json"
        mock_data = [
            {
                "id": f"{platform}_post_{i}",
                "platform": platform,
                "content": f"Sample {platform} post {i}",
                "engagement": {"likes": i * 10, "shares": i * 2},
            }
            for i in range(150)
        ]

        with open(data_file, "w") as f:
            json.dump(mock_data, f, indent=2)

    return result


async def _run_mining_stage(
    config: dict[str, Any], output_dir: Path, skip_cache: bool, verbose: bool
) -> dict[str, Any]:
    """Run REER mining stage."""

    # Mock REER mining
    await asyncio.sleep(3)  # Simulate mining time

    result = {
        "patterns_extracted": 45,
        "strategies_synthesized": 12,
        "candidates_scored": 150,
        "trace_records": 300,
    }

    # Save mock mining results
    patterns_file = output_dir / "extracted_patterns.json"
    mock_patterns = [
        {
            "pattern_id": f"pattern_{i}",
            "type": "engagement",
            "confidence": 0.8 + (i * 0.01),
            "platforms": ["x", "instagram"],
        }
        for i in range(45)
    ]

    with open(patterns_file, "w") as f:
        json.dump(mock_patterns, f, indent=2)

    return result


async def _run_tuning_stage(
    config: dict[str, Any], output_dir: Path, skip_cache: bool, verbose: bool
) -> dict[str, Any]:
    """Run GEPA tuning stage."""

    # Mock GEPA tuning
    await asyncio.sleep(4)  # Simulate tuning time

    result = {
        "best_fitness": 0.847,
        "generations": 75,
        "converged": True,
        "model_file": "best_model.pkl",
    }

    # Save mock model
    model_file = output_dir / "best_model.pkl"
    model_data = {
        "fitness": result["best_fitness"],
        "generations": result["generations"],
        "parameters": {"mutation_rate": 0.1, "crossover_rate": 0.7},
    }

    with open(model_file, "w") as f:
        json.dump(model_data, f, indent=2)

    return result


async def _run_generation_stage(
    config: dict[str, Any],
    output_dir: Path,
    platforms: list[str] | None,
    content_count: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run content generation stage."""

    # Mock content generation
    await asyncio.sleep(2)  # Simulate generation time

    target_platforms = platforms or config.get("platforms", ["x"])

    result = {
        "content_count": content_count,
        "platforms": target_platforms,
        "content_files": [],
    }

    # Generate mock content for each platform
    for platform in target_platforms:
        content_file = output_dir / f"{platform}_content.json"

        mock_content = [
            {
                "id": f"{platform}_content_{i}",
                "platform": platform,
                "content": f"Generated {platform} post {i}: Engaging content with trending hashtags #AI #ML",
                "hashtags": ["#AI", "#ML", "#tech"],
                "scheduled_time": f"2024-01-0{(i % 9) + 1}T10:00:00Z",
                "estimated_engagement": {
                    "likes": 50 + (i * 10),
                    "shares": 10 + (i * 2),
                    "comments": 5 + i,
                },
            }
            for i in range(content_count // len(target_platforms))
        ]

        with open(content_file, "w") as f:
            json.dump(mock_content, f, indent=2)

        result["content_files"].append(str(content_file))

    return result


async def _save_pipeline_results(results: dict[str, Any], output_dir: Path):
    """Save complete pipeline results."""

    results_file = output_dir / "pipeline_results.json"

    # Create summary with timestamps
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stages_completed": results["stages_completed"],
        "total_time": results["total_time"],
        "content_generated": results["content_generated"],
        "errors": results["errors"],
        "stage_summaries": {},
    }

    # Add stage summaries
    for stage, stage_result in results["stage_results"].items():
        if stage == "collect":
            summary["stage_summaries"][stage] = {
                "records_collected": stage_result.get("records_collected", 0),
                "platforms": stage_result.get("platforms", []),
            }
        elif stage == "mine":
            summary["stage_summaries"][stage] = {
                "patterns_extracted": stage_result.get("patterns_extracted", 0),
                "strategies_synthesized": stage_result.get("strategies_synthesized", 0),
            }
        elif stage == "tune":
            summary["stage_summaries"][stage] = {
                "best_fitness": stage_result.get("best_fitness", 0),
                "converged": stage_result.get("converged", False),
            }
        elif stage == "generate":
            summary["stage_summaries"][stage] = {
                "content_count": stage_result.get("content_count", 0),
                "platforms": stage_result.get("platforms", []),
            }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)


def _display_pipeline_config(
    config: dict[str, Any],
    stages: list[str],
    platforms: list[str] | None,
    content_count: int,
    output_dir: Path,
):
    """Display pipeline configuration."""

    config_table = Table(title="Pipeline Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Stages", " → ".join(stages))
    config_table.add_row("Output Directory", str(output_dir))
    config_table.add_row("Content Count", str(content_count))

    if platforms:
        config_table.add_row("Target Platforms", ", ".join(platforms))
    else:
        config_table.add_row(
            "Target Platforms", ", ".join(config.get("platforms", ["x"]))
        )

    # Add config-specific parameters
    if "collection" in config:
        config_table.add_row(
            "Collection Limit", str(config["collection"].get("limit", 100))
        )

    if "mining" in config:
        config_table.add_row(
            "Mining Algorithm", config["mining"].get("algorithm", "reer")
        )

    if "tuning" in config:
        config_table.add_row(
            "Population Size", str(config["tuning"].get("population_size", 50))
        )
        config_table.add_row(
            "Generations", str(config["tuning"].get("generations", 100))
        )

    console.print(config_table)
    console.print()


def _display_pipeline_results(results: dict[str, Any]):
    """Display pipeline execution results."""

    # Summary table
    summary_table = Table(title="Pipeline Execution Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Stages Completed", f"{len(results['stages_completed'])}")
    summary_table.add_row("Total Time", f"{results['total_time']:.1f}s")
    summary_table.add_row("Content Generated", str(results["content_generated"]))
    summary_table.add_row("Errors", str(len(results["errors"])))

    console.print(summary_table)

    # Stage details
    if results["stage_results"]:
        console.print("\n[cyan]Stage Details:[/cyan]")

        stage_tree = Tree("Pipeline Stages")

        for stage, stage_result in results["stage_results"].items():
            stage_branch = stage_tree.add(f"📋 {stage.title()}")

            if stage == "collect":
                stage_branch.add(f"Records: {stage_result.get('records_collected', 0)}")
                stage_branch.add(
                    f"Platforms: {', '.join(stage_result.get('platforms', []))}"
                )

            elif stage == "mine":
                stage_branch.add(
                    f"Patterns: {stage_result.get('patterns_extracted', 0)}"
                )
                stage_branch.add(
                    f"Strategies: {stage_result.get('strategies_synthesized', 0)}"
                )

            elif stage == "tune":
                stage_branch.add(
                    f"Best Fitness: {stage_result.get('best_fitness', 0):.3f}"
                )
                stage_branch.add(f"Converged: {stage_result.get('converged', False)}")

            elif stage == "generate":
                stage_branch.add(
                    f"Content: {stage_result.get('content_count', 0)} posts"
                )
                stage_branch.add(
                    f"Platforms: {', '.join(stage_result.get('platforms', []))}"
                )

        console.print(stage_tree)

    # Show errors if any
    if results["errors"]:
        console.print("\n[red]Errors encountered:[/red]")
        for error in results["errors"]:
            console.print(f"  • {error}")


@app.command()
def status(
    output_dir: Path = typer.Option(
        Path("output"), "--output-dir", "-o", help="Output directory to check"
    )
):
    """Check status of pipeline execution."""

    if not output_dir.exists():
        console.print(f"[red]Output directory not found:[/red] {output_dir}")
        return

    # Check for pipeline results
    results_file = output_dir / "pipeline_results.json"

    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        console.print(f"[green]Pipeline execution found in:[/green] {output_dir}")
        console.print(f"[cyan]Timestamp:[/cyan] {results.get('timestamp', 'Unknown')}")
        console.print(
            f"[cyan]Stages completed:[/cyan] {', '.join(results.get('stages_completed', []))}"
        )
        console.print(
            f"[cyan]Content generated:[/cyan] {results.get('content_generated', 0)} posts"
        )

        if results.get("errors"):
            console.print(f"[red]Errors:[/red] {len(results['errors'])}")

    else:
        console.print(f"[yellow]No pipeline results found in:[/yellow] {output_dir}")

    # List output files
    output_files = list(output_dir.glob("*.json"))

    if output_files:
        console.print(f"\n[cyan]Output files ({len(output_files)}):[/cyan]")

        files_table = Table()
        files_table.add_column("File", style="cyan")
        files_table.add_column("Size", style="green")
        files_table.add_column("Modified", style="yellow")

        for file_path in sorted(output_files):
            stats = file_path.stat()
            size_kb = stats.st_size / 1024
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M")

            files_table.add_row(file_path.name, f"{size_kb:.1f} KB", mod_time)

        console.print(files_table)


@app.command()
def clean(
    output_dir: Path = typer.Option(
        Path("output"), "--output-dir", "-o", help="Output directory to clean"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clean pipeline output directory."""

    if not output_dir.exists():
        console.print(f"[yellow]Output directory not found:[/yellow] {output_dir}")
        return

    # List files to be removed
    files_to_remove = list(output_dir.glob("*"))

    if not files_to_remove:
        console.print(f"[yellow]No files to clean in:[/yellow] {output_dir}")
        return

    console.print(f"[cyan]Files to be removed ({len(files_to_remove)}):[/cyan]")
    for file_path in files_to_remove:
        console.print(f"  • {file_path.name}")

    if not confirm:
        proceed = typer.confirm("\nProceed with cleaning?")
        if not proceed:
            console.print("[yellow]Cleaning cancelled[/yellow]")
            return

    # Remove files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
            elif file_path.is_dir():
                import shutil

                shutil.rmtree(file_path)
                removed_count += 1
        except Exception as e:
            console.print(f"[red]Failed to remove {file_path.name}:[/red] {e}")

    console.print(f"[green]Cleaned {removed_count} items from {output_dir}[/green]")


if __name__ == "__main__":
    app()
