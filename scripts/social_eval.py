#!/usr/bin/env python3
"""
REER × DSPy × MLX - Social Evaluation CLI

Command-line interface for evaluating social media content generation systems.
Provides comprehensive metrics, benchmarking, and performance analysis.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import REERCandidateScorer
from plugins import ContentMetrics, HeuristicScorer
from social import PostMetrics, SocialKPICalculator

app = typer.Typer(
    name="social-eval",
    help="Social media content evaluation CLI",
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
def evaluate(
    content_file: Path = typer.Argument(..., help="Content file to evaluate"),
    reference_file: Path | None = typer.Option(
        None, "--reference", "-r", help="Reference content file for comparison"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for evaluation results"
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "--metric",
        "-m",
        help="Specific metrics to evaluate (can be used multiple times)",
    ),
    platforms: list[str] | None = typer.Option(
        None,
        "--platform",
        "-p",
        help="Filter by platforms (can be used multiple times)",
    ),
    evaluators: list[str] | None = typer.Option(
        None,
        "--evaluator",
        "-e",
        help="Evaluators to use (kpi, heuristic, reer) (can be used multiple times)",
    ),
    include_human_eval: bool = typer.Option(
        False, "--human-eval", help="Include human evaluation metrics"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", help="Generate detailed evaluation report"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Evaluate social media content using multiple metrics and evaluators.

    Examples:

        # Basic content evaluation
        social-eval evaluate generated_content.json

        # Compare against reference content
        social-eval evaluate new_content.json --reference baseline_content.json

        # Evaluate specific metrics
        social-eval evaluate content.json --metric engagement --metric readability

        # Platform-specific evaluation
        social-eval evaluate content.json --platform x --platform instagram

        # Use specific evaluators
        social-eval evaluate content.json --evaluator kpi --evaluator heuristic
    """

    # Validate input files
    if not content_file.exists():
        console.print(f"[red]Content file not found:[/red] {content_file}")
        raise typer.Exit(1)

    if reference_file and not reference_file.exists():
        console.print(f"[red]Reference file not found:[/red] {reference_file}")
        raise typer.Exit(1)

    # Show evaluation parameters
    params_table = Table(title="Evaluation Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Content File", str(content_file))
    if reference_file:
        params_table.add_row("Reference File", str(reference_file))
    if metrics:
        params_table.add_row("Metrics", ", ".join(metrics))
    if platforms:
        params_table.add_row("Platforms", ", ".join(platforms))
    if evaluators:
        params_table.add_row("Evaluators", ", ".join(evaluators))

    params_table.add_row("Human Evaluation", str(include_human_eval))
    params_table.add_row("Generate Report", str(generate_report))

    console.print(params_table)
    console.print()

    try:
        # Run evaluation
        results = asyncio.run(
            _run_evaluation(
                content_file,
                reference_file,
                metrics,
                platforms,
                evaluators,
                include_human_eval,
                verbose,
            )
        )

        # Display results
        _display_evaluation_results(results)

        # Generate report
        if generate_report:
            report_file = (
                output_file
                or content_file.parent
                / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            asyncio.run(_generate_evaluation_report(results, report_file))
            console.print(f"[green]Evaluation report saved to:[/green] {report_file}")

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]Evaluation results saved to:[/green] {output_file}")

        console.print("[green]✓ Content evaluation completed successfully![/green]")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]Error during evaluation:[/red] {e}")
        raise typer.Exit(1)


async def _run_evaluation(
    content_file: Path,
    reference_file: Path | None,
    metrics: list[str] | None,
    platforms: list[str] | None,
    evaluators: list[str] | None,
    include_human_eval: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run comprehensive content evaluation."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "content_file": str(content_file),
        "reference_file": str(reference_file) if reference_file else None,
        "evaluation_results": {},
        "summary_metrics": {},
        "platform_breakdown": {},
        "comparative_analysis": None,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Load content data
        load_task = progress.add_task("Loading content data...", total=None)

        with open(content_file) as f:
            content_data = json.load(f)

        reference_data = None
        if reference_file:
            with open(reference_file) as f:
                reference_data = json.load(f)

        if not isinstance(content_data, list):
            content_data = [content_data]

        if reference_data and not isinstance(reference_data, list):
            reference_data = [reference_data]

        progress.update(
            load_task, description=f"Loaded {len(content_data)} content items"
        )

        # Filter data by platforms if specified
        if platforms:
            filter_task = progress.add_task(
                "Filtering by platforms...", total=len(content_data)
            )
            filtered_content = []

            for item in content_data:
                if item.get("platform") in platforms:
                    filtered_content.append(item)
                progress.update(filter_task, advance=1)

            content_data = filtered_content
            progress.update(
                filter_task, description=f"Filtered to {len(content_data)} items"
            )

        # Initialize evaluators
        init_task = progress.add_task("Initializing evaluators...", total=None)

        # Determine which evaluators to use
        eval_list = evaluators if evaluators else ["kpi", "heuristic", "reer"]

        evaluator_instances = {}
        if "kpi" in eval_list:
            evaluator_instances["kpi"] = SocialKPICalculator()
        if "heuristic" in eval_list:
            evaluator_instances["heuristic"] = HeuristicScorer()
        if "reer" in eval_list:
            evaluator_instances["reer"] = REERCandidateScorer()

        progress.update(
            init_task, description=f"Initialized {len(evaluator_instances)} evaluators"
        )

        # Run evaluations
        eval_task = progress.add_task("Running evaluations...", total=len(content_data))

        all_results = []

        for i, content_item in enumerate(content_data):
            item_results = {
                "content_id": content_item.get("id", f"item_{i}"),
                "platform": content_item.get("platform", "unknown"),
                "evaluator_scores": {},
            }

            # KPI Evaluation
            if "kpi" in evaluator_instances:
                kpi_score = await _evaluate_with_kpi(
                    evaluator_instances["kpi"], content_item
                )
                item_results["evaluator_scores"]["kpi"] = kpi_score

            # Heuristic Evaluation
            if "heuristic" in evaluator_instances:
                heuristic_score = await _evaluate_with_heuristic(
                    evaluator_instances["heuristic"], content_item
                )
                item_results["evaluator_scores"]["heuristic"] = heuristic_score

            # REER Evaluation
            if "reer" in evaluator_instances:
                reer_score = await _evaluate_with_reer(
                    evaluator_instances["reer"], content_item
                )
                item_results["evaluator_scores"]["reer"] = reer_score

            # Human evaluation (if requested)
            if include_human_eval:
                human_score = await _evaluate_with_human(content_item)
                item_results["evaluator_scores"]["human"] = human_score

            all_results.append(item_results)
            progress.update(eval_task, advance=1)

        results["evaluation_results"] = all_results

        # Calculate summary metrics
        summary_task = progress.add_task("Calculating summary metrics...", total=None)

        results["summary_metrics"] = _calculate_summary_metrics(all_results, eval_list)

        # Platform breakdown
        results["platform_breakdown"] = _calculate_platform_breakdown(all_results)

        # Comparative analysis with reference
        if reference_data:
            comp_task = progress.add_task("Running comparative analysis...", total=None)
            results["comparative_analysis"] = await _run_comparative_analysis(
                all_results, reference_data, evaluator_instances
            )
            progress.update(comp_task, description="Comparative analysis completed")

        progress.update(summary_task, description="Summary metrics calculated")

    return results


async def _evaluate_with_kpi(
    kpi_calculator: SocialKPICalculator, content_item: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate content using KPI calculator."""

    # Mock KPI evaluation
    await asyncio.sleep(0.1)  # Simulate calculation time

    # Convert content to PostMetrics format
    post_metrics = PostMetrics(
        post_id=content_item.get("id", "unknown"),
        platform=content_item.get("platform", "unknown"),
        content_length=len(content_item.get("content", "")),
        hashtag_count=len(content_item.get("hashtags", [])),
        mention_count=content_item.get("content", "").count("@"),
        engagement_metrics=content_item.get("engagement", {}),
        reach_metrics=content_item.get("reach", {}),
        conversion_metrics=content_item.get("conversions", {}),
    )

    # Calculate KPIs
    kpi_results = await kpi_calculator.calculate_kpis([post_metrics])

    return {
        "engagement_rate": kpi_results.get("engagement_rate", 0.0),
        "reach_score": kpi_results.get("reach_score", 0.0),
        "conversion_rate": kpi_results.get("conversion_rate", 0.0),
        "content_quality": kpi_results.get("content_quality", 0.0),
        "overall_score": kpi_results.get("overall_score", 0.0),
    }


async def _evaluate_with_heuristic(
    heuristic_scorer: HeuristicScorer, content_item: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate content using heuristic scorer."""

    # Mock heuristic evaluation
    await asyncio.sleep(0.05)

    content_text = content_item.get("content", "")

    # Create ContentMetrics
    content_metrics = ContentMetrics(
        readability_score=min(1.0, len(content_text.split()) / 20),  # Mock readability
        sentiment_score=0.7,  # Mock positive sentiment
        engagement_potential=min(1.0, len(content_item.get("hashtags", [])) / 5),
        platform_optimization=0.8,  # Mock platform optimization
        content_structure=0.75,  # Mock structure score
        call_to_action_strength=(
            0.6 if "!" in content_text or "?" in content_text else 0.3
        ),
    )

    # Score using heuristic scorer
    heuristic_result = await heuristic_scorer.score_content(content_metrics)

    return {
        "readability": content_metrics.readability_score,
        "sentiment": content_metrics.sentiment_score,
        "engagement_potential": content_metrics.engagement_potential,
        "platform_optimization": content_metrics.platform_optimization,
        "structure_score": content_metrics.content_structure,
        "cta_strength": content_metrics.call_to_action_strength,
        "weighted_score": heuristic_result.get("weighted_score", 0.0),
    }


async def _evaluate_with_reer(
    reer_scorer: REERCandidateScorer, content_item: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate content using REER candidate scorer."""

    # Mock REER evaluation
    await asyncio.sleep(0.1)

    # Score candidate using REER methodology
    score = await reer_scorer.score_candidate(content_item)

    return {
        "reer_score": score,
        "trajectory_quality": score * 0.9,  # Mock trajectory component
        "pattern_alignment": score * 1.1,  # Mock pattern component
        "refinement_potential": score * 0.95,  # Mock refinement component
    }


async def _evaluate_with_human(content_item: dict[str, Any]) -> dict[str, Any]:
    """Mock human evaluation (would be manual in real implementation)."""

    # Mock human evaluation scores
    await asyncio.sleep(0.01)

    import random

    return {
        "creativity": random.uniform(0.6, 0.9),
        "relevance": random.uniform(0.7, 0.95),
        "appeal": random.uniform(0.65, 0.9),
        "clarity": random.uniform(0.7, 0.95),
        "overall_quality": random.uniform(0.6, 0.9),
    }


def _calculate_summary_metrics(
    results: list[dict[str, Any]], evaluators: list[str]
) -> dict[str, Any]:
    """Calculate summary metrics across all evaluations."""

    summary = {}

    for evaluator in evaluators:
        evaluator_scores = []

        for result in results:
            if evaluator in result["evaluator_scores"]:
                scores = result["evaluator_scores"][evaluator]

                # Extract numeric scores
                numeric_scores = []
                for _key, value in scores.items():
                    if isinstance(value, int | float):
                        numeric_scores.append(value)

                if numeric_scores:
                    evaluator_scores.extend(numeric_scores)

        if evaluator_scores:
            summary[evaluator] = {
                "mean": statistics.mean(evaluator_scores),
                "median": statistics.median(evaluator_scores),
                "std_dev": (
                    statistics.stdev(evaluator_scores)
                    if len(evaluator_scores) > 1
                    else 0.0
                ),
                "min": min(evaluator_scores),
                "max": max(evaluator_scores),
                "count": len(evaluator_scores),
            }

    return summary


def _calculate_platform_breakdown(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate metrics breakdown by platform."""

    platform_stats = {}

    for result in results:
        platform = result.get("platform", "unknown")

        if platform not in platform_stats:
            platform_stats[platform] = {"count": 0, "scores": {}}

        platform_stats[platform]["count"] += 1

        # Aggregate scores by evaluator
        for evaluator, scores in result["evaluator_scores"].items():
            if evaluator not in platform_stats[platform]["scores"]:
                platform_stats[platform]["scores"][evaluator] = []

            # Extract numeric scores
            numeric_scores = []
            for _key, value in scores.items():
                if isinstance(value, int | float):
                    numeric_scores.append(value)

            if numeric_scores:
                platform_stats[platform]["scores"][evaluator].extend(numeric_scores)

    # Calculate averages
    for platform, stats in platform_stats.items():
        for evaluator, score_list in stats["scores"].items():
            if score_list:
                stats["scores"][evaluator] = {
                    "mean": statistics.mean(score_list),
                    "count": len(score_list),
                }

    return platform_stats


async def _run_comparative_analysis(
    current_results: list[dict[str, Any]],
    reference_data: list[dict[str, Any]],
    evaluator_instances: dict[str, Any],
) -> dict[str, Any]:
    """Run comparative analysis against reference data."""

    # Evaluate reference data with same evaluators
    reference_results = []

    for i, ref_item in enumerate(reference_data):
        ref_result = {
            "content_id": ref_item.get("id", f"ref_{i}"),
            "platform": ref_item.get("platform", "unknown"),
            "evaluator_scores": {},
        }

        # Evaluate each reference item (simplified for demo)
        for evaluator_name, evaluator in evaluator_instances.items():
            if evaluator_name == "kpi":
                score = await _evaluate_with_kpi(evaluator, ref_item)
            elif evaluator_name == "heuristic":
                score = await _evaluate_with_heuristic(evaluator, ref_item)
            elif evaluator_name == "reer":
                score = await _evaluate_with_reer(evaluator, ref_item)
            else:
                continue

            ref_result["evaluator_scores"][evaluator_name] = score

        reference_results.append(ref_result)

    # Calculate comparative metrics
    comparison = {
        "current_summary": _calculate_summary_metrics(
            current_results, list(evaluator_instances.keys())
        ),
        "reference_summary": _calculate_summary_metrics(
            reference_results, list(evaluator_instances.keys())
        ),
        "improvements": {},
        "regressions": {},
    }

    # Calculate improvements and regressions
    for evaluator in evaluator_instances:
        if (
            evaluator in comparison["current_summary"]
            and evaluator in comparison["reference_summary"]
        ):
            current_mean = comparison["current_summary"][evaluator]["mean"]
            reference_mean = comparison["reference_summary"][evaluator]["mean"]

            improvement = current_mean - reference_mean

            if improvement > 0:
                comparison["improvements"][evaluator] = improvement
            elif improvement < 0:
                comparison["regressions"][evaluator] = abs(improvement)

    return comparison


def _display_evaluation_results(results: dict[str, Any]):
    """Display evaluation results in formatted tables."""

    # Summary metrics table
    if results["summary_metrics"]:
        summary_table = Table(title="Evaluation Summary")
        summary_table.add_column("Evaluator", style="cyan")
        summary_table.add_column("Mean", style="green")
        summary_table.add_column("Median", style="yellow")
        summary_table.add_column("Std Dev", style="blue")
        summary_table.add_column("Range", style="magenta")
        summary_table.add_column("Count", style="white")

        for evaluator, stats in results["summary_metrics"].items():
            summary_table.add_row(
                evaluator.upper(),
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['std_dev']:.3f}",
                f"{stats['min']:.3f} - {stats['max']:.3f}",
                str(stats["count"]),
            )

        console.print(summary_table)
        console.print()

    # Platform breakdown
    if results["platform_breakdown"]:
        console.print("[cyan]Platform Breakdown:[/cyan]")

        platform_tree = Tree("Platforms")

        for platform, stats in results["platform_breakdown"].items():
            platform_branch = platform_tree.add(
                f"📱 {platform} ({stats['count']} items)"
            )

            for evaluator, scores in stats["scores"].items():
                if isinstance(scores, dict) and "mean" in scores:
                    platform_branch.add(f"{evaluator}: {scores['mean']:.3f}")

        console.print(platform_tree)
        console.print()

    # Comparative analysis
    if results["comparative_analysis"]:
        comp_analysis = results["comparative_analysis"]

        console.print("[cyan]Comparative Analysis:[/cyan]")

        if comp_analysis["improvements"]:
            console.print("[green]Improvements:[/green]")
            for evaluator, improvement in comp_analysis["improvements"].items():
                console.print(f"  • {evaluator}: +{improvement:.3f}")

        if comp_analysis["regressions"]:
            console.print("[red]Regressions:[/red]")
            for evaluator, regression in comp_analysis["regressions"].items():
                console.print(f"  • {evaluator}: -{regression:.3f}")

        if not comp_analysis["improvements"] and not comp_analysis["regressions"]:
            console.print("[yellow]No significant changes detected[/yellow]")

        console.print()


async def _generate_evaluation_report(results: dict[str, Any], report_file: Path):
    """Generate detailed evaluation report."""

    report = {
        "metadata": {
            "timestamp": results["timestamp"],
            "content_file": results["content_file"],
            "reference_file": results["reference_file"],
            "total_items_evaluated": len(results["evaluation_results"]),
        },
        "executive_summary": {},
        "detailed_metrics": results["summary_metrics"],
        "platform_analysis": results["platform_breakdown"],
        "comparative_analysis": results["comparative_analysis"],
        "recommendations": [],
    }

    # Generate executive summary
    if results["summary_metrics"]:
        best_evaluator = max(
            results["summary_metrics"].items(), key=lambda x: x[1]["mean"]
        )

        report["executive_summary"] = {
            "best_performing_evaluator": best_evaluator[0],
            "highest_mean_score": best_evaluator[1]["mean"],
            "evaluation_consistency": min(
                stats["std_dev"] for stats in results["summary_metrics"].values()
            ),
            "total_evaluations": sum(
                stats["count"] for stats in results["summary_metrics"].values()
            ),
        }

    # Generate recommendations
    recommendations = []

    if results["summary_metrics"]:
        for evaluator, stats in results["summary_metrics"].items():
            if stats["mean"] < 0.7:
                recommendations.append(
                    f"Consider improving {evaluator} scores (current mean: {stats['mean']:.3f})"
                )

            if stats["std_dev"] > 0.2:
                recommendations.append(
                    f"High variability in {evaluator} scores - review content consistency"
                )

    if (
        results["comparative_analysis"]
        and results["comparative_analysis"]["regressions"]
    ):
        for evaluator in results["comparative_analysis"]["regressions"]:
            recommendations.append(f"Address regression in {evaluator} performance")

    report["recommendations"] = recommendations or [
        "No specific recommendations - performance appears satisfactory"
    ]

    # Save report
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


@app.command()
def benchmark(
    models_dir: Path = typer.Argument(
        ..., help="Directory containing models to benchmark"
    ),
    test_data: Path = typer.Argument(..., help="Test dataset file"),
    output_dir: Path = typer.Option(
        Path("benchmarks"),
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metrics to include in benchmark (can be used multiple times)",
    ),
    baseline_model: str | None = typer.Option(
        None, "--baseline", help="Baseline model name for comparison"
    ),
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Number of evaluation iterations per model"
    ),
):
    """Benchmark multiple models against test dataset."""

    if not models_dir.exists():
        console.print(f"[red]Models directory not found:[/red] {models_dir}")
        raise typer.Exit(1)

    if not test_data.exists():
        console.print(f"[red]Test data not found:[/red] {test_data}")
        raise typer.Exit(1)

    # Find model files
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.json"))

    if not model_files:
        console.print(f"[yellow]No model files found in:[/yellow] {models_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[cyan]Benchmarking {len(model_files)} models with {iterations} iterations each...[/cyan]"
    )

    # Mock benchmarking results
    benchmark_results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking models...", total=len(model_files))

        for model_file in model_files:
            model_name = model_file.stem

            # Mock evaluation for each iteration
            iteration_results = []

            for _iteration in range(iterations):
                # Simulate model evaluation
                time.sleep(0.1)

                result = {
                    "accuracy": random.uniform(0.7, 0.95),
                    "precision": random.uniform(0.65, 0.9),
                    "recall": random.uniform(0.6, 0.9),
                    "f1_score": random.uniform(0.65, 0.9),
                    "engagement_score": random.uniform(0.5, 0.85),
                }
                iteration_results.append(result)

            # Calculate statistics across iterations
            model_stats = {}
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "engagement_score",
            ]:
                values = [r[metric] for r in iteration_results]
                model_stats[metric] = {
                    "mean": statistics.mean(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }

            benchmark_results[model_name] = model_stats
            progress.update(task, advance=1)

    # Display benchmark results
    _display_benchmark_results(benchmark_results, baseline_model)

    # Save benchmark results
    results_file = (
        output_dir
        / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    console.print(f"[green]Benchmark results saved to:[/green] {results_file}")


def _display_benchmark_results(results: dict[str, Any], baseline_model: str | None):
    """Display benchmark results in formatted table."""

    # Main results table
    results_table = Table(title="Model Benchmark Results")
    results_table.add_column("Model", style="cyan")
    results_table.add_column("Accuracy", style="green")
    results_table.add_column("Precision", style="yellow")
    results_table.add_column("Recall", style="blue")
    results_table.add_column("F1 Score", style="magenta")
    results_table.add_column("Engagement", style="white")

    # Sort models by accuracy (mean)
    sorted_models = sorted(
        results.items(), key=lambda x: x[1]["accuracy"]["mean"], reverse=True
    )

    for model_name, stats in sorted_models:
        # Highlight baseline model
        style = "bold" if model_name == baseline_model else None

        results_table.add_row(
            model_name,
            f"{stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std_dev']:.3f}",
            f"{stats['precision']['mean']:.3f} ± {stats['precision']['std_dev']:.3f}",
            f"{stats['recall']['mean']:.3f} ± {stats['recall']['std_dev']:.3f}",
            f"{stats['f1_score']['mean']:.3f} ± {stats['f1_score']['std_dev']:.3f}",
            f"{stats['engagement_score']['mean']:.3f} ± {stats['engagement_score']['std_dev']:.3f}",
            style=style,
        )

    console.print(results_table)

    # Show top performer
    best_model = sorted_models[0]
    console.print(
        f"\n[green]🏆 Best performing model:[/green] {best_model[0]} (Accuracy: {best_model[1]['accuracy']['mean']:.3f})"
    )


@app.command()
def compare(
    eval_results_1: Path = typer.Argument(..., help="First evaluation results file"),
    eval_results_2: Path = typer.Argument(..., help="Second evaluation results file"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for comparison results"
    ),
    significance_threshold: float = typer.Option(
        0.05,
        "--threshold",
        "-t",
        help="Significance threshold for detecting meaningful differences",
    ),
):
    """Compare two evaluation result files."""

    # Validate input files
    for file_path in [eval_results_1, eval_results_2]:
        if not file_path.exists():
            console.print(f"[red]Evaluation results file not found:[/red] {file_path}")
            raise typer.Exit(1)

    # Load results
    with open(eval_results_1) as f:
        results_1 = json.load(f)

    with open(eval_results_2) as f:
        results_2 = json.load(f)

    console.print("[cyan]Comparing evaluation results:[/cyan]")
    console.print(f"  File 1: {eval_results_1}")
    console.print(f"  File 2: {eval_results_2}")
    console.print()

    # Perform comparison
    comparison = _compare_evaluation_results(
        results_1, results_2, significance_threshold
    )

    # Display comparison
    _display_comparison_results(comparison)

    # Save comparison
    if output_file:
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        console.print(f"[green]Comparison results saved to:[/green] {output_file}")


def _compare_evaluation_results(
    results_1: dict[str, Any], results_2: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """Compare two evaluation result sets."""

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "file_1": results_1.get("content_file", "Unknown"),
        "file_2": results_2.get("content_file", "Unknown"),
        "metric_comparisons": {},
        "significant_differences": [],
        "summary": {},
    }

    # Compare summary metrics
    summary_1 = results_1.get("summary_metrics", {})
    summary_2 = results_2.get("summary_metrics", {})

    for evaluator in set(summary_1.keys()) | set(summary_2.keys()):
        if evaluator in summary_1 and evaluator in summary_2:
            stats_1 = summary_1[evaluator]
            stats_2 = summary_2[evaluator]

            mean_diff = stats_2["mean"] - stats_1["mean"]
            relative_change = (
                (mean_diff / stats_1["mean"]) * 100 if stats_1["mean"] != 0 else 0
            )

            comparison["metric_comparisons"][evaluator] = {
                "mean_1": stats_1["mean"],
                "mean_2": stats_2["mean"],
                "difference": mean_diff,
                "relative_change_percent": relative_change,
                "significant": abs(relative_change) > (threshold * 100),
            }

            if abs(relative_change) > (threshold * 100):
                comparison["significant_differences"].append(
                    {
                        "evaluator": evaluator,
                        "change": relative_change,
                        "direction": "improvement" if mean_diff > 0 else "regression",
                    }
                )

    # Summary statistics
    improvements = [
        d
        for d in comparison["significant_differences"]
        if d["direction"] == "improvement"
    ]
    regressions = [
        d
        for d in comparison["significant_differences"]
        if d["direction"] == "regression"
    ]

    comparison["summary"] = {
        "total_metrics_compared": len(comparison["metric_comparisons"]),
        "significant_differences": len(comparison["significant_differences"]),
        "improvements": len(improvements),
        "regressions": len(regressions),
    }

    return comparison


def _display_comparison_results(comparison: dict[str, Any]):
    """Display comparison results."""

    # Comparison table
    comp_table = Table(title="Evaluation Results Comparison")
    comp_table.add_column("Evaluator", style="cyan")
    comp_table.add_column("File 1", style="green")
    comp_table.add_column("File 2", style="yellow")
    comp_table.add_column("Difference", style="blue")
    comp_table.add_column("Change %", style="magenta")
    comp_table.add_column("Significant", style="red")

    for evaluator, metrics in comparison["metric_comparisons"].items():
        change_color = "green" if metrics["difference"] > 0 else "red"
        sig_marker = "✓" if metrics["significant"] else "✗"

        comp_table.add_row(
            evaluator.upper(),
            f"{metrics['mean_1']:.3f}",
            f"{metrics['mean_2']:.3f}",
            f"[{change_color}]{metrics['difference']:+.3f}[/{change_color}]",
            f"[{change_color}]{metrics['relative_change_percent']:+.1f}%[/{change_color}]",
            sig_marker,
        )

    console.print(comp_table)

    # Summary
    summary = comparison["summary"]
    console.print("\n[cyan]Summary:[/cyan]")
    console.print(f"  • Total metrics compared: {summary['total_metrics_compared']}")
    console.print(f"  • Significant differences: {summary['significant_differences']}")
    console.print(f"  • Improvements: [green]{summary['improvements']}[/green]")
    console.print(f"  • Regressions: [red]{summary['regressions']}[/red]")


import random  # Add this import at the top level

if __name__ == "__main__":
    app()
