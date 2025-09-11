#!/usr/bin/env python3
"""
REER × DSPy × MLX - GEPA Tuning CLI

Command-line interface for reflective prompt evolution using DSPy GEPA.
This replaces the prior GA-based implementation; only dspy.teleprompt.gepa.GEPA
is used going forward.
"""

import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dspy_program.gepa_runner import run_gepa

app = typer.Typer(
    name="social-gepa",
    help="GEPA tuning CLI for DSPy program optimization",
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
def tune(
    training_data: Path = typer.Argument(
        ..., help="Trainset JSON: list of {topic,audience}"
    ),
    output_dir: Path = typer.Option(
        Path("models/gepa"), "--output-dir", "-o", help="Output directory"
    ),
    gen_model: str = typer.Option(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "--gen-model",
        help="Generation LM (dspy.LM model)",
    ),
    reflection_model: str = typer.Option(
        "gpt-4o", "--reflection-model", help="Reflection LM for GEPA"
    ),
    auto: str = typer.Option(
        "light", "--auto", help="GEPA auto budget: light|medium|heavy"
    ),
    max_full_evals: int | None = typer.Option(
        None, "--max-full-evals", help="Override full eval budget"
    ),
    max_metric_calls: int | None = typer.Option(
        None, "--max-metric-calls", help="Override metric-call budget"
    ),
    track_stats: bool = typer.Option(
        True, "--track-stats/--no-track-stats", help="Attach detailed results"
    ),
    log_dir: Path | None = typer.Option(
        Path("logs/gepa"), "--log-dir", help="GEPA log dir"
    ),
    use_cot: bool = typer.Option(
        True, "--cot/--no-cot", help="Use Chain-of-Thought predictor"
    ),
    use_perplexity: bool = typer.Option(
        False,
        "--perplexity/--no-perplexity",
        help="Enable perplexity during scoring (slower)",
    ),
):
    """Tune a DSPy module using DSPy GEPA with repo-specific scoring as metric."""
    if not training_data.exists():
        console.print(f"[red]Training data not found:[/red] {training_data}")
        raise typer.Exit(1)

    if not training_data.exists():
        console.print(f"[red]Training data not found:[/red] {training_data}")
        raise typer.Exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show tuning parameters
    params_table = Table(title="GEPA Tuning Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Training Data", str(training_data))
    params_table.add_row("Output Directory", str(output_dir))
    params_table.add_row("Gen Model", gen_model)
    params_table.add_row("Reflection Model", reflection_model)
    params_table.add_row("Auto Budget", auto)
    if max_full_evals is not None:
        params_table.add_row("Max Full Evals", str(max_full_evals))
    if max_metric_calls is not None:
        params_table.add_row("Max Metric Calls", str(max_metric_calls))
    params_table.add_row("Track Stats", str(track_stats))
    if log_dir:
        params_table.add_row("Log Dir", str(log_dir))
    params_table.add_row("Chain of Thought", str(use_cot))
    params_table.add_row("Use Perplexity", str(use_perplexity))

    console.print(params_table)
    console.print()

    try:
        # Start GEPA tuning
        # Load training data
        with open(training_data) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Training data must be a list of {topic,audience}")

        # Run GEPA
        optimized = run_gepa(
            data,
            val_tasks=None,
            gen_model=gen_model,
            reflection_model=reflection_model,
            auto=auto,
            max_full_evals=max_full_evals,
            max_metric_calls=max_metric_calls,
            track_stats=track_stats,
            log_dir=str(log_dir) if log_dir else None,
            use_cot=use_cot,
            use_perplexity=use_perplexity,
        )

        # Persist program text/instructions minimally
        output_dir.mkdir(parents=True, exist_ok=True)
        prog_file = output_dir / "optimized_program.json"
        payload = {
            "predictors": {
                name: p.signature.instructions
                for name, p in optimized.named_predictors()
            },
            "detailed_results": (
                getattr(optimized, "detailed_results", None).to_dict()
                if getattr(optimized, "detailed_results", None)
                else None
            ),
        }
        with open(prog_file, "w") as f:
            json.dump(payload, f, indent=2)

        console.print("[green]✓ GEPA (DSPy) tuning completed successfully![/green]")
        console.print(f"[cyan]Saved optimized program to:[/cyan] {prog_file}")

    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        console.print(f"[red]Error during tuning:[/red] {e}")
        raise typer.Exit(1)


"""
Note: GA helper functions were removed. Evaluation and comparison below are adapted
to the JSON output produced by the DSPy GEPA flow (optimized_program.json).
"""


@app.command()
def evaluate(
    model_file: Path = typer.Argument(..., help="Tuned model file to evaluate"),
    test_data: Path = typer.Argument(..., help="Test data file"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for evaluation results"
    ),
    show_instructions: bool = typer.Option(
        False,
        "--show-instructions/--no-show-instructions",
        help="Show predictor instructions summary",
    ),
    metrics: list[str] | None = typer.Option(
        None, "--metric", "-m", help="Evaluation metrics (can be used multiple times)"
    ),
    platforms: list[str] | None = typer.Option(
        None,
        "--platform",
        "-p",
        help="Evaluate on specific platforms (can be used multiple times)",
    ),
):
    """Evaluate an optimized program JSON (DSPy GEPA output)."""

    if not model_file.exists():
        console.print(f"[red]Model file not found:[/red] {model_file}")
        raise typer.Exit(1)

    if not test_data.exists():
        console.print(f"[red]Test data not found:[/red] {test_data}")
        raise typer.Exit(1)

    console.print(f"[cyan]Evaluating model:[/cyan] {model_file}")
    console.print(f"[cyan]Test data:[/cyan] {test_data}")

    # Load program JSON
    with open(model_file) as f:
        prog = json.load(f)

    detailed = prog.get("detailed_results") or {}
    scores = detailed.get("val_aggregate_scores") or []
    best = max(scores) if scores else None
    avg = sum(scores) / len(scores) if scores else None
    num_cands = len(detailed.get("candidates") or [])
    num_val = len(detailed.get("val_subscores") or [[]][0]) if scores else 0

    results = {
        "num_candidates": num_cands,
        "num_val_instances": num_val,
        "best_val_score": best,
        "avg_val_score": avg,
    }

    # Display results
    eval_table = Table(title="Model Evaluation Results")
    eval_table.add_column("Metric", style="cyan")
    eval_table.add_column("Score", style="green")

    for metric, score in results.items():
        val = (
            "-"
            if score is None
            else f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        )
        eval_table.add_row(metric.replace("_", " ").title(), val)

    console.print(eval_table)

    # Optionally display predictor instruction previews
    if show_instructions and prog.get("predictors"):
        instr = prog.get("predictors") or {}
        preview_table = Table(title="Predictor Instructions (Preview)")
        preview_table.add_column("Predictor", style="cyan")
        preview_table.add_column("Instructions (first 120 chars)", style="green")
        for name, text in instr.items():
            snippet = (text or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            preview_table.add_row(name, snippet)
        console.print(preview_table)

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Evaluation results saved to:[/green] {output_file}")


@app.command()
def compare(
    model_dir: Path = typer.Argument(
        ..., help="Directory containing models to compare"
    ),
    metric: str = typer.Option(
        "best_val_score", "--metric", "-m", help="Metric for comparison"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top models to show"),
):
    """Compare multiple tuned models."""

    if not model_dir.exists():
        console.print(f"[red]Model directory not found:[/red] {model_dir}")
        raise typer.Exit(1)

    # Find program files
    model_files = list(model_dir.glob("*.json"))

    if not model_files:
        console.print(f"[yellow]No model files found in:[/yellow] {model_dir}")
        return

    console.print(f"[cyan]Comparing {len(model_files)} models...[/cyan]")

    # Comparison based on detailed_results.val_aggregate_scores
    summary = []
    for mf in model_files:
        try:
            with open(mf) as f:
                prog = json.load(f)
            detailed = prog.get("detailed_results") or {}
            scores = detailed.get("val_aggregate_scores") or []
            best = max(scores) if scores else None
            avg = sum(scores) / len(scores) if scores else None
            summary.append(
                {
                    "file": mf.name,
                    "best_val_score": best or 0.0,
                    "avg_val_score": avg or 0.0,
                    "num_candidates": len(detailed.get("candidates") or []),
                }
            )
        except Exception:
            continue

    summary.sort(key=lambda x: x.get(metric, 0.0), reverse=True)

    comparison_table = Table(title=f"Model Comparison (Top {top_k})")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Best Val Score", style="green")
    comparison_table.add_column("Avg Val Score", style="yellow")
    comparison_table.add_column("Candidates", style="magenta")

    for item in summary[:top_k]:
        comparison_table.add_row(
            item["file"],
            f"{item['best_val_score']:.4f}",
            f"{item['avg_val_score']:.4f}",
            str(item["num_candidates"]),
        )

    console.print(comparison_table)


if __name__ == "__main__":
    app()
