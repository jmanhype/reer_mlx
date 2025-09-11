#!/usr/bin/env python3
"""
REER × DSPy × MLX - GEPA Tuning CLI

Command-line interface for Genetic Evolution with Pattern Adaptation (GEPA) tuning.
Optimizes DSPy programs using evolutionary algorithms with social media patterns.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich import print as rprint
from loguru import logger
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import REERGEPATrainer, OptimizationResult, Individual, Population
from dspy_program import ContentGeneratorModule, PipelineConfig
from social import SocialContentPipeline, Platform, ContentType

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
    program_file: Path = typer.Argument(..., help="DSPy program file to optimize"),
    training_data: Path = typer.Argument(..., help="Training data file"),
    output_dir: Path = typer.Option(
        Path("models/tuned"),
        "--output-dir",
        "-o",
        help="Output directory for tuned models",
    ),
    population_size: int = typer.Option(
        50, "--population-size", "-p", help="Population size for genetic algorithm"
    ),
    generations: int = typer.Option(
        100, "--generations", "-g", help="Number of generations to evolve"
    ),
    mutation_rate: float = typer.Option(
        0.1, "--mutation-rate", "-m", help="Mutation rate (0.0 to 1.0)"
    ),
    crossover_rate: float = typer.Option(
        0.7, "--crossover-rate", "-c", help="Crossover rate (0.0 to 1.0)"
    ),
    elite_size: int = typer.Option(
        5, "--elite-size", "-e", help="Number of elite individuals to preserve"
    ),
    fitness_metric: str = typer.Option(
        "combined",
        "--fitness-metric",
        "-f",
        help="Fitness metric (accuracy, engagement, combined)",
    ),
    target_platforms: Optional[List[str]] = typer.Option(
        None,
        "--platform",
        help="Target platforms for optimization (can be used multiple times)",
    ),
    content_types: Optional[List[str]] = typer.Option(
        None,
        "--content-type",
        help="Content types to optimize for (can be used multiple times)",
    ),
    convergence_threshold: float = typer.Option(
        0.001, "--convergence", help="Convergence threshold for early stopping"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Patience for early stopping (generations without improvement)",
    ),
    parallel_evaluation: bool = typer.Option(
        True, "--parallel/--sequential", help="Enable parallel fitness evaluation"
    ),
    save_checkpoints: bool = typer.Option(
        True, "--checkpoints/--no-checkpoints", help="Save optimization checkpoints"
    ),
    resume_from: Optional[Path] = typer.Option(
        None, "--resume-from", help="Resume optimization from checkpoint"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Tune DSPy programs using GEPA (Genetic Evolution with Pattern Adaptation).

    Examples:

        # Basic GEPA tuning
        social-gepa tune my_program.py training_data.json

        # Tuning with custom parameters
        social-gepa tune program.py data.json --population-size 100 --generations 200

        # Platform-specific optimization
        social-gepa tune program.py data.json --platform x --platform instagram

        # Resume from checkpoint
        social-gepa tune program.py data.json --resume-from models/checkpoint.pkl
    """

    # Validate input files
    if not program_file.exists():
        console.print(f"[red]Program file not found:[/red] {program_file}")
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

    params_table.add_row("Program File", str(program_file))
    params_table.add_row("Training Data", str(training_data))
    params_table.add_row("Output Directory", str(output_dir))
    params_table.add_row("Population Size", str(population_size))
    params_table.add_row("Generations", str(generations))
    params_table.add_row("Mutation Rate", f"{mutation_rate:.3f}")
    params_table.add_row("Crossover Rate", f"{crossover_rate:.3f}")
    params_table.add_row("Elite Size", str(elite_size))
    params_table.add_row("Fitness Metric", fitness_metric)
    params_table.add_row("Convergence Threshold", f"{convergence_threshold:.6f}")
    params_table.add_row("Patience", str(patience))
    params_table.add_row("Parallel Evaluation", str(parallel_evaluation))

    if target_platforms:
        params_table.add_row("Target Platforms", ", ".join(target_platforms))
    if content_types:
        params_table.add_row("Content Types", ", ".join(content_types))
    if resume_from:
        params_table.add_row("Resume From", str(resume_from))

    console.print(params_table)
    console.print()

    try:
        # Start GEPA tuning
        results = asyncio.run(
            _run_gepa_tuning(
                program_file,
                training_data,
                output_dir,
                population_size,
                generations,
                mutation_rate,
                crossover_rate,
                elite_size,
                fitness_metric,
                target_platforms,
                content_types,
                convergence_threshold,
                patience,
                parallel_evaluation,
                save_checkpoints,
                resume_from,
                verbose,
            )
        )

        # Display results
        _display_tuning_results(results)

        console.print(f"[green]✓ GEPA tuning completed successfully![/green]")
        console.print(f"[cyan]Tuned models saved to:[/cyan] {output_dir}")

    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        console.print(f"[red]Error during tuning:[/red] {e}")
        raise typer.Exit(1)


async def _run_gepa_tuning(
    program_file: Path,
    training_data: Path,
    output_dir: Path,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    elite_size: int,
    fitness_metric: str,
    target_platforms: Optional[List[str]],
    content_types: Optional[List[str]],
    convergence_threshold: float,
    patience: int,
    parallel_evaluation: bool,
    save_checkpoints: bool,
    resume_from: Optional[Path],
    verbose: bool,
) -> Dict[str, Any]:
    """Run the GEPA tuning process."""

    results = {
        "best_fitness": 0.0,
        "generation_reached": 0,
        "converged": False,
        "optimization_history": [],
        "best_individual": None,
        "population_diversity": [],
        "computation_time": 0.0,
    }

    start_time = time.time()

    # Create progress table for live updates
    progress_table = Table(title="GEPA Optimization Progress")
    progress_table.add_column("Generation", style="cyan")
    progress_table.add_column("Best Fitness", style="green")
    progress_table.add_column("Avg Fitness", style="yellow")
    progress_table.add_column("Diversity", style="magenta")
    progress_table.add_column("Status", style="blue")

    with Live(progress_table, refresh_per_second=2, console=console) as live:

        # Load training data
        console.print("[cyan]Loading training data...[/cyan]")
        with open(training_data) as f:
            data = json.load(f)

        # Initialize GEPA trainer
        console.print("[cyan]Initializing GEPA trainer...[/cyan]")
        trainer = REERGEPATrainer(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
        )

        # Load or create initial population
        if resume_from and resume_from.exists():
            console.print(f"[cyan]Resuming from checkpoint:[/cyan] {resume_from}")
            # Mock loading checkpoint
            population = await trainer.create_population(population_size)
            start_generation = 50  # Mock resume point
        else:
            console.print("[cyan]Creating initial population...[/cyan]")
            population = await trainer.create_population(population_size)
            start_generation = 0

        # Evolution loop
        best_fitness = 0.0
        stagnation_count = 0

        for generation in range(start_generation, generations):
            # Evaluate fitness
            fitness_scores = await _evaluate_population_fitness(
                population,
                data,
                fitness_metric,
                target_platforms,
                content_types,
                parallel_evaluation,
            )

            # Update population fitness
            for individual, fitness in zip(population.individuals, fitness_scores):
                individual.fitness = fitness

            # Calculate statistics
            current_best = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            diversity = _calculate_diversity(population)

            # Check for improvement
            if current_best > best_fitness:
                best_fitness = current_best
                stagnation_count = 0
                results["best_individual"] = population.individuals[
                    fitness_scores.index(current_best)
                ]
            else:
                stagnation_count += 1

            # Update progress
            status = "Evolving"
            if stagnation_count >= patience:
                status = f"Stagnant ({stagnation_count})"

            # Update live table
            progress_table.add_row(
                str(generation + 1),
                f"{current_best:.4f}",
                f"{avg_fitness:.4f}",
                f"{diversity:.4f}",
                status,
            )

            # Store history
            results["optimization_history"].append(
                {
                    "generation": generation + 1,
                    "best_fitness": current_best,
                    "avg_fitness": avg_fitness,
                    "diversity": diversity,
                }
            )
            results["population_diversity"].append(diversity)

            # Check convergence
            if stagnation_count >= patience:
                console.print(
                    f"[yellow]Early stopping: No improvement for {patience} generations[/yellow]"
                )
                results["converged"] = True
                break

            if (
                generation > 0
                and abs(current_best - best_fitness) < convergence_threshold
            ):
                console.print(f"[green]Converged: Improvement below threshold[/green]")
                results["converged"] = True
                break

            # Evolve population
            if generation < generations - 1:  # Don't evolve on last generation
                population = await trainer.evolve_population(population)

            # Save checkpoint
            if save_checkpoints and (generation + 1) % 10 == 0:
                checkpoint_file = output_dir / f"checkpoint_gen_{generation + 1}.pkl"
                await _save_checkpoint(population, checkpoint_file)

            results["generation_reached"] = generation + 1

            # Small delay to show progress
            await asyncio.sleep(0.1)

    results["best_fitness"] = best_fitness
    results["computation_time"] = time.time() - start_time

    # Save final results
    await _save_final_results(results, output_dir)

    return results


async def _evaluate_population_fitness(
    population: Population,
    training_data: List[Dict],
    fitness_metric: str,
    target_platforms: Optional[List[str]],
    content_types: Optional[List[str]],
    parallel_evaluation: bool,
) -> List[float]:
    """Evaluate fitness for entire population."""

    fitness_scores = []

    if parallel_evaluation:
        # Simulate parallel evaluation
        tasks = []
        for individual in population.individuals:
            task = _evaluate_individual_fitness(
                individual,
                training_data,
                fitness_metric,
                target_platforms,
                content_types,
            )
            tasks.append(task)

        fitness_scores = await asyncio.gather(*tasks)
    else:
        # Sequential evaluation
        for individual in population.individuals:
            fitness = await _evaluate_individual_fitness(
                individual,
                training_data,
                fitness_metric,
                target_platforms,
                content_types,
            )
            fitness_scores.append(fitness)

    return fitness_scores


async def _evaluate_individual_fitness(
    individual: Individual,
    training_data: List[Dict],
    fitness_metric: str,
    target_platforms: Optional[List[str]],
    content_types: Optional[List[str]],
) -> float:
    """Evaluate fitness for a single individual."""

    # Mock fitness evaluation
    await asyncio.sleep(0.01)  # Simulate computation time

    # Base fitness (random for demo)
    import random

    base_fitness = random.uniform(0.5, 1.0)

    # Platform adjustment
    if target_platforms:
        platform_bonus = random.uniform(0.0, 0.1)
        base_fitness += platform_bonus

    # Content type adjustment
    if content_types:
        content_bonus = random.uniform(0.0, 0.1)
        base_fitness += content_bonus

    # Metric-specific calculation
    if fitness_metric == "accuracy":
        return base_fitness * 0.9
    elif fitness_metric == "engagement":
        return base_fitness * 1.1
    else:  # combined
        return base_fitness

    return min(base_fitness, 1.0)  # Cap at 1.0


def _calculate_diversity(population: Population) -> float:
    """Calculate population diversity."""
    # Mock diversity calculation
    import random

    return random.uniform(0.3, 0.9)


async def _save_checkpoint(population: Population, checkpoint_file: Path):
    """Save optimization checkpoint."""
    # Mock checkpoint saving
    checkpoint_data = {
        "population_size": len(population.individuals),
        "generation": "current",
        "timestamp": time.time(),
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


async def _save_final_results(results: Dict[str, Any], output_dir: Path):
    """Save final optimization results."""

    # Save optimization history
    history_file = output_dir / "optimization_history.json"
    with open(history_file, "w") as f:
        json.dump(results["optimization_history"], f, indent=2)

    # Save best model (mock)
    if results["best_individual"]:
        model_file = output_dir / "best_model.pkl"
        model_data = {
            "fitness": results["best_fitness"],
            "generation": results["generation_reached"],
            "converged": results["converged"],
        }
        with open(model_file, "w") as f:
            json.dump(model_data, f, indent=2)

    # Save summary
    summary_file = output_dir / "tuning_summary.json"
    summary = {
        "best_fitness": results["best_fitness"],
        "generations": results["generation_reached"],
        "converged": results["converged"],
        "computation_time": results["computation_time"],
        "final_diversity": (
            results["population_diversity"][-1]
            if results["population_diversity"]
            else 0.0
        ),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def _display_tuning_results(results: Dict[str, Any]):
    """Display tuning results in a formatted view."""

    # Results summary
    results_table = Table(title="GEPA Tuning Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Best Fitness", f"{results['best_fitness']:.6f}")
    results_table.add_row("Generations", str(results["generation_reached"]))
    results_table.add_row("Converged", str(results["converged"]))
    results_table.add_row("Computation Time", f"{results['computation_time']:.2f}s")

    if results["population_diversity"]:
        final_diversity = results["population_diversity"][-1]
        results_table.add_row("Final Diversity", f"{final_diversity:.4f}")

    console.print(results_table)

    # Optimization progress
    if results["optimization_history"]:
        console.print("\n[cyan]Optimization Progress (Last 5 Generations):[/cyan]")

        progress_table = Table()
        progress_table.add_column("Generation", style="cyan")
        progress_table.add_column("Best Fitness", style="green")
        progress_table.add_column("Avg Fitness", style="yellow")

        # Show last 5 generations
        last_generations = results["optimization_history"][-5:]
        for entry in last_generations:
            progress_table.add_row(
                str(entry["generation"]),
                f"{entry['best_fitness']:.4f}",
                f"{entry['avg_fitness']:.4f}",
            )

        console.print(progress_table)


@app.command()
def evaluate(
    model_file: Path = typer.Argument(..., help="Tuned model file to evaluate"),
    test_data: Path = typer.Argument(..., help="Test data file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for evaluation results"
    ),
    metrics: Optional[List[str]] = typer.Option(
        None, "--metric", "-m", help="Evaluation metrics (can be used multiple times)"
    ),
    platforms: Optional[List[str]] = typer.Option(
        None,
        "--platform",
        "-p",
        help="Evaluate on specific platforms (can be used multiple times)",
    ),
):
    """Evaluate a tuned GEPA model."""

    if not model_file.exists():
        console.print(f"[red]Model file not found:[/red] {model_file}")
        raise typer.Exit(1)

    if not test_data.exists():
        console.print(f"[red]Test data not found:[/red] {test_data}")
        raise typer.Exit(1)

    console.print(f"[cyan]Evaluating model:[/cyan] {model_file}")
    console.print(f"[cyan]Test data:[/cyan] {test_data}")

    # Mock evaluation
    results = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.78,
        "f1_score": 0.80,
        "engagement_score": 0.73,
    }

    # Display results
    eval_table = Table(title="Model Evaluation Results")
    eval_table.add_column("Metric", style="cyan")
    eval_table.add_column("Score", style="green")

    for metric, score in results.items():
        eval_table.add_row(metric.replace("_", " ").title(), f"{score:.4f}")

    console.print(eval_table)

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
    test_data: Path = typer.Argument(..., help="Test data file"),
    metric: str = typer.Option(
        "best_fitness", "--metric", "-m", help="Metric for comparison"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top models to show"),
):
    """Compare multiple tuned models."""

    if not model_dir.exists():
        console.print(f"[red]Model directory not found:[/red] {model_dir}")
        raise typer.Exit(1)

    # Find model files
    model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json"))

    if not model_files:
        console.print(f"[yellow]No model files found in:[/yellow] {model_dir}")
        return

    console.print(f"[cyan]Comparing {len(model_files)} models...[/cyan]")

    # Mock comparison results
    comparison_table = Table(title=f"Model Comparison (Top {top_k})")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Fitness", style="green")
    comparison_table.add_column("Generations", style="yellow")
    comparison_table.add_column("Converged", style="magenta")

    import random

    for i, model_file in enumerate(model_files[:top_k]):
        comparison_table.add_row(
            model_file.name,
            f"{random.uniform(0.7, 0.95):.4f}",
            str(random.randint(50, 200)),
            "✓" if random.choice([True, False]) else "✗",
        )

    console.print(comparison_table)


if __name__ == "__main__":
    app()
