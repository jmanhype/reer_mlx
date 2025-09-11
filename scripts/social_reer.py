#!/usr/bin/env python3
"""
REER × DSPy × MLX - REER Mining CLI

Command-line interface for REER (Retrieve-Extract-Execute-Refine) mining operations.
Processes social media data to extract insights and patterns.
"""

import asyncio
import json
from pathlib import Path
import sys
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


from core import REERCandidateScorer, REERTraceStore, REERTrajectorySynthesizer
from core.trace_store import TraceRecord

app = typer.Typer(
    name="social-reer",
    help="REER mining CLI for social media data processing",
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
def mine(
    input_file: Path = typer.Argument(..., help="Input data file to process"),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output-dir",
        "-o",
        help="Output directory for processed data",
    ),
    trace_store: Path = typer.Option(
        Path("data/traces"), "--trace-store", "-t", help="Trace store directory"
    ),
    algorithm: str = typer.Option(
        "reer", "--algorithm", "-a", help="Mining algorithm (reer, extract, synthesize)"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="Batch size for processing"
    ),
    min_engagement: int = typer.Option(
        10, "--min-engagement", help="Minimum engagement threshold"
    ),
    platforms: list[str] | None = typer.Option(
        None,
        "--platform",
        "-p",
        help="Filter by platforms (can be used multiple times)",
    ),
    content_types: list[str] | None = typer.Option(
        None,
        "--content-type",
        "-c",
        help="Filter by content types (can be used multiple times)",
    ),
    extract_patterns: bool = typer.Option(
        True, "--extract-patterns/--no-patterns", help="Extract content patterns"
    ),
    synthesize_strategies: bool = typer.Option(
        True, "--synthesize/--no-synthesize", help="Synthesize content strategies"
    ),
    score_candidates: bool = typer.Option(
        True, "--score/--no-score", help="Score content candidates"
    ),
    parallel_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of parallel workers"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Mine social media data using REER methodology.

    Examples:

        # Basic REER mining
        social-reer mine data/raw/x_data.json

        # Extract patterns only
        social-reer mine data.json --algorithm extract --no-synthesize --no-score

        # Mine with platform filtering
        social-reer mine data.json --platform x --platform instagram --min-engagement 50

        # Parallel processing
        social-reer mine large_data.json --workers 8 --batch-size 500
    """

    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Input file not found:[/red] {input_file}")
        raise typer.Exit(1)

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_store.mkdir(parents=True, exist_ok=True)

    # Show mining parameters
    params_table = Table(title="REER Mining Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Input File", str(input_file))
    params_table.add_row("Output Directory", str(output_dir))
    params_table.add_row("Algorithm", algorithm)
    params_table.add_row("Batch Size", str(batch_size))
    params_table.add_row("Min Engagement", str(min_engagement))
    params_table.add_row("Workers", str(parallel_workers))
    params_table.add_row("Extract Patterns", str(extract_patterns))
    params_table.add_row("Synthesize Strategies", str(synthesize_strategies))
    params_table.add_row("Score Candidates", str(score_candidates))

    if platforms:
        params_table.add_row("Platforms", ", ".join(platforms))
    if content_types:
        params_table.add_row("Content Types", ", ".join(content_types))

    console.print(params_table)
    console.print()

    try:
        # Start mining process
        results = asyncio.run(
            _run_reer_mining(
                input_file,
                output_dir,
                trace_store,
                algorithm,
                batch_size,
                min_engagement,
                platforms,
                content_types,
                extract_patterns,
                synthesize_strategies,
                score_candidates,
                parallel_workers,
                verbose,
            )
        )

        # Display results
        _display_mining_results(results)

        console.print("[green]✓ REER mining completed successfully![/green]")
        console.print(f"[cyan]Results saved to:[/cyan] {output_dir}")

    except Exception as e:
        logger.error(f"Mining failed: {e}")
        console.print(f"[red]Error during mining:[/red] {e}")
        raise typer.Exit(1)


async def _run_reer_mining(
    input_file: Path,
    output_dir: Path,
    trace_store_path: Path,
    algorithm: str,
    batch_size: int,
    min_engagement: int,
    platforms: list[str] | None,
    content_types: list[str] | None,
    extract_patterns: bool,
    synthesize_strategies: bool,
    score_candidates: bool,
    parallel_workers: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run the REER mining process."""

    results = {
        "processed_records": 0,
        "extracted_patterns": [],
        "synthesized_strategies": [],
        "scored_candidates": [],
        "trace_records": 0,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Load input data
        load_task = progress.add_task("Loading input data...", total=None)

        try:
            with open(input_file) as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load input data: {e}")

        if not isinstance(data, list):
            data = [data]

        progress.update(load_task, description=f"Loaded {len(data)} records")

        # Filter data
        if platforms or content_types or min_engagement:
            filter_task = progress.add_task("Filtering data...", total=len(data))
            filtered_data = []

            for i, record in enumerate(data):
                # Apply filters
                if platforms and record.get("platform") not in platforms:
                    continue

                if content_types and record.get("content_type") not in content_types:
                    continue

                engagement = record.get("engagement", {})
                total_engagement = (
                    sum(engagement.values()) if isinstance(engagement, dict) else 0
                )
                if total_engagement < min_engagement:
                    continue

                filtered_data.append(record)
                progress.update(filter_task, advance=1)

            data = filtered_data
            progress.update(filter_task, description=f"Filtered to {len(data)} records")

        # Initialize REER components
        init_task = progress.add_task("Initializing REER components...", total=None)

        trace_store = REERTraceStore(str(trace_store_path))
        synthesizer = REERTrajectorySynthesizer()
        scorer = REERCandidateScorer()

        progress.update(init_task, description="REER components initialized")

        # Process data in batches
        process_task = progress.add_task("Processing data...", total=len(data))

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # RETRIEVE: Store traces
            if algorithm in ["reer", "retrieve"]:
                for record in batch:
                    trace_record = TraceRecord(
                        trace_id=f"trace_{i}_{record.get('id', 'unknown')}",
                        execution_path=[record],
                        input_data=record,
                        output_data=record,
                        metadata={"platform": record.get("platform", "unknown")},
                    )
                    await trace_store.store_trace(trace_record)
                    results["trace_records"] += 1

            # EXTRACT: Pattern extraction
            if extract_patterns and algorithm in ["reer", "extract"]:
                for record in batch:
                    # Mock pattern extraction
                    pattern = {
                        "content_length": len(record.get("content", "")),
                        "hashtag_count": len(record.get("hashtags", [])),
                        "engagement_rate": sum(record.get("engagement", {}).values()),
                        "platform": record.get("platform"),
                        "time_pattern": record.get("timestamp", "")[:10],  # Date part
                    }
                    results["extracted_patterns"].append(pattern)

            # EXECUTE: Strategy synthesis
            if synthesize_strategies and algorithm in ["reer", "synthesize"]:
                strategies = await synthesizer.synthesize_strategies(batch)
                results["synthesized_strategies"].extend(strategies)

            # REFINE: Candidate scoring
            if score_candidates and algorithm in ["reer", "score"]:
                for record in batch:
                    score = await scorer.score_candidate(record)
                    results["scored_candidates"].append(
                        {
                            "id": record.get("id"),
                            "score": score,
                            "platform": record.get("platform"),
                        }
                    )

            results["processed_records"] += len(batch)
            progress.update(process_task, advance=len(batch))

        # Save results
        save_task = progress.add_task("Saving results...", total=None)

        # Save extracted patterns
        if results["extracted_patterns"]:
            patterns_file = output_dir / "extracted_patterns.json"
            with open(patterns_file, "w") as f:
                json.dump(results["extracted_patterns"], f, indent=2)

        # Save synthesized strategies
        if results["synthesized_strategies"]:
            strategies_file = output_dir / "synthesized_strategies.json"
            with open(strategies_file, "w") as f:
                json.dump(results["synthesized_strategies"], f, indent=2)

        # Save scored candidates
        if results["scored_candidates"]:
            candidates_file = output_dir / "scored_candidates.json"
            with open(candidates_file, "w") as f:
                json.dump(results["scored_candidates"], f, indent=2)

        # Save mining summary
        summary_file = output_dir / "mining_summary.json"
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "algorithm": algorithm,
                    "processed_records": results["processed_records"],
                    "patterns_count": len(results["extracted_patterns"]),
                    "strategies_count": len(results["synthesized_strategies"]),
                    "candidates_count": len(results["scored_candidates"]),
                    "trace_records": results["trace_records"],
                },
                f,
                indent=2,
            )

        progress.update(save_task, description="Results saved")

    return results


def _display_mining_results(results: dict[str, Any]):
    """Display mining results in a formatted table."""

    results_table = Table(title="REER Mining Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Count", style="green")

    results_table.add_row("Processed Records", str(results["processed_records"]))
    results_table.add_row("Extracted Patterns", str(len(results["extracted_patterns"])))
    results_table.add_row(
        "Synthesized Strategies", str(len(results["synthesized_strategies"]))
    )
    results_table.add_row("Scored Candidates", str(len(results["scored_candidates"])))
    results_table.add_row("Trace Records", str(results["trace_records"]))

    console.print(results_table)


@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Results directory to analyze"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for analysis report"
    ),
    include_charts: bool = typer.Option(
        False, "--charts", help="Include visualization charts"
    ),
):
    """Analyze REER mining results."""

    if not results_dir.exists():
        console.print(f"[red]Results directory not found:[/red] {results_dir}")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing results in:[/cyan] {results_dir}")

    # Find result files
    patterns_file = results_dir / "extracted_patterns.json"
    strategies_file = results_dir / "synthesized_strategies.json"
    candidates_file = results_dir / "scored_candidates.json"
    results_dir / "mining_summary.json"

    analysis = {}

    # Load and analyze patterns
    if patterns_file.exists():
        with open(patterns_file) as f:
            patterns = json.load(f)

        analysis["patterns"] = {
            "total": len(patterns),
            "avg_content_length": (
                sum(p.get("content_length", 0) for p in patterns) / len(patterns)
                if patterns
                else 0
            ),
            "avg_hashtags": (
                sum(p.get("hashtag_count", 0) for p in patterns) / len(patterns)
                if patterns
                else 0
            ),
            "platforms": list(
                {p.get("platform") for p in patterns if p.get("platform")}
            ),
        }

    # Load and analyze strategies
    if strategies_file.exists():
        with open(strategies_file) as f:
            strategies = json.load(f)

        analysis["strategies"] = {
            "total": len(strategies),
            "types": list({s.get("type") for s in strategies if s.get("type")}),
        }

    # Load and analyze candidates
    if candidates_file.exists():
        with open(candidates_file) as f:
            candidates = json.load(f)

        scores = [
            c.get("score", 0)
            for c in candidates
            if isinstance(c.get("score"), int | float)
        ]
        analysis["candidates"] = {
            "total": len(candidates),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
        }

    # Display analysis
    _display_analysis(analysis)

    # Save analysis report
    if output_file:
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        console.print(f"[green]Analysis saved to:[/green] {output_file}")


def _display_analysis(analysis: dict[str, Any]):
    """Display analysis results."""

    tree = Tree("REER Mining Analysis")

    if "patterns" in analysis:
        patterns_branch = tree.add("📊 Extracted Patterns")
        patterns_branch.add(f"Total: {analysis['patterns']['total']}")
        patterns_branch.add(
            f"Avg Content Length: {analysis['patterns']['avg_content_length']:.1f}"
        )
        patterns_branch.add(f"Avg Hashtags: {analysis['patterns']['avg_hashtags']:.1f}")
        patterns_branch.add(
            f"Platforms: {', '.join(analysis['patterns']['platforms'])}"
        )

    if "strategies" in analysis:
        strategies_branch = tree.add("🎯 Synthesized Strategies")
        strategies_branch.add(f"Total: {analysis['strategies']['total']}")
        strategies_branch.add(f"Types: {', '.join(analysis['strategies']['types'])}")

    if "candidates" in analysis:
        candidates_branch = tree.add("⭐ Scored Candidates")
        candidates_branch.add(f"Total: {analysis['candidates']['total']}")
        candidates_branch.add(f"Avg Score: {analysis['candidates']['avg_score']:.2f}")
        candidates_branch.add(
            f"Score Range: {analysis['candidates']['min_score']:.2f} - {analysis['candidates']['max_score']:.2f}"
        )

    console.print(tree)


@app.command()
def traces(
    trace_store: Path = typer.Option(
        Path("data/traces"), "--trace-store", "-t", help="Trace store directory"
    ),
    action: str = typer.Option(
        "list", "--action", "-a", help="Action to perform (list, search, export)"
    ),
    trace_id: str | None = typer.Option(
        None, "--trace-id", help="Specific trace ID to query"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Limit number of results"),
):
    """Manage and query trace store."""

    if not trace_store.exists():
        console.print(f"[red]Trace store not found:[/red] {trace_store}")
        raise typer.Exit(1)

    REERTraceStore(str(trace_store))

    if action == "list":
        console.print(f"[cyan]Listing traces in:[/cyan] {trace_store}")
        # Mock listing traces
        console.print("[yellow]Trace listing functionality coming soon...[/yellow]")

    elif action == "search":
        console.print("[cyan]Searching traces...[/cyan]")
        # Mock search functionality
        console.print("[yellow]Trace search functionality coming soon...[/yellow]")

    elif action == "export":
        console.print("[cyan]Exporting traces...[/cyan]")
        # Mock export functionality
        console.print("[yellow]Trace export functionality coming soon...[/yellow]")


if __name__ == "__main__":
    app()
