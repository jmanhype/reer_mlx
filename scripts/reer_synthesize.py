#!/usr/bin/env python3
"""
REER Trajectory Synthesis CLI (MVP)

Reads a small corpus of (x,y) pairs and runs local-search to produce (x,z,y)
triples with pseudo-perplexity diagnostics. Designed for offline experimentation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table
import typer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reer.trajectory_search import TrajectorySearch, TrajectorySearchConfig

app = typer.Typer(help="REER trajectory synthesis (local-search MVP)")
console = Console()


@app.command()
def synthesize(
    input_file: Path = typer.Argument(..., help="JSON file with a list of records"),
    x_field: str = typer.Option("topic", help="Field name for x/query"),
    y_field: str = typer.Option("content", help="Field name for y/answer"),
    limit: int = typer.Option(10, help="Max items to process"),
    output_jsonl: Path = typer.Option(
        Path("data/reer_triples.jsonl"), help="Output JSONL of (x,z,y)"
    ),
    auto: str = typer.Option("light", help="Budget preset: light|medium|heavy"),
    backend: str = typer.Option("mlx", help="PPL backend: mlx|together"),
    model: str = typer.Option(
        "mlx-community/Llama-3.2-3B-Instruct-4bit", help="Model name for backend"
    ),
):
    """Run local-search REER synthesis on a small corpus."""
    if not input_file.exists():
        console.print(f"[red]Input not found:[/red] {input_file}")
        raise typer.Exit(1)

    with open(input_file) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    # Budget presets
    if auto == "light":
        cfg = TrajectorySearchConfig(max_iters=6, max_candidates_per_segment=3)
    elif auto == "medium":
        cfg = TrajectorySearchConfig(max_iters=10, max_candidates_per_segment=4)
    else:
        cfg = TrajectorySearchConfig(max_iters=14, max_candidates_per_segment=5)

    # Select evaluator backend with early validation
    console.print(f"[cyan]Initializing {backend} backend with model: {model}[/cyan]")

    try:
        if backend == "mlx":
            from tools.ppl_eval import make_mlx_ppl_evaluator

            ppl = make_mlx_ppl_evaluator(model)
            console.print(
                "[green]✓ MLX backend initialized with sliding window support[/green]"
            )
        elif backend == "together":
            from tools.ppl_eval import make_together_dspy_ppl_evaluator

            ppl = make_together_dspy_ppl_evaluator(model)
            console.print(
                "[green]✓ Together backend initialized with loglikelihood support[/green]"
            )
        else:
            raise typer.BadParameter("backend must be one of: mlx, together")
    except RuntimeError as e:
        console.print(f"[red]Backend initialization failed:[/red]\n{e}")
        raise typer.Exit(1)

    search = TrajectorySearch(ppl, cfg)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    table = Table(title="REER Synthesis Progress")
    table.add_column("Idx", style="cyan")
    table.add_column("PPL Final", style="green")
    table.add_column("Segments", style="yellow")

    with open(output_jsonl, "w") as out:
        for i, rec in enumerate(data):
            if count >= limit:
                break
            x = str(rec.get(x_field, "")).strip()
            y = str(rec.get(y_field, "")).strip()
            if not x or not y:
                continue

            result = search.search(x, y)
            out.write(json.dumps(result) + "\n")
            count += 1

            table.add_row(
                str(i), f"{result['ppl_final']:.3f}", str(len(result["z_segments"]))
            )

    console.print(table)
    console.print(f"[green]✓ Wrote {count} triples to[/green] {output_jsonl}")


if __name__ == "__main__":
    app()
