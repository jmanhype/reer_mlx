#!/usr/bin/env python3
"""
Tiny benchmark comparing 'with plan (z)' vs 'no plan' using the repo scorer.

Generates baseline content (no plan) vs content hinted by a short plan, then
scores both with REERCandidateScorer (perplexity disabled for speed) and prints
summary stats. This is illustrative (not a rigorous benchmark).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from core.candidate_scorer import (
    ContentCandidate,
    REERCandidateScorer,
    ScoringConfig,
)
from dspy_program.reasoning_composer import ReasoningComposer

app = typer.Typer(help="REER mini-benchmark")
console = Console()


@dataclass
class Topic:
    topic: str
    audience: str


def _baseline_content(t: Topic) -> str:
    return f"{t.topic} tips for {t.audience}: share one actionable idea and a CTA."


@app.command()
def run():
    topics = [
        Topic("AI productivity", "developers"),
        Topic("Startup growth", "founders"),
        Topic("MLX best practices", "apple silicon users"),
    ]

    composer = ReasoningComposer()
    scorer = REERCandidateScorer(ScoringConfig(use_perplexity=False))

    async def score(text: str) -> float:
        m = await scorer.score_candidate(ContentCandidate("tmp", text))
        return m.overall_score

    rows: list[tuple[str, float, float]] = []
    for t in topics:
        base = _baseline_content(t)
        with_plan = composer.compose(t.topic, t.audience)["content"]
        base_s = asyncio.run(score(base))
        plan_s = asyncio.run(score(with_plan))
        rows.append((t.topic, base_s, plan_s))

    table = Table(title="Plan vs No Plan (Overall Score)")
    table.add_column("Topic", style="cyan")
    table.add_column("Baseline", style="yellow")
    table.add_column("With Plan", style="green")
    for topic, b, p in rows:
        table.add_row(topic, f"{b:.3f}", f"{p:.3f}")

    console.print(table)
    console.print(
        f"Avg baseline={mean(x[1] for x in rows):.3f} | Avg with-plan={mean(x[2] for x in rows):.3f}"
    )


if __name__ == "__main__":
    app()
