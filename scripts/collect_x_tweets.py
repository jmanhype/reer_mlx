#!/usr/bin/env python3
"""
Collect X (Twitter) tweets using twscrape and save to JSONL/CSV.

Usage examples:
  python scripts/collect_x_tweets.py search --query "ai productivity" --limit 300 --min-likes 200 \
      --output data/raw/x_ai.jsonl

  python scripts/collect_x_tweets.py user --username elonmusk --limit 200 \
      --output data/raw/x_elon.jsonl

Requires: `pip install twscrape`
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path

import typer

from social.collectors.twscrape_collector import (
    TWScrapeCollector,
    TWScrapeNotInstalledError,
)

app = typer.Typer(name="collect-x-tweets", help="Collect X tweets via twscrape")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.command()
def search(
    query: str = typer.Option(..., help="Search query (X advanced syntax supported)"),
    limit: int = typer.Option(300, help="Max tweets to fetch"),
    days_back: int = typer.Option(7, help="Restrict to last N days"),
    lang: str | None = typer.Option("en", help="Language filter (e.g., en)"),
    min_likes: int | None = typer.Option(None, help="Minimum likes threshold"),
    include_retweets: bool = typer.Option(False, help="Include retweets"),
    session_db: Path | None = typer.Option(
        None, help="Path to twscrape session DB (SQLite)"
    ),
    output: Path = typer.Option(..., help="Output file (.jsonl or .csv)"),
):
    """Search tweets and save to disk."""

    async def _run() -> None:
        collector = TWScrapeCollector(session_db=session_db)
        await collector.ensure_logged_in()
        records: list[dict] = []
        async for rec in collector.search(
            query=query,
            limit=limit,
            days_back=days_back,
            lang=lang,
            min_likes=min_likes,
            include_retweets=include_retweets,
        ):
            records.append(asdict(rec))

        if output.suffix.lower() == ".jsonl":
            _write_jsonl(output, records)
        elif output.suffix.lower() == ".csv":
            try:
                import pandas as pd  # type: ignore
            except Exception as e:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "pandas required for CSV output: pip install pandas"
                ) from e
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output, index=False)
        else:
            raise ValueError("Output must end with .jsonl or .csv")

        typer.echo(f"Saved {len(records)} tweets to {output}")

    try:
        asyncio.run(_run())
    except TWScrapeNotInstalledError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)


@app.command()
def user(
    username: str = typer.Option(..., help="X handle without @"),
    limit: int = typer.Option(300, help="Max tweets to fetch"),
    days_back: int | None = typer.Option(30, help="Restrict to last N days"),
    include_replies: bool = typer.Option(False, help="Include replies"),
    include_retweets: bool = typer.Option(False, help="Include retweets"),
    session_db: Path | None = typer.Option(
        None, help="Path to twscrape session DB (SQLite)"
    ),
    output: Path = typer.Option(..., help="Output file (.jsonl or .csv)"),
):
    """Collect tweets from a user's timeline and save to disk."""

    async def _run() -> None:
        collector = TWScrapeCollector(session_db=session_db)
        await collector.ensure_logged_in()
        records: list[dict] = []
        async for rec in collector.user_tweets(
            username=username,
            limit=limit,
            days_back=days_back,
            include_replies=include_replies,
            include_retweets=include_retweets,
        ):
            records.append(asdict(rec))

        if output.suffix.lower() == ".jsonl":
            _write_jsonl(output, records)
        elif output.suffix.lower() == ".csv":
            try:
                import pandas as pd  # type: ignore
            except Exception as e:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "pandas required for CSV output: pip install pandas"
                ) from e
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output, index=False)
        else:
            raise ValueError("Output must end with .jsonl or .csv")

        typer.echo(f"Saved {len(records)} tweets to {output}")

    try:
        asyncio.run(_run())
    except TWScrapeNotInstalledError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
