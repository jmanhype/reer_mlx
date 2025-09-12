#!/usr/bin/env python3
"""
REER × DSPy × MLX - Social Data Collection CLI

Command-line interface for collecting social media data from various platforms.
Supports X (Twitter), Instagram, TikTok, and other social platforms.
"""

import asyncio
from datetime import UTC
import json
from pathlib import Path
import sys

from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


try:  # Optional import; fail gracefully if not installed
    from social.collectors import TWScrapeCollector  # type: ignore

    _HAS_TWSCRAPE = True
except Exception:
    TWScrapeCollector = None  # type: ignore
    _HAS_TWSCRAPE = False

app = typer.Typer(
    name="social-collect",
    help="Data collection CLI for social media platforms",
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
def collect(
    platform: str = typer.Argument(
        ..., help="Platform to collect from (x, instagram, tiktok)"
    ),
    session_db: Path | None = typer.Option(
        None,
        "--session-db",
        help="Path to twscrape session DB (SQLite). Enables logged-in crawling.",
    ),
    lang: str | None = typer.Option("en", "--lang", help="Language filter (e.g., en)"),
    min_likes: int | None = typer.Option(
        None, "--min-likes", help="Minimum likes threshold for search"
    ),
    include_retweets: bool = typer.Option(
        False,
        "--include-retweets/--no-include-retweets",
        help="Include retweets in search",
    ),
    min_likes_per_hour: float | None = typer.Option(
        None,
        "--min-likes-per-hour",
        help="Filter to tweets with at least this likes/hour (recency-adjusted)",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output-dir",
        "-o",
        help="Output directory for collected data",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    hashtags: list[str] | None = typer.Option(
        None,
        "--hashtag",
        "-h",
        help="Hashtags to search for (can be used multiple times)",
    ),
    keywords: list[str] | None = typer.Option(
        None,
        "--keyword",
        "-k",
        help="Keywords to search for (can be used multiple times)",
    ),
    limit: int = typer.Option(
        100, "--limit", "-l", help="Maximum number of posts to collect"
    ),
    days_back: int = typer.Option(
        7, "--days-back", "-d", help="Number of days back to collect data"
    ),
    include_engagement: bool = typer.Option(
        True, "--include-engagement/--no-engagement", help="Include engagement metrics"
    ),
    output_format: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, csv, parquet)"
    ),
    user: str | None = typer.Option(
        None,
        "--user",
        help="Collect a specific user's timeline instead of keyword/hashtag search",
    ),
    include_replies: bool = typer.Option(
        False,
        "--include-replies/--no-include-replies",
        help="Include replies when collecting a user's timeline",
    ),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Enable simple cache/dedupe by tweet id at the output path",
    ),
    sort_by: str = typer.Option(
        "likes_per_hour",
        "--sort-by",
        help=(
            "Sort field for output: likes_per_hour, like_count, retweet_count, "
            "reply_count, quote_count, created_at"
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be collected without actually collecting",
    ),
):
    """
    Collect social media data from specified platform.

    Examples:

        # Collect from X with hashtags
        social-collect x --hashtag "ai" --hashtag "machinelearning" --limit 500

        # Collect with keywords and custom output
        social-collect instagram --keyword "technology" --output-dir ./my_data --format csv

        # Dry run to see what would be collected
        social-collect tiktok --hashtag "viral" --dry-run
    """

    # Validate platform
    supported_platforms = ["x", "twitter", "instagram", "tiktok"]
    if platform.lower() not in supported_platforms:
        console.print(
            f"[red]Error:[/red] Platform '{platform}' not supported. Use one of: {', '.join(supported_platforms)}"
        )
        raise typer.Exit(1)

    # Normalize platform name
    platform_name = "x" if platform.lower() in ["x", "twitter"] else platform.lower()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show collection parameters
    params_table = Table(title=f"Collection Parameters - {platform_name.upper()}")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Platform", platform_name)
    params_table.add_row("Output Directory", str(output_dir))
    params_table.add_row("Limit", str(limit))
    params_table.add_row("Days Back", str(days_back))
    params_table.add_row("Output Format", output_format)
    params_table.add_row("Include Engagement", str(include_engagement))
    if session_db:
        params_table.add_row("Session DB", str(session_db))
    if lang:
        params_table.add_row("Language", lang)
    if min_likes is not None:
        params_table.add_row("Min Likes", str(min_likes))
    params_table.add_row("Include Retweets", str(include_retweets))
    if user:
        params_table.add_row("User", user)
        params_table.add_row("Include Replies", str(include_replies))
    params_table.add_row("Cache", str(cache))
    if min_likes_per_hour is not None:
        params_table.add_row("Min Likes/Hour", str(min_likes_per_hour))
    params_table.add_row("Sort By", sort_by)

    if hashtags:
        params_table.add_row("Hashtags", ", ".join(hashtags))
    if keywords:
        params_table.add_row("Keywords", ", ".join(keywords))

    console.print(params_table)
    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No data will be collected[/yellow]")
        console.print("Collection would proceed with the parameters shown above.")
        return

    # Start collection
    console.print(f"[green]Starting data collection from {platform_name}...[/green]")

    try:
        asyncio.run(
            _collect_data(
                platform_name,
                output_dir,
                hashtags,
                keywords,
                limit,
                days_back,
                include_engagement,
                output_format,
                session_db,
                lang,
                min_likes,
                include_retweets,
                user,
                include_replies,
                cache,
                min_likes_per_hour,
                sort_by,
            )
        )

        console.print("[green]✓ Data collection completed successfully![/green]")
        console.print(f"[cyan]Output saved to:[/cyan] {output_dir}")

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        console.print(f"[red]Error during collection:[/red] {e}")
        raise typer.Exit(1)


async def _collect_data(
    platform: str,
    output_dir: Path,
    hashtags: list[str] | None,
    keywords: list[str] | None,
    limit: int,
    days_back: int,
    include_engagement: bool,
    output_format: str,
    session_db: Path | None,
    lang: str | None,
    min_likes: int | None,
    include_retweets: bool,
    user: str | None,
    include_replies: bool,
    cache: bool,
    min_likes_per_hour: float | None,
    sort_by: str,
):
    """Perform the actual data collection."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Initialize collector based on platform
        if platform == "x":
            if not _HAS_TWSCRAPE:
                progress.add_task(
                    "twscrape not installed. Install with `pip install twscrape`.",
                    total=None,
                )
                await asyncio.sleep(1)
                return
            task = progress.add_task(
                "Collecting X/Twitter data via twscrape...", total=None
            )
        else:
            # For now, we only have X collector implemented
            progress.add_task(f"Platform {platform} not yet implemented", total=None)
            await asyncio.sleep(1)
            return

        # Collect using twscrape
        collected_data: list[dict] = []

        # Build a simple query from hashtags/keywords for the search endpoint.
        def _build_query(
            hashtags: list[str] | None,
            keywords: list[str] | None,
            default_lang: str | None,
        ) -> str:
            parts: list[str] = []
            if keywords:
                for kw in keywords:
                    kw = kw.strip()
                    if not kw:
                        continue
                    if " " in kw:
                        parts.append(f'"{kw}"')
                    else:
                        parts.append(kw)
            if hashtags:
                for tag in hashtags:
                    tag = tag.lstrip("#").strip()
                    if tag:
                        parts.append(f"#{tag}")
            if parts:
                return " ".join(parts)
            # Fallback query to avoid empty search
            return f"lang:{default_lang}" if default_lang else "lang:en"

        query = _build_query(hashtags, keywords, lang)

        collector = TWScrapeCollector(session_db=session_db)  # type: ignore
        # Attempt to ensure sessions are logged in (if configured); ignore failures.
        try:
            await collector.ensure_logged_in()  # type: ignore[attr-defined]
        except Exception:
            pass

        # Stream search results
        from dataclasses import asdict

        # Prepare cache: load existing ids if enabled
        def _compute_output_file() -> Path:
            base = f"{platform}_data"
            if user:
                safe_user = user.replace("/", "_")
                base = f"{platform}_user_{safe_user}"
            return output_dir / f"{base}.{output_format}"

        output_file = _compute_output_file()

        existing_ids: set[str] = set()
        existing_records: list[dict] = []
        if cache and output_file.exists():
            try:
                if output_format == "json":
                    with open(output_file, encoding="utf-8") as f:
                        existing_records = json.load(f)
                    for r in existing_records:
                        rid = str(r.get("id"))
                        if rid:
                            existing_ids.add(rid)
                elif output_format == "csv":
                    import pandas as pd

                    df = pd.read_csv(output_file)
                    if "id" in df.columns:
                        existing_ids = set(df["id"].astype(str).tolist())
                elif output_format == "parquet":
                    import pandas as pd

                    df = pd.read_parquet(output_file)
                    if "id" in df.columns:
                        existing_ids = set(df["id"].astype(str).tolist())
            except Exception:
                # If cache read fails, continue without cache
                existing_ids = set()
                existing_records = []

        def _rec_to_dict(rec) -> dict:
            d = asdict(rec)
            # Convert datetime to ISO string for JSON compatibility
            if isinstance(d.get("created_at"), (str, bytes)):
                pass
            else:
                try:
                    d["created_at"] = rec.created_at.isoformat()
                except Exception:
                    d["created_at"] = str(rec.created_at)
            return d

        idx = 0
        if user:
            async for rec in collector.user_tweets(  # type: ignore[attr-defined]
                username=user,
                limit=limit,
                days_back=days_back,
                include_replies=include_replies,
                include_retweets=include_retweets,
            ):
                idx += 1
                rid = str(rec.id)
                if rid in existing_ids:
                    continue
                if idx % 10 == 0:
                    progress.update(
                        task,
                        description=f"Collected {idx}/{limit} (user='{user}')",
                    )
                collected_data.append(_rec_to_dict(rec))
        else:
            async for rec in collector.search(  # type: ignore[attr-defined]
                query=query,
                limit=limit,
                days_back=days_back,
                lang=lang,
                min_likes=min_likes,
                include_retweets=include_retweets,
            ):
                idx += 1
                rid = str(rec.id)
                if rid in existing_ids:
                    continue
                if idx % 10 == 0:
                    progress.update(
                        task,
                        description=f"Collected {idx}/{limit} (query='{query[:30]}...')",
                    )
                collected_data.append(_rec_to_dict(rec))

        # Save collected data
        # Merge with existing cache (for JSON) when enabled
        if output_format == "json":
            all_records = (
                (existing_records + collected_data)
                if cache and existing_records
                else collected_data
            )

            # Compute likes/hour for ranking/filtering
            from datetime import datetime

            def _parse_dt(s: str) -> datetime:
                try:
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s)
                except Exception:
                    return datetime.now(UTC)

            now = datetime.now(UTC)
            MIN_AGE_HOURS = 0.1  # avoid division blow-ups for very fresh tweets

            for r in all_records:
                try:
                    created_at = r.get("created_at")
                    if isinstance(created_at, str):
                        dt = _parse_dt(created_at)
                    else:
                        dt = now
                    age_h = max((now - dt).total_seconds() / 3600.0, MIN_AGE_HOURS)
                    likes = int(r.get("like_count", 0) or 0)
                    r["likes_per_hour"] = round(likes / age_h, 4)
                except Exception:
                    r["likes_per_hour"] = 0.0

            # Apply optional cutoff
            if min_likes_per_hour is not None:
                all_records = [
                    r
                    for r in all_records
                    if r.get("likes_per_hour", 0.0) >= min_likes_per_hour
                ]

            # Dedupe by id while preserving order (existing first)
            seen: set[str] = set()
            merged: list[dict] = []
            for r in all_records:
                rid = str(r.get("id"))
                if rid and rid not in seen:
                    seen.add(rid)
                    merged.append(r)

            # Sort by selected field
            def _sort_key(rec: dict):
                if sort_by == "created_at":
                    val = rec.get("created_at")
                    try:
                        s = str(val)
                        if s.endswith("Z"):
                            s = s[:-1] + "+00:00"
                        from datetime import datetime

                        return datetime.fromisoformat(s)
                    except Exception:
                        return ""
                return rec.get(sort_by, 0)

            merged.sort(key=_sort_key, reverse=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
        elif output_format == "csv":
            import pandas as pd

            if cache and output_file.exists():
                try:
                    df_existing = pd.read_csv(output_file)
                except Exception:
                    df_existing = None
            else:
                df_existing = None
            df_new = pd.json_normalize(collected_data)
            df = (
                pd.concat([df_existing, df_new], ignore_index=True)
                if df_existing is not None
                else df_new
            )
            if "id" in df.columns:
                df.drop_duplicates(subset=["id"], inplace=True, keep="first")

            # Compute likes_per_hour
            try:
                import numpy as np
            except Exception:
                np = None
            from datetime import datetime

            def to_dt(val):
                try:
                    s = str(val)
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s)
                except Exception:
                    return datetime.now(UTC)

            now = datetime.now(UTC)
            ages_h = (now - df["created_at"].apply(to_dt)).dt.total_seconds() / 3600.0
            ages_h = ages_h.clip(lower=0.1)
            df["likes_per_hour"] = df["like_count"].fillna(0).astype(int) / ages_h

            # Apply cutoff and rank
            if min_likes_per_hour is not None:
                df = df[df["likes_per_hour"] >= float(min_likes_per_hour)]
            if sort_by not in df.columns:
                # Fallback to likes_per_hour if requested column missing
                sort_col = "likes_per_hour"
            else:
                sort_col = sort_by
            df.sort_values(by=sort_col, ascending=False, inplace=True)
            df.to_csv(output_file, index=False)
        elif output_format == "parquet":
            import pandas as pd

            if cache and output_file.exists():
                try:
                    df_existing = pd.read_parquet(output_file)
                except Exception:
                    df_existing = None
            else:
                df_existing = None
            df_new = pd.json_normalize(collected_data)
            df = (
                pd.concat([df_existing, df_new], ignore_index=True)
                if df_existing is not None
                else df_new
            )
            if "id" in df.columns:
                df.drop_duplicates(subset=["id"], inplace=True, keep="first")
            # Compute likes_per_hour
            from datetime import datetime

            def to_dt(val):
                try:
                    s = str(val)
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s)
                except Exception:
                    return datetime.now(UTC)

            now = datetime.now(UTC)
            ages_h = (now - df["created_at"].apply(to_dt)).dt.total_seconds() / 3600.0
            ages_h = ages_h.clip(lower=0.1)
            df["likes_per_hour"] = df["like_count"].fillna(0).astype(int) / ages_h

            if min_likes_per_hour is not None:
                df = df[df["likes_per_hour"] >= float(min_likes_per_hour)]
            if sort_by not in df.columns:
                sort_col = "likes_per_hour"
            else:
                sort_col = sort_by
            df.sort_values(by=sort_col, ascending=False, inplace=True)
            df.to_parquet(output_file, index=False)

        progress.update(
            task,
            description=f"Saved {len(collected_data)} new posts to {output_file}",
        )


@app.command()
def platforms():
    """List supported platforms and their capabilities."""

    platforms_table = Table(title="Supported Platforms")
    platforms_table.add_column("Platform", style="cyan")
    platforms_table.add_column("Status", style="green")
    platforms_table.add_column("Features", style="yellow")

    platforms_table.add_row(
        "X (Twitter)", "✓ Available", "Posts, engagement, hashtags, keywords"
    )
    platforms_table.add_row(
        "Instagram", "🚧 Planned", "Posts, stories, reels, engagement"
    )
    platforms_table.add_row(
        "TikTok", "🚧 Planned", "Videos, trends, hashtags, engagement"
    )

    console.print(platforms_table)


@app.command()
def status(
    data_dir: Path = typer.Option(
        Path("data/raw"), "--data-dir", "-d", help="Data directory to check"
    ),
):
    """Show status of collected data."""

    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        return

    # Find data files
    data_files = list(data_dir.glob("*_data.*"))

    if not data_files:
        console.print("[yellow]No collected data found[/yellow]")
        return

    status_table = Table(title="Collected Data Status")
    status_table.add_column("File", style="cyan")
    status_table.add_column("Size", style="green")
    status_table.add_column("Modified", style="yellow")

    for file_path in sorted(data_files):
        stats = file_path.stat()
        size_mb = stats.st_size / (1024 * 1024)

        status_table.add_row(file_path.name, f"{size_mb:.2f} MB", f"{stats.st_mtime}")

    console.print(status_table)


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Data file to validate"),
    schema_file: Path | None = typer.Option(
        None, "--schema", "-s", help="Schema file for validation"
    ),
):
    """Validate collected data format and quality."""

    if not input_file.exists():
        console.print(f"[red]Input file not found:[/red] {input_file}")
        raise typer.Exit(1)

    console.print(f"[cyan]Validating:[/cyan] {input_file}")

    try:
        # Load and validate data
        if input_file.suffix == ".json":
            with open(input_file) as f:
                data = json.load(f)
        elif input_file.suffix == ".csv":
            import pandas as pd

            data = pd.read_csv(input_file).to_dict("records")
        else:
            console.print(f"[red]Unsupported file format:[/red] {input_file.suffix}")
            raise typer.Exit(1)

        # Basic validation
        if isinstance(data, list) and len(data) > 0:
            console.print(f"[green]✓ Valid data file with {len(data)} records[/green]")

            # Show sample record
            if len(data) > 0:
                console.print("\n[cyan]Sample record:[/cyan]")
                rprint(data[0])
        else:
            console.print("[red]✗ Invalid or empty data file[/red]")

    except Exception as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
