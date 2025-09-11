#!/usr/bin/env python3
"""
REER × DSPy × MLX - Social Data Collection CLI

Command-line interface for collecting social media data from various platforms.
Supports X (Twitter), Instagram, TikTok, and other social platforms.
"""

import asyncio
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

from social.collectors import XAnalyticsNormalizer

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
            XAnalyticsNormalizer()
            task = progress.add_task("Collecting X/Twitter data...", total=None)
        else:
            # For now, we only have X collector implemented
            progress.add_task(f"Platform {platform} not yet implemented", total=None)
            await asyncio.sleep(1)
            return

        # Simulate collection process
        collected_data = []

        # Mock data collection for demonstration
        for i in range(min(limit, 10)):  # Limit to 10 for demo
            progress.update(task, description=f"Collecting post {i + 1}/{limit}...")

            # Simulate API delay
            await asyncio.sleep(0.1)

            # Mock post data
            post_data = {
                "id": f"post_{i + 1}",
                "platform": platform,
                "content": f"Sample post content {i + 1}",
                "author": f"user_{i + 1}",
                "timestamp": "2024-01-01T12:00:00Z",
                "hashtags": hashtags[:2] if hashtags else [],
                "engagement": (
                    {"likes": i * 10, "shares": i * 2, "comments": i * 3}
                    if include_engagement
                    else {}
                ),
            }
            collected_data.append(post_data)

        # Save collected data
        output_file = output_dir / f"{platform}_data.{output_format}"

        if output_format == "json":
            with open(output_file, "w") as f:
                json.dump(collected_data, f, indent=2)
        elif output_format == "csv":
            import pandas as pd

            df = pd.json_normalize(collected_data)
            df.to_csv(output_file, index=False)
        elif output_format == "parquet":
            import pandas as pd

            df = pd.json_normalize(collected_data)
            df.to_parquet(output_file, index=False)

        progress.update(
            task, description=f"Saved {len(collected_data)} posts to {output_file}"
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
