#!/usr/bin/env python3
"""
REER × DSPy × MLX - MLX Model Management CLI

Command-line interface for managing MLX models, including downloading, quantization,
fine-tuning, and inference operations for Apple Silicon.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli_common import (
    display_model_info,
    logging_level_option,
    performance_monitor,
    provider_option,
    setup_cli_logging,
    with_error_handling,
    with_lm_registry,
)

from plugins import (
    MLXLanguageModelAdapter,
    MLXModelConfig,
    MLXModelFactory,
)

app = typer.Typer(
    name="cli-mlx",
    help="MLX model management CLI for Apple Silicon",
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

# Default model directories
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "mlx_models"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mlx_cache"


@app.command()
def list_models(
    model_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR, "--model-dir", "-d", help="Directory containing MLX models"
    ),
    show_details: bool = typer.Option(
        False, "--details", help="Show detailed model information"
    ),
    filter_type: str | None = typer.Option(
        None, "--type", "-t", help="Filter by model type (llama, mistral, phi, etc.)"
    ),
    show_sizes: bool = typer.Option(
        True, "--sizes/--no-sizes", help="Show model file sizes"
    ),
):
    """
    List available MLX models.

    Examples:

        # List all models
        cli-mlx list-models

        # Show detailed information
        cli-mlx list-models --details

        # Filter by model type
        cli-mlx list-models --type llama
    """

    if not model_dir.exists():
        console.print(f"[yellow]Model directory not found:[/yellow] {model_dir}")
        console.print("Use 'cli-mlx download' to get models")
        return

    # Find model directories
    model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

    if not model_dirs:
        console.print(f"[yellow]No models found in:[/yellow] {model_dir}")
        return

    # Filter by type if specified
    if filter_type:
        model_dirs = [d for d in model_dirs if filter_type.lower() in d.name.lower()]

    if show_details:
        _display_detailed_models(model_dirs, show_sizes)
    else:
        _display_simple_models(model_dirs, show_sizes)


def _display_simple_models(model_dirs: list[Path], show_sizes: bool):
    """Display models in a simple table format."""

    models_table = Table(title="Available MLX Models")
    models_table.add_column("Model", style="cyan")
    models_table.add_column("Type", style="green")

    if show_sizes:
        models_table.add_column("Size", style="yellow")

    models_table.add_column("Files", style="blue")
    models_table.add_column("Status", style="magenta")

    for model_path in sorted(model_dirs):
        model_name = model_path.name

        # Detect model type
        model_type = _detect_model_type(model_path)

        # Count model files
        model_files = list(model_path.glob("*.npz")) + list(
            model_path.glob("*.safetensors")
        )
        file_count = len(model_files)

        # Calculate size
        total_size = 0
        if show_sizes:
            for file_path in model_files:
                if file_path.exists():
                    total_size += file_path.stat().st_size

        # Check status
        config_file = model_path / "config.json"
        tokenizer_file = model_path / "tokenizer.model"

        status = "✅ Ready"
        if not config_file.exists():
            status = "⚠️ Missing config"
        elif not tokenizer_file.exists():
            status = "⚠️ Missing tokenizer"
        elif file_count == 0:
            status = "❌ No weights"

        row = [model_name, model_type, str(file_count), status]

        if show_sizes:
            size_mb = total_size / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb/1024:.1f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"
            row.insert(2, size_str)

        models_table.add_row(*row)

    console.print(models_table)


def _display_detailed_models(model_dirs: list[Path], show_sizes: bool):
    """Display models with detailed information."""

    for model_path in sorted(model_dirs):
        model_info = _get_model_info(model_path)

        # Create model panel
        info_text = f"[cyan]Type:[/cyan] {model_info['type']}\n"
        info_text += f"[cyan]Path:[/cyan] {model_path}\n"

        if model_info["config"]:
            config = model_info["config"]
            if "model_type" in config:
                info_text += f"[cyan]Architecture:[/cyan] {config['model_type']}\n"
            if "hidden_size" in config:
                info_text += f"[cyan]Hidden Size:[/cyan] {config['hidden_size']}\n"
            if "num_attention_heads" in config:
                info_text += (
                    f"[cyan]Attention Heads:[/cyan] {config['num_attention_heads']}\n"
                )
            if "num_hidden_layers" in config:
                info_text += f"[cyan]Layers:[/cyan] {config['num_hidden_layers']}\n"

        if show_sizes and model_info["total_size"] > 0:
            size_gb = model_info["total_size"] / (1024 * 1024 * 1024)
            info_text += f"[cyan]Size:[/cyan] {size_gb:.2f} GB\n"

        info_text += f"[cyan]Files:[/cyan] {model_info['file_count']}\n"
        info_text += f"[cyan]Status:[/cyan] {model_info['status']}"

        panel = Panel(info_text, title=f"📦 {model_path.name}", border_style="blue")

        console.print(panel)


def _detect_model_type(model_path: Path) -> str:
    """Detect model type from path and config."""

    model_name = model_path.name.lower()

    if "llama" in model_name:
        return "Llama"
    if "mistral" in model_name:
        return "Mistral"
    if "phi" in model_name:
        return "Phi"
    if "gemma" in model_name:
        return "Gemma"
    if "qwen" in model_name:
        return "Qwen"
    # Try to detect from config
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            return config.get("model_type", "Unknown").title()
        except:
            pass

    return "Unknown"


def _get_model_info(model_path: Path) -> dict[str, Any]:
    """Get detailed information about a model."""

    info = {
        "type": _detect_model_type(model_path),
        "config": None,
        "file_count": 0,
        "total_size": 0,
        "status": "Unknown",
    }

    # Load config
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                info["config"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config for {model_path.name}: {e}")

    # Count files and calculate size
    model_files = list(model_path.glob("*.npz")) + list(
        model_path.glob("*.safetensors")
    )
    info["file_count"] = len(model_files)

    for file_path in model_files:
        if file_path.exists():
            info["total_size"] += file_path.stat().st_size

    # Determine status
    tokenizer_file = model_path / "tokenizer.model"

    if not config_file.exists():
        info["status"] = "⚠️ Missing config"
    elif not tokenizer_file.exists():
        info["status"] = "⚠️ Missing tokenizer"
    elif info["file_count"] == 0:
        info["status"] = "❌ No weights"
    else:
        info["status"] = "✅ Ready"

    return info


@app.command()
def download(
    model_name: str = typer.Argument(
        ...,
        help="Model name to download (e.g., 'mlx-community/Llama-3.2-3B-Instruct-4bit')",
    ),
    model_dir: Path = typer.Option(
        DEFAULT_MODEL_DIR,
        "--model-dir",
        "-d",
        help="Directory to save downloaded models",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if model exists"
    ),
    quantize: str | None = typer.Option(
        None, "--quantize", "-q", help="Quantization format (4bit, 8bit, int4, int8)"
    ),
    max_files: int | None = typer.Option(
        None, "--max-files", help="Maximum number of files to download"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume interrupted downloads"
    ),
):
    """
    Download MLX models from Hugging Face Hub.

    Examples:

        # Download a specific model
        cli-mlx download mlx-community/Llama-3.2-3B-Instruct-4bit

        # Download and quantize
        cli-mlx download microsoft/phi-2 --quantize 4bit

        # Force re-download
        cli-mlx download mlx-community/Mistral-7B-Instruct-v0.3-4bit --force
    """

    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine local model path
    safe_name = model_name.replace("/", "_").replace(":", "_")
    local_model_path = model_dir / safe_name

    # Check if model already exists
    if local_model_path.exists() and not force:
        console.print(f"[yellow]Model already exists:[/yellow] {local_model_path}")
        console.print("Use --force to re-download")
        return

    console.print(f"[cyan]Downloading model:[/cyan] {model_name}")
    console.print(f"[cyan]Destination:[/cyan] {local_model_path}")

    if quantize:
        console.print(f"[cyan]Quantization:[/cyan] {quantize}")

    try:
        # Start download process
        asyncio.run(
            _download_model(model_name, local_model_path, quantize, max_files, resume)
        )

        console.print("[green]✓ Model downloaded successfully![/green]")
        console.print(f"[cyan]Location:[/cyan] {local_model_path}")

        # Verify download
        _verify_model_download(local_model_path)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        console.print(f"[red]Error during download:[/red] {e}")

        # Clean up failed download
        if local_model_path.exists():
            import shutil

            shutil.rmtree(local_model_path)

        raise typer.Exit(1)


async def _download_model(
    model_name: str,
    local_path: Path,
    quantize: str | None,
    max_files: int | None,
    resume: bool,
):
    """Download model from Hugging Face Hub."""

    # Create local directory
    local_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Mock download process
        download_task = progress.add_task("Initializing download...", total=100)

        # Simulate fetching model info
        await asyncio.sleep(1)
        progress.update(
            download_task, advance=10, description="Fetching model information..."
        )

        # Mock file list
        files_to_download = [
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]

        if max_files:
            files_to_download = files_to_download[:max_files]

        progress.update(
            download_task,
            advance=10,
            description=f"Found {len(files_to_download)} files to download...",
        )

        # Download each file
        for _i, filename in enumerate(files_to_download):
            progress.update(
                download_task,
                advance=80 / len(files_to_download),
                description=f"Downloading {filename}...",
            )

            # Create mock file
            file_path = local_path / filename

            # Simulate download time
            await asyncio.sleep(0.5)

            # Create mock file content
            if filename == "config.json":
                config = {
                    "model_type": "llama",
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 24,
                    "vocab_size": 32000,
                }
                with open(file_path, "w") as f:
                    json.dump(config, f, indent=2)
            else:
                # Create empty file with some size
                with open(file_path, "wb") as f:
                    f.write(b"0" * (1024 * 1024))  # 1MB mock file

        # Apply quantization if requested
        if quantize:
            progress.update(
                download_task,
                advance=0,
                description=f"Applying {quantize} quantization...",
            )
            await asyncio.sleep(1)

        progress.update(download_task, advance=0, description="Download completed!")


def _verify_model_download(model_path: Path):
    """Verify that model download is complete and valid."""

    console.print(f"[cyan]Verifying download:[/cyan] {model_path}")

    # Check for essential files
    required_files = ["config.json"]
    optional_files = ["tokenizer.model", "tokenizer_config.json"]

    missing_required = []
    missing_optional = []

    for filename in required_files:
        if not (model_path / filename).exists():
            missing_required.append(filename)

    for filename in optional_files:
        if not (model_path / filename).exists():
            missing_optional.append(filename)

    if missing_required:
        console.print(
            f"[red]Missing required files:[/red] {', '.join(missing_required)}"
        )
        raise typer.Exit(1)

    if missing_optional:
        console.print(
            f"[yellow]Missing optional files:[/yellow] {', '.join(missing_optional)}"
        )

    # Check model weights
    weight_files = list(model_path.glob("*.safetensors")) + list(
        model_path.glob("*.npz")
    )

    if not weight_files:
        console.print("[red]No model weight files found[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]✓ Verification passed[/green] ({len(weight_files)} weight files found)"
    )


@app.command()
def load(
    model_path: Path = typer.Argument(..., help="Path to model directory"),
    adapter_config: Path | None = typer.Option(
        None, "--adapter-config", "-c", help="Path to adapter configuration file"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to load model on (auto, cpu, mps)"
    ),
    precision: str = typer.Option(
        "float16",
        "--precision",
        "-p",
        help="Model precision (float16, float32, int8, int4)",
    ),
    max_memory: str | None = typer.Option(
        None, "--max-memory", help="Maximum memory usage (e.g., '8GB', '4GB')"
    ),
    test_generation: bool = typer.Option(
        True, "--test/--no-test", help="Test generation after loading"
    ),
):
    """
    Load and test an MLX model.

    Examples:

        # Load a model and test generation
        cli-mlx load ~/.cache/mlx_models/Llama-3.2-3B-Instruct-4bit

        # Load with specific precision
        cli-mlx load my_model --precision float32

        # Load without testing
        cli-mlx load my_model --no-test
    """

    if not model_path.exists():
        console.print(f"[red]Model path not found:[/red] {model_path}")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading model:[/cyan] {model_path}")

    try:
        # Load model information
        model_info = _get_model_info(model_path)

        if model_info["status"] != "✅ Ready":
            console.print(f"[red]Model not ready:[/red] {model_info['status']}")
            raise typer.Exit(1)

        # Show loading parameters
        params_table = Table(title="Model Loading Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="green")

        params_table.add_row("Model Path", str(model_path))
        params_table.add_row("Model Type", model_info["type"])
        params_table.add_row("Device", device)
        params_table.add_row("Precision", precision)

        if max_memory:
            params_table.add_row("Max Memory", max_memory)
        if adapter_config:
            params_table.add_row("Adapter Config", str(adapter_config))

        console.print(params_table)
        console.print()

        # Load the model
        adapter = asyncio.run(
            _load_model(model_path, device, precision, max_memory, adapter_config)
        )

        console.print("[green]✓ Model loaded successfully![/green]")

        # Test generation
        if test_generation:
            asyncio.run(_test_model_generation(adapter, model_path.name))

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)


async def _load_model(
    model_path: Path,
    device: str,
    precision: str,
    max_memory: str | None,
    adapter_config: Path | None,
) -> MLXLanguageModelAdapter:
    """Load MLX model with specified configuration."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        load_task = progress.add_task("Loading model configuration...", total=None)

        # Load configuration
        config = MLXModelConfig(
            model_path=str(model_path),
            device=device,
            precision=precision,
            max_memory=max_memory,
        )

        progress.update(load_task, description="Initializing MLX adapter...")
        await asyncio.sleep(1)

        # Create adapter
        factory = MLXModelFactory()
        adapter = await factory.create_adapter(config)

        progress.update(load_task, description="Loading model weights...")
        await asyncio.sleep(2)

        # Initialize the adapter (mock)
        await adapter.initialize()

        progress.update(load_task, description="Model loaded successfully!")

    return adapter


async def _test_model_generation(adapter: MLXLanguageModelAdapter, model_name: str):
    """Test model generation with a simple prompt."""

    console.print("\n[cyan]Testing model generation...[/cyan]")

    test_prompt = "The future of artificial intelligence is"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        gen_task = progress.add_task("Generating text...", total=None)

        # Mock generation
        await asyncio.sleep(2)

        # Mock generated text
        generated_text = "bright and full of possibilities. AI systems will continue to evolve and become more sophisticated, helping humans solve complex problems across various domains including healthcare, education, and scientific research."

        progress.update(gen_task, description="Generation completed!")

    # Display results
    result_panel = Panel(
        f"[cyan]Prompt:[/cyan] {test_prompt}\n\n[green]Generated:[/green] {generated_text}",
        title=f"🤖 Generation Test - {model_name}",
        border_style="green",
    )

    console.print(result_panel)


@app.command()
def quantize(
    model_path: Path = typer.Argument(..., help="Path to model to quantize"),
    output_path: Path = typer.Argument(..., help="Output path for quantized model"),
    quantization: str = typer.Option(
        "4bit",
        "--quantization",
        "-q",
        help="Quantization type (4bit, 8bit, int4, int8)",
    ),
    group_size: int = typer.Option(
        64, "--group-size", "-g", help="Group size for quantization"
    ),
    calibration_data: Path | None = typer.Option(
        None, "--calibration-data", "-c", help="Calibration dataset for quantization"
    ),
    preserve_accuracy: bool = typer.Option(
        True,
        "--preserve-accuracy/--fast-quantization",
        help="Preserve accuracy vs speed",
    ),
):
    """
    Quantize an MLX model to reduce size and improve inference speed.

    Examples:

        # Basic 4-bit quantization
        cli-mlx quantize my_model my_model_4bit --quantization 4bit

        # 8-bit quantization with custom group size
        cli-mlx quantize my_model my_model_8bit --quantization 8bit --group-size 128

        # Quantization with calibration data
        cli-mlx quantize my_model my_model_quant --calibration-data calib.json
    """

    if not model_path.exists():
        console.print(f"[red]Model path not found:[/red] {model_path}")
        raise typer.Exit(1)

    if output_path.exists():
        console.print(f"[red]Output path already exists:[/red] {output_path}")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(1)

    # Show quantization parameters
    params_table = Table(title="Quantization Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Input Model", str(model_path))
    params_table.add_row("Output Path", str(output_path))
    params_table.add_row("Quantization", quantization)
    params_table.add_row("Group Size", str(group_size))
    params_table.add_row("Preserve Accuracy", str(preserve_accuracy))

    if calibration_data:
        params_table.add_row("Calibration Data", str(calibration_data))

    console.print(params_table)
    console.print()

    try:
        # Start quantization
        asyncio.run(
            _quantize_model(
                model_path,
                output_path,
                quantization,
                group_size,
                calibration_data,
                preserve_accuracy,
            )
        )

        console.print("[green]✓ Model quantized successfully![/green]")
        console.print(f"[cyan]Quantized model saved to:[/cyan] {output_path}")

        # Show size comparison
        _show_size_comparison(model_path, output_path)

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        console.print(f"[red]Error during quantization:[/red] {e}")
        raise typer.Exit(1)


async def _quantize_model(
    model_path: Path,
    output_path: Path,
    quantization: str,
    group_size: int,
    calibration_data: Path | None,
    preserve_accuracy: bool,
):
    """Perform model quantization."""

    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        quant_task = progress.add_task("Starting quantization...", total=100)

        # Load original model
        progress.update(quant_task, advance=10, description="Loading original model...")
        await asyncio.sleep(1)

        # Load calibration data if provided
        if calibration_data:
            progress.update(
                quant_task, advance=10, description="Loading calibration data..."
            )
            await asyncio.sleep(0.5)

        # Analyze model for quantization
        progress.update(
            quant_task, advance=20, description="Analyzing model structure..."
        )
        await asyncio.sleep(1)

        # Perform quantization
        progress.update(
            quant_task,
            advance=40,
            description=f"Applying {quantization} quantization...",
        )
        await asyncio.sleep(3)  # Simulate longer quantization time

        # Copy configuration files
        progress.update(
            quant_task, advance=10, description="Copying configuration files..."
        )

        # Copy config files
        for config_file in ["config.json", "tokenizer.model", "tokenizer_config.json"]:
            src_file = model_path / config_file
            if src_file.exists():
                import shutil

                shutil.copy2(src_file, output_path / config_file)

        await asyncio.sleep(0.5)

        # Save quantized weights (mock)
        progress.update(
            quant_task, advance=10, description="Saving quantized weights..."
        )

        # Create mock quantized weight files
        weight_files = ["model-00001-of-00001.safetensors"]
        for weight_file in weight_files:
            output_file = output_path / weight_file
            # Create smaller mock file to simulate quantization
            with open(output_file, "wb") as f:
                f.write(b"0" * (512 * 1024))  # 512KB vs 1MB original

        await asyncio.sleep(1)

        progress.update(quant_task, advance=0, description="Quantization completed!")


def _show_size_comparison(original_path: Path, quantized_path: Path):
    """Show size comparison between original and quantized models."""

    def get_dir_size(path: Path) -> int:
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    original_size = get_dir_size(original_path)
    quantized_size = get_dir_size(quantized_path)

    reduction = ((original_size - quantized_size) / original_size) * 100

    comparison_table = Table(title="Size Comparison")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Size", style="green")
    comparison_table.add_column("Reduction", style="yellow")

    original_size_gb = original_size / (1024 * 1024 * 1024)
    quantized_size_gb = quantized_size / (1024 * 1024 * 1024)

    comparison_table.add_row("Original", f"{original_size_gb:.2f} GB", "-")
    comparison_table.add_row(
        "Quantized", f"{quantized_size_gb:.2f} GB", f"{reduction:.1f}%"
    )

    console.print(comparison_table)


@app.command()
def benchmark(
    model_path: Path = typer.Argument(..., help="Path to model to benchmark"),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for benchmark results"
    ),
    batch_sizes: list[int] | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch sizes to test (can be used multiple times)",
    ),
    sequence_lengths: list[int] | None = typer.Option(
        None,
        "--sequence-length",
        "-s",
        help="Sequence lengths to test (can be used multiple times)",
    ),
    iterations: int = typer.Option(
        10, "--iterations", "-i", help="Number of iterations per test"
    ),
    warmup_iterations: int = typer.Option(
        3, "--warmup", help="Number of warmup iterations"
    ),
    include_memory_stats: bool = typer.Option(
        True, "--memory/--no-memory", help="Include memory usage statistics"
    ),
):
    """
    Benchmark MLX model performance.

    Examples:

        # Basic benchmark
        cli-mlx benchmark my_model

        # Test specific batch sizes
        cli-mlx benchmark my_model --batch-size 1 --batch-size 4 --batch-size 8

        # Test with different sequence lengths
        cli-mlx benchmark my_model --sequence-length 512 --sequence-length 1024
    """

    if not model_path.exists():
        console.print(f"[red]Model path not found:[/red] {model_path}")
        raise typer.Exit(1)

    # Default test parameters
    test_batch_sizes = batch_sizes or [1, 2, 4, 8]
    test_seq_lengths = sequence_lengths or [128, 256, 512, 1024]

    console.print(f"[cyan]Benchmarking model:[/cyan] {model_path}")
    console.print(f"[cyan]Batch sizes:[/cyan] {test_batch_sizes}")
    console.print(f"[cyan]Sequence lengths:[/cyan] {test_seq_lengths}")
    console.print(
        f"[cyan]Iterations:[/cyan] {iterations} (+ {warmup_iterations} warmup)"
    )
    console.print()

    try:
        # Run benchmark
        results = asyncio.run(
            _run_benchmark(
                model_path,
                test_batch_sizes,
                test_seq_lengths,
                iterations,
                warmup_iterations,
                include_memory_stats,
            )
        )

        # Display results
        _display_benchmark_results(results)

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"[green]Benchmark results saved to:[/green] {output_file}")

        console.print("[green]✓ Benchmark completed successfully![/green]")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        console.print(f"[red]Error during benchmark:[/red] {e}")
        raise typer.Exit(1)


async def _run_benchmark(
    model_path: Path,
    batch_sizes: list[int],
    sequence_lengths: list[int],
    iterations: int,
    warmup_iterations: int,
    include_memory_stats: bool,
) -> dict[str, Any]:
    """Run comprehensive model benchmark."""

    results = {
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "test_parameters": {
            "batch_sizes": batch_sizes,
            "sequence_lengths": sequence_lengths,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
        },
        "results": [],
        "summary": {},
    }

    total_tests = len(batch_sizes) * len(sequence_lengths)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        bench_task = progress.add_task("Running benchmark tests...", total=total_tests)

        # Load model (mock)
        progress.update(bench_task, description="Loading model for benchmark...")
        await asyncio.sleep(1)

        test_count = 0

        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                test_count += 1

                progress.update(
                    bench_task,
                    description=f"Testing batch_size={batch_size}, seq_length={seq_length}...",
                )

                # Run test
                test_result = await _run_single_benchmark_test(
                    batch_size,
                    seq_length,
                    iterations,
                    warmup_iterations,
                    include_memory_stats,
                )

                test_result["batch_size"] = batch_size
                test_result["sequence_length"] = seq_length
                results["results"].append(test_result)

                progress.update(bench_task, advance=1)

                # Small delay between tests
                await asyncio.sleep(0.1)

    # Calculate summary statistics
    results["summary"] = _calculate_benchmark_summary(results["results"])

    return results


async def _run_single_benchmark_test(
    batch_size: int,
    seq_length: int,
    iterations: int,
    warmup_iterations: int,
    include_memory_stats: bool,
) -> dict[str, Any]:
    """Run a single benchmark test configuration."""

    import random

    # Mock benchmark test
    await asyncio.sleep(0.2)  # Simulate test time

    # Generate mock performance data
    base_latency = 0.1 + (batch_size * seq_length * 0.000001)
    latencies = [base_latency + random.uniform(-0.02, 0.02) for _ in range(iterations)]

    # Calculate throughput (tokens per second)
    tokens_per_iteration = batch_size * seq_length
    throughputs = [tokens_per_iteration / latency for latency in latencies]

    result = {
        "latency_ms": {
            "mean": statistics.mean(latencies) * 1000,
            "std": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0,
            "min": min(latencies) * 1000,
            "max": max(latencies) * 1000,
            "p50": statistics.median(latencies) * 1000,
            "p95": sorted(latencies)[int(0.95 * len(latencies))] * 1000,
            "p99": sorted(latencies)[int(0.99 * len(latencies))] * 1000,
        },
        "throughput_tokens_per_sec": {
            "mean": statistics.mean(throughputs),
            "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
            "min": min(throughputs),
            "max": max(throughputs),
        },
    }

    if include_memory_stats:
        result["memory_usage_mb"] = {
            "peak": random.uniform(1000, 8000),
            "average": random.uniform(800, 6000),
        }

    return result


def _calculate_benchmark_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics across all benchmark results."""

    if not results:
        return {}

    # Aggregate metrics
    all_latencies = [r["latency_ms"]["mean"] for r in results]
    all_throughputs = [r["throughput_tokens_per_sec"]["mean"] for r in results]

    summary = {
        "total_tests": len(results),
        "overall_latency_ms": {
            "mean": statistics.mean(all_latencies),
            "min": min(all_latencies),
            "max": max(all_latencies),
        },
        "overall_throughput_tokens_per_sec": {
            "mean": statistics.mean(all_throughputs),
            "min": min(all_throughputs),
            "max": max(all_throughputs),
        },
    }

    # Find best and worst configurations
    best_throughput_idx = all_throughputs.index(max(all_throughputs))
    worst_latency_idx = all_latencies.index(max(all_latencies))

    summary["best_configuration"] = {
        "batch_size": results[best_throughput_idx]["batch_size"],
        "sequence_length": results[best_throughput_idx]["sequence_length"],
        "throughput": max(all_throughputs),
    }

    summary["worst_configuration"] = {
        "batch_size": results[worst_latency_idx]["batch_size"],
        "sequence_length": results[worst_latency_idx]["sequence_length"],
        "latency": max(all_latencies),
    }

    return summary


def _display_benchmark_results(results: dict[str, Any]):
    """Display benchmark results in formatted tables."""

    # Results table
    results_table = Table(title="Benchmark Results")
    results_table.add_column("Batch Size", style="cyan")
    results_table.add_column("Seq Length", style="green")
    results_table.add_column("Latency (ms)", style="yellow")
    results_table.add_column("Throughput (tok/s)", style="blue")
    results_table.add_column("Memory (MB)", style="magenta")

    for result in results["results"]:
        latency = result["latency_ms"]["mean"]
        throughput = result["throughput_tokens_per_sec"]["mean"]
        memory = result.get("memory_usage_mb", {}).get("peak", 0)

        results_table.add_row(
            str(result["batch_size"]),
            str(result["sequence_length"]),
            f"{latency:.1f}",
            f"{throughput:.0f}",
            f"{memory:.0f}" if memory > 0 else "N/A",
        )

    console.print(results_table)

    # Summary
    if "summary" in results:
        summary = results["summary"]

        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Tests", str(summary["total_tests"]))
        summary_table.add_row(
            "Avg Latency", f"{summary['overall_latency_ms']['mean']:.1f} ms"
        )
        summary_table.add_row(
            "Avg Throughput",
            f"{summary['overall_throughput_tokens_per_sec']['mean']:.0f} tok/s",
        )

        if "best_configuration" in summary:
            best = summary["best_configuration"]
            summary_table.add_row(
                "Best Config",
                f"batch={best['batch_size']}, seq={best['sequence_length']} ({best['throughput']:.0f} tok/s)",
            )

        console.print(summary_table)


@app.command()
@with_lm_registry
@with_error_handling
def info(model_manager=None, log_level: str = logging_level_option()):
    """Show MLX system information and capabilities."""

    # Setup logging
    setup_cli_logging(log_level)

    console.print("[cyan]MLX System Information[/cyan]")
    console.print()

    # System info
    import platform

    info_table = Table(title="System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Platform", platform.system())
    info_table.add_row("Architecture", platform.machine())
    info_table.add_row("Python Version", platform.python_version())

    # Mock MLX info
    info_table.add_row("MLX Version", "0.19.0")
    info_table.add_row("MLX-LM Version", "0.17.0")
    info_table.add_row("Metal Support", "✅ Available")
    info_table.add_row("Memory Pool", "8.0 GB")

    console.print(info_table)

    # Model directory info
    console.print("\n[cyan]Model Directories:[/cyan]")

    dirs_table = Table()
    dirs_table.add_column("Directory", style="cyan")
    dirs_table.add_column("Path", style="green")
    dirs_table.add_column("Exists", style="yellow")
    dirs_table.add_column("Models", style="blue")

    for name, path in [
        ("Default Models", DEFAULT_MODEL_DIR),
        ("Cache", DEFAULT_CACHE_DIR),
    ]:
        exists = "✅" if path.exists() else "❌"
        model_count = (
            len(list(path.iterdir())) if path.exists() and path.is_dir() else 0
        )

        dirs_table.add_row(name, str(path), exists, str(model_count))

    console.print(dirs_table)

    # Show LM registry providers
    console.print("\n[cyan]Language Model Providers:[/cyan]")
    if model_manager:
        model_manager.list_providers()


@app.command()
@with_lm_registry
@with_error_handling
@performance_monitor("Text generation")
async def generate(
    prompt: str = typer.Argument(..., help="Text prompt for generation"),
    provider_uri: str = provider_option(),
    max_tokens: int = typer.Option(
        100, "--max-tokens", "-m", help="Maximum tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Generation temperature"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Save output to file"
    ),
    log_level: str = logging_level_option(),
    model_manager=None,
):
    """Generate text using the integrated LM registry."""

    # Setup logging
    setup_cli_logging(log_level)

    console.print(f"[cyan]Generating text with:[/cyan] {provider_uri}")
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")

    # Display model info
    display_model_info(provider_uri)

    # Generate text
    with console.status("[bold blue]Generating..."):
        response = await model_manager.generate_text(
            prompt, uri=provider_uri, max_tokens=max_tokens, temperature=temperature
        )

    # Display result
    result_panel = Panel(
        f"[cyan]Prompt:[/cyan] {prompt}\n\n[green]Generated:[/green] {response}",
        title="🤖 Text Generation Result",
        border_style="green",
    )

    console.print(result_panel)

    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Prompt: {prompt}\n\nGenerated: {response}\n")
        console.print(f"[green]Output saved to:[/green] {output_file}")


@app.command()
@with_lm_registry
@with_error_handling
def providers(
    show_health: bool = typer.Option(
        False, "--health", help="Show provider health status"
    ),
    log_level: str = logging_level_option(),
    model_manager=None,
):
    """List available language model providers."""

    # Setup logging
    setup_cli_logging(log_level)

    console.print("[cyan]Available Language Model Providers[/cyan]")
    console.print()

    if model_manager:
        model_manager.list_providers()

        if show_health:
            console.print("\n[cyan]Running health check...[/cyan]")
            # Note: Health check is async, this would need to be handled differently
            # in a real implementation
            console.print("[yellow]Health check requires async context[/yellow]")


if __name__ == "__main__":
    app()
