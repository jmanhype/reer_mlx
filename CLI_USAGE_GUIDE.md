# REER × DSPy × MLX - CLI Usage Guide

This guide covers all the command-line interfaces (CLIs) available in the REER × DSPy × MLX Social Posting Pack.

## Overview of CLI Tools

The project provides six main CLI scripts:

1. **`social_collect.py`** - Data collection from social media platforms
2. **`social_reer.py`** - REER mining operations on collected data
3. **`social_gepa.py`** - GEPA tuning for DSPy program optimization
4. **`social_run.py`** - Complete pipeline execution orchestrator
5. **`social_eval.py`** - Content evaluation and benchmarking
6. **`cli_mlx.py`** - MLX model management for Apple Silicon

## 1. Social Data Collection (`social_collect.py`)

### Purpose
Collect social media data from various platforms (X/Twitter, Instagram, TikTok).

### Basic Usage
```bash
# Show supported platforms
python scripts/social_collect.py platforms

# Basic data collection from X
python scripts/social_collect.py collect x --hashtag "ai" --limit 100

# Collect with multiple hashtags and keywords
python scripts/social_collect.py collect x \
  --hashtag "ai" --hashtag "machinelearning" \
  --keyword "artificial intelligence" \
  --limit 500 \
  --output-dir data/raw \
  --format json

# Dry run to see what would be collected
python scripts/social_collect.py collect x --hashtag "viral" --dry-run
```

### Advanced Features
```bash
# Collect with engagement metrics
python scripts/social_collect.py collect x \
  --hashtag "tech" \
  --include-engagement \
  --days-back 14

# Export to different formats
python scripts/social_collect.py collect x \
  --hashtag "ai" \
  --format csv  # or parquet

# Validate collected data
python scripts/social_collect.py validate data/raw/x_data.json

# Check collection status
python scripts/social_collect.py status --data-dir data/raw
```

## 2. REER Mining (`social_reer.py`)

### Purpose
Process collected social media data using REER (Retrieve-Extract-Execute-Refine) methodology.

### Basic Usage
```bash
# Full REER mining pipeline
python scripts/social_reer.py mine data/raw/x_data.json

# Mining with custom parameters
python scripts/social_reer.py mine data/raw/x_data.json \
  --output-dir data/processed \
  --batch-size 200 \
  --min-engagement 50 \
  --workers 8
```

### Specific Operations
```bash
# Extract patterns only
python scripts/social_reer.py mine data/raw/x_data.json \
  --algorithm extract \
  --no-synthesize \
  --no-score

# Platform-specific mining
python scripts/social_reer.py mine data/raw/combined_data.json \
  --platform x \
  --platform instagram

# Parallel processing for large datasets
python scripts/social_reer.py mine large_dataset.json \
  --workers 8 \
  --batch-size 500

# Analyze mining results
python scripts/social_reer.py analyze data/processed \
  --include-charts
```

### Trace Management
```bash
# List trace records
python scripts/social_reer.py traces --action list

# Search traces
python scripts/social_reer.py traces --action search --trace-id specific_id

# Export traces
python scripts/social_reer.py traces --action export
```

## 3. GEPA Tuning (`social_gepa.py`)

### Purpose
Reflective prompt evolution using DSPy GEPA (no GA). GEPA edits predictor instructions guided by our scorer.

### Basic Usage
```bash
# Trainset: JSON list of {"topic": ..., "audience": ...}
python scripts/social_gepa.py tune data/train_tasks.json \
  --gen-model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --reflection-model gpt-4o \
  --auto light \
  --output-dir models/gepa
```

### Options
- `--gen-model`: LM used for composing posts (via `dspy.settings`).
- `--reflection-model`: LM used by GEPA reflection.
- `--auto`: budget preset `light|medium|heavy` (or use `--max-full-evals` / `--max-metric-calls`).
- `--cot/--no-cot`: enable Chain-of-Thought.
- `--perplexity/--no-perplexity`: enable perplexity in scoring (slower).

Output: `optimized_program.json` with predictor instructions and optional GEPA stats.

## 4. Pipeline Execution (`social_run.py`)

### Purpose
Orchestrate complete end-to-end social media content generation pipelines.

### Basic Usage
```bash
# Run complete pipeline
python scripts/social_run.py pipeline config.json

# Run specific stages
python scripts/social_run.py pipeline config.json \
  --stage collect \
  --stage mine

# Target specific platforms
python scripts/social_run.py pipeline config.json \
  --platform x \
  --platform instagram \
  --content-count 20
```

### Pipeline Management
```bash
# Dry run to see execution plan
python scripts/social_run.py pipeline config.json --dry-run

# Parallel execution where possible
python scripts/social_run.py pipeline config.json \
  --parallel \
  --save-intermediates

# Skip cached results
python scripts/social_run.py pipeline config.json \
  --skip-cache

# Check pipeline status
python scripts/social_run.py status --output-dir output

# Clean pipeline output
python scripts/social_run.py clean --output-dir output --yes
```

### Example Configuration File
```json
{
  "platforms": ["x", "instagram"],
  "collection": {
    "limit": 100,
    "days_back": 7,
    "hashtags": ["ai", "machinelearning"],
    "keywords": ["artificial intelligence"]
  },
  "mining": {
    "algorithm": "reer",
    "batch_size": 50,
    "extract_patterns": true,
    "synthesize_strategies": true
  },
  "tuning": {
    "engine": "dspy-gepa",
    "auto": "light",
    "reflection_model": "gpt-4o",
    "gen_model": "mlx-community/Llama-3.2-3B-Instruct-4bit"
  },
  "generation": {
    "content_count": 20,
    "optimization_level": "high"
  }
}
```

## 5. Content Evaluation (`social_eval.py`)

### Purpose
Evaluate social media content using multiple metrics and evaluators.

### Basic Usage
```bash
# Basic content evaluation
python scripts/social_eval.py evaluate generated_content.json

# Compare against reference content
python scripts/social_eval.py evaluate new_content.json \
  --reference baseline_content.json

# Platform-specific evaluation
python scripts/social_eval.py evaluate content.json \
  --platform x \
  --platform instagram
```

### Advanced Evaluation
```bash
# Use specific evaluators and metrics
python scripts/social_eval.py evaluate content.json \
  --evaluator kpi \
  --evaluator heuristic \
  --metric engagement \
  --metric readability

# Include human evaluation
python scripts/social_eval.py evaluate content.json \
  --human-eval

# Generate detailed report
python scripts/social_eval.py evaluate content.json \
  --output results.json \
  --report
```

### Benchmarking
```bash
# Benchmark multiple models
python scripts/social_eval.py benchmark \
  models/ \
  test_data.json \
  --iterations 5 \
  --baseline baseline_model

# Compare evaluation results
python scripts/social_eval.py compare \
  eval_results_1.json \
  eval_results_2.json \
  --threshold 0.05
```

## 6. MLX Model Management (`cli_mlx.py`)

### Purpose
Manage MLX models for Apple Silicon, including downloading, quantization, and inference.

### Model Discovery
```bash
# Show system information
python scripts/cli_mlx.py info

# List available models
python scripts/cli_mlx.py list-models

# List with detailed information
python scripts/cli_mlx.py list-models --details --sizes

# Filter by model type
python scripts/cli_mlx.py list-models --type llama
```

### Model Download
```bash
# Download a model
python scripts/cli_mlx.py download mlx-community/Llama-3.2-3B-Instruct-4bit

# Download with quantization
python scripts/cli_mlx.py download microsoft/phi-2 --quantize 4bit

# Force re-download
python scripts/cli_mlx.py download model_name --force

# Limit download files
python scripts/cli_mlx.py download model_name --max-files 5
```

### Model Operations
```bash
# Load and test a model
python scripts/cli_mlx.py load ~/.cache/mlx_models/Llama-3.2-3B-Instruct-4bit

# Load with specific settings
python scripts/cli_mlx.py load my_model \
  --precision float32 \
  --device mps \
  --max-memory 8GB

# Quantize a model
python scripts/cli_mlx.py quantize \
  my_model \
  my_model_4bit \
  --quantization 4bit \
  --group-size 128

# Benchmark model performance
python scripts/cli_mlx.py benchmark my_model \
  --batch-size 1 \
  --batch-size 4 \
  --sequence-length 512 \
  --iterations 10
```

## Common Workflow Examples

### 1. Complete Content Generation Workflow
```bash
# 1. Collect data
python scripts/social_collect.py collect x --hashtag "ai" --limit 500

# 2. Mine patterns
python scripts/social_reer.py mine data/raw/x_data.json

# 3. Tune DSPy program
python scripts/social_gepa.py tune my_program.py data/processed/patterns.json

# 4. Generate content
python scripts/social_run.py pipeline config.json --stage generate

# 5. Evaluate results
python scripts/social_eval.py evaluate output/generated_content.json
```

### 2. Model Optimization Workflow
```bash
# 1. Download base model
python scripts/cli_mlx.py download mlx-community/Llama-3.2-3B-Instruct

# 2. Quantize for efficiency
python scripts/cli_mlx.py quantize \
  ~/.cache/mlx_models/Llama-3.2-3B-Instruct \
  ~/.cache/mlx_models/Llama-3.2-3B-Instruct-4bit \
  --quantization 4bit

# 3. Benchmark performance
python scripts/cli_mlx.py benchmark \
  ~/.cache/mlx_models/Llama-3.2-3B-Instruct-4bit

# 4. Use in GEPA tuning
python scripts/social_gepa.py tune program.py data.json --model-path ~/.cache/mlx_models/Llama-3.2-3B-Instruct-4bit
```

### 3. Evaluation and Comparison Workflow
```bash
# 1. Generate content with different models
python scripts/social_run.py pipeline config_model_a.json --output-dir results_a
python scripts/social_run.py pipeline config_model_b.json --output-dir results_b

# 2. Evaluate both
python scripts/social_eval.py evaluate results_a/content.json --output eval_a.json
python scripts/social_eval.py evaluate results_b/content.json --output eval_b.json

# 3. Compare results
python scripts/social_eval.py compare eval_a.json eval_b.json

# 4. Benchmark models
python scripts/social_eval.py benchmark models/ test_data.json
```

## Tips and Best Practices

### Performance Optimization
- Use `--parallel` flags where available for faster processing
- Adjust `--batch-size` based on your system memory
- Use `--workers` parameter to control parallelization
- Consider `--skip-cache` for fresh results

### Data Management
- Use consistent output directory structures
- Save intermediate results with `--save-intermediates`
- Validate data quality with validation commands
- Use appropriate file formats (JSON for flexibility, Parquet for large datasets)

### Model Management
- Start with quantized models for faster inference
- Benchmark models before production use
- Use appropriate precision settings for your hardware
- Monitor memory usage with `--memory` flags

### Pipeline Orchestration
- Use dry runs to verify configuration
- Start with smaller datasets for testing
- Use status commands to monitor progress
- Clean up intermediate files regularly

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
2. **Memory Issues**: Reduce batch sizes or use quantized models
3. **Permission Errors**: Check file permissions and directory access
4. **Platform Issues**: Some platforms may require authentication keys

### Debug Options
- Use `--verbose` flags for detailed output
- Check log files in the `logs/` directory
- Use `--dry-run` to test configurations
- Validate data files before processing

### Getting Help
```bash
# Get help for any command
python scripts/[script_name].py --help

# Get help for specific subcommands
python scripts/social_collect.py collect --help
python scripts/cli_mlx.py download --help
```

This guide provides comprehensive coverage of all CLI functionality. Each tool is designed to work independently or as part of larger workflows, giving you flexibility in how you approach social media content generation and optimization.
