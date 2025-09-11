# Quickstart: REER × DSPy × MLX Social Posting Pack

**Time to first post**: ~10 minutes  
**Prerequisites**: Python 3.11+, API keys for at least one LM provider

## Installation

```bash
# Clone repository (adjust URL to your fork/repo)
git clone https://github.com/reer-team/reer-dspy-mlx-social.git
cd reer-dspy-mlx-social

# Install with all dependencies
pip install -e .

# Or install with specific dependency groups
pip install -e .[dev,social,cloud]

# Set up environment
cp .env.example .env
# Edit .env with your API keys (see below)
```

## Verify Installation

```bash
# Test import
python -c "import core, plugins, social; print('✓ All modules imported successfully')"

# Check CLI tools are available
python scripts/social_collect.py --version
python scripts/social_run.py --version
```

## Required API Keys

Edit `.env` with at least one provider (for complete list see `.env.example`):

```bash
# Language Model Providers (choose at least one)
OPENAI_API_KEY="sk-your-openai-api-key"        # For GPT models
ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"  # For Claude models  
TOGETHER_API_KEY="your-together-ai-key"        # For open source models
HF_API_KEY="hf_your-huggingface-token"         # For HuggingFace Hub

# Social Platform APIs (optional for content generation)
TWITTER_API_KEY="your-twitter-api-key"         # For X/Twitter posting
TWITTER_BEARER_TOKEN="your-twitter-bearer-token"

# MLX Configuration (local models, Apple Silicon only)
MLX_MODEL_PATH="/path/to/your/mlx/models"      # Local model storage
MLX_DEFAULT_MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
```

## Quick Test: Generate Your First Post

### 1. Create Data Directories

```bash
# Create required directories
mkdir -p data/social data/traces output models
```

### 2. Import Your X Analytics (Optional)

```bash
# Option A: Use your own X/Twitter analytics
# Download CSV from analytics.twitter.com, place in examples/
python scripts/social_collect.py normalize \
  --input examples/your_x_analytics.csv \
  --output data/social/normalized.jsonl \
  --format twitter-analytics

# Option B: Use demo data (for testing)
python scripts/social_collect.py demo \
  --output data/social/normalized.jsonl \
  --count 50
```

### 3. Extract Strategies with REER

```bash
# Mine strategies from high-performing posts
python scripts/social_reer.py extract \
  --input data/social/normalized.jsonl \
  --output data/traces/traces.jsonl \
  --top-k 100 \
  --lm "dspy::openai/gpt-4o-mini"  # or "mlx::mistral-7b" for local
```

### 4. Generate Optimized Content

```bash
# Generate new post using extracted strategies
python scripts/social_run.py generate \
  --topic "AI engineering trends" \
  --traces data/traces/traces.jsonl \
  --output output/generated_post.json \
  --lm "dspy::openai/gpt-4o-mini"
```

**Expected Output:**
```json
{
  "content": "Just shipped a REER × DSPy pipeline that learns from your best tweets.\n\nThe twist? It runs on MLX locally for 50% cost reduction vs cloud-only.\n\nHere's what 1000 posts taught me about AI content optimization 🧵\n\n[1/5]",
  "metadata": {
    "score": 0.82,
    "provider": "dspy::openai/gpt-4o-mini",
    "strategy_ids": ["high_engagement", "technical_thread"],
    "platform": "twitter",
    "timestamp": "2024-09-11T12:00:00Z"
  }
}
```

## Provider Switching

### Use Cloud Providers (Fast, Requires API Keys)

```bash
# OpenAI (Recommended for quality)
python scripts/social_run.py generate \
  --topic "your topic" \
  --lm "dspy::openai/gpt-4o-mini" \
  --output output/post.json

# Anthropic Claude (Good balance)
python scripts/social_run.py generate \
  --topic "your topic" \
  --lm "dspy::anthropic/claude-3-haiku" \
  --output output/post.json

# Together AI (Cost-effective for open source models)
python scripts/social_run.py generate \
  --topic "your topic" \
  --lm "dspy::together_ai/meta-llama-3.1-8b-instruct" \
  --output output/post.json
```

### Use Local MLX (Apple Silicon Only, Free)

```bash
# Install MLX model conversion tools
pip install mlx-lm

# Download and convert a model (one-time setup)
python -m mlx_lm.convert \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 \
  --mlx-path models/mistral-7b-mlx \
  --quantize

# Run with local MLX model
python scripts/social_run.py generate \
  --topic "your topic" \
  --lm "mlx::mistral-7b" \
  --output output/post.json
```

## Complete Workflow Example

```bash
# Step 1: Setup directories and collect data
mkdir -p data/social data/traces output models
python scripts/social_collect.py normalize \
  --input examples/your_x_analytics.csv \
  --output data/social/normalized.jsonl \
  --format twitter-analytics

# Step 2: Extract strategies from top posts with REER
python scripts/social_reer.py extract \
  --input data/social/normalized.jsonl \
  --output data/traces/traces.jsonl \
  --top-k 100 \
  --min-engagement 10 \
  --lm "dspy::together_ai/meta-llama-3.1-8b-instruct"

# Step 3: Optimize DSPy pipeline with GEPA
python scripts/social_gepa.py train \
  --traces data/traces/traces.jsonl \
  --output models/tuned_pipeline.pkl \
  --iterations 20 \
  --population-size 50 \
  --lm "dspy::openai/gpt-4o-mini"

# Step 4: Generate multiple content candidates
python scripts/social_run.py generate \
  --topic "AI product launch announcement" \
  --model models/tuned_pipeline.pkl \
  --traces data/traces/traces.jsonl \
  --output output/candidates.jsonl \
  --num-candidates 5 \
  --lm "mlx::mistral-7b"

# Step 5: Evaluate and select best candidate
python scripts/social_eval.py rank \
  --input output/candidates.jsonl \
  --output output/ranked_results.json \
  --metrics engagement,readability,sentiment
```

## Verify Installation

Run the test suite to ensure everything is working:

```bash
# Install test dependencies
pip install -e .[dev]

# Run contract/schema tests (no API keys required)
python -m pytest tests/contract/ -v --tb=short

# Run unit tests
python -m pytest tests/unit/ -v --tb=short

# Test core imports and basic functionality
python -c "
from core import REERTraceStore, REERTrajectorySynthesizer
from plugins import get_registry
from social.collectors import XAnalyticsNormalizer
print('✓ All core modules imported successfully')
"
```

**Expected Contract Test Output:**
```bash
tests/contract/test_candidate_schema.py ............ [ 35%]
tests/contract/test_timeline_schema.py ............ [ 71%]
tests/contract/test_trace_schema.py ............ [100%]
=================== 32 passed in 1.24s ===================
```

**Note:** Integration tests requiring API keys are in `tests/integration/` and can be run after configuring your `.env` file.

## Troubleshooting

### Installation Issues

**"Module not found" errors:**
```bash
# Ensure you're in the project directory
cd reer-dspy-mlx-social

# Install in editable mode with all dependencies
pip install -e .[dev,social,cloud]

# Verify installation
python -c "import core; print('✓ Installation successful')"
```

**"No API key found" errors:**
```bash
# Check your .env file exists and has valid keys
ls -la .env
grep -v "^#\|^$" .env  # Show non-empty, non-comment lines

# Test API key validation
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('OpenAI key:', os.getenv('OPENAI_API_KEY', 'Not set')[:20] + '...')
"
```

### Model Provider Issues

**MLX not available:**
- MLX requires Apple Silicon (M1/M2/M3/M4)
- Install MLX dependencies: `pip install mlx mlx-lm`
- Fallback to cloud providers if not on Apple Silicon

**Rate limiting errors:**
```bash
# Add delays between requests
python scripts/social_run.py generate \
  --topic "your topic" \
  --delay 2.0 \
  --lm "dspy::openai/gpt-4o-mini"

# Use different provider with higher limits
python scripts/social_run.py generate \
  --topic "your topic" \
  --lm "dspy::together_ai/meta-llama-3.1-8b-instruct"
```

### Data Quality Issues

**Low content quality:**
- Ensure you have sufficient training traces (>50 recommended)
- Use higher quality language models (GPT-4 over GPT-3.5)
- Run GEPA optimization: `python scripts/social_gepa.py train`

**Import/parsing errors:**
```bash
# Validate your input data format
python tools/schema_check.py --input data/social/normalized.jsonl --schema trace

# Check for common issues
head -5 data/social/normalized.jsonl  # Inspect first few lines
wc -l data/social/normalized.jsonl    # Count total records
```

## Next Steps

### 1. Optimize Your Pipeline
```bash
# Run GEPA optimization for better performance
python scripts/social_gepa.py train \
  --traces data/traces/traces.jsonl \
  --output models/optimized_pipeline.pkl \
  --iterations 30

# A/B test different models
python scripts/social_eval.py compare \
  --models models/tuned_pipeline.pkl,models/optimized_pipeline.pkl \
  --test-topics "AI trends,product launches,tech insights" \
  --output output/model_comparison.json
```

### 2. Scale Content Production
```bash
# Batch generate content for multiple topics
python scripts/social_run.py batch \
  --topics-file examples/content_topics.txt \
  --output-dir output/batch_content/ \
  --lm "dspy::anthropic/claude-3-haiku"

# Schedule generated content
python scripts/social_schedule.py \
  --content-dir output/batch_content/ \
  --platform twitter \
  --schedule-file examples/posting_schedule.yaml
```

### 3. Monitor and Improve
```bash
# Evaluate content performance
python scripts/social_eval.py analyze \
  --posted-content output/posted_content.jsonl \
  --metrics engagement,reach,sentiment \
  --output output/performance_analysis.json

# Update strategies based on new data
python scripts/social_reer.py update \
  --existing-traces data/traces/traces.jsonl \
  --new-data data/social/recent_posts.jsonl \
  --output data/traces/updated_traces.jsonl
```

## Learning Resources

- **Architecture Overview**: `/docs/ARCHITECTURE.md`
- **API Documentation**: `/docs/api/`
- **Example Configurations**: `/examples/`
- **System Constitution**: `/memory/constitution.md`

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and best practices
- **Documentation**: Comprehensive guides in `/docs/`

---

🎉 **You're ready to start!** Begin with the Quick Test above, then explore the complete workflow to optimize your social media content generation.