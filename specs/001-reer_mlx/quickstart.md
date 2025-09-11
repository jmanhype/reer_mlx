# Quickstart: REER × DSPy × MLX Social Posting Pack

**Time to first post**: ~10 minutes  
**Prerequisites**: Python 3.11+, X (Twitter) analytics export

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/reer-dspy-mlx-pack.git
cd reer-dspy-mlx-pack

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Required API Keys

Add to `.env`:
```bash
# At least one provider required
OPENAI_API_KEY=sk-...           # For OpenAI
TOGETHER_API_KEY=...             # For Together AI
ANTHROPIC_API_KEY=sk-ant-...    # For Anthropic
HF_API_KEY=hf_...               # For HuggingFace
```

## Quick Test: Generate Your First Post

### 1. Import Your X Analytics (Optional but Recommended)

```bash
# Download your X analytics CSV from analytics.twitter.com
# Place in examples/ directory

# Normalize to JSONL format
python scripts/social_collect.py \
  --input examples/x_analytics.csv \
  --output data/social/normalized.jsonl
```

### 2. Extract Strategies with REER

```bash
# Mine strategies from your top posts (or use demo data)
python scripts/social_reer.py \
  --input data/social/normalized.jsonl \
  --output data/traces/traces.jsonl \
  --lm dspy::openai/gpt-4o-mini  # or mlx::mistral-7b for local
```

### 3. Generate Optimized Content

```bash
# Generate a new post using extracted strategies
python scripts/social_run.py \
  --topic "AI engineering" \
  --traces data/traces/traces.jsonl \
  --lm dspy::openai/gpt-4o-mini
```

Expected output:
```
🎯 Generated Post:
─────────────────
Just shipped a REER × DSPy pipeline that learns from your best tweets.

The twist? It runs on MLX locally for 50% cost reduction vs cloud-only.

Here's what 1000 posts taught me about AI content optimization 🧵

[1/5]
─────────────────
Score: 0.82 | Provider: dspy::openai/gpt-4o-mini
```

## Provider Switching

### Use Cloud Providers (Fast, Requires API Keys)

```bash
# OpenAI
python scripts/social_run.py --topic "your topic" --lm dspy::openai/gpt-4o-mini

# Together AI (Cheaper)
python scripts/social_run.py --topic "your topic" --lm dspy::together_ai/meta-llama-3.1-8b-instruct

# Anthropic
python scripts/social_run.py --topic "your topic" --lm dspy::anthropic/claude-3-haiku
```

### Use Local MLX (Apple Silicon Only, Free)

```bash
# First, download a model
python -m mlx_lm.convert --hf-model mistralai/Mistral-7B-Instruct-v0.2 -q

# Run with MLX
python scripts/social_run.py --topic "your topic" --lm mlx::mistral-7b
```

## Complete Workflow Example

```bash
# Step 1: Collect your X data
python scripts/social_collect.py \
  --input examples/your_x_export.csv \
  --output data/social/normalized.jsonl

# Step 2: Extract strategies from top 100 posts
python scripts/social_reer.py \
  --input data/social/normalized.jsonl \
  --output data/traces/traces.jsonl \
  --top-k 100 \
  --lm dspy::together_ai/meta-llama-3.1-8b-instruct

# Step 3: Tune DSPy pipeline with GEPA
python scripts/social_gepa.py \
  --traces data/traces/traces.jsonl \
  --output models/tuned_pipeline.pkl \
  --iterations 10

# Step 4: Generate optimized content
python scripts/social_run.py \
  --topic "product launch" \
  --model models/tuned_pipeline.pkl \
  --traces data/traces/traces.jsonl \
  --lm mlx::mistral-7b \
  --num-drafts 5

# Step 5: Evaluate performance
python scripts/social_eval.py \
  --predictions output/predictions.jsonl \
  --metrics output/metrics.json
```

## Verify Installation

Run the test suite:
```bash
# Schema validation
pytest tests/contract/test_schema_validation.py -v

# Integration test (requires API key)
pytest tests/integration/test_pipeline.py -v

# Full test suite
pytest tests/ -v
```

Expected output:
```
tests/contract/test_schema_validation.py::test_trace_schema PASSED
tests/contract/test_schema_validation.py::test_candidate_schema PASSED
tests/contract/test_schema_validation.py::test_timeline_schema PASSED
tests/integration/test_pipeline.py::test_end_to_end_flow PASSED
========================= 4 passed in 2.34s =========================
```

## Common Issues

### "No API key found"
- Ensure `.env` file exists with at least one provider key
- Source the environment: `source .env`

### "MLX not available"
- MLX requires Apple Silicon (M1/M2/M3)
- Fallback to cloud providers: use `dspy::` prefix instead

### "Rate limit exceeded"
- Add delays: `--delay 1.0` (seconds between requests)
- Use different provider or wait

### "Low quality output"
- Ensure you have sufficient traces (>50 recommended)
- Try different provider or model
- Run GEPA tuning for optimization

## Next Steps

1. **Import more data**: The more historical data, the better strategies
2. **Tune with GEPA**: Run `social_gepa.py` for 10-20 iterations
3. **Experiment with providers**: Compare quality and cost
4. **Schedule posts**: Integrate with your posting workflow
5. **Track performance**: Use `social_eval.py` to measure uplift

## Support

- Documentation: `/docs/`
- Examples: `/examples/`
- Issues: GitHub Issues
- Constitution: `/memory/constitution.md`