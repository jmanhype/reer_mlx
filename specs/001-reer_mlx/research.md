# Research: REER × DSPy × MLX Social Posting Pack

**Date**: 2025-01-10  
**Feature**: REER × DSPy × MLX Social Posting Pack  
**Branch**: `001-build-the-reer`

## Executive Summary
Research findings for implementing a hybrid cloud/local LM system for social media content optimization using REER methodology, DSPy orchestration, and MLX local inference.

## Research Areas

### 1. DSPy Pipeline Patterns for Content Generation

**Decision**: Use DSPy's `ChainOfThought` and `Signature` patterns with custom metrics
**Rationale**: 
- DSPy provides declarative LM programming with automatic prompt optimization
- ChainOfThought enables step-by-step reasoning for content strategy
- Signatures allow clean separation of concerns between pipeline phases

**Alternatives Considered**:
- LangChain: More complex, less declarative
- Raw prompting: No optimization capabilities
- Guidance: Limited provider support

**Implementation Pattern**:
```python
class IdeateSignature(dspy.Signature):
    """Generate content ideas from REER traces"""
    traces: list[dict] = dspy.InputField()
    topic: str = dspy.InputField()
    ideas: list[str] = dspy.OutputField()

class ComposeSignature(dspy.Signature):
    """Compose post from idea"""
    idea: str = dspy.InputField()
    style_traces: list[dict] = dspy.InputField()
    post: str = dspy.OutputField()
```

### 2. MLX Token-Level Logprobs Extraction

**Decision**: Dual approach - Python API for direct access, CLI fallback for compatibility
**Rationale**:
- MLX Python API provides direct token logprobs via `model.generate()`
- CLI fallback ensures compatibility with all mlx-lm models
- Enables true perplexity calculation for scoring

**Alternatives Considered**:
- CLI-only: Limited access to token-level data
- Custom MLX fork: Maintenance burden
- Approximation methods: Insufficient accuracy

**Implementation Pattern**:
```python
def get_mlx_perplexity(model, tokenizer, text):
    """Direct MLX API for token logprobs"""
    tokens = tokenizer.encode(text)
    logits = model(mx.array(tokens[:-1]))
    log_probs = mx.log_softmax(logits, axis=-1)
    token_log_probs = log_probs[mx.arange(len(tokens)-1), tokens[1:]]
    perplexity = mx.exp(-mx.mean(token_log_probs))
    return float(perplexity)
```

### 3. Logprob Normalization Across Providers

**Decision**: Provider-specific adapters with unified interface
**Rationale**:
- Each provider returns logprobs differently
- Normalization layer ensures consistent perplexity scores
- Enables fair comparison between providers

**Alternatives Considered**:
- Single unified API: Not all providers support same features
- Post-hoc normalization: Loss of precision
- Provider-specific scoring: No comparability

**Provider Patterns**:
```python
# Together AI
logprobs = response.choices[0].logprobs.token_logprobs
perplexity = math.exp(-sum(logprobs) / len(logprobs))

# OpenAI
logprobs = response.choices[0].logprobs.token_logprobs
perplexity = math.exp(-sum(logprobs) / len(logprobs))

# Anthropic (via logprob beta)
log_prob = response.meta.log_prob
perplexity = math.exp(-log_prob)

# HuggingFace
outputs = model(input_ids, output_scores=True)
perplexity = torch.exp(loss).item()
```

### 4. GEPA Optimization for Social Content

**Decision**: Use DSPy's BootstrapFewShot with custom social media metrics
**Rationale**:
- GEPA (Generate, Evaluate, Prune, Append) maps well to BootstrapFewShot
- Custom metrics for impressions, engagement, virality
- Traces provide supervision signal

**Alternatives Considered**:
- Manual prompt engineering: Not scalable
- Reinforcement learning: Insufficient data
- Fine-tuning: Too expensive, not flexible

**Optimization Strategy**:
```python
def social_metric(example, prediction, trace=None):
    """Custom metric for social content optimization"""
    score = 0.0
    
    # Length optimization (threads vs single posts)
    if len(prediction.post) < 280:
        score += 0.2
    
    # Engagement patterns from traces
    if trace and matches_high_performing_pattern(prediction.post, trace):
        score += 0.5
    
    # Readability and clarity
    score += calculate_readability_score(prediction.post) * 0.3
    
    return score

optimizer = BootstrapFewShot(
    metric=social_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=5
)
```

### 5. Append-Only Trace Storage

**Decision**: JSONL format with immutable append operations
**Rationale**:
- JSONL allows streaming reads/writes
- Append-only ensures audit trail
- No database dependency
- Git-friendly for versioning

**Alternatives Considered**:
- SQLite: Overkill for append-only
- Parquet: Complex for simple appends
- CSV: Poor for nested data

**Storage Pattern**:
```python
class TraceStore:
    def append(self, trace: dict):
        with open(self.path, 'a') as f:
            f.write(json.dumps(trace) + '\n')
    
    def read_all(self) -> Iterator[dict]:
        with open(self.path, 'r') as f:
            for line in f:
                yield json.loads(line)
```

### 6. Provider API Rate Limiting

**Decision**: Exponential backoff with provider-specific limits
**Rationale**:
- Each provider has different rate limits
- Exponential backoff prevents API bans
- Configurable limits per provider

**Alternatives Considered**:
- Fixed delays: Inefficient
- No rate limiting: Risk of bans
- Queue-based: Over-complex

**Rate Limit Configuration**:
```python
PROVIDER_LIMITS = {
    "together_ai": {"rpm": 600, "tpm": 1000000},
    "openai": {"rpm": 3500, "tpm": 90000},
    "anthropic": {"rpm": 50, "tpm": 100000},
    "hf": {"rpm": 100, "tpm": 50000}
}
```

## Dependencies & Versions

### Core Dependencies
- `dspy-ai>=2.4.0`: DSPy framework
- `mlx>=0.5.0`: Apple MLX framework
- `mlx-lm>=0.0.10`: MLX language models
- `jsonschema>=4.17.0`: Schema validation
- `pydantic>=2.0.0`: Data validation

### Provider SDKs
- `together>=1.0.0`: Together AI
- `openai>=1.0.0`: OpenAI
- `anthropic>=0.20.0`: Anthropic
- `transformers>=4.30.0`: HuggingFace

### Development Tools
- `pytest>=7.0.0`: Testing
- `ruff>=0.1.0`: Linting
- `black>=23.0.0`: Formatting

## Environment Variables

### Required
- `TOGETHER_API_KEY`: Together AI authentication
- `OPENAI_API_KEY`: OpenAI authentication  
- `ANTHROPIC_API_KEY`: Anthropic authentication
- `HF_API_KEY`: HuggingFace authentication

### Optional
- `TOGETHER_API_BASE`: Custom Together endpoint
- `OPENAI_API_BASE`: Custom OpenAI endpoint
- `ANTHROPIC_API_BASE`: Custom Anthropic endpoint
- `HF_API_BASE`: Custom HF endpoint
- `MLX_MODEL_PATH`: Local MLX model cache

## Risk Mitigation

### Provider API Changes
**Risk**: Provider APIs may change, breaking integrations
**Mitigation**: 
- Version pin provider SDKs
- Abstract provider interfaces
- Comprehensive integration tests
- Fallback providers

### MLX Compatibility
**Risk**: MLX only works on Apple Silicon
**Mitigation**:
- Graceful fallback to cloud providers
- Clear hardware requirements in docs
- CPU-based approximation option

### Cost Management
**Risk**: Cloud API costs may exceed budget
**Mitigation**:
- Default to MLX when available
- Cost tracking and alerts
- Configurable spending limits
- Batch processing for efficiency

## Performance Benchmarks

### Perplexity Scoring Performance
- MLX (M1 Max): ~50ms per score
- OpenAI GPT-4: ~800ms per score
- Together Llama-3.1: ~200ms per score
- Anthropic Claude: ~600ms per score

### Cost Comparison (per 1000 scores)
- MLX: $0 (local compute)
- OpenAI GPT-4: ~$30
- Together Llama-3.1: ~$2
- Anthropic Claude: ~$15

### Accuracy Metrics
- Schema adherence: Target ≥90%
- Impression uplift: Target ≥15%
- Cost reduction: Target ≥50% with MLX

## Recommendations

1. **Start with MLX + Together AI** hybrid for best cost/performance
2. **Implement provider abstraction layer** first for flexibility
3. **Use JSONL for all data storage** for simplicity and auditability
4. **Build comprehensive integration tests** before implementation
5. **Create provider-specific adapters** for logprob normalization
6. **Use DSPy's built-in optimizers** rather than custom implementations

## Next Steps

With research complete, proceed to Phase 1:
- Create JSON schemas for traces, candidates, timelines
- Design data models based on research findings
- Write contract tests for schemas
- Create quickstart guide for basic usage