# REER × DSPy × MLX Social Posting Pack

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Overview
Advanced social media strategy learning system using Reverse-Engineered Reasoning (REER) with DSPy and MLX frameworks for Apple Silicon optimization.

## Features
- **REER Mining**: Extract winning posting strategies from X analytics
- **DSPy Integration**: Structured prompting and LM orchestration
- **MLX Optimization**: Apple Silicon GPU acceleration
- **Provider Routing**: Flexible model switching (mlx:/dspy:/dummy:)
- **Memory Profiling**: Advanced optimization tools

## Installation
```bash
git clone https://github.com/jmanhype/reer_mlx.git
cd reer_mlx
pip install -r requirements.txt
```

## Quick Start
```python
from core import REERPipeline

pipeline = REERPipeline()
results = pipeline.mine_strategies(analytics_data)
```

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [Memory Profiling](docs/memory_profiling.md)
- [API Reference](docs/api/)

## Testing
```bash
pytest tests/
```

## License
MIT
