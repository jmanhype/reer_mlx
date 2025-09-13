"""REER configuration dataclass and defaults.

Provides tuning knobs for REER iterative refinement, aligned with
the official REER synthesis procedure (stop threshold, steps, expansions,
and model used for conditional perplexity).
"""

from dataclasses import dataclass


@dataclass
class REERConfig:
    stop_thresh: float = 0.25
    max_steps: int = 5
    num_expansion: int = 2
    ppl_model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
