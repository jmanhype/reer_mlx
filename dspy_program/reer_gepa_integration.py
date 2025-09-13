"""REER + GEPA Integration: Intelligent thinking evolution using DSPy's GEPA.

Instead of random segment variants, use GEPA to evolve thinking (z) that
minimizes conditional log-perplexity log-PPL(y|x,z).
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from dspy import Example
from dspy.teleprompt import GEPA

from core.candidate_scorer import PerplexityCalculator


class ThinkingSignature(dspy.Signature):
    """Generate reasoning that explains an answer given context."""

    context = dspy.InputField(desc="The question or context (x)")
    answer = dspy.OutputField(desc="The target answer (y)")
    thinking = dspy.OutputField(desc="Step-by-step reasoning that leads to the answer")


class ThinkingModule(dspy.Module):
    """DSPy module that generates thinking to minimize perplexity."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.think = (
            dspy.ChainOfThought(ThinkingSignature)
            if use_cot
            else dspy.Predict(ThinkingSignature)
        )

    def forward(self, context: str, answer: str) -> Any:
        return self.think(context=context, answer=answer)


def create_perplexity_metric(ppl_calc: PerplexityCalculator):
    """Create a GEPA metric that minimizes conditional log-perplexity.

    The metric returns:
    - score: 1 / (1 + log_ppl) to convert minimization to maximization
    - feedback: Textual description of perplexity improvement
    """

    async def _calculate_ppl(x: str, z: str, y: str) -> float:
        """Calculate log-PPL(y|x,z)."""
        return await ppl_calc.calculate_conditional_log_perplexity(x, z, y)

    def metric(
        gold: Example,
        pred: Any,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> dict[str, Any]:
        """GEPA metric that evaluates thinking quality via perplexity."""

        # Extract fields
        context = gold.get("context", "")
        answer = gold.get("answer", "")

        # Extract generated thinking
        if isinstance(pred, dict):
            thinking = pred.get("thinking", "")
        else:
            thinking = getattr(pred, "thinking", "")

        # Calculate perplexity - use asyncio.run since GEPA calls this synchronously
        try:
            log_ppl = asyncio.run(_calculate_ppl(context, thinking, answer))
        except RuntimeError:
            # If there's already an event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            log_ppl = loop.run_until_complete(_calculate_ppl(context, thinking, answer))

        # Convert to score (lower PPL is better, but GEPA maximizes)
        # Use 1/(1+log_ppl) to map [0,∞) to (0,1] with lower PPL = higher score
        score = 1.0 / (1.0 + max(0, log_ppl))

        # Generate feedback
        if log_ppl < 0.5:
            feedback = f"Excellent! Very low perplexity ({log_ppl:.3f}). The thinking strongly explains the answer."
        elif log_ppl < 1.0:
            feedback = f"Good perplexity ({log_ppl:.3f}). The thinking explains the answer well."
        elif log_ppl < 2.0:
            feedback = f"Moderate perplexity ({log_ppl:.3f}). The thinking partially explains the answer."
        else:
            feedback = f"High perplexity ({log_ppl:.3f}). The thinking doesn't explain the answer well. Try being more specific and systematic."

        return {"score": score, "feedback": feedback}

    return metric


async def evolve_thinking_with_gepa(
    x: str,
    y: str,
    z_init: str | None = None,
    gen_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    reflection_model: str = "gpt-5",
    max_iterations: int = 10,
) -> dict[str, Any]:
    """Use GEPA to evolve thinking that minimizes log-PPL(y|x,z).

    Args:
        x: Context/question
        y: Target answer
        z_init: Initial thinking (optional)
        gen_model: Model for generating thinking
        reflection_model: Model for GEPA reflection
        max_iterations: Maximum optimization iterations

    Returns:
        Dictionary with evolved thinking and metrics
    """
    from mlx_server_config import configure_mlx_server

    # Configure DSPy with MLX server
    configure_mlx_server()

    # Create training example
    train_example = Example(context=x, answer=y).with_inputs("context", "answer")

    # Initialize thinking module
    student = ThinkingModule(use_cot=True)

    # If we have initial thinking, we could warm-start somehow
    # For now, let GEPA evolve from scratch

    # Initialize perplexity calculator
    ppl_calc = PerplexityCalculator(gen_model)
    await ppl_calc.initialize()

    # Create perplexity metric
    metric = create_perplexity_metric(ppl_calc)

    # Configure GEPA optimizer - GEPA uses GPT-5 for reflection
    # The reflection model analyzes failures and proposes improvements
    reflection_lm = dspy.LM(
        model=reflection_model,  # GPT-5
        temperature=1.0,  # Higher temp for creative reflection
        max_tokens=32000,  # Allow detailed analysis
    )

    gepa = GEPA(
        metric=metric,
        auto="light",  # Light budget for demo
        reflection_lm=reflection_lm,
        track_stats=True,
        track_best_outputs=True,
        num_threads=4,  # Parallel evaluation
        reflection_minibatch_size=3,  # Batch reflection
        seed=42,
    )

    # Compile (optimize) the thinking module
    optimized = gepa.compile(
        student,
        trainset=[train_example],
        valset=[train_example],  # Use same example for validation
    )

    # Generate optimized thinking
    result = optimized(context=x, answer=y)

    # Extract evolved thinking
    if isinstance(result, dict):
        z_evolved = result.get("thinking", "")
    else:
        z_evolved = getattr(result, "thinking", "")

    # Calculate final perplexity
    ppl_calc = PerplexityCalculator(gen_model)
    await ppl_calc.initialize()

    ppl_init = float("inf")
    if z_init:
        ppl_init = await ppl_calc.calculate_conditional_log_perplexity(x, z_init, y)

    ppl_final = await ppl_calc.calculate_conditional_log_perplexity(x, z_evolved, y)

    return {
        "z_init": z_init or "",
        "z_evolved": z_evolved,
        "ppl_init": ppl_init,
        "ppl_final": ppl_final,
        "improvement": ppl_init - ppl_final,
        "optimization_stats": getattr(optimized, "detailed_results", None),
    }


async def demo_gepa_reer():
    """Demo: Use GEPA to evolve thinking that minimizes perplexity."""

    print("=" * 60)
    print("GEPA-Powered REER: Intelligent Thinking Evolution")
    print("=" * 60)

    # Define problem
    x = "How can I improve my productivity?"
    y = "Improving productivity requires prioritizing tasks, time-blocking for focus, minimizing distractions, taking breaks, and tracking progress."

    # Initial (poor) thinking
    z_init = "Work hard and don't waste time."

    print(f"Context: {x}")
    print(f"Target: {y}")
    print(f"Initial thinking: {z_init}")
    print("\nEvolving thinking with GEPA...")

    # Evolve thinking
    result = await evolve_thinking_with_gepa(x, y, z_init)

    print("\nResults:")
    print(f"Initial log-PPL: {result['ppl_init']:.3f}")
    print(f"Final log-PPL: {result['ppl_final']:.3f}")
    print(f"Improvement: {result['improvement']:.3f}")
    print(f"\nEvolved thinking: {result['z_evolved']}")

    return result


if __name__ == "__main__":
    print("Make sure MLX server is running:")
    print("mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080")
    print()

    asyncio.run(demo_gepa_reer())
