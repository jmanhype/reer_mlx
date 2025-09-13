#!/usr/bin/env python3
"""
Complete GEPA + MLX + REER Integration Demo

Architecture:
1. Task Model: MLX 3B (via MLX server) - generates thinking
2. Reflection Model: GPT-5 - analyzes failures and proposes improvements
3. Metric: Conditional perplexity using MLX - evaluates thinking quality

This demonstrates the correct GEPA usage pattern where:
- A smaller, efficient model (MLX 3B) is optimized
- A powerful model (GPT-5) provides reflection and guidance
- The metric provides both score and textual feedback
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dspy
from dspy import Example
from dspy.teleprompt import GEPA

from core.candidate_scorer import PerplexityCalculator
from dspy_program.mlx_server_config import configure_mlx_server


class ThinkingSignature(dspy.Signature):
    """Generate reasoning that explains an answer."""

    context = dspy.InputField(desc="The question or context")
    answer = dspy.InputField(desc="The target answer to explain")
    thinking = dspy.OutputField(
        desc="Clear step-by-step reasoning that leads to the answer"
    )


class ThinkingModule(dspy.Module):
    """DSPy module optimized by GEPA to minimize perplexity."""

    def __init__(self):
        super().__init__()
        # Use Chain of Thought for structured reasoning
        self.think = dspy.ChainOfThought(ThinkingSignature)

    def forward(self, context: str, answer: str):
        return self.think(context=context, answer=answer)


def create_perplexity_metric_with_feedback():
    """Create a GEPA metric with rich textual feedback.

    GEPA works best when the metric provides:
    1. A score (for optimization)
    2. Textual feedback (for reflection)
    """
    # We'll initialize the perplexity calculator lazily
    ppl_calc = None

    def metric(gold: Example, pred, trace=None, pred_name=None, pred_trace=None):
        nonlocal ppl_calc

        # Lazy initialization
        if ppl_calc is None:
            ppl_calc = PerplexityCalculator("mlx-community/Llama-3.2-3B-Instruct-4bit")
            # Initialize synchronously (GEPA calls this in sync context)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(ppl_calc.initialize())

        # Extract fields
        context = gold.get("context", "")
        answer = gold.get("answer", "")
        thinking = (
            pred.thinking if hasattr(pred, "thinking") else pred.get("thinking", "")
        )

        # Calculate perplexity
        async def calc_ppl():
            return await ppl_calc.calculate_conditional_log_perplexity(
                context, thinking, answer
            )

        loop = asyncio.new_event_loop()
        log_ppl = loop.run_until_complete(calc_ppl())

        # Convert to score (lower PPL is better, GEPA maximizes)
        score = 1.0 / (1.0 + max(0, log_ppl))

        # Generate rich feedback for GPT-5 reflection
        if log_ppl < 0.5:
            feedback = (
                f"Excellent reasoning (log-PPL: {log_ppl:.3f})! "
                f"The thinking clearly explains how to arrive at the answer. "
                f"Key strengths: systematic approach, clear logic flow."
            )
        elif log_ppl < 1.0:
            feedback = (
                f"Good reasoning (log-PPL: {log_ppl:.3f}). "
                f"The thinking mostly explains the answer. "
                f"To improve: add more specific steps or examples."
            )
        elif log_ppl < 2.0:
            feedback = (
                f"Moderate reasoning (log-PPL: {log_ppl:.3f}). "
                f"The thinking partially explains the answer but lacks clarity. "
                f"Issues: missing logical connections, vague statements. "
                f"Try: breaking down into smaller steps, being more specific."
            )
        else:
            feedback = (
                f"Poor reasoning (log-PPL: {log_ppl:.3f}). "
                f"The thinking doesn't effectively explain the answer. "
                f"Major issues: disconnected from answer, too vague, missing key points. "
                f"Needed: complete restructuring with clear cause-effect relationships."
            )

        # GEPA expects either a float score or a dict with 'score' and 'feedback'
        # But the DSPy evaluator expects a float for parallel execution
        return score  # Just return the score for now

    return metric


async def demo_gepa_mlx_reer():
    """Demonstrate GEPA optimizing MLX model with GPT-5 reflection."""

    print("=" * 60)
    print("GEPA + MLX + REER: Complete Integration Demo")
    print("=" * 60)
    print("\nArchitecture:")
    print("- Task Model: MLX 3B (generates thinking)")
    print("- Reflection Model: GPT-5 (proposes improvements)")
    print("- Metric: Conditional perplexity (evaluates quality)")
    print("=" * 60)

    # Configure MLX server for task model
    print("\n1. Configuring MLX server...")
    if not configure_mlx_server():
        print("❌ MLX server not running!")
        print(
            "Start with: mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080"
        )
        return
    print("✓ MLX 3B model ready")

    # Create training examples
    examples = [
        Example(
            context="How can I improve my productivity?",
            answer="Improving productivity requires prioritizing tasks using methods like Eisenhower Matrix, implementing time-blocking for focused work, minimizing distractions, taking regular breaks, and tracking progress.",
        ).with_inputs("context", "answer"),
        Example(
            context="What makes a successful startup?",
            answer="Successful startups solve real problems, have strong teams, iterate quickly based on feedback, maintain sufficient runway, and achieve product-market fit.",
        ).with_inputs("context", "answer"),
        Example(
            context="How do neural networks learn?",
            answer="Neural networks learn through backpropagation, adjusting weights based on error gradients, using optimization algorithms like SGD or Adam to minimize loss functions.",
        ).with_inputs("context", "answer"),
    ]

    print(f"\n2. Created {len(examples)} training examples")

    # Initialize student module (uses MLX 3B)
    student = ThinkingModule()

    # Create metric with rich feedback
    print("\n3. Creating perplexity metric with feedback...")
    metric = create_perplexity_metric_with_feedback()

    # Configure GEPA with GPT-5 reflection
    print("\n4. Configuring GEPA optimizer...")
    print("   - Reflection model: GPT-5 (analyzes failures)")
    print("   - Optimization: light budget")
    print("   - Parallel threads: 4")

    reflection_lm = dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000)

    gepa = GEPA(
        metric=metric,
        auto="light",  # Light budget for demo
        reflection_lm=reflection_lm,
        track_stats=True,
        track_best_outputs=True,
        num_threads=4,
        reflection_minibatch_size=2,
        seed=42,
    )

    print("\n5. Starting GEPA optimization...")
    print("   This will:")
    print("   - Use MLX 3B to generate thinking")
    print("   - Evaluate with perplexity metric")
    print("   - Use GPT-5 to reflect on failures")
    print("   - Evolve better prompts iteratively")

    # Compile (optimize) the module
    optimized = gepa.compile(
        student,
        trainset=examples,
        valset=examples[:1],  # Use first example for validation
    )

    print("\n6. Optimization complete!")

    # Test the optimized module
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZED MODULE")
    print("=" * 60)

    test_context = "How can I learn programming effectively?"
    test_answer = "Learn programming through hands-on projects, consistent practice, understanding fundamentals before frameworks, debugging skills, and engaging with the community."

    print(f"\nContext: {test_context}")
    print(f"Answer: {test_answer}")

    print("\n7. Generating optimized thinking...")
    result = optimized(context=test_context, answer=test_answer)

    if hasattr(result, "thinking"):
        print(f"\nOptimized Thinking:\n{result.thinking}")

    # Show optimization stats if available
    if hasattr(optimized, "detailed_results"):
        stats = optimized.detailed_results
        print("\n" + "=" * 60)
        print("OPTIMIZATION STATISTICS")
        print("=" * 60)
        if hasattr(stats, "best_score"):
            print(f"Best score achieved: {stats.best_score:.3f}")
        if hasattr(stats, "iterations"):
            print(f"Iterations: {stats.iterations}")

    print("\n✅ Demo complete!")
    print("\nKey Insights:")
    print("1. MLX 3B generates thinking efficiently on-device")
    print("2. GPT-5 provides intelligent reflection for improvement")
    print("3. Perplexity metric ensures thinking explains answers well")
    print("4. GEPA evolves prompts to minimize perplexity")


if __name__ == "__main__":
    print("Prerequisites:")
    print(
        "1. MLX server running: mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080"
    )
    print("2. OpenAI API key configured for GPT-5 reflection")
    print()

    asyncio.run(demo_gepa_mlx_reer())
