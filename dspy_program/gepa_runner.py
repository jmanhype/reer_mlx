"""DSPy GEPA runner integrating the repo's scorer as feedback metric.

This module wires dspy.teleprompt.gepa.GEPA to our ContentCandidate scoring,
allowing reflective prompt evolution driven by `REERCandidateScorer`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
import json
from typing import Any

try:
    import dspy
    from dspy import Example
    from dspy.primitives import Prediction
    from dspy.teleprompt.gepa import GEPA as DSPyGEPA
except Exception as e:  # pragma: no cover - import-time guard
    dspy = None  # type: ignore[assignment]
    DSPyGEPA = None  # type: ignore[assignment]
    Example = None  # type: ignore[assignment]
    Prediction = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from core.candidate_scorer import (
    ContentCandidate,
    REERCandidateScorer,
    ScoringConfig,
)


class ComposeSignature(dspy.Signature):  # type: ignore[name-defined]
    topic = dspy.InputField(desc="Topic or prompt")
    audience = dspy.InputField(desc="Target audience descriptor")
    post = dspy.OutputField(desc="One social post text")


class ComposeModule(dspy.Module):  # type: ignore[name-defined]
    def __init__(self, use_cot: bool = True) -> None:
        super().__init__()
        self.compose = (
            dspy.ChainOfThought(ComposeSignature)
            if use_cot
            else dspy.Predict(ComposeSignature)
        )

    def forward(self, topic: str, audience: str) -> Any:
        return self.compose(topic=topic, audience=audience)


def _make_metric(scorer: REERCandidateScorer):
    """Return a GEPA-compatible metric that yields score and textual feedback."""
    loop = asyncio.new_event_loop()

    async def _score_text(text: str) -> float:
        metrics = await scorer.score_candidate(ContentCandidate("tmp", text))
        return float(metrics.overall_score)

    def metric(
        gold: Example,  # type: ignore[name-defined]
        pred: Prediction,  # type: ignore[name-defined]
        trace: Any | None,
        pred_name: str | None,
        pred_trace: Any | None,
    ) -> dict[str, Any]:
        # Extract text from pred in either dict or object form
        if isinstance(pred, dict):
            text = pred.get("post", "")
        else:
            text = getattr(pred, "post", "")

        try:
            score = loop.run_until_complete(_score_text(text))
        except RuntimeError:
            # Fallback if event loop policy differs
            score = asyncio.run(_score_text(text))

        feedback = f"This trajectory got a score of {score:.3f}."
        return {"score": score, "feedback": feedback}

    return metric


def run_gepa(
    train_tasks: Iterable[dict[str, str]],
    *,
    val_tasks: Iterable[dict[str, str]] | None = None,
    gen_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    reflection_model: str = "gpt-5",
    auto: str | None = "light",
    max_full_evals: int | None = None,
    max_metric_calls: int | None = None,
    track_stats: bool = True,
    log_dir: str | None = None,
    use_cot: bool = True,
    use_perplexity: bool = False,
    seed: int | None = None,
):
    """Compile an optimized DSPy program using GEPA.

    Args:
        train_tasks: iterable of {"topic": str, "audience": str}
        val_tasks: optional iterable of the same shape; defaults to train_tasks
        gen_model: LM used for composing posts (dspy.settings)
        reflection_model: LM used by GEPA reflection
        auto: one of {"light","medium","heavy"} or None; otherwise use explicit budgets
        max_full_evals: explicit full evaluation budget
        max_metric_calls: explicit metric call budget
        track_stats: whether to attach detailed results
        log_dir: directory for GEPA logs/checkpoints
        use_cot: enable Chain-of-Thought predictor
        use_perplexity: enable perplexity during scoring (slower)
    Returns:
        Optimized DSPy Module
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "DSPy GEPA dependencies are missing. Install extras: 'pip install dspy-ai gepa'"
        ) from _IMPORT_ERROR

    # Configure generation LM
    dspy.settings.configure(lm=dspy.LM(model=gen_model))

    # Student program
    student = ComposeModule(use_cot=use_cot)

    # Build train/val sets
    def to_examples(items: Iterable[dict[str, str]]):
        return [
            Example(
                topic=i.get("topic", ""), audience=i.get("audience", "")
            ).with_inputs("topic", "audience")
            for i in items
        ]

    trainset = to_examples(list(train_tasks))
    valset = to_examples(list(val_tasks)) if val_tasks is not None else None

    # Scorer-backed metric
    scorer = REERCandidateScorer(ScoringConfig(use_perplexity=use_perplexity))

    # Reflection LM
    reflection_lm = dspy.LM(model=reflection_model)

    # GEPA optimizer
    gepa = DSPyGEPA(  # type: ignore[operator]
        metric=_make_metric(scorer),
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        track_stats=track_stats,
        log_dir=log_dir,
        seed=seed or 0,
    )

    compiled = gepa.compile(student, trainset=trainset, valset=valset or trainset)

    class ProgramAdapter:
        """Adapter to match pipeline ContentGeneratorModule interface.

        Exposes call semantics compatible with pipeline:
        - __call__(topic, platform, audience, style, requirements) -> obj with .content
        - refine(original_content, feedback, target_metrics, platform_constraints) -> obj with .refined_content
        - to_dict() -> forwards to underlying compiled.to_dict() if available
        """

        def __init__(self, prog):
            self._prog = prog

        def __call__(
            self,
            *,
            topic: str,
            platform: str,
            audience: str,
            style: str,
            requirements: str,
        ):
            # Only topic/audience supported by ComposeSignature
            pred = self._prog(topic=topic, audience=audience)
            text = (
                pred["post"]
                if isinstance(pred, dict)
                else getattr(pred, "post", str(pred))
            )

            class _Res:
                def __init__(self, content: str):
                    self.content = content
                    self.reasoning = ""

            return _Res(text)

        def refine(
            self,
            *,
            original_content: str,
            feedback: str,
            target_metrics: str,
            platform_constraints: str,
        ):
            # Simple refinement: re-compose from hint in feedback; fallback to original
            # In a full system, we'd add feedback into prompt; for now return original.
            class _Res:
                def __init__(self, refined: str):
                    self.refined_content = refined

            return _Res(original_content)

        def to_dict(self):
            fn = getattr(self._prog, "to_dict", None)
            return fn() if callable(fn) else {"program": str(self._prog)}

    return ProgramAdapter(compiled)


def build_examples_from_traces(
    traces_path: str | Path, max_examples: int = 200
) -> list[dict[str, str]]:
    """Construct lightweight train tasks from REER traces JSON/JSONL.

    Extracts `seed_params.topic` and produces {"topic","audience"} pairs.
    Audience defaults to "general".
    """
    p = Path(traces_path)
    tasks: list[dict[str, str]] = []
    if not p.exists():
        return tasks

    def _push(topic: str):
        if topic:
            tasks.append({"topic": topic, "audience": "general"})

    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if len(tasks) >= max_examples:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = (
                    obj.get("seed_params", {}).get("topic") or obj.get("topic") or ""
                )
                _push(str(topic))
    else:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for obj in data[:max_examples]:
                    topic = (
                        (obj or {}).get("seed_params", {}).get("topic")
                        or obj.get("topic")
                        or ""
                    )
                    _push(str(topic))
        except Exception:
            pass

    return tasks
