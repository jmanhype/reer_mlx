"""REER trajectory local-search MVP.

This module provides a minimal, pluggable implementation for reverse-engineering
reasoning trajectories z for known (x, y) pairs. It is designed for offline use
in this repository and avoids network calls by delegating perplexity evaluation to
an injected evaluator (see tools.ppl_eval).

The algorithm initializes a coarse trajectory z(0), then iteratively refines
segments by proposing small edits and selecting the candidate that reduces a
proxy PPL(y|x,z). Caching ensures efficiency on repeated evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class ReasoningSegment:
    """A single reasoning step/segment."""

    text: str


@dataclass
class ReasoningTrajectory:
    """A trajectory is an ordered list of segments."""

    segments: list[ReasoningSegment] = field(default_factory=list)

    def to_text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text)


@dataclass
class TrajectorySearchConfig:
    max_iters: int = 8
    max_candidates_per_segment: int = 3
    patience: int = 2
    target_ppl: float = 1.1  # stop early if under threshold
    segment_window: int = 1  # how many neighbors to consider (future use)


class TrajectorySearch:
    """Local search over trajectories minimizing PPL(y|x,z).

    The evaluator should expose: ppl_fn(x: str, y: str, z_text: str) -> float.
    """

    def __init__(
        self,
        ppl_fn: Callable[[str, str, str], float],
        config: TrajectorySearchConfig | None = None,
    ) -> None:
        self.ppl_fn = ppl_fn
        self.cfg = config or TrajectorySearchConfig()
        self._cache: dict[tuple[str, str, str], float] = {}

    def _ppl(self, x: str, y: str, z: ReasoningTrajectory) -> float:
        key = (x, y, z.to_text())
        if key not in self._cache:
            self._cache[key] = float(self.ppl_fn(x, y, key[2]))
        return self._cache[key]

    def _init_trajectory(self, x: str, y: str) -> ReasoningTrajectory:
        """Create a coarse initial plan z(0) from (x,y) without copying y.

        Heuristic: derive 3–5 generic steps conditioned on x length and keywords.
        """
        steps: list[str] = []
        topic = (x or "topic").strip()
        steps.append(f"Clarify objective for: {topic} (audience, tone, outcome)")
        steps.append("Outline: Hook → Context → Value → CTA")
        steps.append("Select 1–2 concrete examples and platform-appropriate format")
        steps.append("Anticipate objections; add a question to drive replies")
        steps.append("Tighten copy (length, hashtags/mentions policy, emoji)")
        return ReasoningTrajectory([ReasoningSegment(s) for s in steps])

    def _propose_edits(self, seg: ReasoningSegment) -> list[ReasoningSegment]:
        """Generate small candidate refinements for a segment."""
        base = seg.text.strip()
        variants = [
            base,
            base + " (add example)",
            base.replace("→", "→ refine →"),
        ]
        # Deduplicate and cap
        uniq: list[str] = []
        for v in variants:
            if v not in uniq:
                uniq.append(v)
        return [
            ReasoningSegment(v) for v in uniq[: self.cfg.max_candidates_per_segment]
        ]

    def search(self, x: str, y: str) -> dict[str, Any]:
        """Run local search and return the best trajectory with stats."""
        z = self._init_trajectory(x, y)
        best_ppl = self._ppl(x, y, z)
        history: list[dict[str, Any]] = [
            {"iter": 0, "ppl": best_ppl, "segments": len(z.segments)}
        ]

        stagnant = 0
        for it in range(1, self.cfg.max_iters + 1):
            improved = False
            # Segment-wise refinement
            for idx, seg in enumerate(z.segments):
                candidates = self._propose_edits(seg)
                best_local = seg
                best_local_ppl = best_ppl

                for cand in candidates:
                    trial = ReasoningTrajectory(segments=z.segments.copy())
                    trial.segments[idx] = cand
                    ppl = self._ppl(x, y, trial)
                    if ppl < best_local_ppl:
                        best_local = cand
                        best_local_ppl = ppl

                if best_local is not seg:
                    z.segments[idx] = best_local
                    best_ppl = best_local_ppl
                    improved = True

            history.append({"iter": it, "ppl": best_ppl})

            if best_ppl <= self.cfg.target_ppl:
                break
            if not improved:
                stagnant += 1
                if stagnant >= self.cfg.patience:
                    break
            else:
                stagnant = 0

        return {
            "x": x,
            "y": y,
            "z_segments": [s.text for s in z.segments],
            "ppl_final": best_ppl,
            "search_history": history,
        }
