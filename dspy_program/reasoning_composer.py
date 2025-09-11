"""ReasoningComposer (MVP)

Generates content by first emitting a short plan (z) and then composing output (y).
This is a lightweight, local-only scaffold to demonstrate the integration point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComposeConfig:
    include_plan: bool = False  # If True, return (z,y); otherwise only y


class ReasoningComposer:
    def __init__(self, config: ComposeConfig | None = None) -> None:
        self.config = config or ComposeConfig()

    def plan(self, topic: str, audience: str) -> list[str]:
        steps = [
            f"Define objective for {audience}",
            "Hook → Context → Value → CTA",
            "Add concrete example and question",
        ]
        return steps

    def compose(self, topic: str, audience: str) -> dict[str, Any]:
        z = self.plan(topic, audience)
        y = (
            f"{topic}: Quick insight for {audience}. "
            f"Hook with a question, give one concrete example, and end with a clear CTA."
        )
        if self.config.include_plan:
            return {"plan": z, "content": y}
        return {"content": y}
