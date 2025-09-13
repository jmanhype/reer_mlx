from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timezone, datetime
import os
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    provider: str
    model_id: str
    version: str | None = None


def get_git_sha() -> str:
    # Prefer env override
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha
    # Try reading .git/HEAD to avoid spawning subprocesses
    try:
        git_dir = Path(__file__).resolve().parents[1] / ".git"
        head_file = git_dir / "HEAD"
        if head_file.exists():
            head = head_file.read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref_path = head.split(" ", 1)[1].strip()
                ref_file = git_dir / ref_path
                if ref_file.exists():
                    return ref_file.read_text(encoding="utf-8").strip()
            else:
                return head
    except Exception:
        pass
    return "0000000"


def make_provenance(
    seed: int,
    models: list[ModelInfo],
    run_id: str | None = None,
    template_id: str | None = None,
    trace_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "run_id": run_id or os.getenv("RUN_ID", "local"),
        "template_id": template_id,
        "models": [asdict(m) for m in models],
        "git_sha": get_git_sha(),
        "config": config or {},
    }
    if trace_url:
        prov["trace_url"] = trace_url
    return prov
