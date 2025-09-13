from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import re


def load_policy(path: Path | str = "policies/default.yml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # Try JSON sibling if YAML missing
        pj = p.with_suffix(".json")
        if pj.exists():
            return json.loads(pj.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"Policy not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".json", ".jsonc"}:
        return json.loads(text)
    # Try YAML, fallback to JSON sibling
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        pj = p.with_suffix(".json")
        if pj.exists():
            return json.loads(pj.read_text(encoding="utf-8"))
        try:
            return json.loads(text)
        except Exception:
            raise RuntimeError(
                f"Failed to load policy from {p}. Install pyyaml or provide JSON (looked for {pj.name})."
            )


def make_dedup_key(text: str) -> str:
    norm = text.lower().strip()
    norm = norm.replace("\r", "").replace("\n", " ")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def extract_conversation_seed(text: str) -> str | None:
    # Return the last question if exists, else None
    parts = [seg.strip() for seg in text.split("?") if seg.strip()]
    if len(parts) == 0:
        return None
    # Re-append the question mark
    return parts[-1] + "?"


def has_placeholders(text: str, policy: dict[str, Any]) -> bool:
    patterns = policy.get("constraints", {}).get("placeholder_patterns", [])
    return any(y in text for y in [x.replace("\\", "") for x in patterns])


def strip_claims(text: str) -> tuple[str, list[dict[str, str]]]:
    """Remove unsupported claim-like spans (percentages, 10x, 99.99%).
    Returns sanitized text and extracted claims list.
    """
    claims: list[dict[str, str]] = []
    # Percentage and x-multiplier patterns
    patterns = [
        (r"\b\d{1,3}%\b", "percentage"),
        (r"\b\d{1,2}x\b", "multiplier"),
        (r"\b99\.99%\b", "uptime"),
    ]
    new_text = text
    for pat, cat in patterns:
        for m in list(re.finditer(pat, new_text)):
            span = m.group(0)
            claims.append({"text": span, "category": cat})
        new_text = re.sub(pat, "", new_text)
    # Clean double spaces left by removals
    new_text = re.sub(r"\s{2,}", " ", new_text).strip()
    return new_text, claims


def detect_claims(text: str) -> list[dict[str, str]]:
    """Detect claim-like spans without modifying text.
    Matches percentages, multipliers like 10x, and 99.99% uptime.
    """
    claims: list[dict[str, str]] = []
    patterns = [
        (r"\b\d{1,3}%\b", "percentage"),
        (r"\b\d{1,2}x\b", "multiplier"),
        (r"\b99\.99%\b", "uptime"),
    ]
    for pat, cat in patterns:
        for m in re.finditer(pat, text):
            claims.append({"text": m.group(0), "category": cat})
    return claims


def check_policy(
    text: str,
    style: str,
    policy: dict[str, Any],
    seen_keys: set[str] | None = None,
) -> tuple[list[str], str]:
    """Return (violations, dedup_key) for a text under a given policy.

    Violations are data-driven using constraints/checks in the provided policy.
    """
    violations: list[str] = []
    # placeholders
    if has_placeholders(text, policy):
        violations.append("placeholder_present")
    # claims
    if detect_claims(text) and policy.get("checks", {}).get(
        "evidence_required_for_claims", False
    ):
        violations.append("claims_without_evidence")
    # questions
    q = text.count("?")
    if policy.get("checks", {}).get("question_required", False) and q == 0:
        violations.append("missing_question")
    max_q = policy.get("checks", {}).get("question_max")
    if isinstance(max_q, int) and q > max_q:
        violations.append("too_many_questions")
    # caps
    emoji_cap = policy.get("constraints", {}).get("emoji_max", {}).get(style, 2)
    if sum(1 for ch in text if ord(ch) > 127) > emoji_cap:
        violations.append("excessive_emoji")
    hash_cap = policy.get("constraints", {}).get("hashtags_max", {}).get(style, 2)
    if text.count("#") > hash_cap:
        violations.append("excessive_hashtags")
    # forbidden literals
    if any(
        lit.lower() in text.lower()
        for lit in policy.get("constraints", {}).get("forbidden_literals", [])
    ):
        violations.append("forbidden_literal")
    # length
    if len(text) > 280:
        violations.append("content_too_long")
    # dedup
    key = make_dedup_key(text)
    if seen_keys is not None and key in seen_keys:
        violations.append("duplicate_content")
    return violations, key
