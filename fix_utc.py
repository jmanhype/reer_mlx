#!/usr/bin/env python3
import os
import re


def fix_utc_import(filepath):
    with open(filepath) as f:
        content = f.read()

    # Replace UTC import pattern
    pattern = r"from datetime import (.*?)UTC(.*?)(?=\n|$)"

    def replace_func(match):
        parts = match.group(1).split(",")
        after = match.group(2).split(",") if match.group(2) else []

        # Remove UTC from parts
        all_parts = [
            p.strip() for p in parts + after if p.strip() and p.strip() != "UTC"
        ]

        if all_parts:
            return f"from datetime import {', '.join(all_parts)}"
        return "from datetime import datetime"

    content = re.sub(pattern, replace_func, content)

    # Add timezone import if UTC was used
    if "UTC" in content and "from datetime import timezone" not in content:
        # Find the last datetime import
        import_pos = content.rfind("from datetime import")
        if import_pos != -1:
            end_pos = content.find("\n", import_pos)
            if end_pos != -1:
                content = (
                    content[: end_pos + 1]
                    + "from datetime import timezone\n"
                    + content[end_pos + 1 :]
                )

    # Replace UTC usage with timezone.utc
    content = re.sub(r"\bUTC\b", "timezone.utc", content)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Fixed {filepath}")


# Fix all Python files
files = [
    "./tools/memory_profiler.py",
    "./tools/example_usage.py",
    "./tools/schema_check.py",
    "./core/trajectory_synthesizer.py",
    "./core/integration.py",
    "./core/trainer.py",
    "./core/trace_store.py",
    "./config/logging_config.py",
    "./tests/unit/test_trace_store.py",
    "./tests/contract/test_schema_validator_integration.py",
    "./tests/contract/test_trace_schema.py",
    "./tests/contract/test_timeline_schema.py",
    "./tests/contract/test_candidate_schema.py",
    "./tests/integration/test_provider_switching.py",
    "./tests/integration/test_data_collection.py",
    "./tests/integration/test_reer_mining.py",
    "./tests/integration/test_gepa_tuning.py",
    "./tests/integration/test_pipeline.py",
    "./social/collectors/x_normalize.py",
    "./social/example_usage.py",
    "./dspy_program/reer_module.py",
    "./dspy_program/pipeline.py",
    "./dspy_program/evaluator.py",
]

for f in files:
    if os.path.exists(f):
        fix_utc_import(f)
