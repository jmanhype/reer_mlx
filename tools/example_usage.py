#!/usr/bin/env python3
"""Example usage of the schema validation tool.

This script demonstrates how to use the schema validation utility
programmatically and via CLI commands.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

# Add tools directory to path for imports
sys.path.append(str(Path(__file__).parent))
from schema_check import AutoFixer, FileValidator, SchemaValidator


def demonstrate_programmatic_usage():
    """Demonstrate using the schema validator programmatically."""

    # Initialize the validator
    validator = SchemaValidator()

    # List available schemas

    # Create test data
    valid_trace = {
        "id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "source_post_id": "example_post_123",
        "seed_params": {
            "topic": "Schema Validation",
            "style": "educational",
            "length": 250,
            "thread_size": 1,
        },
        "score": 0.78,
        "metrics": {
            "impressions": 500,
            "engagement_rate": 4.2,
            "retweets": 8,
            "likes": 21,
        },
        "strategy_features": ["hashtag_usage", "educational_content"],
        "provider": "mlx::example-model",
        "metadata": {"extraction_method": "example_v1.0", "confidence": 0.85},
    }

    # Validate the data
    result = validator.validate_data(valid_trace, "traces")

    # Create invalid data for testing auto-fix
    invalid_trace = valid_trace.copy()
    invalid_trace.update(
        {
            "id": "invalid-uuid",
            "score": 1.5,  # Above maximum
            "seed_params": {
                **invalid_trace["seed_params"],
                "length": "250",  # Wrong type
                "thread_size": 0,  # Below minimum
            },
            "strategy_features": [],  # Empty array
            "provider": "invalid-provider",  # Wrong pattern
        }
    )

    # Validate invalid data
    result = validator.validate_data(invalid_trace, "traces")
    for error in result.errors[:3]:  # Show first 3 errors
        ".".join(str(p) for p in error["path"]) if error["path"] else "root"

    # Auto-fix the data
    auto_fixer = AutoFixer(validator)
    fixed_data, fixes = auto_fixer.auto_fix_data(invalid_trace, "traces")
    for _fix in fixes:
        pass

    # Validate fixed data
    result = validator.validate_data(fixed_data, "traces")


def demonstrate_file_operations():
    """Demonstrate file validation operations."""

    # Create temporary test files
    test_dir = Path("temp_examples")
    test_dir.mkdir(exist_ok=True)

    try:
        # Create a valid JSON file
        valid_file = test_dir / "valid_trace.json"
        valid_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source_post_id": "file_example_001",
            "seed_params": {
                "topic": "File Validation",
                "style": "technical",
                "length": 180,
                "thread_size": 1,
            },
            "score": 0.92,
            "metrics": {
                "impressions": 1200,
                "engagement_rate": 6.8,
                "retweets": 25,
                "likes": 82,
            },
            "strategy_features": ["technical_content", "clear_examples"],
            "provider": "mlx::file-demo",
            "metadata": {"extraction_method": "file_demo_v1.0", "confidence": 0.94},
        }

        with open(valid_file, "w") as f:
            json.dump(valid_data, f, indent=2)

        # Create a JSONL file with mixed valid/invalid data
        jsonl_file = test_dir / "mixed_traces.jsonl"
        invalid_data = valid_data.copy()
        invalid_data["score"] = 1.8  # Invalid

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(valid_data) + "\n")
            f.write(json.dumps(invalid_data) + "\n")

        # Validate files using FileValidator
        validator = SchemaValidator()
        file_validator = FileValidator(validator)

        # Validate JSON file
        results = file_validator.validate_file(valid_file, "traces")

        # Validate JSONL file
        results = file_validator.validate_file(jsonl_file, "traces")
        for _i, _result in enumerate(results):
            pass

    except Exception:
        pass

    finally:
        # Clean up
        if test_dir.exists():
            for file in test_dir.iterdir():
                file.unlink()
            test_dir.rmdir()


def demonstrate_cli_usage():
    """Demonstrate CLI usage with examples."""

    # Check if we can run the CLI
    script_path = Path(__file__).parent / "schema_check.py"
    if not script_path.exists():
        return

    # Actually run the list-schemas command as a demo
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "list-schemas"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            # Show first few lines of output
            lines = result.stdout.strip().split("\n")[:8]
            for _line in lines:
                pass
            if len(result.stdout.strip().split("\n")) > 8:
                pass
        else:
            pass

    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass


def demonstrate_integration_patterns():
    """Demonstrate integration patterns for the validation tool."""


def main():
    """Main demonstration function."""

    try:
        demonstrate_programmatic_usage()
        demonstrate_file_operations()
        demonstrate_cli_usage()
        demonstrate_integration_patterns()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
