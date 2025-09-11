#!/usr/bin/env python3
"""Example usage of the schema validation tool.

This script demonstrates how to use the schema validation utility
programmatically and via CLI commands.
"""

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

# Add tools directory to path for imports
sys.path.append(str(Path(__file__).parent))
from schema_check import SchemaValidator, FileValidator, AutoFixer


def demonstrate_programmatic_usage():
    """Demonstrate using the schema validator programmatically."""
    print("=== Programmatic Usage Examples ===\n")

    # Initialize the validator
    print("1. Initializing schema validator...")
    validator = SchemaValidator()

    # List available schemas
    print(f"Available schemas: {validator.get_available_schemas()}")

    # Create test data
    valid_trace = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    print("\n2. Validating trace data...")
    result = validator.validate_data(valid_trace, "traces")
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")

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
    print("\n3. Validating invalid trace data...")
    result = validator.validate_data(invalid_trace, "traces")
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    for error in result.errors[:3]:  # Show first 3 errors
        path = ".".join(str(p) for p in error["path"]) if error["path"] else "root"
        print(f"   - {path}: {error['message']}")

    # Auto-fix the data
    print("\n4. Auto-fixing invalid data...")
    auto_fixer = AutoFixer(validator)
    fixed_data, fixes = auto_fixer.auto_fix_data(invalid_trace, "traces")
    print(f"   Applied {len(fixes)} fixes:")
    for fix in fixes:
        print(f"   - {fix['type']}: {fix['description']}")

    # Validate fixed data
    print("\n5. Validating fixed data...")
    result = validator.validate_data(fixed_data, "traces")
    print(f"   Valid: {result.is_valid}")
    print(f"   Remaining errors: {len(result.errors)}")


def demonstrate_file_operations():
    """Demonstrate file validation operations."""
    print("\n=== File Operations Examples ===\n")

    # Create temporary test files
    test_dir = Path("temp_examples")
    test_dir.mkdir(exist_ok=True)

    try:
        # Create a valid JSON file
        valid_file = test_dir / "valid_trace.json"
        valid_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        print("1. File validation using FileValidator...")
        validator = SchemaValidator()
        file_validator = FileValidator(validator)

        # Validate JSON file
        results = file_validator.validate_file(valid_file, "traces")
        print(f"   {valid_file.name}: {results[0].is_valid}")

        # Validate JSONL file
        results = file_validator.validate_file(jsonl_file, "traces")
        print(f"   {jsonl_file.name}: {len(results)} records")
        for i, result in enumerate(results):
            print(f"     Line {i+1}: {'✓' if result.is_valid else '✗'}")

        print(f"\n2. Created test files in {test_dir}/")
        print(f"   - {valid_file.name}")
        print(f"   - {jsonl_file.name}")

    except Exception as e:
        print(f"Error in file operations: {e}")

    finally:
        # Clean up
        if test_dir.exists():
            for file in test_dir.iterdir():
                file.unlink()
            test_dir.rmdir()


def demonstrate_cli_usage():
    """Demonstrate CLI usage with examples."""
    print("\n=== CLI Usage Examples ===\n")

    # Check if we can run the CLI
    script_path = Path(__file__).parent / "schema_check.py"
    if not script_path.exists():
        print("schema_check.py not found - CLI examples skipped")
        return

    print("1. List available schemas:")
    print(f"   python {script_path} list-schemas")

    print("\n2. Validate a single file:")
    print(f"   python {script_path} validate --file data.json --schema traces")

    print("\n3. Batch validate directory:")
    print(f"   python {script_path} batch --directory data/ --recursive")

    print("\n4. Auto-fix violations:")
    print(
        f"   python {script_path} fix --file invalid.json --schema traces --output fixed/"
    )

    print("\n5. Generate validation report:")
    print(f"   python {script_path} report --directory data/ --output report.json")

    print("\n6. Use custom schema mapping:")
    print(
        f'   python {script_path} batch --directory data/ --schema-mapping \'{{"trace": "traces"}}\''
    )

    # Actually run the list-schemas command as a demo
    try:
        print("\n7. Running list-schemas command:")
        result = subprocess.run(
            [sys.executable, str(script_path), "list-schemas"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # Show first few lines of output
            lines = result.stdout.strip().split("\n")[:8]
            for line in lines:
                print(f"   {line}")
            if len(result.stdout.strip().split("\n")) > 8:
                print("   ...")
        else:
            print(f"   Command failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("   Command timed out")
    except Exception as e:
        print(f"   Error running command: {e}")


def demonstrate_integration_patterns():
    """Demonstrate integration patterns for the validation tool."""
    print("\n=== Integration Patterns ===\n")

    print("1. Test Integration:")
    print(
        """
   # In your test files
   from tools.schema_check import SchemaValidator
   
   @pytest.fixture
   def validator():
       return SchemaValidator()
   
   def test_data_validation(validator):
       result = validator.validate_data(test_data, "traces")
       assert result.is_valid
   """
    )

    print("2. Pipeline Integration:")
    print(
        """
   # In CI/CD pipeline
   python tools/schema_check.py batch --directory data/ --output validation.json
   if [ $? -ne 0 ]; then
       echo "Schema validation failed"
       exit 1
   fi
   """
    )

    print("3. Data Processing Integration:")
    print(
        """
   # In data processing scripts
   from tools.schema_check import SchemaValidator, AutoFixer
   
   validator = SchemaValidator()
   fixer = AutoFixer(validator)
   
   for record in data_stream:
       result = validator.validate_data(record, schema_name)
       if not result.is_valid:
           fixed_record, fixes = fixer.auto_fix_data(record, schema_name)
           record = fixed_record
       
       process_record(record)
   """
    )

    print("4. API Integration:")
    print(
        """
   # In API endpoints
   from tools.schema_check import SchemaValidator
   
   validator = SchemaValidator()
   
   @app.post("/traces")
   def create_trace(trace_data: dict):
       result = validator.validate_data(trace_data, "traces")
       if not result.is_valid:
           return {"errors": result.errors}, 400
       
       # Process valid data
       return create_trace_record(trace_data)
   """
    )


def main():
    """Main demonstration function."""
    print("Schema Validation Tool - Usage Examples")
    print("=" * 50)

    try:
        demonstrate_programmatic_usage()
        demonstrate_file_operations()
        demonstrate_cli_usage()
        demonstrate_integration_patterns()

        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nFor more information, see tools/README.md")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
