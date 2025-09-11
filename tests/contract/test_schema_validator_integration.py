"""Integration tests for schema_check.py tool with existing contract tests.

Tests that the schema validation utility integrates correctly with the existing
contract test framework and provides consistent validation behavior.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from uuid import uuid4

import pytest

# Import the schema validator directly for integration testing
sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
from schema_check import AutoFixer, BatchValidator, FileValidator, SchemaValidator


class TestSchemaValidatorIntegration:
    """Integration tests for the schema validation utility."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def schema_validator(self, project_root: Path) -> SchemaValidator:
        """Create a schema validator instance."""
        schema_dir = project_root / "specs" / "001-reer_mlx" / "contracts"
        return SchemaValidator(schema_dir)

    @pytest.fixture
    def file_validator(self, schema_validator: SchemaValidator) -> FileValidator:
        """Create a file validator instance."""
        return FileValidator(schema_validator)

    @pytest.fixture
    def auto_fixer(self, schema_validator: SchemaValidator) -> AutoFixer:
        """Create an auto-fixer instance."""
        return AutoFixer(schema_validator)

    @pytest.fixture
    def batch_validator(self, schema_validator: SchemaValidator) -> BatchValidator:
        """Create a batch validator instance."""
        return BatchValidator(schema_validator)

    @pytest.fixture
    def valid_trace_data(self) -> dict[str, Any]:
        """Valid trace data for testing."""
        return {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source_post_id": "x_post_12345",
            "seed_params": {
                "topic": "AI development",
                "style": "technical",
                "length": 280,
                "thread_size": 1,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": [
                "hashtag_usage",
                "question_pattern",
                "call_to_action",
            ],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {"extraction_method": "reer_v1.0", "confidence": 0.92},
        }

    @pytest.fixture
    def temp_test_dir(self) -> Path:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    # Schema Discovery and Loading Tests

    def test_schema_discovery(self, schema_validator: SchemaValidator):
        """Test that all expected schemas are discovered and loaded."""
        available_schemas = schema_validator.get_available_schemas()
        expected_schemas = {"traces", "timeline", "candidate"}

        assert set(available_schemas) == expected_schemas

        # Verify each schema is properly loaded
        for schema_name in expected_schemas:
            assert schema_name in schema_validator.schemas
            assert schema_name in schema_validator.validators

            # Verify schema has required metadata
            schema = schema_validator.schemas[schema_name]
            assert "$schema" in schema
            assert "title" in schema
            assert "version" in schema
            assert "type" in schema

    def test_schema_validation_consistency(
        self, schema_validator: SchemaValidator, valid_trace_data: dict[str, Any]
    ):
        """Test that schema validation is consistent with jsonschema library."""
        from jsonschema import ValidationError, validate

        # Get the traces schema
        traces_schema = schema_validator.schemas["traces"]

        # Validate using jsonschema directly
        try:
            validate(valid_trace_data, traces_schema)
            direct_validation_passed = True
        except ValidationError:
            direct_validation_passed = False

        # Validate using our tool
        result = schema_validator.validate_data(valid_trace_data, "traces")
        tool_validation_passed = result.is_valid

        # Results should be consistent
        assert direct_validation_passed == tool_validation_passed

    # File Validation Tests

    def test_json_file_validation(
        self,
        file_validator: FileValidator,
        valid_trace_data: dict[str, Any],
        temp_test_dir: Path,
    ):
        """Test validation of JSON files."""
        test_file = temp_test_dir / "test_trace.json"

        # Write valid data
        with open(test_file, "w") as f:
            json.dump(valid_trace_data, f)

        # Validate the file
        results = file_validator.validate_file(test_file, "traces")

        assert len(results) == 1
        result = results[0]
        assert result.is_valid
        assert result.file_path == str(test_file)
        assert result.schema_name == "traces"
        assert len(result.errors) == 0

    def test_jsonl_file_validation(
        self,
        file_validator: FileValidator,
        valid_trace_data: dict[str, Any],
        temp_test_dir: Path,
    ):
        """Test validation of JSONL files."""
        test_file = temp_test_dir / "test_traces.jsonl"

        # Create test data with one valid and one invalid record
        valid_data = valid_trace_data.copy()
        invalid_data = valid_trace_data.copy()
        invalid_data["score"] = 1.5  # Invalid score

        # Write JSONL data
        with open(test_file, "w") as f:
            f.write(json.dumps(valid_data) + "\n")
            f.write(json.dumps(invalid_data) + "\n")

        # Validate the file
        results = file_validator.validate_file(test_file, "traces")

        assert len(results) == 2
        assert results[0].is_valid  # First record should be valid
        assert not results[1].is_valid  # Second record should be invalid
        assert results[1].line_number == 2
        assert len(results[1].errors) > 0

    def test_malformed_json_handling(
        self, file_validator: FileValidator, temp_test_dir: Path
    ):
        """Test handling of malformed JSON files."""
        test_file = temp_test_dir / "malformed.json"

        # Write malformed JSON
        with open(test_file, "w") as f:
            f.write('{"invalid": json}')  # Missing quotes around 'json'

        # Validate the file
        results = file_validator.validate_file(test_file, "traces")

        assert len(results) == 1
        result = results[0]
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "JSON decode error" in result.errors[0]["message"]

    # Auto-Fixing Tests

    def test_auto_fix_type_conversions(self, auto_fixer: AutoFixer):
        """Test automatic type conversion fixes."""
        invalid_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source_post_id": "x_post_12345",
            "seed_params": {
                "topic": "AI development",
                "style": "technical",
                "length": "280",  # String instead of integer
                "thread_size": "1",  # String instead of integer
            },
            "score": "0.85",  # String instead of number
            "metrics": {
                "impressions": "1000",  # String instead of integer
                "engagement_rate": 5.5,
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": ["hashtag_usage"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {"extraction_method": "reer_v1.0", "confidence": 0.92},
        }

        fixed_data, fixes = auto_fixer.auto_fix_data(invalid_data, "traces")

        # Check that type conversions were applied
        assert isinstance(fixed_data["seed_params"]["length"], int)
        assert isinstance(fixed_data["seed_params"]["thread_size"], int)
        assert isinstance(fixed_data["score"], float)
        assert isinstance(fixed_data["metrics"]["impressions"], int)

        # Check that fixes were recorded
        assert len(fixes) > 0
        fix_types = [fix["type"] for fix in fixes]
        assert "type_conversion" in fix_types

    def test_auto_fix_constraint_violations(self, auto_fixer: AutoFixer):
        """Test automatic constraint violation fixes."""
        invalid_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source_post_id": "x_post_12345",
            "seed_params": {
                "topic": "AI development",
                "style": "technical",
                "length": 0,  # Below minimum
                "thread_size": 50,  # Above maximum
            },
            "score": 1.5,  # Above maximum
            "metrics": {
                "impressions": -100,  # Below minimum
                "engagement_rate": 150.0,  # Above maximum
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": ["hashtag_usage"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {
                "extraction_method": "reer_v1.0",
                "confidence": 2.0,  # Above maximum
            },
        }

        fixed_data, fixes = auto_fixer.auto_fix_data(invalid_data, "traces")

        # Check that constraints were enforced
        assert fixed_data["seed_params"]["length"] >= 1  # Minimum enforced
        assert fixed_data["seed_params"]["thread_size"] <= 25  # Maximum enforced
        assert fixed_data["score"] <= 1.0  # Maximum enforced
        assert fixed_data["metrics"]["impressions"] >= 0  # Minimum enforced
        assert fixed_data["metrics"]["engagement_rate"] <= 100.0  # Maximum enforced
        assert fixed_data["metadata"]["confidence"] <= 1.0  # Maximum enforced

        # Check that constraint fixes were recorded
        fix_types = [fix["type"] for fix in fixes]
        assert "constraint_fix" in fix_types

    def test_auto_fix_missing_fields(self, auto_fixer: AutoFixer):
        """Test automatic addition of missing required fields."""
        incomplete_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            # Missing source_post_id, seed_params, score, metrics, strategy_features, provider, metadata
        }

        fixed_data, fixes = auto_fixer.auto_fix_data(incomplete_data, "traces")

        # Check that missing required fields were added
        required_fields = [
            "source_post_id",
            "seed_params",
            "score",
            "metrics",
            "strategy_features",
            "provider",
            "metadata",
        ]
        for field in required_fields:
            assert field in fixed_data

        # Check that nested required fields were added
        assert "topic" in fixed_data["seed_params"]
        assert "style" in fixed_data["seed_params"]
        assert "length" in fixed_data["seed_params"]
        assert "thread_size" in fixed_data["seed_params"]

        assert "impressions" in fixed_data["metrics"]
        assert "engagement_rate" in fixed_data["metrics"]
        assert "retweets" in fixed_data["metrics"]
        assert "likes" in fixed_data["metrics"]

        assert "extraction_method" in fixed_data["metadata"]
        assert "confidence" in fixed_data["metadata"]

        # Check that fixes were recorded
        fix_types = [fix["type"] for fix in fixes]
        assert "missing_field" in fix_types

    # Batch Validation Tests

    def test_batch_validation_mixed_files(
        self,
        batch_validator: BatchValidator,
        valid_trace_data: dict[str, Any],
        temp_test_dir: Path,
    ):
        """Test batch validation with mixed valid and invalid files."""
        # Create test files
        valid_file = temp_test_dir / "valid_trace.json"
        invalid_file = temp_test_dir / "invalid_trace.json"
        jsonl_file = temp_test_dir / "mixed_traces.jsonl"

        # Write valid JSON file
        with open(valid_file, "w") as f:
            json.dump(valid_trace_data, f)

        # Write invalid JSON file
        invalid_data = valid_trace_data.copy()
        invalid_data["score"] = 1.5  # Invalid score
        with open(invalid_file, "w") as f:
            json.dump(invalid_data, f)

        # Write mixed JSONL file
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(valid_trace_data) + "\n")
            f.write(json.dumps(invalid_data) + "\n")

        # Set up schema mapping for all files
        schema_mapping = {"trace": "traces"}

        # Run batch validation
        report = batch_validator.validate_directory(
            temp_test_dir, schema_mapping, recursive=False
        )

        # Check report summary
        assert report.total_files == 3
        assert report.total_records == 4  # 1 + 1 + 2 records
        assert report.valid_files >= 1  # At least the valid file
        assert report.invalid_files >= 1  # At least the invalid file
        assert "traces" in report.schema_coverage

    # CLI Integration Tests

    def test_cli_list_schemas(self, project_root: Path):
        """Test the CLI list-schemas command."""
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "tools" / "schema_check.py"),
                "list-schemas",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0
        output = result.stdout

        # Check that all expected schemas are listed
        assert "traces" in output
        assert "timeline" in output
        assert "candidate" in output
        assert "v1.0.0" in output  # Version should be shown

    def test_cli_validate_command(
        self, project_root: Path, valid_trace_data: dict[str, Any], temp_test_dir: Path
    ):
        """Test the CLI validate command."""
        test_file = temp_test_dir / "test_trace.json"

        # Write test data
        with open(test_file, "w") as f:
            json.dump(valid_trace_data, f)

        # Run validation
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "tools" / "schema_check.py"),
                "validate",
                "--file",
                str(test_file),
                "--schema",
                "traces",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0
        output = result.stdout
        assert "Valid: True" in output

    def test_cli_fix_command(self, project_root: Path, temp_test_dir: Path):
        """Test the CLI fix command."""
        test_file = temp_test_dir / "invalid_trace.json"
        output_dir = temp_test_dir / "fixed"

        # Create invalid test data
        invalid_data = {
            "id": "invalid-uuid",
            "timestamp": "invalid-date",
            "source_post_id": "",
            "seed_params": {
                "topic": "AI",
                "style": "tech",
                "length": "280",  # Wrong type
                "thread_size": 0,  # Below minimum
            },
            "score": 1.5,  # Above maximum
            "metrics": {
                "impressions": -1,  # Below minimum
                "engagement_rate": 5.5,
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": [],  # Empty array
            "provider": "invalid-provider",
            "metadata": {
                "extraction_method": "reer_v1.0"
                # Missing confidence field
            },
        }

        # Write invalid data
        with open(test_file, "w") as f:
            json.dump(invalid_data, f)

        # Run fix command
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "tools" / "schema_check.py"),
                "fix",
                "--file",
                str(test_file),
                "--schema",
                "traces",
                "--output",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0
        output = result.stdout
        assert "Fixed:" in output
        assert "Applied" in output

        # Check that fixed file exists
        fixed_file = output_dir / "invalid_trace.json"
        assert fixed_file.exists()

        # Verify the fixed file has some improvements
        with open(fixed_file) as f:
            fixed_data = json.load(f)

        # Some basic checks that fixes were applied
        assert fixed_data["score"] <= 1.0  # Score should be clamped
        assert len(fixed_data["strategy_features"]) > 0  # Array should be populated

    def test_cli_report_command(
        self, project_root: Path, valid_trace_data: dict[str, Any], temp_test_dir: Path
    ):
        """Test the CLI report command."""
        test_file = temp_test_dir / "test_trace.json"
        report_file = temp_test_dir / "report.json"

        # Write test data
        with open(test_file, "w") as f:
            json.dump(valid_trace_data, f)

        # Run report command (note: this might not find the file due to schema mapping)
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "tools" / "schema_check.py"),
                "report",
                "--directory",
                str(temp_test_dir),
                "--output",
                str(report_file),
                "--schema-mapping",
                '{"trace": "traces"}',
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            check=False,
        )

        assert result.returncode == 0

        # Check that report file was created
        assert report_file.exists()

        # Verify report structure
        with open(report_file) as f:
            report_data = json.load(f)

        assert "timestamp" in report_data
        assert "total_files" in report_data
        assert "total_records" in report_data
        assert "valid_files" in report_data
        assert "invalid_files" in report_data
        assert "results" in report_data

    # Error Handling Tests

    def test_invalid_schema_name_handling(
        self, schema_validator: SchemaValidator, valid_trace_data: dict[str, Any]
    ):
        """Test handling of invalid schema names."""
        with pytest.raises(ValueError, match="Schema 'nonexistent' not found"):
            schema_validator.validate_data(valid_trace_data, "nonexistent")

    def test_missing_file_handling(self, file_validator: FileValidator):
        """Test handling of missing files."""
        missing_file = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            file_validator.validate_file(missing_file, "traces")

    def test_unsupported_file_type_handling(
        self, file_validator: FileValidator, temp_test_dir: Path
    ):
        """Test handling of unsupported file types."""
        unsupported_file = temp_test_dir / "test.xml"

        # Create a test file
        with open(unsupported_file, "w") as f:
            f.write("<xml>test</xml>")

        with pytest.raises(ValueError, match="Unsupported file type"):
            file_validator.validate_file(unsupported_file, "traces")

    # Performance and Edge Case Tests

    def test_large_file_handling(
        self,
        file_validator: FileValidator,
        valid_trace_data: dict[str, Any],
        temp_test_dir: Path,
    ):
        """Test handling of large JSONL files."""
        large_file = temp_test_dir / "large_traces.jsonl"

        # Create a large JSONL file with many records
        num_records = 1000
        with open(large_file, "w") as f:
            for i in range(num_records):
                record = valid_trace_data.copy()
                record["id"] = str(uuid4())
                record["source_post_id"] = f"post_{i}"
                f.write(json.dumps(record) + "\n")

        # Validate the large file
        results = file_validator.validate_file(large_file, "traces")

        assert len(results) == num_records
        assert all(result.is_valid for result in results)

        # Check that line numbers are correct
        for i, result in enumerate(results):
            assert result.line_number == i + 1
            assert result.record_index == i

    def test_unicode_content_handling(
        self, file_validator: FileValidator, temp_test_dir: Path
    ):
        """Test handling of files with Unicode content."""
        unicode_file = temp_test_dir / "unicode_trace.json"

        unicode_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "source_post_id": "post_🚀_测试",
            "seed_params": {
                "topic": "AI 🤖 development",
                "style": "technical",
                "length": 280,
                "thread_size": 1,
            },
            "score": 0.85,
            "metrics": {
                "impressions": 1000,
                "engagement_rate": 5.5,
                "retweets": 15,
                "likes": 55,
            },
            "strategy_features": ["hashtag_💡", "emoji_usage_🎉"],
            "provider": "mlx::llama-3.2-3b",
            "metadata": {"extraction_method": "reer_v1.0", "confidence": 0.92},
        }

        # Write Unicode data
        with open(unicode_file, "w", encoding="utf-8") as f:
            json.dump(unicode_data, f, ensure_ascii=False)

        # Validate the Unicode file
        results = file_validator.validate_file(unicode_file, "traces")

        assert len(results) == 1
        assert results[0].is_valid
