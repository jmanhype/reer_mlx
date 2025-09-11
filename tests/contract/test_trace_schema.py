"""T006: Contract tests for traces.schema.json validation.

Tests the REER Trace schema contract for append-only records capturing
strategy extraction events from historical posts. Following London School TDD
with mock-first approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

from jsonschema import Draft7Validator, ValidationError, validate
from jsonschema.exceptions import SchemaError
import pytest


class TestTraceSchemaContract:
    """Contract tests for REER Trace schema validation.

    Tests schema structure, required fields, type checking, edge cases,
    and schema version compatibility using mock-driven development.
    """

    @pytest.fixture
    def schema_path(self) -> Path:
        """Return the path to traces.schema.json."""
        return (
            Path(__file__).parent.parent.parent
            / "specs"
            / "001-reer_mlx"
            / "contracts"
            / "traces.schema.json"
        )

    @pytest.fixture
    def trace_schema(self, schema_path: Path) -> dict[str, Any]:
        """Load the traces JSON schema."""
        with open(schema_path) as f:
            return json.load(f)

    @pytest.fixture
    def mock_validator(self) -> Mock:
        """Mock JSON schema validator for interaction testing."""
        validator = Mock(spec=Draft7Validator)
        validator.validate = Mock()
        validator.check_schema = Mock()
        validator.iter_errors = Mock(return_value=[])
        return validator

    @pytest.fixture
    def valid_trace_data(self) -> dict[str, Any]:
        """Valid trace data that should pass schema validation."""
        return {
            "id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    # Schema Structure and Metadata Tests

    def test_schema_has_required_metadata(self, trace_schema: dict[str, Any]):
        """Test that schema contains required JSON Schema Draft 7 metadata."""
        assert trace_schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert trace_schema["$id"] == "https://reer-dspy-mlx/schemas/traces.schema.json"
        assert trace_schema["title"] == "REER Trace"
        assert trace_schema["version"] == "1.0.0"
        assert trace_schema["type"] == "object"
        assert "description" in trace_schema

    def test_schema_validation_contract(
        self, trace_schema: dict[str, Any], mock_validator: Mock
    ):
        """Test schema validation behavior contract."""
        # This will fail initially - testing the contract, not implementation
        with patch("jsonschema.Draft7Validator", return_value=mock_validator):
            Draft7Validator(trace_schema)

            # Verify validator was created with our schema
            mock_validator.check_schema.assert_called_once()

    def test_schema_is_valid_json_schema(self, trace_schema: dict[str, Any]):
        """Test that the schema itself is a valid JSON Schema Draft 7."""
        # This will fail if schema has structural issues
        try:
            Draft7Validator.check_schema(trace_schema)
        except SchemaError as e:
            pytest.fail(f"Schema is not valid JSON Schema: {e}")

    # Required Fields Validation Tests

    def test_all_required_fields_present(self, trace_schema: dict[str, Any]):
        """Test that all required fields are defined in schema."""
        required_fields = {
            "id",
            "timestamp",
            "source_post_id",
            "seed_params",
            "score",
            "metrics",
            "strategy_features",
            "provider",
            "metadata",
        }
        schema_required = set(trace_schema["required"])
        assert schema_required == required_fields

    def test_missing_required_field_validation_fails(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that missing any required field causes validation failure."""
        required_fields = trace_schema["required"]

        for field in required_fields:
            invalid_data = valid_trace_data.copy()
            del invalid_data[field]

            # This should fail validation - testing the contract
            with pytest.raises(
                ValidationError, match=f"'{field}' is a required property"
            ):
                validate(invalid_data, trace_schema)

    def test_seed_params_required_fields(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that seed_params has all required nested fields."""
        seed_params_schema = trace_schema["properties"]["seed_params"]
        required_seed_fields = {"topic", "style", "length", "thread_size"}

        assert set(seed_params_schema["required"]) == required_seed_fields

        # Test missing nested required fields
        for field in required_seed_fields:
            invalid_data = valid_trace_data.copy()
            del invalid_data["seed_params"][field]

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_metrics_required_fields(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that metrics has all required nested fields."""
        metrics_schema = trace_schema["properties"]["metrics"]
        required_metrics_fields = {
            "impressions",
            "engagement_rate",
            "retweets",
            "likes",
        }

        assert set(metrics_schema["required"]) == required_metrics_fields

        # Test missing nested required fields
        for field in required_metrics_fields:
            invalid_data = valid_trace_data.copy()
            del invalid_data["metrics"][field]

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_metadata_required_fields(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that metadata has all required nested fields."""
        metadata_schema = trace_schema["properties"]["metadata"]
        required_metadata_fields = {"extraction_method", "confidence"}

        assert set(metadata_schema["required"]) == required_metadata_fields

        # Test missing nested required fields
        for field in required_metadata_fields:
            invalid_data = valid_trace_data.copy()
            del invalid_data["metadata"][field]

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    # Type Checking Tests

    def test_id_must_be_uuid_format(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that id field must be valid UUID format."""
        id_schema = trace_schema["properties"]["id"]
        assert id_schema["type"] == "string"
        assert id_schema["format"] == "uuid"

        # Test invalid UUID formats
        invalid_ids = ["not-a-uuid", "123", "", "12345678-1234-1234-1234"]

        for invalid_id in invalid_ids:
            invalid_data = valid_trace_data.copy()
            invalid_data["id"] = invalid_id

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_timestamp_must_be_datetime_format(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that timestamp field must be valid ISO 8601 datetime."""
        timestamp_schema = trace_schema["properties"]["timestamp"]
        assert timestamp_schema["type"] == "string"
        assert timestamp_schema["format"] == "date-time"

        # Test invalid datetime formats
        invalid_timestamps = [
            "not-a-date",
            "2024-13-01",
            "2024-01-01",
            "2024-01-01T25:00:00",
        ]

        for invalid_timestamp in invalid_timestamps:
            invalid_data = valid_trace_data.copy()
            invalid_data["timestamp"] = invalid_timestamp

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_score_must_be_number_in_range(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that score must be number between 0.0 and 1.0."""
        score_schema = trace_schema["properties"]["score"]
        assert score_schema["type"] == "number"
        assert score_schema["minimum"] == 0.0
        assert score_schema["maximum"] == 1.0

        # Test invalid score values
        invalid_scores = [-0.1, 1.1, "0.5", None]

        for invalid_score in invalid_scores:
            invalid_data = valid_trace_data.copy()
            invalid_data["score"] = invalid_score

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_seed_params_length_constraints(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that seed_params.length has proper integer constraints."""
        length_schema = trace_schema["properties"]["seed_params"]["properties"][
            "length"
        ]
        assert length_schema["type"] == "integer"
        assert length_schema["minimum"] == 1
        assert length_schema["maximum"] == 10000

        # Test invalid length values
        invalid_lengths = [0, -1, 10001, 0.5, "280"]

        for invalid_length in invalid_lengths:
            invalid_data = valid_trace_data.copy()
            invalid_data["seed_params"]["length"] = invalid_length

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_seed_params_thread_size_constraints(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that seed_params.thread_size has proper integer constraints."""
        thread_size_schema = trace_schema["properties"]["seed_params"]["properties"][
            "thread_size"
        ]
        assert thread_size_schema["type"] == "integer"
        assert thread_size_schema["minimum"] == 1
        assert thread_size_schema["maximum"] == 25

        # Test invalid thread_size values
        invalid_sizes = [0, -1, 26, 1.5, "5"]

        for invalid_size in invalid_sizes:
            invalid_data = valid_trace_data.copy()
            invalid_data["seed_params"]["thread_size"] = invalid_size

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_metrics_non_negative_integers(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that impression metrics must be non-negative integers."""
        for field in ["impressions", "retweets", "likes"]:
            field_schema = trace_schema["properties"]["metrics"]["properties"][field]
            assert field_schema["type"] == "integer"
            assert field_schema["minimum"] == 0

            # Test negative values
            invalid_data = valid_trace_data.copy()
            invalid_data["metrics"][field] = -1

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_engagement_rate_percentage_range(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that engagement_rate must be number between 0.0 and 100.0."""
        engagement_schema = trace_schema["properties"]["metrics"]["properties"][
            "engagement_rate"
        ]
        assert engagement_schema["type"] == "number"
        assert engagement_schema["minimum"] == 0.0
        assert engagement_schema["maximum"] == 100.0

        # Test invalid engagement rates
        invalid_rates = [-0.1, 100.1, "5.5"]

        for invalid_rate in invalid_rates:
            invalid_data = valid_trace_data.copy()
            invalid_data["metrics"]["engagement_rate"] = invalid_rate

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_strategy_features_array_validation(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that strategy_features is array of strings with minimum items."""
        features_schema = trace_schema["properties"]["strategy_features"]
        assert features_schema["type"] == "array"
        assert features_schema["items"]["type"] == "string"
        assert features_schema["minItems"] == 1

        # Test invalid arrays
        invalid_arrays = [[], ["valid", 123], "not-an-array", None]

        for invalid_array in invalid_arrays:
            invalid_data = valid_trace_data.copy()
            invalid_data["strategy_features"] = invalid_array

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_provider_pattern_validation(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that provider follows the required pattern."""
        provider_schema = trace_schema["properties"]["provider"]
        assert provider_schema["type"] == "string"
        assert provider_schema["pattern"] == "^(mlx|dspy)::.+$"

        # Test invalid provider patterns
        invalid_providers = [
            "mlx",
            "dspy",
            "openai::gpt-4",
            "mlx:",
            "dspy::",
            "invalid::model",
        ]

        for invalid_provider in invalid_providers:
            invalid_data = valid_trace_data.copy()
            invalid_data["provider"] = invalid_provider

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_metadata_confidence_range(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that metadata.confidence is number between 0.0 and 1.0."""
        confidence_schema = trace_schema["properties"]["metadata"]["properties"][
            "confidence"
        ]
        assert confidence_schema["type"] == "number"
        assert confidence_schema["minimum"] == 0.0
        assert confidence_schema["maximum"] == 1.0

        # Test invalid confidence values
        invalid_confidences = [-0.1, 1.1, "0.9"]

        for invalid_confidence in invalid_confidences:
            invalid_data = valid_trace_data.copy()
            invalid_data["metadata"]["confidence"] = invalid_confidence

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    # Edge Cases and Invalid Data Tests

    def test_additional_properties_not_allowed(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that additional properties are not allowed."""
        assert trace_schema["additionalProperties"] is False

        # Test with additional property
        invalid_data = valid_trace_data.copy()
        invalid_data["extra_field"] = "not allowed"

        with pytest.raises(
            ValidationError, match="Additional properties are not allowed"
        ):
            validate(invalid_data, trace_schema)

    def test_null_values_rejected(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that null values are rejected for required fields."""
        required_fields = trace_schema["required"]

        for field in required_fields:
            invalid_data = valid_trace_data.copy()
            invalid_data[field] = None

            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_empty_strings_validation(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test behavior with empty strings for string fields."""
        string_fields = ["source_post_id"]

        for field in string_fields:
            invalid_data = valid_trace_data.copy()
            invalid_data[field] = ""

            # Empty strings should be invalid for these fields
            with pytest.raises(ValidationError):
                validate(invalid_data, trace_schema)

    def test_boundary_values_validation(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test boundary values for numeric constraints."""
        # Test boundary values that should be valid
        boundary_tests = [
            ("score", 0.0),
            ("score", 1.0),
            ("seed_params.length", 1),
            ("seed_params.length", 10000),
            ("seed_params.thread_size", 1),
            ("seed_params.thread_size", 25),
            ("metrics.engagement_rate", 0.0),
            ("metrics.engagement_rate", 100.0),
            ("metadata.confidence", 0.0),
            ("metadata.confidence", 1.0),
        ]

        for field_path, value in boundary_tests:
            test_data = valid_trace_data.copy()
            if "." in field_path:
                parent, child = field_path.split(".", 1)
                test_data[parent][child] = value
            else:
                test_data[field_path] = value

            # These should validate successfully
            try:
                validate(test_data, trace_schema)
            except ValidationError as e:
                pytest.fail(
                    f"Boundary value {value} for {field_path} should be valid: {e}"
                )

    # Schema Version Compatibility Tests

    def test_schema_version_compatibility(self, trace_schema: dict[str, Any]):
        """Test schema version is properly defined for compatibility tracking."""
        assert "version" in trace_schema
        assert trace_schema["version"] == "1.0.0"

        # Version should follow semantic versioning pattern
        version_pattern = r"^\d+\.\d+\.\d+$"
        import re

        assert re.match(version_pattern, trace_schema["version"])

    def test_schema_id_uniqueness(self, trace_schema: dict[str, Any]):
        """Test that schema has unique identifier for version tracking."""
        assert "$id" in trace_schema
        schema_id = trace_schema["$id"]
        assert "traces.schema.json" in schema_id
        assert schema_id.startswith("https://")

    # Mock-based Interaction Tests (London School TDD)

    @patch("jsonschema.validate")
    def test_validation_interaction_contract(
        self,
        mock_validate: Mock,
        trace_schema: dict[str, Any],
        valid_trace_data: dict[str, Any],
    ):
        """Test the interaction contract with jsonschema validation."""
        # This tests HOW validation is called, not WHAT it validates
        mock_validate.return_value = None  # Successful validation

        # Code under test would call this
        validate(valid_trace_data, trace_schema)

        # Verify the interaction occurred with correct parameters
        mock_validate.assert_called_once_with(valid_trace_data, trace_schema)

    def test_error_handling_contract(self, trace_schema: dict[str, Any]):
        """Test that validation errors are properly raised and structured."""
        invalid_data = {"invalid": "data"}

        try:
            validate(invalid_data, trace_schema)
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            # Verify error has expected structure
            assert hasattr(e, "message")
            assert hasattr(e, "path")
            assert hasattr(e, "absolute_path")
            assert hasattr(e, "schema_path")

    def test_valid_data_contract_success(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test that valid data passes validation contract."""
        # This will fail initially since we're testing the contract
        try:
            validate(valid_trace_data, trace_schema)
            # If this passes, our schema implementation is working
        except ValidationError as e:
            # Expected in RED phase - implementation doesn't exist yet
            pytest.fail(f"Valid data should pass validation: {e}")
        except Exception as e:
            # Any other error indicates a problem with our test setup
            pytest.fail(f"Unexpected error in validation: {e}")

    # Performance and Edge Case Tests

    def test_large_strategy_features_array(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test validation with large strategy_features array."""
        test_data = valid_trace_data.copy()
        test_data["strategy_features"] = [f"feature_{i}" for i in range(100)]

        # Should handle large arrays efficiently
        try:
            validate(test_data, trace_schema)
        except ValidationError as e:
            pytest.fail(f"Large strategy_features array should be valid: {e}")

    def test_unicode_string_handling(
        self, trace_schema: dict[str, Any], valid_trace_data: dict[str, Any]
    ):
        """Test validation with Unicode strings."""
        test_data = valid_trace_data.copy()
        test_data["source_post_id"] = "post_🚀_测试_🎯"
        test_data["seed_params"]["topic"] = "AI 🤖 entwicklung"
        test_data["strategy_features"] = ["hashtag_💡", "question_❓", "emoji_usage_🎉"]

        # Should handle Unicode properly
        try:
            validate(test_data, trace_schema)
        except ValidationError as e:
            pytest.fail(f"Unicode strings should be valid: {e}")
