"""T007: Contract tests for candidate.schema.json validation.

Tests the Post Candidate schema contract for generated content awaiting
optimization and scheduling. Following London School TDD with mock-first
approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import timezone, datetime

UTC = timezone.utc
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

from jsonschema import Draft7Validator, ValidationError, validate
from jsonschema.exceptions import SchemaError
import pytest


class TestCandidateSchemaContract:
    """Contract tests for Post Candidate schema validation.

    Tests schema structure, required fields, type checking, edge cases,
    and schema version compatibility using mock-driven development.
    """

    @pytest.fixture
    def schema_path(self) -> Path:
        """Return the path to candidate.schema.json."""
        return (
            Path(__file__).parent.parent.parent
            / "specs"
            / "001-reer_mlx"
            / "contracts"
            / "candidate.schema.json"
        )

    @pytest.fixture
    def candidate_schema(self, schema_path: Path) -> dict[str, Any]:
        """Load the candidate JSON schema."""
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
    def valid_candidate_data(self) -> dict[str, Any]:
        """Valid candidate data that should pass schema validation."""
        return {
            "id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "content": "Exploring the latest trends in AI development! What's your take on the future of machine learning? #AI #TechTrends",
            "thread_parts": [
                "Exploring the latest trends in AI development!",
                "What's your take on the future of machine learning?",
                "Join the conversation! #AI #TechTrends",
            ],
            "features": {
                "length": 142,
                "hashtags": ["#AI", "#TechTrends"],
                "mentions": ["@techguru", "@airesearcher"],
                "links": ["https://example.com/article"],
                "media_type": "image",
            },
            "score": 0.78,
            "scoring_details": {
                "perplexity": 45.2,
                "style_match": 0.85,
                "engagement_prediction": 0.72,
            },
            "provider": "dspy::llama-3.2-3b-instruct",
            "trace_ids": [str(uuid4()), str(uuid4())],
            "status": "draft",
        }

    # Schema Structure and Metadata Tests

    def test_schema_has_required_metadata(self, candidate_schema: dict[str, Any]):
        """Test that schema contains required JSON Schema Draft 7 metadata."""
        assert candidate_schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert (
            candidate_schema["$id"]
            == "https://reer-dspy-mlx/schemas/candidate.schema.json"
        )
        assert candidate_schema["title"] == "Post Candidate"
        assert candidate_schema["version"] == "1.0.0"
        assert candidate_schema["type"] == "object"
        assert "description" in candidate_schema

    def test_schema_validation_contract(
        self, candidate_schema: dict[str, Any], mock_validator: Mock
    ):
        """Test schema validation behavior contract."""
        # This will fail initially - testing the contract, not implementation
        with patch("jsonschema.Draft7Validator", return_value=mock_validator):
            Draft7Validator(candidate_schema)

            # Verify validator was created with our schema
            mock_validator.check_schema.assert_called_once()

    def test_schema_is_valid_json_schema(self, candidate_schema: dict[str, Any]):
        """Test that the schema itself is a valid JSON Schema Draft 7."""
        # This will fail if schema has structural issues
        try:
            Draft7Validator.check_schema(candidate_schema)
        except SchemaError as e:
            pytest.fail(f"Schema is not valid JSON Schema: {e}")

    # Required Fields Validation Tests

    def test_all_required_fields_present(self, candidate_schema: dict[str, Any]):
        """Test that all required fields are defined in schema."""
        required_fields = {
            "id",
            "created_at",
            "content",
            "features",
            "score",
            "scoring_details",
            "provider",
            "trace_ids",
            "status",
        }
        schema_required = set(candidate_schema["required"])
        assert schema_required == required_fields

    def test_missing_required_field_validation_fails(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that missing any required field causes validation failure."""
        required_fields = candidate_schema["required"]

        for field in required_fields:
            invalid_data = valid_candidate_data.copy()
            del invalid_data[field]

            # This should fail validation - testing the contract
            with pytest.raises(
                ValidationError, match=f"'{field}' is a required property"
            ):
                validate(invalid_data, candidate_schema)

    def test_features_required_fields(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that features has all required nested fields."""
        features_schema = candidate_schema["properties"]["features"]
        required_features_fields = {
            "length",
            "hashtags",
            "mentions",
            "links",
            "media_type",
        }

        assert set(features_schema["required"]) == required_features_fields

        # Test missing nested required fields
        for field in required_features_fields:
            invalid_data = valid_candidate_data.copy()
            del invalid_data["features"][field]

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_scoring_details_required_fields(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that scoring_details has all required nested fields."""
        scoring_schema = candidate_schema["properties"]["scoring_details"]
        required_scoring_fields = {"perplexity", "style_match", "engagement_prediction"}

        assert set(scoring_schema["required"]) == required_scoring_fields

        # Test missing nested required fields
        for field in required_scoring_fields:
            invalid_data = valid_candidate_data.copy()
            del invalid_data["scoring_details"][field]

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    # Type Checking Tests

    def test_id_must_be_uuid_format(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that id field must be valid UUID format."""
        id_schema = candidate_schema["properties"]["id"]
        assert id_schema["type"] == "string"
        assert id_schema["format"] == "uuid"

        # Test invalid UUID formats
        invalid_ids = ["not-a-uuid", "123", "", "12345678-1234-1234-1234"]

        for invalid_id in invalid_ids:
            invalid_data = valid_candidate_data.copy()
            invalid_data["id"] = invalid_id

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_created_at_must_be_datetime_format(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that created_at field must be valid ISO 8601 datetime."""
        created_at_schema = candidate_schema["properties"]["created_at"]
        assert created_at_schema["type"] == "string"
        assert created_at_schema["format"] == "date-time"

        # Test invalid datetime formats
        invalid_timestamps = [
            "not-a-date",
            "2024-13-01",
            "2024-01-01",
            "2024-01-01T25:00:00",
        ]

        for invalid_timestamp in invalid_timestamps:
            invalid_data = valid_candidate_data.copy()
            invalid_data["created_at"] = invalid_timestamp

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_content_max_length_constraint(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that content has maximum length constraint for Twitter."""
        content_schema = candidate_schema["properties"]["content"]
        assert content_schema["type"] == "string"
        assert content_schema["maxLength"] == 280

        # Test content that's too long
        invalid_data = valid_candidate_data.copy()
        invalid_data["content"] = "x" * 281  # One character too long

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

    def test_thread_parts_array_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that thread_parts is array of strings with constraints."""
        thread_schema = candidate_schema["properties"]["thread_parts"]
        assert thread_schema["type"] == "array"
        assert thread_schema["items"]["type"] == "string"
        assert thread_schema["items"]["maxLength"] == 280
        assert thread_schema["maxItems"] == 25

        # Test too many thread parts
        invalid_data = valid_candidate_data.copy()
        invalid_data["thread_parts"] = ["part"] * 26  # Too many parts

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

        # Test thread part too long
        invalid_data = valid_candidate_data.copy()
        invalid_data["thread_parts"] = ["x" * 281]  # Part too long

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

    def test_score_must_be_number_in_range(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that score must be number between 0.0 and 1.0."""
        score_schema = candidate_schema["properties"]["score"]
        assert score_schema["type"] == "number"
        assert score_schema["minimum"] == 0.0
        assert score_schema["maximum"] == 1.0

        # Test invalid score values
        invalid_scores = [-0.1, 1.1, "0.5", None]

        for invalid_score in invalid_scores:
            invalid_data = valid_candidate_data.copy()
            invalid_data["score"] = invalid_score

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_features_length_constraints(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that features.length has proper integer constraints."""
        length_schema = candidate_schema["properties"]["features"]["properties"][
            "length"
        ]
        assert length_schema["type"] == "integer"
        assert length_schema["minimum"] == 1
        assert length_schema["maximum"] == 7000

        # Test invalid length values
        invalid_lengths = [0, -1, 7001, 1.5, "280"]

        for invalid_length in invalid_lengths:
            invalid_data = valid_candidate_data.copy()
            invalid_data["features"]["length"] = invalid_length

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_hashtags_pattern_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that hashtags follow the required pattern."""
        hashtags_schema = candidate_schema["properties"]["features"]["properties"][
            "hashtags"
        ]
        assert hashtags_schema["type"] == "array"
        assert hashtags_schema["items"]["type"] == "string"
        assert hashtags_schema["items"]["pattern"] == "^#\\w+"

        # Test invalid hashtag patterns
        invalid_hashtags = ["AI", "#", "#with-dash", "#123", ""]

        for invalid_hashtag in invalid_hashtags:
            invalid_data = valid_candidate_data.copy()
            invalid_data["features"]["hashtags"] = [invalid_hashtag]

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_mentions_pattern_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that mentions follow the required pattern."""
        mentions_schema = candidate_schema["properties"]["features"]["properties"][
            "mentions"
        ]
        assert mentions_schema["type"] == "array"
        assert mentions_schema["items"]["type"] == "string"
        assert mentions_schema["items"]["pattern"] == "^@\\w+"

        # Test invalid mention patterns
        invalid_mentions = ["username", "@", "@with-dash", "@123", ""]

        for invalid_mention in invalid_mentions:
            invalid_data = valid_candidate_data.copy()
            invalid_data["features"]["mentions"] = [invalid_mention]

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_links_uri_format_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that links are valid URIs."""
        links_schema = candidate_schema["properties"]["features"]["properties"]["links"]
        assert links_schema["type"] == "array"
        assert links_schema["items"]["type"] == "string"
        assert links_schema["items"]["format"] == "uri"

        # Test invalid URI formats
        invalid_links = ["not-a-url", "ftp://", "//example.com", "example.com"]

        for invalid_link in invalid_links:
            invalid_data = valid_candidate_data.copy()
            invalid_data["features"]["links"] = [invalid_link]

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_media_type_enum_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that media_type is one of allowed enum values."""
        media_type_schema = candidate_schema["properties"]["features"]["properties"][
            "media_type"
        ]
        assert media_type_schema["type"] == "string"
        assert set(media_type_schema["enum"]) == {"image", "video", "none"}

        # Test invalid media types
        invalid_media_types = ["audio", "gif", "document", ""]

        for invalid_media_type in invalid_media_types:
            invalid_data = valid_candidate_data.copy()
            invalid_data["features"]["media_type"] = invalid_media_type

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_scoring_details_constraints(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that scoring_details fields have proper constraints."""
        scoring_schema = candidate_schema["properties"]["scoring_details"]["properties"]

        # Test perplexity (number, minimum 0.0)
        perplexity_schema = scoring_schema["perplexity"]
        assert perplexity_schema["type"] == "number"
        assert perplexity_schema["minimum"] == 0.0

        invalid_data = valid_candidate_data.copy()
        invalid_data["scoring_details"]["perplexity"] = -1.0

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

        # Test style_match (number, 0.0-1.0 range)
        style_match_schema = scoring_schema["style_match"]
        assert style_match_schema["type"] == "number"
        assert style_match_schema["minimum"] == 0.0
        assert style_match_schema["maximum"] == 1.0

        invalid_values = [-0.1, 1.1]
        for invalid_value in invalid_values:
            invalid_data = valid_candidate_data.copy()
            invalid_data["scoring_details"]["style_match"] = invalid_value

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

        # Test engagement_prediction (number, 0.0-1.0 range)
        engagement_schema = scoring_schema["engagement_prediction"]
        assert engagement_schema["type"] == "number"
        assert engagement_schema["minimum"] == 0.0
        assert engagement_schema["maximum"] == 1.0

        for invalid_value in invalid_values:
            invalid_data = valid_candidate_data.copy()
            invalid_data["scoring_details"]["engagement_prediction"] = invalid_value

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_provider_pattern_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that provider follows the required pattern."""
        provider_schema = candidate_schema["properties"]["provider"]
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
            invalid_data = valid_candidate_data.copy()
            invalid_data["provider"] = invalid_provider

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_trace_ids_array_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that trace_ids is array of UUIDs with minimum items."""
        trace_ids_schema = candidate_schema["properties"]["trace_ids"]
        assert trace_ids_schema["type"] == "array"
        assert trace_ids_schema["items"]["type"] == "string"
        assert trace_ids_schema["items"]["format"] == "uuid"
        assert trace_ids_schema["minItems"] == 1

        # Test empty array
        invalid_data = valid_candidate_data.copy()
        invalid_data["trace_ids"] = []

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

        # Test invalid UUID in array
        invalid_data = valid_candidate_data.copy()
        invalid_data["trace_ids"] = ["not-a-uuid"]

        with pytest.raises(ValidationError):
            validate(invalid_data, candidate_schema)

    def test_status_enum_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that status is one of allowed enum values."""
        status_schema = candidate_schema["properties"]["status"]
        assert status_schema["type"] == "string"
        assert set(status_schema["enum"]) == {
            "draft",
            "scheduled",
            "published",
            "archived",
        }

        # Test invalid status values
        invalid_statuses = ["pending", "failed", "processing", ""]

        for invalid_status in invalid_statuses:
            invalid_data = valid_candidate_data.copy()
            invalid_data["status"] = invalid_status

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    # Edge Cases and Invalid Data Tests

    def test_additional_properties_not_allowed(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that additional properties are not allowed."""
        assert candidate_schema["additionalProperties"] is False

        # Test with additional property
        invalid_data = valid_candidate_data.copy()
        invalid_data["extra_field"] = "not allowed"

        with pytest.raises(
            ValidationError, match="Additional properties are not allowed"
        ):
            validate(invalid_data, candidate_schema)

    def test_null_values_rejected(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that null values are rejected for required fields."""
        required_fields = candidate_schema["required"]

        for field in required_fields:
            invalid_data = valid_candidate_data.copy()
            invalid_data[field] = None

            with pytest.raises(ValidationError):
                validate(invalid_data, candidate_schema)

    def test_empty_arrays_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test behavior with empty arrays for array fields."""
        # Empty hashtags, mentions, links should be valid
        test_data = valid_candidate_data.copy()
        test_data["features"]["hashtags"] = []
        test_data["features"]["mentions"] = []
        test_data["features"]["links"] = []

        try:
            validate(test_data, candidate_schema)
        except ValidationError as e:
            pytest.fail(f"Empty hashtags/mentions/links arrays should be valid: {e}")

    def test_boundary_values_validation(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test boundary values for numeric and string constraints."""
        # Test boundary values that should be valid
        boundary_tests = [
            ("content", "x" * 280),  # Max length content
            ("score", 0.0),
            ("score", 1.0),
            ("features.length", 1),
            ("features.length", 7000),
            ("scoring_details.perplexity", 0.0),
            ("scoring_details.style_match", 0.0),
            ("scoring_details.style_match", 1.0),
            ("scoring_details.engagement_prediction", 0.0),
            ("scoring_details.engagement_prediction", 1.0),
        ]

        for field_path, value in boundary_tests:
            test_data = valid_candidate_data.copy()
            if "." in field_path:
                parent, child = field_path.split(".", 1)
                test_data[parent][child] = value
            else:
                test_data[field_path] = value

            # These should validate successfully
            try:
                validate(test_data, candidate_schema)
            except ValidationError as e:
                pytest.fail(
                    f"Boundary value {value} for {field_path} should be valid: {e}"
                )

    def test_thread_parts_optional_field(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that thread_parts is optional."""
        # thread_parts should not be in required fields
        assert "thread_parts" not in candidate_schema["required"]

        # Data without thread_parts should be valid
        test_data = valid_candidate_data.copy()
        del test_data["thread_parts"]

        try:
            validate(test_data, candidate_schema)
        except ValidationError as e:
            pytest.fail(f"Data without thread_parts should be valid: {e}")

    # Schema Version Compatibility Tests

    def test_schema_version_compatibility(self, candidate_schema: dict[str, Any]):
        """Test schema version is properly defined for compatibility tracking."""
        assert "version" in candidate_schema
        assert candidate_schema["version"] == "1.0.0"

        # Version should follow semantic versioning pattern
        version_pattern = r"^\d+\.\d+\.\d+$"
        import re

        assert re.match(version_pattern, candidate_schema["version"])

    def test_schema_id_uniqueness(self, candidate_schema: dict[str, Any]):
        """Test that schema has unique identifier for version tracking."""
        assert "$id" in candidate_schema
        schema_id = candidate_schema["$id"]
        assert "candidate.schema.json" in schema_id
        assert schema_id.startswith("https://")

    # Mock-based Interaction Tests (London School TDD)

    @patch("jsonschema.validate")
    def test_validation_interaction_contract(
        self,
        mock_validate: Mock,
        candidate_schema: dict[str, Any],
        valid_candidate_data: dict[str, Any],
    ):
        """Test the interaction contract with jsonschema validation."""
        # This tests HOW validation is called, not WHAT it validates
        mock_validate.return_value = None  # Successful validation

        # Code under test would call this
        validate(valid_candidate_data, candidate_schema)

        # Verify the interaction occurred with correct parameters
        mock_validate.assert_called_once_with(valid_candidate_data, candidate_schema)

    def test_error_handling_contract(self, candidate_schema: dict[str, Any]):
        """Test that validation errors are properly raised and structured."""
        invalid_data = {"invalid": "data"}

        try:
            validate(invalid_data, candidate_schema)
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            # Verify error has expected structure
            assert hasattr(e, "message")
            assert hasattr(e, "path")
            assert hasattr(e, "absolute_path")
            assert hasattr(e, "schema_path")

    def test_valid_data_contract_success(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test that valid data passes validation contract."""
        # This will fail initially since we're testing the contract
        try:
            validate(valid_candidate_data, candidate_schema)
            # If this passes, our schema implementation is working
        except ValidationError as e:
            # Expected in RED phase - implementation doesn't exist yet
            pytest.fail(f"Valid data should pass validation: {e}")
        except Exception as e:
            # Any other error indicates a problem with our test setup
            pytest.fail(f"Unexpected error in validation: {e}")

    # Performance and Edge Case Tests

    def test_large_thread_parts_array(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test validation with maximum thread_parts array."""
        test_data = valid_candidate_data.copy()
        test_data["thread_parts"] = [
            f"Thread part {i}" for i in range(25)
        ]  # Max allowed

        # Should handle max thread parts
        try:
            validate(test_data, candidate_schema)
        except ValidationError as e:
            pytest.fail(f"Maximum thread_parts array should be valid: {e}")

    def test_unicode_content_handling(
        self, candidate_schema: dict[str, Any], valid_candidate_data: dict[str, Any]
    ):
        """Test validation with Unicode content."""
        test_data = valid_candidate_data.copy()
        test_data["content"] = "AI 🤖 development with émojis and ñoñó! 测试 🚀"
        test_data["features"]["hashtags"] = ["#AI", "#émojis", "#测试"]
        test_data["features"]["mentions"] = ["@técnico", "@测试用户"]

        # Should handle Unicode properly
        try:
            validate(test_data, candidate_schema)
        except ValidationError as e:
            pytest.fail(f"Unicode content should be valid: {e}")

    def test_minimal_valid_candidate(self, candidate_schema: dict[str, Any]):
        """Test minimal candidate with only required fields."""
        minimal_data = {
            "id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "content": "Minimal post",
            "features": {
                "length": 12,
                "hashtags": [],
                "mentions": [],
                "links": [],
                "media_type": "none",
            },
            "score": 0.5,
            "scoring_details": {
                "perplexity": 10.0,
                "style_match": 0.5,
                "engagement_prediction": 0.5,
            },
            "provider": "mlx::model",
            "trace_ids": [str(uuid4())],
            "status": "draft",
        }

        # Minimal valid data should pass
        try:
            validate(minimal_data, candidate_schema)
        except ValidationError as e:
            pytest.fail(f"Minimal valid candidate should pass: {e}")
