#!/usr/bin/env python3
"""T035: JSON schema validator utility for REER × DSPy × MLX project.

Comprehensive JSON schema validation utility that:
- Validates JSON/JSONL files against schemas in specs/001-reer_mlx/contracts/ directory
- Supports batch validation of multiple files
- Provides detailed error reporting with line numbers
- Can auto-fix common schema violations
- Generates validation reports
- Integrates with the existing contract tests

Usage:
    python tools/schema_check.py validate --file data.json --schema traces
    python tools/schema_check.py batch --directory data/ --schema-dir schemas/
    python tools/schema_check.py fix --file data.json --schema traces --output fixed.json
    python tools/schema_check.py report --directory data/ --output validation_report.json
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any
from uuid import UUID, uuid4

import jsonlines

# Third-party imports for schema validation
try:
    from jsonschema import Draft7Validator, FormatChecker, ValidationError, validate
    from jsonschema.exceptions import RefResolutionError, SchemaError
except ImportError:
    sys.exit(1)


@dataclass
class ValidationResult:
    """Result of a single validation operation."""

    file_path: str
    schema_name: str
    is_valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    line_number: int | None = None
    record_index: int | None = None
    validation_time: float | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for batch operations."""

    timestamp: str
    total_files: int
    total_records: int
    valid_files: int
    invalid_files: int
    files_with_warnings: int
    schema_coverage: dict[str, int]
    error_summary: dict[str, int]
    results: list[ValidationResult]
    auto_fixes_applied: list[dict[str, Any]]


class SchemaValidator:
    """Core JSON schema validation engine."""

    def __init__(self, schema_dir: Path | None = None):
        """Initialize validator with schema directory."""
        self.schema_dir = schema_dir or self._find_schema_dir()
        self.schemas = {}
        self.validators = {}
        self.format_checker = FormatChecker()
        self.logger = self._setup_logging()

        # Load all available schemas
        self._load_schemas()

    def _find_schema_dir(self) -> Path:
        """Find the schema directory in the project."""
        current = Path(__file__).resolve()
        project_root = None

        # Look for project root indicators
        for parent in current.parents:
            if (parent / "specs" / "001-reer_mlx" / "contracts").exists():
                project_root = parent
                break

        if not project_root:
            raise FileNotFoundError(
                "Could not find schema directory. Please specify --schema-dir"
            )

        return project_root / "specs" / "001-reer_mlx" / "contracts"

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation operations."""
        logger = logging.getLogger("schema_validator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_schemas(self) -> None:
        """Load all JSON schemas from the schema directory."""
        if not self.schema_dir.exists():
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")

        schema_files = list(self.schema_dir.glob("*.schema.json"))
        if not schema_files:
            raise FileNotFoundError(f"No schema files found in {self.schema_dir}")

        self.logger.info(f"Loading schemas from {self.schema_dir}")

        for schema_file in schema_files:
            try:
                with open(schema_file, encoding="utf-8") as f:
                    schema = json.load(f)

                # Extract schema name from filename (e.g., "traces.schema.json" -> "traces")
                schema_name = schema_file.stem.replace(".schema", "")

                # Validate the schema itself
                Draft7Validator.check_schema(schema)

                # Create validator with format checking
                validator = Draft7Validator(schema, format_checker=self.format_checker)

                self.schemas[schema_name] = schema
                self.validators[schema_name] = validator

                self.logger.info(f"Loaded schema: {schema_name}")

            except (json.JSONDecodeError, SchemaError) as e:
                self.logger.exception(f"Error loading schema {schema_file}: {e}")
                raise

    def get_available_schemas(self) -> list[str]:
        """Get list of available schema names."""
        return list(self.schemas.keys())

    def validate_data(
        self,
        data: Any,
        schema_name: str,
        file_path: str = "",
        line_number: int | None = None,
        record_index: int | None = None,
    ) -> ValidationResult:
        """Validate a single data object against a schema."""
        if schema_name not in self.validators:
            raise ValueError(
                f"Schema '{schema_name}' not found. Available: {list(self.schemas.keys())}"
            )

        validator = self.validators[schema_name]
        errors = []
        warnings = []

        start_time = datetime.now()

        try:
            # Perform validation
            validation_errors = list(validator.iter_errors(data))

            for error in validation_errors:
                error_detail = {
                    "message": error.message,
                    "path": list(error.path),
                    "schema_path": list(error.schema_path),
                    "invalid_value": error.instance,
                    "validator": error.validator,
                    "validator_value": error.validator_value,
                }

                # Classify some errors as warnings instead of errors
                if self._is_warning_level_error(error):
                    warnings.append(error_detail)
                else:
                    errors.append(error_detail)

            # Check for potential warnings (non-schema violations)
            self._check_for_warnings(data, schema_name, warnings)

        except Exception as e:
            errors.append(
                {
                    "message": f"Validation failed with exception: {str(e)}",
                    "path": [],
                    "schema_path": [],
                    "invalid_value": None,
                    "validator": "exception",
                    "validator_value": None,
                }
            )

        validation_time = (datetime.now() - start_time).total_seconds()

        return ValidationResult(
            file_path=file_path,
            schema_name=schema_name,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            line_number=line_number,
            record_index=record_index,
            validation_time=validation_time,
        )

    def _is_warning_level_error(self, error: ValidationError) -> bool:
        """Determine if a validation error should be treated as a warning."""
        # These are errors that might be auto-fixable or non-critical
        warning_validators = {
            "format",  # Format errors for non-critical fields
        }

        # Additional property errors for optional sections
        if error.validator == "additionalProperties":
            # Allow additional properties in metadata sections
            if "metadata" in error.absolute_path:
                return True

        return error.validator in warning_validators

    def _check_for_warnings(
        self, data: dict[str, Any], schema_name: str, warnings: list[dict[str, Any]]
    ) -> None:
        """Check for potential warnings that aren't schema violations."""
        if not isinstance(data, dict):
            return

        # Check for performance-related warnings
        if schema_name == "traces":
            self._check_trace_warnings(data, warnings)
        elif schema_name == "candidate":
            self._check_candidate_warnings(data, warnings)
        elif schema_name == "timeline":
            self._check_timeline_warnings(data, warnings)

    def _check_trace_warnings(
        self, data: dict[str, Any], warnings: list[dict[str, Any]]
    ) -> None:
        """Check for warnings specific to trace data."""
        # Warn about low confidence scores
        if "metadata" in data and "confidence" in data["metadata"]:
            confidence = data["metadata"]["confidence"]
            if confidence < 0.5:
                warnings.append(
                    {
                        "message": f"Low confidence score: {confidence}",
                        "path": ["metadata", "confidence"],
                        "type": "performance_warning",
                    }
                )

        # Warn about very low engagement
        if "metrics" in data and "engagement_rate" in data["metrics"]:
            engagement = data["metrics"]["engagement_rate"]
            if engagement < 1.0:
                warnings.append(
                    {
                        "message": f"Very low engagement rate: {engagement}%",
                        "path": ["metrics", "engagement_rate"],
                        "type": "performance_warning",
                    }
                )

    def _check_candidate_warnings(
        self, data: dict[str, Any], warnings: list[dict[str, Any]]
    ) -> None:
        """Check for warnings specific to candidate data."""
        # Warn about very long content
        if "content" in data and len(data["content"]) > 250:
            warnings.append(
                {
                    "message": f"Content length {len(data['content'])} approaching Twitter limit",
                    "path": ["content"],
                    "type": "length_warning",
                }
            )

        # Warn about low scores
        if "score" in data and data["score"] < 0.3:
            warnings.append(
                {
                    "message": f"Low predicted performance score: {data['score']}",
                    "path": ["score"],
                    "type": "performance_warning",
                }
            )

    def _check_timeline_warnings(
        self, data: dict[str, Any], warnings: list[dict[str, Any]]
    ) -> None:
        """Check for warnings specific to timeline data."""
        # Warn about scheduling in the past
        if "scheduled_time" in data:
            try:
                scheduled = datetime.fromisoformat(
                    data["scheduled_time"].replace("Z", "+00:00")
                )
                if scheduled < datetime.now(timezone.utc):
                    warnings.append(
                        {
                            "message": "Scheduled time is in the past",
                            "path": ["scheduled_time"],
                            "type": "scheduling_warning",
                        }
                    )
            except (ValueError, AttributeError):
                pass  # Invalid format will be caught by schema validation


class FileValidator:
    """File-level validation handler for JSON and JSONL files."""

    def __init__(self, schema_validator: SchemaValidator):
        self.schema_validator = schema_validator
        self.logger = schema_validator.logger

    def validate_file(
        self, file_path: Path, schema_name: str
    ) -> list[ValidationResult]:
        """Validate a single JSON or JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Validating {file_path} against {schema_name} schema")

        if file_path.suffix == ".jsonl":
            return self._validate_jsonl_file(file_path, schema_name)
        if file_path.suffix == ".json":
            return self._validate_json_file(file_path, schema_name)
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _validate_json_file(
        self, file_path: Path, schema_name: str
    ) -> list[ValidationResult]:
        """Validate a single JSON file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            result = self.schema_validator.validate_data(
                data, schema_name, str(file_path)
            )
            return [result]

        except json.JSONDecodeError as e:
            return [
                ValidationResult(
                    file_path=str(file_path),
                    schema_name=schema_name,
                    is_valid=False,
                    errors=[
                        {
                            "message": f"JSON decode error: {e.msg}",
                            "path": [],
                            "schema_path": [],
                            "invalid_value": None,
                            "validator": "json_decode",
                            "validator_value": None,
                            "line_number": e.lineno,
                            "column": e.colno,
                        }
                    ],
                    warnings=[],
                    line_number=e.lineno,
                )
            ]

    def _validate_jsonl_file(
        self, file_path: Path, schema_name: str
    ) -> list[ValidationResult]:
        """Validate a JSONL file (one JSON object per line)."""
        results = []

        try:
            with jsonlines.open(file_path, mode="r") as reader:
                for line_num, data in enumerate(reader, 1):
                    result = self.schema_validator.validate_data(
                        data,
                        schema_name,
                        str(file_path),
                        line_number=line_num,
                        record_index=line_num - 1,
                    )
                    results.append(result)

        except (json.JSONDecodeError, jsonlines.InvalidLineError) as e:
            # Handle parsing errors
            line_num = getattr(e, "lineno", None) or 1
            results.append(
                ValidationResult(
                    file_path=str(file_path),
                    schema_name=schema_name,
                    is_valid=False,
                    errors=[
                        {
                            "message": f"JSONL parsing error: {str(e)}",
                            "path": [],
                            "schema_path": [],
                            "invalid_value": None,
                            "validator": "jsonl_parse",
                            "validator_value": None,
                        }
                    ],
                    warnings=[],
                    line_number=line_num,
                )
            )

        return results


class AutoFixer:
    """Automatic schema violation fixer."""

    def __init__(self, schema_validator: SchemaValidator):
        self.schema_validator = schema_validator
        self.logger = schema_validator.logger

    def auto_fix_data(
        self, data: Any, schema_name: str
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Attempt to automatically fix common schema violations."""
        if schema_name not in self.schema_validator.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")

        fixes_applied = []
        fixed_data = self._deep_copy(data)

        # Apply type fixes
        fixed_data = self._fix_type_errors(fixed_data, schema_name, fixes_applied)

        # Apply format fixes
        fixed_data = self._fix_format_errors(fixed_data, schema_name, fixes_applied)

        # Apply missing field fixes
        fixed_data = self._fix_missing_fields(fixed_data, schema_name, fixes_applied)

        # Apply constraint fixes
        fixed_data = self._fix_constraint_violations(
            fixed_data, schema_name, fixes_applied
        )

        return fixed_data, fixes_applied

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy an object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj

    def _fix_type_errors(
        self, data: Any, schema_name: str, fixes: list[dict[str, Any]]
    ) -> Any:
        """Fix common type conversion errors."""
        if not isinstance(data, dict):
            return data

        schema = self.schema_validator.schemas[schema_name]
        properties = schema.get("properties", {})

        for field, field_schema in properties.items():
            if field in data:
                data[field] = self._convert_field_type(
                    data[field], field_schema, field, fixes
                )

        return data

    def _convert_field_type(
        self,
        value: Any,
        field_schema: dict[str, Any],
        field_name: str,
        fixes: list[dict[str, Any]],
    ) -> Any:
        """Convert a field to the correct type."""
        expected_type = field_schema.get("type")

        if expected_type == "string" and not isinstance(value, str):
            if value is not None:
                fixed_value = str(value)
                fixes.append(
                    {
                        "type": "type_conversion",
                        "field": field_name,
                        "original_value": value,
                        "fixed_value": fixed_value,
                        "description": f"Converted {type(value).__name__} to string",
                    }
                )
                return fixed_value

        elif expected_type == "integer" and not isinstance(value, int):
            if isinstance(value, float | str):
                try:
                    fixed_value = int(float(value))
                    fixes.append(
                        {
                            "type": "type_conversion",
                            "field": field_name,
                            "original_value": value,
                            "fixed_value": fixed_value,
                            "description": f"Converted {type(value).__name__} to integer",
                        }
                    )
                    return fixed_value
                except (ValueError, TypeError):
                    pass

        elif expected_type == "number" and not isinstance(value, int | float):
            if isinstance(value, str):
                try:
                    fixed_value = float(value)
                    fixes.append(
                        {
                            "type": "type_conversion",
                            "field": field_name,
                            "original_value": value,
                            "fixed_value": fixed_value,
                            "description": "Converted string to number",
                        }
                    )
                    return fixed_value
                except ValueError:
                    pass

        elif expected_type == "array" and not isinstance(value, list):
            if isinstance(value, str):
                # Try to parse as JSON array
                try:
                    fixed_value = json.loads(value)
                    if isinstance(fixed_value, list):
                        fixes.append(
                            {
                                "type": "type_conversion",
                                "field": field_name,
                                "original_value": value,
                                "fixed_value": fixed_value,
                                "description": "Parsed string as JSON array",
                            }
                        )
                        return fixed_value
                except json.JSONDecodeError:
                    # Try splitting on common delimiters
                    if "," in value:
                        fixed_value = [item.strip() for item in value.split(",")]
                        fixes.append(
                            {
                                "type": "type_conversion",
                                "field": field_name,
                                "original_value": value,
                                "fixed_value": fixed_value,
                                "description": "Split string into array on commas",
                            }
                        )
                        return fixed_value

        return value

    def _fix_format_errors(
        self, data: Any, schema_name: str, fixes: list[dict[str, Any]]
    ) -> Any:
        """Fix common format errors."""
        if not isinstance(data, dict):
            return data

        # Fix UUID format issues
        if "id" in data and isinstance(data["id"], str):
            if not self._is_valid_uuid(data["id"]):
                # Generate a new UUID if the current one is invalid
                fixed_value = str(uuid4())
                fixes.append(
                    {
                        "type": "format_fix",
                        "field": "id",
                        "original_value": data["id"],
                        "fixed_value": fixed_value,
                        "description": "Generated new UUID for invalid ID",
                    }
                )
                data["id"] = fixed_value

        # Fix datetime format issues
        for field in [
            "timestamp",
            "created_at",
            "scheduled_time",
            "actual_publish_time",
        ]:
            if field in data and isinstance(data[field], str):
                fixed_datetime = self._fix_datetime_format(data[field])
                if fixed_datetime != data[field]:
                    fixes.append(
                        {
                            "type": "format_fix",
                            "field": field,
                            "original_value": data[field],
                            "fixed_value": fixed_datetime,
                            "description": "Fixed datetime format to ISO 8601",
                        }
                    )
                    data[field] = fixed_datetime

        return data

    def _fix_missing_fields(
        self, data: Any, schema_name: str, fixes: list[dict[str, Any]]
    ) -> Any:
        """Add missing required fields with sensible defaults."""
        if not isinstance(data, dict):
            return data

        schema = self.schema_validator.schemas[schema_name]
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required_fields:
            if field not in data:
                default_value = self._generate_default_value(
                    field, properties.get(field, {})
                )
                if default_value is not None:
                    data[field] = default_value
                    fixes.append(
                        {
                            "type": "missing_field",
                            "field": field,
                            "original_value": None,
                            "fixed_value": default_value,
                            "description": "Added missing required field with default value",
                        }
                    )

        return data

    def _fix_constraint_violations(
        self, data: Any, schema_name: str, fixes: list[dict[str, Any]]
    ) -> Any:
        """Fix constraint violations like min/max values."""
        if not isinstance(data, dict):
            return data

        schema = self.schema_validator.schemas[schema_name]
        properties = schema.get("properties", {})

        for field, field_schema in properties.items():
            if field in data:
                fixed_value = self._fix_field_constraints(
                    data[field], field_schema, field, fixes
                )
                if fixed_value != data[field]:
                    data[field] = fixed_value

        return data

    def _fix_field_constraints(
        self,
        value: Any,
        field_schema: dict[str, Any],
        field_name: str,
        fixes: list[dict[str, Any]],
    ) -> Any:
        """Fix constraint violations for a specific field."""
        if isinstance(value, int | float):
            # Fix minimum/maximum constraints
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")

            if minimum is not None and value < minimum:
                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": value,
                        "fixed_value": minimum,
                        "description": f"Clamped value to minimum: {minimum}",
                    }
                )
                return minimum

            if maximum is not None and value > maximum:
                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": value,
                        "fixed_value": maximum,
                        "description": f"Clamped value to maximum: {maximum}",
                    }
                )
                return maximum

        elif isinstance(value, str):
            # Fix length constraints
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")

            if min_length is not None and len(value) < min_length:
                # Pad with appropriate content
                if field_name == "source_post_id":
                    fixed_value = value + "_" + str(uuid4())[:8]
                else:
                    fixed_value = value + "." * (min_length - len(value))

                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": value,
                        "fixed_value": fixed_value,
                        "description": f"Padded string to meet minimum length: {min_length}",
                    }
                )
                return fixed_value

            if max_length is not None and len(value) > max_length:
                fixed_value = value[:max_length]
                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": value,
                        "fixed_value": fixed_value,
                        "description": f"Truncated string to maximum length: {max_length}",
                    }
                )
                return fixed_value

        elif isinstance(value, list):
            # Fix array constraints
            min_items = field_schema.get("minItems")
            max_items = field_schema.get("maxItems")

            if min_items is not None and len(value) < min_items:
                # Add default items
                item_schema = field_schema.get("items", {})
                default_item = self._generate_default_value("item", item_schema)

                while len(value) < min_items:
                    value.append(default_item)

                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": len(value) - (min_items - len(value)),
                        "fixed_value": len(value),
                        "description": f"Added items to meet minimum length: {min_items}",
                    }
                )

            if max_items is not None and len(value) > max_items:
                original_length = len(value)
                value = value[:max_items]
                fixes.append(
                    {
                        "type": "constraint_fix",
                        "field": field_name,
                        "original_value": original_length,
                        "fixed_value": len(value),
                        "description": f"Truncated array to maximum length: {max_items}",
                    }
                )
                return value

        return value

    def _generate_default_value(
        self, field_name: str, field_schema: dict[str, Any]
    ) -> Any:
        """Generate a sensible default value for a field."""
        field_type = field_schema.get("type", "string")

        if field_type == "string":
            if field_schema.get("format") == "uuid":
                return str(uuid4())
            if field_schema.get("format") == "date-time":
                return datetime.now(timezone.utc).isoformat()
            if field_name in ["provider"]:
                return "mlx::default-model"
            if field_name in ["source_post_id"]:
                return f"auto_generated_{uuid4().hex[:8]}"
            return f"default_{field_name}"

        if field_type == "number":
            minimum = field_schema.get("minimum", 0.0)
            maximum = field_schema.get("maximum", 1.0)
            return (minimum + maximum) / 2

        if field_type == "integer":
            minimum = field_schema.get("minimum", 0)
            maximum = field_schema.get("maximum", 100)
            return (minimum + maximum) // 2

        if field_type == "boolean":
            return False

        if field_type == "array":
            min_items = field_schema.get("minItems", 1)
            item_schema = field_schema.get("items", {})
            return [
                self._generate_default_value("item", item_schema)
                for _ in range(min_items)
            ]

        if field_type == "object":
            obj = {}
            required_fields = field_schema.get("required", [])
            properties = field_schema.get("properties", {})

            for req_field in required_fields:
                if req_field in properties:
                    obj[req_field] = self._generate_default_value(
                        req_field, properties[req_field]
                    )

            return obj

        return None

    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            UUID(uuid_string)
            return True
        except ValueError:
            return False

    def _fix_datetime_format(self, datetime_string: str) -> str:
        """Fix common datetime format issues."""
        # Common patterns to fix
        patterns = [
            # Add Z suffix for timezone.utc if missing
            (r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$", r"\1Z"),
            # Replace space with T
            (r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})$", r"\1T\2Z"),
            # Add seconds if missing
            (r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z?$", r"\1:00Z"),
        ]

        result = datetime_string
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        return result


class BatchValidator:
    """Batch validation operations for multiple files."""

    def __init__(self, schema_validator: SchemaValidator):
        self.schema_validator = schema_validator
        self.file_validator = FileValidator(schema_validator)
        self.auto_fixer = AutoFixer(schema_validator)
        self.logger = schema_validator.logger

    def validate_directory(
        self,
        directory: Path,
        schema_mapping: dict[str, str] | None = None,
        recursive: bool = True,
    ) -> ValidationReport:
        """Validate all JSON/JSONL files in a directory."""
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        self.logger.info(f"Starting batch validation of {directory}")

        # Find all JSON/JSONL files
        patterns = ["*.json", "*.jsonl"]
        files = []

        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))

        if not files:
            self.logger.warning(f"No JSON/JSONL files found in {directory}")

        # Determine schema for each file
        results = []
        schema_coverage = defaultdict(int)
        error_summary = defaultdict(int)
        auto_fixes_applied = []

        for file_path in files:
            schema_name = self._determine_schema(file_path, schema_mapping)
            if not schema_name:
                self.logger.warning(f"Could not determine schema for {file_path}")
                continue

            schema_coverage[schema_name] += 1

            try:
                file_results = self.file_validator.validate_file(file_path, schema_name)
                results.extend(file_results)

                # Count errors by type
                for result in file_results:
                    for error in result.errors:
                        error_summary[error.get("validator", "unknown")] += 1

            except Exception as e:
                self.logger.exception(f"Error validating {file_path}: {e}")
                results.append(
                    ValidationResult(
                        file_path=str(file_path),
                        schema_name=schema_name,
                        is_valid=False,
                        errors=[
                            {
                                "message": f"Validation failed: {str(e)}",
                                "path": [],
                                "schema_path": [],
                                "invalid_value": None,
                                "validator": "exception",
                                "validator_value": None,
                            }
                        ],
                        warnings=[],
                    )
                )

        # Generate report
        total_files = len({r.file_path for r in results})
        total_records = len(results)
        valid_files = len({r.file_path for r in results if r.is_valid})
        invalid_files = total_files - valid_files
        files_with_warnings = len({r.file_path for r in results if r.warnings})

        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_files=total_files,
            total_records=total_records,
            valid_files=valid_files,
            invalid_files=invalid_files,
            files_with_warnings=files_with_warnings,
            schema_coverage=dict(schema_coverage),
            error_summary=dict(error_summary),
            results=results,
            auto_fixes_applied=auto_fixes_applied,
        )

        self.logger.info(
            f"Batch validation complete: {valid_files}/{total_files} files valid"
        )

        return report

    def _determine_schema(
        self, file_path: Path, schema_mapping: dict[str, str] | None = None
    ) -> str | None:
        """Determine which schema to use for a file."""
        if schema_mapping:
            # Check explicit mapping
            for pattern, schema in schema_mapping.items():
                if pattern in str(file_path):
                    return schema

        # Infer from filename
        filename = file_path.stem.lower()
        available_schemas = self.schema_validator.get_available_schemas()

        for schema in available_schemas:
            if schema in filename:
                return schema

        # Default to first available schema if only one
        if len(available_schemas) == 1:
            return available_schemas[0]

        return None

    def fix_files(
        self,
        files: list[Path],
        output_dir: Path,
        schema_mapping: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Auto-fix multiple files and save to output directory."""
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        fix_results = []

        for file_path in files:
            schema_name = self._determine_schema(file_path, schema_mapping)
            if not schema_name:
                continue

            try:
                fix_result = self._fix_single_file(file_path, schema_name, output_dir)
                fix_results.append(fix_result)
            except Exception as e:
                self.logger.exception(f"Error fixing {file_path}: {e}")
                fix_results.append(
                    {
                        "file_path": str(file_path),
                        "schema_name": schema_name,
                        "success": False,
                        "error": str(e),
                        "fixes_applied": [],
                    }
                )

        return fix_results

    def _fix_single_file(
        self, file_path: Path, schema_name: str, output_dir: Path
    ) -> dict[str, Any]:
        """Fix a single file and save to output directory."""
        output_file = output_dir / file_path.name

        if file_path.suffix == ".jsonl":
            return self._fix_jsonl_file(file_path, schema_name, output_file)
        return self._fix_json_file(file_path, schema_name, output_file)

    def _fix_json_file(
        self, file_path: Path, schema_name: str, output_file: Path
    ) -> dict[str, Any]:
        """Fix a JSON file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        fixed_data, fixes_applied = self.auto_fixer.auto_fix_data(data, schema_name)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)

        return {
            "file_path": str(file_path),
            "output_path": str(output_file),
            "schema_name": schema_name,
            "success": True,
            "fixes_applied": fixes_applied,
        }

    def _fix_jsonl_file(
        self, file_path: Path, schema_name: str, output_file: Path
    ) -> dict[str, Any]:
        """Fix a JSONL file."""
        all_fixes = []

        with (
            jsonlines.open(file_path, mode="r") as reader,
            jsonlines.open(output_file, mode="w") as writer,
        ):

            for line_num, data in enumerate(reader, 1):
                fixed_data, fixes_applied = self.auto_fixer.auto_fix_data(
                    data, schema_name
                )

                # Add line number to fixes
                for fix in fixes_applied:
                    fix["line_number"] = line_num

                all_fixes.extend(fixes_applied)
                writer.write(fixed_data)

        return {
            "file_path": str(file_path),
            "output_path": str(output_file),
            "schema_name": schema_name,
            "success": True,
            "fixes_applied": all_fixes,
        }


class CLI:
    """Command-line interface for the schema validator."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="JSON schema validator for REER × DSPy × MLX project",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Validate a single file
  python tools/schema_check.py validate --file data/traces.json --schema traces

  # Batch validate all files in a directory
  python tools/schema_check.py batch --directory data/ --recursive

  # Auto-fix files with schema violations
  python tools/schema_check.py fix --file data/invalid.json --schema traces --output fixed/

  # Generate validation report
  python tools/schema_check.py report --directory data/ --output validation_report.json

  # List available schemas
  python tools/schema_check.py list-schemas
""",
        )

        parser.add_argument(
            "--schema-dir",
            type=Path,
            help="Directory containing schema files (auto-detected if not specified)",
        )

        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Validate command
        validate_parser = subparsers.add_parser(
            "validate", help="Validate a single file"
        )
        validate_parser.add_argument(
            "--file", type=Path, required=True, help="File to validate"
        )
        validate_parser.add_argument(
            "--schema", required=True, help="Schema name to validate against"
        )

        # Batch command
        batch_parser = subparsers.add_parser(
            "batch", help="Batch validate multiple files"
        )
        batch_parser.add_argument(
            "--directory", type=Path, required=True, help="Directory to scan"
        )
        batch_parser.add_argument(
            "--recursive", action="store_true", help="Scan recursively"
        )
        batch_parser.add_argument(
            "--schema-mapping",
            type=str,
            help="JSON mapping of file patterns to schemas",
        )

        # Fix command
        fix_parser = subparsers.add_parser("fix", help="Auto-fix schema violations")
        fix_parser.add_argument("--file", type=Path, help="Single file to fix")
        fix_parser.add_argument(
            "--directory", type=Path, help="Directory of files to fix"
        )
        fix_parser.add_argument(
            "--schema", help="Schema name (required for single file)"
        )
        fix_parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory for fixed files",
        )
        fix_parser.add_argument(
            "--schema-mapping",
            type=str,
            help="JSON mapping of file patterns to schemas",
        )

        # Report command
        report_parser = subparsers.add_parser(
            "report", help="Generate validation report"
        )
        report_parser.add_argument(
            "--directory", type=Path, required=True, help="Directory to validate"
        )
        report_parser.add_argument(
            "--output", type=Path, required=True, help="Output file for report"
        )
        report_parser.add_argument(
            "--recursive", action="store_true", help="Scan recursively"
        )
        report_parser.add_argument(
            "--schema-mapping",
            type=str,
            help="JSON mapping of file patterns to schemas",
        )

        # List schemas command
        subparsers.add_parser("list-schemas", help="List available schemas")

        return parser

    def run(self, args: list[str] | None = None) -> int:
        """Run the CLI with the given arguments."""
        parsed_args = self.parser.parse_args(args)

        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            if parsed_args.command == "validate":
                return self._cmd_validate(parsed_args)
            if parsed_args.command == "batch":
                return self._cmd_batch(parsed_args)
            if parsed_args.command == "fix":
                return self._cmd_fix(parsed_args)
            if parsed_args.command == "report":
                return self._cmd_report(parsed_args)
            if parsed_args.command == "list-schemas":
                return self._cmd_list_schemas(parsed_args)
            self.parser.print_help()
            return 1

        except Exception:
            if parsed_args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    def _cmd_validate(self, args) -> int:
        """Handle validate command."""
        validator = SchemaValidator(args.schema_dir)
        file_validator = FileValidator(validator)

        results = file_validator.validate_file(args.file, args.schema)

        # Print results
        for result in results:

            if result.errors:
                for error in result.errors:
                    (
                        ".".join(str(p) for p in error["path"])
                        if error["path"]
                        else "root"
                    )

            if result.warnings:
                for warning in result.warnings:
                    (
                        ".".join(str(p) for p in warning["path"])
                        if warning["path"]
                        else "root"
                    )

        return 0 if all(r.is_valid for r in results) else 1

    def _cmd_batch(self, args) -> int:
        """Handle batch command."""
        validator = SchemaValidator(args.schema_dir)
        batch_validator = BatchValidator(validator)

        schema_mapping = None
        if args.schema_mapping:
            schema_mapping = json.loads(args.schema_mapping)

        report = batch_validator.validate_directory(
            args.directory, schema_mapping, args.recursive
        )

        # Print summary

        if report.schema_coverage:
            for _schema, _count in report.schema_coverage.items():
                pass

        if report.error_summary:
            for _error_type, _count in report.error_summary.items():
                pass

        return 0 if report.invalid_files == 0 else 1

    def _cmd_fix(self, args) -> int:
        """Handle fix command."""
        validator = SchemaValidator(args.schema_dir)
        batch_validator = BatchValidator(validator)

        schema_mapping = None
        if args.schema_mapping:
            schema_mapping = json.loads(args.schema_mapping)

        files = []
        if args.file:
            if not args.schema:
                return 1
            files = [args.file]
            # Create a schema mapping for the single file
            schema_mapping = {str(args.file): args.schema}
        elif args.directory:
            files = list(args.directory.rglob("*.json")) + list(
                args.directory.rglob("*.jsonl")
            )
        else:
            return 1

        fix_results = batch_validator.fix_files(files, args.output, schema_mapping)

        # Print results
        total_fixes = 0
        for result in fix_results:
            if result["success"]:
                total_fixes += len(result["fixes_applied"])
            else:
                pass

        return 0

    def _cmd_report(self, args) -> int:
        """Handle report command."""
        validator = SchemaValidator(args.schema_dir)
        batch_validator = BatchValidator(validator)

        schema_mapping = None
        if args.schema_mapping:
            schema_mapping = json.loads(args.schema_mapping)

        report = batch_validator.validate_directory(
            args.directory, schema_mapping, args.recursive
        )

        # Save report to file
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

        return 0 if report.invalid_files == 0 else 1

    def _cmd_list_schemas(self, args) -> int:
        """Handle list-schemas command."""
        validator = SchemaValidator(args.schema_dir)
        schemas = validator.get_available_schemas()

        for schema in sorted(schemas):
            schema_info = validator.schemas[schema]
            schema_info.get("title", schema)
            schema_info.get("version", "unknown")
            schema_info.get("description", "No description")

        return 0


def main():
    """Main entry point."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
