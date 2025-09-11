# Schema Validation Tool

A comprehensive JSON schema validation utility for the REER × DSPy × MLX project that validates JSON/JSONL files against schemas, provides detailed error reporting, auto-fixes common violations, and generates validation reports.

## Features

- **Multi-Schema Support**: Validates against traces, timeline, and candidate schemas
- **Batch Validation**: Process multiple files and directories recursively  
- **Detailed Error Reporting**: Line-by-line errors with schema paths and validation details
- **Auto-Fix Capabilities**: Automatically fixes common schema violations
- **Report Generation**: Comprehensive validation reports in JSON format
- **CLI Interface**: Easy-to-use command-line interface
- **Integration Ready**: Integrates with existing contract tests

## Installation

Install required dependencies:

```bash
pip install jsonschema jsonlines
```

## Usage

### List Available Schemas

```bash
python tools/schema_check.py list-schemas
```

Output:
```
Available schemas:
  candidate (v1.0.0)
    Title: Post Candidate
    Description: Generated content awaiting optimization and scheduling
  timeline (v1.0.0)
    Title: Timeline Entry
    Description: Scheduled post with publication metadata
  traces (v1.0.0)
    Title: REER Trace
    Description: Append-only record capturing strategy extraction events from historical posts
```

### Validate a Single File

```bash
# Validate a JSON file
python tools/schema_check.py validate --file data/trace.json --schema traces

# Validate a JSONL file
python tools/schema_check.py validate --file data/traces.jsonl --schema traces
```

Output:
```
File: data/trace.json
Schema: traces
Valid: True
```

For invalid files:
```
File: data/invalid_trace.json
Schema: traces
Valid: False
Errors (3):
  - seed_params.length: '280' is not of type 'integer'
  - score: 1.5 is greater than the maximum of 1.0
  - provider: 'invalid-provider' does not match '^(mlx|dspy)::.+$'
Warnings (1):
  - id: 'not-a-valid-uuid' is not a 'uuid'
```

### Batch Validation

```bash
# Validate all JSON/JSONL files in a directory
python tools/schema_check.py batch --directory data/ --recursive

# With custom schema mapping
python tools/schema_check.py batch --directory data/ --schema-mapping '{"trace": "traces", "candidate": "candidate"}'
```

Output:
```
Validation Report
================
Total files: 5
Total records: 12
Valid files: 3
Invalid files: 2
Files with warnings: 1

Schema coverage:
  traces: 3 files
  candidate: 2 files

Error summary:
  type: 5
  minimum: 3
  pattern: 2
```

### Auto-Fix Schema Violations

```bash
# Fix a single file
python tools/schema_check.py fix --file data/invalid.json --schema traces --output fixed/

# Fix all files in a directory
python tools/schema_check.py fix --directory data/ --output fixed/
```

Output:
```
Fixed: data/invalid.json -> fixed/invalid.json
  Applied 4 fixes

Total fixes applied: 4
```

### Generate Validation Reports

```bash
python tools/schema_check.py report --directory data/ --output validation_report.json --recursive
```

Creates a detailed JSON report with:
- Summary statistics
- Error breakdowns by type
- Individual file validation results
- Line-by-line error details
- Performance metrics

## Auto-Fix Capabilities

The tool can automatically fix many common schema violations:

### Type Conversions
- String numbers to integers/floats
- String arrays to proper arrays
- Invalid format corrections

### Constraint Violations  
- Clamp values to min/max ranges
- Truncate strings to max length
- Pad strings to min length
- Fill arrays to min items

### Missing Fields
- Add required fields with sensible defaults
- Generate UUIDs for ID fields
- Create ISO timestamps for date fields
- Add default nested objects

### Format Fixes
- Generate valid UUIDs
- Fix datetime formats to ISO 8601
- Correct provider patterns

## Schema Files

The tool automatically discovers schemas from:
```
specs/001-reer_mlx/contracts/
├── traces.schema.json
├── timeline.schema.json  
└── candidate.schema.json
```

## Integration with Tests

The validation tool integrates with existing contract tests:

```python
# Import the validator in your tests
from tools.schema_check import SchemaValidator, FileValidator

# Use in test fixtures
@pytest.fixture
def schema_validator():
    return SchemaValidator()

def test_custom_validation(schema_validator):
    result = schema_validator.validate_data(test_data, "traces")
    assert result.is_valid
```

## Error Types and Meanings

### Validation Errors
- **type**: Wrong data type (string vs integer)
- **minimum/maximum**: Value outside allowed range
- **minLength/maxLength**: String length violations
- **minItems/maxItems**: Array size violations  
- **pattern**: String doesn't match required pattern
- **required**: Missing required field
- **additionalProperties**: Extra fields not allowed

### Warnings
- **format**: Format violations (UUID, date-time) that don't break functionality
- **performance**: Low confidence scores or engagement rates
- **length**: Content approaching limits

## Configuration

### Schema Directory
Override the default schema directory:
```bash
python tools/schema_check.py --schema-dir /path/to/schemas validate ...
```

### Schema Mapping
Map file patterns to schemas:
```bash
--schema-mapping '{"trace": "traces", "_candidate": "candidate", "timeline": "timeline"}'
```

The tool will match files containing these patterns to the specified schemas.

## Output Formats

### Validation Results
```json
{
  "file_path": "data/trace.json",
  "schema_name": "traces", 
  "is_valid": false,
  "errors": [
    {
      "message": "1.5 is greater than the maximum of 1.0",
      "path": ["score"],
      "schema_path": ["properties", "score", "maximum"],
      "invalid_value": 1.5,
      "validator": "maximum",
      "validator_value": 1.0
    }
  ],
  "warnings": [],
  "line_number": null,
  "validation_time": 0.002
}
```

### Validation Reports
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "total_files": 5,
  "total_records": 12,
  "valid_files": 3,
  "invalid_files": 2,
  "files_with_warnings": 1,
  "schema_coverage": {
    "traces": 3,
    "candidate": 2
  },
  "error_summary": {
    "type": 5,
    "minimum": 3
  },
  "results": [...],
  "auto_fixes_applied": [...]
}
```

## Performance

The tool is optimized for:
- **Large Files**: Efficiently processes JSONL files with thousands of records
- **Batch Operations**: Parallel validation of multiple files
- **Memory Usage**: Streaming processing for large datasets
- **Speed**: Fast validation with detailed error reporting

Typical performance:
- ~1000 records/second for validation
- ~500 records/second for auto-fixing
- Line-by-line error tracking with minimal overhead

## Error Handling

The tool handles various error conditions gracefully:
- **Malformed JSON**: Reports parse errors with line numbers
- **Missing Files**: Clear error messages for file not found
- **Invalid Schemas**: Validates schema files themselves
- **Permission Issues**: Helpful messages for file access problems

## Contributing

To extend the schema validator:

1. **Add New Schemas**: Place `.schema.json` files in the contracts directory
2. **Extend Auto-Fixes**: Add new fix patterns in `AutoFixer` class
3. **Add Validation Logic**: Extend warning checks in `_check_for_warnings`
4. **Update Tests**: Add integration tests for new functionality

## Examples

See the `test_data/` directory for example files:
- `valid_trace.json`: Valid trace data
- `invalid_trace.json`: Invalid data with multiple violations  
- `sample_traces.jsonl`: Mixed valid/invalid JSONL data

## Troubleshooting

### Common Issues

1. **Schema not found**: Ensure schema files exist in `specs/001-reer_mlx/contracts/`
2. **File not processed**: Check schema mapping patterns match your filenames
3. **Auto-fix not working**: Some complex violations may need manual fixing
4. **Memory issues with large files**: Use `--recursive` for directory processing

### Debug Mode
Use Python's verbose flag for detailed debugging:
```bash
python -v tools/schema_check.py validate --file data.json --schema traces
```

### Log Output
The tool provides detailed logging for troubleshooting:
```
2024-01-15 10:30:00 - schema_validator - INFO - Loading schemas from /path/to/contracts
2024-01-15 10:30:00 - schema_validator - INFO - Loaded schema: traces  
2024-01-15 10:30:00 - schema_validator - INFO - Validating data.json against traces schema
```