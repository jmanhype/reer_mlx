# Contract Tests for REER × DSPy × MLX JSON Schemas

## Overview

This directory contains comprehensive contract tests for the three core JSON schemas in the REER (Reverse Engineering for Engagement Rates) project:

- **T006**: `test_trace_schema.py` - Tests for `traces.schema.json`
- **T007**: `test_candidate_schema.py` - Tests for `candidate.schema.json`  
- **T008**: `test_timeline_schema.py` - Tests for `timeline.schema.json`

## Test Design Philosophy

These tests follow the **London School TDD (Test-Driven Development)** approach with:

- **Mock-first development**: Tests use mocks to define contracts and interactions
- **Behavior verification**: Focus on HOW objects collaborate, not just state
- **Outside-in development**: Start with contract definition and work inward
- **Red-Green-Refactor**: Tests MUST fail initially (RED phase) before implementation

## Expected Failures (RED Phase)

**These tests are DESIGNED to fail initially** as part of the TDD cycle. Current failing tests indicate:

### Format Validation Failures
- UUID format validation not enforced (requires `FormatChecker`)
- ISO 8601 datetime format validation not enforced
- URI format validation not working
- Regex pattern validation issues

### Mock Interaction Failures
- Mock contracts for jsonschema validation behavior
- Schema validator interaction testing
- Error handling contract verification

### Schema Structure Tests
Some tests pass because they validate schema structure itself, which exists.

## Running the Tests

### Basic Test Run
```bash
# Run all contract tests
python -m pytest tests/contract/ -v

# Run specific schema tests
python -m pytest tests/contract/test_trace_schema.py -v
python -m pytest tests/contract/test_candidate_schema.py -v
python -m pytest tests/contract/test_timeline_schema.py -v
```

### Format Validation (Implementation Phase)
When implementing validation, you'll need to use `FormatChecker`:

```python
from jsonschema import validate, FormatChecker

# Enable format validation
validate(data, schema, format_checker=FormatChecker())
```

### Contract Markers
Tests are marked for different contract types:

```bash
# Run only interaction contract tests
python -m pytest tests/contract/ -v -m "interaction"

# Run only validation contract tests  
python -m pytest tests/contract/ -v -m "validation"
```

## Test Categories

### 1. Schema Structure Tests
- Verify JSON Schema Draft 7 compliance
- Check required metadata fields
- Validate schema structure itself

### 2. Required Fields Validation
- Test all required fields are enforced
- Verify nested object requirements
- Check missing field error handling

### 3. Type Checking Tests
- UUID format validation
- ISO 8601 datetime validation
- Numeric range constraints
- String pattern matching
- Array validation rules

### 4. Edge Cases and Invalid Data
- Boundary value testing
- Unicode string handling
- Large array processing
- Null value rejection
- Additional properties restriction

### 5. Mock-Based Interaction Tests (London School)
- Validator creation contracts
- Validation call interactions
- Error handling behavior
- Success path verification

### 6. Schema Version Compatibility
- Semantic versioning checks
- Schema ID uniqueness
- Compatibility tracking

## Implementation Guidance

### Phase 1: Fix Format Validation
1. Update validation calls to use `FormatChecker()`
2. Ensure UUID validation works properly
3. Fix datetime format checking
4. Resolve regex pattern issues

### Phase 2: Implement Schema Validators
1. Create validator classes for each schema
2. Implement proper error handling
3. Add comprehensive logging
4. Create factory methods

### Phase 3: Integration
1. Wire validators into main application
2. Add configuration for validation rules
3. Implement performance optimization
4. Add monitoring and metrics

## Test Data

Each test suite includes fixtures for:
- **Valid data samples**: Should pass validation
- **Invalid data samples**: Should fail validation
- **Edge cases**: Boundary conditions
- **Unicode samples**: International character support

## Dependencies

Required packages:
- `pytest>=8.0.0`
- `jsonschema>=4.19.0` 
- `pytest-mock>=3.12.0`

## Contract Testing Benefits

1. **Early Error Detection**: Catch schema issues before implementation
2. **Clear API Contracts**: Define expected behavior explicitly  
3. **Regression Prevention**: Ensure changes don't break contracts
4. **Documentation**: Tests serve as executable specifications
5. **Quality Assurance**: Comprehensive validation coverage

## Next Steps

1. **GREEN Phase**: Implement validators to make tests pass
2. **REFACTOR Phase**: Optimize and clean up implementation
3. **Integration**: Connect validators to main application
4. **Production**: Deploy with comprehensive validation

Remember: **Failing tests are GOOD** in the RED phase - they define what needs to be implemented!