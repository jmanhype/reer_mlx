import json
from pathlib import Path


def test_evaluate_handles_missing_fields(tmp_path: Path):
    # Create minimal optimized program JSON (no detailed_results)
    prog = {"predictors": {"compose": "Instruction text."}}
    prog_file = tmp_path / "optimized_program.json"
    prog_file.write_text(json.dumps(prog))

    # Create dummy test_data file (content unused, only existence checked)
    test_data = tmp_path / "test_data.json"
    test_data.write_text("[]")

    # Import evaluate command and invoke directly
    from scripts.social_gepa import evaluate

    # Should not raise even with missing detailed_results
    evaluate(
        model_file=prog_file,
        test_data=test_data,
        output_file=None,
        show_instructions=True,
        metrics=None,
        platforms=None,
    )
