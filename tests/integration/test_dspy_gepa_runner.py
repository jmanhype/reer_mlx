import pytest


@pytest.mark.integration
def test_run_gepa_returns_module(monkeypatch):
    dspy = pytest.importorskip("dspy")

    # Import runner after ensuring dspy is available
    import dspy_program.gepa_runner as runner

    class FakeGEPA:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, student, *, trainset, valset=None):  # noqa: ARG002
            # Return the student program unchanged
            return student

    # Patch DSPy GEPA class used by the runner
    monkeypatch.setattr(runner, "DSPyGEPA", FakeGEPA)

    # Avoid real LM/config in tests
    monkeypatch.setattr(dspy.settings, "configure", lambda **_: None)
    monkeypatch.setattr(dspy, "LM", lambda *_, **__: (lambda x: ("ok", {})))

    # Minimal trainset (2-3 tasks)
    train = [
        {"topic": "AI productivity", "audience": "developers"},
        {"topic": "Startup growth", "audience": "founders"},
        {"topic": "MLX tips", "audience": "apple silicon users"},
    ]

    optimized = runner.run_gepa(
        train,
        gen_model="dummy",
        reflection_model="dummy",
        auto="light",
        track_stats=False,
    )

    # Assert a DSPy Module is returned and has predictors
    assert hasattr(optimized, "named_predictors")
    preds = dict(optimized.named_predictors())
    assert isinstance(preds, dict)
    assert len(preds) >= 1
