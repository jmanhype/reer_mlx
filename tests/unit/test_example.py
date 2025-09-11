"""Example unit test to verify testing setup."""

import pytest


def test_example_passes():
    """Test that passes to verify testing setup."""
    assert True


def test_example_with_fixture(sample_content):
    """Test using a fixture to verify fixture setup."""
    assert sample_content["title"] == "Test Post"
    assert "test" in sample_content["tags"]


@pytest.mark.unit
def test_marked_as_unit():
    """Test explicitly marked as unit test."""
    assert 1 + 1 == 2


class TestExampleClass:
    """Example test class."""

    def test_method_in_class(self):
        """Test method within a class."""
        assert "hello".upper() == "HELLO"

    def test_another_method(self, sample_config):
        """Test using configuration fixture."""
        assert sample_config["mlx"]["model_path"] == "/test/models/llama-3.2-3b"
