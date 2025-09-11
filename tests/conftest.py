"""Pytest configuration and shared fixtures for REER × DSPy × MLX Social Posting Pack."""

from collections.abc import Generator
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Set test environment variables
os.environ["TEST_MODE"] = "true"
os.environ["MOCK_EXTERNAL_APIS"] = "true"

# Configure pytest-asyncio for async test support
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_configs_dir() -> Path:
    """Return the sample configurations directory path."""
    return Path(__file__).parent.parent / "data" / "configs"


@pytest.fixture
def mock_mlx_model() -> Generator[Mock, None, None]:
    """Mock MLX model for testing."""
    with patch("mlx.core.load") as mock_load:
        mock_model = Mock()
        mock_model.generate.return_value = "Generated content"
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_dspy_module() -> Generator[Mock, None, None]:
    """Mock DSPy module for testing."""
    with patch("dspy.Module") as mock_module:
        mock_instance = Mock()
        mock_instance.forward.return_value = "DSPy generated content"
        mock_module.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_twitter_api() -> Generator[Mock, None, None]:
    """Mock Twitter API for testing."""
    with patch("tweepy.API") as mock_api:
        mock_instance = Mock()
        mock_instance.update_status.return_value = {"id": "123", "text": "Posted!"}
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_facebook_api() -> Generator[Mock, None, None]:
    """Mock Facebook API for testing."""
    with patch("facebook.GraphAPI") as mock_api:
        mock_instance = Mock()
        mock_instance.put_object.return_value = {"id": "123"}
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_content() -> dict[str, Any]:
    """Sample content for testing."""
    return {
        "title": "Test Post",
        "body": "This is a test post for social media.",
        "tags": ["test", "social", "ai"],
        "platform": "twitter",
        "scheduled_time": "2024-01-01T12:00:00Z",
    }


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "mlx": {
            "model_path": "/test/models/llama-3.2-3b",
            "max_tokens": 512,
            "temperature": 0.7,
        },
        "dspy": {
            "provider": "mlx",
            "model_name": "llama-3.2-3b-instruct",
            "max_tokens": 256,
        },
        "social": {
            "platforms": ["twitter", "facebook"],
            "rate_limits": {
                "twitter": 300,
                "facebook": 200,
            },
        },
    }


@pytest.fixture(autouse=True)
def mock_environment_variables() -> Generator[None, None, None]:
    """Mock environment variables for testing."""
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "TEST_MODE": "true",
        "MOCK_EXTERNAL_APIS": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "MLX_MODEL_PATH": "/test/models",
        "DSPY_LM_PROVIDER": "mlx",
    }

    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file for testing."""
    config_file = tmp_path / "test_config.json"
    config_content = {
        "app_name": "REER Test",
        "version": "0.1.0",
        "debug": True,
    }

    import json

    with open(config_file, "w") as f:
        json.dump(config_content, f)

    return config_file


@pytest.fixture
def mock_file_system(tmp_path: Path) -> Path:
    """Create a mock file system structure for testing."""
    # Create directories
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "logs").mkdir()

    # Create sample files
    (tmp_path / "data" / "sample.json").write_text('{"test": "data"}')
    (tmp_path / "outputs" / "generated.txt").write_text("Generated content")

    return tmp_path


# Pytest markers and async mode configuration
asyncio_mode = "auto"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in e2e/ directory as e2e tests
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
