"""T043: Comprehensive unit tests for lm_registry.py

Tests for provider routing, URI parsing, adapter initialization and caching,
model switching behavior, fallback mechanisms, and lifecycle management.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.exceptions import ValidationError
from plugins.lm_registry import (
    URI_PATTERN,
    DummyLanguageModelAdapter,
    LanguageModelRegistry,
    ModelReference,
    ProviderConfig,
    calculate_perplexity,
    generate_text,
    get_recommended_model,
    get_registry,
    validate_model_uri,
)


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating ProviderConfig instance."""
        config = ProviderConfig(
            name="test_provider",
            scheme="test",
            adapter_class=DummyLanguageModelAdapter,
            default_model="test-model",
            supported_models=["model1", "model2"],
            capabilities=["local", "streaming"],
            priority=50,
        )

        assert config.name == "test_provider"
        assert config.scheme == "test"
        assert config.adapter_class == DummyLanguageModelAdapter
        assert config.priority == 50

    def test_provider_config_defaults(self):
        """Test ProviderConfig with default values."""
        config = ProviderConfig(
            name="minimal", scheme="min", adapter_class=DummyLanguageModelAdapter
        )

        assert config.factory_class is None
        assert config.default_model is None
        assert config.supported_models == []
        assert config.capabilities == []
        assert config.priority == 100


class TestModelReference:
    """Test ModelReference dataclass."""

    def test_model_reference_creation(self):
        """Test creating ModelReference instance."""
        ref = ModelReference(
            uri="mlx://test-model",
            provider="mlx",
            model_path="test-model",
            parameters={"temperature": 0.7, "max_tokens": 512},
        )

        assert ref.uri == "mlx://test-model"
        assert ref.provider == "mlx"
        assert ref.model_path == "test-model"
        assert ref.parameters["temperature"] == 0.7

    def test_model_reference_defaults(self):
        """Test ModelReference with default parameters."""
        ref = ModelReference(uri="dspy://gpt-4", provider="dspy", model_path="gpt-4")

        assert ref.parameters == {}


class TestDummyLanguageModelAdapter:
    """Test DummyLanguageModelAdapter functionality."""

    @pytest.fixture
    def dummy_adapter(self):
        """Create DummyLanguageModelAdapter instance."""
        return DummyLanguageModelAdapter("test-dummy-model")

    def test_initialization(self, dummy_adapter):
        """Test DummyLanguageModelAdapter initialization."""
        assert dummy_adapter.model_name == "test-dummy-model"
        assert len(dummy_adapter._responses) > 0
        assert dummy_adapter._response_index == 0

    def test_initialization_with_kwargs(self):
        """Test initialization with additional parameters."""
        adapter = DummyLanguageModelAdapter(
            "test-model", temperature=0.8, max_tokens=100
        )

        assert adapter.model_name == "test-model"
        assert adapter.parameters["temperature"] == 0.8
        assert adapter.parameters["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_basic(self, dummy_adapter):
        """Test basic text generation."""
        prompt = "Generate a social media post"
        result = await dummy_adapter.generate(prompt)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "test-dummy-model" in result
        assert prompt[:50] in result or (prompt + "...") in result

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self, dummy_adapter):
        """Test generation with max_tokens constraint."""
        prompt = "Short prompt"
        result = await dummy_adapter.generate(prompt, max_tokens=10)

        # Should be truncated if exceeds max_tokens * 4 characters
        assert len(result) <= 10 * 4 + 3  # +3 for "..."

    @pytest.mark.asyncio
    async def test_generate_response_cycling(self, dummy_adapter):
        """Test that responses cycle through available options."""
        results = []
        for i in range(len(dummy_adapter._responses) + 2):
            result = await dummy_adapter.generate(f"Prompt {i}")
            results.append(result)

        # Should have different responses (cycling through)
        assert len(set(results)) > 1

    @pytest.mark.asyncio
    async def test_generate_stream(self, dummy_adapter):
        """Test streaming text generation."""
        prompt = "Test streaming"
        chunks = []

        async for chunk in dummy_adapter.generate_stream(prompt):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks).strip()
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_chunking(self, dummy_adapter):
        """Test that streaming produces multiple chunks."""
        prompt = "This is a longer prompt to test streaming functionality"
        chunk_count = 0

        async for _chunk in dummy_adapter.generate_stream(prompt):
            chunk_count += 1

        assert chunk_count > 1  # Should be split into multiple chunks

    @pytest.mark.asyncio
    async def test_get_perplexity(self, dummy_adapter):
        """Test perplexity calculation."""
        short_text = "Short"
        medium_text = "This is a medium length text for testing"
        long_text = "This is a much longer text that exceeds the typical short length threshold and should return different perplexity values based on length"

        short_perp = await dummy_adapter.get_perplexity(short_text)
        medium_perp = await dummy_adapter.get_perplexity(medium_text)
        long_perp = await dummy_adapter.get_perplexity(long_text)

        assert all(
            isinstance(p, float) and p > 0 for p in [short_perp, medium_perp, long_perp]
        )
        # Short text should have higher perplexity
        assert short_perp > medium_perp

    def test_is_available(self, dummy_adapter):
        """Test availability check."""
        assert dummy_adapter.is_available() is True


class TestLanguageModelRegistry:
    """Test LanguageModelRegistry functionality."""

    @pytest.fixture
    def fresh_registry(self):
        """Create a fresh registry instance for testing."""
        return LanguageModelRegistry()

    @pytest.fixture
    def mock_mlx_adapter(self):
        """Mock MLX adapter."""
        mock_adapter = Mock()
        mock_adapter.generate = AsyncMock(return_value="MLX generated text")
        mock_adapter.get_perplexity = AsyncMock(return_value=10.0)
        mock_adapter.is_available.return_value = True
        return mock_adapter

    @pytest.fixture
    def mock_dspy_adapter(self):
        """Mock DSPy adapter."""
        mock_adapter = Mock()
        mock_adapter.generate = AsyncMock(return_value="DSPy generated text")
        mock_adapter.get_perplexity = AsyncMock(return_value=12.0)
        mock_adapter.is_available.return_value = True
        return mock_adapter

    def test_initialization(self, fresh_registry):
        """Test registry initialization."""
        assert len(fresh_registry._providers) >= 3  # MLX, DSPy, Dummy
        assert fresh_registry._default_provider == "mlx"
        assert "mlx" in fresh_registry._providers
        assert "dspy" in fresh_registry._providers
        assert "dummy" in fresh_registry._providers

    def test_builtin_providers_registration(self, fresh_registry):
        """Test that builtin providers are properly registered."""
        # Check MLX provider
        mlx_config = fresh_registry.get_provider("mlx")
        assert mlx_config is not None
        assert mlx_config.scheme == "mlx"
        assert "local_inference" in mlx_config.capabilities

        # Check DSPy provider
        dspy_config = fresh_registry.get_provider("dspy")
        assert dspy_config is not None
        assert dspy_config.scheme == "dspy"
        assert "cloud_api" in dspy_config.capabilities

        # Check Dummy provider
        dummy_config = fresh_registry.get_provider("dummy")
        assert dummy_config is not None
        assert dummy_config.scheme == "dummy"
        assert dummy_config.priority == 1000  # Lowest priority

    def test_register_provider(self, fresh_registry):
        """Test registering a new provider."""
        new_config = ProviderConfig(
            name="custom",
            scheme="custom",
            adapter_class=DummyLanguageModelAdapter,
            priority=25,
        )

        fresh_registry.register_provider(new_config)

        assert "custom" in fresh_registry._providers
        assert fresh_registry.get_provider("custom") == new_config

    def test_get_provider_existing(self, fresh_registry):
        """Test getting existing provider."""
        mlx_config = fresh_registry.get_provider("mlx")
        assert mlx_config is not None
        assert mlx_config.name == "mlx"

    def test_get_provider_nonexistent(self, fresh_registry):
        """Test getting non-existent provider."""
        result = fresh_registry.get_provider("nonexistent")
        assert result is None

    def test_list_providers(self, fresh_registry):
        """Test listing all providers."""
        providers = fresh_registry.list_providers()

        assert len(providers) >= 3
        provider_names = [p.name for p in providers]
        assert "mlx" in provider_names
        assert "dspy" in provider_names
        assert "dummy" in provider_names

    def test_parse_model_uri_full_uri(self, fresh_registry):
        """Test parsing full URI with scheme."""
        uri = "mlx://llama-3.2-3b-instruct?temperature=0.7&max_tokens=512"
        ref = fresh_registry.parse_model_uri(uri)

        assert ref.uri == uri
        assert ref.provider == "mlx"
        assert ref.model_path == "llama-3.2-3b-instruct"
        assert ref.parameters["temperature"] == 0.7
        assert ref.parameters["max_tokens"] == 512

    def test_parse_model_uri_scheme_less(self, fresh_registry):
        """Test parsing URI without scheme."""
        uri = "simple-model-name"
        ref = fresh_registry.parse_model_uri(uri)

        assert ref.uri == uri
        assert ref.provider == "mlx"  # Default provider
        assert ref.model_path == "simple-model-name"
        assert ref.parameters == {}

    def test_parse_model_uri_with_path(self, fresh_registry):
        """Test parsing URI with provider path."""
        uri = "dspy://openai/gpt-4"
        ref = fresh_registry.parse_model_uri(uri)

        assert ref.provider == "dspy"
        assert ref.model_path == "openai/gpt-4"

    def test_parse_model_uri_boolean_parameters(self, fresh_registry):
        """Test parsing URI with boolean parameters."""
        uri = "mlx://model?streaming=true&debug=false"
        ref = fresh_registry.parse_model_uri(uri)

        assert ref.parameters["streaming"] is True
        assert ref.parameters["debug"] is False

    def test_parse_model_uri_invalid_scheme(self, fresh_registry):
        """Test parsing URI with invalid scheme."""
        uri = "invalid://model-name"

        with pytest.raises(ValidationError) as exc_info:
            fresh_registry.parse_model_uri(uri)
        assert "Unknown provider scheme" in str(exc_info.value)

    def test_parse_model_uri_malformed(self, fresh_registry):
        """Test parsing malformed URI."""
        # This should be handled gracefully by urlparse
        uri = "malformed:://invalid::uri"

        with pytest.raises(ValidationError) as exc_info:
            fresh_registry.parse_model_uri(uri)
        assert "Invalid model URI" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_adapter_cached(self, fresh_registry, mock_mlx_adapter):
        """Test adapter caching functionality."""
        uri = "dummy://test-model"

        # First call
        adapter1 = await fresh_registry.get_adapter(uri)
        # Second call
        adapter2 = await fresh_registry.get_adapter(uri)

        # Should return the same cached instance
        assert adapter1 is adapter2

    @pytest.mark.asyncio
    async def test_get_adapter_different_params(self, fresh_registry):
        """Test that different parameters create different cache entries."""
        uri = "dummy://test-model"

        adapter1 = await fresh_registry.get_adapter(uri, temperature=0.7)
        adapter2 = await fresh_registry.get_adapter(uri, temperature=0.9)

        # Should be different instances due to different parameters
        assert adapter1 is not adapter2

    @pytest.mark.asyncio
    async def test_get_adapter_unknown_provider(self, fresh_registry):
        """Test getting adapter for unknown provider."""
        uri = "unknown://model"

        with pytest.raises(ValidationError) as exc_info:
            await fresh_registry.get_adapter(uri)
        assert "Unknown provider scheme" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_adapter_dummy_provider(self, fresh_registry):
        """Test getting dummy adapter."""
        uri = "dummy://test-model"
        adapter = await fresh_registry.get_adapter(uri)

        assert isinstance(adapter, DummyLanguageModelAdapter)
        assert adapter.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_get_adapter_mlx_provider(self, fresh_registry):
        """Test getting MLX adapter."""
        uri = "mlx://test-model"

        with patch.object(fresh_registry, "_create_mlx_adapter") as mock_create:
            mock_adapter = Mock()
            mock_create.return_value = mock_adapter

            adapter = await fresh_registry.get_adapter(uri)

            assert adapter == mock_adapter
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_adapter_dspy_provider(self, fresh_registry):
        """Test getting DSPy adapter."""
        uri = "dspy://openai/gpt-4"

        with patch.object(fresh_registry, "_create_dspy_adapter") as mock_create:
            mock_adapter = Mock()
            mock_create.return_value = mock_adapter

            adapter = await fresh_registry.get_adapter(uri)

            assert adapter == mock_adapter
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_adapter_with_factory(self, fresh_registry):
        """Test getting adapter using factory class."""
        # Register a provider with factory
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_factory.create_adapter.return_value = mock_adapter

        config = ProviderConfig(
            name="factory_test",
            scheme="factory",
            adapter_class=DummyLanguageModelAdapter,
            factory_class=mock_factory,
        )
        fresh_registry.register_provider(config)

        uri = "factory://test-model"
        adapter = await fresh_registry.get_adapter(uri)

        assert adapter == mock_adapter
        mock_factory.create_adapter.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_get_adapter_creation_failure(self, fresh_registry):
        """Test adapter creation failure."""

        # Register a provider that will fail
        class FailingAdapter:
            def __init__(self, *args, **kwargs):
                raise Exception("Adapter creation failed")

        config = ProviderConfig(
            name="failing", scheme="failing", adapter_class=FailingAdapter
        )
        fresh_registry.register_provider(config)

        uri = "failing://test-model"

        with pytest.raises(ValidationError) as exc_info:
            await fresh_registry.get_adapter(uri)
        assert "Adapter creation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_mlx_adapter(self, fresh_registry):
        """Test MLX adapter creation."""
        model_ref = ModelReference(
            uri="mlx://test-model",
            provider="mlx",
            model_path="test-model",
            parameters={"temperature": 0.7, "max_tokens": 512},
        )

        with patch("plugins.lm_registry.MLXModelConfig"):
            with patch("plugins.lm_registry.MLXGenerationConfig"):
                with patch(
                    "plugins.lm_registry.MLXLanguageModelAdapter"
                ) as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.load_model = AsyncMock()
                    mock_adapter_class.return_value = mock_adapter

                    result = await fresh_registry._create_mlx_adapter(
                        model_ref, model_ref.parameters
                    )

                    assert result == mock_adapter
                    mock_adapter.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_dspy_adapter(self, fresh_registry):
        """Test DSPy adapter creation."""
        model_ref = ModelReference(
            uri="dspy://openai/gpt-4",
            provider="dspy",
            model_path="openai/gpt-4",
            parameters={"temperature": 0.8},
        )

        with patch("plugins.lm_registry.DSPyConfig"):
            with patch(
                "plugins.lm_registry.DSPyLanguageModelAdapter"
            ) as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.initialize = AsyncMock()
                mock_adapter_class.return_value = mock_adapter

                result = await fresh_registry._create_dspy_adapter(
                    model_ref, model_ref.parameters
                )

                assert result == mock_adapter
                mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_dspy_adapter_default_provider(self, fresh_registry):
        """Test DSPy adapter creation with default provider."""
        model_ref = ModelReference(
            uri="dspy://gpt-4",
            provider="dspy",
            model_path="gpt-4",  # No provider prefix
            parameters={},
        )

        with patch("plugins.lm_registry.DSPyConfig") as mock_config_class:
            with patch(
                "plugins.lm_registry.DSPyLanguageModelAdapter"
            ) as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.initialize = AsyncMock()
                mock_adapter_class.return_value = mock_adapter

                await fresh_registry._create_dspy_adapter(
                    model_ref, model_ref.parameters
                )

                # Should default to openai provider
                mock_config_class.assert_called_once()
                call_args = mock_config_class.call_args
                assert call_args[1]["provider"] == "openai"
                assert call_args[1]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_route_generate(self, fresh_registry):
        """Test routing generation request."""
        uri = "dummy://test-model"
        prompt = "Generate text"

        adapter = await fresh_registry.get_adapter(uri)

        with patch.object(
            adapter, "generate", return_value="Generated text"
        ) as mock_generate:
            result = await fresh_registry.route_generate(uri, prompt, temperature=0.7)

            assert result == "Generated text"
            mock_generate.assert_called_once_with(prompt, temperature=0.7)

    @pytest.mark.asyncio
    async def test_route_perplexity(self, fresh_registry):
        """Test routing perplexity calculation."""
        uri = "dummy://test-model"
        text = "Calculate perplexity for this text"

        adapter = await fresh_registry.get_adapter(uri)

        with patch.object(
            adapter, "get_perplexity", return_value=15.0
        ) as mock_perplexity:
            result = await fresh_registry.route_perplexity(uri, text)

            assert result == 15.0
            mock_perplexity.assert_called_once_with(text)

    def test_get_recommended_model_social_media_local(self, fresh_registry):
        """Test getting recommended model for social media (local preference)."""
        model = fresh_registry.get_recommended_model("social_media", prefer_local=True)
        assert model.startswith("mlx://")

    def test_get_recommended_model_social_media_cloud(self, fresh_registry):
        """Test getting recommended model for social media (cloud preference)."""
        model = fresh_registry.get_recommended_model("social_media", prefer_local=False)
        assert model.startswith("dspy://")

    def test_get_recommended_model_creative(self, fresh_registry):
        """Test getting recommended model for creative use case."""
        local_model = fresh_registry.get_recommended_model(
            "creative", prefer_local=True
        )
        cloud_model = fresh_registry.get_recommended_model(
            "creative", prefer_local=False
        )

        assert local_model.startswith("mlx://")
        assert cloud_model.startswith("dspy://")

    def test_get_recommended_model_analysis(self, fresh_registry):
        """Test getting recommended model for analysis use case."""
        local_model = fresh_registry.get_recommended_model(
            "analysis", prefer_local=True
        )
        cloud_model = fresh_registry.get_recommended_model(
            "analysis", prefer_local=False
        )

        assert local_model.startswith("mlx://")
        assert cloud_model.startswith("dspy://")

    def test_get_recommended_model_unknown_use_case(self, fresh_registry):
        """Test getting recommended model for unknown use case."""
        model = fresh_registry.get_recommended_model("unknown_use_case")
        # Should fall back to general
        assert model.startswith("mlx://")

    def test_set_default_provider(self, fresh_registry):
        """Test setting default provider."""
        fresh_registry.set_default_provider("dspy")
        assert fresh_registry._default_provider == "dspy"

    def test_set_default_provider_invalid(self, fresh_registry):
        """Test setting invalid default provider."""
        with pytest.raises(ValidationError) as exc_info:
            fresh_registry.set_default_provider("nonexistent")
        assert "Provider 'nonexistent' not found" in str(exc_info.value)

    def test_clear_cache(self, fresh_registry):
        """Test clearing adapter cache."""
        # Manually add something to cache
        fresh_registry._adapters["test_key"] = Mock()
        assert len(fresh_registry._adapters) > 0

        fresh_registry.clear_cache()
        assert len(fresh_registry._adapters) == 0

    @pytest.mark.asyncio
    async def test_health_check(self, fresh_registry):
        """Test health check functionality."""
        # Mock adapters to control health check behavior
        with patch.object(fresh_registry, "get_adapter") as mock_get_adapter:
            mock_adapter = Mock()
            mock_adapter.is_available.return_value = True
            mock_adapter.generate = AsyncMock(return_value="Test response")
            mock_get_adapter.return_value = mock_adapter

            health_status = await fresh_registry.health_check()

            assert isinstance(health_status, dict)
            assert len(health_status) >= 3  # At least mlx, dspy, dummy

            # Check that all providers have status info
            for _provider_name, status in health_status.items():
                assert "available" in status
                assert "scheme" in status

    @pytest.mark.asyncio
    async def test_health_check_with_failures(self, fresh_registry):
        """Test health check with some providers failing."""

        def mock_get_adapter(uri):
            if "mlx" in uri:
                raise Exception("MLX failed to load")

            mock_adapter = Mock()
            mock_adapter.is_available.return_value = True
            mock_adapter.generate = AsyncMock(return_value="Test response")
            return mock_adapter

        with patch.object(fresh_registry, "get_adapter", side_effect=mock_get_adapter):
            health_status = await fresh_registry.health_check()

            # MLX should show as unavailable
            assert health_status["mlx"]["available"] is False
            assert "error" in health_status["mlx"]

            # Others should be available
            assert health_status["dummy"]["available"] is True

    @pytest.mark.asyncio
    async def test_health_check_generation_failure(self, fresh_registry):
        """Test health check when generation fails but adapter is available."""
        with patch.object(fresh_registry, "get_adapter") as mock_get_adapter:
            mock_adapter = Mock()
            mock_adapter.is_available.return_value = True
            mock_adapter.generate = AsyncMock(
                side_effect=Exception("Generation failed")
            )
            mock_get_adapter.return_value = mock_adapter

            health_status = await fresh_registry.health_check()

            # Should still show as available but with generation error
            for _provider_name, status in health_status.items():
                if status["available"]:
                    assert status["test_generation"] is False
                    assert "generation_error" in status


class TestGlobalRegistryFunctions:
    """Test global registry functions."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_generate_text_convenience_function(self):
        """Test generate_text convenience function."""
        uri = "dummy://test-model"
        prompt = "Test prompt"

        with patch("plugins.lm_registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.route_generate = AsyncMock(return_value="Generated text")
            mock_get_registry.return_value = mock_registry

            result = await generate_text(uri, prompt, temperature=0.7)

            assert result == "Generated text"
            mock_registry.route_generate.assert_called_once_with(
                uri, prompt, temperature=0.7
            )

    @pytest.mark.asyncio
    async def test_calculate_perplexity_convenience_function(self):
        """Test calculate_perplexity convenience function."""
        uri = "dummy://test-model"
        text = "Test text"

        with patch("plugins.lm_registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.route_perplexity = AsyncMock(return_value=12.5)
            mock_get_registry.return_value = mock_registry

            result = await calculate_perplexity(uri, text)

            assert result == 12.5
            mock_registry.route_perplexity.assert_called_once_with(uri, text)

    def test_get_recommended_model_convenience_function(self):
        """Test get_recommended_model convenience function."""
        with patch("plugins.lm_registry.get_registry") as mock_get_registry:
            mock_registry = Mock()
            mock_registry.get_recommended_model.return_value = "mlx://recommended-model"
            mock_get_registry.return_value = mock_registry

            result = get_recommended_model("social_media", prefer_local=True)

            assert result == "mlx://recommended-model"
            mock_registry.get_recommended_model.assert_called_once_with(
                "social_media", True
            )


class TestURIValidation:
    """Test URI pattern validation."""

    def test_validate_model_uri_valid_full_uri(self):
        """Test validation of valid full URIs."""
        valid_uris = [
            "mlx://model-name",
            "dspy://openai/gpt-4",
            "https://example.com/model",
            "custom://path/to/model?param=value",
            "scheme+extension://host/path",
        ]

        for uri in valid_uris:
            assert validate_model_uri(uri) is True

    def test_validate_model_uri_scheme_less(self):
        """Test validation of scheme-less URIs."""
        scheme_less_uris = ["model-name", "path/to/model", "simple-name"]

        for uri in scheme_less_uris:
            assert validate_model_uri(uri) is True

    def test_validate_model_uri_invalid(self):
        """Test validation of invalid URIs."""
        invalid_uris = [
            "://missing-scheme",
            "scheme:/missing-slash",
            "123://invalid-scheme-start",
        ]

        for uri in invalid_uris:
            assert validate_model_uri(uri) is False

    def test_uri_pattern_regex(self):
        """Test URI_PATTERN regex directly."""
        valid_matches = [
            "http://example.com",
            "https://example.com/path",
            "mlx://model-name",
            "custom+ext://host/path?query=value#fragment",
        ]

        invalid_matches = [
            "://missing-scheme",
            "123://invalid-start",
            "scheme:/single-slash",
        ]

        for uri in valid_matches:
            assert URI_PATTERN.match(uri) is not None

        for uri in invalid_matches:
            assert URI_PATTERN.match(uri) is None


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def registry_with_mocks(self):
        """Create registry with mocked adapters for integration testing."""
        registry = LanguageModelRegistry()

        # Clear existing adapters
        registry._adapters.clear()

        return registry

    @pytest.mark.asyncio
    async def test_full_workflow_dummy_provider(self, registry_with_mocks):
        """Test full workflow with dummy provider."""
        uri = "dummy://integration-test"

        # Get adapter
        adapter = await registry_with_mocks.get_adapter(uri)
        assert isinstance(adapter, DummyLanguageModelAdapter)

        # Generate text
        text = await adapter.generate("Test prompt")
        assert isinstance(text, str)
        assert len(text) > 0

        # Calculate perplexity
        perplexity = await adapter.get_perplexity(text)
        assert isinstance(perplexity, float)
        assert perplexity > 0

    @pytest.mark.asyncio
    async def test_provider_switching(self, registry_with_mocks):
        """Test switching between different providers."""
        dummy_uri = "dummy://test-model"

        # Get dummy adapter
        dummy_adapter = await registry_with_mocks.get_adapter(dummy_uri)
        assert isinstance(dummy_adapter, DummyLanguageModelAdapter)

        # Verify that different URIs create different adapters
        dummy_adapter2 = await registry_with_mocks.get_adapter(
            "dummy://different-model"
        )
        assert dummy_adapter is not dummy_adapter2
        assert dummy_adapter.model_name != dummy_adapter2.model_name

    @pytest.mark.asyncio
    async def test_parameter_impact_on_caching(self, registry_with_mocks):
        """Test that parameters affect adapter caching."""
        base_uri = "dummy://param-test"

        # Same URI, different parameters
        adapter1 = await registry_with_mocks.get_adapter(base_uri, temperature=0.7)
        adapter2 = await registry_with_mocks.get_adapter(base_uri, temperature=0.9)
        adapter3 = await registry_with_mocks.get_adapter(
            base_uri, temperature=0.7
        )  # Same as first

        # adapter1 and adapter3 should be the same (same parameters)
        assert adapter1 is adapter3
        # adapter2 should be different (different temperature)
        assert adapter1 is not adapter2

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, registry_with_mocks):
        """Test error recovery and fallback scenarios."""
        # Test with dummy provider (should always work)
        dummy_result = await registry_with_mocks.route_generate(
            "dummy://fallback-test", "Test prompt"
        )
        assert isinstance(dummy_result, str)

        # Test invalid provider (should raise error)
        with pytest.raises(ValidationError):
            await registry_with_mocks.route_generate(
                "invalid://provider", "Test prompt"
            )

    def test_configuration_persistence(self, registry_with_mocks):
        """Test that configuration changes persist."""
        # Change default provider
        original_default = registry_with_mocks._default_provider
        registry_with_mocks.set_default_provider("dummy")

        # Parse scheme-less URI
        ref = registry_with_mocks.parse_model_uri("test-model")
        assert ref.provider == "dummy"

        # Restore original default
        registry_with_mocks.set_default_provider(original_default)

    @pytest.mark.asyncio
    async def test_concurrent_adapter_access(self, registry_with_mocks):
        """Test concurrent access to the same adapter."""
        uri = "dummy://concurrent-test"

        # Create multiple concurrent requests
        async def get_adapter_task():
            return await registry_with_mocks.get_adapter(uri)

        tasks = [get_adapter_task() for _ in range(5)]
        adapters = await asyncio.gather(*tasks)

        # All should return the same cached instance
        first_adapter = adapters[0]
        for adapter in adapters[1:]:
            assert adapter is first_adapter

    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, registry_with_mocks):
        """Test comprehensive health check scenario."""
        health_status = await registry_with_mocks.health_check()

        # Should have status for all registered providers
        assert "dummy" in health_status

        # Dummy provider should be available
        dummy_status = health_status["dummy"]
        assert dummy_status["available"] is True
        assert dummy_status["scheme"] == "dummy"
        assert dummy_status.get("test_generation") is True
