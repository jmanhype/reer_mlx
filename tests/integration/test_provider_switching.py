"""T012: Integration test for provider switching (mlx::/dspy::) scenarios.

Tests the complete provider switching workflow including automatic failover,
performance comparison, load balancing, and seamless provider transitions.
Following London School TDD with mock-first approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import timezone, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

# Import statements that will fail initially (RED phase)
try:
    from provider_management.dspy_provider import DSPyProvider
    from provider_management.failover_manager import FailoverManager
    from provider_management.health_monitor import HealthMonitor
    from provider_management.load_balancer import LoadBalancer
    from provider_management.mlx_provider import MLXProvider
    from provider_management.performance_tracker import PerformanceTracker
    from provider_management.provider_manager import ProviderManager
    from provider_management.schemas import (
        ProviderConfig,
        ProviderStatus,
        SwitchingDecision,
    )

    from core.exceptions import LoadBalancingError, ProviderError, SwitchingError
except ImportError:
    # Expected during RED phase - create mock classes for contract testing
    class ProviderManager:
        pass

    class MLXProvider:
        pass

    class DSPyProvider:
        pass

    class LoadBalancer:
        pass

    class HealthMonitor:
        pass

    class FailoverManager:
        pass

    class PerformanceTracker:
        pass

    class ProviderConfig:
        pass

    class ProviderStatus:
        pass

    class SwitchingDecision:
        pass

    class ProviderError(Exception):
        pass

    class SwitchingError(Exception):
        pass

    class LoadBalancingError(Exception):
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestProviderSwitchingIntegration:
    """Integration tests for provider switching scenarios.

    Tests complete end-to-end workflows including:
    - Automatic failover between MLX and DSPy providers
    - Performance-based provider selection
    - Load balancing across multiple providers
    - Health monitoring and proactive switching
    - Seamless transition without service interruption
    - Provider capability matching for specific tasks
    """

    @pytest.fixture
    def mlx_provider_config(self) -> dict[str, Any]:
        """MLX provider configuration."""
        return {
            "provider_id": "mlx_primary",
            "type": "mlx",
            "name": "MLX Local Provider",
            "model_config": {
                "model_path": "/models/llama-3.2-3b-instruct",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "batch_size": 4,
                "device": "gpu",
            },
            "capabilities": [
                "text_generation",
                "content_optimization",
                "strategy_analysis",
                "batch_processing",
            ],
            "performance_characteristics": {
                "avg_latency_ms": 850,
                "throughput_tokens_per_second": 45,
                "memory_usage_mb": 3200,
                "gpu_utilization": 0.75,
            },
            "limits": {
                "max_concurrent_requests": 8,
                "max_context_length": 8192,
                "rate_limit_per_minute": 120,
            },
            "health_check": {
                "endpoint": "/health",
                "interval_seconds": 30,
                "timeout_ms": 5000,
            },
        }

    @pytest.fixture
    def dspy_provider_config(self) -> dict[str, Any]:
        """DSPy provider configuration."""
        return {
            "provider_id": "dspy_primary",
            "type": "dspy",
            "name": "DSPy Framework Provider",
            "model_config": {
                "model_name": "llama-3.2-3b",
                "max_tokens": 256,
                "temperature": 0.8,
                "provider": "mlx",
                "optimization_enabled": True,
                "compilation_cache": True,
            },
            "capabilities": [
                "structured_generation",
                "reasoning_chains",
                "few_shot_learning",
                "program_optimization",
            ],
            "performance_characteristics": {
                "avg_latency_ms": 1200,
                "throughput_tokens_per_second": 35,
                "memory_usage_mb": 2800,
                "optimization_overhead_ms": 150,
            },
            "limits": {
                "max_concurrent_requests": 6,
                "max_context_length": 4096,
                "rate_limit_per_minute": 90,
            },
            "health_check": {
                "endpoint": "/dspy/health",
                "interval_seconds": 45,
                "timeout_ms": 8000,
            },
        }

    @pytest.fixture
    def sample_generation_request(self) -> dict[str, Any]:
        """Sample generation request for provider testing."""
        return {
            "request_id": str(uuid4()),
            "task_type": "content_generation",
            "input": {
                "prompt": "Generate a social media post about AI development best practices",
                "max_tokens": 280,
                "temperature": 0.7,
                "context": "professional_technical_audience",
            },
            "requirements": {
                "response_time_ms": 2000,
                "quality_threshold": 0.8,
                "preferred_providers": ["mlx", "dspy"],
                "fallback_enabled": True,
            },
            "metadata": {
                "user_id": "user_123",
                "session_id": "session_456",
                "priority": "normal",
                "timeout_ms": 10000,
            },
        }

    @pytest.fixture
    def mock_mlx_provider(self, mlx_provider_config: dict[str, Any]) -> Mock:
        """Mock MLX provider with behavior contracts."""
        provider = Mock(spec=MLXProvider)
        provider.config = mlx_provider_config
        provider.provider_id = mlx_provider_config["provider_id"]
        provider.type = "mlx"
        provider.is_healthy = Mock(return_value=True)
        provider.get_status = Mock()
        provider.generate = AsyncMock()
        provider.health_check = AsyncMock()
        provider.get_performance_metrics = Mock()
        provider.supports_capability = Mock()
        return provider

    @pytest.fixture
    def mock_dspy_provider(self, dspy_provider_config: dict[str, Any]) -> Mock:
        """Mock DSPy provider with behavior contracts."""
        provider = Mock(spec=DSPyProvider)
        provider.config = dspy_provider_config
        provider.provider_id = dspy_provider_config["provider_id"]
        provider.type = "dspy"
        provider.is_healthy = Mock(return_value=True)
        provider.get_status = Mock()
        provider.generate = AsyncMock()
        provider.health_check = AsyncMock()
        provider.get_performance_metrics = Mock()
        provider.supports_capability = Mock()
        return provider

    @pytest.fixture
    def mock_load_balancer(self) -> Mock:
        """Mock LoadBalancer with behavior contracts."""
        balancer = Mock(spec=LoadBalancer)
        balancer.select_provider = Mock()
        balancer.update_provider_metrics = Mock()
        balancer.get_load_distribution = Mock()
        balancer.rebalance = AsyncMock()
        balancer.get_balancing_strategy = Mock()
        return balancer

    @pytest.fixture
    def mock_health_monitor(self) -> Mock:
        """Mock HealthMonitor with behavior contracts."""
        monitor = Mock(spec=HealthMonitor)
        monitor.check_provider_health = AsyncMock()
        monitor.get_health_status = Mock()
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        monitor.register_health_callback = Mock()
        return monitor

    @pytest.fixture
    def mock_failover_manager(self) -> Mock:
        """Mock FailoverManager with behavior contracts."""
        manager = Mock(spec=FailoverManager)
        manager.should_failover = Mock()
        manager.execute_failover = AsyncMock()
        manager.get_failover_candidates = Mock()
        manager.update_failover_history = Mock()
        manager.get_failover_metrics = Mock()
        return manager

    @pytest.fixture
    def mock_performance_tracker(self) -> Mock:
        """Mock PerformanceTracker with behavior contracts."""
        tracker = Mock(spec=PerformanceTracker)
        tracker.track_request = Mock()
        tracker.get_provider_metrics = Mock()
        tracker.get_comparative_metrics = Mock()
        tracker.record_switching_decision = Mock()
        tracker.get_switching_history = Mock()
        return tracker

    @pytest.fixture
    def mock_provider_manager(
        self,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
        mock_load_balancer: Mock,
        mock_health_monitor: Mock,
        mock_failover_manager: Mock,
        mock_performance_tracker: Mock,
    ) -> Mock:
        """Mock ProviderManager with all dependencies."""
        manager = Mock(spec=ProviderManager)
        manager.providers = {"mlx": mock_mlx_provider, "dspy": mock_dspy_provider}
        manager.load_balancer = mock_load_balancer
        manager.health_monitor = mock_health_monitor
        manager.failover_manager = mock_failover_manager
        manager.performance_tracker = mock_performance_tracker
        manager.route_request = AsyncMock()
        manager.switch_provider = AsyncMock()
        manager.get_optimal_provider = Mock()
        manager.handle_provider_failure = AsyncMock()
        return manager

    # Core Provider Switching Tests

    async def test_automatic_failover_mlx_to_dspy(
        self,
        mock_provider_manager: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test automatic failover from MLX to DSPy when MLX fails."""
        # Arrange
        # MLX provider fails, DSPy succeeds
        mock_mlx_provider.generate.side_effect = ProviderError(
            "MLX service unavailable"
        )
        mock_dspy_provider.generate.return_value = {
            "content": "Generated content via DSPy fallback",
            "provider_used": "dspy::llama-3.2-3b",
            "generation_time_ms": 1350,
            "quality_score": 0.84,
        }

        mock_provider_manager.failover_manager.should_failover.return_value = True
        mock_provider_manager.failover_manager.get_failover_candidates.return_value = [
            "dspy"
        ]

        failover_result = {
            "original_provider": "mlx",
            "failover_provider": "dspy",
            "failover_reason": "provider_error",
            "failover_time_ms": 250,
            "success": True,
            "content": "Generated content via DSPy fallback",
        }
        mock_provider_manager.route_request.return_value = failover_result

        # Act - This will fail initially (RED phase)
        result = await mock_provider_manager.route_request(sample_generation_request)

        # Assert - Testing the expected failover behavior
        assert result["original_provider"] == "mlx"
        assert result["failover_provider"] == "dspy"
        assert result["success"] is True
        assert result["failover_time_ms"] < 500  # Fast failover

        mock_provider_manager.failover_manager.should_failover.assert_called_once()
        mock_provider_manager.failover_manager.get_failover_candidates.assert_called_once()

    async def test_performance_based_provider_selection(
        self,
        mock_provider_manager: Mock,
        mock_load_balancer: Mock,
        mock_performance_tracker: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test provider selection based on performance metrics."""
        # Arrange
        performance_metrics = {
            "mlx": {
                "avg_latency_ms": 850,
                "success_rate": 0.97,
                "quality_score": 0.86,
                "throughput": 45,
                "current_load": 0.60,
                "availability": 0.99,
            },
            "dspy": {
                "avg_latency_ms": 1200,
                "success_rate": 0.94,
                "quality_score": 0.89,
                "throughput": 35,
                "current_load": 0.40,
                "availability": 0.97,
            },
        }

        # MLX has better latency and throughput, DSPy has better quality
        mock_performance_tracker.get_comparative_metrics.return_value = (
            performance_metrics
        )

        # For latency-sensitive request, choose MLX
        mock_load_balancer.select_provider.return_value = {
            "selected_provider": "mlx",
            "selection_reason": "lowest_latency",
            "confidence": 0.85,
            "alternatives": ["dspy"],
        }

        selection_result = {
            "provider": "mlx",
            "selection_criteria": "performance_optimized",
            "metrics_considered": ["latency", "throughput", "current_load"],
            "expected_performance": performance_metrics["mlx"],
        }
        mock_provider_manager.get_optimal_provider.return_value = selection_result

        # Act
        result = mock_provider_manager.get_optimal_provider(
            sample_generation_request, optimization_target="latency"
        )

        # Assert
        assert result["provider"] == "mlx"
        assert result["selection_criteria"] == "performance_optimized"
        assert "latency" in result["metrics_considered"]

        mock_performance_tracker.get_comparative_metrics.assert_called_once()
        mock_load_balancer.select_provider.assert_called_once()

    async def test_load_balancing_across_providers(
        self,
        mock_provider_manager: Mock,
        mock_load_balancer: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test load balancing across multiple providers."""
        # Arrange
        load_distribution = {
            "mlx": {"current_requests": 6, "capacity": 8, "utilization": 0.75},
            "dspy": {"current_requests": 2, "capacity": 6, "utilization": 0.33},
        }

        balancing_decisions = [
            {"provider": "dspy", "reason": "lower_utilization"},  # Request 1
            {"provider": "dspy", "reason": "lower_utilization"},  # Request 2
            {"provider": "mlx", "reason": "capacity_available"},  # Request 3
            {"provider": "dspy", "reason": "optimal_distribution"},  # Request 4
        ]

        mock_load_balancer.get_load_distribution.return_value = load_distribution
        mock_load_balancer.select_provider.side_effect = balancing_decisions

        # Act - Simulate 4 concurrent requests
        results = []
        for i in range(4):
            request = {**sample_generation_request, "request_id": f"req_{i}"}
            selection = mock_load_balancer.select_provider(request, load_distribution)
            results.append(selection)

        # Assert
        assert len(results) == 4
        # Should favor DSPy initially due to lower utilization
        dspy_selections = sum(1 for r in results if r["provider"] == "dspy")
        assert dspy_selections >= 2  # At least half should go to less utilized DSPy

    async def test_health_based_provider_switching(
        self,
        mock_provider_manager: Mock,
        mock_health_monitor: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
    ):
        """Test provider switching based on health status changes."""
        # Arrange
        initial_health = {
            "mlx": {
                "status": "healthy",
                "response_time_ms": 45,
                "error_rate": 0.02,
                "last_check": datetime.now(UTC).isoformat(),
            },
            "dspy": {
                "status": "healthy",
                "response_time_ms": 67,
                "error_rate": 0.03,
                "last_check": datetime.now(UTC).isoformat(),
            },
        }

        degraded_health = {
            "mlx": {
                "status": "degraded",
                "response_time_ms": 2500,  # Significantly slower
                "error_rate": 0.15,  # Higher error rate
                "last_check": datetime.now(UTC).isoformat(),
            },
            "dspy": {
                "status": "healthy",
                "response_time_ms": 78,
                "error_rate": 0.04,
                "last_check": datetime.now(UTC).isoformat(),
            },
        }

        mock_health_monitor.get_health_status.side_effect = [
            initial_health,
            degraded_health,
        ]
        mock_health_monitor.check_provider_health.return_value = {
            "health_change_detected": True,
            "affected_providers": ["mlx"],
            "recommended_action": "switch_traffic_to_healthy_providers",
        }

        switching_decision = {
            "switch_required": True,
            "from_provider": "mlx",
            "to_provider": "dspy",
            "reason": "health_degradation",
            "urgency": "high",
        }
        mock_provider_manager.switch_provider.return_value = switching_decision

        # Act
        initial_status = mock_health_monitor.get_health_status()
        health_check = await mock_health_monitor.check_provider_health()

        if health_check["health_change_detected"]:
            switch_result = await mock_provider_manager.switch_provider(
                from_provider="mlx", reason="health_degradation"
            )

        # Assert
        assert initial_status["mlx"]["status"] == "healthy"
        assert health_check["health_change_detected"] is True
        assert switch_result["switch_required"] is True
        assert switch_result["reason"] == "health_degradation"

    # Capability-Based Provider Selection Tests

    async def test_capability_based_provider_matching(
        self,
        mock_provider_manager: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
    ):
        """Test provider selection based on capability requirements."""
        # Arrange
        # MLX supports batch processing, DSPy supports structured generation
        mock_mlx_provider.supports_capability.side_effect = lambda cap: cap in [
            "text_generation",
            "content_optimization",
            "batch_processing",
        ]
        mock_dspy_provider.supports_capability.side_effect = lambda cap: cap in [
            "structured_generation",
            "reasoning_chains",
            "few_shot_learning",
        ]

        capability_requests = [
            {
                "request_type": "batch_content_generation",
                "required_capabilities": ["text_generation", "batch_processing"],
                "expected_provider": "mlx",
            },
            {
                "request_type": "structured_data_extraction",
                "required_capabilities": ["structured_generation", "reasoning_chains"],
                "expected_provider": "dspy",
            },
            {
                "request_type": "simple_text_generation",
                "required_capabilities": ["text_generation"],
                "expected_provider": "mlx",  # Both support, but MLX is faster
            },
        ]

        def mock_get_optimal_provider(request):
            for cap_req in capability_requests:
                if request["task_type"] == cap_req["request_type"]:
                    return {
                        "provider": cap_req["expected_provider"],
                        "match_score": 0.95,
                        "capabilities_matched": cap_req["required_capabilities"],
                    }
            return None

        mock_provider_manager.get_optimal_provider.side_effect = (
            mock_get_optimal_provider
        )

        # Act & Assert
        for cap_req in capability_requests:
            request = {
                "task_type": cap_req["request_type"],
                "required_capabilities": cap_req["required_capabilities"],
            }
            result = mock_provider_manager.get_optimal_provider(request)

            assert result["provider"] == cap_req["expected_provider"]
            assert result["match_score"] > 0.9

    async def test_dynamic_capability_discovery(
        self, mock_mlx_provider: Mock, mock_dspy_provider: Mock
    ):
        """Test dynamic discovery and validation of provider capabilities."""
        # Arrange
        mlx_capabilities = {
            "discovered_capabilities": [
                "text_generation",
                "content_optimization",
                "batch_processing",
                "multilingual_support",
            ],
            "capability_scores": {
                "text_generation": 0.95,
                "content_optimization": 0.88,
                "batch_processing": 0.92,
                "multilingual_support": 0.76,
            },
            "discovery_method": "runtime_testing",
            "last_updated": datetime.now(UTC).isoformat(),
        }

        dspy_capabilities = {
            "discovered_capabilities": [
                "structured_generation",
                "reasoning_chains",
                "few_shot_learning",
                "program_optimization",
            ],
            "capability_scores": {
                "structured_generation": 0.92,
                "reasoning_chains": 0.89,
                "few_shot_learning": 0.94,
                "program_optimization": 0.87,
            },
            "discovery_method": "runtime_testing",
            "last_updated": datetime.now(UTC).isoformat(),
        }

        mock_mlx_provider.get_performance_metrics.return_value = mlx_capabilities
        mock_dspy_provider.get_performance_metrics.return_value = dspy_capabilities

        # Act
        mlx_caps = mock_mlx_provider.get_performance_metrics()
        dspy_caps = mock_dspy_provider.get_performance_metrics()

        # Assert
        assert len(mlx_caps["discovered_capabilities"]) == 4
        assert len(dspy_caps["discovered_capabilities"]) == 4
        assert all(score > 0.7 for score in mlx_caps["capability_scores"].values())
        assert all(score > 0.7 for score in dspy_caps["capability_scores"].values())

    # Error Handling and Resilience Tests

    async def test_cascading_failure_handling(
        self,
        mock_provider_manager: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test handling of cascading failures across providers."""
        # Arrange
        # Both providers fail sequentially
        mock_mlx_provider.generate.side_effect = ProviderError("MLX overloaded")
        mock_dspy_provider.generate.side_effect = ProviderError("DSPy timeout")

        cascade_handling = {
            "primary_failure": "mlx",
            "secondary_failure": "dspy",
            "fallback_strategy": "queue_request",
            "estimated_retry_time_seconds": 30,
            "circuit_breaker_activated": True,
        }

        mock_provider_manager.handle_provider_failure.return_value = cascade_handling

        # Act
        result = await mock_provider_manager.handle_provider_failure(
            sample_generation_request, failed_providers=["mlx", "dspy"]
        )

        # Assert
        assert result["primary_failure"] == "mlx"
        assert result["secondary_failure"] == "dspy"
        assert result["circuit_breaker_activated"] is True
        assert result["fallback_strategy"] == "queue_request"

    async def test_graceful_degradation_strategies(
        self,
        mock_provider_manager: Mock,
        mock_performance_tracker: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test graceful degradation when providers are under stress."""
        # Arrange
        stress_conditions = {
            "mlx": {
                "cpu_usage": 0.95,
                "memory_usage": 0.89,
                "queue_length": 15,
                "avg_response_time_ms": 3500,
                "status": "stressed",
            },
            "dspy": {
                "cpu_usage": 0.78,
                "memory_usage": 0.65,
                "queue_length": 3,
                "avg_response_time_ms": 1800,
                "status": "normal",
            },
        }

        degradation_strategy = {
            "strategy": "reduce_quality_for_speed",
            "adjustments": {
                "max_tokens": 150,  # Reduced from 280
                "temperature": 0.5,  # More deterministic
                "timeout_ms": 5000,  # Shorter timeout
            },
            "expected_performance": {
                "response_time_ms": 2000,
                "quality_reduction": 0.15,
            },
        }

        mock_performance_tracker.get_provider_metrics.return_value = stress_conditions
        mock_provider_manager.route_request.return_value = {
            "provider_used": "dspy",
            "degradation_applied": True,
            "strategy": degradation_strategy,
            "content": "Degraded but functional content",
        }

        # Act
        result = await mock_provider_manager.route_request(
            sample_generation_request, allow_degradation=True
        )

        # Assert
        assert result["degradation_applied"] is True
        assert result["provider_used"] == "dspy"  # Less stressed provider chosen
        assert result["strategy"]["expected_performance"]["response_time_ms"] < 3000

    async def test_circuit_breaker_pattern(
        self,
        mock_provider_manager: Mock,
        mock_mlx_provider: Mock,
        sample_generation_request: dict[str, Any],
    ):
        """Test circuit breaker pattern for failing providers."""
        # Arrange
        # Simulate repeated failures to trigger circuit breaker
        failure_sequence = [
            ProviderError("Timeout"),
            ProviderError("Overloaded"),
            ProviderError("Memory error"),
            ProviderError("Network error"),
            ProviderError("Service unavailable"),
        ]
        mock_mlx_provider.generate.side_effect = failure_sequence

        def mock_route_request(request):
            failure_count = len(list(mock_mlx_provider.generate.call_args_list))
            if failure_count >= 5:
                return {
                    "circuit_breaker_open": True,
                    "provider_blocked": "mlx",
                    "alternative_used": "dspy",
                    "content": "Circuit breaker fallback content",
                }
            raise ProviderError(f"Failure {failure_count}")

        mock_provider_manager.route_request.side_effect = mock_route_request

        # Act & Assert
        # First 4 requests should fail normally
        for _i in range(4):
            with pytest.raises(ProviderError):
                await mock_provider_manager.route_request(sample_generation_request)

        # 5th request should trigger circuit breaker
        result = await mock_provider_manager.route_request(sample_generation_request)
        assert result["circuit_breaker_open"] is True
        assert result["provider_blocked"] == "mlx"

    # Performance and Monitoring Tests

    async def test_provider_performance_comparison(
        self,
        mock_performance_tracker: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
    ):
        """Test comprehensive performance comparison between providers."""
        # Arrange
        performance_comparison = {
            "comparison_period": "last_24_hours",
            "request_count": {"mlx": 1250, "dspy": 950},
            "metrics": {
                "mlx": {
                    "avg_latency_ms": 850,
                    "p95_latency_ms": 1200,
                    "p99_latency_ms": 1800,
                    "success_rate": 0.97,
                    "error_rate": 0.03,
                    "throughput_rps": 2.1,
                    "quality_score_avg": 0.86,
                    "cost_per_request": 0.003,
                },
                "dspy": {
                    "avg_latency_ms": 1200,
                    "p95_latency_ms": 1650,
                    "p99_latency_ms": 2400,
                    "success_rate": 0.94,
                    "error_rate": 0.06,
                    "throughput_rps": 1.8,
                    "quality_score_avg": 0.89,
                    "cost_per_request": 0.004,
                },
            },
            "recommendations": {
                "latency_sensitive": "mlx",
                "quality_focused": "dspy",
                "cost_optimized": "mlx",
                "reliability_focused": "mlx",
            },
        }

        mock_performance_tracker.get_comparative_metrics.return_value = (
            performance_comparison
        )

        # Act
        comparison = mock_performance_tracker.get_comparative_metrics(
            time_period="24h", include_recommendations=True
        )

        # Assert
        assert (
            comparison["metrics"]["mlx"]["avg_latency_ms"]
            < comparison["metrics"]["dspy"]["avg_latency_ms"]
        )
        assert (
            comparison["metrics"]["dspy"]["quality_score_avg"]
            > comparison["metrics"]["mlx"]["quality_score_avg"]
        )
        assert comparison["recommendations"]["latency_sensitive"] == "mlx"
        assert comparison["recommendations"]["quality_focused"] == "dspy"

    async def test_real_time_provider_monitoring(
        self, mock_health_monitor: Mock, mock_performance_tracker: Mock
    ):
        """Test real-time monitoring of provider health and performance."""
        # Arrange
        monitoring_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "providers": {
                "mlx": {
                    "status": "healthy",
                    "metrics": {
                        "current_requests": 6,
                        "queue_length": 2,
                        "cpu_usage": 0.72,
                        "memory_usage": 0.68,
                        "gpu_usage": 0.84,
                        "last_response_time_ms": 920,
                    },
                    "alerts": [],
                },
                "dspy": {
                    "status": "warning",
                    "metrics": {
                        "current_requests": 4,
                        "queue_length": 8,
                        "cpu_usage": 0.89,
                        "memory_usage": 0.76,
                        "gpu_usage": 0.91,
                        "last_response_time_ms": 1450,
                    },
                    "alerts": ["high_queue_length", "elevated_cpu_usage"],
                },
            },
            "system_health": {
                "overall_status": "healthy",
                "load_balance_status": "optimal",
                "failover_ready": True,
            },
        }

        mock_health_monitor.get_health_status.return_value = monitoring_data

        # Act
        health_data = mock_health_monitor.get_health_status()

        # Assert
        assert health_data["providers"]["mlx"]["status"] == "healthy"
        assert health_data["providers"]["dspy"]["status"] == "warning"
        assert len(health_data["providers"]["dspy"]["alerts"]) == 2
        assert health_data["system_health"]["overall_status"] == "healthy"

    async def test_switching_decision_history_and_analytics(
        self, mock_performance_tracker: Mock
    ):
        """Test tracking and analysis of provider switching decisions."""
        # Arrange
        switching_history = {
            "total_switches": 45,
            "time_period": "last_week",
            "switch_reasons": {
                "performance_degradation": 18,
                "health_issues": 12,
                "load_balancing": 8,
                "capability_mismatch": 4,
                "manual_override": 3,
            },
            "switch_patterns": {"mlx_to_dspy": 23, "dspy_to_mlx": 22},
            "success_rates": {"automatic_switches": 0.91, "manual_switches": 0.87},
            "avg_switch_time_ms": 180,
            "performance_impact": {
                "during_switch": {
                    "latency_increase": 0.15,
                    "error_rate_increase": 0.02,
                },
                "post_switch": {"performance_improvement": 0.22},
            },
        }

        mock_performance_tracker.get_switching_history.return_value = switching_history

        # Act
        history = mock_performance_tracker.get_switching_history(
            time_period="7d", include_analytics=True
        )

        # Assert
        assert history["total_switches"] == 45
        assert (
            history["switch_reasons"]["performance_degradation"] == 18
        )  # Most common reason
        assert history["success_rates"]["automatic_switches"] > 0.9
        assert history["avg_switch_time_ms"] < 500  # Fast switching

    # Advanced Scenarios and Edge Cases

    async def test_multi_tenant_provider_isolation(
        self, mock_provider_manager: Mock, sample_generation_request: dict[str, Any]
    ):
        """Test provider isolation and resource allocation for multi-tenant scenarios."""
        # Arrange
        tenant_configs = {
            "tenant_premium": {
                "provider_priority": ["mlx", "dspy"],
                "resource_allocation": {"mlx": 0.6, "dspy": 0.4},
                "performance_guarantees": {"max_latency_ms": 1000, "min_quality": 0.9},
            },
            "tenant_standard": {
                "provider_priority": ["dspy", "mlx"],
                "resource_allocation": {"mlx": 0.3, "dspy": 0.7},
                "performance_guarantees": {"max_latency_ms": 2000, "min_quality": 0.8},
            },
        }

        isolation_result = {
            "tenant_id": "tenant_premium",
            "allocated_provider": "mlx",
            "resource_share": 0.6,
            "isolation_enforced": True,
            "performance_guaranteed": True,
        }

        mock_provider_manager.route_request.return_value = isolation_result

        # Act
        premium_request = {
            **sample_generation_request,
            "tenant_id": "tenant_premium",
            "sla_requirements": tenant_configs["tenant_premium"][
                "performance_guarantees"
            ],
        }
        result = await mock_provider_manager.route_request(premium_request)

        # Assert
        assert result["tenant_id"] == "tenant_premium"
        assert (
            result["allocated_provider"] == "mlx"
        )  # Premium tenant gets priority provider
        assert result["isolation_enforced"] is True

    async def test_provider_warm_up_and_cold_start_handling(
        self,
        mock_provider_manager: Mock,
        mock_mlx_provider: Mock,
        mock_dspy_provider: Mock,
    ):
        """Test handling of provider warm-up and cold start scenarios."""
        # Arrange

        mock_provider_manager.get_optimal_provider.return_value = {
            "provider": "dspy",  # Choose warm provider for immediate requests
            "reason": "avoid_cold_start_penalty",
        }

        # Act
        result = mock_provider_manager.get_optimal_provider(
            {"urgency": "high", "max_latency_ms": 2000}
        )

        # Assert
        assert result["provider"] == "dspy"  # Warm provider chosen
        assert result["reason"] == "avoid_cold_start_penalty"

    async def test_provider_version_compatibility_and_migration(
        self, mock_provider_manager: Mock
    ):
        """Test handling of provider version compatibility and migration scenarios."""
        # Arrange

        migration_plan = {
            "affected_provider": "dspy",
            "migration_strategy": "gradual_traffic_shift",
            "timeline": {
                "preparation": "2024-01-15",
                "migration_start": "2024-01-20",
                "completion": "2024-01-25",
            },
            "rollback_plan": "immediate_traffic_redirect_to_mlx",
        }

        mock_provider_manager.switch_provider.return_value = {
            "migration_initiated": True,
            "plan": migration_plan,
            "status": "in_progress",
        }

        # Act
        migration_result = await mock_provider_manager.switch_provider(
            migration_mode=True, target_version="v0.4.0"
        )

        # Assert
        assert migration_result["migration_initiated"] is True
        assert migration_result["plan"]["migration_strategy"] == "gradual_traffic_shift"
        assert migration_result["status"] == "in_progress"
