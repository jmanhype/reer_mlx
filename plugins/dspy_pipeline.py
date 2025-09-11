"""T038: Enhanced DSPy pipeline integration with provider routing.

Provides advanced DSPy integration with:
- Provider routing and fallback
- Structured prompt templates
- Optimization and fine-tuning
- Performance monitoring
- Rate limiting integration
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import json

try:
    import dspy
    from dspy import OpenAI, Anthropic, Together, ChatAdapter
    from dspy.teleprompt import BootstrapFewShot, MIPRO, LabeledFewShot
    from dspy.evaluate import Evaluate

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

from .dspy_lm import DSPyConfig, DSPyLanguageModelAdapter
from .lm_registry import LanguageModelRegistry, get_registry
from core.exceptions import ValidationError, ScoringError, OptimizationError
from core.integration import (
    RateLimiter,
    RateLimitConfig,
    StructuredLogger,
    LoggingConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Enhanced DSPy Configuration
# ============================================================================


@dataclass
class DSPyPipelineConfig:
    """Configuration for DSPy pipeline with provider routing."""

    primary_provider_uri: str
    fallback_provider_uris: List[str] = field(default_factory=list)
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    optimization_enabled: bool = True
    cache_enabled: bool = True
    retry_attempts: int = 3
    timeout_seconds: int = 30
    performance_tracking: bool = True


@dataclass
class DSPyTemplate:
    """Enhanced template for DSPy modules."""

    name: str
    description: str
    input_signature: str  # e.g., "question, context -> answer"
    system_prompt: Optional[str] = None
    reasoning_mode: bool = False
    examples: List[Dict[str, Any]] = field(default_factory=list)
    optimization_metric: str = "accuracy"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSPyExample:
    """Training/validation example for DSPy optimization."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DSPy Module Factory
# ============================================================================


class DSPyModuleFactory:
    """Factory for creating optimized DSPy modules."""

    def __init__(self, config: DSPyPipelineConfig):
        self.config = config
        self.registry = get_registry()
        self.rate_limiter = RateLimiter(config.rate_limit_config)
        self.logger = StructuredLogger(
            f"{__name__}.DSPyModuleFactory", config.logging_config
        )

        # Performance tracking
        self._metrics = {
            "modules_created": 0,
            "optimizations_run": 0,
            "successful_optimizations": 0,
            "total_generation_time": 0.0,
            "total_optimization_time": 0.0,
        }

    async def create_module(
        self,
        template: DSPyTemplate,
        provider_uri: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> "DSPyModule":
        """Create a DSPy module from template.

        Args:
            template: DSPy template configuration
            provider_uri: Override provider URI
            trace_id: Optional trace ID for logging

        Returns:
            Configured DSPy module
        """
        start_time = time.time()

        try:
            self.logger.log_with_context(
                "INFO",
                f"Creating DSPy module: {template.name}",
                trace_id=trace_id,
                template_name=template.name,
            )

            # Determine provider
            if not provider_uri:
                provider_uri = self.config.primary_provider_uri

            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Get adapter from registry
            adapter = await self.registry.get_adapter(provider_uri)

            # Create DSPy module
            module = DSPyModule(
                template=template,
                adapter=adapter,
                provider_uri=provider_uri,
                config=self.config,
                trace_id=trace_id,
            )

            await module.initialize()

            # Record success
            self.rate_limiter.record_success()
            self._metrics["modules_created"] += 1

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "INFO",
                f"DSPy module created successfully",
                trace_id=trace_id,
                duration_ms=duration_ms,
                provider_uri=provider_uri,
            )

            return module

        except Exception as e:
            self.rate_limiter.record_failure()

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "ERROR",
                f"Failed to create DSPy module: {e}",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
            )

            # Try fallback providers
            if self.config.fallback_provider_uris:
                for fallback_uri in self.config.fallback_provider_uris:
                    try:
                        self.logger.log_with_context(
                            "INFO",
                            f"Trying fallback provider: {fallback_uri}",
                            trace_id=trace_id,
                        )

                        return await self.create_module(
                            template, fallback_uri, trace_id
                        )
                    except Exception as fallback_error:
                        self.logger.log_with_context(
                            "WARNING",
                            f"Fallback provider failed: {fallback_error}",
                            trace_id=trace_id,
                            fallback_uri=fallback_uri,
                        )
                        continue

            raise ValidationError(
                f"Failed to create DSPy module {template.name}: {e}",
                details={"template": template.name, "provider_uri": provider_uri},
                original_error=e,
            )

    async def optimize_module(
        self,
        module: "DSPyModule",
        training_examples: List[DSPyExample],
        validation_examples: Optional[List[DSPyExample]] = None,
        optimization_method: str = "bootstrap",
        trace_id: Optional[str] = None,
    ) -> "DSPyModule":
        """Optimize a DSPy module using training examples.

        Args:
            module: Module to optimize
            training_examples: Training examples
            validation_examples: Optional validation examples
            optimization_method: Optimization method (bootstrap, mipro, labeled)
            trace_id: Optional trace ID for logging

        Returns:
            Optimized module
        """
        if not DSPY_AVAILABLE:
            raise OptimizationError("DSPy is not available for optimization")

        start_time = time.time()

        try:
            self.logger.log_with_context(
                "INFO",
                f"Starting optimization for module: {module.template.name}",
                trace_id=trace_id,
                method=optimization_method,
                training_examples=len(training_examples),
            )

            # Convert examples to DSPy format
            dspy_examples = []
            for example in training_examples:
                dspy_example = dspy.Example(**example.inputs, **example.outputs)
                dspy_examples.append(dspy_example)

            # Choose optimization method
            if optimization_method == "bootstrap":
                optimizer = BootstrapFewShot(
                    metric=module._create_metric(),
                    max_bootstrapped_demos=8,
                    max_labeled_demos=8,
                )
            elif optimization_method == "mipro":
                optimizer = MIPRO(
                    metric=module._create_metric(),
                    num_candidates=10,
                    init_temperature=1.0,
                )
            elif optimization_method == "labeled":
                optimizer = LabeledFewShot(k=min(8, len(training_examples)))
            else:
                raise OptimizationError(
                    f"Unknown optimization method: {optimization_method}"
                )

            # Apply rate limiting for optimization
            await self.rate_limiter.acquire()

            # Run optimization
            optimized_module = optimizer.compile(
                module.dspy_module, trainset=dspy_examples
            )

            # Update module with optimized version
            module.dspy_module = optimized_module
            module.is_optimized = True
            module.optimization_method = optimization_method

            # Record success
            self.rate_limiter.record_success()
            self._metrics["successful_optimizations"] += 1

            duration_ms = (time.time() - start_time) * 1000
            self._metrics["total_optimization_time"] += duration_ms / 1000

            self.logger.log_with_context(
                "INFO",
                f"Module optimization completed successfully",
                trace_id=trace_id,
                duration_ms=duration_ms,
                method=optimization_method,
            )

            return module

        except Exception as e:
            self.rate_limiter.record_failure()

            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_with_context(
                "ERROR",
                f"Module optimization failed: {e}",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(e).__name__,
            )

            raise OptimizationError(
                f"Optimization failed for module {module.template.name}: {e}",
                details={
                    "method": optimization_method,
                    "training_examples": len(training_examples),
                },
                original_error=e,
            )

        finally:
            self._metrics["optimizations_run"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get factory performance metrics."""
        return self._metrics.copy()


# ============================================================================
# Enhanced DSPy Module
# ============================================================================


class DSPyModule:
    """Enhanced DSPy module with provider routing and monitoring."""

    def __init__(
        self,
        template: DSPyTemplate,
        adapter: DSPyLanguageModelAdapter,
        provider_uri: str,
        config: DSPyPipelineConfig,
        trace_id: Optional[str] = None,
    ):
        self.template = template
        self.adapter = adapter
        self.provider_uri = provider_uri
        self.config = config
        self.trace_id = trace_id

        # DSPy components
        self.dspy_module: Optional[dspy.Module] = None
        self.predictor: Optional[dspy.Predict] = None

        # State
        self.is_initialized = False
        self.is_optimized = False
        self.optimization_method: Optional[str] = None

        # Performance tracking
        self._metrics = {
            "predictions": 0,
            "successful_predictions": 0,
            "total_prediction_time": 0.0,
            "average_prediction_time": 0.0,
            "cache_hits": 0,
        }

        # Setup logger
        self.logger = StructuredLogger(f"{__name__}.DSPyModule", config.logging_config)

    async def initialize(self) -> None:
        """Initialize the DSPy module."""
        if self.is_initialized:
            return

        if not DSPY_AVAILABLE:
            raise ValidationError("DSPy is not available")

        try:
            # Set DSPy language model
            # This would require converting our adapter to DSPy format
            # For now, we'll create a mock DSPy module

            # Create predictor based on template
            if self.template.reasoning_mode:
                self.predictor = dspy.ChainOfThought(self.template.input_signature)
            else:
                self.predictor = dspy.Predict(self.template.input_signature)

            # Create module wrapper
            self.dspy_module = self._create_module_wrapper()

            self.is_initialized = True

            self.logger.log_with_context(
                "INFO",
                f"DSPy module initialized: {self.template.name}",
                trace_id=self.trace_id,
                provider_uri=self.provider_uri,
            )

        except Exception as e:
            raise ValidationError(
                f"Failed to initialize DSPy module {self.template.name}: {e}",
                details={"template": self.template.name},
                original_error=e,
            )

    def _create_module_wrapper(self) -> dspy.Module:
        """Create a DSPy module wrapper."""

        class ModuleWrapper(dspy.Module):
            def __init__(self, predictor, template):
                super().__init__()
                self.predictor = predictor
                self.template = template

            def forward(self, **kwargs):
                return self.predictor(**kwargs)

        return ModuleWrapper(self.predictor, self.template)

    def _create_metric(self) -> Callable:
        """Create evaluation metric for optimization."""

        def accuracy_metric(example, prediction, trace=None):
            # Simple accuracy metric - in practice, this would be more sophisticated
            if hasattr(example, "answer") and hasattr(prediction, "answer"):
                return example.answer.lower() == prediction.answer.lower()
            return False

        def quality_metric(example, prediction, trace=None):
            # Quality-based metric using length and coherence
            if hasattr(prediction, "answer"):
                answer = prediction.answer
                # Simple quality heuristics
                if len(answer) < 10:
                    return 0.0
                if len(answer) > 1000:
                    return 0.5
                return 1.0
            return 0.0

        # Return appropriate metric based on template
        if self.template.optimization_metric == "accuracy":
            return accuracy_metric
        elif self.template.optimization_metric == "quality":
            return quality_metric
        else:
            return accuracy_metric

    async def predict(self, **inputs) -> Dict[str, Any]:
        """Make a prediction using the DSPy module.

        Args:
            **inputs: Input arguments for prediction

        Returns:
            Prediction result
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            self.logger.log_with_context(
                "DEBUG",
                f"Making prediction with module: {self.template.name}",
                trace_id=self.trace_id,
                input_keys=list(inputs.keys()),
            )

            # Check cache if enabled
            cache_key = None
            if self.config.cache_enabled:
                cache_key = self._generate_cache_key(inputs)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self._metrics["cache_hits"] += 1
                    return cached_result

            # Make prediction
            prediction = self.dspy_module(**inputs)

            # Convert prediction to dict
            if hasattr(prediction, "__dict__"):
                result = {
                    key: value
                    for key, value in prediction.__dict__.items()
                    if not key.startswith("_")
                }
            else:
                result = {"output": str(prediction)}

            # Cache result if enabled
            if self.config.cache_enabled and cache_key:
                self._cache_result(cache_key, result)

            # Update metrics
            self._metrics["successful_predictions"] += 1
            duration = time.time() - start_time
            self._metrics["total_prediction_time"] += duration
            self._metrics["average_prediction_time"] = (
                self._metrics["total_prediction_time"]
                / self._metrics["successful_predictions"]
            )

            self.logger.log_with_context(
                "DEBUG",
                f"Prediction completed successfully",
                trace_id=self.trace_id,
                duration_ms=duration * 1000,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.logger.log_with_context(
                "ERROR",
                f"Prediction failed: {e}",
                trace_id=self.trace_id,
                duration_ms=duration * 1000,
                error_type=type(e).__name__,
            )

            raise ScoringError(
                f"Prediction failed for module {self.template.name}: {e}",
                details={"inputs": inputs},
                original_error=e,
            )

        finally:
            self._metrics["predictions"] += 1

    def _generate_cache_key(self, inputs: Dict[str, Any]) -> str:
        """Generate cache key for inputs."""
        import hashlib

        # Create deterministic string from inputs
        input_str = json.dumps(inputs, sort_keys=True)

        # Add module signature to key
        key_data = f"{self.template.name}:{input_str}"

        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result (stub implementation)."""
        # In a real implementation, this would use Redis or similar
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache result (stub implementation)."""
        # In a real implementation, this would use Redis or similar
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get module performance metrics."""
        return {
            **self._metrics,
            "template_name": self.template.name,
            "provider_uri": self.provider_uri,
            "is_optimized": self.is_optimized,
            "optimization_method": self.optimization_method,
        }


# ============================================================================
# DSPy Pipeline Manager
# ============================================================================


class DSPyPipelineManager:
    """Manager for DSPy pipelines with provider routing."""

    def __init__(self, config: DSPyPipelineConfig):
        self.config = config
        self.factory = DSPyModuleFactory(config)
        self.modules: Dict[str, DSPyModule] = {}
        self.templates: Dict[str, DSPyTemplate] = {}

        # Performance tracking
        self._pipeline_metrics = {
            "pipelines_executed": 0,
            "successful_pipelines": 0,
            "total_pipeline_time": 0.0,
        }

        # Setup logger
        self.logger = StructuredLogger(
            f"{__name__}.DSPyPipelineManager", config.logging_config
        )

    def register_template(self, template: DSPyTemplate) -> None:
        """Register a DSPy template."""
        self.templates[template.name] = template

        self.logger.log_with_context(
            "INFO",
            f"Registered DSPy template: {template.name}",
            template_name=template.name,
        )

    async def get_module(
        self,
        template_name: str,
        provider_uri: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> DSPyModule:
        """Get or create a DSPy module."""

        cache_key = f"{template_name}:{provider_uri or 'default'}"

        if cache_key in self.modules:
            return self.modules[cache_key]

        if template_name not in self.templates:
            raise ValidationError(f"Template not found: {template_name}")

        template = self.templates[template_name]
        module = await self.factory.create_module(template, provider_uri, trace_id)

        self.modules[cache_key] = module
        return module

    async def execute_pipeline(
        self,
        template_name: str,
        inputs: Dict[str, Any],
        provider_uri: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a DSPy pipeline."""

        start_time = time.time()

        try:
            self.logger.log_with_context(
                "INFO",
                f"Executing DSPy pipeline: {template_name}",
                trace_id=trace_id,
                template_name=template_name,
            )

            # Get module
            module = await self.get_module(template_name, provider_uri, trace_id)

            # Execute prediction
            result = await module.predict(**inputs)

            # Add pipeline metadata
            result["_pipeline_metadata"] = {
                "template_name": template_name,
                "provider_uri": provider_uri or self.config.primary_provider_uri,
                "execution_time": time.time() - start_time,
                "is_optimized": module.is_optimized,
            }

            # Update metrics
            self._pipeline_metrics["successful_pipelines"] += 1
            duration = time.time() - start_time
            self._pipeline_metrics["total_pipeline_time"] += duration

            self.logger.log_with_context(
                "INFO",
                f"Pipeline executed successfully",
                trace_id=trace_id,
                duration_ms=duration * 1000,
                template_name=template_name,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.logger.log_with_context(
                "ERROR",
                f"Pipeline execution failed: {e}",
                trace_id=trace_id,
                duration_ms=duration * 1000,
                error_type=type(e).__name__,
            )

            raise

        finally:
            self._pipeline_metrics["pipelines_executed"] += 1

    async def optimize_module(
        self,
        template_name: str,
        training_examples: List[DSPyExample],
        validation_examples: Optional[List[DSPyExample]] = None,
        optimization_method: str = "bootstrap",
        trace_id: Optional[str] = None,
    ) -> None:
        """Optimize a module using training examples."""

        module = await self.get_module(template_name, trace_id=trace_id)

        await self.factory.optimize_module(
            module,
            training_examples,
            validation_examples,
            optimization_method,
            trace_id,
        )

        self.logger.log_with_context(
            "INFO",
            f"Module optimization completed: {template_name}",
            trace_id=trace_id,
            template_name=template_name,
            method=optimization_method,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""

        # Collect module metrics
        module_metrics = {}
        for key, module in self.modules.items():
            module_metrics[key] = module.get_metrics()

        return {
            "pipeline_metrics": self._pipeline_metrics,
            "factory_metrics": self.factory.get_metrics(),
            "module_metrics": module_metrics,
            "templates_registered": len(self.templates),
            "modules_cached": len(self.modules),
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_dspy_pipeline(primary_provider_uri: str, **kwargs) -> DSPyPipelineManager:
    """Factory function to create DSPy pipeline manager.

    Args:
        primary_provider_uri: Primary provider URI
        **kwargs: Additional configuration options

    Returns:
        Configured DSPyPipelineManager
    """
    config = DSPyPipelineConfig(primary_provider_uri=primary_provider_uri, **kwargs)

    return DSPyPipelineManager(config)


# ============================================================================
# Pre-built Templates
# ============================================================================

SOCIAL_MEDIA_TEMPLATES = {
    "content_generation": DSPyTemplate(
        name="content_generation",
        description="Generate social media content",
        input_signature="topic, style, target_audience -> content",
        reasoning_mode=True,
        optimization_metric="quality",
    ),
    "strategy_extraction": DSPyTemplate(
        name="strategy_extraction",
        description="Extract strategy features from content",
        input_signature="content, platform -> features, confidence",
        reasoning_mode=True,
        optimization_metric="accuracy",
    ),
    "engagement_prediction": DSPyTemplate(
        name="engagement_prediction",
        description="Predict engagement for content",
        input_signature="content, audience_data -> engagement_score, reasoning",
        reasoning_mode=True,
        optimization_metric="accuracy",
    ),
}


def get_social_media_templates() -> Dict[str, DSPyTemplate]:
    """Get pre-built social media templates."""
    return SOCIAL_MEDIA_TEMPLATES.copy()
