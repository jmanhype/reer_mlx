"""Example usage of DSPy pipeline modules.

This script demonstrates how to use the REER × DSPy × MLX pipeline
for end-to-end social media content generation and optimization.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

try:
    from .pipeline import (
        PipelineFactory,
        ContentRequest,
        PipelineConfig,
        REERDSPyPipeline,
    )
    from .reer_module import REERSearchModule, SearchStrategy
    from .evaluator import KPIEvaluator
    from ..plugins.dspy_lm import DSPyConfig
    from ..core.trainer import OptimizationConfig
except ImportError:
    # Fallback for standalone usage
    from pipeline import (
        PipelineFactory,
        ContentRequest,
        PipelineConfig,
        REERDSPyPipeline,
    )
    from reer_module import REERSearchModule, SearchStrategy
    from evaluator import KPIEvaluator
    from plugins.dspy_lm import DSPyConfig
    from core.trainer import OptimizationConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def basic_pipeline_example():
    """Example of basic content generation pipeline."""
    logger.info("=== Basic Pipeline Example ===")

    try:
        # Create basic pipeline (no REER search or optimization)
        pipeline = PipelineFactory.create_basic_pipeline(
            provider="openai", model="gpt-3.5-turbo", temperature=0.7
        )

        # Create content request
        request = ContentRequest(
            request_id="example_001",
            topic="AI automation in productivity",
            platform="twitter",
            audience="tech professionals",
            style="informative",
            requirements={
                "include_hashtags": True,
                "character_limit": 280,
                "call_to_action": True,
            },
        )

        # Generate content
        result = await pipeline.generate_content(request)

        if result.success:
            logger.info(f"Generated content: {result.content}")
            logger.info(f"Overall score: {result.scores.overall_score:.3f}")
            logger.info(f"Generation time: {result.generation_time:.2f}s")
        else:
            logger.error(f"Generation failed: {result.error_message}")

    except Exception as e:
        logger.error(f"Basic pipeline example failed: {e}")


async def full_pipeline_example():
    """Example of full pipeline with all components."""
    logger.info("=== Full Pipeline Example ===")

    try:
        # Create output directory
        output_dir = Path("./pipeline_output")

        # Create full pipeline with all features
        pipeline = PipelineFactory.create_full_pipeline(
            provider="openai",
            model="gpt-3.5-turbo",
            output_directory=output_dir,
            temperature=0.7,
            reasoning_mode=True,
        )

        # Create multiple content requests
        requests = [
            ContentRequest(
                request_id="full_001",
                topic="Remote work productivity tips",
                platform="linkedin",
                audience="remote workers",
                style="professional",
                requirements={"professional_tone": True, "actionable_tips": True},
            ),
            ContentRequest(
                request_id="full_002",
                topic="AI trends 2024",
                platform="twitter",
                audience="tech enthusiasts",
                style="engaging",
                requirements={"trending_hashtags": True, "future_focused": True},
            ),
            ContentRequest(
                request_id="full_003",
                topic="Startup growth strategies",
                platform="instagram",
                audience="entrepreneurs",
                style="inspirational",
                requirements={"visual_elements": True, "story_format": True},
            ),
        ]

        # Generate batch of content
        pipeline_result = await pipeline.generate_batch(requests)

        logger.info(f"Batch completed: {pipeline_result.success_rate:.1%} success rate")
        logger.info(f"Total time: {pipeline_result.total_time:.2f}s")
        logger.info(
            f"Average score: {pipeline_result.summary_metrics.get('avg_overall_score', 0):.3f}"
        )

        # Show best content
        if pipeline_result.best_content:
            best = pipeline_result.best_content
            logger.info(f"Best content ({best.request_id}): {best.content[:100]}...")
            logger.info(f"Best score: {best.scores.overall_score:.3f}")

    except Exception as e:
        logger.error(f"Full pipeline example failed: {e}")


async def reer_search_example():
    """Example of REER search module usage."""
    logger.info("=== REER Search Example ===")

    try:
        # Initialize REER search module
        search_module = REERSearchModule()
        await search_module.initialize()

        # Perform different types of searches
        searches = [
            {
                "query": "AI productivity tools",
                "platform": "twitter",
                "strategy": SearchStrategy.HYBRID,
                "depth": 3,
            },
            {
                "query": "remote work trends",
                "platform": "linkedin",
                "strategy": SearchStrategy.TRENDING,
                "depth": 2,
            },
            {
                "query": "startup growth hacks",
                "platform": "instagram",
                "strategy": SearchStrategy.COMPETITIVE,
                "depth": 2,
            },
        ]

        for search_config in searches:
            result = await search_module.search(**search_config)

            if result.success:
                logger.info(
                    f"Search '{result.query}' found {len(result.results)} results"
                )
                logger.info(f"Context summary: {result.context_summary[:100]}...")
                logger.info(f"Trends identified: {result.trends_identified}")

                # Show top result
                if result.results:
                    top_result = result.results[0]
                    logger.info(f"Top result: {top_result.content[:150]}...")
                    logger.info(f"Relevance: {top_result.relevance_score:.3f}")
            else:
                logger.error(f"Search failed: {result.error_message}")

    except Exception as e:
        logger.error(f"REER search example failed: {e}")


async def kpi_evaluation_example():
    """Example of KPI evaluator usage."""
    logger.info("=== KPI Evaluation Example ===")

    try:
        # Initialize KPI evaluator
        evaluator = KPIEvaluator(
            metrics=[
                "engagement_rate",
                "quality_score",
                "viral_potential",
                "brand_alignment",
                "reach_efficiency",
            ],
            use_dspy_analysis=True,
        )
        await evaluator.initialize()

        # Sample content for evaluation
        content_samples = [
            {
                "content": "🚀 Just automated my entire workflow with AI - saved 10 hours this week! Here's how you can do it too: [link] #AI #productivity #automation",
                "metadata": {
                    "platform": "twitter",
                    "topic": "AI automation",
                    "audience": "professionals",
                },
            },
            {
                "content": "The future of remote work is here. Companies that embrace flexible work arrangements are seeing 40% higher retention rates. What's your remote work strategy? #RemoteWork #Leadership #FutureOfWork",
                "metadata": {
                    "platform": "linkedin",
                    "topic": "remote work",
                    "audience": "business leaders",
                },
            },
            {
                "content": "Behind the scenes of building a startup 💯 The real journey: 80% problems, 15% breakthroughs, 5% celebration. But every challenge teaches you something new ✨ #StartupLife #Entrepreneurship #GrowthMindset",
                "metadata": {
                    "platform": "instagram",
                    "topic": "startup journey",
                    "audience": "entrepreneurs",
                },
            },
        ]

        # Evaluate each content sample
        for i, sample in enumerate(content_samples):
            result = await evaluator.evaluate(
                content=sample["content"],
                metadata=sample["metadata"],
                benchmark_comparison=True,
            )

            logger.info(f"Content {i+1} evaluation:")
            logger.info(
                f"  Overall Score: {result.overall_score:.2f}/10 (Grade: {result.grade})"
            )
            logger.info(f"  Platform: {result.platform}")

            # Show metric breakdown
            for metric_result in result.metric_results:
                logger.info(
                    f"  {metric_result.metric_name}: {metric_result.value:.2f} "
                    f"(Target: {'✓' if metric_result.target_achieved else '✗'})"
                )

            # Show recommendations
            if result.recommendations:
                logger.info(f"  Recommendations: {result.recommendations[0]}")

            # Show benchmark insights
            if result.insights.get("benchmark_comparison"):
                benchmark_perf = result.insights.get("benchmark_performance", 0)
                logger.info(
                    f"  Benchmark Performance: {benchmark_perf:.1f}% above average"
                )

        # Get evaluation summary
        summary = await evaluator.get_evaluation_summary(time_window_hours=1)
        logger.info(
            f"Evaluation Summary: {summary.get('total_evaluations', 0)} evaluations"
        )
        logger.info(f"Average Score: {summary.get('average_score', 0):.2f}")

    except Exception as e:
        logger.error(f"KPI evaluation example failed: {e}")


async def optimization_example():
    """Example of GEPA optimization with pipeline."""
    logger.info("=== Optimization Example ===")

    try:
        # Create optimization configuration
        optimization_config = OptimizationConfig(
            population_size=20,
            max_generations=10,
            mutation_rate=0.2,
            crossover_rate=0.8,
            quality_threshold=0.8,
        )

        # Create optimization-focused pipeline
        pipeline = PipelineFactory.create_optimization_pipeline(
            provider="openai",
            model="gpt-3.5-turbo",
            optimization_config=optimization_config,
        )

        # Define optimization target
        optimization_target = {
            "topic": "AI productivity tools",
            "audience": "tech professionals",
            "platform": "twitter",
            "engagement_target": 8.0,
            "quality_target": 8.5,
            "viral_target": 7.0,
        }

        # Run GEPA optimization
        logger.info("Starting GEPA optimization (this may take a while)...")
        optimization_result = await pipeline.optimize_with_gepa(optimization_target)

        if optimization_result.success:
            best_individual = optimization_result.best_individual
            logger.info(f"Optimization completed successfully!")
            logger.info(f"Best fitness: {best_individual.overall_fitness:.3f}")
            logger.info(f"Generations: {optimization_result.total_generations}")
            logger.info(
                f"Total time: {optimization_result.optimization_time_seconds:.1f}s"
            )
            logger.info(f"Best content: {best_individual.phenotype}")

            # Show fitness breakdown
            for metric, score in best_individual.fitness_scores.items():
                logger.info(f"  {metric}: {score:.3f}")

        else:
            logger.error(f"Optimization failed: {optimization_result.error_message}")

    except Exception as e:
        logger.error(f"Optimization example failed: {e}")


async def pipeline_status_example():
    """Example of checking pipeline and module status."""
    logger.info("=== Pipeline Status Example ===")

    try:
        # Create pipeline
        pipeline = PipelineFactory.create_full_pipeline()
        await pipeline.initialize()

        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        logger.info("Pipeline Status:")
        logger.info(f"  Initialized: {status['initialized']}")
        logger.info(f"  Pipeline ID: {status['pipeline_id']}")
        logger.info(f"  Components: {list(status['components'].keys())}")

        # Check individual modules
        if pipeline.reer_search:
            reer_status = await pipeline.reer_search.get_module_status()
            logger.info(f"REER Search Status: {reer_status['initialized']}")
            logger.info(f"  DSPy Available: {reer_status['dspy_available']}")
            logger.info(f"  Cache Size: {reer_status['cache_size']}")

        if pipeline.kpi_evaluator:
            evaluator_status = await pipeline.kpi_evaluator.get_evaluator_status()
            logger.info(f"KPI Evaluator Status: {evaluator_status['initialized']}")
            logger.info(f"  Metrics: {evaluator_status['metrics_configured']}")
            logger.info(f"  Total Evaluations: {evaluator_status['total_evaluations']}")

    except Exception as e:
        logger.error(f"Status example failed: {e}")


async def main():
    """Run all examples."""
    logger.info("Starting DSPy Pipeline Examples")

    examples = [
        ("Basic Pipeline", basic_pipeline_example),
        ("REER Search", reer_search_example),
        ("KPI Evaluation", kpi_evaluation_example),
        ("Pipeline Status", pipeline_status_example),
        ("Full Pipeline", full_pipeline_example),
        # ("Optimization", optimization_example),  # Commented out as it takes longer
    ]

    for name, example_func in examples:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*50}")
            await example_func()
            logger.info(f"✓ {name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")

        # Small delay between examples
        await asyncio.sleep(1)

    logger.info(f"\n{'='*50}")
    logger.info("All examples completed!")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    # Note: This script requires proper API keys and DSPy setup
    # Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    asyncio.run(main())
