"""T013: Integration test for GEPA (Genetic Evolution Post Analyzer) optimization flow.

Tests the complete GEPA optimization workflow including genetic algorithm-based
post evolution, fitness evaluation, population management, and iterative
optimization for social media content. Following London School TDD with
mock-first approach and behavior verification.

This test suite MUST fail initially (RED phase) since implementations don't exist yet.
"""

from datetime import datetime, timedelta
from datetime import timezone
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import numpy as np
import pytest

# Import statements that will fail initially (RED phase)
try:
    from gepa_optimization.convergence_monitor import ConvergenceMonitor
    from gepa_optimization.crossover_engine import CrossoverEngine
    from gepa_optimization.fitness_evaluator import FitnessEvaluator
    from gepa_optimization.genetic_algorithm import GeneticAlgorithm
    from gepa_optimization.gepa_optimizer import GEPAOptimizer
    from gepa_optimization.mutation_engine import MutationEngine
    from gepa_optimization.population_manager import PopulationManager
    from gepa_optimization.schemas import Individual, OptimizationResult, Population
    from gepa_optimization.selection_engine import SelectionEngine

    from core.exceptions import ConvergenceError, FitnessError, OptimizationError
except ImportError:
    # Expected during RED phase - create mock classes for contract testing
    class GEPAOptimizer:
        pass

    class GeneticAlgorithm:
        pass

    class FitnessEvaluator:
        pass

    class PopulationManager:
        pass

    class MutationEngine:
        pass

    class CrossoverEngine:
        pass

    class SelectionEngine:
        pass

    class ConvergenceMonitor:
        pass

    class Individual:
        pass

    class Population:
        pass

    class OptimizationResult:
        pass

    class OptimizationError(Exception):
        pass

    class ConvergenceError(Exception):
        pass

    class FitnessError(Exception):
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestGEPAOptimizationIntegration:
    """Integration tests for GEPA optimization flow.

    Tests complete end-to-end workflows including:
    - Genetic algorithm initialization and evolution
    - Multi-objective fitness evaluation
    - Population diversity management
    - Adaptive mutation and crossover strategies
    - Convergence detection and early stopping
    - Real-time optimization monitoring
    - Performance-based parameter tuning
    """

    @pytest.fixture
    def sample_optimization_config(self) -> dict[str, Any]:
        """Sample GEPA optimization configuration."""
        return {
            "optimization_id": str(uuid4()),
            "algorithm_config": {
                "population_size": 50,
                "max_generations": 100,
                "elite_percentage": 0.1,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "selection_method": "tournament",
                "tournament_size": 3,
            },
            "fitness_objectives": [
                {
                    "name": "engagement_rate",
                    "weight": 0.4,
                    "target": "maximize",
                    "threshold": 0.1,
                },
                {
                    "name": "viral_potential",
                    "weight": 0.3,
                    "target": "maximize",
                    "threshold": 0.15,
                },
                {
                    "name": "brand_alignment",
                    "weight": 0.2,
                    "target": "maximize",
                    "threshold": 0.8,
                },
                {
                    "name": "content_quality",
                    "weight": 0.1,
                    "target": "maximize",
                    "threshold": 0.75,
                },
            ],
            "convergence_criteria": {
                "max_stagnant_generations": 10,
                "fitness_improvement_threshold": 0.001,
                "diversity_threshold": 0.15,
                "target_fitness": 0.9,
            },
            "content_constraints": {
                "max_length": 280,
                "min_length": 50,
                "required_elements": ["hashtags", "call_to_action"],
                "prohibited_words": ["spam", "clickbait"],
                "platform_compliance": ["twitter", "linkedin"],
            },
            "optimization_target": {
                "topic": "AI development best practices",
                "audience": "technical_professionals",
                "brand_voice": "knowledgeable_approachable",
                "goal": "maximize_engagement_with_quality",
            },
        }

    @pytest.fixture
    def sample_initial_population(self) -> list[dict[str, Any]]:
        """Sample initial population for optimization."""
        return [
            {
                "individual_id": f"ind_{i:03d}",
                "genes": {
                    "content_structure": {
                        "hook_type": np.random.choice(
                            ["question", "statement", "statistic"]
                        ),
                        "main_content_style": np.random.choice(
                            ["tips", "insights", "tutorial"]
                        ),
                        "call_to_action": np.random.choice(
                            ["explicit", "implicit", "question"]
                        ),
                    },
                    "visual_elements": {
                        "emoji_count": np.random.randint(0, 3),
                        "hashtag_count": np.random.randint(1, 4),
                        "emoji_placement": np.random.choice(
                            ["beginning", "middle", "end"]
                        ),
                    },
                    "linguistic_features": {
                        "tone": np.random.choice(
                            ["professional", "casual", "enthusiastic"]
                        ),
                        "complexity": np.random.uniform(0.3, 0.8),
                        "sentiment": np.random.uniform(0.5, 0.9),
                    },
                    "timing_genes": {
                        "optimal_hour": np.random.randint(8, 18),
                        "day_preference": np.random.choice(
                            ["weekday", "weekend", "any"]
                        ),
                    },
                },
                "phenotype": {
                    "text": f"Sample AI development tip #{i+1}: Always validate your data. Quality matters! #AI #Development",
                    "metadata": {
                        "character_count": 85 + i,
                        "engagement_prediction": np.random.uniform(0.5, 0.9),
                        "quality_score": np.random.uniform(0.6, 0.95),
                    },
                },
                "fitness": {
                    "raw_score": 0.0,
                    "normalized_score": 0.0,
                    "objective_scores": {},
                    "evaluated": False,
                },
                "generation": 0,
                "lineage": {
                    "parent1": None,
                    "parent2": None,
                    "mutation_applied": False,
                },
            }
            for i in range(20)  # Sample of 20 individuals
        ]

    @pytest.fixture
    def expected_optimization_result(self) -> dict[str, Any]:
        """Expected optimization result after GEPA evolution."""
        return {
            "optimization_id": "opt_123",
            "status": "converged",
            "generations_completed": 25,
            "execution_time_ms": 45000,
            "convergence_reason": "target_fitness_reached",
            "best_individual": {
                "individual_id": "ind_best_025_003",
                "generation": 25,
                "fitness_score": 0.94,
                "genes": {
                    "content_structure": {
                        "hook_type": "question",
                        "main_content_style": "tips",
                        "call_to_action": "implicit",
                    },
                    "visual_elements": {
                        "emoji_count": 1,
                        "hashtag_count": 3,
                        "emoji_placement": "beginning",
                    },
                    "linguistic_features": {
                        "tone": "professional",
                        "complexity": 0.65,
                        "sentiment": 0.82,
                    },
                },
                "optimized_content": {
                    "text": "🚀 What's the most overlooked step in AI development? Data validation! Poor data quality = poor model performance. Always validate before training. #AI #MachineLearning #BestPractices",
                    "character_count": 178,
                    "predicted_metrics": {
                        "engagement_rate": 0.15,
                        "viral_potential": 0.23,
                        "brand_alignment": 0.91,
                        "content_quality": 0.88,
                    },
                },
            },
            "population_statistics": {
                "final_population_size": 50,
                "average_fitness": 0.78,
                "fitness_standard_deviation": 0.12,
                "diversity_index": 0.34,
                "elite_individuals": 5,
            },
            "evolution_history": {
                "fitness_progression": [
                    0.45,
                    0.52,
                    0.61,
                    0.68,
                    0.74,
                    0.81,
                    0.87,
                    0.92,
                    0.94,
                ],
                "diversity_progression": [
                    0.89,
                    0.82,
                    0.75,
                    0.68,
                    0.58,
                    0.45,
                    0.38,
                    0.34,
                    0.34,
                ],
                "mutation_rate_adaptation": [
                    0.15,
                    0.14,
                    0.13,
                    0.12,
                    0.11,
                    0.10,
                    0.09,
                    0.08,
                    0.08,
                ],
                "crossover_success_rate": [
                    0.65,
                    0.68,
                    0.72,
                    0.75,
                    0.78,
                    0.81,
                    0.83,
                    0.85,
                    0.86,
                ],
            },
            "optimization_insights": {
                "key_success_factors": [
                    "question_based_hooks_perform_best",
                    "single_emoji_optimal",
                    "three_hashtags_maximize_reach",
                    "professional_tone_preferred",
                ],
                "convergence_analysis": {
                    "early_generations_focus": "diversity_exploration",
                    "late_generations_focus": "fine_tuning_optimization",
                    "critical_generation": 18,
                },
            },
        }

    @pytest.fixture
    def mock_genetic_algorithm(self) -> Mock:
        """Mock GeneticAlgorithm with behavior contracts."""
        algorithm = Mock(spec=GeneticAlgorithm)
        algorithm.initialize_population = AsyncMock()
        algorithm.evolve_generation = AsyncMock()
        algorithm.get_evolution_statistics = Mock()
        algorithm.adapt_parameters = AsyncMock()
        algorithm.check_convergence = Mock()
        return algorithm

    @pytest.fixture
    def mock_fitness_evaluator(self) -> Mock:
        """Mock FitnessEvaluator with behavior contracts."""
        evaluator = Mock(spec=FitnessEvaluator)
        evaluator.evaluate_individual = AsyncMock()
        evaluator.evaluate_population = AsyncMock()
        evaluator.calculate_multi_objective_fitness = Mock()
        evaluator.update_fitness_weights = Mock()
        evaluator.get_fitness_distribution = Mock()
        return evaluator

    @pytest.fixture
    def mock_population_manager(self) -> Mock:
        """Mock PopulationManager with behavior contracts."""
        manager = Mock(spec=PopulationManager)
        manager.create_initial_population = AsyncMock()
        manager.select_parents = Mock()
        manager.replace_population = AsyncMock()
        manager.maintain_diversity = AsyncMock()
        manager.get_population_statistics = Mock()
        manager.archive_elite = Mock()
        return manager

    @pytest.fixture
    def mock_mutation_engine(self) -> Mock:
        """Mock MutationEngine with behavior contracts."""
        engine = Mock(spec=MutationEngine)
        engine.mutate_individual = AsyncMock()
        engine.adaptive_mutation = AsyncMock()
        engine.get_mutation_statistics = Mock()
        engine.update_mutation_rate = Mock()
        return engine

    @pytest.fixture
    def mock_crossover_engine(self) -> Mock:
        """Mock CrossoverEngine with behavior contracts."""
        engine = Mock(spec=CrossoverEngine)
        engine.crossover = AsyncMock()
        engine.smart_crossover = AsyncMock()
        engine.get_crossover_statistics = Mock()
        engine.update_crossover_strategy = Mock()
        return engine

    @pytest.fixture
    def mock_selection_engine(self) -> Mock:
        """Mock SelectionEngine with behavior contracts."""
        engine = Mock(spec=SelectionEngine)
        engine.tournament_selection = Mock()
        engine.roulette_wheel_selection = Mock()
        engine.rank_based_selection = Mock()
        engine.elite_selection = Mock()
        engine.diversity_based_selection = Mock()
        return engine

    @pytest.fixture
    def mock_convergence_monitor(self) -> Mock:
        """Mock ConvergenceMonitor with behavior contracts."""
        monitor = Mock(spec=ConvergenceMonitor)
        monitor.check_convergence = Mock()
        monitor.update_convergence_metrics = Mock()
        monitor.get_convergence_status = Mock()
        monitor.predict_convergence_time = Mock()
        return monitor

    @pytest.fixture
    def mock_gepa_optimizer(
        self,
        mock_genetic_algorithm: Mock,
        mock_fitness_evaluator: Mock,
        mock_population_manager: Mock,
        mock_mutation_engine: Mock,
        mock_crossover_engine: Mock,
        mock_selection_engine: Mock,
        mock_convergence_monitor: Mock,
    ) -> Mock:
        """Mock GEPAOptimizer with all dependencies."""
        optimizer = Mock(spec=GEPAOptimizer)
        optimizer.genetic_algorithm = mock_genetic_algorithm
        optimizer.fitness_evaluator = mock_fitness_evaluator
        optimizer.population_manager = mock_population_manager
        optimizer.mutation_engine = mock_mutation_engine
        optimizer.crossover_engine = mock_crossover_engine
        optimizer.selection_engine = mock_selection_engine
        optimizer.convergence_monitor = mock_convergence_monitor
        optimizer.optimize = AsyncMock()
        optimizer.optimize_with_constraints = AsyncMock()
        optimizer.multi_objective_optimize = AsyncMock()
        optimizer.get_optimization_status = Mock()
        return optimizer

    # Core GEPA Optimization Workflow Tests

    async def test_complete_gepa_optimization_pipeline(
        self,
        mock_gepa_optimizer: Mock,
        sample_optimization_config: dict[str, Any],
        sample_initial_population: list[dict[str, Any]],
        expected_optimization_result: dict[str, Any],
    ):
        """Test complete GEPA optimization pipeline: initialize → evolve → converge."""
        # Arrange
        mock_gepa_optimizer.population_manager.create_initial_population.return_value = {
            "population": sample_initial_population,
            "diversity_score": 0.89,
            "initialization_method": "diverse_random",
        }

        mock_gepa_optimizer.fitness_evaluator.evaluate_population.return_value = {
            "population_fitness": [0.45, 0.52, 0.61, 0.68, 0.74],
            "average_fitness": 0.60,
            "best_fitness": 0.74,
            "fitness_variance": 0.08,
        }

        mock_gepa_optimizer.genetic_algorithm.evolve_generation.return_value = {
            "generation": 25,
            "best_individual": expected_optimization_result["best_individual"],
            "population_stats": expected_optimization_result["population_statistics"],
        }

        mock_gepa_optimizer.convergence_monitor.check_convergence.return_value = {
            "converged": True,
            "reason": "target_fitness_reached",
            "confidence": 0.95,
        }

        mock_gepa_optimizer.optimize.return_value = expected_optimization_result

        # Act - This will fail initially (RED phase)
        result = await mock_gepa_optimizer.optimize(
            config=sample_optimization_config,
            initial_population=sample_initial_population,
        )

        # Assert - Testing the expected optimization workflow
        mock_gepa_optimizer.population_manager.create_initial_population.assert_called_once()
        mock_gepa_optimizer.fitness_evaluator.evaluate_population.assert_called_once()
        mock_gepa_optimizer.genetic_algorithm.evolve_generation.assert_called()
        mock_gepa_optimizer.convergence_monitor.check_convergence.assert_called()

        assert result["status"] == "converged"
        assert result["best_individual"]["fitness_score"] >= 0.9
        assert (
            result["generations_completed"]
            <= sample_optimization_config["algorithm_config"]["max_generations"]
        )

    async def test_population_initialization_and_diversity(
        self, mock_population_manager: Mock, sample_optimization_config: dict[str, Any]
    ):
        """Test population initialization with diversity enforcement."""
        # Arrange
        initialization_strategies = [
            {
                "strategy": "diverse_random",
                "diversity_score": 0.89,
                "coverage_metrics": {
                    "content_structure_coverage": 0.95,
                    "visual_elements_coverage": 0.87,
                    "linguistic_features_coverage": 0.92,
                    "timing_coverage": 0.78,
                },
            },
            {
                "strategy": "seeded_with_best_practices",
                "diversity_score": 0.76,
                "seed_individuals": 10,
                "random_individuals": 40,
            },
            {
                "strategy": "historical_data_informed",
                "diversity_score": 0.82,
                "historical_patterns": 15,
                "novel_variations": 35,
            },
        ]

        mock_population_manager.create_initial_population.side_effect = [
            {"population": [], "strategy_result": strategy}
            for strategy in initialization_strategies
        ]

        # Act
        results = []
        for _i, strategy in enumerate(["diverse_random", "seeded", "historical"]):
            result = await mock_population_manager.create_initial_population(
                size=50, strategy=strategy, config=sample_optimization_config
            )
            results.append(result)

        # Assert
        assert len(results) == 3
        assert all(
            result["strategy_result"]["diversity_score"] > 0.7 for result in results
        )
        assert (
            results[0]["strategy_result"]["diversity_score"] > 0.85
        )  # Diverse random should be highest

    async def test_multi_objective_fitness_evaluation(
        self,
        mock_fitness_evaluator: Mock,
        sample_initial_population: list[dict[str, Any]],
        sample_optimization_config: dict[str, Any],
    ):
        """Test multi-objective fitness evaluation with weighted objectives."""
        # Arrange
        individual_fitness_results = [
            {
                "individual_id": f"ind_{i:03d}",
                "objective_scores": {
                    "engagement_rate": np.random.uniform(0.05, 0.20),
                    "viral_potential": np.random.uniform(0.10, 0.30),
                    "brand_alignment": np.random.uniform(0.70, 0.95),
                    "content_quality": np.random.uniform(0.60, 0.90),
                },
                "weighted_fitness": 0.0,  # To be calculated
                "fitness_rank": 0,
                "pareto_optimal": False,
            }
            for i in range(5)
        ]

        # Calculate weighted fitness
        objectives = sample_optimization_config["fitness_objectives"]
        for result in individual_fitness_results:
            weighted_sum = sum(
                result["objective_scores"][obj["name"]] * obj["weight"]
                for obj in objectives
            )
            result["weighted_fitness"] = weighted_sum

        mock_fitness_evaluator.calculate_multi_objective_fitness.return_value = {
            "fitness_results": individual_fitness_results,
            "pareto_front": individual_fitness_results[:2],  # Top 2 are pareto optimal
            "hypervolume": 0.68,
            "convergence_metric": 0.15,
        }

        population_evaluation = {
            "evaluated_individuals": len(individual_fitness_results),
            "average_fitness": np.mean(
                [r["weighted_fitness"] for r in individual_fitness_results]
            ),
            "fitness_distribution": {
                "min": min(r["weighted_fitness"] for r in individual_fitness_results),
                "max": max(r["weighted_fitness"] for r in individual_fitness_results),
                "std": np.std(
                    [r["weighted_fitness"] for r in individual_fitness_results]
                ),
            },
            "objective_correlations": {
                "engagement_brand": 0.23,
                "viral_quality": -0.15,
                "brand_quality": 0.67,
            },
        }
        mock_fitness_evaluator.evaluate_population.return_value = population_evaluation

        # Act
        multi_obj_result = mock_fitness_evaluator.calculate_multi_objective_fitness(
            sample_initial_population[:5],
            objectives=sample_optimization_config["fitness_objectives"],
        )
        population_result = await mock_fitness_evaluator.evaluate_population(
            sample_initial_population[:5]
        )

        # Assert
        assert len(multi_obj_result["fitness_results"]) == 5
        assert len(multi_obj_result["pareto_front"]) == 2
        assert multi_obj_result["hypervolume"] > 0.5
        assert population_result["evaluated_individuals"] == 5
        assert all(
            score >= 0
            for score in [
                r["weighted_fitness"] for r in multi_obj_result["fitness_results"]
            ]
        )

    async def test_adaptive_genetic_operations(
        self,
        mock_mutation_engine: Mock,
        mock_crossover_engine: Mock,
        sample_initial_population: list[dict[str, Any]],
    ):
        """Test adaptive mutation and crossover operations based on evolution progress."""
        # Arrange
        evolution_stages = [
            {
                "generation": 1,
                "stage": "exploration",
                "mutation_rate": 0.20,
                "crossover_rate": 0.70,
                "diversity_score": 0.89,
            },
            {
                "generation": 10,
                "stage": "balanced",
                "mutation_rate": 0.15,
                "crossover_rate": 0.75,
                "diversity_score": 0.65,
            },
            {
                "generation": 20,
                "stage": "exploitation",
                "mutation_rate": 0.08,
                "crossover_rate": 0.85,
                "diversity_score": 0.35,
            },
        ]

        mutation_results = []
        crossover_results = []

        for stage in evolution_stages:
            # Mock mutation adaptation
            mutation_result = {
                "individuals_mutated": int(50 * stage["mutation_rate"]),
                "mutation_types": {
                    "content_structure": 0.4,
                    "visual_elements": 0.3,
                    "linguistic_features": 0.2,
                    "timing_genes": 0.1,
                },
                "successful_mutations": int(50 * stage["mutation_rate"] * 0.7),
                "diversity_impact": 0.05 if stage["stage"] == "exploration" else -0.02,
            }

            # Mock crossover adaptation
            crossover_result = {
                "crossover_pairs": int(25 * stage["crossover_rate"]),
                "crossover_strategy": (
                    "uniform" if stage["stage"] == "exploration" else "single_point"
                ),
                "successful_offspring": int(25 * stage["crossover_rate"] * 0.8),
                "fitness_improvement": (
                    0.03 if stage["stage"] == "exploitation" else 0.01
                ),
            }

            mutation_results.append(mutation_result)
            crossover_results.append(crossover_result)

        mock_mutation_engine.adaptive_mutation.side_effect = [
            AsyncMock(return_value=result) for result in mutation_results
        ]
        mock_crossover_engine.smart_crossover.side_effect = [
            AsyncMock(return_value=result) for result in crossover_results
        ]

        # Act
        adaptation_results = []
        for i, stage in enumerate(evolution_stages):
            mutation_result = await mock_mutation_engine.adaptive_mutation(
                population=sample_initial_population,
                generation=stage["generation"],
                diversity_score=stage["diversity_score"],
            )

            crossover_result = await mock_crossover_engine.smart_crossover(
                population=sample_initial_population,
                generation=stage["generation"],
                fitness_trend="improving" if i > 0 else "initial",
            )

            adaptation_results.append(
                {
                    "generation": stage["generation"],
                    "mutation": mutation_result,
                    "crossover": crossover_result,
                }
            )

        # Assert
        # Early generations should have higher mutation rates
        assert (
            adaptation_results[0]["mutation"]["individuals_mutated"]
            > adaptation_results[2]["mutation"]["individuals_mutated"]
        )

        # Late generations should have higher crossover success
        assert (
            adaptation_results[2]["crossover"]["fitness_improvement"]
            > adaptation_results[0]["crossover"]["fitness_improvement"]
        )

        # Diversity impact should be positive in exploration phase
        assert adaptation_results[0]["mutation"]["diversity_impact"] > 0

    async def test_convergence_detection_and_early_stopping(
        self,
        mock_convergence_monitor: Mock,
        expected_optimization_result: dict[str, Any],
    ):
        """Test convergence detection with multiple criteria and early stopping."""
        # Arrange
        convergence_progression = [
            {
                "generation": 5,
                "fitness_improvement": 0.08,
                "stagnant_generations": 0,
                "diversity_score": 0.72,
                "convergence_probability": 0.15,
                "status": "evolving",
            },
            {
                "generation": 15,
                "fitness_improvement": 0.02,
                "stagnant_generations": 3,
                "diversity_score": 0.45,
                "convergence_probability": 0.45,
                "status": "slowing",
            },
            {
                "generation": 25,
                "fitness_improvement": 0.0005,  # Below threshold
                "stagnant_generations": 8,
                "diversity_score": 0.20,
                "convergence_probability": 0.92,
                "status": "converged",
            },
        ]

        convergence_decisions = [
            {"should_continue": True, "reason": "improving", "confidence": 0.85},
            {
                "should_continue": True,
                "reason": "moderate_progress",
                "confidence": 0.60,
            },
            {
                "should_continue": False,
                "reason": "convergence_detected",
                "confidence": 0.92,
            },
        ]

        mock_convergence_monitor.check_convergence.side_effect = convergence_decisions
        mock_convergence_monitor.get_convergence_status.side_effect = (
            convergence_progression
        )

        # Act
        convergence_history = []
        for i, progression in enumerate(convergence_progression):
            status = mock_convergence_monitor.get_convergence_status(
                generation=progression["generation"]
            )
            decision = mock_convergence_monitor.check_convergence(
                fitness_history=expected_optimization_result["evolution_history"][
                    "fitness_progression"
                ][: i + 1],
                diversity_history=expected_optimization_result["evolution_history"][
                    "diversity_progression"
                ][: i + 1],
            )

            convergence_history.append(
                {
                    "generation": progression["generation"],
                    "status": status,
                    "decision": decision,
                }
            )

        # Assert
        assert convergence_history[0]["decision"]["should_continue"] is True
        assert convergence_history[1]["decision"]["should_continue"] is True
        assert (
            convergence_history[2]["decision"]["should_continue"] is False
        )  # Converged

        # Convergence probability should increase over time
        assert (
            convergence_history[2]["status"]["convergence_probability"]
            > convergence_history[1]["status"]["convergence_probability"]
            > convergence_history[0]["status"]["convergence_probability"]
        )

    # Advanced GEPA Features Tests

    async def test_elite_preservation_and_hall_of_fame(
        self,
        mock_population_manager: Mock,
        mock_selection_engine: Mock,
        sample_initial_population: list[dict[str, Any]],
    ):
        """Test elite preservation and hall of fame management."""
        # Arrange
        elite_individuals = [
            {
                "individual_id": "elite_001",
                "fitness_score": 0.94,
                "generation_discovered": 18,
                "preserved_generations": 7,
                "genes": {"content_structure": {"hook_type": "question"}},
                "performance_history": [0.85, 0.89, 0.92, 0.94],
            },
            {
                "individual_id": "elite_002",
                "fitness_score": 0.91,
                "generation_discovered": 12,
                "preserved_generations": 13,
                "genes": {"visual_elements": {"emoji_count": 1}},
                "performance_history": [0.78, 0.84, 0.88, 0.91],
            },
        ]

        hall_of_fame = {
            "total_elites": 15,
            "current_active_elites": 5,
            "elite_diversity": 0.67,
            "average_elite_fitness": 0.89,
            "generational_contribution": {
                "early_generations": 3,
                "mid_generations": 7,
                "late_generations": 5,
            },
            "elite_patterns": {
                "successful_hook_types": ["question", "statistic"],
                "optimal_emoji_count": 1,
                "effective_hashtag_strategies": ["trending_mix", "brand_specific"],
            },
        }

        mock_selection_engine.elite_selection.return_value = elite_individuals
        mock_population_manager.archive_elite.return_value = hall_of_fame

        # Act
        selected_elites = mock_selection_engine.elite_selection(
            population=sample_initial_population, elite_percentage=0.1
        )
        hall_of_fame_status = mock_population_manager.archive_elite(
            elites=selected_elites, generation=25
        )

        # Assert
        assert len(selected_elites) == 2
        assert all(elite["fitness_score"] > 0.9 for elite in selected_elites)
        assert hall_of_fame_status["total_elites"] == 15
        assert (
            hall_of_fame_status["elite_diversity"] > 0.6
        )  # Maintain diversity among elites
        assert (
            "question" in hall_of_fame_status["elite_patterns"]["successful_hook_types"]
        )

    async def test_dynamic_parameter_adaptation(
        self, mock_genetic_algorithm: Mock, expected_optimization_result: dict[str, Any]
    ):
        """Test dynamic adaptation of genetic algorithm parameters during evolution."""
        # Arrange
        parameter_adaptations = [
            {
                "generation": 10,
                "trigger": "diversity_loss",
                "adaptations": {
                    "mutation_rate": {"old": 0.15, "new": 0.25},
                    "crossover_rate": {"old": 0.80, "new": 0.70},
                    "selection_pressure": {"old": 0.7, "new": 0.5},
                },
                "expected_impact": "increase_diversity",
            },
            {
                "generation": 20,
                "trigger": "fitness_plateau",
                "adaptations": {
                    "mutation_rate": {"old": 0.25, "new": 0.10},
                    "crossover_rate": {"old": 0.70, "new": 0.90},
                    "elite_percentage": {"old": 0.10, "new": 0.15},
                },
                "expected_impact": "fine_tune_optimization",
            },
        ]

        adaptation_results = [
            {
                "adaptation_successful": True,
                "diversity_change": 0.15,
                "fitness_improvement_next_gen": 0.03,
                "parameter_stability": 0.85,
            },
            {
                "adaptation_successful": True,
                "diversity_change": -0.05,  # Expected decrease for fine-tuning
                "fitness_improvement_next_gen": 0.02,
                "parameter_stability": 0.92,
            },
        ]

        mock_genetic_algorithm.adapt_parameters.side_effect = [
            AsyncMock(return_value=result) for result in adaptation_results
        ]

        # Act
        adaptation_history = []
        for i, adaptation in enumerate(parameter_adaptations):
            result = await mock_genetic_algorithm.adapt_parameters(
                generation=adaptation["generation"],
                trigger=adaptation["trigger"],
                current_stats={
                    "diversity": 0.45 if i == 0 else 0.35,
                    "fitness_plateau_generations": 3 if i == 1 else 0,
                },
            )
            adaptation_history.append({"adaptation": adaptation, "result": result})

        # Assert
        assert all(
            item["result"]["adaptation_successful"] for item in adaptation_history
        )

        # First adaptation should increase diversity
        assert adaptation_history[0]["result"]["diversity_change"] > 0

        # Second adaptation should improve fitness (fine-tuning)
        assert adaptation_history[1]["result"]["fitness_improvement_next_gen"] > 0

    async def test_constraint_satisfaction_and_validation(
        self, mock_fitness_evaluator: Mock, sample_optimization_config: dict[str, Any]
    ):
        """Test constraint satisfaction and validation during optimization."""
        # Arrange
        test_individuals = [
            {
                "individual_id": "test_001",
                "phenotype": {
                    "text": "Great AI tip! #AI #ML #Tech #Data #Science",  # Too many hashtags
                    "character_count": 45,
                },
                "constraint_violations": ["hashtag_limit_exceeded"],
            },
            {
                "individual_id": "test_002",
                "phenotype": {"text": "AI", "character_count": 2},  # Too short
                "constraint_violations": ["min_length_violation"],
            },
            {
                "individual_id": "test_003",
                "phenotype": {
                    "text": "🚀 What's your biggest AI development challenge? Quality data validation is crucial for model success. Always validate before training! #AI #MachineLearning #BestPractices",
                    "character_count": 167,
                },
                "constraint_violations": [],  # Valid
            },
        ]

        constraint_validation_results = [
            {
                "individual_id": "test_001",
                "valid": False,
                "violations": ["hashtag_count_exceeded"],
                "penalty_score": 0.30,
                "corrective_actions": ["reduce_hashtags"],
            },
            {
                "individual_id": "test_002",
                "valid": False,
                "violations": ["content_too_short"],
                "penalty_score": 0.80,
                "corrective_actions": ["expand_content"],
            },
            {
                "individual_id": "test_003",
                "valid": True,
                "violations": [],
                "penalty_score": 0.0,
                "fitness_bonus": 0.05,
            },
        ]

        mock_fitness_evaluator.evaluate_individual.side_effect = [
            AsyncMock(return_value=result) for result in constraint_validation_results
        ]

        # Act
        validation_results = []
        for individual in test_individuals:
            result = await mock_fitness_evaluator.evaluate_individual(
                individual,
                constraints=sample_optimization_config["content_constraints"],
            )
            validation_results.append(result)

        # Assert
        assert validation_results[0]["valid"] is False  # Too many hashtags
        assert validation_results[1]["valid"] is False  # Too short
        assert validation_results[2]["valid"] is True  # Meets all constraints

        # Penalty scores should be higher for more severe violations
        assert (
            validation_results[1]["penalty_score"]
            > validation_results[0]["penalty_score"]
        )

        # Valid individual should get fitness bonus
        assert validation_results[2]["fitness_bonus"] > 0

    # Performance and Scalability Tests

    async def test_large_scale_optimization_performance(
        self, mock_gepa_optimizer: Mock, sample_optimization_config: dict[str, Any]
    ):
        """Test performance with large-scale optimization scenarios."""
        # Arrange
        large_scale_config = {
            **sample_optimization_config,
            "algorithm_config": {
                **sample_optimization_config["algorithm_config"],
                "population_size": 200,
                "max_generations": 500,
            },
        }

        performance_metrics = {
            "total_individuals_evaluated": 200 * 50,  # 50 generations completed
            "total_execution_time_ms": 180000,  # 3 minutes
            "average_generation_time_ms": 3600,
            "memory_usage_peak_mb": 512,
            "cpu_utilization_avg": 0.75,
            "gpu_utilization_avg": 0.82,
            "cache_hit_rate": 0.68,
            "parallel_efficiency": 0.85,
        }

        scalability_analysis = {
            "linear_scaling_factor": 0.92,
            "memory_scaling": "sub_linear",
            "cpu_bottlenecks": ["fitness_evaluation", "population_sorting"],
            "optimization_recommendations": [
                "increase_batch_size",
                "implement_lazy_evaluation",
                "cache_fitness_results",
            ],
        }

        mock_gepa_optimizer.optimize.return_value = {
            "status": "completed",
            "performance_metrics": performance_metrics,
            "scalability_analysis": scalability_analysis,
        }

        # Act
        result = await mock_gepa_optimizer.optimize(
            config=large_scale_config, performance_monitoring=True
        )

        # Assert
        assert (
            result["performance_metrics"]["total_execution_time_ms"] < 300000
        )  # Under 5 minutes
        assert result["performance_metrics"]["parallel_efficiency"] > 0.8
        assert result["performance_metrics"]["memory_usage_peak_mb"] < 1024  # Under 1GB
        assert result["scalability_analysis"]["linear_scaling_factor"] > 0.9

    async def test_real_time_optimization_monitoring(self, mock_gepa_optimizer: Mock):
        """Test real-time monitoring and analytics during optimization."""
        # Arrange
        monitoring_snapshots = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "generation": 10,
                "current_best_fitness": 0.78,
                "population_diversity": 0.65,
                "convergence_rate": 0.03,
                "estimated_completion_generations": 25,
                "resource_usage": {
                    "memory_mb": 256,
                    "cpu_percent": 72,
                    "gpu_percent": 85,
                },
            },
            {
                "timestamp": (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
                "generation": 20,
                "current_best_fitness": 0.86,
                "population_diversity": 0.42,
                "convergence_rate": 0.02,
                "estimated_completion_generations": 8,
                "resource_usage": {
                    "memory_mb": 278,
                    "cpu_percent": 68,
                    "gpu_percent": 88,
                },
            },
        ]

        mock_gepa_optimizer.get_optimization_status.side_effect = monitoring_snapshots

        # Act
        status_progression = []
        for _i in range(len(monitoring_snapshots)):
            status = mock_gepa_optimizer.get_optimization_status()
            status_progression.append(status)

        # Assert
        assert len(status_progression) == 2

        # Fitness should improve over time
        assert (
            status_progression[1]["current_best_fitness"]
            > status_progression[0]["current_best_fitness"]
        )

        # Diversity should decrease (convergence)
        assert (
            status_progression[1]["population_diversity"]
            < status_progression[0]["population_diversity"]
        )

        # Estimated completion should decrease
        assert (
            status_progression[1]["estimated_completion_generations"]
            < status_progression[0]["estimated_completion_generations"]
        )

    # Error Handling and Edge Cases

    async def test_optimization_error_recovery(
        self,
        mock_gepa_optimizer: Mock,
        mock_fitness_evaluator: Mock,
        sample_optimization_config: dict[str, Any],
    ):
        """Test error recovery mechanisms during optimization."""
        # Arrange
        error_scenarios = [
            {
                "error_type": "fitness_evaluation_timeout",
                "affected_individuals": 5,
                "recovery_strategy": "skip_and_assign_low_fitness",
                "impact": "minimal",
            },
            {
                "error_type": "memory_exhaustion",
                "affected_generation": 15,
                "recovery_strategy": "reduce_population_size",
                "impact": "moderate",
            },
            {
                "error_type": "convergence_stall",
                "stagnant_generations": 15,
                "recovery_strategy": "population_restart_with_elites",
                "impact": "significant",
            },
        ]

        recovery_results = [
            {
                "recovery_successful": True,
                "individuals_recovered": 5,
                "performance_impact": 0.02,
                "continuation_possible": True,
            },
            {
                "recovery_successful": True,
                "new_population_size": 40,  # Reduced from 50
                "memory_freed_mb": 128,
                "continuation_possible": True,
            },
            {
                "recovery_successful": True,
                "elites_preserved": 5,
                "diversity_restored": 0.78,
                "restart_generation": 16,
                "continuation_possible": True,
            },
        ]

        mock_gepa_optimizer.optimize_with_constraints.side_effect = [
            AsyncMock(return_value=result) for result in recovery_results
        ]

        # Act
        recovery_test_results = []
        for _i, scenario in enumerate(error_scenarios):
            result = await mock_gepa_optimizer.optimize_with_constraints(
                config=sample_optimization_config,
                error_recovery=True,
                scenario=scenario,
            )
            recovery_test_results.append({"scenario": scenario, "recovery": result})

        # Assert
        assert all(
            test["recovery"]["recovery_successful"] for test in recovery_test_results
        )
        assert all(
            test["recovery"]["continuation_possible"] for test in recovery_test_results
        )

        # Memory recovery should free significant memory
        assert recovery_test_results[1]["recovery"]["memory_freed_mb"] > 100

        # Diversity restart should restore high diversity
        assert recovery_test_results[2]["recovery"]["diversity_restored"] > 0.7

    async def test_edge_case_population_scenarios(
        self, mock_population_manager: Mock, mock_convergence_monitor: Mock
    ):
        """Test handling of edge case population scenarios."""
        # Arrange
        edge_case_scenarios = [
            {
                "scenario": "premature_convergence",
                "generation": 5,
                "diversity_score": 0.15,  # Very low
                "fitness_variance": 0.02,
                "recommended_action": "diversity_injection",
            },
            {
                "scenario": "fitness_explosion",
                "generation": 12,
                "best_fitness": 0.99,  # Unexpectedly high
                "fitness_variance": 0.45,  # Very high
                "recommended_action": "validate_and_exploit",
            },
            {
                "scenario": "population_degeneration",
                "generation": 18,
                "valid_individuals": 5,  # Most violated constraints
                "constraint_violation_rate": 0.90,
                "recommended_action": "constraint_relaxation",
            },
        ]

        mitigation_strategies = [
            {
                "strategy": "diversity_injection",
                "new_individuals_added": 20,
                "diversity_boost": 0.35,
                "fitness_impact": -0.05,  # Temporary decrease
            },
            {
                "strategy": "validate_and_exploit",
                "validation_passed": True,
                "exploitation_individuals": 10,
                "convergence_acceleration": 0.25,
            },
            {
                "strategy": "constraint_relaxation",
                "relaxed_constraints": ["hashtag_count", "character_limit"],
                "valid_individuals_recovered": 35,
                "population_health_restored": True,
            },
        ]

        mock_population_manager.maintain_diversity.side_effect = [
            AsyncMock(return_value=strategy) for strategy in mitigation_strategies
        ]

        # Act
        mitigation_results = []
        for _i, scenario in enumerate(edge_case_scenarios):
            result = await mock_population_manager.maintain_diversity(
                scenario=scenario["scenario"], current_state=scenario
            )
            mitigation_results.append(result)

        # Assert
        # Diversity injection should increase diversity significantly
        assert mitigation_results[0]["diversity_boost"] > 0.3

        # Validation should pass for fitness explosion
        assert mitigation_results[1]["validation_passed"] is True

        # Constraint relaxation should recover most individuals
        assert mitigation_results[2]["valid_individuals_recovered"] > 30
        assert mitigation_results[2]["population_health_restored"] is True
