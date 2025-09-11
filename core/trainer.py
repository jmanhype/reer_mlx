"""T017: GEPA trainer for optimization implementation.

Implements Genetic Evolution Post Analyzer (GEPA) trainer that uses genetic algorithms
to optimize social media content generation and strategy evolution. Provides
multi-objective optimization, population management, and convergence monitoring.
"""

import asyncio
import random
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import json
from abc import ABC, abstractmethod

from .exceptions import TrainingError, OptimizationError, ConvergenceError, FitnessError
from .candidate_scorer import REERCandidateScorer, ContentCandidate, ScoringMetrics


class SelectionMethod(Enum):
    """Available selection methods for genetic algorithm."""

    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITISM = "elitism"


class CrossoverMethod(Enum):
    """Available crossover methods for genetic algorithm."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SEMANTIC = "semantic"


class MutationMethod(Enum):
    """Available mutation methods for genetic algorithm."""

    RANDOM_WORD = "random_word"
    SYNONYM_REPLACEMENT = "synonym_replacement"
    STRUCTURE_MODIFICATION = "structure_modification"
    FEATURE_INJECTION = "feature_injection"


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""

    individual_id: str
    genes: Dict[str, Any]  # Content generation parameters
    phenotype: str  # Generated content
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    overall_fitness: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """Represents a population of individuals."""

    population_id: str
    individuals: List[Individual]
    generation: int
    population_size: int
    diversity_score: float = 0.0
    avg_fitness: float = 0.0
    best_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptimizationConfig:
    """Configuration for GEPA optimization."""

    population_size: int = 50
    max_generations: int = 100
    elite_percentage: float = 0.1
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SEMANTIC
    mutation_method: MutationMethod = MutationMethod.FEATURE_INJECTION
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnant_generations: int = 10
    diversity_threshold: float = 0.15
    target_fitness: float = 0.9
    fitness_objectives: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "engagement_rate", "weight": 0.4, "target": "maximize"},
            {"name": "quality_score", "weight": 0.3, "target": "maximize"},
            {"name": "viral_potential", "weight": 0.2, "target": "maximize"},
            {"name": "brand_alignment", "weight": 0.1, "target": "maximize"},
        ]
    )


@dataclass
class OptimizationResult:
    """Result of GEPA optimization process."""

    optimization_id: str
    best_individual: Individual
    final_population: Population
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    total_generations: int
    total_evaluations: int
    optimization_time_seconds: float
    success: bool
    error_message: Optional[str] = None


class FitnessEvaluator:
    """Evaluates fitness of individuals using multiple objectives."""

    def __init__(self, scorer: REERCandidateScorer, objectives: List[Dict[str, Any]]):
        """Initialize fitness evaluator.

        Args:
            scorer: Content scorer for evaluation
            objectives: List of fitness objectives with weights
        """
        self.scorer = scorer
        self.objectives = objectives
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize objective weights to sum to 1.0."""
        total_weight = sum(obj["weight"] for obj in self.objectives)
        if total_weight > 0:
            for obj in self.objectives:
                obj["weight"] /= total_weight

    async def evaluate_individual(self, individual: Individual) -> float:
        """Evaluate fitness of an individual.

        Args:
            individual: Individual to evaluate

        Returns:
            Overall fitness score
        """
        try:
            # Create content candidate
            candidate = ContentCandidate(
                candidate_id=individual.individual_id,
                text=individual.phenotype,
                metadata=individual.metadata,
            )

            # Score the candidate
            metrics = await self.scorer.score_candidate(candidate)

            # Calculate multi-objective fitness
            fitness_scores = {}
            overall_fitness = 0.0

            for objective in self.objectives:
                obj_name = objective["name"]
                weight = objective["weight"]
                target = objective["target"]

                # Map objective names to scoring metrics
                score = self._get_objective_score(obj_name, metrics)
                fitness_scores[obj_name] = score

                # Apply weight
                if target == "maximize":
                    overall_fitness += score * weight
                elif target == "minimize":
                    overall_fitness += (1.0 - score) * weight

            # Store fitness scores
            individual.fitness_scores = fitness_scores
            individual.overall_fitness = overall_fitness

            return overall_fitness

        except Exception as e:
            raise FitnessError(
                f"Failed to evaluate individual {individual.individual_id}: {str(e)}",
                details={"individual_id": individual.individual_id},
                original_error=e,
            )

    def _get_objective_score(
        self, objective_name: str, metrics: ScoringMetrics
    ) -> float:
        """Map objective name to scoring metric."""
        mapping = {
            "engagement_rate": metrics.engagement_score,
            "quality_score": metrics.quality_score,
            "viral_potential": metrics.viral_potential,
            "brand_alignment": metrics.brand_alignment,
            "fluency": metrics.fluency_score,
            "coherence": metrics.coherence_score,
            "relevance": metrics.relevance_score,
            "perplexity": metrics.perplexity,
            "overall": metrics.overall_score,
        }

        return mapping.get(objective_name, 0.5)  # Default to neutral score

    async def evaluate_population(self, population: Population) -> None:
        """Evaluate fitness for entire population."""
        for individual in population.individuals:
            await self.evaluate_individual(individual)

        # Update population statistics
        fitness_scores = [ind.overall_fitness for ind in population.individuals]
        population.avg_fitness = float(np.mean(fitness_scores))
        population.best_fitness = float(np.max(fitness_scores))
        population.worst_fitness = float(np.min(fitness_scores))
        population.fitness_std = float(np.std(fitness_scores))


class ContentGenerator:
    """Generates content based on genetic parameters."""

    def __init__(self):
        """Initialize content generator."""
        self.content_templates = {
            "announcement": [
                "🚀 Excited to share {content}! {cta}",
                "Just launched {content}! What do you think? {cta}",
                "Big news: {content} {cta}",
            ],
            "tip": [
                "Pro tip: {content} {cta}",
                "Quick tip for {audience}: {content} {cta}",
                "Here's how to {content} {cta}",
            ],
            "question": [
                "What's your take on {content}? {cta}",
                "How do you handle {content}? {cta}",
                "Anyone else struggling with {content}? {cta}",
            ],
            "insight": [
                "Just realized that {content} {cta}",
                "Interesting insight: {content} {cta}",
                "Key learning: {content} {cta}",
            ],
        }

        self.cta_options = [
            "Let me know your thoughts!",
            "What's your experience?",
            "Share your tips below 👇",
            "Tag someone who needs this!",
            "Drop a comment if you agree!",
            "Try it and let me know how it goes!",
            "What would you add to this list?",
        ]

        self.hashtag_pools = {
            "tech": ["#AI", "#MachineLearning", "#Tech", "#Innovation", "#Programming"],
            "business": [
                "#Business",
                "#Entrepreneurship",
                "#Startup",
                "#Leadership",
                "#Growth",
            ],
            "marketing": [
                "#Marketing",
                "#SocialMedia",
                "#ContentStrategy",
                "#Branding",
                "#DigitalMarketing",
            ],
            "productivity": [
                "#Productivity",
                "#Tips",
                "#Efficiency",
                "#TimeManagement",
                "#Optimization",
            ],
        }

    def generate_content(self, genes: Dict[str, Any]) -> str:
        """Generate content based on genetic parameters.

        Args:
            genes: Genetic parameters for content generation

        Returns:
            Generated content string
        """
        # Extract genetic parameters
        content_type = genes.get("content_type", "announcement")
        topic = genes.get("topic", "innovation")
        audience = genes.get("audience", "professionals")
        use_hashtags = genes.get("use_hashtags", True)
        hashtag_category = genes.get("hashtag_category", "tech")
        hashtag_count = genes.get("hashtag_count", 2)
        use_cta = genes.get("use_cta", True)
        emoji_intensity = genes.get("emoji_intensity", 0.5)

        # Select content template
        templates = self.content_templates.get(
            content_type, self.content_templates["announcement"]
        )
        template = random.choice(templates)

        # Generate content based on topic
        content_part = self._generate_content_part(topic, content_type)

        # Select CTA
        cta = random.choice(self.cta_options) if use_cta else ""

        # Generate content
        content = template.format(content=content_part, audience=audience, cta=cta)

        # Add hashtags
        if use_hashtags:
            hashtags = self._select_hashtags(hashtag_category, hashtag_count)
            if hashtags:
                content += " " + " ".join(hashtags)

        # Add emojis based on intensity
        if emoji_intensity > 0.3:
            content = self._add_emojis(content, emoji_intensity)

        return content.strip()

    def _generate_content_part(self, topic: str, content_type: str) -> str:
        """Generate the main content part based on topic and type."""
        content_variants = {
            "AI": [
                "our new AI-powered feature",
                "breakthrough AI technology",
                "AI automation that saves hours",
                "intelligent AI solutions",
                "next-gen AI capabilities",
            ],
            "productivity": [
                "boost your productivity by 50%",
                "streamline your workflow",
                "optimize your daily routine",
                "maximize your efficiency",
                "eliminate time-wasting tasks",
            ],
            "innovation": [
                "revolutionary innovation",
                "game-changing technology",
                "cutting-edge solutions",
                "innovative approaches",
                "breakthrough methodologies",
            ],
            "business": [
                "scale your business",
                "drive business growth",
                "improve business processes",
                "optimize business operations",
                "enhance business performance",
            ],
        }

        # Find best matching topic
        best_match = "innovation"  # default
        for key in content_variants.keys():
            if key.lower() in topic.lower():
                best_match = key
                break

        variants = content_variants[best_match]
        return random.choice(variants)

    def _select_hashtags(self, category: str, count: int) -> List[str]:
        """Select hashtags from specified category."""
        hashtags = self.hashtag_pools.get(category, self.hashtag_pools["tech"])
        selected_count = min(count, len(hashtags))
        return random.sample(hashtags, selected_count)

    def _add_emojis(self, content: str, intensity: float) -> str:
        """Add emojis to content based on intensity."""
        emoji_options = ["🚀", "💡", "⭐", "🔥", "💯", "✨", "🎯", "🔧", "📈", "🏆"]

        if intensity > 0.7:
            # High intensity - add 2-3 emojis
            emoji_count = random.randint(2, 3)
        elif intensity > 0.5:
            # Medium intensity - add 1-2 emojis
            emoji_count = random.randint(1, 2)
        else:
            # Low intensity - add 0-1 emojis
            emoji_count = random.randint(0, 1)

        if emoji_count > 0:
            selected_emojis = random.sample(
                emoji_options, min(emoji_count, len(emoji_options))
            )
            # Add emojis at the beginning or end
            if random.choice([True, False]):
                content = " ".join(selected_emojis) + " " + content
            else:
                content = content + " " + " ".join(selected_emojis)

        return content


class GeneticOperators:
    """Implements genetic algorithm operators (selection, crossover, mutation)."""

    def __init__(self, config: OptimizationConfig):
        """Initialize genetic operators.

        Args:
            config: Optimization configuration
        """
        self.config = config

    def select_parents(
        self, population: Population, num_parents: int
    ) -> List[Individual]:
        """Select parents for reproduction.

        Args:
            population: Current population
            num_parents: Number of parents to select

        Returns:
            List of selected parent individuals
        """
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, num_parents)
        elif self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(population, num_parents)
        elif self.config.selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(population, num_parents)
        else:
            return self._tournament_selection(population, num_parents)

    def _tournament_selection(
        self, population: Population, num_parents: int
    ) -> List[Individual]:
        """Tournament selection method."""
        parents = []

        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_candidates = random.sample(
                population.individuals,
                min(self.config.tournament_size, len(population.individuals)),
            )

            # Select best from tournament
            best = max(tournament_candidates, key=lambda ind: ind.overall_fitness)
            parents.append(best)

        return parents

    def _roulette_wheel_selection(
        self, population: Population, num_parents: int
    ) -> List[Individual]:
        """Roulette wheel selection method."""
        # Calculate selection probabilities
        fitness_scores = [ind.overall_fitness for ind in population.individuals]
        min_fitness = min(fitness_scores)

        # Shift fitness scores to be positive
        adjusted_scores = [score - min_fitness + 0.01 for score in fitness_scores]
        total_fitness = sum(adjusted_scores)

        probabilities = [score / total_fitness for score in adjusted_scores]

        # Select parents
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative_prob = 0.0

            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population.individuals[i])
                    break

        return parents

    def _rank_based_selection(
        self, population: Population, num_parents: int
    ) -> List[Individual]:
        """Rank-based selection method."""
        # Sort by fitness
        sorted_individuals = sorted(
            population.individuals, key=lambda ind: ind.overall_fitness, reverse=True
        )

        # Assign rank-based probabilities
        pop_size = len(sorted_individuals)
        probabilities = [
            (pop_size - rank) / sum(range(1, pop_size + 1)) for rank in range(pop_size)
        ]

        # Select parents
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative_prob = 0.0

            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(sorted_individuals[i])
                    break

        return parents

    def crossover(
        self, parent1: Individual, parent2: Individual, generation: int
    ) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent
            generation: Current generation number

        Returns:
            Tuple of two offspring individuals
        """
        if self.config.crossover_method == CrossoverMethod.SEMANTIC:
            return self._semantic_crossover(parent1, parent2, generation)
        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2, generation)
        else:
            return self._semantic_crossover(parent1, parent2, generation)

    def _semantic_crossover(
        self, parent1: Individual, parent2: Individual, generation: int
    ) -> Tuple[Individual, Individual]:
        """Semantic crossover for content generation genes."""
        # Create offspring genes by combining parent genes
        offspring1_genes = parent1.genes.copy()
        offspring2_genes = parent2.genes.copy()

        # Crossover genetic parameters
        gene_keys = list(parent1.genes.keys())
        crossover_point = random.randint(1, len(gene_keys) - 1)

        for i, key in enumerate(gene_keys):
            if i >= crossover_point:
                offspring1_genes[key] = parent2.genes.get(key, parent1.genes[key])
                offspring2_genes[key] = parent1.genes.get(key, parent2.genes[key])

        # Create offspring individuals
        content_generator = ContentGenerator()

        offspring1 = Individual(
            individual_id=f"ind_{generation}_{random.randint(1000, 9999)}",
            genes=offspring1_genes,
            phenotype=content_generator.generate_content(offspring1_genes),
            generation=generation,
            parent_ids=[parent1.individual_id, parent2.individual_id],
        )

        offspring2 = Individual(
            individual_id=f"ind_{generation}_{random.randint(1000, 9999)}",
            genes=offspring2_genes,
            phenotype=content_generator.generate_content(offspring2_genes),
            generation=generation,
            parent_ids=[parent1.individual_id, parent2.individual_id],
        )

        return offspring1, offspring2

    def _uniform_crossover(
        self, parent1: Individual, parent2: Individual, generation: int
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover method."""
        # Randomly select genes from each parent
        offspring1_genes = {}
        offspring2_genes = {}

        for key in parent1.genes.keys():
            if random.random() < 0.5:
                offspring1_genes[key] = parent1.genes[key]
                offspring2_genes[key] = parent2.genes.get(key, parent1.genes[key])
            else:
                offspring1_genes[key] = parent2.genes.get(key, parent1.genes[key])
                offspring2_genes[key] = parent1.genes[key]

        # Generate content
        content_generator = ContentGenerator()

        offspring1 = Individual(
            individual_id=f"ind_{generation}_{random.randint(1000, 9999)}",
            genes=offspring1_genes,
            phenotype=content_generator.generate_content(offspring1_genes),
            generation=generation,
            parent_ids=[parent1.individual_id, parent2.individual_id],
        )

        offspring2 = Individual(
            individual_id=f"ind_{generation}_{random.randint(1000, 9999)}",
            genes=offspring2_genes,
            phenotype=content_generator.generate_content(offspring2_genes),
            generation=generation,
            parent_ids=[parent1.individual_id, parent2.individual_id],
        )

        return offspring1, offspring2

    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if random.random() < self.config.mutation_rate:
            if self.config.mutation_method == MutationMethod.FEATURE_INJECTION:
                return self._feature_injection_mutation(individual)
            elif self.config.mutation_method == MutationMethod.STRUCTURE_MODIFICATION:
                return self._structure_modification_mutation(individual)
            else:
                return self._feature_injection_mutation(individual)

        return individual

    def _feature_injection_mutation(self, individual: Individual) -> Individual:
        """Feature injection mutation - modify genetic parameters."""
        mutated_genes = individual.genes.copy()

        # Randomly select a gene to mutate
        gene_keys = list(mutated_genes.keys())
        if gene_keys:
            key_to_mutate = random.choice(gene_keys)

            # Apply mutation based on gene type
            if key_to_mutate == "hashtag_count":
                mutated_genes[key_to_mutate] = random.randint(0, 5)
            elif key_to_mutate == "emoji_intensity":
                mutated_genes[key_to_mutate] = random.uniform(0.0, 1.0)
            elif key_to_mutate == "use_hashtags":
                mutated_genes[key_to_mutate] = not mutated_genes[key_to_mutate]
            elif key_to_mutate == "use_cta":
                mutated_genes[key_to_mutate] = not mutated_genes[key_to_mutate]
            elif key_to_mutate == "content_type":
                types = ["announcement", "tip", "question", "insight"]
                mutated_genes[key_to_mutate] = random.choice(types)
            elif key_to_mutate == "hashtag_category":
                categories = ["tech", "business", "marketing", "productivity"]
                mutated_genes[key_to_mutate] = random.choice(categories)

        # Regenerate content
        content_generator = ContentGenerator()
        individual.genes = mutated_genes
        individual.phenotype = content_generator.generate_content(mutated_genes)

        return individual

    def _structure_modification_mutation(self, individual: Individual) -> Individual:
        """Structure modification mutation - modify content structure."""
        # This is a simpler mutation that just regenerates content with same genes
        content_generator = ContentGenerator()
        individual.phenotype = content_generator.generate_content(individual.genes)
        return individual


class ConvergenceMonitor:
    """Monitors convergence of the genetic algorithm."""

    def __init__(self, config: OptimizationConfig):
        """Initialize convergence monitor.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.stagnant_generations = 0
        self.converged = False
        self.convergence_reason = ""

    def update(self, population: Population) -> bool:
        """Update convergence metrics and check for convergence.

        Args:
            population: Current population

        Returns:
            True if algorithm has converged
        """
        # Update fitness history
        self.fitness_history.append(population.best_fitness)
        self.diversity_history.append(population.diversity_score)

        # Check for target fitness reached
        if population.best_fitness >= self.config.target_fitness:
            self.converged = True
            self.convergence_reason = "Target fitness reached"
            return True

        # Check for fitness stagnation
        if len(self.fitness_history) >= 2:
            fitness_improvement = self.fitness_history[-1] - self.fitness_history[-2]

            if fitness_improvement < self.config.convergence_threshold:
                self.stagnant_generations += 1
            else:
                self.stagnant_generations = 0

            if self.stagnant_generations >= self.config.max_stagnant_generations:
                self.converged = True
                self.convergence_reason = "Fitness stagnation"
                return True

        # Check for diversity loss
        if population.diversity_score < self.config.diversity_threshold:
            self.converged = True
            self.convergence_reason = "Diversity loss"
            return True

        return False

    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information."""
        return {
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "stagnant_generations": self.stagnant_generations,
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
            "final_fitness": self.fitness_history[-1] if self.fitness_history else 0.0,
            "final_diversity": (
                self.diversity_history[-1] if self.diversity_history else 0.0
            ),
        }


class REERGEPATrainer:
    """Main GEPA trainer for content optimization using genetic algorithms.

    Implements a complete genetic algorithm system for optimizing social media
    content generation with multi-objective fitness evaluation, adaptive operators,
    and convergence monitoring.
    """

    def __init__(
        self, scorer: REERCandidateScorer, config: Optional[OptimizationConfig] = None
    ):
        """Initialize GEPA trainer.

        Args:
            scorer: Content scorer for fitness evaluation
            config: Optional optimization configuration
        """
        self.scorer = scorer
        self.config = config or OptimizationConfig()
        self.fitness_evaluator = FitnessEvaluator(
            scorer, self.config.fitness_objectives
        )
        self.genetic_operators = GeneticOperators(self.config)
        self.convergence_monitor = ConvergenceMonitor(self.config)
        self.content_generator = ContentGenerator()

        # Optimization state
        self.current_population: Optional[Population] = None
        self.optimization_history: List[Dict[str, Any]] = []
        self.total_evaluations = 0

    def _create_initial_population(
        self, optimization_target: Dict[str, Any]
    ) -> Population:
        """Create initial population with diverse individuals.

        Args:
            optimization_target: Target parameters for optimization

        Returns:
            Initial population
        """
        individuals = []

        for i in range(self.config.population_size):
            # Create diverse genetic parameters
            genes = {
                "content_type": random.choice(
                    ["announcement", "tip", "question", "insight"]
                ),
                "topic": optimization_target.get("topic", "innovation"),
                "audience": optimization_target.get("audience", "professionals"),
                "use_hashtags": random.choice([True, False]),
                "hashtag_category": random.choice(
                    ["tech", "business", "marketing", "productivity"]
                ),
                "hashtag_count": random.randint(0, 5),
                "use_cta": random.choice([True, False]),
                "emoji_intensity": random.uniform(0.0, 1.0),
            }

            # Generate content
            phenotype = self.content_generator.generate_content(genes)

            # Create individual
            individual = Individual(
                individual_id=f"ind_0_{i:04d}",
                genes=genes,
                phenotype=phenotype,
                generation=0,
            )

            individuals.append(individual)

        return Population(
            population_id=f"pop_0_{int(datetime.now().timestamp())}",
            individuals=individuals,
            generation=0,
            population_size=self.config.population_size,
        )

    def _calculate_population_diversity(self, population: Population) -> float:
        """Calculate diversity score for population.

        Args:
            population: Population to analyze

        Returns:
            Diversity score between 0.0 and 1.0
        """
        if len(population.individuals) < 2:
            return 0.0

        # Calculate diversity based on genetic parameters
        gene_diversity_scores = []

        # Get all gene keys
        all_gene_keys = set()
        for individual in population.individuals:
            all_gene_keys.update(individual.genes.keys())

        for gene_key in all_gene_keys:
            gene_values = []
            for individual in population.individuals:
                if gene_key in individual.genes:
                    gene_values.append(individual.genes[gene_key])

            if gene_values:
                # Calculate diversity for this gene
                if all(isinstance(v, bool) for v in gene_values):
                    # Boolean diversity
                    true_count = sum(gene_values)
                    diversity = min(true_count, len(gene_values) - true_count) / len(
                        gene_values
                    )
                elif all(isinstance(v, (int, float)) for v in gene_values):
                    # Numeric diversity (normalized standard deviation)
                    if len(set(gene_values)) > 1:
                        diversity = np.std(gene_values) / (np.mean(gene_values) + 0.001)
                        diversity = min(diversity, 1.0)
                    else:
                        diversity = 0.0
                else:
                    # Categorical diversity
                    unique_values = len(set(str(v) for v in gene_values))
                    diversity = unique_values / len(gene_values)

                gene_diversity_scores.append(diversity)

        # Return average diversity across all genes
        return float(np.mean(gene_diversity_scores)) if gene_diversity_scores else 0.0

    async def optimize(
        self,
        optimization_target: Dict[str, Any],
        initial_population: Optional[Population] = None,
    ) -> OptimizationResult:
        """Run GEPA optimization process.

        Args:
            optimization_target: Target parameters for optimization
            initial_population: Optional initial population

        Returns:
            OptimizationResult with optimization results
        """
        start_time = datetime.now()
        optimization_id = f"gepa_{int(start_time.timestamp())}"

        try:
            # Initialize scorer
            await self.scorer.initialize()

            # Create or use provided initial population
            if initial_population is None:
                population = self._create_initial_population(optimization_target)
            else:
                population = initial_population

            self.current_population = population

            # Evaluate initial population
            await self.fitness_evaluator.evaluate_population(population)
            population.diversity_score = self._calculate_population_diversity(
                population
            )
            self.total_evaluations += len(population.individuals)

            # Record initial generation
            self.optimization_history.append(
                {
                    "generation": 0,
                    "best_fitness": population.best_fitness,
                    "avg_fitness": population.avg_fitness,
                    "diversity_score": population.diversity_score,
                    "evaluations": self.total_evaluations,
                }
            )

            # Evolution loop
            generation = 1
            while generation <= self.config.max_generations:
                # Check for convergence
                if self.convergence_monitor.update(population):
                    break

                # Create new generation
                new_population = await self._create_next_generation(
                    population, generation
                )

                # Evaluate new population
                await self.fitness_evaluator.evaluate_population(new_population)
                new_population.diversity_score = self._calculate_population_diversity(
                    new_population
                )
                self.total_evaluations += len(new_population.individuals)

                # Record generation
                self.optimization_history.append(
                    {
                        "generation": generation,
                        "best_fitness": new_population.best_fitness,
                        "avg_fitness": new_population.avg_fitness,
                        "diversity_score": new_population.diversity_score,
                        "evaluations": self.total_evaluations,
                    }
                )

                # Update current population
                population = new_population
                self.current_population = population
                generation += 1

            # Find best individual
            best_individual = max(
                population.individuals, key=lambda ind: ind.overall_fitness
            )

            # Calculate performance metrics
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()

            performance_metrics = {
                "generations_completed": generation - 1,
                "total_evaluations": self.total_evaluations,
                "optimization_time_seconds": optimization_time,
                "evaluations_per_second": (
                    self.total_evaluations / optimization_time
                    if optimization_time > 0
                    else 0
                ),
                "final_best_fitness": best_individual.overall_fitness,
                "fitness_improvement": best_individual.overall_fitness
                - self.optimization_history[0]["best_fitness"],
                "convergence_generation": generation - 1,
            }

            return OptimizationResult(
                optimization_id=optimization_id,
                best_individual=best_individual,
                final_population=population,
                optimization_history=self.optimization_history,
                convergence_info=self.convergence_monitor.get_convergence_info(),
                performance_metrics=performance_metrics,
                total_generations=generation - 1,
                total_evaluations=self.total_evaluations,
                optimization_time_seconds=optimization_time,
                success=True,
            )

        except Exception as e:
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()

            return OptimizationResult(
                optimization_id=optimization_id,
                best_individual=Individual("error", {}, ""),
                final_population=Population("error", [], 0, 0),
                optimization_history=self.optimization_history,
                convergence_info=self.convergence_monitor.get_convergence_info(),
                performance_metrics={},
                total_generations=0,
                total_evaluations=self.total_evaluations,
                optimization_time_seconds=optimization_time,
                success=False,
                error_message=str(e),
            )

    async def _create_next_generation(
        self, current_population: Population, generation: int
    ) -> Population:
        """Create the next generation using genetic operators.

        Args:
            current_population: Current population
            generation: Generation number

        Returns:
            New population for next generation
        """
        new_individuals = []

        # Elitism - keep best individuals
        elite_count = int(self.config.elite_percentage * self.config.population_size)
        if elite_count > 0:
            elite_individuals = sorted(
                current_population.individuals,
                key=lambda ind: ind.overall_fitness,
                reverse=True,
            )[:elite_count]

            # Create copies of elite individuals for new generation
            for elite in elite_individuals:
                new_individual = Individual(
                    individual_id=f"ind_{generation}_{len(new_individuals):04d}",
                    genes=elite.genes.copy(),
                    phenotype=elite.phenotype,
                    generation=generation,
                    parent_ids=[elite.individual_id],
                )
                new_individuals.append(new_individual)

        # Generate offspring to fill remaining population
        offspring_needed = self.config.population_size - len(new_individuals)
        offspring_count = 0

        while offspring_count < offspring_needed:
            # Select parents
            parents = self.genetic_operators.select_parents(current_population, 2)

            if len(parents) >= 2:
                parent1, parent2 = parents[0], parents[1]

                # Crossover
                if random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = self.genetic_operators.crossover(
                        parent1, parent2, generation
                    )
                else:
                    # No crossover - create copies
                    offspring1 = Individual(
                        individual_id=f"ind_{generation}_{len(new_individuals) + offspring_count:04d}",
                        genes=parent1.genes.copy(),
                        phenotype=parent1.phenotype,
                        generation=generation,
                        parent_ids=[parent1.individual_id],
                    )
                    offspring2 = Individual(
                        individual_id=f"ind_{generation}_{len(new_individuals) + offspring_count + 1:04d}",
                        genes=parent2.genes.copy(),
                        phenotype=parent2.phenotype,
                        generation=generation,
                        parent_ids=[parent2.individual_id],
                    )

                # Mutation
                offspring1 = self.genetic_operators.mutate(offspring1)
                offspring2 = self.genetic_operators.mutate(offspring2)

                # Add offspring to new generation
                if offspring_count < offspring_needed:
                    new_individuals.append(offspring1)
                    offspring_count += 1

                if offspring_count < offspring_needed:
                    new_individuals.append(offspring2)
                    offspring_count += 1

        return Population(
            population_id=f"pop_{generation}_{int(datetime.now().timestamp())}",
            individuals=new_individuals[: self.config.population_size],
            generation=generation,
            population_size=self.config.population_size,
        )

    async def export_results(
        self, result: OptimizationResult, output_path: Path
    ) -> None:
        """Export optimization results to JSON file.

        Args:
            result: Optimization result to export
            output_path: Output file path
        """
        try:
            # Convert result to serializable format
            export_data = {
                "optimization_id": result.optimization_id,
                "success": result.success,
                "error_message": result.error_message,
                "total_generations": result.total_generations,
                "total_evaluations": result.total_evaluations,
                "optimization_time_seconds": result.optimization_time_seconds,
                "best_individual": {
                    "individual_id": result.best_individual.individual_id,
                    "genes": result.best_individual.genes,
                    "phenotype": result.best_individual.phenotype,
                    "fitness_scores": result.best_individual.fitness_scores,
                    "overall_fitness": result.best_individual.overall_fitness,
                    "generation": result.best_individual.generation,
                },
                "optimization_history": result.optimization_history,
                "convergence_info": result.convergence_info,
                "performance_metrics": result.performance_metrics,
                "config": {
                    "population_size": self.config.population_size,
                    "max_generations": self.config.max_generations,
                    "mutation_rate": self.config.mutation_rate,
                    "crossover_rate": self.config.crossover_rate,
                    "fitness_objectives": self.config.fitness_objectives,
                },
            }

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        except Exception as e:
            raise TrainingError(
                f"Failed to export results: {str(e)}",
                details={"output_path": str(output_path)},
                original_error=e,
            )
