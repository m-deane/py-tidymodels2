"""
Genetic Algorithm implementation for feature selection.

This module provides a flexible genetic algorithm framework for optimizing
binary feature selection problems with support for constraints and convergence detection.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for genetic algorithm parameters."""

    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: float = 0.1
    tournament_size: int = 3
    convergence_threshold: float = 1e-4
    convergence_patience: int = 10
    adaptive_mutation: bool = False
    adaptive_crossover: bool = False
    random_state: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0 <= self.elitism <= 1):
            raise ValueError("elitism must be in [0, 1]")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        if self.convergence_threshold < 0:
            raise ValueError("convergence_threshold must be >= 0")
        if self.convergence_patience < 1:
            raise ValueError("convergence_patience must be >= 1")


class GeneticAlgorithm:
    """
    Genetic Algorithm for binary feature selection optimization.

    This class implements a standard genetic algorithm with:
    - Binary chromosome encoding (0/1 for feature excluded/included)
    - Tournament selection
    - Single-point crossover
    - Bit-flip mutation
    - Elitism
    - Convergence detection

    Parameters
    ----------
    n_features : int
        Number of features (chromosome length)
    fitness_function : Callable[[np.ndarray], float]
        Function that evaluates a chromosome and returns fitness score (higher is better)
    config : GAConfig, optional
        Configuration parameters for the GA

    Attributes
    ----------
    population_ : np.ndarray
        Current population of chromosomes (shape: population_size × n_features)
    fitness_scores_ : np.ndarray
        Fitness scores for current population
    best_chromosome_ : np.ndarray
        Best chromosome found so far
    best_fitness_ : float
        Fitness of best chromosome
    fitness_history_ : List[float]
        Best fitness per generation
    converged_ : bool
        Whether the algorithm converged before max generations
    n_generations_ : int
        Number of generations executed

    Examples
    --------
    >>> def fitness_fn(chromosome):
    ...     # Simple fitness: number of selected features
    ...     return np.sum(chromosome)
    >>>
    >>> ga = GeneticAlgorithm(n_features=10, fitness_function=fitness_fn)
    >>> best_chromosome, best_fitness, history = ga.evolve()
    >>> print(f"Selected features: {np.where(best_chromosome)[0]}")
    """

    def __init__(
        self,
        n_features: int,
        fitness_function: Callable[[np.ndarray], float],
        config: Optional[GAConfig] = None,
        mandatory_indices: Optional[List[int]] = None,
        forbidden_indices: Optional[List[int]] = None,
        feature_costs: Optional[np.ndarray] = None,
        max_cost: Optional[float] = None,
        seed_chromosomes: Optional[np.ndarray] = None,
        generation_callback: Optional[Callable[[int], None]] = None
    ):
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.n_features = n_features
        self.fitness_function = fitness_function
        self.config = config if config is not None else GAConfig()

        # Feature constraints
        self.mandatory_indices = mandatory_indices if mandatory_indices is not None else []
        self.forbidden_indices = forbidden_indices if forbidden_indices is not None else []
        self.feature_costs = feature_costs
        self.max_cost = max_cost

        # Warm start chromosomes
        self.seed_chromosomes = seed_chromosomes
        if self.seed_chromosomes is not None:
            # Validate seed chromosomes
            if len(self.seed_chromosomes.shape) != 2:
                raise ValueError("seed_chromosomes must be 2D array (n_seeds × n_features)")
            if self.seed_chromosomes.shape[1] != n_features:
                raise ValueError(f"seed_chromosomes must have {n_features} features")
            if not np.all(np.isin(self.seed_chromosomes, [0, 1])):
                raise ValueError("seed_chromosomes must be binary (0/1 values)")

        # Validate constraints
        if set(self.mandatory_indices) & set(self.forbidden_indices):
            raise ValueError("Feature cannot be both mandatory and forbidden")

        # Generation callback for relaxation
        self.generation_callback = generation_callback

        # Set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        # State attributes (set during evolution)
        self.population_: Optional[np.ndarray] = None
        self.fitness_scores_: Optional[np.ndarray] = None
        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.fitness_history_: List[float] = []
        self.converged_: bool = False
        self.n_generations_: int = 0

        # Adaptive rate tracking
        self.initial_mutation_rate_: float = self.config.mutation_rate
        self.initial_crossover_rate_: float = self.config.crossover_rate
        self.current_mutation_rate_: float = self.config.mutation_rate
        self.current_crossover_rate_: float = self.config.crossover_rate
        self.mutation_rate_history_: List[float] = []
        self.crossover_rate_history_: List[float] = []

    def initialize_population(self) -> np.ndarray:
        """
        Initialize binary population with optional warm start.

        Each chromosome has 50% probability of each bit being 1.
        If seed_chromosomes are provided, they replace initial individuals.
        Ensures at least one feature is selected per chromosome.
        Respects mandatory and forbidden feature constraints.

        Returns
        -------
        population : np.ndarray
            Binary population matrix (population_size × n_features)
        """
        # Start with random population
        population = np.random.randint(
            0, 2,
            size=(self.config.population_size, self.n_features)
        )

        # Incorporate seed chromosomes if provided (warm start)
        if self.seed_chromosomes is not None:
            n_seeds = min(len(self.seed_chromosomes), self.config.population_size)
            # Replace first n_seeds individuals with seeds
            population[:n_seeds] = self.seed_chromosomes[:n_seeds].copy()

        # Apply mandatory and forbidden constraints to ALL individuals
        for i in range(self.config.population_size):
            # Set mandatory features to 1
            if self.mandatory_indices:
                population[i, self.mandatory_indices] = 1

            # Set forbidden features to 0
            if self.forbidden_indices:
                population[i, self.forbidden_indices] = 0

            # Ensure at least one non-forbidden feature is selected
            selectable_indices = [j for j in range(self.n_features) if j not in self.forbidden_indices]
            if np.sum(population[i, selectable_indices]) == 0 and selectable_indices:
                # Select random selectable feature
                population[i, np.random.choice(selectable_indices)] = 1

        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for all chromosomes in population.

        Parameters
        ----------
        population : np.ndarray
            Population matrix (population_size × n_features)

        Returns
        -------
        fitness_scores : np.ndarray
            Fitness scores (population_size,)
        """
        fitness_scores = np.array([
            self.fitness_function(chromosome)
            for chromosome in population
        ])
        return fitness_scores

    def tournament_selection(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray
    ) -> np.ndarray:
        """
        Select one parent using tournament selection.

        Randomly selects tournament_size individuals and returns the best one.

        Parameters
        ----------
        population : np.ndarray
            Population matrix
        fitness_scores : np.ndarray
            Fitness scores

        Returns
        -------
        parent : np.ndarray
            Selected parent chromosome
        """
        # Select random tournament participants
        tournament_indices = np.random.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False
        )

        # Get fitness of tournament participants
        tournament_fitness = fitness_scores[tournament_indices]

        # Return best individual from tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            First parent chromosome
        parent2 : np.ndarray
            Second parent chromosome

        Returns
        -------
        offspring1, offspring2 : Tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes
        """
        # Apply crossover with probability (use adaptive rate if enabled)
        if np.random.random() < self.current_crossover_rate_:
            # Select random crossover point
            point = np.random.randint(1, self.n_features)

            # Create offspring
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            # No crossover - return copies of parents
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        return offspring1, offspring2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform bit-flip mutation on a chromosome.

        Each bit flips with probability mutation_rate.
        Ensures at least one feature remains selected.
        Respects mandatory and forbidden feature constraints.

        Parameters
        ----------
        chromosome : np.ndarray
            Chromosome to mutate

        Returns
        -------
        mutated : np.ndarray
            Mutated chromosome
        """
        mutated = chromosome.copy()

        # Flip each bit with mutation_rate probability (except constrained features)
        for i in range(self.n_features):
            # Skip mandatory and forbidden features
            if i in self.mandatory_indices or i in self.forbidden_indices:
                continue

            if np.random.random() < self.current_mutation_rate_:
                mutated[i] = 1 - mutated[i]

        # Ensure mandatory features are set
        if self.mandatory_indices:
            mutated[self.mandatory_indices] = 1

        # Ensure forbidden features are unset
        if self.forbidden_indices:
            mutated[self.forbidden_indices] = 0

        # Ensure at least one non-forbidden feature is selected
        selectable_indices = [j for j in range(self.n_features) if j not in self.forbidden_indices]
        if np.sum(mutated[selectable_indices]) == 0 and selectable_indices:
            mutated[np.random.choice(selectable_indices)] = 1

        return mutated

    def apply_elitism(
        self,
        old_population: np.ndarray,
        old_fitness: np.ndarray,
        new_population: np.ndarray,
        new_fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply elitism by preserving top individuals from old generation.

        Parameters
        ----------
        old_population : np.ndarray
            Previous generation population
        old_fitness : np.ndarray
            Previous generation fitness scores
        new_population : np.ndarray
            New generation population
        new_fitness : np.ndarray
            New generation fitness scores

        Returns
        -------
        final_population, final_fitness : Tuple[np.ndarray, np.ndarray]
            Population and fitness after applying elitism
        """
        n_elite = int(self.config.elitism * self.config.population_size)

        if n_elite == 0:
            return new_population, new_fitness

        # Get indices of top individuals from old population
        elite_indices = np.argsort(old_fitness)[-n_elite:]

        # Replace worst individuals in new population with elite
        worst_indices = np.argsort(new_fitness)[:n_elite]

        new_population[worst_indices] = old_population[elite_indices]
        new_fitness[worst_indices] = old_fitness[elite_indices]

        return new_population, new_fitness

    def check_convergence(self) -> bool:
        """
        Check if the algorithm has converged.

        Convergence is detected if the best fitness improvement is below
        convergence_threshold for convergence_patience generations.

        Returns
        -------
        converged : bool
            True if converged
        """
        if len(self.fitness_history_) < self.config.convergence_patience + 1:
            return False

        # Get recent fitness values
        recent = self.fitness_history_[-self.config.convergence_patience - 1:]

        # Check if improvement is below threshold
        improvements = np.diff(recent)
        max_improvement = np.max(improvements) if len(improvements) > 0 else np.inf

        return max_improvement < self.config.convergence_threshold

    def adapt_rates(self) -> None:
        """
        Adapt mutation and crossover rates based on evolution progress.

        Adaptation strategy:
        - Low fitness variance (converging) → increase mutation (explore)
        - High improvement rate → decrease mutation (exploit)
        - Rates bounded between 0.05 and 0.5 for stability

        Updates self.current_mutation_rate_ and self.current_crossover_rate_.
        """
        if len(self.fitness_history_) < 5:
            # Need at least 5 generations for meaningful adaptation
            return

        # Calculate fitness variance across population
        fitness_variance = np.var(self.fitness_scores_) if self.fitness_scores_ is not None else 1.0

        # Calculate recent improvement rate
        recent_history = self.fitness_history_[-5:]
        improvement_rate = (recent_history[-1] - recent_history[0]) / len(recent_history)

        # Normalize metrics for adaptation
        # High variance = diverse population (exploring) → decrease mutation
        # Low variance = converging population → increase mutation
        variance_factor = 1.0 / (1.0 + fitness_variance) if fitness_variance > 0 else 1.0

        # High improvement = making progress → decrease mutation (exploit)
        # Low improvement = stuck → increase mutation (explore)
        improvement_factor = 1.0 / (1.0 + abs(improvement_rate)) if improvement_rate != 0 else 1.0

        # Adapt mutation rate
        if self.config.adaptive_mutation:
            # Increase mutation when converging (low variance) or not improving
            adaptation = variance_factor * improvement_factor
            self.current_mutation_rate_ = self.initial_mutation_rate_ * (1.0 + 2.0 * adaptation)

            # Bound mutation rate
            self.current_mutation_rate_ = np.clip(self.current_mutation_rate_, 0.05, 0.5)

        # Adapt crossover rate (inverse of mutation - more crossover when exploiting)
        if self.config.adaptive_crossover:
            # Increase crossover when making progress (low adaptation)
            adaptation = variance_factor * improvement_factor
            self.current_crossover_rate_ = self.initial_crossover_rate_ * (1.0 + adaptation)

            # Bound crossover rate
            self.current_crossover_rate_ = np.clip(self.current_crossover_rate_, 0.5, 0.95)

        # Track rate history
        self.mutation_rate_history_.append(self.current_mutation_rate_)
        self.crossover_rate_history_.append(self.current_crossover_rate_)

    def evolve(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the genetic algorithm evolution process.

        Returns
        -------
        best_chromosome : np.ndarray
            Best chromosome found
        best_fitness : float
            Fitness of best chromosome
        fitness_history : List[float]
            Best fitness per generation
        """
        # Initialize population
        self.population_ = self.initialize_population()
        self.fitness_scores_ = self.evaluate_population(self.population_)

        # Track best individual
        best_idx = np.argmax(self.fitness_scores_)
        self.best_chromosome_ = self.population_[best_idx].copy()
        self.best_fitness_ = self.fitness_scores_[best_idx]
        self.fitness_history_ = [self.best_fitness_]

        if self.config.verbose:
            print(f"Generation 0: Best fitness = {self.best_fitness_:.6f}")

        # Evolution loop
        for generation in range(1, self.config.generations + 1):
            self.n_generations_ = generation

            # Create new population through selection, crossover, and mutation
            new_population = []

            while len(new_population) < self.config.population_size:
                # Select parents
                parent1 = self.tournament_selection(self.population_, self.fitness_scores_)
                parent2 = self.tournament_selection(self.population_, self.fitness_scores_)

                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)

                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                new_population.append(offspring1)
                if len(new_population) < self.config.population_size:
                    new_population.append(offspring2)

            new_population = np.array(new_population[:self.config.population_size])
            new_fitness = self.evaluate_population(new_population)

            # Apply elitism
            self.population_, self.fitness_scores_ = self.apply_elitism(
                self.population_, self.fitness_scores_,
                new_population, new_fitness
            )

            # Update best individual
            best_idx = np.argmax(self.fitness_scores_)
            if self.fitness_scores_[best_idx] > self.best_fitness_:
                self.best_chromosome_ = self.population_[best_idx].copy()
                self.best_fitness_ = self.fitness_scores_[best_idx]

            self.fitness_history_.append(self.best_fitness_)

            # Call generation callback if provided (for constraint relaxation)
            if self.generation_callback is not None:
                self.generation_callback(generation)

            # Adapt mutation and crossover rates if enabled
            if self.config.adaptive_mutation or self.config.adaptive_crossover:
                self.adapt_rates()

            if self.config.verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness_:.6f}")

            # Check convergence
            if self.check_convergence():
                self.converged_ = True
                if self.config.verbose:
                    print(f"Converged at generation {generation}")
                break

        if self.config.verbose and not self.converged_:
            print(f"Reached max generations ({self.config.generations})")

        return self.best_chromosome_, self.best_fitness_, self.fitness_history_
