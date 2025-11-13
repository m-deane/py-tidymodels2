"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.

This module implements the NSGA-II algorithm for optimizing multiple objectives
simultaneously, maintaining a diverse set of Pareto-optimal solutions.

References:
    Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II."
    IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple, Any, Dict
from dataclasses import dataclass, field


@dataclass
class NSGA2Config:
    """Configuration for NSGA-II parameters."""

    population_size: int = 100  # Should be even for NSGA-II
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    tournament_size: int = 2  # Binary tournament is standard for NSGA-II
    random_state: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4")
        if self.population_size % 2 != 0:
            raise ValueError("population_size must be even for NSGA-II")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate must be in [0, 1]")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")


class NSGAII:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective optimization.

    This class implements the NSGA-II algorithm for optimizing multiple objectives
    simultaneously. It maintains a diverse set of Pareto-optimal solutions.

    Parameters
    ----------
    n_features : int
        Number of features (chromosome length)
    objective_functions : List[Callable[[np.ndarray], float]]
        List of objective functions. Each function evaluates a chromosome and returns
        a score. Convention: minimization (lower is better) for all objectives.
    config : NSGA2Config, optional
        Configuration parameters for NSGA-II

    Attributes
    ----------
    population_ : np.ndarray
        Current population of chromosomes (shape: population_size × n_features)
    objective_values_ : np.ndarray
        Objective values for current population (shape: population_size × n_objectives)
    pareto_front_ : List[int]
        Indices of solutions in the first Pareto front (rank 0)
    ranks_ : np.ndarray
        Pareto rank for each solution (0 = best)
    crowding_distances_ : np.ndarray
        Crowding distance for each solution (higher = more diverse)
    n_generations_ : int
        Number of generations executed

    Examples
    --------
    >>> def obj1(chromosome):
    ...     # Minimize number of features
    ...     return np.sum(chromosome)
    >>>
    >>> def obj2(chromosome):
    ...     # Minimize error (simulated)
    ...     return 1.0 / (np.sum(chromosome) + 1)
    >>>
    >>> nsga2 = NSGAII(n_features=10, objective_functions=[obj1, obj2])
    >>> pareto_front, obj_values = nsga2.evolve()
    """

    def __init__(
        self,
        n_features: int,
        objective_functions: List[Callable[[np.ndarray], float]],
        config: Optional[NSGA2Config] = None,
    ):
        self.n_features = n_features
        self.objective_functions = objective_functions
        self.n_objectives = len(objective_functions)
        self.config = config if config is not None else NSGA2Config()

        if self.n_objectives < 2:
            raise ValueError("NSGA-II requires at least 2 objective functions")

        # Set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        # Initialize state
        self.population_ = None
        self.objective_values_ = None
        self.ranks_ = None
        self.crowding_distances_ = None
        self.pareto_front_ = []
        self.n_generations_ = 0

    def initialize_population(self) -> np.ndarray:
        """
        Initialize random population with at least one feature selected per chromosome.

        Returns
        -------
        np.ndarray
            Initial population (population_size × n_features)
        """
        population = np.random.randint(0, 2, size=(self.config.population_size, self.n_features))

        # Ensure at least one feature is selected in each chromosome
        for i in range(len(population)):
            if np.sum(population[i]) == 0:
                # Randomly set one feature to 1
                random_idx = np.random.randint(0, self.n_features)
                population[i, random_idx] = 1

        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate all objectives for each chromosome in the population.

        Parameters
        ----------
        population : np.ndarray
            Population to evaluate (population_size × n_features)

        Returns
        -------
        np.ndarray
            Objective values (population_size × n_objectives)
        """
        objective_values = np.zeros((len(population), self.n_objectives))

        for i, chromosome in enumerate(population):
            for j, obj_fn in enumerate(self.objective_functions):
                objective_values[i, j] = obj_fn(chromosome)

        return objective_values

    def dominates(self, obj_values_a: np.ndarray, obj_values_b: np.ndarray) -> bool:
        """
        Check if solution A dominates solution B (Pareto dominance).

        A dominates B if:
        - A is no worse than B in all objectives
        - A is strictly better than B in at least one objective

        Parameters
        ----------
        obj_values_a : np.ndarray
            Objective values for solution A
        obj_values_b : np.ndarray
            Objective values for solution B

        Returns
        -------
        bool
            True if A dominates B
        """
        # All objectives must be no worse (<=) and at least one strictly better (<)
        no_worse = np.all(obj_values_a <= obj_values_b)
        strictly_better = np.any(obj_values_a < obj_values_b)
        return no_worse and strictly_better

    def fast_non_dominated_sort(self, objective_values: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        Fast non-dominated sorting algorithm from NSGA-II paper.

        This assigns a Pareto rank to each solution:
        - Rank 0: non-dominated solutions (Pareto front)
        - Rank 1: solutions dominated only by rank 0
        - Rank 2: solutions dominated only by rank 0 or 1
        - ...

        Parameters
        ----------
        objective_values : np.ndarray
            Objective values for all solutions (n_solutions × n_objectives)

        Returns
        -------
        fronts : List[List[int]]
            List of fronts, where each front is a list of solution indices
        ranks : np.ndarray
            Pareto rank for each solution
        """
        n_solutions = len(objective_values)
        domination_counts = np.zeros(n_solutions, dtype=int)  # Number of solutions that dominate this one
        dominated_solutions = [[] for _ in range(n_solutions)]  # Solutions dominated by this one
        ranks = np.zeros(n_solutions, dtype=int)
        fronts = [[]]

        # Build domination relationships
        for i in range(n_solutions):
            for j in range(i + 1, n_solutions):
                if self.dominates(objective_values[i], objective_values[j]):
                    # i dominates j
                    dominated_solutions[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(objective_values[j], objective_values[i]):
                    # j dominates i
                    dominated_solutions[j].append(i)
                    domination_counts[i] += 1

        # Find first front (non-dominated solutions)
        for i in range(n_solutions):
            if domination_counts[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        # Find remaining fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        ranks[j] = current_front + 1
                        next_front.append(j)
            current_front += 1
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break

        return fronts, ranks

    def calculate_crowding_distance(
        self, objective_values: np.ndarray, front_indices: List[int]
    ) -> np.ndarray:
        """
        Calculate crowding distance for solutions in a Pareto front.

        Crowding distance measures how close a solution is to its neighbors.
        Higher values indicate more isolated solutions (better for diversity).

        Parameters
        ----------
        objective_values : np.ndarray
            Objective values for all solutions
        front_indices : List[int]
            Indices of solutions in this front

        Returns
        -------
        np.ndarray
            Crowding distances for all solutions (infinity for boundary points)
        """
        n_solutions = len(objective_values)
        n_front = len(front_indices)
        distances = np.zeros(n_solutions)

        if n_front <= 2:
            # Boundary solutions get infinite distance
            for idx in front_indices:
                distances[idx] = np.inf
            return distances

        # Calculate crowding distance for each objective
        for obj_idx in range(self.n_objectives):
            # Sort front by this objective
            sorted_indices = sorted(front_indices, key=lambda i: objective_values[i, obj_idx])

            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Normalize by objective range
            obj_min = objective_values[sorted_indices[0], obj_idx]
            obj_max = objective_values[sorted_indices[-1], obj_idx]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue  # Skip if all values are the same

            # Calculate distance for interior points
            for i in range(1, n_front - 1):
                idx = sorted_indices[i]
                idx_prev = sorted_indices[i - 1]
                idx_next = sorted_indices[i + 1]

                distance = (
                    objective_values[idx_next, obj_idx] - objective_values[idx_prev, obj_idx]
                ) / obj_range
                distances[idx] += distance

        return distances

    def binary_tournament_selection(
        self, population: np.ndarray, ranks: np.ndarray, crowding_distances: np.ndarray
    ) -> np.ndarray:
        """
        Binary tournament selection using rank and crowding distance.

        Selects better solution based on:
        1. Lower rank (better Pareto front)
        2. If same rank, higher crowding distance (more diverse)

        Parameters
        ----------
        population : np.ndarray
            Current population
        ranks : np.ndarray
            Pareto ranks
        crowding_distances : np.ndarray
            Crowding distances

        Returns
        -------
        np.ndarray
            Selected chromosome
        """
        # Select two random indices
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)

        # Compare by rank first
        if ranks[idx1] < ranks[idx2]:
            return population[idx1].copy()
        elif ranks[idx2] < ranks[idx1]:
            return population[idx2].copy()
        else:
            # Same rank: compare by crowding distance (higher is better)
            if crowding_distances[idx1] > crowding_distances[idx2]:
                return population[idx1].copy()
            else:
                return population[idx2].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover.

        Parameters
        ----------
        parent1 : np.ndarray
            First parent chromosome
        parent2 : np.ndarray
            Second parent chromosome

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes
        """
        if np.random.random() < self.config.crossover_rate:
            point = np.random.randint(1, self.n_features)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Bit-flip mutation ensuring at least one feature is selected.

        Parameters
        ----------
        chromosome : np.ndarray
            Chromosome to mutate

        Returns
        -------
        np.ndarray
            Mutated chromosome
        """
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.config.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit

        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            # Randomly set one feature to 1
            random_idx = np.random.randint(0, len(mutated))
            mutated[random_idx] = 1

        return mutated

    def create_offspring_population(
        self, population: np.ndarray, ranks: np.ndarray, crowding_distances: np.ndarray
    ) -> np.ndarray:
        """
        Create offspring population using selection, crossover, and mutation.

        Parameters
        ----------
        population : np.ndarray
            Parent population
        ranks : np.ndarray
            Pareto ranks
        crowding_distances : np.ndarray
            Crowding distances

        Returns
        -------
        np.ndarray
            Offspring population (same size as parent population)
        """
        offspring = []
        n_offspring_needed = len(population)

        while len(offspring) < n_offspring_needed:
            # Select two parents
            parent1 = self.binary_tournament_selection(population, ranks, crowding_distances)
            parent2 = self.binary_tournament_selection(population, ranks, crowding_distances)

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            offspring.append(child1)
            if len(offspring) < n_offspring_needed:
                offspring.append(child2)

        return np.array(offspring[: n_offspring_needed])

    def environmental_selection(
        self,
        combined_population: np.ndarray,
        combined_objectives: np.ndarray,
        target_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select next generation using non-dominated sorting and crowding distance.

        Parameters
        ----------
        combined_population : np.ndarray
            Combined parent + offspring population
        combined_objectives : np.ndarray
            Objective values for combined population
        target_size : int
            Desired population size

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Selected population and their objective values
        """
        # Perform non-dominated sorting
        fronts, ranks = self.fast_non_dominated_sort(combined_objectives)

        # Calculate crowding distances
        crowding_distances = np.zeros(len(combined_population))
        for front in fronts:
            distances = self.calculate_crowding_distance(combined_objectives, front)
            crowding_distances += distances

        # Select individuals for next generation
        next_population = []
        next_objectives = []
        front_idx = 0

        while len(next_population) < target_size and front_idx < len(fronts):
            front = fronts[front_idx]

            if len(next_population) + len(front) <= target_size:
                # Include entire front
                for idx in front:
                    next_population.append(combined_population[idx])
                    next_objectives.append(combined_objectives[idx])
            else:
                # Include only part of front (select by crowding distance)
                remaining = target_size - len(next_population)
                front_crowding = [(idx, crowding_distances[idx]) for idx in front]
                front_crowding.sort(key=lambda x: x[1], reverse=True)  # Higher distance first

                for idx, _ in front_crowding[:remaining]:
                    next_population.append(combined_population[idx])
                    next_objectives.append(combined_objectives[idx])

            front_idx += 1

        return np.array(next_population), np.array(next_objectives)

    def evolve(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Run the NSGA-II algorithm.

        Returns
        -------
        pareto_population : np.ndarray
            Solutions in the final Pareto front (first front)
        pareto_objectives : np.ndarray
            Objective values for Pareto front solutions
        pareto_indices : List[int]
            Indices of Pareto front solutions in final population
        """
        # Initialize population
        self.population_ = self.initialize_population()
        self.objective_values_ = self.evaluate_population(self.population_)

        # Main evolution loop
        for generation in range(self.config.generations):
            self.n_generations_ = generation + 1

            # Non-dominated sorting and crowding distance
            fronts, self.ranks_ = self.fast_non_dominated_sort(self.objective_values_)
            self.crowding_distances_ = np.zeros(len(self.population_))
            for front in fronts:
                distances = self.calculate_crowding_distance(self.objective_values_, front)
                self.crowding_distances_ += distances

            # Create offspring population
            offspring = self.create_offspring_population(
                self.population_, self.ranks_, self.crowding_distances_
            )
            offspring_objectives = self.evaluate_population(offspring)

            # Combine parent and offspring
            combined_population = np.vstack([self.population_, offspring])
            combined_objectives = np.vstack([self.objective_values_, offspring_objectives])

            # Environmental selection
            self.population_, self.objective_values_ = self.environmental_selection(
                combined_population, combined_objectives, self.config.population_size
            )

            if self.config.verbose and generation % 10 == 0:
                print(f"Generation {generation}: Pareto front size = {len(fronts[0])}")

        # Final non-dominated sorting to get Pareto front
        fronts, self.ranks_ = self.fast_non_dominated_sort(self.objective_values_)
        self.pareto_front_ = fronts[0]

        # Extract Pareto front solutions
        pareto_population = self.population_[self.pareto_front_]
        pareto_objectives = self.objective_values_[self.pareto_front_]

        if self.config.verbose:
            print(f"Final Pareto front size: {len(self.pareto_front_)}")

        return pareto_population, pareto_objectives, self.pareto_front_
