"""
Genetic algorithm-based feature selection step.

This step uses genetic algorithms to optimize feature selection based on
model performance and optional statistical constraints.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Union, List, Callable, Dict, Any
import pandas as pd
import numpy as np

from py_recipes.utils import (
    GeneticAlgorithm,
    GAConfig,
    create_model_fitness_evaluator,
    create_constrained_fitness_function
)
from py_recipes.utils.nsga2 import NSGAII, NSGA2Config


@dataclass
class StepSelectGeneticAlgorithm:
    """
    Select features using genetic algorithm optimization.

    This step uses a genetic algorithm to find an optimal subset of features
    that maximizes model performance while optionally satisfying statistical
    constraints (p-value significance, coefficient stability, multicollinearity, etc.).

    Parameters
    ----------
    columns : selector, optional
        Which columns to consider for selection. If None, uses all numeric predictors
    outcome : str
        Name of the outcome variable
    model : ModelSpec
        Parsnip model specification to use for fitness evaluation
    metric : str, default='rmse'
        Metric to optimize ('rmse', 'mae', 'r_squared', 'accuracy', etc.)
    maximize : bool, default=False
        Whether to maximize metric (True for R², accuracy) or minimize (False for RMSE, MAE)
    top_n : int, optional
        Number of features to select. If None, GA determines optimal number
    constraints : dict, default={}
        Statistical constraints to apply. Keys can be:
        - 'p_value': {'max': 0.05, 'method': 'bonferroni'/'fdr_bh'/'none'}
        - 'coef_stability': {'min': 0.7, 'method': 'correlation'/'cv', 'cv_folds': 5}
        - 'vif': {'max': 5.0, 'exclude_if_exceeded': True/False}
        - 'effect_size': {'min': 0.1, 'method': 'cohens_f2'}
        - 'outcome_correlation': {'min': 0.1, 'method': 'pearson'/'spearman'/'kendall'}
    population_size : int, default=50
        GA population size
    generations : int, default=100
        Maximum number of GA generations
    mutation_rate : float, default=0.1
        Probability of bit-flip mutation
    crossover_rate : float, default=0.8
        Probability of crossover between parents
    elitism : float, default=0.1
        Proportion of top individuals to preserve
    tournament_size : int, default=3
        Tournament selection size
    cv_folds : int, default=5
        Number of cross-validation folds for fitness evaluation
    convergence_threshold : float, default=1e-4
        Minimum fitness improvement to continue evolution
    convergence_patience : int, default=10
        Generations to wait for improvement before stopping
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Print GA progress
    skip : bool, default=False
        Skip this step
    id : str, optional
        Unique identifier for this step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_select_genetic_algorithm
    >>> from py_parsnip import linear_reg
    >>>
    >>> # Basic usage: Select top 10 features optimizing RMSE
    >>> rec = recipe(data, "y ~ .").step_select_genetic_algorithm(
    ...     outcome='y',
    ...     model=linear_reg(),
    ...     metric='rmse',
    ...     top_n=10
    ... )
    >>>
    >>> # With statistical constraints
    >>> rec = recipe(data, "y ~ .").step_select_genetic_algorithm(
    ...     outcome='y',
    ...     model=linear_reg(),
    ...     metric='rmse',
    ...     top_n=15,
    ...     constraints={
    ...         'p_value': {'max': 0.05, 'method': 'bonferroni'},
    ...         'vif': {'max': 5.0, 'exclude_if_exceeded': True},
    ...         'coef_stability': {'min': 0.7, 'method': 'correlation'}
    ...     },
    ...     cv_folds=5,
    ...     verbose=True
    ... )
    """
    outcome: str
    model: Any  # ModelSpec type
    columns: Union[None, str, List[str], Callable] = None
    metric: str = "rmse"
    maximize: bool = False
    top_n: Optional[int] = None
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mandatory_features: List[str] = field(default_factory=list)
    forbidden_features: List[str] = field(default_factory=list)
    feature_costs: Dict[str, float] = field(default_factory=dict)
    max_total_cost: Optional[float] = None
    cost_weight: float = 0.0
    sparsity_weight: float = 0.0
    warm_start: Union[None, str, np.ndarray] = None  # None, "importance", "low_correlation", or custom seeds
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: float = 0.1
    tournament_size: int = 3
    cv_folds: int = 5
    convergence_threshold: float = 1e-4
    convergence_patience: int = 10
    adaptive_mutation: bool = False
    adaptive_crossover: bool = False
    relax_constraints_after: Optional[int] = None
    relaxation_rate: float = 0.05
    n_jobs: int = 1
    n_ensemble: int = 1
    ensemble_strategy: str = "voting"
    ensemble_threshold: float = 0.5
    use_nsga2: bool = False
    nsga2_objectives: List[str] = field(default_factory=lambda: ["performance", "sparsity"])
    nsga2_selection_method: str = "knee_point"  # "knee_point", "min_features", "best_performance", "index"
    nsga2_selection_index: int = 0  # Used when method="index"
    random_state: Optional[int] = None
    verbose: bool = False
    skip: bool = False
    id: Optional[str] = None

    # Prepared state
    _selected_features: List[str] = field(default_factory=list, init=False, repr=False)
    _feature_names: List[str] = field(default_factory=list, init=False, repr=False)
    _ga_history: List[float] = field(default_factory=list, init=False, repr=False)
    _final_fitness: float = field(default=0.0, init=False, repr=False)
    _best_chromosome: np.ndarray = field(default=None, init=False, repr=False)
    _converged: bool = field(default=False, init=False, repr=False)
    _n_generations: int = field(default=0, init=False, repr=False)
    _is_prepared: bool = field(default=False, init=False, repr=False)
    _ensemble_results: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _feature_frequencies: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pareto_front: np.ndarray = field(default=None, init=False, repr=False)
    _pareto_objectives: np.ndarray = field(default=None, init=False, repr=False)
    _pareto_indices: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.model is None:
            raise ValueError("model parameter is required")

        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("mutation_rate must be in [0, 1]")

        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("crossover_rate must be in [0, 1]")

        if not (0 <= self.elitism <= 1):
            raise ValueError("elitism must be in [0, 1]")

        if self.top_n is not None and self.top_n < 1:
            raise ValueError("top_n must be >= 1")

        if self.n_ensemble < 1:
            raise ValueError("n_ensemble must be >= 1")

        valid_strategies = ["voting", "frequency", "union", "intersection"]
        if self.ensemble_strategy not in valid_strategies:
            raise ValueError(f"ensemble_strategy must be one of {valid_strategies}")

        if not (0 < self.ensemble_threshold <= 1.0):
            raise ValueError("ensemble_threshold must be in (0, 1]")

        if self.use_nsga2:
            if len(self.nsga2_objectives) < 2:
                raise ValueError("NSGA-II requires at least 2 objectives")

            valid_objectives = ["performance", "sparsity", "cost"]
            for obj in self.nsga2_objectives:
                if obj not in valid_objectives:
                    raise ValueError(f"nsga2_objectives must be from {valid_objectives}, got '{obj}'")

            if "cost" in self.nsga2_objectives and not self.feature_costs:
                raise ValueError("nsga2_objectives includes 'cost' but feature_costs is not provided")

            valid_methods = ["knee_point", "min_features", "best_performance", "index"]
            if self.nsga2_selection_method not in valid_methods:
                raise ValueError(f"nsga2_selection_method must be one of {valid_methods}")

            if self.nsga2_selection_method == "index" and self.nsga2_selection_index < 0:
                raise ValueError("nsga2_selection_index must be >= 0")

            if self.n_ensemble > 1:
                raise ValueError("Ensemble mode (n_ensemble > 1) is not compatible with NSGA-II (use_nsga2=True)")

    def prep(self, data: pd.DataFrame, training: bool = True):
        """
        Prepare the step by running genetic algorithm to select features.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        training : bool, default=True
            Whether this is training mode

        Returns
        -------
        prepared_step : StepSelectGeneticAlgorithm
            Prepared step with selected features
        """
        if self.skip or not training:
            return self

        # Validate outcome exists
        if self.outcome not in data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found in data")

        # Resolve columns to consider
        if self.columns is None:
            # Use all columns except outcome
            candidate_cols = [c for c in data.columns if c != self.outcome]
        elif isinstance(self.columns, str):
            candidate_cols = [self.columns]
        elif callable(self.columns):
            candidate_cols = self.columns(data)
        else:
            candidate_cols = list(self.columns)

        # Remove outcome if accidentally included
        candidate_cols = [c for c in candidate_cols if c != self.outcome]

        # Filter to numeric columns only
        numeric_cols = data[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns to select from after resolving selector")

        if self.verbose:
            print(f"Genetic Algorithm Feature Selection")
            print(f"  Candidate features: {len(numeric_cols)}")
            print(f"  Model: {self.model.model_type}")
            print(f"  Metric: {self.metric} ({'maximize' if self.maximize else 'minimize'})")
            print(f"  Constraints: {len(self.constraints)}")
            print(f"  Population size: {self.population_size}")
            print(f"  Max generations: {self.generations}")
            print()

        # Create fitness function
        base_fitness_fn = create_model_fitness_evaluator(
            data=data,
            outcome_col=self.outcome,
            model_spec=self.model,
            metric=self.metric,
            maximize=self.maximize,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )

        # Add constraints if specified
        if len(self.constraints) > 0:
            # Create relaxation factor getter if relaxation is enabled
            current_generation = [0]  # Mutable to allow updates from nested function

            def get_relaxation_factor():
                """Compute relaxation factor based on current generation."""
                if self.relax_constraints_after is None:
                    return 1.0  # No relaxation

                gen = current_generation[0]
                if gen < self.relax_constraints_after:
                    return 1.0  # Full constraint strength

                # Linear relaxation after threshold
                generations_since_relax = gen - self.relax_constraints_after
                relaxation_factor = max(0.0, 1.0 - (generations_since_relax * self.relaxation_rate))
                return relaxation_factor

            # Store current_generation reference so GA can update it
            self._current_generation = current_generation

            fitness_fn = create_constrained_fitness_function(
                data=data,
                outcome_col=self.outcome,
                feature_names=numeric_cols,
                model_spec=self.model,
                base_fitness_fn=base_fitness_fn,
                constraints=self.constraints,
                relaxation_factor_getter=get_relaxation_factor
            )
        else:
            fitness_fn = base_fitness_fn
            self._current_generation = None  # No constraints, no relaxation

        # Map mandatory and forbidden features to indices
        mandatory_indices = [numeric_cols.index(f) for f in self.mandatory_features if f in numeric_cols]
        forbidden_indices = [numeric_cols.index(f) for f in self.forbidden_features if f in numeric_cols]

        # Create cost array if specified
        cost_array = None
        if self.feature_costs:
            cost_array = np.array([self.feature_costs.get(f, 0.0) for f in numeric_cols])

        # Wrap fitness function with cost and sparsity penalties if needed
        if self.cost_weight > 0 or self.sparsity_weight > 0:
            base_fitness = fitness_fn

            def enhanced_fitness(chromosome):
                fitness = base_fitness(chromosome)

                # Add cost penalty
                if self.cost_weight > 0 and cost_array is not None:
                    total_cost = np.sum(chromosome * cost_array)
                    if self.max_total_cost and total_cost > self.max_total_cost:
                        # Heavy penalty for exceeding budget
                        fitness -= (total_cost - self.max_total_cost) * self.cost_weight * 10
                    else:
                        # Soft penalty proportional to cost
                        fitness -= total_cost * self.cost_weight

                # Add sparsity penalty (prefer fewer features)
                if self.sparsity_weight > 0:
                    n_selected = np.sum(chromosome)
                    sparsity_penalty = (n_selected / len(chromosome)) * self.sparsity_weight
                    fitness -= sparsity_penalty

                return fitness

            fitness_fn = enhanced_fitness

        # Configure GA
        ga_config = GAConfig(
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            tournament_size=self.tournament_size,
            convergence_threshold=self.convergence_threshold,
            convergence_patience=self.convergence_patience,
            adaptive_mutation=self.adaptive_mutation,
            adaptive_crossover=self.adaptive_crossover,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )

        # Generate seed chromosomes if warm start is enabled
        seed_chromosomes = None
        if self.warm_start is not None:
            from py_recipes.utils import create_importance_based_seeds, create_low_correlation_seeds

            if isinstance(self.warm_start, str):
                if self.warm_start == "importance":
                    seed_chromosomes = create_importance_based_seeds(
                        data=data,
                        outcome_col=self.outcome,
                        feature_names=numeric_cols,
                        n_seeds=min(10, self.population_size // 2),
                        top_n=self.top_n
                    )
                elif self.warm_start == "low_correlation":
                    seed_chromosomes = create_low_correlation_seeds(
                        data=data,
                        feature_names=numeric_cols,
                        n_seeds=min(10, self.population_size // 2)
                    )
                else:
                    raise ValueError(f"Invalid warm_start method: '{self.warm_start}'. "
                                   "Use 'importance', 'low_correlation', or provide custom seeds.")
            elif isinstance(self.warm_start, np.ndarray):
                seed_chromosomes = self.warm_start
            else:
                raise ValueError("warm_start must be None, 'importance', 'low_correlation', or np.ndarray")

        # Create generation callback for constraint relaxation
        def update_generation(gen):
            """Update current generation for relaxation."""
            if self._current_generation is not None:
                self._current_generation[0] = gen

        # Check if NSGA-II mode
        if self.use_nsga2:
            # NSGA-II multi-objective optimization
            if self.verbose:
                print(f"\nRunning NSGA-II with objectives: {self.nsga2_objectives}")

            # Create objective functions (using factory functions to avoid closure issues)
            objective_functions = []

            def create_performance_objective(fitness_func, should_maximize):
                """Factory for performance objective."""
                if should_maximize:
                    return lambda chromosome: -fitness_func(chromosome)
                else:
                    return fitness_func

            def create_sparsity_objective():
                """Factory for sparsity objective."""
                def sparsity_obj(chromosome):
                    n_features = np.sum(chromosome)
                    if n_features == 0:
                        return 1000.0  # Large penalty for no features
                    return float(n_features)
                return sparsity_obj

            def create_cost_objective(costs):
                """Factory for cost objective."""
                def cost_obj(chromosome):
                    if costs is not None:
                        total = np.sum(chromosome * costs)
                        if np.sum(chromosome) == 0:
                            return 1000.0  # Penalty for no features
                        return float(total)
                    return 0.0
                return cost_obj

            for obj_name in self.nsga2_objectives:
                if obj_name == "performance":
                    objective_functions.append(create_performance_objective(fitness_fn, self.maximize))
                elif obj_name == "sparsity":
                    objective_functions.append(create_sparsity_objective())
                elif obj_name == "cost":
                    objective_functions.append(create_cost_objective(cost_array))

            # Configure NSGA-II
            nsga2_config = NSGA2Config(
                population_size=self.population_size if self.population_size % 2 == 0 else self.population_size + 1,
                generations=self.generations,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                tournament_size=self.tournament_size,
                random_state=self.random_state,
                verbose=self.verbose
            )

            # Run NSGA-II
            nsga2 = NSGAII(
                n_features=len(numeric_cols),
                objective_functions=objective_functions,
                config=nsga2_config
            )

            pareto_population, pareto_objectives, pareto_indices = nsga2.evolve()

            # Select solution from Pareto front
            if self.nsga2_selection_method == "knee_point":
                # Find knee point (best trade-off)
                selected_idx = self._find_knee_point(pareto_objectives)
            elif self.nsga2_selection_method == "min_features":
                # Select solution with fewest features
                feature_counts = np.sum(pareto_population, axis=1)
                selected_idx = np.argmin(feature_counts)
            elif self.nsga2_selection_method == "best_performance":
                # Select solution with best performance (first objective)
                selected_idx = np.argmin(pareto_objectives[:, 0])
            elif self.nsga2_selection_method == "index":
                # Select by index
                selected_idx = min(self.nsga2_selection_index, len(pareto_population) - 1)
            else:
                selected_idx = 0

            best_chromosome = pareto_population[selected_idx]
            selected_indices = np.where(best_chromosome == 1)[0]
            selected_features = [numeric_cols[i] for i in selected_indices]

            # Store Pareto front results
            best_fitness = -pareto_objectives[selected_idx, 0] if self.maximize else pareto_objectives[selected_idx, 0]
            history = []  # NSGA-II doesn't have single fitness history

            # Store for prepared instance
            ensemble_results = []
            feature_frequency = {}
            pareto_front_storage = pareto_population
            pareto_objectives_storage = pareto_objectives
            pareto_indices_storage = pareto_indices

            if self.verbose:
                print(f"\nNSGA-II Complete:")
                print(f"  Pareto front size: {len(pareto_population)}")
                print(f"  Selected solution index: {selected_idx}")
                print(f"  Selected features: {len(selected_features)}/{len(numeric_cols)}")
                print(f"  Objective values: {pareto_objectives[selected_idx]}")
                print(f"  Features: {selected_features}")

        # Standard GA mode (single run)
        elif self.n_ensemble == 1:
            # Single GA run
            ensemble_results = []
            feature_frequency = {}

            ga = GeneticAlgorithm(
                n_features=len(numeric_cols),
                fitness_function=fitness_fn,
                config=ga_config,
                mandatory_indices=mandatory_indices,
                forbidden_indices=forbidden_indices,
                feature_costs=cost_array,
                max_cost=self.max_total_cost,
                seed_chromosomes=seed_chromosomes,
                generation_callback=update_generation
            )

            best_chromosome, best_fitness, history = ga.evolve()

            # Extract selected features
            selected_indices = np.where(best_chromosome == 1)[0]
            selected_features = [numeric_cols[i] for i in selected_indices]

            # Initialize for consistency with other modes
            pareto_front_storage = None
            pareto_objectives_storage = None
            pareto_indices_storage = []
        else:
            # Ensemble mode: run GA multiple times with different seeds (n_ensemble > 1)
            if self.verbose:
                print(f"\nRunning ensemble with {self.n_ensemble} GA instances...")

            all_selected_features = []
            feature_frequency = {feat: 0 for feat in numeric_cols}

            base_seed = self.random_state if self.random_state is not None else 42

            for run_idx in range(self.n_ensemble):
                # Use different random seed for each run
                run_seed = base_seed + run_idx
                run_config = replace(ga_config, random_state=run_seed)

                if self.verbose:
                    print(f"  [{run_idx + 1}/{self.n_ensemble}] Running GA with seed={run_seed}...")

                ga = GeneticAlgorithm(
                    n_features=len(numeric_cols),
                    fitness_function=fitness_fn,
                    config=run_config,
                    mandatory_indices=mandatory_indices,
                    forbidden_indices=forbidden_indices,
                    feature_costs=cost_array,
                    max_cost=self.max_total_cost,
                    seed_chromosomes=seed_chromosomes,
                    generation_callback=update_generation
                )

                run_chromosome, run_fitness, run_history = ga.evolve()
                run_indices = np.where(run_chromosome == 1)[0]
                run_features = [numeric_cols[i] for i in run_indices]

                # Store run results
                ensemble_results.append({
                    'run_idx': run_idx,
                    'seed': run_seed,
                    'chromosome': run_chromosome,
                    'fitness': run_fitness,
                    'history': run_history,
                    'features': run_features,
                    'converged': ga.converged_,
                    'n_generations': ga.n_generations_
                })

                all_selected_features.append(set(run_features))

                # Update feature frequencies
                for feat in run_features:
                    feature_frequency[feat] += 1

                if self.verbose:
                    print(f"      Fitness: {run_fitness:.6f}, Features: {len(run_features)}")

            # Aggregate results based on strategy
            if self.ensemble_strategy == "voting":
                # Select features that appear in majority of runs
                threshold_count = int(np.ceil(self.n_ensemble * self.ensemble_threshold))
                selected_features = [feat for feat, count in feature_frequency.items()
                                   if count >= threshold_count]

            elif self.ensemble_strategy == "frequency":
                # Select features based on frequency threshold
                threshold_count = int(np.ceil(self.n_ensemble * self.ensemble_threshold))
                selected_features = [feat for feat, count in feature_frequency.items()
                                   if count >= threshold_count]

            elif self.ensemble_strategy == "union":
                # Select all features that appear in at least one run
                selected_features = [feat for feat, count in feature_frequency.items()
                                   if count > 0]

            elif self.ensemble_strategy == "intersection":
                # Select only features that appear in ALL runs
                selected_features = [feat for feat, count in feature_frequency.items()
                                   if count == self.n_ensemble]

            # Use best run's fitness and chromosome for reporting
            best_run = max(ensemble_results, key=lambda x: x['fitness'])
            best_chromosome = best_run['chromosome']
            best_fitness = best_run['fitness']
            history = best_run['history']

            # Initialize for consistency with other modes
            pareto_front_storage = None
            pareto_objectives_storage = None
            pareto_indices_storage = []

            if self.verbose:
                print(f"\nEnsemble aggregation ({self.ensemble_strategy}):")
                print(f"  Threshold: {self.ensemble_threshold}")
                print(f"  Selected features: {len(selected_features)}")
                print(f"  Feature frequencies: {sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:10]}")

        # Extract selected features (either single run or ensemble)

        # If top_n is specified, take top N features
        if self.top_n is not None and len(selected_features) > self.top_n:
            # Re-evaluate each feature individually to rank them
            feature_scores = {}
            for feature in selected_features:
                feature_idx = numeric_cols.index(feature)
                single_feature_chromosome = np.zeros(len(numeric_cols))
                single_feature_chromosome[feature_idx] = 1
                score = fitness_fn(single_feature_chromosome)
                feature_scores[feature] = score

            # Sort by score and take top N
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, s in sorted_features[:self.top_n]]

        if self.verbose and not self.use_nsga2:
            if self.n_ensemble == 1:
                print(f"\nGA Complete:")
                print(f"  Converged: {ga.converged_}")
                print(f"  Generations: {ga.n_generations_}")
            else:
                print(f"\nGA Ensemble Complete:")
                print(f"  Converged: {best_run['converged']}")
                print(f"  Generations: {best_run['n_generations']}")

            print(f"  Final fitness: {best_fitness:.6f}")
            print(f"  Selected features: {len(selected_features)}/{len(numeric_cols)}")
            print(f"  Features: {selected_features}")

        # Create prepared instance
        prepared = replace(self)
        prepared._selected_features = selected_features
        prepared._feature_names = numeric_cols
        prepared._ga_history = history
        prepared._final_fitness = best_fitness
        prepared._best_chromosome = best_chromosome

        if self.use_nsga2:
            prepared._converged = False  # NSGA-II doesn't have convergence
            prepared._n_generations = nsga2.n_generations_
        elif self.n_ensemble == 1:
            prepared._converged = ga.converged_
            prepared._n_generations = ga.n_generations_
        else:
            prepared._converged = best_run['converged']
            prepared._n_generations = best_run['n_generations']

        prepared._is_prepared = True
        prepared._ensemble_results = ensemble_results
        prepared._feature_frequencies = feature_frequency
        prepared._pareto_front = pareto_front_storage
        prepared._pareto_objectives = pareto_objectives_storage
        prepared._pareto_indices = pareto_indices_storage

        return prepared

    def _find_knee_point(self, pareto_objectives: np.ndarray) -> int:
        """
        Find knee point in Pareto front (best trade-off solution).

        Uses the maximum distance method: finds the point with maximum
        perpendicular distance from the line connecting the extreme points.

        Parameters
        ----------
        pareto_objectives : np.ndarray
            Objective values for Pareto front solutions (n_solutions × n_objectives)

        Returns
        -------
        knee_idx : int
            Index of knee point solution
        """
        if len(pareto_objectives) == 1:
            return 0

        # Normalize objectives to [0, 1] range
        obj_min = np.min(pareto_objectives, axis=0)
        obj_max = np.max(pareto_objectives, axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0  # Avoid division by zero

        normalized_obj = (pareto_objectives - obj_min) / obj_range

        # For 2 objectives: find point with max distance from line
        if pareto_objectives.shape[1] == 2:
            # Sort by first objective
            sorted_indices = np.argsort(normalized_obj[:, 0])
            sorted_obj = normalized_obj[sorted_indices]

            # Line from first to last point
            p1 = sorted_obj[0]
            p2 = sorted_obj[-1]

            # Calculate perpendicular distance for each point
            distances = []
            for point in sorted_obj:
                # Distance from point to line
                dist = np.abs(np.cross(p2 - p1, point - p1)) / np.linalg.norm(p2 - p1)
                distances.append(dist)

            # Return original index of point with max distance
            max_dist_idx = np.argmax(distances)
            return sorted_indices[max_dist_idx]
        else:
            # For 3+ objectives: use distance from origin in normalized space
            distances = np.linalg.norm(normalized_obj, axis=1)
            return np.argmin(distances)  # Closest to origin (best trade-off)

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to transform

        Returns
        -------
        transformed : pd.DataFrame
            Data with only selected features (plus outcome if present)
        """
        if not self._is_prepared:
            raise RuntimeError("Step must be prepared before baking")

        # Keep outcome if present
        keep_cols = self._selected_features.copy()
        if self.outcome in data.columns:
            keep_cols = [self.outcome] + keep_cols

        # Filter to selected columns
        return data[keep_cols].copy()

    def get_selected_features(self) -> List[str]:
        """
        Get list of selected features.

        Returns
        -------
        features : List[str]
            Selected feature names
        """
        if not self._is_prepared:
            raise RuntimeError("Step must be prepared before accessing selected features")
        return self._selected_features.copy()

    def get_fitness_history(self) -> List[float]:
        """
        Get GA fitness history (best fitness per generation).

        Returns
        -------
        history : List[float]
            Fitness values per generation
        """
        if not self._is_prepared:
            raise RuntimeError("Step must be prepared before accessing history")
        return self._ga_history.copy()

    def plot_convergence(self):
        """
        Plot GA convergence curve.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Convergence plot
        """
        if not self._is_prepared:
            raise RuntimeError("Step must be prepared before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(self._ga_history)), self._ga_history, linewidth=2)
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Best Fitness", fontsize=12)
        ax.set_title(f"GA Convergence ({self.metric})", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add convergence annotation
        if self._converged:
            ax.axvline(self._n_generations, color='red', linestyle='--', alpha=0.5)
            ax.text(
                self._n_generations, ax.get_ylim()[1] * 0.95,
                f'Converged\n(gen {self._n_generations})',
                ha='center', fontsize=10, color='red'
            )

        plt.tight_layout()
        return fig


def step_select_genetic_algorithm(
    recipe,
    outcome: str,
    model,
    columns=None,
    metric: str = "rmse",
    maximize: bool = False,
    top_n: Optional[int] = None,
    constraints: Dict[str, Dict[str, Any]] = None,
    mandatory_features: List[str] = None,
    forbidden_features: List[str] = None,
    feature_costs: Dict[str, float] = None,
    max_total_cost: Optional[float] = None,
    cost_weight: float = 0.0,
    sparsity_weight: float = 0.0,
    warm_start: Union[None, str, np.ndarray] = None,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: float = 0.1,
    tournament_size: int = 3,
    cv_folds: int = 5,
    convergence_threshold: float = 1e-4,
    convergence_patience: int = 10,
    adaptive_mutation: bool = False,
    adaptive_crossover: bool = False,
    relax_constraints_after: Optional[int] = None,
    relaxation_rate: float = 0.05,
    n_jobs: int = 1,
    n_ensemble: int = 1,
    ensemble_strategy: str = "voting",
    ensemble_threshold: float = 0.5,
    use_nsga2: bool = False,
    nsga2_objectives: List[str] = None,
    nsga2_selection_method: str = "knee_point",
    nsga2_selection_index: int = 0,
    random_state: Optional[int] = None,
    verbose: bool = False,
    skip: bool = False,
    id: Optional[str] = None
):
    """
    Add genetic algorithm feature selection step to recipe.

    Parameters
    ----------
    recipe : Recipe
        Recipe object to add step to
    outcome : str
        Name of outcome variable
    model : ModelSpec
        Parsnip model specification for fitness evaluation
    columns : selector, optional
        Columns to consider for selection
    metric : str, default='rmse'
        Metric to optimize
    maximize : bool, default=False
        Whether to maximize metric
    top_n : int, optional
        Number of features to select
    constraints : dict, optional
        Statistical constraints
    population_size : int, default=50
        GA population size
    generations : int, default=100
        Maximum generations
    mutation_rate : float, default=0.1
        Mutation probability
    crossover_rate : float, default=0.8
        Crossover probability
    elitism : float, default=0.1
        Elite proportion
    tournament_size : int, default=3
        Tournament size
    cv_folds : int, default=5
        CV folds for fitness evaluation
    convergence_threshold : float, default=1e-4
        Convergence threshold
    convergence_patience : int, default=10
        Convergence patience
    adaptive_mutation : bool, default=False
        Enable adaptive mutation rate
    adaptive_crossover : bool, default=False
        Enable adaptive crossover rate
    relax_constraints_after : int, optional
        Generation to start relaxing constraints
    relaxation_rate : float, default=0.05
        Rate of constraint relaxation per generation
    n_jobs : int, default=1
        Number of parallel jobs for fitness evaluation (-1 = all cores)
    n_ensemble : int, default=1
        Number of GA runs with different seeds for ensemble mode
    ensemble_strategy : str, default='voting'
        Strategy for aggregating ensemble results: 'voting', 'frequency', 'union', 'intersection'
    ensemble_threshold : float, default=0.5
        Threshold for voting/frequency strategies (proportion of runs)
    random_state : int, optional
        Random seed
    verbose : bool, default=False
        Print progress
    skip : bool, default=False
        Skip step
    id : str, optional
        Step ID

    Returns
    -------
    recipe : Recipe
        Recipe with added step

    Examples
    --------
    >>> from py_recipes import recipe
    >>> from py_recipes.steps import step_select_genetic_algorithm
    >>> from py_parsnip import linear_reg
    >>>
    >>> rec = (recipe(data, "y ~ .")
    ...        .step_select_genetic_algorithm(
    ...            outcome='y',
    ...            model=linear_reg(),
    ...            metric='rmse',
    ...            top_n=10,
    ...            generations=50
    ...        ))
    """
    if constraints is None:
        constraints = {}
    if mandatory_features is None:
        mandatory_features = []
    if forbidden_features is None:
        forbidden_features = []
    if feature_costs is None:
        feature_costs = {}
    if nsga2_objectives is None:
        nsga2_objectives = ["performance", "sparsity"]

    step = StepSelectGeneticAlgorithm(
        outcome=outcome,
        model=model,
        columns=columns,
        metric=metric,
        maximize=maximize,
        top_n=top_n,
        constraints=constraints,
        mandatory_features=mandatory_features,
        forbidden_features=forbidden_features,
        feature_costs=feature_costs,
        max_total_cost=max_total_cost,
        cost_weight=cost_weight,
        sparsity_weight=sparsity_weight,
        warm_start=warm_start,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism=elitism,
        tournament_size=tournament_size,
        cv_folds=cv_folds,
        convergence_threshold=convergence_threshold,
        convergence_patience=convergence_patience,
        adaptive_mutation=adaptive_mutation,
        adaptive_crossover=adaptive_crossover,
        relax_constraints_after=relax_constraints_after,
        relaxation_rate=relaxation_rate,
        n_jobs=n_jobs,
        n_ensemble=n_ensemble,
        ensemble_strategy=ensemble_strategy,
        ensemble_threshold=ensemble_threshold,
        use_nsga2=use_nsga2,
        nsga2_objectives=nsga2_objectives,
        nsga2_selection_method=nsga2_selection_method,
        nsga2_selection_index=nsga2_selection_index,
        random_state=random_state,
        verbose=verbose,
        skip=skip,
        id=id
    )
    recipe.steps.append(step)
    return recipe
