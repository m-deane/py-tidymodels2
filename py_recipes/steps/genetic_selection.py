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
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism: float = 0.1
    tournament_size: int = 3
    cv_folds: int = 5
    convergence_threshold: float = 1e-4
    convergence_patience: int = 10
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
            fitness_fn = create_constrained_fitness_function(
                data=data,
                outcome_col=self.outcome,
                feature_names=numeric_cols,
                model_spec=self.model,
                base_fitness_fn=base_fitness_fn,
                constraints=self.constraints
            )
        else:
            fitness_fn = base_fitness_fn

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
            random_state=self.random_state,
            verbose=self.verbose
        )

        # Run GA
        ga = GeneticAlgorithm(
            n_features=len(numeric_cols),
            fitness_function=fitness_fn,
            config=ga_config,
            mandatory_indices=mandatory_indices,
            forbidden_indices=forbidden_indices,
            feature_costs=cost_array,
            max_cost=self.max_total_cost
        )

        best_chromosome, best_fitness, history = ga.evolve()

        # Extract selected features
        selected_indices = np.where(best_chromosome == 1)[0]
        selected_features = [numeric_cols[i] for i in selected_indices]

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

        if self.verbose:
            print(f"\nGA Complete:")
            print(f"  Converged: {ga.converged_}")
            print(f"  Generations: {ga.n_generations_}")
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
        prepared._converged = ga.converged_
        prepared._n_generations = ga.n_generations_
        prepared._is_prepared = True

        return prepared

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
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: float = 0.1,
    tournament_size: int = 3,
    cv_folds: int = 5,
    convergence_threshold: float = 1e-4,
    convergence_patience: int = 10,
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
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism=elitism,
        tournament_size=tournament_size,
        cv_folds=cv_folds,
        convergence_threshold=convergence_threshold,
        convergence_patience=convergence_patience,
        random_state=random_state,
        verbose=verbose,
        skip=skip,
        id=id
    )
    recipe.steps.append(step)
    return recipe
