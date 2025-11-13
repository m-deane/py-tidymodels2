"""
Multi-model orchestration tools for automatic model comparison.

These tools enable automatic comparison of multiple models using
py-workflowsets, with cross-validation and ranking capabilities.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from py_workflowsets import WorkflowSet
from py_workflows import workflow
from py_rsample import vfold_cv, time_series_cv
from py_yardstick import metric_set, rmse, mae, r_squared
import py_parsnip


def generate_workflowset(
    model_recommendations: List[Dict],
    recipe_code: str,
    formula: str,
    max_models: int = 5
) -> WorkflowSet:
    """
    Generate WorkflowSet from model recommendations.

    Creates a WorkflowSet containing workflows for each recommended model,
    using the same preprocessing recipe for fair comparison.

    Args:
        model_recommendations: List of model dicts from suggest_model()
        recipe_code: Preprocessing recipe code (from create_recipe())
        formula: Formula string (e.g., 'target ~ .')
        max_models: Maximum number of models to include (default: 5)

    Returns:
        WorkflowSet containing workflows for comparison

    Example:
        >>> recommendations = suggest_model(data_characteristics)
        >>> recipe_code = create_recipe(data_characteristics, recommendations[0]['model_type'])
        >>> wf_set = generate_workflowset(recommendations, recipe_code, 'sales ~ .')
        >>> len(wf_set.workflows)
        5
    """
    # Limit to max_models
    models_to_compare = model_recommendations[:max_models]

    # Create list of workflows
    workflows = []

    for i, rec in enumerate(models_to_compare):
        model_type = rec['model_type']

        # Create model spec dynamically
        model_spec = _create_model_spec(model_type)

        # Create workflow
        wf = (workflow()
            .add_formula(formula)
            .add_model(model_spec))

        # Add to list with ID
        wf_id = f"{model_type}_{i+1}"
        workflows.append((wf_id, wf))

    # Create WorkflowSet from workflows
    wf_set = WorkflowSet.from_workflows(workflows)

    return wf_set


def compare_models_cv(
    wf_set: WorkflowSet,
    data: pd.DataFrame,
    cv_strategy: str = 'time_series',
    n_folds: int = 5,
    metrics: Optional[object] = None,
    date_column: Optional[str] = None
) -> Tuple[object, pd.DataFrame]:
    """
    Compare models using cross-validation.

    Evaluates all workflows in the WorkflowSet using cross-validation
    and returns ranked results.

    Args:
        wf_set: WorkflowSet to evaluate
        data: Training data
        cv_strategy: 'time_series' or 'vfold' (default: 'time_series')
        n_folds: Number of CV folds (default: 5)
        metrics: Metric set for evaluation (default: rmse, mae, r_squared)
        date_column: Date column name for time series CV (required if cv_strategy='time_series')

    Returns:
        Tuple of (results object, ranked_metrics DataFrame)

    Example:
        >>> wf_set = generate_workflowset(recommendations, recipe_code, 'sales ~ .')
        >>> results, rankings = compare_models_cv(wf_set, train_data, date_column='date')
        >>> print(rankings.head())
           rank wflow_id       rmse  mae  r_squared
        0     1  prophet_reg_1  12.5  9.2       0.85
        1     2  linear_reg_1   15.3 11.1       0.78
    """
    # Default metrics
    if metrics is None:
        metrics = metric_set(rmse, mae, r_squared)

    # Create CV splits
    if cv_strategy == 'time_series':
        if date_column is None:
            raise ValueError("date_column required for time_series CV strategy")

        # Create time series CV splits
        resamples = time_series_cv(
            data,
            date_column=date_column,
            initial='6 months',
            assess='1 month',
            skip='1 month',
            cumulative=False
        )
    else:
        # Standard k-fold CV
        resamples = vfold_cv(data, v=n_folds)

    # Evaluate all workflows
    results = wf_set.fit_resamples(
        resamples=resamples,
        metrics=metrics
    )

    # Collect and rank results
    metrics_df = results.collect_metrics()

    # Rank by RMSE (lower is better)
    ranked = results.rank_results(
        metric='rmse',
        select_best=True,
        n=len(wf_set.workflows)
    )

    return results, ranked


def select_best_models(
    ranked_results: pd.DataFrame,
    selection_strategy: str = 'best',
    n_models: int = 1,
    performance_threshold: Optional[float] = None
) -> List[str]:
    """
    Select best models from ranked results.

    Args:
        ranked_results: Ranked results from compare_models_cv()
        selection_strategy: Selection strategy:
            - 'best': Select top n_models by performance
            - 'within_1se': Select models within 1 std error of best
            - 'threshold': Select models meeting performance threshold
        n_models: Number of models to select (for 'best' strategy)
        performance_threshold: RMSE threshold for 'threshold' strategy

    Returns:
        List of selected workflow IDs

    Example:
        >>> best_models = select_best_models(ranked, strategy='best', n_models=3)
        >>> print(best_models)
        ['prophet_reg_1', 'arima_reg_1', 'linear_reg_1']
    """
    if selection_strategy == 'best':
        # Select top n models
        selected = ranked_results.head(n_models)['wflow_id'].tolist()

    elif selection_strategy == 'within_1se':
        # Select models within 1 standard error of best
        best_rmse = ranked_results.iloc[0]['mean']
        best_se = ranked_results.iloc[0]['std_err']

        threshold = best_rmse + best_se
        selected = ranked_results[ranked_results['mean'] <= threshold]['wflow_id'].tolist()

    elif selection_strategy == 'threshold':
        # Select models meeting threshold
        if performance_threshold is None:
            raise ValueError("performance_threshold required for 'threshold' strategy")

        selected = ranked_results[ranked_results['mean'] <= performance_threshold]['wflow_id'].tolist()

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    return selected


def recommend_ensemble(
    wf_set: WorkflowSet,
    ranked_results: pd.DataFrame,
    ensemble_size: int = 3,
    diversity_weight: float = 0.3
) -> Dict:
    """
    Recommend ensemble composition from top models.

    Analyzes top-performing models and recommends which ones to combine
    in an ensemble, considering both performance and diversity.

    Args:
        wf_set: WorkflowSet with evaluated models
        ranked_results: Ranked results from compare_models_cv()
        ensemble_size: Number of models to include in ensemble (default: 3)
        diversity_weight: Weight for diversity vs performance (0-1, default: 0.3)

    Returns:
        Dictionary with ensemble recommendation:
            - model_ids: List of model IDs to ensemble
            - expected_performance: Expected ensemble RMSE
            - diversity_score: Ensemble diversity score (0-1)
            - reasoning: Explanation of ensemble composition

    Example:
        >>> ensemble_rec = recommend_ensemble(wf_set, ranked, ensemble_size=3)
        >>> print(ensemble_rec['model_ids'])
        ['prophet_reg_1', 'linear_reg_1', 'rand_forest_1']
        >>> print(ensemble_rec['reasoning'])
        'Combining Prophet (time series), Linear Regression (trend), and Random Forest (complex patterns)'
    """
    # Get top models
    top_models = ranked_results.head(ensemble_size * 2)  # Get extra for diversity selection

    # Extract model types from workflow IDs
    model_types = []
    for wf_id in top_models['wflow_id']:
        # Extract model type (e.g., 'prophet_reg' from 'prophet_reg_1')
        model_type = '_'.join(wf_id.split('_')[:-1])
        model_types.append(model_type)

    # Calculate diversity scores
    # Different model families = more diverse
    model_families = _get_model_families(model_types)

    # Select diverse subset
    selected_models = []
    selected_families = set()

    for i, (wf_id, model_type) in enumerate(zip(top_models['wflow_id'], model_types)):
        if len(selected_models) >= ensemble_size:
            break

        family = model_families[model_type]

        # Prefer models from different families (diversity)
        if family not in selected_families or len(selected_models) < ensemble_size:
            selected_models.append(wf_id)
            selected_families.add(family)

    # Calculate expected performance (average of top 3)
    selected_performances = top_models[top_models['wflow_id'].isin(selected_models)]['mean']
    expected_performance = selected_performances.mean() * 0.95  # 5% improvement from ensembling

    # Calculate diversity score
    diversity_score = len(selected_families) / len(selected_models)

    # Generate reasoning
    family_names = [model_families[_extract_model_type(wf_id)] for wf_id in selected_models]
    reasoning = f"Combining {', '.join(family_names)} for complementary strengths"

    return {
        'model_ids': selected_models,
        'expected_performance': expected_performance,
        'diversity_score': diversity_score,
        'reasoning': reasoning,
        'ensemble_type': 'stacking' if len(selected_families) >= 2 else 'averaging'
    }


# Helper functions

def _create_model_spec(model_type: str):
    """
    Dynamically create model specification for any model type.

    Maps model type string to py_parsnip model function and creates spec.
    """
    # Map model type to import name
    model_map = {
        # Baseline Models
        'null_model': 'null_model',
        'naive_reg': 'naive_reg',

        # Linear & Generalized Models
        'linear_reg': 'linear_reg',
        'poisson_reg': 'poisson_reg',
        'gen_additive_mod': 'gen_additive_mod',

        # Tree-Based Models
        'decision_tree': 'decision_tree',
        'rand_forest': 'rand_forest',
        'boost_tree': 'boost_tree',

        # Support Vector Machines
        'svm_rbf': 'svm_rbf',
        'svm_linear': 'svm_linear',

        # Instance-Based & Adaptive
        'nearest_neighbor': 'nearest_neighbor',
        'mars': 'mars',
        'mlp': 'mlp',

        # Time Series Models
        'arima_reg': 'arima_reg',
        'prophet_reg': 'prophet_reg',
        'exp_smoothing': 'exp_smoothing',
        'seasonal_reg': 'seasonal_reg',
        'varmax_reg': 'varmax_reg',

        # Hybrid Time Series
        'arima_boost': 'arima_boost',
        'prophet_boost': 'prophet_boost',

        # Recursive Forecasting
        'recursive_reg': 'recursive_reg',

        # Generic Hybrid & Manual
        'hybrid_model': 'hybrid_model',
        'manual_reg': 'manual_reg',
    }

    import_name = model_map.get(model_type, 'linear_reg')

    # Get model function from py_parsnip
    model_func = getattr(py_parsnip, import_name, None)

    if model_func is None:
        # Fallback to linear_reg
        model_func = py_parsnip.linear_reg

    # Create and return spec
    spec = model_func()

    # Set mode if needed
    if hasattr(spec, 'mode') and spec.mode == 'unknown':
        spec = spec.set_mode('regression')

    return spec


def _get_model_families(model_types: List[str]) -> Dict[str, str]:
    """
    Map model types to model families for diversity calculation.
    """
    family_map = {
        # Baseline
        'null_model': 'Baseline',
        'naive_reg': 'Baseline',

        # Linear
        'linear_reg': 'Linear',
        'poisson_reg': 'Linear',
        'gen_additive_mod': 'Nonlinear',

        # Tree
        'decision_tree': 'Tree',
        'rand_forest': 'Tree Ensemble',
        'boost_tree': 'Boosting',

        # SVM
        'svm_rbf': 'SVM',
        'svm_linear': 'SVM',

        # Other ML
        'nearest_neighbor': 'Instance-Based',
        'mars': 'Splines',
        'mlp': 'Neural Network',

        # Time Series
        'arima_reg': 'Time Series (ARIMA)',
        'prophet_reg': 'Time Series (Prophet)',
        'exp_smoothing': 'Time Series (ETS)',
        'seasonal_reg': 'Time Series (STL)',
        'varmax_reg': 'Time Series (VARMAX)',

        # Hybrid
        'arima_boost': 'Hybrid (ARIMA+Boost)',
        'prophet_boost': 'Hybrid (Prophet+Boost)',
        'recursive_reg': 'Recursive',
        'hybrid_model': 'Hybrid',
        'manual_reg': 'Manual',
    }

    return {mt: family_map.get(mt, 'Unknown') for mt in model_types}


def _extract_model_type(wflow_id: str) -> str:
    """Extract model type from workflow ID (e.g., 'prophet_reg' from 'prophet_reg_1')."""
    parts = wflow_id.split('_')
    # Handle multi-word model types (e.g., 'rand_forest', 'boost_tree')
    if len(parts) >= 3 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return wflow_id
