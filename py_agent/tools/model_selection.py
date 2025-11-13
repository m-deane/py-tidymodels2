"""
Model selection tools for recommending appropriate models.

These tools analyze data characteristics and constraints to recommend
the most suitable py-tidymodels models for forecasting tasks.
"""

import pandas as pd
from typing import Dict, List, Optional


def suggest_model(
    data_characteristics: Dict,
    constraints: Optional[Dict] = None
) -> List[Dict]:
    """
    Recommend appropriate models based on data and constraints.

    Uses a rule-based system to match data characteristics with
    model capabilities and filter by user constraints.

    Args:
        data_characteristics: Output from analyze_temporal_patterns()
        constraints: Optional dict with:
            - max_train_time: Maximum training time in seconds
            - interpretability: 'low', 'medium', or 'high'
            - max_memory: Maximum memory in MB
            - min_accuracy: Minimum acceptable accuracy tier

    Returns:
        List of recommended models, sorted by confidence

    Example:
        >>> char = {
        ...     'frequency': 'daily',
        ...     'seasonality': {'detected': True, 'strength': 0.8},
        ...     'n_observations': 365
        ... }
        >>> models = suggest_model(char)
        >>> models[0]['model_type']
        'prophet_reg'
    """
    if constraints is None:
        constraints = {}

    # Get model profiles
    profiles = get_model_profiles()

    # Score each model
    scored_models = []

    for model_type, profile in profiles.items():
        # Check hard constraints first
        if not _meets_constraints(profile, constraints, data_characteristics):
            continue

        # Calculate suitability score
        score = _calculate_suitability_score(
            profile,
            data_characteristics
        )

        # Generate reasoning
        reasoning = _generate_reasoning(
            model_type,
            profile,
            data_characteristics
        )

        scored_models.append({
            'model_type': model_type,
            'reasoning': reasoning,
            'expected_performance': profile['accuracy_tier'],
            'confidence': score,
            'train_time_estimate': _estimate_train_time(
                profile,
                data_characteristics['n_observations']
            ),
            'interpretability': profile['interpretability']
        })

    # Sort by confidence score
    scored_models.sort(key=lambda x: x['confidence'], reverse=True)

    return scored_models[:5]  # Return top 5


def get_model_profiles() -> Dict:
    """
    Get profiles for all supported models.

    Returns dictionary mapping model type to its characteristics:
    - train_time_per_1k: Training time per 1000 samples (seconds)
    - predict_time_per_1k: Prediction time per 1000 samples (seconds)
    - memory_per_feature: Memory usage per feature (MB)
    - interpretability: 'low', 'medium', or 'high'
    - accuracy_tier: 'low', 'medium', 'high', 'very_high'
    - strengths: List of data pattern strengths
    - weaknesses: List of limitations

    Returns:
        Dictionary of model profiles

    Example:
        >>> profiles = get_model_profiles()
        >>> profiles['prophet_reg']['interpretability']
        'medium'
    """
    return {
        # Baseline Models
        'null_model': {
            'train_time_per_1k': 0.001,
            'predict_time_per_1k': 0.0001,
            'memory_per_feature': 0.001,
            'interpretability': 'high',
            'accuracy_tier': 'low',
            'strengths': ['speed', 'simplicity', 'baseline'],
            'weaknesses': ['accuracy', 'no_pattern_capture'],
            'good_for_seasonality': False,
            'good_for_trend': False,
            'good_for_interactions': False
        },
        'naive_reg': {
            'train_time_per_1k': 0.002,
            'predict_time_per_1k': 0.0001,
            'memory_per_feature': 0.002,
            'interpretability': 'high',
            'accuracy_tier': 'low',
            'strengths': ['speed', 'simplicity', 'baseline', 'seasonality'],
            'weaknesses': ['accuracy', 'limited_patterns'],
            'good_for_seasonality': True,
            'good_for_trend': False,
            'good_for_interactions': False
        },

        # Linear & Generalized Models
        'linear_reg': {
            'train_time_per_1k': 0.05,
            'predict_time_per_1k': 0.001,
            'memory_per_feature': 0.008,
            'interpretability': 'high',
            'accuracy_tier': 'medium',
            'strengths': ['linear_trends', 'interpretability', 'speed'],
            'weaknesses': ['nonlinear_patterns', 'complex_interactions'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'poisson_reg': {
            'train_time_per_1k': 0.08,
            'predict_time_per_1k': 0.002,
            'memory_per_feature': 0.01,
            'interpretability': 'high',
            'accuracy_tier': 'medium',
            'strengths': ['count_data', 'interpretability', 'nonnegative'],
            'weaknesses': ['continuous_data', 'overdispersion'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'gen_additive_mod': {
            'train_time_per_1k': 1.2,
            'predict_time_per_1k': 0.01,
            'memory_per_feature': 0.1,
            'interpretability': 'high',
            'accuracy_tier': 'medium-high',
            'strengths': ['nonlinear_trends', 'interpretability', 'smoothness'],
            'weaknesses': ['interactions', 'high_dimensional'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': False
        },

        # Tree-Based Models
        'decision_tree': {
            'train_time_per_1k': 0.5,
            'predict_time_per_1k': 0.001,
            'memory_per_feature': 0.05,
            'interpretability': 'high',
            'accuracy_tier': 'medium',
            'strengths': ['interpretability', 'interactions', 'nonlinear'],
            'weaknesses': ['overfitting', 'instability'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'rand_forest': {
            'train_time_per_1k': 3.5,
            'predict_time_per_1k': 0.005,
            'memory_per_feature': 0.2,
            'interpretability': 'medium',
            'accuracy_tier': 'high',
            'strengths': ['nonlinear_patterns', 'interactions', 'robustness'],
            'weaknesses': ['extrapolation', 'interpretability'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'boost_tree': {
            'train_time_per_1k': 5.2,
            'predict_time_per_1k': 0.003,
            'memory_per_feature': 0.15,
            'interpretability': 'medium',
            'accuracy_tier': 'very_high',
            'strengths': ['complex_patterns', 'interactions', 'accuracy'],
            'weaknesses': ['speed', 'overfitting_risk'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },

        # Support Vector Machines
        'svm_rbf': {
            'train_time_per_1k': 8.0,
            'predict_time_per_1k': 0.02,
            'memory_per_feature': 0.3,
            'interpretability': 'low',
            'accuracy_tier': 'high',
            'strengths': ['nonlinear_patterns', 'robustness', 'accuracy'],
            'weaknesses': ['speed', 'interpretability', 'hyperparameters'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'svm_linear': {
            'train_time_per_1k': 0.8,
            'predict_time_per_1k': 0.002,
            'memory_per_feature': 0.05,
            'interpretability': 'medium',
            'accuracy_tier': 'medium',
            'strengths': ['linear_patterns', 'robustness', 'speed'],
            'weaknesses': ['nonlinear_patterns', 'hyperparameters'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': False
        },

        # Instance-Based & Adaptive
        'nearest_neighbor': {
            'train_time_per_1k': 0.01,
            'predict_time_per_1k': 0.5,
            'memory_per_feature': 0.8,
            'interpretability': 'high',
            'accuracy_tier': 'medium',
            'strengths': ['simplicity', 'nonlinear', 'local_patterns'],
            'weaknesses': ['prediction_speed', 'memory', 'curse_of_dimensionality'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'mars': {
            'train_time_per_1k': 2.5,
            'predict_time_per_1k': 0.005,
            'memory_per_feature': 0.12,
            'interpretability': 'high',
            'accuracy_tier': 'high',
            'strengths': ['nonlinear', 'interpretability', 'automatic_interactions'],
            'weaknesses': ['computational_cost', 'overfitting_risk'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'mlp': {
            'train_time_per_1k': 10.0,
            'predict_time_per_1k': 0.01,
            'memory_per_feature': 0.4,
            'interpretability': 'low',
            'accuracy_tier': 'high',
            'strengths': ['complex_patterns', 'nonlinear', 'flexibility'],
            'weaknesses': ['interpretability', 'hyperparameters', 'overfitting'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },

        # Time Series Models
        'arima_reg': {
            'train_time_per_1k': 1.5,
            'predict_time_per_1k': 0.008,
            'memory_per_feature': 0.1,
            'interpretability': 'medium',
            'accuracy_tier': 'medium-high',
            'strengths': ['autocorrelation', 'short_term_forecasts'],
            'weaknesses': ['seasonality', 'long_series', 'nonstationary'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'prophet_reg': {
            'train_time_per_1k': 2.1,
            'predict_time_per_1k': 0.01,
            'memory_per_feature': 0.5,
            'interpretability': 'medium',
            'accuracy_tier': 'high',
            'strengths': ['seasonality', 'holidays', 'missing_data', 'trend_changes'],
            'weaknesses': ['short_series', 'irregular_spacing'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'exp_smoothing': {
            'train_time_per_1k': 0.3,
            'predict_time_per_1k': 0.002,
            'memory_per_feature': 0.02,
            'interpretability': 'high',
            'accuracy_tier': 'medium',
            'strengths': ['simple', 'seasonal', 'trend', 'speed'],
            'weaknesses': ['complex_patterns', 'limited_exogenous'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'seasonal_reg': {
            'train_time_per_1k': 1.8,
            'predict_time_per_1k': 0.01,
            'memory_per_feature': 0.15,
            'interpretability': 'high',
            'accuracy_tier': 'medium-high',
            'strengths': ['seasonality', 'decomposition', 'interpretability'],
            'weaknesses': ['complex_interactions', 'irregular_patterns'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': False
        },
        'varmax_reg': {
            'train_time_per_1k': 3.0,
            'predict_time_per_1k': 0.015,
            'memory_per_feature': 0.25,
            'interpretability': 'medium',
            'accuracy_tier': 'high',
            'strengths': ['multivariate', 'cross_correlations', 'exogenous'],
            'weaknesses': ['computational_cost', 'requires_multiple_outcomes'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        },

        # Hybrid Time Series
        'arima_boost': {
            'train_time_per_1k': 6.0,
            'predict_time_per_1k': 0.01,
            'memory_per_feature': 0.2,
            'interpretability': 'low',
            'accuracy_tier': 'very_high',
            'strengths': ['hybrid', 'accuracy', 'captures_residuals'],
            'weaknesses': ['complexity', 'interpretability'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'prophet_boost': {
            'train_time_per_1k': 7.0,
            'predict_time_per_1k': 0.012,
            'memory_per_feature': 0.25,
            'interpretability': 'low',
            'accuracy_tier': 'very_high',
            'strengths': ['hybrid', 'seasonality', 'accuracy'],
            'weaknesses': ['complexity', 'speed'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': True
        },

        # Recursive Forecasting
        'recursive_reg': {
            'train_time_per_1k': 4.0,
            'predict_time_per_1k': 0.02,
            'memory_per_feature': 0.18,
            'interpretability': 'medium',
            'accuracy_tier': 'high',
            'strengths': ['multi_step', 'autoregression', 'any_ml_model'],
            'weaknesses': ['error_accumulation', 'computational_cost'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': True
        },

        # Generic Hybrid & Manual
        'hybrid_model': {
            'train_time_per_1k': 8.0,
            'predict_time_per_1k': 0.015,
            'memory_per_feature': 0.35,
            'interpretability': 'low',
            'accuracy_tier': 'very_high',
            'strengths': ['flexibility', 'ensemble', 'adaptive'],
            'weaknesses': ['complexity', 'tuning_required'],
            'good_for_seasonality': True,
            'good_for_trend': True,
            'good_for_interactions': True
        },
        'manual_reg': {
            'train_time_per_1k': 0.001,
            'predict_time_per_1k': 0.0001,
            'memory_per_feature': 0.005,
            'interpretability': 'high',
            'accuracy_tier': 'varies',
            'strengths': ['domain_knowledge', 'reproducibility', 'external_models'],
            'weaknesses': ['requires_coefficients', 'no_learning'],
            'good_for_seasonality': False,
            'good_for_trend': True,
            'good_for_interactions': True
        }
    }


# Helper functions

def _meets_constraints(
    profile: Dict,
    constraints: Dict,
    data_characteristics: Dict
) -> bool:
    """Check if model meets user constraints."""
    # Training time constraint
    if 'max_train_time' in constraints:
        estimated_time = _estimate_train_time(
            profile,
            data_characteristics['n_observations']
        )
        if estimated_time > constraints['max_train_time']:
            return False

    # Interpretability constraint
    if 'interpretability' in constraints:
        required = constraints['interpretability']
        interp_ranking = {'low': 0, 'medium': 1, 'high': 2}
        if interp_ranking.get(profile['interpretability'], 0) < interp_ranking.get(required, 0):
            return False

    # Memory constraint
    if 'max_memory' in constraints:
        # Rough estimate: memory scales with features
        n_features = data_characteristics.get('n_features', 10)
        estimated_memory = profile['memory_per_feature'] * n_features
        if estimated_memory > constraints['max_memory']:
            return False

    return True


def _calculate_suitability_score(
    profile: Dict,
    data_characteristics: Dict
) -> float:
    """
    Calculate how suitable a model is for the data.

    Returns score between 0 and 1.
    """
    score = 0.5  # Base score

    # Seasonality bonus
    if data_characteristics['seasonality']['detected']:
        if profile['good_for_seasonality']:
            score += 0.2 * data_characteristics['seasonality']['strength']
        else:
            score -= 0.1 * data_characteristics['seasonality']['strength']

    # Trend bonus
    if data_characteristics['trend']['significant']:
        if profile['good_for_trend']:
            score += 0.1 * data_characteristics['trend']['strength']

    # Autocorrelation bonus (for ARIMA-type models)
    if 'autocorrelation' in profile['strengths']:
        avg_autocorr = sum(data_characteristics['autocorrelation'].values()) / len(
            data_characteristics['autocorrelation']
        )
        if avg_autocorr > 0.5:
            score += 0.15

    # Data quality penalties
    if data_characteristics['missing_rate'] > 0.1:
        if 'missing_data' not in profile['strengths']:
            score -= 0.1

    if data_characteristics['outlier_rate'] > 0.05:
        if 'robustness' not in profile['strengths']:
            score -= 0.05

    # Sample size considerations
    n_obs = data_characteristics['n_observations']
    if n_obs < 100:
        # Prefer simpler models for small samples
        if profile['interpretability'] == 'high':
            score += 0.1
        if profile['accuracy_tier'] == 'very_high':
            score -= 0.1  # High capacity models may overfit

    # Clip to [0, 1]
    return max(0.0, min(1.0, score))


def _generate_reasoning(
    model_type: str,
    profile: Dict,
    data_characteristics: Dict
) -> str:
    """Generate human-readable reasoning for model recommendation."""
    reasons = []

    # Seasonality
    if data_characteristics['seasonality']['detected']:
        strength = data_characteristics['seasonality']['strength']
        if profile['good_for_seasonality']:
            reasons.append(
                f"Strong seasonality detected (strength={strength:.2f}), "
                f"{model_type} excels at handling seasonal patterns"
            )
        else:
            reasons.append(
                f"Seasonality detected (strength={strength:.2f}), "
                f"consider adding seasonal features if using {model_type}"
            )

    # Trend
    if data_characteristics['trend']['significant']:
        if profile['good_for_trend']:
            direction = data_characteristics['trend']['direction']
            reasons.append(
                f"Clear {direction} trend, {model_type} can model this effectively"
            )

    # Sample size
    n_obs = data_characteristics['n_observations']
    if n_obs < 100:
        reasons.append(
            f"Small sample size (n={n_obs}), {model_type} is appropriate for limited data"
        )
    elif n_obs > 1000:
        if profile['accuracy_tier'] == 'very_high':
            reasons.append(
                f"Large dataset (n={n_obs}) enables complex model like {model_type}"
            )

    # Interpretability
    if profile['interpretability'] == 'high':
        reasons.append(f"{model_type} provides high interpretability")

    # Combine reasons
    if reasons:
        return ". ".join(reasons) + "."
    else:
        return f"{model_type} is suitable for this forecasting task."


def _estimate_train_time(profile: Dict, n_observations: int) -> float:
    """
    Estimate training time in seconds.

    Args:
        profile: Model profile dictionary
        n_observations: Number of observations in dataset

    Returns:
        Estimated training time in seconds
    """
    # Scale training time based on sample size
    time_per_1k = profile['train_time_per_1k']
    estimated_time = time_per_1k * (n_observations / 1000)

    return estimated_time
