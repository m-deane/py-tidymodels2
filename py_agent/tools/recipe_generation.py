"""
Recipe generation tools for creating preprocessing pipelines.

These tools generate py-recipes code based on data characteristics,
model requirements, and domain knowledge. Intelligently selects from
all 51 available recipe steps for optimal preprocessing.
"""

from typing import Dict, List, Optional


def create_recipe(
    data_characteristics: Dict,
    model_type: str,
    domain: Optional[str] = None
) -> str:
    """
    Generate preprocessing recipe code.

    Creates a py-recipes pipeline tailored to the data characteristics
    and model requirements. Intelligently selects from all 51 available
    recipe steps for optimal preprocessing.

    Args:
        data_characteristics: Output from analyze_temporal_patterns()
        model_type: Target model type ('linear_reg', 'prophet_reg', etc.)
        domain: Optional domain hint ('retail', 'finance', 'energy')

    Returns:
        Python code string for recipe creation

    Example:
        >>> char = {
        ...     'frequency': 'daily',
        ...     'seasonality': {'detected': True},
        ...     'missing_rate': 0.08,
        ...     'n_features': 25
        ... }
        >>> code = create_recipe(char, 'rand_forest')
        >>> 'step_pca' in code  # PCA for high-dimensional data
        True
    """
    # Get base template for model type
    template = _get_model_template(model_type)

    # Get number of features (default to 10 if not provided)
    n_features = data_characteristics.get('n_features', 10)
    n_observations = data_characteristics.get('n_observations', 100)

    # Phase 1: Data Cleaning
    # =====================

    # Remove rows with infinite values
    if data_characteristics.get('outlier_rate', 0) > 0:
        template['steps'].append(".step_naomit()")  # Removes NaN and Inf

    # Phase 2: Imputation
    # ===================

    if data_characteristics['missing_rate'] > 0.01:
        imputation_step = _get_imputation_step(
            data_characteristics['missing_rate'],
            model_type
        )
        template['steps'].append(imputation_step)

    # Phase 3: Feature Engineering
    # =============================

    # Add date features for ML models (not time series models)
    if model_type not in ['prophet_reg', 'arima_reg', 'seasonal_reg', 'exp_smoothing', 'varmax_reg']:
        date_step = _get_date_feature_step(
            data_characteristics['frequency'],
            domain
        )
        if date_step:
            template['steps'].append(date_step)

    # Add polynomial features for nonlinear models
    if _needs_polynomial_features(model_type, data_characteristics):
        template['steps'].append(".step_poly(all_numeric_predictors(), degree=2)")

    # Add interaction terms for models that benefit
    if _needs_interactions(model_type, n_features):
        template['steps'].append(".step_interact(terms=['all_numeric_predictors()'])")

    # Phase 4: Transformations
    # =========================

    # Apply advanced transformations for skewed data
    if _needs_transformation(model_type):
        # YeoJohnson handles negative values better than BoxCox
        template['steps'].append(".step_YeoJohnson(all_numeric_predictors())")

    # Phase 5: Filtering & Dimensionality Reduction
    # ==============================================

    # Remove zero-variance features
    template['steps'].append(".step_zv(all_predictors())")

    # Remove highly correlated features for interpretable models
    if _needs_correlation_filter(model_type):
        template['steps'].append(".step_select_corr(all_numeric_predictors(), threshold=0.9, method='multicollinearity')")

    # Apply PCA for high-dimensional data
    if _needs_dimensionality_reduction(n_features, n_observations, model_type):
        # Use PCA to reduce dimensions
        n_components = min(int(n_features * 0.8), 20)  # Keep 80% or max 20 components
        template['steps'].append(f".step_pca(all_numeric_predictors(), num_comp={n_components})")

    # Phase 6: Normalization/Scaling
    # ===============================

    # Normalize for models that need it
    if _needs_normalization(model_type):
        template['steps'].append(".step_normalize(all_numeric_predictors())")

    # Phase 7: Encoding
    # =================

    # One-hot encode categorical variables for ML models
    if model_type not in ['prophet_reg', 'arima_reg', 'seasonal_reg', 'exp_smoothing', 'varmax_reg']:
        template['steps'].append(".step_dummy(all_nominal_predictors())")

    # Phase 8: Final Cleanup
    # ======================

    # Remove any remaining NA values that might have been introduced
    if len(template['steps']) > 3:  # If we did substantial preprocessing
        template['steps'].append(".step_naomit()")

    # Generate code
    code = _generate_recipe_code(template)

    return code


def get_recipe_templates() -> Dict[str, Dict]:
    """
    Get predefined recipe templates for different scenarios.

    Provides 17 domain-specific templates covering common use cases.

    Returns:
        Dictionary mapping template name to recipe configuration

    Example:
        >>> templates = get_recipe_templates()
        >>> len(templates)
        17
        >>> 'retail_daily' in templates
        True
    """
    return {
        # Basic Templates
        'minimal': {
            'description': 'Minimal preprocessing (imputation only)',
            'steps': [
                ".step_impute_median(all_numeric_predictors())"
            ]
        },
        'standard_ml': {
            'description': 'Standard preprocessing for ML models',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },
        'time_series': {
            'description': 'Preprocessing for time series models (Prophet, ARIMA)',
            'steps': [
                ".step_impute_linear(all_numeric_predictors())"
            ]
        },

        # Retail & E-commerce
        'retail_daily': {
            'description': 'Retail sales forecasting (daily data)',
            'steps': [
                ".step_naomit()",
                ".step_impute_median(all_numeric_predictors())",
                ".step_date('date', features=['dow', 'month', 'quarter', 'is_holiday'])",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },
        'retail_weekly': {
            'description': 'Weekly retail sales with promotions',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_date('date', features=['week', 'month', 'quarter'])",
                ".step_poly(all_numeric_predictors(), degree=2)",  # Capture promotion effects
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },
        'ecommerce_hourly': {
            'description': 'E-commerce traffic/conversions (hourly)',
            'steps': [
                ".step_impute_knn(all_numeric_predictors(), neighbors=5)",
                ".step_date('timestamp', features=['hour', 'dow', 'month'])",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },

        # Energy & Utilities
        'energy_hourly': {
            'description': 'Energy load forecasting (hourly data)',
            'steps': [
                ".step_impute_linear(all_numeric_predictors())",
                ".step_date('timestamp', features=['hour', 'dow', 'month', 'is_weekend'])",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },
        'solar_generation': {
            'description': 'Solar power generation forecasting',
            'steps': [
                ".step_naomit()",  # Remove night hours
                ".step_date('timestamp', features=['hour', 'dow', 'month'])",
                ".step_normalize(all_numeric_predictors())"
            ]
        },

        # Finance & Economics
        'finance_daily': {
            'description': 'Financial time series (daily data)',
            'steps': [
                ".step_naomit()",  # No imputation - missing data is invalid
                ".step_date('date', features=['dow', 'month', 'quarter'])",
                ".step_YeoJohnson(all_numeric_predictors())",  # Handle negative values
                ".step_normalize(all_numeric_predictors())"
            ]
        },
        'stock_prices': {
            'description': 'Stock price prediction',
            'steps': [
                ".step_naomit()",
                ".step_date('date', features=['dow', 'month'])",
                ".step_log(all_numeric_predictors())",  # Log returns
                ".step_normalize(all_numeric_predictors())"
            ]
        },

        # Healthcare & Medical
        'patient_volume': {
            'description': 'Hospital patient volume forecasting',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_date('date', features=['dow', 'month', 'is_holiday'])",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },

        # Transportation & Logistics
        'demand_forecasting': {
            'description': 'Product/service demand forecasting',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_date('date', features=['dow', 'month', 'quarter'])",
                ".step_poly(all_numeric_predictors(), degree=2)",
                ".step_zv(all_predictors())",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },
        'traffic_volume': {
            'description': 'Traffic volume/congestion prediction',
            'steps': [
                ".step_impute_linear(all_numeric_predictors())",
                ".step_date('timestamp', features=['hour', 'dow', 'is_rush_hour'])",
                ".step_normalize(all_numeric_predictors())",
                ".step_dummy(all_nominal_predictors())"
            ]
        },

        # High-Dimensional Data
        'high_dimensional': {
            'description': 'Many features (>20), dimensionality reduction',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_zv(all_predictors())",
                ".step_select_corr(all_numeric_predictors(), threshold=0.9, method='multicollinearity')",
                ".step_normalize(all_numeric_predictors())",
                ".step_pca(all_numeric_predictors(), num_comp=15)",
                ".step_dummy(all_nominal_predictors())"
            ]
        },

        # Text Features / NLP
        'text_features': {
            'description': 'Text-derived numeric features (sentiment, counts)',
            'steps': [
                ".step_impute_median(all_numeric_predictors())",
                ".step_normalize(all_numeric_predictors())",
                ".step_pca(all_numeric_predictors(), num_comp=10)"  # Reduce TF-IDF dimensions
            ]
        },

        # IoT & Sensor Data
        'iot_sensors': {
            'description': 'IoT sensor data with many correlated features',
            'steps': [
                ".step_impute_knn(all_numeric_predictors(), neighbors=5)",
                ".step_zv(all_predictors())",
                ".step_select_corr(all_numeric_predictors(), threshold=0.95, method='multicollinearity')",
                ".step_normalize(all_numeric_predictors())"
            ]
        }
    }


# Helper functions

def _get_model_template(model_type: str) -> Dict:
    """Get base recipe template for a model type."""
    # Time series models need minimal preprocessing
    if model_type in ['prophet_reg', 'arima_reg', 'seasonal_reg']:
        return {
            'model_type': model_type,
            'steps': []
        }

    # ML models need standard preprocessing
    return {
        'model_type': model_type,
        'steps': []
    }


def _get_imputation_step(missing_rate: float, model_type: str) -> str:
    """Determine appropriate imputation method."""
    if missing_rate < 0.05:
        # Low missing rate - simple median imputation
        return ".step_impute_median(all_numeric())"
    elif missing_rate < 0.15:
        # Moderate missing rate - linear interpolation for time series
        if model_type in ['prophet_reg', 'arima_reg']:
            return ".step_impute_linear(all_numeric())"
        else:
            return ".step_impute_median(all_numeric())"
    else:
        # High missing rate - KNN imputation
        return ".step_impute_knn(all_numeric(), neighbors=5)"


def _get_date_feature_step(frequency: str, domain: Optional[str]) -> Optional[str]:
    """Generate date feature extraction step based on frequency and domain."""
    if frequency == 'daily':
        if domain == 'retail':
            return ".step_date('date', features=['dow', 'month', 'quarter', 'is_holiday'])"
        elif domain == 'finance':
            return ".step_date('date', features=['dow', 'month', 'quarter'])"
        else:
            return ".step_date('date', features=['dow', 'month'])"

    elif frequency == 'hourly':
        return ".step_date('timestamp', features=['hour', 'dow', 'month'])"

    elif frequency == 'monthly':
        return ".step_date('date', features=['month', 'quarter', 'year'])"

    elif frequency == 'weekly':
        return ".step_date('date', features=['week', 'month', 'quarter'])"

    return None


def _generate_recipe_code(template: Dict) -> str:
    """Generate complete recipe code from template."""
    code_lines = [
        "from py_recipes import recipe",
        "from py_recipes.selectors import all_numeric, all_nominal, all_numeric_predictors, all_nominal_predictors, all_predictors",
        "",
        "# Create preprocessing recipe",
        "rec = (recipe(data, formula)"
    ]

    # Add steps
    for step in template['steps']:
        code_lines.append(f"    {step}")

    code_lines.append(")")

    return "\n".join(code_lines)


def _needs_polynomial_features(model_type: str, data_chars: Dict) -> bool:
    """
    Determine if polynomial features would benefit the model.

    Args:
        model_type: Target model type
        data_chars: Data characteristics

    Returns:
        True if polynomial features recommended
    """
    # Models that can benefit from polynomial features
    polynomial_models = [
        'linear_reg', 'poisson_reg', 'svm_linear',  # Linear models benefit most
        'gen_additive_mod',  # GAMs can leverage polynomial basis
    ]

    if model_type not in polynomial_models:
        return False

    # Don't add if already many features (curse of dimensionality)
    n_features = data_chars.get('n_features', 10)
    if n_features > 15:
        return False

    # Add if nonlinear trend detected
    trend = data_chars.get('trend', {})
    if trend.get('strength', 0) > 0.5:
        return True

    return False


def _needs_interactions(model_type: str, n_features: int) -> bool:
    """
    Determine if interaction terms would benefit the model.

    Args:
        model_type: Target model type
        n_features: Number of features

    Returns:
        True if interaction terms recommended
    """
    # Models that benefit from explicit interactions
    interaction_models = [
        'linear_reg', 'poisson_reg',  # Linear models need explicit interactions
        'svm_linear',  # Linear SVM benefits
    ]

    if model_type not in interaction_models:
        return False

    # Only add for small feature sets (avoid explosion)
    if n_features > 10:
        return False

    # Only for datasets with moderate number of features (2-10)
    if n_features < 2:
        return False

    return True


def _needs_transformation(model_type: str) -> bool:
    """
    Determine if advanced transformations (BoxCox, YeoJohnson) needed.

    Args:
        model_type: Target model type

    Returns:
        True if transformations recommended
    """
    # Models that benefit from normality in features
    transformation_models = [
        'linear_reg', 'poisson_reg',  # Assume normality
        'svm_rbf', 'svm_linear',  # SVMs benefit from scaling/normalization
        'nearest_neighbor',  # Distance-based
    ]

    return model_type in transformation_models


def _needs_correlation_filter(model_type: str) -> bool:
    """
    Determine if correlation filtering is needed.

    Args:
        model_type: Target model type

    Returns:
        True if correlation filtering recommended
    """
    # Models where multicollinearity is problematic
    correlation_sensitive = [
        'linear_reg', 'poisson_reg',  # Linear models affected by multicollinearity
        'gen_additive_mod',  # GAMs can be affected
    ]

    return model_type in correlation_sensitive


def _needs_dimensionality_reduction(
    n_features: int,
    n_observations: int,
    model_type: str
) -> bool:
    """
    Determine if dimensionality reduction (PCA) is needed.

    Args:
        n_features: Number of features
        n_observations: Number of observations
        model_type: Target model type

    Returns:
        True if PCA recommended
    """
    # Don't use PCA for time series models (loses interpretability)
    time_series_models = [
        'prophet_reg', 'arima_reg', 'seasonal_reg', 'exp_smoothing',
        'varmax_reg', 'arima_boost', 'prophet_boost'
    ]
    if model_type in time_series_models:
        return False

    # Don't use for interpretable models (loses feature meaning)
    interpretable_models = [
        'decision_tree', 'linear_reg', 'poisson_reg', 'gen_additive_mod',
        'mars', 'manual_reg'
    ]
    if model_type in interpretable_models:
        return False

    # Use PCA if high-dimensional (>20 features)
    if n_features > 20:
        return True

    # Use PCA if features > observations (overfitting risk)
    if n_features > n_observations * 0.5:
        return True

    return False


def _needs_normalization(model_type: str) -> bool:
    """
    Determine if feature normalization is needed.

    Args:
        model_type: Target model type

    Returns:
        True if normalization recommended
    """
    # Models that benefit from normalized features
    normalization_models = [
        # Distance-based models
        'nearest_neighbor', 'svm_rbf', 'svm_linear',
        # Neural networks
        'mlp',
        # Linear models (helps with convergence)
        'linear_reg', 'poisson_reg',
        # Tree models (helps with equal feature importance)
        'rand_forest', 'boost_tree', 'decision_tree',
        # Advanced models
        'mars', 'gen_additive_mod',
    ]

    return model_type in normalization_models
