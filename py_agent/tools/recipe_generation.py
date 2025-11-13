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
    and model requirements.

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
        ...     'missing_rate': 0.08
        ... }
        >>> code = create_recipe(char, 'prophet_reg')
        >>> 'step_impute' in code
        True
    """
    # Get base template for model type
    template = _get_model_template(model_type)

    # Add imputation if needed
    if data_characteristics['missing_rate'] > 0.01:
        imputation_step = _get_imputation_step(
            data_characteristics['missing_rate'],
            model_type
        )
        template['steps'].insert(0, imputation_step)

    # Add date features if time series model
    if model_type in ['prophet_reg', 'arima_reg', 'seasonal_reg']:
        # Minimal preprocessing for time series models
        pass
    else:
        # Add date feature extraction for ML models
        date_step = _get_date_feature_step(
            data_characteristics['frequency'],
            domain
        )
        if date_step:
            template['steps'].append(date_step)

    # Add outlier handling if needed
    if data_characteristics['outlier_rate'] > 0.05:
        outlier_step = ".step_filter_outliers(method='iqr', factor=3.0)"
        template['steps'].append(outlier_step)

    # Add normalization for ML models
    if model_type in ['rand_forest', 'boost_tree', 'linear_reg']:
        template['steps'].append(".step_normalize(all_numeric())")

    # Generate code
    code = _generate_recipe_code(template)

    return code


def get_recipe_templates() -> Dict[str, Dict]:
    """
    Get predefined recipe templates for different scenarios.

    Returns:
        Dictionary mapping template name to recipe configuration

    Example:
        >>> templates = get_recipe_templates()
        >>> 'retail_daily' in templates
        True
    """
    return {
        'minimal': {
            'description': 'Minimal preprocessing (imputation only)',
            'steps': [
                ".step_impute_median(all_numeric())"
            ]
        },
        'standard_ml': {
            'description': 'Standard preprocessing for ML models',
            'steps': [
                ".step_impute_median(all_numeric())",
                ".step_normalize(all_numeric())",
                ".step_dummy(all_nominal())"
            ]
        },
        'time_series': {
            'description': 'Preprocessing for time series models (Prophet, ARIMA)',
            'steps': [
                ".step_impute_median(all_numeric())"
            ]
        },
        'retail_daily': {
            'description': 'Retail sales forecasting (daily data)',
            'steps': [
                ".step_impute_median(all_numeric())",
                ".step_date('date', features=['dow', 'month', 'quarter', 'is_holiday'])",
                ".step_normalize(all_numeric())",
                ".step_dummy(all_nominal())"
            ]
        },
        'energy_hourly': {
            'description': 'Energy load forecasting (hourly data)',
            'steps': [
                ".step_impute_linear(all_numeric())",
                ".step_date('timestamp', features=['hour', 'dow', 'month'])",
                ".step_normalize(all_numeric())"
            ]
        },
        'finance_daily': {
            'description': 'Financial time series (daily data)',
            'steps': [
                ".step_naomit()",  # No imputation - missing data is invalid
                ".step_date('date', features=['dow', 'month', 'quarter'])",
                ".step_YeoJohnson(all_numeric())",  # Handle negative values
                ".step_normalize(all_numeric())"
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
        "from py_recipes.selectors import all_numeric, all_nominal",
        "",
        "# Create preprocessing recipe",
        "rec = (recipe(data, formula)"
    ]

    # Add steps
    for step in template['steps']:
        code_lines.append(f"    {step}")

    code_lines.append(")")

    return "\n".join(code_lines)
