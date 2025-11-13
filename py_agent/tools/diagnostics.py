"""
Diagnostic tools for identifying performance issues.

These tools analyze model performance to detect problems like
overfitting, data leakage, and distribution shift.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def diagnose_performance(
    model_fit: object,
    test_data: pd.DataFrame = None
) -> Dict:
    """
    Analyze model performance and identify issues.

    Performs comprehensive diagnostics including overfitting detection,
    data quality assessment, and model complexity analysis.

    Args:
        model_fit: Fitted workflow object
        test_data: Optional test data for overfitting detection

    Returns:
        Dictionary containing:
        - metrics: Performance metrics
        - issues_detected: List of identified issues
        - diagnostics: Detailed diagnostic information

    Example:
        >>> fit = workflow().add_model(linear_reg()).fit(train)
        >>> diag = diagnose_performance(fit, test)
        >>> len(diag['issues_detected']) > 0
        True  # If issues found
    """
    issues = []
    diagnostics = {}

    # Extract outputs
    try:
        outputs, coefficients, stats = model_fit.extract_outputs()
    except Exception as e:
        return {
            'metrics': {},
            'issues_detected': [{
                'type': 'extraction_error',
                'severity': 'high',
                'evidence': str(e),
                'recommendation': 'Check model fit status'
            }],
            'diagnostics': {}
        }

    # Calculate metrics
    metrics = _calculate_metrics_from_outputs(outputs, stats)

    # Check for overfitting if test data available
    if test_data is not None or 'test' in stats['split'].values:
        overfitting_issue = detect_overfitting(outputs, stats)
        if overfitting_issue:
            issues.append(overfitting_issue)

    # Analyze feature-to-sample ratio
    feature_issue = _check_feature_to_sample_ratio(outputs, coefficients)
    if feature_issue:
        issues.append(feature_issue)

    # Check for data quality issues
    quality_issues = _check_data_quality(outputs)
    issues.extend(quality_issues)

    # Analyze residuals
    residual_diagnostics = _analyze_residuals(outputs)
    diagnostics.update(residual_diagnostics)

    # Model complexity assessment
    complexity_info = _assess_model_complexity(coefficients)
    diagnostics['model_complexity'] = complexity_info

    return {
        'metrics': metrics,
        'issues_detected': issues,
        'diagnostics': diagnostics
    }


def detect_overfitting(
    outputs: pd.DataFrame,
    stats: pd.DataFrame
) -> Dict:
    """
    Detect overfitting by comparing train and test performance.

    Args:
        outputs: Outputs DataFrame from extract_outputs()
        stats: Stats DataFrame from extract_outputs()

    Returns:
        Issue dictionary if overfitting detected, None otherwise

    Example:
        >>> # Simulated overfitting scenario
        >>> outputs_df = pd.DataFrame({
        ...     'split': ['train']*50 + ['test']*50,
        ...     'actuals': range(100),
        ...     'fitted': range(100)  # Perfect fit on train
        ... })
        >>> stats_df = pd.DataFrame({
        ...     'split': ['train', 'test'],
        ...     'rmse': [0.1, 50.0]  # Big difference
        ... })
        >>> issue = detect_overfitting(outputs_df, stats_df)
        >>> issue['type']
        'overfitting'
    """
    # Get train and test metrics
    train_stats = stats[stats['split'] == 'train']
    test_stats = stats[stats['split'] == 'test']

    if len(train_stats) == 0 or len(test_stats) == 0:
        return None  # Can't detect without both splits

    # Compare RMSE
    if 'rmse' in stats.columns:
        train_rmse = train_stats['rmse'].mean()
        test_rmse = test_stats['rmse'].mean()

        # Check for significant gap
        if test_rmse > train_rmse * 1.5:  # 50% worse on test
            severity = 'high' if test_rmse > train_rmse * 2 else 'medium'

            return {
                'type': 'overfitting',
                'severity': severity,
                'evidence': f'train_rmse={train_rmse:.2f}, test_rmse={test_rmse:.2f}',
                'recommendation': (
                    'Model performs much worse on test data. Consider: '
                    '(1) Add regularization, '
                    '(2) Reduce model complexity, '
                    '(3) Feature selection, '
                    '(4) Use cross-validation'
                )
            }

    return None


# Helper functions

def _calculate_metrics_from_outputs(outputs: pd.DataFrame, stats: pd.DataFrame) -> Dict:
    """Extract key metrics from outputs and stats."""
    metrics = {}

    # From stats
    if 'rmse' in stats.columns:
        metrics['overall_rmse'] = float(stats['rmse'].mean())
    if 'mae' in stats.columns:
        metrics['overall_mae'] = float(stats['mae'].mean())
    if 'r_squared' in stats.columns:
        metrics['overall_r2'] = float(stats['r_squared'].mean())

    # From outputs
    if 'actuals' in outputs.columns and 'fitted' in outputs.columns:
        valid_mask = ~(outputs['actuals'].isna() | outputs['fitted'].isna())
        valid_outputs = outputs[valid_mask]

        if len(valid_outputs) > 0:
            residuals = valid_outputs['actuals'] - valid_outputs['fitted']
            metrics['residual_mean'] = float(residuals.mean())
            metrics['residual_std'] = float(residuals.std())

    return metrics


def _check_feature_to_sample_ratio(
    outputs: pd.DataFrame,
    coefficients: pd.DataFrame
) -> Dict:
    """Check if feature-to-sample ratio indicates overfitting risk."""
    n_samples = len(outputs)
    n_features = len(coefficients) if coefficients is not None else 0

    if n_features == 0:
        return None

    ratio = n_features / n_samples

    # Rule of thumb: want at least 10 samples per feature
    if ratio > 0.1:  # Less than 10 samples per feature
        severity = 'high' if ratio > 0.3 else 'medium'

        return {
            'type': 'high_dimensional',
            'severity': severity,
            'evidence': f'{n_features} features / {n_samples} samples = {ratio:.2f}',
            'recommendation': (
                f'Feature-to-sample ratio is high. Consider: '
                f'(1) Feature selection (reduce to ~{n_samples//10} features), '
                f'(2) Dimensionality reduction (PCA), '
                f'(3) Collect more data'
            )
        }

    return None


def _check_data_quality(outputs: pd.DataFrame) -> List[Dict]:
    """Check for data quality issues in outputs."""
    issues = []

    # Check for missing actuals
    if 'actuals' in outputs.columns:
        missing_rate = outputs['actuals'].isna().mean()
        if missing_rate > 0.1:
            issues.append({
                'type': 'missing_data',
                'severity': 'medium',
                'evidence': f'{missing_rate*100:.1f}% missing values in actuals',
                'recommendation': 'Review data collection process or imputation strategy'
            })

    # Check for constant predictions (model not learning)
    if 'fitted' in outputs.columns:
        fitted_std = outputs['fitted'].std()
        if fitted_std < 1e-6:
            issues.append({
                'type': 'constant_predictions',
                'severity': 'high',
                'evidence': 'Model produces constant predictions',
                'recommendation': 'Model may not be learning. Check: (1) Feature scaling, (2) Learning rate, (3) Data preprocessing'
            })

    return issues


def _analyze_residuals(outputs: pd.DataFrame) -> Dict:
    """Analyze residual patterns."""
    diagnostics = {}

    if 'residuals' in outputs.columns:
        residuals = outputs['residuals'].dropna()

        if len(residuals) > 0:
            # Basic statistics
            diagnostics['residual_mean'] = float(residuals.mean())
            diagnostics['residual_std'] = float(residuals.std())
            diagnostics['residual_skew'] = float(residuals.skew())

            # Check for systematic bias
            if abs(residuals.mean()) > residuals.std() * 0.2:
                diagnostics['systematic_bias'] = True
                diagnostics['bias_direction'] = 'positive' if residuals.mean() > 0 else 'negative'
            else:
                diagnostics['systematic_bias'] = False

    return diagnostics


def _assess_model_complexity(coefficients: pd.DataFrame) -> str:
    """Assess model complexity level."""
    if coefficients is None or len(coefficients) == 0:
        return 'unknown'

    n_params = len(coefficients)

    if n_params < 5:
        return 'low'
    elif n_params < 20:
        return 'medium'
    else:
        return 'high'
