"""
Workflow execution tools for running and evaluating forecasting workflows.

These tools execute py-tidymodels workflows and extract results
in a format suitable for analysis and presentation.
"""

import pandas as pd
import time
from typing import Dict, Optional
import traceback


def fit_workflow(
    workflow_code: str,
    data: pd.DataFrame,
    formula: str,
    group_col: Optional[str] = None
) -> Dict:
    """
    Execute workflow and return results.

    Runs the provided workflow code in a safe namespace and
    captures execution time, success status, and results.

    Args:
        workflow_code: Python code string creating workflow
        data: DataFrame for training
        formula: Model formula (e.g., 'y ~ x1 + x2')
        group_col: Optional grouping column for nested modeling

    Returns:
        Dictionary containing:
        - success: Whether execution succeeded
        - fit_time: Training time in seconds
        - model_fit: Fitted workflow object (if successful)
        - error: Error message (if failed)
        - metrics: Basic performance metrics

    Example:
        >>> code = '''
        ... from py_workflows import workflow
        ... from py_parsnip import linear_reg
        ... wf = workflow().add_model(linear_reg())
        ... '''
        >>> result = fit_workflow(code, train_data, 'y ~ x')
        >>> result['success']
        True
    """
    start_time = time.time()

    # Create execution namespace with required imports
    namespace = {
        'pd': pd,
        'data': data,
        'formula': formula,
        'group_col': group_col
    }

    try:
        # Execute workflow creation code
        exec(workflow_code, namespace)

        # Workflow should be assigned to 'wf' variable
        if 'wf' not in namespace:
            return {
                'success': False,
                'error': "Workflow code must assign to variable 'wf'",
                'fit_time': 0.0
            }

        workflow = namespace['wf']

        # Fit workflow
        if group_col:
            fit = workflow.fit_nested(data, group_col=group_col)
        else:
            fit = workflow.fit(data)

        fit_time = time.time() - start_time

        # Extract basic metrics if possible
        try:
            outputs, coefficients, stats = fit.extract_outputs()
            metrics = _extract_summary_metrics(stats)
        except:
            metrics = {}

        return {
            'success': True,
            'fit_time': fit_time,
            'model_fit': fit,
            'metrics': metrics,
            'error': None
        }

    except Exception as e:
        fit_time = time.time() - start_time
        return {
            'success': False,
            'fit_time': fit_time,
            'model_fit': None,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def evaluate_workflow(
    model_fit: object,
    test_data: pd.DataFrame
) -> Dict:
    """
    Evaluate fitted workflow on test data.

    Args:
        model_fit: Fitted workflow object
        test_data: DataFrame for testing

    Returns:
        Dictionary containing:
        - predictions: DataFrame with predictions
        - metrics: Performance metrics
        - success: Whether evaluation succeeded
        - error: Error message (if failed)

    Example:
        >>> fit = workflow().add_model(linear_reg()).fit(train)
        >>> result = evaluate_workflow(fit, test)
        >>> 'rmse' in result['metrics']
        True
    """
    try:
        # Make predictions
        predictions = model_fit.predict(test_data)

        # Evaluate if possible
        try:
            eval_result = model_fit.evaluate(test_data)
            outputs, coefficients, stats = eval_result.extract_outputs()
            metrics = _extract_summary_metrics(stats)
        except:
            # If evaluate not available, just return predictions
            metrics = {}

        return {
            'success': True,
            'predictions': predictions,
            'metrics': metrics,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'predictions': None,
            'metrics': {},
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# Helper functions

def _extract_summary_metrics(stats: pd.DataFrame) -> Dict:
    """
    Extract summary metrics from stats DataFrame.

    Args:
        stats: Stats DataFrame from extract_outputs()

    Returns:
        Dictionary of key metrics
    """
    if stats is None or len(stats) == 0:
        return {}

    metrics = {}

    # Extract training metrics
    train_stats = stats[stats['split'] == 'train']
    if len(train_stats) > 0:
        if 'rmse' in train_stats.columns:
            metrics['train_rmse'] = float(train_stats['rmse'].mean())
        if 'mae' in train_stats.columns:
            metrics['train_mae'] = float(train_stats['mae'].mean())
        if 'r_squared' in train_stats.columns:
            metrics['train_r2'] = float(train_stats['r_squared'].mean())

    # Extract test metrics if available
    test_stats = stats[stats['split'] == 'test']
    if len(test_stats) > 0:
        if 'rmse' in test_stats.columns:
            metrics['test_rmse'] = float(test_stats['rmse'].mean())
        if 'mae' in test_stats.columns:
            metrics['test_mae'] = float(test_stats['mae'].mean())
        if 'r_squared' in test_stats.columns:
            metrics['test_r2'] = float(test_stats['r_squared'].mean())

    return metrics
