"""
Performance monitoring utilities for deployed models.

This module provides functions for monitoring model performance in production,
including prediction logging, data drift detection, and performance tracking.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError(
        "MLflow is required for monitoring utilities. "
        "Install with: pip install mlflow"
    )

try:
    from scipy import stats
except ImportError:
    warnings.warn(
        "scipy not installed. Data drift detection will use simplified methods. "
        "Install with: pip install scipy"
    )
    stats = None


def log_prediction_batch(
    model_name: str,
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Log batch predictions for monitoring.

    This function logs predictions to MLflow for performance tracking,
    debugging, and audit purposes.

    Args:
        model_name: Name of the model
        predictions: DataFrame with predictions (must have '.pred' column)
        actuals: Optional DataFrame with actual values for comparison
        features: Optional DataFrame with input features
        metadata: Optional dict with batch metadata (e.g., batch_id, timestamp)
        run_name: Optional run name (defaults to timestamp)

    Returns:
        Run ID of logged predictions

    Examples:
        >>> from py_mlflow import log_prediction_batch
        >>>
        >>> # Log predictions with actuals
        >>> run_id = log_prediction_batch(
        ...     model_name="SalesForecast",
        ...     predictions=pred_df,
        ...     actuals=actual_df,
        ...     metadata={"batch_id": "2024-01-15", "source": "production"}
        ... )
    """
    # Set experiment
    experiment_name = f"{model_name}_monitoring"
    mlflow.set_experiment(experiment_name)

    # Generate run name
    if run_name is None:
        run_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log metadata
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_predictions", len(predictions))
        mlflow.log_param("timestamp", datetime.now().isoformat())

        if metadata:
            for key, value in metadata.items():
                mlflow.log_param(f"meta_{key}", value)

        # Log prediction statistics
        if '.pred' in predictions.columns:
            pred_values = predictions['.pred'].values
            mlflow.log_metric("pred_mean", float(np.mean(pred_values)))
            mlflow.log_metric("pred_std", float(np.std(pred_values)))
            mlflow.log_metric("pred_min", float(np.min(pred_values)))
            mlflow.log_metric("pred_max", float(np.max(pred_values)))
            mlflow.log_metric("pred_nan_count", int(np.isnan(pred_values).sum()))

        # Log actuals and compute metrics if provided
        if actuals is not None:
            from py_yardstick import rmse, mae, r_squared

            # Extract actual values
            if isinstance(actuals, pd.DataFrame):
                if '.pred' in actuals.columns:
                    actual_values = actuals['.pred'].values
                else:
                    actual_values = actuals.iloc[:, 0].values
            else:
                actual_values = actuals

            # Align predictions and actuals
            min_len = min(len(pred_values), len(actual_values))
            pred_aligned = pred_values[:min_len]
            actual_aligned = actual_values[:min_len]

            # Compute metrics
            try:
                rmse_value = rmse(actual_aligned, pred_aligned).iloc[0]['value']
                mae_value = mae(actual_aligned, pred_aligned).iloc[0]['value']
                r2_value = r_squared(actual_aligned, pred_aligned).iloc[0]['value']

                mlflow.log_metric("rmse", float(rmse_value))
                mlflow.log_metric("mae", float(mae_value))
                mlflow.log_metric("r_squared", float(r2_value))
            except Exception as e:
                warnings.warn(f"Failed to compute metrics: {str(e)}")

        # Log feature statistics if provided
        if features is not None:
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    mlflow.log_metric(f"feature_{col}_mean", float(features[col].mean()))
                    mlflow.log_metric(f"feature_{col}_std", float(features[col].std()))

        # Log artifacts
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save predictions
            pred_path = f"{tmpdir}/predictions.csv"
            predictions.to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path, "predictions")

            # Save actuals if provided
            if actuals is not None:
                actual_path = f"{tmpdir}/actuals.csv"
                if isinstance(actuals, pd.DataFrame):
                    actuals.to_csv(actual_path, index=False)
                else:
                    pd.DataFrame({'actual': actuals}).to_csv(actual_path, index=False)
                mlflow.log_artifact(actual_path, "actuals")

            # Save features if provided
            if features is not None:
                features_path = f"{tmpdir}/features.csv"
                features.to_csv(features_path, index=False)
                mlflow.log_artifact(features_path, "features")

        return run.info.run_id


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
    method: str = "ks"
) -> Dict[str, Any]:
    """
    Detect data drift using statistical tests.

    Compares distributions of current data against reference (training) data
    to detect potential drift that might affect model performance.

    Args:
        reference_data: Reference dataset (e.g., training data)
        current_data: Current dataset (e.g., production data)
        threshold: P-value threshold for drift detection (default: 0.05)
        method: Statistical test method - "ks" (Kolmogorov-Smirnov) or
                "simple" (mean/std comparison)

    Returns:
        Dict with drift detection results:
        - drifted_features: List of features with detected drift
        - drift_scores: Dict mapping features to drift scores
        - summary: Summary statistics

    Examples:
        >>> from py_mlflow import detect_data_drift
        >>>
        >>> # Detect drift
        >>> drift_results = detect_data_drift(
        ...     reference_data=train_data,
        ...     current_data=production_data,
        ...     threshold=0.05
        ... )
        >>>
        >>> if drift_results['drifted_features']:
        ...     print(f"Warning: Drift detected in {drift_results['drifted_features']}")
        ...     # Trigger retraining or alerting
    """
    results = {
        "drifted_features": [],
        "drift_scores": {},
        "summary": {
            "n_features_checked": 0,
            "n_features_drifted": 0,
            "method": method,
            "threshold": threshold
        }
    }

    # Get common numeric columns
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    numeric_cols = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(reference_data[col]) and
           pd.api.types.is_numeric_dtype(current_data[col])
    ]

    if not numeric_cols:
        warnings.warn("No common numeric columns found for drift detection")
        return results

    results["summary"]["n_features_checked"] = len(numeric_cols)

    for col in numeric_cols:
        ref_values = reference_data[col].dropna().values
        curr_values = current_data[col].dropna().values

        if len(ref_values) == 0 or len(curr_values) == 0:
            continue

        if method == "ks" and stats is not None:
            # Kolmogorov-Smirnov test
            statistic, pvalue = stats.ks_2samp(ref_values, curr_values)
            drift_score = pvalue
            drifted = pvalue < threshold

        else:
            # Simple method: compare means and standard deviations
            ref_mean = np.mean(ref_values)
            ref_std = np.std(ref_values)
            curr_mean = np.mean(curr_values)
            curr_std = np.std(curr_values)

            # Compute normalized differences
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-10)
            std_diff = abs(curr_std - ref_std) / (ref_std + 1e-10)

            # Combine differences (simple heuristic)
            drift_score = max(mean_diff, std_diff)
            drifted = drift_score > (1 / threshold)  # Invert threshold logic

        results["drift_scores"][col] = {
            "score": float(drift_score),
            "drifted": bool(drifted),
            "ref_mean": float(np.mean(ref_values)),
            "curr_mean": float(np.mean(curr_values)),
            "ref_std": float(np.std(ref_values)),
            "curr_std": float(np.std(curr_values))
        }

        if drifted:
            results["drifted_features"].append(col)

    results["summary"]["n_features_drifted"] = len(results["drifted_features"])

    return results


def monitor_model_performance(
    model_name: str,
    window: str = "7d",
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get model performance metrics over time window.

    Retrieves logged performance metrics from MLflow for a specified
    model and time window, useful for monitoring performance degradation.

    Args:
        model_name: Name of the model
        window: Time window - format: "7d", "1w", "30d", "1m" (default: "7d")
        metrics: List of metric names to retrieve (default: ["rmse", "mae", "r_squared"])

    Returns:
        DataFrame with performance metrics over time including:
        - timestamp: When metrics were logged
        - run_id: MLflow run ID
        - metric columns: One column per requested metric

    Examples:
        >>> from py_mlflow import monitor_model_performance
        >>>
        >>> # Get last 7 days of performance
        >>> perf_df = monitor_model_performance(
        ...     "SalesForecast",
        ...     window="7d",
        ...     metrics=["rmse", "mae"]
        ... )
        >>>
        >>> # Check for performance degradation
        >>> recent_rmse = perf_df.tail(5)['rmse'].mean()
        >>> baseline_rmse = perf_df.head(5)['rmse'].mean()
        >>> if recent_rmse > baseline_rmse * 1.1:
        ...     print("Warning: RMSE degraded by >10%")
    """
    if metrics is None:
        metrics = ["rmse", "mae", "r_squared"]

    # Parse time window
    window_map = {
        'd': 'days',
        'w': 'weeks',
        'm': 'months'
    }

    window_value = int(window[:-1])
    window_unit = window[-1]

    if window_unit not in window_map:
        raise ValueError(f"Invalid window unit '{window_unit}'. Use 'd' (days), 'w' (weeks), or 'm' (months)")

    # Calculate start time
    end_time = datetime.now()
    if window_unit == 'd':
        start_time = end_time - timedelta(days=window_value)
    elif window_unit == 'w':
        start_time = end_time - timedelta(weeks=window_value)
    elif window_unit == 'm':
        start_time = end_time - timedelta(days=window_value * 30)  # Approximate

    # Get MLflow client
    client = MlflowClient()

    # Get experiment
    experiment_name = f"{model_name}_monitoring"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        warnings.warn(f"No monitoring experiment found for model '{model_name}'")
        return pd.DataFrame()

    # Search runs in time window
    filter_string = f"attributes.start_time >= {int(start_time.timestamp() * 1000)}"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"]
    )

    if runs.empty:
        warnings.warn(f"No runs found for model '{model_name}' in window '{window}'")
        return pd.DataFrame()

    # Extract metrics
    perf_data = []

    for _, run in runs.iterrows():
        run_data = {
            "timestamp": pd.to_datetime(run["start_time"]),
            "run_id": run["run_id"]
        }

        # Extract requested metrics
        for metric in metrics:
            metric_col = f"metrics.{metric}"
            if metric_col in run.index:
                run_data[metric] = run[metric_col]
            else:
                run_data[metric] = None

        perf_data.append(run_data)

    perf_df = pd.DataFrame(perf_data)

    # Sort by timestamp
    perf_df = perf_df.sort_values("timestamp")

    return perf_df


def create_monitoring_dashboard_data(
    model_name: str,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Create comprehensive monitoring dashboard data.

    Aggregates drift detection, performance metrics, and prediction statistics
    into a single data structure suitable for dashboard visualization.

    Args:
        model_name: Name of the model
        reference_data: Reference/training dataset
        current_data: Current production dataset
        predictions: Recent predictions
        actuals: Recent actual values (optional)

    Returns:
        Dict with dashboard data:
        - model_info: Model metadata
        - drift_status: Drift detection results
        - performance: Performance metrics
        - prediction_stats: Prediction statistics
        - alerts: List of alert messages

    Examples:
        >>> from py_mlflow import create_monitoring_dashboard_data
        >>>
        >>> dashboard_data = create_monitoring_dashboard_data(
        ...     model_name="SalesForecast",
        ...     reference_data=train_data,
        ...     current_data=recent_prod_data,
        ...     predictions=recent_preds,
        ...     actuals=recent_actuals
        ... )
        >>>
        >>> # Use dashboard_data to populate monitoring UI
        >>> if dashboard_data['alerts']:
        ...     for alert in dashboard_data['alerts']:
        ...         send_alert(alert)
    """
    dashboard_data = {
        "model_info": {
            "name": model_name,
            "timestamp": datetime.now().isoformat()
        },
        "drift_status": {},
        "performance": {},
        "prediction_stats": {},
        "alerts": []
    }

    # Drift detection
    try:
        drift_results = detect_data_drift(reference_data, current_data)
        dashboard_data["drift_status"] = drift_results

        if drift_results["drifted_features"]:
            dashboard_data["alerts"].append({
                "type": "drift",
                "severity": "warning",
                "message": f"Data drift detected in {len(drift_results['drifted_features'])} features",
                "features": drift_results["drifted_features"]
            })
    except Exception as e:
        dashboard_data["drift_status"]["error"] = str(e)
        dashboard_data["alerts"].append({
            "type": "error",
            "severity": "error",
            "message": f"Drift detection failed: {str(e)}"
        })

    # Prediction statistics
    if '.pred' in predictions.columns:
        pred_values = predictions['.pred'].values
        dashboard_data["prediction_stats"] = {
            "count": int(len(pred_values)),
            "mean": float(np.mean(pred_values)),
            "std": float(np.std(pred_values)),
            "min": float(np.min(pred_values)),
            "max": float(np.max(pred_values)),
            "nan_count": int(np.isnan(pred_values).sum())
        }

        # Check for anomalies
        if np.isnan(pred_values).any():
            nan_pct = np.isnan(pred_values).sum() / len(pred_values) * 100
            dashboard_data["alerts"].append({
                "type": "prediction_quality",
                "severity": "error",
                "message": f"{nan_pct:.1f}% of predictions are NaN"
            })

    # Performance metrics
    if actuals is not None:
        try:
            from py_yardstick import rmse, mae, r_squared

            # Extract values
            if isinstance(actuals, pd.DataFrame):
                if '.pred' in actuals.columns:
                    actual_values = actuals['.pred'].values
                else:
                    actual_values = actuals.iloc[:, 0].values
            else:
                actual_values = actuals

            # Align
            min_len = min(len(pred_values), len(actual_values))
            pred_aligned = pred_values[:min_len]
            actual_aligned = actual_values[:min_len]

            # Compute metrics
            rmse_value = rmse(actual_aligned, pred_aligned).iloc[0]['value']
            mae_value = mae(actual_aligned, pred_aligned).iloc[0]['value']
            r2_value = r_squared(actual_aligned, pred_aligned).iloc[0]['value']

            dashboard_data["performance"] = {
                "rmse": float(rmse_value),
                "mae": float(mae_value),
                "r_squared": float(r2_value)
            }

            # Check for performance degradation (simple heuristic)
            if r2_value < 0.5:
                dashboard_data["alerts"].append({
                    "type": "performance",
                    "severity": "warning",
                    "message": f"Low R² value: {r2_value:.3f}"
                })

        except Exception as e:
            dashboard_data["performance"]["error"] = str(e)

    return dashboard_data
