"""
Prior Sensitivity Analysis for Bayesian Models

Compare how different prior specifications affect posterior inference and predictions.
"""

from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np
import warnings

from py_parsnip import ModelSpec
from py_yardstick import metric_set, rmse, mae


def compare_priors(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    formula: str,
    priors: Dict[str, Dict[str, str]],
    test_data: Optional[pd.DataFrame] = None,
    metrics: Optional[Any] = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple prior specifications on the same model and data.

    Fits the same Bayesian model with different prior specifications and
    compares:
    - Posterior means and SDs
    - Prediction performance (if test_data provided)
    - Convergence diagnostics

    Args:
        model_spec: Base ModelSpec (e.g., linear_reg().set_engine("pymc"))
        data: Training data
        formula: Model formula
        priors: Dict mapping prior names to prior specifications
            Example:
                {
                    "weak": {"prior_coefs": "normal(0, 10)"},
                    "medium": {"prior_coefs": "normal(0, 5)"},
                    "strong": {"prior_coefs": "normal(0, 1)"}
                }
        test_data: Optional test data for prediction metrics
        metrics: Optional metric_set for evaluation (default: rmse + mae)
        draws: MCMC draws per chain (default: 1000)
        tune: MCMC tuning steps (default: 500)
        chains: Number of MCMC chains (default: 2)
        verbose: Print progress (default: True)

    Returns:
        DataFrame with one row per prior specification, showing:
        - prior_name: Name of prior specification
        - Metrics (if test_data provided): rmse, mae, etc.
        - Posterior summaries: posterior_mean_*, posterior_sd_*
        - Diagnostics: max_rhat, min_ess_bulk, n_divergences

    Examples:
        >>> from py_parsnip import linear_reg
        >>> from py_bayes.analysis import compare_priors
        >>> from py_yardstick import metric_set, rmse, mae, r_squared
        >>>
        >>> # Define prior specifications to compare
        >>> priors = {
        ...     "weak": {"prior_coefs": "normal(0, 10)", "prior_sigma": "half_cauchy(10)"},
        ...     "medium": {"prior_coefs": "normal(0, 5)", "prior_sigma": "half_cauchy(5)"},
        ...     "strong": {"prior_coefs": "normal(0, 1)", "prior_sigma": "half_cauchy(1)"}
        ... }
        >>>
        >>> # Compare priors
        >>> spec = linear_reg().set_engine("pymc")
        >>> results = compare_priors(
        ...     model_spec=spec,
        ...     data=train_data,
        ...     formula="y ~ x1 + x2",
        ...     priors=priors,
        ...     test_data=test_data,
        ...     metrics=metric_set(rmse, mae, r_squared)
        ... )
        >>>
        >>> # Analyze results
        >>> print(results[['prior_name', 'rmse', 'mae', 'max_rhat']])
    """
    if metrics is None:
        metrics = metric_set(rmse, mae)

    results = []

    for prior_name, prior_spec in priors.items():
        if verbose:
            print(f"\nFitting with prior: {prior_name}")
            print(f"  Specification: {prior_spec}")

        # Create model with these priors
        # Update model args with prior specification
        updated_args = {**model_spec.args}
        updated_args.update(prior_spec)

        # Override MCMC settings for faster comparison
        updated_args["draws"] = draws
        updated_args["tune"] = tune
        updated_args["chains"] = chains
        updated_args["progressbar"] = verbose

        # Create new spec with updated args
        updated_spec = ModelSpec(
            model_type=model_spec.model_type,
            engine=model_spec.engine,
            mode=model_spec.mode,
            args=updated_args
        )

        # Fit model
        try:
            fit = updated_spec.fit(data, formula)

            # Extract outputs
            outputs, coefficients, stats = fit.extract_outputs()

            # Extract posterior summaries for each coefficient
            result_dict = {"prior_name": prior_name}

            # Add posterior means and SDs
            for _, row in coefficients.iterrows():
                term = row["term"]
                result_dict[f"posterior_mean_{term}"] = row["estimate"]
                result_dict[f"posterior_sd_{term}"] = row["std_error"]

            # Add diagnostics from stats
            if len(stats) > 0:
                stats_row = stats.iloc[0]
                result_dict["max_rhat"] = stats_row.get("max_rhat", np.nan)
                result_dict["min_ess_bulk"] = stats_row.get("min_ess_bulk", np.nan)
                result_dict["min_ess_tail"] = stats_row.get("min_ess_tail", np.nan)
                result_dict["n_divergences"] = stats_row.get("n_divergences", 0)
                result_dict["train_rmse"] = stats_row.get("rmse", np.nan)
                result_dict["train_mae"] = stats_row.get("mae", np.nan)
                result_dict["train_r_squared"] = stats_row.get("r_squared", np.nan)

            # Add test metrics if test_data provided
            if test_data is not None:
                if verbose:
                    print(f"  Evaluating on test data...")

                try:
                    fit = fit.evaluate(test_data)

                    # Extract test metrics
                    test_outputs, _, test_stats = fit.extract_outputs()
                    test_rows = test_outputs[test_outputs["split"] == "test"]

                    if len(test_rows) > 0:
                        # Compute metrics using metric_set
                        test_actuals = test_rows["actuals"].values
                        test_predictions = test_rows["fitted"].values

                        # Create DataFrame for metrics
                        metric_df = pd.DataFrame({
                            "truth": test_actuals,
                            "estimate": test_predictions
                        })

                        # Compute metrics
                        metric_results = metrics(metric_df, "truth", "estimate")

                        # Add to result dict
                        for _, metric_row in metric_results.iterrows():
                            metric_name = metric_row[".metric"]
                            metric_value = metric_row["value"]
                            result_dict[f"test_{metric_name}"] = metric_value

                except Exception as e:
                    warnings.warn(f"Could not evaluate test data for prior '{prior_name}': {e}")

            results.append(result_dict)

            if verbose:
                print(f"  ✓ Complete")

        except Exception as e:
            warnings.warn(f"Failed to fit model with prior '{prior_name}': {e}")
            if verbose:
                print(f"  ✗ Failed: {e}")
            continue

    if len(results) == 0:
        raise ValueError("All prior specifications failed to fit")

    return pd.DataFrame(results)


def compare_prior_predictive(
    model_spec: ModelSpec,
    data: pd.DataFrame,
    formula: str,
    priors: Dict[str, Dict[str, str]],
    n_samples: int = 1000,
    draws: int = 500,
    tune: int = 250,
    chains: int = 2,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compare prior predictive distributions for different prior specifications.

    Samples from the prior predictive distribution (before seeing data) to
    understand what each prior implies about the outcome distribution.

    Args:
        model_spec: Base ModelSpec
        data: Data (used only for formula parsing and covariate structure)
        formula: Model formula
        priors: Dict mapping prior names to specifications
        n_samples: Number of prior predictive samples per observation
        draws: MCMC draws for prior sampling
        tune: MCMC tuning steps
        chains: Number of chains
        verbose: Print progress

    Returns:
        Dict mapping prior_name to DataFrame of prior predictive samples
        Each DataFrame has shape (n_obs, n_samples) with prior predictions

    Note:
        This is computationally expensive as it requires fitting each model
        to sample from the prior before seeing the data.
    """
    # This would require PyMC's prior predictive sampling
    # Implementation deferred as it requires more complex PyMC API usage
    raise NotImplementedError(
        "Prior predictive comparison not yet implemented. "
        "Use compare_priors() for posterior comparison."
    )
