"""
PyMC engine for Bayesian linear regression.

Implements MCMC-based Bayesian inference for linear models using PyMC.
Supports flexible prior specifications and comprehensive posterior analysis.
"""

from typing import Dict, Any, Literal, Tuple, Optional, Union
import pandas as pd
import numpy as np
import warnings

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_bayes.priors import parse_prior, get_default_priors


@register_engine("linear_reg", "pymc")
class PymcLinearRegEngine(Engine):
    """
    PyMC engine for Bayesian linear regression.

    Supports:
    - Flexible prior specifications via string DSL
    - MCMC sampling with multiple chains
    - Credible intervals
    - Posterior and posterior predictive sampling
    - Comprehensive diagnostics (Rhat, ESS, WAIC, LOO)
    """

    param_map = {
        "penalty": None,  # Not applicable for Bayesian
        "mixture": None,  # Not applicable for Bayesian
    }

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit Bayesian linear regression via PyMC.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors
            original_training_data: Original training DataFrame (optional)

        Returns:
            Dict containing:
                - model: PyMC model object
                - posterior_samples: Posterior samples (arviz.InferenceData)
                - summary: Posterior summary statistics
                - diagnostics: Convergence diagnostics
                - y_train: Training outcomes
                - fitted: Posterior mean predictions
                - residuals: Training residuals
                - X_train: Training predictors
                - feature_names: Predictor column names
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            raise ImportError(
                "PyMC and ArviZ are required for Bayesian models. "
                "Install with: pip install pymc>=5.10.0 arviz>=0.16.0"
            )

        # Extract data
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]

        # Remove Intercept column if present (PyMC will handle it separately)
        feature_names = list(X.columns)
        if "Intercept" in feature_names:
            X = X.drop(columns=["Intercept"])
            feature_names = [f for f in feature_names if f != "Intercept"]

        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values if isinstance(y, pd.Series) else y

        # Parse prior specifications
        priors = self._parse_priors(spec.args, feature_names)

        # Build PyMC model
        with pm.Model() as model:
            # Priors for intercept
            intercept = self._create_prior("Intercept", priors["intercept"])

            # Priors for coefficients
            if priors["per_coef_priors"]:
                # Different prior per coefficient
                beta_list = []
                for i, col in enumerate(feature_names):
                    beta_i = self._create_prior(f"beta_{col}", priors["coefs"][col])
                    beta_list.append(beta_i)
                beta = pm.math.stack(beta_list)
            else:
                # Shared prior for all coefficients
                beta = self._create_prior("beta", priors["coefs"], shape=X_array.shape[1])

            # Prior for error standard deviation
            sigma = self._create_prior("sigma", priors["sigma"])

            # Likelihood
            mu = intercept + pm.math.dot(X_array, beta)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_array)

            # Sample posterior
            draws = spec.args.get("draws", 2000)
            tune = spec.args.get("tune", 1000)
            chains = spec.args.get("chains", 4)
            target_accept = spec.args.get("target_accept", 0.95)
            random_seed = spec.args.get("random_seed", None)
            progressbar = spec.args.get("progressbar", True)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                return_inferencedata=True
            )

        # Compute fitted values (posterior mean)
        posterior_mean = trace.posterior.mean(dim=["chain", "draw"])
        intercept_mean = float(posterior_mean["Intercept"].values)

        if priors["per_coef_priors"]:
            # Extract individual beta means
            beta_mean = np.array([
                float(posterior_mean[f"beta_{col}"].values)
                for col in feature_names
            ])
        else:
            beta_mean = posterior_mean["beta"].values

        fitted = intercept_mean + X_array @ beta_mean
        residuals = y_array - fitted

        # Store results
        fit_data = {
            "model": model,
            "posterior_samples": trace,
            "summary": az.summary(trace),
            "diagnostics": {
                "rhat": az.rhat(trace),
                "ess": az.ess(trace),
                "divergences": int(trace.sample_stats.diverging.sum().item()) if hasattr(trace, 'sample_stats') else 0
            },
            "y_train": y_array,
            "fitted": fitted,
            "residuals": residuals,
            "X_train": X_array,
            "feature_names": feature_names,
            "original_training_data": original_training_data,
            "n_obs": len(y_array),
            "n_features": X_array.shape[1],
            "prior_coefs_dict": priors["per_coef_priors"]
        }

        return fit_data

    def _parse_priors(
        self,
        args: Dict[str, Any],
        feature_names: list
    ) -> Dict[str, Any]:
        """Parse prior specifications from model args."""
        defaults = get_default_priors()

        # Get prior strings (use defaults if not provided)
        prior_intercept_str = args.get("prior_intercept", defaults["prior_intercept"])
        prior_coefs_spec = args.get("prior_coefs", defaults["prior_coefs"])
        prior_sigma_str = args.get("prior_sigma", defaults["prior_sigma"])

        # Parse intercept prior
        intercept_prior = parse_prior(prior_intercept_str)

        # Parse coefficient priors (can be string or dict)
        per_coef_priors = isinstance(prior_coefs_spec, dict)

        if isinstance(prior_coefs_spec, str):
            # Shared prior for all coefficients
            coefs_prior = parse_prior(prior_coefs_spec)
        elif isinstance(prior_coefs_spec, dict):
            # Per-coefficient priors
            coefs_prior = {}
            for feature in feature_names:
                if feature in prior_coefs_spec:
                    coefs_prior[feature] = parse_prior(prior_coefs_spec[feature])
                else:
                    # Use default for missing features
                    coefs_prior[feature] = parse_prior(defaults["prior_coefs"])
        else:
            raise ValueError(
                f"prior_coefs must be string or dict, got {type(prior_coefs_spec)}"
            )

        # Parse sigma prior
        sigma_prior = parse_prior(prior_sigma_str)

        return {
            "intercept": intercept_prior,
            "coefs": coefs_prior,
            "sigma": sigma_prior,
            "per_coef_priors": per_coef_priors
        }

    def _create_prior(
        self,
        name: str,
        prior_spec: Dict[str, Any],
        shape: Optional[int] = None
    ):
        """Create PyMC prior distribution from parsed specification."""
        import pymc as pm

        dist_name = prior_spec["dist"]

        if dist_name == "normal":
            return pm.Normal(name, mu=prior_spec["mu"], sigma=prior_spec["sigma"], shape=shape)
        elif dist_name == "student_t":
            return pm.StudentT(
                name,
                nu=prior_spec["nu"],
                mu=prior_spec["mu"],
                sigma=prior_spec["sigma"],
                shape=shape
            )
        elif dist_name == "half_cauchy":
            return pm.HalfCauchy(name, beta=prior_spec["beta"], shape=shape)
        elif dist_name == "exponential":
            return pm.Exponential(name, lam=prior_spec["lam"], shape=shape)
        elif dist_name == "gamma":
            return pm.Gamma(name, alpha=prior_spec["alpha"], beta=prior_spec["beta"], shape=shape)
        elif dist_name == "beta":
            return pm.Beta(name, alpha=prior_spec["alpha"], beta=prior_spec["beta"], shape=shape)
        elif dist_name == "uniform":
            return pm.Uniform(name, lower=prior_spec["lower"], upper=prior_spec["upper"], shape=shape)
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "conf_int", "posterior", "predictive"],
    ) -> pd.DataFrame:
        """
        Generate predictions from Bayesian model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type
                - "numeric": Posterior mean
                - "conf_int": Credible intervals
                - "posterior": Posterior samples (no error term)
                - "predictive": Posterior predictive samples (with error)

        Returns:
            DataFrame with predictions
        """
        try:
            import pymc as pm
        except ImportError:
            raise ImportError("PyMC is required. Install with: pip install pymc>=5.10.0")

        # Extract test data
        X_test = molded.predictors

        # Remove Intercept column if present
        if "Intercept" in X_test.columns:
            X_test = X_test.drop(columns=["Intercept"])

        X_test_array = X_test.values

        # Extract posterior
        trace = fit.fit_data["posterior_samples"]
        feature_names = fit.fit_data["feature_names"]
        prior_coefs_dict = fit.fit_data["prior_coefs_dict"]

        # Posterior mean prediction
        if type == "numeric":
            posterior_mean = trace.posterior.mean(dim=["chain", "draw"])
            intercept = float(posterior_mean["Intercept"].values)

            if prior_coefs_dict:
                beta = np.array([
                    float(posterior_mean[f"beta_{col}"].values)
                    for col in feature_names
                ])
            else:
                beta = posterior_mean["beta"].values

            pred = intercept + X_test_array @ beta
            return pd.DataFrame({".pred": pred})

        # Credible intervals
        elif type == "conf_int":
            level = fit.spec.args.get("level", 0.95)
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q

            # Get posterior samples
            intercept_samples = trace.posterior["Intercept"].values.flatten()

            if prior_coefs_dict:
                # Stack individual beta samples
                beta_samples_list = []
                for col in feature_names:
                    beta_col_samples = trace.posterior[f"beta_{col}"].values.flatten()
                    beta_samples_list.append(beta_col_samples)
                beta_samples = np.stack(beta_samples_list, axis=1)
            else:
                beta_samples = trace.posterior["beta"].values
                # Reshape to (n_samples, n_features)
                beta_samples = beta_samples.reshape(-1, len(feature_names))

            # Compute predictions for each sample
            pred_samples = intercept_samples[:, None] + beta_samples @ X_test_array.T

            # Compute quantiles
            pred_mean = pred_samples.mean(axis=0)
            pred_lower = np.percentile(pred_samples, lower_q * 100, axis=0)
            pred_upper = np.percentile(pred_samples, upper_q * 100, axis=0)

            return pd.DataFrame({
                ".pred": pred_mean,
                ".pred_lower": pred_lower,
                ".pred_upper": pred_upper
            })

        # Posterior samples (no error term)
        elif type == "posterior":
            n_samples = fit.spec.args.get("n_samples", 1000)

            # Get posterior samples
            intercept_samples = trace.posterior["Intercept"].values.flatten()[:n_samples]

            if prior_coefs_dict:
                beta_samples_list = []
                for col in feature_names:
                    beta_col_samples = trace.posterior[f"beta_{col}"].values.flatten()[:n_samples]
                    beta_samples_list.append(beta_col_samples)
                beta_samples = np.stack(beta_samples_list, axis=1)
            else:
                beta_samples = trace.posterior["beta"].values
                beta_samples = beta_samples.reshape(-1, len(feature_names))[:n_samples]

            # Compute predictions
            pred_samples = intercept_samples[:, None] + beta_samples @ X_test_array.T

            # Return as wide DataFrame
            cols = {f".pred_sample_{i+1}": pred_samples[i] for i in range(n_samples)}
            return pd.DataFrame(cols)

        # Posterior predictive (with error term)
        elif type == "predictive":
            n_samples = fit.spec.args.get("n_samples", 500)

            # Sample from posterior
            intercept_samples = trace.posterior["Intercept"].values.flatten()[:n_samples]

            if prior_coefs_dict:
                beta_samples_list = []
                for col in feature_names:
                    beta_col_samples = trace.posterior[f"beta_{col}"].values.flatten()[:n_samples]
                    beta_samples_list.append(beta_col_samples)
                beta_samples = np.stack(beta_samples_list, axis=1)
            else:
                beta_samples = trace.posterior["beta"].values
                beta_samples = beta_samples.reshape(-1, len(feature_names))[:n_samples]

            # Sample sigma (error standard deviation)
            sigma_samples = trace.posterior["sigma"].values.flatten()[:n_samples]

            # Compute predictions + error
            mu_samples = intercept_samples[:, None] + beta_samples @ X_test_array.T

            # Add error term
            np.random.seed(42)
            pred_samples = mu_samples + np.random.normal(0, 1, mu_samples.shape) * sigma_samples[:, None]

            cols = {f".pred_sample_{i+1}": pred_samples[i] for i in range(n_samples)}
            return pd.DataFrame(cols)

        else:
            raise ValueError(
                f"Unsupported prediction type: {type}. "
                f"Supported: numeric, conf_int, posterior, predictive"
            )

    def extract_outputs(
        self, fit: ModelFit
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output for Bayesian models.

        Returns:
            Tuple of (outputs, coefficients, stats)
        """
        try:
            import arviz as az
        except ImportError:
            raise ImportError("ArviZ is required. Install with: pip install arviz>=0.16.0")

        trace = fit.fit_data["posterior_samples"]
        summary = fit.fit_data["summary"]
        feature_names = fit.fit_data["feature_names"]
        prior_coefs_dict = fit.fit_data["prior_coefs_dict"]

        # ===============
        # OUTPUTS
        # ===============
        X_train = fit.fit_data["X_train"]
        y_train = fit.fit_data["y_train"]
        fitted = fit.fit_data["fitted"]
        residuals = fit.fit_data["residuals"]

        # Compute credible intervals for fitted values
        intercept_samples = trace.posterior["Intercept"].values.flatten()

        if prior_coefs_dict:
            beta_samples_list = []
            for col in feature_names:
                beta_col_samples = trace.posterior[f"beta_{col}"].values.flatten()
                beta_samples_list.append(beta_col_samples)
            beta_samples = np.stack(beta_samples_list, axis=1)
        else:
            beta_samples = trace.posterior["beta"].values
            beta_samples = beta_samples.reshape(-1, len(feature_names))

        fitted_samples = intercept_samples[:, None] + beta_samples @ X_train.T
        fitted_lower = np.percentile(fitted_samples, 2.5, axis=0)
        fitted_upper = np.percentile(fitted_samples, 97.5, axis=0)

        # Create forecast: actuals where available, fitted otherwise
        forecast_train = pd.Series(y_train).combine_first(pd.Series(fitted)).values

        outputs = pd.DataFrame({
            "actuals": y_train,
            "fitted": fitted,
            "fitted_lower": fitted_lower,
            "fitted_upper": fitted_upper,
            "forecast": forecast_train,
            "residuals": residuals,
            "split": "train",
            "model": fit.spec.model_type,
            "model_group_name": fit.spec.engine,
            "group": None
        })

        # Add test data if available
        if "evaluation_data" in fit.__dict__ and "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Credible intervals for test predictions (if conf_int was used)
            if ".pred_lower" in test_preds.columns:
                test_lower = test_preds[".pred_lower"].values
                test_upper = test_preds[".pred_upper"].values
            else:
                test_lower = np.full_like(test_predictions, np.nan)
                test_upper = np.full_like(test_predictions, np.nan)

            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "fitted_lower": test_lower,
                "fitted_upper": test_upper,
                "forecast": forecast_test,
                "residuals": test_residuals,
                "split": "test",
                "model": fit.spec.model_type,
                "model_group_name": fit.spec.engine,
                "group": None
            })

            outputs = pd.concat([outputs, test_df], ignore_index=True)

        # ===============
        # COEFFICIENTS
        # ===============
        coef_rows = []

        # Intercept
        intercept_summary = summary.loc["Intercept"]
        intercept_samples_full = trace.posterior["Intercept"].values.flatten()

        coef_rows.append({
            "term": "Intercept",
            "estimate": intercept_summary["mean"],
            "std_error": intercept_summary["sd"],
            "lower_ci": intercept_summary["hdi_2.5%"] if "hdi_2.5%" in intercept_summary.index else np.percentile(intercept_samples_full, 2.5),
            "upper_ci": intercept_summary["hdi_97.5%"] if "hdi_97.5%" in intercept_summary.index else np.percentile(intercept_samples_full, 97.5),
            "rhat": intercept_summary["r_hat"],
            "ess_bulk": intercept_summary["ess_bulk"],
            "ess_tail": intercept_summary["ess_tail"],
            "prob_positive": float((intercept_samples_full > 0).mean()),
            "model": fit.spec.model_type,
            "model_group_name": fit.spec.engine,
            "group": None
        })

        # Coefficients
        if prior_coefs_dict:
            # Individual beta parameters
            for col in feature_names:
                beta_name = f"beta_{col}"
                beta_summary = summary.loc[beta_name]
                beta_samples_full = trace.posterior[beta_name].values.flatten()

                coef_rows.append({
                    "term": col,
                    "estimate": beta_summary["mean"],
                    "std_error": beta_summary["sd"],
                    "lower_ci": beta_summary["hdi_2.5%"] if "hdi_2.5%" in beta_summary.index else np.percentile(beta_samples_full, 2.5),
                    "upper_ci": beta_summary["hdi_97.5%"] if "hdi_97.5%" in beta_summary.index else np.percentile(beta_samples_full, 97.5),
                    "rhat": beta_summary["r_hat"],
                    "ess_bulk": beta_summary["ess_bulk"],
                    "ess_tail": beta_summary["ess_tail"],
                    "prob_positive": float((beta_samples_full > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })
        else:
            # Vectorized beta parameter
            for i, col in enumerate(feature_names):
                beta_name = f"beta[{i}]"
                beta_summary = summary.loc[beta_name]
                beta_i_samples = trace.posterior["beta"].values[:, :, i].flatten()

                coef_rows.append({
                    "term": col,
                    "estimate": beta_summary["mean"],
                    "std_error": beta_summary["sd"],
                    "lower_ci": beta_summary["hdi_2.5%"] if "hdi_2.5%" in beta_summary.index else np.percentile(beta_i_samples, 2.5),
                    "upper_ci": beta_summary["hdi_97.5%"] if "hdi_97.5%" in beta_summary.index else np.percentile(beta_i_samples, 97.5),
                    "rhat": beta_summary["r_hat"],
                    "ess_bulk": beta_summary["ess_bulk"],
                    "ess_tail": beta_summary["ess_tail"],
                    "prob_positive": float((beta_i_samples > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })

        coefficients = pd.DataFrame(coef_rows)

        # ===============
        # STATS
        # ===============
        # Compute WAIC and LOO
        try:
            waic = az.waic(trace, scale="deviance")
            loo = az.loo(trace, scale="deviance")
        except Exception as e:
            warnings.warn(f"Could not compute WAIC/LOO: {e}")
            waic = None
            loo = None

        # Training metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        r_squared = 1 - np.var(residuals) / np.var(y_train)

        # Extract diagnostics safely
        rhat_data = fit.fit_data["diagnostics"]["rhat"]
        ess_data = fit.fit_data["diagnostics"]["ess"]

        # Convert to float properly
        # rhat and ess are xarray Datasets - convert to numpy first
        max_rhat = float(np.max(rhat_data.to_array().values))

        if hasattr(ess_data, 'dims'):
            ess_array = ess_data.to_array().values
            if hasattr(ess_data, 'data_vars') and len(list(ess_data.data_vars)) > 1:
                # Separate bulk and tail ESS
                min_ess_bulk = float(np.min(ess_array[0]))  # First variable
                min_ess_tail = float(np.min(ess_array[1]))  # Second variable
            else:
                min_ess_bulk = float(np.min(ess_array))
                min_ess_tail = min_ess_bulk
        else:
            min_ess_bulk = float(np.min(ess_data))
            min_ess_tail = min_ess_bulk

        stats_dict = {
            "split": "train",
            "n_obs": len(y_train),
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "n_divergences": fit.fit_data["diagnostics"]["divergences"],
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "model": fit.spec.model_type,
            "model_group_name": fit.spec.engine,
            "group": None
        }

        # Add WAIC/LOO if available
        if waic is not None:
            stats_dict["waic"] = waic.elpd_waic * -2  # Convert to deviance scale
            stats_dict["waic_se"] = waic.se * 2

        if loo is not None:
            stats_dict["loo"] = loo.elpd_loo * -2  # Convert to deviance scale
            stats_dict["loo_se"] = loo.se * 2
            stats_dict["loo_p"] = loo.p_loo

        stats = pd.DataFrame([stats_dict])

        return outputs, coefficients, stats
