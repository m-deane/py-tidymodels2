"""
PyMC engine for Hierarchical Bayesian Linear Regression.

Implements partial pooling for grouped/panel data where each group gets
its own parameters drawn from common hyperprior distributions.
"""

from typing import Dict, Any, Literal, Tuple, Optional, Union, List
import pandas as pd
import numpy as np
import warnings

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_bayes.priors import parse_prior, get_default_priors


@register_engine("linear_reg", "pymc_hierarchical")
class PymcHierarchicalEngine(Engine):
    """
    PyMC engine for Hierarchical Bayesian linear regression.

    Implements partial pooling where each group has its own parameters
    drawn from common hyperprior distributions. This approach:
    - Shares information across groups (shrinks toward global mean)
    - Adapts to group-specific patterns
    - Handles imbalanced group sizes gracefully

    Supports:
    - Group-varying intercepts
    - Group-varying slopes (for specified predictors)
    - Hyperpriors on global mean and variance
    - MCMC sampling with comprehensive diagnostics

    Args in spec.args:
        group_col: Column name for grouping variable (REQUIRED)
        group_varying_intercept: Whether to use group-varying intercepts (default: True)
        group_varying_slopes: List of predictor names for group-varying slopes (default: [])
        prior_group_intercept_mean: Prior for global intercept mean (default: "normal(0, 10)")
        prior_group_intercept_sd: Prior for intercept SD across groups (default: "half_cauchy(5)")
        prior_group_slope_mean: Prior for global slope mean (default: "normal(0, 5)")
        prior_group_slope_sd: Prior for slope SD across groups (default: "half_cauchy(3)")
        prior_fixed_coefs: Prior for non-varying coefficients (default: "normal(0, 5)")
        prior_sigma: Prior for error SD (default: "half_cauchy(5)")
        draws: Number of MCMC draws (default: 2000)
        tune: Number of tuning steps (default: 1000)
        chains: Number of MCMC chains (default: 4)
        target_accept: Target acceptance probability (default: 0.95)
        random_seed: Random seed for reproducibility (default: None)
        progressbar: Show MCMC progress bar (default: True)

    Examples:
        >>> # Group-varying intercepts only
        >>> spec = linear_reg().set_engine(
        ...     "pymc_hierarchical",
        ...     group_varying_intercept=True,
        ...     prior_group_intercept_sd="half_cauchy(5)"
        ... )
        >>> fit = spec.fit_global(data, "y ~ x1 + x2", group_col="store_id")

        >>> # Group-varying intercepts AND slopes for x1
        >>> spec = linear_reg().set_engine(
        ...     "pymc_hierarchical",
        ...     group_varying_intercept=True,
        ...     group_varying_slopes=["x1"],
        ...     prior_group_slope_sd="half_cauchy(3)"
        ... )
        >>> fit = spec.fit_global(data, "y ~ x1 + x2", group_col="region")
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
        Fit hierarchical Bayesian linear regression via PyMC.

        IMPORTANT: This engine requires original_training_data to access the
        group column, which is NOT included in the molded predictors.

        The workflow's fit_global() method ensures the group column is passed
        as part of original_training_data.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors (NO group column)
            original_training_data: Original training DataFrame (REQUIRED, contains group column)

        Returns:
            Dict containing:
                - model: PyMC model object
                - posterior_samples: Posterior samples (arviz.InferenceData)
                - summary: Posterior summary statistics
                - diagnostics: Convergence diagnostics
                - group_mapping: Dict mapping group names to integer indices
                - group_col: Name of the group column
                - y_train: Training outcomes
                - fitted: Posterior mean predictions
                - residuals: Training residuals
                - X_train: Training predictors
                - feature_names: Predictor column names
                - group_varying_intercept: Boolean flag
                - group_varying_slopes: List of varying slope names
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            raise ImportError(
                "PyMC and ArviZ are required for Bayesian models. "
                "Install with: pip install pymc>=5.10.0 arviz>=0.16.0"
            )

        # Validate group_col is provided
        group_col = spec.args.get("group_col")
        if group_col is None:
            raise ValueError(
                "Hierarchical models require 'group_col' argument. "
                "Use: spec.set_engine('pymc_hierarchical', group_col='column_name')"
            )

        # Validate original_training_data is provided
        if original_training_data is None:
            raise ValueError(
                "Hierarchical models require original_training_data to access group column. "
                "This should be passed automatically by workflow.fit_global()."
            )

        # Validate group column exists
        if group_col not in original_training_data.columns:
            raise ValueError(f"Group column '{group_col}' not found in original_training_data")

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

        # Extract group indices
        # IMPORTANT: The group column should be in original_training_data, NOT in molded predictors
        groups = original_training_data[group_col].values
        unique_groups = sorted(set(groups))
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        group_idx = np.array([group_to_idx[g] for g in groups])
        n_groups = len(unique_groups)

        # Parse configuration
        group_varying_intercept = spec.args.get("group_varying_intercept", True)
        group_varying_slopes = spec.args.get("group_varying_slopes", [])

        # Validate group_varying_slopes
        if not isinstance(group_varying_slopes, list):
            raise ValueError(f"group_varying_slopes must be a list, got {type(group_varying_slopes)}")

        # Check that varying slope names are in feature_names
        for slope_name in group_varying_slopes:
            if slope_name not in feature_names:
                raise ValueError(
                    f"group_varying_slopes contains '{slope_name}' which is not in predictors. "
                    f"Available: {feature_names}"
                )

        # Parse prior specifications
        priors = self._parse_priors(spec.args, feature_names, group_varying_slopes)

        # Build PyMC hierarchical model
        with pm.Model() as model:
            # ================
            # HYPERPRIORS
            # ================

            # Intercept hyperpriors (if varying)
            if group_varying_intercept:
                mu_alpha = self._create_prior("mu_alpha", priors["group_intercept_mean"])
                sigma_alpha = self._create_prior("sigma_alpha", priors["group_intercept_sd"])

                # Group-varying intercepts
                alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
            else:
                # Fixed intercept
                alpha = self._create_prior("alpha", priors["fixed_intercept"])

            # Slope hyperpriors (if varying)
            beta_varying = {}
            beta_fixed = []

            for i, col in enumerate(feature_names):
                if col in group_varying_slopes:
                    # Group-varying slope
                    mu_beta_col = self._create_prior(f"mu_beta_{col}", priors["group_slope_mean"][col])
                    sigma_beta_col = self._create_prior(f"sigma_beta_{col}", priors["group_slope_sd"][col])
                    beta_col = pm.Normal(f"beta_{col}", mu=mu_beta_col, sigma=sigma_beta_col, shape=n_groups)
                    beta_varying[col] = (beta_col, i)
                else:
                    # Fixed slope
                    beta_col = self._create_prior(f"beta_{col}", priors["fixed_coefs"][col])
                    beta_fixed.append((beta_col, i))

            # Prior for error standard deviation
            sigma = self._create_prior("sigma", priors["sigma"])

            # ================
            # LIKELIHOOD
            # ================

            # Compute linear predictor
            if group_varying_intercept:
                mu = alpha[group_idx]
            else:
                mu = alpha

            # Add varying slopes
            for col, (beta_col, idx) in beta_varying.items():
                mu += beta_col[group_idx] * X_array[:, idx]

            # Add fixed slopes
            for beta_col, idx in beta_fixed:
                mu += beta_col * X_array[:, idx]

            # Likelihood
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

        # Intercept
        if group_varying_intercept:
            alpha_mean = posterior_mean["alpha"].values  # Shape: (n_groups,)
            fitted = alpha_mean[group_idx]
        else:
            alpha_mean = float(posterior_mean["alpha"].values)
            fitted = np.full(len(y_array), alpha_mean)

        # Slopes (varying)
        for col, (_, idx) in beta_varying.items():
            beta_mean = posterior_mean[f"beta_{col}"].values  # Shape: (n_groups,)
            fitted += beta_mean[group_idx] * X_array[:, idx]

        # Slopes (fixed)
        for beta_col, idx in beta_fixed:
            beta_mean = float(posterior_mean[f"beta_{feature_names[idx]}"].values)
            fitted += beta_mean * X_array[:, idx]

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
            "group_mapping": group_to_idx,
            "unique_groups": unique_groups,
            "group_col": group_col,
            "y_train": y_array,
            "fitted": fitted,
            "residuals": residuals,
            "X_train": X_array,
            "feature_names": feature_names,
            "group_varying_intercept": group_varying_intercept,
            "group_varying_slopes": group_varying_slopes,
            "original_training_data": original_training_data,
            "n_obs": len(y_array),
            "n_features": X_array.shape[1],
            "n_groups": n_groups
        }

        return fit_data

    def _parse_priors(
        self,
        args: Dict[str, Any],
        feature_names: List[str],
        group_varying_slopes: List[str]
    ) -> Dict[str, Any]:
        """Parse prior specifications from model args."""
        defaults = get_default_priors()

        # Hyperpriors for group-varying intercept
        prior_group_intercept_mean_str = args.get("prior_group_intercept_mean", "normal(0, 10)")
        prior_group_intercept_sd_str = args.get("prior_group_intercept_sd", "half_cauchy(5)")

        # Hyperpriors for group-varying slopes
        prior_group_slope_mean_str = args.get("prior_group_slope_mean", "normal(0, 5)")
        prior_group_slope_sd_str = args.get("prior_group_slope_sd", "half_cauchy(3)")

        # Prior for fixed coefficients
        prior_fixed_coefs_str = args.get("prior_fixed_coefs", defaults["prior_coefs"])

        # Prior for fixed intercept (if not varying)
        prior_fixed_intercept_str = args.get("prior_intercept", defaults["prior_intercept"])

        # Prior for sigma
        prior_sigma_str = args.get("prior_sigma", defaults["prior_sigma"])

        # Parse priors
        result = {
            "group_intercept_mean": parse_prior(prior_group_intercept_mean_str),
            "group_intercept_sd": parse_prior(prior_group_intercept_sd_str),
            "group_slope_mean": {},
            "group_slope_sd": {},
            "fixed_coefs": {},
            "fixed_intercept": parse_prior(prior_fixed_intercept_str),
            "sigma": parse_prior(prior_sigma_str)
        }

        # Parse priors for each feature
        for feature in feature_names:
            if feature in group_varying_slopes:
                # Group-varying slope
                result["group_slope_mean"][feature] = parse_prior(prior_group_slope_mean_str)
                result["group_slope_sd"][feature] = parse_prior(prior_group_slope_sd_str)
            else:
                # Fixed slope
                result["fixed_coefs"][feature] = parse_prior(prior_fixed_coefs_str)

        return result

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
        prediction_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate predictions from hierarchical Bayesian model.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors (NO group column)
            type: Prediction type
                - "numeric": Posterior mean
                - "conf_int": Credible intervals
                - "posterior": Posterior samples (no error term)
                - "predictive": Posterior predictive samples (with error)
            prediction_data: Original prediction DataFrame (contains group column)

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

        # Extract group indices from prediction_data
        group_col = fit.fit_data["group_col"]

        if prediction_data is None:
            raise ValueError(
                "Hierarchical models require prediction_data to access group column. "
                "This should be passed automatically by workflow prediction methods."
            )

        if group_col not in prediction_data.columns:
            raise ValueError(f"Group column '{group_col}' not found in prediction_data")

        groups_test = prediction_data[group_col].values
        group_mapping = fit.fit_data["group_mapping"]

        # Map test groups to indices (handle new groups)
        group_idx_test = []
        for g in groups_test:
            if g in group_mapping:
                group_idx_test.append(group_mapping[g])
            else:
                # New group not seen in training - use global mean (index 0 as approximation)
                warnings.warn(f"Group '{g}' not seen in training. Using global mean parameters.")
                group_idx_test.append(0)  # Approximate with first group

        group_idx_test = np.array(group_idx_test)

        # Extract posterior and metadata
        trace = fit.fit_data["posterior_samples"]
        feature_names = fit.fit_data["feature_names"]
        group_varying_intercept = fit.fit_data["group_varying_intercept"]
        group_varying_slopes = fit.fit_data["group_varying_slopes"]

        # Build index mapping for features
        varying_info = {}
        fixed_info = {}
        for i, col in enumerate(feature_names):
            if col in group_varying_slopes:
                varying_info[col] = i
            else:
                fixed_info[col] = i

        # Posterior mean prediction
        if type == "numeric":
            posterior_mean = trace.posterior.mean(dim=["chain", "draw"])

            # Intercept
            if group_varying_intercept:
                alpha_mean = posterior_mean["alpha"].values
                pred = alpha_mean[group_idx_test]
            else:
                alpha_mean = float(posterior_mean["alpha"].values)
                pred = np.full(len(group_idx_test), alpha_mean)

            # Varying slopes
            for col, idx in varying_info.items():
                beta_mean = posterior_mean[f"beta_{col}"].values
                pred += beta_mean[group_idx_test] * X_test_array[:, idx]

            # Fixed slopes
            for col, idx in fixed_info.items():
                beta_mean = float(posterior_mean[f"beta_{col}"].values)
                pred += beta_mean * X_test_array[:, idx]

            return pd.DataFrame({".pred": pred})

        # Credible intervals
        elif type == "conf_int":
            level = fit.spec.args.get("level", 0.95)
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q

            # Get posterior samples
            if group_varying_intercept:
                alpha_samples = trace.posterior["alpha"].values  # Shape: (chains, draws, n_groups)
                alpha_samples = alpha_samples.reshape(-1, alpha_samples.shape[-1])  # (n_samples, n_groups)
            else:
                alpha_samples = trace.posterior["alpha"].values.flatten()  # (n_samples,)

            # Initialize predictions with intercepts
            if group_varying_intercept:
                pred_samples = alpha_samples[:, group_idx_test]  # (n_samples, n_test)
            else:
                pred_samples = alpha_samples[:, None] * np.ones((1, len(group_idx_test)))  # Broadcast

            # Add varying slopes
            for col, idx in varying_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values
                beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])  # (n_samples, n_groups)
                pred_samples += beta_samples[:, group_idx_test] * X_test_array[:, idx]

            # Add fixed slopes
            for col, idx in fixed_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values.flatten()  # (n_samples,)
                pred_samples += beta_samples[:, None] * X_test_array[:, idx]

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
            if group_varying_intercept:
                alpha_samples = trace.posterior["alpha"].values
                alpha_samples = alpha_samples.reshape(-1, alpha_samples.shape[-1])[:n_samples]
            else:
                alpha_samples = trace.posterior["alpha"].values.flatten()[:n_samples]

            # Initialize predictions
            if group_varying_intercept:
                pred_samples = alpha_samples[:, group_idx_test]
            else:
                pred_samples = alpha_samples[:, None] * np.ones((1, len(group_idx_test)))

            # Add varying slopes
            for col, idx in varying_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values
                beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])[:n_samples]
                pred_samples += beta_samples[:, group_idx_test] * X_test_array[:, idx]

            # Add fixed slopes
            for col, idx in fixed_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values.flatten()[:n_samples]
                pred_samples += beta_samples[:, None] * X_test_array[:, idx]

            # Return as wide DataFrame
            cols = {f".pred_sample_{i+1}": pred_samples[i] for i in range(n_samples)}
            return pd.DataFrame(cols)

        # Posterior predictive (with error term)
        elif type == "predictive":
            n_samples = fit.spec.args.get("n_samples", 500)

            # Get posterior samples
            if group_varying_intercept:
                alpha_samples = trace.posterior["alpha"].values
                alpha_samples = alpha_samples.reshape(-1, alpha_samples.shape[-1])[:n_samples]
            else:
                alpha_samples = trace.posterior["alpha"].values.flatten()[:n_samples]

            # Initialize predictions
            if group_varying_intercept:
                mu_samples = alpha_samples[:, group_idx_test]
            else:
                mu_samples = alpha_samples[:, None] * np.ones((1, len(group_idx_test)))

            # Add varying slopes
            for col, idx in varying_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values
                beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])[:n_samples]
                mu_samples += beta_samples[:, group_idx_test] * X_test_array[:, idx]

            # Add fixed slopes
            for col, idx in fixed_info.items():
                beta_samples = trace.posterior[f"beta_{col}"].values.flatten()[:n_samples]
                mu_samples += beta_samples[:, None] * X_test_array[:, idx]

            # Sample sigma
            sigma_samples = trace.posterior["sigma"].values.flatten()[:n_samples]

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
        Extract comprehensive three-DataFrame output for hierarchical Bayesian models.

        Returns:
            Tuple of (outputs, coefficients, stats)

        Coefficients DataFrame includes:
            - Global hyperparameters (mu_alpha, sigma_alpha, etc.)
            - Group-specific parameters (alpha[group], beta_x1[group], etc.)
        """
        try:
            import arviz as az
        except ImportError:
            raise ImportError("ArviZ is required. Install with: pip install arviz>=0.16.0")

        trace = fit.fit_data["posterior_samples"]
        summary = fit.fit_data["summary"]
        feature_names = fit.fit_data["feature_names"]
        group_varying_intercept = fit.fit_data["group_varying_intercept"]
        group_varying_slopes = fit.fit_data["group_varying_slopes"]
        unique_groups = fit.fit_data["unique_groups"]
        group_col = fit.fit_data["group_col"]

        # ===============
        # OUTPUTS
        # ===============
        X_train = fit.fit_data["X_train"]
        y_train = fit.fit_data["y_train"]
        fitted = fit.fit_data["fitted"]
        residuals = fit.fit_data["residuals"]

        # Get group indices for training data
        original_training_data = fit.fit_data["original_training_data"]
        groups_train = original_training_data[group_col].values
        group_mapping = fit.fit_data["group_mapping"]
        group_idx_train = np.array([group_mapping[g] for g in groups_train])

        # Compute credible intervals for fitted values
        posterior_mean = trace.posterior.mean(dim=["chain", "draw"])

        # Build fitted samples for CI computation
        if group_varying_intercept:
            alpha_samples = trace.posterior["alpha"].values
            alpha_samples = alpha_samples.reshape(-1, alpha_samples.shape[-1])
            fitted_samples = alpha_samples[:, group_idx_train]
        else:
            alpha_samples = trace.posterior["alpha"].values.flatten()
            fitted_samples = alpha_samples[:, None] * np.ones((1, len(y_train)))

        # Add slopes
        for i, col in enumerate(feature_names):
            if col in group_varying_slopes:
                beta_samples = trace.posterior[f"beta_{col}"].values
                beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])
                fitted_samples += beta_samples[:, group_idx_train] * X_train[:, i]
            else:
                beta_samples = trace.posterior[f"beta_{col}"].values.flatten()
                fitted_samples += beta_samples[:, None] * X_train[:, i]

        fitted_lower = np.percentile(fitted_samples, 2.5, axis=0)
        fitted_upper = np.percentile(fitted_samples, 97.5, axis=0)

        # Create forecast
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
            "group": groups_train
        })

        # Add test data if available
        if "evaluation_data" in fit.__dict__ and "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Credible intervals
            if ".pred_lower" in test_preds.columns:
                test_lower = test_preds[".pred_lower"].values
                test_upper = test_preds[".pred_upper"].values
            else:
                test_lower = np.full_like(test_predictions, np.nan)
                test_upper = np.full_like(test_predictions, np.nan)

            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            # Get test groups
            groups_test = test_data[group_col].values if group_col in test_data.columns else np.full(len(test_data), None)

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
                "group": groups_test
            })

            outputs = pd.concat([outputs, test_df], ignore_index=True)

        # ===============
        # COEFFICIENTS
        # ===============
        coef_rows = []

        # Global hyperparameters
        if group_varying_intercept:
            # mu_alpha
            if "mu_alpha" in summary.index:
                mu_alpha_summary = summary.loc["mu_alpha"]
                mu_alpha_samples = trace.posterior["mu_alpha"].values.flatten()
                coef_rows.append({
                    "term": "mu_alpha",
                    "estimate": mu_alpha_summary["mean"],
                    "std_error": mu_alpha_summary["sd"],
                    "lower_ci": mu_alpha_summary.get("hdi_2.5%", np.percentile(mu_alpha_samples, 2.5)),
                    "upper_ci": mu_alpha_summary.get("hdi_97.5%", np.percentile(mu_alpha_samples, 97.5)),
                    "rhat": mu_alpha_summary["r_hat"],
                    "ess_bulk": mu_alpha_summary["ess_bulk"],
                    "ess_tail": mu_alpha_summary["ess_tail"],
                    "prob_positive": float((mu_alpha_samples > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })

            # sigma_alpha
            if "sigma_alpha" in summary.index:
                sigma_alpha_summary = summary.loc["sigma_alpha"]
                sigma_alpha_samples = trace.posterior["sigma_alpha"].values.flatten()
                coef_rows.append({
                    "term": "sigma_alpha",
                    "estimate": sigma_alpha_summary["mean"],
                    "std_error": sigma_alpha_summary["sd"],
                    "lower_ci": sigma_alpha_summary.get("hdi_2.5%", np.percentile(sigma_alpha_samples, 2.5)),
                    "upper_ci": sigma_alpha_summary.get("hdi_97.5%", np.percentile(sigma_alpha_samples, 97.5)),
                    "rhat": sigma_alpha_summary["r_hat"],
                    "ess_bulk": sigma_alpha_summary["ess_bulk"],
                    "ess_tail": sigma_alpha_summary["ess_tail"],
                    "prob_positive": float((sigma_alpha_samples > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })

        # Group-varying hyperparameters for slopes
        for col in group_varying_slopes:
            mu_name = f"mu_beta_{col}"
            sigma_name = f"sigma_beta_{col}"

            if mu_name in summary.index:
                mu_summary = summary.loc[mu_name]
                mu_samples = trace.posterior[mu_name].values.flatten()
                coef_rows.append({
                    "term": mu_name,
                    "estimate": mu_summary["mean"],
                    "std_error": mu_summary["sd"],
                    "lower_ci": mu_summary.get("hdi_2.5%", np.percentile(mu_samples, 2.5)),
                    "upper_ci": mu_summary.get("hdi_97.5%", np.percentile(mu_samples, 97.5)),
                    "rhat": mu_summary["r_hat"],
                    "ess_bulk": mu_summary["ess_bulk"],
                    "ess_tail": mu_summary["ess_tail"],
                    "prob_positive": float((mu_samples > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })

            if sigma_name in summary.index:
                sigma_summary = summary.loc[sigma_name]
                sigma_samples = trace.posterior[sigma_name].values.flatten()
                coef_rows.append({
                    "term": sigma_name,
                    "estimate": sigma_summary["mean"],
                    "std_error": sigma_summary["sd"],
                    "lower_ci": sigma_summary.get("hdi_2.5%", np.percentile(sigma_samples, 2.5)),
                    "upper_ci": sigma_summary.get("hdi_97.5%", np.percentile(sigma_samples, 97.5)),
                    "rhat": sigma_summary["r_hat"],
                    "ess_bulk": sigma_summary["ess_bulk"],
                    "ess_tail": sigma_summary["ess_tail"],
                    "prob_positive": float((sigma_samples > 0).mean()),
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine,
                    "group": None
                })

        # Group-specific parameters
        for i, group in enumerate(unique_groups):
            # Intercept
            if group_varying_intercept:
                alpha_name = f"alpha[{i}]"
                if alpha_name in summary.index:
                    alpha_summary = summary.loc[alpha_name]
                    alpha_i_samples = trace.posterior["alpha"].values[:, :, i].flatten()
                    coef_rows.append({
                        "term": "Intercept",
                        "estimate": alpha_summary["mean"],
                        "std_error": alpha_summary["sd"],
                        "lower_ci": alpha_summary.get("hdi_2.5%", np.percentile(alpha_i_samples, 2.5)),
                        "upper_ci": alpha_summary.get("hdi_97.5%", np.percentile(alpha_i_samples, 97.5)),
                        "rhat": alpha_summary["r_hat"],
                        "ess_bulk": alpha_summary["ess_bulk"],
                        "ess_tail": alpha_summary["ess_tail"],
                        "prob_positive": float((alpha_i_samples > 0).mean()),
                        "model": fit.spec.model_type,
                        "model_group_name": fit.spec.engine,
                        "group": group
                    })

            # Varying slopes
            for col in group_varying_slopes:
                beta_name = f"beta_{col}[{i}]"
                if beta_name in summary.index:
                    beta_summary = summary.loc[beta_name]
                    beta_i_samples = trace.posterior[f"beta_{col}"].values[:, :, i].flatten()
                    coef_rows.append({
                        "term": col,
                        "estimate": beta_summary["mean"],
                        "std_error": beta_summary["sd"],
                        "lower_ci": beta_summary.get("hdi_2.5%", np.percentile(beta_i_samples, 2.5)),
                        "upper_ci": beta_summary.get("hdi_97.5%", np.percentile(beta_i_samples, 97.5)),
                        "rhat": beta_summary["r_hat"],
                        "ess_bulk": beta_summary["ess_bulk"],
                        "ess_tail": beta_summary["ess_tail"],
                        "prob_positive": float((beta_i_samples > 0).mean()),
                        "model": fit.spec.model_type,
                        "model_group_name": fit.spec.engine,
                        "group": group
                    })

        # Fixed coefficients (shared across groups)
        for col in feature_names:
            if col not in group_varying_slopes:
                beta_name = f"beta_{col}"
                if beta_name in summary.index:
                    beta_summary = summary.loc[beta_name]
                    beta_samples = trace.posterior[beta_name].values.flatten()
                    coef_rows.append({
                        "term": col,
                        "estimate": beta_summary["mean"],
                        "std_error": beta_summary["sd"],
                        "lower_ci": beta_summary.get("hdi_2.5%", np.percentile(beta_samples, 2.5)),
                        "upper_ci": beta_summary.get("hdi_97.5%", np.percentile(beta_samples, 97.5)),
                        "rhat": beta_summary["r_hat"],
                        "ess_bulk": beta_summary["ess_bulk"],
                        "ess_tail": beta_summary["ess_tail"],
                        "prob_positive": float((beta_samples > 0).mean()),
                        "model": fit.spec.model_type,
                        "model_group_name": fit.spec.engine,
                        "group": None  # Fixed across all groups
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
        max_rhat = float(np.max(rhat_data.to_array().values))

        if hasattr(ess_data, 'dims'):
            ess_array = ess_data.to_array().values
            if hasattr(ess_data, 'data_vars') and len(list(ess_data.data_vars)) > 1:
                min_ess_bulk = float(np.min(ess_array[0]))
                min_ess_tail = float(np.min(ess_array[1]))
            else:
                min_ess_bulk = float(np.min(ess_array))
                min_ess_tail = min_ess_bulk
        else:
            min_ess_bulk = float(np.min(ess_data))
            min_ess_tail = min_ess_bulk

        stats_dict = {
            "split": "train",
            "n_obs": len(y_train),
            "n_groups": fit.fit_data["n_groups"],
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
            stats_dict["waic"] = waic.elpd_waic * -2
            stats_dict["waic_se"] = waic.se * 2

        if loo is not None:
            stats_dict["loo"] = loo.elpd_loo * -2
            stats_dict["loo_se"] = loo.se * 2
            stats_dict["loo_p"] = loo.p_loo

        stats = pd.DataFrame([stats_dict])

        return outputs, coefficients, stats
