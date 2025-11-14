"""
Statsmodels engine for panel regression (Linear Mixed Effects Models)

Maps panel_reg to statsmodels MixedLM for panel regression with random effects.
Provides full statistical inference for fixed effects and variance components for
random effects.

Supports:
- Random intercepts: Each group has its own baseline level
- Random slopes: Each group has its own slope for a specified variable
- Random intercepts + slopes: Both group-specific intercepts and slopes
- ICC (Intraclass Correlation Coefficient): Proportion of variance due to groups
"""

from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


@register_engine("panel_reg", "statsmodels")
class StatsmodelsPanelEngine(Engine):
    """
    Statsmodels engine for panel regression (Linear Mixed Effects).

    Provides panel regression with random effects:
    - Random intercepts (default): Each group has its own baseline level
    - Random slopes: Each group has its own slope for specified variable
    - Random intercepts + slopes: Both group-specific intercepts and slopes
    - Full statistical inference for fixed effects
    - Variance components for random effects
    - Group-level random effects accessible

    Requirements:
    - Must use fit_global() to specify group column
    - Group column must have at least 2 groups
    - Each group must have at least 2 observations
    - Random slopes require .set_args(slope_var='variable_name')
    """

    param_map = {
        "intercept": "fit_intercept",
    }

    def fit(
        self,
        spec: ModelSpec,
        molded: MoldedData,
        original_training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit mixed linear effects model using statsmodels MixedLM.

        Implementation follows the standard mold/forge path (not raw path).
        Supports both random intercepts and random slopes.

        Args:
            spec: ModelSpec with panel_reg configuration
            molded: MoldedData with outcomes and predictors (from mold/forge)
            original_training_data: Original training data with group column

        Returns:
            Dict containing:
                - model: Fitted MixedLM result object
                - groups: Group labels array
                - random_effects: Dict mapping group → random effects
                - cov_re: Random effects covariance matrix
                - n_groups: Number of groups
                - group_sizes: Series with observations per group
                - [standard fit_data fields]

        Raises:
            ValueError: If group column missing, insufficient groups, or invalid random_effects
        """
        from statsmodels.regression.mixed_linear_model import MixedLM
        import statsmodels.api as sm

        # Check for unsupported parameters
        args = spec.args
        if "penalty" in args and args["penalty"] is not None:
            raise ValueError(
                "Regularized mixed models not yet implemented. "
                "penalty parameter is reserved for future enhancement."
            )

        # =====================
        # 1. EXTRACT AND VALIDATE GROUP COLUMN
        # =====================
        if original_training_data is None:
            raise ValueError(
                "panel_reg requires original_training_data with group column. "
                "Use fit_global(data, group_col='column_name')"
            )

        # The group_col should be stored in spec.args by fit_global()
        group_col_name = args.get("_group_col")
        if group_col_name is None:
            raise ValueError(
                "panel_reg requires group column. Use fit_global(data, group_col='column_name')"
            )

        if group_col_name not in original_training_data.columns:
            raise ValueError(
                f"Group column '{group_col_name}' not found in training data. "
                f"Available columns: {list(original_training_data.columns)}"
            )

        groups = original_training_data[group_col_name].values

        # Validate group structure
        unique_groups = pd.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError(
                f"Need at least 2 groups for panel regression, got {len(unique_groups)}. "
                f"Found only group: {unique_groups[0] if len(unique_groups) == 1 else 'none'}"
            )

        group_sizes = pd.Series(groups).value_counts()
        min_group_size = group_sizes.min()
        if min_group_size < 2:
            min_group = group_sizes.idxmin()
            raise ValueError(
                f"All groups must have at least 2 observations. "
                f"Group '{min_group}' has only {min_group_size} observation(s). "
                f"Consider: (1) Removing singleton groups, (2) Using linear_reg() for small datasets"
            )

        # =====================
        # 2. EXTRACT PREDICTORS AND OUTCOMES
        # =====================
        X = molded.predictors
        y = molded.outcomes

        # Flatten y if it's a DataFrame with single column
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]

        # =====================
        # 3. HANDLE INTERCEPT PARAMETER
        # =====================
        intercept = args.get("intercept", True)

        # Check if X already has an intercept column (from patsy)
        has_intercept_col = False
        intercept_col_name = None
        if isinstance(X, pd.DataFrame):
            for col in ["Intercept", "const", "intercept"]:
                if col in X.columns:
                    has_intercept_col = True
                    intercept_col_name = col
                    break

        # Modify X based on intercept parameter
        if not intercept and has_intercept_col:
            # User wants no intercept, but X has one - remove it
            import warnings
            X = X.drop(columns=[intercept_col_name])
            warnings.warn(
                f"intercept=False: Removing '{intercept_col_name}' column from design matrix.",
                UserWarning
            )
        elif intercept and not has_intercept_col:
            # User wants intercept, but X doesn't have one - add it
            X = sm.add_constant(X, has_constant='skip')
            has_intercept_col = True
            intercept_col_name = "const"

        # =====================
        # 4. BUILD RANDOM EFFECTS DESIGN MATRIX (exog_re)
        # =====================
        random_effects = args.get("random_effects", "intercept")
        exog_re = None
        slope_var = args.get("slope_var")

        if random_effects == "intercept" or random_effects is None:
            # Random intercepts only (default)
            # MixedLM interprets exog_re=None as random intercepts
            exog_re = None

        elif random_effects in ("slope", "both"):
            # Random intercepts + random slope
            if slope_var is None:
                raise ValueError(
                    f"random_effects='{random_effects}' requires slope_var parameter. "
                    f"Use .set_args(slope_var='variable_name'). "
                    f"Available variables: {list(X.columns)}"
                )

            if slope_var not in X.columns:
                raise ValueError(
                    f"slope_var '{slope_var}' not found in predictors. "
                    f"Available predictors: {list(X.columns)}. "
                    f"Check your formula or recipe preprocessing."
                )

            # Build exog_re: intercept column + slope variable
            # This creates random intercepts AND random slopes
            if has_intercept_col and intercept_col_name:
                exog_re = X[[intercept_col_name, slope_var]]
            else:
                # Add constant column for random intercept
                slope_data = X[[slope_var]]
                exog_re = sm.add_constant(slope_data, has_constant='skip')
                # Rename columns for clarity
                exog_re.columns = ["const", slope_var]

        else:
            raise ValueError(
                f"Invalid random_effects: '{random_effects}'. "
                f"Must be 'intercept', 'slope', 'both', or None"
            )

        # =====================
        # 5. FIT MIXEDLM MODEL
        # =====================
        model = MixedLM(endog=y, exog=X, groups=groups, exog_re=exog_re)
        fitted_model = model.fit(method=['lbfgs'])

        # Calculate fitted values and residuals
        fitted = fitted_model.fittedvalues.values
        residuals = fitted_model.resid.values

        # =====================
        # 6. EXTRACT RANDOM EFFECTS AND COVARIANCE
        # =====================
        random_effects_dict = fitted_model.random_effects
        cov_re = fitted_model.cov_re

        # =====================
        # 7. RETURN FIT_DATA DICT
        # =====================
        return {
            "model": fitted_model,
            "n_features": X.shape[1],
            "n_obs": X.shape[0],
            "X_train": X,
            "y_train": y,
            "fitted": fitted,
            "residuals": residuals,
            "groups": groups,
            "group_col": group_col_name,
            "random_effects": random_effects_dict,
            "cov_re": cov_re,
            "n_groups": len(unique_groups),
            "group_sizes": group_sizes,
            "random_effects_spec": random_effects,
            "slope_var": slope_var if random_effects in ("slope", "both") else None,
            "original_training_data": original_training_data,
        }

    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted MixedLM model.

        For groups seen during training: Uses fixed effects + group-specific random effects
        For new groups: Uses fixed effects only (population average)

        Args:
            fit: ModelFit with fitted MixedLM model
            molded: MoldedData with predictors
            type: Prediction type
                - "numeric": Point predictions
                - "conf_int": Predictions with confidence intervals

        Returns:
            DataFrame with predictions (.pred column)
        """
        import statsmodels.api as sm

        if type not in ("numeric", "conf_int"):
            raise ValueError(
                f"panel_reg supports type='numeric' or 'conf_int', got '{type}'"
            )

        model = fit.fit_data["model"]
        X = molded.predictors

        # =====================
        # HANDLE INTERCEPT (match fit())
        # =====================
        intercept = fit.spec.args.get("intercept", True)

        # Check if X has an intercept column
        has_intercept_col = False
        intercept_col_name = None
        if isinstance(X, pd.DataFrame):
            for col in ["Intercept", "const", "intercept"]:
                if col in X.columns:
                    has_intercept_col = True
                    intercept_col_name = col
                    break

        # Modify X to match training data
        if not intercept and has_intercept_col:
            X = X.drop(columns=[intercept_col_name])
        elif intercept and not has_intercept_col:
            X = sm.add_constant(X, has_constant='skip')
            has_intercept_col = True
            intercept_col_name = "const"

        # =====================
        # MAKE PREDICTIONS
        # =====================
        # MixedLM.predict() automatically handles:
        # - Training groups: Uses fixed + random effects
        # - New groups: Uses fixed effects only (population average)
        # Note: statsmodels MixedLM.predict() doesn't take exog_re parameter
        # It uses the fitted random effects structure automatically
        predictions = model.predict(X)

        if type == "numeric":
            return pd.DataFrame({".pred": predictions.values})
        else:  # conf_int
            # Get prediction intervals
            # Note: MixedLM prediction intervals are complex (involve both fixed and random effects uncertainty)
            # For now, we use approximate intervals from get_prediction()
            try:
                pred_summary = model.get_prediction(X)
                pred_int = pred_summary.conf_int(alpha=0.05)  # 95% CI

                return pd.DataFrame({
                    ".pred": predictions.values,
                    ".pred_lower": pred_int[:, 0],
                    ".pred_upper": pred_int[:, 1],
                })
            except Exception as e:
                # If get_prediction fails, return predictions only
                import warnings
                warnings.warn(
                    f"Could not compute confidence intervals: {e}. "
                    f"Returning point predictions only.",
                    UserWarning
                )
                return pd.DataFrame({
                    ".pred": predictions.values,
                    ".pred_lower": np.nan,
                    ".pred_upper": np.nan,
                })

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        residuals = actuals - predictions
        n = len(actuals)

        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # MAPE (avoid division by zero)
        mask = actuals != 0
        mape = np.mean(np.abs(residuals[mask] / actuals[mask]) * 100) if mask.any() else np.nan

        # SMAPE
        smape = np.mean(2 * np.abs(residuals) / (np.abs(actuals) + np.abs(predictions)) * 100)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        # MDA (Mean Directional Accuracy)
        if n > 1:
            actual_direction = np.diff(actuals) > 0
            pred_direction = np.diff(predictions) > 0
            mda = np.mean(actual_direction == pred_direction) * 100
        else:
            mda = np.nan

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
            "r_squared": r_squared,
            "mda": mda,
        }

    def _calculate_residual_diagnostics(self, residuals: np.ndarray, model) -> Dict[str, float]:
        """Calculate residual diagnostic statistics"""
        from scipy import stats as scipy_stats
        import statsmodels.stats.diagnostic as sm_diag

        results = {}
        n = len(residuals)

        # Durbin-Watson statistic
        if n > 1:
            diff_resid = np.diff(residuals)
            dw = np.sum(diff_resid ** 2) / np.sum(residuals ** 2)
            results["durbin_watson"] = dw
        else:
            results["durbin_watson"] = np.nan

        # Shapiro-Wilk test for normality
        if n >= 3:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
            results["shapiro_wilk_stat"] = shapiro_stat
            results["shapiro_wilk_p"] = shapiro_p
        else:
            results["shapiro_wilk_stat"] = np.nan
            results["shapiro_wilk_p"] = np.nan

        # Ljung-Box test for autocorrelation
        try:
            n_lags = max(1, min(10, n // 5))
            lb_result = sm_diag.acorr_ljungbox(residuals, lags=n_lags)
            results["ljung_box_stat"] = lb_result['lb_stat'].iloc[-1]
            results["ljung_box_p"] = lb_result['lb_pvalue'].iloc[-1]
        except Exception:
            results["ljung_box_stat"] = np.nan
            results["ljung_box_p"] = np.nan

        # Breusch-Pagan test for heteroskedasticity
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'exog'):
                bp_result = sm_diag.het_breuschpagan(residuals, model.model.exog)
                results["breusch_pagan_stat"] = bp_result[0]
                results["breusch_pagan_p"] = bp_result[1]
            else:
                results["breusch_pagan_stat"] = np.nan
                results["breusch_pagan_p"] = np.nan
        except Exception:
            results["breusch_pagan_stat"] = np.nan
            results["breusch_pagan_p"] = np.nan

        return results

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame output for panel regression.

        Returns:
            Tuple of (outputs, coefficients, stats)

        Outputs DataFrame:
            - actuals, fitted, forecast, residuals, split (standard)
            - group: Group identifier for each observation

        Coefficients DataFrame:
            Fixed effects:
                - variable, coefficient, std_error, t_stat, p_value, ci (standard)
                - vif: Variance inflation factor
                - type: "fixed"
            Random effects variance components:
                - variable="RE: Intercept Variance", coefficient=variance, type="random"
                - variable="RE: slope_var Variance", coefficient=variance, type="random"
                - variable="RE: Cov(Intercept, slope_var)", coefficient=covariance, type="random"
                - variable="Residual Variance", coefficient=variance, type="residual"

        Stats DataFrame:
            - Standard metrics: rmse, mae, mape, r_squared, etc. (per split)
            - Model info: aic, bic, log_likelihood, n_groups
            - ICC (Intraclass Correlation): proportion of variance due to groups
            - Group statistics: min_group_size, max_group_size, mean_group_size
        """
        model = fit.fit_data["model"]

        # ====================
        # 1. OUTPUTS DataFrame
        # ====================
        outputs_list = []

        # Training data
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        groups = fit.fit_data.get("groups")

        if y_train is not None and fitted is not None:
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            # Create forecast: actuals where they exist, fitted where they don't
            forecast_train = pd.Series(y_train_array).combine_first(pd.Series(fitted)).values

            train_df = pd.DataFrame({
                "actuals": y_train_array,
                "fitted": fitted,
                "forecast": forecast_train,
                "residuals": residuals if residuals is not None else y_train_array - fitted,
                "split": "train",
                "group": groups,  # Add group column
            })

            # Add model metadata
            train_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            train_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""

            outputs_list.append(train_df)

        # Test data (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_data = fit.evaluation_data["test_data"]
            test_preds = fit.evaluation_data["test_predictions"]
            outcome_col = fit.evaluation_data["outcome_col"]

            test_actuals = test_data[outcome_col].values
            test_predictions = test_preds[".pred"].values
            test_residuals = test_actuals - test_predictions

            # Create forecast
            forecast_test = pd.Series(test_actuals).combine_first(pd.Series(test_predictions)).values

            # Extract group column from test data
            group_col = fit.fit_data.get("group_col")
            test_groups = fit.evaluation_data.get("test_groups")
            if test_groups is None and "original_test_data" in fit.evaluation_data:
                orig_test = fit.evaluation_data["original_test_data"]
                if group_col and group_col in orig_test.columns:
                    test_groups = orig_test[group_col].values
                else:
                    test_groups = ["unknown"] * len(test_actuals)
            elif test_groups is None:
                test_groups = ["unknown"] * len(test_actuals)

            test_df = pd.DataFrame({
                "actuals": test_actuals,
                "fitted": test_predictions,
                "forecast": forecast_test,
                "residuals": test_residuals,
                "split": "test",
                "group": test_groups,
            })

            # Add model metadata
            test_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
            test_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""

            outputs_list.append(test_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # Try to add date column if available
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=None,
                    fit_date_col=None
                )

                # Extract date values for training data
                if date_col == '__index__':
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values

                # Handle test data if present
                if fit.evaluation_data and 'original_test_data' in fit.evaluation_data:
                    test_data_orig = fit.evaluation_data['original_test_data']
                    if date_col == '__index__':
                        test_dates = test_data_orig.index.values
                    else:
                        test_dates = test_data_orig[date_col].values

                    all_dates = np.concatenate([train_dates, test_dates])
                else:
                    all_dates = train_dates

                # Insert date column at position 0
                outputs.insert(0, 'date', all_dates)

        except (ValueError, ImportError):
            pass  # Skip date column if not available

        # ====================
        # 2. COEFFICIENTS DataFrame
        # ====================
        # Fixed effects (from model.params)
        coef_names = list(model.params.index)
        coef_values = model.params.values
        std_errors = model.bse.values
        t_stats = model.tvalues.values
        p_values = model.pvalues.values

        # Get confidence intervals
        conf_int = model.conf_int(alpha=0.05)  # 95% CI
        ci_lower = conf_int.iloc[:, 0].values
        ci_upper = conf_int.iloc[:, 1].values

        # Calculate VIF for each predictor (skip intercept)
        X_train = fit.fit_data.get("X_train")
        vifs = [np.nan] * len(coef_names)

        if X_train is not None and X_train.shape[1] > 1:
            for i, col_name in enumerate(coef_names):
                if col_name not in ["Intercept", "const", "intercept"]:
                    try:
                        X_i = X_train.iloc[:, i:i+1]
                        X_not_i = X_train.drop(X_train.columns[i], axis=1)

                        if X_not_i.shape[1] > 0:
                            from sklearn.linear_model import LinearRegression as LR
                            vif_model = LR()
                            vif_model.fit(X_not_i, X_i)
                            r_squared_i = vif_model.score(X_not_i, X_i)
                            if r_squared_i < 0.9999:
                                vifs[i] = 1 / (1 - r_squared_i)
                    except:
                        pass

        fixed_coefs = pd.DataFrame({
            "variable": coef_names,
            "coefficient": coef_values,
            "std_error": std_errors,
            "t_stat": t_stats,
            "p_value": p_values,
            "ci_0.025": ci_lower,
            "ci_0.975": ci_upper,
            "vif": vifs,
            "type": "fixed",
        })

        # Random effects variance components
        re_rows = []
        cov_re = fit.fit_data["cov_re"]
        random_effects_spec = fit.fit_data.get("random_effects_spec")
        slope_var = fit.fit_data.get("slope_var")

        # Variance of random intercept
        # Note: cov_re can be a DataFrame or ndarray depending on statsmodels version
        cov_re_values = cov_re.values if isinstance(cov_re, pd.DataFrame) else cov_re

        if cov_re_values.shape[0] >= 1:
            re_rows.append({
                "variable": "RE: Intercept Variance",
                "coefficient": cov_re_values[0, 0],
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
                "type": "random",
            })

        # Variance of random slope (if applicable)
        if random_effects_spec in ("slope", "both") and cov_re_values.shape[0] >= 2:
            re_rows.append({
                "variable": f"RE: {slope_var} Variance",
                "coefficient": cov_re_values[1, 1],
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
                "type": "random",
            })

            # Covariance between intercept and slope
            re_rows.append({
                "variable": f"RE: Cov(Intercept, {slope_var})",
                "coefficient": cov_re_values[0, 1],
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_0.025": np.nan,
                "ci_0.975": np.nan,
                "vif": np.nan,
                "type": "random",
            })

        # Residual variance
        re_rows.append({
            "variable": "Residual Variance",
            "coefficient": model.scale,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_0.025": np.nan,
            "ci_0.975": np.nan,
            "vif": np.nan,
            "type": "residual",
        })

        re_coefs = pd.DataFrame(re_rows)
        coefficients = pd.concat([fixed_coefs, re_coefs], ignore_index=True)

        # Add model metadata
        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"  # Panel models are global

        # ====================
        # 3. STATS DataFrame
        # ====================
        stats_rows = []
        n_obs = fit.fit_data.get("n_obs", 0)
        n_features = fit.fit_data.get("n_features", 0)

        # Training metrics
        if y_train is not None and fitted is not None:
            train_metrics = self._calculate_metrics(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                fitted
            )

            # Add adjusted R-squared
            r_sq = train_metrics["r_squared"]
            if not np.isnan(r_sq) and n_obs > n_features:
                adj_r_sq = 1 - (1 - r_sq) * (n_obs - 1) / (n_obs - n_features - 1)
                train_metrics["adj_r_squared"] = adj_r_sq
            else:
                train_metrics["adj_r_squared"] = np.nan

            for metric_name, value in train_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Test metrics (if evaluated)
        if "test_predictions" in fit.evaluation_data:
            test_actuals = test_data[outcome_col].values
            test_forecast = test_preds[".pred"].values

            test_metrics = self._calculate_metrics(test_actuals, test_forecast)

            # Add adjusted R-squared for test
            r_sq_test = test_metrics["r_squared"]
            n_test = len(test_actuals)
            if not np.isnan(r_sq_test) and n_test > n_features:
                adj_r_sq_test = 1 - (1 - r_sq_test) * (n_test - 1) / (n_test - n_features - 1)
                test_metrics["adj_r_squared"] = adj_r_sq_test
            else:
                test_metrics["adj_r_squared"] = np.nan

            for metric_name, value in test_metrics.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "test",
                })

        # Residual diagnostics (on training data)
        if residuals is not None and len(residuals) > 0:
            resid_diag = self._calculate_residual_diagnostics(residuals, model)
            for metric_name, value in resid_diag.items():
                stats_rows.append({
                    "metric": metric_name,
                    "value": value,
                    "split": "train",
                })

        # Panel-specific statistics
        # Calculate ICC (Intraclass Correlation Coefficient)
        # ICC = var(random_intercept) / (var(random_intercept) + var(residual))
        cov_re_for_icc = fit.fit_data["cov_re"]
        cov_re_values_icc = cov_re_for_icc.values if isinstance(cov_re_for_icc, pd.DataFrame) else cov_re_for_icc

        if cov_re_values_icc.shape[0] >= 1:
            random_intercept_var = cov_re_values_icc[0, 0]
            residual_var = model.scale
            icc = random_intercept_var / (random_intercept_var + residual_var)
        else:
            icc = np.nan

        # Model-level statistics
        group_sizes = fit.fit_data.get("group_sizes")
        stats_rows.extend([
            {"metric": "aic", "value": model.aic, "split": ""},
            {"metric": "bic", "value": model.bic, "split": ""},
            {"metric": "log_likelihood", "value": model.llf, "split": ""},
            {"metric": "n_groups", "value": fit.fit_data.get("n_groups"), "split": ""},
            {"metric": "min_group_size", "value": group_sizes.min() if group_sizes is not None else np.nan, "split": ""},
            {"metric": "max_group_size", "value": group_sizes.max() if group_sizes is not None else np.nan, "split": ""},
            {"metric": "mean_group_size", "value": group_sizes.mean() if group_sizes is not None else np.nan, "split": ""},
            {"metric": "icc", "value": icc, "split": ""},  # KEY PANEL METRIC
            {"metric": "n_obs_train", "value": n_obs, "split": "train"},
            {"metric": "n_features", "value": n_features, "split": ""},
        ])

        # Add training date range if available
        train_dates = None
        try:
            from py_parsnip.utils import _infer_date_column

            if fit.fit_data.get("original_training_data") is not None:
                date_col = _infer_date_column(
                    fit.fit_data["original_training_data"],
                    spec_date_col=None,
                    fit_date_col=None
                )

                if date_col == '__index__':
                    train_dates = fit.fit_data["original_training_data"].index.values
                else:
                    train_dates = fit.fit_data["original_training_data"][date_col].values
        except (ValueError, ImportError):
            pass

        if train_dates is not None and len(train_dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(train_dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(train_dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)

        # Add model metadata
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
