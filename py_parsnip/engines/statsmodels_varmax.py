"""
Statsmodels engine for VARMAX regression

VARMAX (Vector AutoRegressive Moving Average with eXogenous variables) for
multivariate time series where multiple dependent variables influence each other.
"""

from typing import Dict, Any, Literal
import pandas as pd
import numpy as np

from py_parsnip.engine_registry import Engine, register_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData
from py_parsnip.utils.time_series_utils import _infer_date_column, _parse_ts_formula


@register_engine("varmax_reg", "statsmodels")
class StatsmodelsVARMAXEngine(Engine):
    """
    Statsmodels engine for VARMAX models.

    Parameter mapping:
    - non_seasonal_ar → order[0] (p)
    - non_seasonal_ma → order[1] (q)
    - trend → trend parameter
    """

    param_map = {
        "non_seasonal_ar": "p",
        "non_seasonal_ma": "q",
        "trend": "trend",
    }

    def fit_raw(
        self, spec: ModelSpec, data: pd.DataFrame, formula: str, date_col: str
    ) -> tuple[Dict[str, Any], Any]:
        """
        Fit VARMAX model using statsmodels.

        Args:
            spec: ModelSpec with VARMAX configuration
            data: Training data DataFrame
            formula: Formula like "y1 + y2 ~ x1 + x2" (multiple outcomes)
            date_col: Name of date column or '__index__' for DatetimeIndex

        Returns:
            Tuple of (fit_data dict, blueprint)
        """
        from statsmodels.tsa.statespace.varmax import VARMAX

        # Parse formula - VARMAX has multiple outcomes, so we need custom parsing
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(f"Invalid formula: {formula}")

        outcome_part = parts[0].strip()
        predictor_part = parts[1].strip()

        # Parse multiple outcomes
        outcome_names = [o.strip() for o in outcome_part.split("+")]
        if len(outcome_names) < 2:
            raise ValueError(
                "VARMAX requires at least 2 outcome variables. "
                f"Got: {outcome_names}. Use 'y1 + y2 ~ ...' formula syntax."
            )

        # Validate outcomes exist
        missing = [o for o in outcome_names if o not in data.columns]
        if missing:
            raise ValueError(f"Outcomes {missing} not found in data")

        # Get outcome matrix
        y = data[outcome_names]

        # Parse exogenous variables (excluding date column)
        # We use _parse_ts_formula with first outcome for consistency
        _, exog_vars = _parse_ts_formula(f"{outcome_names[0]} ~ {predictor_part}", date_col)

        # Handle exogenous variables and time index
        if date_col == '__index__':
            # Data is already indexed by datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(
                    f"date_col is '__index__' but data does not have DatetimeIndex. "
                    f"Got index type: {type(data.index).__name__}"
                )
            y = data[outcome_names]
            if exog_vars:
                missing = [p for p in exog_vars if p not in data.columns]
                if missing:
                    raise ValueError(f"Exogenous variables {missing} not found in data")
                exog = data[exog_vars]
            else:
                exog = None
        elif date_col is not None:
            # Set datetime column as index
            y = data.set_index(date_col)[outcome_names]
            if exog_vars:
                missing = [p for p in exog_vars if p not in data.columns]
                if missing:
                    raise ValueError(f"Exogenous variables {missing} not found in data")
                exog = data.set_index(date_col)[exog_vars]
            else:
                exog = None
        else:
            # No date column
            if exog_vars:
                missing = [p for p in exog_vars if p not in data.columns]
                if missing:
                    raise ValueError(f"Exogenous variables {missing} not found in data")
                exog = data[exog_vars]
            else:
                exog = None

        # Get VARMAX parameters
        args = spec.args
        order = (
            args.get("non_seasonal_ar", 1),
            args.get("non_seasonal_ma", 0)
        )
        trend = args.get("trend", "c")

        # Create and fit VARMAX model
        model = VARMAX(y, exog=exog, order=order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit(disp=False)

        # Get fitted values - this will be a DataFrame with multiple columns
        fitted_values = fitted_model.fittedvalues
        actuals = y.values
        residuals = actuals - fitted_values.values

        # Extract dates
        if date_col == '__index__':
            dates = data.index.values
        elif date_col is not None:
            dates = data[date_col].values
        else:
            # Fallback to integer index
            dates = np.arange(len(y))

        blueprint = {
            "formula": formula,
            "outcome_names": outcome_names,
            "predictor_names": exog_vars if exog_vars else [],
            "date_col": date_col,
            "order": order,
            "trend": trend,
        }

        fit_data = {
            "model": fitted_model,
            "outcome_names": outcome_names,
            "predictor_names": exog_vars if exog_vars else [],
            "order": order,
            "trend": trend,
            "n_obs": len(y),
            "n_outcomes": len(outcome_names),
            "y_train": actuals,
            "fitted": fitted_values.values,
            "residuals": residuals,
            "dates": dates,
        }

        return fit_data, blueprint

    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """Not used for VARMAX - use fit_raw() instead"""
        raise NotImplementedError("VARMAX uses fit_raw() instead of fit()")

    def predict(self, fit: ModelFit, molded: MoldedData, type: str) -> pd.DataFrame:
        """Not used for VARMAX - use predict_raw() instead"""
        raise NotImplementedError("VARMAX uses predict_raw() instead of predict()")

    def predict_raw(
        self,
        fit: ModelFit,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions using fitted VARMAX model.

        Args:
            fit: ModelFit with fitted VARMAX model
            new_data: DataFrame with exogenous variables if needed
            type: Prediction type ('numeric' or 'conf_int')

        Returns:
            DataFrame with predictions for all outcome variables
        """
        if type not in ("numeric", "conf_int"):
            raise ValueError(f"varmax_reg supports type='numeric' or 'conf_int', got '{type}'")

        model = fit.fit_data["model"]
        predictor_names = fit.fit_data["predictor_names"]
        outcome_names = fit.fit_data["outcome_names"]
        date_col = fit.blueprint.get("date_col") if isinstance(fit.blueprint, dict) else None

        # predictor_names already excludes date_col (handled in fit_raw via _parse_ts_formula)
        exog_predictor_names = predictor_names

        n_periods = len(new_data)

        # Get exogenous variables
        if exog_predictor_names:
            missing = [p for p in exog_predictor_names if p not in new_data.columns]
            if missing:
                raise ValueError(f"Exogenous variables {missing} not found in new_data")
            exog = new_data[exog_predictor_names]
        else:
            exog = None

        # Extract date index
        if date_col == '__index__':
            date_index = new_data.index
        elif date_col and date_col in new_data.columns:
            date_index = new_data[date_col]
        else:
            date_index = None

        if type == "numeric":
            forecast = model.forecast(steps=n_periods, exog=exog)
            # forecast will be a DataFrame with columns for each outcome
            result = pd.DataFrame(forecast.values, columns=[f".pred_{name}" for name in outcome_names])
            if date_index is not None:
                result.index = date_index
            return result
        else:  # conf_int
            forecast_obj = model.get_forecast(steps=n_periods, exog=exog)
            pred_mean = forecast_obj.predicted_mean
            pred_int = forecast_obj.conf_int(alpha=0.05)

            # Create columns for each outcome with mean and intervals
            result_dict = {}
            for i, name in enumerate(outcome_names):
                result_dict[f".pred_{name}"] = pred_mean.iloc[:, i].values
                result_dict[f".pred_{name}_lower"] = pred_int.iloc[:, i*2].values
                result_dict[f".pred_{name}_upper"] = pred_int.iloc[:, i*2+1].values

            result = pd.DataFrame(result_dict)
            if date_index is not None:
                result.index = date_index
            return result

    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract comprehensive three-DataFrame output for VARMAX"""
        model = fit.fit_data["model"]
        outcome_names = fit.fit_data["outcome_names"]
        n_outcomes = fit.fit_data["n_outcomes"]

        # OUTPUTS DataFrame
        outputs_list = []
        y_train = fit.fit_data.get("y_train")
        fitted = fit.fit_data.get("fitted")
        residuals = fit.fit_data.get("residuals")
        dates = fit.fit_data.get("dates")

        if y_train is not None and fitted is not None:
            for i, outcome_name in enumerate(outcome_names):
                train_df = pd.DataFrame({
                    "outcome_variable": outcome_name,
                    "date": dates if dates is not None else np.arange(len(y_train)),
                    "actuals": y_train[:, i],
                    "fitted": fitted[:, i],
                    "residuals": residuals[:, i] if residuals is not None else y_train[:, i] - fitted[:, i],
                    "split": "train",
                })
                train_df["model"] = fit.model_name if fit.model_name else fit.spec.model_type
                train_df["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
                train_df["group"] = "global"
                outputs_list.append(train_df)

        outputs = pd.concat(outputs_list, ignore_index=True) if outputs_list else pd.DataFrame()

        # COEFFICIENTS DataFrame
        if hasattr(model, "params") and model.params is not None:
            param_names = model.param_names if hasattr(model, "param_names") else list(model.params.index)
            coefficients = pd.DataFrame({
                "variable": param_names,
                "coefficient": model.params.values,
                "std_error": model.bse.values if hasattr(model, "bse") else [np.nan] * len(param_names),
                "t_stat": model.tvalues.values if hasattr(model, "tvalues") else [np.nan] * len(param_names),
                "p_value": model.pvalues.values if hasattr(model, "pvalues") else [np.nan] * len(param_names),
                "ci_0.025": [np.nan] * len(param_names),
                "ci_0.975": [np.nan] * len(param_names),
                "vif": [np.nan] * len(param_names),
            })
        else:
            coefficients = pd.DataFrame()

        coefficients["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        coefficients["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        coefficients["group"] = "global"

        # STATS DataFrame
        stats_rows = [
            {"metric": "formula", "value": fit.blueprint.get("formula", "") if isinstance(fit.blueprint, dict) else "", "split": ""},
            {"metric": "model_type", "value": fit.spec.model_type, "split": ""},
            {"metric": "order", "value": str(fit.fit_data.get("order", (1, 0))), "split": ""},
            {"metric": "trend", "value": fit.fit_data.get("trend", "c"), "split": ""},
            {"metric": "n_outcomes", "value": n_outcomes, "split": ""},
            {"metric": "n_obs_train", "value": fit.fit_data.get("n_obs", 0), "split": "train"},
            {"metric": "aic", "value": model.aic if hasattr(model, "aic") else np.nan, "split": ""},
            {"metric": "bic", "value": model.bic if hasattr(model, "bic") else np.nan, "split": ""},
        ]

        # Add training date range
        dates = fit.fit_data.get("dates")
        if dates is not None and len(dates) > 0:
            stats_rows.extend([
                {"metric": "train_start_date", "value": str(dates[0]), "split": "train"},
                {"metric": "train_end_date", "value": str(dates[-1]), "split": "train"},
            ])

        stats = pd.DataFrame(stats_rows)
        stats["model"] = fit.model_name if fit.model_name else fit.spec.model_type
        stats["model_group_name"] = fit.model_group_name if fit.model_group_name else ""
        stats["group"] = "global"

        return outputs, coefficients, stats
