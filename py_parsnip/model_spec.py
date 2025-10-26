"""
ModelSpec and ModelFit: Core model specification and fitted model containers

ModelSpec is an IMMUTABLE specification of a model (type + engine + args).
ModelFit is a MUTABLE container for fitted model artifacts.

This separation ensures:
- Specs can be reused without side effects
- Fitted models contain all necessary artifacts
- Clear distinction between specification and execution
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Any, Optional, Literal
import pandas as pd

from py_hardhat import MoldedData


@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable model specification.

    A ModelSpec defines WHAT model to fit, but doesn't contain fitted artifacts.
    It's frozen (immutable) so it can be safely reused across multiple fits.

    Attributes:
        model_type: Type of model (e.g., "linear_reg", "rand_forest")
        engine: Backend to use (e.g., "sklearn", "statsmodels")
        mode: "regression" or "classification"
        args: Dict of model arguments (e.g., {"penalty": 0.1})

    Example:
        >>> spec = ModelSpec(
        ...     model_type="linear_reg",
        ...     engine="sklearn",
        ...     mode="regression",
        ...     args={"penalty": 0.1, "mixture": 0.5}
        ... )
        >>> # Spec is immutable - use replace() to modify
        >>> new_spec = replace(spec, args={"penalty": 0.2})
    """

    model_type: str
    engine: str = "sklearn"
    mode: Literal["regression", "classification", "unknown"] = "unknown"
    args: Dict[str, Any] = field(default_factory=dict)

    def set_engine(self, engine: str, **engine_args) -> "ModelSpec":
        """
        Set the computational engine.

        Args:
            engine: Engine name (e.g., "sklearn", "statsmodels")
            **engine_args: Engine-specific arguments

        Returns:
            New ModelSpec with engine set
        """
        # Merge engine_args into args
        new_args = {**self.args, **engine_args}
        return replace(self, engine=engine, args=new_args)

    def set_mode(self, mode: Literal["regression", "classification"]) -> "ModelSpec":
        """
        Set the model mode.

        Args:
            mode: "regression" or "classification"

        Returns:
            New ModelSpec with mode set
        """
        return replace(self, mode=mode)

    def set_args(self, **kwargs) -> "ModelSpec":
        """
        Set model arguments.

        Args:
            **kwargs: Model arguments (tidymodels naming convention)

        Returns:
            New ModelSpec with args updated

        Example:
            >>> spec = linear_reg()
            >>> spec = spec.set_args(penalty=0.1, mixture=0.5)
        """
        new_args = {**self.args, **kwargs}
        return replace(self, args=new_args)

    def fit(self, data: pd.DataFrame, formula: Optional[str] = None) -> "ModelFit":
        """
        Fit the model to data.

        Args:
            data: Training data DataFrame
            formula: Optional formula (e.g., "y ~ x1 + x2")

        Returns:
            ModelFit containing fitted model and metadata

        Example:
            >>> spec = linear_reg()
            >>> fit = spec.fit(train_data, "sales ~ price + advertising")
        """
        from py_parsnip.engine_registry import get_engine
        from py_hardhat import mold

        # Get engine first to check if it needs special handling
        engine = get_engine(self.model_type, self.engine)

        # Check if engine has custom fit_raw method (for models like Prophet)
        if hasattr(engine, "fit_raw"):
            # Engine handles data directly without molding
            fit_data, blueprint = engine.fit_raw(self, data, formula)
        else:
            # Standard molding path
            if formula is not None:
                molded = mold(formula, data)
            else:
                # Assume data is already molded (MoldedData object)
                if isinstance(data, MoldedData):
                    molded = data
                else:
                    raise ValueError(
                        "Either provide a formula or pass MoldedData directly"
                    )

            fit_data = engine.fit(self, molded)
            blueprint = molded.blueprint

        # Create ModelFit
        return ModelFit(
            spec=self,
            fit_data=fit_data,
            blueprint=blueprint,
        )


@dataclass
class ModelFit:
    """
    Fitted model container.

    Unlike ModelSpec, ModelFit is MUTABLE and contains fitted artifacts.
    It stores everything needed for prediction and extraction.

    Attributes:
        spec: Original ModelSpec used for fitting
        fit_data: Dict containing engine-specific fitted objects
        blueprint: Blueprint from mold() for consistent prediction
        fit_time: Optional fit time in seconds
        evaluation_data: Dict storing evaluation results from evaluate()
        model_name: Optional name for this model
        model_group_name: Optional group name for organizing models

    Example:
        >>> fit = spec.fit(train_data, "sales ~ price")
        >>> fit = fit.evaluate(test_data)  # Add test data evaluation
        >>> predictions = fit.predict(test_data)
        >>> outputs, coefs, stats = fit.extract_outputs()
    """

    spec: ModelSpec
    fit_data: Dict[str, Any]
    blueprint: Any  # Blueprint from hardhat
    fit_time: Optional[float] = None
    evaluation_data: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None
    model_group_name: Optional[str] = None

    def predict(
        self,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"] = "numeric",
    ) -> pd.DataFrame:
        """
        Predict on new data.

        Args:
            new_data: New data DataFrame
            type: Prediction type:
                - "numeric": Numeric predictions (regression)
                - "class": Class predictions (classification)
                - "prob": Class probabilities (classification)
                - "conf_int": Confidence intervals (if supported)

        Returns:
            DataFrame with predictions

        Example:
            >>> predictions = fit.predict(test_data)
            >>> predictions.head()
                 .pred
            0   123.45
            1   234.56
        """
        from py_parsnip.engine_registry import get_engine
        from py_hardhat import forge

        # Get engine
        engine = get_engine(self.spec.model_type, self.spec.engine)

        # Check if engine uses raw data (like Prophet)
        if hasattr(engine, "predict_raw"):
            # Engine handles data directly without forging
            predictions = engine.predict_raw(self, new_data, type=type)
        else:
            # Standard forging path
            forged = forge(new_data, self.blueprint)
            predictions = engine.predict(self, forged, type=type)

        return predictions

    def extract_fit_engine(self) -> Any:
        """
        Extract the underlying fitted engine object.

        Returns:
            The fitted model object from the engine (e.g., sklearn model)

        Example:
            >>> sklearn_model = fit.extract_fit_engine()
            >>> sklearn_model.coef_
        """
        return self.fit_data.get("model")

    def evaluate(
        self,
        test_data: pd.DataFrame,
        outcome_col: Optional[str] = None,
    ) -> "ModelFit":
        """
        Evaluate model on test data with actuals.

        This method stores test predictions and actuals for later extraction
        via extract_outputs(). It enables comprehensive train/test metrics.

        Args:
            test_data: Test data DataFrame with actuals
            outcome_col: Name of outcome column (auto-detected if None)

        Returns:
            Self (for method chaining)

        Example:
            >>> fit = spec.fit(train, "sales ~ price")
            >>> fit = fit.evaluate(test)  # test has 'sales' column
            >>> outputs, coefs, stats = fit.extract_outputs()  # Now has test metrics
        """
        # Detect outcome column from blueprint if not provided
        if outcome_col is None:
            # For standard hardhat blueprints
            if hasattr(self.blueprint, "roles"):
                outcome_col = self.blueprint.roles.get("outcome", [None])[0]
            # For raw data models (Prophet, ARIMA) that use dict blueprints
            elif isinstance(self.blueprint, dict) and "outcome_name" in self.blueprint:
                outcome_col = self.blueprint["outcome_name"]
            # For other blueprints with outcome_name attribute
            elif hasattr(self.blueprint, "outcome_name"):
                outcome_col = self.blueprint.outcome_name
            else:
                raise ValueError("Cannot auto-detect outcome column. Please specify outcome_col.")

        # Strip one-hot encoding suffix from outcome column name
        # e.g., "species[setosa]" -> "species"
        if outcome_col and "[" in outcome_col:
            outcome_col = outcome_col.split("[")[0]

        if outcome_col not in test_data.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in test_data")

        # Determine prediction type based on mode
        if self.spec.mode == "regression":
            pred_type = "numeric"
        elif self.spec.mode == "classification":
            pred_type = "class"
        else:
            # Default to numeric for unknown mode
            pred_type = "numeric"

        # Make predictions on test data
        predictions = self.predict(test_data, type=pred_type)

        # Store evaluation results
        self.evaluation_data["test_data"] = test_data
        self.evaluation_data["test_predictions"] = predictions
        self.evaluation_data["outcome_col"] = outcome_col

        return self

    def extract_outputs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract standardized three-DataFrame output.

        Returns:
            Tuple of (outputs, coefficients, stats) DataFrames

        The Outputs DataFrame contains observation-level results with:
        - date (for time series)
        - actuals
        - fitted (training) / forecast (test)
        - residuals
        - split (train/test/forecast)

        Example:
            >>> outputs, coefs, stats = fit.extract_outputs()
            >>> print(outputs)  # Observation-level results
            >>> print(coefs)  # Coefficients with p-values, CI
            >>> print(stats)  # Metrics by split
        """
        from py_parsnip.engine_registry import get_engine

        engine = get_engine(self.spec.model_type, self.spec.engine)
        return engine.extract_outputs(self)
