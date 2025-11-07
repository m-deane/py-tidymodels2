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
from py_parsnip.utils import _infer_date_column


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
        date_col: Optional date column name for time series models

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
    date_col: Optional[str] = None

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
        # Extract date_col if provided in kwargs
        date_col = kwargs.pop('date_col', None)
        new_args = {**self.args, **kwargs}

        # Update date_col if provided, otherwise keep existing
        if date_col is not None:
            return replace(self, args=new_args, date_col=date_col)
        return replace(self, args=new_args)

    def fit(
        self,
        data: pd.DataFrame,
        formula: Optional[str] = None,
        original_training_data: Optional[pd.DataFrame] = None,
        date_col: Optional[str] = None
    ) -> "ModelFit":
        """
        Fit the model to data.

        Args:
            data: Training data DataFrame
            formula: Optional formula (e.g., "y ~ x1 + x2")
            original_training_data: Original unpreprocessed training data (for raw datetime/categorical values)
            date_col: Optional date column name (for time series models)

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
            import inspect

            # Check if fit_raw accepts date_col parameter
            fit_raw_signature = inspect.signature(engine.fit_raw)
            accepts_date_col = 'date_col' in fit_raw_signature.parameters

            # Build kwargs for fit_raw
            fit_raw_kwargs = {}
            if original_training_data is not None:
                fit_raw_kwargs['original_training_data'] = original_training_data

            # Only infer and pass date_col if engine supports it
            if accepts_date_col:
                # Infer date column using priority-based detection
                # Priority: fit date_col param > spec.date_col > auto-detect
                inferred_date = _infer_date_column(data, self.date_col, date_col)
                fit_raw_kwargs['date_col'] = inferred_date

            # Engine handles data directly without molding
            fit_data, blueprint = engine.fit_raw(
                self, data, formula, **fit_raw_kwargs
            )
        else:
            # Standard molding path
            # Special case: if data is a dict (for custom_data strategy in hybrid_model)
            # skip molding and pass dict directly to engine
            if isinstance(data, dict):
                import inspect
                fit_signature = inspect.signature(engine.fit)
                accepts_original_data = 'original_training_data' in fit_signature.parameters

                if not accepts_original_data:
                    raise ValueError(
                        "Engine does not support dict input (missing original_training_data parameter)"
                    )

                # Validate dict has required keys
                if 'model1' not in data or 'model2' not in data:
                    raise ValueError(
                        "When using dict data, must provide 'model1' and 'model2' keys. "
                        f"Got keys: {list(data.keys())}"
                    )

                # Create blueprint from first dataset in dict for metadata and formula storage
                if isinstance(data['model1'], pd.DataFrame):
                    from py_hardhat import mold as create_mold
                    temp_molded = create_mold(formula, data['model1'])
                    blueprint = temp_molded.blueprint
                else:
                    raise ValueError(
                        "Dict data['model1'] must be a DataFrame for blueprint creation"
                    )

                # Create a minimal MoldedData with just the blueprint (no actual molding)
                # This allows the engine to access the formula via molded.blueprint.formula
                from py_hardhat import MoldedData
                minimal_molded = MoldedData(
                    predictors=None,
                    outcomes=None,
                    blueprint=blueprint,
                    extras={}
                )

                # Pass dict as original_training_data and minimal molded for formula
                fit_data = engine.fit(self, minimal_molded, original_training_data=data)
            elif formula is not None:
                molded = mold(formula, data)
                # Pass original_training_data to engine.fit() for datetime column extraction
                # If not provided, use data itself (direct fit() calls have original data)
                # Check if engine.fit() accepts original_training_data parameter
                import inspect
                fit_signature = inspect.signature(engine.fit)
                accepts_original_data = 'original_training_data' in fit_signature.parameters

                if accepts_original_data:
                    # Pass original_training_data (defaults to data for consistency)
                    orig_data = original_training_data if original_training_data is not None else data
                    fit_data = engine.fit(self, molded, original_training_data=orig_data)
                else:
                    fit_data = engine.fit(self, molded)
                blueprint = molded.blueprint
            else:
                # Assume data is already molded (MoldedData object)
                if isinstance(data, MoldedData):
                    molded = data
                    # Check if engine.fit() accepts original_training_data parameter
                    import inspect
                    fit_signature = inspect.signature(engine.fit)
                    accepts_original_data = 'original_training_data' in fit_signature.parameters

                    if accepts_original_data:
                        orig_data = original_training_data if original_training_data is not None else None
                        fit_data = engine.fit(self, molded, original_training_data=orig_data)
                    else:
                        fit_data = engine.fit(self, molded)
                    blueprint = molded.blueprint
                else:
                    raise ValueError(
                        "Either provide a formula or pass MoldedData directly"
                    )

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
        original_test_data: Optional[pd.DataFrame] = None,
    ) -> "ModelFit":
        """
        Evaluate model on test data with actuals.

        This method stores test predictions and actuals for later extraction
        via extract_outputs(). It enables comprehensive train/test metrics.

        Args:
            test_data: Test data DataFrame with actuals (may be preprocessed)
            outcome_col: Name of outcome column (auto-detected if None)
            original_test_data: Original test data before preprocessing (for raw values)

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

        # Store evaluation results (including original test data for raw values)
        self.evaluation_data["test_data"] = test_data
        self.evaluation_data["test_predictions"] = predictions
        self.evaluation_data["outcome_col"] = outcome_col
        # Store original test data (defaults to test_data for consistency)
        self.evaluation_data["original_test_data"] = original_test_data if original_test_data is not None else test_data

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
