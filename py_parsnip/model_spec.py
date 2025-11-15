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
from typing import Dict, Any, Optional, Literal, Tuple
import pandas as pd
import numpy as np
import warnings

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

            # Check if fit_raw accepts optional parameters
            fit_raw_signature = inspect.signature(engine.fit_raw)
            accepts_date_col = 'date_col' in fit_raw_signature.parameters
            accepts_original_data = 'original_training_data' in fit_raw_signature.parameters

            # Build kwargs for fit_raw
            fit_raw_kwargs = {}
            if original_training_data is not None and accepts_original_data:
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
                # Expand dot notation before molding (if present)
                # This prevents datetime columns from being included, which causes
                # patsy to treat them as categorical and fail on new dates in test data
                if ' . ' in formula or formula.endswith(' .') or ' ~ .' in formula:
                    # Parse formula to extract outcome
                    if '~' in formula:
                        outcome_str, predictor_str = formula.split('~', 1)
                        outcome_str = outcome_str.strip()
                        predictor_str = predictor_str.strip()

                        # Check if using dot notation
                        if predictor_str == '.' or predictor_str.startswith('. +') or ' + .' in predictor_str:
                            # Expand dot notation to all columns except outcome and datetime
                            # Get all columns except outcome
                            all_cols = [col for col in data.columns if col != outcome_str]

                            # Exclude datetime columns (they cause patsy categorical errors on new dates)
                            predictor_cols = [
                                col for col in all_cols
                                if not pd.api.types.is_datetime64_any_dtype(data[col])
                            ]

                            # Handle different dot notation patterns
                            if predictor_str == '.':
                                # Pure dot notation: "y ~ ."
                                expanded_formula = f"{outcome_str} ~ {' + '.join(predictor_cols)}"
                            elif predictor_str.startswith('. +'):
                                # Dot notation with additions: "y ~ . + I(x1*x2)"
                                extra_terms = predictor_str[2:].strip()  # Remove ". +"
                                if predictor_cols:
                                    expanded_formula = f"{outcome_str} ~ {' + '.join(predictor_cols)} + {extra_terms}"
                                else:
                                    expanded_formula = f"{outcome_str} ~ {extra_terms}"
                            elif ' + .' in predictor_str:
                                # Additions before dot: "y ~ x1 + ."
                                prefix_terms = predictor_str.split(' + .')[0].strip()
                                if predictor_cols:
                                    expanded_formula = f"{outcome_str} ~ {prefix_terms} + {' + '.join(predictor_cols)}"
                                else:
                                    expanded_formula = f"{outcome_str} ~ {prefix_terms}"
                            else:
                                # Shouldn't reach here, but use original formula
                                expanded_formula = formula

                            formula = expanded_formula

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

    def fit_nested(
        self,
        data: pd.DataFrame,
        formula: str,
        group_col: str,
        original_training_data: Optional[pd.DataFrame] = None,
        date_col: Optional[str] = None
    ) -> "NestedModelFit":
        """
        Fit separate models for each group in the data (panel/grouped modeling).

        This method enables grouped/panel modeling directly on ModelSpec without
        requiring a workflow wrapper. It fits one independent model per group value,
        useful for:
        - Multi-store sales forecasting (one model per store)
        - Multi-product demand forecasting (one model per product)
        - Multi-region time series (one model per region)

        Args:
            data: Training data DataFrame with group column
            formula: Model formula (e.g., "y ~ x1 + x2")
            group_col: Column name containing group identifiers
            original_training_data: Original unpreprocessed training data (optional)
            date_col: Optional date column name (for time series models)

        Returns:
            NestedModelFit containing dict of fitted models per group

        Raises:
            ValueError: If group_col not in data

        Examples:
            >>> # Fit separate models for each store
            >>> spec = linear_reg()
            >>> nested_fit = spec.fit_nested(
            ...     data,
            ...     "sales ~ date + price",
            ...     group_col="store_id"
            ... )
            >>>
            >>> # Predict for all groups
            >>> predictions = nested_fit.predict(test_data)
            >>>
            >>> # Extract outputs with group column
            >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        """
        # Validate group column exists
        if group_col not in data.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")

        # Get unique groups
        groups = data[group_col].unique()

        # Warn if only one group
        if len(groups) == 1:
            warnings.warn(
                f"Only one group found in '{group_col}'. Consider using fit() instead of fit_nested().",
                UserWarning
            )

        # Fit separate model for each group
        group_fits = {}
        group_train_data = {}  # Store original training data per group

        for group in groups:
            group_data = data[data[group_col] == group].copy()

            # Store original group training data (before dropping group column)
            # This preserves date column for later extraction
            group_train_data[group] = group_data.copy()

            # For recursive models, set date as index if needed
            is_recursive = self.model_type == "recursive_reg"
            if is_recursive and "date" in group_data.columns and not isinstance(group_data.index, pd.DatetimeIndex):
                # Set date as index before removing group column (recursive models need this)
                group_data = group_data.set_index("date")
                group_data = group_data.drop(columns=[group_col])
            else:
                # Remove group column before fitting (it's not a predictor)
                group_data = group_data.drop(columns=[group_col])

            # Fit this group's model
            group_fits[group] = self.fit(
                group_data,
                formula,
                original_training_data=original_training_data,
                date_col=date_col
            )

        return NestedModelFit(
            spec=self,
            group_col=group_col,
            group_fits=group_fits,
            formula=formula,
            group_train_data=group_train_data
        )

    def fit_global(
        self,
        data: pd.DataFrame,
        formula: str,
        group_col: str,
        original_training_data: Optional[pd.DataFrame] = None,
        date_col: Optional[str] = None
    ) -> "ModelFit":
        """
        Fit a single global model using group as a feature.

        This method fits one model using all groups together, with the group
        column as an additional predictor. Useful when:
        - Groups share common patterns
        - Insufficient data per group for separate models
        - Want to capture cross-group effects

        Args:
            data: Training data DataFrame with group column
            formula: Model formula (e.g., "sales ~ price + advertising")
            group_col: Column name containing group identifiers (used as feature)
            original_training_data: Original unpreprocessed training data (optional)
            date_col: Optional date column name (for time series models)

        Returns:
            ModelFit with group column included as predictor

        Raises:
            ValueError: If group_col not in data or invalid formula format

        Examples:
            >>> # Fit single model using store_id as a feature
            >>> spec = linear_reg()
            >>> global_fit = spec.fit_global(
            ...     data,
            ...     "sales ~ price + advertising",
            ...     group_col="store_id"
            ... )
            >>>
            >>> # Group column is automatically included as predictor
            >>> predictions = global_fit.predict(test_data)
        """
        # Validate group column exists
        if group_col not in data.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")

        # Validate formula structure
        if '~' not in formula:
            raise ValueError(f"Invalid formula format: {formula}. Formula must contain '~'")

        # Parse formula to add group column
        lhs, rhs = formula.split('~', 1)
        outcome = lhs.strip()
        predictors = rhs.strip()

        # Check if group_col is already in the formula
        predictor_terms = [term.strip() for term in predictors.split('+')]

        if group_col in predictor_terms or predictors == ".":
            # group_col already in formula or using "." notation (includes all columns)
            updated_formula = formula
        else:
            # Add group_col explicitly
            updated_formula = f"{outcome} ~ {predictors} + {group_col}"

        # Fit with updated formula
        return self.fit(
            data,
            updated_formula,
            original_training_data=original_training_data,
            date_col=date_col
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
        outputs, coefficients, stats = engine.extract_outputs(self)

        # Reorder columns for consistent ordering: date first, then core columns
        from py_parsnip.utils.output_ordering import (
            reorder_outputs_columns,
            reorder_coefficients_columns,
            reorder_stats_columns
        )

        outputs = reorder_outputs_columns(outputs, group_col=None)
        coefficients = reorder_coefficients_columns(coefficients, group_col=None)
        stats = reorder_stats_columns(stats, group_col=None)

        return outputs, coefficients, stats

    def explain(
        self,
        data: pd.DataFrame,
        method: Literal["auto", "tree", "linear", "kernel"] = "auto",
        background_size: int = 100,
        background: Literal["sample", "kmeans"] = "sample",
        background_data: Optional[pd.DataFrame] = None,
        check_additivity: bool = True
    ) -> pd.DataFrame:
        """
        Compute SHAP values to explain model predictions.

        Uses SHAP (SHapley Additive exPlanations) to compute feature contributions
        for each observation. Auto-selects best explainer based on model type.

        Args:
            data: Data to explain (must contain all features used in model)
            method: Explainer method:
                - "auto": Auto-select based on model type (default)
                - "tree": TreeExplainer (for tree-based models)
                - "linear": LinearExplainer (for linear models)
                - "kernel": KernelExplainer (model-agnostic, slower)
            background_size: Number of background samples for KernelExplainer
            background: Background sampling strategy ("sample" or "kmeans")
            background_data: Custom background data (overrides background_size)
            check_additivity: Verify SHAP values sum to prediction - base_value

        Returns:
            DataFrame with SHAP values per variable per observation.
            Columns: observation_id, variable, shap_value, abs_shap, feature_value,
                     base_value, prediction, model, model_group_name

        Examples:
            >>> # Auto-select best explainer
            >>> shap_df = fit.explain(test_data)
            >>>
            >>> # Global feature importance
            >>> importance = shap_df.groupby("variable")["abs_shap"].mean()
            >>>
            >>> # Force specific method
            >>> shap_df = fit.explain(test_data, method="kernel")
        """
        from py_interpret import ShapEngine

        return ShapEngine.explain(
            fit=self,
            data=data,
            method=method,
            background_size=background_size,
            background=background,
            background_data=background_data,
            check_additivity=check_additivity
        )

    def explain_plot(
        self,
        data: pd.DataFrame,
        plot_type: Literal["summary", "waterfall", "force", "dependence", "temporal"] = "summary",
        observation_id: Optional[int] = None,
        feature: Optional[str] = None,
        method: Literal["auto", "tree", "linear", "kernel"] = "auto",
        **kwargs
    ):
        """
        Generate SHAP visualization for model interpretability.

        Convenience method that combines explain() with visualization functions.
        Supports 5 plot types for different analysis needs.

        Args:
            data: Data to explain
            plot_type: Type of visualization:
                - "summary": Global feature importance (beeswarm or bar)
                - "waterfall": Local explanation for single observation
                - "force": Interactive force plot for single observation
                - "dependence": Partial dependence for single feature
                - "temporal": SHAP values over time (requires date column)
            observation_id: Required for waterfall/force plots (which observation to explain)
            feature: Required for dependence plots (which feature to analyze)
            method: SHAP explainer method ("auto", "tree", "linear", "kernel")
            **kwargs: Additional arguments passed to plotting function:
                - summary: plot_type="beeswarm"|"bar", max_display=20
                - waterfall: max_display=10
                - force: matplotlib=False
                - dependence: interaction_feature="auto"
                - temporal: features=None, aggregation="mean", plot_type="line"

        Returns:
            matplotlib Figure object (or HTML for force plot with matplotlib=False)

        Raises:
            ValueError: If required arguments missing (e.g., observation_id for waterfall)

        Examples:
            >>> # Global feature importance
            >>> fig = fit.explain_plot(test_data, plot_type="summary")
            >>>
            >>> # Explain single prediction
            >>> fig = fit.explain_plot(test_data, plot_type="waterfall", observation_id=0)
            >>>
            >>> # Partial dependence
            >>> fig = fit.explain_plot(test_data, plot_type="dependence", feature="temperature")
            >>>
            >>> # Time series evolution
            >>> fig = fit.explain_plot(test_data, plot_type="temporal")
        """
        from py_interpret.visualizations import (
            summary_plot,
            waterfall_plot,
            force_plot,
            dependence_plot,
            temporal_plot
        )

        # Get SHAP values
        shap_df = self.explain(data, method=method)

        # Route to appropriate plot function
        if plot_type == "summary":
            return summary_plot(shap_df, **kwargs)

        elif plot_type == "waterfall":
            if observation_id is None:
                raise ValueError(
                    "observation_id required for waterfall plot. "
                    "Specify which observation to explain (e.g., observation_id=0)"
                )
            return waterfall_plot(shap_df, observation_id=observation_id, **kwargs)

        elif plot_type == "force":
            if observation_id is None:
                raise ValueError(
                    "observation_id required for force plot. "
                    "Specify which observation to explain (e.g., observation_id=0)"
                )
            return force_plot(shap_df, observation_id=observation_id, **kwargs)

        elif plot_type == "dependence":
            if feature is None:
                raise ValueError(
                    "feature required for dependence plot. "
                    "Specify which feature to analyze (e.g., feature='temperature')"
                )
            return dependence_plot(shap_df, feature=feature, **kwargs)

        elif plot_type == "temporal":
            if "date" not in data.columns:
                raise ValueError(
                    "Temporal plot requires 'date' column in data. "
                    "Ensure your data has a date/datetime column."
                )
            return temporal_plot(shap_df, **kwargs)

        else:
            raise ValueError(
                f"Unknown plot_type: {plot_type}. "
                f"Must be 'summary', 'waterfall', 'force', 'dependence', or 'temporal'"
            )

    def explain_interactions(
        self,
        data: pd.DataFrame,
        background_size: int = 100,
        background: Literal["sample", "kmeans"] = "sample",
        background_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Compute SHAP interaction values showing feature pair interactions.

        Only works with tree-based models (rand_forest, decision_tree, boost_tree).
        Returns 3D array where interaction_values[i, j, k] is the interaction
        between features j and k for observation i.

        Args:
            data: Data to explain
            background_size: Number of background samples (for fallback methods)
            background: Background sampling strategy ("sample" or "kmeans")
            background_data: Custom background data (optional)

        Returns:
            3D numpy array of shape (n_observations, n_features, n_features)

        Raises:
            NotImplementedError: If model type doesn't support interaction values

        Examples:
            >>> # Compute interactions for random forest
            >>> spec = rand_forest().set_mode('regression')
            >>> fit = spec.fit(train_data, 'y ~ x1 + x2 + x3')
            >>> interactions = fit.explain_interactions(test_data)
            >>>
            >>> # Get interaction between x1 and x2 for first observation
            >>> x1_x2_interaction = interactions[0, 0, 1]
        """
        from py_interpret import ShapEngine

        return ShapEngine.explain_interactions(
            fit=self,
            data=data,
            background_size=background_size,
            background=background,
            background_data=background_data
        )

    def save_mlflow(
        self,
        path: str,
        conda_env: Optional[Any] = None,
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model to MLflow format for versioning and deployment.

        This method provides a convenient way to persist the ModelFit object
        using MLflow's model format, which enables:
        - Model versioning and lineage tracking
        - Deployment to various serving platforms
        - Model registry integration
        - Reproducible model artifacts

        Args:
            path: Directory path where model will be saved
            conda_env: Conda environment specification (optional)
            signature: Model signature for input/output schema validation.
                      Use "auto" to infer from input_example.
            input_example: Example input DataFrame for signature inference (optional)
            registered_model_name: Name to register model in MLflow Model Registry (optional)
            metadata: Additional custom metadata dict (optional)

        Returns:
            None

        Examples:
            >>> # Basic save
            >>> fit = spec.fit(train_data, "y ~ x1 + x2")
            >>> fit.save_mlflow("models/my_model")
            >>>
            >>> # Save with auto signature and registry
            >>> fit.save_mlflow(
            ...     path="models/my_model",
            ...     input_example=train_data.head(5),
            ...     signature="auto",
            ...     registered_model_name="MyModel"
            ... )
            >>>
            >>> # Load later
            >>> from py_mlflow import load_model
            >>> loaded = load_model("models/my_model")
            >>> predictions = loaded.predict(test_data)
        """
        from py_mlflow import save_model

        save_model(
            model=self,
            path=path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            metadata=metadata
        )


@dataclass
class NestedModelFit:
    """
    Fitted model with separate fits for each group (panel/grouped modeling).

    This class is parallel to NestedWorkflowFit but operates at the ModelSpec level,
    enabling grouped/panel modeling without requiring workflow wrapper.
    It holds multiple fitted models, one per group, while maintaining a unified interface.

    Attributes:
        spec: Original ModelSpec specification
        group_col: Column name containing group identifiers
        group_fits: Dict mapping group values to ModelFit objects
        formula: Formula used for fitting
        group_train_data: Dict mapping group values to original training data (for date extraction)

    Examples:
        >>> spec = linear_reg()
        >>> nested_fit = spec.fit_nested(data, "sales ~ date", group_col="store_id")
        >>>
        >>> # Predict for all groups
        >>> predictions = nested_fit.predict(test_data)
        >>>
        >>> # Extract outputs with group column
        >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        >>> print(outputs[["date", group_col, "actuals", "forecast"]])
    """
    spec: ModelSpec
    group_col: str
    group_fits: Dict[Any, ModelFit]
    formula: str
    group_train_data: Dict[Any, pd.DataFrame]

    def predict(
        self,
        new_data: pd.DataFrame,
        type: Literal["numeric", "class", "prob", "conf_int"] = "numeric"
    ) -> pd.DataFrame:
        """
        Generate predictions for all groups.

        Automatically routes each row to the appropriate group model.

        Args:
            new_data: New data with group column
            type: Prediction type ("numeric", "conf_int", "pred_int", etc.)

        Returns:
            DataFrame with predictions and group column

        Raises:
            ValueError: If group_col not in new_data or no matching groups found

        Examples:
            >>> predictions = nested_fit.predict(test_data)
            >>> predictions = nested_fit.predict(test_data, type="conf_int")
        """
        if self.group_col not in new_data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in new_data")

        # Get predictions for each group
        all_predictions = []
        is_recursive = self.spec.model_type == "recursive_reg"

        for group, group_fit in self.group_fits.items():
            # Filter data for this group
            group_data = new_data[new_data[self.group_col] == group].copy()

            if len(group_data) == 0:
                continue  # Skip groups not in new_data

            # For recursive models, set date as index if needed
            if is_recursive and "date" in group_data.columns and not isinstance(group_data.index, pd.DatetimeIndex):
                # Set date as index before removing group column (recursive models need this)
                group_data = group_data.set_index("date")
                group_data_no_group = group_data.drop(columns=[self.group_col])
            else:
                # Remove group column before prediction
                group_data_no_group = group_data.drop(columns=[self.group_col])

            # Get predictions
            group_preds = group_fit.predict(group_data_no_group, type=type)

            # Add group column back
            group_preds[self.group_col] = group

            all_predictions.append(group_preds)

        if len(all_predictions) == 0:
            raise ValueError("No matching groups found in new_data")

        # Combine predictions from all groups
        return pd.concat(all_predictions, ignore_index=True)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        outcome_col: Optional[str] = None
    ) -> "NestedModelFit":
        """
        Evaluate all group models on test data.

        Args:
            test_data: Test data with actual outcomes and group column
            outcome_col: Name of outcome column (auto-detected if None)

        Returns:
            Self for method chaining

        Examples:
            >>> nested_fit = spec.fit_nested(train, "y ~ x", "store_id")
            >>> nested_fit = nested_fit.evaluate(test)
            >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        """
        if self.group_col not in test_data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in test_data")

        # Evaluate each group model
        is_recursive = self.spec.model_type == "recursive_reg"

        for group, group_fit in self.group_fits.items():
            # Filter data for this group
            group_data = test_data[test_data[self.group_col] == group].copy()

            if len(group_data) == 0:
                continue  # Skip groups not in test_data

            # For recursive models, set date as index if needed
            if is_recursive and "date" in group_data.columns and not isinstance(group_data.index, pd.DatetimeIndex):
                # Set date as index before removing group column (recursive models need this)
                group_data = group_data.set_index("date")
                group_data_no_group = group_data.drop(columns=[self.group_col])
            else:
                # Remove group column before evaluation
                group_data_no_group = group_data.drop(columns=[self.group_col])

            # Evaluate this group's model
            self.group_fits[group] = group_fit.evaluate(group_data_no_group, outcome_col)

        return self

    def extract_outputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame outputs for all groups.

        Combines outputs from all group models and adds group column.

        Returns:
            Tuple of (outputs, coefficients, stats) DataFrames
            - outputs: Includes group_col showing which group each row belongs to
            - coefficients: Includes group_col
            - stats: Includes group_col

        Examples:
            >>> outputs, coefficients, stats = nested_fit.extract_outputs()
            >>>
            >>> # Filter to specific group
            >>> store_a_outputs = outputs[outputs[group_col] == "A"]
            >>>
            >>> # Compare metrics across groups
            >>> test_rmse = stats[
            ...     (stats["metric"] == "rmse") &
            ...     (stats["split"] == "test")
            ... ][[group_col, "value"]]
        """
        all_outputs = []
        all_coefficients = []
        all_stats = []

        for group, group_fit in self.group_fits.items():
            # Extract outputs for this group
            outputs, coefficients, stats = group_fit.extract_outputs()

            # Preserve date information if available
            # This is needed for plot_forecast() to work
            if "date" not in outputs.columns:
                # Make a copy to avoid SettingWithCopyWarning
                outputs = outputs.copy()

                # Try to get training dates from stored original training data
                if hasattr(self, 'group_train_data') and group in self.group_train_data:
                    train_data_orig = self.group_train_data[group]
                    if "date" in train_data_orig.columns:
                        train_dates = train_data_orig["date"].values
                        train_mask = outputs['split'] == 'train'
                        if train_mask.sum() == len(train_dates):
                            outputs.loc[train_mask, 'date'] = train_dates
                # Fallback: Try to get date from the fit's molded data (if date was used as index)
                elif hasattr(group_fit, 'molded') and group_fit.molded is not None:
                    molded_outcomes = group_fit.molded.outcomes
                    if isinstance(molded_outcomes, pd.DataFrame) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
                        # Date is in the outcomes index
                        date_index = molded_outcomes.index
                        train_mask = outputs['split'] == 'train'
                        if train_mask.sum() == len(date_index):
                            outputs.loc[train_mask, 'date'] = date_index.values
                    elif isinstance(molded_outcomes, pd.Series) and isinstance(molded_outcomes.index, pd.DatetimeIndex):
                        # Outcomes is a Series with date index
                        date_index = molded_outcomes.index
                        train_mask = outputs['split'] == 'train'
                        if train_mask.sum() == len(date_index):
                            outputs.loc[train_mask, 'date'] = date_index.values

                # Try to get test dates from evaluation data
                if hasattr(group_fit, 'evaluation_data') and "test_data" in group_fit.evaluation_data:
                    test_data = group_fit.evaluation_data["test_data"]
                    if "date" in test_data.columns:
                        test_dates = test_data["date"].values
                        test_mask = outputs['split'] == 'test'
                        if test_mask.sum() == len(test_dates):
                            outputs.loc[test_mask, 'date'] = test_dates
                    elif isinstance(test_data.index, pd.DatetimeIndex):
                        test_dates = test_data.index.values
                        test_mask = outputs['split'] == 'test'
                        if test_mask.sum() == len(test_dates):
                            outputs.loc[test_mask, 'date'] = test_dates

            # Add group column
            outputs[self.group_col] = group
            coefficients[self.group_col] = group
            stats[self.group_col] = group

            all_outputs.append(outputs)
            all_coefficients.append(coefficients)
            all_stats.append(stats)

        # Combine all groups (preserve dates if present)
        combined_outputs = pd.concat(all_outputs, ignore_index=True)
        combined_coefficients = pd.concat(all_coefficients, ignore_index=True)
        combined_stats = pd.concat(all_stats, ignore_index=True)

        # Reorder columns for consistent ordering: date first, group second, then core columns
        from py_parsnip.utils.output_ordering import (
            reorder_outputs_columns,
            reorder_coefficients_columns,
            reorder_stats_columns
        )

        combined_outputs = reorder_outputs_columns(combined_outputs, group_col=self.group_col)
        combined_coefficients = reorder_coefficients_columns(combined_coefficients, group_col=self.group_col)
        combined_stats = reorder_stats_columns(combined_stats, group_col=self.group_col)

        return combined_outputs, combined_coefficients, combined_stats

    def explain(
        self,
        data: pd.DataFrame,
        method: Literal["auto", "tree", "linear", "kernel"] = "auto",
        background_size: int = 100,
        background: Literal["sample", "kmeans"] = "sample",
        background_data: Optional[pd.DataFrame] = None,
        check_additivity: bool = True
    ) -> pd.DataFrame:
        """
        Compute SHAP values for all group models.

        Computes SHAP values separately for each group and combines results
        with group column for comparison across groups.

        Args:
            data: Data to explain (must include group column)
            method: Explainer method ("auto", "tree", "linear", or "kernel")
            background_size: Number of background samples for KernelExplainer
            background: Background sampling strategy ("sample" or "kmeans")
            background_data: Custom background data (overrides background_size)
            check_additivity: Verify SHAP values sum to prediction - base_value

        Returns:
            DataFrame with SHAP values per variable per observation per group.
            Includes "group" column to identify which group each row belongs to.

        Examples:
            >>> # Compute SHAP for all groups
            >>> shap_df = nested_fit.explain(test_data)
            >>>
            >>> # Compare feature importance across groups
            >>> importance_by_group = shap_df.groupby(["group", "variable"])["abs_shap"].mean()
            >>> print(importance_by_group.unstack())
            >>>
            >>> # Filter to specific group
            >>> group_a_shap = shap_df[shap_df["group"] == "A"]
        """
        from py_interpret import ShapEngine

        if self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        # Compute SHAP for each group
        all_shap = []

        for group, group_fit in self.group_fits.items():
            # Filter data for this group
            group_data = data[data[self.group_col] == group].copy()

            if len(group_data) == 0:
                continue  # Skip groups not in data

            # Remove group column before SHAP computation
            group_data_no_group = group_data.drop(columns=[self.group_col])

            # Compute SHAP for this group's model
            group_shap = ShapEngine.explain(
                fit=group_fit,
                data=group_data_no_group,
                method=method,
                background_size=background_size,
                background=background,
                background_data=background_data,
                check_additivity=check_additivity
            )

            # Add group column
            group_shap[self.group_col] = group

            all_shap.append(group_shap)

        if len(all_shap) == 0:
            raise ValueError("No matching groups found in data")

        # Combine SHAP values from all groups
        combined_shap = pd.concat(all_shap, ignore_index=True)

        # Rename group_col to "group" for consistency with extract_outputs()
        combined_shap = combined_shap.rename(columns={self.group_col: "group"})

        return combined_shap

    def save_mlflow(
        self,
        path: str,
        conda_env: Optional[Any] = None,
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save nested model (per-group models) to MLflow format.

        This method saves all group models into a single MLflow model package,
        preserving the grouped structure for later restoration.

        Args:
            path: Directory path where model will be saved
            conda_env: Conda environment specification (optional)
            signature: Model signature for input/output schema validation.
                      Use "auto" to infer from input_example.
            input_example: Example input DataFrame for signature inference (optional)
            registered_model_name: Name to register model in MLflow Model Registry (optional)
            metadata: Additional custom metadata dict (optional)

        Returns:
            None

        Examples:
            >>> # Fit nested model
            >>> spec = linear_reg()
            >>> nested_fit = spec.fit_nested(data, "sales ~ date", group_col="store_id")
            >>>
            >>> # Save nested model
            >>> nested_fit.save_mlflow("models/store_models")
            >>>
            >>> # Load and predict
            >>> from py_mlflow import load_model
            >>> loaded = load_model("models/store_models")
            >>> predictions = loaded.predict(test_data)  # Routes to correct group models
        """
        from py_mlflow import save_model

        save_model(
            model=self,
            path=path,
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            metadata=metadata
        )
