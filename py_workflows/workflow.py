"""
Workflow composition for preprocessing and modeling

Provides Workflow and WorkflowFit classes for composing preprocessing steps
with model specifications into complete pipelines.
"""

from dataclasses import dataclass, replace, field
from typing import Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from py_parsnip import ModelSpec, ModelFit
from py_recipes import Recipe, PreparedRecipe


@dataclass(frozen=True)
class Workflow:
    """
    Immutable workflow composition.

    Workflows combine preprocessing (formula or recipe) with a model specification
    into a single pipeline that can be fitted and used for prediction.

    Attributes:
        preprocessor: Formula string or Recipe object (or None)
        spec: ModelSpec for the model to fit
        post: Post-processing steps (future use)
        case_weights: Column name for case weights (optional)

    Examples:
        >>> # Create workflow with formula and model
        >>> wf = (
        ...     workflow()
        ...     .add_formula("sales ~ price + advertising")
        ...     .add_model(linear_reg().set_engine("sklearn"))
        ... )
        >>>
        >>> # Fit workflow
        >>> wf_fit = wf.fit(train_data)
        >>>
        >>> # Predict on new data
        >>> predictions = wf_fit.predict(test_data)
    """
    preprocessor: Optional[Any] = None  # Formula string or Recipe
    spec: Optional[ModelSpec] = None
    post: Optional[Any] = None
    case_weights: Optional[str] = None

    def add_formula(self, formula: str) -> "Workflow":
        """
        Add a formula for preprocessing.

        Args:
            formula: R-style formula string (e.g., "y ~ x1 + x2")

        Returns:
            New Workflow with formula added

        Raises:
            ValueError: If workflow already has a preprocessor

        Examples:
            >>> wf = workflow().add_formula("sales ~ price + advertising")
        """
        if self.preprocessor is not None:
            raise ValueError("Workflow already has a preprocessor")
        return replace(self, preprocessor=formula)

    def add_model(self, spec: ModelSpec) -> "Workflow":
        """
        Add a model specification.

        Args:
            spec: ModelSpec instance (from linear_reg(), rand_forest(), etc.)

        Returns:
            New Workflow with model added

        Raises:
            ValueError: If workflow already has a model

        Examples:
            >>> wf = workflow().add_model(linear_reg().set_engine("sklearn"))
        """
        if self.spec is not None:
            raise ValueError("Workflow already has a model")
        return replace(self, spec=spec)

    def add_recipe(self, recipe: Any) -> "Workflow":
        """
        Add a recipe for preprocessing.

        Args:
            recipe: Recipe object with preprocessing steps

        Returns:
            New Workflow with recipe added

        Raises:
            ValueError: If workflow already has a preprocessor

        Examples:
            >>> from py_recipes import recipe
            >>> rec = recipe().step_normalize().step_dummy(["category"])
            >>> wf = workflow().add_recipe(rec).add_model(linear_reg())
        """
        if self.preprocessor is not None:
            raise ValueError("Workflow already has a preprocessor")
        return replace(self, preprocessor=recipe)

    def remove_formula(self) -> "Workflow":
        """
        Remove the preprocessor.

        Returns:
            New Workflow without preprocessor
        """
        return replace(self, preprocessor=None)

    def remove_model(self) -> "Workflow":
        """
        Remove the model specification.

        Returns:
            New Workflow without model
        """
        return replace(self, spec=None)

    def update_formula(self, formula: str) -> "Workflow":
        """
        Replace the formula.

        Args:
            formula: New formula string

        Returns:
            New Workflow with updated formula
        """
        return replace(self, preprocessor=formula)

    def update_model(self, spec: ModelSpec) -> "Workflow":
        """
        Replace the model specification.

        Args:
            spec: New ModelSpec

        Returns:
            New Workflow with updated model
        """
        return replace(self, spec=spec)

    def fit(self, data: pd.DataFrame) -> "WorkflowFit":
        """
        Fit the entire workflow to training data.

        Args:
            data: Training data DataFrame

        Returns:
            WorkflowFit object containing fitted pipeline

        Raises:
            ValueError: If workflow doesn't have a model

        Examples:
            >>> wf = (
            ...     workflow()
            ...     .add_formula("sales ~ price")
            ...     .add_model(linear_reg().set_engine("sklearn"))
            ... )
            >>> wf_fit = wf.fit(train_data)
        """
        if self.spec is None:
            raise ValueError("Workflow must have a model specification")

        # Handle preprocessor
        if self.preprocessor is not None:
            if isinstance(self.preprocessor, str):
                # It's a formula - pass directly to model
                formula = self.preprocessor
                processed_data = data
                fitted_preprocessor = self.preprocessor
            elif isinstance(self.preprocessor, Recipe):
                # It's a recipe - prep and bake
                prepared_recipe = self.preprocessor.prep(data)
                processed_data = prepared_recipe.bake(data)

                # Auto-detect outcome column
                # Try common names first, then use first numeric column
                outcome_col = None
                for common_name in ['y', 'target', 'outcome']:
                    if common_name in processed_data.columns:
                        outcome_col = common_name
                        break

                # If no common name found, use the first column from original data
                # that's still present and numeric
                if outcome_col is None:
                    for col in data.columns:
                        if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                            outcome_col = col
                            break

                if outcome_col is None:
                    raise ValueError(
                        "Could not auto-detect outcome column. Please ensure your data has "
                        "a column named 'y', 'target', or 'outcome', or that the first numeric "
                        "column is the outcome variable."
                    )

                # Build explicit formula (patsy doesn't support "y ~ ." notation)
                predictor_cols = [col for col in processed_data.columns if col != outcome_col]
                if len(predictor_cols) == 0:
                    raise ValueError("No predictor columns found after recipe preprocessing")
                formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"
                fitted_preprocessor = prepared_recipe
            else:
                raise ValueError(f"Unknown preprocessor type: {type(self.preprocessor)}")
        else:
            # No preprocessor - use default formula
            raise ValueError("Workflow must have a formula (via add_formula()) or recipe (via add_recipe())")

        # Fit the model (data first, then formula)
        model_fit = self.spec.fit(processed_data, formula)

        return WorkflowFit(
            workflow=self,
            pre=fitted_preprocessor,
            fit=model_fit,
            post=self.post
        )

    def fit_nested(self, data: pd.DataFrame, group_col: str) -> "NestedWorkflowFit":
        """
        Fit separate models for each group in the data (panel/grouped modeling).

        This method fits one independent model per group value, useful for:
        - Multi-store sales forecasting (one model per store)
        - Multi-product demand forecasting (one model per product)
        - Multi-region time series (one model per region)

        Args:
            data: Training data DataFrame with group column
            group_col: Column name containing group identifiers

        Returns:
            NestedWorkflowFit containing dict of fitted models per group

        Raises:
            ValueError: If group_col not in data or workflow doesn't have model

        Examples:
            >>> # Fit separate models for each store
            >>> wf = (
            ...     workflow()
            ...     .add_formula("sales ~ date")
            ...     .add_model(recursive_reg(base_model=rand_forest(), lags=7))
            ... )
            >>> nested_fit = wf.fit_nested(data, group_col="store_id")
            >>>
            >>> # Predict for all groups
            >>> predictions = nested_fit.predict(test_data)
            >>> # Extract outputs with group column
            >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        """
        if self.spec is None:
            raise ValueError("Workflow must have a model specification")

        if group_col not in data.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")

        # Get unique groups
        groups = data[group_col].unique()

        # Fit separate model for each group
        group_fits = {}
        for group in groups:
            group_data = data[data[group_col] == group].copy()

            # For recursive models, set date as index if needed
            is_recursive = self.spec and self.spec.model_type == "recursive_reg"
            if is_recursive and "date" in group_data.columns and not isinstance(group_data.index, pd.DatetimeIndex):
                # Set date as index before removing group column (recursive models need this)
                group_data = group_data.set_index("date")
                group_data = group_data.drop(columns=[group_col])
            else:
                # Remove group column before fitting (it's not a predictor)
                group_data = group_data.drop(columns=[group_col])

            # Fit workflow for this group
            group_fits[group] = self.fit(group_data)

        return NestedWorkflowFit(
            workflow=self,
            group_col=group_col,
            group_fits=group_fits
        )

    def fit_global(self, data: pd.DataFrame, group_col: str) -> "WorkflowFit":
        """
        Fit a single global model using group as a feature.

        This method fits one model using all groups together, with the group
        column as an additional predictor. Useful when:
        - Groups share common patterns
        - Insufficient data per group for separate models
        - Want to capture cross-group effects

        Args:
            data: Training data DataFrame with group column
            group_col: Column name containing group identifiers (used as feature)

        Returns:
            WorkflowFit with group column included as predictor

        Raises:
            ValueError: If group_col not in data or workflow doesn't have model

        Examples:
            >>> # Fit single model using store_id as a feature
            >>> wf = (
            ...     workflow()
            ...     .add_formula("sales ~ date + store_id")
            ...     .add_model(rand_forest())
            ... )
            >>> global_fit = wf.fit_global(data, group_col="store_id")
            >>>
            >>> # Group column is automatically included as predictor
            >>> predictions = global_fit.predict(test_data)
        """
        if self.spec is None:
            raise ValueError("Workflow must have a model specification")

        if group_col not in data.columns:
            raise ValueError(f"Group column '{group_col}' not found in data")

        # Update formula to include group column as a feature
        if isinstance(self.preprocessor, str):
            formula = self.preprocessor
            # Check if group_col is already in the formula
            if group_col not in formula:
                # Add group_col to the formula
                if " ~ " in formula:
                    outcome, predictors = formula.split(" ~ ", 1)
                    if predictors == ".":
                        # Keep "." notation (will include group_col automatically)
                        updated_formula = formula
                    else:
                        # Add group_col explicitly
                        updated_formula = f"{outcome} ~ {predictors} + {group_col}"
                else:
                    raise ValueError(f"Invalid formula format: {formula}")
            else:
                updated_formula = formula

            # Update workflow with new formula
            updated_workflow = self.update_formula(updated_formula)
            return updated_workflow.fit(data)
        elif isinstance(self.preprocessor, Recipe):
            # For recipes, group column will be included automatically if present in data
            return self.fit(data)
        else:
            raise ValueError("Workflow must have a formula or recipe preprocessor")


@dataclass
class WorkflowFit:
    """
    Fitted workflow containing preprocessing and model fit.

    Attributes:
        workflow: Original Workflow specification
        pre: Fitted preprocessor (formula or recipe)
        fit: ModelFit object from parsnip
        post: Post-processing steps (future use)

    Examples:
        >>> wf_fit = workflow().add_formula("y ~ x").add_model(spec).fit(train)
        >>> predictions = wf_fit.predict(test)
        >>> outputs, coeffs, stats = wf_fit.extract_outputs()
    """
    workflow: Workflow
    pre: Any  # Fitted preprocessor (formula string or PreparedRecipe)
    fit: ModelFit
    post: Optional[Any] = None

    def predict(
        self,
        new_data: pd.DataFrame,
        type: str = "numeric"
    ) -> pd.DataFrame:
        """
        Generate predictions using the fitted workflow.

        Automatically applies preprocessing before prediction.

        Args:
            new_data: New data for prediction
            type: Prediction type ("numeric", "conf_int", "pred_int", etc.)

        Returns:
            DataFrame with predictions

        Examples:
            >>> predictions = wf_fit.predict(test_data)
            >>> # With prediction intervals
            >>> predictions = wf_fit.predict(test_data, type="pred_int")
        """
        # Apply preprocessing to new data
        if isinstance(self.pre, str):
            # Formula - pass data directly to model (forge() handles it)
            processed_data = new_data
        elif isinstance(self.pre, PreparedRecipe):
            # Recipe - apply fitted transformations
            processed_data = self.pre.bake(new_data)
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self.pre)}")

        # Model prediction (handles forge() internally)
        predictions = self.fit.predict(processed_data, type=type)

        return predictions

    def evaluate(
        self,
        test_data: pd.DataFrame,
        outcome_col: Optional[str] = None
    ) -> "WorkflowFit":
        """
        Evaluate workflow on test data.

        Stores test predictions for comprehensive train/test metrics.

        Args:
            test_data: Test data with actual outcomes
            outcome_col: Name of outcome column (auto-detected if None)

        Returns:
            Self for method chaining

        Examples:
            >>> wf_fit = wf.fit(train).evaluate(test)
            >>> outputs, coeffs, stats = wf_fit.extract_outputs()
            >>> # Now outputs includes both train and test observations
        """
        # Apply preprocessing to test data
        if isinstance(self.pre, str):
            # Formula - pass data directly to model (forge() handles it)
            processed_test_data = test_data
        elif isinstance(self.pre, PreparedRecipe):
            # Recipe - apply fitted transformations
            processed_test_data = self.pre.bake(test_data)
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self.pre)}")

        # Delegate to the underlying ModelFit with preprocessed data
        self.fit = self.fit.evaluate(processed_test_data, outcome_col)
        return self

    def extract_fit_parsnip(self) -> ModelFit:
        """
        Extract the underlying parsnip ModelFit.

        Returns:
            ModelFit object

        Examples:
            >>> model_fit = wf_fit.extract_fit_parsnip()
            >>> # Access underlying model
            >>> sklearn_model = model_fit.fit_data["model"]
        """
        return self.fit

    def extract_preprocessor(self) -> Any:
        """
        Extract the fitted preprocessor.

        Returns:
            Formula string or PreparedRecipe
        """
        return self.pre

    def extract_spec_parsnip(self) -> ModelSpec:
        """
        Extract the model specification.

        Returns:
            ModelSpec object
        """
        return self.workflow.spec

    def extract_outputs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract comprehensive three-DataFrame outputs.

        Returns:
            Tuple of (outputs, coefficients, stats) DataFrames per
            .claude_plans/model_outputs.md specification

        Examples:
            >>> outputs, coefficients, stats = wf_fit.extract_outputs()
            >>>
            >>> # Outputs: observation-level results
            >>> print(outputs[["actuals", "fitted", "forecast", "split"]])
            >>>
            >>> # Coefficients: variable-level parameters
            >>> print(coefficients[["variable", "coefficient", "p_value"]])
            >>>
            >>> # Stats: model-level metrics by split
            >>> print(stats[stats["metric"].isin(["rmse", "mae", "r_squared"])])
        """
        return self.fit.extract_outputs()


@dataclass
class NestedWorkflowFit:
    """
    Fitted workflow with separate models for each group (panel/grouped modeling).

    This class holds multiple fitted models, one per group, enabling
    independent forecasting for each group while maintaining a unified interface.

    Attributes:
        workflow: Original Workflow specification
        group_col: Column name containing group identifiers
        group_fits: Dict mapping group values to WorkflowFit objects

    Examples:
        >>> wf = workflow().add_formula("sales ~ date").add_model(spec)
        >>> nested_fit = wf.fit_nested(data, group_col="store_id")
        >>>
        >>> # Predict for all groups
        >>> predictions = nested_fit.predict(test_data)
        >>>
        >>> # Extract outputs with group column
        >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        >>> print(outputs[["date", "group", "actuals", "forecast"]])
    """
    workflow: Workflow
    group_col: str
    group_fits: dict  # {group_value: WorkflowFit}

    def predict(
        self,
        new_data: pd.DataFrame,
        type: str = "numeric"
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
            ValueError: If group_col not in new_data

        Examples:
            >>> predictions = nested_fit.predict(test_data)
            >>> predictions = nested_fit.predict(test_data, type="pred_int")
        """
        if self.group_col not in new_data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in new_data")

        # Get predictions for each group
        all_predictions = []
        is_recursive = self.workflow.spec and self.workflow.spec.model_type == "recursive_reg"

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
    ) -> "NestedWorkflowFit":
        """
        Evaluate all group models on test data.

        Args:
            test_data: Test data with actual outcomes and group column
            outcome_col: Name of outcome column (auto-detected if None)

        Returns:
            Self for method chaining

        Examples:
            >>> nested_fit = wf.fit_nested(train, "store_id")
            >>> nested_fit = nested_fit.evaluate(test)
            >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        """
        if self.group_col not in test_data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in test_data")

        # Evaluate each group model
        is_recursive = self.workflow.spec and self.workflow.spec.model_type == "recursive_reg"

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
            - outputs: Includes 'group' column showing which group each row belongs to
            - coefficients: Includes 'group' column
            - stats: Includes 'group' column

        Examples:
            >>> outputs, coefficients, stats = nested_fit.extract_outputs()
            >>>
            >>> # Filter to specific group
            >>> store_a_outputs = outputs[outputs["group"] == "A"]
            >>>
            >>> # Compare metrics across groups
            >>> test_rmse = stats[
            ...     (stats["metric"] == "rmse") &
            ...     (stats["split"] == "test")
            ... ][["group", "value"]]
        """
        all_outputs = []
        all_coefficients = []
        all_stats = []

        for group, group_fit in self.group_fits.items():
            # Extract outputs for this group
            outputs, coefficients, stats = group_fit.extract_outputs()

            # Add group column
            outputs[self.group_col] = group
            coefficients[self.group_col] = group
            stats[self.group_col] = group

            all_outputs.append(outputs)
            all_coefficients.append(coefficients)
            all_stats.append(stats)

        # Combine all groups
        combined_outputs = pd.concat(all_outputs, ignore_index=True)
        combined_coefficients = pd.concat(all_coefficients, ignore_index=True)
        combined_stats = pd.concat(all_stats, ignore_index=True)

        return combined_outputs, combined_coefficients, combined_stats


def workflow() -> Workflow:
    """
    Create a new empty workflow.

    Returns:
        Empty Workflow ready for composition

    Examples:
        >>> wf = (
        ...     workflow()
        ...     .add_formula("sales ~ price + advertising")
        ...     .add_model(linear_reg(penalty=0.1).set_engine("sklearn"))
        ... )
        >>> wf_fit = wf.fit(train_data)
    """
    return Workflow()
