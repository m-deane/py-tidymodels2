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
