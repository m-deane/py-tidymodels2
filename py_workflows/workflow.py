"""
Workflow composition for preprocessing and modeling

Provides Workflow and WorkflowFit classes for composing preprocessing steps
with model specifications into complete pipelines.
"""

from dataclasses import dataclass, replace, field
from typing import Any, Optional, Tuple, Union
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
        model_name: Optional name for the model (used in extract_outputs())
        model_group_name: Optional group name for organizing models (used in extract_outputs())

    Examples:
        >>> # Create workflow with formula and model
        >>> wf = (
        ...     workflow()
        ...     .add_formula("sales ~ price + advertising")
        ...     .add_model(linear_reg().set_engine("sklearn"))
        ...     .add_model_name("baseline")
        ...     .add_model_group_name("linear_models")
        ... )
        >>>
        >>> # Fit workflow
        >>> wf_fit = wf.fit(train_data)
        >>>
        >>> # Predict on new data
        >>> predictions = wf_fit.predict(test_data)
        >>>
        >>> # Extract outputs with custom model names
        >>> outputs, _, _ = wf_fit.extract_outputs()
        >>> print(outputs["model"].unique())  # ['baseline']
        >>> print(outputs["model_group_name"].unique())  # ['linear_models']
    """
    preprocessor: Optional[Any] = None  # Formula string or Recipe
    spec: Optional[ModelSpec] = None
    post: Optional[Any] = None
    case_weights: Optional[str] = None
    model_name: Optional[str] = None
    model_group_name: Optional[str] = None

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

    def add_model_name(self, name: str) -> "Workflow":
        """
        Add a model name for identification in outputs.

        The model name will appear in the "model" column of DataFrames
        returned by extract_outputs().

        Args:
            name: Name for this model (e.g., "baseline", "poly", "interaction")

        Returns:
            New Workflow with model_name set

        Examples:
            >>> wf = (
            ...     workflow()
            ...     .add_model(linear_reg())
            ...     .add_model_name("baseline")
            ... )
            >>> fit = wf.fit(train_data)
            >>> outputs, _, _ = fit.extract_outputs()
            >>> print(outputs["model"].unique())  # ['baseline']
        """
        return replace(self, model_name=name)

    def add_model_group_name(self, group_name: str) -> "Workflow":
        """
        Add a model group name for organizing related models.

        The model group name will appear in the "model_group_name" column of
        DataFrames returned by extract_outputs(). Useful for organizing models
        into logical groups (e.g., "linear_models", "tree_models", "ensemble").

        Args:
            group_name: Group name for organizing models (e.g., "linear_models", "polynomial")

        Returns:
            New Workflow with model_group_name set

        Examples:
            >>> wf = (
            ...     workflow()
            ...     .add_model(linear_reg())
            ...     .add_model_name("baseline")
            ...     .add_model_group_name("linear_models")
            ... )
            >>> fit = wf.fit(train_data)
            >>> outputs, _, _ = fit.extract_outputs()
            >>> print(outputs["model_group_name"].unique())  # ['linear_models']
        """
        return replace(self, model_group_name=group_name)

    def remove_formula(self) -> "Workflow":
        """
        Remove the preprocessor.

        Returns:
            New Workflow without preprocessor
        """
        return replace(self, preprocessor=None)

    def _recipe_requires_outcome(self, recipe: Any) -> bool:
        """
        Check if recipe contains supervised steps that require outcome during prep/bake.

        Supervised feature selection steps need the outcome column to calculate
        feature importance or relevance scores.

        Args:
            recipe: Recipe or PreparedRecipe object to check

        Returns:
            True if recipe has supervised steps requiring outcome, False otherwise
        """
        # Handle both Recipe and PreparedRecipe
        steps = None
        if hasattr(recipe, 'steps'):
            steps = recipe.steps
        elif hasattr(recipe, 'prepared_steps'):
            # PreparedRecipe uses prepared_steps
            steps = recipe.prepared_steps

        if steps is None:
            return False

        # Supervised step class names that require outcome during prep/bake
        supervised_step_types = {
            'StepFilterAnova',
            'StepFilterRfImportance',
            'StepFilterMutualInfo',
            'StepFilterRocAuc',
            'StepFilterChisq',
            'StepSelectShap',
            'StepSelectPermutation',
            'StepSafe',
            'StepSafeV2',
        }

        for step in steps:
            step_class_name = step.__class__.__name__
            if step_class_name in supervised_step_types:
                return True

        return False

    def _get_outcome_from_recipe(self, recipe: Any) -> Optional[str]:
        """
        Extract outcome column name from supervised feature selection steps.

        Args:
            recipe: Recipe or PreparedRecipe object

        Returns:
            Outcome column name if found in supervised steps, None otherwise
        """
        # Handle both Recipe and PreparedRecipe
        steps = None
        if hasattr(recipe, 'steps'):
            steps = recipe.steps
        elif hasattr(recipe, 'prepared_steps'):
            steps = recipe.prepared_steps

        if steps is None:
            return None

        # Check each step for outcome attribute
        for step in steps:
            if hasattr(step, 'outcome') and step.outcome is not None:
                return step.outcome

        return None

    def _detect_outcome(self, original_data: pd.DataFrame) -> str:
        """
        Auto-detect outcome column from original data.

        Args:
            original_data: Original data before preprocessing

        Returns:
            Name of outcome column

        Raises:
            ValueError: If outcome column cannot be detected
        """
        # Try common names first
        for common_name in ['y', 'target', 'outcome']:
            if common_name in original_data.columns:
                return common_name

        # If no common name found, use the first numeric column
        for col in original_data.columns:
            if pd.api.types.is_numeric_dtype(original_data[col]):
                return col

        raise ValueError(
            "Could not auto-detect outcome column. Please ensure your data has "
            "a column named 'y', 'target', or 'outcome', or that the first numeric "
            "column is the outcome variable."
        )

    def _prep_and_bake_with_outcome(self, recipe, data: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
        """
        Prep and bake a recipe while preserving the outcome column.

        Args:
            recipe: Recipe or PreparedRecipe
            data: Data to process
            outcome_col: Name of outcome column to preserve

        Returns:
            Processed data with outcome column preserved
        """
        from py_recipes.recipe import PreparedRecipe

        # Check if recipe has supervised steps that need outcome during bake
        needs_outcome = self._recipe_requires_outcome(recipe if isinstance(recipe, PreparedRecipe) else recipe)

        # DEBUG
        import sys
        if hasattr(sys, '_workflow_debug'):
            print(f"   [DEBUG] _prep_and_bake_with_outcome:")
            print(f"           needs_outcome={needs_outcome}")
            print(f"           input columns={list(data.columns)}")

        if needs_outcome:
            # Bake with outcome included (for supervised feature selection)
            if isinstance(recipe, PreparedRecipe):
                processed_data = recipe.bake(data)
            else:
                prepped = recipe.prep(data)
                processed_data = prepped.bake(data)

            # DEBUG
            if hasattr(sys, '_workflow_debug'):
                print(f"           output columns={list(processed_data.columns)}")
        else:
            # Separate outcome from predictors
            outcome = data[outcome_col].copy()
            predictors = data.drop(columns=[outcome_col])

            # Prep/bake predictors only
            if isinstance(recipe, PreparedRecipe):
                processed_predictors = recipe.bake(predictors)
            else:
                prepped = recipe.prep(predictors)
                processed_predictors = prepped.bake(predictors)

            # Recombine with outcome (align by index in case rows were removed by step_naomit)
            processed_data = processed_predictors.copy()
            processed_data[outcome_col] = outcome.loc[processed_predictors.index].values

        return processed_data

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

        # Store original training data for engines that need raw datetime/categorical values
        original_data = data.copy()

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
                # Exclude datetime columns - they should be indices, not predictors
                predictor_cols = [
                    col for col in processed_data.columns
                    if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
                ]
                if len(predictor_cols) == 0:
                    raise ValueError("No predictor columns found after recipe preprocessing")

                # Escape column names that contain Patsy special characters
                # Special chars: ^ * : + - / ( ) ** [ ] { }
                import re
                def escape_column_name(col):
                    if re.search(r'[\^\*\:\+\-\/\(\)\[\]\{\}]', col):
                        # Wrap in Q() for Patsy to treat as literal column name
                        return f'Q("{col}")'
                    return col

                escaped_cols = [escape_column_name(col) for col in predictor_cols]
                formula = f"{outcome_col} ~ {' + '.join(escaped_cols)}"
                fitted_preprocessor = prepared_recipe
            else:
                raise ValueError(f"Unknown preprocessor type: {type(self.preprocessor)}")
        else:
            # No preprocessor - use default formula
            raise ValueError("Workflow must have a formula (via add_formula()) or recipe (via add_recipe())")

        # Fit the model (data first, then formula)
        # Pass original training data for engines that need raw datetime/categorical values
        model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)

        # Set model_name and model_group_name from workflow if provided
        if self.model_name is not None or self.model_group_name is not None:
            model_fit = replace(
                model_fit,
                model_name=self.model_name if self.model_name is not None else model_fit.model_name,
                model_group_name=self.model_group_name if self.model_group_name is not None else model_fit.model_group_name
            )

        return WorkflowFit(
            workflow=self,
            pre=fitted_preprocessor,
            fit=model_fit,
            post=self.post,
            formula=formula  # Store formula for easy access
        )

    def fit_nested(
        self,
        data: pd.DataFrame,
        group_col: str,
        per_group_prep: bool = True,
        min_group_size: int = 30
    ) -> "NestedWorkflowFit":
        """
        Fit separate models for each group in the data (panel/grouped modeling).

        This method fits one independent model per group value, useful for:
        - Multi-store sales forecasting (one model per store)
        - Multi-product demand forecasting (one model per product)
        - Multi-region time series (one model per region)

        Args:
            data: Training data DataFrame with group column
            group_col: Column name containing group identifiers
            per_group_prep: If True, prep recipe separately for each group.
                           Enables group-specific feature engineering (PCA, feature selection).
                           Default: True (each group gets optimized preprocessing).
            min_group_size: Minimum samples required for per-group preprocessing.
                           Groups smaller than this use global recipe (if per_group_prep=True).
                           Default: 30.

        Returns:
            NestedWorkflowFit containing dict of fitted models per group

        Raises:
            ValueError: If group_col not in data or workflow doesn't have model

        Examples:
            >>> # Fit separate models for each store (shared preprocessing)
            >>> wf = (
            ...     workflow()
            ...     .add_formula("sales ~ date")
            ...     .add_model(recursive_reg(base_model=rand_forest(), lags=7))
            ... )
            >>> nested_fit = wf.fit_nested(data, group_col="store_id")
            >>>
            >>> # Fit with per-group feature engineering (e.g., PCA)
            >>> wf_pca = (
            ...     workflow()
            ...     .add_recipe(recipe().step_pca(num_comp=5))
            ...     .add_model(linear_reg())
            ... )
            >>> nested_fit = wf_pca.fit_nested(
            ...     data,
            ...     group_col="store_id",
            ...     per_group_prep=True  # Each group gets own PCA components
            ... )
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

        # For per-group prep, we need a recipe (not just formula)
        if per_group_prep and not isinstance(self.preprocessor, Recipe):
            import warnings
            warnings.warn(
                "per_group_prep=True requires a recipe preprocessor. "
                "Workflow uses formula-only preprocessing. "
                "Falling back to per_group_prep=False.",
                UserWarning
            )
            per_group_prep = False

        # Prep global/shared recipe if needed
        global_recipe = None
        if per_group_prep or isinstance(self.preprocessor, Recipe):
            # Detect outcome column from full data
            # For supervised feature selection, get outcome from recipe; otherwise auto-detect
            outcome_col_global = self._get_outcome_from_recipe(self.preprocessor)
            if outcome_col_global is None:
                outcome_col_global = self._detect_outcome(data)

            # Check if recipe has supervised steps that need outcome during prep
            needs_outcome = self._recipe_requires_outcome(self.preprocessor)

            if needs_outcome:
                # Prep with outcome included (for supervised feature selection)
                prep_data = data.drop(columns=[group_col])
                global_recipe = self.preprocessor.prep(prep_data)
            else:
                # Prep on predictors only (excluding outcome)
                predictors_global = data.drop(columns=[outcome_col_global, group_col])
                global_recipe = self.preprocessor.prep(predictors_global)

        # Fit separate model for each group
        group_fits = {}
        group_recipes = {} if per_group_prep else None
        group_train_data = {}  # Store original training data per group (for date extraction)

        for group in groups:
            group_data = data[data[group_col] == group].copy()

            # Store original group training data (BEFORE removing group column)
            # This preserves date column for later extraction in extract_outputs()
            group_train_data[group] = group_data.copy()

            # Determine if we should use per-group preprocessing for this group
            use_group_recipe = per_group_prep and len(group_data) >= min_group_size

            if not use_group_recipe and per_group_prep and len(group_data) < min_group_size:
                import warnings
                warnings.warn(
                    f"Group '{group}' has only {len(group_data)} samples "
                    f"(minimum: {min_group_size}). Using global recipe instead.",
                    UserWarning
                )

            # For recursive models, set date as index if needed
            is_recursive = self.spec and self.spec.model_type == "recursive_reg"
            if is_recursive and "date" in group_data.columns and not isinstance(group_data.index, pd.DatetimeIndex):
                # Set date as index before removing group column (recursive models need this)
                group_data = group_data.set_index("date")
                group_data_no_group = group_data.drop(columns=[group_col])
            else:
                # Remove group column before fitting (it's not a predictor)
                group_data_no_group = group_data.drop(columns=[group_col])

            # Detect outcome column (before preprocessing)
            # For supervised feature selection, get outcome from recipe; otherwise auto-detect
            outcome_col = self._get_outcome_from_recipe(self.preprocessor)
            if outcome_col is None:
                outcome_col = self._detect_outcome(group_data_no_group)

            # Fit based on preprocessing strategy
            if use_group_recipe:
                # Prep recipe on THIS group's data only
                try:
                    # Check if recipe needs outcome during prep
                    needs_outcome = self._recipe_requires_outcome(self.preprocessor)

                    # DEBUG
                    import sys
                    if hasattr(sys, '_workflow_debug'):
                        print(f"\n[DEBUG] Prepping recipe for group '{group}':")
                        print(f"        needs_outcome={needs_outcome}")
                        print(f"        group_data_no_group columns={list(group_data_no_group.columns)}")

                    if needs_outcome:
                        # Prep with outcome included (for supervised feature selection)
                        group_recipe = self.preprocessor.prep(group_data_no_group)
                    else:
                        # Prep on predictors only (excluding outcome)
                        predictors = group_data_no_group.drop(columns=[outcome_col])
                        group_recipe = self.preprocessor.prep(predictors)

                    group_recipes[group] = group_recipe

                    # DEBUG
                    if hasattr(sys, '_workflow_debug'):
                        print(f"        recipe prepped successfully")
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Recipe prep failed for group '{group}': {str(e)}\n"
                        f"Falling back to global recipe for this group.",
                        UserWarning
                    )
                    # Fallback to global recipe
                    group_recipes[group] = global_recipe

                # Bake data with group's recipe (preserving outcome)
                processed_data = self._prep_and_bake_with_outcome(
                    group_recipes[group],
                    group_data_no_group,
                    outcome_col
                )

                # Build formula from processed data
                predictor_cols = [
                    col for col in processed_data.columns
                    if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
                ]
                formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"

                # Fit model directly
                model_fit = self.spec.fit(processed_data, formula, original_training_data=group_data_no_group)

                # Set model_name and model_group_name from workflow if provided
                if self.model_name is not None or self.model_group_name is not None:
                    model_fit = replace(
                        model_fit,
                        model_name=self.model_name if self.model_name is not None else model_fit.model_name,
                        model_group_name=self.model_group_name if self.model_group_name is not None else model_fit.model_group_name
                    )

                # Wrap in WorkflowFit
                group_fits[group] = WorkflowFit(
                    workflow=self,
                    pre=group_recipes[group],
                    fit=model_fit,
                    post=self.post,
                    formula=formula,
                    recipe_prepped_without_outcome=True  # Per-group recipe prepped on predictors only
                )

            elif per_group_prep:
                # Small group - use global recipe
                group_recipes[group] = global_recipe

                # Bake data with global recipe (preserving outcome)
                processed_data = self._prep_and_bake_with_outcome(
                    global_recipe,
                    group_data_no_group,
                    outcome_col
                )

                # Build formula from processed data
                predictor_cols = [
                    col for col in processed_data.columns
                    if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
                ]
                formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"

                # Fit model directly
                model_fit = self.spec.fit(processed_data, formula, original_training_data=group_data_no_group)

                # Set model_name and model_group_name from workflow if provided
                if self.model_name is not None or self.model_group_name is not None:
                    model_fit = replace(
                        model_fit,
                        model_name=self.model_name if self.model_name is not None else model_fit.model_name,
                        model_group_name=self.model_group_name if self.model_group_name is not None else model_fit.model_group_name
                    )

                # Wrap in WorkflowFit
                group_fits[group] = WorkflowFit(
                    workflow=self,
                    pre=global_recipe,
                    fit=model_fit,
                    post=self.post,
                    formula=formula,
                    recipe_prepped_without_outcome=True  # Global recipe also prepped on predictors only
                )

            else:
                # Standard shared preprocessing
                if global_recipe is not None:
                    # Recipe-based: use shared prepped recipe (preserving outcome)
                    processed_data = self._prep_and_bake_with_outcome(
                        global_recipe,
                        group_data_no_group,
                        outcome_col
                    )

                    # Build formula from processed data
                    predictor_cols = [
                        col for col in processed_data.columns
                        if col != outcome_col and not pd.api.types.is_datetime64_any_dtype(processed_data[col])
                    ]
                    formula = f"{outcome_col} ~ {' + '.join(predictor_cols)}"

                    # Fit model directly
                    model_fit = self.spec.fit(processed_data, formula, original_training_data=group_data_no_group)

                    # Set model_name and model_group_name from workflow if provided
                    if self.model_name is not None or self.model_group_name is not None:
                        model_fit = replace(
                            model_fit,
                            model_name=self.model_name if self.model_name is not None else model_fit.model_name,
                            model_group_name=self.model_group_name if self.model_group_name is not None else model_fit.model_group_name
                        )

                    # Wrap in WorkflowFit
                    group_fits[group] = WorkflowFit(
                        workflow=self,
                        pre=global_recipe,
                        fit=model_fit,
                        post=self.post,
                        formula=formula
                    )
                else:
                    # Formula-based: fit normally (formula applied per group)
                    group_fits[group] = self.fit(group_data_no_group)

        return NestedWorkflowFit(
            workflow=self,
            group_col=group_col,
            group_fits=group_fits,
            group_recipes=group_recipes,
            group_train_data=group_train_data
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

            # Validate formula structure
            if '~' not in formula:
                raise ValueError(f"Invalid formula format: {formula}. Formula must contain '~'")

            # Robust formula parsing with whitespace normalization
            lhs, rhs = formula.split('~', 1)
            outcome = lhs.strip()
            predictors = rhs.strip()

            # Check if group_col is already in the formula
            # Split predictors on '+' and check each term
            predictor_terms = [term.strip() for term in predictors.split('+')]

            if group_col in predictor_terms or predictors == ".":
                # group_col already in formula or using "." notation (includes all columns)
                updated_formula = formula
            else:
                # Add group_col explicitly
                updated_formula = f"{outcome} ~ {predictors} + {group_col}"

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
        formula: Formula used for model fitting (stored for convenience)
        recipe_prepped_without_outcome: Whether recipe was prepped excluding outcome column

    Examples:
        >>> wf_fit = workflow().add_formula("y ~ x").add_model(spec).fit(train)
        >>> predictions = wf_fit.predict(test)
        >>> outputs, coeffs, stats = wf_fit.extract_outputs()
    """
    workflow: Workflow
    pre: Any  # Fitted preprocessor (formula string or PreparedRecipe)
    fit: ModelFit
    post: Optional[Any] = None
    formula: Optional[str] = None  # Formula used for model fitting
    recipe_prepped_without_outcome: bool = False  # True only for per-group preprocessing

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

            # Only separate outcome if recipe was prepped WITHOUT outcome (per-group case)
            if self.recipe_prepped_without_outcome:
                # Per-group preprocessing: recipe prepped on predictors only
                # IMPORTANT: Preserve outcome column during baking

                # Detect outcome column if not provided
                if outcome_col is None:
                    # For supervised steps, get outcome from recipe; otherwise auto-detect
                    outcome_col = self.workflow._get_outcome_from_recipe(self.pre)
                    if outcome_col is None:
                        outcome_col = self.workflow._detect_outcome(test_data)

                # Check if outcome exists in test data
                if outcome_col in test_data.columns:
                    # Check if recipe has supervised steps that need outcome during bake
                    needs_outcome = self.workflow._recipe_requires_outcome(self.pre)

                    if needs_outcome:
                        # Bake with outcome included (for supervised feature selection)
                        processed_test_data = self.pre.bake(test_data)
                    else:
                        # Separate outcome from predictors
                        outcome = test_data[outcome_col].copy()
                        predictors = test_data.drop(columns=[outcome_col])

                        # Bake predictors only
                        processed_predictors = self.pre.bake(predictors)

                        # Recombine with outcome (align by index to handle step_naomit)
                        processed_test_data = processed_predictors.copy()
                        processed_test_data[outcome_col] = outcome.loc[processed_predictors.index].values
                else:
                    # No outcome in test data (prediction only scenario)
                    processed_test_data = self.pre.bake(test_data)
            else:
                # Standard workflow: recipe prepped on all data including outcome
                # Bake normally (recipe expects outcome to be present)
                processed_test_data = self.pre.bake(test_data)
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self.pre)}")

        # Delegate to the underlying ModelFit with both original and preprocessed data
        # Pass original test data so engines can access raw datetime/categorical values
        self.fit = self.fit.evaluate(processed_test_data, outcome_col, original_test_data=test_data)
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
        outputs, coefficients, stats = self.fit.extract_outputs()

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

    def extract_formula(self) -> str:
        """
        Extract the formula used for model fitting.

        Returns:
            Formula string (e.g., "y ~ x1 + x2")

        Examples:
            >>> wf_fit = workflow().add_formula("sales ~ price").add_model(spec).fit(train)
            >>> formula = wf_fit.extract_formula()
            >>> print(formula)
            'sales ~ price'
            >>>
            >>> # For recipes, returns auto-generated formula
            >>> rec = recipe().step_normalize()
            >>> wf_fit = workflow().add_recipe(rec).add_model(spec).fit(train)
            >>> formula = wf_fit.extract_formula()
            >>> print(formula)
            'y ~ x1 + x2 + x3'
        """
        if self.formula is None:
            raise ValueError("No formula stored in WorkflowFit")
        return self.formula

    def extract_preprocessed_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessor to data and return transformed data.

        This is a convenience function to get the preprocessed data that would
        be used by the model, useful for:
        - Inspecting transformed features
        - Understanding what the model actually sees
        - Debugging preprocessing pipelines
        - Manual analysis of transformed data

        Args:
            data: Data to preprocess (train or test data)

        Returns:
            DataFrame with preprocessing applied

        Examples:
            >>> # With formula
            >>> wf_fit = workflow().add_formula("y ~ x1 + x2").add_model(spec).fit(train)
            >>> train_transformed = wf_fit.extract_preprocessed_data(train)
            >>> test_transformed = wf_fit.extract_preprocessed_data(test)
            >>>
            >>> # With recipe
            >>> rec = recipe().step_normalize().step_dummy()
            >>> wf_fit = workflow().add_recipe(rec).add_model(spec).fit(train)
            >>> train_transformed = wf_fit.extract_preprocessed_data(train)
            >>>
            >>> # Inspect transformed columns
            >>> print(train_transformed.columns)
            >>> print(train_transformed.head())
        """
        if isinstance(self.pre, str):
            # Formula - use mold() to get preprocessed data
            from py_hardhat import mold
            molded = mold(self.pre, data)
            # Return predictors and outcomes combined
            result = molded.predictors.copy()
            # Add outcome column(s) if present
            if molded.outcomes is not None and not molded.outcomes.empty:
                for col in molded.outcomes.columns:
                    result[col] = molded.outcomes[col]
            return result
        elif isinstance(self.pre, PreparedRecipe):
            # Recipe - use bake()
            return self.pre.bake(data)
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self.pre)}")


@dataclass
class NestedWorkflowFit:
    """
    Fitted workflow with separate models for each group (panel/grouped modeling).

    This class holds multiple fitted models, one per group, enabling
    independent forecasting for each group while maintaining a unified interface.

    When per_group_prep=True, also stores separate recipes per group, enabling
    group-specific feature engineering (e.g., different PCA components per group).

    Attributes:
        workflow: Original Workflow specification
        group_col: Column name containing group identifiers
        group_fits: Dict mapping group values to WorkflowFit objects
        group_recipes: Optional dict mapping group values to PreparedRecipe objects
                      (None if per_group_prep=False)
        group_train_data: Dict mapping group values to original training data
                         (for date extraction in extract_outputs)

    Examples:
        >>> # Standard nested fit (shared preprocessing)
        >>> wf = workflow().add_formula("sales ~ date").add_model(spec)
        >>> nested_fit = wf.fit_nested(data, group_col="store_id")
        >>>
        >>> # Nested fit with per-group feature engineering
        >>> wf_pca = (
        ...     workflow()
        ...     .add_recipe(recipe().step_pca(num_comp=5))
        ...     .add_model(linear_reg())
        ... )
        >>> nested_fit = wf_pca.fit_nested(
        ...     data,
        ...     group_col="store_id",
        ...     per_group_prep=True
        ... )
        >>>
        >>> # Predict for all groups (automatic routing)
        >>> predictions = nested_fit.predict(test_data)
        >>>
        >>> # Compare features across groups
        >>> feature_comp = nested_fit.get_feature_comparison()
        >>>
        >>> # Extract outputs with group column
        >>> outputs, coeffs, stats = nested_fit.extract_outputs()
        >>> print(outputs[["date", "group", "actuals", "forecast"]])
    """
    workflow: Workflow
    group_col: str
    group_fits: dict  # {group_value: WorkflowFit}
    group_recipes: Optional[dict]  # {group_value: PreparedRecipe} or None
    group_train_data: dict  # {group_value: DataFrame} - original training data with dates

    def predict(
        self,
        new_data: pd.DataFrame,
        type: str = "numeric"
    ) -> pd.DataFrame:
        """
        Generate predictions for all groups.

        Automatically routes each row to the appropriate group model.
        If per_group_prep=True was used during fitting, applies group-specific
        preprocessing (recipes) before prediction.

        Args:
            new_data: New data with group column
            type: Prediction type ("numeric", "conf_int", "pred_int", etc.)

        Returns:
            DataFrame with predictions and group column

        Raises:
            ValueError: If group_col not in new_data or new groups encountered

        Examples:
            >>> predictions = nested_fit.predict(test_data)
            >>> predictions = nested_fit.predict(test_data, type="pred_int")
        """
        if self.group_col not in new_data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in new_data")

        # Check for new groups not seen during training
        new_groups = set(new_data[self.group_col].unique())
        training_groups = set(self.group_fits.keys())
        unseen_groups = new_groups - training_groups

        if unseen_groups:
            raise ValueError(
                f"New group(s) not seen during training: {sorted(unseen_groups)}\n"
                f"Available groups: {sorted(training_groups)}\n"
                f"Cannot predict for unseen groups."
            )

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

            # Apply group-specific preprocessing if available
            if self.group_recipes is not None:
                # Per-group preprocessing: bake with group's recipe
                group_recipe = self.group_recipes[group]
                processed_data = group_recipe.bake(group_data_no_group)
            else:
                # Standard preprocessing: will be applied inside group_fit.predict()
                processed_data = group_data_no_group

            # Get predictions
            group_preds = group_fit.predict(processed_data, type=type)

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

        # Store original test data per group for date extraction later
        # This is needed because recipes may exclude date columns during preprocessing
        if not hasattr(self, 'group_test_data'):
            self.group_test_data = {}

        # Evaluate each group model
        is_recursive = self.workflow.spec and self.workflow.spec.model_type == "recursive_reg"

        for group, group_fit in self.group_fits.items():
            # Filter data for this group
            group_data = test_data[test_data[self.group_col] == group].copy()

            if len(group_data) == 0:
                continue  # Skip groups not in test_data

            # Store original group test data (WITH group column, date, etc.)
            # This preserves date column for later extraction in extract_outputs()
            self.group_test_data[group] = group_data.copy()

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

    def get_feature_comparison(self) -> pd.DataFrame:
        """
        Compare which features are used by each group.

        Only applicable when per_group_prep=True was used during fitting.
        Returns a DataFrame showing which features are present for each group,
        enabling cross-group comparison of feature engineering results.

        Returns:
            DataFrame with groups as rows, features as columns,
            bool values indicating if feature is present for that group.
            Returns None if per_group_prep=False (shared preprocessing).

        Examples:
            >>> # After fitting with per_group_prep=True
            >>> nested_fit = wf.fit_nested(data, group_col="country", per_group_prep=True)
            >>> comparison = nested_fit.get_feature_comparison()
            >>> print(comparison)
            #           PC1    PC2    PC3    PC4    PC5
            # USA      True   True   True   True  False
            # UK       True   True   True  False  False
            # Canada   True   True   True   True   True
            >>>
            >>> # See which features are shared vs group-specific
            >>> shared_features = comparison.columns[comparison.all()]
            >>> group_specific = comparison.columns[~comparison.all()]
        """
        if self.group_recipes is None:
            print("No per-group preprocessing. All groups use the same features.")
            return None

        # Get feature names from each group's workflow fit
        feature_usage = {}
        all_features = set()

        for group in self.group_fits.keys():
            try:
                # Method 1: Try to get from model fit's molded data
                group_fit = self.group_fits[group]
                if hasattr(group_fit.fit, 'molded') and group_fit.fit.molded is not None:
                    features = list(group_fit.fit.molded.predictors.columns)
                    feature_usage[group] = set(features)
                    all_features.update(features)
                    continue

                # Method 2: Try to get from formula
                if hasattr(group_fit, 'formula') and group_fit.formula:
                    # Parse formula to get predictor names
                    # Formula format: "outcome ~ pred1 + pred2 + pred3"
                    formula = group_fit.formula
                    if '~' in formula:
                        rhs = formula.split('~')[1].strip()
                        # Split by + and clean up whitespace
                        features = [f.strip() for f in rhs.split('+')]
                        feature_usage[group] = set(features)
                        all_features.update(features)
                        continue

                # Method 3: Get from model's training data or fit_data
                if hasattr(group_fit.fit, 'fit_data') and 'X' in group_fit.fit.fit_data:
                    X = group_fit.fit.fit_data['X']
                    if hasattr(X, 'columns'):
                        features = list(X.columns)
                        feature_usage[group] = set(features)
                        all_features.update(features)
                        continue

                # If all methods fail, mark as empty
                feature_usage[group] = set()

            except Exception as e:
                import warnings
                warnings.warn(
                    f"Could not extract feature names for group '{group}': {e}",
                    UserWarning
                )
                feature_usage[group] = set()

        if not all_features:
            print("Could not determine feature names from any group.")
            return None

        # Create comparison DataFrame
        comparison = pd.DataFrame(
            {feature: [feature in feature_usage.get(group, set())
                      for group in sorted(self.group_fits.keys())]
             for feature in sorted(all_features)},
            index=sorted(self.group_fits.keys())
        )

        return comparison

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

            # Preserve date information if available in index or molded data
            # This is needed for plot_forecast() to work
            if "date" not in outputs.columns:
                # Make a copy to avoid SettingWithCopyWarning
                outputs = outputs.copy()

                # Try to get TRAINING dates from stored group_train_data (PRIMARY source)
                # This is more reliable than molded data because it preserves original unprocessed data
                if hasattr(self, 'group_train_data') and group in self.group_train_data:
                    train_data_orig = self.group_train_data[group]
                    if "date" in train_data_orig.columns:
                        train_dates = train_data_orig["date"].values
                        train_mask = outputs['split'] == 'train'
                        if train_mask.sum() == len(train_dates):
                            outputs.loc[train_mask, 'date'] = train_dates
                    elif isinstance(train_data_orig.index, pd.DatetimeIndex):
                        train_dates = train_data_orig.index.values
                        train_mask = outputs['split'] == 'train'
                        if train_mask.sum() == len(train_dates):
                            outputs.loc[train_mask, 'date'] = train_dates
                # Fallback: Try to get date from the fit's molded data (training data)
                elif hasattr(group_fit.fit, 'molded') and group_fit.fit.molded is not None:
                    molded_outcomes = group_fit.fit.molded.outcomes
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

                # Try to get TEST dates from stored group_test_data (PRIMARY source)
                # This is more reliable than evaluation_data because it preserves original unprocessed data
                if hasattr(self, 'group_test_data') and group in self.group_test_data:
                    test_data_orig = self.group_test_data[group]
                    if "date" in test_data_orig.columns:
                        test_dates = test_data_orig["date"].values
                        test_mask = outputs['split'] == 'test'
                        if test_mask.sum() == len(test_dates):
                            outputs.loc[test_mask, 'date'] = test_dates
                    elif isinstance(test_data_orig.index, pd.DatetimeIndex):
                        test_dates = test_data_orig.index.values
                        test_mask = outputs['split'] == 'test'
                        if test_mask.sum() == len(test_dates):
                            outputs.loc[test_mask, 'date'] = test_dates
                # Fallback: Try to get test dates from evaluation data (processed data - may not have date)
                elif hasattr(group_fit.fit, 'evaluation_data') and "test_data" in group_fit.fit.evaluation_data:
                    test_data = group_fit.fit.evaluation_data["test_data"]
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

    def extract_preprocessed_data(
        self,
        data: pd.DataFrame,
        split: str = "train"
    ) -> pd.DataFrame:
        """
        Extract preprocessed data showing what the models see after recipe transformations.

        This is useful for inspecting how recipes transform your data, especially with
        per-group preprocessing where each group may have different transformations
        (e.g., different PCA components, different selected features).

        Args:
            data: Original data to preprocess (train or test data with group column)
            split: Which data split this is ("train" or "test") - for informational purposes

        Returns:
            DataFrame with preprocessed data for all groups, including group column

        Examples:
            >>> # Fit nested workflow with per-group preprocessing
            >>> wf = workflow().add_recipe(recipe().step_normalize()).add_model(linear_reg())
            >>> nested_fit = wf.fit_nested(train_data, group_col='country')
            >>>
            >>> # Extract preprocessed training data
            >>> processed_train = nested_fit.extract_preprocessed_data(train_data, split='train')
            >>> print(processed_train.head())
            >>>
            >>> # After evaluation, extract preprocessed test data
            >>> nested_fit = nested_fit.evaluate(test_data)
            >>> processed_test = nested_fit.extract_preprocessed_data(test_data, split='test')
            >>>
            >>> # Compare preprocessing across groups
            >>> for group in processed_train[nested_fit.group_col].unique():
            ...     group_data = processed_train[processed_train[nested_fit.group_col] == group]
            ...     print(f"{group}: mean={group_data['x1'].mean():.4f}, std={group_data['x1'].std():.4f}")
        """
        if self.group_col not in data.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")

        # Process each group
        processed_groups = []

        for group in data[self.group_col].unique():
            # Get data for this group
            group_data = data[data[self.group_col] == group].copy()

            # Remove group column before preprocessing (it's metadata, not a predictor)
            group_data_no_group = group_data.drop(columns=[self.group_col])

            # Apply preprocessing based on whether per-group recipes exist
            if self.group_recipes is not None and group in self.group_recipes:
                # Per-group preprocessing: use group-specific recipe
                group_recipe = self.group_recipes[group]
                processed = group_recipe.bake(group_data_no_group)
            elif self.group_fits[group].pre is not None:
                # Shared preprocessing: use the recipe from group_fit
                preprocessor = self.group_fits[group].pre
                if hasattr(preprocessor, 'bake'):
                    # It's a PreparedRecipe
                    processed = preprocessor.bake(group_data_no_group)
                else:
                    # It's a formula string - return original data (formula applied during fit/predict)
                    processed = group_data_no_group.copy()
            else:
                # No preprocessing
                processed = group_data_no_group.copy()

            # Add group column back
            processed[self.group_col] = group

            # Add metadata columns if they exist in original data
            for col in ['date', 'split']:
                if col in group_data.columns and col not in processed.columns:
                    processed[col] = group_data[col].values

            processed_groups.append(processed)

        # Combine all groups
        result = pd.concat(processed_groups, ignore_index=True)

        # Add split column if not already present
        if 'split' not in result.columns:
            result['split'] = split

        # Reorder columns: date, group_col, then others
        cols = list(result.columns)
        priority_cols = []

        if 'date' in cols:
            priority_cols.append('date')
            cols.remove('date')

        if self.group_col in cols:
            priority_cols.append(self.group_col)
            cols.remove(self.group_col)

        final_cols = priority_cols + cols
        result = result[final_cols]

        return result

    def extract_formula(self) -> dict:
        """
        Extract the formula used for each group.

        Returns a dictionary mapping group values to their formulas.
        Useful for understanding what formula was used for each group,
        especially when auto-generated or with per-group preprocessing.

        Returns:
            Dict mapping group values to formula strings

        Examples:
            >>> nested_fit = wf.fit_nested(train_data, group_col='country')
            >>> formulas = nested_fit.extract_formula()
            >>> print(formulas)
            >>> # {'USA': 'sales ~ x1 + x2 + x3', 'UK': 'sales ~ x1 + x2 + x3'}
            >>>
            >>> # Check if all groups use same formula
            >>> if len(set(formulas.values())) == 1:
            ...     print(f"All groups use: {list(formulas.values())[0]}")
        """
        formulas = {}
        for group, group_fit in self.group_fits.items():
            formulas[group] = group_fit.extract_formula()
        return formulas

    def extract_spec_parsnip(self) -> ModelSpec:
        """
        Extract the ModelSpec specification.

        This returns the shared model specification used across all groups.
        The spec is the same for all groups (e.g., linear_reg(), prophet_reg()).

        Returns:
            ModelSpec object

        Examples:
            >>> nested_fit = wf.fit_nested(train_data, group_col='country')
            >>> spec = nested_fit.extract_spec_parsnip()
            >>> print(f"Model type: {spec.model_type}")
            >>> print(f"Engine: {spec.engine}")
        """
        return self.workflow.spec

    def extract_preprocessor(self, group: Optional[str] = None) -> Union[Any, dict]:
        """
        Extract the preprocessor (formula or recipe) for groups.

        Args:
            group: If specified, returns preprocessor for that group only.
                  If None, returns dict mapping all groups to their preprocessors.

        Returns:
            If group is specified: Preprocessor for that group (formula string or PreparedRecipe)
            If group is None: Dict mapping group values to their preprocessors

        Raises:
            ValueError: If specified group not found

        Examples:
            >>> # Get preprocessor for specific group
            >>> usa_preprocessor = nested_fit.extract_preprocessor(group='USA')
            >>> if isinstance(usa_preprocessor, str):
            ...     print(f"USA uses formula: {usa_preprocessor}")
            >>> else:
            ...     print(f"USA uses recipe with {len(usa_preprocessor.steps)} steps")
            >>>
            >>> # Get all preprocessors
            >>> all_preprocessors = nested_fit.extract_preprocessor()
            >>> for group, prep in all_preprocessors.items():
            ...     print(f"{group}: {type(prep).__name__}")
        """
        if group is not None:
            # Return specific group's preprocessor
            if group not in self.group_fits:
                raise ValueError(
                    f"Group '{group}' not found. Available groups: {sorted(self.group_fits.keys())}"
                )
            return self.group_fits[group].extract_preprocessor()
        else:
            # Return dict of all preprocessors
            preprocessors = {}
            for grp, group_fit in self.group_fits.items():
                preprocessors[grp] = group_fit.extract_preprocessor()
            return preprocessors

    def extract_fit_parsnip(self, group: Optional[str] = None) -> Union[ModelFit, dict]:
        """
        Extract the fitted ModelFit object(s) for groups.

        Args:
            group: If specified, returns ModelFit for that group only.
                  If None, returns dict mapping all groups to their ModelFits.

        Returns:
            If group is specified: ModelFit for that group
            If group is None: Dict mapping group values to their ModelFits

        Raises:
            ValueError: If specified group not found

        Examples:
            >>> # Get ModelFit for specific group
            >>> usa_fit = nested_fit.extract_fit_parsnip(group='USA')
            >>> print(f"USA model type: {usa_fit.spec.model_type}")
            >>>
            >>> # Get all ModelFits
            >>> all_fits = nested_fit.extract_fit_parsnip()
            >>> for group, fit in all_fits.items():
            ...     outputs, _, stats = fit.extract_outputs()
            ...     print(f"{group} RMSE: {stats[stats['split']=='test']['rmse'].values[0]:.2f}")
        """
        if group is not None:
            # Return specific group's ModelFit
            if group not in self.group_fits:
                raise ValueError(
                    f"Group '{group}' not found. Available groups: {sorted(self.group_fits.keys())}"
                )
            return self.group_fits[group].extract_fit_parsnip()
        else:
            # Return dict of all ModelFits
            model_fits = {}
            for grp, group_fit in self.group_fits.items():
                model_fits[grp] = group_fit.extract_fit_parsnip()
            return model_fits


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
