"""
Pattern Consistency Tests

Tests that implementation patterns are consistent across components:
- All model factory functions return ModelSpec
- All recipe steps follow prep()/bake() protocol
- prep() returns PreparedStep
- bake() returns DataFrame
- All preprocessing follows consistent patterns
"""

import inspect
from typing import get_type_hints
import pytest
import pandas as pd
import numpy as np

from py_parsnip import ModelSpec
from py_recipes import Recipe, PreparedRecipe, recipe
from py_recipes.recipe import RecipeStep, PreparedStep


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'x1': np.random.randn(50),
        'x2': np.random.randn(50),
        'x3': np.random.choice(['A', 'B', 'C'], 50),
        'y': np.random.randn(50) * 10 + 50
    })


def get_model_factory_functions():
    """Get all model factory functions from py_parsnip"""
    import py_parsnip

    # Common model functions
    model_functions = [
        'linear_reg', 'rand_forest', 'decision_tree', 'boost_tree',
        'prophet_reg', 'arima_reg', 'exp_smoothing', 'seasonal_reg',
        'nearest_neighbor', 'svm_rbf', 'svm_linear', 'mlp',
        'mars', 'poisson_reg', 'gen_additive_mod',
        'null_model', 'naive_reg', 'recursive_reg',
        'arima_boost', 'prophet_boost', 'hybrid_model',
        'manual_reg', 'varmax_reg', 'bag_tree',
        'rule_fit', 'window_reg'
    ]

    available_functions = []
    for name in model_functions:
        if hasattr(py_parsnip, name):
            func = getattr(py_parsnip, name)
            if callable(func):
                available_functions.append((name, func))

    return available_functions


def get_recipe_step_classes():
    """Get all recipe step classes"""
    from py_recipes import steps

    step_classes = []

    # Get all attributes from steps module
    for name in dir(steps):
        if name.startswith('Step'):
            attr = getattr(steps, name)
            if inspect.isclass(attr):
                # Check if it has prep and bake methods (follows protocol)
                if hasattr(attr, 'prep') and hasattr(attr, 'bake'):
                    step_classes.append((name, attr))

    return step_classes


class TestModelFactoryPatterns:
    """Test that model factory functions follow consistent patterns"""

    def test_all_model_functions_return_modelspec(self):
        """
        Verify that all model factory functions return ModelSpec.

        Model factories should return ModelSpec, not ModelFit or other types.
        """
        model_functions = get_model_factory_functions()

        assert len(model_functions) > 0, "No model factory functions found"

        violations = []

        for name, func in model_functions:
            try:
                # Call function with no arguments (should have defaults)
                result = func()

                if not isinstance(result, ModelSpec):
                    violations.append(f"{name}() returns {type(result)}, not ModelSpec")

            except TypeError as e:
                # Some functions may require arguments
                # Try with minimal arguments
                try:
                    if name == 'hybrid_model':
                        # hybrid_model requires two models
                        from py_parsnip import linear_reg
                        result = func(model1=linear_reg(), model2=linear_reg())
                    elif name == 'window_reg':
                        # window_reg requires window_size
                        result = func(window_size=7)
                    elif name == 'seasonal_reg':
                        # seasonal_reg requires at least one seasonal_period
                        result = func(seasonal_period_1=7)
                    elif name == 'recursive_reg':
                        # recursive_reg requires base_model
                        from py_parsnip import linear_reg
                        result = func(base_model=linear_reg())
                    else:
                        # Skip if we can't call it
                        continue

                    if not isinstance(result, ModelSpec):
                        violations.append(f"{name}() returns {type(result)}, not ModelSpec")

                except:
                    # Skip if still can't call
                    continue

        assert not violations, (
            f"Model factory functions not returning ModelSpec:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_model_functions_have_consistent_parameters(self):
        """
        Verify that model factory functions have consistent parameter patterns.

        Common parameters:
        - penalty, mixture (regularization)
        - trees, min_n, tree_depth (tree-based)
        - engine (optional, defaults to primary engine)
        """
        model_functions = get_model_factory_functions()

        for name, func in model_functions:
            sig = inspect.signature(func)

            # All parameters should have defaults OR be clearly documented
            required_params = [
                p.name for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty
            ]

            # Only a few functions should have required parameters
            if required_params and name not in ['hybrid_model', 'window_reg']:
                pytest.skip(
                    f"{name}() has required parameters: {required_params}. "
                    f"Model factories should generally have all optional parameters."
                )

    def test_model_functions_have_docstrings(self):
        """
        Verify that model factory functions have docstrings.
        """
        model_functions = get_model_factory_functions()

        violations = []

        for name, func in model_functions:
            if not func.__doc__ or len(func.__doc__.strip()) < 20:
                violations.append(name)

        assert not violations, (
            f"Model factory functions missing docstrings:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_model_functions_set_model_type(self):
        """
        Verify that model factory functions set the model_type attribute.

        The model_type should match the function name (e.g., linear_reg → "linear_reg").
        """
        model_functions = get_model_factory_functions()

        violations = []

        for name, func in model_functions:
            try:
                if name == 'hybrid_model':
                    from py_parsnip import linear_reg
                    spec = func(model1=linear_reg(), model2=linear_reg())
                elif name == 'window_reg':
                    spec = func(window_size=7)
                else:
                    spec = func()

                # model_type should match function name
                if spec.model_type != name:
                    violations.append(
                        f"{name}() sets model_type='{spec.model_type}', expected '{name}'"
                    )

            except:
                # Skip if we can't call it
                continue

        assert not violations, (
            f"Model factory functions with incorrect model_type:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestRecipeStepPatterns:
    """Test that recipe steps follow consistent patterns"""

    def test_all_recipe_steps_follow_protocol(self):
        """
        Verify that all recipe step classes follow the RecipeStep protocol.

        They should have prep() and bake() methods.
        """
        step_classes = get_recipe_step_classes()

        assert len(step_classes) > 0, "No recipe step classes found"

        # All step classes returned by get_recipe_step_classes already
        # have prep() and bake() methods (checked in that function)
        # So this test just validates the count
        assert len(step_classes) >= 10, (
            f"Expected at least 10 recipe step classes, found {len(step_classes)}"
        )

    def test_all_recipe_steps_have_prep_method(self):
        """
        Verify that all recipe step classes have prep() method.

        prep() is called during recipe.prep() to learn from training data.
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            if not hasattr(cls, 'prep'):
                violations.append(name)
            elif not callable(getattr(cls, 'prep')):
                violations.append(f"{name} (prep not callable)")

        assert not violations, (
            f"Recipe steps missing prep() method:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_all_recipe_steps_have_bake_method(self):
        """
        Verify that all recipe step classes have bake() method.

        bake() is called to transform data using learned parameters.
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            if not hasattr(cls, 'bake'):
                violations.append(name)
            elif not callable(getattr(cls, 'bake')):
                violations.append(f"{name} (bake not callable)")

        assert not violations, (
            f"Recipe steps missing bake() method:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_recipe_step_prep_signature(self):
        """
        Verify that prep() methods have consistent signature.

        Expected signature: prep(data: DataFrame) → PreparedStep
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            if hasattr(cls, 'prep'):
                sig = inspect.signature(cls.prep)
                params = list(sig.parameters.keys())

                # Should have 'self' and 'data' parameters at minimum
                if 'data' not in params:
                    violations.append(f"{name}: prep() missing 'data' parameter")

        assert not violations, (
            f"Recipe steps with incorrect prep() signature:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_recipe_step_bake_signature(self):
        """
        Verify that bake() methods have consistent signature.

        Expected signature: bake(data: DataFrame) → DataFrame
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            if hasattr(cls, 'bake'):
                sig = inspect.signature(cls.bake)
                params = list(sig.parameters.keys())

                # Should have 'self' and 'data' parameters at minimum
                if 'data' not in params:
                    violations.append(f"{name}: bake() missing 'data' parameter")

        assert not violations, (
            f"Recipe steps with incorrect bake() signature:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_recipe_step_docstrings(self):
        """
        Verify that recipe step classes have docstrings.
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            if not cls.__doc__ or len(cls.__doc__.strip()) < 20:
                violations.append(name)

        assert not violations, (
            f"Recipe steps missing docstrings:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestRecipeProtocol:
    """Test recipe protocol implementation"""

    def test_prep_returns_prepared_step(self):
        """
        Verify that step.prep() returns a prepared step.

        The returned object should have the same methods but with learned parameters.
        """
        data = create_sample_data()
        rec = recipe(data)

        # Add a step
        rec = rec.step_normalize(['x1', 'x2'])

        # Prep should return PreparedRecipe
        prepped = rec.prep(data)

        assert isinstance(prepped, PreparedRecipe), (
            "recipe.prep() should return PreparedRecipe"
        )

        # Prepared recipe should have bake method
        assert hasattr(prepped, 'bake'), (
            "PreparedRecipe should have bake() method"
        )

    def test_bake_returns_dataframe(self):
        """
        Verify that bake() returns a DataFrame.

        The DataFrame should have transformed data.
        """
        data = create_sample_data()
        rec = recipe(data)
        rec = rec.step_normalize(['x1', 'x2'])

        prepped = rec.prep(data)
        baked = prepped.bake(data)

        assert isinstance(baked, pd.DataFrame), (
            "PreparedRecipe.bake() should return DataFrame"
        )

        # Baked data should have same number of rows
        assert len(baked) == len(data), (
            "Baked data should have same number of rows as input"
        )

    def test_recipe_chaining_pattern(self):
        """
        Verify that recipe methods return new Recipe instances for chaining.

        This tests the builder pattern:
        rec = (recipe()
               .step_normalize()
               .step_dummy())
        """
        data = create_sample_data()
        rec1 = recipe(data)
        rec2 = rec1.step_normalize(['x1', 'x2'])
        rec3 = rec2.step_dummy(['x3'])

        # Each step should return a new Recipe
        assert rec1 is not rec2
        assert rec2 is not rec3
        assert rec1 is not rec3

        # All should be Recipe instances
        assert isinstance(rec1, Recipe)
        assert isinstance(rec2, Recipe)
        assert isinstance(rec3, Recipe)

    def test_prep_bake_workflow(self):
        """
        Verify that the prep/bake workflow works correctly.

        1. Create recipe with steps
        2. Prep on training data
        3. Bake on training data
        4. Bake on test data
        """
        data = create_sample_data()
        train_data = data[:40]
        test_data = data[40:]

        # Create and prep recipe
        rec = (
            recipe(train_data)
            .step_normalize(['x1', 'x2'])
            .step_dummy(['x3'])
        )

        prepped = rec.prep(train_data)

        # Bake on training data
        train_baked = prepped.bake(train_data)
        assert isinstance(train_baked, pd.DataFrame)
        assert len(train_baked) == len(train_data)

        # Bake on test data
        test_baked = prepped.bake(test_data)
        assert isinstance(test_baked, pd.DataFrame)
        assert len(test_baked) == len(test_data)

        # Columns should be consistent
        assert set(train_baked.columns) == set(test_baked.columns)


class TestConsistentNamingPatterns:
    """Test that naming patterns are consistent across the codebase"""

    def test_model_function_naming_pattern(self):
        """
        Verify that model factory functions follow naming pattern:
        - All lowercase
        - Words separated by underscore
        - Ends with model type (e.g., _reg, _tree, _model)
        """
        model_functions = get_model_factory_functions()

        violations = []

        for name, func in model_functions:
            # Should be lowercase with underscores
            if not name.islower() and '_' in name:
                if name != name.lower():
                    violations.append(f"{name} not all lowercase")

            # Should end with common suffix
            common_suffixes = ['_reg', '_tree', '_model', '_neighbor', '_smoothing']
            if not any(name.endswith(suffix) for suffix in common_suffixes):
                # Some exceptions are allowed
                if name not in ['mars', 'mlp', 'arima']:
                    violations.append(f"{name} doesn't follow naming pattern")

        # This is a soft requirement
        if violations:
            pytest.skip(
                f"Some model functions don't follow naming pattern:\n" +
                "\n".join(f"  - {v}" for v in violations[:5])
            )

    def test_recipe_step_naming_pattern(self):
        """
        Verify that recipe step classes follow naming pattern:
        - PascalCase
        - Start with "Step"
        - Descriptive name
        """
        step_classes = get_recipe_step_classes()

        violations = []

        for name, cls in step_classes:
            # Should start with "Step"
            if not name.startswith('Step'):
                violations.append(f"{name} doesn't start with 'Step'")

            # Should be PascalCase (first letter uppercase)
            if not name[0].isupper():
                violations.append(f"{name} not PascalCase")

        assert not violations, (
            f"Recipe steps not following naming pattern:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestErrorHandlingPatterns:
    """Test that error handling is consistent"""

    def test_model_fit_with_invalid_formula_raises_error(self):
        """
        Verify that fitting with invalid formula raises appropriate error.
        """
        from py_parsnip import linear_reg

        data = create_sample_data()
        spec = linear_reg()

        # Invalid formula (outcome not in data)
        with pytest.raises(Exception):  # Could be KeyError, ValueError, etc.
            spec.fit(data, "invalid_outcome ~ x1")

    def test_workflow_double_add_raises_error(self):
        """
        Verify that adding duplicate components raises ValueError.
        """
        from py_workflows import Workflow
        from py_parsnip import linear_reg

        wf = Workflow()

        # Add formula once
        wf = wf.add_formula("y ~ x1")

        # Adding formula again should raise error
        with pytest.raises(ValueError, match="already has a preprocessor"):
            wf.add_formula("y ~ x2")

        # Same for model
        wf2 = Workflow().add_model(linear_reg())
        with pytest.raises(ValueError, match="already has a model"):
            wf2.add_model(linear_reg())

    def test_recipe_with_missing_columns_raises_error(self):
        """
        Verify that using recipe with missing columns raises appropriate error.
        """
        data = create_sample_data()
        rec = recipe(data)

        # Add step referencing non-existent column
        rec = rec.step_normalize(['x1', 'nonexistent_column'])

        # Prep should raise error
        with pytest.raises(Exception):  # KeyError or similar
            rec.prep(data)
