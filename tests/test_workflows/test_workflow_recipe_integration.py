"""
Tests for workflow and recipe integration
"""

import pytest
import pandas as pd
import numpy as np

from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg, rand_forest


class TestWorkflowRecipeIntegration:
    """Test workflows with recipe preprocessing"""

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data"""
        np.random.seed(42)
        return pd.DataFrame({
            "x1": np.random.randn(100) * 10 + 50,
            "x2": np.random.randn(100) * 5 + 20,
            "category": np.random.choice(["A", "B", "C"], 100),
            "y": np.random.randn(100) * 2 + 10
        })


    def test_workflow_with_basic_recipe(self, regression_data):
        """Test workflow with simple recipe"""
        # Create recipe
        rec = recipe().step_normalize()

        # Create workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit workflow
        wf_fit = wf.fit(regression_data)

        # Make predictions
        predictions = wf_fit.predict(regression_data)

        assert ".pred" in predictions.columns
        assert len(predictions) == len(regression_data)

    def test_workflow_multi_step_recipe(self, regression_data):
        """Test workflow with multi-step recipe"""
        # Create multi-step recipe
        rec = (
            recipe()
            .step_normalize(columns=["x1", "x2"])
            .step_dummy(["category"])
        )

        # Create workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit workflow
        wf_fit = wf.fit(regression_data)

        # Make predictions
        predictions = wf_fit.predict(regression_data)

        assert ".pred" in predictions.columns
        assert len(predictions) == len(regression_data)

    def test_recipe_imputation_with_workflow(self):
        """Test recipe with imputation in workflow"""
        # Data with missing values
        data = pd.DataFrame({
            "x1": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "x2": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0, 90.0, 100.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })

        # Recipe with imputation
        rec = recipe().step_impute_mean().step_normalize()

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit
        wf_fit = wf.fit(data)

        # Predict
        predictions = wf_fit.predict(data)

        assert len(predictions) == len(data)
        assert not predictions[".pred"].isna().any()

    def test_train_test_consistency(self, regression_data):
        """Test that recipe transformations are consistent for train/test"""
        # Split data
        train = regression_data[:80]
        test = regression_data[80:]

        # Recipe
        rec = (
            recipe()
            .step_normalize(columns=["x1", "x2"])
            .step_dummy(["category"])
        )

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit on train
        wf_fit = wf.fit(train)

        # Predict on test
        test_predictions = wf_fit.predict(test)

        assert len(test_predictions) == len(test)
        assert ".pred" in test_predictions.columns

    def test_recipe_with_mutate(self, regression_data):
        """Test recipe with custom transformations"""
        # Recipe with mutate
        rec = (
            recipe()
            .step_mutate({
                "x1_squared": lambda df: df["x1"] ** 2,
                "x1_x2": lambda df: df["x1"] * df["x2"]
            })
            .step_normalize()
        )

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit
        wf_fit = wf.fit(regression_data)

        # Extract preprocessor to check transformations
        prep_recipe = wf_fit.extract_preprocessor()
        transformed = prep_recipe.bake(regression_data)

        assert "x1_squared" in transformed.columns
        assert "x1_x2" in transformed.columns

    def test_recipe_with_random_forest(self, regression_data):
        """Test recipe integration with random forest"""
        # Recipe
        rec = (
            recipe()
            .step_normalize()
            .step_dummy(["category"])
        )

        # Workflow with random forest
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(rand_forest().set_engine("sklearn").set_mode("regression"))
        )

        # Fit
        wf_fit = wf.fit(regression_data)

        # Predict
        predictions = wf_fit.predict(regression_data)

        assert len(predictions) == len(regression_data)

    def test_extract_preprocessor(self, regression_data):
        """Test extracting fitted recipe from workflow"""
        # Recipe
        rec = recipe().step_normalize()

        # Workflow
        wf = workflow().add_recipe(rec).add_model(linear_reg().set_engine("sklearn"))

        # Fit
        wf_fit = wf.fit(regression_data)

        # Extract preprocessor
        prep_recipe = wf_fit.extract_preprocessor()

        # Should be a PreparedRecipe
        from py_recipes import PreparedRecipe
        assert isinstance(prep_recipe, PreparedRecipe)

        # Should be able to bake new data
        transformed = prep_recipe.bake(regression_data)
        assert len(transformed) == len(regression_data)

    def test_workflow_extract_outputs_with_recipe(self, regression_data):
        """Test extract_outputs with recipe workflow"""
        # Recipe
        rec = recipe().step_normalize().step_dummy(["category"])

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit
        wf_fit = wf.fit(regression_data)

        # Extract outputs
        outputs, coefficients, stats = wf_fit.extract_outputs()

        # Check outputs
        assert len(outputs) == len(regression_data)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "split" in outputs.columns

        # Check coefficients
        assert len(coefficients) > 0
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Check stats
        assert len(stats) > 0
        assert "metric" in stats.columns
        assert "value" in stats.columns

    def test_recipe_preserves_extra_columns(self, regression_data):
        """Test that recipe preserves columns not in the transformations"""
        # Add extra column
        data = regression_data.copy()
        data["id"] = range(len(data))

        # Recipe that doesn't touch 'id'
        rec = recipe().step_normalize(columns=["x1", "x2"])

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit
        wf_fit = wf.fit(data)

        # Extract preprocessor and bake
        prep_recipe = wf_fit.extract_preprocessor()
        transformed = prep_recipe.bake(data)

        # 'id' should still be there
        assert "id" in transformed.columns
        assert transformed["id"].tolist() == data["id"].tolist()

    def test_empty_recipe_with_workflow(self, regression_data):
        """Test workflow with empty recipe (no-op preprocessing)"""
        # Empty recipe
        rec = recipe()

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit
        wf_fit = wf.fit(regression_data)

        # Predict
        predictions = wf_fit.predict(regression_data)

        assert len(predictions) == len(regression_data)

    def test_workflow_evaluate_with_recipe(self, regression_data):
        """Test workflow.evaluate() with recipe preprocessing"""
        # Split data
        train = regression_data[:80]
        test = regression_data[80:]

        # Recipe
        rec = recipe().step_normalize().step_dummy(["category"])

        # Workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg().set_engine("sklearn"))
        )

        # Fit and evaluate
        wf_fit = wf.fit(train).evaluate(test)

        # Extract outputs
        outputs, coefficients, stats = wf_fit.extract_outputs()

        # Should have both train and test data
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Stats should have both splits
        assert "train" in stats["split"].values
        assert "test" in stats["split"].values
