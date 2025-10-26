"""
Tests for workflow composition
"""

import pytest
import pandas as pd
import numpy as np

from py_workflows import workflow, Workflow, WorkflowFit
from py_parsnip import linear_reg, rand_forest


class TestWorkflowCreation:
    """Test workflow creation and composition"""

    def test_create_empty_workflow(self):
        """Test creating an empty workflow"""
        wf = workflow()
        assert isinstance(wf, Workflow)
        assert wf.preprocessor is None
        assert wf.spec is None
        assert wf.post is None

    def test_add_formula(self):
        """Test adding a formula to workflow"""
        wf = workflow().add_formula("y ~ x1 + x2")
        assert wf.preprocessor == "y ~ x1 + x2"
        assert wf.spec is None

    def test_add_model(self):
        """Test adding a model to workflow"""
        spec = linear_reg().set_engine("sklearn")
        wf = workflow().add_model(spec)
        assert wf.spec == spec
        assert wf.preprocessor is None

    def test_add_formula_and_model(self):
        """Test adding both formula and model"""
        spec = linear_reg().set_engine("sklearn")
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(spec)
        )
        assert wf.preprocessor == "y ~ x1 + x2"
        assert wf.spec == spec

    def test_add_formula_twice_raises_error(self):
        """Test that adding formula twice raises error"""
        wf = workflow().add_formula("y ~ x1")
        with pytest.raises(ValueError, match="already has a preprocessor"):
            wf.add_formula("y ~ x2")

    def test_add_model_twice_raises_error(self):
        """Test that adding model twice raises error"""
        spec = linear_reg().set_engine("sklearn")
        wf = workflow().add_model(spec)
        with pytest.raises(ValueError, match="already has a model"):
            wf.add_model(spec)

    def test_workflow_immutability(self):
        """Test that workflow is immutable"""
        wf1 = workflow()
        wf2 = wf1.add_formula("y ~ x")
        assert wf1.preprocessor is None
        assert wf2.preprocessor == "y ~ x"

    def test_remove_formula(self):
        """Test removing formula"""
        wf = workflow().add_formula("y ~ x")
        wf_removed = wf.remove_formula()
        assert wf.preprocessor == "y ~ x"
        assert wf_removed.preprocessor is None

    def test_remove_model(self):
        """Test removing model"""
        spec = linear_reg().set_engine("sklearn")
        wf = workflow().add_model(spec)
        wf_removed = wf.remove_model()
        assert wf.spec == spec
        assert wf_removed.spec is None

    def test_update_formula(self):
        """Test updating formula"""
        wf = workflow().add_formula("y ~ x1")
        wf_updated = wf.update_formula("y ~ x1 + x2")
        assert wf.preprocessor == "y ~ x1"
        assert wf_updated.preprocessor == "y ~ x1 + x2"

    def test_update_model(self):
        """Test updating model"""
        spec1 = linear_reg().set_engine("sklearn")
        spec2 = linear_reg(penalty=0.1).set_engine("sklearn")
        wf = workflow().add_model(spec1)
        wf_updated = wf.update_model(spec2)
        assert wf.spec == spec1
        assert wf_updated.spec == spec2


class TestWorkflowFit:
    """Test fitting workflows"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": np.random.randn(100) + 10,
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
        })

    def test_fit_workflow(self, sample_data):
        """Test fitting a workflow"""
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(sample_data)
        assert isinstance(wf_fit, WorkflowFit)
        assert wf_fit.workflow == wf
        assert wf_fit.pre == "y ~ x1 + x2"
        assert wf_fit.fit is not None

    def test_fit_without_model_raises_error(self, sample_data):
        """Test that fitting without model raises error"""
        wf = workflow().add_formula("y ~ x1 + x2")
        with pytest.raises(ValueError, match="must have a model"):
            wf.fit(sample_data)

    def test_fit_without_formula_raises_error(self, sample_data):
        """Test that fitting without formula raises error"""
        wf = workflow().add_model(linear_reg().set_engine("sklearn"))
        with pytest.raises(ValueError, match="must have a formula"):
            wf.fit(sample_data)

    def test_predict_after_fit(self, sample_data):
        """Test prediction after fitting"""
        train = sample_data[:80]
        test = sample_data[80:]

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(train)
        predictions = wf_fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test)

    def test_evaluate_after_fit(self, sample_data):
        """Test evaluation after fitting"""
        train = sample_data[:80]
        test = sample_data[80:]

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(train).evaluate(test)

        # Check that evaluation data is stored
        assert "test_predictions" in wf_fit.fit.evaluation_data
        assert "test_data" in wf_fit.fit.evaluation_data

    def test_extract_fit_parsnip(self, sample_data):
        """Test extracting parsnip fit"""
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(sample_data)
        model_fit = wf_fit.extract_fit_parsnip()

        assert model_fit is wf_fit.fit
        assert "model" in model_fit.fit_data

    def test_extract_preprocessor(self, sample_data):
        """Test extracting preprocessor"""
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(sample_data)
        preprocessor = wf_fit.extract_preprocessor()

        assert preprocessor == "y ~ x1 + x2"

    def test_extract_spec_parsnip(self, sample_data):
        """Test extracting model spec"""
        spec = linear_reg().set_engine("sklearn")
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(spec)
        )
        wf_fit = wf.fit(sample_data)
        extracted_spec = wf_fit.extract_spec_parsnip()

        assert extracted_spec == spec

    def test_extract_outputs(self, sample_data):
        """Test extracting comprehensive outputs"""
        train = sample_data[:80]
        test = sample_data[80:]

        wf = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(train).evaluate(test)
        outputs, coefficients, stats = wf_fit.extract_outputs()

        # Check outputs DataFrame
        assert isinstance(outputs, pd.DataFrame)
        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "forecast" in outputs.columns
        assert "split" in outputs.columns

        # Check that both train and test are present
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Check coefficients DataFrame
        assert isinstance(coefficients, pd.DataFrame)
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns

        # Check stats DataFrame
        assert isinstance(stats, pd.DataFrame)
        assert "metric" in stats.columns
        assert "value" in stats.columns
        assert "split" in stats.columns


class TestWorkflowWithDifferentModels:
    """Test workflow with different model types"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": np.random.randn(100) + 10,
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "x3": np.random.randn(100),
        })

    def test_workflow_with_regularized_model(self, sample_data):
        """Test workflow with Ridge regression"""
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(linear_reg(penalty=0.1, mixture=0.0).set_engine("sklearn"))
        )
        wf_fit = wf.fit(sample_data)
        predictions = wf_fit.predict(sample_data)

        assert ".pred" in predictions.columns
        assert len(predictions) == len(sample_data)

    def test_workflow_with_random_forest(self, sample_data):
        """Test workflow with random forest"""
        wf = (
            workflow()
            .add_formula("y ~ x1 + x2 + x3")
            .add_model(
                rand_forest(trees=10, min_n=5)
                .set_mode("regression")
                .set_engine("sklearn")
            )
        )
        wf_fit = wf.fit(sample_data)
        predictions = wf_fit.predict(sample_data)

        assert ".pred" in predictions.columns
        assert len(predictions) == len(sample_data)

    def test_workflow_method_chaining(self, sample_data):
        """Test method chaining for workflow operations"""
        train = sample_data[:80]
        test = sample_data[80:]

        # Full pipeline in one chain
        outputs, coefficients, stats = (
            workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
            .fit(train)
            .evaluate(test)
            .extract_outputs()
        )

        assert isinstance(outputs, pd.DataFrame)
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values


class TestWorkflowEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": np.random.randn(50),
            "x": np.random.randn(50),
        })

    def test_workflow_with_simple_formula(self, sample_data):
        """Test workflow with simple one-predictor formula"""
        wf = (
            workflow()
            .add_formula("y ~ x")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(sample_data)
        predictions = wf_fit.predict(sample_data)

        assert len(predictions) == len(sample_data)

    def test_workflow_prediction_on_different_sized_data(self, sample_data):
        """Test prediction on data with different size"""
        train = sample_data[:40]
        test = sample_data[40:]

        wf = (
            workflow()
            .add_formula("y ~ x")
            .add_model(linear_reg().set_engine("sklearn"))
        )
        wf_fit = wf.fit(train)
        predictions = wf_fit.predict(test)

        assert len(predictions) == len(test)

    def test_workflow_preserves_model_properties(self, sample_data):
        """Test that workflow preserves model properties"""
        spec = linear_reg(penalty=0.5, mixture=0.5).set_engine("sklearn")
        wf = workflow().add_formula("y ~ x").add_model(spec)
        wf_fit = wf.fit(sample_data)

        # Check that spec is preserved
        assert wf_fit.workflow.spec == spec
        assert dict(wf_fit.workflow.spec.args)["penalty"] == 0.5
        assert dict(wf_fit.workflow.spec.args)["mixture"] == 0.5
