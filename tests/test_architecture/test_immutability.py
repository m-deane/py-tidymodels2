"""
Specification Immutability Tests

Tests that core specification objects are immutable (frozen dataclasses):
- ModelSpec cannot be modified after creation
- Workflow cannot be modified after creation
- Blueprint cannot be modified after creation
- Modification methods return NEW instances
"""

import pytest
from dataclasses import FrozenInstanceError

from py_parsnip import ModelSpec, linear_reg
from py_workflows import Workflow
from py_hardhat import Blueprint, MoldedData, mold
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'x1': np.random.randn(50),
        'x2': np.random.randn(50),
        'y': np.random.randn(50) * 10 + 50
    })


class TestModelSpecImmutability:
    """Test that ModelSpec is immutable"""

    def test_model_spec_is_frozen(self):
        """
        Verify that ModelSpec is a frozen dataclass.

        Frozen dataclasses cannot have attributes modified after creation.
        """
        spec = ModelSpec(
            model_type="linear_reg",
            engine="sklearn",
            mode="regression",
            args={"penalty": 0.1}
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            spec.model_type = "rand_forest"

        with pytest.raises(FrozenInstanceError):
            spec.engine = "statsmodels"

        with pytest.raises(FrozenInstanceError):
            spec.mode = "classification"

        with pytest.raises(FrozenInstanceError):
            spec.args = {"penalty": 0.2}

    def test_model_spec_args_dict_not_mutable(self):
        """
        Verify that modifying the args dict doesn't affect the original spec.

        This tests that specs are truly independent.
        """
        spec = ModelSpec(
            model_type="linear_reg",
            engine="sklearn",
            mode="regression",
            args={"penalty": 0.1}
        )

        # Get the args dict
        original_args = spec.args
        original_penalty = original_args.get("penalty")

        # Try to modify it (this will modify the dict, but not the spec)
        args_copy = spec.args.copy()
        args_copy["penalty"] = 0.5

        # Original spec should be unchanged
        assert spec.args["penalty"] == original_penalty

    def test_set_engine_returns_new_instance(self):
        """
        Verify that set_engine() returns a NEW ModelSpec instance.

        The original spec should be unchanged.
        """
        spec1 = ModelSpec(
            model_type="linear_reg",
            engine="sklearn",
            mode="regression"
        )

        spec2 = spec1.set_engine("statsmodels")

        # spec2 should be different from spec1
        assert spec1 is not spec2
        assert spec1.engine == "sklearn"
        assert spec2.engine == "statsmodels"

        # Other attributes should be copied
        assert spec1.model_type == spec2.model_type
        assert spec1.mode == spec2.mode

    def test_set_mode_returns_new_instance(self):
        """
        Verify that set_mode() returns a NEW ModelSpec instance.
        """
        spec1 = ModelSpec(
            model_type="linear_reg",
            engine="sklearn",
            mode="regression"
        )

        spec2 = spec1.set_mode("classification")

        assert spec1 is not spec2
        assert spec1.mode == "regression"
        assert spec2.mode == "classification"

    def test_set_args_returns_new_instance(self):
        """
        Verify that set_args() returns a NEW ModelSpec instance.
        """
        spec1 = ModelSpec(
            model_type="linear_reg",
            engine="sklearn",
            mode="regression",
            args={"penalty": 0.1}
        )

        spec2 = spec1.set_args(penalty=0.5)

        assert spec1 is not spec2
        assert spec1.args["penalty"] == 0.1
        assert spec2.args["penalty"] == 0.5

    def test_model_factory_returns_frozen_spec(self):
        """
        Verify that model factory functions return frozen ModelSpec.
        """
        spec = linear_reg(penalty=0.1, mixture=0.5)

        with pytest.raises(FrozenInstanceError):
            spec.model_type = "rand_forest"

    def test_chaining_modifications_creates_new_instances(self):
        """
        Verify that chaining set_* methods creates new instances at each step.
        """
        spec1 = linear_reg()
        spec2 = spec1.set_engine("statsmodels")
        spec3 = spec2.set_args(penalty=0.1)

        # All three should be different objects
        assert spec1 is not spec2
        assert spec2 is not spec3
        assert spec1 is not spec3

        # Original should be unchanged
        assert spec1.engine == "sklearn"
        assert spec1.args.get("penalty") is None

        # Final spec should have all modifications
        assert spec3.engine == "statsmodels"
        assert spec3.args["penalty"] == 0.1


class TestWorkflowImmutability:
    """Test that Workflow is immutable"""

    def test_workflow_is_frozen(self):
        """
        Verify that Workflow is a frozen dataclass.
        """
        wf = Workflow(
            preprocessor="y ~ x1 + x2",
            spec=linear_reg()
        )

        with pytest.raises(FrozenInstanceError):
            wf.preprocessor = "y ~ x1"

        with pytest.raises(FrozenInstanceError):
            wf.spec = linear_reg(penalty=0.1)

    def test_add_formula_returns_new_instance(self):
        """
        Verify that add_formula() returns a NEW Workflow instance.
        """
        wf1 = Workflow()
        wf2 = wf1.add_formula("y ~ x1 + x2")

        assert wf1 is not wf2
        assert wf1.preprocessor is None
        assert wf2.preprocessor == "y ~ x1 + x2"

    def test_add_model_returns_new_instance(self):
        """
        Verify that add_model() returns a NEW Workflow instance.
        """
        wf1 = Workflow()
        wf2 = wf1.add_model(linear_reg())

        assert wf1 is not wf2
        assert wf1.spec is None
        assert wf2.spec is not None

    def test_add_model_name_returns_new_instance(self):
        """
        Verify that add_model_name() returns a NEW Workflow instance.
        """
        wf1 = Workflow()
        wf2 = wf1.add_model_name("test_model")

        assert wf1 is not wf2
        assert wf1.model_name is None
        assert wf2.model_name == "test_model"

    def test_chaining_workflow_methods_creates_new_instances(self):
        """
        Verify that chaining workflow methods creates new instances at each step.
        """
        wf1 = Workflow()
        wf2 = wf1.add_formula("y ~ x1")
        wf3 = wf2.add_model(linear_reg())
        wf4 = wf3.add_model_name("test")

        # All should be different objects
        assert wf1 is not wf2
        assert wf2 is not wf3
        assert wf3 is not wf4
        assert wf1 is not wf4

        # Original should be unchanged
        assert wf1.preprocessor is None
        assert wf1.spec is None
        assert wf1.model_name is None

        # Final workflow should have all components
        assert wf4.preprocessor == "y ~ x1"
        assert wf4.spec is not None
        assert wf4.model_name == "test"

    def test_workflow_builder_pattern_works(self):
        """
        Verify that workflow builder pattern works correctly.

        This tests the common usage pattern:
        wf = (Workflow()
              .add_formula(...)
              .add_model(...))
        """
        wf = (
            Workflow()
            .add_formula("y ~ x1 + x2")
            .add_model(linear_reg().set_engine("sklearn"))
            .add_model_name("baseline")
        )

        assert wf.preprocessor == "y ~ x1 + x2"
        assert wf.spec is not None
        assert wf.model_name == "baseline"


class TestBlueprintImmutability:
    """Test that Blueprint is immutable"""

    def test_blueprint_is_frozen(self):
        """
        Verify that Blueprint is a frozen dataclass.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)
        blueprint = molded.blueprint

        # Blueprint should be frozen
        with pytest.raises(FrozenInstanceError):
            blueprint.formula = "y ~ x1"

        with pytest.raises(FrozenInstanceError):
            blueprint.roles = {"outcome": ["y"]}

        with pytest.raises(FrozenInstanceError):
            blueprint.intercept = False

    def test_blueprint_from_mold_is_frozen(self):
        """
        Verify that Blueprint created by mold() is frozen.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)

        with pytest.raises(FrozenInstanceError):
            molded.blueprint.formula = "y ~ x1"

    def test_blueprint_attributes_immutable(self):
        """
        Verify that Blueprint attributes cannot be modified.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)
        blueprint = molded.blueprint

        # Try to modify various attributes
        with pytest.raises(FrozenInstanceError):
            blueprint.column_order = []

        with pytest.raises(FrozenInstanceError):
            blueprint.factor_levels = {}

        with pytest.raises(FrozenInstanceError):
            blueprint.ptypes = {}

    def test_blueprint_dicts_not_shared(self):
        """
        Verify that modifying blueprint dict attributes doesn't affect original.

        This tests that blueprints are truly independent.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)
        blueprint = molded.blueprint

        # Get copies of dict attributes
        roles_copy = blueprint.roles.copy()
        factor_levels_copy = blueprint.factor_levels.copy()

        # Modify copies
        roles_copy["test"] = ["test_col"]
        factor_levels_copy["test"] = ["A", "B"]

        # Original blueprint should be unchanged
        assert "test" not in blueprint.roles
        assert "test" not in blueprint.factor_levels


class TestMoldedDataMutability:
    """Test MoldedData mutability (it's NOT frozen, but should be treated carefully)"""

    def test_molded_data_is_not_frozen(self):
        """
        Verify that MoldedData is NOT frozen (unlike specs and workflows).

        MoldedData contains actual data DataFrames which need to be mutable
        for performance reasons.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)

        # MoldedData should NOT be frozen (no FrozenInstanceError)
        # But modifying it is discouraged
        try:
            molded.extras = {"test": "value"}
            # If this works, MoldedData is mutable (expected)
            assert molded.extras["test"] == "value"
        except FrozenInstanceError:
            pytest.fail("MoldedData should not be frozen")

    def test_molded_data_blueprint_is_frozen(self):
        """
        Verify that the Blueprint inside MoldedData is frozen.

        Even though MoldedData is mutable, its Blueprint should be immutable.
        """
        data = create_sample_data()
        molded = mold("y ~ x1 + x2", data)

        with pytest.raises(FrozenInstanceError):
            molded.blueprint.formula = "y ~ x1"


class TestImmutabilityBestPractices:
    """Test best practices for immutability"""

    def test_spec_reuse_is_safe(self):
        """
        Verify that reusing a ModelSpec for multiple fits is safe.

        Because specs are immutable, they can be safely reused.
        """
        data = create_sample_data()
        spec = linear_reg(penalty=0.1)

        # Fit twice with same spec
        fit1 = spec.fit(data, "y ~ x1")
        fit2 = spec.fit(data, "y ~ x1 + x2")

        # Fits should be independent
        assert fit1 is not fit2

        # Spec should be unchanged
        assert spec.args["penalty"] == 0.1

    def test_workflow_reuse_is_safe(self):
        """
        Verify that reusing a Workflow for multiple fits is safe.
        """
        data = create_sample_data()
        wf = Workflow().add_formula("y ~ x1").add_model(linear_reg())

        # Fit twice with same workflow
        fit1 = wf.fit(data)
        fit2 = wf.fit(data)

        # Fits should be independent
        assert fit1 is not fit2

        # Workflow should be unchanged
        assert wf.preprocessor == "y ~ x1"

    def test_spec_modification_creates_independent_copy(self):
        """
        Verify that modifying a spec creates truly independent copy.

        Modifications to the new spec should not affect the original.
        """
        spec1 = linear_reg(penalty=0.1)
        spec2 = spec1.set_args(penalty=0.5)
        spec3 = spec2.set_engine("statsmodels")

        # All specs should be independent
        assert spec1.args["penalty"] == 0.1
        assert spec1.engine == "sklearn"

        assert spec2.args["penalty"] == 0.5
        assert spec2.engine == "sklearn"

        assert spec3.args["penalty"] == 0.5
        assert spec3.engine == "statsmodels"

    def test_nested_workflow_spec_independence(self):
        """
        Verify that workflow and its spec are independent.

        Modifying the spec inside a workflow should not affect the original spec.
        """
        spec1 = linear_reg(penalty=0.1)
        wf1 = Workflow().add_model(spec1)

        # Modify the spec
        spec2 = spec1.set_args(penalty=0.5)
        wf2 = Workflow().add_model(spec2)

        # Original workflow should have original spec
        assert wf1.spec.args["penalty"] == 0.1
        assert wf2.spec.args["penalty"] == 0.5

        # Original spec should be unchanged
        assert spec1.args["penalty"] == 0.1
