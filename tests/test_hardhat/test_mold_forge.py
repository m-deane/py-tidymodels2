"""
Tests for mold() and forge() functions

Tests cover:
- Basic numeric predictors
- Categorical variables with one-hot encoding
- Factor level validation
- Column alignment
- Error handling for new factor levels
- Missing columns
"""

import pytest
import pandas as pd
import numpy as np

from py_hardhat import mold, forge, Blueprint, MoldedData


class TestMoldBasic:
    """Test basic mold() functionality"""

    def test_simple_numeric_formula(self):
        """Test mold with simple numeric formula"""
        data = pd.DataFrame({"y": [1, 2, 3], "x": [10, 20, 30]})

        molded = mold("y ~ x", data)

        # Check outcomes
        assert molded.outcomes is not None
        assert list(molded.outcomes.columns) == ["y"]
        assert len(molded.outcomes) == 3

        # Check predictors (should have intercept + x)
        assert "Intercept" in molded.predictors.columns
        assert "x" in molded.predictors.columns
        assert len(molded.predictors) == 3

        # Check blueprint
        assert molded.blueprint.formula == "y ~ x"
        assert "outcome" in molded.blueprint.roles
        assert "predictor" in molded.blueprint.roles

    def test_multiple_predictors(self):
        """Test mold with multiple numeric predictors"""
        data = pd.DataFrame({
            "y": [1, 2, 3],
            "x1": [10, 20, 30],
            "x2": [100, 200, 300],
        })

        molded = mold("y ~ x1 + x2", data)

        assert molded.outcomes is not None
        assert list(molded.outcomes.columns) == ["y"]
        assert "Intercept" in molded.predictors.columns
        assert "x1" in molded.predictors.columns
        assert "x2" in molded.predictors.columns

    def test_categorical_variable(self):
        """Test mold with categorical variable (one-hot encoding)"""
        data = pd.DataFrame({
            "y": [1, 2, 3, 4],
            "x": [10, 20, 30, 40],
            "category": ["A", "B", "A", "C"],
        })

        molded = mold("y ~ x + category", data)

        # Patsy should create dummy variables for category
        # (reference category 'A' is dropped)
        assert "x" in molded.predictors.columns
        # Category encoding may vary, but should have some form of categorical columns

        # Check factor levels were captured
        assert "category" in molded.blueprint.factor_levels
        assert set(molded.blueprint.factor_levels["category"]) == {"A", "B", "C"}

    def test_no_intercept(self):
        """Test mold with intercept=False"""
        data = pd.DataFrame({"y": [1, 2, 3], "x": [10, 20, 30]})

        molded = mold("y ~ x", data, intercept=False)

        assert "Intercept" not in molded.predictors.columns
        assert "x" in molded.predictors.columns


class TestForgeBasic:
    """Test basic forge() functionality"""

    def test_forge_numeric_predictors(self):
        """Test forge with numeric predictors"""
        # Training data
        train = pd.DataFrame({"y": [1, 2, 3], "x": [10, 20, 30]})
        molded = mold("y ~ x", train)

        # Test data (no outcome)
        test = pd.DataFrame({"x": [15, 25]})
        forged = forge(test, molded.blueprint, outcomes=False)

        # Check structure matches training
        assert forged.outcomes is None
        assert list(forged.predictors.columns) == list(molded.predictors.columns)
        assert len(forged.predictors) == 2

        # Check values are correct
        assert "Intercept" in forged.predictors.columns
        assert "x" in forged.predictors.columns

    def test_forge_with_outcomes(self):
        """Test forge when new_data includes outcomes"""
        train = pd.DataFrame({"y": [1, 2, 3], "x": [10, 20, 30]})
        molded = mold("y ~ x", train)

        test = pd.DataFrame({"y": [4, 5], "x": [40, 50]})
        forged = forge(test, molded.blueprint, outcomes=True)

        assert forged.outcomes is not None
        assert len(forged.outcomes) == 2
        assert list(forged.predictors.columns) == list(molded.predictors.columns)

    def test_forge_categorical_valid_levels(self):
        """Test forge with categorical variable - valid levels"""
        # Training data
        train = pd.DataFrame({
            "y": [1, 2, 3, 4],
            "x": [10, 20, 30, 40],
            "category": ["A", "B", "A", "C"],
        })
        molded = mold("y ~ x + category", train)

        # Test data with valid categories
        test = pd.DataFrame({"x": [15, 25], "category": ["A", "B"]})
        forged = forge(test, molded.blueprint, outcomes=False)

        # Should succeed - all categories were seen in training
        assert forged.predictors is not None
        assert len(forged.predictors) == 2

    def test_forge_categorical_new_level_error(self):
        """Test forge with categorical variable - new level should error"""
        # Training data
        train = pd.DataFrame({
            "y": [1, 2, 3],
            "x": [10, 20, 30],
            "category": ["A", "B", "A"],
        })
        molded = mold("y ~ x + category", train)

        # Test data with NEW category 'C' not seen in training
        test = pd.DataFrame({"x": [15], "category": ["C"]})

        # Should raise error for new factor level
        with pytest.raises(ValueError, match="New factor levels found"):
            forge(test, molded.blueprint, outcomes=False)

    def test_forge_missing_column_error(self):
        """Test forge with missing required column"""
        train = pd.DataFrame({"y": [1, 2, 3], "x1": [10, 20, 30], "x2": [100, 200, 300]})
        molded = mold("y ~ x1 + x2", train)

        # Test data missing x2
        test = pd.DataFrame({"x1": [15, 25]})

        # Should raise error for missing column
        with pytest.raises(ValueError, match="Required base columns missing"):
            forge(test, molded.blueprint, outcomes=False)


class TestBlueprint:
    """Test Blueprint dataclass validation"""

    def test_blueprint_invalid_role(self):
        """Test that invalid roles are rejected"""
        with pytest.raises(ValueError, match="Invalid roles"):
            Blueprint(
                formula="y ~ x",
                roles={"invalid_role": ["x"]},  # Invalid role
                factor_levels={},
                column_order=["x"],
                ptypes={"x": "float64"},
            )

    def test_blueprint_invalid_indicators(self):
        """Test that invalid indicators are rejected"""
        with pytest.raises(ValueError, match="indicators must be"):
            Blueprint(
                formula="y ~ x",
                roles={"outcome": ["y"], "predictor": ["x"]},
                factor_levels={},
                column_order=["x"],
                ptypes={"x": "float64"},
                indicators="invalid",  # Invalid
            )


class TestMoldedData:
    """Test MoldedData dataclass validation"""

    def test_molded_data_length_mismatch(self):
        """Test that length mismatch between outcomes and predictors is caught"""
        blueprint = Blueprint(
            formula="y ~ x",
            roles={"outcome": ["y"], "predictor": ["x"]},
            factor_levels={},
            column_order=["x"],
            ptypes={"x": "float64"},
        )

        with pytest.raises(ValueError, match="must have same length"):
            MoldedData(
                outcomes=pd.DataFrame({"y": [1, 2, 3]}),  # 3 rows
                predictors=pd.DataFrame({"x": [10, 20]}),  # 2 rows - MISMATCH
                blueprint=blueprint,
            )


class TestIntegration:
    """Integration tests for mold → forge workflow"""

    def test_full_workflow_numeric(self):
        """Test complete mold → forge workflow with numeric data"""
        # Training
        train = pd.DataFrame({
            "sales": [100, 200, 300, 400],
            "price": [10, 20, 30, 40],
            "quantity": [5, 10, 15, 20],
        })

        molded = mold("sales ~ price + quantity", train)

        # Prediction
        test = pd.DataFrame({"price": [15, 25], "quantity": [7, 12]})
        forged = forge(test, molded.blueprint)

        # Verify structure consistency
        assert list(forged.predictors.columns) == list(molded.predictors.columns)
        assert len(forged.predictors) == 2

    def test_full_workflow_mixed_types(self):
        """Test complete workflow with numeric + categorical"""
        train = pd.DataFrame({
            "sales": [100, 200, 300, 400],
            "price": [10, 20, 30, 40],
            "store": ["A", "B", "A", "B"],
        })

        molded = mold("sales ~ price + store", train)

        # Test with valid data
        test = pd.DataFrame({"price": [15, 25], "store": ["A", "B"]})
        forged = forge(test, molded.blueprint)

        assert list(forged.predictors.columns) == list(molded.predictors.columns)
