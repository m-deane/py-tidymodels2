"""
Tests for panel/grouped models (nested and global fitting)

Tests cover:
- fit_nested() with multiple groups
- fit_global() with group as a feature
- NestedWorkflowFit predict() for all groups
- NestedWorkflowFit extract_outputs() with group column
- Evaluation on test data
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest, recursive_reg


class TestNestedFitting:
    """Test fit_nested() method for per-group modeling"""

    def test_nested_basic(self):
        """Test basic nested fitting with multiple groups"""
        # Create data with 3 groups
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        data = pd.DataFrame({
            "date": dates.tolist() * 3,
            "store_id": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
            "sales": np.concatenate([
                np.linspace(100, 150, 100) + np.random.normal(0, 5, 100),
                np.linspace(200, 250, 100) + np.random.normal(0, 5, 100),
                np.linspace(50, 100, 100) + np.random.normal(0, 5, 100),
            ])
        })

        # Fit nested model
        wf = (
            workflow()
            .add_formula("sales ~ date")
            .add_model(linear_reg())
        )

        nested_fit = wf.fit_nested(data, group_col="store_id")

        # Check structure
        assert nested_fit.group_col == "store_id"
        assert len(nested_fit.group_fits) == 3
        assert "A" in nested_fit.group_fits
        assert "B" in nested_fit.group_fits
        assert "C" in nested_fit.group_fits

    def test_nested_with_recursive_model(self):
        """Test nested fitting with recursive forecasting model"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        data = []
        for store in ["Store1", "Store2"]:
            store_data = pd.DataFrame({
                "date": dates,
                "store_id": store,
                "sales": np.cumsum(np.random.randn(60)) + 100
            })
            data.append(store_data)

        data = pd.concat(data, ignore_index=True)

        # For recursive models, set date as index before fitting
        data_indexed = data.set_index("date")

        # Fit nested recursive model
        wf = (
            workflow()
            .add_formula("sales ~ .")
            .add_model(recursive_reg(base_model=linear_reg(), lags=7))
        )

        # Reset index for fit_nested, which will re-index per group
        nested_fit = wf.fit_nested(data_indexed.reset_index(), group_col="store_id")

        assert len(nested_fit.group_fits) == 2
        assert "Store1" in nested_fit.group_fits
        assert "Store2" in nested_fit.group_fits

    def test_nested_missing_group_col(self):
        """Test error when group column not in data"""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())

        with pytest.raises(ValueError, match="Group column 'group' not found"):
            wf.fit_nested(data, group_col="group")


class TestNestedPrediction:
    """Test prediction with nested models"""

    def test_nested_predict_basic(self):
        """Test basic prediction for all groups"""
        np.random.seed(42)

        # Create train data with numeric time variable
        train = pd.DataFrame({
            "time": list(range(100)) * 2,
            "store_id": ["A"] * 100 + ["B"] * 100,
            "sales": np.concatenate([
                np.linspace(100, 150, 100),
                np.linspace(200, 250, 100),
            ])
        })

        # Create test data
        test = pd.DataFrame({
            "time": list(range(100, 120)) * 2,
            "store_id": ["A"] * 20 + ["B"] * 20,
        })

        # Fit and predict
        wf = workflow().add_formula("sales ~ time").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="store_id")

        predictions = nested_fit.predict(test)

        # Check predictions
        assert len(predictions) == 40
        assert ".pred" in predictions.columns
        assert "store_id" in predictions.columns
        assert set(predictions["store_id"].unique()) == {"A", "B"}

    def test_nested_predict_missing_group(self):
        """Test prediction when test has different groups"""
        np.random.seed(42)
        train = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 8, 10, 12],
            "group": ["A", "A", "A", "B", "B", "B"]
        })

        test = pd.DataFrame({
            "x": [7, 8],
            "group": ["C", "C"]  # Group C wasn't in training data
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="group")

        # Should raise error for groups not in training
        with pytest.raises(ValueError, match="No matching groups found"):
            nested_fit.predict(test)

    def test_nested_predict_intervals(self):
        """Test prediction intervals with nested models"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        # Create train data
        train_data = []
        for store in ["X", "Y"]:
            store_data = pd.DataFrame({
                "date": dates[:50],
                "store_id": store,
                "sales": np.cumsum(np.random.randn(50)) + 100
            })
            train_data.append(store_data)
        train = pd.concat(train_data, ignore_index=True)
        train = train.set_index("date")

        # Create test data
        test_data = []
        for store in ["X", "Y"]:
            store_data = pd.DataFrame({
                "date": dates[50:],
                "store_id": store,
            })
            test_data.append(store_data)
        test = pd.concat(test_data, ignore_index=True)
        test = test.set_index("date")

        # Fit nested recursive model
        wf = (
            workflow()
            .add_formula("sales ~ .")
            .add_model(recursive_reg(base_model=rand_forest(trees=50), lags=7))
        )
        nested_fit = wf.fit_nested(train.reset_index(), group_col="store_id")

        # Predict with intervals
        predictions = nested_fit.predict(test.reset_index(), type="pred_int")

        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns
        assert "store_id" in predictions.columns


class TestNestedEvaluation:
    """Test evaluation with nested models"""

    def test_nested_evaluate(self):
        """Test evaluate() method for nested models"""
        np.random.seed(42)

        # Create train and test data with numeric time
        full_data = pd.DataFrame({
            "time": list(range(120)) * 2,
            "store_id": ["A"] * 120 + ["B"] * 120,
            "sales": np.concatenate([
                np.linspace(100, 150, 120) + np.random.normal(0, 5, 120),
                np.linspace(200, 250, 120) + np.random.normal(0, 5, 120),
            ])
        })

        train = full_data[full_data["time"] < 100]
        test = full_data[full_data["time"] >= 100]

        # Fit and evaluate
        wf = workflow().add_formula("sales ~ time").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="store_id")
        nested_fit = nested_fit.evaluate(test)

        # Extract outputs should now include test data
        outputs, coeffs, stats = nested_fit.extract_outputs()

        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values
        assert "store_id" in outputs.columns


class TestNestedExtractOutputs:
    """Test extract_outputs() with grouped data"""

    def test_extract_outputs_structure(self):
        """Test that extract_outputs returns three DataFrames with group column"""
        np.random.seed(42)
        data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [2, 4, 6, 8, 10, 12, 14, 16, 18],
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        nested_fit = wf.fit_nested(data, group_col="category")

        outputs, coefficients, stats = nested_fit.extract_outputs()

        # Check all three DataFrames have group column
        assert "category" in outputs.columns
        assert "category" in coefficients.columns
        assert "category" in stats.columns

        # Check all groups present
        assert set(outputs["category"].unique()) == {"A", "B", "C"}
        assert set(coefficients["category"].unique()) == {"A", "B", "C"}
        assert set(stats["category"].unique()) == {"A", "B", "C"}

    def test_extract_outputs_group_comparison(self):
        """Test comparing metrics across groups"""
        np.random.seed(42)

        # Create data with different patterns per group
        data = pd.DataFrame({
            "time": list(range(120)) * 3,
            "region": ["North"] * 120 + ["South"] * 120 + ["East"] * 120,
            "sales": np.concatenate([
                np.linspace(100, 150, 120) + np.random.normal(0, 2, 120),  # North: low noise
                np.linspace(200, 250, 120) + np.random.normal(0, 10, 120),  # South: high noise
                np.linspace(150, 200, 120) + np.random.normal(0, 5, 120),  # East: medium noise
            ])
        })

        # Split train/test
        train = data[data["time"] < 100]
        test = data[data["time"] >= 100]

        # Fit, evaluate, extract
        wf = workflow().add_formula("sales ~ time").add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col="region").evaluate(test)

        outputs, coefficients, stats = nested_fit.extract_outputs()

        # Get test RMSE for each region
        test_rmse = stats[
            (stats["metric"] == "rmse") &
            (stats["split"] == "test")
        ][["region", "value"]].sort_values("value")

        # North should have lowest RMSE (lowest noise)
        assert test_rmse.iloc[0]["region"] == "North"


class TestGlobalFitting:
    """Test fit_global() method with group as feature"""

    def test_global_basic(self):
        """Test basic global fitting with group as feature"""
        np.random.seed(42)
        data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 10, 12, 14],  # Different intercept per group
            "group": ["A", "A", "A", "B", "B", "B"]
        })

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())
        global_fit = wf.fit_global(data, group_col="group")

        # Should be a regular WorkflowFit, not nested
        from py_workflows.workflow import WorkflowFit
        assert isinstance(global_fit, WorkflowFit)

        # Test prediction with group column
        test = pd.DataFrame({
            "x": [7, 8],
            "group": ["A", "B"]
        })

        predictions = global_fit.predict(test)
        assert len(predictions) == 2

    def test_global_with_explicit_group(self):
        """Test global fitting when formula already includes group"""
        data = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "y": [2, 4, 8, 10],
            "category": ["A", "A", "B", "B"]
        })

        # Formula already includes category
        wf = workflow().add_formula("y ~ x + category").add_model(linear_reg())
        global_fit = wf.fit_global(data, group_col="category")

        # Should work without error
        assert global_fit is not None

    def test_global_missing_group_col(self):
        """Test error when group column not in data"""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        wf = workflow().add_formula("y ~ x").add_model(linear_reg())

        with pytest.raises(ValueError, match="Group column 'group' not found"):
            wf.fit_global(data, group_col="group")


class TestNestedVsGlobal:
    """Compare nested vs global modeling approaches"""

    def test_nested_vs_global_comparison(self):
        """Compare performance of nested vs global models"""
        np.random.seed(42)

        # Create data with clear group differences
        data = pd.DataFrame({
            "time": list(range(100)) * 2,
            "store": ["Premium"] * 100 + ["Discount"] * 100,
            "sales": np.concatenate([
                np.linspace(200, 250, 100) + np.random.normal(0, 5, 100),  # Premium
                np.linspace(100, 120, 100) + np.random.normal(0, 5, 100),  # Discount
            ])
        })

        # Split train/test
        train = data[data["time"] < 80]
        test = data[data["time"] >= 80]

        # Nested approach
        wf_nested = workflow().add_formula("sales ~ time").add_model(linear_reg())
        nested_fit = wf_nested.fit_nested(train, group_col="store").evaluate(test)
        _, _, stats_nested = nested_fit.extract_outputs()

        # Global approach
        wf_global = workflow().add_formula("sales ~ time").add_model(linear_reg())
        global_fit = wf_global.fit_global(train, group_col="store").evaluate(test)
        _, _, stats_global = global_fit.extract_outputs()

        # Both should produce valid metrics
        nested_rmse = stats_nested[
            (stats_nested["metric"] == "rmse") &
            (stats_nested["split"] == "test")
        ]["value"]

        global_rmse = stats_global[
            (stats_global["metric"] == "rmse") &
            (stats_global["split"] == "test")
        ]["value"]

        # Nested should generally be better when groups are very different
        # (though not guaranteed in all random seeds)
        assert len(nested_rmse) == 2  # One per group
        assert len(global_rmse) == 1  # Single model
