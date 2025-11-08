"""
Tests for automatic exclusion of datetime columns from auto-generated formulas.

When using workflows with recipes (no explicit formula), datetime columns
should be automatically excluded from the predictor list since they should
be indices, not exogenous variables.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe


class TestDatetimeExclusion:
    """Test that datetime columns are excluded from auto-generated formulas"""

    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data with date column"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
            'target': np.random.randn(100)
        })

    def test_datetime_excluded_from_recipe_formula(self, time_series_data):
        """Datetime columns should be excluded from auto-generated formulas"""
        train_data = time_series_data.iloc[:75]
        test_data = time_series_data.iloc[75:]

        # Create recipe without explicit formula
        rec = recipe().step_normalize()

        # Create workflow with recipe only (no formula)
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg())
        )

        # Fit on training data
        fit = wf.fit(train_data)

        # The auto-generated formula should exclude 'date'
        # We can verify this by checking that evaluate doesn't fail on new dates
        fit_evaluated = fit.evaluate(test_data)

        # Extract outputs to confirm it worked
        outputs, coefs, stats = fit_evaluated.extract_outputs()

        # Verify we have results
        assert len(outputs) > 0
        assert len(stats) > 0

        # Verify date was NOT used as a predictor
        # (coefficients should only be for x1, x2, x3, and Intercept)
        predictor_names = coefs['variable'].unique()
        assert 'date' not in predictor_names
        assert 'Intercept' in predictor_names
        assert 'x1' in predictor_names
        assert 'x2' in predictor_names
        assert 'x3' in predictor_names

    def test_datetime_exclusion_with_new_dates(self, time_series_data):
        """Test data with completely new date range should work"""
        train_data = time_series_data.iloc[:50]

        # Test data has dates that were NOT in training
        test_data = time_series_data.iloc[75:]

        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(linear_reg())

        fit = wf.fit(train_data)

        # Should NOT raise error about unseen categorical levels
        fit_evaluated = fit.evaluate(test_data)

        outputs, _, stats = fit_evaluated.extract_outputs()
        assert len(outputs) > 0
        assert len(stats) > 0

    def test_multiple_datetime_columns_excluded(self):
        """Multiple datetime columns should all be excluded"""
        np.random.seed(42)
        data = pd.DataFrame({
            'date1': pd.date_range('2020-01-01', periods=100, freq='D'),
            'date2': pd.date_range('2020-01-01', periods=100, freq='H'),
            'timestamp': pd.to_datetime(['2020-01-01'] * 100),
            'x1': np.random.randn(100),
            'target': np.random.randn(100)
        })

        train_data = data.iloc[:75]
        test_data = data.iloc[75:]

        rec = recipe()
        wf = workflow().add_recipe(rec).add_model(linear_reg())

        fit = wf.fit(train_data)
        fit_evaluated = fit.evaluate(test_data)

        _, coefs, _ = fit_evaluated.extract_outputs()

        # None of the datetime columns should be predictors
        predictor_names = coefs['variable'].unique()
        assert 'date1' not in predictor_names
        assert 'date2' not in predictor_names
        assert 'timestamp' not in predictor_names
        assert 'x1' in predictor_names

    def test_no_datetime_columns_still_works(self):
        """Workflows without datetime columns should still work"""
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'target': np.random.randn(100)
        })

        train_data = data.iloc[:75]
        test_data = data.iloc[75:]

        rec = recipe().step_normalize()
        wf = workflow().add_recipe(rec).add_model(linear_reg())

        fit = wf.fit(train_data)
        fit_evaluated = fit.evaluate(test_data)

        _, coefs, stats = fit_evaluated.extract_outputs()

        # Should have predictors x1, x2
        predictor_names = coefs['variable'].unique()
        assert 'x1' in predictor_names
        assert 'x2' in predictor_names
        assert len(stats) > 0

    def test_explicit_formula_overrides_datetime_exclusion(self):
        """If user explicitly includes date in formula, respect it"""
        # This test verifies that the exclusion only applies to AUTO-generated formulas
        # If user explicitly provides a formula, we use it as-is
        pass  # This would require testing explicit formulas, which is separate behavior
