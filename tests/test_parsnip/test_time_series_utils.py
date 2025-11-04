"""
Tests for time series utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from py_parsnip.utils import _infer_date_column, _parse_ts_formula, _validate_frequency


class TestInferDateColumn:
    """Tests for _infer_date_column function."""

    def test_explicit_spec_date_col(self):
        """Test using explicitly specified date column."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        result = _infer_date_column(df, spec_date_col='date')
        assert result == 'date'

    def test_fit_date_col_priority(self):
        """Test that fit_date_col takes priority over spec_date_col."""
        df = pd.DataFrame({
            'date1': pd.date_range('2020-01-01', periods=10),
            'date2': pd.date_range('2021-01-01', periods=10),
            'value': range(10)
        })
        result = _infer_date_column(
            df,
            spec_date_col='date2',
            fit_date_col='date1'
        )
        assert result == 'date1'

    def test_datetime_index(self):
        """Test detection of DatetimeIndex."""
        df = pd.DataFrame({
            'value': range(10)
        }, index=pd.date_range('2020-01-01', periods=10))

        result = _infer_date_column(df)
        assert result == '__index__'

    def test_datetime_index_with_fit_date_col(self):
        """Test fit_date_col='__index__' validates DatetimeIndex exists."""
        df = pd.DataFrame({
            'value': range(10)
        }, index=pd.date_range('2020-01-01', periods=10))

        result = _infer_date_column(df, fit_date_col='__index__')
        assert result == '__index__'

    def test_auto_detect_single_datetime(self):
        """Test auto-detection of single datetime column."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10),
            'category': ['A'] * 10
        })
        result = _infer_date_column(df)
        assert result == 'date'

    def test_error_multiple_datetime_columns(self):
        """Test error when multiple datetime columns exist without specification."""
        df = pd.DataFrame({
            'date1': pd.date_range('2020-01-01', periods=10),
            'date2': pd.date_range('2021-01-01', periods=10),
            'value': range(10)
        })
        with pytest.raises(ValueError, match="Multiple datetime columns found"):
            _infer_date_column(df)

    def test_error_no_datetime_columns(self):
        """Test error when no datetime columns exist."""
        df = pd.DataFrame({
            'value': range(10),
            'category': ['A'] * 10
        })
        with pytest.raises(ValueError, match="No datetime column found"):
            _infer_date_column(df)

    def test_error_spec_date_col_not_found(self):
        """Test error when specified date column doesn't exist."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        with pytest.raises(ValueError, match="spec_date_col 'nonexistent' not found"):
            _infer_date_column(df, spec_date_col='nonexistent')

    def test_error_spec_date_col_not_datetime(self):
        """Test error when specified column is not datetime type."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        with pytest.raises(ValueError, match="is not a datetime type"):
            _infer_date_column(df, spec_date_col='value')

    def test_error_fit_date_col_not_found(self):
        """Test error when fit_date_col doesn't exist in data."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        with pytest.raises(ValueError, match="fit_date_col 'nonexistent' not found"):
            _infer_date_column(df, fit_date_col='nonexistent')

    def test_error_fit_date_col_index_mismatch(self):
        """Test error when fit_date_col='__index__' but no DatetimeIndex."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        with pytest.raises(ValueError, match="does not have a DatetimeIndex"):
            _infer_date_column(df, fit_date_col='__index__')


class TestParseTsFormula:
    """Tests for _parse_ts_formula function."""

    def test_standard_formula_with_date(self):
        """Test parsing standard formula with date column."""
        outcome, exog = _parse_ts_formula("sales ~ lag1 + lag2 + date", "date")
        assert outcome == "sales"
        assert exog == ["lag1", "lag2"]

    def test_intercept_only_formula(self):
        """Test parsing intercept-only formula."""
        outcome, exog = _parse_ts_formula("sales ~ 1", "date")
        assert outcome == "sales"
        assert exog == []

    def test_formula_with_date_only(self):
        """Test formula where only date is on right side."""
        outcome, exog = _parse_ts_formula("sales ~ date", "date")
        assert outcome == "sales"
        assert exog == []

    def test_formula_with_all_predictors(self):
        """Test formula with . (all predictors)."""
        outcome, exog = _parse_ts_formula("target ~ .", "date")
        assert outcome == "target"
        assert exog == ["."]

    def test_formula_with_datetime_index(self):
        """Test formula with DatetimeIndex (date_col='__index__')."""
        outcome, exog = _parse_ts_formula("y ~ lag1 + lag2", "__index__")
        assert outcome == "y"
        assert exog == ["lag1", "lag2"]

    def test_multiple_outcomes(self):
        """Test formula with multiple outcome variables (e.g., VARMAX)."""
        outcome, exog = _parse_ts_formula("y1 + y2 ~ x1 + x2 + date", "date")
        assert outcome == "y1 + y2"
        assert exog == ["x1", "x2"]

    def test_formula_with_whitespace(self):
        """Test formula parsing handles extra whitespace."""
        outcome, exog = _parse_ts_formula("  sales  ~  lag1  +  lag2  +  date  ", "date")
        assert outcome == "sales"
        assert exog == ["lag1", "lag2"]

    def test_formula_single_predictor(self):
        """Test formula with single predictor."""
        outcome, exog = _parse_ts_formula("y ~ x1 + date", "date")
        assert outcome == "y"
        assert exog == ["x1"]

    def test_formula_no_date_column(self):
        """Test formula without date column (non-time-series case)."""
        outcome, exog = _parse_ts_formula("y ~ x1 + x2 + x3", "date")
        assert outcome == "y"
        assert exog == ["x1", "x2", "x3"]

    def test_error_missing_tilde(self):
        """Test error when formula missing ~ separator."""
        with pytest.raises(ValueError, match="must contain '~' separator"):
            _parse_ts_formula("sales", "date")

    def test_error_multiple_tildes(self):
        """Test error when formula has multiple ~ separators."""
        with pytest.raises(ValueError, match="exactly one '~' separator"):
            _parse_ts_formula("sales ~ lag1 ~ lag2", "date")

    def test_error_empty_outcome(self):
        """Test error when outcome (left side) is empty."""
        with pytest.raises(ValueError, match="Outcome .* cannot be empty"):
            _parse_ts_formula("~ x1 + x2", "date")

    def test_error_empty_predictors(self):
        """Test error when predictors (right side) is empty."""
        with pytest.raises(ValueError, match="Predictors .* cannot be empty"):
            _parse_ts_formula("sales ~ ", "date")

    def test_complex_outcome_expression(self):
        """Test formula with complex outcome expression."""
        outcome, exog = _parse_ts_formula("log(sales) ~ lag1 + date", "date")
        assert outcome == "log(sales)"
        assert exog == ["lag1"]

    def test_interaction_terms(self):
        """Test formula with interaction terms."""
        outcome, exog = _parse_ts_formula("y ~ x1*x2 + date", "date")
        assert outcome == "y"
        # Note: interaction is treated as single term
        assert "x1*x2" in exog
        assert "date" not in exog


class TestValidateFrequency:
    """Tests for _validate_frequency function."""

    def test_series_with_explicit_frequency(self):
        """Test series that already has frequency set."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        s = pd.Series(range(10), index=dates)

        result = _validate_frequency(s)
        assert result.index.freq is not None
        assert result.index.freq.freqstr == 'D'

    def test_series_infer_daily_frequency(self):
        """Test frequency inference for daily data."""
        dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10, freq='D'))
        dates.freq = None  # Remove frequency
        s = pd.Series(range(10), index=dates)

        result = _validate_frequency(s)
        assert result.index.freq is not None

    def test_series_infer_monthly_frequency(self):
        """Test frequency inference for monthly data."""
        dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=12, freq='MS'))
        dates.freq = None
        s = pd.Series(range(12), index=dates)

        result = _validate_frequency(s)
        assert result.index.freq is not None

    def test_series_infer_hourly_frequency(self):
        """Test frequency inference for hourly data."""
        dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=24, freq='h'))
        dates.freq = None
        s = pd.Series(range(24), index=dates)

        result = _validate_frequency(s)
        assert result.index.freq is not None

    def test_series_fallback_to_common_diff(self):
        """Test fallback to most common difference when inference fails."""
        # Create irregular but mostly regular series
        dates = pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03',
            '2020-01-04', '2020-01-05'
        ])
        s = pd.Series([1, 2, 3, 4, 5], index=dates)

        result = _validate_frequency(s, require_freq=False)
        # Should return series (may or may not have frequency)
        assert isinstance(result, pd.Series)

    def test_require_freq_false_returns_as_is(self):
        """Test that require_freq=False returns series even without frequency."""
        dates = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-08'])
        s = pd.Series([1, 2, 3], index=dates)

        result = _validate_frequency(s, require_freq=False)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_error_not_datetime_index(self):
        """Test error when series doesn't have DatetimeIndex."""
        s = pd.Series(range(10))

        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            _validate_frequency(s)

    def test_error_irregular_frequency_required(self):
        """Test error when frequency required but cannot be inferred."""
        dates = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-08'])
        s = pd.Series([1, 2, 3], index=dates)

        with pytest.raises(ValueError, match="Could not infer frequency"):
            _validate_frequency(s, require_freq=True)

    def test_infer_freq_false_preserves_none(self):
        """Test that infer_freq=False doesn't attempt to infer frequency."""
        dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10, freq='D'))
        dates.freq = None
        s = pd.Series(range(10), index=dates)

        result = _validate_frequency(s, infer_freq=False, require_freq=False)
        assert result.index.freq is None

    def test_series_name_preserved(self):
        """Test that series name is preserved after validation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        s = pd.Series(range(10), index=dates, name='test_series')

        result = _validate_frequency(s)
        assert result.name == 'test_series'

    def test_series_values_preserved(self):
        """Test that series values are preserved after validation."""
        dates = pd.DatetimeIndex(pd.date_range('2020-01-01', periods=10, freq='D'))
        dates.freq = None
        original_values = np.array([1.5, 2.3, 3.1, 4.6, 5.2, 6.8, 7.4, 8.9, 9.1, 10.7])
        s = pd.Series(original_values, index=dates)

        result = _validate_frequency(s)
        np.testing.assert_array_almost_equal(result.values, original_values)


class TestIntegrationScenarios:
    """Integration tests combining multiple utility functions."""

    def test_workflow_with_datetime_index(self):
        """Test typical workflow with DatetimeIndex."""
        # Create data with DatetimeIndex
        df = pd.DataFrame({
            'value': range(10),
            'lag1': range(1, 11)
        }, index=pd.date_range('2020-01-01', periods=10, freq='D'))

        # Infer date column
        date_col = _infer_date_column(df)
        assert date_col == '__index__'

        # Parse formula
        outcome, exog = _parse_ts_formula("value ~ lag1", date_col)
        assert outcome == "value"
        assert exog == ["lag1"]

        # Validate frequency
        series = df['value']
        validated = _validate_frequency(series)
        assert validated.index.freq is not None

    def test_workflow_with_date_column(self):
        """Test typical workflow with explicit date column."""
        # Create data with date column
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10),
            'lag1': range(1, 11),
            'lag2': range(2, 12)
        })

        # Infer date column
        date_col = _infer_date_column(df)
        assert date_col == 'date'

        # Parse formula
        outcome, exog = _parse_ts_formula("value ~ lag1 + lag2 + date", date_col)
        assert outcome == "value"
        assert exog == ["lag1", "lag2"]

    def test_workflow_with_explicit_spec(self):
        """Test workflow with explicitly specified date column."""
        # Create data with multiple datetime columns
        df = pd.DataFrame({
            'date1': pd.date_range('2020-01-01', periods=10),
            'date2': pd.date_range('2021-01-01', periods=10),
            'value': range(10)
        })

        # Specify date column explicitly
        date_col = _infer_date_column(df, spec_date_col='date1')
        assert date_col == 'date1'

        # Parse formula
        outcome, exog = _parse_ts_formula("value ~ date1 + date2", date_col)
        assert outcome == "value"
        assert exog == ["date2"]  # date1 excluded, date2 included

    def test_workflow_prediction_consistency(self):
        """Test that fit_date_col ensures consistency between fit and predict."""
        # Training data
        train_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })

        # Infer date column during fit
        fit_date_col = _infer_date_column(train_df)
        assert fit_date_col == 'date'

        # Prediction data (different structure but same date column)
        pred_df = pd.DataFrame({
            'date': pd.date_range('2020-01-11', periods=5),
            'value': range(5)
        })

        # Use fit_date_col to ensure consistency
        pred_date_col = _infer_date_column(pred_df, fit_date_col=fit_date_col)
        assert pred_date_col == fit_date_col


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
