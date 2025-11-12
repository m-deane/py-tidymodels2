"""
Test extract_formulas() with time series models (prophet, arima).

Time series models use fit_raw() path and store exog_vars instead of X_train.
"""

import pandas as pd
import numpy as np
import pytest
from py_workflowsets import WorkflowSet
from py_parsnip import prophet_reg, arima_reg


def test_extract_formulas_prophet():
    """Test extract_formulas with Prophet model (stores exog_vars)."""
    # Create time series data
    np.random.seed(42)
    n_per_group = 100

    # USA
    usa_dates = pd.date_range('2020-01-01', periods=n_per_group, freq='D')
    usa_data = pd.DataFrame({
        'country': ['USA'] * n_per_group,
        'date': usa_dates,
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'x2': np.random.randn(n_per_group) * 5 + 20,
        'y': np.random.randn(n_per_group) * 100 + 500
    })

    # Germany
    germany_dates = pd.date_range('2020-01-01', periods=n_per_group, freq='D')
    germany_data = pd.DataFrame({
        'country': ['Germany'] * n_per_group,
        'date': germany_dates,
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'x2': np.random.randn(n_per_group) * 5 + 20,
        'y': np.random.randn(n_per_group) * 100 + 500
    })

    data = pd.concat([usa_data, germany_data], ignore_index=True)

    # Create WorkflowSet with Prophet
    try:
        wf_set = WorkflowSet.from_cross(
            preproc=["y ~ date + x1", "y ~ date + x1 + x2"],
            models=[prophet_reg()]
        )

        # Fit on all groups
        results = wf_set.fit_nested(data, group_col='country')

        # Extract formulas
        formulas_df = results.extract_formulas()

        # Verify structure
        assert isinstance(formulas_df, pd.DataFrame), "Should return DataFrame"
        assert 'n_features' in formulas_df.columns, "Should have n_features column"

        # Verify we have entries for both groups
        assert set(formulas_df['group'].unique()) == {'USA', 'Germany'}, \
            "Should have formulas for both groups"

        # Check formulas
        prep_1_df = formulas_df[formulas_df['wflow_id'].str.contains('prep_1')]
        prep_2_df = formulas_df[formulas_df['wflow_id'].str.contains('prep_2')]

        # prep_1 should have 1 exog var (x1) + date
        # Prophet doesn't count date as a feature, so n_features should be 1
        for _, row in prep_1_df.iterrows():
            assert 'x1' in row['formula'], f"prep_1 should include x1"
            assert row['n_features'] == 1, f"prep_1 should have 1 exog feature, got {row['n_features']}"

        # prep_2 should have 2 exog vars (x1 + x2) + date
        for _, row in prep_2_df.iterrows():
            assert 'x1' in row['formula'], f"prep_2 should include x1"
            assert 'x2' in row['formula'], f"prep_2 should include x2"
            assert row['n_features'] == 2, f"prep_2 should have 2 exog features, got {row['n_features']}"

        print("✅ Prophet extract_formulas() test passed")
        print(f"\nFormulas extracted:")
        print(formulas_df[['wflow_id', 'group', 'formula', 'n_features']].to_string(index=False))

    except ImportError as e:
        pytest.skip(f"Prophet not available: {e}")


def test_extract_formulas_arima():
    """Test extract_formulas with ARIMA model (stores exog_vars)."""
    # Create time series data
    np.random.seed(42)
    n_per_group = 100

    # USA
    usa_dates = pd.date_range('2020-01-01', periods=n_per_group, freq='D')
    usa_data = pd.DataFrame({
        'country': ['USA'] * n_per_group,
        'date': usa_dates,
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'y': np.random.randn(n_per_group) * 100 + 500
    })

    # Germany
    germany_dates = pd.date_range('2020-01-01', periods=n_per_group, freq='D')
    germany_data = pd.DataFrame({
        'country': ['Germany'] * n_per_group,
        'date': germany_dates,
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'y': np.random.randn(n_per_group) * 100 + 500
    })

    data = pd.concat([usa_data, germany_data], ignore_index=True)

    # Create WorkflowSet with ARIMA
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ date + x1"],
        models=[arima_reg(non_seasonal_ar=1, non_seasonal_differences=0, non_seasonal_ma=0)]
    )

    # Fit on all groups
    results = wf_set.fit_nested(data, group_col='country')

    # Extract formulas
    formulas_df = results.extract_formulas()

    # Verify structure
    assert isinstance(formulas_df, pd.DataFrame), "Should return DataFrame"
    assert 'n_features' in formulas_df.columns, "Should have n_features column"

    # Verify we have entries for both groups
    assert set(formulas_df['group'].unique()) == {'USA', 'Germany'}, \
        "Should have formulas for both groups"

    # Check formulas include x1
    for _, row in formulas_df.iterrows():
        assert 'x1' in row['formula'], f"Formula should include x1: {row['formula']}"
        assert row['n_features'] == 1, f"Should have 1 exog feature, got {row['n_features']}"

    print("✅ ARIMA extract_formulas() test passed")
    print(f"\nFormulas extracted:")
    print(formulas_df[['wflow_id', 'group', 'formula', 'n_features']].to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing extract_formulas() with Time Series Models")
    print("="*60 + "\n")

    try:
        print("Test 1: Prophet (stores exog_vars)")
        print("-" * 60)
        test_extract_formulas_prophet()
        print()

        print("Test 2: ARIMA (stores exog_vars)")
        print("-" * 60)
        test_extract_formulas_arima()
        print()

        print("="*60)
        print("✅ ALL TIME SERIES TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
