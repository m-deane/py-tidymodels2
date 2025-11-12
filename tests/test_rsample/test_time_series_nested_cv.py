"""
Test time_series_nested_cv() and time_series_global_cv() functions for group-aware CV splits.
"""

import pandas as pd
import numpy as np
import pytest
from py_rsample import time_series_nested_cv, time_series_global_cv, TimeSeriesCV


def create_panel_data():
    """Create sample panel data with multiple groups."""
    np.random.seed(42)
    data_frames = []

    for country in ['USA', 'Germany', 'Japan']:
        df = pd.DataFrame({
            'country': [country] * 200,
            'date': pd.date_range('2020-01-01', periods=200, freq='D'),
            'value': np.random.randn(200) * 10 + 100
        })
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def test_time_series_nested_cv_basic():
    """Test basic functionality of time_series_nested_cv."""
    data = create_panel_data()

    cv_by_group = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='100 days',
        assess='30 days'
    )

    # Should have 3 groups
    assert len(cv_by_group) == 3
    assert set(cv_by_group.keys()) == {'USA', 'Germany', 'Japan'}

    # Each group should have TimeSeriesCV object
    for group, cv in cv_by_group.items():
        assert isinstance(cv, TimeSeriesCV)
        assert len(cv) > 0  # Should have at least one fold

    print("✅ time_series_nested_cv basic test passed")


def test_time_series_nested_cv_parameters():
    """Test that all time_series_cv parameters work."""
    data = create_panel_data()

    cv_by_group = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='80 days',
        assess='20 days',
        skip='10 days',
        cumulative=False,  # Rolling window
        lag='5 days'
    )

    # Verify each group got the parameters
    for group, cv in cv_by_group.items():
        assert cv.cumulative == False
        assert len(cv) > 0

    print("✅ time_series_nested_cv parameters test passed")


def test_time_series_nested_cv_fold_count():
    """Test that fold count varies by group based on data size."""
    # Create data with different sizes per group
    data_frames = []

    # USA: 200 days (should get more folds)
    data_frames.append(pd.DataFrame({
        'country': ['USA'] * 200,
        'date': pd.date_range('2020-01-01', periods=200, freq='D'),
        'value': np.random.randn(200)
    }))

    # Germany: 150 days (should get fewer folds)
    data_frames.append(pd.DataFrame({
        'country': ['Germany'] * 150,
        'date': pd.date_range('2020-01-01', periods=150, freq='D'),
        'value': np.random.randn(150)
    }))

    data = pd.concat(data_frames, ignore_index=True)

    cv_by_group = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='80 days',
        assess='20 days',
        skip='10 days'
    )

    # USA should have more folds than Germany
    assert len(cv_by_group['USA']) > len(cv_by_group['Germany'])

    print("✅ time_series_nested_cv fold count test passed")


def test_time_series_nested_cv_invalid_group_col():
    """Test error handling for invalid group column."""
    data = create_panel_data()

    with pytest.raises(ValueError, match="Group column 'invalid' not found"):
        time_series_nested_cv(
            data=data,
            group_col='invalid',
            date_column='date',
            initial='100 days',
            assess='30 days'
        )

    print("✅ time_series_nested_cv invalid group_col test passed")


def test_time_series_nested_cv_insufficient_data():
    """Test error handling when groups have insufficient data."""
    # Create data with one group too small
    data_frames = []

    # USA: enough data
    data_frames.append(pd.DataFrame({
        'country': ['USA'] * 200,
        'date': pd.date_range('2020-01-01', periods=200, freq='D'),
        'value': np.random.randn(200)
    }))

    # Germany: too little data
    data_frames.append(pd.DataFrame({
        'country': ['Germany'] * 50,
        'date': pd.date_range('2020-01-01', periods=50, freq='D'),
        'value': np.random.randn(50)
    }))

    data = pd.concat(data_frames, ignore_index=True)

    with pytest.raises(ValueError, match="Failed to create CV splits"):
        time_series_nested_cv(
            data=data,
            group_col='country',
            date_column='date',
            initial='100 days',
            assess='30 days'
        )

    print("✅ time_series_nested_cv insufficient data test passed")


def test_time_series_nested_cv_return_format():
    """Test that return format is correct (dict with group names as keys)."""
    data = create_panel_data()

    cv_by_group = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='100 days',
        assess='30 days'
    )

    # Should be a dictionary
    assert isinstance(cv_by_group, dict)

    # Keys should be group names
    assert all(isinstance(k, str) for k in cv_by_group.keys())

    # Values should be TimeSeriesCV objects
    assert all(isinstance(v, TimeSeriesCV) for v in cv_by_group.values())

    print("✅ time_series_nested_cv return format test passed")


def test_time_series_nested_cv_slice_limit():
    """Test that slice_limit parameter limits number of folds per group."""
    data = create_panel_data()

    # Without slice_limit - should create many folds per group
    cv_all = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='80 days',
        assess='20 days',
        skip='10 days'
    )

    # With slice_limit=3 - should only create 3 folds per group
    cv_limited = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='80 days',
        assess='20 days',
        skip='10 days',
        slice_limit=3
    )

    # Each group should have only 3 folds
    for group in ['USA', 'Germany', 'Japan']:
        assert len(cv_limited[group]) == 3
        assert len(cv_limited[group]) < len(cv_all[group])

        # Verify the limited folds are the first 3 folds (not random)
        for i in range(3):
            train_all = cv_all[group][i].training()
            train_limited = cv_limited[group][i].training()
            test_all = cv_all[group][i].testing()
            test_limited = cv_limited[group][i].testing()

            # Should be identical to first 3 folds from full CV
            assert len(train_all) == len(train_limited)
            assert len(test_all) == len(test_limited)

    print("✅ time_series_nested_cv slice_limit test passed")


# ============================================================================
# Tests for time_series_global_cv()
# ============================================================================

def test_time_series_global_cv_basic():
    """Test basic functionality of time_series_global_cv."""
    data = create_panel_data()

    cv_by_group = time_series_global_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial=400,  # Row count
        assess=100
    )

    # Should have 3 groups
    assert len(cv_by_group) == 3
    assert set(cv_by_group.keys()) == {'USA', 'Germany', 'Japan'}

    # Each group should have TimeSeriesCV object
    for group, cv in cv_by_group.items():
        assert isinstance(cv, TimeSeriesCV)
        assert len(cv) > 0  # Should have at least one fold

    # All groups should share the SAME CV object (not different objects)
    cv_objects = list(cv_by_group.values())
    assert cv_objects[0] is cv_objects[1]  # Same object reference
    assert cv_objects[1] is cv_objects[2]  # Same object reference

    print("✅ time_series_global_cv basic test passed")


def test_time_series_global_cv_same_splits():
    """Test that all groups get the same CV splits."""
    data = create_panel_data()

    cv_by_group = time_series_global_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial=400,  # Row count
        assess=100
    )

    # Get CV objects for each group
    cv_usa = cv_by_group['USA']
    cv_germany = cv_by_group['Germany']
    cv_japan = cv_by_group['Japan']

    # All should have same number of folds
    assert len(cv_usa) == len(cv_germany) == len(cv_japan)

    # All should be the exact same object
    assert cv_usa is cv_germany is cv_japan

    print("✅ time_series_global_cv same splits test passed")


def test_time_series_global_cv_vs_nested():
    """Test difference between global and nested CV."""
    data = create_panel_data()

    # Global CV (same splits for all groups)
    cv_global = time_series_global_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial=400,  # Row count
        assess=100
    )

    # Nested CV (different splits per group)
    cv_nested = time_series_nested_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial='100 days',
        assess='30 days'
    )

    # Global: all groups share same object
    assert cv_global['USA'] is cv_global['Germany']

    # Nested: each group has different object
    assert cv_nested['USA'] is not cv_nested['Germany']

    print("✅ time_series_global_cv vs nested test passed")


def test_time_series_global_cv_parameters():
    """Test that all time_series_cv parameters work with global CV."""
    data = create_panel_data()

    cv_by_group = time_series_global_cv(
        data=data,
        group_col='country',
        date_column='date',
        initial=300,  # Row count
        assess=50,
        skip=25,
        cumulative=False,  # Rolling window
        lag=10
    )

    # Verify parameters were passed correctly
    cv = cv_by_group['USA']
    assert cv.cumulative == False
    assert len(cv) > 0

    print("✅ time_series_global_cv parameters test passed")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    print("\n" + "="*60)
    print("Testing time_series_nested_cv() and time_series_global_cv()")
    print("="*60 + "\n")

    print("NESTED CV TESTS:")
    print("-" * 60)
    test_time_series_nested_cv_basic()
    test_time_series_nested_cv_parameters()
    test_time_series_nested_cv_fold_count()
    test_time_series_nested_cv_invalid_group_col()
    test_time_series_nested_cv_insufficient_data()
    test_time_series_nested_cv_return_format()
    test_time_series_nested_cv_slice_limit()

    print("\nGLOBAL CV TESTS:")
    print("-" * 60)
    test_time_series_global_cv_basic()
    test_time_series_global_cv_same_splits()
    test_time_series_global_cv_vs_nested()
    test_time_series_global_cv_parameters()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED (11 total)")
    print("="*60)
