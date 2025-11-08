"""
Tests for step_rm and step_select
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe


def test_step_rm_single_column():
    """Test removing a single column"""
    data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })

    rec = recipe().step_rm("b")
    rec_fit = rec.prep(data)
    result = rec_fit.bake(data)

    assert 'a' in result.columns
    assert 'b' not in result.columns
    assert 'c' in result.columns
    assert len(result.columns) == 2


def test_step_rm_multiple_columns():
    """Test removing multiple columns"""
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'date': pd.date_range('2020-01-01', periods=3),
        'feature': [10, 20, 30],
        'target': [100, 200, 300]
    })

    rec = recipe().step_rm(["id", "date"])
    rec_fit = rec.prep(data)
    result = rec_fit.bake(data)

    assert 'id' not in result.columns
    assert 'date' not in result.columns
    assert 'feature' in result.columns
    assert 'target' in result.columns
    assert len(result.columns) == 2


def test_step_rm_column_not_found():
    """Test error when column doesn't exist"""
    data = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })

    rec = recipe().step_rm("c")

    with pytest.raises(ValueError, match="Columns not found"):
        rec.prep(data)


def test_step_select_keep_columns():
    """Test selecting specific columns to keep"""
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'date': pd.date_range('2020-01-01', periods=3),
        'feature': [10, 20, 30],
        'target': [100, 200, 300]
    })

    rec = recipe().step_select(["feature", "target"])
    rec_fit = rec.prep(data)
    result = rec_fit.bake(data)

    assert 'id' not in result.columns
    assert 'date' not in result.columns
    assert 'feature' in result.columns
    assert 'target' in result.columns
    assert len(result.columns) == 2


def test_step_rm_with_other_steps():
    """Test step_rm in combination with other steps"""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'feature1': [10, 20, 30, 40, 50],
        'feature2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'target': [100, 200, 300, 400, 500]
    })

    rec = (
        recipe()
        .step_rm("id")
        .step_normalize()
    )
    rec_fit = rec.prep(data)
    result = rec_fit.bake(data)

    # ID should be removed
    assert 'id' not in result.columns

    # Other columns should be normalized
    assert 'feature1' in result.columns
    assert 'feature2' in result.columns
    assert 'target' in result.columns

    # Check normalization worked (mean ~0, std ~1)
    assert abs(result['feature1'].mean()) < 0.01
    assert abs(result['feature1'].std(ddof=0) - 1.0) < 0.01  # Use population std


def test_step_rm_bake_on_new_data():
    """Test that step_rm works on new data"""
    train_data = pd.DataFrame({
        'id': [1, 2, 3],
        'feature': [10, 20, 30],
        'target': [100, 200, 300]
    })

    test_data = pd.DataFrame({
        'id': [4, 5],
        'feature': [40, 50],
        'target': [400, 500]
    })

    rec = recipe().step_rm("id")
    rec_fit = rec.prep(train_data)

    # Bake on new data
    result = rec_fit.bake(test_data)

    assert 'id' not in result.columns
    assert 'feature' in result.columns
    assert 'target' in result.columns
    assert len(result) == 2
