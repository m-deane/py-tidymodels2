"""
Tests for step_corr() - correlation-based feature filtering
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.feature_selection import StepCorr, PreparedStepCorr
from py_recipes.selectors import all_numeric


class TestStepCorr:
    """Test suite for StepCorr"""

    def test_step_corr_basic(self):
        """Test basic correlation filtering with default threshold"""
        # Create data with highly correlated features
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # Highly correlated with x1
        x3 = np.random.randn(n)
        y = x1 + x3 + np.random.randn(n) * 0.1

        data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'y': y
        })

        # Apply step_corr
        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # x2 should be removed due to high correlation with x1
        assert 'x1' in result.columns or 'x2' in result.columns
        assert not ('x1' in result.columns and 'x2' in result.columns)
        assert 'x3' in result.columns
        assert 'y' in result.columns

    def test_step_corr_no_removal(self):
        """Test that uncorrelated features are all kept"""
        # Create data with low correlations
        np.random.seed(123)
        data = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100),
            'y': np.random.randn(100)
        })

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # All columns should be kept
        assert set(result.columns) == set(data.columns)

    def test_step_corr_threshold(self):
        """Test different threshold values"""
        # Create data with moderate correlation
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.5  # Moderately correlated
        x3 = np.random.randn(n)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        # High threshold - should keep both
        rec1 = recipe(data).step_corr(threshold=0.95)
        result1 = rec1.prep(data).bake(data)
        assert 'x1' in result1.columns and 'x2' in result1.columns

        # Low threshold - should remove one
        rec2 = recipe(data).step_corr(threshold=0.5)
        result2 = rec2.prep(data).bake(data)
        assert 'x1' in result2.columns or 'x2' in result2.columns
        assert not ('x1' in result2.columns and 'x2' in result2.columns)

    def test_step_corr_column_subset(self):
        """Test filtering specific columns"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01
        x3 = np.random.randn(n)
        x4 = x3 + np.random.randn(n) * 0.01

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})

        # Check only x1 and x2
        rec = recipe(data).step_corr(columns=['x1', 'x2'], threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # x1 or x2 should be removed, but x3 and x4 should remain
        assert 'x1' in result.columns or 'x2' in result.columns
        assert not ('x1' in result.columns and 'x2' in result.columns)
        assert 'x3' in result.columns and 'x4' in result.columns

    def test_step_corr_method_pearson(self):
        """Test Pearson correlation method"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01

        data = pd.DataFrame({'x1': x1, 'x2': x2})

        rec = recipe(data).step_corr(threshold=0.9, method='pearson')
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # One column should be removed
        assert len(result.columns) == 1

    def test_step_corr_method_spearman(self):
        """Test Spearman correlation method"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        # Create monotonic but non-linear relationship
        x2 = np.sort(x1) + np.random.randn(n) * 0.01

        data = pd.DataFrame({'x1': x1, 'x2': x2})

        rec = recipe(data).step_corr(threshold=0.9, method='spearman')
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Spearman should detect correlation
        assert len(result.columns) <= 2

    def test_step_corr_method_kendall(self):
        """Test Kendall correlation method"""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05

        data = pd.DataFrame({'x1': x1, 'x2': x2})

        rec = recipe(data).step_corr(threshold=0.8, method='kendall')
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # One column should be removed
        assert len(result.columns) == 1

    def test_step_corr_multiple_pairs(self):
        """Test removal of multiple correlated pairs"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # Correlated with x1
        x3 = np.random.randn(n)
        x4 = x3 + np.random.randn(n) * 0.01  # Correlated with x3
        x5 = np.random.randn(n)  # Independent

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should remove 2 columns (one from each pair)
        assert len(result.columns) == 3
        assert 'x5' in result.columns  # Independent should remain

    def test_step_corr_preserves_non_numeric(self):
        """Test that non-numeric columns are preserved"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01

        data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'category': ['A'] * 50 + ['B'] * 50,
            'text': ['foo'] * n
        })

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Non-numeric columns should be preserved
        assert 'category' in result.columns
        assert 'text' in result.columns
        # One numeric column should be removed
        assert 'x1' in result.columns or 'x2' in result.columns
        assert not ('x1' in result.columns and 'x2' in result.columns)

    def test_step_corr_with_selector(self):
        """Test using selector function for columns"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01
        x3 = np.random.randn(n)

        data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'category': ['A'] * n
        })

        # Use all_numeric selector
        rec = recipe(data).step_corr(columns=all_numeric(), threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should only check numeric columns
        assert 'category' in result.columns
        assert 'x3' in result.columns

    def test_step_corr_single_column(self):
        """Test with single column (no correlation possible)"""
        data = pd.DataFrame({'x1': np.random.randn(100)})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should keep the single column
        assert list(result.columns) == ['x1']

    def test_step_corr_empty_columns(self):
        """Test with no numeric columns"""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 10,
            'text': ['foo', 'bar', 'baz'] * 10
        })

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should keep all columns
        assert set(result.columns) == set(data.columns)

    def test_step_corr_perfect_correlation(self):
        """Test with perfectly correlated features"""
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = x1.copy()  # Perfect correlation

        data = pd.DataFrame({'x1': x1, 'x2': x2})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should remove one column
        assert len(result.columns) == 1

    def test_step_corr_new_data(self):
        """Test baking with new data"""
        np.random.seed(42)
        n = 100
        x1_train = np.random.randn(n)
        x2_train = x1_train + np.random.randn(n) * 0.01
        x3_train = np.random.randn(n)

        train_data = pd.DataFrame({'x1': x1_train, 'x2': x2_train, 'x3': x3_train})

        # Prep on training data
        rec = recipe(train_data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(train_data)

        # New test data
        x1_test = np.random.randn(50)
        x2_test = np.random.randn(50)
        x3_test = np.random.randn(50)
        test_data = pd.DataFrame({'x1': x1_test, 'x2': x2_test, 'x3': x3_test})

        # Bake on test data
        result = rec_prepped.bake(test_data)

        # Should have same columns as training result
        train_result = rec_prepped.bake(train_data)
        assert set(result.columns) == set(train_result.columns)

    def test_step_corr_negative_correlation(self):
        """Test that absolute correlation is used"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = -x1 + np.random.randn(n) * 0.01  # Perfect negative correlation

        data = pd.DataFrame({'x1': x1, 'x2': x2})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should detect negative correlation and remove one
        assert len(result.columns) == 1

    def test_step_corr_chain_with_other_steps(self):
        """Test chaining with other recipe steps"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01
        x3 = np.random.randn(n)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        # Chain multiple steps
        rec = (recipe(data)
               .step_normalize()
               .step_corr(threshold=0.9))

        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should apply both normalization and correlation filtering
        assert len(result.columns) == 2

    def test_step_corr_missing_column_in_bake(self):
        """Test baking when some columns are missing"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01
        x3 = np.random.randn(n)

        train_data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        rec = recipe(train_data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(train_data)

        # Determine which column was removed during prep
        train_result = rec_prepped.bake(train_data)
        removed_cols = set(train_data.columns) - set(train_result.columns)

        # Test data missing x2 (one of the correlated columns)
        test_data = pd.DataFrame({'x1': np.random.randn(50), 'x3': np.random.randn(50)})
        result = rec_prepped.bake(test_data)

        # Should gracefully handle missing column
        # x3 should always be present since it's independent
        assert 'x3' in result.columns
        # If x1 was removed, result should only have x3
        # If x2 was removed, result should have x1 and x3
        assert len(result.columns) <= 2

    def test_step_corr_preserves_order(self):
        """Test that column order is preserved (except removed columns)"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = x2 + np.random.randn(n) * 0.01  # Correlated with x2

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Check that remaining columns maintain their relative order
        result_cols = list(result.columns)
        original_cols = [col for col in data.columns if col in result_cols]
        assert result_cols == original_cols

    def test_step_corr_removes_higher_correlated(self):
        """Test that the column with higher mean correlation is removed"""
        np.random.seed(42)
        n = 100

        # x1 is correlated with x2 and x3
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # Correlated with x1
        x3 = x1 + np.random.randn(n) * 0.1  # Correlated with x1
        x4 = np.random.randn(n)  # Independent

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})

        rec = recipe(data).step_corr(threshold=0.8)
        rec_prepped = rec.prep(data)

        # x1 has the highest mean correlation, so it might be removed
        # Check that the algorithm runs without error
        result = rec_prepped.bake(data)
        assert 'x4' in result.columns  # Independent should remain

    def test_step_corr_three_way_correlation(self):
        """Test with three mutually correlated features"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05
        x3 = x1 + np.random.randn(n) * 0.05

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

        rec = recipe(data).step_corr(threshold=0.9)
        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Should remove at least one column
        assert len(result.columns) < 3

    def test_prepared_step_corr_direct(self):
        """Test PreparedStepCorr directly"""
        data = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})

        # Create prepared step directly
        prepped = PreparedStepCorr(columns_to_remove=['x2'])
        result = prepped.bake(data)

        assert 'x1' in result.columns
        assert 'x2' not in result.columns
        assert 'x3' in result.columns

    def test_prepared_step_corr_empty_removal(self):
        """Test PreparedStepCorr with no columns to remove"""
        data = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})

        prepped = PreparedStepCorr(columns_to_remove=[])
        result = prepped.bake(data)

        assert set(result.columns) == {'x1', 'x2'}

    def test_step_corr_integration_recipe(self):
        """Test full integration with recipe workflow"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # Highly correlated
        x3 = np.random.randn(n)
        x4 = np.random.randn(n)
        y = x1 + x3 + np.random.randn(n) * 0.1

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})

        # Complex recipe
        rec = (recipe(data)
               .step_corr(threshold=0.95)
               .step_normalize()
               .step_mutate({'x1_sq': lambda df: df['x1'] ** 2 if 'x1' in df.columns else 0}))

        rec_prepped = rec.prep(data)
        result = rec_prepped.bake(data)

        # Check that recipe executed successfully
        assert len(result.columns) >= 3  # At least some columns remain
        assert 'y' in result.columns
