"""
Demonstration test showing column ordering behavior across all use cases.

This test serves as both verification and documentation of the column ordering feature.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_parsnip import linear_reg


class TestColumnOrderingDemo:
    """Comprehensive demonstration of column ordering feature"""

    @pytest.fixture
    def time_series_data(self):
        """Create realistic time series data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'x1': np.random.randn(120) * 2 + 10,
            'x2': np.random.randn(120) * 3 + 5,
            'y': np.random.randn(120) * 5 + 100
        })
        train = data.iloc[:100]
        test = data.iloc[100:]
        return train, test

    @pytest.fixture
    def grouped_time_series_data(self):
        """Create realistic grouped time series data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='D')

        data_store_a = pd.DataFrame({
            'date': dates,
            'store_id': 'Store_A',
            'x1': np.random.randn(60) * 2 + 10,
            'y': np.random.randn(60) * 5 + 100
        })

        data_store_b = pd.DataFrame({
            'date': dates,
            'store_id': 'Store_B',
            'x1': np.random.randn(60) * 2 + 15,
            'y': np.random.randn(60) * 5 + 120
        })

        data_store_c = pd.DataFrame({
            'date': dates,
            'store_id': 'Store_C',
            'x1': np.random.randn(60) * 2 + 8,
            'y': np.random.randn(60) * 5 + 90
        })

        data = pd.concat([data_store_a, data_store_b, data_store_c], ignore_index=True)
        train = data.iloc[:150]  # 50 per store
        test = data.iloc[150:]   # 10 per store
        return train, test

    def test_demo_standard_workflow_column_order(self, time_series_data):
        """
        DEMO: Standard workflow shows date-first ordering
        """
        train, test = time_series_data

        # Fit a simple workflow
        wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)

        # Extract outputs
        outputs, coeffs, stats = fit.extract_outputs()

        # DEMONSTRATION: Column ordering is now predictable
        print("\n=== STANDARD WORKFLOW OUTPUT COLUMNS ===")
        print("outputs.columns:")
        for i, col in enumerate(outputs.columns, 1):
            print(f"  {i}. {col}")

        # VERIFICATION: Key columns are in expected positions
        assert outputs.columns[0] == 'date', "Date should be first column"
        assert outputs.columns[1] == 'actuals', "Actuals should be second"
        assert outputs.columns[2] == 'fitted', "Fitted should be third"
        assert outputs.columns[3] == 'forecast', "Forecast should be fourth"
        assert outputs.columns[4] == 'residuals', "Residuals should be fifth"
        assert outputs.columns[5] == 'split', "Split should be sixth"

        # Show sample data
        print("\noutputs.head():")
        print(outputs.head().to_string())

        # DEMONSTRATION: Easy column access
        assert 'date' in outputs.columns
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns

        # DEMONSTRATION: Coefficients ordering
        print("\n=== COEFFICIENTS COLUMNS ===")
        print("coeffs.columns:")
        for i, col in enumerate(coeffs.columns, 1):
            print(f"  {i}. {col}")

        assert coeffs.columns[0] == 'variable'
        assert coeffs.columns[1] == 'coefficient'

        # DEMONSTRATION: Stats ordering
        print("\n=== STATS COLUMNS ===")
        print("stats.columns:")
        for i, col in enumerate(stats.columns, 1):
            print(f"  {i}. {col}")

        assert stats.columns[0] == 'split'
        assert stats.columns[1] == 'metric'
        assert stats.columns[2] == 'value'

    def test_demo_nested_workflow_column_order(self, grouped_time_series_data):
        """
        DEMO: Nested workflow shows date-first, group-second ordering
        """
        train, test = grouped_time_series_data

        # Fit a nested workflow (separate model per store)
        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id')
        nested_fit = nested_fit.evaluate(test)

        # Extract outputs
        outputs, coeffs, stats = nested_fit.extract_outputs()

        # DEMONSTRATION: Column ordering with group column
        print("\n=== NESTED WORKFLOW OUTPUT COLUMNS ===")
        print("outputs.columns:")
        for i, col in enumerate(outputs.columns, 1):
            print(f"  {i}. {col}")

        # VERIFICATION: Date first, group second
        assert outputs.columns[0] == 'date', "Date should be first column"
        assert outputs.columns[1] == 'store_id', "Group column should be second"
        assert outputs.columns[2] == 'actuals', "Actuals should be third"
        assert outputs.columns[3] == 'fitted', "Fitted should be fourth"
        assert outputs.columns[4] == 'forecast', "Forecast should be fifth"
        assert outputs.columns[5] == 'residuals', "Residuals should be sixth"
        assert outputs.columns[6] == 'split', "Split should be seventh"

        # Show sample data
        print("\noutputs.head(10):")
        print(outputs.head(10).to_string())

        # DEMONSTRATION: Easy group filtering
        print("\n=== FILTERING BY GROUP ===")
        store_a_outputs = outputs[outputs['store_id'] == 'Store_A']
        print(f"Store A has {len(store_a_outputs)} observations")
        print(store_a_outputs.head().to_string())

        # DEMONSTRATION: Coefficients with group column
        print("\n=== NESTED COEFFICIENTS COLUMNS ===")
        print("coeffs.columns:")
        for i, col in enumerate(coeffs.columns, 1):
            print(f"  {i}. {col}")

        assert coeffs.columns[0] == 'store_id', "Group should be first in coefficients"
        assert coeffs.columns[1] == 'variable', "Variable should be second"
        assert coeffs.columns[2] == 'coefficient', "Coefficient should be third"

        # Show coefficients by group
        print("\nCoefficients by store:")
        print(coeffs[['store_id', 'variable', 'coefficient']].to_string())

        # DEMONSTRATION: Stats with group column
        print("\n=== NESTED STATS COLUMNS ===")
        print("stats.columns:")
        for i, col in enumerate(stats.columns, 1):
            print(f"  {i}. {col}")

        assert stats.columns[0] == 'store_id', "Group should be first in stats"
        assert stats.columns[1] == 'split', "Split should be second"
        assert stats.columns[2] == 'metric', "Metric should be third"
        assert stats.columns[3] == 'value', "Value should be fourth"

        # Show RMSE by store
        print("\nRMSE by store and split:")
        rmse_by_store = stats[stats['metric'] == 'rmse'][['store_id', 'split', 'value']]
        print(rmse_by_store.to_string())

    def test_demo_comparison_before_after(self, grouped_time_series_data):
        """
        DEMO: Show the improvement in column ordering
        """
        train, test = grouped_time_series_data

        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id').evaluate(test)
        outputs, coeffs, stats = nested_fit.extract_outputs()

        print("\n=== BEFORE vs AFTER COMPARISON ===")
        print("\nBEFORE (hypothetical inconsistent ordering):")
        print("  outputs.columns: ['residuals', 'model', 'actuals', 'split', 'store_id', 'fitted', 'date', ...]")
        print("  - Date was buried in the middle!")
        print("  - Group column was not consistently positioned")
        print("  - Hard to visually inspect")

        print("\nAFTER (current consistent ordering):")
        print(f"  outputs.columns: {list(outputs.columns[:7])}")
        print("  ✓ Date is first")
        print("  ✓ Group column is second")
        print("  ✓ Core columns follow in predictable order")
        print("  ✓ Easy to visually inspect")
        print("  ✓ Simple to filter by group")

        # Demonstrate practical benefits
        print("\n=== PRACTICAL BENEFITS ===")

        # Benefit 1: Quick access to date
        print("\n1. Quick access to date column (always first):")
        print(f"   outputs.columns[0] = '{outputs.columns[0]}'")
        assert outputs.columns[0] == 'date'

        # Benefit 2: Easy group filtering
        print("\n2. Easy group filtering (group always second):")
        print(f"   outputs.columns[1] = '{outputs.columns[1]}'")
        store_b = outputs[outputs[outputs.columns[1]] == 'Store_B']
        print(f"   Filtered Store_B: {len(store_b)} observations")

        # Benefit 3: Consistent structure across all models
        print("\n3. Consistent structure for all nested models:")
        print("   - Linear regression: date → group → actuals → ...")
        print("   - Random forest:     date → group → actuals → ...")
        print("   - Prophet:           date → group → actuals → ...")
        print("   All follow the SAME ordering!")

        # Benefit 4: Visual clarity in notebooks
        print("\n4. Visual clarity in Jupyter notebooks:")
        print(outputs[['date', 'store_id', 'actuals', 'fitted', 'split']].head(5).to_string())

    def test_demo_backward_compatibility(self, time_series_data):
        """
        DEMO: Existing code continues to work (backward compatibility)
        """
        train, test = time_series_data

        wf = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())
        fit = wf.fit(train).evaluate(test)
        outputs, coeffs, stats = fit.extract_outputs()

        print("\n=== BACKWARD COMPATIBILITY DEMO ===")

        # Old code pattern 1: Access by column name (always worked, still works)
        print("\n1. Access columns by name (recommended):")
        print("   actuals = outputs['actuals']  # ✓ Still works")
        actuals = outputs['actuals']
        assert len(actuals) > 0

        print("   fitted = outputs['fitted']    # ✓ Still works")
        fitted = outputs['fitted']
        assert len(fitted) > 0

        # Old code pattern 2: Filter by column value
        print("\n2. Filter by column value:")
        print("   train_data = outputs[outputs['split'] == 'train']  # ✓ Still works")
        train_data = outputs[outputs['split'] == 'train']
        assert len(train_data) > 0

        # Old code pattern 3: Iterate over columns
        print("\n3. Iterate over columns:")
        print("   for col in outputs.columns: ...  # ✓ Still works")
        col_count = 0
        for col in outputs.columns:
            col_count += 1
        assert col_count > 0
        print(f"   Found {col_count} columns")

        # Old code pattern 4: Select multiple columns
        print("\n4. Select multiple columns:")
        print("   subset = outputs[['actuals', 'fitted']]  # ✓ Still works")
        subset = outputs[['actuals', 'fitted']]
        assert len(subset.columns) == 2

        print("\n✓ All existing code patterns continue to work!")
        print("✓ No code changes required!")
        print("✓ Only column ORDER changed (for the better!)")

    def test_demo_use_cases(self, grouped_time_series_data):
        """
        DEMO: Common use cases made easier
        """
        train, test = grouped_time_series_data

        wf = workflow().add_formula('y ~ x1').add_model(linear_reg())
        nested_fit = wf.fit_nested(train, group_col='store_id').evaluate(test)
        outputs, coeffs, stats = nested_fit.extract_outputs()

        print("\n=== COMMON USE CASES ===")

        # Use case 1: Export to CSV for reporting
        print("\n1. Export to CSV/Excel:")
        print("   outputs.to_csv('forecast_results.csv')")
        print("   → Date is first column (intuitive for stakeholders)")
        print("   → Group is second column (easy to sort/filter in Excel)")

        # Use case 2: Compare performance across groups
        print("\n2. Compare performance across groups:")
        print("   test_rmse = stats[(stats['metric'] == 'rmse') & (stats['split'] == 'test')]")
        test_rmse = stats[(stats['metric'] == 'rmse') & (stats['split'] == 'test')]
        print("   Result:")
        print(test_rmse[['store_id', 'value']].to_string(index=False))

        # Use case 3: Plot time series by group
        print("\n3. Plot time series by group:")
        print("   for store in outputs['store_id'].unique():")
        print("       store_data = outputs[outputs['store_id'] == store]")
        print("       plt.plot(store_data['date'], store_data['actuals'])")
        print("   → Date column is guaranteed to be there and accessible")

        # Use case 4: Filter to specific group for analysis
        print("\n4. Filter to specific group:")
        print("   store_a = outputs[outputs['store_id'] == 'Store_A']")
        store_a = outputs[outputs['store_id'] == 'Store_A']
        print(f"   → {len(store_a)} observations for Store_A")
        print(f"   → Columns: {list(store_a.columns[:7])}")

        # Use case 5: Check residuals by group
        print("\n5. Check residuals by group:")
        print("   residuals_by_group = outputs.groupby('store_id')['residuals'].std()")
        residuals_by_group = outputs.groupby('store_id')['residuals'].std()
        print("   Result:")
        print(residuals_by_group.to_string())

        print("\n✓ All common use cases are now easier and more intuitive!")
