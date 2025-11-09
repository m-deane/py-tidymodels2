#!/usr/bin/env python3
"""
Test dot notation expansion for standard models (sklearn path).

Verifies that "target ~ ." excludes datetime columns to prevent
patsy categorical errors when test data has new dates.
"""

import pandas as pd
import numpy as np
from py_parsnip import linear_reg

def test_linear_reg_dot_notation_excludes_date():
    """Test that dot notation excludes datetime columns in standard models"""
    print("=" * 70)
    print("TEST: linear_reg with 'target ~ .' excludes date column")
    print("=" * 70)

    # Create training data with date column
    np.random.seed(42)
    train_dates = pd.date_range('2020-04-01', periods=40, freq='MS')  # Apr 2020 - Jul 2023
    train_data = pd.DataFrame({
        'date': train_dates,
        'x1': np.random.randn(40),
        'x2': np.random.randn(40),
        'x3': np.random.randn(40),
        'target': np.random.randn(40).cumsum() + 100
    })

    # Create test data with NEW dates (not in training)
    test_dates = pd.date_range('2023-10-01', periods=10, freq='MS')  # Oct 2023 onwards
    test_data = pd.DataFrame({
        'date': test_dates,
        'x1': np.random.randn(10),
        'x2': np.random.randn(10),
        'x3': np.random.randn(10),
        'target': np.random.randn(10).cumsum() + 100
    })

    try:
        # Fit with dot notation (should exclude date automatically)
        spec = linear_reg()
        fit = spec.fit(train_data, "target ~ .")

        print(f"✅ Training successful with dot notation")
        print(f"   Blueprint formula: {fit.blueprint.formula}")

        # Verify date was excluded from formula
        formula_lower = fit.blueprint.formula.lower()
        if 'date' in formula_lower:
            print(f"❌ FAILED: Date column was included in formula!")
            print(f"   Formula: {fit.blueprint.formula}")
            return False

        # The critical test: evaluate on test data with NEW dates
        # This previously raised: PatsyError: observation with value Timestamp('2023-10-01')
        # does not match any of the expected levels
        fit = fit.evaluate(test_data)

        print(f"✅ Evaluation successful on test data with new dates")
        print(f"   Test date range: {test_data['date'].min()} to {test_data['date'].max()}")
        print(f"   No patsy categorical error!")

        # Make predictions (should also work)
        predictions = fit.predict(test_data)
        print(f"   Predictions shape: {predictions.shape}")

        print("\n✅ PASSED: Dot notation excludes date, no categorical errors\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_linear_reg_explicit_formula_still_works():
    """Verify backward compatibility - explicit variable listing"""
    print("=" * 70)
    print("TEST: Backward compatibility (explicit variables, no date)")
    print("=" * 70)

    np.random.seed(42)
    train_dates = pd.date_range('2020-04-01', periods=40, freq='MS')
    train_data = pd.DataFrame({
        'date': train_dates,
        'x1': np.random.randn(40),
        'x2': np.random.randn(40),
        'target': np.random.randn(40).cumsum() + 100
    })

    test_dates = pd.date_range('2023-10-01', periods=10, freq='MS')
    test_data = pd.DataFrame({
        'date': test_dates,
        'x1': np.random.randn(10),
        'x2': np.random.randn(10),
        'target': np.random.randn(10).cumsum() + 100
    })

    try:
        # Explicit formula without date
        spec = linear_reg()
        fit = spec.fit(train_data, "target ~ x1 + x2")

        print(f"✅ Training successful with explicit formula")

        # Should work fine
        fit = fit.evaluate(test_data)
        predictions = fit.predict(test_data)

        print(f"✅ Evaluation successful")
        print(f"   Predictions shape: {predictions.shape}")

        print("\n✅ PASSED: Backward compatibility maintained\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        return False


def test_statsmodels_linear_reg():
    """Test with statsmodels engine (user's original scenario)"""
    print("=" * 70)
    print("TEST: linear_reg with statsmodels engine and dot notation")
    print("=" * 70)

    np.random.seed(42)
    train_dates = pd.date_range('2020-04-01', periods=40, freq='MS')
    train_data = pd.DataFrame({
        'date': train_dates,
        'x1': np.random.randn(40),
        'x2': np.random.randn(40),
        'x3': np.random.randn(40),
        'target': np.random.randn(40).cumsum() + 100
    })

    test_dates = pd.date_range('2023-10-01', periods=10, freq='MS')
    test_data = pd.DataFrame({
        'date': test_dates,
        'x1': np.random.randn(10),
        'x2': np.random.randn(10),
        'x3': np.random.randn(10),
        'target': np.random.randn(10).cumsum() + 100
    })

    try:
        # Use statsmodels engine like in the forecasting notebook
        spec = linear_reg().set_engine("statsmodels")
        fit = spec.fit(train_data, "target ~ .")

        print(f"✅ Training successful with statsmodels engine")
        print(f"   Blueprint formula: {fit.blueprint.formula}")

        # Evaluate on test data with new dates
        fit = fit.evaluate(test_data)

        print(f"✅ Evaluation successful on test data")
        print(f"   No patsy categorical error!")

        print("\n✅ PASSED: Statsmodels engine with dot notation works\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all standard model dot notation tests"""
    print("\n" + "=" * 70)
    print("STANDARD MODEL DOT NOTATION TESTS")
    print("Testing fix for: date column in formula causes patsy categorical error")
    print("=" * 70 + "\n")

    results = []
    results.append(("linear_reg dot notation", test_linear_reg_dot_notation_excludes_date()))
    results.append(("Backward compatibility", test_linear_reg_explicit_formula_still_works()))
    results.append(("statsmodels engine", test_statsmodels_linear_reg()))

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Standard model dot notation fix verified!")
        print("   The forecasting.ipynb issue is now resolved for linear_reg models.")
    else:
        print(f"\n❌ {total - passed} test(s) failed - fix needs review")

    print("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
