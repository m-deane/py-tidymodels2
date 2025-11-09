#!/usr/bin/env python3
"""
Verification test for dot notation fix in forecasting.ipynb

Tests that "target ~ ." formula now works with Prophet and other time series models.
"""

import pandas as pd
import numpy as np
from py_parsnip import prophet_reg, arima_reg, seasonal_reg

def test_prophet_dot_notation():
    """Test Prophet with dot notation (user's original issue)"""
    print("=" * 70)
    print("TEST 1: Prophet with 'target ~ .'")
    print("=" * 70)

    # Create test data matching forecasting notebook structure
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'x3': np.random.randn(100),
        'target': np.random.randn(100).cumsum() + 100
    })

    try:
        # This previously raised: ValueError: Exogenous variable '.' not found in data
        spec = prophet_reg()
        fit = spec.fit(data, "target ~ .")

        # Verify exogenous variables were expanded correctly
        exog_vars = fit.fit_data.get('exog_vars', [])
        print(f"✅ SUCCESS: Prophet fit with dot notation")
        print(f"   Exogenous variables detected: {exog_vars}")
        print(f"   Expected: ['x1', 'x2', 'x3']")

        # Verify predictions work
        test_data = data.iloc[-10:].copy()
        predictions = fit.predict(test_data)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction columns: {list(predictions.columns)}")

        assert set(exog_vars) == {'x1', 'x2', 'x3'}, f"Wrong exog vars: {exog_vars}"
        print("\n✅ PASSED: Prophet dot notation test\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        return False


def test_arima_dot_notation():
    """Test ARIMA with dot notation"""
    print("=" * 70)
    print("TEST 2: ARIMA with 'target ~ .'")
    print("=" * 70)

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'target': np.random.randn(100).cumsum() + 100
    })

    try:
        spec = arima_reg(
            non_seasonal_ar=1,
            non_seasonal_differences=1,
            non_seasonal_ma=1
        )
        fit = spec.fit(data, "target ~ .")

        exog_vars = fit.fit_data.get('exog_vars', [])
        print(f"✅ SUCCESS: ARIMA fit with dot notation")
        print(f"   Exogenous variables detected: {exog_vars}")
        print(f"   Expected: ['x1', 'x2']")

        # Verify predictions work
        test_data = data.iloc[-10:].copy()
        predictions = fit.predict(test_data)
        print(f"   Predictions shape: {predictions.shape}")

        assert set(exog_vars) == {'x1', 'x2'}, f"Wrong exog vars: {exog_vars}"
        print("\n✅ PASSED: ARIMA dot notation test\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        return False


def test_seasonal_reg_dot_notation():
    """Test Seasonal Regression (STL) with dot notation"""
    print("=" * 70)
    print("TEST 3: Seasonal Regression with 'target ~ .'")
    print("=" * 70)

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'target': np.random.randn(100).cumsum() + 100
    })

    try:
        spec = seasonal_reg(seasonal_period_1=7)
        fit = spec.fit(data, "target ~ .")

        exog_vars = fit.fit_data.get('exog_vars', [])
        print(f"✅ SUCCESS: Seasonal regression fit with dot notation")
        print(f"   Exogenous variables detected: {exog_vars}")

        # Verify predictions work
        test_data = data.iloc[-10:].copy()
        predictions = fit.predict(test_data)
        print(f"   Predictions shape: {predictions.shape}")

        print("\n✅ PASSED: Seasonal regression dot notation test\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        return False


def test_explicit_variables_still_work():
    """Verify backward compatibility - explicit variable listing"""
    print("=" * 70)
    print("TEST 4: Backward Compatibility (explicit variables)")
    print("=" * 70)

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'x3': np.random.randn(100),
        'target': np.random.randn(100).cumsum() + 100
    })

    try:
        # Old style should still work
        spec = prophet_reg()
        fit = spec.fit(data, "target ~ x1 + x2")

        exog_vars = fit.fit_data.get('exog_vars', [])
        print(f"✅ SUCCESS: Explicit variable listing still works")
        print(f"   Exogenous variables: {exog_vars}")
        print(f"   Expected: ['x1', 'x2']")

        assert set(exog_vars) == {'x1', 'x2'}, f"Wrong exog vars: {exog_vars}"
        print("\n✅ PASSED: Backward compatibility test\n")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {type(e).__name__}: {e}\n")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("DOT NOTATION FIX VERIFICATION TESTS")
    print("Testing fix for: ValueError: Exogenous variable '.' not found in data")
    print("=" * 70 + "\n")

    results = []
    results.append(("Prophet dot notation", test_prophet_dot_notation()))
    results.append(("ARIMA dot notation", test_arima_dot_notation()))
    results.append(("Seasonal regression dot notation", test_seasonal_reg_dot_notation()))
    results.append(("Backward compatibility", test_explicit_variables_still_work()))

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
        print("\n✅ ALL TESTS PASSED - Dot notation fix verified successfully!")
        print("   The forecasting.ipynb issue is now resolved.")
    else:
        print(f"\n❌ {total - passed} test(s) failed - fix needs review")

    print("=" * 70 + "\n")

    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
