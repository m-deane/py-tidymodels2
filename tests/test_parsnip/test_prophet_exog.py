"""
Quick verification test for Prophet with exogenous regressors
"""

import pandas as pd
import numpy as np
from py_parsnip import prophet_reg

# Create test data with exogenous variables
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
trend = np.linspace(100, 200, 100)
seasonality = 10 * np.sin(np.arange(100) * 2 * np.pi / 7)  # Weekly pattern

# Create exogenous variables
exog1 = np.random.randn(100) * 5  # Random regressor 1
exog2 = np.linspace(0, 1, 100) * 20  # Trend regressor 2

# Target variable influenced by trend, seasonality, and exog variables
target = trend + seasonality + 2 * exog1 + 1.5 * exog2 + np.random.randn(100) * 3

train_data = pd.DataFrame({
    "date": dates,
    "sales": target,
    "promo": exog1,
    "temperature": exog2
})

# Test 1: Fit with exogenous regressors
print("Test 1: Fitting Prophet with exogenous regressors...")
spec = prophet_reg()
fit = spec.fit(train_data, "sales ~ promo + temperature + date")

assert fit is not None
assert "model" in fit.fit_data
assert fit.fit_data["exog_vars"] == ["promo", "temperature"]
print("✓ Fit successful with exog vars:", fit.fit_data["exog_vars"])

# Test 2: Predict with exogenous regressors
print("\nTest 2: Predicting with exogenous regressors...")
future_dates = pd.date_range(start="2022-04-11", periods=10, freq="D")
future_exog1 = np.random.randn(10) * 5
future_exog2 = np.linspace(1, 1.1, 10) * 20

future_data = pd.DataFrame({
    "date": future_dates,
    "promo": future_exog1,
    "temperature": future_exog2
})

predictions = fit.predict(future_data)
assert ".pred" in predictions.columns
assert len(predictions) == 10
print("✓ Predictions successful, shape:", predictions.shape)

# Test 3: Prediction intervals
print("\nTest 3: Prediction intervals with exogenous regressors...")
pred_intervals = fit.predict(future_data, type="conf_int")
assert ".pred" in pred_intervals.columns
assert ".pred_lower" in pred_intervals.columns
assert ".pred_upper" in pred_intervals.columns
print("✓ Prediction intervals successful")
print("Columns:", pred_intervals.columns.tolist())

# Test 4: Fit without exogenous regressors (backward compatibility)
print("\nTest 4: Backward compatibility (no exog vars)...")
spec_simple = prophet_reg()
fit_simple = spec_simple.fit(train_data, "sales ~ date")
assert fit_simple.fit_data["exog_vars"] == []
print("✓ Backward compatibility maintained")

# Test 5: Extract outputs
print("\nTest 5: Extracting outputs...")
outputs, coefficients, stats = fit.extract_outputs()
assert len(outputs) > 0
assert len(coefficients) > 0
assert len(stats) > 0
print("✓ Extract outputs successful")
print(f"  Outputs shape: {outputs.shape}")
print(f"  Coefficients shape: {coefficients.shape}")
print(f"  Stats shape: {stats.shape}")

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("Prophet engine successfully supports exogenous regressors!")
print("="*60)
