"""
Quick Start Guide for Hybrid Time Series Models

This script provides minimal examples to get started with arima_boost and prophet_boost.
"""

import pandas as pd
import numpy as np
from py_parsnip import arima_boost, prophet_boost

# ============================================================================
# Create sample data
# ============================================================================
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=200, freq="D")
trend = np.linspace(100, 200, 200)
seasonality = 10 * np.sin(np.arange(200) * 2 * np.pi / 30)
non_linear = 5 * np.sin(np.arange(200) * 0.05) ** 2
noise = np.random.normal(0, 3, 200)
sales = trend + seasonality + non_linear + noise

data = pd.DataFrame({"date": dates, "sales": sales})
train = data.iloc[:150]
test = data.iloc[150:]

# ============================================================================
# Example 1: ARIMA + XGBoost (arima_boost)
# ============================================================================
print("Example 1: ARIMA + XGBoost")
print("-" * 50)

# Create model specification
model1 = arima_boost(
    # ARIMA parameters
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    # XGBoost parameters
    trees=50,
    tree_depth=3,
    learn_rate=0.1,
)

# Fit model
fit1 = model1.fit(train, "sales ~ date")
print("Model fitted!")

# Predict
predictions1 = fit1.predict(test[["date"]], type="numeric")
rmse1 = np.sqrt(np.mean((test["sales"] - predictions1[".pred"]) ** 2))
print(f"Test RMSE: {rmse1:.2f}")

# Extract outputs
outputs1, coefficients1, stats1 = fit1.extract_outputs()
print(f"Training observations: {len(outputs1[outputs1['split'] == 'train'])}")
print(f"ARIMA parameters: {len(coefficients1[coefficients1['variable'].str.contains('arima')])}")
print(f"XGBoost parameters: {len(coefficients1[coefficients1['variable'].str.contains('xgb')])}")

# ============================================================================
# Example 2: Prophet + XGBoost (prophet_boost)
# ============================================================================
print("\nExample 2: Prophet + XGBoost")
print("-" * 50)

# Create model specification
model2 = prophet_boost(
    # Prophet parameters
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    # XGBoost parameters
    trees=50,
    tree_depth=3,
    learn_rate=0.1,
)

# Fit model
fit2 = model2.fit(train, "sales ~ date")
print("Model fitted!")

# Predict
predictions2 = fit2.predict(test[["date"]], type="numeric")
rmse2 = np.sqrt(np.mean((test["sales"] - predictions2[".pred"]) ** 2))
print(f"Test RMSE: {rmse2:.2f}")

# Extract outputs
outputs2, coefficients2, stats2 = fit2.extract_outputs()
print(f"Training observations: {len(outputs2[outputs2['split'] == 'train'])}")
print(f"Prophet parameters: {len(coefficients2[coefficients2['variable'].str.contains('prophet')])}")
print(f"XGBoost parameters: {len(coefficients2[coefficients2['variable'].str.contains('xgb')])}")

# ============================================================================
# Compare performance
# ============================================================================
print("\nComparison")
print("-" * 50)
print(f"ARIMA + XGBoost RMSE:  {rmse1:.2f}")
print(f"Prophet + XGBoost RMSE: {rmse2:.2f}")

# ============================================================================
# Key Takeaways
# ============================================================================
print("\nKey Takeaways")
print("-" * 50)
print("1. Both models combine classical forecasting with gradient boosting")
print("2. arima_boost: Best for data with autocorrelation + non-linear patterns")
print("3. prophet_boost: Best for data with strong seasonality + non-linear patterns")
print("4. Extract outputs to see individual components (base model + XGBoost)")
print("5. Tune both base model and XGBoost parameters for best performance")
