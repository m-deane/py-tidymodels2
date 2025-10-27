"""
Demonstration of hybrid time series models: arima_boost and prophet_boost

This script shows how to use the hybrid models that combine classical
forecasting with gradient boosting to capture both linear and non-linear patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_parsnip import arima_boost, prophet_boost

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic time series data with both linear and non-linear patterns
print("=" * 70)
print("Creating synthetic time series data...")
print("=" * 70)

n = 500
dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

# Linear trend (will be captured by ARIMA/Prophet)
trend = np.linspace(100, 300, n)

# Seasonal pattern (will be captured by ARIMA/Prophet)
yearly_seasonality = 30 * np.sin(np.arange(n) * 2 * np.pi / 365)
weekly_seasonality = 10 * np.sin(np.arange(n) * 2 * np.pi / 7)

# Non-linear pattern (will be captured by XGBoost)
non_linear = 20 * np.sin(np.arange(n) * 0.03) ** 2

# Random noise
noise = np.random.normal(0, 5, n)

# Combine all components
sales = trend + yearly_seasonality + weekly_seasonality + non_linear + noise

data = pd.DataFrame({"date": dates, "sales": sales})

# Split into train and test
train_size = 400
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

print(f"\nData shape: {data.shape}")
print(f"Training data: {train_size} observations")
print(f"Test data: {len(test_data)} observations")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")

# ============================================================================
# Example 1: ARIMA + XGBoost Hybrid
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: ARIMA + XGBoost Hybrid (arima_boost)")
print("=" * 70)

# Create and fit ARIMA+XGBoost model
print("\nFitting ARIMA(1,1,1) + XGBoost model...")
spec_arima_boost = arima_boost(
    # ARIMA parameters
    non_seasonal_ar=1,
    non_seasonal_differences=1,
    non_seasonal_ma=1,
    # XGBoost parameters
    trees=100,
    tree_depth=5,
    learn_rate=0.1,
)

fit_arima_boost = spec_arima_boost.fit(train_data, "sales ~ date")
print("Model fitted successfully!")

# Make predictions
print("\nMaking predictions on test data...")
preds_arima_boost = fit_arima_boost.predict(test_data[["date"]], type="numeric")

# Calculate metrics
test_actuals = test_data["sales"].values
arima_boost_rmse = np.sqrt(
    np.mean((test_actuals - preds_arima_boost[".pred"].values) ** 2)
)
arima_boost_mae = np.mean(np.abs(test_actuals - preds_arima_boost[".pred"].values))

print(f"\nTest Set Performance:")
print(f"  RMSE: {arima_boost_rmse:.2f}")
print(f"  MAE:  {arima_boost_mae:.2f}")

# Extract comprehensive outputs
outputs_ab, coefficients_ab, stats_ab = fit_arima_boost.extract_outputs()

print(f"\nModel components:")
print(f"  ARIMA fitted values shape: {outputs_ab['arima_fitted'].shape}")
print(f"  XGBoost fitted values shape: {outputs_ab['xgb_fitted'].shape}")
print(f"  Combined fitted values shape: {outputs_ab['fitted'].shape}")

# Show some ARIMA parameters
arima_params = coefficients_ab[coefficients_ab["variable"].str.contains("arima_")]
print(f"\nARIMA parameters (sample):")
print(arima_params[["variable", "coefficient"]].head())

# Show XGBoost parameters
xgb_params = coefficients_ab[coefficients_ab["variable"].str.contains("xgb_")]
print(f"\nXGBoost parameters:")
print(xgb_params[["variable", "coefficient"]])

# ============================================================================
# Example 2: Prophet + XGBoost Hybrid
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Prophet + XGBoost Hybrid (prophet_boost)")
print("=" * 70)

# Create and fit Prophet+XGBoost model
print("\nFitting Prophet + XGBoost model...")
spec_prophet_boost = prophet_boost(
    # Prophet parameters
    growth="linear",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    seasonality_mode="additive",
    # XGBoost parameters
    trees=100,
    tree_depth=5,
    learn_rate=0.1,
)

fit_prophet_boost = spec_prophet_boost.fit(train_data, "sales ~ date")
print("Model fitted successfully!")

# Make predictions
print("\nMaking predictions on test data...")
preds_prophet_boost = fit_prophet_boost.predict(test_data[["date"]], type="numeric")

# Calculate metrics
prophet_boost_rmse = np.sqrt(
    np.mean((test_actuals - preds_prophet_boost[".pred"].values) ** 2)
)
prophet_boost_mae = np.mean(np.abs(test_actuals - preds_prophet_boost[".pred"].values))

print(f"\nTest Set Performance:")
print(f"  RMSE: {prophet_boost_rmse:.2f}")
print(f"  MAE:  {prophet_boost_mae:.2f}")

# Extract comprehensive outputs
outputs_pb, coefficients_pb, stats_pb = fit_prophet_boost.extract_outputs()

print(f"\nModel components:")
print(f"  Prophet fitted values shape: {outputs_pb['prophet_fitted'].shape}")
print(f"  XGBoost fitted values shape: {outputs_pb['xgb_fitted'].shape}")
print(f"  Combined fitted values shape: {outputs_pb['fitted'].shape}")

# Show Prophet parameters
prophet_params = coefficients_pb[coefficients_pb["variable"].str.contains("prophet_")]
print(f"\nProphet parameters:")
print(prophet_params[["variable", "coefficient"]])

# Show XGBoost parameters
xgb_params_pb = coefficients_pb[coefficients_pb["variable"].str.contains("xgb_")]
print(f"\nXGBoost parameters:")
print(xgb_params_pb[["variable", "coefficient"]])

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "=" * 70)
print("Model Comparison")
print("=" * 70)

print(f"\nTest Set Performance:")
print(f"  ARIMA + XGBoost:")
print(f"    RMSE: {arima_boost_rmse:.2f}")
print(f"    MAE:  {arima_boost_mae:.2f}")
print(f"\n  Prophet + XGBoost:")
print(f"    RMSE: {prophet_boost_rmse:.2f}")
print(f"    MAE:  {prophet_boost_mae:.2f}")

# ============================================================================
# Visualization (optional - only if matplotlib available)
# ============================================================================
try:
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Hybrid Time Series Models: ARIMA+XGBoost vs Prophet+XGBoost", fontsize=16)

    # Plot 1: ARIMA+XGBoost training fit
    ax1 = axes[0, 0]
    train_outputs = outputs_ab[outputs_ab["split"] == "train"]
    ax1.plot(train_outputs["date"], train_outputs["actuals"], label="Actuals", alpha=0.7)
    ax1.plot(
        train_outputs["date"],
        train_outputs["arima_fitted"],
        label="ARIMA Only",
        alpha=0.7,
        linestyle="--",
    )
    ax1.plot(train_outputs["date"], train_outputs["fitted"], label="ARIMA+XGBoost", alpha=0.9)
    ax1.set_title("ARIMA+XGBoost: Training Fit")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ARIMA+XGBoost test predictions
    ax2 = axes[0, 1]
    ax2.plot(test_data["date"], test_actuals, label="Actuals", alpha=0.7)
    ax2.plot(
        test_data["date"],
        preds_arima_boost[".pred"],
        label="ARIMA+XGBoost Predictions",
        alpha=0.9,
    )
    ax2.set_title(f"ARIMA+XGBoost: Test Predictions (RMSE={arima_boost_rmse:.2f})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prophet+XGBoost training fit
    ax3 = axes[1, 0]
    train_outputs_pb = outputs_pb[outputs_pb["split"] == "train"]
    ax3.plot(train_outputs_pb["date"], train_outputs_pb["actuals"], label="Actuals", alpha=0.7)
    ax3.plot(
        train_outputs_pb["date"],
        train_outputs_pb["prophet_fitted"],
        label="Prophet Only",
        alpha=0.7,
        linestyle="--",
    )
    ax3.plot(
        train_outputs_pb["date"], train_outputs_pb["fitted"], label="Prophet+XGBoost", alpha=0.9
    )
    ax3.set_title("Prophet+XGBoost: Training Fit")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Sales")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prophet+XGBoost test predictions
    ax4 = axes[1, 1]
    ax4.plot(test_data["date"], test_actuals, label="Actuals", alpha=0.7)
    ax4.plot(
        test_data["date"],
        preds_prophet_boost[".pred"],
        label="Prophet+XGBoost Predictions",
        alpha=0.9,
    )
    ax4.set_title(f"Prophet+XGBoost: Test Predictions (RMSE={prophet_boost_rmse:.2f})")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Sales")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hybrid_models_comparison.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'hybrid_models_comparison.png'")

except Exception as e:
    print(f"\nCould not create visualization: {e}")

print("\n" + "=" * 70)
print("Demo completed successfully!")
print("=" * 70)
