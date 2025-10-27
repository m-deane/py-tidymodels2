"""
Demo script for boost_tree model with XGBoost, LightGBM, and CatBoost engines
"""

import pandas as pd
import numpy as np
from py_parsnip import boost_tree

# Create sample data
np.random.seed(42)
n = 100
x1 = np.random.uniform(10, 30, n)
x2 = np.random.uniform(5, 15, n)
x3 = np.random.uniform(2, 6, n)
# Create target with non-linear relationships
y = 50 + 5 * x1 + 3 * x2 + 2 * x3 + 0.1 * x1 * x2 + np.random.normal(0, 10, n)

data = pd.DataFrame({
    "sales": y,
    "price": x1,
    "advertising": x2,
    "competition": x3,
})

# Split data
train = data.iloc[:80]
test = data.iloc[80:]

print("=" * 70)
print("BOOST_TREE MODEL DEMONSTRATION")
print("=" * 70)

# ============================================================================
# 1. XGBoost Engine
# ============================================================================
print("\n" + "=" * 70)
print("1. XGBoost Engine")
print("=" * 70)

spec_xgb = boost_tree(
    trees=100,
    tree_depth=6,
    learn_rate=0.1,
    mtry=0.8,
    min_n=5
)

print(f"\nModel Specification:")
print(f"  Model Type: {spec_xgb.model_type}")
print(f"  Engine: {spec_xgb.engine}")
print(f"  Parameters: {spec_xgb.args}")

fit_xgb = spec_xgb.fit(train, "sales ~ price + advertising + competition")
print(f"\nFitted Model:")
print(f"  Class: {fit_xgb.fit_data['model_class']}")
print(f"  Features: {fit_xgb.fit_data['n_features']}")
print(f"  Observations: {fit_xgb.fit_data['n_obs']}")

predictions_xgb = fit_xgb.predict(test)
print(f"\nPredictions (first 5):")
print(predictions_xgb.head())

# Extract outputs
fit_xgb = fit_xgb.evaluate(test)
outputs_xgb, importance_xgb, stats_xgb = fit_xgb.extract_outputs()

print(f"\nFeature Importance:")
print(importance_xgb[["variable", "importance"]].to_string(index=False))

print(f"\nTraining Metrics:")
train_metrics = stats_xgb[stats_xgb["split"] == "train"][["metric", "value"]]
train_metrics = train_metrics[train_metrics["metric"].isin(["rmse", "mae", "r_squared"])]
print(train_metrics.to_string(index=False))

print(f"\nTest Metrics:")
test_metrics = stats_xgb[stats_xgb["split"] == "test"][["metric", "value"]]
test_metrics = test_metrics[test_metrics["metric"].isin(["rmse", "mae", "r_squared"])]
print(test_metrics.to_string(index=False))

# ============================================================================
# 2. LightGBM Engine
# ============================================================================
print("\n" + "=" * 70)
print("2. LightGBM Engine")
print("=" * 70)

spec_lgb = boost_tree(
    trees=100,
    tree_depth=6,
    learn_rate=0.1,
    mtry=0.8,
    min_n=5
).set_engine("lightgbm")

print(f"\nModel Specification:")
print(f"  Model Type: {spec_lgb.model_type}")
print(f"  Engine: {spec_lgb.engine}")
print(f"  Parameters: {spec_lgb.args}")

fit_lgb = spec_lgb.fit(train, "sales ~ price + advertising + competition")
print(f"\nFitted Model:")
print(f"  Class: {fit_lgb.fit_data['model_class']}")
print(f"  Features: {fit_lgb.fit_data['n_features']}")
print(f"  Observations: {fit_lgb.fit_data['n_obs']}")

predictions_lgb = fit_lgb.predict(test)
print(f"\nPredictions (first 5):")
print(predictions_lgb.head())

# Extract outputs
fit_lgb = fit_lgb.evaluate(test)
outputs_lgb, importance_lgb, stats_lgb = fit_lgb.extract_outputs()

print(f"\nFeature Importance:")
print(importance_lgb[["variable", "importance"]].to_string(index=False))

print(f"\nTest Metrics:")
test_metrics = stats_lgb[stats_lgb["split"] == "test"][["metric", "value"]]
test_metrics = test_metrics[test_metrics["metric"].isin(["rmse", "mae", "r_squared"])]
print(test_metrics.to_string(index=False))

# ============================================================================
# 3. CatBoost Engine
# ============================================================================
print("\n" + "=" * 70)
print("3. CatBoost Engine")
print("=" * 70)

spec_cat = boost_tree(
    trees=100,
    tree_depth=6,
    learn_rate=0.1,
    mtry=0.8,
    min_n=5
).set_engine("catboost")

print(f"\nModel Specification:")
print(f"  Model Type: {spec_cat.model_type}")
print(f"  Engine: {spec_cat.engine}")
print(f"  Parameters: {spec_cat.args}")

fit_cat = spec_cat.fit(train, "sales ~ price + advertising + competition")
print(f"\nFitted Model:")
print(f"  Class: {fit_cat.fit_data['model_class']}")
print(f"  Features: {fit_cat.fit_data['n_features']}")
print(f"  Observations: {fit_cat.fit_data['n_obs']}")

predictions_cat = fit_cat.predict(test)
print(f"\nPredictions (first 5):")
print(predictions_cat.head())

# Extract outputs
fit_cat = fit_cat.evaluate(test)
outputs_cat, importance_cat, stats_cat = fit_cat.extract_outputs()

print(f"\nFeature Importance:")
print(importance_cat[["variable", "importance"]].to_string(index=False))

print(f"\nTest Metrics:")
test_metrics = stats_cat[stats_cat["split"] == "test"][["metric", "value"]]
test_metrics = test_metrics[test_metrics["metric"].isin(["rmse", "mae", "r_squared"])]
print(test_metrics.to_string(index=False))

# ============================================================================
# 4. Compare All Engines
# ============================================================================
print("\n" + "=" * 70)
print("4. Engine Comparison")
print("=" * 70)

comparison = pd.DataFrame({
    "Engine": ["XGBoost", "LightGBM", "CatBoost"],
    "RMSE": [
        stats_xgb[stats_xgb["metric"] == "rmse"][stats_xgb["split"] == "test"]["value"].iloc[0],
        stats_lgb[stats_lgb["metric"] == "rmse"][stats_lgb["split"] == "test"]["value"].iloc[0],
        stats_cat[stats_cat["metric"] == "rmse"][stats_cat["split"] == "test"]["value"].iloc[0],
    ],
    "R²": [
        stats_xgb[stats_xgb["metric"] == "r_squared"][stats_xgb["split"] == "test"]["value"].iloc[0],
        stats_lgb[stats_lgb["metric"] == "r_squared"][stats_lgb["split"] == "test"]["value"].iloc[0],
        stats_cat[stats_cat["metric"] == "r_squared"][stats_cat["split"] == "test"]["value"].iloc[0],
    ],
})

print("\nTest Set Performance:")
print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
