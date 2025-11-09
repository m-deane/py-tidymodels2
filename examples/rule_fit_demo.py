"""
Demo of rule_fit model with rule extraction

This script demonstrates:
1. Fitting a RuleFit model for regression
2. Extracting interpretable rules
3. Making predictions
4. Classification with rules
"""

import pandas as pd
import numpy as np
from py_parsnip import rule_fit

# Set random seed for reproducibility
np.random.seed(42)

# ======================
# REGRESSION EXAMPLE
# ======================
print("=" * 60)
print("REGRESSION EXAMPLE")
print("=" * 60)

# Create sample data with nonlinear relationships
n = 200
train_data = pd.DataFrame({
    "y": np.random.randn(n) * 10 + 50 +
         5 * (np.random.randn(n) > 0) +  # Nonlinear effect
         3 * (np.random.randn(n) * 5 > 2),  # Another nonlinear effect
    "x1": np.random.randn(n) * 5,
    "x2": np.random.randn(n) * 3,
    "x3": np.random.randn(n) * 2,
})

# Fit RuleFit model
print("\nFitting RuleFit model...")
spec = rule_fit(max_rules=15, tree_depth=3, penalty=0.01).set_mode("regression")
fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

print(f"✓ Model fitted successfully")
print(f"  Model type: {fit.spec.model_type}")
print(f"  Mode: {fit.spec.mode}")
print(f"  Max rules: {fit.fit_data['model'].max_rules}")

# Extract outputs to see rules
outputs, coefficients, stats = fit.extract_outputs()

print("\n" + "=" * 60)
print("EXTRACTED RULES (Interpretability)")
print("=" * 60)
print(f"\nTotal rules/features: {len(coefficients)}")
print("\nTop 10 most important rules:")
print(coefficients.nlargest(10, 'importance')[['variable', 'coefficient', 'importance']])

# Get statistics
print("\n" + "=" * 60)
print("MODEL STATISTICS")
print("=" * 60)
train_stats = stats[stats['split'] == 'train']
print(f"\nTraining Metrics:")
print(f"  RMSE: {train_stats[train_stats['metric'] == 'rmse']['value'].values[0]:.4f}")
print(f"  MAE: {train_stats[train_stats['metric'] == 'mae']['value'].values[0]:.4f}")
print(f"  R²: {train_stats[train_stats['metric'] == 'r_squared']['value'].values[0]:.4f}")

n_rules = stats[stats['metric'] == 'n_rules']['value'].values[0]
print(f"\n  Number of rules: {int(n_rules)}")

# Make predictions
print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)
test_data = pd.DataFrame({
    "x1": [2.0, -1.0, 0.5],
    "x2": [1.5, -0.5, 0.2],
    "x3": [0.8, -0.3, 0.1],
})

predictions = fit.predict(test_data, type="numeric")
print("\nTest predictions:")
print(predictions)

# ======================
# CLASSIFICATION EXAMPLE
# ======================
print("\n\n" + "=" * 60)
print("CLASSIFICATION EXAMPLE")
print("=" * 60)

# Create sample classification data
n = 200
train_data_class = pd.DataFrame({
    "y": np.random.choice([0, 1], n),
    "x1": np.random.randn(n) * 5,
    "x2": np.random.randn(n) * 3,
    "x3": np.random.randn(n) * 2,
})

# Fit classification model
print("\nFitting RuleFit classifier...")
spec_class = rule_fit(max_rules=12, tree_depth=3, penalty=0.001).set_mode("classification")
fit_class = spec_class.fit(train_data_class, "y ~ x1 + x2 + x3")

print(f"✓ Model fitted successfully")
print(f"  Model class: {fit_class.fit_data['model_class']}")

# Extract rules
outputs_class, coefficients_class, stats_class = fit_class.extract_outputs()

print("\n" + "=" * 60)
print("CLASSIFICATION RULES")
print("=" * 60)
print(f"\nTotal rules/features: {len(coefficients_class)}")
print("\nTop 8 most important rules:")
print(coefficients_class.nlargest(8, 'importance')[['variable', 'coefficient', 'importance']])

# Classification predictions
test_data_class = pd.DataFrame({
    "x1": [2.0, -1.0, 0.5, -2.0],
    "x2": [1.5, -0.5, 0.2, -1.0],
    "x3": [0.8, -0.3, 0.1, -0.5],
})

print("\n" + "=" * 60)
print("CLASSIFICATION PREDICTIONS")
print("=" * 60)

# Class predictions
class_preds = fit_class.predict(test_data_class, type="class")
print("\nClass predictions:")
print(class_preds)

# Probability predictions
prob_preds = fit_class.predict(test_data_class, type="prob")
print("\nProbability predictions:")
print(prob_preds)

# Get classification metrics
train_stats_class = stats_class[stats_class['split'] == 'train']
print("\n" + "=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)
print(f"Accuracy: {train_stats_class[train_stats_class['metric'] == 'accuracy']['value'].values[0]:.4f}")
print(f"Precision: {train_stats_class[train_stats_class['metric'] == 'precision']['value'].values[0]:.4f}")
print(f"Recall: {train_stats_class[train_stats_class['metric'] == 'recall']['value'].values[0]:.4f}")
print(f"F1 Score: {train_stats_class[train_stats_class['metric'] == 'f1_score']['value'].values[0]:.4f}")

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("\nKey Features Demonstrated:")
print("✓ Regression and classification support")
print("✓ Rule extraction for interpretability")
print("✓ Comprehensive three-DataFrame outputs")
print("✓ Parameter tuning (max_rules, tree_depth, penalty)")
print("✓ Multiple prediction types (numeric, class, prob)")
