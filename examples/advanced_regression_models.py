"""
Example usage of advanced regression models in py-tidymodels

This script demonstrates:
1. MARS (Multivariate Adaptive Regression Splines)
2. Poisson Regression (for count data)
3. Generalized Additive Models (GAM)
"""

import numpy as np
import pandas as pd
from py_parsnip import mars, poisson_reg, gen_additive_mod

print("=" * 80)
print("ADVANCED REGRESSION MODELS IN PY-TIDYMODELS")
print("=" * 80)

# =============================================================================
# 1. MARS - Multivariate Adaptive Regression Splines
# =============================================================================
print("\n1. MARS (Multivariate Adaptive Regression Splines)")
print("-" * 80)
print("MARS automatically detects non-linear relationships using piecewise")
print("linear basis functions (hinge functions).\n")

# Create non-linear data
np.random.seed(42)
x = np.linspace(0, 10, 60)
y_mars = x**2 + np.sin(x) * 5 + np.random.normal(0, 3, 60)
data_mars = pd.DataFrame({"y": y_mars, "x": x})

# Split into train/test
train_mars = data_mars.iloc[:45]
test_mars = data_mars.iloc[45:]

# Fit MARS model
print("Fitting MARS model with 15 terms and pairwise interactions...")
spec_mars = mars(num_terms=15, prod_degree=2)
fit_mars = spec_mars.fit(train_mars, "y ~ x")

# Predict
pred_mars = fit_mars.predict(test_mars)
print(f"Predictions shape: {pred_mars.shape}")
print(f"First 5 predictions:\n{pred_mars.head()}")

# Extract outputs
outputs_mars, basis_funcs, stats_mars = fit_mars.extract_outputs()
print(f"\nModel used {len(basis_funcs)} basis functions")
print(f"Training R-squared: {stats_mars[stats_mars['metric'] == 'r_squared']['value'].iloc[0]:.4f}")

# =============================================================================
# 2. Poisson Regression - For Count Data
# =============================================================================
print("\n\n2. Poisson Regression (for count data)")
print("-" * 80)
print("Poisson regression models count outcomes using a log link function.")
print("Ideal for: event counts, rare events, non-negative integer outcomes.\n")

# Create count data
np.random.seed(42)
x1_pois = np.random.uniform(0, 5, 50)
x2_pois = np.random.uniform(0, 3, 50)
lambda_true = np.exp(0.3 + 0.4 * x1_pois + 0.5 * x2_pois)
counts = np.random.poisson(lambda_true)
data_pois = pd.DataFrame({"count": counts, "x1": x1_pois, "x2": x2_pois})

# Split into train/test
train_pois = data_pois.iloc[:35]
test_pois = data_pois.iloc[35:]

# Fit Poisson model
print("Fitting Poisson GLM...")
spec_pois = poisson_reg()
fit_pois = spec_pois.fit(train_pois, "count ~ x1 + x2")

# Predict with confidence intervals
pred_pois = fit_pois.predict(test_pois, type="conf_int")
print(f"Predictions with confidence intervals:\n{pred_pois.head()}")

# Extract outputs
outputs_pois, coefs_pois, stats_pois = fit_pois.extract_outputs()
print(f"\nCoefficients (with z-statistics):")
print(coefs_pois[["variable", "coefficient", "z_stat", "p_value"]].to_string())
print(f"\nAIC: {stats_pois[stats_pois['metric'] == 'aic']['value'].iloc[0]:.2f}")
print(f"Deviance: {stats_pois[stats_pois['metric'] == 'deviance']['value'].iloc[0]:.2f}")

# =============================================================================
# 3. Generalized Additive Model (GAM)
# =============================================================================
print("\n\n3. Generalized Additive Model (GAM)")
print("-" * 80)
print("GAMs fit smooth non-parametric functions to each predictor,")
print("automatically detecting non-linear relationships.\n")

# Create data with non-linear relationships
np.random.seed(42)
x1_gam = np.linspace(0, 10, 50)
x2_gam = np.linspace(0, 5, 50)
y_gam = np.sin(x1_gam) * 10 + x2_gam**2 + np.random.normal(0, 2, 50)
data_gam = pd.DataFrame({"y": y_gam, "x1": x1_gam, "x2": x2_gam})

# Split into train/test
train_gam = data_gam.iloc[:35]
test_gam = data_gam.iloc[35:]

# Fit GAM with moderate smoothing
print("Fitting GAM with 12 splines per feature...")
spec_gam = gen_additive_mod(adjust_deg_free=12)
fit_gam = spec_gam.fit(train_gam, "y ~ x1 + x2")

# Predict
pred_gam = fit_gam.predict(test_gam)
print(f"Predictions shape: {pred_gam.shape}")
print(f"First 5 predictions:\n{pred_gam.head()}")

# Extract outputs
outputs_gam, partial_effects, stats_gam = fit_gam.extract_outputs()
print(f"\nPartial effects (feature contributions):")
print(partial_effects[["feature", "effect_range", "data_range"]].to_string())
print(f"\nGCV score: {stats_gam[stats_gam['metric'] == 'gcv']['value'].iloc[0]:.4f}")
print(f"R-squared: {stats_gam[stats_gam['metric'] == 'r_squared']['value'].iloc[0]:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nAll three advanced regression models successfully fitted and predicted!")
print("\nModel Characteristics:")
print("  - MARS: Best for automatic non-linearity and interaction detection")
print("  - Poisson: Best for count data and rare events")
print("  - GAM: Best for flexible non-parametric smooth relationships")
print("\nAll models support:")
print("  - Three-DataFrame output structure (outputs, coefficients/effects, stats)")
print("  - Standard predict() interface")
print("  - Evaluation with train/test splits")
print("=" * 80)
