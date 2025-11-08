"""
Demo: step_corr() - Correlation-based Feature Filtering

This script demonstrates how to use step_corr() to remove highly correlated features
from datasets, which helps reduce multicollinearity in modeling.
"""

import pandas as pd
import numpy as np
from py_recipes import recipe, all_numeric, all_numeric_predictors

# =============================================================================
# 1. BASIC USAGE
# =============================================================================

print("=" * 70)
print("EXAMPLE 1: Basic Correlation Filtering")
print("=" * 70)

# Create data with highly correlated features
np.random.seed(42)
n = 100
x1 = np.random.randn(n)
x2 = x1 + np.random.randn(n) * 0.01  # Very high correlation with x1
x3 = np.random.randn(n)
y = x1 + x3 + np.random.randn(n) * 0.1

data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

print("\nOriginal correlation matrix:")
print(data.corr().round(3))

# Apply step_corr with default threshold (0.9)
rec = recipe(data).step_corr(threshold=0.9)
rec_prepped = rec.prep(data)
result = rec_prepped.bake(data)

print(f"\nOriginal shape: {data.shape}")
print(f"After step_corr: {result.shape}")
print(f"Removed columns: {set(data.columns) - set(result.columns)}")
print(f"\nRemaining correlation matrix:")
print(result.corr().round(3))

# =============================================================================
# 2. DIFFERENT THRESHOLDS
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Different Correlation Thresholds")
print("=" * 70)

# Create data with varying correlation levels
np.random.seed(123)
n = 100
x1 = np.random.randn(n)
x2 = x1 + np.random.randn(n) * 0.3   # High correlation (~0.95)
x3 = x1 + np.random.randn(n) * 0.8   # Moderate correlation (~0.75)
x4 = x1 + np.random.randn(n) * 2.0   # Low correlation (~0.45)
x5 = np.random.randn(n)               # Independent

data2 = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

print("\nCorrelation matrix:")
print(data2.corr().round(3))

for threshold in [0.95, 0.80, 0.60]:
    rec = recipe(data2).step_corr(threshold=threshold)
    result = rec.prep(data2).bake(data2)
    removed = set(data2.columns) - set(result.columns)
    print(f"\nThreshold {threshold}: {len(result.columns)} columns retained")
    if removed:
        print(f"  Removed: {removed}")
    else:
        print(f"  Removed: None")

# =============================================================================
# 3. CORRELATION METHODS
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Different Correlation Methods")
print("=" * 70)

# Create data with non-linear but monotonic relationship
np.random.seed(456)
n = 100
x1 = np.random.uniform(0, 10, n)
x2 = np.log(x1 + 1) + np.random.randn(n) * 0.1  # Non-linear relationship
x3 = np.random.randn(n)

data3 = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

print("\nPearson correlation (linear):")
print(data3.corr(method='pearson').round(3))

print("\nSpearman correlation (monotonic):")
print(data3.corr(method='spearman').round(3))

for method in ['pearson', 'spearman', 'kendall']:
    rec = recipe(data3).step_corr(threshold=0.85, method=method)
    result = rec.prep(data3).bake(data3)
    removed = set(data3.columns) - set(result.columns)
    print(f"\nMethod '{method}': {len(result.columns)} columns retained")
    if removed:
        print(f"  Removed: {removed}")

# =============================================================================
# 4. COLUMN SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Selective Column Filtering")
print("=" * 70)

# Create data with multiple correlated groups
np.random.seed(789)
n = 100
# Group 1: correlated features
g1_x1 = np.random.randn(n)
g1_x2 = g1_x1 + np.random.randn(n) * 0.01

# Group 2: correlated features
g2_x1 = np.random.randn(n)
g2_x2 = g2_x1 + np.random.randn(n) * 0.01

# Independent feature
x_ind = np.random.randn(n)

data4 = pd.DataFrame({
    'g1_x1': g1_x1, 'g1_x2': g1_x2,
    'g2_x1': g2_x1, 'g2_x2': g2_x2,
    'x_ind': x_ind,
    'category': ['A'] * 50 + ['B'] * 50
})

print("\nOriginal data:")
print(f"  Shape: {data4.shape}")
print(f"  Columns: {list(data4.columns)}")

# Filter only group 1 columns
rec1 = recipe(data4).step_corr(columns=['g1_x1', 'g1_x2'], threshold=0.9)
result1 = rec1.prep(data4).bake(data4)
print(f"\nFiltering only ['g1_x1', 'g1_x2']:")
print(f"  Result columns: {list(result1.columns)}")
print(f"  Note: g2_x1 and g2_x2 both retained despite high correlation")

# Filter using selector
rec2 = recipe(data4).step_corr(columns=all_numeric(), threshold=0.9)
result2 = rec2.prep(data4).bake(data4)
print(f"\nFiltering all_numeric() columns:")
print(f"  Result columns: {list(result2.columns)}")
print(f"  Note: 'category' (non-numeric) preserved")

# =============================================================================
# 5. CHAINING WITH OTHER STEPS
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Chaining with Other Recipe Steps")
print("=" * 70)

np.random.seed(999)
n = 100
x1 = np.random.randn(n)
x2 = x1 + np.random.randn(n) * 0.01
x3 = np.random.randn(n) * 10  # Different scale
x4 = np.random.randn(n)
y = x1 + x3 + x4 + np.random.randn(n)

data5 = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})

print("\nOriginal data summary:")
print(data5.describe().loc[['mean', 'std']].round(2))

# Create pipeline: normalize -> remove correlated -> create interactions
rec = (recipe(data5)
       .step_normalize()
       .step_corr(threshold=0.9)
       .step_mutate({'x3_x4': lambda df: df['x3'] * df['x4']}))

rec_prepped = rec.prep(data5)
result = rec_prepped.bake(data5)

print(f"\nPipeline result:")
print(f"  Shape: {result.shape}")
print(f"  Columns: {list(result.columns)}")
print(f"\nTransformed data summary:")
print(result.describe().loc[['mean', 'std']].round(2))

# =============================================================================
# 6. HANDLING MULTIPLE CORRELATED GROUPS
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Multiple Correlated Feature Groups")
print("=" * 70)

np.random.seed(111)
n = 100

# Create 3 groups of highly correlated features
base1 = np.random.randn(n)
group1 = pd.DataFrame({
    'g1_v1': base1,
    'g1_v2': base1 + np.random.randn(n) * 0.01,
    'g1_v3': base1 + np.random.randn(n) * 0.01
})

base2 = np.random.randn(n)
group2 = pd.DataFrame({
    'g2_v1': base2,
    'g2_v2': base2 + np.random.randn(n) * 0.01,
})

base3 = np.random.randn(n)
group3 = pd.DataFrame({
    'g3_v1': base3,
    'g3_v2': base3 + np.random.randn(n) * 0.01,
    'g3_v3': base3 + np.random.randn(n) * 0.01,
    'g3_v4': base3 + np.random.randn(n) * 0.01,
})

# Independent features
independent = pd.DataFrame({
    'ind1': np.random.randn(n),
    'ind2': np.random.randn(n),
})

data6 = pd.concat([group1, group2, group3, independent], axis=1)

print(f"\nOriginal data: {data6.shape[1]} features")
print(f"  Group 1: 3 correlated features")
print(f"  Group 2: 2 correlated features")
print(f"  Group 3: 4 correlated features")
print(f"  Independent: 2 features")

rec = recipe(data6).step_corr(threshold=0.9)
rec_prepped = rec.prep(data6)
result = rec_prepped.bake(data6)

removed_per_group = {
    'Group 1': sum(1 for c in ['g1_v1', 'g1_v2', 'g1_v3'] if c not in result.columns),
    'Group 2': sum(1 for c in ['g2_v1', 'g2_v2'] if c not in result.columns),
    'Group 3': sum(1 for c in ['g3_v1', 'g3_v2', 'g3_v3', 'g3_v4'] if c not in result.columns),
    'Independent': sum(1 for c in ['ind1', 'ind2'] if c not in result.columns)
}

print(f"\nAfter step_corr: {result.shape[1]} features")
print("\nRemoved features per group:")
for group, count in removed_per_group.items():
    print(f"  {group}: {count} removed")

# =============================================================================
# 7. COMPARISON WITH OUTCOME-BASED SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: step_corr() vs step_select_corr()")
print("=" * 70)

np.random.seed(222)
n = 100
x1 = np.random.randn(n)
x2 = x1 + np.random.randn(n) * 0.01  # Correlated with x1, relevant to y
x3 = np.random.randn(n)               # Independent, not relevant to y
y = x1 + np.random.randn(n) * 0.1

data7 = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})

print("\nCorrelations:")
print(data7.corr().round(3))

# step_corr: removes multicollinearity
rec1 = recipe(data7).step_corr(threshold=0.9)
result1 = rec1.prep(data7).bake(data7)
print(f"\nstep_corr() (removes multicollinearity):")
print(f"  Retained: {list(result1.columns)}")
print(f"  Note: Removes one of {x1, x2} due to high correlation")

# step_select_corr: keeps features correlated with outcome
rec2 = recipe(data7).step_select_corr(outcome='y', threshold=0.8, method='outcome')
result2 = rec2.prep(data7).bake(data7)
print(f"\nstep_select_corr() with method='outcome':")
print(f"  Retained: {list(result2.columns)}")
print(f"  Note: Keeps features correlated with 'y', removes uncorrelated x3")

print("\n" + "=" * 70)
print("Demo completed!")
print("=" * 70)
