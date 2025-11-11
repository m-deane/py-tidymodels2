"""
Test workflow with step_naomit to debug why NaN rows aren't being removed.
"""

import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe, starts_with
from py_parsnip import linear_reg

# Create grouped data similar to notebook
np.random.seed(42)
n_per_group = 50

data_list = []
for country in ['USA', 'UK']:
    group_data = pd.DataFrame({
        'country': [country] * n_per_group,
        'date': pd.date_range('2020-01-01', periods=n_per_group),
        'x1': np.random.randn(n_per_group) * 10 + 50,
        'x2': np.random.randn(n_per_group) * 5 + 20,
        'refinery_kbd': np.random.randn(n_per_group) * 15 + 100
    })
    data_list.append(group_data)

data = pd.concat(data_list, ignore_index=True)
train_data = data.iloc[:80]

print("=" * 70)
print("Debug: Workflow with step_lag + step_naomit")
print("=" * 70)
print(f"Train data shape: {train_data.shape}")
print(f"Train columns: {list(train_data.columns)}")

# Create recipe with lag and naomit (like notebook)
rec_lag = (
    recipe()
    .step_lag(starts_with(""), lags=[1, 2, 3])
    .step_naomit()
)

wf_lag = (
    workflow()
    .add_recipe(rec_lag)
    .add_model(linear_reg())
)

print("\nFitting nested workflow...")
try:
    fit_lag = wf_lag.fit_nested(train_data, group_col='country', per_group_prep=True)
    print("✓ SUCCESS: fit_nested() completed")
except Exception as e:
    print(f"✗ FAILED: {str(e)[:200]}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
