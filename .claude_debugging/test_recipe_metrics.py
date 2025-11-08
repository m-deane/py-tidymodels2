"""
Quick test to verify recipe workflows collect metrics correctly
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

from py_workflows import workflow
from py_parsnip import linear_reg
from py_recipes import recipe
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae, r_squared
from py_tune import fit_resamples

# Create simple test data
df = pd.DataFrame({
    'target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
})

# Create recipe workflow
rec = recipe().step_normalize(['x1', 'x2'])
wf = workflow().add_recipe(rec).add_model(linear_reg())

# Create CV folds
folds = vfold_cv(df, v=3)

# Fit resamples
print("Testing recipe workflow with fit_resamples...")
results = fit_resamples(
    wf,
    folds,
    metrics=metric_set(rmse, mae, r_squared)
)

# Check metrics
metrics_df = results.collect_metrics()
print(f"\n✓ Metrics collected: {len(metrics_df)} rows")
print(f"Metrics DataFrame:\n{metrics_df}")

if len(metrics_df) > 0:
    print("\n✓ SUCCESS: Recipe workflows now collect metrics correctly!")
    print(f"Metrics found: {metrics_df['metric'].unique().tolist()}")
else:
    print("\n✗ FAILURE: No metrics collected")
    sys.exit(1)
