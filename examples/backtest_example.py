"""
Simple example demonstrating backtesting with data vintages.

This example shows how to:
1. Create synthetic vintage data
2. Set up vintage cross-validation
3. Backtest multiple workflows
4. Analyze results
"""

import pandas as pd
import numpy as np

from py_backtest import VintageCV, create_vintage_data
from py_workflows import workflow
from py_parsnip import linear_reg
from py_yardstick import metric_set, rmse, mae

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("BACKTESTING WITH DATA VINTAGES - EXAMPLE")
print("=" * 60)

# 1. Create final data
print("\n1. Creating final data...")
n = 365
dates = pd.date_range('2023-01-01', periods=n, freq='D')
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = 2.0 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5

final_data = pd.DataFrame({
    'date': dates,
    'x1': x1,
    'x2': x2,
    'y': y
})

print(f"   - Created {len(final_data)} days of final data")

# 2. Create vintage data (simulate data revisions)
print("\n2. Creating vintage data (simulating revisions)...")
vintage_df = create_vintage_data(
    final_data,
    date_col='date',
    n_revisions=3,
    revision_std=0.05,  # 5% measurement noise
    revision_lag='7 days'
)

print(f"   - Created {len(vintage_df)} vintage rows")
print(f"   - {len(vintage_df) / len(final_data)} vintages per observation")

# 3. Create workflows
print("\n3. Creating workflows...")
wf1 = workflow().add_formula('y ~ x1').add_model(linear_reg())
wf2 = workflow().add_formula('y ~ x1 + x2').add_model(linear_reg())

from py_workflowsets import WorkflowSet
wf_set = WorkflowSet.from_workflows(
    [wf1, wf2],
    ids=['simple', 'full']
)

print(f"   - Created {len(wf_set.workflow_ids)} workflows")
print(f"   - IDs: {wf_set.workflow_ids}")

# 4. Create vintage CV
print("\n4. Creating vintage CV...")
vintage_cv = VintageCV(
    data=vintage_df,
    as_of_col='as_of_date',
    date_col='date',
    initial='180 days',    # 6 months training
    assess='30 days',      # 1 month test
    skip='15 days',        # 2-week gap between folds
    lag='7 days'           # 1-week-ahead forecasts
)

print(f"   - Created {len(vintage_cv)} vintage folds")

# Show first fold info
first_fold = vintage_cv[0]
info = first_fold.get_vintage_info()
print(f"\n   First fold:")
print(f"   - Vintage date: {info['vintage_date'].date()}")
print(f"   - Training: {info['training_start'].date()} to {info['training_end'].date()}")
print(f"   - Test: {info['test_start'].date()} to {info['test_end'].date()}")
print(f"   - Forecast horizon: {info['forecast_horizon']}")
print(f"   - Training obs: {info['n_train_obs']}")
print(f"   - Test obs: {info['n_test_obs']}")

# 5. Backtest all workflows
print("\n5. Backtesting workflows...")
print("   (This may take a minute...)")

results = wf_set.fit_backtests(
    vintage_cv,
    metrics=metric_set(rmse, mae),
    verbose=False
)

print(f"   - Completed backtesting {len(results.workflow_ids)} workflows")

# 6. Analyze results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Rank workflows
print("\n6a. Top Models (by RMSE):")
ranked = results.rank_results('rmse', n=2)
print(ranked.to_string(index=False))

# Analyze vintage drift
print("\n6b. Accuracy Degradation Over Time:")
drift = results.analyze_vintage_drift('rmse')
print("\nSimple model drift:")
simple_drift = drift[drift['wflow_id'] == 'simple'].head(3)
print(simple_drift[['vintage_date', 'metric_value', 'drift_from_start', 'drift_pct']].to_string(index=False))

print("\nFull model drift:")
full_drift = drift[drift['wflow_id'] == 'full'].head(3)
print(full_drift[['vintage_date', 'metric_value', 'drift_from_start', 'drift_pct']].to_string(index=False))

# Extract best workflow
best = results.extract_best_workflow('rmse')
print(f"\n6c. Best Workflow: {best}")

# Collect metrics summary
print("\n6d. Metrics Summary:")
metrics = results.collect_metrics(by_vintage=False, summarize=True)
print(metrics.to_string(index=False))

print("\n" + "=" * 60)
print("EXAMPLE COMPLETE")
print("=" * 60)
print("\nKey Takeaways:")
print("- Created vintage data with 3 revisions per observation")
print("- Backtested 2 workflows across multiple vintage folds")
print("- Analyzed forecast accuracy degradation over time")
print("- Identified best workflow for production use")
print("\nThe full model (y ~ x1 + x2) should perform better than")
print("the simple model (y ~ x1) since the true relationship uses both features.")
