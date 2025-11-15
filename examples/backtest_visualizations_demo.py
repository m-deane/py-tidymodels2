"""
Demo: Backtesting Visualizations

Shows how to use the 4 visualization functions from py_backtest to analyze
backtest performance across vintages, workflows, and forecast horizons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from py_backtest import VintageCV, BacktestResults, create_vintage_data
from py_workflows import workflow
from py_parsnip import linear_reg, rand_forest
from py_workflowsets import WorkflowSet
from py_yardstick import metric_set, rmse, mae

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Create Sample Data with Vintages
# ============================================================================

# Create base time series data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Simulated features
df = pd.DataFrame({
    'date': dates,
    'x1': np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.randn(n) * 0.1,
    'x2': np.cos(np.arange(n) * 2 * np.pi / 365) + np.random.randn(n) * 0.1,
    'x3': np.random.randn(n) * 0.5,
    'target': np.sin(np.arange(n) * 2 * np.pi / 365) * 10 + 50 + np.random.randn(n) * 2
})

# Create vintage data (simulates data revisions)
vintage_df = create_vintage_data(
    final_data=df,
    date_col='date',
    n_revisions=3,
    revision_std=0.5
)

print(f"Created vintage data: {len(vintage_df)} rows")
print(f"Vintages: {vintage_df['as_of_date'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# 2. Create Workflows to Compare
# ============================================================================

# Define multiple workflows
workflows_list = [
    workflow()
        .add_formula("target ~ x1 + x2")
        .add_model(linear_reg()),

    workflow()
        .add_formula("target ~ x1 + x2 + x3")
        .add_model(linear_reg()),

    workflow()
        .add_formula("target ~ x1 + x2 + x3")
        .add_model(rand_forest(trees=50).set_mode("regression")),
]

# Create WorkflowSet
wf_set = WorkflowSet.from_workflows(workflows_list)

print(f"\nCreated {len(wf_set.workflow_ids)} workflows:")
for wf_id in wf_set.workflow_ids:
    print(f"  - {wf_id}")

# ============================================================================
# 3. Setup Vintage Cross-Validation
# ============================================================================

vintage_cv = VintageCV(
    data=vintage_df,
    as_of_col='as_of_date',
    date_col='date',
    initial='2 years',
    assess='3 months',
    skip='2 months',
    lag='1 week',
    vintage_selection='latest',
    slice_limit=6  # Limit to 6 folds for demo
)

print(f"\nVintage CV: {len(vintage_cv)} folds")
print(f"Initial training: 2 years")
print(f"Assessment: 3 months")
print(f"Skip: 2 months")

# ============================================================================
# 4. Run Backtests
# ============================================================================

print("\nRunning backtests...")

results = wf_set.fit_backtests(
    resamples=vintage_cv,
    metrics=metric_set(rmse, mae)
)

print(f"✓ Backtest complete!")

# ============================================================================
# 5. Analyze Results
# ============================================================================

print("\n" + "="*70)
print("BACKTEST ANALYSIS")
print("="*70)

# Collect metrics
metrics_df = results.collect_metrics(by_vintage=False)
print("\nAverage Performance Across Vintages:")
print(metrics_df)

# Rank workflows
ranked = results.rank_results("rmse", n=3)
print("\nTop Workflows by RMSE:")
print(ranked)

# Analyze vintage drift
drift = results.analyze_vintage_drift("rmse")
print("\nVintage Drift Analysis (RMSE):")
print(drift.head(10))

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# ---------------------------------------------------------------------------
# Plot 1: Accuracy Over Time
# ---------------------------------------------------------------------------
print("\n1. Accuracy Over Time")

# By workflow (separate lines)
fig1 = results.plot_accuracy_over_time(
    metric="rmse",
    by_workflow=True,
    show=False,
    figsize=(14, 6)
)
fig1.savefig("examples/backtest_accuracy_over_time.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_accuracy_over_time.png")

# Aggregated (with confidence bands)
fig2 = results.plot_accuracy_over_time(
    metric="rmse",
    by_workflow=False,
    show=False,
    figsize=(14, 6)
)
fig2.savefig("examples/backtest_accuracy_aggregated.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_accuracy_aggregated.png")

# ---------------------------------------------------------------------------
# Plot 2: Horizon Comparison
# ---------------------------------------------------------------------------
print("\n2. Forecast Horizon Comparison")

fig3 = results.plot_horizon_comparison(
    metric="rmse",
    show=False,
    figsize=(12, 6)
)
fig3.savefig("examples/backtest_horizon_comparison.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_horizon_comparison.png")

# ---------------------------------------------------------------------------
# Plot 3: Vintage Drift
# ---------------------------------------------------------------------------
print("\n3. Vintage Drift Analysis")

fig4 = results.plot_vintage_drift(
    metric="rmse",
    show=False,
    figsize=(14, 8)
)
fig4.savefig("examples/backtest_vintage_drift.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_vintage_drift.png")

# ---------------------------------------------------------------------------
# Plot 4: Revision Impact (Placeholder - no final data)
# ---------------------------------------------------------------------------
print("\n4. Data Revision Impact")

fig5 = results.plot_revision_impact(
    metric="rmse",
    show=False,
    figsize=(10, 6)
)
fig5.savefig("examples/backtest_revision_impact_placeholder.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_revision_impact_placeholder.png")
print("   Note: Shows placeholder (no final data provided)")

# ---------------------------------------------------------------------------
# Plot 4b: Revision Impact with Simulated Final Data
# ---------------------------------------------------------------------------
print("\n5. Data Revision Impact (with simulated final data)")

# Simulate final data metrics (vintage metrics + small random change)
vintage_metrics = results.collect_metrics(by_vintage=False, summarize=False)
vintage_rmse = vintage_metrics[vintage_metrics["metric"] == "rmse"].copy()

# Create simulated final metrics
vintage_vs_final = pd.DataFrame({
    'wflow_id': vintage_rmse['wflow_id'].values,
    'vintage_rmse': vintage_rmse['value'].values,
    'final_rmse': vintage_rmse['value'].values - np.random.uniform(0.1, 0.5, len(vintage_rmse))
})

fig6 = results.plot_revision_impact(
    metric="rmse",
    vintage_vs_final_data=vintage_vs_final,
    show=False,
    figsize=(10, 6)
)
fig6.savefig("examples/backtest_revision_impact_simulated.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/backtest_revision_impact_simulated.png")
print("   Note: Uses simulated final data for demonstration")

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nBacktest Results:")
print(f"  - Workflows tested: {len(wf_set.workflow_ids)}")
print(f"  - Vintages evaluated: {len(vintage_cv)}")
print(f"  - Metrics computed: rmse, mae")

print(f"\nVisualizations Created:")
print(f"  1. Accuracy over time (by workflow)")
print(f"  2. Accuracy over time (aggregated)")
print(f"  3. Forecast horizon comparison")
print(f"  4. Vintage drift analysis")
print(f"  5. Revision impact (placeholder)")
print(f"  6. Revision impact (with simulated data)")

print(f"\nBest Workflow by RMSE:")
best_wf = ranked.iloc[0]["wflow_id"]
best_rmse = ranked.iloc[0]["mean_rmse"]
print(f"  {best_wf}: RMSE = {best_rmse:.4f}")

print("\n✓ Demo complete!")
print("\nTo view plots, open the PNG files in examples/ directory")
