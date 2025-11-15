"""
Simple Demo: Backtesting Visualizations

Demonstrates the 4 visualization functions using mock BacktestResults.
This is a standalone demo that doesn't require running actual backtests.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from py_backtest import BacktestResults

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Create Mock Backtest Results
# ============================================================================

print("Creating mock backtest results...")

vintage_dates = pd.date_range(start='2023-01-01', end='2023-06-01', freq='MS')

results = {}

for wf_id in ['linear_reg', 'rand_forest', 'xgboost']:
    folds = []

    # Simulate different performance characteristics
    base_rmse = {'linear_reg': 5.0, 'rand_forest': 4.5, 'xgboost': 4.2}[wf_id]
    drift_rate = {'linear_reg': 0.5, 'rand_forest': 0.3, 'xgboost': 0.2}[wf_id]

    for i, vintage_date in enumerate(vintage_dates):
        # Simulate metrics with drift over time
        metrics = pd.DataFrame({
            'metric': ['rmse', 'mae', 'r_squared'],
            'value': [
                base_rmse + i * drift_rate + np.random.randn() * 0.2,
                base_rmse * 0.8 + i * drift_rate * 0.8 + np.random.randn() * 0.15,
                0.85 - i * 0.02 - np.random.rand() * 0.03
            ]
        })

        # Create varying forecast horizons
        horizon = timedelta(days=1 + i * 3)

        vintage_info = {
            'vintage_date': vintage_date,
            'training_start': vintage_date - timedelta(days=730),  # 2 years
            'training_end': vintage_date,
            'test_start': vintage_date + horizon,
            'test_end': vintage_date + horizon + timedelta(days=90),  # 3 months
            'n_train_obs': 730,
            'n_test_obs': 90,
            'forecast_horizon': horizon
        }

        folds.append({
            'fold_idx': i,
            'vintage_info': vintage_info,
            'metrics': metrics
        })

    results[wf_id] = {
        'wflow_id': wf_id,
        'folds': folds
    }

backtest_results = BacktestResults(results)

print(f"✓ Created mock results for {len(backtest_results.workflow_ids)} workflows")
print(f"  Workflows: {', '.join(backtest_results.workflow_ids)}")
print(f"  Vintages: {len(vintage_dates)}")

# ============================================================================
# Analyze Results
# ============================================================================

print("\n" + "="*70)
print("BACKTEST ANALYSIS")
print("="*70)

# Collect metrics
metrics_df = backtest_results.collect_metrics(by_vintage=False)
print("\nAverage Performance Across Vintages:")
print(metrics_df.to_string())

# Rank workflows
ranked = backtest_results.rank_results("rmse", n=3)
print("\n\nTop Workflows by RMSE:")
print(ranked.to_string())

# Analyze vintage drift
drift = backtest_results.analyze_vintage_drift("rmse")
print("\n\nVintage Drift Analysis (RMSE):")
print(drift.to_string())

# Analyze forecast horizon
horizon_perf = backtest_results.analyze_forecast_horizon("rmse")
print("\n\nPerformance by Forecast Horizon:")
print(horizon_perf.to_string())

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# ---------------------------------------------------------------------------
# Plot 1: Accuracy Over Time (by workflow)
# ---------------------------------------------------------------------------
print("\n1. Accuracy Over Time (by workflow)")

fig1 = backtest_results.plot_accuracy_over_time(
    metric="rmse",
    by_workflow=True,
    show=False,
    figsize=(14, 6)
)
fig1.savefig("examples/viz_demo_accuracy_by_workflow.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_accuracy_by_workflow.png")

# ---------------------------------------------------------------------------
# Plot 2: Accuracy Over Time (aggregated)
# ---------------------------------------------------------------------------
print("\n2. Accuracy Over Time (aggregated with confidence bands)")

fig2 = backtest_results.plot_accuracy_over_time(
    metric="rmse",
    by_workflow=False,
    show=False,
    figsize=(14, 6)
)
fig2.savefig("examples/viz_demo_accuracy_aggregated.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_accuracy_aggregated.png")

# ---------------------------------------------------------------------------
# Plot 3: Forecast Horizon Comparison
# ---------------------------------------------------------------------------
print("\n3. Forecast Horizon Comparison")

fig3 = backtest_results.plot_horizon_comparison(
    metric="rmse",
    show=False,
    figsize=(12, 6)
)
fig3.savefig("examples/viz_demo_horizon_comparison.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_horizon_comparison.png")

# ---------------------------------------------------------------------------
# Plot 4: Vintage Drift
# ---------------------------------------------------------------------------
print("\n4. Vintage Drift Analysis")

fig4 = backtest_results.plot_vintage_drift(
    metric="rmse",
    show=False,
    figsize=(14, 8)
)
fig4.savefig("examples/viz_demo_vintage_drift.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_vintage_drift.png")

# ---------------------------------------------------------------------------
# Plot 5: Revision Impact (with simulated final data)
# ---------------------------------------------------------------------------
print("\n5. Data Revision Impact (with simulated final data)")

# Simulate final data metrics
vintage_metrics = backtest_results.collect_metrics(by_vintage=False, summarize=False)
vintage_rmse = vintage_metrics[vintage_metrics["metric"] == "rmse"].copy()

# Create simulated final metrics (vintage performance - small improvement)
vintage_vs_final = pd.DataFrame({
    'wflow_id': vintage_rmse['wflow_id'].values,
    'vintage_rmse': vintage_rmse['value'].values,
    'final_rmse': vintage_rmse['value'].values - np.random.uniform(0.1, 0.3, len(vintage_rmse))
})

fig5 = backtest_results.plot_revision_impact(
    metric="rmse",
    vintage_vs_final_data=vintage_vs_final,
    show=False,
    figsize=(10, 6)
)
fig5.savefig("examples/viz_demo_revision_impact.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_revision_impact.png")

# ---------------------------------------------------------------------------
# Demonstrate filtering and customization
# ---------------------------------------------------------------------------
print("\n6. Custom Styling Example (filtered to top 2 workflows)")

# Filter to top 2 workflows
top_2_workflows = ranked.iloc[:2]["wflow_id"].tolist()

fig6 = backtest_results.plot_accuracy_over_time(
    metric="rmse",
    by_workflow=True,
    workflows=top_2_workflows,
    show=False,
    figsize=(12, 5),
    linewidth=3,
    markersize=8,
    marker="D"
)
fig6.savefig("examples/viz_demo_custom_style.png", dpi=150, bbox_inches="tight")
print("   ✓ Saved: examples/viz_demo_custom_style.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nBacktest Results:")
print(f"  - Workflows: {len(backtest_results.workflow_ids)}")
print(f"  - Vintages: {len(vintage_dates)}")
print(f"  - Metrics: rmse, mae, r_squared")

print(f"\nVisualizations Created:")
print(f"  1. Accuracy over time (by workflow)")
print(f"  2. Accuracy over time (aggregated with confidence)")
print(f"  3. Forecast horizon comparison")
print(f"  4. Vintage drift analysis (2 subplots)")
print(f"  5. Revision impact scatter plot")
print(f"  6. Custom styled plot (filtered workflows)")

print(f"\nBest Workflow by RMSE:")
best_wf = ranked.iloc[0]["wflow_id"]
best_rmse = ranked.iloc[0]["mean_rmse"]
best_std = ranked.iloc[0]["std_rmse"]
print(f"  {best_wf}: {best_rmse:.3f} ± {best_std:.3f}")

print("\n✓ Demo complete!")
print("\nAll plots saved to examples/ directory:")
print("  - viz_demo_accuracy_by_workflow.png")
print("  - viz_demo_accuracy_aggregated.png")
print("  - viz_demo_horizon_comparison.png")
print("  - viz_demo_vintage_drift.png")
print("  - viz_demo_revision_impact.png")
print("  - viz_demo_custom_style.png")

print("\nKey Features Demonstrated:")
print("  ✓ Method-based API (results.plot_*())")
print("  ✓ Workflow filtering")
print("  ✓ Custom styling via **kwargs")
print("  ✓ Aggregated vs per-workflow views")
print("  ✓ Confidence bands")
print("  ✓ Date formatting")
print("  ✓ Multiple subplot layouts")
