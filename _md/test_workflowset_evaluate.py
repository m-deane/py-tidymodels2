"""
Test to verify WorkflowSetNestedResults.evaluate() method works
"""
import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg, rand_forest
from py_rsample import initial_split, training, testing

# Create simple test data
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'country': np.repeat(['A', 'B'], n // 2),
    'x1': np.random.randn(n) * 10 + 50,
    'x2': np.random.randn(n) * 5 + 20,
    'y': np.random.randn(n) * 100 + 500
})

# Split train/test
split = initial_split(data, prop=0.75, seed=123)
train_data = training(split)
test_data = testing(split)

print("="*60)
print("Testing WorkflowSetNestedResults.evaluate()")
print("="*60)
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print()

# Create WorkflowSet
try:
    wf_set = WorkflowSet.from_cross(
        preproc=["y ~ x1 + x2"],
        models=[
            linear_reg().set_engine("sklearn"),
            rand_forest(trees=50).set_mode('regression').set_engine("sklearn")
        ]
    )
    print(f"✓ WorkflowSet created with {len(wf_set.workflows)} workflows")
    print()

    # Fit nested
    print("Fitting nested workflows...")
    results = wf_set.fit_nested(train_data, group_col='country', per_group_prep=False)
    print("✓ fit_nested completed")
    print()

    # Check metrics before evaluation (should only have train split)
    print("Before evaluation:")
    metrics_before = results.collect_metrics(split='all')
    splits_before = metrics_before['split'].unique()
    print(f"  Splits available: {sorted(splits_before)}")
    print(f"  Metrics shape: {metrics_before.shape}")
    print()

    # Evaluate on test data
    print("Evaluating on test data...")
    results = results.evaluate(test_data)
    print("✓ evaluate() completed")
    print()

    # Check metrics after evaluation (should have both train and test)
    print("After evaluation:")
    metrics_after = results.collect_metrics(split='all')
    splits_after = metrics_after['split'].unique()
    print(f"  Splits available: {sorted(splits_after)}")
    print(f"  Metrics shape: {metrics_after.shape}")
    print()

    # Show test metrics
    test_metrics = results.collect_metrics(split='test', by_group=False)
    print("Test metrics (averaged across groups):")
    print(test_metrics[['wflow_id', 'metric', 'mean', 'std', 'model']])
    print()

    # Verify test split exists
    if 'test' in splits_after:
        print("✅ SUCCESS! evaluate() method working correctly")
        print("   Test metrics are now available")
    else:
        print("❌ FAILED: Test split not found after evaluation")

except Exception as e:
    print("❌ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
