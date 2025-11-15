"""
Quick test to verify conformal prediction fix with MAPIE 0.9.2
"""
import pandas as pd
import numpy as np
from py_parsnip import linear_reg

# Create simple test data
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(n)*0.5

train_data = pd.DataFrame({
    'x1': X[:150, 0],
    'x2': X[:150, 1],
    'y': y[:150]
})

test_data = pd.DataFrame({
    'x1': X[150:, 0],
    'x2': X[150:, 1],
    'y': y[150:]
})

print("="*80)
print("TESTING CONFORMAL PREDICTION FIX")
print("="*80)

# Test 1: Split conformal
print("\n1. Testing split conformal...")
try:
    spec = linear_reg()
    fit = spec.fit(train_data, 'y ~ x1 + x2')
    conformal_preds = fit.conformal_predict(test_data, alpha=0.05, method='split')
    print(f"✅ Split conformal: Generated {len(conformal_preds)} predictions")
    print(f"   Columns: {list(conformal_preds.columns)}")
    print(f"   Sample interval width: {conformal_preds['.pred_upper'].iloc[0] - conformal_preds['.pred_lower'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ Split conformal FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 2: CV+ conformal
print("\n2. Testing cv+ conformal...")
try:
    conformal_preds = fit.conformal_predict(test_data, alpha=0.05, method='cv+', cv=5)
    print(f"✅ CV+ conformal: Generated {len(conformal_preds)} predictions")
    print(f"   Sample interval width: {conformal_preds['.pred_upper'].iloc[0] - conformal_preds['.pred_lower'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ CV+ conformal FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 3: Jackknife+ conformal
print("\n3. Testing jackknife+ conformal...")
try:
    conformal_preds = fit.conformal_predict(test_data, alpha=0.05, method='jackknife+')
    print(f"✅ Jackknife+ conformal: Generated {len(conformal_preds)} predictions")
    print(f"   Sample interval width: {conformal_preds['.pred_upper'].iloc[0] - conformal_preds['.pred_lower'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ Jackknife+ conformal FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 4: Auto method selection
print("\n4. Testing auto method selection...")
try:
    conformal_preds = fit.conformal_predict(test_data, alpha=0.05, method='auto')
    method_used = conformal_preds['.conf_method'].iloc[0]
    print(f"✅ Auto conformal: Generated {len(conformal_preds)} predictions")
    print(f"   Method selected: {method_used}")
    print(f"   Sample interval width: {conformal_preds['.pred_upper'].iloc[0] - conformal_preds['.pred_lower'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ Auto conformal FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 5: Multiple alphas
print("\n5. Testing multiple confidence levels...")
try:
    conformal_preds = fit.conformal_predict(test_data, alpha=[0.05, 0.1, 0.2], method='split')
    print(f"✅ Multiple alphas: Generated {len(conformal_preds)} predictions")
    print(f"   Columns: {[c for c in conformal_preds.columns if 'pred' in c]}")
    print(f"   95% interval width: {conformal_preds['.pred_upper_95'].iloc[0] - conformal_preds['.pred_lower_95'].iloc[0]:.3f}")
    print(f"   80% interval width: {conformal_preds['.pred_upper_80'].iloc[0] - conformal_preds['.pred_lower_80'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ Multiple alphas FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 6: Cached wrapper (reuse with different alpha)
print("\n6. Testing cached wrapper reuse...")
try:
    # First call caches wrapper
    conformal_preds1 = fit.conformal_predict(test_data, alpha=0.05, method='split')
    # Second call should reuse wrapper
    conformal_preds2 = fit.conformal_predict(test_data, alpha=0.1, method='split')
    print(f"✅ Cached wrapper: Both predictions generated successfully")
    print(f"   First (α=0.05) interval width: {conformal_preds1['.pred_upper'].iloc[0] - conformal_preds1['.pred_lower'].iloc[0]:.3f}")
    print(f"   Second (α=0.10) interval width: {conformal_preds2['.pred_upper'].iloc[0] - conformal_preds2['.pred_lower'].iloc[0]:.3f}")
except Exception as e:
    print(f"❌ Cached wrapper FAILED: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("BASIC CONFORMAL TESTS COMPLETE")
print("="*80)
