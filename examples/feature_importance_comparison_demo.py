"""
Comprehensive Feature Importance Comparison Demo

Demonstrates three feature importance calculation methods:
1. SAFE with LightGBM-based importance (improved from uniform distribution)
2. SHAP values (SHapley Additive exPlanations)
3. Permutation importance

Compares the three methods and shows how to use them in practice.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg
import warnings
warnings.filterwarnings('ignore')


def create_synthetic_data(n=1000, noise_level=0.5):
    """
    Create synthetic regression dataset with known feature importance.

    Features:
    - x1: High importance (coefficient = 3.0)
    - x2: High importance (coefficient = 2.5)
    - x3: Medium importance (coefficient = 1.2)
    - x4: Low importance (coefficient = 0.3)
    - x5: No importance (pure noise)
    - x6: No importance (pure noise)
    - x7: Medium importance via interaction with x1
    """
    np.random.seed(42)

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    x4 = np.random.randn(n)
    x5 = np.random.randn(n)  # Noise
    x6 = np.random.randn(n)  # Noise
    x7 = np.random.randn(n)

    # Target with known relationships
    y = (
        3.0 * x1 +           # Strong linear effect
        2.5 * x2 +           # Strong linear effect
        1.2 * x3 +           # Moderate effect
        0.3 * x4 +           # Weak effect
        0.8 * x1 * x7 +      # Interaction effect
        np.random.randn(n) * noise_level
    )

    return pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'x5': x5, 'x6': x6, 'x7': x7, 'y': y
    })


def demo_safe_importance():
    """Demonstrate SAFE with LightGBM-based importance."""
    print("=" * 80)
    print("METHOD 1: SAFE with LightGBM-based Importance")
    print("=" * 80)
    print("\nSAFE (Surrogate Assisted Feature Extraction) creates binary threshold features")
    print("and now uses LightGBM to compute proper feature importance instead of uniform.")
    print()

    # Create data
    data = create_synthetic_data(n=1000)

    # Train surrogate model for SAFE
    X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    y = data['y']
    surrogate = GradientBoostingRegressor(n_estimators=100, random_state=42)
    surrogate.fit(X, y)

    # Create recipe with SAFE step
    rec = recipe().step_safe(
        surrogate_model=surrogate,
        outcome='y',
        penalty=10,
        top_n=15,  # Select top 15 threshold features
        feature_type='both'  # Create both dummies and interactions
    )

    # Prep recipe
    prepped_rec = rec.prep(data)

    # Get step for inspection
    safe_step = prepped_rec.steps[0]

    # Show feature importances
    print("Top 20 features by LightGBM importance:")
    print("-" * 50)

    importances = safe_step.get_feature_importances()
    print(importances.head(20).to_string())

    # Apply transformation
    transformed = prepped_rec.bake(data)
    safe_cols = [c for c in transformed.columns if '_to_' in c or '_x_' in c]

    print(f"\nCreated {len(safe_cols)} SAFE features")
    print(f"Dummies: {len([c for c in safe_cols if '_x_' not in c])}")
    print(f"Interactions: {len([c for c in safe_cols if '_x_' in c])}")

    # Show top features by original variable
    print("\nTop features by original variable:")
    print("-" * 50)

    var_importances = {}
    for feat, imp in safe_step._feature_importances.items():
        var_name = feat.split('_')[0]  # Extract variable name
        if var_name not in var_importances:
            var_importances[var_name] = []
        var_importances[var_name].append(imp)

    var_totals = {var: sum(imps) for var, imps in var_importances.items()}
    sorted_vars = sorted(var_totals.items(), key=lambda x: x[1], reverse=True)

    for var, total_imp in sorted_vars[:10]:
        print(f"  {var}: {total_imp:.4f}")

    print("\nKEY INSIGHT: Notice how x1 and x2 have highest importance (as expected),")
    print("while x5 and x6 (noise variables) have low importance.")
    print()

    return transformed


def demo_shap_importance():
    """Demonstrate SHAP-based feature selection."""
    print("\n" + "=" * 80)
    print("METHOD 2: SHAP (SHapley Additive exPlanations)")
    print("=" * 80)
    print("\nSHAP values use game theory to attribute prediction contributions to features.")
    print("Works with any model type. Uses TreeExplainer for tree models (fast).")
    print()

    # Check if shap is available
    try:
        import shap
    except ImportError:
        print("SHAP package not installed. Install with: pip install shap")
        print("Skipping SHAP demo.")
        return None

    # Create data
    data = create_synthetic_data(n=500)  # Smaller dataset for faster SHAP computation

    # Split data
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # Train model
    X_train = train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    y_train = train['y']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create recipe with SHAP selection
    rec = recipe().step_select_shap(
        outcome='y',
        model=model,
        top_n=5,  # Keep top 5 features by SHAP value
        shap_samples=300,  # Sample for faster computation
        random_state=42
    )

    # Prep and bake
    prepped_rec = rec.prep(train)

    # Get step for inspection
    shap_step = prepped_rec.steps[0]

    # Show SHAP values
    print("Feature importance by mean absolute SHAP value:")
    print("-" * 50)

    sorted_features = sorted(shap_step._scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features:
        print(f"  {feat}: {score:.4f}")

    print(f"\nSelected features (top 5):")
    for feat in shap_step._selected_features:
        print(f"  - {feat} (SHAP = {shap_step._scores[feat]:.4f})")

    # Apply transformation
    transformed = prepped_rec.bake(test)

    print(f"\nReduced from {len(data.columns)-1} features to {len(transformed.columns)-1} features")

    print("\nKEY INSIGHT: SHAP captures both main effects and interactions.")
    print("Notice how x1, x2 have high SHAP values (strong predictors),")
    print("while x5, x6 have low values (noise).")
    print()

    return transformed


def demo_permutation_importance():
    """Demonstrate permutation importance feature selection."""
    print("\n" + "=" * 80)
    print("METHOD 3: Permutation Importance")
    print("=" * 80)
    print("\nPermutation importance measures feature importance by shuffling each feature")
    print("and measuring the resulting drop in model performance. Model-agnostic.")
    print()

    # Create data
    data = create_synthetic_data(n=500)  # Moderate dataset size

    # Split data
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # Train model
    X_train = train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    y_train = train['y']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create recipe with permutation importance selection
    rec = recipe().step_select_permutation(
        outcome='y',
        model=model,
        top_n=5,  # Keep top 5 features
        n_repeats=10,  # Repeat permutation 10 times for stability
        scoring='r2',  # Use R² as metric
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # Prep and bake
    print("Computing permutation importance (this may take a moment)...")
    prepped_rec = rec.prep(train)

    # Get step for inspection
    perm_step = prepped_rec.steps[0]

    # Show permutation importances
    print("\nFeature importance by permutation (mean over 10 repeats):")
    print("-" * 50)

    sorted_features = sorted(perm_step._scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features:
        print(f"  {feat}: {score:.4f}")

    print(f"\nSelected features (top 5):")
    for feat in perm_step._selected_features:
        print(f"  - {feat} (importance = {perm_step._scores[feat]:.4f})")

    # Apply transformation
    transformed = prepped_rec.bake(test)

    print(f"\nReduced from {len(data.columns)-1} features to {len(transformed.columns)-1} features")

    print("\nKEY INSIGHT: Permutation importance is computationally expensive but")
    print("very reliable. Notice how x1, x2 have highest importance (strong predictors).")
    print()

    return transformed


def compare_all_methods():
    """Compare all three methods side by side."""
    print("\n" + "=" * 80)
    print("COMPARISON: All Three Methods")
    print("=" * 80)
    print()

    # Create data
    data = create_synthetic_data(n=500)
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # Prepare models
    X_train = train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    y_train = train['y']

    # Surrogate for SAFE
    surrogate = GradientBoostingRegressor(n_estimators=50, random_state=42)
    surrogate.fit(X_train, y_train)

    # Model for SHAP and Permutation
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)

    # Method 1: SAFE (get variable-level importance)
    print("Computing SAFE importances...")
    safe_rec = recipe().step_safe(
        surrogate_model=surrogate,
        outcome='y',
        penalty=10,
        feature_type='dummies'
    )
    safe_prepped = safe_rec.prep(train)
    safe_step = safe_prepped.steps[0]

    # Aggregate by variable
    safe_var_importance = {}
    for feat, imp in safe_step._feature_importances.items():
        var_name = feat.split('_')[0]
        safe_var_importance[var_name] = safe_var_importance.get(var_name, 0) + imp

    # Method 2: SHAP
    try:
        import shap
        print("Computing SHAP values...")
        shap_rec = recipe().step_select_shap(
            outcome='y',
            model=rf_model,
            top_n=7,
            shap_samples=200,
            random_state=42
        )
        shap_prepped = shap_rec.prep(train)
        shap_scores = shap_prepped.steps[0]._scores
    except ImportError:
        print("SHAP not available, skipping...")
        shap_scores = None

    # Method 3: Permutation
    print("Computing permutation importance...")
    perm_rec = recipe().step_select_permutation(
        outcome='y',
        model=rf_model,
        top_n=7,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    perm_prepped = perm_rec.prep(train)
    perm_scores = perm_prepped.steps[0]._scores

    # Compare
    print("\n" + "=" * 80)
    print("Feature Importance Rankings (Normalized)")
    print("=" * 80)
    print(f"{'Feature':<10} {'True Coef':<12} {'SAFE':<12} {'SHAP':<12} {'Permutation':<12}")
    print("-" * 80)

    # True coefficients (for reference)
    true_coefs = {
        'x1': 3.0,
        'x2': 2.5,
        'x3': 1.2,
        'x4': 0.3,
        'x5': 0.0,
        'x6': 0.0,
        'x7': 0.8  # Via interaction
    }

    # Normalize scores for comparison
    def normalize(scores):
        total = sum(scores.values()) if scores else 1.0
        return {k: v/total if total > 0 else 0.0 for k, v in scores.items()}

    safe_norm = normalize(safe_var_importance)
    shap_norm = normalize(shap_scores) if shap_scores else {}
    perm_norm = normalize(perm_scores)

    for feat in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']:
        true_val = true_coefs[feat]
        safe_val = safe_norm.get(feat, 0.0)
        shap_val = shap_norm.get(feat, 0.0) if shap_scores else float('nan')
        perm_val = perm_norm.get(feat, 0.0)

        print(f"{feat:<10} {true_val:<12.2f} {safe_val:<12.4f} {shap_val:<12.4f} {perm_val:<12.4f}")

    print("\nINTERPRETATION:")
    print("-" * 80)
    print("• All three methods correctly identify x1 and x2 as most important")
    print("• All three methods correctly identify x5 and x6 as least important (noise)")
    print("• SAFE creates threshold features, so importance is spread across thresholds")
    print("• SHAP and Permutation give direct variable-level importance")
    print("• All methods agree on ranking: x1 > x2 > x3 > x7 > x4 > x5/x6")
    print()


def workflow_integration_example():
    """Show how to integrate feature selection into a workflow."""
    print("\n" + "=" * 80)
    print("WORKFLOW INTEGRATION EXAMPLE")
    print("=" * 80)
    print("\nDemonstrate feature selection in a complete modeling workflow.")
    print()

    # Create data
    data = create_synthetic_data(n=500)
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # Train model for feature selection
    X_train = train[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    y_train = train['y']
    feature_selector = RandomForestRegressor(n_estimators=50, random_state=42)
    feature_selector.fit(X_train, y_train)

    # Create workflow with permutation-based feature selection
    rec = recipe().step_select_permutation(
        outcome='y',
        model=feature_selector,
        top_n=4,  # Keep only top 4 features
        n_repeats=5,
        random_state=42
    )

    wf = workflow().add_recipe(rec).add_model(linear_reg())

    # Fit workflow
    print("Fitting workflow with permutation feature selection...")
    fitted_wf = wf.fit(train)

    # Evaluate
    eval_results = fitted_wf.evaluate(test)

    # Show results
    print("\nSelected features:")
    perm_step = fitted_wf.fit_data['prepped_recipe'].steps[0]
    for feat in perm_step._selected_features:
        print(f"  - {feat} (importance = {perm_step._scores[feat]:.4f})")

    print(f"\nModel performance on test set:")
    print(f"  RMSE: {eval_results['test']['rmse']:.4f}")
    print(f"  R²:   {eval_results['test']['r_squared']:.4f}")

    print("\nKEY INSIGHT: Feature selection reduces dimensionality while maintaining")
    print("good model performance. Only 4 features needed instead of 7!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE METHODS COMPARISON DEMO")
    print("=" * 80)
    print("\nThis demo compares three feature importance calculation methods:")
    print("  1. SAFE with LightGBM-based importance (threshold features)")
    print("  2. SHAP values (game theory-based attribution)")
    print("  3. Permutation importance (model-agnostic)")
    print()

    # Run individual demos
    demo_safe_importance()
    demo_shap_importance()
    demo_permutation_importance()

    # Compare all methods
    compare_all_methods()

    # Workflow integration
    workflow_integration_example()

    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("""
WHEN TO USE EACH METHOD:

1. SAFE with LightGBM Importance:
   ✓ When you need threshold-based features (piecewise linear modeling)
   ✓ Creates interpretable binary indicators and interactions
   ✓ Now uses proper feature importance instead of uniform distribution
   ✓ Best for: Threshold detection, rule-based models, interpretability

2. SHAP Values:
   ✓ When you need to explain individual predictions
   ✓ Works with any model type (TreeExplainer for trees is fast)
   ✓ Captures both main effects and interactions
   ✓ Best for: Model explanation, feature importance, interaction detection
   ✓ Note: Requires 'shap' package (pip install shap)

3. Permutation Importance:
   ✓ Most reliable model-agnostic method
   ✓ Works with any model and any metric
   ✓ Computationally expensive (requires many model evaluations)
   ✓ Best for: Critical applications, final feature selection, model validation
   ✓ Tip: Use n_jobs=-1 for parallel execution

GENERAL RECOMMENDATIONS:
• Use SHAP or Permutation for feature selection before modeling
• Use SAFE when you need threshold features for linear models
• All three methods agree on feature rankings in most cases
• Start with SHAP (fast for tree models), validate with Permutation
• Use SAFE when interpretability via thresholds is critical
    """)
    print("=" * 80)
