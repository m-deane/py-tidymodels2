"""
Debug test to see what X_train actually contains after supervised feature selection.
"""

import pandas as pd
import numpy as np
from py_workflowsets import WorkflowSet
from py_parsnip import linear_reg
from py_recipes import recipe


def test_debug_x_train_contents():
    """Check what's actually in X_train after feature selection."""
    # Create data with correlated features
    np.random.seed(42)
    n_per_group = 100

    # USA: x1 and x2 highly correlated (0.95)
    usa_data = pd.DataFrame({
        'country': ['USA'] * n_per_group,
        'x1': np.random.randn(n_per_group),
    })
    usa_data['x2'] = usa_data['x1'] * 0.95 + np.random.randn(n_per_group) * 0.1  # Highly correlated with x1
    usa_data['x3'] = np.random.randn(n_per_group)  # Independent
    usa_data['y'] = usa_data['x1'] * 2 + usa_data['x3'] * 1.5 + np.random.randn(n_per_group) * 0.5

    # Germany: x1 and x3 highly correlated (0.95)
    germany_data = pd.DataFrame({
        'country': ['Germany'] * n_per_group,
        'x1': np.random.randn(n_per_group),
    })
    germany_data['x3'] = germany_data['x1'] * 0.95 + np.random.randn(n_per_group) * 0.1  # Highly correlated with x1
    germany_data['x2'] = np.random.randn(n_per_group)  # Independent
    germany_data['y'] = germany_data['x1'] * 2 + germany_data['x2'] * 1.5 + np.random.randn(n_per_group) * 0.5

    data = pd.concat([usa_data, germany_data], ignore_index=True)

    # Create workflow with supervised feature selection
    rec = recipe().step_select_corr(outcome='y', threshold=0.9, method='multicollinearity')

    wf_set = WorkflowSet.from_cross(
        preproc=[rec],
        models=[linear_reg()]
    )

    # Fit with per-group preprocessing
    results = wf_set.fit_nested(data, group_col='country', per_group_prep=True)

    # Get the nested fit
    nested_fit = results.results[0]['nested_fit']

    # Check X_train for each group
    print("\n" + "="*60)
    print("DEBUG: X_train contents after feature selection")
    print("="*60)

    for group_name, wf_fit in nested_fit.group_fits.items():
        model_fit = wf_fit.fit
        X_train = model_fit.fit_data.get('X_train')

        print(f"\n{group_name}:")
        print(f"  X_train type: {type(X_train)}")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_train columns: {list(X_train.columns)}")

        # Check what the recipe kept
        if wf_fit.pre is not None and hasattr(wf_fit.pre, 'steps'):
            print(f"  Recipe has {len(wf_fit.pre.steps)} steps")
            for i, step in enumerate(wf_fit.pre.steps):
                print(f"    Step {i}: {type(step).__name__}")
                if hasattr(step, 'columns_to_keep'):
                    print(f"      columns_to_keep: {step.columns_to_keep}")

    # Now check extract_formulas output
    formulas_df = results.extract_formulas()
    print("\n" + "="*60)
    print("extract_formulas() output:")
    print("="*60)
    print(formulas_df[['wflow_id', 'group', 'formula', 'n_features']].to_string(index=False))

    # What SHOULD happen:
    # USA: x2 highly correlated with x1 → should be removed → X_train should have [x1, x3]
    # Germany: x3 highly correlated with x1 → should be removed → X_train should have [x1, x2]

    print("\n" + "="*60)
    print("Expected:")
    print("="*60)
    print("USA should have: x1, x3 (x2 removed due to correlation)")
    print("Germany should have: x1, x2 (x3 removed due to correlation)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

    test_debug_x_train_contents()
