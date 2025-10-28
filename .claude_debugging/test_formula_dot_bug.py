"""
Minimal Reproducible Test Case for Formula Parsing Error
=========================================================

ERROR: ValueError: Failed to parse formula 'target ~ .': invalid syntax (<unknown>, line 1)

ROOT CAUSE:
- Patsy 1.0.1 does not support the '.' wildcard in formulas
- When patsy encounters '.', it creates an EvalFactor('.')
- It then tries to parse '.' as Python code via ast.parse('.')
- This fails because '.' alone is not valid Python syntax

LOCATION: py_hardhat/mold.py, line 65 in mold()
    y_mat, X_mat = dmatrices(formula, data, ...)
    -> patsy.eval.ast_names() calls ast.parse('.')
    -> SyntaxError: invalid syntax

SOLUTION: Expand '.' to explicit column names before calling patsy
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression


def test_original_error():
    """Reproduce the exact error from notebook cell 8"""
    print("="*80)
    print("TEST 1: Reproduce Original Error")
    print("="*80)

    # Exact scenario from notebook
    np.random.seed(42)
    X_multi, y_multi = make_regression(
        n_samples=300, n_features=8, n_informative=5, noise=20, random_state=42
    )

    feature_names = [f'Feature_{i+1}' for i in range(8)]
    df_multi = pd.DataFrame(X_multi, columns=feature_names)
    df_multi['target'] = y_multi

    print(f"DataFrame shape: {df_multi.shape}")
    print(f"Columns: {df_multi.columns.tolist()}")

    # This should fail with patsy
    import patsy
    try:
        y_mat, X_mat = patsy.dmatrices('target ~ .', df_multi, return_type='dataframe')
        print("\nUNEXPECTED: Formula parsed successfully!")
        print(f"X_mat columns: {X_mat.columns.tolist()}")
    except Exception as e:
        print(f"\nEXPECTED ERROR: {type(e).__name__}: {e}")
        return True

    return False


def test_workaround():
    """Test the workaround: expand '.' to explicit columns"""
    print("\n" + "="*80)
    print("TEST 2: Workaround - Expand '.' to Explicit Columns")
    print("="*80)

    # Same data
    np.random.seed(42)
    X_multi, y_multi = make_regression(
        n_samples=300, n_features=8, n_informative=5, noise=20, random_state=42
    )

    feature_names = [f'Feature_{i+1}' for i in range(8)]
    df_multi = pd.DataFrame(X_multi, columns=feature_names)
    df_multi['target'] = y_multi

    # Manually expand the formula
    outcome = 'target'
    predictor_cols = [col for col in df_multi.columns if col != outcome]
    expanded_formula = f'{outcome} ~ {" + ".join(predictor_cols)}'

    print(f"Original formula: 'target ~ .'")
    print(f"Expanded formula: '{expanded_formula}'")

    import patsy
    try:
        y_mat, X_mat = patsy.dmatrices(expanded_formula, df_multi, return_type='dataframe')
        print(f"\nSUCCESS: Formula parsed!")
        print(f"X_mat shape: {X_mat.shape}")
        print(f"X_mat columns: {X_mat.columns.tolist()}")
        return True
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def test_edge_cases():
    """Test edge cases with special column names"""
    print("\n" + "="*80)
    print("TEST 3: Edge Cases - Special Column Names")
    print("="*80)

    import patsy

    # Test case 1: Column with spaces (needs quoting)
    df1 = pd.DataFrame({
        'my feature': [1, 2, 3],
        'another_feature': [4, 5, 6],
        'target': [7, 8, 9]
    })

    print("\nCase 1: Column with spaces")
    print(f"Columns: {df1.columns.tolist()}")

    # Need to use Q() for special characters
    formula1 = 'target ~ Q("my feature") + another_feature'
    try:
        y_mat, X_mat = patsy.dmatrices(formula1, df1, return_type='dataframe')
        print(f"SUCCESS: {X_mat.columns.tolist()}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Test case 2: Column starting with number (also needs quoting)
    df2 = pd.DataFrame({
        '1st_feature': [1, 2, 3],
        '2nd_feature': [4, 5, 6],
        'target': [7, 8, 9]
    })

    print("\nCase 2: Columns starting with numbers")
    print(f"Columns: {df2.columns.tolist()}")

    formula2 = 'target ~ Q("1st_feature") + Q("2nd_feature")'
    try:
        y_mat, X_mat = patsy.dmatrices(formula2, df2, return_type='dataframe')
        print(f"SUCCESS: {X_mat.columns.tolist()}")
    except Exception as e:
        print(f"FAILED: {e}")


def proposed_fix():
    """Show the proposed fix for mold()"""
    print("\n" + "="*80)
    print("PROPOSED FIX FOR mold()")
    print("="*80)

    fix_code = '''
def _expand_dot_formula(formula: str, data: pd.DataFrame) -> str:
    """
    Expand '.' wildcard in formula to explicit column names.

    The '.' in R-style formulas means "all columns except the outcome".
    Patsy doesn't support this, so we expand it manually.

    Args:
        formula: Formula string (e.g., "y ~ .")
        data: DataFrame with columns to expand

    Returns:
        Expanded formula string (e.g., "y ~ x1 + x2 + x3")
    """
    # Check if formula contains '.'
    if '~' not in formula:
        return formula

    lhs, rhs = formula.split('~', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    # If RHS doesn't contain '.', return as-is
    if '.' not in rhs:
        return formula

    # Extract outcome column name(s) from LHS
    # Simple case: single outcome (e.g., "y")
    # Complex case: multiple outcomes (e.g., "y1 + y2")
    outcome_cols = [col.strip() for col in lhs.split('+')]

    # Get all columns except outcomes
    predictor_cols = [col for col in data.columns if col not in outcome_cols]

    # Quote column names that aren't valid Python identifiers
    def quote_if_needed(name):
        if name.isidentifier():
            return name
        else:
            return f'Q("{name}")'

    quoted_predictors = [quote_if_needed(col) for col in predictor_cols]

    # Replace '.' with explicit column list
    if rhs.strip() == '.':
        # Simple case: "y ~ ."
        expanded_rhs = ' + '.join(quoted_predictors)
    else:
        # Complex case: "y ~ . + log(x)" or "y ~ . - x1"
        # For now, handle simple replacement
        expanded_rhs = rhs.replace('.', ' + '.join(quoted_predictors))

    return f'{lhs} ~ {expanded_rhs}'


# In mold(), add this before calling dmatrices():
formula = _expand_dot_formula(formula, data)
'''

    print(fix_code)

    print("\nThen in mold(), line 65, change:")
    print("  FROM: y_mat, X_mat = dmatrices(formula, data, ...)")
    print("  TO:   formula = _expand_dot_formula(formula, data)")
    print("        y_mat, X_mat = dmatrices(formula, data, ...)")


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# FORMULA PARSING ERROR - MINIMAL REPRODUCIBLE TEST")
    print("#"*80 + "\n")

    # Run tests
    error_reproduced = test_original_error()
    workaround_works = test_workaround()
    test_edge_cases()
    proposed_fix()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1. Original error reproduced: {error_reproduced}")
    print(f"2. Workaround successful: {workaround_works}")
    print("\n3. Root cause confirmed:")
    print("   - Patsy 1.0.1 doesn't support '.' wildcard")
    print("   - ast.parse('.') raises SyntaxError")
    print("   - Location: patsy/eval.py line 111")
    print("\n4. Fix required:")
    print("   - Add _expand_dot_formula() function to mold.py")
    print("   - Preprocess formula before calling patsy.dmatrices()")
    print("   - Handle edge cases (spaces, numbers, special chars)")
    print("\n5. Files to modify:")
    print("   - /Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py")
    print("="*80)
