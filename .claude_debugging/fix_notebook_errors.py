"""
Fix all errors in forecasting_recipes_grouped.ipynb

This script fixes 13 errors across multiple cells:
1. Cell 32: step_select_corr incorrect usage
2. Cell 49-50: Missing step_naomit for lag/diff
3. Cells 57-59, 81-87: Outcome column handling (already fixed in workflow)
4. Cell 69: sqrt NaN handling
5. Cell 76: Wrong outcome column name
"""

import json
import re

def fix_notebook(input_path, output_path):
    with open(input_path, 'r') as f:
        nb = json.load(f)

    # Fix Cell 32: step_select_corr
    cell32 = nb['cells'][32]
    if 'step_select_corr' in ''.join(cell32['source']):
        cell32['source'] = [
            "# Recipe with correlation filter\n",
            "rec_corr = (\n",
            "    recipe()\n",
            "    .step_select_corr(outcome='refinery_kbd', threshold=0.4, method='multicollinearity')  # Remove correlated features\n",
            "    .step_normalize(all_numeric_predictors())\n",
            "\n",
            ")\n",
            "\n",
            "# Workflow\n",
            "wf_corr = (\n",
            "    workflow()\n",
            "    .add_recipe(rec_corr)\n",
            "    .add_model(linear_reg().set_engine(\"sklearn\"))\n",
            "    .add_model_name(\"corr\")\n",
            "    .add_model_group_name(\"linear_reg_correlation_filter\")\n",
            ")\n",
            "\n",
            "# Fit and evaluate\n",
            "fit_corr = wf_corr.fit_nested(train_data, group_col='country', per_group_prep=True)\n",
            "fit_corr = fit_corr.evaluate(test_data)\n",
            "\n",
            "outputs_corr, coefs_corr, stats_corr = fit_corr.extract_outputs()\n",
            "\n",
            "display(outputs_corr)\n",
            "display(coefs_corr)\n",
            "display(stats_corr)\n",
            "\n",
            "# Plot forecast\n",
            "fig = plot_forecast(fit_corr, title=\"With Correlation Filtering (threshold=0.4)\")\n",
            "fig.show()\n",
            "\n",
            "# Extract preprocessed training data\n",
            "processed_train = fit_corr.extract_preprocessed_data(train_data, split='train')\n",
            "processed_test = fit_corr.extract_preprocessed_data(test_data, split='test')    \n",
            "\n",
            "display(processed_train.tail(10))\n",
            "display(processed_test.head(10))"
        ]
        # Clear outputs
        cell32['outputs'] = []
        cell32['execution_count'] = None
        print("✓ Fixed Cell 32: step_select_corr usage")

    # Fix Cell 49: Uncomment step_naomit for lag features
    cell49 = nb['cells'][49]
    if 'step_lag' in ''.join(cell49['source']):
        source = ''.join(cell49['source'])
        # Uncomment step_naomit
        source = source.replace('# .step_naomit()', '.step_naomit()')
        cell49['source'] = [source]
        cell49['outputs'] = []
        cell49['execution_count'] = None
        print("✓ Fixed Cell 49: Uncommented step_naomit for lag features")

    # Fix Cell 50: Uncomment step_naomit for diff features
    cell50 = nb['cells'][50]
    if 'step_diff' in ''.join(cell50['source']):
        source = ''.join(cell50['source'])
        # Uncomment step_naomit
        source = source.replace('# .step_naomit()', '.step_naomit()')
        cell50['source'] = [source]
        cell50['outputs'] = []
        cell50['execution_count'] = None
        print("✓ Fixed Cell 50: Uncommented step_naomit for diff features")

    # Fix Cell 57-59: Add note that outcome column handling is fixed in workflow
    # These should now work with the workflow fix
    for cell_idx in [57, 58, 59]:
        cell = nb['cells'][cell_idx]
        cell['outputs'] = []
        cell['execution_count'] = None
    print("✓ Cleared Cell 57-59: Supervised filter steps (fixed in workflow)")

    # Fix Cell 69: Add step_naomit before sqrt
    cell69 = nb['cells'][69]
    if 'step_sqrt' in ''.join(cell69['source']):
        source = ''.join(cell69['source'])
        # Add step_naomit before the problematic transformations, and remove inplace
        source = source.replace('.step_sqrt(all_numeric_predictors(), inplace=True)',
                                '.step_naomit()  # Remove rows with NaN before sqrt\\n    .step_sqrt(all_numeric_predictors())')
        cell69['source'] = [source]
        cell69['outputs'] = []
        cell69['execution_count'] = None
        print("✓ Fixed Cell 69: Added step_naomit before sqrt and removed inplace")

    # Fix Cell 76: Change outcome from "target" to "refinery_kbd"
    cell76 = nb['cells'][76]
    if 'step_pls' in ''.join(cell76['source']):
        source = ''.join(cell76['source'])
        source = source.replace('outcome="target"', 'outcome="refinery_kbd"')
        cell76['source'] = [source]
        cell76['outputs'] = []
        cell76['execution_count'] = None
        print("✓ Fixed Cell 76: Changed outcome from 'target' to 'refinery_kbd'")

    # Clear outputs for cells 81-87 (supervised selection - should work with workflow fix)
    for cell_idx in [81, 83, 85, 87]:
        cell = nb['cells'][cell_idx]
        cell['outputs'] = []
        cell['execution_count'] = None
    print("✓ Cleared Cells 81-87: Supervised selection steps (fixed in workflow)")

    # Save fixed notebook
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\\n✓ Fixed notebook saved to: {output_path}")
    return True

if __name__ == "__main__":
    input_path = "_md/forecasting_recipes_grouped.ipynb"
    output_path = "_md/forecasting_recipes_grouped.ipynb"

    fix_notebook(input_path, output_path)
