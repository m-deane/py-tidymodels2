"""
Revert Phase 3 recipe demonstrations to use per_group_prep=True
Now that workflow.py correctly handles outcome column for supervised steps
"""

import nbformat

# Load the notebook
nb_path = '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_recipes_grouped.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Find and fix code cells - restore per_group_prep=True
fixed_count = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Check if this is a Phase 3 demonstration cell
        if 'fit_nested(train_data, group_col=\'country\')' in cell.source and any(x in cell.source for x in [
            'fit_vif', 'fit_pvalue', 'fit_stability', 'fit_lofo',
            'fit_granger', 'fit_stepwise', 'fit_probe'
        ]):
            old_source = cell.source

            # Restore per_group_prep=True parameter
            new_source = old_source.replace(
                "fit_nested(train_data, group_col='country')",
                "fit_nested(train_data, group_col='country', per_group_prep=True)"
            )

            # Update the comment
            new_source = new_source.replace(
                "# Fit with nested models (one per group)",
                "# Fit with per-group preprocessing"
            )

            if new_source != old_source:
                cell.source = new_source
                fixed_count += 1

                # Find which step this is
                if 'fit_vif' in new_source:
                    step_name = 'VIF'
                elif 'fit_pvalue' in new_source:
                    step_name = 'P-value'
                elif 'fit_stability' in new_source:
                    step_name = 'Stability'
                elif 'fit_lofo' in new_source:
                    step_name = 'LOFO'
                elif 'fit_granger' in new_source:
                    step_name = 'Granger'
                elif 'fit_stepwise' in new_source:
                    step_name = 'Stepwise'
                elif 'fit_probe' in new_source:
                    step_name = 'Probe'
                else:
                    step_name = 'Unknown'

                print(f"✓ Restored cell {i+1}: {step_name} selection - per_group_prep=True")

# Remove the explanatory notes about global preprocessing
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Phase 3 Step' in cell.source:
        if '**Note:**' in cell.source and 'global preprocessing' in cell.source:
            # Remove the note paragraph
            parts = cell.source.split('\n\n')
            new_parts = [p for p in parts if not (p.startswith('**Note:**') and 'global preprocessing' in p)]
            cell.source = '\n\n'.join(new_parts)
            print(f"✓ Removed global preprocessing note from cell {i+1}")

# Save the updated notebook
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print(f"\n✅ Restored {fixed_count} code cells to use per_group_prep=True")
print(f"✅ Phase 3 steps now use per-group feature selection")
print(f"✅ Workflow.py has been updated to preserve outcome column for these steps")
