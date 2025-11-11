"""
Fix Phase 3 recipe demonstrations by removing per_group_prep=True
These steps require the outcome column during prep, which is not available with per_group_prep
"""

import nbformat

# Load the notebook
nb_path = '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_recipes_grouped.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Find and fix code cells with per_group_prep=True in the Phase 3 section
fixed_count = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Check if this is a Phase 3 demonstration cell
        if 'per_group_prep=True' in cell.source and any(x in cell.source for x in [
            'fit_vif', 'fit_pvalue', 'fit_stability', 'fit_lofo',
            'fit_granger', 'fit_stepwise', 'fit_probe'
        ]):
            # Replace per_group_prep=True with nothing (use default False)
            # Also update the comment
            old_source = cell.source

            # Remove per_group_prep=True parameter and adjust formatting
            new_source = old_source.replace(
                "fit_nested(train_data, group_col='country', per_group_prep=True)",
                "fit_nested(train_data, group_col='country')"
            )

            # Update the comment
            new_source = new_source.replace(
                "# Fit with per-group preprocessing",
                "# Fit with nested models (one per group)"
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

                print(f"✓ Fixed cell {i+1}: {step_name} selection")

# Update markdown cells to explain why per_group_prep is not used
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Phase 3 Step' in cell.source:
        # Add note about why per_group_prep is not used
        if '**Note:**' not in cell.source:
            # Add note after the description
            parts = cell.source.split('\n\n')
            if len(parts) >= 2:
                # Insert note after first paragraph
                note = "**Note:** These selection steps require the outcome variable during preprocessing, so they use global preprocessing (not per-group)."
                parts.insert(2, note)
                cell.source = '\n\n'.join(parts)
                print(f"✓ Updated markdown cell {i+1}")

# Save the updated notebook
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print(f"\n✅ Fixed {fixed_count} code cells")
print(f"✅ Removed per_group_prep=True from supervised selection steps")
print(f"✅ These steps now use global preprocessing (outcome column required)")
