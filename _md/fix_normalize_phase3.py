"""
Fix normalization in Phase 3 demonstrations to exclude outcome column

The issue: all_numeric_predictors() uses hardcoded outcome names {'y', 'target', 'outcome'}
but doesn't know about 'refinery_kbd', so it normalizes the outcome when it shouldn't.

Solution: Use difference(all_numeric(), one_of('refinery_kbd')) to explicitly exclude outcome
"""

import nbformat

# Load the notebook
nb_path = '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/_md/forecasting_recipes_grouped.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Step 1: Update selector imports at the top of notebook
import_updated = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'from py_recipes.selectors import (' in cell.source:
        # Check if difference and one_of are already imported
        if 'difference' not in cell.source or 'one_of' not in cell.source:
            old_import = cell.source

            # Add difference and one_of to imports
            new_import = old_import.replace(
                "    contains, starts_with, ends_with, matches  # ← These are now available",
                "    contains, starts_with, ends_with, matches,\n    difference, one_of  # For excluding specific columns"
            )

            if new_import != old_import:
                cell.source = new_import
                import_updated = True
                print(f"✓ Updated imports in cell {i+1}")
                break

# Step 2: Fix normalization in Phase 3 cells
fixed_count = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Check if this is a Phase 3 demonstration cell with normalization
        if 'step_normalize(all_numeric_predictors())' in cell.source and any(x in cell.source for x in [
            'rec_vif', 'rec_pvalue', 'rec_stability', 'rec_lofo',
            'rec_granger', 'rec_stepwise', 'rec_probe'
        ]):
            old_source = cell.source

            # Use difference selector to explicitly exclude outcome
            new_source = old_source.replace(
                "    .step_normalize(all_numeric_predictors())  # Normalize first\n",
                "    .step_normalize(difference(all_numeric(), one_of('refinery_kbd')))  # Normalize predictors (exclude outcome)\n"
            )

            if new_source != old_source:
                cell.source = new_source
                fixed_count += 1

                # Find which step this is
                if 'rec_vif' in new_source:
                    step_name = 'VIF'
                elif 'rec_pvalue' in new_source:
                    step_name = 'P-value'
                elif 'rec_stability' in new_source:
                    step_name = 'Stability'
                elif 'rec_lofo' in new_source:
                    step_name = 'LOFO'
                elif 'rec_granger' in new_source:
                    step_name = 'Granger'
                elif 'rec_stepwise' in new_source:
                    step_name = 'Stepwise'
                elif 'rec_probe' in new_source:
                    step_name = 'Probe'
                else:
                    step_name = 'Unknown'

                print(f"✓ Fixed cell {i+1}: {step_name} - normalize excludes outcome")

# Save the updated notebook
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print(f"\n{'='*60}")
print(f"✅ Updated imports: {'Yes' if import_updated else 'Already up to date'}")
print(f"✅ Fixed {fixed_count} normalization steps")
print(f"✅ Normalization now excludes 'refinery_kbd' outcome column")
print(f"✅ Using: difference(all_numeric(), one_of('refinery_kbd'))")
print(f"{'='*60}")
