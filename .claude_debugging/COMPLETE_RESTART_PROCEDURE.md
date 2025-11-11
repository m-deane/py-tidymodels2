# Complete Restart Procedure - 2025-11-10

## Issue

You're still seeing errors even after all fixes have been applied. This is because your Jupyter session has:
1. Old code cached in memory
2. Old data/variables cached from previous runs
3. Old bytecode cache on disk

## Complete Restart Procedure

### Step 1: Close Jupyter Completely
1. In Jupyter browser: Click **File** → **Close and Halt**
2. In your terminal running Jupyter: Press **Ctrl+C** twice to stop the server
3. Wait for "Server stopped" message

### Step 2: Clear All Caches
In your terminal:
```bash
cd /Users/matthewdeane/Documents/Data\ Science/python/_projects/py-tidymodels

# Clear Python bytecode cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Force reinstall the package
source py-tidymodels2/bin/activate
pip install -e . --force-reinstall --no-deps

# Restart Python to clear any module caches
deactivate
source py-tidymodels2/bin/activate
```

### Step 3: Restart Jupyter
```bash
cd /Users/matthewdeane/Documents/Data\ Science/python/_projects/py-tidymodels
jupyter notebook
```

### Step 4: Open Notebook Fresh
1. In Jupyter browser, navigate to `_md/`
2. Click `forecasting_recipes_grouped.ipynb`
3. **Kernel** → **Restart & Clear Output**
4. Confirm the restart

### Step 5: Verify the Fix is Loaded
Add this cell at the TOP of your notebook (before running anything else):

```python
# Verification cell - check if fixes are loaded
import py_workflows.workflow as wf_module
import inspect

# Check if _recipe_requires_outcome exists
has_method = hasattr(wf_module.Workflow, '_recipe_requires_outcome')
print(f"✓ _recipe_requires_outcome method exists: {has_method}")

# Check if evaluate has the fix
eval_source = inspect.getsource(wf_module.WorkflowFit.evaluate)
has_needs_outcome = 'needs_outcome = self.workflow._recipe_requires_outcome' in eval_source
print(f"✓ evaluate() has supervised step fix: {has_needs_outcome}")

if has_method and has_needs_outcome:
    print("\n✅ ALL FIXES ARE LOADED - Safe to proceed")
else:
    print("\n❌ FIXES NOT LOADED - Need to restart properly")
```

If this shows "FIXES NOT LOADED", then Python is still using cached code.

### Step 6: Run Notebook from Beginning
1. **Cell** → **Run All**
2. Or run cells sequentially from top to bottom

## Alternative: Nuclear Option

If the above doesn't work, do a complete environment reset:

```bash
cd /Users/matthewdeane/Documents/Data\ Science/python/_projects/py-tidymodels

# Deactivate current environment
deactivate

# Remove and recreate environment
rm -rf py-tidymodels2
python -m venv py-tidymodels2
source py-tidymodels2/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Reinstall Jupyter kernel
python -m ipykernel install --user --name=py-tidymodels2 --display-name="py-tidymodels2"

# Start Jupyter
jupyter notebook
```

## Why This is Necessary

### Python Module Caching
When you `import py_workflows`, Python:
1. Loads the module from disk
2. Caches it in `sys.modules`
3. Reuses the cached version for all subsequent imports

Even with `pip install -e .`, the cached version stays in memory until Python restarts.

### Jupyter Notebook State
Jupyter maintains:
1. Kernel state (all variables)
2. Module cache (all imports)
3. Execution history

Selecting "Restart Kernel" clears variables but keeps some cached state. A full server restart is more thorough.

### Bytecode Cache
Python compiles `.py` files to `.pyc` bytecode for faster loading. Old `.pyc` files can cause issues.

## Verification Checklist

After restarting, verify:
- [ ] Verification cell shows "ALL FIXES ARE LOADED"
- [ ] Cell 49 shows `.step_naomit()` uncommented
- [ ] Cell 32 shows `step_select_corr(outcome='refinery_kbd', ...)`
- [ ] Running cells doesn't produce import errors

## If Still Failing

If you still get errors after a complete restart:

1. **Check the specific error message** - is it the same or different?
2. **Run the diagnostic script** (`.claude_debugging/diagnose_missing_columns.py`)
3. **Check data quality**:
   - Do train_data and test_data have the same columns?
   - Are there NaN values in test data?
4. **Report back** with:
   - The exact cell number failing
   - The complete error message
   - Output from the verification cell

## Expected Behavior After Restart

Cell 57 (ANOVA filter) should:
1. Fit successfully with per-group preprocessing
2. Evaluate test data without errors
3. Display outputs, coefficients, and stats
4. Show forecast plot

All 13 previously failing cells should execute without errors.

---

**Status**: Restart required for fixes to take effect
**Date**: 2025-11-10
**Next Action**: Complete restart procedure above
