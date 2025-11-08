import pandas as pd
import sys
sys.path.insert(0, '/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels')

# This would be run in Jupyter to check what's in wf_results_recipes
# Add this cell after cell 32:

print("Checking recipe workflow results...")
print()

# Check what metrics were collected
recipe_metrics = wf_results_recipes.collect_metrics()
print(f"Metrics collected: {len(recipe_metrics)} rows")
print(f"Columns: {list(recipe_metrics.columns)}")
print()

if len(recipe_metrics) > 0:
    print("Sample metrics:")
    print(recipe_metrics.head(10))
    print()
    
    # Check which metrics are present
    unique_metrics = recipe_metrics['metric'].unique() if 'metric' in recipe_metrics.columns else []
    print(f"Unique metrics: {unique_metrics}")
else:
    print("NO METRICS COLLECTED - All workflows failed during CV")
    print()
    print("This means the fits succeeded but metrics computation failed.")
    print("Likely cause: The workflows completed but didn't return valid predictions.")

