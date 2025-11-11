"""
Comprehensive audit of all recipe steps for mutation bugs.
"""
import re
import sys
from pathlib import Path

def extract_step_classes(file_path):
    """Extract all Step class names from a file."""
    with open(file_path) as f:
        content = f.read()
    
    # Find all Step class definitions
    pattern = r'^class (Step\w+).*:'
    matches = re.findall(pattern, content, re.MULTILINE)
    return matches

def check_prep_method(file_path, class_name):
    """Check if a Step class has mutation bugs in prep()."""
    with open(file_path) as f:
        content = f.read()
    
    # Find the class definition
    class_pattern = rf'^class {class_name}.*?(?=^class |\Z)'
    match = re.search(class_pattern, content, re.MULTILINE | re.DOTALL)
    if not match:
        return "NOT_FOUND", None
    
    class_content = match.group(0)
    
    # Check if it has a prep method
    prep_pattern = r'def prep\(self.*?\n(?=    def |\Z)'
    prep_match = re.search(prep_pattern, class_content, re.DOTALL)
    if not prep_match:
        return "NO_PREP", None
    
    prep_content = prep_match.group(0)
    
    # Check for mutation patterns
    has_self_assignment = bool(re.search(r'\n\s+self\._\w+\s*=', prep_content))
    returns_self = 'return self' in prep_content
    returns_prepared = bool(re.search(r'return Prepared\w+', prep_content))
    uses_replace = 'replace(self)' in prep_content
    
    # Classify
    if has_self_assignment and returns_self and not uses_replace:
        return "BROKEN", prep_content
    elif returns_prepared or (uses_replace and returns_self):
        return "SAFE", None
    elif not has_self_assignment and returns_self:
        return "TRIVIAL", None
    else:
        return "UNCLEAR", prep_content

# Step files to audit
step_files = [
    "basis.py",
    "categorical_extended.py",
    "discretization.py",
    "dummy.py",
    "feature_extraction.py",
    "feature_selection.py",
    "feature_selection_advanced.py",
    "filter_supervised.py",
    "filters.py",
    "financial_oscillators.py",
    "impute.py",
    "interaction_detection.py",
    "interactions.py",
    "mutate.py",
    "naomit.py",
    "normalize.py",
    "reduction.py",
    "remove.py",
    "scaling.py",
    "splitwise.py",
    "timeseries.py",
    "timeseries_extended.py",
    "transformations.py",
]

base_path = Path("/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps")

results = {
    "SAFE": [],
    "BROKEN": [],
    "TRIVIAL": [],
    "NO_PREP": [],
    "UNCLEAR": [],
    "NOT_FOUND": [],
}

total_steps = 0

for file_name in step_files:
    file_path = base_path / file_name
    if not file_path.exists():
        continue
    
    classes = extract_step_classes(file_path)
    for class_name in classes:
        total_steps += 1
        status, content = check_prep_method(file_path, class_name)
        results[status].append((file_name, class_name))
        
        if status == "BROKEN":
            print(f"\n{'='*80}")
            print(f"BROKEN: {file_name}::{class_name}")
            print(f"{'='*80}")
            if content:
                print(content[:500])  # First 500 chars

# Summary
print(f"\n\n{'='*80}")
print(f"AUDIT SUMMARY")
print(f"{'='*80}")
print(f"Total Steps: {total_steps}\n")

for status in ["SAFE", "BROKEN", "TRIVIAL", "NO_PREP", "UNCLEAR", "NOT_FOUND"]:
    count = len(results[status])
    print(f"{status}: {count}")
    if count > 0 and count < 20:
        for file_name, class_name in results[status]:
            print(f"  - {file_name}::{class_name}")

print(f"\n{'='*80}")
print(f"PRIORITY FIXES NEEDED: {len(results['BROKEN'])}")
print(f"{'='*80}")
