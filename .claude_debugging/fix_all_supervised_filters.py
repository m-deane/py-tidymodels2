"""
Fix all supervised filter steps to return copies instead of mutating self.

This script will fix the prep() method in all 7 supervised filter steps.
"""

import re

# Read the file
with open('py_recipes/steps/filter_supervised.py', 'r') as f:
    content = f.read()

# Pattern to find prep() methods that return self after mutation
# We need to replace:
#   self._something = value
#   self._is_prepared = True
#   return self
#
# With:
#   from dataclasses import replace
#   prepared = replace(self)
#   prepared._something = value
#   prepared._is_prepared = True
#   return prepared

# List of classes to fix
classes_to_fix = [
    'StepFilterAnova',
    'StepFilterRfImportance',
    # 'StepFilterMutualInfo',  # Already fixed
    'StepFilterRocAuc',
    'StepFilterChisq',
    'StepSelectShap',
    'StepSelectPermutation'
]

print("=" * 80)
print("FIXING SUPERVISED FILTER STEPS")
print("=" * 80)

for class_name in classes_to_fix:
    print(f"\nSearching for {class_name}.prep() method...")

    # Find the class definition
    class_pattern = rf'class {class_name}:'
    class_match = re.search(class_pattern, content)

    if not class_match:
        print(f"  ✗ Could not find class {class_name}")
        continue

    print(f"  ✓ Found class at position {class_match.start()}")

    # Find the prep() method within this class
    # Look for: def prep(self, data: pd.DataFrame, training: bool = True):
    prep_start = content.find('def prep(self, data: pd.DataFrame, training: bool = True):', class_match.start())

    if prep_start == -1:
        print(f"  ✗ Could not find prep() method for {class_name}")
        continue

    # Check if this prep() belongs to the current class or a later one
    # by ensuring no new class definition appears before it
    next_class_pattern = r'\nclass \w+:'
    next_class_match = re.search(next_class_pattern, content[class_match.end():])

    if next_class_match and (class_match.end() + next_class_match.start()) < prep_start:
        print(f"  ✗ prep() found belongs to next class, skipping")
        continue

    print(f"  ✓ Found prep() method at position {prep_start}")

    # Find the return self statement in this prep() method
    # Look for pattern: self._is_prepared = True\n        return self
    return_pattern = r'(self\._is_prepared = True)\n        (return self)'

    # Search within the prep() method (next 2000 chars should be enough)
    prep_section = content[prep_start:prep_start+2000]
    return_match = re.search(return_pattern, prep_section)

    if not return_match:
        print(f"  ✗ Could not find 'return self' pattern in prep() method")
        continue

    print(f"  ✓ Found 'return self' pattern")

    # Now we need to identify all self._ assignments before return
    # to replace them with prepared._

    # Find the start of the assignments section
    # Look backwards from return self to find where assignments start
    # Usually after the "if len(score_cols) == 0:" check

    assignment_start = prep_section.rfind('raise ValueError("No columns to score")', 0, return_match.start())
    if assignment_start == -1:
        # Try alternative patterns
        assignment_start = prep_section.rfind('if len(', 0, return_match.start())

    if assignment_start == -1:
        print(f"  ✗ Could not find assignment section start")
        continue

    # Get the assignments section
    assignments_section = prep_section[assignment_start:return_match.start()]

    # Find all self._ = patterns
    self_assignments = re.findall(r'self\.(_\w+)\s*=', assignments_section)

    print(f"  ✓ Found {len(self_assignments)} self._ assignments: {self_assignments}")

    # Construct the replacement
    # 1. Add "from dataclasses import replace" and "prepared = replace(self)" before assignments
    # 2. Replace all "self._" with "prepared._" in assignments
    # 3. Replace "return self" with "return prepared"

    # For now, just report what we found
    print(f"  → Would replace 'return self' with 'return prepared'")
    print(f"  → Would add 'from dataclasses import replace' and 'prepared = replace(self)'")
    print(f"  → Would replace {len(self_assignments)} 'self._' with 'prepared._'")

print("\n" + "=" * 80)
print("\nManual fix required - the pattern is too complex for automated replacement.")
print("Please manually update each class's prep() method following the pattern:")
print("""
OLD:
    self._scores = ...
    self._selected_features = ...
    self._is_prepared = True
    return self

NEW:
    from dataclasses import replace
    prepared = replace(self)
    prepared._scores = ...
    prepared._selected_features = ...
    prepared._is_prepared = True
    return prepared
""")
