"""
Script to systematically update all remaining WorkflowSet methods with parallel validation utilities.
"""

import re

def update_parallel_execution_block(content, method_name, task_type):
    """
    Update a method's parallel execution block to use validation utilities.

    Args:
        content: File content string
        method_name: Name of the method (e.g., 'fit_nested', 'fit_nested_resamples')
        task_type: Description for format_parallel_info (e.g., 'workflow-group combinations')

    Returns:
        Updated content string
    """
    # Pattern 1: Find "if n_jobs is None or n_jobs == 1:" blocks
    # Pattern 2: Replace with validation + effective_n_jobs

    # For fit_nested, fit_nested_resamples, and fit_global_resamples:
    # Look for the pattern just before "# Sequential or parallel execution"

    patterns = {
        'fit_nested': {
            'before': r'(        # Prepare work items[\s\S]*?for wf_id, wf in self\.workflows\.items\(\).*?\n.*?for group_name in groups\n        \])\n\n        # Sequential or parallel execution\n        if n_jobs is None or n_jobs == 1:',
            'replace': r'\1\n\n        # Validate and resolve n_jobs with warnings\n        effective_n_jobs = validate_n_jobs(n_jobs, len(work_items), verbose=verbose)\n\n        # Windows compatibility check\n        if effective_n_jobs > 1:\n            check_windows_compatibility(verbose=verbose and n_jobs is not None)\n\n        # Sequential or parallel execution\n        if effective_n_jobs == 1:'
        },
        'fit_nested_resamples': {
            'before': r'(        # Prepare work items[\s\S]*?for wf_id, wf in self\.workflows\.items\(\).*?\n.*?for group_name in cv_by_group\.keys\(\)\n        \])\n\n        # Sequential or parallel execution\n        if n_jobs is None or n_jobs == 1:',
            'replace': r'\1\n\n        # Validate and resolve n_jobs with warnings\n        effective_n_jobs = validate_n_jobs(n_jobs, len(work_items), verbose=verbose)\n\n        # Windows compatibility check\n        if effective_n_jobs > 1:\n            check_windows_compatibility(verbose=verbose and n_jobs is not None)\n\n        # Sequential or parallel execution\n        if effective_n_jobs == 1:'
        },
        'fit_global_resamples': {
            'before': r'(        # Prepare work items[\s\S]*?for wf_id, wf in self\.workflows\.items\(\).*?\n.*?for group_name in groups\n        \])\n\n        # Sequential or parallel execution\n        if n_jobs is None or n_jobs == 1:',
            'replace': r'\1\n\n        # Validate and resolve n_jobs with warnings\n        effective_n_jobs = validate_n_jobs(n_jobs, len(work_items), verbose=verbose)\n\n        # Windows compatibility check\n        if effective_n_jobs > 1:\n            check_windows_compatibility(verbose=verbose and n_jobs is not None)\n\n        # Sequential or parallel execution\n        if effective_n_jobs == 1:'
        }
    }

    if method_name in patterns:
        content = re.sub(patterns[method_name]['before'], patterns[method_name]['replace'], content)

    # Now update all "n_jobs=n_jobs" to "n_jobs=effective_n_jobs"
    # and add backend specification for Parallel calls

    # Update sequential blocks
    content = re.sub(
        r'if n_jobs == 1:(\s+)# Sequential execution(\s+)if verbose:(\s+)print\(f"(Fitting .* \(sequential\)\.\.\.)"',
        r'if effective_n_jobs == 1:\1# Sequential execution\2if verbose:\3info = format_parallel_info(1, len(work_items), "' + task_type + r'")\3print(f"{info}..."',
        content
    )

    # Update parallel blocks
    content = re.sub(
        r'else:(\s+)# Parallel execution(\s+)if verbose:(\s+)print\(f"(Fitting .* \(n_jobs=\{n_jobs\}\)\.\.\.)"',
        r'else:\1# Parallel execution\2if verbose:\3info = format_parallel_info(effective_n_jobs, len(work_items), "' + task_type + r'")\3print(f"{info}..."',
        content
    )

    # Update Parallel calls
    content = re.sub(
        r'results = Parallel\(n_jobs=n_jobs, verbose=joblib_verbose\)\(',
        r'results = Parallel(n_jobs=effective_n_jobs, verbose=joblib_verbose, backend=backend)(\n            backend = get_joblib_backend()',
        content
    )

    return content


# Read the file
with open('/home/user/py-tidymodels2/py_workflowsets/workflowset.py', 'r') as f:
    content = f.read()

print("Original file length:", len(content))

# This approach is too complex for regex. Let me do manual edits instead.
print("\nPlease continue with manual edits for the remaining 3 methods.")
print("Methods to update: fit_nested, fit_nested_resamples, fit_global_resamples")
