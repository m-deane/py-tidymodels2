#!/usr/bin/env python
"""
Fix test files to handle long format stats DataFrame.

Changes assertions from wide format (rmse as column) to long format (metric/value columns).
"""

import re
from pathlib import Path

def fix_file(filepath):
    """Fix test file to handle long format stats."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Pattern 1: assert 'rmse' in stats.columns
    content = re.sub(
        r"assert 'rmse' in stats\.columns",
        "assert 'rmse' in stats['metric'].values",
        content
    )

    # Pattern 2: assert 'mae' in stats.columns
    content = re.sub(
        r"assert 'mae' in stats\.columns",
        "assert 'mae' in stats['metric'].values",
        content
    )

    # Pattern 3: assert 'r_squared' in stats.columns
    content = re.sub(
        r"assert 'r_squared' in stats\.columns",
        "assert 'r_squared' in stats['metric'].values",
        content
    )

    # Pattern 4: stats[stats['split'] == 'test']['rmse'].iloc[0]
    content = re.sub(
        r"stats\[stats\['split'\] == 'test'\]\['rmse'\]\.iloc\[0\]",
        "stats[(stats['split'] == 'test') & (stats['metric'] == 'rmse')]['value'].iloc[0]",
        content
    )

    # Pattern 5: stats[stats['split'] == 'train']['rmse'].iloc[0]
    content = re.sub(
        r"stats\[stats\['split'\] == 'train'\]\['rmse'\]\.iloc\[0\]",
        "stats[(stats['split'] == 'train') & (stats['metric'] == 'rmse')]['value'].iloc[0]",
        content
    )

    # Pattern 6: test_stats['rmse'].iloc[0]
    content = re.sub(
        r"test_stats\['rmse'\]\.iloc\[0\]",
        "test_stats[test_stats['metric'] == 'rmse']['value'].iloc[0]",
        content
    )

    # Pattern 7: train_stats['rmse'].iloc[0]
    content = re.sub(
        r"train_stats\['rmse'\]\.iloc\[0\]",
        "train_stats[train_stats['metric'] == 'rmse']['value'].iloc[0]",
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Fix all test files
test_dir = Path('/home/user/py-tidymodels2/tests/test_comprehensive_combinations')
files_fixed = []

for test_file in test_dir.glob('test_*.py'):
    if fix_file(test_file):
        files_fixed.append(test_file.name)
        print(f"Fixed: {test_file.name}")

print(f"\nTotal files fixed: {len(files_fixed)}")
