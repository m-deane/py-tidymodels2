"""
Script to add extract_preprocessed_data() example cells to the notebook.
"""

import json
import sys

notebook_path = "_md/forecasting_recipes_grouped.ipynb"

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find the cell after the baseline fit_nested
target_code = "fit_baseline = wf_baseline.fit_nested(train_data, group_col='country')"
insert_index = None

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if target_code in source and 'evaluate(test_data)' in source:
            insert_index = i + 1
            break

if insert_index is None:
    print("ERROR: Could not find target cell")
    sys.exit(1)

print(f"Found target cell, will insert at index {insert_index}")

# Create new markdown cell
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Inspecting Preprocessed Data\n",
        "\n",
        "The `.extract_preprocessed_data()` method lets you see what data the models actually see after recipe transformations. This is especially useful with per-group preprocessing where each group may have different transformations.\n",
        "\n",
        "**Key Points:**\n",
        "- Shows the data AFTER recipe steps are applied\n",
        "- Useful for verifying normalization, scaling, feature engineering\n",
        "- Works with both per-group and shared preprocessing\n",
        "- Returns a DataFrame with the group column and metadata (date, split)"
    ]
}

# Create new code cell
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Extract preprocessed training data\n",
        "processed_train = fit_baseline.extract_preprocessed_data(train_data, split='train')\n",
        "\n",
        "print(f\"Preprocessed training data shape: {processed_train.shape}\")\n",
        "print(f\"\\nColumns: {list(processed_train.columns)}\")\n",
        "print(f\"\\nFirst few rows:\")\n",
        "print(processed_train.head(10))\n",
        "\n",
        "# Compare preprocessing statistics across groups\n",
        "print(f\"\\n{'='*70}\")\n",
        "print(\"Preprocessing Statistics by Group:\")\n",
        "print(f\"{'='*70}\")\n",
        "\n",
        "for group in processed_train['country'].unique():\n",
        "    group_data = processed_train[processed_train['country'] == group]\n",
        "    print(f\"\\n{group}:\")\n",
        "    print(f\"  Rows: {len(group_data)}\")\n",
        "    print(f\"  refinery_kbd mean: {group_data['refinery_kbd'].mean():.2f}\")\n",
        "    print(f\"  (Note: Since this is baseline with no preprocessing, values should match original data)\")"
    ]
}

# Create another markdown cell
markdown_cell_2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Extracting Preprocessed Test Data\n",
        "\n",
        "You can also extract preprocessed test data after calling `.evaluate()`. This shows how test data is transformed using the training statistics (e.g., training means/stds for normalization)."
    ]
}

# Create test data extraction cell
code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Extract preprocessed test data\n",
        "processed_test = fit_baseline.extract_preprocessed_data(test_data, split='test')\n",
        "\n",
        "print(f\"Preprocessed test data shape: {processed_test.shape}\")\n",
        "print(f\"\\nFirst few rows:\")\n",
        "print(processed_test.head())\n",
        "\n",
        "# Verify split column\n",
        "print(f\"\\nSplit values: {processed_test['split'].unique()}\")"
    ]
}

# Insert cells
notebook['cells'].insert(insert_index, markdown_cell)
notebook['cells'].insert(insert_index + 1, code_cell)
notebook['cells'].insert(insert_index + 2, markdown_cell_2)
notebook['cells'].insert(insert_index + 3, code_cell_2)

# Save the updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Successfully added 4 cells to notebook at index {insert_index}")
print(f"  - Markdown: Inspecting Preprocessed Data")
print(f"  - Code: Extract train data")
print(f"  - Markdown: Extracting Test Data")
print(f"  - Code: Extract test data")
