"""
Add a summary cell at the end of forecasting_recipes_grouped.ipynb
listing all recipe variables with descriptions
"""
import nbformat

# Recipe descriptions based on notebook analysis
recipe_descriptions = {
    # Basic Normalization & Scaling
    'rec_normalize': 'Normalize numeric features to mean=0, sd=1',
    'rec_scale': 'Min-max scaling to [0, 1] range',
    'rec_center_scale': 'Center (mean=0) and scale (sd=1) numeric features',

    # Transformations
    'rec_log': 'Log transformation for specific columns (handles skewness)',
    'rec_sqrt': 'Square root transformation (moderate skewness correction)',
    'rec_boxcox': 'Box-Cox transformation (automated skewness correction)',
    'rec_yj': 'Yeo-Johnson transformation (handles negative values)',
    'rec_percentile': 'Percentile transformation (robust to outliers)',

    # Polynomial & Interactions
    'rec_poly': 'Polynomial features (degree 2) for non-linear relationships',
    'rec_interact': 'Interaction terms between features (x1 * x2)',
    'rec_bs': 'B-splines for flexible non-linear transformations',
    'rec_ns': 'Natural splines with boundary constraints',

    # Dimensionality Reduction
    'rec_pca': 'PCA for dimensionality reduction (unsupervised)',
    'rec_ica': 'ICA to find independent signal components',
    'rec_kpca': 'Kernel PCA for non-linear dimensionality reduction',
    'rec_pls': 'PLS for supervised dimensionality reduction',

    # Feature Filtering
    'rec_corr': 'Correlation filter (remove highly correlated features)',
    'rec_zv': 'Zero variance filter (remove constant features)',
    'rec_nzv': 'Near-zero variance filter (remove nearly constant features)',
    'rec_lincomb': 'Linear combinations filter (remove redundant features)',
    'rec_missing': 'Missing data filter (remove features with too many NAs)',

    # Supervised Feature Selection (Phase 2)
    'rec_anova': 'ANOVA F-test filter (keep top 50% of features)',
    'rec_rf': 'Random Forest importance filter (keep top 5 features)',
    'rec_rf_imp': 'Random Forest importance-based selection',
    'rec_mi': 'Mutual information filter (keep top 6 features)',
    'rec_perm': 'Permutation importance feature selection',
    'rec_shap': 'SHAP-based feature selection (model-agnostic)',
    'rec_safe': 'SAFE v2 feature engineering + selection',

    # Advanced Feature Selection (Phase 3)
    'rec_vif': 'VIF multicollinearity removal (remove high VIF features)',
    'rec_pvalue': 'P-value based feature selection (statistical significance)',
    'rec_stability': 'Bootstrap stability selection (consistent features)',
    'rec_lofo': 'LOFO importance selection (leave-one-feature-out)',
    'rec_granger': 'Granger causality selection (time series causation)',
    'rec_stepwise': 'Stepwise selection based on AIC/BIC',
    'rec_probe': 'Probe-based selection (compare to random features)',
    'rec_splitwise': 'Splitwise feature selection',

    # Imputation
    'rec_impute': 'Median imputation for missing values',

    # Time Series Features
    'rec_lag': 'Lag features (previous time period values)',
    'rec_diff': 'Differencing for stationarity (removes trends)',
    'rec_rolling': 'Rolling window features (moving statistics)',
    'rec_ewm': 'Exponentially weighted moving features (recent emphasis)',
    'rec_date': 'Date features (year, month, day, etc.)',
    'rec_ts_sig': 'Timeseries signature (comprehensive date features)',
    'rec_fourier': 'Fourier features for seasonal patterns',

    # Discretization
    'rec_discretize': 'Discretization into quantile bins',

    # Complex/Multi-Step
    'rec_complex': 'Complex multi-step recipe (impute, normalize, PCA, select)',

    # Model-Specific
    'rec_xgb': 'Recipe optimized for XGBoost (impute + normalize)',
}

# Read notebook
nb_path = 'forecasting_recipes_grouped.ipynb'
with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Create markdown cell with title
markdown_cell = nbformat.v4.new_markdown_cell(
    source="# Recipe Variables Summary\n\n"
           "This notebook demonstrates **47 different recipe preprocessing strategies** across 8 categories.\n\n"
           "Below is a complete index of all recipe variables created in this notebook:"
)

# Create code cell with recipe summary
code_lines = [
    "# Complete Recipe Variable Index",
    "# ==============================",
    "",
    "import pandas as pd",
    "",
    "recipes_index = {",
]

# Group recipes by category
categories = {
    'Basic Normalization & Scaling': ['rec_normalize', 'rec_scale', 'rec_center_scale'],
    'Transformations': ['rec_log', 'rec_sqrt', 'rec_boxcox', 'rec_yj', 'rec_percentile'],
    'Polynomial & Interactions': ['rec_poly', 'rec_interact', 'rec_bs', 'rec_ns'],
    'Dimensionality Reduction': ['rec_pca', 'rec_ica', 'rec_kpca', 'rec_pls'],
    'Feature Filtering': ['rec_corr', 'rec_zv', 'rec_nzv', 'rec_lincomb', 'rec_missing'],
    'Supervised Feature Selection': [
        'rec_anova', 'rec_rf', 'rec_rf_imp', 'rec_mi', 'rec_perm',
        'rec_shap', 'rec_safe', 'rec_splitwise'
    ],
    'Advanced Selection (Phase 3)': [
        'rec_vif', 'rec_pvalue', 'rec_stability', 'rec_lofo',
        'rec_granger', 'rec_stepwise', 'rec_probe'
    ],
    'Imputation': ['rec_impute'],
    'Time Series Features': [
        'rec_lag', 'rec_diff', 'rec_rolling', 'rec_ewm',
        'rec_date', 'rec_ts_sig', 'rec_fourier'
    ],
    'Discretization': ['rec_discretize'],
    'Complex Workflows': ['rec_complex'],
    'Model-Specific': ['rec_xgb'],
}

for category, recipes in categories.items():
    code_lines.append(f"    # {category}")
    for rec in recipes:
        if rec in recipe_descriptions:
            desc = recipe_descriptions[rec]
            code_lines.append(f"    '{rec}': '{desc}',")
    code_lines.append("")

code_lines.append("}")
code_lines.append("")
code_lines.append("# Display as DataFrame")
code_lines.append("df_recipes = pd.DataFrame([")
code_lines.append("    {'Variable': var, 'Description': desc}")
code_lines.append("    for var, desc in recipes_index.items()")
code_lines.append("])")
code_lines.append("")
code_lines.append("print(f\"Total Recipes Demonstrated: {len(df_recipes)}\")")
code_lines.append("print(\"\\n\" + \"=\"*80)")
code_lines.append("display(df_recipes)")
code_lines.append("")
code_lines.append("# Count by category")
code_lines.append("print(\"\\n\" + \"=\"*80)")
code_lines.append("print(\"Recipe Count by Category:\")")
code_lines.append("print(\"=\"*80)")

for category, recipes in categories.items():
    code_lines.append(f"print(f\"{category:.<50} {{len({recipes})}}\")")

code_cell = nbformat.v4.new_code_cell(source='\n'.join(code_lines))

# Add cells to notebook
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)

# Write back
with open(nb_path, 'w') as f:
    nbformat.write(nb, f)

print(f"✓ Added 2 summary cells to notebook")
print(f"✓ Total cells now: {len(nb.cells)}")
print(f"✓ Total recipes indexed: {len(recipe_descriptions)}")
print("\nCategories:")
for cat, recs in categories.items():
    print(f"  - {cat}: {len(recs)} recipes")
