"""
Script to create synthetic data vintages for refinery_margins.csv
Generates weekly vintages over one year for each original observation.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Read the original dataset
print("Reading refinery_margins.csv...")
df = pd.read_csv('_md/__data/refinery_margins.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

print(f"Original dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique dates: {df['date'].nunique()}")
print(f"Unique countries: {df['country'].nunique()}")

# Identify numeric columns (excluding date and country)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns to revise: {len(numeric_cols)}")

# Identify price columns (brent, dubai, wti) and margin columns
price_cols = [col for col in numeric_cols if col in ['brent', 'dubai', 'wti']]
margin_cols = [col for col in numeric_cols if 'margin' in col.lower() or 'cracking' in col.lower() or 'hydroskimming' in col.lower() or 'coking' in col.lower()]
other_numeric_cols = [col for col in numeric_cols if col not in price_cols + margin_cols and col != 'refinery_kbd']
refinery_col = ['refinery_kbd'] if 'refinery_kbd' in numeric_cols else []

print(f"Price columns: {price_cols}")
print(f"Margin columns: {len(margin_cols)}")
print(f"Refinery column: {refinery_col}")
print(f"Other numeric columns: {other_numeric_cols}")

# Function to create vintages for a single row
def create_vintages_for_row(row, num_weeks=52):
    """
    Create synthetic vintages for a single row.
    Returns a DataFrame with num_weeks rows (one per vintage).
    """
    # Generate as_of_dates (starting 7 days after the original date)
    base_date = row['date']
    as_of_dates = [base_date + timedelta(days=7 * (i + 1)) for i in range(num_weeks)]
    
    # Create list to store vintage rows
    vintage_rows = []
    
    # For each vintage, create a revised version of the row
    for week_idx, as_of_date in enumerate(as_of_dates):
        # Create a copy of the original row
        vintage_row = row.copy()
        vintage_row['as_of_date'] = as_of_date
        
        # Calculate revision factor (decreases over time)
        # Early vintages: larger revisions, later vintages: smaller revisions
        # Week 1: ~2-3% variation, Week 52: ~0.1-0.3% variation
        progress = week_idx / (num_weeks - 1)  # 0 to 1
        early_noise_level = 0.025  # 2.5% for early vintages
        late_noise_level = 0.002   # 0.2% for late vintages
        noise_level = early_noise_level * (1 - progress) + late_noise_level * progress
        
        # Apply revisions to price columns (percentage-based)
        for col in price_cols:
            if pd.notna(vintage_row[col]):
                noise = np.random.normal(0, noise_level)
                vintage_row[col] = vintage_row[col] * (1 + noise)
        
        # Apply revisions to refinery_kbd (percentage-based, smaller range)
        for col in refinery_col:
            if pd.notna(vintage_row[col]):
                refinery_noise_level = noise_level * 0.5  # Smaller variation for refinery capacity
                noise = np.random.normal(0, refinery_noise_level)
                vintage_row[col] = vintage_row[col] * (1 + noise)
        
        # Apply revisions to margin columns (absolute changes)
        # Early: ±0.3 to ±0.5, Late: ±0.05 to ±0.15
        for col in margin_cols:
            if pd.notna(vintage_row[col]):
                early_margin_noise = 0.4
                late_margin_noise = 0.1
                margin_noise_level = early_margin_noise * (1 - progress) + late_margin_noise * progress
                noise = np.random.normal(0, margin_noise_level)
                vintage_row[col] = vintage_row[col] + noise
        
        # Apply revisions to other numeric columns (percentage-based, smaller)
        for col in other_numeric_cols:
            if pd.notna(vintage_row[col]):
                noise = np.random.normal(0, noise_level * 0.7)
                vintage_row[col] = vintage_row[col] * (1 + noise)
        
        vintage_rows.append(vintage_row)
    
    return pd.DataFrame(vintage_rows)

# Process each row and create vintages
print("\nCreating vintages for each row...")
print("This may take a few minutes...")

vintage_dfs = []
total_rows = len(df)

for idx, (_, row) in enumerate(df.iterrows()):
    if (idx + 1) % 100 == 0:
        print(f"Processing row {idx + 1}/{total_rows}...")
    
    vintage_df = create_vintages_for_row(row, num_weeks=52)
    vintage_dfs.append(vintage_df)

# Combine all vintages into a single DataFrame
print("\nCombining all vintages...")
df_with_vintages = pd.concat(vintage_dfs, ignore_index=True)

# Reorder columns to put as_of_date after date
cols = df.columns.tolist()
date_idx = cols.index('date')
cols.insert(date_idx + 1, 'as_of_date')
df_with_vintages = df_with_vintages[cols]

print(f"\nFinal dataset shape: {df_with_vintages.shape}")
print(f"Expected rows: {len(df) * 52}")
print(f"Actual rows: {len(df_with_vintages)}")

# Verify the structure
print(f"\nSample of first few vintages for first row:")
print(df_with_vintages[df_with_vintages['date'] == df['date'].iloc[0]].head(5)[['date', 'as_of_date', 'country', 'brent', 'dubai', 'wti']])

# Save to CSV
output_path = '_md/__data/refinery_margins_with_vintages.csv'
print(f"\nSaving to {output_path}...")
df_with_vintages.to_csv(output_path, index=False)
print("Done!")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original rows: {len(df):,}")
print(f"Vintages per row: 52")
print(f"Total output rows: {len(df_with_vintages):,}")
print(f"Date range in output: {df_with_vintages['date'].min()} to {df_with_vintages['date'].max()}")
print(f"As-of-date range: {df_with_vintages['as_of_date'].min()} to {df_with_vintages['as_of_date'].max()}")
print(f"Output file: {output_path}")

