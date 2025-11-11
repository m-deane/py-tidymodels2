# Prompt: Create Data Vintages for Refinery Margins Dataset

## Task
Create a Python script that transforms the `refinery_margins.csv` dataset to include synthetic data vintages with an `as_of_date` column. The script should generate multiple versions of each observation representing how the data might have appeared at different points in time.

## Dataset Structure
- **Source file**: `_md/__data/refinery_margins.csv`
- **Date range**: 2006-01-01 to 2021-12-01 (monthly data)
- **Key columns**: `date`, `country`, and various refinery metrics (brent, dubai, wti, refinery_kbd, and multiple margin columns)
- **Structure**: Each date has multiple country rows (approximately 10 countries per date)

## Requirements

### 1. Add `as_of_date` Column
- Add a new column called `as_of_date` to the dataset
- This column represents when the data was "known" or "available"

### 2. Generate Weekly Vintages Over One Year
- For each original row (unique combination of `date` and `country`), create multiple versions with different `as_of_date` values
- Generate vintages spaced **one week apart** over a **one-year period**
- The `as_of_date` should be:
  - **At least one week after** the original `date` (data cannot be known before it occurs)
  - Spaced exactly 7 days apart
  - Cover a full year (52 weeks = 52 vintages per original row)

### 3. Synthetic Data Revisions
To make the vintages realistic, introduce small synthetic revisions to the numeric columns:
- **Early vintages** (closer to the original date): Add small random noise (±1-3% variation)
- **Later vintages** (further from the original date): Gradually converge toward the "final" values with smaller revisions (±0.1-0.5% variation)
- This simulates how data gets revised and refined over time as more information becomes available

### 4. Implementation Details
- Use pandas for data manipulation
- For each original row:
  1. Calculate the first `as_of_date` as `date + 7 days`
  2. Generate 52 weekly `as_of_date` values
  3. For each vintage, apply synthetic revisions to numeric columns:
     - Price columns (brent, dubai, wti): Small percentage changes
     - Margin columns: Small absolute changes (±0.1 to ±0.5)
     - refinery_kbd: Small percentage changes (±1-2%)
  4. Ensure revisions are consistent (later vintages should be closer to final values)

### 5. Output
- Save the result as `refinery_margins_with_vintages.csv` in the same directory
- The output should have all original columns plus the `as_of_date` column
- Each original row should appear 52 times (once for each weekly vintage)

## Example Logic
For a row with `date = '2020-01-01'`:
- Vintage 1: `as_of_date = '2020-01-08'` (original values + early revision noise)
- Vintage 2: `as_of_date = '2020-01-15'` (slightly refined values)
- ...
- Vintage 52: `as_of_date = '2020-12-30'` (values very close to final)

## Notes
- Preserve all original columns
- Maintain data types (dates as datetime, numeric as float)
- Ensure no `as_of_date` is before the original `date`
- The revisions should be realistic (not too large, trending toward stability over time)

