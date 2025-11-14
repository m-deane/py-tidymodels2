"""
Test script to reproduce the conformal prediction error in notebook 23a.
"""
import pandas as pd
import numpy as np
from py_parsnip import linear_reg

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Testing Conformal Prediction with Nested Models")
print("="*80)

# Load data
print("\n1. Loading data...")
gas_data = pd.read_csv('_md/__data/european_gas_demand_weather_data.csv')
gas_data['date'] = pd.to_datetime(gas_data['date'])
gas_data = gas_data.sort_values(['country', 'date']).reset_index(drop=True)

# Create lag features
def create_lag_features(df, lags=[1, 7, 30]):
    df = df.copy()
    for lag in lags:
        df[f'demand_lag_{lag}'] = df.groupby('country')['gas_demand'].shift(lag)
    df['demand_ma_7'] = df.groupby('country')['gas_demand'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    return df

gas_data = create_lag_features(gas_data, lags=[1, 7, 30])
gas_data['month'] = gas_data['date'].dt.month
gas_data_clean = gas_data.dropna().copy()

# Split
split_date = gas_data_clean['date'].max() - pd.Timedelta(days=90)
train_data = gas_data_clean[gas_data_clean['date'] <= split_date].copy()
test_data = gas_data_clean[gas_data_clean['date'] > split_date].copy()

print(f"Train: {train_data.shape}, Test: {test_data.shape}")
print(f"Train countries: {train_data['country'].nunique()}")
print(f"Test countries: {test_data['country'].nunique()}")

# Fit nested models
print("\n2. Fitting nested models...")
formula = 'gas_demand ~ temperature + wind_speed + demand_lag_1 + demand_lag_7 + demand_lag_30 + demand_ma_7 + month'
spec = linear_reg()
nested_fit = spec.fit_nested(train_data, formula, group_col='country')

print(f"Fitted {len(nested_fit.group_fits)} models")
print(f"Groups in nested_fit: {list(nested_fit.group_fits.keys())[:5]}...")

# Check if each group model has training data
print("\n3. Checking if group models have training data...")
for i, (group, group_fit) in enumerate(list(nested_fit.group_fits.items())[:3]):
    print(f"\nGroup: {group}")
    print(f"  fit_data keys: {list(group_fit.fit_data.keys())}")

    if 'original_training_data' in group_fit.fit_data:
        otd = group_fit.fit_data['original_training_data']
        if otd is not None:
            print(f"  original_training_data: shape={otd.shape}, columns={list(otd.columns)}")
        else:
            print(f"  original_training_data: None ⚠")
    else:
        print(f"  original_training_data: Not found ⚠")

# Try conformal prediction
print("\n4. Attempting conformal prediction...")
try:
    conformal_preds = nested_fit.conformal_predict(
        test_data,
        alpha=0.05,
        method='split',
        per_group_calibration=True
    )
    print(f"✅ SUCCESS! Generated {len(conformal_preds)} predictions")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {str(e)}")

    # Try single group to debug
    print("\n5. Testing single group directly...")
    test_country = list(nested_fit.group_fits.keys())[0]
    print(f"Testing country: {test_country}")

    group_fit = nested_fit.group_fits[test_country]
    group_test = test_data[test_data['country'] == test_country].copy()
    group_test_no_group = group_test.drop(columns=['country'])

    print(f"Group test data shape: {group_test_no_group.shape}")
    print(f"Group test columns: {list(group_test_no_group.columns)}")

    try:
        single_pred = group_fit.conformal_predict(
            group_test_no_group,
            alpha=0.05,
            method='split'
        )
        print(f"✅ Single group SUCCESS! Generated {len(single_pred)} predictions")
    except Exception as e2:
        print(f"❌ Single group FAILED: {type(e2).__name__}: {str(e2)}")

        # Check training data retrieval
        print("\n6. Checking training data retrieval...")
        if 'training_data' in group_fit.fit_data:
            td = group_fit.fit_data['training_data']
            print(f"training_data: {type(td)}, shape={td.shape if td is not None else None}")
        elif 'original_training_data' in group_fit.fit_data:
            otd = group_fit.fit_data['original_training_data']
            print(f"original_training_data: {type(otd)}, shape={otd.shape if otd is not None else None}")
        else:
            print("No training data found in fit_data!")

print("\n" + "="*80)
print("Test Complete")
print("="*80)
