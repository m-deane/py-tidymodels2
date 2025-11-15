"""
Debug notebook 23a coefficients array length issue
"""
import pandas as pd
import numpy as np
from py_parsnip import linear_reg

# Load data
print("Loading data...")
gas_data = pd.read_csv('_md/__data/european_gas_demand_weather_data.csv')
gas_data['date'] = pd.to_datetime(gas_data['date'])

# Create lag features
def create_lag_features(df, lags=[1, 7, 30]):
    df_out = df.copy()
    for country in df['country'].unique():
        country_mask = df_out['country'] == country
        for lag in lags:
            df_out.loc[country_mask, f'demand_lag_{lag}'] = (
                df_out.loc[country_mask, 'gas_demand'].shift(lag)
            )
            df_out.loc[country_mask, f'demand_ma_7'] = (
                df_out.loc[country_mask, 'gas_demand']
                .rolling(window=7, min_periods=1).mean()
            )
    df_out['month'] = df_out['date'].dt.month
    return df_out.dropna()

gas_data_clean = create_lag_features(gas_data, lags=[1, 7, 30])
print(f"Data shape after lag features: {gas_data_clean.shape}")
print(f"Countries: {gas_data_clean['country'].unique()}")

# Split data
split_date = gas_data_clean['date'].max() - pd.Timedelta(days=90)
train_data = gas_data_clean[gas_data_clean['date'] <= split_date].copy()
test_data = gas_data_clean[gas_data_clean['date'] > split_date].copy()

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# Fit nested model
formula = 'gas_demand ~ temperature + wind_speed + demand_lag_1 + demand_lag_7 + demand_lag_30 + demand_ma_7 + month'
print(f"\nFormula: {formula}")

spec = linear_reg()
print("\nFitting nested model...")
nested_fit = spec.fit_nested(train_data, formula, group_col='country')

print("\nExtract outputs WITHOUT conformal...")
try:
    outputs, coeffs, stats = nested_fit.extract_outputs()
    print(f"✅ Success WITHOUT conformal")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Coefficients shape: {coeffs.shape}")
    print(f"   Stats shape: {stats.shape}")
except Exception as e:
    print(f"❌ FAILED without conformal: {type(e).__name__}: {str(e)}")

print("\nExtract outputs WITH conformal (alpha=0.25)...")
try:
    outputs, coeffs, stats = nested_fit.extract_outputs(conformal_alpha=0.25)
    print(f"✅ Success WITH conformal")
    print(f"   Outputs shape: {outputs.shape}")
    print(f"   Coefficients shape: {coeffs.shape}")
    print(f"   Stats shape: {stats.shape}")
except Exception as e:
    print(f"❌ FAILED with conformal: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nDebug: Testing each group individually...")
for country in train_data['country'].unique():
    print(f"\n  Testing {country}...")
    country_train = train_data[train_data['country'] == country]
    try:
        spec = linear_reg()
        fit = spec.fit(country_train, formula)
        outputs, coeffs, stats = fit.extract_outputs()
        print(f"    ✅ {country}: coeffs shape {coeffs.shape}")
        print(f"       Column lengths: variable={len(coeffs['variable'])}, coefficient={len(coeffs['coefficient'])}, vif={len(coeffs['vif'])}")
    except Exception as e:
        print(f"    ❌ {country} FAILED: {type(e).__name__}: {str(e)[:100]}")
