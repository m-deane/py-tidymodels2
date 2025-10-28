"""
Quick test script to understand MSTL API structure in statsmodels
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import MSTL

# Create simple time series with multiple seasonalities
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=730, freq='D')
# Weekly (period 7) + Yearly (period 365) patterns
weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(730) / 7)
yearly_pattern = 20 * np.sin(2 * np.pi * np.arange(730) / 365)
trend = np.linspace(100, 150, 730)
noise = np.random.normal(0, 5, 730)
values = trend + weekly_pattern + yearly_pattern + noise

ts = pd.Series(values, index=dates)

# Fit MSTL
mstl = MSTL(ts, periods=[7, 365], windows=[7, 365], iterate=2)
result = mstl.fit()

# Inspect the structure
print("=" * 60)
print("MSTL Result Object Structure")
print("=" * 60)
print(f"\nType of result: {type(result)}")
print(f"\nAll attributes (non-private):")
attrs = [attr for attr in dir(result) if not attr.startswith('_')]
for attr in attrs:
    print(f"  - {attr}")

print(f"\n{'=' * 60}")
print("Seasonal Component Analysis")
print("=" * 60)
print(f"\nType of result.seasonal: {type(result.seasonal)}")
print(f"Shape of result.seasonal: {result.seasonal.shape}")

# Check if it's a DataFrame or Series
if hasattr(result.seasonal, 'columns'):
    print(f"It's a DataFrame with columns: {result.seasonal.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(result.seasonal.head())
else:
    print("It's a Series")
    print(f"Index type: {type(result.seasonal.index)}")
    print(f"\nFirst 10 values:")
    print(result.seasonal.head(10))

# Check for alternative attributes
print(f"\n{'=' * 60}")
print("Looking for individual seasonal components")
print("=" * 60)

# Check if there's a seasonal_ attribute
if hasattr(result, 'seasonal_'):
    print(f"\nFound 'seasonal_' attribute!")
    print(f"Type: {type(result.seasonal_)}")
    print(f"Shape: {result.seasonal_.shape if hasattr(result.seasonal_, 'shape') else 'N/A'}")
    if isinstance(result.seasonal_, (list, tuple)):
        print(f"Length: {len(result.seasonal_)}")
        for i, component in enumerate(result.seasonal_):
            print(f"\n  Component {i} (period {[7, 365][i]}):")
            print(f"    Type: {type(component)}")
            print(f"    Shape: {component.shape if hasattr(component, 'shape') else 'N/A'}")
            if hasattr(component, 'head'):
                print(f"    First 5 values: {component.head().values}")
elif hasattr(result, 'components'):
    print(f"\nFound 'components' attribute!")
    print(f"Type: {type(result.components)}")
    print(result.components)
else:
    print("\nNo 'seasonal_' or 'components' attribute found")

# Check if seasonal is sum of components
print(f"\n{'=' * 60}")
print("Other Components")
print("=" * 60)
print(f"\nTrend type: {type(result.trend)}")
print(f"Trend shape: {result.trend.shape}")
print(f"\nResid type: {type(result.resid)}")
print(f"Resid shape: {result.resid.shape}")

print(f"\n{'=' * 60}")
print("Statsmodels Version")
print("=" * 60)
import statsmodels
print(f"statsmodels version: {statsmodels.__version__}")
