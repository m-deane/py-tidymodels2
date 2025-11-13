"""
Shared fixtures for comprehensive combination tests.

These fixtures load real data from _md/__data/ directory and provide
preprocessed versions for testing various modeling scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "_md" / "__data"


@pytest.fixture
def refinery_data():
    """
    Load refinery margins data with grouped structure.

    Columns: date, country, refinery_kbd, brent, dubai, wti, various margin columns
    Groups: Multiple countries (Algeria, Denmark, Germany, Italy, etc.)
    Time range: 2006-01-01 onwards
    Use case: Panel/grouped modeling with multiple countries
    """
    df = pd.read_csv(DATA_DIR / "refinery_margins.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture
def refinery_data_ungrouped(refinery_data):
    """
    Aggregate refinery data by date (remove country grouping).

    Use case: Standard time series modeling without groups
    """
    # Aggregate numeric columns by date
    numeric_cols = refinery_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')

    agg_dict = {col: 'mean' for col in numeric_cols}
    df = refinery_data.groupby('date').agg(agg_dict).reset_index()
    return df


@pytest.fixture
def refinery_data_small_groups(refinery_data):
    """
    Filter refinery data to 3 countries for faster testing.

    Use case: Quick grouped model tests
    """
    countries = refinery_data['country'].unique()[:3]
    return refinery_data[refinery_data['country'].isin(countries)].copy()


@pytest.fixture
def gas_demand_data():
    """
    Load European gas demand data with weather variables.

    Columns: date, temperature, wind_speed, gas_demand, country
    Groups: Multiple European countries
    Time range: 2013-01-01 onwards
    Use case: Panel/grouped modeling with weather predictors
    """
    df = pd.read_csv(DATA_DIR / "european_gas_demand_weather_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture
def gas_demand_ungrouped(gas_demand_data):
    """
    Aggregate gas demand data by date (remove country grouping).

    Use case: Standard time series modeling without groups
    """
    numeric_cols = gas_demand_data.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_cols}
    df = gas_demand_data.groupby('date').agg(agg_dict).reset_index()
    return df


@pytest.fixture
def gas_demand_small_groups(gas_demand_data):
    """
    Filter gas demand data to 3 countries for faster testing.

    Use case: Quick grouped model tests
    """
    countries = gas_demand_data['country'].unique()[:3]
    return gas_demand_data[gas_demand_data['country'].isin(countries)].copy()


@pytest.fixture
def jodi_production_data():
    """
    Load JODI refinery production data.

    Columns: date, category, subcategory, country, unit, value, mean_production, pct_zero
    Groups: Multiple countries
    Time range: 2002-01-01 onwards
    Use case: Panel/grouped modeling with production data
    """
    df = pd.read_csv(DATA_DIR / "jodi_refinery_production_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture
def train_test_split_80_20():
    """
    Return a function that splits data 80/20 train/test.

    Usage: train, test = train_test_split_80_20()(data)
    """
    def splitter(data):
        split_idx = int(len(data) * 0.8)
        return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()
    return splitter


@pytest.fixture
def train_test_split_by_group():
    """
    Return a function that splits grouped data 80/20 within each group.

    Usage: train, test = train_test_split_by_group()(data, 'country')
    """
    def splitter(data, group_col):
        train_list = []
        test_list = []

        for group_name, group_data in data.groupby(group_col):
            split_idx = int(len(group_data) * 0.8)
            train_list.append(group_data.iloc[:split_idx])
            test_list.append(group_data.iloc[split_idx:])

        train = pd.concat(train_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)
        return train, test

    return splitter


@pytest.fixture
def sample_recipes():
    """
    Return a dictionary of commonly used recipe configurations.

    Keys: 'basic_normalize', 'pca', 'poly', 'interact', 'select_corr', etc.
    """
    from py_recipes import recipe
    from py_recipes.selectors import all_numeric, all_numeric_predictors

    return {
        'none': None,  # No recipe (formula only)
        'normalize': recipe().step_normalize(all_numeric_predictors()),
        'impute_normalize': (
            recipe()
            .step_impute_median(all_numeric())
            .step_normalize(all_numeric_predictors())
        ),
        'pca_3': (
            recipe()
            .step_normalize(all_numeric_predictors())
            .step_pca(num_comp=3)
        ),
        'pca_5': (
            recipe()
            .step_normalize(all_numeric_predictors())
            .step_pca(num_comp=5)
        ),
        'poly_2': (
            recipe()
            .step_poly(['brent', 'dubai'], degree=2)
            .step_normalize(all_numeric_predictors())
        ),
        'select_corr': (
            recipe()
            .step_select_corr(method='spearman', threshold=0.8)
            .step_normalize(all_numeric_predictors())
        ),
        'ica_3': (
            recipe()
            .step_normalize(all_numeric_predictors())
            .step_ica(num_comp=3)
        ),
        'complex_pipeline': (
            recipe()
            .step_impute_median(all_numeric())
            .step_naomit()
            .step_normalize(all_numeric_predictors())
            .step_pca(num_comp=5)
        ),
    }


@pytest.fixture
def sample_models():
    """
    Return a dictionary of commonly used model specifications.

    Keys: 'linear_reg', 'lasso', 'ridge', 'rand_forest', 'boost_tree', etc.
    """
    from py_parsnip import (
        linear_reg, rand_forest, boost_tree, decision_tree,
        nearest_neighbor, svm_rbf, null_model, naive_reg,
        arima_reg, prophet_reg
    )

    return {
        'linear_reg': linear_reg(),
        'lasso': linear_reg(penalty=0.1, mixture=1.0),
        'elasticnet': linear_reg(penalty=0.1, mixture=0.5),
        'ridge': linear_reg(penalty=0.1, mixture=0.0),
        'rand_forest': rand_forest(trees=50, min_n=5).set_mode('regression'),
        'boost_tree': boost_tree(trees=50, tree_depth=3, learn_rate=0.1).set_mode('regression'),
        'decision_tree': decision_tree(tree_depth=5, min_n=10).set_mode('regression'),
        'knn': nearest_neighbor(neighbors=5).set_mode('regression'),
        'svm': svm_rbf(cost=1.0, rbf_sigma=0.1).set_mode('regression'),
        'null_mean': null_model(strategy='mean'),
        'naive': naive_reg(strategy='naive'),
        'arima': arima_reg(non_seasonal_ar=2, non_seasonal_ma=1),
        'prophet': prophet_reg(seasonality_yearly=True),
    }


@pytest.fixture
def sample_formulas():
    """
    Return a list of commonly used formula specifications.
    """
    return [
        'brent ~ dubai + wti',
        'brent ~ .',
        'brent ~ dubai + wti + I(dubai*wti)',  # Interaction
        'brent ~ dubai + wti + I(dubai**2)',   # Polynomial
        'gas_demand ~ temperature + wind_speed',
        'gas_demand ~ .',
    ]


@pytest.fixture
def metric_set_basic():
    """Return basic metric set for regression."""
    from py_yardstick import metric_set, rmse, mae, r_squared
    return metric_set(rmse, mae, r_squared)


@pytest.fixture
def metric_set_extended():
    """Return extended metric set for regression."""
    from py_yardstick import metric_set, rmse, mae, r_squared, mape, rse
    return metric_set(rmse, mae, r_squared, mape, rse)
