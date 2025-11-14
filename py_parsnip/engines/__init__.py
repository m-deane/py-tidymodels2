"""Engine implementations"""

# Import engines to trigger registration
from py_parsnip.engines import sklearn_linear_reg  # noqa: F401
from py_parsnip.engines import sklearn_rand_forest  # noqa: F401
from py_parsnip.engines import statsmodels_linear_reg  # noqa: F401
from py_parsnip.engines import statsmodels_panel  # noqa: F401
from py_parsnip.engines import prophet_engine  # noqa: F401
from py_parsnip.engines import statsmodels_arima  # noqa: F401
from py_parsnip.engines import pmdarima_auto_arima  # noqa: F401
from py_parsnip.engines import statsforecast_auto_arima  # noqa: F401
from py_parsnip.engines import statsmodels_varmax  # noqa: F401
from py_parsnip.engines import statsmodels_exp_smoothing  # noqa: F401
from py_parsnip.engines import statsmodels_seasonal_reg  # noqa: F401
from py_parsnip.engines import skforecast_recursive  # noqa: F401
from py_parsnip.engines import parsnip_null_model  # noqa: F401
from py_parsnip.engines import parsnip_naive_reg  # noqa: F401
from py_parsnip.engines import hybrid_arima_boost  # noqa: F401
from py_parsnip.engines import hybrid_prophet_boost  # noqa: F401
from py_parsnip.engines import xgboost_boost_tree  # noqa: F401
from py_parsnip.engines import lightgbm_boost_tree  # noqa: F401
from py_parsnip.engines import catboost_boost_tree  # noqa: F401
from py_parsnip.engines import pyearth_mars  # noqa: F401
from py_parsnip.engines import statsmodels_poisson_reg  # noqa: F401
from py_parsnip.engines import pygam_gam  # noqa: F401
from py_parsnip.engines import sklearn_decision_tree  # noqa: F401
from py_parsnip.engines import sklearn_nearest_neighbor  # noqa: F401
from py_parsnip.engines import sklearn_svm_rbf  # noqa: F401
from py_parsnip.engines import sklearn_svm_linear  # noqa: F401
from py_parsnip.engines import sklearn_svm_poly  # noqa: F401
from py_parsnip.engines import sklearn_mlp  # noqa: F401
from py_parsnip.engines import sklearn_pls  # noqa: F401
from py_parsnip.engines import sklearn_bag_tree  # noqa: F401
from py_parsnip.engines import generic_hybrid  # noqa: F401
from py_parsnip.engines import parsnip_manual_reg  # noqa: F401
from py_parsnip.engines import imodels_rule_fit  # noqa: F401
from py_parsnip.engines import parsnip_window_reg  # noqa: F401
