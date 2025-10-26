"""Engine implementations"""

# Import engines to trigger registration
from py_parsnip.engines import sklearn_linear_reg  # noqa: F401
from py_parsnip.engines import sklearn_rand_forest  # noqa: F401
from py_parsnip.engines import statsmodels_linear_reg  # noqa: F401
from py_parsnip.engines import prophet_engine  # noqa: F401
from py_parsnip.engines import statsmodels_arima  # noqa: F401
