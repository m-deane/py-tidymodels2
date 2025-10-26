"""
Engine Registry: Pluggable backend system for models

The engine registry allows different computational backends to be registered
for each model type. For example:
- linear_reg + sklearn → LinearRegression
- linear_reg + statsmodels → OLS
- rand_forest + sklearn → RandomForestRegressor

This uses a decorator pattern for clean registration.
"""

from typing import Dict, Tuple, Type, Any, Literal
from abc import ABC, abstractmethod
import pandas as pd

from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


# Global registry: (model_type, engine) → Engine class
ENGINE_REGISTRY: Dict[Tuple[str, str], Type["Engine"]] = {}


class Engine(ABC):
    """
    Abstract base class for model engines.

    Each engine must implement:
    - fit(): Fit model to molded data
    - predict(): Make predictions on forged data
    - extract_outputs(): Extract standardized three-DataFrame output
    - translate_params(): Translate tidymodels params → engine params
    """

    # Parameter mapping: tidymodels name → engine name
    # Override in subclasses
    param_map: Dict[str, str] = {}

    @abstractmethod
    def fit(self, spec: ModelSpec, molded: MoldedData) -> Dict[str, Any]:
        """
        Fit model to molded data.

        Args:
            spec: ModelSpec with model configuration
            molded: MoldedData with outcomes and predictors

        Returns:
            Dict containing fitted model and metadata
        """
        pass

    @abstractmethod
    def predict(
        self,
        fit: ModelFit,
        molded: MoldedData,
        type: Literal["numeric", "class", "prob", "conf_int"],
    ) -> pd.DataFrame:
        """
        Make predictions on forged data.

        Args:
            fit: ModelFit with fitted model
            molded: MoldedData with predictors
            type: Prediction type

        Returns:
            DataFrame with predictions
        """
        pass

    @abstractmethod
    def extract_outputs(
        self, fit: ModelFit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract standardized three-DataFrame output.

        Args:
            fit: ModelFit with fitted model

        Returns:
            Tuple of (model_outputs, coefficients, stats)
        """
        pass

    def translate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate tidymodels params to engine-specific params.

        Args:
            params: Tidymodels-style parameters

        Returns:
            Engine-specific parameters

        Example:
            >>> # For sklearn Ridge:
            >>> params = {"penalty": 0.1, "mixture": 0.5}
            >>> translated = engine.translate_params(params)
            >>> # {"alpha": 0.1, "l1_ratio": 0.5}
        """
        translated = {}
        for key, value in params.items():
            # Use param_map if available, otherwise keep original name
            engine_key = self.param_map.get(key, key)
            translated[engine_key] = value
        return translated


def register_engine(model_type: str, engine: str):
    """
    Decorator to register an engine for a model type.

    Args:
        model_type: Model type (e.g., "linear_reg")
        engine: Engine name (e.g., "sklearn")

    Example:
        >>> @register_engine("linear_reg", "sklearn")
        ... class SklearnLinearEngine(Engine):
        ...     def fit(self, spec, molded):
        ...         ...
    """

    def decorator(engine_class: Type[Engine]):
        ENGINE_REGISTRY[(model_type, engine)] = engine_class
        return engine_class

    return decorator


def get_engine(model_type: str, engine: str) -> Engine:
    """
    Get an engine instance for a model type.

    Args:
        model_type: Model type (e.g., "linear_reg")
        engine: Engine name (e.g., "sklearn")

    Returns:
        Engine instance

    Raises:
        ValueError: If engine not registered

    Example:
        >>> engine = get_engine("linear_reg", "sklearn")
        >>> fit_data = engine.fit(spec, molded)
    """
    key = (model_type, engine)
    if key not in ENGINE_REGISTRY:
        available = list(ENGINE_REGISTRY.keys())
        raise ValueError(
            f"No engine registered for model_type='{model_type}', engine='{engine}'. "
            f"Available: {available}"
        )

    engine_class = ENGINE_REGISTRY[key]
    return engine_class()


def list_engines() -> pd.DataFrame:
    """
    List all registered engines.

    Returns:
        DataFrame with model_type and engine columns

    Example:
        >>> list_engines()
           model_type    engine
        0  linear_reg   sklearn
        1  linear_reg   statsmodels
        2  rand_forest  sklearn
    """
    data = [
        {"model_type": model_type, "engine": engine}
        for model_type, engine in ENGINE_REGISTRY.keys()
    ]
    return pd.DataFrame(data)
