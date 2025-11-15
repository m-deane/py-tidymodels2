"""
ShapEngine: Core SHAP computation engine

Auto-selects appropriate SHAP explainer based on model type and provides
standardized DataFrame output format compatible with py-tidymodels ecosystem.
"""

from typing import Optional, Literal, TYPE_CHECKING
import pandas as pd
import numpy as np
import warnings

if TYPE_CHECKING:
    from py_parsnip import ModelFit


class ShapEngine:
    """
    Core SHAP computation engine.

    Auto-selects appropriate explainer based on model type:
    - TreeExplainer: Fast, exact for tree-based models
    - LinearExplainer: Fast, exact for linear models without regularization
    - KernelExplainer: Model-agnostic, slower, universal fallback

    All methods are static for stateless computation.
    """

    @staticmethod
    def explain(
        fit: "ModelFit",
        data: pd.DataFrame,
        method: Literal["auto", "tree", "linear", "kernel"] = "auto",
        background_size: int = 100,
        background: Literal["sample", "kmeans"] = "sample",
        background_data: Optional[pd.DataFrame] = None,
        check_additivity: bool = True
    ) -> pd.DataFrame:
        """
        Compute SHAP values for model predictions.

        Args:
            fit: Fitted ModelFit object
            data: Data to explain (must contain all features used in model)
            method: Explainer method to use:
                - "auto": Auto-select based on model type
                - "tree": TreeExplainer (for tree-based models)
                - "linear": LinearExplainer (for linear models)
                - "kernel": KernelExplainer (model-agnostic)
            background_size: Number of background samples for KernelExplainer
            background: Background sampling strategy ("sample" or "kmeans")
            background_data: Custom background data (overrides background_size)
            check_additivity: Verify SHAP values sum to prediction - base_value

        Returns:
            DataFrame with SHAP values per variable per observation.
            Columns: observation_id, variable, shap_value, abs_shap, feature_value,
                     base_value, prediction, model, model_group_name, [date], [group]

        Raises:
            ImportError: If shap package not installed
            ValueError: If method not supported or training data not available
            NotImplementedError: If model type not supported for requested method

        Examples:
            >>> # Auto-select best explainer
            >>> shap_df = fit.explain(test_data)
            >>>
            >>> # Force specific method
            >>> shap_df = fit.explain(test_data, method="kernel", background_size=50)
            >>>
            >>> # Use custom background data
            >>> shap_df = fit.explain(test_data, background_data=custom_bg)
        """
        # Try importing shap
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package not installed. Install with: pip install shap>=0.43.0"
            )

        # Select explainer method
        if method == "auto":
            method = ShapEngine._auto_select_method(fit)

        # Get model predict function
        predict_fn = ShapEngine._get_predict_function(fit)

        # Prepare data (apply recipe if workflow)
        X = ShapEngine._prepare_data(fit, data)

        # Create explainer
        if method == "tree":
            explainer = ShapEngine._create_tree_explainer(fit, X)
        elif method == "linear":
            explainer = ShapEngine._create_linear_explainer(fit, X)
        elif method == "kernel":
            bg_data = ShapEngine._get_background_data(
                fit, background, background_size, background_data
            )
            explainer = shap.KernelExplainer(predict_fn, bg_data)
        else:
            raise ValueError(
                f"Unknown method: {method}. Must be 'auto', 'tree', 'linear', or 'kernel'"
            )

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

        # Handle potential 3D output for multi-class (use first class for now)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Get expected value (base value)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[0]
        else:
            base_value = explainer.expected_value

        # Get predictions
        predictions = predict_fn(X)

        # Convert to DataFrame
        shap_df = ShapEngine._format_shap_dataframe(
            shap_values=shap_values,
            X=X,
            data=data,
            base_value=base_value,
            predictions=predictions,
            fit=fit
        )

        # Verify additivity
        if check_additivity:
            ShapEngine._check_additivity(shap_df)

        return shap_df

    @staticmethod
    def _auto_select_method(fit: "ModelFit") -> str:
        """
        Auto-select best SHAP explainer based on model type.

        Selection logic:
        - Tree models → TreeExplainer (fast, exact)
        - Linear models without regularization → LinearExplainer (fast, exact)
        - Everything else → KernelExplainer (slow, universal)

        Args:
            fit: Fitted ModelFit object

        Returns:
            Method name: "tree", "linear", or "kernel"
        """
        model_type = fit.spec.model_type

        # Tree-based models → TreeExplainer
        tree_models = [
            "rand_forest", "decision_tree", "boost_tree",
            "bag_tree", "cubist_rules"
        ]
        if model_type in tree_models:
            return "tree"

        # Linear models → LinearExplainer (only if no regularization)
        linear_models = ["linear_reg", "logistic_reg"]
        if model_type in linear_models:
            engine = fit.spec.engine
            # Check if no regularization
            if engine == "sklearn":
                penalty = fit.spec.args.get("penalty", 0)
                if penalty == 0 or penalty is None:
                    return "linear"
            elif engine == "statsmodels":
                # statsmodels OLS has no regularization
                return "linear"

        # Default: KernelExplainer (model-agnostic)
        return "kernel"

    @staticmethod
    def _get_predict_function(fit: "ModelFit"):
        """
        Get model prediction function compatible with SHAP.

        Args:
            fit: Fitted ModelFit object

        Returns:
            Callable that takes X (array or DataFrame) and returns predictions
        """
        def predict(X):
            # Convert to DataFrame if numpy array
            if isinstance(X, np.ndarray):
                # Get feature names from prepared data (without Intercept)
                from py_hardhat import forge
                # Get a sample to determine feature names
                if "X_train" in fit.fit_data and fit.fit_data["X_train"] is not None:
                    X_train = fit.fit_data["X_train"]
                    # Remove Intercept if present
                    if 'Intercept' in X_train.columns:
                        feature_names = [col for col in X_train.columns if col != 'Intercept']
                    else:
                        feature_names = X_train.columns.tolist()
                else:
                    # Fallback: generic names
                    feature_names = [f"x{i}" for i in range(X.shape[1])]

                X = pd.DataFrame(X, columns=feature_names)

            # Get predictions
            preds = fit.predict(X, type="numeric")
            return preds[".pred"].values

        return predict

    @staticmethod
    def _prepare_data(fit: "ModelFit", data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for SHAP computation.

        If fitted via WorkflowFit with recipe, applies recipe transformations.
        Otherwise, uses mold/forge for standard preprocessing.

        Args:
            fit: Fitted ModelFit object
            data: Raw data to prepare

        Returns:
            Preprocessed DataFrame ready for SHAP computation
        """
        # Check if this is a WorkflowFit (has workflow attribute)
        # We need to handle this carefully since fit might be ModelFit or wrapped in WorkflowFit
        # For now, use standard mold/forge approach
        from py_hardhat import forge

        # Use blueprint from fit to preprocess data
        forged = forge(data, fit.blueprint)
        X = forged.predictors

        # Remove Intercept column if present (patsy adds it, but models don't use it for SHAP)
        if 'Intercept' in X.columns:
            X = X.drop(columns=['Intercept'])

        return X

    @staticmethod
    def _create_tree_explainer(fit: "ModelFit", X: pd.DataFrame):
        """
        Create TreeExplainer for tree-based models.

        Args:
            fit: Fitted ModelFit object
            X: Preprocessed feature data (not used for tree explainer)

        Returns:
            shap.TreeExplainer instance

        Raises:
            ImportError: If shap not installed
            ValueError: If model object not found in fit_data
        """
        import shap

        # Get underlying model object
        model = fit.fit_data.get("model")
        if model is None:
            raise ValueError("Model object not found in fit_data. Cannot create TreeExplainer.")

        # TreeExplainer works directly with sklearn/xgboost/lightgbm models
        explainer = shap.TreeExplainer(model)
        return explainer

    @staticmethod
    def _create_linear_explainer(fit: "ModelFit", X: pd.DataFrame):
        """
        Create LinearExplainer for linear models.

        Requires model to have coef_ and intercept_ attributes (sklearn style).

        Args:
            fit: Fitted ModelFit object
            X: Preprocessed feature data (used for masker)

        Returns:
            shap.LinearExplainer instance

        Raises:
            ImportError: If shap not installed
            ValueError: If model doesn't have coef_/intercept_ attributes
        """
        import shap

        # Get coefficients and intercept
        model = fit.fit_data.get("model")
        if model is None:
            raise ValueError("Model object not found in fit_data")

        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            # sklearn model
            W = model.coef_
            b = model.intercept_

            # Get training data mean for masker
            X_train = fit.fit_data.get("X_train")
            if X_train is not None:
                # Remove Intercept column if present
                if 'Intercept' in X_train.columns:
                    X_train = X_train.drop(columns=['Intercept'])
                X_mean = X_train.mean(axis=0).values
            else:
                # Fallback: zeros
                X_mean = np.zeros(X.shape[1])

            # LinearExplainer needs (coef, intercept) and training data for masking
            explainer = shap.LinearExplainer((W, b), (X_mean, X_mean))
        else:
            raise ValueError(
                "Model does not have coef_/intercept_ attributes. "
                "Cannot create LinearExplainer. Try method='kernel' instead."
            )

        return explainer

    @staticmethod
    def _get_background_data(
        fit: "ModelFit",
        background: str,
        background_size: int,
        background_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get background data for KernelExplainer.

        Args:
            fit: Fitted ModelFit object
            background: Sampling strategy ("sample" or "kmeans")
            background_size: Number of background samples
            background_data: Custom background data (overrides other params)

        Returns:
            Background data DataFrame

        Raises:
            ValueError: If training data not available and no custom background provided
        """
        # Custom background data provided
        if background_data is not None:
            return ShapEngine._prepare_data(fit, background_data)

        # Get training data
        X_train = fit.fit_data.get("X_train")
        if X_train is None:
            raise ValueError(
                "Training data not available in fit object. "
                "Provide background_data parameter for KernelExplainer."
            )

        # Remove Intercept column if present
        if 'Intercept' in X_train.columns:
            X_train = X_train.drop(columns=['Intercept'])

        # Random sampling
        if background == "sample":
            if len(X_train) <= background_size:
                return X_train
            return X_train.sample(n=background_size, random_state=42)

        # K-means clustering
        elif background == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=background_size, random_state=42, n_init=10)
            kmeans.fit(X_train)
            return pd.DataFrame(kmeans.cluster_centers_, columns=X_train.columns)

        else:
            raise ValueError(
                f"Unknown background method: {background}. Must be 'sample' or 'kmeans'"
            )

    @staticmethod
    def _format_shap_dataframe(
        shap_values: np.ndarray,
        X: pd.DataFrame,
        data: pd.DataFrame,
        base_value: float,
        predictions: np.ndarray,
        fit: "ModelFit"
    ) -> pd.DataFrame:
        """
        Format SHAP values into standard DataFrame structure.

        Args:
            shap_values: SHAP values array (n_samples, n_features)
            X: Preprocessed feature data
            data: Original data (for date column if present)
            base_value: Expected value (SHAP baseline)
            predictions: Model predictions
            fit: Fitted ModelFit object

        Returns:
            DataFrame with standardized SHAP output format
        """
        # Standard SHAP values are 2D: (n_samples, n_features)
        rows = []
        for i in range(len(X)):
            for j, var in enumerate(X.columns):
                row = {
                    "observation_id": i,
                    "variable": var,
                    "shap_value": shap_values[i, j],
                    "abs_shap": abs(shap_values[i, j]),
                    "feature_value": X.iloc[i, j],
                    "base_value": base_value,
                    "prediction": predictions[i],
                    "model": fit.spec.model_type,
                    "model_group_name": fit.spec.engine
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # Add date column if present in original data
        if "date" in data.columns:
            date_map = {i: data.iloc[i]["date"] for i in range(len(data))}
            df["date"] = df["observation_id"].map(date_map)

        # Add custom model names if set
        if fit.model_name is not None:
            df["model"] = fit.model_name
        if fit.model_group_name is not None:
            df["model_group_name"] = fit.model_group_name

        return df

    @staticmethod
    def _check_additivity(shap_df: pd.DataFrame):
        """
        Verify SHAP values sum to prediction - base_value.

        Issues warning if additivity property violated (within numerical tolerance).

        Args:
            shap_df: SHAP DataFrame from _format_shap_dataframe

        Returns:
            None (issues warnings if additivity violated)
        """
        for obs_id in shap_df["observation_id"].unique():
            obs_data = shap_df[shap_df["observation_id"] == obs_id]

            shap_sum = obs_data["shap_value"].sum()
            base = obs_data["base_value"].iloc[0]
            pred = obs_data["prediction"].iloc[0]

            expected = pred - base
            diff = abs(shap_sum - expected)

            # Tolerance for numerical errors
            if diff > 1e-3:
                warnings.warn(
                    f"Observation {obs_id}: SHAP values don't sum to prediction.\n"
                    f"  SHAP sum: {shap_sum:.6f}\n"
                    f"  Expected (pred - base): {expected:.6f}\n"
                    f"  Difference: {diff:.6f}\n"
                    f"This may indicate numerical precision issues."
                )
