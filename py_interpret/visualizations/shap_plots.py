"""
SHAP visualization functions for model interpretability

Provides 5 core plot types:
1. summary_plot: Global feature importance (beeswarm or bar chart)
2. waterfall_plot: Local explanation for single prediction
3. force_plot: Interactive HTML force plot for single prediction
4. dependence_plot: Partial dependence for single feature
5. temporal_plot: SHAP values over time (for time series)

All functions accept SHAP DataFrames from explain() method and return
matplotlib Figure objects (or HTML for force_plot).
"""

from typing import Optional, Literal, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def summary_plot(
    shap_df: pd.DataFrame,
    plot_type: Literal["beeswarm", "bar"] = "beeswarm",
    max_display: int = 20,
    show: bool = True,
    figsize: tuple = (10, 6),
    **kwargs
) -> plt.Figure:
    """
    Create SHAP summary plot showing global feature importance.

    For beeswarm plots, each dot represents one observation's SHAP value for a feature.
    Color indicates feature value (red = high, blue = low). For bar plots, shows
    mean absolute SHAP value per feature.

    Args:
        shap_df: SHAP DataFrame from explain() method with columns:
                 observation_id, variable, shap_value, abs_shap, feature_value
        plot_type: "beeswarm" for detailed distribution or "bar" for simple ranking
        max_display: Maximum number of features to display (ranked by importance)
        show: Whether to call plt.show() (default: True)
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments passed to SHAP's summary_plot

    Returns:
        matplotlib.figure.Figure object

    Raises:
        ImportError: If shap package not installed
        ValueError: If required columns missing from shap_df

    Examples:
        >>> # Get SHAP values
        >>> shap_df = fit.explain(test_data)
        >>>
        >>> # Beeswarm plot (default)
        >>> fig = summary_plot(shap_df, max_display=15)
        >>>
        >>> # Bar chart for simple ranking
        >>> fig = summary_plot(shap_df, plot_type="bar")
    """
    # Import SHAP
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package not installed. Install with: pip install shap>=0.43.0"
        )

    # Validate input
    required_cols = ["observation_id", "variable", "shap_value", "feature_value"]
    _validate_shap_dataframe(shap_df, required_cols)

    # Convert long-format DataFrame to SHAP matrix format
    shap_values, feature_values, feature_names = _convert_to_shap_matrix(shap_df)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create summary plot using SHAP's built-in function
    if plot_type == "beeswarm":
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
            **kwargs
        )
    elif plot_type == "bar":
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="bar",
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Must be 'beeswarm' or 'bar'"
        )

    # Get current figure (SHAP creates its own)
    fig = plt.gcf()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def waterfall_plot(
    shap_df: pd.DataFrame,
    observation_id: int,
    max_display: int = 10,
    show: bool = True,
    figsize: tuple = (10, 6),
    **kwargs
) -> plt.Figure:
    """
    Create waterfall plot showing how features contribute to a single prediction.

    Shows the base value (expected prediction), individual feature contributions,
    and final prediction. Features are ordered by absolute SHAP value.

    Args:
        shap_df: SHAP DataFrame from explain() method
        observation_id: Which observation to explain (row index)
        max_display: Maximum number of features to display
        show: Whether to call plt.show()
        figsize: Figure size as (width, height)
        **kwargs: Additional arguments passed to SHAP's waterfall_plot

    Returns:
        matplotlib.figure.Figure object

    Raises:
        ImportError: If shap package not installed
        ValueError: If observation_id not found in shap_df

    Examples:
        >>> # Explain first observation
        >>> fig = waterfall_plot(shap_df, observation_id=0)
        >>>
        >>> # Explain specific observation with more features
        >>> fig = waterfall_plot(shap_df, observation_id=42, max_display=15)
    """
    # Import SHAP
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package not installed. Install with: pip install shap>=0.43.0"
        )

    # Validate input
    required_cols = ["observation_id", "variable", "shap_value", "feature_value", "base_value"]
    _validate_shap_dataframe(shap_df, required_cols)

    # Check observation_id exists
    if observation_id not in shap_df["observation_id"].values:
        raise ValueError(
            f"observation_id {observation_id} not found in shap_df. "
            f"Valid range: 0 to {shap_df['observation_id'].max()}"
        )

    # Filter to single observation
    obs_data = shap_df[shap_df["observation_id"] == observation_id].copy()

    # Sort by absolute SHAP value (descending)
    obs_data = obs_data.sort_values("abs_shap", ascending=False)

    # Get base value and prediction
    base_value = obs_data["base_value"].iloc[0]
    prediction = obs_data["prediction"].iloc[0] if "prediction" in obs_data.columns else None

    # Prepare data for waterfall plot
    feature_names = obs_data["variable"].values
    shap_values = obs_data["shap_value"].values
    feature_values = obs_data["feature_value"].values

    # Create Explanation object for waterfall plot
    # SHAP's waterfall_plot expects an Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names
    )

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create waterfall plot
    shap.waterfall_plot(explanation, max_display=max_display, show=False, **kwargs)

    # Get current figure
    fig = plt.gcf()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def force_plot(
    shap_df: pd.DataFrame,
    observation_id: int,
    matplotlib: bool = False,
    show: bool = True,
    figsize: tuple = (20, 3),
    **kwargs
) -> Union[str, plt.Figure]:
    """
    Create interactive force plot for a single prediction.

    Force plots show how features push prediction from base value to final value.
    Red features push prediction higher, blue features push it lower.

    Args:
        shap_df: SHAP DataFrame from explain() method
        observation_id: Which observation to explain
        matplotlib: If True, return matplotlib figure instead of HTML (default: False)
        show: Whether to display plot
        figsize: Figure size (only used if matplotlib=True)
        **kwargs: Additional arguments passed to SHAP's force_plot

    Returns:
        HTML string (if matplotlib=False) or matplotlib.figure.Figure (if matplotlib=True)

    Raises:
        ImportError: If shap package not installed
        ValueError: If observation_id not found

    Examples:
        >>> # Interactive HTML force plot
        >>> html = force_plot(shap_df, observation_id=0)
        >>>
        >>> # Static matplotlib version
        >>> fig = force_plot(shap_df, observation_id=0, matplotlib=True)
    """
    # Import SHAP
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package not installed. Install with: pip install shap>=0.43.0"
        )

    # Validate input
    required_cols = ["observation_id", "variable", "shap_value", "feature_value", "base_value"]
    _validate_shap_dataframe(shap_df, required_cols)

    # Check observation_id exists
    if observation_id not in shap_df["observation_id"].values:
        raise ValueError(
            f"observation_id {observation_id} not found in shap_df"
        )

    # Filter to single observation
    obs_data = shap_df[shap_df["observation_id"] == observation_id].copy()

    # Get data
    base_value = obs_data["base_value"].iloc[0]
    feature_names = obs_data["variable"].values
    shap_values = obs_data["shap_value"].values
    feature_values = obs_data["feature_value"].values

    # Create force plot
    force_plot_obj = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=feature_values,
        feature_names=feature_names,
        matplotlib=matplotlib,
        show=False,
        **kwargs
    )

    if matplotlib:
        # Return matplotlib figure
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
    else:
        # Return HTML
        if show:
            # Display in notebook
            try:
                from IPython.display import display
                display(force_plot_obj)
            except ImportError:
                warnings.warn("IPython not available. Cannot display force plot inline.")
        return force_plot_obj


def dependence_plot(
    shap_df: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = "auto",
    show: bool = True,
    figsize: tuple = (10, 6),
    alpha: float = 0.5,
    **kwargs
) -> plt.Figure:
    """
    Create SHAP dependence plot for a single feature.

    Shows how SHAP values for a feature vary with the feature's value.
    Color can indicate interaction with another feature.

    Args:
        shap_df: SHAP DataFrame from explain() method
        feature: Which feature to plot
        interaction_feature: Feature to use for color coding:
                             - "auto": Auto-select strongest interaction
                             - Feature name: Use specific feature
                             - None: No color coding
        show: Whether to call plt.show()
        figsize: Figure size as (width, height)
        alpha: Transparency of scatter points (0-1)
        **kwargs: Additional arguments passed to SHAP's dependence_plot

    Returns:
        matplotlib.figure.Figure object

    Raises:
        ImportError: If shap package not installed
        ValueError: If feature not found in shap_df

    Examples:
        >>> # Dependence plot with auto-detected interaction
        >>> fig = dependence_plot(shap_df, feature="temperature")
        >>>
        >>> # Specify interaction feature
        >>> fig = dependence_plot(shap_df, feature="price", interaction_feature="demand")
        >>>
        >>> # No interaction coloring
        >>> fig = dependence_plot(shap_df, feature="age", interaction_feature=None)
    """
    # Import SHAP
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package not installed. Install with: pip install shap>=0.43.0"
        )

    # Validate input
    required_cols = ["observation_id", "variable", "shap_value", "feature_value"]
    _validate_shap_dataframe(shap_df, required_cols)

    # Check feature exists
    if feature not in shap_df["variable"].values:
        available = shap_df["variable"].unique().tolist()
        raise ValueError(
            f"Feature '{feature}' not found in shap_df. Available: {available}"
        )

    # Convert to SHAP matrix format
    shap_values, feature_values, feature_names = _convert_to_shap_matrix(shap_df)

    # Get feature index
    feature_idx = feature_names.index(feature)

    # Handle interaction feature
    if interaction_feature == "auto":
        interaction_idx = "auto"
    elif interaction_feature is None:
        interaction_idx = None
    else:
        if interaction_feature not in feature_names:
            raise ValueError(
                f"Interaction feature '{interaction_feature}' not found. "
                f"Available: {feature_names}"
            )
        interaction_idx = feature_names.index(interaction_feature)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create dependence plot
    shap.dependence_plot(
        ind=feature_idx,
        shap_values=shap_values,
        features=feature_values,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        alpha=alpha,
        show=False,
        **kwargs
    )

    # Get current figure
    fig = plt.gcf()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def temporal_plot(
    shap_df: pd.DataFrame,
    features: Optional[Union[str, list]] = None,
    aggregation: Literal["mean", "sum", "abs_mean"] = "mean",
    show: bool = True,
    figsize: tuple = (12, 6),
    plot_type: Literal["line", "area", "bar"] = "line",
    **kwargs
) -> plt.Figure:
    """
    Create temporal plot showing how SHAP values evolve over time.

    Visualizes SHAP values across time for time series models. Useful for
    understanding how feature importance changes over time.

    Args:
        shap_df: SHAP DataFrame from explain() with 'date' column
        features: Features to plot:
                  - None: Plot all features (aggregated)
                  - str: Single feature name
                  - list: Multiple feature names
        aggregation: How to aggregate SHAP values:
                     - "mean": Mean SHAP value per timestep
                     - "sum": Sum of SHAP values per timestep
                     - "abs_mean": Mean absolute SHAP value
        show: Whether to call plt.show()
        figsize: Figure size as (width, height)
        plot_type: Type of plot ("line", "area", or "bar")
        **kwargs: Additional matplotlib arguments (color, alpha, etc.)

    Returns:
        matplotlib.figure.Figure object

    Raises:
        ValueError: If 'date' column not in shap_df or features not found

    Examples:
        >>> # Plot all features aggregated over time
        >>> fig = temporal_plot(shap_df)
        >>>
        >>> # Plot single feature evolution
        >>> fig = temporal_plot(shap_df, features="temperature")
        >>>
        >>> # Plot multiple features
        >>> fig = temporal_plot(shap_df, features=["price", "demand"], plot_type="area")
        >>>
        >>> # Show absolute importance over time
        >>> fig = temporal_plot(shap_df, aggregation="abs_mean")
    """
    # Validate date column exists
    if "date" not in shap_df.columns:
        raise ValueError(
            "SHAP DataFrame must contain 'date' column for temporal plots. "
            "Ensure data passed to explain() had a 'date' column."
        )

    # Validate required columns
    required_cols = ["date", "variable", "shap_value", "observation_id"]
    _validate_shap_dataframe(shap_df, required_cols)

    # Filter features if specified
    if features is not None:
        if isinstance(features, str):
            features = [features]

        # Validate features exist
        available_features = shap_df["variable"].unique()
        missing = set(features) - set(available_features)
        if missing:
            raise ValueError(
                f"Features not found: {missing}. Available: {available_features.tolist()}"
            )

        # Filter data
        plot_data = shap_df[shap_df["variable"].isin(features)].copy()
    else:
        plot_data = shap_df.copy()

    # Aggregate by date and feature
    if aggregation == "mean":
        agg_data = plot_data.groupby(["date", "variable"])["shap_value"].mean().reset_index()
    elif aggregation == "sum":
        agg_data = plot_data.groupby(["date", "variable"])["shap_value"].sum().reset_index()
    elif aggregation == "abs_mean":
        plot_data["abs_shap_value"] = plot_data["shap_value"].abs()
        agg_data = plot_data.groupby(["date", "variable"])["abs_shap_value"].mean().reset_index()
        agg_data = agg_data.rename(columns={"abs_shap_value": "shap_value"})
    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation}. Must be 'mean', 'sum', or 'abs_mean'"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot for plotting
    pivot_data = agg_data.pivot(index="date", columns="variable", values="shap_value")

    # Create plot based on type
    if plot_type == "line":
        pivot_data.plot(ax=ax, **kwargs)
    elif plot_type == "area":
        # Use stacked=False to allow mixed positive/negative SHAP values
        pivot_data.plot.area(ax=ax, alpha=0.7, stacked=False, **kwargs)
    elif plot_type == "bar":
        pivot_data.plot.bar(ax=ax, **kwargs)
    else:
        raise ValueError(
            f"Unknown plot_type: {plot_type}. Must be 'line', 'area', or 'bar'"
        )

    # Customize plot
    ax.set_xlabel("Date", fontsize=12)

    ylabel = {
        "mean": "Mean SHAP Value",
        "sum": "Sum of SHAP Values",
        "abs_mean": "Mean |SHAP Value|"
    }[aggregation]
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_title("SHAP Values Over Time", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


# Helper functions
def _validate_shap_dataframe(shap_df: pd.DataFrame, required_cols: list):
    """
    Validate that SHAP DataFrame has required columns.

    Args:
        shap_df: DataFrame to validate
        required_cols: List of required column names

    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if shap_df is None or len(shap_df) == 0:
        raise ValueError("SHAP DataFrame is empty")

    missing_cols = set(required_cols) - set(shap_df.columns)
    if missing_cols:
        raise ValueError(
            f"SHAP DataFrame missing required columns: {missing_cols}. "
            f"Expected columns from explain() method."
        )


def _convert_to_shap_matrix(shap_df: pd.DataFrame) -> tuple:
    """
    Convert long-format SHAP DataFrame to matrix format for SHAP plots.

    Args:
        shap_df: SHAP DataFrame in long format (from explain())

    Returns:
        Tuple of (shap_values, feature_values, feature_names):
        - shap_values: 2D array (n_observations, n_features)
        - feature_values: 2D array (n_observations, n_features)
        - feature_names: List of feature names

    Examples:
        >>> shap_values, feature_values, feature_names = _convert_to_shap_matrix(shap_df)
        >>> print(shap_values.shape)  # (100, 5) for 100 obs, 5 features
    """
    # Get unique observations and features
    observation_ids = sorted(shap_df["observation_id"].unique())
    feature_names = shap_df.groupby("variable")["abs_shap"].mean().sort_values(ascending=False).index.tolist()

    n_obs = len(observation_ids)
    n_features = len(feature_names)

    # Initialize matrices
    shap_values = np.zeros((n_obs, n_features))
    feature_values = np.zeros((n_obs, n_features))

    # Fill matrices
    for i, obs_id in enumerate(observation_ids):
        obs_data = shap_df[shap_df["observation_id"] == obs_id]

        for j, feature in enumerate(feature_names):
            feature_data = obs_data[obs_data["variable"] == feature]

            if len(feature_data) > 0:
                shap_values[i, j] = feature_data["shap_value"].iloc[0]
                feature_values[i, j] = feature_data["feature_value"].iloc[0]

    return shap_values, feature_values, feature_names
