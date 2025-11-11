"""
Core Recipe and PreparedRecipe classes for feature engineering

Recipes specify a sequence of preprocessing steps to be applied to data.
They are fitted on training data (prep) and then applied to new data (bake).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Union, Callable
import pandas as pd
import numpy as np


def _get_datetime_columns(data: pd.DataFrame) -> List[str]:
    """
    Get list of datetime columns in dataframe.

    Used internally to exclude datetime columns from normalization/scaling
    operations, as these should be handled by specialized time feature
    extraction steps instead.

    Args:
        data: DataFrame to check for datetime columns

    Returns:
        List of column names with datetime dtype
    """
    return [c for c in data.columns
            if pd.api.types.is_datetime64_any_dtype(data[c])]


class RecipeStep(Protocol):
    """
    Protocol for recipe steps.

    All recipe steps must implement prep() and return a PreparedStep.
    """

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStep":
        """
        Fit the step to training data.

        Args:
            data: Training data
            training: Whether this is training data (vs application)

        Returns:
            PreparedStep ready to bake new data
        """
        ...


class PreparedStep(Protocol):
    """
    Protocol for fitted recipe steps.

    All prepared steps must implement bake() to transform data.
    """

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted transformation to new data.

        Args:
            data: Data to transform

        Returns:
            Transformed DataFrame
        """
        ...


@dataclass
class Recipe:
    """
    Feature engineering specification.

    A Recipe is a specification of preprocessing steps that can be:
    1. Fitted to training data via prep()
    2. Applied to new data via PreparedRecipe.bake()

    Recipes ensure consistent preprocessing between training and test data,
    preventing data leakage.

    Attributes:
        steps: List of RecipeStep objects to apply
        template: Optional template DataFrame for role inference
        roles: Dictionary mapping role names to column lists

    Examples:
        >>> # Create recipe
        >>> rec = (
        ...     Recipe()
        ...     .step_normalize(["feature1", "feature2"])
        ...     .step_dummy(["category"])
        ... )
        >>>
        >>> # Fit to training data
        >>> rec_fit = rec.prep(train_data)
        >>>
        >>> # Apply to test data
        >>> test_transformed = rec_fit.bake(test_data)
    """

    steps: List[Any] = field(default_factory=list)
    template: Optional[pd.DataFrame] = None
    roles: Dict[str, List[str]] = field(default_factory=dict)

    def add_step(self, step: RecipeStep) -> "Recipe":
        """
        Add a preprocessing step to the recipe.

        Args:
            step: RecipeStep to add

        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self

    def step_rm(
        self,
        columns: Union[str, List[str], Callable]
    ) -> "Recipe":
        """
        Remove/drop columns from the dataset.

        This step removes specified columns from the data. Useful for
        removing columns that shouldn't be used in modeling (like IDs, dates,
        or other non-predictive features).

        Args:
            columns: Column(s) to remove. Can be a string, list of strings,
                    or selector function

        Returns:
            Self for method chaining

        Examples:
            >>> # Remove single column
            >>> rec = recipe().step_rm("date")
            >>>
            >>> # Remove multiple columns
            >>> rec = recipe().step_rm(["id", "date", "timestamp"])
        """
        from py_recipes.steps.remove import StepRm
        return self.add_step(StepRm(columns=columns))

    def step_select(
        self,
        columns: Union[str, List[str], Callable]
    ) -> "Recipe":
        """
        Select (keep) only specified columns from the dataset.

        This is the inverse of step_rm() - it keeps only the specified
        columns and removes everything else.

        Args:
            columns: Column(s) to keep. Can be a string, list of strings,
                    or selector function

        Returns:
            Self for method chaining

        Examples:
            >>> # Keep only specific columns
            >>> rec = recipe().step_select(["feature1", "feature2", "target"])
        """
        from py_recipes.steps.remove import StepSelect
        return self.add_step(StepSelect(columns=columns))

    def step_normalize(
        self,
        columns: Optional[List[str]] = None,
        method: str = "zscore"
    ) -> "Recipe":
        """
        Normalize numeric columns.

        Centers and scales numeric features using StandardScaler (zscore)
        or MinMaxScaler (minmax).

        Args:
            columns: Columns to normalize (None = all numeric)
            method: Normalization method ("zscore" or "minmax")

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_normalize(["feature1", "feature2"])
            >>> rec = Recipe().step_normalize(method="minmax")
        """
        from py_recipes.steps.normalize import StepNormalize
        return self.add_step(StepNormalize(columns=columns, method=method))

    def step_dummy(
        self,
        columns: Union[List[str], Callable],
        one_hot: bool = True
    ) -> "Recipe":
        """
        Create dummy variables from categorical columns.

        Converts categorical variables to numeric using one-hot encoding
        or integer encoding.

        Args:
            columns: Categorical columns to encode (list or selector function)
            one_hot: Use one-hot encoding (True) or integer encoding (False)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_dummy(["category", "group"])
            >>> rec = Recipe().step_dummy(all_nominal_predictors())
        """
        from py_recipes.steps.dummy import StepDummy
        return self.add_step(StepDummy(columns=columns, one_hot=one_hot))

    def step_impute_mean(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Impute missing values using mean.

        Replaces NA values in numeric columns with the training mean.

        Args:
            columns: Columns to impute (None = all numeric with NA)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_impute_mean(["feature1", "feature2"])
        """
        from py_recipes.steps.impute import StepImputeMean
        return self.add_step(StepImputeMean(columns=columns))

    def step_impute_median(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Impute missing values using median.

        Replaces NA values in numeric columns with the training median.

        Args:
            columns: Columns to impute (None = all numeric with NA)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.impute import StepImputeMedian
        return self.add_step(StepImputeMedian(columns=columns))

    def step_mutate(
        self,
        transformations: Dict[str, callable]
    ) -> "Recipe":
        """
        Create or modify columns using custom functions.

        Args:
            transformations: Dict mapping column names to transformation functions

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_mutate({
            ...     "log_feature": lambda df: np.log(df["feature"] + 1),
            ...     "interaction": lambda df: df["x1"] * df["x2"]
            ... })
        """
        from py_recipes.steps.mutate import StepMutate
        return self.add_step(StepMutate(transformations=transformations))

    def step_lag(
        self,
        columns: List[str],
        lags: List[int]
    ) -> "Recipe":
        """
        Create lag features for time series data.

        Creates lagged versions of specified columns to capture temporal patterns.

        Args:
            columns: Columns to create lags for
            lags: List of lag periods (e.g., [1, 2, 7] for 1-day, 2-day, 7-day lags)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_lag(["sales", "price"], lags=[1, 7, 30])
        """
        from py_recipes.steps.timeseries import StepLag
        return self.add_step(StepLag(columns=columns, lags=lags))

    def step_diff(
        self,
        columns: Optional[List[str]] = None,
        lag: int = 1,
        differences: int = 1
    ) -> "Recipe":
        """
        Create differenced features.

        Computes differences between consecutive observations to make
        time series stationary.

        Args:
            columns: Columns to difference (None = all numeric)
            lag: Period for differencing (default 1)
            differences: Number of times to difference (default 1)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_diff(["sales"], lag=1)
            >>> rec = Recipe().step_diff(["price"], lag=7, differences=2)
        """
        from py_recipes.steps.timeseries import StepDiff
        return self.add_step(StepDiff(columns=columns, lag=lag, differences=differences))

    def step_pct_change(
        self,
        columns: Optional[List[str]] = None,
        periods: int = 1
    ) -> "Recipe":
        """
        Create percent change features.

        Computes percentage changes between consecutive observations.

        Args:
            columns: Columns to compute percent changes for (None = all numeric)
            periods: Number of periods for change calculation (default 1)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_pct_change(["sales", "revenue"])
            >>> rec = Recipe().step_pct_change(["price"], periods=7)
        """
        from py_recipes.steps.timeseries import StepPctChange
        return self.add_step(StepPctChange(columns=columns, periods=periods))

    def step_rolling(
        self,
        columns: List[str],
        window: int,
        stats: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Create rolling window statistics.

        Computes statistics over rolling windows (mean, std, min, max, sum).

        Args:
            columns: Columns to compute rolling stats for
            window: Size of rolling window
            stats: Statistics to compute (default ["mean"])
                   Options: "mean", "std", "min", "max", "sum"

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_rolling(["sales"], window=7, stats=["mean", "std"])
            >>> rec = Recipe().step_rolling(["price"], window=30, stats=["min", "max"])
        """
        from py_recipes.steps.timeseries import StepRolling
        return self.add_step(StepRolling(columns=columns, window=window, stats=stats))

    def step_date(
        self,
        column: str,
        features: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Extract date/time features from datetime columns.

        Creates features like year, month, day, dayofweek, quarter, etc.

        Args:
            column: Datetime column to extract features from
            features: List of features to extract (default ["year", "month", "day", "dayofweek"])
                     Options: "year", "month", "day", "dayofweek", "dayofyear", "quarter",
                             "week", "hour", "minute", "is_weekend", "is_month_start",
                             "is_month_end", "is_quarter_start", "is_quarter_end",
                             "is_year_start", "is_year_end"

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_date("date", features=["year", "month", "dayofweek"])
            >>> rec = Recipe().step_date("timestamp", features=["hour", "is_weekend"])
        """
        from py_recipes.steps.timeseries import StepDate
        return self.add_step(StepDate(column=column, features=features))

    def step_pca(
        self,
        columns: Optional[List[str]] = None,
        num_comp: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> "Recipe":
        """
        Principal Component Analysis (PCA) transformation.

        Reduces dimensionality by projecting data onto principal components.

        Args:
            columns: Columns to apply PCA to (None = all numeric)
            num_comp: Number of components to keep
            threshold: Variance threshold (alternative to num_comp)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_pca(num_comp=5)
            >>> rec = Recipe().step_pca(["x1", "x2", "x3"], num_comp=2)
            >>> rec = Recipe().step_pca(threshold=0.95)  # Keep 95% of variance
        """
        from py_recipes.steps.feature_selection import StepPCA
        return self.add_step(StepPCA(columns=columns, num_comp=num_comp, threshold=threshold))

    def step_select_corr(
        self,
        outcome: str,
        threshold: float = 0.9,
        method: str = "multicollinearity"
    ) -> "Recipe":
        """
        Select features based on correlation.

        Removes features with low correlation to the outcome variable or
        high correlation with other predictors (multicollinearity).

        Args:
            outcome: Outcome column name
            threshold: Correlation threshold (default 0.9 for multicollinearity)
            method: Selection method - "multicollinearity" or "outcome"

        Returns:
            Self for method chaining

        Examples:
            >>> # Remove highly correlated predictors
            >>> rec = Recipe().step_select_corr("sales", threshold=0.9, method="multicollinearity")
            >>> # Keep only features correlated with outcome
            >>> rec = Recipe().step_select_corr("sales", threshold=0.3, method="outcome")
        """
        from py_recipes.steps.feature_selection import StepSelectCorr
        return self.add_step(StepSelectCorr(outcome=outcome, threshold=threshold, method=method))

    # ========== Transformation Steps ==========

    def step_log(
        self,
        columns: Optional[List[str]] = None,
        base: float = np.e,
        offset: float = 0.0,
        signed: bool = False,
        inplace: bool = True
    ) -> "Recipe":
        """
        Apply logarithmic transformation.

        Args:
            columns: Columns to transform (None = all numeric)
            base: Logarithm base (default: natural log)
            offset: Value added before transformation (default: 0)
            signed: If True, preserves sign (default: False)
            inplace: If True, replace original columns; if False, create new columns with suffix (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.transformations import StepLog
        return self.add_step(StepLog(columns=columns, base=base, offset=offset, signed=signed, inplace=inplace))

    def step_sqrt(
        self,
        columns: Optional[List[str]] = None,
        inplace: bool = True
    ) -> "Recipe":
        """
        Apply square root transformation.

        Args:
            columns: Columns to transform (None = all numeric)
            inplace: If True, replace original columns; if False, create new columns with suffix (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.transformations import StepSqrt
        return self.add_step(StepSqrt(columns=columns, inplace=inplace))

    def step_boxcox(
        self,
        columns: Optional[List[str]] = None,
        lambdas: Optional[Dict[str, float]] = None,
        inplace: bool = True
    ) -> "Recipe":
        """
        Apply Box-Cox power transformation.

        Args:
            columns: Columns to transform (None = all numeric)
            lambdas: Optional dict of lambda parameters
            inplace: If True, replace original columns; if False, create new columns with suffix (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.transformations import StepBoxCox
        return self.add_step(StepBoxCox(columns=columns, lambdas=lambdas, inplace=inplace))

    def step_yeojohnson(
        self,
        columns: Optional[List[str]] = None,
        lambdas: Optional[Dict[str, float]] = None,
        inplace: bool = True
    ) -> "Recipe":
        """
        Apply Yeo-Johnson power transformation.

        Args:
            columns: Columns to transform (None = all numeric)
            lambdas: Optional dict of lambda parameters
            inplace: If True, replace original columns; if False, create new columns with suffix (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.transformations import StepYeoJohnson
        return self.add_step(StepYeoJohnson(columns=columns, lambdas=lambdas, inplace=inplace))

    def step_inverse(
        self,
        columns: Optional[List[str]] = None,
        offset: float = 0.0,
        inplace: bool = True
    ) -> "Recipe":
        """
        Apply inverse transformation (1/x).

        Args:
            columns: Columns to transform (None = all numeric)
            offset: Value added before inversion to avoid division by zero (default: 0)
            inplace: If True, replace original columns; if False, create new columns with suffix (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.transformations import StepInverse
        return self.add_step(StepInverse(columns=columns, offset=offset, inplace=inplace))

    # ========== Scaling Steps ==========

    def step_center(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Center numeric columns to have mean zero.

        Args:
            columns: Columns to center (None = all numeric)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.scaling import StepCenter
        return self.add_step(StepCenter(columns=columns))

    def step_scale(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Scale numeric columns to have standard deviation of one.

        Args:
            columns: Columns to scale (None = all numeric)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.scaling import StepScale
        return self.add_step(StepScale(columns=columns))

    def step_range(
        self,
        columns: Optional[List[str]] = None,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> "Recipe":
        """
        Scale numeric columns to a custom range.

        Args:
            columns: Columns to scale (None = all numeric)
            min_val: Minimum value of scaled range
            max_val: Maximum value of scaled range

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.scaling import StepRange
        return self.add_step(StepRange(columns=columns, min_val=min_val, max_val=max_val))

    # ========== Filter Steps ==========

    def step_zv(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Remove zero variance columns.

        Args:
            columns: Columns to check (None = all numeric)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filters import StepZv
        return self.add_step(StepZv(columns=columns))

    def step_nzv(
        self,
        columns: Optional[List[str]] = None,
        freq_cut: float = 19.0,
        unique_cut: float = 10.0
    ) -> "Recipe":
        """
        Remove near-zero variance columns.

        Args:
            columns: Columns to check (None = all numeric)
            freq_cut: Frequency ratio threshold
            unique_cut: Unique value percentage threshold

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filters import StepNzv
        return self.add_step(StepNzv(columns=columns, freq_cut=freq_cut, unique_cut=unique_cut))

    def step_lincomb(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 1e-5
    ) -> "Recipe":
        """
        Remove linearly dependent columns.

        Args:
            columns: Columns to check (None = all numeric)
            threshold: Tolerance for linear dependency detection

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filters import StepLinComb
        return self.add_step(StepLinComb(columns=columns, threshold=threshold))

    def step_filter_missing(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> "Recipe":
        """
        Remove columns with high proportion of missing values.

        Args:
            columns: Columns to check (None = all columns)
            threshold: Maximum proportion of missing values

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filters import StepFilterMissing
        return self.add_step(StepFilterMissing(columns=columns, threshold=threshold))

    def step_filter_anova(
        self,
        outcome: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        use_pvalue: bool = True
    ) -> "Recipe":
        """
        Filter features using ANOVA F-test.

        Args:
            outcome: Outcome column name
            threshold: Minimum score to keep (either F-statistic or -log10(p-value))
            top_n: Keep top N features
            top_p: Keep top proportion of features (0-1)
            use_pvalue: Use -log10(p-value) if True, else F-statistic

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filter_supervised import StepFilterAnova
        return self.add_step(StepFilterAnova(
            outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p, use_pvalue=use_pvalue
        ))

    def step_filter_rf_importance(
        self,
        outcome: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        trees: int = 100,
        mtry: Optional[int] = None,
        min_n: int = 2
    ) -> "Recipe":
        """
        Filter features using Random Forest feature importance.

        Args:
            outcome: Outcome column name
            threshold: Minimum importance score to keep
            top_n: Keep top N features
            top_p: Keep top proportion of features (0-1)
            trees: Number of trees in random forest
            mtry: Number of variables to sample at each split
            min_n: Minimum number of samples in leaf nodes

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filter_supervised import StepFilterRfImportance
        return self.add_step(StepFilterRfImportance(
            outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
            trees=trees, mtry=mtry, min_n=min_n
        ))

    def step_filter_mutual_info(
        self,
        outcome: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        n_neighbors: int = 3
    ) -> "Recipe":
        """
        Filter features using mutual information (information gain).

        Args:
            outcome: Outcome column name
            threshold: Minimum mutual information score to keep
            top_n: Keep top N features
            top_p: Keep top proportion of features (0-1)
            n_neighbors: Number of neighbors for MI estimation

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filter_supervised import StepFilterMutualInfo
        return self.add_step(StepFilterMutualInfo(
            outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p, n_neighbors=n_neighbors
        ))

    def step_filter_roc_auc(
        self,
        outcome: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        multiclass_strategy: str = 'ovr'
    ) -> "Recipe":
        """
        Filter features using ROC AUC scores (classification only).

        Args:
            outcome: Outcome column name
            threshold: Minimum ROC AUC score to keep
            top_n: Keep top N features
            top_p: Keep top proportion of features (0-1)
            multiclass_strategy: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filter_supervised import StepFilterRocAuc
        return self.add_step(StepFilterRocAuc(
            outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
            multiclass_strategy=multiclass_strategy
        ))

    def step_filter_chisq(
        self,
        outcome: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        method: str = 'chisq',
        use_pvalue: bool = True
    ) -> "Recipe":
        """
        Filter features using chi-squared or Fisher exact test.

        Args:
            outcome: Outcome column name
            threshold: Minimum score to keep
            top_n: Keep top N features
            top_p: Keep top proportion of features (0-1)
            method: 'chisq' or 'fisher'
            use_pvalue: Use -log10(p-value) if True, else test statistic

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.filter_supervised import StepFilterChisq
        return self.add_step(StepFilterChisq(
            outcome=outcome, threshold=threshold, top_n=top_n, top_p=top_p,
            method=method, use_pvalue=use_pvalue
        ))

    def step_select_shap(
        self,
        outcome: str,
        model: Any,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        shap_samples: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> "Recipe":
        """
        Filter features using SHAP (SHapley Additive exPlanations) values.

        Uses SHAP values to measure feature importance. Works with tree-based
        models (fast TreeExplainer) and other models (slower KernelExplainer).

        Args:
            outcome: Outcome column name
            model: Trained scikit-learn compatible model to explain
            threshold: Minimum absolute SHAP value to keep feature
            top_n: Keep top N features by mean absolute SHAP value
            top_p: Keep top proportion of features (0-1)
            shap_samples: Number of samples for SHAP calculation (None = all)
            random_state: Random seed for sampling

        Returns:
            Self for method chaining

        Examples:
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> rf = RandomForestRegressor(n_estimators=100, random_state=42)
            >>> rf.fit(X_train, y_train)
            >>> rec = recipe(data, "price ~ .").step_select_shap(
            ...     outcome='price', model=rf, top_n=10, shap_samples=500
            ... )

        Notes:
            Requires shap package: pip install shap
        """
        from py_recipes.steps.filter_supervised import StepSelectShap
        return self.add_step(StepSelectShap(
            outcome=outcome, model=model, threshold=threshold, top_n=top_n,
            top_p=top_p, shap_samples=shap_samples, random_state=random_state
        ))

    def step_select_permutation(
        self,
        outcome: str,
        model: Any,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        top_p: Optional[float] = None,
        n_repeats: int = 10,
        scoring: Optional[Union[str, Callable]] = None,
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ) -> "Recipe":
        """
        Filter features using permutation importance.

        Measures feature importance by shuffling each feature and measuring
        the resulting decrease in model performance. Model-agnostic.

        Args:
            outcome: Outcome column name
            model: Trained scikit-learn compatible model to evaluate
            threshold: Minimum permutation importance to keep feature
            top_n: Keep top N features by permutation importance
            top_p: Keep top proportion of features (0-1)
            n_repeats: Number of times to permute each feature (default: 10)
            scoring: Scoring metric (e.g., 'r2', 'neg_mean_squared_error', 'accuracy')
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            Self for method chaining

        Examples:
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> rf = RandomForestRegressor(n_estimators=100, random_state=42)
            >>> rf.fit(X_train, y_train)
            >>> rec = recipe(data, "price ~ .").step_select_permutation(
            ...     outcome='price', model=rf, top_n=15, n_repeats=10
            ... )

        Notes:
            Computationally expensive (n_repeats × n_features model evaluations).
            Parallel execution recommended for large datasets.
        """
        from py_recipes.steps.filter_supervised import StepSelectPermutation
        return self.add_step(StepSelectPermutation(
            outcome=outcome, model=model, threshold=threshold, top_n=top_n,
            top_p=top_p, n_repeats=n_repeats, scoring=scoring,
            random_state=random_state, n_jobs=n_jobs
        ))

    def step_splitwise(
        self,
        outcome: str,
        transformation_mode: str = 'univariate',
        min_support: float = 0.1,
        min_improvement: float = 3.0,
        criterion: str = 'AIC',
        feature_type: str = 'dummies',
        exclude_vars: Optional[List[str]] = None,
        columns: Union[None, str, List[str], Callable] = None
    ) -> "Recipe":
        """
        Adaptive dummy encoding for numeric predictors using shallow decision trees.

        SplitWise automatically transforms numeric predictors into either binary
        dummy variables (with 1 or 2 split points) or keeps them linear based on
        AIC/BIC improvement. This is a supervised transformation step.

        Args:
            outcome: Outcome column name (required for supervised transformation)
            transformation_mode: 'univariate' (independent) or 'iterative' (adaptive)
            min_support: Minimum fraction of observations in each dummy group (0-0.5)
            min_improvement: Minimum AIC/BIC improvement to prefer dummy over linear
            criterion: 'AIC' or 'BIC' for model selection
            feature_type: 'dummies' (binary only), 'interactions' (dummy*value), or 'both'
            exclude_vars: Variables forced to stay linear (no transformation)
            columns: Columns to consider (None = all numeric except outcome)

        Returns:
            Self for method chaining

        Examples:
            >>> # Basic usage - dummies only
            >>> rec = recipe().step_splitwise(outcome='price')
            >>>
            >>> # With interactions
            >>> rec = recipe().step_splitwise(
            ...     outcome='sales',
            ...     feature_type='interactions'
            ... )
            >>>
            >>> # Both dummies and interactions
            >>> rec = recipe().step_splitwise(
            ...     outcome='price',
            ...     feature_type='both',
            ...     min_support=0.15
            ... )

        References:
            Kurbucz et al. (2025). SplitWise Regression: Stepwise Modeling with
            Adaptive Dummy Encoding. arXiv:2505.15423
        """
        from py_recipes.steps.splitwise import StepSplitwise
        return self.add_step(StepSplitwise(
            outcome=outcome,
            transformation_mode=transformation_mode,
            min_support=min_support,
            min_improvement=min_improvement,
            criterion=criterion,
            feature_type=feature_type,
            exclude_vars=exclude_vars,
            columns=columns
        ))

    def step_safe(
        self,
        surrogate_model,
        outcome: str,
        penalty: float = 3.0,
        pelt_model: str = 'l2',
        no_changepoint_strategy: str = 'median',
        feature_type: str = 'dummies',
        keep_original_cols: bool = False,
        top_n: Optional[int] = None,
        grid_resolution: int = 1000
    ) -> "Recipe":
        """
        Surrogate Assisted Feature Extraction (SAFE) for interpretable models.

        SAFE uses a complex surrogate model to guide feature transformation,
        creating interpretable features by:
        - Detecting changepoints in numeric variable partial dependence plots
        - Merging similar categorical levels via hierarchical clustering

        The transformed features retain information from the complex surrogate
        while being more interpretable for simpler models.

        Args:
            surrogate_model: Pre-fitted surrogate model (e.g., GradientBoostingRegressor)
                Must implement predict() for regression or predict_proba() for classification
            outcome: Name of outcome variable (required for supervised transformation)
            penalty: Penalty for adding changepoints (default: 3.0).
                Higher values = fewer intervals. Typical range: 0.1-10.0
            pelt_model: Cost function for Pelt algorithm ('l2', 'l1', 'rbf')
            no_changepoint_strategy: Strategy when no changepoint detected:
                - 'median': Create one split at median
                - 'drop': Remove feature from output
            feature_type: 'dummies' (binary only), 'interactions' (dummy*value), or 'both'
            keep_original_cols: Whether to keep original columns alongside
                transformed features (default: False)
            top_n: If specified, select only top N most important transformed
                features based on variance explained in surrogate predictions
            grid_resolution: Number of points for partial dependence grid (default: 1000)

        Returns:
            Recipe with step_safe added

        Examples:
            >>> from sklearn.ensemble import GradientBoostingRegressor
            >>> surrogate = GradientBoostingRegressor(n_estimators=100)
            >>> surrogate.fit(train_data.drop('target', axis=1), train_data['target'])
            >>>
            >>> # Basic usage - dummies only
            >>> rec = recipe().step_safe(
            ...     surrogate_model=surrogate,
            ...     outcome='target',
            ...     penalty=3.0
            ... )
            >>>
            >>> # With interactions
            >>> rec = recipe().step_safe(
            ...     surrogate_model=surrogate,
            ...     outcome='target',
            ...     feature_type='interactions',
            ...     top_n=10
            ... )
            >>>
            >>> # Both dummies and interactions
            >>> rec = recipe().step_safe(
            ...     surrogate_model=surrogate,
            ...     outcome='target',
            ...     feature_type='both'
            ... )

        Notes:
            - Requires ruptures, scipy, and kneed packages
            - Surrogate model must be pre-fitted
            - Numeric features: changepoint detection via Pelt algorithm
            - Categorical features: hierarchical clustering (Ward linkage)
            - Output is one-hot encoded with p-1 scheme

        References:
            SAFE library: https://github.com/ModelOriented/SAFE

        Deprecation:
            step_safe() now uses StepSafeV2 internally. Parameters pelt_model and
            no_changepoint_strategy are ignored. Consider using step_safe_v2() directly
            for more control and clarity.
        """
        import warnings
        from py_recipes.steps.feature_extraction import StepSafeV2

        # Deprecation warning
        warnings.warn(
            "step_safe() is deprecated and now uses step_safe_v2() internally. "
            "Parameters 'pelt_model' and 'no_changepoint_strategy' are ignored. "
            "Consider using step_safe_v2() directly for more control. "
            "Old step_safe() with PELT will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )

        # Parameter mapping:
        # - Old feature_type (dummies/interactions/both) → new output_mode
        # - Set new feature_type='both' (always process numeric + categorical)
        # - Ignore pelt_model and no_changepoint_strategy (not in V2)

        # Map penalty default: old=3.0, new=10.0
        # If user didn't specify (using default 3.0), use new default 10.0
        if penalty == 3.0:
            penalty_v2 = 10.0
        else:
            penalty_v2 = penalty

        # Map grid_resolution default: old=1000, new=100
        if grid_resolution == 1000:
            grid_resolution_v2 = 100
        else:
            grid_resolution_v2 = grid_resolution

        return self.add_step(StepSafeV2(
            surrogate_model=surrogate_model,
            outcome=outcome,
            penalty=penalty_v2,
            top_n=top_n,
            max_thresholds=5,  # V2 default
            keep_original_cols=keep_original_cols,
            grid_resolution=grid_resolution_v2,
            feature_type='both',  # V2: process both numeric and categorical
            output_mode=feature_type,  # OLD feature_type → NEW output_mode
            columns=None
        ))

    def step_safe_v2(
        self,
        surrogate_model,
        outcome: str,
        penalty: float = 10.0,
        top_n: Optional[int] = None,
        max_thresholds: int = 5,
        keep_original_cols: bool = True,
        grid_resolution: int = 100,
        feature_type: str = 'both',
        output_mode: str = 'dummies',
        importance_method: str = 'lasso',
        columns=None
    ) -> "Recipe":
        """
        SAFE v2: Surrogate Assisted Feature Extraction with UNFITTED model.

        This version accepts an UNFITTED surrogate model (fitted during prep()),
        adds max_thresholds parameter to control threshold quantity, sanitizes
        feature names for compatibility, and recalculates importances
        on TRANSFORMED features using multiple methods.

        Key differences from step_safe:
        - Accepts UNFITTED surrogate model (fitted during prep)
        - Adds max_thresholds parameter (default=5)
        - Sanitizes feature names with regex
        - Recalculates importances on TRANSFORMED features using lasso/ridge/permutation/hybrid

        Args:
            surrogate_model: UNFITTED sklearn-compatible model (will be fitted during prep)
            outcome: Name of outcome variable (required)
            penalty: Changepoint penalty (default: 10.0). Higher = fewer thresholds
            top_n: Select top N most important TRANSFORMED features (None = keep all)
            max_thresholds: Maximum thresholds per numeric feature (default: 5)
            keep_original_cols: Keep original features alongside transformations (default: True)
            grid_resolution: PDP grid points (default: 100)
            feature_type: Which variable types to process - 'numeric', 'categorical', or 'both' (default: 'both')
            output_mode: Type of features to create - 'dummies', 'interactions', or 'both' (default: 'dummies')
            importance_method: Method for feature importance - 'lasso', 'ridge', 'permutation', or 'hybrid' (default: 'lasso')
            columns: Which columns to transform (None = all except outcome)

        Returns:
            Recipe with step_safe_v2 added

        Examples:
            >>> from sklearn.ensemble import GradientBoostingRegressor
            >>>
            >>> # UNFITTED model - fitted during prep()
            >>> surrogate = GradientBoostingRegressor(n_estimators=100)
            >>>
            >>> # Basic usage
            >>> rec = recipe().step_safe_v2(
            ...     surrogate_model=surrogate,
            ...     outcome='target',
            ...     penalty=10.0,
            ...     max_thresholds=5
            ... )
            >>>
            >>> # Select top 10 most important transformed features
            >>> rec = recipe().step_safe_v2(
            ...     surrogate_model=surrogate,
            ...     outcome='target',
            ...     top_n=10,
            ...     keep_original_cols=False
            ... )

        Notes:
            - Requires ruptures package (pip install ruptures)
            - Surrogate model fitted during prep() - do NOT fit beforehand
            - Creates binary threshold features (feature > threshold)
            - Feature names sanitized for compatibility
            - Importances calculated on TRANSFORMED features using lasso/ridge/permutation/hybrid
        """
        from py_recipes.steps.feature_extraction import StepSafeV2

        return self.add_step(StepSafeV2(
            surrogate_model=surrogate_model,
            outcome=outcome,
            penalty=penalty,
            top_n=top_n,
            max_thresholds=max_thresholds,
            keep_original_cols=keep_original_cols,
            grid_resolution=grid_resolution,
            feature_type=feature_type,
            output_mode=output_mode,
            importance_method=importance_method,
            columns=columns
        ))

    def step_eix(
        self,
        tree_model,
        outcome: str,
        option: str = 'both',
        top_n: Optional[int] = None,
        min_gain: float = 0.0,
        create_interactions: bool = True,
        keep_original_cols: bool = False
    ) -> "Recipe":
        """
        EIX - Explain Interactions in XGBoost/LightGBM for feature selection.

        Analyzes tree structure from XGBoost or LightGBM models to identify
        important variable interactions and creates interaction features based
        on tree model gain.

        Args:
            tree_model: Pre-fitted XGBoost or LightGBM model (REQUIRED)
            outcome: Outcome variable name (REQUIRED for data validation)
            option: What to extract - 'variables', 'interactions', or 'both' (default: 'both')
            top_n: Select top N most important features/interactions (None = keep all)
            min_gain: Minimum sumGain threshold for keeping features (default: 0.0)
            create_interactions: Whether to create interaction features (parent × child) (default: True)
            keep_original_cols: Keep original columns alongside EIX features (default: False)

        Returns:
            Self for method chaining

        Notes:
            - Requires pre-fitted XGBoost or LightGBM model
            - Model must be trained on the same variables in the data
            - Creates interaction features by multiplying parent × child variables
            - Interactions are identified where child gain > parent gain
            - Strong interactions indicate the child variable adds significant
              information beyond the parent variable

        Algorithm:
            1. Extract tree structure from XGBoost/LightGBM model
            2. For each tree, analyze parent-child node relationships
            3. Identify strong interactions: child gain > parent gain
            4. Calculate importance metrics: sumGain, frequency, meanGain
            5. Select top features/interactions by importance
            6. Create interaction features: parent × child

        Examples:
            >>> from xgboost import XGBRegressor
            >>> from py_recipes import recipe
            >>>
            >>> # Fit tree model (REQUIRED)
            >>> tree_model = XGBRegressor(n_estimators=100, max_depth=3)
            >>> tree_model.fit(X_train, y_train)
            >>>
            >>> # Basic usage - find and create top interactions
            >>> rec = recipe().step_eix(
            ...     tree_model=tree_model,
            ...     outcome='target',
            ...     option='interactions',
            ...     top_n=10
            ... )
            >>>
            >>> # Conservative - only strong interactions
            >>> rec = recipe().step_eix(
            ...     tree_model=tree_model,
            ...     outcome='sales',
            ...     option='interactions',
            ...     min_gain=0.1,
            ...     top_n=5
            ... )
            >>>
            >>> # Select important variables only (no interactions)
            >>> rec = recipe().step_eix(
            ...     tree_model=tree_model,
            ...     outcome='revenue',
            ...     option='variables',
            ...     top_n=15,
            ...     create_interactions=False
            ... )
            >>>
            >>> # Both variables and interactions
            >>> rec = recipe().step_eix(
            ...     tree_model=tree_model,
            ...     outcome='target',
            ...     option='both',
            ...     top_n=20,
            ...     min_gain=0.05
            ... )
            >>>
            >>> # Inspect importance after prep
            >>> prepped = rec.prep(train_data)
            >>> eix_step = prepped.prepared_steps[0]
            >>>
            >>> # Get importance table
            >>> importance = eix_step.get_importance()
            >>> print(importance)
            >>>
            >>> # Get interactions to be created
            >>> interactions = eix_step.get_interactions()
            >>> for inter in interactions:
            ...     print(f"{inter['parent']} × {inter['child']} → {inter['name']}")

        Use cases:
            - Feature selection based on tree model analysis
            - Identifying important variable interactions
            - Creating interaction features for linear models
            - Transfer knowledge from tree models to simpler models
            - Understanding which variable pairs are most informative

        Comparison:
            - vs. step_interact(): EIX uses tree model gain (data-driven),
              step_interact() creates all pairwise interactions (exhaustive)
            - vs. step_safe(): EIX uses tree structure directly,
              SAFE uses partial dependence plots (PDP)
            - vs. step_poly(): EIX creates multiplicative interactions,
              step_poly() creates polynomial features

        Dependencies:
            Requires: xgboost or lightgbm

            >>> pip install xgboost lightgbm
        """
        from py_recipes.steps.interaction_detection import StepEIX

        return self.add_step(StepEIX(
            tree_model=tree_model,
            outcome=outcome,
            option=option,
            top_n=top_n,
            min_gain=min_gain,
            create_interactions=create_interactions,
            keep_original_cols=keep_original_cols
        ))

    def step_naomit(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Remove rows with missing values.

        Filters out rows containing NA/NaN values in specified columns.
        Commonly used after creating lag features which introduce NaN at
        the beginning of time series.

        Args:
            columns: Columns to check for NAs (None = check all columns)

        Returns:
            Self for method chaining

        Examples:
            >>> # Remove rows with any NA values
            >>> recipe().step_lag(['value'], lags=[1, 7]).step_naomit()
            >>>
            >>> # Remove rows with NA in specific lag columns
            >>> recipe().step_naomit(columns=['value_lag_1', 'value_lag_7'])
        """
        from py_recipes.steps.naomit import StepNaOmit
        return self.add_step(StepNaOmit(columns=columns))

    # ========== Extended Categorical Steps ==========

    def step_other(
        self,
        columns: Optional[List[str]] = None,
        threshold: float = 0.05,
        other_label: str = "other"
    ) -> "Recipe":
        """
        Pool infrequent categorical levels into "other".

        Args:
            columns: Categorical columns (None = all categorical)
            threshold: Minimum frequency to keep level
            other_label: Label for pooled category

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.categorical_extended import StepOther
        return self.add_step(StepOther(columns=columns, threshold=threshold, other_label=other_label))

    def step_novel(
        self,
        columns: Optional[List[str]] = None,
        novel_label: str = "new"
    ) -> "Recipe":
        """
        Handle novel categorical levels in new data.

        Args:
            columns: Categorical columns (None = all categorical)
            novel_label: Label for novel levels

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.categorical_extended import StepNovel
        return self.add_step(StepNovel(columns=columns, novel_label=novel_label))

    def step_unknown(
        self,
        columns: Optional[List[str]] = None,
        unknown_label: str = "_unknown_"
    ) -> "Recipe":
        """
        Assign missing categorical values to "unknown" level.

        Args:
            columns: Categorical columns (None = all categorical)
            unknown_label: Label for missing values

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.categorical_extended import StepUnknown
        return self.add_step(StepUnknown(columns=columns, unknown_label=unknown_label))

    def step_indicate_na(
        self,
        columns: Optional[List[str]] = None,
        prefix: str = "na_ind"
    ) -> "Recipe":
        """
        Create indicator columns for missing values.

        Args:
            columns: Columns to create indicators for (None = all with NA)
            prefix: Prefix for indicator columns

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.categorical_extended import StepIndicateNa
        return self.add_step(StepIndicateNa(columns=columns, prefix=prefix))

    def step_integer(
        self,
        columns: Optional[List[str]] = None,
        zero_based: bool = True
    ) -> "Recipe":
        """
        Integer encode categorical variables.

        Args:
            columns: Categorical columns (None = all categorical)
            zero_based: Use zero-based indexing

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.categorical_extended import StepInteger
        return self.add_step(StepInteger(columns=columns, zero_based=zero_based))

    # ========== Extended Imputation Steps ==========

    def step_impute_mode(
        self,
        columns: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Impute missing values using mode (most frequent value).

        Args:
            columns: Columns to impute (None = all with NA)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.impute import StepImputeMode
        return self.add_step(StepImputeMode(columns=columns))

    def step_impute_knn(
        self,
        columns: Optional[List[str]] = None,
        neighbors: int = 5,
        weights: str = "uniform"
    ) -> "Recipe":
        """
        Impute missing values using K-Nearest Neighbors.

        Args:
            columns: Columns to impute (None = all numeric with NA)
            neighbors: Number of neighbors
            weights: Weight function ('uniform' or 'distance')

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.impute import StepImputeKnn
        return self.add_step(StepImputeKnn(columns=columns, neighbors=neighbors, weights=weights))

    def step_impute_linear(
        self,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        limit_direction: str = "both"
    ) -> "Recipe":
        """
        Impute missing values using linear interpolation.

        Args:
            columns: Columns to impute (None = all numeric with NA)
            limit: Maximum consecutive NAs to fill
            limit_direction: Direction to fill ('forward', 'backward', 'both')

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.impute import StepImputeLinear
        return self.add_step(StepImputeLinear(columns=columns, limit=limit, limit_direction=limit_direction))

    # ========== Basis Function Steps ==========

    def step_bs(
        self,
        column: str,
        degree: int = 3,
        df: Optional[int] = None,
        knots: Optional[int] = None
    ) -> "Recipe":
        """
        Create B-spline basis functions.

        Args:
            column: Column to create splines for
            degree: Degree of spline
            df: Degrees of freedom
            knots: Number of internal knots

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.basis import StepBs
        return self.add_step(StepBs(column=column, degree=degree, df=df, knots=knots))

    def step_ns(
        self,
        column: str,
        df: Optional[int] = None,
        knots: Optional[int] = None
    ) -> "Recipe":
        """
        Create natural spline basis functions.

        Args:
            column: Column to create splines for
            df: Degrees of freedom
            knots: Number of internal knots

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.basis import StepNs
        return self.add_step(StepNs(column=column, df=df, knots=knots))

    def step_poly(
        self,
        columns: List[str],
        degree: int = 2,
        include_interactions: bool = False,
        inplace: bool = True
    ) -> "Recipe":
        """
        Create polynomial features.

        Args:
            columns: Columns to create polynomials for
            degree: Maximum polynomial degree
            include_interactions: Include cross terms
            inplace: If True, replace original columns; if False, keep originals (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.basis import StepPoly
        return self.add_step(StepPoly(columns=columns, degree=degree, include_interactions=include_interactions, inplace=inplace))

    def step_harmonic(
        self,
        column: str,
        frequency: int = 1,
        period: float = 1.0
    ) -> "Recipe":
        """
        Create harmonic (Fourier) basis functions.

        Args:
            column: Column to create harmonics for
            frequency: Number of harmonics/cycles
            period: Period of seasonality

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.basis import StepHarmonic
        return self.add_step(StepHarmonic(column=column, frequency=frequency, period=period))

    # ========== Interaction Steps ==========

    def step_interact(
        self,
        interactions: Union[List[tuple], List[str], Callable],
        separator: str = "_x_"
    ) -> "Recipe":
        """
        Create interaction features between columns.

        Args:
            interactions: Can be:
                - List of column pairs: [("x1", "x2"), ("x1", "x3")]
                - List of columns: ["x1", "x2", "x3"] (creates all pairwise interactions)
                - Selector function: all_numeric_predictors() (creates all pairwise interactions)
            separator: Separator for interaction names (default: "_x_")

        Returns:
            Self for method chaining

        Examples:
            >>> # Specific pairs
            >>> rec.step_interact([("x1", "x2"), ("x1", "x3")])
            >>>
            >>> # All pairs from list
            >>> rec.step_interact(["x1", "x2", "x3"])
            >>>
            >>> # All pairs from selector
            >>> rec.step_interact(all_numeric_predictors())
        """
        from py_recipes.steps.interactions import StepInteract

        # Pass directly to StepInteract - it will handle resolution during prep()
        return self.add_step(StepInteract(interactions=interactions, separator=separator))

    def step_ratio(
        self,
        ratios: List[tuple],
        offset: float = 1e-10,
        separator: str = "_per_"
    ) -> "Recipe":
        """
        Create ratio features between columns.

        Args:
            ratios: List of (numerator, denominator) pairs
            offset: Small value to avoid division by zero
            separator: Separator for ratio names

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.interactions import StepRatio
        return self.add_step(StepRatio(ratios=ratios, offset=offset, separator=separator))

    # ========== Discretization Steps ==========

    def step_discretize(
        self,
        columns: Optional[List[str]] = None,
        num_breaks: int = 4,
        method: str = "quantile",
        labels: Optional[List[str]] = None
    ) -> "Recipe":
        """
        Discretize numeric columns into bins.

        Args:
            columns: Columns to discretize (None = all numeric)
            num_breaks: Number of bins
            method: Binning method ('quantile' or 'width')
            labels: Custom bin labels

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.discretization import StepDiscretize
        return self.add_step(StepDiscretize(columns=columns, num_breaks=num_breaks, method=method, labels=labels))

    def step_cut(
        self,
        columns: List[str],
        breaks: dict,
        labels: Optional[dict] = None,
        include_lowest: bool = True
    ) -> "Recipe":
        """
        Cut numeric columns at specified thresholds.

        Args:
            columns: Columns to cut
            breaks: Dict of column -> list of breakpoints
            labels: Dict of column -> list of labels
            include_lowest: Include lowest value in first bin

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.discretization import StepCut
        return self.add_step(StepCut(columns=columns, breaks=breaks, labels=labels, include_lowest=include_lowest))

    def step_percentile(
        self,
        columns: Optional[List[str]] = None,
        num_breaks: int = 100,
        as_integer: bool = True
    ) -> "Recipe":
        """
        Convert numeric columns to percentile ranks.

        Args:
            columns: Columns to convert (None = all numeric)
            num_breaks: Number of percentile bins (default: 100 for 0-100 scale)
            as_integer: Return integer percentiles (default: True)

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.discretization import StepPercentile
        return self.add_step(StepPercentile(columns=columns, num_breaks=num_breaks, as_integer=as_integer))

    # ========== Advanced Dimensionality Reduction Steps ==========

    def step_ica(
        self,
        columns: Optional[List[str]] = None,
        num_comp: Optional[int] = None,
        algorithm: str = "parallel",
        max_iter: int = 200
    ) -> "Recipe":
        """
        Independent Component Analysis transformation.

        Args:
            columns: Columns to apply ICA to (None = all numeric)
            num_comp: Number of components
            algorithm: ICA algorithm ('parallel', 'deflation')
            max_iter: Maximum iterations

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.reduction import StepIca
        return self.add_step(StepIca(columns=columns, num_comp=num_comp, algorithm=algorithm, max_iter=max_iter))

    def step_kpca(
        self,
        columns: Optional[List[str]] = None,
        num_comp: Optional[int] = None,
        kernel: str = "rbf",
        gamma: Optional[float] = None
    ) -> "Recipe":
        """
        Kernel Principal Component Analysis.

        Args:
            columns: Columns to apply kernel PCA to (None = all numeric)
            num_comp: Number of components
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.reduction import StepKpca
        return self.add_step(StepKpca(columns=columns, num_comp=num_comp, kernel=kernel, gamma=gamma))

    def step_pls(
        self,
        columns: Optional[List[str]] = None,
        outcome: str = None,
        num_comp: Optional[int] = None
    ) -> "Recipe":
        """
        Partial Least Squares transformation.

        Args:
            columns: Predictor columns (None = all numeric except outcome)
            outcome: Outcome column name
            num_comp: Number of components

        Returns:
            Self for method chaining
        """
        from py_recipes.steps.reduction import StepPls
        return self.add_step(StepPls(columns=columns, outcome=outcome, num_comp=num_comp))

    def step_holiday(
        self,
        date_column: str,
        country: str = "US",
        holidays: Optional[List[str]] = None,
        prefix: str = "holiday_"
    ) -> "Recipe":
        """
        Add holiday indicator features.

        Creates binary indicators for holidays, useful for capturing
        holiday effects in time series models. Uses pytimetk if available.

        Args:
            date_column: Column containing dates
            country: Country code for holidays (e.g., 'US', 'UK', 'CA')
            holidays: List of specific holidays (None = all major holidays)
            prefix: Prefix for created columns

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe().step_holiday('date', country='US')
            >>> rec = Recipe().step_holiday('date', holidays=["Christmas Day", "New Year's Day"])
        """
        from py_recipes.steps.timeseries_extended import StepHoliday
        return self.add_step(StepHoliday(
            date_column=date_column,
            country=country,
            holidays=holidays,
            prefix=prefix
        ))

    def step_fourier(
        self,
        date_column: str,
        period: float,
        K: int = 5,
        prefix: str = "fourier_"
    ) -> "Recipe":
        """
        Add Fourier features for seasonality.

        Creates sine and cosine features at different frequencies to capture
        seasonal patterns. Works with pytimetk if available.

        Args:
            date_column: Column containing dates or numeric time index
            period: Period of seasonality (e.g., 365 for yearly, 12 for monthly)
            K: Number of Fourier term pairs to include
            prefix: Prefix for created columns

        Returns:
            Self for method chaining

        Examples:
            >>> # Yearly seasonality with 5 Fourier pairs
            >>> rec = Recipe().step_fourier('date', period=365, K=5)
            >>> # Monthly seasonality
            >>> rec = Recipe().step_fourier('month', period=12, K=3)
        """
        from py_recipes.steps.timeseries_extended import StepFourier
        return self.add_step(StepFourier(
            date_column=date_column,
            period=period,
            K=K,
            prefix=prefix
        ))

    def step_timeseries_signature(
        self,
        date_column: str,
        features: Optional[List[str]] = None,
        prefix: str = ""
    ) -> "Recipe":
        """
        Extract comprehensive time-based features from dates.

        Creates 15+ time features including hour, day, month, quarter, year,
        day of week, weekend indicators, etc. Uses pytimetk if available.

        Args:
            date_column: Column containing dates
            features: List of specific features to extract (None = all)
            prefix: Prefix for created columns

        Returns:
            Self for method chaining

        Examples:
            >>> # Extract all time features
            >>> rec = Recipe().step_timeseries_signature('date')
            >>> # Extract specific features
            >>> rec = Recipe().step_timeseries_signature('date', features=['month', 'day_of_week'])
        """
        from py_recipes.steps.timeseries_extended import StepTimeseriesSignature
        return self.add_step(StepTimeseriesSignature(
            date_column=date_column,
            features=features,
            prefix=prefix
        ))

    def step_lead(
        self,
        columns: List[str],
        leads: List[int],
        prefix: str = "lead_"
    ) -> "Recipe":
        """
        Create lead (future) features.

        Creates features that look ahead in time, useful for prediction tasks.
        Works with pytimetk if available.

        Args:
            columns: Columns to create leads for
            leads: List of lead periods (e.g., [1, 2, 7])
            prefix: Prefix for created columns

        Returns:
            Self for method chaining

        Examples:
            >>> # 1-step and 7-step ahead features
            >>> rec = Recipe().step_lead(['sales', 'price'], leads=[1, 7])
        """
        from py_recipes.steps.timeseries_extended import StepLead
        return self.add_step(StepLead(
            columns=columns,
            leads=leads,
            prefix=prefix
        ))

    def step_ewm(
        self,
        columns: List[str],
        span: int = 10,
        statistics: Optional[List[str]] = None,
        prefix: str = "ewm_"
    ) -> "Recipe":
        """
        Create exponentially weighted moving (EWM) features.

        Computes EWM statistics giving more weight to recent observations.
        Useful for capturing trends and momentum.

        Args:
            columns: Columns to compute EWM for
            span: Span for exponential weighting (smaller = more recent weight)
            statistics: Statistics to compute ('mean', 'std', 'var')
            prefix: Prefix for created columns

        Returns:
            Self for method chaining

        Examples:
            >>> # EWM mean with span of 10
            >>> rec = Recipe().step_ewm(['sales'], span=10, statistics=['mean'])
            >>> # EWM mean and std
            >>> rec = Recipe().step_ewm(['price'], span=20, statistics=['mean', 'std'])
        """
        from py_recipes.steps.timeseries_extended import StepEwm
        return self.add_step(StepEwm(
            columns=columns,
            span=span,
            statistics=statistics,
            prefix=prefix
        ))

    def step_expanding(
        self,
        columns: List[str],
        statistics: Optional[List[str]] = None,
        prefix: str = "expanding_",
        min_periods: int = 1
    ) -> "Recipe":
        """
        Create expanding window features.

        Computes cumulative statistics from the start to each point.
        Useful for running totals, cumulative averages, etc.

        Args:
            columns: Columns to compute expanding stats for
            statistics: Statistics to compute ('mean', 'std', 'sum', 'min', 'max')
            prefix: Prefix for created columns
            min_periods: Minimum periods required

        Returns:
            Self for method chaining

        Examples:
            >>> # Cumulative mean
            >>> rec = Recipe().step_expanding(['sales'], statistics=['mean'])
            >>> # Running total and cumulative average
            >>> rec = Recipe().step_expanding(['revenue'], statistics=['sum', 'mean'])
        """
        from py_recipes.steps.timeseries_extended import StepExpanding
        return self.add_step(StepExpanding(
            columns=columns,
            statistics=statistics,
            prefix=prefix,
            min_periods=min_periods
        ))

    # Role Management Methods

    def update_role(
        self,
        columns: List[str],
        new_role: str,
        old_role: Optional[str] = None
    ) -> "Recipe":
        """
        Update the role of specified columns.

        Args:
            columns: Column names to update
            new_role: New role to assign
            old_role: Optional old role to remove from (if None, removes from all roles)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe()
            >>> rec = rec.update_role(['id'], 'ID')
            >>> rec = rec.update_role(['outcome'], 'outcome')
            >>> rec = rec.update_role(['x1', 'x2'], 'predictor')
        """
        # Remove from old role if specified
        if old_role is not None:
            if old_role in self.roles:
                self.roles[old_role] = [
                    col for col in self.roles[old_role] if col not in columns
                ]
        else:
            # Remove from all roles
            for role in self.roles:
                self.roles[role] = [
                    col for col in self.roles[role] if col not in columns
                ]

        # Add to new role
        if new_role not in self.roles:
            self.roles[new_role] = []

        # Add columns that aren't already in the new role
        for col in columns:
            if col not in self.roles[new_role]:
                self.roles[new_role].append(col)

        return self

    def add_role(
        self,
        columns: List[str],
        new_role: str
    ) -> "Recipe":
        """
        Add a role to specified columns (doesn't remove existing roles).

        Args:
            columns: Column names to add role to
            new_role: Role to add

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe()
            >>> rec = rec.add_role(['id'], 'ID')
            >>> rec = rec.add_role(['timestamp'], 'time_index')
        """
        if new_role not in self.roles:
            self.roles[new_role] = []

        # Add columns that aren't already in the role
        for col in columns:
            if col not in self.roles[new_role]:
                self.roles[new_role].append(col)

        return self

    def remove_role(
        self,
        columns: List[str],
        old_role: str
    ) -> "Recipe":
        """
        Remove a role from specified columns.

        Args:
            columns: Column names to remove role from
            old_role: Role to remove

        Returns:
            Self for method chaining

        Examples:
            >>> rec = Recipe()
            >>> rec = rec.remove_role(['x1'], 'predictor')
        """
        if old_role in self.roles:
            self.roles[old_role] = [
                col for col in self.roles[old_role] if col not in columns
            ]

        return self

    def has_role(
        self,
        columns: Optional[List[str]] = None,
        role: str = "predictor"
    ) -> List[str]:
        """
        Get columns with a specific role.

        Args:
            columns: Optional subset of columns to filter (None = all columns)
            role: Role to filter by

        Returns:
            List of column names with the specified role

        Examples:
            >>> rec = Recipe()
            >>> rec = rec.update_role(['x1', 'x2'], 'predictor')
            >>> rec.has_role(role='predictor')
            ['x1', 'x2']
        """
        role_cols = self.roles.get(role, [])

        if columns is None:
            return role_cols
        else:
            return [col for col in columns if col in role_cols]

    # Advanced Feature Selection Methods

    def step_vip(
        self,
        outcome: str,
        threshold: float = 1.0,
        num_comp: int = 2
    ) -> "Recipe":
        """
        Variable Importance in Projection (VIP) feature selection.

        Calculates VIP scores from a PLS model and selects features based on threshold.
        VIP scores measure the importance of each variable in the projection used in a PLS model.

        Args:
            outcome: Name of outcome column (required for supervised selection)
            threshold: VIP threshold for feature selection (default 1.0)
                Variables with VIP > threshold are kept
            num_comp: Number of PLS components to use (default 2)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = recipe()
            >>> rec = rec.step_vip(outcome='y', threshold=1.0, num_comp=2)
        """
        from py_recipes.steps.feature_selection_advanced import StepVip
        return self.add_step(StepVip(
            threshold=threshold,
            num_comp=num_comp,
            outcome=outcome
        ))

    def step_boruta(
        self,
        outcome: str,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        perc: int = 100,
        alpha: float = 0.05
    ) -> "Recipe":
        """
        Boruta all-relevant feature selection.

        Uses Boruta algorithm to identify all features that are statistically relevant
        to the outcome. Creates shadow features and compares real feature importance
        to maximum shadow importance.

        Args:
            outcome: Name of outcome column (required)
            max_iter: Maximum number of iterations (default 100)
            random_state: Random seed for reproducibility (default None)
            perc: Percentile of shadow feature importance to compare against (default 100)
            alpha: P-value threshold for feature importance test (default 0.05)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = recipe()
            >>> rec = rec.step_boruta(outcome='y', max_iter=100)
        """
        from py_recipes.steps.feature_selection_advanced import StepBoruta
        return self.add_step(StepBoruta(
            outcome=outcome,
            max_iter=max_iter,
            random_state=random_state,
            perc=perc,
            alpha=alpha
        ))

    def step_rfe(
        self,
        outcome: str,
        n_features: Optional[int] = None,
        step: int = 1,
        estimator: Optional[Any] = None
    ) -> "Recipe":
        """
        Recursive Feature Elimination (RFE) feature selection.

        Recursively removes features and builds models to find the optimal subset.
        Uses model coefficients or feature importances to rank features.

        Args:
            outcome: Name of outcome column (required)
            n_features: Number of features to select (default None = select half)
            step: Number of features to remove at each iteration (default 1)
            estimator: Sklearn estimator to use (default None = LogisticRegression/LinearRegression)

        Returns:
            Self for method chaining

        Examples:
            >>> rec = recipe()
            >>> rec = rec.step_rfe(outcome='y', n_features=10)
        """
        from py_recipes.steps.feature_selection_advanced import StepRfe
        return self.add_step(StepRfe(
            outcome=outcome,
            n_features=n_features,
            step=step,
            estimator=estimator
        ))

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedRecipe":
        """
        Fit recipe to training data.

        Executes each step's prep() method sequentially, fitting parameters
        on the training data.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedRecipe ready to bake new data

        Examples:
            >>> rec = Recipe().step_normalize()
            >>> rec_fit = rec.prep(train_data)
        """
        prepared_steps = []
        current_data = data.copy()

        for step in self.steps:
            # Prep the step
            prepared_step = step.prep(current_data, training=training)
            prepared_steps.append(prepared_step)

            # Apply step to current data for next step
            current_data = prepared_step.bake(current_data)

        return PreparedRecipe(
            recipe=self,
            prepared_steps=prepared_steps,
            template=data
        )


@dataclass
class PreparedRecipe:
    """
    Fitted recipe ready to transform new data.

    Created by Recipe.prep(), contains fitted preprocessing steps
    that can be applied to new data via bake().

    Attributes:
        recipe: Original Recipe specification
        prepared_steps: List of fitted PreparedStep objects
        template: Template DataFrame from training

    Examples:
        >>> rec_fit = recipe.prep(train_data)
        >>> test_transformed = rec_fit.bake(test_data)
    """

    recipe: Recipe
    prepared_steps: List[Any]
    template: pd.DataFrame

    def bake(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted recipe to new data.

        Applies each prepared step sequentially to transform the data
        using parameters fitted during prep().

        Args:
            new_data: Data to transform

        Returns:
            Transformed DataFrame

        Examples:
            >>> test_transformed = rec_fit.bake(test_data)
        """
        result = new_data.copy()

        for prepared_step in self.prepared_steps:
            result = prepared_step.bake(result)

        return result

    def juice(self) -> pd.DataFrame:
        """
        Extract transformed training data.

        Returns the training data after all transformations have been applied.
        Equivalent to bake(template).

        Returns:
            Transformed training DataFrame
        """
        return self.bake(self.template)


def recipe(data: Optional[pd.DataFrame] = None) -> Recipe:
    """
    Create a new recipe.

    Args:
        data: Optional template DataFrame for role inference

    Returns:
        Empty Recipe ready for steps

    Examples:
        >>> rec = recipe().step_normalize().step_dummy(["category"])
    """
    return Recipe(template=data)
