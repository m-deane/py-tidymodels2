"""
Extended time series preprocessing steps with pytimetk wrappers

Provides advanced time series features including holiday detection and Fourier terms.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class StepHoliday:
    """
    Add holiday indicator features.

    Creates binary indicators for holidays using pytimetk's holiday detection.
    Useful for capturing holiday effects in time series models.

    Attributes:
        date_column: Column containing dates
        country: Country code for holidays (e.g., 'US', 'UK', 'CA')
        holidays: List of specific holidays to include (None = all major holidays)
        prefix: Prefix for created columns (default: 'holiday_')
    """

    date_column: str
    country: str = "US"
    holidays: Optional[List[str]] = None
    prefix: str = "holiday_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepHoliday":
        """
        Prepare holiday features.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepHoliday ready to create features
        """
        if self.date_column not in data.columns:
            return PreparedStepHoliday(
                date_column=self.date_column,
                country=self.country,
                holidays=self.holidays or [],
                prefix=self.prefix,
                feature_names=[]
            )

        # Define default holidays if not specified
        if self.holidays is None:
            # Common US holidays
            default_holidays = [
                "New Year's Day",
                "Martin Luther King Jr. Day",
                "Presidents' Day",
                "Memorial Day",
                "Independence Day",
                "Labor Day",
                "Columbus Day",
                "Veterans Day",
                "Thanksgiving",
                "Christmas Day"
            ]
            holidays_to_use = default_holidays
        else:
            holidays_to_use = self.holidays

        # Generate feature names
        feature_names = []
        for h in holidays_to_use:
            clean_name = h.lower().replace(' ', '_').replace("'", '')
            feature_names.append(f"{self.prefix}{clean_name}")

        return PreparedStepHoliday(
            date_column=self.date_column,
            country=self.country,
            holidays=holidays_to_use,
            prefix=self.prefix,
            feature_names=feature_names
        )


@dataclass
class PreparedStepHoliday:
    """
    Fitted holiday feature creator.

    Attributes:
        date_column: Date column name
        country: Country code
        holidays: List of holidays to detect
        prefix: Column name prefix
        feature_names: Names of created features
    """

    date_column: str
    country: str
    holidays: List[str]
    prefix: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday indicator features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with holiday features added
        """
        result = data.copy()

        if self.date_column not in result.columns or len(self.feature_names) == 0:
            return result

        # Try to use pytimetk if available
        try:
            import pytimetk as tk

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result[self.date_column]):
                dates = pd.to_datetime(result[self.date_column])
            else:
                dates = result[self.date_column]

            # Create holiday features using pytimetk
            for i, holiday_name in enumerate(self.holidays):
                feature_name = self.feature_names[i]

                # Use pytimetk's augment_holiday_signature if available
                # Otherwise, create simple holiday indicators
                try:
                    # Create a temporary dataframe with the date column
                    temp_df = pd.DataFrame({self.date_column: dates})

                    # Use pytimetk to detect holidays
                    holiday_df = tk.augment_holiday_signature(
                        temp_df,
                        date_column=self.date_column,
                        country_name=self.country
                    )

                    # Extract the specific holiday indicator if it exists
                    # pytimetk creates columns like 'holiday_name'
                    holiday_cols = [col for col in holiday_df.columns if holiday_name.lower().replace(' ', '_') in col.lower()]

                    if holiday_cols:
                        result[feature_name] = holiday_df[holiday_cols[0]].astype(int)
                    else:
                        # If specific holiday not found, create zeros
                        result[feature_name] = 0

                except Exception:
                    # Fallback: create simple holiday indicators
                    result[feature_name] = self._simple_holiday_indicator(
                        dates, holiday_name, self.country
                    )

        except ImportError:
            # pytimetk not available, use simple holiday detection
            if not pd.api.types.is_datetime64_any_dtype(result[self.date_column]):
                dates = pd.to_datetime(result[self.date_column])
            else:
                dates = result[self.date_column]

            for i, holiday_name in enumerate(self.holidays):
                feature_name = self.feature_names[i]
                result[feature_name] = self._simple_holiday_indicator(
                    dates, holiday_name, self.country
                )

        return result

    def _simple_holiday_indicator(
        self,
        dates: pd.Series,
        holiday_name: str,
        country: str
    ) -> pd.Series:
        """
        Simple holiday indicator without pytimetk.

        Args:
            dates: Series of dates
            holiday_name: Name of holiday
            country: Country code

        Returns:
            Binary series indicating holiday
        """
        # Simple implementation for common US holidays
        month = dates.dt.month
        day = dates.dt.day

        indicator = pd.Series(0, index=dates.index)

        if country == "US":
            if "New Year" in holiday_name:
                indicator = ((month == 1) & (day == 1)).astype(int)
            elif "Independence" in holiday_name or "July 4" in holiday_name:
                indicator = ((month == 7) & (day == 4)).astype(int)
            elif "Christmas" in holiday_name:
                indicator = ((month == 12) & (day == 25)).astype(int)
            elif "Thanksgiving" in holiday_name:
                # 4th Thursday of November (approximation)
                indicator = ((month == 11) & (dates.dt.day >= 22) &
                           (dates.dt.day <= 28) & (dates.dt.dayofweek == 3)).astype(int)
            elif "Memorial" in holiday_name:
                # Last Monday of May (approximation)
                indicator = ((month == 5) & (dates.dt.day >= 25) &
                           (dates.dt.dayofweek == 0)).astype(int)
            elif "Labor" in holiday_name:
                # First Monday of September (approximation)
                indicator = ((month == 9) & (dates.dt.day <= 7) &
                           (dates.dt.dayofweek == 0)).astype(int)

        return indicator


@dataclass
class StepFourier:
    """
    Add Fourier features for seasonality.

    Creates sine and cosine features at different frequencies to capture
    seasonal patterns in time series data.

    Attributes:
        date_column: Column containing dates or numeric time index
        period: Period of seasonality (e.g., 365 for yearly, 12 for monthly)
        K: Number of Fourier term pairs to include (default: 5)
        prefix: Prefix for created columns (default: 'fourier_')
    """

    date_column: str
    period: float
    K: int = 5
    prefix: str = "fourier_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepFourier":
        """
        Prepare Fourier features.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepFourier ready to create features
        """
        if self.date_column not in data.columns:
            return PreparedStepFourier(
                date_column=self.date_column,
                period=self.period,
                K=self.K,
                prefix=self.prefix,
                feature_names=[]
            )

        # Generate feature names
        feature_names = []
        for k in range(1, self.K + 1):
            feature_names.append(f"{self.prefix}sin_{k}")
            feature_names.append(f"{self.prefix}cos_{k}")

        return PreparedStepFourier(
            date_column=self.date_column,
            period=self.period,
            K=self.K,
            prefix=self.prefix,
            feature_names=feature_names
        )


@dataclass
class PreparedStepFourier:
    """
    Fitted Fourier feature creator.

    Attributes:
        date_column: Date or time column name
        period: Seasonality period
        K: Number of Fourier pairs
        prefix: Column name prefix
        feature_names: Names of created features
    """

    date_column: str
    period: float
    K: int
    prefix: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with Fourier features added
        """
        result = data.copy()

        if self.date_column not in result.columns or len(self.feature_names) == 0:
            return result

        # Try to use pytimetk if available
        try:
            import pytimetk as tk

            # Convert to datetime if needed and not numeric
            if pd.api.types.is_datetime64_any_dtype(result[self.date_column]):
                # For datetime, create numeric time index
                time_idx = (result[self.date_column] - result[self.date_column].min()).dt.total_seconds()
                time_idx = time_idx / (24 * 3600)  # Convert to days
            elif pd.api.types.is_numeric_dtype(result[self.date_column]):
                time_idx = result[self.date_column]
            else:
                # Try to convert to datetime
                dates = pd.to_datetime(result[self.date_column])
                time_idx = (dates - dates.min()).dt.total_seconds() / (24 * 3600)

            # Try pytimetk's Fourier features
            try:
                temp_df = pd.DataFrame({'time': time_idx})
                fourier_df = tk.augment_fourier(
                    temp_df,
                    date_column='time',
                    periods=self.period,
                    max_order=self.K
                )

                # Extract Fourier features
                for k in range(1, self.K + 1):
                    sin_name = f"{self.prefix}sin_{k}"
                    cos_name = f"{self.prefix}cos_{k}"

                    # pytimetk might use different naming
                    sin_cols = [col for col in fourier_df.columns if f'sin' in col.lower() and f'{k}' in col]
                    cos_cols = [col for col in fourier_df.columns if f'cos' in col.lower() and f'{k}' in col]

                    if sin_cols:
                        result[sin_name] = fourier_df[sin_cols[0]]
                    else:
                        result[sin_name] = self._create_fourier_term(time_idx, k, 'sin')

                    if cos_cols:
                        result[cos_name] = fourier_df[cos_cols[0]]
                    else:
                        result[cos_name] = self._create_fourier_term(time_idx, k, 'cos')

            except Exception:
                # Fallback to manual creation
                for k in range(1, self.K + 1):
                    sin_name = f"{self.prefix}sin_{k}"
                    cos_name = f"{self.prefix}cos_{k}"
                    result[sin_name] = self._create_fourier_term(time_idx, k, 'sin')
                    result[cos_name] = self._create_fourier_term(time_idx, k, 'cos')

        except ImportError:
            # pytimetk not available, create manually
            if pd.api.types.is_datetime64_any_dtype(result[self.date_column]):
                time_idx = (result[self.date_column] - result[self.date_column].min()).dt.total_seconds()
                time_idx = time_idx / (24 * 3600)
            elif pd.api.types.is_numeric_dtype(result[self.date_column]):
                time_idx = result[self.date_column]
            else:
                dates = pd.to_datetime(result[self.date_column])
                time_idx = (dates - dates.min()).dt.total_seconds() / (24 * 3600)

            for k in range(1, self.K + 1):
                sin_name = f"{self.prefix}sin_{k}"
                cos_name = f"{self.prefix}cos_{k}"
                result[sin_name] = self._create_fourier_term(time_idx, k, 'sin')
                result[cos_name] = self._create_fourier_term(time_idx, k, 'cos')

        return result

    def _create_fourier_term(
        self,
        time_idx: pd.Series,
        k: int,
        term_type: str
    ) -> pd.Series:
        """
        Create Fourier term manually.

        Args:
            time_idx: Numeric time index
            k: Fourier order
            term_type: 'sin' or 'cos'

        Returns:
            Fourier feature series
        """
        freq = 2 * np.pi * k / self.period

        if term_type == 'sin':
            return np.sin(freq * time_idx)
        else:
            return np.cos(freq * time_idx)


@dataclass
class StepTimeseriesSignature:
    """
    Extract comprehensive time-based features from dates.

    Creates a wide range of time-based features including:
    - Hour, minute, second
    - Day, day of week, day of year
    - Week, month, quarter, year
    - Weekend/weekday indicators
    - Month/quarter start/end indicators

    Attributes:
        date_column: Column containing dates
        features: List of features to extract (None = all)
        prefix: Prefix for created columns (default: '')
    """

    date_column: str
    features: Optional[List[str]] = None
    prefix: str = ""

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepTimeseriesSignature":
        """
        Prepare timeseries signature extraction.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepTimeseriesSignature ready to extract features
        """
        if self.date_column not in data.columns:
            return PreparedStepTimeseriesSignature(
                date_column=self.date_column,
                features=self.features or [],
                prefix=self.prefix,
                feature_names=[]
            )

        # Default feature set
        if self.features is None:
            available_features = [
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'quarter', 'day_of_week', 'day_of_year', 'week_of_year',
                'is_weekend', 'is_month_start', 'is_month_end',
                'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'
            ]
        else:
            available_features = self.features

        # Generate feature names
        feature_names = [f"{self.prefix}{feat}" for feat in available_features]

        return PreparedStepTimeseriesSignature(
            date_column=self.date_column,
            features=available_features,
            prefix=self.prefix,
            feature_names=feature_names
        )


@dataclass
class PreparedStepTimeseriesSignature:
    """
    Fitted timeseries signature extractor.

    Attributes:
        date_column: Date column name
        features: Features to extract
        prefix: Column name prefix
        feature_names: Names of created features
    """

    date_column: str
    features: List[str]
    prefix: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract timeseries signature features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with time features added
        """
        result = data.copy()

        if self.date_column not in result.columns or len(self.features) == 0:
            return result

        # Convert to datetime
        if not pd.api.types.is_datetime64_any_dtype(result[self.date_column]):
            dates = pd.to_datetime(result[self.date_column])
        else:
            dates = result[self.date_column]

        # Try pytimetk first
        try:
            import pytimetk as tk

            temp_df = pd.DataFrame({self.date_column: dates})
            signature_df = tk.augment_timeseries_signature(temp_df, date_column=self.date_column)

            # Map pytimetk column names to our feature names
            for i, feature in enumerate(self.features):
                feature_name = self.feature_names[i]

                # Try to find matching column in signature_df
                matching_cols = [col for col in signature_df.columns
                               if feature.replace('_', '').lower() in col.replace('_', '').lower()]

                if matching_cols:
                    result[feature_name] = signature_df[matching_cols[0]]
                else:
                    # Fallback to manual extraction
                    result[feature_name] = self._extract_feature(dates, feature)

        except ImportError:
            # Manual extraction
            for i, feature in enumerate(self.features):
                feature_name = self.feature_names[i]
                result[feature_name] = self._extract_feature(dates, feature)

        return result

    def _extract_feature(self, dates: pd.Series, feature: str) -> pd.Series:
        """
        Manually extract a time feature.

        Args:
            dates: Series of dates
            feature: Feature name to extract

        Returns:
            Extracted feature series
        """
        if feature == 'year':
            return dates.dt.year
        elif feature == 'month':
            return dates.dt.month
        elif feature == 'day':
            return dates.dt.day
        elif feature == 'hour':
            return dates.dt.hour
        elif feature == 'minute':
            return dates.dt.minute
        elif feature == 'second':
            return dates.dt.second
        elif feature == 'quarter':
            return dates.dt.quarter
        elif feature == 'day_of_week':
            return dates.dt.dayofweek
        elif feature == 'day_of_year':
            return dates.dt.dayofyear
        elif feature == 'week_of_year':
            return dates.dt.isocalendar().week
        elif feature == 'is_weekend':
            return (dates.dt.dayofweek >= 5).astype(int)
        elif feature == 'is_month_start':
            return dates.dt.is_month_start.astype(int)
        elif feature == 'is_month_end':
            return dates.dt.is_month_end.astype(int)
        elif feature == 'is_quarter_start':
            return dates.dt.is_quarter_start.astype(int)
        elif feature == 'is_quarter_end':
            return dates.dt.is_quarter_end.astype(int)
        elif feature == 'is_year_start':
            return dates.dt.is_year_start.astype(int)
        elif feature == 'is_year_end':
            return dates.dt.is_year_end.astype(int)
        else:
            # Default: zeros
            return pd.Series(0, index=dates.index)


@dataclass
class StepLead:
    """
    Create lead (future) features.

    Creates features that look ahead in time, useful for prediction tasks
    where future information is being predicted.

    Attributes:
        columns: Columns to create leads for
        leads: List of lead periods (e.g., [1, 2, 7] for 1-step, 2-step, 7-step ahead)
        prefix: Prefix for created columns (default: 'lead_')
    """

    columns: List[str]
    leads: List[int]
    prefix: str = "lead_"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepLead":
        """
        Prepare lead feature creation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepLead ready to create features
        """
        cols = [col for col in self.columns if col in data.columns]

        # Generate feature names
        feature_names = []
        for col in cols:
            for lead in self.leads:
                feature_names.append(f"{self.prefix}{col}_lead_{lead}")

        return PreparedStepLead(
            columns=cols,
            leads=self.leads,
            prefix=self.prefix,
            feature_names=feature_names
        )


@dataclass
class PreparedStepLead:
    """
    Fitted lead feature creator.

    Attributes:
        columns: Columns to lead
        leads: Lead periods
        prefix: Column name prefix
        feature_names: Names of created features
    """

    columns: List[str]
    leads: List[int]
    prefix: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create lead features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with lead features added
        """
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            for lead in self.leads:
                feature_name = f"{self.prefix}{col}_lead_{lead}"

                # Try pytimetk first
                try:
                    import pytimetk as tk

                    temp_df = pd.DataFrame({col: result[col]})
                    lead_df = tk.augment_leads(temp_df, date_column=col, lags=lead)

                    # Find the lead column
                    lead_cols = [c for c in lead_df.columns if f'lead_{lead}' in c.lower()]

                    if lead_cols:
                        result[feature_name] = lead_df[lead_cols[0]]
                    else:
                        # Manual creation
                        result[feature_name] = result[col].shift(-lead)

                except ImportError:
                    # Manual creation: shift backwards (negative)
                    result[feature_name] = result[col].shift(-lead)

        return result


@dataclass
class StepEwm:
    """
    Create exponentially weighted moving (EWM) features.

    Computes exponentially weighted statistics, giving more weight to recent
    observations. Useful for capturing trends and momentum.

    Attributes:
        columns: Columns to compute EWM for
        span: Span for exponential weighting (smaller = more weight on recent)
        statistics: Statistics to compute ('mean', 'std', 'var')
        prefix: Prefix for created columns (default: 'ewm_')
    """

    columns: List[str]
    span: int = 10
    statistics: List[str] = None
    prefix: str = "ewm_"

    def __post_init__(self):
        if self.statistics is None:
            self.statistics = ['mean']

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepEwm":
        """
        Prepare EWM feature creation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepEwm ready to create features
        """
        cols = [col for col in self.columns if col in data.columns]

        # Generate feature names
        feature_names = []
        for col in cols:
            for stat in self.statistics:
                feature_names.append(f"{self.prefix}{col}_{stat}_{self.span}")

        return PreparedStepEwm(
            columns=cols,
            span=self.span,
            statistics=self.statistics,
            prefix=self.prefix,
            feature_names=feature_names
        )


@dataclass
class PreparedStepEwm:
    """
    Fitted EWM feature creator.

    Attributes:
        columns: Columns to compute EWM for
        span: EWM span
        statistics: Statistics to compute
        prefix: Column name prefix
        feature_names: Names of created features
    """

    columns: List[str]
    span: int
    statistics: List[str]
    prefix: str
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create EWM features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with EWM features added
        """
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            for stat in self.statistics:
                feature_name = f"{self.prefix}{col}_{stat}_{self.span}"

                # Use pandas EWM directly (more reliable than pytimetk for this use case)
                result[feature_name] = self._compute_ewm(result[col], stat)

        return result

    def _compute_ewm(self, series: pd.Series, stat: str) -> pd.Series:
        """
        Manually compute EWM statistic.

        Args:
            series: Series to compute EWM for
            stat: Statistic to compute

        Returns:
            EWM feature series
        """
        ewm = series.ewm(span=self.span)

        if stat == 'mean':
            return ewm.mean()
        elif stat == 'std':
            return ewm.std()
        elif stat == 'var':
            return ewm.var()
        else:
            return ewm.mean()


@dataclass
class StepExpanding:
    """
    Create expanding window features.

    Computes cumulative statistics from the start of the series to each point.
    Useful for computing running totals, cumulative averages, etc.

    Attributes:
        columns: Columns to compute expanding stats for
        statistics: Statistics to compute ('mean', 'std', 'sum', 'min', 'max')
        prefix: Prefix for created columns (default: 'expanding_')
        min_periods: Minimum periods required (default: 1)
    """

    columns: List[str]
    statistics: List[str] = None
    prefix: str = "expanding_"
    min_periods: int = 1

    def __post_init__(self):
        if self.statistics is None:
            self.statistics = ['mean']

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepExpanding":
        """
        Prepare expanding window feature creation.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepExpanding ready to create features
        """
        cols = [col for col in self.columns if col in data.columns]

        # Generate feature names
        feature_names = []
        for col in cols:
            for stat in self.statistics:
                feature_names.append(f"{self.prefix}{col}_{stat}")

        return PreparedStepExpanding(
            columns=cols,
            statistics=self.statistics,
            prefix=self.prefix,
            min_periods=self.min_periods,
            feature_names=feature_names
        )


@dataclass
class PreparedStepExpanding:
    """
    Fitted expanding window feature creator.

    Attributes:
        columns: Columns to compute expanding stats for
        statistics: Statistics to compute
        prefix: Column name prefix
        min_periods: Minimum periods
        feature_names: Names of created features
    """

    columns: List[str]
    statistics: List[str]
    prefix: str
    min_periods: int
    feature_names: List[str]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create expanding window features.

        Args:
            data: Data to transform

        Returns:
            DataFrame with expanding features added
        """
        result = data.copy()

        for col in self.columns:
            if col not in result.columns:
                continue

            for stat in self.statistics:
                feature_name = f"{self.prefix}{col}_{stat}"

                # Try pytimetk first
                try:
                    import pytimetk as tk

                    temp_df = pd.DataFrame({col: result[col]})
                    expanding_df = tk.augment_expanding(
                        temp_df,
                        date_column=col
                    )

                    # Find matching column
                    exp_cols = [c for c in expanding_df.columns
                               if 'expanding' in c.lower() and stat in c.lower()]

                    if exp_cols:
                        result[feature_name] = expanding_df[exp_cols[0]]
                    else:
                        # Manual creation
                        result[feature_name] = self._compute_expanding(result[col], stat)

                except ImportError:
                    # Manual creation
                    result[feature_name] = self._compute_expanding(result[col], stat)

        return result

    def _compute_expanding(self, series: pd.Series, stat: str) -> pd.Series:
        """
        Manually compute expanding statistic.

        Args:
            series: Series to compute expanding stats for
            stat: Statistic to compute

        Returns:
            Expanding feature series
        """
        expanding = series.expanding(min_periods=self.min_periods)

        if stat == 'mean':
            return expanding.mean()
        elif stat == 'std':
            return expanding.std()
        elif stat == 'var':
            return expanding.var()
        elif stat == 'sum':
            return expanding.sum()
        elif stat == 'min':
            return expanding.min()
        elif stat == 'max':
            return expanding.max()
        else:
            return expanding.mean()
