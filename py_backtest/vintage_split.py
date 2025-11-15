"""
Vintage-aware split classes for backtesting.

Extends py_rsample Split to handle data vintages for point-in-time backtesting.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from py_rsample.split import Split, RSplit
from py_backtest.vintage_utils import select_vintage


@dataclass(frozen=True)
class VintageSplit(Split):
    """
    Vintage-aware split containing train/test indices with vintage metadata.

    Extends Split to add vintage-specific attributes and methods for
    point-in-time backtesting.

    Attributes:
        data: Original vintage DataFrame
        in_id: Training indices (numpy array)
        out_id: Testing indices (numpy array)
        id: Split identifier string (e.g., "Vintage001")
        vintage_date: Date of data vintage used for training
        training_start: First date in training period
        training_end: Last date in training period
        test_start: First date in test period
        test_end: Last date in test period
        as_of_col: Name of vintage date column
        date_col: Name of observation date column
        vintage_selection: Strategy for selecting vintage ("latest", "exact")

    Example:
        >>> split = VintageSplit(
        ...     data=vintage_df,
        ...     in_id=np.array([0, 1, 2]),
        ...     out_id=np.array([3, 4]),
        ...     id="Vintage001",
        ...     vintage_date=pd.Timestamp("2023-01-15"),
        ...     training_start=pd.Timestamp("2021-01-01"),
        ...     training_end=pd.Timestamp("2023-01-01"),
        ...     test_start=pd.Timestamp("2023-01-08"),
        ...     test_end=pd.Timestamp("2023-04-01"),
        ...     as_of_col="as_of_date",
        ...     date_col="date",
        ...     vintage_selection="latest"
        ... )
    """

    vintage_date: pd.Timestamp
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    as_of_col: str
    date_col: str
    vintage_selection: str


class VintageRSplit(RSplit):
    """
    Vintage-aware RSplit for extracting training/testing data with correct vintages.

    Extends RSplit to handle vintage selection when accessing training/testing data.
    Training data uses the vintage available at vintage_date, while testing data
    uses the final/latest vintage (for evaluation against actual outcomes).

    Attributes:
        split: Underlying VintageSplit object

    Methods:
        training(): Get training DataFrame using correct vintage
        testing(): Get testing DataFrame using final vintage
        get_vintage_info(): Get metadata about this fold's vintage

    Example:
        >>> rsplit = VintageRSplit(vintage_split)
        >>> train_df = rsplit.training()  # Uses vintage_date vintage
        >>> test_df = rsplit.testing()    # Uses final vintage
        >>> info = rsplit.get_vintage_info()
    """

    def __init__(self, split: VintageSplit):
        """
        Initialize VintageRSplit from a VintageSplit.

        Args:
            split: VintageSplit object containing data and vintage metadata
        """
        if not isinstance(split, VintageSplit):
            raise TypeError(
                f"VintageRSplit requires VintageSplit object, got {type(split)}"
            )
        super().__init__(split)

    @property
    def split(self) -> VintageSplit:
        """Get the underlying VintageSplit object"""
        return self._split

    def training(self) -> pd.DataFrame:
        """
        Get training data using correct vintage.

        Returns training data where:
        - Observation dates in [training_start, training_end]
        - Using vintage available at vintage_date (simulates point-in-time data)

        Returns:
            DataFrame with training observations using historical vintage
        """
        vsplit = self._split

        # Get data for training period
        training_period_data = vsplit.data[
            (vsplit.data[vsplit.date_col] >= vsplit.training_start) &
            (vsplit.data[vsplit.date_col] <= vsplit.training_end)
        ]

        # Select appropriate vintage for each observation
        # (simulates data available at vintage_date)
        vintage_data = select_vintage(
            data=training_period_data,
            as_of_col=vsplit.as_of_col,
            date_col=vsplit.date_col,
            vintage_date=vsplit.vintage_date,
            vintage_selection=vsplit.vintage_selection
        )

        # Drop as_of_col (not needed for modeling)
        result = vintage_data.drop(columns=[vsplit.as_of_col]).reset_index(drop=True)

        return result

    def testing(self) -> pd.DataFrame:
        """
        Get testing data for evaluation.

        Returns test data where:
        - Observation dates in [test_start, test_end]
        - Using FINAL vintage (most recent as_of_date)
          Because we're evaluating against actual outcomes

        Returns:
            DataFrame with testing observations using final vintage
        """
        vsplit = self._split

        # Get data for test period
        test_period_data = vsplit.data[
            (vsplit.data[vsplit.date_col] >= vsplit.test_start) &
            (vsplit.data[vsplit.date_col] <= vsplit.test_end)
        ]

        # Use final vintage (most recent as_of_date for each observation)
        # This represents the "actual" values for evaluation
        idx = test_period_data.groupby(vsplit.date_col)[vsplit.as_of_col].idxmax()
        final_vintage_data = test_period_data.loc[idx]

        # Drop as_of_col (not needed for modeling)
        result = final_vintage_data.drop(columns=[vsplit.as_of_col]).reset_index(drop=True)

        return result

    def get_vintage_info(self) -> dict:
        """
        Get metadata about this fold's vintage.

        Returns:
            Dictionary with:
            - vintage_date: Date of vintage used
            - training_start: First training date
            - training_end: Last training date
            - test_start: First test date
            - test_end: Last test date
            - n_train_obs: Number of training observations
            - n_test_obs: Number of test observations
            - forecast_horizon: Gap between training end and test start

        Example:
            >>> info = rsplit.get_vintage_info()
            >>> print(info)
            {
                'vintage_date': Timestamp('2023-01-15'),
                'training_start': Timestamp('2021-01-01'),
                'training_end': Timestamp('2023-01-01'),
                'test_start': Timestamp('2023-01-08'),
                'test_end': Timestamp('2023-04-01'),
                'n_train_obs': 730,
                'n_test_obs': 90,
                'forecast_horizon': Timedelta('7 days')
            }
        """
        vsplit = self._split

        # Get actual data to count observations
        train_data = self.training()
        test_data = self.testing()

        return {
            "vintage_date": vsplit.vintage_date,
            "training_start": vsplit.training_start,
            "training_end": vsplit.training_end,
            "test_start": vsplit.test_start,
            "test_end": vsplit.test_end,
            "n_train_obs": len(train_data),
            "n_test_obs": len(test_data),
            "forecast_horizon": vsplit.test_start - vsplit.training_end
        }

    def __repr__(self) -> str:
        """String representation"""
        vsplit = self._split
        return (
            f"VintageRSplit(id={vsplit.id}, "
            f"vintage_date={vsplit.vintage_date.date()}, "
            f"train={vsplit.training_start.date()} to {vsplit.training_end.date()}, "
            f"test={vsplit.test_start.date()} to {vsplit.test_end.date()})"
        )
