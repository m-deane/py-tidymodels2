"""
Financial oscillator indicators for time series feature engineering

Provides RSI, MACD, and Stochastic Oscillator indicators using pytimetk.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import pandas as pd


@dataclass
class StepOscillators:
    """
    Add financial oscillator indicators (RSI, MACD, Stochastic).

    Uses pytimetk to calculate momentum oscillators for financial time series.
    Supports panel data with grouping by ticker/symbol.

    Attributes:
        date_col: Name of date/time column
        close_col: Name of close price column
        group_col: Optional grouping column (e.g., ticker symbol for panel data)
        rsi_periods: RSI periods to calculate (default: [14])
        macd: Whether to calculate MACD (default: True)
        macd_fast: MACD fast period (default: 12)
        macd_slow: MACD slow period (default: 26)
        macd_signal: MACD signal period (default: 9)
        stochastic: Whether to calculate Stochastic Oscillator (default: False)
        stoch_period: Stochastic period (default: 14)
        stoch_smooth_k: Stochastic %K smoothing (default: 3)
        high_col: High price column for Stochastic (default: None)
        low_col: Low price column for Stochastic (default: None)
        handle_na: How to handle NaN values ("keep", "drop", "fill_forward")
    """

    date_col: str
    close_col: str
    group_col: Optional[str] = None
    rsi_periods: Optional[Union[int, List[int]]] = field(default_factory=lambda: [14])
    macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stochastic: bool = False
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    high_col: Optional[str] = None
    low_col: Optional[str] = None
    handle_na: str = "keep"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepOscillators":
        """
        Validate columns and prepare for baking.

        Args:
            data: Training data
            training: Whether this is training data

        Returns:
            PreparedStepOscillators with validated configuration
        """
        # Validate required columns
        required = [self.date_col, self.close_col]
        if self.group_col:
            required.append(self.group_col)

        if self.stochastic:
            if self.high_col is None or self.low_col is None:
                raise ValueError("Stochastic oscillator requires high_col and low_col parameters")
            required.extend([self.high_col, self.low_col])

        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate handle_na parameter
        if self.handle_na not in ["keep", "drop", "fill_forward"]:
            raise ValueError(f"handle_na must be 'keep', 'drop', or 'fill_forward', got: {self.handle_na}")

        # Convert single RSI period to list
        if isinstance(self.rsi_periods, int):
            rsi_periods = [self.rsi_periods]
        else:
            rsi_periods = self.rsi_periods

        return PreparedStepOscillators(
            date_col=self.date_col,
            close_col=self.close_col,
            group_col=self.group_col,
            rsi_periods=rsi_periods,
            macd=self.macd,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            stochastic=self.stochastic,
            stoch_period=self.stoch_period,
            stoch_smooth_k=self.stoch_smooth_k,
            high_col=self.high_col,
            low_col=self.low_col,
            handle_na=self.handle_na,
        )


@dataclass
class PreparedStepOscillators:
    """
    Fitted oscillator calculation step.

    Attributes:
        date_col: Name of date/time column
        close_col: Name of close price column
        group_col: Optional grouping column
        rsi_periods: RSI periods to calculate
        macd: Whether to calculate MACD
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        stochastic: Whether to calculate Stochastic Oscillator
        stoch_period: Stochastic period
        stoch_smooth_k: Stochastic %K smoothing
        high_col: High price column
        low_col: Low price column
        handle_na: How to handle NaN values
    """

    date_col: str
    close_col: str
    group_col: Optional[str]
    rsi_periods: Optional[List[int]]
    macd: bool
    macd_fast: int
    macd_slow: int
    macd_signal: int
    stochastic: bool
    stoch_period: int
    stoch_smooth_k: int
    high_col: Optional[str]
    low_col: Optional[str]
    handle_na: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate oscillator indicators.

        Args:
            data: Data to transform

        Returns:
            DataFrame with oscillator indicator columns added
        """
        try:
            from pytimetk.finance import augment_rsi, augment_macd, augment_stochastic_oscillator
        except ImportError:
            raise ImportError(
                "pytimetk is required for financial oscillator indicators. "
                "Install with: pip install pytimetk"
            )

        result = data.copy()

        # Calculate RSI
        if self.rsi_periods is not None and len(self.rsi_periods) > 0:
            if self.group_col:
                result = (
                    result.groupby(self.group_col, group_keys=False)
                    .apply(
                        lambda x: augment_rsi(
                            x,
                            date_column=self.date_col,
                            close_column=self.close_col,
                            periods=self.rsi_periods,
                            reduce_memory=False,
                        )
                    )
                )
            else:
                result = augment_rsi(
                    result,
                    date_column=self.date_col,
                    close_column=self.close_col,
                    periods=self.rsi_periods,
                    reduce_memory=False,
                )

        # Calculate MACD
        if self.macd:
            if self.group_col:
                result = (
                    result.groupby(self.group_col, group_keys=False)
                    .apply(
                        lambda x: augment_macd(
                            x,
                            date_column=self.date_col,
                            close_column=self.close_col,
                            fast_period=self.macd_fast,
                            slow_period=self.macd_slow,
                            signal_period=self.macd_signal,
                            reduce_memory=False,
                        )
                    )
                )
            else:
                result = augment_macd(
                    result,
                    date_column=self.date_col,
                    close_column=self.close_col,
                    fast_period=self.macd_fast,
                    slow_period=self.macd_slow,
                    signal_period=self.macd_signal,
                    reduce_memory=False,
                )

        # Calculate Stochastic Oscillator
        if self.stochastic:
            if self.group_col:
                result = (
                    result.groupby(self.group_col, group_keys=False)
                    .apply(
                        lambda x: augment_stochastic_oscillator(
                            x,
                            date_column=self.date_col,
                            high_column=self.high_col,
                            low_column=self.low_col,
                            close_column=self.close_col,
                            k_periods=self.stoch_period,
                            d_periods=self.stoch_smooth_k,
                            reduce_memory=False,
                        )
                    )
                )
            else:
                result = augment_stochastic_oscillator(
                    result,
                    date_column=self.date_col,
                    high_column=self.high_col,
                    low_column=self.low_col,
                    close_column=self.close_col,
                    k_periods=self.stoch_period,
                    d_periods=self.stoch_smooth_k,
                    reduce_memory=False,
                )

        # Handle NaN values from lookback windows
        if self.handle_na == "drop":
            result = result.dropna()
        elif self.handle_na == "fill_forward":
            result = result.fillna(method='ffill')
        # "keep" does nothing

        return result
