# Financial Time Series Recipe Steps: pytimetk Integration Project Plan

**Project**: Integrate pytimetk financial indicators into py-tidymodels recipes
**Date**: 2025-10-31
**Status**: Planning Phase
**Estimated Duration**: 3-4 weeks

---

## Executive Summary

Extend py-tidymodels recipes with financial time series feature engineering capabilities by leveraging pytimetk's 16 financial indicators. Implement categorical recipe steps (`step_oscillators()`, `step_volatility()`, etc.) that wrap pytimetk functions while maintaining the prep/bake two-phase pattern.

**Recommended Approach**: Option 2 - Categorical Indicator Steps
**Implementation Complexity**: Medium (3-4 weeks)
**Confidence Level**: High (85%)

---

## Problem Analysis

### Core Challenge

Create recipe steps for financial time series indicators (RSI, MACD, Bollinger Bands, ATR, etc.) that:
- Conform to prep/bake two-phase pattern
- Support panel data (multiple securities)
- Handle lookback window NaN values correctly
- Enable hyperparameter tuning via workflows
- Integrate seamlessly with existing py-tidymodels ecosystem

### Key Constraints

1. **Architectural**: Must conform to prep/bake two-phase pattern
2. **Immutability**: Recipe specifications must be frozen dataclasses
3. **Group-Awareness**: Must support panel data (multiple securities/tickers)
4. **Lookback Dependency**: Financial indicators have varying lookback windows
5. **NaN Handling**: Initial periods will have missing values due to warmup
6. **OHLC Support**: Some indicators need High/Low/Open, not just Close prices

### Critical Success Factors

1. Correct calculation of financial indicators matching industry standards
2. Efficient handling of panel data (multiple tickers in one dataset)
3. Proper NaN handling during warmup periods
4. Compatibility with existing recipe ecosystem
5. Clear documentation for financial practitioners
6. Testability against known financial datasets

---

## Available pytimetk Financial Indicators

### Discovered Indicators (16 total)

**Oscillators & Momentum (7)**:
- `augment_rsi()` - Relative Strength Index
- `augment_macd()` - Moving Average Convergence Divergence
- `augment_ppo()` - Percentage Price Oscillator
- `augment_cmo()` - Chande Momentum Oscillator
- `augment_stochastic_oscillator()` - Stochastic Oscillator
- `augment_fip_momentum()` - FIP Momentum
- `augment_qsmomentum()` - QS Momentum

**Volatility Measures (4)**:
- `augment_atr()` - Average True Range
- `augment_bbands()` - Bollinger Bands
- `augment_ewma_volatility()` - EWMA Volatility
- `augment_rolling_risk_metrics()` - Rolling Risk Metrics

**Trend Indicators (1)**:
- `augment_adx()` - Average Directional Index

**Risk Measures (2)**:
- `augment_drawdown()` - Drawdown calculation
- `augment_hurst_exponent()` - Hurst Exponent

**Pattern Detection (1)**:
- `augment_regime_detection()` - Regime Detection

**Location**: `reference/pytimetk-master/src/pytimetk/finance/`

---

## Multi-Dimensional Analysis

### Technical Perspective

**Prep/Bake Pattern Insight**:
Financial indicators DON'T fit traditional prep/bake pattern cleanly:
- Traditional steps (e.g., scaling): Learn parameters (mean, std) from training data
- Financial indicators: Calculate based on price history, no learning required
- Solution: `prep()` validates columns exist, `bake()` calculates indicators

**Technical Challenges**:
1. **Lookback Windows**: RSI(14) needs 14 periods, creates NaN for initial rows
2. **Multi-Asset Support**: Must handle grouped data (multiple tickers)
3. **OHLC Dependency**: Some indicators need High/Low/Open, not just Close
4. **State Dependency**: Indicators depend on historical values
5. **Runtime Calculation**: Unlike scaling, these calculate on-the-fly

### Business Perspective

**Value Proposition**:
- Automated feature engineering for financial ML models
- Consistent indicator calculations across workflows
- Reproducible trading strategy backtesting
- Integration with modeling pipelines

**Target Users**:
- Quantitative analysts building trading models
- Risk managers creating volatility forecasts
- Data scientists in fintech
- Academic researchers in finance

### User Perspective

**Typical Workflow**:
```python
# Current (manual):
df = df.augment_rsi(date_column="date", close_column="close", periods=14)
df = df.augment_macd(date_column="date", close_column="close")
df = df.augment_bbands(date_column="date", close_column="close")

# Desired (recipe):
rec = recipe(stock_data, formula="return ~ .")
    .step_rsi(date_col="date", close_col="close", periods=[14, 21])
    .step_macd(date_col="date", close_col="close")
    .step_bbands(date_col="date", close_col="close", periods=20)

prepped = rec.prep()
features = prepped.bake(new_data)
```

**Pain Points**:
- Manual feature engineering is error-prone
- Hard to track which indicators were used
- Difficult to apply same transformations to test/validation data
- No built-in handling of panel data

### System Perspective

**Integration Points**:
- Works within workflow pipelines
- Compatible with tune grid search (indicator parameters as hyperparameters)
- Outputs feed into parsnip models
- Must handle datetime indexing correctly

---

## Solution Options

### Option 1: Individual Indicator Steps (Granular Control)

Create separate step for each indicator: `step_rsi()`, `step_macd()`, `step_atr()`, etc.

**Pros**:
- Maximum flexibility and control
- Easy to understand and test individually
- Users can cherry-pick specific indicators
- Clear parameter interfaces

**Cons**:
- Requires implementing ~15 separate steps
- Verbose for users who want multiple indicators
- Code duplication across similar indicators

**Implementation Complexity**: High (5-6 weeks for 15 indicators)

**Example**:
```python
@dataclass
class StepRSI:
    date_col: str
    close_col: str
    periods: Union[int, List[int]] = 14

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepRSI":
        # Validate columns exist
        if self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found")
        return PreparedStepRSI(
            date_col=self.date_col,
            close_col=self.close_col,
            periods=self.periods if isinstance(self.periods, list) else [self.periods]
        )

@dataclass
class PreparedStepRSI:
    date_col: str
    close_col: str
    periods: List[int]

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        from pytimetk.finance import augment_rsi
        result = augment_rsi(
            data=data,
            date_column=self.date_col,
            close_column=self.close_col,
            periods=self.periods
        )
        return result
```

---

### Option 2: Categorical Indicator Steps (Balanced Approach) ⭐ **RECOMMENDED**

Group indicators by category: `step_oscillators()`, `step_volatility()`, `step_trend()`, `step_risk()`.

**Pros**:
- Reduces number of steps from 15 to 4-5
- Logical grouping matches financial practitioner mental models
- Still allows fine-grained control via parameters
- Easier to maintain than 15 separate steps

**Cons**:
- Less granular than individual steps
- Slightly more complex parameter interface
- Must handle multiple indicator types in one step

**Implementation Complexity**: Medium (3-4 weeks for 4 category steps)

**Categories**:
1. **`step_oscillators()`**: RSI, MACD, PPO, CMO, Stochastic
2. **`step_volatility()`**: ATR, Bollinger Bands, EWMA Volatility, Rolling Risk
3. **`step_trend()`**: ADX
4. **`step_risk()`**: Drawdown, Hurst Exponent

**Example**:
```python
@dataclass
class StepOscillators:
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
    handle_na: str = "drop"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepOscillators":
        # Validate required columns
        return PreparedStepOscillators(...)

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        if self.rsi_periods:
            result = augment_rsi(result, self.date_col, self.close_col, self.rsi_periods)
        if self.macd:
            result = augment_macd(result, self.date_col, self.close_col, **self.macd_params)
        # ... etc
        return result
```

---

### Option 3: Unified Financial Features Step (Simplified)

Single `step_financial_features()` that enables/disables indicators via flags.

**Pros**:
- Single step to learn
- Easiest for beginners
- Minimal code to maintain
- Quick to implement

**Cons**:
- Less flexible for advanced users
- Parameter explosion (many optional args)
- Harder to tune individual indicators
- All-or-nothing approach limiting

**Implementation Complexity**: Low (1-2 weeks)

---

### Option 4: Pytimetk Wrapper Step (Delegation Pattern)

Create `step_pytimetk()` that directly wraps pytimetk functions with minimal abstraction.

**Pros**:
- Leverages existing pytimetk implementation
- No duplication of indicator logic
- Easy to add new indicators as pytimetk adds them
- Minimal maintenance

**Cons**:
- Tight coupling to pytimetk API
- Less "recipe-like" interface
- Harder to integrate with tune parameter search
- Breaks abstraction layer

**Implementation Complexity**: Very Low (3-5 days)

---

## Recommendation: Option 2 - Categorical Steps with Phased Rollout

### Rationale

1. **Practitioner-Friendly**: Financial analysts think in terms of indicator categories
2. **Manageable Scope**: 4-5 steps vs 15 individual steps
3. **Flexibility**: Parameters allow fine-tuning without explosion
4. **Maintainability**: Grouped logic easier to test and update
5. **Extensibility**: Easy to add new indicators to existing categories

### Implementation Roadmap

#### Phase 1: Foundation (Week 1)
- Create base financial step infrastructure
- Implement `step_oscillators()` with RSI, MACD
- Build comprehensive tests (model after test_varmax_reg.py and test_auto_arima.py)
- Document prep/bake pattern for financial indicators

**Deliverables**:
- `py_recipes/steps/financial_oscillators.py`
- `tests/test_recipes/test_step_oscillators.py`
- Basic documentation

#### Phase 2: Core Indicators (Week 2)
- Add Bollinger Bands, Stochastic to `step_oscillators()`
- Implement `step_volatility()` with ATR, EWMA
- Create panel data tests (multiple tickers)
- Handle NaN in lookback windows

**Deliverables**:
- `py_recipes/steps/financial_volatility.py`
- `tests/test_recipes/test_step_volatility.py`
- Panel data test suite

#### Phase 3: Advanced Features (Week 3)
- Implement `step_trend()` with ADX
- Implement `step_risk()` with Drawdown
- Add support for OHLC columns (not just Close)
- Integration with workflows

**Deliverables**:
- `py_recipes/steps/financial_trend.py`
- `py_recipes/steps/financial_risk.py`
- Workflow integration tests

#### Phase 4: Polish & Documentation (Week 4)
- Create demo notebook showing financial ML pipeline
- Performance optimization for large panels
- Parameter tuning examples
- Update CLAUDE.md

**Deliverables**:
- `examples/financial_indicators_demo.ipynb`
- Updated CLAUDE.md
- Performance benchmarks

---

## Critical Design Decisions

### 1. Handling Lookback NaNs

**Options**:
- A: Drop NaN rows (default)
- B: Keep NaN rows
- C: Forward fill

**Recommendation**: Configurable with "drop" as default

```python
handle_na: str = "drop"  # "drop", "keep", "fill_forward"
```

**Rationale**: Financial practitioners expect NaN for insufficient history, but need option to preserve row count for certain workflows.

---

### 2. Panel Data Support

**Implementation**:
```python
group_col: Optional[str] = None  # e.g., "ticker" or "symbol"

# In bake():
if group_col:
    result = data.groupby(group_col).apply(
        lambda x: augment_rsi(x, date_col, close_col, periods)
    ).reset_index(drop=True)
else:
    result = augment_rsi(data, date_col, close_col, periods)
```

**Rationale**: Essential for multi-asset portfolios and cross-sectional strategies.

---

### 3. Column Naming Convention

Follow pytimetk conventions:
- RSI(14) on 'close' → `close_rsi_14`
- MACD on 'close' → `close_macd`, `close_macd_signal`, `close_macd_hist`
- Bollinger on 'close' → `close_bbands_20_2_lower`, `close_bbands_20_2_middle`, `close_bbands_20_2_upper`

**Rationale**: Consistency with pytimetk, clear parameter identification.

---

## Complete Implementation Example: step_oscillators()

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import pandas as pd

@dataclass
class StepOscillators:
    """
    Calculate oscillator-based technical indicators.

    Attributes:
        date_col: Date column name
        close_col: Closing price column name
        group_col: Optional grouping column (e.g., 'ticker' for panel data)
        rsi_periods: RSI lookback periods (None = skip RSI)
        macd: Calculate MACD (default True)
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        stochastic: Calculate Stochastic Oscillator
        stoch_period: Stochastic lookback period
        handle_na: How to handle NaN from lookback ("drop", "keep")
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
    handle_na: str = "drop"

    def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepOscillators":
        """Validate columns and prepare step."""
        # Validate required columns
        required = [self.date_col, self.close_col]
        if self.group_col:
            required.append(self.group_col)

        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return PreparedStepOscillators(
            date_col=self.date_col,
            close_col=self.close_col,
            group_col=self.group_col,
            rsi_periods=self.rsi_periods,
            macd=self.macd,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            stochastic=self.stochastic,
            stoch_period=self.stoch_period,
            handle_na=self.handle_na
        )


@dataclass
class PreparedStepOscillators:
    """Prepared oscillators step - ready to calculate indicators."""
    date_col: str
    close_col: str
    group_col: Optional[str]
    rsi_periods: Optional[Union[int, List[int]]]
    macd: bool
    macd_fast: int
    macd_slow: int
    macd_signal: int
    stochastic: bool
    stoch_period: int
    handle_na: str

    def bake(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators."""
        from pytimetk.finance import augment_rsi, augment_macd, augment_stochastic_oscillator

        result = data.copy()

        # Calculate RSI if requested
        if self.rsi_periods is not None:
            if self.group_col:
                result = result.groupby(self.group_col).apply(
                    lambda x: augment_rsi(
                        x, self.date_col, self.close_col,
                        periods=self.rsi_periods
                    )
                ).reset_index(drop=True)
            else:
                result = augment_rsi(
                    result, self.date_col, self.close_col,
                    periods=self.rsi_periods
                )

        # Calculate MACD if requested
        if self.macd:
            if self.group_col:
                result = result.groupby(self.group_col).apply(
                    lambda x: augment_macd(
                        x, self.date_col, self.close_col,
                        fast_period=self.macd_fast,
                        slow_period=self.macd_slow,
                        signal_period=self.macd_signal
                    )
                ).reset_index(drop=True)
            else:
                result = augment_macd(
                    result, self.date_col, self.close_col,
                    fast_period=self.macd_fast,
                    slow_period=self.macd_slow,
                    signal_period=self.macd_signal
                )

        # Calculate Stochastic if requested
        if self.stochastic:
            if self.group_col:
                result = result.groupby(self.group_col).apply(
                    lambda x: augment_stochastic_oscillator(
                        x, self.date_col,
                        high_column="high",  # Requires OHLC data
                        low_column="low",
                        close_column=self.close_col,
                        period=self.stoch_period
                    )
                ).reset_index(drop=True)
            else:
                result = augment_stochastic_oscillator(
                    result, self.date_col,
                    high_column="high",
                    low_column="low",
                    close_column=self.close_col,
                    period=self.stoch_period
                )

        # Handle NaN rows from lookback
        if self.handle_na == "drop":
            result = result.dropna()

        return result
```

---

## Risk Assessment

### High Risk ✗

**Risk**: Incorrect indicator calculations
**Impact**: Wrong signals, bad trading decisions, user distrust
**Mitigation**:
- Extensive tests against known financial datasets
- Validation against TA-Lib reference implementations
- Test with real market data from multiple time periods

**Risk**: Panel data edge cases
**Impact**: Incorrect calculations for multi-ticker datasets
**Mitigation**:
- Comprehensive multi-ticker test suite
- Test with varying history lengths per ticker
- Validate groupby operations preserve order

### Medium Risk ⚠

**Risk**: Performance with large datasets
**Impact**: Slow preprocessing, user frustration
**Mitigation**:
- Profile with 1M+ row datasets
- Implement lazy evaluation where possible
- Consider chunking for very large panels

**Risk**: NaN handling confusion
**Impact**: Unexpected row drops, data loss
**Mitigation**:
- Clear documentation of default behavior
- Warning messages when rows are dropped
- Sensible defaults ("drop" for most use cases)

### Low Risk ✓

**Risk**: API changes in pytimetk
**Impact**: Breaking changes in recipe steps
**Mitigation**:
- Version pinning for pytimetk dependency
- Wrapper abstractions isolate breaking changes
- Comprehensive test suite catches regressions

---

## Success Metrics

1. **Correctness**: All indicators match TA-Lib/pytimetk reference implementations (100% accuracy)
2. **Performance**: Handle 1M+ row panels in <30 seconds
3. **Usability**: Financial practitioners can build ML pipelines in <50 lines of code
4. **Coverage**: 80%+ of common financial indicators available (12+ of 16 discovered indicators)
5. **Test Coverage**: >90% code coverage with comprehensive edge case tests
6. **Documentation**: Complete examples for all 4 step categories

---

## Test Strategy

### Unit Tests (Model after test_varmax_reg.py and test_auto_arima.py)

```python
class TestStepOscillators:
    """Test step_oscillators specification and preparation"""

    def test_default_spec(self):
        """Test default oscillators specification"""
        step = StepOscillators(date_col="date", close_col="close")
        assert step.rsi_periods == [14]
        assert step.macd == True
        assert step.handle_na == "drop"

    def test_prep_validates_columns(self):
        """Test prep() validates required columns exist"""
        data = pd.DataFrame({"wrong_col": [1, 2, 3]})
        step = StepOscillators(date_col="date", close_col="close")

        with pytest.raises(ValueError, match="Missing columns"):
            step.prep(data)

    def test_bake_calculates_rsi(self, stock_data):
        """Test bake() calculates RSI correctly"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            rsi_periods=[14]
        )
        prepped = step.prep(stock_data)
        result = prepped.bake(stock_data)

        assert "close_rsi_14" in result.columns
        assert result["close_rsi_14"].between(0, 100).all()

    def test_panel_data_support(self, multi_ticker_data):
        """Test panel data with multiple tickers"""
        step = StepOscillators(
            date_col="date",
            close_col="close",
            group_col="ticker",
            rsi_periods=[14]
        )
        prepped = step.prep(multi_ticker_data)
        result = prepped.bake(multi_ticker_data)

        # Each ticker should have RSI calculated independently
        for ticker in multi_ticker_data["ticker"].unique():
            ticker_data = result[result["ticker"] == ticker]
            assert "close_rsi_14" in ticker_data.columns
```

### Integration Tests

```python
class TestFinancialWorkflow:
    """Test financial indicators in complete workflow"""

    def test_workflow_with_oscillators(self, stock_data):
        """Test workflow with oscillator features"""
        rec = recipe(stock_data, formula="return ~ .")
            .step_oscillators(
                date_col="date",
                close_col="close",
                rsi_periods=[14, 21],
                macd=True
            )

        wf = workflow()
            .add_recipe(rec)
            .add_model(linear_reg())

        fit = wf.fit(stock_data)
        predictions = fit.predict(test_data)

        assert predictions is not None
```

---

## Alternative Perspectives

### Contrarian View

"Why wrap pytimetk at all? Users can call it directly."

### Counter-Arguments

Recipe steps provide:
1. **Reproducibility**: Same transformations on train/test/validation splits
2. **Pipeline Integration**: Works seamlessly with workflows and tune
3. **Hyperparameter Search**: Indicator parameters become tunable in grid search
4. **Consistency**: Standard interface across all preprocessing steps
5. **Workflow Tracking**: Clear record of feature engineering in model metadata

---

## Future Considerations

### Potential Enhancements (Post-MVP)

1. **Custom Indicator Support**: Allow users to define custom indicator functions
2. **Indicator Combinations**: Pre-built indicator suites (e.g., "momentum_suite")
3. **Automatic Parameter Tuning**: Grid search over indicator parameters
4. **Visualization Integration**: Built-in plotting of indicators on price charts
5. **Backtesting Support**: Direct integration with backtesting frameworks
6. **Real-time Calculation**: Streaming indicator updates for live trading

### Research Areas

1. **Optimal Indicator Selection**: Which indicators provide most signal for ML models?
2. **Parameter Sensitivity**: How sensitive are models to indicator parameters?
3. **Computational Efficiency**: Vectorized implementations for large-scale backtesting
4. **Alternative Libraries**: Integration with other TA libraries (TA-Lib, pandas-ta)

---

## Meta-Analysis

### Confidence Level

**High (85%)** - Strong understanding of:
- pytimetk API and available indicators
- py-tidymodels recipe architecture
- Financial practitioner workflows
- Implementation patterns from existing steps

### Uncertainties

1. **User Preference**: Granular vs. grouped steps (mitigate with user survey/beta testing)
2. **Performance**: Bottlenecks with very large panels (needs profiling on real data)
3. **Tune Integration**: Complexity of parameter grids (needs prototype)

### Acknowledged Biases

1. **Categorical Grouping Preference**: May reflect implementer's mental model, not users'
2. **Stock Data Focus**: May miss crypto/forex-specific requirements
3. **Python Ecosystem**: Assumes pytimetk is best choice vs. TA-Lib or pandas-ta

---

## Dependencies

### Required Packages

- `pytimetk>=0.3.0` - Core financial indicators
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical operations

### Internal Dependencies

- `py_recipes.recipe` - Base recipe class
- `py_recipes.steps` - Existing step infrastructure
- Test fixtures from `test_varmax_reg.py` and `test_auto_arima.py` as models

---

## Next Steps

1. **Approval & Refinement**: Review this plan, gather feedback
2. **Phase 1 Kickoff**: Begin implementation of `step_oscillators()`
3. **Test Framework Setup**: Create financial data fixtures and test utilities
4. **Documentation Template**: Establish documentation standards for financial steps

---

## References

- pytimetk documentation: `reference/pytimetk-master/`
- Existing recipe steps: `py_recipes/steps/scaling.py`
- Test patterns: `tests/test_parsnip/test_varmax_reg.py`, `test_auto_arima.py`
- Recipe architecture: Current 51 steps across 8 categories

---

**Document Version**: 1.0
**Last Updated**: 2025-10-31
**Next Review**: After Phase 1 completion
