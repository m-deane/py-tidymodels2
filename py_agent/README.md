# py_agent: AI-Powered Forecasting Agent

AI agent system for automated time series forecasting workflow generation using py-tidymodels.

## Overview

`py_agent` is an intelligent agent that can automatically:
- Analyze temporal patterns in your data
- Recommend appropriate forecasting models
- Generate preprocessing recipes
- Create complete py-tidymodels workflows
- Diagnose performance issues and suggest improvements

## Features

### Phase 1: MVP (v0.1.0) - Rule-Based ✅ COMPLETE

- ✅ Rule-based workflow generation (no LLM required)
- ✅ Support for **ALL 23 model types** (baseline, linear, tree, SVM, neural nets, time series, hybrid)
- ✅ Automated data analysis (seasonality, trend, autocorrelation)
- ✅ Basic recipe generation (10+ preprocessing steps)
- ✅ Performance diagnostics and debugging
- ✅ Conversational session interface
- 🎯 Target: 70%+ workflow success rate
- 💰 Cost: $0 (no API calls)

### Phase 2: LLM Integration (v0.2.0) - ✅ COMPLETE

- ✅ Claude Sonnet 4.5 integration via Anthropic SDK
- ✅ Multi-agent architecture with specialized agents:
  - **DataAnalyzer**: LLM-enhanced temporal pattern analysis
  - **FeatureEngineer**: Advanced recipe optimization with reasoning
  - **ModelSelector**: Intelligent model selection with trade-off analysis
  - **Orchestrator**: High-level workflow coordination
- ✅ Tool-calling architecture for structured LLM interactions
- ✅ Budget management and cost tracking ($100/day default)
- ✅ Dual-mode support: Switch between rule-based (Phase 1) and LLM-enhanced (Phase 2)
- ✅ Comprehensive test coverage (50+ tests for Phase 2)
- 💰 Cost: ~$4-10 per workflow (with LLM)

### Phase 3: Advanced Features (In Progress)

- ✅ **Phase 3.1 Complete**: Model Expansion - All 23 model types!
- ✅ **Phase 3.2 Complete**: Enhanced Recipe Generation - Intelligent 51-step selection!
- ⏳ Phase 3.3: Multi-model WorkflowSet orchestration
- ⏳ Phase 3.4: RAG knowledge base with 500+ forecasting examples
- ⏳ Phase 3.5: Autonomous iteration and self-improvement

## Quick Start

### Phase 1: Rule-Based Mode (Default)

```python
from py_agent import ForecastAgent
import pandas as pd

# Load your data
sales_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=365, freq='D'),
    'sales': [...],  # Your sales values
    'store_id': [...]  # Optional: for grouped forecasting
})

# Initialize agent in rule-based mode (no API costs)
agent = ForecastAgent(verbose=True)

# Generate workflow from natural language
workflow = agent.generate_workflow(
    data=sales_data,
    request="Forecast next quarter sales for each store with seasonality"
)

# Fit and predict
fit = workflow.fit(sales_data)
predictions = fit.predict(future_data)
```

### Phase 2: LLM-Enhanced Mode

```python
from py_agent import ForecastAgent
import os

# Set API key (or use ANTHROPIC_API_KEY environment variable)
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Initialize agent in LLM mode
agent = ForecastAgent(
    use_llm=True,              # Enable LLM-enhanced reasoning
    model="claude-sonnet-4.5",  # LLM model to use
    budget_per_day=100.0,       # Daily budget in USD
    verbose=True
)

# Generate workflow with LLM orchestration
workflow = agent.generate_workflow(
    data=sales_data,
    request="Forecast sales with weekly seasonality and account for holidays",
    constraints={
        'max_train_time': 60,  # Maximum 60 seconds training
        'interpretability': 'high'  # Prefer interpretable models
    }
)

# Access LLM reasoning
print("Data Analysis Insights:", agent.last_workflow_info['data_analysis_reasoning'])
print("Model Selection Reasoning:", agent.last_workflow_info['model_selection_reasoning'])
print("Feature Engineering Reasoning:", agent.last_workflow_info['feature_engineering_reasoning'])
print(f"API Cost: ${agent.llm_client.total_cost:.4f}")

# Fit and predict
fit = workflow.fit(sales_data)
predictions = fit.predict(future_data)
```

## Architecture

### Core Components

```
py_agent/
├── tools/              # Analysis and recommendation functions (Phase 1)
│   ├── data_analysis.py       # Temporal pattern detection
│   ├── model_selection.py     # Model recommendation engine
│   ├── recipe_generation.py   # Preprocessing recipe creation
│   ├── workflow_execution.py  # Workflow fitting and evaluation
│   └── diagnostics.py         # Performance analysis
│
├── llm/                # LLM integration (Phase 2)
│   ├── client.py              # Anthropic SDK wrapper with budget management
│   └── tool_schemas.py        # Tool calling schemas for LLM
│
├── agents/             # Agent implementations
│   ├── forecast_agent.py      # Main ForecastAgent class (dual-mode)
│   └── specialized_agents.py  # Multi-agent system (Phase 2)
│       ├── BaseAgent          # Base class for all agents
│       ├── DataAnalyzer       # LLM-enhanced data analysis
│       ├── FeatureEngineer    # LLM-enhanced recipe optimization
│       ├── ModelSelector      # LLM-enhanced model selection
│       └── Orchestrator       # High-level coordination
│
└── README.md           # This file
```

### Tools

#### Data Analysis Tools
- `analyze_temporal_patterns()` - Comprehensive temporal analysis
- `detect_seasonality()` - Seasonal pattern detection
- `detect_trend()` - Trend identification
- `calculate_autocorrelation()` - Lag correlation analysis

#### Model Selection Tools
- `suggest_model()` - Recommend models based on data characteristics
- `get_model_profiles()` - Access model capability profiles

#### Recipe Generation Tools
- `create_recipe()` - Generate preprocessing code
- `get_recipe_templates()` - Access predefined recipe templates

#### Workflow Execution Tools
- `fit_workflow()` - Execute workflow on data
- `evaluate_workflow()` - Test workflow performance

#### Diagnostic Tools
- `diagnose_performance()` - Identify performance issues
- `detect_overfitting()` - Check train/test performance gap

## Usage Examples

### Example 1: Single-Shot Workflow Generation

```python
agent = ForecastAgent()

workflow = agent.generate_workflow(
    data=sales_data,
    request="Forecast daily sales with weekly seasonality",
    constraints={
        'max_train_time': 60,  # 1 minute max
        'interpretability': 'high'
    }
)

fit = workflow.fit(train_data)
predictions = fit.predict(test_data)
```

### Example 2: Conversational Session

```python
agent = ForecastAgent()
session = agent.start_session()

# Multi-turn conversation
session.send("I need to forecast sales")
session.send("Daily data for 50 stores")
session.send("3 years of history")

# Generate workflow from conversation
workflow = session.get_workflow()
```

### Example 3: Performance Debugging

```python
# Fit a workflow
fit = workflow.fit(train_data)

# Debug performance
diagnostics = agent.debug_session(fit, test_data)

# View issues and recommendations
for issue in diagnostics['issues_detected']:
    print(f"⚠️  {issue['type']}: {issue['evidence']}")
    print(f"💡 {issue['recommendation']}\n")
```

## Data Analysis Capabilities

The agent automatically analyzes your data to identify:

- **Frequency**: Daily, weekly, monthly, quarterly, yearly
- **Seasonality**: Detection, period, and strength (0-1)
- **Trend**: Direction (increasing/decreasing/stable) and significance
- **Autocorrelation**: At lags 1, 7, 30 (configurable)
- **Data Quality**: Missing value rate, outlier rate

Example output:
```python
{
    'frequency': 'daily',
    'seasonality': {
        'detected': True,
        'period': 7,
        'strength': 0.73
    },
    'trend': {
        'direction': 'increasing',
        'strength': 0.45,
        'significant': True
    },
    'autocorrelation': {
        'lag_1': 0.82,
        'lag_7': 0.45
    },
    'missing_rate': 0.08,
    'outlier_rate': 0.02
}
```

## Model Recommendation System

The agent uses a rule-based system to match data characteristics with model capabilities:

### Supported Models (All 23 Models)

#### Baseline Models (2)
- **`null_model`** - Null baseline (mean/median)
- **`naive_reg`** - Naive forecasting baselines

#### Linear & Generalized Models (3)
- **`linear_reg`** - Linear Regression (HIGH interpretability, VERY FAST)
- **`poisson_reg`** - Poisson Regression (for count data)
- **`gen_additive_mod`** - Generalized Additive Models (nonlinear trends)

#### Tree-Based Models (3)
- **`decision_tree`** - Decision Trees (HIGH interpretability)
- **`rand_forest`** - Random Forest (complex patterns, robustness)
- **`boost_tree`** - Gradient Boosting (XGBoost, LightGBM, CatBoost)

#### Support Vector Machines (2)
- **`svm_rbf`** - RBF Kernel SVM (nonlinear patterns)
- **`svm_linear`** - Linear Kernel SVM (linear patterns)

#### Instance-Based & Adaptive (3)
- **`nearest_neighbor`** - k-Nearest Neighbors
- **`mars`** - Multivariate Adaptive Regression Splines
- **`mlp`** - Multi-Layer Perceptron Neural Network

#### Time Series Models (5)
- **`arima_reg`** - ARIMA/SARIMAX (autocorrelation, short-term forecasts)
- **`prophet_reg`** - Facebook Prophet (seasonality, holidays, MEDIUM interpretability)
- **`exp_smoothing`** - Exponential Smoothing / ETS
- **`seasonal_reg`** - STL Decomposition Models
- **`varmax_reg`** - Multivariate VARMAX (multiple outcomes)

#### Hybrid Time Series (2)
- **`arima_boost`** - ARIMA + XGBoost Hybrid
- **`prophet_boost`** - Prophet + XGBoost Hybrid

#### Recursive Forecasting (1)
- **`recursive_reg`** - ML models for multi-step forecasting (skforecast)

#### Generic Hybrid & Manual (2)
- **`hybrid_model`** - Combines any two models (residual, sequential, weighted, custom_data strategies)
- **`manual_reg`** - User-specified coefficients (for external forecasts)

### Recommendation Logic

```python
# Strong seasonality → Prophet
if seasonality_strength > 0.6:
    recommend('prophet_reg', confidence=0.85)

# Linear trend only → Linear Regression
if trend_significant and not seasonality_detected:
    recommend('linear_reg', confidence=0.75)

# Complex patterns → Random Forest
if feature_interactions or nonlinear_patterns:
    recommend('rand_forest', confidence=0.80)
```

## Recipe Generation

### Enhanced Recipe Generation (Phase 3.2) ✅ COMPLETE

Intelligently generates preprocessing recipes using **all 51 available recipe steps** with an 8-phase pipeline:

#### 8-Phase Preprocessing Pipeline

1. **Phase 1: Data Cleaning**
   - Remove rows with infinite values (`step_naomit()`)
   - Triggered by outlier rate > 0

2. **Phase 2: Imputation**
   - **Low missing rate (<5%)**: Median imputation
   - **Moderate missing rate (5-15%)**: Linear interpolation (time series) or median (ML)
   - **High missing rate (>15%)**: KNN imputation with 5 neighbors

3. **Phase 3: Feature Engineering**
   - **Date features**: Extract temporal features based on frequency and domain
   - **Polynomial features**: For linear models with nonlinear trends (<15 features)
   - **Interaction terms**: For linear models with 2-10 features

4. **Phase 4: Transformations**
   - **YeoJohnson transformation**: For models assuming normality (linear, SVM, k-NN)
   - Handles negative values better than BoxCox

5. **Phase 5: Filtering & Dimensionality Reduction**
   - **Zero-variance filter**: Always applied (`step_zv()`)
   - **Correlation filter**: For linear models (threshold=0.9, multicollinearity removal)
   - **PCA**: For high-dimensional data (>20 features OR features > 50% of observations)
     - Excludes: Time series models (loses interpretability)
     - Excludes: Interpretable models (linear_reg, decision_tree, mars)
     - Components: min(n_features × 0.8, 20)

6. **Phase 6: Normalization/Scaling**
   - Applied to: Distance-based models (k-NN, SVM), neural networks, linear models, tree models
   - Uses `step_normalize()` for mean=0, std=1

7. **Phase 7: Encoding**
   - One-hot encoding for ML models (`step_dummy()`)
   - Skipped for time series models (prophet, ARIMA)

8. **Phase 8: Final Cleanup**
   - Remove any NAs introduced by preprocessing (`step_naomit()`)

#### Intelligent Preprocessing Decisions

The system uses 6 decision functions to optimize preprocessing:

- **`_needs_polynomial_features()`**: Adds polynomial features for linear models with nonlinear trends (max 15 features to avoid curse of dimensionality)
- **`_needs_interactions()`**: Adds interaction terms for linear models with 2-10 features (avoids explosion)
- **`_needs_transformation()`**: Applies YeoJohnson for models assuming normality
- **`_needs_correlation_filter()`**: Removes multicollinear features for linear models
- **`_needs_dimensionality_reduction()`**: Applies PCA for high-dimensional data (>20 features)
- **`_needs_normalization()`**: Normalizes features for distance-based and neural network models

#### 17 Domain-Specific Templates

Pre-configured recipes for common use cases:

**Basic (3):**
- `minimal`: Imputation only
- `standard_ml`: Imputation + normalization + encoding
- `time_series`: Linear interpolation for time series

**Retail & E-commerce (3):**
- `retail_daily`: Daily sales with holidays
- `retail_weekly`: Weekly sales with promotions
- `ecommerce_hourly`: Hourly traffic/conversions

**Energy & Utilities (2):**
- `energy_hourly`: Energy load forecasting
- `solar_generation`: Solar power generation

**Finance & Economics (2):**
- `finance_daily`: Financial time series (no imputation)
- `stock_prices`: Stock price prediction with log returns

**Healthcare (1):**
- `patient_volume`: Hospital patient volume

**Transportation & Logistics (2):**
- `demand_forecasting`: Product/service demand
- `traffic_volume`: Traffic congestion prediction

**High-Dimensional & Specialized (4):**
- `high_dimensional`: PCA + correlation filter (>20 features)
- `text_features`: TF-IDF dimension reduction
- `iot_sensors`: Correlated sensor data filtering

#### Example Generated Recipes

**Simple ML Model (Linear Regression):**
```python
from py_recipes import recipe
from py_recipes.selectors import all_numeric_predictors, all_nominal_predictors

rec = (recipe(data, formula)
    .step_impute_median(all_numeric())
    .step_date('date', features=['dow', 'month'])
    .step_YeoJohnson(all_numeric_predictors())
    .step_zv(all_predictors())
    .step_select_corr(all_numeric_predictors(), threshold=0.9, method='multicollinearity')
    .step_normalize(all_numeric_predictors())
    .step_dummy(all_nominal_predictors())
    .step_naomit())
```

**High-Dimensional Data (Random Forest):**
```python
rec = (recipe(data, formula)
    .step_naomit()
    .step_impute_median(all_numeric())
    .step_zv(all_predictors())
    .step_pca(all_numeric_predictors(), num_comp=20)  # Reduce from 50 to 20
    .step_normalize(all_numeric_predictors())
    .step_dummy(all_nominal_predictors())
    .step_naomit())
```

**Time Series Model (Prophet):**
```python
rec = (recipe(data, formula)
    .step_impute_linear(all_numeric()))  # Minimal preprocessing
```

## Testing

Comprehensive test suite with 50+ tests:

```bash
# Run all agent tests
pytest tests/test_agent/ -v

# Run specific test modules
pytest tests/test_agent/test_data_analysis.py -v
pytest tests/test_agent/test_model_selection.py -v

# Run with coverage
pytest tests/test_agent/ --cov=py_agent --cov-report=html
```

## Roadmap

### Phase 1: MVP ✅ COMPLETE
- ✅ Rule-based workflow generation
- ✅ 3 model types supported
- ✅ Basic recipe generation
- ✅ Performance diagnostics
- ✅ 50+ tests passing
- ✅ Demo notebook with 4 examples

### Phase 2: LLM Integration ✅ COMPLETE
- ✅ Claude Sonnet 4.5 integration via Anthropic SDK
- ✅ LLM-based reasoning for model selection
- ✅ Advanced recipe optimization with reasoning
- ✅ Natural language explanations
- ✅ Multi-agent architecture (DataAnalyzer, FeatureEngineer, ModelSelector, Orchestrator)
- ✅ Tool-calling pattern for structured interactions
- ✅ Budget management and cost tracking
- ✅ 50+ Phase 2 tests passing
- ✅ Dual-mode support (switch between rule-based and LLM)

### Phase 3: Advanced Features (In Progress)
- ✅ **Phase 3.1 Complete**: Model Expansion - All 23 model types!
  - Baseline, Linear/GLM, Tree-based, SVM, Neural nets, Time series, Hybrid models
- ✅ **Phase 3.2 Complete**: Enhanced Recipe Generation - Intelligent 51-step selection!
  - 8-phase preprocessing pipeline
  - 6 intelligent decision functions
  - 17 domain-specific templates
  - Model-specific optimizations (PCA, polynomial, interactions, transformations)
- ⏳ RAG knowledge base with 500+ forecasting examples
- ⏳ Multi-model WorkflowSet orchestration
- ⏳ Ensemble recommendations
- ⏳ Autonomous iteration and self-improvement
- ⏳ Performance profiling and auto-optimization

## Performance Targets

Based on research report (`.claude_plans/AI_AGENT_RESEARCH_REPORT.md`):

- **Success Rate**: 70%+ of generated workflows execute successfully
- **Accuracy**: 60%+ produce results within 20% of expert baseline
- **Time Savings**: 80% reduction (4 hours → 45 minutes for beginners)
- **Cost**: <$10 per forecast (when LLM integrated)

## Dependencies

### Phase 1 (Rule-Based)
- pandas >= 2.0
- numpy >= 1.24
- statsmodels >= 0.14
- scipy >= 1.10
- py_tidymodels (all packages: hardhat, parsnip, recipes, workflows, etc.)

### Phase 2 (LLM Integration) - Additional Dependencies
- anthropic >= 0.40.0 (Claude SDK)
- python-dotenv >= 1.0.0 (for environment variable management)

## Contributing

This is an MVP implementation. Future enhancements welcome:
- Additional model types support
- Domain-specific templates
- Advanced diagnostic capabilities
- Integration with external LLM APIs

## License

Same as py-tidymodels parent project.

## References

- Research Report: `.claude_plans/AI_AGENT_RESEARCH_REPORT.md`
- Project Guidelines: `.claude/CLAUDE.md`
- py-tidymodels Documentation: See parent README
