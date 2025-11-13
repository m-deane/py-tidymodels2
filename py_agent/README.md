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
- ✅ **Phase 3.3 Complete**: Multi-Model WorkflowSet Orchestration - Automatic model comparison!
- ✅ **Phase 3.4 Complete**: RAG knowledge base with 8 foundational forecasting examples!
- ✅ **Phase 3.5 Complete**: Autonomous iteration with try-evaluate-improve loops!

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

### Phase 3.3: Multi-Model Comparison

```python
from py_agent import ForecastAgent
import pandas as pd

# Load your data
sales_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=730, freq='D'),
    'sales': [...],
    'temperature': [...],
    'promotion': [...]
})

# Initialize agent
agent = ForecastAgent(verbose=True)

# Compare top 5 models automatically
results = agent.compare_models(
    data=sales_data,
    request="Forecast daily sales with seasonality",
    n_models=5,               # Compare top 5 recommended models
    cv_strategy='time_series', # Use time series CV
    n_folds=5,                # 5 CV folds
    date_column='date',
    return_ensemble=True      # Also get ensemble recommendation
)

# View rankings
print("\n🏆 Model Rankings:")
print(results['rankings'])
#      rank           wflow_id     mean  std_err
# 0       1     prophet_reg_1   12.45     0.82
# 1       2      arima_reg_4   13.21     1.15
# 2       3     linear_reg_2   15.33     1.42
# 3       4   rand_forest_3   16.78     1.88
# 4       5    boost_tree_5   17.12     2.01

# Get best model
best_model_id = results['best_model_id']
print(f"\n✅ Best Model: {best_model_id}")

# Access the WorkflowSet
wf_set = results['workflowset']

# Fit best model on full data
best_workflow = wf_set[best_model_id]
fit = best_workflow.fit(sales_data)
predictions = fit.predict(future_data)

# Get ensemble recommendation
if results.get('ensemble_recommendation'):
    ensemble = results['ensemble_recommendation']
    print(f"\n🤝 Ensemble: {', '.join(ensemble['model_ids'])}")
    print(f"   Expected RMSE: {ensemble['expected_performance']:.2f}")
    print(f"   Diversity: {ensemble['diversity_score']:.2f}")
    print(f"   Type: {ensemble['ensemble_type']}")
```

**Key Benefits:**
- **Automatic Comparison**: Evaluates 5+ models in parallel
- **Cross-Validation**: Robust performance estimates with CV
- **Ranking**: Models sorted by performance (RMSE, MAE, R²)
- **Ensemble Recommendations**: Suggests optimal model combinations
- **Time Savings**: 1-2 hours of manual model testing → 5 minutes automated

### Phase 3.4: RAG Knowledge Base (Example-Driven Recommendations)

```python
from py_agent import ForecastAgent
import pandas as pd

# Load your data
sales_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=730, freq='D'),
    'sales': [...],  # Daily sales with weekly seasonality
    'temperature': [...],
    'promotion': [...]
})

# Initialize agent with RAG enabled
agent = ForecastAgent(verbose=True, use_rag=True)
# Output: ✅ RAG knowledge base initialized with 8 examples

# Generate workflow with RAG-enhanced recommendations
workflow = agent.generate_workflow(
    data=sales_data,
    request="Forecast daily sales with strong weekly patterns"
)

# RAG automatically:
# 1. Searches knowledge base for similar forecasting scenarios
# 2. Retrieves relevant examples (e.g., retail daily sales with seasonality)
# 3. Boosts confidence for models that worked in similar cases
# 4. Shows key lessons learned from comparable scenarios

# Example verbose output:
# 📚 Searching knowledge base for similar examples...
# ✓ Found 3 similar examples:
#   1. Retail Daily Sales - Strong Weekly Seasonality (similarity: 0.87)
#      E-commerce platform with 2 years of daily sales data...
#      Domain: retail, Difficulty: easy
#   2. Healthcare Patient Volume - Daily Admissions (similarity: 0.72)
#      Daily patient admissions for regional hospital...
#      Domain: healthcare, Difficulty: easy
#   3. Transportation Highway Traffic - Hourly Counts (similarity: 0.68)
#      Hourly vehicle counts on major highway...
#      Domain: transportation, Difficulty: medium
#
# 💡 RAG-recommended models:
#   • prophet_reg (confidence: 0.92)
#   • arima_reg (confidence: 0.78)
#   • linear_reg (confidence: 0.65)
#
# 💭 Key Lessons from Similar Cases:
#   • Prophet excels with weekly seasonality and holiday effects
#   • Add is_holiday feature for better promotional handling
#   • Log transformation helped stabilize variance during promotions

# Fit and predict
fit = workflow.fit(sales_data)
predictions = fit.predict(future_data)
```

**Advanced RAG Usage: Direct API**

```python
from py_agent.knowledge import ExampleLibrary, RAGRetriever, DEFAULT_LIBRARY_PATH

# Load example library
library = ExampleLibrary(DEFAULT_LIBRARY_PATH)
print(f"Loaded {len(library)} examples")  # 8 foundational examples

# Create retriever
retriever = RAGRetriever(library)

# Query by data characteristics
data_chars = {
    'frequency': 'daily',
    'seasonality': {'detected': True, 'strength': 0.8},
    'trend': {'direction': 'increasing'},
    'n_observations': 730
}

results = retriever.retrieve_by_data_characteristics(data_chars, top_k=3)

for result in results:
    print(f"\n{result.rank}. {result.example.title}")
    print(f"   Similarity: {result.similarity_score:.2f}")
    print(f"   Domain: {result.example.domain}")
    print(f"   Recommended: {result.example.recommended_models}")

# Extract model recommendations from similar examples
models = retriever.get_model_recommendations_from_examples(results, top_n=3)
print("\nTop models from similar cases:")
for model, score in models:
    print(f"  {model}: {score:.2f}")

# Extract key lessons
lessons = retriever.get_key_lessons(results)
print("\nKey lessons:")
for lesson in lessons:
    print(f"  • {lesson}")
```

**Foundational Example Set (8 Examples)**

| Domain | Frequency | Seasonality | Difficulty | Top Model |
|--------|-----------|-------------|------------|-----------|
| Retail | Daily | Strong (weekly) | Easy | prophet_reg |
| Finance | Daily | None | Hard | arima_reg |
| Energy | Hourly | Strong (daily+weekly) | Medium | boost_tree |
| Manufacturing | Monthly | Moderate (annual) | Medium | arima_reg |
| IoT | Minute | Weak | Hard | arima_reg |
| Healthcare | Daily | Strong (flu season) | Easy | prophet_reg |
| Transportation | Hourly | Very Strong (commute) | Medium | boost_tree |
| Agriculture | Yearly | None | Hard | linear_reg |

**Key Benefits:**
- **Faster Learning**: See similar examples automatically, learn from past successes
- **Better Model Selection**: Evidence-based recommendations from proven cases
- **Knowledge Transfer**: Key lessons and preprocessing insights surface automatically
- **Confidence Boosting**: Models recommended by similar examples get up to +10% confidence
- **Retrieval Speed**: Sub-100ms with caching, scalable to 500+ examples
- **Extensible**: Easy to add new examples (JSON format), no code changes needed

### Phase 3.5: Autonomous Iteration (Self-Improving Workflows)

```python
from py_agent import ForecastAgent
import pandas as pd

# Load your data
train_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=500, freq='D'),
    'sales': [...],  # Daily sales with seasonality
    'temperature': [...],
    'promotion': [...]
})

test_data = pd.DataFrame({
    'date': pd.date_range('2021-05-15', periods=100, freq='D'),
    'sales': [...],
    'temperature': [...],
    'promotion': [...]
})

# Initialize agent
agent = ForecastAgent(verbose=True)

# Autonomous iteration: Try multiple approaches until target reached
best_workflow, history = agent.iterate(
    data=train_data,
    request="Forecast daily sales with seasonality",
    target_metric='rmse',     # Optimize RMSE
    target_value=10.0,        # Stop when RMSE < 10
    max_iterations=5,         # Try up to 5 different approaches
    test_data=test_data
)

# Agent automatically:
# 1. Generates initial workflow with default recommendation
# 2. Fits and evaluates on test data
# 3. If performance unsatisfactory, diagnoses issues
# 4. Tries different approaches (simpler model, regularization, more features, etc.)
# 5. Stops when target reached or max iterations exhausted

# Example iteration output:
# 🔄 Starting autonomous iteration loop...
#    Target: rmse < 10.0
#    Max iterations: 5
#
# ============================================================
# Iteration 1/5
# ============================================================
# 🎯 Approach: default_recommendation
#    Fitting workflow...
#    Evaluating on test data...
#    RMSE: 15.234
#    MAE: 12.456
#    R²: 0.752
# ⚠️  Issues detected: 1
#    • overfitting: Train RMSE much lower than test
#
# ============================================================
# Iteration 2/5
# ============================================================
# 🎯 Approach: regularization_or_simpler_model
#    Fitting workflow...
#    Evaluating on test data...
#    RMSE: 11.234
#    MAE: 9.123
#    R²: 0.821
# ✨ New best rmse: 11.234
#
# ============================================================
# Iteration 3/5
# ============================================================
# 🎯 Approach: try_tree_based_model
#    Fitting workflow...
#    Evaluating on test data...
#    RMSE: 9.456
#    MAE: 7.234
#    R²: 0.892
# ✨ New best rmse: 9.456
# ✅ Stopping: Target RMSE of 10.0 reached

# Access best workflow
print(f"Best RMSE: {best_workflow.extract_outputs()[2]['rmse'].iloc[0]:.2f}")

# Analyze iteration history
print(f"\nIterations: {len(history)}")
for i, result in enumerate(history, 1):
    print(f"{i}. {result.approach}: RMSE={result.performance.get('rmse', 'N/A'):.2f}, Success={result.success}")

# Use best workflow for predictions
predictions = best_workflow.predict(future_data)
```

**Iteration Approaches (Automatic Selection)**

The agent automatically tries different approaches based on detected issues:

1. **default_recommendation** (Iteration 1): Agent's best guess based on data analysis
2. **regularization_or_simpler_model**: If overfitting detected
3. **more_complex_model_or_features**: If underfitting detected
4. **try_tree_based_model**: Try gradient boosting or random forest
5. **try_time_series_model**: Try Prophet or ARIMA
6. **try_ensemble_or_boosting**: Try ensemble methods
7. **try_advanced_preprocessing**: Add polynomial features, interactions, PCA

**Stopping Criteria**

Iteration stops when:
- **Target reached**: Performance meets or exceeds target value
- **No improvement**: Less than 5% improvement over previous best
- **Max iterations**: Maximum number of iterations exhausted

**Key Benefits:**
- **Autonomous Improvement**: Agent tries multiple approaches automatically
- **Self-Debugging**: Detects overfitting, underfitting, and other issues
- **Adaptive Strategies**: Different approaches based on previous results
- **Performance Guarantees**: Stops when target performance reached
- **Time Efficient**: Avoids manual trial-and-error (hours → minutes)

## Architecture

### Core Components

```
py_agent/
├── tools/              # Analysis and recommendation functions
│   ├── data_analysis.py            # Temporal pattern detection
│   ├── model_selection.py          # Model recommendation engine
│   ├── recipe_generation.py        # Preprocessing recipe creation (Phase 3.2: 51 steps)
│   ├── multi_model_orchestration.py # Multi-model comparison (Phase 3.3)
│   ├── autonomous_iteration.py     # Autonomous iteration loops (Phase 3.5)
│   ├── workflow_execution.py       # Workflow fitting and evaluation
│   └── diagnostics.py              # Performance analysis
│
├── knowledge/          # RAG knowledge base (Phase 3.4)
│   ├── example_library.py          # ForecastingExample and ExampleLibrary
│   ├── rag_retrieval.py            # RAGRetriever with vector similarity
│   ├── forecasting_examples.json   # 8 foundational examples
│   └── __init__.py                 # Module exports
│
├── llm/                # LLM integration (Phase 2)
│   ├── client.py              # Anthropic SDK wrapper with budget management
│   └── tool_schemas.py        # Tool calling schemas for LLM
│
├── agents/             # Agent implementations
│   ├── forecast_agent.py      # Main ForecastAgent class (dual-mode + RAG)
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
- `create_recipe()` - Generate preprocessing code (Phase 3.2: Intelligent 51-step selection)
- `get_recipe_templates()` - Access 17 domain-specific recipe templates

#### Multi-Model Orchestration Tools (Phase 3.3)
- `generate_workflowset()` - Create WorkflowSet from model recommendations
- `compare_models_cv()` - Evaluate models with cross-validation
- `select_best_models()` - Select best models from rankings (best, within_1se, threshold)
- `recommend_ensemble()` - Suggest optimal ensemble composition

#### RAG Knowledge Base Tools (Phase 3.4)
- `ForecastingExample` - Dataclass for forecasting examples with metadata
- `ExampleLibrary` - Manage collection of forecasting examples (load, save, filter)
- `RAGRetriever` - Vector similarity search for example retrieval
  - `retrieve()` - Retrieve similar examples by text query
  - `retrieve_by_data_characteristics()` - Retrieve by data characteristics dict
  - `get_model_recommendations_from_examples()` - Extract model recommendations
  - `get_preprocessing_insights()` - Extract preprocessing strategies
  - `get_key_lessons()` - Extract key lessons learned
- `create_foundational_examples()` - Load 8 foundational forecasting examples

#### Autonomous Iteration Tools (Phase 3.5)
- `IterationLoop` - Autonomous try-evaluate-improve loop
  - `iterate()` - Iteratively improve workflow until target or max iterations
  - Automatic approach selection based on previous issues
  - Performance-based stopping criteria
- `IterationResult` - Result from single iteration attempt
  - Tracks performance, issues, approach, success/failure
  - Includes duration and error information
- `iterate_until_target()` - Convenience function for autonomous iteration

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
- ✅ **Phase 3.3 Complete**: Multi-Model WorkflowSet Orchestration!
  - Automatic comparison of 5+ models with cross-validation
  - Model ranking by performance (RMSE, MAE, R²)
  - Ensemble recommendations with diversity scoring
  - Selection strategies: best, within_1se, threshold
  - Time savings: 1-2 hours → 5 minutes
- ⏳ Phase 3.4: RAG knowledge base with 500+ forecasting examples
- ⏳ Phase 3.5: Autonomous iteration and self-improvement

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
