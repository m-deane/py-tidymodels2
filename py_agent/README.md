# py_agent: AI-Powered Forecasting Agent

AI agent system for automated time series forecasting workflow generation using py-tidymodels.

## Overview

`py_agent` is an intelligent agent that can automatically:
- Analyze temporal patterns in your data
- Recommend appropriate forecasting models
- Generate preprocessing recipes
- Create complete py-tidymodels workflows
- Diagnose performance issues and suggest improvements

## MVP Features (v0.1.0)

This is the initial MVP implementation with:
- ✅ Rule-based workflow generation
- ✅ Support for 3 model types: `linear_reg`, `prophet_reg`, `rand_forest`
- ✅ Automated data analysis (seasonality, trend, autocorrelation)
- ✅ Basic recipe generation (10+ preprocessing steps)
- ✅ Performance diagnostics and debugging
- ✅ Conversational session interface
- 🎯 Target: 70%+ workflow success rate

## Quick Start

```python
from py_agent import ForecastAgent
import pandas as pd

# Load your data
sales_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=365, freq='D'),
    'sales': [...],  # Your sales values
    'store_id': [...]  # Optional: for grouped forecasting
})

# Initialize agent
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

## Architecture

### Core Components

```
py_agent/
├── tools/              # Analysis and recommendation functions
│   ├── data_analysis.py       # Temporal pattern detection
│   ├── model_selection.py     # Model recommendation engine
│   ├── recipe_generation.py   # Preprocessing recipe creation
│   ├── workflow_execution.py  # Workflow fitting and evaluation
│   └── diagnostics.py         # Performance analysis
│
├── agents/             # Agent implementations
│   └── forecast_agent.py      # Main ForecastAgent class
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

### Supported Models (MVP)

1. **`linear_reg`** - Linear Regression
   - Best for: Linear trends, high interpretability
   - Interpretability: HIGH
   - Speed: VERY FAST

2. **`prophet_reg`** - Facebook Prophet
   - Best for: Strong seasonality, holidays, missing data
   - Interpretability: MEDIUM
   - Speed: FAST

3. **`rand_forest`** - Random Forest
   - Best for: Complex patterns, feature interactions
   - Interpretability: MEDIUM
   - Speed: MODERATE

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

Automatically generates preprocessing recipes based on:
- Data quality (missing values, outliers)
- Model requirements
- Domain knowledge (retail, finance, energy)

Example generated recipe:
```python
from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_nominal

rec = (recipe(data, formula)
    .step_impute_median(all_numeric())
    .step_date('date', features=['dow', 'month', 'quarter'])
    .step_normalize(all_numeric())
    .step_dummy(all_nominal()))
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

### Phase 1: MVP (Current)
- ✅ Rule-based workflow generation
- ✅ 3 model types supported
- ✅ Basic recipe generation
- ✅ Performance diagnostics

### Phase 2: LLM Integration (Planned)
- ⏳ Claude Sonnet 4.5 integration
- ⏳ LLM-based reasoning for model selection
- ⏳ Advanced recipe optimization
- ⏳ Natural language explanations

### Phase 3: Multi-Agent System (Planned)
- ⏳ Specialized agents (DataAnalyzer, FeatureEngineer, ModelSelector)
- ⏳ RAG knowledge base
- ⏳ Ensemble recommendations
- ⏳ Autonomous iteration

## Performance Targets

Based on research report (`.claude_plans/AI_AGENT_RESEARCH_REPORT.md`):

- **Success Rate**: 70%+ of generated workflows execute successfully
- **Accuracy**: 60%+ produce results within 20% of expert baseline
- **Time Savings**: 80% reduction (4 hours → 45 minutes for beginners)
- **Cost**: <$10 per forecast (when LLM integrated)

## Dependencies

- pandas >= 2.0
- numpy >= 1.24
- statsmodels >= 0.14
- scipy >= 1.10
- py_tidymodels (all packages: hardhat, parsnip, recipes, workflows, etc.)

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
