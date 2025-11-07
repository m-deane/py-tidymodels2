# py-tidymodels Architecture Documentation (arc42)

**Version:** 1.0
**Date:** 2025-10-26
**Status:** Planning Phase
**Authors:** Architecture Team

---

## Table of Contents

1. [Introduction and Goals](#1-introduction-and-goals)
2. [Constraints](#2-constraints)
3. [Context and Scope](#3-context-and-scope)
4. [Solution Strategy](#4-solution-strategy)
5. [Building Block View](#5-building-block-view)
6. [Runtime View](#6-runtime-view)
7. [Deployment View](#7-deployment-view)
8. [Cross-cutting Concepts](#8-cross-cutting-concepts)
9. [Architecture Decisions](#9-architecture-decisions)
10. [Quality Requirements](#10-quality-requirements)
11. [Risks and Technical Debt](#11-risks-and-technical-debt)
12. [Glossary](#12-glossary)

---

## 1. Introduction and Goals

### 1.1 Requirements Overview

**py-tidymodels** is a Python port of R's tidymodels ecosystem, focused on time series regression and forecasting. The system provides a unified, composable interface for:

- **Model Specification**: Consistent API across 50+ statistical and ML models
- **Preprocessing**: Feature engineering via recipes (pytimetk integration)
- **Workflows**: Composable pipelines of preprocessing + modeling
- **Cross-Validation**: Time series-aware resampling strategies
- **Multi-Model Comparison**: Workflow sets for comparing 100+ model configurations
- **Standardized Outputs**: Consistent DataFrame schemas for all models

**Target Users:**
- Data scientists working with time series data
- ML engineers building forecasting pipelines
- Researchers comparing modeling approaches
- Production teams deploying forecasting models

### 1.2 Quality Goals

| Priority | Quality Goal | Motivation |
|----------|-------------|------------|
| 1 | **Extensibility** | New models/engines must be trivial to add (<1 hour) |
| 2 | **Consistency** | Identical API across all model types (ML, statistical, time series) |
| 3 | **Performance** | <10% overhead vs direct sklearn/statsmodels usage |
| 4 | **Type Safety** | Full type hints for IDE support and early error detection |
| 5 | **Usability** | Clear error messages, comprehensive documentation |

### 1.3 Stakeholders

| Role | Expectations | Influence |
|------|--------------|-----------|
| **End Users** | Easy-to-use API, consistent behavior, good docs | High |
| **Contributors** | Clear extension points, well-tested code | Medium |
| **Python ML Community** | Integration with existing tools (sklearn, statsmodels) | Medium |
| **R tidymodels Users** | Familiar concepts, similar workflow | Low |

---

## 2. Constraints

### 2.1 Technical Constraints

| Constraint | Description | Motivation |
|------------|-------------|------------|
| **TC-1** | Python 3.10+ | Type hints, dataclass features |
| **TC-2** | Pandas DataFrames | Standard data structure in Python ML ecosystem |
| **TC-3** | Patsy for formulas | R-style formula parsing in Python |
| **TC-4** | pytimetk v2.2.0+ | Existing time series feature engineering (do not rebuild) |
| **TC-5** | skforecast | Recursive forecasting backend |
| **TC-6** | No R dependencies | Pure Python (optional rpy2 for ranger engine) |

### 2.2 Organizational Constraints

| Constraint | Description |
|------------|-------------|
| **OC-1** | Open source (MIT/Apache 2.0) |
| **OC-2** | Maintained on GitHub |
| **OC-3** | PyPI distribution |
| **OC-4** | Semantic versioning |

### 2.3 Conventions

| Convention | Description |
|------------|-------------|
| **CON-1** | PEP 8 style guide |
| **CON-2** | NumPy docstring format |
| **CON-3** | Type hints on all public APIs |
| **CON-4** | pytest for testing (>90% coverage) |
| **CON-5** | Black formatter, flake8 linter |

---

## 3. Context and Scope

### 3.1 Business Context

```
┌─────────────────────────────────────────────────────────────────┐
│                     py-tidymodels System                        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  py-recipes  │  │  py-parsnip  │  │ py-workflows │         │
│  │ (pytimetk)   │→ │  (engines)   │→ │ (pipelines)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         ↓                  ↓                  ↓                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ py-rsample   │  │ py-yardstick │  │py-workflowsets│        │
│  │    (CV)      │  │  (metrics)   │  │  (compare)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
         ↑                                           ↓
    ┌─────────┐                                 ┌─────────┐
    │   User  │                                 │ Results │
    │  (API)  │                                 │(3 DFs)  │
    └─────────┘                                 └─────────┘
         ↑
┌────────┴────────┐
│ External Systems│
├─────────────────┤
│ • sklearn       │
│ • statsmodels   │
│ • prophet       │
│ • pmdarima      │
│ • pytimetk      │
│ • skforecast    │
└─────────────────┘
```

**External Interfaces:**

| System | Purpose | Interface Type |
|--------|---------|----------------|
| **sklearn** | ML models (RF, XGBoost, etc.) | Python API (fit/predict) |
| **statsmodels** | Statistical models (ARIMA, OLS) | Python API |
| **prophet** | Time series forecasting | Python API |
| **pytimetk** | Time series feature engineering | Python API |
| **skforecast** | Recursive forecasting | Python API (ForecasterAutoreg) |
| **patsy** | Formula parsing | Python API |
| **pandas** | Data manipulation | DataFrame/Series |

### 3.2 Technical Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    py-tidymodels Technical Context              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Code                                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ workflow()                                                 │ │
│  │   .add_recipe(recipe(...).step_lag(...))                  │ │
│  │   .add_model(arima_reg(...).set_engine("statsmodels"))    │ │
│  │   .fit(train_data)                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          py-tidymodels Core Layers                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Layer 4: Workflows (composition)                        │   │
│  │ Layer 3: Recipes (prep/bake - user preprocessing)      │   │
│  │ Layer 2: Hardhat (mold/forge - internal preprocessing) │   │
│  │ Layer 1: Parsnip (model specs + engine registry)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Engine Layer (adapters)                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ SklearnLinearEngine → sklearn.linear_model.Ridge       │   │
│  │ StatsmodelsARIMAEngine → statsmodels.SARIMAX           │   │
│  │ ProphetEngine → prophet.Prophet                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          External ML Libraries                          │   │
│  │  (sklearn, statsmodels, prophet, etc.)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Solution Strategy

### 4.1 Core Architectural Decisions

#### Decision 1: Registry-Based Engine System
**Problem:** How to support 50+ models across multiple libraries with consistent API?

**Solution:** Centralized engine registry with decorator-based registration.

```python
@register_engine(model_type="linear_reg", engine="sklearn")
class SklearnLinearEngine(Engine):
    param_map = {"penalty": "alpha"}

    def fit(self, spec, data): ...
    def predict(self, fit, data, type): ...
    def extract_outputs(self, fit): ...
```

**Benefits:**
- ✅ Engines are independent, testable modules
- ✅ Easy to add new engines without modifying core
- ✅ Clear separation of concerns

**Trade-offs:**
- ❌ Global registry (addressed with proper namespacing)
- ❌ Decorator "magic" (documented thoroughly)

---

#### Decision 2: Two-Layer Preprocessing (recipes + hardhat)
**Problem:** Separate user-facing feature engineering from internal data preparation.

**Solution:** Two preprocessing layers with distinct responsibilities.

```
Raw Data
  → recipes (prep/bake) - Feature engineering, transformations
  → hardhat (mold/forge) - Formula → matrix, factor levels
  → Model fit/predict
```

**recipes (user-facing):**
- Feature engineering (lags, date features, normalization)
- Wraps pytimetk, sklearn transformers
- **Optional** - users can skip

**hardhat (internal):**
- Formula parsing → design matrix
- Factor level enforcement, column alignment
- **Always happens** - ensures data consistency

**Benefits:**
- ✅ Clean separation of concerns
- ✅ User controls preprocessing, system ensures consistency
- ✅ Prevents train/test leakage

---

#### Decision 3: Immutable Specs, Mutable Fits
**Problem:** Balance between functional programming principles and Python conventions.

**Solution:**
- `ModelSpec` is immutable (frozen dataclass)
- `ModelFit` is mutable (for caching)

```python
@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    engine: str
    args: Tuple[Tuple[str, Any], ...]

    def set_engine(self, engine) -> "ModelSpec":
        return replace(self, engine=engine)  # New instance

@dataclass
class ModelFit:
    spec: ModelSpec
    fit_data: Dict[str, Any]
    _cache: Dict = field(default_factory=dict)  # Mutable cache
```

**Benefits:**
- ✅ Specs are shareable, serializable
- ✅ Method chaining feels natural
- ✅ Fits can cache predictions for performance

---

#### Decision 4: Standardized Three-DataFrame Output
**Problem:** Different models return different output formats (sklearn: arrays, statsmodels: rich objects).

**Solution:** All models MUST return three standardized DataFrames.

```python
class Engine(ABC):
    @abstractmethod
    def extract_outputs(self, fit) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            model_outputs: actuals, fitted, predicted, residuals
            coefficients: parameters, hyperparameters, importances
            stats: metrics, residual tests, model info
        """
        pass
```

**Benefits:**
- ✅ Consistent interface for all models
- ✅ Multiple models stack without schema changes
- ✅ Easy to compare across model types

---

#### Decision 5: Workflows Over Model Tables
**Problem:** R's modeltime uses table-based model organization (modeltime_table/calibrate).

**Solution:** Use workflows + workflowsets instead.

❌ **Avoid:**
```r
# R modeltime pattern (DO NOT port)
models_tbl <- modeltime_table(m1, m2, m3)
calibrated <- modeltime_calibrate(models_tbl, test)
```

✅ **Use:**
```python
# Workflows + workflowsets pattern
wf_set = workflow_set(
    workflows=[wf1, wf2, wf3],
    ids=["ARIMA", "Prophet", "RF"]
)
results = wf_set.fit_resamples(cv_splits)
```

**Benefits:**
- ✅ More composable, less clunky
- ✅ Scales to 100+ model configurations
- ✅ Integrates with tune for hyperparameter optimization

---

### 4.2 Integration Strategy

**Leverage Existing Packages (Don't Rebuild):**

| Package | Purpose | Integration Strategy |
|---------|---------|---------------------|
| **pytimetk** | Time series feature engineering | Wrap in recipe steps (`step_lag()` → `augment_lags()`) |
| **skforecast** | Recursive forecasting | Use as backend for `recursive()` wrapper |
| **patsy** | Formula parsing | Use directly in hardhat layer |
| **sklearn** | ML models | Wrap as engines |
| **statsmodels** | Statistical models | Wrap as engines |

---

## 5. Building Block View

### 5.1 Level 1: System Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                      py-tidymodels System                         │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐           │
│  │ py-recipes  │──→│ py-hardhat  │──→│  py-parsnip  │           │
│  │ (Phase 2)   │   │  (Phase 1)  │   │  (Phase 1)   │           │
│  └─────────────┘   └─────────────┘   └──────────────┘           │
│         │                                      │                  │
│         └──────────────────┬───────────────────┘                  │
│                            ↓                                      │
│                   ┌─────────────────┐                            │
│                   │  py-workflows   │                            │
│                   │   (Phase 1)     │                            │
│                   └─────────────────┘                            │
│                            │                                      │
│         ┌──────────────────┼──────────────────┐                  │
│         ↓                  ↓                  ↓                  │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐           │
│  │ py-rsample  │   │ py-tune     │   │py-workflowsets│          │
│  │  (Phase 1)  │   │  (Phase 2)  │   │  (Phase 2)   │          │
│  └─────────────┘   └─────────────┘   └──────────────┘           │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ↓                                      │
│                   ┌─────────────────┐                            │
│                   │  py-yardstick   │                            │
│                   │   (Phase 2)     │                            │
│                   └─────────────────┘                            │
└───────────────────────────────────────────────────────────────────┘
```

**Package Dependencies:**
- **py-hardhat**: No dependencies (foundational)
- **py-parsnip**: Depends on py-hardhat
- **py-recipes**: Depends on pytimetk
- **py-workflows**: Depends on py-hardhat, py-parsnip, py-recipes
- **py-rsample**: Independent (time series splitting)
- **py-tune**: Depends on py-workflows, py-rsample
- **py-workflowsets**: Depends on py-workflows, py-rsample
- **py-yardstick**: Independent (metrics)

---

### 5.2 Level 2: Component Details

#### 5.2.1 py-hardhat (Data Preprocessing Abstraction)

```
┌─────────────────────────────────────────────────────────────┐
│                     py-hardhat                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                          │
│  │  Blueprint   │  (immutable preprocessing metadata)      │
│  ├──────────────┤                                          │
│  │ • formula    │                                          │
│  │ • roles      │  {outcome: [...], predictor: [...]}     │
│  │ • levels     │  {col: [cat1, cat2, ...]}               │
│  │ • ptypes     │  {col: dtype}                           │
│  └──────────────┘                                          │
│         ↑                                                   │
│         │                                                   │
│  ┌──────┴───────┐        ┌──────────────┐                 │
│  │   mold()     │───────→│  MoldedData  │                 │
│  ├──────────────┤        ├──────────────┤                 │
│  │ Input:       │        │ • predictors │ (X matrix)      │
│  │ • formula    │        │ • outcomes   │ (y)             │
│  │ • data       │        │ • extras     │ (weights, etc)  │
│  │              │        │ • blueprint  │                 │
│  │ Process:     │        └──────────────┘                 │
│  │ 1. Parse     │                                          │
│  │ 2. Extract   │                                          │
│  │ 3. Create    │                                          │
│  │    matrices  │                                          │
│  └──────────────┘                                          │
│                                                             │
│  ┌──────────────┐                                          │
│  │   forge()    │  (apply blueprint to new data)          │
│  ├──────────────┤                                          │
│  │ Input:       │                                          │
│  │ • new_data   │                                          │
│  │ • blueprint  │                                          │
│  │              │                                          │
│  │ Process:     │                                          │
│  │ 1. Apply     │                                          │
│  │    formula   │                                          │
│  │ 2. Enforce   │                                          │
│  │    levels    │                                          │
│  │ 3. Align     │                                          │
│  │    columns   │                                          │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Convert formulas to design matrices
- Manage factor levels (categorical handling)
- Ensure column consistency across train/test
- Extract data roles (outcome, predictor, group, time_index)

**Key Classes:**
- `Blueprint`: Immutable preprocessing recipe
- `MoldedData`: Preprocessed data ready for modeling
- `mold()`: Fit-time preprocessing
- `forge()`: Predict-time preprocessing

---

#### 5.2.2 py-parsnip (Unified Model Interface)

```
┌──────────────────────────────────────────────────────────────────┐
│                         py-parsnip                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Engine Registry                         │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  ENGINE_REGISTRY = {                                       │ │
│  │    ("linear_reg", "sklearn"): SklearnLinearEngine,        │ │
│  │    ("linear_reg", "statsmodels"): StatsmodelsLinearEngine,│ │
│  │    ("arima_reg", "statsmodels"): StatsmodelsARIMAEngine,  │ │
│  │    ("prophet_reg", "prophet"): ProphetEngine,             │ │
│  │    ...                                                     │ │
│  │  }                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↑                                     │
│                            │                                     │
│  ┌─────────────────────────┴──────────────────────────────────┐ │
│  │              Model Specification Layer                     │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  @dataclass(frozen=True)                                   │ │
│  │  class ModelSpec:                                          │ │
│  │    model_type: str                                         │ │
│  │    mode: str                                               │ │
│  │    engine: str | None                                      │ │
│  │    args: Tuple[Tuple[str, Any], ...]                      │ │
│  │                                                            │ │
│  │    def set_engine(self, engine) -> ModelSpec               │ │
│  │    def set_mode(self, mode) -> ModelSpec                   │ │
│  │    def fit(self, formula, data) -> ModelFit                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Model Fit Layer                               │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  @dataclass                                                │ │
│  │  class ModelFit:                                           │ │
│  │    spec: ModelSpec                                         │ │
│  │    blueprint: Blueprint                                    │ │
│  │    fit_data: Dict[str, Any]  # Engine-specific            │ │
│  │    fit_time: datetime                                      │ │
│  │                                                            │ │
│  │    def predict(self, new_data, type) -> pd.DataFrame      │ │
│  │    def extract_outputs(self) -> (DFs...)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ↓                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Engine Interface                           │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  class Engine(ABC):                                        │ │
│  │    param_map: Dict[str, str] = {}                         │ │
│  │                                                            │ │
│  │    @abstractmethod                                         │ │
│  │    def fit(spec, molded) -> Dict[str, Any]                │ │
│  │                                                            │ │
│  │    @abstractmethod                                         │ │
│  │    def predict(fit, molded, type) -> pd.DataFrame         │ │
│  │                                                            │ │
│  │    @abstractmethod                                         │ │
│  │    def extract_outputs(fit) -> (3 DataFrames)             │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Model specification with consistent API
- Engine registration and parameter translation
- Delegating fit/predict to engines
- Extracting standardized outputs

**Key Classes:**
- `ModelSpec`: Immutable specification
- `ModelFit`: Fitted model artifact
- `Engine`: Abstract base for all engines
- `register_engine()`: Decorator for registration

---

#### 5.2.3 py-workflows (Composition Layer)

```
┌─────────────────────────────────────────────────────────────┐
│                      py-workflows                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  @dataclass(frozen=True)                             │  │
│  │  class Workflow:                                     │  │
│  │    preprocessor: Recipe | str | None                │  │
│  │    spec: ModelSpec | None                           │  │
│  │    post: Any | None                                 │  │
│  │                                                      │  │
│  │  Methods:                                            │  │
│  │  • add_recipe(recipe) -> Workflow                   │  │
│  │  • add_model(spec) -> Workflow                      │  │
│  │  • add_formula(formula) -> Workflow                 │  │
│  │  • fit(data) -> WorkflowFit                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Fit Process                         │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  1. Apply recipe (if present)                       │  │
│  │     recipe_fit = recipe.prep(data)                  │  │
│  │     data_prepped = recipe_fit.bake(data)            │  │
│  │                                                      │  │
│  │  2. Fit model                                        │  │
│  │     model_fit = spec.fit(formula, data_prepped)     │  │
│  │                                                      │  │
│  │  3. Apply postprocessor (future)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  @dataclass                                          │  │
│  │  class WorkflowFit:                                  │  │
│  │    workflow: Workflow                                │  │
│  │    pre: Recipe | str | None                         │  │
│  │    fit: ModelFit                                     │  │
│  │    post: Any | None                                  │  │
│  │                                                      │  │
│  │  Methods:                                            │  │
│  │  • predict(new_data, type) -> pd.DataFrame          │  │
│  │  • extract_fit_parsnip() -> ModelFit                │  │
│  │  • extract_preprocessor() -> Recipe                 │  │
│  │  • extract_outputs() -> (3 DataFrames)              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Predict Process                         │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  1. Apply recipe (if present)                       │  │
│  │     new_data_prepped = recipe_fit.bake(new_data)    │  │
│  │                                                      │  │
│  │  2. Model prediction                                 │  │
│  │     predictions = model_fit.predict(               │  │
│  │         new_data_prepped, type                      │  │
│  │     )                                                │  │
│  │                                                      │  │
│  │  3. Postprocessing (future)                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Responsibilities:**
- Compose recipe + model into pipeline
- Ensure preprocessing applied at predict time
- Delegate to components
- Provide consistent interface

**Key Classes:**
- `Workflow`: Immutable composition
- `WorkflowFit`: Fitted pipeline
- `workflow()`: Factory function

---

## 6. Runtime View

### 6.1 Scenario 1: Basic Model Fitting

```
User                 Workflow            ModelSpec           Engine              sklearn
 │                       │                    │                 │                   │
 │ workflow()            │                    │                 │                   │
 │──────────────────────→│                    │                 │                   │
 │                       │                    │                 │                   │
 │ .add_model(           │                    │                 │                   │
 │   linear_reg()        │                    │                 │                   │
 │     .set_engine(      │                    │                 │                   │
 │       "sklearn"       │                    │                 │                   │
 │     )                 │                    │                 │                   │
 │ )                     │                    │                 │                   │
 │──────────────────────→│                    │                 │                   │
 │                       │                    │                 │                   │
 │ .fit(train_data)      │                    │                 │                   │
 │──────────────────────→│                    │                 │                   │
 │                       │                    │                 │                   │
 │                       │ spec.fit()         │                 │                   │
 │                       │───────────────────→│                 │                   │
 │                       │                    │                 │                   │
 │                       │                    │ mold(formula, data)                 │
 │                       │                    │──────────────────────────────────┐  │
 │                       │                    │←─────────────────────────────────┘  │
 │                       │                    │ MoldedData + Blueprint              │
 │                       │                    │                 │                   │
 │                       │                    │ get engine      │                   │
 │                       │                    │────────────────→│                   │
 │                       │                    │                 │                   │
 │                       │                    │                 │ engine.fit()      │
 │                       │                    │                 │──────────────────→│
 │                       │                    │                 │                   │
 │                       │                    │                 │                   │ Ridge().fit()
 │                       │                    │                 │                   │──────────────┐
 │                       │                    │                 │                   │←─────────────┘
 │                       │                    │                 │                   │
 │                       │                    │                 │ fitted model      │
 │                       │                    │                 │←──────────────────│
 │                       │                    │                 │                   │
 │                       │                    │ ModelFit        │                   │
 │                       │                    │←────────────────│                   │
 │                       │ WorkflowFit        │                 │                   │
 │                       │←───────────────────│                 │                   │
 │ WorkflowFit           │                    │                 │                   │
 │←──────────────────────│                    │                 │                   │
```

**Steps:**
1. User creates workflow
2. User adds model specification with engine
3. User calls fit()
4. Workflow delegates to ModelSpec.fit()
5. ModelSpec uses hardhat.mold() to preprocess
6. ModelSpec gets engine from registry
7. Engine translates parameters and calls sklearn
8. sklearn fits Ridge model
9. Engine wraps result in fit_data dict
10. ModelSpec creates ModelFit
11. Workflow creates WorkflowFit
12. Returns to user

---

### 6.2 Scenario 2: Prediction with Workflow

```
User           WorkflowFit        ModelFit         Engine          sklearn
 │                   │                │               │               │
 │ .predict(test)    │                │               │               │
 │──────────────────→│                │               │               │
 │                   │                │               │               │
 │                   │ (no recipe,    │               │               │
 │                   │  so skip)      │               │               │
 │                   │                │               │               │
 │                   │ model_fit      │               │               │
 │                   │  .predict()    │               │               │
 │                   │───────────────→│               │               │
 │                   │                │               │               │
 │                   │                │ forge(        │               │
 │                   │                │   new_data,   │               │
 │                   │                │   blueprint   │               │
 │                   │                │ )             │               │
 │                   │                │──────────────────────────┐    │
 │                   │                │←─────────────────────────┘    │
 │                   │                │ MoldedData                    │
 │                   │                │               │               │
 │                   │                │ get engine    │               │
 │                   │                │──────────────→│               │
 │                   │                │               │               │
 │                   │                │               │ engine        │
 │                   │                │               │  .predict()   │
 │                   │                │               │──────────────→│
 │                   │                │               │               │
 │                   │                │               │               │ model.predict()
 │                   │                │               │               │────────────────┐
 │                   │                │               │               │←───────────────┘
 │                   │                │               │               │
 │                   │                │               │ predictions   │
 │                   │                │               │←──────────────│
 │                   │                │ DataFrame     │               │
 │                   │                │←──────────────│               │
 │                   │ DataFrame      │               │               │
 │                   │←───────────────│               │               │
 │ DataFrame         │                │               │               │
 │←──────────────────│                │               │               │
```

**Steps:**
1. User calls predict() on WorkflowFit
2. WorkflowFit checks for recipe (none in this case)
3. WorkflowFit delegates to ModelFit.predict()
4. ModelFit uses hardhat.forge() to preprocess
5. ModelFit gets engine from registry
6. Engine calls sklearn's predict()
7. sklearn returns numpy array
8. Engine wraps in DataFrame
9. Returns through layers to user

---

### 6.3 Scenario 3: Time Series CV with WorkflowSet

```
User              WorkflowSet         RSample        Workflow         Models
 │                      │                 │              │               │
 │ workflow_set(        │                 │              │               │
 │   workflows=[        │                 │              │               │
 │     wf1, wf2, wf3    │                 │              │               │
 │   ]                  │                 │              │               │
 │ )                    │                 │              │               │
 │─────────────────────→│                 │              │               │
 │                      │                 │              │               │
 │ .fit_resamples(      │                 │              │               │
 │   cv_splits          │                 │              │               │
 │ )                    │                 │              │               │
 │─────────────────────→│                 │              │               │
 │                      │                 │              │               │
 │                      │ for split in cv_splits:        │               │
 │                      │────────────────→│              │               │
 │                      │                 │              │               │
 │                      │                 │ split        │               │
 │                      │                 │  .training() │               │
 │                      │                 │──────────┐   │               │
 │                      │                 │←─────────┘   │               │
 │                      │                 │              │               │
 │                      │                 │ split        │               │
 │                      │                 │  .testing()  │               │
 │                      │                 │──────────┐   │               │
 │                      │                 │←─────────┘   │               │
 │                      │                 │              │               │
 │                      │ for wf in workflows:           │               │
 │                      │────────────────────────────────→│               │
 │                      │                 │              │               │
 │                      │                 │              │ wf.fit(train) │
 │                      │                 │              │──────────────→│
 │                      │                 │              │               │
 │                      │                 │              │               │ (fit process)
 │                      │                 │              │               │───────────┐
 │                      │                 │              │               │←──────────┘
 │                      │                 │              │               │
 │                      │                 │              │ wf_fit        │
 │                      │                 │              │←──────────────│
 │                      │                 │              │               │
 │                      │                 │              │ wf_fit        │
 │                      │                 │              │  .predict()   │
 │                      │                 │              │──────────────→│
 │                      │                 │              │               │
 │                      │                 │              │               │ (predict)
 │                      │                 │              │               │───────────┐
 │                      │                 │              │               │←──────────┘
 │                      │                 │              │               │
 │                      │                 │              │ predictions   │
 │                      │                 │              │←──────────────│
 │                      │                 │              │               │
 │                      │ collect results│              │               │
 │                      │←───────────────────────────────│               │
 │                      │                 │              │               │
 │ WorkflowSetResults   │                 │              │               │
 │←─────────────────────│                 │              │               │
```

**Steps:**
1. User creates WorkflowSet with multiple workflows
2. User calls fit_resamples() with CV splits
3. WorkflowSet iterates over each CV split
4. For each split, get training and testing data
5. For each workflow, fit on training data
6. Predict on testing data
7. Collect all results (predictions, metrics)
8. Return WorkflowSetResults

---

## 7. Deployment View

### 7.1 Installation

```
┌────────────────────────────────────────────────────────────┐
│                   User's Environment                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Create virtual environment                             │
│     python -m venv py-tidymodels2                         │
│                                                            │
│  2. Install py-tidymodels from PyPI                       │
│     pip install py-tidymodels                             │
│                                                            │
│  3. Dependencies automatically installed:                  │
│     ┌──────────────────────────────────────────────┐      │
│     │ pandas, numpy, scipy                         │      │
│     │ scikit-learn, statsmodels, prophet          │      │
│     │ pytimetk, skforecast                        │      │
│     │ plotly, patsy                                │      │
│     └──────────────────────────────────────────────┘      │
│                                                            │
│  4. Optional GPU support:                                  │
│     pip install py-tidymodels[gpu]                        │
│     (installs cudf, xgboost[gpu])                         │
│                                                            │
│  5. Development install:                                   │
│     pip install py-tidymodels[dev]                        │
│     (installs pytest, black, mypy, etc.)                  │
└────────────────────────────────────────────────────────────┘
```

### 7.2 Package Structure

```
py-tidymodels/
├── py_hardhat/
│   ├── __init__.py
│   ├── blueprint.py
│   ├── mold.py
│   └── forge.py
├── py_parsnip/
│   ├── __init__.py
│   ├── model_spec.py
│   ├── model_fit.py
│   ├── engine.py
│   ├── registry.py
│   ├── models/
│   │   ├── linear_reg.py
│   │   ├── rand_forest.py
│   │   ├── arima_reg.py
│   │   └── prophet_reg.py
│   └── engines/
│       ├── sklearn_linear.py
│       ├── statsmodels_linear.py
│       ├── statsmodels_arima.py
│       └── prophet.py
├── py_recipes/
│   ├── __init__.py
│   ├── recipe.py
│   ├── step.py
│   └── steps/
│       ├── step_lag.py          # pytimetk wrapper
│       ├── step_date.py         # pytimetk wrapper
│       ├── step_normalize.py    # sklearn wrapper
│       └── ...
├── py_workflows/
│   ├── __init__.py
│   ├── workflow.py
│   └── workflow_fit.py
├── py_rsample/
│   ├── __init__.py
│   ├── split.py
│   ├── time_series_cv.py
│   └── periods.py
├── py_tune/
├── py_yardstick/
└── py_workflowsets/
```

---

## 8. Cross-cutting Concepts

### 8.1 Immutability Pattern

**Principle:** Specifications are immutable, fits are mutable.

```python
# Immutable spec
spec = linear_reg(penalty=0.1)
new_spec = spec.set_engine("sklearn")  # Returns NEW instance
assert spec != new_spec  # Different objects

# Mutable fit (for caching)
fit = spec.fit(data)
fit._cache["predictions"] = preds  # Allowed
```

**Benefits:**
- Thread-safe specifications
- Cacheable, shareable
- Easier to reason about

---

### 8.2 Parameter Translation

**Principle:** Unified tidymodels parameters translate to engine-specific names.

```python
# User code (engine-agnostic)
model = rand_forest(trees=500, mtry=10)

# Engine translation
class SklearnRandForestEngine(Engine):
    param_map = {
        "trees": "n_estimators",
        "mtry": "max_features"
    }

    def fit(self, spec, data):
        args = self.translate_params(dict(spec.args))
        # args = {"n_estimators": 500, "max_features": 10}
        model = RandomForestRegressor(**args)
```

**Benefits:**
- Consistent parameter names across engines
- Easy to switch engines
- Less confusion for users

---

### 8.3 Type Safety

**Principle:** All public APIs have type hints.

```python
def mold(
    formula: str,
    data: pd.DataFrame
) -> Tuple[MoldedData, Blueprint]:
    ...

@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    mode: str = "unknown"
    engine: str | None = None

    def fit(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        x: pd.DataFrame | None = None,
        y: pd.Series | None = None
    ) -> "ModelFit":
        ...
```

**Benefits:**
- IDE autocomplete
- Early error detection
- Self-documenting code

---

### 8.4 Error Handling

**Principle:** Clear, actionable error messages.

```python
# Bad
AttributeError: 'NoneType' object has no attribute 'fit'

# Good
ValueError:
  Model 'linear_reg' requires an engine.

  Available engines for linear_reg:
  • 'sklearn' - Ridge regression via scikit-learn
  • 'statsmodels' - OLS via statsmodels

  Use: linear_reg().set_engine("sklearn")
```

**Standards:**
- Always validate at entry points
- Suggest fixes in error messages
- Include available options when applicable

---

## 9. Architecture Decisions

### ADR-001: Registry-Based Engine System

**Status:** Accepted
**Date:** 2025-10-26
**Context:** Need to support 50+ models across multiple libraries with consistent API.

**Decision:** Use decorator-based engine registry.

**Consequences:**
- ✅ Easy to add new engines
- ✅ Engines are independent modules
- ❌ Global state (registry)
- ❌ Magic decorator pattern

**Alternatives Considered:**
- Class hierarchy with adapters (rejected: deep inheritance, hard to extend)
- Functional composition (rejected: unfamiliar to ML practitioners)

---

### ADR-002: Two-Layer Preprocessing

**Status:** Accepted
**Date:** 2025-10-26
**Context:** Need to separate user-facing feature engineering from internal data preparation.

**Decision:** Implement both recipes (prep/bake) and hardhat (mold/forge).

**Consequences:**
- ✅ Clean separation of concerns
- ✅ Prevents train/test leakage
- ✅ User controls preprocessing
- ❌ Two layers to understand
- ❌ More complexity

**Alternatives Considered:**
- Single preprocessing layer (rejected: mixing concerns)
- sklearn pipelines (rejected: doesn't match tidymodels philosophy)

---

### ADR-003: Avoid modeltime_table Pattern

**Status:** Accepted
**Date:** 2025-10-26
**Context:** R's modeltime uses table-based model organization which is clunky and doesn't scale.

**Decision:** Use workflows + workflowsets instead of modeltime_table/calibrate.

**Consequences:**
- ✅ More composable
- ✅ Scales to 100+ models
- ✅ Integrates with tune
- ❌ Different from R (but better)

**Alternatives Considered:**
- Port modeltime_table exactly (rejected: user feedback says it's clunky)
- Custom model container (rejected: reinventing workflows)

---

### ADR-004: Leverage pytimetk Instead of Rebuilding

**Status:** Accepted
**Date:** 2025-10-26
**Context:** Time series feature engineering is complex. pytimetk already exists (v2.2.0).

**Decision:** Wrap pytimetk functions in recipe steps instead of rebuilding.

**Consequences:**
- ✅ Saves 2-3 months development time
- ✅ Production-ready code (66 tests)
- ✅ GPU acceleration available
- ❌ External dependency
- ❌ Less control over implementation

**Alternatives Considered:**
- Build py-timetk from scratch (rejected: too much effort, pytimetk exists)
- Use tsfresh (rejected: different API philosophy)

---

### ADR-005: Standardized Three-DataFrame Output

**Status:** Accepted
**Date:** 2025-10-26
**Context:** Different models return different formats. Need consistency for comparison.

**Decision:** All models MUST return (model_outputs, coefficients, stats) DataFrames.

**Consequences:**
- ✅ Consistent interface
- ✅ Multiple models stack without schema changes
- ✅ Easy to compare across model types
- ❌ Overhead for simple models
- ❌ Schema enforcement complexity

**Alternatives Considered:**
- Different outputs per model type (rejected: inconsistent)
- Single DataFrame (rejected: loses structure)

---

## 10. Quality Requirements

### 10.1 Performance

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Overhead** | <10% vs direct sklearn/statsmodels | Benchmark suite |
| **Prediction Speed** | <1ms for 1000 rows | Profiling |
| **Memory** | <2x of underlying library | Memory profiler |
| **mold/forge Cache** | 90%+ cache hit rate | Cache metrics |

### 10.2 Reliability

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Test Coverage** | >90% line coverage | pytest-cov |
| **Type Coverage** | 100% of public API | mypy |
| **Error Handling** | 100% of user-facing functions | Code review |
| **Regression Tests** | All past bugs have tests | Test suite |

### 10.3 Usability

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **API Consistency** | 100% consistent across models | API review |
| **Documentation** | All public functions | Sphinx docs |
| **Examples** | 2+ per function | Doc review |
| **Error Messages** | Actionable suggestions | User testing |

### 10.4 Maintainability

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| **Code Style** | PEP 8 compliant | flake8 |
| **Formatting** | Black formatted | Black check |
| **Complexity** | Cyclomatic complexity <10 | Radon |
| **Modularity** | Single responsibility per class | Code review |

---

## 11. Risks and Technical Debt

### 11.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Performance overhead** | Medium | High | Profile early, optimize hot paths, allow direct access |
| **Schema violations** | Low | Medium | Runtime validation, OutputBuilder helpers |
| **Engine translation bugs** | Medium | Medium | Extensive tests, documentation, fallback to raw params |
| **Patsy complexity** | Low | Medium | Support DataFrame input, cache parsed formulas |

### 11.2 Known Technical Debt

| Debt Item | Priority | Plan |
|-----------|----------|------|
| **Global engine registry** | Medium | Consider ContextVars for isolation |
| **No async support** | Low | Defer until user demand |
| **Limited GPU support** | Medium | Expand in Phase 4 |
| **No streaming data** | Low | Defer until use case emerges |

### 11.3 Future Enhancements

| Enhancement | Phase | Priority |
|-------------|-------|----------|
| **Distributed computing (Dask/Ray)** | 4+ | Medium |
| **Model explainability (SHAP)** | 3-4 | High |
| **AutoML integration** | 4+ | Low |
| **Production deployment tools** | 4 | High |

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **Engine** | Backend implementation (sklearn, statsmodels, etc.) |
| **Model Spec** | Immutable model specification (type, engine, params) |
| **Model Fit** | Fitted model artifact |
| **Workflow** | Composition of recipe + model + postprocessor |
| **Recipe** | Feature engineering specification (prep/bake) |
| **Blueprint** | Hardhat preprocessing metadata (mold/forge) |
| **mold()** | Convert formula + data to model matrix (fit time) |
| **forge()** | Apply blueprint to new data (predict time) |
| **prep()** | Fit recipe to training data |
| **bake()** | Apply fitted recipe to data |
| **WorkflowSet** | Collection of workflows for comparison |
| **Resamples** | Cross-validation splits |
| **model_outputs** | DataFrame with actuals, fitted, predicted, residuals |
| **coefficients** | DataFrame with parameters, hyperparameters, importances |
| **stats** | DataFrame with metrics, model info |

---

**End of Architecture Documentation**
