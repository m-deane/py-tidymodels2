# Model Gap Analysis: R Tidymodels vs. py-tidymodels

**Analysis Date:** 2025-10-27
**py-tidymodels Version:** Phase 3 Complete (657+ tests)

## Executive Summary

R's tidymodels/modeltime ecosystem contains **43 unique model types** across 11 categories. py-tidymodels currently implements **5 model types**, leaving **38 models** available for future implementation.

**Current Coverage:** 11.6% (5/43 models)

---

## Currently Implemented in py-tidymodels ✅

| Model | Category | Engines | Status |
|-------|----------|---------|--------|
| `linear_reg` | Linear Models | sklearn, statsmodels | ✅ Fully implemented |
| `prophet_reg` | Time Series | prophet | ✅ Fully implemented |
| `arima_reg` | Time Series | statsmodels | ✅ Fully implemented |
| `rand_forest` | Tree Ensemble | sklearn | ✅ Fully implemented |
| `recursive_reg` | Time Series | skforecast | ✅ Fully implemented (custom) |

**Total Implemented:** 5 models

---

## Priority 1: High-Impact Models (Recommended Next) 🎯

These models are commonly used, have excellent Python libraries available, and would significantly expand py-tidymodels' capabilities.

### A. Tree-Based Models (3 models)

| Model | Python Engines | Priority | Rationale |
|-------|---------------|----------|-----------|
| `boost_tree` | xgboost, lightgbm, catboost | **CRITICAL** | Industry-standard ensemble method, excellent libraries |
| `decision_tree` | sklearn.tree | High | Fundamental ML algorithm, simple implementation |
| `bag_tree` | sklearn.ensemble (BaggingClassifier/Regressor) | Medium | Useful ensemble baseline |

**Implementation Effort:** Medium
**Python Library Support:** Excellent
**User Demand:** Very High

### B. Linear Classification Models (2 models)

| Model | Python Engines | Priority | Rationale |
|-------|---------------|----------|-----------|
| `logistic_reg` | sklearn.linear_model, statsmodels | **CRITICAL** | Essential for classification, mirrors linear_reg |
| `multinom_reg` | sklearn.linear_model | High | Multi-class classification, natural extension |

**Implementation Effort:** Low (similar to linear_reg)
**Python Library Support:** Excellent
**User Demand:** Very High

### C. Support Vector Machines (3 models)

| Model | Python Engines | Priority | Rationale |
|-------|---------------|----------|-----------|
| `svm_rbf` | sklearn.svm | High | Most popular SVM kernel |
| `svm_linear` | sklearn.svm | Medium | Fast for high-dimensional data |
| `svm_poly` | sklearn.svm | Low | Less commonly used |

**Implementation Effort:** Low (sklearn provides all)
**Python Library Support:** Excellent
**User Demand:** High

### D. Additional Time Series Models (3 models)

| Model | Python Engines | Priority | Rationale |
|-------|---------------|----------|-----------|
| `exp_smoothing` | statsmodels.holtwinters | High | Classic forecasting method |
| `naive_reg` | Custom (simple implementation) | Medium | Essential baseline for benchmarking |
| `nnetar_reg` | darts, pytorch_forecasting | Low | Neural network forecasting |

**Implementation Effort:** Low to Medium
**Python Library Support:** Good
**User Demand:** Medium

### E. Other Essential Models (4 models)

| Model | Python Engines | Priority | Rationale |
|-------|---------------|----------|-----------|
| `naive_Bayes` | sklearn.naive_bayes | High | Fast, interpretable classification |
| `nearest_neighbor` | sklearn.neighbors | High | Simple, powerful baseline |
| `mlp` | sklearn.neural_network, pytorch | Medium | Neural networks for tabular data |
| `poisson_reg` | statsmodels.genmod | Medium | Count data regression |

**Implementation Effort:** Low to Medium
**Python Library Support:** Excellent
**User Demand:** High

---

## Priority 2: Valuable But Specialized Models 📊

### A. Spline & Adaptive Models (3 models)

| Model | Python Engines | Complexity | Notes |
|-------|---------------|------------|-------|
| `mars` | py-earth | Medium | Multivariate Adaptive Regression Splines |
| `gen_additive_mod` | pygam | Medium | Non-parametric smoothing |
| `bag_mars` | py-earth | Medium | Ensemble MARS |

**Implementation Effort:** Medium
**Python Library Support:** Good (py-earth, pygam available)

### B. Discriminant Analysis (4 models)

| Model | Python Engines | Complexity | Notes |
|-------|---------------|------------|-------|
| `discrim_linear` | sklearn.discriminant_analysis | Low | Linear Discriminant Analysis |
| `discrim_quad` | sklearn.discriminant_analysis | Low | Quadratic Discriminant Analysis |
| `discrim_flexible` | Custom + sklearn | Medium | Requires additional implementation |
| `discrim_regularized` | Custom | Medium | Regularized discriminant analysis |

**Implementation Effort:** Low to Medium
**Python Library Support:** Good (sklearn provides LDA/QDA)

### C. Additional Time Series (4 models)

| Model | Python Engines | Complexity | Notes |
|-------|---------------|------------|-------|
| `seasonal_reg` | statsmodels.tsa.seasonal | Medium | TBATS, STL decomposition |
| `arima_boost` | statsmodels + xgboost | Medium | Hybrid model |
| `prophet_boost` | prophet + xgboost | Medium | Hybrid model |
| `adam` | Custom (port from R smooth package) | High | Advanced exponential smoothing |

**Implementation Effort:** Medium to High
**Python Library Support:** Mixed (some require custom implementation)

### D. Other Models (2 models)

| Model | Python Engines | Complexity | Notes |
|-------|---------------|------------|-------|
| `pls` | sklearn.cross_decomposition | Low | Partial Least Squares |
| `null_model` | Custom (simple) | Very Low | Baseline predictor |

**Implementation Effort:** Low
**Python Library Support:** Good

---

## Priority 3: Advanced/Specialized Models 🔬

These models are valuable but require significant implementation effort or have limited Python library support.

### A. Rule-Based Models (3 models)

| Model | Python Availability | Challenge |
|-------|-------------------|-----------|
| `cubist_rules` | No direct Python equivalent | Would require porting Cubist from R/C |
| `rule_fit` | imodels library | Moderate - library exists but needs integration |
| `C5_rules` | No Python equivalent | C5.0 is proprietary, would need custom implementation |

**Implementation Effort:** High
**Recommendation:** Lower priority unless user demand increases

### B. Advanced Tree Models (2 models)

| Model | Python Availability | Challenge |
|-------|-------------------|-----------|
| `bart` | pymc-bart | Requires Bayesian framework integration |
| `bag_mlp` | Custom implementation | Needs wrapper around sklearn.neural_network |

**Implementation Effort:** High
**Recommendation:** Phase 4-5 features

### C. Advanced Time Series (3 models)

| Model | Python Availability | Challenge |
|-------|-------------------|-----------|
| `window_reg` | Custom | Sliding window aggregation framework |
| `temporal_hierarchy` | No direct Python equivalent | Complex hierarchical forecasting |
| `nnetar_reg` | darts, pytorch_forecasting | Requires deep learning framework |

**Implementation Effort:** High
**Recommendation:** Consider if time series becomes primary focus

### D. Survival Analysis (3 models)

| Model | Python Availability | Challenge |
|-------|-------------------|-----------|
| `survival_reg` | lifelines, scikit-survival | Requires survival analysis framework |
| `proportional_hazards` | lifelines.CoxPHFitter | Different paradigm from standard ML |
| `surv_reg` | lifelines | Parametric survival models |

**Implementation Effort:** High (new domain)
**Recommendation:** Phase 5+ or separate package

### E. Platform-Specific Models (1 model)

| Model | Python Availability | Challenge |
|-------|-------------------|-----------|
| `auto_ml` | H2O AutoML (requires Java), auto-sklearn | Platform dependencies, complex integration |

**Implementation Effort:** Very High
**Recommendation:** Low priority (many alternatives exist)

---

## Recommended Implementation Roadmap 🗺️

### Phase 4A: Core Expansion (5-8 models)
**Estimated Effort:** 2-3 weeks
**Goal:** Cover most common ML use cases

1. ✅ `boost_tree` (xgboost, lightgbm, catboost)
2. ✅ `logistic_reg` (sklearn, statsmodels)
3. ✅ `decision_tree` (sklearn)
4. ✅ `naive_Bayes` (sklearn)
5. ✅ `nearest_neighbor` (sklearn)
6. `svm_rbf` (sklearn)
7. `multinom_reg` (sklearn)
8. `exp_smoothing` (statsmodels)

**Deliverables:**
- Implement 5-8 new model specs
- Add engine registrations
- Write comprehensive tests (20+ per model)
- Create demo notebooks
- Update documentation

### Phase 4B: Intermediate Expansion (6-10 models)
**Estimated Effort:** 2-3 weeks
**Goal:** Add specialized but valuable models

1. `svm_linear` (sklearn)
2. `poisson_reg` (statsmodels)
3. `mlp` (sklearn, pytorch)
4. `discrim_linear` (sklearn)
5. `discrim_quad` (sklearn)
6. `mars` (py-earth)
7. `gen_additive_mod` (pygam)
8. `pls` (sklearn)
9. `naive_reg` (custom)
10. `null_model` (custom)

**Deliverables:**
- Implement 6-10 additional models
- Comprehensive testing
- Demo notebooks for each category
- Performance benchmarking

### Phase 4C: Advanced Features (5+ models)
**Estimated Effort:** 3-4 weeks
**Goal:** Advanced and hybrid models

1. `bag_tree` (sklearn)
2. `arima_boost` (statsmodels + xgboost)
3. `prophet_boost` (prophet + xgboost)
4. `seasonal_reg` (statsmodels)
5. `nnetar_reg` (darts or pytorch_forecasting)
6. `rule_fit` (imodels)
7. `bart` (pymc-bart)

**Deliverables:**
- Advanced model implementations
- Hybrid model framework
- Extensive testing
- Advanced demo notebooks

### Phase 5: Specialized Domains (Optional)
**Estimated Effort:** Variable
**Goal:** Domain-specific models as needed

- Survival analysis models (if demand exists)
- Additional rule-based models
- Temporal hierarchical models
- AutoML integration

---

## Engine Implementation Status

### Engines Currently Supported ✅
1. **sklearn** - linear_reg, rand_forest
2. **statsmodels** - linear_reg (OLS), arima_reg
3. **prophet** - prophet_reg
4. **skforecast** - recursive_reg

### Engines Ready to Add (Phase 4A)
5. **xgboost** - boost_tree 🎯 PRIORITY
6. **lightgbm** - boost_tree 🎯 PRIORITY
7. **catboost** - boost_tree 🎯 PRIORITY
8. **pmdarima** - arima_reg (auto_arima) 🎯 PRIORITY

### Engines for Future Phases
9. **pytorch** - mlp, nnetar_reg
10. **py-earth** - mars, bag_mars
11. **pygam** - gen_additive_mod
12. **imodels** - rule_fit
13. **lifelines** - survival models
14. **darts** - advanced time series
15. **pymc-bart** - bart

---

## Gap Analysis by Category

| Category | Total in R | Implemented | Gap | Coverage % |
|----------|-----------|-------------|-----|------------|
| Time Series | 11 | 3 | 8 | 27% |
| Tree-Based | 6 | 1 | 5 | 17% |
| Linear Models | 4 | 1 | 3 | 25% |
| SVM | 3 | 0 | 3 | 0% |
| Neural Networks | 2 | 0 | 2 | 0% |
| Rule-Based | 3 | 0 | 3 | 0% |
| Spline/Adaptive | 3 | 0 | 3 | 0% |
| Discriminant | 4 | 0 | 4 | 0% |
| Other Classifiers | 2 | 0 | 2 | 0% |
| Dimensionality | 1 | 0 | 1 | 0% |
| Survival | 3 | 0 | 3 | 0% |
| Baseline/AutoML | 2 | 0 | 2 | 0% |
| **TOTAL** | **43** | **5** | **38** | **11.6%** |

---

## Recommendations Summary

### Immediate Next Steps (Phase 4A)

1. **Implement boost_tree with 3 engines** (xgboost, lightgbm, catboost)
   - Highest demand model not yet implemented
   - Excellent Python library support
   - Essential for competitive ML workflows

2. **Implement logistic_reg** (sklearn, statsmodels)
   - Natural complement to linear_reg
   - Essential for classification tasks
   - Low implementation effort (mirrors linear_reg)

3. **Implement decision_tree** (sklearn)
   - Fundamental ML algorithm
   - Quick implementation
   - Foundation for understanding tree ensembles

4. **Implement svm_rbf** (sklearn)
   - Popular kernel SVM
   - sklearn provides complete implementation
   - Moderate implementation effort

5. **Implement naive_Bayes and nearest_neighbor** (sklearn)
   - Essential baselines for classification
   - Very low implementation effort
   - Complete toolkit for beginners

### Strategic Priorities

**Quick Wins (Low Effort, High Value):**
- logistic_reg, decision_tree, naive_Bayes, nearest_neighbor, null_model
- **Estimated Time:** 1-2 weeks total

**High Impact (Medium Effort, Very High Value):**
- boost_tree (3 engines), svm_rbf, multinom_reg, exp_smoothing
- **Estimated Time:** 2-3 weeks total

**Long-term Value (High Effort, Specialized Value):**
- MARS, GAMs, rule-based models, survival analysis
- **Estimated Time:** 3-4 weeks each category

---

## Python Library Availability Assessment

### Excellent Support ✅ (Ready to Use)
- sklearn (all tree, linear, SVM, discriminant, neighbors, naive Bayes models)
- statsmodels (ARIMA, exponential smoothing, GLMs)
- xgboost, lightgbm, catboost (gradient boosting)
- prophet (Facebook Prophet)
- pmdarima (auto ARIMA)

### Good Support ✓ (May Need Adaptation)
- py-earth (MARS)
- pygam (GAMs)
- scikit-survival, lifelines (survival analysis)
- imodels (rule-based models)

### Limited Support ⚠️ (Requires Significant Work)
- Cubist rules (no Python equivalent - would need port)
- C5.0 (proprietary)
- Temporal hierarchy (complex implementation)
- BART (requires Bayesian framework)

### Not Available ❌ (Would Need Custom Implementation)
- Some specialized time series (window_reg, some seasonal methods)
- Flexible discriminant analysis (needs custom implementation)
- Some advanced ensemble methods

---

## Conclusion

py-tidymodels has successfully implemented the **core foundation** (11.6% of R tidymodels), focusing on essential time series and basic ML models. The ecosystem is ready for rapid expansion.

**Recommended Focus for Phase 4:**
1. Boost tree models (xgboost, lightgbm, catboost) - fills major gap
2. Classification models (logistic_reg, naive_Bayes, nearest_neighbor) - essential ML
3. SVM models (all three variants) - powerful non-linear methods
4. Decision tree (foundation) and additional time series (exp_smoothing)

**Expected Outcome:**
- Coverage: 11.6% → 35-40% (15-17 models)
- Comprehensive ML toolkit for regression AND classification
- Competitive with other Python ML frameworks
- Strong differentiation via tidymodels API + production-ready outputs

**Timeline Estimate:**
- Phase 4A (8 models): 2-3 weeks
- Phase 4B (10 models): 2-3 weeks
- Phase 4C (7 models): 3-4 weeks
- **Total to 30 models:** 7-10 weeks

---

*This analysis provides a complete roadmap for expanding py-tidymodels to cover the most valuable models from R's tidymodels ecosystem while prioritizing those with excellent Python library support and high user demand.*
