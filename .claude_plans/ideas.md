# !! example recipes with workflows
Models

sktime - PieceWiseTrendForecaster
piecewise linear regression
panel regression
auto_arima
auto_ets

Ideas

1. pytimetk -> create steps out of feature engineering functions
2. parallel processing for workflowsets
7. integrate steps from timetk
8. tuning enhnacements from finetune package
9. model interpretibilty - SHAP suite
10. mlflow integration
11. conformal prediction intervals
12. check any remaining functionality in dials package
13. implement tailor package
14. bayesian models
19. helper to create large number of recipes
24. comprehensive testing for all recipes - behaving as expected
26. feature-engine integration
30. workflowsets - leave_out_formula_vars

enhancements from testing and to check
13. explode formula
18. mars - pyearth doesnt work -- ignore
20. py-rsample - plot splits function?
22. inverse transformations
24. fit_global - outcome has to be renamed "target"
25. test inplace recipe transformations

## RECIPES

# Feature Engineering

1. step_h_stat - create interactions based on h stat score above a certain threshold, or top_n highest scoring interactions
2. featureengine - DecisionTreeDiscretiser
3. anomaly - winsoriser
4. anomaly - outlier trimmer
5. feature-engine - DecisionTreeFeatures
6. best_lag_creator() - granger lag selection
7. anomaly - clean_anomalies based on pytimetk
8. step_stationary()
9. step_deseasonalise()
10. step_detrend()

# Feature Selection
1. step_vif - select for vif values based on a threshold based on fitting a linear model
2. step_pvalue - select for p values based on a threshold based on fitting a linear model
3. feature selection - stability score selection
4. feature selection - lofo importance
5. important library - - step_predictor_desirability()
6. step_granger_causality()
7. step_causality()
8. feature-engine - smart correlated selection
9. feature-engine - drophighpsifeatures
10. step_rfe()
11. step_stepwise_selection()
11. step_probe_feature_selection()
12. step_select_lasso()/ridge()
13. step_select_rulefit()
14. step_select_skope_rules()
15. step_select_boruta()
16. step_select_stationarity()
17. causal - transfer entropy

# Dimensionality Reduction

1. t-sne
2. umap

## To Test

statsforecast - auto_arima
daily grouped data - gas
