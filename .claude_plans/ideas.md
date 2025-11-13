# !! example recipes with workflows
Models

sktime - PieceWiseTrendForecaster
piecewise linear regression
panel regression - IP
auto_arima?
auto_ets

Ideas

1. pytimetk -> create steps out of feature engineering functions
2. parallel processing for workflowsets - IP
7. integrate steps from timetk
8. tuning enhnacements from finetune package - IP
9. model interpretibilty - SHAP suite
10. mlflow integration
11. conformal prediction intervals - IP
13. implement tailor package
14. bayesian models
19. helper to create large number of recipes
24. comprehensive testing for all recipes - behaving as expected - IP
30. workflowsets - leave_out_formula_vars

enhancements from testing and to check
13. explode formula
18. mars - pyearth doesnt work -- ignore
20. py-rsample - plot splits function?
22. inverse transformations
24. fit_global - outcome has to be renamed "target"
25. test inplace recipe transformations
26. trelliscopejs integration - IP
27. recipe - explode by list of recipes AND formulas

## RECIPES

# Feature Engineering

# Feature Selection
5. important library - - step_predictor_desirability()
13. step_select_rulefit()
14. step_select_skope_rules()
17. causal - transfer entropy#
18. step_select_genetic_algorithm() - IP
19. steps from advances in financial machine learning - IP

# Dimensionality Reduction

1. t-sne
2. umap

## To Test

statsforecast - auto_arima
daily grouped data - gas
