# !! example recipes with workflows
Models

sktime - PieceWiseTrendForecaster
piecewise linear regression

Ideas

1. pytimetk -> create steps out of feature engineering functions
2. parallel processing for workflowsets
3. check statsforecasts models in reference folder
4. check neuralforecast models in reference folder
5. check hierarchicalforecast in reference folder
7. integrate steps from timetk
8. tuning enhnacements from finetune package
9. model interpretibilty - SHAP suite
10. mlflow integration
11. conformal prediction intervals
12. check any remaining functionality in dials package
13. implement tailor package
14. bayesian models
15. step_splitwise() -- done -- add interactions argument
16. step_safe - surrogate assisted feature extraction -- done -- deprecate original step_safe() with step_safe_v2()
19. helper to create large number of recipes
21. step_causal()
22. step_granger_causality()
23. step_h_stat()
24. comprehensive testing for all recipes - behaving as expected
26. feature-engine integration
27. step_select_pvalue() - filter for p values based on a threshold
28. select_select_vif() - filter for vif values based on a threshold


enhancements from testing and to check
13. explode formula
18. mars - pyearth doesnt work -- ignore
20. py-rsample - plot splits function?
22. inverse transformations
23. time series cv/ train test split as a recipe step

to test

13. step_normalize() etc are group specific transformations
14. order of recipe steps

