# !! example recipes with workflows
Models

bag_tree
window_reg
rule_fit
svm_poly

Ideas

1. pytimetk -> create steps out of feature engineering functions
2. parallel processing for workflowsets
3. check statsforecasts models in reference folder
4. check neuralforecast models in reference folder
5. check hierarchicalforecast in reference folder
6. double check filtro package integration - done -- test
7. integrate steps from timetk
8. tuning enhnacements from finetune package
9. model interpretibilty - SHAP suite
10. mlflow integration
11. conformal prediction intervals
12. check any remaining functionality in dials package
13. implement tailor package
14. bayesian models
15. step_splitwise() -- done -- add interactions argument
16. step_safe - surrogate assisted feature extraction
17. step_spline() - xspliner
18. step_eix()
19. helper to create large number of recipes
20. permutation feature selection step

enhancements from testing and to check
12. time series cv - done -- test
13. explode formula
14. time series forecasting with workflows - date treating as categorical -- done -- test
15. splits - should be indexed by date? -- done -- test
16. formula doesnt handle multiple predictors -- done -- test
17. hybrid models as parsnip models  -- done -- test
18. mars - pyearth doesnt work -- ignore
19. pygam error -- done -- test
20. py-rsample - plot splits function?
21. selectors - all_numeric_predictors etc -- !! test
22. inverse transformations


9. ### Phase 2 Documentation
**Documentation Deliverables:**
- [ ] API reference for all Phase 2 packages
- [ ] Tutorial: `02_recipes_and_feature_engineering.ipynb`
- [ ] Tutorial: `03_hyperparameter_tuning.ipynb`
- [ ] Tutorial: `04_multi_model_comparison.ipynb`
- [ ] Demo: `examples/feature_selection_demo.py`
- [ ] Demo: `examples/workflowsets_demo.py`
- [ ] Update README with Phase 2 capabilities
- [ ] Update requirements.txt

### Phase 4 Documentation
**Documentation Deliverables:**
- [ ] Tutorial: `09_dashboard_usage.ipynb`
- [ ] Tutorial: `10_mlflow_integration.ipynb`
- [ ] Comprehensive user guide
- [ ] Complete API reference
- [ ] Comparison guides (vs R tidymodels, sklearn, skforecast)
- [ ] Video tutorials (optional)
- [ ] Final requirements files

