Bugs

From examples in "/_md/forecasting.ipynb"

1. when trying to use a recipe with step_corr()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[49], line 17
      3 # Recipe-based workflow with advanced feature engineering
      4 # Advantages: Normalization, encoding, imputation, feature creation, etc.
      5 
      6 # For recipes, we prep/bake outside the workflow, then use formula in workflow
      7 rec = (
      8     recipe()  # Create empty recipe
      9     .step_normalize()  # Normalize numeric features (z-score) - None = all numeric
     10 )
     12 rec = (recipe()
     13     .step_impute_median()      # 1. Impute
     14     .step_log(["totaltar"])       # 2. Transform
     15     .step_normalize()          # 3. Normalize
     16     #.step_dummy(["category"])  # 4. Encode
---> 17     .step_corr(threshold=0.9)  # 5. Filter
     18 )
     20 # Prep the recipe on training data
     21 rec_prepped = rec.prep(train_data)

AttributeError: 'Recipe' object has no attribute 'step_corr'

2. when using all_numeric() in a recipe step:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[53], line 21
     12 rec = (recipe()
     13     .step_impute_median()      # 1. Impute
     14     .step_boxcox(all_numeric())       # 2. Transform
   (...)
     17     #.step_corr(threshold=0.9)  # 5. Filter
     18 )
     20 # Prep the recipe on training data
---> 21 rec_prepped = rec.prep(train_data)
     23 # Apply to both train and test
     24 train_processed_recipe = rec_prepped.bake(train_data)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/recipe.py:1542, in Recipe.prep(self, data, training)
   1538 current_data = data.copy()
   1540 for step in self.steps:
   1541     # Prep the step
-> 1542     prepared_step = step.prep(current_data, training=training)
   1543     prepared_steps.append(prepared_step)
   1545     # Apply step to current data for next step

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps/transformations.py:192, in StepBoxCox.prep(self, data, training)
    190     cols = data.select_dtypes(include=[np.number]).columns.tolist()
    191 else:
--> 192     cols = [col for col in self.columns if col in data.columns]
    194 # Fit transformers
    195 transformers = {}

TypeError: 'function' object is not iterable

3. cannot import recipe selector helpers 

---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
Cell In[56], line 1
----> 1 from py_recipes.selectors import (
      2     all_numeric,      # All numeric columns
      3     all_nominal,      # All categorical columns
      4     all_predictors,   # All predictor columns
      5     all_outcomes,     # All outcome columns
      6     has_role,         # Columns with specific role
      7     has_type,         # Columns with specific type
      8 )
     10 print("APPROACH 2: Recipe-Based Workflow (Powerful)")
     12 # Recipe-based workflow with advanced feature engineering
     13 # Advantages: Normalization, encoding, imputation, feature creation, etc.
     14 
     15 # For recipes, we prep/bake outside the workflow, then use formula in workflow

ImportError: cannot import name 'all_predictors' from 'py_recipes.selectors' (/Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/selectors.py)

4. in this example - do not rename the transformed variable - i.e. totaltar becomes totaltar^2 - or in the recipe step include an argument for inplace=True which will not rename the column, whereas when it is left "False" it will rename the transformed column and add it to the dataset. Always default this to False

---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/compat.py:40, in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
     39 try:
---> 40     return f(*args, **kwargs)
     41 except Exception as e:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:179, in EvalEnvironment.eval(self, expr, source_name, inner_namespace)
    178 code = compile(expr, source_name, "eval", self.flags, False)
--> 179 return eval(code, {}, VarLookupDict([inner_namespace] + self._namespaces))

File <string>:1

NameError: name 'totaltar' is not defined

The above exception was the direct cause of the following exception:

PatsyError                                Traceback (most recent call last)
File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:161, in mold(formula, data, intercept, indicators)
    159 try:
    160     # Create design matrices and capture design info
--> 161     y_mat, X_mat = dmatrices(
    162         expanded_formula,
    163         data,
    164         return_type="dataframe",
    165         NA_action="raise",  # Fail on missing values - user should handle this
    166     )
    168     # Extract design info from the matrices for later use in forge()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:317, in dmatrices(formula_like, data, eval_env, NA_action, return_type)
    316 eval_env = EvalEnvironment.capture(eval_env, reference=1)
--> 317 (lhs, rhs) = _do_highlevel_design(
    318     formula_like, data, eval_env, NA_action, return_type
    319 )
    320 if lhs.shape[1] == 0:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:162, in _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type)
    160     return iter([data])
--> 162 design_infos = _try_incr_builders(
    163     formula_like, data_iter_maker, eval_env, NA_action
    164 )
    165 if design_infos is not None:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:56, in _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
     55     assert isinstance(eval_env, EvalEnvironment)
---> 56     return design_matrix_builders(
     57         [formula_like.lhs_termlist, formula_like.rhs_termlist],
     58         data_iter_maker,
     59         eval_env,
     60         NA_action,
     61     )
     62 else:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:746, in design_matrix_builders(termlists, data_iter_maker, eval_env, NA_action)
    744 # Now all the factors have working eval methods, so we can evaluate them
    745 # on some data to find out what type of data they return.
--> 746 (num_column_counts, cat_levels_contrasts) = _examine_factor_types(
    747     all_factors, factor_states, data_iter_maker, NA_action
    748 )
    749 # Now we need the factor infos, which encapsulate the knowledge of
    750 # how to turn any given factor into a chunk of data:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:491, in _examine_factor_types(factors, factor_states, data_iter_maker, NA_action)
    490 for factor in list(examine_needed):
--> 491     value = factor.eval(factor_states[factor], data)
    492     if factor in cat_sniffers or guess_categorical(value):

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:599, in EvalFactor.eval(self, memorize_state, data)
    598 def eval(self, memorize_state, data):
--> 599     return self._eval(memorize_state["eval_code"], memorize_state, data)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:582, in EvalFactor._eval(self, code, memorize_state, data)
    581 inner_namespace = VarLookupDict([data, memorize_state["transforms"]])
--> 582 return call_and_wrap_exc(
    583     "Error evaluating factor",
    584     self,
    585     memorize_state["eval_env"].eval,
    586     code,
    587     inner_namespace=inner_namespace,
    588 )

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/compat.py:43, in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
     42 new_exc = PatsyError("%s: %s: %s" % (msg, e.__class__.__name__, e), origin)
---> 43 raise new_exc from e

PatsyError: Error evaluating factor: NameError: name 'totaltar' is not defined
    target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1
             ^^^^^^^^

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
Cell In[67], line 51
     42 print(train_processed_recipe.columns)
     44 wf_recipe = (
     45     workflow()
     46     # Formula on preprocessed data
     47     .add_formula(FORMULA_STR)
     48     .add_model(linear_reg())
     49 )
---> 51 fit_recipe = wf_recipe.fit(train_processed_recipe)
     52 fit_recipe = fit_recipe.evaluate(test_processed_recipe)
     54 outputs_recipe, coefs_recipe, stats_recipe = fit_recipe.extract_outputs()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py:230, in Workflow.fit(self, data)
    226     raise ValueError("Workflow must have a formula (via add_formula()) or recipe (via add_recipe())")
    228 # Fit the model (data first, then formula)
    229 # Pass original training data for engines that need raw datetime/categorical values
--> 230 model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)
    232 return WorkflowFit(
    233     workflow=self,
    234     pre=fitted_preprocessor,
    235     fit=model_fit,
    236     post=self.post
    237 )

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/model_spec.py:200, in ModelSpec.fit(self, data, formula, original_training_data, date_col)
    198     fit_data = engine.fit(self, minimal_molded, original_training_data=data)
    199 elif formula is not None:
--> 200     molded = mold(formula, data)
    201     # Pass original_training_data to engine.fit() for datetime column extraction
    202     # If not provided, use data itself (direct fit() calls have original data)
    203     # Check if engine.fit() accepts original_training_data parameter
    204     import inspect

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:173, in mold(formula, data, intercept, indicators)
    170     outcome_design_info = y_mat.design_info
    172 except Exception as e:
--> 173     raise ValueError(
    174         f"Failed to parse formula '{formula}': {str(e)}\n"
    175         f"(Expanded to: '{expanded_formula}')"
    176     ) from e
    178 # Handle intercept option
    179 if not intercept and "Intercept" in X_mat.columns:

ValueError: Failed to parse formula 'target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1': Error evaluating factor: NameError: name 'totaltar' is not defined
    target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1
             ^^^^^^^^
(Expanded to: 'target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1')


5. for steps that create new variables or transform existing - give me option to drop the original columns and/or update the formula with the new variable names and drop the old ones

---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/compat.py:40, in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
     39 try:
---> 40     return f(*args, **kwargs)
     41 except Exception as e:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:179, in EvalEnvironment.eval(self, expr, source_name, inner_namespace)
    178 code = compile(expr, source_name, "eval", self.flags, False)
--> 179 return eval(code, {}, VarLookupDict([inner_namespace] + self._namespaces))

File <string>:1

NameError: name 'totaltar' is not defined

The above exception was the direct cause of the following exception:

PatsyError                                Traceback (most recent call last)
File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:161, in mold(formula, data, intercept, indicators)
    159 try:
    160     # Create design matrices and capture design info
--> 161     y_mat, X_mat = dmatrices(
    162         expanded_formula,
    163         data,
    164         return_type="dataframe",
    165         NA_action="raise",  # Fail on missing values - user should handle this
    166     )
    168     # Extract design info from the matrices for later use in forge()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:317, in dmatrices(formula_like, data, eval_env, NA_action, return_type)
    316 eval_env = EvalEnvironment.capture(eval_env, reference=1)
--> 317 (lhs, rhs) = _do_highlevel_design(
    318     formula_like, data, eval_env, NA_action, return_type
    319 )
    320 if lhs.shape[1] == 0:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:162, in _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type)
    160     return iter([data])
--> 162 design_infos = _try_incr_builders(
    163     formula_like, data_iter_maker, eval_env, NA_action
    164 )
    165 if design_infos is not None:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:56, in _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
     55     assert isinstance(eval_env, EvalEnvironment)
---> 56     return design_matrix_builders(
     57         [formula_like.lhs_termlist, formula_like.rhs_termlist],
     58         data_iter_maker,
     59         eval_env,
     60         NA_action,
     61     )
     62 else:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:746, in design_matrix_builders(termlists, data_iter_maker, eval_env, NA_action)
    744 # Now all the factors have working eval methods, so we can evaluate them
    745 # on some data to find out what type of data they return.
--> 746 (num_column_counts, cat_levels_contrasts) = _examine_factor_types(
    747     all_factors, factor_states, data_iter_maker, NA_action
    748 )
    749 # Now we need the factor infos, which encapsulate the knowledge of
    750 # how to turn any given factor into a chunk of data:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:491, in _examine_factor_types(factors, factor_states, data_iter_maker, NA_action)
    490 for factor in list(examine_needed):
--> 491     value = factor.eval(factor_states[factor], data)
    492     if factor in cat_sniffers or guess_categorical(value):

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:599, in EvalFactor.eval(self, memorize_state, data)
    598 def eval(self, memorize_state, data):
--> 599     return self._eval(memorize_state["eval_code"], memorize_state, data)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:582, in EvalFactor._eval(self, code, memorize_state, data)
    581 inner_namespace = VarLookupDict([data, memorize_state["transforms"]])
--> 582 return call_and_wrap_exc(
    583     "Error evaluating factor",
    584     self,
    585     memorize_state["eval_env"].eval,
    586     code,
    587     inner_namespace=inner_namespace,
    588 )

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/compat.py:43, in call_and_wrap_exc(msg, origin, f, *args, **kwargs)
     42 new_exc = PatsyError("%s: %s: %s" % (msg, e.__class__.__name__, e), origin)
---> 43 raise new_exc from e

PatsyError: Error evaluating factor: NameError: name 'totaltar' is not defined
    target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1
             ^^^^^^^^

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
Cell In[74], line 54
     45 print(train_processed_recipe.columns)
     47 wf_recipe = (
     48     workflow()
     49     # Formula on preprocessed data
     50     .add_formula(FORMULA_STR)
     51     .add_model(linear_reg())
     52 )
---> 54 fit_recipe = wf_recipe.fit(train_processed_recipe)
     55 fit_recipe = fit_recipe.evaluate(test_processed_recipe)
     57 outputs_recipe, coefs_recipe, stats_recipe = fit_recipe.extract_outputs()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py:230, in Workflow.fit(self, data)
    226     raise ValueError("Workflow must have a formula (via add_formula()) or recipe (via add_recipe())")
    228 # Fit the model (data first, then formula)
    229 # Pass original training data for engines that need raw datetime/categorical values
--> 230 model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)
    232 return WorkflowFit(
    233     workflow=self,
    234     pre=fitted_preprocessor,
    235     fit=model_fit,
    236     post=self.post
    237 )

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/model_spec.py:200, in ModelSpec.fit(self, data, formula, original_training_data, date_col)
    198     fit_data = engine.fit(self, minimal_molded, original_training_data=data)
    199 elif formula is not None:
--> 200     molded = mold(formula, data)
    201     # Pass original_training_data to engine.fit() for datetime column extraction
    202     # If not provided, use data itself (direct fit() calls have original data)
    203     # Check if engine.fit() accepts original_training_data parameter
    204     import inspect

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:173, in mold(formula, data, intercept, indicators)
    170     outcome_design_info = y_mat.design_info
    172 except Exception as e:
--> 173     raise ValueError(
    174         f"Failed to parse formula '{formula}': {str(e)}\n"
    175         f"(Expanded to: '{expanded_formula}')"
    176     ) from e
    178 # Handle intercept option
    179 if not intercept and "Intercept" in X_mat.columns:

ValueError: Failed to parse formula 'target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1': Error evaluating factor: NameError: name 'totaltar' is not defined
    target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1
             ^^^^^^^^
(Expanded to: 'target ~ totaltar +mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1')

6. in step_lag

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[79], line 44
     21 rec = (
     22     recipe()
     23     .step_impute_median()      # 1. Impute
   (...)
     40     .step_timeseries_signature(["date"])        # Extract time features
     41 )
     43 # Prep the recipe on training data
---> 44 rec_prepped = rec.prep(train_data)
     46 # Apply to both train and test
     47 train_processed_recipe = rec_prepped.bake(train_data)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/recipe.py:1542, in Recipe.prep(self, data, training)
   1538 current_data = data.copy()
   1540 for step in self.steps:
   1541     # Prep the step
-> 1542     prepared_step = step.prep(current_data, training=training)
   1543     prepared_steps.append(prepared_step)
   1545     # Apply step to current data for next step

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps/timeseries_extended.py:444, in StepTimeseriesSignature.prep(self, data, training)
    433 def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepTimeseriesSignature":
    434     """
    435     Prepare timeseries signature extraction.
    436 
   (...)
    442         PreparedStepTimeseriesSignature ready to extract features
    443     """
--> 444     if self.date_column not in data.columns:
    445         return PreparedStepTimeseriesSignature(
    446             date_column=self.date_column,
    447             features=self.features or [],
    448             prefix=self.prefix,
    449             feature_names=[]
    450         )
    452     # Default feature set

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/pandas/core/indexes/base.py:5370, in Index.__contains__(self, key)
   5335 def __contains__(self, key: Any) -> bool:
   5336     """
   5337     Return a boolean indicating whether the provided key is in the index.
   5338 
   (...)
   5368     False
   5369     """
-> 5370     hash(key)
   5371     try:
   5372         return key in self._engine

TypeError: unhashable type: 'list'

7. in step_diff 

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[81], line 44
     21 rec = (
     22     recipe()
     23     .step_impute_median()      # 1. Impute
   (...)
     40     .step_timeseries_signature(["date"])        # Extract time features
     41 )
     43 # Prep the recipe on training data
---> 44 rec_prepped = rec.prep(train_data)
     46 # Apply to both train and test
     47 train_processed_recipe = rec_prepped.bake(train_data)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/recipe.py:1542, in Recipe.prep(self, data, training)
   1538 current_data = data.copy()
   1540 for step in self.steps:
   1541     # Prep the step
-> 1542     prepared_step = step.prep(current_data, training=training)
   1543     prepared_steps.append(prepared_step)
   1545     # Apply step to current data for next step

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_recipes/steps/timeseries_extended.py:444, in StepTimeseriesSignature.prep(self, data, training)
    433 def prep(self, data: pd.DataFrame, training: bool = True) -> "PreparedStepTimeseriesSignature":
    434     """
    435     Prepare timeseries signature extraction.
    436 
   (...)
    442         PreparedStepTimeseriesSignature ready to extract features
    443     """
--> 444     if self.date_column not in data.columns:
    445         return PreparedStepTimeseriesSignature(
    446             date_column=self.date_column,
    447             features=self.features or [],
    448             prefix=self.prefix,
    449             feature_names=[]
    450         )
    452     # Default feature set

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/pandas/core/indexes/base.py:5370, in Index.__contains__(self, key)
   5335 def __contains__(self, key: Any) -> bool:
   5336     """
   5337     Return a boolean indicating whether the provided key is in the index.
   5338 
   (...)
   5368     False
   5369     """
-> 5370     hash(key)
   5371     try:
   5372         return key in self._engine

TypeError: unhashable type: 'list'

8. the pattern matching using the recipe selectors does not work:

# Pattern matching
rec = recipe().step_log(starts_with("price_"))
rec = recipe().step_center(ends_with("_amount"))

9. ---------------------------------------------------------------------------
SyntaxError                               Traceback (most recent call last)
File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:161, in mold(formula, data, intercept, indicators)
    159 try:
    160     # Create design matrices and capture design info
--> 161     y_mat, X_mat = dmatrices(
    162         expanded_formula,
    163         data,
    164         return_type="dataframe",
    165         NA_action="raise",  # Fail on missing values - user should handle this
    166     )
    168     # Extract design info from the matrices for later use in forge()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:317, in dmatrices(formula_like, data, eval_env, NA_action, return_type)
    316 eval_env = EvalEnvironment.capture(eval_env, reference=1)
--> 317 (lhs, rhs) = _do_highlevel_design(
    318     formula_like, data, eval_env, NA_action, return_type
    319 )
    320 if lhs.shape[1] == 0:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:162, in _do_highlevel_design(formula_like, data, eval_env, NA_action, return_type)
    160     return iter([data])
--> 162 design_infos = _try_incr_builders(
    163     formula_like, data_iter_maker, eval_env, NA_action
    164 )
    165 if design_infos is not None:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/highlevel.py:56, in _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
     55     assert isinstance(eval_env, EvalEnvironment)
---> 56     return design_matrix_builders(
     57         [formula_like.lhs_termlist, formula_like.rhs_termlist],
     58         data_iter_maker,
     59         eval_env,
     60         NA_action,
     61     )
     62 else:

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:743, in design_matrix_builders(termlists, data_iter_maker, eval_env, NA_action)
    742         all_factors.update(term.factors)
--> 743 factor_states = _factors_memorize(all_factors, data_iter_maker, eval_env)
    744 # Now all the factors have working eval methods, so we can evaluate them
    745 # on some data to find out what type of data they return.

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/build.py:393, in _factors_memorize(factors, data_iter_maker, eval_env)
    392 state = {}
--> 393 which_pass = factor.memorize_passes_needed(state, eval_env)
    394 factor_states[factor] = state

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:504, in EvalFactor.memorize_passes_needed(self, state, eval_env)
    503 env_namespace = eval_env.namespace
--> 504 subset_names = [name for name in ast_names(self.code) if name in env_namespace]
    505 eval_env = eval_env.subset(subset_names)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:504, in <listcomp>(.0)
    503 env_namespace = eval_env.namespace
--> 504 subset_names = [name for name in ast_names(self.code) if name in env_namespace]
    505 eval_env = eval_env.subset(subset_names)

File ~/Documents/Data Science/python/_projects/py-tidymodels/py-tidymodels2/lib/python3.10/site-packages/patsy/eval.py:111, in ast_names(code)
    109 disallowed_ast_nodes += (ast.DictComp, ast.SetComp)
--> 111 for node in ast.walk(ast.parse(code)):
    112     if isinstance(node, disallowed_ast_nodes):

File ~/anaconda3/lib/python3.10/ast.py:50, in parse(source, filename, mode, type_comments, feature_version)
     49 # Else it should be an int giving the minor version for 3.x.
---> 50 return compile(source, filename, mode, flags,
     51                _feature_version=feature_version)

SyntaxError: invalid syntax (<unknown>, line 1)

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
Cell In[107], line 61
     51 print(train_processed_recipe.columns)
     53 wf_recipe = (
     54     workflow()
     55     # Formula on preprocessed data
   (...)
     58     .add_model(linear_reg())
     59 )
---> 61 fit_recipe = wf_recipe.fit(train_data)
     62 fit_recipe = fit_recipe.evaluate(test_data)
     64 outputs_recipe, coefs_recipe, stats_recipe = fit_recipe.extract_outputs()

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_workflows/workflow.py:230, in Workflow.fit(self, data)
    226     raise ValueError("Workflow must have a formula (via add_formula()) or recipe (via add_recipe())")
    228 # Fit the model (data first, then formula)
    229 # Pass original training data for engines that need raw datetime/categorical values
--> 230 model_fit = self.spec.fit(processed_data, formula, original_training_data=original_data)
    232 return WorkflowFit(
    233     workflow=self,
    234     pre=fitted_preprocessor,
    235     fit=model_fit,
    236     post=self.post
    237 )

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_parsnip/model_spec.py:200, in ModelSpec.fit(self, data, formula, original_training_data, date_col)
    198     fit_data = engine.fit(self, minimal_molded, original_training_data=data)
    199 elif formula is not None:
--> 200     molded = mold(formula, data)
    201     # Pass original_training_data to engine.fit() for datetime column extraction
    202     # If not provided, use data itself (direct fit() calls have original data)
    203     # Check if engine.fit() accepts original_training_data parameter
    204     import inspect

File ~/Documents/Data Science/python/_projects/py-tidymodels/py_hardhat/mold.py:173, in mold(formula, data, intercept, indicators)
    170     outcome_design_info = y_mat.design_info
    172 except Exception as e:
--> 173     raise ValueError(
    174         f"Failed to parse formula '{formula}': {str(e)}\n"
    175         f"(Expanded to: '{expanded_formula}')"
    176     ) from e
    178 # Handle intercept option
    179 if not intercept and "Intercept" in X_mat.columns:

ValueError: Failed to parse formula 'target ~ date + mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1 + mean_nwe_lsfo_crack_trade_month + mean_nwe_ulsfo_crack_trade_month lag3 + mean_sing_gasoline_vs_vlsfo_trade_month + new_sweet_sr_margin + totaltar': invalid syntax (<unknown>, line 1)
(Expanded to: 'target ~ date + mean_med_diesel_crack_input1_trade_month_lag2 + mean_nwe_hsfo_crack_trade_month_lag1 + mean_nwe_lsfo_crack_trade_month + mean_nwe_ulsfo_crack_trade_month lag3 + mean_sing_gasoline_vs_vlsfo_trade_month + new_sweet_sr_margin + totaltar')

Enhancements

1. add all_numeric_predictors(), all_nominal_predictors() selectors to allow control over transformation of exogenous variables but not the target variable


To Check

1. how to reverse recipe steps such as step_normalize to return the predictions in the same scale as the original input_data - inverse transformation to return the correct absolute values of predictions for instance
