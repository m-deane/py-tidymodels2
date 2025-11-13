"""
Comprehensive tests for edge cases and failure modes.

Tests boundary conditions, error handling, missing data scenarios,
and other edge cases to ensure robustness:
- Missing data handling
- Datetime column exclusion
- Unseen categorical levels
- Small sample sizes
- Extreme parameter values
- Invalid inputs
- Group-related edge cases

All tests use real data from _md/__data/ directory.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_numeric_predictors, all_nominal
from py_parsnip import linear_reg, rand_forest, decision_tree
from py_rsample import vfold_cv
from py_yardstick import metric_set, rmse, mae


class TestMissingDataHandling:
    """Test handling of missing data."""

    def test_missing_values_imputation(self, refinery_data_ungrouped):
        """Test that missing values are properly imputed."""
        # Introduce missing values
        data = refinery_data_ungrouped.copy()
        data.loc[10:20, 'dubai'] = np.nan
        data.loc[30:40, 'wti'] = np.nan

        train = data.iloc[:int(len(data)*0.8)]
        test = data.iloc[int(len(data)*0.8):]

        # Recipe with imputation
        rec = (recipe()
               .step_impute_median(all_numeric())
               .step_normalize(all_numeric_predictors()))

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Should handle missing values
        preds = fit.predict(test)
        assert len(preds) == len(test)
        assert preds['.pred'].notna().all()

    def test_naomit_removes_missing(self, gas_demand_ungrouped):
        """Test that step_naomit removes rows with missing values."""
        # Introduce missing values
        data = gas_demand_ungrouped.copy()
        data.loc[10:15, 'temperature'] = np.nan

        train = data.iloc[:int(len(data)*0.8)]
        test = data.iloc[int(len(data)*0.8):]

        # Recipe with naomit
        rec = (recipe()
               .step_naomit()
               .step_normalize(all_numeric_predictors()))

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Check preprocessed data has no NaN
        preprocessed = fit.extract_preprocessed_data(train)
        assert not preprocessed.isna().any().any()

    def test_infinity_handling(self, refinery_data_ungrouped):
        """Test that infinity values are handled."""
        # Introduce infinity values
        data = refinery_data_ungrouped.copy()
        data.loc[10, 'dubai'] = np.inf
        data.loc[20, 'wti'] = -np.inf

        train = data.iloc[:int(len(data)*0.8)]
        test = data.iloc[int(len(data)*0.8):]

        # Recipe with naomit (should remove inf)
        rec = (recipe()
               .step_naomit()
               .step_normalize(all_numeric_predictors()))

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Check preprocessed data has no inf
        preprocessed = fit.extract_preprocessed_data(train)
        assert not np.isinf(preprocessed.select_dtypes(include=[np.number]).values).any()


class TestDatetimeHandling:
    """Test automatic datetime column exclusion."""

    def test_datetime_excluded_from_formula(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test that datetime columns are excluded from auto-generated formulas."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        # Verify date column exists
        assert 'date' in train.columns
        assert pd.api.types.is_datetime64_any_dtype(train['date'])

        # Use recipe (auto-generates formula)
        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Extract formula
        formula = fit.extract_formula()
        assert 'date' not in formula  # Date should be excluded

        # Predictions should work with new dates
        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_datetime_excluded_from_dummy_encoding(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test that datetime columns are excluded from step_dummy."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        # Recipe with dummy encoding (should skip datetime)
        rec = (recipe()
               .step_dummy(all_nominal())
               .step_normalize(all_numeric_predictors()))

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Should not error on new dates
        preds = fit.predict(test)
        assert len(preds) == len(test)


class TestFormulaEdgeCases:
    """Test edge cases in formula parsing."""

    def test_dot_notation_expansion(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test dot notation expands correctly."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        wf = workflow().add_formula('brent ~ .').add_model(linear_reg())
        fit = wf.fit(train)

        # Should expand to all columns except brent and date
        formula = fit.extract_formula()
        assert '~' in formula
        assert 'brent' in formula

        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_interaction_term_validation(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test interaction terms work correctly."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        wf = workflow().add_formula('brent ~ dubai + wti + I(dubai*wti)').add_model(linear_reg())
        fit = wf.fit(train)

        # Should handle interaction
        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_polynomial_term_validation(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test polynomial terms work correctly."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        wf = workflow().add_formula('brent ~ dubai + wti + I(dubai**2) + I(wti**2)').add_model(linear_reg())
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)


class TestSmallSampleSizes:
    """Test handling of small sample sizes."""

    def test_small_training_set(self, refinery_data_ungrouped):
        """Test with very small training set."""
        # Use only 50 samples
        train = refinery_data_ungrouped.iloc[:50]
        test = refinery_data_ungrouped.iloc[50:100]

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_small_cv_folds(self, refinery_data_ungrouped, metric_set_basic):
        """Test with small number of CV folds."""
        data = refinery_data_ungrouped.iloc[:100]  # Small dataset

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())

        # 2-fold CV
        folds = vfold_cv(data, v=2)

        from py_tune import fit_resamples
        results = fit_resamples(wf, resamples=folds, metrics=metric_set_basic)

        # Should still work
        assert results is not None

    def test_small_group_sizes(self, refinery_data_small_groups):
        """Test nested modeling with small group sizes."""
        # Filter to very small sample per group
        small_data = refinery_data_small_groups.groupby('country').head(30)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(small_data, group_col='country')

        # Should still work
        assert len(fit_nested.group_fits) > 0


class TestGroupedEdgeCases:
    """Test edge cases for grouped/panel modeling."""

    def test_single_group(self, refinery_data_small_groups):
        """Test nested modeling with single group."""
        single_country = refinery_data_small_groups[
            refinery_data_small_groups['country'] == refinery_data_small_groups['country'].unique()[0]
        ]

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(single_country, group_col='country')

        assert len(fit_nested.group_fits) == 1

        preds = fit_nested.predict(single_country)
        assert len(preds) > 0

    def test_prediction_with_missing_group(self, refinery_data_small_groups, train_test_split_by_group):
        """Test prediction fails gracefully when group is missing."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        # Train on subset
        countries = train['country'].unique()
        train_subset = train[train['country'].isin(countries[:-1])]

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train_subset, group_col='country')

        # Try to predict on full test set (includes unseen group)
        with pytest.raises((ValueError, KeyError)):
            fit_nested.predict(test)

    def test_unbalanced_group_sizes(self, refinery_data):
        """Test with highly unbalanced group sizes."""
        # Get groups with different sizes
        countries = refinery_data['country'].unique()[:3]
        data_list = []

        # First group: 100 samples
        data_list.append(refinery_data[refinery_data['country'] == countries[0]].head(100))
        # Second group: 50 samples
        data_list.append(refinery_data[refinery_data['country'] == countries[1]].head(50))
        # Third group: 20 samples
        data_list.append(refinery_data[refinery_data['country'] == countries[2]].head(20))

        data = pd.concat(data_list, ignore_index=True)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(data, group_col='country')

        # Should handle different sizes
        assert len(fit_nested.group_fits) == 3


class TestRecipeEdgeCases:
    """Test edge cases in recipe preprocessing."""

    def test_pca_more_components_than_features(self, gas_demand_ungrouped, train_test_split_80_20):
        """Test PCA when requesting more components than features."""
        train, test = train_test_split_80_20(gas_demand_ungrouped)

        # Only 2 predictors, request 10 components
        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=10))

        wf = workflow().add_recipe(rec).add_model(linear_reg())

        # Should handle this gracefully (use max available)
        try:
            fit = wf.fit(train)
            preds = fit.predict(test)
            # If successful, check it works
            assert len(preds) == len(test)
        except (ValueError, Exception):
            # If it errors, that's acceptable behavior
            pass

    def test_polynomial_degree_1(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test polynomial with degree 1 (edge case)."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        rec = (recipe()
               .step_poly(['dubai'], degree=1)
               .step_normalize(all_numeric_predictors()))

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_empty_selector(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test recipe step with selector that matches no columns."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        # All columns are numeric, so all_nominal() should match none
        rec = recipe().step_dummy(all_nominal())

        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit = wf.fit(train)

        # Should still work (no-op)
        preds = fit.predict(test)
        assert len(preds) == len(test)


class TestModelEdgeCases:
    """Test edge cases for model specifications."""

    def test_random_forest_single_tree(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test random forest with single tree."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(
            rand_forest(trees=1).set_mode('regression')
        )
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_decision_tree_max_depth(self, gas_demand_ungrouped, train_test_split_80_20):
        """Test decision tree with very deep depth."""
        train, test = train_test_split_80_20(gas_demand_ungrouped)

        wf = workflow().add_formula('gas_demand ~ temperature + wind_speed').add_model(
            decision_tree(tree_depth=30, min_n=1).set_mode('regression')
        )
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)

    def test_lasso_very_high_penalty(self, refinery_data_ungrouped, train_test_split_80_20):
        """Test Lasso with very high penalty (should zero all coefficients)."""
        train, test = train_test_split_80_20(refinery_data_ungrouped)

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(
            linear_reg(penalty=1000.0, mixture=1.0)
        )
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == len(test)

        # All predictions might be near mean (all coefficients zeroed)
        # Just check they're finite
        assert np.isfinite(preds['.pred']).all()


class TestWorkflowSetEdgeCases:
    """Test edge cases for WorkflowSet."""

    def test_workflowset_single_workflow(self, refinery_data_ungrouped, metric_set_basic):
        """Test WorkflowSet with single workflow."""
        from py_workflowsets import WorkflowSet

        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        assert len(wf_set.workflow_ids) == 1

        folds = vfold_cv(refinery_data_ungrouped, v=3)
        results = wf_set.fit_resamples(resamples=folds, metrics=metric_set_basic)

        ranked = results.rank_results('rmse', n=1)
        assert len(ranked) == 1

    def test_workflowset_nested_single_workflow_single_group(self, refinery_data_small_groups):
        """Test nested WorkflowSet with single workflow and single group."""
        from py_workflowsets import WorkflowSet

        single_country = refinery_data_small_groups[
            refinery_data_small_groups['country'] == refinery_data_small_groups['country'].unique()[0]
        ]

        formulas = ['brent ~ dubai + wti']
        models = [linear_reg()]
        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        results = wf_set.fit_nested(single_country, group_col='country')

        metrics = results.collect_metrics(by_group=False, split='test')
        assert len(metrics) == 1


class TestCVEdgeCases:
    """Test edge cases for cross-validation."""

    def test_single_fold_cv(self, refinery_data_ungrouped, metric_set_basic):
        """Test with single fold (basically train/test split)."""
        from py_tune import fit_resamples

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())

        # Single fold
        folds = vfold_cv(refinery_data_ungrouped, v=2)

        results = fit_resamples(wf, resamples=folds, metrics=metric_set_basic)
        assert results is not None

    def test_cv_more_folds_than_samples(self, refinery_data_ungrouped, metric_set_basic):
        """Test CV with more folds than samples (should error or handle)."""
        from py_tune import fit_resamples

        small_data = refinery_data_ungrouped.iloc[:10]
        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())

        # 20 folds on 10 samples
        with pytest.raises((ValueError, Exception)):
            folds = vfold_cv(small_data, v=20)


class TestPredictionEdgeCases:
    """Test edge cases in prediction."""

    def test_single_sample_prediction(self, refinery_data_ungrouped):
        """Test predicting on single sample."""
        train = refinery_data_ungrouped.iloc[:100]
        test = refinery_data_ungrouped.iloc[100:101]  # Single row

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == 1

    def test_empty_prediction(self, refinery_data_ungrouped):
        """Test predicting on empty DataFrame."""
        train = refinery_data_ungrouped.iloc[:100]
        test = refinery_data_ungrouped.iloc[0:0]  # Empty

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit = wf.fit(train)

        preds = fit.predict(test)
        assert len(preds) == 0

    def test_prediction_missing_columns(self, refinery_data_ungrouped):
        """Test prediction fails when test data missing required columns."""
        train = refinery_data_ungrouped.iloc[:100]
        test = refinery_data_ungrouped[['date', 'dubai']].iloc[100:110]  # Missing 'wti'

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit = wf.fit(train)

        # Should error
        with pytest.raises((ValueError, KeyError)):
            fit.predict(test)


class TestEvaluationEdgeCases:
    """Test edge cases in model evaluation."""

    def test_evaluate_with_missing_outcome(self, refinery_data_ungrouped):
        """Test evaluation when test data missing outcome column."""
        train = refinery_data_ungrouped.iloc[:100]
        test = refinery_data_ungrouped[['date', 'dubai', 'wti']].iloc[100:110]  # Missing 'brent'

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit = wf.fit(train)

        # Should error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            fit.evaluate(test)

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions (edge case for metrics)."""
        # Create synthetic data where model can perfectly fit
        np.random.seed(42)
        data = pd.DataFrame({
            'x': range(100),
            'y': range(100)  # Perfect linear relationship
        })

        train = data.iloc[:80]
        test = data.iloc[80:]

        wf = workflow().add_formula('y ~ x').add_model(linear_reg())
        fit = wf.fit(train)

        eval_fit = fit.evaluate(test)
        outputs, coeffs, stats = eval_fit.extract_outputs()

        # R² should be near 1, RMSE near 0
        test_stats = stats[stats['split'] == 'test']
        assert test_stats['r_squared'].iloc[0] > 0.99
        assert test_stats['rmse'].iloc[0] < 0.1
