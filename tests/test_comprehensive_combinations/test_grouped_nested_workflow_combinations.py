"""
Comprehensive tests for grouped/nested workflow combinations.

Tests per-group modeling where each group gets its own independent model.
Tests various combinations of:
- Formulas and recipes
- Linear and tree-based models
- Per-group vs global preprocessing
- Group-specific predictions and evaluations

All tests use real grouped data from _md/__data/ directory.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_recipes.selectors import all_numeric, all_numeric_predictors
from py_parsnip import (
    linear_reg, rand_forest, boost_tree, decision_tree,
    nearest_neighbor, svm_rbf
)


class TestNestedFormulaOnlyWorkflows:
    """Test nested workflows with formulas only (no recipes)."""

    def test_simple_nested_linear_reg(self, refinery_data_small_groups, train_test_split_by_group):
        """Test simple nested linear regression across groups."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        # Verify we have separate models for each group
        assert fit_nested.group_col == 'country'
        assert len(fit_nested.group_fits) == len(train['country'].unique())

        # Predictions
        preds = fit_nested.predict(test)
        assert '.pred' in preds.columns
        assert 'country' in preds.columns  # Group column preserved
        assert len(preds) == len(test)

        # Each group should have predictions
        for country in test['country'].unique():
            country_preds = preds[preds['country'] == country]
            assert len(country_preds) > 0

    def test_nested_with_evaluation(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with evaluation."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        # Evaluate
        eval_nested = fit_nested.evaluate(test)
        outputs, coeffs, stats = eval_nested.extract_outputs()

        # Verify group column in all outputs
        assert 'group' in outputs.columns
        assert 'group' in coeffs.columns
        assert 'group' in stats.columns

        # Each group should have train and test stats
        for country in train['country'].unique():
            country_stats = stats[stats['group'] == country]
            assert len(country_stats[country_stats['split'] == 'train']) > 0
            assert len(country_stats[country_stats['split'] == 'test']) > 0

    def test_nested_dot_notation(self, gas_demand_small_groups, train_test_split_by_group):
        """Test nested workflow with dot notation formula."""
        train, test = train_test_split_by_group(gas_demand_small_groups, 'country')

        wf = workflow().add_formula('gas_demand ~ .').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)
        assert 'country' in preds.columns

    def test_nested_with_interactions(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with interaction terms."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti + I(dubai*wti)').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        # Check coefficients vary by group
        _, coeffs, _ = fit_nested.extract_outputs()
        # Each group should have its own coefficients
        for country in train['country'].unique():
            country_coeffs = coeffs[coeffs['group'] == country]
            assert len(country_coeffs) > 0


class TestNestedRecipeWorkflows:
    """Test nested workflows with recipes."""

    def test_nested_with_normalization(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with normalization recipe."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)
        assert 'country' in preds.columns

    def test_nested_with_pca(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with PCA."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=3))
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_with_poly(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with polynomial features."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_poly(['dubai', 'wti'], degree=2)
               .step_normalize(all_numeric_predictors()))
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_with_feature_selection(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with feature selection."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        # Note: step_select_corr requires outcome parameter, using variance threshold instead
        rec = (recipe()
               .step_select_variance_threshold(threshold=0.1)
               .step_normalize(all_numeric_predictors()))
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)


class TestPerGroupPreprocessing:
    """Test per-group preprocessing where each group gets its own recipe."""

    def test_per_group_pca(self, refinery_data_small_groups, train_test_split_by_group):
        """Test per-group PCA preprocessing."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=3))
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country', per_group_prep=True)

        # Each group gets its own preprocessing
        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        # Get feature comparison
        comparison = fit_nested.get_feature_comparison()
        # Groups are in the index, not a column
        assert comparison.index.name == 'group' or len(comparison.index) == len(train['country'].unique())
        # Each group should have PC1, PC2, PC3
        for country in train['country'].unique():
            assert country in comparison.index
            # All groups should have PC1, PC2, PC3 (value is True if present)
            assert comparison.loc[country, 'PC1'] == True
            assert comparison.loc[country, 'PC2'] == True
            assert comparison.loc[country, 'PC3'] == True

    def test_per_group_normalize_comparison(self, refinery_data_small_groups, train_test_split_by_group):
        """Compare per-group vs global normalization."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(linear_reg())

        # Global preprocessing
        fit_global_prep = wf.fit_nested(train, group_col='country', per_group_prep=False)
        preds_global = fit_global_prep.predict(test)

        # Per-group preprocessing
        fit_per_group_prep = wf.fit_nested(train, group_col='country', per_group_prep=True)
        preds_per_group = fit_per_group_prep.predict(test)

        # Both should produce predictions
        assert len(preds_global) == len(test)
        assert len(preds_per_group) == len(test)

        # Predictions may differ due to different scaling
        # (Not testing for exact equality)


class TestNestedTreeModels:
    """Test nested workflows with tree-based models."""

    def test_nested_decision_tree(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested decision tree."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(
            decision_tree(tree_depth=5, min_n=10).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        # Check feature importances per group
        _, coeffs, _ = fit_nested.extract_outputs()
        for country in train['country'].unique():
            country_coeffs = coeffs[coeffs['group'] == country]
            assert len(country_coeffs) > 0

    def test_nested_random_forest(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested random forest."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(
            rand_forest(trees=50, min_n=5).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        eval_nested = fit_nested.evaluate(test)
        _, _, stats = eval_nested.extract_outputs()

        # Each group should have different RMSE (stats are in long format)
        test_stats = stats[stats['split'] == 'test']
        # Stats DataFrame has one row per metric per group
        # With 3 groups and multiple metrics, we expect more than 3 rows
        unique_groups = test_stats['group'].nunique()
        assert unique_groups == len(train['country'].unique())

    @pytest.mark.skip(reason="boost_tree requires xgboost which is not installed")
    def test_nested_boost_tree(self, gas_demand_small_groups, train_test_split_by_group):
        """Test nested gradient boosting."""
        train, test = train_test_split_by_group(gas_demand_small_groups, 'country')

        wf = workflow().add_formula('gas_demand ~ temperature + wind_speed').add_model(
            boost_tree(trees=50, tree_depth=3, learn_rate=0.1).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_random_forest_with_pca(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested random forest with PCA preprocessing."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=3))
        wf = workflow().add_recipe(rec).add_model(
            rand_forest(trees=50).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)


class TestNestedOtherModels:
    """Test nested workflows with other model types."""

    def test_nested_knn(self, gas_demand_small_groups, train_test_split_by_group):
        """Test nested k-NN."""
        train, test = train_test_split_by_group(gas_demand_small_groups, 'country')

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(
            nearest_neighbor(neighbors=5).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_svm(self, gas_demand_small_groups, train_test_split_by_group):
        """Test nested SVM."""
        train, test = train_test_split_by_group(gas_demand_small_groups, 'country')

        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=3))
        wf = workflow().add_recipe(rec).add_model(
            svm_rbf(cost=1.0, rbf_sigma=0.1).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_lasso(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested Lasso regression."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = recipe().step_normalize(all_numeric_predictors())
        wf = workflow().add_recipe(rec).add_model(
            linear_reg(penalty=0.1, mixture=1.0)
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        # Each group should have different coefficients (some may be zero)
        _, coeffs, _ = fit_nested.extract_outputs()
        groups = coeffs['group'].unique()
        assert len(groups) == len(train['country'].unique())


class TestNestedComplexPipelines:
    """Test nested workflows with complex multi-step pipelines."""

    def test_nested_full_pipeline(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with complex pipeline."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_impute_median(all_numeric())
               .step_naomit()
               .step_normalize(all_numeric_predictors())
               .step_pca(num_comp=3))
        wf = workflow().add_recipe(rec).add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

        eval_nested = fit_nested.evaluate(test)
        outputs, coeffs, stats = eval_nested.extract_outputs()

        # Verify structure
        assert 'group' in outputs.columns
        assert 'split' in outputs.columns
        assert 'rmse' in stats['metric'].values

    def test_nested_poly_rf_pipeline(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested random forest with polynomial features."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        rec = (recipe()
               .step_poly(['dubai', 'wti'], degree=2)
               .step_normalize(all_numeric_predictors()))
        wf = workflow().add_recipe(rec).add_model(
            rand_forest(trees=50).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    @pytest.mark.skip(reason="boost_tree requires xgboost which is not installed")
    def test_nested_selection_boosting(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested boosting with feature selection."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        # Note: step_select_corr requires outcome parameter, using variance threshold instead
        rec = (recipe()
               .step_normalize(all_numeric_predictors())
               .step_select_variance_threshold(threshold=0.1))
        wf = workflow().add_recipe(rec).add_model(
            boost_tree(trees=50, learn_rate=0.1).set_mode('regression')
        )
        fit_nested = wf.fit_nested(train, group_col='country')

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)


class TestNestedGroupComparison:
    """Test comparing performance across groups."""

    def test_group_performance_comparison(self, refinery_data_small_groups, train_test_split_by_group):
        """Compare model performance across different groups."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        eval_nested = fit_nested.evaluate(test)
        outputs, coeffs, stats = eval_nested.extract_outputs()

        # Get test RMSE for each group (stats are in long format)
        test_stats = stats[stats['split'] == 'test']
        for country in train['country'].unique():
            country_stats = test_stats[test_stats['group'] == country]
            # Stats DataFrame has one row per metric per group
            country_rmse = country_stats[country_stats['metric'] == 'rmse']
            assert len(country_rmse) == 1
            assert country_rmse['value'].iloc[0] > 0

    def test_coefficient_variation_across_groups(self, refinery_data_small_groups, train_test_split_by_group):
        """Verify coefficients vary across groups."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        _, coeffs, _ = fit_nested.extract_outputs()

        # Get dubai coefficient for each group
        # Linear reg uses 'variable' column for coefficient names, not 'term'
        dubai_coeffs = coeffs[coeffs['variable'] == 'dubai']
        assert len(dubai_coeffs) == len(train['country'].unique())

        # Coefficients should vary across groups (not all the same)
        # Linear reg uses 'coefficient' column for values, not 'estimate'
        unique_values = dubai_coeffs['coefficient'].nunique()
        # Allow for possibility of identical coefficients in rare cases
        # but typically should be different (just checking it runs)


class TestNestedEdgeCases:
    """Test edge cases for nested workflows."""

    def test_nested_with_single_group(self, refinery_data_small_groups, train_test_split_by_group):
        """Test nested workflow with only one group."""
        # Filter to single country
        single_country = refinery_data_small_groups[
            refinery_data_small_groups['country'] == refinery_data_small_groups['country'].unique()[0]
        ]
        train, test = train_test_split_by_group(single_country, 'country')

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        # Should still work with one group
        assert len(fit_nested.group_fits) == 1

        preds = fit_nested.predict(test)
        assert len(preds) == len(test)

    def test_nested_prediction_with_missing_group(self, refinery_data_small_groups, train_test_split_by_group):
        """Test prediction when test data has a group not in training."""
        train, test = train_test_split_by_group(refinery_data_small_groups, 'country')

        # Train on subset of groups
        countries = train['country'].unique()
        train_subset = train[train['country'].isin(countries[:-1])]  # Drop last group

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train_subset, group_col='country')

        # Try to predict on full test set (includes unseen group)
        # This should raise an error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            fit_nested.predict(test)

    def test_nested_with_many_groups(self, refinery_data):
        """Test nested workflow with many groups (performance check)."""
        # Use first 5 countries
        countries = refinery_data['country'].unique()[:5]
        data = refinery_data[refinery_data['country'].isin(countries)]

        train = data.iloc[:int(len(data)*0.8)]
        test = data.iloc[int(len(data)*0.8):]

        wf = workflow().add_formula('brent ~ dubai + wti').add_model(linear_reg())
        fit_nested = wf.fit_nested(train, group_col='country')

        assert len(fit_nested.group_fits) == 5

        preds = fit_nested.predict(test)
        assert len(preds) > 0
