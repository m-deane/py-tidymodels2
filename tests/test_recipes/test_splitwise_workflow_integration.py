"""
Workflow integration tests for step_splitwise().

Tests the complete end-to-end workflow:
- Recipe with step_splitwise()
- Workflow composition
- Model fitting and prediction
- Evaluation metrics
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_workflows import workflow
from py_parsnip import linear_reg


class TestSplitwiseWorkflowIntegration:
    """Test step_splitwise integration with workflows."""

    @pytest.fixture
    def nonlinear_data(self):
        """Create dataset with non-linear relationships."""
        np.random.seed(42)
        n = 300

        # Predictor with threshold effect
        x1 = np.random.uniform(-5, 5, n)
        # Predictor with U-shaped effect
        x2 = np.random.uniform(-3, 3, n)
        # Linear predictor
        x3 = np.random.randn(n)

        # Outcome with mixed relationships
        y = (
            10 * (x1 > 0).astype(int) +  # Threshold effect
            5 * ((x2 < -1) | (x2 > 1)).astype(int) +  # U-shaped
            2 * x3 +  # Linear
            np.random.randn(n) * 0.5  # Noise
        )

        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'y': y
        })

    def test_workflow_with_splitwise_basic(self, nonlinear_data):
        """Test basic workflow with step_splitwise."""
        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe with splitwise
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=2.0)
        )

        # Create workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg())
        )

        # Fit on training data
        fit = wf.fit(train)

        # Should have fitted model (accessible via fit.fit attribute)
        assert fit.fit is not None
        assert hasattr(fit.fit, 'spec')

        # Make predictions
        preds = fit.predict(test)

        # Should have predictions
        assert '.pred' in preds.columns
        assert len(preds) == len(test)

    def test_workflow_splitwise_transformations(self, nonlinear_data):
        """Test that splitwise transformations are applied correctly."""
        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe with splitwise
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=1.0)
        )

        # Prep the recipe to inspect transformations
        prepped = rec.prep(train)

        # Access the prepared step (correct attribute: prepared_steps)
        splitwise_step = prepped.prepared_steps[0]
        decisions = splitwise_step.get_decisions()

        # Should have decisions for all predictors
        assert 'x1' in decisions
        assert 'x2' in decisions
        assert 'x3' in decisions

        # x1 and x2 likely transformed (non-linear), x3 likely linear
        # At least one should be transformed to dummy
        transformed_count = sum(
            1 for col, info in decisions.items()
            if info['decision'] in ['single_split', 'double_split']
        )
        assert transformed_count >= 1

        # Bake the data to see transformations
        baked = prepped.bake(train)

        # Should have dummy columns for transformed variables
        dummy_cols = [c for c in baked.columns if '_ge_' in c or '_between_' in c]
        assert len(dummy_cols) >= 1

        # Outcome should be preserved
        assert 'y' in baked.columns

    def test_workflow_splitwise_evaluation(self, nonlinear_data):
        """Test workflow evaluation with splitwise."""
        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe with splitwise
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=2.0)
        )

        # Create workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg())
        )

        # Fit and evaluate
        fit = wf.fit(train)
        fit = fit.evaluate(test)

        # Extract outputs to verify evaluation worked
        outputs, coefficients, stats = fit.extract_outputs()

        # Should have data for both splits
        assert 'train' in outputs['split'].values
        assert 'test' in outputs['split'].values

        # Should have model statistics for both splits
        # Stats are in long format with 'metric' and 'value' columns
        assert len(stats) > 0
        assert 'split' in stats.columns
        assert 'metric' in stats.columns
        assert 'value' in stats.columns

        # Verify we have train and test statistics
        train_stats = stats[stats['split'] == 'train']
        test_stats = stats[stats['split'] == 'test']
        assert len(train_stats) > 0
        assert len(test_stats) > 0

        # Check that rmse metric exists
        assert 'rmse' in stats['metric'].values

    def test_workflow_splitwise_with_other_steps(self, nonlinear_data):
        """Test splitwise combined with other recipe steps."""
        from py_recipes.selectors import all_numeric_predictors

        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe with multiple steps
        rec = (
            recipe()
            .step_splitwise(outcome='y', min_improvement=2.0)
            .step_normalize(all_numeric_predictors())
        )

        # Create workflow
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg())
        )

        # Fit on training data
        fit = wf.fit(train)

        # Make predictions
        preds = fit.predict(test)

        # Should work correctly
        assert '.pred' in preds.columns
        assert len(preds) == len(test)
        assert not preds['.pred'].isna().any()

    def test_workflow_splitwise_exclude_vars(self, nonlinear_data):
        """Test excluding specific variables from transformation."""
        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe excluding x3 from transformation
        rec = (
            recipe()
            .step_splitwise(outcome='y', exclude_vars=['x3'], min_improvement=1.0)
        )

        # Prep and inspect
        prepped = rec.prep(train)
        splitwise_step = prepped.prepared_steps[0]
        decisions = splitwise_step.get_decisions()

        # x3 should be kept linear
        assert decisions['x3']['decision'] == 'linear'

        # Create workflow and fit
        wf = (
            workflow()
            .add_recipe(rec)
            .add_model(linear_reg())
        )

        fit = wf.fit(train)
        preds = fit.predict(test)

        # Should work correctly
        assert '.pred' in preds.columns
        assert len(preds) == len(test)

    def test_workflow_splitwise_criterion_comparison(self, nonlinear_data):
        """Test AIC vs BIC criterion selection."""
        # Split train/test
        train = nonlinear_data.iloc[:200]
        test = nonlinear_data.iloc[200:]

        # Create recipe with AIC
        rec_aic = (
            recipe()
            .step_splitwise(outcome='y', criterion='AIC', min_improvement=1.0)
        )

        # Create recipe with BIC (more conservative)
        rec_bic = (
            recipe()
            .step_splitwise(outcome='y', criterion='BIC', min_improvement=1.0)
        )

        # Prep both
        prepped_aic = rec_aic.prep(train)
        prepped_bic = rec_bic.prep(train)

        # Get decisions
        decisions_aic = prepped_aic.prepared_steps[0].get_decisions()
        decisions_bic = prepped_bic.prepared_steps[0].get_decisions()

        # BIC should be more conservative (fewer transformations or same)
        aic_transformed = sum(
            1 for info in decisions_aic.values()
            if info['decision'] in ['single_split', 'double_split']
        )
        bic_transformed = sum(
            1 for info in decisions_bic.values()
            if info['decision'] in ['single_split', 'double_split']
        )

        # BIC should transform same or fewer variables
        assert bic_transformed <= aic_transformed

    def test_workflow_splitwise_min_support_effect(self, nonlinear_data):
        """Test min_support parameter effect."""
        # Split train/test
        train = nonlinear_data.iloc[:200]

        # Create recipe with high min_support
        rec_high_support = (
            recipe()
            .step_splitwise(outcome='y', min_support=0.3, min_improvement=1.0)
        )

        # Create recipe with low min_support
        rec_low_support = (
            recipe()
            .step_splitwise(outcome='y', min_support=0.1, min_improvement=1.0)
        )

        # Prep both
        prepped_high = rec_high_support.prep(train)
        prepped_low = rec_low_support.prep(train)

        # Get decisions
        decisions_high = prepped_high.prepared_steps[0].get_decisions()
        decisions_low = prepped_low.prepared_steps[0].get_decisions()

        # Higher support should be more restrictive (fewer transformations or same)
        high_transformed = sum(
            1 for info in decisions_high.values()
            if info['decision'] in ['single_split', 'double_split']
        )
        low_transformed = sum(
            1 for info in decisions_low.values()
            if info['decision'] in ['single_split', 'double_split']
        )

        # High support should transform same or fewer variables
        assert high_transformed <= low_transformed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
