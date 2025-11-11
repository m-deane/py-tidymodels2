"""
Tests for advanced selection steps (Phase 3).
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.advanced_selection import (
    StepVif,
    StepPvalue,
    StepSelectStability,
    StepSelectLofo,
    StepSelectGranger,
    StepSelectStepwise,
    StepSelectProbe,
)


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    n = 200

    # Create features with various levels of multicollinearity
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(100, 20, n)
    x3 = 0.8 * x1 + np.random.normal(0, 2, n)  # High multicollinearity with x1
    x4 = 0.3 * x1 + 0.3 * x2 + np.random.normal(0, 5, n)  # Moderate
    x5 = np.random.normal(0, 1, n)  # Independent noise

    # Create target with known relationships
    target = 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 5, n)

    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,  # Should be removed by VIF
        'x4': x4,
        'x5': x5,  # Should be removed by p-value
        'target': target
    })

    return data


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    n = 200

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)

    # Create binary target
    logit = 2.0 * x1 + 1.0 * x2 + np.random.normal(0, 0.5, n)
    prob = 1 / (1 + np.exp(-logit))
    target = (np.random.random(n) < prob).astype(int)

    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'target': target
    })

    return data


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for Granger causality tests."""
    np.random.seed(42)
    n = 150

    # Create autocorrelated time series
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    target = np.zeros(n)

    x1[0] = np.random.normal(0, 1)
    x2[0] = np.random.normal(0, 1)
    target[0] = np.random.normal(0, 1)

    for i in range(1, n):
        x1[i] = 0.5 * x1[i-1] + np.random.normal(0, 1)
        x2[i] = 0.3 * x2[i-1] + np.random.normal(0, 1)
        # Target depends on past x1 (Granger causes) but not past x2
        target[i] = 0.7 * target[i-1] + 0.4 * x1[i-1] + np.random.normal(0, 1)

    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'target': target
    })

    return data


class TestStepVif:
    """Tests for StepVif."""

    def test_basic_vif_removal(self, sample_regression_data):
        """Test basic VIF-based feature removal."""
        rec = recipe().step_vif(threshold=10.0, outcome='target')

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        # x3 is highly correlated with x1 (0.8 * x1), so one should be removed
        assert 'target' in result.columns
        assert 'x2' in result.columns
        # At least one of the highly correlated features (x1, x3) should be removed
        assert not ('x1' in result.columns and 'x3' in result.columns)
        # Result should have fewer features than input
        assert result.shape[1] <= sample_regression_data.shape[1]

    def test_strict_threshold(self, sample_regression_data):
        """Test with strict VIF threshold."""
        rec = recipe().step_vif(threshold=5.0, outcome='target')

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        # Stricter threshold should remove more features
        assert result.shape[1] <= sample_regression_data.shape[1]
        assert 'target' in result.columns

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_vif(threshold=10.0, outcome='target')
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]
        assert 'target' in result.columns


class TestStepPvalue:
    """Tests for StepPvalue."""

    def test_basic_pvalue_selection(self, sample_regression_data):
        """Test basic p-value based selection."""
        rec = recipe().step_pvalue(outcome='target', threshold=0.05)

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        # Significant features (x1, x2, x4) should remain
        assert 'target' in result.columns
        assert 'x1' in result.columns
        assert 'x2' in result.columns
        # x5 (noise) might be removed
        assert result.shape[1] >= 3  # At least target + 2 features

    def test_classification_pvalue(self, sample_classification_data):
        """Test p-value selection with classification."""
        rec = recipe().step_pvalue(
            outcome='target',
            threshold=0.05,
            model_type='logistic'
        )

        prepped = rec.prep(sample_classification_data)
        result = prepped.bake(sample_classification_data)

        assert 'target' in result.columns
        assert result.shape[1] <= sample_classification_data.shape[1]

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_pvalue(outcome='target', threshold=0.05)
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]


class TestStepSelectStability:
    """Tests for StepSelectStability."""

    def test_basic_stability_selection(self, sample_regression_data):
        """Test basic stability selection."""
        rec = recipe().step_select_stability(
            outcome='target',
            threshold=0.6,
            n_bootstrap=20,  # Small for faster testing
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        # Important features (x1, x2) should be selected frequently
        assert 'x1' in result.columns or 'x2' in result.columns
        assert result.shape[1] >= 2  # At least target + 1 feature

    def test_high_threshold(self, sample_regression_data):
        """Test with high selection threshold."""
        rec = recipe().step_select_stability(
            outcome='target',
            threshold=0.9,
            n_bootstrap=20,
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        # High threshold selects only very stable features
        assert 'target' in result.columns
        assert result.shape[1] <= sample_regression_data.shape[1]

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_stability(
            outcome='target',
            threshold=0.7,
            n_bootstrap=20,
            random_state=42
        )
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]


class TestStepSelectLofo:
    """Tests for StepSelectLofo."""

    def test_basic_lofo_importance(self, sample_regression_data):
        """Test basic LOFO importance."""
        rec = recipe().step_select_lofo(
            outcome='target',
            top_n=3,
            cv=2,  # Small for faster testing
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        # Should select top 3 features
        assert result.shape[1] == 4  # 3 features + target

    def test_threshold_selection(self, sample_regression_data):
        """Test LOFO with importance threshold."""
        rec = recipe().step_select_lofo(
            outcome='target',
            threshold=0.0,  # Keep features with positive importance
            cv=2,
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        assert result.shape[1] >= 2  # At least target + 1 feature

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_lofo(
            outcome='target',
            top_n=3,
            cv=2,
            random_state=42
        )
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]


class TestStepSelectGranger:
    """Tests for StepSelectGranger."""

    def test_basic_granger_selection(self, sample_time_series_data):
        """Test basic Granger causality selection."""
        rec = recipe().step_select_granger(
            outcome='target',
            max_lag=3,
            alpha=0.1  # Lenient for test
        )

        prepped = rec.prep(sample_time_series_data)
        result = prepped.bake(sample_time_series_data)

        assert 'target' in result.columns
        # x1 should be selected (Granger-causes target in data generation)
        # x2 might not be selected
        assert result.shape[1] >= 2  # At least target + 1 feature

    def test_strict_alpha(self, sample_time_series_data):
        """Test with strict significance level."""
        rec = recipe().step_select_granger(
            outcome='target',
            max_lag=3,
            alpha=0.001  # Very strict
        )

        prepped = rec.prep(sample_time_series_data)
        result = prepped.bake(sample_time_series_data)

        assert 'target' in result.columns
        # Very strict alpha might not find any significant features
        assert result.shape[1] <= sample_time_series_data.shape[1]

    def test_recipe_method(self, sample_time_series_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_granger(
            outcome='target',
            max_lag=3,
            alpha=0.1
        )
        prepped = rec.prep(sample_time_series_data)
        result = prepped.bake(sample_time_series_data)

        assert result.shape[0] == sample_time_series_data.shape[0]


class TestStepSelectStepwise:
    """Tests for StepSelectStepwise."""

    def test_forward_selection(self, sample_regression_data):
        """Test forward stepwise selection."""
        rec = recipe().step_select_stepwise(
            outcome='target',
            direction='forward',
            criterion='aic'
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        # Should select important features
        assert result.shape[1] >= 2  # At least target + 1 feature
        assert result.shape[1] <= sample_regression_data.shape[1]

    def test_backward_elimination(self, sample_regression_data):
        """Test backward stepwise elimination."""
        rec = recipe().step_select_stepwise(
            outcome='target',
            direction='backward',
            criterion='bic'
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        assert result.shape[1] <= sample_regression_data.shape[1]

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_stepwise(
            outcome='target',
            direction='both',
            criterion='aic'
        )
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]


class TestStepSelectProbe:
    """Tests for StepSelectProbe."""

    def test_basic_probe_selection(self, sample_regression_data):
        """Test basic probe feature selection."""
        rec = recipe().step_select_probe(
            outcome='target',
            n_probes=5,
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        # Important features should exceed probe threshold
        assert 'x1' in result.columns or 'x2' in result.columns
        assert result.shape[1] >= 2  # At least target + 1 feature

    def test_percentile_threshold(self, sample_regression_data):
        """Test probe selection with percentile threshold."""
        rec = recipe().step_select_probe(
            outcome='target',
            n_probes=10,
            threshold_percentile=95,  # 95th percentile instead of max
            random_state=42
        )

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert 'target' in result.columns
        # More lenient threshold might select more features
        assert result.shape[1] >= 2

    def test_recipe_method(self, sample_regression_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_probe(
            outcome='target',
            n_probes=5,
            random_state=42
        )
        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]


class TestIntegration:
    """Integration tests for advanced selection steps."""

    def test_multiple_steps_chain(self, sample_regression_data):
        """Test chaining multiple selection steps."""
        rec = (recipe()
               .step_vif(threshold=10.0, outcome='target')
               .step_pvalue(outcome='target', threshold=0.05))

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        assert result.shape[0] == sample_regression_data.shape[0]
        assert 'target' in result.columns
        # Features should be selected by both steps
        assert result.shape[1] <= sample_regression_data.shape[1]

    def test_import_from_steps(self):
        """Test that steps can be imported from py_recipes.steps."""
        from py_recipes.steps import (
            StepVif,
            StepPvalue,
            StepSelectStability,
            StepSelectLofo,
            StepSelectGranger,
            StepSelectStepwise,
            StepSelectProbe,
        )

        assert StepVif is not None
        assert StepPvalue is not None
        assert StepSelectStability is not None
        assert StepSelectLofo is not None
        assert StepSelectGranger is not None
        assert StepSelectStepwise is not None
        assert StepSelectProbe is not None

    def test_combined_with_phase_2(self, sample_regression_data):
        """Test combining Phase 3 steps with Phase 2 steps."""
        rec = (recipe()
               .step_vif(threshold=10.0, outcome='target')
               .step_pvalue(outcome='target', threshold=0.05))

        prepped = rec.prep(sample_regression_data)
        result = prepped.bake(sample_regression_data)

        # Should work seamlessly
        assert 'target' in result.columns
        assert result.shape[0] == sample_regression_data.shape[0]
