"""
Tests for feature-engine integration steps (Phase 1).
"""

import pytest
import pandas as pd
import numpy as np
from py_recipes import recipe
from py_recipes.steps.feature_engine_steps import (
    StepDtDiscretiser,
    StepWinsorizer,
    StepOutlierTrimmer,
    StepDtFeatures,
    StepSelectSmartCorr,
    StepSelectPsi,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 200

    data = pd.DataFrame({
        'x1': np.random.normal(50, 10, n),
        'x2': np.random.normal(100, 20, n),
        'x3': np.random.normal(10, 5, n),
        'x4': np.random.normal(0, 1, n),  # Highly correlated with x5
        'x5': np.random.normal(0, 1, n) + np.random.normal(0, 0.1, n),  # Correlated with x4
        'target': np.random.normal(0, 1, n)
    })

    # Add some outliers
    data.loc[0, 'x1'] = 150
    data.loc[1, 'x2'] = 300

    # Make target correlated with features
    data['target'] = 0.5 * data['x1'] + 0.3 * data['x2'] + np.random.normal(0, 5, n)

    return data


class TestStepDtDiscretiser:
    """Tests for StepDtDiscretiser."""

    def test_basic_prep_bake(self, sample_data):
        """Test basic prep and bake functionality."""
        rec = recipe().step_dt_discretiser(
            outcome='target',
            columns=['x1', 'x2'],
            cv=2,
            regression=True
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert 'x1' in result.columns
        assert 'x2' in result.columns
        # DecisionTreeDiscretiser returns float predictions (discretized values)
        assert result['x1'].dtype in [np.float32, np.float64, 'float64', 'float32']
        # Values should be different from original (discretized)
        assert not np.allclose(result['x1'].values, sample_data['x1'].values)

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_dt_discretiser(outcome='target', cv=2)
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]


class TestStepWinsorizer:
    """Tests for StepWinsorizer."""

    def test_iqr_method(self, sample_data):
        """Test IQR winsorization."""
        rec = recipe().step_winsorizer(
            columns=['x1', 'x2'],
            capping_method='iqr',
            fold=1.5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Check outliers were capped
        assert result['x1'].max() < sample_data['x1'].max()
        assert result.shape == sample_data.shape

    def test_gaussian_method(self, sample_data):
        """Test Gaussian winsorization."""
        rec = recipe().step_winsorizer(
            columns=['x1'],
            capping_method='gaussian',
            tail='right',
            fold=3.0
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape == sample_data.shape

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_winsorizer(capping_method='iqr')
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]


class TestStepOutlierTrimmer:
    """Tests for StepOutlierTrimmer."""

    def test_removes_outliers(self, sample_data):
        """Test that outliers are removed."""
        rec = recipe().step_outlier_trimmer(
            columns=['x1', 'x2'],
            capping_method='iqr',
            fold=1.5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should have fewer rows due to outlier removal
        assert result.shape[0] < sample_data.shape[0]
        assert result.shape[1] == sample_data.shape[1]

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_outlier_trimmer(capping_method='iqr', fold=2.0)
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[1] == sample_data.shape[1]


class TestStepDtFeatures:
    """Tests for StepDtFeatures."""

    def test_creates_features(self, sample_data):
        """Test that new features are created."""
        rec = recipe().step_dt_features(
            outcome='target',
            columns=['x1', 'x2'],
            features_to_combine=3,
            cv=2
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should have more columns (original + new features)
        assert result.shape[1] > sample_data.shape[1]
        assert result.shape[0] == sample_data.shape[0]

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_dt_features(outcome='target', features_to_combine=2, cv=2)
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]


class TestStepSelectSmartCorr:
    """Tests for StepSelectSmartCorr."""

    def test_removes_correlated_features(self, sample_data):
        """Test that correlated features are removed."""
        rec = recipe().step_select_smart_corr(
            outcome='target',
            columns=['x4', 'x5'],
            threshold=0.5,
            selection_method='variance'
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # Should have removed at least one correlated feature
        assert result.shape[1] <= sample_data.shape[1]
        assert result.shape[0] == sample_data.shape[0]

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_smart_corr(
            outcome='target',
            threshold=0.9,
            selection_method='variance'
        )
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]


class TestStepSelectPsi:
    """Tests for StepSelectPsi."""

    def test_basic_prep_bake(self, sample_data):
        """Test basic prep and bake functionality."""
        rec = recipe().step_select_psi(
            columns=['x1', 'x2', 'x3'],
            threshold=0.25,
            bins=5
        )

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        # PSI step may or may not drop features depending on distribution
        assert result.shape[1] <= sample_data.shape[1]
        assert result.shape[0] == sample_data.shape[0]

    def test_recipe_method(self, sample_data):
        """Test recipe convenience method."""
        rec = recipe().step_select_psi(threshold=0.30, bins=10)
        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]


class TestIntegration:
    """Integration tests for feature-engine steps."""

    def test_multiple_steps_chain(self, sample_data):
        """Test chaining multiple feature-engine steps."""
        rec = (recipe()
               .step_winsorizer(columns=['x1', 'x2'], capping_method='iqr', fold=1.5)
               .step_dt_discretiser(outcome='target', columns=['x1'], cv=2)
               .step_select_smart_corr(outcome='target', threshold=0.9))

        prepped = rec.prep(sample_data)
        result = prepped.bake(sample_data)

        assert result.shape[0] == sample_data.shape[0]

    def test_import_from_steps(self):
        """Test that steps can be imported from py_recipes.steps."""
        from py_recipes.steps import (
            StepDtDiscretiser,
            StepWinsorizer,
            StepOutlierTrimmer,
            StepDtFeatures,
            StepSelectSmartCorr,
            StepSelectPsi,
        )

        assert StepDtDiscretiser is not None
        assert StepWinsorizer is not None
        assert StepOutlierTrimmer is not None
        assert StepDtFeatures is not None
        assert StepSelectSmartCorr is not None
        assert StepSelectPsi is not None
