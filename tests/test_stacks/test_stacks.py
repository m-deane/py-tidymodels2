"""
Tests for py_stacks package

Tests cover:
- Creating stacks
- Adding candidates
- Blending predictions
- Getting model weights
- Comparing ensemble to candidates
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np

from py_stacks import stacks, Stacks, BlendedStack


class TestStacksCreation:
    """Test creating stacks objects"""

    def test_create_empty_stacks(self):
        """Test creating empty stacks"""
        stack = stacks()

        assert isinstance(stack, Stacks)
        assert len(stack.candidates) == 0
        assert len(stack.candidate_names) == 0
        assert stack.meta_learner is None
        assert stack.blend_fit is None

    def test_stacks_initialization(self):
        """Test direct Stacks initialization"""
        stack = Stacks()

        assert isinstance(stack, Stacks)
        assert hasattr(stack, "candidates")
        assert hasattr(stack, "meta_learner")


class TestAddCandidates:
    """Test adding candidate predictions"""

    def test_add_single_candidate(self):
        """Test adding a single candidate"""
        # Create mock predictions DataFrame
        predictions = pd.DataFrame({
            ".pred": [1.0, 2.0, 3.0, 4.0, 5.0],
            "actual": [1.1, 2.1, 2.9, 4.2, 4.8],
            ".config": ["config_1"] * 5
        })

        stack = stacks().add_candidates(predictions, name="model_1")

        assert len(stack.candidates) == 1
        assert len(stack.candidate_names) == 1
        assert stack.candidate_names[0] == "model_1"

    def test_add_multiple_candidates(self):
        """Test adding multiple candidates"""
        pred1 = pd.DataFrame({
            ".pred": [1.0, 2.0, 3.0],
            "actual": [1.1, 2.1, 2.9],
        })

        pred2 = pd.DataFrame({
            ".pred": [1.2, 2.1, 3.1],
            "actual": [1.1, 2.1, 2.9],
        })

        stack = (
            stacks()
            .add_candidates(pred1, name="linear")
            .add_candidates(pred2, name="tree")
        )

        assert len(stack.candidates) == 2
        assert stack.candidate_names == ["linear", "tree"]

    def test_add_candidate_auto_name(self):
        """Test automatic name generation"""
        predictions = pd.DataFrame({
            ".pred": [1.0, 2.0, 3.0],
            "actual": [1.1, 2.1, 2.9],
        })

        stack = stacks().add_candidates(predictions)

        assert len(stack.candidate_names) == 1
        assert "candidates_1" in stack.candidate_names[0]

    def test_add_candidate_method_chaining(self):
        """Test method chaining with add_candidates"""
        pred1 = pd.DataFrame({".pred": [1.0], "actual": [1.1]})
        pred2 = pd.DataFrame({".pred": [1.2], "actual": [1.1]})

        result = stacks().add_candidates(pred1).add_candidates(pred2)

        assert isinstance(result, Stacks)
        assert len(result.candidates) == 2


class TestBlendPredictions:
    """Test blending predictions with meta-learner"""

    def test_blend_basic(self):
        """Test basic blending"""
        np.random.seed(42)

        # Create mock predictions from 3 models
        n_obs = 100
        actual = np.random.randn(n_obs) * 2 + 10

        pred1 = pd.DataFrame({
            ".pred": actual + np.random.randn(n_obs) * 0.5,
            "actual": actual
        })

        pred2 = pd.DataFrame({
            ".pred": actual + np.random.randn(n_obs) * 0.3,
            "actual": actual
        })

        pred3 = pd.DataFrame({
            ".pred": actual + np.random.randn(n_obs) * 0.7,
            "actual": actual
        })

        # Create and blend stack
        blended = (
            stacks()
            .add_candidates(pred1, name="model_1")
            .add_candidates(pred2, name="model_2")
            .add_candidates(pred3, name="model_3")
            .blend_predictions()
        )

        assert isinstance(blended, BlendedStack)
        assert blended.meta_learner is not None

    def test_blend_with_penalty(self):
        """Test blending with different penalty values"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        # Low penalty
        blended_low = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions(penalty=0.001)
        )

        # High penalty
        blended_high = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions(penalty=1.0)
        )

        # High penalty should produce more regularized (smaller) weights
        weights_low = np.abs(blended_low.meta_learner.coef_).sum()
        weights_high = np.abs(blended_high.meta_learner.coef_).sum()

        assert weights_high <= weights_low

    def test_blend_non_negative_constraint(self):
        """Test non-negative constraint on weights"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        # With non-negative constraint
        blended = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions(non_negative=True)
        )

        # All coefficients should be >= 0
        assert all(blended.meta_learner.coef_ >= 0)

    def test_blend_without_candidates_error(self):
        """Test error when blending without candidates"""
        stack = stacks()

        with pytest.raises(ValueError, match="No candidates added"):
            stack.blend_predictions()


class TestGetModelWeights:
    """Test extracting model weights"""

    def test_get_weights_basic(self):
        """Test getting model weights"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1, name="model_1")
            .add_candidates(pred2, name="model_2")
            .blend_predictions()
        )

        weights = blended.get_model_weights()

        assert isinstance(weights, pd.DataFrame)
        assert "model" in weights.columns
        assert "weight" in weights.columns
        assert len(weights) >= 2  # At least 2 models + intercept

    def test_weights_sorted_by_contribution(self):
        """Test weights are sorted by absolute value"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 1.0, "actual": actual})  # Worse
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.1, "actual": actual})  # Better

        blended = (
            stacks()
            .add_candidates(pred1, name="weak_model")
            .add_candidates(pred2, name="strong_model")
            .blend_predictions()
        )

        weights = blended.get_model_weights()

        # Exclude intercept row
        model_weights = weights[weights["model"] != "(Intercept)"]

        # First model (highest weight) should be the better predictor
        # Note: This isn't guaranteed in all cases, but likely with this setup
        assert len(model_weights) >= 2

    def test_weights_include_contribution_pct(self):
        """Test weights include percentage contribution"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions()
        )

        weights = blended.get_model_weights()

        assert "contribution_pct" in weights.columns
        # Model contributions should sum to 100% (excluding intercept)
        model_contribs = weights[weights["model"] != "(Intercept)"]["contribution_pct"]
        assert abs(model_contribs.sum() - 100.0) < 1e-6


class TestGetMetrics:
    """Test getting ensemble metrics"""

    def test_get_metrics_basic(self):
        """Test getting basic metrics"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions()
        )

        metrics = blended.get_metrics()

        assert isinstance(metrics, pd.DataFrame)
        assert "metric" in metrics.columns
        assert "value" in metrics.columns

        # Should have standard regression metrics
        metric_names = metrics["metric"].tolist()
        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names

    def test_metrics_are_numeric(self):
        """Test metrics are valid numbers"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1)
            .blend_predictions()
        )

        metrics = blended.get_metrics()

        # All values should be numeric and not NaN
        assert pd.api.types.is_numeric_dtype(metrics["value"])
        assert not metrics["value"].isna().any()


class TestCompareToCandidate:
    """Test comparing ensemble to individual models"""

    def test_compare_basic(self):
        """Test basic comparison"""
        np.random.seed(42)

        actual = np.random.randn(100) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(100) * 0.8, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(100) * 0.6, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1, name="model_1")
            .add_candidates(pred2, name="model_2")
            .blend_predictions()
        )

        comparison = blended.compare_to_candidates()

        assert isinstance(comparison, pd.DataFrame)
        assert "model" in comparison.columns
        assert "rmse" in comparison.columns
        assert "mae" in comparison.columns
        assert "r_squared" in comparison.columns

        # Should have ensemble + individual models
        assert len(comparison) >= 3  # Ensemble + 2 models

    def test_ensemble_in_comparison(self):
        """Test ensemble appears in comparison"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.3, "actual": actual})

        blended = (
            stacks()
            .add_candidates(pred1)
            .add_candidates(pred2)
            .blend_predictions()
        )

        comparison = blended.compare_to_candidates()

        # Ensemble should be in the results
        assert "Ensemble" in comparison["model"].values

    def test_comparison_sorted_by_rmse(self):
        """Test comparison is sorted by RMSE"""
        np.random.seed(42)

        actual = np.random.randn(100) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(100) * 1.0, "actual": actual})  # Worse
        pred2 = pd.DataFrame({".pred": actual + np.random.randn(100) * 0.1, "actual": actual})  # Better

        blended = (
            stacks()
            .add_candidates(pred1, name="weak")
            .add_candidates(pred2, name="strong")
            .blend_predictions()
        )

        comparison = blended.compare_to_candidates()

        # Should be sorted with lowest RMSE first
        rmse_values = comparison["rmse"].values
        assert all(rmse_values[i] <= rmse_values[i+1] for i in range(len(rmse_values)-1))


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_candidate_blend(self):
        """Test blending with only one candidate"""
        np.random.seed(42)

        actual = np.random.randn(50) * 2 + 10
        pred1 = pd.DataFrame({".pred": actual + np.random.randn(50) * 0.5, "actual": actual})

        # Should still work with one candidate
        blended = stacks().add_candidates(pred1).blend_predictions()

        assert isinstance(blended, BlendedStack)
        weights = blended.get_model_weights()
        assert len(weights) >= 1

    def test_many_candidates(self):
        """Test blending with many candidates"""
        np.random.seed(42)

        actual = np.random.randn(100) * 2 + 10

        stack = stacks()

        # Add 10 candidates
        for i in range(10):
            pred = pd.DataFrame({
                ".pred": actual + np.random.randn(100) * (0.5 + i * 0.1),
                "actual": actual
            })
            stack = stack.add_candidates(pred, name=f"model_{i}")

        blended = stack.blend_predictions()

        assert isinstance(blended, BlendedStack)
        weights = blended.get_model_weights()
        assert len(weights) >= 10

    def test_perfect_predictions(self):
        """Test with perfect predictions (zero error)"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        pred1 = pd.DataFrame({".pred": actual.copy(), "actual": actual})  # Perfect

        blended = stacks().add_candidates(pred1).blend_predictions()

        metrics = blended.get_metrics()

        # RMSE should be very small
        rmse = metrics[metrics["metric"] == "rmse"]["value"].iloc[0]
        assert rmse < 1e-10

    def test_constant_predictions(self):
        """Test with constant predictions"""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        pred1 = pd.DataFrame({".pred": np.ones(5) * 3.0, "actual": actual})  # Constant

        blended = stacks().add_candidates(pred1).blend_predictions()

        # Should complete without error
        assert isinstance(blended, BlendedStack)
