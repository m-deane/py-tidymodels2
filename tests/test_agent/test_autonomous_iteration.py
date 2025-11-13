"""
Tests for autonomous iteration system (Phase 3.5).

Tests autonomous workflow improvement through:
- Try-evaluate-improve loops
- Self-debugging and issue detection
- Automatic retry with different approaches
- Performance-based stopping criteria
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from py_agent.tools.autonomous_iteration import (
    IterationResult,
    IterationLoop,
    iterate_until_target
)


class TestIterationResult:
    """Test IterationResult dataclass."""

    def test_create_iteration_result(self):
        """Test creating iteration result."""
        result = IterationResult(
            iteration_num=1,
            workflow=Mock(),
            fit=Mock(),
            performance={'rmse': 10.5, 'mae': 8.2, 'r_squared': 0.85},
            issues=[],
            approach="default_recommendation",
            success=True,
            error=None,
            duration=5.2
        )

        assert result.iteration_num == 1
        assert result.success is True
        assert result.performance['rmse'] == 10.5
        assert result.approach == "default_recommendation"

    def test_failed_iteration_result(self):
        """Test failed iteration result."""
        result = IterationResult(
            iteration_num=2,
            workflow=None,
            fit=None,
            performance={},
            issues=[],
            approach="retry_with_simpler_model",
            success=False,
            error="Model fitting failed",
            duration=1.0
        )

        assert result.success is False
        assert result.error == "Model fitting failed"
        assert result.fit is None


class TestIterationLoop:
    """Test IterationLoop class."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock ForecastAgent."""
        agent = Mock()
        agent.verbose = False
        return agent

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        return pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100) * 10 + 100,
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })

    def test_create_iteration_loop(self, mock_agent):
        """Test creating iteration loop."""
        loop = IterationLoop(
            agent=mock_agent,
            max_iterations=5,
            target_metric='rmse',
            target_value=10.0
        )

        assert loop.agent == mock_agent
        assert loop.max_iterations == 5
        assert loop.target_metric == 'rmse'
        assert loop.target_value == 10.0
        assert len(loop.history) == 0

    def test_determine_approach_first_iteration(self, mock_agent):
        """Test approach determination for first iteration."""
        loop = IterationLoop(agent=mock_agent)

        approach = loop._determine_approach(1, [])

        assert approach == "default_recommendation"

    def test_determine_approach_after_failure(self, mock_agent):
        """Test approach determination after failure."""
        loop = IterationLoop(agent=mock_agent)

        # Create failed result
        failed_result = IterationResult(
            iteration_num=1,
            workflow=None,
            fit=None,
            performance={},
            issues=[],
            approach="default_recommendation",
            success=False,
            error="Fitting failed",
            duration=1.0
        )

        approach = loop._determine_approach(2, [failed_result])

        assert approach == "retry_with_simpler_model"

    def test_determine_approach_after_overfitting(self, mock_agent):
        """Test approach determination after overfitting detected."""
        loop = IterationLoop(agent=mock_agent)

        # Create result with overfitting
        overfit_result = IterationResult(
            iteration_num=1,
            workflow=Mock(),
            fit=Mock(),
            performance={'rmse': 15.0, 'mae': 12.0, 'r_squared': 0.75},
            issues=[{'type': 'overfitting', 'evidence': 'Train RMSE much lower than test'}],
            approach="default_recommendation",
            success=True,
            error=None,
            duration=5.0
        )

        approach = loop._determine_approach(2, [overfit_result])

        assert approach == "regularization_or_simpler_model"

    def test_determine_approach_after_underfitting(self, mock_agent):
        """Test approach determination after underfitting."""
        loop = IterationLoop(agent=mock_agent)

        # Create result with underfitting
        underfit_result = IterationResult(
            iteration_num=1,
            workflow=Mock(),
            fit=Mock(),
            performance={'rmse': 25.0, 'mae': 20.0, 'r_squared': 0.35},
            issues=[{'type': 'underfitting', 'evidence': 'High training error'}],
            approach="default_recommendation",
            success=True,
            error=None,
            duration=5.0
        )

        approach = loop._determine_approach(2, [underfit_result])

        assert approach == "more_complex_model_or_features"

    def test_is_better_performance_rmse(self, mock_agent):
        """Test performance comparison for RMSE."""
        loop = IterationLoop(agent=mock_agent, target_metric='rmse')

        # Lower RMSE is better
        assert loop._is_better_performance(10.0, 15.0) is True
        assert loop._is_better_performance(15.0, 10.0) is False
        assert loop._is_better_performance(10.0, 10.0) is False

    def test_is_better_performance_r_squared(self, mock_agent):
        """Test performance comparison for R²."""
        loop = IterationLoop(agent=mock_agent, target_metric='r_squared')

        # Higher R² is better
        assert loop._is_better_performance(0.85, 0.75) is True
        assert loop._is_better_performance(0.75, 0.85) is False
        assert loop._is_better_performance(0.85, 0.85) is False

    def test_should_stop_target_reached_rmse(self, mock_agent):
        """Test stopping when target RMSE reached."""
        loop = IterationLoop(
            agent=mock_agent,
            target_metric='rmse',
            target_value=10.0
        )

        # RMSE below target should stop
        should_stop = loop._should_stop(
            current_metric=9.5,
            best_metric=9.5,
            iteration=2
        )

        assert should_stop is True

    def test_should_stop_target_not_reached(self, mock_agent):
        """Test not stopping when target not reached."""
        loop = IterationLoop(
            agent=mock_agent,
            target_metric='rmse',
            target_value=10.0
        )

        # RMSE above target should not stop
        should_stop = loop._should_stop(
            current_metric=12.0,
            best_metric=12.0,
            iteration=2
        )

        assert should_stop is False

    def test_should_stop_no_improvement(self, mock_agent):
        """Test stopping when no improvement."""
        loop = IterationLoop(
            agent=mock_agent,
            target_metric='rmse',
            improvement_threshold=0.05
        )

        # Create history with no improvement
        loop.history = [
            IterationResult(
                iteration_num=1,
                workflow=Mock(),
                fit=Mock(),
                performance={'rmse': 15.0},
                issues=[],
                approach="default",
                success=True,
                error=None,
                duration=5.0
            ),
            IterationResult(
                iteration_num=2,
                workflow=Mock(),
                fit=Mock(),
                performance={'rmse': 14.9},  # < 5% improvement
                issues=[],
                approach="retry",
                success=True,
                error=None,
                duration=5.0
            )
        ]

        should_stop = loop._should_stop(
            current_metric=14.9,
            best_metric=14.9,
            iteration=3
        )

        # With less than 5% improvement, should stop
        assert should_stop is True

    def test_get_best_metric_from_history_rmse(self, mock_agent):
        """Test getting best RMSE from history."""
        loop = IterationLoop(agent=mock_agent, target_metric='rmse')

        history = [
            IterationResult(
                iteration_num=1,
                workflow=Mock(),
                fit=Mock(),
                performance={'rmse': 15.0},
                issues=[],
                approach="default",
                success=True,
                error=None,
                duration=5.0
            ),
            IterationResult(
                iteration_num=2,
                workflow=Mock(),
                fit=Mock(),
                performance={'rmse': 12.0},
                issues=[],
                approach="retry",
                success=True,
                error=None,
                duration=5.0
            )
        ]

        best = loop._get_best_metric_from_history(history)

        # Best RMSE is lowest
        assert best == 12.0

    def test_get_best_metric_from_history_r_squared(self, mock_agent):
        """Test getting best R² from history."""
        loop = IterationLoop(agent=mock_agent, target_metric='r_squared')

        history = [
            IterationResult(
                iteration_num=1,
                workflow=Mock(),
                fit=Mock(),
                performance={'r_squared': 0.75},
                issues=[],
                approach="default",
                success=True,
                error=None,
                duration=5.0
            ),
            IterationResult(
                iteration_num=2,
                workflow=Mock(),
                fit=Mock(),
                performance={'r_squared': 0.85},
                issues=[],
                approach="retry",
                success=True,
                error=None,
                duration=5.0
            )
        ]

        best = loop._get_best_metric_from_history(history)

        # Best R² is highest
        assert best == 0.85

    def test_get_best_metric_no_successful_iterations(self, mock_agent):
        """Test getting best metric when no successful iterations."""
        loop = IterationLoop(agent=mock_agent)

        history = [
            IterationResult(
                iteration_num=1,
                workflow=None,
                fit=None,
                performance={},
                issues=[],
                approach="default",
                success=False,
                error="Failed",
                duration=1.0
            )
        ]

        best = loop._get_best_metric_from_history(history)

        assert best is None


class TestIterationIntegration:
    """Integration tests for iteration loop."""

    @pytest.fixture
    def sample_data(self):
        """Create realistic sample data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        # Generate data with trend and seasonality
        t = np.arange(200)
        trend = 100 + 0.5 * t
        seasonality = 20 * np.sin(2 * np.pi * t / 7)
        noise = np.random.randn(200) * 5

        values = trend + seasonality + noise

        return pd.DataFrame({
            'date': dates,
            'sales': values,
            'temperature': np.random.randn(200) * 10 + 20,
            'promotion': np.random.choice([0, 1], 200)
        })

    @pytest.fixture
    def split_data(self, sample_data):
        """Split data into train/test."""
        train = sample_data.iloc[:150]
        test = sample_data.iloc[150:]
        return train, test

    @patch('py_agent.tools.autonomous_iteration.diagnose_performance')
    def test_iteration_loop_basic(self, mock_diagnose, split_data):
        """Test basic iteration loop execution."""
        train, test = split_data

        # Mock diagnose_performance to return no issues
        mock_diagnose.return_value = {'issues_detected': []}

        # Create mock agent
        agent = Mock()
        agent.verbose = False

        # Create mock workflow and fit
        mock_workflow = Mock()
        mock_fit = Mock()

        # Mock extract_outputs to return stats
        stats_df = pd.DataFrame({
            'split': ['test'],
            'rmse': [12.5],
            'mae': [10.0],
            'r_squared': [0.82]
        })
        mock_fit.extract_outputs.return_value = (Mock(), Mock(), stats_df)
        mock_fit.evaluate.return_value = mock_fit

        mock_workflow.fit.return_value = mock_fit
        agent.generate_workflow.return_value = mock_workflow

        # Create iteration loop
        loop = IterationLoop(
            agent=agent,
            max_iterations=3,
            target_metric='rmse',
            target_value=None,
            verbose=False
        )

        # Run iteration
        best_fit, history = loop.iterate(
            data=train,
            request="Forecast sales",
            test_data=test
        )

        # Check results
        assert best_fit is not None
        assert len(history) > 0
        assert history[0].success is True
        assert history[0].performance['rmse'] == 12.5

        # Agent should have been called
        agent.generate_workflow.assert_called()

    @patch('py_agent.tools.autonomous_iteration.diagnose_performance')
    def test_iteration_stops_at_target(self, mock_diagnose, split_data):
        """Test iteration stops when target reached."""
        train, test = split_data

        mock_diagnose.return_value = {'issues_detected': []}

        # Create agent
        agent = Mock()
        agent.verbose = False

        # First iteration: RMSE = 15.0
        # Second iteration: RMSE = 9.0 (below target)
        call_count = [0]

        def generate_workflow_side_effect(*args, **kwargs):
            call_count[0] += 1

            mock_workflow = Mock()
            mock_fit = Mock()

            rmse = 15.0 if call_count[0] == 1 else 9.0

            stats_df = pd.DataFrame({
                'split': ['test'],
                'rmse': [rmse],
                'mae': [rmse * 0.8],
                'r_squared': [0.75 if call_count[0] == 1 else 0.88]
            })
            mock_fit.extract_outputs.return_value = (Mock(), Mock(), stats_df)
            mock_fit.evaluate.return_value = mock_fit

            mock_workflow.fit.return_value = mock_fit
            return mock_workflow

        agent.generate_workflow.side_effect = generate_workflow_side_effect

        # Create loop with target
        loop = IterationLoop(
            agent=agent,
            max_iterations=5,
            target_metric='rmse',
            target_value=10.0,
            verbose=False
        )

        # Run iteration
        best_fit, history = loop.iterate(
            data=train,
            request="Forecast sales",
            test_data=test
        )

        # Should stop after 2 iterations (target reached)
        assert len(history) == 2
        assert history[-1].performance['rmse'] == 9.0
        assert agent.generate_workflow.call_count == 2


class TestIterateUntilTarget:
    """Test iterate_until_target convenience function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        return pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100) * 10 + 100
        })

    @patch('py_agent.tools.autonomous_iteration.IterationLoop')
    def test_iterate_until_target_function(self, mock_loop_class, sample_data):
        """Test iterate_until_target convenience function."""
        # Mock IterationLoop
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop

        mock_loop.iterate.return_value = (Mock(), [])

        # Mock agent
        agent = Mock()

        # Call convenience function
        best_wf, history = iterate_until_target(
            agent=agent,
            data=sample_data,
            request="Forecast values",
            target_metric='rmse',
            target_value=10.0,
            max_iterations=3
        )

        # Check loop was created correctly
        mock_loop_class.assert_called_once()
        call_kwargs = mock_loop_class.call_args[1]
        assert call_kwargs['agent'] == agent
        assert call_kwargs['max_iterations'] == 3
        assert call_kwargs['target_metric'] == 'rmse'
        assert call_kwargs['target_value'] == 10.0

        # Check iterate was called
        mock_loop.iterate.assert_called_once()


class TestForecastAgentIntegration:
    """Test integration with ForecastAgent.iterate() method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        return pd.DataFrame({
            'date': dates,
            'sales': np.random.randn(100) * 10 + 100,
            'feature1': np.random.randn(100)
        })

    def test_agent_iterate_method_exists(self):
        """Test that ForecastAgent has iterate() method."""
        from py_agent import ForecastAgent

        agent = ForecastAgent(verbose=False)

        assert hasattr(agent, 'iterate')
        assert callable(agent.iterate)

    @patch('py_agent.tools.autonomous_iteration.IterationLoop')
    def test_agent_iterate_creates_loop(self, mock_loop_class, sample_data):
        """Test that agent.iterate() creates IterationLoop."""
        from py_agent import ForecastAgent

        # Mock IterationLoop
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        mock_loop.iterate.return_value = (Mock(), [])

        # Create agent
        agent = ForecastAgent(verbose=False)

        # Call iterate
        agent.iterate(
            data=sample_data,
            request="Forecast sales",
            target_metric='rmse',
            target_value=10.0,
            max_iterations=3
        )

        # Check IterationLoop was created
        mock_loop_class.assert_called_once()
        call_kwargs = mock_loop_class.call_args[1]
        assert call_kwargs['agent'] == agent
        assert call_kwargs['max_iterations'] == 3
        assert call_kwargs['target_metric'] == 'rmse'
        assert call_kwargs['target_value'] == 10.0
        assert call_kwargs['verbose'] == agent.verbose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
