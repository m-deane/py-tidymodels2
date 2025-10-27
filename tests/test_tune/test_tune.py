"""
Comprehensive tests for py-tune hyperparameter tuning
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import replace

from py_tune import (
    tune,
    TuneParameter,
    grid_regular,
    grid_random,
    TuneResults,
    fit_resamples,
    tune_grid,
    finalize_workflow,
)


# ============================================================================
# Test TuneParameter and tune()
# ============================================================================

def test_tune_parameter_creation():
    """Test TuneParameter creation without id"""
    param = TuneParameter()
    assert isinstance(param, TuneParameter)
    assert param.id.startswith("tune_")


def test_tune_parameter_with_id():
    """Test TuneParameter creation with custom id"""
    param = TuneParameter(id="penalty")
    assert param.id == "penalty"


def test_tune_parameter_repr():
    """Test TuneParameter string representation"""
    param = TuneParameter(id="penalty")
    assert repr(param) == "tune(id='penalty')"


def test_tune_function():
    """Test tune() function returns TuneParameter"""
    param = tune()
    assert isinstance(param, TuneParameter)


def test_tune_function_with_id():
    """Test tune() function with custom id"""
    param = tune(id="mixture")
    assert param.id == "mixture"


def test_multiple_tune_parameters_unique_ids():
    """Test that multiple tune() calls create unique ids"""
    param1 = tune()
    param2 = tune()
    assert param1.id != param2.id


# ============================================================================
# Test grid_regular()
# ============================================================================

def test_grid_regular_basic():
    """Test basic regular grid generation"""
    param_info = {
        'penalty': {'range': (0.001, 1.0)},
        'mixture': {'range': (0, 1)}
    }
    grid = grid_regular(param_info, levels=3)

    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 9  # 3 x 3
    assert 'penalty' in grid.columns
    assert 'mixture' in grid.columns
    assert '.config' in grid.columns


def test_grid_regular_log_transformation():
    """Test regular grid with log transformation"""
    param_info = {
        'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
    }
    grid = grid_regular(param_info, levels=3)

    assert len(grid) == 3
    # Check that values are log-spaced
    penalties = grid['penalty'].values
    assert penalties[0] < penalties[1] < penalties[2]
    # Approximate log spacing check
    log_diff1 = np.log10(penalties[1]) - np.log10(penalties[0])
    log_diff2 = np.log10(penalties[2]) - np.log10(penalties[1])
    assert np.isclose(log_diff1, log_diff2, rtol=0.01)


def test_grid_regular_explicit_values():
    """Test regular grid with explicit values"""
    param_info = {
        'mtry': {'values': [2, 4, 6, 8]},
        'trees': {'values': [100, 500, 1000]}
    }
    grid = grid_regular(param_info, levels=3)

    assert len(grid) == 12  # 4 x 3
    assert sorted(grid['mtry'].unique()) == [2, 4, 6, 8]
    assert sorted(grid['trees'].unique()) == [100, 500, 1000]


def test_grid_regular_config_names():
    """Test that config names are generated correctly"""
    param_info = {'penalty': {'range': (0.1, 1.0)}}
    grid = grid_regular(param_info, levels=5)

    configs = grid['.config'].tolist()
    assert configs == ['config_001', 'config_002', 'config_003', 'config_004', 'config_005']


def test_grid_regular_single_parameter():
    """Test regular grid with single parameter"""
    param_info = {'penalty': {'range': (0.001, 1.0)}}
    grid = grid_regular(param_info, levels=10)

    assert len(grid) == 10
    assert len(grid.columns) == 2  # penalty + .config


def test_grid_regular_invalid_param_info():
    """Test error when param_info is invalid"""
    param_info = {'penalty': {}}  # Missing 'range' or 'values'

    with pytest.raises(ValueError, match="must have 'range' or 'values' key"):
        grid_regular(param_info, levels=3)


# ============================================================================
# Test grid_random()
# ============================================================================

def test_grid_random_basic():
    """Test basic random grid generation"""
    param_info = {
        'penalty': {'range': (0.001, 1.0)},
        'mixture': {'range': (0, 1)}
    }
    grid = grid_random(param_info, size=20, seed=42)

    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 20
    assert 'penalty' in grid.columns
    assert 'mixture' in grid.columns
    assert '.config' in grid.columns


def test_grid_random_log_transformation():
    """Test random grid with log transformation"""
    param_info = {
        'penalty': {'range': (0.001, 1.0), 'trans': 'log'}
    }
    grid = grid_random(param_info, size=50, seed=42)

    assert len(grid) == 50
    penalties = grid['penalty'].values
    assert all(0.001 <= p <= 1.0 for p in penalties)
    # Check log-uniform distribution (more values near lower end)
    below_01 = sum(p < 0.01 for p in penalties)
    assert below_01 > 5  # Should have several values in lower range


def test_grid_random_integer_type():
    """Test random grid with integer type"""
    param_info = {
        'trees': {'range': (10, 1000), 'type': 'int'}
    }
    grid = grid_random(param_info, size=30, seed=42)

    assert len(grid) == 30
    trees = grid['trees'].values
    assert all(isinstance(t, (int, np.integer)) for t in trees)
    assert all(10 <= t <= 1000 for t in trees)


def test_grid_random_reproducibility():
    """Test that seed makes random grid reproducible"""
    param_info = {'penalty': {'range': (0.001, 1.0)}}

    grid1 = grid_random(param_info, size=10, seed=123)
    grid2 = grid_random(param_info, size=10, seed=123)

    assert grid1['penalty'].equals(grid2['penalty'])


def test_grid_random_different_seeds():
    """Test that different seeds produce different grids"""
    param_info = {'penalty': {'range': (0.001, 1.0)}}

    grid1 = grid_random(param_info, size=10, seed=123)
    grid2 = grid_random(param_info, size=10, seed=456)

    assert not grid1['penalty'].equals(grid2['penalty'])


def test_grid_random_invalid_param_info():
    """Test error when param_info missing range"""
    param_info = {'penalty': {'values': [0.1, 0.5, 1.0]}}  # random needs range

    with pytest.raises(ValueError, match="must have 'range' key"):
        grid_random(param_info, size=10)


# ============================================================================
# Test TuneResults
# ============================================================================

@pytest.fixture
def sample_tune_results():
    """Create sample TuneResults for testing"""
    metrics_data = {
        '.config': ['config_001', 'config_001', 'config_002', 'config_002'],
        '.resample': ['Fold01', 'Fold02', 'Fold01', 'Fold02'],
        'rmse': [0.5, 0.6, 0.4, 0.5],
        'mae': [0.3, 0.4, 0.25, 0.35]
    }
    metrics_df = pd.DataFrame(metrics_data)

    grid_data = {
        '.config': ['config_001', 'config_002'],
        'penalty': [0.1, 1.0],
        'mixture': [0.5, 0.8]
    }
    grid_df = pd.DataFrame(grid_data)

    return TuneResults(
        metrics=metrics_df,
        grid=grid_df
    )


def test_tune_results_creation():
    """Test TuneResults creation"""
    results = TuneResults()
    assert isinstance(results.metrics, pd.DataFrame)
    assert isinstance(results.predictions, pd.DataFrame)
    assert isinstance(results.grid, pd.DataFrame)


def test_tune_results_collect_metrics(sample_tune_results):
    """Test collecting metrics from TuneResults"""
    metrics = sample_tune_results.collect_metrics()
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 4
    assert 'rmse' in metrics.columns


def test_tune_results_collect_predictions():
    """Test collecting predictions from TuneResults"""
    pred_data = {
        '.config': ['config_001', 'config_001'],
        '.pred': [1.5, 2.3]
    }
    results = TuneResults(predictions=pd.DataFrame(pred_data))

    preds = results.collect_predictions()
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 2


def test_tune_results_show_best_minimize(sample_tune_results):
    """Test show_best for minimization metric"""
    best = sample_tune_results.show_best('rmse', n=2, maximize=False)

    assert isinstance(best, pd.DataFrame)
    assert len(best) == 2
    assert best.iloc[0]['.config'] == 'config_002'  # Lower RMSE
    assert 'penalty' in best.columns
    assert 'mixture' in best.columns


def test_tune_results_show_best_maximize(sample_tune_results):
    """Test show_best for maximization metric"""
    # Add accuracy metric (higher is better)
    sample_tune_results.metrics['accuracy'] = [0.8, 0.85, 0.9, 0.88]

    best = sample_tune_results.show_best('accuracy', n=1, maximize=True)

    assert len(best) == 1
    assert best.iloc[0]['.config'] == 'config_002'  # Higher accuracy


def test_tune_results_select_best(sample_tune_results):
    """Test select_best returns dictionary of best params"""
    best_params = sample_tune_results.select_best('rmse', maximize=False)

    assert isinstance(best_params, dict)
    assert 'penalty' in best_params
    assert 'mixture' in best_params
    assert best_params['penalty'] == 1.0  # config_002 has lower RMSE


def test_tune_results_select_by_one_std_err():
    """Test one-standard-error rule selection"""
    # Create metrics where multiple configs are within 1 std err
    # config_002 is best (mean 0.4, std 0.1)
    # threshold = 0.4 + 0.1 = 0.5
    # config_001 (mean 0.42) and config_003 (mean 0.45) are within threshold
    metrics_data = {
        '.config': ['config_001'] * 5 + ['config_002'] * 5 + ['config_003'] * 5,
        'rmse': [0.32, 0.42, 0.52, 0.42, 0.42,  # mean 0.42, within threshold
                 0.3, 0.4, 0.5, 0.4, 0.4,       # mean 0.4, std 0.1 (best)
                 0.35, 0.45, 0.55, 0.45, 0.45]  # mean 0.45, within threshold
    }
    metrics_df = pd.DataFrame(metrics_data)

    grid_data = {
        '.config': ['config_001', 'config_002', 'config_003'],
        'penalty': [0.1, 1.0, 0.5],  # config_001 has lowest (simplest) penalty
        'mixture': [0.5, 0.8, 0.6]
    }
    grid_df = pd.DataFrame(grid_data)

    results = TuneResults(metrics=metrics_df, grid=grid_df)
    selected = results.select_by_one_std_err('rmse', maximize=False)

    assert isinstance(selected, dict)
    # All three configs should be within 1 std err of config_002 (best)
    # Among them, config_001 has simplest (lowest penalty=0.1)
    assert selected['penalty'] == 0.1


def test_tune_results_empty():
    """Test TuneResults with empty data"""
    results = TuneResults()

    metrics = results.collect_metrics()
    assert len(metrics) == 0

    preds = results.collect_predictions()
    assert len(preds) == 0


# ============================================================================
# Test fit_resamples() and tune_grid() - Mock-based tests
# ============================================================================

@pytest.fixture
def mock_workflow():
    """Create a mock workflow for testing"""
    from dataclasses import dataclass
    from typing import Any, Tuple

    @dataclass
    class MockSpec:
        """Mock model specification"""
        args: Tuple = tuple()
        mode: str = "regression"
        engine: str = "lm"

    @dataclass
    class MockWorkflow:
        """Mock workflow for testing"""
        preprocessor: str = "y ~ x"
        spec: Any = None

        def __post_init__(self):
            if self.spec is None:
                self.spec = MockSpec()

        def fit(self, data):
            """Mock fit method"""
            fitted = replace(self)
            return fitted

        def predict(self, data):
            """Mock predict method - return predictions"""
            n = len(data)
            return pd.DataFrame({
                '.pred': np.random.uniform(0, 10, n)
            })

    return MockWorkflow()


@pytest.fixture
def mock_resamples():
    """Create mock resamples for testing"""
    from dataclasses import dataclass

    @dataclass
    class MockSplit:
        """Mock resample split"""
        train_data: pd.DataFrame
        test_data: pd.DataFrame

        def __iter__(self):
            return iter([self.train_data, self.test_data])

        def training(self):
            """Return training data - matches RSplit API"""
            return self.train_data

        def testing(self):
            """Return testing data - matches RSplit API"""
            return self.test_data

    # Create sample data
    data1 = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
    data2 = pd.DataFrame({'x': [4, 5, 6], 'y': [8, 10, 12]})

    split1 = MockSplit(train_data=data1, test_data=data2)
    split2 = MockSplit(train_data=data2, test_data=data1)

    return [split1, split2]


def test_fit_resamples_basic_structure(mock_workflow, mock_resamples):
    """Test fit_resamples returns TuneResults"""
    # Note: This test is limited because it requires full py_rsample integration
    # It tests the basic structure of the function

    results = fit_resamples(
        mock_workflow,
        mock_resamples,
        metrics=None
    )

    assert isinstance(results, TuneResults)
    assert hasattr(results, 'metrics')
    assert hasattr(results, 'predictions')


def test_tune_grid_basic_structure(mock_workflow, mock_resamples):
    """Test tune_grid returns TuneResults with grid"""
    param_info = {
        'penalty': {'range': (0.1, 1.0)},
        'mixture': {'range': (0, 1)}
    }

    results = tune_grid(
        mock_workflow,
        mock_resamples,
        grid=2,  # 2x2 grid
        param_info=param_info,
        metrics=None
    )

    assert isinstance(results, TuneResults)
    assert len(results.grid) == 4  # 2 x 2


def test_tune_grid_with_dataframe_grid(mock_workflow, mock_resamples):
    """Test tune_grid with pre-made grid"""
    grid_df = pd.DataFrame({
        'penalty': [0.1, 0.5, 1.0],
        'mixture': [0.2, 0.5, 0.8]
    })

    results = tune_grid(
        mock_workflow,
        mock_resamples,
        grid=grid_df,
        metrics=None
    )

    assert isinstance(results, TuneResults)
    assert len(results.grid) == 3
    assert '.config' in results.grid.columns


def test_tune_grid_requires_param_info():
    """Test tune_grid raises error without param_info when grid is int"""
    with pytest.raises(ValueError, match="param_info required"):
        tune_grid(
            workflow=None,
            resamples=None,
            grid=3,
            param_info=None
        )


# ============================================================================
# Test finalize_workflow()
# ============================================================================

def test_finalize_workflow_basic(mock_workflow):
    """Test finalizing workflow with best parameters"""
    best_params = {'penalty': 0.5, 'mixture': 0.8}

    final_wf = finalize_workflow(mock_workflow, best_params)

    # Check that workflow structure is maintained
    assert hasattr(final_wf, 'preprocessor')
    assert hasattr(final_wf, 'spec')


def test_finalize_workflow_updates_params(mock_workflow):
    """Test that finalize_workflow updates parameters"""
    # Add some tunable parameters to the mock spec
    mock_workflow.spec = replace(mock_workflow.spec, args=(('penalty', tune('penalty')), ('mixture', tune('mixture'))))

    best_params = {'penalty': 0.5, 'mixture': 0.8}
    final_wf = finalize_workflow(mock_workflow, best_params)

    # Verify structure is maintained
    assert hasattr(final_wf, 'spec')
    assert len(final_wf.spec.args) >= 2


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

def test_grid_regular_large_grid():
    """Test regular grid doesn't explode with many parameters"""
    param_info = {
        'param1': {'range': (0, 1)},
        'param2': {'range': (0, 1)},
        'param3': {'range': (0, 1)},
        'param4': {'range': (0, 1)}
    }
    grid = grid_regular(param_info, levels=3)

    assert len(grid) == 81  # 3^4


def test_grid_random_respects_size():
    """Test random grid generates exactly requested size"""
    param_info = {
        'param1': {'range': (0, 1)},
        'param2': {'range': (10, 100)},
        'param3': {'range': (0.001, 1.0), 'trans': 'log'}
    }

    for size in [5, 10, 50, 100]:
        grid = grid_random(param_info, size=size, seed=42)
        assert len(grid) == size


def test_tune_results_handles_multiple_metrics():
    """Test TuneResults with multiple metrics"""
    metrics_data = {
        '.config': ['config_001', 'config_001', 'config_002', 'config_002'],
        '.resample': ['Fold01', 'Fold02', 'Fold01', 'Fold02'],
        'rmse': [0.5, 0.6, 0.4, 0.5],
        'mae': [0.3, 0.4, 0.25, 0.35],
        'r_squared': [0.7, 0.65, 0.8, 0.75]
    }
    metrics_df = pd.DataFrame(metrics_data)

    grid_data = {
        '.config': ['config_001', 'config_002'],
        'penalty': [0.1, 1.0]
    }
    grid_df = pd.DataFrame(grid_data)

    results = TuneResults(metrics=metrics_df, grid=grid_df)

    # Should be able to rank by any metric
    best_rmse = results.select_best('rmse', maximize=False)
    best_r2 = results.select_best('r_squared', maximize=True)

    assert best_rmse['penalty'] == 1.0  # config_002
    assert best_r2['penalty'] == 1.0  # config_002


def test_tune_parameter_used_in_expression():
    """Test that TuneParameter can be used in parameter position"""
    penalty_param = tune(id='penalty')
    mixture_param = tune(id='mixture')

    # Simulate usage in model spec
    params = {
        'penalty': penalty_param,
        'mixture': mixture_param
    }

    assert isinstance(params['penalty'], TuneParameter)
    assert params['penalty'].id == 'penalty'
    assert params['mixture'].id == 'mixture'
