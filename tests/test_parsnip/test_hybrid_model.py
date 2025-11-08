"""
Tests for hybrid_model() and generic_hybrid engine
"""

import pytest
import pandas as pd
import numpy as np
from py_parsnip import hybrid_model, linear_reg, rand_forest, decision_tree


@pytest.fixture
def simple_data():
    """Simple regression data for testing"""
    np.random.seed(42)
    data = pd.DataFrame({
        'x': range(100),
        'y': np.linspace(0, 10, 100) + np.random.normal(0, 1, 100)
    })
    return data


@pytest.fixture
def time_series_data():
    """Time series data with date column"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    # Trend with regime change at midpoint
    y = np.concatenate([
        np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100),  # Period 1
        np.linspace(10, 5, 100) + np.random.normal(0, 0.5, 100),  # Period 2
    ])
    data = pd.DataFrame({
        'date': dates,
        'x': range(200),
        'y': y
    })
    return data


class TestHybridModelSpec:
    """Test hybrid_model() specification function"""

    def test_create_basic_spec(self):
        """Test creating basic hybrid model spec"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='residual'
        )

        assert spec.model_type == 'hybrid_model'
        assert spec.engine == 'generic_hybrid'
        assert spec.mode == 'regression'
        assert spec.args['strategy'] == 'residual'

    def test_residual_strategy(self):
        """Test residual strategy configuration"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='residual'
        )

        assert spec.args['strategy'] == 'residual'
        assert spec.args['model1_spec'].model_type == 'linear_reg'
        assert spec.args['model2_spec'].model_type == 'decision_tree'

    def test_sequential_strategy(self):
        """Test sequential strategy configuration"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='sequential',
            split_point=0.7
        )

        assert spec.args['strategy'] == 'sequential'
        assert spec.args['split_point'] == 0.7

    def test_weighted_strategy(self):
        """Test weighted strategy configuration"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='weighted',
            weight1=0.6,
            weight2=0.4
        )

        assert spec.args['strategy'] == 'weighted'
        assert spec.args['weight1'] == 0.6
        assert spec.args['weight2'] == 0.4

    def test_validation_missing_models(self):
        """Test validation when models are missing"""
        with pytest.raises(ValueError, match="Both model1 and model2 are required"):
            hybrid_model(model1=linear_reg(), model2=None)

        with pytest.raises(ValueError, match="Both model1 and model2 are required"):
            hybrid_model(model1=None, model2=linear_reg())

    def test_validation_invalid_strategy(self):
        """Test validation with invalid strategy"""
        with pytest.raises(ValueError, match="strategy must be one of"):
            hybrid_model(
                model1=linear_reg(),
                model2=linear_reg(),
                strategy='invalid'
            )

    def test_validation_sequential_missing_split(self):
        """Test validation for sequential without split_point"""
        with pytest.raises(ValueError, match="split_point is required for sequential"):
            hybrid_model(
                model1=linear_reg(),
                model2=linear_reg(),
                strategy='sequential'
            )

    def test_validation_weights_out_of_bounds(self):
        """Test validation for weights out of bounds"""
        with pytest.raises(ValueError, match="Weights must be between 0 and 1"):
            hybrid_model(
                model1=linear_reg(),
                model2=linear_reg(),
                strategy='weighted',
                weight1=1.5,
                weight2=0.5
            )

    def test_validation_weights_warning(self):
        """Test warning when weights don't sum to 1"""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = hybrid_model(
                model1=linear_reg(),
                model2=linear_reg(),
                strategy='weighted',
                weight1=0.7,
                weight2=0.5
            )
            assert len(w) == 1
            assert "sum to" in str(w[0].message)


class TestResidualStrategy:
    """Test residual strategy (model2 trains on residuals)"""

    def test_residual_fit(self, simple_data):
        """Test fitting with residual strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='residual'
        )

        fit = spec.fit(simple_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'residual'
        assert 'model1_fit' in fit.fit_data
        assert 'model2_fit' in fit.fit_data
        assert 'fitted' in fit.fit_data
        assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_residual_predictions(self, simple_data):
        """Test predictions with residual strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='residual'
        )

        fit = spec.fit(simple_data, 'y ~ x')

        # Predict on new data
        test_data = pd.DataFrame({'x': range(100, 110)})
        predictions = fit.predict(test_data)

        assert '.pred' in predictions.columns
        assert len(predictions) == 10
        assert not predictions['.pred'].isna().any()

    def test_residual_improves_over_single_model(self, simple_data):
        """Test that hybrid improves over single model (in general)"""
        # Fit single linear model
        single_spec = linear_reg()
        single_fit = single_spec.fit(simple_data, 'y ~ x')
        single_rmse = np.sqrt(np.mean(single_fit.fit_data['residuals']**2))

        # Fit hybrid model
        hybrid_spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='residual'
        )
        hybrid_fit = hybrid_spec.fit(simple_data, 'y ~ x')
        hybrid_rmse = np.sqrt(np.mean(hybrid_fit.fit_data['residuals']**2))

        # Hybrid should generally be better (captures residual patterns)
        # Not strict assertion as it depends on data, but should usually be true
        print(f"Single RMSE: {single_rmse:.4f}, Hybrid RMSE: {hybrid_rmse:.4f}")
        assert hybrid_rmse <= single_rmse * 1.1  # Within 10% at worst

    def test_residual_extract_outputs(self, simple_data):
        """Test extract_outputs with residual strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='residual'
        )

        fit = spec.fit(simple_data, 'y ~ x')
        outputs, coefficients, stats = fit.extract_outputs()

        # Check outputs
        assert 'actuals' in outputs.columns
        assert 'fitted' in outputs.columns
        assert 'residuals' in outputs.columns
        assert len(outputs) == len(simple_data)

        # Check coefficients (hyperparameters)
        assert 'variable' in coefficients.columns
        assert 'strategy' in coefficients['variable'].values
        assert 'model1_type' in coefficients['variable'].values
        assert 'model2_type' in coefficients['variable'].values

        # Check stats
        assert 'metric' in stats.columns
        assert 'rmse' in stats['metric'].values
        assert 'r_squared' in stats['metric'].values


class TestSequentialStrategy:
    """Test sequential strategy (different models for different periods)"""

    def test_sequential_fit_int_split(self, simple_data):
        """Test sequential with integer split point"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='sequential',
            split_point=50
        )

        fit = spec.fit(simple_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'sequential'
        assert fit.fit_data['split_point'] == 50
        assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_sequential_fit_float_split(self, simple_data):
        """Test sequential with float split point (proportion)"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='sequential',
            split_point=0.7
        )

        fit = spec.fit(simple_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'sequential'
        assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_sequential_fit_date_split(self, time_series_data):
        """Test sequential with date string split point"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='sequential',
            split_point='2020-04-10'
        )

        fit = spec.fit(time_series_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'sequential'
        assert len(fit.fit_data['fitted']) == len(time_series_data)

    def test_sequential_predictions(self, simple_data):
        """Test predictions with sequential strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='sequential',
            split_point=0.6
        )

        fit = spec.fit(simple_data, 'y ~ x')

        test_data = pd.DataFrame({'x': range(100, 110)})
        predictions = fit.predict(test_data)

        assert '.pred' in predictions.columns
        assert len(predictions) == 10


class TestWeightedStrategy:
    """Test weighted strategy (weighted combination of predictions)"""

    def test_weighted_fit(self, simple_data):
        """Test fitting with weighted strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='weighted',
            weight1=0.6,
            weight2=0.4
        )

        fit = spec.fit(simple_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'weighted'
        assert fit.fit_data['weight1'] == 0.6
        assert fit.fit_data['weight2'] == 0.4
        assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_weighted_predictions(self, simple_data):
        """Test predictions with weighted strategy"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='weighted',
            weight1=0.7,
            weight2=0.3
        )

        fit = spec.fit(simple_data, 'y ~ x')

        test_data = pd.DataFrame({'x': range(100, 110)})
        predictions = fit.predict(test_data)

        assert '.pred' in predictions.columns
        assert len(predictions) == 10

    def test_weighted_equal_weights_same_model(self, simple_data):
        """Test that equal weights with same model gives same result as single model"""
        # Single model
        single_spec = linear_reg()
        single_fit = single_spec.fit(simple_data, 'y ~ x')
        single_preds = single_fit.predict(simple_data)

        # Hybrid with equal weights, same model
        hybrid_spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='weighted',
            weight1=0.5,
            weight2=0.5
        )
        hybrid_fit = hybrid_spec.fit(simple_data, 'y ~ x')
        hybrid_preds = hybrid_fit.predict(simple_data)

        # Should be approximately equal
        np.testing.assert_allclose(
            single_preds['.pred'].values,
            hybrid_preds['.pred'].values,
            rtol=1e-5
        )

    def test_weighted_extract_outputs_includes_weights(self, simple_data):
        """Test that extract_outputs includes weight coefficients"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='weighted',
            weight1=0.6,
            weight2=0.4
        )

        fit = spec.fit(simple_data, 'y ~ x')
        _, coefficients, _ = fit.extract_outputs()

        # Check weights in coefficients
        weight_vars = coefficients[coefficients['variable'].isin(['weight1', 'weight2'])]
        assert len(weight_vars) == 2

        weight1_val = weight_vars[weight_vars['variable'] == 'weight1']['coefficient'].values[0]
        weight2_val = weight_vars[weight_vars['variable'] == 'weight2']['coefficient'].values[0]

        assert weight1_val == 0.6
        assert weight2_val == 0.4


class TestCustomDataStrategy:
    """Test custom_data strategy (different/overlapping training datasets)"""

    @pytest.fixture
    def overlapping_data(self):
        """Create overlapping training datasets"""
        np.random.seed(42)
        # Full dataset: 150 observations
        dates = pd.date_range('2020-01-01', periods=150, freq='D')
        y = np.linspace(0, 15, 150) + np.random.normal(0, 1, 150)

        full_df = pd.DataFrame({
            'date': dates,
            'x': range(150),
            'y': y
        })

        # Model 1 trains on first 100 observations (2020-01-01 to 2020-04-09)
        data1 = full_df.iloc[:100].copy()

        # Model 2 trains on last 80 observations (2020-03-22 to 2020-05-29)
        # This creates 30-day overlap (2020-03-22 to 2020-04-09)
        data2 = full_df.iloc[70:].copy()

        return {'model1': data1, 'model2': data2, 'full': full_df}

    @pytest.fixture
    def non_overlapping_data(self):
        """Create non-overlapping training datasets"""
        np.random.seed(42)
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2020-07-01', periods=100, freq='D')

        data1 = pd.DataFrame({
            'date': dates1,
            'x': range(100),
            'y': np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100)
        })

        data2 = pd.DataFrame({
            'date': dates2,
            'x': range(100, 200),
            'y': np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
        })

        return {'model1': data1, 'model2': data2}

    def test_custom_data_spec_creation(self):
        """Test creating custom_data strategy spec"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='custom_data',
            blend_predictions='weighted',
            weight1=0.4,
            weight2=0.6
        )

        assert spec.model_type == 'hybrid_model'
        assert spec.args['strategy'] == 'custom_data'
        assert spec.args['blend_predictions'] == 'weighted'
        assert spec.args['weight1'] == 0.4
        assert spec.args['weight2'] == 0.6

    def test_custom_data_validation_blend_predictions(self):
        """Test validation of blend_predictions parameter"""
        with pytest.raises(ValueError, match="blend_predictions must be one of"):
            hybrid_model(
                model1=linear_reg(),
                model2=linear_reg(),
                strategy='custom_data',
                blend_predictions='invalid'
            )

    def test_custom_data_fit_overlapping(self, overlapping_data):
        """Test fitting with overlapping training periods"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='weighted',
            weight1=0.5,
            weight2=0.5
        )

        # Fit with dict input
        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        assert fit.fit_data['strategy'] == 'custom_data'
        assert 'model1_fit' in fit.fit_data
        assert 'model2_fit' in fit.fit_data
        assert fit.fit_data['n_obs_1'] == 100
        assert fit.fit_data['n_obs_2'] == 80

    def test_custom_data_fit_non_overlapping(self, non_overlapping_data):
        """Test fitting with non-overlapping training periods"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=decision_tree().set_mode('regression'),
            strategy='custom_data',
            blend_predictions='weighted',
            weight1=0.3,
            weight2=0.7
        )

        fit = spec.fit(non_overlapping_data, 'y ~ x')

        assert fit.fit_data['strategy'] == 'custom_data'
        assert fit.fit_data['n_obs_1'] == 100
        assert fit.fit_data['n_obs_2'] == 100

    def test_custom_data_missing_keys(self, overlapping_data):
        """Test error when dict missing required keys"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data'
        )

        # Missing 'model2' key
        with pytest.raises(ValueError, match="must provide 'model1' and 'model2' keys"):
            spec.fit({'model1': overlapping_data['model1']}, 'y ~ x')

        # Missing 'model1' key
        with pytest.raises(ValueError, match="must provide 'model1' and 'model2' keys"):
            spec.fit({'model2': overlapping_data['model2']}, 'y ~ x')

    def test_custom_data_predictions_weighted(self, overlapping_data):
        """Test predictions with weighted blend"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='weighted',
            weight1=0.6,
            weight2=0.4
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        # Predict on new data
        test_data = pd.DataFrame({'x': range(150, 160)})
        predictions = fit.predict(test_data)

        assert '.pred' in predictions.columns
        assert len(predictions) == 10
        assert not predictions['.pred'].isna().any()

        # Manually verify weighted blend
        pred1 = fit.fit_data['model1_fit'].predict(test_data)['.pred'].values
        pred2 = fit.fit_data['model2_fit'].predict(test_data)['.pred'].values
        expected = 0.6 * pred1 + 0.4 * pred2

        np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-5)

    def test_custom_data_predictions_avg(self, overlapping_data):
        """Test predictions with average blend"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='avg'
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        test_data = pd.DataFrame({'x': range(150, 160)})
        predictions = fit.predict(test_data)

        # Verify average blend
        pred1 = fit.fit_data['model1_fit'].predict(test_data)['.pred'].values
        pred2 = fit.fit_data['model2_fit'].predict(test_data)['.pred'].values
        expected = 0.5 * (pred1 + pred2)

        np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-5)

    def test_custom_data_predictions_sum(self, overlapping_data):
        """Test predictions with sum blend"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='sum'
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        test_data = pd.DataFrame({'x': range(150, 160)})
        predictions = fit.predict(test_data)

        # Verify sum blend
        pred1 = fit.fit_data['model1_fit'].predict(test_data)['.pred'].values
        pred2 = fit.fit_data['model2_fit'].predict(test_data)['.pred'].values
        expected = pred1 + pred2

        np.testing.assert_allclose(predictions['.pred'].values, expected, rtol=1e-5)

    def test_custom_data_predictions_model1_only(self, overlapping_data):
        """Test predictions with model1 only"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='model1'
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        test_data = pd.DataFrame({'x': range(150, 160)})
        predictions = fit.predict(test_data)

        # Verify only model1 predictions used
        pred1 = fit.fit_data['model1_fit'].predict(test_data)['.pred'].values

        np.testing.assert_allclose(predictions['.pred'].values, pred1, rtol=1e-5)

    def test_custom_data_predictions_model2_only(self, overlapping_data):
        """Test predictions with model2 only"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='model2'
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        test_data = pd.DataFrame({'x': range(150, 160)})
        predictions = fit.predict(test_data)

        # Verify only model2 predictions used
        pred2 = fit.fit_data['model2_fit'].predict(test_data)['.pred'].values

        np.testing.assert_allclose(predictions['.pred'].values, pred2, rtol=1e-5)

    def test_custom_data_extract_outputs_coefficients(self, overlapping_data):
        """Test that extract_outputs includes custom_data parameters"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='custom_data',
            blend_predictions='weighted',
            weight1=0.4,
            weight2=0.6
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        _, coefficients, _ = fit.extract_outputs()

        # Check that coefficients include weights and blend_predictions
        coef_vars = coefficients['variable'].values
        assert 'weight1' in coef_vars
        assert 'weight2' in coef_vars
        assert 'blend_predictions' in coef_vars

        # Verify values
        weight1_val = coefficients[coefficients['variable'] == 'weight1']['coefficient'].values[0]
        weight2_val = coefficients[coefficients['variable'] == 'weight2']['coefficient'].values[0]
        blend_val = coefficients[coefficients['variable'] == 'blend_predictions']['coefficient'].values[0]

        assert weight1_val == 0.4
        assert weight2_val == 0.6
        assert blend_val == 'weighted'

    def test_custom_data_extract_outputs_stats(self, overlapping_data):
        """Test that extract_outputs includes per-model observation counts"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=linear_reg(),
            strategy='custom_data',
            blend_predictions='avg'
        )

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }
        fit = spec.fit(data_dict, 'y ~ x')

        _, _, stats = fit.extract_outputs()

        # Check that stats include n_obs for each model
        stat_metrics = stats['metric'].values
        assert 'n_obs_model1' in stat_metrics
        assert 'n_obs_model2' in stat_metrics

        # Verify values
        n_obs_1 = stats[stats['metric'] == 'n_obs_model1']['value'].values[0]
        n_obs_2 = stats[stats['metric'] == 'n_obs_model2']['value'].values[0]

        assert n_obs_1 == 100
        assert n_obs_2 == 80

    def test_custom_data_different_model_types(self, overlapping_data):
        """Test custom_data with different model type combinations"""
        combinations = [
            (linear_reg(), decision_tree().set_mode('regression')),
            (decision_tree().set_mode('regression'), rand_forest().set_mode('regression')),
            (linear_reg(), rand_forest().set_mode('regression')),
        ]

        data_dict = {
            'model1': overlapping_data['model1'],
            'model2': overlapping_data['model2']
        }

        for model1, model2 in combinations:
            spec = hybrid_model(
                model1=model1,
                model2=model2,
                strategy='custom_data',
                blend_predictions='weighted'
            )
            fit = spec.fit(data_dict, 'y ~ x')

            assert fit.fit_data['model1_fit'] is not None
            assert fit.fit_data['model2_fit'] is not None

            # Test predictions work
            test_data = pd.DataFrame({'x': range(150, 160)})
            predictions = fit.predict(test_data)
            assert len(predictions) == 10
            assert not predictions['.pred'].isna().any()


class TestHybridModelEdgeCases:
    """Test edge cases and error handling"""

    def test_mode_auto_set_for_unknown_mode(self, simple_data):
        """Test that models with unknown mode get set to regression"""
        # rand_forest without mode specified
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest(),  # No .set_mode() called
            strategy='residual'
        )

        # Should auto-set mode to regression and fit successfully
        fit = spec.fit(simple_data, 'y ~ x')

        assert fit.fit_data['model1_spec'].mode == 'regression'
        assert fit.fit_data['model2_spec'].mode == 'regression'

    def test_different_model_combinations(self, simple_data):
        """Test various model combinations"""
        combinations = [
            (linear_reg(), linear_reg()),
            (linear_reg(), decision_tree().set_mode('regression')),
            (decision_tree().set_mode('regression'), rand_forest().set_mode('regression')),
        ]

        for model1, model2 in combinations:
            spec = hybrid_model(model1=model1, model2=model2, strategy='residual')
            fit = spec.fit(simple_data, 'y ~ x')

            assert fit.fit_data['fitted'] is not None
            assert len(fit.fit_data['fitted']) == len(simple_data)

    def test_model_metadata_columns(self, simple_data):
        """Test that outputs include model metadata columns"""
        spec = hybrid_model(
            model1=linear_reg(),
            model2=rand_forest().set_mode('regression'),
            strategy='residual'
        )

        fit = spec.fit(simple_data, 'y ~ x')
        outputs, coefficients, stats = fit.extract_outputs()

        # All DataFrames should have metadata columns
        for df in [outputs, coefficients, stats]:
            assert 'model' in df.columns
            assert 'model_group_name' in df.columns
            assert 'group' in df.columns
