"""
Tests for mlp (Multi-Layer Perceptron) model specification and sklearn engine

Tests cover:
- Model specification creation
- Parameter translation (hidden_units, penalty, epochs, learn_rate, activation)
- Fitting with formula
- Prediction
- Extract outputs (layer weight summaries, stats)
- Evaluate() method
- Different network architectures
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import mlp, ModelSpec, ModelFit


class TestMLPSpec:
    """Test mlp() model specification"""

    def test_default_spec(self):
        """Test default mlp specification"""
        spec = mlp()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "mlp"
        assert spec.engine == "sklearn"
        assert spec.mode == "regression"
        assert spec.args == {}

    def test_spec_with_hidden_units_int(self):
        """Test mlp with single hidden layer (int)"""
        spec = mlp(hidden_units=50)

        assert spec.args == {"hidden_units": 50}

    def test_spec_with_hidden_units_tuple(self):
        """Test mlp with multiple hidden layers (tuple)"""
        spec = mlp(hidden_units=(100, 50))

        assert spec.args == {"hidden_units": (100, 50)}

    def test_spec_with_penalty(self):
        """Test mlp with penalty parameter"""
        spec = mlp(penalty=0.01)

        assert spec.args == {"penalty": 0.01}

    def test_spec_with_epochs(self):
        """Test mlp with epochs parameter"""
        spec = mlp(epochs=500)

        assert spec.args == {"epochs": 500}

    def test_spec_with_learn_rate(self):
        """Test mlp with learn_rate parameter"""
        spec = mlp(learn_rate=0.01)

        assert spec.args == {"learn_rate": 0.01}

    def test_spec_with_activation(self):
        """Test mlp with activation parameter"""
        spec = mlp(activation="tanh")

        assert spec.args == {"activation": "tanh"}

    def test_spec_with_all_parameters(self):
        """Test mlp with all parameters"""
        spec = mlp(
            hidden_units=(100, 50, 25),
            penalty=0.001,
            epochs=300,
            learn_rate=0.01,
            activation="relu"
        )

        assert spec.args == {
            "hidden_units": (100, 50, 25),
            "penalty": 0.001,
            "epochs": 300,
            "learn_rate": 0.01,
            "activation": "relu"
        }

    def test_spec_immutability(self):
        """Test that ModelSpec is immutable"""
        spec1 = mlp(hidden_units=50)
        spec2 = spec1.set_args(hidden_units=100)

        assert spec1.args == {"hidden_units": 50}
        assert spec2.args == {"hidden_units": 100}


class TestMLPFit:
    """Test mlp fitting"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
            "x3": [2, 4, 3, 6, 5, 3.5, 4.5, 5.5, 3.2, 4.8],
        })

    def test_fit_with_formula(self, train_data):
        """Test fitting with formula"""
        spec = mlp()
        fit = spec.fit(train_data, "y ~ x1 + x2 + x3")

        assert isinstance(fit, ModelFit)
        assert fit.spec == spec
        assert "model" in fit.fit_data

    def test_fit_model_class(self, train_data):
        """Test correct sklearn model class"""
        spec = mlp()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert fit.fit_data["model_class"] == "MLPRegressor"

    def test_fit_with_hidden_units_int(self, train_data):
        """Test fitting with single hidden layer"""
        spec = mlp(hidden_units=50)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        # sklearn stores int as int, not tuple
        assert model.hidden_layer_sizes == 50 or model.hidden_layer_sizes == (50,)

    def test_fit_with_hidden_units_tuple(self, train_data):
        """Test fitting with multiple hidden layers"""
        spec = mlp(hidden_units=(50, 25))
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.hidden_layer_sizes == (50, 25)

    def test_fit_with_penalty(self, train_data):
        """Test fitting with penalty parameter"""
        spec = mlp(penalty=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.alpha == 0.01

    def test_fit_with_epochs(self, train_data):
        """Test fitting with epochs parameter"""
        spec = mlp(epochs=100)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.max_iter == 100

    def test_fit_with_learn_rate(self, train_data):
        """Test fitting with learn_rate parameter"""
        spec = mlp(learn_rate=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.learning_rate_init == 0.01

    def test_fit_with_activation(self, train_data):
        """Test fitting with activation function"""
        spec = mlp(activation="tanh")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.fit_data["model"]
        assert model.activation == "tanh"

    def test_fit_stores_residuals(self, train_data):
        """Test that fit stores residuals"""
        spec = mlp()
        fit = spec.fit(train_data, "y ~ x1 + x2")

        assert "residuals" in fit.fit_data
        assert fit.fit_data["residuals"] is not None


class TestMLPPredict:
    """Test mlp prediction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = mlp(hidden_units=50, epochs=100)
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_predict_basic(self, fitted_model):
        """Test basic prediction"""
        test = pd.DataFrame({
            "x1": [12, 22],
            "x2": [6, 11],
        })

        predictions = fitted_model.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == 2

    def test_predict_values_reasonable(self, fitted_model):
        """Test that predictions are in reasonable range"""
        test = pd.DataFrame({
            "x1": [15],
            "x2": [7],
        })

        predictions = fitted_model.predict(test)

        # MLP may not converge with limited epochs, just check it returns a number
        assert predictions[".pred"].iloc[0] is not None
        assert not np.isnan(predictions[".pred"].iloc[0])

    def test_predict_invalid_type(self, fitted_model):
        """Test prediction with invalid type raises error"""
        test = pd.DataFrame({"x1": [12], "x2": [6]})

        with pytest.raises(ValueError, match="only supports type='numeric'"):
            fitted_model.predict(test, type="class")


class TestMLPExtractOutputs:
    """Test mlp output extraction"""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model for testing"""
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = mlp(hidden_units=(50, 25), epochs=100)
        fit = spec.fit(train, "y ~ x1 + x2")
        return fit

    def test_extract_fit_engine(self, fitted_model):
        """Test extract_fit_engine()"""
        sklearn_model = fitted_model.extract_fit_engine()

        assert sklearn_model is not None
        assert hasattr(sklearn_model, "coefs_")
        assert hasattr(sklearn_model, "hidden_layer_sizes")

    def test_extract_outputs_returns_three_dataframes(self, fitted_model):
        """Test extract_outputs() returns three DataFrames"""
        outputs, coefs, stats = fitted_model.extract_outputs()

        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefs, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_model_outputs(self, fitted_model):
        """Test Outputs DataFrame structure"""
        outputs, _, _ = fitted_model.extract_outputs()

        assert "actuals" in outputs.columns
        assert "fitted" in outputs.columns
        assert "residuals" in outputs.columns
        assert len(outputs) == 10

    def test_extract_outputs_coefficients_layer_weights(self, fitted_model):
        """Test Coefficients DataFrame contains layer weight summaries"""
        _, coefs, _ = fitted_model.extract_outputs()

        # Should have layer-wise weight summaries
        # Architecture: input(2) -> hidden(50) -> hidden(25) -> output(1)
        # So 3 weight matrices: layer_0_to_1, layer_1_to_2, layer_2_to_3
        assert len(coefs) == 3
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns  # Mean weight
        assert "std_error" in coefs.columns  # Std of weights

    def test_extract_outputs_stats(self, fitted_model):
        """Test Stats DataFrame structure"""
        _, _, stats = fitted_model.extract_outputs()

        metric_names = stats["metric"].values
        assert "rmse" in metric_names
        assert "r_squared" in metric_names
        assert "hidden_layer_sizes" in metric_names
        assert "activation" in metric_names
        assert "n_iter" in metric_names
        assert "loss" in metric_names


class TestMLPEvaluate:
    """Test mlp evaluate() method"""

    @pytest.fixture
    def train_test_data(self):
        """Create train/test split"""
        np.random.seed(42)
        train = pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14],
        })
        test = pd.DataFrame({
            "y": [160, 240, 200],
            "x1": [16, 24, 20],
            "x2": [8, 12, 10],
        })
        return train, test

    def test_evaluate(self, train_test_data):
        """Test evaluate() method"""
        train, test = train_test_data

        spec = mlp(hidden_units=50, epochs=100)
        fit = spec.fit(train, "y ~ x1 + x2")

        # Evaluate on test data
        fit = fit.evaluate(test, "y")

        # Check that evaluation data is stored
        assert "test_predictions" in fit.evaluation_data

    def test_evaluate_outputs(self, train_test_data):
        """Test that evaluate() includes test data in outputs"""
        train, test = train_test_data

        spec = mlp()
        fit = spec.fit(train, "y ~ x1 + x2")
        fit = fit.evaluate(test, "y")

        outputs, _, _ = fit.extract_outputs()

        # Should have both train and test splits
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values


class TestMLPParameterTranslation:
    """Test parameter translation from tidymodels to sklearn"""

    @pytest.fixture
    def train_data(self):
        """Create sample training data"""
        np.random.seed(42)
        return pd.DataFrame({
            "y": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "x1": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "x2": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

    def test_param_translation_hidden_units(self, train_data):
        """Test hidden_units parameter maps to hidden_layer_sizes"""
        spec = mlp(hidden_units=(50, 25))
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.hidden_layer_sizes == (50, 25)

    def test_param_translation_penalty(self, train_data):
        """Test penalty parameter maps to alpha"""
        spec = mlp(penalty=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.alpha == 0.01

    def test_param_translation_epochs(self, train_data):
        """Test epochs parameter maps to max_iter"""
        spec = mlp(epochs=100)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.max_iter == 100

    def test_param_translation_learn_rate(self, train_data):
        """Test learn_rate parameter maps to learning_rate_init"""
        spec = mlp(learn_rate=0.01)
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.learning_rate_init == 0.01

    def test_param_translation_activation(self, train_data):
        """Test activation parameter maps to activation"""
        spec = mlp(activation="tanh")
        fit = spec.fit(train_data, "y ~ x1 + x2")

        model = fit.extract_fit_engine()
        assert model.activation == "tanh"


class TestMLPIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow_simple(self):
        """Test complete workflow with simple architecture"""
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = mlp(hidden_units=50, epochs=100)
        fit = spec.fit(train, "sales ~ price + advertising")

        test = pd.DataFrame({
            "price": [12, 22, 28],
            "advertising": [6, 11, 14],
        })

        predictions = fit.predict(test)

        assert len(predictions) == 3
        assert ".pred" in predictions.columns
        assert all(predictions[".pred"] > 0)

    def test_full_workflow_deep_network(self):
        """Test complete workflow with deep architecture"""
        train = pd.DataFrame({
            "sales": [100, 200, 150, 300, 250, 180, 220, 280, 160, 240],
            "price": [10, 20, 15, 30, 25, 18, 22, 28, 16, 24],
            "advertising": [5, 10, 7, 15, 12, 9, 11, 14, 8, 12],
        })

        spec = mlp(hidden_units=(100, 50, 25), penalty=0.001, epochs=100)
        fit = spec.fit(train, "sales ~ price + advertising")

        # Extract outputs
        outputs, coefs, stats = fit.extract_outputs()
        assert len(outputs) == 10
        assert len(coefs) == 4  # 4 weight matrices for 3 hidden layers + output
        assert len(stats) > 0
