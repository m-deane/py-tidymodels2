"""
Tests for enhanced recipe generation with intelligent 51-step selection.

Validates that the recipe generation system intelligently selects preprocessing
steps based on data characteristics and model requirements.
"""

import pytest
from py_agent.tools.recipe_generation import (
    create_recipe,
    get_recipe_templates,
    _needs_polynomial_features,
    _needs_interactions,
    _needs_transformation,
    _needs_correlation_filter,
    _needs_dimensionality_reduction,
    _needs_normalization,
)


class TestHelperFunctions:
    """Tests for intelligent preprocessing decision functions."""

    # Tests for _needs_polynomial_features()
    def test_polynomial_features_for_linear_models(self):
        """Linear models with nonlinear trend should get polynomial features."""
        data_chars = {
            'n_features': 5,
            'trend': {'strength': 0.7}  # Strong trend
        }
        assert _needs_polynomial_features('linear_reg', data_chars) is True
        assert _needs_polynomial_features('poisson_reg', data_chars) is True
        assert _needs_polynomial_features('svm_linear', data_chars) is True

    def test_polynomial_features_avoids_high_dimensions(self):
        """Should not add polynomial features if already many features."""
        data_chars = {
            'n_features': 20,  # Too many features
            'trend': {'strength': 0.8}
        }
        assert _needs_polynomial_features('linear_reg', data_chars) is False

    def test_polynomial_features_not_for_tree_models(self):
        """Tree models don't benefit from polynomial features."""
        data_chars = {
            'n_features': 5,
            'trend': {'strength': 0.7}
        }
        assert _needs_polynomial_features('rand_forest', data_chars) is False
        assert _needs_polynomial_features('boost_tree', data_chars) is False
        assert _needs_polynomial_features('decision_tree', data_chars) is False

    def test_polynomial_features_weak_trend(self):
        """Weak trend should not trigger polynomial features."""
        data_chars = {
            'n_features': 5,
            'trend': {'strength': 0.3}  # Weak trend
        }
        assert _needs_polynomial_features('linear_reg', data_chars) is False

    # Tests for _needs_interactions()
    def test_interactions_for_linear_models(self):
        """Linear models with moderate feature count should get interactions."""
        assert _needs_interactions('linear_reg', n_features=5) is True
        assert _needs_interactions('poisson_reg', n_features=8) is True
        assert _needs_interactions('svm_linear', n_features=3) is True

    def test_interactions_boundary_conditions(self):
        """Interactions only for 2-10 features."""
        # Too few features
        assert _needs_interactions('linear_reg', n_features=1) is False

        # Too many features (explosion risk)
        assert _needs_interactions('linear_reg', n_features=15) is False
        assert _needs_interactions('linear_reg', n_features=11) is False

        # Edge cases
        assert _needs_interactions('linear_reg', n_features=2) is True
        assert _needs_interactions('linear_reg', n_features=10) is True

    def test_interactions_not_for_tree_models(self):
        """Tree models capture interactions automatically."""
        assert _needs_interactions('rand_forest', n_features=5) is False
        assert _needs_interactions('boost_tree', n_features=5) is False

    # Tests for _needs_transformation()
    def test_transformation_for_normality_assuming_models(self):
        """Models assuming normality should get transformations."""
        assert _needs_transformation('linear_reg') is True
        assert _needs_transformation('poisson_reg') is True
        assert _needs_transformation('svm_rbf') is True
        assert _needs_transformation('svm_linear') is True
        assert _needs_transformation('nearest_neighbor') is True

    def test_transformation_not_for_tree_models(self):
        """Tree models don't require normality."""
        assert _needs_transformation('rand_forest') is False
        assert _needs_transformation('boost_tree') is False
        assert _needs_transformation('decision_tree') is False

    def test_transformation_not_for_time_series(self):
        """Time series models handle transformations internally."""
        assert _needs_transformation('prophet_reg') is False
        assert _needs_transformation('arima_reg') is False

    # Tests for _needs_correlation_filter()
    def test_correlation_filter_for_linear_models(self):
        """Linear models affected by multicollinearity need filtering."""
        assert _needs_correlation_filter('linear_reg') is True
        assert _needs_correlation_filter('poisson_reg') is True
        assert _needs_correlation_filter('gen_additive_mod') is True

    def test_correlation_filter_not_for_tree_models(self):
        """Tree models robust to multicollinearity."""
        assert _needs_correlation_filter('rand_forest') is False
        assert _needs_correlation_filter('boost_tree') is False

    # Tests for _needs_dimensionality_reduction()
    def test_pca_for_high_dimensional_data(self):
        """High-dimensional data (>20 features) should trigger PCA."""
        assert _needs_dimensionality_reduction(
            n_features=25,
            n_observations=100,
            model_type='rand_forest'
        ) is True

        assert _needs_dimensionality_reduction(
            n_features=50,
            n_observations=200,
            model_type='mlp'
        ) is True

    def test_pca_for_features_exceeding_observations(self):
        """PCA when features > 50% of observations."""
        assert _needs_dimensionality_reduction(
            n_features=15,
            n_observations=20,  # 15 > 20 * 0.5
            model_type='rand_forest'
        ) is True

    def test_pca_not_for_time_series(self):
        """Time series models should not use PCA."""
        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='prophet_reg'
        ) is False

        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='arima_reg'
        ) is False

        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='seasonal_reg'
        ) is False

    def test_pca_not_for_interpretable_models(self):
        """Interpretable models should not use PCA (loses feature meaning)."""
        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='linear_reg'
        ) is False

        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='decision_tree'
        ) is False

        assert _needs_dimensionality_reduction(
            n_features=30,
            n_observations=100,
            model_type='mars'
        ) is False

    def test_pca_not_for_low_dimensions(self):
        """Low-dimensional data should not trigger PCA."""
        assert _needs_dimensionality_reduction(
            n_features=10,
            n_observations=100,
            model_type='rand_forest'
        ) is False

    # Tests for _needs_normalization()
    def test_normalization_for_distance_based_models(self):
        """Distance-based models need normalization."""
        assert _needs_normalization('nearest_neighbor') is True
        assert _needs_normalization('svm_rbf') is True
        assert _needs_normalization('svm_linear') is True

    def test_normalization_for_neural_networks(self):
        """Neural networks benefit from normalization."""
        assert _needs_normalization('mlp') is True

    def test_normalization_for_linear_models(self):
        """Linear models benefit from normalization (convergence)."""
        assert _needs_normalization('linear_reg') is True
        assert _needs_normalization('poisson_reg') is True

    def test_normalization_for_tree_models(self):
        """Tree models benefit from normalization (equal feature importance)."""
        assert _needs_normalization('rand_forest') is True
        assert _needs_normalization('boost_tree') is True
        assert _needs_normalization('decision_tree') is True


class TestRecipeGeneration:
    """Tests for create_recipe() function with 8-phase pipeline."""

    def test_high_dimensional_triggers_pca(self):
        """High-dimensional data should trigger PCA."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.02,
            'outlier_rate': 0.01,
            'n_features': 25,  # High-dimensional
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'rand_forest')

        assert 'step_pca' in code
        # Should have PCA with ~20 components (min(25*0.8, 20))
        assert 'num_comp=20' in code

    def test_nonlinear_trend_triggers_polynomial(self):
        """Strong trend should trigger polynomial features for linear models."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.8},  # Strong trend
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 5,  # Low enough for polynomial
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_poly' in code
        assert 'degree=2' in code

    def test_moderate_features_triggers_interactions(self):
        """Moderate feature count (2-10) should trigger interactions for linear models."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.3},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 5,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_interact' in code

    def test_linear_model_gets_correlation_filter(self):
        """Linear models should get correlation filtering."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_select_corr' in code
        assert 'threshold=0.9' in code
        assert 'multicollinearity' in code

    def test_linear_model_gets_transformation(self):
        """Linear models should get YeoJohnson transformation."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_YeoJohnson' in code

    def test_distance_model_gets_normalization(self):
        """Distance-based models should get normalization."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'nearest_neighbor')

        assert 'step_normalize' in code

    def test_outliers_trigger_naomit(self):
        """Outliers should trigger step_naomit()."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.05,  # Outliers present
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_naomit' in code

    def test_missing_data_triggers_imputation(self):
        """Missing data should trigger appropriate imputation."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.08,  # Moderate missing rate
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        # Should get median imputation for moderate missing rate
        assert 'step_impute_median' in code

    def test_high_missing_rate_uses_knn(self):
        """High missing rate should use KNN imputation."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.20,  # High missing rate
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_impute_knn' in code
        assert 'neighbors=5' in code

    def test_ml_models_get_dummy_encoding(self):
        """ML models should get dummy encoding."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_dummy' in code
        assert 'all_nominal_predictors' in code

    def test_time_series_no_dummy_encoding(self):
        """Time series models should not get dummy encoding."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True},
            'trend': {'strength': 0.5},
            'missing_rate': 0.02,
            'outlier_rate': 0.0,
            'n_features': 5,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'prophet_reg')

        assert 'step_dummy' not in code

    def test_zero_variance_filter_always_present(self):
        """Zero-variance filter should always be present."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_zv' in code

    def test_tree_models_no_polynomial(self):
        """Tree models should not get polynomial features."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.8},  # Strong trend
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 5,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'rand_forest')

        assert 'step_poly' not in code

    def test_daily_data_gets_date_features(self):
        """Daily data should get date feature extraction."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 5,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'step_date' in code

    def test_retail_domain_adds_holiday_features(self):
        """Retail domain should add holiday features."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 5,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg', domain='retail')

        assert 'step_date' in code
        assert 'is_holiday' in code


class TestRecipeTemplates:
    """Tests for get_recipe_templates() function."""

    def test_all_17_templates_present(self):
        """Verify all 17 templates are available."""
        templates = get_recipe_templates()

        assert len(templates) == 17

        # Basic templates (3)
        assert 'minimal' in templates
        assert 'standard_ml' in templates
        assert 'time_series' in templates

        # Retail & E-commerce (3)
        assert 'retail_daily' in templates
        assert 'retail_weekly' in templates
        assert 'ecommerce_hourly' in templates

        # Energy & Utilities (2)
        assert 'energy_hourly' in templates
        assert 'solar_generation' in templates

        # Finance & Economics (2)
        assert 'finance_daily' in templates
        assert 'stock_prices' in templates

        # Healthcare (1)
        assert 'patient_volume' in templates

        # Transportation & Logistics (2)
        assert 'demand_forecasting' in templates
        assert 'traffic_volume' in templates

        # High-dimensional & specialized (4)
        assert 'high_dimensional' in templates
        assert 'text_features' in templates
        assert 'iot_sensors' in templates

    def test_templates_have_required_fields(self):
        """All templates should have description and steps."""
        templates = get_recipe_templates()

        for name, template in templates.items():
            assert 'description' in template, f"{name} missing description"
            assert 'steps' in template, f"{name} missing steps"
            assert isinstance(template['steps'], list), f"{name} steps not a list"

    def test_minimal_template_structure(self):
        """Minimal template should have only imputation."""
        templates = get_recipe_templates()
        minimal = templates['minimal']

        assert len(minimal['steps']) == 1
        assert 'step_impute_median' in minimal['steps'][0]

    def test_standard_ml_template_structure(self):
        """Standard ML template should have imputation, normalization, encoding."""
        templates = get_recipe_templates()
        standard = templates['standard_ml']

        steps_str = ''.join(standard['steps'])
        assert 'step_impute_median' in steps_str
        assert 'step_normalize' in steps_str
        assert 'step_dummy' in steps_str

    def test_time_series_template_no_dummy(self):
        """Time series template should not have dummy encoding."""
        templates = get_recipe_templates()
        ts = templates['time_series']

        steps_str = ''.join(ts['steps'])
        assert 'step_dummy' not in steps_str
        assert 'step_impute_linear' in steps_str

    def test_retail_template_has_date_features(self):
        """Retail templates should have date feature extraction."""
        templates = get_recipe_templates()
        retail_daily = templates['retail_daily']

        steps_str = ''.join(retail_daily['steps'])
        assert 'step_date' in steps_str
        assert 'is_holiday' in steps_str

    def test_high_dimensional_template_has_pca(self):
        """High-dimensional template should have PCA."""
        templates = get_recipe_templates()
        high_dim = templates['high_dimensional']

        steps_str = ''.join(high_dim['steps'])
        assert 'step_pca' in steps_str
        assert 'step_select_corr' in steps_str

    def test_finance_template_has_yeojohnson(self):
        """Finance template should have YeoJohnson for negative values."""
        templates = get_recipe_templates()
        finance = templates['finance_daily']

        steps_str = ''.join(finance['steps'])
        assert 'step_YeoJohnson' in steps_str

    def test_iot_template_has_correlation_filter(self):
        """IoT template should filter correlated sensor data."""
        templates = get_recipe_templates()
        iot = templates['iot_sensors']

        steps_str = ''.join(iot['steps'])
        assert 'step_select_corr' in steps_str
        assert 'threshold=0.95' in steps_str

    def test_template_descriptions_exist(self):
        """All templates should have non-empty descriptions."""
        templates = get_recipe_templates()

        for name, template in templates.items():
            assert len(template['description']) > 10, \
                f"{name} has too short description"


class TestGeneratedRecipeCode:
    """Tests for recipe code generation."""

    def test_generated_code_has_imports(self):
        """Generated code should have necessary imports."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'from py_recipes import recipe' in code
        assert 'from py_recipes.selectors import' in code

    def test_generated_code_creates_recipe(self):
        """Generated code should create recipe object."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        assert 'rec = (recipe(data, formula)' in code

    def test_generated_code_properly_indented(self):
        """Generated code should have proper indentation for steps."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.05,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        # Steps should be indented
        lines = code.split('\n')
        step_lines = [l for l in lines if l.strip().startswith('.step_')]
        assert all(l.startswith('    ') for l in step_lines), \
            "Steps not properly indented"

    def test_generated_code_closes_parentheses(self):
        """Generated code should properly close recipe parentheses."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')

        # Should end with closing parenthesis
        assert code.strip().endswith(')')


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_features_handled(self):
        """Zero features should be handled gracefully."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 0,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')
        assert code is not None
        assert 'recipe' in code

    def test_zero_observations_handled(self):
        """Zero observations should be handled gracefully."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 0
        }

        code = create_recipe(data_chars, 'linear_reg')
        assert code is not None

    def test_missing_optional_fields(self):
        """Missing optional data characteristics should use defaults."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'missing_rate': 0.0,
            # Missing: trend, outlier_rate, n_features, n_observations
        }

        code = create_recipe(data_chars, 'linear_reg')
        assert code is not None
        assert 'recipe' in code

    def test_unknown_model_type_handled(self):
        """Unknown model type should be handled gracefully."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'unknown_model_type')
        assert code is not None
        assert 'recipe' in code

    def test_extreme_missing_rate(self):
        """Extreme missing rate (100%) should be handled."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 1.0,  # 100% missing
            'outlier_rate': 0.0,
            'n_features': 10,
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'linear_reg')
        # Should use KNN imputation for very high missing rate
        assert 'step_impute_knn' in code

    def test_very_high_dimensions(self):
        """Very high-dimensional data should trigger PCA."""
        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': False},
            'trend': {'strength': 0.5},
            'missing_rate': 0.0,
            'outlier_rate': 0.0,
            'n_features': 100,  # Very high
            'n_observations': 500
        }

        code = create_recipe(data_chars, 'rand_forest')
        assert 'step_pca' in code
        # Should cap at 20 components
        assert 'num_comp=20' in code
