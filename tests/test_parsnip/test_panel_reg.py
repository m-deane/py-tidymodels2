"""
Tests for panel_reg model specification and statsmodels MixedLM engine

Tests cover:
- Model specification creation
- Engine registration
- Fitting with random intercepts/slopes
- Prediction for training and new groups
- Extract outputs (three-DataFrame format)
- ICC calculation and variance components
- Integration with workflows
- Error handling for invalid configurations
"""

import pytest
import pandas as pd
import numpy as np

from py_parsnip import panel_reg, linear_reg, ModelSpec, ModelFit
from py_workflows import workflow
from py_recipes import recipe
from py_recipes.selectors import all_numeric


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def balanced_panel():
    """Create balanced panel data: 3 stores × 10 weeks = 30 observations"""
    np.random.seed(42)

    # 3 stores, 10 weeks each
    groups = ['Store_A', 'Store_B', 'Store_C'] * 10
    week = sorted(list(range(1, 11)) * 3)

    # Group effects (baseline sales by store)
    group_effects = {'Store_A': 100, 'Store_B': 150, 'Store_C': 80}
    group_effect_values = [group_effects[g] for g in groups]

    # Price sensitivity (same for all stores)
    price = np.random.uniform(8, 12, 30)

    # Generate sales with group effects
    sales = (
        np.array(group_effect_values) +
        -2 * price +  # Price effect
        np.random.normal(0, 5, 30)  # Random noise
    )

    return pd.DataFrame({
        'store_id': groups,
        'week': week,
        'price': price,
        'sales': sales
    })


@pytest.fixture
def unbalanced_panel():
    """Create unbalanced panel: different sizes per group"""
    np.random.seed(42)

    data = []

    # Store_A: 12 observations
    for i in range(12):
        data.append({
            'store_id': 'Store_A',
            'week': i + 1,
            'price': np.random.uniform(8, 12),
            'sales': 100 - 2 * np.random.uniform(8, 12) + np.random.normal(0, 5)
        })

    # Store_B: 8 observations
    for i in range(8):
        data.append({
            'store_id': 'Store_B',
            'week': i + 1,
            'price': np.random.uniform(8, 12),
            'sales': 150 - 2 * np.random.uniform(8, 12) + np.random.normal(0, 5)
        })

    # Store_C: 6 observations
    for i in range(6):
        data.append({
            'store_id': 'Store_C',
            'week': i + 1,
            'price': np.random.uniform(8, 12),
            'sales': 80 - 2 * np.random.uniform(8, 12) + np.random.normal(0, 5)
        })

    return pd.DataFrame(data)


@pytest.fixture
def panel_with_slopes():
    """Create panel data suitable for random slopes (time variable)"""
    np.random.seed(42)

    data = []

    # Different time trends for different stores
    for store, base_sales, trend in [
        ('Store_A', 100, 2.0),   # Strong growth
        ('Store_B', 150, -1.0),  # Declining
        ('Store_C', 120, 0.5),   # Moderate growth
    ]:
        for time in range(1, 21):  # 20 time periods
            sales = base_sales + trend * time + np.random.normal(0, 5)
            data.append({
                'store_id': store,
                'time': time,
                'sales': sales
            })

    return pd.DataFrame(data)


@pytest.fixture
def categorical_groups():
    """Create panel with categorical (string) group identifiers"""
    np.random.seed(42)

    regions = ['North', 'South', 'East', 'West']
    data = []

    for region_idx, region in enumerate(regions):
        base_sales = 100 + region_idx * 20
        for obs in range(15):
            data.append({
                'region': region,
                'temperature': np.random.uniform(60, 90),
                'sales': base_sales + 0.5 * np.random.uniform(60, 90) + np.random.normal(0, 3)
            })

    return pd.DataFrame(data)


@pytest.fixture
def many_groups():
    """Create panel with many groups (50 groups)"""
    np.random.seed(42)

    data = []

    for group_id in range(50):
        base_value = 50 + group_id * 2
        for obs in range(5):
            data.append({
                'group_id': f'Group_{group_id:02d}',
                'x': np.random.uniform(0, 10),
                'y': base_value + 2 * np.random.uniform(0, 10) + np.random.normal(0, 2)
            })

    return pd.DataFrame(data)


# ============================================================================
# 1. Specification Tests (5 tests)
# ============================================================================

class TestPanelRegSpec:
    """Test panel_reg() model specification"""

    def test_default_spec(self):
        """Test default panel_reg specification"""
        spec = panel_reg()

        assert isinstance(spec, ModelSpec)
        assert spec.model_type == "panel_reg"
        assert spec.engine == "statsmodels"
        assert spec.mode == "regression"
        assert spec.args["intercept"] is True
        assert spec.args["random_effects"] == "intercept"

    def test_spec_with_random_slopes(self):
        """Test panel_reg with random slopes parameter"""
        spec = panel_reg(random_effects="both")

        assert spec.args["random_effects"] == "both"
        assert spec.args["intercept"] is True

    def test_spec_set_args_slope_var(self):
        """Test .set_args(slope_var='time')"""
        spec = panel_reg(random_effects="both")
        spec = spec.set_args(slope_var='time')

        assert spec.args["slope_var"] == "time"
        assert spec.args["random_effects"] == "both"

    def test_spec_immutability(self):
        """Test that ModelSpec is frozen/immutable"""
        spec1 = panel_reg(random_effects="intercept")
        spec2 = spec1.set_args(random_effects="both")

        # Original spec should be unchanged
        assert spec1.args["random_effects"] == "intercept"
        # New spec should have new value
        assert spec2.args["random_effects"] == "both"

    def test_spec_engine_setting(self):
        """Test .set_engine() method"""
        spec = panel_reg()
        spec = spec.set_engine("statsmodels")

        assert spec.engine == "statsmodels"


# ============================================================================
# 2. Fit Tests (10 tests)
# ============================================================================

class TestPanelRegFit:
    """Test panel_reg fitting with random effects"""

    def test_fit_random_intercept(self, balanced_panel):
        """Test fit with random intercepts only (default)"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        fit = wf.fit_global(balanced_panel, group_col='store_id')

        assert isinstance(fit.extract_fit_parsnip(), ModelFit)
        parsnip_fit = fit.extract_fit_parsnip()
        assert "model" in parsnip_fit.fit_data
        assert "random_effects" in parsnip_fit.fit_data
        assert "cov_re" in parsnip_fit.fit_data

    def test_fit_random_slope(self, panel_with_slopes):
        """Test fit with random intercepts + slopes"""
        spec = panel_reg(random_effects="both")
        spec = spec.set_args(slope_var='time')

        wf = workflow().add_formula("sales ~ time").add_model(spec)
        fit = wf.fit_global(panel_with_slopes, group_col='store_id')

        parsnip_fit = fit.extract_fit_parsnip()
        assert parsnip_fit.fit_data["random_effects_spec"] == "both"
        assert parsnip_fit.fit_data["slope_var"] == "time"

        # Random effects covariance should be 2x2 (intercept + slope)
        cov_re = parsnip_fit.fit_data["cov_re"]
        assert cov_re.shape == (2, 2)

    def test_fit_requires_group_col(self, balanced_panel):
        """Test error when group column missing"""
        spec = panel_reg()

        # Direct fit without group column should fail
        with pytest.raises(ValueError, match="requires original_training_data"):
            spec.fit(balanced_panel, "sales ~ price")

    def test_fit_requires_original_training_data(self, balanced_panel):
        """Test error when original_training_data is None"""
        spec = panel_reg()

        # This will fail during engine.fit() because original_training_data is None
        with pytest.raises(ValueError, match="requires original_training_data"):
            spec.fit(balanced_panel, "sales ~ price")

    def test_fit_requires_min_groups(self):
        """Test error with < 2 groups"""
        # Only 1 group
        data = pd.DataFrame({
            'store_id': ['Store_A'] * 10,
            'price': np.random.uniform(8, 12, 10),
            'sales': np.random.uniform(50, 150, 10)
        })

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            wf.fit_global(data, group_col='store_id')

    def test_fit_requires_min_obs_per_group(self):
        """Test error when group has < 2 observations"""
        # Store_C has only 1 observation
        data = pd.DataFrame({
            'store_id': ['Store_A', 'Store_A', 'Store_B', 'Store_B', 'Store_C'],
            'price': [10, 11, 9, 10, 12],
            'sales': [100, 105, 150, 155, 80]
        })

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        with pytest.raises(ValueError, match="at least 2 observations"):
            wf.fit_global(data, group_col='store_id')

    def test_fit_invalid_random_effects(self, balanced_panel):
        """Test error with invalid random_effects parameter"""
        spec = panel_reg(random_effects="invalid_option")
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        with pytest.raises(ValueError, match="Invalid random_effects"):
            wf.fit_global(balanced_panel, group_col='store_id')

    def test_fit_slope_var_required(self, panel_with_slopes):
        """Test error when random_effects='both' but slope_var missing"""
        spec = panel_reg(random_effects="both")
        # Don't set slope_var

        wf = workflow().add_formula("sales ~ time").add_model(spec)

        with pytest.raises(ValueError, match="requires slope_var parameter"):
            wf.fit_global(panel_with_slopes, group_col='store_id')

    def test_fit_slope_var_not_in_predictors(self, panel_with_slopes):
        """Test error when slope_var not in data"""
        spec = panel_reg(random_effects="both")
        spec = spec.set_args(slope_var='nonexistent_var')

        wf = workflow().add_formula("sales ~ time").add_model(spec)

        with pytest.raises(ValueError, match="not found in predictors"):
            wf.fit_global(panel_with_slopes, group_col='store_id')

    def test_fit_stores_random_effects(self, balanced_panel):
        """Test that fit_data contains random_effects dict and cov_re"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        parsnip_fit = fit.extract_fit_parsnip()

        # Check random effects dict
        assert "random_effects" in parsnip_fit.fit_data
        random_effects = parsnip_fit.fit_data["random_effects"]
        assert isinstance(random_effects, dict)
        assert len(random_effects) == 3  # 3 stores
        assert 'Store_A' in random_effects
        assert 'Store_B' in random_effects
        assert 'Store_C' in random_effects

        # Check covariance matrix
        assert "cov_re" in parsnip_fit.fit_data
        cov_re = parsnip_fit.fit_data["cov_re"]
        assert cov_re.shape[0] >= 1  # At least random intercept variance


# ============================================================================
# 3. Predict Tests (5 tests)
# ============================================================================

class TestPanelRegPredict:
    """Test panel_reg prediction"""

    def test_predict_training_groups(self, balanced_panel):
        """Test predictions for groups seen during training (uses fixed + random effects)"""
        # Split into train/test
        train = balanced_panel[balanced_panel['week'] <= 7]
        test = balanced_panel[balanced_panel['week'] > 7]

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(train, group_col='store_id')

        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test)
        assert all(predictions[".pred"].notna())

    def test_predict_new_groups(self, balanced_panel):
        """Test predictions for new groups (uses fixed effects only, population average)"""
        # Train on Store_A and Store_B only
        train = balanced_panel[balanced_panel['store_id'].isin(['Store_A', 'Store_B'])]

        # Test on Store_C (new group)
        test = balanced_panel[balanced_panel['store_id'] == 'Store_C'].copy()

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(train, group_col='store_id')

        # Should work - uses population average for new groups
        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test)
        assert all(predictions[".pred"].notna())

    def test_predict_with_random_slopes(self, panel_with_slopes):
        """Test predictions with random slopes model"""
        train = panel_with_slopes[panel_with_slopes['time'] <= 15]
        test = panel_with_slopes[panel_with_slopes['time'] > 15]

        spec = panel_reg(random_effects="both")
        spec = spec.set_args(slope_var='time')

        wf = workflow().add_formula("sales ~ time").add_model(spec)
        fit = wf.fit_global(train, group_col='store_id')

        predictions = fit.predict(test)

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert len(predictions) == len(test)

    def test_predict_conf_int(self, balanced_panel):
        """Test confidence intervals for predictions"""
        train = balanced_panel[balanced_panel['week'] <= 7]
        test = balanced_panel[balanced_panel['week'] > 7]

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(train, group_col='store_id')

        predictions = fit.predict(test, type="conf_int")

        assert isinstance(predictions, pd.DataFrame)
        assert ".pred" in predictions.columns
        assert ".pred_lower" in predictions.columns
        assert ".pred_upper" in predictions.columns

        # Verify interval structure
        assert all(predictions[".pred_lower"] <= predictions[".pred"])
        assert all(predictions[".pred"] <= predictions[".pred_upper"])

    def test_predict_invalid_type(self, balanced_panel):
        """Test error with unsupported prediction type"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        test = balanced_panel.iloc[:5]

        with pytest.raises(ValueError, match="supports type='numeric' or 'conf_int'"):
            fit.predict(test, type="prob")


# ============================================================================
# 4. Extract Outputs Tests (8 tests)
# ============================================================================

class TestPanelRegExtractOutputs:
    """Test extract_outputs() for panel regression"""

    def test_extract_outputs_structure(self, balanced_panel):
        """Test that extract_outputs returns three DataFrames"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        outputs, coefficients, stats = fit.extract_outputs()

        # All three should be DataFrames
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

    def test_extract_outputs_includes_group(self, balanced_panel):
        """Test that outputs DataFrame has group column"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        outputs, _, _ = fit.extract_outputs()

        # Check for group column
        assert "group" in outputs.columns

        # Check that all groups are present
        assert set(outputs["group"].unique()) == {'Store_A', 'Store_B', 'Store_C'}

    def test_extract_coefficients_fixed_effects(self, balanced_panel):
        """Test that coefficients DataFrame includes fixed effects"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        _, coefficients, _ = fit.extract_outputs()

        # Check for fixed effects
        fixed_coefs = coefficients[coefficients["type"] == "fixed"]
        assert len(fixed_coefs) >= 2  # At least Intercept + price

        # Check column structure
        assert "variable" in coefficients.columns
        assert "coefficient" in coefficients.columns
        assert "std_error" in coefficients.columns
        assert "t_stat" in coefficients.columns
        assert "p_value" in coefficients.columns
        assert "type" in coefficients.columns

        # Check that Intercept and price are in fixed effects
        variable_names = fixed_coefs["variable"].tolist()
        assert any("Intercept" in v or "const" in v for v in variable_names)
        assert "price" in variable_names

    def test_extract_coefficients_random_variance(self, balanced_panel):
        """Test that coefficients DataFrame includes random intercept variance"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        _, coefficients, _ = fit.extract_outputs()

        # Check for random effects variance components
        random_coefs = coefficients[coefficients["type"] == "random"]
        assert len(random_coefs) >= 1  # At least random intercept variance

        # Check for specific variance component
        assert any("RE: Intercept Variance" in v for v in random_coefs["variable"].tolist())

    def test_extract_coefficients_slope_variance(self, panel_with_slopes):
        """Test that coefficients DataFrame includes random slope variance components"""
        spec = panel_reg(random_effects="both")
        spec = spec.set_args(slope_var='time')

        wf = workflow().add_formula("sales ~ time").add_model(spec)
        fit = wf.fit_global(panel_with_slopes, group_col='store_id')

        _, coefficients, _ = fit.extract_outputs()

        # Check for random slope variance components
        random_coefs = coefficients[coefficients["type"] == "random"]

        variable_names = random_coefs["variable"].tolist()
        assert any("RE: Intercept Variance" in v for v in variable_names)
        assert any("RE: time Variance" in v for v in variable_names)
        assert any("RE: Cov(Intercept, time)" in v for v in variable_names)

    def test_extract_stats_panel_metrics(self, balanced_panel):
        """Test that stats DataFrame includes ICC and group statistics"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        _, _, stats = fit.extract_outputs()

        metric_names = stats["metric"].tolist()

        # Check for panel-specific metrics
        assert "icc" in metric_names
        assert "n_groups" in metric_names
        assert "min_group_size" in metric_names
        assert "max_group_size" in metric_names
        assert "mean_group_size" in metric_names

    def test_extract_stats_standard_metrics(self, balanced_panel):
        """Test that stats DataFrame includes RMSE, MAE, R²"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        _, _, stats = fit.extract_outputs()

        # Check for standard metrics
        train_stats = stats[stats["split"] == "train"]
        metric_names = train_stats["metric"].tolist()

        assert "rmse" in metric_names
        assert "mae" in metric_names
        assert "r_squared" in metric_names
        assert "adj_r_squared" in metric_names

    def test_extract_icc_calculation(self, balanced_panel):
        """Test that ICC is calculated correctly: var(RE) / (var(RE) + var(residual))"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        _, coefficients, stats = fit.extract_outputs()

        # Extract ICC from stats
        icc_row = stats[stats["metric"] == "icc"]
        assert len(icc_row) == 1
        icc = icc_row.iloc[0]["value"]

        # Extract variance components from coefficients
        re_intercept_var_row = coefficients[
            coefficients["variable"].str.contains("RE: Intercept Variance", na=False)
        ]
        residual_var_row = coefficients[
            coefficients["variable"] == "Residual Variance"
        ]

        assert len(re_intercept_var_row) == 1
        assert len(residual_var_row) == 1

        re_var = re_intercept_var_row.iloc[0]["coefficient"]
        resid_var = residual_var_row.iloc[0]["coefficient"]

        # Calculate expected ICC
        expected_icc = re_var / (re_var + resid_var)

        # Verify ICC is between 0 and 1
        assert 0 <= icc <= 1

        # Verify ICC matches calculation (with tolerance for numerical precision)
        assert abs(icc - expected_icc) < 1e-6


# ============================================================================
# 5. Integration Tests (5 tests)
# ============================================================================

class TestPanelRegIntegration:
    """Integration tests with workflows and WorkflowSets"""

    def test_workflow_fit_global_formula(self, balanced_panel):
        """Test panel_reg with workflow.fit_global() using formula"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        fit = wf.fit_global(balanced_panel, group_col='store_id')

        # Should work without errors
        assert fit is not None

        # Test prediction
        predictions = fit.predict(balanced_panel.iloc[:10])
        assert len(predictions) == 10

    def test_workflow_fit_global_recipe(self, balanced_panel):
        """Test panel_reg with recipe preprocessing"""
        # Note: Recipe doesn't support explicit formulas, it auto-generates them
        # The recipe will normalize all numeric columns
        rec = recipe().step_normalize(all_numeric())
        spec = panel_reg()

        # When using recipe, don't add formula - it will be auto-generated
        wf = workflow().add_recipe(rec).add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        # Should work with recipe preprocessing
        assert fit is not None

        # Test prediction
        predictions = fit.predict(balanced_panel.iloc[:10])
        assert len(predictions) == 10

    def test_workflowset_comparison(self, balanced_panel):
        """Test comparing panel_reg vs linear_reg in WorkflowSet"""
        from py_workflowsets import WorkflowSet

        formulas = ["sales ~ price"]
        models = [
            linear_reg(),
            panel_reg(),
        ]

        wf_set = WorkflowSet.from_cross(preproc=formulas, models=models)

        # Should create 2 workflows (1 formula × 2 models)
        assert len(wf_set.workflows) == 2

        # Fit both workflows globally
        results = wf_set.fit_global(balanced_panel, group_col='store_id')

        # Should have results for both workflows
        assert len(results.results) == 2

        # Check that both workflows succeeded
        assert all(r["fit"] is not None for r in results.results)

        # Check workflow IDs
        wflow_ids = [r["wflow_id"] for r in results.results]
        assert "prep_1_linear_reg_1" in wflow_ids
        assert "prep_1_panel_reg_2" in wflow_ids

    def test_workflow_predict_evaluate(self, balanced_panel):
        """Test full workflow: fit → predict → evaluate"""
        # Split data
        train = balanced_panel[balanced_panel['week'] <= 7]
        test = balanced_panel[balanced_panel['week'] > 7]

        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        # Fit
        fit = wf.fit_global(train, group_col='store_id')

        # Evaluate on test
        fit = fit.evaluate(test)

        # Extract outputs should show both train and test
        outputs, _, stats = fit.extract_outputs()

        assert "split" in outputs.columns
        assert "train" in outputs["split"].values
        assert "test" in outputs["split"].values

        # Stats should have both train and test metrics
        train_rmse = stats[(stats["metric"] == "rmse") & (stats["split"] == "train")]
        test_rmse = stats[(stats["metric"] == "rmse") & (stats["split"] == "test")]

        assert len(train_rmse) == 1
        assert len(test_rmse) == 1

    def test_workflow_extract_outputs(self, balanced_panel):
        """Test extract_outputs() through workflow"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)
        fit = wf.fit_global(balanced_panel, group_col='store_id')

        # Extract via workflow
        outputs, coefficients, stats = fit.extract_outputs()

        # Should return three DataFrames
        assert isinstance(outputs, pd.DataFrame)
        assert isinstance(coefficients, pd.DataFrame)
        assert isinstance(stats, pd.DataFrame)

        # Outputs should have group column
        assert "group" in outputs.columns

        # Coefficients should have fixed and random effects
        assert "fixed" in coefficients["type"].values
        assert "random" in coefficients["type"].values

        # Stats should have ICC
        assert "icc" in stats["metric"].values


# ============================================================================
# 6. Edge Case Tests (5 tests)
# ============================================================================

class TestPanelRegEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_unbalanced_panels(self, unbalanced_panel):
        """Test with different group sizes (unbalanced panel)"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ price").add_model(spec)

        fit = wf.fit_global(unbalanced_panel, group_col='store_id')

        # Should work with unbalanced panels
        assert fit is not None

        # Check group sizes in stats
        _, _, stats = fit.extract_outputs()

        min_size_row = stats[stats["metric"] == "min_group_size"]
        max_size_row = stats[stats["metric"] == "max_group_size"]

        assert len(min_size_row) == 1
        assert len(max_size_row) == 1

        # Min should be 6, max should be 12
        assert min_size_row.iloc[0]["value"] == 6
        assert max_size_row.iloc[0]["value"] == 12

    def test_categorical_groups(self, categorical_groups):
        """Test with categorical group column (strings, not integers)"""
        spec = panel_reg()
        wf = workflow().add_formula("sales ~ temperature").add_model(spec)

        fit = wf.fit_global(categorical_groups, group_col='region')

        # Should work with string group identifiers
        assert fit is not None

        # Extract random effects
        parsnip_fit = fit.extract_fit_parsnip()
        random_effects = parsnip_fit.fit_data["random_effects"]

        # Should have all 4 regions
        assert len(random_effects) == 4
        assert 'North' in random_effects
        assert 'South' in random_effects
        assert 'East' in random_effects
        assert 'West' in random_effects

    def test_many_groups(self, many_groups):
        """Test with many groups (50+ groups)"""
        spec = panel_reg()
        wf = workflow().add_formula("y ~ x").add_model(spec)

        fit = wf.fit_global(many_groups, group_col='group_id')

        # Should handle many groups
        assert fit is not None

        # Check n_groups in stats
        _, _, stats = fit.extract_outputs()
        n_groups_row = stats[stats["metric"] == "n_groups"]

        assert len(n_groups_row) == 1
        assert n_groups_row.iloc[0]["value"] == 50

    def test_small_within_group_variance(self):
        """Test when all variation is between groups (high ICC)"""
        np.random.seed(42)

        # Create data with very different group means but low within-group variance
        data = []
        for group_id, base_value in enumerate([50, 150, 250]):
            for obs in range(20):
                data.append({
                    'group': f'Group_{group_id}',
                    'x': np.random.uniform(0, 10),
                    'y': base_value + 0.1 * np.random.uniform(0, 10) + np.random.normal(0, 0.5)
                })

        df = pd.DataFrame(data)

        spec = panel_reg()
        wf = workflow().add_formula("y ~ x").add_model(spec)
        fit = wf.fit_global(df, group_col='group')

        # Extract ICC
        _, _, stats = fit.extract_outputs()
        icc_row = stats[stats["metric"] == "icc"]
        icc = icc_row.iloc[0]["value"]

        # ICC should be high (most variance is between groups)
        assert icc > 0.7  # Expect ICC > 0.7 for high between-group variance

    def test_no_between_group_variance(self):
        """Test when all variation is within groups (low ICC)"""
        np.random.seed(42)

        # Create data with same group means but high within-group variance
        data = []
        for group_id in range(3):
            for obs in range(20):
                data.append({
                    'group': f'Group_{group_id}',
                    'x': np.random.uniform(0, 10),
                    'y': 100 + 5 * np.random.uniform(0, 10) + np.random.normal(0, 10)
                })

        df = pd.DataFrame(data)

        spec = panel_reg()
        wf = workflow().add_formula("y ~ x").add_model(spec)
        fit = wf.fit_global(df, group_col='group')

        # Extract ICC
        _, _, stats = fit.extract_outputs()
        icc_row = stats[stats["metric"] == "icc"]
        icc = icc_row.iloc[0]["value"]

        # ICC should be low (most variance is within groups, not between)
        assert icc < 0.3  # Expect ICC < 0.3 for low between-group variance
