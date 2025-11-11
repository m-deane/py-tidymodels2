"""
Tests for NestedWorkflowFit extract methods.

Tests the extract_formula(), extract_spec_parsnip(), extract_preprocessor(),
and extract_fit_parsnip() methods on NestedWorkflowFit.
"""

import pytest
import pandas as pd
import numpy as np
from py_workflows import workflow
from py_recipes import recipe
from py_parsnip import linear_reg, ModelSpec, ModelFit
from py_recipes import PreparedRecipe


@pytest.fixture
def sample_grouped_data():
    """Create sample grouped data for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'group': ['A'] * 50 + ['B'] * 50,
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randn(100)
    })
    return data


def test_extract_formula_returns_dict(sample_grouped_data):
    """Test that extract_formula() returns a dictionary."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    formulas = nested_fit.extract_formula()

    assert isinstance(formulas, dict)
    assert len(formulas) == 2
    assert 'A' in formulas
    assert 'B' in formulas


def test_extract_formula_with_explicit_formula(sample_grouped_data):
    """Test extract_formula() returns correct formula for each group."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    formulas = nested_fit.extract_formula()

    assert formulas['A'] == "y ~ x1 + x2"
    assert formulas['B'] == "y ~ x1 + x2"

    # Check that all groups use same formula
    assert len(set(formulas.values())) == 1


def test_extract_formula_with_recipe(sample_grouped_data):
    """Test extract_formula() with recipe (auto-generated formula)."""
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    formulas = nested_fit.extract_formula()

    # Should have auto-generated formulas
    assert 'A' in formulas
    assert 'B' in formulas
    # Auto-generated formulas should include y, x1, x2 but not date, group
    for group, formula in formulas.items():
        assert 'y' in formula
        assert '~' in formula
        assert 'x1' in formula
        assert 'x2' in formula
        assert 'date' not in formula
        assert 'group' not in formula


def test_extract_spec_parsnip_returns_modelspec(sample_grouped_data):
    """Test that extract_spec_parsnip() returns ModelSpec."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    spec = nested_fit.extract_spec_parsnip()

    assert isinstance(spec, ModelSpec)
    assert spec.model_type == "linear_reg"
    assert spec.mode == "regression"


def test_extract_spec_parsnip_shared_across_groups(sample_grouped_data):
    """Test that extract_spec_parsnip() returns same spec for all groups."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    spec = nested_fit.extract_spec_parsnip()

    # Verify it's the same spec used by individual groups
    group_a_fit = nested_fit.group_fits['A']
    group_a_spec = group_a_fit.extract_spec_parsnip()

    assert spec.model_type == group_a_spec.model_type
    assert spec.engine == group_a_spec.engine


def test_extract_preprocessor_no_group_returns_dict(sample_grouped_data):
    """Test extract_preprocessor() without group argument returns dict."""
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    preprocessors = nested_fit.extract_preprocessor()

    assert isinstance(preprocessors, dict)
    assert len(preprocessors) == 2
    assert 'A' in preprocessors
    assert 'B' in preprocessors


def test_extract_preprocessor_with_group_returns_single(sample_grouped_data):
    """Test extract_preprocessor(group='A') returns single preprocessor."""
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    prep_a = nested_fit.extract_preprocessor(group='A')

    assert isinstance(prep_a, PreparedRecipe)


def test_extract_preprocessor_with_formula(sample_grouped_data):
    """Test extract_preprocessor() with formula returns formula string."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    preprocessors = nested_fit.extract_preprocessor()

    # With formula, preprocessors should be strings
    assert isinstance(preprocessors['A'], str)
    assert isinstance(preprocessors['B'], str)
    assert preprocessors['A'] == "y ~ x1 + x2"


def test_extract_preprocessor_invalid_group_raises_error(sample_grouped_data):
    """Test extract_preprocessor() with invalid group raises ValueError."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    with pytest.raises(ValueError, match="Group 'C' not found"):
        nested_fit.extract_preprocessor(group='C')


def test_extract_preprocessor_with_per_group_prep(sample_grouped_data):
    """Test extract_preprocessor() with per-group preprocessing."""
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group', per_group_prep=True)

    preprocessors = nested_fit.extract_preprocessor()

    # Each group should have its own PreparedRecipe
    assert isinstance(preprocessors['A'], PreparedRecipe)
    assert isinstance(preprocessors['B'], PreparedRecipe)
    # They should be different objects (different instances)
    assert preprocessors['A'] is not preprocessors['B']


def test_extract_fit_parsnip_no_group_returns_dict(sample_grouped_data):
    """Test extract_fit_parsnip() without group argument returns dict."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    model_fits = nested_fit.extract_fit_parsnip()

    assert isinstance(model_fits, dict)
    assert len(model_fits) == 2
    assert 'A' in model_fits
    assert 'B' in model_fits


def test_extract_fit_parsnip_with_group_returns_single(sample_grouped_data):
    """Test extract_fit_parsnip(group='A') returns single ModelFit."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    fit_a = nested_fit.extract_fit_parsnip(group='A')

    assert isinstance(fit_a, ModelFit)
    assert fit_a.spec.model_type == "linear_reg"


def test_extract_fit_parsnip_invalid_group_raises_error(sample_grouped_data):
    """Test extract_fit_parsnip() with invalid group raises ValueError."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    with pytest.raises(ValueError, match="Group 'C' not found"):
        nested_fit.extract_fit_parsnip(group='C')


def test_extract_fit_parsnip_can_extract_outputs(sample_grouped_data):
    """Test that extracted ModelFit can call extract_outputs()."""
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    fit_a = nested_fit.extract_fit_parsnip(group='A')
    outputs, coeffs, stats = fit_a.extract_outputs()

    assert isinstance(outputs, pd.DataFrame)
    assert isinstance(coeffs, pd.DataFrame)
    assert isinstance(stats, pd.DataFrame)
    assert len(outputs) > 0


def test_all_extract_methods_work_together(sample_grouped_data):
    """Test that all extract methods work together cohesively."""
    rec = recipe().step_normalize()
    wf = workflow().add_recipe(rec).add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group', per_group_prep=True)

    # Extract formula for all groups
    formulas = nested_fit.extract_formula()
    assert len(formulas) == 2

    # Extract shared spec
    spec = nested_fit.extract_spec_parsnip()
    assert spec.model_type == "linear_reg"

    # Extract preprocessor for specific group
    prep_a = nested_fit.extract_preprocessor(group='A')
    assert isinstance(prep_a, PreparedRecipe)

    # Extract all preprocessors
    all_preps = nested_fit.extract_preprocessor()
    assert len(all_preps) == 2

    # Extract ModelFit for specific group
    fit_a = nested_fit.extract_fit_parsnip(group='A')
    assert isinstance(fit_a, ModelFit)

    # Extract all ModelFits
    all_fits = nested_fit.extract_fit_parsnip()
    assert len(all_fits) == 2


def test_extract_methods_with_three_groups(sample_grouped_data):
    """Test extract methods with more than two groups."""
    # Add a third group
    np.random.seed(42)
    extra_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=50),
        'group': ['C'] * 50,
        'x1': np.random.randn(50),
        'x2': np.random.randn(50),
        'y': np.random.randn(50)
    })
    data = pd.concat([sample_grouped_data, extra_data], ignore_index=True)

    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(data, group_col='group')

    # Test all extract methods
    formulas = nested_fit.extract_formula()
    assert len(formulas) == 3
    assert set(formulas.keys()) == {'A', 'B', 'C'}

    preprocessors = nested_fit.extract_preprocessor()
    assert len(preprocessors) == 3

    model_fits = nested_fit.extract_fit_parsnip()
    assert len(model_fits) == 3


def test_extract_methods_work_after_evaluate(sample_grouped_data):
    """Test that extract methods still work after calling evaluate()."""
    train = sample_grouped_data[:80]
    test = sample_grouped_data[80:]

    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(train, group_col='group')
    nested_fit = nested_fit.evaluate(test)

    # All extract methods should still work
    formulas = nested_fit.extract_formula()
    assert len(formulas) == 2

    spec = nested_fit.extract_spec_parsnip()
    assert spec.model_type == "linear_reg"

    preprocessors = nested_fit.extract_preprocessor()
    assert len(preprocessors) == 2

    model_fits = nested_fit.extract_fit_parsnip()
    assert len(model_fits) == 2


def test_extract_formula_shows_different_formulas_if_different(sample_grouped_data):
    """Test that extract_formula() shows differences if groups have different formulas."""
    # This is a theoretical test - in current implementation all groups get same formula
    # But this tests the dict structure supports it
    wf = workflow().add_formula("y ~ x1 + x2").add_model(linear_reg())
    nested_fit = wf.fit_nested(sample_grouped_data, group_col='group')

    formulas = nested_fit.extract_formula()

    # Get unique formulas
    unique_formulas = set(formulas.values())

    # Currently all groups use same formula
    assert len(unique_formulas) == 1

    # But the dict structure supports different formulas per group
    assert isinstance(formulas, dict)
    for group, formula in formulas.items():
        assert isinstance(formula, str)
        assert '~' in formula
