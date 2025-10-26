"""
Simple tests for py-parsnip without fixtures

These tests verify core functionality without complex pytest fixtures.
"""

import pandas as pd
import numpy as np

from py_parsnip import linear_reg


def test_linear_reg_spec_creation():
    """Test basic spec creation"""
    spec = linear_reg()
    assert spec.model_type == "linear_reg"
    assert spec.engine == "sklearn"


def test_linear_reg_fit_simple():
    """Test simple fit"""
    train = pd.DataFrame({
        "y": [100, 200, 150],
        "x": [10, 20, 15],
    })

    spec = linear_reg()
    fit = spec.fit(train, "y ~ x")

    assert fit is not None
    assert fit.spec == spec


def test_linear_reg_predict_simple():
    """Test simple prediction"""
    train = pd.DataFrame({
        "y": [100, 200, 150],
        "x": [10, 20, 15],
    })

    spec = linear_reg()
    fit = spec.fit(train, "y ~ x")

    test = pd.DataFrame({"x": [12, 18]})
    predictions = fit.predict(test)

    assert len(predictions) == 2
    assert ".pred" in predictions.columns


def test_linear_reg_with_penalty():
    """Test Ridge regression"""
    train = pd.DataFrame({
        "y": [100, 200, 150, 300, 250],
        "x1": [10, 20, 15, 30, 25],
        "x2": [5, 10, 7, 15, 12],
    })

    spec = linear_reg(penalty=0.1, mixture=0.0)
    fit = spec.fit(train, "y ~ x1 + x2")

    assert fit.fit_data["model_class"] == "Ridge"


def test_full_workflow():
    """Test complete workflow"""
    train = pd.DataFrame({
        "sales": [100, 200, 150, 300, 250],
        "price": [10, 20, 15, 30, 25],
        "advertising": [5, 10, 7, 15, 12],
    })

    spec = linear_reg()
    fit = spec.fit(train, "sales ~ price + advertising")

    test = pd.DataFrame({
        "price": [12, 22],
        "advertising": [6, 11],
    })

    predictions = fit.predict(test)

    assert len(predictions) == 2
    assert all(predictions[".pred"] > 0)
