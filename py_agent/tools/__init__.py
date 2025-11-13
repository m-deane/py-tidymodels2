"""
Tools for AI agent to analyze data and make recommendations.

This module provides functions that the LLM agent can call to:
- Analyze temporal patterns in time series data
- Detect seasonality and trends
- Suggest appropriate models based on data characteristics
- Generate preprocessing recipes
- Execute and evaluate workflows
- Diagnose performance issues
"""

from py_agent.tools.data_analysis import (
    analyze_temporal_patterns,
    detect_seasonality,
    detect_trend,
    calculate_autocorrelation
)
from py_agent.tools.model_selection import suggest_model, get_model_profiles
from py_agent.tools.recipe_generation import create_recipe, get_recipe_templates
from py_agent.tools.workflow_execution import fit_workflow, evaluate_workflow
from py_agent.tools.diagnostics import diagnose_performance, detect_overfitting

__all__ = [
    # Data analysis
    "analyze_temporal_patterns",
    "detect_seasonality",
    "detect_trend",
    "calculate_autocorrelation",
    # Model selection
    "suggest_model",
    "get_model_profiles",
    # Recipe generation
    "create_recipe",
    "get_recipe_templates",
    # Workflow execution
    "fit_workflow",
    "evaluate_workflow",
    # Diagnostics
    "diagnose_performance",
    "detect_overfitting",
]
