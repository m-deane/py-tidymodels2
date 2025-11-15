"""
py_interpret: Model interpretability using SHAP

Provides comprehensive model interpretability framework using SHAP
(SHapley Additive exPlanations) for variable-level contribution analysis.

Main components:
- ShapEngine: Core SHAP computation engine
- Visualization functions: summary_plot, waterfall_plot, force_plot, dependence_plot, temporal_plot
- Integration with ModelFit, WorkflowFit, and NestedWorkflowFit via explain() and explain_plot() methods
"""

from py_interpret.shap_engine import ShapEngine
from py_interpret.visualizations import (
    summary_plot,
    waterfall_plot,
    force_plot,
    dependence_plot,
    temporal_plot
)

__all__ = [
    "ShapEngine",
    "summary_plot",
    "waterfall_plot",
    "force_plot",
    "dependence_plot",
    "temporal_plot"
]
