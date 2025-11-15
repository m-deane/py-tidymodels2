"""
py_interpret: Model interpretability using SHAP

Provides comprehensive model interpretability framework using SHAP
(SHapley Additive exPlanations) for variable-level contribution analysis.

Main components:
- ShapEngine: Core SHAP computation engine
- Integration with ModelFit, WorkflowFit, and NestedWorkflowFit via explain() method
"""

from py_interpret.shap_engine import ShapEngine

__all__ = ["ShapEngine"]
