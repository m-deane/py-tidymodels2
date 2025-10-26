"""
py-workflows: Workflow composition for py-tidymodels

Provides tools for composing preprocessing and models into complete pipelines.

Main Components:
    workflow(): Create a new workflow
    Workflow: Immutable workflow specification
    WorkflowFit: Fitted workflow with trained model

Example:
    >>> from py_workflows import workflow
    >>> from py_parsnip import linear_reg
    >>>
    >>> # Create and fit workflow
    >>> wf = (
    ...     workflow()
    ...     .add_formula("sales ~ price + advertising")
    ...     .add_model(linear_reg().set_engine("sklearn"))
    ... )
    >>> wf_fit = wf.fit(train_data)
    >>>
    >>> # Predict and evaluate
    >>> predictions = wf_fit.predict(test_data)
    >>> wf_fit = wf_fit.evaluate(test_data)
    >>> outputs, coefficients, stats = wf_fit.extract_outputs()
"""

from py_workflows.workflow import workflow, Workflow, WorkflowFit

__all__ = [
    "workflow",
    "Workflow",
    "WorkflowFit",
]

__version__ = "0.1.0"
