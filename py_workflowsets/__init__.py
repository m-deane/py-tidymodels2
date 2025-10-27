"""
py-workflowsets: Multi-model comparison for py-tidymodels

Provides WorkflowSet for comparing multiple workflows across different
preprocessing strategies and model specifications.
"""

from py_workflowsets.workflowset import WorkflowSet, WorkflowSetResults

__all__ = [
    "WorkflowSet",
    "WorkflowSetResults",
]

__version__ = "0.1.0"
