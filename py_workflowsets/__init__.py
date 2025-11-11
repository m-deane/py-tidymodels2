"""
py-workflowsets: Multi-model comparison for py-tidymodels

Provides WorkflowSet for comparing multiple workflows across different
preprocessing strategies and model specifications.
"""

from py_workflowsets.workflowset import (
    WorkflowSet,
    WorkflowSetResults,
    WorkflowSetNestedResults,
)

__all__ = [
    "WorkflowSet",
    "WorkflowSetResults",
    "WorkflowSetNestedResults",
]

__version__ = "0.1.0"
