"""
py_stacks: Model ensembling via stacking/meta-learning

Provides tools for combining multiple models using elastic net regularization.
"""

from .stacks import stacks, Stacks, BlendedStack

__all__ = [
    "stacks",
    "Stacks",
    "BlendedStack",
]

__version__ = "0.1.0"
