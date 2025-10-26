"""
py-hardhat: Internal preprocessing layer for py-tidymodels

This package provides the low-level formula → matrix conversion layer
that sits between recipes (user-facing feature engineering) and models.

Key functions:
- mold(): Convert formula + data → model-ready format with Blueprint
- forge(): Apply Blueprint to new data for prediction

The Blueprint ensures consistent data structure between training and prediction.
"""

from py_hardhat.blueprint import Blueprint, MoldedData
from py_hardhat.mold import mold
from py_hardhat.forge import forge

__all__ = ["Blueprint", "MoldedData", "mold", "forge"]
__version__ = "0.1.0"
