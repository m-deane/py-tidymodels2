"""
py_agent: AI-powered forecasting agent for py-tidymodels

This package provides LLM-based agents that can automatically generate,
optimize, and debug time series forecasting workflows.

Main components:
- tools: Core analysis and recommendation functions
- agents: LLM agent implementations
"""

from py_agent.agents.forecast_agent import ForecastAgent

__version__ = "0.1.0"
__all__ = ["ForecastAgent"]
