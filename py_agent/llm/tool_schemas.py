"""
Tool schemas for LLM tool calling.

Defines the interface between LLM and py_agent tools in Anthropic's format.
"""

from typing import Dict, List


def get_tool_schemas() -> List[Dict]:
    """
    Get tool schemas for all available py_agent tools.

    Returns tool definitions in Anthropic's tool calling format.

    Returns:
        List of tool schema dictionaries

    Example:
        >>> schemas = get_tool_schemas()
        >>> client.call(messages, tools=schemas)
    """
    return [
        {
            "name": "analyze_temporal_patterns",
            "description": (
                "Analyze temporal patterns in time series data. Detects frequency, "
                "seasonality, trend, autocorrelation, and data quality issues. "
                "This is the primary analysis function to understand data characteristics."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date_col": {
                        "type": "string",
                        "description": "Name of the date/time column in the data"
                    },
                    "value_col": {
                        "type": "string",
                        "description": "Name of the value column to analyze (e.g., 'sales', 'demand')"
                    }
                },
                "required": ["date_col", "value_col"]
            }
        },
        {
            "name": "suggest_model",
            "description": (
                "Recommend appropriate forecasting models based on data characteristics "
                "and user constraints. Returns ranked list of models with confidence scores "
                "and reasoning. Use after analyzing data patterns."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_characteristics": {
                        "type": "object",
                        "description": "Output from analyze_temporal_patterns function"
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional constraints dict with max_train_time, interpretability, max_memory",
                        "properties": {
                            "max_train_time": {
                                "type": "number",
                                "description": "Maximum training time in seconds"
                            },
                            "interpretability": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Required interpretability level"
                            },
                            "max_memory": {
                                "type": "number",
                                "description": "Maximum memory usage in MB"
                            }
                        }
                    }
                },
                "required": ["data_characteristics"]
            }
        },
        {
            "name": "create_recipe",
            "description": (
                "Generate preprocessing recipe code based on data characteristics, "
                "model requirements, and optional domain knowledge. Returns executable "
                "Python code for py_recipes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_characteristics": {
                        "type": "object",
                        "description": "Output from analyze_temporal_patterns function"
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Target model type (e.g., 'prophet_reg', 'linear_reg', 'rand_forest')"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["retail", "finance", "energy"],
                        "description": "Optional domain hint for specialized preprocessing"
                    }
                },
                "required": ["data_characteristics", "model_type"]
            }
        },
        {
            "name": "diagnose_performance",
            "description": (
                "Analyze model performance and identify issues like overfitting, "
                "data quality problems, or model complexity issues. Returns detected "
                "issues with severity levels and actionable recommendations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_fit": {
                        "type": "string",
                        "description": "Reference to fitted model object (use 'current_fit')"
                    },
                    "test_data_available": {
                        "type": "boolean",
                        "description": "Whether test data is available for overfitting detection"
                    }
                },
                "required": ["model_fit"]
            }
        },
        {
            "name": "get_model_profiles",
            "description": (
                "Get detailed profiles for all supported models including training time, "
                "interpretability, accuracy tier, strengths, and weaknesses. Use to "
                "understand model capabilities."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_recipe_templates",
            "description": (
                "Get predefined recipe templates for common forecasting scenarios. "
                "Includes templates for retail, energy, finance, and general time series."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


def get_data_analyzer_tools() -> List[Dict]:
    """
    Get tools for DataAnalyzer specialized agent.

    Returns:
        Tool schemas for data analysis tasks
    """
    return [
        schema for schema in get_tool_schemas()
        if schema['name'] in [
            'analyze_temporal_patterns',
            'get_model_profiles'
        ]
    ]


def get_feature_engineer_tools() -> List[Dict]:
    """
    Get tools for FeatureEngineer specialized agent.

    Returns:
        Tool schemas for feature engineering tasks
    """
    return [
        schema for schema in get_tool_schemas()
        if schema['name'] in [
            'create_recipe',
            'get_recipe_templates'
        ]
    ]


def get_model_selector_tools() -> List[Dict]:
    """
    Get tools for ModelSelector specialized agent.

    Returns:
        Tool schemas for model selection tasks
    """
    return [
        schema for schema in get_tool_schemas()
        if schema['name'] in [
            'suggest_model',
            'get_model_profiles'
        ]
    ]


def get_debugger_tools() -> List[Dict]:
    """
    Get tools for Debugger specialized agent.

    Returns:
        Tool schemas for debugging tasks
    """
    return [
        schema for schema in get_tool_schemas()
        if schema['name'] in [
            'diagnose_performance'
        ]
    ]
