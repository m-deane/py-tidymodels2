"""
Specialized agents for Phase 2 multi-agent system.

Implements specialized agents with LLM-based reasoning:
- DataAnalyzer: Data profiling and pattern detection
- FeatureEngineer: Recipe optimization and feature engineering
- ModelSelector: Model recommendation with advanced reasoning
- Orchestrator: High-level planning and coordination
"""

import pandas as pd
from typing import Dict, List, Optional, Any

from py_agent.llm.client import LLMClient
from py_agent.llm.tool_schemas import (
    get_data_analyzer_tools,
    get_feature_engineer_tools,
    get_model_selector_tools
)
from py_agent.tools.data_analysis import analyze_temporal_patterns
from py_agent.tools.model_selection import suggest_model, get_model_profiles
from py_agent.tools.recipe_generation import create_recipe, get_recipe_templates


class BaseAgent:
    """
    Base class for specialized agents.

    Provides common functionality for LLM interaction and tool calling.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        role: str,
        tools: Dict[str, callable],
        tool_schemas: List[Dict]
    ):
        """
        Initialize base agent.

        Args:
            llm_client: LLM client for API calls
            role: Agent's role description
            tools: Dict mapping tool name to function
            tool_schemas: Tool schemas for LLM
        """
        self.llm_client = llm_client
        self.role = role
        self.tools = tools
        self.tool_schemas = tool_schemas

    def process(
        self,
        task: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a task using LLM reasoning and tools.

        Args:
            task: Task description
            context: Optional context information

        Returns:
            Dict with result and reasoning
        """
        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Build user message
        user_message = self._build_user_message(task, context)

        # Call LLM with tools
        response = self.llm_client.call_with_tools(
            messages=[{"role": "user", "content": user_message}],
            tools=self.tools,
            tool_definitions=self.tool_schemas,
            system=system_prompt
        )

        return {
            'result': response['content'],
            'tool_calls_made': len(response.get('tool_calls', [])),
            'reasoning': self._extract_reasoning(response)
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for this agent's role."""
        raise NotImplementedError("Subclasses must implement _build_system_prompt")

    def _build_user_message(self, task: str, context: Optional[Dict]) -> str:
        """Build user message with task and context."""
        message = f"Task: {task}\n"

        if context:
            message += f"\nContext:\n{self._format_context(context)}"

        return message

    def _format_context(self, context: Dict) -> str:
        """Format context dict as readable text."""
        lines = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                import json
                lines.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _extract_reasoning(self, response: Dict) -> str:
        """Extract reasoning from LLM response."""
        # Extract text content
        text_blocks = [
            block['text']
            for block in response['content']
            if block['type'] == 'text'
        ]
        return "\n".join(text_blocks)


class DataAnalyzer(BaseAgent):
    """
    Specialized agent for data analysis and pattern detection.

    Uses LLM reasoning to interpret data patterns and provide insights.

    Example:
        >>> analyzer = DataAnalyzer(llm_client)
        >>> result = analyzer.analyze(data, date_col='date', value_col='sales')
        >>> print(result['insights'])
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize DataAnalyzer agent."""
        tools = {
            'analyze_temporal_patterns': self._analyze_wrapper,
            'get_model_profiles': get_model_profiles
        }

        super().__init__(
            llm_client=llm_client,
            role="Data Analysis Expert",
            tools=tools,
            tool_schemas=get_data_analyzer_tools()
        )

        # Store data for tool access
        self._current_data = None

    def analyze(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> Dict[str, Any]:
        """
        Analyze time series data with LLM-enhanced insights.

        Args:
            data: DataFrame containing time series
            date_col: Name of date column
            value_col: Name of value column

        Returns:
            Dict with analysis results and LLM insights
        """
        self._current_data = data

        task = (
            f"Analyze the time series data to understand its characteristics. "
            f"The date column is '{date_col}' and the value column is '{value_col}'. "
            f"Provide comprehensive insights about patterns, seasonality, trends, "
            f"and any notable characteristics that would inform forecasting model selection."
        )

        context = {
            'n_observations': len(data),
            'date_range': f"{data[date_col].min()} to {data[date_col].max()}",
            'columns': data.columns.tolist()
        }

        result = self.process(task, context)

        return {
            'analysis': result['result'],
            'insights': result['reasoning'],
            'tool_calls': result['tool_calls_made']
        }

    def _analyze_wrapper(self, date_col: str, value_col: str) -> Dict:
        """Wrapper for analyze_temporal_patterns with current data."""
        if self._current_data is None:
            raise ValueError("No data available. Call analyze() first.")

        return analyze_temporal_patterns(
            self._current_data,
            date_col,
            value_col
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for DataAnalyzer."""
        return """You are an expert data analyst specializing in time series forecasting.

Your role is to:
1. Analyze temporal patterns in time series data
2. Identify seasonality, trends, and autocorrelation
3. Assess data quality (missing values, outliers)
4. Provide actionable insights for model selection

When analyzing data:
- Use the analyze_temporal_patterns tool to get quantitative analysis
- Interpret results in business context
- Highlight patterns relevant to forecasting
- Note any data quality concerns
- Suggest data characteristics that would inform model choice

Be precise, data-driven, and focus on patterns that impact forecast accuracy."""


class FeatureEngineer(BaseAgent):
    """
    Specialized agent for feature engineering and recipe optimization.

    Uses LLM reasoning to create optimal preprocessing pipelines.

    Example:
        >>> engineer = FeatureEngineer(llm_client)
        >>> result = engineer.create_recipe(data_chars, 'prophet_reg')
        >>> print(result['recipe_code'])
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize FeatureEngineer agent."""
        tools = {
            'create_recipe': create_recipe,
            'get_recipe_templates': get_recipe_templates
        }

        super().__init__(
            llm_client=llm_client,
            role="Feature Engineering Expert",
            tools=tools,
            tool_schemas=get_feature_engineer_tools()
        )

    def engineer_features(
        self,
        data_characteristics: Dict,
        model_type: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create optimized preprocessing recipe with LLM reasoning.

        Args:
            data_characteristics: Output from data analysis
            model_type: Target model type
            domain: Optional domain hint

        Returns:
            Dict with recipe code and reasoning
        """
        task = (
            f"Create an optimal preprocessing recipe for a {model_type} model. "
            f"Consider the data characteristics and any domain-specific requirements. "
            f"Explain your preprocessing strategy and why each step is included."
        )

        context = {
            'data_characteristics': data_characteristics,
            'model_type': model_type,
            'domain': domain
        }

        result = self.process(task, context)

        return {
            'recipe': result['result'],
            'reasoning': result['reasoning'],
            'tool_calls': result['tool_calls_made']
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for FeatureEngineer."""
        return """You are an expert feature engineer specializing in time series preprocessing.

Your role is to:
1. Design optimal preprocessing pipelines (recipes) for forecasting models
2. Select appropriate imputation strategies based on data quality
3. Create domain-specific features (retail, finance, energy)
4. Balance preprocessing complexity with model requirements

When creating recipes:
- Use create_recipe tool to generate code
- Consider data quality issues (missing values, outliers)
- Match preprocessing to model requirements:
  * Time series models (Prophet, ARIMA): minimal preprocessing
  * ML models (Random Forest, XGBoost): extensive feature engineering
- Include domain knowledge when available
- Explain reasoning for each preprocessing step

Be thoughtful about preprocessing choices - they significantly impact forecast quality."""


class ModelSelector(BaseAgent):
    """
    Specialized agent for model selection with advanced reasoning.

    Uses LLM to recommend models considering multiple factors beyond
    simple pattern matching.

    Example:
        >>> selector = ModelSelector(llm_client)
        >>> result = selector.select_model(data_chars, constraints)
        >>> print(result['recommendation'])
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize ModelSelector agent."""
        tools = {
            'suggest_model': suggest_model,
            'get_model_profiles': get_model_profiles
        }

        super().__init__(
            llm_client=llm_client,
            role="Model Selection Expert",
            tools=tools,
            tool_schemas=get_model_selector_tools()
        )

    def select_model(
        self,
        data_characteristics: Dict,
        constraints: Optional[Dict] = None,
        business_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select optimal model with LLM-enhanced reasoning.

        Args:
            data_characteristics: Output from data analysis
            constraints: Optional user constraints
            business_context: Optional business context

        Returns:
            Dict with model recommendation and detailed reasoning
        """
        task = (
            "Recommend the best forecasting model(s) for this data. "
            "Consider data patterns, constraints, and business context. "
            "Provide detailed reasoning for your recommendation including "
            "pros/cons and expected performance."
        )

        context = {
            'data_characteristics': data_characteristics,
            'constraints': constraints,
            'business_context': business_context
        }

        result = self.process(task, context)

        return {
            'recommendation': result['result'],
            'reasoning': result['reasoning'],
            'tool_calls': result['tool_calls_made']
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for ModelSelector."""
        return """You are an expert in time series forecasting model selection.

Your role is to:
1. Match data patterns to appropriate forecasting models
2. Consider user constraints (time, interpretability, resources)
3. Provide detailed reasoning for recommendations
4. Explain trade-offs between model choices

When selecting models:
- Use suggest_model tool to get quantitative recommendations
- Use get_model_profiles to understand model capabilities
- Consider multiple factors:
  * Data patterns (seasonality, trend, complexity)
  * Sample size and feature count
  * Computational constraints
  * Interpretability requirements
  * Business context and use case
- Recommend 2-3 models with clear pros/cons
- Explain expected performance and limitations

Be thorough in reasoning - model selection is critical for forecast success."""


class Orchestrator:
    """
    Orchestrator agent for coordinating specialized agents.

    Plans high-level workflow and delegates to specialized agents.

    Example:
        >>> orchestrator = Orchestrator(llm_client, agents)
        >>> workflow = orchestrator.generate_workflow(data, request)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        data_analyzer: DataAnalyzer,
        feature_engineer: FeatureEngineer,
        model_selector: ModelSelector
    ):
        """
        Initialize Orchestrator.

        Args:
            llm_client: LLM client
            data_analyzer: DataAnalyzer agent
            feature_engineer: FeatureEngineer agent
            model_selector: ModelSelector agent
        """
        self.llm_client = llm_client
        self.data_analyzer = data_analyzer
        self.feature_engineer = feature_engineer
        self.model_selector = model_selector

    def generate_workflow(
        self,
        data: pd.DataFrame,
        request: str,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate complete workflow by orchestrating specialized agents.

        Args:
            data: Input data
            request: User's natural language request
            constraints: Optional constraints

        Returns:
            Dict with workflow and detailed explanations
        """
        # Step 1: Analyze data
        analysis_result = self.data_analyzer.analyze(
            data,
            date_col=self._detect_date_col(data),
            value_col=self._detect_value_col(data)
        )

        # Step 2: Select model
        model_result = self.model_selector.select_model(
            data_characteristics=analysis_result['analysis'],
            constraints=constraints,
            business_context=request
        )

        # Step 3: Engineer features
        recipe_result = self.feature_engineer.engineer_features(
            data_characteristics=analysis_result['analysis'],
            model_type=model_result['recommendation'].get('model_type', 'prophet_reg')
        )

        return {
            'data_analysis': analysis_result,
            'model_selection': model_result,
            'feature_engineering': recipe_result,
            'workflow': self._build_workflow_code(model_result, recipe_result)
        }

    def _detect_date_col(self, data: pd.DataFrame) -> str:
        """Detect date column from data."""
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                return col
            if any(word in col.lower() for word in ['date', 'time']):
                return col
        return data.columns[0]

    def _detect_value_col(self, data: pd.DataFrame) -> str:
        """Detect value column from data."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if any(word in col.lower() for word in ['value', 'y', 'target', 'sales']):
                return col
        return numeric_cols[0] if len(numeric_cols) > 0 else data.columns[1]

    def _build_workflow_code(
        self,
        model_result: Dict,
        recipe_result: Dict
    ) -> str:
        """Build complete executable workflow code."""
        import json

        # Extract model type from recommendation
        recommendation = model_result.get('recommendation', {})
        if isinstance(recommendation, dict):
            model_type = recommendation.get('model_type', 'linear_reg')
        else:
            # If recommendation is string, try to parse it
            model_type = 'linear_reg'  # Default fallback

        # Extract recipe code
        recipe_code = recipe_result.get('recipe', '')

        # Build complete workflow
        workflow_code = f"""
# AI-Generated Workflow
# Generated by py_agent Orchestrator

from py_workflows import workflow
from py_parsnip import {model_type}
from py_recipes import recipe

# Create model specification
model_spec = {model_type}()

# Create preprocessing recipe
rec = {recipe_code if recipe_code else 'recipe()'}

# Build workflow
wf = workflow().add_recipe(rec).add_model(model_spec)

# Workflow is ready to fit
# Usage: fit = wf.fit(data)
#        predictions = fit.predict(new_data)
"""

        return workflow_code
