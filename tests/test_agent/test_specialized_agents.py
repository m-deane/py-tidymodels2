"""
Tests for specialized agents (Phase 2).

Tests the multi-agent system with LLM-based reasoning.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from py_agent.agents.specialized_agents import (
    BaseAgent,
    DataAnalyzer,
    FeatureEngineer,
    ModelSelector,
    Orchestrator
)
from py_agent.llm.client import LLMClient


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    values = (
        np.sin(np.arange(365) * 2 * np.pi / 7) * 10 +  # Weekly seasonality
        np.arange(365) * 0.1 +  # Trend
        np.random.randn(365) * 2  # Noise
    )
    return pd.DataFrame({'date': dates, 'sales': values})


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=LLMClient)
    client.call_with_tools = Mock(return_value={
        'content': [{'type': 'text', 'text': 'Analysis result'}],
        'tool_calls': [{'name': 'analyze_temporal_patterns'}],
    })
    return client


class TestBaseAgent:
    """Tests for BaseAgent base class."""

    def test_init(self, mock_llm_client):
        """Test BaseAgent initialization."""
        agent = BaseAgent(
            llm_client=mock_llm_client,
            role="Test Agent",
            tools={'test_tool': lambda x: x},
            tool_schemas=[{'name': 'test_tool'}]
        )

        assert agent.llm_client == mock_llm_client
        assert agent.role == "Test Agent"
        assert 'test_tool' in agent.tools
        assert len(agent.tool_schemas) == 1

    def test_format_context(self, mock_llm_client):
        """Test context formatting."""
        agent = BaseAgent(
            llm_client=mock_llm_client,
            role="Test",
            tools={},
            tool_schemas=[]
        )

        context = {
            'simple': 'value',
            'nested': {'key': 'val'},
            'list': [1, 2, 3]
        }

        formatted = agent._format_context(context)

        assert 'simple: value' in formatted
        assert 'nested:' in formatted
        assert 'list:' in formatted

    def test_extract_reasoning(self, mock_llm_client):
        """Test reasoning extraction from LLM response."""
        agent = BaseAgent(
            llm_client=mock_llm_client,
            role="Test",
            tools={},
            tool_schemas=[]
        )

        response = {
            'content': [
                {'type': 'text', 'text': 'First thought'},
                {'type': 'tool_use', 'name': 'test'},
                {'type': 'text', 'text': 'Second thought'}
            ]
        }

        reasoning = agent._extract_reasoning(response)

        assert 'First thought' in reasoning
        assert 'Second thought' in reasoning


class TestDataAnalyzer:
    """Tests for DataAnalyzer agent."""

    def test_init(self, mock_llm_client):
        """Test DataAnalyzer initialization."""
        analyzer = DataAnalyzer(mock_llm_client)

        assert analyzer.role == "Data Analysis Expert"
        assert 'analyze_temporal_patterns' in analyzer.tools
        assert 'get_model_profiles' in analyzer.tools
        assert len(analyzer.tool_schemas) > 0

    def test_analyze(self, mock_llm_client, sample_data):
        """Test data analysis with LLM."""
        # Mock LLM response
        mock_llm_client.call_with_tools.return_value = {
            'content': [
                {'type': 'text', 'text': 'The data shows strong weekly seasonality and increasing trend.'}
            ],
            'tool_calls': [{'name': 'analyze_temporal_patterns', 'result': {}}]
        }

        analyzer = DataAnalyzer(mock_llm_client)

        result = analyzer.analyze(sample_data, 'date', 'sales')

        # Check result structure
        assert 'analysis' in result
        assert 'insights' in result
        assert 'tool_calls' in result

        # Check LLM was called
        assert mock_llm_client.call_with_tools.called

    def test_analyze_wrapper(self, mock_llm_client, sample_data):
        """Test analyze_temporal_patterns wrapper."""
        analyzer = DataAnalyzer(mock_llm_client)
        analyzer._current_data = sample_data

        result = analyzer._analyze_wrapper('date', 'sales')

        # Should return data analysis dict
        assert 'frequency' in result
        assert 'seasonality' in result
        assert 'trend' in result

    def test_analyze_wrapper_no_data(self, mock_llm_client):
        """Test wrapper fails when no data set."""
        analyzer = DataAnalyzer(mock_llm_client)

        with pytest.raises(ValueError, match="No data available"):
            analyzer._analyze_wrapper('date', 'value')

    def test_system_prompt(self, mock_llm_client):
        """Test DataAnalyzer has proper system prompt."""
        analyzer = DataAnalyzer(mock_llm_client)

        prompt = analyzer._build_system_prompt()

        assert 'data analyst' in prompt.lower()
        assert 'time series' in prompt.lower()
        assert 'seasonality' in prompt.lower()
        assert 'trend' in prompt.lower()


class TestFeatureEngineer:
    """Tests for FeatureEngineer agent."""

    def test_init(self, mock_llm_client):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(mock_llm_client)

        assert engineer.role == "Feature Engineering Expert"
        assert 'create_recipe' in engineer.tools
        assert 'get_recipe_templates' in engineer.tools

    def test_engineer_features(self, mock_llm_client):
        """Test feature engineering with LLM."""
        mock_llm_client.call_with_tools.return_value = {
            'content': [
                {'type': 'text', 'text': 'Created preprocessing recipe with normalization and dummy encoding.'}
            ],
            'tool_calls': [{'name': 'create_recipe'}]
        }

        engineer = FeatureEngineer(mock_llm_client)

        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.8},
            'missing_rate': 0.05
        }

        result = engineer.engineer_features(
            data_characteristics=data_chars,
            model_type='prophet_reg'
        )

        assert 'recipe' in result
        assert 'reasoning' in result
        assert 'tool_calls' in result

    def test_system_prompt(self, mock_llm_client):
        """Test FeatureEngineer has proper system prompt."""
        engineer = FeatureEngineer(mock_llm_client)

        prompt = engineer._build_system_prompt()

        assert 'feature engineer' in prompt.lower()
        assert 'preprocessing' in prompt.lower()
        assert 'recipe' in prompt.lower()


class TestModelSelector:
    """Tests for ModelSelector agent."""

    def test_init(self, mock_llm_client):
        """Test ModelSelector initialization."""
        selector = ModelSelector(mock_llm_client)

        assert selector.role == "Model Selection Expert"
        assert 'suggest_model' in selector.tools
        assert 'get_model_profiles' in selector.tools

    def test_select_model(self, mock_llm_client):
        """Test model selection with LLM."""
        mock_llm_client.call_with_tools.return_value = {
            'content': [
                {'type': 'text', 'text': 'Recommend prophet_reg due to strong seasonality.'}
            ],
            'tool_calls': [{'name': 'suggest_model'}]
        }

        selector = ModelSelector(mock_llm_client)

        data_chars = {
            'frequency': 'daily',
            'seasonality': {'detected': True, 'strength': 0.9},
            'n_observations': 730
        }

        result = selector.select_model(data_characteristics=data_chars)

        assert 'recommendation' in result
        assert 'reasoning' in result
        assert 'tool_calls' in result

    def test_select_model_with_constraints(self, mock_llm_client):
        """Test model selection with constraints."""
        mock_llm_client.call_with_tools.return_value = {
            'content': [{'type': 'text', 'text': 'Selected linear_reg for high interpretability.'}],
            'tool_calls': []
        }

        selector = ModelSelector(mock_llm_client)

        data_chars = {'frequency': 'daily', 'n_observations': 365}
        constraints = {'interpretability': 'high', 'max_train_time': 10}

        result = selector.select_model(
            data_characteristics=data_chars,
            constraints=constraints
        )

        # Check LLM received constraints in context
        call_args = mock_llm_client.call_with_tools.call_args
        assert call_args is not None

    def test_system_prompt(self, mock_llm_client):
        """Test ModelSelector has proper system prompt."""
        selector = ModelSelector(mock_llm_client)

        prompt = selector._build_system_prompt()

        assert 'model selection' in prompt.lower()
        assert 'forecasting' in prompt.lower()
        assert 'constraints' in prompt.lower()


class TestOrchestrator:
    """Tests for Orchestrator coordination agent."""

    def test_init(self, mock_llm_client):
        """Test Orchestrator initialization."""
        analyzer = DataAnalyzer(mock_llm_client)
        engineer = FeatureEngineer(mock_llm_client)
        selector = ModelSelector(mock_llm_client)

        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            data_analyzer=analyzer,
            feature_engineer=engineer,
            model_selector=selector
        )

        assert orchestrator.data_analyzer == analyzer
        assert orchestrator.feature_engineer == engineer
        assert orchestrator.model_selector == selector

    @patch.object(DataAnalyzer, 'analyze')
    @patch.object(ModelSelector, 'select_model')
    @patch.object(FeatureEngineer, 'engineer_features')
    def test_generate_workflow(
        self,
        mock_engineer,
        mock_selector,
        mock_analyzer,
        mock_llm_client,
        sample_data
    ):
        """Test workflow generation orchestration."""
        # Mock agent responses
        mock_analyzer.return_value = {
            'analysis': {'frequency': 'daily', 'seasonality': {'detected': True}},
            'insights': 'Strong weekly pattern',
            'tool_calls': 1
        }

        mock_selector.return_value = {
            'recommendation': {'model_type': 'prophet_reg'},
            'reasoning': 'Good for seasonal data',
            'tool_calls': 1
        }

        mock_engineer.return_value = {
            'recipe': "recipe().step_normalize()",
            'reasoning': 'Normalize features',
            'tool_calls': 1
        }

        # Create orchestrator
        analyzer = DataAnalyzer(mock_llm_client)
        engineer = FeatureEngineer(mock_llm_client)
        selector = ModelSelector(mock_llm_client)

        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            data_analyzer=analyzer,
            feature_engineer=engineer,
            model_selector=selector
        )

        # Generate workflow
        result = orchestrator.generate_workflow(
            data=sample_data,
            request="Forecast sales with weekly seasonality"
        )

        # Check result structure
        assert 'data_analysis' in result
        assert 'model_selection' in result
        assert 'feature_engineering' in result
        assert 'workflow' in result

        # Check workflow code is generated
        assert 'from py_workflows import workflow' in result['workflow']
        assert 'from py_parsnip import' in result['workflow']

    def test_detect_date_col(self, mock_llm_client, sample_data):
        """Test automatic date column detection."""
        analyzer = DataAnalyzer(mock_llm_client)
        engineer = FeatureEngineer(mock_llm_client)
        selector = ModelSelector(mock_llm_client)

        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            data_analyzer=analyzer,
            feature_engineer=engineer,
            model_selector=selector
        )

        date_col = orchestrator._detect_date_col(sample_data)

        assert date_col == 'date'

    def test_detect_value_col(self, mock_llm_client, sample_data):
        """Test automatic value column detection."""
        analyzer = DataAnalyzer(mock_llm_client)
        engineer = FeatureEngineer(mock_llm_client)
        selector = ModelSelector(mock_llm_client)

        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            data_analyzer=analyzer,
            feature_engineer=engineer,
            model_selector=selector
        )

        value_col = orchestrator._detect_value_col(sample_data)

        assert value_col == 'sales'

    def test_build_workflow_code(self, mock_llm_client):
        """Test workflow code generation."""
        analyzer = DataAnalyzer(mock_llm_client)
        engineer = FeatureEngineer(mock_llm_client)
        selector = ModelSelector(mock_llm_client)

        orchestrator = Orchestrator(
            llm_client=mock_llm_client,
            data_analyzer=analyzer,
            feature_engineer=engineer,
            model_selector=selector
        )

        model_result = {
            'recommendation': {'model_type': 'linear_reg'},
            'reasoning': 'Test'
        }

        recipe_result = {
            'recipe': "recipe().step_normalize()",
            'reasoning': 'Test'
        }

        workflow_code = orchestrator._build_workflow_code(model_result, recipe_result)

        # Check code structure
        assert 'from py_workflows import workflow' in workflow_code
        assert 'from py_parsnip import linear_reg' in workflow_code
        assert 'from py_recipes import recipe' in workflow_code
        assert 'model_spec = linear_reg()' in workflow_code
        assert 'step_normalize' in workflow_code


def test_integration_orchestrator_workflow():
    """Integration test: Full orchestration workflow."""
    # Skip if no API key
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No API key for integration test")

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.sin(np.arange(100) * 2 * np.pi / 7) * 10 + 50
    data = pd.DataFrame({'date': dates, 'sales': values})

    # Create LLM client with low budget
    llm_client = LLMClient(budget_per_day=1.0, max_tokens=500)

    # Create specialized agents
    analyzer = DataAnalyzer(llm_client)
    engineer = FeatureEngineer(llm_client)
    selector = ModelSelector(llm_client)

    # Create orchestrator
    orchestrator = Orchestrator(
        llm_client=llm_client,
        data_analyzer=analyzer,
        feature_engineer=engineer,
        model_selector=selector
    )

    # This would make real API calls - skip for now
    # result = orchestrator.generate_workflow(
    #     data=data,
    #     request="Forecast sales"
    # )
