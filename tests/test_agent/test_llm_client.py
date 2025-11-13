"""
Tests for LLM client wrapper.

Tests the Anthropic SDK integration, tool calling, and budget management.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from py_agent.llm.client import LLMClient


class TestLLMClientInit:
    """Tests for LLM client initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        # Skip if no API key (CI environment)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("No API key available")

        client = LLMClient()

        assert client.model == "claude-sonnet-4.5"
        assert client.max_tokens == 4096
        assert client.temperature == 0.7
        assert client.budget_per_day == 100.0
        assert client.use_cache == True

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("No API key available")

        client = LLMClient(
            model="claude-sonnet-4",
            max_tokens=2048,
            temperature=0.5,
            budget_per_day=50.0,
            use_cache=False
        )

        assert client.model == "claude-sonnet-4"
        assert client.max_tokens == 2048
        assert client.temperature == 0.5
        assert client.budget_per_day == 50.0
        assert client.use_cache == False

    def test_init_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                LLMClient()


class TestLLMClientBudget:
    """Tests for budget management."""

    @patch('anthropic.Anthropic')
    def test_budget_tracking(self, mock_anthropic):
        """Test that budget is tracked correctly."""
        # Mock API response
        mock_response = Mock()
        mock_response.content = [Mock(type='text', text='Test response')]
        mock_response.usage = Mock(
            input_tokens=100,
            output_tokens=50
        )
        mock_anthropic.return_value.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(budget_per_day=10.0)

            # Make a call
            client.call([{"role": "user", "content": "test"}])

            # Check budget tracking
            assert client.total_cost > 0
            assert client.usage_stats['total_requests'] == 1
            assert client.usage_stats['total_input_tokens'] == 100
            assert client.usage_stats['total_output_tokens'] == 50

    @patch('anthropic.Anthropic')
    def test_budget_exceeded(self, mock_anthropic):
        """Test that calls fail when budget is exceeded."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(budget_per_day=0.001)  # Very low budget

            # Manually set high cost
            client.total_cost = 10.0

            # Should raise error
            with pytest.raises(RuntimeError, match="Budget exceeded"):
                client.call([{"role": "user", "content": "test"}])


class TestLLMClientToolCalling:
    """Tests for tool calling functionality."""

    @patch('anthropic.Anthropic')
    def test_call_with_tools_success(self, mock_anthropic):
        """Test successful tool calling with execution."""
        # Mock tool function
        def test_tool(param1: str) -> dict:
            return {"result": f"processed {param1}"}

        tools = {"test_tool": test_tool}
        tool_definitions = [{
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        }]

        # Mock API responses
        # First response: tool call
        mock_tool_call = Mock()
        mock_tool_call.type = 'tool_use'
        mock_tool_call.id = 'tool_1'
        mock_tool_call.name = 'test_tool'
        mock_tool_call.input = {"param1": "test_value"}

        response1 = Mock()
        response1.content = [mock_tool_call]
        response1.usage = Mock(input_tokens=50, output_tokens=25)
        response1.stop_reason = 'tool_use'

        # Second response: final answer
        response2 = Mock()
        response2.content = [Mock(type='text', text='Final answer')]
        response2.usage = Mock(input_tokens=75, output_tokens=30)
        response2.stop_reason = 'end_turn'

        mock_anthropic.return_value.messages.create.side_effect = [response1, response2]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient()

            result = client.call_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=tools,
                tool_definitions=tool_definitions
            )

            # Check result
            assert 'content' in result
            assert len(result['tool_calls']) == 1
            assert result['tool_calls'][0]['name'] == 'test_tool'

    @patch('anthropic.Anthropic')
    def test_call_with_tools_max_iterations(self, mock_anthropic):
        """Test that tool calling stops after max iterations."""
        def test_tool(param1: str) -> dict:
            return {"result": "test"}

        tools = {"test_tool": test_tool}
        tool_definitions = [{
            "name": "test_tool",
            "description": "Test tool",
            "input_schema": {
                "type": "object",
                "properties": {"param1": {"type": "string"}},
                "required": ["param1"]
            }
        }]

        # Mock infinite tool calls
        mock_tool_call = Mock()
        mock_tool_call.type = 'tool_use'
        mock_tool_call.id = 'tool_1'
        mock_tool_call.name = 'test_tool'
        mock_tool_call.input = {"param1": "test"}

        response = Mock()
        response.content = [mock_tool_call]
        response.usage = Mock(input_tokens=50, output_tokens=25)
        response.stop_reason = 'tool_use'

        mock_anthropic.return_value.messages.create.return_value = response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient()

            with pytest.raises(RuntimeError, match="Maximum iterations"):
                client.call_with_tools(
                    messages=[{"role": "user", "content": "test"}],
                    tools=tools,
                    tool_definitions=tool_definitions,
                    max_iterations=3
                )


class TestLLMClientRetry:
    """Tests for retry logic."""

    @patch('anthropic.Anthropic')
    def test_retry_on_rate_limit(self, mock_anthropic):
        """Test retry on rate limit error."""
        from anthropic import RateLimitError

        # First two calls fail, third succeeds
        success_response = Mock()
        success_response.content = [Mock(type='text', text='Success')]
        success_response.usage = Mock(input_tokens=50, output_tokens=25)

        mock_anthropic.return_value.messages.create.side_effect = [
            RateLimitError("Rate limited", response=Mock(), body={}),
            RateLimitError("Rate limited", response=Mock(), body={}),
            success_response
        ]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('time.sleep'):  # Don't actually sleep in tests
                client = LLMClient()

                result = client.call(
                    messages=[{"role": "user", "content": "test"}],
                    max_retries=3
                )

                # Should succeed on third try
                assert result['content'][0]['text'] == 'Success'

    @patch('anthropic.Anthropic')
    def test_retry_exhausted(self, mock_anthropic):
        """Test that retry eventually fails."""
        from anthropic import RateLimitError

        # All calls fail
        mock_anthropic.return_value.messages.create.side_effect = RateLimitError(
            "Rate limited",
            response=Mock(),
            body={}
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('time.sleep'):
                client = LLMClient()

                with pytest.raises(RateLimitError):
                    client.call(
                        messages=[{"role": "user", "content": "test"}],
                        max_retries=2
                    )


class TestLLMClientCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation_sonnet_4_5(self):
        """Test cost calculation for Claude Sonnet 4.5."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(model="claude-sonnet-4.5")

            cost = client._calculate_cost(1000, 500)

            # Sonnet 4.5: $3/MTok input, $15/MTok output
            expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
            assert abs(cost - expected) < 0.000001

    def test_cost_calculation_sonnet_4(self):
        """Test cost calculation for Claude Sonnet 4."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(model="claude-sonnet-4")

            cost = client._calculate_cost(1000, 500)

            # Sonnet 4: $3/MTok input, $15/MTok output
            expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
            assert abs(cost - expected) < 0.000001

    def test_cost_calculation_opus(self):
        """Test cost calculation for Claude Opus."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(model="claude-opus-4")

            cost = client._calculate_cost(1000, 500)

            # Opus 4: $15/MTok input, $75/MTok output
            expected = (1000 * 15.0 / 1_000_000) + (500 * 75.0 / 1_000_000)
            assert abs(cost - expected) < 0.000001


def test_integration_simple_call():
    """Integration test: Simple LLM call."""
    # Only run if API key is available
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No API key available for integration test")

    client = LLMClient(max_tokens=100)

    result = client.call([
        {"role": "user", "content": "Say 'test successful' and nothing else."}
    ])

    assert 'content' in result
    assert len(result['content']) > 0
    assert result['content'][0]['type'] == 'text'
    assert 'test successful' in result['content'][0]['text'].lower()
