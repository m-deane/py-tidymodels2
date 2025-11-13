"""
LLM client wrapper for Anthropic Claude API.

Provides a unified interface for LLM interactions with:
- Tool calling support
- Caching for cost optimization
- Token budget management
- Error handling and retries
"""

import os
import json
from typing import Dict, List, Optional, Any, Callable
import time


class LLMClient:
    """
    Client for interacting with Claude API via Anthropic SDK.

    Handles tool calling, caching, and cost management for the AI agent.

    Example:
        >>> client = LLMClient(api_key="sk-...")
        >>> response = client.call(
        ...     messages=[{"role": "user", "content": "Analyze this data"}],
        ...     tools=data_analysis_tools
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4.5",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        budget_per_day: float = 100.0,
        use_cache: bool = True
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env)
            model: Model to use (claude-sonnet-4.5, claude-opus-4, etc.)
            max_tokens: Maximum tokens per request
            temperature: Sampling temperature (0-1)
            budget_per_day: Daily budget in USD
            use_cache: Whether to use prompt caching
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.budget_per_day = budget_per_day
        self.use_cache = use_cache

        # Initialize Anthropic client (lazy import)
        self._client = None

        # Track usage
        self.total_cost = 0.0
        self.request_count = 0
        self.token_usage = {
            'input': 0,
            'output': 0,
            'cache_write': 0,
            'cache_read': 0
        }

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for Phase 2. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def call(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        system: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make LLM API call with tool calling support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            system: Optional system prompt
            max_retries: Number of retries on failure

        Returns:
            Dict with response content and tool calls

        Example:
            >>> response = client.call(
            ...     messages=[{"role": "user", "content": "Analyze sales data"}],
            ...     tools=[{
            ...         "name": "analyze_data",
            ...         "description": "Analyze temporal patterns",
            ...         "input_schema": {...}
            ...     }]
            ... )
        """
        # Check budget
        if self.total_cost >= self.budget_per_day:
            raise BudgetExceededError(
                f"Daily budget of ${self.budget_per_day} exceeded. "
                f"Current cost: ${self.total_cost:.2f}"
            )

        # Prepare request
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }

        if system:
            request_params["system"] = system

        if tools:
            request_params["tools"] = tools

        # Retry logic
        for attempt in range(max_retries):
            try:
                # Make API call
                response = self.client.messages.create(**request_params)

                # Update usage tracking
                self._update_usage(response)

                # Parse response
                parsed = self._parse_response(response)

                return parsed

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise LLMError(f"LLM call failed after {max_retries} attempts: {e}")

    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: Dict[str, Callable],
        tool_definitions: List[Dict],
        system: Optional[str] = None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Make LLM call with automatic tool execution loop.

        Handles the tool calling loop:
        1. LLM returns tool calls
        2. Execute tools
        3. Send results back to LLM
        4. Repeat until LLM returns final answer

        Args:
            messages: Initial messages
            tools: Dict mapping tool name to function
            tool_definitions: Tool schemas for LLM
            system: System prompt
            max_iterations: Maximum tool calling iterations

        Returns:
            Final response from LLM

        Example:
            >>> def analyze_data(data):
            ...     return {"seasonality": True}
            >>>
            >>> response = client.call_with_tools(
            ...     messages=[{"role": "user", "content": "Analyze this"}],
            ...     tools={"analyze_data": analyze_data},
            ...     tool_definitions=[...]
            ... )
        """
        conversation = messages.copy()

        for iteration in range(max_iterations):
            # Call LLM
            response = self.call(
                messages=conversation,
                tools=tool_definitions,
                system=system
            )

            # Check if LLM wants to use tools
            if not response.get('tool_calls'):
                # No more tool calls, return final answer
                return response

            # Execute tool calls
            tool_results = []
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['input']

                if tool_name not in tools:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # Execute tool
                try:
                    result = tools[tool_name](**tool_args)
                    tool_results.append({
                        'tool_use_id': tool_call['id'],
                        'content': json.dumps(result)
                    })
                except Exception as e:
                    tool_results.append({
                        'tool_use_id': tool_call['id'],
                        'content': json.dumps({'error': str(e)}),
                        'is_error': True
                    })

            # Add assistant's tool calls to conversation
            conversation.append({
                'role': 'assistant',
                'content': response['content']
            })

            # Add tool results to conversation
            conversation.append({
                'role': 'user',
                'content': tool_results
            })

        raise MaxIterationsError(
            f"Tool calling loop exceeded {max_iterations} iterations"
        )

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse Anthropic API response."""
        parsed = {
            'content': [],
            'tool_calls': [],
            'stop_reason': response.stop_reason
        }

        for block in response.content:
            if block.type == 'text':
                parsed['content'].append({
                    'type': 'text',
                    'text': block.text
                })
            elif block.type == 'tool_use':
                parsed['tool_calls'].append({
                    'id': block.id,
                    'name': block.name,
                    'input': block.input
                })

        return parsed

    def _update_usage(self, response):
        """Update token usage and cost tracking."""
        usage = response.usage

        # Update token counts
        self.token_usage['input'] += usage.input_tokens
        self.token_usage['output'] += usage.output_tokens

        # Update cache tokens if available
        if hasattr(usage, 'cache_creation_input_tokens'):
            self.token_usage['cache_write'] += usage.cache_creation_input_tokens
        if hasattr(usage, 'cache_read_input_tokens'):
            self.token_usage['cache_read'] += usage.cache_read_input_tokens

        # Calculate cost (Claude Sonnet 4.5 pricing)
        cost = 0.0
        cost += (usage.input_tokens / 1_000_000) * 3.0  # $3 per 1M input
        cost += (usage.output_tokens / 1_000_000) * 15.0  # $15 per 1M output

        if hasattr(usage, 'cache_creation_input_tokens'):
            cost += (usage.cache_creation_input_tokens / 1_000_000) * 3.75  # Cache write
        if hasattr(usage, 'cache_read_input_tokens'):
            cost += (usage.cache_read_input_tokens / 1_000_000) * 0.30  # Cache read

        self.total_cost += cost
        self.request_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_cost': self.total_cost,
            'request_count': self.request_count,
            'tokens': self.token_usage,
            'avg_cost_per_request': self.total_cost / max(self.request_count, 1)
        }

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_cost = 0.0
        self.request_count = 0
        self.token_usage = {
            'input': 0,
            'output': 0,
            'cache_write': 0,
            'cache_read': 0
        }


class BudgetExceededError(Exception):
    """Raised when daily budget is exceeded."""
    pass


class LLMError(Exception):
    """Raised when LLM call fails."""
    pass


class MaxIterationsError(Exception):
    """Raised when tool calling loop exceeds max iterations."""
    pass
