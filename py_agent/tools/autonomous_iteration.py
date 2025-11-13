"""
Autonomous workflow iteration system for py_agent.

This module enables the agent to iteratively improve workflows through:
- Try-evaluate-improve loops
- Self-debugging based on performance issues
- Automatic retry with different approaches
- Performance-based stopping criteria

Phase 3.5: Autonomous Iteration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from py_agent.tools.diagnostics import diagnose_performance


@dataclass
class IterationResult:
    """
    Result from a single iteration attempt.

    Attributes:
        iteration_num: Iteration number (1-indexed)
        workflow: The workflow object created
        fit: Fitted workflow (if successful)
        performance: Performance metrics dict (rmse, mae, r_squared, etc.)
        issues: List of detected issues
        approach: Description of approach used
        success: Whether iteration was successful
        error: Error message if failed
        duration: Time taken in seconds
    """
    iteration_num: int
    workflow: Optional[object]
    fit: Optional[object]
    performance: Dict[str, float]
    issues: List[Dict]
    approach: str
    success: bool
    error: Optional[str]
    duration: float


class IterationLoop:
    """
    Autonomous iteration loop for workflow improvement.

    Iteratively generates and evaluates workflows, improving based on
    performance feedback until target is met or max iterations reached.

    Example:
        >>> from py_agent import ForecastAgent
        >>> from py_agent.tools.autonomous_iteration import IterationLoop

        >>> agent = ForecastAgent(verbose=True)
        >>> loop = IterationLoop(
        ...     agent=agent,
        ...     max_iterations=5,
        ...     target_metric='rmse',
        ...     target_value=10.0  # Stop when RMSE < 10
        ... )

        >>> best_workflow, history = loop.iterate(
        ...     data=train_data,
        ...     request="Forecast sales with seasonality",
        ...     test_data=test_data
        ... )
    """

    def __init__(
        self,
        agent: 'ForecastAgent',
        max_iterations: int = 5,
        target_metric: str = 'rmse',
        target_value: Optional[float] = None,
        improvement_threshold: float = 0.05,
        verbose: bool = True
    ):
        """
        Initialize iteration loop.

        Args:
            agent: ForecastAgent instance
            max_iterations: Maximum number of iterations (default: 5)
            target_metric: Metric to optimize ('rmse', 'mae', 'r_squared')
            target_value: Optional target value to achieve (stops when reached)
            improvement_threshold: Minimum relative improvement to continue (default: 5%)
            verbose: Whether to print progress messages
        """
        self.agent = agent
        self.max_iterations = max_iterations
        self.target_metric = target_metric
        self.target_value = target_value
        self.improvement_threshold = improvement_threshold
        self.verbose = verbose
        self.history: List[IterationResult] = []

        # Track what's been tried
        self.tried_models = set()
        self.tried_approaches = set()

    def iterate(
        self,
        data: pd.DataFrame,
        request: str,
        test_data: Optional[pd.DataFrame] = None,
        formula: Optional[str] = None,
        constraints: Optional[Dict] = None
    ) -> Tuple[object, List[IterationResult]]:
        """
        Iteratively improve workflow until target or max iterations.

        Args:
            data: Training data
            request: Natural language request
            test_data: Optional test data for evaluation
            formula: Optional explicit formula
            constraints: Optional constraints dict

        Returns:
            (best_workflow, iteration_history) tuple
            - best_workflow: Best workflow found
            - iteration_history: List of IterationResult objects
        """
        self._log("🔄 Starting autonomous iteration loop...")
        self._log(f"   Target: {self.target_metric} {'<' if self.target_metric != 'r_squared' else '>'} {self.target_value if self.target_value else 'N/A'}")
        self._log(f"   Max iterations: {self.max_iterations}")

        best_workflow = None
        best_performance = float('inf') if self.target_metric != 'r_squared' else 0.0

        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n{'='*60}")
            self._log(f"Iteration {iteration}/{self.max_iterations}")
            self._log(f"{'='*60}")

            # Generate workflow for this iteration
            result = self._try_iteration(
                iteration_num=iteration,
                data=data,
                request=request,
                test_data=test_data,
                formula=formula,
                constraints=constraints,
                previous_results=self.history
            )

            self.history.append(result)

            # Check if this is the best so far
            if result.success:
                current_metric = result.performance.get(self.target_metric, float('inf'))

                is_better = self._is_better_performance(current_metric, best_performance)

                if is_better:
                    self._log(f"\n✨ New best {self.target_metric}: {current_metric:.4f}")
                    best_workflow = result.fit
                    best_performance = current_metric

                # Check stopping criteria
                if self._should_stop(current_metric, best_performance, iteration):
                    self._log(f"\n✅ Stopping: {self._get_stop_reason(current_metric, iteration)}")
                    break
            else:
                self._log(f"\n⚠️  Iteration {iteration} failed: {result.error}")

        # Summary
        self._log(f"\n{'='*60}")
        self._log("Iteration Summary")
        self._log(f"{'='*60}")
        self._log(f"Iterations completed: {len(self.history)}")
        self._log(f"Successful iterations: {sum(1 for r in self.history if r.success)}")

        if best_workflow:
            self._log(f"Best {self.target_metric}: {best_performance:.4f}")
            self._log(f"Best approach: {[r.approach for r in self.history if r.success and r.performance.get(self.target_metric) == best_performance][0]}")
        else:
            self._log("❌ No successful workflows generated")

        return best_workflow, self.history

    def _try_iteration(
        self,
        iteration_num: int,
        data: pd.DataFrame,
        request: str,
        test_data: Optional[pd.DataFrame],
        formula: Optional[str],
        constraints: Optional[Dict],
        previous_results: List[IterationResult]
    ) -> IterationResult:
        """Try a single iteration with error handling."""
        start_time = time.time()

        try:
            # Determine approach for this iteration
            approach = self._determine_approach(iteration_num, previous_results)
            self._log(f"\n🎯 Approach: {approach}")

            # Generate workflow
            workflow = self._generate_workflow(
                data=data,
                request=request,
                formula=formula,
                constraints=constraints,
                approach=approach,
                iteration_num=iteration_num
            )

            # Fit workflow
            self._log("   Fitting workflow...")
            fit = workflow.fit(data)

            # Evaluate performance
            if test_data is not None:
                self._log("   Evaluating on test data...")
                fit_eval = fit.evaluate(test_data)
                outputs, coeffs, stats = fit_eval.extract_outputs()
            else:
                outputs, coeffs, stats = fit.extract_outputs()

            # Get performance metrics
            test_stats = stats[stats['split'] == 'test'] if 'split' in stats.columns and test_data is not None else stats
            performance = {
                'rmse': float(test_stats['rmse'].iloc[0]) if 'rmse' in test_stats.columns else float('inf'),
                'mae': float(test_stats['mae'].iloc[0]) if 'mae' in test_stats.columns else float('inf'),
                'r_squared': float(test_stats['r_squared'].iloc[0]) if 'r_squared' in test_stats.columns else 0.0,
            }

            self._log(f"   RMSE: {performance['rmse']:.4f}")
            self._log(f"   MAE: {performance['mae']:.4f}")
            self._log(f"   R²: {performance['r_squared']:.4f}")

            # Diagnose issues
            issues = self._diagnose_issues(fit, test_data, performance)
            if issues:
                self._log(f"\n⚠️  Issues detected: {len(issues)}")
                for issue in issues[:3]:  # Show top 3
                    self._log(f"   • {issue['type']}: {issue['evidence']}")

            duration = time.time() - start_time

            return IterationResult(
                iteration_num=iteration_num,
                workflow=workflow,
                fit=fit,
                performance=performance,
                issues=issues,
                approach=approach,
                success=True,
                error=None,
                duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self._log(f"   ❌ Error: {str(e)}")

            return IterationResult(
                iteration_num=iteration_num,
                workflow=None,
                fit=None,
                performance={},
                issues=[],
                approach=approach if 'approach' in locals() else "unknown",
                success=False,
                error=str(e),
                duration=duration
            )

    def _determine_approach(
        self,
        iteration_num: int,
        previous_results: List[IterationResult]
    ) -> str:
        """
        Determine what approach to try for this iteration.

        Iteration 1: Use agent's default recommendation
        Iteration 2+: Try different approaches based on previous issues
        """
        if iteration_num == 1:
            return "default_recommendation"

        # Analyze previous failures and issues
        if len(previous_results) == 0:
            return "default_recommendation"

        last_result = previous_results[-1]

        if not last_result.success:
            return "retry_with_simpler_model"

        # Check for specific issues
        if last_result.issues:
            issue_types = [issue['type'] for issue in last_result.issues]

            if 'overfitting' in issue_types:
                return "regularization_or_simpler_model"
            elif 'underfitting' in issue_types or 'high_bias' in issue_types:
                return "more_complex_model_or_features"
            elif 'high_variance' in issue_types:
                return "increase_regularization"
            elif 'residual_autocorrelation' in issue_types:
                return "add_lag_features_or_arima"
            elif 'heteroscedasticity' in issue_types:
                return "log_transform_or_weighted_regression"

        # Try different models
        if iteration_num == 2:
            return "try_tree_based_model"
        elif iteration_num == 3:
            return "try_time_series_model"
        elif iteration_num == 4:
            return "try_ensemble_or_boosting"
        else:
            return "try_advanced_preprocessing"

    def _generate_workflow(
        self,
        data: pd.DataFrame,
        request: str,
        formula: Optional[str],
        constraints: Optional[Dict],
        approach: str,
        iteration_num: int
    ) -> object:
        """Generate workflow based on approach."""
        # Modify constraints based on approach
        adjusted_constraints = constraints.copy() if constraints else {}

        if approach == "retry_with_simpler_model":
            adjusted_constraints['prefer_simple'] = True
            adjusted_constraints['interpretability'] = 'high'
        elif approach == "regularization_or_simpler_model":
            adjusted_constraints['regularization'] = 'high'
        elif approach == "more_complex_model_or_features":
            adjusted_constraints['max_complexity'] = 'high'
        elif approach == "try_tree_based_model":
            # Will be handled by agent's recommendation system
            pass

        # Generate workflow
        workflow = self.agent.generate_workflow(
            data=data,
            request=request,
            formula=formula,
            constraints=adjusted_constraints
        )

        return workflow

    def _diagnose_issues(
        self,
        fit: object,
        test_data: Optional[pd.DataFrame],
        performance: Dict[str, float]
    ) -> List[Dict]:
        """Diagnose performance issues."""
        try:
            diagnostics = diagnose_performance(fit, test_data)
            return diagnostics.get('issues_detected', [])
        except Exception:
            # If diagnostics fail, return empty list
            return []

    def _is_better_performance(self, current: float, best: float) -> bool:
        """Check if current performance is better than best."""
        if self.target_metric == 'r_squared':
            return current > best
        else:
            # For error metrics (rmse, mae), lower is better
            return current < best

    def _should_stop(
        self,
        current_metric: float,
        best_metric: float,
        iteration: int
    ) -> bool:
        """Check if iteration should stop."""
        # Stop if target value reached
        if self.target_value is not None:
            if self.target_metric == 'r_squared':
                if current_metric >= self.target_value:
                    return True
            else:
                if current_metric <= self.target_value:
                    return True

        # Stop if no improvement
        if iteration > 1 and len(self.history) >= 2:
            prev_best = self._get_best_metric_from_history(self.history[:-1])
            if prev_best is not None:
                if self.target_metric == 'r_squared':
                    improvement = (best_metric - prev_best) / (prev_best + 1e-8)
                else:
                    improvement = (prev_best - best_metric) / (prev_best + 1e-8)

                if improvement < self.improvement_threshold:
                    return True

        return False

    def _get_best_metric_from_history(
        self,
        history: List[IterationResult]
    ) -> Optional[float]:
        """Get best metric value from history."""
        successful = [r for r in history if r.success and self.target_metric in r.performance]
        if not successful:
            return None

        if self.target_metric == 'r_squared':
            return max(r.performance[self.target_metric] for r in successful)
        else:
            return min(r.performance[self.target_metric] for r in successful)

    def _get_stop_reason(self, current_metric: float, iteration: int) -> str:
        """Get reason for stopping."""
        if self.target_value is not None:
            if self.target_metric == 'r_squared':
                if current_metric >= self.target_value:
                    return f"Target R² of {self.target_value:.2f} reached"
            else:
                if current_metric <= self.target_value:
                    return f"Target {self.target_metric} of {self.target_value:.2f} reached"

        if iteration >= self.max_iterations:
            return "Maximum iterations reached"

        return "No significant improvement"

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(message)


def iterate_until_target(
    agent: 'ForecastAgent',
    data: pd.DataFrame,
    request: str,
    target_metric: str = 'rmse',
    target_value: float = None,
    max_iterations: int = 5,
    test_data: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    constraints: Optional[Dict] = None
) -> Tuple[object, List[IterationResult]]:
    """
    Convenience function for autonomous iteration.

    Args:
        agent: ForecastAgent instance
        data: Training data
        request: Natural language request
        target_metric: Metric to optimize ('rmse', 'mae', 'r_squared')
        target_value: Target value to achieve (stops when reached)
        max_iterations: Maximum iterations (default: 5)
        test_data: Optional test data for evaluation
        formula: Optional explicit formula
        constraints: Optional constraints dict

    Returns:
        (best_workflow, iteration_history) tuple

    Example:
        >>> from py_agent import ForecastAgent
        >>> from py_agent.tools.autonomous_iteration import iterate_until_target

        >>> agent = ForecastAgent()
        >>> best_wf, history = iterate_until_target(
        ...     agent=agent,
        ...     data=train_data,
        ...     request="Forecast sales",
        ...     target_metric='rmse',
        ...     target_value=10.0,
        ...     max_iterations=5,
        ...     test_data=test_data
        ... )
    """
    loop = IterationLoop(
        agent=agent,
        max_iterations=max_iterations,
        target_metric=target_metric,
        target_value=target_value
    )

    return loop.iterate(
        data=data,
        request=request,
        test_data=test_data,
        formula=formula,
        constraints=constraints
    )
