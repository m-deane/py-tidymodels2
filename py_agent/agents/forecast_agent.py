"""
ForecastAgent: AI-powered forecasting workflow generator.

This agent analyzes data, recommends models, generates preprocessing recipes,
and creates complete py-tidymodels workflows through natural language interaction.

MVP Implementation (v0.1.0):
- Rule-based workflow generation
- Supports 3 model types: linear_reg, prophet_reg, rand_forest
- Basic recipe generation
- Conversational debugging
- Success rate target: 70%+

Future versions will integrate LLM-based reasoning for improved recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import json
import os

from py_agent.tools.data_analysis import analyze_temporal_patterns
from py_agent.tools.model_selection import suggest_model, get_model_profiles
from py_agent.tools.recipe_generation import create_recipe, get_recipe_templates
from py_agent.tools.workflow_execution import fit_workflow, evaluate_workflow
from py_agent.tools.diagnostics import diagnose_performance
from py_agent.tools.multi_model_orchestration import (
    generate_workflowset, compare_models_cv, select_best_models, recommend_ensemble
)

# Phase 2 imports (optional)
try:
    from py_agent.llm.client import LLMClient
    from py_agent.agents.specialized_agents import (
        DataAnalyzer, FeatureEngineer, ModelSelector, Orchestrator
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class ForecastAgent:
    """
    AI-powered agent for automated forecasting workflow generation.

    This agent can:
    - Analyze temporal patterns in your data
    - Recommend appropriate models
    - Generate preprocessing recipes
    - Create and execute complete workflows
    - Diagnose performance issues

    Example:
        >>> agent = ForecastAgent()
        >>> workflow = agent.generate_workflow(
        ...     data=sales_data,
        ...     request="Forecast next quarter sales for each store"
        ... )
        >>> fit = workflow.fit(train_data)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4.5",
        verbose: bool = True,
        use_llm: bool = False,
        use_rag: bool = False,
        budget_per_day: float = 100.0
    ):
        """
        Initialize ForecastAgent.

        Args:
            api_key: Optional API key for LLM provider (Phase 2)
            model: LLM model to use (Phase 2)
            verbose: Whether to print progress messages
            use_llm: Whether to use LLM-enhanced reasoning (Phase 2) or rule-based (Phase 1)
            use_rag: Whether to use RAG knowledge base for example-driven recommendations (Phase 3.4)
            budget_per_day: Daily budget for LLM API calls in USD (Phase 2)
        """
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        self.use_llm = use_llm
        self.use_rag = use_rag
        self.session_history = []

        # Initialize Phase 2 components if LLM mode requested
        self.llm_client = None
        self.orchestrator = None

        # Initialize Phase 3.4 RAG components if requested
        self.example_library = None
        self.rag_retriever = None

        if use_rag:
            try:
                from py_agent.knowledge import (
                    ExampleLibrary,
                    RAGRetriever,
                    DEFAULT_LIBRARY_PATH
                )

                self.example_library = ExampleLibrary(DEFAULT_LIBRARY_PATH)
                self.rag_retriever = RAGRetriever(
                    self.example_library,
                    cache_embeddings=True
                )

                if self.verbose:
                    print(f"✅ RAG knowledge base initialized with {len(self.example_library)} examples")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to initialize RAG: {e}")
                    print("   Continuing without RAG enhancement...")
                self.use_rag = False

        if use_llm:
            if not LLM_AVAILABLE:
                raise ImportError(
                    "Phase 2 LLM components not available. "
                    "Make sure py_agent.llm and py_agent.agents.specialized_agents are installed."
                )

            # Check for API key
            if not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "API key required for LLM mode. "
                    "Provide api_key parameter or set ANTHROPIC_API_KEY environment variable."
                )

            # Initialize LLM client
            self.llm_client = LLMClient(
                api_key=api_key,
                model=model,
                budget_per_day=budget_per_day
            )

            # Initialize specialized agents
            data_analyzer = DataAnalyzer(self.llm_client)
            feature_engineer = FeatureEngineer(self.llm_client)
            model_selector = ModelSelector(self.llm_client)

            # Initialize orchestrator
            self.orchestrator = Orchestrator(
                llm_client=self.llm_client,
                data_analyzer=data_analyzer,
                feature_engineer=feature_engineer,
                model_selector=model_selector
            )

            if self.verbose:
                print("✅ ForecastAgent initialized in LLM mode (Phase 2)")
        else:
            if self.verbose:
                print("✅ ForecastAgent initialized in rule-based mode (Phase 1)")

    def generate_workflow(
        self,
        data: pd.DataFrame,
        request: str,
        formula: Optional[str] = None,
        constraints: Optional[Dict] = None
    ) -> object:
        """
        Generate complete workflow from natural language request.

        This is the primary method for single-shot workflow generation.

        Args:
            data: DataFrame containing your data
            request: Natural language description of forecasting task
            formula: Optional explicit formula (if not provided, will infer)
            constraints: Optional constraints dict with:
                - max_train_time: Maximum training time in seconds
                - interpretability: 'low', 'medium', or 'high'
                - max_memory: Maximum memory in MB

        Returns:
            Workflow object ready to fit

        Example:
            >>> agent = ForecastAgent()
            >>> workflow = agent.generate_workflow(
            ...     data=sales_data,
            ...     request="Forecast sales for 50 stores with seasonality"
            ... )
            >>> fit = workflow.fit(train_data)
        """
        # Route to LLM-based orchestrator if in LLM mode
        if self.use_llm and self.orchestrator:
            return self._generate_workflow_llm(data, request, formula, constraints)

        # Otherwise use Phase 1 rule-based implementation
        self._log("🔍 Analyzing your data...")

        # Parse request to extract key information
        task_info = self._parse_request(request, data)

        # Analyze data characteristics
        date_col, value_col = self._detect_columns(data, task_info)
        data_chars = analyze_temporal_patterns(data, date_col, value_col)

        self._log(f"✓ Detected {data_chars['frequency']} data")
        if data_chars['seasonality']['detected']:
            self._log(f"✓ Found seasonality (strength={data_chars['seasonality']['strength']:.2f})")
        if data_chars['trend']['significant']:
            self._log(f"✓ Detected {data_chars['trend']['direction']} trend")

        # RAG: Retrieve similar examples (Phase 3.4)
        rag_models = []
        rag_insights = []
        if self.use_rag and self.rag_retriever:
            self._log("\n📚 Searching knowledge base for similar examples...")
            try:
                similar_examples = self.rag_retriever.retrieve_by_data_characteristics(
                    data_chars, top_k=3
                )

                if similar_examples:
                    self._log(f"✓ Found {len(similar_examples)} similar examples:")
                    for i, result in enumerate(similar_examples, 1):
                        self._log(f"  {i}. {result.example.title} (similarity: {result.similarity_score:.2f})")
                        self._log(f"     {result.example.description[:80]}...")
                        self._log(f"     Domain: {result.example.domain}, Difficulty: {result.example.difficulty}")

                    # Extract model recommendations from examples
                    rag_models = self.rag_retriever.get_model_recommendations_from_examples(
                        similar_examples, top_n=3
                    )

                    # Extract preprocessing insights
                    rag_insights = self.rag_retriever.get_preprocessing_insights(similar_examples)

                    # Extract key lessons
                    rag_lessons = self.rag_retriever.get_key_lessons(similar_examples)

                    if rag_models:
                        self._log("\n💡 RAG-recommended models:")
                        for model, score in rag_models:
                            self._log(f"  • {model} (confidence: {score:.2f})")

                    if rag_lessons:
                        self._log("\n💭 Key Lessons from Similar Cases:")
                        for lesson in rag_lessons[:3]:  # Show top 3 lessons
                            self._log(f"  • {lesson}")
                else:
                    self._log("  No similar examples found in knowledge base")
            except Exception as e:
                self._log(f"⚠️  RAG retrieval failed: {e}")

        # Suggest models
        self._log("\n📊 Recommending models...")
        model_suggestions = suggest_model(data_chars, constraints)

        # Boost confidence for models recommended by RAG
        if rag_models:
            rag_model_names = {model for model, _ in rag_models}
            for suggestion in model_suggestions:
                if suggestion['model_type'] in rag_model_names:
                    # Boost confidence by up to 10% for RAG-recommended models
                    rag_score = next((score for model, score in rag_models
                                     if model == suggestion['model_type']), 0.0)
                    boost = 0.1 * rag_score  # Max boost of 0.1 for perfect RAG match
                    suggestion['confidence'] = min(1.0, suggestion['confidence'] + boost)
                    suggestion['rag_boosted'] = True

        if len(model_suggestions) == 0:
            raise ValueError("No suitable models found for your data and constraints")

        # Select best model
        best_model = model_suggestions[0]
        self._log(f"✓ Recommended: {best_model['model_type']}")
        self._log(f"  Reason: {best_model['reasoning']}")
        self._log(f"  Confidence: {best_model['confidence']:.0%}")

        # Generate recipe
        self._log("\n🔧 Creating preprocessing recipe...")
        recipe_code = create_recipe(
            data_chars,
            best_model['model_type'],
            domain=task_info.get('domain')
        )
        self._log("✓ Recipe generated")

        # Generate complete workflow code
        self._log("\n⚙️  Building workflow...")
        workflow_code = self._generate_workflow_code(
            model_type=best_model['model_type'],
            recipe_code=recipe_code,
            formula=formula or self._infer_formula(data, value_col, date_col),
            group_col=task_info.get('group_col')
        )

        # Execute workflow creation
        namespace = {'pd': pd}
        exec(workflow_code, namespace)

        if 'wf' not in namespace:
            raise ValueError("Failed to create workflow")

        workflow = namespace['wf']
        self._log("✓ Workflow ready!")

        # Store workflow info for later reference
        self.last_workflow_info = {
            'model_type': best_model['model_type'],
            'data_characteristics': data_chars,
            'recipe_code': recipe_code,
            'workflow_code': workflow_code,
            'task_info': task_info
        }

        return workflow

    def debug_session(
        self,
        model_fit: object,
        test_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Start interactive debugging session for a fitted model.

        Analyzes model performance and provides recommendations for improvement.

        Args:
            model_fit: Fitted workflow object
            test_data: Optional test data for overfitting detection

        Returns:
            Dictionary with diagnostics and recommendations

        Example:
            >>> fit = workflow.fit(train_data)
            >>> diagnostics = agent.debug_session(fit, test_data)
            >>> print(diagnostics['recommendations'])
        """
        self._log("🔍 Diagnosing model performance...")

        # Run diagnostics
        diagnostics = diagnose_performance(model_fit, test_data)

        # Format results
        self._log("\n📊 Performance Metrics:")
        for metric, value in diagnostics['metrics'].items():
            self._log(f"  {metric}: {value:.4f}")

        # Report issues
        if len(diagnostics['issues_detected']) > 0:
            self._log("\n⚠️  Issues Detected:")
            for issue in diagnostics['issues_detected']:
                severity_emoji = '🔴' if issue['severity'] == 'high' else '🟡'
                self._log(f"\n{severity_emoji} {issue['type'].upper()}")
                self._log(f"  Evidence: {issue['evidence']}")
                self._log(f"  💡 Recommendation: {issue['recommendation']}")
        else:
            self._log("\n✅ No major issues detected!")

        # Generate recommendations
        recommendations = self._generate_recommendations(diagnostics)
        diagnostics['recommendations'] = recommendations

        return diagnostics

    def compare_models(
        self,
        data: pd.DataFrame,
        request: str,
        n_models: int = 5,
        cv_strategy: str = 'time_series',
        n_folds: int = 5,
        date_column: Optional[str] = None,
        formula: Optional[str] = None,
        constraints: Optional[Dict] = None,
        return_ensemble: bool = False
    ) -> Dict:
        """
        Compare multiple models and select the best.

        This is Phase 3.3: Multi-Model WorkflowSet Orchestration.
        Automatically generates, evaluates, and compares multiple models
        using cross-validation, then returns the best model(s).

        Args:
            data: DataFrame containing your data
            request: Natural language description of forecasting task
            n_models: Number of models to compare (default: 5)
            cv_strategy: 'time_series' or 'vfold' (default: 'time_series')
            n_folds: Number of CV folds (default: 5)
            date_column: Date column name (required for time_series CV)
            formula: Optional explicit formula
            constraints: Optional constraints dict
            return_ensemble: Whether to return ensemble recommendation (default: False)

        Returns:
            Dictionary with:
                - best_model_id: ID of best performing model
                - rankings: DataFrame with all model rankings
                - workflowset: WorkflowSet object with all models
                - cv_results: Cross-validation results
                - ensemble_recommendation: Optional ensemble recommendation

        Example:
            >>> agent = ForecastAgent()
            >>> results = agent.compare_models(
            ...     data=sales_data,
            ...     request="Forecast daily sales with seasonality",
            ...     n_models=5,
            ...     date_column='date'
            ... )
            >>> print(results['best_model_id'])
            'prophet_reg_1'
            >>> print(results['rankings'].head())
        """
        self._log("🚀 Starting multi-model comparison (Phase 3.3)...")

        # Parse request
        task_info = self._parse_request(request, data)

        # Analyze data
        self._log("\n🔍 Analyzing data characteristics...")
        if date_column is None:
            date_column, value_col = self._detect_columns(data, task_info)
        else:
            _, value_col = self._detect_columns(data, task_info)

        data_chars = analyze_temporal_patterns(data, date_column, value_col)

        self._log(f"✓ Detected {data_chars['frequency']} data")
        if data_chars['seasonality']['detected']:
            self._log(f"✓ Found seasonality (strength={data_chars['seasonality']['strength']:.2f})")

        # RAG: Retrieve similar examples (Phase 3.4)
        rag_models = []
        if self.use_rag and self.rag_retriever:
            self._log("\n📚 Searching knowledge base for similar examples...")
            try:
                similar_examples = self.rag_retriever.retrieve_by_data_characteristics(
                    data_chars, top_k=3
                )

                if similar_examples:
                    self._log(f"✓ Found {len(similar_examples)} similar examples")
                    # Extract model recommendations
                    rag_models = self.rag_retriever.get_model_recommendations_from_examples(
                        similar_examples, top_n=5
                    )
                    if rag_models:
                        self._log("💡 RAG-recommended models:")
                        for model, score in rag_models[:3]:
                            self._log(f"  • {model} (confidence: {score:.2f})")
            except Exception as e:
                self._log(f"⚠️  RAG retrieval failed: {e}")

        # Get model recommendations
        self._log(f"\n📊 Recommending top {n_models} models...")
        model_recommendations = suggest_model(data_chars, constraints)

        # Boost confidence for models recommended by RAG
        if rag_models:
            rag_model_names = {model for model, _ in rag_models}
            for suggestion in model_recommendations:
                if suggestion['model_type'] in rag_model_names:
                    rag_score = next((score for model, score in rag_models
                                     if model == suggestion['model_type']), 0.0)
                    boost = 0.1 * rag_score
                    suggestion['confidence'] = min(1.0, suggestion['confidence'] + boost)
                    suggestion['rag_boosted'] = True

        if len(model_recommendations) == 0:
            raise ValueError("No suitable models found for your data and constraints")

        # Limit to n_models
        models_to_compare = model_recommendations[:n_models]

        for i, rec in enumerate(models_to_compare, 1):
            self._log(f"  {i}. {rec['model_type']} (confidence: {rec['confidence']:.0%})")

        # Generate recipe (use recipe for best model)
        self._log("\n🔧 Creating preprocessing recipe...")
        recipe_code = create_recipe(
            data_chars,
            models_to_compare[0]['model_type'],
            domain=task_info.get('domain')
        )
        self._log("✓ Recipe generated")

        # Generate WorkflowSet
        self._log("\n⚙️  Building WorkflowSet...")
        inferred_formula = formula or self._infer_formula(data, value_col, date_column)

        wf_set = generate_workflowset(
            model_recommendations=models_to_compare,
            recipe_code=recipe_code,
            formula=inferred_formula,
            max_models=n_models
        )
        self._log(f"✓ Created {len(wf_set.workflows)} workflows")

        # Run cross-validation
        self._log(f"\n🔄 Running {cv_strategy} cross-validation ({n_folds} folds)...")
        self._log("   This may take a few minutes...")

        cv_results, rankings = compare_models_cv(
            wf_set=wf_set,
            data=data,
            cv_strategy=cv_strategy,
            n_folds=n_folds,
            date_column=date_column
        )

        self._log("✓ Cross-validation complete!")

        # Display rankings
        self._log("\n🏆 Model Rankings:")
        self._log("-" * 60)
        for i, row in rankings.head(n_models).iterrows():
            self._log(f"  {i+1}. {row['wflow_id']}")
            self._log(f"     RMSE: {row['mean']:.4f} (±{row['std_err']:.4f})")

        # Select best model
        best_model_ids = select_best_models(
            rankings,
            selection_strategy='best',
            n_models=1
        )
        best_model_id = best_model_ids[0]

        self._log(f"\n✅ Best Model: {best_model_id}")

        # Prepare results
        results = {
            'best_model_id': best_model_id,
            'rankings': rankings,
            'workflowset': wf_set,
            'cv_results': cv_results,
            'data_characteristics': data_chars,
            'formula': inferred_formula,
            'models_compared': [rec['model_type'] for rec in models_to_compare]
        }

        # Generate ensemble recommendation if requested
        if return_ensemble:
            self._log("\n🤝 Generating ensemble recommendation...")
            ensemble_rec = recommend_ensemble(
                wf_set=wf_set,
                ranked_results=rankings,
                ensemble_size=min(3, len(models_to_compare))
            )
            results['ensemble_recommendation'] = ensemble_rec

            self._log(f"✓ Ensemble: {', '.join(ensemble_rec['model_ids'])}")
            self._log(f"  Expected RMSE: {ensemble_rec['expected_performance']:.4f}")
            self._log(f"  Diversity: {ensemble_rec['diversity_score']:.2f}")
            self._log(f"  Type: {ensemble_rec['ensemble_type']}")

        return results

    def start_session(self) -> 'ConversationalSession':
        """
        Start conversational session for iterative workflow building.

        Returns:
            ConversationalSession object for multi-turn interaction

        Example:
            >>> agent = ForecastAgent()
            >>> session = agent.start_session()
            >>> session.send("I need to forecast sales")
            >>> session.send("Monthly data, 3 years history")
            >>> workflow = session.get_workflow()
        """
        return ConversationalSession(self)

    # Helper methods

    def _parse_request(self, request: str, data: pd.DataFrame) -> Dict:
        """
        Parse natural language request to extract task information.

        Extracts:
        - Whether it's grouped/panel data
        - Domain hints (retail, finance, etc.)
        - Forecasting horizon
        - Special requirements (seasonality, etc.)
        """
        request_lower = request.lower()

        task_info = {}

        # Detect grouped data
        group_keywords = ['store', 'stores', 'product', 'products', 'region', 'regions',
                          'country', 'countries', 'entity', 'entities']
        for keyword in group_keywords:
            if keyword in request_lower:
                task_info['is_grouped'] = True
                task_info['group_keyword'] = keyword
                # Try to find group column
                for col in data.columns:
                    if keyword in col.lower():
                        task_info['group_col'] = col
                        break
                break

        # Detect domain
        if any(word in request_lower for word in ['retail', 'sales', 'store']):
            task_info['domain'] = 'retail'
        elif any(word in request_lower for word in ['finance', 'stock', 'price']):
            task_info['domain'] = 'finance'
        elif any(word in request_lower for word in ['energy', 'load', 'consumption']):
            task_info['domain'] = 'energy'

        # Detect forecasting horizon
        if 'quarter' in request_lower:
            task_info['horizon'] = 'quarterly'
        elif 'month' in request_lower:
            task_info['horizon'] = 'monthly'
        elif 'week' in request_lower:
            task_info['horizon'] = 'weekly'
        elif 'day' in request_lower:
            task_info['horizon'] = 'daily'

        return task_info

    def _detect_columns(
        self,
        data: pd.DataFrame,
        task_info: Dict
    ) -> tuple:
        """
        Detect date and value columns from data.

        Returns:
            (date_col, value_col) tuple
        """
        # Find date column
        date_col = None
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                date_col = col
                break
            elif any(word in col.lower() for word in ['date', 'time', 'timestamp']):
                date_col = col
                break

        if date_col is None:
            raise ValueError("Could not find date column. Please specify date_col.")

        # Find value column (outcome)
        value_col = None
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove date-related columns
        numeric_cols = [col for col in numeric_cols if col != date_col]

        # Look for common target names
        for col in numeric_cols:
            if any(word in col.lower() for word in ['target', 'y', 'outcome', 'sales', 'value']):
                value_col = col
                break

        # If not found, use first numeric column
        if value_col is None and len(numeric_cols) > 0:
            value_col = numeric_cols[0]

        if value_col is None:
            raise ValueError("Could not find value column. Please specify value_col.")

        return date_col, value_col

    def _infer_formula(
        self,
        data: pd.DataFrame,
        value_col: str,
        date_col: str
    ) -> str:
        """
        Infer model formula from data columns.

        Returns formula string like "sales ~ ."
        """
        # Use dot notation to include all predictors
        return f"{value_col} ~ ."

    def _generate_workflow_code(
        self,
        model_type: str,
        recipe_code: str,
        formula: str,
        group_col: Optional[str]
    ) -> str:
        """Generate complete workflow code."""
        lines = [
            "from py_workflows import workflow",
            f"from py_parsnip import {model_type}",
            "",
            "# Preprocessing recipe",
            recipe_code,
            "",
            "# Model specification",
            f"spec = {model_type}()",
            "",
            "# Create workflow",
            "wf = workflow().add_recipe(rec).add_model(spec)"
        ]

        return "\n".join(lines)

    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate actionable recommendations from diagnostics."""
        recommendations = []

        for issue in diagnostics['issues_detected']:
            recommendations.append(issue['recommendation'])

        if len(recommendations) == 0:
            recommendations.append("Model looks good! Consider validating on holdout data.")

        return recommendations

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)

    def _generate_workflow_llm(
        self,
        data: pd.DataFrame,
        request: str,
        formula: Optional[str] = None,
        constraints: Optional[Dict] = None
    ) -> object:
        """
        Generate workflow using LLM-based orchestrator (Phase 2).

        Args:
            data: DataFrame containing your data
            request: Natural language description of forecasting task
            formula: Optional explicit formula
            constraints: Optional constraints dict

        Returns:
            Workflow object ready to fit
        """
        self._log("🤖 Using LLM-enhanced workflow generation (Phase 2)...")
        self._log("🔍 Orchestrating specialized agents...")

        # Use orchestrator to generate workflow
        orchestration_result = self.orchestrator.generate_workflow(
            data=data,
            request=request,
            constraints=constraints
        )

        # Store workflow info for debugging
        self.last_workflow_info = {
            'mode': 'llm',
            'data_characteristics': orchestration_result['data_analysis']['analysis'],
            'model_type': orchestration_result['model_selection']['recommendation'].get('model_type', 'unknown'),
            'recipe_code': orchestration_result['feature_engineering']['recipe'],
            'workflow_code': orchestration_result['workflow'],
            'data_analysis_reasoning': orchestration_result['data_analysis']['insights'],
            'model_selection_reasoning': orchestration_result['model_selection']['reasoning'],
            'feature_engineering_reasoning': orchestration_result['feature_engineering']['reasoning']
        }

        self._log("\n✅ Workflow generation complete!")
        self._log(f"📊 Data Analysis: {orchestration_result['data_analysis']['tool_calls']} tool calls")
        self._log(f"🎯 Model Selection: {orchestration_result['model_selection']['tool_calls']} tool calls")
        self._log(f"🔧 Feature Engineering: {orchestration_result['feature_engineering']['tool_calls']} tool calls")

        if self.llm_client:
            self._log(f"\n💰 API Cost: ${self.llm_client.total_cost:.4f}")
            self._log(f"🪙  Tokens: {self.llm_client.usage_stats['total_input_tokens']} in, {self.llm_client.usage_stats['total_output_tokens']} out")

        # Build actual workflow object from generated code
        from py_workflows import workflow
        from py_recipes import recipe

        # Parse model type from recommendation
        recommendation = orchestration_result['model_selection']['recommendation']
        if isinstance(recommendation, dict):
            model_type = recommendation.get('model_type', 'linear_reg')
        else:
            model_type = 'linear_reg'

        # Import and create model dynamically
        spec = self._create_model_spec(model_type)

        # Create recipe (simplified - in full implementation would parse recipe_code)
        rec = recipe()

        # Build workflow
        wf = workflow().add_recipe(rec).add_model(spec)

        return wf

    def _create_model_spec(self, model_type: str) -> object:
        """
        Dynamically create model specification for any model type.

        Args:
            model_type: Model type string (e.g., 'linear_reg', 'prophet_reg')

        Returns:
            Model specification object

        Supports all 23 py-tidymodels models.
        """
        # Map of all 23 model types to their import names
        model_map = {
            # Baseline
            'null_model': 'null_model',
            'naive_reg': 'naive_reg',
            # Linear & Generalized
            'linear_reg': 'linear_reg',
            'poisson_reg': 'poisson_reg',
            'gen_additive_mod': 'gen_additive_mod',
            # Tree-Based
            'decision_tree': 'decision_tree',
            'rand_forest': 'rand_forest',
            'boost_tree': 'boost_tree',
            # SVM
            'svm_rbf': 'svm_rbf',
            'svm_linear': 'svm_linear',
            # Instance-Based & Adaptive
            'nearest_neighbor': 'nearest_neighbor',
            'mars': 'mars',
            'mlp': 'mlp',
            # Time Series
            'arima_reg': 'arima_reg',
            'prophet_reg': 'prophet_reg',
            'exp_smoothing': 'exp_smoothing',
            'seasonal_reg': 'seasonal_reg',
            'varmax_reg': 'varmax_reg',
            # Hybrid Time Series
            'arima_boost': 'arima_boost',
            'prophet_boost': 'prophet_boost',
            # Recursive
            'recursive_reg': 'recursive_reg',
            # Hybrid & Manual
            'hybrid_model': 'hybrid_model',
            'manual_reg': 'manual_reg'
        }

        # Get import name (default to linear_reg if unknown)
        import_name = model_map.get(model_type, 'linear_reg')

        # Dynamic import and instantiation
        import py_parsnip
        model_func = getattr(py_parsnip, import_name, None)

        if model_func is None:
            # Fallback to linear_reg
            if self.verbose:
                print(f"⚠️  Unknown model type '{model_type}', falling back to linear_reg")
            model_func = py_parsnip.linear_reg

        # Create and return model specification
        return model_func()


class ConversationalSession:
    """
    Multi-turn conversational session for iterative workflow building.

    Allows users to provide information incrementally through conversation.
    """

    def __init__(self, agent: ForecastAgent):
        """Initialize session with parent agent."""
        self.agent = agent
        self.context = {}
        self.messages = []

    def send(self, message: str):
        """
        Send message to agent and update context.

        Args:
            message: User message

        Example:
            >>> session = agent.start_session()
            >>> session.send("I have sales data")
            >>> session.send("Daily frequency")
        """
        self.messages.append({'role': 'user', 'content': message})

        # Extract information from message
        self._update_context(message)

        # Generate response
        response = self._generate_response(message)
        self.messages.append({'role': 'assistant', 'content': response})

        if self.agent.verbose:
            print(f"Agent: {response}")

    def get_workflow(self) -> object:
        """
        Generate workflow from collected context.

        Returns:
            Workflow object ready to fit

        Example:
            >>> session = agent.start_session()
            >>> session.send("Forecast monthly sales")
            >>> workflow = session.get_workflow()
        """
        if 'data' not in self.context:
            raise ValueError("Please provide data before generating workflow")

        # Generate workflow from context
        workflow = self.agent.generate_workflow(
            data=self.context['data'],
            request=" ".join([m['content'] for m in self.messages if m['role'] == 'user']),
            constraints=self.context.get('constraints')
        )

        return workflow

    def _update_context(self, message: str):
        """Extract information from message and update context."""
        message_lower = message.lower()

        # Extract frequency
        if 'daily' in message_lower:
            self.context['frequency'] = 'daily'
        elif 'weekly' in message_lower:
            self.context['frequency'] = 'weekly'
        elif 'monthly' in message_lower:
            self.context['frequency'] = 'monthly'

        # Extract horizon
        if 'next' in message_lower:
            # Extract number
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() == 'next' and i + 1 < len(words):
                    try:
                        self.context['horizon_steps'] = int(words[i + 1])
                    except:
                        pass

    def _generate_response(self, message: str) -> str:
        """Generate agent response based on context."""
        # Simple rule-based responses for MVP
        if len(self.messages) == 1:
            return "I'll help you build a forecasting workflow. Can you tell me about your data frequency (daily/weekly/monthly)?"

        if 'frequency' in self.context and 'horizon_steps' not in self.context:
            return "How far ahead do you need to forecast?"

        if 'data' not in self.context:
            return "Please provide your data using context['data'] = your_dataframe"

        return "I have enough information. Call get_workflow() to generate your workflow."
