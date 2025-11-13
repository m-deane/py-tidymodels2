# py_agent Tutorial Notebooks

Welcome to the **py_agent tutorial collection**! These Jupyter notebooks provide comprehensive, hands-on education for using py_agent to build intelligent forecasting workflows.

---

## 📚 Tutorial Overview

We provide **4 progressive tutorials** covering beginner to advanced topics:

| # | Notebook | Level | Time | Topics Covered |
|---|----------|-------|------|----------------|
| **22** | [**Complete Overview**](22_agent_complete_tutorial.ipynb) | Beginner | 30-45 min | All phases (1, 2, 3.1-3.5), basic workflow, multi-model comparison, RAG, autonomous iteration |
| **23** | [**LLM-Enhanced Mode**](23_agent_llm_mode_tutorial.ipynb) | Intermediate | 45-60 min | Claude Sonnet 4.5 integration, explainable reasoning, budget management, constraints |
| **24** | [**Domain Examples**](24_agent_domain_specific_examples.ipynb) | Intermediate | 60-90 min | Retail, finance, energy forecasting; domain-adapted preprocessing; industry best practices |
| **25** | [**Advanced Features**](25_agent_advanced_features.ipynb) | Advanced | 90-120 min | Performance debugging, ensembles, grouped modeling, production best practices |

**Total Learning Time**: 3.5-5 hours for complete mastery

---

## 🎯 Which Tutorial Should I Start With?

### I'm New to py_agent
**Start here**: [Notebook 22 - Complete Overview](22_agent_complete_tutorial.ipynb)
- Get a comprehensive introduction to all features
- Learn the basics of workflow generation
- See examples of each major phase
- ~30-45 minutes

### I Want Explainable Forecasts
**Start here**: [Notebook 23 - LLM-Enhanced Mode](23_agent_llm_mode_tutorial.ipynb)
- Learn how Claude Sonnet 4.5 enhances forecasting
- See model selection reasoning
- Compare rule-based ($0) vs LLM ($4-10) modes
- Understand budget management
- ~45-60 minutes

### I Have Industry-Specific Needs
**Start here**: [Notebook 24 - Domain Examples](24_agent_domain_specific_examples.ipynb)
- See examples from retail, finance, energy
- Learn domain-adapted preprocessing
- Understand industry best practices
- Apply to your specific use case
- ~60-90 minutes

### I'm Deploying to Production
**Start here**: [Notebook 25 - Advanced Features](25_agent_advanced_features.ipynb)
- Learn performance debugging techniques
- Understand ensemble methods
- Master grouped/panel modeling
- Get production best practices (monitoring, error handling, versioning)
- ~90-120 minutes

---

## 📖 Tutorial Details

### Notebook 22: Complete Overview ⭐ START HERE

**File**: `22_agent_complete_tutorial.ipynb`

**What You'll Learn**:
- Basic workflow generation (Phase 1)
- Multi-model comparison with 5 models (Phase 3.3)
- RAG knowledge base usage (Phase 3.4)
- Autonomous iteration for performance improvement (Phase 3.5)
- Combining all features together

**Dataset**: Daily sales with trend, seasonality, promotions (2 years)

**Key Takeaways**:
- Understand py_agent architecture
- Know when to use each phase
- Compare rule-based vs advanced features
- Get started quickly

**Prerequisites**: None - complete beginner friendly

---

### Notebook 23: LLM-Enhanced Mode

**File**: `23_agent_llm_mode_tutorial.ipynb`

**What You'll Learn**:
- Initialize LLM mode with Claude Sonnet 4.5
- View explainable model selection reasoning
- Handle complex natural language constraints
- Manage API costs with budget controls
- Compare rule-based vs LLM performance
- Combine LLM with multi-model comparison

**Dataset**: E-commerce sales with complex patterns (multiple seasonality, external factors, non-linear interactions)

**Key Takeaways**:
- When to use LLM mode ($4-10) vs rule-based ($0)
- How to set domain-specific constraints
- Budget management best practices
- Explainability for stakeholders

**Prerequisites**: Complete Notebook 22 first, ANTHROPIC_API_KEY required

---

### Notebook 24: Domain-Specific Examples

**File**: `24_agent_domain_specific_examples.ipynb`

**What You'll Learn**:
- **Retail**: Weekly seasonality, promotions, holidays, inventory optimization
- **Finance**: Stock prices, volatility, non-stationarity, risk management
- **Energy**: Multiple seasonality, temperature effects, extreme weather, grid stability

**Datasets**:
- Retail: Daily store sales with promotions and weather
- Finance: Stock prices with market correlations and momentum
- Energy: Electricity demand with temperature and extreme events

**Key Takeaways**:
- Domain-adapted preprocessing strategies
- Industry-specific model selection
- Business requirement constraints
- Best practices per domain
- Universal principles across domains

**Prerequisites**: Complete Notebook 22, recommended to complete Notebook 23

---

### Notebook 25: Advanced Features & Production

**File**: `25_agent_advanced_features.ipynb`

**What You'll Learn**:
- **Performance Debugging**: Autonomous iteration with try-evaluate-improve loops
- **Multi-Model Ensembles**: Diversity scoring, ensemble selection
- **Grouped/Panel Modeling**: Multi-entity forecasting at scale
- **Production Best Practices**: Error handling, monitoring, budget controls, versioning

**Dataset**: Complex scenario with non-linear relationships, structural breaks, heteroscedasticity, outliers, missing data

**Key Takeaways**:
- Debug poor forecasts automatically
- Build robust ensembles
- Scale to multiple entities (stores, regions, products)
- Deploy to production safely
- Monitor model performance
- Version control configurations

**Prerequisites**: Complete Notebooks 22-24, production deployment knowledge helpful

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source py-tidymodels2/bin/activate

# Install package in editable mode (if not already)
pip install -e .

# Install Jupyter (if not already)
pip install jupyter ipykernel

# Create Jupyter kernel
python -m ipykernel install --user --name=py-tidymodels2

# Launch Jupyter
jupyter notebook
```

### 2. Navigate to Examples

- Open `examples/` directory in Jupyter
- Start with `22_agent_complete_tutorial.ipynb`

### 3. Run the Notebooks

- Select kernel: **py-tidymodels2**
- Run cells sequentially (**Shift+Enter**)
- Modify examples with your own data

---

## 💡 Tips for Success

1. **Run sequentially**: Execute cells in order from top to bottom
2. **Modify examples**: Try changing parameters, data, constraints
3. **Save checkpoints**: Use **File → Save Checkpoint** before major changes
4. **Restart kernel**: If issues occur, **Kernel → Restart & Clear Output**
5. **Clear cache**: After code changes, clear `__pycache__` directories:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   ```

---

## 📊 Learning Path by Goal

### Goal: Learn the Basics
- ✅ Notebook 22 (Complete Overview)
- Try on your own data
- Experiment with different models

### Goal: Use in Production
- ✅ Notebook 22 (Complete Overview)
- ✅ Notebook 23 (LLM Mode) - if budget allows
- ✅ Notebook 25 (Advanced Features)
- Focus on monitoring, error handling, versioning

### Goal: Apply to Specific Domain
- ✅ Notebook 22 (Complete Overview)
- ✅ Notebook 24 (Domain Examples) - focus on your industry
- ✅ Notebook 25 (Advanced Features) - for production

### Goal: Maximize Accuracy
- ✅ Notebook 22 (Complete Overview)
- ✅ Notebook 23 (LLM Mode) - for explainable reasoning
- ✅ Notebook 25 (Advanced Features) - for ensembles and iteration
- Use RAG knowledge base (`use_rag=True`)
- Use autonomous iteration (`agent.iterate()`)

---

## 📝 Additional Resources

### Documentation
- **Main README**: `py_agent/README.md` - Complete API reference
- **Planning Docs**: `.claude_plans/` - Implementation summaries
- **Tests**: `tests/test_agent/` - Comprehensive test suite (252+ tests)

### Example Notebooks (Legacy)
- `01_hardhat_demo.ipynb` - Data preprocessing basics
- `02_parsnip_demo.ipynb` - Model interface
- `03_time_series_models.ipynb` - Prophet and ARIMA
- `04_rand_forest_demo.ipynb` - Random Forest
- `05_recipes_comprehensive_demo.ipynb` - Feature engineering (51 steps)
- `08_workflows_demo.ipynb` - Workflow pipelines
- `09_yardstick_demo.ipynb` - Evaluation metrics
- `10_tune_demo.ipynb` - Hyperparameter tuning
- `11_workflowsets_demo.ipynb` - Multi-model comparison
- `12_recursive_forecasting_demo.ipynb` - Recursive forecasting
- `13_panel_models_demo.ipynb` - Grouped/panel models

### Community
- **GitHub Issues**: Report bugs, request features
- **Discussions**: Ask questions, share use cases

---

## ✅ Completion Checklist

Track your progress through the tutorials:

- [ ] Completed Notebook 22 (Complete Overview)
- [ ] Completed Notebook 23 (LLM-Enhanced Mode)
- [ ] Completed Notebook 24 (Domain Examples)
- [ ] Completed Notebook 25 (Advanced Features)
- [ ] Applied to my own data
- [ ] Created custom workflow
- [ ] Deployed to production

---

## 🎯 Next Steps After Tutorials

Once you've completed the tutorials:

1. **Apply to Your Data**: Use what you've learned on real forecasting problems
2. **Experiment**: Try different models, preprocessing, constraints
3. **Share Results**: Contribute back to the community
4. **Production Deploy**: Follow best practices from Notebook 25
5. **Monitor Performance**: Track metrics, retrain periodically

---

## 🏆 Tutorial Statistics

- **Total Notebooks**: 4
- **Total Content**: ~5 hours
- **Total Cells**: 94
- **Total Visualizations**: 30+
- **Domains Covered**: Retail, Finance, Energy, E-commerce
- **Phases Covered**: All (1, 2, 3.1-3.5)
- **Models Demonstrated**: 10+ different model types
- **Features Covered**: All major py_agent capabilities

---

**Happy Forecasting! 🚀**

For questions or issues, please open a GitHub issue or refer to the main documentation.

---

**Last Updated**: 2025-11-13
**Version**: v0.3.0 (All Phases Complete)
