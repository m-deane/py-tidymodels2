Architecture Documentation
==========================

py-tidymodels follows a layered architecture.

Layers
------

1. **py-hardhat**: Data preprocessing (mold/forge)
2. **py-parsnip**: Model specification interface
3. **py-rsample**: Resampling and cross-validation
4. **py-workflows**: Pipeline composition
5. **py-recipes**: Feature engineering (51 steps)
6. **py-yardstick**: Evaluation metrics (17 metrics)
7. **py-tune**: Hyperparameter tuning
8. **py-workflowsets**: Multi-model comparison

Key Patterns
------------

- **Immutable Specifications**: ModelSpec frozen dataclass
- **Registry Pattern**: Engine registration via decorators
- **Dual-Path Processing**: Standard vs raw data handling
- **Three-DataFrame Output**: outputs, coefficients, stats

See Also
--------

* :doc:`../user_guide/concepts` - Detailed architecture guide
* CLAUDE.md - Implementation details
