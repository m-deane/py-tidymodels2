# py-tidymodels Documentation

This directory contains the Sphinx documentation for py-tidymodels.

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment activated
- py-tidymodels installed in editable mode

### Build Documentation Locally

```bash
# Install documentation dependencies
cd docs
pip install -r requirements.txt

# Build HTML documentation
make html

# View documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Using the Build Script

```bash
# Activate virtual environment first
source ../py-tidymodels2/bin/activate

# Run build script
./build_docs.sh
```

The script will:
1. Install dependencies
2. Clean previous builds
3. Build HTML documentation
4. Run quality checks
5. Display the output location

### Serve Documentation Locally

```bash
# Build and serve on http://localhost:8000
make serve
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst               # Main documentation index
├── Makefile                # Build commands
├── requirements.txt        # Documentation dependencies
├── build_docs.sh           # Automated build script
│
├── api/                    # API Reference
│   ├── parsnip.rst         # Model specifications
│   ├── workflows.rst       # Workflow pipelines
│   ├── recipes.rst         # Feature engineering
│   ├── hardhat.rst         # Data preprocessing
│   ├── rsample.rst         # Resampling & CV
│   ├── yardstick.rst       # Model metrics
│   ├── tune.rst            # Hyperparameter tuning
│   ├── workflowsets.rst    # Multi-model comparison
│   ├── visualize.rst       # Visualizations
│   └── stacks.rst          # Model ensembling
│
├── user_guide/             # User Guides
│   ├── installation.rst    # Installation instructions
│   ├── quickstart.rst      # Quick start guide
│   ├── concepts.rst        # Core concepts
│   ├── recipes.rst         # Recipe guide
│   ├── time_series.rst     # Time series modeling
│   ├── tuning.rst          # Hyperparameter tuning
│   └── workflows.rst       # Workflow guide
│
├── models/                 # Model Reference
│   ├── linear_models.rst
│   ├── tree_models.rst
│   ├── time_series.rst
│   ├── ensemble_models.rst
│   └── baseline_models.rst
│
├── examples/               # Examples
│   ├── basic_regression.rst
│   ├── time_series_forecasting.rst
│   ├── hyperparameter_tuning.rst
│   ├── panel_models.rst
│   └── model_stacking.rst
│
└── development/            # Development Docs
    ├── contributing.rst
    ├── architecture.rst
    ├── testing.rst
    └── changelog.rst
```

## Available Make Commands

```bash
# Build commands
make html          # Build HTML documentation
make latexpdf      # Build PDF documentation
make epub          # Build EPUB documentation
make clean         # Clean build directory

# Quality checks
make check         # Check links and coverage
make linkcheck     # Check for broken links
make coverage      # Generate documentation coverage report

# Development
make serve         # Serve documentation locally (port 8000)
make watch         # Watch for changes and rebuild
make quick         # Quick rebuild (no clean)

# Full build
make all           # Build all formats (HTML, PDF, EPUB)
```

## Writing Documentation

### Adding New API Reference

1. Create `.rst` file in `api/` directory
2. Use Sphinx autodoc directives:

```rst
.. automodule:: py_package.module
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Add to `index.rst` table of contents

### Adding Examples

1. Create `.rst` file in `examples/` directory
2. Use code-block directives with Python highlighting:

```rst
.. code-block:: python

   from py_parsnip import linear_reg

   spec = linear_reg()
   fit = spec.fit(data, "y ~ x")
```

3. Add to `index.rst` table of contents

### Docstring Style

Use Google-style docstrings:

```python
def function(arg1, arg2):
    """Brief description.

    Longer description with more details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Example:
        >>> result = function(1, 2)
        >>> print(result)
        3
    """
    return arg1 + arg2
```

## CI/CD Integration

Documentation is automatically built and deployed via GitHub Actions:

- **On Push to Main**: Builds and deploys to GitHub Pages
- **On Pull Request**: Builds and checks for errors
- **Manual Trigger**: Can be triggered manually via Actions tab

### GitHub Pages Setup

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

Documentation will be available at:
`https://<username>.github.io/<repository>/`

## Documentation Quality

### Coverage Report

Check which code is documented:

```bash
make coverage
open _build/coverage/python.txt
```

### Link Checking

Check for broken links:

```bash
make linkcheck
open _build/linkcheck/output.txt
```

### Docstring Coverage

Check docstring coverage with interrogate:

```bash
pip install interrogate
interrogate -v py_parsnip py_workflows py_recipes
```

## Troubleshooting

### Sphinx Build Errors

**Problem**: `ModuleNotFoundError` during build

**Solution**: Ensure package is installed in editable mode:
```bash
pip install -e ..
```

**Problem**: Warnings about missing docstrings

**Solution**: Add docstrings to all public functions/classes, or exclude from documentation.

### LaTeX/PDF Issues

**Problem**: PDF build fails

**Solution**: Install LaTeX:
```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
```

### Theme Issues

**Problem**: Theme doesn't look right

**Solution**: Reinstall theme:
```bash
pip install --force-reinstall sphinx-rtd-theme
```

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [Google Docstring Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Support

For documentation issues:
- Open an issue on GitHub
- Tag with `documentation` label
- Include build output if relevant
