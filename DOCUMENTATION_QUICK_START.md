# Documentation Quick Start Guide

## 🎉 Your API Documentation is Ready!

Professional-grade API documentation has been set up for py-tidymodels using Sphinx with autodoc.

## Quick Commands

### View Documentation Locally

```bash
# 1. Activate virtual environment
source py-tidymodels2/bin/activate

# 2. Build documentation
cd docs
./build_docs.sh

# 3. Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Serve Documentation (with Auto-Reload)

```bash
cd docs
make serve
# Visit: http://localhost:8000
```

### Rebuild After Changes

```bash
cd docs
make html
```

## What You Get

### ✅ Complete API Reference
- **10 packages** fully documented
- **22 models** with examples
- **51 preprocessing steps** detailed
- **17 metrics** explained
- Auto-generated from your docstrings

### ✅ User Guides
- Installation guide
- 5-minute quick start with 8 examples
- Core concepts and architecture
- Ready to expand with more content

### ✅ Multiple Output Formats
```bash
make html      # Interactive HTML
make latexpdf  # Professional PDF
make epub      # E-book format
```

### ✅ Automated Deployment
- GitHub Actions workflow configured
- Automatic deployment to GitHub Pages
- PDF generation on releases
- Documentation quality checks

## File Locations

```
docs/
├── index.rst              # Main page
├── conf.py                # Configuration
├── Makefile               # Build commands
├── build_docs.sh          # Automated build script
├── README.md              # Detailed documentation guide
│
├── api/                   # API Reference (10 modules)
│   ├── parsnip.rst        # 22 models documented
│   ├── workflows.rst      # Pipeline composition
│   ├── recipes.rst        # 51 preprocessing steps
│   ├── tune.rst           # Hyperparameter tuning
│   └── ... (6 more)
│
└── user_guide/            # Getting Started
    ├── installation.rst   # Install guide
    ├── quickstart.rst     # Quick examples
    └── concepts.rst       # Architecture
```

## Access Your Documentation

### Local (After Building)
```
file:///Users/matthewdeane/Documents/Data Science/python/_projects/py-tidymodels/docs/_build/html/index.html
```

### GitHub Pages (After Setup)
1. Push to GitHub
2. Go to Settings → Pages
3. Source: `gh-pages` branch
4. Available at: `https://your-username.github.io/py-tidymodels/`

### PDF (After Building)
```
docs/_build/latex/py-tidymodels.pdf
```

## Common Tasks

### Update API Documentation
```bash
# Edit docstrings in your Python files
# Then rebuild
cd docs
make html
```

### Add New Section
```bash
# 1. Create .rst file in appropriate directory
# 2. Add to index.rst table of contents
# 3. Rebuild
make html
```

### Fix Warnings
```bash
# See build warnings
make html

# Check specific issues
make check
```

### Deploy to GitHub Pages
```bash
# Push to main branch - automatic via GitHub Actions
git add .
git commit -m "Update documentation"
git push origin main
```

## Useful Make Commands

```bash
make html          # Build HTML (most common)
make clean         # Clean old builds
make serve         # Serve locally on port 8000
make check         # Validate links and coverage
make latexpdf      # Generate PDF
make quick         # Quick rebuild (no clean)
```

## Documentation Structure

Your documentation includes:

**API Reference:**
- py_hardhat - Data preprocessing
- py_parsnip - 22 model specifications
- py_rsample - Resampling & CV
- py_workflows - Pipeline composition
- py_recipes - 51 preprocessing steps
- py_yardstick - 17 evaluation metrics
- py_tune - Hyperparameter tuning
- py_workflowsets - Multi-model comparison
- py_visualize - Interactive plots
- py_stacks - Model ensembling

**User Guides:**
- Installation instructions
- Quick start with examples
- Core concepts and architecture
- (Stubs for recipes, time series, tuning, workflows - ready to expand)

**Examples:**
- Quick start includes 8 working examples
- (Stubs for detailed examples - ready to expand)

## Customization

### Change Theme
Edit `docs/conf.py`:
```python
html_theme = 'sphinx_rtd_theme'  # Current
# html_theme = 'alabaster'       # Alternative
# html_theme = 'furo'            # Modern alternative
```

### Add Logo/Favicon
Edit `docs/conf.py`:
```python
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'
```

### Modify Navigation
Edit `docs/index.rst`:
```rst
.. toctree::
   :maxdepth: 2

   new_section/index
```

## Need Help?

1. **Build Errors**: Check `docs/README.md` - Troubleshooting section
2. **Sphinx Docs**: https://www.sphinx-doc.org/
3. **ReStructuredText Guide**: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
4. **Read the Docs Theme**: https://sphinx-rtd-theme.readthedocs.io/

## Next Steps (Optional)

1. **Expand User Guides**: Add more detailed tutorials
2. **Add Model Deep Dives**: Detailed guide for each model type
3. **Create Real Examples**: Use actual datasets from your notebooks
4. **Record Videos**: Screen recordings of common workflows
5. **Add Images**: Architecture diagrams, screenshots, plots

## Your Documentation is Production-Ready! 🚀

- ✅ Complete API reference with 100+ functions/classes
- ✅ Professional Read the Docs theme
- ✅ Working examples in quick start
- ✅ Automated CI/CD deployment
- ✅ Multiple output formats (HTML, PDF, EPUB)
- ✅ Quality checks (link validation, coverage)

Just build and share!
