#!/bin/bash
# Script to build py-tidymodels documentation

set -e  # Exit on error

echo "======================================================================"
echo "py-tidymodels Documentation Build Script"
echo "======================================================================"

# Change to docs directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment not activated!"
    echo "Please activate your virtual environment first:"
    echo "  source ../py-tidymodels2/bin/activate"
    exit 1
fi

# Install documentation dependencies
echo ""
echo "Step 1: Installing documentation dependencies..."
echo "----------------------------------------------------------------------"
pip install -q -r requirements.txt
echo "✓ Documentation dependencies installed"

# Clean previous builds
echo ""
echo "Step 2: Cleaning previous builds..."
echo "----------------------------------------------------------------------"
make clean
echo "✓ Clean complete"

# Build HTML documentation
echo ""
echo "Step 3: Building HTML documentation..."
echo "----------------------------------------------------------------------"
make html

if [ $? -eq 0 ]; then
    echo "✓ HTML documentation built successfully!"
else
    echo "✗ HTML build failed!"
    exit 1
fi

# Check for broken links and coverage
echo ""
echo "Step 4: Running documentation checks..."
echo "----------------------------------------------------------------------"
make check || echo "⚠ Some documentation issues found (see output above)"

# Display results
echo ""
echo "======================================================================"
echo "Documentation Build Complete!"
echo "======================================================================"
echo ""
echo "HTML documentation: file://$SCRIPT_DIR/_build/html/index.html"
echo ""
echo "To view the documentation:"
echo "  1. Open in browser:"
echo "     open _build/html/index.html (macOS)"
echo "     xdg-open _build/html/index.html (Linux)"
echo "     start _build/html/index.html (Windows)"
echo ""
echo "  2. Or start local server:"
echo "     make serve"
echo "     Then visit: http://localhost:8000"
echo ""
echo "======================================================================"
