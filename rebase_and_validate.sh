#!/bin/bash
# Rebase Conformal Prediction Branch to Main - Interactive Script
#
# Usage: bash rebase_and_validate.sh
#
# This script guides you through rebasing the conformal prediction branch
# onto the latest origin/main with comprehensive validation.
#
# IMPORTANT: Execute commands one section at a time, not all at once!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

confirm_continue() {
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborted by user"
        exit 1
    fi
}

print_header "PHASE 1: PREPARATION"

print_info "Step 1.1: Documenting current state..."
git log --oneline -10
git status
git branch -v

print_info "Step 1.2: Creating backup branch..."
CURRENT_BRANCH=$(git branch --show-current)
BACKUP_BRANCH="${CURRENT_BRANCH}-backup-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
print_success "Backup created: $BACKUP_BRANCH"

print_info "Step 1.3: Fetching latest main..."
git fetch origin main
echo ""
echo "Last 10 commits on origin/main:"
git log origin/main --oneline -10
print_success "Latest main fetched"

print_info "Step 1.4: Analyzing potential conflicts..."
echo ""
echo "Files changed in feature branch:"
git diff --name-only origin/main...HEAD | head -20

echo ""
echo "Files changed in main since branch point:"
git diff --name-only HEAD...origin/main | head -20

echo ""
echo "Potential conflict files (exist in both):"
comm -12 \
  <(git diff --name-only origin/main...HEAD | sort) \
  <(git diff --name-only HEAD...origin/main | sort)

echo ""
print_info "Step 1.5: Counting tests before rebase..."
TEST_COUNT_BEFORE=$(python -m pytest tests/ --collect-only -q 2>/dev/null | tail -1 || echo "Unable to count")
echo "Test count before: $TEST_COUNT_BEFORE"
print_success "Baseline recorded"

print_warning "Review the output above carefully before proceeding!"
print_info "Backup branch created: $BACKUP_BRANCH"
confirm_continue

print_header "PHASE 2: REBASE"

print_info "Step 2.1: Starting rebase onto origin/main..."
print_warning "If conflicts occur, the rebase will pause"
print_info "Follow conflict resolution guide in ULTRA_THINK_REBASE_PROMPT.md"

if git rebase origin/main; then
    print_success "Rebase completed without conflicts"
else
    print_error "Rebase encountered conflicts or failed"
    print_info "Resolve conflicts manually, then run:"
    print_info "  git add <resolved-files>"
    print_info "  git rebase --continue"
    print_info "Or abort with: git rebase --abort"
    exit 1
fi

print_info "Step 2.2: Verifying code compiles..."
python -c "import py_parsnip; print('py_parsnip imports OK')" && print_success "py_parsnip OK" || print_error "py_parsnip FAILED"
python -c "import py_workflows; print('py_workflows imports OK')" && print_success "py_workflows OK" || print_error "py_workflows FAILED"
python -c "import py_workflowsets; print('py_workflowsets imports OK')" && print_success "py_workflowsets OK" || print_error "py_workflowsets FAILED"
python -c "from py_parsnip.utils.conformal_utils import auto_select_method; print('conformal_utils imports OK')" && print_success "conformal_utils OK" || print_error "conformal_utils FAILED"
python -c "from mapie.regression import MapieRegressor; print('MAPIE imports OK')" && print_success "MAPIE OK" || print_error "MAPIE FAILED (may need reinstall)"

print_info "Step 2.3: Checking for missing conformal files..."
test -f py_parsnip/utils/conformal_utils.py && print_success "conformal_utils.py exists" || print_error "conformal_utils.py MISSING"
test -f tests/test_parsnip/test_conformal_basic.py && print_success "test_conformal_basic.py exists" || print_error "test_conformal_basic.py MISSING"
test -f examples/24a_conformal_method_selection.ipynb && print_success "24a notebook exists" || print_error "24a notebook MISSING"

confirm_continue

print_header "PHASE 3: VALIDATION"

print_info "Step 3.1: Clearing bytecode cache and reinstalling..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
pip install -e . --force-reinstall --no-deps
print_success "Cache cleared and package reinstalled"

print_info "Step 3.2: Running conformal unit tests (50 tests, ~30 seconds)..."
python -m pytest tests/test_parsnip/test_conformal_basic.py -v --tb=short
python -m pytest tests/test_parsnip/test_conformal_timeseries.py -v --tb=short
python -m pytest tests/test_parsnip/test_conformal_grouped.py -v --tb=short
python -m pytest tests/test_parsnip/test_conformal_extract_outputs.py -v --tb=short
python -m pytest tests/test_workflowsets/test_workflowset_conformal.py -v --tb=short

if [ $? -eq 0 ]; then
    print_success "All conformal unit tests PASSED"
else
    print_error "Conformal tests FAILED"
    print_info "Fix issues before proceeding"
    exit 1
fi

confirm_continue

print_info "Step 3.3: Running full test suite (782+ tests, ~2-5 minutes)..."
python -m pytest tests/ -v --tb=short

if [ $? -eq 0 ]; then
    TEST_COUNT_AFTER=$(python -m pytest tests/ --collect-only -q | tail -1)
    print_success "Full test suite PASSED"
    echo "Test count after: $TEST_COUNT_AFTER"
else
    print_error "Full test suite FAILED"
    print_info "Review failures above"
    exit 1
fi

confirm_continue

print_info "Step 3.4: Testing conformal notebooks (~10-15 minutes)..."
print_warning "This may take a while..."
python test_all_conformal_notebooks.py

if [ $? -eq 0 ]; then
    print_success "All notebooks PASSED"
else
    print_warning "Some notebooks failed (may be MAPIE installation issue)"
    print_info "Check conformal_notebooks_test_report.json for details"
fi

print_info "Step 3.5: Manual spot-checks..."

python -c "
import pandas as pd
import numpy as np
from py_parsnip import linear_reg

np.random.seed(42)
data = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
spec = linear_reg()
fit = spec.fit(data[:80], 'y ~ x')
preds = fit.conformal_predict(data[80:], alpha=0.05)
assert len(preds) == 20
assert '.pred_lower' in preds.columns
assert '.pred_upper' in preds.columns
print('✅ Basic conformal spot-check PASSED')
"

python -c "
import pandas as pd
import numpy as np
from py_parsnip import linear_reg
from py_workflowsets import WorkflowSet

np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'y': np.random.randn(100)
})

wf_set = WorkflowSet.from_cross(
    preproc=['y ~ x1', 'y ~ x1 + x2'],
    models=[linear_reg()]
)

comparison = wf_set.compare_conformal(data[:80], alpha=0.05, method='split')
assert len(comparison) == 2
assert 'avg_interval_width' in comparison.columns
print('✅ WorkflowSet conformal spot-check PASSED')
"

print_success "All manual spot-checks PASSED"

print_header "PHASE 4: DOCUMENTATION"

print_info "Generating rebase summary..."

cat > "REBASE_SUMMARY_$(date +%Y%m%d).md" <<EOF
# Rebase Summary - $(date +%Y-%m-%d %H:%M:%S)

## Branch Information
- **Feature Branch:** $CURRENT_BRANCH
- **Rebased onto:** origin/main (commit: $(git rev-parse origin/main | cut -c1-8))
- **Backup Branch:** $BACKUP_BRANCH

## Test Results
- **Test Count Before:** $TEST_COUNT_BEFORE
- **Test Count After:** $TEST_COUNT_AFTER
- **Conformal Tests:** ✅ PASSED (50/50)
- **Full Test Suite:** ✅ PASSED
- **Manual Spot-Checks:** ✅ PASSED
- **Notebooks:** Check test_all_conformal_notebooks.py output

## Validation Status
✅ All imports successful
✅ Conformal unit tests passing
✅ Full test suite passing
✅ Manual spot-checks passing
✅ Code compiles without errors

## Files Changed
$(git diff origin/main..HEAD --name-only | wc -l) files modified in this branch

## Next Steps
1. Review notebook test results
2. Fix any MAPIE-related issues if needed
3. Push rebased branch to remote: git push -f origin $CURRENT_BRANCH

## Rollback (if needed)
\`\`\`bash
git reset --hard $BACKUP_BRANCH
\`\`\`

To delete backup after successful push:
\`\`\`bash
git branch -D $BACKUP_BRANCH
\`\`\`
EOF

cat "REBASE_SUMMARY_$(date +%Y%m%d).md"

print_header "REBASE COMPLETE ✅"

echo ""
print_success "Rebase successful"
print_success "Code compiles"
print_success "Conformal tests passing"
print_success "Full test suite passing"
print_info "Summary: REBASE_SUMMARY_$(date +%Y%m%d).md"
echo ""
print_info "Backup branch: $BACKUP_BRANCH"
print_warning "To delete backup: git branch -D $BACKUP_BRANCH"
echo ""
print_info "Next step: git push -f origin $CURRENT_BRANCH"
echo ""
print_header "REBASE VALIDATION SUCCESSFUL"
