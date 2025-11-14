#!/usr/bin/env python
"""
Comprehensive test script for all conformal prediction notebooks.

This script:
1. Tests each notebook individually
2. Captures detailed error information
3. Generates summary report
4. Identifies MAPIE installation issues
5. Provides actionable fix recommendations

Usage:
    python test_all_conformal_notebooks.py
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

# Configuration
NOTEBOOKS_DIR = Path("examples")
TIMEOUT_SECONDS = 900  # 15 minutes per notebook

# Conformal prediction notebooks (in logical testing order)
CONFORMAL_NOTEBOOKS = [
    # Priority 1: Simple demonstrations (created 2025-11-13)
    "24a_conformal_method_selection.ipynb",
    "24b_timeseries_conformal_enbpi.ipynb",
    "24c_feature_selection_conformal.ipynb",
    "24e_per_group_conformal.ipynb",

    # Priority 2: Real-world applications (existing)
    "23a_european_gas_conformal.ipynb",
    "23b_refinery_workflowset_conformal.ipynb",
    "23c_crude_production_method_comparison.ipynb",

    # Priority 3: Advanced (if created)
    "24d_conformal_calibration_strategies.ipynb",
    "24f_conformal_with_recipes.ipynb",
    "24g_multiple_confidence_levels.ipynb",
    "24h_cv_conformal_integration.ipynb",
    "24i_production_conformal_workflow.ipynb",
]

def check_mapie_installation():
    """Check if MAPIE is correctly installed."""
    try:
        result = subprocess.run(
            ["python", "-c", "from mapie.regression import MapieRegressor; print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception as e:
        return False

def test_notebook(notebook_path):
    """
    Test a single notebook execution.

    Returns:
        dict: Test result with status, error info, and execution time
    """
    print(f"\n{'='*80}")
    print(f"Testing: {notebook_path.name}")
    print(f"{'='*80}")

    if not notebook_path.exists():
        print(f"⚠️  SKIP: File not found")
        return {
            "notebook": notebook_path.name,
            "status": "SKIP",
            "reason": "File not found",
            "error": None,
            "time": 0
        }

    output_path = Path("/tmp") / notebook_path.name

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                str(notebook_path),
                "--output", str(output_path),
                "--ExecutePreprocessor.timeout", str(TIMEOUT_SECONDS)
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS + 60
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            print(f"✅ PASS ({elapsed:.1f}s)")
            return {
                "notebook": notebook_path.name,
                "status": "PASS",
                "reason": None,
                "error": None,
                "time": elapsed
            }
        else:
            # Extract error from stderr
            error_msg = result.stderr

            # Check for specific error types
            if "MapieRegressor" in error_msg or "mapie.regression" in error_msg:
                error_type = "MAPIE_IMPORT_ERROR"
                reason = "MAPIE installation issue"
            elif "FileNotFoundError" in error_msg:
                error_type = "FILE_NOT_FOUND"
                reason = "Missing data file"
            elif "ValueError: No conformal predictions" in error_msg:
                error_type = "NO_PREDICTIONS"
                reason = "Conformal prediction failed"
            elif "ModuleNotFoundError" in error_msg:
                error_type = "MODULE_MISSING"
                reason = "Missing Python module"
            else:
                error_type = "EXECUTION_ERROR"
                reason = "Notebook execution failed"

            print(f"❌ FAIL ({elapsed:.1f}s): {reason}")
            print(f"\nError excerpt:")
            print("-" * 80)
            # Print last 500 chars of error
            print(error_msg[-500:] if len(error_msg) > 500 else error_msg)
            print("-" * 80)

            return {
                "notebook": notebook_path.name,
                "status": "FAIL",
                "reason": reason,
                "error_type": error_type,
                "error": error_msg[-1000:],  # Last 1000 chars
                "time": elapsed
            }

    except subprocess.TimeoutExpired:
        elapsed = TIMEOUT_SECONDS
        print(f"⏱️  TIMEOUT ({elapsed:.1f}s)")
        return {
            "notebook": notebook_path.name,
            "status": "TIMEOUT",
            "reason": f"Exceeded {TIMEOUT_SECONDS}s timeout",
            "error": None,
            "time": elapsed
        }

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"💥 ERROR ({elapsed:.1f}s): {str(e)}")
        return {
            "notebook": notebook_path.name,
            "status": "ERROR",
            "reason": str(e),
            "error": str(e),
            "time": elapsed
        }

def generate_report(results, mapie_ok):
    """Generate comprehensive test report."""
    print("\n\n" + "="*80)
    print("CONFORMAL PREDICTION NOTEBOOKS - TEST REPORT")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MAPIE Installation: {'✅ OK' if mapie_ok else '❌ FAILED'}")
    print("="*80)

    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    timeout = sum(1 for r in results if r["status"] == "TIMEOUT")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\nSUMMARY:")
    print(f"  Total notebooks tested: {total}")
    print(f"  ✅ Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  ❌ Failed: {failed}/{total}")
    print(f"  ⏱️  Timeout: {timeout}/{total}")
    print(f"  ⚠️  Skipped: {skipped}/{total}")
    print(f"  💥 Errors: {errors}/{total}")

    # Detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS:")
    print(f"{'='*80}\n")

    for r in results:
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⚠️",
            "TIMEOUT": "⏱️",
            "ERROR": "💥"
        }.get(r["status"], "❓")

        print(f"{status_icon} {r['notebook']:<50} {r['status']:<10} ({r['time']:.1f}s)")
        if r["reason"]:
            print(f"   └─ {r['reason']}")

    # Error analysis
    if failed > 0 or errors > 0:
        print(f"\n{'='*80}")
        print("ERROR ANALYSIS:")
        print(f"{'='*80}\n")

        error_types = {}
        for r in results:
            if r["status"] in ["FAIL", "ERROR"] and "error_type" in r:
                et = r.get("error_type", "UNKNOWN")
                if et not in error_types:
                    error_types[et] = []
                error_types[et].append(r["notebook"])

        for error_type, notebooks in error_types.items():
            print(f"{error_type}:")
            for nb in notebooks:
                print(f"  - {nb}")
            print()

        # Recommendations
        print(f"{'='*80}")
        print("RECOMMENDED FIXES:")
        print(f"{'='*80}\n")

        if not mapie_ok or any(r.get("error_type") == "MAPIE_IMPORT_ERROR" for r in results):
            print("🔧 MAPIE Installation Issue:")
            print("   1. pip uninstall -y mapie")
            print("   2. pip cache purge")
            print("   3. pip install mapie==0.9.2")
            print("   4. Restart Jupyter kernel")
            print("   See: .claude_plans/CONFORMAL_NOTEBOOKS_FIX.md\n")

        if any(r.get("error_type") == "FILE_NOT_FOUND" for r in results):
            print("📁 Missing Data Files:")
            print("   Check that all data files exist in _md/__data/")
            print("   Verify file paths in notebook cells\n")

        if any(r.get("error_type") == "NO_PREDICTIONS" for r in results):
            print("🔍 Conformal Prediction Failures:")
            print("   This may be caused by MAPIE issues (see above)")
            print("   Or insufficient calibration data per group\n")

    # Success message
    if passed == total:
        print(f"\n{'='*80}")
        print("🎉 ALL TESTS PASSED! 🎉")
        print(f"{'='*80}\n")
        print("All conformal prediction notebooks are working correctly.")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "timeout": timeout,
        "errors": errors,
        "mapie_ok": mapie_ok
    }

def main():
    """Main test execution."""
    print("="*80)
    print("CONFORMAL PREDICTION NOTEBOOKS - COMPREHENSIVE TEST")
    print("="*80)

    # Check MAPIE installation
    print("\n1. Checking MAPIE installation...")
    mapie_ok = check_mapie_installation()
    if mapie_ok:
        print("   ✅ MAPIE is correctly installed")
    else:
        print("   ❌ MAPIE installation issue detected")
        print("   See: .claude_plans/CONFORMAL_NOTEBOOKS_FIX.md for fix instructions")
        print("\n   Continuing with tests (expect failures)...\n")

    # Test each notebook
    print("\n2. Testing notebooks...")
    results = []

    for notebook_name in CONFORMAL_NOTEBOOKS:
        notebook_path = NOTEBOOKS_DIR / notebook_name
        result = test_notebook(notebook_path)
        results.append(result)

    # Generate report
    print("\n3. Generating report...")
    summary = generate_report(results, mapie_ok)

    # Save JSON report
    report_path = Path("conformal_notebooks_test_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "results": results
        }, f, indent=2)

    print(f"\n📄 Detailed JSON report saved to: {report_path}")

    # Return exit code
    return 0 if summary["passed"] == summary["total"] else 1

if __name__ == "__main__":
    exit(main())
