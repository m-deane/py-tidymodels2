"""
Tests for parallel processing utilities.

Tests validation, CPU detection, and Windows compatibility functions.
"""

import pytest
import warnings
import os
from unittest.mock import patch

from py_tune.parallel_utils import (
    get_cpu_count,
    validate_n_jobs,
    get_joblib_backend,
    check_windows_compatibility,
    format_parallel_info
)


class TestGetCpuCount:
    """Test CPU count detection."""

    def test_returns_positive_integer(self):
        """CPU count should be positive integer."""
        count = get_cpu_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_fallback_on_error(self):
        """Should return 1 if detection fails."""
        with patch('os.cpu_count', return_value=None):
            count = get_cpu_count()
            assert count == 1


class TestValidateNJobs:
    """Test n_jobs validation and warnings."""

    def test_none_returns_1(self):
        """n_jobs=None should return 1 (sequential)."""
        result = validate_n_jobs(None, task_count=10)
        assert result == 1

    def test_1_returns_1(self):
        """n_jobs=1 should return 1 (sequential)."""
        result = validate_n_jobs(1, task_count=10)
        assert result == 1

    def test_positive_integer_returned(self):
        """Positive n_jobs should be returned as-is."""
        result = validate_n_jobs(4, task_count=10)
        assert result == 4

    def test_minus_1_resolves_to_all_cores(self):
        """n_jobs=-1 should resolve to all CPU cores."""
        cpu_count = get_cpu_count()
        result = validate_n_jobs(-1, task_count=10)
        assert result == cpu_count

    def test_minus_2_resolves_correctly(self):
        """n_jobs=-2 should resolve to all cores - 1."""
        cpu_count = get_cpu_count()
        expected = max(1, cpu_count - 1)
        result = validate_n_jobs(-2, task_count=10)
        assert result == expected

    def test_warns_when_n_jobs_exceeds_cpu_count(self):
        """Should warn when n_jobs > available CPU cores."""
        cpu_count = get_cpu_count()
        excessive_jobs = cpu_count + 2

        with pytest.warns(UserWarning, match="requested but only.*available"):
            validate_n_jobs(excessive_jobs, task_count=10)

    def test_warns_when_n_jobs_exceeds_task_count(self):
        """Should warn when n_jobs > task count."""
        with pytest.warns(UserWarning, match="greater than task count"):
            validate_n_jobs(10, task_count=5)

    def test_warns_when_minus_1_with_few_cores(self):
        """Should warn when n_jobs=-1 but only 1-2 cores available."""
        with patch('py_tune.parallel_utils.get_cpu_count', return_value=2):
            with pytest.warns(UserWarning, match="may not provide significant speedup"):
                validate_n_jobs(-1, task_count=10)

    def test_no_warning_for_reasonable_values(self):
        """Should not warn for reasonable n_jobs values."""
        cpu_count = get_cpu_count()
        reasonable_jobs = min(2, cpu_count)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            try:
                validate_n_jobs(reasonable_jobs, task_count=10)
            except UserWarning:
                pytest.fail("Unexpected warning for reasonable n_jobs value")

    def test_verbose_output(self, capsys):
        """Should print resolution message when verbose=True."""
        validate_n_jobs(-2, task_count=10, verbose=True)
        captured = capsys.readouterr()
        assert "Resolved n_jobs=" in captured.out


class TestGetJoblibBackend:
    """Test joblib backend selection."""

    def test_returns_loky(self):
        """Should return 'loky' backend for all platforms."""
        backend = get_joblib_backend()
        assert backend == 'loky'


class TestCheckWindowsCompatibility:
    """Test Windows compatibility checking."""

    def test_returns_true(self):
        """Should always return True with loky backend."""
        result = check_windows_compatibility()
        assert result is True

    def test_verbose_output_on_windows(self, capsys):
        """Should print Windows-specific info when verbose=True on Windows."""
        with patch('sys.platform', 'win32'):
            check_windows_compatibility(verbose=True)
            captured = capsys.readouterr()
            assert "Windows" in captured.out
            assert "loky" in captured.out

    def test_no_output_when_not_verbose(self, capsys):
        """Should not print when verbose=False."""
        check_windows_compatibility(verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestFormatParallelInfo:
    """Test parallel execution info formatting."""

    def test_sequential_format(self):
        """Should format sequential execution info."""
        info = format_parallel_info(1, 10, "CV folds")
        assert info == "Processing 10 CV folds (sequential)"

    def test_parallel_format_fewer_cores_than_available(self):
        """Should format parallel execution with cores used/total."""
        cpu_count = get_cpu_count()
        n_jobs = max(2, cpu_count - 1)  # Use fewer cores than available

        info = format_parallel_info(n_jobs, 10, "workflows")
        assert "Processing 10 workflows using" in info
        assert f"{n_jobs}/{cpu_count} cores" in info
        assert "(loky backend)" in info

    def test_parallel_format_all_cores(self):
        """Should format parallel execution when using all cores."""
        cpu_count = get_cpu_count()

        info = format_parallel_info(cpu_count, 10, "grid fits")
        assert "Processing 10 grid fits using" in info
        assert f"{cpu_count} cores" in info
        assert "(loky backend)" in info

    def test_parallel_format_more_cores_than_available(self):
        """Should format parallel execution even when n_jobs > cpu_count."""
        cpu_count = get_cpu_count()
        excessive_jobs = cpu_count + 2

        info = format_parallel_info(excessive_jobs, 10, "tasks")
        assert "Processing 10 tasks using" in info
        assert f"{excessive_jobs} cores" in info
        assert "(loky backend)" in info


class TestValidateNJobsIntegration:
    """Integration tests for validate_n_jobs with various scenarios."""

    def test_typical_grid_search_scenario(self):
        """Test typical grid search: 25 configs × 5 folds = 125 tasks."""
        cpu_count = get_cpu_count()
        # User wants to use all cores
        effective = validate_n_jobs(-1, task_count=125)
        assert effective == cpu_count

    def test_small_task_count_scenario(self):
        """Test scenario where task count < n_jobs."""
        # Only 3 CV folds, but user requests 8 cores
        with pytest.warns(UserWarning, match="greater than task count"):
            effective = validate_n_jobs(8, task_count=3)
            assert effective == 8  # Still returns requested value

    def test_sequential_default_scenario(self):
        """Test default sequential behavior."""
        effective = validate_n_jobs(None, task_count=100)
        assert effective == 1

    def test_explicit_sequential_scenario(self):
        """Test explicit sequential request."""
        effective = validate_n_jobs(1, task_count=100)
        assert effective == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
