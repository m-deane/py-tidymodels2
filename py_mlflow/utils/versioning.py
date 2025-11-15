"""
Version compatibility checking for MLflow model persistence.
"""

import warnings
from packaging import version
from typing import Dict, Any


def check_version_compatibility(flavor_conf: Dict[str, Any], current_version: str) -> None:
    """
    Check if current py-tidymodels version is compatible with saved model.

    Args:
        flavor_conf: Flavor configuration from MLmodel file
        current_version: Current py-tidymodels version string

    Raises:
        ValueError: If version incompatibility would cause errors

    Warnings:
        UserWarning: If version mismatch might cause issues but isn't fatal
    """
    model_version_str = flavor_conf.get("py_tidymodels_version", "0.0.0")

    current_ver = version.parse(current_version)
    model_ver = version.parse(model_version_str)

    # Check major version compatibility
    if current_ver.major != model_ver.major:
        warnings.warn(
            f"Model was trained with py-tidymodels {model_ver}, "
            f"but current version is {current_ver}. "
            "Major version mismatch may cause compatibility issues.",
            UserWarning
        )

    # Check for version constraints
    min_version = flavor_conf.get("min_py_tidymodels_version")
    max_version = flavor_conf.get("max_py_tidymodels_version")

    if min_version and current_ver < version.parse(min_version):
        raise ValueError(
            f"Model requires py-tidymodels >= {min_version}, "
            f"but current version is {current_ver}"
        )

    if max_version and current_ver > version.parse(max_version):
        warnings.warn(
            f"Model was tested with py-tidymodels <= {max_version}, "
            f"but current version is {current_ver}. "
            "Compatibility not guaranteed.",
            UserWarning
        )


def get_version_metadata(current_version: str) -> Dict[str, str]:
    """
    Get version metadata for storing in MLflow model.

    Args:
        current_version: Current py-tidymodels version

    Returns:
        Dict with version metadata
    """
    return {
        "py_tidymodels_version": current_version,
        "min_py_tidymodels_version": current_version,
        # Allow patch version upgrades
        "max_py_tidymodels_version": f"{version.parse(current_version).major}.{version.parse(current_version).minor + 1}.0"
    }
