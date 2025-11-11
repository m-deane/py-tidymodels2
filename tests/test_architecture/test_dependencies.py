"""
Architecture Layer Dependency Tests

Tests that enforce layered architecture constraints:
- Layer 1 (hardhat) has no upward dependencies
- Layer 2 (parsnip) depends only on hardhat
- Layer 3 (rsample) has no cross-layer dependencies
- Layer 4 (workflows) depends only on parsnip and recipes
- Layer 5 (recipes) has appropriate dependencies
- No circular dependencies exist
"""

import ast
import importlib
import sys
from pathlib import Path
from typing import Set, Dict, List
import pytest


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def get_module_imports(module_path: Path) -> Set[str]:
    """
    Extract import statements from a Python file.

    Args:
        module_path: Path to Python file

    Returns:
        Set of imported module names
    """
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(module_path))
    except SyntaxError:
        # Skip files with syntax errors
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    return imports


def get_all_python_files(directory: Path) -> List[Path]:
    """Get all Python files in a directory recursively"""
    return [p for p in directory.rglob("*.py") if p.name != "__init__.py"]


def get_layer_dependencies(layer_name: str) -> Set[str]:
    """
    Get all imports from a layer's Python files.

    Args:
        layer_name: Layer directory name (e.g., "py_hardhat", "py_parsnip")

    Returns:
        Set of imported module names
    """
    project_root = get_project_root()
    layer_path = project_root / layer_name

    if not layer_path.exists():
        return set()

    all_imports = set()
    for py_file in get_all_python_files(layer_path):
        all_imports.update(get_module_imports(py_file))

    return all_imports


class TestLayerDependencies:
    """Test suite for layer dependency constraints"""

    def test_hardhat_has_no_upward_dependencies(self):
        """
        Layer 1 (hardhat) should not depend on any higher layers.

        Hardhat is the foundation layer and should only depend on:
        - Standard library
        - Third-party packages (pandas, numpy, patsy)
        """
        hardhat_imports = get_layer_dependencies("py_hardhat")

        # Higher layers that hardhat must NOT import
        forbidden_imports = {
            "py_parsnip",
            "py_rsample",
            "py_workflows",
            "py_recipes",
            "py_yardstick",
            "py_tune",
            "py_workflowsets",
            "py_stacks",
            "py_visualize"
        }

        violations = hardhat_imports & forbidden_imports
        assert not violations, (
            f"py_hardhat imports from higher layers: {violations}. "
            f"Hardhat is Layer 1 and must not depend on upper layers."
        )

    def test_parsnip_depends_only_on_hardhat(self):
        """
        Layer 2 (parsnip) should only depend on Layer 1 (hardhat).

        Parsnip should not depend on:
        - workflows (Layer 4)
        - recipes (Layer 5)
        - tune (Layer 7)
        - workflowsets (Layer 8)
        """
        parsnip_imports = get_layer_dependencies("py_parsnip")

        # Allowed: hardhat
        # Forbidden: workflows, recipes, tune, workflowsets, etc.
        forbidden_imports = {
            "py_workflows",
            "py_recipes",
            "py_tune",
            "py_workflowsets",
            "py_stacks",
            "py_visualize"
        }

        violations = parsnip_imports & forbidden_imports
        assert not violations, (
            f"py_parsnip imports from inappropriate layers: {violations}. "
            f"Parsnip should only depend on hardhat."
        )

    def test_rsample_has_minimal_dependencies(self):
        """
        Layer 3 (rsample) should have minimal cross-layer dependencies.

        Rsample provides resampling infrastructure and should not depend on:
        - parsnip (Layer 2)
        - workflows (Layer 4)
        - recipes (Layer 5)
        """
        rsample_imports = get_layer_dependencies("py_rsample")

        forbidden_imports = {
            "py_parsnip",
            "py_workflows",
            "py_recipes",
            "py_tune",
            "py_workflowsets"
        }

        violations = rsample_imports & forbidden_imports
        assert not violations, (
            f"py_rsample imports from inappropriate layers: {violations}. "
            f"Rsample should be independent of model layers."
        )

    def test_workflows_depends_only_on_parsnip_recipes(self):
        """
        Layer 4 (workflows) should depend only on parsnip and recipes.

        Workflows orchestrate preprocessing and modeling, so it should depend on:
        - parsnip (Layer 2) - for models
        - recipes (Layer 5) - for preprocessing

        But NOT on:
        - tune (Layer 7) - higher layer
        - workflowsets (Layer 8) - higher layer
        """
        workflows_imports = get_layer_dependencies("py_workflows")

        # Should import parsnip and recipes
        assert "py_parsnip" in workflows_imports, (
            "py_workflows should import py_parsnip for model specs"
        )
        assert "py_recipes" in workflows_imports, (
            "py_workflows should import py_recipes for preprocessing"
        )

        # Should NOT import higher layers
        forbidden_imports = {
            "py_tune",
            "py_workflowsets",
            "py_stacks"
        }

        violations = workflows_imports & forbidden_imports
        assert not violations, (
            f"py_workflows imports from higher layers: {violations}. "
            f"Workflows should not depend on tune/workflowsets."
        )

    def test_recipes_has_no_model_dependencies(self):
        """
        Layer 5 (recipes) should not depend on model layers.

        Recipes provide preprocessing and should not depend on:
        - parsnip (Layer 2)
        - workflows (Layer 4)
        """
        recipes_imports = get_layer_dependencies("py_recipes")

        forbidden_imports = {
            "py_parsnip",
            "py_workflows",
            "py_tune",
            "py_workflowsets"
        }

        violations = recipes_imports & forbidden_imports
        assert not violations, (
            f"py_recipes imports from model layers: {violations}. "
            f"Recipes should be independent preprocessing layer."
        )

    def test_yardstick_is_independent(self):
        """
        Layer 6 (yardstick) should be independent of model layers.

        Yardstick provides metrics and should not depend on:
        - parsnip
        - workflows
        - recipes
        """
        yardstick_imports = get_layer_dependencies("py_yardstick")

        forbidden_imports = {
            "py_parsnip",
            "py_workflows",
            "py_recipes",
            "py_tune",
            "py_workflowsets"
        }

        violations = yardstick_imports & forbidden_imports
        assert not violations, (
            f"py_yardstick imports from model layers: {violations}. "
            f"Yardstick should be independent metrics layer."
        )

    def test_tune_depends_on_lower_layers(self):
        """
        Layer 7 (tune) should depend on lower layers appropriately.

        Tune should depend on:
        - workflows (Layer 4)
        - rsample (Layer 3)
        - yardstick (Layer 6)

        But NOT on:
        - workflowsets (Layer 8) - higher layer
        """
        tune_imports = get_layer_dependencies("py_tune")

        # Should import workflows, rsample, yardstick
        assert "py_workflows" in tune_imports or "py_parsnip" in tune_imports, (
            "py_tune should import py_workflows or py_parsnip for model specs"
        )

        # Should NOT import higher layers
        forbidden_imports = {"py_workflowsets"}

        violations = tune_imports & forbidden_imports
        assert not violations, (
            f"py_tune imports from higher layers: {violations}. "
            f"Tune should not depend on workflowsets."
        )

    def test_no_circular_imports(self):
        """
        Test that no circular import dependencies exist.

        This test attempts to import all main modules in sequence to detect
        circular dependencies that would cause ImportError.
        """
        project_root = get_project_root()

        # Add project root to path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # List of modules to test in dependency order
        modules_to_test = [
            "py_hardhat",
            "py_parsnip",
            "py_rsample",
            "py_recipes",
            "py_workflows",
            "py_yardstick",
            "py_tune",
            "py_workflowsets"
        ]

        # Try importing each module
        for module_name in modules_to_test:
            try:
                # Remove from cache if already imported
                if module_name in sys.modules:
                    del sys.modules[module_name]

                # Try importing
                module = importlib.import_module(module_name)
                assert module is not None, f"Failed to import {module_name}"

            except ImportError as e:
                pytest.fail(f"Circular import detected in {module_name}: {e}")

    def test_layer_dependency_order(self):
        """
        Test that layers follow strict ordering:

        Layer 1: hardhat (foundation)
        Layer 2: parsnip (models)
        Layer 3: rsample (resampling)
        Layer 4: workflows (orchestration)
        Layer 5: recipes (preprocessing)
        Layer 6: yardstick (metrics)
        Layer 7: tune (hyperparameter tuning)
        Layer 8: workflowsets (multi-model comparison)

        Lower layers should never depend on higher layers.
        """
        layer_hierarchy = {
            "py_hardhat": 1,
            "py_parsnip": 2,
            "py_rsample": 3,
            "py_workflows": 4,
            "py_recipes": 5,
            "py_yardstick": 6,
            "py_tune": 7,
            "py_workflowsets": 8
        }

        violations = []

        for layer_name, layer_level in layer_hierarchy.items():
            layer_imports = get_layer_dependencies(layer_name)

            # Check if this layer imports any higher layer
            for imported_layer, imported_level in layer_hierarchy.items():
                if imported_layer in layer_imports and imported_level > layer_level:
                    violations.append(
                        f"Layer {layer_level} ({layer_name}) imports from "
                        f"Layer {imported_level} ({imported_layer})"
                    )

        assert not violations, (
            f"Layer ordering violations detected:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestModuleImportability:
    """Test that all modules can be imported without errors"""

    def test_hardhat_imports_cleanly(self):
        """Test that py_hardhat can be imported"""
        import py_hardhat
        assert hasattr(py_hardhat, 'mold')
        assert hasattr(py_hardhat, 'forge')
        assert hasattr(py_hardhat, 'Blueprint')
        assert hasattr(py_hardhat, 'MoldedData')

    def test_parsnip_imports_cleanly(self):
        """Test that py_parsnip can be imported"""
        import py_parsnip
        assert hasattr(py_parsnip, 'ModelSpec')
        assert hasattr(py_parsnip, 'ModelFit')
        assert hasattr(py_parsnip, 'linear_reg')

    def test_workflows_imports_cleanly(self):
        """Test that py_workflows can be imported"""
        import py_workflows
        assert hasattr(py_workflows, 'Workflow')
        assert hasattr(py_workflows, 'WorkflowFit')

    def test_recipes_imports_cleanly(self):
        """Test that py_recipes can be imported"""
        import py_recipes
        assert hasattr(py_recipes, 'Recipe')
        assert hasattr(py_recipes, 'recipe')

    def test_rsample_imports_cleanly(self):
        """Test that py_rsample can be imported"""
        import py_rsample
        assert hasattr(py_rsample, 'initial_time_split')
        assert hasattr(py_rsample, 'time_series_cv')

    def test_yardstick_imports_cleanly(self):
        """Test that py_yardstick can be imported"""
        import py_yardstick
        assert hasattr(py_yardstick, 'rmse')
        assert hasattr(py_yardstick, 'mae')

    def test_tune_imports_cleanly(self):
        """Test that py_tune can be imported"""
        import py_tune
        assert hasattr(py_tune, 'tune')
        assert hasattr(py_tune, 'tune_grid')

    def test_workflowsets_imports_cleanly(self):
        """Test that py_workflowsets can be imported"""
        import py_workflowsets
        assert hasattr(py_workflowsets, 'WorkflowSet')
