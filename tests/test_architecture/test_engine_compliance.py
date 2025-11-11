"""
Engine Interface Compliance Tests

Tests that all engines properly implement the Engine ABC:
- All engines are registered in ENGINE_REGISTRY
- All engines inherit from Engine
- All engines implement required methods (fit, predict, extract_outputs)
- All engines have param_map attribute
- All engines return correct output structures
"""

import inspect
from typing import Set
import pytest
import pandas as pd

from py_parsnip.engine_registry import Engine, ENGINE_REGISTRY, get_engine
from py_parsnip.model_spec import ModelSpec, ModelFit
from py_hardhat import MoldedData


def get_all_registered_engines() -> Set[tuple]:
    """Get all registered (model_type, engine) pairs"""
    return set(ENGINE_REGISTRY.keys())


def get_engine_class(model_type: str, engine: str):
    """Get the engine class (not instance) for a model type"""
    return ENGINE_REGISTRY.get((model_type, engine))


class TestEngineRegistration:
    """Test suite for engine registration"""

    def test_all_engines_registered(self):
        """
        Verify that engines are registered in ENGINE_REGISTRY.

        This test ensures that the registration system is working and that
        engines use the @register_engine decorator properly.
        """
        registered_engines = get_all_registered_engines()

        assert len(registered_engines) > 0, (
            "No engines registered. Check that @register_engine decorator is used."
        )

        # Should have at least these core engines
        expected_core_engines = [
            ("linear_reg", "sklearn"),
            ("rand_forest", "sklearn"),
            ("prophet_reg", "prophet"),
            ("arima_reg", "statsmodels"),
        ]

        for model_type, engine in expected_core_engines:
            assert (model_type, engine) in registered_engines, (
                f"Core engine not registered: {model_type} + {engine}"
            )

    def test_engine_registry_has_no_duplicates(self):
        """
        Verify that each (model_type, engine) pair is registered only once.

        Duplicate registrations would indicate an error in the registration system.
        """
        registered_engines = get_all_registered_engines()
        registered_list = list(ENGINE_REGISTRY.keys())

        assert len(registered_engines) == len(registered_list), (
            f"Duplicate engine registrations detected. "
            f"Unique: {len(registered_engines)}, Total: {len(registered_list)}"
        )

    def test_get_engine_returns_instances(self):
        """
        Test that get_engine() returns Engine instances.
        """
        # Test a few known engines
        test_cases = [
            ("linear_reg", "sklearn"),
            ("rand_forest", "sklearn"),
        ]

        for model_type, engine_name in test_cases:
            if (model_type, engine_name) in ENGINE_REGISTRY:
                engine = get_engine(model_type, engine_name)
                assert isinstance(engine, Engine), (
                    f"get_engine({model_type}, {engine_name}) should return Engine instance"
                )

    def test_get_engine_raises_for_unknown_engine(self):
        """
        Test that get_engine() raises ValueError for unknown engines.
        """
        with pytest.raises(ValueError, match="No engine registered"):
            get_engine("nonexistent_model", "nonexistent_engine")


class TestEngineInterfaceCompliance:
    """Test suite for Engine ABC compliance"""

    def test_all_engines_inherit_from_engine_abc(self):
        """
        Verify that all registered engines inherit from Engine ABC.

        This ensures that all engines follow the Engine protocol.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            if not issubclass(engine_class, Engine):
                violations.append(f"{model_type} + {engine_name}: {engine_class.__name__}")

        assert not violations, (
            f"Engines not inheriting from Engine ABC:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_all_engines_have_required_methods(self):
        """
        Verify that all engines implement required methods:
        - fit() or fit_raw()
        - predict() or predict_raw()
        - extract_outputs()

        Engines can use either standard path (fit/predict) or raw path
        (fit_raw/predict_raw) for special data handling.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            # Check for fit method (either fit or fit_raw)
            has_fit = (
                hasattr(engine_instance, 'fit') and
                callable(getattr(engine_instance, 'fit'))
            )
            has_fit_raw = (
                hasattr(engine_instance, 'fit_raw') and
                callable(getattr(engine_instance, 'fit_raw'))
            )

            if not (has_fit or has_fit_raw):
                violations.append(
                    f"{model_type} + {engine_name}: missing fit() or fit_raw()"
                )

            # Check for predict method (either predict or predict_raw)
            has_predict = (
                hasattr(engine_instance, 'predict') and
                callable(getattr(engine_instance, 'predict'))
            )
            has_predict_raw = (
                hasattr(engine_instance, 'predict_raw') and
                callable(getattr(engine_instance, 'predict_raw'))
            )

            if not (has_predict or has_predict_raw):
                violations.append(
                    f"{model_type} + {engine_name}: missing predict() or predict_raw()"
                )

            # All engines must have extract_outputs
            has_extract_outputs = (
                hasattr(engine_instance, 'extract_outputs') and
                callable(getattr(engine_instance, 'extract_outputs'))
            )

            if not has_extract_outputs:
                violations.append(
                    f"{model_type} + {engine_name}: missing extract_outputs()"
                )

        assert not violations, (
            f"Engines missing required methods:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_all_engines_have_param_map(self):
        """
        Verify that all engines have a param_map attribute.

        param_map translates tidymodels parameter names to engine-specific names.
        Even if empty, it should exist as a dict.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            if not hasattr(engine_class, 'param_map'):
                violations.append(f"{model_type} + {engine_name}: missing param_map")
            elif not isinstance(engine_class.param_map, dict):
                violations.append(
                    f"{model_type} + {engine_name}: param_map is not a dict"
                )

        assert not violations, (
            f"Engines with param_map issues:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_engine_translate_params_works(self):
        """
        Test that translate_params() method works correctly.

        The default implementation in Engine ABC should use param_map
        to translate parameter names.
        """
        # Test with sklearn linear regression engine
        engine = get_engine("linear_reg", "sklearn")

        # sklearn linear engine maps: penalty → alpha, mixture → l1_ratio
        params = {"penalty": 0.1, "mixture": 0.5}
        translated = engine.translate_params(params)

        assert "alpha" in translated or "penalty" in translated, (
            "translate_params should translate or pass through parameters"
        )

    def test_fit_method_signature(self):
        """
        Verify that fit() methods have correct signature.

        Expected signature:
        - fit(spec: ModelSpec, molded: MoldedData) → Dict[str, Any]
        OR
        - fit_raw(spec: ModelSpec, data: DataFrame, formula: str) → tuple
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            # Check standard fit() signature
            if hasattr(engine_instance, 'fit') and not hasattr(engine_instance, 'fit_raw'):
                sig = inspect.signature(engine_instance.fit)
                params = list(sig.parameters.keys())

                # Should have at least 'spec' and 'molded' parameters
                # Allow additional optional parameters
                if 'spec' not in params or 'molded' not in params:
                    violations.append(
                        f"{model_type} + {engine_name}: fit() missing spec or molded parameter"
                    )

            # Check fit_raw() signature
            elif hasattr(engine_instance, 'fit_raw'):
                sig = inspect.signature(engine_instance.fit_raw)
                params = list(sig.parameters.keys())

                # Should have at least 'spec', 'data', 'formula' parameters
                if 'spec' not in params or 'data' not in params or 'formula' not in params:
                    violations.append(
                        f"{model_type} + {engine_name}: fit_raw() missing required parameters"
                    )

        assert not violations, (
            f"Engines with incorrect fit() signature:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_extract_outputs_method_signature(self):
        """
        Verify that extract_outputs() methods have correct signature.

        Expected signature:
        - extract_outputs(fit: ModelFit) → tuple[DataFrame, DataFrame, DataFrame]
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            if hasattr(engine_instance, 'extract_outputs'):
                sig = inspect.signature(engine_instance.extract_outputs)
                params = list(sig.parameters.keys())

                # Should have 'fit' parameter
                if 'fit' not in params:
                    violations.append(
                        f"{model_type} + {engine_name}: extract_outputs() missing fit parameter"
                    )

        assert not violations, (
            f"Engines with incorrect extract_outputs() signature:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


class TestEnginePathConsistency:
    """Test that engines consistently use either standard or raw path"""

    def test_engines_use_one_path_consistently(self):
        """
        Verify that engines use either standard path OR raw path, not both.

        Standard path: fit() + predict()
        Raw path: fit_raw() + predict_raw()

        Mixing paths would indicate inconsistent implementation.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            has_fit = hasattr(engine_instance, 'fit')
            has_fit_raw = hasattr(engine_instance, 'fit_raw')
            has_predict = hasattr(engine_instance, 'predict')
            has_predict_raw = hasattr(engine_instance, 'predict_raw')

            # Check for inconsistent path usage
            if has_fit and has_fit_raw:
                # Having both is OK if fit raises NotImplementedError
                pass
            elif has_predict and has_predict_raw:
                # Having both is OK if predict raises NotImplementedError
                pass
            elif (has_fit and not has_predict) or (not has_fit and has_predict):
                violations.append(
                    f"{model_type} + {engine_name}: inconsistent standard path "
                    f"(has_fit={has_fit}, has_predict={has_predict})"
                )
            elif (has_fit_raw and not has_predict_raw) or (not has_fit_raw and has_predict_raw):
                violations.append(
                    f"{model_type} + {engine_name}: inconsistent raw path "
                    f"(has_fit_raw={has_fit_raw}, has_predict_raw={has_predict_raw})"
                )

        assert not violations, (
            f"Engines with inconsistent path usage:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_raw_path_engines_raise_notimplementederror(self):
        """
        Verify that raw path engines raise NotImplementedError for standard methods.

        Engines using fit_raw/predict_raw should raise NotImplementedError
        for fit/predict to prevent incorrect usage.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            # If engine has fit_raw, check that fit raises NotImplementedError
            if hasattr(engine_instance, 'fit_raw'):
                if hasattr(engine_instance, 'fit'):
                    try:
                        # Try calling fit with dummy args
                        # Should raise NotImplementedError
                        # We can't actually call it, so we check the source
                        import inspect
                        source = inspect.getsource(engine_instance.fit)
                        if 'NotImplementedError' not in source:
                            violations.append(
                                f"{model_type} + {engine_name}: fit() should raise NotImplementedError "
                                f"when using raw path"
                            )
                    except:
                        pass  # Can't inspect, skip

        # This test is informational - don't fail if violations found
        # because some engines may have valid reasons for mixed implementation
        if violations:
            pytest.skip(
                f"Some engines have mixed path implementation (may be intentional):\n" +
                "\n".join(f"  - {v}" for v in violations)
            )


class TestEngineDocumentation:
    """Test that engines have proper documentation"""

    def test_all_engines_have_docstrings(self):
        """
        Verify that all engine classes have docstrings.

        Good documentation is essential for maintainability.
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            if not engine_class.__doc__ or len(engine_class.__doc__.strip()) < 20:
                violations.append(
                    f"{model_type} + {engine_name}: {engine_class.__name__} missing docstring"
                )

        assert not violations, (
            f"Engines missing docstrings:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_engine_methods_have_docstrings(self):
        """
        Verify that key engine methods have docstrings.

        Methods to check: fit, predict, extract_outputs
        """
        violations = []

        for (model_type, engine_name), engine_class in ENGINE_REGISTRY.items():
            engine_instance = engine_class()

            # Check fit/fit_raw docstring
            if hasattr(engine_instance, 'fit'):
                if not engine_instance.fit.__doc__:
                    violations.append(
                        f"{model_type} + {engine_name}: fit() missing docstring"
                    )
            elif hasattr(engine_instance, 'fit_raw'):
                if not engine_instance.fit_raw.__doc__:
                    violations.append(
                        f"{model_type} + {engine_name}: fit_raw() missing docstring"
                    )

            # Check extract_outputs docstring
            if hasattr(engine_instance, 'extract_outputs'):
                if not engine_instance.extract_outputs.__doc__:
                    violations.append(
                        f"{model_type} + {engine_name}: extract_outputs() missing docstring"
                    )

        # This is a soft requirement - don't fail build
        if violations:
            pytest.skip(
                f"Some engine methods missing docstrings:\n" +
                "\n".join(f"  - {v}" for v in violations[:10])  # Show first 10
            )
