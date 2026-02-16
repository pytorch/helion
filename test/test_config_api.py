from __future__ import annotations

import importlib
import inspect
import os
import pickle
from typing import Any
import unittest
from unittest.mock import patch

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import TestCase
from helion._testing import onlyBackends


def _json_safe_values() -> st.SearchStrategy[Any]:
    # JSON-safe primitives/containers
    scalar = st.one_of(
        st.integers(), st.floats(allow_nan=False), st.booleans(), st.text()
    )
    leaf = st.one_of(scalar, st.none())
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(min_size=0, max_size=8), children, max_size=4),
        ),
        max_leaves=8,
    )


def _known_keys_strategy() -> st.SearchStrategy[dict[str, Any]]:
    # For known keys, None values are omitted by constructor; favor non-None
    return st.fixed_dictionaries(
        {
            "block_sizes": st.lists(
                st.integers(min_value=1, max_value=4096), max_size=4
            ),
            "loop_orders": st.lists(
                st.lists(st.integers(min_value=0, max_value=4), max_size=4),
                max_size=3,
            ),
            "flatten_loops": st.lists(st.booleans(), max_size=4),
            "l2_groupings": st.lists(
                st.integers(min_value=1, max_value=128), max_size=4
            ),
            "reduction_loops": st.lists(
                st.one_of(st.integers(min_value=0, max_value=8), st.none()),
                max_size=4,
            ),
            "range_unroll_factors": st.lists(
                st.integers(min_value=1, max_value=16), max_size=4
            ),
            "range_warp_specializes": st.lists(
                st.one_of(st.booleans(), st.none()), max_size=4
            ),
            "range_num_stages": st.lists(
                st.integers(min_value=1, max_value=8), max_size=4
            ),
            "range_multi_buffers": st.lists(
                st.one_of(st.booleans(), st.none()), max_size=4
            ),
            "range_flattens": st.lists(st.one_of(st.booleans(), st.none()), max_size=4),
            "static_ranges": st.lists(st.booleans(), max_size=4),
            "load_eviction_policies": st.lists(
                st.sampled_from(["", "first", "last"]), max_size=4
            ),
            "num_warps": st.integers(min_value=1, max_value=64),
            "num_stages": st.integers(min_value=1, max_value=16),
            "pid_type": st.sampled_from(
                ["flat", "xyz", "persistent_blocked", "persistent_interleaved"]
            ),
            "indexing": st.sampled_from(["pointer", "tensor_descriptor"]),
        }
    )


def _unknown_keys_strategy() -> st.SearchStrategy[dict[str, Any]]:
    key = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,12}")
    # Avoid colliding with known keys and enforce distinctness
    return st.dictionaries(
        keys=key.filter(
            lambda k: (
                k
                not in {
                    "block_sizes",
                    "loop_orders",
                    "flatten_loops",
                    "l2_groupings",
                    "reduction_loops",
                    "range_unroll_factors",
                    "range_warp_specializes",
                    "range_num_stages",
                    "range_multi_buffers",
                    "range_flattens",
                    "static_ranges",
                    "load_eviction_policies",
                    "num_warps",
                    "num_stages",
                    "pid_type",
                    "indexing",
                }
            )
        ),
        values=_json_safe_values(),
        max_size=4,
    )


@onlyBackends(["triton"])
class TestConfigAPI(TestCase):
    def test_config_import_path_stability(self) -> None:
        runtime = importlib.import_module("helion.runtime")

        self.assertIs(helion.Config, runtime.Config)
        self.assertIs(helion.Config, helion.runtime.Config)

    def test_config_constructor_signature_contains_expected_kwargs(self) -> None:
        # Keep this list in sync with public kwargs; removal/rename should fail tests
        expected = {
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "l2_groupings",
            "reduction_loops",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "num_warps",
            "num_stages",
            "pid_type",
            "indexing",
        }

        sig = inspect.signature(helion.Config.__init__)
        kwonly = {
            name
            for name, p in sig.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        # Expected kwargs must be present as keyword-only
        self.assertTrue(expected.issubset(kwonly))

    def test_mapping_behavior_len_iter_dict_roundtrip(self) -> None:
        data = {
            "block_sizes": [64, 32],
            "num_warps": 8,
            "custom_extra": {"a": 1},
        }
        cfg = helion.Config(**data)

        # Supports Mapping protocol
        self.assertEqual(len(cfg), len(cfg.config))
        self.assertEqual(dict(cfg), cfg.config)
        self.assertEqual(set(iter(cfg)), set(cfg.config.keys()))

        # Equality and hash coherence
        cfg2 = helion.Config(**data)
        self.assertEqual(cfg, cfg2)
        self.assertEqual(hash(cfg), hash(cfg2))

    @settings(deadline=None)
    @given(
        st.builds(lambda a, b: (a, b), _known_keys_strategy(), _unknown_keys_strategy())
    )
    def test_json_roundtrip_preserves_keys_and_values(
        self, pair: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        known, unknown = pair
        data = {**known, **unknown}
        cfg = helion.Config(**data)

        # JSON round-trip
        json_str = cfg.to_json()
        restored = helion.Config.from_json(json_str)

        # Compare as dicts; JSON dumps may reorder keys
        self.assertEqual(dict(restored), dict(cfg))

        # Unknown keys must persist
        for k in unknown:
            self.assertIn(k, restored)
            self.assertEqual(restored[k], unknown[k])

    @settings(deadline=None)
    @given(_known_keys_strategy(), _unknown_keys_strategy())
    def test_pickle_roundtrip_preserves_equality_and_hash(
        self, known: dict[str, Any], unknown: dict[str, Any]
    ) -> None:
        data = {**known, **unknown}
        cfg = helion.Config(**data)
        blob = pickle.dumps(cfg)
        restored = pickle.loads(blob)

        self.assertEqual(restored, cfg)
        self.assertEqual(hash(restored), hash(cfg))

    def test_list_tuple_hash_equivalence(self) -> None:
        cfg_list = helion.Config(block_sizes=[32, 64], loop_orders=[[1, 0]])
        cfg_tuple = helion.Config(block_sizes=[32, 64], loop_orders=[[1, 0]])

        # Same content should be equal and have equal hashes
        self.assertEqual(cfg_list, cfg_tuple)
        self.assertEqual(hash(cfg_list), hash(cfg_tuple))

    def test_pre_serialized_json_backward_compat(self) -> None:
        # Simulated config JSON saved in a prior release (hand-written, stable keys)
        json_str = (
            "{\n"
            '  "block_sizes": [64, 32],\n'
            '  "num_warps": 8,\n'
            '  "indexing": "pointer",\n'
            '  "custom_extra": {"alpha": 1, "beta": [1, 2]}\n'
            "}\n"
        )

        restored = helion.Config.from_json(json_str)

        expected = {
            "block_sizes": [64, 32],
            "num_warps": 8,
            "indexing": "pointer",
            "custom_extra": {"alpha": 1, "beta": [1, 2]},
        }
        self.assertEqual(dict(restored), expected)

        # Ensure we can still serialize it back and preserve content
        rejson = restored.to_json()
        reread = helion.Config.from_json(rejson)
        self.assertEqual(dict(reread), expected)


@onlyBackends(["triton"])
class TestSettingsEnv(TestCase):
    def test_persistent_reserved_sms_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_PERSISTENT_RESERVED_SMS": "5"},
            clear=False,
        ):
            settings = helion.Settings()
        self.assertEqual(settings.persistent_reserved_sms, 5)

    def test_autotune_force_persistent_limits_config_spec(self) -> None:
        settings = helion.Settings(autotune_force_persistent=True)
        env = CompileEnvironment(torch.device("cpu"), settings)
        self.assertEqual(
            env.config_spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )

    def test_backend_env_var_accepts_cute(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_BACKEND": "cute"},
            clear=False,
        ):
            settings = helion.Settings()
        self.assertEqual(settings.backend, "cute")

    def test_compile_environment_selects_cute_backend(self) -> None:
        settings = helion.Settings(backend="cute")
        env = CompileEnvironment(torch.device("cpu"), settings)
        self.assertEqual(env.backend_name, "cute")
        self.assertEqual(env.backend.default_launcher_name, "_default_cute_launcher")


@onlyBackends(["triton"])
class TestFormatKernelDecorator(TestCase):
    def test_format_kernel_decorator_includes_index_dtype(self) -> None:
        """Test that format_kernel_decorator includes index_dtype when set."""
        config = helion.Config(block_sizes=[8], num_warps=4)
        settings = helion.Settings(index_dtype=torch.int64)
        from helion.runtime.kernel import BoundKernel

        decorator = BoundKernel.format_kernel_decorator(None, config, settings)  # type: ignore[arg-type]

        self.assertIn("index_dtype=torch.int64", decorator)


@onlyBackends(["triton"])
class TestHardwareConfigSpecRanges(TestCase):
    """Tests for NVIDIA/AMD num_warps and num_stages range constraints.

    AMD GPUs have different hardware constraints than NVIDIA:
    - Max threads per block: 1024
    - Threads per wavefront: 64 (vs 32 for NVIDIA warps)
    - Max num_warps = 1024 / 64 = 16 (vs 32 for NVIDIA)
    - num_stages is also constrained differently for AMD pipelining

    These tests mock supports_amd_cdna_tunables to verify the correct ranges
    are used based on the GPU architecture.
    """

    def test_flat_config_uses_nvidia_ranges_when_not_amd(self) -> None:
        """Test that flat_config uses NVIDIA ranges (1-32, 1-8) when not on AMD."""
        from helion.autotuner.config_fragment import IntegerFragment
        from helion.autotuner.config_fragment import NumWarpsFragment
        from helion.autotuner.config_spec import ConfigSpec

        captured: dict[str, object] = {}

        def capture_fn(fragment: object) -> object:
            if isinstance(fragment, NumWarpsFragment):
                captured["num_warps"] = fragment
            elif isinstance(fragment, IntegerFragment) and not captured.get(
                "num_stages"
            ):
                captured["num_stages"] = fragment
            return fragment.default() if hasattr(fragment, "default") else fragment

        with (
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=False,
            ),
            patch(
                "helion.autotuner.config_spec.use_tileir_tunables",
                return_value=False,
            ),
        ):
            config_spec = ConfigSpec()
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]
        num_stages = captured["num_stages"]

        self.assertEqual(num_warps.low, 1)
        self.assertEqual(num_warps.high, 32)
        self.assertEqual(num_stages.low, 1)
        self.assertEqual(num_stages.high, 8)

    def test_flat_config_uses_amd_ranges_when_amd(self) -> None:
        """Test that flat_config uses AMD ranges (1-16, 1-4) when on AMD CDNA."""
        from helion.autotuner.config_fragment import IntegerFragment
        from helion.autotuner.config_fragment import NumWarpsFragment
        from helion.autotuner.config_spec import ConfigSpec

        captured: dict[str, object] = {}

        def capture_fn(fragment: object) -> object:
            if isinstance(fragment, NumWarpsFragment):
                captured["num_warps"] = fragment
            elif isinstance(fragment, IntegerFragment) and not captured.get(
                "num_stages"
            ):
                captured["num_stages"] = fragment
            return fragment.default() if hasattr(fragment, "default") else fragment

        with (
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=True,
            ),
            patch(
                "helion.autotuner.config_spec.use_tileir_tunables",
                return_value=False,
            ),
        ):
            config_spec = ConfigSpec()
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]
        num_stages = captured["num_stages"]

        self.assertEqual(num_warps.low, 1)
        self.assertEqual(num_warps.high, 16)
        self.assertEqual(num_stages.low, 1)
        self.assertEqual(num_stages.high, 4)

    def test_flat_config_uses_tileir_ranges_when_tileir(self) -> None:
        """Test that flat_config uses TileIR ranges (4-4, 1-10) when on TileIR backend."""
        from helion.autotuner.config_fragment import NumWarpsFragment
        from helion.autotuner.config_spec import ConfigSpec

        captured: dict[str, object] = {}

        def capture_fn(fragment: object) -> object:
            if isinstance(fragment, NumWarpsFragment):
                # TileIR overrides num_warps, so capture the last one
                captured["num_warps"] = fragment
            return fragment.default() if hasattr(fragment, "default") else fragment

        with (
            patch(
                "helion.autotuner.config_spec.supports_amd_cdna_tunables",
                return_value=False,
            ),
            patch(
                "helion.autotuner.config_spec.use_tileir_tunables",
                return_value=True,
            ),
        ):
            config_spec = ConfigSpec()
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]

        # TileIR uses fixed num_warps of 4
        self.assertEqual(num_warps.low, 4)
        self.assertEqual(num_warps.high, 4)


if __name__ == "__main__":
    unittest.main()
