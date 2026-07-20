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
from helion import exc
from helion._compiler.backend import PallasBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER,
)
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl


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
            "num_threads": st.one_of(
                st.integers(min_value=1, max_value=128),
                st.lists(st.integers(min_value=1, max_value=128), max_size=4),
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
            "pallas_load_buffer_count": st.lists(
                st.integers(min_value=1, max_value=2), max_size=4
            ),
            "load_eviction_policies": st.lists(
                st.sampled_from(["", "first", "last"]), max_size=4
            ),
            "load_cache_modifiers": st.lists(st.sampled_from(["", ".cg"]), max_size=4),
            "store_cache_modifiers": st.lists(
                st.sampled_from(["", ".cs", ".wt"]), max_size=4
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
                    "num_threads",
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
                    "pallas_load_buffer_count",
                    "load_eviction_policies",
                    "load_cache_modifiers",
                    "store_cache_modifiers",
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


class TestPallasLoadBufferCountConfig(TestCase):
    @staticmethod
    def _config_spec(
        num_tensors: int, *, has_pallas_inner_loops: bool = True
    ) -> ConfigSpec:
        spec = ConfigSpec(
            backend=PallasBackend(),
            target_device_capability=None,
            device=torch.device("cpu"),
            num_sm=1,
        )
        spec.pallas_load_buffer_count.length = num_tensors
        spec.has_pallas_inner_loops = has_pallas_inner_loops
        return spec

    def test_default_and_search_surface(self) -> None:
        spec = self._config_spec(2)

        field = spec._flat_fields()["pallas_load_buffer_count"]

        assert isinstance(field, ListOf)
        assert isinstance(field.inner, IntegerFragment)
        self.assertEqual(
            (field.inner.low, field.inner.high, field.inner.default_val), (1, 2, 1)
        )
        self.assertEqual(field.default(), [1, 1])
        self.assertEqual(
            field.pattern_neighbors([1, 1]),
            [[2, 1], [1, 2]],
        )
        with patch("helion.autotuner.config_fragment.random.randint", return_value=2):
            self.assertEqual(field.random(), [2, 2])
        self.assertIn(
            ("pallas_load_buffer_count", *field.fingerprint()),
            spec.structural_fingerprint(),
        )
        self.assertNotIn("pallas_load_buffer_count", spec.default_config())

        fori_config = helion.Config(pallas_loop_type="fori_loop")
        spec.normalize(fori_config)
        self.assertEqual(fori_config.pallas_load_buffer_count, [1, 1])

        config = helion.Config(
            pallas_loop_type="fori_loop",
            pallas_load_buffer_count=[2, 1],
        )

        spec.normalize(config)

        self.assertEqual(config.pallas_load_buffer_count, [2, 1])

    def test_inactive_field_is_ignored(self) -> None:
        spec = self._config_spec(2)
        for values in ([2, 1], [2]):
            config = helion.Config.from_dict(
                {
                    "pallas_loop_type": "emit_pipeline",
                    "pallas_load_buffer_count": values,
                }
            )
            spec.normalize(config)
            self.assertNotIn("pallas_load_buffer_count", config)

    def test_inactive_search_surface_omits_field(self) -> None:
        cases = (
            (self._config_spec(2, has_pallas_inner_loops=False), [2]),
            (self._config_spec(0), []),
        )
        for spec, values in cases:
            with self.subTest(num_tensors=spec.pallas_load_buffer_count.length):
                self.assertNotIn("pallas_load_buffer_count", spec._flat_fields())
                config = helion.Config(
                    pallas_loop_type="fori_loop",
                    pallas_load_buffer_count=values,
                )
                spec.normalize(config)
                self.assertNotIn("pallas_load_buffer_count", config)

    def test_non_pallas_backend_rejects_the_field(self) -> None:
        spec = ConfigSpec(
            backend=TritonBackend(),
            target_device_capability=None,
            device=torch.device("cpu"),
            num_sm=1,
        )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "Unsupported config keys for backend 'triton'",
        ):
            spec.normalize(helion.Config(pallas_load_buffer_count=[]))

    def test_rejects_invalid_explicit_lists(self) -> None:
        spec = self._config_spec(2)
        invalid_values = (
            (2, 1),
            [1],
            [1, 1, 1],
            [1, True],
            [0, 1],
            [3, 1],
        )

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(exc.InvalidConfig):
                spec.normalize(
                    helion.Config.from_dict(
                        {
                            "pallas_loop_type": "fori_loop",
                            "pallas_load_buffer_count": value,
                        }
                    )
                )

        zero_tensor_spec = self._config_spec(0)
        with self.assertRaises(exc.InvalidConfig):
            zero_tensor_spec.normalize(
                helion.Config(
                    pallas_loop_type="fori_loop", pallas_load_buffer_count=[2]
                )
            )


@onlyBackends(["triton", "cute"])
class TestConfigAPI(TestCase):
    def test_config_import_path_stability(self) -> None:
        runtime = importlib.import_module("helion.runtime")

        self.assertIs(helion.Config, runtime.Config)
        self.assertIs(helion.Config, helion.runtime.Config)

    def test_cuda_device_capability_specializes_bound_kernel_cache_key(self) -> None:
        @helion.kernel()
        def device_key_kernel(device: hl.constexpr) -> None:
            pass

        device = torch.device("cuda:0")
        # Patch the helion seam (target_device_capability, imported into
        # runtime.kernel) rather than torch.cuda.get_device_capability: the
        # latter is memoized behind _target_device_capability, mirroring the
        # is_hip / _is_hip pattern where tests mock the public wrapper, not
        # the cached inner query.
        with patch(
            "helion.runtime.kernel.target_device_capability", return_value=(9, 0)
        ):
            sm90_key = device_key_kernel._base_specialization_key((device,))
        with patch(
            "helion.runtime.kernel.target_device_capability", return_value=(10, 0)
        ):
            sm100_key = device_key_kernel._base_specialization_key((device,))

        self.assertEqual(sm90_key[-2:], ("cuda", (9, 0)))
        self.assertEqual(sm100_key[-2:], ("cuda", (10, 0)))
        self.assertNotEqual(sm90_key, sm100_key)

    def test_config_constructor_signature_contains_expected_kwargs(self) -> None:
        # Keep this list in sync with public kwargs; removal/rename should fail tests
        expected = {
            "block_sizes",
            "num_threads",
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
            "pallas_load_buffer_count",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
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

    def test_epilogue_subtile_rewrites_only_store_slots(self) -> None:
        env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
        spec = env.config_spec
        spec.epilogue_subtile_candidate_enabled = True
        spec.store_indices = [1, 3]
        config = {
            "epilogue_subtile": 2,
            "indexing": ["pointer", "block_ptr", "pointer", "block_ptr"],
        }

        spec.fix_epilogue_subtile_store_indexing(config)

        self.assertEqual(
            config["indexing"],
            ["pointer", "tensor_descriptor", "pointer", "tensor_descriptor"],
        )


@onlyBackends(["triton", "cute"])
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

    @skipIfXPU("Uses torch.device('cuda') directly")
    def test_distributed_limits_pid_types_to_persistent(self) -> None:
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
        self.assertEqual(
            env.config_spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )

    def test_persistent_block_limit_caps_num_sm_multiplier(self) -> None:
        # max_blocks=10000, 200 SMs -> 10000 // 200 = 50 -> floor pow2 = 32
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 32)

    def test_persistent_block_limit_handles_zero_raw_max(self) -> None:
        # max_blocks=144, 148 SMs -> 144 // 148 = 0 -> must clamp to 1
        # without crashing on `1 << -1`.
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=144),
            patch("helion.runtime.get_num_sm", return_value=148),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 1)

    def test_backend_env_var_accepts_cute(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_BACKEND": "cute"},
            clear=False,
        ):
            settings = helion.Settings()
        self.assertEqual(settings.backend, "cute")

    def test_backend_tileir_requires_enable_tile(self) -> None:
        env = {"HELION_BACKEND": "tileir", "ENABLE_TILE": "0"}
        with (
            patch.dict(os.environ, env, clear=False),
            self.assertRaises(exc.MissingEnableTile),
        ):
            helion.Settings()

    def test_backend_tileir_kwarg_requires_enable_tile(self) -> None:
        with (
            patch.dict(os.environ, {"ENABLE_TILE": "0"}, clear=False),
            self.assertRaises(exc.MissingEnableTile),
        ):
            helion.Settings(backend="tileir")

    def test_backend_tileir_with_enable_tile(self) -> None:
        env = {"HELION_BACKEND": "tileir", "ENABLE_TILE": "1"}
        with patch.dict(os.environ, env, clear=False):
            settings = helion.Settings()
        self.assertEqual(settings.backend, "tileir")

    @skipUnlessCuteAvailable("Constructs a cute CompileEnvironment")
    def test_compile_environment_selects_cute_backend(self) -> None:
        settings = helion.Settings(backend="cute")
        env = CompileEnvironment(torch.device("cpu"), settings)
        self.assertEqual(env.backend_name, "cute")
        self.assertEqual(env.backend.default_launcher_name, "_default_cute_launcher")

    @skipUnlessCuteAvailable("Constructs a cute CompileEnvironment")
    def test_num_threads_support_is_backend_specific(self) -> None:
        triton_env = CompileEnvironment(
            torch.device("cpu"), helion.Settings(backend="triton")
        )
        self.assertFalse(triton_env.config_spec.supports_config_key("num_threads"))
        self.assertNotIn("num_threads", triton_env.config_spec.supported_config_keys())

        cute_env = CompileEnvironment(
            torch.device("cpu"), helion.Settings(backend="cute")
        )
        self.assertTrue(cute_env.config_spec.supports_config_key("num_threads"))

    def test_pallas_backend_uses_exact_factory_and_static_reduction_dims(
        self,
    ) -> None:
        from helion._compiler.backend import PallasBackend
        from helion._compiler.backend import TritonBackend

        triton = TritonBackend()
        pallas = PallasBackend()

        self.assertTrue(triton.pad_factory_tensors_to_power_of_2)
        self.assertEqual(triton.static_rdim_size(384), 512)
        self.assertFalse(pallas.pad_factory_tensors_to_power_of_2)
        self.assertEqual(pallas.static_rdim_size(384), 384)

    def test_triton_rejects_num_threads_in_normalize(self) -> None:
        env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            rf"Unsupported config keys for backend '{env.backend_name}'",
        ):
            env.config_spec.normalize({"num_threads": [2]})

    def test_block_size_spec_max_size_bounded_by_world_size(self) -> None:
        """Regression test: BlockSizeSpec.max_size must be bounded by size_hint//world_size
        in a distributed setting, not the raw size_hint."""
        from helion.autotuner.config_spec import BlockSizeSpec

        size_hint = 1024
        world_size = 4

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
        ):
            spec = BlockSizeSpec(block_id=0, size_hint=size_hint)

        # max_size should be bounded by size_hint // world_size = 256, not 1024
        self.assertLessEqual(spec.max_size, size_hint // world_size)

    def test_bounded_inner_block_size_clamped_to_outer_value(self) -> None:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        # Use Triton only as a concrete backend; this normalize behavior is
        # backend-agnostic.
        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=1024, bounded_by_block_id=0)
        )

        config = {"block_sizes": [64, 256]}
        spec.normalize(config)

        self.assertEqual(config["block_sizes"][:2], [64, 64])

    def test_bounded_inner_block_size_keeps_valid_inner_value(self) -> None:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=1024, bounded_by_block_id=0)
        )

        config = {"block_sizes": [256, 64]}
        spec.normalize(config)

        self.assertEqual(config["block_sizes"][:2], [256, 64])

    def test_bounded_inner_block_size_clamps_multi_level_nesting(self) -> None:
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=TritonBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=1024, bounded_by_block_id=0)
        )
        spec.block_sizes.append(
            BlockSizeSpec(block_id=2, size_hint=1024, bounded_by_block_id=1)
        )

        config = {"block_sizes": [64, 256, 512]}
        spec.normalize(config)

        self.assertEqual(config["block_sizes"][:3], [64, 64, 64])

    def test_bounded_inner_block_size_repairs_cute_num_threads(self) -> None:
        from helion._compiler.backend import CuteBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import NumThreadsSpec

        spec = ConfigSpec(backend=CuteBackend())
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=1024, bounded_by_block_id=0)
        )
        spec.num_threads.append(NumThreadsSpec(block_id=0, size_hint=1024))
        spec.num_threads.append(NumThreadsSpec(block_id=1, size_hint=1024))

        config = {"block_sizes": [64, 256], "num_threads": [64, 256]}
        spec.normalize(config)

        self.assertEqual(config["block_sizes"][:2], [64, 64])
        self.assertEqual(config["num_threads"][:2], [64, 64])

    def test_detect_outer_block_bound_requires_end_minus_begin(self) -> None:
        from types import SimpleNamespace

        import sympy

        from helion._compiler.host_function import SymbolOrigin
        from helion._compiler.type_info import _detect_outer_block_bound
        from helion._compiler.variable_origin import TileBeginOrigin
        from helion._compiler.variable_origin import TileEndOrigin

        begin = sympy.Symbol("begin")
        end = sympy.Symbol("end")
        fake_host = SimpleNamespace(
            expr_to_origin={
                begin: SymbolOrigin(TileBeginOrigin(3)),
                end: SymbolOrigin(TileEndOrigin(3)),
            }
        )
        fake_env = SimpleNamespace(get_block_id=lambda _numel: None)

        with (
            patch(
                "helion._compiler.type_info.HostFunction.current",
                return_value=fake_host,
            ),
            patch(
                "helion._compiler.type_info._symint_expr",
                side_effect=lambda expr: expr,
            ),
        ):
            self.assertEqual(_detect_outer_block_bound(end - begin, fake_env), 3)
            self.assertIsNone(_detect_outer_block_bound(begin + end, fake_env))

    def test_detect_outer_block_bound_accepts_direct_block_size(self) -> None:
        from types import SimpleNamespace

        from helion._compiler.type_info import _detect_outer_block_bound

        numel = object()
        fake_env = SimpleNamespace(
            get_block_id=lambda value: 5 if value is numel else None
        )

        with patch(
            "helion._compiler.type_info._symint_expr",
            side_effect=AssertionError("_symint_expr should not be called"),
        ):
            self.assertEqual(_detect_outer_block_bound(numel, fake_env), 5)

    def test_bounded_block_size_repr_includes_bound(self) -> None:
        from helion.autotuner.config_spec import BlockSizeSpec

        self.assertIn(
            "bounded_by_block_id=7",
            repr(BlockSizeSpec(block_id=1, size_hint=64, bounded_by_block_id=7)),
        )

    def test_autotune_search_acf_env_var_strips_whitespace(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_SEARCH_ACF": "/a/first.bin, /b/second.bin ,/c/third.bin"},
            clear=False,
        ):
            settings = helion.Settings()
        self.assertEqual(
            settings.autotune_search_acf,
            ["/a/first.bin", "/b/second.bin", "/c/third.bin"],
        )


@onlyBackends(["triton", "cute"])
class TestFormatKernelDecorator(TestCase):
    def test_format_kernel_decorator_includes_index_dtype(self) -> None:
        """Test that format_kernel_decorator includes index_dtype when set."""
        config = helion.Config(block_sizes=[8], num_warps=4)
        settings = helion.Settings(index_dtype=torch.int64)
        from helion.runtime.kernel import BoundKernel

        decorator = BoundKernel.format_kernel_decorator(None, config, settings)  # type: ignore[arg-type]

        self.assertIn("index_dtype=torch.int64", decorator)


@onlyBackends(["triton", "cute"])
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
        from helion._compiler.backend import TritonBackend
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
        ):
            config_spec = ConfigSpec(backend=TritonBackend())
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]
        num_stages = captured["num_stages"]

        self.assertEqual(num_warps.low, 1)
        self.assertEqual(num_warps.high, 32)
        self.assertEqual(num_stages.low, 1)
        self.assertEqual(num_stages.high, 8)

    def test_flat_config_uses_amd_ranges_when_amd(self) -> None:
        """Test that flat_config uses AMD ranges (1-16, 1-4) when on AMD CDNA."""
        from helion._compiler.backend import TritonBackend
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
        ):
            config_spec = ConfigSpec(backend=TritonBackend())
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]
        num_stages = captured["num_stages"]

        self.assertEqual(num_warps.low, 1)
        self.assertEqual(num_warps.high, 16)
        self.assertEqual(num_stages.low, 1)
        self.assertEqual(num_stages.high, 4)

    def test_flat_config_uses_tileir_ranges_when_tileir(self) -> None:
        """Test that flat_config uses TileIR ranges (4-4, 1-10) when on TileIR backend."""
        from helion._compiler.backend import TileIRBackend
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
        ):
            config_spec = ConfigSpec(backend=TileIRBackend())
            config_spec.flat_config(capture_fn)

        num_warps = captured["num_warps"]

        # TileIR uses fixed num_warps of 4
        self.assertEqual(num_warps.low, 4)
        self.assertEqual(num_warps.high, 4)

    def test_eviction_policy_choices_do_not_leak_mocked_amd_state(self) -> None:
        """Mocked AMD capability detection should not poison later Triton specs."""
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            amd_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(amd_spec.load_eviction_policies.inner.choices, ("",))

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=False,
        ):
            nvidia_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(
            nvidia_spec.load_eviction_policies.inner.choices,
            ("", "first", "last"),
        )

    def test_load_cache_modifier_choices_do_not_leak_mocked_amd_state(self) -> None:
        """Mocked AMD capability detection should not poison later Triton specs."""
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            amd_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(amd_spec.load_cache_modifiers.inner.choices, ("", ".cg"))

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=False,
        ):
            nvidia_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(nvidia_spec.load_cache_modifiers.inner.choices, ("",))

    def test_store_cache_modifier_choices_do_not_leak_mocked_amd_state(self) -> None:
        """Mocked AMD capability detection should not poison later Triton specs."""
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_spec import ConfigSpec

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=True,
        ):
            amd_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(
            amd_spec.store_cache_modifiers.inner.choices,
            ("", ".cs", ".wt"),
        )

        with patch(
            "helion.autotuner.config_spec.supports_amd_cdna_tunables",
            return_value=False,
        ):
            nvidia_spec = ConfigSpec(backend=TritonBackend())
        self.assertEqual(nvidia_spec.store_cache_modifiers.inner.choices, ("",))


class TestCuteTcgen05ConfigSpecSplit(TestCase):
    @staticmethod
    def _make_cute_tcgen05_spec():
        from helion._compiler.backend import CuteBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        spec = ConfigSpec(backend=CuteBackend())
        spec.cute_tcgen05_search_enabled = True
        for block_id, size_hint in enumerate((4096, 4096, 4096)):
            spec.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=size_hint,
                    max_size=256 if block_id < 2 else 128,
                )
            )
        return spec

    def test_cute_tcgen05_search_fields_and_default_flat_roundtrip(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._make_cute_tcgen05_spec()
        flat_keys = [key for key, _count, _is_sequence in spec.flat_key_layout()]

        self.assertIn("tcgen05_cluster_m", flat_keys)
        self.assertIn("tcgen05_ab_stages", flat_keys)
        self.assertIn("tcgen05_strategy", flat_keys)
        self.assertIn("tcgen05_warp_spec_c_input_warps", flat_keys)

        gen = ConfigGeneration(spec)
        default_flat = gen.default_flat()
        self.assertEqual(
            gen.flatten(gen.unflatten([*default_flat])),
            default_flat,
        )

    def test_tcgen05_search_fields_do_not_leak_to_other_backends(self) -> None:
        from helion._compiler.backend import MetalBackend
        from helion._compiler.backend import PallasBackend
        from helion._compiler.backend import TileIRBackend
        from helion._compiler.backend import TritonBackend
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        for backend in (
            TritonBackend(),
            PallasBackend(),
            TileIRBackend(),
            MetalBackend(),
        ):
            spec = ConfigSpec(backend=backend)
            spec.cute_tcgen05_search_enabled = True
            spec.block_sizes.append(
                BlockSizeSpec(block_id=0, size_hint=128, max_size=128)
            )
            flat_keys = {key for key, _count, _is_sequence in spec.flat_key_layout()}
            self.assertFalse(
                any(key.startswith("tcgen05_") for key in flat_keys),
                f"{backend.name} search surface leaked tcgen05 keys: {flat_keys}",
            )
            default_config = spec.default_config()
            self.assertFalse(
                any(key.startswith("tcgen05_") for key in default_config.config),
                f"{backend.name} default config leaked tcgen05 keys: "
                f"{default_config.config}",
            )
            gen = ConfigGeneration(spec)
            generated_config = gen.unflatten(gen.default_flat())
            self.assertFalse(
                any(key.startswith("tcgen05_") for key in generated_config.config),
                f"{backend.name} generated config leaked tcgen05 keys: "
                f"{generated_config.config}",
            )

    def test_explicit_tcgen05_strategy_config_validation(self) -> None:
        spec = self._make_cute_tcgen05_spec()

        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05 strategy invariants violated",
        ):
            spec.normalize(
                helion.Config(
                    block_sizes=[128, 128, 64],
                    tcgen05_strategy="role_local_monolithic",
                    tcgen05_warp_spec_c_input_warps=1,
                )
            )

        with self.assertRaises(exc.InvalidConfig):
            spec.normalize(
                helion.Config(
                    block_sizes=[128, 128, 64],
                    **{
                        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY: (
                            TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER
                        )
                    },
                )
            )

        config = helion.Config(
            block_sizes=[128, 128, 64],
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_warp_spec_c_input_warps=1,
        )
        spec.normalize(config)
        self.assertEqual(config.config["tcgen05_strategy"], "role_local_with_scheduler")
        self.assertEqual(config.config["tcgen05_warp_spec_c_input_warps"], 1)

    def test_direct_cute_config_spec_enforces_clc_arch_gate(self) -> None:
        from helion._compiler.cute.strategies import (
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
        )
        from helion._compiler.cute.strategies import Tcgen05PersistenceModel

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            # Patch the helion seam: torch.cuda.get_device_capability is
            # memoized behind _target_device_capability (is_hip / _is_hip
            # pattern), so config_spec's get_target_device_capability() must
            # be patched at the wrapper, not the cached torch query.
            patch(
                "helion.autotuner.config_spec.get_target_device_capability",
                return_value=(9, 0),
            ),
        ):
            spec = self._make_cute_tcgen05_spec()

        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "requires CUDA compute capability major >= 10",
        ):
            spec.normalize(
                helion.Config(
                    block_sizes=[128, 128, 64],
                    pid_type="persistent_interleaved",
                    tcgen05_strategy="role_local_with_scheduler",
                    tcgen05_warp_spec_scheduler_warps=1,
                    tcgen05_warp_spec_c_input_warps=1,
                    **{
                        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                            Tcgen05PersistenceModel.CLC_PERSISTENT.value
                        )
                    },
                )
            )

    def test_aux_kernel_detection_routes_strategy_search_surface(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        tcgen05_config = spec._cute_tcgen05_config

        narrow = tcgen05_config.strategy_autotune_fragments()
        self.assertEqual(narrow["tcgen05_strategy"].choices, ("role_local_monolithic",))
        self.assertEqual(narrow["tcgen05_warp_spec_c_input_warps"].choices, (0,))

        tcgen05_config.aux_kernel_detected = True
        widened = tcgen05_config.strategy_autotune_fragments()
        self.assertEqual(
            widened["tcgen05_strategy"].choices,
            ("role_local_monolithic", "role_local_with_scheduler"),
        )
        self.assertEqual(widened["tcgen05_warp_spec_c_input_warps"].choices, (0, 1))


if __name__ == "__main__":
    unittest.main()
