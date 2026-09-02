from __future__ import annotations

import importlib
import inspect
import os
import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
import unittest
from unittest.mock import patch

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import torch

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import (
    _tcgen05_grouped_worklist_seed_family,
)
from helion._compiler.backend import PallasBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.grouped_worklist_policy import GroupedWorklistTargetPolicy
from helion._compiler.cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from helion._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_DIRECT
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_DYNAMIC
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_STATIC
from helion._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_MODE_WORKLIST_NM
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from helion._compiler.cute.tcgen05_constants import (
    resolve_tcgen05_grouped_worklist_mma_profile,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.config_spec import MatmulFact
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Mapping


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
            "pallas_indirect_access_mode": st.sampled_from(["dma", "one_hot"]),
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
            "cross_loop_schedule": st.sampled_from(["barrier", "static_pipeline"]),
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
                    "pallas_indirect_access_mode",
                    "load_eviction_policies",
                    "load_cache_modifiers",
                    "store_cache_modifiers",
                    "num_warps",
                    "num_stages",
                    "pid_type",
                    "cross_loop_schedule",
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
        spec = ConfigSpec(backend=PallasBackend())
        spec.pallas_load_buffer_count.length = num_tensors
        spec.has_pallas_inner_loops = has_pallas_inner_loops
        return spec

    def test_default_and_search_surface(self) -> None:
        spec = self._config_spec(2)
        field = spec._flat_fields()["pallas_load_buffer_count"]
        self.assertEqual(field.default(), [1, 1])
        self.assertEqual(field.pattern_neighbors([1, 1]), [[2, 1], [1, 2]])
        self.assertIn(
            ("pallas_load_buffer_count", *field.fingerprint()),
            spec.structural_fingerprint(),
        )
        self.assertNotIn("pallas_load_buffer_count", spec.default_config())

        fori_config = helion.Config(pallas_loop_type="fori_loop")
        spec.normalize(fori_config)
        self.assertEqual(fori_config.pallas_load_buffer_count, [1, 1])

        config = helion.Config(
            pallas_loop_type="fori_loop", pallas_load_buffer_count=[2, 1]
        )
        spec.normalize(config)
        self.assertEqual(config.pallas_load_buffer_count, [2, 1])

    def test_inactive_field_is_ignored(self) -> None:
        cases = (
            (self._config_spec(2), "emit_pipeline", [2], True),
            (
                self._config_spec(2, has_pallas_inner_loops=False),
                "fori_loop",
                [2],
                False,
            ),
            (self._config_spec(0), "fori_loop", [], False),
        )
        for spec, loop_type, values, present_in_search in cases:
            with self.subTest(num_tensors=spec.pallas_load_buffer_count.length):
                self.assertEqual(
                    "pallas_load_buffer_count" in spec._flat_fields(),
                    present_in_search,
                )
                config = helion.Config.from_dict(
                    {
                        "pallas_loop_type": loop_type,
                        "pallas_load_buffer_count": values,
                    }
                )
                spec.normalize(config)
                self.assertNotIn("pallas_load_buffer_count", config)

    def test_non_pallas_backend_rejects_the_field(self) -> None:
        spec = ConfigSpec(backend=TritonBackend())
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
        with (
            patch(
                "helion.runtime.kernel.target_device_capability", return_value=(9, 0)
            ),
            patch(
                "helion.runtime.kernel.compiler_promotion_specialization_key",
                return_value=(),
            ),
        ):
            sm90_key = device_key_kernel._base_specialization_key((device,))
        with (
            patch(
                "helion.runtime.kernel.target_device_capability",
                return_value=(10, 0),
            ),
            patch(
                "helion.runtime.kernel.compiler_promotion_specialization_key",
                return_value=(),
            ),
        ):
            sm100_key = device_key_kernel._base_specialization_key((device,))

        self.assertEqual(sm90_key[-3:], ("cuda", (9, 0), False))
        self.assertEqual(sm100_key[-3:], ("cuda", (10, 0), False))
        self.assertNotEqual(sm90_key, sm100_key)

        promotion = (("synthetic", "NVIDIA B200"),)
        with (
            patch(
                "helion.runtime.kernel.target_device_capability",
                return_value=(10, 0),
            ),
            patch(
                "helion.runtime.kernel.compiler_promotion_specialization_key",
                return_value=promotion,
            ),
        ):
            promoted_key = device_key_kernel._base_specialization_key((device,))
        self.assertEqual(
            promoted_key[-4:],
            ("cuda", (10, 0), promotion, False),
        )

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
            "cross_loop_schedule",
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

    def test_cross_loop_schedule_is_an_admitted_triton_field(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        self.assertEqual(helion.Config().cross_loop_schedule, "barrier")
        self.assertEqual(
            helion.Config(cross_loop_schedule="static_pipeline").cross_loop_schedule,
            "static_pipeline",
        )

        with patch("helion._compat.is_hip", return_value=False):
            spec = ConfigSpec(backend=TritonBackend())
            self.assertTrue(spec.supports_config_key("cross_loop_schedule"))
            self.assertNotIn("cross_loop_schedule", spec._flat_fields())
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "only for kernels with compiler-inferred cross-loop dependencies",
            ):
                spec.normalize(helion.Config(cross_loop_schedule="barrier"))

            spec.enable_cross_loop_schedule()
            field = spec._flat_fields()["cross_loop_schedule"]
            self.assertIsInstance(field, EnumFragment)
            assert isinstance(field, EnumFragment)
            self.assertIs(field, spec.cross_loop_schedule)
            self.assertEqual(field.choices, ("barrier", "static_pipeline"))
            self.assertEqual(
                spec.default_config()["cross_loop_schedule"],
                "barrier",
            )

            static_config = spec.default_config()
            static_config.config["cross_loop_schedule"] = "static_pipeline"
            spec.normalize(static_config)
            generation = ConfigGeneration(spec)
            round_trip = generation.unflatten(generation.flatten(static_config))
            self.assertEqual(
                round_trip["cross_loop_schedule"],
                "static_pipeline",
            )

            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "must be one of",
            ):
                spec.normalize(
                    helion.Config.from_dict({"cross_loop_schedule": "unknown"})
                )

    def test_cross_loop_schedule_is_not_supported_on_amd(self) -> None:
        with patch("helion._compat.is_hip", return_value=True):
            spec = ConfigSpec(backend=TritonBackend())
            self.assertFalse(spec.supports_config_key("cross_loop_schedule"))
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "is not supported by backend",
            ):
                spec.enable_cross_loop_schedule()

    def test_cross_loop_schedule_is_not_supported_on_xpu(self) -> None:
        with patch("helion._compat.is_hip", return_value=False):
            spec = ConfigSpec(
                backend=TritonBackend(),
                device=torch.device("xpu"),
                num_sm=1,
            )
            self.assertFalse(spec.supports_config_key("cross_loop_schedule"))

    def test_warp_specialization_uses_effective_launcher_warp_count(self) -> None:
        backend = TritonBackend()
        config = helion.Config(
            num_warps=1,
            range_warp_specializes=[None, True],
        )

        self.assertEqual(backend.effective_num_warps(config), 4)

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
    def test_autotune_force_persistent_no_symm_mem_keeps_multiplier(self) -> None:
        # force_persistent + distributed but no symm-mem signal must NOT clamp the
        # signal-pad budget; the clamp is symm-mem-specific.
        settings = helion.Settings(autotune_force_persistent=True)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.restrict_pid_types_for_persistent(())
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 128)
        self.assertEqual(
            env.config_spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )

    @skipIfXPU("Uses torch.device('cuda') directly")
    def test_autotune_force_persistent_clamps_on_symm_mem(self) -> None:
        # force_persistent + a symm-mem arg must clamp to the signal-pad budget.
        settings = helion.Settings(autotune_force_persistent=True)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
            patch("helion._dist_utils.is_symm_mem_tensor", return_value=True),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            # is_symm_mem_tensor is mocked, so a plain CPU tensor is enough and
            # avoids requiring a CUDA build on the runner.
            env.restrict_pid_types_for_persistent((torch.empty(1),))
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 32)

    @skipIfXPU("Uses torch.device('cuda') directly")
    def test_distributed_alone_keeps_all_pid_types(self) -> None:
        # A distributed process alone must NOT restrict pid_types; the kernel
        # must actually require a persistent kernel (barrier / symm-mem arg).
        settings = helion.Settings()
        with (
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.restrict_pid_types_for_persistent(())
        self.assertEqual(
            env.config_spec.allowed_pid_types,
            ("flat", "xyz", "persistent_blocked", "persistent_interleaved"),
        )

    @skipIfXPU("Uses torch.device('cuda') directly")
    def test_distributed_barrier_limits_pid_types_to_persistent(self) -> None:
        # A barrier restricts pid_types but is NOT a symm-mem signal, so it must
        # leave max_num_sm_multiplier untouched.
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.has_barrier = True
            env.restrict_pid_types_for_persistent(())
        self.assertEqual(
            env.config_spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 128)

    @skipIfXPU("Uses torch.device('cuda') directly")
    def test_distributed_symm_mem_arg_limits_pid_types_to_persistent(self) -> None:
        # A symm-mem tensor arg (no barrier) is the other persistent signal and
        # must restrict pid_types just like a barrier does.
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
            patch("helion._dist_utils.is_symm_mem_tensor", return_value=True),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.restrict_pid_types_for_persistent((torch.empty(1),))
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
            patch("helion._dist_utils.is_symm_mem_tensor", return_value=True),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.restrict_pid_types_for_persistent((torch.empty(1),))
        self.assertEqual(env.config_spec.max_num_sm_multiplier, 32)

    def test_persistent_block_limit_handles_zero_raw_max(self) -> None:
        # max_blocks=144, 148 SMs -> 144 // 148 = 0 -> must clamp to 1
        # without crashing on `1 << -1`.
        settings = helion.Settings()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=144),
            patch("helion.runtime.get_num_sm", return_value=148),
            patch("helion._dist_utils.is_symm_mem_tensor", return_value=True),
        ):
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            env.restrict_pid_types_for_persistent((torch.empty(1),))
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


class TestConfigSpecNormalizedCopy(TestCase):
    def test_nested_containers_are_copied_without_copying_opaque_leaves(self) -> None:
        class NonCopyable:
            def __deepcopy__(self, memo: object) -> None:
                raise AssertionError("opaque config leaves must not be deep-copied")

        spec = ConfigSpec(
            backend=TritonBackend(),
            user_defined_tunables={"metadata": EnumFragment((None,))},
        )
        spec.loop_orders.append(LoopOrderSpec([0, 1]))
        opaque = NonCopyable()
        tensor = torch.empty(0)
        requested = helion.Config(
            loop_orders=[[0, 1]],
            metadata={"nested": ([opaque, tensor],)},
        )
        requested_values = dict(requested.config)

        normalized = spec.normalized_config(requested)

        self.assertIsNot(normalized, requested)
        self.assertEqual(requested.config, requested_values)
        self.assertEqual(normalized.config["num_warps"], 4)
        self.assertEqual(normalized.config["num_stages"], 1)
        self.assertEqual(normalized.config["pid_type"], "flat")
        normalized.loop_orders[0][0] = 1
        normalized_metadata = cast("dict[str, object]", normalized["metadata"])
        normalized_tuple = cast("tuple[object, ...]", normalized_metadata["nested"])
        normalized_nested = cast("list[object]", normalized_tuple[0])
        normalized_nested.append("normalized-only")
        requested_metadata = cast("dict[str, object]", requested["metadata"])
        requested_tuple = cast("tuple[object, ...]", requested_metadata["nested"])
        requested_nested = cast("list[object]", requested_tuple[0])
        self.assertEqual(requested.loop_orders, [[0, 1]])
        self.assertEqual(len(requested_nested), 2)
        self.assertIs(normalized_nested[0], opaque)
        self.assertIs(normalized_nested[1], tensor)


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

    @staticmethod
    def _make_permuted_cute_tcgen05_spec():
        from helion._compiler.backend import CuteBackend
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        # Config order is [N, K, M], while semantic MMA axes are M=2, N=0, K=1.
        spec = ConfigSpec(backend=CuteBackend())
        spec.cute_tcgen05_search_enabled = True
        for block_id, max_size in enumerate((256, 128, 256)):
            spec.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=4096,
                    max_size=max_size,
                )
            )
        spec.register_cute_tcgen05_mma_analysis(
            m_block_id=2,
            n_block_id=0,
            k_block_id=1,
            compile_time_static_extents=(4096, 4096, 4096),
            input_dtype=torch.bfloat16,
            has_leading_passthrough=False,
            explicit_epi_tile_compatible=True,
        )
        return spec

    @staticmethod
    def _register_default_cute_tcgen05_mma_analysis(spec):
        spec.register_cute_tcgen05_mma_analysis(
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            compile_time_static_extents=(4096, 4096, 4096),
            input_dtype=torch.bfloat16,
            has_leading_passthrough=False,
            explicit_epi_tile_compatible=True,
        )
        return spec

    @staticmethod
    def _add_cute_tcgen05_matmul_fact(
        spec,
        *,
        static_k: int,
        k_block_id: int = 2,
    ) -> None:
        spec.matmul_facts.append(
            MatmulFact(
                lhs_ndim=2,
                rhs_ndim=3,
                m_block_id=0,
                n_block_id=1,
                k_block_id=k_block_id,
                static_m=256,
                static_n=None,
                static_k=static_k,
                lhs_dtype=torch.bfloat16,
                rhs_dtype=torch.bfloat16,
            )
        )

    @staticmethod
    def _grouped_worklist_config(
        *,
        block_sizes: tuple[int, int, int] | list[int] = (256, 128, 64),
        source_m_tile: int = TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        cluster_m: int = 2,
        **overrides: Any,
    ) -> helion.Config:
        values: dict[str, Any] = {
            "block_sizes": list(block_sizes),
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": cluster_m,
            "tcgen05_cluster_n": 1,
            "tcgen05_grouped_mode": TCGEN05_GROUPED_MODE_WORKLIST_NM,
            "tcgen05_grouped_worklist_source_m_tile": source_m_tile,
        }
        values.update(overrides)
        return helion.Config(**values)

    def _grouped_worklist_generation(
        self,
        spec: Any,
        *,
        source_m_tile: int,
        cluster_m: int = 2,
    ) -> tuple[Any, list[Any], int, int, int]:
        from helion.autotuner.config_generation import ConfigGeneration

        spec.compiler_seed_configs = [
            self._grouped_worklist_config(
                source_m_tile=source_m_tile,
                cluster_m=cluster_m,
            )
        ]
        generation = ConfigGeneration(spec)
        [(flat, _normalized)] = generation.seed_flat_config_pairs()
        _m_flat, n_flat, k_flat = generation.block_size_indices[:3]
        [cluster_flat], _ = generation._key_to_flat_indices["tcgen05_cluster_m"]
        return generation, flat, n_flat, k_flat, cluster_flat

    def _constrained_grouped_worklist_spec(self, smem_budget: int) -> Any:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        spec._cute_tcgen05_config.ab_stages_three_search_constraints = (
            Tcgen05AbStagesThreeSearchConstraints(
                dtype_bytes=2,
                per_cta_smem_budget_bytes=smem_budget,
            )
        )
        return spec

    @staticmethod
    def _enum_fragment(fragment: object) -> EnumFragment:
        assert isinstance(fragment, EnumFragment)
        return fragment

    def test_normalized_config_drops_none_core_values(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        requested = helion.Config.from_dict(
            {"block_sizes": [256, 128, 128], "xcd_remap": None}
        )

        normalized = spec.normalized_config(requested)

        self.assertNotIn("xcd_remap", normalized.config)
        self.assertIsNone(requested.config["xcd_remap"])

    def test_grouped_static_seed_representation(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._make_cute_tcgen05_spec()
        spec.allowed_pid_types = ("flat",)
        seeds: list[helion.Config] = []
        modes = (TCGEN05_GROUPED_MODE_STATIC, TCGEN05_GROUPED_MODE_DYNAMIC)
        for pid_type, mode in zip(
            ("persistent_blocked", "persistent_interleaved"), modes, strict=True
        ):
            seed = helion.Config(block_sizes=[128, 64, 64], pid_type=pid_type)
            seed.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = mode
            seeds.append(seed)
        spec.compiler_seed_configs = seeds

        generation = ConfigGeneration(spec)
        pairs = generation.seed_flat_config_pairs()
        self.assertEqual(len(pairs), 2)
        [pid_index], _ = generation._key_to_flat_indices["pid_type"]
        [mode_index], _ = generation._key_to_flat_indices[
            TCGEN05_GROUPED_MODE_CONFIG_KEY
        ]
        pid_fragment = generation.flat_spec[pid_index]
        mode_fragment = generation.flat_spec[mode_index]
        for (flat, normalized), mode in zip(pairs, modes, strict=True):
            self.assertEqual(normalized.config[TCGEN05_GROUPED_MODE_CONFIG_KEY], mode)
            generation.encode_config(flat)
            self.assertEqual(pid_fragment.pattern_neighbors(flat[pid_index]), ["flat"])
            self.assertEqual(mode_fragment.pattern_neighbors(flat[mode_index]), [None])

    def test_grouped_worklist_source_tile_normalization(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        self.assertEqual(
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
            tuple(sorted(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES)),
        )
        self.assertEqual(
            spec._tcgen05_optional_fragments()[
                TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
            ].default(),
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
        )
        for source_m_tile in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES:
            with self.subTest(source_m_tile=source_m_tile):
                config = helion.Config(
                    block_sizes=[256, 128, 128],
                    pid_type="persistent_interleaved",
                    tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                    tcgen05_grouped_worklist_source_m_tile=source_m_tile,
                )
                spec.normalize(config)
                self.assertEqual(
                    config.config[TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY],
                    source_m_tile,
                )

        for invalid_value in (256.0, True, 225):
            wrong_type_config = helion.Config(
                block_sizes=[256, 128, 128],
                pid_type="persistent_interleaved",
                tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                tcgen05_grouped_worklist_source_m_tile=256,
            )
            wrong_type_config.config[
                TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY
            ] = invalid_value
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                r"source_m_tile.*\(32, 224, 256\)",
            ):
                spec.normalize(wrong_type_config)

        invalid = helion.Config(
            block_sizes=[256, 128, 128],
            tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_DYNAMIC,
            tcgen05_grouped_worklist_source_m_tile=256,
        )
        with self.assertRaisesRegex(exc.InvalidConfig, "source_m_tile"):
            spec.normalize(invalid)
        spec.normalize(invalid, _fix_invalid=True)
        self.assertNotIn(
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
            invalid.config,
        )

    def test_grouped_runtime_direct_normalization(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        spec.target_device_capability = (10, 0)

        def grouped_config(
            *,
            block_k: int = 64,
            grouped_mode: str = TCGEN05_GROUPED_MODE_WORKLIST_NM,
            runtime_direct: bool = True,
            clc: bool = False,
            static_signature: bool = False,
        ) -> helion.Config:
            config = helion.Config(
                block_sizes=[256, 128, block_k],
                pid_type="persistent_interleaved",
                tcgen05_grouped_mode=grouped_mode,
                tcgen05_grouped_runtime_direct=runtime_direct,
                **(
                    {
                        "tcgen05_strategy": "role_local_with_scheduler",
                        "tcgen05_warp_spec_scheduler_warps": 1,
                        "tcgen05_persistence_model": "clc_persistent",
                    }
                    if clc
                    else {}
                ),
            )
            if static_signature:
                config.config[TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY] = [
                    1,
                    256,
                    128,
                    64,
                ]
            return config

        def rejected_and_repaired(
            factory: Callable[[], helion.Config], message: str
        ) -> helion.Config:
            with self.assertRaisesRegex(exc.InvalidConfig, message):
                spec.normalize(factory())
            repaired = factory()
            spec.normalize(repaired, _fix_invalid=True)
            return repaired

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = grouped_config(block_k=128, runtime_direct=enabled)
                spec.normalize(config)
                self.assertIs(
                    config.config[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY],
                    enabled,
                )

        invalid = grouped_config(block_k=128)
        invalid.config[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = 1
        with self.assertRaisesRegex(exc.InvalidConfig, "must be a boolean"):
            spec.normalize(invalid)

        unsupported_cases = (
            {"grouped_mode": TCGEN05_GROUPED_MODE_DYNAMIC},
            {"static_signature": True},
        )
        for case in unsupported_cases:
            with self.subTest(unsupported=case):
                repaired = rejected_and_repaired(
                    lambda case=case: grouped_config(**case),
                    "must not silently fall back",
                )
            self.assertNotIn(
                TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
                repaired.config,
            )

        for fix_invalid in (False, True):
            valid_clc = grouped_config(clc=True)
            spec.normalize(valid_clc, _fix_invalid=fix_invalid)
            self.assertEqual(
                valid_clc.config["tcgen05_persistence_model"], "clc_persistent"
            )
            self.assertEqual(valid_clc.config["tcgen05_warp_spec_scheduler_warps"], 1)

        invalid_clc_cases = (
            {"runtime_direct": False},
            {"grouped_mode": TCGEN05_GROUPED_MODE_DYNAMIC},
            {"static_signature": True},
        )
        for case in invalid_clc_cases:
            with self.subTest(invalid_clc=case):
                repaired_clc = rejected_and_repaired(
                    lambda case=case: grouped_config(clc=True, **case),
                    "exact one-record-per-cluster tile table",
                )
            self.assertNotEqual(
                repaired_clc.config.get("tcgen05_persistence_model"),
                "clc_persistent",
            )

        repaired_with_reservation = grouped_config(clc=True, runtime_direct=False)
        repaired_with_reservation.config[
            TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
        ] = 52
        spec.normalize(repaired_with_reservation, _fix_invalid=True)
        self.assertNotEqual(
            repaired_with_reservation.config.get("tcgen05_persistence_model"),
            "clc_persistent",
        )
        self.assertEqual(
            repaired_with_reservation.config[
                TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
            ],
            52,
        )

    def test_grouped_worklist_panel_requires_runtime_direct(self) -> None:
        spec = self._make_cute_tcgen05_spec()

        def panel_config(*, runtime_direct: bool | None = None) -> helion.Config:
            config = helion.Config(
                block_sizes=[256, 128, 64],
                pid_type="persistent_interleaved",
                tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                tcgen05_l2_swizzle_size=8,
            )
            if runtime_direct is not None:
                config.config[TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY] = (
                    runtime_direct
                )
            return config

        for runtime_direct in (None, False):
            with (
                self.subTest(runtime_direct=runtime_direct),
                self.assertRaisesRegex(
                    exc.InvalidConfig,
                    "requires tcgen05_grouped_runtime_direct=True",
                ),
            ):
                spec.normalize(panel_config(runtime_direct=runtime_direct))

            repaired = panel_config(runtime_direct=runtime_direct)
            spec.normalize(repaired, _fix_invalid=True)
            self.assertEqual(
                repaired.config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY],
                1,
            )

        runtime_direct = panel_config(runtime_direct=True)
        spec.normalize(runtime_direct)
        self.assertEqual(
            runtime_direct.config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY],
            8,
        )

    def test_grouped_worklist_seed_fields_round_trip(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        cases = (
            (
                "source_m_tile",
                128,
                TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
                TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
            ),
            (
                "runtime_direct",
                64,
                TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY,
                True,
            ),
        )
        for name, block_k, key, expected in cases:
            with self.subTest(name=name):
                spec = self._make_cute_tcgen05_spec()
                seed = helion.Config(
                    block_sizes=[256, 128, block_k],
                    pid_type="persistent_interleaved",
                    tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                )
                seed.config[key] = expected
                spec.compiler_seed_configs = [seed]
                generation = ConfigGeneration(spec)
                [(flat, normalized)] = generation.seed_flat_config_pairs()
                self.assertEqual(normalized.config[key], expected)
                generation.encode_config(flat)
                self.assertEqual(generation.unflatten(flat).config[key], expected)

    def test_grouped_worklist_one_cta_accepts_bk128_ab5(self) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        config = self._grouped_worklist_config(
            block_sizes=[256, 128, 128],
            source_m_tile=TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
            cluster_m=1,
            num_stages=7,
            num_warps=8,
            tcgen05_ab_stages=5,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_consumer_regs=256,
        )

        spec.normalize(config)

        self.assertEqual(config.block_sizes[:3], [256, 128, 128])
        self.assertEqual(config.config["tcgen05_ab_stages"], 5)
        self.assertEqual(config.config["tcgen05_consumer_regs"], 256)

    def test_grouped_worklist_two_cta_fix_preserves_logical_tile(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        spec.register_cute_tcgen05_mma_analysis(
            m_block_id=0,
            n_block_id=1,
            k_block_id=2,
            compile_time_static_extents=(4096, 4096, 4096),
            input_dtype=torch.bfloat16,
            has_leading_passthrough=False,
            explicit_epi_tile_compatible=True,
        )
        spec.allow_tcgen05_cluster_m2_search(static_k=128)

        for source_m_tile in (
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
            TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
        ):
            for block_k in (64, 128):
                with self.subTest(
                    source_m_tile=source_m_tile,
                    block_k=block_k,
                ):
                    config = self._grouped_worklist_config(
                        block_sizes=[256, 128, block_k],
                        source_m_tile=source_m_tile,
                    )

                    spec.normalize(config, _fix_invalid=True)

                    self.assertEqual(config.block_sizes[:3], [256, 128, block_k])
                    self.assertEqual(config.config["tcgen05_cluster_m"], 2)
                    self.assertEqual(
                        config.config["pid_type"],
                        "persistent_interleaved",
                    )

    def test_grouped_worklist_search_fix_projects_semantic_tile(self) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        cases = (
            (32, 1, [128, 256, 32], [256, 128, 64], 1),
            (32, 2, [128, 256, 256], [256, 128, 128], 2),
            # BK=96 is equidistant; the deterministic tie breaks toward 64.
            (224, 1, [128, 256, 96], [256, 128, 64], 2),
            # Exact canary family: a BN neighbor must return to logical BN=128.
            (224, 2, [256, 256, 128], [256, 128, 128], 2),
            (256, 1, [64, 64, 256], [256, 128, 128], 2),
        )
        for source_m_tile, cluster_m, block_sizes, expected, expected_cluster in cases:
            with self.subTest(
                source_m_tile=source_m_tile,
                cluster_m=cluster_m,
                block_sizes=block_sizes,
            ):
                config = self._grouped_worklist_config(
                    block_sizes=[*block_sizes],
                    source_m_tile=source_m_tile,
                    cluster_m=cluster_m,
                )

                spec._cute_tcgen05_config.fix_search_config(config.config)

                self.assertEqual(config.block_sizes[:3], expected)
                self.assertEqual(config.config["tcgen05_cluster_m"], expected_cluster)

    def test_grouped_worklist_search_fix_rejects_no_divisible_block_k(self) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        spec.allow_tcgen05_cluster_m2_search(static_k=96)
        config = self._grouped_worklist_config(
            block_sizes=[256, 256, 128],
            source_m_tile=224,
        )

        with self.assertRaisesRegex(exc.InvalidConfig, "no supported block_k"):
            spec.normalize(config, _fix_invalid=True)

    def test_grouped_worklist_config_generation_canonicalizes_neighbors(
        self,
    ) -> None:
        cases = (
            (32, 1, 256, 32, 1, 64),
            (32, 2, 256, 256, 2, 128),
            (224, 1, 256, 32, 2, 64),
            (224, 2, 256, 128, 2, 128),
            (256, 1, 64, 256, 2, 128),
        )
        for (
            source_m_tile,
            sampled_cluster,
            sampled_bn,
            sampled_bk,
            expected_cluster,
            expected_bk,
        ) in cases:
            with self.subTest(
                source_m_tile=source_m_tile,
                sampled_cluster=sampled_cluster,
                sampled_bk=sampled_bk,
            ):
                spec = self._register_default_cute_tcgen05_mma_analysis(
                    self._make_cute_tcgen05_spec()
                )
                generation, flat, n_flat, k_flat, cluster_flat = (
                    self._grouped_worklist_generation(
                        spec,
                        source_m_tile=source_m_tile,
                    )
                )
                neighbor = [*flat]
                neighbor[n_flat] = sampled_bn
                neighbor[k_flat] = sampled_bk
                neighbor[cluster_flat] = sampled_cluster

                canonical_flat, normalized = generation.canonicalize_flat(neighbor)

                self.assertEqual(normalized.block_sizes[:3], [256, 128, expected_bk])
                self.assertEqual(
                    normalized.config["tcgen05_cluster_m"], expected_cluster
                )
                self.assertEqual(canonical_flat[n_flat], 128)
                self.assertEqual(canonical_flat[k_flat], expected_bk)
                self.assertEqual(canonical_flat[cluster_flat], expected_cluster)

    def test_grouped_worklist_config_generation_uses_fact_static_k(self) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        self._add_cute_tcgen05_matmul_fact(spec, static_k=192)
        # A fact for another semantic K block must not constrain this MMA.
        self._add_cute_tcgen05_matmul_fact(
            spec,
            static_k=96,
            k_block_id=1,
        )
        self.assertIsNone(spec._cute_tcgen05_config.cluster_m2_search_constraints)
        generation, flat, _n_flat, k_flat, _cluster_flat = (
            self._grouped_worklist_generation(
                spec,
                source_m_tile=224,
            )
        )
        neighbor = [*flat]
        neighbor[k_flat] = 128

        canonical_flat, normalized = generation.canonicalize_flat(neighbor)

        self.assertEqual(normalized.block_sizes[:3], [256, 128, 64])
        self.assertEqual(canonical_flat[k_flat], 64)

    def test_grouped_worklist_config_generation_uses_aliased_fact_static_k(
        self,
    ) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        k_spec = spec.block_sizes[2]
        k_spec.block_ids.append(7)
        spec.block_sizes[2] = k_spec
        self._add_cute_tcgen05_matmul_fact(spec, static_k=192, k_block_id=7)
        generation, flat, _n_flat, k_flat, _cluster_flat = (
            self._grouped_worklist_generation(
                spec,
                source_m_tile=224,
            )
        )
        neighbor = [*flat]
        neighbor[k_flat] = 128

        canonical_flat, normalized = generation.canonicalize_flat(neighbor)

        self.assertEqual(normalized.block_sizes[:3], [256, 128, 64])
        self.assertEqual(canonical_flat[k_flat], 64)

    def test_grouped_worklist_fact_static_k_rejects_without_constraints(
        self,
    ) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        self._add_cute_tcgen05_matmul_fact(spec, static_k=96)
        self.assertIsNone(spec._cute_tcgen05_config.cluster_m2_search_constraints)
        config = self._grouped_worklist_config(source_m_tile=224)

        with self.assertRaisesRegex(exc.InvalidConfig, "no supported block_k"):
            spec.normalize(config, _fix_invalid=True)

    def test_grouped_worklist_config_generation_respects_k_constraints(
        self,
    ) -> None:
        cases = (
            # BK128 does not divide K192, so choose the exact BK64 profile.
            (192, None, False, 224, 2, 1, 128, 2, 64),
            # Generic max-K-tile limits must not rewrite a valid one-CTA BK64.
            (32768, None, False, 32, 1, 1, 64, 1, 64),
            # Generic edge-tail policy must not select a non-dividing worklist BK.
            (192, None, True, 224, 2, 2, 128, 2, 64),
            # The source-32 two-CTA profile owns the same exact-divisibility rule.
            (192, None, True, 32, 2, 2, 128, 2, 64),
            # Generic constraints and matching semantic facts are intersected.
            (256, 192, False, 224, 2, 2, 128, 2, 64),
        )
        for (
            static_k,
            fact_static_k,
            allow_edge_k_tail_family,
            source_m_tile,
            seed_cluster,
            sampled_cluster,
            sampled_bk,
            expected_cluster,
            expected_bk,
        ) in cases:
            with self.subTest(
                static_k=static_k,
                allow_edge_k_tail_family=allow_edge_k_tail_family,
                source_m_tile=source_m_tile,
                sampled_cluster=sampled_cluster,
            ):
                spec = self._register_default_cute_tcgen05_mma_analysis(
                    self._make_cute_tcgen05_spec()
                )
                if fact_static_k is not None:
                    self._add_cute_tcgen05_matmul_fact(
                        spec,
                        static_k=fact_static_k,
                    )
                spec.allow_tcgen05_cluster_m2_search(
                    static_k=static_k,
                    allow_edge_k_tail_family=allow_edge_k_tail_family,
                )
                generation, flat, n_flat, k_flat, cluster_flat = (
                    self._grouped_worklist_generation(
                        spec,
                        source_m_tile=source_m_tile,
                        cluster_m=seed_cluster,
                    )
                )
                neighbor = [*flat]
                neighbor[n_flat] = 256
                neighbor[k_flat] = sampled_bk
                neighbor[cluster_flat] = sampled_cluster

                canonical_flat, normalized = generation.canonicalize_flat(neighbor)

                self.assertEqual(normalized.block_sizes[:3], [256, 128, expected_bk])
                self.assertEqual(
                    normalized.config["tcgen05_cluster_m"], expected_cluster
                )
                self.assertEqual(canonical_flat[n_flat], 128)
                self.assertEqual(canonical_flat[k_flat], expected_bk)
                self.assertEqual(canonical_flat[cluster_flat], expected_cluster)

    def test_grouped_worklist_two_cta_seed_survives_without_generic_constraints(
        self,
    ) -> None:
        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        self.assertIsNone(spec._cute_tcgen05_config.cluster_m2_search_constraints)

        for source_m_tile in (
            TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
            TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
        ):
            with self.subTest(source_m_tile=source_m_tile):
                generation, flat, _n_flat, _k_flat, _cluster_flat = (
                    self._grouped_worklist_generation(
                        spec,
                        source_m_tile=source_m_tile,
                    )
                )
                [seed] = spec.compiler_seed_configs
                spec.compiler_default_config = seed

                effective = spec.default_config()
                self.assertEqual(effective.config["tcgen05_cluster_m"], 2)
                self.assertEqual(effective.config["block_sizes"][:3], [256, 128, 64])

                [(seed_flat, normalized)] = generation.seed_flat_config_pairs()
                self.assertEqual(seed_flat, flat)
                self.assertEqual(normalized.config["tcgen05_cluster_m"], 2)
                unflattened = generation.unflatten([*flat])
                self.assertEqual(unflattened.config["tcgen05_cluster_m"], 2)
                self.assertEqual(unflattened.config["block_sizes"][:3], [256, 128, 64])

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

    def test_grouped_static_reserved_sms_envelope(self) -> None:
        spec = self._make_cute_tcgen05_spec()

        def grouped_config(reserved_sms: object) -> helion.Config:
            config = helion.Config(
                block_sizes=[128, 64, 64],
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
            )
            config.config.update(
                {
                    TCGEN05_GROUPED_MODE_CONFIG_KEY: TCGEN05_GROUPED_MODE_DYNAMIC,
                    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY: reserved_sms,
                }
            )
            return config

        for mode in (True, "unknown", None):
            invalid = helion.Config(
                block_sizes=[128, 64, 64],
                **{TCGEN05_GROUPED_MODE_CONFIG_KEY: mode},
            )
            with self.assertRaisesRegex(exc.InvalidConfig, "grouped_mode"):
                spec.normalize(invalid)
            spec.normalize(invalid, _fix_invalid=True)
            self.assertNotIn(TCGEN05_GROUPED_MODE_CONFIG_KEY, invalid.config)

        for reserved_sms in (4, 0):
            config = grouped_config(reserved_sms)
            spec.normalize(config)
            if reserved_sms:
                self.assertEqual(
                    config.config[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY],
                    reserved_sms,
                )
            else:
                self.assertNotIn(
                    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
                    config.config,
                )

        cases: tuple[tuple[str, dict[str, object]], ...] = (
            ("missing_grouped_mode", {}),
            (
                "static_mode",
                {
                    "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
                    TCGEN05_GROUPED_MODE_CONFIG_KEY: TCGEN05_GROUPED_MODE_STATIC,
                },
            ),
            (
                "nonpersistent_pid",
                {
                    "pid_type": "flat",
                    TCGEN05_GROUPED_MODE_CONFIG_KEY: TCGEN05_GROUPED_MODE_DYNAMIC,
                },
            ),
        )

        for name, extra_config in cases:
            with self.subTest(name=name):
                config = helion.Config(block_sizes=[128, 64, 64])
                config.config.update(extra_config)
                config.config[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = 5
                spec.normalize(config)
                self.assertNotIn(
                    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
                    config.config,
                )

        for value in (-1, True, "4", None, TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX + 1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(exc.InvalidConfig, "reserved_sms"),
            ):
                spec.normalize(grouped_config(value))
            repaired = grouped_config(value)
            spec.normalize(repaired, _fix_invalid=True)
            self.assertNotIn(
                TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
                repaired.config,
            )

    def test_grouped_rejects_num_sm_multiplier(self) -> None:
        spec = self._make_cute_tcgen05_spec()

        def make_config(grouped_mode: str) -> helion.Config:
            return helion.Config(
                block_sizes=[256, 128, 64],
                num_sm_multiplier=2,
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                tcgen05_grouped_mode=grouped_mode,
            )

        for grouped_mode in (
            TCGEN05_GROUPED_MODE_STATIC,
            TCGEN05_GROUPED_MODE_DYNAMIC,
            TCGEN05_GROUPED_MODE_DIRECT,
            TCGEN05_GROUPED_MODE_WORKLIST_NM,
        ):
            with self.subTest(grouped_mode=grouped_mode):
                with self.assertRaisesRegex(exc.InvalidConfig, "num_sm_multiplier=1"):
                    spec.normalize(make_config(grouped_mode))

                repaired = make_config(grouped_mode)
                spec.normalize(repaired, _fix_invalid=True)
                self.assertNotIn("num_sm_multiplier", repaired.config)

    def test_grouped_clc_rejects_ignored_occupancy_limits(self) -> None:
        from helion._compiler.cute.strategies import (
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
        )
        from helion._compiler.cute.strategies import Tcgen05PersistenceModel

        spec = self._make_cute_tcgen05_spec()

        def make_config() -> helion.Config:
            return helion.Config(
                block_sizes=[256, 128, 64],
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
                **{
                    TCGEN05_GROUPED_MODE_CONFIG_KEY: (TCGEN05_GROUPED_MODE_WORKLIST_NM),
                    TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY: True,
                    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY: 4,
                    TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                        Tcgen05PersistenceModel.CLC_PERSISTENT.value
                    ),
                },
            )

        with self.assertRaisesRegex(exc.InvalidConfig, "exact full tile-record grid"):
            spec.normalize(make_config())

        repaired = make_config()
        spec.normalize(repaired, _fix_invalid=True)
        self.assertNotIn(
            TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
            repaired.config,
        )

    def test_grouped_static_problem_signature_envelope(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        key = TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
        signature = [2, 128, 64, 32, 16, 128, 1536]

        def make_config(
            value: object, mode: object = TCGEN05_GROUPED_MODE_DIRECT
        ) -> helion.Config:
            config = helion.Config(
                block_sizes=[128, 64, 64],
                pid_type="persistent_interleaved",
            )
            config.config[key] = value
            if mode is not None:
                config.config[TCGEN05_GROUPED_MODE_CONFIG_KEY] = mode
            return config

        config = make_config(signature)
        spec.normalize(config)
        self.assertEqual(config.config[key], signature)
        minimized = config.minimize(spec)
        spec.normalize(minimized)
        self.assertEqual(minimized.config[key], signature)

        dynamic = make_config(signature, TCGEN05_GROUPED_MODE_DYNAMIC)
        spec.normalize(dynamic)
        self.assertEqual(dynamic.config[key], signature)

        static = make_config(signature, TCGEN05_GROUPED_MODE_STATIC)
        spec.normalize(static)
        self.assertEqual(static.config[key], signature)

        invalid_values = (
            (2, 128, 64, 32, 16, 128, 1536),
            [True, 128, 64, 32],
            [2, 128, 64, 32],
            [1, 128, 0, 32],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                invalid = make_config(value)
                with self.assertRaisesRegex(exc.InvalidConfig, key):
                    spec.normalize(invalid)
                fixed = make_config(value)
                spec.normalize(fixed, _fix_invalid=True)
                self.assertNotIn(key, fixed.config)

        for mode in (None, TCGEN05_GROUPED_MODE_WORKLIST_NM):
            with self.subTest(mode=mode):
                invalid = make_config(signature, mode)
                with self.assertRaisesRegex(exc.InvalidConfig, key):
                    spec.normalize(invalid)
                fixed = make_config(signature, mode)
                spec.normalize(fixed, _fix_invalid=True)
                self.assertNotIn(key, fixed.config)

    def test_deep_ab_stage_mode_envelopes_and_stage7_roundtrip(self) -> None:
        def selected_config() -> helion.Config:
            return self._grouped_worklist_config(
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                num_stages=7,
                num_warps=8,
                tcgen05_ab_stages=7,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                tcgen05_num_epi_warps=4,
                tcgen05_consumer_regs=240,
            )

        def dynamic_config(
            ab_stages: int, c_stages: int | None = None
        ) -> helion.Config:
            config = helion.Config(
                block_sizes=[128, 64, 64],
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                tcgen05_ab_stages=ab_stages,
                tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_DYNAMIC,
            )
            if c_stages is not None:
                config.config["tcgen05_c_stages"] = c_stages
            return config

        spec = self._register_default_cute_tcgen05_mma_analysis(
            self._make_cute_tcgen05_spec()
        )
        grouped_ab4 = dynamic_config(4)
        spec.normalize(grouped_ab4)
        grouped_ab4.config["tcgen05_ab_stages"] = 4.0
        with self.assertRaisesRegex(exc.InvalidConfig, "tcgen05_ab_stages"):
            spec.normalize(grouped_ab4)

        grouped_ab8 = dynamic_config(8, 4)
        spec.normalize(grouped_ab8)
        self.assertEqual(grouped_ab8.config["tcgen05_ab_stages"], 8)
        minimized_ab8 = grouped_ab8.minimize(spec)
        spec.normalize(minimized_ab8)
        self.assertEqual(minimized_ab8.config["tcgen05_ab_stages"], 8)
        self.assertEqual(minimized_ab8.config["tcgen05_c_stages"], 4)

        for ab_stages, c_stages in ((8, 2), (4, 4)):
            mismatched = dynamic_config(ab_stages, c_stages)
            with (
                self.subTest(ab_stages=ab_stages, c_stages=c_stages),
                self.assertRaisesRegex(exc.InvalidConfig, "tcgen05_ab_stages"),
            ):
                spec.normalize(mismatched)

        config = selected_config()
        spec.normalize(config)
        self.assertEqual(config.block_sizes, [256, 128, 64])
        self.assertEqual(config.config["tcgen05_ab_stages"], 7)
        minimized = config.minimize(spec)
        self.assertNotIn("tcgen05_cluster_n", minimized.config)
        self.assertNotIn("tcgen05_acc_stages", minimized.config)
        self.assertNotIn("tcgen05_c_stages", minimized.config)
        spec.normalize(minimized)
        self.assertEqual(minimized.config["tcgen05_ab_stages"], 7)

        constrained_spec = self._constrained_grouped_worklist_spec(1)
        with self.assertRaisesRegex(exc.InvalidConfig, "tcgen05_ab_stages"):
            constrained_spec.normalize(selected_config())

    def test_grouped_worklist_two_cta_deep_ab_uses_conservative_smem_capacity(
        self,
    ) -> None:
        def selected_config(source_m_tile: int, ab_stages: int) -> helion.Config:
            return self._grouped_worklist_config(
                source_m_tile=source_m_tile,
                pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
                tcgen05_ab_stages=ab_stages,
            )

        for invalid_cluster_m in (True, 4):
            with self.subTest(invalid_cluster_m=invalid_cluster_m):
                invalid = selected_config(
                    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
                    7,
                )
                invalid.config["tcgen05_cluster_m"] = invalid_cluster_m
                self.assertIsNone(
                    resolve_tcgen05_grouped_worklist_mma_profile(
                        invalid,
                        block_k=64,
                    )
                )

        b200_capacity_bytes = TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
        constrained_budget_bytes = (
            b200_capacity_bytes - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        cases = (
            (TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT, 7, True, 232_448),
            (TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE, 6, True, 214_016),
            (TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE, 7, False, 246_784),
        )
        for source_m_tile, ab_stages, should_fit, required_bytes in cases:
            with self.subTest(
                source_m_tile=source_m_tile,
                ab_stages=ab_stages,
                should_fit=should_fit,
            ):
                config = selected_config(source_m_tile, ab_stages)
                profile = resolve_tcgen05_grouped_worklist_mma_profile(
                    config,
                    block_k=64,
                )
                assert profile is not None
                self.assertEqual((profile.mma_m, profile.mma_n), (256, source_m_tile))
                self.assertEqual(
                    tcgen05_grouped_worklist_smem_bytes(
                        group_count=1,
                        device_split_sizes=False,
                        sched_stage_count=1,
                        bm=profile.mma_m,
                        bn=profile.mma_n,
                        bk=64,
                        dtype_bytes=2,
                        ab_stages=ab_stages,
                        acc_stages=2,
                        c_stages=2,
                        cluster_m=2,
                    ),
                    required_bytes,
                )

                spec = self._constrained_grouped_worklist_spec(constrained_budget_bytes)
                spec.register_cute_tcgen05_grouped_worklist_smem_facts(
                    group_count=1,
                    device_split_sizes=False,
                )
                if should_fit:
                    spec.normalize(config)
                else:
                    with self.assertRaisesRegex(exc.InvalidConfig, "tcgen05_ab_stages"):
                        spec.normalize(config)

    def test_grouped_worklist_deep_ab_normalization_uses_device_split_facts(
        self,
    ) -> None:
        config = self._grouped_worklist_config(
            pid_type=TCGEN05_TWO_CTA_SEED_PID_TYPE,
            tcgen05_ab_stages=7,
        )
        constrained_budget_bytes = (
            TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
            - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        for group_count, should_fit in ((20, True), (21, False)):
            with self.subTest(group_count=group_count, should_fit=should_fit):
                spec = self._constrained_grouped_worklist_spec(constrained_budget_bytes)
                spec.register_cute_tcgen05_grouped_worklist_smem_facts(
                    group_count=group_count,
                    device_split_sizes=True,
                )
                if should_fit:
                    spec.normalize(helion.Config.from_dict(config.config))
                else:
                    with self.assertRaisesRegex(exc.InvalidConfig, "tcgen05_ab_stages"):
                        spec.normalize(helion.Config.from_dict(config.config))

    def test_reviewed_grouped_worklist_seeds_fit_conservative_b200_smem(self) -> None:
        seed_inputs = (
            (6, 6 * 32, 4096, 4096, "k", 32),
            (6, 6 * 224, 7168, 3072, "k", 224),
            (8, 8 * 4096, 4096, 2048, "k", 256),
            (8, 8 * 4096, 4096, 2048, "n", 256),
        )
        reviewed_configs = [
            config
            for groups, packed_m, n, k, b_major, source_m_tile in seed_inputs
            for config in _tcgen05_grouped_worklist_seed_family(
                groups=groups,
                packed_m=packed_m,
                n=n,
                k=k,
                b_major=b_major,
                source_m_tile=source_m_tile,
                num_sm=148,
                target_policy=GroupedWorklistTargetPolicy(),
            )[0]
        ]
        spec = self._constrained_grouped_worklist_spec(
            TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
            - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        spec.register_cute_tcgen05_grouped_worklist_smem_facts(
            group_count=8,
            device_split_sizes=False,
        )

        for config in reviewed_configs:
            with self.subTest(config=config.config):
                ab_stages = config.config["tcgen05_ab_stages"]
                assert isinstance(ab_stages, int)
                # AB1-3 use the ordinary admission path; deeper reviewed seeds
                # must pass the conservative all-scheduler SMEM upper bound.
                self.assertTrue(
                    ab_stages < 4
                    or spec._cute_tcgen05_config._grouped_worklist_nm_ab_config_matches(
                        config.config, ab_stages
                    )
                )

    def test_grouped_dynamic_deep_stage_smem_accounts_for_output_dtype(self) -> None:
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.tcgen05_config import CuteTcgen05Config

        config = CuteTcgen05Config(ConfigSpec(backend=CuteBackend()))
        with (
            patch.object(
                CuteTcgen05Config,
                "per_cta_smem_capacity_bytes",
                return_value=232448,
            ),
            patch(
                "helion._compiler.cute.tcgen05_config.tcgen05_default_epilogue_tile_size",
                return_value=(128, 32),
            ),
        ):
            self.assertTrue(
                config.grouped_dynamic_stages_fit_for_target(
                    dtype_bytes=2,
                    output_dtype_bytes=2,
                    device=torch.device("cuda"),
                    bm=128,
                    bn=64,
                    bk=64,
                    cluster_m=1,
                    ab_stages=8,
                    c_stages=4,
                )
            )
            self.assertFalse(
                config.grouped_dynamic_stages_fit_for_target(
                    dtype_bytes=2,
                    output_dtype_bytes=4,
                    device=torch.device("cuda"),
                    bm=128,
                    bn=64,
                    bk=64,
                    cluster_m=1,
                    ab_stages=8,
                    c_stages=4,
                )
            )

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

        def choices(fragments: Mapping[str, object], key: str) -> tuple[object, ...]:
            return self._enum_fragment(fragments[key]).choices

        narrow = tcgen05_config.strategy_autotune_fragments()
        self.assertEqual(
            choices(narrow, "tcgen05_strategy"), ("role_local_monolithic",)
        )
        self.assertEqual(choices(narrow, "tcgen05_warp_spec_c_input_warps"), (0,))

        tcgen05_config.aux_kernel_detected = True
        widened = tcgen05_config.strategy_autotune_fragments()
        self.assertEqual(
            choices(widened, "tcgen05_strategy"),
            ("role_local_monolithic", "role_local_with_scheduler"),
        )
        self.assertEqual(
            choices(widened, "tcgen05_warp_spec_c_input_warps"),
            (0, 1),
        )

    def test_compiler_seed_fragments_preserve_base_search_choices(self) -> None:
        spec = self._make_cute_tcgen05_spec()
        spec.compiler_seed_configs = [
            helion.Config(
                pid_type="flat",
                tcgen05_consumer_regs=240,
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
                tcgen05_persistence_model="non_persistent",
            ),
            helion.Config(
                pid_type="persistent_interleaved",
                tcgen05_persistence_model="clc_persistent",
            ),
            helion.Config.from_dict(
                {
                    "tcgen05_consumer_regs": True,
                    "tcgen05_strategy": "invalid",
                    "tcgen05_warp_spec_scheduler_warps": True,
                    "tcgen05_persistence_model": "invalid",
                }
            ),
        ]
        tcgen05_config = spec._cute_tcgen05_config

        consumer = self._enum_fragment(
            tcgen05_config.consumer_regs_autotune_fragments()["tcgen05_consumer_regs"]
        )
        strategy = tcgen05_config.strategy_autotune_fragments()
        persistence = self._enum_fragment(
            tcgen05_config.persistence_model_autotune_fragments()[
                "tcgen05_persistence_model"
            ]
        )
        strategy_fragment = self._enum_fragment(strategy["tcgen05_strategy"])
        scheduler_fragment = self._enum_fragment(
            strategy["tcgen05_warp_spec_scheduler_warps"]
        )

        self.assertEqual(consumer.choices, (256, 240))
        self.assertEqual(consumer.search_choices, (256,))
        self.assertEqual(
            strategy_fragment.choices,
            ("role_local_monolithic", "role_local_with_scheduler"),
        )
        self.assertEqual(strategy_fragment.search_choices, ("role_local_monolithic",))
        self.assertEqual(scheduler_fragment.choices, (0, 1))
        self.assertEqual(scheduler_fragment.search_choices, (0,))
        self.assertEqual(
            persistence.choices,
            ("non_persistent", "static_persistent", "clc_persistent"),
        )
        self.assertEqual(
            persistence.search_choices,
            ("non_persistent", "static_persistent"),
        )

    def test_default_seed_values_do_not_add_degenerate_fragments(self) -> None:
        baseline = self._make_cute_tcgen05_spec()
        baseline_hash = baseline.structural_fingerprint_hash()

        spec = self._make_cute_tcgen05_spec()
        spec.compiler_seed_configs = [
            helion.Config(
                pid_type="flat",
                tcgen05_consumer_regs=256,
                tcgen05_persistence_model="non_persistent",
            )
        ]
        fields = spec._cute_tcgen05_config.flat_fields()

        self.assertNotIn("tcgen05_consumer_regs", fields)
        self.assertNotIn("tcgen05_persistence_model", fields)
        self.assertEqual(spec.structural_fingerprint_hash(), baseline_hash)

        spec.compiler_seed_configs = [
            helion.Config(
                pid_type="flat",
                tcgen05_persistence_model="static_persistent",
            )
        ]
        persistence = self._enum_fragment(
            spec._cute_tcgen05_config.flat_fields()["tcgen05_persistence_model"]
        )
        self.assertEqual(
            persistence.choices,
            ("non_persistent", "static_persistent"),
        )
        self.assertIsNone(persistence.search_choices)
        self.assertNotEqual(spec.structural_fingerprint_hash(), baseline_hash)
        self.assertIn(
            (
                "tcgen05_persistence_model",
                "enum",
                repr("non_persistent"),
                repr("static_persistent"),
            ),
            spec.structural_fingerprint(),
        )

    def test_mixed_grouped_persistence_seed_search_domain(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._make_cute_tcgen05_spec()
        spec.target_device_capability = (10, 0)
        spec.allowed_pid_types = ("flat",)
        spec.compiler_seed_configs = [
            helion.Config(
                pid_type="persistent_interleaved",
                tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                tcgen05_grouped_runtime_direct=True,
            ),
            helion.Config(
                pid_type="persistent_interleaved",
                tcgen05_grouped_mode=TCGEN05_GROUPED_MODE_WORKLIST_NM,
                tcgen05_grouped_runtime_direct=True,
                tcgen05_persistence_model="clc_persistent",
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
            ),
        ]

        generation = ConfigGeneration(spec)
        [persistence_index], _ = generation._key_to_flat_indices[
            "tcgen05_persistence_model"
        ]
        persistence = self._enum_fragment(generation.flat_spec[persistence_index])
        self.assertEqual(
            persistence.choices,
            ("non_persistent", "static_persistent", "clc_persistent"),
        )
        self.assertEqual(
            persistence.search_choices,
            ("non_persistent",),
        )
        self.assertEqual(
            persistence.search_values(),
            ["non_persistent"],
        )
        self.assertIn(
            (
                "tcgen05_persistence_model",
                "enum",
                repr("non_persistent"),
                repr("static_persistent"),
                repr("clc_persistent"),
                "search",
                repr("non_persistent"),
            ),
            spec.structural_fingerprint(),
        )

        pairs = generation.seed_flat_config_pairs()
        self.assertEqual(len(pairs), 2)
        by_model = {
            normalized.config["tcgen05_persistence_model"]: flat
            for flat, normalized in pairs
        }
        self.assertEqual(set(by_model), {"static_persistent", "clc_persistent"})
        feature_dim = sum(fragment.dim() for fragment in generation.flat_spec)
        for flat in by_model.values():
            self.assertEqual(len(generation.encode_config(flat)), feature_dim)
        for model in ("static_persistent", "clc_persistent"):
            self.assertEqual(
                persistence.pattern_neighbors(by_model[model][persistence_index]),
                ["non_persistent"],
            )


if __name__ == "__main__":
    unittest.main()
