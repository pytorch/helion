from __future__ import annotations

import copy
import pickle
import tempfile
from types import SimpleNamespace
from typing import ClassVar
import unittest
from unittest.mock import patch

import torch

from test._autotuner_test_utils import _SM90_HARDWARE
from test._autotuner_test_utils import AutotunerTestCase
from test._autotuner_test_utils import _get_examples_matmul
from test._autotuner_test_utils import _pin_sm90
from test._autotuner_test_utils import basic_kernels

import helion
from helion import _compat
from helion import exc
from helion._compiler.tile_dispatch import BlockIDStrategyMapping
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import NumThreadsFragment
from helion.autotuner.config_fragment import NumWarpsFragment
from helion.autotuner.config_fragment import PermutationFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import SMALL_DIM_BLOCK_SIZE_OVERSHOOT
import helion.language as hl
from helion.language import loops


@onlyBackends(["triton", "cute"])
class TestAutotunerConfigs(AutotunerTestCase):
    @_pin_sm90
    @patch.object(_compat, "_supports_host_tensor_descriptor", lambda: False)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    @skipIfCute("golden snapshot of the Triton config space")
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_host_tensor_descriptor", lambda: False)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @patch("helion._hardware.get_hardware_info", return_value=_SM90_HARDWARE)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    @skipIfCute("golden snapshot of the Triton config space")
    def test_config_fragment1(self, _mock_hardware):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_host_tensor_descriptor", lambda: False)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @patch("helion._hardware.get_hardware_info", return_value=_SM90_HARDWARE)
    @skipIfTileIR("tileir backend will ignore `warp specialization` hint")
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    @skipIfCute("golden snapshot of the Triton config space")
    def test_config_warp_specialize_unroll(self, _mock_hardware):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"range_unroll_factors": [4], "range_warp_specializes": ([True])}
        # We expect all the unroll factors to be set to 0
        configs = ConfigGeneration(spec, overrides=overrides).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @_pin_sm90
    @patch.object(_compat, "_supports_host_tensor_descriptor", lambda: False)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    @skipIfTileIR("block-size overshoot is gated to the triton backend")
    @skipIfCute("block-size overshoot is gated to the triton backend")
    def test_small_dim_block_size_overshoot(self):
        # All dims are 16, smaller than SMALL_DIM_BLOCK_SIZE_OVERSHOOT, so the
        # generated configs may use block sizes larger than the dimensions
        # themselves (e.g. 32 or 64) -- the extra rows/cols are masked off.
        self.assertEqual(SMALL_DIM_BLOCK_SIZE_OVERSHOOT, 64)
        args = (
            torch.randn([16, 16], device=DEVICE),
            torch.randn([16, 16], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch.object(_compat, "_supports_host_tensor_descriptor", lambda: False)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @skipIfCute("overrides the Triton-only 'indexing' config key")
    def test_config_generation_overrides(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"indexing": "tensor_descriptor"}
        gen = ConfigGeneration(spec, overrides=overrides)

        flat = gen.default_flat()
        config = gen.unflatten([*flat])
        self.assertEqual(config["indexing"], "tensor_descriptor")
        configs = [gen.unflatten(gen.random_flat()) for _ in range(3)]
        self.assertEqual({cfg["indexing"] for cfg in configs}, {"tensor_descriptor"})
        indexing_choices = spec.valid_indexing_types()
        indexing_index = next(
            i
            for i, fragment in enumerate(gen.flat_spec)
            if isinstance(fragment, ListOf)
            and isinstance(fragment.inner, EnumFragment)
            and fragment.inner.choices == tuple(indexing_choices)
        )
        mutated = gen.random_flat()
        mutated[indexing_index] = "pointer"
        new_config = gen.unflatten(mutated)
        self.assertEqual(new_config["indexing"], "tensor_descriptor")
        self.assertEqual(mutated[indexing_index], "pointer")

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_save_load_config(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=2,
            num_stages=1,
            indexing="block_ptr",
            l2_grouping=32,
        )
        with tempfile.NamedTemporaryFile() as f:
            config.save(f.name)
            loaded_config = helion.Config.load(f.name)
            self.assertEqual(config, loaded_config)
        self.assertExpectedJournal(config.to_json())

    def test_config_pickle_roundtrip(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=4,
            num_stages=2,
            indexing="tensor_descriptor",
            extra_metadata={"nested": [1, 2, 3]},
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertIsInstance(restored, helion.Config)
        self.assertEqual(config, restored)
        self.assertIsNot(config, restored)
        self.assertIsNot(config.config, restored.config)

    def test_run_fixed_config(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1024, 1, 1],
                flatten_loops=[True],
                loop_orders=[[0, 2, 1]],
                num_warps=8,
            )
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))

    def test_population_member_canonicalizes_overridden_flat_values(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        key, value, other = self._scalar_override()
        gen = ConfigGeneration(spec, overrides={key: value})
        (key_idx,), _ = gen._key_to_flat_indices[key]
        raw_flat = gen.default_flat()
        original_flat = copy.deepcopy(raw_flat)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = spec
        search.config_gen = gen

        member = search.make_unbenchmarked(raw_flat)

        self.assertIsNotNone(member)
        assert member is not None
        self.assertEqual(raw_flat, original_flat)
        self.assertIsNot(member.flat_values, raw_flat)
        self.assertEqual(member.config[key], value)
        self.assertEqual(member.flat_values[key_idx], value)
        self.assertEqual(member.flat_values, gen.flatten(member.config))
        gen.encode_config(member.flat_values)

        [(seed_flat, seed_config)] = gen.user_seed_flat_config_pairs(
            [helion.Config(**{key: other})]
        )
        self.assertEqual(seed_config[key], value)
        self.assertEqual(seed_flat, gen.flatten(seed_config))

    @skipIfCute("overrides the Triton-only 'indexing' config key")
    def test_scalar_list_override_has_encodable_flat_values(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        gen = ConfigGeneration(spec, overrides={"indexing": "pointer"})

        flat, config = gen.canonicalize_flat(gen.default_flat())

        indexing_indices, is_sequence = gen._key_to_flat_indices["indexing"]
        self.assertFalse(is_sequence)
        self.assertEqual(len(indexing_indices), 1)
        self.assertEqual(flat[indexing_indices[0]], ["pointer"] * spec.indexing.length)
        self.assertEqual(flat, gen.flatten(config))
        gen.encode_config(flat)

    def test_tile_strategy_dispatch_compact_shape_uses_cached_block_lookup(self):
        """Fallback block-id lookups should reuse the precomputed strategy cache."""

        class DummyStrategy:
            block_ids: ClassVar[list[int]] = [3, 4]

            def block_size_var(self, block_idx: int) -> str:
                return f"_BLOCK_{block_idx}"

            def compact_shape(self, shapes):
                return shapes

        dispatch = TileStrategyDispatch.__new__(TileStrategyDispatch)
        dispatch.strategies = [DummyStrategy()]
        dispatch.block_id_to_strategy = BlockIDStrategyMapping()
        dispatch.block_id_to_strategy[(3, 4)] = dispatch.strategies[0]

        with patch(
            "helion._compiler.tile_dispatch.CompileEnvironment.current",
            return_value=SimpleNamespace(
                get_block_id=lambda _shape: 3,
                resolve_block_id=lambda _shape: 3,
            ),
        ):
            compacted = dispatch.compact_shape([object()])

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].size_str, "_BLOCK_3")
        self.assertEqual(compacted[0].block_ids, [3])

    def test_fragment_encoding(self):
        """Test encoding functionality for all ConfigSpecFragment types."""
        # Test BooleanFragment
        bool_frag = BooleanFragment()
        self.assertEqual(bool_frag.dim(), 1)
        self.assertEqual(bool_frag.encode(True), [1.0])
        self.assertEqual(bool_frag.encode(False), [0.0])

        # Test IntegerFragment
        int_frag = IntegerFragment(low=1, high=10, default_val=5)
        self.assertEqual(int_frag.dim(), 1)
        self.assertEqual(int_frag.encode(5), [5.0])

        # Test PowerOfTwoFragment (log2 transformation)
        pow2_frag = PowerOfTwoFragment(low=2, high=128, default_val=8)
        self.assertEqual(pow2_frag.dim(), 1)
        self.assertEqual(pow2_frag.encode(8), [3.0])  # log2(8) = 3
        self.assertEqual(pow2_frag.encode(16), [4.0])  # log2(16) = 4

        # Test NumThreadsFragment (0 is the CuTe auto-thread sentinel)
        num_threads_frag = NumThreadsFragment(high=128)
        self.assertEqual(num_threads_frag.dim(), 1)
        self.assertEqual(num_threads_frag.encode(0), [0.0])
        self.assertEqual(num_threads_frag.encode(8), [4.0])

        # Test EnumFragment (one-hot encoding)
        enum_frag = EnumFragment(choices=("a", "b", "c"))
        self.assertEqual(enum_frag.dim(), 3)
        self.assertEqual(enum_frag.encode("a"), [1.0, 0.0, 0.0])
        self.assertEqual(enum_frag.encode("b"), [0.0, 1.0, 0.0])

        # Test PermutationFragment
        perm_frag = PermutationFragment(length=3)
        self.assertEqual(perm_frag.dim(), 3)
        encoded = perm_frag.encode([0, 1, 2])
        self.assertEqual(encoded, [0.0, 1.0, 2.0])

        # Test ListOf with BooleanFragment
        list_frag = ListOf(inner=BooleanFragment(), length=3)
        self.assertEqual(list_frag.dim(), 3)
        self.assertEqual(list_frag.encode([True, False, True]), [1.0, 0.0, 1.0])

        # Test encode_dim consistency
        for fragment, value in [
            (BooleanFragment(), True),
            (IntegerFragment(1, 10, 5), 5),
            (PowerOfTwoFragment(2, 128, 8), 16),
            (NumThreadsFragment(128), 0),
            (EnumFragment(choices=("a", "b")), "b"),
        ]:
            dim = fragment.dim()
            encoded = fragment.encode(value)
            self.assertEqual(len(encoded), dim)

    def test_block_size_fragment_autotuner_min_clamp(self):
        """random_config() must not crash when autotuner_min > max_size."""
        from examples.attention import attention

        q, k, v = [
            torch.randn(4, 48, 128, 128, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        ]
        bound = attention.bind((q, k, v))
        config_spec = bound.config_spec
        config_spec.raise_grid_block_minimums()
        gen = ConfigGeneration(config_spec)
        config = gen.random_config()
        self.assertEqual(config["block_sizes"][0], 1)


class TestConfigFragmentCardinality(TestCase):
    """cardinality()/search_values() are pure, GPU-independent fragment
    methods, so they run in both normal and ref-eager modes."""

    def test_fragment_cardinality(self):
        # Number of distinct search values per fragment type. Feeds the
        # search-space size product and coverage denominators.
        self.assertIsNone(ConfigSpecFragment().cardinality())  # unknown by default
        self.assertEqual(PermutationFragment(1).cardinality(), 1)
        self.assertEqual(PermutationFragment(4).cardinality(), 24)  # 4!
        self.assertEqual(IntegerFragment(1, 8).cardinality(), 8)
        self.assertEqual(PowerOfTwoFragment(1, 64).cardinality(), 7)  # 1..64
        self.assertEqual(PowerOfTwoFragment(1, 1).cardinality(), 1)  # boundary
        self.assertEqual(BlockSizeFragment(16, 256).cardinality(), 5)
        self.assertEqual(NumWarpsFragment(1, 32).cardinality(), 6)
        self.assertEqual(EnumFragment(("a", "b", "c")).cardinality(), 3)
        # search_choices restrict the searched cardinality to the subset.
        self.assertEqual(
            EnumFragment(("a", "b", "c", "d"), search_choices=("a", "c")).cardinality(),
            2,
        )
        self.assertEqual(BooleanFragment().cardinality(), 2)
        # NumThreads: "0" (auto) plus powers of two up to high.
        self.assertEqual(NumThreadsFragment(256).cardinality(), 10)
        # ListOf is combinatorial (inner ** length), not linear.
        self.assertEqual(
            ListOf(EnumFragment(("a", "b", "c")), length=4).cardinality(), 81
        )
        # ListOf over an unknown-cardinality inner is itself unknown.
        self.assertIsNone(ListOf(ConfigSpecFragment(), length=3).cardinality())

    def test_fragment_search_values(self):
        # Explicit enumerable values, and the limit guard that avoids
        # materializing very large ranges.
        self.assertIsNone(ConfigSpecFragment().search_values())
        self.assertEqual(
            IntegerFragment(1, 8).search_values(), [1, 2, 3, 4, 5, 6, 7, 8]
        )
        self.assertEqual(
            PowerOfTwoFragment(1, 64).search_values(), [1, 2, 4, 8, 16, 32, 64]
        )
        self.assertEqual(
            BlockSizeFragment(16, 256).search_values(), [16, 32, 64, 128, 256]
        )
        self.assertEqual(EnumFragment(("a", "b", "c")).search_values(), ["a", "b", "c"])
        self.assertEqual(
            EnumFragment(
                ("a", "b", "c", "d"), search_choices=("a", "c")
            ).search_values(),
            ["a", "c"],
        )
        self.assertEqual(BooleanFragment().search_values(), [False, True])
        self.assertEqual(
            NumThreadsFragment(256).search_values(),
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        # limit guard: None (not a materialized list) above the limit, full below.
        self.assertIsNone(IntegerFragment(0, 200).search_values())
        self.assertEqual(len(IntegerFragment(0, 200).search_values(limit=1000)), 201)
        # ListOf combinations are not enumerated.
        self.assertIsNone(
            ListOf(EnumFragment(("a", "b", "c")), length=4).search_values()
        )

    def test_power_of_two_fragment_off_grid_seed(self) -> None:
        fragment = PowerOfTwoFragment(1, 128)

        # Explicit configs may start off the autotuned power-of-two surface.
        # Pattern and differential search must re-enter that surface without
        # changing any of its generated candidates.
        self.assertEqual(fragment.pattern_neighbors(3), [2, 4])
        self.assertEqual(fragment.pattern_neighbors(5), [4, 8])
        self.assertEqual(fragment.pattern_neighbors(5, radius=2), [2, 4, 8, 16])
        self.assertEqual(fragment.differential_mutation(3, 1, 2), 2)
        self.assertEqual(fragment.differential_mutation(3, 2, 1), 4)
        self.assertEqual(fragment.differential_mutation(3, 1, 1), 2)
        self.assertEqual(fragment.differential_mutation(6, 1, 1), 4)

        clamped = PowerOfTwoFragment(1, 32)
        self.assertEqual(clamped.pattern_neighbors(48), [32])
        self.assertEqual(clamped.pattern_neighbors(64), [32])
        self.assertEqual(clamped.differential_mutation(48, 1, 2), 32)
        self.assertEqual(clamped.differential_mutation(64, 2, 1), 32)

        # Existing on-grid behavior is unchanged.
        self.assertEqual(fragment.pattern_neighbors(4), [2, 8])
        self.assertEqual(fragment.differential_mutation(4, 1, 2), 2)
        self.assertEqual(fragment.differential_mutation(4, 2, 1), 8)
        self.assertEqual(fragment.differential_mutation(4, 1, 1), 4)

    def test_enum_fragment_coverage_choices(self):
        fragment = EnumFragment(
            ("a", "b", "c"),
            search_choices=("a", "b", "c"),
            coverage_choices=("a", "c"),
        )

        self.assertEqual(fragment.cardinality(), 3)
        self.assertEqual(fragment.search_values(), ["a", "b", "c"])
        self.assertEqual(fragment.pattern_neighbors("a"), ["b", "c"])
        with patch("helion.autotuner.config_fragment.random.choice", return_value="b"):
            self.assertEqual(fragment.random(), "b")
        self.assertEqual(
            fragment.fingerprint(),
            (
                "enum",
                "'a'",
                "'b'",
                "'c'",
                "search",
                "'a'",
                "'b'",
                "'c'",
                "coverage",
                "'a'",
                "'c'",
            ),
        )

        with self.assertRaisesRegex(ValueError, "coverage_choices must not be empty"):
            EnumFragment(("a", "b"), coverage_choices=())
        with self.assertRaisesRegex(
            ValueError, "coverage_choices must be a subset of active search choices"
        ):
            EnumFragment(("a", "b"), coverage_choices=("c",))
        with self.assertRaisesRegex(
            ValueError, "coverage_choices must be a subset of active search choices"
        ):
            EnumFragment(
                ("a", "b", "c"),
                search_choices=("a", "b"),
                coverage_choices=("c",),
            )

    def test_block_id_sequence_cardinality(self):
        # Product of per-item fragment cardinalities; empty -> 1 (neutral);
        # an unknown-cardinality item makes the whole sequence unknown.
        from helion.autotuner.block_id_sequence import BlockIdSequence
        from helion.autotuner.block_id_sequence import _BlockIdItem

        class _Item(_BlockIdItem):
            def __init__(self, ids, frag):
                super().__init__(ids)
                self._f = frag

            def _fragment(self, base):
                return self._f

        seq = BlockIdSequence()
        seq.append(_Item([0], EnumFragment(("a", "b", "c"))))
        seq.append(_Item([1], EnumFragment(("x", "y"))))
        self.assertEqual(seq.cardinality(None), 6)  # 3 * 2
        self.assertEqual(BlockIdSequence().cardinality(None), 1)

        unknown = BlockIdSequence()
        unknown.append(_Item([0], ConfigSpecFragment()))
        self.assertIsNone(unknown.cardinality(None))


class TestConfigValuePriors(TestCase):
    """Backend-supplied per-key priors bias the random half of the initial
    population (config_generation), while the other half stays uniform."""

    def _add_config_gen(self) -> tuple[ConfigGeneration, object]:
        @helion.kernel(autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(256, 256, device=DEVICE)
        bound = add.bind((a, a))
        return ConfigGeneration(bound.config_spec), bound

    @staticmethod
    def _forceable_slot(gen: ConfigGeneration) -> tuple[str, int, object]:
        """Pick a config key the live kernel exposes and a value to force it to.

        Backends expose different keys (e.g. the default backend has a scalar
        ``num_warps`` slot; the cute backend exposes only sequence keys such as
        ``block_sizes``), so the priors tests select a target dynamically rather
        than hard-coding ``num_warps``. Returns ``(key, flat_index, value)``
        where ``value`` is representable by the slot's fragment.
        """
        # Prefer a scalar ``num_warps`` slot when present: a power-of-two value
        # in its range that won't be altered by repair logic.
        for key, (indices, is_sequence) in gen._key_to_flat_indices.items():
            if key == "num_warps" and not is_sequence and indices:
                return key, indices[0], 4
        # Otherwise use the first block-size slot, which every backend exposes.
        block_key, (block_indices, _is_seq) = next(
            (k, v) for k, v in gen._key_to_flat_indices.items() if k == "block_sizes"
        )
        flat_idx = block_indices[0]
        fragment = gen.flat_spec[flat_idx]
        # Use the fragment's own default: guaranteed representable and stable.
        return block_key, flat_idx, fragment.default()

    def test_no_priors_falls_through_to_uniform(self) -> None:
        gen, _ = self._add_config_gen()
        # With no priors, biased sampling is exactly uniform sampling and the
        # population fill is unchanged. Control the priors explicitly rather than
        # relying on the ambient backend default (the cute backend, for example,
        # supplies non-empty priors).
        gen._config_value_priors = {}
        self.assertEqual(gen._config_value_priors, {})
        flat = gen.biased_random_flat()
        self.assertEqual(len(flat), len(gen.flat_spec))

    def test_prior_forces_value(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, flat_idx, value = self._forceable_slot(gen)
        # Control the priors directly on the instance so the test is independent
        # of whatever priors the active backend supplies.
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        # Disable size repair/shrink so the forced value is observed verbatim.
        with (
            patch.object(gen, "shrink_config", lambda *a, **k: None),
            patch.object(gen, "_repair_cute_num_threads", lambda *a, **k: None),
        ):
            for _ in range(25):
                self.assertEqual(gen.biased_random_flat()[flat_idx], value)

    def test_population_fill_is_half_biased(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, _flat_idx, value = self._forceable_slot(gen)
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        with (
            # Suppress backend-supplied compiler seeds so the random-fill count
            # is deterministic across backends (cute seeds a few configs).
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(
                gen, "biased_random_flat", wraps=gen.biased_random_flat
            ) as biased,
            patch.object(gen, "random_flat", wraps=gen.random_flat) as uniform,
        ):
            # 1 default + 10 random fill slots (seeds suppressed above).
            gen.random_population_flat(11)
        # Duplicate or invalid draws are retried, but attempts continue to
        # alternate between biased and uniform sampling.
        self.assertGreaterEqual(biased.call_count + uniform.call_count, 10)
        self.assertLessEqual(abs(biased.call_count - uniform.call_count), 1)

    def test_population_fill_pads_after_unique_configs_exhausted(self) -> None:
        gen, _ = self._add_config_gen()
        default_flat = gen.default_flat()
        normalized = gen.unflatten(default_flat)
        gen.config_spec.cute_flash_search_enabled = True
        with (
            patch.object(gen, "default_flat", return_value=default_flat),
            patch.object(gen, "user_seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "biased_random_flat", return_value=default_flat),
            patch.object(gen, "random_flat", return_value=default_flat),
            patch.object(gen, "unflatten", return_value=normalized),
            patch.object(gen, "flatten", return_value=default_flat),
        ):
            population = gen.random_population_flat(6)
        self.assertEqual(population, [default_flat] * 6)

    def test_flash_population_stores_normalized_flat_configs(self) -> None:
        gen, _ = self._add_config_gen()
        default_flat = gen.default_flat()
        normalized = gen.unflatten(default_flat)
        gen.config_spec.cute_flash_search_enabled = True
        canonical_flat = [*default_flat]
        with (
            patch.object(gen, "default_flat", return_value=default_flat),
            patch.object(gen, "user_seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "unflatten", return_value=normalized),
            patch.object(gen, "flatten", return_value=canonical_flat) as flatten,
        ):
            population = gen.random_population_flat(1)
        self.assertEqual(population, [canonical_flat])
        flatten.assert_called_once_with(normalized)

    def test_non_flash_population_propagates_generation_failure(self) -> None:
        gen, _ = self._add_config_gen()
        gen.config_spec.cute_flash_search_enabled = False
        with (
            patch.object(
                gen,
                "random_population_flat",
                side_effect=exc.InvalidConfig("invalid default"),
            ),
            self.assertRaisesRegex(exc.InvalidConfig, "invalid default"),
        ):
            gen.random_population(1)


if __name__ == "__main__":
    unittest.main()
