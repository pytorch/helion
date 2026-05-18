from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.cute.strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from helion._compiler.cute.strategies import Tcgen05LayoutOverrides
from helion._compiler.cute.strategies import Tcgen05LayoutStrategy
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.strategies import Tcgen05WarpSpec
from helion._compiler.cute.strategies import validate_tcgen05_strategy_invariants
from helion._compiler.cute.tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfMTIA
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
import helion.language as hl


@helion.kernel
def _matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def _split_k_offset_index_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Split-K reduction whose atomic_add index mixes an offset-constant
    (``tile_m.begin // block_m``) with a tile coord (``tile_n``).

    The non-``BlockSizeOrigin`` first index would short-circuit the
    prior cycle's gated ghost-axis predicate before the cycle that
    lifted the scan above the gate; the inner-K thread axis remains
    live in this scope and causes ``blockDim.z``-multiplier
    over-counting without the ghost-axis leader predicate.
    """
    m, k = x.size()
    _, n = y.size()
    block_m = hl.register_block_size(m)
    out = torch.zeros(
        [(m + 15) // 16, n],
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile(
        [m, n, k], block_size=[block_m, None, k_block]
    ):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        m_block_idx = tile_m.begin // block_m
        hl.atomic_add(out, [m_block_idx, tile_n], acc.sum(dim=0))
    return out


def _cute_two_matmuls_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    m2, k2 = x2.size()
    _, n2 = y2.size()
    out2 = torch.empty([m2, n2], dtype=x2.dtype, device=x2.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)

    for tile_m2, tile_n2 in hl.tile([m2, n2]):
        acc2 = hl.zeros([tile_m2, tile_n2], dtype=torch.float32)
        for tile_k2 in hl.tile(k2):
            acc2 = torch.addmm(
                acc2,
                x2[tile_m2, tile_k2],
                y2[tile_k2, tile_n2],
            )
        out2[tile_m2, tile_n2] = acc2.to(x2.dtype)
    return out, out2


_cute_two_matmuls_kernel = helion.kernel(_cute_two_matmuls_impl, backend="cute")
_cute_two_matmuls_force_persistent_kernel = helion.kernel(
    _cute_two_matmuls_impl,
    backend="cute",
    autotune_force_persistent=True,
)


@helion.kernel(backend="cute")
def _cute_strategy_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute", autotune_force_persistent=True)
def _cute_strategy_matmul_force_persistent_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _cute_4096_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain BF16 4096^3 cute matmul; shared by the SMEM-gate tests below."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(budget_bytes: int):
    """Bind the 4096^3 matmul with the per-CTA AB-SMEM budget mocked.

    The SMEM-budget gate is purely deterministic given a budget value
    (see ``ConfigSpec._cute_per_cta_ab_smem_budget_bytes``). Mocking
    that helper lets the demote/keep/seed unit tests exercise the gate
    on any device, not just hosts that report a B200-sized opt-in
    SMEM cap. ``budget_bytes`` is the per-CTA AB-SMEM budget in bytes
    the gate should treat as available for the AB pipeline staging.

    Clears ``_bound_kernels`` before binding so two tests in the same
    process that mock different budget values do not collide on the
    in-memory bind cache (the cache is keyed by args and would
    otherwise replay the first test's recorded spec).
    """
    args = (
        torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
    )
    _cute_4096_matmul_kernel._bound_kernels.clear()
    with (
        patch_cute_mma_support(),
        patch.object(
            ConfigSpec,
            "_cute_per_cta_ab_smem_budget_bytes",
            staticmethod(lambda device: budget_bytes),
        ),
    ):
        return _cute_4096_matmul_kernel.bind(args)


def _bind_cute_strategy_kernel():
    """Shared bind helper for the G2-A strategy data-model tests.

    The G2-A tests all need a cute_tcgen05-enabled ``config_spec`` with
    the cluster_m=2 search arm exposed (otherwise the cluster_m=2
    fixup / invariant tests below would not have a search arm to
    exercise); hoisting the bind avoids repeating the inline kernel
    definition in every test. The 256² shape would normally fall
    below the cycle-38 small-shape wave-quantization gate
    (cute_plan.md §7.6.3.2), so we mock ``_cuda_num_sms_or_zero``
    to return 0 — that fallback keeps cluster_m=2 search live for
    configuration round-trip tests without depending on the host
    GPU. Tests that intend to exercise the gate live in
    ``test_cute_tcgen05_small_shape_wave_quantization_gate*`` and
    bind their own kernel.

    For tests that exercise codegen (``to_triton_code()``), keep
    the ``patch_cute_mma_support`` context active across the
    codegen call — ``cute_mma.py`` consults
    ``get_cute_mma_support()`` during codegen, and a bare bind
    followed by a codegen call would silently hit the non-tcgen05
    fallback on a host without native tcgen05.
    """
    args = (
        torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
    )
    with (
        patch_cute_mma_support(),
        patch(
            "helion.language.matmul_ops._cuda_num_sms_or_zero",
            return_value=0,
        ),
    ):
        return _cute_strategy_matmul_kernel.bind(args)


@onlyBackends(["triton", "cute"])
class TestDotRequirements(RefEagerTestDisabled, TestCase):
    @patch.object(_compat, "_min_dot_size", lambda *args: (2, 8, 16))
    def test_hl_dot_sets_min_size(self) -> None:
        @helion.kernel
        def k_small(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc += hl.dot(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        m, k, n = 32, 4, 16
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        spec = k_small.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [2, 8, 16])

    @patch.object(_compat, "_min_dot_size", lambda *args: (2, 8, 16))
    def test_matmul_sets_min_size(self) -> None:
        m, k, n = 32, 4, 16
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        spec = _matmul_kernel.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [2, 8, 16])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_matmul_constrains_search_space(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [128, 8, 16])
        # tile_k upper bound was previously hardcoded to 16; the cute tcgen05
        # path now allows multiples of 16 up to min(128, static_k) so the
        # autotuner can pack more cute.gemm instructions per K iteration.
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 128, 64])
        default_block_sizes = spec.default_config().config["block_sizes"]
        self.assertGreaterEqual(default_block_sizes[2], 16)
        self.assertLessEqual(default_block_sizes[2], 64)
        self.assertGreaterEqual(default_block_sizes[0], 128)
        self.assertLessEqual(default_block_sizes[0], 256)
        self.assertGreaterEqual(default_block_sizes[1], 8)
        self.assertLessEqual(default_block_sizes[1], 128)
        self.assertEqual(spec.default_config().config["l2_groupings"], [1])
        # This small-N problem cannot form the validated 256x256 CtaGroup.TWO
        # tile, so the autotuner keeps cluster_m narrowed to 1.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_equal_dims_keep_default_within_max_bound(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [128, 8, 16])
        # tile_k upper bound is now 128 (the static_k=8192 case; capped at 128
        # to keep AB SMEM staging budget sane).
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 256, 128])
        default_block_sizes = spec.default_config().config["block_sizes"]
        self.assertGreaterEqual(default_block_sizes[2], 16)
        self.assertLessEqual(default_block_sizes[2], 128)
        self.assertGreaterEqual(default_block_sizes[0], 128)
        self.assertLessEqual(default_block_sizes[0], 256)
        self.assertGreaterEqual(default_block_sizes[1], 8)
        self.assertLessEqual(default_block_sizes[1], 256)
        self.assertEqual(spec.default_config().config["l2_groupings"], [1])
        # K=8192 can form validated CtaGroup.TWO products at bk >= 32 even
        # though bk=16 is over the K-tile cap. The search exposes cluster_m=2,
        # and normalization drops only the invalid per-bk products.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 1)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        over_cap_config = {
            "block_sizes": [256, 256, 16],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(over_cap_config, _fix_invalid=True)
        self.assertEqual(over_cap_config["tcgen05_cluster_m"], 1)
        self.assertEqual(over_cap_config["pid_type"], "flat")
        valid_config = {
            "block_sizes": [128, 256, 32],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(valid_config, _fix_invalid=True)
        self.assertEqual(valid_config["tcgen05_cluster_m"], 2)
        self.assertEqual(valid_config["pid_type"], "persistent_interleaved")
        self.assertEqual(valid_config["block_sizes"][:3], [256, 256, 32])
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_widened_default_stays_on_tcgen05_path(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([8192, 8192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
            config = bound.config_spec.default_config()
            code = bound.to_triton_code(config)
        self.assertEqual(config.config["block_sizes"][2], 16)
        self.assertGreaterEqual(config.config["block_sizes"][0], 128)
        self.assertLessEqual(config.config["block_sizes"][0], 256)
        self.assertGreaterEqual(config.config["block_sizes"][1], 8)
        self.assertLessEqual(config.config["block_sizes"][1], 256)
        self.assertIn("make_trivial_tiled_mma", code)
        self.assertIn(f"_BLOCK_SIZE_0 = {config.config['block_sizes'][0]}", code)
        self.assertIn(f"_BLOCK_SIZE_1 = {config.config['block_sizes'][1]}", code)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_enters_validated_search_space(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search_fragments["tcgen05_cluster_m"].choices, (1, 2))

        config = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["l2_groupings"], [1])

        for override in (
            {"pid_type": "flat"},
            {"block_sizes": [128, 256, 16]},
            {"block_sizes": [256, 128, 16]},
            {"l2_groupings": [16]},
            {"pid_type": "persistent_interleaved"},
        ):
            with self.subTest(override=override):
                config = {
                    "block_sizes": [256, 256, 16],
                    "l2_groupings": [1],
                    "pid_type": "persistent_blocked",
                    "tcgen05_cluster_m": 2,
                    **override,
                }
                spec.normalize(config, _fix_invalid=True)
                expected_l2_groupings = override.get("l2_groupings", [1])
                self.assertEqual(config["tcgen05_cluster_m"], 2)
                self.assertEqual(config["pid_type"], "persistent_interleaved")
                self.assertEqual(config["block_sizes"][:3], [256, 256, 16])
                self.assertEqual(config["l2_groupings"], expected_l2_groupings)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_small_shape_wave_quantization_gate(self) -> None:
        """Cycle 38 (cute_plan.md §7.6.3.2): the cluster_m=2 search arm
        is narrowed for shapes whose cluster_m=2 work-cluster count
        cannot saturate even one wave of cluster slots.

        The gate measures ``(M / 256) * (N / 256)`` cluster_m=2 work
        clusters and compares against ``num_sms // 2`` (one wave of
        cluster slots when each cluster occupies two SMs). With the
        SM count mocked to B200's 148 the threshold is 74 cluster
        slots; the §7.6.1.1 boost-target shapes 1024^3 and 2048^3
        sit at 16 and 64 cluster slots respectively and therefore
        narrow to ``cluster_m=1`` only. The 4096^3 G2 closure
        baseline sits at 256 cluster slots > 74 and keeps
        cluster_m=2 search exposed (positive control covered by
        ``test_cute_tcgen05_two_cta_enters_validated_search_space``).

        Mocking ``_cuda_num_sms_or_zero`` keeps the test hermetic:
        the gate logic is exercised on any host regardless of the
        live GPU's SM count.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        for size in (1024, 2048):
            with self.subTest(size=size):
                args = (
                    torch.empty([size, size], device=DEVICE, dtype=HALF_DTYPE),
                    torch.empty([size, size], device=DEVICE, dtype=HALF_DTYPE),
                )
                with (
                    patch_cute_mma_support(),
                    patch(
                        "helion.language.matmul_ops._cuda_num_sms_or_zero",
                        return_value=148,
                    ),
                ):
                    bound = cute_matmul_mma.bind(args)
                spec = bound.config_spec
                # Below the one-wave SM-slot threshold: cluster_m=2
                # search is suppressed and the cluster_m2 seed
                # / fixup machinery is disabled so the autotuner
                # never spends budget on the cluster_m=2 seed for a
                # shape where it has no productive lever.
                self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
                self.assertIsNone(spec._tcgen05_cluster_m2_search_constraints)
                # Keep this assertion scoped to the cluster_m=2 seed heuristic:
                # future unrelated heuristics may still apply to these shapes.
                self.assertNotIn(
                    CuteTcgen05ClusterM2Heuristic.name,
                    spec.autotuner_heuristics,
                )
                # Persistent pid types are still allowed (the static-
                # full-tile gate above this is unaffected) — only the
                # cluster_m search arm narrows.
                self.assertIn("persistent_interleaved", spec.allowed_pid_types)
                self.assertIn("persistent_blocked", spec.allowed_pid_types)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m1_persistent_search_caps_m_tile(self) -> None:
        """Search-only cluster_m=1 persistent configs stay on tcgen05 M tiles."""

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec

        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            with self.subTest(pid_type=pid_type):
                config = {
                    "block_sizes": [256, 32, 16],
                    "pid_type": pid_type,
                    "tcgen05_cluster_m": 1,
                }
                spec.normalize(config, _fix_invalid=True)
                self.assertEqual(config["tcgen05_cluster_m"], 1)
                self.assertEqual(config["pid_type"], pid_type)
                self.assertEqual(
                    config["block_sizes"][:3],
                    [TCGEN05_ONE_CTA_MAX_BLOCK_M, 32, 16],
                )

        flat_config = {
            "block_sizes": [256, 32, 16],
            "pid_type": "flat",
            "tcgen05_cluster_m": 1,
        }
        spec.normalize(flat_config, _fix_invalid=True)
        self.assertEqual(flat_config["block_sizes"][:3], [256, 32, 16])

        two_cta_config = {
            "block_sizes": [256, 32, 16],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(two_cta_config, _fix_invalid=True)
        self.assertEqual(two_cta_config["tcgen05_cluster_m"], 2)
        self.assertEqual(two_cta_config["pid_type"], "persistent_interleaved")
        self.assertEqual(two_cta_config["block_sizes"][:3], [256, 256, 16])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_two_cta_projection_falls_back_before_mutation(
        self,
    ) -> None:
        """Invalid cluster_m=2 search products fall back without pid churn."""

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec

        for block_sizes in (
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 8],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 24],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, True],
        ):
            with self.subTest(block_sizes=block_sizes):
                original_block_sizes = list(block_sizes)
                config = {
                    "block_sizes": block_sizes,
                    "l2_groupings": [1],
                    "pid_type": "flat",
                    "tcgen05_cluster_m": 2,
                }
                spec._fix_tcgen05_cluster_m2_search_config(config)
                self.assertEqual(config["tcgen05_cluster_m"], 1)
                self.assertEqual(config["pid_type"], "flat")
                self.assertEqual(config["block_sizes"], original_block_sizes)
                self.assertEqual(config["l2_groupings"], [1])

        original_allowed_pid_types = spec.allowed_pid_types
        try:
            spec.allowed_pid_types = ("flat",)
            config = {
                "block_sizes": [
                    TCGEN05_TWO_CTA_BLOCK_M,
                    TCGEN05_TWO_CTA_BLOCK_N,
                    16,
                ],
                "l2_groupings": [1],
                "pid_type": "flat",
                "tcgen05_cluster_m": 2,
            }
            spec._fix_tcgen05_cluster_m2_search_config(config)
        finally:
            spec.allowed_pid_types = original_allowed_pid_types
        self.assertEqual(config["tcgen05_cluster_m"], 1)
        self.assertEqual(config["pid_type"], "flat")
        self.assertEqual(
            config["block_sizes"],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 16],
        )
        self.assertEqual(config["l2_groupings"], [1])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_three_smem_budget_gate(self) -> None:
        """SMEM-budget gate admits ``tcgen05_ab_stages=3`` into search.

        The 4096^3 BF16 matmul binding admits ``ab=3`` into the autotune
        search arm via the SMEM-budget gate so the canonical fast config
        family (``cluster_m=2`` ``ab=3``) reaches the autotuner without a
        hand-forced override. Search-time normalization demotes ``ab=3``
        candidates whose ``(bm, bn, bk, cluster_m)`` per-CTA AB-SMEM cost
        exceeds the device's optin SMEM cap minus the non-AB reservation
        (see ``cute_plan.md`` §7.0). The validation surface stays
        unchanged: explicit ``helion.Config(tcgen05_ab_stages=3)`` always
        round-trips for explicit user configs.

        The gate is purely deterministic given a budget value, so we
        pin the per-CTA AB-SMEM budget to B200's nominal value via
        ``_bind_cute_4096_matmul_kernel_with_mocked_smem_budget`` —
        that keeps coverage live on any host regardless of the live
        GPU's optin SMEM cap.
        """
        # B200's optin reports 232 448 bytes = 227 KiB; subtract the
        # 28 KiB non-AB reservation to match what
        # ``_cute_per_cta_ab_smem_budget_bytes`` produces in production
        # (203 776 bytes). Tracking the production value exactly is
        # what makes the over-budget vs in-budget boundary in this
        # test mirror the running gate.
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec

        constraints = spec._tcgen05_ab_stages_three_search_constraints
        self.assertIsNotNone(constraints)
        # ``itemsize`` for BF16/FP16 is 2 bytes — matches the matmul
        # binding's ``lhs.dtype.itemsize`` argument.
        self.assertEqual(constraints.dtype_bytes, HALF_DTYPE.itemsize)
        self.assertEqual(constraints.per_cta_smem_budget_bytes, b200_budget_bytes)

        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        # Search surface lifts ab_stages cap from 2 to 3 once the gate
        # admits the arm. The validation surface is independently 3.
        self.assertEqual(search_fragments["tcgen05_ab_stages"].high, 3)
        validation_fragments = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation_fragments["tcgen05_ab_stages"].high, 3)

        # cluster_m=2 256x256x128 ab=3: the canonical 4096^3 fast config —
        # fits the per-CTA budget (196 608 bytes vs B200's 203 776-byte
        # budget). Search-time fixup keeps it.
        keep_config = {
            "block_sizes": [256, 256, 128],
            "l2_groupings": [4],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_ab_stages": 3,
        }
        spec.normalize(keep_config, _fix_invalid=True)
        self.assertEqual(keep_config["tcgen05_ab_stages"], 3)
        self.assertEqual(keep_config["tcgen05_cluster_m"], 2)

        # ab=3 over-budget shapes get demoted to ab=2. The cute_dsl
        # ptxas failure is the loud backstop for explicit user configs
        # that bypass autotune; the search-side fixup keeps the
        # autotuner from blowing the GPU context mid-tuning. The two
        # cases below exercise distinct post-fixup shapes:
        #   * persistent + cluster_m=1 with bm=128: 294 912 bytes — the
        #     ``_fix_tcgen05_cluster_m1_persistent_search_config`` path
        #     already clamps bm to ``TCGEN05_ONE_CTA_MAX_BLOCK_M``;
        #     bm=128 is at the cap, so it survives unchanged.
        #   * flat + cluster_m=1 with bm=256: 393 216 bytes — flat
        #     pid_type bypasses the cluster_m1 cap (the cap only
        #     applies under persistent pid_types) so the unmolested
        #     256x256x128 single-CTA path reaches the new fixup.
        over_budget_cases = (
            ("persistent_interleaved", [128, 256, 128]),  # 294 912 bytes
            ("flat", [256, 256, 128]),  # 393 216 bytes
        )
        for pid_type, over_budget_block_sizes in over_budget_cases:
            with self.subTest(pid_type=pid_type, block_sizes=over_budget_block_sizes):
                config = {
                    "block_sizes": list(over_budget_block_sizes),
                    "pid_type": pid_type,
                    "tcgen05_cluster_m": 1,
                    "tcgen05_ab_stages": 3,
                }
                spec.normalize(config, _fix_invalid=True)
                self.assertEqual(config["tcgen05_ab_stages"], 2)

        # cluster_m=1 ab=3 in-budget shape stays at ab=3.
        in_budget = {
            "block_sizes": [128, 128, 128],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_ab_stages": 3,
        }
        spec.normalize(in_budget, _fix_invalid=True)
        self.assertEqual(in_budget["tcgen05_ab_stages"], 3)

        # Validation surface always accepts ab=3 (no _fix_invalid).
        user_config = {
            "block_sizes": [128, 256, 128],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_ab_stages": 3,
        }
        spec.normalize(user_config)
        self.assertEqual(user_config["tcgen05_ab_stages"], 3)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_three_gate_off_below_b200(self) -> None:
        """Gate stays off when target device's SMEM optin is sub-B200.

        Mocking the budget helper to return 0 — the value the helper
        produces for non-CUDA hosts and any device whose optin cap sits
        below ``TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN`` — must
        keep ``_tcgen05_ab_stages_three_search_constraints`` ``None`` so
        the search surface stays at ``ab_stages_max=2`` and the
        canonical seed does not carry ``ab=3``. This guards against
        broadening the search past the hardware's known-good envelope
        on heterogeneous / multi-GPU setups.
        """
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(0)
        spec = bound.config_spec

        self.assertIsNone(spec._tcgen05_ab_stages_three_search_constraints)
        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search_fragments["tcgen05_ab_stages"].high, 2)
        # Validation surface stays at 3 so explicit user configs still
        # round-trip even on a device the gate is off for.
        validation_fragments = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation_fragments["tcgen05_ab_stages"].high, 3)
        # The cluster_m2 seed exists but does *not* carry ab=3.
        seeds = spec.compiler_seed_configs
        self.assertEqual(len(seeds), 1)
        self.assertNotIn("tcgen05_ab_stages", seeds[0].config)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_three_refused_for_multi_block_size_triple(
        self,
    ) -> None:
        """Gate refuses ``ab=3`` when the spec has more than one (M,N,K) triple.

        ``tcgen05_ab_stages`` is a global config knob across the whole
        bound kernel, but the per-config search-time fixup
        (``_fix_tcgen05_ab_stages_three_search_config``) only inspects
        ``block_sizes[0:3]``. A multi-dot or multi-root kernel with
        more than the single-matmul 3-block-size triple could otherwise
        sit at ``ab=3`` for a later over-budget triple — that path
        would survive the fixup and abort at ptxas mid-tuning. Verify
        ``allow_tcgen05_ab_stages_three_search`` clears the recorded
        constraints whenever the spec's block-size sequence is not the
        validated 3-tuple.
        """
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec

        # Sanity: the matmul-bound spec with a 3-block-size triple
        # already admits the arm (this is the standard production path).
        self.assertIsNotNone(spec._tcgen05_ab_stages_three_search_constraints)

        # Simulate a spec with more than the single-matmul 3-block-size
        # triple (e.g. a multi-dot or multi-root kernel) and re-invoke
        # the gate. The recorded constraints must be cleared so the
        # search surface stays at ``ab_stages_max=2`` — otherwise a
        # later over-budget triple would survive the per-config fixup
        # (which inspects only ``block_sizes[0:3]``) and abort at ptxas.
        # The gate's only check on ``block_sizes`` is its length, so
        # re-running ``allow_tcgen05_ab_stages_three_search`` against
        # a spec whose ``block_sizes`` reports a non-3 length is enough
        # to validate the refusal.
        with patch.object(type(spec.block_sizes), "__len__", lambda self: 9):
            self.assertEqual(len(spec.block_sizes), 9)
            with patch.object(
                ConfigSpec,
                "_cute_per_cta_ab_smem_budget_bytes",
                staticmethod(lambda device: b200_budget_bytes),
            ):
                spec.allow_tcgen05_ab_stages_three_search(
                    dtype_bytes=2,
                    device=torch.device("cuda:0"),
                )
            self.assertIsNone(spec._tcgen05_ab_stages_three_search_constraints)
            search_fragments = spec._tcgen05_optional_fragments(for_search=True)
            self.assertEqual(search_fragments["tcgen05_ab_stages"].high, 2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_three_seeded_in_initial_population(
        self,
    ) -> None:
        """Canonical ``ab=3`` fast config is in the initial autotune seed.

        Acceptance: when the SMEM gate admits ``ab=3`` for the canonical
        ``256x256x128 cluster_m=2`` shape, the cluster_m=2 seed config
        carries ``tcgen05_ab_stages=3`` so the autotuner's initial
        population includes the retained 4096^3 fast config family
        (``cute_plan.md`` §1.1). Without this seed the normal autotune
        would have to discover ``ab=3`` via random mutation, which is
        unreliable across short search budgets.

        Pins the per-CTA AB-SMEM budget to B200's nominal value so the
        seed-path coverage runs on any host (see
        ``test_cute_tcgen05_ab_stages_three_smem_budget_gate``).
        """
        # B200 production value: 227 KiB optin minus 28 KiB non-AB
        # reservation (see _cute_per_cta_ab_smem_budget_bytes).
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec

        seed_configs = spec.compiler_seed_configs
        self.assertEqual(len(seed_configs), 1)
        seed = seed_configs[0].config
        self.assertEqual(
            seed["block_sizes"][:3],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 128],
        )
        self.assertEqual(seed["tcgen05_cluster_m"], 2)
        self.assertEqual(seed["tcgen05_ab_stages"], 3)

    @onlyBackends(["cute"])
    def test_cute_universal_matmul_lane_loop_correctness(self) -> None:
        """Universal-MMA SMEM-load guards stay correct under a lane loop.

        Binds a CuTe matmul with a lane-loop configuration
        (``elements_per_thread=2``) on either the M or the N axis and
        asserts both the launch dim (recovery must divide by ``epT``)
        and ``allclose`` against ``x @ y`` (SMEM-load guards must use
        the physical thread coord so every lane iteration re-populates
        sA / sB). The fix is symmetric across M and N — see the
        ``_local_mma_coord_expr`` → ``_physical_mma_coord_expr``
        switch in ``cute_mma._codegen_cute_mma``.
        """
        torch.manual_seed(0)
        x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32)
        y = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32)
        # Both variants force the universal MMA path (fp32 inputs) with
        # a 2-element lane loop on the named axis. Expected launch dim
        # is ``block=(16, 16, 1)`` in both cases: the non-laned axis
        # carries its ``num_threads`` value directly, the laned axis
        # carries ``block_size // elements_per_thread``.
        cases = (
            (
                "n_axis_lane",
                helion.Config(block_sizes=[16, 32, 32], num_threads=[16, 16, 32]),
            ),
            (
                "m_axis_lane",
                helion.Config(block_sizes=[32, 16, 32], num_threads=[16, 16, 32]),
            ),
        )
        for case_name, config in cases:
            with self.subTest(case=case_name):
                # Fresh bind cache: the in-memory bind cache is keyed
                # by args and other subTest iterations populate it.
                _cute_strategy_matmul_kernel._bound_kernels.clear()
                # ``patch_cute_mma_support`` makes the lowering
                # decision deterministic across hosts — on a
                # tcgen05-capable host these shapes fall to universal
                # MMA via the precondition-check path anyway, but
                # wrapping matches the convention used by every other
                # ``_cute_strategy_matmul_kernel`` binding in this
                # class.
                with patch_cute_mma_support():
                    bound = _cute_strategy_matmul_kernel.bind((x, y))
                bound.set_config(config)
                result = bound(x, y)
                torch.testing.assert_close(result, x @ y, atol=1e-1, rtol=1e-2)

                code = bound.to_triton_code(config)
                for ln in code.splitlines():
                    if "_launcher(" in ln and "block=(" in ln:
                        self.assertIn("block=(16, 16, 1)", ln)
                        break
                else:
                    self.fail("could not locate launcher block=(...) in generated code")

    @onlyBackends(["cute"])
    def test_cute_inactive_grid_block_id_does_not_claim_thread_axis(self) -> None:
        """Grid codegen for an inactive block_id must skip its thread axis.

        Binds ``examples.matmul_split_k`` with a config that places the
        outer K block_id in ``inactive_block_ids`` (the K coordinate is
        only referenced through the inner device-loop's range bounds, so
        the static-analysis pass marks the outer block_id unused inside
        the graph). If the grid emits ``indices_<n> = tile_offset_<n> +
        thread_idx[axis]`` for an inactive block_id, the inner device
        loop's ``_compute_thread_axis_offset`` will reuse that axis (it
        counts only active axes) and produce a ``cudaErrorIllegalAddress``
        at runtime.
        """
        from helion._testing import EXAMPLES_DIR
        from helion._testing import import_path

        torch.manual_seed(0)
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, 64, device=DEVICE, dtype=torch.float32)

        mod = import_path(EXAMPLES_DIR / "matmul_split_k.py")
        config = helion.Config(
            block_sizes=[16, 2, 16],
            num_threads=[0, 2, 8],
            split_k=32,
        )
        # Force a fresh bind so other tests in this class do not poison
        # the in-memory bind cache.
        mod.matmul_split_k._bound_kernels.clear()
        bound = mod.matmul_split_k.bind((x, y))
        bound.set_config(config)

        code = bound.to_triton_code(config)
        # ``indices_2`` corresponds to the inactive outer-K block_id. It
        # must be plain ``tile_offset_2`` — no ``thread_idx`` term —
        # otherwise the launch dim is shared with the inner block_id and
        # the inner indices line addresses past the tile.
        for ln in code.splitlines():
            if ln.strip().startswith("indices_2 = "):
                self.assertNotIn("thread_idx", ln, msg=ln)
                self.assertIn("tile_offset_2", ln, msg=ln)
                break
        else:
            self.fail("could not locate indices_2 = ... in generated code")

        # Crash-survival regression check: the kernel must run without a
        # CUDA illegal memory access so the GPU context survives. This
        # test does NOT assert numerical correctness against
        # ``torch.matmul``: that is pinned separately by
        # ``test_cute_atomic_add_predicates_cta_resident_thread_axis``
        # below, which guards against atomic_add over-counting when the
        # inner-K loop's thread axis remains live in the surrounding
        # scope.
        bound(x, y)
        torch.cuda.synchronize()

    @onlyBackends(["cute"])
    def test_cute_atomic_add_predicates_cta_resident_thread_axis(self) -> None:
        """``hl.atomic_add`` outside an inner device loop must predicate
        on the loop's CTA-resident thread axis.

        ``examples.matmul_split_k`` issues ``hl.atomic_add(out, [tile_m,
        tile_n], acc)`` outside an inner ``for inner_k in hl.tile(...)``
        device loop. When the autotuner picks a config that maps the
        inner-K block_id onto a thread axis (here ``thread_idx[2]``),
        every axis-2 thread continues to execute the post-inner-loop
        code with the same broadcast reduction value. Without a
        ``thread_idx[axis] == 0`` predicate on the atomic, each output
        cell is accumulated ``blockDim.z`` times, producing a result
        that is ``blockDim.z``-x too large.
        """
        from helion._testing import EXAMPLES_DIR
        from helion._testing import import_path

        torch.manual_seed(0)
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, 64, device=DEVICE, dtype=torch.float32)
        expected = torch.matmul(x, y)

        mod = import_path(EXAMPLES_DIR / "matmul_split_k.py")
        config = helion.Config(
            block_sizes=[16, 2, 16],
            num_threads=[0, 2, 8],
            split_k=32,
        )
        mod.matmul_split_k._bound_kernels.clear()
        bound = mod.matmul_split_k.bind((x, y))
        bound.set_config(config)

        code = bound.to_triton_code(config)
        # The atomic_add must be guarded by a CTA-resident leader
        # predicate on axis 2 (the inner-K loop's thread axis). The
        # predicate is emitted on the surrounding ``if`` statement, not
        # on the atomic call itself.
        lines = code.splitlines()
        found = False
        for idx, ln in enumerate(lines):
            if "cute.arch.atomic_add" in ln:
                # Walk back through the enclosing context (a small fixed
                # window is enough; the predicate is the immediately
                # preceding ``if`` statement).
                for prior in reversed(lines[max(0, idx - 4) : idx]):
                    if prior.lstrip().startswith("if "):
                        self.assertIn(
                            "cute.arch.thread_idx()[2] == 0", prior, msg=prior
                        )
                        found = True
                        break
                self.assertTrue(found, msg=f"no enclosing if for: {ln}")
                break
        if not found:
            self.fail("could not locate cute.arch.atomic_add in generated code")

        out = bound(x, y)
        torch.cuda.synchronize()
        # fp32 split-K with 32-way K split + atomic-add over 1024 K
        # elements per output. Loose atol matches the existing
        # ``test_matmul_split_k`` accuracy bar in ``test_examples.py``.
        torch.testing.assert_close(out, expected, atol=1, rtol=0.01)

    @onlyBackends(["cute"])
    def test_cute_atomic_add_predicates_ghost_axis_for_offset_constant_index(
        self,
    ) -> None:
        """Ghost-axis predicate must fire even when the atomic index
        does not flow through a ``BlockSizeOrigin`` symbol.

        The fix is required for index forms beyond
        ``[tile_m, tile_n]`` — e.g. an offset-constant
        ``tile.begin // block_size`` paired with another tile coord.
        The prior cycle's predicate gated the ghost-axis scan behind
        ``has_block_size_index``; if every index were offset-constant
        the gate would return early and miss the ghost axis. This test
        uses a mixed index ``[m_block_idx, tile_n]`` so axis 1 still
        triggers the gate while axis 0 is offset-constant and axis 2
        is a ghost from the exited inner-K device loop. Pre-fix the
        old code only predicated axis 0 (non-indexed active block_m)
        and missed axis 2, producing an 8× over-count.
        """
        torch.manual_seed(0)
        m, k, n = 16, 1024, 64
        x = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.float32)
        expected = torch.matmul(x, y).sum(dim=0).unsqueeze(0)

        _split_k_offset_index_atomic._bound_kernels.clear()
        bound = _split_k_offset_index_atomic.bind((x, y))
        config = helion.Config(
            block_sizes=[16, 2, 16],
            num_threads=[0, 2, 8],
            split_k=32,
            indexing="block_ptr",
        )
        bound.set_config(config)

        code = bound.to_triton_code(config)
        # The ghost-axis predicate on axis 2 (inner-K loop's thread
        # axis after exit) must appear on the atomic_add's enclosing
        # ``if``.
        lines = code.splitlines()
        found_axis_2 = False
        for idx, ln in enumerate(lines):
            if "cute.arch.atomic_add" in ln:
                for prior in reversed(lines[max(0, idx - 4) : idx]):
                    if prior.lstrip().startswith("if "):
                        self.assertIn(
                            "cute.arch.thread_idx()[2] == 0", prior, msg=prior
                        )
                        found_axis_2 = True
                        break
                break
        self.assertTrue(found_axis_2, msg="ghost-axis predicate missing")

        out = bound(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=1, rtol=0.01)

    @skipIfMTIA("MTIA requires tl.dot initial value stride >= 128 bytes")
    def test_matmul_smaller_than_min_dot_size(self) -> None:
        """Test matmul where K and N are smaller than min_dot_size (16 on CUDA).

        If update_min_block() promotes block sizes beyond the tensor dimensions,
        this will fail with shape mismatches.
        """
        m, k, n = 32, 8, 8
        args = (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        _, result = code_and_output(_matmul_kernel, args, block_sizes=[32, 8, 8])
        ref = args[0].float() @ args[1].float()
        torch.testing.assert_close(result, ref, atol=1e-1, rtol=1e-2)

    @skipIfMTIA("MTIA backend does not support 3D dot reshape patterns")
    def test_bmm_constrains_batch_block_to_one(self) -> None:
        """Triton warp-spec only stably supports 2D tl.dot.
        For batched matmul (baddbmm/bmm), the batch dimension block size must
        be constrained to 1 so the codegen an squeeze the 3D operands to 2D
        before emitting tl.dot.

        Without this constraint the autotuner may pick batch block sizes > 1,
        producing a 3D tl.dot that crashes in Triton's LLVM backend with
        "Unsupported DotOp found when converting TritonGPU to LLVM".
        """

        @helion.kernel(static_shapes=True)
        def bmm_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            b, m, k = A.size()
            b, k, n = B.size()
            out = torch.empty(
                [b, m, n],
                device=A.device,
                dtype=torch.promote_types(A.dtype, B.dtype),
            )
            for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
                acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.baddbmm(
                        acc,
                        A[tile_b, tile_m, tile_k],
                        B[tile_b, tile_k, tile_n],
                    )
                out[tile_b, tile_m, tile_n] = acc
            return out

        b, m, k, n = 16, 512, 768, 1024
        args = (
            torch.randn([b, m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([b, k, n], device=DEVICE, dtype=HALF_DTYPE),
        )

        # Use the spec's batch max_size as block_sizes[0], combined with
        # autotuner parameters that trigger a Triton crash when batch > 1.
        # Without the fix, max_size = 16 (full batch dim) and the 3D tl.dot
        # hits "Unsupported DotOp" → RuntimeError: PassManager::run failed.
        # With the fix, max_size = 1 and the codegen squeezes to a 2D tl.dot.
        bound = bmm_kernel.bind(args)
        batch_max = bound.config_spec.block_sizes[0].max_size
        code, result = code_and_output(
            bmm_kernel,
            args,
            block_sizes=[batch_max, 1, 128, 16],
            indexing=["pointer", "pointer", "tensor_descriptor"],
            num_warps=2,
            num_stages=5,
            pid_type="flat",
        )
        expected = torch.bmm(args[0], args[1])
        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_validated_autotune_narrowing(self) -> None:
        """``narrow_tcgen05_autotune_to_validated_configs`` consolidates the
        tcgen05 limitations into a single config_spec call.

        Pin the resulting state so any future change to the helper has to
        update the test as well: persistent pid types stay in the autotune
        search for validated static full-tile shapes, the cluster_m search
        stays narrowed to ``(1,)`` when the problem cannot form the validated
        256x256 CtaGroup.TWO tile, and the num_epi_warps search is narrowed
        to ``(4,)``.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        # Every candidate M/N/K block size divides this static problem, so
        # role-local persistent pid types are admitted back into autotune.
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        # This N=128 problem cannot form a validated 256x256 CtaGroup.TWO
        # tile, so the autotune search stays narrowed to cluster_m=1.
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        # num_epi_warps != 4 currently produces wrong output on B200
        # (only 4 epi warps lowers correctly today). The autotune search
        # is narrowed to num_epi_warps=4 so the autotuner does not
        # converge on a wrong-output config.
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        # The validated narrowing leaves cluster_m=2 still accepted as a
        # legal value for an explicit user-supplied helion.Config
        # (CUDA-launch-failure is loud and won't silently miscompute).
        validation_fragments = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation_fragments["tcgen05_cluster_m"].choices, (1, 2))
        # num_epi_warps is the exception: validation is also tightened
        # to (4,) because non-4 values silently produce wrong output, so
        # an explicit user-supplied helion.Config must be rejected
        # rather than allowed to miscompute.
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        self.assertEqual(validation_fragments["tcgen05_num_epi_warps"].choices, (4,))
        # The search view exposes the same narrowed EnumFragment.
        search_fragments = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search_fragments["tcgen05_num_epi_warps"].choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_partial_tile_search_keeps_persistent_pid_types_out(
        self,
    ) -> None:
        """Autotune excludes persistent pid types when the search can sample
        block sizes that produce partial tiles."""

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 192], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 128, 64])
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_search_keeps_persistent_pid_types_out(
        self,
    ) -> None:
        """Multi-root tcgen05 kernels keep persistent pid types out of autotune
        until the persistent scheduler/grid spans every root case."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_two_matmuls_kernel.bind(args)
        spec = bound.config_spec
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_forced_persistent_raises_invalid_config(
        self,
    ) -> None:
        """Forced-persistent multi-root tcgen05 has no valid pid search choice."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with (
            patch_cute_mma_support(),
            self.assertRaisesRegex(
                InvalidConfig,
                "CuTe tcgen05 multi-root kernels do not support persistent pid types",
            ),
        ):
            _cute_two_matmuls_force_persistent_kernel.bind(args)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_distributed_raises_invalid_config(
        self,
    ) -> None:
        """Distributed mode also makes the pid search persistent-only."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with (
            patch_cute_mma_support(),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.distributed_c10d.get_rank", return_value=0),
            patch("torch.distributed.distributed_c10d.get_world_size", return_value=1),
            patch("torch._logging._internal.dist.get_rank", return_value=0),
            patch(
                "torch.fx.experimental.symbolic_shapes.trace_structured",
                lambda *args, **kwargs: None,
            ),
            patch(
                "helion.runtime.kernel._find_process_group_name",
                return_value="world",
            ),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            self.assertRaisesRegex(
                InvalidConfig,
                "CuTe tcgen05 multi-root kernels do not support persistent pid types",
            ),
        ):
            _cute_two_matmuls_kernel.bind(args)

    def test_narrow_tcgen05_autotune_to_validated_configs_helper(self) -> None:
        """Direct unit test for the narrowing helper that does not depend
        on the dot-requirements bind path. The helper only manipulates the
        autotune search state on the receiver and is safe to invoke on any
        ``ConfigSpec`` instance."""

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        before_pid = set(spec.allowed_pid_types)
        spec.narrow_tcgen05_autotune_to_validated_configs()
        # Both persistent types are dropped (idempotently if they were
        # already absent).
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertNotIn("persistent_interleaved", spec.allowed_pid_types)
        # Other pid types are preserved.
        for pid_type in before_pid - {"persistent_blocked", "persistent_interleaved"}:
            self.assertIn(pid_type, spec.allowed_pid_types)
        # The cluster_m search is narrowed to (1,) unless the matmul caller
        # proves it can form validated CtaGroup.TWO search candidates.
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        # The num_epi_warps search is now narrowed to (4,) -- the only
        # currently-correct value on B200 (1 and 2 are directly verified
        # to produce wrong output, 3 is unsafe by extension).
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        # Validation is also tightened for num_epi_warps because the
        # failure mode is silent wrong output.
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        # Calling it twice is idempotent.
        spec.narrow_tcgen05_autotune_to_validated_configs()
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))

        spec = stub.bind(args).config_spec
        spec.allowed_pid_types = (
            "flat",
            "xyz",
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec.narrow_tcgen05_autotune_to_validated_configs(
            allow_persistent_pid_types=True
        )
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))

        spec = stub.bind(args).config_spec
        spec.allowed_pid_types = (
            "flat",
            "xyz",
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec.narrow_tcgen05_autotune_to_validated_configs(
            allow_persistent_pid_types=True,
            allow_cluster_m2_search=True,
            cluster_m2_static_k=4096,
        )
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))

    def test_restrict_tcgen05_num_epi_warps_search_helper(self) -> None:
        """Direct unit test for ``restrict_tcgen05_num_epi_warps_search``.

        The helper sets the per-instance search-only override and never
        affects the validation view returned by
        ``_tcgen05_optional_fragments(for_search=False)``. The test
        exercises the override on its own (i.e. without going through
        the full ``narrow_tcgen05_autotune_to_validated_configs``
        consolidation) so any future regression to the helper itself is
        caught here directly.
        """

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        # Default state: no override is set, so the search uses the
        # default IntegerFragment range and the validation view keeps
        # the same range.
        self.assertIsNone(spec._tcgen05_num_epi_warps_search_choices)
        default_search = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(default_search["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(default_search["tcgen05_num_epi_warps"].high, 4)

        spec.restrict_tcgen05_num_epi_warps_search((1, 2))
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (1, 2))
        narrowed_search = spec._tcgen05_optional_fragments(for_search=True)
        # Narrowing flips the search view to an EnumFragment so the
        # autotuner samples only the listed values.
        self.assertEqual(narrowed_search["tcgen05_num_epi_warps"].choices, (1, 2))
        # Validation view is unaffected by the search-only helper:
        # user-supplied helion.Config values in [1, 4] still round-trip
        # through normalize() unless ``restrict_tcgen05_num_epi_warps_validation``
        # is also called (see ``test_restrict_tcgen05_num_epi_warps_validation_helper``).
        validation = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(validation["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(validation["tcgen05_num_epi_warps"].high, 4)

        # Empty override raises (a misuse: every search must allow at
        # least one value).
        with self.assertRaises(AssertionError):
            spec.restrict_tcgen05_num_epi_warps_search(())

    def test_restrict_tcgen05_num_epi_warps_validation_helper(self) -> None:
        """Direct unit test for ``restrict_tcgen05_num_epi_warps_validation``.

        Unlike the search-only sibling, this helper tightens what
        ``normalize()`` accepts so user-supplied configs with bad
        values are rejected with ``InvalidConfig`` rather than silently
        accepted. Used by the BF16/FP16 matmul path because non-4
        epi-warp counts produce silent wrong output.
        """

        @helion.kernel
        def stub(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        args = (torch.randn([1024], device=DEVICE),)
        spec = stub.bind(args).config_spec
        # Default state: validation view is the full IntegerFragment.
        self.assertIsNone(spec._tcgen05_num_epi_warps_validation_choices)
        default_validation = spec._tcgen05_optional_fragments(for_search=False)
        self.assertEqual(default_validation["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(default_validation["tcgen05_num_epi_warps"].high, 4)

        spec.restrict_tcgen05_num_epi_warps_validation((4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        narrowed_validation = spec._tcgen05_optional_fragments(for_search=False)
        # Validation view flipped to EnumFragment with the restricted choices.
        self.assertEqual(narrowed_validation["tcgen05_num_epi_warps"].choices, (4,))
        # Search view unaffected by the validation-only helper.
        search = spec._tcgen05_optional_fragments(for_search=True)
        self.assertEqual(search["tcgen05_num_epi_warps"].low, 1)
        self.assertEqual(search["tcgen05_num_epi_warps"].high, 4)

        # Empty override raises.
        with self.assertRaises(AssertionError):
            spec.restrict_tcgen05_num_epi_warps_validation(())

    @onlyBackends(["cute"])
    def test_cute_tcgen05_num_epi_warps_search_routes_through_flat_fields(
        self,
    ) -> None:
        """End-to-end check that the narrowed num_epi_warps search shows
        up in ``_flat_fields()`` (the autotuner's single source of truth
        for the search space). Without this routing, the narrow_helper
        would only flip the per-instance flag while the autotuner kept
        sampling the full IntegerFragment range.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        # cute_tcgen05_search_enabled gates the inclusion of the tcgen05
        # optional fragments in _flat_fields(); enforce_dot_requirements
        # set it during bind, so the narrowed search view should appear.
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        flat_fields = spec._flat_fields()
        self.assertIn("tcgen05_num_epi_warps", flat_fields)
        # The matmul-side narrowing collapses the search to (4,);
        # _flat_fields exposes that as an EnumFragment with a single
        # choice rather than the default IntegerFragment(1, 4, 4).
        self.assertEqual(flat_fields["tcgen05_num_epi_warps"].choices, (4,))
        # This small-N problem cannot form the validated 256x256
        # CtaGroup.TWO tile, so cluster_m is narrowed to 1.
        self.assertEqual(flat_fields["tcgen05_cluster_m"].choices, (1,))
        self.assertIn("persistent_blocked", flat_fields["pid_type"].choices)
        self.assertIn("persistent_interleaved", flat_fields["pid_type"].choices)
        self.assertNotIn("num_threads", flat_fields)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_user_config_num_epi_warps_validation(self) -> None:
        """A user-supplied ``helion.Config(..., tcgen05_num_epi_warps=N)``
        must be rejected by ``normalize()`` for any N != 4 once the
        matmul path has narrowed the validation accept-set to ``(4,)``.
        ``num_epi_warps != 4`` produces silent wrong output today, so
        accepting an explicit user value would silently miscompute —
        the validation tightening is the only loud signal for a user
        bypassing autotune. The legal value 4 must still round-trip.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        # Both the search and validation accept-sets are narrowed to (4,).
        self.assertEqual(spec._tcgen05_num_epi_warps_search_choices, (4,))
        self.assertEqual(spec._tcgen05_num_epi_warps_validation_choices, (4,))
        # Non-4 values are rejected: silent wrong output on the
        # current SIMT-store epilogue.
        for n_epi in (1, 2, 3):
            cfg = helion.Config(
                block_sizes=[128, 16, 16],
                tcgen05_num_epi_warps=n_epi,
            )
            with self.assertRaises(InvalidConfig):
                spec.normalize(cfg)
        # The validated value still round-trips unchanged.
        cfg = helion.Config(
            block_sizes=[128, 16, 16],
            tcgen05_num_epi_warps=4,
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_num_epi_warps"], 4)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_minimize_normalize_round_trip(self) -> None:
        """The autotuner minimizes the winning config by stripping values
        that match ``default_config()`` (built from the *search* view),
        and the cached/minimized config is later re-expanded by
        ``normalize()``. If the fill-missing branch in normalize() used
        the validation-view default instead of the search-view default,
        the narrowed ``tcgen05_num_epi_warps=4`` choice would silently
        round-trip back to ``4`` only by accident (the validation
        IntegerFragment default also happens to be 4 today). Pin the
        search-view default routing so that, when the search view's
        default later diverges from the validation-view default again
        (e.g. when item 2 lifts the narrowing back to a smaller value),
        normalize() picks up the search-view default instead of the
        validation default.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        # The narrowed search default is what default_config() exposes.
        default_cfg = spec.default_config()
        self.assertEqual(default_cfg.config["tcgen05_num_epi_warps"], 4)
        # Simulate the autotuner's minimize step: a winning config of 4
        # matches the search-view default and gets stripped.
        winning = helion.Config(**default_cfg.config)
        minimized = winning.minimize(spec)
        self.assertNotIn("tcgen05_num_epi_warps", minimized.config)
        # Re-normalizing the minimized config (what happens on the next
        # to_code() call after a cache reload) must restore the same
        # effective value via the search-view fill-missing branch.
        spec.normalize(minimized)
        self.assertEqual(minimized.config["tcgen05_num_epi_warps"], 4)
        # Now simulate a future state where the search-view default
        # diverges from the validation-view default. Restrict the
        # search to (2,) (interior of the validation range) and confirm
        # that the fill-missing branch picks up the search-view default
        # of 2 rather than the validation-view default of 4. To do
        # this we must also lift the validation narrowing so that 2 is
        # a legal user-supplied value (otherwise constructing the
        # ``helion.Config(tcgen05_num_epi_warps=2)`` below would be
        # rejected by ``normalize``'s validation pass).
        spec._tcgen05_num_epi_warps_validation_choices = None
        spec.restrict_tcgen05_num_epi_warps_search((2,))
        new_default = spec.default_config()
        self.assertEqual(new_default.config["tcgen05_num_epi_warps"], 2)
        winning_2 = helion.Config(**new_default.config)
        minimized_2 = winning_2.minimize(spec)
        self.assertNotIn("tcgen05_num_epi_warps", minimized_2.config)
        spec.normalize(minimized_2)
        self.assertEqual(minimized_2.config["tcgen05_num_epi_warps"], 2)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_data_model_round_trip(self) -> None:
        """G2-A: ``Tcgen05Strategy`` / ``Tcgen05PersistenceModel`` /
        ``Tcgen05LayoutStrategy`` / ``Tcgen05WarpSpec`` /
        ``Tcgen05LayoutOverrides`` are wired through ``ConfigSpec`` so
        that ``helion.Config(...)`` round-trips them and
        ``default_config()`` exposes the documented defaults
        (``ROLE_LOCAL_MONOLITHIC`` strategy with the pinned 6-warp
        spec; ``epi_warps`` lives in the existing
        ``tcgen05_num_epi_warps`` field).
        """

        spec = _bind_cute_strategy_kernel().config_spec

        # Defaults match the documented G2-A pin: ROLE_LOCAL_MONOLITHIC
        # strategy with the existing 6-warp role-local spec. Persistence
        # model is derived from the active pid_type ("flat" -> non-
        # persistent) so serialized configs cannot encode contradictions.
        default_cfg = spec.default_config()
        self.assertEqual(
            default_cfg.config["tcgen05_strategy"], "role_local_monolithic"
        )
        self.assertEqual(default_cfg.config["pid_type"], "flat")
        self.assertEqual(
            default_cfg.config["tcgen05_persistence_model"], "non_persistent"
        )
        self.assertEqual(default_cfg.config["tcgen05_layout_strategy"], "default")
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_ab_load_warps"], 1)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_mma_warps"], 1)
        # ``epi_warps`` is the existing tcgen05_num_epi_warps knob.
        self.assertEqual(default_cfg.config["tcgen05_num_epi_warps"], 4)
        self.assertNotIn("tcgen05_warp_spec_epi_warps", default_cfg.config)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_epi_load_warps"], 0)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_scheduler_warps"], 0)
        # ``c_input_warps`` is the dedicated C-input / auxiliary-tensor
        # warp slot (``cute_plan.md`` §7.5.3.2). Default is 0 so
        # serialized configs round-trip cleanly; the validator widens
        # the accept set to ``{0, 1}`` under ``ROLE_LOCAL_WITH_SCHEDULER``
        # (inert-body slot) and stays at ``{0}`` under
        # ``ROLE_LOCAL_MONOLITHIC``. The productive TMA producer body
        # is a follow-up.
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_c_input_warps"], 0)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_register_decrease"], 120)
        self.assertEqual(default_cfg.config["tcgen05_warp_spec_register_increase"], 256)
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_smem_swizzle_a",
            "tcgen05_layout_overrides_smem_swizzle_b",
            "tcgen05_layout_overrides_d_store_box_n",
        ):
            self.assertIsNone(default_cfg.config[key])

        # JSON round-trip preserves every strategy field exactly.
        replayed = helion.Config.from_json(default_cfg.to_json())
        self.assertEqual(replayed, default_cfg)

        # An explicit user-supplied config round-trips through
        # normalize. Use persistent pid_type so the explicit
        # ``static_persistent`` agrees.
        cfg = helion.Config(
            block_sizes=[256, 256, 16],
            l2_groupings=[1],
            pid_type="persistent_blocked",
            tcgen05_cluster_m=2,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_monolithic",
            tcgen05_persistence_model="static_persistent",
            tcgen05_layout_strategy="default",
            tcgen05_warp_spec_ab_load_warps=1,
            tcgen05_warp_spec_mma_warps=1,
            tcgen05_warp_spec_epi_load_warps=0,
            tcgen05_warp_spec_scheduler_warps=0,
            tcgen05_warp_spec_register_decrease=120,
            tcgen05_warp_spec_register_increase=256,
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(cfg.config["tcgen05_persistence_model"], "static_persistent")
        self.assertEqual(cfg.config["tcgen05_num_epi_warps"], 4)
        self.assertEqual(cfg.config["tcgen05_warp_spec_register_decrease"], 120)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_reject_illegal(self) -> None:
        """G2-A: validation rejects illegal combinations.

        - ``tcgen05_strategy`` and ``tcgen05_layout_strategy`` are
          narrowed at the autotune fragment to the implemented set so
          unimplemented strategies are loudly rejected at the user
          surface (matches ``restrict_tcgen05_num_epi_warps_*``).
        - ``tcgen05_warp_spec_*`` knobs are narrowed similarly until
          G2-B/C reads them.
        - The cross-fragment validator catches strategy-conditional
          violations that span multiple fragments — exercised
          directly in
          ``test_cute_tcgen05_strategy_invariants_helper_unit``
          for the strategies the autotune fragment narrowing makes
          unreachable from the user surface today.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        base = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
        }

        # ``ROLE_LOCAL_WITH_SCHEDULER`` is now an implemented
        # strategy; explicit user configs that select it must
        # *also* set ``scheduler_warps=1`` to satisfy the
        # cross-fragment invariant.
        with self.assertRaises(InvalidConfig):
            # WITH_SCHEDULER + scheduler_warps=0 (the default) is
            # rejected by the cross-fragment validator.
            spec.normalize(
                helion.Config(**base, tcgen05_strategy="role_local_with_scheduler")
            )
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(**base, tcgen05_layout_strategy="explicit_epi_tile")
            )
        with self.assertRaises(InvalidConfig):
            # MONOLITHIC + scheduler_warps=1 is rejected: MONOLITHIC
            # requires scheduler_warps=0.
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_scheduler_warps=1))
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_ab_load_warps=2))
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_warp_spec_mma_warps=2))

        # ``WITH_SCHEDULER`` + ``cluster_m=2`` is accepted. Each
        # CTA in the cluster runs its own scheduler that publishes
        # locally and consumers release locally; both CTAs converge
        # on the same cluster-level virtual_pid via the
        # ``// cluster_m`` collapse in the consumer. See
        # ``cute_mma._codegen_cute_mma`` ``consumer_mask_to_leader``
        # comment for the full topology.
        with_scheduler_cluster_m2 = helion.Config(
            **base,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
        )
        spec.normalize(with_scheduler_cluster_m2)
        self.assertEqual(
            with_scheduler_cluster_m2.config["tcgen05_strategy"],
            "role_local_with_scheduler",
        )
        self.assertEqual(with_scheduler_cluster_m2.config["tcgen05_cluster_m"], 2)

        # WITH_SCHEDULER + scheduler_warps=1 + cluster_m=1 is also
        # valid and round-trips cleanly.
        cluster_m1_base = {
            **base,
            "tcgen05_cluster_m": 1,
        }
        with_scheduler_cfg = helion.Config(
            **cluster_m1_base,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
        )
        spec.normalize(with_scheduler_cfg)
        self.assertEqual(
            with_scheduler_cfg.config["tcgen05_strategy"],
            "role_local_with_scheduler",
        )
        self.assertEqual(
            with_scheduler_cfg.config["tcgen05_warp_spec_scheduler_warps"], 1
        )
        self.assertEqual(with_scheduler_cfg.config["tcgen05_cluster_m"], 1)

        # ``DYNAMIC_PERSISTENT`` is not in the persistence-model
        # fragment surface today (no codegen supports it).
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(**base, tcgen05_persistence_model="dynamic_persistent")
            )

        # ``epi_warps != 4`` -> rejected via ``tcgen05_num_epi_warps``
        # validation (single source of truth).
        with self.assertRaises(InvalidConfig):
            spec.normalize(helion.Config(**base, tcgen05_num_epi_warps=2))

        # Persistence model must agree with pid_type. The explicit
        # ``static_persistent`` contradicts ``pid_type=flat``.
        flat_base = {**base, "pid_type": "flat", "tcgen05_cluster_m": 1}
        with self.assertRaises(InvalidConfig) as ctx:
            spec.normalize(
                helion.Config(
                    **flat_base, tcgen05_persistence_model="static_persistent"
                )
            )
        self.assertIn("contradicts pid_type", str(ctx.exception))

        # Layout overrides with a concrete value under DEFAULT layout
        # strategy must be rejected — the override would be silently
        # ignored otherwise.
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(
                    **base,
                    tcgen05_layout_strategy="default",
                    tcgen05_layout_overrides_epi_tile_m=64,
                )
            )

        # The pinned ROLE_LOCAL_MONOLITHIC config still normalizes
        # cleanly so the rejection paths are not over-broad.
        cfg = helion.Config(
            **base,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_monolithic",
        )
        spec.normalize(cfg)
        self.assertEqual(cfg.config["tcgen05_strategy"], "role_local_monolithic")

        # G3.1 first slice (``cute_plan.md`` §7.5.3.2, cycle 34):
        # ``tcgen05_warp_spec_c_input_warps=1`` under WITH_SCHEDULER
        # round-trips end-to-end. The validator's accept set now
        # admits the value; the codegen body stays inert until the
        # productive TMA producer body lands.
        c_input_cfg = helion.Config(
            **base,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_warp_spec_c_input_warps=1,
        )
        spec.normalize(c_input_cfg)
        self.assertEqual(
            c_input_cfg.config["tcgen05_strategy"], "role_local_with_scheduler"
        )
        self.assertEqual(c_input_cfg.config["tcgen05_warp_spec_c_input_warps"], 1)

        # MONOLITHIC + c_input_warps=1 is still rejected (no slot in
        # the 6-warp shape for an 8th role warp). Pin the negative
        # path so the per-strategy gate cannot drift.
        with self.assertRaises(InvalidConfig):
            spec.normalize(
                helion.Config(
                    **base,
                    tcgen05_strategy="role_local_monolithic",
                    tcgen05_warp_spec_c_input_warps=1,
                )
            )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_helper_unit(self) -> None:
        """``validate_tcgen05_strategy_invariants`` covers the
        cross-fragment cases the autotune narrowing makes unreachable
        from the user surface today (persistence model not supported
        by the chosen strategy, scheduler_warps mismatching the
        strategy) plus the positive case where ``EXPLICIT_EPI_TILE``
        accepts non-None layout overrides.

        The earlier warpgroup-alignment requirement on
        ``ROLE_LOCAL_WITH_SCHEDULER`` was relaxed once the initial
        7-warp implementation landed (1 ab_load + 1 mma + 4 epi + 1
        scheduler = 7). Cycle 34's c_input lift makes the 8-warp
        variant reachable end-to-end (8 role warps exactly match
        the launched envelope, no padding); the alignment branch
        stays dead-code-tested via patching since neither variant
        triggers it organically today.
        """
        # scheduler_warps=0 under WITH_SCHEDULER is rejected (the
        # strategy demands one scheduler warp).
        wrong_scheduler_count = Tcgen05WarpSpec(
            ab_load_warps=1,
            mma_warps=1,
            epi_warps=4,
            epi_load_warps=0,
            scheduler_warps=0,
            register_split=(120, 256),
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=wrong_scheduler_count,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertTrue(any("scheduler_warps=1" in e for e in errors))

        # DYNAMIC_PERSISTENT under a strategy that does not support it.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.DYNAMIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertTrue(any("dynamic_persistent" in e for e in errors))

        # ``ROLE_LOCAL_WITH_SCHEDULER`` runs at cluster_m ∈ {1, 2}.
        # cluster_m=3+ falls outside the supported set; the
        # validator must reject so a user config can't reach an
        # untested cluster shape.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=dataclasses.replace(
                ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
            ),
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=4,
        )
        self.assertTrue(
            any("tcgen05_cluster_m=4" in e for e in errors), msg=str(errors)
        )

        # Positive control: ROLE_LOCAL_WITH_SCHEDULER + cluster_m=2
        # is now accepted (the per-CTA scheduler-warp topology is
        # cluster-correct).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=dataclasses.replace(
                ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
            ),
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Positive case: EXPLICIT_EPI_TILE + non-None overrides is
        # accepted — the validator must not drift into rejecting all
        # override values regardless of layout strategy.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(epi_tile_m=64, epi_tile_n=32),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [])

        # Negative control: clean ROLE_LOCAL_MONOLITHIC default is
        # always accepted.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_cluster_n(self) -> None:
        """G2 cluster_n=2 validator coverage (cute_plan.md §6.12.7,
        cycle 33 widening for ``ROLE_LOCAL_WITH_SCHEDULER``).

        ``cluster_n=2`` requires the 4-CTA V=2 cluster (``cluster_m=2``
        with ``use_2cta=True``). The validator now accepts cluster_n=2
        under both ``ROLE_LOCAL_MONOLITHIC`` and
        ``ROLE_LOCAL_WITH_SCHEDULER`` (cycle 33 lifted the
        WITH_SCHEDULER restriction so the cluster_n=2 lever exposes
        the G3.1-C step-2 productive C-input warp opportunity); it
        still rejects:
          - ``cluster_n=2`` with ``cluster_m=1`` (V=1 has no 4-CTA path)
        """
        # Positive control: cluster_n=2 + ROLE_LOCAL_MONOLITHIC + cluster_m=2.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # cluster_n=2 with cluster_m=1: rejected (requires the 4-CTA
        # V=2 cluster).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
            cluster_n=2,
        )
        self.assertTrue(
            any("requires tcgen05_cluster_m=2" in e for e in errors),
            msg=str(errors),
        )

        # cluster_n=2 under ROLE_LOCAL_WITH_SCHEDULER: ACCEPTED
        # in cycle 33 (the scheduler-broadcast topology generalizes
        # to cluster_n=2 with the per-CTA-local pattern preserved
        # and the cluster envelope ``cluster_m * cluster_n`` wired
        # through the deferred-init protocol).
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # cluster_n=2 with cluster_m=1 still rejected under
        # ROLE_LOCAL_WITH_SCHEDULER (V=1 has no 4-CTA path).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
            cluster_n=2,
        )
        self.assertTrue(
            any("requires tcgen05_cluster_m=2" in e for e in errors),
            msg=str(errors),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_c_input_warps(self) -> None:
        """G3.1 first slice (cute_plan.md §7.5.3.2) data-model lift:
        ``c_input_warps`` is plumbed through the dataclass + validator,
        and cycle 34 widens the ``ROLE_LOCAL_WITH_SCHEDULER`` accept
        set to ``{0, 1}`` so explicit user configs can opt in to the
        productive C-input warp slot. The codegen body remains inert
        in cycle 34; the productive TMA producer body lands in a
        follow-up cycle.

        - Positive control: ``c_input_warps=0`` accepted under both
          ``ROLE_LOCAL_MONOLITHIC`` and ``ROLE_LOCAL_WITH_SCHEDULER``
          (the field is plumbed through normalize / round-trip and
          defaults to 0 for legacy configs).
        - Positive control (cycle 34): ``c_input_warps=1`` accepted
          under ``ROLE_LOCAL_WITH_SCHEDULER`` — the slot occupies
          what was previously the inert padding warp.
        - Negative control: ``c_input_warps=1`` rejected under
          ``ROLE_LOCAL_MONOLITHIC`` (the 6-warp shape has no slot
          for an 8th role warp).
        """
        # Positive control: c_input_warps=0 under MONOLITHIC.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Positive control: c_input_warps=0 under WITH_SCHEDULER.
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Negative control: c_input_warps=1 under MONOLITHIC.
        c_input_monolithic = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, c_input_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=c_input_monolithic,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertTrue(
            any("c_input_warps in [0]" in e for e in errors),
            msg=str(errors),
        )

        # Positive control: c_input_warps=1 accepted under
        # WITH_SCHEDULER. The slot is reachable end-to-end and the
        # launched-warp accounting recognizes it (see the matching
        # matmul-plan accounting test below); the codegen body
        # stays inert until the productive TMA producer body lands.
        c_input_with_sched = dataclasses.replace(with_sched, c_input_warps=1)
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=c_input_with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # The dataclass total_warps reflects the c_input_warps slot:
        # 4 epi + 1 mma + 1 ab_load + 1 sched + 1 c_input = 8.
        self.assertEqual(c_input_with_sched.total_warps, 8)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_matmul_plan_c_input_warp_accounting(self) -> None:
        """``CuteTcgen05MatmulPlan`` carries ``c_input_warp_count``
        and the launched-warp accounting is invariant under the
        c_input lift because the slot occupies what was previously
        the inert padding warp under ``ROLE_LOCAL_WITH_SCHEDULER``
        (``cute_plan.md`` §7.5.3.2):

        - ``c_input_warp_count=0``: 7 role warps, 8 launched (1 pad).
        - ``c_input_warp_count=1``: 8 role warps, 8 launched (0 pad).

        Existing role warp ids (``exec_warp_id``, ``tma_warp_id``,
        ``scheduler_warp_id``) are unaffected by the lift — codegen
        sites that gate on those ids stay byte-identical.
        """
        from helion._compiler.device_function import CuteTcgen05MatmulPlan

        base_kwargs: dict[str, object] = {
            "bm": 256,
            "bn": 256,
            "bk": 128,
            "k_tile_count": 32,
            "cluster_m": 1,
            "is_two_cta": False,
            "uses_role_local_persistent_body": True,
            "uses_cluster_m2_one_cta_role_local_bridge": False,
            "cta_thread_count": 256,
            "physical_m_threads": 128,
            "acc_stage_count": 2,
            "ab_stage_count": 2,
            "c_stage_count": 2,
            "epi_warp_count": 4,
            "ab_load_warp_count": 1,
            "scheduler_warp_count": 1,
            "sched_stage_count": 1,
        }

        # c_input_warp_count=0 baseline: 7 role warps, 8 launched
        # (one inert padding warp).
        plan_c0 = CuteTcgen05MatmulPlan(**base_kwargs)
        self.assertEqual(plan_c0.c_input_warp_count, 0)
        self.assertEqual(plan_c0.role_warp_count, 7)
        self.assertEqual(plan_c0.launched_warp_count, 8)
        # All existing role warp ids stay pinned regardless of the
        # c_input lift below.
        self.assertEqual(plan_c0.exec_warp_id, 4)
        self.assertEqual(plan_c0.tma_warp_id, 5)
        self.assertEqual(plan_c0.scheduler_warp_id, 6)
        self.assertEqual(plan_c0.persistent_scheduler_owner_warp_id, 6)

        # c_input_warp_count=1 lift: 8 role warps, 8 launched (no
        # padding).
        plan_c1 = CuteTcgen05MatmulPlan(**base_kwargs, c_input_warp_count=1)
        self.assertEqual(plan_c1.c_input_warp_count, 1)
        self.assertEqual(plan_c1.role_warp_count, 8)
        self.assertEqual(plan_c1.launched_warp_count, 8)
        # Existing role warp ids are unaffected by the lift.
        self.assertEqual(plan_c1.exec_warp_id, 4)
        self.assertEqual(plan_c1.tma_warp_id, 5)
        self.assertEqual(plan_c1.scheduler_warp_id, 6)
        self.assertEqual(plan_c1.persistent_scheduler_owner_warp_id, 6)
        # Block shape is invariant in both cases (256 mma threads
        # × 8 launched warps × 1 = the same launch envelope).
        self.assertEqual(plan_c0.block_shape, plan_c1.block_shape)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_clc_persistent_cluster_n(
        self,
    ) -> None:
        """``CLC_PERSISTENT`` + ``cluster_n>1`` is rejected.

        The CLC scheduler-warp body in
        ``program_id._build_scheduler_warp_role_local_while_clc``
        publishes the work tile to peer CTAs by iterating lanes
        ``< cluster_m``; cluster_n>1 CTAs would never receive the
        CLC mailbox publish and would hang at ``producer_acquire``.
        The paired ``(strategy, persistence_model)`` invariant
        rejects this combination at validate time so the runtime
        path is unreachable.
        """
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )

        # Positive control: CLC + cluster_n=1 still accepts.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=1,
            arch_major=10,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Positive control: STATIC_PERSISTENT + cluster_n=2 accepts
        # (the static path's per-CTA-local scheduler topology
        # generalizes to cluster_n=2).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
            arch_major=10,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # Negative control: CLC + cluster_n=2 rejected. The CLC
        # broadcast is cluster_m-only; second-N-lane CTAs never
        # receive the mailbox publish.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            cluster_n=2,
            arch_major=10,
        )
        self.assertTrue(
            any(
                "clc_persistent" in e and "tcgen05_cluster_n in [1]" in e
                for e in errors
            ),
            msg=str(errors),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_clc_persistent(self) -> None:
        """G2-H (cute_plan.md): ``Tcgen05PersistenceModel.CLC_PERSISTENT``
        is only valid under ``ROLE_LOCAL_WITH_SCHEDULER`` on arch >= 100.

        The validator must reject the model under MONOLITHIC (the
        scheduler-warp role only exists in WITH_SCHEDULER) and on
        arch < 100 (CLC is a Blackwell sm_100+ instruction). The
        positive control: WITH_SCHEDULER + arch_major=10 +
        scheduler_warps=1 + persistent_* pid_type accepts cleanly.
        """
        # Positive control: CLC + WITH_SCHEDULER + sm_100 (arch=10).
        with_sched = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, scheduler_warps=1
        )
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            arch_major=10,
        )
        self.assertEqual(errors, [], msg=str(errors))

        # CLC under MONOLITHIC: rejected (the strategy doesn't
        # support CLC because it has no scheduler warp to issue
        # the query).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=1,
            arch_major=10,
        )
        self.assertTrue(any("clc_persistent" in e for e in errors), msg=str(errors))

        # CLC on arch < 100: rejected.
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="persistent_blocked",
            cluster_m=2,
            arch_major=9,
        )
        self.assertTrue(
            any("requires CUDA compute capability major >= 10" in e for e in errors),
            msg=str(errors),
        )

        # CLC overlays a runtime cancel on the persistent-grid
        # launch, so it must agree with ``pid_type=persistent_*``;
        # CLC paired with ``pid_type=flat`` is rejected with the
        # contradiction error (validator asks user to set both
        # consistently).
        errors = validate_tcgen05_strategy_invariants(
            strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT,
            layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
            warp_spec=with_sched,
            layout_overrides=Tcgen05LayoutOverrides(),
            pid_type="flat",
            cluster_m=1,
            arch_major=10,
        )
        self.assertTrue(
            any("contradicts pid_type" in e for e in errors), msg=str(errors)
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_persistence_model_enum_value_pin(self) -> None:
        """Pin the string literal that ``CuteTcgen05MatmulPlan.is_clc_persistent``
        compares against to the actual enum's ``.value``.

        ``CuteTcgen05MatmulPlan.persistence_model`` is stored as a
        ``str`` (the enum's ``.value``) so the dataclass stays free
        of cute-internal imports. The ``is_clc_persistent`` property
        reads the enum value lazily and compares — this test pins
        that the canonical value is ``"clc_persistent"`` so a rename
        of the enum member would either propagate via the lazy
        import or trip this test loudly. Without it a renamed enum
        could silently degrade ``is_clc_persistent`` to always-False
        because all the comparisons would be against a stale string
        literal in serialized configs.
        """
        self.assertEqual(Tcgen05PersistenceModel.CLC_PERSISTENT.value, "clc_persistent")
        self.assertEqual(
            Tcgen05PersistenceModel.STATIC_PERSISTENT.value, "static_persistent"
        )
        # Round-trip via ``CuteTcgen05MatmulPlan`` to confirm the
        # property tracks the enum value.
        from helion._compiler.device_function import CuteTcgen05MatmulPlan

        plan_clc = CuteTcgen05MatmulPlan(
            bm=256,
            bn=256,
            bk=128,
            k_tile_count=4,
            cluster_m=2,
            is_two_cta=True,
            uses_role_local_persistent_body=True,
            uses_cluster_m2_one_cta_role_local_bridge=False,
            cta_thread_count=256,
            physical_m_threads=32,
            acc_stage_count=2,
            ab_stage_count=2,
            c_stage_count=2,
            epi_warp_count=4,
            ab_load_warp_count=1,
            scheduler_warp_count=1,
            sched_stage_count=1,
            persistence_model=Tcgen05PersistenceModel.CLC_PERSISTENT.value,
        )
        self.assertTrue(plan_clc.is_clc_persistent)
        plan_static = dataclasses.replace(
            plan_clc,
            persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
        )
        self.assertFalse(plan_static.is_clc_persistent)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_invariants_warpgroup_alignment_branch(
        self,
    ) -> None:
        """The warpgroup-alignment branch of
        ``validate_tcgen05_strategy_invariants`` is currently dead
        code (``_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL`` is
        empty) because today's two strategies tolerate non-aligned
        role-warp totals via ``CuteTcgen05MatmulPlan.launched_warp_count``
        rounding at the launch boundary. Patch the set to include
        an existing strategy enum and pass a misaligned warp_spec
        to confirm the validator's alignment check still fires —
        so a future strategy that opts in catches misconfigured
        warp counts loudly.
        """
        from helion._compiler.cute import strategies as strategies_module

        misaligned = Tcgen05WarpSpec(
            ab_load_warps=1,
            mma_warps=1,
            epi_warps=4,
            epi_load_warps=0,
            scheduler_warps=1,  # 1+1+4+1 = 7, not warpgroup-aligned
            register_split=(120, 256),
        )
        with patch.object(
            strategies_module,
            "_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL",
            frozenset({Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER}),
        ):
            errors = validate_tcgen05_strategy_invariants(
                strategy=Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
                persistence_model=Tcgen05PersistenceModel.STATIC_PERSISTENT,
                layout_strategy=Tcgen05LayoutStrategy.DEFAULT,
                warp_spec=misaligned,
                layout_overrides=Tcgen05LayoutOverrides(),
                pid_type="persistent_blocked",
                cluster_m=1,
            )
        self.assertTrue(any("warpgroup-aligned" in e for e in errors), msg=str(errors))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_fix_invalid_resets_to_defaults(self) -> None:
        """G2-A: ``normalize(_fix_invalid=True)`` silently rolls a
        broken strategy record back to the documented defaults rather
        than raising. Mirrors the cluster_m=2 search canonicalization
        path used by ``_fix_tcgen05_cluster_m2_search_config``.
        Layout-override values are silently dropped to ``None``.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        # A user-supplied config that hits the cross-fragment
        # validator (DEFAULT layout + concrete override). Without
        # ``_fix_invalid`` this raises; with it, the strategy fields
        # reset to defaults derived from the active pid_type.
        config = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
            "tcgen05_layout_strategy": "default",
            "tcgen05_layout_overrides_epi_tile_m": 64,
        }
        spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(config["tcgen05_persistence_model"], "static_persistent")
        self.assertEqual(config["tcgen05_layout_strategy"], "default")
        # Override that triggered the rollback is now None.
        self.assertIsNone(config["tcgen05_layout_overrides_epi_tile_m"])

        # An out-of-range override under DEFAULT also fixes silently.
        config2 = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
            "tcgen05_layout_overrides_epi_tile_n": "not-an-int",
        }
        spec.normalize(config2, _fix_invalid=True)
        self.assertIsNone(config2["tcgen05_layout_overrides_epi_tile_n"])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_normalize_idempotent_after_pid_type_fixup(
        self,
    ) -> None:
        """G2-A regression: the strategy default/invariant pass must
        run *after* ``pid_type`` canonicalization and the
        ``_fix_tcgen05_cluster_m{1,2}_*_search_config`` rewrites,
        otherwise ``tcgen05_persistence_model`` is derived from the
        pre-fixup ``pid_type`` and a re-``normalize()`` over the
        already-normalized config trips the
        ``contradicts pid_type`` invariant.

        The path: a search config with ``pid_type="flat"`` and
        ``tcgen05_cluster_m=2`` lands in ``_fix_tcgen05_cluster_m2_search_config``,
        which rewrites ``pid_type`` to ``persistent_interleaved``. The
        derived persistence model must follow that rewrite.
        """

        spec = _bind_cute_strategy_kernel().config_spec

        config: dict[str, object] = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        # The cluster_m2 fixup rewrote pid_type; the persistence-model
        # default agrees with the post-fixup pid_type.
        self.assertEqual(config["pid_type"], "persistent_interleaved")
        self.assertEqual(config["tcgen05_persistence_model"], "static_persistent")

        # Re-normalize on the already-normalized config is idempotent
        # — it does not raise and does not change any field.
        snapshot = dict(config)
        spec.normalize(config, _fix_invalid=False)
        self.assertEqual(config, snapshot)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_flat_round_trip_with_force_persistent(
        self,
    ) -> None:
        """G2-A regression: ``flatten(unflatten(default_flat())) ==
        default_flat()`` even when ``autotune_force_persistent`` has
        narrowed ``allowed_pid_types`` so the default ``pid_type``
        is ``persistent_blocked`` rather than ``flat``.

        ``tcgen05_persistence_model`` is fully derived from
        ``pid_type`` (see ``derive_persistence_model_from_pid_type``)
        so giving it its own slot in ``_flat_fields()`` would mean
        the flat default carries ``non_persistent`` (the
        ``EnumFragment`` default) while the post-normalize value is
        ``static_persistent`` (derived from the persistent
        ``pid_type``). The ``flatten``/``unflatten`` round trip would
        then stabilize on the post-normalize value and the
        autotuner's ``default_flat()`` baseline would diverge from
        every other flat config it generates. Pin the round-trip so
        the field stays out of the autotune surface until a strategy
        decouples it.
        """

        args = (
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_strategy_matmul_force_persistent_kernel.bind(args)
        spec = bound.config_spec
        # autotune_force_persistent removes flat/xyz from the
        # allowed pid_types, so the EnumFragment(pid_type) default
        # is "persistent_blocked".
        self.assertEqual(
            spec.allowed_pid_types,
            ("persistent_blocked", "persistent_interleaved"),
        )
        cg = ConfigGeneration(spec)
        default_flat = cg.default_flat()
        round_tripped = cg.flatten(cg.unflatten(default_flat))
        self.assertEqual(default_flat, round_tripped)
        # Cross-check: the unflattened config's persistence model is
        # the derived value (static_persistent), and the autotune
        # surface (``_flat_fields``) excludes the field so it does
        # not carry a stale flat-config default.
        config = cg.unflatten(default_flat)
        self.assertEqual(
            config.config["tcgen05_persistence_model"], "static_persistent"
        )
        self.assertNotIn("tcgen05_persistence_model", spec._flat_fields())

    @onlyBackends(["cute"])
    def test_cute_tcgen05_strategy_default_lowering_byte_identical(
        self,
    ) -> None:
        """G2-A pins generated code byte-identity: the strategy data
        model is plumbed through ``ConfigSpec`` but no codegen path
        reads it yet. The retained role-local seed produces the same
        kernel source whether the strategy fields are explicitly set
        to their documented defaults or omitted entirely.
        """

        # ``cute_mma.py`` consults ``get_cute_mma_support()`` during
        # codegen, so the patch must remain active across both
        # ``to_triton_code()`` calls — without it, on a host without
        # native tcgen05 support both kernels silently fall through to
        # the non-tcgen05 path and the byte-identity check still
        # passes vacuously. The ``make_trivial_tiled_mma`` assert is a
        # tcgen05-specific marker that catches this regression.
        args = (
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_strategy_matmul_kernel.bind(args)
            baseline_seed = {
                "block_sizes": [256, 256, 16],
                "l2_groupings": [1],
                "pid_type": "persistent_interleaved",
                "tcgen05_cluster_m": 2,
                "tcgen05_ab_stages": 2,
                "tcgen05_acc_stages": 2,
                "tcgen05_c_stages": 2,
                "tcgen05_num_epi_warps": 4,
            }
            baseline = helion.Config(**baseline_seed)
            with_strategy = helion.Config(
                **baseline_seed,
                tcgen05_strategy="role_local_monolithic",
                tcgen05_persistence_model="static_persistent",
                tcgen05_layout_strategy="default",
                tcgen05_warp_spec_ab_load_warps=1,
                tcgen05_warp_spec_mma_warps=1,
                tcgen05_warp_spec_epi_load_warps=0,
                tcgen05_warp_spec_scheduler_warps=0,
                tcgen05_warp_spec_register_decrease=120,
                tcgen05_warp_spec_register_increase=256,
            )

            baseline_code = bound.to_triton_code(baseline)
            with_strategy_code = bound.to_triton_code(with_strategy)
        self.assertIn("make_trivial_tiled_mma", baseline_code)
        self.assertIn("make_trivial_tiled_mma", with_strategy_code)
        self.assertEqual(baseline_code, with_strategy_code)


@onlyBackends(["pallas"])
class TestDotRequirementsPallas(RefEagerTestDisabled, TestCase):
    def test_tpu_min_dot_size_constrains_matmul(self) -> None:
        """Verify that TPU min_dot_size (8, 128, 128) is applied to matmul block sizes."""
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),
        )
        spec = _matmul_kernel.bind(args).config_spec
        self.assertEqual([x.min_size for x in spec.block_sizes], [8, 128, 128])


if __name__ == "__main__":
    unittest.main()
