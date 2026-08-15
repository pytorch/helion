from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._compiler.autotuner_heuristics.cute import CuteTcgen05ClusterM2Heuristic
from helion._compiler.cute.strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from helion._compiler.cute.strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from helion._compiler.cute.strategies import Tcgen05LayoutOverrides
from helion._compiler.cute.strategies import Tcgen05LayoutStrategy
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.strategies import Tcgen05WarpSpec
from helion._compiler.cute.strategies import validate_tcgen05_strategy_invariants
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from helion._compiler.cute.tcgen05_constants import TCGEN05_EXPLICIT_D_STORE_BOX_N
from helion._compiler.cute.tcgen05_constants import TCGEN05_EXPLICIT_EPI_TILE_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_EXPLICIT_EPI_TILE_N
from helion._compiler.cute.tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N,
)
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipIfMTIA
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import MatmulFact
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
_cute_two_matmuls_distributed_kernel = helion.kernel(
    _cute_two_matmuls_impl,
    backend="cute",
    distributed=True,
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
    (see ``CuteTcgen05Config.per_cta_ab_smem_budget_bytes``). Mocking
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
            CuteTcgen05Config,
            "per_cta_ab_smem_budget_bytes",
            staticmethod(lambda device: budget_bytes),
        ),
    ):
        return _cute_4096_matmul_kernel.bind(args)


def _bind_cute_k384_matmul_kernel():
    """Bind a plain bf16 4096x4096x384 matmul: the A5 illegal-``bk`` witness.

    ``384 % 256 != 0``, so ``bk=256`` is an ILLEGAL cluster_m=2 K tile on this
    shape -- while the K ``BlockSizeFragment``'s ``high`` is 256, so the sampler
    draws it. That combination is what makes A5's arm live here and dead on
    4096^3 (whose legal set is the whole drawable domain).
    """
    args = (
        torch.empty([4096, 384], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([384, 4096], device=DEVICE, dtype=HALF_DTYPE),
    )
    _cute_k384_matmul_kernel._bound_kernels.clear()
    with patch_cute_mma_support():
        return _cute_k384_matmul_kernel.bind(args)


@helion.kernel(backend="cute")
def _cute_k384_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
def _cute_2048x4096x4096_matmul_kernel(
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
def _cute_batched_matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched bf16 matmul: the family where the FORMULA heuristic declines.

    ``cute_matmul_formula._single_matmul_fact`` requires ``lhs_ndim == rhs_ndim == 2``,
    so the only promoting tcgen05 producer skips this shape — which is what made the
    no-autotune default here depend on a repair-stage side effect.
    """
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=x.dtype, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc.to(x.dtype)
    return out


def _bind_cute_residual_full_tile_4096_kernel():
    """Bind the 4096³ rank-2 residual kernel: the aux-TMA FULL-TILE family.

    Distinct from ``_bind_cute_residual_5000_kernel`` in the one way that matters
    here: 4096 is a multiple of every drawable tile, so ``allow_edge_k_tail_family``
    is False and ``_aux_tma_full_tile_search_enabled()`` is the arm that admits it.
    The edge family pins ``ab_stages`` to 2 through stage 3's override dict before
    stage 6 runs, which would make the depth cap a vacuous no-op there.
    """
    args = (
        torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([4096, 4096], device=DEVICE, dtype=HALF_DTYPE),
    )
    _cute_residual_5000_kernel._bound_kernels.clear()
    with patch_cute_mma_support():
        return _cute_residual_5000_kernel.bind(args)


def _bind_cute_residual_5000_kernel():
    """Bind the 5000^3 bf16 rank-2 residual (source-C aux) kernel on the real device.

    The SMEM budget is NOT mocked here: this family's gates key on
    ``exact_shape_aux_kernel_detected`` and on the real per-CTA budget, and the
    tile-infeasibility case being tested only exists at the true 203 776 B value.
    """
    args = (
        torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
    )
    _cute_residual_5000_kernel._bound_kernels.clear()
    with patch_cute_mma_support():
        return _cute_residual_5000_kernel.bind(args)


@helion.kernel(backend="cute")
def _cute_rowvec_bias_kernel(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Row-vector-bias matmul: BROAD aux detector True, PRECISE detector False.

    The family where ``aux_kernel_detected`` and
    ``exact_shape_aux_kernel_detected`` disagree, which is what makes it the right
    witness for the C-ring SMEM model (item 11). A rank-1 bias broadcast over N has
    no source-C tile to load, yet the emitted epilogue still builds the
    with-source-C ``(128, 64)`` tile.
    """
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def _cute_residual_5000_kernel(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=HALF_DTYPE, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc + c[tile_m, tile_n].to(torch.float32)).to(HALF_DTYPE)
    return out


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
    # The strategy tests mutate the returned config_spec; avoid sharing that
    # state through the bound-kernel cache across test methods.
    _cute_strategy_matmul_kernel._bound_kernels.clear()
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
        # The promote-to-default formula heuristic emits a wave-count-aware
        # l2_grouping: this tiny single-wave 256x64x128 problem gets the few-wave
        # grouping [4] (was [1] under the old fixed-grouping default).
        self.assertEqual(spec.default_config().config["l2_groupings"], [4])
        # The small-N shape cannot form the validated 256x256 CtaGroup.TWO tile, so
        # the SEARCH keeps cluster_m narrowed to 1. The formula seed is orthogonal
        # to that search restriction (cute-seed-orthogonal-to-search): it promotes
        # the best genuinely-valid config, which here is the rectangular cluster_m=2
        # tile [256,64,64] (bn shrunk to N=64) -- GPU-verified to compile and match
        # x@y exactly. So the promoted default is cluster_m=2 even though the search
        # arm stays cluster_m=1.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 2)
        self.assertEqual(spec.default_config().config["block_sizes"][:2], [256, 64])
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
        # tile_k DRAW bound is 256: the compiler ships pretuned entries at bk=256
        # that its own search could not otherwise reach, and with a bound of 128 a
        # seeded bk=256 is a one-way dead end for the hill-climber
        # (``pattern_neighbors(256)`` clamps to ``[128]``). Per-tile SMEM is what
        # actually bounds bk, and it is enforced downstream. Note this is the DRAW
        # bound only: the tiling-divisibility gates keep the old 128-based value,
        # so cluster_m=2 eligibility and the persistent pid types do not move.
        self.assertEqual([x.max_size for x in spec.block_sizes], [256, 256, 256])
        default_block_sizes = spec.default_config().config["block_sizes"]
        self.assertGreaterEqual(default_block_sizes[2], 16)
        self.assertLessEqual(default_block_sizes[2], 128)
        self.assertGreaterEqual(default_block_sizes[0], 128)
        self.assertLessEqual(default_block_sizes[0], 256)
        self.assertGreaterEqual(default_block_sizes[1], 8)
        self.assertLessEqual(default_block_sizes[1], 256)
        # 16-bit 8192^3 is a full-wave compute shape; the promote-to-default
        # formula heuristic emits the DEFAULT-layout deep-AB CtaGroup.TWO tile
        # ([256,256,64] ab=6) with the wave-count-aware many-wave grouping [1]
        # (this many-CTA shape exceeds the _MANY_WAVE crossover; was [2] under the
        # old FFI-envelope default).
        self.assertEqual(spec.default_config().config["l2_groupings"], [1])
        # K=8192 can form validated CtaGroup.TWO products at bk >= 32 even
        # though bk=16 is over the K-tile cap. The compute full-tile default lands
        # on cluster_m=2, and the search exposes both arms.
        self.assertEqual(spec.default_config().config["tcgen05_cluster_m"], 2)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        # An over-K-tile-cap ``bk`` is now REPAIRED, not demoted (item 9,
        # 2026-08-01). ``bk=16`` at K=8192 needs 512 K-tiles against a 256 cap, so
        # A5 used to answer "not a legal cluster_m=2 bk" by setting cluster_m=1 --
        # which additionally cost the tile, because stage 4 then clamped ``bm`` to
        # 128 (measured: this config came out as ``[128,256,256] cm1``). A5 snaps
        # ``bk`` to a legal value instead, and keeps the tile.
        #
        # ⚠ THE REPAIRED VALUE IS NOW 32, NOT 256 (2026-08-10). A5 used to take the
        # LARGEST legal ``bk``; it now takes the NEAREST to the drawn one. Both are
        # legal here -- at K=8192 the legal set is {32, 64, 128, 256} and ``bk=32``
        # gives exactly 256 K-tiles, right at the cap -- so this asserts the repair
        # POLICY, not legality. Nearest-legal moves the key as little as possible
        # ("change only the knobs that must change"), and it is what made the
        # edge-family K pin redundant: on that family every repairable drawn value
        # is below the legal set, so largest-legal answered 256 while the family's
        # measured tile is 128, and a separate pin existed to overwrite it.
        #
        # THE CAP ITSELF IS UNCHANGED AND STILL BINDING -- it is enforced inside
        # ``cluster_m2_bk_is_valid``, which the repair consults, so the repaired
        # value satisfies it by construction whichever candidate is chosen. See
        # ``test_cute_tcgen05_a5_repairs_bk_and_keeps_the_tile_count_cap``.
        over_cap_config = {
            "block_sizes": [256, 256, 16],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(over_cap_config, _fix_invalid=True)
        self.assertEqual(over_cap_config["tcgen05_cluster_m"], 2)
        self.assertEqual(over_cap_config["pid_type"], "persistent_interleaved")
        over_cap_block_sizes = cast("list[int]", over_cap_config["block_sizes"])
        self.assertEqual(over_cap_block_sizes[2], 32)
        cluster_m2_constraints = spec._tcgen05_cluster_m2_search_constraints
        assert cluster_m2_constraints is not None
        self.assertLessEqual(
            -(-cluster_m2_constraints.static_k // over_cap_block_sizes[2]),
            cluster_m2_constraints.max_k_tiles,
            msg=f"repair broke the K-tile cap: {over_cap_config}",
        )
        valid_config = {
            "block_sizes": [128, 256, 32],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(valid_config, _fix_invalid=True)
        self.assertEqual(valid_config["tcgen05_cluster_m"], 2)
        self.assertEqual(valid_config["pid_type"], "persistent_interleaved")
        # Ordinary cluster_m=2 candidates stay distinct from the FFI seed: M/N
        # are projected onto the CtaGroup.TWO tile while a valid sampled K tile
        # is preserved.
        self.assertEqual(valid_config["block_sizes"][:3], [256, 256, 32])
        self.assertIs(valid_config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], False)
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
        # 16-bit 8192^3 is a full-wave compute shape; the promote-to-default
        # formula heuristic emits a DEFAULT-layout deep-AB tile ([256,256,64] ab=6,
        # bk=64) rather than the old bk=128 envelope -- still a validated tcgen05
        # full tile (bk in the 32..128 range), not the old non-persistent bk=16
        # default. It still codegens on the tcgen05 path.
        self.assertIn(config.config["block_sizes"][2], (64, 128))
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

        # The ordinary cluster_m=2 search arm remains distinct from the FFI
        # direct-entry seed. It projects M/N and the pid type onto the validated
        # CtaGroup.TWO envelope while preserving a valid sampled K tile.
        config = {
            "block_sizes": [256, 256, 16],
            "l2_groupings": [1],
            "pid_type": "persistent_blocked",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config["tcgen05_cluster_m"], 2)
        self.assertEqual(config["pid_type"], "persistent_interleaved")
        self.assertEqual(config["block_sizes"][:3], [256, 256, 16])
        self.assertEqual(config["l2_groupings"], [1])
        self.assertIs(config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], False)

        # ── ``tvm_ffi_launch=True`` NO LONGER PROJECTS ONTO THE SEED ──
        #
        # These five subtests used to assert the opposite: whatever pid_type / bk /
        # l2_grouping was asked for, an ``ffi=True`` config came back as the seed's
        # ``[256,256,128] / l2=[2] / persistent_interleaved``. That behaviour was
        # correct only while the flag was undrawable (``search_choices=(False,)``), so
        # a True could only be a deliberate, partial request.
        #
        # The flag is now a FREE SEARCH AXIS, and the projection is what made that
        # worthless: measured, with the flag un-clamped and the projection still in
        # place, 300 draws collapsed to **152 distinct** configs, the ~150 ``ffi=True``
        # draws landing on ~2. The flag has no codegen dependencies *within* the
        # tcgen05 family (bit-exact on 5 varied configs, and on 20 of the 30 sampled
        # newly-reachable draws below), so there is nothing about an ``ffi=True``
        # config on its own tile to repair.
        #
        # What IS enforced is the flag's one real precondition, per-config and at the
        # END of the pipeline: the config must lower on the tcgen05 path, or
        # ``cute/backend.py`` raises when it emits ``--enable-tvm-ffi``. So the
        # assertions invert: the drawn tile SURVIVES, and the flag survives with it
        # when the tile is tcgen05-emittable.
        for override, tcgen05_emittable in (
            ({"pid_type": "flat"}, True),
            ({"block_sizes": [128, 256, 16]}, True),
            ({"block_sizes": [256, 128, 16]}, True),
            ({"l2_groupings": [16]}, True),
            ({"pid_type": "persistent_interleaved"}, True),
            # ⚠ THE OLD NEGATIVE WITNESS WAS RETIRED HERE (2026-08-07), and this comment
            # block is why. It was ``[256,64,256]`` at ``cluster_m=1, pid_type='flat'``,
            # expecting the flag CLEARED because that tile is not tcgen05-emittable
            # (``bn=64`` with ``bk=256`` fails ``_mma_impl_matches_problem_shape``).
            #
            # The comment that used to sit here spelled out the mechanism it depended on:
            # "``pid_type='flat'`` is load-bearing, not decoration: at
            # ``persistent_blocked`` the cluster_m=1 persistent clamp rewrites bm
            # 256 -> 128, and ``[128,64,256]`` IS emittable, so the flag correctly
            # SURVIVES there." Exactly so — and the clamp is no longer gated on a
            # persistent ``pid_type``, because that gate was letting ``bm=256 ∧
            # cluster_m=1`` reach codegen on the non-tensor-core ``universal`` scalar
            # path (~1500-2000x slower, silently, on 6.8-8.8% of draws across six
            # measured shapes). So ``flat`` now takes the same clamp, the final tile is
            # ``[128,64,256]``, and the flag correctly survives.
            #
            # The negative polarity did not disappear — it moved to
            # ``test_cute_tcgen05_tvm_ffi_launch_is_a_free_searchable_axis``, which drives
            # stage 10 directly on an off-envelope FINAL tile. That is the honest home
            # for it: the property is "stage 10 clears the flag on a non-emittable final
            # tile", and after the clamp NO drawable ``cluster_m=1`` tile on this shape is
            # non-emittable (probed 10/10), so the public-``normalize`` route can no
            # longer construct the witness at all.
            (
                {
                    "block_sizes": [256, 64, 256],
                    "tcgen05_cluster_m": 1,
                    "pid_type": "flat",
                },
                True,
            ),
        ):
            with self.subTest(override=override):
                config = {
                    "block_sizes": [256, 256, 16],
                    "l2_groupings": [1],
                    "pid_type": "persistent_blocked",
                    "tcgen05_cluster_m": 2,
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
                    **override,
                }
                requested_tile = list(cast("list[int]", config["block_sizes"]))
                requested_l2 = list(cast("list[int]", config["l2_groupings"]))
                spec.normalize(config, _fix_invalid=True)
                self.assertIs(
                    config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY],
                    tcgen05_emittable,
                    msg=(
                        "tvm_ffi_launch must survive on a tcgen05-emittable tile and "
                        "be cleared otherwise -- it is a free axis with exactly one "
                        "precondition"
                    ),
                )
                if config["tcgen05_cluster_m"] == 2:
                    # The cluster_m=2 shaping still owns the tile (bm=256, bn snapped
                    # to {128,256}, bk pinned on the edge family) -- that is a
                    # different stage and is unchanged here. What must NOT happen is
                    # the FFI stage overwriting l2_groupings with the seed's value.
                    self.assertEqual(
                        config["l2_groupings"],
                        requested_l2,
                        msg=(
                            f"the FFI stage overwrote l2_groupings "
                            f"{requested_l2} -> {config['l2_groupings']}; a launch "
                            f"flag has no business touching the scheduler grouping"
                        ),
                    )
                else:
                    # ⚠ ``bm`` is EXEMPT from this check at cluster_m=1 (2026-08-07).
                    # The property under test is "the FFI/layout stage did not overwrite
                    # the drawn tile", but ``_fix_cluster_m1_persistent_search_config``
                    # (a DIFFERENT stage, and the one whose whole job this is) clamps
                    # ``bm`` to ``TCGEN05_ONE_CTA_MAX_BLOCK_M`` on every cluster_m=1
                    # candidate — it used to skip non-persistent ``pid_type`` values,
                    # which is exactly the defect that let ``bm=256 ∧ cluster_m=1`` reach
                    # the non-tensor-core ``universal`` path. Asserting the full tile here
                    # would re-pin that hole from a test whose subject is a different
                    # stage. ``bn``/``bk`` are still asserted, which is what the FFI stage
                    # could plausibly touch.
                    self.assertEqual(
                        config["block_sizes"][1:3],
                        requested_tile[1:3],
                        msg="the FFI stage overwrote the drawn bn/bk",
                    )
                    self.assertLessEqual(
                        cast("list[int]", config["block_sizes"])[0],
                        TCGEN05_ONE_CTA_MAX_BLOCK_M,
                        msg=(
                            "bm exceeds the CtaGroup.ONE cap at cluster_m=1, so codegen "
                            "will silently emit the non-tensor-core 'universal' path"
                        ),
                    )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_small_shape_wave_quantization_gate(self) -> None:
        """Cycle 38 (cute_plan.md §7.6.3.2): the cluster_m=2 search arm
        is narrowed for shapes whose cluster_m=2 work-cluster count
        cannot saturate even a quarter-wave of cluster slots.

        Stage 2: the gate counts work clusters at the NARROWEST N tile
        the full-tile cluster_m=2 search admits — ``(M / 256) * (N / 128)``
        (block_n=128, a 256x128 output tile) rather than the 256x256
        artifact ``(M / 256) * (N / 256)``. A block_n=128 cm2 tile produces
        2x the clusters and fills the device where a 256x256 cm2 would
        underfill, so counting at the tile the search can actually pick
        keeps the gate honest for the bn=128 default-layout cm2 path (the
        +8.3% 512x4096x4096 win). The comparison is against ``num_sms // 4``
        (lowered from ``num_sms // 2`` because the generalized TVM-FFI
        direct entry wins at ~0.86 of a wave). With the SM count mocked to
        B200's 148 the threshold is 37 cluster slots: 1024^3 sits at 32
        clusters ((1024/256)*(1024/128) = 4*8) < 37 and narrows to
        ``cluster_m=1`` only, while 2048^3 sits at 128 clusters (>= 37) and
        KEEPS cluster_m=2 search exposed. The 512x4096x4096 medium-M shape
        now sits at 64 clusters ((512/256)*(4096/128) = 2*32) >= 37 and is
        ADMITTED (it was suppressed under the old 256x256 count at 32
        clusters) — the Stage-2 behavior change. The 4096^3 G2 closure
        baseline (512 clusters) also keeps cluster_m=2 search (covered by
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

        def bind_at(size: int, n: int | None = None):
            n = size if n is None else n
            args = (
                torch.empty([size, size], device=DEVICE, dtype=HALF_DTYPE),
                torch.empty([size, n], device=DEVICE, dtype=HALF_DTYPE),
            )
            with (
                patch_cute_mma_support(),
                patch(
                    "helion.language.matmul_ops._cuda_num_sms_or_zero",
                    return_value=148,
                ),
            ):
                return cute_matmul_mma.bind(args).config_spec

        # Suppressed: 1024^3 = (1024/256)*(1024/128) = 32 cluster slots < 148 // 4
        # = 37 (counted at the narrow-N block_n=128 tile). cluster_m=2 search is
        # suppressed and the cluster_m2 seed / fixup machinery is disabled so the
        # autotuner never spends budget on the cluster_m=2 seed for a shape where
        # it has no productive lever.
        suppressed_spec = bind_at(1024)
        self.assertEqual(suppressed_spec._tcgen05_cluster_m_search_choices, (1,))
        self.assertIsNone(suppressed_spec._tcgen05_cluster_m2_search_constraints)
        # Keep this assertion scoped to the cluster_m=2 seed heuristic:
        # future unrelated heuristics may still apply to these shapes.
        self.assertNotIn(
            CuteTcgen05ClusterM2Heuristic.name,
            suppressed_spec.autotuner_heuristics,
        )
        # Persistent pid types are still allowed (the static-full-tile gate
        # above this is unaffected) — only the cluster_m search arm narrows.
        self.assertIn("persistent_interleaved", suppressed_spec.allowed_pid_types)
        self.assertIn("persistent_blocked", suppressed_spec.allowed_pid_types)

        # Admitted (positive control for the lowered // 4 boundary): 2048^3 =
        # (2048/256)*(2048/128) = 128 cluster slots >= 37. cluster_m=2 search
        # stays exposed, its constraints are recorded, and the cluster_m=2 seed
        # heuristic is registered.
        admitted_spec = bind_at(2048)
        self.assertEqual(admitted_spec._tcgen05_cluster_m_search_choices, (1, 2))
        self.assertIsNotNone(admitted_spec._tcgen05_cluster_m2_search_constraints)
        self.assertIn(
            CuteTcgen05ClusterM2Heuristic.name,
            admitted_spec.autotuner_heuristics,
        )

        # Stage 2 behavior change: the 512x4096x4096 medium-M shape sits at
        # (512/256)*(4096/128) = 64 clusters counted at the narrow-N tile
        # (>= 37, ADMITTED), where the old 256x256 count put it at
        # (512/256)*(4096/256) = 32 (< 37, suppressed). This is the shape whose
        # bn=128 cm2 + ab4 config runs +8.3% over the cm1 winner.
        s2_spec = bind_at(512, n=4096)
        self.assertEqual(s2_spec._tcgen05_cluster_m_search_choices, (1, 2))
        self.assertIsNotNone(s2_spec._tcgen05_cluster_m2_search_constraints)
        self.assertIn(
            CuteTcgen05ClusterM2Heuristic.name,
            s2_spec.autotuner_heuristics,
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_fp8_small_grid_cluster_m2_search(self) -> None:
        """fp8 small-grid CtaGroup.TWO family enters search at the bm=128 tile.

        The bm=256 full-tile cluster_m=2 projection underfills small/wave-limited
        fp8 GEMMs (512x2048x4096 -> 16 clusters), so the fp8-validated bm=128
        (per-CTA 64xbn) 2-CTA family from ``_tcgen05_use_2cta_instrs`` must enter
        the autotuner: a sampled bm<=128 candidate stays at the small-grid tile
        (bm=128/bn=128) instead of being forced to bm=256, the cluster_m=2 seed
        heuristic is registered, and it seeds the small-grid tile. A sampled
        bm=256 candidate still projects to the full tile (no regression).
        """

        @helion.kernel(backend="cute")
        def cute_fp8_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(torch.bfloat16)
            return out

        args = (
            torch.empty([512, 4096], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([4096, 2048], device=DEVICE, dtype=torch.float8_e4m3fn),
        )
        with (
            patch_cute_mma_support(),
            patch(
                "helion.language.matmul_ops._cuda_num_sms_or_zero",
                return_value=148,
            ),
        ):
            bound = cute_fp8_matmul.bind(args)
        spec = bound.config_spec

        # The cluster_m=2 search arm is exposed with the fp8 small-grid flag set,
        # and the seed heuristic is registered.
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        constraints = spec._tcgen05_cluster_m2_search_constraints
        self.assertIsNotNone(constraints)
        self.assertTrue(constraints.allow_fp8_small_grid)
        self.assertIn(
            CuteTcgen05ClusterM2Heuristic.name,
            spec.autotuner_heuristics,
        )

        # A sampled bm<=128 cluster_m=2 candidate routes to the small-grid tile,
        # NOT the bm=256 full tile. A sampled epilogue_subtile is dropped: the
        # tcgen05 CtaGroup.TWO MMA path does not support a fused epilogue
        # subtile and would otherwise fail to compile and waste autotune budget.
        small_grid = {
            "block_sizes": [128, 128, 128],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
            "epilogue_subtile": 2,
        }
        spec.normalize(small_grid, _fix_invalid=True)
        self.assertEqual(small_grid["tcgen05_cluster_m"], 2)
        self.assertEqual(small_grid["pid_type"], "persistent_interleaved")
        self.assertIsNone(small_grid.get("epilogue_subtile"))
        self.assertEqual(
            small_grid["block_sizes"][:2],
            [
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N,
            ],
        )

        # ⚠ ``bn`` IS NOT PINNED ON THIS ARM ANY MORE (2026-08-10). The case above
        # draws ``bn=128`` and so cannot tell a pin from a floor -- it passes either
        # way. This case draws ``bn=256`` and pins the actual change: the fp8 arm
        # used to write ``block_sizes[n_index] = 128`` unconditionally, and now
        # leaves ``bn`` to the stage's SETTLE, which is the same rule the non-fp8
        # arm gets (``bn <= 128`` snaps to 128, otherwise 256).
        #
        # The deleted write was wave-quantisation steering, not legality: the tuned
        # 128x128 tile is ALREADY SEEDED (asserted below), so it competes on merit
        # either way, and measured 2026-08-10 the seed producer emits
        # ``[128,128,128] cm=2 ab=12`` identically with the write present or gone.
        # What it cost was reach -- on fp8 512x2048x4096 / 512x6144x2048 / 1024^3,
        # 600 pre-fix draws each: the fp8 arm's ``bn`` distribution went
        # ``{128: 149}`` -> ``{128: 119, 256: 30}`` and distinct tiles 4 -> 8, with
        # ``block_sizes`` the ONLY key that changed. All 30 newly-reachable configs
        # emit real tcgen05 (grepped for ``CtaGroup``, not merely compiled) and are
        # BIT-EXACT under the integer-data oracle.
        #
        # Deliberately NOT freed below 128: the SETTLE's floor stands. At ``bm=128``
        # a ``bn=16`` raises ``OpError: expects the N-mode to satisfy 32 <= ...``,
        # and narrow N measured badly on cm2 generally (``bn=64`` is -40% vs 128 on
        # a wave-saturated bf16 4096^3, and +0.2% -- inside a 0.2% null-arm floor --
        # on the wave-starved shape that most favours it).
        small_grid_wide_n = {
            "block_sizes": [128, 256, 128],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(small_grid_wide_n, _fix_invalid=True)
        self.assertEqual(small_grid_wide_n["tcgen05_cluster_m"], 2)
        self.assertEqual(
            small_grid_wide_n["block_sizes"][:2],
            [TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N],
            msg=(
                "a drawn bn=256 on the fp8 small-grid arm was re-pinned to 128; the "
                "arm must leave bn to the stage's SETTLE so the 256-wide tile "
                "competes (measured bit-exact, 30/600 draws reach it)"
            ),
        )

        # The full-tile cluster_m=2 path drops epilogue_subtile for the same
        # reason.
        full_tile_epi = {
            "block_sizes": [256, 256, 128],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
            "epilogue_subtile": 2,
        }
        spec.normalize(full_tile_epi, _fix_invalid=True)
        self.assertEqual(full_tile_epi["tcgen05_cluster_m"], 2)
        self.assertIsNone(full_tile_epi.get("epilogue_subtile"))

        # A sampled bm=256 cluster_m=2 candidate still projects to the full tile.
        full_tile = {
            "block_sizes": [256, 256, 128],
            "l2_groupings": [1],
            "pid_type": "flat",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(full_tile, _fix_invalid=True)
        self.assertEqual(full_tile["tcgen05_cluster_m"], 2)
        self.assertEqual(
            full_tile["block_sizes"][:2],
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N],
        )

        # The seed heuristic seeds the small-grid tile with the deep A/B pipeline.
        seed = CuteTcgen05ClusterM2Heuristic.get_seed_config(bound.env, None)
        self.assertEqual(seed.config.get("tcgen05_cluster_m"), 2)
        self.assertEqual(
            list(seed.block_sizes[:2]),
            [
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
                TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N,
            ],
        )

        # THE LEGAL-bm INVARIANT (2026-08-01). Every cluster_m=2 config that
        # leaves normalize must carry bm in {128, 256}. This is a hard legality
        # property, not a preference: ``_tcgen05_use_2cta_instrs``
        # (``cute_mma.py``) returns a *bool*, so at bm < 128 with cluster_m=2 it
        # returns False and codegen SILENTLY emits ``CtaGroup.ONE`` -- bit-exact
        # output, wrong kernel, no warning, no raise. bm=128 is additionally
        # fp8-only there (bf16 bm=128 cm2 is the legacy CtaGroup.ONE family).
        #
        # Asserted over DRAWN configs, not hand-built ones: 50% of cm2 draws on
        # this family arrive with bm of 16/32/64 and depend on
        # ``_fix_cluster_m2_search_config`` to establish a legal bm before it
        # returns. Two safeguards do that independently -- ``min_search_m = 128``
        # (a fragment bound) and the stage's own explicit write -- and this test
        # covers the composition, so moving either one cannot silently produce a
        # non-CTA-pair kernel.
        config_gen = ConfigGeneration(spec)
        random.seed(20260801)
        cm2_seen = 0
        for _ in range(400):
            drawn = config_gen.random_config()
            if drawn.config.get("tcgen05_cluster_m") != 2:
                continue
            cm2_seen += 1
            drawn_bm = drawn.block_sizes[0]
            self.assertIn(
                drawn_bm,
                (
                    TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M,
                    TCGEN05_TWO_CTA_BLOCK_M,
                ),
                msg=(
                    f"cluster_m=2 config left normalize with block_m={drawn_bm}, "
                    f"which is not a CtaGroup.TWO tile -- codegen will silently "
                    f"emit CtaGroup.ONE. Full config: {drawn.config}"
                ),
            )
        self.assertGreater(
            cm2_seen, 0, "no cluster_m=2 config was drawn, so this proves nothing"
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_fp8_small_grid_one_wave_seed_ceiling(self) -> None:
        """The heuristic stops *seeding* the bm=128 small-grid tile once its
        128x128 cluster grid exceeds ~one wave (clusters > num_sms // 2), while
        the bm=128 search candidates stay reachable.

        Each 128x128 cluster spans 2 CTAs, so the small-grid tile wins as a seed
        only while clusters*2 <= num_sms (B200 cold-L2: 1.00-1.17x at <=72
        clusters / <=0.97 waves, dropping to 0.84-0.94x from 80 clusters / 1.08
        waves up through 4096^3). Above that ceiling the heuristic seeds the
        bm=256 full tile instead -- but search admission is unchanged
        (allow_fp8_small_grid stays True), so the autotuner can still explore
        bm=128 if it wins.
        """

        @helion.kernel(backend="cute")
        def cute_fp8_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(torch.bfloat16)
            return out

        # 2048x2048 -> (2048/128)^2 = 256 clusters, far above 148 // 2 = 74.
        args = (
            torch.empty([2048, 2048], device=DEVICE, dtype=torch.float8_e4m3fn),
            torch.empty([2048, 2048], device=DEVICE, dtype=torch.float8_e4m3fn),
        )
        with (
            patch_cute_mma_support(),
            patch(
                "helion.language.matmul_ops._cuda_num_sms_or_zero",
                return_value=148,
            ),
            patch("helion.runtime.get_num_sm", return_value=148),
        ):
            bound = cute_fp8_matmul.bind(args)
            spec = bound.config_spec

            # Search admission is unchanged: the small-grid arm stays enabled so
            # bm=128 candidates remain reachable during autotuning.
            constraints = spec._tcgen05_cluster_m2_search_constraints
            self.assertIsNotNone(constraints)
            self.assertTrue(constraints.allow_fp8_small_grid)

            # But the heuristic seeds the bm=256 full tile, NOT bm=128, because
            # 256 clusters is well past the one-wave seed ceiling.
            seed = CuteTcgen05ClusterM2Heuristic.get_seed_config(bound.env, None)
        self.assertEqual(seed.config.get("tcgen05_cluster_m"), 2)
        self.assertEqual(
            list(seed.block_sizes[:2]),
            [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N],
        )

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

        # ⚠ THE NON-PERSISTENT ARM IS NOW CLAMPED TOO (2026-08-07). This block used to
        # assert ``[256, 32, 16]`` survives at ``pid_type='flat'`` — i.e. it PINNED the
        # defect. ``bm=256`` at ``cluster_m=1`` fails
        # ``_mma_impl_matches_problem_shape`` (which admits ``bm in {64,128}``, or
        # ``bm==256`` only with ``cluster_m==2``), so ``_choose_mma_impl`` fell through to
        # the non-tensor-core ``universal`` scalar path: numerically correct, ~1500-2000x
        # slower, and emitted with NO warning (the SMEM-downgrade warning is gated on
        # ``tcgen05_ok``, which that path never reaches).
        #
        # It was reachable by the SAMPLER, not just by explicit configs. Measured over
        # 600 post-pipeline draws per shape: plain 4096³ 53/600 (8.8%), rowvec-bias 53/600,
        # resid 41/600, 2048x4096x4096 53/600, 8192x1024x1024 51/600, 512x4096x4096 53/600
        # — all at ``pid_type='flat'``. Only the 5000³ edge shape was immune, because
        # ``max_search_m`` clamps to 128 there. Each such draw spent a compile plus a
        # benchmark on a kernel that can never win.
        #
        # The clamp itself was always correct; the STAGE was gated on a persistent
        # ``pid_type``, so it never ran on these. That gate term is gone. ``xyz`` behaves
        # identically to ``flat`` (both non-persistent), which is why the hazard is stated
        # as "non-persistent" rather than "flat".
        for non_persistent_pid in ("flat", "xyz"):
            with self.subTest(pid_type=non_persistent_pid):
                non_persistent_config = {
                    "block_sizes": [256, 32, 16],
                    "pid_type": non_persistent_pid,
                    "tcgen05_cluster_m": 1,
                }
                spec.normalize(non_persistent_config, _fix_invalid=True)
                self.assertEqual(
                    non_persistent_config["block_sizes"][:3],
                    [TCGEN05_ONE_CTA_MAX_BLOCK_M, 32, 16],
                    msg=(
                        f"bm=256 at cluster_m=1 with pid_type={non_persistent_pid!r} was "
                        "not clamped, so codegen will silently emit the non-tensor-core "
                        "'universal' scalar path (~1500-2000x slower)"
                    ),
                )
                # The redirect in clamp (2) is scoped to the persistent enum, so a
                # non-persistent pid_type is left exactly as drawn.
                self.assertEqual(non_persistent_config["pid_type"], non_persistent_pid)

        two_cta_config = {
            "block_sizes": [256, 32, 16],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
        }
        spec.normalize(two_cta_config, _fix_invalid=True)
        self.assertEqual(two_cta_config["tcgen05_cluster_m"], 2)
        self.assertEqual(two_cta_config["pid_type"], "persistent_interleaved")
        # The ordinary cluster_m=2 arm preserves its valid bk=16 sample instead
        # of collapsing into the distinct bk=128 FFI direct-entry seed. The
        # sub-128 sampled block_n rounds UP to the narrow 128 cm2 tile (the
        # un-seeded [256,128,*] tile that only search can reach), not to 256.
        self.assertEqual(two_cta_config["block_sizes"][:3], [256, 128, 16])
        self.assertIs(two_cta_config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY], False)

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
        ``test_cute_tcgen05_ab_stages_smem_budget_gate``).
        """
        # B200 production value: 227 KiB optin minus 28 KiB non-AB
        # reservation (see CuteTcgen05Config.per_cta_ab_smem_budget_bytes).
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec

        # 16-bit 4096^3 is FFI-eligible (fp16 == bf16 parity). The initial
        # population carries the DEFAULT-layout cluster_m=2 ab=3 seed and the
        # generalized TVM-FFI direct-entry seed, both on the canonical ab=3
        # fast-config envelope. The formula matmul heuristic additionally emits
        # a deep-AB compute seed for this shape ([256,256,64] ab=6, which fills
        # the AB-SMEM isobar and runs faster than the ab=3 tile); that extra seed
        # is legitimate, so this test asserts the canonical ab=3 envelope is
        # PRESENT among the cluster_m=2 seeds (the point of the test — ab=3 is
        # seeded rather than discovered by mutation) rather than requiring every
        # cluster_m=2 seed to be it.
        cluster_m2_seeds = [
            config.config
            for config in spec.compiler_seed_configs
            if config.config.get("tcgen05_cluster_m") == 2
        ]
        self.assertGreaterEqual(len(cluster_m2_seeds), 1)
        canonical_ab3_seeds = [
            seed
            for seed in cluster_m2_seeds
            if seed["block_sizes"][:3]
            == [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 128]
            and seed["tcgen05_ab_stages"] == 3
        ]
        self.assertGreaterEqual(
            len(canonical_ab3_seeds),
            1,
            f"canonical [256,256,128] ab=3 seed missing from cluster_m=2 seeds: "
            f"{[s['block_sizes'] for s in cluster_m2_seeds]}",
        )

    @onlyBackends(["cute"])
    def test_cute_universal_matmul_lane_loop_correctness(self) -> None:
        """Universal-MMA SMEM staging stays correct under a lane loop.

        Binds a CuTe matmul with a lane-loop configuration
        (``elements_per_thread=2``) on either the M or the N axis and
        asserts both the launch dim (recovery must divide by ``epT``)
        and ``allclose`` against ``x @ y``. Two invariants are covered,
        both symmetric across M and N:

        1. SMEM-load guards use the *physical* thread coord so every
           lane iteration re-populates sA / sB — the
           ``_local_mma_coord_expr`` → ``_physical_mma_coord_expr``
           switch in ``cute_mma._codegen_cute_mma``.
        2. The universal-MMA K-loop emits a trailing
           ``cute.arch.sync_threads()`` after ``cute.gemm`` so this
           iteration's SMEM reads complete before the next iteration
           overwrites sA / sB. Without it a thread that finishes its
           gemm early races ahead and clobbers the tile a sibling is
           still reading (write-after-read hazard), producing
           nondeterministic wrong values — see ``_emit_mma_pipeline``.
           The single-run assert below caught this only intermittently;
           the repeat loop makes the race a hard failure.
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
                # Repeat: the SMEM write-after-read race is timing
                # dependent, so a single launch passes intermittently.
                # Several launches make a missing trailing barrier a
                # deterministic failure.
                expected = x @ y
                for _ in range(8):
                    result = bound(x, y)
                    torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)

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
    def test_cute_tcgen05_double_edge_no_divisor_keeps_flat_search(self) -> None:
        """Double-edge flat tcgen05 search no longer needs divisor tiles."""

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
            torch.randn([67, 16], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([16, 67], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        self.assertEqual([x.max_size for x in spec.block_sizes], [64, 64, 16])
        self.assertIn("tcgen05_cluster_m", spec.default_config().config)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_double_edge_keeps_wide_n_search(self) -> None:
        """Double-edge search keeps N wide and caps M to flat tcgen05."""

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
            torch.randn([192, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 67], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        self.assertEqual([x.max_size for x in spec.block_sizes], [128, 64, 64])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_partial_single_edge_search_stays_enabled(self) -> None:
        """A double-edge default tile keeps wide-N tcgen05 candidates searchable."""

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
            torch.randn([5000, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        self.assertEqual([x.max_size for x in spec.block_sizes], [128, 256, 64])

    @onlyBackends(["cute"])
    def test_cute_tcgen05_edge_k_tail_family_admits_cluster_m2_search(self) -> None:
        """The large double-edge + K-tail family exposes CtaGroup.TWO search."""

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
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma.bind(args)
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        # bk DRAW bound is 256 (see the widening note in
        # ``test_cute_tcgen05_equal_dims_keep_default_within_max_bound``); the
        # edge-family gates below still key on the 128-based divisibility value,
        # which is why cluster_m=2 and persistent_interleaved survive here.
        self.assertEqual([x.max_size for x in spec.block_sizes], [128, 256, 256])
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        flat_fields = spec._flat_fields()
        self.assertIn("pid_type", flat_fields)
        self.assertIn("persistent_interleaved", flat_fields["pid_type"].choices)
        constraints = spec._tcgen05_cluster_m2_search_constraints
        assert constraints is not None
        self.assertTrue(constraints.allow_edge_k_tail_family)
        self.assertTrue(
            spec._tcgen05_cluster_m2_bk_is_valid(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                constraints,
            )
        )
        self.assertFalse(spec._tcgen05_cluster_m2_bk_is_valid(64, constraints))

    def test_tcgen05_edge_tile_detection_skips_unknown_dims(self) -> None:
        config_spec = SimpleNamespace(
            block_sizes=SimpleNamespace(
                block_id_to_index=lambda block_id: {0: 0, 1: 1}[block_id],
            ),
            matmul_facts=[
                MatmulFact(
                    lhs_ndim=2,
                    rhs_ndim=2,
                    m_block_id=None,
                    n_block_id=0,
                    k_block_id=1,
                    static_m=None,
                    static_n=130,
                    static_k=128,
                    lhs_dtype=HALF_DTYPE,
                    rhs_dtype=HALF_DTYPE,
                )
            ],
        )
        tcgen05 = CuteTcgen05Config(config_spec)

        self.assertTrue(
            tcgen05._matmul_fact_has_edge_tile(
                {"block_sizes": [128, 64]},
                fact_index=0,
            )
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_search_disables_tcgen05(
        self,
    ) -> None:
        """Distinct analyzed matmul axes cannot share one tcgen05 config."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_two_matmuls_kernel.bind(args)
        spec = bound.config_spec
        self.assertFalse(spec.cute_tcgen05_search_enabled)
        self.assertIsNone(spec._tcgen05_cluster_m_search_choices)
        self.assertIsNone(spec._tcgen05_num_epi_warps_search_choices)
        self.assertIsNone(spec._cute_tcgen05_config.matmul_block_ids)
        self.assertIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_candidate_collection_ignores_ineligible_matmul(
        self,
    ) -> None:
        args = (
            torch.randn([256, 64], device=DEVICE, dtype=torch.float32),
            torch.randn([64, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            spec = _cute_two_matmuls_kernel.bind(args).config_spec
        self.assertTrue(spec.cute_tcgen05_search_enabled)
        second_fact = spec.matmul_facts[1]
        self.assertEqual(
            spec._cute_tcgen05_config.matmul_block_ids,
            (second_fact.m_block_id, second_fact.n_block_id, second_fact.k_block_id),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_forced_persistent_disables_tcgen05(
        self,
    ) -> None:
        """Forced persistence remains valid on the generic CuTe path."""

        args = (
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([256, 64], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_two_matmuls_force_persistent_kernel.bind(args)
        self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_multi_root_distributed_disables_tcgen05_search(
        self,
    ) -> None:
        """Ambiguous distributed matmul axes stay on the generic search."""

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
        ):
            bound = _cute_two_matmuls_distributed_kernel.bind(args)
        self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)

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

        spec = stub.bind(args).config_spec
        with self.assertRaisesRegex(
            AssertionError,
            "cluster_m=2 search requires persistent pid types",
        ):
            spec.narrow_tcgen05_autotune_to_validated_configs(
                allow_cluster_m2_search=True,
                cluster_m2_static_k=5000,
            )

        spec = stub.bind(args).config_spec
        with self.assertRaisesRegex(
            AssertionError,
            "edge/K-tail admission requires cluster_m=2 search",
        ):
            spec.narrow_tcgen05_autotune_to_validated_configs(
                cluster_m2_static_k=5000,
                allow_cluster_m2_edge_k_tail_family=True,
            )

        spec = stub.bind(args).config_spec
        spec.allowed_pid_types = (
            "flat",
            "xyz",
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec.narrow_tcgen05_autotune_to_validated_configs(
            allow_cluster_m2_search=True,
            cluster_m2_static_k=5000,
            allow_cluster_m2_edge_k_tail_family=True,
        )
        self.assertNotIn("persistent_blocked", spec.allowed_pid_types)
        self.assertIn("persistent_interleaved", spec.allowed_pid_types)
        self.assertEqual(spec._tcgen05_cluster_m_search_choices, (1, 2))
        constraints = spec._tcgen05_cluster_m2_search_constraints
        assert constraints is not None
        self.assertTrue(constraints.allow_edge_k_tail_family)
        self.assertTrue(
            spec._tcgen05_cluster_m2_bk_is_valid(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                constraints,
            )
        )
        self.assertFalse(spec._tcgen05_cluster_m2_bk_is_valid(64, constraints))

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
        # optional fragments in _flat_fields(); structural DeviceIR analysis
        # sets it during bind, so the narrowed search view should appear.
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
        # The promote-to-default formula heuristic pins num_epi_warps=4 explicitly
        # in ``compiler_default_config``, which would shadow the search-view default
        # in ``default_config()``. This assertion exercises the search-view
        # fill-missing routing, so clear the promoted seed to expose the raw
        # search-view fragment default (the property under test).
        spec.compiler_default_config = None
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

        # The 256^2 16-bit shape is a full-wave compute shape; the promote-to-
        # default formula heuristic emits the DEFAULT-layout CtaGroup.TWO compute
        # tile. The ROLE_LOCAL_MONOLITHIC strategy is still the pin, and the seed
        # pins persistent_interleaved / static_persistent (vs the old non-eligible
        # flat / non_persistent), but on the DEFAULT layout rather than the FFI
        # explicit_epi_tile envelope. The persistence model agrees with the
        # persistent pid_type so the serialized config is still internally
        # consistent.
        default_cfg = spec.default_config()
        self.assertEqual(
            default_cfg.config["tcgen05_strategy"], "role_local_monolithic"
        )
        self.assertEqual(default_cfg.config["pid_type"], "persistent_interleaved")
        self.assertEqual(
            default_cfg.config["tcgen05_persistence_model"], "static_persistent"
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
        # The DEFAULT-layout compute default leaves every layout override unset so
        # the layout helper derives the epilogue tile / D-store box / SMEM swizzle
        # (the FFI explicit_epi_tile 128/32/32 envelope ships only on the Bucket-B
        # FFI alt-seed, not the promoted DEFAULT-layout default).
        for key in (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_d_store_box_n",
            "tcgen05_layout_overrides_smem_swizzle_a",
            "tcgen05_layout_overrides_smem_swizzle_b",
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
            layout_overrides=Tcgen05LayoutOverrides(
                epi_tile_m=64, epi_tile_n=32, d_store_box_n=32
            ),
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
        sites that gate on those ids keep the same role assignments.
        """
        from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan

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
        from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan

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
        """G2-A: the cross-fragment strategy validator silently rolls a
        broken strategy record back to the documented defaults under
        ``fix_invalid=True`` rather than raising. Layout-override values
        are silently dropped to ``None``.

        The cross-fragment validator (``validate_strategy_invariants``)
        is exercised DIRECTLY rather than through the full ``normalize``
        chain. Because the 256^2 16-bit shape used here is FFI-eligible
        (fp16 == bf16 parity), the full ``normalize`` chain would (a) reject
        an out-of-fragment override (e.g. ``epi_tile_m=64``) at the optional
        fragment narrowing (now ``(None, 128)``) before reaching the
        cross-fragment validator, and (b) re-pin the strategy onto the
        validated FFI ``explicit_epi_tile`` envelope via the TVM-FFI search
        projection — shadowing the DEFAULT-layout rollback this test
        validates. Calling the validator directly keeps the test focused on
        the silent-rollback gate, mirroring the gate-isolation pattern used by
        ``test_cute_tcgen05_c_stages_budget_gate``.
        """

        spec = _bind_cute_strategy_kernel().config_spec
        tcfg = spec._cute_tcgen05_config

        def _base_strategy_config() -> dict[str, object]:
            # The default config already carries every strategy field the
            # cross-fragment validator reads; mutate a copy to inject the
            # violation under test.
            return dict(spec.default_config().config)

        # A config that hits the cross-fragment validator (DEFAULT layout +
        # concrete override). Without ``fix_invalid`` it raises; with it, the
        # strategy fields reset to defaults derived from the active pid_type
        # and the offending override is dropped to None.
        config = _base_strategy_config()
        config["tcgen05_layout_strategy"] = "default"
        config["tcgen05_layout_overrides_epi_tile_m"] = 128
        with self.assertRaisesRegex(
            InvalidConfig, "tcgen05 strategy invariants violated"
        ):
            tcfg.validate_strategy_invariants(dict(config), fix_invalid=False)
        tcfg.validate_strategy_invariants(config, fix_invalid=True)
        self.assertEqual(config["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(config["tcgen05_persistence_model"], "static_persistent")
        self.assertEqual(config["tcgen05_layout_strategy"], "default")
        self.assertIsNone(config["tcgen05_layout_overrides_epi_tile_m"])

        # A concrete epi_tile_n override under DEFAULT also fixes silently.
        config2 = _base_strategy_config()
        config2["tcgen05_layout_strategy"] = "default"
        config2["tcgen05_layout_overrides_epi_tile_n"] = 64
        tcfg.validate_strategy_invariants(config2, fix_invalid=True)
        self.assertIsNone(config2["tcgen05_layout_overrides_epi_tile_n"])
        self.assertEqual(config2["tcgen05_layout_strategy"], "default")

        # A concrete d_store_box_n override under DEFAULT is also dropped so it
        # never round-trips into generated ``cute.make_layout(...)`` calls.
        config3 = _base_strategy_config()
        config3["tcgen05_layout_strategy"] = "default"
        config3["tcgen05_layout_overrides_d_store_box_n"] = 32
        tcfg.validate_strategy_invariants(config3, fix_invalid=True)
        self.assertIsNone(config3["tcgen05_layout_overrides_d_store_box_n"])
        self.assertEqual(config3["tcgen05_layout_strategy"], "default")

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
        # This test guards the persistence-model derivation round-trip on the
        # SEARCH representation. The promote-to-default formula heuristic pins a
        # cluster_m=2 [256,256,*] compute config in ``compiler_default_config``,
        # which ``default_flat()`` would flatten as the baseline; that promoted
        # config is not flat-round-trip-identity in this force-persistent narrowed
        # spec (its block_m=256 projects back to the flat block_m default of 128),
        # which is a general promoted-seed property, not the persistence-model
        # invariant under test. Clear the promoted seed so ``default_flat()`` uses
        # the search-view fragment default (verified idempotent: fragment-default
        # default_flat DOES round-trip to identity).
        spec.compiler_default_config = None
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
    def test_cute_tcgen05_strategy_defaults_normalize_and_codegen(
        self,
    ) -> None:
        """Documented tcgen05 strategy defaults normalize correctly
        and still select the retained role-local codegen path.

        The old coverage compared two complete generated source
        strings. This version checks the config contract directly and
        uses small structural markers for the generated kernels.
        """

        # ``cute_mma.py`` consults ``get_cute_mma_support()`` during
        # codegen, so the patch must remain active across both
        # ``to_triton_code()`` calls — without it, on a host without
        # native tcgen05 support both kernels silently fall through to
        # the non-tcgen05 path. The marker assertions below catch this
        # regression.
        args = (
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([256, 256], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = _cute_strategy_matmul_kernel.bind(args)
            spec = bound.config_spec
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

            baseline_normalized = dict(baseline_seed)
            explicit_normalized = dict(with_strategy.config)
            spec.normalize(baseline_normalized, _fix_invalid=False)
            spec.normalize(explicit_normalized, _fix_invalid=False)
            for key in (
                "tcgen05_strategy",
                "tcgen05_persistence_model",
                "tcgen05_layout_strategy",
                "tcgen05_warp_spec_ab_load_warps",
                "tcgen05_warp_spec_mma_warps",
                "tcgen05_warp_spec_epi_load_warps",
                "tcgen05_warp_spec_scheduler_warps",
                "tcgen05_warp_spec_register_decrease",
                "tcgen05_warp_spec_register_increase",
            ):
                self.assertEqual(baseline_normalized[key], explicit_normalized[key])

            baseline_code = bound.to_triton_code(baseline)
            with_strategy_code = bound.to_triton_code(with_strategy)

        strategy_markers = (
            "cute.arch.setmaxregister_decrease",
            "cute.arch.setmaxregister_increase",
            "tcgen05_tma_warp =",
            "tcgen05_exec_active =",
            "tcgen05_epi_active =",
            "cute.nvgpu.tcgen05.CtaGroup.TWO",
            "make_trivial_tiled_mma",
            "tcgen05_ab_pipeline_consumer_group =",
            "tcgen05_acc_pipeline_consumer_group =",
            "PipelineUmmaAsync.create",
            "PipelineTmaUmma.create",
            "PipelineTmaStore.create",
            "PersistentTileSchedulerParams",
            "StaticPersistentTileScheduler.create",
            "tcgen05_role_local_0_work_tile",
            "while tcgen05_role_local_0_work_tile.is_valid_tile",
            "_helion_cute_cluster_shape",
            "_helion_cute_wrapper_plans",
        )

        def _strategy_lines(code: str) -> list[str]:
            return [
                line.strip()
                for line in code.splitlines()
                if any(marker in line for marker in strategy_markers)
            ]

        baseline_strategy_lines = _strategy_lines(baseline_code)
        explicit_strategy_lines = _strategy_lines(with_strategy_code)
        self.assertGreaterEqual(
            len(baseline_strategy_lines), 16, msg="\n".join(baseline_strategy_lines)
        )
        self.assertEqual(baseline_strategy_lines, explicit_strategy_lines)

        for code in (baseline_code, with_strategy_code):
            self.assertIn("make_trivial_tiled_mma", code)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertIn("PipelineUmmaAsync.create", code)
            self.assertIn("PipelineTmaUmma.create", code)
            self.assertIn("PipelineTmaStore.create", code)

    # ------------------------------------------------------------------
    # Search-surface widening acceptance tests.
    #
    # These prove three things about the tcgen05 search space, using only
    # bind + normalize (no kernel compile, no benchmark):
    #   * the domain widened (what CAN be drawn),
    #   * draws actually reach the new values (what IS drawn),
    #   * the SMEM budget still refuses what overflows (the negative test).
    #
    # Plus the structural invariant that ties them together: NO DAYLIGHT
    # between what a seed may hold and what the search may draw. A value the
    # search can draw but ``normalize`` rejects wastes draws on configs that
    # die in validation; a value a seed can hold but the search cannot draw
    # makes the seed a silent single point of failure, because the search
    # cannot improve on it or correct it.
    # ------------------------------------------------------------------

    @onlyBackends(["cute"])
    def test_cute_tcgen05_ab_stages_no_daylight_between_surfaces(self) -> None:
        """Every numeric search fragment is bounded by its validation fragment.

        The invariant: ``search.high <= validation.high`` and
        ``search.low >= validation.low`` for every shared numeric knob. If the
        search bound were the wider of the two, the sampler would spend draws on
        values ``normalize`` then rejects outright (measured: before the AB caps
        were unified, a 16-bit shape with no direct-entry structure drew
        ``tcgen05_ab_stages=4`` against a validation bound of ``[1, 3]``).

        Written as a LOOP over every numeric fragment rather than a one-off
        assertion on the knob that happened to be wrong, so a future widening of
        one surface cannot silently reintroduce the gap on a different key.
        """
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec
        search = spec._tcgen05_optional_fragments(for_search=True)
        validation = spec._tcgen05_optional_fragments(for_search=False)

        checked = 0
        for key, search_fragment in search.items():
            validation_fragment = validation.get(key)
            if not isinstance(search_fragment, IntegerFragment) or not isinstance(
                validation_fragment, IntegerFragment
            ):
                continue
            checked += 1
            self.assertLessEqual(
                search_fragment.high,
                validation_fragment.high,
                msg=(
                    f"{key}: search high {search_fragment.high} exceeds validation "
                    f"high {validation_fragment.high} — the search can draw a value "
                    f"normalize will reject"
                ),
            )
            self.assertGreaterEqual(
                search_fragment.low,
                validation_fragment.low,
                msg=(
                    f"{key}: search low {search_fragment.low} is below validation "
                    f"low {validation_fragment.low}"
                ),
            )
        # Guard against the loop silently checking nothing.
        self.assertGreater(checked, 0)
        self.assertIn("tcgen05_ab_stages", search)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_a5_repairs_bk_and_keeps_the_tile_count_cap(self) -> None:
        """A5 snaps an illegal ``bk`` to a legal one instead of demoting to cm1.

        Item 9. The old A5 was ``if not cluster_m2_bk_is_valid(bk): cluster_m = 1``,
        inconsistent with the same function snapping ``bm``, snapping ``bn``, and
        re-tuning ``(bn, bk, ab, c)`` in the joint solve -- and the demote cost MORE
        than cluster_m, because stage 4 then clamps ``bm`` to <= 128, so the
        candidate lost its tile too.

        K=384 is the witness: ``384 % 256 != 0`` makes ``bk=256`` illegal for
        cluster_m=2 while the K fragment's ``high`` is 256, so the sampler draws it.
        (On 4096^3 every drawable ``bk`` is legal, so A5 never fires there -- which is
        why this test does not use the usual shape.)

        THE TILE-COUNT CAP IS THE HARD BOUND AND MUST SURVIVE THE REPAIR.
        ``cluster_m2_bk_is_valid`` enforces ``ceil(static_k / bk) <= max_k_tiles``
        internally, so a repaired ``bk`` satisfies it by construction -- the repair
        can only pick a value the predicate already accepts. That cap is GPU-measured
        (512 K-tiles -> runtime ``RuntimeError``), unlike the edge family's
        ``bk in {128, 256}`` whitelist which is coverage. This test asserts the
        post-repair tile count against the cap directly, so a future "repair" that
        picked a ``bk`` outside the predicate would fail here rather than at runtime.
        """
        bound = _bind_cute_k384_matmul_kernel()
        spec = bound.config_spec
        tcgen05 = spec._cute_tcgen05_config
        constraints = spec._tcgen05_cluster_m2_search_constraints
        self.assertIsNotNone(constraints)
        assert constraints is not None
        # Precondition: bk=256 must actually be illegal here, or the test is vacuous.
        self.assertFalse(tcgen05.cluster_m2_bk_is_valid(256, constraints))
        self.assertTrue(tcgen05.cluster_m2_bk_is_valid(128, constraints))

        config = {
            "block_sizes": [256, 256, 256],
            "l2_groupings": [1],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 2,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
        }
        spec.normalize(config, _fix_invalid=True)
        # REPAIRED, not demoted: cluster_m=2 survives and bm keeps 256 (the demote
        # path would have let stage 4 clamp it to <= 128).
        self.assertEqual(
            config["tcgen05_cluster_m"],
            2,
            msg=f"illegal bk demoted to cluster_m=1 instead of being repaired: {config}",
        )
        block_sizes = cast("list[int]", config["block_sizes"])
        self.assertEqual(block_sizes[0], TCGEN05_TWO_CTA_BLOCK_M)
        self.assertNotEqual(block_sizes[2], 256)
        self.assertTrue(
            tcgen05.cluster_m2_bk_is_valid(block_sizes[2], constraints),
            msg=f"repaired bk={block_sizes[2]} is not legal for this shape",
        )
        # The cap, asserted directly rather than trusted.
        k_tiles = -(-constraints.static_k // block_sizes[2])
        self.assertLessEqual(
            k_tiles,
            constraints.max_k_tiles,
            msg=(
                f"repaired bk={block_sizes[2]} needs {k_tiles} K-tiles, over the "
                f"{constraints.max_k_tiles} cap (512 K-tiles is a GPU-measured "
                f"runtime RuntimeError)"
            ),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_deep_ab_stages_are_settable(self) -> None:
        """A deep 16-bit ``ab_stages`` no longer fails validation outright.

        Before the caps were unified, ``set_config``/``normalize`` on a 16-bit
        shape rejected ``tcgen05_ab_stages=8`` with
        ``InvalidConfig: tcgen05_ab_stages must be in [1, 6], got 8``. The depth
        is now admitted by the fragment and bounded per TILE instead: on this
        canonical 256x256 tile it is clamped down to what the per-CTA SMEM budget
        actually fits, rather than being refused for the whole dtype.
        """
        b200_budget_bytes = 227 * 1024 - 28 * 1024
        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(b200_budget_bytes)
        spec = bound.config_spec

        deep = {
            "block_sizes": [256, 256, 128],
            "l2_groupings": [4],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_ab_stages": 8,
        }
        # No raise: the validation fragment admits the depth.
        spec.normalize(deep, _fix_invalid=True)
        # ...and the tile-level budget still binds. [256,256,128] cm2 bf16 costs
        # 65536 B/stage, so only ab=3 fits the 203776 B budget (196608 B); ab=4
        # would be 262144 B.
        self.assertEqual(deep["tcgen05_ab_stages"], 3)
        self.assertEqual(deep["block_sizes"], [256, 256, 128])

        # A tile where the deep pipeline genuinely fits keeps it: bk=32 quarters
        # the per-stage cost, so [256,256,32] cm2 ab8 is 131072 B <= budget.
        # This is the regime the cap used to amputate -- ab8 is reached by a
        # SMALLER bk, not by a deeper pipeline on the same tile.
        deep_small_bk = {
            "block_sizes": [256, 256, 32],
            "l2_groupings": [4],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_ab_stages": 8,
        }
        spec.normalize(deep_small_bk, _fix_invalid=True)
        self.assertEqual(deep_small_bk["tcgen05_ab_stages"], 8)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_block_k_256_is_two_way(self) -> None:
        """The ``bk=256`` neighbourhood is not a one-way street.

        The compiler ships pretuned fp8 entries at ``bk=256``, so the seed can emit
        it. With the fragment capped at 128 the neighbourhood was one-way --
        ``pattern_neighbors(256)`` clamped to ``[128]``, so the hill-climber could
        only ever walk AWAY from such a seed and never back.
        """
        pow2_fragment = PowerOfTwoFragment(16, 256, 64)
        self.assertIn(128, pow2_fragment.pattern_neighbors(256))
        self.assertIn(256, pow2_fragment.pattern_neighbors(128))

    @onlyBackends(["cute"])
    def test_cute_tcgen05_cluster_m1_applies_both_legality_clamps(self) -> None:
        """The cluster_m=1 stage applies its ``bm`` clamp AND its pid redirect.

        These are two independent legality clamps on different keys, and they used
        to sit in an if/else: on an edge-K-tail-family shape the ``pid_type``
        redirect ``return``ed before the ``block_m`` clamp could run. A
        cluster_m=1 config carrying ``bm=256`` therefore kept it, which is outside
        the CtaGroup.ONE validated envelope (that MMA covers 64/128 M tiles) --
        and the recorded failure mode is silent: ``block_m=256`` with
        ``cluster_m=1`` emits plain Triton instead of erroring, so the search
        would benchmark a different BACKEND than the one it believed it measured.

        Reachable through public ``normalize(..., _fix_invalid=True)``; a random
        draw cannot produce it, because the M fragment is ``[128, 128]`` on such
        shapes, so the exposure is explicit configs, cache entries and seeds.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_edge(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        # 5000 % 256 != 0 on all three axes, which is what admits the
        # edge-K-tail family and so selects the pid-redirect arm.
        args = (
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
            torch.empty([5000, 5000], device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_edge.bind(args)
        spec = bound.config_spec
        constraints = spec._tcgen05_cluster_m2_search_constraints
        self.assertIsNotNone(constraints)
        self.assertTrue(
            constraints.allow_edge_k_tail_family,
            msg="shape must admit the edge-K-tail family for this test to bite",
        )

        config = {
            "block_sizes": [256, 128, 128],
            "l2_groupings": [1],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
        }
        spec.normalize(config, _fix_invalid=True)
        # BOTH clamps: the pid redirect (which used to be the only one) and the
        # bm clamp (which used to be skipped).
        self.assertEqual(config["pid_type"], "flat")
        self.assertEqual(config["block_sizes"][0], TCGEN05_ONE_CTA_MAX_BLOCK_M)

    @onlyBackends(["cute"])
    def test_cute_tcgen05_explicit_epi_tile_declines_aux_tma_producer(self) -> None:
        """explicit epilogue tile × productive aux-TMA producer = ILLEGAL MEMORY ACCESS.

        GPU-verified 2026-08-05 on a 4096³ bf16 rank-2 residual, ``[256,256,32]`` cm2,
        one config per process, integer-data oracle:

        | layout | aux_load_mode | c_input_warps | result |
        |---|---|---|---|
        | `explicit_epi_tile` | **tma** | **1** | **CUDA ILLEGAL MEMORY ACCESS** |
        | `explicit_epi_tile` | simt | 1 | bit-exact |
        | `explicit_epi_tile` | simt | 0 | bit-exact |
        | `DEFAULT` | tma | 1 | bit-exact |
        | `DEFAULT` | simt | 1 | bit-exact |

        Root cause is stated in ``cute_mma.py``'s own comment: the explicit-epi-tile
        family admits a rank-2 exact-shape residual *because* "``c_input_warps == 0``
        is enforced below, so the aux-TMA productive body never fires" — but that
        conjunct lives in the **flat-role** guard, so a plain ``explicit_epi_tile``
        config with ``flat_role=False`` never met it. With a productive aux producer
        the aux-TMA body DOES fire, against a D-store box built for the ``(128, 32)``
        explicit tile.

        **PRE-EXISTING**: the same config faults identically on base ``95ec8eb79``. It
        was unreachable there only because ``explicit_epi_tile`` was drawable but
        never survived repair (0/300 draws) — so making that axis reachable turned a
        latent trap into a live one. Guarded in two places, both asserted here:

        1. ``cute_mma.py`` raises ``BackendUnsupported`` (the honest diagnostic for an
           explicit user config);
        2. ``_aux_tma_request_is_satisfiable`` declines, so a DRAW is demoted to SIMT
           and never spends a compile on a config that faults inside the timed
           benchmark.

        ⚠ The guard is narrowed to ``aux_load_mode=tma``, deliberately. ``simt`` +
        ``c_input_warps=1`` under the explicit tile is the cooperative-SIMT producer
        and is bit-exact; guarding on ``c_input_warps`` alone would delete a working
        regime. An earlier, wider version of this guard did exactly that and was
        caught by re-running the 8-combination table above.
        """
        bound = _bind_cute_residual_full_tile_4096_kernel()
        cute_config = bound.config_spec._cute_tcgen05_config

        def candidate(layout: str, aux: str) -> dict[str, object]:
            return {
                "block_sizes": [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, 32],
                "l2_groupings": [2],
                "pid_type": "persistent_interleaved",
                "tcgen05_cluster_m": 2,
                "tcgen05_cluster_n": 1,
                "tcgen05_ab_stages": 2,
                "tcgen05_c_stages": 2,
                TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: aux,
                TCGEN05_STRATEGY_CONFIG_KEY: (
                    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
                ),
                TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
                TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 1,
                "tcgen05_warp_spec_store_warps": 0,
                TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: layout,
            }

        # The faulting pair is DECLINED, and the drawn warp key is left alone.
        faulting = candidate(Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value, "tma")
        cute_config._fix_aux_tma_search_config(faulting)
        self.assertEqual(
            faulting[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            "simt",
            msg=(
                "explicit_epi_tile + aux_load_mode=tma + a productive aux producer "
                "is a CUDA ILLEGAL MEMORY ACCESS; the search must decline it"
            ),
        )

        # ...and the DEFAULT-layout pair, which is bit-exact, still SURVIVES.
        working = candidate(Tcgen05LayoutStrategy.DEFAULT.value, "tma")
        cute_config._fix_aux_tma_search_config(working)
        self.assertEqual(
            working[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY],
            TCGEN05_AUX_LOAD_MODE_TMA,
            msg=(
                "the guard is too wide: DEFAULT layout + aux-TMA + c_input=1 is "
                "bit-exact and must not be declined"
            ),
        )

        # No DRAWN config may carry the faulting pair.
        config_gen = ConfigGeneration(bound.config_spec)
        random.seed(20260805)
        offenders = 0
        for _ in range(300):
            config = config_gen.random_config().config
            if (
                config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
                == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
                and config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                == TCGEN05_AUX_LOAD_MODE_TMA
                and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
            ):
                offenders += 1
        self.assertEqual(
            offenders,
            0,
            msg=(
                f"{offenders}/300 drawn configs carry the faulting "
                f"explicit_epi_tile + aux-TMA producer combination"
            ),
        )

    @onlyBackends(["cute"])
    def test_cute_tcgen05_layout_overrides_are_derived_not_drawn(self) -> None:
        """The 3 epi-tile overrides are DERIVED from ``layout_strategy`` (§1 tier 2).

        Each has exactly ONE legal non-``None`` value — ``(128, 32, 32)``, the only
        triple ``cute_mma.py``'s D-descriptor codegen accepts. Their determinator is
        ``layout_strategy``: ``explicit_epi_tile`` requires all three set,
        ``DEFAULT`` requires all three ``None``, both enforced by
        ``validate_tcgen05_strategy_invariants``.

        **The derivation is the whole mechanism, and it is what this pins.**
        ``_derive_layout_override_bundle`` overwrites all three from the determinator
        UNCONDITIONALLY — it does not merely fill ``None`` — so the drawn value cannot
        affect the emitted kernel. That is what makes the drawn ``layout_strategy``
        axis coherent (measured 0/300 -> 24/300 reach) rather than an incoherent state
        the validator rejects.

        ⚠ SEARCH AND VALIDATION ARE DELIBERATELY EQUAL HERE, and this test asserts
        that rather than the reverse. These keys briefly carried
        ``search_choices=(None,)``, justified by the OLD stage 1, where the ONLY exit
        for an incoherent bundle was the strip back to DEFAULT.

        The strip arm still EXISTS and still fires often (``:3652``; measured 216/300
        draws) — what changed is that it is no longer reachable BY INCOHERENCE.
        ``_settle_layout_group`` runs first and completes the bundle, so by the time
        the strip is considered the overrides already agree with ``layout_strategy``;
        a config now reaches it for a *different* reason (off the seed's envelope and
        off ``_fix_towards_explicit_epi_tile_envelope``'s preconditions), and it takes
        that path identically whether the draw was ``(128,32,32)`` or
        ``(None,None,None)``. Hence both draws normalize to the SAME config and the
        narrowing protects nothing. Withholding a REDUNDANT value from the search is
        not the same as withholding an ILLEGAL one (contrast ``cluster_n``, whose
        ``2`` is search-withheld because it HANGS); parity is the default absent a
        legality reason.

        Also pins the constant-sharing: the derivation, the seed and
        ``cute_mma.py``'s ``_TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE`` must read
        ONE definition, or a future re-validation moves one and not the others.
        """
        from helion._compiler.cute.cute_mma import (
            _TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE,
        )

        self.assertEqual(
            _TCGEN05_EXPLICIT_EPI_TILE_VALIDATED_SHAPE,
            (
                TCGEN05_EXPLICIT_EPI_TILE_M,
                TCGEN05_EXPLICIT_EPI_TILE_N,
                TCGEN05_EXPLICIT_D_STORE_BOX_N,
            ),
            msg="codegen's validated shape and the derivation constants diverged",
        )

        bound = _bind_cute_4096_matmul_kernel_with_mocked_smem_budget(
            227 * 1024 - 28 * 1024
        )
        spec = bound.config_spec
        cute_config = spec._cute_tcgen05_config
        self.assertTrue(
            cute_config.explicit_epi_tile_family_exists(),
            msg="shape not direct-entry eligible, so the layout group is absent",
        )
        override_keys = (
            "tcgen05_layout_overrides_epi_tile_m",
            "tcgen05_layout_overrides_epi_tile_n",
            "tcgen05_layout_overrides_d_store_box_n",
        )

        # (1) SEARCH AND VALIDATION SURFACES ARE EQUAL. The derivation, not a narrowed
        # draw surface, is what keeps the group coherent -- so there is no legality
        # reason to withhold the concrete value from the search, and a key an explicit
        # ``helion.Config`` may set is one the search may set.
        search = cute_config.optional_fragments(for_search=True)
        validation = cute_config.optional_fragments(for_search=False)
        for key in override_keys:
            with self.subTest(key=key, surface="search"):
                self.assertEqual(
                    search[key]._active_choices(),
                    validation[key]._active_choices(),
                    msg=(
                        f"{key} diverges between the search and validation surfaces. "
                        f"The drawn value is overwritten by "
                        f"_derive_layout_override_bundle either way, so narrowing "
                        f"buys no coherence -- only a parity gap and a "
                        f"fingerprint()/cache-key change"
                    ),
                )
            with self.subTest(key=key, surface="validation"):
                self.assertEqual(
                    len(validation[key]._active_choices()),
                    2,
                    msg=(
                        f"{key} lost its concrete value on the validation surface, "
                        f"so an explicit config / the seed can no longer round-trip"
                    ),
                )

        # (2) the derivation makes both layout values coherent, in both directions.
        for layout, expected in (
            (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
                (
                    TCGEN05_EXPLICIT_EPI_TILE_M,
                    TCGEN05_EXPLICIT_EPI_TILE_N,
                    TCGEN05_EXPLICIT_D_STORE_BOX_N,
                ),
            ),
            (Tcgen05LayoutStrategy.DEFAULT.value, (None, None, None)),
        ):
            config: dict[str, object] = {
                TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: layout,
                # Deliberately start from the WRONG values for this layout, so a
                # no-op implementation cannot pass.
                override_keys[0]: None if expected[0] is not None else 128,
                override_keys[1]: None if expected[1] is not None else 32,
                override_keys[2]: None if expected[2] is not None else 32,
            }
            cute_config._derive_layout_override_bundle(config)
            with self.subTest(layout=layout):
                self.assertEqual(
                    tuple(config[key] for key in override_keys),
                    expected,
                    msg=f"overrides do not follow layout_strategy={layout!r}",
                )

        # (3) A DERIVATION MUST NOT INTRODUCE A KEY. On a non-matmul kernel the
        # tcgen05 keys are absent by design and ``normalize_strategy`` raises
        # ``InvalidConfig`` for any that appears -- presence, not value, is what it
        # tests. Writing None here would break 12 pointwise tests.
        absent: dict[str, object] = {
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (Tcgen05LayoutStrategy.DEFAULT.value)
        }
        cute_config._derive_layout_override_bundle(absent)
        for key in override_keys:
            self.assertNotIn(
                key,
                absent,
                msg=(
                    f"the derivation INTRODUCED {key} into a config that did not "
                    f"have it; on a non-matmul kernel that is an InvalidConfig"
                ),
            )


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
