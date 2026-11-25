"""A batched fp32 GEMM computed as tf32 on the tcgen05 MMA.

CPU code-generation tests: the batched fp32 GEMM takes the tcgen05 search
only where its TMA pipeline loads both rank-3 operands (a row-major A, a row-
or column-major B, TensorMap-aligned rows), at a shape the search keeps and
with the tf32 MMA in the device's record; every other batched fp32 dot,
whatever turned the search down, keeps the lowering that took it before. The
bind patches pin the GB300 SMEM model and the MMA record.
"""

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import TestCase
from helion._testing import default_cute_mma_support
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import patch_target_facts
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import gb300_target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

# The tf32 MMA is off in the default test record: fp32 operands would take no
# tcgen05 MMA at all.
TF32_SUPPORT = default_cute_mma_support(tcgen05_tf32=True)


@contextlib.contextmanager
def _gb300(*, tf32: bool = True) -> Iterator[None]:
    """Bind and render on the GB300 stand-in: its SMEM model and SM count,
    and an MMA record with the tf32 MMA (``tf32=False``: the default record,
    without it)."""
    with (
        gb300_target(),
        patch_cute_mma_support(TF32_SUPPORT if tf32 else default_cute_mma_support()),
        patch_target_facts(num_sm=152),
        forbid_cuda_work(),
    ):
        yield


def _make_bmm(dot_precision: str = "tf32") -> helion.Kernel:
    """A fresh kernel per bind: ``Kernel.bind`` memoizes by argument key."""

    @helion.kernel(backend="cute", static_shapes=True, dot_precision=dot_precision)
    def _bmm(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
        batch, m, k = x.size()
        _, _, n = y.size()
        for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
            acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.baddbmm(
                    acc, x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n]
                )
            out[tile_b, tile_m, tile_n] = acc

    return _bmm


def _make_dot_bmm(*, flat_edge_tma: bool = True) -> helion.Kernel:
    """The batched GEMM as an ``hl.dot``, a form the register-MMA DAG lowers
    (``flat_edge_tma``: the ``cute_flat_edge_tma`` setting)."""

    @helion.kernel(
        backend="cute",
        static_shapes=True,
        dot_precision="tf32",
        cute_flat_edge_tma=flat_edge_tma,
    )
    def _dot_bmm(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
        batch, m, k = x.size()
        _, _, n = y.size()
        for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
            acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = hl.dot(
                    x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n], acc=acc
                )
            out[tile_b, tile_m, tile_n] = acc

    return _dot_bmm


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _two_dot_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    out2: torch.Tensor,
) -> None:
    """Two batched GEMMs into two outputs, each over its own K loop: two
    operand analyses, which the tcgen05 search does not take."""
    batch, m, k = x.size()
    _, _, n = y.size()
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc
        acc2 = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k2 in hl.tile(k):
            acc2 = hl.dot(
                x[tile_b, tile_m, tile_k2], z[tile_b, tile_k2, tile_n], acc=acc2
            )
        out2[tile_b, tile_m, tile_n] = acc2


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _shared_b_bmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """Batched A [batch, m, k] times one B [k, n] the batches share."""
    batch, m, k = x.size()
    _, n = w.size()
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_b, tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _shared_a_bmm(w: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """One A [m, k] the batches share times batched B [batch, k, n]."""
    m, k = w.size()
    batch, _, n = y.size()
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_m, tile_k], y[tile_b, tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc


def _operands(
    batch: int, m: int, n: int, k: int, layout: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 ``x`` [batch, m, k] and ``y`` [batch, k, n]: ``layout[0]`` "R" is
    a row-major (K-major) ``x``, "C" an M-major one; ``layout[1]`` "R" an
    N-major ``y``, "C" a K-major one."""
    x = (
        torch.empty((batch, m, k))
        if layout[0] == "R"
        else torch.empty((batch, k, m)).mT
    )
    y = (
        torch.empty((batch, k, n))
        if layout[1] == "R"
        else torch.empty((batch, n, k)).mT
    )
    return x, y, torch.empty((batch, m, n))


@functools.cache
def _bound_bmm(
    batch: int,
    m: int,
    n: int,
    k: int,
    layout: str = "RR",
    dot_precision: str = "tf32",
) -> BoundKernel:
    """One bind per problem for the module (the tests only read and render)."""
    with _gb300():
        return _make_bmm(dot_precision)._bind_isolated(
            _operands(batch, m, n, k, layout)
        )


def _render(bound: BoundKernel, config: helion.Config | None = None) -> str:
    """``bound``'s code at ``config``, by default its default config."""
    with _gb300():
        return bound.to_code(
            bound.config_spec.default_config() if config is None else config
        )


@onlyBackends(["cute"])
class TestCuteTcgen05BatchedTf32(TestCase):
    def test_tma_operands_take_the_tcgen05_search(self) -> None:
        # 512 x [128, 128] @ [128, 128], b N-major and K-major.
        for layout in ("RR", "RC"):
            spec = _bound_bmm(512, 128, 128, 128, layout).config_spec
            self.assertTrue(spec.tcgen05.search_enabled, layout)
            with _gb300():
                self.assertIn("tcgen05_ab_stages", spec._flat_fields())

    def test_search_tile_lowers_to_the_tf32_tma_pipeline(self) -> None:
        # The one-CTA 64x128 tile over 64-deep fp32 k-blocks the full-effort
        # search picks for 512 x 128^3 (GB300: 17.0 us, cuBLAS 18.4).
        config = helion.Config(
            block_sizes=[1, 64, 128, 64],
            pid_type="persistent_interleaved",
            tcgen05_ab_stages=4,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_persistence_model="static_persistent",
            tcgen05_strategy="role_local_monolithic",
        )
        code = _render(_bound_bmm(512, 128, 128, 128), config)
        self.assertIn("PipelineTmaUmma.create(num_stages=4", code)
        self.assertIn(
            "make_smem_layout_a(tiled_mma, (64, 128, 64), cutlass.TFloat32, 4)", code
        )
        self.assertIn("'input_dtype': 'cutlass.TFloat32'", code)

    def test_default_config_lowers_to_the_tf32_tma_pipeline(self) -> None:
        code = _render(_bound_bmm(16, 128, 128, 256, "RC"))
        self.assertIn("'input_dtype': 'cutlass.TFloat32'", code)
        self.assertIn("tma_atom_a", code)

    def test_refused_tcgen05_config_keeps_the_k_lanes(self) -> None:
        # Whole-tile configs the fp32 tcgen05 lowering refuses (a split
        # epilogue store): the universal lowering declines the batched GEMM,
        # and the scalar fallback that runs instead reads one k per lane of
        # the K loop, not one k per K tile, on the generic thread layout and
        # schedule (16 x 16 threads, not the tcgen05 root's 32 x 8 over the
        # default 64 x 32 tile; a persistent config's generic persistent
        # loop), as main ran batched fp32.
        bound = _bound_bmm(512, 128, 128, 128)
        with _gb300():
            flat = bound.config_spec.default_config()
        flat.config["epilogue_subtile"] = 2
        persistent = helion.Config(
            block_sizes=[1, 64, 128, 64],
            pid_type="persistent_interleaved",
            tcgen05_persistence_model="static_persistent",
            epilogue_subtile=2,
        )
        for config in (flat, persistent):
            code = _render(bound, config)
            self.assertNotIn("cutlass.TFloat32", code)
            self.assertIn("indices_3 = tile_offset_3 + cutlass.Int32(lane_3)", code)
            self.assertIn("block=(16, 16, 1)", code)

    def test_rows_no_tensormap_addresses_keep_the_collective_lowering(self) -> None:
        # 132-byte A rows (K = 33): no TMA load, so no tcgen05 search.
        bound = _bound_bmm(8, 100, 72, 33)
        spec = bound.config_spec
        self.assertFalse(spec.tcgen05.search_enabled)
        with _gb300():
            self.assertIn("cute_collective_compute", spec._flat_fields())
        _render(bound)

    def test_m_major_a_keeps_the_collective_lowering(self) -> None:
        bound = _bound_bmm(8, 128, 128, 128, "CR")
        self.assertFalse(bound.config_spec.tcgen05.search_enabled)
        _render(bound)

    def test_ieee_fp32_keeps_the_collective_lowering(self) -> None:
        bound = _bound_bmm(512, 128, 128, 128, dot_precision="ieee")
        self.assertFalse(bound.config_spec.tcgen05.search_enabled)

    def test_dots_off_the_tcgen05_search_keep_the_register_mma_dag(self) -> None:
        # A batched fp32 hl.dot the tcgen05 search does not take gets no
        # operand analysis once the search is planned, so the register-MMA
        # DAG keeps it on its tf32 MMAs (the default config's), whatever
        # turned the search down: a shape it leaves to SIMT (8 x 32^3: M
        # below 64, N and K below 128, few batches), a record without the
        # tf32 MMA, cute_flat_edge_tma=False (the search tiles divide the
        # extents: 96 rows give 32, below the 64-row floor) and two GEMMs in
        # one kernel (two analyses; the DAG is searched, not the default).
        with _gb300():
            bound = _make_dot_bmm()._bind_isolated(_operands(8, 96, 96, 96, "RR"))
        # (with the flat-edge TMA, 8 x 96^3 takes the search)
        self.assertTrue(bound.config_spec.tcgen05.search_enabled)
        flat_edge_off = _make_dot_bmm(flat_edge_tma=False)
        e = torch.empty
        cases = (
            (_make_dot_bmm(), _operands(8, 32, 32, 32, "RR"), True, True),
            (_make_dot_bmm(), _operands(512, 128, 128, 128, "RR"), False, True),
            (flat_edge_off, _operands(8, 96, 96, 96, "RR"), True, True),
            (flat_edge_off, _operands(8, 96, 128, 128, "RR"), True, True),
            (_two_dot_bmm, tuple(e(512, 128, 128) for _ in range(5)), True, False),
        )
        for kernel, args, tf32, dag_default in cases:
            shapes = [tuple(arg.shape) for arg in args[:2]]
            with _gb300(tf32=tf32):
                bound = kernel._bind_isolated(args)
                code = (
                    bound.to_code(bound.config_spec.default_config())
                    if dag_default
                    else ""
                )
            self.assertFalse(bound.config_spec.tcgen05.search_enabled, shapes)
            self.assertIsNotNone(bound.config_spec.cute_mma_dag, shapes)
            if dag_default:
                self.assertIn("mma_tf32(", code)

    def test_mixed_rank_products_keep_the_other_lowerings(self) -> None:
        # A row-major, TensorMap-aligned operand shared by the batches, beside
        # a batched A or B: outside the rank-3 form the batched tf32 pipeline
        # runs.
        cases = (
            (
                _shared_b_bmm,
                (torch.empty(4, 256, 64), torch.empty(64, 256)),
            ),
            (
                _shared_a_bmm,
                (torch.empty(256, 64), torch.empty(4, 64, 256)),
            ),
        )
        for kernel, operands in cases:
            with _gb300():
                spec = kernel._bind_isolated(
                    (*operands, torch.empty(4, 256, 256))
                ).config_spec
            self.assertFalse(spec.tcgen05.search_enabled, kernel.name)
