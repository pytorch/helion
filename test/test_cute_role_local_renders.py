"""CPU render pins for the tcgen05 role-local persistent-loop families of a
plain GEMM (the grouped families are in ``test_cute_role_local_grouped_renders``).

Each test binds one kernel on the fake sm100 target the DeepGEMM-selected
tests use, renders one explicit ``helion.Config``, checks the marker that
proves the render reached its builder in
``helion/_compiler/cute/tcgen05/scheduler_pids.py``, and journals the whole
render in ``test_cute_role_local_renders.expected_cute``:

* F1 ``_build_role_local_while``: per-role ``StaticPersistentTileScheduler``
  (cluster_m 1 / 2, without the PDL wait) and the one-shot form.
* F2 ``_build_role_local_while_with_scheduler`` + P1
  ``_build_scheduler_warp_role_local_while``: the static scheduler warp and
  its mailbox consumers; the store-warp aux merge; the full/edge split.
* F6 ``_build_c_input_warp_role_local_while``: the C-input warp with the
  SIMT and the TMA aux producer.
* P3 ``_build_scheduler_warp_role_local_while_clc``: CLC at cluster_m 1,
  cluster_m 2 (peer broadcast) and cluster_n 2 (four peers).
"""

from __future__ import annotations

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support

import helion
from helion._testing import TestCase
from helion._testing import cpu_render_target
from helion._testing import matchesBackends
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")

# Family markers: the variable or call only that builder emits.
F1_TILE_SCHEDULER = (
    "tcgen05_role_local_0_tile_sched = cutlass.utils.StaticPersistentTileScheduler"
)
MAILBOX_VALID = "tcgen05_role_local_0_valid"
SCHEDULER_WARP_TILE_SCHEDULER = "tcgen05_scheduler_warp_tile_sched"
CLC_BIDX = "tcgen05_clc_bidx"
CLC_PEER_BROADCAST = "_cute_store_shared_remote_x4"
CLC_PEER_N = "tcgen05_clc_sched_peer_n"
C_INPUT_WARP_VALID = "tcgen05_c_input_warp_valid"
FULL_VALID = "_full_valid"
EDGE_VALID = "_edge_valid"
SCHEDULER_WARP_FULL_TILE = "tcgen05_scheduler_warp_full_tile"
ROLE_LOCAL_WHILE = "while tcgen05_role_local"
PDL_WAIT = "cute.arch.griddepcontrol_wait()"
STORE_WARP_AUX_PRODUCER = "tcgen05_aux_pipeline.producer_acquire"
STORE_WARP_C_STORE_EDGE = "tcgen05_c_store_edge.consumer_wait"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _residual_matmul(
    a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(out.dtype)
    return out


def _gemm_args(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
    )


def _residual_gemm_args(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    return (*_gemm_args(m, n, k), torch.empty((m, n), dtype=torch.bfloat16))


def _config(**overrides: object) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [128, 128, 64],
        "l2_groupings": [1],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
    }
    return helion.Config(**(values | overrides))


_SCHEDULER_WARP: dict[str, object] = {
    "tcgen05_strategy": "role_local_with_scheduler",
    "tcgen05_warp_spec_scheduler_warps": 1,
}
_TWO_CTA: dict[str, object] = {"block_sizes": [256, 256, 128], "tcgen05_cluster_m": 2}


def _render(
    kernel: object, args: tuple[torch.Tensor, ...], config: helion.Config
) -> str:
    # ``args`` stays referenced here through ``to_code``: the worklist
    # alignment proofs replay the real bound tensors.
    with cpu_render_target(mma_support=tcgen05_mma_support()):
        bound = cpu_bind(kernel, args)
        return bound.to_code(config, output_origin_lines=False)


class RenderPins(TestCase):
    """Marker checks ahead of the journal; the grouped module shares it."""

    def _pin(self, code: str, *present: str, absent: tuple[str, ...] = ()) -> None:
        for marker in present:
            self.assertIn(marker, code)
        for marker in absent:
            self.assertNotIn(marker, code)
        self.assertExpectedJournal(code)


@onlyBackends(["cute"])
class TestRoleLocalRenders(RenderPins):
    # F1: monolithic static persistent, one StaticPersistentTileScheduler per role.
    def test_f1_static_persistent_cluster_m1(self) -> None:
        code = _render(_matmul, _gemm_args(4096, 4096, 1024), _config())
        self._pin(
            code,
            F1_TILE_SCHEDULER,
            ROLE_LOCAL_WHILE,
            PDL_WAIT,
            absent=(MAILBOX_VALID, SCHEDULER_WARP_TILE_SCHEDULER),
        )

    def test_f1_static_persistent_cluster_m2(self) -> None:
        code = _render(_matmul, _gemm_args(4096, 4096, 1024), _config(**_TWO_CTA))
        self._pin(code, F1_TILE_SCHEDULER, ROLE_LOCAL_WHILE, "(2, 1, 1)")

    def test_f1_cluster_m1_without_pdl(self) -> None:
        # ``cute_pdl`` defaults on for tcgen05 kernels; the knob off removes the
        # dependency wait from the role-local prelude.
        code = _render(_matmul, _gemm_args(4096, 4096, 1024), _config(cute_pdl=False))
        self._pin(code, F1_TILE_SCHEDULER, ROLE_LOCAL_WHILE, absent=(PDL_WAIT,))

    def test_f1_one_shot_role_scheduler(self) -> None:
        # 64 full tiles on 148 SMs: one tile per CTA, the body runs once and
        # the role-local loop is omitted.
        code = _render(_matmul, _gemm_args(1024, 1024, 512), _config())
        self._pin(
            code,
            F1_TILE_SCHEDULER,
            absent=(ROLE_LOCAL_WHILE, "advance_to_next_work"),
        )

    # F2 consumers + P1 static scheduler warp.
    def test_f2_p1_scheduler_warp_one_cta(self) -> None:
        code = _render(
            _matmul, _gemm_args(4096, 4096, 1024), _config(**_SCHEDULER_WARP)
        )
        self._pin(
            code,
            MAILBOX_VALID,
            SCHEDULER_WARP_TILE_SCHEDULER,
            absent=(F1_TILE_SCHEDULER, C_INPUT_WARP_VALID, CLC_BIDX),
        )

    def test_f2_store_warp_aux_merge(self) -> None:
        # store_warps=1 without a C-input warp and a TMA aux ring: the store
        # warp's aux producer body is injected into the epilogue role-local
        # while and the C-store drain rides the same warp.
        code = _render(
            _residual_matmul,
            _residual_gemm_args(4096, 4096, 1024),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_c_stages=4,
                tcgen05_warp_spec_c_input_warps=0,
                tcgen05_warp_spec_store_warps=1,
                tcgen05_aux_load_mode="tma",
            ),
        )
        self._pin(
            code,
            MAILBOX_VALID,
            STORE_WARP_AUX_PRODUCER,
            STORE_WARP_C_STORE_EDGE,
            absent=(C_INPUT_WARP_VALID,),
        )

    # F6: standalone C-input warp.
    def test_f6_c_input_warp_simt_aux(self) -> None:
        code = _render(
            _residual_matmul,
            _residual_gemm_args(4096, 4096, 1024),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_warp_spec_c_input_warps=1,
                tcgen05_aux_load_mode="simt",
            ),
        )
        self._pin(
            code,
            C_INPUT_WARP_VALID,
            MAILBOX_VALID,
            "tcgen05_aux_pipeline = cutlass.pipeline.PipelineAsync.create(",
        )

    def test_f6_c_input_warp_tma_aux(self) -> None:
        code = _render(
            _residual_matmul,
            _residual_gemm_args(4096, 4096, 1024),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_warp_spec_c_input_warps=1,
                tcgen05_aux_load_mode="tma",
            ),
        )
        self._pin(
            code,
            C_INPUT_WARP_VALID,
            MAILBOX_VALID,
            "tcgen05_aux_pipeline = cutlass.pipeline.PipelineTmaAsync.create(",
        )

    def test_full_edge_scheduler_split(self) -> None:
        # Output edges with a TMA store for full tiles only: the scheduler warp
        # publishes the full-tile stream, a sentinel, then the edge stream; the
        # epilogue role gets one loop per phase.
        code = _render(
            _residual_matmul,
            _residual_gemm_args(5000, 5000, 512),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_acc_stages=1,
                tcgen05_warp_spec_c_input_warps=1,
            ),
        )
        self._pin(
            code,
            SCHEDULER_WARP_FULL_TILE,
            FULL_VALID,
            EDGE_VALID,
            C_INPUT_WARP_VALID,
        )

    # P3: CLC scheduler warp.
    def test_p3_clc_cluster_m1(self) -> None:
        code = _render(
            _matmul,
            _gemm_args(4096, 4096, 1024),
            _config(**_SCHEDULER_WARP, tcgen05_persistence_model="clc_persistent"),
        )
        self._pin(code, CLC_BIDX, MAILBOX_VALID, absent=(CLC_PEER_BROADCAST,))

    def test_p3_clc_cluster_m2(self) -> None:
        code = _render(
            _matmul,
            _gemm_args(4096, 4096, 1024),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_persistence_model="clc_persistent",
            ),
        )
        self._pin(code, CLC_BIDX, CLC_PEER_BROADCAST, absent=(CLC_PEER_N,))

    def test_p3_clc_cluster_n2(self) -> None:
        code = _render(
            _matmul,
            _gemm_args(4096, 4096, 1024),
            _config(
                **_SCHEDULER_WARP,
                **_TWO_CTA,
                tcgen05_cluster_n=2,
                tcgen05_persistence_model="clc_persistent",
            ),
        )
        self._pin(code, CLC_BIDX, CLC_PEER_BROADCAST, CLC_PEER_N)
