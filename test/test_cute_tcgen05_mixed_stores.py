"""CPU codegen of tcgen05 epilogues whose stores differ in element type or
read epilogue inputs alone.

One matmul's output stores walk the epilogue tile of the widest store's
element type, each staging and storing it in its own; a store whose value is
an epilogue input load beside them (an "aux-only" store) joins the epilogue
instead of reading a collective-matmul placeholder. The GPU side is
``test_cute_tcgen05_mixed_stores_gpu``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.cute_mma import _tcgen05_epilogue_tile_dtype
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion._testing.cute_mixed_stores import bmm_three_stores
from helion._testing.cute_mixed_stores import bmm_three_stores_args
from helion._testing.cute_mixed_stores import mm_and_copy
from helion._testing.cute_mixed_stores import mm_and_copy_args
from helion._testing.cute_mixed_stores import mm_mixed_stores
from helion._testing.cute_mixed_stores import mm_mixed_stores_args
from helion._testing.cute_mixed_stores import mm_reread_output
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

CPU_DEVICE = "cpu"

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    # B200's 148 SMs, which the default tile heuristics size waves from
    with (
        cpu_codegen_target(),
        forbid_cuda_work(),
        patch(
            "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
            return_value=148,
        ),
    ):
        yield


# A store's epilogue tile sized by FP32, whatever the store's own dtype.
_F32_TILE = re.compile(
    r"(tcgen05_store_epi_tile(?:_\d+)?) = cutlass\.utils\.blackwell_helpers\."
    r"compute_epilogue_tile_shape\(\(\d+, \d+\), False, [\w.]+, cutlass\.Float32,"
)


def _store_bodies(code: str) -> list[str]:
    """Each output store's subtile loop, in store order."""
    loops = re.split(r"\n\s*for _tcgen05_subtile in ", code)[1:]
    return [loop.split("producer_commit()")[0] for loop in loops]


def test_epilogue_tile_dtype_is_the_widest_store() -> None:
    supported = (torch.float16, torch.bfloat16, torch.float32)
    bf16, f16, f32 = torch.bfloat16, torch.float16, torch.float32
    pick = _tcgen05_epilogue_tile_dtype
    assert pick(frozenset({bf16}), bf16, supported) is bf16
    assert pick(frozenset({bf16, f32}), bf16, supported) is f32
    # Equally wide outputs: the input dtype, else the first by name.
    assert pick(frozenset({bf16, f16}), f16, supported) is f16
    assert pick(frozenset({bf16, f16}), f32, supported) is bf16
    # No traced store, or one the epilogue cannot store.
    assert pick(None, bf16, supported) is None
    assert pick(frozenset({bf16, torch.int32}), bf16, supported) is None


# A one-CTA 128x64 tile, and the two-CTA (``tcgen05_cluster_m=2``) 256x64 one.
_MIXED_TILES = {
    "one_cta_128x64": ((1024, 512, 256), {"block_sizes": [128, 64, 64]}),
    "two_cta_256x64": (
        (2048, 1024, 512),
        {
            "block_sizes": [256, 64, 64],
            "tcgen05_cluster_m": 2,
            "pid_type": "persistent_interleaved",
        },
    ),
}


@pytest.mark.parametrize("tile", sorted(_MIXED_TILES))
def test_mixed_dtype_stores_share_the_fp32_subtile(tile: str) -> None:
    """The BF16 store walks the FP32 store's subtiles in its own element
    type, and its host TMA box is built from the same tile."""
    (m, n, k), overrides = _MIXED_TILES[tile]
    bound = mm_mixed_stores._bind_isolated(
        mm_mixed_stores_args(m, n, k, device=CPU_DEVICE, random=False)
    )
    default = {
        key: value
        for key, value in bound.config_spec.default_config().config.items()
        if key != "tcgen05_persistence_model"
    }
    code = bound.to_code(helion.Config(**{**default, **overrides}))
    tiles = _F32_TILE.findall(code)
    assert len(tiles) == 2, tiles
    assert re.search(
        rf"make_smem_layout_epi\(cutlass\.BFloat16, [\w.]+, {tiles[1]},", code
    )
    d_plans = [plan for plan in wrapper_plans(code) if plan["kind"] == "tcgen05_d_tma"]
    assert len(d_plans) == 2
    # The FP32 store's box is sized by its own dtype; the BF16 one names the
    # FP32 tile its device loop stages.
    assert "epi_tile_raw_expr" not in d_plans[0]
    assert "cutlass.Float32" in str(d_plans[1]["epi_tile_raw_expr"])


def test_aux_only_store_joins_the_epilogue() -> None:
    """``aux_out[tile] = aux[tile].to(f32)`` beside the accumulator stores is
    stored by the epilogue: its subtiles read ``aux`` through the
    accumulator's partition and never load TMEM, and the FP32 copy and the
    BF16 sum walk the FP32 accumulator store's subtiles."""
    bound = bmm_three_stores._bind_isolated(
        bmm_three_stores_args(4, 256, 64, 128, device=CPU_DEVICE, random=False)
    )
    config = bound.config_spec.default_config()
    code = bound.to_code(config)
    acc_body, aux_body, sum_body = _store_bodies(code)
    assert "tcgen05_tiled_copy_t2r" in acc_body
    assert "tcgen05_tiled_copy_t2r" not in aux_body
    assert re.search(r"tcgen05_acc_vec_\d+ = tcgen05_aux_loaded_0\.to\(", aux_body)
    assert "tcgen05_tiled_copy_t2r" in sum_body
    # The aux-only store waits for nothing; the last store releases the
    # accumulator after the other two read it.
    assert "consumer_wait" in acc_body
    assert "consumer_wait" not in aux_body + sum_body
    assert "consumer_release" in sum_body
    assert "consumer_release" not in acc_body + aux_body
    assert len(_F32_TILE.findall(code)) == 3
    assert len([p for p in wrapper_plans(code) if p["kind"] == "tcgen05_d_tma"]) == 3


@pytest.mark.parametrize("compute", ("warp", "tcgen05"))
def test_aux_only_store_joins_the_epilogue_of_a_collective_mma_config(
    compute: str,
) -> None:
    """A collective-MMA config takes the tcgen05 store epilogue on SM100,
    which stores the copy as its second TMA store."""
    args = mm_and_copy_args(256, 256, 512, device=CPU_DEVICE, random=False)
    bound = mm_and_copy._bind_isolated((*args, False))
    code = bound.to_code(
        helion.Config(
            block_sizes=[64, 64, 32],
            cute_collective_mma=True,
            cute_collective_compute=compute,
            cute_collective_copy="async_cached",
            cute_collective_stages=2,
        )
    )
    assert "tcgen05_tma_store_atom_1" in code


@pytest.mark.parametrize("copy_first", (True, False), ids=("first", "last"))
def test_aux_only_store_waits_or_releases_without_a_tmem_load(
    copy_first: bool,
) -> None:
    """First among the stores, the copy waits for the accumulator; last, it
    releases it after the accumulator store's TMEM load. It never loads TMEM
    itself."""
    bound = mm_and_copy._bind_isolated(
        (*mm_and_copy_args(1024, 512, 256, device=CPU_DEVICE, random=False), copy_first)
    )
    code = bound.to_code(bound.config_spec.default_config())
    bodies = _store_bodies(code)
    copy_body, acc_body = bodies if copy_first else bodies[::-1]
    assert "tcgen05_tiled_copy_t2r" not in copy_body
    assert "tcgen05_tiled_copy_t2r" in acc_body
    first, last = bodies
    assert "consumer_wait" in first and "consumer_wait" not in last
    assert "consumer_release" in last and "consumer_release" not in first


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_read_back(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
    sum_out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc_out[tile_m, tile_n] = acc
        aux_out[tile_m, tile_n] = aux[tile_m, tile_n].to(torch.float32)
        sum_out[tile_m, tile_n] = (acc + aux_out[tile_m, tile_n]).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_loaded_early(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        early = aux[tile_m, tile_n]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc_out[tile_m, tile_n] = acc
        aux_out[tile_m, tile_n] = early.to(torch.float32)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_scaled_by_overwritten_scalar(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    scale: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        factor = scale[()]
        scale[()] = 2.0
        acc_out[tile_m, tile_n] = acc
        aux_out[tile_m, tile_n] = aux[tile_m, tile_n].to(torch.float32) * factor


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_beside_impure_asm(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
    w: torch.Tensor,
    side: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc_out[tile_m, tile_n] = acc
        aux_out[tile_m, tile_n] = aux[tile_m, tile_n].to(torch.float32) * 2.0
        side[tile_m.id, tile_n.id] = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;",
            "=r,r",
            [w[1]],
            dtype=torch.float32,
            is_pure=False,
            pack=1,
        )


def _not_adopted_cases() -> dict[str, tuple[object, tuple[object, ...]]]:
    a, b, aux, acc_out, sum_out = mm_mixed_stores_args(
        1024, 512, 256, device=CPU_DEVICE, random=False
    )
    return {
        # The copy reads an output the accumulator store writes.
        "source_written": (
            mm_reread_output,
            (
                torch.empty((1024, 256), dtype=torch.bfloat16),
                torch.empty((256, 512), dtype=torch.bfloat16),
                torch.empty((1024, 512), dtype=torch.bfloat16),
                torch.empty((1024, 512), dtype=torch.bfloat16),
            ),
        ),
        # A later chain reads the copy's destination.
        "destination_read": (
            _copy_read_back,
            (a, b, aux, acc_out, torch.empty_like(acc_out), sum_out),
        ),
        # The copy and the accumulator store write one tensor.
        "destination_written": (mm_and_copy, (a, b, aux, acc_out, acc_out, False)),
        # Its operand is loaded ahead of the K loop, before the epilogue
        # exists (refused where that load lowers).
        "loaded_early": (
            _copy_loaded_early,
            (a, b, aux, acc_out, torch.empty_like(acc_out)),
        ),
        # Inline assembly that could hide a store sits in the kernel.
        "opaque_asm": (
            _copy_beside_impure_asm,
            (
                a,
                b,
                aux,
                acc_out,
                torch.empty_like(acc_out),
                torch.zeros(16, dtype=torch.float32),
                torch.zeros((8, 4), dtype=torch.float32),
            ),
        ),
        # Its device scalar is overwritten before the epilogue would read it.
        "scalar_written": (
            _copy_scaled_by_overwritten_scalar,
            (
                a,
                b,
                aux,
                torch.empty((), dtype=torch.float32),
                acc_out,
                torch.empty_like(acc_out),
            ),
        ),
    }


@pytest.mark.parametrize(
    "case",
    (
        "source_written",
        "destination_read",
        "destination_written",
        "loaded_early",
        "opaque_asm",
        "scalar_written",
    ),
)
def test_aux_only_store_the_epilogue_would_reorder_is_not_adopted(case: str) -> None:
    """The epilogue stores the copy with asynchronous TMA stores and reads its
    operands per subtile after the K loop; where that would reorder a read or
    a write of the kernel, or the operand is loaded before the epilogue
    exists, the copy keeps the collective matmul's placeholder refusal."""
    kernel, args = _not_adopted_cases()[case]
    bound = kernel._bind_isolated(args)
    with pytest.raises(exc.BackendUnsupported, match="beside a collective matmul"):
        bound.to_code(bound.config_spec.default_config())


def test_copy_loaded_before_the_k_loop_is_not_an_epilogue_store() -> None:
    """The epilogue cannot store a copy whose operand is loaded ahead of the K
    loop, so the structural facts price no C ring for it and see no auxiliary
    operand in the epilogue."""
    kernel, args = _not_adopted_cases()["loaded_early"]
    tcgen05 = kernel._bind_isolated(args).config_spec.tcgen05
    facts = tcgen05.rowvec_aux_facts
    assert facts is not None and facts.output_count == 1
    assert not tcgen05.aux_kernel_detected
