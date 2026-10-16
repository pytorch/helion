"""CPU proofs that attention storage fits the actual per-CTA CuTe layouts."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_FLAGS

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
ir = pytest.importorskip("cutlass._mlir.ir")
sm100 = pytest.importorskip("cutlass.utils.blackwell_helpers")
flash_fa4_shared_storage = importlib.import_module(
    "helion._compiler.cute._flash_runtime"
).flash_fa4_shared_storage


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    initialized = torch.cuda.is_initialized()
    with _mock_cuda_unavailable():

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("GPU initialization forbidden in the layout proof")

        monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
        yield
    assert torch.cuda.is_initialized() == initialized


@pytest.fixture(autouse=True)
def _context() -> Iterator[None]:
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


def _layouts(
    head_dim: int,
    kv_tile_n: int,
    stages: int,
    dtype: Any,
    cta_group_size: int,
) -> tuple[Any, Any, Any, Any]:
    group = (
        cute.nvgpu.tcgen05.CtaGroup.TWO
        if cta_group_size == 2
        else cute.nvgpu.tcgen05.CtaGroup.ONE
    )
    mma_m = 128 * cta_group_size
    qk = sm100.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        group,
        (mma_m, kv_tile_n),
    )
    pv = sm100.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        cutlass.Float32,
        group,
        (mma_m, head_dim),
        cute.nvgpu.tcgen05.OperandSource.TMEM,
    )
    return (
        sm100.make_smem_layout_a(qk, (mma_m, kv_tile_n, head_dim), dtype, 2),
        sm100.make_smem_layout_b(qk, (mma_m, kv_tile_n, head_dim), dtype, stages),
        sm100.make_smem_layout_b(pv, (mma_m, head_dim, kv_tile_n), dtype, stages),
        sm100.make_smem_layout_epi(
            dtype, cutlass.utils.LayoutEnum.ROW_MAJOR, (128, head_dim), 2
        ),
    )


def _layout_bytes(layout: Any, dtype: Any) -> int:
    return int(cute.cosize(layout)) * dtype.width // 8


def _member_bytes(storage: Any, name: str) -> int:
    # CuTe's static struct metadata includes each aligned MemRange's declared
    # allocation, independently of the layout subsequently applied to its pointer.
    return int(storage._annotations[name].dtype.size_in_bytes)


_LAYOUT_CASES = (
    pytest.param(64, 128, id="d64-n128"),
    pytest.param(64, 160, id="d64-n160"),
    pytest.param(128, 128, id="d128-n128"),
    # These are SDK operand/storage tests, not full-attention legality claims.
    # The current attention TMEM arrangement excludes N192 and N256.
    pytest.param(64, 192, id="sdk-only-d64-n192"),
    pytest.param(64, 256, id="sdk-only-d64-n256"),
    pytest.param(128, 192, id="sdk-only-d128-n192"),
    pytest.param(128, 256, id="sdk-only-d128-n256"),
)
_DTYPES = (
    pytest.param(cutlass.Float16, id="fp16"),
    pytest.param(cutlass.BFloat16, id="bf16"),
)


@pytest.mark.parametrize(("head_dim", "kv_tile_n"), _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stages", (2, 3, 6))
def test_sdk_cooperative_kv_partition_preserves_q_and_o(
    head_dim: int, kv_tile_n: int, dtype: Any, stages: int
) -> None:
    one = _layouts(head_dim, kv_tile_n, stages, dtype, 1)
    two = _layouts(head_dim, kv_tile_n, stages, dtype, 2)
    for index in (0, 3):
        assert _layout_bytes(two[index], dtype) == _layout_bytes(one[index], dtype)
    for index in (1, 2):
        assert 2 * _layout_bytes(two[index], dtype) == _layout_bytes(one[index], dtype)
        # The physical stage stride must agree with the full allocation, not
        # merely the logical element count of one stage.
        layout = two[index]
        stage = cute.select(layout, mode=[0, 1, 2])
        assert int(layout.outer.stride[-1]) == int(cute.cosize(stage))
        assert _layout_bytes(layout, dtype) == stages * _layout_bytes(stage, dtype)


@pytest.mark.parametrize(("head_dim", "kv_tile_n"), _LAYOUT_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stages", (2, 3, 6))
@pytest.mark.parametrize("cta_group_size", (1, 2))
@pytest.mark.parametrize("epi_tma", (False, True))
def test_runtime_storage_matches_sdk_operand_spans(
    head_dim: int,
    kv_tile_n: int,
    dtype: Any,
    stages: int,
    cta_group_size: int,
    epi_tma: bool,
) -> None:
    q, k, v, o = _layouts(head_dim, kv_tile_n, stages, dtype, cta_group_size)
    storage = flash_fa4_shared_storage(
        head_dim,
        stages,
        dtype=dtype,
        epi_tma=epi_tma,
        kv_tile_n=kv_tile_n,
        kv_cta_group_size=cta_group_size,
    )
    assert _member_bytes(storage, "sQ") == _layout_bytes(q, dtype)
    assert _member_bytes(storage, "sK") == max(
        _layout_bytes(k, dtype), _layout_bytes(v, dtype)
    )
    if epi_tma:
        assert _member_bytes(storage, "sO") == _layout_bytes(o, dtype)
    else:
        assert "sO" not in storage._annotations


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("epi_tma", (False, True))
def test_six_cooperative_stages_fit_three_full_stages(
    head_dim: int, dtype: Any, epi_tma: bool
) -> None:
    one = flash_fa4_shared_storage(head_dim, 3, dtype=dtype, epi_tma=epi_tma)
    two = flash_fa4_shared_storage(
        head_dim, 6, dtype=dtype, epi_tma=epi_tma, kv_cta_group_size=2
    )
    _, k_one, v_one, _ = _layouts(head_dim, 128, 3, dtype, 1)
    _, k_two, v_two, _ = _layouts(head_dim, 128, 6, dtype, 2)
    assert _layout_bytes(k_one, dtype) == _layout_bytes(k_two, dtype)
    assert _layout_bytes(v_one, dtype) == _layout_bytes(v_two, dtype)
    assert _member_bytes(one, "sK") == _layout_bytes(k_one, dtype)
    assert _member_bytes(two, "sK") == _layout_bytes(k_two, dtype)
    assert two.size_in_bytes() == one.size_in_bytes()


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("family", ("fa4_cga2_local", "fa4_cga2_local_tma_4d"))
def test_cluster_of_local_ctas_keeps_full_kv_allocation(
    head_dim: int, dtype: Any, family: str
) -> None:
    flags = FLASH_PIPELINE_FAMILY_FLAGS[family]
    assert flags.use_cga2_local_cta and not flags.use_2cta_instrs
    group = 2 if flags.use_2cta_instrs else 1
    _, k, v, _ = _layouts(head_dim, 128, 3, dtype, group)
    default = flash_fa4_shared_storage(head_dim, 3, dtype=dtype)
    explicit = flash_fa4_shared_storage(
        head_dim, 3, dtype=dtype, kv_cta_group_size=group
    )
    assert _member_bytes(explicit, "sK") == _layout_bytes(k, dtype)
    assert _member_bytes(explicit, "sK") == _layout_bytes(v, dtype)
    assert explicit.size_in_bytes() == default.size_in_bytes()


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("epi_tma", (False, True))
def test_separate_one_cta_rings_keep_both_operand_allocations(
    head_dim: int, dtype: Any, epi_tma: bool
) -> None:
    q, k, v, o = _layouts(head_dim, 128, 2, dtype, 1)
    storage = flash_fa4_shared_storage(
        head_dim, 2, dtype=dtype, epi_tma=epi_tma, separate_kv=True
    )
    assert _member_bytes(storage, "sQ") == _layout_bytes(q, dtype)
    assert _member_bytes(storage, "sK") == _layout_bytes(k, dtype)
    assert _member_bytes(storage, "sV") == _layout_bytes(v, dtype)
    assert _member_bytes(storage, "sO") == (_layout_bytes(o, dtype) if epi_tma else 0)
