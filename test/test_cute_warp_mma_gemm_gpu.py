"""Runtime numerics for the ``warp_mma`` matmul family (sm_100 GPU).

Every admitted operand dtype (fp16, bf16, e4m3), both B layouts (N- and
K-contiguous), batched problems, bias rows, residual tiles, fp32 outputs and
every seeded tile / warp count are checked against fp32 torch matmuls of the
upcast operands; repeated launches are bitwise identical (fixed K order).
"""

from __future__ import annotations

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import _batched_gemm
from test.test_cute_tcgen05_fixed_chain import _ordinary_gemm
from test.test_cute_tcgen05_fixed_chain import _two_output_gemm
from test.test_cute_tcgen05_fixed_cost import _row_bias_gemm
from test.test_cute_warp_mma_gemm import _bias_gelu_aux_gemm
from test.test_cute_warp_mma_gemm import _extra_root_store_after
from test.test_cute_warp_mma_gemm import _extra_root_store_before
from test.test_cute_warp_mma_gemm import _f32_out_gemm
from test.test_cute_warp_mma_gemm import _fp8_gemm
from test.test_cute_warp_mma_gemm import _inplace_residual
from test.test_cute_warp_mma_gemm import _rank0_scaled_gemm
from test.test_cute_warp_mma_gemm import _relu_residual_gemm
from test.test_cute_warp_mma_gemm import _sliced_output

import helion
from helion._compiler.cute import cute_warp_mma_gemm as family
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="the register-MMA family targets the sm_100 GPUs of the tcgen05 search",
    ),
]

FAMILY = family.WARP_MMA_FAMILY_KEY
WARPS = family.WARP_MMA_WARPS_KEY
LAUNCHES = 8


def _config(block_sizes: list[int], warps: int) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        **{FAMILY: family.MATMUL_FAMILY_WARP_MMA, WARPS: warps},
    )


def _run(kernel_fn, args: tuple[torch.Tensor, ...], config: helion.Config):
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "_helion_warp_mma.mma_" in source, source
    assert "tcgen05_tmem_allocator" not in source
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    first = outputs[0]
    for other in outputs[1:]:
        if isinstance(first, tuple):
            for lhs, rhs in zip(first, other, strict=True):
                assert torch.equal(lhs, rhs)
        else:
            assert torch.equal(first, other)
    return first, source


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device=DEVICE, dtype=torch.float32).to(dtype)


def _k_major(b: torch.Tensor) -> torch.Tensor:
    return b.transpose(-1, -2).contiguous().transpose(-1, -2)


def _check(out: torch.Tensor, ref: torch.Tensor) -> None:
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


TILES = [
    ([16, 8, 128], 1),
    ([16, 16, 128], 1),
    ([32, 16, 128], 2),
    ([32, 32, 128], 4),
    ([64, 8, 128], 4),
    ([64, 32, 256], 4),
    ([32, 8, 32], 1),
    ([64, 64, 64], 8),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_major", [False, True])
@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_dense_gemm_matches_torch(
    dtype: torch.dtype, k_major: bool, block_sizes: list[int], warps: int
) -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=dtype)
    b = _randn(256, 256, dtype=dtype)
    if k_major:
        b = _k_major(b)
    out, _ = _run(_ordinary_gemm, (a, b), _config(block_sizes, warps))
    _check(out, torch.matmul(a.float(), b.float()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k_major", [False, True])
@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_m_major_a_gemm_matches_torch(
    dtype: torch.dtype, k_major: bool, block_sizes: list[int], warps: int
) -> None:
    # ``mat1.T @ grad_out``: A is the transposed view of a row-major matrix.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=dtype).T
    b = _randn(256, 256, dtype=dtype)
    if k_major:
        b = _k_major(b)
    out, source = _run(_ordinary_gemm, (a, b), _config(block_sizes, warps))
    assert "_helion_warp_mma.ldmatrix_x4_trans(" in source
    _check(out, torch.matmul(a.float(), b.float()))


def test_matmul_backward_pair_runs_on_the_family() -> None:
    """Both GEMMs of the matmul example's backward (``grad_out @ mat2.T`` and
    ``mat1.T @ grad_out``) run on the family's measured 16x32 two-warp tile."""
    import examples.matmul as matmul_mod

    torch.manual_seed(0)
    mat1 = _randn(256, 256, dtype=torch.float16)
    mat2 = _randn(256, 256, dtype=torch.float16)
    grad_out = _randn(256, 256, dtype=torch.float16)
    matmul = helion.kernel(matmul_mod.matmul.fn, backend="cute", static_shapes=True)
    for lhs, rhs in ((grad_out, mat2.T), (mat1.T, grad_out)):
        bound = matmul.bind((lhs, rhs))
        bound.set_config(_config([16, 32, 64], 2))
        with bound.env:
            normalized = bound.config_spec.normalized_config(_config([16, 32, 64], 2))
        assert "_helion_warp_mma.mma_f16(" in bound.to_code(normalized)
        _check(bound(lhs, rhs), torch.matmul(lhs.float(), rhs.float()))


@pytest.mark.parametrize(("block_sizes", "warps"), TILES)
def test_fp8_gemm_matches_scaled_mm(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _k_major(_randn(256, 256, dtype=torch.float8_e4m3fn))
    out, source = _run(_fp8_gemm, (x, y), _config(block_sizes, warps))
    assert "_helion_warp_mma.mma_e4m3(" in source
    scale = torch.tensor(1.0, device=DEVICE)
    ref = torch._scaled_mm(
        x, y, scale, scale, use_fast_accum=False, out_dtype=torch.float16
    )
    _check(out, ref.float())
    _check(out, torch.matmul(x.float(), y.float()))


def test_fp8_gemm_rejects_an_n_major_b() -> None:
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _randn(256, 256, dtype=torch.float8_e4m3fn)
    kernel = helion.kernel(_fp8_gemm, backend="cute", static_shapes=True)
    bound = kernel.bind((x, y))
    with pytest.raises(Exception, match="not admitted|cute_matmul_family"):
        bound.set_config(_config([16, 8, 128], 1))
        with bound.env:
            bound.config_spec.normalized_config(_config([16, 8, 128], 1))


@pytest.mark.parametrize(
    ("block_sizes", "warps"), [([64, 8, 128], 4), ([16, 8, 128], 1), ([32, 32, 128], 4)]
)
def test_bias_row_epilogue(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    out, source = _run(_row_bias_gemm, (a, b, bias), _config(block_sizes, warps))
    assert "_helion_warp_mma.unpack_f16x2(_helion_warp_mma.ldg_b32(" in source
    _check(out, torch.matmul(a.float(), b.float()) + bias.float())


@pytest.mark.parametrize(
    ("block_sizes", "warps"),
    [
        ([1, 64, 8, 128], 4),
        ([1, 16, 8, 128], 1),
        ([1, 32, 32, 128], 4),
        ([1, 32, 16, 64], 2),
    ],
)
def test_batched_gemm(block_sizes: list[int], warps: int) -> None:
    torch.manual_seed(0)
    a = _randn(4, 64, 128, dtype=torch.float16)
    b = _randn(4, 128, 128, dtype=torch.float16)
    out, _ = _run(_batched_gemm, (a, b), _config(block_sizes, warps))
    _check(out, torch.bmm(a.float(), b.float()))


def test_fp32_output() -> None:
    torch.manual_seed(0)
    a = _randn(128, 64, dtype=torch.bfloat16)
    b = _randn(64, 192, dtype=torch.bfloat16)
    out, source = _run(_f32_out_gemm, (a, b), _config([32, 8, 64], 2))
    assert "_helion_warp_mma.stg_v2_f32(" in source
    assert out.dtype == torch.float32
    _check(out, torch.matmul(a.float(), b.float()))


def test_relu_residual_epilogue() -> None:
    torch.manual_seed(0)
    a = _randn(192, 256, dtype=torch.float16)
    b = _randn(256, 128, dtype=torch.float16)
    residual = _randn(192, 128, dtype=torch.float16)
    out, _ = _run(_relu_residual_gemm, (a, b, residual), _config([16, 16, 128], 1))
    ref = torch.relu(torch.matmul(a.float(), b.float()) + residual.float())
    _check(out, ref)


def test_rank0_scale_epilogue() -> None:
    # The fp8_gemm dequantization scales: rank-0 loads are tile-uniform
    # scalar leaves rendered inline on the fragments, and the launcher
    # marshals the 0-d tensors as one-element views.
    torch.manual_seed(0)
    a = _randn(128, 64, dtype=torch.bfloat16)
    b = _randn(64, 128, dtype=torch.bfloat16)
    scale_a = torch.tensor(0.5, device=DEVICE)
    scale_b = torch.tensor(3.0, device=DEVICE)
    out, source = _run(
        _rank0_scaled_gemm, (a, b, scale_a, scale_b), _config([32, 32, 64], 4)
    )
    assert "* cutlass.Float32(scale_a.iterator.load())" in source
    assert "* cutlass.Float32(scale_b.iterator.load())" in source
    _check(out, torch.matmul(a.float(), b.float()) * 1.5)


def test_two_outputs() -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    (out, out2), _ = _run(_two_output_gemm, (a, b), _config([32, 16, 128], 2))
    ref = torch.matmul(a.float(), b.float())
    _check(out, ref)
    _check(out2, torch.relu(ref))


def test_under_aligned_operand_fails_closed() -> None:
    # The cp.async packets need 16-byte-aligned operand bases; a view eight
    # bytes into a buffer is refused with the reason rather than lowered.
    torch.manual_seed(0)
    full = _randn(256, 264, dtype=torch.float16)
    a = full[:, 4:260]
    b = _randn(256, 256, dtype=torch.float16)
    kernel = helion.kernel(_ordinary_gemm, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b))
    with pytest.raises(BackendUnsupported, match="16-byte aligned"):
        bound.set_config(_config([16, 8, 128], 1))
        bound(a, b)


def test_campaign_examples_run_on_the_family() -> None:
    """The three campaign variants on their measured tiles."""
    import examples.bmm as bmm_mod
    import examples.fp8_gemm as fp8_mod
    import examples.matmul as matmul_mod

    torch.manual_seed(0)
    x = _randn(256, 256, dtype=torch.float8_e4m3fn)
    y = _k_major(_randn(256, 256, dtype=torch.float8_e4m3fn))
    fp8 = helion.kernel(fp8_mod.fp8_gemm.fn, backend="cute", static_shapes=True)
    bound = fp8.bind((x, y))
    bound.set_config(_config([16, 8, 128], 1))
    out = bound(x, y)
    _check(out, torch.matmul(x.float(), y.float()))

    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    epilogue = lambda acc, tile: acc + bias[tile[1]]  # noqa: E731
    matmul = helion.kernel(matmul_mod.matmul.fn, backend="cute", static_shapes=True)
    bound = matmul.bind((a, b, epilogue))
    bound.set_config(_config([64, 8, 128], 4))
    out = bound(a, b, epilogue)
    _check(out, torch.addmm(bias.float(), a.float(), b.float()))

    a3 = _randn(4, 64, 128, dtype=torch.float16)
    b3 = _randn(4, 128, 128, dtype=torch.float16)
    bmm = helion.kernel(bmm_mod.bmm.fn, backend="cute", static_shapes=True)
    bound = bmm.bind((a3, b3))
    bound.set_config(_config([1, 64, 8, 128], 4))
    out = bound(a3, b3)
    _check(out, torch.bmm(a3.float(), b3.float()))


def _grouped_reference(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """The jagged kernel's contract: rows of a group hold ``A[rows] @ B``;
    rows outside every group are left as allocated (NaN here)."""
    out = torch.full(
        (a.shape[0], b.shape[1]), float("nan"), dtype=a.dtype, device=a.device
    )
    bounds = offsets.tolist()
    for g in range(len(bounds) - 1):
        start, end = bounds[g], bounds[g + 1]
        if end - start > 0:
            out[start:end] = (a[start:end].float() @ b.float()).to(a.dtype)
    return out


@pytest.mark.parametrize(
    ("block_sizes", "warps"),
    [([32, 32, 64], 4), ([64, 16, 64], 4), ([16, 32, 64], 2), ([16, 8, 128], 1)],
)
def test_grouped_row_union_matches_the_jagged_kernel(
    block_sizes: list[int], warps: int
) -> None:
    """``examples/grouped_gemm.py``'s jagged kernel on the family: the
    catalogue's shape-0 layout, then (same compiled kernel, ``static_shapes=
    False``) a layout with an empty group and one that leaves the tail rows
    outside every group."""
    import examples.grouped_gemm as grouped_mod

    torch.manual_seed(0)
    a = _randn(640, 128, dtype=torch.bfloat16)
    b = _randn(128, 128, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 64, 192, 384, 640], device=DEVICE, dtype=torch.int32)
    kernel = helion.kernel(
        grouped_mod.grouped_gemm_jagged.fn, backend="cute", static_shapes=False
    )
    bound = kernel.bind((a, b, offsets))
    config = _config(block_sizes, warps)
    bound.set_config(config)
    with bound.env:
        assert bound.config_spec._cute_tcgen05_config.warp_mma_admitted
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "_helion_warp_mma.mma_bf16(" in source
    assert "group_offsets.iterator + cutlass.Int64(" in source
    assert "tcgen05_tmem_allocator" not in source
    for layout in (
        [0, 64, 192, 384, 640],
        [0, 100, 100, 300, 640],
        [0, 64, 64, 200, 500],
    ):
        offs = torch.tensor(layout, device=DEVICE, dtype=torch.int32)
        ref = _grouped_reference(a, b, offs)
        covered = ~torch.isnan(ref)
        outs = [bound(a, b, offs) for _ in range(3)]
        torch.cuda.synchronize()
        for other in outs[1:]:
            assert torch.equal(outs[0][covered], other[covered])
        _check(outs[0][covered], ref[covered].float())


def test_grouped_row_union_rebinds_per_shape_and_fails_closed_off_the_family() -> None:
    """``static_shapes=False``: the family is admitted from each binding's
    hints and codegen specializes ``(total_M, G)`` under shape guards, so a
    call with another packed size or group count rebinds (fresh heuristic,
    fresh body) instead of reusing the first binding's clamps.  A shape the
    family does not run (1000 rows) rejects the pinned family strictly and
    repairs to the tcgen05 lowering, which stays correct."""
    import examples.grouped_gemm as grouped_mod

    torch.manual_seed(0)
    b = _randn(128, 128, dtype=torch.bfloat16)
    kernel = helion.kernel(
        grouped_mod.grouped_gemm_jagged.fn, backend="cute", static_shapes=False
    )
    config = _config([32, 32, 64], 4)
    seen: dict[int, int] = {}
    for total_m, layout in (
        (640, [0, 64, 192, 384, 640]),
        (1024, [0, 100, 100, 300, 1024]),
        (768, [0, 64, 64, 200, 500, 700, 768]),
        (640, [0, 10, 300, 300, 640]),
    ):
        a = _randn(total_m, 128, dtype=torch.bfloat16)
        offs = torch.tensor(layout, device=DEVICE, dtype=torch.int32)
        bound = kernel.bind((a, b, offs))
        # One bound kernel per (total_M, G); the same shape reuses it.
        assert seen.setdefault(id(bound), total_m) == total_m
        with bound.env:
            state = bound.config_spec._cute_tcgen05_config
            assert state.warp_mma_admitted
            assert state.warp_mma_grouped_extents == (total_m, 128, 128)
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.mma_bf16(" in source
        assert f"cutlass.Int32({total_m})" in source
        assert source.count("group_offsets.iterator + cutlass.Int64(") == len(layout)
        bound.set_config(normalized)
        out = bound(a, b, offs)
        torch.cuda.synchronize()
        ref = _grouped_reference(a, b, offs)
        covered = ~torch.isnan(ref)
        _check(out[covered], ref[covered].float())
    assert len(seen) == 3

    a = _randn(1000, 128, dtype=torch.bfloat16)
    offs = torch.tensor([0, 100, 100, 300, 1000], device=DEVICE, dtype=torch.int32)
    bound = kernel.bind((a, b, offs))
    assert id(bound) not in seen
    with bound.env:
        assert not bound.config_spec._cute_tcgen05_config.warp_mma_admitted
        with pytest.raises(helion.exc.InvalidConfig, match="not admitted"):
            bound.config_spec.normalized_config(config)
        repaired = dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
    assert repaired[FAMILY] == family.MATMUL_FAMILY_TCGEN05
    bound.set_config(helion.Config(**repaired))
    assert "_helion_warp_mma" not in bound.to_code(bound._config)
    out = bound(a, b, offs)
    torch.cuda.synchronize()
    ref = _grouped_reference(a, b, offs)
    covered = ~torch.isnan(ref)
    _check(out[covered], ref[covered].float())


def test_gelu_aux_epilogue_matches_torch() -> None:
    # The cuBLASLt GELU_AUX pattern (bias prefix, two outputs, the chain's
    # ``_gelu_erf`` helper) on the family: the exact erf on per-fragment
    # scalars against torch's gelu of the fp32 pre-activation.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    bias = _randn(256, dtype=torch.float16)
    pre = torch.matmul(a.float(), b.float()) * 1.25 + bias.float()
    for block_sizes, warps in (([64, 16, 64], 1), ([32, 32, 128], 4)):
        (out, aux), source = _run(
            _bias_gelu_aux_gemm, (a, b, bias), _config(block_sizes, warps)
        )
        assert "gelu_erf" in source
        _check(aux, pre)
        _check(out, torch.nn.functional.gelu(pre))


def test_m_major_int16_cast_operand_matches_torch() -> None:
    # ``x[tile_m, tile_k].to(bf16)`` of an M-contiguous int16 A: the packets
    # are loaded as words, converted in registers into the M-major staging
    # and read by ``ldmatrix.trans``.
    import examples.bf16xint16_gemm as example

    torch.manual_seed(0)
    x = torch.randint(-8, 9, (256, 128), device=DEVICE, dtype=torch.int16).T
    w = _randn(256, 128, dtype=torch.bfloat16)
    assert x.shape == (128, 256) and x.stride() == (1, 128)
    kernel = helion.kernel(example._int16xbf16_gemm.fn, backend="cute")
    bound = kernel.bind((x, w))
    with bound.env:
        state = bound.config_spec._cute_tcgen05_config
        assert state.warp_mma_admitted and not state.tcgen05_family_admitted
    for block_sizes, warps in (([16, 8, 128], 1), ([32, 16, 64], 2)):
        config = _config(block_sizes, warps)
        bound.set_config(config)
        with bound.env:
            normalized = bound.config_spec.normalized_config(config)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.cvt_s16x8_bf16_sts(wm_a" in source
        assert "_helion_warp_mma.ldmatrix_x4_trans(" in source
        out = bound(x, w)
        torch.cuda.synchronize()
        _check(out, torch.matmul(x.float(), w.float()))


@pytest.mark.parametrize(
    "kernel_fn", [_extra_root_store_after, _extra_root_store_before]
)
def test_extra_root_store_fails_closed_on_the_family(kernel_fn) -> None:
    # The family refuses a kernel whose root tile loop holds a store it did
    # not match instead of dropping it.  (The default tcgen05 config of this
    # kernel fails to compile at the base commit as well -- ``indices_0`` is
    # undefined for the SIMT store next to the role body -- a pre-existing
    # defect outside this family, recorded in the lane report.)
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    other = _randn(256, 256, dtype=torch.float16)
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b, other))
    with pytest.raises(
        BackendUnsupported, match="root tile loop may only hold the GEMM"
    ):
        bound.set_config(_config([16, 8, 128], 1))
        bound(a, b, other)


def test_in_place_residual_on_strided_output_views() -> None:
    # The output is also the residual aux: a 2-byte-offset view (element
    # loads / stores, row stride 516 B) and an 8-byte-offset view (pairs)
    # must both be read before they are written, in place, no staging copy.
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    ref = torch.matmul(a.float(), b.float())
    kernel = helion.kernel(_inplace_residual, backend="cute", static_shapes=True)
    for width, offset, block_sizes, warps in (
        (258, 1, [16, 8, 128], 1),
        (264, 4, [32, 16, 128], 2),
        (256, 0, [64, 8, 128], 4),
    ):
        full = _randn(256, width, dtype=torch.float16)
        view = full[:, offset : offset + 256]
        before = full.clone()
        bound = kernel.bind((a, b, view))
        bound.set_config(_config(block_sizes, warps))
        result = bound(a, b, view)
        torch.cuda.synchronize()
        _check(result, ref + before[:, offset : offset + 256].float())
        # Columns outside the view are untouched.
        assert torch.equal(full[:, :offset], before[:, :offset])
        assert torch.equal(full[:, offset + 256 :], before[:, offset + 256 :])


def test_sliced_output_leaves_the_rest() -> None:
    torch.manual_seed(0)
    a = _randn(256, 256, dtype=torch.float16)
    b = _randn(256, 256, dtype=torch.float16)
    full = _randn(256, 512, dtype=torch.float16)
    keep = full[:, 256:].clone()
    kernel = helion.kernel(_sliced_output, backend="cute", static_shapes=True)
    bound = kernel.bind((a, b, full))
    bound.set_config(_config([16, 8, 128], 1))
    result = bound(a, b, full)
    torch.cuda.synchronize()
    _check(result[:, :256], torch.matmul(a.float(), b.float()))
    assert torch.equal(result[:, 256:], keep)


def test_bf16xint16_materialized_operand_on_the_family() -> None:
    # The fissioned bf16 x int16 example: region 1 is a plain bf16 GEMM over
    # the materialized operand, admitted (and seeded) at (128, 256, 128).
    import examples.bf16xint16_gemm as example

    torch.manual_seed(0)
    x = _randn(128, 256, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (256, 128), device=DEVICE, dtype=torch.int16)
    kernel = helion.kernel(
        example._bf16xint16_gemm.fn, backend="cute", static_shapes=True
    )
    bound = kernel.bind((x, w))
    with bound.env:
        seeds = [
            seed
            for seed in bound.config_spec._cute_tcgen05_config.autotune_seed_configs()
            if seed.config.get(FAMILY) == family.MATMUL_FAMILY_WARP_MMA
        ]
    assert seeds
    ref = torch.matmul(x.float(), w.to(torch.bfloat16).float())
    for seed in seeds[:3]:
        bound = kernel.bind((x, w))
        bound.set_config(seed)
        with bound.env:
            normalized = bound.config_spec.normalized_config(seed)
        source = bound.to_code(normalized)
        assert "_helion_warp_mma.mma_bf16(" in source
        out = bound(x, w)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
