"""CPU-only tests of the CuTe unaligned-operand staging pass.

The pass (``helion/_compiler/cute/stage_aligned_operands.py``) pads matmul
operands whose strides fail the TensorMap proof through a host copy so the
tcgen05 pipeline keeps its TMA producers. These tests bind GEMMs on CPU
tensors under the fake sm_100 target and inspect the host function and the
rendered host prologue; no GPU is used.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_rhs_grouped import _target
from test.test_cute_tcgen05_ordinary_exploration import _ordinary_gemm

import helion
from helion._compiler.cute.stage_aligned_operands import _needs_staging
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cuda_fake
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_cp_async_a import gemm_into
from helion._testing.cute_cp_async_a import keyword_addmm_gemm
from helion._testing.cute_cp_async_a import residual_gemm
from helion._testing.cute_cp_async_a import two_output_gemm
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends("cute")]

CPU_DEVICE = "cpu"


def _inplace_gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _dot_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            x_tile = x[tile_m, tile_k]
            y_tile = y[tile_k, tile_n]
            acc = hl.dot(x_tile, y_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _inplace_through_alias(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # ``a`` is a matmul operand and is written through the host alias
    # ``scratch``: it must stay the caller's tensor, not a padded copy.
    m, k = a.size()
    _, n = b.size()
    scratch = a.detach()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _merged_view_operands(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Operands reached through ``view`` / ``contiguous`` are not staged: a
    # view that merges the padded axis fails on the slice and ``contiguous``
    # already copies.
    m, k = a.size()
    _, n = b.size()
    a2 = a.reshape(-1).view(m, k)
    b2 = b.contiguous()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a2[tile_m, tile_k], b2[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _annotated_alias_store(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    scratch: torch.Tensor = a.detach()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _tuple_alias_store(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    scratch, other = a, b
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], other[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        scratch[tile_m, tile_n] = acc.to(a.dtype)
    return out


def _hl_store_into_operand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
        hl.store(a, [tile_m, tile_n], acc.to(a.dtype))
    return out


def _dict_return(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return {"out": out, "a": a}


def _layout_query(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    assert a.is_contiguous()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _bias_rows(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # No matmul: an unaligned bias operand is read, never staged.
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_m, tile_n]
    return out


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HELION_BACKEND", "cute")
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "full")
    with (
        _target(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        mock_cuda_unavailable(),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=cuda_fake,
        ),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda", index=0, cc=100, warp_size=32, multi_processor_count=148
            ),
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(fn: object, *args: torch.Tensor, **settings: object) -> BoundKernel:
    return helion.kernel(fn, backend="cute", static_shapes=True, **settings).bind(args)  # pyrefly: ignore[bad-argument-type]


def _staging_statements(bound: BoundKernel) -> list[str]:
    return [
        ast.unparse(statement)
        for statement in bound.host_function.body
        if isinstance(statement, ast.Assign)
        and "constant_pad_nd" in ast.unparse(statement)
    ]


def test_needs_staging_rule() -> None:
    aligned = torch.empty((64, 4096), dtype=torch.bfloat16)
    assert _needs_staging(aligned) is None
    assert _needs_staging(aligned.T) is None
    # Odd bf16 rows: 8194-byte stride; padded to 4160 elements (128-byte rows).
    odd = torch.empty((4097, 4097), dtype=torch.bfloat16)
    assert _needs_staging(odd) == 63
    # An aligned extent that still fails the proof pads a whole block so the
    # copy is never a zero-pad alias.
    assert _needs_staging(torch.empty((4096, 4097), dtype=torch.bfloat16).T) == 64
    # fp8: 128-byte rows need 128-element multiples.
    assert _needs_staging(torch.empty((128, 4100), dtype=torch.float8_e4m3fn)) == 124
    # Under-aligned base offset.
    assert _needs_staging(torch.empty((64, 4096), dtype=torch.bfloat16)[1:]) is None
    assert _needs_staging(torch.empty((128, 4097), dtype=torch.bfloat16)[1:]) == 63
    assert (
        _needs_staging(torch.empty((8, 64, 4096), dtype=torch.bfloat16)[:, :, 1:]) == 1
    )
    # Rank-1 and rank-4 operands and axes of extent one are ignored.
    assert _needs_staging(torch.empty(4097, dtype=torch.bfloat16)) is None
    assert _needs_staging(torch.empty((2, 4, 1024, 68), dtype=torch.bfloat16)) is None
    # Small operands keep the scalar producers (STAGING_MIN_ELEMENTS).
    assert _needs_staging(torch.empty((16, 4100), dtype=torch.float8_e4m3fn)) is None
    assert _needs_staging(torch.empty((63, 4097), dtype=torch.bfloat16)) is None
    assert _needs_staging(torch.empty((1, 4096), dtype=torch.bfloat16)) is None


def test_odd_gemm_operands_are_staged_and_tma_proven() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, b)
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 63])[:, :1025]",
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]",
    ]
    assert bound.host_function.backend_rewrote_ast
    # The TMA producers the staged layout unlocks are checked on a GPU in
    # test_cute_stage_aligned_operands_gpu.py; the CPU target renders the
    # host prologue only.
    code = bound.to_code(bound.config_spec.default_config())
    assert "constant_pad_nd(a, [0, 63])" in code


def test_only_the_unaligned_operand_is_staged() -> None:
    a = torch.empty((1024, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1024), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_dot_gemm, a, b)
    # ``x`` has 2050-byte rows; ``y`` has 2048-byte rows and stays in place.
    assert _staging_statements(bound) == [
        "x = torch.constant_pad_nd(x, [0, 63])[:, :1025]"
    ]


def test_aligned_gemm_is_left_alone() -> None:
    a = torch.empty((1024, 1024), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, a)
    assert _staging_statements(bound) == []
    assert not bound.host_function.backend_rewrote_ast


def test_written_operand_and_non_matmul_reads_are_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    out = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_inplace_gemm, a, a, out)
    # ``out`` is written and returned: never staged even though unaligned.
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 63])[:, :1025]",
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]",
    ]
    bias = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_bias_rows, bias, bias)
    assert _staging_statements(bound) == []


def test_operand_written_through_an_alias_is_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_inplace_through_alias, a, b)
    assert _staging_statements(bound) == [
        "b = torch.constant_pad_nd(b, [0, 63])[:, :1025]"
    ]


@pytest.mark.parametrize(
    "kernel,staged",
    (
        (_annotated_alias_store, ["b"]),
        # The tuple unpacking aliases both operands.
        (_tuple_alias_store, []),
        (_hl_store_into_operand, ["b"]),
        (_dict_return, ["b"]),
        (_layout_query, ["b"]),
    ),
    ids=lambda value: value.__name__ if callable(value) else "-".join(value) or "none",
)
def test_other_mentions_of_an_operand_keep_it_unstaged(kernel, staged) -> None:
    # Any mention of ``a`` beyond tile reads, metadata and the matmul operand
    # itself (an annotated or tuple alias that is stored into, an ``hl.store``
    # into it, a ``return`` of it, a layout query the copy would answer
    # differently) keeps the argument on its original tensor; an operand
    # mentioned only in those forms is still staged.
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(kernel, a, b)
    assert _staging_statements(bound) == [
        f"{name} = torch.constant_pad_nd({name}, [0, 63])[:, :1025]" for name in staged
    ]


def test_operands_behind_merging_views_are_not_staged() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_merged_view_operands, a, b)
    assert _staging_statements(bound) == []


def test_setting_disables_the_pass() -> None:
    a = torch.empty((1025, 1025), dtype=torch.bfloat16, device=CPU_DEVICE)
    bound = _bind(_ordinary_gemm, a, a, cute_stage_unaligned_operands=False)
    assert _staging_statements(bound) == []
    assert "cute_stage_unaligned_operands=False" in bound.format_kernel_decorator(
        bound.config_spec.default_config(), bound.settings
    )


def _x_transposed_args(m: int, n: int) -> tuple[torch.Tensor, ...]:
    # ``x.T`` of a [512, m] activation: M-contiguous A with 2 * m-byte K rows.
    x = torch.empty((512, m), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((512, n), dtype=torch.bfloat16, device=CPU_DEVICE)
    residual = torch.empty((m, n), dtype=torch.float32, device=CPU_DEVICE)
    return x.T, b, residual


def test_narrow_gemm_copies_an_m_major_a_with_cp_async() -> None:
    # 2036-byte K rows: no TensorMap can address A. Up to N=256 the epilogue
    # warps copy it with cp.async in the kernel instead of the host staging a
    # transposing copy:
    #     cute.copy(tcgen05_cpa_0_0_tc, tcgen05_cpa_0_0_g[None, None, None, k],
    #               tcgen05_cpa_0_0_s[None, None, None, tcgen05_cpa_state.index],
    #               pred=tcgen05_cpa_0_0_p)
    #     cute.arch.cp_async_mbarrier_arrive_noinc(tcgen05_cpa_full + ...)
    bound = _bind(residual_gemm, *_x_transposed_args(1018, 128))
    assert _staging_statements(bound) == []
    assert bound.config_spec.cute_tcgen05_cp_async_a_provable
    config = bound.config_spec.default_config()
    # 4 paired tiles would leave most of 148 SMs idle: 128-row tiles.
    assert config.config["block_sizes"] == [128, 128, 64]
    assert config.config["pid_type"] == "flat"
    code = bound.to_code(config)
    assert "constant_pad_nd" not in code
    assert "'a_cp_async': True" in code
    assert "tma_atom_a" not in code
    # The copies write the stage through the generic proxy, the UMMA reads it
    # through the async proxy: the MMA warp fences after its wait.
    wait = code.index("cute.arch.mbarrier_wait(tcgen05_cpa_full")
    assert code[wait:].split("\n", 2)[1].strip() == (
        "cute.arch.fence_view_async_shared()"
    )
    # Rows k % 4 == 0 start 16-byte aligned, k % 4 == 2 8-byte aligned, the
    # odd rows 4-byte aligned: one copy width per row class.
    for bits in (128, 64, 32):
        assert f"num_bits_per_copy={bits}" in code
    # one arrive ahead of the K loop per stage, one refill in it
    assert (
        code.count("cp_async_mbarrier_arrive_noinc")
        == 1 + config.config["tcgen05_ab_stages"]
    )


def test_cp_async_a_pairs_m_subtiles_on_one_b_stage() -> None:
    bound = _bind(residual_gemm, *_x_transposed_args(1018, 128))
    config = helion.Config(
        **{
            **bound.config_spec.default_config().config,
            "block_sizes": [256, 128, 64],
            "tcgen05_ab_stages": 4,
        }
    )
    code = bound.to_code(config)
    # Both 128-row subtiles' UMMAs read the stage's B, the second through a
    # fragment of its own, in one statement (no CTA barrier between them):
    #     if tcgen05_exec_active:
    #         for _tcgen05_kblk_idx in range(...):
    #             cute.gemm(tiled_mma, acc_frag, [tcgen05_tCrA[...]],
    #                       [tcgen05_tCrB[...]], acc_frag)
    #         for _tcgen05_kblk_idx in range(...):
    #             cute.gemm(tcgen05_msub_tiled_mma, tcgen05_msub_acc_frag,
    #                       [tcgen05_msub_tCrA[...]], [tcgen05_msub_tCrB[...]], ...)
    assert "tcgen05_msub_tCrB = tiled_mma.make_fragment_B(sB)" in code
    loop_start = code.index("for tile_offset_2 in")
    loop = code[
        loop_start : code.index("tcgen05_acc_pipeline.producer_commit", loop_start)
    ]
    assert "tcgen05_msub_tCrB[" in loop
    assert "sync_threads" not in loop
    # The epilogue drains each subtile at its own rows.
    assert "for _tcgen05_msub in cutlass.range(2, unroll_full=True):" in code
    assert (
        "tcgen05_full_tile = tile_offset_0 + cutlass.Int32(_tcgen05_msub) * "
        "cutlass.Int32(128) + cutlass.Int32(128) <= 1018"
    ) in code


def test_cp_async_a_needs_a_16_byte_aligned_bound_base() -> None:
    # ``x[:, 1:1019].T`` starts 2 bytes into ``x``: the producer's 16-byte
    # copies cannot address it, so the host stages A as before (the fake
    # tensor's storage offset is zero; the bound pointer decides).
    x = torch.empty((512, 1020), dtype=torch.bfloat16, device=CPU_DEVICE)
    _, b, residual = _x_transposed_args(1018, 128)
    bound = _bind(residual_gemm, x[:, 1:1019].T, b, residual)
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 64])[:, :512]"
    ]
    assert not bound.config_spec.cute_tcgen05_cp_async_a_provable


def test_cp_async_a_runs_beside_an_output_no_tensormap_addresses() -> None:
    # 520-byte output rows: the output takes the SIMT store, A still the
    # cp.async producer (the scalar producers ran this 0.54 ms -> 47 ms).
    a, b, _ = _x_transposed_args(1018, 128)
    out = torch.empty((1018, 130), dtype=torch.float32, device=CPU_DEVICE)[:, :128]
    bound = _bind(gemm_into, a, b, out)
    assert _staging_statements(bound) == []
    assert bound.config_spec.cute_tcgen05_cp_async_a_provable
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128, 64]
    assert "'a_cp_async': True" in bound.to_code(config)


def test_cp_async_a_keeps_128_row_tiles_beside_several_outputs() -> None:
    # 239 128-row tiles would pair, but the fanout of two output stores has
    # no paired M subtiles: the default and a projected sample keep 128 rows.
    a, b, _ = _x_transposed_args(30522, 128)
    bound = _bind(two_output_gemm, a, b)
    assert bound.config_spec.cute_tcgen05_cp_async_a_provable
    config = bound.config_spec.default_config()
    assert config.config["block_sizes"] == [128, 128, 64]
    assert "'a_cp_async': True" in bound.to_code(config)
    sample = {**config.config, "block_sizes": [256, 128, 128]}
    bound.config_spec.normalize(sample, _fix_invalid=True)
    assert sample["block_sizes"] == [128, 128, 64]


def test_cp_async_a_needs_the_flat_edge_tma_path() -> None:
    # The producer runs whole tiles or the flat-edge TMA path: without the
    # path the host stages A over the 1018-row edge and the default compiles.
    bound = _bind(
        residual_gemm, *_x_transposed_args(1018, 128), cute_flat_edge_tma=False
    )
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 64])[:, :512]"
    ]
    assert not bound.config_spec.cute_tcgen05_cp_async_a_provable
    bound.to_code(bound.config_spec.default_config())


def test_keyword_matmul_operands_keep_their_roles() -> None:
    # ``mat2=`` written before ``mat1=``: B is the unaligned operand (260-byte
    # K-major rows), and the host stages it rather than taking it for A.
    a = torch.empty((1024, 136), dtype=torch.bfloat16, device=CPU_DEVICE)[:, :130]
    b = torch.empty((2048, 130), dtype=torch.bfloat16, device=CPU_DEVICE).T
    bound = _bind(keyword_addmm_gemm, a, b)
    assert [statement.split(" = ")[0] for statement in _staging_statements(bound)] == [
        "b"
    ]


def test_cp_async_a_pairs_m_subtiles_in_both_accumulator_stages() -> None:
    # A one-stage sample of the 256-row tile takes the second stage: its two
    # subtiles' accumulators fill both (an explicit one-stage config is
    # refused).
    bound = _bind(residual_gemm, *_x_transposed_args(30522, 128))
    sample = {
        **bound.config_spec.default_config().config,
        "block_sizes": [256, 128, 64],
        "tcgen05_acc_stages": 1,
    }
    with pytest.raises(helion.exc.InvalidConfig, match="cp.async producer"):
        bound.to_code(helion.Config(**sample))
    bound.config_spec.normalize(sample, _fix_invalid=True)
    assert sample["block_sizes"] == [256, 128, 64]
    assert sample["tcgen05_acc_stages"] == 2
    assert "'a_cp_async': True" in bound.to_code(helion.Config(**sample))


def test_small_unstaged_a_keeps_scalar_producers_without_flat_edge_tma() -> None:
    # Below ``STAGING_MIN_ELEMENTS`` A stays in place whatever the producer;
    # without the flat-edge TMA path the 1018-row edge keeps the scalar
    # producers instead of the cp.async copies, and the config compiles.
    x = torch.empty((192, 1018), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((192, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    residual = torch.empty((1018, 128), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind(residual_gemm, x.T, b, residual, cute_flat_edge_tma=False)
    assert _staging_statements(bound) == []
    # the 256-row tile too: the MMA plan's prediction agrees with codegen
    for block_m in (128, 256):
        config = helion.Config(
            **{
                **bound.config_spec.default_config().config,
                "block_sizes": [block_m, 128, 64],
                "pid_type": "flat",
                "tcgen05_acc_stages": 2,
            }
        )
        assert "cp_async_mbarrier_arrive_noinc" not in bound.to_code(config)


def test_explicit_configs_stay_allowed_where_staging_would_not_pad_a() -> None:
    # A 192-deep A sits below the staging minimum: the host never padded it,
    # so an explicit config off the cp.async envelope keeps the scalar
    # producers it ran on rather than being refused.
    x = torch.empty((192, 1018), dtype=torch.bfloat16, device=CPU_DEVICE)
    b = torch.empty((192, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
    residual = torch.empty((1018, 128), dtype=torch.float32, device=CPU_DEVICE)
    bound = _bind(residual_gemm, x.T, b, residual)
    spec = bound.config_spec
    assert spec.cute_tcgen05_cp_async_a_provable
    assert not spec.cute_tcgen05_cp_async_a_unstaged
    config = helion.Config(
        **{
            **spec.default_config().config,
            "block_sizes": [128, 128, 128],
            "tcgen05_ab_stages": 2,
        }
    )
    assert "cp_async_mbarrier_arrive_noinc" not in bound.to_code(config)


def test_wide_gemm_keeps_staging_an_m_major_a() -> None:
    # Five 128-wide tile columns each read A: the staged copy pays for itself.
    bound = _bind(residual_gemm, *_x_transposed_args(1018, 640))
    # The host relays A out K-major (the staging pass's row-major copy).
    assert _staging_statements(bound) == [
        "a = torch.constant_pad_nd(a, [0, 64])[:, :512]"
    ]
    assert not bound.config_spec.cute_tcgen05_cp_async_a_provable


@pytest.mark.parametrize(
    "overrides",
    (
        {"tcgen05_split_k": 2},
        {"tcgen05_layout_overrides_smem_swizzle_a": 64},
        {"tcgen05_strategy": "pure_matmul_role_lifecycle"},
    ),
    ids=("split_k", "swizzle_a", "pure_lifecycle"),
)
def test_cp_async_a_stays_off_configs_its_views_do_not_cover(
    overrides: dict[str, object],
) -> None:
    # A split K (each split's own k-blocks), another A swizzle (the copies
    # address the SW128 stage) or the pure lifecycle's separate TMA producer
    # (no refill in the shared K loop) would load the unstaged A with the
    # scalar producers: an explicit config is refused, a search sample is
    # projected onto the cp.async envelope.
    if overrides.get("tcgen05_strategy"):
        # the pure lifecycle runs whole tiles of a plain GEMM: 1024 of 1026
        # columns
        x = torch.empty((512, 1026), dtype=torch.bfloat16, device=CPU_DEVICE)
        b = torch.empty((512, 128), dtype=torch.bfloat16, device=CPU_DEVICE)
        bound = _bind(_dot_gemm, x[:, :1024].T, b)
    else:
        bound = _bind(residual_gemm, *_x_transposed_args(1018, 128))
    values = {**bound.config_spec.default_config().config, **overrides}
    with pytest.raises(helion.exc.InvalidConfig, match="cp.async producer"):
        bound.to_code(helion.Config(**values))
    bound.config_spec.normalize(values, _fix_invalid=True)
    assert "cp_async_mbarrier_arrive_noinc" in bound.to_code(helion.Config(**values))
    for key, value in overrides.items():
        assert values.get(key) != value
    code = bound.to_code(helion.Config(**values))
    assert "cp_async_mbarrier_arrive_noinc" in code
