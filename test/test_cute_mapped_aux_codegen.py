from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mapped_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.size(1)
    for rows, cols in hl.tile([m, n]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[rows, reduction], b[reduction, cols])
        first[rows, cols] = acc.to(first.dtype)
        value = aux[
            rows.index.reshape([rows.index.size(0), 1]) // aux.size(1),
            rows.index.reshape([rows.index.size(0), 1]) % aux.size(1),
            cols.index.reshape([1, cols.index.size(0)]),
        ].to(torch.float32)
        second[rows, cols] = (acc * value).to(second.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mapped_cols(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.size(1)
    for rows, cols in hl.tile([m, n]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for reduction in hl.tile(k):
            acc = torch.addmm(acc, a[rows, reduction], b[reduction, cols])
        value = aux[
            cols.index.reshape([1, cols.index.size(0)]) // aux.size(1),
            cols.index.reshape([1, cols.index.size(0)]) % aux.size(1),
            rows.index.reshape([rows.index.size(0), 1]),
        ].to(torch.float32)
        out[rows, cols] = (acc * value).to(out.dtype)


def _row_args(dtype: torch.dtype, period: int = 128) -> tuple[torch.Tensor, ...]:
    m = 2 * period
    a = torch.empty((m, 64), dtype=dtype)
    b = torch.empty_strided((64, 96), (1, 64), dtype=dtype)
    aux = torch.empty_strided((2, period, 96), (period * 96, 1, period), dtype=dtype)
    first = torch.empty((m, 96), dtype=dtype)
    second = torch.empty((m, 96), dtype=dtype)
    return a, b, aux, first, second


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mapped_row_codegen_default_and_saved_collective(dtype):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, _row_args(dtype))
        assert bound.config_spec.cute_tcgen05_search_enabled
        for config in (
            bound.config_spec.default_config(),
            bound.config_spec._base_default_config(),
            helion.Config(block_sizes=[128, 32, 64], pid_type="flat"),
        ):
            source = bound.to_code(config)
            assert "tcgen05_ab_tma" in source
            assert "cute.crd2idx((cutlass.Int32(" in source
            assert "tcgen05_aux_view2d" in source
            assert ".layout.stride[1]" in source and ".layout.stride[2]" in source
            assert "cute.local_tile(tcgen05_aux_view2d" in source
            assert "cute.make_layout((aux.shape[1], aux.shape[2])" in source
        source = bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_collective_mma=True,
                cute_collective_compute="tcgen05",
                cute_collective_tmem_a=True,
            )
        )
        assert "collective_mma" in source
        assert "tcgen05_ab_tma" not in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mapped_col_codegen_uses_original_axis_order(dtype):
    a = torch.empty((128, 64), dtype=dtype)
    b = torch.empty_strided((64, 256), (1, 64), dtype=dtype)
    aux = torch.empty_strided((2, 128, 128), (16384, 1, 128), dtype=dtype)
    out = torch.empty((128, 256), dtype=dtype)
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_cols, (a, b, aux, out))
        source = bound.to_code(
            helion.Config(block_sizes=[128, 32, 64], pid_type="flat")
        )
        assert "tcgen05_ab_tma" in source
        assert ".shape[2]" in source and ".shape[1]" in source
        assert "% 4" in source


@pytest.mark.parametrize(
    ("period", "blocks"), ((192, [128, 32, 64]), (128, [128, 64, 64]))
)
def test_mapped_aux_tile_boundary_and_partial_n_are_rejected(period, blocks):
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, _row_args(torch.bfloat16, period))
        with pytest.raises(exc.InvalidConfig, match="mapped auxiliary"):
            bound.to_code(helion.Config(block_sizes=blocks, pid_type="flat"))


def test_mapped_aux_without_any_complete_native_grid_keeps_fallback():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, _row_args(torch.bfloat16, period=96))
        assert not bound.config_spec.cute_tcgen05_search_enabled


def test_mapped_aux_does_not_claim_unknown_or_overlapping_alias_facts():
    a, b, aux, _first, second = _row_args(torch.bfloat16)
    # The destination really aliases the source; this is not FakeTensor
    # inequality or a made-up dispatch fact. The complete DAG proof must fail.
    first = torch.as_strided(aux, (256, 96), (96, 1))
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, (a, b, aux, first, second))
        assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("n_contiguous", (False, True))
def test_mapped_aux_register_mma_family_is_not_admitted(n_contiguous):
    a, b, aux, first, second = _row_args(torch.bfloat16)
    if n_contiguous:
        aux = torch.empty((2, 128, 96), dtype=torch.bfloat16)
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, (a, b, aux, first, second))
        with pytest.raises(
            (exc.InvalidConfig, exc.BackendUnsupported), match="warp_mma"
        ):
            bound.to_code(
                helion.Config(
                    block_sizes=[32, 16, 16],
                    pid_type="flat",
                    cute_matmul_family="warp_mma",
                    cute_warp_mma_warps=4,
                )
            )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mapped_aux_managed_binding_uses_normal_complete_facts(dtype):
    with (
        cpu_codegen_target(),
        forbid_cuda_work(),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch("helion._compat._is_hip", return_value=False),
    ):
        bound = _mapped_rows.bind(_row_args(dtype))
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(
            helion.Config(block_sizes=[128, 32, 64], pid_type="flat")
        )
        assert "tcgen05_ab_tma" in source
        assert "cute.local_tile(tcgen05_aux_view2d" in source


def test_mapped_aux_unsupported_hoist_flag_is_not_ignored():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = cpu_bind(_mapped_rows, _row_args(torch.bfloat16))
        with pytest.raises(exc.InvalidConfig):
            bound.to_code(
                helion.Config(
                    block_sizes=[128, 32, 64],
                    pid_type="flat",
                    tcgen05_aux_fragment_hoist=True,
                )
            )
