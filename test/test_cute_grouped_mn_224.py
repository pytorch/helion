from __future__ import annotations

import ast
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_d_shift import _assignment
from test.test_cute_grouped_d_shift import _config
from test.test_cute_grouped_d_shift import _plans
from test.test_cute_grouped_d_shift import _unsafe_output
from test.test_cute_grouped_d_shift import _value
from test.test_cute_grouped_paired_clc_layout import _sdk

import helion
from helion._compiler.cute.cute_mma import _tcgen05_grouped_b_rank3_index_domain
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES,
)
from helion._compiler.cute.tcgen05_constants import (
    tcgen05_grouped_worklist_epilogue_shape,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    variant: int = 0,
) -> torch.Tensor:
    hl.specialize(variant)
    m, k = a.shape
    groups, n, _ = b.shape
    bm = hl.register_block_size(256)
    bn = hl.register_block_size(128)
    bk = hl.register_block_size(64)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for expert, rows, columns in hl.tile([groups, m, n], block_size=[1, bm, bn]):
        group = expert.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - start
        indices = start + rows.index
        valid = rows.index < length
        acc = hl.full(
            [rows, columns], 1.0 if variant == 1 else 0.0, dtype=torch.float32
        )
        for inner in hl.tile(k, block_size=bk):
            left = hl.load(a, [indices, inner], extra_mask=valid[:, None])
            if variant == 2:
                left = left * 2
            acc = torch.addmm(acc, left, b[group, columns, inner].T)
            if variant == 3:
                acc = acc + 1
        hl.store(out, [indices, columns], acc.to(out.dtype), extra_mask=valid[:, None])
    return out


def _profile(*, depth: int = 7) -> helion.Config:
    config = _config("mn")
    config.config.update(
        tcgen05_grouped_worklist_source_m_tile=256,
        tcgen05_grouped_worklist_mma_n=224,
        tcgen05_ab_stages=depth,
        tcgen05_consumer_regs=256,
    )
    return config


def _args(*, n: int = 512, groups: int = 8, dtype=torch.int64):
    return (
        torch.empty((640, 256), dtype=torch.bfloat16),
        torch.empty((groups, n, 256), dtype=torch.bfloat16),
        torch.arange(groups + 1, dtype=dtype) * (640 // groups),
    )


def _source(args=None, config=None):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args() if args is None else args)
        config = bound.config_spec.normalized_config(
            _profile() if config is None else config
        )
        source = bound.to_code(config)
        assert config.block_sizes == [256, 128, 64]
        assert bound.config_spec.block_sizes[2].declared_max_size == 64
        return source


def _footprint(*, groups: int = 16, immutable: bool = True, **options) -> int:
    values = {
        "group_count": groups,
        "device_split_sizes": True,
        "sched_stage_count": 1,
        "bm": 256,
        "bn": 224,
        "bk": 64,
        "dtype_bytes": 2,
        "output_dtype_bytes": 2,
        "ab_stages": 7,
        "acc_stages": 2,
        "c_stages": 2,
        "cluster_m": 2,
        "orientation": "mn",
        "immutable_ab_tensormaps": immutable,
        "immutable_d_tensormap": immutable,
    }
    return tcgen05_grouped_worklist_smem_bytes(**(values | options))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_real_medium_metadata_normalizes_at_capacity_without_widening_logical_k(dtype):
    args = (
        torch.empty((8192, 2048), dtype=torch.bfloat16),
        torch.empty((16, 1536, 2048), dtype=torch.bfloat16),
        torch.arange(17, dtype=dtype) * 512,
    )
    source = _source(args)
    grouped, ab, output = _plans(source)
    assert (ab["bm"], ab["bn"], ab["bk"], ab["ab_stage_count"]) == (256, 224, 64, 7)
    assert grouped["immutable_ab_tensormaps"] is True
    assert ab["full_allocation_b_rank3"] is True
    assert output["bounded_output_shift"] is True
    assert output["c_stage_count"] == 2
    assert "TensorMapManager" not in source
    assert "update_tensormap" not in source
    assert "tcgen05_grouped_ab_tensormaps" not in source
    assert _footprint() == 232448


@pytest.mark.parametrize("n", [160, 512, 544, 672])
def test_exact_expert_extent_masks_feature_padding_without_crossing_expert(n):
    source = _source(_args(n=n))
    tree = ast.parse(source)
    _, ab, _ = _plans(source)
    rank3 = n % 224 != 0
    assert bool(ab.get("full_allocation_b_rank3")) is rank3
    body, arguments = [], []
    _append_cute_wrapper_plan(body, arguments, ab)
    host = ast.parse(dedent("\n".join(body)))
    rhs = ab["rhs_idx"]
    values = {
        f"arg{rhs}": SimpleNamespace(iterator=100000000),
        f"arg{rhs}_shape0": 8,
        f"arg{rhs}_shape1": n,
        f"arg{rhs}_shape2": 256,
        f"arg{rhs}_stride0": n * 256,
        f"arg{rhs}_stride1": 256,
        f"arg{rhs}_stride2": 1,
    }
    view = _assignment(host, "tma_atom_b_rhs_tma")
    layout = next(kw.value for kw in view.keywords if kw.arg == "layout")
    shape = _value(layout.args[0], values)
    stride = _value(
        next(kw.value for kw in layout.keywords if kw.arg == "stride"), values
    )
    assert shape == ((n, 256, 8) if rank3 else (8 * n, 256))
    assert stride == ((256, 1, n * 256) if rank3 else (256, 1))
    tile = _assignment(tree, "gB_tma")
    assert _value(tile.args[1], {}) == ((224, 64, 1) if rank3 else (224, 64))
    # Evaluate the actual device group/tile expressions at every expert's tail.
    # Separate N and G descriptor extents reject the padded row even if its
    # byte address would lie inside the neighboring expert's allocation.
    for expert in (0, 1, 7):
        tile_idx = (n - 1) // 224
        coords = _value(
            tile.args[2],
            {
                "tcgen05_grouped_cta_tile_idx_n": tile_idx,
                "tcgen05_grouped_group_idx": expert,
            },
        )
        for local_n in (0, n - tile_idx * 224 - 1, 223):
            row = tile_idx * 224 + local_n
            if rank3:
                actual = (coords[0] * 224 + local_n, 255, coords[2])
                valid = all(
                    0 <= value < extent
                    for value, extent in zip(actual, shape, strict=True)
                )
                assert valid is (row < n)
                if valid:
                    assert sum(a * b for a, b in zip(actual, stride, strict=True)) == (
                        expert * n * 256 + row * 256 + 255
                    )
            else:
                assert coords[0] * 224 + local_n == expert * n + row


def test_resolved_device_store_and_host_ir_agree_on_stage_bytes_and_tma_rank():
    source = _source()
    tree = ast.parse(source)
    _, ab, output = _plans(source)
    with _forbid_cuda_work(), _sdk() as sdk:
        c, cute = sdk.cutlass, sdk.cute
        function = sdk.func.FuncOp("mn224", ([], []))
        with sdk.ir.InsertionPoint(function.add_entry_block()):
            scope = {"cutlass": c, "cute": cute}
            for index, shape, stride in (
                (ab["lhs_idx"], (640, 256), (256, 1)),
                (ab["rhs_idx"], (8, 512, 256), (512 * 256, 256, 1)),
                (output["d_idx"], (640, 512), (512, 1)),
            ):
                scope[f"arg{index}"] = cute.make_tensor(
                    cute.make_ptr(
                        c.BFloat16,
                        (index + 1) * 2**26,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(shape, stride=stride),
                )
                for dim, size in enumerate(shape):
                    scope[f"arg{index}_shape{dim}"] = size
                    scope[f"arg{index}_stride{dim}"] = stride[dim]
            for plan in (ab, output):
                body, arguments = [], []
                _append_cute_wrapper_plan(body, arguments, plan)
                exec(
                    compile(
                        ast.parse(dedent("\n".join(body))), "<actual host plan>", "exec"
                    ),
                    scope,
                )
            assert tuple(
                cute.size(scope["tma_tensor_b"], mode=[i]) for i in range(3)
            ) == (512, 256, 8)
            for name in (
                "tiled_mma",
                "sA_layout",
                "sB_layout",
                "tcgen05_c_layout",
                "tcgen05_epi_tile",
                "acc_frag_base",
                "tcgen05_acc_tmem_cols",
                "tcgen05_store_epi_tile",
                "tcgen05_sD_layout",
            ):
                node = ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=_assignment(tree, name),
                )
                exec(
                    compile(
                        ast.fix_missing_locations(
                            ast.Module(body=[node], type_ignores=[])
                        ),
                        "<actual device layout>",
                        "exec",
                    ),
                    scope,
                )
            for name in (
                "tcgen05_epi_tile",
                "tcgen05_store_epi_tile",
                "tcgen05_tma_store_atom_epi_tile",
            ):
                assert tuple(cute.size(mode) for mode in scope[name]) == (128, 32)
            assert cute.size_in_bytes(c.BFloat16, scope["sA_layout"]) == 114688
            assert cute.size_in_bytes(c.BFloat16, scope["sB_layout"]) == 100352
            assert cute.size_in_bytes(c.BFloat16, scope["tcgen05_sD_layout"]) == 16384
            assert scope["tcgen05_acc_tmem_cols"] == 512
            # Both hardware peers partition the same expert-local tail view;
            # the extra unit group mode must not become another K tile.
            for peer in (0, 1):
                scope.update(
                    tma_thr_mma=scope["tiled_mma"].get_slice(peer),
                    tcgen05_grouped_cta_tile_idx_n=2,
                    tcgen05_grouped_group_idx=7,
                )
                for name in ("gB_tma", "gB_tma_part"):
                    node = ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=_assignment(tree, name),
                    )
                    exec(
                        compile(
                            ast.fix_missing_locations(
                                ast.Module(body=[node], type_ignores=[])
                            ),
                            "<actual producer partition>",
                            "exec",
                        ),
                        scope,
                    )
                assert (
                    cute.size(
                        scope["gB_tma_part"], mode=[cute.rank(scope["gB_tma_part"]) - 1]
                    )
                    == 4
                )
            sdk.func.ReturnOp([])
        assert sdk.module.operation.verify()
    assert tcgen05_grouped_worklist_epilogue_shape(
        bm=256, bn=224, cluster_m=2, output_dtype_bytes=2, orientation="mn"
    ) == (128, 32)


def test_exact_capacity_and_proved_descriptor_discounts_fail_closed():
    assert _footprint(groups=32) == 232448
    assert _footprint(groups=32, immutable=False) == 233472
    assert _footprint(groups=48) == 233472
    assert _footprint(ab_stages=8) > 232448
    assert _footprint(c_stages=3) > 232448
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args(groups=32))
        model = bound.config_spec._cute_tcgen05_config
        assert model.grouped_d_shift_supported
        assert bound.config_spec.normalized_config(_profile())["tcgen05_ab_stages"] == 7
        with patch.object(model, "grouped_d_shift_supported", False):
            assert not model._grouped_worklist_immutable_smem(_profile().config)
            with pytest.raises(InvalidConfig):
                bound.config_spec.normalized_config(_profile())
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(_profile(depth=8))
        constraints = model.ab_stages_three_search_constraints
        assert constraints is not None
        reduced = constraints._replace(
            per_cta_smem_budget_bytes=232447
            - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
        )
        with (
            patch.object(model, "ab_stages_three_search_constraints", reduced),
            pytest.raises(InvalidConfig),
        ):
            bound.config_spec.normalized_config(_profile())


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_tail_descriptor_requires_closed_zero_seeded_pure_contraction(variant):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, (*_args(n=160), variant))
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_profile(depth=2))


def test_feature_tail_keeps_output_tma_stride_alignment_guard():
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped, _args(n=513))
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_profile(depth=2))


@pytest.mark.parametrize("variant", [0, 1])
def test_feature_tail_cannot_relax_fresh_contiguous_output_ownership(variant):
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsafe_output, (*_args(n=256), variant))
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_profile(depth=2))


def test_signed_original_offset_and_padded_coordinate_domain_is_bounded():
    assert _tcgen05_grouped_b_rank3_index_domain(16, 1536, 2048, 224)
    assert not _tcgen05_grouped_b_rank3_index_domain(2048, 256, 4096, 224)
    assert not _tcgen05_grouped_b_rank3_index_domain(1, (1 << 31) - 64, 1, 224)
    assert not _tcgen05_grouped_b_rank3_index_domain(0, 512, 256, 224)


def test_rank3_host_marker_cannot_bypass_immutable_mn_descriptor_contract():
    _, ab, _ = _plans(_source())
    for bad in ({"full_allocation_b": False}, {"orientation": "nm"}):
        with pytest.raises(BackendUnsupported):
            _append_cute_wrapper_plan([], [], ab | bad)
