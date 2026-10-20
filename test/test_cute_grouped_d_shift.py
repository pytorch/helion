from __future__ import annotations

import ast
from copy import deepcopy
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_grouped_ab_k import _observed_logical_k
from test.test_cute_grouped_ab_k import _unsupported_k_loop
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel

import helion
from helion._compiler.cute.grouped_d_shift import bounded_d_shift_index_domain
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _cute_dynamic_tensormap_contexts
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_device_split_sizes
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_dynamic_ab_tensormaps

KEY = "tcgen05_grouped_d_shift"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unsafe_output(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, variant: int
) -> torch.Tensor:
    hl.specialize(variant)
    m, k = a.shape
    groups, n, _ = b.shape
    bm = hl.register_block_size(256)
    bn = hl.register_block_size(128)
    bk = hl.register_block_size(64)
    out = (
        a
        if variant == 0
        else torch.empty_strided((m, n), (n + 32, 1), dtype=a.dtype, device=a.device)
    )
    for group_tile, rows, columns in hl.tile([groups, m, n], block_size=[1, bm, bn]):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - start
        row = start + rows.index
        valid = rows.index < length
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(k, block_size=bk):
            left = hl.load(a, [row, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, b[group, columns, inner].T)
        hl.store(out, [row, columns], acc, extra_mask=valid[:, None])
    return out


def _config(orientation: str, *, cluster_n: int = 1) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_orientation=orientation,
        tcgen05_grouped_worklist_mma_n=256,
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_full_allocation_a=True,
        tcgen05_grouped_d_shift=True,
    )


def _args(kind: str, dtype: torch.dtype):
    metadata = (
        [0, 1, 128, 128, 128, 128, 126, 1]
        if kind == "sizes"
        else [0, 0, 1, 129, 257, 385, 513, 639, 640]
    )
    return (
        torch.empty((640, 256), dtype=torch.bfloat16),
        torch.empty((8, 512, 256), dtype=torch.bfloat16),
        torch.tensor(metadata, dtype=dtype),
    )


def _plans(source: str):
    return next(
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )


def _assignment(tree: ast.AST, name: str):
    return next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


def _value(expression: ast.expr, values):
    return eval(
        compile(ast.Expression(expression), "<bounded TMA coordinate>", "eval"),
        {"cutlass": SimpleNamespace(Int32=int, Int64=int)},
        values,
    )


@pytest.mark.parametrize("orientation", ["mn", "nm"])
@pytest.mark.parametrize("kind", ["offsets", "sizes", "fp32"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fixed_bounded_d_view_and_live_coordinates_mask_exact_rows(
    orientation: str, kind: str, dtype: torch.dtype
) -> None:
    kernel = {
        "offsets": _device_offsets_kernel,
        "sizes": _device_split_sizes_kernel,
        "fp32": _observed_logical_k,
    }[kind]
    args = _args(kind, dtype)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, args)
        assert bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        config = bound.config_spec.normalized_config(_config(orientation))
        source = bound.to_code(config)
        assert config.block_sizes[2] == 64
        if kind == "fp32":
            assert bound.config_spec.block_sizes[2].declared_max_size == 64
    tree = ast.parse(source)
    grouped, ab, output = _plans(source)
    assert grouped["bounded_output_shift"] is True
    assert grouped["immutable_ab_tensormaps"] is True
    assert not grouped.get("dynamic_d_tensormap")
    assert "ab_tensormaps_arg" not in grouped
    assert "tcgen05_grouped_ab_tensormaps" not in source
    assert "TensorMapManager" not in source
    assert "update_tensormap" not in source
    assert "selected_valid_row_mask" not in source
    _validate_tcgen05_grouped_device_split_sizes(grouped, args[2])
    with patch(
        "helion.runtime.cute.launcher._tcgen05_grouped_dynamic_ab_tensor_arg",
        side_effect=lambda plan, values, key, operand: args[
            0 if key == "lhs_idx" else 1
        ],
    ):
        _validate_tcgen05_grouped_dynamic_ab_tensormaps(grouped, args)
    with patch(
        "helion.runtime.cute.launcher._cuda_stream_capture_context",
        side_effect=AssertionError("no mutable TensorMaps need a stream context"),
    ):
        assert (
            _cute_dynamic_tensormap_contexts(
                SimpleNamespace(_helion_cute_wrapper_plans=[grouped]), args
            )
            == ()
        )

    body, call_args = [], []
    _append_cute_wrapper_plan(body, call_args, output)
    wrapper = ast.parse(dedent("\n".join(body)))
    make_tensor = _assignment(
        wrapper,
        "tcgen05_tma_store_atom_d_nm"
        if orientation == "nm"
        else "tcgen05_tma_store_atom_gmem_mnl",
    )
    d_idx = output["d_idx"]
    values = {
        f"arg{d_idx}": SimpleNamespace(iterator=100000000),
        f"arg{d_idx}_shape0": 640,
        f"arg{d_idx}_shape1": 512,
        f"arg{d_idx}_stride0": 512,
        f"arg{d_idx}_stride1": 1,
    }
    assert _value(make_tensor.args[0], values) == 100000000 - 640 * 512
    layout = next(kw.value for kw in make_tensor.keywords if kw.arg == "layout")
    shape = _value(layout.args[0], values)
    stride = _value(
        next(kw.value for kw in layout.keywords if kw.arg == "stride"), values
    )
    assert shape == ((512, 640, 641) if orientation == "nm" else (640, 512, 641))
    assert stride == ((1, 512, 512) if orientation == "nm" else (512, 1, 512))
    shift = _assignment(tree, "tcgen05_bounded_output")
    tile = _assignment(tree, "tcgen05_gC")
    assert ast.unparse(shift.func) == "cute.domain_offset"
    assert ast.unparse(tile.args[0]) == "tcgen05_bounded_output"
    row_axis = 1 if orientation == "nm" else 0
    # Evaluate the emitted expressions against the original allocation and the
    # literal frontend's clipped intervals, including empty/negative/tail starts.
    for raw_start, raw_length in [
        (-64, 192),
        (0, 0),
        (1, 1),
        (1, 128),
        (129, 511),
        (639, 256),
        (640, 1),
        (641, 1),
    ]:
        start = min(max(raw_start, 0), 640)
        end = min(max(raw_start + min(max(raw_length, 0), 640), 0), 640)
        length = max(end - start, 0)
        for tile_m in range(6):
            for local_row in (0, 1, 63, 64, 127):
                for column in (0, 255, 511):
                    coordinates = list(
                        _value(
                            shift.args[0],
                            {
                                "tcgen05_grouped_problem_m": length,
                                "tcgen05_grouped_global_m_start": start,
                            },
                        )
                    )
                    tile_coordinates = _value(
                        tile.args[2],
                        {
                            "tcgen05_grouped_cta_tile_idx_m": tile_m,
                            "tcgen05_grouped_cta_tile_idx_n": column
                            // ab["bn" if orientation == "mn" else "bm"],
                        },
                    )
                    coordinates[row_axis] += (
                        tile_coordinates[row_axis] * 128 + local_row
                    )
                    feature_tile = ab["bn" if orientation == "mn" else "bm"]
                    coordinates[1 - row_axis] += (
                        tile_coordinates[1 - row_axis] * feature_tile
                        + column % feature_tile
                    )
                    valid = all(
                        0 <= coord < extent
                        for coord, extent in zip(coordinates, shape, strict=True)
                    )
                    row = tile_m * 128 + local_row
                    assert valid == (row < length)
                    address = (
                        100000000
                        - 640 * 512
                        + sum(c * s for c, s in zip(coordinates, stride, strict=True))
                    )
                    if valid:
                        assert address == 100000000 + (start + row) * 512 + column


@pytest.mark.parametrize("orientation", ["mn", "nm"])
def test_default_off_matches_the_original_source(orientation: str) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args("fp32", torch.int32))
        config = _config(orientation)
        config.config.pop(KEY)
        explicit_off = deepcopy(config)
        explicit_off.config[KEY] = False
        assert bound.to_code(config) == bound.to_code(explicit_off)
        generated = bound.to_code(config)
        assert "TensorMapManager" in generated
        assert "bounded_output_shift" not in generated


@pytest.mark.parametrize("variant", [1, 2, 3, 5])
def test_noncanonical_reductions_cannot_request_bounded_d_shift(variant: int) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsupported_k_loop, (*_args("fp32", torch.int32), variant))
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config("mn"))


@pytest.mark.parametrize("variant", [0, 1])
def test_owned_contiguous_output_is_required(variant: int) -> None:
    args = (
        torch.empty((640, 256), dtype=torch.bfloat16),
        torch.empty((8, 256, 256), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 257, 385, 513, 639, 640], dtype=torch.int32),
        variant,
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_unsafe_output, args)
        assert not bound.config_spec._cute_tcgen05_config.grouped_d_shift_supported
        with pytest.raises((InvalidConfig, BackendUnsupported)):
            bound.to_code(_config("mn"))


@pytest.mark.parametrize("orientation", ["mn", "nm"])
def test_feature_tail_requires_exact_grouped_input_extents(orientation: str):
    args = list(_args("fp32", torch.int32))
    args[1] = torch.empty((8, 160, 256), dtype=torch.bfloat16)
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, tuple(args))
        if orientation == "mn":
            source = bound.to_code(_config(orientation))
            _, ab, output = _plans(source)
            assert ab["full_allocation_b_rank3"] is True
            assert output["bounded_output_shift"] is True
        else:
            with pytest.raises((InvalidConfig, BackendUnsupported)):
                bound.to_code(_config(orientation))


@pytest.mark.parametrize("bad", [1, "true", "ptr_shift"])
def test_invalid_config_values_are_rejected(bad) -> None:
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, _args("fp32", torch.int32))
        config = _config("mn")
        config.config[KEY] = bad
        with pytest.raises(InvalidConfig):
            bound.config_spec.normalized_config(config)


@pytest.mark.parametrize(
    "m,n,row_tile",
    [
        (0, 512, 128),
        (640, 0, 128),
        (640, 512, 0),
        ((1 << 31) - 1, 1, 1),
        (1 << 20, 1 << 12, 128),
        ((1 << 31) - 64, 1, 128),
    ],
)
def test_signed_pointer_and_virtual_coordinate_bounds_fail_closed(
    m, n, row_tile
) -> None:
    assert not bounded_d_shift_index_domain(m, n, row_tile)


def test_shifted_compiler_seeds_keep_the_old_prefix_and_logical_cap() -> None:
    args = (
        torch.empty((2048, 512), dtype=torch.bfloat16),
        torch.empty((4, 512, 512), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 2048], dtype=torch.int32),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_observed_logical_k, args)
        seeds = bound.config_spec.compiler_seed_configs
        original = [seed for seed in seeds if not seed.get(KEY)]
        shifted = [seed for seed in seeds if seed.get(KEY)]
        assert shifted and seeds[: len(original)] == original
        assert not bound.config_spec.compiler_default_config.get(KEY)
        assert bound.config_spec.block_sizes[2].declared_max_size == 64
        for seed in shifted:
            assert seed.block_sizes[2] == 64
            assert seed["tcgen05_grouped_full_allocation_a"] is True
            ast.parse(bound.to_code(seed))


@pytest.mark.cuda
@pytest.mark.parametrize("orientation", ["mn", "nm"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bounded_output_shift_live_empty_tail_rows_and_graph_replay(
    orientation: str, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family tcgen05")
    generator = torch.Generator(device="cuda").manual_seed(691)
    a = torch.randn((640, 256), generator=generator, device="cuda").bfloat16()
    b = torch.randn((8, 512, 256), generator=generator, device="cuda").bfloat16()
    b.add_(torch.arange(8, device="cuda")[:, None, None])
    boundaries = [0, 0, 1, 129, 257, 385, 513, 639, 640]
    offsets = torch.tensor(boundaries, device="cuda", dtype=dtype)
    args = (a, b, offsets)
    bound = _device_offsets_kernel._bind_isolated(args)
    bound.set_config(_config(orientation))

    def reference():
        out = torch.empty((640, 512), device="cuda", dtype=torch.bfloat16)
        for group in range(8):
            begin, end = boundaries[group : group + 2]
            out[begin:end] = (a[begin:end].float() @ b[group].float().T).bfloat16()
        return out

    torch.testing.assert_close(bound(*args), reference(), rtol=1e-2, atol=1e-2)
    boundaries[1] = 1
    offsets[1] = 1
    torch.testing.assert_close(bound(*args), reference(), rtol=1e-2, atol=1e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound(*args)
    boundaries[1] = 0
    offsets[1] = 0
    graph.replay()
    torch.testing.assert_close(output, reference(), rtol=1e-2, atol=1e-2)
