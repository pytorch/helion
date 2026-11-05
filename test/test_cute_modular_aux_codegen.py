from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_rows(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            row.index.reshape([row.index.size(0), 1]) % aux.size(0),
            col.index.reshape([1, col.index.size(0)]),
        ].to(torch.float32)
        out[row, col] = (acc + value) * value


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_rows_transpose(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            col.index.reshape([1, col.index.size(0)]),
            row.index.reshape([row.index.size(0), 1]) % aux.size(1),
        ].to(torch.float32)
        out[row, col] = (acc + value) * value


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_cols(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            row.index.reshape([row.index.size(0), 1]),
            col.index.reshape([1, col.index.size(0)]) % aux.size(1),
        ].to(torch.float32)
        out[row, col] = (acc + value) * value


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_cols_transpose(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor, out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            col.index.reshape([1, col.index.size(0)]) % aux.size(0),
            row.index.reshape([row.index.size(0), 1]),
        ].to(torch.float32)
        out[row, col] = (acc + value) * value


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mod_shared(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.size(1)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for r in hl.tile(k):
            acc = torch.addmm(acc, a[row, r], b[r, col])
        value = aux[
            row.index.reshape([row.index.size(0), 1]) % aux.size(0),
            col.index.reshape([1, col.index.size(0)]),
        ].to(torch.float32)
        result = (acc + value) * value
        first[row, col] = result
        second[row, col] = result * 2.0


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def arguments(dtype, axis=0, transposed=False, contiguous=1, period=128):
    m, n = (2 * period, 96) if axis == 0 else (128, 2 * period)
    shape = (period, n) if axis == 0 else (m, period)
    if transposed:
        shape = shape[::-1]
    strides = (1, shape[0]) if contiguous == 0 else (shape[1], 1)
    operand_dtype = torch.bfloat16 if dtype is torch.float32 else dtype
    return (
        torch.empty((m, 64), dtype=operand_dtype),
        torch.empty((64, n), dtype=operand_dtype),
        torch.empty_strided(shape, strides, dtype=dtype),
        torch.empty((m, n), dtype=torch.float32),
    )


def chains(bound):
    with bound.env, bound.host_function:
        graphs = bound.host_function.device_ir.graphs
        anchor = next(
            n
            for info in graphs
            for n in info.graph.nodes
            if n.target is torch.ops.aten.addmm.default
        )
        return analyze_tcgen05_matmul_store_chains(graphs, anchor)


@pytest.mark.parametrize(
    "axis,transposed,contiguous",
    [(a, t, s) for a in (0, 1) for t in (False, True) for s in (0, 1)],
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_original_rank2_dtype_axis_layout_and_source_transport(
    axis, transposed, contiguous, dtype
):
    function = {
        (0, False): _mod_rows,
        (0, True): _mod_rows_transpose,
        (1, False): _mod_cols,
        (1, True): _mod_cols_transpose,
    }[(axis, transposed)]
    args = arguments(dtype, axis, transposed, contiguous)
    bound = cpu_bind(function, args)
    assert bound.config_spec.cute_tcgen05_search_enabled
    analyzed = chains(bound)
    assert analyzed is not None
    leaf = analyzed[0][1].auxiliary_tensor_loads[0]
    assert isinstance(leaf, epi._ModularAuxiliaryTensorLoadExpr)
    assert leaf.mapping.split_axis == axis and leaf.mapping.period == 128
    source = bound.to_code(helion.Config(block_sizes=[128, 32, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source
    assert "cute.make_tensor(aux.iterator, cute.make_layout((aux.shape[" in source
    assert "aux.layout.stride[0]" in source and "aux.layout.stride[1]" in source
    assert "cute.local_tile(tcgen05_aux_view2d" in source
    assert f"% {128 // (128, 32)[axis]}" in source
    assert (
        tuple(args[2].shape) == leaf.mapping.shape
        and tuple(args[2].stride()) == leaf.mapping.strides
    )


@pytest.mark.parametrize("period", (128, 192, 256))
@pytest.mark.parametrize("axis", (0, 1))
def test_structurally_admitted_period_defaults_have_valid_device_transport(
    period, axis
):
    function = _mod_rows if axis == 0 else _mod_cols
    bound = cpu_bind(function, arguments(torch.bfloat16, axis, period=period))
    assert bound.config_spec.cute_tcgen05_search_enabled
    for config in (
        bound.config_spec.default_config(),
        bound.config_spec._base_default_config(),
    ):
        source = bound.to_code(config)
        assert "tcgen05_ab_tma" in source
        assert "cute.local_tile(tcgen05_aux_view2d" in source
        assert "cute.make_tensor(aux.iterator" in source


@pytest.mark.parametrize(
    "period,blocks,axis",
    (
        (192, [128, 32, 64], 0),
        (96, [128, 32, 64], 0),
        (128, [64, 256, 64], 1),
        (128, [128, 128, 64], 0),
    ),
)
def test_modular_grid_or_wrap_is_rejected(period, blocks, axis):
    function = _mod_rows if axis == 0 else _mod_cols
    bound = cpu_bind(function, arguments(torch.bfloat16, axis, period=period))
    if period == 96 and axis == 0:
        assert not bound.config_spec.cute_tcgen05_search_enabled
        assert chains(bound) is None
        fallback = bound.to_code(helion.Config(block_sizes=blocks, pid_type="flat"))
        assert "tcgen05_ab_tma" not in fallback
        assert "tcgen05_aux_view2d" not in fallback
    else:
        with pytest.raises(exc.InvalidConfig, match="modular auxiliary"):
            bound.to_code(helion.Config(block_sizes=blocks, pid_type="flat"))


@pytest.mark.parametrize(
    "fault", ("fake", "alias", "cleared_facts", "no_registered_classifier")
)
def test_new_leaf_requires_whole_dag_readonly_disjoint_runtime_proof(fault):
    args = arguments(torch.bfloat16)
    if fault == "alias":
        a, b, _aux, out = args
        args = (a, b, out[:128, :], out)
    elif fault == "fake":
        with FakeTensorMode() as mode:
            args = tuple(mode.from_tensor(x) for x in args)
    bound = cpu_bind(_mod_rows, args)
    if fault in ("fake", "alias"):
        assert not bound.config_spec.cute_tcgen05_search_enabled
        assert chains(bound) is None
        return
    key = "tensor_storage_disjoint_matrix_v1"
    env = bound.env
    if fault == "cleared_facts":
        saved = env.bound_runtime_input_specialization_results.pop(key)
    else:
        saved = env.runtime_input_specializations.pop(key)
    try:
        assert chains(bound) is None
    finally:
        if fault == "cleared_facts":
            env.bound_runtime_input_specialization_results[key] = saved
        else:
            env.runtime_input_specializations[key] = saved


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "axis,transposed,contiguous",
    ((0, False, 0), (0, True, 1), (1, False, 1), (1, True, 0)),
)
def test_modular_transport_and_typed_DAG_preserve_IEEE_and_original_association(
    dtype, axis, transposed, contiguous
):
    import itertools

    from test.test_cute_epilogue_dag import _evaluate

    function = {
        (0, False): _mod_rows,
        (0, True): _mod_rows_transpose,
        (1, False): _mod_cols,
        (1, True): _mod_cols_transpose,
    }[(axis, transposed)]
    args = arguments(dtype, axis, transposed, contiguous)
    bound = cpu_bind(function, args)
    complete = chains(bound)
    assert complete is not None
    chain = complete[0][1]
    (leaf,) = chain.auxiliary_tensor_loads
    assert isinstance(leaf, epi._ModularAuxiliaryTensorLoadExpr)
    mapping = leaf.mapping
    host = args[2]
    host.copy_(torch.linspace(-2, 2, host.numel()).reshape(host.shape))
    host[0, 0] = -0.0
    host[0, 1] = float("inf")
    host[1, 0] = -float("inf")
    host[1, 1] = float("nan")
    rows, cols = 128, 32
    # Cross a logical repeat boundary between tiles, never inside a tile.
    start_m = 128 if axis == 0 else 0
    start_n = 128 if axis == 1 else 0
    gathered = torch.empty((rows, cols), dtype=dtype)
    physical = host.untyped_storage()
    flat = torch.empty(0, dtype=dtype).set_(physical, 0, (host.numel(),), (1,))
    for m in range(rows):
        for n in range(cols):
            gathered[m, n] = flat[mapping.offset(start_m + m, start_n + n)]
    coordinates = (
        torch.arange(start_m, start_m + rows)[:, None],
        torch.arange(start_n, start_n + cols)[None, :],
    )
    expected_index = (coordinates[axis] % mapping.period, coordinates[1 - axis])
    if mapping.order[0] == 1:
        expected_index = expected_index[::-1]
    direct = host[expected_index]
    torch.testing.assert_close(gathered, direct, rtol=0, atol=0, equal_nan=True)
    acc = torch.linspace(-1, 1, rows * cols).reshape(rows, cols)
    acc[0, 0] = -0.0
    names = itertools.count()
    source, expr = chain.render_prelude_and_expr(
        "original_acc",
        lambda prefix: f"{prefix}_{next(names)}",
        "",
        {leaf: "raw_modular_aux"},
    )
    result = _evaluate(source, expr, acc, raw_modular_aux=gathered)
    assert isinstance(result, torch.Tensor)
    reference = (acc + direct.float()) * direct.float()
    torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(reference)
    assert torch.equal(
        result[finite].view(torch.int32), reference[finite].view(torch.int32)
    )


@pytest.mark.parametrize(
    "key,value",
    (
        ("tcgen05_aux_fragment_hoist", True),
        ("tcgen05_aux_prefetch_subtiles", 1),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_epi_warp_groups", 2),
        ("tcgen05_cluster_m", 2),
        ("tcgen05_epi_subtile_n", 32),
    ),
)
def test_uncomposed_modular_protocols_decline_loudly(key, value):
    bound = cpu_bind(_mod_rows, arguments(torch.bfloat16))
    config = helion.Config(
        block_sizes=[128, 32, 64],
        pid_type="persistent_interleaved",
        tcgen05_ab_stages=2,
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    config.config[key] = value
    message = {
        "tcgen05_aux_fragment_hoist": "tcgen05_aux_fragment_hoist=True requires an exact-shape auxiliary operand",
        "tcgen05_aux_prefetch_subtiles": "the auxiliary prefetch ring needs an exact-shape rank-2 auxiliary operand",
        "tcgen05_aux_load_mode": "rank-two modular auxiliary loads require the plain single-CTA per-subtile protocol",
        "tcgen05_epi_warp_groups": "mapped auxiliary loads require the ordinary per-subtile SIMT path",
        "tcgen05_cluster_m": "rank-two modular auxiliary loads require the plain single-CTA per-subtile protocol",
        "tcgen05_epi_subtile_n": "the configured epilogue subtile needs 16-bit outputs",
    }[key]
    with pytest.raises(exc.InvalidConfig, match=message):
        bound.to_code(config)


@pytest.mark.parametrize("contiguous", (0, 1))
def test_register_mma_families_do_not_lower_modular_leaf_as_exact(contiguous):
    bound = cpu_bind(_mod_rows, arguments(torch.bfloat16, contiguous=contiguous))
    with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported), match="warp_mma"):
        bound.to_code(
            helion.Config(
                block_sizes=[32, 16, 16],
                pid_type="flat",
                cute_matmul_family="warp_mma",
                cute_warp_mma_warps=4,
            )
        )


def test_managed_shared_fanout_keeps_mapping_and_all_store_protocols():
    a, b, aux, first = arguments(torch.bfloat16)
    second = torch.empty_like(first)
    with patch.object(_mod_shared, "_bind_isolated", side_effect=_mod_shared.bind):
        bound = cpu_bind(_mod_shared, (a, b, aux, first, second))
    assert bound.config_spec.cute_tcgen05_search_enabled
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            pid_type="persistent_interleaved",
            tcgen05_ab_stages=2,
            tcgen05_epilogue_fanout="shared",
        )
    )
    assert "cute.local_tile(tcgen05_aux_view2d" in source
    assert source.count("PipelineTmaStore.create(") == 2
    assert source.count("producer_commit()") == 2
    assert source.count("producer_tail()") == 2
    complete = chains(bound)
    assert complete is not None and len(complete) == 2
