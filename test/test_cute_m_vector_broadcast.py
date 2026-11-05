"""Bounded explicit per-M-vector singleton view in ordinary epilogue DAGs."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def m_vector_epilogue(a, b, u, v, bias, first, second, mode: hl.constexpr = "view"):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            right = b[inner, col].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, left, right)
        base = acc + bias[col].to(torch.float32)
        if mode == "bare":
            shaped = u[row]
        elif mode == "wrong_index":
            shaped = u[col].reshape([row, 1])
        elif mode == "wrong_view_axis":
            shaped = u[row].reshape([1, row])
        elif mode == "view_copy":
            shaped = u[row].clone().reshape([row, 1])
        elif mode == "mask":
            shaped = hl.load(u, [row], extra_mask=row.index >= 0).reshape([row, 1])
        elif mode == "offset":
            shaped = u[row.index + 1].reshape([row, 1])
        elif mode == "pre_cast":
            shaped = (
                u[row]
                .to(torch.float32)
                .to(torch.float16)
                .to(torch.float32)
                .reshape([row, 1])
            )
        else:
            shaped = u[row].reshape([row, 1]).to(torch.float32)
        branch = base * shaped
        first[row, col] = branch.to(first.dtype)
        second[row, col] = (branch + v[row].reshape([row, 1]).to(torch.float32)).to(
            second.dtype
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def identity_then_m_vector(a, b, u, first, second):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        first[row, col] = acc.to(first.dtype)
        second[row, col] = (acc * u[row].reshape([row, 1]).to(torch.float32)).to(
            second.dtype
        )


def arguments(
    dtype=torch.bfloat16, aux_dtype=torch.float32, m=256, n=128, stride=1, offset=0
):
    a = torch.empty((m, 128), dtype=dtype)
    b = torch.empty((n, 128), dtype=dtype).T
    raw = torch.empty(m * stride + offset, dtype=aux_dtype)
    u = raw[offset : offset + m * stride : stride]
    v = torch.empty(m, dtype=aux_dtype)
    bias = torch.empty(n, dtype=aux_dtype)
    first = torch.empty((m, n), dtype=torch.float32)
    second = torch.empty((m, n), dtype=torch.float32)
    return a, b, u, v, bias, first, second


def chains(bound):
    graphs = bound.host_function.device_ir.graphs
    anchor = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains

    with bound.env, bound.host_function:
        value = analyze_tcgen05_matmul_store_chains(graphs, anchor)
    return graphs, anchor, value


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("aux_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("bm", (64, 128))
def test_explicit_m_singleton_view_has_complete_native_and_fragment_source(
    input_dtype, aux_dtype, bm
):
    bound = cpu_bind(m_vector_epilogue, arguments(input_dtype, aux_dtype))
    assert bound.config_spec.cute_tcgen05_search_enabled
    _, _, analyzed = chains(bound)
    assert analyzed is not None and len(analyzed) == 2
    tagged = [
        leaf
        for _, chain in analyzed
        for leaf in chain.auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    ]
    assert len(tagged) == 3
    assert all(leaf.broadcast_axis == 3 for leaf in tagged)
    source = bound.to_code(helion.Config(block_sizes=[bm, 32, 64], pid_type="flat"))
    ast.parse(source)
    assert "tcgen05_ab_tma" in source
    assert ".layout.stride[0], 0)" in source
    assert "cute.local_tile(tcgen05_aux_view2d" in source
    assert "cute.make_rmem_tensor(cute.make_layout(tcgen05_tRS_rAcc.shape)" in source
    assert "cute.autovec_copy(tcgen05_tTR_gAux_subtile" in source


@pytest.mark.parametrize(
    "mode", ("bare", "wrong_index", "wrong_view_axis", "view_copy", "mask", "offset")
)
def test_unproved_m_shapes_indices_or_loads_do_not_enter_native_surface(mode):
    bound = cpu_bind(m_vector_epilogue, (*arguments(m=128, n=128), mode))
    assert not bound.config_spec.cute_tcgen05_search_enabled


def test_explicit_pre_view_float_cast_association_is_preserved():
    bound = cpu_bind(m_vector_epilogue, (*arguments(), "pre_cast"))
    _, _, analyzed = chains(bound)
    assert analyzed is not None
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    assert "Float16" in leaf.template and "Float32" in leaf.template
    assert bound.to_code(helion.Config(block_sizes=[128, 32, 64], pid_type="flat"))


@pytest.mark.parametrize("aux_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("bm", (64, 128))
def test_m_view_original_positive_strides_offsets_and_full_fragment_addresses(
    aux_dtype, bm
):
    args = arguments(aux_dtype=aux_dtype, stride=2, offset=7)
    a, b, u, v, bias, first, second = args
    u.copy_(torch.arange(u.numel(), dtype=aux_dtype))
    bound = cpu_bind(m_vector_epilogue, args)
    source = bound.to_code(helion.Config(block_sizes=[bm, 32, 64], pid_type="flat"))
    assert ".layout.stride[0], 0)" in source
    # This is exactly the physical address law of the emitted zero-N-stride
    # view. Check every row/column, including both rows of BM64 fragments.
    raw = torch.empty(0, dtype=u.dtype).set_(
        u.untyped_storage(),
        0,
        (u.untyped_storage().nbytes() // u.element_size(),),
        (1,),
    )
    for tile_m in range(a.shape[0] // bm):
        for local_m in range(bm):
            row = tile_m * bm + local_m
            offset = u.storage_offset() + row * u.stride(0)
            for tile_n in range(b.shape[1] // 32):
                for local_n in range(32):
                    column = tile_n * 32 + local_n
                    assert 0 <= column < b.shape[1]
                    assert raw[offset] == u[row]
    _, _, analyzed = chains(bound)
    assert analyzed is not None
    assert all(
        leaf.view_node.args[0].graph is leaf.load_node.graph
        for _, chain in analyzed
        for leaf in chain.auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    # Source materializes M-dependent values into the actual carrier profile;
    # it must never use the single-row scalar shortcut of legacy axis 2.
    assert "cute.make_rmem_tensor(cute.make_layout(tcgen05_tRS_rAcc.shape)" in source
    assert "tcgen05_colvec_scalar" not in source


@pytest.mark.parametrize("target_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_native_stores_keep_original_half_or_fp32_types_and_collective_fallback(
    target_dtype,
):
    args = list(arguments())
    args[-2] = args[-2].to(target_dtype)
    args[-1] = args[-1].to(target_dtype)
    bound = cpu_bind(m_vector_epilogue, tuple(args))
    source = bound.to_code(helion.Config(block_sizes=[128, 32, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source
    dtype_name = {
        torch.float16: "Float16",
        torch.bfloat16: "BFloat16",
        torch.float32: "Float32",
    }[target_dtype]
    assert source.count("'output_dtype': 'cutlass." + dtype_name + "'") == 2
    collective = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_collective_mma=True,
            cute_collective_compute="tcgen05",
            cute_collective_tmem_a=True,
        )
    )
    assert "tcgen05_ab_tma" not in collective and "collective_mma" in collective
    assert "view" in collective or "reshape" in collective


@pytest.mark.parametrize("alias", ("m_input", "outputs", "second_m_input"))
def test_m_vector_readonly_and_every_output_disjointness_fail_closed(alias):
    args = list(arguments())
    if alias == "m_input":
        args[2] = args[-2].view(-1)[: args[0].shape[0]]
    elif alias == "outputs":
        args[-1] = args[-2]
    else:
        args[3] = args[-1].view(-1)[: args[0].shape[0]]
    bound = cpu_bind(m_vector_epilogue, tuple(args))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    _, _, analyzed = chains(bound)
    assert analyzed is None


@pytest.mark.parametrize(
    "fault",
    (
        "inferred_shape",
        "n_dimension",
        "extra_kwargs",
        "view_copy",
        "dtype",
        "stride",
        "cross_graph",
        "unknown_reshape",
    ),
)
def test_m_view_tag_requires_complete_typed_metadata_and_original_graph(fault):
    bound = cpu_bind(m_vector_epilogue, arguments())
    graphs, anchor, analyzed = chains(bound)
    assert analyzed is not None
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    node = leaf.view_node
    indices = epi._carrier_tile_index_nodes(
        next(store.args[2] for store, _ in analyzed)
    )
    args, kwargs, target, value = node.args, node.kwargs, node.target, node.meta["val"]
    try:
        if fault == "inferred_shape":
            node.args = (args[0], [-1, 1])
        elif fault == "n_dimension":
            node.args = (args[0], [indices[1], 1])
        elif fault == "extra_kwargs":
            node.kwargs = {"unknown": True}
        elif fault == "view_copy":
            node.target = torch.ops.aten.view_copy.default
        elif fault == "unknown_reshape":
            node.target = torch.ops.aten.clone.default
        elif fault == "dtype":
            node.meta["val"] = value.to(torch.float64)
        elif fault == "stride":
            node.meta["val"] = value.as_strided(value.shape, (value.stride(0), 0))
        else:
            other = cpu_bind(m_vector_epilogue, arguments())
            _, _, other_chains = chains(other)
            other_leaf = next(
                leaf
                for leaf in other_chains[0][1].auxiliary_tensor_loads
                if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
            )
            node.args = (other_leaf.load_node, args[1])
        with bound.env, bound.host_function:
            assert (
                epi._m_vector_broadcast_view(
                    node,
                    carrier_tile_shape=tuple(anchor.meta["val"].shape),
                    carrier_tile_index_nodes=indices,
                    carrier_global_shape=(256, 128),
                    graphs=graphs,
                )
                is None
            )
    finally:
        node.args, node.kwargs, node.target, node.meta["val"] = (
            args,
            kwargs,
            target,
            value,
        )


def test_m_aux_shape_is_not_a_general_binary_broadcast_permission():
    _, _, analyzed = chains(cpu_bind(m_vector_epilogue, arguments()))
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    assert epi._m_vector_shape_owned(leaf)
    assert not epi._m_vector_shape_owned(
        epi._AuxiliaryTensorLoadExpr(leaf.load_node, 3, "{aux}")
    )
    assert not epi._m_vector_shape_owned(epi._CurrentTensorExpr())


def test_runtime_snapshot_and_prepared_alias_guards_cannot_reuse_m_aux_proof():
    args = arguments()
    with patch.object(
        m_vector_epilogue, "_bind_isolated", side_effect=m_vector_epilogue.bind
    ):
        bound = cpu_bind(m_vector_epilogue, args)
    _, _, analyzed = chains(bound)
    assert analyzed is not None
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is None
    before = dict(bound.env.bound_runtime_input_specialization_results)
    source = bound.to_code(helion.Config(block_sizes=[128, 32, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source
    assert before == bound.env.bound_runtime_input_specialization_results
    with (
        bound.env,
        bound.host_function,
        patch.object(bound.env, "bound_runtime_input_specialization_results", {}),
    ):
        from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains

        anchor = next(
            node
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.addmm.default
        )
        assert (
            analyze_tcgen05_matmul_store_chains(
                bound.host_function.device_ir.graphs, anchor
            )
            is None
        )
    aliased = list(args)
    aliased[2] = args[-2].view(-1)[: args[0].shape[0]]
    with patch.object(
        m_vector_epilogue, "_bind_isolated", side_effect=m_vector_epilogue.bind
    ):
        other = cpu_bind(m_vector_epilogue, tuple(aliased))
    assert other is not bound and not other.config_spec.cute_tcgen05_search_enabled


def test_m_axis_source_does_not_export_unsupported_warp_mma_family():
    bound = cpu_bind(m_vector_epilogue, arguments())
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(
            helion.Config(
                block_sizes=[32, 16, 16],
                pid_type="flat",
                cute_matmul_family="warp_mma",
                cute_warp_mma_warps=4,
            )
        )


@pytest.mark.parametrize("bm", (64, 128))
@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
def test_shared_m_vector_retains_one_real_fragment_coordinate_copy_and_both_stores(
    bm, placement
):
    bound = cpu_bind(m_vector_epilogue, arguments())
    state = bound.config_spec._cute_tcgen05_config
    assert len(state.epilogue_fanout_plans) == 1
    source = bound.to_code(
        helion.Config(
            block_sizes=[bm, 32, 64],
            pid_type="persistent_interleaved",
            tcgen05_epilogue_fanout="shared",
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_aux_load_placement=placement,
        )
    )
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    copies = [
        node
        for node in calls
        if ast.unparse(node.func) == "cute.copy"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id.startswith("tcgen05_tiled_copy_t2r")
    ]
    assert len(copies) == 1
    assert source.count("PipelineTmaStore.create(") == 2
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert ".layout.stride[0], 0)" in source
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(
            helion.Config(
                block_sizes=[bm, 32, 64],
                pid_type="flat",
                tcgen05_epilogue_fanout="shared",
            )
        )


@pytest.mark.parametrize("aux_dtype", (torch.bool, torch.int32, torch.float64))
def test_nonfloating_m_vectors_are_not_a_new_numeric_cast_path(aux_dtype):
    bound = cpu_bind(m_vector_epilogue, arguments(aux_dtype=aux_dtype))
    assert not bound.config_spec.cute_tcgen05_search_enabled


def test_fake_only_m_vector_alias_storage_is_not_a_readonly_proof():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        args = tuple(mode.from_tensor(x) for x in arguments())
    bound = cpu_bind(m_vector_epilogue, args)
    assert not bound.config_spec.cute_tcgen05_search_enabled


def test_m_vector_prepared_guard_keeps_real_alias_pattern_and_registration():
    args = arguments()
    with patch.object(
        m_vector_epilogue, "_bind_isolated", side_effect=m_vector_epilogue.bind
    ):
        bound = cpu_bind(m_vector_epilogue, args)
    overlap = list(args)
    overlap[2] = args[-2].view(-1)[: args[0].shape[0]]
    key = "tensor_storage_disjoint_matrix_v1"
    spec = bound.env.runtime_input_specializations[key]
    result = bound.env.bound_runtime_input_specialization_results[key]
    extractors = tuple(lambda values, i=i: values[i] for i in range(len(spec.sources)))
    guard = spec.prepared_guard_factory(extractors, result)
    assert guard(args) and not guard(tuple(overlap))
    graphs, anchor, analyzed = chains(bound)
    assert analyzed is not None
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    saved = bound.env.runtime_input_specializations.pop(key)
    try:
        with bound.env, bound.host_function:
            assert (
                epi._m_vector_broadcast_view(
                    leaf.view_node,
                    carrier_tile_shape=tuple(anchor.meta["val"].shape),
                    carrier_tile_index_nodes=epi._carrier_tile_index_nodes(
                        analyzed[0][0].args[2]
                    ),
                    carrier_global_shape=(256, 128),
                    graphs=graphs,
                )
                is None
            )
    finally:
        bound.env.runtime_input_specializations[key] = saved


def test_m_singleton_atomic_nodes_cannot_bypass_existing_depth_limit():
    bound = cpu_bind(m_vector_epilogue, arguments())
    graphs, anchor, analyzed = chains(bound)
    assert analyzed is not None
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    view = leaf.view_node
    old = view.args
    source = leaf.load_node
    with view.graph.inserting_before(view):
        for index in range(34):
            dtype = torch.float16 if index % 2 == 0 else torch.float32
            source = view.graph.call_function(
                torch.ops.prims.convert_element_type.default, (source, dtype)
            )
            source.meta["val"] = leaf.load_node.meta["val"].to(dtype)
    view.args = (source, old[1])
    try:
        with bound.env, bound.host_function:
            assert (
                epi._m_vector_broadcast_view(
                    view,
                    carrier_tile_shape=tuple(anchor.meta["val"].shape),
                    carrier_tile_index_nodes=epi._carrier_tile_index_nodes(
                        analyzed[0][0].args[2]
                    ),
                    carrier_global_shape=(256, 128),
                    graphs=graphs,
                )
                is None
            )
    finally:
        view.args = old


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_rendered_m_vector_float_association_and_ieee_values_are_exact(dtype):
    from types import SimpleNamespace

    from test.test_cute_epilogue_dag import _evaluate
    from test.test_cute_epilogue_dag import _render

    bound = cpu_bind(m_vector_epilogue, arguments(aux_dtype=dtype))
    graphs, anchor, analyzed = chains(bound)
    assert analyzed is not None
    acc = (
        torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan"), 1.0e-8]
        )
        .repeat(8)
        .view(8, 8)
    )
    u = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), 0.125, 0.3333], dtype=dtype
    )[:, None].expand(8, 8)
    v = torch.tensor([-0.0, 0.0, 1.0, -1.0, 2.0, 3.0, 0.125, 0.3333], dtype=dtype)[
        :, None
    ].expand(8, 8)
    bias = torch.arange(8, dtype=dtype).repeat(8, 1)
    first = (acc + bias.float()) * u.float()
    for (_store, chain), expected in zip(
        analyzed, (first, first + v.float()), strict=True
    ):
        bindings = {
            leaf: {"u": "u", "v": "v", "bias": "bias"}[leaf.load_node.args[0].args[0]]
            for leaf in chain.auxiliary_tensor_loads
        }
        source, expression = _render((chain, anchor), aux=bindings)
        actual = _evaluate(
            source,
            expression,
            acc,
            u=u,
            v=v,
            bias=bias,
            cutlass=SimpleNamespace(
                Float32=torch.float32, Float16=torch.float16, BFloat16=torch.bfloat16
            ),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = ~torch.isnan(expected)
        assert torch.equal(
            torch.signbit(actual[finite]), torch.signbit(expected[finite])
        )


def test_identity_first_output_retains_both_stores_without_new_shared_identity_proof():
    a, b, u, _v, _bias, first, second = arguments()
    bound = cpu_bind(identity_then_m_vector, (a, b, u, first, second))
    assert bound.config_spec.cute_tcgen05_search_enabled
    assert not bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    _, _, analyzed = chains(bound)
    assert analyzed is not None and len(analyzed) == 2
    source = bound.to_code(helion.Config(block_sizes=[128, 32, 64], pid_type="flat"))
    assert source.count("'kind': 'tcgen05_d_tma'") == 2
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert "first" in source and "second" in source


def test_m_shape_ownership_is_identity_bounded_under_reconvergent_dag():
    bound = cpu_bind(m_vector_epilogue, arguments())
    _, _, analyzed = chains(bound)
    assert analyzed is not None
    leaf = next(
        leaf
        for leaf in analyzed[0][1].auxiliary_tensor_loads
        if isinstance(leaf, epi._MVectorBroadcastAuxLoadExpr)
    )
    expr = leaf
    for _ in range(31):
        expr = epi._BinaryTensorExpr("add", "{lhs}+{rhs}", expr, expr)
    assert epi._m_vector_shape_owned(expr)
    for _ in range(40):
        expr = epi._BinaryTensorExpr("add", "{lhs}+{rhs}", expr, expr)
    assert not epi._m_vector_shape_owned(expr)


@pytest.mark.parametrize("mode", ("tma", "epi_tma"))
def test_m_vector_auxiliary_tma_request_stays_fail_closed(mode):
    bound = cpu_bind(m_vector_epilogue, arguments())
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64], pid_type="flat", tcgen05_aux_load_mode=mode
            )
        )
