"""Predicate-only bool byte leaves and exact FP32 scalar-branch where proof."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_epilogue_dag import _evaluate
from test.test_cute_epilogue_dag import _render

import helion
from helion._compiler.cute import cute_epilogue as epi
from helion._compiler.cute import cute_mma
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bool_where_epilogue(
    a, b, predicate, scalar, mask, out, mode: hl.constexpr = "scalar_true"
):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            right = b[inner, col].to(torch.float32).to(b.dtype)
            acc = torch.addmm(acc, left, right)
        if mode == "bare_M":
            pred = predicate[row]
        elif mode == "offset":
            pred = predicate[row.index + 1, col]
        elif mode == "transpose":
            pred = predicate[col, row].T
        elif mode == "masked":
            pred = hl.load(predicate, [row, col], extra_mask=row.index[:, None] >= 0)
        elif mode == "cloned":
            pred = predicate[row, col].clone()
        elif mode == "computed":
            pred = acc > hl.full([], 0.0, dtype=torch.float32)
        else:
            pred = predicate[row, col]
        scale = mask[row, col].to(torch.float32)
        branch = acc * (scale * hl.full([], 1.1111111111111112, dtype=torch.float32))
        raw_scalar = scalar[()]
        if mode != "implicit_scalar_half":
            selected_scalar = raw_scalar.to(torch.float32)
        else:
            selected_scalar = raw_scalar
        if mode == "scalar_false":
            value = torch.where(pred, branch, selected_scalar)
        elif mode == "predicate_only":
            value = torch.where(pred, selected_scalar, acc)
        elif mode == "full_branches":
            value = torch.where(pred, acc, branch)
        elif mode == "reuse_predicate_cast":
            value = torch.where(pred, selected_scalar, branch * pred.to(torch.float32))
        elif mode == "raw_numeric":
            value = torch.where(pred, selected_scalar, branch * pred)
        elif mode == "raw_numeric_after_where":
            selected = torch.where(pred, selected_scalar, branch)
            value = selected * pred
        elif mode == "raw_store":
            value = pred
        else:
            value = torch.where(pred, selected_scalar, branch)
        out[row, col] = value.to(out.dtype)
    return out.T


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def shared_bool_where(a, b, predicate, scalar, first, second):
    m, k = a.shape
    n = b.shape[1]
    for row, col in hl.tile((m, n)):
        acc = hl.zeros((row, col), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        pred = predicate[row, col]
        selected = torch.where(pred, scalar[()].to(torch.float32), acc)
        first[row, col] = selected
        second[row, col] = selected * pred.to(torch.float32)


def arguments(
    input_dtype=torch.bfloat16,
    scalar_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    m=256,
    n=128,
    alias=None,
    stride=1,
    offset=0,
):
    a = torch.empty((m, 128), dtype=input_dtype)
    b = torch.empty((n, 128), dtype=input_dtype).T
    raw = torch.empty(m * n * stride + offset, dtype=torch.bool)
    predicate = raw[offset : offset + m * n * stride : stride].view(m, n)
    scalar = torch.empty((), dtype=scalar_dtype)
    mask = torch.empty((m, n), dtype=torch.bool)
    out = torch.empty((m, n), dtype=output_dtype)
    if alias == "predicate":
        predicate = (
            out.view(torch.uint8).reshape(-1)[: m * n].view(torch.bool).view(m, n)
        )
    elif alias == "mask":
        mask = out.view(torch.uint8).reshape(-1)[: m * n].view(torch.bool).view(m, n)
    elif alias == "scalar":
        scalar = out.view(scalar_dtype).reshape(-1)[0]
    return a, b, predicate, scalar, mask, out


def analyze(bound):
    graphs = bound.host_function.device_ir.graphs
    anchor = next(
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    )
    with bound.env, bound.host_function:
        chains = analyze_tcgen05_matmul_store_chains(graphs, anchor)
    return graphs, anchor, chains


@pytest.fixture(autouse=True)
def _cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@pytest.mark.parametrize("input_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("scalar_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "mode", ("scalar_true", "scalar_false", "full_branches", "reuse_predicate_cast")
)
def test_exact_predicate_where_keeps_single_original_store_byte_width_and_scalar_cast(
    input_dtype, scalar_dtype, mode
):
    bound = cpu_bind(bool_where_epilogue, (*arguments(input_dtype, scalar_dtype), mode))
    assert bound.config_spec.cute_tcgen05_search_enabled
    graphs, anchor, chains = analyze(bound)
    assert chains is not None and len(chains) == 1
    chain = chains[0][1]
    predicates = [
        leaf
        for leaf in chain.auxiliary_tensor_loads
        if isinstance(leaf, epi._ReadonlyBoolPredicateLoadExpr)
    ]
    assert len(predicates) == 1 and not isinstance(
        predicates[0], epi._ReadonlyBoolAuxLoadExpr
    )
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    assert "tcgen05_ab_tma" in source and "cute.where" in source
    assert "cutlass.Uint8" in source and "!= cutlass.Uint8(0)" in source
    assert source.count("'kind': 'tcgen05_d_tma'") == 1
    assert source.count("PipelineTmaStore.create(") == 1
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert "'output_dtype': 'cutlass.BFloat16'" in source
    assert "ref_mode" not in source or "REF" not in source
    # The original callable still returns the same transpose alias/view.
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "T"
        for node in ast.walk(
            ast.parse(__import__("inspect").getsource(bool_where_epilogue.fn))
        )
    )


@pytest.mark.parametrize(
    "mode",
    (
        "bare_M",
        "offset",
        "transpose",
        "masked",
        "cloned",
        "raw_numeric",
        "raw_numeric_after_where",
        "raw_store",
        "implicit_scalar_half",
    ),
)
def test_unproved_predicate_shape_numeric_reuse_or_scalar_precision_stays_refused(mode):
    kwargs = {"m": 128, "n": 128}
    if mode == "bare_M":
        args = list(arguments(**kwargs))
        args[2] = torch.empty(128, dtype=torch.bool)
    else:
        args = arguments(**kwargs)
    bound = cpu_bind(bool_where_epilogue, (*args, mode))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    _, _, chains = analyze(bound)
    assert chains is None


@pytest.mark.parametrize("alias", ("predicate", "mask", "scalar"))
def test_every_leaf_readonly_source_and_output_alias_facts_remain_required(alias):
    bound = cpu_bind(bool_where_epilogue, arguments(alias=alias))
    assert not bound.config_spec.cute_tcgen05_search_enabled
    assert analyze(bound)[2] is None


@pytest.mark.parametrize("output_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("stride_offset", ((1, 0), (2, 7)))
def test_predicate_transport_keeps_actual_source_strides_offsets_and_final_dtype(
    output_dtype, stride_offset
):
    stride, offset = stride_offset
    inputs = arguments(output_dtype=output_dtype, stride=stride, offset=offset)
    bound = cpu_bind(bool_where_epilogue, inputs)
    assert bound.config_spec.cute_tcgen05_search_enabled
    source = bound.to_code(helion.Config(block_sizes=[128, 64, 64], pid_type="flat"))
    expected = {
        torch.float16: "Float16",
        torch.bfloat16: "BFloat16",
        torch.float32: "Float32",
    }[output_dtype]
    assert "'output_dtype': 'cutlass." + expected + "'" in source
    leaf = next(
        x
        for x in analyze(bound)[2][0][1].auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    host = leaf.load_node.args[0].meta["val"]
    assert host.dtype is torch.bool and host.stride() == (128 * stride, stride)
    assert inputs[2].storage_offset() == offset
    assert inputs[2].stride() == (128 * stride, stride)
    assert "cutlass.Uint8" in source


@pytest.mark.parametrize(
    "scalar_value", (-0.0, 0.0, 1.0, float("inf"), -float("inf"), float("nan"))
)
@pytest.mark.parametrize(
    "mode", ("scalar_true", "scalar_false", "reuse_predicate_cast")
)
def test_rendered_noncanonical_bool_predicate_and_scalar_ieee_selection_is_bitwise(
    scalar_value, mode
):
    bound = cpu_bind(bool_where_epilogue, (*arguments(), mode))
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    chain = chains[0][1]
    acc = (
        torch.tensor(
            [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan"), 1.0e-8]
        )
        .repeat(8)
        .view(8, 8)
    )
    raw = (
        torch.tensor([0, 1, 2, 128, 255, 0, 2, 255], dtype=torch.uint8)
        .repeat(8)
        .view(8, 8)
    )
    maskraw = raw.flip(1)
    scalar = torch.tensor(scalar_value, dtype=torch.bfloat16)
    factor = maskraw.ne(0).float() * torch.tensor(
        1.1111111111111112, dtype=torch.float32
    )
    branch = acc * factor
    if mode == "reuse_predicate_cast":
        branch = branch * raw.ne(0).float()
    expected = (
        torch.where(raw.ne(0), branch, scalar.float())
        if mode == "scalar_false"
        else torch.where(raw.ne(0), scalar.float(), branch)
    )
    # Preserve the final original BF16 cast; the renderer keeps this explicit
    # round trip in opmath before the final store performs its own conversion.
    expected = expected.to(torch.bfloat16).float()
    bindings = {}
    for leaf in chain.auxiliary_tensor_loads:
        name = leaf.load_node.args[0].args[0]
        bindings[leaf] = {"predicate": "raw", "mask": "maskraw"}[name]
    scalar_bindings = dict.fromkeys(chain.runtime_scalars, "scalar")
    bindings.update(scalar_bindings)
    source, expression = _render((chain, anchor), aux=bindings)
    actual = _evaluate(
        source,
        expression,
        acc,
        raw=raw,
        maskraw=maskraw,
        scalar=scalar,
        cutlass=SimpleNamespace(
            Float16=torch.float16,
            BFloat16=torch.bfloat16,
            Float32=torch.float32,
            Uint8=torch.uint8,
        ),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    valid = ~torch.isnan(expected)
    assert torch.equal(torch.signbit(actual[valid]), torch.signbit(expected[valid]))


def test_numeric_comparison_predicate_keeps_original_scalar_broadcast_proof():
    bound = cpu_bind(bool_where_epilogue, (*arguments(), "computed"))
    assert bound.config_spec.cute_tcgen05_search_enabled
    chains = analyze(bound)[2]
    assert chains is not None
    assert not any(
        isinstance(leaf, epi._ReadonlyBoolPredicateLoadExpr)
        for leaf in chains[0][1].auxiliary_tensor_loads
    )


def test_predicate_only_certificate_does_not_steal_explicit_float_bool_seed_subject():
    args = arguments()
    bound = cpu_bind(bool_where_epilogue, args)
    state = bound.config_spec._cute_tcgen05_config
    subject = state._readonly_bool_aux_seed_subject
    assert subject is not None
    chain = analyze(bound)[2][0][1]
    pred = next(
        x
        for x in chain.auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    floats = tuple(
        x.load_node
        for x in chain.auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolAuxLoadExpr)
    )
    assert subject[2] == floats and pred.load_node not in subject[2]
    assert bound.env.bound_runtime_input_specialization_results
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(
            helion.Config(
                block_sizes=[32, 16, 16],
                pid_type="flat",
                cute_matmul_family="warp_mma",
                cute_warp_mma_warps=4,
            )
        )


def test_where_scalar_branch_ownership_rejects_vectors_carriers_and_cycles():
    scalar = epi._ScalarConstantExpr("1.0", torch.float32)
    assert epi._where_scalar_branch_owned(scalar)
    expr = scalar
    for _ in range(31):
        expr = epi._BinaryTensorExpr("add", "{lhs}+{rhs}", expr, expr)
    assert epi._where_scalar_branch_owned(expr)
    for _ in range(40):
        expr = epi._BinaryTensorExpr("add", "{lhs}+{rhs}", expr, expr)
    assert not epi._where_scalar_branch_owned(expr)
    assert not epi._where_scalar_branch_owned(epi._CurrentTensorExpr())
    cyclic = epi._UnaryTensorExpr(epi._UnaryOp("to", "{inner}"), scalar)
    object.__setattr__(cyclic, "operand", cyclic)
    assert not epi._where_scalar_branch_owned(cyclic)


def test_raw_predicate_only_has_no_float_cast_compact_seed_certificate():
    bound = cpu_bind(bool_where_epilogue, (*arguments(), "predicate_only"))
    assert bound.config_spec.cute_tcgen05_search_enabled
    state = bound.config_spec._cute_tcgen05_config
    assert state._readonly_bool_aux_seed_subject is None
    assert state._multi_output_seed_configs() == []
    chain = analyze(bound)[2][0][1]
    assert len(chain.auxiliary_tensor_loads) == 1
    assert isinstance(
        chain.auxiliary_tensor_loads[0], epi._ReadonlyBoolPredicateLoadExpr
    )
    assert chain.auxiliary_tensor_loads[0].template == "(({aux}) != cutlass.Uint8(0))"


@pytest.mark.parametrize(
    "fault",
    (
        "mask",
        "offset",
        "swapped_index",
        "square_repeated_index",
        "host_other_graph",
        "result_dtype",
        "result_shape",
        "host_shape",
        "carrier_device",
        "kwargs",
    ),
)
def test_bool_predicate_leaf_requires_exact_carrier_identity_and_complete_typed_load(
    fault,
):
    bound = cpu_bind(bool_where_epilogue, arguments(m=128, n=128))
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    leaf = next(
        x
        for x in chains[0][1].auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    node = leaf.load_node
    host = node.args[0]
    indices = epi._carrier_tile_index_nodes(chains[0][0].args[2])
    old_args, old_kwargs, old_value, old_host = (
        node.args,
        node.kwargs,
        node.meta["val"],
        host.meta["val"],
    )
    carrier_device = anchor.meta["val"].device
    try:
        if fault == "mask":
            node.args = (*node.args[:2], True, None)
        elif fault == "offset":
            node.args = (host, [0, indices[1]], None, None)
        elif fault == "swapped_index":
            node.args = (host, [indices[1], indices[0]], None, None)
        elif fault == "square_repeated_index":
            node.args = (host, [indices[0], indices[0]], None, None)
        elif fault == "host_other_graph":
            other = cpu_bind(bool_where_epilogue, arguments(m=128, n=128))
            foreign = next(
                x
                for x in analyze(other)[2][0][1].auxiliary_tensor_loads
                if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
            )
            node.args = (foreign.load_node.args[0], *node.args[1:])
        elif fault == "result_dtype":
            node.meta["val"] = old_value.to(torch.float32)
        elif fault == "result_shape":
            node.meta["val"] = old_value[:, 0]
        elif fault == "host_shape":
            host.meta["val"] = old_host[:1, :]
        elif fault == "carrier_device":
            carrier_device = torch.device("cpu")
        else:
            node.kwargs = {"unknown": True}
        with bound.env, bound.host_function:
            assert (
                epi._readonly_bool_predicate_load(
                    node,
                    carrier_tile_shape=tuple(anchor.meta["val"].shape),
                    carrier_tile_index_nodes=indices,
                    carrier_global_shape=(128, 128),
                    carrier_device=carrier_device,
                    graphs=graphs,
                )
                is None
            )
    finally:
        node.args, node.kwargs, node.meta["val"], host.meta["val"] = (
            old_args,
            old_kwargs,
            old_value,
            old_host,
        )


@pytest.mark.parametrize(
    "fault",
    (
        "true_half",
        "false_half",
        "true_vector",
        "condition_vector",
        "result_half",
        "other_graph",
        "where_kwargs",
    ),
)
def test_new_where_broadcast_is_not_general_dtype_shape_or_graph_permission(fault):
    bound = cpu_bind(bool_where_epilogue, arguments())
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    where = next(
        n
        for graph in graphs
        for n in graph.graph.nodes
        if n.target is torch.ops.aten.where.self
    )
    predicate, left, right = where.args
    old_args, kwargs = where.args, where.kwargs
    old_meta = {x: x.meta["val"] for x in (predicate, left, right, where)}
    try:
        if fault == "true_half":
            left.meta["val"] = left.meta["val"].to(torch.bfloat16)
        elif fault == "false_half":
            right.meta["val"] = right.meta["val"].to(torch.bfloat16)
        elif fault == "true_vector":
            left.meta["val"] = left.meta["val"].expand(128)
        elif fault == "condition_vector":
            predicate.meta["val"] = predicate.meta["val"][0, :]
        elif fault == "result_half":
            where.meta["val"] = where.meta["val"].to(torch.bfloat16)
        elif fault == "other_graph":
            other = cpu_bind(bool_where_epilogue, arguments())
            foreign = next(
                n
                for graph in other.host_function.device_ir.graphs
                for n in graph.graph.nodes
                if n.target is torch.ops.aten.where.self
            )
            where.args = (foreign.args[0], left, right)
        else:
            where.kwargs = {"unknown": True}
        with bound.env, bound.host_function:
            assert (
                epi._analyze_tcgen05_pointwise_dag(
                    chains[0][0].args[2], graphs, {anchor}, (256, 128)
                )
                is None
            )
    finally:
        where.args, where.kwargs = old_args, kwargs
        for node, value in old_meta.items():
            node.meta["val"] = value


@pytest.mark.parametrize(
    "fault", ("missing_facts", "missing_registration", "fake_args")
)
def test_predicate_inline_and_whole_graph_alias_proofs_fail_closed(fault):
    if fault == "fake_args":
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            args = tuple(mode.from_tensor(x) for x in arguments())
        bound = cpu_bind(bool_where_epilogue, args)
        assert not bound.config_spec.cute_tcgen05_search_enabled
        return
    bound = cpu_bind(bool_where_epilogue, arguments())
    graphs, anchor, chains = analyze(bound)
    key = "tensor_storage_disjoint_matrix_v1"
    mapping = (
        bound.env.bound_runtime_input_specialization_results
        if fault == "missing_facts"
        else bound.env.runtime_input_specializations
    )
    old = mapping.pop(key)
    try:
        with bound.env, bound.host_function:
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
            assert key not in mapping
    finally:
        mapping[key] = old


def test_managed_predicate_rebinding_prepared_guard_and_post_codegen_facts_are_original():
    args = arguments()
    with patch.object(
        bool_where_epilogue, "_bind_isolated", side_effect=bool_where_epilogue.bind
    ):
        bound = cpu_bind(bool_where_epilogue, args)
    saved = dict(bound.env.bound_runtime_input_specialization_results)
    configs = [
        bound.config_spec.default_config(),
        bound.config_spec._base_default_config(),
        helion.Config(block_sizes=[128, 64, 64], pid_type="flat"),
    ]
    with bound.env, bound.host_function:
        pairs = ConfigGeneration(bound.config_spec).seed_flat_config_pairs()
    configs.extend(cfg for _, cfg in pairs)
    for config in configs:
        source = bound.to_code(config)
        assert "tcgen05_ab_tma" in source
        assert dict(bound.env.bound_runtime_input_specialization_results) == saved
    key = "tensor_storage_disjoint_matrix_v1"
    specialization = bound.env.runtime_input_specializations[key]
    guard = specialization.prepared_guard_factory(
        tuple(
            lambda values, i=i: values[i] for i in range(len(specialization.sources))
        ),
        saved[key],
    )
    overlap = list(args)
    overlap[2] = (
        args[-1]
        .view(torch.uint8)
        .reshape(-1)[: args[2].numel()]
        .view(torch.bool)
        .view(args[2].shape)
    )
    assert guard(args) and not guard(tuple(overlap))
    with patch.object(
        bool_where_epilogue, "_bind_isolated", side_effect=bool_where_epilogue.bind
    ):
        other = cpu_bind(bool_where_epilogue, tuple(overlap))
    assert other is not bound and not other.config_spec.cute_tcgen05_search_enabled


@pytest.mark.parametrize("mode", ("tma", "epi_tma"))
def test_auxiliary_TMA_mode_does_not_marshal_predicate_storage_as_one_bit_bool(mode):
    bound = cpu_bind(bool_where_epilogue, arguments())
    with pytest.raises((helion.exc.InvalidConfig, helion.exc.BackendUnsupported)):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 64, 64], pid_type="flat", tcgen05_aux_load_mode=mode
            )
        )


def test_bool_where_keeps_explicit_saved_collective_fallback_and_exact_scalar_storage():
    bound = cpu_bind(bool_where_epilogue, arguments())
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 64, 64],
            num_threads=[4, 32, 1],
            cute_collective_mma=True,
            cute_collective_compute="tcgen05",
            cute_collective_tmem_a=True,
        )
    )
    assert "tcgen05_ab_tma" not in source
    assert "dot_acc" in source or "collective_mma" in source
    assert "where" in source
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    with bound.env, bound.host_function:
        candidate = cute_mma.analyze_cute_mma_node(
            anchor, device_ir=bound.host_function.device_ir
        )
        assert cute_mma.tcgen05_half_source_store_eligible(
            graphs, anchor, candidate.operands
        )


@pytest.mark.parametrize("fault", ("load_target", "rank", "host_dtype", "cross_graph"))
def test_scalar_branch_ownership_has_real_rank_zero_floating_load_source(fault):
    bound = cpu_bind(bool_where_epilogue, arguments())
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    scalar = chains[0][1].runtime_scalars[0]
    load = scalar.source
    assert isinstance(load, torch.fx.Node)
    host = load.args[0]
    args, target, value, host_value = (
        load.args,
        load.target,
        load.meta["val"],
        host.meta["val"],
    )
    try:
        if fault == "load_target":
            load.target = torch.ops.aten.clone.default
        elif fault == "rank":
            load.meta["val"] = value.expand(1)
        elif fault == "host_dtype":
            host.meta["val"] = host_value.to(torch.int32)
        else:
            other = cpu_bind(bool_where_epilogue, arguments())
            foreign = analyze(other)[2][0][1].runtime_scalars[0].source
            load.args = (foreign.args[0], *args[1:])
        assert not epi._where_scalar_branch_owned(scalar)
    finally:
        load.args, load.target, load.meta["val"], host.meta["val"] = (
            args,
            target,
            value,
            host_value,
        )


def test_shared_predicate_and_explicit_float_cast_reuse_raw_bytes_not_typed_keys():
    a, b, p, s, _mask, out = arguments(output_dtype=torch.float32)
    bound = cpu_bind(shared_bool_where, (a, b, p, s, out, torch.empty_like(out)))
    graphs, anchor, chains = analyze(bound)
    assert chains is not None and len(chains) == 2
    second = chains[1][1]
    pred = next(
        x
        for x in second.auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    numeric = next(
        x
        for x in second.auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolAuxLoadExpr)
    )
    assert pred.load_node is numeric.load_node and pred is not numeric
    from helion._compiler.cute.shared_epilogue_dag import _atom
    from helion._compiler.cute.shared_epilogue_dag import raw_aux_key

    assert _atom(pred) != _atom(numeric)
    assert raw_aux_key(pred) == raw_aux_key(numeric)
    before = dict(bound.env.bound_runtime_input_specialization_results)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            pid_type="persistent_interleaved",
            tcgen05_epilogue_fanout="shared",
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
        )
    )
    assert before == bound.env.bound_runtime_input_specialization_results
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    assert (
        len(
            [
                n
                for n in calls
                if ast.unparse(n.func) == "cute.copy"
                and n.args
                and isinstance(n.args[0], ast.Name)
                and n.args[0].id.startswith("tcgen05_tiled_copy_t2r")
            ]
        )
        == 1
    )
    assert source.count("PipelineTmaStore.create(") == 2
    assert source.count("tcgen05_acc_pipeline.consumer_release(") == 1
    assert (
        len(
            [
                n
                for n in calls
                if ast.unparse(n.func) == "cute.autovec_copy"
                and n.args
                and isinstance(n.args[0], ast.Name)
                and n.args[0].id.startswith("tcgen05_tTR_gAux_subtile")
            ]
        )
        == 1
    )


@pytest.mark.parametrize("kind", ("depth", "cycle", "node_count"))
def test_bool_where_predicate_context_cannot_escape_existing_DAG_bounds(kind):
    bound = cpu_bind(bool_where_epilogue, arguments())
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    store = chains[0][0]
    original = store.args
    root = store.args[2]
    if kind == "cycle":
        where = next(
            n
            for graph in graphs
            for n in graph.graph.nodes
            if n.target is torch.ops.aten.where.self
        )
        old_where = where.args
        where.args = (where, *where.args[1:])
    else:
        with root.graph.inserting_before(store):
            value = root
            if kind == "depth":
                for _ in range(34):
                    new = root.graph.call_function(
                        torch.ops.aten.relu.default, (value,)
                    )
                    new.meta["val"] = root.meta["val"]
                    value = new
            else:
                # Balanced scalar tree has shallow depth but more than64nodes.
                current = []
                for _ in range(40):
                    x = root.graph.call_function(
                        hl.creation_ops.full, ([], 1.0, torch.float32, None)
                    )
                    x.meta["val"] = (
                        chains[0][1].runtime_scalars[0].source.meta["val"].float()
                    )
                    current.append(x)
                while len(current) > 1:
                    later = []
                    for i in range(0, len(current), 2):
                        if i + 1 == len(current):
                            later.append(current[i])
                            continue
                        x = root.graph.call_function(
                            torch.ops.aten.add.Tensor, (current[i], current[i + 1])
                        )
                        x.meta["val"] = current[i].meta["val"]
                        later.append(x)
                    current = later
                where = next(
                    n
                    for graph in graphs
                    for n in graph.graph.nodes
                    if n.target is torch.ops.aten.where.self
                )
                old_where = where.args
                where.args = (where.args[0], current[0], where.args[2])
            if kind == "depth":
                store.args = (*store.args[:2], value, *store.args[3:])
    try:
        with bound.env, bound.host_function:
            assert (
                epi._analyze_tcgen05_pointwise_dag(
                    store.args[2], graphs, {anchor}, (256, 128)
                )
                is None
            )
    finally:
        store.args = original
        if kind in ("cycle", "node_count"):
            where.args = old_where


def test_predicate_proof_reanalysis_does_not_retain_an_old_valid_leaf_or_publish_facts():
    bound = cpu_bind(bool_where_epilogue, arguments())
    graphs, anchor, chains = analyze(bound)
    assert chains is not None
    first = next(
        x
        for x in chains[0][1].auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    source = first.load_node
    original = source.args
    facts = dict(bound.env.bound_runtime_input_specialization_results)
    registration = dict(bound.env.runtime_input_specializations)
    source.args = (*source.args[:2], True, None)
    try:
        with bound.env, bound.host_function:
            assert analyze_tcgen05_matmul_store_chains(graphs, anchor) is None
    finally:
        source.args = original
    repeated = analyze(bound)[2]
    assert repeated is not None
    second = next(
        x
        for x in repeated[0][1].auxiliary_tensor_loads
        if isinstance(x, epi._ReadonlyBoolPredicateLoadExpr)
    )
    assert second is not first and second.load_node is first.load_node
    assert facts == bound.env.bound_runtime_input_specialization_results
    assert registration == bound.env.runtime_input_specializations
