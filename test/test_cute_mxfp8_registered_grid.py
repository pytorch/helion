"""Registered complete-K contractions, renamed tunables and fail-closed proofs."""

from __future__ import annotations

import ast
import inspect
import itertools
import linecache
import operator
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute_block_scaled import (
    CuteBlockScaledMmaHeuristic,
)
from helion._compiler.compile_environment import FixedBlockSizeSource
from helion._compiler.cute.registered_mxfp8 import int32_registered_domain
from helion._compiler.cute.registered_mxfp8 import match_registered_mxfp8
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language import tile_ops


@helion.kernel(
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _generic_registered_contraction(
    left: torch.Tensor,
    right: torch.Tensor,
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
) -> torch.Tensor:
    rows, channels = left.shape
    columns = right.size(0)
    assert left.dtype == right.dtype == torch.float8_e4m3fn
    assert packed_left.dtype == packed_right.dtype == torch.int32
    assert channels == right.size(1) and channels > 0 and channels % 32 == 0
    result = torch.empty((rows, columns), device=left.device, dtype=torch.bfloat16)
    width_r = hl.register_block_size(16, 128)
    width_c = hl.register_block_size(16, 256)
    width_g = hl.register_block_size(2, 32)
    partitions = hl.register_tunable("partition_count", PowerOfTwoFragment(1, 16, 1))
    _unused_tunable = hl.register_tunable(
        "independent_choice", PowerOfTwoFragment(1, 8, 2)
    )
    groups_per_partition = (channels // 32 + partitions - 1) // partitions
    temporary = torch.empty(
        (partitions, rows, columns), device=left.device, dtype=torch.float32
    )
    result_flat = result.view(rows * columns)
    temporary_flat = temporary.view(partitions * rows * columns)
    for task_r, task_c, partition in hl.grid(
        [
            (rows + width_r - 1) // width_r,
            (columns + width_c - 1) // width_c,
            partitions,
        ]
    ):
        r = task_r * width_r + hl.arange(width_r)
        c = task_c * width_c + hl.arange(width_c)
        carry = hl.zeros([width_r, width_c], dtype=torch.float32)
        finish = min((partition + 1) * groups_per_partition, channels // 32)
        for group in hl.tile(
            partition * groups_per_partition, finish, block_size=width_g
        ):
            k = group.begin * 32 + hl.arange(width_g * 32)
            x = hl.load(
                left,
                [r, k],
                extra_mask=(r[:, None] < rows) & (k[None, :] < finish * 32),
            )
            y = hl.load(
                right,
                [c, k],
                extra_mask=(c[:, None] < columns) & (k[None, :] < finish * 32),
            )
            sx_word = hl.load(
                packed_left,
                [r, group.index // 4],
                extra_mask=(r[:, None] < rows) & (group.index[None, :] < finish),
            )
            sy_word = hl.load(
                packed_right,
                [c, group.index // 4],
                extra_mask=(c[:, None] < columns) & (group.index[None, :] < finish),
            )
            sx = ((sx_word >> ((group.index % 4) * 8)[None, :]) & 255).to(torch.uint8)
            sy = ((sy_word >> ((group.index % 4) * 8)[None, :]) & 255).to(torch.uint8)
            carry = hl.dot_scaled(x, sx, "e4m3", y.T, sy, "e4m3", acc=carry)
        if partitions == 1:
            hl.store(
                result_flat,
                [r[:, None] * columns + c[None, :]],
                carry.to(result.dtype),
                extra_mask=(r[:, None] < rows) & (c[None, :] < columns),
            )
        else:
            hl.store(
                temporary_flat,
                [(partition * rows + r[:, None]) * columns + c[None, :]],
                carry,
                extra_mask=(r[:, None] < rows) & (c[None, :] < columns),
            )
    if partitions > 1:
        return temporary.sum(dim=0).to(torch.bfloat16)
    return result


_VARIANTS = itertools.count()


def _host_variant(option):
    tree = ast.parse(inspect.getsource(_generic_registered_contraction.fn))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    function.name = f"renamed_host_variant_{next(_VARIANTS)}"
    allocation = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        )
    )
    if option == "host_write":
        function.body.insert(
            function.body.index(allocation) + 1, ast.parse("result.add_(1)").body[0]
        )
    elif option == "unreachable_host_write":
        branch = next(node for node in function.body if isinstance(node, ast.If))
        branch.body.insert(0, ast.parse("result.add_(1)").body[0])
    elif option == "bad_return":
        function.body[-1] = ast.parse("return result + 1").body[0]
    else:
        function.args.args.append(ast.arg(arg="external_output"))
        allocation.value = ast.Name(id="external_output", ctx=ast.Load())
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(tree) + "\n"
    filename = f"<registered-host-variant-{function.name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    return helion.kernel(
        namespace[function.name],
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _bound(m=7, n=33, k=160, **options):
    args = (
        torch.empty((m, k), dtype=torch.float8_e4m3fn),
        torch.empty((n, k), dtype=torch.float8_e4m3fn),
        torch.empty_strided(
            (m, (k + 127) // 128), (1, (m + 3) // 4 * 4), dtype=torch.int32
        ),
        torch.empty_strided(
            (n, (k + 127) // 128), (1, (n + 3) // 4 * 4), dtype=torch.int32
        ),
    )
    if options:
        option = next(iter(options))
        kernel = _host_variant(option)
        return _cpu_bind(
            kernel, args + ((options[option],) if option == "external_output" else ())
        )
    return _cpu_bind(_generic_registered_contraction, args)


def _match(bound):
    with bound.env, bound.host_function:
        return match_registered_mxfp8(
            bound.host_function.device_ir.graphs, bound.host_function.device_ir
        )


@pytest.fixture(autouse=True)
def _cpu_only():
    with (
        _forbid_cuda_work(),
        _mock_cuda_unavailable(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=228 * 1024
        ),
    ):
        yield


@pytest.mark.parametrize(
    "m,n,k", [(1, 129, 32), (7, 33, 160), (32, 5120, 2048), (73, 273, 512)]
)
def test_registered_mxfp8_positive_cpu(m, n, k):
    bound = _bound(m, n, k)
    plan = _match(bound)
    assert plan is not None and plan.partition_key == "partition_count"
    assert bound.config_spec.cute_mxfp8_mma_available
    assert bound.config_spec.cute_mxfp8_fused_scales_available
    config = bound.config_spec.default_config()
    config.config.update(
        cute_scaled_mma=True,
        cute_scaled_fused_scales=True,
        cute_scaled_tile_n=32,
        cute_scaled_tile_k=128,
        cute_scaled_stages=2,
        cute_scaled_cluster_m=1,
        cute_scaled_persistent=False,
        independent_choice=4,
    )
    source = bound.to_code(config)
    assert config.config["independent_choice"] == 4
    assert config.config["partition_count"] == 1
    assert "_helion_cute_wrapper_plans" in source
    assert "temporary = torch.empty" in source and "result = torch.empty" in source


def test_registered_mxfp8_requires_partition_one_cpu():
    bound = _bound()
    config = bound.config_spec.default_config()
    config.config.update(cute_scaled_mma=True, partition_count=2)
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(config)


@pytest.mark.parametrize(
    "option", ["host_write", "unreachable_host_write", "bad_return", "external_output"]
)
def test_registered_mxfp8_host_alias_effects_reject_cpu(option):
    value = (
        torch.empty((7, 33), dtype=torch.bfloat16)
        if option == "external_output"
        else True
    )
    bound = _bound(**{option: value})
    assert _match(bound) is None
    assert not bound.config_spec.cute_mxfp8_mma_available


def test_registered_mxfp8_seeds_are_explicit_and_other_tunables_survive_cpu():
    bound = _bound()
    with bound.env, bound.host_function:
        seeds = CuteBlockScaledMmaHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    assert all(seed.config["partition_count"] == 1 for seed in seeds)
    assert all(seed.config["independent_choice"] == 2 for seed in seeds)


def test_registered_mxfp8_grid_domains_and_k_bounds_cpu():
    bound = _bound()
    assert _match(bound) is not None
    ir = bound.host_function.device_ir
    grid_ids = ir.grid_block_ids[0]
    for block_id in grid_ids:
        block = bound.env.block_sizes[block_id]
        original_size = block.size
        block.size = original_size + 1
        assert _match(bound) is None
        block.size = original_size
        assert _match(bound) is not None

        original_source = block.block_size_source
        block.block_size_source = FixedBlockSizeSource(2)
        assert _match(bound) is None
        block.block_size_source = original_source
        assert _match(bound) is not None
    iterator = ir.phases[0].root_nodes[0].iter
    original_arguments, original_keywords = iterator.args, iterator.keywords
    iterator.args = [
        ast.List(elts=[ast.Constant(0)] * 3, ctx=ast.Load()),
        *original_arguments,
    ]
    assert _match(bound) is None
    iterator.args = original_arguments
    iterator.keywords = [ast.keyword(arg="step", value=ast.Constant(2))]
    assert _match(bound) is None
    iterator.keywords = original_keywords
    assert _match(bound) is not None
    root = ir.graphs[ir.root_ids[0]]
    loop = next(
        node for node in root.graph.nodes if node.target is _tracing_ops._for_loop
    )
    for argument in (1, 2):
        original_args = loop.args
        original_bound = loop.args[argument][0]
        with root.graph.inserting_before(loop):
            shifted = root.graph.call_function(operator.add, (original_bound, 1))
        shifted.meta["val"] = original_bound.meta["val"] + 1
        altered = list(loop.args)
        altered[argument] = [shifted]
        loop.args = tuple(altered)
        assert _match(bound) is None
        loop.args = original_args
        root.graph.erase_node(shifted)
        assert _match(bound) is not None


def test_registered_mxfp8_int32_domain_metadata_cpu():
    limit = 2**31
    assert int32_registered_domain(32, 5120, 2048, 128, 256, 32)
    assert int32_registered_domain(limit - 128, 1, 128, 128, 256, 32)
    assert int32_registered_domain(1, limit - 256, 128, 128, 256, 32)
    assert int32_registered_domain(1, 1, limit - 1024, 128, 256, 32)
    assert not int32_registered_domain(limit - 127, 1, 128, 128, 256, 32)
    assert not int32_registered_domain(1, limit - 255, 128, 128, 256, 32)
    assert not int32_registered_domain(1, 1, limit - 32, 128, 256, 32)
    assert not int32_registered_domain(65536, 65536, 128, 128, 256, 32)


def test_registered_mxfp8_maximum_width_guard_cpu():
    bound = _bound()
    assert _match(bound) is not None
    for block_id in (0, 1, 2):
        item = bound.config_spec.block_sizes.block_id_lookup(block_id)
        previous = item.max_size
        item.max_size = 2**31
        assert _match(bound) is None
        item.max_size = previous
        assert _match(bound) is not None
        previous = item.declared_max_size
        item.declared_max_size = 2**31
        assert _match(bound) is None
        item.declared_max_size = previous
        assert _match(bound) is not None


def test_registered_mxfp8_actual_index_graph_int32_oracle_cpu():
    """Execute the original FX coordinate expressions with real Int32 lanes."""
    bound = _bound()
    ir = bound.host_function.device_ir
    root = ir.graphs[ir.root_ids[0]]
    reduction = next(
        node for node in root.graph.nodes if node.target is _tracing_ops._for_loop
    )
    body = ir.graphs[reduction.args[0]]
    dot = next(node for node in body.graph.nodes if node.target is hl.dot_scaled)
    expressions = (reduction.args[3][0], reduction.args[3][1], dot.args[0].args[1][1])
    for expression, width, extent in zip(
        expressions,
        (128, 256, 1024),
        (2**31 - 128, 2**31 - 256, 2**31 - 1024),
        strict=True,
    ):
        task = (extent + width - 1) // width - 1
        start = task * width

        def execute(
            node, *, task=task, width=width, expression=expression, start=start
        ):
            if not isinstance(node, torch.fx.Node):
                return node
            if node.target is _tracing_ops._get_symnode:
                value = node.meta["val"]._sympy_()
                origin = bound.host_function.expr_to_origin.get(value)
                if origin is not None and type(origin.origin).__name__ == "GridOrigin":
                    return torch.tensor(task, dtype=torch.int32)
                return width // 32 if expression is expressions[2] else width
            if node.target is tile_ops.tile_begin:
                return torch.tensor(start // 32, dtype=torch.int32)
            if node.target is torch.ops.prims.iota.default:
                return torch.arange(execute(node.args[0]), dtype=torch.int32)
            if node.target in (operator.mul, torch.ops.aten.mul.Tensor):
                return execute(node.args[0]) * execute(node.args[1])
            if node.target in (operator.add, torch.ops.aten.add.Tensor):
                return execute(node.args[0]) + execute(node.args[1])
            raise AssertionError(f"unexpected coordinate op {node.target}")

        actual = execute(expression)
        expected = torch.arange(width, dtype=torch.int64) + start
        assert actual.dtype == torch.int32
        torch.testing.assert_close(actual.to(torch.int64), expected, atol=0, rtol=0)
        assert int(expected[-1]) < 2**31
    # The guard also protects a rounding numerator that need not be widened.
    numerator = torch.tensor(2**31 - 1, dtype=torch.int32) + 127
    assert int(numerator) < 0


def test_registered_mxfp8_graph_mutations_cpu():
    bound = _bound()
    original = _match(bound)
    assert original is not None
    graphs = bound.host_function.device_ir.graphs
    root = graphs[bound.host_function.device_ir.root_ids[0]]
    body = next(info for info in graphs if type(info).__name__ == "ForLoopGraphInfo")
    final = next(info for info in graphs if type(info).__name__ == "IfGraphInfo")
    other = next(info for info in graphs if type(info).__name__ == "ElseGraphInfo")

    def mutation(node, field, value):
        saved = getattr(node, field)
        try:
            setattr(node, field, value)
            assert _match(bound) is None
        finally:
            setattr(node, field, saved)
        assert _match(bound) is not None

    for iota in root.graph.find_nodes(
        op="call_function", target=torch.ops.prims.iota.default
    ):
        for key, value in (("start", 1), ("step", 2), ("dtype", torch.int64)):
            mutation(iota, "kwargs", dict(iota.kwargs) | {key: value})
    initial = next(node for node in root.graph.nodes if node.target is hl.full)
    for value in (1.0, -0.0):
        mutation(initial, "args", (initial.args[0], value, *initial.args[2:]))
    for add in root.graph.find_nodes(
        op="call_function", target=torch.ops.aten.add.Tensor
    ):
        mutation(add, "kwargs", {"alpha": 2})
    dot = next(node for node in body.graph.nodes if node.target is hl.dot_scaled)
    mutation(dot, "args", (*dot.args[:2], "e5m2", *dot.args[3:]))
    for mask in body.graph.find_nodes(
        op="call_function", target=torch.ops.aten.bitwise_and.Scalar
    ):
        mutation(mask, "args", (mask.args[0], 127))
    for divisor in body.graph.find_nodes(
        op="call_function", target=torch.ops.aten.div.Tensor_mode
    ):
        mutation(divisor, "args", (divisor.args[0], 2))
    for comparison in body.graph.find_nodes(
        op="call_function", target=torch.ops.aten.lt.Scalar
    ):
        mutation(comparison, "target", torch.ops.aten.le.Scalar)
    for info in (root, body, final, other):
        output = next(iter(info.graph.find_nodes(op="output")))
        with info.graph.inserting_before(output):
            assertion = info.graph.call_function(
                torch.ops.aten._assert_async.msg,
                (torch.tensor(True), "observable assertion"),
            )
        assert _match(bound) is None
        info.graph.erase_node(assertion)
        assert _match(bound) is not None
    store = next(node for node in final.graph.nodes if node.target is memory_ops.store)
    with final.graph.inserting_before(store):
        duplicate = final.graph.call_function(memory_ops.store, store.args)
    assert _match(bound) is None
    final.graph.erase_node(duplicate)
    assert _match(bound) is not None
