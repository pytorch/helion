from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
import sympy
import torch

from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401

import helion
from helion._compiler.cute.epilogue_fanout import _fresh_returned_tensors
from helion._compiler.cute.row_resident import _matrix
from helion._compiler.cute.row_resident import _stage
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import creation_ops
from helion.language import matmul_ops
from helion.language import memory_ops

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from helion._compiler.cute.row_resident import RowContraction
    from helion.runtime.kernel import BoundKernel


def addmm_contraction(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rows, reduction = left.size()
    columns = right.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = torch.addmm(accumulator, left[row, k], right[k, column])
        result[row, column] = accumulator
    return result


def dot_contraction(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rows, reduction = left.size()
    columns = right.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = hl.dot(left[row, k], right[k, column], acc=accumulator)
        result[row, column] = accumulator
    return result


def specialized_dot_contraction(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    hl.specialize(left.shape)
    hl.specialize(right.shape)
    rows, reduction = left.size()
    columns = right.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row, column in hl.tile((rows, columns)):
        accumulator = hl.zeros([row, column], dtype=torch.float32)
        for k in hl.tile(reduction):
            accumulator = hl.dot(left[row, k], right[k, column], acc=accumulator)
        result[row, column] = accumulator
    return result


def bind_stage(
    dtype: torch.dtype,
    reduction: int,
    *,
    static_shapes: bool = True,
    explicitly_specialized: bool = False,
) -> BoundKernel[Any]:
    function = addmm_contraction if dtype is torch.float16 else dot_contraction
    if explicitly_specialized:
        function = specialized_dot_contraction
    values = (
        torch.empty((32, reduction), dtype=dtype),
        torch.empty((reduction, 12), dtype=dtype),
    )
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="full",
    )
    return kernel._bind_isolated(values)


def actual_stage(bound: BoundKernel[Any], dtype: torch.dtype) -> RowContraction | None:
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        fresh = _fresh_returned_tensors(host)
        if dtype is torch.float16:
            # Exercise the existing default argument as well as its arithmetic.
            return _stage(bound.env, host, 0, fresh)
        return _stage(bound.env, host, 0, fresh, dtype)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("reduction", (64, 128, 256))
def test_actual_stage_preserves_operand_dtype(
    dtype: torch.dtype, reduction: int
) -> None:
    bound = bind_stage(dtype, reduction)
    found = actual_stage(bound, dtype)
    assert found is not None
    assert found.lhs.dtype is found.rhs.dtype is dtype
    assert found.lhs.shape == (32, reduction)
    assert found.rhs.shape == (reduction, 12)
    assert len(found.stores) == 1 and found.stores[0].tensor.dtype is dtype
    assert not found.stores[0].chain.steps


@pytest.mark.parametrize("reduction", (64, 128, 256))
def test_explicit_specialization_preserves_literal_symint_metadata(
    reduction: int,
) -> None:
    bound = bind_stage(
        torch.bfloat16,
        reduction,
        static_shapes=False,
        explicitly_specialized=True,
    )
    found = actual_stage(bound, torch.bfloat16)
    assert found is not None
    assert any(isinstance(size, torch.SymInt) for size in found.lhs.shape)
    for size in (*found.lhs.shape, *found.lhs.stride()):
        if isinstance(size, torch.SymInt):
            assert isinstance(size._sympy_(), sympy.Integer)


def test_unguarded_backed_symbols_keep_the_original_path() -> None:
    bound = bind_stage(torch.bfloat16, 64, static_shapes=False)
    host = bound.host_function
    assert host is not None
    tensor = host.params.arguments["left"]
    assert isinstance(tensor, torch.Tensor)
    assert any(
        isinstance(size, torch.SymInt) and size._sympy_().free_symbols
        for size in tensor.shape
    )
    assert not _matrix(tensor, torch.bfloat16)
    assert actual_stage(bound, torch.bfloat16) is None


@contextmanager
def changed(node: torch.fx.Node, attribute: str, value: Any) -> Iterator[None]:
    before = getattr(node, attribute)
    setattr(node, attribute, value)
    try:
        yield
    finally:
        setattr(node, attribute, before)


@pytest.mark.parametrize(
    "fault",
    (
        "dot_output_dtype",
        "dot_kwargs",
        "wrong_accumulator",
        "accumulator_dtype",
        "negative_zero",
        "short_stop",
        "arbitrary_stop_node",
        "stop_kwargs",
        "stop_bool",
        "input_mask",
        "output_mask",
        "extra_child_effect",
    ),
)
def test_actual_stage_rejects_changed_contraction(fault: str) -> None:
    dtype = torch.bfloat16
    bound = bind_stage(dtype, 64)
    assert actual_stage(bound, dtype) is not None
    assert bound.host_function is not None
    graphs = bound.host_function.device_ir.graphs
    nodes = [node for info in graphs for node in info.graph.nodes]
    loop = next(node for node in nodes if node.target is _tracing_ops._for_loop)
    dot = next(node for node in nodes if node.target is matmul_ops.dot)
    seed = next(node for node in nodes if node.target is creation_ops.full)
    stop = loop.args[2][0]
    if fault == "extra_child_effect":
        graph = dot.graph
        with graph.inserting_before(dot):
            extra = graph.call_function(memory_ops.store, (None, (), None, None))
        try:
            assert actual_stage(bound, dtype) is None
        finally:
            graph.erase_node(extra)
    else:
        attribute = "args"
        if fault.startswith("dot") or fault == "wrong_accumulator":
            node = dot
            if fault == "dot_kwargs":
                attribute, value = "kwargs", {"out_dtype": torch.float32}
            elif fault == "dot_output_dtype":
                value = (*dot.args[:3], torch.bfloat16)
            else:
                value = (dot.args[0], dot.args[1], dot.args[0], dot.args[3])
        elif fault in ("accumulator_dtype", "negative_zero"):
            node = seed
            args = list(seed.args)
            args[2 if fault == "accumulator_dtype" else 1] = (
                torch.bfloat16 if fault == "accumulator_dtype" else -0.0
            )
            value = tuple(args)
        elif fault == "short_stop":
            node, value = loop, (loop.args[0], loop.args[1], [32], loop.args[3])
        elif fault.startswith("stop") or fault == "arbitrary_stop_node":
            # Exercise a canonical typed constant, even when a static converter
            # chose a literal for this particular test's loop extent.
            if isinstance(stop, torch.fx.Node):
                node = stop
            else:
                with loop.graph.inserting_before(loop):
                    node = loop.graph.call_function(_tracing_ops._get_symnode, ("64",))
                node.meta["val"] = 64
            if fault == "arbitrary_stop_node":
                attribute, value = "target", torch.ops.aten.sym_size.int
            elif fault == "stop_kwargs":
                attribute, value = "kwargs", {"incidental": True}
            else:
                attribute, value = "meta", node.meta | {"val": True}
            with changed(loop, "args", (loop.args[0], [0], [node], loop.args[3])):
                assert actual_stage(bound, dtype) is not None
                with changed(node, attribute, value):
                    assert actual_stage(bound, dtype) is None
            if node is not stop:
                loop.graph.erase_node(node)
            assert actual_stage(bound, dtype) is not None
            return
        else:
            target = memory_ops.load if fault == "input_mask" else memory_ops.store
            node = next(node for node in nodes if node.target is target)
            args = list(node.args)
            args[2 if fault == "input_mask" else 3] = True
            value = tuple(args)
        with changed(node, attribute, value):
            assert actual_stage(bound, dtype) is None
    assert actual_stage(bound, dtype) is not None
