"""CPU semantic and refusal tests for preserve-output store folding."""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion._compiler.cute.fold_noop_stores import fold_noop_stores
from helion.language import _tracing_ops
from helion.language import atomic_ops
from helion.language import memory_ops
from helion.language.barrier import barrier
from helion.language.inline_asm_ops import inline_asm_elementwise

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx.node import Argument
    from torch.fx.node import Target


@dataclasses.dataclass
class _Program:
    graph: torch.fx.Graph
    tensor: torch.fx.Node
    index: torch.fx.Node
    update: torch.fx.Node
    condition: torch.fx.Node
    extra_mask: torch.fx.Node
    old: torch.fx.Node
    where: torch.fx.Node
    store: torch.fx.Node


def _program(
    dtype: torch.dtype = torch.float32,
    *,
    masked: bool = False,
    shape: tuple[int, ...] = (8,),
    condition_shape: tuple[int, ...] | None = None,
    extra_shape: tuple[int, ...] | None = None,
) -> _Program:
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    index = graph.placeholder("index")
    update = graph.placeholder("update")
    condition = graph.placeholder("condition")
    extra_mask = graph.placeholder("extra_mask")
    tensor.meta["val"] = torch.empty(16, dtype=dtype)
    index.meta["val"] = torch.empty(shape, dtype=torch.int64)
    update.meta["val"] = torch.empty(shape, dtype=dtype)
    condition.meta["val"] = torch.empty(condition_shape or shape, dtype=torch.bool)
    extra_mask.meta["val"] = torch.empty(extra_shape or shape, dtype=torch.bool)
    old = graph.call_function(memory_ops.load, (tensor, [index]))
    old.meta["val"] = torch.empty(shape, dtype=dtype)
    where = graph.call_function(torch.ops.aten.where.self, (condition, update, old))
    where.meta = {
        "val": torch.empty(shape, dtype=dtype),
        "location": object(),
        "stack_trace": "where update or preserved value",
    }
    store = graph.call_function(
        memory_ops.store,
        (tensor, [index], where, extra_mask if masked else None),
    )
    graph.output(tensor)
    return _Program(
        graph, tensor, index, update, condition, extra_mask, old, where, store
    )


class _MemoryInterpreter(torch.fx.Interpreter):
    """Materialize loads and record the actual element stores on CPU."""

    def __init__(self, graph: torch.fx.Graph) -> None:
        super().__init__(torch.fx.GraphModule({}, graph))
        self.loads = 0
        self.writes: list[int] = []

    def call_function(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if target is memory_ops.load:
            arguments = dict(
                zip(
                    ("tensor", "index", "extra_mask", "eviction_policy"),
                    args,
                    strict=False,
                )
            )
            arguments.update(kwargs)
            tensor = cast("torch.Tensor", arguments["tensor"])
            indices = cast("list[object]", arguments["index"])
            self.loads += 1
            # Snapshot the loaded tile, including when indexing returns a view.
            return tensor[tuple(indices)].clone()  # pyrefly: ignore[bad-index]
        if target is memory_ops.store:
            arguments = dict(
                zip(("tensor", "index", "value", "extra_mask"), args, strict=False)
            )
            arguments.update(kwargs)
            tensor = cast("torch.Tensor", arguments["tensor"])
            indices = cast("list[object]", arguments["index"])
            value = cast("torch.Tensor", arguments["value"])
            mask = cast("torch.Tensor | None", arguments.get("extra_mask"))
            flat_indices = torch.arange(tensor.numel()).reshape(tensor.shape)[
                tuple(indices)  # pyrefly: ignore[bad-index]
            ]
            values = torch.broadcast_to(value, flat_indices.shape).flatten()
            active = (
                torch.ones_like(flat_indices, dtype=torch.bool)
                if mask is None
                else torch.broadcast_to(mask, flat_indices.shape)
            ).flatten()
            for offset, update, enabled in zip(
                flat_indices.flatten(), values, active, strict=True
            ):
                if enabled:
                    self.writes.append(int(offset))
                    tensor.flatten()[offset] = update
            return None
        return super().call_function(target, args, kwargs)


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    if dtype.is_floating_point:
        tensor = torch.tensor(
            [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.5, -2.0, 8.0] * 2,
            dtype=dtype,
        )
        update = torch.tensor(
            [3.5, -0.0, 0.0, -2.5, 7.0, float("nan"), 9.0, -8.0], dtype=dtype
        )
    else:
        tensor = (torch.arange(16) % 2).to(dtype)
        update = (torch.arange(8) % 3).to(dtype)
    index = torch.tensor([7, 3, 0, 5, 1, 6, 2, 4])
    condition = torch.tensor([True, False, True, False, False, True, False, True])
    extra = torch.tensor([True, True, False, False, True, False, True, True])
    return tensor, index, update, condition, extra


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64, torch.bool],
)
@pytest.mark.parametrize("masked", [False, True])
def test_preserves_values_and_removes_false_writes(
    dtype: torch.dtype, masked: bool
) -> None:
    program = _program(dtype, masked=masked)
    tensor, index, update, condition, extra = _inputs(dtype)
    original = _MemoryInterpreter(program.graph).run(
        tensor.clone(), index, update, condition, extra
    )
    assert fold_noop_stores(program.graph) == 1
    interpreter = _MemoryInterpreter(program.graph)
    result = interpreter.run(tensor.clone(), index, update, condition, extra)
    # Byte equality also checks preserved NaN payloads and signed zeros.
    torch.testing.assert_close(result.view(torch.uint8), original.view(torch.uint8))
    active = condition & extra if masked else condition
    assert interpreter.writes == index[active].tolist()
    assert interpreter.loads == 0
    assert program.old not in program.graph.nodes
    assert program.where not in program.graph.nodes
    assert fold_noop_stores(program.graph) == 0


def test_inactive_duplicate_indices_never_write() -> None:
    program = _program()
    tensor = torch.arange(16, dtype=torch.float32)
    index = torch.tensor([0, 1, 2, 0, 0, 0, 0, 0])
    update = torch.arange(100, 108, dtype=torch.float32)
    condition = torch.arange(8) < 3
    assert fold_noop_stores(program.graph) == 1
    interpreter = _MemoryInterpreter(program.graph)
    result = interpreter.run(tensor.clone(), index, update, condition, condition)
    assert interpreter.writes == [0, 1, 2]
    torch.testing.assert_close(result[:3], update[:3])
    torch.testing.assert_close(result[3:], tensor[3:])


def test_combines_broadcast_masks_and_copies_source_location() -> None:
    program = _program(
        masked=True, shape=(2, 4), condition_shape=(2, 1), extra_shape=(1, 4)
    )
    location = program.where.meta["location"]
    assert fold_noop_stores(program.graph) == 1
    mask = program.store.args[3]
    assert isinstance(mask, torch.fx.Node)
    assert mask.target is torch.ops.aten.logical_and.default
    assert mask.meta["val"].shape == (2, 4)
    assert mask.meta["val"].dtype is torch.bool
    assert mask.meta["location"] is location
    tensor = torch.arange(16, dtype=torch.float32)
    index = torch.arange(8).reshape(2, 4)
    condition = torch.tensor([[True], [False]])
    extra = torch.tensor([[True, False, False, True]])
    interpreter = _MemoryInterpreter(program.graph)
    interpreter.run(tensor, index, torch.full((2, 4), 42.0), condition, extra)
    assert interpreter.writes == [0, 3]


def test_symbolic_shapes_use_fake_tensor_metadata_without_data_or_guards() -> None:
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    rows = shape_env.create_unbacked_symint()
    columns = shape_env.create_unbacked_symint()
    with mode:
        program = _program(masked=True)
        for node in (program.update, program.old, program.where):
            node.meta["val"] = torch.empty((rows, columns))
        program.condition.meta["val"] = torch.empty((rows, 1), dtype=torch.bool)
        program.extra_mask.meta["val"] = torch.empty((1, columns), dtype=torch.bool)
        guards = list(shape_env.guards)
        assert fold_noop_stores(program.graph) == 1
        combined = program.store.args[3]
        assert isinstance(combined, torch.fx.Node)
        value = combined.meta["val"]
        assert isinstance(value, FakeTensor)
        assert cast("torch.SymInt", value.shape[0]).node is rows.node
        assert cast("torch.SymInt", value.shape[1]).node is columns.node
        assert shape_env.guards == guards


def test_reuses_identical_store_mask() -> None:
    program = _program()
    program.store.args = (*program.store.args[:3], program.condition)
    assert fold_noop_stores(program.graph) == 1
    assert program.store.args[3] is program.condition
    assert all(
        node.target is not torch.ops.aten.logical_and.default
        for node in program.graph.nodes
    )


def test_accepts_keyword_memory_arguments_and_cache_hint() -> None:
    program = _program(masked=True)
    program.old.args = ()
    program.old.kwargs = {
        "tensor": program.tensor,
        "index": [program.index],
        "extra_mask": None,
        "eviction_policy": "evict_last",
    }
    program.store.args = (program.tensor,)
    program.store.kwargs = {
        "index": [program.index],
        "value": program.where,
        "extra_mask": program.extra_mask,
    }
    assert fold_noop_stores(program.graph) == 1
    assert not program.store.kwargs
    assert program.store.args[2] is program.update


def test_accepts_matching_literal_slice_and_sequence_indices() -> None:
    program = _program()
    program.old.args = (program.tensor, [slice(None, 4, None), program.index, 0])
    program.store.args = (
        program.tensor,
        (slice(None, 4, None), program.index, 0),
        program.where,
    )
    assert fold_noop_stores(program.graph) == 1


def test_keeps_shared_load_and_where_users() -> None:
    program = _program()
    output = next(node for node in program.graph.nodes if node.op == "output")
    output.args = ((program.tensor, program.where, program.old),)
    assert fold_noop_stores(program.graph) == 1
    assert program.old in program.graph.nodes
    assert program.where in program.graph.nodes
    assert program.where.args[2] is program.old


def _unknown_effect(value: torch.Tensor) -> torch.Tensor:
    value.add_(1)
    return value


@pytest.mark.parametrize(
    "target,args",
    [
        (memory_ops.store, lambda p: (p.tensor, [p.index], p.update)),
        (atomic_ops.atomic_add, lambda p: (p.tensor, [p.index], p.update)),
        (barrier, lambda p: ()),
        (_tracing_ops._for_loop, lambda p: (0, [0], [1], [])),
        (_tracing_ops._if, lambda p: (p.condition, 0, 1, [], [])),
        (torch.ops.aten.add_.Tensor, lambda p: (p.tensor, 1)),
        (_unknown_effect, lambda p: (p.tensor,)),
        (inline_asm_elementwise, lambda p: ("", "=r", [], torch.int32, False, 1)),
    ],
)
def test_rejects_intervening_effects(
    target: Callable[..., object], args: Callable[[_Program], tuple[object, ...]]
) -> None:
    program = _program()
    with program.graph.inserting_before(program.where):
        effect = program.graph.call_function(
            target, cast("tuple[Argument, ...]", args(program))
        )
    original = str(program.graph)
    assert fold_noop_stores(program.graph) == 0
    assert str(program.graph) == original
    assert effect in program.graph.nodes


def test_rejects_writes_to_potentially_aliasing_other_tensor() -> None:
    program = _program()
    with program.graph.inserting_before(program.where):
        # A distinct input can alias the output: the pass does not guess.
        program.graph.call_function(
            memory_ops.store, (program.update, [program.index], program.update)
        )
    assert fold_noop_stores(program.graph) == 0


def test_intervening_pure_arithmetic_and_loads_are_allowed() -> None:
    program = _program()
    with program.graph.inserting_before(program.where):
        pure = program.graph.call_function(
            torch.ops.aten.mul.Tensor, (program.update, program.update)
        )
        load = program.graph.call_function(
            memory_ops.load, (program.tensor, [program.index])
        )
    assert fold_noop_stores(program.graph) == 1
    # This rewrite performs targeted DCE, not a global graph cleanup.
    assert pure in program.graph.nodes and load in program.graph.nodes


def test_effect_before_load_does_not_block_fold_or_disappear() -> None:
    program = _program()
    with program.graph.inserting_before(program.old):
        effect = program.graph.call_function(_unknown_effect, (program.tensor,))
    assert fold_noop_stores(program.graph) == 1
    assert effect in program.graph.nodes


@pytest.mark.parametrize(
    "change",
    [
        lambda p: setattr(p.old, "args", (p.update, [p.index])),
        lambda p: setattr(p.old, "args", (p.tensor, [p.condition])),
        lambda p: setattr(p.old, "args", (p.tensor, [p.index, 0])),
        lambda p: setattr(p.old, "args", (p.tensor, [p.index], p.condition)),
        lambda p: setattr(p.old, "kwargs", {"other": 0}),
        lambda p: setattr(p.old, "kwargs", {"tensor": p.tensor}),
        lambda p: setattr(p.store, "kwargs", {"other": 0}),
        lambda p: setattr(p.where, "kwargs", {"out": p.update}),
        lambda p: setattr(p.where, "args", (p.condition, p.old, p.update)),
        lambda p: p.condition.meta.update(val=torch.empty(8, dtype=torch.uint8)),
        lambda p: p.old.meta.update(val=torch.empty(8, dtype=torch.float16)),
        lambda p: p.update.meta.update(val=torch.empty(8, dtype=torch.float16)),
        lambda p: p.where.meta.update(val=torch.empty(8, dtype=torch.float16)),
        lambda p: p.update.meta.update(val=torch.empty(1, dtype=torch.float32)),
        lambda p: p.old.meta.update(val=torch.empty(1, dtype=torch.float32)),
        lambda p: p.tensor.meta.clear(),
    ],
)
def test_rejects_mismatches_without_changing_graph(
    change: Callable[[_Program], None],
) -> None:
    program = _program()
    change(program)
    original = str(program.graph)
    assert fold_noop_stores(program.graph) == 0
    assert str(program.graph) == original


@pytest.mark.parametrize("extra_shape,dtype", [((8,), torch.int32), ((3,), torch.bool)])
def test_rejects_unproven_extra_mask(
    extra_shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    program = _program(masked=True)
    program.extra_mask.meta["val"] = torch.empty(extra_shape, dtype=dtype)
    assert fold_noop_stores(program.graph) == 0


def test_rejects_recomputed_index_even_with_same_runtime_value() -> None:
    program = _program()
    with program.graph.inserting_before(program.old):
        equivalent = program.graph.call_function(operator.add, (program.index, 0))
    program.old.args = (program.tensor, [equivalent])
    assert fold_noop_stores(program.graph) == 0


def test_rejects_cast_between_load_and_where() -> None:
    program = _program()
    with program.graph.inserting_before(program.where):
        converted = program.graph.call_function(
            torch.ops.aten._to_copy.default, (program.old,), {"dtype": torch.float32}
        )
    converted.meta["val"] = torch.empty(8, dtype=torch.float32)
    program.where.args = (program.condition, program.update, converted)
    assert fold_noop_stores(program.graph) == 0


def test_second_preserve_store_cannot_reuse_load_before_first_store() -> None:
    program = _program()
    output = next(node for node in program.graph.nodes if node.op == "output")
    with program.graph.inserting_before(output):
        second = program.graph.call_function(
            memory_ops.store, (program.tensor, [program.index], program.where)
        )
    assert fold_noop_stores(program.graph) == 1
    assert second.args[2] is program.where
    assert program.old in program.graph.nodes
