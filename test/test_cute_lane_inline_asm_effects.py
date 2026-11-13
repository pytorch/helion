"""Opaque assembly must neither disappear nor be replayed by lane splitting."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.lane_loops.markers import HELION_VECTOR_REDUCTION_OWNER_ATTR
from helion._compiler.cute.lane_loops.markers import _create_lane_loop
from helion._compiler.cute.lane_loops.markers import _lane_reduce_marker_expr
from helion._compiler.cute.lane_loops.split import _live_phase2_indices
from helion._compiler.cute.lane_loops.split import (
    interchange_lane_outside_serial_reductions,
)
from helion._compiler.cute.lane_loops.split import split_lane_loop_reductions
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_lane_inline_asm import snapshot_before_opaque_producer
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _assembly(argument: str, purity: str = "False") -> str:
    return (
        f"pair = _cute_inline_asm_elementwise(({argument},), "
        "asm='mov.f32 $0, $2; mov.f32 $1, $2;', constraints='=f,=f,f', "
        "dtype=(cutlass.Float32, cutlass.Float32)"
        f", is_pure={purity}"
        ")"
    )


def _source(body: list[ast.AST]) -> str:
    return ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
        )
    )


def _lane_loop(*, shared: bool, purity: str = "False") -> ast.For:
    marker = _lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    statements = ["value = cutlass.Float32(lane)", _assembly("value", purity)]
    if shared:
        statements.append("value = pair[0]")
    statements += [
        f"reduced = {marker}",
        "(out.iterator + lane).store(value - reduced)",
    ]
    return _create_lane_loop("lane", 4, list(ast.parse("\n".join(statements)).body))


def test_phase2_keeps_an_unused_opaque_assembly_assignment() -> None:
    body = list(ast.parse("value = lane\n" + _assembly("value")).body)
    assert _live_phase2_indices(body) == {0, 1}


def test_phase2_can_drop_unused_declared_pure_assembly() -> None:
    body = list(ast.parse("value = lane\n" + _assembly("value", "True")).body)
    assert _live_phase2_indices(body) == set()


@pytest.mark.parametrize("shared", [False, True], ids=["unused", "shared"])
def test_lane_split_declines_an_opaque_effect(shared: bool) -> None:
    # Merely keeping the unused call in phase2 is insufficient: an opaque
    # effect might mutate the input read in phase1. Sharing its value also
    # replayed the call in both passes on the parent.
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        result = split_lane_loop_reductions([_lane_loop(shared=shared)])
        pytest.fail(_source(result))


def test_lane_split_declines_an_effect_between_reductions() -> None:
    marker = _lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    body = ast.parse(
        "value = cutlass.Float32(lane)\n"
        f"first = {marker}\n"
        f"{_assembly('value - first')}\n"
        "value = pair[0]\n"
        f"second = {marker}\n"
        "(out.iterator + lane).store(value - second)\n"
    ).body
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([_create_lane_loop("lane", 4, body)])


def _serial_loop(purity: str) -> ast.For:
    marker = _lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    body = ast.parse(
        "col = lane\n"
        "total = 0.0\n"
        "for row in range(3):\n"
        "    value = cutlass.Float32(row * 4 + col)\n"
        f"    {_assembly('value', purity)}\n"
        "    value = pair[0]\n"
        "    total = total + value\n"
        f"    reduced = {marker}\n"
        "    (out.iterator + row * 4 + col).store(value - reduced)\n"
        "(column.iterator + col).store(total)\n"
    ).body
    return _create_lane_loop("lane", 4, list(body))


def test_serial_interchange_declines_replaying_an_opaque_producer() -> None:
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        result = interchange_lane_outside_serial_reductions([_serial_loop("False")])
        pytest.fail(_source(result))


def test_pure_producers_keep_both_split_paths() -> None:
    for result in (
        split_lane_loop_reductions([_lane_loop(shared=True, purity="True")]),
        interchange_lane_outside_serial_reductions([_serial_loop("True")]),
    ):
        calls = [
            node
            for statement in result
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_cute_inline_asm_elementwise"
        ]
        assert len(calls) == 2, _source(result)


def test_an_unsplit_loop_preserves_the_effect() -> None:
    loop = _create_lane_loop("lane", 4, list(ast.parse(_assembly("lane")).body))
    original = _source([loop])
    body = split_lane_loop_reductions([loop])
    body = interchange_lane_outside_serial_reductions(body)
    assert _source(body) == original


@pytest.mark.parametrize("argument", ["value - reduced", "reduced"])
@pytest.mark.parametrize("use_output", [False, True])
def test_impure_epilogue_requires_lane_ownership(
    argument: str, use_output: bool
) -> None:
    marker = _lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = _create_lane_loop(
        "lane",
        4,
        ast.parse(
            "value = (x.iterator + lane).load()\n"
            f"reduced = {marker}\n"
            f"{_assembly(argument)}\n"
            f"(out.iterator + lane).store({'pair[0]' if use_output else 'value'})\n"
        ).body,
    )
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions(
            [loop], proven_disjoint_tensor_pairs={frozenset(("x", "out"))}
        )


def test_interchange_keeps_an_opaque_effect_when_it_restores_in_place() -> None:
    marker = _lane_reduce_marker_expr("value", "sum", "cutlass.Float32(0)", 1)
    body = ast.parse(
        "for row in range(3):\n"
        "    value = cutlass.Float32(lane + row)\n"
        f"    reduced = {marker}\n"
        f"    {_assembly('lane')}\n"
    ).body
    loop = _create_lane_loop("lane", 4, body)
    result = interchange_lane_outside_serial_reductions([loop])
    assert result == [loop]
    assert _source(result).count("_cute_inline_asm_elementwise(") == 1
    assert "reduced = value" in _source(result)


def test_interchange_preserves_the_register_tile_fallback() -> None:
    loop = _serial_loop("False")
    loop.iter = ast.parse("cutlass.range_constexpr(4)", mode="eval").body
    inner = next(stmt for stmt in loop.body if isinstance(stmt, ast.For))
    setattr(inner, HELION_VECTOR_REDUCTION_OWNER_ATTR, "lane")
    with pytest.raises(RegisterTileUnsupported, match="register-tile owner"):
        interchange_lane_outside_serial_reductions([loop])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _generated_multi_output(x: torch.Tensor, use_output: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :]
        first, second = hl.inline_asm_elementwise(
            "mov.f32 $0, $2; mov.f32 $1, $2;",
            "=f,=f,f",
            [row],
            (torch.float32, torch.float32),
            False,
            1,
        )
        value = first + second if use_output else row
        total = value.sum(-1, keepdim=True)
        out[tile, :] = value - total
    return out


@pytest.mark.parametrize("use_output", [False, True], ids=["unused", "shared"])
def test_actual_codegen_requires_a_single_execution_of_multi_output_assembly(
    use_output: bool,
) -> None:
    # One row per CTA and eight threads sweep 256 columns. Persistent reduction
    # keeps this in lane splitting, independent of roll_reduction's FX guard.
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_generated_multi_output, (torch.empty((8, 256)), use_output))
        config = helion.Config(
            block_sizes=[1],
            num_threads=[0, 8],
            cute_vector_widths=[1, 1],
        )
        if not use_output:
            with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
                source = bound.to_code(config)
                pytest.fail(source)
        else:
            source = bound.to_code(config)
            assert source.count("_cute_inline_asm_elementwise(") == 1, source
            assert "opaque_result_stash" in source, source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _generated_producer_only(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :]
        value = hl.inline_asm_elementwise(
            "mov.f32 $0, $1;", "=f,f", [row], torch.float32, False, 1
        )
        out[tile] = value.sum(-1)
    return out


def test_actual_codegen_keeps_a_producer_only_effect() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_generated_producer_only, (torch.empty((8, 256)),))
        source = bound.to_code(
            helion.Config(
                block_sizes=[1], num_threads=[0, 8], cute_vector_widths=[1, 1]
            )
        )
    assert source.count("_cute_inline_asm_elementwise(") == 1, source


@pytest.mark.parametrize("shared", [False, True])
def test_actual_opaque_memory_producer_uses_two_complete_cta_joins(
    shared: bool,
) -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            snapshot_before_opaque_producer,
            (
                torch.empty((2, 256)),
                torch.empty((2, 256), dtype=torch.uint64),
                torch.empty((2, 256), dtype=torch.uint64),
                shared,
            ),
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[1],
                num_threads=[0, 64],
                cute_vector_widths=[1, 1],
            )
        )
    assert source.count("_cute_inline_asm_elementwise(") == 1, source
    assert source.count("cute.arch.sync_threads()") == 2, source
    assert "atom.global.add.u32" in source and "st.global.u32" in source


def test_epilogue_cannot_reread_inputs_after_an_opaque_write() -> None:
    marker = _lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 1, owner_lane="lane"
    )
    loop = _create_lane_loop(
        "lane",
        4,
        ast.parse(
            "value = (x.iterator + lane).load()\n"
            f"reduced = {marker}\n"
            f"{_assembly('value - reduced')}\n"
            "(out.iterator + lane).store(value)\n"
        ).body,
    )
    # An opaque call may overwrite the next lane's x. The original whole-row
    # load observes [0, 1, 2, 3], while an interleaved reload observes [0, 100,
    # 100, 100]. Disjoint x/out does not constrain that hidden assembly write.
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions(
            [loop], proven_disjoint_tensor_pairs={frozenset(("x", "out"))}
        )


def _snapshot_producer_loop(dtype: str = "Float32", shared: bool = False) -> ast.For:
    marker = _lane_reduce_marker_expr(
        "value", "sum", f"cutlass.{dtype}(0)", 1, owner_lane="lane"
    )
    return _create_lane_loop(
        "lane",
        4,
        ast.parse(
            "index = tid * 4 + lane\n"
            "input_value = (x.iterator + index).load()\n"
            f"{_assembly('input_value')}\n"
            "value = pair[0]\n"
            f"reduced = {marker}\n"
            + (
                "(out.iterator + index).store(value - reduced)\n"
                if shared
                else "(out.iterator + tid).store(reduced)\n"
            )
        ).body,
    )


@pytest.mark.parametrize("dtype", ["Float32", "Int64"])
@pytest.mark.parametrize("shared", [False, True])
def test_producer_snapshots_all_threads_before_opaque_writes(
    dtype: str, shared: bool
) -> None:
    result = split_lane_loop_reductions([_snapshot_producer_loop(dtype, shared)])
    assert _source(result).count("_cute_inline_asm_elementwise(") == 1

    class PauseAtBarrier(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.AST:
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.arch.sync_threads"
            ):
                return ast.Expr(value=ast.Yield(value=ast.Constant(value="barrier")))
            return self.generic_visit(node)

    # Execute the generated scalar statements for two physical threads, pausing
    # each at a real emitted CTA barrier. A per-thread stash without that join
    # lets thread zero overwrite values thread one has not loaded yet.
    function = ast.parse("def kernel():\n    pass\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = cast("list[ast.stmt]", result)
    module = ast.fix_missing_locations(
        PauseAtBarrier().visit(ast.Module(body=[function], type_ignores=[]))
    )
    code = compile(module, "<opaque-producer-snapshot>", "exec")

    class Pointer:
        def __init__(
            self, values: list[float | int], events: list, tid: int, index: int = 0
        ) -> None:
            self.values, self.events, self.tid, self.index = values, events, tid, index

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.values, self.events, self.tid, self.index + offset)

        def load(self) -> float | int:
            value = self.values[self.index]
            self.events.append(("load", self.tid, self.index, value))
            return value

        def store(self, value: float) -> None:
            self.events.append(("store", self.tid, self.index, value))
            self.values[self.index] = value

    for offset in (0, 20):
        base = 2**60 if dtype == "Int64" else 0.0
        original = [base + offset + i for i in range(8)]
        memory, outputs, events = original.copy(), [-1] * (8 if shared else 2), []
        calls: list[list[float | int]] = [[], []]
        workers = []
        for tid in range(2):

            def assembly(
                values: tuple,
                tid: int = tid,
                calls: list = calls,
                events: list = events,
                memory: list = memory,
                **kwargs: object,
            ) -> tuple:
                (value,) = values
                index = tid * 4 + len(calls[tid])
                calls[tid].append(value)
                events.append(("asm", tid, index, value))
                memory[(index + 4) % 8] = -100
                return value, value

            namespace = {
                "tid": tid,
                "x": SimpleNamespace(iterator=Pointer(memory, events, tid)),
                "out": SimpleNamespace(iterator=Pointer(outputs, events, tid)),
                "cutlass": SimpleNamespace(
                    Float32=float, Int64=int, range_constexpr=range
                ),
                "_cute_inline_asm_elementwise": assembly,
            }
            exec(code, namespace)
            workers.append(namespace["kernel"]())
        for worker in workers:
            assert next(worker) == "barrier"
        assert [event[0] for event in events] == ["load"] * 8
        for worker in workers:
            assert next(worker) == "barrier"
        assert calls == [original[:4], original[4:]]
        assert [event[0] for event in events[8:]] == ["asm"] * 8
        for worker in workers:
            with pytest.raises(StopIteration):
                next(worker)
        expected = [sum(original[:4]), sum(original[4:])]
        if shared:
            expected = [
                value - expected[index // 4] for index, value in enumerate(original)
            ]
        assert outputs == expected
        assert memory == [-100] * 8


def test_producer_barriers_decline_a_divergent_enclosing_guard() -> None:
    wrapper = ast.parse("if tid == 0:\n    pass\n").body[0]
    assert isinstance(wrapper, ast.If)
    wrapper.body = [_snapshot_producer_loop()]
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([wrapper])


def test_producer_declines_an_incomplete_outer_lane_snapshot() -> None:
    outer = _create_lane_loop("row_lane", 2, [_snapshot_producer_loop()])
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([outer])


def test_producer_declines_a_post_effect_load_and_a_carried_prefix() -> None:
    for statement in (
        "value = (x.iterator + lane).load()",
        "value = pair[0] + input_value",
    ):
        loop = _snapshot_producer_loop()
        if "load" in statement:
            loop.body[3] = ast.parse(statement).body[0]
        else:
            loop.body[1] = ast.parse("input_value = input_value + lane").body[0]
        with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
            split_lane_loop_reductions([loop])


def test_producer_declines_an_unproved_tensor_subscript_after_the_effect() -> None:
    loop = _snapshot_producer_loop()
    loop.body[3] = ast.parse("value = shared_memory[lane]").body[0]
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([loop])


def test_producer_declines_a_load_hidden_in_the_opaque_call_operands() -> None:
    loop = _snapshot_producer_loop()
    loop.body[2] = ast.parse(_assembly("(x.iterator + index).load()")).body[0]
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([loop])


def test_producer_barrier_declines_a_preceding_conditional_exit() -> None:
    prefix = ast.parse("if tid == 0:\n    return\n").body
    with pytest.raises(exc.BackendUnsupported, match="inline assembly"):
        split_lane_loop_reductions([*prefix, _snapshot_producer_loop()])


def test_producer_keeps_barriers_outside_reduction_owner_guards() -> None:
    result = split_lane_loop_reductions([_snapshot_producer_loop()])
    barriers = [
        stmt
        for stmt in result
        if isinstance(stmt, ast.Expr)
        and ast.unparse(stmt.value) == "cute.arch.sync_threads()"
    ]
    assert len(barriers) == 2
    assert not any(
        "sync_threads" in ast.unparse(stmt)
        for stmt in result
        if isinstance(stmt, (ast.For, ast.If))
    )
