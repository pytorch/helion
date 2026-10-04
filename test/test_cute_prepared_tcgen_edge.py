from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from textwrap import dedent
from typing import TYPE_CHECKING
from unittest.mock import patch

from cuda.bindings.driver import CUstream
import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_ptr
import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _config
from .test_cute_chained_loop_tmem_transport import _inputs
from .test_cute_chained_loop_tmem_transport import _packed_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chunk_recurrence import _code
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage as stage_module
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute import prepared_tcgen_edge
from helion._compiler.cute.prepared_tcgen_binding import PreparedProjectionHost
from helion._compiler.cute.prepared_tcgen_binding import PreparedTmemEdge
from helion.runtime.cute.launcher import cute_cuda_graph

if TYPE_CHECKING:
    from pathlib import Path


class _Restore(ast.NodeTransformer):
    """Expand only the three selected helper boundaries, not arbitrary math."""

    def visit_ImportFrom(self, node):
        if node.module == "helion._compiler.cute" and any(
            item.name == "prepared_tcgen_edge" for item in node.names
        ):
            assert len(node.names) == 1
            return None
        return node

    def visit_Assign(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "prepared_tcgen_edge.execute_prepared_read"
        ):
            (
                source,
                values,
                copy,
                companion,
                completion,
                phase,
                descriptor,
                shape,
                count,
            ) = call.args
            assert ast.literal_eval(descriptor) is False
            assert all(
                ast.literal_eval(item) is None
                for item in (companion, completion, phase, shape)
            )
            assert ast.literal_eval(count) == 0
            return ast.parse(
                f"cute.copy({ast.unparse(copy)}, {ast.unparse(source)}, {ast.unparse(values)})\n"
                "cute.arch.fence_view_async_tmem_load()"
            ).body
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call):
            return self.generic_visit(node)
        name = ast.unparse(call.func)
        if name == "prepared_tcgen_edge.execute_prepared_issue":
            (
                a,
                b,
                acc,
                operation,
                issuer,
                event,
                phase,
                ready,
                ready_phase,
                descriptor,
                tmem_a,
                begin,
                end,
                atom,
                advance,
                initialized,
                commit,
                wait,
            ) = call.args
            assert ast.literal_eval(descriptor) is False
            assert ast.literal_eval(tmem_a) is False
            assert ast.literal_eval(begin) == 0 and ast.literal_eval(atom) == 16
            assert ast.literal_eval(commit) is True and ast.literal_eval(wait) is True
            assert all(
                ast.literal_eval(item) is None for item in (ready, ready_phase, advance)
            )
            prefix = ast.unparse(a).removesuffix("_ra")
            k = prefix + "_kk"
            return ast.parse(
                f"if {ast.unparse(issuer)}:\n"
                f"    {ast.unparse(operation)}.set(tcgen05.Field.ACCUMULATE, {ast.unparse(initialized)})\n"
                f"    for {k} in cutlass.range_constexpr({ast.unparse(end)}):\n"
                f"        cute.gemm({ast.unparse(operation)}, {ast.unparse(acc)}, {ast.unparse(a)}[None, None, {k}], {ast.unparse(b)}[None, None, {k}], {ast.unparse(acc)})\n"
                f"        {ast.unparse(operation)}.set(tcgen05.Field.ACCUMULATE, True)\n"
                "    with cute.arch.elect_one():\n"
                f"        tcgen05.commit({ast.unparse(event)})\n"
                f"cute.arch.mbarrier_wait({ast.unparse(event)}, {ast.unparse(phase)})"
            ).body
        if name == "prepared_tcgen_edge.execute_prepared_publication":
            values, destination, copy, barrier, arrival, descriptor, shape = call.args
            assert ast.literal_eval(descriptor) is False
            assert ast.literal_eval(arrival) is None and ast.literal_eval(shape) is None
            return ast.parse(
                f"{ast.unparse(barrier)}.arrive_and_wait()\n"
                f"cute.copy({ast.unparse(copy)}, {ast.unparse(values)}, {ast.unparse(destination)})\n"
                "cute.arch.fence_view_async_tmem_store()\n"
                f"{ast.unparse(barrier)}.arrive_and_wait()"
            ).body
        return self.generic_visit(node)


@pytest.fixture(
    scope="module",
    params=[
        (dtype, warps) for dtype in (torch.bfloat16, torch.float16) for warps in (4, 8)
    ],
)
def selected_edge(request):
    dtype, warps = request.param

    def run(probe):
        config = _config(16, pipeline=True, consumer_warps=warps)
        args = _inputs("cpu", dtype)
        before = _source(_packed_sequence, args, config)
        captured = []
        original = stage_module.emit_stage

        def selected(cg, plan, boundaries, stage, geometry, phase, *args, **kwargs):
            transport = kwargs.get("tmem_output")
            if transport is None:
                return original(
                    cg, plan, boundaries, stage, geometry, phase, *args, **kwargs
                )
            edge = PreparedTmemEdge(
                cg,
                plan,
                boundaries,
                stage,
                geometry,
                phase,
                kwargs["execution"],
                transport,
            )
            result = original(
                cg,
                plan,
                boundaries,
                stage,
                geometry,
                phase,
                *args,
                prepared_edge=edge,
                **kwargs,
            )
            assert edge.completed(result)
            # Probe the actual completion boundary, before later stages are
            # permitted to publish additional boundaries or reuse the arena.
            probe(edge, result)
            assert edge.completed(result)
            captured.append((edge, result))
            return result

        with patch.object(stage_module, "emit_stage", selected):
            try:
                after = _source(_packed_sequence, args, config)
            except exc.InternalError as error:
                # Codegen wraps a rejection raised inside the real stage.
                # Expose that same cause to the contract tests; do not accept
                # a different exception or alter compiler fallback behavior.
                if isinstance(error.__cause__, chain._UnsupportedChain):
                    raise error.__cause__ from None
                raise
        assert len(captured) == 1
        return *captured[0], before, after

    return run


def test_actual_shared_source_whole_inverse_and_role_wait(selected_edge):
    edge, result, before, after = selected_edge(lambda edge, result: None)
    assert ast.dump(_Restore().visit(ast.parse(after))) == ast.dump(ast.parse(before))
    tree = ast.parse(after)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == f"{edge.execution.thread} < 128"
        ):
            assert "execute_prepared_issue" not in ast.unparse(node)
            assert edge.execution.sync not in ast.unparse(node)
    assert after.count("execute_prepared_issue(") == 1
    assert after.count("execute_prepared_read(") == 1
    assert after.count("execute_prepared_publication(") == 1


@pytest.mark.parametrize(
    "field", ["phase", "stage", "boundaries", "geometry", "execution", "transport"]
)
def test_foreign_current_binding_rejects(selected_edge, field):
    def probe(edge, result):
        old = getattr(edge, field)
        value = (
            (dict(old) if field == "boundaries" else replace(old))
            if field in ("boundaries", "geometry", "execution", "transport")
            else ("wrong_phase" if field == "phase" else old + 1)
        )
        try:
            setattr(edge, field, value)
            with pytest.raises(chain._UnsupportedChain):
                edge.completed(result)
        finally:
            setattr(edge, field, old)

    selected_edge(probe)


@pytest.mark.parametrize(
    "target", ["dtype", "expression", "mask", "slot", "alias", "nested_arg"]
)
def test_deep_typed_owner_point_changes_reject(selected_edge, target):
    def probe(edge, result):
        if target == "alias":
            mapping = edge.plan.tensor_aliases
            key = "review_extra_alias"
            assert key not in mapping
            mapping[key] = "foreign_view"
            try:
                with pytest.raises(chain._UnsupportedChain):
                    edge.completed(result)
            finally:
                mapping.pop(key)
        elif target == "nested_arg":
            node = edge.plan.dots[edge.stage]
            old = node.kwargs
            node.kwargs = {**old, "review_nested": {"seed": node.args[2]}}
            try:
                with pytest.raises(chain._UnsupportedChain):
                    edge.completed(result)
            finally:
                node.kwargs = old
        else:
            obj, field, value = {
                "dtype": (edge.transport, "dtype", "cutlass.Float32"),
                "expression": (edge.transport, "expression_lines", ("foreign = 0",)),
                "mask": (edge.transport, "masked_value", "cutlass.Float16(0)"),
                "slot": (
                    edge.transport.slot,
                    "column_offset",
                    edge.transport.slot.column_offset + 32,
                ),
            }[target]
            old = getattr(obj, field)
            try:
                object.__setattr__(obj, field, value)
                with pytest.raises(chain._UnsupportedChain):
                    edge.completed(result)
            finally:
                object.__setattr__(obj, field, old)

    selected_edge(probe)


def test_read_publication_duplicate_and_incomplete_reject(selected_edge):
    def probe(edge, result):
        fresh = replace(edge)
        prefix = f"chain_{fresh.stage}"
        with pytest.raises(chain._UnsupportedChain):
            fresh.read(prefix)
        with pytest.raises(chain._UnsupportedChain):
            fresh.finish(result)
        assert not fresh.completed(result)
        assert edge._stage_setup is not None and edge._load_setup is not None
        fresh.capture_stage_setup(list(edge._stage_setup))
        fresh.issue(prefix, True, False)
        assert not fresh.completed(result)
        with pytest.raises(chain._UnsupportedChain):
            fresh.issue(prefix, True, False)
        fresh.capture_load_setup(prefix, list(edge._load_setup))
        fresh.read(prefix)
        assert not fresh.completed(result)
        with pytest.raises(chain._UnsupportedChain):
            fresh.finish(result)
        with pytest.raises(chain._UnsupportedChain):
            edge.finish(result)

    selected_edge(probe)


@pytest.mark.parametrize("operation", ["issue", "read", "publication"])
@pytest.mark.parametrize("predicate", ["False", "first128", "foreign_role"])
def test_complete_steps_under_foreign_scope_reject(selected_edge, operation, predicate):
    def probe(edge, result):
        original = "\n".join(result).splitlines()
        step = edge._steps[{"issue": 0, "read": 1, "publication": 2}[operation]]
        matches = [
            index
            for index in range(len(original) - len(step) + 1)
            if tuple(
                dedent("\n".join(original[index : index + len(step)])).splitlines()
            )
            == step
        ]
        assert len(matches) == 1
        index = matches[0]
        indent = original[index][: len(original[index]) - len(original[index].lstrip())]
        condition = (
            f"{edge.execution.thread} < 128" if predicate == "first128" else predicate
        )
        changed = [
            *original[:index],
            indent + f"if {condition}:",
            *("    " + line for line in original[index : index + len(step)]),
            *original[index + len(step) :],
        ]
        ast.parse("\n".join(changed))
        # Reopen only the completion operation; preserve its actual three
        # successful emission records so this exercises lexical acceptance.
        complete = edge._complete
        edge._complete = None
        try:
            with pytest.raises(chain._UnsupportedChain):
                edge.finish(changed)
            assert edge._complete is None
        finally:
            edge._complete = complete

    selected_edge(probe)


@pytest.mark.parametrize("operation", ["issue", "read", "publication"])
def test_complete_steps_reordered_reject(selected_edge, operation):
    def probe(edge, result):
        changed = "\n".join(result).splitlines()

        def position(step):
            found = [
                index
                for index in range(len(changed) - len(step) + 1)
                if tuple(
                    dedent("\n".join(changed[index : index + len(step)])).splitlines()
                )
                == step
            ]
            assert len(found) == 1
            return found[0]

        ordinal = {"issue": 0, "read": 1, "publication": 2}[operation]
        step = edge._steps[ordinal]
        start = position(step)
        moved = changed[start : start + len(step)]
        changed[start : start + len(step)] = []
        if operation == "issue":
            insert = len(changed) - 1
        elif operation == "read":
            insert = position(edge._steps[2]) + len(edge._steps[2])
        else:
            insert = position(edge._steps[1])
        changed[insert:insert] = moved
        ast.parse("\n".join(changed))
        complete = edge._complete
        edge._complete = None
        try:
            with pytest.raises(chain._UnsupportedChain):
                edge.finish(changed)
            assert edge._complete is None
        finally:
            edge._complete = complete

    selected_edge(probe)


@pytest.mark.parametrize("operation", ["issue", "read", "publish"])
def test_returned_operation_cannot_add_unowned_control_flow(selected_edge, operation):
    original = getattr(PreparedTmemEdge, operation)

    def changed(edge, *args):
        return [*original(edge, *args), "if True:\n    return"]

    with (
        patch.object(PreparedTmemEdge, operation, changed),
        pytest.raises(chain._UnsupportedChain, match="complete stage changed"),
    ):
        selected_edge(lambda edge, lines: None)


@pytest.mark.parametrize("section", ["stage", "load"])
def test_setup_mutated_after_original_capture_rejects(selected_edge, section):
    method = f"capture_{section}_setup"
    original = getattr(PreparedTmemEdge, method)

    def changed(edge, *args):
        original(edge, *args)
        lines = args[-1]
        lines[0] = "review_corrupted_setup = 0"

    with (
        patch.object(PreparedTmemEdge, method, changed),
        pytest.raises(chain._UnsupportedChain, match="complete stage changed"),
    ):
        selected_edge(lambda edge, lines: None)


@pytest.mark.parametrize("change", ["interposed_effect", "early_return", "extra_join"])
def test_whole_stage_rejects_unowned_statements_before_final_join(
    selected_edge, change
):
    def probe(edge, result):
        extra = {
            "interposed_effect": f"chain_{edge.stage}_values.fill(0.0)",
            "early_return": "if True:\n    return",
            "extra_join": edge.execution.sync,
        }[change]
        changed = [*result[:-1], extra, result[-1]]
        complete = edge._complete
        edge._complete = None
        try:
            with pytest.raises(chain._UnsupportedChain):
                edge.finish(changed)
            assert edge._complete is None
        finally:
            edge._complete = complete

    selected_edge(probe)


def test_original_matched_host_default_source_and_mutation():
    before = _code(fp32_state=True, dv_partitions=2, pipeline="wide")
    original = chunk_recurrence._plan_chunk_recurrence
    captures = []

    def selected(*args, **kwargs):
        plan = original(*args, prepared_edge=True, **kwargs)
        if plan is not None:
            captures.append(PreparedProjectionHost(plan))
        return plan

    with patch.object(chunk_recurrence, "_plan_chunk_recurrence", selected):
        after = _code(fp32_state=True, dv_partitions=2, pipeline="wide")
    # Public lowering now consumes the full epoch. The explicitly selected
    # partial-edge route remains a distinct compatibility path; check both
    # complete sources rather than treating partial selection as public default.
    assert hashlib.sha256(before.encode()).hexdigest() == (
        "7183f47591f7bcfbba7f572b6e18bf891045ee859020e5882650c258d8da0a09"
    )
    assert hashlib.sha256(after.encode()).hexdigest() == (
        "c95ceb30c1e17c6522364b2fa42def36081dc2f0ffbda95e138fa0197511f2f5"
    )
    assert len(captures) == 1
    host = captures[0]
    assert host.host_keywords() == {"PREPARED_EDGE": True}
    assert host.plan.prepared_projection is not None
    node = host.plan.prepared_projection.source
    old = node.args
    try:
        node.args = (*old[:2], host.plan.prepared_projection.state, *old[3:])
        with pytest.raises(chain._UnsupportedChain):
            host.host_keywords()
    finally:
        node.args = old
    host.check()


@cute.kernel
def _state_abi_transfer_kernel(
    source: cute.Tensor,
    destination: cute.Tensor,
    valid_rows: cutlass.Int32,
    BEGIN: cutlass.Constexpr,
    COUNT: cutlass.Constexpr,
    READ: cutlass.Constexpr,
    NARROW: cutlass.Constexpr,
) -> None:
    row = cute.arch.thread_idx()[0]
    valid = row < valid_rows
    state_row = row
    if not valid:
        state_row = row + 1024
    if cutlass.const_expr(READ):
        if cutlass.const_expr(NARROW):
            values = prepared_tcgen_edge.read_linear_state_abi(
                source, (0, 0, state_row), BEGIN, COUNT, valid, MAX_BITS=128
            )
        else:
            values = prepared_tcgen_edge.read_linear_state_abi(
                source, (0, 0, state_row), BEGIN, COUNT, valid
            )
        for column in cutlass.range_constexpr(COUNT):
            destination[0, 0, row, column] = values[column]
    else:
        values = cutlass.Array(cutlass.Float32, COUNT, alignment=16)
        for column in cutlass.range_constexpr(COUNT):
            values[column] = source[0, 0, row, column]
        if cutlass.const_expr(NARROW):
            prepared_tcgen_edge.store_linear_state_abi(
                values,
                destination,
                (0, 0, state_row),
                BEGIN,
                COUNT,
                valid,
                MAX_BITS=128,
            )
        else:
            prepared_tcgen_edge.store_linear_state_abi(
                values, destination, (0, 0, state_row), BEGIN, COUNT, valid
            )


@cute.jit
def _state_abi_transfer_host(
    source_pointer: cute.Pointer,
    destination_pointer: cute.Pointer,
    valid_rows: cutlass.Int32,
    ROW_STRIDE: cutlass.Constexpr,
    INNER_STRIDE: cutlass.Constexpr,
    BEGIN: cutlass.Constexpr,
    COUNT: cutlass.Constexpr,
    READ: cutlass.Constexpr,
    NARROW: cutlass.Constexpr,
    stream: CUstream,
) -> None:
    source_stride = ROW_STRIDE if READ else 64
    source_inner = INNER_STRIDE if READ else 1
    destination_stride = 64 if READ else ROW_STRIDE
    destination_inner = 1 if READ else INNER_STRIDE
    source = cute.make_tensor(
        source_pointer,
        cute.make_layout(
            (1, 1, 32, 64),
            stride=(
                32 * source_stride,
                32 * source_stride,
                source_stride,
                source_inner,
            ),
        ),
    )
    destination = cute.make_tensor(
        destination_pointer,
        cute.make_layout(
            (1, 1, 32, 64),
            stride=(
                32 * destination_stride,
                32 * destination_stride,
                destination_stride,
                destination_inner,
            ),
        ),
    )
    _state_abi_transfer_kernel(
        source, destination, valid_rows, BEGIN, COUNT, READ, NARROW
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@pytest.mark.parametrize("read", [True, False], ids=["read", "store"])
@pytest.mark.parametrize(
    "row_stride,inner_stride,begin,count",
    [(132, 1, 0, 32), (132, 1, 4, 32), (132, 1, 1, 12), (260, 2, 0, 32)],
    ids=["divergent_alignment", "begin4", "partial_unaligned", "strided"],
)
def test_linear_state_abi_exact_bits_and_masked_addresses(
    read: bool,
    row_stride: int,
    inner_stride: int,
    begin: int,
    count: int,
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell or newer CUDA device")
    # No conversion through Python floats: preserve NaN payloads, signed zeros
    # and subnormals. Read and store have independent physical-index oracles.
    words = [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7FC12345,
        0xFFC54321,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x3F800000,
        0xBF012345,
    ]
    words = [word if word < 0x80000000 else word - 0x100000000 for word in words]
    source_stride, source_inner = (row_stride, inner_stride) if read else (64, 1)
    output_stride, output_inner = (64, 1) if read else (row_stride, inner_stride)
    source_begin, output_begin = (begin, 0) if read else (0, begin)
    stream = CUstream(torch.cuda.current_stream().cuda_stream)
    for narrow in (False, True):
        # Both orientations put 32-byte and 16-byte-only rows in the same warp.
        for offset in (0, 4):
            source_start = 8 + (offset if read else 0)
            output_start = 8 + (0 if read else offset)
            source_cpu = torch.tensor(
                [words[index % len(words)] for index in range(32 * source_stride + 32)],
                dtype=torch.int32,
            )
            poison_cpu = torch.full(
                (32 * output_stride + 32,), 0x6BADCAFE, dtype=torch.int32
            )
            for row in range(32):
                for column in range(64):
                    poison_cpu[
                        output_start + row * output_stride + column * output_inner
                    ] = 0x5A17DEAD
            source = source_cpu.cuda()
            poison = poison_cpu.cuda()
            output = poison.clone()
            source_pointer = make_ptr(
                cutlass.Float32,
                source.data_ptr() + 4 * source_start,
                cute.AddressSpace.gmem,
                assumed_align=32 if read and offset == 0 else 16,
            )
            destination_pointer = make_ptr(
                cutlass.Float32,
                output.data_ptr() + 4 * output_start,
                cute.AddressSpace.gmem,
                assumed_align=32 if not read and offset == 0 else 16,
            )
            if row_stride == 132 and begin % 4 == 0:
                state_address = (
                    source.data_ptr() + 4 * source_start
                    if read
                    else output.data_ptr() + 4 * output_start
                )
                assert {
                    (state_address + row * row_stride * 4) % 32 for row in range(32)
                } == {0, 16}
            native_dir = tmp_path / f"{narrow}-{offset}"
            native_dir.mkdir()
            compiled = cute.compile(
                _state_abi_transfer_host,
                source_pointer,
                destination_pointer,
                cutlass.Int32(32),
                row_stride,
                inner_stride,
                begin,
                count,
                read,
                narrow,
                stream,
                options=f"--enable-tvm-ffi --keep-ptx --dump-dir {native_dir}",
            )
            instruction = "ld.global.v4.b64" if read else "st.global.v4.b64"
            assert (instruction in compiled.__ptx__) == (
                not narrow and inner_stride == 1 and count % 8 == 0
            )
            for valid_rows in (32, 17, 0):
                expected = poison_cpu.clone()
                for row in range(32):
                    for column in range(count):
                        destination_index = (
                            output_start
                            + row * output_stride
                            + (output_begin + column) * output_inner
                        )
                        if row < valid_rows:
                            source_index = (
                                source_start
                                + row * source_stride
                                + (source_begin + column) * source_inner
                            )
                            expected[destination_index] = source_cpu[source_index]
                        elif read:
                            expected[destination_index] = 0
                call = (source_pointer, destination_pointer, cutlass.Int32(valid_rows))
                output.copy_(poison)
                compiled(*call, stream)
                assert torch.equal(output.cpu(), expected)
                assert torch.equal(source.cpu(), source_cpu)
                with cute_cuda_graph() as graph:
                    compiled(*call, CUstream(torch.cuda.current_stream().cuda_stream))
                for _replay in range(2):
                    output.copy_(poison)
                    graph.replay()
                    assert torch.equal(output.cpu(), expected)
                    assert torch.equal(source.cpu(), source_cpu)
                graph.reset()
