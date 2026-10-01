"""Prepared continuation: original native regions and selected relation authority."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import blackwell_helpers
import pytest
import torch

from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute import chunk_recurrence_sm100 as original
from helion._compiler.cute import prepared_tcgen_edge as shared
from helion._compiler.cute.prepared_continuation import ContinuationIssue
from helion._compiler.cute.prepared_continuation import ContinuationOpcode
from helion._compiler.cute.prepared_continuation import PreparedBodyLowering


@pytest.fixture(autouse=True)
def restore_shared_preprocessing_state():
    # Real IR construction performs CuTe's lazy code replacement. Restore only
    # these called module functions so later tests can observe their own first
    # preprocessing transition, independent of test ordering.
    originals = []
    for function in vars(shared).values():
        if isinstance(function, FunctionType) and "__wrapped__" in vars(function):
            inner = vars(function)["__wrapped__"]
            assert isinstance(inner, FunctionType)
            originals.append((inner, inner.__code__, dict(vars(inner))))
    try:
        yield
    finally:
        for inner, code, fields in originals:
            inner.__code__ = code
            vars(inner).clear()
            vars(inner).update(fields)


def _restore_epoch_edge(tree):
    """Undo only reviewed typed-state leaves and the two bound R operands.

    Added arithmetic/store leaves are checked in full, including decorators.
    The original continuation inverse and its original baseline digest then
    check every remaining statement, equation, argument and control edge.
    """
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    leaves = {
        "pack_layout_f_state": "65cc7a3481b015e171a9efc3f95ea79927493b8b02349fe18dc6f44f54d6a56e",
        "scale_layout_f_state": "af00e90ef708b240f0467f74d6c8c92acbad7accad3c32ead2a074fec3e8326c",
        "execute_prepared_store": "496de6a40f37d46f41670e5daab434f0a84d7d22d31edd77fc9a086fc186cdfc",
        "execute_prepared_store_completion": "a808e3541b19e152b4c5d62f030cf89bcce3031fab534a6d52b1d759e5699da8",
    }
    for name, expected in leaves.items():
        function = functions[name]
        assert hashlib.sha256(ast.dump(function).encode()).hexdigest() == expected
        tree.body.remove(function)
    for name in ("fmul2", "pack_input_b16x2_to_i32"):
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "affine_recurrence_primitives"
            and node.level == 1
            and [(item.name, item.asname) for item in node.names] == [(name, None)]
        ]
        assert len(matches) == 1
        tree.body.remove(matches[0])
    literal_import = ast.parse("from typing import Literal").body[0]
    imports = [node for node in tree.body if ast.dump(node) == ast.dump(literal_import)]
    assert len(imports) == 1
    tree.body.remove(imports[0])
    expected_function = ast.parse(
        "def expected(B_MAJOR: cutlass.Constexpr[Literal[0, 1]] = 0, "
        "B_BASE_BYTES: cutlass.Constexpr[int] = 0): pass"
    ).body[0]
    assert isinstance(expected_function, ast.FunctionDef)
    expected_args = expected_function.args
    for name in ("_prepare_descriptor_issue", "execute_prepared_issue"):
        function = functions[name]
        assert [ast.dump(arg) for arg in function.args.args[-2:]] == [
            ast.dump(arg) for arg in expected_args.args
        ]
        assert [ast.dump(arg) for arg in function.args.defaults[-2:]] == [
            ast.dump(arg) for arg in expected_args.defaults
        ]
        function.args.args = function.args.args[:-2]
        function.args.defaults = function.args.defaults[:-2]
    descriptor = functions["_prepare_descriptor_issue"]
    majors = [
        keyword
        for node in ast.walk(descriptor)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "prims.Tcgen05InstrDesc.build"
        for keyword in node.keywords
        if keyword.arg == "b_major"
    ]
    assert len(majors) == 1 and ast.unparse(majors[0].value) == "B_MAJOR"
    majors[0].value = ast.Constant(0)
    for text in (
        "assert B_MAJOR in (0, 1) and B_BASE_BYTES >= 0",
        (
            "if cutlass.const_expr(B_BASE_BYTES):\n"
            "    b = b.advance_start_address(B_BASE_BYTES)"
        ),
    ):
        expected = ast.dump(ast.parse(text).body[0])
        matches = [node for node in descriptor.body if ast.dump(node) == expected]
        assert len(matches) == 1
        descriptor.body.remove(matches[0])
    calls = [
        node
        for node in ast.walk(functions["execute_prepared_issue"])
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_prepare_descriptor_issue"
    ]
    assert len(calls) == 1 and not calls[0].keywords
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "a",
        "b",
        "accumulator",
        "operation",
        "B_SWIZZLE",
        "B_MAJOR",
        "B_BASE_BYTES",
    ]
    calls[0].args = calls[0].args[:-2]
    publication = functions["execute_prepared_publication"]
    branch = publication.body[1]
    assert isinstance(branch, ast.If)
    for statements, start, descriptor_mode in (
        (branch.body, 1, True),
        (branch.orelse, 2, False),
    ):
        expected = ast.parse(
            "execute_prepared_store(values, destination, copy, "
            f"{descriptor_mode!r}, STORE_SHAPE)\n"
            f"execute_prepared_store_completion({descriptor_mode!r})"
        ).body
        assert [ast.dump(node) for node in statements[start : start + 2]] == [
            ast.dump(node) for node in expected
        ]
        original = (
            "prims.tcgen05_st(STORE_SHAPE, destination, values)\n"
            "prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)\n"
            "prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)"
            if descriptor_mode
            else "cute.copy(copy, values, destination)\n"
            "cute.arch.fence_view_async_tmem_store()"
        )
        statements[start : start + 2] = ast.parse(original).body
    return tree


def _restore_original_edge(tree):
    tree = _restore_epoch_edge(tree)
    added = {"_continuation_control", "execute_prepared_continuation"}
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert added <= functions.keys()
    tree.body = [
        node for node in tree.body if node not in [functions[n] for n in added]
    ]
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "warp_specialized_primitives"
        and [(n.name, n.asname) for n in node.names]
        == [("elect_arrive_mbarrier", None)]
    ]
    assert len(imports) == 1
    tree.body.remove(imports[0])
    for name in ("_prepare_descriptor_issue", "execute_prepared_issue"):
        function = functions[name]
        assert function.args.args[-1].arg == "B_SWIZZLE"
        assert ast.literal_eval(function.args.defaults[-1]) == 128
        function.args.args.pop()
        function.args.defaults.pop()
    descriptor = functions["_prepare_descriptor_issue"]
    guards = [
        node
        for node in descriptor.body
        if isinstance(node, ast.Assert)
        and ast.unparse(node.test) == "B_SWIZZLE in (32, 128)"
    ]
    assert len(guards) == 1
    descriptor.body.remove(guards[0])
    layouts = [
        keyword
        for node in ast.walk(descriptor)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "layout"
    ]
    assert len(layouts) == 1 and isinstance(layouts[0].value, ast.IfExp)
    layout = layouts[0].value
    assert ast.unparse(layout.test) == "B_SWIZZLE == 128"
    assert ast.unparse(layout.orelse) == "prims.Tcgen05SmemSwizzle.SWIZZLE_32B"
    layouts[0].value = layout.body
    calls = [
        node
        for node in ast.walk(functions["execute_prepared_issue"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prepare_descriptor_issue"
    ]
    assert len(calls) == 1 and ast.unparse(calls[0].args[-1]) == "B_SWIZZLE"
    calls[0].args.pop()
    return tree


def _check_original_edge(tree):
    # Complete sealed v14 module, including decorators and every original helper.
    # The existing issue-region digest also covers both newly added executors.
    restored = _restore_original_edge(tree)
    assert hashlib.sha256(ast.dump(restored).encode()).hexdigest() == (
        "6eae1205161f3c62e5b8ae21d4c223b57b938cde9d2c7084c39d9f5939f52668"
    )


def test_complete_original_edge_module_inverse():
    _check_original_edge(ast.parse(Path(shared.__file__).read_text()))


@pytest.mark.parametrize(
    "mutation", ["issue_body", "read_body", "decorator", "default", "layout"]
)
def test_complete_original_edge_inverse_rejects_unrelated_change(mutation):
    tree = ast.parse(Path(shared.__file__).read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if mutation in ("issue_body", "read_body"):
        name = "_issue_atom" if mutation == "issue_body" else "execute_prepared_read"
        functions[name].body.append(ast.Pass())
    elif mutation == "decorator":
        functions["execute_prepared_read"].decorator_list.append(ast.Name(id="foreign"))
    elif mutation == "default":
        functions["execute_prepared_issue"].args.defaults[-1] = ast.Constant(32)
    else:
        descriptor = functions["_prepare_descriptor_issue"]
        layout = next(
            node for node in ast.walk(descriptor) if isinstance(node, ast.IfExp)
        )
        layout.orelse = ast.Name(id="foreign_layout")
    with pytest.raises(AssertionError):
        _check_original_edge(tree)


def walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for view in block.operations:
                yield from walk(view.operation)


@pytest.mark.parametrize("dtype", [cutlass.BFloat16, cutlass.Float16])
@pytest.mark.parametrize("schedule", ["full", "serial64", "overlap64"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_original_m128_fragment_layout_and_alias(tmp_path, dtype, schedule, dynamic):
    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    before_cuda = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "original_fragment", ([ir.IntegerType.get_signless(1)], [])
            )
        with ir.InsertionPoint(function.add_entry_block()):
            half = schedule != "full"
            major_b = (
                cute.nvgpu.OperandMajorMode.MN
                if half
                else cute.nvgpu.OperandMajorMode.K
            )
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                major_b,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, 64),
                tcgen05.OperandSource.SMEM,
            )
            original_slice = mma.get_slice(0)
            trait, original_value = mma._trait, mma._trait.value

            def tensor(shape, address, mn=False):
                atom = (
                    tcgen05.SmemLayoutAtomKind.MN_SW128
                    if mn
                    else tcgen05.SmemLayoutAtomKind.K_SW128
                )
                layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(atom, dtype),
                    shape,
                    order=(1, 0) if mn else (0, 1),
                )
                pointer = cute.make_ptr(
                    dtype, address, cute.AddressSpace.smem, assumed_align=128
                )
                return cute.make_tensor(
                    cute.recast_ptr(pointer, layout.inner, dtype=dtype), layout.outer
                )

            a = mma.make_fragment_A(original_slice.partition_A(tensor((128, 128), 0)))
            b = mma.make_fragment_B(
                original_slice.partition_B(tensor((64, 128), 32768, half))
            )
            accumulator = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C((128, 64))).layout,
            )
            bar = cute.make_ptr(
                cutlass.Int64, 49152, cute.AddressSpace.smem, assumed_align=8
            )
            predicate = cutlass.Boolean(function.arguments[0]) if dynamic else False
            count = 4 if half else 8
            retire = schedule != "overlap64"
            program = ((4, 0, 0, count, half, retire, retire),)
            for index in range(2 if half else 1):
                port = (
                    a,
                    b,
                    accumulator,
                    mma,
                    bar,
                    index,
                    False,
                    index * count,
                    16,
                    (),
                    128,
                )
                shared.execute_prepared_continuation(
                    program, (port,), (bar,), (index,), 0, predicate, False
                )
            if not retire:
                shared.execute_prepared_continuation(
                    ((6, 0, False), (0, 0, False)),
                    (),
                    (bar,),
                    (0,),
                    0,
                    predicate,
                    False,
                )
            assert original_slice._trait is trait and trait.value == original_value
            original_slice.partition_C(cute.make_identity_tensor((128, 64)))
            func.ReturnOp([])
        assert module.operation.verify()
        operations = list(walk(function.operation))
        gemms = [op for op in operations if op.name == "cute.gemm"]
        assert len(gemms) == 8
        for index, gemm in enumerate(gemms):
            setter = gemm.operands[0].owner
            assert setter.name == "cute_nvgpu.atom.set_value"
            flag = ir.IntegerAttr(setter.operands[1].owner.attributes["value"])
            assert bool(flag.value) == (half or index != 0)
        assert sum(op.name == "nvvm.tcgen05.commit" for op in operations) == (
            2 if schedule == "serial64" else 1
        )
        top = [view.operation for view in function.entry_block.operations]
        waits = [op for op in top if op.name == "nvvm.mbarrier.try_wait.parity"]
        assert len(waits) == (2 if schedule == "serial64" else 1)
        assert not any(op.name in ("cute.gemm", "nvvm.tcgen05.commit") for op in top)
        assert [op for op in top if op.name == "cute.tiled.mma.partition"][-1].operands[
            0
        ] == original_value
        (tmp_path / "module.mlir").write_text(str(module))
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize("iteration", [0, 1, "dynamic"])
def test_original_m64_sw128_to_sw32_descriptor_ports(tmp_path, iteration):
    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    before_cuda = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "original_descriptor",
                (
                    [ir.IntegerType.get_signless(32), ir.IntegerType.get_signless(32)],
                    [],
                ),
            )
        with ir.InsertionPoint(function.add_entry_block()):
            raw = cutlass.Int32(function.arguments[0])
            step = (
                cutlass.Int32(function.arguments[1])
                if iteration == "dynamic"
                else cutlass.Int32(iteration)
            )
            barriers = cutlass.Array(
                cutlass.Int64, 16, space=cutlass.AddressSpace.smem, alignment=8
            )
            qd = cutlass.Array(
                cutlass.BFloat16,
                original.TILE_ELEMS,
                space=cutlass.AddressSpace.smem,
                alignment=128,
            )
            qk = cutlass.Array(
                cutlass.BFloat16,
                original.QK_REC_ELEMS,
                space=cutlass.AddressSpace.smem,
                alignment=128,
            )
            acc = (raw, 0, original.tcgen05_qstate_acc_tmem_col_offset(0), 0)
            ports = (
                (
                    (raw, original.KDA_TMEM_STATE_AS_INPUT_COL_OFFSET),
                    (
                        qd,
                        original.TCGEN05_STATE_K_B_LEADING_BYTES,
                        original.TCGEN05_STATE_K_B_STRIDE_BYTES,
                    ),
                    acc,
                    (cutlass.BFloat16, original.DV_HALF, original.BT),
                    barriers.subview(8),
                    0,
                    True,
                    0,
                    original.TCGEN05_F16_K_ATOM,
                    (
                        original.TCGEN05_F16_ELEM_BYTES,
                        original.TCGEN05_SW128_K_PHASES_PER_SLICE,
                        original.BT,
                        original.TCGEN05_SW128_BYTES,
                    ),
                    128,
                ),
                (
                    (raw, original.tcgen05_shared_input_tmem_col_offset(0)),
                    (
                        qk,
                        original.TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES,
                        original.TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES,
                    ),
                    acc,
                    (cutlass.BFloat16, original.DV_HALF, original.BT),
                    barriers.subview(9),
                    0,
                    True,
                    0,
                    original.TCGEN05_F16_K_ATOM,
                    (2, 1, original.BT, 32),
                    32,
                ),
            )
            program = (
                (5, ((0, 0, False), (3, 1, False))),
                (0, 2, False),
                (1, 3, True),
                (0, 4, False),
                (4, 0, 0, 8, False, True, False),
                (1, 5, True),
                (4, 1, 0, 1, True, True, False),
            )
            shared.execute_prepared_continuation(
                program,
                ports,
                (
                    (barriers, 2),
                    (barriers.subview(2), 3),
                    barriers.subview(5),
                    barriers.subview(6),
                    barriers.subview(7),
                    barriers.subview(10),
                ),
                (0, 0, 0, 0, 0, 0),
                step,
                True,
                True,
            )
            func.ReturnOp([])
        assert module.operation.verify()
        text = str(module)
        operations = list(walk(function.operation))
        mmas = [op for op in operations if op.name == "nvvm.tcgen05.mma"]
        assert len(mmas) == 9
        for index, mma in enumerate(mmas):
            assert bool(
                ir.IntegerAttr(mma.operands[4].owner.attributes["value"]).value
            ) == (index != 0)
            assert "cta_1" in str(mma.attributes["ctaGroup"])
        descs = [op for op in operations if op.name == "nvvm.tcgen05.mma_smem_desc"]
        assert len(descs) == 2
        assert [
            ir.IntegerAttr(op.operands[-1].owner.attributes["value"]).value
            for op in descs
        ] == [2, 6]
        ordered = [
            op.name
            for op in operations
            if op.name
            in ("nvvm.tcgen05.mma", "nvvm.tcgen05.commit", "nvvm.tcgen05.fence")
        ]
        assert ordered == [
            "nvvm.tcgen05.fence",
            *(["nvvm.tcgen05.mma"] * 8),
            "nvvm.tcgen05.commit",
            "nvvm.tcgen05.fence",
            "nvvm.tcgen05.mma",
            "nvvm.tcgen05.commit",
        ]
        (tmp_path / "module.mlir").write_text(text)
    assert torch.cuda.is_initialized() == before_cuda


def test_foreign_relation_on_original_plan_rejects():
    from test import test_cute_chunk_recurrence as original

    planner, emit = chunk_recurrence._plan_chunk_recurrence, body.emit_body_program
    observed = []

    def select(*args, **kwargs):
        return planner(*args, **kwargs, prepared_continuation=True)

    def check(*args, **kwargs):
        current = kwargs.get("prepared_body")
        assert current is not None and current.root_action is None
        original_plan = current.owner.plan
        foreign = replace(current.continuation)
        actions = tuple(
            replace(a, continuation=foreign) if isinstance(a, ContinuationIssue) else a
            for a in current.actions()
        )
        with pytest.raises(
            chain._UnsupportedChain, match="foreign prepared external body"
        ):
            PreparedBodyLowering(
                current.codegen,
                current.owner,
                foreign,
                body.BodyProgram(actions),
                current.event_count,
                current.issue_ranges,
            )
        assert (
            current.owner.plan is original_plan
            and original_plan.prepared_continuation is current.continuation
        )
        result = emit(*args, **kwargs)
        assert current.payload
        observed.append(True)
        return result

    with (
        patch.object(chunk_recurrence, "_plan_chunk_recurrence", select),
        patch.object(body, "emit_body_program", check),
    ):
        original._code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert observed == [True]


def test_only_implemented_opcodes_advertised():
    assert {item.name: item.value for item in ContinuationOpcode} == {
        "WAIT": 0,
        "WAIT_TOGGLE": 1,
        "RELEASE": 3,
        "ISSUE": 4,
        "PREVIOUS": 5,
        "COMMIT": 6,
    }


@pytest.mark.parametrize(
    "mutation", ["port", "toggle_type", "fence", "end", "init", "owner", "graph"]
)
def test_actual_selected_action_mutation_rejects(mutation):
    from test import test_cute_chunk_recurrence as original

    planner, emit = chunk_recurrence._plan_chunk_recurrence, body.emit_body_program
    observed = []

    def select(*args, **kwargs):
        return planner(*args, **kwargs, prepared_continuation=True)

    def check(*args, **kwargs):
        assert set(kwargs) == {"prepared_body"}
        current = kwargs["prepared_body"]
        clone = PreparedBodyLowering(
            current.codegen,
            current.owner,
            current.continuation,
            current.program,
            current.event_count,
            current.issue_ranges,
        )
        issue = next(a for a in current.actions() if isinstance(a, ContinuationIssue))
        wait = current.actions()[1]
        changes = {
            "port": (wait, "port", 3),
            "toggle_type": (wait, "toggle", 0),
            "fence": (wait, "fence_after", True),
            "end": (issue, "k_end", issue.k_end - 1),
            "init": (issue, "initialized", True),
            "owner": (clone, "owner", replace(current.owner)),
        }
        if mutation == "graph":
            node = current.continuation.first.node
            original_args = node.args
            node.args = (*node.args[:3], torch.float16)
            try:
                with pytest.raises(chain._UnsupportedChain):
                    emit(*args, prepared_body=clone)
            finally:
                node.args = original_args
        else:
            target, field, value = changes[mutation]
            old = getattr(target, field)
            object.__setattr__(target, field, value)
            try:
                with pytest.raises(chain._UnsupportedChain):
                    emit(*args, prepared_body=clone)
            finally:
                object.__setattr__(target, field, old)
        result = emit(*args, **kwargs)
        assert current.payload
        observed.append(True)
        return result

    with (
        patch.object(chunk_recurrence, "_plan_chunk_recurrence", select),
        patch.object(body, "emit_body_program", check),
    ):
        original._code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert observed == [True]
