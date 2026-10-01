from __future__ import annotations

import ast
from contextlib import contextmanager
from contextlib import nullcontext
import copy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cutlass
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.base_dsl.utils.tree_utils import tree_flatten
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import blackwell_helpers
import pytest
import torch

from .test_cute_prepared_continuation import _check_original_edge
from helion._compiler.cute import prepared_tcgen_edge as shared

_ALIASES = {
    "issue_a": "a",
    "issue_b": "b",
    "issue_accumulator": "accumulator",
    "issue_operation": "operation",
}


def _module() -> ast.Module:
    assert shared.__file__ is not None
    return ast.parse(Path(shared.__file__).read_text())


def _function(tree: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_prepared_issue"
    )


def _issuer(function: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "issuer"
    )


def _without_fragment_receiver(tree: ast.Module) -> ast.Module:
    function = _function(tree)
    branch = _issuer(function)
    index = function.body.index(branch)
    assert [ast.unparse(node) for node in function.body[index - 2 : index]] == [
        "fragment_operation = None",
        "if cutlass.const_expr(not DESCRIPTOR):\n    fragment_operation = operation",
    ]
    function.body[index - 2 : index] = []
    preparation = branch.body[0]
    assert isinstance(preparation, ast.If)
    guard = preparation.orelse.pop(0)
    assert ast.unparse(guard) == "assert fragment_operation is not None"
    loop = branch.body[1]
    assert isinstance(loop, ast.For)
    accumulation = loop.body[-1]
    assert isinstance(accumulation, ast.If)
    guard = accumulation.body.pop(0)
    assert ast.unparse(guard) == "assert fragment_operation is not None"
    for node in ast.walk(branch):
        if isinstance(node, ast.Name) and node.id == "fragment_operation":
            node.id = "operation" if isinstance(node.ctx, ast.Load) else node.id
    # v12 used the issue-local alias as its .set receiver.
    for node in ast.walk(branch):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "operation"
        ):
            node.func.value.id = "issue_operation"
    return tree


def _restore(tree: ast.Module) -> ast.Module:
    tree = _without_fragment_receiver(tree)
    branch = _issuer(_function(tree))
    preparation = branch.body[0]
    assert isinstance(preparation, ast.If)
    alias = preparation.orelse.pop(0)
    assert isinstance(alias, ast.Assign)
    assert ast.unparse(alias) == (
        "issue_a, issue_b, issue_accumulator, issue_operation = "
        "(a, b, accumulator, operation)"
    )
    for node in ast.walk(branch):
        if isinstance(node, ast.Name) and node.id in _ALIASES:
            node.id = _ALIASES[node.id]
    return tree


def test_issue_alias_rewrite_preserves_surrounding_module():
    current = _module()
    _check_original_edge(current)
    restored = _restore(copy.deepcopy(current))
    current_issue, restored_issue = _function(current), _function(restored)
    assert ast.dump(current_issue.args) == ast.dump(restored_issue.args)
    assert [ast.dump(node) for node in current_issue.decorator_list] == [
        ast.dump(node) for node in restored_issue.decorator_list
    ]
    # The inverse changes only the issuer's aliases. Newly shared state/output
    # leaves must not be deleted or hashed as part of that issue-region proof.
    current.body.remove(current_issue)
    restored.body.remove(restored_issue)
    assert ast.dump(current) == ast.dump(restored)


@pytest.mark.parametrize("original", [False, True])
def test_installed_region_analysis(original):
    tree = _module()
    if original:
        tree = _restore(tree)
    function = _function(tree)
    active = {argument.arg for argument in function.args.args}
    if not original:
        active.add("fragment_operation")
    names, full_writes, closures = DSLPreprocessor(
        ["cutlass", "cute"]
    ).analyze_region_variables(_issuer(function), [active], [])
    assert closures == []
    assert names == (
        ["a", "b", "accumulator", "operation"] if original else ["fragment_operation"]
    )
    assert full_writes == (4 if original else 0)


def test_actual_preprocess_does_not_capture_incoming_operands(tmp_path):
    # Other tests can populate CuTe's in-place preprocessing cache. Load the
    # actual source afresh so this test always observes its first transition.
    assert shared.__file__ is not None
    path = tmp_path / "prepared_issue.py"
    path.write_text(Path(shared.__file__).read_text())
    spec = importlib.util.spec_from_file_location(
        "helion._compiler.cute._isolated_issue", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inner = vars(module.execute_prepared_issue)["__wrapped__"]
    code, fields = inner.__code__, dict(vars(inner))
    analyzer = DSLPreprocessor.analyze_region_variables
    observed = []
    before_cuda = torch.cuda.is_initialized()

    def capture(self, node, active_symbols, active_callables):
        result = analyzer(self, node, active_symbols, active_callables)
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "issuer"
        ):
            observed.append(result)
        return result

    try:
        with (
            patch.object(DSLPreprocessor, "analyze_region_variables", capture),
            patch(
                "cutlass.cute.compile", side_effect=AssertionError("native forbidden")
            ),
            patch(
                "torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")
            ),
        ):
            BaseDSL._preprocess_and_replace_code(inner)
        assert observed == [(["fragment_operation"], 0, [])]
        assert vars(inner)["_original_code"] is code
    finally:
        inner.__code__ = code
        vars(inner).clear()
        vars(inner).update(fields)
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize("dtype", [cutlass.BFloat16, cutlass.Float16])
def test_original_numeric_class_capture_fails(dtype):
    # The exact installed flatten path fails on the class, not a numeric value.
    # The correction excludes this readonly descriptor recipe from region args;
    # it does not modify the installed flatten implementation or dtype.
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'self'"
    ):
        tree_flatten((dtype, 64, 128))


@pytest.mark.parametrize("issuer", [False, True])
@pytest.mark.parametrize(
    ("descriptor", "ready", "begin", "end", "initialized", "commit", "wait"),
    [
        (False, False, 0, 2, False, True, True),
        (False, False, 2, 4, True, True, True),
        (True, True, 0, 4, False, False, False),
        (True, True, 4, 8, True, True, False),
        (True, False, 0, 8, False, True, False),
    ],
)
def test_original_order_and_general_mutable_operation(
    issuer, descriptor, ready, begin, end, initialized, commit, wait
):
    inputs = tuple(object() for _ in range(4))
    prepared = tuple(object() for _ in range(4))
    completion, input_ready = object(), object() if ready else None
    advance = object()

    def trace(original):
        events: list[tuple[object, ...]] = []

        class Operation:
            def set(self, field, value):
                events.append(("set", field, value))

        operation = (cutlass.BFloat16, 64, 128) if descriptor else Operation()

        def prepare(*values):
            assert values[:3] == inputs[:3] and values[3] is operation
            events.append(("prepare",))
            return prepared

        def atom(a, b, accumulator, current_operation, k, scale, *policy):
            expected = prepared if descriptor else (*inputs[:3], operation)
            assert (a, b, accumulator, current_operation) == expected
            assert policy == (descriptor, True, 16, advance, False)
            events.append(("atom", k, scale))

        namespace: dict[str, object] = {
            "cutlass": SimpleNamespace(
                const_expr=bool, range_constexpr=range, Int32=int
            ),
            "tcgen05": SimpleNamespace(
                Field=SimpleNamespace(ACCUMULATE="accumulate"),
                commit=lambda event: events.append(("commit", event)),
            ),
            "prims": SimpleNamespace(
                Tcgen05Fence=SimpleNamespace(AFTER_THREAD_SYNC="after"),
                CTAGroup=SimpleNamespace(CTA_1="one"),
                tcgen05_fence=lambda value: events.append(("fence", value)),
                elect_sync=lambda: True,
                tcgen05_commit=lambda event, group: events.append(
                    ("commit", event, group)
                ),
            ),
            "_prepare_descriptor_issue": prepare,
            "_issue_atom": atom,
            "execute_prepared_wait": lambda *values: events.append(("wait", *values)),
        }
        namespace["cute"] = SimpleNamespace(arch=SimpleNamespace(elect_one=nullcontext))
        tree = _restore(_module()) if original else _module()
        function = copy.deepcopy(_function(tree))
        function.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), "<issue-order-only>", "exec"),
            namespace,
        )
        execute = namespace["execute_prepared_issue"]
        assert callable(execute)
        phase = execute(
            *inputs[:3],
            operation,
            issuer,
            completion,
            0,
            input_ready,
            1,
            descriptor,
            True,
            begin,
            end,
            16,
            advance,
            initialized,
            commit,
            wait,
        )
        assert phase == (0 if ready else 1)
        assert sum(event[0] == "atom" for event in events) == (
            end - begin if issuer else 0
        )
        # Completion is outside the issuer predicate in the original common path.
        if wait:
            assert events[-1] == ("wait", completion, 0, False)
        return events

    assert trace(False) == trace(True)


def _old_issue(tmp_path: Path) -> Any:
    # Mechanically restore the actual v12 helper, not a separately written loop.
    path = tmp_path / "old_issue.py"
    path.write_text(ast.unparse(_without_fragment_receiver(_module())))
    spec = importlib.util.spec_from_file_location(
        "helion._compiler.cute._old_issue", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute_prepared_issue


def _issue_function_state():
    functions = (
        shared._issue_atom,
        shared.execute_prepared_wait,
        shared.execute_prepared_issue,
    )
    return tuple(
        (inner, inner.__code__, dict(vars(inner)))
        for inner in (vars(function)["__wrapped__"] for function in functions)
    )


@contextmanager
def _preserve_issue_functions():
    originals = _issue_function_state()
    try:
        yield
    finally:
        for inner, code, fields in originals:
            inner.__code__ = code
            vars(inner).clear()
            vars(inner).update(fields)


def _actual_alias_ir(execute, dtype, dynamic, begin, end, initialized, old=False):
    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    before_cuda = torch.cuda.is_initialized()
    with (
        _preserve_issue_functions(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "issue_alias", ([ir.IntegerType.get_signless(1)], [])
            )
        with ir.InsertionPoint(function.add_entry_block()):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, 32),
                tcgen05.OperandSource.SMEM,
            )
            original_slice = mma.get_slice(0)
            trait = mma._trait
            original_value = trait.value
            assert original_slice._trait is trait

            def tensor(shape, address):
                layout = cute.tile_to_shape(
                    tcgen05.make_smem_layout_atom(
                        tcgen05.SmemLayoutAtomKind.K_SW32, dtype
                    ),
                    shape,
                    order=(0, 1),
                )
                pointer = cute.make_ptr(
                    dtype, address, cute.AddressSpace.smem, assumed_align=128
                )
                return cute.make_tensor(
                    cute.recast_ptr(pointer, layout.inner, dtype=dtype), layout.outer
                )

            a = mma.make_fragment_A(original_slice.partition_A(tensor((128, 32), 0)))
            b = mma.make_fragment_B(original_slice.partition_B(tensor((32, 32), 8192)))
            accumulator = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C((128, 32))).layout,
            )
            bar = cute.make_ptr(
                cutlass.Int64, 16384, cute.AddressSpace.smem, assumed_align=8
            )
            predicate = cutlass.Boolean(function.arguments[0]) if dynamic else False
            execute(
                a,
                b,
                accumulator,
                mma,
                predicate,
                bar,
                0,
                None,
                None,
                False,
                False,
                begin,
                end,
                16,
                None,
                initialized,
                True,
                True,
            )
            # This actual caller alias existed before issue, as in emit_stage.
            assert original_slice._trait is trait and mma._trait is trait
            assert (trait.value == original_value) is (not old)
            original_slice.partition_C(cute.make_identity_tensor((128, 32)))
            func.ReturnOp([])
        if old:
            with pytest.raises(ir.MLIRError, match="does not dominate this use"):
                module.operation.verify()
        else:
            assert module.operation.verify()
            top = [view.operation for view in function.entry_block.operations]
            waits = [op for op in top if op.name == "nvvm.mbarrier.try_wait.parity"]
            assert len(waits) == 1
            partitions = [op for op in top if op.name == "cute.tiled.mma.partition"]
            assert partitions[-1].operands[0] == original_value
            regions = [op for op in top if op.name == "scf.if"]
            assert len(regions) == 1
            if not dynamic:
                flag = ir.IntegerAttr(regions[0].operands[0].owner.attributes["value"])
                assert not bool(flag.value)
            for issue in regions:
                assert top.index(issue) < top.index(waits[0])
                assert len(issue.results) == 1
                assert issue.results[0].type == original_value.type
                body = [
                    view.operation for view in issue.regions[0].blocks[0].operations
                ]
                gemms = [op for op in body if op.name == "cute.gemm"]
                assert len(gemms) == end - begin
                for index, gemm in enumerate(gemms):
                    setter = gemm.operands[0].owner
                    assert setter.name == "cute_nvgpu.atom.set_value"
                    flag = ir.IntegerAttr(setter.operands[1].owner.attributes["value"])
                    assert bool(flag.value) == (initialized or index != 0)
                assert body[-1].name == "scf.yield"
                assert body[-1].operands[0].owner.name == "cute_nvgpu.atom.set_value"
                other = list(issue.regions[1].blocks[0].operations)
                assert other[-1].operation.operands[0] == original_value
            assert not any(
                op.name in ("cute.gemm", "nvvm.tcgen05.commit") for op in top
            )
    assert torch.cuda.is_initialized() == before_cuda


@pytest.mark.parametrize("dtype", [cutlass.BFloat16, cutlass.Float16])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    ("begin", "end", "initialized"), [(0, 1, False), (0, 2, False), (1, 2, True)]
)
def test_actual_thr_mma_alias_after_issue(dtype, dynamic, begin, end, initialized):
    _actual_alias_ir(
        shared.execute_prepared_issue, dtype, dynamic, begin, end, initialized
    )


def test_actual_alias_ir_preserves_original_preprocessing_state():
    before = _issue_function_state()
    _actual_alias_ir(
        shared.execute_prepared_issue, cutlass.BFloat16, False, 0, 1, False
    )
    after = _issue_function_state()
    assert all(
        old_inner is new_inner and old_code is new_code and old_fields == new_fields
        for (old_inner, old_code, old_fields), (new_inner, new_code, new_fields) in zip(
            before, after, strict=True
        )
    )


@pytest.mark.parametrize("dtype", [cutlass.BFloat16, cutlass.Float16])
def test_original_issue_alias_dominance_failure(tmp_path, dtype):
    _actual_alias_ir(_old_issue(tmp_path), dtype, True, 0, 2, False, old=True)
