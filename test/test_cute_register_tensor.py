from __future__ import annotations

import ast
import importlib.util
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_register_program import _run as run_scalar_program

import helion
from helion._compiler.cute.register_tensor import _build_plan as compile_plan
from helion._compiler.cute.register_tensor import _serialize_plan as serialize_plan
from helion._compiler.cute.register_tensor import (
    _UnsupportedRegisterTensor as UnsupportedRegisterPlan,
)
from helion._compiler.cute.register_tensor import emit_register_tensor
from helion._compiler.cute.register_tensor import plan_register_tensor
from helion._compiler.cute.row_fragment import RowFragment
from helion._compiler.cute.row_fragment import RowFragmentLayout
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any


def graph(function, dtype=torch.int64):
    return make_fx(function)(torch.arange(32, dtype=dtype).reshape(4, 8))


def gather(value, offset=1):
    return value.index_select(0, torch.arange(4) ^ offset)


def communicating(plan):
    return [node for node in plan["nodes"] if "live_registers" in node]


def test_prefix_communication_is_pruned():
    plan = compile_plan(graph(lambda value: gather(value)[:, :2]), groups=4)
    assert communicating(plan)[0]["live_registers"] == [0, 1]


def test_local_permutation_scatter_expands_needed_sources():

    def program(value):
        peer = gather(value)
        positions = (torch.arange(8)[None, :] + torch.arange(4)[:, None]) % 8
        return peer.gather(1, positions)[:, :2]

    plan = compile_plan(graph(program), groups=4)
    assert communicating(plan)[0]["live_registers"] == [0, 1, 2, 3, 4]


def test_static_where_prunes_dead_communication():
    plan = compile_plan(
        graph(
            lambda value: torch.where(torch.arange(8) < 1, gather(value), value)[:, :2]
        ),
        groups=4,
    )
    assert communicating(plan)[0]["live_registers"] == [0]


def test_cat_propagates_disjoint_prefixes():

    def program(value):
        peer = gather(value)
        return torch.cat((peer[:, :2], peer[:, 6:]), dim=1)

    plan = compile_plan(graph(program), groups=4)
    assert communicating(plan)[0]["live_registers"] == [0, 1, 6, 7]


@pytest.mark.parametrize("constant_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_cat_with_constant_uses_scalar_fallback(constant_first, dtype):
    def program(value):
        constant = torch.ones((4, 3), dtype=dtype)
        sources = (constant, value) if constant_first else (value, constant)
        return torch.cat(sources, dim=1)

    value = torch.arange(32, dtype=dtype).reshape(4, 8)
    module = make_fx(program)(value)
    fragment = RowFragment("x", dtype, 32, RowFragmentLayout(4, 8, "lane"))
    assert plan_register_tensor(module, [fragment], lanes=4) is None
    outputs, _source = run_scalar_program(module, value)
    torch.testing.assert_close(outputs[0], program(value), rtol=0, atol=0)


def test_duplicate_gathers_share_one_plan_value():
    plan = compile_plan(
        graph(lambda value: torch.maximum(gather(value), gather(value))), groups=4
    )
    assert len(communicating(plan)) == 1


@pytest.mark.parametrize(
    "program",
    [
        lambda value: gather(gather(value, 1), 2),
        lambda value: torch.where(
            torch.arange(4)[:, None] < 2, gather(value, 1), gather(value, 2)
        ),
    ],
)
def test_alias_composition_declines(program):
    with pytest.raises(UnsupportedRegisterPlan, match="composed communication"):
        compile_plan(graph(program), groups=4)


@pytest.mark.parametrize(
    "program",
    [
        lambda value: value.add_(1),
        lambda value: torch.add(value, value, alpha=2),
        lambda value: value.view(torch.float64),
        lambda value: value.sum(dim=1, keepdim=True),
    ],
)
def test_unsupported_effects_and_operations_decline(program):
    with pytest.raises(UnsupportedRegisterPlan):
        compile_plan(graph(program), groups=4)


def test_random_constant_is_not_evaluated():
    module = graph(lambda value: value + torch.rand(value.shape), torch.float32)
    state = torch.random.get_rng_state().clone()
    with pytest.raises(UnsupportedRegisterPlan):
        compile_plan(module, groups=4)
    assert torch.equal(state, torch.random.get_rng_state())


def test_boolean_minimum_declines_signed_integer_ordering():
    module = make_fx(lambda value: torch.minimum(value, ~value))(
        torch.zeros((4, 8), dtype=torch.bool)
    )
    with pytest.raises(UnsupportedRegisterPlan):
        compile_plan(module, groups=4)


def test_owner_lane_permutations_decline():
    module = graph(lambda value: gather(value))
    fragment = RowFragment(
        "x", torch.int64, 32, RowFragmentLayout(4, 8, "lane", owner_lanes=(1, 0, 3, 2))
    )
    with pytest.raises(UnsupportedRegisterPlan, match="owner_lanes"):
        compile_plan(module, groups=4, input_fragments=[fragment])


def test_fake_context_has_identical_constant_plan():
    module = graph(lambda value: gather(value)[:, :2])
    expected = serialize_plan(compile_plan(module, groups=4))
    with FakeTensorMode():
        actual = serialize_plan(compile_plan(module, groups=4))
    assert actual == expected


def test_nested_constant_attributes_preserve_float_bits():
    owner = torch.nn.Module()
    owner.add_module("constants", torch.nn.Module())
    data = torch.tensor([[0.0, -0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 5.0]] * 4)
    owner.constants.register_buffer("data", data)
    ir = Graph()
    value = ir.placeholder("value")
    value.meta["val"] = torch.empty_like(data)
    constant = ir.get_attr("constants.data")
    constant.meta["val"] = data
    result = ir.call_function(torch.ops.aten.maximum.default, (value, constant))
    result.meta["val"] = torch.empty_like(data)
    ir.output(result)
    plan = compile_plan(GraphModule(owner, ir), groups=4)
    node = next(node for node in plan["nodes"] if node["op"] == "constant")
    assert node["rows"] == [data[0].numpy().tobytes().hex()]


def test_emission_names_and_plan_globals_are_isolated():
    statements, modules, names = [], [], []

    def new_var(prefix):
        name = prefix + str(len(names))
        names.append(name)
        return name

    cg = SimpleNamespace(
        module_statements=modules,
        add_statement=statements.append,
        device_function=SimpleNamespace(new_var=new_var),
    )
    fragment = RowFragment("x", torch.int64, 32, RowFragmentLayout(4, 8, "lane"))
    module = graph(lambda value: gather(value)[:, :2])
    plan = plan_register_tensor(module, [fragment], lanes=4)
    assert plan is not None
    first = emit_register_tensor(cg, plan, [fragment], lane_expr="lane")
    second = emit_register_tensor(cg, plan, [fragment], lane_expr="lane")
    assert first[0].name != second[0].name
    assert len(modules) == 2
    snapshot = (
        [ast.dump(item) for item in modules],
        [ast.dump(item) for item in statements],
        list(names),
    )
    unsupported = graph(lambda value: gather(gather(value)))
    assert plan_register_tensor(unsupported, [fragment], lanes=4) is None
    assert snapshot == (
        [ast.dump(item) for item in modules],
        [ast.dump(item) for item in statements],
        list(names),
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sort_rows(value: torch.Tensor):
    output = torch.empty_like(value)
    indices = torch.empty(value.shape, dtype=torch.int64, device=value.device)
    for row in hl.tile(value.size(0)):
        selected, positions = torch.sort(value[row, :], descending=True)
        output[row, :] = selected
        indices[row, :] = positions
    return output, indices


@pytest.mark.parametrize("width", [512, 8192])
def test_wide_sort_has_compact_caller_ast(width):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_sort_rows, (torch.empty((3, width), dtype=torch.float32),))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_topk_lanes_per_row=16,
            cute_topk_selection_layout="replicated",
            cute_topk_sort_network="batcher",
            cute_topk_key_dtype="int64",
            cute_topk_rows_per_block=8,
            cute_topk_vector_width=8,
            cute_topk_key_encoder="dsl",
            cute_topk_merge_schedule="sequential",
        )
        source = bound.to_code(config)
    kernel = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert "_cute_execute_register_plan" in source
    assert sum(1 for _ in ast.walk(kernel)) < 1200


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_sdk_keeps_only_live_communication(dtype, tmp_path):
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    from cutlass._mlir.passmanager import PassManager
    import cutlass.cute as cute

    from helion.runtime.cute.register_tensor import _cute_execute_register_plan

    module = graph(lambda value: gather(value)[:, :2], dtype)
    plan = compile_plan(module, groups=4)
    ctype = {
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.float32: cutlass.Float32,
    }[dtype]
    initialized = torch.cuda.is_initialized()
    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with ir.InsertionPoint(emitted.body):
            function = func.FuncOp(
                "entry",
                (
                    [ir.VectorType.get([8], ctype.mlir_type)],
                    [ir.VectorType.get([2], ctype.mlir_type)],
                ),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                values = cute.TensorSSA(block.arguments[0], (8,), ctype)
                buffer = cute.make_rmem_tensor((8,), ctype)
                buffer.store(values)
                (output,) = _cute_execute_register_plan(
                    serialize_plan(plan), (buffer,), cutlass.Int32(0)
                )
                func.ReturnOp([output.load().ir_value()])
        PassManager.parse("builtin.module(canonicalize,cse)").run(emitted.operation)
        assert emitted.operation.verify()
        source = str(emitted)
        (tmp_path / "register_tensor.mlir").write_text(source)
        assert source.count("nvvm.shfl.sync") == (4 if dtype == torch.int64 else 2)
    assert torch.cuda.is_initialized() == initialized


def test_sdk_float32_extrema_keep_nan_semantics_and_eliminate_dead_elements(tmp_path):
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir import ir
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import gpu
    import cutlass.cute as cute

    def program(value):
        peer = gather(value)
        return torch.minimum(value, peer)[:, :2], torch.maximum(value, peer)[:, :2]

    plan = compile_plan(graph(program, torch.float32), groups=4)
    path = tmp_path / "float32_extrema_sdk.py"
    path.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from helion.runtime.cute.register_tensor import _cute_execute_register_plan\n"
        f"PLAN = {serialize_plan(plan)!r}\n"
        "@cute.kernel\n"
        "def kernel(source, minimum, maximum):\n"
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])\n"
        "    values = cute.make_tensor(source.iterator + thread * 8, "
        "cute.make_layout((8,))).load()\n"
        "    resident = cute.make_rmem_tensor((8,), cutlass.Float32)\n"
        "    resident.store(values)\n"
        "    low, high = _cute_execute_register_plan(PLAN, (resident,), thread)\n"
        "    cute.make_tensor(minimum.iterator + thread * 2, "
        "cute.make_layout((2,))).store(low.load())\n"
        "    cute.make_tensor(maximum.iterator + thread * 2, "
        "cute.make_layout((2,))).store(high.load())\n"
        "@cute.jit\n"
        "def entry(source, minimum, maximum):\n"
        "    kernel(source, minimum, maximum).launch(grid=(1,), block=(32,))\n"
    )
    spec = importlib.util.spec_from_file_location("float32_extrema_sdk", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensors = [
        cute.runtime.make_fake_tensor(
            cutlass.Float32, (32, width), (width, 1), assumed_align=16
        )
        for width in (8, 2, 2)
    ]
    initialized = torch.cuda.is_initialized()
    staged = cute.compile.to_precompiled_mlir(
        module.entry, *tensors, options="--gpu-arch=sm_100a"
    )
    mlir = cast("Any", ir)
    passes = cast("Any", passmanager)
    gpu_ir = cast("Any", gpu)
    with mlir.Context() as context, mlir.Location.unknown():
        context.enable_multithreading(False)
        original = mlir.Module.parse(staged.get_bitcode())
        assert original.operation.verify()
        before = str(original)
        (tmp_path / "float32_extrema.mlir").write_text(before)
        for operation in ("fmin", "fmax"):
            instructions = [
                line for line in before.splitlines() if f"nvvm.{operation} " in line
            ]
            assert len(instructions) == 8
            assert all(" nan " in line and "ftz" not in line for line in instructions)
        device = next(op for op in original.body.operations if op.name == "gpu.module")
        lowered = mlir.Module.create()
        with mlir.InsertionPoint(lowered.body):
            device.operation.clone()
        # Compile to PTX on the CPU. No CUDA context, cubin or kernel launch is
        # needed. Vector element DCE occurs in this downstream SDK pipeline.
        passes.PassManager.parse(
            "builtin.module(cute-to-nvvm{cubin-format=isa cubin-chip=sm_100a})"
        ).run(lowered.operation)
        ptx = (
            gpu_ir.ObjectAttr(lowered.body.operations[0].attributes["objects"][0])
            .object.decode()
            .rstrip("\0")
        )
        (tmp_path / "float32_extrema.ptx").write_text(ptx)
        assert len(re.findall(r"\bmin\.NaN\.f32\b", ptx)) == 2
        assert len(re.findall(r"\bmax\.NaN\.f32\b", ptx)) == 2
        assert len(re.findall(r"\bshfl\.sync\.", ptx)) == 2
        assert ".ftz." not in ptx
    assert torch.cuda.is_initialized() == initialized
