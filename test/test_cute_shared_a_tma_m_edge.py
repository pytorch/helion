"""CPU-only exact Inductor and source/resource tests for shared-A TMA M edges."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag_tcgen05 as tcgen
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration

INDUCTOR_NATIVE = """import torch
import helion
import helion.language as hl

from torch._inductor.runtime.helion import HelionKernelWrapper
@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)
def helion_op1_op3_op2_op4(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
        tmp6 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp6.to(torch.float32)
        tmp7 = acc1.to(torch.float32)
        out1[tile_y, tile_x] = tmp7.to(torch.float32)
helion_op1_op3_op2_op4 = HelionKernelWrapper(helion_op1_op3_op2_op4)

def call_helion_op1_op3_op2_op4(buf0: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf2: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op1_op3_op2_op4(buf0, arg1_1, arg2_1, buf2, buf4)
"""


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def args_for(
    p=2, m=1000, n=768, k=128, dtype=torch.bfloat16, output_dtype=torch.float32
):
    return tuple(
        [torch.empty_strided((m, k), (1, m), dtype=dtype)]
        + [torch.empty((k, n), dtype=dtype) for _ in range(p)]
        + [torch.empty((m, n), dtype=output_dtype) for _ in range(p)]
    )


def synthetic_source(p=2, *, extra=""):
    """A new test fixture, not a production seeder or a rewritten original."""
    params = ["a"] + [f"b{i}" for i in range(p)] + [f"d{i}" for i in range(p)]
    lines = [
        "import torch",
        "import helion",
        "import helion.language as hl",
        '@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full", ref_mode=helion.RefMode.OFF)',
        f"def shared_A({', '.join(params)}):",
        "    for m,n in hl.tile([a.size(0),b0.size(1)]):",
        *[f"        acc{i}=hl.zeros([m,n],dtype=torch.float32)" for i in range(p)],
        "        for k in hl.tile(a.size(1)):",
        "            v=a[m,k].to(torch.float32).to(a.dtype)",
    ]
    for i in range(p):
        lines += [
            f"            w{i}=b{i}[k,n].to(torch.float32).to(a.dtype)",
            f"            acc{i}=torch.addmm(acc{i},v,w{i})",
        ]
    if extra:
        lines.append(extra)
    lines += [f"        d{i}[m,n]=acc{i}.to(d{i}.dtype)" for i in range(p)]
    return "\n".join(lines) + "\n"


def load_kernel(source):
    from torch._inductor.codecache import PyCodeCache

    module = PyCodeCache.load(source)
    original = getattr(module, "shared_A", None)
    if original is None:
        original = module.helion_op1_op3_op2_op4.kernel
    assert isinstance(original, helion.Kernel)
    return helion.kernel(
        original.fn,
        settings=original.settings,
        cute_structural_policy=original.cute_structural_policy,
    )


def bind(kernel, args):
    with patch.object(kernel, "_bind_isolated", side_effect=kernel.bind):
        return cpu_bind(kernel, args)


def pairs(bound):
    with bound.env, bound.host_function:
        bound.config_spec.validate_compiler_coverage_groups()
        gen = ConfigGeneration(bound.config_spec)
        gen.initial_population_view()
        return gen.seed_flat_config_pairs()


def capture_code(bound, config):
    captured = []
    original = tcgen.plan_tcgen05

    def capture(*args, **kwargs):
        plan = original(*args, **kwargs)
        captured.append(plan)
        return plan

    with patch.object(tcgen, "plan_tcgen05", side_effect=capture):
        source = bound.to_code(config)
    assert len(captured) == 1
    return source, captured[0]


def native_calls(source, suffix):
    tree = ast.parse(source) if isinstance(source, str) else source
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(suffix)
    ]


def test_actual_R2_Inductor_module_defaults_and_initial_view_native_source():
    assert hashlib.sha256(INDUCTOR_NATIVE.encode()).hexdigest() == (
        "5e3f321431f7fdcd027a91ad2a86ac1e512da3c4749014567d392a6d22d52e8e"
    )
    kernel = load_kernel(INDUCTOR_NATIVE)
    args = args_for()
    bound = bind(kernel, args)
    assert kernel.settings.ref_mode is helion.RefMode.OFF
    assert kernel.settings.fast_math is False
    assert kernel.settings.disable_autotuner_heuristics is False
    assert kernel.cute_structural_policy is not None
    policy_settings = kernel.cute_structural_policy.to_dict()["settings"]
    assert isinstance(policy_settings, dict)
    assert len(policy_settings) == 5 and all(policy_settings.values())
    seeds = pairs(bound)
    assert len(seeds) == 4
    assert not native_calls(
        bound.to_code(bound.config_spec.default_config()), "tcgen05_mma"
    )
    fields = bound.config_spec._flat_fields()
    with bound.env, bound.host_function:
        bound.config_spec.validate_compiler_coverage_groups()
    assert (
        tuple((key, *value.fingerprint()) for key, value in fields.items())
        == (bound.config_spec._compiler_coverage_fields)
        or not bound.config_spec.compiler_coverage_groups
    )
    for _, config in seeds:
        assert config.get("cute_mma_dag") == "tcgen05"
        assert config.get("cute_mma_dag_persistent", 0) == 0
        source, plan = capture_code(bound, config)
        assert plan.shared_a_m_edge_axis is not None
        assert plan.shared_a_m_edge_axis.extent == 1000
        assert len(plan.loop_tiles) == 3
        names = [
            t.operand.load.name for t in plan.loop_tiles if t.operand.load is not None
        ]
        assert names == ["A", "B", "B1"]
        assert all(t.route == "direct" and t.tma is not None for t in plan.loop_tiles)
        assert plan.tx_bytes == sum(s.bytes for s in plan.tma)
        assert len(plan.tma) == 3
        assert plan.tma[0].dims == (0, 1)
        assert tuple(plan.tma[0].access.fake.shape) == (1000, 128)
        assert plan.tma[0].access.strides == (1, 1000)
        assert plan.tmem_cols <= 512
        assert native_calls(source, "tcgen05_mma")
        assert "cp_async_bulk_tensor_shared_cta_global" in source
        assert "cutlass.Int32(1000)" in source
        assert not native_calls(source, "cp_async_bulk_tensor_global_shared_cta")
        assert [s.access.name for s in plan.staged_stores] == ["out0", "out1"]


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_general_shared_A_P2_3_4_half_types_real_TMA_predicated_stores(p, dtype):
    bound = bind(
        load_kernel(synthetic_source(p)), args_for(p, m=264, n=128, dtype=dtype)
    )
    seeds = pairs(bound)
    assert seeds
    source, plan = capture_code(bound, seeds[0][1])
    assert len(plan.loop_tiles) == p + 1
    assert plan.shared_a_m_edge_axis is not None
    assert plan.shared_a_m_edge_axis.extent == 264
    assert len(plan.slots) == p
    assert len({slot.column for slot in plan.slots}) == p
    assert all(
        left.column + tcgen._align(left.cols, 32) <= right.column
        for left, right in zip(plan.slots, plan.slots[1:], strict=False)
    )
    assert plan.tmem_cols <= 512
    assert native_calls(source, "tcgen05_mma")
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for call in native_calls(tree, "stg_v4_b32"):
        ancestor = parents[call]
        guards = []
        while ancestor in parents:
            if isinstance(ancestor, ast.If):
                guards.append(ast.unparse(ancestor.test))
            ancestor = parents[ancestor]
        assert any("cutlass.Int32(264)" in guard for guard in guards)
    assert source.rfind("stg_v4_b32") < source.rfind("tcgen05_dealloc")


def test_complete_JSON_prebind_and_cached_rebind_keep_native_program():
    args = args_for()
    kernel = load_kernel(INDUCTOR_NATIVE)
    bound = bind(kernel, args)
    config = pairs(bound)[0][1]
    decoded = helion.Config.from_dict(json.loads(config.to_json()))
    original = load_kernel(INDUCTOR_NATIVE)
    fresh = helion.kernel(
        original.fn,
        config=decoded,
        settings=original.settings,
        cute_structural_policy=original.cute_structural_policy,
    )
    rebound = bind(fresh, args)
    assert bind(kernel, args) is bound
    assert ast.dump(ast.parse(rebound.to_code(decoded)), include_attributes=False) == (
        ast.dump(ast.parse(bound.to_code(config)), include_attributes=False)
    )
    pairs(rebound)


def test_wrapper_reaches_autotune_initial_view_without_native_compile():
    from torch._inductor.runtime.helion import HelionKernelWrapper

    from helion.runtime.kernel import BoundKernel

    kernel = load_kernel(INDUCTOR_NATIVE)
    args = args_for()
    bound = bind(kernel, args)
    wrapper = HelionKernelWrapper(kernel)
    observed = []

    class ReachedSearch(Exception):
        pass

    def stop_at_search(self, *args, **kwargs):
        with self.env, self.host_function:
            self.config_spec.validate_compiler_coverage_groups()
            gen = ConfigGeneration(self.config_spec)
            gen.initial_population_view()
            observed.append(len(gen.seed_flat_config_pairs()))
        raise ReachedSearch

    with (
        patch.object(BoundKernel, "autotune", stop_at_search),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.raises(ReachedSearch),
    ):
        wrapper(*args)
    assert observed == [4]
    assert bound.kernel is kernel


def test_actual_descriptor_factory_full_global_dims_pitch_and_zero_fill_IR():
    """Unwrap only the JIT decorator in a test; execute its exact body with
    real CuTe/SDK MLIR objects. No TensorMap driver call or kernel executes.
    """
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    runtime = importlib.import_module("helion._compiler.cute._mma_dag_tcgen05_runtime")
    assert runtime.__file__ is not None
    module = ast.parse(Path(runtime.__file__).read_text())
    fn = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "make_tensor_map"
    )
    fn.decorator_list = []
    namespace = dict(runtime.__dict__)
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), runtime.__file__, "exec"),
        namespace,
    )
    calls = []
    tensor_map_module = importlib.import_module("cutlass.experimental.cuda.tensor_map")
    original = tensor_map_module.create_tensor_map_tiled

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with (
            ir.InsertionPoint(emitted.body),
            patch.object(tensor_map_module, "create_tensor_map_tiled", capture),
        ):
            tensor = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((1000, 128), stride=(1, 1000)),
            )
            namespace["make_tensor_map"](tensor, (0, 1), (64, 128), "s128b")
        assert len(calls) == 1
        assert calls[0]["global_dims"] == [1000, 128]
        assert calls[0]["global_strides"] == [125]
        assert calls[0]["box_dims"] == [64, 128]
        assert calls[0]["oob_fill"] is None
        encoders = [
            op for op in emitted.body.operations if "tensor_map_encode_tiled" in op.name
        ]
        assert len(encoders) == 1
        assert str(encoders[0].operands[-1].owner.attributes["value"]) == "0 : i32"
        assert "1000" in str(emitted)
        assert "1024" not in str(emitted)


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_unary_output_casts_are_kept_not_added_to_FP32(output_dtype):
    bound = bind(
        load_kernel(synthetic_source()),
        args_for(m=264, n=128, output_dtype=output_dtype),
    )
    source, plan = capture_code(bound, pairs(bound)[0][1])
    assert [s.access.dtype for s in plan.staged_stores] == [output_dtype, output_dtype]
    if output_dtype is torch.float32:
        assert not native_calls(source, "pack_bf16x2")
        assert not native_calls(source, "pack_f16x2")
    else:
        assert native_calls(
            source, "pack_f16x2" if output_dtype is torch.float16 else "pack_bf16x2"
        )
    assert "cutlass.Int32(264)" in source


def test_ordered_multiple_K_iterations_and_ring_refill_follow_all_UMMA_consumers():
    bound = bind(load_kernel(synthetic_source()), args_for(m=264, n=128, k=512))
    source, plan = capture_code(bound, pairs(bound)[0][1])
    assert plan.stages > 1
    assert plan.tx_bytes == sum(s.bytes for s in plan.tma)
    assert len(plan.loop_tiles) == 3
    # The existing simple-loop batch emits all ordered sibling MMA commands,
    # then one commit before the refill. The source, not a config flag, proves
    # which path was emitted; hardware synchronization remains a later gate.
    tree = ast.parse(source)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and "md_i" in ast.unparse(n.target)
    ]
    assert loops
    sequence = []

    class OrderedCalls(ast.NodeVisitor):
        def visit_Call(self, node):
            sequence.append(ast.unparse(node))
            self.generic_visit(node)

    OrderedCalls().visit(loops[0])
    mma_positions = [i for i, s in enumerate(sequence) if "tcgen05_mma" in s]
    commit_positions = [i for i, s in enumerate(sequence) if "tcgen05_commit" in s]
    assert mma_positions and commit_positions
    assert max(mma_positions) < min(commit_positions)
    assert "mbar_wait(md_accb" in source


def test_payload_nonfinite_and_signed_zero_does_not_change_admission_or_FP32_source():
    kernel = load_kernel(synthetic_source())
    zero = args_for(m=264, n=128)
    special = args_for(m=264, n=128)
    for tensor in zero[:3]:
        tensor.zero_()
    values = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=torch.bfloat16
    )
    for tensor in special[:3]:
        tensor.copy_(
            values.repeat((tensor.numel() + 4) // 5)[: tensor.numel()].reshape(
                tensor.shape
            )
        )
    plain = bind(kernel, zero)
    changed = bind(kernel, special)
    a = pairs(plain)
    b = pairs(changed)
    assert [dict(c) for _, c in a] == [dict(c) for _, c in b]
    assert ast.dump(
        ast.parse(plain.to_code(a[0][1])), include_attributes=False
    ) == ast.dump(ast.parse(changed.to_code(b[0][1])), include_attributes=False)
    # This verifies content-independent source/admission, NOT GPU arithmetic
    # on NaN/Inf/signed-zero values and never relaxes that future execution gate.
