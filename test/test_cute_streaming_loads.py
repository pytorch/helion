"""Tests for the CuTe ``"streaming"`` load eviction policy.

``"streaming"`` lowers to the ``ld.global.cs`` cache operator (``cop='cs'``
on ``cute.arch.load``): evict-first at both L1 and L2, so single-use
streaming reads stop displacing useful L2 lines.  Scalar sites route
through ``cute.arch.load`` until final emission (``(ptr).load()`` has no hint kwargs), and the
cross-sweep load fuser must keep matching the hinted scalar form against
its unhinted twin in the consume sweep.

Lives in ``helion/_compiler/cute/memory_ops.py`` (emission),
``helion/language/memory_ops.py`` (scalar form), and
``helion/_compiler/cute/fuse_two_pass_loads.py`` (matching).
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(
    backend="cute",
    config={
        "block_sizes": [1],
        "reduction_loops": [1024],
        "load_eviction_policies": ["streaming", "streaming", "streaming"],
    },
)
def _logsumexp_scalar_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_x[:, None]), dim=-1)
        out[tile_m] = max_x + torch.log(sum_exp)
    return out


@helion.kernel(
    backend="cute",
    config={
        "block_sizes": [1],
        "reduction_loops": [512],
        "num_threads": [0, 64],
        "cute_vector_widths": [8, 1],
        "load_eviction_policies": ["streaming", "streaming", "streaming"],
    },
)
def _logsumexp_vec_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_x[:, None]), dim=-1)
        out[tile_m] = max_x + torch.log(sum_exp)
    return out


@onlyBackends(["cute"])
class TestCuteStreamingLoads(TestCase):
    def test_streaming_in_choices(self) -> None:
        from helion.autotuner.config_spec import get_valid_eviction_policies

        self.assertIn("streaming", get_valid_eviction_policies("cute"))
        self.assertNotIn("streaming", get_valid_eviction_policies("triton"))

    def test_scalar_streaming_load_keeps_fusion(self) -> None:
        x = torch.randn(64, 2048, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(_logsumexp_scalar_kernel, (x,))
        # Scalar sites retain the hint through the final policy helper...
        self.assertIn("_cute_scalar_policy_load(", code)
        self.assertIn("'streaming'", code)
        # ...and the cross-sweep register cache must still fire (the hinted
        # reduce-sweep load matches the unhinted consume-sweep load).
        self.assertIn("_fuse_cache_0", code)
        torch.testing.assert_close(
            out, torch.logsumexp(x, dim=-1), rtol=1e-3, atol=1e-3
        )

    def test_vec_streaming_load(self) -> None:
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(_logsumexp_vec_kernel, (x,))
        self.assertIn("cop='cs'", code)
        torch.testing.assert_close(
            out, torch.logsumexp(x.float(), dim=-1), rtol=1e-3, atol=1e-3
        )


@pytest.mark.parametrize(
    "dtype", ("Float32", "Float64", "Int32", "Int64", "Uint32", "Uint64")
)
@pytest.mark.parametrize(
    "keyword, value, policy",
    (
        ("cop", "cs", "streaming"),
        ("level1_eviction_priority", "evict_first", "first"),
        ("level1_eviction_priority", "evict_last", "last"),
    ),
)
def test_scalar_policy_final_lowering(dtype, keyword, value, policy):
    from helion._compiler.cute.scalar_policy_loads import lower_scalar_policy_loads

    body = ast.parse(
        f"x = cute.arch.load(p, cutlass.{dtype}, {keyword}={value!r})"
    ).body
    result = lower_scalar_policy_loads(body)
    assert (
        ast.unparse(result[0])
        == f"x = _cute_scalar_policy_load(p, cutlass.{dtype}, {policy!r})"
    )


@pytest.mark.parametrize(
    "expr",
    (
        "p.load()",
        "cute.arch.load(p, cutlass.Float32)",
        "cute.arch.load(p, cutlass.Float16, cop='cs')",
        "cute.arch.load(p, cutlass.BFloat16, cop='cs')",
        "cute.arch.load(p, cutlass.Uint8, cop='cs')",
        "cute.arch.load(p, cutlass.Boolean, cop='cs')",
        "cute.arch.load(p, ir.VectorType.get([4], cutlass.Float32.mlir_type), cop='cs')",
        "cute.arch.load(p, cutlass.Float32, cop='cg')",
        "cute.arch.load(p, cutlass.Float32, cop=policy)",
        "cute.arch.load(p, cutlass.Float32, cop='cs', extra=True)",
        "other.arch.load(p, cutlass.Float32, cop='cs')",
        "cute.arch.load(p, other.Float32, cop='cs')",
    ),
)
def test_scalar_policy_preserves_other_load_forms(expr):
    from helion._compiler.cute.scalar_policy_loads import lower_scalar_policy_loads

    body = ast.parse(f"x = {expr}").body
    before = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert (
        ast.dump(ast.Module(body=lower_scalar_policy_loads(body), type_ignores=[]))
        == before
    )


def test_scalar_policy_helper_bits_and_memory_effects():
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func

    from helion._compiler.cute.l2_policy import scalar_policy_load

    class Address:
        def __init__(self, value):
            self.value = value

        def toint(self, **kwargs):
            return self

        def ir_value(self, **kwargs):
            return self.value

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for dtype in (
                cutlass.Float32,
                cutlass.Float64,
                cutlass.Int32,
                cutlass.Int64,
                cutlass.Uint32,
                cutlass.Uint64,
            ):
                for policy in ("first", "last", "streaming"):
                    fn = func.FuncOp(
                        f"load_{dtype.__name__}_{policy}",
                        ir.FunctionType.get(
                            [cutlass.Uint64.mlir_type], [dtype.mlir_type]
                        ),
                    )
                    block = fn.add_entry_block()
                    with ir.InsertionPoint(block):
                        value = scalar_policy_load(
                            Address(block.arguments[0]), dtype, policy
                        )
                        assert type(value) is dtype
                        func.ReturnOp([value.ir_value()])
        module.operation.verify()
        text = str(module)
        assert text.count("llvm.inline_asm has_side_effects") == 18
        assert text.count("~{memory}") == 18
        assert text.count("llvm.bitcast") == 6
        assert "sitofp" not in text and "uitofp" not in text and "fptrunc" not in text
        for suffix in (".L1::evict_first", ".L1::evict_last", ".cs"):
            for width in (32, 64):
                assert text.count(f"ld.global{suffix}.b{width}") == 3


@pytest.mark.parametrize("length", (0, 1, 3, 4, 7))
@pytest.mark.parametrize("width", (32, 64))
def test_scalar_policy_mask_tail_alias_and_bit_model(length, width):
    from helion._compiler.cute.scalar_policy_loads import lower_scalar_policy_loads

    # No arithmetic in the transfer: includes signed zero, NaN payloads,
    # infinities, subnormals and signed/unsigned integer high bits.
    words = [0, 1, 1 << (width - 1), (1 << width) - 1]
    words += (
        [0x7F800000, 0x7FC12345, 0xFF800000]
        if width == 32
        else [0x7FF0000000000000, 0x7FF8123456789ABC, 0xFFF0000000000000]
    )
    code = """
for i in range(8):
    if i < length:
        before = cute.arch.load(pointer + i, cutlass.Uint64, cop='cs')
        (pointer + i).store(before ^ 1)
        after = cute.arch.load(pointer + i, cutlass.Uint64, level1_eviction_priority='evict_last')
        outputs.append((before, after))
"""
    if width == 32:
        code = code.replace("Uint64", "Uint32")

    def run(lower):
        values = words[:length]
        events = []

        class Pointer:
            def __init__(self, index=0):
                self.index = index

            def __add__(self, index):
                return Pointer(self.index + index)

            def load(self):
                assert self.index < length
                events.append(("load", self.index, values[self.index]))
                return values[self.index]

            def store(self, value):
                assert self.index < length
                events.append(("store", self.index, value))
                values[self.index] = value

        def load(pointer, dtype, **kwargs):
            return pointer.load()

        def policy_load(pointer, dtype, policy):
            assert policy in ("streaming", "last")
            return pointer.load()

        outputs = []
        scope = {
            "length": length,
            "pointer": Pointer(),
            "outputs": outputs,
            "cute": SimpleNamespace(arch=SimpleNamespace(load=load)),
            "cutlass": SimpleNamespace(Uint32=int, Uint64=int),
            "_cute_scalar_policy_load": policy_load,
        }
        tree = ast.parse(code)
        if lower:
            tree.body = lower_scalar_policy_loads(tree.body)
        exec(
            compile(ast.fix_missing_locations(tree), "<scalar-load-model>", "exec"),
            scope,
        )
        return values, outputs, events

    original = run(False)
    assert run(True) == original
    assert len(original[2]) == length * 3
    assert all(after == (before ^ 1) for before, after in original[1])


def test_scalar_policy_codegen_retains_fusion():
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable

    with _mock_cuda_unavailable(), _forbid_native_compile():
        bound = _cpu_bind(_logsumexp_scalar_kernel, (torch.zeros(64, 2048),))
        code = bound.to_code(bound.configs[0])
    assert "_cute_scalar_policy_load(" in code
    assert "_fuse_cache_0" in code


if __name__ == "__main__":
    import unittest

    unittest.main()
