"""The collective MMA lowering issues loop-invariant read-only epilogue loads early.

CPU-only (fake CUDA target, the ``test_cute_tcgen05_fixed_chain`` fixture): the
jagged dense bmm example's per-column bias is read by the scalar epilogue that
follows the collective pipeline; the lowering moves that load ahead of the
loop it is invariant to so its DRAM latency overlaps the operand loads instead
of following the MMAs in series.
"""

from __future__ import annotations

import ast

import torch

from test.test_cute_tcgen05_fixed_chain import CPU_DEVICE
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401

import helion
from helion._compiler.cute.collective_matmul import _hoist_invariant_epilogue_loads
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _collective_config() -> helion.Config:
    return helion.Config(
        block_sizes=[1, 32, 32, 16],
        num_threads=[0, 4, 32, 1],
        cute_collective_mma=True,
        cute_collective_static_layouts=True,
        cute_collective_compute="warp",
        cute_collective_stages=4,
        cute_collective_copy="async_cached",
        cute_collective_recipe="scalar",
        cute_collective_epilogue="scalar",
        cute_lane_layouts=["blocked", "blocked", "strided", "blocked"],
    )


def test_jagged_dense_bmm_bias_load_precedes_the_pipeline() -> None:
    import examples.jagged_dense_bmm as example

    torch.manual_seed(0)
    batch, features, length = 16, 32, 319
    seq_offsets = torch.tensor(
        [0, *sorted(torch.randint(0, length, (batch - 1,)).tolist()), length],
        dtype=torch.int64,
        device=CPU_DEVICE,
    )
    jagged = torch.randn(length, features, device=CPU_DEVICE)
    dense = torch.randn(batch, features, features, device=CPU_DEVICE)
    bias = torch.randn(batch, features, device=CPU_DEVICE)
    bound = helion.kernel(example.jagged_dense_bmm.fn, backend="cute").bind(  # pyrefly: ignore[bad-argument-type]
        (seq_offsets, jagged, dense, bias)
    )
    with bound.env:
        config = bound.config_spec.normalized_config(_collective_config())
    source = bound.to_code(config)
    body = source[source.index("@cute.kernel") :]
    bias_load = body.index("bias.iterator")
    # Before the operand pipeline's shared allocation and before the jagged
    # tile loop the load used to sit in; the store it feeds stays in place.
    assert bias_load < body.index("alloc_smem")
    assert bias_load < body.index("for tile_offset_1")
    assert body.index("output.iterator") > body.index("for tile_offset_1")
    assert body.count("bias.iterator") == 1


def test_hoist_moves_only_unconditional_invariant_loads_of_unstored_tensors() -> None:
    source = """
for i in range(4):
    acc = out.iterator.load()
    a = (bias.iterator + cutlass.Int32(col) * cutlass.Int32(bias.layout.stride[0])).load()
    b = (bias.iterator + cutlass.Int32(i) * cutlass.Int32(bias.layout.stride[0])).load()
    if mask:
        c = (gate.iterator + cutlass.Int32(col)).load()
    for lane in range(8):
        d = (scale.iterator + cutlass.Int32(col)).load()
        (out.iterator + cutlass.Int32(lane)).store(acc + a + b + d)
"""
    body = ast.parse(source).body
    _hoist_invariant_epilogue_loads(body)  # pyrefly: ignore[bad-argument-type]
    rendered = ast.unparse(ast.Module(body=body, type_ignores=[]))
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    # ``a`` (invariant, bias never stored) and ``d`` (invariant through two
    # loops) move ahead of the outer loop in program order; ``acc`` reads the
    # stored tensor, ``b`` reads the loop variable and ``c`` is conditional.
    assert lines[0].startswith("a = ")
    assert lines[1].startswith("d = ")
    assert lines[2].startswith("for i in range(4):")
    assert any(line.startswith("acc = ") for line in lines[3:])
    assert any(line.startswith("b = ") for line in lines[3:])
    assert any(line.startswith("c = ") for line in lines[3:])


def test_hoist_counts_every_write_form_and_keeps_bound_dependent_loads() -> None:
    # ``counts`` is written by a relaxed atomic add, ``staged`` by a TMA
    # copy through its tensor view: loads of either stay under the loop.
    # ``x`` is read at ``seq_start``, which the row loop's bound depends on
    # (directly, and one arithmetic assignment away through ``rows``): with
    # no trips the address is not known to be valid, so those loads stay
    # too. Only ``bias`` moves.
    source = """
rows = seq_end - seq_start
for r in cutlass.range(rows):
    a = (bias.iterator + cutlass.Int32(col)).load()
    e = (x.iterator + cutlass.Int32(seq_start) * cutlass.Int32(x.layout.stride[0])).load()
    f = (counts.iterator + cutlass.Int32(col)).load()
    g = (staged.iterator + cutlass.Int32(col)).load()
    cute.arch.atomic_add((counts.iterator + cutlass.Int32(col)).llvm_ptr, val=cutlass.Float32(1), sem='relaxed')
    cute.copy(atom, source_view, tma_tensor_staged)
    (out.iterator + cutlass.Int32(r)).store(a + e + f + g)
for r in cutlass.range(seq_start, seq_end):
    h = (x.iterator + cutlass.Int32(seq_start)).load()
    (out.iterator + cutlass.Int32(r)).store(h)
"""
    body = ast.parse(source).body
    _hoist_invariant_epilogue_loads(body)  # pyrefly: ignore[bad-argument-type]
    rendered = ast.unparse(ast.Module(body=body, type_ignores=[]))
    lines = [line.strip() for line in rendered.splitlines() if line.strip()]
    assert lines[0].startswith("rows = ")
    assert lines[1].startswith("a = ")
    assert lines[2].startswith("for r in cutlass.range(rows):")
    first_loop = lines.index("for r in cutlass.range(rows):")
    second_loop = lines.index("for r in cutlass.range(seq_start, seq_end):")
    for name in ("e", "f", "g"):
        assert any(
            line.startswith(f"{name} = ") for line in lines[first_loop:second_loop]
        ), name
    assert any(line.startswith("h = ") for line in lines[second_loop:])
