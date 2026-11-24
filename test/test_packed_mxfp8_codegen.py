"""Generated-source checks of the native packed MXFP8 lowering
(``helion/_compiler/triton/packed_mxfp8.py``), bound on the CPU for a
modelled sm100 target (``packed_mxfp8_cpu_codegen``)."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler._inductor.template_buffer import render_inductor_ast
from helion._compiler.ast_extension import unparse
from helion._compiler.compile_environment import tensor_input_sources
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.packed_mxfp8 import dynamic_packed_mxfp8_gemm
from helion._testing.packed_mxfp8 import packed_mxfp8_cpu_codegen
from helion._testing.packed_mxfp8 import packed_mxfp8_gemm
from helion._testing.packed_mxfp8 import packed_mxfp8_inputs
from helion._testing.packed_mxfp8 import two_output_packed_mxfp8_gemm
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    # The lowering is the Triton backend's on CUDA builds only: a ROCm or XPU
    # build of torch declines it whatever target the fixture models
    # (``mxfp8.native_packed_mxfp8_target``), as TileIR does.
    pytest.mark.skipif(
        torch.version.hip is not None or torch.xpu.is_available(),
        reason="the native packed MXFP8 lowering needs a CUDA build",
    ),
]

_TILEIR = "the native packed MXFP8 lowering is the Triton backend's, not TileIR's"


@helion.kernel(autotune_effort="none", static_shapes=True)
def _stack_write_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    dev_ptrs: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` that, after loading its operands, zeroes the
    ``x`` tile through a stack tensor's device pointers (which a caller may
    aim at ``x``)."""
    m, k = x.size()
    n, _ = y.size()
    hl.specialize(dev_ptrs.size(0))
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            stack = hl.stacktensor_like(scratch, dev_ptrs[:])
            stack[rows, columns] = torch.zeros_like(lhs)[None, :, :]
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            flat_out,
            [rows[:, None] * n + cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _far_loads_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` reading its operand columns ``2**32`` past them,
    in int64: every lane is masked off, and int32 would wrap them back inside."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = (groups.begin * 32 + hl.arange(block_groups * 32)).to(
                torch.int64
            ) + 2**32
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        out_cols = cols
        hl.store(
            flat_out,
            [rows[:, None] * n + out_cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (out_cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _far_store_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` storing its output columns ``2**32`` past them, in
    int64: every lane is masked off, and int32 would wrap them back inside."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        out_cols = cols.to(torch.int64) + 2**32
        hl.store(
            flat_out,
            [rows[:, None] * n + out_cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (out_cols[None, :] < n),
        )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _branch_store_packed_mxfp8_gemm(
    x: torch.Tensor, x_scale: torch.Tensor, y: torch.Tensor, y_scale: torch.Tensor
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` storing its output, or for ``spill`` 2 an FP32
    copy of it, as the pretuned split-K kernel picks its output."""
    m, k = x.size()
    n, _ = y.size()
    spill = hl.register_tunable("spill", PowerOfTwoFragment(1, 2, 1))
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    copy = torch.empty([m, n], dtype=torch.float32, device=x.device)
    flat_copy = copy.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if spill == 1:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
        else:
            hl.store(
                flat_copy,
                [rows[:, None] * n + cols[None, :]],
                acc,
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _runtime_branch_store_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` storing its output, or twice it, in either arm
    of a branch on ``flag``, which only the launch decides."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if flag[0] > 0:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
        else:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                (acc * 2).to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def _release_packed_mxfp8_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    y: torch.Tensor,
    y_scale: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    """``packed_mxfp8_gemm`` that, after loading its operands, publishes
    their use with a release ``flag`` update another program may acquire."""
    m, k = x.size()
    n, _ = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    flat_out = out.view(m * n)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(4)
    for task_m, task_n in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n]
    ):
        rows = hl.arange(block_m) + task_m * block_m
        cols = hl.arange(block_n) + task_n * block_n
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        for groups in hl.tile(k // 32, block_size=block_groups):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                x,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < k),
            )
            rhs = hl.load(
                y,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < k),
            )
            lhs_words = hl.load(
                x_scale, [rows, groups.index // 4], extra_mask=rows[:, None] < m
            )
            rhs_words = hl.load(
                y_scale, [cols, groups.index // 4], extra_mask=cols[:, None] < n
            )
            hl.atomic_xchg(flag, [task_m], 1, sem="release")
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        hl.store(
            flat_out,
            [rows[:, None] * n + cols[None, :]],
            acc.to(out.dtype),
            extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
        )
    return out


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    return packed_mxfp8_inputs(m, n, k, device=torch.device("cpu"))[0]


def _operand_descriptors(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **facts: object
) -> tuple[bool, bool]:
    """Whether ``x`` and ``y`` load through descriptors in the code of
    ``kernel``'s seed without warp specialization (each operand then lowers
    on its own): the binding's own code, or an Inductor render under
    ``facts``."""
    # A fresh kernel, so no binding is shared with the GPU tests' kernels.
    kernel = helion.kernel(
        kernel.fn,
        autotune_effort="none",
        static_shapes=kernel.settings.static_shapes,
    )
    bound = kernel.bind(args)
    assert bound.reads_inductor_call_facts
    seed = bound.config_spec.default_config()
    config = helion.Config(
        **{
            **seed.config,
            "range_warp_specializes": [
                False if enabled else enabled
                for enabled in seed.config["range_warp_specializes"]
            ],
        }
    )
    code = (
        unparse(render_inductor_ast(bound, config, **facts))  # pyrefly: ignore[bad-argument-type]
        if facts
        else bound.to_triton_code(config)
    )
    return "(x_desc.load(" in code, "(y_desc.load(" in code


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_inductor_render_needs_traced_layouts() -> None:
    """An Inductor render (``inductor_call_facts``) loads an operand through a
    descriptor only where its calls hold the layout facts the descriptor
    reads: the traced strides of an input the graph keeps symbolic need not
    hold (``traced_strides_hold``), and a dynamic-shape graph skips the
    dispatch guard that keys a dynamic extent."""
    args = _inputs(256, 512, 1280)
    x = args[0]
    with packed_mxfp8_cpu_codegen():
        # Static shapes: only the strides of an input the graph keeps
        # symbolic stop its descriptor.
        symbolic_x = {
            "symbolic_input_layouts": True,
            "symbolic_strides": tensor_input_sources({"x": x}, [x]),
        }
        for facts, expected in (
            ({}, (True, True)),
            ({"symbolic_input_layouts": False}, (True, True)),
            ({"symbolic_input_layouts": True}, (True, True)),
            (symbolic_x, (False, True)),
        ):
            assert _operand_descriptors(packed_mxfp8_gemm, args, **facts) == (
                expected
            ), facts
        # Dynamic shapes: the scale tensors' dynamic extents, which only the
        # dispatch guards key, keep the whole dot generic in a dynamic graph.
        for facts, expected in (
            ({}, (True, True)),
            ({"symbolic_input_layouts": False}, (True, True)),
            ({"symbolic_input_layouts": True}, (False, False)),
        ):
            assert (
                _operand_descriptors(dynamic_packed_mxfp8_gemm, args, **facts)
                == expected
            ), facts


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("bindings are made only in compiled mode")
def test_dynamic_shapes_bind_once_per_extent_class() -> None:
    """The native lowering's descriptor guards key only what it reads of its
    operands and scale words: across M of 24 to 300 a dynamic-shape kernel
    keeps one binding (a fixed config, or autotuning off), and M = 1, where a
    descriptor would not broadcast the extent of one, gets its own."""
    with packed_mxfp8_cpu_codegen():
        for settings in (
            {"config": helion.Config(block_sizes=[128, 128, 4], num_warps=4)},
            {"autotune_effort": "none"},
        ):
            kernel = helion.kernel(
                packed_mxfp8_gemm.fn, static_shapes=False, **settings
            )
            assert kernel.bind(_inputs(96, 512, 1280)).reads_inductor_call_facts
            bindings = {
                id(kernel.bind(_inputs(m, 512, 1280))) for m in (24, 48, 96, 200, 300)
            }
            assert len(bindings) == 1, settings
            assert id(kernel.bind(_inputs(1, 512, 1280))) not in bindings, settings


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_inductor_render_keeps_warp_specialization_only_for_a_whole_dot() -> None:
    """The seed warp-specializes when the binding's own launches take the dot
    whole.  An Inductor render under which the dot keeps its generic loads
    (a dynamic graph: the scale words' dynamic extents are unguarded) drops
    the warp specialization, which Triton cannot compile around generic
    packed loads; a render that keeps the whole dot keeps it."""
    args = _inputs(256, 512, 1280)
    with packed_mxfp8_cpu_codegen():
        kernel = helion.kernel(
            dynamic_packed_mxfp8_gemm.fn, autotune_effort="none", static_shapes=False
        )
        bound = kernel.bind(args)
        seed = bound.config_spec.default_config()
        assert any(seed.config["range_warp_specializes"])

        def facts(code: str) -> tuple[bool, bool]:
            return "warp_specialize=True" in code, "(x_desc.load(" in code

        assert facts(bound.to_triton_code(seed)) == (True, True)
        for symbolic_input_layouts, expected in (
            (False, (True, True)),
            (True, (False, False)),
        ):
            code = unparse(
                render_inductor_ast(
                    bound, seed, symbolic_input_layouts=symbolic_input_layouts
                )
            )
            assert facts(code) == expected, symbolic_input_layouts


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_inductor_render_keeps_warp_specialization_of_generic_loads() -> None:
    """A config whose own code keeps the generic loads in a warp-specialized
    K loop (a 16-row tile, as the pretuned 16 x 2048 x 4096 config) renders
    the same code under Inductor, warp specialization included."""
    args = _inputs(16, 2048, 4096)
    config = helion.Config(
        block_sizes=[16, 128, 4],
        pid_type="flat",
        range_warp_specializes=[None, True],
    )
    with packed_mxfp8_cpu_codegen():
        kernel = helion.kernel(
            packed_mxfp8_gemm.fn, autotune_effort="none", static_shapes=True
        )
        bound = kernel.bind(args)
        code = bound.to_triton_code(config)
        assert "warp_specialize=True" in code
        assert "mxfp8_lhs_words" not in code
        for symbolic_input_layouts in (False, True):
            render = unparse(
                render_inductor_ast(
                    bound, config, symbolic_input_layouts=symbolic_input_layouts
                )
            )
            assert "warp_specialize=True" in render, symbolic_input_layouts


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_write_through_device_pointers_keeps_generic_loads() -> None:
    """The native loads are issued at the dot, after the loads they replace,
    so a write in the dot's phase that may reach their tensors keeps the
    generic loads: here a store through a stack tensor's device pointers,
    which no storage of the kernel bounds."""
    x, x_scale, y, y_scale = _inputs(128, 256, 256)
    args = (
        x,
        x_scale,
        y,
        y_scale,
        torch.zeros(1, dtype=torch.uint64),
        torch.empty_like(x),
    )
    with packed_mxfp8_cpu_codegen():
        bound = _stack_write_packed_mxfp8_gemm.bind(args)
        code = bound.to_triton_code(bound.config_spec.default_config())
    assert "(x_desc.load(" not in code
    assert "mxfp8_lhs_words" not in code


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("bindings are made only in compiled mode")
def test_misaligned_operand_rebinds() -> None:
    """The native lowering loads its operands and scale words through host
    descriptors, which need a 16-byte aligned base, whatever ``indexing``
    says: an operand at a misaligned base gets its own binding, whose code
    keeps the generic loads."""
    with packed_mxfp8_cpu_codegen():
        for source, static_shapes in (
            (packed_mxfp8_gemm, True),
            (dynamic_packed_mxfp8_gemm, False),
        ):
            kernel = helion.kernel(
                source.fn, autotune_effort="none", static_shapes=static_shapes
            )
            x, x_scale, y, y_scale = _inputs(128, 256, 256)
            bound = kernel.bind((x, x_scale, y, y_scale))
            code = bound.to_triton_code(bound.config_spec.default_config())
            assert "(x_desc.load(" in code, static_shapes
            shifted = torch.empty(x.numel() + 1, dtype=x.dtype)[1:].view(x.shape)
            shifted.copy_(x)
            misaligned = kernel.bind((shifted, x_scale, y, y_scale))
            assert misaligned is not bound, static_shapes
            code = misaligned.to_triton_code(misaligned.config_spec.default_config())
            assert "(x_desc.load(" not in code, static_shapes


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_int64_coordinates_keep_generic_accesses() -> None:
    """A descriptor takes its coordinates cast to int32.  Where an index is
    computed in int64 (here ``2**32`` past the tensor, every lane masked
    off), the cast would wrap it back inside, so the load or flattened
    store keeps its generic form, which reads zeros or writes nothing."""
    args = _inputs(128, 256, 256)
    with packed_mxfp8_cpu_codegen():
        for kernel, access, native in (
            (packed_mxfp8_gemm, "(x_desc.load(", True),
            (packed_mxfp8_gemm, "out_desc.store(", True),
            (_far_loads_packed_mxfp8_gemm, "(x_desc.load(", False),
            (_far_store_packed_mxfp8_gemm, "out_desc.store(", False),
        ):
            kernel = helion.kernel(
                kernel.fn, autotune_effort="none", static_shapes=True
            )
            bound = kernel.bind(args)
            code = bound.to_triton_code(bound.config_spec.default_config())
            assert (access in code) is native, (kernel.name, access)


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_tma_store_needs_a_single_output() -> None:
    """A TMA store's staging tile is budgeted beside the operand ring alone,
    and a second output's store needs shared memory of its own, so a kernel
    writing two outputs keeps pointer stores for both."""
    with packed_mxfp8_cpu_codegen():
        kernel = helion.kernel(
            two_output_packed_mxfp8_gemm.fn, autotune_effort="none", static_shapes=True
        )
        bound = kernel.bind(_inputs(256, 512, 1280))
        code = bound.to_triton_code(bound.config_spec.default_config())
    assert "warp_specialize=True" in code
    assert "(x_desc.load(" in code
    assert "_desc.store(" not in code


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_tma_store_ignores_a_branch_the_config_rules_out() -> None:
    """A store in a branch the config rules out (``if spill == 1``, as the
    pretuned kernel's ``split_k``) writes nothing, so the output's store
    keeps its TMA store."""
    with packed_mxfp8_cpu_codegen():
        kernel = helion.kernel(
            _branch_store_packed_mxfp8_gemm.fn,
            autotune_effort="none",
            static_shapes=True,
        )
        bound = kernel.bind(_inputs(256, 512, 1280))
        seed = bound.config_spec.default_config()
        code = bound.to_triton_code(seed)
    assert seed["spill"] == 1
    assert "warp_specialize=True" in code
    assert "out_desc.store(" in code


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_tma_store_needs_a_single_store() -> None:
    """Two stores of one output, in the arms of a branch the launch decides,
    each need shared memory, so the output keeps pointer stores."""
    args = (*_inputs(256, 512, 1280), torch.ones(1, dtype=torch.int32))
    with packed_mxfp8_cpu_codegen():
        kernel = helion.kernel(
            _runtime_branch_store_packed_mxfp8_gemm.fn,
            autotune_effort="none",
            static_shapes=True,
        )
        bound = kernel.bind(args)
        code = bound.to_triton_code(bound.config_spec.default_config())
    assert "warp_specialize=True" in code
    assert "(x_desc.load(" in code
    assert "_desc.store(" not in code


@skipUnlessBackends(["triton"])
@skipIfTileIR(_TILEIR)
@skipIfRefEager("Codegen inspection not applicable in ref eager mode")
def test_synchronizing_op_keeps_generic_loads() -> None:
    """The native loads are issued at the dot, so an op of the dot's phase
    that may synchronize with other programs (here a release atomic between
    the loads and the dot) keeps the generic loads where they stand."""
    x, x_scale, y, y_scale = _inputs(128, 256, 256)
    config = helion.Config(block_sizes=[128, 256, 8], num_stages=1)
    with packed_mxfp8_cpu_codegen():
        for kernel, args, native in (
            (packed_mxfp8_gemm, (x, x_scale, y, y_scale), True),
            (
                _release_packed_mxfp8_gemm,
                (x, x_scale, y, y_scale, torch.zeros(1, dtype=torch.int32)),
                False,
            ),
        ):
            bound = helion.kernel(
                kernel.fn, autotune_effort="none", static_shapes=True
            ).bind(args)
            code = bound.to_triton_code(config)
            assert ("(x_desc.load(" in code) is native, kernel.name
            assert ("mxfp8_lhs_words" in code) is native, kernel.name
