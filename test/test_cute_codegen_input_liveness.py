"""Code generation does not depend on which example inputs are still alive.

A bound kernel holds the tensors it was bound with weakly, and a direct
``to_code`` reads whichever of them the caller still holds.  Facts the bound
kernel's cache key already backs (a packet's base alignment, an input's sizes
and strides) must render the same code once some or all of them are collected.
They hold only where dispatch checks that key: Inductor's render of a
dynamic-shape graph, bound to placeholder inputs, must not use them, and no
Inductor render proves more base alignment than Inductor gives its inputs.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
import torch
from torch._dynamo.source import ConstantSource
from torch._dynamo.testing import rand_strided
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext

import helion
from helion._compat import supports_torch_compile_template_lowering
from helion._compiler.ast_extension import unparse
from helion._compiler.compile_environment import HELION_STATIC_DIMS_ATTR
from helion._compiler.compile_environment import tensor_input_sources
from helion._compiler.cute.fragment_storage import metadata_guarded
from helion._compiler.cute.memory_ops import tensor_has_specialized_tma_alignment
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterator

    from torch._guards import Source

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _cpu_target() -> Iterator[None]:
    with cpu_render_target():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _head_rms_norm(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tokens, heads, head_dim = q.shape
    half = head_dim // 2
    out = torch.empty_like(q)
    for tile_t, tile_h in hl.tile([tokens, heads]):
        x = q[tile_t, tile_h, :].float()
        rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        out[tile_t, tile_h, :half] = (
            q[tile_t, tile_h, :half].float() * rstd * weight[:half].float()
        ).to(q.dtype)
        out[tile_t, tile_h, half:] = (
            q[tile_t, tile_h, half:].float() * rstd * weight[half:].float()
        ).to(q.dtype)
    return out


def _head_rms_norm_fn(q: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tokens, heads, head_dim = q.shape
    head_dim = hl.specialize(head_dim)
    out = torch.empty_like(q)
    for tile_t, tile_h in hl.tile([tokens, heads]):
        x = q[tile_t, tile_h, :].float()
        rstd = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        # ``weight[:]`` would broadcast against the specialized head_dim and
        # specialize weight's size too; indexed by an arange it stays symbolic.
        cols = hl.arange(head_dim)
        out[tile_t, tile_h, :] = (x * rstd * weight[cols].float()).to(q.dtype)
    return out


_head_rms_norm_dynamic = helion.kernel(
    _head_rms_norm_fn, backend="cute", static_shapes=False, autotune_effort="none"
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _state_update(
    state_indices: torch.Tensor, state: torch.Tensor, packed: torch.Tensor
) -> torch.Tensor:
    # Only state_indices' (dynamic) size gives the batch grid axis its extent.
    batches = state_indices.size(0)
    heads = hl.specialize(state.size(1))
    rows = hl.specialize(state.size(2))
    width = hl.specialize(state.size(3))
    out = torch.empty([batches, heads, rows], dtype=state.dtype, device=state.device)
    for tile_batch, tile_head, tile_row in hl.tile(
        [batches, heads, rows], block_size=[1, 1, None]
    ):
        batch_index = tile_batch.id
        head_index = tile_head.id
        state_index = state_indices[batch_index]
        cols = hl.arange(width)
        state_values = state[state_index, head_index, tile_row, cols].float()
        packed_values = packed[batch_index, head_index, cols].float()
        updated = state_values + packed_values[None, :]
        out[batch_index, head_index, tile_row] = updated.sum(-1).to(state.dtype)
        state[state_index, head_index, tile_row, cols] = updated
    return out


@helion.kernel(
    backend="cute",
    static_shapes=False,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _center_rows_into_a_fresh_view(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))
    # The second half of a fresh buffer: its storage offset, rows * n, follows
    # from x's dynamic row count.
    out = x.new_empty([2, x.size(0), n]).select(0, 1)
    column_sums = x.new_empty(
        [(x.size(0) + m_block - 1) // m_block, n], dtype=torch.float32
    )
    for tile_cta in hl.tile(x.size(0), block_size=m_block):
        acc = weight.new_zeros(n, dtype=torch.float32)
        for tile_m in hl.tile(tile_cta.begin, tile_cta.end):
            x_m = x[tile_m, :].to(torch.float32)
            acc += (x_m * weight[None, :].float()).sum(0)
            out[tile_m, :] = (x_m - x_m.mean(-1)[:, None]).to(x.dtype)
        column_sums[tile_cta.id, :] = acc
    return out, column_sums


def _row_logsumexp_fn(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.shape[0]):
        values = x[row, :].float()
        maximum = values.amax(-1)
        denominator = torch.exp(values - maximum[:, None]).sum(-1)
        out[row] = denominator.log() + maximum
    return out


_row_logsumexp = helion.kernel(
    _row_logsumexp_fn, backend="cute", static_shapes=True, autotune_effort="none"
)
_row_logsumexp_dynamic = helion.kernel(
    _row_logsumexp_fn, backend="cute", static_shapes=False, autotune_effort="none"
)


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _row_sums_of_a_pair(xs: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty(xs[0].shape[0], dtype=torch.float32, device=xs[0].device)
    for row in hl.tile(xs[0].shape[0]):
        out[row] = xs[0][row, :].float().sum(-1) + xs[1][row, :].float().sum(-1)
    return out


def _inductor_render(
    bound: BoundKernel,
    config: helion.Config,
    *,
    symbolic_input_layouts: bool,
    symbolic_strides: Collection[Source] = (),
) -> str:
    """``config``'s code as Inductor renders it into a graph whose input
    sizes and strides are static, or symbolic (those of the inputs in
    ``symbolic_strides`` among them)."""
    from helion._compiler._inductor.template_buffer import render_inductor_ast

    return unparse(
        render_inductor_ast(
            bound,
            config,
            symbolic_input_layouts=symbolic_input_layouts,
            symbolic_strides=symbolic_strides,
        )
    )


def _renders(
    kernel: helion.Kernel,
    make_args: Callable[[], list[torch.Tensor]],
    config: helion.Config,
) -> list[str]:
    """``config``'s code with every input alive, after the first one is
    collected and after all of them are."""
    args = make_args()
    bound = cpu_bind(kernel, tuple(args))
    refs = [
        bound._runtime_tensor_refs_by_name[name] for name in kernel.signature.parameters
    ]
    codes = [bound.to_code(config)]
    del args[0]
    gc.collect()
    assert refs[0]() is None
    codes.append(bound.to_code(config))
    args.clear()
    gc.collect()
    assert all(ref() is None for ref in refs)
    codes.append(bound.to_code(config))
    return codes


def test_persistent_packets_keep_their_alignment_proof() -> None:
    live, partial, dead = _renders(
        _head_rms_norm,
        lambda: [
            torch.randn(64, 8, 256, dtype=torch.bfloat16),
            torch.randn(256, dtype=torch.bfloat16),
        ],
        helion.Config(
            block_sizes=[4, 2],
            cute_vector_widths=[8, 4, 1, 1],
            num_threads=[4, 2, 2, 0],
            reduction_loops=[256],
        ),
    )
    # The q and weight halves load as branch-local packets.
    assert live.count("_persistent_branch_vec_") == 8, live
    assert partial == live
    assert dead == live


def test_dynamic_layout_of_a_collected_input_comes_from_the_bound_signature() -> None:
    """Without static shapes a collected input's traced strides (q's) and
    lane size (weight's) are symbolic; its bound signature still proves the
    packet layout."""
    live, partial, dead = _renders(
        _head_rms_norm_dynamic,
        lambda: [
            torch.randn(64, 8, 256, dtype=torch.bfloat16),
            torch.randn(256, dtype=torch.bfloat16),
        ],
        helion.Config(
            block_sizes=[4, 2],
            cute_vector_widths=[8, 8],
            num_threads=[4, 2, 32],
        ),
    )
    assert "_persistent_branch_vec_0 = cute.arch.load(q.iterator" in live, live
    assert "_persistent_branch_vec_1 = cute.arch.load(weight.iterator" in live, live
    assert partial == live
    assert dead == live


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_inductor_render_of_a_dynamic_graph_takes_no_proof_from_placeholders() -> None:
    """Inductor binds a dynamic-shape graph's kernel to placeholder inputs
    (``lower_helion_kernel``: 64 for a dynamic size, 1 for a dynamic stride)
    and renders code for every size the graph accepts, with no runtime
    arguments.  The placeholder weight's length is a multiple of the packet
    width; a later call's need not be (255 of the 256 lanes ``hl.arange``
    reads, the last one masked), so the render takes no packet from it.  The
    binding's own launches, whose dispatch checks that residue, keep it."""
    # Dynamo keeps the head dim static (``mark_static``), so the lowering
    # binds it as a literal.
    q = rand_strided([64, 64, 256], [1, 256, 1], dtype=torch.bfloat16)
    setattr(q, HELION_STATIC_DIMS_ATTR, frozenset({2}))
    weight = rand_strided([64], [1], dtype=torch.bfloat16)
    bound = cpu_bind(_head_rms_norm_dynamic, (q, weight))
    config = helion.Config(
        block_sizes=[4, 2],
        cute_vector_widths=[8, 8],
        num_threads=[4, 2, 32],
    )

    packet = "cute.arch.load(weight.iterator"
    assert packet in bound.to_code(config)
    assert packet in _inductor_render(bound, config, symbolic_input_layouts=False)
    dynamic = _inductor_render(bound, config, symbolic_input_layouts=True)
    assert packet not in dynamic, dynamic
    assert "weight.iterator" in dynamic, dynamic


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_inductor_render_of_a_dynamic_graph_masks_each_lane_of_a_partial_tail() -> None:
    """A rolled reduction over the 64 columns of a placeholder input takes
    eight-lane packets with one mask per packet only from the binding's
    keyed size residue.  Inductor runs the render at every row length the
    graph accepts: at 61, the last packet's mask (``56 < 61``) would load
    lanes 61-63 and sum them into the row, so every lane keeps its own mask."""
    x = rand_strided([64, 64], [64, 1], dtype=torch.bfloat16)
    bound = cpu_bind(_row_logsumexp_dynamic, (x,))
    config = helion.Config(
        block_sizes=[1],
        reduction_loops=[32],
        num_threads=[1, 4],
        cute_vector_widths=[8, 1],
    )

    packet = "cute.arch.load(x.iterator"
    assert packet in bound.to_code(config)
    dynamic = _inductor_render(bound, config, symbolic_input_layouts=True)
    assert packet not in dynamic, dynamic
    assert "rindex_1 < x_size_1" in dynamic, dynamic


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_inductor_render_proves_no_base_alignment_past_16_bytes() -> None:
    """Inductor aligns graph inputs only to 16 bytes, and the placeholder it
    binds the kernel to, a fresh allocation, reproduces only that.  Packets
    of eight fp32 lanes need 32 bytes, which only the binding's own launches
    check (no row copy is that wide: the launcher declares its pointers at
    most 16-byte aligned); a row copy of four lanes needs 16, which the
    render proves from the bound signature."""
    x = rand_strided([3, 32000], [32000, 1], dtype=torch.float32)
    bound = cpu_bind(_row_logsumexp, (x,))

    def config(vector_width: int) -> helion.Config:
        return helion.Config(
            block_sizes=[1],
            num_threads=[1, 256],
            reduction_loops=[16384],
            cute_vector_widths=[vector_width, 1],
            cute_lane_layouts=["strided", "blocked"],
            cute_reduction_reloads=["register"],
            cute_cluster_n=2,
            cute_cluster_warp_online=True,
            cute_scalar_epilogue=True,
            cute_register_row=True,
            cute_warp_fp32_redux=True,
            cute_register_row_tensor_copy=True,
        )

    copy = "_cute_complete_row_tensor_copy("
    packet = "cute.arch.load(x.iterator"
    wide = bound.to_code(config(8))
    assert packet in wide and copy not in wide, wide
    wide = _inductor_render(bound, config(8), symbolic_input_layouts=False)
    assert packet not in wide and copy not in wide, wide
    narrow = _inductor_render(bound, config(4), symbolic_input_layouts=False)
    assert copy in narrow and "vector_width=4)" in narrow, narrow


def test_a_dynamic_stride_proves_a_tensor_map_only_behind_dispatch_guards() -> None:
    """The bound signature of a contiguous 64 x 256 input proves a TensorMap's
    16-byte outer stride for the binding's own launches, which dispatch
    checks.  Inductor's render of a dynamic-shape graph took that residue from
    a placeholder (its stride of 1) and runs at every row stride."""
    bound = cpu_bind(_row_logsumexp_dynamic, (torch.empty(64, 256),))
    env = bound.env
    traced = next(iter(env.input_sources))
    assert not isinstance(traced.stride(0), int)
    assert tensor_has_specialized_tma_alignment(env, traced)
    with env.without_dispatch_guards():
        assert not tensor_has_specialized_tma_alignment(env, traced)


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
@pytest.mark.parametrize("symbolic_stride", [False, True])
def test_a_symbolic_stride_proves_no_layout_in_a_dynamic_render(
    symbolic_stride: bool,
) -> None:
    """A graph that saw ``x[:, :64]`` and then ``x[:, ::2]`` keeps the inner
    stride symbolic.  ``lower_helion_kernel``'s placeholder gives it 1, which
    the bind traces as a literal (the sizes stay static), and the render runs
    at a stride of 2: an eight-lane packet or a TensorMap proven from that
    stride would read the wrong elements.  The binding's own launches, and a
    graph whose strides are static, keep both."""
    x = rand_strided([64, 64], [128, 1], dtype=torch.bfloat16)
    setattr(x, HELION_STATIC_DIMS_ATTR, frozenset({0, 1}))
    kernel = helion.kernel(
        _row_logsumexp_fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    bound = cpu_bind(kernel, (x,))
    config = helion.Config(
        block_sizes=[1], num_threads=[1, 8], cute_vector_widths=[8, 1]
    )
    # What ``lower_helion_kernel`` passes for a graph that keeps x's inner
    # stride symbolic.
    symbolic_strides = (
        tensor_input_sources({"x": x}, [x]) if symbolic_stride else frozenset()
    )

    packet = "cute.arch.load(x.iterator"
    assert packet in bound.to_code(config)
    dynamic = _inductor_render(
        bound, config, symbolic_input_layouts=True, symbolic_strides=symbolic_strides
    )
    assert (packet in dynamic) is not symbolic_stride, dynamic
    env = bound.env
    traced = next(iter(env.input_sources))
    assert traced.stride() == (128, 1)
    assert tensor_has_specialized_tma_alignment(env, traced)
    with env.without_dispatch_guards(symbolic_strides):
        assert tensor_has_specialized_tma_alignment(env, traced) is not symbolic_stride


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_a_symbolic_stride_of_a_list_item_proves_no_layout_in_a_dynamic_render() -> (
    None
):
    """``lower_helion_kernel`` names the inputs whose strides its graph keeps
    symbolic as the layout proofs name an input (``tensor_input_sources``):
    a list item by its own source, not the list's, which the binding records
    for it.  The item at a symbolic stride loses its packets; its neighbour,
    whose strides are static, keeps them."""
    xs = [rand_strided([64, 64], [128, 1], dtype=torch.bfloat16) for _ in range(2)]
    for x in xs:
        setattr(x, HELION_STATIC_DIMS_ATTR, frozenset({0, 1}))
    bound = cpu_bind(_row_sums_of_a_pair, (xs,))
    config = helion.Config(
        block_sizes=[1], num_threads=[1, 8], cute_vector_widths=[8, 1]
    )

    dynamic = _inductor_render(
        bound,
        config,
        symbolic_input_layouts=True,
        symbolic_strides=tensor_input_sources({"xs": xs}, [xs[1]]),
    )
    assert "cute.arch.load(xs_item_0.iterator" in dynamic, dynamic
    assert "cute.arch.load(xs_item_1.iterator" not in dynamic, dynamic


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_configs_tuned_under_different_call_facts_key_apart() -> None:
    """A static render, a dynamic one and one that keeps an input's strides
    symbolic hold fewer facts than the binding's own launches, each its own
    set (``inductor_call_facts``): a config tuned for one may be refused by
    another, as the radix selection is at a symbolic stride.  So each keys
    its disk-cached configs apart, fused or not."""
    from helion._compiler._inductor.template_buffer import _FusionAutotuneAdapter
    from helion._compiler._inductor.template_buffer import inductor_call_facts

    x = rand_strided([64, 64], [128, 1], dtype=torch.bfloat16)
    setattr(x, HELION_STATIC_DIMS_ATTR, frozenset({0, 1}))
    kernel = helion.kernel(
        _row_logsumexp_fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    bound = cpu_bind(kernel, (x,))

    def factory(config: helion.Config) -> tuple[Callable[..., object], str | None]:
        raise AssertionError("nothing is tuned")

    fused = _FusionAutotuneAdapter(bound, factory, "fusion")
    keys = [(bound.extra_cache_key(), fused.extra_cache_key())]
    for symbolic_input_layouts, symbolic_strides in (
        (False, frozenset()),
        (True, frozenset()),
        (True, tensor_input_sources({"x": x}, [x])),
    ):
        with inductor_call_facts(
            bound,
            symbolic_input_layouts=symbolic_input_layouts,
            symbolic_strides=symbolic_strides,
        ):
            keys.append((bound.extra_cache_key(), fused.extra_cache_key()))
    assert keys[0] == ("", "fusion")
    assert len({plain for plain, _ in keys}) == len(keys)
    assert all(key == plain + "fusion" for plain, key in keys)
    assert bound.extra_cache_key() == ""


def test_an_outer_trace_binding_takes_no_hint_once_its_inputs_are_collected() -> None:
    """A binding made from an outer trace's FakeTensors (Dynamo, make_fx)
    has the hints that trace chose, which no call need share.  The binding
    records that, so its hints stay refused after the FakeTensors are
    collected, as they are while alive."""
    shape_env = ShapeEnv()
    fake = FakeTensorMode(shape_env=shape_env).from_tensor(
        torch.empty(64, 256),
        source=ConstantSource("x"),
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC]
        ),
    )
    bound = cpu_bind(_row_logsumexp_dynamic, (fake,))
    traced = next(iter(bound.env.input_sources))
    size = traced.shape[1]
    assert isinstance(size, torch.SymInt)
    assert bound.env.keyed_hint(size) is None
    del fake
    gc.collect()
    assert bound.env.keyed_hint(size) is None


def test_collective_copies_keep_their_stride_and_alignment_proofs() -> None:
    live, partial, dead = _renders(
        _matmul,
        lambda: [
            torch.randn(256, 256, dtype=torch.bfloat16),
            torch.randn(256, 256, dtype=torch.bfloat16),
        ],
        helion.Config(
            block_sizes=[64, 32, 64],
            num_threads=[4, 32, 1],
            cute_collective_copy="async",
            cute_collective_mma=True,
        ),
    )
    # Both operands stage through asynchronous shared-memory copies.
    assert live.count("cp_async_shared_global(") == 2, live
    assert partial == live
    assert dead == live


def test_proven_bounds_keep_the_exact_sizes_of_collected_inputs() -> None:
    """The ``input_tensor_metadata`` fact keys the bound kernel on exact input
    sizes, so proven bounds take the batch grid extent and ``packed``'s size
    from the trace whether or not the caller still holds those inputs."""
    live, partial, dead = _renders(
        _state_update,
        lambda: [
            torch.tensor([0, 1], dtype=torch.int32),
            torch.empty(3, 3, 128, 128, dtype=torch.bfloat16),
            torch.empty(2, 3, 128, dtype=torch.bfloat16),
        ],
        helion.Config(
            block_sizes=[128],
            num_threads=[8, 16],
            cute_vector_widths=[8, 1, 1, 1],
            pid_type="xyz",
            cute_proven_bounds=True,
        ),
    )
    assert "_cute_proven_bounds_" in live, live
    assert "< packed_size_2" not in live, live
    # state_indices, collected first, sizes the batch grid axis.
    assert partial == live
    # packed, collected with the rest, has its size checks retired.
    assert dead == live


def test_exact_input_metadata_proves_nothing_without_dispatch_guards() -> None:
    """The readers that take an input's size or stride from its trace hint
    under the ``input_tensor_metadata`` fact (computed-fragment capacities
    through ``metadata_guarded``, the gathered, block-scaled and mxfp8 MMA
    plans) ask ``exact_input_metadata_holds``.  Inductor's render of a
    dynamic-shape graph took those hints from placeholders: a capacity sized
    from a placeholder's 64 rows would be overrun by a call that passes 4096."""
    args = [
        torch.tensor([0, 1], dtype=torch.int32),
        torch.empty(3, 3, 128, 128, dtype=torch.bfloat16),
        torch.empty(2, 3, 128, dtype=torch.bfloat16),
    ]
    bound = cpu_bind(_state_update, tuple(args))
    bound.to_code(
        helion.Config(
            block_sizes=[128],
            num_threads=[8, 16],
            cute_vector_widths=[8, 1, 1, 1],
            pid_type="xyz",
            cute_proven_bounds=True,
        )
    )
    env = bound.env
    assert "input_tensor_metadata" in env.compiler_fact_specialization_facts
    (batches,) = [
        tensor.shape[0]
        for tensor, source in env.input_sources.items()
        if getattr(source, "local_name", None) == "state_indices"
    ]
    batches = env.shape_env.replace(batches._sympy_())
    assert env.exact_input_metadata_holds()
    assert metadata_guarded(env, batches)
    with env.without_dispatch_guards():
        assert not env.exact_input_metadata_holds()
        assert not metadata_guarded(env, batches)


def test_resident_stores_keep_the_offset_alignment_of_a_collected_input() -> None:
    """``out`` starts rows * n elements into a fresh buffer; the row count the
    cache key fixes keeps that offset 16-byte aligned, so the resident
    reduction stores ``out`` in vectors whether or not x is still alive."""
    live, partial, dead = _renders(
        _center_rows_into_a_fresh_view,
        lambda: [
            torch.empty(64, 4096, dtype=torch.bfloat16),
            torch.empty(4096, dtype=torch.bfloat16),
        ],
        helion.Config(
            block_sizes=[32, 1],
            num_threads=[1, 1, 256],
            cute_vector_widths=[8, 1, 1],
            cute_reduction_schedule="resident",
        ),
    )
    assert "_cute_resident_store_vector(out," in live, live
    assert partial == live
    assert dead == live
