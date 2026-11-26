"""Lower flex_attention's forward and backward to the Helion kernels.

``lower_forward`` / ``lower_backward`` take the call Inductor's flex_attention
lowering offers a backend (``FlexAttentionCall`` /
``FlexAttentionBackwardCall``) and first decide, adding no IR node, whether
the Helion kernels run it: they raise :class:`Declined` with the reason when
they do not, and Inductor lowers the call itself. Otherwise they lower it and
return its outputs. Deciding realizes the inputs it reads, as Inductor's own
template lowering does: their layouts are frozen, and a view no
ReinterpretView can express gets a copy (dead code if unused). This stays
when the call is declined. The kernels compile for static shapes; a call with
a symbolic size is declined rather than specialized, since the guards would
also change the graph Inductor's kernels lower (the forward's guards reach
the backward's).

The score_mod, mask_mod and joint subgraphs go to the kernels as FX
GraphModules, which Helion inlines at trace time; each kernel call lowers
through Helion's own Inductor template lowering.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import logging
import math
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config as inductor_config
from torch._inductor.ir import ExternKernel
from torch._inductor.ir import TemplateBuffer
from torch._inductor.ir import TensorBox
from torch._inductor.kernel.flex.backend import FlexAttentionDeclined
from torch._inductor.lowering import full
from torch._inductor.lowering import lowerings
from torch._inductor.lowering import to_dtype
from torch._inductor.virtualized import V
from torch._prims_common import compute_required_storage_length
from torch._prims_common import make_contiguous_strides_for
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree

from . import _kernels
import helion
from helion._compiler._dynamo.higher_order_ops import _rebuild_container_args
from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
from helion._compiler._dynamo.variables import _infer_output_spec
from helion._compiler._inductor.template_buffer import lower_helion_kernel
from helion._compiler.cute.backend import _attention_config_envelope_ok
from helion._compiler.cute.bounded_gather import integer_bounds
from helion._compiler.cute.cutedsl_compat import _cute_backend_requirement_error
from helion._compiler.cute.flex_match import flex_operand_strides
from helion.autotuner.base_cache import AutotuneCacheBase
import helion.language as hl
from helion.runtime.ref_mode import is_ref_mode_enabled
from helion.runtime.settings import autotuning_disallowed_reason

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch._inductor.kernel.flex.backend import FlexAttentionBackwardCall
    from torch._inductor.kernel.flex.backend import FlexAttentionCall
    from torch._inductor.kernel.flex.backend import FlexBlockMask

    from ._choices import Options
    from helion.autotuner.config_spec import ConfigSpec

log = logging.getLogger("helion.flex_attention")

aten = torch.ops.aten
prims = torch.ops.prims

# The fused kernels walk 128-wide KV blocks; a BlockMask of one
# _LARGE_SPARSE_BLOCK_SIZE block (what the frontend builds for "no BlockMask")
# becomes dense tables of this block size.
_DENSE_BLOCK_SIZE = 128


class Declined(FlexAttentionDeclined):
    """The Helion kernels do not run this flex_attention call; the message
    says why."""


def _check_supported(
    query: TensorBox, key: TensorBox, value: TensorBox, scale: object
) -> None:
    device = query.get_device()
    if device is None or device.type != "cuda":
        raise Declined("the kernels need CUDA tensors")
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise Declined("the CuTe flex kernels need a Blackwell (sm100) GPU")
    # After the device checks: it imports the CuTe DSL (about 0.6 s).
    if (reason := _cute_backend_requirement_error()) is not None:
        raise Declined(reason)
    dtype = query.get_dtype()
    if (
        key.get_dtype() != dtype
        or value.get_dtype() != dtype
        or dtype not in (torch.float16, torch.bfloat16)
    ):
        raise Declined("query, key and value must share one dtype, float16 or bfloat16")
    if not isinstance(scale, (int, float)) or not scale > 0:
        raise Declined(f"the scale must be a positive number (got {scale})")
    if query.get_size()[0] != key.get_size()[0]:
        raise Declined("NYI: query and key/value batch sizes differ")
    if V.graph.cpp_wrapper or V.graph.aot_mode:
        raise Declined(
            "NYI: cpp_wrapper and AOTInductor (the Helion kernel's call is "
            "Python wrapper code)"
        )
    if inductor_config.deterministic or inductor_config.batch_invariant:
        raise Declined(
            "Inductor's deterministic mode is on, which rules out the kernels' "
            "on-device autotuning (and the backward's atomic grad_query)"
        )


@dataclasses.dataclass
class _Tensor:
    """A kernel tensor argument: the fake tensor the kernel is checked with,
    and how to build its IR node when the call is lowered."""

    fake: torch.Tensor
    build: Callable[[], TensorBox]


def _fake(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | None,
    offset: int = 0,
) -> torch.Tensor:
    with V.fake_mode:
        storage = torch.empty(
            compute_required_storage_length(list(size), list(stride), offset),
            dtype=dtype,
            device=device,
        )
        return storage.as_strided(size, stride, offset)


def _contiguous_fake(
    size: Sequence[int], dtype: torch.dtype, device: torch.device | None
) -> torch.Tensor:
    return _fake(size, make_contiguous_strides_for(list(size)), dtype, device)


# -- planning: the static sizes of a call's arguments, declining symbolic ones


def _static_int(expr: sympy.Expr | int) -> int:
    if isinstance(expr, int):
        return expr
    expr = V.graph.sizevars.shape_env.replace(expr)
    if not expr.is_number:
        raise Declined(
            f"NYI: dynamic shapes; size {expr} is symbolic, and the kernels "
            "compile for static shapes"
        )
    return int(expr)


def _static_layout(node: TensorBox) -> tuple[list[int], list[int], int]:
    layout = TemplateBuffer.realize_template_input(node).get_layout()
    return (
        [_static_int(s) for s in layout.size],
        [_static_int(s) for s in layout.stride],
        _static_int(layout.offset),
    )


def _plan_input(node: TensorBox) -> _Tensor:
    size, stride, offset = _static_layout(node)
    fake = _fake(size, stride, node.get_dtype(), node.get_device(), offset)
    return _Tensor(fake, lambda: node)


def _plan_operand(node: TensorBox) -> _Tensor:
    """query, key, value or grad_out in a layout the fused kernels address
    with TMA (``flex_operand_strides``, the fused matcher's rule: a unit
    inner stride, and a storage offset and outer strides of dimensions
    wider than 1 that are non-zero multiples of 16 bytes). Anything else,
    such as an ``expand``ed GQA key with a zero head stride or a view at a
    misaligned storage offset, is copied into a fresh contiguous buffer.
    (A graph input at a misaligned address, which Inductor gives offset 0,
    is staged into an aligned copy with its strides by the launcher.)"""
    planned = _plan_input(node)
    if flex_operand_strides(planned.fake) is not None:
        return planned
    size = list(planned.fake.size())

    @functools.cache
    def contiguous_copy() -> TensorBox:
        # A fresh buffer: require_contiguous alone keeps a contiguous
        # view's storage offset. Built once: grad_out is read by the
        # kernel call and by the backward's row terms.
        return cast(
            "TensorBox",
            ExternKernel.require_contiguous(ExternKernel.copy_input(node)),
        )

    return _Tensor(
        _contiguous_fake(size, node.get_dtype(), node.get_device()), contiguous_copy
    )


def _plan_captured(
    buffers: Sequence[object], device: torch.device
) -> tuple[list[_Tensor], dict[int, int]]:
    """A mod's captured tensors, and its captured integers (under dynamic
    shapes, the symbolic sizes it reads) by position."""
    tensors = []
    ints = {}
    for i, buf in enumerate(buffers):
        if isinstance(buf, TensorBox):
            if buf.get_device() != device:
                raise Declined(
                    f"NYI: score_mod or mask_mod captures a tensor on "
                    f"{buf.get_device()}, not on {device} with query, key "
                    "and value"
                )
            if _is_fp8(buf.get_dtype()):
                # The CuTe DSL fails to compile the fused kernels' read
                # of an fp8 table (it aborts the process).
                raise Declined(
                    "NYI: score_mod or mask_mod captures an fp8 tensor "
                    f"({buf.get_dtype()})"
                )
            tensors.append(_plan_input(buf))
        elif isinstance(buf, int) or (
            isinstance(buf, sympy.Expr) and buf.is_integer  # pyrefly: ignore[missing-attribute]
        ):
            ints[i] = _static_int(buf)
        else:
            raise Declined("NYI: score_mod or mask_mod captures a non-integer scalar")
    return tensors, ints


def _plan_block_tables(
    tables: Sequence[TensorBox | None],
    sparse_q: int,
    sparse_kv: int,
    mask_mod: torch.fx.GraphModule,
    rows: int,
    cols: int,
    device: torch.device,
) -> tuple[list[_Tensor], int, int]:
    """The four (num_blocks, indices, full_num_blocks, full_indices) tables
    of one walk direction and its (row, column) block sizes, with
    full-block tables for a BlockMask without full blocks, and for one of
    a single ``_LARGE_SPARSE_BLOCK_SIZE`` block (the frontend's "no
    BlockMask") that block's counts spread over dense
    ``_DENSE_BLOCK_SIZE`` tables (``_dense_tables``)."""
    i32 = torch.int32
    num_blocks, indices, full_num_blocks, full_indices = tables
    if num_blocks is None or indices is None:
        raise Declined("NYI: a BlockMask built with compute_q_blocks=False")
    if sparse_q == sparse_kv == _LARGE_SPARSE_BLOCK_SIZE:
        nrows = -(-rows // _DENSE_BLOCK_SIZE)
        ncols = -(-cols // _DENSE_BLOCK_SIZE)
        counts = _plan_input(num_blocks)
        full_counts = None if full_num_blocks is None else _plan_input(full_num_blocks)
        *lead, blocks = (int(s) for s in counts.fake.size())
        if blocks != 1:
            raise Declined(
                f"NYI: a BlockMask of {blocks} {_LARGE_SPARSE_BLOCK_SIZE}-wide rows"
            )
        all_full = _masks_nothing(mask_mod)
        dense = functools.cache(
            lambda: _dense_tables(
                counts.build(),
                None if full_counts is None else full_counts.build(),
                lead,
                nrows,
                ncols,
                device,
                all_full=all_full,
            )
        )
        shapes = ([*lead, nrows], [*lead, nrows, ncols]) * 2
        return (
            [
                _Tensor(
                    _contiguous_fake(shape, i32, device),
                    lambda i=i: dense()[i],
                )
                for i, shape in enumerate(shapes)
            ],
            _DENSE_BLOCK_SIZE,
            _DENSE_BLOCK_SIZE,
        )
    planned = [_plan_input(num_blocks), _plan_input(indices)]
    if full_num_blocks is None or full_indices is None:
        # A BlockMask without full blocks: every block is partial.
        for partial in planned.copy():
            size = list(partial.fake.size())
            planned.append(
                _Tensor(
                    _contiguous_fake(size, i32, device),
                    lambda size=size: _realized(
                        full(size, 0, dtype=i32, device=device)
                    ),
                )
            )
    else:
        planned += [_plan_input(full_num_blocks), _plan_input(full_indices)]
    return planned, sparse_q, sparse_kv


# -- the mods ------------------------------------------------------------------


class _ModTracer(torch.fx.Tracer):
    """The tracer FX rebuilds a prepared mod with when it is unpickled (as the
    Helion autotuner's benchmark subprocess loads the kernel args): it re-traces
    the mod's code, where ``hl.load`` must stay a call, not run on proxies."""

    def __init__(self) -> None:
        super().__init__(autowrap_functions=(hl.load,))


def _prepare_mod(
    gm: torch.fx.GraphModule,
    num_coords: int,
    captured_ints: Mapping[int, int],
    coord_sizes: Sequence[int],
) -> torch.fx.GraphModule:
    """A copy of a score_mod / mask_mod / score_mod_bwd subgraph a Helion kernel
    can inline: the placeholders after the ``num_coords`` leading ones are the
    captured values; the integers (``captured_ints``, by position) become
    constants and the tensors stay inputs, which a Helion device loop reads
    through ``hl.load``. ``coord_sizes`` are the sizes the coordinates (b,
    h, q_idx, kv_idx) index."""
    gm = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    graph = gm.graph
    captured = list(graph.find_nodes(op="placeholder"))[num_coords:]
    constants = {captured[i]: value for i, value in captured_ints.items()}
    if constants:
        for node in graph.nodes:
            node.args = torch.fx.node.map_arg(node.args, lambda n: constants.get(n, n))
            node.kwargs = torch.fx.node.map_arg(
                node.kwargs, lambda n: constants.get(n, n)
            )
        for node in constants:
            graph.erase_node(node)
    # The subgraphs were traced at the input dtype; the kernels evaluate them
    # in fp32, as eager does, so a cast among fp16 / bf16 and fp32 of a value
    # computed from the score (or the gradient), which the kernels hold in
    # fp32, only adds rounding (and in the joint graph it splits the region
    # the fused backward matches): it is dropped. A cast to fp16 / bf16 of
    # anything else becomes one to fp32: Inductor computes fp16 / bf16 in
    # fp32 too, unless emulate_precision_casts asks to keep the casts. Other
    # casts (to fp32 of anything else, to fp8 or fp64) stay; a captured fp16
    # / bf16 tensor's load is computed in fp32 (below), so ``bias[h].float()``
    # keeps its fp32 range whether or not the cast is a no-op.
    if not inductor_config.emulate_precision_casts:
        upcast = (torch.float16, torch.bfloat16, torch.float32)
        derived = OrderedSet(
            p
            for p in list(graph.find_nodes(op="placeholder"))[:num_coords]
            if isinstance(v := p.meta.get("val"), torch.Tensor)
            and v.dtype.is_floating_point
        )
        for node in graph.nodes:
            if any(n in derived for n in node.all_input_nodes):
                derived.add(node)
        for node in graph.find_nodes(
            op="call_function", target=prims.convert_element_type.default
        ):
            src, dtype = node.args[:2]
            src_val = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
            if not isinstance(src_val, torch.Tensor) or dtype not in upcast:
                continue
            if src in derived and src_val.dtype in upcast:
                node.replace_all_uses_with(src)
                graph.erase_node(node)
            elif dtype != torch.float32:
                # A cast to fp16 / bf16 of anything else (``(q_idx -
                # kv_idx).to(score.dtype)``, ``bias[h].to(score.dtype)`` of an
                # fp16 table, an fp8 round trip's way back) computes in fp32,
                # as Inductor does: exact for the integers and fp8, and
                # without fp16's range for the table.
                node.args = (src, torch.float32, *node.args[2:])
    if graph.find_nodes(op="call_function", target=inline_asm_elementwise):
        raise Declined("NYI: inline_asm_elementwise in score_mod or mask_mod")
    placeholders = list(graph.find_nodes(op="placeholder"))
    # The coordinates (b, h, q_idx, kv_idx), each below its size.
    coords = [
        p
        for p in placeholders[:num_coords]
        if isinstance(v := p.meta.get("val"), torch.Tensor)
        and not v.dtype.is_floating_point
    ]
    bounds = {p: size - 1 for p, size in zip(coords, coord_sizes, strict=True)}
    buffers = OrderedSet(placeholders[num_coords:])
    for node in list(graph.nodes):
        if node.target is aten.index.Tensor and node.args[0] in buffers:
            indices = list(node.args[1])
            if any(i is None for i in indices):
                raise Declined(
                    "NYI: score_mod or mask_mod indexes a captured tensor with a slice"
                )
            # A negative index wraps, as in eager and Inductor's Triton
            # lowering (the kernels' gathers read zero out of range).
            sizes = node.args[0].meta["val"].shape
            with graph.inserting_before(node):
                indices = [
                    i
                    if _non_negative(i, bounds)
                    else graph.call_function(
                        aten.where.self,
                        (
                            graph.call_function(aten.lt.Scalar, (i, 0)),
                            graph.call_function(aten.add.Tensor, (i, int(size))),
                            i,
                        ),
                    )
                    for i, size in zip(indices, sizes, strict=False)
                ]
            node.target = hl.load
            node.args = (node.args[0], indices)
            if node.args[0].meta["val"].dtype in (torch.float16, torch.bfloat16):
                if inductor_config.emulate_precision_casts:
                    # Inductor then rounds what is computed from the load to
                    # its dtype, as eager does; the kernels would compute it
                    # in fp32.
                    raise Declined(
                        "NYI: emulate_precision_casts with a float16 / bfloat16 "
                        "tensor captured by score_mod or mask_mod"
                    )
                # Computed in fp32, as Inductor computes fp16 / bf16 loads
                # (and as the cast rules above assume).
                with graph.inserting_after(node):
                    upcast_load = graph.call_function(
                        prims.convert_element_type.default, (node, torch.float32)
                    )
                node.replace_all_uses_with(
                    upcast_load,
                    delete_user_cb=lambda user, load=upcast_load: user is not load,
                )
    for buf in buffers:
        for user in buf.users:
            if user.target is not hl.load:
                raise Declined(
                    "NYI: score_mod or mask_mod uses a captured tensor other than "
                    "by indexing it with the coordinates"
                )
    gm.recompile()
    gm._tracer_cls = _ModTracer
    return gm


def _non_negative(index: object, bounds: Mapping[torch.fx.Node, int]) -> bool:
    """Whether a mod's gather index is non-negative by construction:
    ``integer_bounds``' interval analysis (each intermediate fitting its
    dtype, so a sum or product that may wrap negative is not) proves it at
    least 0, with each coordinate in ``bounds`` between 0 and its largest
    value."""
    interval = integer_bounds(
        index, captured_bounds={node: (0, high) for node, high in bounds.items()}
    )
    return interval is not None and interval[0] >= 0


def _score_mod_bwd(
    joint_gm: torch.fx.GraphModule,
    captured_ints: Mapping[int, int],
    coord_sizes: Sequence[int],
) -> torch.fx.GraphModule:
    """Helion's ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx, *buffers)`` from
    the joint graph ``(score, b, h, m, n, grad_score_mod, *buffers) ->
    (grad_score, grad_b, ..., *buffer_grads)``."""
    gm = torch.fx.GraphModule(joint_gm, copy.deepcopy(joint_gm.graph))
    graph = gm.graph
    placeholders = list(graph.find_nodes(op="placeholder"))
    placeholders[0].prepend(placeholders[5])
    (output,) = graph.find_nodes(op="output")
    outs = output.args[0]
    if any(o is not None for o in outs[5:]):
        raise Declined("NYI: gradients of tensors captured by score_mod")
    output.args = (outs[0],)
    graph.eliminate_dead_code()
    gm.recompile()
    return _prepare_mod(gm, 6, captured_ints, coord_sizes)


# -- BlockMask tables ------------------------------------------------------------


def _realized(node: TensorBox) -> TensorBox:
    node.realize()
    return node


def _masks_nothing(graph_module: torch.fx.GraphModule) -> bool:
    """Whether a mask_mod graph returns a constant True (the frontend's
    ``noop_mask``)."""
    (output,) = graph_module.graph.find_nodes(op="output")
    value = output.args[0]
    return (
        isinstance(value, torch.fx.Node)
        and value.target is aten.full.default
        and value.args[1] is True
    )


def _dense_tables(
    counts: TensorBox,
    full_counts: TensorBox | None,
    lead: Sequence[int],
    nrows: int,
    ncols: int,
    device: torch.device,
    *,
    all_full: bool,
) -> list[TensorBox]:
    """(num_blocks, indices, full_num_blocks, full_indices) over ``nrows`` x
    ``ncols`` ``_DENSE_BLOCK_SIZE`` blocks of a BlockMask of one block per
    (batch, head) (``lead``), whose partial / full counts (``[*lead, 1]``,
    0 or 1) are ``counts`` / ``full_counts``: a row lists every block as
    partial where that block is partial, as full where it is full (both as
    full under a mask_mod that masks nothing), and none where it is
    neither."""
    i32 = torch.int32
    partial: TensorBox | None = to_dtype(counts, i32)
    whole = None if full_counts is None else to_dtype(full_counts, i32)
    if all_full:
        whole = partial if whole is None else lowerings[aten.add.Tensor](partial, whole)
        partial = None

    def num_blocks(count: TensorBox | None) -> TensorBox:
        if count is None:
            return _realized(full([*lead, nrows], 0, dtype=i32, device=device))
        return _realized(
            lowerings[aten.clone.default](
                lowerings[aten.expand.default](
                    lowerings[aten.mul.Tensor](count, ncols), [*lead, nrows]
                )
            )
        )

    def indices() -> TensorBox:
        iota = lowerings[prims.iota.default](
            ncols, start=0, step=1, dtype=i32, device=device, requires_grad=False
        )
        return _realized(
            lowerings[aten.clone.default](
                lowerings[aten.expand.default](
                    lowerings[aten.view.default](iota, [1, 1, 1, ncols]),
                    [*lead, nrows, ncols],
                )
            )
        )

    return [num_blocks(partial), indices(), num_blocks(whole), indices()]


# -- kernel selection and lowering ------------------------------------------------


def _is_fp8(dtype: torch.dtype) -> bool:
    """Whether ``dtype`` is an 8-bit float (float8_e4m3fn, float8_e5m2,
    float8_e8m0fnu, ...)."""
    return dtype.is_floating_point and dtype.itemsize == 1


def _example(node: TensorBox) -> torch.Tensor:
    """A real input like ``node`` for autotuning, in its layout (strides, zero
    strides of a broadcast captured tensor included, and storage offset):
    random floats and bools, zero integers (a captured integer tensor may
    index another)."""
    realized = TemplateBuffer.realize_template_input(node)
    size = [int(s) for s in realized.get_size()]
    stride = [int(s) for s in realized.get_stride()]
    offset = int(realized.get_layout().offset)
    dtype = realized.get_dtype()
    storage = torch.empty(
        compute_required_storage_length(size, stride, offset),
        dtype=dtype,
        device=realized.get_device(),
    )
    if dtype.is_floating_point:
        storage.normal_()
    elif dtype == torch.bool:
        storage.random_(2)
    else:
        storage.zero_()
    return storage.as_strided(size, stride, offset)


def _dense_example_tables(
    tables: Sequence[TensorBox], *, partial: bool
) -> list[torch.Tensor]:
    """Real BlockMask tables shaped like ``tables`` listing every block, for
    autotuning (random example inputs would index out of range): as partial
    blocks, which apply the mask_mod, when ``partial``, else as full."""
    examples = [_example(t) for t in tables]
    listed, unlisted = (0, 2) if partial else (2, 0)
    examples[unlisted].zero_()
    examples[unlisted + 1].zero_()
    indices = examples[listed + 1]
    ncols = indices.size(-1)
    examples[listed].fill_(ncols)
    # Written through one element of each broadcast (zero-stride) dimension,
    # which copy_ cannot write into.
    _distinct(indices).copy_(
        torch.arange(ncols, dtype=indices.dtype, device=indices.device)
    )
    return examples


def _distinct(t: torch.Tensor) -> torch.Tensor:
    """``t`` with its zero-stride dimensions narrowed to one element: a view
    whose elements are all distinct memory."""
    return t[tuple(slice(0, 1) if s == 0 else slice(None) for s in t.stride())]


def _listed_tables(
    rows: int, cols: int, device: torch.device, *, partial: bool
) -> list[torch.Tensor]:
    """(num_blocks, indices, full_num_blocks, full_indices) of a ``rows`` x
    ``cols`` block grid listing every block, as partial when ``partial``,
    else as full."""
    i32 = torch.int32
    listed = [
        torch.full([1, 1, rows], cols, dtype=i32, device=device),
        torch.arange(cols, dtype=i32, device=device).expand(1, 1, rows, cols).clone(),
    ]
    unlisted = [
        torch.zeros([1, 1, rows], dtype=i32, device=device),
        torch.zeros([1, 1, rows, cols], dtype=i32, device=device),
    ]
    return listed + unlisted if partial else unlisted + listed


def _container_specs(
    tensors: dict[str, _Tensor],
    constant_args: dict[str, object],
    score_mod_buffers: Sequence[_Tensor],
    mask_mod_buffers: Sequence[_Tensor],
) -> None:
    """Add the captured tensors as the kernels' tuple params, flattened like any
    Helion container arg (``name.i`` tensor args plus a treespec)."""
    specs = {}
    for name, bufs in (
        ("score_mod_buffers", score_mod_buffers),
        ("mask_mod_buffers", mask_mod_buffers),
    ):
        if not bufs:
            constant_args[name] = ()
            continue
        for i, buf in enumerate(bufs):
            tensors[f"{name}.{i}"] = buf
        specs[name] = pytree.treespec_dumps(
            pytree.tree_flatten(tuple(b.fake for b in bufs))[1]
        )
    if specs:
        constant_args["__container_specs"] = specs


def _kernel_args(
    kernel: helion.Kernel,
    tensor_args: Mapping[str, object],
    constant_args: Mapping[str, object],
) -> tuple[object, ...]:
    all_args = {**constant_args, **tensor_args}
    _rebuild_container_args(all_args)
    return tuple(all_args[n] for n in kernel.signature.parameters)


def _fused_config_spec(
    direction: str, fake_args: tuple[object, ...]
) -> tuple[ConfigSpec, dict[str, Any]]:
    """The config spec and HOP output spec of the ``direction`` kernel bound
    to ``fake_args``, which must lower to Helion's fused CuTe flash kernel
    (one binding for both)."""
    kernel = _kernels.get_kernel(direction)
    try:
        bound, output_spec = _infer_output_spec(kernel, fake_args)
    except helion.exc.InternalError as e:
        # A Helion bug, not an unsupported call: keep Triton's kernel, but
        # say so where it is seen.
        log.warning(
            "Helion failed to compile this flex_attention %s (keeping "
            "Inductor's kernel): %s",
            direction,
            e,
            exc_info=True,
        )
        raise Declined(
            f"Helion failed to compile this flex_attention {direction}: {e}"
        ) from e
    except helion.exc.BaseError as e:
        raise Declined(
            f"Helion cannot compile this flex_attention {direction}: {e}"
        ) from e
    spec = bound.config_spec
    # The fused matchers' decision, as the config spec records it.
    if direction == "forward":
        fused = spec.cute_flash_search_enabled
    else:
        fused = spec.cute_flash_bwd_search_enabled
    if not fused:
        matcher = "flex_match" if direction == "forward" else "flex_bwd_match"
        hint = ""
        if direction == "backward" and inductor_config.emulate_precision_casts:
            hint = (
                "; NYI: emulate_precision_casts=True keeps the float casts in "
                "score_mod's gradient, which the fused backward does not fit"
            )
        # TODO(yushangdi): put the matcher's decline reason (logged at DEBUG as
        # "<label> declined: <reason>") in this message.
        raise Declined(
            f"it does not fit Helion's fused CuTe flash kernel; "
            f"HELION_LOGS=+helion._compiler.cute.{matcher} logs why{hint}"
        )
    return spec, output_spec


def _select_kernel(
    direction: str,
    options: Options,
    config_spec: ConfigSpec,
    tuning_args: Callable[..., tuple[object, ...]],
) -> helion.Kernel:
    """The kernel to lower, fixed to the configured config for ``direction``
    or else to the best config autotuning finds (Helion's default config at
    ``HELION_AUTOTUNE_EFFORT=none``, or when the examples give NaN at the
    default config: NaN outputs compare equal in the accuracy check, so no
    candidate could be validated). A config in Helion's autotune cache is
    read first, before anything compiles; ``tuning_args(prepare=False)``
    skips the example fix-ups (the backward's example forward), which the
    cache key does not read, and a miss then prepares the same examples
    rather than building a second set."""
    config = options.config(direction)
    if config is not None:
        try:
            normalized = config_spec.normalized_config(config)
        except (TypeError, helion.exc.InvalidConfig) as e:
            raise ValueError(
                f"{direction}_config (helion.flex_attention.configure or "
                "BACKEND_OPTIONS) does not apply to this flex_attention "
                f"{direction}: {e}"
            ) from e
        if not _attention_config_envelope_ok(normalized):
            # Outside the fused kernel's envelope the forward would quietly
            # run Helion's generic kernel, and the backward fail to compile.
            raise ValueError(
                f"{direction}_config (helion.flex_attention.configure or "
                "BACKEND_OPTIONS) does not apply to this flex_attention "
                f"{direction}: the fused "
                "kernel takes only pid_type='flat', l2_groupings of 1, "
                "identity loop_orders, num_threads of 0, cute_vector_widths "
                "of 1 and no epilogue_subtile"
            )
        return _kernels.get_kernel(direction, config=dict(config))
    kernel = _kernels.get_kernel(direction, tolerance=options.autotune_tolerance)
    if kernel._fixed_config_kind() is not None:
        return kernel
    if (cached := _cached_config(kernel, tuning_args(prepare=False))) is not None:
        return _kernels.get_kernel(direction, config=dict(cached))
    args = tuning_args()
    if os.environ.get("HELION_ASSERT_CACHE_HIT") == "1":
        # A miss must raise CacheAssertionError: the autotuner raises it (with
        # its key diagnostics) before anything runs, which the default-config
        # probe below would otherwise run, or skip on degenerate examples.
        tuned = kernel.autotune(args, force=False)
        return _kernels.get_kernel(direction, config=dict(tuned.config))
    default = _kernels.get_kernel(direction, config=dict(config_spec.default_config()))
    if (degenerate := _degenerate(default(*args))) is not None:
        log.info(
            "flex_attention %s keeps Helion's default config: its autotuning "
            "examples %s, so no candidate config could be validated",
            direction,
            degenerate,
        )
        return default
    tuned = kernel.autotune(args, force=False)
    return _kernels.get_kernel(direction, config=dict(tuned.config))


def _degenerate(outputs: object) -> str | None:
    """How the outputs of the autotuning examples keep the accuracy check
    from telling configs apart, or None. Captured tensors' random examples
    outside a score_mod's domain (torch.log of a negative bias) can make
    every row NaN, which the check counts as a match; a mask_mod reading a
    captured integer tensor (kv_idx < lengths[b]) on its zero example can
    mask every entry, leaving nothing to compare (zero outputs and
    gradients, a logsumexp of -inf)."""
    tensors = [t for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)]
    if any(t.isnan().any() for t in tensors):
        return "give NaN"
    if not any((t.isfinite() & (t != 0)).any() for t in tensors):
        return "mask every entry"
    return None


def _cached_config(
    kernel: helion.Kernel, args: tuple[object, ...]
) -> helion.Config | None:
    """The config Helion's autotune cache holds for ``kernel`` on ``args``
    (its key reads their shapes and dtypes, not their values), read without
    compiling, as autotuning reads it; None on a miss, where the cache is not
    read, or for an autotuner without a cache."""
    bound = kernel.bind(args)
    tuner = bound.settings.autotuner_fn(bound, args)
    if not isinstance(tuner, AutotuneCacheBase):
        return None
    return tuner.cached_config(skip_cache=kernel.settings.force_autotune)


def check_compiled(direction: str) -> None:
    """Raise :class:`Declined` under Helion's reference mode
    (``HELION_INTERPRET=1``), where the ``direction`` kernel runs eagerly and
    binds no host function to lower."""
    if is_ref_mode_enabled(_kernels.get_kernel(direction).settings):
        raise Declined("NYI: Helion's reference mode (HELION_INTERPRET=1)")


def check_tuning_allowed(direction: str, options: Options) -> None:
    """Raise :class:`Declined` when the ``direction`` kernel would autotune (no
    configured config, and its settings search) where autotuning is
    disallowed (``HELION_DISALLOW_AUTOTUNING=1``, a MAST job): Helion's
    autotuner raises there before reading any cache, which would fail the
    compile after the choice replaced Triton's."""
    config = options.config(direction)
    if config is not None or _kernels.get_kernel(direction)._fixed_config_kind():
        return
    if (reason := autotuning_disallowed_reason()) is not None:
        raise Declined(
            f"autotuning is disallowed ({reason}) and no {direction} config "
            "is configured (helion.flex_attention.configure or BACKEND_OPTIONS)"
        )


# The forward kernel's BlockMask table parameters.
_FORWARD_TABLE_NAMES = (
    "block_mask_kv_num_blocks",
    "block_mask_kv_indices",
    "block_mask_full_kv_num_blocks",
    "block_mask_full_kv_indices",
)


@dataclasses.dataclass
class _Call:
    """One kernel call: its tensor and constant arguments."""

    direction: str
    tensors: dict[str, _Tensor]
    constant_args: dict[str, object]
    table_names: tuple[str, ...]
    # Whether the mask_mod can mask an entry: the autotuning examples then
    # list their blocks as partial, so they apply it, as the call does (a
    # score_mod finite only where the mask keeps an entry gives NaN outside
    # it, and NaN outputs pass the accuracy check).
    masks: bool
    # Fixes up the random example tensors autotuning runs on, in place.
    prepare_examples: Callable[[dict[str, object]], None] | None = None

    def lower(self, options: Options) -> tuple[TensorBox, ...]:
        """Check that the kernel runs this call (or raise :class:`Declined`,
        before adding any IR node), then lower it under ``options``."""
        signature = _kernels.get_kernel(self.direction)
        fake_args = _kernel_args(
            signature,
            {n: t.fake for n, t in self.tensors.items()},
            self.constant_args,
        )
        config_spec, output_spec = _fused_config_spec(self.direction, fake_args)

        tensor_args = {n: t.build() for n, t in self.tensors.items()}
        tables = [tensor_args[n] for n in self.table_names]

        examples: dict[str, object] = {}
        prepared = False

        def tuning_args(*, prepare: bool = True) -> tuple[object, ...]:
            # One set of examples: the cache lookup reads it unprepared,
            # and a miss prepares the same tensors once.
            nonlocal prepared
            if not examples:
                tables_by_name = dict(
                    zip(
                        self.table_names,
                        _dense_example_tables(tables, partial=self.masks),
                        strict=True,
                    )
                )
                examples.update(
                    (n, tables_by_name[n] if n in tables_by_name else _example(t))
                    for n, t in tensor_args.items()
                )
            if prepare and not prepared and self.prepare_examples is not None:
                self.prepare_examples(examples)
                prepared = True
            return _kernel_args(signature, examples, self.constant_args)

        kernel = _select_kernel(self.direction, options, config_spec, tuning_args)
        return lower_helion_kernel(
            kernel_idx=helion_kernel_side_table.add_kernel(kernel),
            constant_args=self.constant_args,
            tensor_args=tensor_args,
            output_spec=output_spec,
        )


@dataclasses.dataclass
class _Planned:
    """What the forward and backward plans share: the planned operands
    (query, key, value, then the extra ones in order), the BlockMask tables
    and the constant arguments both kernels take."""

    operands: list[_Tensor]
    device: torch.device
    tables: list[_Tensor]
    block_m: int
    block_n: int
    score_mod_ints: dict[int, int]
    coord_sizes: tuple[int, ...]
    constant_args: dict[str, object]
    score_mod_buffers: list[_Tensor]
    mask_mod_buffers: list[_Tensor]
    masks: bool


def _plan_shared(
    direction: str,
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    extra_operands: Sequence[TensorBox],
    score_mod_graph: torch.fx.GraphModule,
    block_mask: FlexBlockMask,
    scale: object,
    score_mod_other_buffers: Sequence[object],
    mask_mod_other_buffers: Sequence[object],
    options: Options,
) -> _Planned:
    """Check what both directions decline (raising :class:`Declined`) before
    realizing any input, then plan the operands, the captured buffers, the
    ``direction``'s BlockMask tables (the forward's list KV blocks per Q
    block; the backward's Q-side tables list Q blocks per KV block, so their
    rows are KV blocks) and the constant arguments both kernels take."""
    _check_supported(query, key, value, scale)
    check_compiled(direction)
    check_tuning_allowed(direction, options)
    if direction == "backward" and torch.are_deterministic_algorithms_enabled():
        raise Declined(
            "deterministic algorithms are on, and the fused backward accumulates "
            "grad_query with atomics"
        )
    kv_tables = [
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
    ]
    q_tables = [
        block_mask.q_num_blocks,
        block_mask.q_indices,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
    ]
    mask_mod = block_mask.mask_mod
    operands = [_plan_operand(t) for t in (query, key, value, *extra_operands)]
    q, k = operands[0], operands[1]
    device = query.get_device_or_error()
    score_mod_buffers, score_mod_ints = _plan_captured(score_mod_other_buffers, device)
    mask_mod_buffers, mask_mod_ints = _plan_captured(mask_mod_other_buffers, device)
    m, n = q.fake.size(2), k.fake.size(2)
    side, rows, cols = (kv_tables, m, n) if direction == "forward" else (q_tables, n, m)
    tables, block_m, block_n = _plan_block_tables(
        side,
        block_mask.q_block_size,
        block_mask.kv_block_size,
        mask_mod,
        rows,
        cols,
        device,
    )
    coord_sizes = (*q.fake.size()[:3], n)
    constant_args: dict[str, object] = {
        "score_mod": _prepare_mod(score_mod_graph, 5, score_mod_ints, coord_sizes),
        "block_mask_mask_mod": _prepare_mod(mask_mod, 4, mask_mod_ints, coord_sizes),
        "block_mask_m": block_m,
        "block_mask_n": block_n,
        "scale": float(scale),  # pyrefly: ignore[bad-argument-type]
    }
    return _Planned(
        operands=operands,
        device=device,
        tables=tables,
        block_m=block_m,
        block_n=block_n,
        score_mod_ints=score_mod_ints,
        coord_sizes=coord_sizes,
        constant_args=constant_args,
        score_mod_buffers=score_mod_buffers,
        mask_mod_buffers=mask_mod_buffers,
        masks=not _masks_nothing(mask_mod),
    )


def lower_forward(
    call: FlexAttentionCall, options: Options
) -> tuple[TensorBox, TensorBox, TensorBox | None]:
    """Lower a flex_attention forward to the Helion forward kernel, or raise
    :class:`Declined` before adding any IR node. Returns (out, logsumexp in
    base 2, max_scores in base 2); max_scores is None unless the call reads it
    (``call.output_max``)."""
    planned = _plan_shared(
        "forward",
        call.query,
        call.key,
        call.value,
        (),
        call.score_mod,
        call.block_mask,
        call.scale,
        call.score_mod_buffers,
        call.mask_mod_buffers,
        options,
    )
    q, k, v = planned.operands
    table_names = _FORWARD_TABLE_NAMES
    tensors = {
        "query": q,
        "key": k,
        "value": v,
        **dict(zip(table_names, planned.tables, strict=True)),
    }
    # A row-max store keeps the fused plan off its dense bodies, so the
    # kernel stores it only on request.
    constant_args = {**planned.constant_args, "output_max": call.output_max}
    _container_specs(
        tensors, constant_args, planned.score_mod_buffers, planned.mask_mod_buffers
    )
    out, lse, max_scores = _Call(
        "forward", tensors, constant_args, table_names, planned.masks
    ).lower(options)
    return out, lse, max_scores if call.output_max else None


def lower_backward(
    call: FlexAttentionBackwardCall, options: Options
) -> tuple[TensorBox, TensorBox, TensorBox]:
    """Lower a flex_attention backward to the Helion backward kernel, or raise
    :class:`Declined` before adding any IR node. Returns (grad_query,
    grad_key, grad_value)."""
    query = call.query
    planned = _plan_shared(
        "backward",
        query,
        call.key,
        call.value,
        (call.grad_out,),
        call.fw_graph,
        call.block_mask,
        call.scale,
        call.score_mod_buffers,
        call.mask_mod_buffers,
        options,
    )
    device = planned.device
    block_m, block_n, masks = planned.block_m, planned.block_n, planned.masks
    q, k, v, do = planned.operands
    o, lse = (_plan_input(t) for t in (call.out, call.logsumexp))
    dlse = None if call.grad_logsumexp is None else _plan_input(call.grad_logsumexp)
    f32 = torch.float32

    @functools.cache
    def row_terms() -> tuple[TensorBox, TensorBox]:
        """The softmax-backward row terms: delta = rowsum(out * grad_out), and
        the forward's base-2 logsumexp with no-visible-key rows mapped to 0."""
        delta = lowerings[aten.sum.dim_IntList](
            lowerings[aten.mul.Tensor](
                to_dtype(o.build(), f32), to_dtype(do.build(), f32)
            ),
            [-1],
        )
        if dlse is not None:
            # d lse / d score = P, so the logsumexp's gradient adds
            # P * grad_lse to dS = P * (dP - delta): fold it into delta. The
            # forward's logsumexp is base 2, so its gradient in natural units
            # is grad_logsumexp * log2(e).
            delta = lowerings[aten.sub.Tensor](
                delta, lowerings[aten.mul.Tensor](dlse.build(), 1.0 / math.log(2.0))
            )
        logsumexp = lse.build()
        lse2 = lowerings[aten.where.self](
            lowerings[aten.eq.Scalar](logsumexp, float("-inf")), 0.0, logsumexp
        )
        # The fused backward reads these row terms as contiguous; left
        # flexible, Inductor may pad their strides (a sequence length that is
        # not a multiple of 8), which the kernel declines.
        return (
            cast("TensorBox", ExternKernel.require_contiguous(lse2)),
            cast("TensorBox", ExternKernel.require_contiguous(delta)),
        )

    rows = list(lse.fake.size())
    table_names = (
        "block_mask_q_num_blocks",
        "block_mask_q_indices",
        "block_mask_full_q_num_blocks",
        "block_mask_full_q_indices",
    )
    tensors = {
        "query": q,
        "key": k,
        "value": v,
        "grad_out": do,
        "lse2": _Tensor(_contiguous_fake(rows, f32, device), lambda: row_terms()[0]),
        "delta": _Tensor(_contiguous_fake(rows, f32, device), lambda: row_terms()[1]),
        **dict(zip(table_names, planned.tables, strict=True)),
    }
    constant_args = {
        **planned.constant_args,
        "score_mod_bwd": _score_mod_bwd(
            call.joint_graph, planned.score_mod_ints, planned.coord_sizes
        ),
    }
    _container_specs(
        tensors, constant_args, planned.score_mod_buffers, planned.mask_mod_buffers
    )

    def example_row_terms(examples: dict[str, object]) -> None:
        """Replace the random lse2 / delta examples with the row terms of the
        examples' forward, run at Helion's default config over every block
        (listed as the backward's example tables list them). The kernel
        recomputes P = exp2(score - lse2), which a random lse2 lets overflow,
        and a candidate config is checked against the default config's
        outputs, so the inf or NaN gradients of both would pass the check."""
        query, key, grad_out = (
            cast("torch.Tensor", examples[n]) for n in ("query", "key", "grad_out")
        )
        forward_tables = _listed_tables(
            -(-query.size(2) // block_m),
            -(-key.size(2) // block_n),
            query.device,
            partial=masks,
        )
        backward_only = {"grad_out", "lse2", "delta", *table_names}
        forward = _kernels.get_kernel("forward")
        forward_args = _kernel_args(
            forward,
            {
                **{n: t for n, t in examples.items() if n not in backward_only},
                **dict(zip(_FORWARD_TABLE_NAMES, forward_tables, strict=True)),
            },
            {
                **{n: a for n, a in constant_args.items() if n != "score_mod_bwd"},
                "output_max": False,
            },
        )
        default = forward._bind_isolated(forward_args).config_spec.default_config()
        out, lse2, _ = _kernels.get_kernel("forward", config=dict(default))(
            *forward_args
        )
        # Rows with no visible key map to 0, as in row_terms.
        examples["lse2"] = torch.where(lse2 == float("-inf"), 0.0, lse2).contiguous()
        # (Without grad_logsumexp's term: an example of it is noise as well.)
        examples["delta"] = (out.float() * grad_out.float()).sum(-1).contiguous()

    dq, dk, dv = _Call(
        "backward", tensors, constant_args, table_names, masks, example_row_terms
    ).lower(options)
    return to_dtype(dq, query.get_dtype()), dk, dv
