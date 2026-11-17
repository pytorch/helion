# ruff: noqa: ANN001, ANN202, ANN401
"""Gluon lowerings for Helion's ``gluon`` backend: a gluon kernel is Helion's
Triton lowering traced as a native Gluon kernel (``tl_namespace`` as ``tl``,
layouts from ``gluon_layout``), with every tcgen05 dot from ``codegen_tmem_dot``.
"""

from __future__ import annotations

import ast
import contextvars
import dataclasses
import functools
import hashlib
import threading
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonJITFunction
from triton.experimental.gluon.language._core import builtin as gl_builtin
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryScalesLayout
from triton.experimental.gluon.language.nvidia.blackwell import allocate_tensor_memory
from triton.experimental.gluon.language.nvidia.blackwell import tcgen05_commit
from triton.experimental.gluon.language.nvidia.blackwell import tcgen05_copy
from triton.experimental.gluon.language.nvidia.blackwell import tcgen05_mma
from triton.experimental.gluon.language.nvidia.blackwell import tcgen05_mma_scaled
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.hopper import tma
import triton.language as tl
from triton.runtime.jit import JITFunction

# pyrefly: ignore [missing-import]
from triton.tools.triton_to_gluon_translator import blackwell_helpers

# pyrefly: ignore [missing-import]
from triton.tools.triton_to_gluon_translator import common_helpers

# pyrefly: ignore [missing-import]
from triton.tools.triton_to_gluon_translator import nvidia_helpers

from ..._compat import min_dot_size
from ..._compat import torch_dtype_to_tl
from ...language import memory_ops
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _for_loop_step
from ...language._tracing_ops import _new_var
from ...language.creation_ops import full
from ...language.matmul_ops import dot
from ...language.matmul_ops import dot_scaled
from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..cross_loop_codegen import _is_load
from ..cross_loop_codegen import _is_tl_call
from ..device_function import DeviceFunction
from ..device_ir import ForLoopGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import control_flow_parent_entries
from ..device_ir_analysis import matmul_operand_positions
from ..indexing_strategy import _resolve_configured_extent
from ..matmul_utils import _pad_tensor
from . import gluon_layout

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import GraphInterpreter

# The generated code's name for this module (``GluonBackend.library_imports``).
ALIAS = "_helion_gluon"


_COMPILING_NATIVE = contextvars.ContextVar("_helion_compiling_native", default=False)
# ``gluon_layout``'s trace state is process-wide: one native compile at a time
# (a thread compiling another kernel sees Triton's code generator unchanged).
_NATIVE_COMPILE_LOCK = threading.RLock()


class NativeJITFunction(GluonJITFunction):
    """A native Gluon kernel: Helion's Triton lowering traced with ``gluon_layout``,
    launched with ``maxnreg`` registers a thread (None: Triton's)."""

    maxnreg: int | None = None

    def _pack_args(  # pyrefly: ignore [bad-override]
        self,
        backend: Any,
        kwargs: dict[str, Any],
        bound_args: Any,
        specialization: Any,
        options: Any,
    ) -> Any:
        # Launches and the autotuner's precompiler both take compile options here.
        if self.maxnreg is not None:
            kwargs = {"maxnreg": self.maxnreg, **kwargs}
        return super()._pack_args(backend, kwargs, bound_args, specialization, options)

    @property
    def cache_key(self) -> str:
        # Triton keys compiled kernels on source; the layouts come from here too.
        return hashlib.sha256(
            (super().cache_key + _native_source_hash()).encode()
        ).hexdigest()

    def create_binder(self) -> Any:
        binder = super().create_binder()
        compile_source = self.compile

        def compile_native(*args: Any, **kwargs: Any) -> Any:
            with _NATIVE_COMPILE_LOCK:
                token = _COMPILING_NATIVE.set(True)
                try:
                    return compile_source(*args, **kwargs)
                finally:
                    _COMPILING_NATIVE.reset(token)

        self.compile = compile_native  # pyrefly: ignore [bad-override-param-name]
        return binder


def native_kernel(fn: Any = None, *, maxnreg: int | None = None, **options: Any) -> Any:
    """Decorator of a generated gluon kernel (``maxnreg``: ``NativeJITFunction``'s;
    ``options``: ``triton.jit``'s)."""
    if fn is None:
        return functools.partial(native_kernel, maxnreg=maxnreg, **options)
    _install_native()
    kernel = NativeJITFunction(fn, **options)
    kernel.maxnreg = maxnreg
    return kernel


@functools.cache
def _native_source_hash() -> str:
    sources = (gluon_layout.__file__, __file__)
    return hashlib.sha256(b"".join(open(f, "rb").read() for f in sources)).hexdigest()


@functools.cache
def _install_native() -> None:
    gluon_layout.install(lambda generator: _COMPILING_NATIVE.get())


vector_layout = gluon_layout.vector_layout
_shared = gluon_layout._shared


# ---- software pipelining (``gluon_pipeline.software_pipeline``) -------------------------


def _pipe_layout(ptr: Any, sem: Any) -> Any:
    return sem._coalesced(ptr) if gluon_layout._is_auto(ptr) else ptr.type.layout


def _pipe_in(x: Any, shape: Any, layout: Any, sem: Any) -> Any:
    """``x`` broadcast to ``shape`` in ``layout``."""
    x = sem.settle(sem.broadcast_impl_shape(x, shape), layout)
    return x if x.type.layout == layout else sem.convert_layout(x, layout)


def _pipe_view(ptr: Any, mask: Any, shape: Any, sem: Any) -> tuple[Any, Any]:
    """``ptr`` and ``mask`` in a slot's ``shape`` (an MMA operand's: ``_mma_shape``;
    a scalar's one element, as Triton's pipeliner copies it)."""
    if not ptr.type.is_block():
        ptr = sem.splat(ptr, [1])
    if list(ptr.type.shape) == list(shape):
        return ptr, mask
    if mask is not None:
        mask = sem.reshape(sem.broadcast_impl_shape(mask, ptr.type.shape), shape, False)
    return sem.reshape(ptr, shape, False), mask


@gl_builtin
def pipe_alloc(ptr, mask, S, OPERAND=None, STAGED=False, _semantic: Any = None):
    """The ``S``-slot shared-memory ring a load of ``ptr`` streams through, None if
    ``cp.async`` cannot copy it (under 4 contiguous bytes a thread).
    An MMA operand's (``OPERAND``: ``stage``'s ``TRANSPOSED``) is in ``stage``'s
    layout when ``cp.async`` writes that (its MMA reads the slot); else an
    asynchronous MMA's (``STAGED``) is (ring or None, the restaged slots)."""
    mask = tl.core._unwrap_if_constexpr(mask)
    OPERAND, STAGED, S = map(tl.core._unwrap_if_constexpr, (OPERAND, STAGED, S))
    if OPERAND is not None:
        ptr, mask = _pipe_view(ptr, mask, _mma_shape(ptr.type.shape), _semantic)
    elif not ptr.type.is_block():
        ptr, mask = _pipe_view(ptr, mask, [1], _semantic)
    layout = _pipe_layout(ptr, _semantic)
    ty, shape = ptr.type.element_ty.element_ty, [S, *ptr.shape]
    ring = None
    if isinstance(layout, gl.BlockedLayout):
        order = layout.order
        vec = layout.size_per_thread[order[0]]
        if mask is not None and gluon_layout._is_block(mask):
            vec = min(
                vec, _semantic._ai(mask, gluon_layout._shape(ptr)).const[order[0]]
            )
        if vec * ty.primitive_bitwidth >= 32:
            ring = gl.SwizzledSharedLayout(vec, 1, 1, order)
    if OPERAND is not None:
        regs = gl.to_linear_layout(layout, ptr.shape, _semantic=_semantic)
        mma = gl.NVMMASharedLayout.get_default_for(
            ptr.shape, ty, _first_contiguous(regs, ty, OPERAND)
        )
        if ring is not None and ring.order[0] == (0 if mma.transposed else 1):  # pyrefly: ignore [missing-attribute]
            return gl.allocate_shared_memory(ty, shape, mma, _semantic=_semantic)
        if STAGED:
            staged = gl.allocate_shared_memory(ty, shape, mma, _semantic=_semantic)
            if ring is not None:
                ring = gl.allocate_shared_memory(ty, shape, ring, _semantic=_semantic)
            return tl.tuple([tl.constexpr(None) if ring is None else ring, staged])
    if ring is None:
        return None
    return gl.allocate_shared_memory(ty, shape, ring, _semantic=_semantic)


@gl_builtin
def pipe_put(
    ring, ptr, mask, pred, slot, cache_modifier="", eviction_policy="", _semantic=None
):
    """The copy of one step's load into ``slot`` of ``ring`` (masked off past the loop's end)."""
    ring, mask = (tl.core._unwrap_if_constexpr(x) for x in (ring, mask))
    if isinstance(ring, tl.tuple):
        ring = tl.core._unwrap_if_constexpr(ring[0])  # pyrefly: ignore [bad-index]
    if ring is None:
        return
    ptr, mask = _pipe_view(ptr, mask, ring.type.shape[1:], _semantic)  # pyrefly: ignore [missing-attribute]
    layout = _pipe_layout(ptr, _semantic)
    ptr = _semantic.settle(ptr, layout)  # pyrefly: ignore [missing-attribute]
    live = _semantic.splat(pred, ptr.shape, layout)  # pyrefly: ignore [missing-attribute]
    if mask is not None:
        live = _pipe_in(
            _semantic.and_(_pipe_in(mask, ptr.shape, layout, _semantic), live),  # pyrefly: ignore [missing-attribute]
            ptr.shape,
            layout,
            _semantic,
        )
    async_copy.async_load(  # pyrefly: ignore [missing-attribute]
        ring.index(slot, _semantic=_semantic),  # pyrefly: ignore [unexpected-keyword]
        ptr,
        live,
        tl.core._unwrap_if_constexpr(cache_modifier),
        tl.core._unwrap_if_constexpr(eviction_policy),
        _semantic=_semantic,
    )


@gl_builtin
def pipe_take(
    ring, slot, ptr, mask, other, cache_modifier="", eviction_policy="", _semantic=None
):
    """A pipelined load's value: read from its slot (a plain load without a ring);
    an MMA operand's slot itself, or (a pair) the value restaged in its slot."""
    ring, mask, other = (tl.core._unwrap_if_constexpr(x) for x in (ring, mask, other))
    if isinstance(ring, tl.tuple):
        # Concrete operands: a load of auto ones would anchor them (no replay
        # after the slot's store).
        ptr, mask = _pipe_view(ptr, mask, ring[1].type.shape[1:], _semantic)  # pyrefly: ignore [bad-index, missing-attribute]
        layout = _pipe_layout(ptr, _semantic)
        ptr = _semantic.settle(ptr, layout)  # pyrefly: ignore [missing-attribute]
        if mask is not None:
            mask = _pipe_in(mask, ptr.shape, layout, _semantic)
        value = pipe_take(
            ring[0],  # pyrefly: ignore [bad-index]
            slot,
            ptr,
            mask,
            other,
            cache_modifier,
            eviction_policy,
            _semantic=_semantic,
        )
        staged = ring[1].index(slot, _semantic=_semantic)  # pyrefly: ignore [bad-index]
        staged.store(value, _semantic=_semantic)
        fence_async_shared(_semantic=_semantic)
        return staged
    if ring is None:
        return tl.load(
            ptr, mask, other, cache_modifier=cache_modifier,
            eviction_policy=eviction_policy, _semantic=_semantic,
        )  # fmt: skip
    if isinstance(ring.type.layout, gl.NVMMASharedLayout):  # pyrefly: ignore [missing-attribute]
        return ring.index(slot, _semantic=_semantic)  # pyrefly: ignore [unexpected-keyword]
    scalar = not ptr.type.is_block()
    ptr, mask = _pipe_view(ptr, mask, ring.type.shape[1:], _semantic)  # pyrefly: ignore [missing-attribute]
    # (cp.async zero-fills masked lanes: a pipelined load's ``other`` is zero.)
    # The slot is refilled steps later: a use in another layout re-reads it.
    memory = ring.index(slot, _semantic=_semantic)  # pyrefly: ignore [unexpected-keyword]
    value = _semantic.slot_load(memory, _pipe_layout(ptr, _semantic))  # pyrefly: ignore [missing-attribute]
    return _semantic.unsplat(value) if scalar else value  # pyrefly: ignore [missing-attribute]


@gluon.jit
def pipe_wait(n: gl.constexpr, MMA: gl.constexpr):
    """Every thread's copies but the last ``n`` steps' landed: visible to the CTA
    (Membar puts a barrier after the wait), and with ``MMA`` (a tcgen05 MMA reads
    a slot) to the async proxy too (Membar's barrier before the MMA, one thread
    publishing every thread's fenced copies)."""
    async_copy.wait_group(n)
    if MMA:
        fence_async_shared()


@gluon.jit
def pipe_release(empty, i, S: gl.constexpr, LAST: gl.constexpr = False):  # pyrefly: ignore [bad-function-definition]
    """The MMAs of step ``i - 1`` complete (their slots free); after the loop's
    last step (``LAST``) its barriers retire."""
    mma_wait(empty, i, S)
    if LAST:
        for s in gl.static_range(S):
            mbarrier.invalidate(empty.index(s))


pipe_commit = async_copy.commit_group


# ---- ``tl`` of a gluon kernel ----------------------------------------------------
# Each ``tl.X`` Helion's Triton lowering emits is an entry of ``_TL``, else Gluon's
# namesake, else Triton's @jit standard library (traced with the Gluon semantic);
# any other name fails at trace time.


def _method(name: str) -> object:
    """``tl.<name>(x, ...)`` as the tensor method Gluon has instead."""

    @gl_builtin
    def call(x, *args: Any, _semantic: Any = None) -> Any:
        return getattr(x, name)(*args, _semantic=_semantic)

    return call


def _loaded(x: Any, _semantic: Any) -> Any:
    """``x`` in registers: a slot an MMA reads in place (``dot_input``) is read
    first (``tl`` ops leaving it unchanged return it as is)."""
    if not _shared(x.type):
        return x
    layout = gluon_layout.default_layout(x.shape, _semantic._warps())
    return x.load(layout, _semantic=_semantic)


def _slot_aware(op: Any, unchanged: Any) -> Any:
    """``tl.<op>``: a slot the op leaves as is (``unchanged``) is itself, else the
    op reads it first (``_loaded``)."""

    @gl_builtin
    def call(x: Any, *args: Any, _semantic: Any = None, **kwargs: Any) -> Any:
        if _shared(x.type) and unchanged(x, *args):
            return x
        return op(_loaded(x, _semantic), *args, _semantic=_semantic, **kwargs)

    return call


def _same_shape(x: Any, *shape: Any) -> bool:
    """(A unit batch ``pipe_alloc`` squeezed is the slot's own shape.)"""
    return list(x.shape) == tl.core._unwrap_shape(tl.core._unwrap_iterable(shape))


@gl_builtin
def _zeros(shape, dtype, layout=None, _semantic=None):
    """``tl.zeros`` as a builtin: Gluon's is a function returning an auto value."""
    return gl.full(shape, 0, dtype, layout, _semantic=_semantic)


@gluon.jit
def _descriptor_smem(desc, offsets):
    """A synchronous TMA copy of a box into shared memory."""
    smem = gl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, offsets, bar, smem)  # pyrefly: ignore [missing-attribute]
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    return smem


@gluon.jit
def _mma_drain():
    """Wait for the MMAs this thread issued so far: buffers they read are free."""
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)


@gluon.jit
def _descriptor_load(desc, offsets, TRANSPOSED: gl.constexpr = None):  # pyrefly: ignore [bad-function-definition]
    """``desc.load``: a synchronous TMA copy, read in the tile's vector layout; one
    an MMA reads in place (``TRANSPOSED``, ``stage``'s: ``plan_load``) is the copy
    where tcgen05 reads its order (``_k_major``), else restaged.  That MMA reads
    asynchronously, so the copy waits for it first: a loop's next copy or
    restage rewrites the buffer the last one read."""
    if TRANSPOSED is not None:
        _mma_drain()  # pyrefly: ignore [bad-argument-count]
    smem = _descriptor_smem(desc, offsets)  # pyrefly: ignore [bad-argument-count]
    if TRANSPOSED is None:
        value = smem.load(vector_layout(desc.block_shape, desc.dtype, gl.num_warps()))  # pyrefly: ignore [missing-attribute]
    elif _k_major(desc.dtype) and TRANSPOSED:
        value = stage(
            smem.load(vector_layout(smem.shape, smem.dtype, gl.num_warps())), True  # pyrefly: ignore [bad-argument-count, missing-attribute]
        )  # fmt: skip
    else:
        value = smem
    return value


@gluon.constexpr_function
def _mma_v2_layout(shape, num_warps):
    """mma.sync's encoding as Triton splits warps (``getMmaV2WarpsPerCTA``):
    balancing the 16-row and 8-column repetitions, ties toward M; a batched
    dot's along the batch."""
    if len(shape) == 3:
        return nvidia_helpers.tl_dot_mma_sync_layout(shape, num_warps)
    reps, warps = [-(-shape[0] // 16), -(-shape[1] // 8)], [1, 1]
    while warps[0] * warps[1] < num_warps:
        if reps[0] >= reps[1]:
            warps[0] *= 2
            reps[0] = reps[0] // 2 if reps[0] != 1 else 1
        else:
            warps[1] *= 2
            reps[1] //= 2
    return gl.NVMMADistributedLayout([2, 0], warps, [16, 8])


@gluon.constexpr_function
def _mma_shape(shape: Any) -> list[int]:
    """An MMA operand's shape: its unit batch squeezed."""
    return list(shape[1:] if len(shape) == 3 and shape[0] == 1 else shape)


@gluon.constexpr_function
def _register_dot_layout(a_ty, b_ty, precision, num_warps):
    """A register dot's result layout, operands' k width and result shape: FMA
    (fp64, fp32 ``ieee``) in the default layout, else mma.sync (a chained dot's
    warps those of the mma.sync result it reads, as Triton's)."""
    shape = [*a_ty.shape[:-1], b_ty.shape[-1]]
    dtype = a_ty.element_ty
    if dtype.is_fp64() or (dtype.is_fp32() and precision not in (None, "tf32")):
        return gluon_layout.default_layout(shape, num_warps), 0, shape
    k_width = nvidia_helpers.tl_dot_mma_sync_k_width(a_ty, b_ty)
    for lay in (a_ty.layout, b_ty.layout):
        if isinstance(lay, gl.NVMMADistributedLayout) and len(lay.warps_per_cta) == len(
            shape
        ):
            return (
                gl.NVMMADistributedLayout([2, 0], lay.warps_per_cta, lay.instr_shape),
                k_width,
                shape,
            )
    return _mma_v2_layout(shape, num_warps), k_width, shape


@gl_builtin
def _register_dot(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=gl.float32,
    _semantic: Any = None,
):
    """``tl.dot`` of a dot ``codegen_tmem_dot`` does not take: mma.sync (fp8
    operands: ptxas upcasts them), or FMA.  A builtin, so the operand conversions
    are the caller's (``gluon_layout`` hoists loop-invariant ones); a shared-memory
    operand (a pipelined or TMA slot) is read in its operand layout."""
    precision, acc = map(tl.core._unwrap_if_constexpr, (input_precision, acc))
    layout, k_width, shape = _register_dot_layout(
        a.type, b.type, precision, _semantic._warps()
    )
    a, b = _semantic.convert_layouts(
        [(x, gl.DotOperandLayout(i, layout, k_width)) for i, x in enumerate((a, b))]
    )
    if acc is None:
        acc = gl.full(
            shape,
            0,
            tl.core._unwrap_if_constexpr(out_dtype),
            layout,
            _semantic=_semantic,
        )
    acc = gl.convert_layout(acc, layout, _semantic=_semantic)
    if k_width == 0:
        return gl.dot_fma(a, b, acc, _semantic=_semantic)
    return mma_v2(a, b, acc, precision, _semantic=_semantic)


@gluon.constexpr_function
def _e4m3(scale) -> bool:
    return scale is not None and scale.dtype == gl.float8e4nv


@gluon.jit
def _e4m3_scaled(v, scale, FORMAT: gl.constexpr, OPERAND: gl.constexpr):
    """An operand times its e4m3 block scales, in bf16 (the translator's
    decomposition reads e8m0 exponents only; Triton's casts e4m3)."""
    RANK: gl.constexpr = len(v.type.shape)  # pyrefly: ignore [bad-assignment]
    K: gl.constexpr = RANK - 1 - OPERAND
    if FORMAT == "e2m1":
        v = gl.fp4_to_fp(v, gl.bfloat16, K)
    else:
        v = v.to(gl.bfloat16)
    if OPERAND == 1:
        # pyrefly: ignore [bad-assignment]
        order: gl.constexpr = common_helpers.tl_dot_decomposed_get_transposed_order(
            RANK
        )
        scale = gl.permute(scale, order)
    scale = common_helpers.tl_dot_decomposed_broadcast_scale(
        scale.to(gl.bfloat16), K, v.shape[K] // scale.shape[K]
    )
    return v * gl.convert_layout(scale, v.type.layout)


@gluon.jit
def _decomposed_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=gl.float32,
):
    """``tl.dot_scaled`` of a dot ``codegen_tmem_dot`` does not take: Triton's
    decomposition (the translator's; fast math only: NaN scales are not masked)."""
    A_E4M3: gl.constexpr = _e4m3(lhs_scale)  # pyrefly: ignore [bad-assignment]
    B_E4M3: gl.constexpr = _e4m3(rhs_scale)  # pyrefly: ignore [bad-assignment]
    if A_E4M3:
        lhs = _e4m3_scaled(lhs, lhs_scale, lhs_format, 0)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
    if B_E4M3:
        rhs = _e4m3_scaled(rhs, rhs_scale, rhs_format, 1)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
    return common_helpers.tl_dot_decomposed_block_scales_impl(
        _decomposed_dot_scaled, _register_dot,  # pyrefly: ignore [bad-argument-type]
        lhs, None if A_E4M3 else lhs_scale, "bf16" if A_E4M3 else lhs_format,
        rhs, None if B_E4M3 else rhs_scale, "bf16" if B_E4M3 else rhs_format,
        acc, True, lhs_k_pack, rhs_k_pack, out_dtype,
    )  # fmt: skip


@gl_builtin
def _dot_scaled(
    *args: Any,
    acc: Any = None,
    _semantic: Any = None,
    _generator: Any = None,
    **kwargs: Any,
) -> Any:
    """``_decomposed_dot_scaled`` with its accumulator in the register dot's layout
    first: a loop's dry run then calls it with the loop's own types (a callee
    of auto values left uncalled is one Gluon rejects)."""
    if acc is not None:
        layout = _mma_v2_layout(acc.shape, _semantic._warps())
        acc = _semantic.settle(acc, layout)
    kwargs["acc"] = acc
    return _generator.call_JitFunction(_decomposed_dot_scaled, args, kwargs)


_TL: dict[str, object] = {
    "extra": types.SimpleNamespace(cuda=tl.extra.cuda, libdevice=gl.extra.libdevice),
    "range": tl.range,
    "PropagateNan": tl.PropagateNan,
    "debug_barrier": gl.barrier,
    **{name: _method(name) for name in ("permute", "trans", "view")},
    "broadcast_to": _slot_aware(tl.broadcast_to, _same_shape),
    "reshape": _slot_aware(tl.reshape, _same_shape),
    "zeros": _zeros,
    "cast": _slot_aware(
        tl.cast, lambda x, dtype, *_: x.dtype == tl.core._unwrap_if_constexpr(dtype)
    ),
    "dot": _register_dot,
    "dot_scaled": _dot_scaled,
    "load_tensor_descriptor": _descriptor_load,
    "store_tensor_descriptor": nvidia_helpers.tl_store_tensor_descriptor,
}


def _tl(name: str) -> object:
    if name in _TL:
        return _TL[name]
    if hasattr(gl, name):
        return getattr(gl, name)
    if isinstance(value := getattr(tl, name, None), JITFunction):
        return value
    raise AttributeError(f"tl.{name} is not supported on gluon")


# The generated code's ``tl`` (``GluonBackend.library_imports``).
tl_namespace = types.ModuleType("helion_gluon_tl")
tl_namespace.__getattr__ = _tl  # pyrefly: ignore [missing-attribute]


# ---- single ops ----------------------------------------------------------------
# A tcgen05 dot is MMA ``count`` of its thread, arriving on ``done[count % S]``.
# Its tensor-memory accumulator stays through the loop (``acc_alloc`` /
# ``acc_read``), goes in and out per statement (``acc_put`` / ``acc_get``), or is
# a slot of an MMA partition's accumulator ring (``acc_claim`` / ``acc_take``).


@gluon.jit
def view_reshape(x, SHAPE: gl.constexpr):
    return x.reshape(SHAPE)


@gluon.jit
def view_permute(x, ORDER: gl.constexpr):
    return x.permute(ORDER)


@gl_builtin
def reinterpret(x, dtype, shape, layout, _semantic: Any = None):
    """A memory descriptor's view as ``shape`` of ``dtype`` in ``layout``
    (``_reinterpret`` on Triton 3.8, ``reinterpret`` from 3.9)."""
    method = getattr(x, "reinterpret", None) or x._reinterpret
    return method(dtype, shape, layout, _semantic=_semantic)


@gluon.jit
def view_dtype(x, DTYPE: gl.constexpr):
    return reinterpret(x, DTYPE, x.shape, x.layout)  # pyrefly: ignore [bad-argument-count]


@gl_builtin
def stage(
    x,
    TRANSPOSED,
    SHAPE=None,
    PADDED=False,
    done=None,
    count=0,
    S=1,
    _semantic: Any = None,
):
    """A body value staged as a shared-memory MMA operand, broadcast to ``SHAPE``
    there, as Triton stages one: in its registers' contiguous dimension, K-major
    where tcgen05 reads no other order (``_k_major``: first dimension contiguous
    if ``TRANSPOSED``);
    ``PADDED``: fp4 beside an 8-bit operand; a unit batch squeezed.  A
    shared-memory value (a slot read in place) is its own operand.  With ``done``
    (its MMAs' completion, number ``count`` next), the store waits for the MMA
    that read the previous restage, into memory allocated ahead of the loop
    (Triton's: nothing else reuses it while the MMA reads)."""
    TRANSPOSED, SHAPE, PADDED, done = map(
        tl.core._unwrap_if_constexpr, (TRANSPOSED, SHAPE, PADDED, done)
    )
    shape = _mma_shape(x.type.shape)
    if len(x.type.shape) != len(shape):  # pyrefly: ignore [bad-argument-type]
        x = x.reshape(shape, _semantic=_semantic)
    if _shared(x.type):
        return x
    if done is not None:
        mma_wait(done, count, S, _semantic=_semantic)
    t = _semantic.settle(x, gluon_layout.default_layout(shape, _semantic._warps()))
    if SHAPE is not None:
        t = t.broadcast_to(SHAPE, _semantic=_semantic)
    regs = gl.to_linear_layout(t.type.layout, t.shape, _semantic=_semantic)
    layout = gl.NVMMASharedLayout.get_default_for(
        t.shape, t.dtype, _first_contiguous(regs, t.dtype, TRANSPOSED), PADDED
    )
    if done is None:
        smem = gl.allocate_shared_memory(
            t.dtype, t.shape, layout, t, _semantic=_semantic
        )
    else:
        smem = _semantic.loop_buffer(t.dtype, t.shape, layout)
        smem.store(t, _semantic=_semantic)
    fence_async_shared(_semantic=_semantic)
    return smem


@gluon.constexpr_function
def _k_major(dtype: Any) -> bool:
    """Whether tcgen05 reads ``dtype`` operands K-major only (Triton's MMAv5 rule:
    tf32, and fp4 packed in bytes); others in either order."""
    return dtype.primitive_bitwidth == 32 or dtype.is_int()


@gluon.constexpr_function
def _first_contiguous(regs: Any, dtype: Any, transposed: Any) -> bool:
    if _k_major(dtype) or len(regs.shape) != 2:
        return transposed
    basis = next((b for b in [*regs.reg_bases, *regs.lane_bases] if any(b)), [0, 1])
    return basis[1] == 0


@gluon.constexpr_function
def acc_layout(M: Any, N: Any, dtype: Any = gl.float32) -> TensorMemoryLayout:
    """Triton's tensor-memory layout of a dot accumulator ([M, N] of ``dtype``)."""
    block = [128 if M >= 128 else 64, min(N, 256)]
    return TensorMemoryLayout(block, col_stride=32 // dtype.primitive_bitwidth)  # pyrefly: ignore [bad-argument-type]


@gluon.jit
def acc_alloc(M: gl.constexpr, N: gl.constexpr, DTYPE: gl.constexpr = gl.float32):  # pyrefly: ignore [bad-function-definition]
    """A dot's tensor-memory accumulator and its barrier."""
    tm = allocate_tensor_memory(DTYPE, [M, N], acc_layout(M, N, DTYPE))
    return tm, mma_barrier()  # pyrefly: ignore [bad-argument-count]


@gluon.jit
def mma_barrier():
    done = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    mbarrier.init(done.index(0), count=1)
    return done


@gl_builtin
def acc_put(tm, value, _semantic: Any = None):
    """``value`` into tensor memory ``tm``: a builtin, so the caller's semantic
    converts it (a constant is created in the register layout instead)."""
    layout = tm.get_reg_layout(_semantic=_semantic, _generator=_semantic.generator)
    tm.store(gl.convert_layout(value, layout, _semantic=_semantic), _semantic=_semantic)


@gluon.jit
def acc_get(tm):
    return tm.load(tm.get_reg_layout())


@gluon.jit
def acc_free(done):
    mbarrier.invalidate(done.index(0))


@gluon.jit
def acc_read(tm, done, SYNCED: gl.constexpr):
    """The accumulator once this thread's MMAs into it complete (``SYNCED``: each
    was waited for)."""
    if not SYNCED:
        tcgen05_commit(done.index(0))
        mbarrier.wait(done.index(0), 0)
    mbarrier.invalidate(done.index(0))
    return acc_get(tm)


@gluon.jit
def acc_buffers(M: gl.constexpr, N: gl.constexpr, B: gl.constexpr):
    """An accumulator ring: ``B`` tensor-memory accumulators of ``[M, N]``."""
    return allocate_tensor_memory(gl.float32, [B, M, N], acc_layout(M, N))


@gluon.jit
def acc_claim(buffers, empty, tile):
    """MMA side: accumulator ``tile``, once its reader released it."""
    B: gl.constexpr = buffers.shape[0]
    mbarrier.wait(empty.index(tile % B), ((tile // B) & 1) ^ 1)
    return buffers.index(tile % B)


@gluon.jit
def acc_commit(full, tile):
    """MMA side: accumulator ``tile`` is ready once this thread's MMAs complete."""
    tcgen05_commit(full.index(tile % full.shape[0]))


@gluon.jit
def acc_take(buffers, full, empty, tile):
    """Reader side: accumulator ``tile``, read into the body, then released."""
    B: gl.constexpr = buffers.shape[0]
    mbarrier.wait(full.index(tile % B), (tile // B) & 1)
    acc = acc_get(buffers.index(tile % B))
    mbarrier.arrive(empty.index(tile % B))
    return acc


@gluon.jit
def tmem_dot(
    a, b, tm, fresh, done, count, S: gl.constexpr, SYNC: gl.constexpr, own=None
):
    """``tm += a @ b`` (``=`` while ``fresh``), arriving on ``done[count % S]``
    and on ``own`` (its restages' barrier beside a ring channel's); with
    ``SYNC`` it waits for the MMA."""
    s = count % S
    if own is None:
        tcgen05_mma(a, b, tm, use_acc=fresh == 0, mbarriers=[done.index(s)])
    else:
        tcgen05_mma(
            a, b, tm, use_acc=fresh == 0, mbarriers=[done.index(s), own.index(0)]
        )
    if SYNC:
        mma_wait(done, count + 1, S)


@gl_builtin
def mma_wait(done, count, S, _semantic: Any = None):
    """The MMA before number ``count`` complete (operands it reads free)."""
    last = _semantic.to_tensor(count).__sub__(1, _semantic=_semantic)
    S = tl.core._unwrap_if_constexpr(S)
    mbarrier.wait(
        done.index(last.__mod__(S, _semantic=_semantic), _semantic=_semantic),
        last.__floordiv__(S, _semantic=_semantic).__and__(1, _semantic=_semantic),
        pred=last.__ge__(0, _semantic=_semantic),
        _semantic=_semantic,
    )


@gluon.jit
def _scale_put(x, COPY: gl.constexpr, sc):
    """A dot's block scales into tensor memory ``sc``: by tcgen05.cp (``COPY``,
    ``copied_scale``), else through registers."""
    if COPY:
        tcgen05_copy(x, sc)
    elif _shared(x.type):
        sc.store(x.load(sc.get_reg_layout()))  # pyrefly: ignore [missing-attribute]
        fence_async_shared()  # the read ahead of the slot's next TMA write
    else:
        sc.store(gl.convert_layout(x, sc.get_reg_layout()))  # pyrefly: ignore [missing-attribute]


@gluon.jit
def _scales(dtype: gl.constexpr, shape: gl.constexpr):
    """Tensor memory for block scales of ``shape`` (at least 32 rows: Triton gives a
    16-row buffer half the lanes, which may sit at lane 64, where MMAs read none)."""
    ROWS: gl.constexpr = 32 if shape[0] < 32 else shape[0]  # noqa: FURB136  # pyrefly: ignore [bad-assignment]
    SHAPE: gl.constexpr = [ROWS, shape[1]]  # pyrefly: ignore [bad-assignment]
    sc = allocate_tensor_memory(dtype, SHAPE, TensorMemoryScalesLayout())
    if shape[0] != ROWS:
        sc = reinterpret(sc, dtype, shape, TensorMemoryScalesLayout())  # pyrefly: ignore [bad-argument-count]
    return sc


@gluon.constexpr_function
def _restaged(layout: Any, fmt: Any, other: Any, b: Any) -> bool:
    """Whether tcgen05 cannot read an fp4 operand's slot (``layout``) as is:
    it reads fp4 K-major, padded beside an 8-bit operand (kind::f8f6f4)."""
    return (
        fmt == "e2m1"
        and isinstance(layout, gl.NVMMASharedLayout)
        and (layout.transposed != b or layout.fp4_padded != (fmt != other))
    )


@gluon.jit
def _fp4_operand(x, FORMAT: gl.constexpr, OTHER: gl.constexpr, B: gl.constexpr):
    """An fp4 operand's slot, restaged if tcgen05 cannot read it (as Triton's)."""
    if _restaged(x.type.layout, FORMAT, OTHER, B):
        x = stage(
            x.load(vector_layout(x.shape, x.dtype, gl.num_warps())), B, None, FORMAT != OTHER  # pyrefly: ignore [bad-argument-count]
        )  # fmt: skip
    return x


@gluon.jit
def tmem_dot_scaled(
    a,
    a_scale,
    a_format: gl.constexpr,
    b,
    b_scale,
    b_format: gl.constexpr,
    tm,
    fresh,
    done,
    count,
    S: gl.constexpr,
    A_COPY: gl.constexpr,
    B_COPY: gl.constexpr,
    SYNC: gl.constexpr,
    PIPELINE: gl.constexpr,
):
    """``tm += (a * a_scale) @ (b * b_scale)`` (``=`` while ``fresh``).  Scales
    stored from registers wait for the MMA reading the previous ones, or
    (``PIPELINE``) alternate two buffers, waiting for the MMA two back; with a
    restaged operand (one buffer: the next call's restage) it waits as ``SYNC``."""
    A_RESTAGED: gl.constexpr = _restaged(a.type.layout, a_format, b_format, False)  # pyrefly: ignore [bad-assignment]
    WAIT: gl.constexpr = (
        # pyrefly: ignore [bad-assignment]
        SYNC or A_RESTAGED or _restaged(b.type.layout, b_format, a_format, True)
    )
    a = _fp4_operand(a, a_format, b_format, False)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
    b = _fp4_operand(b, b_format, a_format, True)  # pyrefly: ignore [bad-argument-count, bad-argument-type]
    s = count % S
    A_STORED: gl.constexpr = not A_COPY  # pyrefly: ignore [bad-assignment]
    B_STORED: gl.constexpr = not B_COPY  # pyrefly: ignore [bad-assignment]
    a_sc = _scales(a_scale.dtype, a_scale.shape)  # pyrefly: ignore [bad-argument-count]
    b_sc = _scales(b_scale.dtype, b_scale.shape)  # pyrefly: ignore [bad-argument-count]
    # Scales stored from registers in two alternating buffers (``PIPELINE``).
    ALTERNATE: gl.constexpr = (A_STORED or B_STORED) and PIPELINE and S > 1 and not WAIT  # pyrefly: ignore [bad-assignment]
    if ALTERNATE:
        a_alt = a_sc
        b_alt = b_sc
        if A_STORED:
            a_alt = _scales(a_scale.dtype, a_scale.shape)  # pyrefly: ignore [bad-argument-count]
        if B_STORED:
            b_alt = _scales(b_scale.dtype, b_scale.shape)  # pyrefly: ignore [bad-argument-count]
        back = count - 2 + fresh
        if back >= 0:
            mbarrier.wait(done.index(back % S), (back // S) & 1)
        if count % 2 == 1:
            a_sc = a_alt
            b_sc = b_alt
    _scale_put(a_scale, A_COPY, a_sc)  # pyrefly: ignore [bad-argument-count]
    _scale_put(b_scale, B_COPY, b_sc)  # pyrefly: ignore [bad-argument-count]
    tcgen05_mma_scaled(
        a,
        b,
        tm,
        a_sc,
        b_sc,
        a_format,
        b_format,
        use_acc=fresh == 0,
        mbarriers=[done.index(s)],
    )
    if not ALTERNATE and (A_STORED or B_STORED or WAIT):
        mbarrier.wait(done.index(s), (count // S) & 1)


# ---- the compiler side ---------------------------------------------------------

aten = torch.ops.aten
# The dot targets' (``matmul_operand_positions``) block scales and accumulators.
_SCALES: dict[object, tuple[int, ...]] = {dot_scaled: (1, 4)}
_FORMATS = (2, 5)  # ``dot_scaled``'s operand formats
_ACCUMULATOR: dict[object, int] = {
    dot: 2,
    dot_scaled: 6,
    aten.addmm.default: 0,
    aten.baddbmm.default: 0,
}
_TRANSPOSES = (aten.permute.default, aten.t.default, aten.transpose.int)
_RESHAPES = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)
# Value copies: a memory view keeps reading the same memory.
_COPIES = (_new_var, aten.clone.default, aten.alias.default)
_SHAPE_QUERIES = (aten.sym_size.int, aten.sym_stride.int)
_CONVERTS = (torch.ops.prims.convert_element_type.default, aten._to_copy.default)


@dataclasses.dataclass(frozen=True)
class SharedValue:
    """A Gluon shared-memory descriptor in the body, or a memory view of one:
    a tensor-memory dot's operand, read by nothing else."""

    expr: ast.AST


def is_dot(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in matmul_operand_positions()


def operands(target: object) -> tuple[int, ...]:
    """A dot's A and B, then its block scales: what a shared-memory value may feed."""
    return (*matmul_operand_positions()[target], *_SCALES.get(target, ()))


def is_scale(dot_node: torch.fx.Node, position: int) -> bool:
    return position in _SCALES.get(dot_node.target, ())


def copied_scale(dot_node: torch.fx.Node, position: int) -> bool:
    """Whether a shared-memory operand is a scale tcgen05.cp copies: 128 rows or
    more viewed (reshape / permute) from their packed blocks, the order tcgen05.cp
    reads (Triton's tmem_copy pattern); a block as loaded goes through registers."""
    arg = cast("torch.fx.Node", dot_node.args[position])
    return (
        is_scale(dot_node, position)
        and concrete(arg.meta["val"].shape)[0] >= 128
        and arg.target in (*_TRANSPOSES, *_RESHAPES)
    )


def _tuple(items: Iterable[object]) -> str:
    return f"({''.join(f'{item}, ' for item in items)})"


def reshape(value: ast.AST, dims: Iterable[str]) -> ast.AST:
    return expr_from_string(f"{ALIAS}.view_reshape({{x}}, {_tuple(dims)})", x=value)


def permute(value: ast.AST, order: Iterable[int]) -> ast.AST:
    return expr_from_string(f"{ALIAS}.view_permute({{x}}, {_tuple(order)})", x=value)


def _permutation(node: torch.fx.Node) -> list[int]:
    rank = len(node.meta["val"].shape)
    if node.target is aten.t.default:
        return [1, 0]
    if node.target is aten.transpose.int:
        order = list(range(rank))
        i, j = (d % rank for d in cast("tuple[int, int]", node.args[1:3]))
        order[i], order[j] = order[j], order[i]
        return order
    return [d % rank for d in cast("list[int]", node.args[1])]


def _is_identity(node: torch.fx.Node) -> bool:
    return node.target in _COPIES or (
        node.target in _CONVERTS
        and node.meta["val"].dtype
        == cast("torch.fx.Node", node.args[0]).meta["val"].dtype
    )


def _is_view(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and (
        node.target in _TRANSPOSES
        or node.target in _RESHAPES
        or node.target is aten.view.dtype
        or _is_identity(node)
    )


def dot_operands(node: torch.fx.Node) -> list[tuple[torch.fx.Node, int]] | None:
    """The dots and operand positions a value feeds through memory views: all of
    its uses, [] if it feeds no dot, None if it feeds one and something else."""
    uses: list[tuple[torch.fx.Node, int]] = []
    other = False
    for user in node.users:
        positions = [i for i, arg in enumerate(user.args) if arg is node]
        if is_dot(user) and positions[:1] and positions[0] in operands(user.target):
            other |= len(positions) != 1
            uses.append((user, positions[0]))
        elif _is_view(user) and user.args[0] is node:
            if (more := dot_operands(user)) is None:
                return None
            other |= not more
            uses += more
        else:
            other = True
    return None if uses and other else uses


def reaches(node: torch.fx.Node) -> set[torch.fx.Node]:
    """The nodes ``node``'s value reaches, not through memory (a load's index or a
    stored value is no use of it)."""
    seen, todo = set(), [node]
    while todo:
        for user in todo.pop().users:
            if user not in seen and user.target not in (
                memory_ops.load,
                memory_ops.store,
            ):
                seen.add(user)
                todo.append(user)
    return seen


def dot_input(
    device_function: DeviceFunction, node: torch.fx.Node
) -> tuple[bool | None] | None:
    """How load ``node`` feeds its graph's dots (``gluon_pipeline.software_pipeline``
    reads it off the load's code): None if it feeds none (a load's index or a
    stored value is no feed); ``(TRANSPOSED,)`` if every use is an operand an MMA
    reads in place (``stage``'s orientation), directly or through one 2-D
    transpose; else ``(None,)``."""
    if not any(map(is_dot, reaches(node))):
        return None
    orientations = {
        _read_in_place(device_function, node, *use) for use in dot_operands(node) or ()
    }
    return (orientations.pop(),) if len(orientations) == 1 else (None,)


def _read_in_place(
    device_function: DeviceFunction,
    node: torch.fx.Node,
    dot_node: torch.fx.Node,
    position: int,
) -> bool | None:
    """``stage``'s TRANSPOSED for a dot reading ``node`` (operand ``position``)
    from its slot as is, or through one 2-D transpose (Triton's memdesc_trans);
    None if it cannot (a scale, a padded K, a decomposed ``dot_scaled``, another
    view)."""
    arg = cast("torch.fx.Node", dot_node.args[position])
    flip = (
        arg is not node
        and arg.target is aten.permute.default
        and (arg.args[0] is node and _permutation(arg) == [1, 0])
    )
    if (
        is_scale(dot_node, position)
        or padded_k(dot_node)
        or (arg is not node and not flip)
        or (
            dot_node.target is dot_scaled
            and not tmem_dot_supported(device_function, dot_node)
        )
    ):
        return None
    lhs = position == matmul_operand_positions()[dot_node.target][0]
    return _transposed(lhs, [1, 0] if flip else [0, 1])


def _transposed(lhs: bool, view: list[int]) -> bool:
    """``stage``'s TRANSPOSED for an operand whose dims are its source's ``view``
    (``view[d]``: the source dim of dim ``d``): the source's contiguous dim is not K."""
    rank = len(view)
    return view.index(rank - 1) != (rank - 1 if lhs else rank - 2)


def plan_load(
    device_function: DeviceFunction, node: torch.fx.Node, value: ast.AST
) -> None:
    """Put ``dot_input`` on the load calls of ``value`` (a load's code); a TMA load
    MMAs read in place (``value`` itself) copies for them (``_descriptor_load``)."""
    plan = dot_input(device_function, node)
    for call in ast.walk(value):
        if _is_load(call) and isinstance(call, ExtendedAST):
            call._plan = plan
    if _is_tl_call(value, frozenset(["load_tensor_descriptor"])) and plan:
        if plan[0] is not None:
            value.keywords.append(ast.keyword("TRANSPOSED", ast.Constant(plan[0])))


def concrete(shape: Iterable[int | torch.SymInt]) -> list[int]:
    env = CompileEnvironment.current()
    config = DeviceFunction.current().config
    return [
        size
        if (size := _resolve_configured_extent(d, config)) is not None
        else env.size_hint(d)
        for d in shape
    ]


def _tl_type(value: torch.Tensor, shape: list[int]) -> tl.block_type:
    dtype = (
        tl.uint8
        if value.dtype == torch.float4_e2m1fn_x2
        else cast("tl.dtype", torch_dtype_to_tl(value.dtype))
    )
    return tl.block_type(dtype, shape)


def _accumulator(node: torch.fx.Node) -> object:
    position = _ACCUMULATOR.get(node.target, len(node.args))
    return node.args[position] if position < len(node.args) else None


def _zero_fill(value: object) -> bool:
    """Whether ``value`` is a zero fill (through copies)."""
    while isinstance(value, torch.fx.Node) and value.target is _new_var:
        value = value.args[0]
    return (
        isinstance(value, torch.fx.Node)
        and value.target is full
        and cast("float", value.args[1]) == 0
    )


def _carried(node: torch.fx.Node) -> torch.fx.Node | None:
    """The loop-carried accumulator a dot reads (through its phi copies)."""
    accumulator = _accumulator(node)
    while (
        isinstance(accumulator, torch.fx.Node)
        and accumulator.target is _new_var
        and len(accumulator.users) == 1
    ):
        accumulator = accumulator.args[0]
    if isinstance(accumulator, torch.fx.Node) and accumulator.op == "placeholder":
        return accumulator
    return None


def _graph_info(device_function: DeviceFunction, graph: torch.fx.Graph) -> GraphInfo:
    return next(i for i in device_function.graph_infos() if i.graph is graph)


def padded_k(node: torch.fx.Node) -> int | None:
    """The K a dot's Triton lowering zero-pads its operands to
    (``emit_tl_dot_with_padding``: doubling up to ``min_dot_size``), else None."""
    if node.target is dot_scaled:
        return None
    a, b = (
        cast("torch.fx.Node", node.args[i]).meta["val"]
        for i in matmul_operand_positions()[node.target]
    )
    k = concrete(a.shape)[-1]
    min_k = min_dot_size(CompileEnvironment.current().device, a.dtype, b.dtype)[2]
    return min_k if k < min_k else None


def tmem_dot_supported(device_function: DeviceFunction, node: torch.fx.Node) -> bool:
    """Whether Triton lowers this dot to a tcgen05 MMA, in a form
    ``codegen_tmem_dot`` emits (one dtype, fp32 or no accumulator, no ``addmm``
    scaling, rank 2 once a unit batch squeezes)."""
    a, b = (
        cast("torch.fx.Node", node.args[i]).meta["val"]
        for i in matmul_operand_positions()[node.target]
    )
    acc = _accumulator(node)
    scaled = node.target is dot_scaled
    if (
        node.kwargs.get("alpha", 1) != 1
        or node.kwargs.get("beta", 1) != 1
        or (a.dtype != b.dtype and not scaled)
        or (acc is not None and not isinstance(acc, torch.fx.Node))
        or (
            isinstance(acc, torch.fx.Node)
            and {acc.meta["val"].dtype, node.meta["val"].dtype} != {torch.float32}
        )
    ):
        return False
    shapes = [concrete(t.shape) for t in (a, b)]
    if len(shapes[0]) == 3:
        if shapes[0][0] != 1 or shapes[1][0] != 1:
            return False
        shapes = [shape[1:] for shape in shapes]
    if k := padded_k(node):
        shapes = [[shapes[0][0], k], [k, shapes[1][1]]]
    # The translator's rule (Triton's; Gluon has none); a block-scaled dot also
    # needs fp8 / fp4 operands.
    a_ty, b_ty = _tl_type(a, shapes[0]), _tl_type(b, shapes[1])
    warps = device_function.config.num_warps
    if scaled:
        return (
            all(isinstance(node.args[i], torch.fx.Node) for i in _SCALES[dot_scaled])
            and {node.args[i] for i in _FORMATS} <= {"e2m1", "e4m3", "e5m2"}
            and bool(blackwell_helpers.tl_dot_scaled_mmav5_supported(a_ty, b_ty, warps))
        )
    env = CompileEnvironment.current()
    precision = env.backend.map_dot_precision(env.settings.dot_precision)
    return bool(
        blackwell_helpers.tl_dot_mmav5_supported(
            a_ty, b_ty, warps, precision, None, None
        )
    )


def resident(device_function: DeviceFunction, node: torch.fx.Node) -> bool:
    """Whether the dot's accumulator stays in tensor memory through its loop: a
    loop-carried fp32 one nothing else in the loop touches."""
    accumulator = _carried(node)
    info = _graph_info(device_function, node.graph)
    return (
        accumulator is not None
        and len([u for u in accumulator.users if u.target not in _SHAPE_QUERIES]) == 1
        and all(user.op == "output" for user in node.users)
        and isinstance(info, ForLoopGraphInfo)
        and not isinstance(info, ReductionLoopGraphInfo)
    )


def codegen_node(interp: GraphInterpreter, node: torch.fx.Node) -> object:
    """The gluon lowering of a ring slot's consumers (``gluon_ring``) and of every
    tcgen05 dot; NotImplemented for other nodes (Triton's lowering)."""
    device_function = cast("GenerateAST", interp.cg).device_function
    if device_function.ring is not None:
        result = device_function.ring.codegen_node(interp, node)
        if result is not NotImplemented:
            return result
    if is_dot(node) and tmem_dot_supported(device_function, node):
        return codegen_tmem_dot(interp, node)
    return NotImplemented


def codegen_view(interp: GraphInterpreter, node: torch.fx.Node) -> SharedValue:
    """A memory view of a shared-memory value."""
    source = interp.env[cast("torch.fx.Node", node.args[0])]
    assert isinstance(source, SharedValue) and _is_view(node), node
    if _is_identity(node):
        return source
    if node.target in _TRANSPOSES:
        return SharedValue(permute(source.expr, _permutation(node)))
    if node.target is aten.view.dtype:
        dtype = CompileEnvironment.current().backend.dtype_str(
            cast("torch.dtype", node.args[1])
        )
        return SharedValue(
            expr_from_string(f"{ALIAS}.view_dtype({{x}}, {dtype})", x=source.expr)
        )
    dims = interp.cg.device_function.tile_strategy.shape_dims(node.meta["val"].shape)
    return SharedValue(reshape(source.expr, dims))


def fresh_accumulator(device_function: DeviceFunction, node: torch.fx.Node) -> bool:
    """Whether the dot's loop starts its accumulator at zero and statically runs."""
    accumulator = cast("torch.fx.Node", _carried(node))
    info = cast("ForLoopGraphInfo", _graph_info(device_function, accumulator.graph))
    loop, _ = control_flow_parent_entries(device_function.graph_infos())[info.graph_id]
    if loop.target not in (_for_loop, _for_loop_step):
        return False
    initial = info.placeholder_to_outer_arg(accumulator)
    bounds = zip(
        *cast("tuple[list[object], list[object]]", loop.args[1:3]), strict=True
    )
    return _zero_fill(initial) and all(
        type(b) is int and type(e) is int and e > b for b, e in bounds
    )


def codegen_tmem_dot(
    interp: GraphInterpreter,
    node: torch.fx.Node,
    completion: tuple[str, str, str] | None = None,
    ring: tuple[str, ...] | None = None,
) -> ast.AST:
    """The dot as a tcgen05 MMA, its accumulator in tensor memory through its loop
    (``resident``, or ``ring``'s), else per statement.  It completes on
    ``completion`` (a ring channel's ``done``, ``count``, ``S``), else on its own."""
    cg = cast("GenerateAST", interp.cg)
    device_function = cg.device_function
    a_pos, b_pos = matmul_operand_positions()[node.target]
    shape = concrete(node.meta["val"].shape)
    M, N = shape[-2:]
    squeeze = len(shape) == 3  # a unit batch (``tmem_dot_supported``)

    def view(x: str, *batch: int) -> str:
        """The accumulator ``x`` with ``batch`` (squeezed without)."""
        return f"tl.reshape({x}, {[*batch, M, N]})" if squeeze else x

    k_pad = padded_k(node)
    loops = cg._active_loop_stack()
    in_place = ring is not None or resident(device_function, node)
    barrier = device_function.new_var("tmem_done")
    prefix: list[str] = []
    # A loop-resident accumulator's MMA reading a restaged operand completes by
    # the operand's next restage (Triton's): ``stage`` waits for it then, on the
    # MMAs' own barrier ``own`` (the accumulator's commits once after the loop).
    # Beside a ring channel's barrier the MMA arrives on both: a step of the
    # channel may be a register read, which orders no MMA.
    lazy = in_place and node.target is not dot_scaled
    beside = (
        completion is not None
        and ring is None
        and lazy
        and not all(
            isinstance(interp.env[cast("torch.fx.Node", node.args[p])], SharedValue)
            for p in (a_pos, b_pos)
        )
    )
    own: tuple[str, ...] = ()
    if (completion is None and ring is None) or beside:
        own = (
            device_function.new_var("tmem_mma") if lazy else barrier,
            device_function.new_var("tmem_count"),
            "1",
        )
        prefix.append(f"{own[1]} = 0")
        if lazy:
            prefix.append(f"{own[0]} = {ALIAS}.mma_barrier()")
    done, count, depth = completion or own
    operand_asts: dict[int, ast.AST] = {}
    in_smem: dict[int, bool] = {}
    for position in operands(node.target):
        arg = cast("torch.fx.Node", node.args[position])
        value = interp.env[arg]
        in_smem[position] = isinstance(value, SharedValue)
        if isinstance(value, SharedValue):
            operand_asts[position] = value.expr
            if squeeze and not is_scale(node, position):
                dims = map(str, concrete(arg.meta["val"].shape)[1:])
                operand_asts[position] = reshape(value.expr, dims)
        elif is_scale(node, position):
            operand_asts[position] = cast("ast.AST", value)
        else:
            operand_asts[position] = _staged(
                interp,
                arg,
                position == a_pos,
                squeeze,
                k_pad,
                own if lazy else (),
            )
    if ring is None and device_function.ring is not None:
        device_function.ring.tmem.append((M, N))  # beside the ring's: its ledger
    # Else an operand staged from registers waits for its MMA right away.
    sync = not lazy and not (in_smem[a_pos] and in_smem[b_pos])
    acc_node = _accumulator(node)
    # A statement's zero fill the MMA overwrites instead (Triton's
    # OptimizeAccumulatorInit).
    accumulator = (
        interp.env[acc_node]
        if isinstance(acc_node, torch.fx.Node)
        and (in_place or not _zero_fill(acc_node))
        else None
    )
    tm = device_function.new_var("tmem_acc")
    name = device_function.new_var(node.name)
    if ring is not None:
        buffers, full_barriers, empty, tile = ring
        # Set until the dot's first MMA (of each loop) issues.
        fresh = device_function.new_var("tmem_fresh", dce=True)
        prefix += [
            f"{tm} = {ALIAS}.acc_claim({buffers}, {empty}, {tile})",
            f"{fresh} = 1",
        ]
        suffix = [
            f"{ALIAS}.acc_commit({full_barriers}, {tile})",
            f"{name} = {view(f'{ALIAS}.acc_take({buffers}, {full_barriers}, {empty}, {tile})', 1)}",
            f"{tile} = {tile} + 1",
        ]
    elif in_place:
        # ``name`` joins the loop-carried accumulator's rename group, so before
        # the loop it names the accumulator's initial value: a zero fill the
        # loop's first MMA overwrites instead (Triton's OptimizeAccumulatorInit).
        prefix.append(f"{tm}, {barrier} = {ALIAS}.acc_alloc({M}, {N})")
        if fresh_accumulator(device_function, node):
            fresh = device_function.new_var("tmem_fresh", dce=True)
            prefix.append(f"{fresh} = 1")
        else:
            fresh = "0"
            prefix.append(f"{ALIAS}.acc_put({tm}, {view(name)})")
        suffix = [f"{name} = {view(f'{ALIAS}.acc_read({tm}, {barrier}, {sync})', 1)}"]
        if lazy and own:  # (its MMAs completed: the accumulator's commit followed)
            suffix.append(f"{ALIAS}.acc_free({own[0]})")
    else:
        fresh = "0" if accumulator is not None else "1"
        prefix += [f"{tm}, {barrier} = {ALIAS}.acc_alloc({M}, {N})"]
        suffix = [f"{ALIAS}.acc_free({barrier})"]
        sync = True
    if loops:
        loops[-1].outer_prefix.extend(map(statement_from_string, prefix))
        loops[-1].outer_suffix.extend(map(statement_from_string, suffix))
    else:
        for statement in prefix:
            cg.add_statement(statement_from_string(statement))
    if not in_place and accumulator is not None:
        cg.add_statement(
            statement_from_string(
                f"{ALIAS}.acc_put({tm}, {view('{acc}')})",
                acc=cast("ast.AST", accumulator),
            )
        )
    tmem = f"{tm}, {fresh}, {done}, {count}, {depth}"
    if node.target is dot_scaled:
        a_scale, b_scale = _SCALES[dot_scaled]
        a_format, b_format = (repr(node.args[i]) for i in _FORMATS)
        call = expr_from_string(
            f"{ALIAS}.tmem_dot_scaled({{a}}, {{a_scale}}, {a_format}, "
            f"{{b}}, {{b_scale}}, {b_format}, {tmem}, "
            f"{in_smem[a_scale] and copied_scale(node, a_scale)}, "
            f"{in_smem[b_scale] and copied_scale(node, b_scale)}, {sync}, "
            f"{ring is not None and _scaled_dots(node.graph) == 1})",
            a=operand_asts[a_pos],
            a_scale=operand_asts[a_scale],
            b=operand_asts[b_pos],
            b_scale=operand_asts[b_scale],
        )
    else:
        call = expr_from_string(
            f"{ALIAS}.tmem_dot({{a}}, {{b}}, {tmem}, {sync}"
            f"{f', {own[0]}' if beside else ''})",
            a=operand_asts[a_pos],
            b=operand_asts[b_pos],
        )
    cg.add_statement(create(ast.Expr, value=call))
    for counter in (count, own[1]) if beside else (count,):
        cg.add_statement(statement_from_string(f"{counter} = {counter} + 1"))
    if fresh not in ("0", "1"):
        cg.add_statement(statement_from_string(f"{fresh} = 0"))
    if in_place:
        # The loop carries the accumulator unchanged; it lives in tensor memory.
        cg.add_statement(
            statement_from_string(f"{name} = {{acc}}", acc=cast("ast.AST", accumulator))
        )
        return create(ast.Name, id=name, ctx=ast.Load())
    result = view(f"{ALIAS}.acc_get({tm})", 1)
    if (dtype := node.meta["val"].dtype) != torch.float32:
        result = f"{result}.to({CompileEnvironment.current().backend.dtype_str(dtype)})"
    cg.add_statement(statement_from_string(f"{name} = {result}"))
    if not loops:
        for statement in suffix:
            cg.add_statement(statement_from_string(statement))
    return create(ast.Name, id=name, ctx=ast.Load())


def _scaled_dots(graph: torch.fx.Graph) -> int:
    return sum(n.target is dot_scaled for n in graph.nodes)


def _stage_call(
    x: ast.AST, transposed: bool, shape: str | None, completion: tuple[str, ...]
) -> ast.AST:
    args = [str(transposed), *([shape] if shape else [])]
    if completion:
        args = [str(transposed), shape or "None", "False", *completion]
    return expr_from_string(f"{ALIAS}.stage({{x}}, {', '.join(args)})", x=x)


def _staged(
    interp: GraphInterpreter,
    node: torch.fx.Node,
    lhs: bool,
    squeeze: bool,
    k_pad: int | None,
    completion: tuple[str, ...],
) -> ast.AST:
    """A register operand staged K-major, as Triton stages tcgen05 operands:
    before its transposes, or (``squeeze``, ``k_pad``) as its value (``stage``
    squeezes the unit batch), K zero-padded to ``k_pad`` as
    ``emit_tl_dot_with_padding`` does."""
    shape = cast("torch.Tensor", node.meta["val"]).shape
    if squeeze or k_pad:
        value = cast("ast.AST", interp.env[node])
        if k_pad:
            dims = concrete(shape)[squeeze:]
            if squeeze:
                value = expr_from_string(f"tl.reshape({{x}}, {dims})", x=value)
            axis = 1 if lhs else 0
            value = _pad_tensor(value, axis, dims[axis], k_pad, dims[1 - axis])
        return _stage_call(value, not lhs, None, completion)
    orders: list[list[int]] = []
    while node.op == "call_function" and node.target in _TRANSPOSES:
        orders.append(_permutation(node))
        node = cast("torch.fx.Node", node.args[0])
    shape = cast("torch.Tensor", node.meta["val"]).shape
    rank = len(shape)
    view = list(range(rank))
    for order in reversed(orders):
        view = [view[d] for d in order]
    source, broadcast = node, None
    if (
        node.target is aten.expand.default
        and cast("torch.Tensor", node.args[0].meta["val"]).ndim == rank  # pyrefly: ignore [missing-attribute]
    ):
        device_function = cast("GenerateAST", interp.cg).device_function
        broadcast = _tuple(device_function.tile_strategy.shape_dims(list(shape)))
        source = cast("torch.fx.Node", node.args[0])
    value = _stage_call(
        cast("ast.AST", interp.env[source]),
        _transposed(lhs, view),
        broadcast,
        completion,
    )
    for order in reversed(orders):
        value = permute(value, order)
    return value
