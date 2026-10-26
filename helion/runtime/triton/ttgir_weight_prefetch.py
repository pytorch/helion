"""Prefetch weights into shared memory across megakernel readiness waits.

Triton pipelines a K loop by issuing its first stages' ``cp.async`` or TMA
copies in a prologue right before the loop.  In a cross-loop megakernel a task that waits
for another root's output spins right before that loop, so the prologue, and
with it every copy of a weight that no task writes, is issued only after the
spin.  This pass moves those copies, with the pure index and allocation ops
that feed them, above the spin.  Copies of tensors that the kernel writes
(other roots' outputs) stay after it.

The pass runs on the TritonGPU IR after Triton's own pipeline, installed
through ``triton.knobs.runtime.add_stages_inspection_hook`` only while Helion
compiles a kernel that asks for it.  Shared memory allocation and barrier
insertion still run afterwards, so buffer lifetimes and barriers are computed
for the new order.  Every ``ttg.async_commit_group`` stays in place: a moved
copy joins the first group committed after its new position, later groups keep
their order and count, and every ``ttg.async_wait`` waits for at least the
copies it waited for before.  A moved TMA copy takes its stage barrier's
``barrier_expect`` (and the barrier's initialization) along; the stage's other
copies still arrive on that barrier after the wait, so the phase completes
only when every byte it expects has landed.

Copies move only when it is safe without knowing the schedule:

* the copy's pointer derives from a kernel argument through which no op of the
  kernel (or of a callee) stores, atomically updates, or runs an external
  function or inline assembly that may write memory.  Inline assembly is not
  read as PTX: an op is recognized only if its template (with MLIR's string
  escapes decoded, nothing else), constraint string, purity, packing and
  operand count are exactly those of an entry of :data:`_ALLOWLIST`, the
  assembly that Helion's code generator and Triton's language helpers emit
  (plus a few measurement and kernel idioms, each listed with its source).
  An allowlisted op writes nothing, or (a stamp) only through one pointer
  operand, which then writes that operand's argument.  Any other assembly,
  and any external function with side effects, may load or build any
  address, so it writes every argument; arguments that may share a storage
  (``alias_classes``, the launch's pointer and descriptor views of one
  tensor) are written together;
* everything between the spin and the copy is pure, an allocation or view, a
  ``cp.async`` copy or commit, a tensor-memory or barrier initialization, or
  an op that uses shared memory (an ``async_wait``, an MMA, a barrier wait, a
  shared load or store) none of whose operands may be the buffer slot the copy
  writes; a copy after any other op stays in place;
* the spin is the wait Helion emits for in-kernel producers (its allowlisted
  ``ld.acquire.gpu`` polls, or ``ld.relaxed.gpu`` polls of several counters
  with one acquire fence after them), the function has no system-scope
  operation and no integer-to-pointer cast (peer memory written by other
  devices), and no op of the module calls it (a caller may pass one pointer
  twice);
* in a module that may wait on its PDL predecessor (Triton's ``gdc_wait``,
  any assembly off the allowlist, an external function with side effects, or
  any other op mentioning ``griddepcontrol.wait``), Triton's ``gdc_wait``
  comes before the spin in the same or an enclosing block, in a module
  whose assembly is all allowlisted (other assembly may branch around it): a
  task may otherwise read the predecessor's output only because the producer
  it waits for did wait, and a read moved above the spin could run before
  the predecessor finished.

The pass reads the printed module and re-parses it.  Anything it does not
recognize leaves the module unchanged.

With the kernel's ``ttgir_weight_ring`` setting (on by default; the launch
passes it as ``_ttgir_weight_ring``, so it is part of the compile key), the
warp-specialized weight ring of :mod:`.ttgir_weight_ring` comes first: each
compile tries the two-producer ring, the weight-only ring, this hoist and the
module unchanged, in that order (:func:`_candidates`); a candidate that a
later stage rejects (``make_llir``, ``ptxas``) or whose binary spills falls
back to the next one (:class:`_Fallback`), and a ring binary whose code
exceeds the instruction-cache budget moves on to its smaller twins.  The ring
reuses this module's analyses (written arguments with their storage classes,
the assembly allowlist, the PDL rule), so it is never less conservative than
the hoist.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator

log: logging.Logger = logging.getLogger(__name__)

_VALUE = re.compile(r"%[A-Za-z0-9_$.#-]+")
_RESULTS = re.compile(r"^\s*((?:%[A-Za-z0-9_$.#-]+(?::\d+)?(?:,\s*)?)+)\s*=")
_OP_NAME = re.compile(r"=\s*\"?([a-z_]+\.[a-z_.0-9]+)")
_BARE_OP = re.compile(r"^\s*\"?([a-z_]+\.[a-z_.0-9]+)")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_TYPED_ARG = re.compile(r"(%[A-Za-z0-9_$.-]+)\s*:(?!\d)")
_BOUND_ARG = re.compile(r"(%[A-Za-z0-9_$.-]+)\s*=(?!=)")
_FUNC = re.compile(r"^\s*tt\.func\b")
_FUNC_NAME = re.compile(r"@(\"[^\"]+\"|[A-Za-z0-9_$.]+)")

# Ops without side effects or memory access; they may move with a copy.
_PURE = frozenset(
    {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrsi",
        "arith.shrui",
        "arith.cmpi",
        "arith.select",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.index_cast",
        "arith.minsi",
        "arith.maxsi",
        "arith.minui",
        "arith.maxui",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.make_range",
        "tt.addptr",
        "tt.reshape",
        "tt.trans",
        "ttg.convert_layout",
    }
)
# Ops that create or view a shared memory buffer; they may move with a copy.
_BUFFER = frozenset({"ttg.local_alloc", "ttg.memdesc_index"})
# Ops a copy may move above if none of their memory operands may be the slot
# the copy writes: they read, write, or wait for shared memory, or only order
# the copies (an ``async_wait`` then also waits for the moved copy).
_SLOT_USERS = frozenset(
    {
        "ttg.async_wait",
        "ttg.local_load",
        "ttg.local_store",
        "ttng.tc_gen5_mma",
        "ttng.tc_gen5_mma_scaled",
        "ttng.tc_gen5_commit",
        "ttng.warp_group_dot",
        "ttng.warp_group_dot_wait",
        "ttng.wait_barrier",
        "ttng.arrive_barrier",
        "ttng.barrier_expect",
    }
)
_VIEWS = frozenset(
    {"ttg.memdesc_trans", "ttg.memdesc_subslice", "ttg.memdesc_reinterpret"}
)
# Ops a copy may move above, besides the above: other copies and commits, views
# of buffers, and initialization of tensor memory and barriers, none of which
# touch a buffer a copy writes.
_PASSABLE = (
    _PURE
    | _BUFFER
    | _VIEWS
    | frozenset(
        {
            "ttg.async_copy_global_to_local",
            "ttg.async_commit_group",
            "ttng.async_tma_copy_global_to_local",
            "ttng.tmem_alloc",
            "ttng.tmem_store",
            "ttng.init_barrier",
        }
    )
)
# Ops through which a pointer flows from their first operand.  An integer made
# from a pointer still names that pointer's memory: inline assembly can store
# through it.
_POINTER_FLOW = frozenset(
    {
        "tt.addptr",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.reshape",
        "tt.trans",
        "tt.bitcast",
        "tt.ptr_to_int",
        "ttg.convert_layout",
        "tt.make_tensor_descriptor",
    }
)
# Ops that write global memory through their first operand.
_WRITES = frozenset(
    {
        "tt.store",
        "tt.atomic_rmw",
        "tt.atomic_cas",
        "tt.descriptor_store",
        "tt.descriptor_reduce",
        "tt.descriptor_scatter",
        "ttng.async_tma_copy_local_to_global",
        "ttng.async_tma_reduce",
        "ttng.async_tma_scatter",
    }
)
_COPY = "ttg.async_copy_global_to_local"
_TMA_COPY = "ttng.async_tma_copy_global_to_local"
_TMA_OPERANDS = re.compile(
    r"async_tma_copy_global_to_local (%[\w$.-]+)\[[^\]]*\]\s+(%[\w$.-]+),\s*(%[\w$.-]+),"
)
_ASM = "tt.elementwise_inline_asm"
_EXTERN = "tt.extern_elementwise"
_ASM_ESCAPE = re.compile(r"\\([0-9A-Fa-f]{2})")
_PURE_ATTRIBUTE = re.compile(r"\bpure\s*=\s*(true|false)\b")


def _decoded(text: str) -> str:
    # MLIR prints other characters (quote, newline, tab) as two hex digits.
    return _ASM_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _is_pure(line: str) -> bool:
    """Whether an op declares ``pure = true`` (in its attributes, not in a
    string); an op that does not say is not."""
    attribute = _PURE_ATTRIBUTE.search(_code(line))
    return attribute is not None and attribute.group(1) == "true"


# --- Inline assembly: an exact-template allowlist -------------------------
#
# The pass does not read PTX.  An inline assembly op counts as writing
# nothing (or as writing only through one operand) only if it is, exactly,
# one of the ops below: the same template once MLIR's string escapes are
# decoded, the same constraint string, purity, packing and operand count.
# Everything else writes every argument and may wait on the PDL predecessor.
# An entry missing here costs hoists, never soundness;
# ``test_ttgir_weight_prefetch.py`` fails when Helion's code generator emits
# assembly that is not listed.


@dataclasses.dataclass(frozen=True)
class _AsmTemplate:
    """One allowlisted inline assembly op and what it does."""

    template: str
    constraints: str
    pure: bool
    operands: int
    # Where it comes from (the code that emits it).
    source: str
    # "quiet": writes no memory and does not wait on the PDL predecessor.
    # "poll": quiet; the readiness poll Helion emits (a spin's seed).
    # "sync": quiet; the warp sync or fence Helion emits after a spin.
    # "pdl_wait": writes no memory; waits on the PDL predecessor, always.
    # "writes": writes memory only through operand ``written_operand``.
    kind: str = "quiet"
    written_operand: int | None = None
    packed: int = 1
    # Whether ``{int}`` in the template stands for any decimal immediate.
    holes: bool = False

    def key(self) -> tuple[str, str, bool, int, int]:
        return (self.template, self.constraints, self.pure, self.operands, self.packed)


_CROSS_LOOP = "helion/_compiler/cross_loop_codegen.py"
_DEVICE_FUNCTION = "helion/_compiler/device_function.py"
_QUANTIZED = "helion/_compiler/triton/quantized_ops.py"
_GDC = "triton/language/extra/cuda/gdc.py"
_CUDA_UTILS = "triton/language/extra/cuda/utils.py"
_SGLANG_NVFP4 = "sglang/kernels/ops/moe/glm5_nvfp4_verify_helion.py (GLM-5 MoE)"
_STAMP = (
    "GLM harness globaltimer stamps "
    "(glm53_runs/artifacts/smem_prefetch: attn_mk/build/vattn.py _gt, "
    "attn_v4/verify/vv.py, xrw/meas/v4/xattn.py)"
)
_READY_WAIT = (
    "{ .reg .pred p; .reg .b32 v; READY_WAIT: "
    "ld.acquire.gpu.global.u32 v, [$1]; setp.ne.u32 p, v, $2; "
    "@p bra READY_WAIT; bar.warp.sync 0xffffffff; "
)

_HELION_FP4X2_TO_F32 = (
    "\n"
    "{\n"
    "    .reg .b8 b0;\n"
    "    .reg .b16 lo, hi;\n"
    "    .reg .b32 h2;\n"
    "    mov.b16 {b0, _}, $2;\n"
    "    cvt.rn.f16x2.e2m1x2 h2, b0;\n"
    "    mov.b32 {lo, hi}, h2;\n"
    "    cvt.f32.f16 $0, lo;\n"
    "    cvt.f32.f16 $1, hi;\n"
    "}\n"
)

_HELION_FP4X16_TO_F16 = (
    "\n"
    "{\n"
    "    .reg .b32 lo, hi;\n"
    "    .reg .b8 c0,c1,c2,c3,c4,c5,c6,c7;\n"
    "    .reg .b32 h0,h1,h2,h3,h4,h5,h6,h7;\n"
    "    mov.b64 {lo, hi}, $16;\n"
    "    mov.b32 {c0,c1,c2,c3}, lo;\n"
    "    cvt.rn.f16x2.e2m1x2 h0, c0; cvt.rn.f16x2.e2m1x2 h1, c1;\n"
    "    cvt.rn.f16x2.e2m1x2 h2, c2; cvt.rn.f16x2.e2m1x2 h3, c3;\n"
    "    mov.b32 {c4,c5,c6,c7}, hi;\n"
    "    cvt.rn.f16x2.e2m1x2 h4, c4; cvt.rn.f16x2.e2m1x2 h5, c5;\n"
    "    cvt.rn.f16x2.e2m1x2 h6, c6; cvt.rn.f16x2.e2m1x2 h7, c7;\n"
    "    mov.b32 {$0,$1}, h0; mov.b32 {$2,$3}, h1;\n"
    "    mov.b32 {$4,$5}, h2; mov.b32 {$6,$7}, h3;\n"
    "    mov.b32 {$8,$9}, h4; mov.b32 {$10,$11}, h5;\n"
    "    mov.b32 {$12,$13}, h6; mov.b32 {$14,$15}, h7;\n"
    "}\n"
)

_HELION_BF16X16_TO_F16 = (
    "\n"
    "{\n"
    "    .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7;\n"
    "    .reg .b16 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;\n"
    "    mov.b64 {w0,w1}, $16; mov.b64 {w2,w3}, $17;\n"
    "    mov.b64 {w4,w5}, $18; mov.b64 {w6,w7}, $19;\n"
    "    mov.b32 {b0,b1}, w0; mov.b32 {b2,b3}, w1;\n"
    "    mov.b32 {b4,b5}, w2; mov.b32 {b6,b7}, w3;\n"
    "    mov.b32 {b8,b9}, w4; mov.b32 {b10,b11}, w5;\n"
    "    mov.b32 {b12,b13}, w6; mov.b32 {b14,b15}, w7;\n"
    "    cvt.rn.f16.bf16 $0, b0; cvt.rn.f16.bf16 $1, b1;\n"
    "    cvt.rn.f16.bf16 $2, b2; cvt.rn.f16.bf16 $3, b3;\n"
    "    cvt.rn.f16.bf16 $4, b4; cvt.rn.f16.bf16 $5, b5;\n"
    "    cvt.rn.f16.bf16 $6, b6; cvt.rn.f16.bf16 $7, b7;\n"
    "    cvt.rn.f16.bf16 $8, b8; cvt.rn.f16.bf16 $9, b9;\n"
    "    cvt.rn.f16.bf16 $10, b10; cvt.rn.f16.bf16 $11, b11;\n"
    "    cvt.rn.f16.bf16 $12, b12; cvt.rn.f16.bf16 $13, b13;\n"
    "    cvt.rn.f16.bf16 $14, b14; cvt.rn.f16.bf16 $15, b15;\n"
    "}\n"
)

_NVFP4_SCALE_GUARDED = (
    "\n"
    "        {\n"
    "          .reg .pred nonzero, scale_nonzero;\n"
    "          .reg .b16 fp8_pair;\n"
    "          .reg .b32 h2_32;\n"
    "          .reg .b16 h_lo, h_hi;\n"
    "          .reg .f32 encode_scale, block_scaled, scale_f;\n"
    "          .reg .f32 scale_f32, rcp_gs, product, result;\n"
    "          setp.ne.f32 nonzero, $2, 0f00000000;\n"
    "          @nonzero rcp.approx.ftz.f32 encode_scale, $2;\n"
    "          @!nonzero mov.b32 encode_scale, 0x7f7fffff;\n"
    "          mul.f32 block_scaled, $3, 0f3e2aaaab;\n"
    "          mul.f32 scale_f, encode_scale, block_scaled;\n"
    "          cvt.rn.satfinite.e4m3x2.f32 fp8_pair, 0f00000000, scale_f;\n"
    "          @!nonzero mov.b16 fp8_pair, 0x0000;\n"
    "          cvt.u32.u16 $0, fp8_pair;\n"
    "          cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
    "          mov.b32 {h_lo, h_hi}, h2_32;\n"
    "          cvt.f32.f16 scale_f32, h_lo;\n"
    "          rcp.approx.ftz.f32 rcp_gs, encode_scale;\n"
    "          mul.f32 product, scale_f32, rcp_gs;\n"
    "          rcp.approx.ftz.f32 result, product;\n"
    "          setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;\n"
    "          selp.f32 $1, result, 0f00000000, scale_nonzero;\n"
    "        }\n"
    "        "
)

_NVFP4_SCALE = (
    "\n"
    "        {\n"
    "          .reg .pred scale_nonzero;\n"
    "          .reg .b16 fp8_pair;\n"
    "          .reg .b32 h2_32;\n"
    "          .reg .b16 h_lo, h_hi;\n"
    "          .reg .f32 block_scaled, scale_f, scale_f32, rcp_gs, product, result;\n"
    "          mul.f32 block_scaled, $3, 0f3e2aaaab;\n"
    "          mul.f32 scale_f, $2, block_scaled;\n"
    "          cvt.rn.satfinite.e4m3x2.f32 fp8_pair, 0f00000000, scale_f;\n"
    "          cvt.u32.u16 $0, fp8_pair;\n"
    "          cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
    "          mov.b32 {h_lo, h_hi}, h2_32;\n"
    "          cvt.f32.f16 scale_f32, h_lo;\n"
    "          rcp.approx.ftz.f32 rcp_gs, $2;\n"
    "          mul.f32 product, scale_f32, rcp_gs;\n"
    "          rcp.approx.ftz.f32 result, product;\n"
    "          setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;\n"
    "          selp.f32 $1, result, 0f00000000, scale_nonzero;\n"
    "        }\n"
    "        "
)

_NVFP4_PACK_E2M1X2 = (
    "\n"
    "        {\n"
    "          .reg .b8 b;\n"
    "          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;\n"
    "          cvt.u32.u8 $0, b;\n"
    "        }\n"
    "        "
)

_ALLOWLIST: tuple[_AsmTemplate, ...] = (
    # Helion: readiness polls, the sync after them, the TMA proxy fences.
    _AsmTemplate(
        "ld.acquire.gpu.global.u32 $0, [$1];",
        "=r,l",
        False,
        1,
        f"{_CROSS_LOOP}: _wait_for_counter, _wait_for_dependencies "
        "(before sm_90), the persistent epoch read",
        "poll",
    ),
    _AsmTemplate(
        "ld.relaxed.gpu.global.u32 $0, [$1];",
        "=r,l",
        False,
        1,
        f"{_CROSS_LOOP}: _wait_for_dependencies (several counters, sm_90+)",
        "poll",
    ),
    _AsmTemplate(
        "bar.warp.sync 0xffffffff; mov.u32 $0, $1;",
        "=r,r",
        False,
        1,
        f"{_CROSS_LOOP}: _wait_for_counter, _wait_for_dependencies, _publication_sync",
        "sync",
    ),
    _AsmTemplate(
        "fence.acquire.gpu; bar.warp.sync 0xffffffff; mov.u32 $0, $1;",
        "=r,r",
        False,
        1,
        f"{_CROSS_LOOP}: _wait_for_dependencies (after relaxed polls)",
        "sync",
    ),
    _AsmTemplate(
        "fence.proxy.async.global; mov.u32 $0, $1;",
        "=r,r",
        False,
        1,
        f"{_DEVICE_FUNCTION}: DeviceFunction.async_load_fence",
        "sync",
    ),
    # Waits for this thread's bulk (TMA) stores, whose writes are the TMA
    # store ops' own.
    _AsmTemplate(
        "cp.async.bulk.wait_group 0; fence.proxy.async.global; mov.u32 $0, $1;",
        "=r,r",
        False,
        1,
        f"{_DEVICE_FUNCTION}: DeviceFunction.async_store_drain",
        "sync",
    ),
    _AsmTemplate(
        "fence.acq_rel.gpu; // $0",
        "=r",
        False,
        0,
        f"{_DEVICE_FUNCTION}: the release fence before an exit PDL trigger",
    ),
    _AsmTemplate(
        _READY_WAIT + "mov.u32 $0, 0; }",
        "=r,l,r",
        False,
        2,
        f"{_CROSS_LOOP}: the readiness spin of a warp-specialized task loop",
    ),
    _AsmTemplate(
        _READY_WAIT + "fence.proxy.async.global; mov.u32 $0, 0; }",
        "=r,l,r",
        False,
        2,
        f"{_CROSS_LOOP}: the same, before TMA global accesses",
    ),
    _AsmTemplate(
        "redux.sync.min.s32 $0, $1, 0xffffffff;",
        "=r,r",
        True,
        1,
        f"{_CROSS_LOOP}: guarded_live_tasks",
    ),
    _AsmTemplate(
        "bar.sync 0;",
        "=r",
        False,
        0,
        "helion/runtime/triton/helpers.py: triton_wait_signal",
    ),
    _AsmTemplate(
        _HELION_FP4X2_TO_F32,
        "=f,=f,h",
        True,
        1,
        f"{_QUANTIZED}: float4_e2m1fn_x2_to_float32",
    ),
    _AsmTemplate(
        _HELION_FP4X16_TO_F16,
        ",".join(["=h"] * 16 + ["l"]),
        True,
        1,
        f"{_QUANTIZED}: load_float4_e2m1fn_x16_to_float16",
    ),
    _AsmTemplate(
        _HELION_BF16X16_TO_F16,
        ",".join(["=h"] * 16 + ["l"] * 4),
        True,
        4,
        f"{_QUANTIZED}: load_bfloat16_x16_to_float16",
    ),
    # Triton's language helpers.  Helion emits gdc_wait / gdc_launch_dependents
    # for PDL entry and exit (device_function.py).
    _AsmTemplate(
        "griddepcontrol.wait; // dummy $0",
        "=r",
        False,
        0,
        f"{_GDC}: gdc_wait",
        "pdl_wait",
    ),
    _AsmTemplate(
        "griddepcontrol.launch_dependents; // dummy $0",
        "=r",
        False,
        0,
        f"{_GDC}: gdc_launch_dependents",
    ),
    _AsmTemplate(
        "mov.u64 $0, %globaltimer;",
        "=l",
        False,
        0,
        f"{_CUDA_UTILS}: globaltimer; {_STAMP} and their delay spins",
    ),
    _AsmTemplate("mov.u32 $0, %smid;", "=r", True, 0, f"{_CUDA_UTILS}: smid"),
    # Measurement idioms of the GLM harnesses (stamps store through tl.store).
    _AsmTemplate(
        "mov.u32 $0, %smid;",
        "=r",
        False,
        0,
        "GLM harness smid stamp (attn_mk/build/vattn.py tx_smid)",
    ),
    *(
        _AsmTemplate(
            "mov.u64 $0, %globaltimer; // $1",
            f"=l,{operand}",
            False,
            1,
            "GLM MoE harness stamps ordered after a value (moe_v3 traced kernels)",
        )
        for operand in ("f", "l", "r")
    ),
    _AsmTemplate(
        "{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$1], t; "
        "mov.u32 $0, 0; }",
        "=r,l",
        False,
        1,
        "GLM harness stamp stored by the assembly (ring/ttgir/*_stamped_code.py)",
        "writes",
        written_operand=0,
    ),
    _AsmTemplate(
        "{ .reg .pred p; .reg .u32 t; .reg .u64 a; mov.u32 t, %tid.x; "
        "setp.lt.u32 p, t, {int}; mul.wide.u32 a, t, {int}; add.u64 a, a, $1; "
        "@p cp.async.bulk.prefetch.L2.global [a], {int}; mov.u32 $0, 0; }",
        "=r,l",
        False,
        1,
        "GLM harness L2 weight prefetch (attn_mk/build/vattn.py _bulk_pf)",
        holes=True,
    ),
    # Register-only NVFP4 quantization of the GLM-5 MoE kernels.
    _AsmTemplate(_NVFP4_SCALE_GUARDED, "=r,=f,f,f", True, 2, _SGLANG_NVFP4),
    _AsmTemplate(_NVFP4_SCALE, "=r,=f,f,f", True, 2, _SGLANG_NVFP4),
    _AsmTemplate(_NVFP4_PACK_E2M1X2, "=r,f,f", True, 2, _SGLANG_NVFP4),
)
_EXACT: dict[tuple[str, str, bool, int, int], _AsmTemplate] = {
    entry.key(): entry for entry in _ALLOWLIST if not entry.holes
}
_FAMILIES: list[tuple[re.Pattern[str], _AsmTemplate]] = [
    (
        re.compile("[0-9]+".join(map(re.escape, entry.template.split("{int}")))),
        entry,
    )
    for entry in _ALLOWLIST
    if entry.holes
]


def _mlir_string(text: str) -> str | None:
    """The value of an MLIR string literal's body, decoded as MLIR's lexer
    does (``\\"``, ``\\\\``, ``\\n``, ``\\t`` and two hex digits), or None if
    it is not well formed."""
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        escape = text[i + 1 : i + 2]
        if escape in ('"', "\\"):
            out.append(escape)
            i += 2
        elif escape == "n":
            out.append("\n")
            i += 2
        elif escape == "t":
            out.append("\t")
            i += 2
        elif re.fullmatch(r"[0-9A-Fa-f]{2}", text[i + 1 : i + 3]):
            out.append(chr(int(text[i + 1 : i + 3], 16)))
            i += 3
        else:
            return None
    return "".join(out)


@dataclasses.dataclass(frozen=True)
class _InlineAsm:
    template: str
    constraints: str
    pure: bool
    packed: int
    operands: tuple[str, ...]


_ASM_HEAD = re.compile(
    r"^\s*(?:%[\w$.#-]+(?::\d+)?(?:\s*,\s*%[\w$.#-]+(?::\d+)?)*\s*=\s*)?"
    r'tt\.elementwise_inline_asm\s+"((?:[^"\\]|\\.)*)"\s*\{'
)
_ASM_ATTRIBUTES = {
    "constraints": re.compile(r'"((?:[^"\\]|\\.)*)"'),
    "packed_element": re.compile(r"(\d+)\s*:\s*i32"),
    "pure": re.compile(r"(true|false)"),
}
_ASM_OPERANDS = re.compile(r"\s*((?:%[\w$.#-]+)(?:\s*,\s*%[\w$.#-]+)*)?\s*(?::|->)")


def _parse_asm(line: str) -> _InlineAsm | None:
    """An inline assembly op printed in Triton's custom form, or None if
    ``line`` is not exactly that (generic form, other attributes, ...)."""
    head = _ASM_HEAD.match(line)
    if head is None:
        return None
    template = _mlir_string(head.group(1))
    # The attribute dictionary, up to its closing brace (outside strings).
    body, k, quoted = [], head.end(), False
    while k < len(line) and (quoted or line[k] != "}"):
        if line[k] == "\\" and quoted:
            body.append(line[k : k + 2])
            k += 2
            continue
        quoted ^= line[k] == '"'
        body.append(line[k])
        k += 1
    if k == len(line):
        return None
    attributes: dict[str, str] = {}
    for item in re.split(r",\s*(?=[a-z_]+\s*=)", "".join(body).strip()):
        name, sep, value = item.partition("=")
        name, value = name.strip(), value.strip()
        pattern = _ASM_ATTRIBUTES.get(name)
        found = pattern.fullmatch(value) if sep and pattern else None
        if found is None or name in attributes:
            return None
        attributes[name] = found.group(1)
    operands = _ASM_OPERANDS.match(line, k + 1)
    constraints = _mlir_string(attributes.get("constraints", "\\"))
    if (
        template is None
        or constraints is None
        or operands is None
        or set(attributes) != set(_ASM_ATTRIBUTES)
    ):
        return None
    return _InlineAsm(
        template,
        constraints,
        attributes["pure"] == "true",
        int(attributes["packed_element"]),
        tuple(re.split(r"\s*,\s*", operands.group(1))) if operands.group(1) else (),
    )


def _allowlisted(asm: _InlineAsm) -> _AsmTemplate | None:
    entry = _EXACT.get(
        (asm.template, asm.constraints, asm.pure, len(asm.operands), asm.packed)
    )
    if entry is not None:
        return entry
    for pattern, family in _FAMILIES:
        if (
            pattern.fullmatch(asm.template)
            and (asm.constraints, asm.pure, len(asm.operands), asm.packed)
            == family.key()[1:]
        ):
            return family
    return None


@functools.lru_cache(maxsize=1 << 14)
def _asm_entry(line: str) -> _AsmTemplate | None:
    """The allowlist entry an inline assembly op is, or None."""
    asm = _parse_asm(line)
    return None if asm is None else _allowlisted(asm)


def _asm_kind(line: str) -> str | None:
    entry = _asm_entry(line) if _op(line) == _ASM else None
    return entry.kind if entry is not None else None


def _asm_may_write(line: str) -> bool:
    """Whether an inline assembly op may write memory: any op off the
    allowlist (which writes every argument), or an allowlisted stamp."""
    entry = _asm_entry(line)
    return entry is None or entry.kind == "writes"


def _asm_targets(line: str) -> list[str] | None:
    """The values an inline assembly op writes through, or None if it may
    write through any address (an op off the allowlist)."""
    asm = _parse_asm(line)
    entry = None if asm is None else _allowlisted(asm)
    if asm is None or entry is None:
        return None
    if entry.written_operand is None:
        return []
    return [asm.operands[entry.written_operand].split("#")[0]]


@dataclasses.dataclass
class Report:
    """What :func:`rewrite` did, for logs and tests."""

    hoisted: int = 0
    sites: int = 0
    skipped: list[str] = dataclasses.field(default_factory=list)
    # The arguments (by source name where known) of the functions the pass
    # considered that some op may write.
    written: list[str] = dataclasses.field(default_factory=list)
    # Inline assembly off the allowlist (it writes every argument).
    foreign_asm: list[str] = dataclasses.field(default_factory=list)


def _code(line: str) -> str:
    """``line`` without its trailing location, string literals blanked."""
    body = _STRING.sub(lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', line).rstrip()
    if body.endswith(")"):
        depth = 0
        for k in range(len(body) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(body[k], 0)
            if depth == 0:
                if body[:k].endswith(" loc"):
                    return body[: k - 4]
                break
    return body


def _results(line: str) -> list[str]:
    m = _RESULTS.match(line)
    if not m:
        return []
    return [tok.strip().split(":")[0] for tok in m.group(1).split(",") if tok.strip()]


def _op(line: str) -> str | None:
    code = _code(line)
    m = _OP_NAME.search(code) if _RESULTS.match(code) else None
    if m is None:
        m = _BARE_OP.match(code)
    return m.group(1) if m else None


def _uses(line: str) -> list[str]:
    code = _code(line)
    if _RESULTS.match(code):
        code = code.split("=", 1)[1]
    return [v.split("#")[0] for v in _VALUE.findall(code)]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _uniquify(text: str) -> str:
    """Give every SSA value a unique name.

    The MLIR printer reuses names across sibling regions (the bodies of two
    ``scf.if`` ops may both define ``%18``), so each use resolves through the
    stack of enclosing regions.
    """
    scopes: list[dict[str, str]] = [{}]
    counter = 0
    out = []

    def define(name: str, scope: dict[str, str]) -> str:
        nonlocal counter
        counter += 1
        scope[name] = f"%u{counter}_{name[1:]}"
        return scope[name]

    def resolve(name: str) -> str:
        base, _, index = name.partition("#")
        for scope in reversed(scopes):
            if base in scope:
                return scope[base] + (f"#{index}" if index else "")
        return name

    for line in text.splitlines():
        code = _code(line)
        stripped = code.strip()
        closes = stripped.startswith("}")
        opens = stripped.endswith("{")
        if closes:
            scopes.pop()
        results = _RESULTS.match(code)
        head_at = results.end() if results else 0
        head = code[head_at:]
        # Values a line binds for the region it opens: loop induction variables
        # and iter_args, while-loop arguments, function and partition
        # parameters.  Block labels bind their arguments in the current region.
        region_args: list[re.Match[str]] = []
        block_args: list[re.Match[str]] = []
        if stripped.startswith("^"):
            block_args = list(_TYPED_ARG.finditer(head))
        elif opens:
            region_args = list(_BOUND_ARG.finditer(head))
            if _FUNC.match(code) or "partition" in head:
                region_args += _TYPED_ARG.finditer(head)
        bound = {(m.start(1), m.end(1)) for m in region_args + block_args}
        renames: dict[tuple[int, int], str] = {}
        for m in _VALUE.finditer(head):
            if (m.start(), m.end()) not in bound:
                renames[(head_at + m.start(), head_at + m.end())] = resolve(m.group(0))
        for m in block_args:
            renames[(head_at + m.start(1), head_at + m.end(1))] = define(
                m.group(1), scopes[-1]
            )
        if results:
            for m in _VALUE.finditer(results.group(1)):
                at = results.start(1)
                renames[(at + m.start(), at + m.end())] = define(m.group(0), scopes[-1])
        if opens:
            scopes.append({})
        for m in region_args:
            renames[(head_at + m.start(1), head_at + m.end(1))] = define(
                m.group(1), scopes[-1]
            )
        new = line
        for (a, b), name in sorted(renames.items(), reverse=True):
            new = new[:a] + name + new[b:]
        out.append(new)
    if len(scopes) != 1:
        raise ValueError("unbalanced regions")
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


@dataclasses.dataclass
class _Function:
    name: str
    start: int
    end: int  # exclusive
    params: list[str]
    # Each parameter's source name (its ``loc("name")``), None if unknown.
    param_names: list[str | None]
    roots: dict[str, frozenset[str]]
    defs: dict[str, int]  # value -> line defining it
    # Parameters that are not pointers or descriptors (integers, floats): an
    # address carried in one names no argument's storage.
    scalars: frozenset[str] = frozenset()


_LOC_NAME = re.compile(r'loc\("((?:[^"\\]|\\.)*)"')
_LOC_ALIAS = re.compile(r"loc\((#[\w$.-]+)\)")
_LOC_DEFINITION = re.compile(r'^\s*(#[\w$.-]+)\s*=\s*loc\("((?:[^"\\]|\\.)*)"')


def _param_names(
    line: str, code: str, aliases: dict[str, str]
) -> tuple[list[str], list[str | None]]:
    """The parameters of the function a ``tt.func`` line opens, and the
    source name each carries in its location (``%x: ... loc("x")``, directly
    or through a ``#loc`` alias)."""
    start = code.index("(", code.index("@"))
    depth, close = 0, len(code)
    for k in range(start, len(code)):
        depth += {"(": 1, ")": -1}.get(code[k], 0)
        if depth == 0:
            close = k
            break
    found = list(_TYPED_ARG.finditer(code, start, close))
    params, names = [], []
    _param_names.scalars = {  # type: ignore[attr-defined]
        m.group(1)
        for m in found
        if re.match(r"\s*(i\d+|f\d+|bf16|f16|index)\b", code[m.end() :])
    }
    for k, m in enumerate(found):
        # ``code`` blanks strings in place, so positions index ``line`` too.
        segment = line[m.end() : found[k + 1].start() if k + 1 < len(found) else close]
        name = _LOC_NAME.search(segment)
        alias = _LOC_ALIAS.search(segment)
        params.append(m.group(1))
        names.append(
            _decoded(name.group(1))
            if name
            else aliases.get(alias.group(1))
            if alias
            else None
        )
    return params, names


def _functions(lines: list[str]) -> list[_Function]:
    aliases = {
        m.group(1): _decoded(m.group(2))
        for line in lines
        if line.startswith("#")
        for m in [_LOC_DEFINITION.match(line)]
        if m
    }
    funcs = []
    for i, line in enumerate(lines):
        if not _FUNC.match(line):
            continue
        code = _code(line)
        name = _FUNC_NAME.search(code)
        assert name is not None
        params, param_names = _param_names(line, code, aliases)
        scalars = frozenset(_param_names.scalars)  # type: ignore[attr-defined]
        depth, j = 0, i
        while j < len(lines):
            stripped = _code(lines[j]).strip()
            depth += stripped.endswith("{") - stripped.startswith("}")
            if depth == 0:
                break
            j += 1
        body = lines[i + 1 : j]
        defs = {r: n for n in range(i + 1, j) for r in _results(lines[n])}
        funcs.append(
            _Function(
                name.group(1).strip('"'),
                i,
                j + 1,
                params,
                param_names,
                _pointer_roots(body),
                defs,
                scalars,
            )
        )
    return funcs


def _pointer_roots(body: list[str]) -> dict[str, frozenset[str]]:
    """Value -> the values its pointer derives from.

    A value produced by anything but pointer arithmetic is its own root, so a
    pointer whose roots are all parameters points into those arguments.
    """
    roots: dict[str, frozenset[str]] = {}
    # Integers derived from a pointer's address (tt.ptr_to_int, then arithmetic):
    # an offset made of them may reach another argument's storage.
    tainted: set[str] = set()
    # Device-built TMA descriptors: tensormap_create writes a tensormap of its
    # base pointer into kernel-private scratch, which reinterpret_tensor_descriptor
    # turns into the descriptor.
    described: dict[str, frozenset[str]] = {}
    for line in body:
        results = _results(line)
        op, uses = _op(line), _uses(line)
        if op == "ttng.tensormap_create" and len(uses) >= 2:
            described[uses[0]] = roots.get(uses[1], frozenset({uses[1]}))
        if not results:
            continue
        if op == "tt.ptr_to_int" or (op != "tt.addptr" and any(u in tainted for u in uses)):
            tainted.update(results)
        if op == "tt.addptr" and any(u in tainted for u in uses[1:]):
            src = frozenset({results[0]})
        elif op == "ttng.reinterpret_tensor_descriptor" and uses:
            src = described.get(uses[0], frozenset({results[0]}))
        elif op in _POINTER_FLOW and uses:
            src = roots.get(uses[0], frozenset({uses[0]}))
        else:
            src = frozenset({results[0]})
        for r in results:
            roots[r] = src
    return roots


def _op_write_hits(
    line: str,
    f: _Function,
    by_name: dict[str, _Function],
    written: dict[str, set[int]],
    wild: set[str],
) -> set[int] | None:
    """Indices of ``f``'s parameters one op of ``f`` may write through (before
    storage classes), or None if it may write any argument.  ``written`` and
    ``wild``: the callee summaries of :func:`_write_analysis`."""
    op = _op(line)
    targets: list[str] = []
    if op in _WRITES:
        targets = _uses(line)[:1]
    elif op == "tt.call":
        callee = _FUNC_NAME.search(_code(line))
        name = callee.group(1).strip('"') if callee else ""
        if name not in by_name or name in wild:
            # A function outside the module may write anything; so may one
            # that writes through an address it builds.
            return None
        args = _uses(line)
        targets = [a for k, a in enumerate(args) if k in written[name]]
    elif op == _ASM:
        asm_targets = _asm_targets(line)
        if asm_targets is None:
            # Assembly off the allowlist may load or build the address it
            # stores through: it may write any argument.
            return None
        targets = asm_targets
    elif op == _EXTERN and not _is_pure(line):
        # So may an external function with side effects.
        return None
    hit: set[int] = set()
    for t in targets:
        roots = f.roots.get(t, frozenset({t}))
        if not all(r in f.params and r not in f.scalars for r in roots):
            # An unattributable target may be any argument.
            return None
        hit |= {f.params.index(r) for r in roots}
    return hit


def _write_analysis(
    lines: list[str],
    funcs: list[_Function],
    classes: dict[str, list[set[int]]] | None = None,
) -> tuple[dict[str, set[int]], set[str]]:
    """(:func:`_written_params`, the functions that may write any argument).

    ``classes``: per function, sets of parameters that may share a storage;
    writing one member writes them all.
    """
    by_name = {f.name: f for f in funcs}
    written: dict[str, set[int]] = {f.name: set() for f in funcs}
    # Functions that may write memory no parameter of theirs names.
    wild: set[str] = {
        f.name
        for f in funcs
        if any(_op(line) == "tt.int_to_ptr" for line in lines[f.start + 1 : f.end])
    }
    changed = True
    while changed:
        changed = False
        for f in funcs:
            everything = set(range(len(f.params)))
            hit: set[int] = set()
            for line in lines[f.start + 1 : f.end]:
                op_hit = _op_write_hits(line, f, by_name, written, wild)
                if op_hit is None:
                    hit = everything
                    break
                hit |= op_hit
                if hit == everything:
                    break
            if hit == everything and f.name not in wild:
                wild.add(f.name)
                changed = True
            for members in (classes or {}).get(f.name, ()):
                if hit & members:
                    hit |= members
            if not hit <= written[f.name]:
                written[f.name] |= hit
                changed = True
    return written, wild


def _written_params(
    lines: list[str],
    funcs: list[_Function],
    classes: dict[str, list[set[int]]] | None = None,
) -> dict[str, set[int]]:
    """Function name -> indices of the pointer parameters it may write through.

    ``classes``: per function, sets of parameters that may share a storage;
    writing one member writes them all.
    """
    return _write_analysis(lines, funcs, classes)[0]


def _unsupported(lines: list[str], f: _Function) -> str | None:
    for line in lines[f.start + 1 : f.end]:
        op = _op(line)
        if op == "tt.int_to_ptr":
            return "integer-to-pointer cast"
        if (
            op == _ASM
            and ".sys" in line
            or (
                op in ("tt.atomic_rmw", "tt.atomic_cas")
                and re.search(r",\s*sys\b", _code(line))
            )
        ):
            return "system-scope operation"
    return None


def _is_seed(line: str) -> bool:
    return _asm_kind(line) == "poll" and len(_results(line)) == 1


def _is_sync(line: str) -> bool:
    return _asm_kind(line) == "sync"


def _region_end(lines: list[str], i: int, end: int) -> int:
    """Index of the line closing the region opened on line ``i``."""
    depth, j = 0, i
    while j < end:
        stripped = _code(lines[j]).strip()
        depth += stripped.endswith("{") - stripped.startswith("}")
        if depth == 0:
            return j
        j += 1
    raise ValueError("unbalanced regions")


def _wait_at(lines: list[str], i: int, end: int) -> int | None:
    """If line ``i`` seeds a readiness spin, the index just past its sync."""
    if not _is_seed(lines[i]):
        return None
    ind = _indent(lines[i])
    # One spin may poll several counters: seeds (and their targets) then a
    # single scf.while carrying the polled values.
    seeds = {_results(lines[i])[0]}
    j = i + 1
    while j < end and _indent(lines[j]) == ind:
        if _is_seed(lines[j]):
            seeds.add(_results(lines[j])[0])
        elif _op(lines[j]) not in _PURE:
            break
        j += 1
    if not (
        j < end
        and _indent(lines[j]) == ind
        and re.match(r"^\s*(%\S+\s*=\s*)?scf\.while\b", lines[j])
        and seeds <= set(_uses(lines[j]))
    ):
        return None
    close = _region_end(lines, j, end)
    if not any(_is_seed(lines[k]) for k in range(j + 1, close)):
        return None
    k, synced = close + 1, None
    while k < end and _indent(lines[k]) == ind:
        if _is_sync(lines[k]):
            synced = k = k + 1
        elif _op(lines[k]) in ("tt.make_range", "arith.constant"):
            k += 1
        else:
            break
    return synced


_UNKNOWN = ("?", None)


def _slot(
    lines: list[str], defs: dict[str, int], value: str
) -> tuple[str, int | None] | None:
    """(buffer, constant slot or None for any) a memory descriptor names, None
    for a value that is not one, ``_UNKNOWN`` if that cannot be told."""
    d = defs.get(value)
    if d is None:
        return _UNKNOWN
    op, uses = _op(lines[d]), _uses(lines[d])
    if op in ("ttg.local_alloc", "ttng.tmem_alloc"):
        return (value, None)
    if op in _VIEWS:
        return _slot(lines, defs, uses[0])
    if op == "ttg.memdesc_index":
        base = _slot(lines, defs, uses[0])
        if base is None or base is _UNKNOWN or base[1] is not None:
            return _UNKNOWN
        index = defs.get(uses[1])
        m = (
            re.search(r"arith\.constant (-?\d+) :", lines[index])
            if index is not None
            else None
        )
        return (base[0], int(m.group(1)) if m else None)
    if op is not None and op.startswith(("ttg.", "ttng.")) and "memdesc" in lines[d]:
        return _UNKNOWN
    return None


def _conflicts(
    used: tuple[str, int | None] | None, written: tuple[str, int | None] | None
) -> bool:
    if used is None:
        return False
    if used is _UNKNOWN or written is None or written is _UNKNOWN:
        return True
    return used[0] == written[0] and (
        used[1] is None or written[1] is None or used[1] == written[1]
    )


def _tma_stage(
    lines: list[str], f: _Function, window: list[int], copy: int, barrier: str
) -> list[int] | None:
    """The ``barrier_expect`` (and any ``init_barrier``) of a TMA copy's stage
    barrier, which move with the copy; None if anything else before the expect
    uses that barrier (an earlier phase of a reused barrier)."""
    slot = _slot(lines, f.defs, barrier)
    if slot is None or slot is _UNKNOWN or slot[1] is None:
        return None

    def on_barrier(i: int) -> bool:
        return any(_conflicts(_slot(lines, f.defs, u), slot) for u in _uses(lines[i]))

    expects = [
        i
        for i in window
        if i < copy and _op(lines[i]) == "ttng.barrier_expect" and on_barrier(i)
    ]
    if not expects:
        return None
    inits = [
        i
        for i in window
        if i < expects[-1] and _op(lines[i]) == "ttng.init_barrier" and on_barrier(i)
    ]
    if any(
        on_barrier(i)
        for i in window
        if i < expects[-1]
        and i not in inits
        and _op(lines[i]) in _SLOT_USERS | {"ttng.init_barrier"}
    ):
        return None
    return [expects[-1], *inits]


def _hoist_site(
    lines: list[str],
    f: _Function,
    read_only: set[str],
    w0: int,
    k: int,
    rep: Report,
) -> list[int]:
    """Lines to move from the prologue after the wait ``[w0, k)`` to ``w0``."""
    ind = _indent(lines[w0])
    window: list[int] = []
    j = k
    while (
        j < f.end
        and _indent(lines[j]) == ind
        and _op(lines[j]) in _PASSABLE | _SLOT_USERS
    ):
        window.append(j)
        j += 1
    copies = [i for i in window if _op(lines[i]) in (_COPY, _TMA_COPY)]
    if not copies:
        return []
    # Pure ops of the wait itself (CSE shares index ranges with the warp sync)
    # may move with a copy; the poll, the spin and the sync may not.
    defs = {
        r: i
        for i in range(w0, j)
        if _indent(lines[i]) == ind
        for r in _results(lines[i])
    }
    chosen: set[int] = set()
    for copy in copies:
        seeds = [copy]
        if _op(lines[copy]) == _COPY:
            src, dst = _uses(lines[copy])[:2]
        else:
            tma = _TMA_OPERANDS.search(_code(lines[copy]))
            if tma is None:
                rep.skipped.append(f"{f.name}: unrecognized TMA copy")
                continue
            src, dst, barrier = tma.groups()
            stage = _tma_stage(lines, f, window, copy, barrier)
            if stage is None:
                rep.skipped.append(f"{f.name}: TMA copy's barrier is in use before it")
                continue
            seeds += stage
        roots = f.roots.get(src, frozenset({src}))
        if not roots <= read_only:
            rep.skipped.append(f"{f.name}: copy of {sorted(roots)} (written)")
            continue
        slot = _slot(lines, f.defs, dst)
        if any(
            _conflicts(_slot(lines, f.defs, u), slot)
            for i in window
            if i < copy and _op(lines[i]) in _SLOT_USERS
            for u in _uses(lines[i])
        ):
            rep.skipped.append(f"{f.name}: copy's slot is in use before it")
            continue
        stack, cone, ok = list(seeds), set(), True
        while stack and ok:
            cur = stack.pop()
            if cur in cone:
                continue
            cone.add(cur)
            for u in _uses(lines[cur]):
                d = defs.get(u)
                if d is None or d in cone:
                    continue
                if _op(lines[d]) not in _PURE | _BUFFER:
                    ok = False
                    break
                stack.append(d)
        if ok:
            chosen |= cone
            rep.hoisted += 1
        else:
            rep.skipped.append(f"{f.name}: copy depends on the wait or an effect")
    if chosen:
        rep.sites += 1
    return sorted(chosen)


def _may_wait_for_predecessor(line: str) -> bool:
    """Whether an op may wait on the PDL predecessor: Triton's ``gdc_wait``,
    any inline assembly off the allowlist, an external function with side
    effects, or any other op whose text mentions ``griddepcontrol.wait``."""
    op = _op(line)
    if op == _ASM:
        entry = _asm_entry(line)
        return entry is None or entry.kind == "pdl_wait"
    if op == _EXTERN and not _is_pure(line):
        return True
    return "griddepcontrol.wait" in line or "griddepcontrol.wait" in _decoded(line)


def _waits_for_predecessor(line: str) -> bool:
    """Whether an op is Triton's ``gdc_wait``, exactly: it waits on the PDL
    predecessor wherever control reaches it (a function holding other
    assembly is ruled out by :func:`_pdl_wait_dominates`)."""
    return _asm_kind(line) == "pdl_wait"


def _foreign_asm(lines: list[str], f: _Function | None = None) -> list[str]:
    """The inline assembly ops of ``f`` (default: of the module) that are not
    on the allowlist."""
    span = lines[f.start + 1 : f.end] if f is not None else lines
    return [line for line in span if _op(line) == _ASM and _asm_entry(line) is None]


def _pdl_wait_dominates(lines: list[str], f: _Function, at: int) -> bool:
    """Whether Triton's ``gdc_wait`` precedes line ``at`` in its block or an
    enclosing block of the function (structured control flow: dominance).
    Never in a module holding assembly off the allowlist: control may leave
    one assembly block for a label in another (also once LLVM inlines a
    callee), around the wait."""
    if _foreign_asm(lines):
        return False
    if any(_op(line) in ("cf.br", "cf.cond_br", "cf.switch") for line in lines[f.start + 1 : f.end]):
        # Several blocks (cf branches, e.g. an early return): a wait in a
        # sibling block does not dominate.
        return False
    level = _indent(lines[at])
    for j in range(at - 1, f.start, -1):
        ind = _indent(lines[j])
        if ind > level or not _code(lines[j]).strip():
            continue
        if ind < level:
            # Leaving the block: only the line that opened it encloses ``at``.
            if not _code(lines[j]).strip().endswith("{"):
                continue
            level = ind
            if _code(lines[j]).strip().startswith("}"):
                # ``} else {`` / ``} do {``: the region's sibling arm does not
                # dominate; continue from the op that owns both arms.
                continue
        if _waits_for_predecessor(lines[j]):
            return True
    return False


def _resolved_classes(
    f: _Function, alias_classes: tuple[tuple[str, ...], ...]
) -> list[set[int]] | None:
    """``alias_classes`` (argument names) as sets of ``f``'s parameters, or
    None if a member names no parameter of ``f``."""
    index = {n: k for k, n in enumerate(f.param_names) if n is not None}
    if any(name not in index for members in alias_classes for name in members):
        return None
    return [{index[name] for name in members} for members in alias_classes]


def rewrite(
    text: str, alias_classes: tuple[tuple[str, ...], ...] = ()
) -> tuple[str, Report]:
    """Hoist read-only prologue copies above readiness spins in ``text``.

    ``alias_classes``: kernel arguments (by name) that may share a storage at
    launch, e.g. a tensor passed as a pointer and as a TMA descriptor; an op
    writing one of them writes all.
    """
    rep = Report()
    text = _uniquify(text)
    lines = text.splitlines()
    funcs = _functions(lines)
    called = set()
    for line in lines:
        if _op(line) == "tt.call":
            callee = _FUNC_NAME.search(_code(line))
            called.add(callee.group(1).strip('"') if callee else "")
    eligible: list[_Function] = []
    classes: dict[str, list[set[int]]] = {}
    for f in funcs:
        reason = _unsupported(lines, f)
        resolved = _resolved_classes(f, alias_classes)
        if f.name in called:
            # A caller may pass one pointer for several parameters.
            reason = reason or "called function (its parameters may alias)"
        if resolved is None:
            reason = reason or "argument storage classes name unknown parameters"
        if reason is not None:
            rep.skipped.append(f"{f.name}: {reason}")
            continue
        eligible.append(f)
        classes[f.name] = resolved or []
    written = _written_params(lines, funcs, classes)
    known = {f.name for f in funcs}
    # A function outside the module may wait on the predecessor too.
    waits_on_predecessor = bool(called - known) or any(
        _may_wait_for_predecessor(line) for line in lines
    )
    for f in funcs:
        for line in _foreign_asm(lines, f):
            asm = _parse_asm(line)
            rep.foreign_asm.append(
                f"{f.name}: {asm.template[:80]!r}" if asm else f"{f.name}: unparsed"
            )
    moves: dict[int, list[int]] = {}
    for f in eligible:
        rep.written += [
            f.param_names[k] or f.params[k] for k in sorted(written[f.name])
        ]
        read_only = {p for k, p in enumerate(f.params) if k not in written[f.name]}
        i = f.start + 1
        while i < f.end:
            k = _wait_at(lines, i, f.end)
            if k is None:
                i += 1
                continue
            # Back-to-back waits (one per producer) form one site.
            while True:
                j = k
                while j < f.end and _op(lines[j]) in _PURE:
                    j += 1
                nxt = _wait_at(lines, j, f.end) if j < f.end else None
                if nxt is None:
                    break
                k = nxt
            if waits_on_predecessor and not _pdl_wait_dominates(lines, f, i):
                rep.skipped.append(
                    f"{f.name}: no griddepcontrol.wait before the wait"
                    + (" (assembly off the allowlist)" if _foreign_asm(lines) else "")
                )
                i = k
                continue
            moved = _hoist_site(lines, f, read_only, i, k, rep)
            if moved:
                moves[i] = moved
            i = k
    if not moves:
        return text, rep
    moved_lines = {m for ms in moves.values() for m in ms}
    out = []
    for idx, line in enumerate(lines):
        if idx in moves:
            pad = " " * _indent(line)
            out.extend(pad + lines[m].lstrip(" ") for m in moves[idx])
        if idx not in moved_lines:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), rep


def _reparse(mod: object, text: str) -> object:
    """``text`` parsed in ``mod``'s context and verified; raises on failure."""
    from triton._C.libtriton import ir  # pyrefly: ignore[missing-module-attribute]

    context = mod.context  # pyrefly: ignore[missing-attribute]
    fd, path = tempfile.mkstemp(suffix=".ttgir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        new = ir.parse_mlir_module(path, context)
        new.context = context  # as triton.compiler.compiler.parse does
        if not new.verify():
            raise RuntimeError("verification failed")
        return new
    finally:
        os.unlink(path)


def _hoisted(text: str, alias_classes: tuple[tuple[str, ...], ...]) -> str | None:
    """:func:`rewrite`'s text if it moved a copy, else None (logged)."""
    try:
        hoisted, rep = rewrite(text, alias_classes)
    except Exception as e:
        log.debug("ttgir weight prefetch: analysis failed (%r)", e)
        return None
    if rep.foreign_asm:
        log.debug(
            "ttgir weight prefetch: assembly off the allowlist: %s", rep.foreign_asm
        )
    if not rep.hoisted:
        if rep.skipped:
            log.debug("ttgir weight prefetch: nothing moved: %s", rep.skipped)
        return None
    log.debug(
        "ttgir weight prefetch: %d copies above %d waits (written: %s)",
        rep.hoisted,
        rep.sites,
        rep.written,
    )
    return hoisted


def _code_budget() -> int:
    """SASS code bytes above which a ring form moves on to its next smaller
    twin (:data:`ttgir_weight_ring.CODE_BUDGET`)."""
    from . import ttgir_weight_ring

    return ttgir_weight_ring.CODE_BUDGET


def _size_twin_suffixes() -> tuple[str, ...]:
    """Label suffixes of a ring form's smaller twins, longest first
    (:data:`ttgir_weight_ring.SIZE_LADDER`)."""
    from . import ttgir_weight_ring

    return tuple(
        sorted((s for s, _ in ttgir_weight_ring.SIZE_LADDER if s), key=len)[::-1]
    )


def _ring_form(label: str) -> str:
    """The ring form a candidate label belongs to (its size twins share it)."""
    for suffix in _size_twin_suffixes():
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _candidates(
    text: str, alias_classes: tuple[tuple[str, ...], ...], ring: bool = False
) -> list[tuple[str, str | None]]:
    """The fallback chain of one compile: (label, TTGIR text, or None for the
    module as is).

    With ``ring`` (the kernel's ``ttgir_weight_ring`` setting): the
    two-producer ring, the weight-only ring, the hoist, the module unchanged.
    A later stage (``make_llir``, ``ptxas``) that fails on one candidate, or a
    ring binary that spills, falls back to the next (:class:`_Fallback`).
    Each ring form is followed by its smaller twins
    (:data:`ttgir_weight_ring.SIZE_LADDER`: ``<label>_roll``, rolled
    producers; ``<label>_roll_u0``, also without the consumer unroll) where
    they change it: the trial ``ptxas`` of a ring form moves on to its next
    twin while the binary's code exceeds
    :data:`ttgir_weight_ring.CODE_BUDGET` (automatic, size-aware; no
    per-kernel switch)."""
    out: list[tuple[str, str | None]] = []
    if ring:
        from . import ttgir_weight_ring

        def with_twins(label: str, **opts: object) -> dict[str, object]:
            info: dict[str, object] = {}
            texts: list[str] = []
            for suffix, twin in ttgir_weight_ring.SIZE_LADDER:
                kwargs: dict[str, Any] = {"roll": False, **opts, **twin}
                try:
                    ring_text, twin_info = ttgir_weight_ring.rewrite_budgeted(
                        text, alias_classes=alias_classes, **kwargs
                    )
                except Exception as e:
                    if not suffix:
                        raise
                    log.debug("ttgir weight ring: no %s%s (%r)", label, suffix, e)
                    continue
                if not suffix:
                    info = twin_info
                    coverage = ttgir_weight_ring.copy_coverage(ring_text)
                    if coverage < ttgir_weight_ring.MIN_COPY_COVERAGE:
                        # Profitability: the ring would stream too little of
                        # the kernel's copy traffic to pay for its warp
                        # specialization.
                        log.debug(
                            "ttgir weight ring: %s moves %.1f%% of the copy bytes; "
                            "not a candidate",
                            label,
                            100 * coverage,
                        )
                        return info
                elif ring_text in texts:
                    continue
                texts.append(ring_text)
                out.append((label + suffix, ring_text))
            return info

        try:
            info = with_twins("ring")
            log.debug(
                "ttgir weight ring: %s", {k: v for k, v in info.items() if k != "rings"}
            )
            if info.get("act"):
                # If the two-producer ring fails later (make_llir, ptxas,
                # spills), the weight-only ring comes before the hoist.
                with_twins("ring_wonly", act=False)
        except Exception as e:
            log.debug("ttgir weight ring: not applied (%r)", e)
    hoisted = _hoisted(text, alias_classes)
    if hoisted is not None:
        out.append(("hoist", hoisted))
    out.append(("unchanged", None))
    return out


def _alias_classes_of_context(
    alias_classes: tuple[tuple[str, ...], ...] | None,
) -> tuple[tuple[str, ...], ...]:
    if alias_classes is None:
        facts = _current_facts()
        alias_classes = facts.alias_classes if facts is not None else ()
    return alias_classes


def transform_module(
    mod: object,
    alias_classes: tuple[tuple[str, ...], ...] | None = None,
    ring: bool | None = None,
) -> object:
    """Apply :func:`rewrite` to a TritonGPU module (with ``ring``, the first
    candidate of :func:`_candidates` that re-parses and verifies); unchanged
    on any failure.

    ``alias_classes`` and ``ring`` default to those of the :func:`enabled`
    context this thread compiles in.
    """
    alias_classes = _alias_classes_of_context(alias_classes)
    if ring is None:
        facts = _current_facts()
        ring = facts.ring if facts is not None else False
    text = mod.str()  # pyrefly: ignore[missing-attribute]
    if not ring:
        hoisted = _hoisted(text, alias_classes)
        if hoisted is None:
            return mod
        try:
            return _reparse(mod, hoisted)
        except Exception as e:
            log.debug("ttgir weight prefetch: re-parse failed (%r)", e)
            return mod
    for label, candidate in _candidates(text, alias_classes, ring=True):
        if candidate is None:
            return mod
        try:
            return _reparse(mod, candidate)
        except Exception as e:
            log.debug("ttgir weight prefetch: %s re-parse failed (%r)", label, e)
    return mod


def _ptxas_spills(ptx: str, arch: int, options: object) -> int:
    """Spill-store bytes ptxas reports for ``ptx`` (the tool and register
    allocator options of Triton's ``make_cubin``)."""
    return _ptxas_trial(ptx, arch, options)[0]


def _code_bytes(cubin: bytes) -> int:
    """Total size of the ``.text.*`` (SASS) sections of an ELF64 cubin."""
    import struct

    shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
    hdrs = [
        struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize)
        for i in range(shnum)
    ]
    strtab = hdrs[shstrndx][4]

    def name(off: int) -> bytes:
        return cubin[strtab + off : cubin.index(b"\0", strtab + off)]

    return sum(h[5] for h in hdrs if name(h[0]).startswith(b".text."))


def _ptxas_trial(ptx: str, arch: int, options: object) -> tuple[int, int]:
    """(spill-store bytes, SASS code bytes) of ``ptx`` compiled by ptxas with
    the tool and register allocator options of Triton's ``make_cubin``."""
    import subprocess

    from triton.backends.nvidia.compiler import get_ptxas
    from triton.backends.nvidia.compiler import sm_arch_from_capability

    ptxas = get_ptxas(arch).path
    extra = getattr(options, "ptx_options", None)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        Path(src).write_text(ptx)
        cmd = [
            ptxas,
            "-v",
            "--regAllocOptLevel=2",
            *(extra.split(" ") if extra else []),
            f"--gpu-name={sm_arch_from_capability(arch)}",
            src,
            "-o",
            src + ".o",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        code = _code_bytes(Path(src + ".o").read_bytes())
    return sum(int(x) for x in re.findall(r"(\d+) bytes spill stores", r.stderr)), code


class _Fallback:
    """Per-compile state of the candidate chain across Triton's later stages
    (installed only for kernels that ask for the ring)."""

    def __init__(
        self,
        stages: dict[str, Callable[..., object]],
        options: object,
        capability: object,
        alias_classes: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.make_ttgir = stages.get("ttgir")
        self.make_llir = stages.get("llir")
        self.make_ptx = stages.get("ptx")
        self.make_cubin = stages.get("cubin")
        self.options = options
        self.capability = capability
        self.alias_classes = alias_classes
        self.cands: list[tuple[str, str | None]] = []
        self.k = 0
        self.original: object = None
        self.original_text = ""
        self.num_warps: object = None

    @property
    def label(self) -> str:
        return self.cands[self.k][0] if self.cands else "unchanged"

    def _module(self, md: dict[str, object]) -> object:
        """Module of the current candidate, advancing past any that fails to
        re-parse."""
        while True:
            label, text = self.cands[self.k]
            if text is None:
                return _reparse(self.original, self.original_text)
            try:
                return _reparse(self.original, text)
            except Exception as e:
                log.debug("ttgir weight prefetch: %s re-parse failed (%r)", label, e)
                self.k += 1

    def advance(self, md: dict[str, object], why: BaseException) -> object:
        """Move to the next candidate and return its freshly parsed module
        (``make_llir`` mutates its input); re-raises ``why`` if none is left."""
        if self.k + 1 >= len(self.cands):
            raise why
        log.debug(
            "ttgir weight prefetch: %s failed after TTGIR (%r); falling back",
            self.label,
            why,
        )
        self.k += 1
        # make_llir overwrites it only for warp-specialized modules.
        md["num_warps"] = self.num_warps
        return self._module(md)

    def ttgir(self, src: object, md: dict[str, object]) -> object:
        mod = self.make_ttgir(src, md)  # pyrefly: ignore[not-callable]
        self.original = mod
        self.original_text = mod.str()  # pyrefly: ignore[missing-attribute]
        self.num_warps = md.get("num_warps")
        self.cands = _candidates(self.original_text, self.alias_classes, ring=True)
        self.k = 0
        if self.cands[0][1] is None:
            return mod
        return self._module(md)

    def llir(self, src: object, md: dict[str, object]) -> object:
        while True:
            try:
                return self.make_llir(src, md)  # pyrefly: ignore[not-callable]
            except Exception as e:
                src = self.advance(md, e)

    def ptx(self, src: object, md: dict[str, object]) -> object:
        """For a ring candidate, a trial ``ptxas`` here (not only in the cubin
        stage) keeps the PTX Triton stores consistent with the binary on
        fallback."""
        while True:
            out = self.make_ptx(src, md)  # pyrefly: ignore[not-callable]
            if not self.label.startswith("ring"):
                return out
            try:
                spills, code = _ptxas_trial(
                    cast("str", out), cast("int", self.capability), self.options
                )
                if spills:
                    raise RuntimeError(f"ring binary spills {spills} bytes")
                if code > _code_budget() and self._smaller_twin_next():
                    raise RuntimeError(
                        f"ring binary code {code} B exceeds the retained "
                        "instruction-cache budget"
                    )
                return out
            except Exception as e:
                src = self.llir(self.advance(md, e), md)

    def _next_label(self) -> str | None:
        return self.cands[self.k + 1][0] if self.k + 1 < len(self.cands) else None

    def _smaller_twin_next(self) -> bool:
        """The next candidate is a smaller twin of the current ring form
        (:data:`ttgir_weight_ring.SIZE_LADDER`; :func:`_candidates` lists a
        form's twins right after it, smallest last)."""
        nxt = self._next_label()
        return nxt is not None and _ring_form(nxt) == _ring_form(self.label)

    def cubin(self, src: object, md: dict[str, object]) -> object:
        while True:
            try:
                return self.make_cubin(src, md)  # pyrefly: ignore[not-callable]
            except Exception as e:
                src = self.ptx(self.llir(self.advance(md, e), md), md)


_DIGEST: str = hashlib.sha256(
    Path(__file__).read_bytes()
    + (Path(__file__).parent / "ttgir_weight_ring.py").read_bytes()
).hexdigest()


def _ring_key(ring: bool) -> str:
    """The ring's part of the cache key: off, or on (with the private
    negative-control test hook when it is set)."""
    if not ring:
        return "ring0"
    from . import ttgir_weight_ring

    options = ttgir_weight_ring.options_key()
    return f"ring1[{options}]" if options else "ring1"


@dataclasses.dataclass(frozen=True)
class _Facts:
    """What the launch tells the pass about the kernel it compiles."""

    # Kernel arguments (by name) that may share a storage, per storage.
    alias_classes: tuple[tuple[str, ...], ...] = ()
    # The kernel's ``ttgir_weight_ring`` setting: try the ring first.
    ring: bool = False

    def key(self) -> str:
        if not self.alias_classes:
            return ""
        digest = hashlib.sha256(repr(self.alias_classes).encode()).hexdigest()
        return f"-aliases-{digest[:16]}"


def normalize_alias_classes(
    classes: Iterable[Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    """Classes of two or more distinct names, each sorted, in sorted order."""
    return tuple(
        sorted(
            members
            for members in (tuple(sorted(set(c))) for c in classes)
            if len(members) > 1
        )
    )


_local = threading.local()


def _current_facts() -> _Facts | None:
    """The facts of the innermost :func:`enabled` context of this thread, or
    None outside one (a compile on another thread, e.g. Triton's async
    compile or an unrelated kernel, runs without the pass)."""
    stack = getattr(_local, "facts", None)
    return stack[-1] if stack else None


class _StagesHook:
    """``add_stages_inspection_hook`` that runs the pass after ``make_ttgir``,
    chained after any hook installed before it.  It acts only for compiles
    on a thread inside :func:`enabled`, and keys them with the pass version
    and the launch's facts."""

    def __init__(self, previous: Callable[..., object] | None) -> None:
        self.previous = previous

    def __call__(self, *args: object) -> object:
        facts = _current_facts()
        if not args:
            key, digest = (
                ("", "")
                if self.previous is None
                else cast("tuple[str, str]", self.previous())
            )
            if facts is None:
                return key, digest
            version = f"{_DIGEST}{facts.key()}-{_ring_key(facts.ring)}"
            combined = hashlib.sha256(f"{digest}:{version}".encode()).hexdigest()
            # Triton keys its on-disk cache with the first value and its
            # in-memory cache with the second: both name this version.
            return f"{key}helion-ttgir-weight-prefetch-{version}", combined
        if self.previous is not None:
            self.previous(*args)
        stages = args[1]
        assert isinstance(stages, dict)
        make_ttgir = stages.get("ttgir")
        if make_ttgir is None or facts is None:
            return None
        if not facts.ring or not all(k in stages for k in ("llir", "ptx", "cubin")):
            # The hoist alone; a ring needs the fallback through the later
            # stages (a backend without them gets no ring).
            stages["ttgir"] = lambda src, metadata: transform_module(
                make_ttgir(src, metadata), ring=False
            )
            return None
        # ring -> weight-only ring -> hoist -> unchanged, decided again by
        # every later stage that fails (make_llir, ptxas) or, for a ring,
        # spills.
        fb = _Fallback(
            stages,
            args[2] if len(args) > 2 else None,
            args[4] if len(args) > 4 else None,
            facts.alias_classes,
        )
        stages["ttgir"] = fb.ttgir
        stages["llir"] = fb.llir
        stages["ptx"] = fb.ptx
        stages["cubin"] = fb.cubin
        return None


_lock = threading.Lock()
_depth = 0
_saved: Callable[..., object] | None = None


@contextlib.contextmanager
def enabled(
    alias_classes: Iterable[Iterable[str]] = (), ring: bool = False
) -> Iterator[None]:
    """Run the pass on kernels this thread compiles (and keys) inside this
    context.  ``alias_classes``: kernel arguments (by name) that may share a
    storage at launch; ``ring``: try the warp-specialized weight ring first
    (the kernel's ``ttgir_weight_ring`` setting).  Both are part of the key."""
    global _depth, _saved
    from triton import knobs

    if not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        yield
        return
    facts = _Facts(normalize_alias_classes(alias_classes), ring=bool(ring))
    with _lock:
        if _depth == 0:
            _saved = knobs.runtime.add_stages_inspection_hook
            # Triton's PipelineStagesHook protocol lists four parameters, but
            # the backend calls the hook with five (and with none for the key).
            knobs.runtime.add_stages_inspection_hook = _StagesHook(_saved)  # pyrefly: ignore[bad-assignment]
        _depth += 1
    stack = _local.__dict__.setdefault("facts", [])
    stack.append(facts)
    try:
        yield
    finally:
        stack.pop()
        with _lock:
            _depth -= 1
            if _depth == 0:
                knobs.runtime.add_stages_inspection_hook = _saved
                _saved = None
