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

The pass reads the printed module as a tree of ops, blocks and SSA values
(:mod:`.ttgir_tree`), checked op by op and use by use against Triton's own
parse of it; operand order, types and attributes (assembly templates,
purity, callees, argument names) come from Triton's parse, not from the
printed syntax.  It moves ops in that tree, prints it and has Triton re-parse
and verify the result.  A module whose tree disagrees with Triton's parse,
and anything else the pass does not recognize, is left unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
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

from . import ttgir_tree

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator

log: logging.Logger = logging.getLogger(__name__)

_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')

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
_ASM = "tt.elementwise_inline_asm"
_EXTERN = "tt.extern_elementwise"
_ASM_ESCAPE = re.compile(r"\\([0-9A-Fa-f]{2})")


def _decoded(text: str) -> str:
    # MLIR prints other characters (quote, newline, tab) as two hex digits.
    return _ASM_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)


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


def _allowlist_entry(
    template: str, constraints: str, pure: bool, operands: int, packed: int
) -> _AsmTemplate | None:
    """The allowlist entry an inline assembly op with these attributes and
    operand count is, or None."""
    entry = _EXACT.get((template, constraints, pure, operands, packed))
    if entry is not None:
        return entry
    for pattern, family in _FAMILIES:
        if (
            pattern.fullmatch(template)
            and (constraints, pure, operands, packed) == family.key()[1:]
        ):
            return family
    return None


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


def _resolved_classes(
    f: _Kernel, alias_classes: tuple[tuple[str, ...], ...]
) -> list[set[int]] | None:
    """``alias_classes`` (argument names) as sets of ``f``'s parameters, or
    None if a member names no parameter of ``f``."""
    index = {n: k for k, n in enumerate(f.param_names) if n is not None}
    if any(name not in index for members in alias_classes for name in members):
        return None
    return [{index[name] for name in members} for members in alias_classes]


# --- The hoist ---------------------------------------------------------------
#
# The hoist reads the module through :mod:`.ttgir_tree`: ops, blocks and
# use-def chains from the tree, operand order, types and attributes from
# Triton's own parse of the same text (:func:`ttgir_tree.cross_check`).

_UNKNOWN = ("?", None)
_SCALAR = re.compile(r"i\d+|f\d+|bf16|f16|index")
_BARRIER = re.compile(r"!ttg\.memdesc<(?:\d+x)*i64\b")
_NAME_LOC = re.compile(r'loc\("((?:[^"\\]|\\.)*)"[()]')

Value = ttgir_tree.Value
Op = ttgir_tree.Op


def _conflicts(
    used: tuple[object, int | None] | None, written: tuple[object, int | None] | None
) -> bool:
    if used is None:
        return False
    if used is _UNKNOWN or written is None or written is _UNKNOWN:
        return True
    return used[0] is written[0] and (
        used[1] is None or written[1] is None or used[1] == written[1]
    )


def _location_name(value: Any) -> str | None:  # noqa: ANN401 - a Triton ir.value
    """The name of a Triton value's location (``loc("x")``), None if it has
    none."""
    loc = value.get_loc()
    if hasattr(loc, "get_name"):
        return loc.get_name() or None
    # Triton 3.7 has no getter; a name location prints ``loc("x")`` or
    # ``loc("x"(<child>))``, a file location ``loc("file":line:col)``.
    m = _NAME_LOC.match(str(loc))
    return _mlir_string(m.group(1)) if m else None


@dataclasses.dataclass(eq=False)
class _Kernel:
    """A function of the module (a ``tt.func`` with a body)."""

    op: Op
    name: str
    params: list[Value]
    # Each parameter's source name (its location's), None if unknown.
    param_names: list[str | None]
    # Parameters that are not pointers or descriptors (integers, floats): an
    # address carried in one names no argument's storage.
    scalars: frozenset[Value]
    body: list[Op]
    roots: dict[Value, frozenset[Value]] = dataclasses.field(default_factory=dict)

    def position(self, v: Value) -> int | None:
        return next((k for k, p in enumerate(self.params) if p is v), None)


class _Hoist:
    """The hoist's analyses of one module."""

    def __init__(self, m: ttgir_tree.Module, al: ttgir_tree.Alignment) -> None:
        self.m = m
        self.al = al
        self._operands: dict[Op, list[Value]] = {}
        self._asm: dict[Op, _AsmTemplate | None] = {}
        self.funcs = [
            self._kernel(op) for op in m.walk() if op.name == "tt.func" and op.regions
        ]
        # Inline assembly off the allowlist, anywhere in the module.
        self.foreign = [
            op for op in m.walk() if op.name == _ASM and self.asm(op) is None
        ]

    def operands(self, op: Op) -> list[Value]:
        """``op``'s operand values, in Triton's operand order."""
        found = self._operands.get(op)
        if found is None:
            found = self._operands[op] = [u.value for u in self.al.operands(op)]
        return found

    def attr(self, op: Op, kind: str, name: str) -> Any:  # noqa: ANN401
        return getattr(self.al.triton(op), f"get_{kind}_attr")(name)

    def asm(self, op: Op) -> _AsmTemplate | None:
        """The allowlist entry an inline assembly op is (its attributes as
        Triton reads them), or None."""
        if op not in self._asm:
            template = self.attr(op, "str", "asm_string")
            constraints = self.attr(op, "str", "constraints")
            pure = self.attr(op, "bool", "pure")
            packed = self.attr(op, "int", "packed_element")
            self._asm[op] = (
                None
                if None in (template, constraints, pure, packed)
                else _allowlist_entry(
                    template, constraints, pure, len(self.operands(op)), packed
                )
            )
        return self._asm[op]

    def asm_kind(self, op: Op) -> str | None:
        entry = self.asm(op) if op.name == _ASM else None
        return entry.kind if entry is not None else None

    def pure(self, op: Op) -> bool:
        return self.attr(op, "bool", "pure") is True

    def callee(self, op: Op) -> str:
        return self.attr(op, "flat_symbol_ref", "callee") or ""

    def type(self, v: Value) -> str:
        return self.al.type(v)

    # -- functions and the arguments they write ---------------------------------
    def _kernel(self, op: Op) -> _Kernel:
        entry = op.regions[0]
        params = [*entry.header_arguments, *entry.blocks[0].arguments]
        k = _Kernel(
            op,
            self.attr(op, "str", "sym_name") or "",
            params,
            [_location_name(self.al.value(p)) for p in params],
            frozenset(p for p in params if _SCALAR.fullmatch(self.type(p))),
            list(op.walk())[1:],
        )
        k.roots = self._roots(k.body)
        return k

    def _tainted(self, body: list[Op]) -> set[Value]:
        """Integers derived from a pointer's address (``tt.ptr_to_int``, then
        arithmetic): an offset made of them may reach another argument's
        storage.  Ops with regions pass them on to their results and region
        arguments (as initial or yielded values)."""
        tainted: set[Value] = set()
        changed = True
        while changed:
            changed = False
            for op in body:
                uses = self.operands(op)
                hit = op.name == "tt.ptr_to_int" or (
                    op.name != "tt.addptr" and any(u in tainted for u in uses)
                )
                if op.regions and not hit:
                    hit = any(
                        u in tainted
                        for r in op.regions
                        for b in r.blocks
                        if b.ops
                        for u in self.operands(b.ops[-1])
                    )
                if not hit:
                    continue
                new = set(op.results)
                for r in op.regions:
                    for b in r.blocks:
                        new.update(r.header_arguments if b is r.blocks[0] else ())
                        new.update(b.arguments)
                if not new <= tainted:
                    tainted |= new
                    changed = True
        return tainted

    def _roots(self, body: list[Op]) -> dict[Value, frozenset[Value]]:
        """Value -> the values its pointer derives from.

        A value produced by anything but pointer arithmetic is its own root,
        so a pointer whose roots are all parameters points into those
        arguments.
        """
        tainted = self._tainted(body)
        roots: dict[Value, frozenset[Value]] = {}
        # Device-built TMA descriptors: tensormap_create writes a tensormap of
        # its base pointer into kernel-private scratch, which
        # reinterpret_tensor_descriptor turns into the descriptor.
        described: dict[Value, frozenset[Value]] = {}
        for op in body:
            uses = self.operands(op)
            if op.name == "ttng.tensormap_create" and len(uses) >= 2:
                described[uses[0]] = roots.get(uses[1], frozenset({uses[1]}))
            if not op.results:
                continue
            own = frozenset({op.results[0]})
            if op.name == "tt.addptr" and any(u in tainted for u in uses[1:]):
                src = own
            elif op.name == "ttng.reinterpret_tensor_descriptor" and uses:
                src = described.get(uses[0], own)
            elif op.name in _POINTER_FLOW and uses:
                src = roots.get(uses[0], frozenset({uses[0]}))
            else:
                src = own
            for r in op.results:
                roots[r] = src
        return roots

    def _write_hits(
        self,
        op: Op,
        f: _Kernel,
        by_name: dict[str, _Kernel],
        written: dict[str, set[int]],
        wild: set[str],
    ) -> set[int] | None:
        """Indices of ``f``'s parameters ``op`` may write through (before
        storage classes), or None if it may write any argument."""
        targets: list[Value] = []
        if op.name in _WRITES:
            targets = self.operands(op)[:1]
        elif op.name == "tt.call":
            callee = self.callee(op)
            if callee not in by_name or callee in wild:
                # A function outside the module (or only declared in it) may
                # write anything; so may one that writes through an address it
                # builds.
                return None
            targets = [
                a for k, a in enumerate(self.operands(op)) if k in written[callee]
            ]
        elif op.name == _ASM:
            entry = self.asm(op)
            if entry is None:
                # Assembly off the allowlist may load or build the address it
                # stores through: it may write any argument.
                return None
            if entry.written_operand is not None:
                targets = [self.operands(op)[entry.written_operand]]
        elif op.name == _EXTERN and not self.pure(op):
            # So may an external function with side effects.
            return None
        hit: set[int] = set()
        for t in targets:
            for r in f.roots.get(t, frozenset({t})):
                k = f.position(r)
                if k is None or r in f.scalars:
                    # An unattributable target may be any argument.
                    return None
                hit.add(k)
        return hit

    def write_analysis(
        self, classes: dict[str, list[set[int]]]
    ) -> tuple[dict[str, set[int]], set[str]]:
        """(Function name -> indices of the pointer parameters it may write
        through, the functions that may write any argument).  ``classes``:
        per function, sets of parameters that may share a storage; writing
        one member writes them all."""
        by_name = {f.name: f for f in self.funcs}
        written: dict[str, set[int]] = {f.name: set() for f in self.funcs}
        # Functions that may write memory no parameter of theirs names.
        wild = {
            f.name
            for f in self.funcs
            if any(op.name == "tt.int_to_ptr" for op in f.body)
        }
        changed = True
        while changed:
            changed = False
            for f in self.funcs:
                everything = set(range(len(f.params)))
                hit: set[int] = set()
                for op in f.body:
                    op_hit = self._write_hits(op, f, by_name, written, wild)
                    if op_hit is None:
                        hit = everything
                        break
                    hit |= op_hit
                    if hit == everything:
                        break
                if hit == everything and f.name not in wild:
                    wild.add(f.name)
                    changed = True
                for members in classes.get(f.name, ()):
                    if hit & members:
                        hit |= members
                if not hit <= written[f.name]:
                    written[f.name] |= hit
                    changed = True
        return written, wild

    def unsupported(self, f: _Kernel) -> str | None:
        for op in f.body:
            if op.name == "tt.int_to_ptr":
                return "integer-to-pointer cast"
            if (
                op.name == _ASM
                and ".sys" in (self.attr(op, "str", "asm_string") or "")
                or (
                    op.name in ("tt.atomic_rmw", "tt.atomic_cas")
                    and re.search(r",\s*sys\b", _code(op.first_line))
                )
            ):
                return "system-scope operation"
        return None

    def may_wait_for_predecessor(self, op: Op) -> bool:
        """Whether an op may wait on the PDL predecessor: Triton's
        ``gdc_wait``, any inline assembly off the allowlist, an external
        function with side effects, or any other op whose text mentions
        ``griddepcontrol.wait``."""
        if op.name == _ASM:
            entry = self.asm(op)
            return entry is None or entry.kind == "pdl_wait"
        if op.name == _EXTERN and not self.pure(op):
            return True
        own = "".join(
            line for p in op.parts if isinstance(p, ttgir_tree.Text) for line in p.lines
        )
        return "griddepcontrol.wait" in own or "griddepcontrol.wait" in _decoded(own)

    def pdl_wait_dominates(self, f: _Kernel, op: Op) -> bool:
        """Whether Triton's ``gdc_wait`` precedes ``op`` in its block or an
        enclosing block of the function (structured control flow:
        dominance).  Never in a module holding assembly off the allowlist:
        control may leave one assembly block for a label in another (also
        once LLVM inlines a callee), around the wait."""
        if self.foreign:
            return False
        if any(x.name in ("cf.br", "cf.cond_br", "cf.switch") for x in f.body):
            # Several blocks (cf branches, e.g. an early return): a wait in a
            # sibling block does not dominate.
            return False
        cur: Op | None = op
        while cur is not None and cur is not f.op and cur.parent is not None:
            ops = cur.parent.ops
            if any(self.asm_kind(x) == "pdl_wait" for x in ops[: ops.index(cur)]):
                return True
            cur = cur.parent_op
        return False

    # -- readiness spins --------------------------------------------------------
    def is_seed(self, op: Op) -> bool:
        return self.asm_kind(op) == "poll" and sum(v.count for v in op.results) == 1

    def wait_at(self, ops: list[Op], i: int) -> int | None:
        """If ``ops[i]`` seeds a readiness spin, the index just past its
        sync."""
        if not self.is_seed(ops[i]):
            return None
        # One spin may poll several counters: seeds (and their targets) then a
        # single scf.while carrying the polled values.
        seeds = {ops[i].results[0]}
        j = i + 1
        while j < len(ops):
            if self.is_seed(ops[j]):
                seeds.add(ops[j].results[0])
            elif ops[j].name not in _PURE:
                break
            j += 1
        if not (
            j < len(ops)
            and ops[j].name == "scf.while"
            and seeds <= set(self.operands(ops[j]))
            and any(self.is_seed(x) for x in list(ops[j].walk())[1:])
        ):
            return None
        k, synced = j + 1, None
        while k < len(ops):
            if self.asm_kind(ops[k]) == "sync":
                synced = k = k + 1
            elif ops[k].name in ("tt.make_range", "arith.constant"):
                k += 1
            else:
                break
        return synced

    # -- shared memory slots ----------------------------------------------------
    def slot(self, v: Value) -> tuple[object, int | None] | None:
        """(buffer, constant slot or None for any) a memory descriptor names,
        None for a value that is not one, ``_UNKNOWN`` if that cannot be
        told."""
        d = v.owner_op
        if d is None:
            return _UNKNOWN
        uses = self.operands(d)
        if d.name in ("ttg.local_alloc", "ttng.tmem_alloc"):
            return (v, None)
        if d.name in _VIEWS:
            return self.slot(uses[0])
        if d.name == "ttg.memdesc_index":
            base = self.slot(uses[0])
            if base is None or base is _UNKNOWN or base[1] is not None:
                return _UNKNOWN
            index = uses[1].owner_op
            constant = (
                self.attr(index, "int", "value")
                if index is not None and index.name == "arith.constant"
                else None
            )
            return (base[0], constant)
        if self.type(v).startswith("!ttg.memdesc<"):
            return _UNKNOWN
        return None

    def copy_roles(self, op: Op) -> tuple[Value, Value, Value | None] | None:
        """(source pointer or descriptor, destination buffer, TMA barrier or
        None) of a copy, told apart by their types; None if they are not
        what a copy has."""
        uses = self.operands(op)
        types = [self.type(u) for u in uses]
        if op.name == _COPY:
            if (
                len(uses) >= 2
                and "!tt.ptr<" in types[0]
                and types[1].startswith("!ttg.memdesc<")
            ):
                return uses[0], uses[1], None
            return None
        buffers = [k for k, t in enumerate(types) if t.startswith("!ttg.memdesc<")]
        barriers = [k for k in buffers if _BARRIER.match(types[k])]
        if (
            not uses
            or not types[0].startswith("!tt.tensordesc<")
            or len(buffers) != 2
            or len(barriers) != 1
        ):
            return None
        (dst,) = (k for k in buffers if k != barriers[0])
        return uses[0], uses[dst], uses[barriers[0]]

    def tma_stage(
        self, ops: list[Op], window: list[int], copy: int, barrier: Value
    ) -> list[int] | None:
        """The ``barrier_expect`` (and any ``init_barrier``) of a TMA copy's
        stage barrier, which move with the copy; None if anything else before
        the expect uses that barrier (an earlier phase of a reused barrier)."""
        slot = self.slot(barrier)
        if slot is None or slot is _UNKNOWN or slot[1] is None:
            return None

        def on_barrier(i: int) -> bool:
            return any(_conflicts(self.slot(u), slot) for u in self.operands(ops[i]))

        expects = [
            i
            for i in window
            if i < copy and ops[i].name == "ttng.barrier_expect" and on_barrier(i)
        ]
        if not expects:
            return None
        inits = [
            i
            for i in window
            if i < expects[-1] and ops[i].name == "ttng.init_barrier" and on_barrier(i)
        ]
        if any(
            on_barrier(i)
            for i in window
            if i < expects[-1]
            and i not in inits
            and ops[i].name in _SLOT_USERS | {"ttng.init_barrier"}
        ):
            return None
        return [expects[-1], *inits]

    def hoist_site(
        self,
        f: _Kernel,
        read_only: set[Value],
        ops: list[Op],
        w0: int,
        k: int,
        rep: Report,
    ) -> list[int]:
        """Indices of the ops to move from after the wait ``ops[w0:k]`` to
        before it."""
        window: list[int] = []
        j = k
        while j < len(ops) and ops[j].name in _PASSABLE | _SLOT_USERS:
            window.append(j)
            j += 1
        copies = [i for i in window if ops[i].name in (_COPY, _TMA_COPY)]
        if not copies:
            return []
        # Pure ops of the wait itself (CSE shares index ranges with the warp
        # sync) may move with a copy; the poll, the spin and the sync may not.
        defs = {r: i for i in range(w0, j) for r in ops[i].results}
        chosen: set[int] = set()
        for copy in copies:
            seeds = [copy]
            roles = self.copy_roles(ops[copy])
            if roles is None:
                rep.skipped.append(f"{f.name}: unrecognized copy")
                continue
            src, dst, barrier = roles
            if barrier is not None:
                stage = self.tma_stage(ops, window, copy, barrier)
                if stage is None:
                    rep.skipped.append(
                        f"{f.name}: TMA copy's barrier is in use before it"
                    )
                    continue
                seeds += stage
            roots = f.roots.get(src, frozenset({src}))
            if not roots <= read_only:
                names = sorted(f"%{r.name}" for r in roots)
                rep.skipped.append(f"{f.name}: copy of {names} (written)")
                continue
            slot = self.slot(dst)
            if any(
                _conflicts(self.slot(u), slot)
                for i in window
                if i < copy and ops[i].name in _SLOT_USERS
                for u in self.operands(ops[i])
            ):
                rep.skipped.append(f"{f.name}: copy's slot is in use before it")
                continue
            stack, cone, ok = list(seeds), set(), True
            while stack and ok:
                cur = stack.pop()
                if cur in cone:
                    continue
                cone.add(cur)
                for u in self.operands(ops[cur]):
                    d = defs.get(u)
                    if d is None or d in cone:
                        continue
                    if ops[d].name not in _PURE | _BUFFER:
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


def rewrite(
    text: str,
    alias_classes: tuple[tuple[str, ...], ...] = (),
    mod: object = None,
) -> tuple[str, Report]:
    """Hoist read-only prologue copies above readiness spins in ``text``.

    ``alias_classes``: kernel arguments (by name) that may share a storage at
    launch, e.g. a tensor passed as a pointer and as a TMA descriptor; an op
    writing one of them writes all.  ``mod``: Triton's parse of ``text`` (the
    stages hook's module), parsed here if not given.  A module whose tree
    disagrees with it (:func:`ttgir_tree.cross_check`) is left unchanged.
    """
    rep = Report()
    try:
        m = ttgir_tree.parse(text)
        al = ttgir_tree.cross_check(
            m, mod if mod is not None else ttgir_tree.triton_parse(text)
        )
    except ttgir_tree.ParseError as e:
        rep.skipped.append(f"module: {e}")
        return text, rep
    h = _Hoist(m, al)
    called = {h.callee(op) for op in m.walk() if op.name == "tt.call"}
    eligible: list[_Kernel] = []
    classes: dict[str, list[set[int]]] = {}
    for f in h.funcs:
        reason = h.unsupported(f)
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
    written, _ = h.write_analysis(classes)
    # A function outside the module may wait on the predecessor too.
    waits_on_predecessor = bool(called - {f.name for f in h.funcs}) or any(
        h.may_wait_for_predecessor(op) for op in m.walk()
    )
    for f in h.funcs:
        rep.foreign_asm += [
            f"{f.name}: {(h.attr(op, 'str', 'asm_string') or '')[:80]!r}"
            for op in f.body
            if op in h.foreign
        ]
    moves: list[tuple[Op, list[Op]]] = []

    def scan(f: _Kernel, read_only: set[Value], block: ttgir_tree.Block) -> None:
        ops = block.ops
        i = 0
        while i < len(ops):
            k = h.wait_at(ops, i)
            if k is None:
                for r in ops[i].regions:
                    for b in r.blocks:
                        scan(f, read_only, b)
                i += 1
                continue
            # Back-to-back waits (one per producer) form one site.
            while True:
                j = k
                while j < len(ops) and ops[j].name in _PURE:
                    j += 1
                nxt = h.wait_at(ops, j) if j < len(ops) else None
                if nxt is None:
                    break
                k = nxt
            if waits_on_predecessor and not h.pdl_wait_dominates(f, ops[i]):
                rep.skipped.append(
                    f"{f.name}: no griddepcontrol.wait before the wait"
                    + (" (assembly off the allowlist)" if h.foreign else "")
                )
                i = k
                continue
            moved = h.hoist_site(f, read_only, ops, i, k, rep)
            if moved:
                moves.append((ops[i], [ops[x] for x in moved]))
            i = k

    for f in eligible:
        rep.written += [
            f.param_names[k] or f"%{f.params[k].name}" for k in sorted(written[f.name])
        ]
        read_only = {p for k, p in enumerate(f.params) if k not in written[f.name]}
        for r in f.op.regions:
            for b in r.blocks:
                scan(f, read_only, b)
    if not moves:
        return text, rep
    for anchor, ops in moves:
        for op in ops:
            op.move_before(anchor)
    m.analyze()
    if m.unresolved:
        op, name = m.unresolved[0]
        rep.skipped.append(f"module: moving {op.name} left %{name} undefined")
        rep.hoisted = rep.sites = 0
        return text, rep
    return m.text(), rep


def inline_assembly(text: str) -> list[tuple[str, _AsmTemplate | None]]:
    """Each inline assembly op of a module, as the hoist reads it: (its
    template, its allowlist entry or None)."""
    m = ttgir_tree.parse(text)
    h = _Hoist(m, ttgir_tree.cross_check(m, ttgir_tree.triton_parse(text)))
    return [
        (h.attr(op, "str", "asm_string") or "", h.asm(op))
        for op in m.walk()
        if op.name == _ASM
    ]


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


def _hoisted(
    text: str, alias_classes: tuple[tuple[str, ...], ...], mod: object = None
) -> str | None:
    """:func:`rewrite`'s text if it moved a copy, else None (logged)."""
    try:
        hoisted, rep = rewrite(text, alias_classes, mod)
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


def transform_module(
    mod: object, alias_classes: tuple[tuple[str, ...], ...] | None = None
) -> object:
    """Apply :func:`rewrite` to a TritonGPU module; unchanged on any failure.

    ``alias_classes`` defaults to that of the :func:`enabled` context this
    thread compiles in.
    """
    if alias_classes is None:
        facts = _current_facts()
        alias_classes = facts.alias_classes if facts is not None else ()
    text = mod.str()  # pyrefly: ignore[missing-attribute]
    hoisted = _hoisted(text, alias_classes, mod)
    if hoisted is None:
        return mod
    try:
        return _reparse(mod, hoisted)
    except Exception as e:
        log.debug("ttgir weight prefetch: re-parse failed (%r)", e)
        return mod


def _pass_sources() -> list[Path]:
    """The pass and every module of its family (``ttgir_*.py`` next to it),
    by name."""
    return sorted(Path(__file__).parent.glob("ttgir_*.py"))


_DIGEST: str = hashlib.sha256(
    b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in _pass_sources())
).hexdigest()


@dataclasses.dataclass(frozen=True)
class _Facts:
    """What the launch tells the pass about the kernel it compiles."""

    # Kernel arguments (by name) that may share a storage, per storage.
    alias_classes: tuple[tuple[str, ...], ...] = ()

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
            version = f"{_DIGEST}{facts.key()}"
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
        stages["ttgir"] = lambda src, metadata: transform_module(
            make_ttgir(src, metadata)
        )
        return None


_lock = threading.Lock()
_depth = 0
_saved: Callable[..., object] | None = None


@contextlib.contextmanager
def enabled(alias_classes: Iterable[Iterable[str]] = ()) -> Iterator[None]:
    """Run the pass on kernels this thread compiles (and keys) inside this
    context.  ``alias_classes``: kernel arguments (by name) that may share a
    storage at launch; part of the key."""
    global _depth, _saved
    from triton import knobs

    if not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        yield
        return
    facts = _Facts(normalize_alias_classes(alias_classes))
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
