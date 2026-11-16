"""MSL side of ticketed ``hl.barrier()`` phases (``ticketed_phases.py``):
the device helpers and the kernel-body rewrites ``metal_jit`` applies."""

from __future__ import annotations

import ast

from ..ast_read_writes import ast_rename
from .lane_fold import is_store
from .lane_fold import names_in
from .lane_fold import pointer_base

#: The MSL namespace of the phase helpers; a kernel that names it gets
#: :data:`PHASE_PREAMBLE`, and ``coherent(device)`` access to the buffers
#: :func:`coherent_buffers` picks.
PHASE_NAMESPACE = "helion_phase"

#: ``int32`` words between two counters of the phase state: each counter
#: (the ticket and every phase's) on its own 128-byte line, so polling one
#: does not contend with updates to another.
COUNTER_STRIDE = 32

PHASE_PREAMBLE = f"""\
namespace helion_phase {{
// Ticketed phases of a flat launch; see helion/_compiler/metal/ticketed_phases.py.
// Counter 0 hands out tickets, counter 1 + p counts the finished tasks of phase p, and the
// last of them raises flag p (counter phases + p), which later phases poll.
constant constexpr uint STRIDE = {COUNTER_STRIDE};
inline bool leader(uint3 tid) {{ return tid.x == 0 && tid.y == 0 && tid.z == 0; }}
inline uint3 layout(uint flat, uint x, uint y) {{ return uint3(flat % x, flat / x % y, flat / (x * y)); }}
inline device atomic_uint* counter(device int* state, uint i) {{
  return reinterpret_cast<device atomic_uint*>(state + i * STRIDE);
}}
inline uint ticket(device int* state, threadgroup uint* slot, uint3 tid) {{
  if (leader(tid)) {{
    slot[0] = atomic_fetch_add_explicit(counter(state, 0), 1u, memory_order_relaxed);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return slot[0];
}}
inline void wait(device int* state, uint phases, uint phase, uint3 tid) {{
  if (leader(tid)) {{
    device atomic_uint* flag = counter(state, phases + phase);
    // Poll with a read-modify-write: on an M4 Pro a relaxed atomic load kept
    // returning a stale 0 for up to a second after the flag was raised
    // (about 1 in 20 launches stalled), while the RMW reads the current value.
    while (atomic_fetch_add_explicit(flag, 0u, memory_order_relaxed) == 0u) {{
    }}
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }}
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}}
inline void arrive(device int* state, uint phases, uint phase, uint count, uint3 tid) {{
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (leader(tid)) {{
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (atomic_fetch_add_explicit(counter(state, 1 + phase), 1u, memory_order_relaxed) == count - 1u) {{
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(counter(state, phases + phase), 1u, memory_order_relaxed);
    }}
  }}
}}
}}  // namespace helion_phase
"""


#: The generated module global (``ticketed_phases._record_phase_access``)
#: mapping each tensor argument of a phased kernel to the phases that read it
#: and the phases that store to it.
PHASE_ACCESS_GLOBAL = "_METAL_PHASE_ACCESS"

#: The generated module global (``ticketed_phases.order_phases``) mapping each
#: tensor argument that a store of a phased kernel writes on several threads at
#: once to the tensor arguments that store's value and mask load
#: (``multi_root.GuardResult.repeated_reads``).
REPEATED_READS_GLOBAL = "_METAL_REPEATED_READS"

#: The generated module global (``ticketed_phases.relayout_roots``) mapping
#: each reduction site of a phased launch whose roots lay their threads out
#: differently to its root's layout, which sizes the site's collective.
SITE_DIMS_GLOBAL = "_METAL_SITE_DIMS"


def coherent_buffers(
    access: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    storage_of: dict[str, int],
) -> set[str]:
    """The tensor arguments to access through ``coherent(device)`` pointers:
    every buffer that a phase writes and a later phase reads or writes.

    Within one dispatch, Metal makes a threadgroup's device-memory accesses
    visible to other threadgroups only for ``coherent(device)`` memory,
    written and read through such pointers; the fences of ``wait`` and
    ``arrive`` order those accesses but do not make plain ones visible, and
    plain writes are only published when the dispatch ends.  So the writer
    and every later phase reach such a buffer through its coherent alias
    (:func:`coherent_param_prologue`).  A read in a phase before the write
    needs nothing, since the reading phase finished before the writing one
    began, and neither do accesses within one phase, since two threadgroups
    of a phase are unordered anyway.  Views of one storage (``storage_of``)
    are one buffer.
    """
    groups: dict[int, tuple[set[int], set[int]]] = {}
    for name, (reads, writes) in access.items():
        if name in storage_of:
            group = groups.setdefault(storage_of[name], (set(), set()))
            group[0].update(reads)
            group[1].update(writes)
    handed_on = {
        ptr
        for ptr, (reads, writes) in groups.items()
        if any(w < a for w in writes for a in reads | writes)
    }
    return {name for name, ptr in storage_of.items() if ptr in handed_on}


def coherent_param_prologue(
    body_stmts: list[ast.stmt],
    coherent: set[str],
    metal_dtypes: dict[str, str],
    param_names: list[str],
) -> tuple[list[ast.stmt], list[str]]:
    """Route every access to the ``coherent`` buffers (:func:`coherent_buffers`)
    through a ``coherent(device)`` alias, so a phase sees another
    threadgroup's writes from an earlier phase.  Each alias is named apart
    from every parameter and every name of the body.  Returns the renamed
    body and the MSL declarations of the aliases, which open the kernel
    body.  An MPP matmul stages every operand of a buffer codegen sees an
    earlier phase write (``mpp_matmul._order_memory``), here through the
    alias.  An operand whose storage only the call shares with such a buffer
    stays a device view, which ``metal_jit._reject_coherent_views``
    refuses."""
    taken = set(param_names) | names_in(body_stmts)
    renames: dict[str, str] = {}
    for name in sorted(coherent):
        alias = base = f"{name}__coherent"
        suffix = 0
        while alias in taken:
            suffix += 1
            alias = f"{base}{suffix}"
        taken.add(alias)
        renames[name] = alias

    renamed = [ast_rename(stmt, renames) for stmt in body_stmts]
    decls = [
        f"device coherent(device) {metal_dtypes[name]}* {alias} = "
        f"reinterpret_cast<device coherent(device) {metal_dtypes[name]}*>({name});"
        for name, alias in renames.items()
    ]
    return renamed, decls


def stored_params(body_stmts: list[ast.stmt], params: set[str]) -> set[str]:
    """The tensor parameters the body writes: every one named in a
    ``tl.store``'s address.  Its base is the one it writes; counting any
    other errs toward a coherent buffer, never away from one."""
    stored: set[str] = set()
    for node in ast.walk(ast.Module(body=body_stmts, type_ignores=[])):
        if is_store(node):
            stored.update(
                name.id
                for name in ast.walk(node.args[0])
                if isinstance(name, ast.Name) and name.id in params
            )
    return stored


def read_params(body_stmts: list[ast.stmt], params: set[str]) -> set[str]:
    """The tensor parameters the body reads: named anywhere but as the base
    of a ``tl.store``'s address, the tensor it writes."""
    read: set[str] = set()

    class _Reads(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if is_store(node):
                written = pointer_base(node.args[0])
                read.update(
                    name.id
                    for name in ast.walk(node.args[0])
                    if isinstance(name, ast.Name)
                    and name.id in params
                    and name.id != written
                )
                for arg in [*node.args[1:], *(k.value for k in node.keywords)]:
                    self.visit(arg)
                return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in params:
                read.add(node.id)

    visitor = _Reads()
    for stmt in body_stmts:
        visitor.visit(stmt)
    return read


def uses_phases(body_stmts: list[ast.stmt]) -> bool:
    """Whether the kernel body calls the phase helpers."""
    return PHASE_NAMESPACE in names_in(body_stmts)
