"""Matcher and renderer of the memory-access calls the CuTe emitters generate.

The lane-loop, pipelining and vectorization passes run over the generated
device body and have to recognize its loads and stores again.  This module
is the one description of that grammar (``match_packet_access`` is the
analysis, ``render_packet_load`` / ``render_masked_packet_load`` /
``vector_type_source`` the emitter side), replacing the per-pass parsers
of the same call text.  The grammar, pinned by the render corpus of
``test/test_cute_*.py`` (``test/test_cute_packet_access.py`` holds every
call form the corpus contains):

======================================================================  ================
call                                                                    ``kind``
======================================================================  ================
``(PTR).load()``                                                        ``load``
``(PTR).store(VALUE)``                                                  ``store``
``cute.arch.load(PTR, cutlass.T[, cop='cs' | level1_eviction_priority=...])``  ``arch_load``
``cute.arch.load(PTR, ir.VectorType.get([V], cutlass.T.mlir_type)[, kw])``     ``packet``
``<L2 helper>(PTR, ir.VectorType.get([V], cutlass.T.mlir_type))``              ``hinted_packet``
``_cute_load_vNb_if(PTR, PRED, ir.VectorType.get([V], cutlass.T.mlir_type)[, 'prio'])``  ``masked_packet``
``<flush helper>(PTR, VALUES)``                                         ``flush``
======================================================================  ================

``arch_load`` is the scalar ``cute.arch.load``: a cache-hinted 4- or 8-byte
scalar (``cutlass.Float32``), or a byte-packed word carrier
(``cutlass.Uint8`` for an fp8/fp4 element, ``cutlass.Uint32`` /
``cutlass.Uint64`` for a packed fp8 chunk) whose element count the call
does not show.  The L2 helpers are the names of
``cache_policy_loads._CUTE_CACHE_LOAD_HELPER_NAMES``, the predicated loads
those of ``cache_policy_loads.MASKED_PACKET_LOAD_HELPER_NAMES``, the flushes
those of ``packet_flush_names`` (the ``_cute_store_*_vec`` family).

A consumer narrows the match to what it handles (a plain ``packet`` with
no keywords and a ``Uint16`` / ``Uint32`` carrier, say) by reading the
record's fields; the matcher itself only parses.

The memory writes are wider than the packet stores: ``is_store_call`` names
every generated write (stores, atomics, vector reductions into memory, bulk
copies and the persistent vectorizer's store marker), ``store_pointer`` the
address one writes through, and ``call_write_roots`` the tensors any call
may write, the calls whose effects are unknown included.  The passes that
move or reuse memory accesses ask these instead of keeping lists of their
own.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path
from ._ast_pass_utils import call_path_str
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cache_policy_loads import MASKED_PACKET_LOAD_HELPER_NAMES
from .cache_policy_loads import CachePolicy
from .packet_flush_names import PACKET_FLUSH_HELPER_NAMES
from .reduce_call import GROUPED_REDUCE_HELPERS
from .reduce_call import ReduceScope

if TYPE_CHECKING:
    from collections.abc import Callable

PacketKind = Literal[
    "load", "store", "arch_load", "packet", "hinted_packet", "masked_packet", "flush"
]

_ARCH_LOAD = ("cute", "arch", "load")
_ARCH_STORE = ("cute", "arch", "store")
_VECTOR_TYPE = ("ir", "VectorType", "get")
# The branch-local persistent vectorizer's markers (``memory_ops``):
# ``LOAD(block, width, dtype, eviction, PTR, scalar_load)`` and
# ``STORE(block, width, dtype, PTR, value, mask)``.
PERSISTENT_BRANCH_VEC_LOAD = "_helion_persistent_branch_vec_load"
PERSISTENT_BRANCH_VEC_STORE = "_helion_persistent_branch_vec_store"
# Calls that write no tensor besides those the relocation proofs accept:
# the loops' iterators, a thread's fresh register tensor, the vectorizer's
# load marker, the register conversions of packed bf16 and fp4, warp
# shuffles and the pure selects and extrema of ``cutlass``, and the
# calls that touch only shared memory or barrier state, which no tensor a
# load reads aliases: shared-memory allocation and tensor views, the CTA,
# mbarrier and cluster barriers, the signal that lets dependent grids
# launch, and the grouped reductions, whose scratch is shared memory.  (The
# wait for the grids before is none: it makes their writes visible.)
_NO_MEMORY_WRITE = frozenset(
    {
        PERSISTENT_BRANCH_VEC_LOAD,
        *GROUPED_REDUCE_HELPERS,
        "_cute_bfloat16_x16_to_float16",
        "_cute_float4_e2m1fn_x16_to_float16",
        "cute.arch.alloc_smem",
        "cute.arch.cluster_arrive",
        "cute.arch.cluster_arrive_relaxed",
        "cute.arch.cluster_wait",
        "cute.arch.griddepcontrol_launch_dependents",
        "cute.arch.make_warp_uniform",
        "cute.arch.mbarrier_init",
        "cute.arch.mbarrier_init_fence",
        "cute.arch.shuffle_sync",
        "cute.arch.shuffle_sync_bfly",
        "cute.arch.shuffle_sync_down",
        "cute.arch.shuffle_sync_up",
        "cute.arch.sync_threads",
        "cute.make_layout",
        "cute.make_rmem_tensor",
        "cute.make_tensor",
        "cutlass.max",
        "cutlass.min",
        "cutlass.range",
        "cutlass.range_constexpr",
        "cutlass.select_",
        "range",
    }
)


@dataclasses.dataclass(frozen=True)
class PacketAccess:
    """One generated load or store call, parsed.

    ``pointer`` is the addressed operand (the receiver of ``.load()`` /
    ``.store()``, the first argument of every other form).  ``width`` and
    ``carrier`` describe the ``ir.VectorType.get([V], cutlass.T.mlir_type)``
    operand of the vector forms (``carrier`` is ``"cutlass.T"``); for an
    ``arch_load`` ``carrier`` is the scalar type text and ``width`` is None
    (one element, or a packed word).  ``helper`` is the called helper name
    of the hinted / masked / flush forms, ``predicate`` the guard of a
    masked packet, ``keywords`` the call's keywords as written.
    """

    kind: PacketKind
    call: ast.Call
    pointer: ast.expr
    width: int | None = None
    carrier: str | None = None
    helper: str | None = None
    predicate: ast.expr | None = None
    keywords: tuple[ast.keyword, ...] = ()

    @property
    def is_load(self) -> bool:
        return self.kind not in ("store", "flush")

    @property
    def is_vector(self) -> bool:
        """A packet of ``width`` elements (plain, hinted or predicated)."""
        return self.kind in ("packet", "hinted_packet", "masked_packet")

    @property
    def vector_type(self) -> ast.Call | None:
        """The ``ir.VectorType.get(...)`` operand of a vector form."""
        if self.kind == "masked_packet":
            operand = self.call.args[2]
        elif self.is_vector:
            operand = self.call.args[1]
        else:
            return None
        assert isinstance(operand, ast.Call)
        return operand

    def cache_policy(self) -> CachePolicy | None:
        """The hint spelled by the call's keywords (``arch_load`` / ``packet``),
        or None when a keyword is not a string-valued cache hint."""
        return CachePolicy.from_load_keywords(self.keywords)


def vector_type_parts(node: ast.AST) -> tuple[int | None, str | None] | None:
    """``(V, "cutlass.T")`` of ``ir.VectorType.get([V], cutlass.T.mlir_type)``;
    None when ``node`` is not that call, a None component when the shape or
    element operand is not the generated spelling."""
    if not (
        isinstance(node, ast.Call)
        and call_path(node.func) == _VECTOR_TYPE
        and len(node.args) == 2
        and not node.keywords
    ):
        return None
    shape, element = node.args
    width = None
    if (
        isinstance(shape, ast.List)
        and len(shape.elts) == 1
        and isinstance(shape.elts[0], ast.Constant)
        and type(shape.elts[0].value) is int
    ):
        width = shape.elts[0].value
    carrier = None
    if isinstance(element, ast.Attribute) and element.attr == "mlir_type":
        carrier = ast.unparse(element.value)
    return width, carrier


def match_packet_access(node: ast.AST) -> PacketAccess | None:
    """Parse ``node`` as one of the generated load / store call forms."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("load", "store"):
        path = call_path(func)
        if path not in (_ARCH_LOAD, _ARCH_STORE):
            if func.attr == "load" and not node.args and not node.keywords:
                return PacketAccess("load", node, func.value)
            if func.attr == "store" and len(node.args) == 1 and not node.keywords:
                return PacketAccess("store", node, func.value)
            return None
        if path == _ARCH_STORE or len(node.args) != 2:
            return None
        pointer, operand = node.args
        keywords = tuple(node.keywords)
        if (parts := vector_type_parts(operand)) is not None:
            width, carrier = parts
            return PacketAccess(
                "packet", node, pointer, width, carrier, None, None, keywords
            )
        if isinstance(operand, (ast.Attribute, ast.Name)):
            return PacketAccess(
                "arch_load",
                node,
                pointer,
                None,
                ast.unparse(operand),
                None,
                None,
                keywords,
            )
        return None
    if not isinstance(func, ast.Name):
        return None
    name = func.id
    if name in _CUTE_CACHE_LOAD_HELPER_NAMES:
        if len(node.args) != 2 or (parts := vector_type_parts(node.args[1])) is None:
            return None
        width, carrier = parts
        return PacketAccess(
            "hinted_packet",
            node,
            node.args[0],
            width,
            carrier,
            name,
            None,
            tuple(node.keywords),
        )
    if name in MASKED_PACKET_LOAD_HELPER_NAMES:
        if (
            len(node.args) not in (3, 4)
            or (parts := vector_type_parts(node.args[2])) is None
        ):
            return None
        width, carrier = parts
        return PacketAccess(
            "masked_packet",
            node,
            node.args[0],
            width,
            carrier,
            name,
            node.args[1],
            tuple(node.keywords),
        )
    if name in PACKET_FLUSH_HELPER_NAMES:
        if len(node.args) != 2 or node.keywords:
            return None
        return PacketAccess("flush", node, node.args[0], None, None, name)
    return None


def is_store_call(call: ast.Call) -> bool:
    """Whether ``call`` is a generated memory write: a store, an atomic or
    vector reduction into memory, a bulk copy, or the persistent
    vectorizer's store marker."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return (
            func.attr in ("store", "__setitem__")
            or func.attr.startswith("atomic_")
            # ``cute.copy(atom, src, dst)``: a bulk write to its destination.
            or (
                func.attr == "copy"
                and isinstance(func.value, ast.Name)
                and func.value.id == "cute"
            )
        )
    return isinstance(func, ast.Name) and (
        func.id == PERSISTENT_BRANCH_VEC_STORE
        # ``_cute_red_add_f32_vec(ptr, vals)``: a vector reduction into memory.
        or func.id.startswith(
            ("_cute_store_", "_cute_rank1_store_", "_cute_atomic_", "_cute_red_")
        )
    )


def store_pointer(call: ast.Call) -> ast.expr | None:
    """The address a generated write (``is_store_call``) writes through;
    None when its destination is not an address operand (``cute.copy``'s
    tensor view, the ``_cute_rank1_store_*`` helpers) and for any other
    call."""
    func = call.func
    if isinstance(func, ast.Name):
        if func.id == PERSISTENT_BRANCH_VEC_STORE:
            return call.args[3] if len(call.args) == 6 else None
        if func.id.startswith(("_cute_store_", "_cute_atomic_", "_cute_red_")):
            return call.args[0] if call.args else None
        return None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr.startswith("atomic_"):
        return call.args[0] if call.args else None
    if func.attr == "store":
        # ``cute.arch.store(ptr, value, ...)`` is a free function, while
        # ``ptr.store(value)`` carries its destination on ``func.value``.
        if call_path(func) == _ARCH_STORE:
            return call.args[0] if call.args else None
        return func.value
    if func.attr == "__setitem__":
        return func.value
    return None


def call_write_roots(
    call: ast.Call, roots: Callable[[ast.expr], frozenset[str] | None]
) -> frozenset[str] | None:
    """The tensors ``call`` may write, ``roots`` naming those an address
    reaches (None: not resolved): empty when the call writes no tensor,
    None when it may write memory it does not name.

    A generated write (``is_store_call``) writes the ``roots`` of its
    address, or memory it does not name when it has none
    (``store_pointer``).  A call writes no tensor when the relocation proofs
    accept it with its loads and CTA reductions, when ``_NO_MEMORY_WRITE``
    names it (iterators, fresh register tensors, the load marker, and the
    calls that touch only shared memory or barrier state), when it is a
    trace-time list's ``name.append(value)``, and when it is inline asm
    declared ``is_pure``: that asm is lowered without side effects
    (``inline_asm_helpers``), so the compiler already assumes it writes
    nothing.  Any other call, impure inline asm among them, may write
    anything.
    """
    if is_store_call(call):
        pointer = store_pointer(call)
        return None if pointer is None else roots(pointer)
    func = call.func
    name = call_path_str(func)
    if name == "_cute_inline_asm_elementwise":
        pure = any(
            keyword.arg == "is_pure"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        return frozenset() if pure else None
    if (
        name in _NO_MEMORY_WRITE
        or (
            isinstance(func, ast.Attribute)
            and func.attr == "append"
            and isinstance(func.value, ast.Name)
        )
        or _is_proven_relocatable_call(call, allow_load=True, allow_reduction=True)
    ):
        return frozenset()
    return None


def fragment_writes(call: ast.Call) -> set[str]:
    """The names of the register fragments ``call`` stores into in place,
    which ``call_write_roots`` leaves out since they are no tensor: the
    ``results`` of the fragment grouped reduce (all the names its argument
    reads, when it is no plain name)."""
    if call_path_str(call.func) != ReduceScope.SHARED_TWO_STAGE_FRAGMENT.value:
        return set()
    if len(call.args) < 2:
        return set()
    return {node.id for node in ast.walk(call.args[1]) if isinstance(node, ast.Name)}


def vector_type_source(width: int, carrier: str) -> str:
    """``ir.VectorType.get([V], cutlass.T.mlir_type)``."""
    return f"ir.VectorType.get([{width}], {carrier}.mlir_type)"


def render_packet_load(
    pointer: str,
    width: int,
    carrier: str,
    *,
    helper: str | None = None,
    kwargs: str = "",
) -> str:
    """The ``packet`` form (``cute.arch.load`` with ``kwargs``, the
    ``CachePolicy.load_kwargs`` fragment) or, with ``helper``, the
    ``hinted_packet`` form of a ``width`` x ``carrier`` packet at ``pointer``."""
    vector_type = vector_type_source(width, carrier)
    if helper is not None:
        return f"{helper}({pointer}, {vector_type})"
    return f"cute.arch.load({pointer}, {vector_type}{kwargs})"


def render_masked_packet_load(
    helper: str, pointer: str, predicate: str, width: int, carrier: str, priority: str
) -> str:
    """The ``masked_packet`` form: ``helper(pointer, predicate, vector_type[, priority])``."""
    arguments = f"{pointer}, {predicate}, {vector_type_source(width, carrier)}"
    if priority:
        arguments += f", {priority!r}"
    return f"{helper}({arguments})"
