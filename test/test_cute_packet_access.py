"""The generated memory-access call grammar (``cute/packet_access.py``).

The table below is the acceptance set of the matcher: every load / store
call shape the CPU render corpus of ``test/test_cute_*.py`` contains
(pointer and value operands abstracted to ``PTR`` / ``V``), harvested from a
codedump of the whole cute suite on 2026-10-31, plus the forms the emitters
can produce that the corpus happened not to exercise (a predicated packet
with a priority, the L2 evict-last flushes).  A new call form an emitter
starts producing belongs in this table.
"""

from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.cache_policy_loads import CachePolicy
from helion._compiler.cute.packet_access import PacketAccess
from helion._compiler.cute.packet_access import call_write_roots
from helion._compiler.cute.packet_access import is_store_call
from helion._compiler.cute.packet_access import match_packet_access
from helion._compiler.cute.packet_access import render_masked_packet_load
from helion._compiler.cute.packet_access import render_packet_load
from helion._compiler.cute.packet_access import store_pointer
from helion._compiler.cute.packet_access import vector_type_parts
from helion._compiler.cute.packet_flush_names import PACKET_FLUSH_HELPER_NAMES
from helion._compiler.cute.packet_flush_names import PACKET_FLUSH_L2_EVICT_LAST
from helion._compiler.cute.packet_flush_names import PACKET_FLUSH_STREAMING
from helion._compiler.cute.persistent_branch_vec import PACKET_STORE_BYTES

# Grammar and table pins over the AST and the emitted strings: no GPU.
pytestmark = pytest.mark.cpu_only

_U16 = "cutlass.Uint16"
_U32 = "cutlass.Uint32"


def _vt(width: int, carrier: str) -> str:
    return f"ir.VectorType.get([{width}], {carrier}.mlir_type)"


# (call source, expected (kind, width, carrier, helper, has predicate, keywords))
CORPUS_FORMS: list[tuple[str, tuple[object, ...] | None]] = [
    # scalar forms
    ("(PTR).load()", ("load", None, None, None, False, ())),
    ("(PTR).store(V)", ("store", None, None, None, False, ())),
    (
        "cute.arch.load(PTR, cutlass.Uint8)",
        ("arch_load", None, "cutlass.Uint8", None, False, ()),
    ),
    ("cute.arch.load(PTR, cutlass.Uint32)", ("arch_load", None, _U32, None, False, ())),
    (
        "cute.arch.load(PTR, cutlass.Uint64)",
        ("arch_load", None, "cutlass.Uint64", None, False, ()),
    ),
    (
        "cute.arch.load(PTR, cutlass.Float32, level1_eviction_priority='evict_last')",
        (
            "arch_load",
            None,
            "cutlass.Float32",
            None,
            False,
            (("level1_eviction_priority", "evict_last"),),
        ),
    ),
    (
        "cute.arch.load(PTR, cutlass.Uint64, cop='cs')",
        ("arch_load", None, "cutlass.Uint64", None, False, (("cop", "cs"),)),
    ),
    # plain packets
    (f"cute.arch.load(PTR, {_vt(8, _U16)})", ("packet", 8, _U16, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(4, _U32)})", ("packet", 4, _U32, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(4, _U16)})", ("packet", 4, _U16, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(2, _U32)})", ("packet", 2, _U32, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(2, _U16)})", ("packet", 2, _U16, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(8, _U32)})", ("packet", 8, _U32, None, False, ())),
    (f"cute.arch.load(PTR, {_vt(16, _U16)})", ("packet", 16, _U16, None, False, ())),
    (
        f"cute.arch.load(PTR, {_vt(8, 'cutlass.Uint8')})",
        ("packet", 8, "cutlass.Uint8", None, False, ()),
    ),
    (
        f"cute.arch.load(PTR, {_vt(8, _U16)}, level1_eviction_priority='evict_first')",
        (
            "packet",
            8,
            _U16,
            None,
            False,
            (("level1_eviction_priority", "evict_first"),),
        ),
    ),
    (
        f"cute.arch.load(PTR, {_vt(4, _U32)}, level1_eviction_priority='evict_first')",
        (
            "packet",
            4,
            _U32,
            None,
            False,
            (("level1_eviction_priority", "evict_first"),),
        ),
    ),
    (
        f"cute.arch.load(PTR, {_vt(2, _U32)}, level1_eviction_priority='evict_last')",
        ("packet", 2, _U32, None, False, (("level1_eviction_priority", "evict_last"),)),
    ),
    (
        f"cute.arch.load(PTR, {_vt(8, _U16)}, cop='cs')",
        ("packet", 8, _U16, None, False, (("cop", "cs"),)),
    ),
    # L2-policy helper packets
    (
        f"_cute_load_l2_evict_last(PTR, {_vt(8, _U16)})",
        ("hinted_packet", 8, _U16, "_cute_load_l2_evict_last", False, ()),
    ),
    (
        f"_cute_load_l2_evict_last(PTR, {_vt(4, _U32)})",
        ("hinted_packet", 4, _U32, "_cute_load_l2_evict_last", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_last(PTR, {_vt(8, _U16)})",
        ("hinted_packet", 8, _U16, "_cute_load_l1_l2_evict_last", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_last(PTR, {_vt(4, _U32)})",
        ("hinted_packet", 4, _U32, "_cute_load_l1_l2_evict_last", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_first(PTR, {_vt(4, _U32)})",
        ("hinted_packet", 4, _U32, "_cute_load_l1_l2_evict_first", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_first(PTR, {_vt(8, _U16)})",
        ("hinted_packet", 8, _U16, "_cute_load_l1_l2_evict_first", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_last_8b(PTR, {_vt(2, _U32)})",
        ("hinted_packet", 2, _U32, "_cute_load_l1_l2_evict_last_8b", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_last_8b(PTR, {_vt(4, _U16)})",
        ("hinted_packet", 4, _U16, "_cute_load_l1_l2_evict_last_8b", False, ()),
    ),
    (
        f"_cute_load_l2_evict_last_8b(PTR, {_vt(4, _U16)})",
        ("hinted_packet", 4, _U16, "_cute_load_l2_evict_last_8b", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_first_8b(PTR, {_vt(2, _U32)})",
        ("hinted_packet", 2, _U32, "_cute_load_l1_l2_evict_first_8b", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_first_8b(PTR, {_vt(4, _U16)})",
        ("hinted_packet", 4, _U16, "_cute_load_l1_l2_evict_first_8b", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_last_4b(PTR, {_vt(2, _U16)})",
        ("hinted_packet", 2, _U16, "_cute_load_l1_l2_evict_last_4b", False, ()),
    ),
    (
        f"_cute_load_l1_l2_evict_first_4b(PTR, {_vt(2, _U16)})",
        ("hinted_packet", 2, _U16, "_cute_load_l1_l2_evict_first_4b", False, ()),
    ),
    (
        f"_cute_load_nc_l2_evict_first_32b(PTR, {_vt(16, _U16)})",
        ("hinted_packet", 16, _U16, "_cute_load_nc_l2_evict_first_32b", False, ()),
    ),
    (
        f"_cute_load_nc_l2_evict_first_16b(PTR, {_vt(8, _U16)})",
        ("hinted_packet", 8, _U16, "_cute_load_nc_l2_evict_first_16b", False, ()),
    ),
    # predicated packets of the V-loop sink
    (
        f"_cute_load_v16b_if(PTR, PRED, {_vt(4, _U32)})",
        ("masked_packet", 4, _U32, "_cute_load_v16b_if", True, ()),
    ),
    (
        f"_cute_load_v16b_if(PTR, PRED, {_vt(8, _U16)})",
        ("masked_packet", 8, _U16, "_cute_load_v16b_if", True, ()),
    ),
    (
        f"_cute_load_v8b_if(PTR, PRED, {_vt(4, _U16)})",
        ("masked_packet", 4, _U16, "_cute_load_v8b_if", True, ()),
    ),
    (
        f"_cute_load_v32b_if(PTR, PRED, {_vt(8, _U32)})",
        ("masked_packet", 8, _U32, "_cute_load_v32b_if", True, ()),
    ),
    (
        f"_cute_load_v16b_if(PTR, PRED, {_vt(4, _U32)}, 'evict_last')",
        ("masked_packet", 4, _U32, "_cute_load_v16b_if", True, ()),
    ),
    # packet flushes
    (
        "_cute_store_u32_vec(PTR, V)",
        ("flush", None, None, "_cute_store_u32_vec", False, ()),
    ),
    (
        "_cute_store_u16_vec(PTR, V)",
        ("flush", None, None, "_cute_store_u16_vec", False, ()),
    ),
    (
        "_cute_store_fp8e4m3fn_vec(PTR, V)",
        ("flush", None, None, "_cute_store_fp8e4m3fn_vec", False, ()),
    ),
    (
        "_cute_store_fp8e5m2_vec(PTR, V)",
        ("flush", None, None, "_cute_store_fp8e5m2_vec", False, ()),
    ),
    (
        "_cute_store_u16x8_l2_evict_last(PTR, V)",
        ("flush", None, None, "_cute_store_u16x8_l2_evict_last", False, ()),
    ),
    (
        "_cute_store_u32x4_l2_evict_last(PTR, V)",
        ("flush", None, None, "_cute_store_u32x4_l2_evict_last", False, ()),
    ),
    # not accesses: other helpers, host code, pointer-table reads with arguments
    (
        "_cute_store_shared_remote_x4(PTR, V, V, V, smem_ptr=S, mbar_ptr=M, peer_cta_rank_in_cluster=R)",
        None,
    ),
    ('PyCodeCache.load("source")', None),
    ("cute.arch.store(PTR, V)", None),
    ("cute.arch.thread_idx()", None),
    ("cutlass.Float32(V)", None),
    ("x.bitcast(cutlass.Float32)", None),
]


def _parse(source: str) -> ast.Call:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


@pytest.mark.parametrize(
    ("source", "expected"), CORPUS_FORMS, ids=[s for s, _ in CORPUS_FORMS]
)
def test_corpus_forms(source: str, expected: tuple[object, ...] | None) -> None:
    access = match_packet_access(_parse(source))
    if expected is None:
        assert access is None
        return
    assert access is not None
    kind, width, carrier, helper, has_predicate, keywords = expected
    assert access.kind == kind
    assert access.width == width
    assert access.carrier == carrier
    assert access.helper == helper
    assert (access.predicate is not None) == has_predicate
    assert (
        tuple((k.arg, ast.literal_eval(k.value)) for k in access.keywords) == keywords
    )
    assert ast.unparse(access.pointer) == "PTR"
    assert access.is_load == (kind not in ("store", "flush"))
    assert access.is_vector == (kind in ("packet", "hinted_packet", "masked_packet"))
    if access.is_vector:
        assert access.vector_type is not None
        assert vector_type_parts(access.vector_type) == (width, carrier)
    else:
        assert access.vector_type is None


def test_flush_helper_names_pin_the_store_tables() -> None:
    """The flushes the matcher recognizes are exactly the plain flushes the
    branch-local vectorizer emits (its dtype table, ``PACKET_STORE_BYTES``)
    and their L2 evict-last and streaming twins; a helper added to one table
    must reach the others."""
    assert set(PACKET_STORE_BYTES) == set(PACKET_FLUSH_L2_EVICT_LAST)
    assert set(PACKET_STORE_BYTES) == set(PACKET_FLUSH_STREAMING)
    every_flush = (
        set(PACKET_FLUSH_L2_EVICT_LAST)
        | set(PACKET_FLUSH_L2_EVICT_LAST.values())
        | set(PACKET_FLUSH_STREAMING.values())
    )
    assert every_flush == PACKET_FLUSH_HELPER_NAMES
    for table in (PACKET_FLUSH_L2_EVICT_LAST, PACKET_FLUSH_STREAMING):
        for plain, hinted in table.items():
            for helper in (plain, hinted):
                access = match_packet_access(_parse(f"{helper}(PTR, V)"))
                assert access is not None and access.kind == "flush"
                assert access.helper == helper and not access.is_load


def test_non_calls_and_malformed_vector_types() -> None:
    assert match_packet_access(ast.parse("PTR", mode="eval").body) is None
    # A vector type whose shape is not one literal: still the packet form,
    # with the parts a consumer must check.
    access = match_packet_access(
        _parse(
            "cute.arch.load(PTR, ir.VectorType.get([4, 2], cutlass.Uint16.mlir_type))"
        )
    )
    assert access is not None and access.kind == "packet"
    assert (access.width, access.carrier) == (None, "cutlass.Uint16")
    # Helpers called with anything but a vector type are not packets.
    assert (
        match_packet_access(_parse("_cute_load_l2_evict_last(PTR, cutlass.Uint16)"))
        is None
    )
    assert match_packet_access(_parse("_cute_load_v16b_if(PTR, PRED)")) is None
    assert match_packet_access(_parse("_cute_store_u16_vec(PTR)")) is None
    # ``.load`` / ``.store`` with other arities are not the scalar forms.
    assert match_packet_access(_parse("x.load(1)")) is None
    assert match_packet_access(_parse("x.store()")) is None
    assert match_packet_access(_parse("x.store(a, b)")) is None


def _inline_asm(pure: bool) -> str:
    return (
        "_cute_inline_asm_elementwise(a, asm='', constraints='=f,f', "
        f"dtype=cutlass.Float32, is_pure={pure})"
    )


def _iterator_roots(pointer: ast.expr) -> frozenset[str] | None:
    """The tensors whose ``.iterator`` an address reads, None if none."""
    roots = frozenset(
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    )
    return roots or None


@pytest.mark.parametrize(
    ("source", "store", "pointer", "roots"),
    [
        # ``ptr.store(value)`` writes through its receiver, ``cute.arch``
        # calls in the address included; ``cute.arch.store`` takes it first.
        (
            "(x.iterator + cute.arch.thread_idx()[0]).store(v)",
            True,
            "x.iterator + cute.arch.thread_idx()[0]",
            frozenset({"x"}),
        ),
        (
            "cute.arch.store(x.iterator + 1, v)",
            True,
            "x.iterator + 1",
            frozenset({"x"}),
        ),
        (
            "_cute_store_u32_vec(y.iterator + 4, values)",
            True,
            "y.iterator + 4",
            frozenset({"y"}),
        ),
        # A write whose destination is no address operand may write memory
        # it does not name.
        ("cute.copy(atom, source, target)", True, None, None),
        ("_cute_rank1_store_f32(target, v)", True, None, None),
        # Inline asm declared pure writes nothing, impure asm and an unknown
        # call may write anything, and a load writes nothing.
        (
            _inline_asm(pure=True),
            False,
            None,
            frozenset(),
        ),
        (
            _inline_asm(pure=False),
            False,
            None,
            None,
        ),
        # Warp shuffles and the pure selects of ``cutlass`` write nothing.
        ("cute.arch.shuffle_sync_bfly(value, offset=1)", False, None, frozenset()),
        ("cutlass.select_(mask, a, b)", False, None, frozenset()),
        ("mystery(x.iterator)", False, None, None),
        ("(x.iterator + 1).load()", False, None, frozenset()),
    ],
)
def test_the_write_recognizers(
    source: str, store: bool, pointer: str | None, roots: frozenset[str] | None
) -> None:
    call = ast.parse(source, mode="eval").body
    assert isinstance(call, ast.Call)
    assert is_store_call(call) is store
    address = store_pointer(call)
    assert (None if address is None else ast.unparse(address)) == pointer
    assert call_write_roots(call, _iterator_roots) == roots


def test_cache_policy_of_keywords() -> None:
    plain = match_packet_access(_parse(f"cute.arch.load(PTR, {_vt(8, _U16)})"))
    assert plain is not None and plain.cache_policy() == CachePolicy.NONE
    hinted = match_packet_access(
        _parse(
            f"cute.arch.load(PTR, {_vt(8, _U16)}, level1_eviction_priority='evict_first')"
        )
    )
    assert hinted is not None and hinted.cache_policy() == CachePolicy(
        l1_priority="evict_first"
    )
    streaming = match_packet_access(
        _parse("cute.arch.load(PTR, cutlass.Float32, cop='cs')")
    )
    assert streaming is not None and streaming.cache_policy() == CachePolicy(
        streaming=True
    )
    other = match_packet_access(_parse("cute.arch.load(PTR, cutlass.Float32, cop=x)"))
    assert other is not None and other.cache_policy() is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (s, e)
        for s, e in CORPUS_FORMS
        if e is not None and e[0] in ("packet", "hinted_packet", "masked_packet")
    ],
    ids=[
        s
        for s, e in CORPUS_FORMS
        if e is not None and e[0] in ("packet", "hinted_packet", "masked_packet")
    ],
)
def test_render_round_trips_the_corpus(
    source: str, expected: tuple[object, ...]
) -> None:
    access = match_packet_access(_parse(source))
    assert isinstance(access, PacketAccess)
    assert access.width is not None and access.carrier is not None
    if access.kind == "masked_packet":
        assert access.predicate is not None and access.helper is not None
        priority = (
            ast.literal_eval(access.call.args[3]) if len(access.call.args) == 4 else ""
        )
        rendered = render_masked_packet_load(
            access.helper,
            "PTR",
            ast.unparse(access.predicate),
            access.width,
            access.carrier,
            priority,
        )
    else:
        policy = access.cache_policy()
        assert policy is not None
        rendered = render_packet_load(
            "PTR",
            access.width,
            access.carrier,
            helper=access.helper,
            kwargs=policy.load_kwargs,
        )
    assert rendered == source


def test_cache_policy_suffix_round_trip() -> None:
    for policy in (
        CachePolicy.NONE,
        CachePolicy(l1_priority="evict_first"),
        CachePolicy(l1_priority="evict_last"),
        CachePolicy(streaming=True),
        CachePolicy(l2_helper="l2_last"),
        CachePolicy(l2_helper="l1_l2_first"),
        CachePolicy(l2_helper="l1_l2_last"),
        CachePolicy(l2_helper="nc_l2_first"),
    ):
        assert CachePolicy.from_suffix(policy.suffix) == policy
        assert policy.is_l2_marker == bool(policy.l2_helper)
        assert policy.load_kwargs == ("" if policy.l2_helper else policy.suffix)
    assert CachePolicy.tunable("first") == CachePolicy(l1_priority="evict_first")
    assert CachePolicy.tunable("last") == CachePolicy(l1_priority="evict_last")
    assert CachePolicy.tunable("streaming") == CachePolicy(streaming=True)
    assert CachePolicy.tunable("l2_last").suffix == "__l2_last__"
    assert CachePolicy.tunable("nc_l2_first").suffix == "__nc_l2_first__"
    assert CachePolicy.tunable("") == CachePolicy.NONE
    assert (
        CachePolicy.explicit("evict_last").suffix
        == ", level1_eviction_priority='evict_last'"
    )
    assert CachePolicy.explicit("") == CachePolicy.NONE
    # Helper selection by packet width: 16/8/4 bytes for the L2 families,
    # 32/16 for the read-only one, none for other widths and other hints.
    assert (
        CachePolicy(l2_helper="l2_last").vector_helper(16) == "_cute_load_l2_evict_last"
    )
    assert (
        CachePolicy(l2_helper="l2_last").vector_helper(8)
        == "_cute_load_l2_evict_last_8b"
    )
    assert (
        CachePolicy(l2_helper="l1_l2_first").vector_helper(4)
        == "_cute_load_l1_l2_evict_first_4b"
    )
    assert CachePolicy(l2_helper="l2_last").vector_helper(2) is None
    assert (
        CachePolicy(l2_helper="nc_l2_first").vector_helper(32)
        == "_cute_load_nc_l2_evict_first_32b"
    )
    assert CachePolicy(l2_helper="nc_l2_first").vector_helper(8) is None
    assert CachePolicy(l1_priority="evict_last").vector_helper(16) is None
    assert CachePolicy(l1_priority="evict_last").masked_priority == "evict_last"
    assert CachePolicy(streaming=True).masked_priority == "cs"
    assert CachePolicy(l2_helper="l2_last").masked_priority == ""
    assert CachePolicy(l1_priority="evict_first").scalar_policy == "first"
    assert CachePolicy(streaming=True).scalar_policy == "streaming"
    assert CachePolicy(l1_priority="evict_normal").scalar_policy is None
