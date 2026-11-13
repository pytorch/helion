from __future__ import annotations

import ast
from itertools import combinations
from types import SimpleNamespace

import pytest
import torch

from helion import exc
from helion._compiler.cute.computed_fragment import FragmentCompiler
from helion._compiler.cute.fragment_context import Fragment
from helion._compiler.cute.fragment_context import FragmentContext
from helion._compiler.cute.fragment_shared_arena import SharedArenaRequest
from helion._compiler.cute.fragment_shared_arena import plan_shared_arena
from helion._compiler.cute.fragment_shared_arena import shared_schedule_is_unchanged
from helion._compiler.cute.fuse_fma import fuse_fma

pytestmark = pytest.mark.cpu_only

BUFFERS = (("a", torch.float32, 32), ("b", torch.int16, 63))
SAFE = """
a[0] = 1
cute.arch.sync_threads()
out[0] = a[0]
cute.arch.sync_threads()
b[0] = 2
cute.arch.sync_threads()
out[1] = b[0]
cute.arch.sync_threads()
"""


def test_byte_plan_preserves_live_values_and_aligned_spans():
    body = ast.parse(SAFE).body
    plan = plan_shared_arena(BUFFERS, set(), body)
    assert plan is not None
    assert plan.byte_size == 128
    assert [view.offset for view in plan.views] == [0, 0]
    assert all(
        view.offset % 16 == 0
        and view.count * view.dtype.itemsize <= view.byte_size
        and view.offset + view.byte_size <= plan.byte_size
        for view in plan.views
    )
    # An old value used after the other buffer is written stays live even if
    # accesses are separated by barriers. Value interference is independent.
    assert plan_shared_arena(BUFFERS, {frozenset({"a", "b"})}, body) is None
    assert shared_schedule_is_unchanged(plan, body, {})


def test_constrained_equal_size_slots_pack_without_name_order_fragmentation():
    # Path a-c-d-b needs two slots. Visiting a,b,c,d instead wastes a third.
    buffers = tuple((name, torch.float32, 32) for name in "abcd")
    edges = {frozenset(pair) for pair in ("ac", "cd", "db")}
    body = ast.parse(
        "\n".join(f"{name}[0] = 1\ncute.arch.sync_threads()" for name in "abcd")
    ).body
    plan = plan_shared_arena(buffers, edges, body)
    assert plan is not None and plan.byte_size == 256
    slots = {
        view.name: (view.offset, view.offset + view.byte_size) for view in plan.views
    }
    for left, right in ("ac", "cd", "db"):
        assert slots[left][1] <= slots[right][0] or slots[right][1] <= slots[left][0]


@pytest.mark.parametrize(
    "source",
    [
        "a[0] = 1\nb[0] = 2",
        "for i in range(3):\n a[i] = 1\n cute.arch.sync_threads()\n b[i] = 2",
        "a[0] = 1\nif flag:\n cute.arch.sync_threads()\nb[0] = 2",
        "a[0] = 1\ncute.arch.sync_threads()\nptr = a\nb[0] = 2",
        "a[0] = 1\ncute.arch.sync_threads()\nconsume(a)\nb[0] = 2",
        "a[0] = 1\ncute.arch.sync_threads()\nwhile flag:\n break\nb[0] = 2",
        "a[0] = 1\ncute.arch.sync_threads()\nb[0] = _cute_inline_asm_elementwise((), is_pure=False)",
        "a[0] = 1\ncute.arch.sync_threads()\nopaque()\nb[0] = 2",
    ],
)
def test_arena_refuses_missing_retirement_and_pointer_escape(source):
    assert plan_shared_arena(BUFFERS, set(), ast.parse(source).body) is None


@pytest.mark.parametrize(
    "index", ["None", ":", "...", "0, None", "0, :", "0, ...", "(0, (None,))"]
)
def test_arena_refuses_alias_producing_subscripts(index):
    # Each expression can retain a view across the apparent last use of a.
    # Nested indices must not evade the same flat scalar-access contract.
    source = (
        f"a[0] = 1\nview = a[{index}]\ncute.arch.sync_threads()\n"
        "b[0] = 2\ncute.arch.sync_threads()\nout[0] = view[0]"
    )
    assert plan_shared_arena(BUFFERS, set(), ast.parse(source).body) is None


@pytest.mark.parametrize(
    "index", ["0", "i * 32 + j", "cutlass.Int32(i)", "indices[i] * 1"]
)
def test_arena_accepts_flat_scalar_subscripts(index):
    body = ast.parse(SAFE.replace("a[0]", f"a[{index}]")).body
    plan = plan_shared_arena(BUFFERS, set(), body)
    assert plan is not None
    assert shared_schedule_is_unchanged(plan, body, {})


def test_allocation_records_each_new_live_interference_pair_once():
    class CountingEdges(set):
        attempted = 0

        def update(self, edges):
            for edge in edges:
                self.attempted += 1
                self.add(edge)

    context = object.__new__(FragmentContext)
    context.buffers = []
    context.smem_bytes = 0
    context.scopes = []
    context.held = []
    context.buffer_interference = CountingEdges()
    context.df = SimpleNamespace(new_var=lambda _: f"buffer{len(context.buffers)}")
    for _ in range(64):
        context.held.append(
            context.allocate(Fragment((4,), torch.float32, lambda _: "0"))
        )
    names = [name for name, _, _ in context.buffers]
    expected = {frozenset(pair) for pair in combinations(names, 2)}
    assert context.buffer_interference == expected
    # Count attempted edge insertions, not wall time: rebuilding old pairs at
    # every allocation grows cubically even though the final graph is quadratic.
    assert context.buffer_interference.attempted == len(expected)


@pytest.mark.parametrize("control", ["for i in range(3)", "while flag"])
def test_loop_backedge_with_complete_retirement_can_reuse(control):
    body = ast.parse(
        f"{control}:\n a[0] = 1\n cute.arch.sync_threads()\n"
        " b[0] = 2\n cute.arch.sync_threads()"
    ).body
    plan = plan_shared_arena(BUFFERS, set(), body)
    assert plan is not None and plan.byte_size == 128


def test_both_conditional_arms_must_retire():
    body = ast.parse(
        "a[0] = 1\nif flag:\n cute.arch.sync_threads()\n"
        "else:\n cute.arch.sync_threads()\nb[0] = 2"
    ).body
    plan = plan_shared_arena(BUFFERS, set(), body)
    assert plan is not None and plan.byte_size == 128


def test_opposite_arms_cannot_alias_without_a_uniform_guard_proof():
    # Barriers inside each arm do not make its guard uniform across threads.
    body = ast.parse(
        "if thread % 2:\n a[0] = 1\n cute.arch.sync_threads()\n"
        "else:\n b[0] = 2\n cute.arch.sync_threads()"
    ).body
    assert plan_shared_arena(BUFFERS, set(), body) is None


@pytest.mark.parametrize(
    "changed",
    [
        SAFE.replace("b[0] = 2", "b[1] = 2"),
        SAFE.replace("cute.arch.sync_threads()", "pass", 1),
        SAFE.replace("b[0] = 2", "if flag:\n b[0] = 2"),
        SAFE.replace("b[0] = 2", "ptr = b\nb[0] = 2"),
        "return\n" + SAFE,
    ],
)
def test_late_shared_schedule_mutation_refuses_packed_storage(changed):
    plan = plan_shared_arena(BUFFERS, set(), ast.parse(SAFE).body)
    assert plan is not None
    assert not shared_schedule_is_unchanged(plan, ast.parse(changed).body, {})


def test_final_schedule_uses_exact_renames_and_controls():
    source = "for i in range(n):\n" + "".join(
        " " + line + "\n" for line in SAFE.strip().splitlines()
    )
    plan = plan_shared_arena(BUFFERS, set(), ast.parse(source).body)
    assert plan is not None
    renamed = (
        source.replace("a[", "first[")
        .replace("b[", "second[")
        .replace("range(n)", "range(limit)")
    )
    names = {"a": "first", "b": "second", "n": "limit"}
    assert shared_schedule_is_unchanged(plan, ast.parse(renamed).body, names)
    assert not shared_schedule_is_unchanged(
        plan, ast.parse(renamed.replace("range(limit)", "range(limit + 1)")).body, names
    )
    assert not shared_schedule_is_unchanged(
        plan, ast.parse(renamed).body, {"a": "first", "b": "first"}
    )


def _allocation_fixture():
    plan = plan_shared_arena(BUFFERS, set(), ast.parse(SAFE).body)
    assert plan is not None
    context = object.__new__(FragmentContext)
    context.shared_arena_plan = plan
    context.shared_arena_name = "arena"
    context.allocator = "allocator"
    context.env = SimpleNamespace(
        backend=SimpleNamespace(
            dtype_str={
                torch.float32: "cutlass.Float32",
                torch.int16: "cutlass.Int16",
            }.__getitem__
        )
    )
    declarations = ast.unparse(ast.Module(body=context.allocations(), type_ignores=[]))
    return SharedArenaRequest(plan, "arena", declarations)


def test_generated_byte_arena_views_preserve_typed_values():
    """Execute the actual allocation AST with CPU pointers, including replay."""
    request = _allocation_fixture()
    allocations = []

    class Pointer:
        def __init__(self, memory, offset=0, dtype=torch.uint8):
            self.memory, self.offset, self.dtype = memory, offset, dtype

        def __add__(self, offset):
            return Pointer(self.memory, self.offset + offset, self.dtype)

    class Allocator:
        def allocate_tensor(self, dtype, layout, byte_alignment):
            assert dtype is torch.uint8 and byte_alignment == 16
            allocations.append((layout, byte_alignment))
            return SimpleNamespace(iterator=Pointer(bytearray(layout[0])))

    cutlass = SimpleNamespace(
        Uint8=torch.uint8,
        Float32=torch.float32,
        Int16=torch.int16,
        utils=SimpleNamespace(SmemAllocator=Allocator),
    )
    cute = SimpleNamespace(
        make_layout=lambda shape: shape,
        recast_ptr=lambda p, dtype: Pointer(p.memory, p.offset, dtype),
        make_tensor=lambda p, shape: torch.frombuffer(
            p.memory, dtype=p.dtype, count=shape[0], offset=p.offset
        ),
        arch=SimpleNamespace(sync_threads=lambda: None),
    )
    namespace = {"cutlass": cutlass, "cute": cute, "out": [0.0, 0]}
    exec(request.allocation_source, namespace)
    assert allocations == [((128,), 16)]
    assert namespace["a"].data_ptr() == namespace["b"].data_ptr()
    body = SAFE.replace("a[0] = 1", "a[0] = 1.25").replace("b[0] = 2", "b[0] = 32767")
    # Snapshot the output scalars just like separate global stores. A tensor
    # view in this CPU model would alias the arena after the next write.
    body = body.replace("= a[0]", "= float(a[0])").replace("= b[0]", "= int(b[0])")
    for _ in range(2):
        exec(body, namespace)
        assert namespace["out"] == [1.25, 32767]


@pytest.mark.parametrize(
    "mutation", ["offset", "alignment", "span", "duplicate", "escape", "order", "late"]
)
def test_final_arena_allocation_mutations_refuse(mutation):
    request = _allocation_fixture()
    source = request.allocation_source
    if mutation == "offset":
        source = source.replace("arena.iterator + 0", "arena.iterator + 16", 1)
    elif mutation == "alignment":
        source = source.replace("byte_alignment=16", "byte_alignment=8")
    elif mutation == "span":
        source = source.replace("(128,)", "(64,)", 1)
    elif mutation == "duplicate":
        source += "\n" + source.splitlines()[-1]
    elif mutation == "escape":
        source += "\nescaped = arena.iterator"
    elif mutation == "order":
        lines = source.splitlines()
        lines[-2], lines[-1] = lines[-1], lines[-2]
        source = "\n".join(lines)
    else:
        source = "a[0] = 1\n" + source
    with pytest.raises(exc.InvalidConfig, match="shared arena allocation"):
        request.validate(ast.parse(source + "\n" + SAFE).body, {})


def test_final_arena_declarations_and_renames_are_checked():
    request = _allocation_fixture()
    body = request.allocation_source + "\n" + SAFE
    request.validate(ast.parse(body).body, {})
    names = {"arena": "renamed_arena", "a": "first"}
    renamed = ast.parse(body)
    for node in ast.walk(renamed):
        if isinstance(node, ast.Name):
            node.id = names.get(node.id, node.id)
    request.validate(renamed.body, names)
    with pytest.raises(exc.InvalidConfig, match="access schedule"):
        request.validate(ast.parse(body.replace("b[0] = 2", "b[1] = 2")).body, {})


def test_final_scalar_fma_pass_preserves_shared_access_proof():
    request = _allocation_fixture()
    source = (
        request.allocation_source
        + "\n"
        + SAFE
        + "\nf = cutlass.Float32(2.0)\nt = f * 3.0\nresult = t + 4.0\n"
    )
    body = fuse_fma(ast.parse(source).body)
    assert "cute.math.fma(" in ast.unparse(ast.Module(body=body, type_ignores=[]))
    request.validate(body, {})
    # Even a call after the last explicit use can address shared memory
    # opaquely. Scalar fusion is not permission to ignore unknown effects.
    body.append(ast.parse("opaque_effect(result)").body[0])
    with pytest.raises(exc.InvalidConfig, match="access schedule"):
        request.validate(body, {})


def test_final_arena_allows_disjoint_prologue_and_other_root():
    request = _allocation_fixture()
    body = (
        "thread = cute.arch.thread_idx()[0]\ncute.arch.sync_threads()\n"
        + request.allocation_source
        + "\n"
        + SAFE
        + "\nother[0] = 9\ncute.arch.sync_threads()\nother[1] = other[0]\ncute.arch.sync_threads()\n"
    )
    request.validate(ast.parse(body).body, {})
    for changed in (
        "escape(a)\n" + body,
        body + "escape(b)\n",
        body + "alias = b[...]\n",
        body + "b[0] = 9\n",
        "opaque_prologue()\n" + body,
        body + "opaque_tail()\n",
        body.replace("b[0] = 2", "unknown_in_arena()\nb[0] = 2"),
    ):
        with pytest.raises(exc.InvalidConfig, match="shared arena"):
            request.validate(ast.parse(changed).body, {})


def _packet_shared_proof_fixture():
    from helion._compiler.cute.packet_loads import _published_dependencies

    slot = Fragment((), torch.float32, lambda _: "slot[0]", True, storage="slot")
    view = Fragment((), torch.float32, lambda _: "slot[0]", True, (slot,))
    # A lazy view is not register-backed: it has transitive shared storage.
    lazy = Fragment((), torch.float32, lambda _: "slot[0]", dependencies=(slot,))
    compiler = object.__new__(FragmentCompiler)
    compiler.thread = "thread"
    compiler.threads = 128
    compiler.buffers = [("slot", torch.float32, 1), ("unused", torch.float32, 1)]
    compiler.buffer_interference = set()
    compiler.scopes = [{"lazy": lazy}]
    compiler.held = []
    compiler.pending_local_atomics = set()
    compiler.cg = SimpleNamespace(
        statements_stack=[
            ast.parse(
                "for owner in range(thread, 1, 128):\n    slot[0] = cutlass.Float32(3)\ncute.arch.sync_threads()\n"
            ).body
        ]
    )
    return compiler, slot, view, lazy, _published_dependencies


def test_fragment_packet_shared_publication_and_transitive_lifetime():
    compiler, slot, view, lazy, proof = _packet_shared_proof_fixture()
    assert proof(compiler, (lazy,)) == frozenset({"slot"})
    assert proof(compiler, (view,)) == frozenset({"slot"})
    result = Fragment((4,), torch.float32, lambda _: "0")
    compiler.held.append((result, lazy))
    compiler.scopes.clear()
    compiler.smem_bytes = 0
    selected = compiler.allocate(result)
    assert selected.storage == "unused" and selected.storage != slot.storage
    assert compiler.live_buffers() == {"slot"}
    assert compiler.buffer_interference == {frozenset({"slot", "unused"})}


@pytest.mark.parametrize(
    "bad",
    [
        "pending",
        "alias",
        "later_write",
        "multiple",
        "missing_publication",
        "unknown_storage",
        "register",
    ],
)
def test_fragment_packet_shared_publication_declines(bad):
    compiler, slot, _view, lazy, proof = _packet_shared_proof_fixture()
    dependencies = (lazy,)
    body = compiler.cg.statements_stack[-1]
    if bad == "pending":
        compiler.pending_local_atomics.add("slot")
    elif bad == "alias":
        body.extend(ast.parse("alias = slot").body)
    elif bad == "later_write":
        body.extend(
            ast.parse(
                "for owner in range(thread, 1, 128):\n    slot[0] = cutlass.Float32(9)"
            ).body
        )
    elif bad == "multiple":
        body.extend(
            ast.parse("slot[0] = cutlass.Float32(9)\ncute.arch.sync_threads()").body
        )
    elif bad == "missing_publication":
        body.pop()
    elif bad == "unknown_storage":
        compiler.buffers.clear()
    else:
        register = Fragment((), torch.float32, lambda _: "private", True)
        dependencies = (lazy, register)
    assert not proof(compiler, dependencies)
