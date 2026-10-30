"""Device IR graph passes for the Spyre backend.

Stage 1: **destination binding.** The producing node learns where its result
must land *before* emission, so the output operand already **is** the store
destination and the emitter never has to go back and patch the tree.

Why this is a graph pass and not emitter bookkeeping
----------------------------------------------------

The emitter is not append-only. It emits ``OpSpec``s and then rewrites them:
``SpyreEmitter.redirect_output`` walks every already-emitted spec, including
inside closed ``LoopSpec`` bodies, and replaces ``spec.args``. The root cause is
that the emitter must decide a buffer's identity at a point where it does not
yet know it, so it guesses and repairs. Buffering the specs would not help --
that keeps the decision late. Making the decision **earlier**, as a graph
annotation, is what turns emission into a pure function of the annotated graph.

Slot: ``DeviceIR.build_codegen_graphs``, mirroring ``metal``'s
``rewrite_mpp_graphs``. That function is a graph *producer* -- it owns the
copies and its contract is "return the graphs codegen should use" -- whereas
``pre_codegen`` is a codegen participant that receives a list it does not own.
The deciding fact is that ``_generate_ast`` can run up to three times for one
kernel and passes ``_codegen_graphs`` through unchanged on retry, which
``generate_ast.py`` already records as hazardous for an in-place graph
annotation. Here the question cannot arise: ``build_codegen_graphs`` makes
fresh copies on every call.

This pass must also run *after* ``_apply_rolling``, because a rolled reduction
is what creates the loop carry the pass binds -- only the producer can
guarantee that order.

**The ``meta`` trap, measured in this checkout.** ``GraphInfo.copy()`` uses
``fx.Graph.node_copy``, which gives each copied node a *new* ``meta`` dict
holding the *same* values::

    meta dict shared        = False
    leaked to original      = False     # writing a NEW meta key is safe
    nested mutation leaked  = 99        # mutating a nested value DOES leak

So this module may only ever **assign fresh keys** and must never mutate
anything reached through an existing key. Violating that corrupts the original
Device IR, which is shared across every config the autotuner tries -- a
cache-poisoning bug that would present as config-dependent nondeterminism.

Decline, do not raise
---------------------

An unrecognised or ambiguous chain leaves the graph unchanged and the emitter
takes its existing rewrite path, following metal's rule that "unsupported or
ambiguous shapes return no candidate and leave the graph unchanged". That is
sound *only while the fallback exists*: once ``redirect_output`` is deleted, a
case that cannot be annotated must raise instead, or it would emit a kernel
with no destination bound -- precisely the silent-wrongness class this work
exists to remove.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language.creation_ops import full
from ...language.memory_ops import store
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ...runtime.config import Config
    from ..device_ir import DeviceIR


#: ``node.meta`` key carrying the bound destination. Read by the Spyre
#: emitters; absent means "not annotated", which is a decline, not an error.
SPYRE_DEST = "spyre_dest"


@dataclasses.dataclass(frozen=True)
class SpyreDest:
    """Where a producing node's result must be written.

    The **subscript is carried, not just the tensor**: a rolled reduction stores
    ``out[tile_m, :] = acc``, so the accumulator is tile-shaped while ``out`` is
    not, and the subscript is what tells the emitter which host dims the value's
    symbols index. That is the information ``_output_symbols`` reconstructs
    today.

    ``tensor`` is the store's ``proxy_arg(0)``, and it is the *identical*
    ``FakeTensor`` object codegen resolves -- verified in this checkout: the
    object at ``store_node.args[0].meta["val"]`` is the same ``id()`` as
    ``state.proxy_arg(0)``, and both key the same ``device_function.tensor_arg``
    entry. That identity is what lets the emitter turn this annotation back into
    the destination ``Operand``.
    """

    tensor: torch.Tensor
    subscript: tuple[object, ...]
    #: Provenance, for error messages and for the agreement check.
    store_node: Node


def rewrite_spyre_graphs(device_ir: DeviceIR, config: Config) -> None:
    """Annotate each single-owner stored value with its destination.

    Adds no node, removes no node, rebuilds no graph and rewrites no
    ``meta["lowering"]`` -- unlike metal's pass, which is a graph
    *restructurer*. This is a graph *annotator*: the graph's shape is identical
    before and after, and the only difference is one new ``meta`` key. That is
    what makes the annotation separately testable against the behaviour it
    replaces.

    ``config`` is unused in Stage 1 and is taken so the signature does not have
    to change when allocation (Stage 3) needs ``spyre_lx_buffers``.
    """
    for graph_info in list(device_ir.graphs):
        if not isinstance(graph_info, RootGraphInfo):
            continue
        for store_node in _store_nodes(graph_info):
            producer = _producer_for_store(device_ir, store_node)
            if producer is None:
                continue
            producer.meta[SPYRE_DEST] = SpyreDest(
                tensor=store_node.args[0].meta["val"],
                subscript=tuple(store_node.args[1]),
                store_node=store_node,
            )


def _store_nodes(graph_info: RootGraphInfo) -> list[Node]:
    """Every ``hl.store`` call in a root graph, in program order."""
    return [
        node
        for node in graph_info.graph.nodes
        if node.op == "call_function" and node.target is store
    ]


def _producer_for_store(device_ir: DeviceIR, store_node: Node) -> Node | None:
    """The node whose output buffer should *be* this store's destination.

    Two chains, both present in the drivers:

    1. **Direct**: ``<compute> -> store``. The compute node is annotated.
    2. **Loop carry**: ``hl.full -> _for_loop -> getitem -> _phi -> store``. The
       ``hl.full`` node is annotated, so the accumulator is allocated *as* the
       destination from the start and the combine then accumulates in place.
       This is what the emitter's own reference says is required ("the
       accumulator must *be* the store destination"), and it is structurally the
       same shape metal matches as ``getitem -> _phi -> ... -> store``.

    Returns ``None`` -- a decline, leaving the graph unchanged -- for anything
    else, or when the chain is not single-owner.
    """
    if len(store_node.args) < 3:
        return None
    dest_node = store_node.args[0]
    value = store_node.args[2]
    if not isinstance(dest_node, torch.fx.Node) or not isinstance(value, torch.fx.Node):
        return None
    if not isinstance(dest_node.meta.get("val"), torch.Tensor):
        return None
    if not _subscript_is_modelled(store_node.args[1]):
        return None

    # Single-owner, exactly as metal requires: a value with another user needs
    # its own buffer plus an explicit copy, so binding its producer straight to
    # the destination would make the second user read the store target.
    if list(value.users) != [store_node]:
        return None

    if _is_phi(value):
        return _carry_root(device_ir, value)
    if value.op == "call_function" and _is_compute(value):
        return value
    return None


def _carry_root(device_ir: DeviceIR, phi_node: Node) -> Node | None:
    """The ``hl.full`` behind a ``_phi``, if the carry chain is single-owner.

    The chain is ``full -> _for_loop -> getitem -> _phi``, and ``_phi``'s two
    args are the pre-loop value and the loop result -- the same node reached two
    ways, which is what makes binding the ``full`` sufficient to bind the whole
    carry.
    """
    args = [a for a in phi_node.args if isinstance(a, torch.fx.Node)]
    if len(args) != 2:
        return None
    pre_loop, loop_result = args

    if not _is_getitem(loop_result):
        return None
    for_loop = loop_result.args[0]
    if not isinstance(for_loop, torch.fx.Node) or not _is_for_loop(for_loop):
        return None
    # The loop result must be consumed only by this phi: a second consumer would
    # read the accumulator after it has become the store destination.
    if list(loop_result.users) != [phi_node]:
        return None
    if list(for_loop.users) != [loop_result]:
        return None

    if not _is_full(pre_loop):
        return None
    # ``full`` legitimately has two users -- the _for_loop that carries it and
    # the _phi that closes it -- and nothing else may read it.
    if set(pre_loop.users) != {for_loop, phi_node}:
        return None
    if not _loop_body_accumulates_in_place(device_ir, for_loop):
        return None
    return pre_loop


def _loop_body_accumulates_in_place(device_ir: DeviceIR, for_loop: Node) -> bool:
    """Does the loop body combine into the carry rather than rebuild it?

    Binding the ``hl.full`` to the destination is only correct if every
    iteration *updates* the carry in place. The emitter already relies on this
    (``_carried_input`` routes a carried operand to the op's output), but the
    pass must not assume it: a body whose returned value does not descend from
    the carry placeholder would have the store destination written once and the
    rest of the loop's work discarded.
    """
    if not for_loop.args or not isinstance(for_loop.args[0], int):
        return False
    graph_id = for_loop.args[0]
    if not (0 <= graph_id < len(device_ir.graphs)):
        return False
    body = device_ir.graphs[graph_id]
    if not isinstance(body, ForLoopGraphInfo):
        return False

    outputs = [n for n in body.graph.nodes if n.op == "output"]
    if len(outputs) != 1:
        return False
    returned = outputs[0].args[0]
    if not isinstance(returned, (list, tuple)) or len(returned) != 1:
        return False
    result = returned[0]
    if not isinstance(result, torch.fx.Node):
        return False

    placeholders = [n for n in body.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != 1:
        return False
    return _descends_from(result, placeholders[0])


def _descends_from(node: Node, ancestor: Node) -> bool:
    """Is ``ancestor`` reachable from ``node`` through input edges?"""
    seen: set[Node] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current is ancestor:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(current.all_input_nodes)
    return False


def _subscript_is_modelled(subscript: object) -> bool:
    """Would ``memory_ops._reject_narrowing`` accept this subscript?

    Mirrored rather than called, because at **graph** time a subscript element
    is not yet the value the codegen handler sees. ``_reject_narrowing`` runs on
    ``state.proxy_arg(1)`` and so inspects resolved ``SymInt``/``int``/``slice``
    objects; here each index is either a ``slice`` or an ``fx.Node`` whose
    ``meta["val"]`` holds the block symbol. Checking for ``SymInt`` directly at
    this point rejects every store, which is how this was found: the first
    version of the pass annotated nothing at all and every gate still passed.

    A **tensor**-valued index is a scatter, which has no single destination to
    bind, so it is declined -- matching ``_indirect_dim``'s test on the load
    side.
    """
    if not isinstance(subscript, (list, tuple)):
        return False
    for index in subscript:
        if isinstance(index, slice):
            if index != slice(None):
                return False
        elif isinstance(index, torch.fx.Node):
            val = index.meta.get("val")
            if isinstance(val, torch.Tensor):
                return False
            if not isinstance(val, (torch.SymInt, int)):
                return False
        elif not isinstance(index, (torch.SymInt, int)):
            return False
    return True


def _is_compute(node: Node) -> bool:
    """Is this a node the Spyre compute lowerings emit an OpSpec for?

    A ``load`` returns a buffer handle and emits nothing, so a bare
    ``out[...] = x[...]`` copy has no producing OpSpec to bind and must stay on
    the existing path.
    """
    from .optable import spyre_op_name

    return spyre_op_name(node.target) is not None


def _is_phi(node: Node) -> bool:
    return node.op == "call_function" and node.target is _tracing_ops._phi


def _is_full(node: Node) -> bool:
    return node.op == "call_function" and node.target is full


def _is_getitem(node: object) -> bool:
    import operator

    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is operator.getitem
        and len(node.args) >= 2
    )


def _is_for_loop(node: Node) -> bool:
    return node.op == "call_function" and _tracing_ops.is_for_loop_target(node.target)


# -- the codegen side: resolving an annotation to a buffer --------------------


@dataclasses.dataclass(frozen=True)
class BoundDest:
    """A resolved destination, and whether writing it needs a relayout.

    ``relayout`` is the **widened** half of Stage 1. When the computed value's
    stick axis differs from the destination's, the value cannot be written into
    the destination directly -- but the decision can still be made early, by
    giving the value ``hbm_pool`` residency instead of ``lx``. That is what
    retires ``restage_to_pool``: the closing ``ReStickifyOpHBM`` needs an HBM
    *source*, and today the emitter discovers that only once the store arrives,
    by which time the value is already in ``lx`` and the producing spec has to
    be rewritten. Deciding residency here makes the relayout a plain append.
    """

    operand: object
    relayout: bool


def bound_destination(
    node: Node | None,
    sizes: list[int],
    stick_dim: int,
    dtype: torch.dtype,
) -> BoundDest | None:
    """The destination buffer this node's output should *be*, if one is bound.

    Returns ``None`` -- decline, leaving the caller on its existing
    ``add_intermediate`` path -- when the node carries no annotation or when the
    destination's shape does not match what the op is about to produce.

    The shape check is not a formality. ``memory_ops`` already raises a good
    message on a rank mismatch (a Spyre reduction output is keepdim on the
    device, so a rank-dropped destination is a genuinely different layout), and
    that check must keep firing: declining here leaves the store to raise it
    rather than binding a destination of the wrong rank.
    """
    if node is None:
        return None
    annotation = node.meta.get(SPYRE_DEST)
    if annotation is None:
        return None

    tensor = annotation.tensor
    if list(tensor.size()) != list(sizes) or tensor.dtype != dtype:
        return None

    from ..device_function import DeviceFunction

    # ``CompileEnvironment.backend`` is typed as the base ``Backend``, which has
    # no ``emitter``; ``compute_ops._spyre_backend`` exists to narrow it in one
    # place rather than repeating the cast. Sound by construction: this is only
    # reached from the Spyre lowerings.
    from .compute_ops import _spyre_backend

    name = DeviceFunction.current().tensor_arg(tensor).name
    operand = _spyre_backend().emitter.add_arg(name, tensor)
    return BoundDest(operand=operand, relayout=operand.stick_dim != stick_dim)


#: How many stores took the bound-destination path and how many fell back.
#:
#: This is the plan's exit criterion as a live counter rather than a one-off
#: measurement: the fallback may only be deleted once this shows it unused.
#: Measured before any of this landed, the three rewrite entry points were
#: called 37 (``redirect_output``), 10 (``restage_to_pool``) and 10
#: (``emit_restickify_into``) times across the driver set.
BINDING: dict[str, int] = {"bound": 0, "relayout": 0, "fallback": 0}
