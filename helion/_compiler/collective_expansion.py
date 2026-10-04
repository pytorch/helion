"""Expands ``hl.all_reduce`` statements into remote-copy exchanges.

An all-reduce is a statement of the host body, not a device op: each
algorithm is a few top-level device loops (stage the source, push it to the
peers, combine what arrived), and expanding it into them lets the megakernel
planner schedule, fuse and overlap those loops like any other.

Two algorithms are emitted for every site under a host ``if`` on the payload
size, which type propagation resolves from the shapes, so the dead one is
dropped before lowering (see ``fold_static_host_ifs``):

* one-shot: every rank pushes its whole source to every peer, then sums the
  ``world`` copies.  One exchange round, ``world - 1`` full payloads out.
* reduce-scatter + all-gather: rank ``c`` receives every rank's column chunk
  ``c`` and sums it, then pushes the reduced chunk, in the destination dtype,
  to every peer.  Two rounds, but only ``2 (world - 1) / world`` payloads out.

The latency of the extra round decides small payloads, the bandwidth large
ones.  Both sum the ``world`` partials in rank order before the cast to the
destination dtype, so they give bitwise-equal results on every rank.

A tile loop right after a site that reads the destination only at its own
tile (a residual add) moves into the one-shot sum's loop when its tiles cover
the destination, another static host ``if``: one root instead of two, and
the summed tile is used without a reload.

Exchange buffers are per site.  A peer can push the next execution of a site
into a rank that is still reading the last one, unless another exchange runs
in between (every rank waits for every peer in each).  A site that is the
only one in its host loop body therefore alternates two buffer slots by the
loop index.
"""

from __future__ import annotations

import ast
import itertools
from string import Template
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from .. import exc
from .. import language as language_module
from .ast_extension import ExtendedAST
from .ast_extension import convert
from .ast_extension import expr_from_string
from .program_id import _clone_stmt
from .static_loop_unroller import _HostStaticRangeUnroller
from .static_loop_unroller import mark_static_host_if

if TYPE_CHECKING:
    from collections.abc import Callable

    from .host_function import HostFunction


# Measured on a TPU v6 slice (8 chips, ICI): an exchange round costs about
# 3.9 us beyond its data, and a push moves about 61 GB/s per peer.  One-shot
# wins while the bytes reduce-scatter + all-gather saves move in less time than
# its extra round; the product is the payload break-even in bytes.
_ROUND_LATENCY_BYTES = 238_000

_ALGORITHMS = ("one_shot", "reduce_scatter")


class _Site:
    def __init__(
        self,
        index: int,
        call: ast.Call,
        slots: int,
        slot: str,
        hl: str,
        torch_name: str,
        constant: Callable[[ast.expr], object],
    ) -> None:
        self.prefix = f"_all_reduce{index}_"
        self.torch_name = torch_name
        self.slots = slots
        self.slot = slot
        args = [*call.args]
        keywords = {kw.arg: kw.value for kw in call.keywords}
        names = ("dst", "src", "peers", "rank")
        for name in names[len(args) :]:
            if name not in keywords:
                raise exc.InvalidAPIUsage(f"hl.all_reduce: missing argument {name!r}")
            args.append(keywords.pop(name))
        algorithm = keywords.pop("algorithm", None)
        if len(args) != len(names) or keywords:
            raise exc.InvalidAPIUsage(
                "hl.all_reduce takes (dst, src, peers, rank, *, algorithm=None)"
            )
        self.dst, self.src, self.peers, self.rank = (ast.unparse(a) for a in args)
        self.peers_node = args[2]
        self.hl = hl
        value = None if algorithm is None else constant(algorithm)
        if value is None or value in _ALGORITHMS:
            self.algorithm = value
        else:
            raise exc.InvalidAPIUsage(
                f"hl.all_reduce: algorithm must be None or a constant in {_ALGORITHMS}"
            )

    @property
    def fields(self) -> dict[str, str]:
        return {
            "P": self.prefix,
            "hl": self.hl,
            "torch": self.torch_name,
            "dst": self.dst,
            "src": self.src,
            "peers": self.peers,
            "rank": self.rank,
            "slot": self.slot,
            "peer": self.peer(f"{self.prefix}j.begin"),
        }

    def peer(self, index: str) -> str:
        """The expression for peer ``index`` of the ``peers`` vector, which may
        be a full slice of a larger tensor (``peers[0, :]``)."""
        node = self.peers_node
        if isinstance(node, ast.Subscript):
            parts = (
                [*node.slice.elts]
                if isinstance(node.slice, ast.Tuple)
                else [node.slice]
            )
            last = parts[-1]
            if (
                isinstance(last, ast.Slice)
                and last.lower is None
                and last.upper is None
                and last.step is None
            ):
                lead = [ast.unparse(p) for p in parts[:-1]]
                return f"{ast.unparse(node.value)}[{', '.join([*lead, index])}]"
        return f"({self.peers})[{index}]"

    def one_shot_test(self) -> str:
        if self.algorithm is not None:
            return str(self.algorithm == "one_shot")
        return Template(_ONE_SHOT_TEST).substitute(
            self.fields, threshold=_ROUND_LATENCY_BYTES
        )

    def allocations(self) -> str:
        return Template(_ALLOCATIONS).substitute(
            self.fields, one_shot=self.one_shot_test(), slots=self.slots
        )

    def dispatch(self) -> str:
        """An ``if`` on the payload over the device loops of both algorithms."""
        fields = self.fields

        def push(src: str, index: str, dst: str) -> str:
            return Template(_PUSH).substitute(
                fields, push_src=src, push_index=index, push_dst=dst
            )

        p = self.prefix
        one_shot = Template(_ONE_SHOT).substitute(
            fields, push=push(self.src, "[]", f"{p}recv")
        )
        reduce_scatter = Template(_REDUCE_SCATTER).substitute(
            fields,
            push_scatter=push(f"{p}chunks", f"[{p}peer]", f"{p}scatter"),
            push_gather=push(f"{p}gather", f"[{self.slot}, {p}me]", f"{p}gather"),
        )
        return (
            f"if {self.one_shot_test()}:\n{textwrap.indent(one_shot, '    ')}"
            f"else:\n{textwrap.indent(reduce_scatter, '    ')}"
        )


# Templates over ``$P`` (the site's name prefix), the call's arguments and
# ``$hl`` / ``$torch``, the module names the kernel uses.

# Bytes that reduce-scatter + all-gather saves per rank, (world - 1) * numel *
# (src_bytes - (src_bytes + dst_bytes) / world), against the round it adds.
# Its chunks must be whole 128-lane tiles, which TPU DMAs move efficiently.
_ONE_SHOT_TEST = (
    "${P}world <= 2 or $src.size(1) % (128 * ${P}world) != 0 or "
    "(${P}world - 1) * $src.numel() * (${P}world * $src.element_size() "
    "- $src.element_size() - $dst.element_size()) <= $threshold * ${P}world"
)

_ALLOCATIONS = """\
${P}world = $peers.numel() + 1
${P}cols = $src.size(1) // ${P}world
if $one_shot:
    # Symmetric: [slot, sender rank, rows, cols].
    ${P}recv = $torch.empty(
        [$slots, ${P}world, $src.size(0), $src.size(1)],
        dtype=$src.dtype,
        device=$src.device,
    )
else:
    # The source split into column chunks: [chunk, rows, cols / world].
    ${P}chunks = $torch.empty(
        [${P}world, $src.size(0), ${P}cols], dtype=$src.dtype, device=$src.device
    )
    # Symmetric: [slot, sender rank, rows, cols / world].  Rank c receives
    # every rank's chunk c, then every rank receives rank c's reduced chunk c.
    ${P}scatter = $torch.empty(
        [$slots, ${P}world, $src.size(0), ${P}cols],
        dtype=$src.dtype,
        device=$src.device,
    )
    ${P}gather = $torch.empty(
        [$slots, ${P}world, $src.size(0), ${P}cols],
        dtype=$dst.dtype,
        device=$src.device,
    )
"""

# Push ``$push_src[$push_index]`` into ``$push_dst[$slot, me]`` on every
# peer, then drain all the sends and receives: the body of a device loop.
_PUSH = """\
    for ${P}j in $hl.tile(${P}world - 1, block_size=1):
        ${P}peer = $peer
        ${P}copy = $hl.make_async_remote_copy(
            $push_src, $push_index, ${P}peer, dst=$push_dst, dst_index=[$slot, ${P}me]
        )
        ${P}copy.start()
        if ${P}j.begin == ${P}world - 2:
            for ${P}w in $hl.static_range(${P}world - 1):
                ${P}copy.wait()
"""

_ONE_SHOT = """\
for ${P}_ in $hl.grid(1):
    ${P}me = $rank
    ${P}recv[$slot, ${P}me, :, :] = $src[:, :]
$push
for ${P}tm, ${P}tn in $hl.tile([$src.size(0), $src.size(1)]):
    ${P}total = $torch.sum(${P}recv[$slot, :, ${P}tm, ${P}tn], 0)
    $dst[${P}tm, ${P}tn] = ${P}total.to($dst.dtype)
"""

_REDUCE_SCATTER = """\
for ${P}_ in $hl.grid(1):
    for ${P}c in $hl.static_range(${P}world):
        ${P}chunks[${P}c, :, :] = $src[:, ${P}c * ${P}cols : (${P}c + 1) * ${P}cols]
    ${P}me = $rank
    ${P}scatter[$slot, ${P}me, :, :] = ${P}chunks[${P}me, :, :]
$push_scatter
for ${P}_ in $hl.grid(1):
    ${P}me = $rank
    ${P}total = $torch.sum(${P}scatter[$slot, :, :, :], 0)
    ${P}gather[$slot, ${P}me, :, :] = ${P}total.to($dst.dtype)
$push_gather
for ${P}_ in $hl.grid(1):
    for ${P}c in $hl.static_range(${P}world):
        $dst[:, ${P}c * ${P}cols : (${P}c + 1) * ${P}cols] = ${P}gather[$slot, ${P}c, :, :]
"""


class _Expander:
    def __init__(self, func: HostFunction) -> None:
        self.func = func
        self.resolver = _HostStaticRangeUnroller(func)
        self.counter = itertools.count()
        # pyrefly: ignore [missing-attribute]
        scope = func.fn.__globals__
        self.torch_name = next(
            (name for name, value in scope.items() if value is torch), None
        )

    def _site_call(self, stmt: ast.stmt) -> ast.Call | None:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and self.resolver._resolve(stmt.value.func) is language_module.all_reduce
        ):
            return stmt.value
        return None

    def _is_device_loop(self, stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.For)
            and isinstance(stmt.iter, ast.Call)
            and self.resolver._is_device_loop(self.resolver._resolve(stmt.iter.func))
        )

    def _direct_sites(self, body: list[ast.stmt]) -> int:
        """Sites that run on every pass through ``body``: not under a nested
        host loop, which may run zero times."""
        count = 0
        for stmt in body:
            if self._site_call(stmt) is not None:
                count += 1
            elif isinstance(stmt, ast.If):
                count += min(
                    self._direct_sites(stmt.body), self._direct_sites(stmt.orelse)
                )
        return count

    def _parity(self, loop: ast.For | None) -> tuple[int, str]:
        if loop is None or self._direct_sites(loop.body) >= 2:
            return 1, "0"
        call = loop.iter
        if (
            isinstance(loop.target, ast.Name)
            and isinstance(call, ast.Call)
            and self.resolver._resolve(call.func)
            in (range, language_module.static_range)
            and 1 <= len(call.args) <= 2
            and not call.keywords
        ):
            start = ast.unparse(call.args[0]) if len(call.args) == 2 else "0"
            return 2, f"({loop.target.id} - {start}) % 2"
        raise exc.InvalidAPIUsage(
            "hl.all_reduce: a site alone in a host loop alternates two buffer "
            "slots by the loop index, which needs `for i in range(start, stop)` "
            "or `hl.static_range(start, stop)`"
        )

    def _pointwise_consumer(self, stmt: ast.stmt, dst: ast.expr) -> ast.For | None:
        """``stmt`` if it is a 2-D tile loop that reads ``dst`` only at its own
        tile (``dst[tm, tn]``) and never writes it, so it can run in the
        one-shot sum's loop, on the sum of each tile as soon as it is made."""
        if not (
            self._is_device_loop(stmt)
            and isinstance(dst, ast.Name)
            and isinstance(stmt, ast.For)
            and self.resolver._resolve(cast("ast.Call", stmt.iter).func)
            is language_module.tile
            and isinstance(stmt.target, ast.Tuple)
            and len(stmt.target.elts) == 2
            and all(isinstance(t, ast.Name) for t in stmt.target.elts)
        ):
            return None
        call = cast("ast.Call", stmt.iter)
        if not (
            len(call.args) == 1
            and isinstance(call.args[0], (ast.List, ast.Tuple))
            and len(call.args[0].elts) == 2
        ):
            return None
        tile = [cast("ast.Name", t).id for t in stmt.target.elts]
        reads = 0
        for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
            if isinstance(node, ast.Name) and node.id in tile:
                if not isinstance(node.ctx, ast.Load):
                    return None
            elif isinstance(node, ast.Subscript) and self._is_tile_read(
                node, dst.id, tile
            ):
                reads += 1
        uses = sum(
            isinstance(node, ast.Name) and node.id == dst.id
            for node in ast.walk(ast.Module(body=stmt.body, type_ignores=[]))
        )
        return stmt if reads and reads == uses else None

    @staticmethod
    def _is_tile_read(node: ast.Subscript, dst: str, tile: list[str]) -> bool:
        index = node.slice
        return (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == dst
            and isinstance(index, ast.Tuple)
            and [getattr(e, "id", None) for e in index.elts] == tile
        )

    def body(
        self, body: list[ast.stmt], loop: ast.For | None, prelude: list[str]
    ) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        consumed: ast.stmt | None = None
        for i, stmt in enumerate(body):
            if stmt is consumed:
                continue
            call = self._site_call(stmt)
            if call is not None:
                consumer = None
                if i + 1 < len(body):
                    dst = next(
                        (kw.value for kw in call.keywords if kw.arg == "dst"),
                        call.args[0] if call.args else None,
                    )
                    if dst is not None:
                        consumer = self._pointwise_consumer(body[i + 1], dst)
                result.extend(self._expand(stmt, call, loop, prelude, consumer))
                consumed = consumer
                continue
            if self._is_device_loop(stmt):
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Expr) and self._site_call(child):
                        raise exc.InvalidAPIUsage(
                            "hl.all_reduce is a host statement; call it outside "
                            "device loops"
                        )
            elif isinstance(stmt, ast.For):
                stmt.body = self.body(stmt.body, stmt, prelude)
            elif isinstance(stmt, ast.If):
                stmt.body = self.body(stmt.body, loop, prelude)
                stmt.orelse = self.body(stmt.orelse, loop, prelude)
            result.append(stmt)
        return result

    def _expand(
        self,
        stmt: ast.stmt,
        call: ast.Call,
        loop: ast.For | None,
        prelude: list[str],
        consumer: ast.For | None,
    ) -> list[ast.stmt]:
        """The site's exchange loops.  A ``consumer`` (the statement after the
        site, see ``_pointwise_consumer``) is moved into them: into the
        one-shot sum's loop when its tiles cover the destination, and after
        the exchange otherwise."""
        if self.torch_name is None:
            raise exc.InvalidAPIUsage(
                "hl.all_reduce needs `torch` imported in the kernel's module"
            )
        func = call.func
        if not isinstance(func, ast.Attribute):
            raise exc.InvalidAPIUsage(
                "call hl.all_reduce through the helion.language module"
            )
        slots, slot = self._parity(loop)
        site = _Site(
            next(self.counter),
            call,
            slots,
            slot,
            ast.unparse(func.value),
            self.torch_name,
            self.resolver._constant,
        )
        prelude.append(site.allocations())
        assert isinstance(stmt, ExtendedAST)
        with stmt:
            statements = _statements(site.dispatch())
        if consumer is None:
            return statements
        (dispatch,) = statements
        assert isinstance(dispatch, ast.If)
        dispatch.orelse.append(_clone_stmt(consumer))
        sum_root = dispatch.body.pop()
        rows, cols = (
            ast.unparse(e)
            for e in cast("ast.List", cast("ast.Call", consumer.iter).args[0]).elts
        )
        tm, tn = (
            cast("ast.Name", t).id for t in cast("ast.Tuple", consumer.target).elts
        )
        p = site.prefix
        assert isinstance(consumer, ExtendedAST)
        with consumer:
            (covers,) = _statements(
                f"if ({rows}) == {site.src}.size(0) and ({cols}) == {site.src}.size(1):\n"
                f"    for {ast.unparse(consumer.target)} in {ast.unparse(consumer.iter)}:\n"
                f"        {p}total = {site.torch_name}.sum({p}recv[{site.slot}, :, {tm}, {tn}], 0)\n"
                f"        {p}value = {p}total.to({site.dst}.dtype)\n"
                f"        {site.dst}[{tm}, {tn}] = {p}value\n"
            )
        assert isinstance(covers, ast.If)
        (fused,) = covers.body
        assert isinstance(fused, ast.For)
        reads = _TileReads(site.dst, [tm, tn], f"{p}value")
        fused.body.extend(reads.visit(_clone_stmt(s)) for s in consumer.body)
        covers.orelse = [sum_root, _clone_stmt(consumer)]
        dispatch.body.append(covers)
        return [dispatch]


class _TileReads(ast.NodeTransformer):
    """Replaces the reads ``dst[tm, tn]`` with ``name``."""

    def __init__(self, dst: str, tile: list[str], name: str) -> None:
        self.dst = dst
        self.tile = tile
        self.name = name

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if _Expander._is_tile_read(node, self.dst, self.tile):
            assert isinstance(node, ExtendedAST)
            with node:
                return expr_from_string(self.name)
        self.generic_visit(node)
        return node


def _statements(source: str) -> list[ast.stmt]:
    """``source`` parsed, at the current location; its ``if``s are static."""
    statements = []
    for stmt in ast.parse(source).body:
        stmt = convert(stmt)
        assert isinstance(stmt, ast.stmt)
        statements.append(
            mark_static_host_if(stmt) if isinstance(stmt, ast.If) else stmt
        )
    return statements


def expand_collectives(func: HostFunction) -> None:
    """Replace each ``hl.all_reduce`` statement with its exchange loops and
    allocate their buffers."""
    if not any(
        isinstance(node, ast.Name)
        and node.id == "all_reduce"
        or isinstance(node, ast.Attribute)
        and node.attr == "all_reduce"
        for stmt in func.body
        for node in ast.walk(stmt)
    ):
        return
    prelude: list[str] = []
    body = _Expander(func).body(func.body, None, prelude)
    # Host statements may not follow a top-level loop: allocate before the
    # first, where every tensor the sites name is already defined.
    first = next(
        (i for i, stmt in enumerate(body) if isinstance(stmt, ast.For)), len(body)
    )
    anchor = body[min(first, len(body) - 1)]
    assert isinstance(anchor, ExtendedAST)
    with anchor:
        allocations = _statements("".join(prelude))
    func.body = [*body[:first], *allocations, *body[first:]]
