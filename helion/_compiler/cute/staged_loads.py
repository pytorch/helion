"""Fill a shared-memory image of a CTA's loads of one tensor with packets.

Role: kernel-definition pass (``cute_staged_load``), before
``cute_staged_store``.

A pointwise kernel whose stores are contiguous along one tile and whose loads
are contiguous along another (a transposing copy, ``out[z, y, x] = in[z, y,
x]`` with ``in`` of stride 1 along ``y``) has no thread layout that makes both
sides packets.  With the stores in 16-byte packets (a thread holds V
consecutive ``x``), a thread's loads are V scalars ``stride`` elements apart,
and each warp-level load instruction reads a 32-byte sector from each of up to
32 lines.  Yet the CTA as a whole reads a box of the input: R rows of L
contiguous elements, S elements apart.  With the knob on, the CTA first copies
that box into a shared-memory image with 16-byte packets, thread ``t`` taking
packets ``t``, ``t + T``, ... in the box's row-major order (a warp reads 512
contiguous bytes of image: whole 128-byte lines of the input), meets at one
barrier, and each load reads its element from the image.

Three choices make the image cheap to read:

* The index.  A load's address is a sum of ``index * stride`` terms; the
  terms whose stride is a multiple of S give the element's image row, the
  others its column, so the image index is a sum of the same indices with
  other coefficients: no division per element.
* The swizzle.  The image's 16-byte chunk ``c`` of row ``r`` sits at ``c ^ ((r
  >> s) & (C - 1))`` (C chunks a row), with the shift ``s`` (or none) under
  which the warp's reads of every load touch the fewest shared-memory
  wavefronts: a warp reading 8 rows 8 apart at one column otherwise hits one
  bank 8 times.
* The warp.  A kernel without warp-level collectives computes nothing that
  depends on which hardware thread holds which coordinates, so the pass may
  relabel the CTA's threads (a bijection of the linear thread id onto the
  coordinates ``cute.arch.thread_idx()`` returns): it picks the order of the
  thread axes under which the warp-level stores touch the fewest 128-byte
  lines, the store packets' axis fastest.  The fill then reads, and the
  stores write, whole lines per warp instruction.

Soundness.  The pass replaces the loads of one tensor ``T`` only when it
proves, for the generated body:

* Effects.  Every load of ``T`` is a statement ``name = (T.iterator +
  ...).load()``, run outside any branch the threads of a CTA may disagree
  on; ``T`` is named nowhere else and is proven not to overlap any tensor the
  kernel writes, so no store can change what the image holds.  The body has
  no early ``return`` or other irregular control flow, no barrier, shared
  memory or warp collective, and every other call is a load, a store, or one
  the relocation proofs call pure.
* Footprint.  The body is run symbolically over the thread block as
  ``cute_staged_store``'s proof runs it.  Every load's element offset must be
  the same combination of CTA-uniform symbols plus a per-thread number, and
  the loads of the launch's first CTA must read exactly the elements of a box
  ``{lo + r * S + c : r < R, c < L}`` (each at least once), ``L`` a multiple
  of the 16-byte packet, ``S`` too when ``R > 1``, and the image within the
  static shared-memory budget.  No load is masked, so every CTA reads its
  box shifted by its symbols' value: the fill reads exactly the elements the
  kernel read, in bounds wherever the kernel was.  Every CTA's box starts on
  a 16-byte boundary of ``T``, whose base is proven aligned.
* Index.  Each term of every load's address is ``index * stride`` with a
  proven stride, or a stride-1 index; the row part of the offset divides by
  S exactly, and for every thread of every load its row and column lie in the
  box.  Every integer of the proof and of the staged code stays within signed
  32 bits at every CTA, as ``cute_staged_store`` requires.

Anything else declines (dynamic shapes, masked loads, a footprint with
holes, a tensor the kernel also writes), and the body is left as it was.  Of
several loaded tensors the first that proves is staged; the others keep
their direct loads.  The staged kernel reads the same values and
computes the same results; only the instructions that read them change.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from ..ast_extension import statement_from_string
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path_str
from .emitters.lane_packets import _CUTE_VECTOR_UNROLL_CARRIER
from .literal_offset_loads import _stride_reference
from .packet_access import is_store_call
from .packet_access import match_packet_access
from .packet_access import store_pointer
from .packet_access import vector_type_source
from .staged_stores import _INDEX_RANGE
from .staged_stores import _LOOP_CALLS
from .staged_stores import _MAX_BLOCKS
from .staged_stores import _PDL_CALLS
from .staged_stores import PACKET_BYTES
from .staged_stores import _Affine
from .staged_stores import _combination_source
from .staged_stores import _Decline
from .staged_stores import _Grid
from .staged_stores import _if
from .staged_stores import _in_range
from .staged_stores import _index_axis
from .staged_stores import _Interpreter
from .staged_stores import _pointer_terms
from .staged_stores import _with_static_strides

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

# Shared-memory banks and their width, and the lines the stores are counted in.
_BANKS = 32
_BANK_BYTES = 4
_LINE_BYTES = 128
_WARP = 32


@dataclasses.dataclass(frozen=True)
class _Load:
    """One execution of a load statement by every thread: its terms (the
    index expression and stride of each, the stride 1 for a bare index) and
    the value of each term."""

    site: ast.Assign
    terms: tuple[tuple[ast.expr, int], ...]
    values: tuple[_Affine, ...]


@dataclasses.dataclass(frozen=True)
class _Store:
    """One execution of a store by every thread: its element offset in the
    stored tensor and the bytes it writes per thread."""

    offset: _Affine
    itemsize: int
    width: int


class _LoadInterpreter(_Interpreter):
    """``cute_staged_store``'s interpreter, recording the loads of ``sites``
    (their terms' values) and the stores of the other tensors."""

    def __init__(
        self,
        *,
        tensor: str,
        sites: Collection[int],
        terms: Mapping[int, tuple[tuple[ast.expr, int], ...]],
        stores: Mapping[int, tuple[str, int, int]],
        strides: Mapping[tuple[str, int], int],
        constexpr_values: Mapping[str, int],
        thread_block_dims: tuple[int, int, int],
    ) -> None:
        super().__init__(
            tensor=tensor,
            sites=sites,
            strides=strides,
            constexpr_values=constexpr_values,
            thread_block_dims=thread_block_dims,
        )
        self.site_terms = terms
        self.store_calls = stores
        self.loads: list[_Load] = []
        self.stores: list[_Store] = []

    def execute(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign) and id(statement) in self.sites:
            self.steps += 1
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            terms = self.site_terms[id(statement)]
            values = []
            for index, stride in terms:
                value = self.eval(index)
                if not isinstance(value, _Affine):
                    raise _Decline
                scaled = self.held(value.scaled(stride))
                if scaled is None:
                    raise _Decline
                values.append(scaled)
            self.loads.append(_Load(statement, terms, tuple(values)))
            self.bind(target.id, None)
            return
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            store = self.store_calls.get(id(statement.value))
            if store is not None:
                self.record_store(statement.value, store)
                return
        super().execute(statement)

    def record(self, site: ast.Expr, mask: object) -> None:
        # ``_Interpreter.branch`` records the statements of an undecided
        # ``if`` that are all sites, as masked stores: a staged load runs
        # under no mask.
        raise _Decline

    def record_store(self, call: ast.Call, store: tuple[str, int, int]) -> None:
        """A store of another tensor, for the warp layout choice only: an
        address the proof cannot follow just leaves it out of the count."""
        name, itemsize, width = store
        pointer = store_pointer(call)
        terms = None if pointer is None else _pointer_terms(pointer, name)
        if terms is None:
            return
        offset = _Affine(0)
        for term in terms:
            value = self.evaluate(term)
            if not isinstance(value, _Affine):
                return
            offset = offset.plus(value)
        self.stores.append(_Store(offset, itemsize, width))


def _root_tensor(pointer: ast.expr) -> str | None:
    root = pointer
    while isinstance(root, ast.BinOp) and isinstance(root.op, ast.Add):
        root = root.left
    name = call_path_str(root)
    if name is None or not name.endswith(".iterator"):
        return None
    return name.removesuffix(".iterator")


def _term_parts(
    term: ast.expr, strides: Mapping[tuple[str, int], int]
) -> tuple[ast.expr, int] | None:
    """``(index, stride)`` of an address term ``index * stride`` with a proven
    stride, or ``(term, 1)`` for a term without one."""
    if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
        for index, stride in ((term.left, term.right), (term.right, term.left)):
            node = stride
            if isinstance(node, ast.Call) and len(node.args) == 1:
                node = node.args[0]
            reference = _stride_reference(node)
            if reference is not None:
                if reference not in strides:
                    return None
                return index, strides[reference]
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return index, node.value
    return term, 1


def _sites(
    body: Sequence[ast.stmt],
    tensors: Mapping[str, torch.Tensor],
    strides: Mapping[tuple[str, int], int],
) -> tuple[
    dict[str, list[ast.Assign]],
    dict[int, tuple[tuple[ast.expr, int], ...]],
    dict[int, tuple[str, int, int]],
    set[str],
]:
    """Per tensor, its load statements of the staged form; their terms; the
    stores (call, tensor, itemsize, packet width); the tensors written.
    ``_Decline`` when the body's effects allow no staging at all."""
    loads: dict[str, list[ast.Assign]] = {}
    other_loads: set[str] = set()
    terms: dict[int, tuple[tuple[ast.expr, int], ...]] = {}
    site_calls: set[int] = set()
    stores: dict[int, tuple[str, int, int]] = {}
    written: set[str] = set()
    for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
        if isinstance(
            node, (ast.Return, ast.While, ast.With, ast.Try, ast.Break, ast.Continue)
        ):
            raise _Decline
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            access = match_packet_access(node.value)
            if access is not None and access.kind == "load":
                name = _root_tensor(access.pointer)
                pointer_terms = (
                    None if name is None else _pointer_terms(access.pointer, name)
                )
                if name in tensors and pointer_terms is not None:
                    parts = [_term_parts(term, strides) for term in pointer_terms]
                    if all(part is not None for part in parts):
                        loads.setdefault(name, []).append(node)
                        terms[id(node)] = tuple(
                            cast("tuple[ast.expr, int]", part) for part in parts
                        )
                        site_calls.add(id(node.value))
                        continue
        if isinstance(node, ast.Call) and id(node) not in site_calls:
            access = match_packet_access(node)
            if access is not None and access.kind != "store":
                name = _root_tensor(access.pointer)
                if name is not None:
                    other_loads.add(name)
            if is_store_call(node):
                pointer = store_pointer(node)
                name = None if pointer is None else _root_tensor(pointer)
                if name is None or name not in tensors:
                    raise _Decline
                written.add(name)
                itemsize = tensors[name].dtype.itemsize
                width = (
                    len(node.args[1].elts)
                    if (len(node.args) > 1 and isinstance(node.args[1], ast.List))
                    else 1
                )
                stores[id(node)] = (name, itemsize, width)
    for name in other_loads:
        loads.pop(name, None)
    return loads, terms, stores, written


def _effects_allow(
    body: Sequence[ast.stmt],
    tensor: str,
    sites: Collection[ast.Assign],
    stores: Mapping[int, tuple[str, int, int]],
    tensors: Collection[str],
) -> bool:
    """No call but loads, the stores of other tensors, pure ones, trace-time
    list appends and dependent-launch markers; ``tensor`` named only in its
    staged loads' addresses."""
    pointers = {
        id(node)
        for site in sites
        for node in ast.walk(cast("ast.Call", site.value).func)
    }
    calls = {id(site.value) for site in sites}
    loop_calls = {
        id(node.iter)
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and call_path_str(node.iter.func) in _LOOP_CALLS
    }
    trace_lists = {
        node.targets[0].id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.List)
        and not node.value.elts
    } - set(tensors)
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                if id(node) in calls or id(node) in loop_calls:
                    continue
                if id(node) in stores:
                    if stores[id(node)][0] == tensor:
                        return False
                    continue
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in trace_lists
                ):
                    continue
                if call_path_str(node.func) in _PDL_CALLS:
                    continue
                if not _is_proven_relocatable_call(node, allow_load=True):
                    return False
            elif isinstance(node, ast.Name) and node.id == tensor:
                if id(node) not in pointers:
                    return False
    return True


@dataclasses.dataclass(frozen=True)
class _Plan:
    """What the proof found: the load statements, the box (``lo`` less the
    symbols ``symbols``; ``rows`` x ``length`` elements, rows ``stride``
    apart), the row and column bases of the image index, the swizzle shift
    (None for none), the thread axis order and the first statement holding a
    staged load."""

    sites: tuple[ast.Assign, ...]
    symbols: tuple[tuple[str, int], ...]
    lo: int
    rows: int
    length: int
    stride: int
    row_symbols: tuple[tuple[str, int], ...]
    row_lo: int
    col_symbols: tuple[tuple[str, int], ...]
    col_lo: int
    swizzle: int | None
    # The swizzle as one value per thread, when every load of a thread
    # takes the same: ``(coordinate[axis] >> shift) & (C - 1)``.
    swizzle_coordinate: tuple[int, int] | None
    axis_order: tuple[int, ...]
    first_site: int
    # The CTA order: the block index's mixed-radix digits (radix of each, in
    # the kernel's order, fastest first) and the order to run them in.
    cta_radices: tuple[int, ...] = ()
    cta_order: tuple[int, ...] = ()


def _linear_ids(shape: tuple[int, ...], order: Sequence[int]) -> np.ndarray:
    """The hardware linear thread id of each thread coordinate (an array
    over ``[z, y, x]``) when the axes ``order`` run fastest first."""
    dims = (shape[2], shape[1], shape[0])
    index = [
        np.arange(dims[0]).reshape(1, 1, dims[0]),
        np.arange(dims[1]).reshape(1, dims[1], 1),
        np.arange(dims[2]).reshape(dims[2], 1, 1),
    ]
    linear = np.zeros(shape, dtype=np.int64)
    scale = 1
    for axis in order:
        linear = linear + index[axis] * scale
        scale *= dims[axis]
    return linear


def _store_lines(
    stores: Sequence[_Store], warps: np.ndarray, shape: tuple[int, ...]
) -> int:
    """The 128-byte lines the warp-level stores touch, summed."""
    total = 0
    for store in stores:
        offsets = np.broadcast_to(np.asarray(store.offset.const), shape).ravel()
        lines = offsets * store.itemsize // _LINE_BYTES
        last = (offsets * store.itemsize + store.itemsize * store.width - 1) // (
            _LINE_BYTES
        )
        pairs = np.unique(
            np.concatenate(
                [np.stack([warps.ravel(), lines]), np.stack([warps.ravel(), last])],
                axis=1,
            ),
            axis=1,
        )
        total += pairs.shape[1]
    return total


def _wavefronts(addresses: Sequence[np.ndarray], warps: np.ndarray) -> int:
    """The shared-memory wavefronts the warp-level loads at ``addresses``
    (byte addresses per thread, one array per load) take, summed: per warp,
    the most distinct 4-byte words any one bank serves."""
    total = 0
    flat_warps = warps.ravel()
    for address in addresses:
        words = address.ravel() // _BANK_BYTES
        for warp in np.unique(flat_warps):
            mine = np.unique(words[flat_warps == warp])
            counts = np.bincount(mine % _BANKS, minlength=_BANKS)
            total += int(counts.max())
    return total


def _thread_index_subscripted_only(body: Sequence[ast.stmt]) -> bool:
    """Whether the body reads the thread's identity only as
    ``cute.arch.thread_idx()[k]`` with a literal ``k``, which the relabeling
    rewrites: no lane or warp index, no ``thread_idx()`` unpacked whole or
    subscripted by a name."""
    subscripted: set[int] = set()
    for statement in body:
        for node in ast.walk(statement):
            # ``thread_idx()[k]`` with a literal axis: what
            # ``_ThreadRelabel`` rewrites (``thread_idx()[axis]`` is not).
            if _index_axis(node, "cute.arch.thread_idx") is not None:
                assert isinstance(node, ast.Subscript)
                subscripted.add(id(node.value))
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                name = call_path_str(node.func)
                if name in ("cute.arch.lane_idx", "cute.arch.warp_idx"):
                    return False
                if name == "cute.arch.thread_idx" and id(node) not in subscripted:
                    return False
    return True


def _prove(
    body: Sequence[ast.stmt],
    *,
    tensor: str,
    sites: Sequence[ast.Assign],
    terms: Mapping[int, tuple[tuple[ast.expr, int], ...]],
    stores: Mapping[int, tuple[str, int, int]],
    itemsize: int,
    strides: Mapping[tuple[str, int], int],
    constexpr_values: Mapping[str, int],
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int, int, int],
    smem_budget: int,
) -> _Plan:
    interpreter = _LoadInterpreter(
        tensor=tensor,
        sites={id(site) for site in sites},
        terms=terms,
        stores=stores,
        strides=strides,
        constexpr_values=constexpr_values,
        thread_block_dims=thread_block_dims,
    )
    interpreter.run(body)
    loads = interpreter.loads
    if not loads:
        raise _Decline
    shape = interpreter.shape
    offsets = []
    symbols = None
    for load in loads:
        total = _Affine(0)
        for value in load.values:
            total = total.plus(value)
        if symbols is None:
            symbols = total.symbols
        elif total.symbols != symbols:
            raise _Decline
        offsets.append(np.broadcast_to(np.asarray(total.const), shape))
    assert symbols is not None
    grid = _Grid(interpreter.symbols, constexpr_values, block_grid_dims)
    for combination, (low, high) in interpreter.spans.items():
        values = grid.combination(combination)
        if not _in_range(
            np.asarray([int(values.min()) + low, int(values.max()) + high]),
            _INDEX_RANGE,
        ):
            raise _Decline

    # The box: R rows of L contiguous elements, S apart, read exactly.
    elements = np.unique(np.concatenate([offset.ravel() for offset in offsets]))
    lo = int(elements[0])
    run = 1
    while run < elements.size and int(elements[run]) == lo + run:
        run += 1
    length = run
    if elements.size % length:
        raise _Decline
    rows = elements.size // length
    stride = int(elements[length]) - lo if rows > 1 else length
    relative = elements - lo
    if (
        stride < length
        or np.any(relative % stride >= length)
        or not np.array_equal(
            np.unique(relative // stride), np.arange(rows, dtype=relative.dtype)
        )
        or (length * itemsize) % PACKET_BYTES
        or (rows > 1 and (stride * itemsize) % PACKET_BYTES)
        or rows * length * itemsize > smem_budget
    ):
        raise _Decline
    starts = (lo + grid.combination(symbols)) * itemsize
    if np.any(starts % PACKET_BYTES):
        raise _Decline

    # Each load's row and column, from its terms: the image index.
    row_parts: list[_Affine] = []
    col_parts: list[_Affine] = []
    for load in loads:
        row = _Affine(0)
        col = _Affine(0)
        for (_, term_stride), value in zip(load.terms, load.values, strict=True):
            if rows > 1 and term_stride % stride == 0:
                # ``index * stride`` with stride a multiple of S: the row
                # part ``index * (stride / S)``, exactly.
                if np.any(np.asarray(value.const) % stride) or any(
                    c % stride for _, c in value.symbols
                ):
                    raise _Decline
                row = row.plus(
                    _Affine(
                        np.asarray(value.const) // stride,
                        tuple((name, c // stride) for name, c in value.symbols),
                    )
                )
            else:
                col = col.plus(value)
        row_parts.append(row)
        col_parts.append(col)
    row_symbols = row_parts[0].symbols
    col_symbols = col_parts[0].symbols
    if any(part.symbols != row_symbols for part in row_parts) or any(
        part.symbols != col_symbols for part in col_parts
    ):
        raise _Decline
    row_consts = [np.broadcast_to(np.asarray(part.const), shape) for part in row_parts]
    col_consts = [np.broadcast_to(np.asarray(part.const), shape) for part in col_parts]
    row_lo = min(int(const.min()) for const in row_consts)
    col_lo = min(int(const.min()) for const in col_consts)
    if row_lo * stride + col_lo != lo and rows > 1:
        raise _Decline
    image_rows: list[np.ndarray] = []
    image_cols: list[np.ndarray] = []
    for row_const, col_const, offset in zip(
        row_consts, col_consts, offsets, strict=True
    ):
        r = row_const - row_lo
        c = col_const - col_lo
        if (
            np.any(r < 0)
            or np.any(r >= rows)
            or np.any(c < 0)
            or np.any(c >= length)
            or not np.array_equal(r * stride + c, offset - lo)
        ):
            raise _Decline
        image_rows.append(r)
        image_cols.append(c)
    # The staged code computes the bases from the symbols and each row and
    # column as the sum of its terms: within signed 32 bits at every CTA.
    for combination, low in (
        (symbols, lo),
        (row_symbols, row_lo),
        (col_symbols, col_lo),
    ):
        value = low + grid.combination(combination)
        top = value + max(stride * rows, length)
        if not (_in_range(value, _INDEX_RANGE) and _in_range(top, _INDEX_RANGE)):
            raise _Decline

    # The thread axis order: the one whose warps' stores touch the fewest
    # lines (the order as generated wins ties).
    dims = thread_block_dims
    axes = [axis for axis in range(3) if dims[axis] > 1]
    orders = [tuple(range(3))]
    if _thread_index_subscripted_only(body):
        orders += [
            tuple(order) + tuple(a for a in range(3) if a not in order)
            for order in itertools.permutations(axes)
        ]
    best_order = orders[0]
    best_lines = None
    for order in orders:
        warps = _linear_ids(shape, order) // _WARP
        lines = _store_lines(interpreter.stores, warps, shape)
        if best_lines is None or lines < best_lines:
            best_order, best_lines = order, lines
    warps = _linear_ids(shape, best_order) // _WARP

    # The swizzle: the shift whose reads take the fewest wavefronts.
    width = PACKET_BYTES // itemsize
    chunks = length // width
    candidates: list[int | None] = [None]
    if chunks > 1 and chunks & (chunks - 1) == 0:
        candidates += list(range(max(1, rows).bit_length()))

    def image(r: np.ndarray, c: np.ndarray, shift: int | None) -> np.ndarray:
        chunk = c // width
        if shift is not None:
            chunk = chunk ^ ((r >> shift) & (chunks - 1))
        return (r * length + chunk * width + c % width) * itemsize

    # No swizzle where it ties with the best (no XOR per load); among shifts
    # that tie, the largest: its swizzle changes least along a
    # thread's own run of rows (a lane loop over 8 consecutive rows keeps one
    # chunk under a shift of 3, two under a shift of 2), so fewer addresses
    # are computed per load (35b8's copy: 5.4 us under the larger of two
    # tied shifts, 6.2 us under the smaller).
    fronts = {
        shift: _wavefronts(
            [image(r, c, shift) for r, c in zip(image_rows, image_cols, strict=True)],
            warps,
        )
        for shift in candidates
    }
    fewest = min(fronts.values())
    tied = [shift for shift in candidates if fronts[shift] == fewest]
    best_shift = None if None in tied else max(cast("list[int]", tied))
    swizzle_coordinate = None
    if best_shift is not None:
        # Every load of a thread in the same swizzled chunk column (35b8's
        # copy: the 8 rows of a thread's vector lane share their bits 3-5):
        # the value is one thread coordinate's bits, computed once.
        mask = chunks - 1
        values = [(r >> best_shift) & mask for r in image_rows]
        if all(np.array_equal(value, values[0]) for value in values[1:]):
            for axis in range(3):
                if thread_block_dims[axis] <= 1:
                    continue
                coordinate = np.broadcast_to(interpreter.thread_index[axis], shape)
                found = next(
                    (
                        shift
                        for shift in range(max(1, thread_block_dims[axis]).bit_length())
                        if np.array_equal((coordinate >> shift) & mask, values[0])
                    ),
                    None,
                )
                if found is not None:
                    swizzle_coordinate = (axis, found)
                    break

    cta_radices, cta_order = _cta_order(
        interpreter, grid, block_grid_dims, interpreter.stores
    )

    staged = {id(site) for site in sites}
    first_site = min(
        position
        for position, statement in enumerate(body)
        if any(id(node) in staged for node in ast.walk(statement))
    )
    used = {name for name, _ in symbols}
    used.update(name for name, _ in row_symbols)
    used.update(name for name, _ in col_symbols)
    if any(
        name in interpreter.rebound or interpreter.symbols[name].position >= first_site
        for name in used
    ):
        raise _Decline
    return _Plan(
        sites=tuple(sites),
        symbols=symbols,
        lo=lo,
        rows=rows,
        length=length,
        stride=stride,
        row_symbols=row_symbols,
        row_lo=row_lo,
        col_symbols=col_symbols,
        col_lo=col_lo,
        swizzle=best_shift,
        swizzle_coordinate=swizzle_coordinate,
        axis_order=best_order,
        first_site=first_site,
        cta_radices=cta_radices,
        cta_order=cta_order,
    )


def _cta_order(
    interpreter: _LoadInterpreter,
    grid: _Grid,
    block_grid_dims: tuple[int, int, int],
    stores: Sequence[_Store],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The block index's digits and the order to run them in: the digits
    are the CTA-uniform symbols the kernel computes from the block index
    alone, when the index is their mixed-radix number (``bid = d0 + n0 * (d1
    + n1 * d2)``, the flat grid's form); the order puts the digit along which
    the stores' addresses move least first, so consecutive CTAs write
    neighbouring memory.  ``((), ())`` when the grid is not one-dimensional,
    the digits are not found, or the order is the kernel's."""
    total = int(np.prod(block_grid_dims))
    if block_grid_dims[1] != 1 or block_grid_dims[2] != 1 or not stores:
        return (), ()
    names = [
        name
        for name, symbol in interpreter.symbols.items()
        if any(
            _index_axis(node, "cute.arch.block_idx") is not None
            for node in ast.walk(symbol.definition)
        )
        and all(not value.symbols for value in symbol.inputs.values())
    ]
    if not 1 < len(names) <= 3:
        return (), ()
    try:
        values = {name: grid.symbol(name).reshape(-1) for name in names}
    except _Decline:
        return (), ()
    bid = np.arange(total, dtype=np.int64)
    digits: list[str] | None = None
    radices: list[int] = []
    for candidate in itertools.permutations(names):
        sizes = [int(values[name].max()) + 1 for name in candidate]
        composed = np.zeros(total, dtype=np.int64)
        scale = 1
        for name, size in zip(candidate, sizes, strict=True):
            composed = composed + values[name] * scale
            scale *= size
        if scale == total and np.array_equal(composed, bid):
            digits, radices = list(candidate), sizes
            break
    if digits is None:
        return (), ()
    # The address step per digit of the first store's base (its symbols'
    # value at each CTA): the base must be linear in the digits.
    store = stores[0]
    base = grid.combination(store.offset.symbols).reshape(-1)
    steps: list[int] = []
    origin = int(base[0])
    scale = 1
    for size in radices:
        steps.append(int(base[scale]) - origin if size > 1 else 0)
        scale *= size
    predicted = np.full(total, origin, dtype=np.int64)
    for name, step in zip(digits, steps, strict=True):
        predicted = predicted + values[name] * step
    if not np.array_equal(predicted, base):
        return (), ()
    order = tuple(
        sorted(
            range(len(digits)),
            key=lambda k: (abs(steps[k]) if radices[k] > 1 else 0, k),
        )
    )
    if order == tuple(range(len(digits))):
        return (), ()
    return tuple(radices), order


class _ThreadRelabel(ast.NodeTransformer):
    """``cute.arch.thread_idx()[k]`` as coordinate ``k`` of the hardware
    linear thread id under ``order``."""

    def __init__(self, coordinates: Mapping[int, str]) -> None:
        self.coordinates = coordinates

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (
            isinstance(node.value, ast.Call)
            and call_path_str(node.value.func) == "cute.arch.thread_idx"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in self.coordinates
        ):
            return ast.Name(id=self.coordinates[node.slice.value], ctx=ast.Load())
        return self.generic_visit(node)


def stage_cta_loads(
    body: list[ast.stmt],
    *,
    tensors: Mapping[str, torch.Tensor],
    base_aligned: Callable[[torch.Tensor], bool],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    constexpr_values: Mapping[str, int],
    thread_block_dims: tuple[int, int, int] | None,
    block_grid_dims: tuple[int, int, int] | None,
    index_type: str,
    dtype_str: Callable[[torch.dtype], str],
    smem_budget: int,
    new_var: Callable[[str], str],
) -> bool:
    """Stage the loads of one tensor of the kernel ``body`` (in place) when
    the proofs of the module docstring hold; False, with ``body`` as it was,
    when none does.  The arguments are ``stage_cta_stores``'s."""
    if thread_block_dims is None or block_grid_dims is None:
        return False
    if not 0 < int(np.prod(block_grid_dims)) <= _MAX_BLOCKS:
        return False
    try:
        loads, terms, stores, written = _sites(
            body, tensors, proven_tensor_stride_values
        )
    except _Decline:
        return False
    for tensor, sites in loads.items():
        dtype = tensors[tensor].dtype
        if (
            tensor in written
            or dtype not in _CUTE_VECTOR_UNROLL_CARRIER
            or not base_aligned(tensors[tensor])
            or any(
                frozenset((tensor, other)) not in proven_disjoint_tensor_pairs
                for other in written
            )
            or not _effects_allow(body, tensor, sites, stores, tensors)
        ):
            continue
        try:
            plan = _prove(
                body,
                tensor=tensor,
                sites=sites,
                terms=terms,
                stores=stores,
                itemsize=dtype.itemsize,
                strides=proven_tensor_stride_values,
                constexpr_values=constexpr_values,
                thread_block_dims=thread_block_dims,
                block_grid_dims=block_grid_dims,
                smem_budget=smem_budget,
            )
        except _Decline:
            continue
        _emit(
            body,
            plan,
            tensor=tensor,
            dtype=dtype,
            strides=proven_tensor_stride_values,
            thread_block_dims=thread_block_dims,
            index_type=index_type,
            dtype_str=dtype_str,
            new_var=new_var,
        )
        return True
    return False


def _emit(
    body: list[ast.stmt],
    plan: _Plan,
    *,
    tensor: str,
    dtype: torch.dtype,
    strides: Mapping[tuple[str, int], int],
    thread_block_dims: tuple[int, int, int],
    index_type: str,
    dtype_str: Callable[[torch.dtype], str],
    new_var: Callable[[str], str],
) -> None:
    stage = new_var("_cta_image")
    base = new_var("_cta_image_base")
    row_base = new_var("_cta_image_row")
    col_base = new_var("_cta_image_col")
    tid = new_var("_cta_image_thread")
    width = PACKET_BYTES // dtype.itemsize
    chunks = plan.length // width
    log_width = width.bit_length() - 1
    carrier = _CUTE_VECTOR_UNROLL_CARRIER[dtype]
    vector = vector_type_source(width, carrier)

    swizzle_name = new_var("_cta_image_swizzle") if plan.swizzle_coordinate else None

    def swizzled(row: str, chunk: str, per_thread: bool = False) -> str:
        if plan.swizzle is None:
            return chunk
        if per_thread and swizzle_name is not None:
            return f"(({chunk}) ^ {swizzle_name})"
        return f"(({chunk}) ^ ((({row}) >> {plan.swizzle}) & {chunks - 1}))"

    # Each staged load reads its element from the image.
    for site in plan.sites:
        row_terms: list[str] = []
        col_terms: list[str] = []
        for index, term_stride in _site_terms(site, tensor, strides):
            source = ast.unparse(_with_static_strides(index, strides))
            if plan.rows > 1 and term_stride % plan.stride == 0:
                factor = term_stride // plan.stride
                row_terms.append(
                    f"({source}) * {factor}" if factor != 1 else f"({source})"
                )
            else:
                col_terms.append(
                    f"({source}) * {term_stride}" if term_stride != 1 else f"({source})"
                )
        row = f"(({' + '.join(row_terms) or '0'}) - {row_base})"
        col = f"(({' + '.join(col_terms) or '0'}) - {col_base})"
        index = (
            f"{row} * {plan.length} + "
            f"{swizzled(row, f'({col} >> {log_width})', per_thread=True)} * {width} + "
            f"({col} & {width - 1})"
        )
        site.value = cast(
            "ast.expr",
            cast(
                "ast.Expr", statement_from_string(f"({stage} + ({index})).load()")
            ).value,
        )

    # The fill, ahead of the first staged load, and the relabeled threads.
    dims = thread_block_dims
    threads = int(np.prod(dims))
    hardware = " + ".join(
        f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) * {scale}"
        for axis, scale in enumerate((1, dims[0], dims[0] * dims[1]))
        if dims[axis] > 1
    )
    setup = [
        (
            f"{stage} = cute.arch.alloc_smem({dtype_str(dtype)}, "
            f"{plan.rows * plan.length}, alignment={PACKET_BYTES})"
        ),
        f"{base} = {_combination_source(plan.symbols, index_type, plan.lo)}",
        f"{row_base} = {_combination_source(plan.row_symbols, index_type, plan.row_lo)}",
        f"{col_base} = {_combination_source(plan.col_symbols, index_type, plan.col_lo)}",
    ]
    # The thread id and the relabeled coordinates read nothing but the
    # thread index: they open the body, ahead of any statement that reads a
    # coordinate; the image's bases read the symbols, defined before the
    # first staged load.
    head = [f"{tid} = {hardware or 'cutlass.Int32(0)'}"]
    relabeled = plan.axis_order != tuple(range(3))
    coordinates: dict[int, str] = {}
    if relabeled:
        scale = 1
        for axis in plan.axis_order:
            if dims[axis] <= 1:
                continue
            name = new_var(f"_cta_thread_{axis}")
            coordinates[axis] = name
            head.append(f"{name} = ({tid} // {scale}) % {dims[axis]}")
            scale *= dims[axis]
    if swizzle_name is not None:
        assert plan.swizzle_coordinate is not None
        axis, shift = plan.swizzle_coordinate
        coordinate = coordinates.get(
            axis, f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"
        )
        setup.append(f"{swizzle_name} = ({coordinate} >> {shift}) & {chunks - 1}")
    statements = [statement_from_string(line) for line in setup]
    packets = plan.rows * chunks
    log_chunks = chunks.bit_length() - 1 if chunks & (chunks - 1) == 0 else None
    for first in range(0, packets, threads):
        packet = new_var("_cta_image_packet")
        row_of = (
            f"(({tid} + {first}) >> {log_chunks})"
            if log_chunks is not None
            else f"(({tid} + {first}) // {chunks})"
        )
        chunk_of = (
            f"(({tid} + {first}) & {chunks - 1})"
            if log_chunks is not None
            else f"(({tid} + {first}) % {chunks})"
        )
        moves = [
            statement_from_string(
                f"{packet} = cute.arch.load({tensor}.iterator + ({base} + "
                f"{index_type}({row_of} * {plan.stride} + {chunk_of} * {width})), "
                f"{vector})"
            ),
            statement_from_string(
                f"cute.arch.store({stage} + ({row_of} * {plan.length} + "
                f"{swizzled(row_of, chunk_of)} * {width}), {packet})"
            ),
        ]
        if first + threads > packets:
            statements.append(_if(f"({tid} + {first}) < {packets}", moves, []))
        else:
            statements.extend(moves)
    statements.append(statement_from_string("cute.arch.sync_threads()"))
    if relabeled:
        relabel = _ThreadRelabel(coordinates)
        for position in range(len(body)):
            body[position] = cast("ast.stmt", relabel.visit(body[position]))
    body[plan.first_site : plan.first_site] = statements
    body[0:0] = [statement_from_string(line) for line in head]
    if plan.cta_order:
        _reorder_ctas(body, plan, new_var)


def _site_terms(
    site: ast.Assign, tensor: str, strides: Mapping[tuple[str, int], int]
) -> list[tuple[ast.expr, int]]:
    call = cast("ast.Call", site.value)
    access = match_packet_access(call)
    assert access is not None
    pointer_terms = _pointer_terms(access.pointer, tensor)
    assert pointer_terms is not None
    return [
        cast("tuple[ast.expr, int]", _term_parts(term, strides))
        for term in pointer_terms
    ]


def transposed_load_sites(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Whether a graph loads a tensor of a dtype the image's fill moves in
    16-byte packets whose stride-1 dims a store of that graph does not share:
    the tiles on the dims of stride 1 of the load and of the store (size-1
    dims included, whose stride may be 1 or not) are not the same set, so no
    thread layout need make both sides packets and ``cute_staged_load`` is
    tunable.  A dim of size 1 changes neither set's meaning across shapes: a
    copy keeps equal sets at every shape, and a transposing kernel unequal
    ones unless every dim but one has size 1, so the search space stays the
    same for the shapes a kernel runs at.  Only static strides are read; a
    symbolic one is left alone (comparing it would add a guard)."""
    from ...language import memory_ops
    from .analysis.match_common import node_tensor
    from .flat_gather_lane import _block_id_of_size
    from .flat_gather_lane import _scalar

    def unit_blocks(subscript: object, tensor: torch.Tensor) -> frozenset[int] | None:
        if not isinstance(subscript, (list, tuple)):
            return None
        if not all(type(value) is int for value in (*tensor.stride(), *tensor.size())):
            return None
        items = [item for item in subscript if item is not None]
        if len(items) != tensor.ndim or tensor.ndim < 2:
            return None
        blocks = [_block_id_of_size(env, _scalar(item)) for item in items]
        if None in blocks or len(set(blocks)) != len(blocks):
            return None
        units = frozenset(
            cast("int", blocks[dim])
            for dim in range(tensor.ndim)
            if tensor.stride(dim) == 1
        )
        return units or None

    for graph_info in device_ir.graphs:
        loads: set[frozenset[int]] = set()
        stores: set[frozenset[int]] = set()
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (
                memory_ops.load,
                memory_ops.store,
            ):
                continue
            tensor = node_tensor(node.args[0])
            if tensor is None:
                continue
            units = unit_blocks(node.args[1], tensor)
            if units is None:
                continue
            if node.target is memory_ops.store:
                stores.add(units)
            elif tensor.dtype in _CUTE_VECTOR_UNROLL_CARRIER:
                loads.add(units)
        if any(load != store for load in loads for store in stores):
            return True
    return False


class _BlockRelabel(ast.NodeTransformer):
    """``cute.arch.block_idx()[0]`` as the name of the reordered index."""

    def __init__(self, name: str) -> None:
        self.name = name

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (
            isinstance(node.value, ast.Call)
            and call_path_str(node.value.func) == "cute.arch.block_idx"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 0
        ):
            return ast.Name(id=self.name, ctx=ast.Load())
        return self.generic_visit(node)


def _reorder_ctas(
    body: list[ast.stmt], plan: _Plan, new_var: Callable[[str], str]
) -> None:
    """Run the CTAs in ``plan.cta_order``: the hardware block index ``h`` is
    read as the mixed-radix number of the digits in that order, and the
    kernel's block index becomes the number of the same digits in its own
    order (a bijection of the grid: the CTAs share nothing)."""
    hardware = new_var("_cta_order_block")
    lines = [f"{hardware} = cutlass.Int32(cute.arch.block_idx()[0])"]
    digit_names = [
        new_var(f"_cta_order_digit_{k}") for k in range(len(plan.cta_radices))
    ]
    rest = hardware
    for position, k in enumerate(plan.cta_order):
        size = plan.cta_radices[k]
        if position == len(plan.cta_order) - 1:
            lines.append(f"{digit_names[k]} = {rest}")
        else:
            lines.append(f"{digit_names[k]} = {rest} % {size}")
            nxt = new_var("_cta_order_rest")
            lines.append(f"{nxt} = {rest} // {size}")
            rest = nxt
    composed = digit_names[-1]
    for k in range(len(plan.cta_radices) - 2, -1, -1):
        composed = f"({digit_names[k]} + {plan.cta_radices[k]} * {composed})"
    block = new_var("_cta_order_index")
    lines.append(f"{block} = {composed}")
    relabel = _BlockRelabel(block)
    for position in range(len(body)):
        body[position] = cast("ast.stmt", relabel.visit(body[position]))
    body[0:0] = [statement_from_string(line) for line in lines]
