"""Merge scalar loads at consecutive literal offsets into vector loads.

A ``hl.static_range`` over the packed bytes of a group (``w[row, k, byte]``
for ``byte`` in ``0..7``) or over the lanes of an activation group
(``x[k, lane]`` for ``lane`` in ``0..15``) lowers to one scalar load per
literal: eight ``cute.arch.load(ptr + byte * stride, Uint8)`` of one qword,
sixteen ``(ptr + lane * stride).load()`` of thirty-two contiguous bytes.  Each
is its own memory instruction, and a warp's instruction touches one byte per
thread spread over as many sectors as there are threads.  The lane vectorizer
(``affine_vector_io``) cannot help: it widens along a tile lane, and these
loads are contiguous along a literal dim the tile does not iterate.

This pass runs after scalar code generation on the kernel body.  In every
statement list it groups the unmasked scalar loads of one tensor whose pointer
expressions differ only in the literal ``c`` of one ``c * T.layout.stride[d]``
term, and replaces each run of consecutive literals by vector loads of at most
16 bytes hoisted to the run's first load, with per-element extracts in place
of the scalar loads.  A run is widened only when the dim's stride is a proven
unit stride, every other stride term is a proven multiple of the vector's
bytes, the tensor's base alignment covers the vector, and no statement between
the first and the last load of the run can write memory (the loads then move
earlier past pure arithmetic only).  Masked loads (``x if guard else 0``) and
loads inside loop headers keep their scalar form.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..compile_environment import CompileEnvironment
from .memory_ops import tensor_has_specialized_base_alignment
from .persistent_branch_vec import _plain_scalar_load_pointer
from .scalar_recipe import PURE_HELPERS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction

# Element carrier of a typed vector load, as the lane vectorizer uses: the
# load is issued as unsigned words of the element width and each extract is
# bitcast back to the scalar form's dtype.
_CARRIERS: dict[str, tuple[str, int]] = {
    "cutlass.Uint8": ("cutlass.Uint8", 1),
    "cutlass.Int8": ("cutlass.Uint8", 1),
    "cutlass.Float8E4M3FN": ("cutlass.Uint8", 1),
    "cutlass.Float8E5M2": ("cutlass.Uint8", 1),
    "cutlass.BFloat16": ("cutlass.Uint16", 2),
    "cutlass.Float16": ("cutlass.Uint16", 2),
    "cutlass.Int16": ("cutlass.Uint16", 2),
    "cutlass.Uint16": ("cutlass.Uint16", 2),
    "cutlass.Float32": ("cutlass.Uint32", 4),
    "cutlass.Int32": ("cutlass.Uint32", 4),
    "cutlass.Uint32": ("cutlass.Uint32", 4),
}
_MAX_VECTOR_BYTES = 16
_INT_CASTS = frozenset(("cutlass.Int32", "cutlass.Int64"))
# Calls a statement between two merged loads may make: scalar casts, the
# quantized decode and integer helpers, pure math, coordinates and loads.
_PURE_CALLS = (
    frozenset(
        f"cutlass.{name}"
        for name in (
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Uint8",
            "Uint16",
            "Uint32",
            "Uint64",
            "Float32",
            "Float16",
            "BFloat16",
            "Boolean",
        )
    )
    | frozenset(
        f"cute.math.{name}"
        for name in (
            "abs",
            "exp",
            "exp2",
            "log",
            "log2",
            "max",
            "min",
            "rcp",
            "rsqrt",
            "sqrt",
            "tanh",
        )
    )
    | frozenset(
        f"operator.{name}"
        for name in ("lt", "le", "gt", "ge", "eq", "ne", "and_", "or_")
    )
    | frozenset(("cute.arch.thread_idx", "cute.arch.block_idx", "cute.arch.load"))
    | PURE_HELPERS
)


@dataclass(frozen=True)
class _TensorFacts:
    dtype: str | None
    strides: dict[int, int]
    alignment: int


@dataclass
class _Site:
    call: ast.Call
    statement: int
    literal: int
    dtype: str


@dataclass(frozen=True)
class _GroupKey:
    tensor: str
    dim: int
    others: tuple[str, ...]
    element: str
    plain: bool


def _tensor_facts(device_function: DeviceFunction) -> dict[str, _TensorFacts]:
    from ..device_function import TensorArg

    env = CompileEnvironment.current()
    inputs = list(env.input_sources)
    input_storages = {id(tensor.untyped_storage()) for tensor in inputs}
    proven = device_function.proven_tensor_stride_values()
    facts: dict[str, _TensorFacts] = {}
    for arg in device_function.arguments:
        if not isinstance(arg, TensorArg):
            continue
        tensor = arg.fake_value
        strides = {
            dim: stride
            for dim, stride in enumerate(tensor.stride())
            if isinstance(stride, int)
        }
        for (name, dim), stride in proven.items():
            if name == arg.name:
                strides.setdefault(dim, stride)
        storage = id(tensor.untyped_storage())
        offset = tensor.storage_offset() * tensor.element_size()
        alignment = 1
        for candidate in (16, 8, 4, 2):
            aligned = storage not in input_storages and statically_known_true(
                offset % candidate == 0
            )
            if not aligned:
                aligned = any(
                    id(source.untyped_storage()) == storage
                    and tensor_has_specialized_base_alignment(env, source, candidate)
                    and statically_known_true(
                        (offset - source.storage_offset() * source.element_size())
                        % candidate
                        == 0
                    )
                    for source in inputs
                )
            if aligned:
                alignment = candidate
                break
        try:
            dtype = env.backend.dtype_str(tensor.dtype)
        except Exception:
            dtype = None
        if tensor.dtype in (torch.float4_e2m1fn_x2, torch.uint8):
            dtype = "cutlass.Uint8"
        facts[arg.name] = _TensorFacts(dtype, strides, alignment)
    return facts


def _add_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_add_terms(node.left), *_add_terms(node.right)]
    return [node]


def _strip_int_cast(node: ast.expr) -> ast.expr:
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in _INT_CASTS
    ):
        node = node.args[0]
    return node


def _stride_reference(node: ast.expr) -> tuple[str, int] | None:
    node = _strip_int_cast(node)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "stride"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "layout"
        and isinstance(node.value.value.value, ast.Name)
    ):
        return node.value.value.value.id, node.slice.value
    return None


def _stride_term(node: ast.expr) -> tuple[ast.expr, str, int] | None:
    """``coefficient * T.layout.stride[d]`` (either order) as ``(coefficient, T, d)``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    for coefficient, stride in ((node.left, node.right), (node.right, node.left)):
        reference = _stride_reference(stride)
        if reference is not None:
            return coefficient, *reference
    return None


def _literal(node: ast.expr) -> int | None:
    node = _strip_int_cast(node)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


@dataclass(frozen=True)
class _Pointer:
    tensor: str
    dim: int
    literal: int
    # ``ast.dump`` of every other term, sorted: the group identity.
    others: tuple[str, ...]
    # Dims whose stride multiplies a non-literal coefficient.
    stride_dims: tuple[int, ...]


def _pointer(node: ast.expr) -> _Pointer | None:
    terms = _add_terms(node)
    roots = [
        term.value.id
        for term in terms
        if isinstance(term, ast.Attribute)
        and term.attr == "iterator"
        and isinstance(term.value, ast.Name)
    ]
    if len(roots) != 1:
        return None
    tensor = roots[0]
    literal: tuple[int, int] | None = None
    others: list[str] = []
    stride_dims: list[int] = []
    for term in terms:
        if isinstance(term, ast.Attribute) and term.attr == "iterator":
            others.append(ast.dump(term))
            continue
        stride = _stride_term(term)
        if stride is None or stride[1] != tensor:
            return None
        coefficient, _, dim = stride
        value = _literal(coefficient)
        if value is not None and literal is None:
            literal = (dim, value)
            continue
        if value is not None:
            # Two literal terms: keep the first as the merge axis only when
            # the second is the same for every load, which the group identity
            # (``others``) enforces.
            others.append(ast.dump(term))
            stride_dims.append(dim)
            continue
        others.append(ast.dump(term))
        stride_dims.append(dim)
    if literal is None:
        return None
    return _Pointer(
        tensor, literal[0], literal[1], tuple(sorted(others)), tuple(stride_dims)
    )


def _load_call(node: ast.Call) -> tuple[ast.expr, str | None] | None:
    """``(pointer, element dtype)`` of a scalar load; dtype ``None`` for ``.load()``."""
    name = ast.unparse(node.func)
    if name == "cute.arch.load":
        if len(node.args) != 2 or node.keywords:
            return None
        dtype = node.args[1]
        if not isinstance(dtype, ast.Attribute):
            return None
        return node.args[0], ast.unparse(dtype)
    pointer = _plain_scalar_load_pointer(node)
    if pointer is not None:
        return pointer, None
    return None


class _Collector(ast.NodeVisitor):
    """Unmasked scalar loads of one statement, outside nested control flow."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []
        self.pure = True

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # A guarded load keeps its scalar form; its guard may still be pure.
        self.visit(node.test)
        self._check_pure(node.body)
        self._check_pure(node.orelse)

    def _check_pure(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if (
                    _load_call(child) is None
                    and ast.unparse(child.func) not in _PURE_CALLS
                ):
                    self.pure = False
            elif isinstance(child, (ast.Subscript, ast.Attribute)) and isinstance(
                child.ctx, (ast.Store, ast.Del)
            ):
                self.pure = False

    def visit_Call(self, node: ast.Call) -> None:
        load = _load_call(node)
        if load is not None:
            self.calls.append(node)
            for arg in node.args:
                self.visit(arg)
            return
        if ast.unparse(node.func) not in _PURE_CALLS:
            self.pure = False
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.pure = False
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.pure = False
        self.generic_visit(node)


def _statement_loads(
    statement: ast.stmt,
) -> tuple[list[ast.Call], bool, frozenset[str]]:
    """Loads of a statement, whether it is free of memory effects, and the
    names it binds."""
    if isinstance(statement, ast.Assign):
        if not all(isinstance(target, ast.Name) for target in statement.targets):
            return [], False, frozenset()
        collector = _Collector()
        collector.visit(statement.value)
        writes = frozenset(
            target.id for target in statement.targets if isinstance(target, ast.Name)
        )
        return collector.calls, collector.pure, writes
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        if not isinstance(statement.target, ast.Name):
            return [], False, frozenset()
        collector = _Collector()
        collector.visit(statement.value)
        return collector.calls, collector.pure, frozenset({statement.target.id})
    if isinstance(statement, ast.Pass):
        return [], True, frozenset()
    return [], False, frozenset()


def _names(node: ast.AST) -> frozenset[str]:
    return frozenset(
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    )


class _Merger:
    def __init__(self, facts: dict[str, _TensorFacts]) -> None:
        self.facts = facts
        self.counter = 0

    def fresh(self) -> str:
        self.counter += 1
        return f"_helion_literal_vec_{self.counter}"

    def block(self, body: list[ast.stmt]) -> list[ast.stmt]:
        for statement in body:
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if (
                    isinstance(nested, list)
                    and nested
                    and isinstance(nested[0], ast.stmt)
                ):
                    setattr(statement, field, self.block(nested))
        return self.merge(body)

    def merge(self, body: list[ast.stmt]) -> list[ast.stmt]:
        pure: list[bool] = []
        writes: list[frozenset[str]] = []
        groups: dict[_GroupKey, list[_Site]] = defaultdict(list)
        for index, statement in enumerate(body):
            calls, is_pure, bound = _statement_loads(statement)
            pure.append(is_pure)
            writes.append(bound)
            for call in calls:
                load = _load_call(call)
                assert load is not None
                pointer, dtype = load
                parsed = _pointer(pointer)
                if parsed is None:
                    continue
                facts = self.facts.get(parsed.tensor)
                if facts is None:
                    continue
                element = dtype if dtype is not None else facts.dtype
                if element is None or element not in _CARRIERS:
                    continue
                key = _GroupKey(
                    parsed.tensor, parsed.dim, parsed.others, element, dtype is None
                )
                groups[key].append(_Site(call, index, parsed.literal, element))
        inserts: dict[int, list[ast.stmt]] = defaultdict(list)
        replacements: dict[int, ast.expr] = {}
        for key, sites in groups.items():
            facts = self.facts[key.tensor]
            carrier, itemsize = _CARRIERS[key.element]
            if facts.strides.get(key.dim) != 1:
                continue
            seen: dict[int, _Site] = {}
            for site in sites:
                # The same literal twice in one block is a duplicate load the
                # scalar generator kept; merging only the first is still exact.
                seen.setdefault(site.literal, site)
            ordered = sorted(seen.values(), key=lambda site: site.literal)
            run: list[_Site] = []
            for site in ordered:
                if run and site.literal != run[-1].literal + 1:
                    self.emit(
                        run,
                        facts,
                        carrier,
                        itemsize,
                        pure,
                        writes,
                        inserts,
                        replacements,
                    )
                    run = []
                run.append(site)
            if run:
                self.emit(
                    run, facts, carrier, itemsize, pure, writes, inserts, replacements
                )
        if not replacements:
            return body

        class Replace(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                replacement = replacements.get(id(node))
                if replacement is not None:
                    return replacement
                return self.generic_visit(node)

        result: list[ast.stmt] = []
        for index, statement in enumerate(body):
            result.extend(inserts.get(index, ()))
            result.append(Replace().visit(statement))
        return result

    def emit(
        self,
        run: Sequence[_Site],
        facts: _TensorFacts,
        carrier: str,
        itemsize: int,
        pure: Sequence[bool],
        writes: Sequence[frozenset[str]],
        inserts: dict[int, list[ast.stmt]],
        replacements: dict[int, ast.expr],
    ) -> None:
        width = 1
        while width * 2 <= len(run) and width * 2 * itemsize <= _MAX_VECTOR_BYTES:
            width *= 2
        start = 0
        while start + width <= len(run):
            piece = run[start : start + width]
            if width > 1 and self.admits(piece, width, itemsize, facts, pure, writes):
                self.widen(piece, width, carrier, itemsize, inserts, replacements)
            start += width

    def admits(
        self,
        piece: Sequence[_Site],
        width: int,
        itemsize: int,
        facts: _TensorFacts,
        pure: Sequence[bool],
        writes: Sequence[frozenset[str]],
    ) -> bool:
        vector_bytes = width * itemsize
        if facts.alignment % vector_bytes:
            return False
        if (piece[0].literal * itemsize) % vector_bytes:
            return False
        load = _load_call(piece[0].call)
        assert load is not None
        pointer = _pointer(load[0])
        assert pointer is not None
        for dim in pointer.stride_dims:
            stride = facts.strides.get(dim)
            if stride is None or (stride * itemsize) % vector_bytes:
                return False
        first = min(site.statement for site in piece)
        last = max(site.statement for site in piece)
        if not all(pure[first : last + 1]):
            return False
        # The hoisted load reads the first load's pointer; a name it reads
        # must not be rebound before the last load (the same text would
        # then name a different address).
        reads = _names(load[0])
        return not any(reads & writes[index] for index in range(first, last + 1))

    def widen(
        self,
        piece: Sequence[_Site],
        width: int,
        carrier: str,
        itemsize: int,
        inserts: dict[int, list[ast.stmt]],
        replacements: dict[int, ast.expr],
    ) -> None:
        first = min(piece, key=lambda site: site.statement)
        load = _load_call(first.call)
        assert load is not None
        base_pointer, _dtype = load
        parsed = _pointer(base_pointer)
        assert parsed is not None
        base = ast.parse(ast.unparse(base_pointer), mode="eval").body
        # Rewrite the merge dim's literal term to the piece's first literal.
        target = piece[0].literal
        retargeted = False

        class Retarget(ast.NodeTransformer):
            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                nonlocal retargeted
                stride = _stride_term(node)
                if (
                    not retargeted
                    and stride is not None
                    and stride[2] == parsed.dim
                    and _literal(stride[0]) == first.literal
                ):
                    constant = _strip_int_cast(stride[0])
                    assert isinstance(constant, ast.Constant)
                    constant.value = target
                    retargeted = True
                    return node
                return self.generic_visit(node)

        base = Retarget().visit(base)
        assert retargeted
        vector = self.fresh()
        inserts[first.statement].append(
            ast.parse(
                f"{vector} = cute.arch.load({ast.unparse(base)}, "
                f"ir.VectorType.get([{width}], {carrier}.mlir_type))"
            ).body[0]
        )
        for site in piece:
            lane = site.literal - target
            extract = f"{carrier}({vector}[{lane}])"
            if site.dtype != carrier:
                extract = f"{extract}.bitcast({site.dtype})"
            replacements[id(site.call)] = ast.parse(extract, mode="eval").body


def merge_literal_offset_loads(
    body: list[ast.stmt], device_function: DeviceFunction
) -> list[ast.stmt]:
    merger = _Merger(_tensor_facts(device_function))
    return ast.fix_missing_locations(
        ast.Module(body=merger.block(body), type_ignores=[])
    ).body
