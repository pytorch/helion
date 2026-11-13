"""Runtime tensor-storage proofs and their dispatch-cache guards."""

from __future__ import annotations

import ast
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._inductor.codegen.wrapper import pexpr
from torch._subclasses import FakeTensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .compile_environment import RuntimeInputSpecialization
from .compile_environment import _is_supported_tensor_input_source
from .compile_environment import _replay_tensor_input_source
from .compile_environment import _symint_expr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable
    from collections.abc import Sequence

    from torch._guards import Source

    from .compile_environment import CompileEnvironment
    from .compile_environment import PreparedGuardFactory


def _prepared_guard_source(
    namespace: dict[str, object],
    name: str,
    extract: Callable[[Sequence[object]], Hashable],
) -> str:
    """The expression a generated prepared guard reads a source through:
    ``args[i]`` for a plain argument (the extractor carries its index), else
    a call of the extractor bound into ``namespace``."""
    index = getattr(extract, "_helion_arg_index", None)
    if type(index) is int:
        return f"args[{index}]"
    namespace[name] = extract
    return f"{name}(args)"


def _tensor_storage_disjoint_matrix_signature(
    values: Sequence[object],
) -> Hashable:
    """Classify every pair while reading each tensor's storage only once."""
    spans: list[tuple[torch.device, int, int] | None] = []
    for value in values:
        if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
            spans.append(None)
            continue
        storage = value.untyped_storage()
        start = int(storage.data_ptr())
        spans.append((value.device, start, start + storage.nbytes()))

    result: list[bool] = []
    for left, right in itertools.combinations(spans, 2):
        if left is None or right is None:
            result.append(False)
        elif left[1] == left[2] or right[1] == right[2] or left[0] != right[0]:
            result.append(True)
        else:
            result.append(left[2] <= right[1] or right[2] <= left[1])
    return tuple(result)


def _tensor_storage_disjoint_prepared_guard(
    source_extractors: Sequence[Callable[[Sequence[object]], Hashable]],
    expected: Hashable,
) -> Callable[[Sequence[object]], bool]:
    """Prepared-call predicate for ``_tensor_storage_disjoint_matrix_signature``.

    Source types and devices are pinned by the prepared argument guard, so a
    pair's entry can only change through the storage spans themselves.  Each
    span is read once and every pair is re-evaluated against its pinned
    entry.  A pair the classifier proved disjoint through differing devices
    may be rejected here (the spans are compared regardless), which only
    costs a fallback to the full dispatch path.
    """
    namespace: dict[str, object] = {}
    lines = ["def guard(args):"]
    for index, extract in enumerate(source_extractors):
        source = _prepared_guard_source(namespace, f"extract_{index}", extract)
        lines.extend(
            (
                f"    storage_{index} = {source}.untyped_storage()",
                f"    start_{index} = storage_{index}.data_ptr()",
                f"    end_{index} = start_{index} + storage_{index}.nbytes()",
            )
        )
    conditions: list[str] = []
    for (left, right), disjoint in zip(
        itertools.combinations(range(len(source_extractors)), 2),
        cast("tuple[bool, ...]", expected),
        strict=True,
    ):
        predicate = (
            f"(start_{left} == end_{left} or start_{right} == end_{right}"
            f" or end_{left} <= start_{right} or end_{right} <= start_{left})"
        )
        conditions.append(predicate if disjoint else f"not {predicate}")
    lines.append(f"    return {' and '.join(conditions) or 'True'}")
    exec("\n".join(lines), namespace)
    return cast("Callable[[Sequence[object]], bool]", namespace["guard"])


def _tensor_alias_sources(env: CompileEnvironment) -> tuple[Source, ...]:
    sources: list[Source] = []
    # Enumerate direct host tensor arguments by their argument names rather
    # than only asking FakeTensor -> Source.  FakeTensorConverter deliberately
    # reuses one FakeTensor when the same real tensor is passed in two argument
    # slots.  That makes the reverse mapping ambiguous, but the two explicit
    # argument Sources remain distinct and must both participate in the
    # cache-specialized alias matrix.
    from .host_function import HostFunction

    for name, value in HostFunction.current().params.arguments.items():
        if isinstance(value, torch.Tensor):
            sources.append(LocalSource(name, is_input=True))
    for tensor in env.input_sources:
        source = env.tensor_input_source(tensor)
        if source is not None and source not in sources:
            sources.append(source)
    return tuple(sources)


_TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY = "tensor_storage_disjoint_matrix_v1"


def _canonical_contiguous_layout(values: Sequence[object]) -> bool:
    """Check canonical strides, including dimensions is_contiguous ignores."""
    if len(values) != 1 or not isinstance(values[0], torch.Tensor):
        return False
    tensor = values[0]
    if tensor.layout != torch.strided:
        return False
    expected: int | torch.SymInt = 1
    for size, stride in reversed(
        tuple(zip(tensor.size(), tensor.stride(), strict=True))
    ):
        if not statically_known_true(stride == expected):
            return False
        expected = expected * torch.sym_max(size, 1)
    return True


def _layout_extent_class(value: int | torch.SymInt) -> int | None:
    # Symbolic outer-tracer metadata must prove its class without new guards.
    if statically_known_true(value == 0):
        return 0
    if statically_known_true(value == 1):
        return 1
    if statically_known_true(value > 1):
        return 2
    return None


def _layout_min(*values: int | torch.SymInt) -> int | torch.SymInt:
    if all(type(value) is int for value in values):
        return min(values)
    result = values[0]
    for value in values[1:]:
        result = torch.sym_min(result, value)
    return result


def _layout_max(*values: int | torch.SymInt) -> int | torch.SymInt:
    if all(type(value) is int for value in values):
        return max(values)
    result = values[0]
    for value in values[1:]:
        result = torch.sym_max(result, value)
    return result


@dataclasses.dataclass(frozen=True)
class _CanonicalLayoutClassifier:
    """Relocatable arithmetic with separately proved dispatch eligibility.

    The type identifies the supported evaluator. Fast/prepared reuse also
    requires covered source paths and a predicate for any unpinned layout.
    """

    require_canonical_input: bool
    expressions: tuple[str, ...]
    parameter_count: int
    _evaluate: Callable[..., tuple[int | torch.SymInt, ...]] = dataclasses.field(
        init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        names = tuple(f"v{i}" for i in range(self.parameter_count))
        # pexpr also prints operators we deliberately do not replay as metadata.
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
        )
        for expression in self.expressions:
            nodes = ast.walk(ast.parse(expression, mode="eval"))
            if any(
                not isinstance(node, allowed)
                or isinstance(node, ast.Name)
                and node.id not in (*names, "min", "max")
                or isinstance(node, ast.Call)
                and (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in ("min", "max")
                )
                or isinstance(node, ast.Constant)
                and type(node.value) is not int
                for node in nodes
            ):
                raise ValueError("Unsupported canonical layout metadata expression")
        result = f"({', '.join(self.expressions)},)" if self.expressions else "()"
        evaluate = cast(
            "Callable[..., tuple[int | torch.SymInt, ...]]",
            eval(
                f"lambda {', '.join(names)}: {result}",
                {"min": _layout_min, "max": _layout_max},
            ),
        )
        object.__setattr__(self, "_evaluate", evaluate)

    def __call__(self, values: Sequence[object]) -> tuple[bool, tuple[int, ...]] | None:
        offset = int(self.require_canonical_input)
        canonical = not offset or _canonical_contiguous_layout(values[:1])
        if not canonical:
            # This binding uses runtime strides, so extent classes cannot
            # enable any part of the unused canonical-layout proof.
            return False, ()
        try:
            classes = tuple(
                _layout_extent_class(size) for size in self._evaluate(*values[offset:])
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return None if None in classes else (True, cast("tuple[int, ...]", classes))


def _canonical_metadata_guard_factory(
    env: CompileEnvironment,
    sources: tuple[Source, ...],
    require_canonical_input: bool,
) -> PreparedGuardFactory | None:
    # Fast/prepared argument metadata covers exact ordinary containers and
    # Tensor/Parameter leaves. It does not cover arbitrary attributes or
    # custom __getitem__ behavior, even when those sources are replayable.
    for source in sources:
        base = source.base if isinstance(source, TensorPropertySource) else source
        indices: list[int | str] = []
        while isinstance(base, GetItemSource):
            if base.index_is_slice or type(base.index) not in (int, str):
                return None
            indices.append(cast("int | str", base.index))
            base = base.base
        if not isinstance(base, LocalSource):
            return None
        value = env.runtime_arg_values_by_name.get(base.local_name)
        for index in reversed(indices):
            if type(value) is dict:
                value = cast("dict[int | str, object]", value)[index]
            elif type(value) in (tuple, list) and isinstance(index, int):
                value = cast("Sequence[object]", value)[index]
            else:
                return None
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            return None

    def factory(
        extractors: Sequence[Callable[[Sequence[object]], Hashable]],
        _expected: Hashable,
    ) -> Callable[[Sequence[object]], bool]:
        if require_canonical_input:
            extract = extractors[0]

            def guard(args: Sequence[object]) -> bool:
                # Python's prepared metadata fallback does not pin layout.
                # Keep this residual check even when sizes/strides are equal.
                return cast("torch.Tensor", extract(args)).layout == torch.strided

            return guard
        return lambda _args: True

    return factory


def _guard_canonical_layout(
    env: CompileEnvironment,
    tensor: torch.Tensor,
    *,
    require_input: bool,
    coalesce_degenerate: bool = False,
) -> tuple[int | None, ...] | None:
    """Key the exact derived extent classes and, optionally, input layout.

    Canonical formulas may have lost max(size, 1) under ShapeEnv's ranges.
    Guard the derived sizes themselves, not the input sizes: n-3 can cross
    one while n stays nondegenerate. Compile metadata-only arithmetic once;
    dispatch performs no symbolic substitution or tensor computation.
    A rank-one compaction proof can coalesce zero/one because both decline
    exact layout; the returned class is not used to rewrite those extents.
    """
    if not env.runtime_arg_values_by_name:
        return None
    source = None
    input_view = False
    if require_input:
        source = env.tensor_stride_input_source(tensor)
        if source is None:
            input_view = True
            owners = [
                value
                for value in env.input_sources
                if value.untyped_storage() == tensor.untyped_storage()
            ]
            if len(owners) != 1:
                return None
            source = env.tensor_stride_input_source(owners[0])
        if source is None:
            return None
    # Direct input canonical strides are checked on every call. The leading
    # size enters no stride product, so retain it symbolically without asking
    # an outer tracer to decide whether a compound size is zero/one/larger.
    preserve_leading = require_input and not input_view and tensor.ndim > 0
    expressions: list[sympy.Expr] = []
    for size in tensor.size()[int(preserve_leading) :]:
        expression = (
            _symint_expr(size)
            if isinstance(size, torch.SymInt)
            else sympy.Integer(size)
        )
        if expression is None:
            return None
        expressions.append(expression)
    # A foreign input can have a compound size such as s0 - 1. Its source
    # names that complete size; the recreated s0 deliberately has no input
    # projection. Prefer the whole metadata expression before symbol leaves.
    whole_properties: dict[sympy.Expr, list[TensorPropertySource]] = {}
    for input_tensor, input_source in env.input_sources.items():
        if not _is_supported_tensor_input_source(input_source):
            continue
        # Inferred stride expressions can equal size products only for the
        # representative layout. They are not projections of that arithmetic
        # on a later padded or empty input.
        for dim, value in enumerate(input_tensor.size()):
            if isinstance(value, torch.SymInt):
                expression = _symint_expr(value)
                if expression is not None:
                    whole_properties.setdefault(expression, []).append(
                        TensorPropertySource(input_source, TensorProperty.SIZE, dim)
                    )
    properties: dict[sympy.Expr, TensorPropertySource] = {
        expr: min(whole_properties[expr], key=repr)
        for expr in expressions
        if expr in whole_properties
    }
    symbols = set().union(
        *(expr.free_symbols for expr in expressions if expr not in properties)
    )
    for symbol in symbols:
        if not isinstance(symbol, sympy.Symbol):
            return None
        candidates = [
            item
            for item in env.shape_env.var_to_sources.get(symbol, ())
            if isinstance(item, TensorPropertySource)
            and item.prop in (TensorProperty.SIZE, TensorProperty.STRIDE)
            and _is_supported_tensor_input_source(item.base)
        ]
        if not candidates:
            return None
        properties[symbol] = min(candidates, key=repr)
    leaves = tuple(sorted(set(properties.values()), key=repr))
    names = tuple(f"v{i}" for i in range(len(leaves)))
    replacements = {
        symbol: sympy.Symbol(names[leaves.index(prop)], integer=True)
        for symbol, prop in properties.items()
    }
    code = tuple(
        pexpr(expr.xreplace(replacements), simplify=False) for expr in expressions
    )
    if coalesce_degenerate:
        assert not require_input and tensor.ndim == 1
        # Wrap after source normalization so symbolic range assumptions cannot
        # simplify away the clamp before it is replayed on actual metadata.
        code = tuple(f"max(1, {expr})" for expr in code)
    try:
        classifier = _CanonicalLayoutClassifier(source is not None, code, len(leaves))
    except ValueError:
        return None
    sources = ((source,) if source is not None else ()) + leaves

    identity = ("canonical_layout_v2", tuple(map(repr, sources)), code)
    env.register_runtime_input_specialization(
        repr(identity),
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=identity,
            classifier=classifier,
            prepared_guard_factory=_canonical_metadata_guard_factory(
                env, sources, source is not None
            ),
        ),
    )
    values: list[object] = []
    for item in sources:
        if isinstance(item, TensorPropertySource):
            value = _replay_tensor_input_source(
                item.base, env.runtime_arg_values_by_name
            )
            if not isinstance(value, torch.Tensor):
                return None
            assert item.idx is not None
            values.append(
                value.size(item.idx)
                if item.prop == TensorProperty.SIZE
                else value.stride(item.idx)
            )
        else:
            values.append(
                _replay_tensor_input_source(item, env.runtime_arg_values_by_name)
            )
    result = classifier(values)
    env.bound_metadata_specialization_results.setdefault(repr(identity), result)
    if result is None or not result[0]:
        return None
    classes = result[1]
    # A view can preserve noncanonical singleton/empty strides even when its
    # owner is canonical. Its representative formula is not a proof there.
    if input_view and any(value != 2 for value in classes):
        return None
    return (None, *classes) if preserve_leading else classes


def register_tensor_alias_specializations(env: CompileEnvironment) -> None:
    """Cache-key runtime alias facts used by memory-reordering passes.

    Different kernel argument names are not a non-aliasing proof: callers may
    pass the same tensor or overlapping views.  Recording the storage-overlap
    predicate in the bound-kernel key lets codegen use a positive runtime fact
    without reusing that code for a later aliasing launch.
    """
    sources = _tensor_alias_sources(env)
    if len(sources) < 2:
        return
    env.register_runtime_input_specialization(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                "storage_span_disjoint_matrix_v1",
                tuple(map(repr, sources)),
            ),
            classifier=_tensor_storage_disjoint_matrix_signature,
            reusable_tensor_properties=frozenset(("storage_span",)),
            prepared_guard_factory=_tensor_storage_disjoint_prepared_guard,
        ),
    )


def runtime_tensors_have_disjoint_storage(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    recordable: Sequence[Source],
) -> bool:
    """Inspect live storage without registering a cache guard.

    Reuse ``recordable`` only within one read-only candidate proof.
    A transform using this provisional fact must register the alias
    specialization before committing its rewrite.
    """
    sources: list[Source] = []
    for tensor in (left, right):
        source = env.tensor_storage_input_source(tensor)
        if source is None or source not in recordable or source in sources:
            return False
        sources.append(source)
    values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in sources
    )
    return _tensor_storage_disjoint_matrix_signature(values) == (True,)


def runtime_tensors_are_proven_disjoint(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized positive runtime storage-disjointness fact."""
    left_source = env.tensor_input_source(left)
    right_source = env.tensor_input_source(right)
    if left_source is None:
        left_source = env.tensor_storage_input_source(left)
    if right_source is None:
        right_source = env.tensor_storage_input_source(right)
    if left_source is None or right_source is None:
        return False
    return runtime_tensor_sources_are_proven_disjoint(
        env,
        left_source,
        right_source,
        allow_unbound=allow_unbound,
    )


def runtime_tensor_sources_are_proven_disjoint(
    env: CompileEnvironment,
    left_source: Source,
    right_source: Source,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized storage fact for explicit input Sources.

    Search registration runs during binding, before its immutable facts are
    recorded. It may opt into live facts from the registered classifier.
    Codegen must retain the default requirement for a bound fact.
    """
    if left_source == right_source:
        return False
    specialization = env.runtime_input_specializations.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    sources = _tensor_alias_sources(env)
    if (
        specialization is None
        or specialization.sources != sources
        or specialization.classifier_identity
        != ("storage_span_disjoint_matrix_v1", tuple(map(repr, sources)))
        or specialization.reusable_tensor_properties != frozenset(("storage_span",))
    ):
        return False
    # This immutable matrix was recorded from the bound kernel's dispatch key.
    # Recompilation may outlive its construction tensors, which are held only
    # weakly. Missing live arguments do not invalidate a cache-specialized fact.
    facts = env.bound_runtime_input_specialization_results.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    runtime_values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in sources
    )
    live_facts = specialization.classifier(runtime_values)
    assert isinstance(live_facts, tuple)
    if (
        allow_unbound
        and _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
        not in env.bound_runtime_input_specialization_results
    ):
        facts = live_facts
    if (
        not isinstance(facts, tuple)
        or len(facts) != len(sources) * (len(sources) - 1) // 2
        or any(type(fact) is not bool for fact in facts)
    ):
        return False
    live = tuple(
        isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor)
        for value in runtime_values
    )
    # Keep rejecting contradictory live arguments, including a partial weak
    # fallback where only some construction tensors are still alive.
    if any(
        left_live and right_live and live_fact != fact
        for (left_live, right_live), live_fact, fact in zip(
            itertools.combinations(live, 2), live_facts, facts, strict=True
        )
    ):
        return False
    wanted = frozenset((left_source, right_source))
    for pair, fact in zip(itertools.combinations(sources, 2), facts, strict=True):
        if frozenset(pair) == wanted:
            return fact is True
    return False
