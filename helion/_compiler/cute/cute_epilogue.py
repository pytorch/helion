"""Whitelisted unary-chain detector for tcgen05 fused epilogues (G3.1.1).

A user-level epilogue lambda over a tcgen05 matmul result, e.g.

    out[tile_m, tile_n] = relu(acc).to(x.dtype)

is structurally an *identity-style* store the role-local tcgen05 epilogue
*could* emit if we splice the unary chain inline at the per-thread T2R
register. The G3.1.0 reachability check
(``reach_tcgen05_matmul_anchors`` in :mod:`cute_fx_walk`) only proves
*reachability* — that the value depends on a tcgen05-registered matmul
fx_node — and is the loud-failure backstop. This module owns the
*narrower* whitelist-based classifier that produces a renderable inline
expression for the simplest case: the value chain is purely

    matmul -> [whitelisted unary op]* -> convert_element_type -> store

where every op in the chain has exactly one tensor input (the prior
tensor result) and zero or more compile-time scalar arguments. Any op
outside the whitelist (including unary ops that read auxiliary tensors,
reductions, or shape changes) bails to ``None`` so the G3.1.0 diagnostic
keeps firing for those shapes — they are G3.1.2 territory.

The classifier is intentionally side-effect-free; the splice site in
``_codegen_cute_store_tcgen05_tile`` is responsible for substituting the
rendered expression in place of the existing
``tRS_rAcc.load().to(target_dtype)`` line.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch

from .cute_fx_walk import build_inner_outputs_index
from .cute_fx_walk import walk_carrier_to_tcgen05_matmul

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


# Whitelist semantics: every accepted op must (a) have exactly one tensor
# input (the chain's carrier), (b) read no global memory, (c) produce a
# tensor of the same shape and same dtype as the input. This rules out
# broadcast, reductions, and ops that read auxiliary tensors.
# Compile-time scalar arguments (e.g., ``aten.add.Tensor(x, 1.0)``) are
# folded into the rendered Python expression as numeric literals.


@dataclasses.dataclass(frozen=True)
class _UnaryStep:
    """A single accepted op rendered as ``template.format(inner=...)``.

    ``op_name`` is the human-readable op name used in ``__repr__`` for
    test diagnostics (e.g. ``"relu"``). ``template`` contains one or
    more ``{inner}`` placeholders and is the Python source the splice
    substitutes for the prior carrier expression. When
    ``inner_ref_count > 1``, the renderer hoists ``{inner}`` to a
    dedicated local first so the rendered ``template`` only references
    the carrier identifier, never the inbound expression directly.
    This keeps the template's correctness independent of whether the
    caller's ``{inner}`` is a bound local or an arbitrary expression
    — each step gets the same self-contained shape.
    """

    op_name: str
    template: str
    inner_ref_count: int = 1


# The cute DSL surface for whitelisted unary operations. Renderings are
# inline Python expressions on a TensorSSA value. All renderings
# preserve dtype (we only operate on float accumulators here), so the
# trailing ``.to(target_dtype)`` downcast in the existing splice site
# stays correct.

# `relu` must propagate NaN (`torch.relu(NaN) = NaN`) and zero out
# negative inputs including `-inf` (`torch.relu(-inf) = 0`). Naive
# `where(x > 0, x, 0)` returns 0 for NaN (`NaN > 0` is False), so we
# guard with `x != x` (the canonical NaN-detection idiom on TensorSSA;
# `cute` does not expose a generic `isnan` for it) and feed NaN through
# unchanged. The `(x + abs(x)) * 0.5` shortcut would be one expression
# but produces NaN for `-inf` (`-inf + inf = NaN`), which mismatches
# `torch.relu(-inf) = 0`; the explicit double-where is the only inline
# rendering that matches torch on the full IEEE float input range. The
# 5 inner references are bound to a single hoisted local via
# ``inner_ref_count = 5`` so the template's correctness does not
# depend on the caller passing a bound local for ``{inner}``.
_RELU_TEMPLATE = (
    "cute.where(({inner}) != ({inner}), ({inner}),"
    " cute.where(({inner}) > 0.0, ({inner}), 0.0))"
)
_RELU_INNER_REF_COUNT = 5
_ABS_TEMPLATE = "cute.math.absf({inner})"
_NEG_TEMPLATE = "(-({inner}))"
_TANH_TEMPLATE = "cute.math.tanh({inner})"
_EXP_TEMPLATE = "cute.math.exp({inner})"
_LOG_TEMPLATE = "cute.math.log({inner})"
_SQRT_TEMPLATE = "cute.math.sqrt({inner})"


def _add_const_template(scalar: float) -> str:
    return f"(({{inner}}) + {scalar!r})"


def _sub_const_template(scalar: float) -> str:
    return f"(({{inner}}) - {scalar!r})"


def _rsub_const_template(scalar: float) -> str:
    # `scalar - x`. aten.sub.Tensor with a tensor as second arg can
    # appear via `c - acc`; the lambda walks the FX user side and only
    # accepts the form where the carrier is one of the two args.
    return f"({scalar!r} - ({{inner}}))"


def _mul_const_template(scalar: float) -> str:
    return f"(({{inner}}) * {scalar!r})"


def _div_const_template(scalar: float) -> str:
    return f"(({{inner}}) / {scalar!r})"


def _rdiv_const_template(scalar: float) -> str:
    return f"({scalar!r} / ({{inner}}))"


# Mapping of accepted aten/prims targets to ``_UnaryStep`` rows. The
# classifier looks the row up at match time and emits it directly;
# binary scalar ops are handled separately below since their template
# depends on the extracted constant.
_ZERO_ARG_TARGETS: dict[object, _UnaryStep] = {
    torch.ops.aten.relu.default: _UnaryStep(
        op_name="relu",
        template=_RELU_TEMPLATE,
        inner_ref_count=_RELU_INNER_REF_COUNT,
    ),
    torch.ops.aten.abs.default: _UnaryStep(op_name="abs", template=_ABS_TEMPLATE),
    torch.ops.aten.neg.default: _UnaryStep(op_name="neg", template=_NEG_TEMPLATE),
    torch.ops.aten.tanh.default: _UnaryStep(op_name="tanh", template=_TANH_TEMPLATE),
    torch.ops.aten.exp.default: _UnaryStep(op_name="exp", template=_EXP_TEMPLATE),
    torch.ops.aten.log.default: _UnaryStep(op_name="log", template=_LOG_TEMPLATE),
    torch.ops.aten.sqrt.default: _UnaryStep(op_name="sqrt", template=_SQRT_TEMPLATE),
    # NOTE: ``prims.convert_element_type.default`` is intentionally
    # absent. A user-explicit intermediate cast (e.g.,
    # ``out[tile] = relu(acc).to(d_inter)`` written to a
    # ``d_target != d_inter`` tensor) shows up as a second
    # ``convert_element_type`` mid-chain; rejecting it by whitelist
    # falls through to the G3.1.0 backstop and prevents silently
    # dropping the intermediate cast (which would change rounding).
    # Pinned by
    # ``test_tcgen05_fused_chain_rejects_intermediate_cast_dtype_mismatch``.
}


# Scalar binary ops where one of the two FX args is the chain carrier
# and the other is a compile-time constant (``int`` / ``float``,
# possibly arriving as a 0-d tensor). The renderer folds the constant
# into the Python expression. Both arg positions are checked so
# ``acc * 2.0`` and ``2.0 * acc`` both fuse; for non-commutative ops
# (``sub``, ``div``) we render ``acc - c`` vs ``c - acc`` separately.
# These targets are also rejected if any unexpected ``kwargs`` are
# present (e.g. ``aten.add.Tensor`` accepts ``alpha=k`` which would
# silently change the rendered scalar — see the
# ``_has_only_default_kwargs`` filter at the call site).
_SCALAR_BINARY_TARGETS: frozenset[object] = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
    }
)


def _extract_scalar(arg: object) -> float | None:
    """Return ``float(arg)`` if ``arg`` is a finite Python int/float;
    else ``None``. Booleans intentionally return ``None`` (the scalar
    binary whitelist does not accept boolean arithmetic, which has
    different semantics than the float arithmetic the templates
    render). Non-finite floats (``inf``, ``-inf``, ``nan``) are also
    rejected because they ``repr`` to bare identifiers (``inf``,
    ``nan``) that are not valid Python expressions in the rendered
    cute DSL source — a whitelisted ``acc + float("inf")`` would
    splice into invalid code. Users wanting non-finite arithmetic
    can pass it through an aux-tensor lambda (G3.1-C territory)
    where the value is materialized as a tensor element.

    Helion's host-tensor guard
    (:class:`exc.HostTensorDirectUsage`) prevents 0-d tensor scalars
    from reaching the analyzer through the normal store path — a user
    writing ``acc + scalar_t`` against a 0-d host tensor is rejected
    upstream before this analyzer runs. Inside-kernel 0-d tensors are
    rare and would require explicit ``hl.zeros([])`` or similar; if a
    workload appears that materializes a fold-time-constant 0-d
    tensor mid-chain, this function can be extended to accept FX
    nodes whose ``meta['val']`` is a 0-d tensor with a constant
    backing storage. Until then, restricting to Python scalars is
    correct and removes a dead-code branch.
    """
    if isinstance(arg, bool):
        return None  # Boolean scalars are not whitelisted.
    if isinstance(arg, (int, float)):
        val = float(arg)
        if not math.isfinite(val):
            # ``repr(float("inf")) == "inf"`` is a bare identifier
            # in Python; rendering it as a literal in the splice
            # site emits invalid source. Bail to the G3.1.0
            # backstop so the user sees a structured error.
            return None
        return val
    return None


@dataclasses.dataclass(frozen=True)
class Tcgen05UnaryEpilogueChain:
    """A renderable whitelisted unary chain rooted at a tcgen05 matmul.

    ``steps`` is in *application order*: ``steps[0]`` is the op closest
    to the matmul; ``steps[-1]`` is the op closest to the
    ``convert_element_type`` cast at the store.

    :meth:`render_prelude_and_expr` emits one bound local per step so
    chain composition uses single-name references at each level. This
    avoids quadratic-or-worse source blowup from templates that
    duplicate ``{inner}`` (the relu template substitutes ``{inner}`` 5
    times, so a 3-deep chain of relus would otherwise be 125x; with
    per-step binding it stays linear). CuTe CSEs identical reads at
    compile time, but the source-side blowup pessimizes Python parse
    time and the cute-DSL JIT IR build, both of which scan the source
    text linearly. Per-step locals keep generated source size O(N) in
    the chain depth.
    """

    steps: tuple[_UnaryStep, ...]

    def render_prelude_and_expr(
        self,
        carrier_name: str,
        local_name_factory: object,
        prelude_indent: str,
    ) -> tuple[str, str]:
        """Return ``(prelude, final_expr)``.

        - ``prelude`` is one-or-more ``<local> = <step_expr>\\n`` lines
          (each indented with ``prelude_indent``).
        - ``final_expr`` is the carrier-name reference for the last
          step.

        ``local_name_factory`` must be a callable taking a single
        prefix string and returning a fresh AST variable name; in
        practice the splice site passes ``df.new_var`` so each chain
        step gets a unique name even across multiple kernels.

        Identity epilogues do not reach this method: the analyzer in
        :func:`analyze_tcgen05_unary_epilogue_chain` returns ``None``
        for the no-step case so the splice site stays out of the
        picture (the ``ast.Name`` fast path already handles identity
        stores). Empty chains here would indicate a caller bug.
        """
        assert self.steps, (
            "render_prelude_and_expr is only valid for chains with at "
            "least one step; identity epilogues should never reach the "
            "splice site (the analyzer returns None for them so the "
            "ast.Name fast path handles them)"
        )
        prelude_lines: list[str] = []
        cur_expr = carrier_name
        local = carrier_name
        for step in self.steps:
            # Hoist ``{inner}`` to a fresh local when the template
            # references it more than once (e.g. relu's double-where
            # rendering with 5 inner refs). This keeps the template
            # self-contained: a single hoisted local is referenced
            # from the rendered expression, so the rendered source
            # never duplicates the inbound expression — even if a
            # caller were to pass a non-trivial expression for
            # ``cur_expr``. Single-ref templates skip the hoist to
            # keep the no-op identity shape unchanged.
            inner_name: str
            if step.inner_ref_count > 1:
                inner_name = local_name_factory(  # type: ignore[operator]
                    "tcgen05_chain_step_in"
                )
                assert isinstance(inner_name, str)
                prelude_lines.append(f"{prelude_indent}{inner_name} = {cur_expr}\n")
            else:
                inner_name = cur_expr
            local = local_name_factory("tcgen05_chain_step")  # type: ignore[operator]
            assert isinstance(local, str)
            step_expr = step.template.format(inner=inner_name)
            prelude_lines.append(f"{prelude_indent}{local} = {step_expr}\n")
            cur_expr = local
        return ("".join(prelude_lines), local)


def _classify_scalar_binary(
    cur: torch.fx.Node,
) -> tuple[_UnaryStep, torch.fx.Node] | None:
    """Classify ``cur`` (a ``call_function`` node whose target is on the
    scalar-binary whitelist) as a single ``_UnaryStep`` plus its FX
    carrier node. Returns ``None`` if the node cannot be folded into a
    scalar-binary step (unexpected kwargs, both args are tensors, both
    args are scalars, etc.).

    Reject any non-empty kwargs: ``aten.add.Tensor`` and
    ``aten.sub.Tensor`` accept an ``alpha`` kwarg whose default is ``1``
    and that scales the second argument; silently rendering
    ``carrier + scalar`` when the FX node is ``add(carrier, scalar,
    alpha=2.0)`` would emit incorrect arithmetic. The conservative
    response is to bail to ``None`` so the G3.1.0 backstop fires.
    """
    if cur.kwargs:
        return None
    if len(cur.args) < 2:
        return None
    target = cur.target
    lhs = cur.args[0]
    rhs = cur.args[1]
    scalar: float | None
    carrier: torch.fx.Node
    forward_form: bool  # True => `carrier <op> scalar`
    if isinstance(lhs, torch.fx.Node) and not isinstance(rhs, torch.fx.Node):
        scalar = _extract_scalar(rhs)
        carrier = lhs
        forward_form = True
    elif isinstance(rhs, torch.fx.Node) and not isinstance(lhs, torch.fx.Node):
        scalar = _extract_scalar(lhs)
        carrier = rhs
        forward_form = False
    else:
        # Both args are FX nodes (auxiliary-tensor lambda — G3.1.2
        # territory) or neither is (would mean a constant binary op
        # that should have been folded upstream — shouldn't reach
        # us). Either way, bail.
        return None
    if scalar is None:
        return None
    if target is torch.ops.aten.add.Tensor:
        return _UnaryStep(op_name="add", template=_add_const_template(scalar)), carrier
    if target is torch.ops.aten.mul.Tensor:
        return _UnaryStep(op_name="mul", template=_mul_const_template(scalar)), carrier
    if target is torch.ops.aten.sub.Tensor:
        template = (
            _sub_const_template(scalar)
            if forward_form
            else _rsub_const_template(scalar)
        )
        return _UnaryStep(op_name="sub", template=template), carrier
    if target is torch.ops.aten.div.Tensor:
        template = (
            _div_const_template(scalar)
            if forward_form
            else _rdiv_const_template(scalar)
        )
        return _UnaryStep(op_name="div", template=template), carrier
    return None


def analyze_tcgen05_unary_epilogue_chain(
    state: CodegenState, value_node: torch.fx.Node
) -> tuple[Tcgen05UnaryEpilogueChain, torch.fx.Node] | None:
    """Classify ``value_node``'s producer chain as a whitelisted unary
    epilogue rooted at a tcgen05 matmul.

    Expected shape: ``value_node`` is the user-side store value, which
    Helion has wrapped in an implicit ``convert_element_type`` to the
    store-target tensor's dtype. The chain we accept is, walking
    upstream from ``value_node``:

        convert_element_type (the implicit cast) ->
        [whitelisted unary op]* ->
        accumulator carrier (phi / getitem on for_loop output ->
        registered tcgen05 matmul fx_node).

    A user-written intermediate cast like
    ``out[tile] = relu(acc).to(d_inter)`` with ``d_inter`` not equal
    to the store target's dtype shows up as a *second*
    ``convert_element_type`` inside the chain. The chain step loop
    rejects that because ``convert_element_type`` is not on the
    whitelist; the G3.1.0 backstop then fires. This means the splice
    cannot silently drop a user-explicit intermediate cast — pinned
    by ``test_tcgen05_fused_chain_rejects_intermediate_cast_dtype_mismatch``.

    Returns ``(chain, matmul_anchor)`` on success — the rendered chain
    excludes the trailing ``convert_element_type`` (the splice site
    already emits ``.to(target_dtype)`` where ``target_dtype`` is the
    store-target tensor's dtype), and the anchor is the unique
    tcgen05 matmul fx_node whose ``result_var`` the splice should
    target. Returns ``None`` if the chain is not in the whitelist or
    multiple matmul anchors are reachable along the carrier path
    (multi-input epilogues are G3.1.2 territory). The caller falls back
    to the G3.1.0 ``BackendUnsupported`` raise on ``None`` so the
    diagnostic keeps firing for non-whitelisted shapes.

    The chain may have ``steps == ()`` — i.e. ``out[tile] =
    acc.to(x.dtype)`` — in which case the splice site emits the
    existing identity ``.to(target_dtype)`` line unchanged. The fast-
    path ``ast.Name``-matching code in ``store_codegen`` handles the
    identity case earlier; the empty-chain return from this function
    is a defensive belt-and-suspenders.
    """
    df = state.device_function
    target_fx_nodes = df.cute_tcgen05_matmul_fx_nodes
    if not target_fx_nodes:
        return None

    if (
        value_node.op != "call_function"
        or value_node.target is not torch.ops.prims.convert_element_type.default
    ):
        return None
    if value_node.kwargs:
        # Defensive: ``prims.convert_element_type.default`` takes
        # ``(input, dtype)`` positionally with no kwargs in the FX
        # forms we trace; reject anything unexpected.
        return None
    cast_input = value_node.args[0] if value_node.args else None
    if not isinstance(cast_input, torch.fx.Node):
        return None

    inner_outputs_by_graph_id = build_inner_outputs_index(state)

    matmul_anchor = walk_carrier_to_tcgen05_matmul(
        cast_input, target_fx_nodes, inner_outputs_by_graph_id
    )
    if matmul_anchor is not None:
        # Identity epilogue (`out[tile] = acc.to(dtype)`). Return
        # ``None`` so the splice site stays out of the picture — the
        # ast.Name fast path in ``store_codegen`` already routes
        # identity stores through ``_codegen_cute_store_tcgen05_tile``
        # without an epilogue chain. Routing them through the splice
        # would emit a no-op prelude + identical RHS, but the
        # contract that identity stores never see an
        # ``epilogue_chain`` keeps the no-chain code path
        # byte-identical to the pre-G3.1-B golden.
        return None

    steps: list[_UnaryStep] = []
    cur: torch.fx.Node = cast_input
    # Bound the walk so a pathological FX graph cannot loop forever.
    # 32 unary ops between the matmul and the cast is an absurd upper
    # bound for any realistic activation chain.
    for _ in range(32):
        if cur.op != "call_function":
            return None
        target = cur.target
        # Zero-arg unary ops.
        if target in _ZERO_ARG_TARGETS:
            if cur.kwargs:
                return None  # Reject unexpected kwargs.
            arg = cur.args[0] if cur.args else None
            if not isinstance(arg, torch.fx.Node):
                return None
            steps.append(_ZERO_ARG_TARGETS[target])
            anchor = walk_carrier_to_tcgen05_matmul(
                arg, target_fx_nodes, inner_outputs_by_graph_id
            )
            if anchor is not None:
                steps.reverse()
                return Tcgen05UnaryEpilogueChain(steps=tuple(steps)), anchor
            cur = arg
            continue
        # Scalar binary ops (commutative + non-commutative).
        if target in _SCALAR_BINARY_TARGETS:
            classified = _classify_scalar_binary(cur)
            if classified is None:
                return None
            step, carrier = classified
            steps.append(step)
            anchor = walk_carrier_to_tcgen05_matmul(
                carrier, target_fx_nodes, inner_outputs_by_graph_id
            )
            if anchor is not None:
                steps.reverse()
                return Tcgen05UnaryEpilogueChain(steps=tuple(steps)), anchor
            cur = carrier
            continue
        # Op not on the whitelist. Surfacing None lets the G3.1.0
        # diagnostic raise so the user sees the actionable message.
        return None
    return None
