"""Whitelisted chain detector for tcgen05 fused epilogues.

A user-level epilogue lambda over a tcgen05 matmul result, e.g.

    out[tile_m, tile_n] = relu(acc).to(x.dtype)
    out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(x.dtype)

is structurally an *identity-style* store the role-local tcgen05 epilogue
*could* emit if we splice the chain inline at the per-thread T2R
register. The reachability check
(``reach_tcgen05_matmul_anchors`` in :mod:`cute_fx_walk`) only proves
*reachability* — that the value depends on a tcgen05-registered matmul
fx_node — and is the loud-failure backstop. This module owns the
*narrower* whitelist-based classifier that produces a renderable inline
expression for two cases:

- Unary chains: ``matmul -> [whitelisted unary op]* ->
  convert_element_type -> store``, where every op in the chain has
  exactly one tensor input (the prior tensor result) and zero or more
  compile-time scalar arguments.
- Auxiliary-tensor binary ops: same shape as above, but one or more
  steps are ``add/sub/mul/div`` with the chain carrier as one
  operand and a ``helion.language.load(aux_tensor, [...])``
  call as the other. Two aux load shapes are accepted: the
  exact-shape rank-2 form (``residual[tile_m, tile_n]``) and the
  rank-1 trailing-axis (rowvec) broadcast form (``bias[tile_n]``).
  See :class:`_AuxiliaryTensorStep` for the canonical contract.
  Forms outside these two — 3-D underlying tensors with a static
  collapse, mismatched indices, leading-axis rank-1
  (``bias[tile_m]``), kwargs — are rejected to the loud-failure
  backstop.

Any op outside the whitelist (reductions, shape changes,
auxiliary-tensor loads with unsupported indexing, etc.) bails to
``None`` so the loud-failure diagnostic keeps firing for those shapes.

The classifier is intentionally side-effect-free; the splice site in
``_codegen_cute_store_tcgen05_tile`` is responsible for substituting the
rendered expression in place of the existing
``tRS_rAcc.load().to(target_dtype)`` line. Auxiliary-tensor steps carry
the FX node of their ``load`` call so the splice site can recover the
auxiliary tensor + index expressions and emit the per-thread GMEM read
inline.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch

from .cute_fx_walk import aux_tensor_load_kind
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


@dataclasses.dataclass(frozen=True)
class _AuxiliaryTensorStep:
    """A binary scalar op fused as ``carrier <op> aux_load``.

    Unlike ``_UnaryStep`` (whose other operand is a compile-time
    scalar literal), this step's other operand is the result of a
    ``helion.language.load`` call against an auxiliary GMEM tensor.
    The splice site is responsible for emitting per-thread aux load
    code from the ``load_node`` data captured here — including the
    auxiliary tensor name, the index expressions, and the dtype.

    ``op_name`` is the user-visible op name (``"add"`` / ``"sub"`` /
    ``"mul"`` / ``"div"``) used in test diagnostics. ``op_template``
    is the binary-op Python expression the splice renders, with
    placeholders ``{carrier}`` (the chain carrier identifier) and
    ``{aux}`` (the per-thread auxiliary load expression). Renderers
    do **not** parenthesize ``{aux}`` because the splice site emits
    ``aux`` as a bound local first — there is no precedence
    ambiguity inside the rendered binary expression.

    ``forward_form`` is ``True`` when the carrier is the *left*
    operand (``carrier <op> aux``) and ``False`` when it is the
    right (``aux <op> carrier``). Commutative ops (``add`` / ``mul``)
    only need one form; non-commutative ones (``sub`` / ``div``)
    use the flag to render the correct direction.

    ``load_node`` is the FX node for the ``helion.language.load``
    call. The splice site reads ``load_node.args[0]`` (the auxiliary
    tensor's host tensor FX node) and ``load_node.args[1]`` (the
    index list, which the analyzer pins to exactly the carrier's
    tile-id symbol nodes — broader index shapes are rejected at
    classify time).

    ``broadcast_axis`` is ``None`` when the aux tensor matches the
    carrier rank exactly (``residual[tile_m, tile_n]``) or ``1`` for
    a trailing-axis (rowvec) broadcast aux load (``bias[tile_n]``
    with shape ``(N,)``). The leading-axis (colvec / M-axis) form is
    not accepted: a bare rank-1 operand on the RHS aligns to the
    *last* dimension under PyTorch broadcasting rules
    (``acc + bias[tile_m]`` is either a shape error when BM ≠ BN
    or a rowvec broadcast when BM == BN), so accepting the colvec
    pattern would silently rewrite the user's broadcast direction.
    Users wanting an explicit colvec broadcast must spell it out
    with ``[:, None]`` / ``.unsqueeze(-1)``; that is a separate
    pattern handler not yet wired up. The splice site
    (``memory_ops._codegen_cute_store_tcgen05_tile``) owns the
    canonical broadcast-view contract — it builds a 2-D logical
    view of the rank-1 tensor with stride 0 on the orthogonal axis
    so the existing ``partition_C → flat_divide → partition_D``
    pipeline can run unchanged. Mirrors Quack's ``RowVecLoad``
    epilogue (``quack/quack/epi_ops.py``).
    """

    op_name: str
    op_template: str
    forward_form: bool
    load_node: torch.fx.Node
    broadcast_axis: int | None = None


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
    # falls through to the loud-failure backstop and prevents
    # silently dropping the intermediate cast (which would change
    # rounding). Pinned by
    # ``test_tcgen05_fused_chain_rejects_intermediate_cast_dtype_mismatch``.
}


# Binary ops that the chain analyzer accepts. Both scalar (the
# other operand is a compile-time literal int/float) and
# auxiliary-tensor (the other operand is a
# ``helion.language.load`` of a 2-D auxiliary GMEM tensor matching
# the output tile shape) forms are routed through ``_classify_binary``
# below. Both arg positions are checked so ``acc <op> other`` and
# ``other <op> acc`` both fuse; for non-commutative ops
# (``sub``, ``div``) the renderer picks the correct direction. These
# targets are also rejected if any unexpected ``kwargs`` are present
# (e.g. ``aten.add.Tensor`` accepts ``alpha=k`` which would silently
# change the rendered expression — see the kwarg-rejection branch in
# ``_classify_binary``).
_SCALAR_BINARY_TARGETS: frozenset[object] = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
    }
)


# Per-target ``op_template`` for the auxiliary-tensor renderer. The
# placeholders ``{carrier}`` and ``{aux}`` are bound to splice-site
# locals (the chain carrier and the per-thread aux load
# respectively). The classifier picks ``_AUX_FORWARD_OP_TEMPLATES``
# when the carrier is the *left* operand (``carrier <op> aux``) and
# ``_AUX_REVERSE_OP_TEMPLATES`` when it is the right (``aux <op>
# carrier``). For commutative ``add`` / ``mul`` the two tables are
# value-equivalent; the symmetric handling keeps non-commutative
# ``sub`` / ``div`` correct without a per-step branch.
_AUX_FORWARD_OP_TEMPLATES: dict[object, str] = {
    torch.ops.aten.add.Tensor: "(({carrier}) + ({aux}))",
    torch.ops.aten.mul.Tensor: "(({carrier}) * ({aux}))",
    torch.ops.aten.sub.Tensor: "(({carrier}) - ({aux}))",
    torch.ops.aten.div.Tensor: "(({carrier}) / ({aux}))",
}
_AUX_REVERSE_OP_TEMPLATES: dict[object, str] = {
    torch.ops.aten.add.Tensor: "(({aux}) + ({carrier}))",
    torch.ops.aten.mul.Tensor: "(({aux}) * ({carrier}))",
    torch.ops.aten.sub.Tensor: "(({aux}) - ({carrier}))",
    torch.ops.aten.div.Tensor: "(({aux}) / ({carrier}))",
}


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
    can pass it through an aux-tensor lambda where the value is
    materialized as a tensor element.

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
            # site emits invalid source. Bail to the loud-failure
            # backstop so the user sees a structured error.
            return None
        return val
    return None


@dataclasses.dataclass(frozen=True)
class Tcgen05UnaryEpilogueChain:
    """A renderable whitelisted chain rooted at a tcgen05 matmul.

    ``steps`` is in *application order*: ``steps[0]`` is the op closest
    to the matmul; ``steps[-1]`` is the op closest to the
    ``convert_element_type`` cast at the store. A step is either a
    ``_UnaryStep`` (zero-arg unary or scalar binary) or an
    ``_AuxiliaryTensorStep`` (binary op with the chain carrier +
    a ``helion.language.load`` of an auxiliary GMEM tensor). The
    classname (``Tcgen05UnaryEpilogueChain``) is preserved from the
    earlier unary-only implementation for byte-identity in goldens;
    conceptually the type is now a "tcgen05 epilogue chain" that
    may include auxiliary-tensor steps.

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

    For auxiliary-tensor steps, the renderer expects the splice site
    to provide a per-step pre-bound local for the ``aux`` operand;
    the renderer just emits ``carrier <op> aux_local``. The aux load
    code (cute partition + per-thread read) is the splice site's
    responsibility because it depends on the per-subtile loop layout
    and the partitioned tile, neither of which this module knows about.
    """

    steps: tuple[_UnaryStep | _AuxiliaryTensorStep, ...]

    @property
    def auxiliary_tensor_steps(self) -> tuple[_AuxiliaryTensorStep, ...]:
        """All auxiliary-tensor steps in application order.

        Used by the splice site to request per-step ``aux_local``
        locals before calling :meth:`render_prelude_and_expr` so the
        per-thread aux load setup runs once per output tile (outside
        the per-subtile chain rendering).
        """
        return tuple(s for s in self.steps if isinstance(s, _AuxiliaryTensorStep))

    def render_prelude_and_expr(
        self,
        carrier_name: str,
        local_name_factory: object,
        prelude_indent: str,
        aux_locals_by_step: tuple[str, ...] | None = None,
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

        ``aux_locals_by_step`` is required when any step is an
        ``_AuxiliaryTensorStep`` and supplies, per-aux-step in
        application order, the splice-side pre-bound local that
        carries the per-thread auxiliary load value. Pure unary
        chains pass ``None``.

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
        aux_steps = self.auxiliary_tensor_steps
        if aux_steps:
            assert aux_locals_by_step is not None and len(aux_locals_by_step) == len(
                aux_steps
            ), (
                "auxiliary-tensor chain steps require one aux local per "
                f"aux step; got {len(aux_locals_by_step) if aux_locals_by_step is not None else None} "
                f"aux_locals for {len(aux_steps)} aux steps"
            )
        else:
            assert aux_locals_by_step is None or aux_locals_by_step == (), (
                "non-auxiliary chains must not be passed aux_locals_by_step"
            )
        aux_local_iter = iter(aux_locals_by_step or ())
        prelude_lines: list[str] = []
        cur_expr = carrier_name
        local = carrier_name
        for step in self.steps:
            if isinstance(step, _AuxiliaryTensorStep):
                # Auxiliary-tensor binary op. The splice site has
                # already bound the per-thread aux load to a local
                # (so the renderer's job is just substituting the
                # binary expression). The carrier is bound here as
                # well, even though templates only reference it once
                # — the symmetric handling with the unary path keeps
                # the prelude shape uniform and CuTe CSEs the load
                # reference at compile time.
                aux_local = next(aux_local_iter)
                local = local_name_factory(  # type: ignore[operator]
                    "tcgen05_chain_step"
                )
                assert isinstance(local, str)
                step_expr = step.op_template.format(carrier=cur_expr, aux=aux_local)
                prelude_lines.append(f"{prelude_indent}{local} = {step_expr}\n")
                cur_expr = local
                continue
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


def _classify_binary(
    cur: torch.fx.Node,
    *,
    carrier_tile_shape: tuple[object, ...] | None,
    carrier_tile_index_nodes: tuple[torch.fx.Node, ...] | None = None,
    carrier_global_shape: tuple[object, ...] | None = None,
) -> tuple[_UnaryStep | _AuxiliaryTensorStep, torch.fx.Node] | None:
    """Classify ``cur`` (a ``call_function`` node whose target is on the
    binary whitelist) as a single chain step plus its FX carrier node.

    Returns ``None`` if the node cannot be folded — unexpected
    kwargs, multiple chain inputs, both args are scalars, both args
    are tensors but neither is a recognized auxiliary load, etc. The
    ``_AuxiliaryTensorStep`` branch is gated on ``carrier_tile_shape``
    being available *and* matching the auxiliary load's tile shape;
    pass ``None`` when no carrier tile shape is known (e.g. the
    chain entry point) and the auxiliary branch will be skipped.

    Reject any non-empty kwargs: ``aten.add.Tensor`` and
    ``aten.sub.Tensor`` accept an ``alpha`` kwarg whose default is ``1``
    and that scales the second argument; silently rendering
    ``carrier + other`` when the FX node is ``add(carrier, other,
    alpha=2.0)`` would emit incorrect arithmetic. The conservative
    response is to bail to ``None`` so the loud-failure backstop
    fires. Pinned for both scalar and auxiliary-tensor forms by
    ``test_tcgen05_fused_chain_rejects_alpha_kwarg``.
    """
    if cur.kwargs:
        return None
    if len(cur.args) < 2:
        return None
    target = cur.target
    lhs = cur.args[0]
    rhs = cur.args[1]
    # Determine which arg is the chain carrier. The carrier is the
    # FX node that walks back to the matmul anchor; the other is
    # either a Python scalar literal or a recognized aux load FX
    # node. We branch on the arg shapes here and let the caller's
    # subsequent ``walk_carrier_to_tcgen05_matmul`` confirm the
    # chosen carrier reaches the matmul.
    lhs_is_node = isinstance(lhs, torch.fx.Node)
    rhs_is_node = isinstance(rhs, torch.fx.Node)
    if lhs_is_node and rhs_is_node:
        assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
        # Both args are FX nodes. One must be a recognized auxiliary
        # tensor load; the other is the chain carrier. If both look
        # like aux loads or neither does, bail — the chain has no
        # unique carrier.
        lhs_kind = aux_tensor_load_kind(
            lhs,
            carrier_tile_shape=carrier_tile_shape,
            carrier_tile_index_nodes=carrier_tile_index_nodes,
            carrier_global_shape=carrier_global_shape,
        )
        rhs_kind = aux_tensor_load_kind(
            rhs,
            carrier_tile_shape=carrier_tile_shape,
            carrier_tile_index_nodes=carrier_tile_index_nodes,
            carrier_global_shape=carrier_global_shape,
        )
        aux_load: torch.fx.Node
        carrier: torch.fx.Node
        forward_form: bool
        if lhs_kind is not None and rhs_kind is None:
            aux_load = lhs
            carrier = rhs
            forward_form = False  # carrier is the right operand
            aux_kind = lhs_kind
        elif rhs_kind is not None and lhs_kind is None:
            aux_load = rhs
            carrier = lhs
            forward_form = True  # carrier is the left operand
            aux_kind = rhs_kind
        else:
            return None
        op_template_table: dict[object, str] = (
            _AUX_FORWARD_OP_TEMPLATES if forward_form else _AUX_REVERSE_OP_TEMPLATES
        )
        op_template = op_template_table[target]
        op_name_table: dict[object, str] = {
            torch.ops.aten.add.Tensor: "add",
            torch.ops.aten.mul.Tensor: "mul",
            torch.ops.aten.sub.Tensor: "sub",
            torch.ops.aten.div.Tensor: "div",
        }
        op_name = op_name_table[target]
        broadcast_axis = aux_kind[1] if aux_kind[0] == "broadcast" else None
        return (
            _AuxiliaryTensorStep(
                op_name=op_name,
                op_template=op_template,
                forward_form=forward_form,
                load_node=aux_load,
                broadcast_axis=broadcast_axis,
            ),
            carrier,
        )
    # One arg is a tensor and the other a scalar literal. Extract
    # the scalar and render a ``_UnaryStep`` row.
    scalar: float | None
    forward_form_scalar: bool  # True => `carrier <op> scalar`
    scalar_carrier: torch.fx.Node
    if lhs_is_node and not rhs_is_node:
        assert isinstance(lhs, torch.fx.Node)
        scalar = _extract_scalar(rhs)
        scalar_carrier = lhs
        forward_form_scalar = True
    elif rhs_is_node and not lhs_is_node:
        assert isinstance(rhs, torch.fx.Node)
        scalar = _extract_scalar(lhs)
        scalar_carrier = rhs
        forward_form_scalar = False
    else:
        # Neither is an FX node — would mean a constant binary op
        # that should have been folded upstream. Bail.
        return None
    if scalar is None:
        return None
    if target is torch.ops.aten.add.Tensor:
        return (
            _UnaryStep(op_name="add", template=_add_const_template(scalar)),
            scalar_carrier,
        )
    if target is torch.ops.aten.mul.Tensor:
        return (
            _UnaryStep(op_name="mul", template=_mul_const_template(scalar)),
            scalar_carrier,
        )
    if target is torch.ops.aten.sub.Tensor:
        template = (
            _sub_const_template(scalar)
            if forward_form_scalar
            else _rsub_const_template(scalar)
        )
        return _UnaryStep(op_name="sub", template=template), scalar_carrier
    if target is torch.ops.aten.div.Tensor:
        template = (
            _div_const_template(scalar)
            if forward_form_scalar
            else _rdiv_const_template(scalar)
        )
        return _UnaryStep(op_name="div", template=template), scalar_carrier
    return None


def _carrier_tile_shape(node: torch.fx.Node) -> tuple[object, ...] | None:
    """Extract the carrier's tile shape from ``meta['val'].shape``.

    Returns ``None`` when the meta is missing. The classifier uses
    this shape to decide whether an auxiliary-tensor load matches
    the carrier's rank/extents — only exact-shape aux loads are
    accepted; broadcast / rank mismatches drop to the loud-failure
    backstop.
    """
    val = node.meta.get("val")
    if val is None:
        return None
    return tuple(val.shape)


def _carrier_tile_index_nodes(
    cast_input: torch.fx.Node,
) -> tuple[torch.fx.Node, ...] | None:
    """Extract the tile-id symbol FX nodes that index the chain carrier.

    Walks back from the post-matmul chain entry to the ``hl.zeros``
    initial-value node and reads its tile-shape list. Returns the
    tuple of FX symint nodes (one per tile axis), or ``None`` when
    the walk cannot recover them — the classifier then falls back
    to the looser shape-only check.

    For binary chain steps the walk picks the first
    ``all_input_nodes`` entry that is *not* a ``helion.language.load``
    call. The chain analyzer accepts both ``add(carrier, aux_load)``
    and ``add(aux_load, carrier)``; descending into the aux load side
    breaks the walk-back to ``hl.zeros``. Skipping aux load nodes
    keeps the walk on the carrier side regardless of operand order,
    so the reverse-form chain (``aux + carrier``) recovers the tile
    index symbols just like the forward form.

    Invariant: any ``hl.load`` input encountered during the walk is
    necessarily an aux load (never a carrier passthrough), because
    the carrier always originates at ``hl.zeros`` and is propagated
    through ``_phi`` / ``_new_var`` / ``getitem`` plus the accepted
    chain ops — none of which produces a load node along the
    carrier side.
    """
    import operator

    from ...language import _tracing_ops
    from ...language.memory_ops import load as helion_load

    cur: torch.fx.Node | None = cast_input
    visited: set[torch.fx.Node] = set()
    # Walk through identity-shape passthroughs to the loop entry,
    # then take the ``_phi`` initial-value branch to the
    # ``hl.zeros`` (``full``) node whose ``args[0]`` is the tile-
    # shape list.
    while cur is not None and cur not in visited:
        visited.add(cur)
        if cur.op != "call_function":
            return None
        target = cur.target
        if target is _tracing_ops._phi:
            init = cur.args[0] if cur.args else None
            if not isinstance(init, torch.fx.Node):
                return None
            shape_arg = init.args[0] if init.args else None
            if not isinstance(shape_arg, (list, tuple)):
                return None
            nodes: list[torch.fx.Node] = []
            for entry in shape_arg:
                if not isinstance(entry, torch.fx.Node):
                    return None
                nodes.append(entry)
            return tuple(nodes)
        if target is _tracing_ops._new_var or target is operator.getitem:
            arg = cur.args[0] if cur.args else None
            if not isinstance(arg, torch.fx.Node):
                return None
            cur = arg
            continue
        # The chain may carry a binary op whose carrier we want to
        # follow back. Pick the first ``all_input_nodes`` entry that
        # is not a ``helion.language.load`` call so we descend into
        # the carrier side regardless of operand order. Reverse-form
        # binaries (``aux_load <op> carrier``) put the aux load
        # first; without this skip the walk would descend into the
        # aux tensor and never find ``hl.zeros``.
        chosen: torch.fx.Node | None = None
        for inp in cur.all_input_nodes:
            if inp.op == "call_function" and inp.target is helion_load:
                continue
            chosen = inp
            break
        if chosen is None:
            return None
        cur = chosen
    return None


def analyze_tcgen05_unary_epilogue_chain(
    state: CodegenState,
    value_node: torch.fx.Node,
    *,
    output_global_shape: tuple[object, ...] | None = None,
) -> tuple[Tcgen05UnaryEpilogueChain, torch.fx.Node] | None:
    """Classify ``value_node``'s producer chain as a whitelisted
    epilogue rooted at a tcgen05 matmul.

    Expected shape: ``value_node`` is the user-side store value, which
    Helion has wrapped in an implicit ``convert_element_type`` to the
    store-target tensor's dtype. The chain we accept is, walking
    upstream from ``value_node``:

        convert_element_type (the implicit cast) ->
        [whitelisted op]* ->
        accumulator carrier (phi / getitem on for_loop output ->
        registered tcgen05 matmul fx_node).

    Whitelisted ops are zero-arg unary (``relu`` / ``tanh`` / ``exp``
    / ``log`` / ``sqrt`` / ``abs`` / ``neg``), scalar binary
    (``add`` / ``sub`` / ``mul`` / ``div`` against a compile-time
    Python literal), and auxiliary-tensor binary in two forms:
    exact-shape (``residual[tile_m, tile_n]``, rank-2 aux matching
    the carrier tile shape) and rank-1 trailing-axis (rowvec)
    broadcast (``bias[tile_n]``, where the single load index
    symbol matches the carrier's trailing tile-id symbol). Other
    shapes — 3-D collapsed loads, indices that are not exactly the
    carrier trailing tile-id symbol, leading-axis rank-1
    (``bias[tile_m]``), kwargs — are rejected so the loud-failure
    backstop fires.

    A user-written intermediate cast like
    ``out[tile] = relu(acc).to(d_inter)`` with ``d_inter`` not equal
    to the store target's dtype shows up as a *second*
    ``convert_element_type`` inside the chain. The chain step loop
    rejects that because ``convert_element_type`` is not on the
    whitelist; the loud-failure backstop then fires. This means the
    splice cannot silently drop a user-explicit intermediate cast —
    pinned by ``test_tcgen05_fused_chain_rejects_intermediate_cast_dtype_mismatch``.

    Returns ``(chain, matmul_anchor)`` on success — the rendered chain
    excludes the trailing ``convert_element_type`` (the splice site
    already emits ``.to(target_dtype)`` where ``target_dtype`` is the
    store-target tensor's dtype), and the anchor is the unique
    tcgen05 matmul fx_node whose ``result_var`` the splice should
    target. Returns ``None`` if the chain is not in the whitelist or
    multiple matmul anchors are reachable along the carrier path
    (multi-input epilogues are deferred). The caller falls back
    to the loud-failure ``BackendUnsupported`` raise on ``None`` so
    the diagnostic keeps firing for non-whitelisted shapes.

    The chain may have ``steps == ()`` — i.e. ``out[tile] =
    acc.to(x.dtype)`` — in which case the splice site emits the
    existing identity ``.to(target_dtype)`` line unchanged. The fast-
    path ``ast.Name``-matching code in ``store_codegen`` handles the
    identity case earlier; the empty-chain return from this function
    is a defensive belt-and-suspenders.

    ``output_global_shape`` is the user-side store target tensor's
    full (non-tile) shape, threaded into the rank-1 broadcast aux
    classifier so an aux whose extent only happens to match the
    tile but not the global axis is rejected at classify time.
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
        # byte-identical to the unary-only golden.
        return None

    # The chain carrier's expected tile shape and tile-id symbol
    # nodes, used to validate auxiliary-tensor loads. Read once at
    # the top — both are invariant along the chain because every
    # accepted op preserves shape (zero-arg unary, scalar-binary,
    # and exact-shape aux-binary all produce the same shape as
    # their input).
    carrier_tile_shape = _carrier_tile_shape(cast_input)
    carrier_tile_index_nodes = _carrier_tile_index_nodes(cast_input)

    steps: list[_UnaryStep | _AuxiliaryTensorStep] = []
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
        # Binary ops (scalar literal *or* aux-tensor load on the
        # other side). The classifier rejects multi-tensor cases
        # where neither operand is a recognized aux load and any
        # kwarg-bearing form (``alpha=k``, etc.).
        if target in _SCALAR_BINARY_TARGETS:
            classified = _classify_binary(
                cur,
                carrier_tile_shape=carrier_tile_shape,
                carrier_tile_index_nodes=carrier_tile_index_nodes,
                carrier_global_shape=output_global_shape,
            )
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
        # Op not on the whitelist. Surfacing None lets the
        # loud-failure diagnostic raise so the user sees the
        # actionable message.
        return None
    return None
