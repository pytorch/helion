"""Internal tanh-approximation GELU op.

Not a user-facing API. The user-facing surface is
``torch.nn.functional.gelu(x, approximate="tanh")``; ``device_ir`` installs
a decomposition (see ``install_gelu_decomp`` below) that maps the
``approximate="tanh"`` overload onto the single-FX-node
``_gelu_tanh_approx`` op defined here. The ``approximate="none"`` path
keeps the inductor default decomposition (the erf form).

The polynomial form ``0.5 * x * (1 + tanh(x * (sqrt(2/pi) +
sqrt(2/pi) * 0.044715 * x * x)))`` references ``x`` four times. Spelled
out as four primitive ops, this breaks the linear-chain assumption of
Helion's tcgen05 epilogue chain analyzer
(``helion/_compiler/cute/cute_epilogue.py``) and falls back to the
loud-failure backstop. Folding the whole expression behind a single
op lets the chain analyzer see exactly one ``_gelu_tanh_approx`` FX
node and splice the polynomial inline as one chain step (single
hoisted local for ``x``, single rendered expression — register-resident
throughout).

Inside the tcgen05 chain analyzer, ``_gelu_tanh_approx`` is registered
as a ``_UnaryStep`` row in ``_ZERO_ARG_TARGETS`` keyed on the api
wrapper itself (the FX target). The template binds ``x`` to a single
hoisted local (``inner_ref_count = 4``) so the rendered expression
references the input exactly once even though the polynomial has four
occurrences of ``x``.

Backend support: ``cute`` and ``triton`` only. The ``pallas`` backend
raises :class:`exc.BackendUnsupported` because Mosaic does not have a
direct ``cute.math.tanh`` analog and the polynomial would need a
separate Pallas-flavoured lowering (the same primitive can be
spelled directly with ``jax.nn.gelu(x, approximate=True)`` in user
code today).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

import torch
from torch._inductor.utils import triton_type

from .. import exc
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState


# Tanh-approximation GELU constants. Spelled out as ``float`` literals
# rather than ``math.sqrt(2.0 / math.pi)`` at import time so the
# rendered Python literals are byte-identical across machines and
# pinned by the codegen-marker tests in ``test_cute_lowerings.py``.
#
# ``kappa = sqrt(2/pi)``, ``lambda = sqrt(2/pi) * 0.044715``. Both are
# the same constants Quack uses in ``quack.activation.gelu_tanh_approx``
# (``quack/quack/activation.py``); pinned here so the rendered
# expression matches PyTorch's
# ``torch.nn.functional.gelu(x, approximate="tanh")`` to bf16
# precision.
GELU_TANH_APPROX_KAPPA: float = 0.7978845608028654
GELU_TANH_APPROX_LAMBDA: float = 0.035677408136300125


# Templates for backend codegen.
#
# The cute template uses ``{inner}`` directly so it plugs into the
# tcgen05 chain analyzer's ``_UnaryStep.template`` slot
# (``cute_epilogue.py`` substitutes ``{inner}`` after hoisting the
# carrier expression to a single local). The cute-backend codegen
# below also calls ``.format(inner=...)`` against the lifted local
# name to share the same template across the splice site and
# pointwise paths.
#
# The triton template is rendered as two layers: the inner ``{x32}``
# placeholder is fp32 (cast happens at the codegen-site for fp16 /
# bf16 inputs to satisfy ``libdevice.tanh``'s fp32-only contract);
# the outer ``{x}`` is the original-dtype value preserved for the
# leading ``0.5 * x`` factor that has no transcendental component.
# This keeps the fp32 round-trip narrow — only the ``tanh`` argument
# is cast — and matches Helion's existing ``tanh`` lowering shape
# (``inductor_lowering_extra.FP32_FALLBACK_OPS_UNARY``). For fp32
# inputs the codegen treats ``{x32}`` and ``{x}`` as the same local.
# NOTE: the cute template has no bf16/fp16 round-trip on the carrier,
# unlike the triton template below. The tcgen05 chain analyzer splices
# this template at a per-thread T2R register where the accumulator is
# always fp32 (the carrier is the matmul accumulator), so no cast is
# needed. The standalone cute pointwise codegen path also calls this
# template; that path runs through cute_dsl which broadens bf16/fp16
# inputs to fp32 around ``cute.math.tanh`` automatically, so the
# absence of an explicit cast is intentional and safe for both call
# sites.
_GELU_TANH_APPROX_EXPR_CUTE = (
    f"(0.5 * ({{inner}}) * (1.0 + cute.math.tanh(({{inner}}) *"
    f" ({GELU_TANH_APPROX_KAPPA!r} + {GELU_TANH_APPROX_LAMBDA!r}"
    f" * ({{inner}}) * ({{inner}})))))"
)
_GELU_TANH_APPROX_EXPR_TRITON = (
    f"(0.5 * ({{x}}) * (1.0 + libdevice.tanh(({{x32}}) * ({GELU_TANH_APPROX_KAPPA!r}"
    f" + {GELU_TANH_APPROX_LAMBDA!r} * ({{x32}}) * ({{x32}})))))"
)


# Number of times the rendered cute expression substitutes ``{inner}``.
# The polynomial references ``x`` four times: once in the leading
# ``0.5 * x``, once in ``x * (...)`` inside tanh, and twice in
# ``x * x``. The chain renderer (``_UnaryStep.inner_ref_count > 1``)
# uses this count to hoist ``{inner}`` to a single local first so the
# template references only that local — keeping rendered source O(1)
# in the input expression size.
GELU_TANH_APPROX_INNER_REF_COUNT: int = 4


@_decorators.api(is_device_only=True)
def _gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    """Internal tanh-approximation GELU op (see module docstring).

    Computes ``0.5 * x * (1 + tanh(x * (sqrt(2/pi) + sqrt(2/pi) *
    0.044715 * x * x)))``. Not user-facing — invoked via the
    ``aten.gelu.default`` decomposition installed by
    :func:`install_gelu_decomp`. For fp16 / bf16 inputs the polynomial
    runs in fp32 (Triton's ``libdevice.tanh`` is fp32-only) and the
    result is cast back to the input dtype.

    Backend support: ``cute`` and ``triton``. ``pallas`` raises
    :class:`exc.BackendUnsupported`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(_gelu_tanh_approx)
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@_decorators.codegen(_gelu_tanh_approx, "triton")
def _(state: CodegenState) -> ast.AST:
    # Lift the input to a local so the four ``{x}`` references all
    # bind to the same Triton SSA name; without the lift, the
    # rendered expression would textually duplicate the inbound
    # expression four times (Triton compile-time CSE handles the
    # values, but generated source size grows quadratically with
    # chain depth, which we explicitly avoid in cute_epilogue.py too).
    input_ast = state.codegen.lift(
        state.ast_arg(0), dce=True, prefix="gelu_tanh_approx_in"
    )
    # ``libdevice.tanh`` is fp32-only on Triton/CUDA. For fp16 / bf16
    # inputs cast the *tanh argument* (``{x32}``) to fp32 (the
    # surrounding multiplies promote through to fp32 too) and then
    # narrow the whole result back to the input dtype before
    # returning, matching ``register_fake``'s ``torch.empty_like(x)``
    # contract. Mirrors the round-trip Helion installs for
    # ``aten.tanh.default`` via
    # ``inductor_lowering_extra.FP32_FALLBACK_OPS_UNARY``.
    proxy = state.proxy_args[0]
    orig_dtype: torch.dtype | None = None
    if isinstance(proxy, torch.Tensor) and proxy.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        orig_dtype = proxy.dtype
    if orig_dtype is not None:
        x32_local = state.codegen.lift(
            expr_from_string(f"{input_ast.id}.to(tl.float32)"),
            dce=True,
            prefix="gelu_tanh_approx_fp32",
        )
        x32_id = x32_local.id
    else:
        x32_id = input_ast.id
    expr = _GELU_TANH_APPROX_EXPR_TRITON.replace("{x}", input_ast.id).replace(
        "{x32}", x32_id
    )
    if orig_dtype is not None:
        # Narrow the fp32 polynomial result back to the input dtype so
        # the FX-level same-dtype contract holds for callers that do
        # not immediately follow with ``.to(x.dtype)``.
        expr = f"({expr}).to({triton_type(orig_dtype)})"
    return expr_from_string(expr)


@_decorators.codegen(_gelu_tanh_approx, "cute")
def _(state: CodegenState) -> ast.AST:
    # Same lift-to-single-local rationale as the triton path: see the
    # ``inner_ref_count`` discussion above and the docstring on
    # :class:`Tcgen05UnaryEpilogueChain` (``cute_epilogue.py``).
    input_ast = state.codegen.lift(
        state.ast_arg(0), dce=True, prefix="gelu_tanh_approx_in"
    )
    return expr_from_string(epilogue_unary_step_template().format(inner=input_ast.id))


@_decorators.codegen(_gelu_tanh_approx, "pallas")
def _(state: CodegenState) -> ast.AST:
    # Pallas does not have a ``cute.math.tanh`` analog wired through
    # Helion today; raise a structured ``BackendUnsupported`` so the
    # diagnostic is actionable rather than failing at codegen lookup
    # with a missing-implementation error. Users can spell
    # ``jax.nn.gelu(x, approximate=True)`` directly when targeting
    # Pallas.
    raise exc.BackendUnsupported(
        "pallas",
        "F.gelu(x, approximate='tanh') (cute and triton only)",
    )


@_decorators.ref(_gelu_tanh_approx)
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def epilogue_unary_step_template() -> str:
    """Return the cute-backend template (``{inner}``-keyed) for the
    tcgen05 epilogue chain analyzer.

    The chain renderer in ``cute_epilogue.py`` substitutes ``{inner}``
    after hoisting the carrier expression to a fresh local; the
    template returned here is the same one the cute-backend codegen
    in this module uses, so the two paths cannot drift apart.
    """
    return _GELU_TANH_APPROX_EXPR_CUTE


def install_gelu_decomp(
    decomp_table: dict[torch._ops.OpOverload, Callable[..., object]],
) -> None:
    """Route ``F.gelu(x, approximate='tanh')`` through ``_gelu_tanh_approx``.

    ``aten.gelu.default`` is the dispatch target for both
    ``approximate='none'`` (default, erf form) and ``approximate='tanh'``.
    Inductor's default decomposition expands the tanh form into the
    4-x-reference polynomial, which the cute epilogue chain analyzer
    cannot fuse. We replace the entry with a wrapper that branches on
    the kwarg: tanh maps to a single ``_gelu_tanh_approx`` FX node, and
    every other approximate value falls through to the original
    inductor decomp (or the bare aten op if none was registered).
    """
    original_decomp = decomp_table[torch.ops.aten.gelu.default]

    def _gelu_decomp(x: torch.Tensor, *, approximate: str = "none") -> torch.Tensor:
        if approximate == "tanh":
            return _gelu_tanh_approx(x)
        # pyrefly: ignore [bad-return]
        return original_decomp(x, approximate=approximate)

    decomp_table[torch.ops.aten.gelu.default] = _gelu_decomp
