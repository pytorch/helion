"""Spyre codegen for ops defined in ``helion.language.creation_ops``.

Plan 2: an ``hl.zeros``/``hl.full`` accumulator becomes **an explicit fill
OpSpec into a real buffer, emitted before the enclosing LoopSpec**, because
SuperDSC has no ``iter_args``. This is the same shape torch-spyre's own coarse
tiler produces: the K-tiled reference opens with an ``identity`` writing the
accumulator (``ir_chain=('mm', 'coarse_tile_fill_buf0')``) ahead of its
``LoopSpec``, then combines into it inside the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.creation_ops import full

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .backend import SpyreBackend
    from .emitter import Operand


@_decorators.codegen(full, "spyre")
def _(state: CodegenState) -> object:
    """Allocate the accumulator buffer and record its fill value.

    The fill OpSpec itself is emitted by the ``_for_loop`` handler, which is
    the only place that knows where "before the enclosing loop" is. Emitting it
    here would put it *inside* the loop body and re-zero the carry every
    iteration.

    Returns an ``ast.Name`` placeholder rather than the buffer: an accumulator
    is typically captured by a ``_for_loop``, whose ``codegen`` asserts every
    captured arg is an ``ast.AST`` (device_ir.py:382). The buffer is recovered
    from emitter state by that name.
    """
    from ..ast_extension import expr_from_string
    from ..compile_environment import CompileEnvironment
    from .compute_ops import tile_space

    value = state.proxy_arg(1)
    if not isinstance(value, (int, float, bool)):
        raise exc.BackendUnsupported("spyre", "hl.full with a dynamic fill value")
    if value != 0:
        # A non-zero fill needs a constant operand, which means an op_info
        # constant this milestone does not model. Fail loudly (plan 2).
        raise exc.BackendUnsupported(
            "spyre", f"hl.full fill value {value!r}; only 0 is modelled"
        )

    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    sizes, _ = tile_space(fake_value)

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    node = state.fx_node
    assert node is not None
    operand = backend.emitter.add_intermediate(
        node.name,
        sizes,
        fake_value.dtype,
        # The accumulator sticks on its trailing dim, matching the reference's
        # accumulator layout ([1, 64, 64] for a 64x32 fp16 acc).
        len(sizes) - 1,
        residency="lx",
    )
    backend.emitter.mark_accumulator(operand, float(value))
    return expr_from_string(backend.emitter.bind_carry(operand))