"""aten target -> Spyre op name.

Plan 7 "Op-name reuse": the table is lifted from torch-spyre's ``SpyreOpFuncs``
(torch_spyre/_inductor/spyre_kernel.py:237-419), which is the one genuinely
reusable asset from the inductor backend -- and it is reused here as what it
always was, a table. Canonical op-name constants:
torch_spyre/_inductor/constants.py:19-37.

Unmapped targets are not silently dropped: ``spyre_op_name`` returns ``None``
and the caller raises ``BackendUnsupported``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

aten = torch.ops.aten

#: Pointwise aten overloads. Names verified against SpyreOpFuncs by extracting
#: every ``PointwiseOp("<name>", ...)`` literal from the installed package.
_POINTWISE: dict[object, str] = {
    aten.add.Tensor: "add",
    aten.sub.Tensor: "sub",
    aten.mul.Tensor: "mul",
    aten.div.Tensor: "realdiv",
    aten.abs.default: "abs",
    aten.neg.default: "neg",
    aten.exp.default: "exp",
    aten.log.default: "log",
    aten.sqrt.default: "sqrt",
    aten.rsqrt.default: "rsqrt",
    aten.reciprocal.default: "reciprocal",
    aten.floor.default: "floor",
    aten.tanh.default: "tanh",
    aten.sigmoid.default: "sigmoid",
    aten.relu.default: "relufwd",
    aten.gelu.default: "gelufwd",
    aten.silu.default: "silu",
    aten.maximum.default: "maximum",
    aten.minimum.default: "minimum",
    aten.eq.Tensor: "equal",
    aten.ne.Tensor: "notequal",
    aten.lt.Tensor: "lesserthan",
    aten.le.Tensor: "lesserequal",
    aten.gt.Tensor: "greaterthan",
    aten.ge.Tensor: "greaterequal",
    aten.where.self: "where3",
    aten.clamp.default: "clip",
}

#: Matmul-family overloads. All lower to ``batchmatmul``
#: (constants.BATCH_MATMUL_OP); the captured reference for ``torch.mm`` at
#: 64x256 @ 256x128 is ``op='batchmatmul', is_reduction=True``, not "matmul".
_MATMUL: dict[object, str] = {
    aten.mm.default: "batchmatmul",
    aten.bmm.default: "batchmatmul",
    aten.matmul.default: "batchmatmul",
}

#: ``addmm(acc, x, y)`` is a matmul *plus* an accumulate into an existing
#: buffer, so it emits two OpSpecs (batchmatmul into a partial, then add into
#: the accumulator) rather than one -- which is exactly the shape torch-spyre's
#: own coarse tiler produces inside a K-tiled LoopSpec.
_ACCUMULATING_MATMUL: dict[object, str] = {
    aten.addmm.default: "batchmatmul",
    aten.baddbmm.default: "batchmatmul",
}

#: Reductions. ``prod`` cannot reduce along the stick dim
#: (constants.REDUCTIONS_NON_STICK_DIM_ONLY), which the layout code must honour.
_REDUCTION: dict[object, str] = {
    aten.sum.default: "add",
    aten.sum.dim_IntList: "add",
    aten.amax.default: "maximum",
    aten.amin.default: "minimum",
}


def spyre_op_name(target: object) -> str | None:
    """Spyre op name for an aten target, or ``None`` when unmapped."""
    for table in (_POINTWISE, _MATMUL, _ACCUMULATING_MATMUL, _REDUCTION):
        name = table.get(target)
        if name is not None:
            return name
    return None


def is_matmul(target: object) -> bool:
    return target in _MATMUL


def is_accumulating_matmul(target: object) -> bool:
    return target in _ACCUMULATING_MATMUL


def is_reduction(target: object) -> bool:
    return target in _REDUCTION
