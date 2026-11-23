"""Shared by the tests of the cache hints on single-element loads:
``test_cute_scalar_load_hints.py`` (the 4- and 8-byte elements of
``cute_scalar_load_hints``) and ``test_cute_narrow_load_hints.py`` (the
2-byte elements of ``cute_narrow_load_hints``).

Two kernels whose loads are single elements, bound once per element type and
size on CPU; the load lines of their generated code; the MLIR the two load
helpers build; and, for the GPU numerics, inputs with every special value
and a comparison of every bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from . import DEVICE
from .cute_codegen import cpu_bind
import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def component_sum(pairs: torch.Tensor) -> torch.Tensor:
    # ``pairs`` is [n, 2]: the tile lane sits on a stride-2 dim, so both
    # components load element by element, from two load slots of one tensor.
    n = pairs.size(0)
    out = torch.empty([n], dtype=pairs.dtype, device=pairs.device)
    for tile in hl.tile(n):
        out[tile] = pairs[tile, 0] + pairs[tile, 1]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def component_add(pairs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # One strided component and a contiguous operand: two tensors, a load
    # slot each.
    n = pairs.size(0)
    out = torch.empty([n], dtype=pairs.dtype, device=pairs.device)
    for tile in hl.tile(n):
        out[tile] = pairs[tile, 0] + x[tile]
    return out


# One bound kernel per (kernel, dtype, n): 1000 elements leave a partial last
# tile, so the loads are masked; 1024 are unmasked.
_BOUND: dict[tuple[str, torch.dtype, int], BoundKernel[object]] = {}


def bound_components(
    dtype: torch.dtype, n: int, kernel: Kernel[torch.Tensor] = component_sum
) -> BoundKernel[object]:
    """``component_sum`` or ``component_add`` bound on CPU for ``n`` pairs of
    ``dtype`` elements, once per module run."""
    key = (kernel.name, dtype, n)
    if key not in _BOUND:
        arguments: tuple[torch.Tensor, ...] = (torch.empty((n, 2), dtype=dtype),)
        if kernel is component_add:
            arguments = (*arguments, torch.empty((n,), dtype=dtype))
        _BOUND[key] = cpu_bind(cast("Kernel[object]", kernel), arguments)
    return _BOUND[key]


def policy_config(
    bound: BoundKernel[object], policies: list[str], **extra: object
) -> helion.Config:
    """The default config of ``bound`` in tiles of 256 with the load slots'
    ``policies`` and the ``extra`` keys, normalized."""
    spec = bound.config_spec
    return spec.normalized_config(
        helion.Config.from_dict(
            spec.default_config().config
            | {"block_sizes": [256], "load_eviction_policies": policies}
            | extra
        )
    )


def load_lines(code: str) -> list[str]:
    """The kernel body's load statements: the element loads and the packets
    element loads merged into."""
    body = code.split("def _helion_")[1].split("\ndef ")[0]
    return [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith(("load = ", "load_1 = ", "_helion_literal_vec"))
    ]


def lines_without_loads(code: str) -> list[str]:
    """Every line of the module but the element loads and the imports of
    their helpers: what a hint must leave as it was."""
    return [
        line
        for line in code.splitlines()
        if "load = " not in line
        and "load_1 = " not in line
        and "_policy_load as _cute_" not in line
        and "_policy_load_if as _cute_" not in line
    ]


def element_load_helper_ir(dtype_names: Sequence[str]) -> tuple[str, int]:
    """The verified MLIR of ``element_policy_load`` and
    ``element_policy_load_if`` for each cutlass type of ``dtype_names`` with
    an L1 priority and an L2 family, and the number of loads in it."""
    import cutlass
    from cutlass._mlir import ir as mlir_ir
    from cutlass._mlir.dialects import func as mlir_func

    from helion._compiler.cute.element_policy_loads import element_policy_load
    from helion._compiler.cute.element_policy_loads import element_policy_load_if

    class Address:
        def __init__(self, value: object) -> None:
            self.value = value

        def toint(self, **kwargs: object) -> Address:
            return self

        def ir_value(self, **kwargs: object) -> object:
            return self.value

    # The MLIR bindings carry no type information.
    ir: Any = mlir_ir
    func: Any = mlir_func
    dtypes = [getattr(cutlass, name) for name in dtype_names]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            for dtype in dtypes:
                for policy in ("last", "l1_l2_last"):
                    for predicated in (False, True):
                        fn = func.FuncOp(
                            f"load_{dtype.__name__}_{policy}_{int(predicated)}",
                            ir.FunctionType.get(
                                [cutlass.Uint64.mlir_type, cutlass.Int32.mlir_type],
                                [dtype.mlir_type],
                            ),
                        )
                        block = fn.add_entry_block()
                        with ir.InsertionPoint(block):
                            address = Address(block.arguments[0])
                            if predicated:
                                value = element_policy_load_if(
                                    address,
                                    cutlass.Int32(block.arguments[1]) != 0,
                                    dtype,
                                    policy,
                                )
                            else:
                                value = element_policy_load(address, dtype, policy)
                            assert type(value) is dtype
                            func.ReturnOp([value.ir_value()])
        module.operation.verify()
        return str(module), len(dtypes) * 2 * 2


def adversarial_pairs(dtype: torch.dtype, n: int, layout: str) -> torch.Tensor:
    """``[n, 2]`` elements of ``dtype`` on the test device, as a
    ``contiguous`` tensor, a view at a storage ``offset`` or a ``permuted``
    view: floats with a NaN, an inf and a negative zero, integers over
    their upper bits (two of them still add without overflow)."""
    generator = torch.Generator().manual_seed(n)
    if dtype.is_floating_point:
        base = torch.randn((n + 5, 2), generator=generator, dtype=torch.float32)
        base[1, 0] = float("nan")
        base[2, 1] = float("inf")
        base[3, 0] = -0.0
    else:
        bound = 2 ** (8 * dtype.itemsize - 2)
        base = torch.randint(-bound, bound, (n + 5, 2), generator=generator)
    base = base.to(dtype).to(DEVICE)
    if layout == "contiguous":
        return base[:n].contiguous()
    if layout == "offset":
        # A storage offset of an odd number of elements.
        return base[5:]
    assert layout == "permuted"
    # Components along the stride-1 dim become rows of a transposed view.
    return base[:n].t().contiguous().t()


def assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Loads only: every bit of every element must survive (NaN payloads
    aside, signed zeros, integer high bits)."""
    assert actual.dtype == expected.dtype
    if not expected.dtype.is_floating_point:
        assert torch.equal(actual, expected)
        return
    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[expected.dtype.itemsize]
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(width), expected[~nan].view(width))
