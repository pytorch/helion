"""Spyre layout decisions: host shape/dtype -> Spyre stick layout.

The decisions plan 4.2/4.3 assigns to Helion. ``device_coordinates[-1]`` is the
stick axis and ``superdsc._get_device_dim_order`` reads it without recomputing
(torch_spyre/_inductor/codegen/superdsc.py:367-382), so a layout chosen here is
final.

The dtype and stick-size mappings are torch-spyre's own (``get_device_dtype`` /
``get_elem_in_stick`` from ``torch_spyre._C``) rather than a second table here,
so they cannot drift from what the validator and SDSC codegen assume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    import sympy
    import torch


def device_dtype(dtype: torch.dtype) -> object:
    """The ``DataFormats`` member torch-spyre uses for a torch dtype.

    ``get_device_dtype`` returns ``DataFormats.INVALID`` for an unsupported
    dtype rather than raising (fp64 and complex both land there), and an
    INVALID format would travel all the way into the bundle. Reject it here so
    the failure names the dtype.
    """
    from torch_spyre._C import DataFormats, get_device_dtype

    try:
        resolved = get_device_dtype(dtype)
    except Exception as e:
        raise exc.BackendUnsupported("spyre", f"dtype: {dtype}") from e
    if resolved == DataFormats.INVALID:
        raise exc.BackendUnsupported("spyre", f"dtype: {dtype}")
    return resolved


def elems_per_stick(dtype: torch.dtype) -> int:
    """Elements per 128-byte stick: 64 at fp16/bf16, 32 at fp32.

    Validated through ``device_dtype`` first: ``get_elem_in_stick`` reports 0
    for an unsupported dtype, which would make a stick count divide by zero.
    """
    from torch_spyre._C import get_elem_in_stick

    device_dtype(dtype)
    eps = int(get_elem_in_stick(dtype))
    if eps <= 0:
        raise exc.BackendUnsupported("spyre", f"dtype: {dtype}")
    return eps


def stickify(
    sizes: list[int],
    symbols: list[sympy.Expr],
    stick_dim: int,
    dtype: torch.dtype,
) -> tuple[list[int], list[sympy.Expr]]:
    """Return ``(device_size, device_coordinates)`` for one operand.

    ``sizes[i]`` is the host extent of host dim ``i`` and ``symbols[i]`` the
    iteration symbol indexing it. ``stick_dim`` selects the host dim laid out
    along sticks, which plan 4.2 makes an operand-local choice: a matmul's three
    operands stick on three different host dims.

    Produces the layout torch-spyre's own propagate_layouts does, verified
    against the captured references in experiments/spyre_opspec_emitter/golden/:
    a 128x256 fp16 operand sticking on its last dim becomes device_size
    [4, 128, 64] with coordinates [floor(c1/64), c0, Mod(c1, 64)].
    """
    import sympy

    eps = elems_per_stick(dtype)
    stick_sym = symbols[stick_dim]
    n_sticks = -(-sizes[stick_dim] // eps)

    others = [i for i in range(len(sizes)) if i != stick_dim]
    device_size = [n_sticks, *(sizes[i] for i in others), eps]
    device_coordinates = [
        sympy.floor(stick_sym / eps),
        *(symbols[i] for i in others),
        sympy.Mod(stick_sym, eps),
    ]
    return device_size, device_coordinates


def stick_dim_of(tensor: torch.Tensor) -> int:
    """The host dim a device tensor is stickified on, read from its layout.

    Plan 5.3: input layouts are *read*, not chosen. ``device_tensor_layout()``
    is a torch-spyre monkey-patch (torch_spyre/_monkey_patch.py:130) returning
    ``SpyreTensorLayout | None``. ``None`` means the tensor carries no device
    layout (a fake tensor at trace time, or a host tensor), where the trailing
    dim is the only defensible choice -- torch-spyre's own default tiled layout.
    """
    getter = getattr(tensor, "device_tensor_layout", None)
    if getter is None:
        return tensor.dim() - 1
    try:
        stl = getter()
    except Exception:
        return tensor.dim() - 1
    if stl is None:
        return tensor.dim() - 1
    strides = list(stl.stride_map)
    ones = [i for i, s in enumerate(strides) if s == 1]
    if len(ones) != 1:
        raise exc.BackendUnsupported(
            "spyre", f"cannot identify stick axis from stride_map {strides}"
        )
    return ones[0]
