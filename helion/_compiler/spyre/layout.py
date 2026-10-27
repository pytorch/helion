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


#: A stick is a 128-byte aligned chunk, so a pool buffer starting mid-stick
#: would split one. Used to align ``hbm_pool`` byte offsets.
STICK_BYTES = 128


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
    if len(device_size) < 3:
        # A device layout is at least rank 3 (outer sticks, middle dims, within
        # stick), so a 1-D operand gets a leading unit axis at a constant
        # coordinate -- the shape the captured gather reference uses for its
        # 1-D index tensor: device_size [1, 2, 32], coords ['0', ...].
        device_size = [1, *device_size]
        device_coordinates = [sympy.Integer(0), *device_coordinates]
    return device_size, device_coordinates


def stick_dim_of(tensor: torch.Tensor) -> int:
    """The host dim a device tensor is stickified on, read from its layout.

    Plan 5.3: input layouts are *read*, not chosen. ``device_tensor_layout()``
    is a torch-spyre monkey-patch (torch_spyre/_monkey_patch.py:130) returning
    ``SpyreTensorLayout | None``. ``None`` means the tensor carries no device
    layout (a fake tensor at trace time, or a host tensor), where the trailing
    dim is the only defensible choice -- torch-spyre's own default tiled layout.

    The stick dim is recovered from ``device_size``, not ``stride_map``: both
    are **device**-dim indexed (length = host rank + 1), so indexing them by a
    host dim is a category error. ``stickify`` lays an operand out as
    ``[ceil(stick_extent / eps), *other_extents, eps]``, so the stick dim is
    the host dim whose extent is ``device_size[0] * device_size[-1]`` rounded
    to a stick, with the remaining host extents appearing in the middle in
    order.
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

    device_size = [int(d) for d in stl.device_size]
    host_size = [int(d) for d in tensor.size()]
    if len(device_size) != len(host_size) + 1:
        raise exc.BackendUnsupported(
            "spyre",
            f"device_size {device_size} does not match host shape {host_size}",
        )

    eps = device_size[-1]
    middle = device_size[1:-1]
    # Search from the trailing dim inwards. When two host dims both fit the
    # device layout the tensor is square-ish and the layout alone cannot say
    # which axis is the stick -- e.g. a 128x128 fp16 tensor has
    # device_size [2, 128, 64], and ceil(128/64) == 2 holds for *either* dim
    # with the other left as the middle extent. Taking the first match then
    # reports dim 0, which swaps c0/c1 in every coordinate the operand
    # contributes: the kernel computes correctly and writes its result
    # transposed, with nothing in the stack raising
    # (repros/square_pointwise_transposed.py).
    #
    # The trailing dim is the right tie-break: it is torch-spyre's own default
    # tiled layout and the same choice made above when there is no layout to
    # read at all, so an ambiguous tensor is treated exactly like an
    # unannotated one.
    for candidate in reversed(range(len(host_size))):
        others = [host_size[i] for i in range(len(host_size)) if i != candidate]
        if others != middle:
            continue
        if -(-host_size[candidate] // eps) == device_size[0]:
            return candidate

    raise exc.BackendUnsupported(
        "spyre",
        f"cannot identify stick axis: device_size {device_size}"
        f" for host shape {host_size}",
    )
