"""Small, side-effect-free helpers for exact CuTe FX graph matchers."""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..device_ir import DeviceIR


@dataclasses.dataclass(frozen=True)
class _TensorRef:
    fake: torch.Tensor
    name: str


def _same_ref(lhs: _TensorRef | None, rhs: _TensorRef | None) -> bool:
    return (
        lhs is not None
        and rhs is not None
        and lhs.name == rhs.name
        and lhs.fake is rhs.fake
    )


def _is_call(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


def _strip_value_casts(node: object) -> torch.fx.Node | None:
    if not isinstance(node, torch.fx.Node):
        return None
    current = node
    cast_targets = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach.default,
        torch.ops.prims.convert_element_type.default,
    }
    for _ in range(8):
        if current.op != "call_function" or current.target not in cast_targets:
            return current
        if not current.args or not isinstance(current.args[0], torch.fx.Node):
            return None
        current = current.args[0]
    return None


def _is_matrix_transpose_of(node: object, source: torch.fx.Node) -> bool:
    current = _strip_value_casts(node)
    if not isinstance(current, torch.fx.Node) or current.op != "call_function":
        return False
    if current.target is torch.ops.aten.t.default:
        return bool(current.args and current.args[0] is source)
    if current.target is torch.ops.aten.permute.default:
        permutation = current.args[1] if len(current.args) >= 2 else None
        return (
            isinstance(permutation, (list, tuple))
            and current.args[0] is source
            and tuple(permutation) == (1, 0)
        )
    if current.target is torch.ops.aten.transpose.int:
        first_dim = current.args[1] if len(current.args) >= 2 else None
        second_dim = current.args[2] if len(current.args) >= 3 else None
        return (
            isinstance(first_dim, int)
            and isinstance(second_dim, int)
            and current.args[0] is source
            and {first_dim, second_dim} in ({0, 1}, {-2, -1})
        )
    return False


def _host_tensor_ref(node: object) -> _TensorRef | None:
    from ...language._tracing_ops import _host_tensor

    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not _host_tensor
        or not node.args
        or not isinstance(node.args[0], str)
    ):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return _TensorRef(value, node.args[0])


def _load_ref(node: object) -> _TensorRef | None:
    from ...language import memory_ops

    if not _is_call(node, memory_ops.load):
        return None
    assert isinstance(node, torch.fx.Node)
    return _host_tensor_ref(node.args[0] if node.args else None)


def _store_ref(node: object) -> _TensorRef | None:
    from ...language import memory_ops

    if not _is_call(node, memory_ops.store):
        return None
    assert isinstance(node, torch.fx.Node)
    return _host_tensor_ref(node.args[0] if node.args else None)


def _memory_indices(node: torch.fx.Node) -> tuple[object, ...] | None:
    if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
        return None
    return tuple(node.args[1])


def _linear_offsets_fit_i32(
    work_items: int,
    tensor_numels: Sequence[int],
) -> bool:
    """Prove a flattened launch and all emitted signed offsets fit Int32."""

    max_i32 = torch.iinfo(torch.int32).max
    return 0 < work_items <= max_i32 and all(
        numel > 0 and numel - 1 <= max_i32 for numel in tensor_numels
    )


def _xyz_grid_fits(extents: Sequence[int]) -> bool:
    """Prove one physical CUDA XYZ launch fits each architectural dimension."""

    from torch._inductor.runtime.triton_heuristics import (  # type: ignore[import-untyped]
        get_max_y_grid,
    )

    if len(extents) != 3:
        return False
    max_i32 = torch.iinfo(torch.int32).max
    max_yz = get_max_y_grid()
    return (
        0 < extents[0] <= max_i32
        and 0 < extents[1] < max_yz
        and 0 < extents[2] < max_yz
    )


class _GeneratedCodeTemplate:
    """Render raw generated locals without colliding with kernel arguments."""

    def __init__(
        self,
        prefix: str,
        occupied_names: Sequence[str],
        new_var: Callable[[str], str],
    ) -> None:
        self._local_pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(prefix)}_[A-Za-z0-9_]+"
        )
        self._occupied_names = frozenset(occupied_names)
        self._new_var = new_var
        self._local_names: dict[str, str] = {}
        self._values: list[tuple[str, str]] = []

    def protect(self, value: str) -> str:
        """Return an invalid-Python sentinel replaced after local renaming."""

        token = f"<HELION_EXACT_VALUE_{len(self._values)}>"
        self._values.append((token, value))
        return token

    def render(self, source: str) -> str:
        def replacement(match: re.Match[str]) -> str:
            name = match.group(0)
            if name not in self._local_names:
                generated = self._new_var(name)
                while generated in self._occupied_names:
                    generated = self._new_var(name)
                self._local_names[name] = generated
            return self._local_names[name]

        source = self._local_pattern.sub(replacement, source)
        for token, value in self._values:
            source = source.replace(token, value)
        return source


def _canonical_root_axis_ids(
    device_ir: DeviceIR,
    *,
    root_phase_index: int,
    coordinates: Sequence[torch.fx.Node | None],
    expected_extents: Sequence[object],
) -> tuple[int, ...] | None:
    """Prove one whole-root replacement owns its exact logical task domain.

    Whole-root schedules synthesize their launch grid independently of the
    carrier.  Matching the body alone is therefore insufficient: a carrier
    with reordered coordinates, a non-zero start, or a truncated extent would
    otherwise be replaced by a kernel that executes a different task set.
    ``None`` coordinates intentionally leave one axis identified by its
    position only, while still checking its extent and canonical origin.
    """

    import sympy

    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from ..variable_origin import AttributeOrigin
    from ..variable_origin import GlobalOrigin

    if (
        root_phase_index != 0
        or len(device_ir.grid_block_ids) != 1
        or len(device_ir.task_families) != 1
    ):
        return None
    root_grid_ids = tuple(device_ir.grid_block_ids[0])
    task_family = device_ir.task_families[0]
    if (
        len(root_grid_ids) != len(coordinates)
        or len(coordinates) != len(expected_extents)
        or task_family.logical_axis_order != root_grid_ids
    ):
        return None

    env = CompileEnvironment.current()

    def extent_expr(value: object) -> sympy.Expr | None:
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        if not isinstance(value, (int, sympy.Expr)):
            return None
        result = env.specialize_expr(sympy.sympify(value))
        replacements: dict[sympy.Basic, int] = {}
        for symbol in result.free_symbols:
            origin_info = HostFunction.current().expr_to_origin.get(symbol)
            if origin_info is None:
                continue
            origin = origin_info.origin
            if isinstance(origin, AttributeOrigin) and isinstance(
                origin.value, GlobalOrigin
            ):
                constant = HostFunction.current().fn.__globals__.get(origin.key)
                if type(constant) is int:
                    replacements[symbol] = constant
        if replacements:
            result = env.specialize_expr(result.xreplace(replacements))
        return sympy.simplify(result)

    for block_id, coordinate, expected_extent in zip(
        root_grid_ids,
        coordinates,
        expected_extents,
        strict=True,
    ):
        axis = task_family.axis(block_id)
        axis_extent = extent_expr(axis.extent) if axis is not None else None
        expected_extent_expr = extent_expr(expected_extent)
        if (
            axis is None
            or not axis.canonical_origin
            or axis_extent is None
            or expected_extent_expr is None
            or axis_extent != expected_extent_expr
        ):
            return None
        if coordinate is not None:
            coordinate_value = coordinate.meta.get("val")
            if env.resolve_block_id(coordinate_value) != block_id:
                return None
    return root_grid_ids
