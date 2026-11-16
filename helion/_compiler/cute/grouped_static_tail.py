"""Grouped static-persistent GEMM layout tails: codegen analysis and bind-time key.

A grouped static-persistent tcgen05 GEMM (``tcgen05_grouped_mode``) reads
its per-group M extents from the ``layout`` tensor and its per-group N extents
from ``n_sizes``. When those tensors are available at codegen time
(``CompileEnvironment.runtime_arg_values_by_name``),
:func:`analyze_grouped_static_tail` proves whether any group has a partial M or
N tile. The pipeline planner (``cute_mma.plan_tcgen05_pipeline``) then drops
the tail-handling store protocol (``tcgen05_tma_store_full_tiles_only`` and
the ``EDGE_ONLY`` D TensorMap) for layouts without tails and publishes the two
facts on the ``tcgen05_grouped_static_persistent`` wrapper plan as
``grouped_static_has_m_tail`` / ``grouped_static_has_n_tail``.

Generated code therefore depends on tensor *values*, so every value read
through ``runtime_arg_values_by_name`` must be covered by a bind-time
specialization descriptor: :func:`tail_descriptors` turns the published
wrapper plans into ``cute_grouped_static_tail`` descriptors and
:func:`make_tail_extractor` recomputes the same two facts from the launch
arguments (through :func:`ordered_group_m_tail`, the walker both halves share)
so a call with a different layout re-binds instead of running the tail-free
kernel. The facts have three consumers: the planner flips in ``cute_mma.py``,
the bound-kernel key extension
(``runtime_hooks.register_grouped_static_tail_specializations``, run from
``finish_codegen``) and the launcher's re-checks
(``helion.runtime.cute.launcher``). ``runtime_hooks`` resolves this module's
functions at call time, so tests patch them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable
from typing import Hashable
from typing import cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils.weak import WeakIdKeyDictionary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment


def ordered_group_m_tail(
    layout_values: Sequence[int],
    *,
    group_count: int,
    bm: int,
    m_tail_preserve: bool,
) -> bool | None:
    """Walk an ordered group layout; ``None`` when it is not the proved shape.

    The layout lists each row's group id in group order. With
    ``m_tail_preserve`` every group but the first may be preceded by negative
    padding up to the next ``bm`` boundary and trailing padding is allowed;
    without it the groups must be adjacent and cover the whole layout. Every
    group must start on a ``bm`` boundary. Returns whether any group's row
    count is not a whole number of ``bm`` tiles.
    """
    cursor = 0
    has_m_tail = False
    for expected_group in range(group_count):
        if m_tail_preserve and expected_group > 0:
            next_m_boundary = ((cursor + bm - 1) // bm) * bm
            while (
                cursor < len(layout_values)
                and cursor < next_m_boundary
                and layout_values[cursor] < 0
            ):
                cursor += 1
            if cursor != next_m_boundary or (
                cursor < len(layout_values) and layout_values[cursor] < 0
            ):
                return None
        if cursor >= len(layout_values) or layout_values[cursor] != expected_group:
            return None
        start = cursor
        while cursor < len(layout_values) and layout_values[cursor] == expected_group:
            cursor += 1
        actual_m = cursor - start
        if start % bm != 0:
            return None
        has_m_tail = has_m_tail or actual_m % bm != 0
    if cursor != len(layout_values):
        if m_tail_preserve and all(value < 0 for value in layout_values[cursor:]):
            cursor = len(layout_values)
    if cursor != len(layout_values):
        return None
    return has_m_tail


def runtime_int_tensor_values(
    env: CompileEnvironment,
    arg_name: str,
    *,
    expected_numel: int,
) -> list[int] | None:
    """The values of a 1-D int tensor argument known at codegen time."""
    value = env.runtime_arg_values_by_name.get(arg_name)
    if not (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and int(value.numel()) == expected_numel
        and value.dtype in (torch.int32, torch.int64)
    ):
        return None
    with unset_fake_temporarily():
        return [int(item) for item in value.detach().cpu().tolist()]


def runtime_ordered_group_m_tail(
    env: CompileEnvironment,
    layout_arg_name: str,
    *,
    group_count: int,
    bm: int,
    m_tail_preserve: bool,
) -> bool | None:
    """:func:`ordered_group_m_tail` over the codegen-time ``layout`` values."""
    layout_value = env.runtime_arg_values_by_name.get(layout_arg_name)
    if not (
        isinstance(layout_value, torch.Tensor)
        and layout_value.ndim == 1
        and layout_value.dtype in (torch.int32, torch.int64)
    ):
        return None
    with unset_fake_temporarily():
        layout_values = [int(item) for item in layout_value.detach().cpu().tolist()]
    return ordered_group_m_tail(
        layout_values,
        group_count=group_count,
        bm=bm,
        m_tail_preserve=m_tail_preserve,
    )


@dataclass(frozen=True)
class GroupedStaticTailFacts:
    """Whether any group of a grouped static layout has a partial M / N tile."""

    has_m_tail: bool
    has_n_tail: bool


def analyze_grouped_static_tail(
    env: CompileEnvironment,
    *,
    layout_arg_name: str,
    n_sizes_arg_name: str | None,
    group_count: int,
    bm: int,
    bn: int,
    n_size: int,
    m_tail_preserve: bool,
) -> GroupedStaticTailFacts | None:
    """Tail facts of the grouped layout bound at codegen time, or ``None``.

    ``None`` when the layout / ``n_sizes`` values are unavailable, when a
    group's N extent is outside ``(0, n_size]`` or when the layout is not the
    proved ordered shape (:func:`ordered_group_m_tail`).
    """
    n_sizes_values = (
        runtime_int_tensor_values(
            env,
            n_sizes_arg_name,
            expected_numel=group_count,
        )
        if n_sizes_arg_name is not None
        else [n_size] * group_count
    )
    if n_sizes_values is None or any(
        group_n <= 0 or group_n > n_size for group_n in n_sizes_values
    ):
        return None
    has_m_tail = runtime_ordered_group_m_tail(
        env,
        layout_arg_name,
        group_count=group_count,
        bm=bm,
        m_tail_preserve=m_tail_preserve,
    )
    if has_m_tail is None:
        return None
    has_n_tail = any(group_n % bn != 0 for group_n in n_sizes_values)
    return GroupedStaticTailFacts(has_m_tail, has_n_tail)


def tail_descriptors(
    plans: Sequence[dict[str, object]],
) -> tuple[Hashable, ...]:
    """The ``cute_grouped_static_tail`` descriptors of the published wrapper plans."""
    descriptors: list[Hashable] = []
    for plan in plans:
        if plan.get("kind") != "tcgen05_grouped_static_persistent" or bool(
            plan.get("worklist_metadata")
        ):
            continue
        if not (
            isinstance(plan.get("grouped_static_has_m_tail"), bool)
            or isinstance(plan.get("grouped_static_has_n_tail"), bool)
        ):
            continue
        layout_idx = plan.get("layout_bind_idx")
        group_count = plan.get("group_count")
        bm = plan.get("bm")
        if not (
            isinstance(layout_idx, int)
            and isinstance(group_count, int)
            and isinstance(bm, int)
        ):
            continue
        n_sizes_idx = plan.get("n_sizes_bind_idx")
        bn = plan.get("bn")
        descriptors.append(
            (
                "cute_grouped_static_tail",
                layout_idx,
                group_count,
                bm,
                n_sizes_idx if isinstance(n_sizes_idx, int) else None,
                bn if isinstance(bn, int) else None,
            )
        )
    return tuple(sorted(descriptors, key=repr))


def int_1d_tensor_values(
    value: object,
    cache: WeakIdKeyDictionary,
) -> tuple[int, ...] | None:
    """The values of a 1-D int launch argument, cached by (version, ptr, layout)."""
    if not (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and value.dtype in (torch.int32, torch.int64)
    ):
        return None
    if torch.is_inference(value):
        return tuple(int(v) for v in value.detach().cpu().tolist())
    signature = (
        int(value._version),
        int(value.data_ptr()),
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
    )
    try:
        cached_signature, cached_values = cache[value]
    except KeyError:
        pass
    else:
        if cached_signature == signature:
            return cached_values
    values = tuple(int(v) for v in value.detach().cpu().tolist())
    cache[value] = (signature, values)
    return values


def layout_has_m_tail(
    layout_values: tuple[int, ...],
    *,
    bm: int,
    group_count: int,
) -> bool | None:
    """The bind-time M-tail fact; the key always preserves the M tail."""
    return ordered_group_m_tail(
        layout_values, group_count=group_count, bm=bm, m_tail_preserve=True
    )


def make_tail_extractor(
    descriptor: Hashable,
) -> Callable[[Sequence[object]], Hashable]:
    """The specialization extractor recomputing a descriptor's tail facts."""
    (
        _label,
        layout_idx,
        group_count,
        bm,
        n_sizes_idx,
        bn,
    ) = cast("tuple[object, int, int, int, int | None, int | None]", descriptor)
    tensor_values_cache: WeakIdKeyDictionary = WeakIdKeyDictionary()

    def cute_grouped_static_tail_extractor(
        args: Sequence[object],
        *,
        _descriptor: Hashable = descriptor,
        _layout_idx: int = layout_idx,
        _group_count: int = group_count,
        _bm: int = bm,
        _n_sizes_idx: int | None = n_sizes_idx,
        _bn: int | None = bn,
    ) -> Hashable:
        layout_has_m_tail_fact: bool | None = None
        if _layout_idx < len(args):
            layout_values = int_1d_tensor_values(
                args[_layout_idx],
                tensor_values_cache,
            )
            if layout_values is not None:
                layout_has_m_tail_fact = layout_has_m_tail(
                    layout_values,
                    bm=_bm,
                    group_count=_group_count,
                )
        n_sizes_has_n_tail: bool | None = None
        if _n_sizes_idx is not None and _bn is not None and _n_sizes_idx < len(args):
            n_sizes_values = int_1d_tensor_values(
                args[_n_sizes_idx],
                tensor_values_cache,
            )
            if n_sizes_values is not None and len(n_sizes_values) == _group_count:
                n_sizes_has_n_tail = any(
                    group_n % _bn != 0 for group_n in n_sizes_values
                )
        return (_descriptor, layout_has_m_tail_fact, n_sizes_has_n_tail)

    return cute_grouped_static_tail_extractor
