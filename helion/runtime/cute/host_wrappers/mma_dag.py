"""Host side of the ``cute_mma_dag`` emitters: the alias guard over the
tensor arguments of a mma_dag launch (and, from the registry, the host
wrapper emitters of its TMA, alias, fixup and split-K plan kinds).
"""

from __future__ import annotations

import bisect
import operator
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch

from .... import exc
from ..arg_signature import schema_pointer_alignment
from ..launch_descriptor import cute_launch_descriptor
from .context import HostWrapperContext
from .context import HostWrapperEmitter

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def tensor_byte_span(tensor: torch.Tensor) -> int:
    """Bytes from the tensor's first element to one past its last (the span
    of storage its view can touch; 0 for an empty tensor)."""
    if tensor.numel() == 0:
        return 0
    last = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        last += (size - 1) * stride
    return (last + 1) * tensor.element_size()


ALIAS_PASSED_LIMIT = 64


class MmaDagAliasGuard:
    """Refuses a cute_mma_dag launch whose tensor arguments alias.

    The planner admits a program whose loads and stores touch distinct
    storages and the emitter hoists side loads and stages tiles ahead of
    the stores on that proof.  The proof is over the fake tensors the
    kernel is compiled against (one storage per argument), so a caller
    passing one tensor, or two views of one buffer, under two names is
    only visible here: the byte span of every stored argument must not
    overlap any other tensor argument's.  Two loads may alias freely (the
    engine passes views of one tensor and the same tensor twice).

    The spans follow from the pinned tensor metadata, so a guard built for
    one launch holds for the fast relaunches that share that metadata; the
    per-launch work is the pointers, and a launch repeating a pointer set
    that passed recently (the allocator cycles through a few blocks for a
    workload's fresh outputs) is not checked again.  A stored span overlaps
    another iff more than one span (its own) begins before it ends and
    does not end before it begins: two bisections per store over the
    sorted begins and ends; only a failure names the pair.
    """

    __slots__ = ("indices", "names", "passed", "pick", "spans", "stores")

    def __init__(self, plan: dict[str, object], args: tuple[object, ...]) -> None:
        indices = plan.get("tensor_idx")
        names = plan.get("tensor_names")
        stored = plan.get("stored")
        if (
            not isinstance(indices, list)
            or not isinstance(names, list)
            or not isinstance(stored, list)
            or len(indices) != len(names)
            or len(indices) != len(stored)
        ):
            raise exc.BackendUnsupported("cute", "invalid mma_dag alias plan")
        kept: list[int] = []
        kept_names: list[str] = []
        stores: list[int] = []
        spans: list[int] = []
        for index, name, is_store in zip(indices, names, stored, strict=True):
            if type(index) is not int or not isinstance(name, str):
                raise exc.BackendUnsupported("cute", "invalid mma_dag alias plan")
            tensor = args[index]
            if not isinstance(tensor, torch.Tensor):
                raise exc.BackendUnsupported(
                    "cute", f"mma_dag argument {name!r} is not a tensor"
                )
            span = tensor_byte_span(tensor)
            if span == 0:
                continue
            if is_store:
                stores.append(len(kept))
            kept.append(index)
            kept_names.append(name)
            spans.append(span)
        self.indices = tuple(kept)
        self.names = tuple(kept_names)
        self.spans = tuple(spans)
        self.stores = tuple(stores)
        # The guarded tensors of a launch (``args[index]`` for each index).
        self.pick = cast(
            "Callable[[tuple[object, ...]], Sequence[torch.Tensor]]",
            operator.itemgetter(*kept)
            if len(kept) > 1
            else (lambda args: tuple(args[index] for index in kept)),
        )
        self.passed: set[tuple[int, ...]] = set()

    def check(self, args: tuple[object, ...]) -> None:
        self.check_begins(tuple(map(torch.Tensor.data_ptr, self.pick(args))))

    def check_begins(self, begins: tuple[int, ...]) -> None:
        """``check`` on the guarded tensors' data pointers in ``indices``
        order (the fast relaunch gathers them once for every guard)."""
        if begins in self.passed:
            return
        ends = tuple(map(operator.add, begins, self.spans))
        sorted_begins = sorted(begins)
        sorted_ends = sorted(ends)
        for store in self.stores:
            if (
                bisect.bisect_left(sorted_begins, ends[store])
                - bisect.bisect_right(sorted_ends, begins[store])
                > 1
            ):
                self.refuse(store, begins, ends)
        if len(self.passed) >= ALIAS_PASSED_LIMIT:
            self.passed.clear()
        self.passed.add(begins)

    def refuse(
        self, store: int, begins: tuple[int, ...], ends: tuple[int, ...]
    ) -> NoReturn:
        other = next(
            index
            for index in range(len(begins))
            if index != store
            and begins[index] < ends[store]
            and begins[store] < ends[index]
        )
        first, second = sorted((store, other))
        raise exc.BackendUnsupported(
            "cute",
            f"cute_mma_dag arguments {self.names[first]!r} and "
            f"{self.names[second]!r} alias (their storage overlaps); the "
            "kernel reads and writes them as distinct tensors",
        )


def mma_dag_alias_guard(
    cute_kernel: object, args: tuple[object, ...]
) -> MmaDagAliasGuard | None:
    plan = cute_launch_descriptor(cute_kernel).alias_plan
    if plan is None:
        return None
    return MmaDagAliasGuard(plan, args)


def emit_alias(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_mma_dag_alias``: launch-time argument metadata for the alias guard; nothing to prepare on the host."""
    # cute_mma_dag: launch-time argument metadata for the alias guard
    # (``MmaDagAliasGuard``); nothing to prepare on the host.


def emit_split_k(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_tcgen05_split_k``: the partial/counter workspace pointer, a wrapper host scalar passed on after the ordinary arguments."""
    # tcgen05 split-K: the partial/counter workspace pointer, a wrapper
    # host scalar of ``_build_cute_schema_and_args`` passed on to the
    # kernel after the ordinary arguments (and the tensor maps).
    param = plan.get("param")
    if not isinstance(param, str):
        raise exc.BackendUnsupported("cute", "invalid tcgen05 split-K wrapper plan")
    ctx.call_args.append(param)


def emit_fixup(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_mma_dag_fixup``: the exchange workspace's pointer, a wrapper host scalar passed on after the ordinary arguments."""
    # cute_mma_dag fixup split: the exchange workspace's pointer, a
    # wrapper host scalar of ``_build_cute_schema_and_args`` passed on
    # to the kernel after the ordinary arguments (and the tensor maps).
    param = plan.get("param")
    if not isinstance(param, str):
        raise exc.BackendUnsupported("cute", "invalid mma_dag fixup wrapper plan")
    ctx.call_args.append(param)


def emit_tma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_mma_dag_tma``: one tiled TMA descriptor per operand, built over the wrapper's tensor views and passed as grid constants."""
    # cute_mma_dag tcgen05 mode: one tiled TMA descriptor per operand,
    # built here over the wrapper's tensor views and passed to the kernel
    # as grid constants after the ordinary arguments.  The launch keeps
    # the generic flat grid.
    count = ctx.plan_int("tma_count")
    if count <= 0 or count > 16:
        raise exc.BackendUnsupported("cute", "invalid mma_dag tensor-map count")
    tma_params: list[str] = []
    for index in range(count):
        param = plan.get(f"tma{index}_param")
        dims = plan.get(f"tma{index}_dims")
        box = plan.get(f"tma{index}_box")
        swizzle = plan.get(f"tma{index}_swizzle")
        tensor_idx = ctx.plan_int(f"tma{index}_idx")
        if (
            not isinstance(param, str)
            or not isinstance(dims, (list, tuple))
            or not isinstance(box, (list, tuple))
            or len(dims) != len(box)
            or not 1 <= len(dims) <= 5
            or not all(type(d) is int and d >= 0 for d in dims)
            or len(set(dims)) != len(dims)
            or not all(type(b) is int and 1 <= b <= 256 for b in box)
            or swizzle not in ("none", "s32b", "s64b", "s128b", "s128b_atom_32b")
        ):
            raise exc.BackendUnsupported(
                "cute", "invalid mma_dag tensor-map wrapper plan"
            )
        if ctx.schema_key is not None and (
            schema_pointer_alignment(ctx.schema_key[tensor_idx]) % 16
        ):
            raise exc.BackendUnsupported(
                "cute", "mma_dag tensor maps require 16-byte-aligned bases"
            )
        ctx.body.append(
            f"    {param} = _helion_mma_dag_tensor_map(arg{tensor_idx}, "
            f"{tuple(dims)!r}, {tuple(box)!r}, {swizzle!r})"
        )
        tma_params.append(param)
    ctx.call_args.extend(tma_params)


def emit_register_subblocks(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_register_subblocks``: ``warps`` consecutive work items per CTA."""
    # Register sub-block programs: ``warps`` consecutive work items
    # (batch row, channel tile, independent group) per CTA.
    ctx.body.extend(
        [
            f"    grid_x = cutlass.Int32({ctx.plan_int('total_ctas')})",
            "    grid_y = cutlass.Int32(1)",
            "    grid_z = cutlass.Int32(1)",
        ]
    )


def emit_warp_mma_gemm(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_warp_mma_gemm``: one CTA per (batch, m, n) output tile."""
    # Register-MMA GEMM tiles: one CTA per (batch, m, n) output tile,
    # plain tensor arguments, no host-side descriptors.
    ctx.body.extend(
        [
            f"    grid_x = cutlass.Int32({ctx.plan_int('total_tiles')})",
            "    grid_y = cutlass.Int32(1)",
            "    grid_z = cutlass.Int32(1)",
        ]
    )


def tma_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute._mma_dag_tcgen05_runtime import make_tensor_map

    return {"_helion_mma_dag_tensor_map": make_tensor_map}


def emit_paged_attention(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """Build raw-byte and packed-scale maps over the actual argument views."""
    params: list[str] = []
    indices: list[int] = []
    for index in range(4):
        param = plan.get(f"tma{index}_param")
        tensor_idx = ctx.plan_int(f"tma{index}_idx")
        if not isinstance(param, str):
            raise exc.BackendUnsupported(
                "cute", "invalid paged attention map parameter"
            )
        if ctx.schema_key is not None and (
            schema_pointer_alignment(ctx.schema_key[tensor_idx]) % 16
        ):
            raise exc.BackendUnsupported(
                "cute", "paged attention tensor maps require 16-byte-aligned bases"
            )
        params.append(param)
        indices.append(tensor_idx)
    for pair in (0, 2):
        raw, scale = (f"arg{indices[pair + offset]}" for offset in (0, 1))
        ctx.body.append(
            f"    {params[pair]}, {params[pair + 1]} = _helion_paged_maps("
            f"{raw}, {scale}, "
            f"tuple({raw}.shape[i] for i in (4, 2, 3, 1, 0)), "
            f"tuple({raw}.stride[i] for i in (4, 2, 3, 1, 0)), "
            f"tuple({scale}.shape[i] for i in (2, 3, 1, 0)), "
            f"tuple({scale}.stride[i] // 4 for i in (2, 3, 1, 0)), 128)"
        )
    ctx.call_args.extend(params)


def paged_attention_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute._paged_attention_runtime import make_paged_tma_operands

    return {"_helion_paged_maps": make_paged_tma_operands}


ROW_MMA_DEPENDENCIES = (
    "_compiler/cute/_flash_row_mma_runtime.py",
    "_compiler/cute/device/ptx.py",
)

EMITTERS = (
    HostWrapperEmitter("helion_mma_dag_alias", emit_alias),
    HostWrapperEmitter(
        "helion_paged_attention",
        emit_paged_attention,
        namespace=paged_attention_namespace,
        source_dependencies=(
            "_compiler/cute/_paged_attention_runtime.py",
            "_compiler/cute/_paged_attention_compute.py",
            "_compiler/cute/_paged_attention_pipeline.py",
            "_compiler/cute/_paged_attention_partial_compute.py",
            "_compiler/cute/_paged_attention_partial_pipeline.py",
            "_compiler/cute/_paged_attention_reference_compute.py",
            "_compiler/cute/_paged_attention_io.py",
            "_compiler/cute/_mma_dag_tcgen05_runtime.py",
            "_compiler/cute/_mma_dag_runtime.py",
            "_compiler/cute/_warp_mma_runtime.py",
            *ROW_MMA_DEPENDENCIES,
        ),
    ),
    HostWrapperEmitter("helion_tcgen05_split_k", emit_split_k),
    HostWrapperEmitter("helion_mma_dag_fixup", emit_fixup),
    HostWrapperEmitter(
        "helion_mma_dag_tma",
        emit_tma,
        namespace=tma_namespace,
        source_dependencies=(
            "_compiler/cute/_mma_dag_runtime.py",
            "_compiler/cute/_mma_dag_tcgen05_runtime.py",
            *ROW_MMA_DEPENDENCIES,
            "_compiler/cute/_warp_mma_runtime.py",
        ),
    ),
    HostWrapperEmitter(
        "helion_register_subblocks",
        emit_register_subblocks,
        source_dependencies=(
            "_compiler/cute/_warp_mma_runtime.py",
            "_compiler/cute/_mma_dag_runtime.py",
            *ROW_MMA_DEPENDENCIES,
        ),
    ),
    HostWrapperEmitter(
        "helion_warp_mma_gemm",
        emit_warp_mma_gemm,
        source_dependencies=(
            "_compiler/cute/_warp_mma_runtime.py",
            *ROW_MMA_DEPENDENCIES,
        ),
    ),
)
