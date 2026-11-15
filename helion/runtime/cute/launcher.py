"""CuTe DSL (cutlass) runtime: launcher, wrapper codegen, and launch caching.

Extracted from ``helion/runtime/__init__.py``; the cutedsl-specific launch path
lives here so the package root stays backend-neutral. ``helion.runtime`` re-exports
the public entry points (``default_cute_launcher``, ``cute_cuda_graph``).
``cutlass`` is imported lazily inside the functions that need it, so importing this
module never requires a CuTe/cutlass install.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import dataclass
from dataclasses import field
import hashlib
import importlib
import linecache
import logging
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
import weakref

import torch
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils.weak import WeakIdKeyDictionary

from ... import exc
from ..._compiler.cute.device_state import Tcgen05GroupedSchedulerMode
from ..._compiler.cute.dtypes import cutlass_pointer_dtype
from ..._compiler.cute.grouped_balanced_workers import balanced_workers_index_domain
from ..._compiler.cute.grouped_d_shift import bounded_d_shift_index_domain
from ..._compiler.cute.grouped_full_coverage import full_coverage_index_domain
from ..._compiler.cute.grouped_panel_raster import GROUPED_PANEL_ALTERNATE_KEY
from ..._compiler.cute.grouped_panel_raster import GROUPED_PANEL_RATIO_CHOICES
from ..._compiler.cute.grouped_panel_raster import GROUPED_PANEL_RATIO_KEY
from ..._compiler.cute.grouped_panel_raster import grouped_device_panel_index_domain
from ..._compiler.cute.grouped_panel_raster import grouped_panel_ratio_index_domain
from ..._compiler.cute.grouped_worklist import GroupedWorklistRows
from ..._compiler.cute.grouped_worklist import Tcgen05GroupedWorklistValidationError
from ..._compiler.cute.grouped_worklist import (
    _tcgen05_grouped_worklist_rows_from_flattened,
)
from ..._compiler.cute.grouped_worklist import (
    tcgen05_grouped_worklist_compatible_source_m_tiles,
)
from ..._compiler.cute.grouped_worklist import validate_tcgen05_grouped_worklist_rows
from ..._compiler.cute.strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL,
)
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS,
)
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_RUNTIME_TILE_FIELD_COUNT
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE,
)
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
)
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_STORE_SHAPE
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE,
)
from ..._compiler.cute.tcgen05_constants import Tcgen05GroupedRuntimeTileField
from ..triton.launcher import get_num_sm
from .arg_signature import ArgSignature
from .arg_signature import FastpathTensorGuard
from .arg_signature import ScalarArgFact
from .arg_signature import _CuteLastScalarArgGuard
from .arg_signature import _CuteLastTensorArgGuard
from .arg_signature import kernel_param_is_constexpr as _cute_kernel_param_is_constexpr
from .arg_signature import pointer_alignment as _cute_pointer_alignment
from .arg_signature import scalar_annotation as _cute_scalar_annotation
from .arg_signature import schema_pointer_alignment as _cute_schema_pointer_alignment
from .arg_signature import (
    schema_with_pointer_alignment as _cute_schema_with_pointer_alignment,
)
from .arg_signature import validate_launcher_tensor as _validate_cute_launcher_tensor
from .capture_pdl_launch import prepare_capture_launch
from .compile_cache import CompiledCuteLauncher as _CompiledCuteLauncher
from .compile_cache import disk_cache_key as _cute_disk_cache_key
from .compile_cache import function_globals
from .compile_cache import (
    merge_default_compile_options as _merge_default_compile_options,
)
from .compile_cache import (
    patch_cutlass_jit_shutdown_unload as _patch_cutlass_jit_shutdown_unload,
)
from .fast_relaunch import HOST_CELL_ADDRESS_FLOOR as _CUTE_HOST_CELL_ADDRESS_FLOOR
from .fast_relaunch import CuteFastRelaunch as _CuteFastRelaunch
from .fast_relaunch import PackedLaunch as _CutePackedLaunch
from .fast_relaunch import cxx_tensor_guards as _cute_cxx_tensor_guards
from .fast_relaunch import packed_launch as _cute_packed_launch
from .host_wrappers import HOST_WRAPPER_EMITTERS
from .host_wrappers import HostWrapperContext
from .host_wrappers import wrapped_grouped_descriptors
from .host_wrappers.grouped_static import grouped_static_active_clusters
from .host_wrappers.mma_dag import mma_dag_alias_guard as _mma_dag_alias_guard
from .host_wrappers.recurrences import build_chunk_prepare_tensor_maps
from .host_wrappers.recurrences import build_chunk_recurrence_tensor_maps
from .host_wrappers.recurrences import chunk_recurrence_tensor_map_specs
from .host_wrappers.recurrences import validate_gdn_recurrence_launch_args
from .launch_caches import (
    DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT as _TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT,
)
from .launch_caches import (
    GROUPED_STATIC_METADATA_CACHE_LIMIT as _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT,
)
from .launch_caches import LAUNCH_ARG_CACHE_LIMIT as _CUTE_LAUNCH_ARG_CACHE_LIMIT
from .launch_caches import (
    MMA_DAG_FIXUP_WORKSPACE_CACHE_LIMIT as _MMA_DAG_FIXUP_WORKSPACE_CACHE_LIMIT,
)
from .launch_caches import (
    SPLIT_K_WORKSPACE_CACHE_LIMIT as _TCGEN05_SPLIT_K_WORKSPACE_CACHE_LIMIT,
)
from .launch_caches import GroupedLaunchContext as _CuteGroupedLaunchContext
from .launch_caches import (
    GroupedStaticMetadataCacheEntry as _Tcgen05GroupedStaticMetadataCacheEntry,
)
from .launch_caches import (
    GroupedStaticMetadataResult as _Tcgen05GroupedStaticMetadataResult,
)
from .launch_caches import LaunchArgCacheEntry as _CuteLaunchArgCacheEntry
from .launch_caches import cached_exchange_workspace as _cached_exchange_workspace
from .launch_caches import cuda_stream_capture_context as _cuda_stream_capture_context
from .launch_caches import cute_cuda_graph as cute_cuda_graph
from .launch_caches import (
    track_cuda_graph_cache_entry as _track_cute_cuda_graph_cache_entry,
)
from .launch_descriptor import (
    STAGED_OPERAND_ALIGNMENT as _CUTE_STAGED_OPERAND_ALIGNMENT,
)
from .launch_descriptor import bake_tensor_shapes as _cute_bake_tensor_shapes_guard
from .launch_descriptor import cluster_shape as _cute_cluster_shape
from .launch_descriptor import cute_launch_descriptor
from .launch_descriptor import min_blocks_per_mp as _cute_min_blocks_per_mp
from .launch_descriptor import preferred_smem_carveout as _cute_preferred_smem_carveout
from .launch_descriptor import use_pdl as _cute_use_pdl
from .occupancy import resident_clusters as _cute_resident_clusters
from .plan_facts import plan_int_value as _plan_int_value
from .plan_facts import plan_str_value as _plan_str_value
from .plan_facts import (
    tcgen05_grouped_dynamic_ab_tensormap_rank as _tcgen05_grouped_dynamic_ab_tensormap_rank,
)
from .plan_facts import (
    tcgen05_grouped_scheduler_mode as _tcgen05_grouped_scheduler_mode,
)
from .plan_facts import tcgen05_plan_orientation as _tcgen05_plan_orientation

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable
    from collections.abc import Sequence


log: logging.Logger = logging.getLogger(__name__)

# The element type of a global-memory pointer to a torch dtype
# (``helion._compiler.cute.dtypes``); tests patch this name on the launcher.
_torch_dtype_to_cutlass = cutlass_pointer_dtype


def _append_cute_wrapper_plan(
    body: list[str],
    call_args: list[str],
    plan: dict[str, object],
    num_sm: int | None = None,
    schema_key: Sequence[tuple[object, ...]] | None = None,
    device: torch.device | None = None,
) -> None:
    """Append one wrapper plan's host setup to the ``@cute.jit`` wrapper body.

    Dispatches to the :class:`HostWrapperEmitter` registered for the plan's
    kind (``helion.runtime.cute.host_wrappers``).
    """
    context = HostWrapperContext(
        body=body,
        call_args=call_args,
        plan=plan,
        num_sm=num_sm,
        schema_key=schema_key,
        device=device,
        descriptors=wrapped_grouped_descriptors(plan),
    )
    kind = plan["kind"]
    emitter = HOST_WRAPPER_EMITTERS.get(cast("str", kind))
    if emitter is None:
        raise exc.BackendUnsupported("cute", f"wrapper plan kind: {kind}")
    emitter.emit(context, plan)


def _create_cute_wrapper(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    num_sm: int | None = None,
    device: torch.device | None = None,
) -> object:
    _patch_cutlass_jit_shutdown_unload()
    import cutlass
    import cutlass.cute as cute

    cuda_driver = importlib.import_module("cuda.bindings.driver")
    kernel_name = getattr(cast("Any", cute_kernel), "__name__", "cute_kernel")
    kernel_tag = f"{kernel_name}_{id(cute_kernel):x}"
    func_name = f"_helion_cute_launch_{kernel_tag}"
    params: list[str] = []
    body: list[str] = []
    call_args: list[str] = []

    for i, entry in enumerate(schema_key):
        kind = entry[0]
        if kind == "tensor":
            ptr_name = f"arg{i}_ptr"
            params.append(f"{ptr_name}: cute.Pointer")
            if len(entry) in (5, 6):
                # ("tensor", dtype, rank, sizes, strides[, alignment]) — baked
                # layout. An omitted alignment denotes the original 16-byte ABI.
                # Wrapper plans (matmul TMA) also reference
                # ``arg{i}_shape{d}`` / ``arg{i}_stride{d}`` names, so we
                # bind those names to their literal values in the wrapper
                # body before constructing the tensor.
                (_, _dtype, rank, sizes_t, strides_t) = entry[:5]
                assert isinstance(rank, int)
                assert isinstance(sizes_t, tuple) and len(sizes_t) == rank
                assert isinstance(strides_t, tuple) and len(strides_t) == rank
                shape_literals = [repr(int(s)) for s in sizes_t]
                stride_literals = [repr(int(s)) for s in strides_t]
                for d, lit in enumerate(shape_literals):
                    body.append(f"    arg{i}_shape{d} = {lit}")
                for d, lit in enumerate(stride_literals):
                    body.append(f"    arg{i}_stride{d} = {lit}")
                shape_tuple = (
                    f"({shape_literals[0]},)"
                    if rank == 1
                    else f"({', '.join(shape_literals)})"
                )
                stride_tuple = (
                    f"({stride_literals[0]},)"
                    if rank == 1
                    else f"({', '.join(stride_literals)})"
                )
                body.append(
                    f"    arg{i} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
                )
                call_args.append(f"arg{i}")
                continue
            (_, _dtype, rank) = entry[:3]
            assert isinstance(rank, int)
            shape_names = [f"arg{i}_shape{d}" for d in range(rank)]
            stride_names = [f"arg{i}_stride{d}" for d in range(rank)]
            params.extend(f"{name}: cutlass.Int64" for name in shape_names)
            params.extend(f"{name}: cutlass.Int64" for name in stride_names)
            shape_tuple = (
                f"({shape_names[0]},)" if rank == 1 else f"({', '.join(shape_names)})"
            )
            stride_tuple = (
                f"({stride_names[0]},)" if rank == 1 else f"({', '.join(stride_names)})"
            )
            body.append(
                f"    arg{i} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
            )
            call_args.append(f"arg{i}")
            continue

        if kind == "wrapper_tensor":
            (_, name, _dtype, rank, sizes_t, strides_t) = entry[:6]
            assert isinstance(name, str)
            assert isinstance(rank, int)
            assert isinstance(sizes_t, tuple) and len(sizes_t) == rank
            assert isinstance(strides_t, tuple) and len(strides_t) == rank
            ptr_name = f"{name}_ptr"
            params.append(f"{ptr_name}: cute.Pointer")
            shape_literals = [repr(int(s)) for s in sizes_t]
            stride_literals = [repr(int(s)) for s in strides_t]
            shape_tuple = (
                f"({shape_literals[0]},)"
                if rank == 1
                else f"({', '.join(shape_literals)})"
            )
            stride_tuple = (
                f"({stride_literals[0]},)"
                if rank == 1
                else f"({', '.join(stride_literals)})"
            )
            body.append(
                f"    {name} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
            )
            call_args.append(name)
            continue

        if kind == "wrapper_tensor_runtime_leading_extent":
            (_, name, _dtype, rank, tail_sizes, strides) = entry[:6]
            assert isinstance(name, str)
            assert isinstance(rank, int)
            assert isinstance(tail_sizes, tuple) and len(tail_sizes) == rank - 1
            assert isinstance(strides, tuple) and len(strides) == rank
            ptr_name = f"{name}_ptr"
            leading_extent_name = f"{name}_shape0"
            params.extend(
                (
                    f"{ptr_name}: cute.Pointer",
                    f"{leading_extent_name}: cutlass.Int64",
                )
            )
            shape_values = [
                leading_extent_name,
                *(repr(int(size)) for size in tail_sizes),
            ]
            stride_values = [repr(int(stride)) for stride in strides]
            shape_tuple = (
                f"({shape_values[0]},)" if rank == 1 else f"({', '.join(shape_values)})"
            )
            stride_tuple = (
                f"({stride_values[0]},)"
                if rank == 1
                else f"({', '.join(stride_values)})"
            )
            body.append(
                f"    {name} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
            )
            call_args.append(name)
            continue

        if kind == "wrapper_host_scalar":
            (_, name, scalar_kind) = entry
            assert isinstance(name, str)
            assert isinstance(scalar_kind, str)
            params.append(f"{name}: {_cute_scalar_annotation(scalar_kind)}")
            continue

        if kind == "scalar_constexpr":
            (_, scalar_kind, _scalar_key_value, scalar_value) = entry
            assert isinstance(scalar_kind, str)
            literal = repr(scalar_value)
            body.append(f"    arg{i} = {literal}")
            call_args.append(f"arg{i}")
            continue

        assert kind == "scalar"
        (_, scalar_kind) = entry
        assert isinstance(scalar_kind, str)
        scalar_name = f"arg{i}"
        params.append(f"{scalar_name}: {_cute_scalar_annotation(scalar_kind)}")
        call_args.append(scalar_name)

    params.extend(
        (
            "grid_x: cutlass.Int32",
            "grid_y: cutlass.Int32",
            "grid_z: cutlass.Int32",
            "stream: CUstream",
        )
    )
    wrapper_plans = [
        cast("dict[str, object]", plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    ]
    wrapped = [plan.get("wrapped_grouped_descriptors") for plan in wrapper_plans]
    if any(identity is not None for identity in wrapped) and (
        len(wrapper_plans) != 2
        or [plan.get("kind") for plan in wrapper_plans]
        not in (
            ["tcgen05_grouped_rna", "tcgen05_d_tma"],
            ["tcgen05_grouped_tma_rn", "tcgen05_d_tma"],
        )
        or wrapped[0] is None
        or wrapped[0] != wrapped[1]
    ):
        raise exc.BackendUnsupported(
            "cute", "wrapped input and output descriptors require one shared proof"
        )
    if any(
        plan.get("kind") == "tcgen05_grouped_tma_rn" for plan in wrapper_plans
    ) and block != (32, 8, 1):
        raise exc.BackendUnsupported(
            "cute", "TF32 TMA RN requires its proved 256-thread block"
        )
    for plan in wrapper_plans:
        _append_cute_wrapper_plan(
            body,
            call_args,
            plan,
            num_sm=num_sm,
            schema_key=schema_key,
            device=device,
        )
    gathered_plans = [
        plan for plan in wrapper_plans if plan.get("kind") == "gathered_mma_tma"
    ]
    if gathered_plans:
        if (
            len(wrapper_plans) != 1
            or tuple(cast("Sequence[int]", gathered_plans[0]["source_block"])) != block
        ):
            raise exc.BackendUnsupported(
                "cute", "gathered MMA requires its proved source launch"
            )
        block = (288, 1, 1)
    sm100_recurrence_plans = [
        plan for plan in wrapper_plans if plan.get("kind") == "chunk_recurrence_sm100"
    ]
    warp_dv4_recurrence_plans = [
        plan
        for plan in wrapper_plans
        if plan.get("kind") == "chunk_recurrence_warp_dv4"
    ]
    block_scaled_plans = [
        plan for plan in wrapper_plans if plan.get("kind") == "block_scaled_mma"
    ]
    gdn_recurrence_plans = [
        plan for plan in wrapper_plans if plan.get("kind") == "gdn_recurrence_sm100"
    ]
    if block_scaled_plans and (len(wrapper_plans) != 1 or block != (192, 1, 1)):
        raise exc.BackendUnsupported(
            "cute", "native block scaling must own the complete device root"
        )
    if len(gdn_recurrence_plans) > 1:
        raise exc.BackendUnsupported("cute", "multiple gdn recurrence plans")
    if gdn_recurrence_plans and len(wrapper_plans) != 1:
        raise exc.BackendUnsupported(
            "cute", "gdn recurrence must own the complete device root"
        )
    if len(sm100_recurrence_plans) > 1:
        raise exc.BackendUnsupported("cute", "multiple SM100 recurrence plans")
    if len(warp_dv4_recurrence_plans) > 1:
        raise exc.BackendUnsupported("cute", "multiple SM100 warp-DV4 plans")
    if (sm100_recurrence_plans or warp_dv4_recurrence_plans) and len(
        wrapper_plans
    ) != 1:
        raise exc.BackendUnsupported(
            "cute", "SM100 recurrence must own the complete device root"
        )
    launch_suffix = f", block={block!r}"
    cluster_shape = _cute_cluster_shape(cute_kernel, wrapper_plans)
    if cluster_shape is not None:
        launch_suffix += f", cluster={list(cluster_shape)!r}"
    preferred_smem_carveout = _cute_preferred_smem_carveout(cute_kernel, wrapper_plans)
    if preferred_smem_carveout is not None:
        launch_suffix += f", preferred_smem_carveout={preferred_smem_carveout}"
    # G2-H (cute_plan.md, see plan: G2-H CLC): CLC kernels need PDL
    # enabled at the host launch so ``nvvm.clusterlaunchcontrol_try_cancel``
    # returns valid responses. ``use_pdl`` is set on the per-matmul
    # wrapper plan in ``cute_mma._codegen_cute_mma`` when
    # ``Tcgen05PersistenceModel.CLC_PERSISTENT`` is active. Reading
    # from the plan rather than a kernel-level side-channel attribute
    # mirrors how ``cluster_m``/``cluster_n`` flow through this layer.
    if any(plan.get("use_pdl") for plan in wrapper_plans) or _cute_use_pdl(cute_kernel):
        launch_suffix += ", use_pdl=True"
    # An emitter's count, else one CTA per SM for the fa4 flash topology, whose
    # setmaxnreg reallocation ptxas honors only then (``min_blocks_per_mp``).
    min_blocks = _cute_min_blocks_per_mp(cute_kernel, wrapper_plans)
    if min_blocks is not None:
        launch_suffix += f", min_blocks_per_mp={min_blocks}"
    # A whole-root plan (block-scaled GEMM, the recurrence schedules) replaces
    # the generic launch with its host entry call; the checks above leave at
    # most one such plan.
    whole_root = [
        (plan, emitter)
        for plan in wrapper_plans
        if (emitter := HOST_WRAPPER_EMITTERS[cast("str", plan["kind"])]).host_call
        is not None
    ]
    if whole_root:
        plan, emitter = whole_root[0]
        assert emitter.host_call is not None
        emitter.host_call(body, plan)
    else:
        body.extend(
            (
                f"    _helion_cute_kernel_tag = {kernel_tag!r}",
                "    _kernel("
                + ", ".join(call_args)
                + f").launch(grid=(grid_x, grid_y, grid_z){launch_suffix}, stream=stream)",
            )
        )

    source = "\n".join(
        [
            "@cute.jit",
            f"def {func_name}({', '.join(params)}) -> None:",
            *body,
        ]
    )

    namespace: dict[str, Any] = {
        "cutlass": cutlass,
        "cute": cute,
        "CUstream": cuda_driver.CUstream,
        "_kernel": cute_kernel,
    }
    for plan in wrapper_plans:
        emitter = HOST_WRAPPER_EMITTERS[cast("str", plan["kind"])]
        if emitter.namespace is not None:
            namespace.update(emitter.namespace(plan))
    filename = f"<helion_cute_launcher:{kernel_tag}:{schema_key!r}:{block!r}>"
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    try:
        exec(compile(source, filename, "exec"), namespace)
    except BaseException:
        linecache.cache.pop(filename, None)
        raise
    jit_func = namespace[func_name]
    # The SDK's default mangler includes runtime values such as "1e+16",
    # which are not valid in its unquoted TVM FFI symbols. Generic Pointer
    # annotations omit dtype and alignment, so include the full schema along
    # with the source's kernel identity and launch geometry.
    symbol_key = repr((source, schema_key)).encode()
    jit_func.set_name_prefix(
        f"_helion_cute_launch_{hashlib.sha256(symbol_key).hexdigest()}",
        keep_mangled_name=False,
    )
    return jit_func


_CUTE_NUM_SM_CACHE: dict[tuple[str, int | None], int] = {}


def _cute_device_from_arch_args(
    arch_args: tuple[object, ...] | None,
) -> torch.device | None:
    """The CUDA device of the launch's first tensor argument (``None`` off GPU)."""
    if arch_args is None:
        return None
    for arg in arch_args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
            return arg.device
    return None


def _cute_num_sm_from_arch_args(arch_args: tuple[object, ...] | None) -> int | None:
    device = _cute_device_from_arch_args(arch_args)
    if device is None:
        return None
    cache_key = (device.type, device.index)
    cached = _CUTE_NUM_SM_CACHE.get(cache_key)
    if cached is None:
        cached = get_num_sm(device)
        _CUTE_NUM_SM_CACHE[cache_key] = cached
    return cached


def _cute_compiled_launcher_discriminator(
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    compile_options: str | None,
    arch_args: tuple[object, ...] | None,
    cluster_shape: object = None,
) -> tuple[tuple[object, ...], str, int | None]:
    merged_compile_options = _merge_default_compile_options(compile_options)
    num_sm = _cute_num_sm_from_arch_args(arch_args)
    resident_clusters = _cute_resident_clusters(
        cluster_shape, _cute_device_from_arch_args(arch_args)
    )
    return (
        (
            schema_key,
            block,
            merged_compile_options,
            num_sm,
            *(() if resident_clusters is None else (resident_clusters,)),
        ),
        merged_compile_options,
        num_sm,
    )


def _get_compiled_cute_launcher(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    compile_options: str | None = None,
    arch_args: tuple[object, ...] | None = None,
) -> object:
    # Always ensure ``--enable-tvm-ffi`` is present on the generic launcher
    # path: the generated wrapper signature (``cute.Pointer`` + scalars) is
    # TVM-FFI compatible and the FFI bridge bypasses CUTLASS-DSL's per-arg
    # cast/pointer work in ``generate_execution_args``. We merge rather
    # than replace because other flags (e.g. ``--generate-line-info`` when
    # ``tcgen05_cubin_lineinfo`` is True) can already be in
    # ``compile_options``.  The division option is merged the same way
    # (``merge_default_compile_options``): ``div.full.f32`` unless the
    # kernel's launch chose the IEEE division.
    cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    cache_key, compile_options, num_sm = _cute_compiled_launcher_discriminator(
        schema_key,
        block,
        compile_options,
        arch_args,
        cluster_shape,
    )
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_compiled_launchers
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_compiled_launchers = cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if arch_args is not None:
        _ensure_cute_dsl_arch_env(arch_args)
    wrapper_plans = tuple(
        repr(plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", ())
    )
    device = _cute_device_from_arch_args(arch_args)
    jit_func = _create_cute_wrapper(
        cute_kernel,
        schema_key,
        block,
        num_sm=num_sm,
        device=device,
    )
    disk_cache_key = _cute_disk_cache_key(
        cute_kernel,
        schema_key,
        block,
        wrapper_plans,
        cluster_shape,
        compile_options,
        num_sm,
        resident_clusters=_cute_resident_clusters(cluster_shape, device),
        wrapper_globals=function_globals(jit_func),
    )
    launcher = _CompiledCuteLauncher(
        jit_func,
        compile_options,
        cache_key=disk_cache_key,
        sass_checks=tuple(
            getattr(cast("Any", cute_kernel), "_helion_cute_sass_checks", ())
        ),
        kernel_name=getattr(cast("Any", cute_kernel), "__name__", "cute_kernel"),
    )
    cache[cache_key] = launcher
    return launcher


_CUTE_LAUNCHER_IMPORTS: tuple[object, ...] | None = None


def _get_cute_launcher_imports() -> tuple[object, ...]:
    global _CUTE_LAUNCHER_IMPORTS
    cached = _CUTE_LAUNCHER_IMPORTS
    if cached is not None:
        return cached
    _patch_cutlass_jit_shutdown_unload()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    import cutlass.torch as cutlass_torch

    cached = (cute.AddressSpace.gmem, make_ptr, cutlass_torch.current_stream)
    _CUTE_LAUNCHER_IMPORTS = cached
    return cached


def _cute_current_stream() -> object:
    """Sample the *current* CUDA stream for a cute kernel launch.

    Must be called fresh on every launch and never cached: under CUDA graph
    capture ``torch.cuda.current_stream()`` is redirected to a dedicated capture
    stream, so a stream baked into the cached launch args (during eager warmup)
    would make the kernel launch on the wrong, non-capturing stream — the graph
    then records no work and replays as a no-op (empty-graph capture). Sampling
    here keeps the launch on whatever stream is current at call time.
    """
    _gmem, _make_ptr, current_stream_obj = _get_cute_launcher_imports()
    return cast("Any", current_stream_obj)()


# ``standalone_launcher._standalone_ensure_cuda_context``, imported on the first
# launch after torch initializes CUDA: that module loads cutlass and
# ``cuda.bindings``, which a CPU-only process may not have.
_STANDALONE_ENSURE_CUDA_CONTEXT: Callable[[], None] | None = None


def _ensure_cuda_context() -> None:
    """Make the current device's primary CUDA context current on this thread,
    before every launch (the fast relaunch included).

    Without a current context the compiled host code of a launch that sets
    function attributes (``cudaFuncSetAttribute``, ``cuKernelGetAttribute``)
    aborts the process (the CuTe DSL runtime's ``cuda_dialect_assert_success``:
    "invalid device context"), and a thread can lack one at any launch.  A new
    thread starts without one until a runtime-API call initializes it there,
    PyTorch's autograd device threads among them (c10 skips ``cudaSetDevice``
    for the device a thread already defaults to), and other code can clear
    the context, or pop one it pushed on a thread that had none, after one was
    made current.  The check is the exported launcher's
    (``standalone_launcher._standalone_ensure_cuda_context``, one
    ``cuCtxGetCurrent`` call when a context is current, which keeps that
    module Helion-free), imported and run once torch has initialized CUDA.
    """
    global _STANDALONE_ENSURE_CUDA_CONTEXT
    check = _STANDALONE_ENSURE_CUDA_CONTEXT
    if check is None:
        if not torch.cuda.is_initialized():
            # No CUDA work in this process yet (a stub launch in a CPU-only
            # test): nothing to make current, and libcuda may not even be
            # installed.
            return
        from .standalone_launcher import _standalone_ensure_cuda_context as check

        _STANDALONE_ENSURE_CUDA_CONTEXT = check
    check()


@dataclass(frozen=True)
class _CuteLastLaunchCacheEntry:
    arg_guard: _CuteLastLaunchArgGuard
    compiled_discriminator: tuple[object, ...]
    launch: _CuteLaunchArgCacheEntry
    compiled: object


@dataclass(frozen=True)
class _CuteLastGroupedMutationGuard:
    index: int
    tensor_id: int
    mutation_key: tuple[object, ...]

    def matches(self, args: tuple[object, ...]) -> bool:
        if self.index >= len(args) or not isinstance(args[self.index], torch.Tensor):
            return False
        tensor = args[self.index]
        assert isinstance(tensor, torch.Tensor)
        return (
            id(tensor) == self.tensor_id
            and _tcgen05_grouped_tensor_mutation_key(tensor) == self.mutation_key
        )


@dataclass(frozen=True)
class _CuteLastLaunchArgGuard:
    arg_count: int
    grid: tuple[int, int, int]
    bake_tensor_shapes: bool
    arg_guards: tuple[_CuteLastTensorArgGuard | _CuteLastScalarArgGuard, ...]
    grouped_mutation_guards: tuple[_CuteLastGroupedMutationGuard, ...]
    grouped_launch_contexts: tuple[_CuteGroupedLaunchContext, ...]

    def matches(
        self,
        cute_kernel: object,
        args: tuple[object, ...],
        grid: tuple[int, int, int],
    ) -> bool:
        if (
            len(args) != self.arg_count
            or grid != self.grid
            or _cute_bake_tensor_shapes_guard(cute_kernel) != self.bake_tensor_shapes
            or _cute_grouped_launch_contexts(cute_kernel, args)
            != self.grouped_launch_contexts
        ):
            return False
        constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
        for guard in self.arg_guards:
            if isinstance(guard, _CuteLastTensorArgGuard):
                if not guard.matches(args):
                    return False
            elif not guard.matches(args, constexpr_flags):
                return False
        return all(guard.matches(args) for guard in self.grouped_mutation_guards)


def _cute_staged_plan_operands(
    cute_kernel: object,
) -> tuple[tuple[int, bool], ...]:
    """The (argument index, is_output) operands the kernel's plans stage."""
    return cute_launch_descriptor(cute_kernel).staged_operands


def _cute_stage_under_aligned_operands(
    cute_kernel: object, args: tuple[object, ...]
) -> tuple[tuple[object, ...], list[tuple[torch.Tensor, torch.Tensor]]] | None:
    """Replace under-aligned staged operands by 16-byte-aligned copies.

    Returns the launch arguments with each such operand cloned (a fresh
    allocation is at least 16-byte aligned) and the (destination, copy) pairs
    to write back after the launch, or None when every operand is aligned.
    """
    staged: list[object] | None = None
    copy_backs: list[tuple[torch.Tensor, torch.Tensor]] = []
    descriptor = cute_launch_descriptor(cute_kernel)
    for index, is_output in descriptor.staged_operands:
        tensor = args[index]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.data_ptr() % _CUTE_STAGED_OPERAND_ALIGNMENT == 0
        ):
            continue
        if staged is None:
            staged = list(args)
        if index in descriptor.staged_strided_copies:
            copy = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            copy.copy_(tensor)
        else:
            copy = tensor.clone()
        staged[index] = copy
        if is_output:
            copy_backs.append((tensor, copy))
    if staged is None:
        return None
    return tuple(staged), copy_backs


def _validate_tcgen05_grouped_tensor_devices(
    layout: torch.Tensor,
    args: tuple[object, ...],
) -> None:
    mismatched_indices = [
        index
        for index, arg in enumerate(args)
        if isinstance(arg, torch.Tensor) and arg.device != layout.device
    ]
    if mismatched_indices:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped scheduler requires every tensor argument to be on "
            f"{layout.device}; mismatched argument indices: {mismatched_indices}",
        )


def _tcgen05_grouped_static_plans(cute_kernel: object) -> list[dict[str, object]]:
    return list(cute_launch_descriptor(cute_kernel).grouped_static_plans)


def _tcgen05_split_k_plan_present(cute_kernel: object) -> bool:
    return cute_launch_descriptor(cute_kernel).split_k_plan is not None


def _tcgen05_grouped_device_split_sizes(plan: dict[str, object]) -> bool:
    return bool(plan.get("device_split_sizes"))


def _tcgen05_grouped_device_layout_kind(
    plan: dict[str, object],
) -> str:
    kind = plan.get("device_layout_kind", "split_sizes")
    if kind not in ("split_sizes", "offsets"):
        raise exc.BackendUnsupported(
            "cute",
            f"unknown tcgen05 grouped device layout kind: {kind!r}",
        )
    assert isinstance(kind, str)
    return kind


def _tcgen05_grouped_host_metadata_plans(
    cute_kernel: object,
) -> list[dict[str, object]]:
    return [
        plan
        for plan in _tcgen05_grouped_static_plans(cute_kernel)
        if not _tcgen05_grouped_device_split_sizes(plan)
    ]


def _cute_grouped_static_metadata_matches(
    grouped_static_metadata: tuple[_Tcgen05GroupedStaticMetadataCacheEntry, ...],
    cute_kernel: object,
    args: tuple[object, ...],
) -> bool:
    plans = _tcgen05_grouped_host_metadata_plans(cute_kernel)
    if len(grouped_static_metadata) != len(plans):
        return False
    for plan, entry in zip(plans, grouped_static_metadata, strict=True):
        layout = _tcgen05_grouped_static_layout_arg(plan, args)
        n_sizes_arg = _tcgen05_grouped_static_size_arg(plan, args, "n_sizes")
        k_sizes_arg = _tcgen05_grouped_static_size_arg(plan, args, "k_sizes")
        if not entry.matches(layout, n_sizes_arg, k_sizes_arg):
            return False
        expected_has_m_tail = plan.get("grouped_static_has_m_tail")
        if isinstance(expected_has_m_tail, bool) and (
            entry.has_m_tail != expected_has_m_tail
        ):
            return False
        expected_has_n_tail = plan.get("grouped_static_has_n_tail")
        if isinstance(expected_has_n_tail, bool) and (
            entry.has_n_tail != expected_has_n_tail
        ):
            return False
    return True


def _tcgen05_split_k_workspace(
    cute_kernel: object,
    *,
    device: torch.device,
    tiles: int,
    splits: int,
    tile_elems: int,
) -> torch.Tensor:
    """The exchange workspace of a ``tcgen05_split_k`` GEMM: per output tile
    the ``splits`` fp32 partial slabs of ``tile_elems`` elements, then one
    u32 arrival counter per tile. The counters are zeroed once (the slabs
    are written before they are read) and the workspace is kept per
    (device, geometry, stream, capture): the counters only ever grow (a
    launch's CTAs derive their base from their ticket's period), so no
    launch resets them, and the cached pointer stays valid for a captured
    graph."""
    partial_bytes = tiles * splits * tile_elems * 4
    counter_bytes = -(-tiles * 4 // 256) * 256
    stream_handle, capture_id = _cuda_stream_capture_context(device)
    return _cached_exchange_workspace(
        cute_kernel,
        "_helion_tcgen05_split_k_workspace_cache",
        (
            device.type,
            device.index,
            tiles,
            splits,
            tile_elems,
            stream_handle,
            capture_id,
        ),
        device=device,
        nbytes=partial_bytes + counter_bytes,
        zero_from=partial_bytes,
        capture_id=capture_id,
        limit=_TCGEN05_SPLIT_K_WORKSPACE_CACHE_LIMIT,
    )


def _mma_dag_fixup_workspace(
    cute_kernel: object,
    *,
    device: torch.device,
    row_tiles: int,
    tiles: int,
    rounds: int,
    slabs: int,
    rows: int,
) -> torch.Tensor:
    """The exchange workspace of a ``cute_mma_dag`` fixup split: per partial
    slab and row tile the ``tiles x rows`` fp32 partials, then one u32
    arrival counter per row tile (``rounds`` arrivals per launch).  Zeroed once and kept per (device, geometry,
    stream, capture): the counters only ever grow (a launch's CTAs derive
    their base from their first ticket), so no launch resets them, and the
    cached pointer stays valid for a captured graph."""
    partial_bytes = slabs * tiles * row_tiles * rows * 4
    counter_bytes = -(-row_tiles * 4 // 256) * 256
    stream_handle, capture_id = _cuda_stream_capture_context(device)
    return _cached_exchange_workspace(
        cute_kernel,
        "_helion_mma_dag_fixup_workspace_cache",
        (
            device.type,
            device.index,
            row_tiles,
            tiles,
            rounds,
            slabs,
            rows,
            stream_handle,
            capture_id,
        ),
        device=device,
        nbytes=partial_bytes + counter_bytes,
        zero_from=0,
        capture_id=capture_id,
        limit=_MMA_DAG_FIXUP_WORKSPACE_CACHE_LIMIT,
    )


def _tcgen05_grouped_dynamic_tensormap_workspace(
    cute_kernel: object,
    *,
    device: torch.device,
    tensormap_count: int,
    num_sm_multiplier: int = 1,
) -> torch.Tensor:
    num_sm = get_num_sm(device)
    workspace_rows = grouped_static_active_clusters(
        num_sm=num_sm,
        cluster_m=1,
        reserved_sms=0,
        num_sm_multiplier=num_sm_multiplier,
    )
    stream_handle, capture_id = _cuda_stream_capture_context(device)
    cache_key = (
        device.type,
        device.index,
        workspace_rows,
        tensormap_count,
        stream_handle,
        capture_id,
    )
    # A captured graph embeds this raw pointer and can outlive the launch LRU.
    # cute_cuda_graph ties tracked entries to graph lifetime; raw captures stay
    # conservatively retained by this cache because their graph is unobservable.
    try:
        cache = cast(
            "OrderedDict[tuple[object, ...], torch.Tensor]",
            cast("Any", cute_kernel)._helion_tcgen05_dynamic_tensormap_workspace_cache,
        )
    except AttributeError:
        cache = OrderedDict()
        cast(
            "Any", cute_kernel
        )._helion_tcgen05_dynamic_tensormap_workspace_cache = cache
    workspace = cache.get(cache_key)
    if workspace is not None:
        return workspace
    workspace = torch.empty(
        (workspace_rows, tensormap_count, 128 // 8),
        dtype=torch.int64,
        device=device,
    )
    cache[cache_key] = workspace
    if capture_id is not None:
        _track_cute_cuda_graph_cache_entry(
            cute_kernel,
            "_helion_tcgen05_dynamic_tensormap_workspace_cache",
            cache_key,
            workspace,
            (workspace,),
        )
    else:
        eager_keys = [key for key in cache if key[-1] is None]
        while len(eager_keys) > _TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT:
            cache.pop(eager_keys.pop(0))
    return workspace


def _tcgen05_grouped_static_layout_arg(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> torch.Tensor:
    layout_idx = _plan_int_value(plan, "layout_idx")
    if layout_idx >= len(args) or not isinstance(args[layout_idx], torch.Tensor):
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped scheduler layout argument is not a tensor"
        )
    layout = args[layout_idx]
    assert isinstance(layout, torch.Tensor)
    if layout.device.type != "cuda":
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped scheduler layout must be a CUDA tensor"
        )
    worklist_metadata = bool(plan.get("worklist_metadata"))
    if _tcgen05_grouped_device_split_sizes(plan):
        if layout.ndim != 1:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped compact device layout must be rank 1",
            )
    elif worklist_metadata:
        if layout.ndim != 2 or layout.size(1) != 4:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped worklist scheduler metadata must have shape [W, 4]",
            )
    elif layout.ndim != 1:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped scheduler layout must be rank 1"
        )
    if layout.dtype not in (torch.int32, torch.int64):
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped scheduler layout must be int32 or int64"
        )
    return layout


def _tcgen05_grouped_device_source_m_tile_supported(plan: dict[str, object]) -> bool:
    """Admit compact ONE profiles and retain the proved source64 contract."""
    source_m_tile = plan.get("source_m_tile")
    if plan.get("orientation", "mn") == "mn":
        return (
            plan.get("kind") == "tcgen05_grouped_static_persistent"
            and plan.get("scheduler_mode") == "device_group_search"
            and source_m_tile in (128, 256)
            and plan.get("bm") == source_m_tile
            and plan.get("bn") in (128, 192, 224, 256)
            and plan.get("cluster_m") == 2
            and plan.get("cluster_n") in (1, 2)
            and plan.get("device_split_sizes") is True
            and plan.get("device_layout_kind") in ("offsets", "split_sizes")
        )
    if (
        type(source_m_tile) is int
        and source_m_tile == TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE
    ):
        cluster_m = plan.get("cluster_m")
        return (
            type(cluster_m) is int
            and cluster_m in (1, 2)
            and plan.get("kind") == "tcgen05_grouped_static_persistent"
            and plan.get("scheduler_mode") == "device_group_search"
            and plan.get("orientation") == "nm"
            and plan.get("bm") == 128 * cluster_m
            and plan.get("bn") == source_m_tile
            and plan.get("cluster_n") == 1
            and plan.get("device_split_sizes") is True
            and plan.get("device_layout_kind") in ("offsets", "split_sizes")
        )
    if type(source_m_tile) is int and source_m_tile in (8, 16):
        return (
            plan.get("kind") == "tcgen05_grouped_static_persistent"
            and plan.get("scheduler_mode") == "device_group_search"
            and plan.get("orientation") == "nm"
            and plan.get("bm") == 128
            and plan.get("bn") == source_m_tile
            and plan.get("cluster_m") == 1
            and plan.get("cluster_n") == 1
        )
    if source_m_tile in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES:
        return True
    if (
        type(source_m_tile) is not int
        or source_m_tile != TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
    ):
        return False
    profile = plan.get("source64_profile")
    expected_profile = {
        "full_coverage_mode": TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL,
        "consumer_local": True,
        "use_2cta_instrs": False,
        "ab_stage_count": 4,
        "acc_stage_count": 2,
        "c_stage_count": 2,
        "consumer_regs": 256,
        "input_dtype": "cutlass.BFloat16",
        "acc_dtype": "cutlass.Float32",
        "offset_dtype": "torch.int32",
    }
    if type(profile) is not dict or profile.keys() != expected_profile.keys():
        return False
    if any(
        type(profile[key]) is not type(value) or profile[key] != value
        for key, value in expected_profile.items()
    ):
        return False
    required_plan = {
        "kind": "tcgen05_grouped_static_persistent",
        "scheduler_mode": "device_group_search",
        "bm": 128,
        "bn": 64,
        "bk": 128,
        "cluster_m": 1,
        "cluster_n": 1,
        "orientation": "nm",
        "shared_rhs": True,
        "worklist_metadata": True,
        "device_split_sizes": True,
        "device_layout_kind": "offsets",
        "dynamic_ab_tensormaps": True,
        "dynamic_ab_tensormap_rank": 2,
        "dynamic_d_tensormap": True,
    }
    if any(
        type(plan.get(key)) is not type(value) or plan.get(key) != value
        for key, value in required_plan.items()
    ):
        return False
    if any(
        plan.get(key, False) is not False
        for key in (
            "fixed_tensormaps",
            "direct_pointer_metadata",
            "external_direct_pointer_metadata",
            "use_2cta_instrs",
        )
    ):
        return False
    return full_coverage_index_domain(
        cast("int", plan.get("group_count")),
        cast("int", plan.get("m_size")),
        cast("int", plan.get("n_size")),
        cast("int", plan.get("k_total_size")),
        source_m_tile,
        128,
        128,
    )


def _validate_tcgen05_grouped_device_split_sizes(
    plan: dict[str, object],
    split_sizes: torch.Tensor,
) -> None:
    group_count = _plan_int_value(plan, "group_count")
    layout_kind = _tcgen05_grouped_device_layout_kind(plan)
    expected_values = group_count + (layout_kind == "offsets")
    if int(split_sizes.numel()) != expected_values:
        expected_shape = "[G + 1]" if layout_kind == "offsets" else "[G]"
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped device {layout_kind} must have shape {expected_shape}",
        )
    if _tcgen05_plan_orientation(
        plan
    ) != "nm" and not _tcgen05_grouped_device_source_m_tile_supported(plan):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires N,M orientation",
        )
    bk = _plan_int_value(plan, "bk")
    source_m_tile = _plan_int_value(plan, "source_m_tile")
    m_size = _plan_int_value(plan, "m_size")
    n_size = _plan_int_value(plan, "n_size")
    k_total_size = _plan_int_value(plan, "k_total_size")
    int32_max = torch.iinfo(torch.int32).max
    if any(
        extent <= 0 or extent > int32_max
        for extent in (group_count, m_size, n_size, k_total_size)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires G, M, N, and K "
            "dimensions to be positive signed Int32 values",
        )
    if (
        bk not in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
        or not _tcgen05_grouped_device_source_m_tile_supported(plan)
        or n_size % TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[2] != 0
        or k_total_size % bk != 0
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires block_k 64 or 128, "
            "a validated source M tile, output N divisible by 32, and K "
            "divisible by the CTA K tile",
        )
    if (
        source_m_tile == TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
        and split_sizes.dtype != torch.int32
    ):
        raise exc.BackendUnsupported(
            "cute", "source64 device offsets require the proved Int32 layout dtype"
        )
    if (
        not bool(plan.get("dynamic_ab_tensormaps"))
        or _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) != 2
        or not (
            bool(plan.get("dynamic_d_tensormap"))
            or plan.get("bounded_output_shift") is True
        )
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires rank-2 dynamic A/B "
            "TensorMaps and a dynamic D TensorMap",
        )
    if plan.get("bounded_output_shift") is True and not (
        plan.get("immutable_ab_tensormaps") is True
        and not plan.get("dynamic_d_tensormap")
        and not plan.get("fixed_tensormaps")
        and not plan.get("direct_pointer_metadata")
        and _tcgen05_grouped_scheduler_mode(plan)
        is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
        and bounded_d_shift_index_domain(m_size, n_size, source_m_tile)
    ):
        raise exc.BackendUnsupported(
            "cute", "bounded output shift requires immutable pure device-worklist maps"
        )
    if plan.get("balanced_workers") is True and not (
        _tcgen05_plan_orientation(plan) == "mn"
        and plan.get("cluster_m") == 2
        and plan.get("cluster_n", 1) in (1, 2)
        and (
            plan.get("cluster_n", 1) == 1
            or n_size
            % (_plan_int_value(plan, "bn") * _plan_int_value(plan, "cluster_n"))
            == 0
        )
        and plan.get("num_sm_multiplier", 1) == 1
        and not plan.get("fixed_tensormaps")
        and not plan.get("direct_pointer_metadata")
        and _tcgen05_grouped_scheduler_mode(plan)
        is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
        and balanced_workers_index_domain(
            group_count,
            m_size,
            n_size,
            _plan_int_value(plan, "bm"),
            _plan_int_value(plan, "bn") * _plan_int_value(plan, "cluster_n"),
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "balanced output workers require the proved device M,N profile"
        )
    if (
        plan.get("l2_swizzle_size", 1) != 1
        or plan.get("device_panel_swizzle") is True
        or plan.get(GROUPED_PANEL_ALTERNATE_KEY) is not None
        or plan.get(GROUPED_PANEL_RATIO_KEY) is not None
    ):
        bm, bn = _plan_int_value(plan, "bm"), _plan_int_value(plan, "bn")
        cluster_n = _plan_int_value(plan, "cluster_n")
        if not (
            plan.get("device_panel_swizzle") is True
            and (
                _plan_int_value(plan, "l2_swizzle_size") > 1
                or plan.get(GROUPED_PANEL_ALTERNATE_KEY) is not None
            )
            and _tcgen05_plan_orientation(plan) == "mn"
            and plan.get("device_layout_kind") == "offsets"
            and _tcgen05_grouped_scheduler_mode(plan)
            is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
            and (cluster_n == 1 or n_size % (bn * cluster_n) == 0)
            and grouped_device_panel_index_domain(
                group_count, m_size, n_size, bm, bn, cluster_n
            )
        ):
            raise exc.BackendUnsupported(
                "cute",
                "device panel raster requires the bounded complete-cluster MN plan",
            )
        if (
            plan.get(GROUPED_PANEL_ALTERNATE_KEY) is not None
            or plan.get(GROUPED_PANEL_RATIO_KEY) is not None
        ):
            alternate = plan.get(GROUPED_PANEL_ALTERNATE_KEY)
            ratio = plan.get(GROUPED_PANEL_RATIO_KEY)
            if not (
                type(alternate) is int
                and alternate in TCGEN05_LEGAL_L2_SWIZZLE_SIZES
                and type(ratio) is int
                and ratio in GROUPED_PANEL_RATIO_CHOICES
                and grouped_panel_ratio_index_domain(n_size, bn, cluster_n, ratio)
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "adaptive panel requires a legal alternate width and bounded positive integer ratio",
                )


def _tcgen05_grouped_device_split_total_clusters(
    plan: dict[str, object],
) -> int:
    """Return a value-independent bound for device split-size metadata.

    The kernel clips every raw group interval to the packed ``[0, M)`` extent,
    so any one group can expose at most ``ceil(M / source_m_tile)`` row tiles,
    regardless of split signs, sums, or overlaps.  Multiply that bound by G
    and by the output-column cluster count.  The wrapper caps the physical grid
    at the active-cluster limit, so this conservative logical bound does not
    increase resident clusters and requires no host read of split values.
    """
    group_count = _plan_int_value(plan, "group_count")
    m_size = _plan_int_value(plan, "m_size")
    n_size = _plan_int_value(plan, "n_size")
    source_m_tile = _plan_int_value(plan, "source_m_tile")
    physical_mma_m = _plan_int_value(plan, "bm")
    if (
        group_count <= 0
        or m_size <= 0
        or n_size <= 0
        or physical_mma_m <= 0
        or not _tcgen05_grouped_device_source_m_tile_supported(plan)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires positive G, M, and N "
            "and a validated source M tile",
        )
    physical_columns = (
        _plan_int_value(plan, "bn") * _plan_int_value(plan, "cluster_n")
        if _tcgen05_plan_orientation(plan) == "mn"
        else physical_mma_m
    )
    n_clusters = (n_size + physical_columns - 1) // physical_columns
    packed_m_clusters = (m_size + source_m_tile - 1) // source_m_tile
    total_clusters = n_clusters * group_count * packed_m_clusters
    if total_clusters > torch.iinfo(torch.int32).max:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes cluster bound must fit signed Int32",
        )
    return total_clusters


def _cute_dynamic_tensormap_contexts(
    cute_kernel: object,
    args: tuple[object, ...],
) -> tuple[_CuteGroupedLaunchContext, ...]:
    """Return stream/capture contexts that isolate mutable TensorMaps."""
    contexts: list[_CuteGroupedLaunchContext] = []
    for plan in _tcgen05_grouped_static_plans(cute_kernel):
        if not (
            (
                bool(plan.get("dynamic_ab_tensormaps"))
                and not bool(plan.get("immutable_ab_tensormaps"))
            )
            or bool(plan.get("dynamic_d_tensormap"))
        ):
            continue
        layout = _tcgen05_grouped_static_layout_arg(plan, args)
        stream_handle, capture_id = _cuda_stream_capture_context(layout.device)
        contexts.append(
            (layout.device.type, layout.device.index, stream_handle, capture_id)
        )
    return tuple(contexts)


def _cute_grouped_launch_contexts(
    cute_kernel: object,
    args: tuple[object, ...],
    *,
    dynamic_tensormap_contexts: tuple[_CuteGroupedLaunchContext, ...] | None = None,
) -> tuple[_CuteGroupedLaunchContext, ...]:
    """Return contexts that isolate mutable or capture-owned launch tensors."""
    if dynamic_tensormap_contexts is None:
        dynamic_tensormap_contexts = _cute_dynamic_tensormap_contexts(cute_kernel, args)
    contexts = list(dynamic_tensormap_contexts)
    contexts_by_device = {
        (context[0], context[1]): context for context in dynamic_tensormap_contexts
    }
    for plan in _tcgen05_grouped_static_plans(cute_kernel):
        if plan.get("dynamic_ab_tensormaps") or plan.get("dynamic_d_tensormap"):
            continue
        layout = _tcgen05_grouped_static_layout_arg(plan, args)
        device_key = (layout.device.type, layout.device.index)
        context = contexts_by_device.get(device_key)
        if context is None:
            stream_handle, capture_id = _cuda_stream_capture_context(layout.device)
            context = (*device_key, stream_handle, capture_id)
            contexts_by_device[device_key] = context
        if context[3] is None:
            # Static grouped metadata is read-only across eager streams, so
            # keep the existing bounded shared cache on the eager hot path.
            continue
        if context not in contexts:
            contexts.append(context)
    if _tcgen05_split_k_plan_present(cute_kernel):
        # The split-K workspace is cached per (stream, capture)
        # (``_tcgen05_split_k_workspace``): the same pointers relaunched on
        # another stream or under capture must miss the last-launch cache
        # too, or they would reuse the earlier launch's workspace pointer.
        device = next(
            (arg.device for arg in args if isinstance(arg, torch.Tensor)), None
        )
        if device is not None:
            context = (
                device.type,
                device.index,
                *_cuda_stream_capture_context(device),
            )
            if context not in contexts:
                contexts.append(context)
    return tuple(contexts)


def _tcgen05_grouped_static_size_arg(
    plan: dict[str, object],
    args: tuple[object, ...],
    name: Literal["n_sizes", "k_sizes"],
) -> torch.Tensor | None:
    index = plan.get(f"{name}_idx")
    if index is None:
        return None
    assert isinstance(index, int)
    if index >= len(args) or not isinstance(args[index], torch.Tensor):
        raise exc.BackendUnsupported(
            "cute", f"tcgen05 grouped scheduler {name} argument is not a tensor"
        )
    tensor = args[index]
    assert isinstance(tensor, torch.Tensor)
    if tensor.device.type != "cuda":
        raise exc.BackendUnsupported(
            "cute", f"tcgen05 grouped scheduler {name} must be a CUDA tensor"
        )
    if tensor.ndim != 1:
        raise exc.BackendUnsupported(
            "cute", f"tcgen05 grouped scheduler {name} must be rank 1"
        )
    if tensor.dtype not in (torch.int32, torch.int64):
        raise exc.BackendUnsupported(
            "cute", f"tcgen05 grouped scheduler {name} must be int32 or int64"
        )
    return tensor


def _tcgen05_grouped_dynamic_ab_tensor_arg(
    plan: dict[str, object],
    args: tuple[object, ...],
    key: str,
    operand: str,
) -> torch.Tensor:
    idx = plan.get(key)
    if (
        not isinstance(idx, int)
        or idx >= len(args)
        or not isinstance(args[idx], torch.Tensor)
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped dynamic A/B TensorMaps require tensor operand {operand}",
        )
    tensor = args[idx]
    assert isinstance(tensor, torch.Tensor)
    if tensor.device.type != "cuda":
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped dynamic A/B TensorMap operand {operand} must be CUDA",
        )
    return tensor


def _tcgen05_grouped_external_direct_tensor_arg(
    plan: dict[str, object],
    args: tuple[object, ...],
    *,
    key: str,
    tensor_name: str,
    dtype: torch.dtype,
    ndim: int,
) -> torch.Tensor:
    idx = plan.get(key)
    if (
        not isinstance(idx, int)
        or idx >= len(args)
        or not isinstance(args[idx], torch.Tensor)
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped external direct metadata requires {tensor_name} tensor",
        )
    tensor = args[idx]
    assert isinstance(tensor, torch.Tensor)
    if tensor.device.type != "cuda":
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped external direct metadata {tensor_name} must be CUDA",
        )
    if tensor.dtype != dtype:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped external direct metadata {tensor_name} dtype must be "
            f"{dtype}",
        )
    if tensor.ndim != ndim:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped external direct metadata {tensor_name} must be "
            f"rank {ndim}",
        )
    return tensor


def _tcgen05_grouped_external_direct_metadata_args(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not bool(plan.get("external_direct_pointer_metadata")):
        return None
    pointers = _tcgen05_grouped_external_direct_tensor_arg(
        plan,
        args,
        key="direct_pointers_idx",
        tensor_name="direct_pointers",
        dtype=torch.int64,
        ndim=2,
    )
    strides = _tcgen05_grouped_external_direct_tensor_arg(
        plan,
        args,
        key="direct_strides_idx",
        tensor_name="direct_strides",
        dtype=torch.int32,
        ndim=3,
    )
    if strides.device != pointers.device:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped external direct metadata pointers and strides "
            "must be on the same CUDA device",
        )
    return pointers, strides


def _tcgen05_grouped_external_direct_cache_key(
    plan: dict[str, object],
    args: tuple[object, ...] | None,
) -> tuple[object, ...]:
    if args is None or not bool(plan.get("external_direct_pointer_metadata")):
        return ()
    metadata_args = _tcgen05_grouped_external_direct_metadata_args(plan, args)
    assert metadata_args is not None
    pointers, strides = metadata_args
    return (
        *_tcgen05_grouped_tensor_cache_key(
            "direct_pointers", pointers, include_version=True
        ),
        *_tcgen05_grouped_tensor_cache_key(
            "direct_strides", strides, include_version=True
        ),
    )


def _tcgen05_grouped_direct_d_idx(
    cute_kernel: object,
    _plan: dict[str, object],
) -> int:
    d_plans = [
        cast("dict[str, object]", candidate)
        for candidate in getattr(
            cast("Any", cute_kernel), "_helion_cute_wrapper_plans", ()
        )
        if cast("dict[str, object]", candidate).get("kind") == "tcgen05_d_tma"
        and bool(cast("dict[str, object]", candidate).get("rank3_mnl_tensor"))
    ]
    if len(d_plans) != 1:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires exactly one "
            "rank-3 grouped D TensorMap wrapper plan",
        )
    d_idx = d_plans[0].get("d_idx")
    if not isinstance(d_idx, int):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata D TensorMap plan is missing "
            "the output tensor index",
        )
    return d_idx


def _tcgen05_grouped_direct_d_tensor_arg(
    cute_kernel: object,
    plan: dict[str, object],
    args: tuple[object, ...],
) -> torch.Tensor:
    d_idx = _tcgen05_grouped_direct_d_idx(cute_kernel, plan)
    if d_idx >= len(args) or not isinstance(args[d_idx], torch.Tensor):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires tensor operand D",
        )
    tensor = args[d_idx]
    assert isinstance(tensor, torch.Tensor)
    if tensor.device.type != "cuda":
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata operand D must be CUDA",
        )
    return tensor


def _tcgen05_grouped_tensor_cache_key(
    label: str,
    tensor: torch.Tensor,
    *,
    include_version: bool = False,
) -> tuple[object, ...]:
    return (
        label,
        id(tensor),
        int(tensor.data_ptr()),
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        int(tensor.storage_offset()),
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        *(_tcgen05_grouped_tensor_mutation_key(tensor) if include_version else ()),
    )


def _tcgen05_grouped_tensor_mutation_key(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    if not torch.is_inference(tensor):
        return ("version", int(tensor._version))
    if (
        tensor.device.type == "cuda"
        and _cuda_stream_capture_context(tensor.device)[1] is not None
    ):
        raise exc.BackendUnsupported(
            "cute",
            "inference tensor grouped metadata is unsupported during CUDA graph "
            "capture; use ordinary tensors with stable version counters",
        )
    return ("values", tuple(tensor.detach().reshape(-1).cpu().tolist()))


@dataclass(frozen=True)
class _Tcgen05GroupedWorklistCompatibilityClassifier:
    """Project packed worklist values onto compatible source-M tile families."""

    static_group_count: int | None
    static_packed_m: int | None
    reviewed_rows: frozenset[GroupedWorklistRows] = frozenset()
    cache: WeakIdKeyDictionary = field(
        default_factory=WeakIdKeyDictionary,
        compare=False,
        repr=False,
    )

    def __call__(self, values: Sequence[object]) -> Hashable:
        tensor = values[0]
        if not isinstance(tensor, torch.Tensor):
            return (
                "not_tensor",
                type(tensor).__module__,
                type(tensor).__name__,
            )
        if isinstance(tensor, FakeTensor):
            if not isinstance(tensor.constant, torch.Tensor):
                return ()
            tensor = tensor.constant

        value_index = 1
        group_count = self.static_group_count
        if group_count is None:
            group_count = cast("int", values[value_index])
            value_index += 1
        packed_m = self.static_packed_m
        if packed_m is None:
            packed_m = cast("int", values[value_index])

        inference_values: tuple[int, ...] | None = None
        is_inference = torch.is_inference(tensor)
        with unset_fake_temporarily():
            tensor_key = _tcgen05_grouped_tensor_cache_key(
                "worklist",
                tensor,
                include_version=not is_inference,
            )
            if is_inference:
                mutation_key = _tcgen05_grouped_tensor_mutation_key(tensor)
                inference_values = cast("tuple[int, ...]", mutation_key[1])
        input_key = (tensor_key, inference_values, group_count, packed_m)
        try:
            cached_input_key, cached_result = self.cache[tensor]
        except KeyError:
            pass
        else:
            if cached_input_key == input_key:
                return cast("Hashable", cached_result)
        if tensor.ndim != 2 or tensor.shape[1] != 4:
            rows: GroupedWorklistRows = ()
        elif inference_values is not None:
            rows = _tcgen05_grouped_worklist_rows_from_flattened(inference_values)
        else:
            with unset_fake_temporarily():
                copied_rows = cast("list[list[int]]", tensor.detach().cpu().tolist())
            rows = tuple(
                cast(
                    "tuple[int, int, int, int]",
                    tuple(int(item) for item in row),
                )
                for row in copied_rows
            )
        compatible = tcgen05_grouped_worklist_compatible_source_m_tiles(
            rows,
            group_count=group_count,
            packed_m=packed_m,
        )
        result: Hashable = (
            compatible
            if not self.reviewed_rows
            else (compatible, rows if rows in self.reviewed_rows else None)
        )
        self.cache[tensor] = (input_key, result)
        return result


def _validate_tcgen05_grouped_direct_d_tensormap(
    tensor: torch.Tensor,
) -> None:
    if tensor.ndim != 2:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires rank-2 D",
        )
    if any(
        int(stride) < 0 or int(stride) > torch.iinfo(torch.int32).max
        for stride in tensor.stride()
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires non-negative "
            "int32 D strides",
        )
    alignment = 16
    if int(tensor.data_ptr()) % alignment != 0:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires 16-byte-aligned D",
        )
    if int(tensor.stride(0)) * tensor.element_size() % alignment != 0:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped direct pointer metadata requires 16-byte-aligned "
            "D row starts",
        )


def _validate_tcgen05_grouped_dynamic_ab_tensormaps(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> None:
    lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
    rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
    rank = _tcgen05_grouped_dynamic_ab_tensormap_rank(plan)
    if lhs.ndim != 2:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require rank-2 A",
        )
    if bool(plan.get("shared_rhs")):
        if (
            rank != 2
            or _tcgen05_plan_orientation(plan) != "nm"
            or rhs.ndim != 2
            or lhs.dtype not in (torch.float16, torch.bfloat16)
            or rhs.dtype != lhs.dtype
            or lhs.shape
            != (_plan_int_value(plan, "m_size"), _plan_int_value(plan, "k_total_size"))
            or rhs.shape
            != (_plan_int_value(plan, "k_total_size"), _plan_int_value(plan, "n_size"))
        ):
            raise exc.BackendUnsupported(
                "cute",
                "shared RHS worklist requires matching rank-2 FP16/BF16 matrices",
            )
        rhs_k_contiguous = rhs.stride(0) == 1 and rhs.stride(1) == rhs.size(0)
        rhs_n_contiguous = rhs.stride(1) == 1 and rhs.stride(0) == rhs.size(1)
        if lhs.stride() != (lhs.size(1), 1) or not (
            rhs_k_contiguous or rhs_n_contiguous
        ):
            raise exc.BackendUnsupported(
                "cute", "shared RHS worklist requires contiguous matrix storage"
            )
        rhs_outer_stride = rhs.stride(1) if rhs_k_contiguous else rhs.stride(0)
        if (
            int(lhs.data_ptr()) % 16
            or int(rhs.data_ptr()) % 16
            or int(lhs.stride(0)) * lhs.element_size() % 16
            or int(rhs_outer_stride) * rhs.element_size() % 16
        ):
            raise exc.BackendUnsupported(
                "cute",
                "shared RHS worklist requires 16-byte-aligned bases and outer strides",
            )
        return
    if rhs.ndim != 3:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require rank-3 grouped B",
        )
    if lhs.stride(1) != 1:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require K-contiguous A",
        )
    rhs_k_contiguous = rhs.stride(2) == 1
    rhs_mn_contiguous = (
        rhs.stride(1) == 1
        and rhs.stride(2) == rhs.size(1)
        and rhs.stride(0) == rhs.size(1) * rhs.size(2)
    )
    if rhs_mn_contiguous and _tcgen05_plan_orientation(plan) != "nm":
        raise exc.BackendUnsupported(
            "cute",
            "MN-major grouped B is validated only for the N,M worklist path",
        )
    if not (rhs_k_contiguous or rhs_mn_contiguous):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require contiguous K-major "
            "or MN-major grouped B",
        )
    if rank == 2:
        if lhs.stride(0) != lhs.size(1):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped rank-2 dynamic A/B TensorMaps require "
                "contiguous A[M,K] outer stride",
            )
        if not (
            (
                rhs_k_contiguous
                and rhs.stride(1) == rhs.size(2)
                and rhs.stride(0) == rhs.size(1) * rhs.size(2)
            )
            or rhs_mn_contiguous
        ):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped rank-2 dynamic A/B TensorMaps require "
                "contiguous B[G,N,K] outer strides",
            )
    alignment = 16
    lhs_stride0_bytes = int(lhs.stride(0)) * lhs.element_size()
    rhs_stride0_bytes = int(rhs.stride(0)) * rhs.element_size()
    rhs_outer_matrix_stride_bytes = (
        int(rhs.stride(1) if rhs_k_contiguous else rhs.stride(2)) * rhs.element_size()
    )
    if (
        int(lhs.data_ptr()) % alignment != 0
        or int(rhs.data_ptr()) % alignment != 0
        or lhs_stride0_bytes % alignment != 0
        or rhs_stride0_bytes % alignment != 0
        or rhs_outer_matrix_stride_bytes % alignment != 0
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require 16-byte-aligned "
            "A/B bases and outer strides",
        )


def _validate_tcgen05_grouped_fixed_tensormaps(
    cute_kernel: object,
    plan: dict[str, object],
    args: tuple[object, ...],
) -> None:
    """Validate the immutable full-allocation worklist TensorMap envelope."""
    if not bool(plan.get("fixed_tensormaps")):
        return
    if (
        _tcgen05_plan_orientation(plan) != "nm"
        or not bool(plan.get("worklist_metadata"))
        or _tcgen05_grouped_device_split_sizes(plan)
        or bool(plan.get("dynamic_ab_tensormaps"))
        or bool(plan.get("dynamic_d_tensormap"))
        or _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) != 2
    ):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require a host N,M worklist with "
            "rank-2 immutable A/B/D descriptors",
        )
    _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, args)
    lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
    rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
    n_size = _plan_int_value(plan, "n_size")
    k_total_size = _plan_int_value(plan, "k_total_size")
    bk = _plan_int_value(plan, "bk")
    physical_mma_m = _plan_int_value(plan, "bm")
    if (
        lhs.dtype not in (torch.float16, torch.bfloat16)
        or rhs.dtype != lhs.dtype
        or int(lhs.size(1)) != k_total_size
        or int(rhs.size(1)) != n_size
        or int(rhs.size(2)) != k_total_size
        or physical_mma_m not in (128, 256)
        or n_size % physical_mma_m != 0
        or k_total_size % bk != 0
    ):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require matching contiguous FP16/BF16 "
            "A[Mtotal,K] and B[G,N,K], N divisible by the physical MMA-M "
            "tile, and K divisible by block_k",
        )

    d_plans = [
        cast("dict[str, object]", candidate)
        for candidate in getattr(
            cast("Any", cute_kernel), "_helion_cute_wrapper_plans", ()
        )
        if cast("dict[str, object]", candidate).get("kind") == "tcgen05_d_tma"
        and bool(cast("dict[str, object]", candidate).get("fixed_tensormap"))
    ]
    if len(d_plans) != 1:
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require exactly one packed D "
            "TensorMap wrapper plan",
        )
    d_idx = d_plans[0].get("d_idx")
    if not isinstance(d_idx, int) or d_idx >= len(args):
        raise exc.BackendUnsupported(
            "cute", "fixed full-allocation D TensorMap argument is missing"
        )
    output = args[d_idx]
    if not isinstance(output, torch.Tensor):
        raise exc.BackendUnsupported(
            "cute", "fixed full-allocation D TensorMap argument must be a tensor"
        )
    if (
        output.device.type != "cuda"
        or output.dtype != lhs.dtype
        or output.ndim != 2
        or tuple(int(size) for size in output.shape) != (int(lhs.size(0)), n_size)
        or tuple(int(stride) for stride in output.stride()) != (n_size, 1)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require matching contiguous FP16/BF16 "
            "D[Mtotal,N] matching packed A and grouped B",
        )
    alignment = 16
    if int(output.data_ptr()) % alignment != 0:
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require a 16-byte-aligned D base",
        )


def _validate_tcgen05_grouped_worklist_nm(
    plan: dict[str, object],
    args: tuple[object, ...],
    rows: list[list[int]],
) -> None:
    if _tcgen05_plan_orientation(plan) != "nm":
        return
    lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
    rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
    source_m_tile = _plan_int_value(plan, "source_m_tile")
    try:
        validate_tcgen05_grouped_worklist_rows(
            rows,
            group_count=int(rhs.size(0)),
            packed_m=int(lhs.size(0)),
            source_m_tile=source_m_tile,
        )
    except Tcgen05GroupedWorklistValidationError as error:
        raise exc.BackendUnsupported(
            "cute",
            str(error),
        ) from None


def _validate_tcgen05_grouped_runtime_direct_clc_grid(total_clusters: int) -> None:
    if total_clusters > TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped runtime-direct CLC requires at most "
            f"{TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS} exact tile "
            "records because CUDA grid.z has the same limit",
        )


def _tcgen05_grouped_runtime_nm_tile_records(
    worklist_rows: list[list[int]],
    problem_sizes: list[tuple[int, int, int, int]],
    *,
    source_tile_m: int,
    source_tile_n: int,
    l2_swizzle_size: int,
) -> torch.Tensor:
    """Expand runtime per-group worklist rows into logical N,M tile records."""
    metadata: list[tuple[int, ...]] = []
    tile_counts: list[int] = []
    for row, problem_size in zip(worklist_rows, problem_sizes, strict=True):
        real_group, global_m_start, actual_m, aligned_m = (int(value) for value in row)
        problem_n, problem_m, problem_k, batch = problem_size
        if batch != 1 or problem_m != aligned_m:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped runtime N,M tile records require batch 1 and "
                "problem M matching the aligned worklist extent",
            )
        m_tiles = aligned_m // source_tile_m
        n_tiles = (problem_n + source_tile_n - 1) // source_tile_n
        metadata.append(
            (
                real_group,
                global_m_start,
                actual_m,
                aligned_m,
                problem_n,
                problem_k,
                m_tiles,
                n_tiles,
            )
        )
        tile_counts.append(m_tiles * n_tiles)

    total_tiles = sum(tile_counts)
    metadata_tensor = torch.tensor(metadata, dtype=torch.int32)
    counts = torch.tensor(tile_counts, dtype=torch.int64)
    metadata_idx = torch.repeat_interleave(
        torch.arange(len(metadata), dtype=torch.int64), counts
    )
    group_starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    local_idx = torch.arange(total_tiles, dtype=torch.int64) - group_starts
    selected = metadata_tensor[metadata_idx]
    (
        real_group,
        global_m_start,
        actual_m,
        aligned_m,
        problem_n,
        problem_k,
        m_tiles,
        n_tiles,
    ) = (selected[:, field] for field in range(8))
    # CUTLASS scheduling excludes zero-tile groups: repeat_interleave omits their
    # zero-count rows, so every source-M raster divisor below is strictly positive.
    if l2_swizzle_size > 1:
        panel_size = torch.minimum(torch.full_like(n_tiles, l2_swizzle_size), n_tiles)
        panel_span = panel_size * m_tiles
        panel_idx = torch.div(local_idx, panel_span, rounding_mode="floor")
        panel_linear = local_idx % panel_span
        panel_width = torch.minimum(panel_size, n_tiles - panel_idx * panel_size)
        cta_tile_idx_m = torch.div(panel_linear, panel_width, rounding_mode="floor")
        cta_tile_idx_n = panel_idx * panel_size + panel_linear % panel_width
        cta_tile_idx_m = torch.where(
            panel_idx % 2 == 1,
            m_tiles - 1 - cta_tile_idx_m,
            cta_tile_idx_m,
        )
    else:
        m_first = m_tiles <= n_tiles
        cta_tile_idx_m = torch.where(
            m_first,
            local_idx % m_tiles,
            torch.div(local_idx, n_tiles, rounding_mode="floor"),
        )
        cta_tile_idx_n = torch.where(
            m_first,
            torch.div(local_idx, m_tiles, rounding_mode="floor"),
            local_idx % n_tiles,
        )
    tile_start = cta_tile_idx_m * source_tile_m
    # STORE_M remains explicit for parity with the mailbox/device schema.
    # Host-expanded rows always reserve one full physical source tile.
    columns = {
        Tcgen05GroupedRuntimeTileField.CTA_M: cta_tile_idx_m,
        Tcgen05GroupedRuntimeTileField.CTA_N: cta_tile_idx_n,
        Tcgen05GroupedRuntimeTileField.METADATA_IDX: metadata_idx,
        Tcgen05GroupedRuntimeTileField.GROUP_IDX: real_group,
        Tcgen05GroupedRuntimeTileField.PROBLEM_M: aligned_m,
        Tcgen05GroupedRuntimeTileField.PROBLEM_N: problem_n,
        Tcgen05GroupedRuntimeTileField.PROBLEM_K: problem_k,
        Tcgen05GroupedRuntimeTileField.GLOBAL_M_START: global_m_start,
        Tcgen05GroupedRuntimeTileField.VALID_M: (actual_m - tile_start).clamp(
            0, source_tile_m
        ),
        Tcgen05GroupedRuntimeTileField.STORE_M: torch.full_like(
            cta_tile_idx_m, source_tile_m
        ),
    }
    return torch.stack(
        tuple(columns[field] for field in Tcgen05GroupedRuntimeTileField),
        dim=1,
    ).to(torch.int32)


def _tcgen05_grouped_static_metadata_cache_key(
    cute_kernel: object,
    plan: dict[str, object],
    layout: torch.Tensor,
    n_sizes: torch.Tensor | None,
    k_sizes: torch.Tensor | None,
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Cache key for scheduler metadata copied from grouped-static tensors.

    The hot path guards ordinary layout/n_sizes/k_sizes tensors by identity,
    metadata, and ``_version``. Inference tensors have no version counter, so
    their exact values form the mutation key instead. Inference metadata is
    unsupported during CUDA graph capture because its current values cannot be
    safely copied to the host there. Version-bypassing writes to ordinary tensors
    remain unsupported; users must rewarm with the final metadata values.
    During CUDA graph capture/replay these metadata tensors must match the exact
    prewarmed values and remain immutable. Frozen wrapper plans supply the static
    dependencies; source extents are included only when worklist validation reads
    them, and full source tensor metadata only when generated pointer tables embed
    it.
    """
    wrapper_plans = getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", ())
    plan_index = next(
        (index for index, candidate in enumerate(wrapper_plans) if candidate is plan),
        None,
    )
    if plan_index is None:
        raise exc.BackendUnsupported(
            "cute", "grouped scheduler plan is not registered on the CuTe kernel"
        )

    worklist_source_extents: list[object] = []
    if bool(plan.get("worklist_metadata")):
        lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
        rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
        worklist_source_extents = [
            "worklist_source_extents",
            int(lhs.size(0)),
            int(rhs.size(0)),
        ]

    return (
        "tcgen05_grouped_static_metadata",
        plan_index,
        *_tcgen05_grouped_tensor_cache_key("layout", layout, include_version=True),
        *worklist_source_extents,
        *(
            (
                *_tcgen05_grouped_tensor_cache_key(
                    "direct_lhs",
                    _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A"),
                ),
                *_tcgen05_grouped_tensor_cache_key(
                    "direct_rhs",
                    _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B"),
                ),
                *_tcgen05_grouped_tensor_cache_key(
                    "direct_d",
                    _tcgen05_grouped_direct_d_tensor_arg(cute_kernel, plan, args),
                ),
            )
            if bool(plan.get("direct_pointer_metadata"))
            else ()
        ),
        *_tcgen05_grouped_external_direct_cache_key(plan, args),
        *(
            _tcgen05_grouped_tensor_cache_key("n_sizes", n_sizes, include_version=True)
            if n_sizes is not None
            else ()
        ),
        *(
            _tcgen05_grouped_tensor_cache_key("k_sizes", k_sizes, include_version=True)
            if k_sizes is not None
            else ()
        ),
    )


def _append_tcgen05_grouped_static_mutation_guards(
    key: list[object],
    cute_kernel: object,
    args: tuple[object, ...],
) -> None:
    for plan in _tcgen05_grouped_static_plans(cute_kernel):
        if _tcgen05_grouped_device_split_sizes(plan):
            continue
        for label, index in (
            ("layout", _plan_int_value(plan, "layout_idx")),
            ("n_sizes", plan.get("n_sizes_idx")),
            ("k_sizes", plan.get("k_sizes_idx")),
        ):
            if not (
                isinstance(index, int)
                and index < len(args)
                and isinstance(args[index], torch.Tensor)
            ):
                continue
            tensor = args[index]
            assert isinstance(tensor, torch.Tensor)
            key.append(
                (
                    f"tcgen05_grouped_static_{label}",
                    index,
                    id(tensor),
                    *_tcgen05_grouped_tensor_mutation_key(tensor),
                )
            )


def _tcgen05_grouped_static_metadata_cache(
    cute_kernel: object,
) -> OrderedDict[tuple[object, ...], _Tcgen05GroupedStaticMetadataCacheEntry] | None:
    try:
        return cast(
            "OrderedDict[tuple[object, ...], _Tcgen05GroupedStaticMetadataCacheEntry]",
            cast("Any", cute_kernel)._helion_tcgen05_grouped_static_metadata_cache,
        )
    except AttributeError:
        return None


def _tcgen05_grouped_static_metadata_cache_entry(
    cute_kernel: object,
    cache_key: tuple[object, ...],
    layout: torch.Tensor,
    n_sizes: torch.Tensor | None,
    k_sizes: torch.Tensor | None = None,
) -> _Tcgen05GroupedStaticMetadataCacheEntry | None:
    cache = _tcgen05_grouped_static_metadata_cache(cute_kernel)
    if cache is None:
        return None
    cached = cache.get(cache_key)
    if cached is not None and cached.matches(layout, n_sizes, k_sizes):
        cache.move_to_end(cache_key)
        return cached
    if cached is not None:
        cache.pop(cache_key, None)
    return None


def _build_tcgen05_grouped_static_metadata(
    cute_kernel: object,
    plan: dict[str, object],
    args: tuple[object, ...],
) -> _Tcgen05GroupedStaticMetadataCacheEntry:
    layout = _tcgen05_grouped_static_layout_arg(plan, args)
    _validate_tcgen05_grouped_tensor_devices(layout, args)
    _validate_tcgen05_grouped_fixed_tensormaps(cute_kernel, plan, args)
    n_sizes_arg = _tcgen05_grouped_static_size_arg(plan, args, "n_sizes")
    k_sizes_arg = _tcgen05_grouped_static_size_arg(plan, args, "k_sizes")
    cache_key = _tcgen05_grouped_static_metadata_cache_key(
        cute_kernel,
        plan,
        layout,
        n_sizes_arg,
        k_sizes_arg,
        args,
    )
    cache = _tcgen05_grouped_static_metadata_cache(cute_kernel)
    if cache is None:
        cache = OrderedDict()
        cast("Any", cute_kernel)._helion_tcgen05_grouped_static_metadata_cache = cache
    cached = _tcgen05_grouped_static_metadata_cache_entry(
        cute_kernel,
        cache_key,
        layout,
        n_sizes_arg,
        k_sizes_arg,
    )
    if cached is not None:
        return cached
    if _cuda_stream_capture_context(layout.device)[1] is not None:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped scheduler metadata is not cached for this layout; "
            "call the kernel once with the final grouped metadata values before "
            "CUDA graph capture",
        )

    group_count = _plan_int_value(plan, "group_count")
    bm = _plan_int_value(plan, "bm")
    bn = _plan_int_value(plan, "bn")
    bk = _plan_int_value(plan, "bk")
    worklist_nm = _tcgen05_plan_orientation(plan) == "nm"
    n_size = _plan_int_value(plan, "n_size")
    k_total_size = _plan_int_value(plan, "k_total_size")
    if worklist_nm:
        worklist_m_tile = _plan_int_value(plan, "source_m_tile")
        if worklist_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 N,M worklist scheduler requires a validated source M tile",
            )
        scheduler_bm = bm
        scheduler_bn = worklist_m_tile
    else:
        worklist_m_tile = scheduler_bm = bm
        scheduler_bn = bn
    scheduler_mode = _tcgen05_grouped_scheduler_mode(plan)
    runtime_direct_tile_table = scheduler_mode in (
        Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
        Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
    )
    dynamic_ab_tensormaps = bool(plan.get("dynamic_ab_tensormaps"))
    dynamic_d_tensormap = bool(plan.get("dynamic_d_tensormap"))
    fixed_tensormaps = bool(plan.get("fixed_tensormaps"))
    direct_pointer_metadata = bool(plan.get("direct_pointer_metadata"))
    external_direct_metadata = _tcgen05_grouped_external_direct_metadata_args(
        plan,
        args,
    )
    worklist_metadata = bool(plan.get("worklist_metadata"))
    m_tail_preserve = bool(plan.get("m_tail_preserve"))
    n_tail_preserve = bool(plan.get("n_tail_preserve"))
    if worklist_nm and not worklist_metadata:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 N,M orientation requires grouped worklist metadata"
        )
    if dynamic_ab_tensormaps and bk != 64 and not (worklist_nm and bk == 128):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps are validated only for "
            "BK64, or BK128 on the N,M worklist path",
        )
    direct_lhs: torch.Tensor | None = None
    direct_rhs: torch.Tensor | None = None
    direct_d: torch.Tensor | None = None
    if direct_pointer_metadata:
        if not dynamic_ab_tensormaps or not bool(plan.get("dynamic_d_tensormap")):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped direct pointer metadata requires dynamic A/B "
                "and D TensorMaps",
            )
        if worklist_nm:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped direct pointer metadata supports only the "
                "default M,N TensorMap orientation",
            )
        _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, args)
        direct_lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
        direct_rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
        direct_d = _tcgen05_grouped_direct_d_tensor_arg(cute_kernel, plan, args)
        _validate_tcgen05_grouped_direct_d_tensormap(direct_d)
    if worklist_nm:
        if (
            n_size <= 0
            or n_size % TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[2] != 0
            or k_total_size % bk != 0
        ):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 N,M worklist scheduler requires output N "
                "divisible by 32 and K divisible by the CTA K tile",
            )
    elif n_size % bn != 0 or k_total_size % bk != 0:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped scheduler requires common N/K dimensions divisible "
            "by the CTA tile",
        )
    if worklist_nm and not (
        (
            dynamic_ab_tensormaps
            and _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) == 2
            and dynamic_d_tensormap
            and not fixed_tensormaps
        )
        or (
            fixed_tensormaps
            and not dynamic_ab_tensormaps
            and not dynamic_d_tensormap
            and _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) == 2
        )
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M worklist metadata requires either rank-2 dynamic "
            "per-group A/B/D TensorMaps or fixed full-allocation A/B/D "
            "TensorMaps",
        )
    n_sizes_values: list[int] | None = None
    if n_sizes_arg is not None:
        if int(n_sizes_arg.numel()) != group_count:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler n_sizes length must match group count",
            )
        n_sizes_values = [int(value) for value in n_sizes_arg.detach().cpu().tolist()]
    k_sizes_values: list[int] | None = None
    if k_sizes_arg is not None:
        if int(k_sizes_arg.numel()) != group_count:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler k_sizes length must match group count",
            )
        k_sizes_values = [int(value) for value in k_sizes_arg.detach().cpu().tolist()]
    if dynamic_ab_tensormaps and k_sizes_values is None and not worklist_metadata:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped dynamic A/B TensorMaps require exact per-group k_sizes",
        )

    starts: list[int] = []
    sizes: list[int] = []
    has_m_tail = False
    real_groups: list[int] | None = [] if worklist_metadata else None
    worklist_rows: list[list[int]] | None = None
    if worklist_metadata:
        if int(layout.size(0)) != group_count:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped worklist scheduler row count must match group count",
            )
        if n_sizes_values is not None or k_sizes_values is not None:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped worklist scheduler uses common N/K sizes",
            )
        lhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "lhs_idx", "A")
        rhs = _tcgen05_grouped_dynamic_ab_tensor_arg(plan, args, "rhs_idx", "B")
        worklist_rows = cast("list[list[int]]", layout.detach().cpu().tolist())
        _validate_tcgen05_grouped_worklist_nm(plan, args, worklist_rows)
        for row in worklist_rows:
            real_group, start, valid_m, reserved_or_store_m = (
                int(value) for value in row
            )
            if worklist_nm:
                aligned_m = reserved_or_store_m
                starts.append(start)
                sizes.append(aligned_m)
                assert real_groups is not None
                real_groups.append(real_group)
                continue
            if reserved_or_store_m != 0:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped worklist scheduler requires reserved "
                    "metadata column to be zero",
                )
            if real_group < 0 or real_group >= int(rhs.size(0)):
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped worklist scheduler real group id is "
                    "outside B_grouped",
                )
            if start < 0 or valid_m <= 0 or valid_m > worklist_m_tile:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped worklist scheduler requires each row to "
                    "describe one CTA-M tile",
                )
            if start + valid_m > int(lhs.size(0)):
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped worklist scheduler row exceeds A extent",
                )
            starts.append(start)
            sizes.append(valid_m)
            assert real_groups is not None
            real_groups.append(real_group)
    else:
        layout_values = [int(value) for value in layout.detach().cpu().tolist()]
        cursor = 0
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
                    raise exc.BackendUnsupported(
                        "cute",
                        "tcgen05 grouped scheduler requires interior M-tail padding "
                        "to end at the next CTA M tile boundary",
                    )
            if cursor >= len(layout_values) or layout_values[cursor] != expected_group:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped scheduler requires ordered complete groups "
                    "without row holes or skipped group indices",
                )
            start = cursor
            while (
                cursor < len(layout_values) and layout_values[cursor] == expected_group
            ):
                cursor += 1
            actual_m = cursor - start
            if start % bm != 0:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped scheduler requires each group start to be "
                    "divisible by the CTA M tile",
                )
            if actual_m % bm != 0 and not m_tail_preserve:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped scheduler requires each group M size to be "
                    "divisible by the CTA M tile unless the source store proves "
                    "grouped M-tail preservation",
                )
            has_m_tail = has_m_tail or actual_m % bm != 0
            starts.append(start)
            sizes.append(actual_m)
        if cursor != len(layout_values):
            if m_tail_preserve and all(value < 0 for value in layout_values[cursor:]):
                cursor = len(layout_values)
        if cursor != len(layout_values):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler requires layout rows to end after the "
                "last ordered group",
            )

    if not any(sizes):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped scheduler does not launch an all-empty worklist; "
            "all groups are empty, so the caller must skip the GEMM invocation",
        )

    problem_sizes = []
    total_clusters = 0
    has_n_tail = False
    for group_idx, actual_m in enumerate(sizes):
        group_n = n_size if n_sizes_values is None else n_sizes_values[group_idx]
        group_k = k_total_size if k_sizes_values is None else k_sizes_values[group_idx]
        if group_n <= 0 or group_n > n_size:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler requires each per-group N size to be "
                "positive and within the output N extent",
            )
        if group_k <= 0 or group_k > k_total_size:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler requires each per-group K size to be "
                "positive and within the padded K extent",
            )
        if worklist_nm:
            if group_n % TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[2] != 0:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 N,M worklist scheduler requires each "
                    "per-group output N size to be divisible by 32",
                )
            has_n_tail = has_n_tail or group_n % scheduler_bm != 0
        else:
            if group_n % bn != 0 and not n_tail_preserve:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped scheduler requires each per-group N size to be "
                    "divisible by the CTA N tile unless the source store proves "
                    "grouped N-tail preservation",
                )
            has_n_tail = has_n_tail or group_n % bn != 0
        if group_k % bk != 0 and not (dynamic_ab_tensormaps and group_k % 16 == 0):
            if dynamic_ab_tensormaps:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped dynamic A/B TensorMaps require each "
                    "per-group K size to be a multiple of 16",
                )
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped scheduler requires each per-group K size to be "
                "divisible by the CTA K tile",
            )
        if worklist_nm:
            problem_sizes.append((group_n, actual_m, group_k, 1))
            total_clusters += ((group_n + scheduler_bm - 1) // scheduler_bm) * (
                (actual_m + scheduler_bn - 1) // scheduler_bn
            )
        else:
            problem_sizes.append((actual_m, group_n, group_k, 1))
            total_clusters += ((actual_m + bm - 1) // bm) * ((group_n + bn - 1) // bn)
    if total_clusters <= 0:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped scheduler found zero work clusters"
        )
    if scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC:
        _validate_tcgen05_grouped_runtime_direct_clc_grid(total_clusters)
    expected_has_m_tail = plan.get("grouped_static_has_m_tail")
    if isinstance(expected_has_m_tail, bool) and has_m_tail != expected_has_m_tail:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped M-tail specialization does not match the current "
            "layout metadata; rebind and prewarm the grouped kernel with the "
            "final metadata before launch or CUDA graph capture",
        )
    expected_has_n_tail = plan.get("grouped_static_has_n_tail")
    if isinstance(expected_has_n_tail, bool) and has_n_tail != expected_has_n_tail:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped N-tail specialization does not match the current "
            "n_sizes metadata; rebind and prewarm the grouped kernel with the "
            "final metadata before launch or CUDA graph capture",
        )
    if (expected_shapes := plan.get("static_problem_shapes")) is not None:
        actual_shapes = tuple(
            (problem_m, problem_n, problem_k)
            for problem_m, problem_n, problem_k, _batch in problem_sizes
        )
        if actual_shapes != expected_shapes:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped static problem-shape specialization does not "
                "match the current layout/n_sizes/k_sizes metadata; expected "
                f"{expected_shapes!r}, got {actual_shapes!r}; rebind and prewarm "
                "with the final grouped shapes before launch or CUDA graph capture",
            )

    device = layout.device
    direct_pointers_tensor: torch.Tensor | None = None
    direct_strides_tensor: torch.Tensor | None = None
    if direct_pointer_metadata:
        assert direct_lhs is not None
        assert direct_rhs is not None
        assert direct_d is not None
        if external_direct_metadata is not None:
            direct_pointers_tensor, direct_strides_tensor = external_direct_metadata
            if direct_pointers_tensor.device != device:
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped external direct metadata must be on the "
                    "grouped scheduler metadata device",
                )
            if tuple(int(size) for size in direct_pointers_tensor.shape) != (
                len(sizes),
                3,
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped external direct pointer metadata must have "
                    "shape [metadata_rows, 3]",
                )
            if tuple(int(size) for size in direct_strides_tensor.shape) != (
                len(sizes),
                3,
                2,
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "tcgen05 grouped external direct stride metadata must have "
                    "shape [metadata_rows, 3, 2]",
                )
        else:
            direct_pointer_rows: list[tuple[int, int, int]] = []
            direct_stride_rows: list[
                tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
            ] = []
            for metadata_idx, (start, actual_m) in enumerate(
                zip(starts, sizes, strict=True)
            ):
                real_group = (
                    real_groups[metadata_idx]
                    if real_groups is not None
                    else metadata_idx
                )
                group_n = (
                    n_size if n_sizes_values is None else n_sizes_values[metadata_idx]
                )
                group_k = (
                    k_total_size
                    if k_sizes_values is None
                    else k_sizes_values[metadata_idx]
                )
                if start + actual_m > int(direct_lhs.size(0)):
                    raise exc.BackendUnsupported(
                        "cute",
                        "tcgen05 grouped direct pointer metadata A segment exceeds "
                        "the source extent",
                    )
                if start + actual_m > int(direct_d.size(0)) or group_n > int(
                    direct_d.size(1)
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        "tcgen05 grouped direct pointer metadata D segment exceeds "
                        "the output extent",
                    )
                if real_group < 0 or real_group >= int(direct_rhs.size(0)):
                    raise exc.BackendUnsupported(
                        "cute",
                        "tcgen05 grouped direct pointer metadata B group is out "
                        "of range",
                    )
                if group_n > int(direct_rhs.size(1)) or group_k > int(
                    direct_rhs.size(2)
                ):
                    raise exc.BackendUnsupported(
                        "cute",
                        "tcgen05 grouped direct pointer metadata B segment exceeds "
                        "the grouped B extent",
                    )
                direct_pointer_rows.append(
                    (
                        int(direct_lhs.data_ptr())
                        + start * int(direct_lhs.stride(0)) * direct_lhs.element_size(),
                        int(direct_rhs.data_ptr())
                        + real_group
                        * int(direct_rhs.stride(0))
                        * direct_rhs.element_size(),
                        int(direct_d.data_ptr())
                        + start * int(direct_d.stride(0)) * direct_d.element_size(),
                    )
                )
                direct_stride_rows.append(
                    (
                        (int(direct_lhs.stride(0)), int(direct_lhs.stride(1))),
                        (int(direct_rhs.stride(1)), int(direct_rhs.stride(2))),
                        (int(direct_d.stride(0)), int(direct_d.stride(1))),
                    )
                )
            direct_pointers_tensor = torch.tensor(
                direct_pointer_rows, dtype=torch.int64, device=device
            )
            direct_strides_tensor = torch.tensor(
                direct_stride_rows, dtype=torch.int32, device=device
            )
    problem_tensor: torch.Tensor | None = None
    starts_tensor: torch.Tensor | None = None
    real_groups_tensor: torch.Tensor | None = None
    if not runtime_direct_tile_table:
        problem_tensor = torch.tensor(problem_sizes, dtype=torch.int32, device=device)
        starts_tensor = torch.tensor(starts, dtype=torch.int32, device=device)
        if real_groups is not None:
            real_groups_tensor = torch.tensor(
                real_groups, dtype=torch.int32, device=device
            )
    runtime_tile_records_tensor: torch.Tensor | None = None
    if runtime_direct_tile_table:
        if not worklist_nm or worklist_rows is None:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 grouped runtime tile tables require N,M worklist metadata",
            )
        l2_swizzle_size = _plan_int_value(plan, "l2_swizzle_size")
        runtime_tile_records = _tcgen05_grouped_runtime_nm_tile_records(
            worklist_rows,
            problem_sizes,
            source_tile_m=worklist_m_tile,
            source_tile_n=scheduler_bm,
            l2_swizzle_size=l2_swizzle_size,
        )
        if len(runtime_tile_records) != total_clusters:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 runtime N,M tile table row count does not match "
                "the grouped launch cluster count",
            )
        if (
            runtime_tile_records.ndim != 2
            or runtime_tile_records.size(1) != TCGEN05_GROUPED_RUNTIME_TILE_FIELD_COUNT
        ):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 runtime N,M tile table does not match the shared field schema",
            )
        runtime_tile_records_tensor = runtime_tile_records.to(device=device)
    result = _Tcgen05GroupedStaticMetadataResult(
        problem_sizes=problem_tensor,
        starts=starts_tensor,
        total_clusters=total_clusters,
        real_groups=real_groups_tensor,
        runtime_tile_records=runtime_tile_records_tensor,
        direct_pointers=direct_pointers_tensor,
        direct_strides=direct_strides_tensor,
    )
    entry = _Tcgen05GroupedStaticMetadataCacheEntry(
        layout_ref=weakref.ref(layout),
        n_sizes_ref=weakref.ref(n_sizes_arg) if n_sizes_arg is not None else None,
        k_sizes_ref=weakref.ref(k_sizes_arg) if k_sizes_arg is not None else None,
        has_m_tail=has_m_tail,
        has_n_tail=has_n_tail,
        result=result,
    )
    cache[cache_key] = entry
    cache.move_to_end(cache_key)
    while len(cache) > _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT:
        cache.popitem(last=False)
    return entry


def _cute_arg_signature(
    cute_kernel: object, args: tuple[object, ...], *, validate: bool = True
) -> ArgSignature:
    """The per-argument facts of a launch (``arg_signature.ArgSignature``),
    built through the launcher's seams: the kernel's Constexpr flags and the
    CUDA-tensor check tests patch on this module.  The fast relaunch probe
    declines non-CUDA tensors itself (``validate=False``)."""
    return ArgSignature.build(
        args,
        _cute_kernel_param_is_constexpr(cute_kernel),
        _validate_cute_launcher_tensor if validate else _accept_launcher_tensor,
    )


def _accept_launcher_tensor(tensor: torch.Tensor) -> None:
    """The no-op tensor check of callers that decline non-CUDA tensors."""


def _cute_launch_arg_cache_key(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    *,
    dynamic_tensormap_contexts: tuple[_CuteGroupedLaunchContext, ...] | None = None,
    signature: ArgSignature | None = None,
) -> tuple[object, ...]:
    if signature is None:
        signature = _cute_arg_signature(cute_kernel, args)
    key: list[object] = [len(args), grid, _cute_bake_tensor_shapes_guard(cute_kernel)]
    if _tcgen05_split_k_plan_present(cute_kernel):
        # The split-K workspace pointer in the launch arguments is cached per
        # (stream, capture) (``_tcgen05_split_k_workspace``); a captured
        # graph must not reuse the arguments of an eager launch.
        device = next(
            (arg.device for arg in args if isinstance(arg, torch.Tensor)), None
        )
        if device is not None:
            key.append(("workspace_context", *_cuda_stream_capture_context(device)))
    if dynamic_tensormap_contexts is None:
        dynamic_tensormap_contexts = _cute_dynamic_tensormap_contexts(cute_kernel, args)
    if dynamic_tensormap_contexts:
        key.append(("dynamic_tensormap_contexts", dynamic_tensormap_contexts))
    key.extend(signature.cache_key_entries())
    _append_tcgen05_grouped_static_mutation_guards(key, cute_kernel, args)
    return tuple(key)


def _retain_cute_capture_owned_launch_tensors(
    cute_kernel: object,
    *,
    grouped_launch_contexts: tuple[_CuteGroupedLaunchContext, ...],
    owned_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Keep raw-pointer launch tensors alive for captured graph replays."""
    if not owned_tensors:
        return
    contexts = list(grouped_launch_contexts)
    contexts_by_device = {
        (context[0], context[1]): context for context in grouped_launch_contexts
    }
    # Grouped tcgen05 plans already contribute their stream/capture context.
    # Other launcher-owned allocations (notably KDA's raw TensorMap descriptor
    # storage) have no grouped plan, so derive the missing contexts from the
    # tensors whose pointers the captured launch embeds.
    for tensor in owned_tensors:
        device_key = (tensor.device.type, tensor.device.index)
        if device_key in contexts_by_device:
            continue
        stream_handle, capture_id = _cuda_stream_capture_context(tensor.device)
        context = (*device_key, stream_handle, capture_id)
        contexts_by_device[device_key] = context
        contexts.append(context)
    capture_contexts = tuple(context for context in contexts if context[3] is not None)
    if not capture_contexts:
        return
    cache = cast(
        "dict[tuple[_CuteGroupedLaunchContext, ...], dict[int, torch.Tensor]]",
        cast("Any", cute_kernel).__dict__.setdefault(
            "_helion_cute_capture_owned_launch_tensors", {}
        ),
    )
    capture_tensors = cache.setdefault(capture_contexts, {})
    for tensor in owned_tensors:
        capture_tensors[id(tensor)] = tensor
    _track_cute_cuda_graph_cache_entry(
        cute_kernel,
        "_helion_cute_capture_owned_launch_tensors",
        capture_contexts,
        capture_tensors,
        owned_tensors,
    )


def _record_cute_owned_launch_tensors(
    owned_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Associate raw-pointer launch tensors with their current CUDA streams."""
    streams: dict[tuple[str, int | None], torch.cuda.Stream] = {}
    for tensor in owned_tensors:
        device_key = (tensor.device.type, tensor.device.index)
        stream = streams.get(device_key)
        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
            streams[device_key] = stream
        tensor.record_stream(stream)


def _build_cached_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    *,
    signature: ArgSignature | None = None,
) -> _CuteLaunchArgCacheEntry:
    if signature is None:
        signature = _cute_arg_signature(cute_kernel, args)
    dynamic_tensormap_contexts = _cute_dynamic_tensormap_contexts(cute_kernel, args)
    grouped_launch_contexts = _cute_grouped_launch_contexts(
        cute_kernel,
        args,
        dynamic_tensormap_contexts=dynamic_tensormap_contexts,
    )
    cache_key = _cute_launch_arg_cache_key(
        cute_kernel,
        args,
        grid,
        dynamic_tensormap_contexts=dynamic_tensormap_contexts,
        signature=signature,
    )
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_launch_arg_cache
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_launch_arg_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        if isinstance(
            cached, _CuteLaunchArgCacheEntry
        ) and _cute_grouped_static_metadata_matches(
            cached.grouped_static_metadata, cute_kernel, args
        ):
            cache[cache_key] = cache.pop(cache_key)
            _retain_cute_capture_owned_launch_tensors(
                cute_kernel,
                grouped_launch_contexts=grouped_launch_contexts,
                owned_tensors=cached.owned_tensors,
            )
            return cached
        cache.pop(cache_key)

    built = _build_cute_schema_and_args(cute_kernel, args, grid, signature=signature)
    cache[cache_key] = built
    if built.owned_tensors:
        _retain_cute_capture_owned_launch_tensors(
            cute_kernel,
            grouped_launch_contexts=grouped_launch_contexts,
            owned_tensors=built.owned_tensors,
        )
    if len(cache) > _CUTE_LAUNCH_ARG_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    return built


def _build_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    bake_tensor_shapes: bool = True,
    *,
    signature: ArgSignature | None = None,
) -> _CuteLaunchArgCacheEntry:
    # NOTE: the returned launch args deliberately EXCLUDE the CUDA stream. The
    # stream is the only launch arg that is not a pure function of
    # (grid, tensor metadata, scalars), so it must not be baked into the cached
    # args — the caller appends a freshly sampled ``_cute_current_stream()`` on
    # every launch (see ``default_cute_launcher``). Caching the stream would
    # break CUDA graph capture (empty-graph / no-op replay).
    gmem_space, make_ptr_obj, _current_stream_obj = _get_cute_launcher_imports()
    make_ptr = cast("Any", make_ptr_obj)
    if signature is None:
        signature = _cute_arg_signature(cute_kernel, args)
    descriptor = cute_launch_descriptor(cute_kernel)
    # Universal MMA needs runtime tensor layouts for its SMEM-load guards.
    # Full-tile tcgen05 wrapper schemas are specialized by problem shape and
    # stride, while partial-tile paths still propagate runtime tensor layouts.
    if bake_tensor_shapes:
        if not _cute_bake_tensor_shapes_guard(cute_kernel):
            bake_tensor_shapes = False
    schema: list[tuple[object, ...]] = []
    launch_args: list[object] = []
    for fact in signature.entries:
        if isinstance(fact, ScalarArgFact):
            schema.append(fact.schema_entry())
            if not fact.is_constexpr:
                launch_args.append(fact.value)
            continue
        launch_args.append(
            make_ptr(
                cast("Any", _torch_dtype_to_cutlass(fact.dtype)),
                fact.data_ptr,
                gmem_space,
                assumed_align=fact.alignment,
            )
        )
        # Bake the shape / stride tuple into the schema key.  The generated
        # wrapper substitutes literal Int values for each dimension, so the
        # CuTe DSL sees a fully static tensor layout and the per-load offset
        # arithmetic collapses to constant strides — typically a 2-3x
        # reduction in ``smsp__inst_executed`` for reduction kernels where
        # the inner loop is dominated by stride multiplies.  Zero-sized
        # tensors keep the runtime-shape path (``TensorArgFact.bakes``).
        bake = bake_tensor_shapes and fact.bakes()
        schema.append(fact.schema_entry(bake))
        if not bake:
            _ndim, sizes_t, strides_t = fact.layout()
            launch_args.extend(sizes_t)
            launch_args.extend(strides_t)

    owned_tensors: list[torch.Tensor] = []
    grouped_static_metadata: list[_Tcgen05GroupedStaticMetadataCacheEntry] = []

    def append_wrapper_tensor(
        name: str,
        tensor: torch.Tensor,
        *,
        owned: bool = False,
        runtime_leading_extent: bool = False,
    ) -> None:
        _validate_cute_launcher_tensor(tensor)
        sizes = tuple(int(tensor.size(d)) for d in range(tensor.ndim))
        strides = tuple(int(tensor.stride(d)) for d in range(tensor.ndim))
        data_ptr = int(tensor.data_ptr())
        alignment = _cute_pointer_alignment(data_ptr)
        launch_args.append(
            make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                data_ptr,
                gmem_space,
                assumed_align=alignment,
            )
        )
        if runtime_leading_extent:
            schema.append(
                _cute_schema_with_pointer_alignment(
                    (
                        "wrapper_tensor_runtime_leading_extent",
                        name,
                        str(tensor.dtype),
                        tensor.ndim,
                        sizes[1:],
                        strides,
                    ),
                    alignment,
                )
            )
            launch_args.append(sizes[0])
        else:
            schema.append(
                _cute_schema_with_pointer_alignment(
                    (
                        "wrapper_tensor",
                        name,
                        str(tensor.dtype),
                        tensor.ndim,
                        sizes,
                        strides,
                    ),
                    alignment,
                )
            )
        if owned:
            owned_tensors.append(tensor)

    def append_grouped_tensormap_workspace(
        plan: dict[str, object],
        layout: torch.Tensor,
    ) -> None:
        workspace_name: str | None = None
        tensormap_count = 0
        if plan.get("bounded_output_shift") is True:
            _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, args)
            return
        if bool(plan.get("dynamic_ab_tensormaps")):
            _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, args)
            workspace_name = _plan_str_value(plan, "ab_tensormaps_arg")
            tensormap_count = 3 if bool(plan.get("dynamic_d_tensormap")) else 2
        elif bool(plan.get("dynamic_d_tensormap")):
            workspace_name = _plan_str_value(plan, "d_tensormaps_arg")
            tensormap_count = 1
        if workspace_name is not None:
            workspace = _tcgen05_grouped_dynamic_tensormap_workspace(
                cute_kernel,
                device=layout.device,
                tensormap_count=tensormap_count,
                num_sm_multiplier=cast("int", plan.get("num_sm_multiplier", 1)),
            )
            append_wrapper_tensor(workspace_name, workspace, owned=True)

    for plan in _tcgen05_grouped_static_plans(cute_kernel):
        layout = _tcgen05_grouped_static_layout_arg(plan, args)
        if _tcgen05_grouped_device_split_sizes(plan):
            _validate_tcgen05_grouped_tensor_devices(layout, args)
            _validate_tcgen05_grouped_device_split_sizes(plan, layout)
            append_grouped_tensormap_workspace(plan, layout)
            total_name = _plan_str_value(plan, "total_clusters_arg")
            schema.append(("wrapper_host_scalar", total_name, "int"))
            launch_args.append(_tcgen05_grouped_device_split_total_clusters(plan))
            continue

        metadata_entry = _build_tcgen05_grouped_static_metadata(cute_kernel, plan, args)
        metadata_result = metadata_entry.result
        problem_tensor = metadata_result.problem_sizes
        starts_tensor = metadata_result.starts
        real_groups_tensor = metadata_result.real_groups
        runtime_tile_records_tensor = metadata_result.runtime_tile_records
        direct_pointers_tensor = metadata_result.direct_pointers
        direct_strides_tensor = metadata_result.direct_strides
        total_clusters = metadata_result.total_clusters
        grouped_static_metadata.append(metadata_entry)
        if problem_tensor is not None and starts_tensor is not None:
            append_wrapper_tensor(
                _plan_str_value(plan, "problem_sizes_arg"), problem_tensor
            )
            append_wrapper_tensor(_plan_str_value(plan, "starts_arg"), starts_tensor)
        if real_groups_tensor is not None:
            append_wrapper_tensor(
                _plan_str_value(plan, "real_groups_arg"), real_groups_tensor
            )
        if runtime_tile_records_tensor is not None:
            append_wrapper_tensor(
                _plan_str_value(plan, "runtime_tile_records_arg"),
                runtime_tile_records_tensor,
                runtime_leading_extent=True,
            )
        append_grouped_tensormap_workspace(plan, layout)
        if direct_pointers_tensor is not None and direct_strides_tensor is not None:
            append_wrapper_tensor(
                _plan_str_value(plan, "direct_pointers_arg"),
                direct_pointers_tensor,
            )
            append_wrapper_tensor(
                _plan_str_value(plan, "direct_strides_arg"),
                direct_strides_tensor,
            )
        total_name = _plan_str_value(plan, "total_clusters_arg")
        schema.append(("wrapper_host_scalar", total_name, "int"))
        launch_args.append(total_clusters)
        owned_tensors.extend(metadata_result.tensors())

    if descriptor.split_k_plan is not None:
        plan = descriptor.split_k_plan
        splits, tile_elems = (
            int(cast("int", plan["splits"])),
            int(cast("int", plan["tile_elems"])),
        )
        param = _plan_str_value(plan, "param")
        tiles = int(cast("int", plan["tiles"]))
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        # The kernel baked the counter offset from its tile count: the grid
        # must be exactly that many tiles times the split.
        if (
            not tensor_args
            or splits <= 1
            or tile_elems <= 0
            or tiles <= 0
            or grid[0] != tiles * splits
        ):
            raise exc.BackendUnsupported("cute", "invalid tcgen05 split-K launch")
        workspace = _tcgen05_split_k_workspace(
            cute_kernel,
            device=tensor_args[0].device,
            tiles=tiles,
            splits=splits,
            tile_elems=tile_elems,
        )
        schema.append(("wrapper_host_scalar", param, "int"))
        launch_args.append(int(workspace.data_ptr()))
        owned_tensors.append(workspace)

    if descriptor.fixup_plan is not None:
        plan = descriptor.fixup_plan
        tiles, rounds, slabs, rows = (
            int(cast("int", plan["tiles"])),
            int(cast("int", plan["rounds"])),
            int(cast("int", plan["slabs"])),
            int(cast("int", plan["rows"])),
        )
        param = _plan_str_value(plan, "param")
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if not tensor_args or tiles <= 0 or grid[0] % tiles:
            raise exc.BackendUnsupported("cute", "invalid mma_dag fixup launch")
        workspace = _mma_dag_fixup_workspace(
            cute_kernel,
            device=tensor_args[0].device,
            row_tiles=grid[0] // tiles,
            tiles=tiles,
            rounds=rounds,
            slabs=slabs,
            rows=rows,
        )
        schema.append(("wrapper_host_scalar", param, "int"))
        launch_args.append(int(workspace.data_ptr()))
        owned_tensors.append(workspace)

    if descriptor.chunk_prepare_plan is not None:
        plan = descriptor.chunk_prepare_plan
        storage, descriptors = build_chunk_prepare_tensor_maps(plan, args)
        desc_args = plan.get("desc_args")
        if not isinstance(desc_args, (list, tuple)) or len(desc_args) != len(
            descriptors
        ):
            raise exc.BackendUnsupported(
                "cute", "invalid chunk-prepare descriptor arguments"
            )
        for name, address in zip(desc_args, descriptors, strict=True):
            if not isinstance(name, str):
                raise exc.BackendUnsupported(
                    "cute", "invalid chunk-prepare descriptor argument name"
                )
            schema.append(("wrapper_host_scalar", name, "int"))
            launch_args.append(address)
        owned_tensors.append(storage)

    if descriptor.chunk_recurrence_plan is not None:
        plan = descriptor.chunk_recurrence_plan
        storage, descriptors = build_chunk_recurrence_tensor_maps(plan, args)
        desc_args = plan.get("desc_args")
        if not isinstance(desc_args, (list, tuple)) or len(desc_args) != len(
            descriptors
        ):
            raise exc.BackendUnsupported(
                "cute", "invalid chunk-recurrence descriptor arguments"
            )
        for name, address in zip(desc_args, descriptors, strict=True):
            if not isinstance(name, str):
                raise exc.BackendUnsupported(
                    "cute", "invalid chunk-recurrence descriptor argument name"
                )
            schema.append(("wrapper_host_scalar", name, "int"))
            launch_args.append(address)
        owned_tensors.append(storage)

    if descriptor.sm100_recurrence_plan is not None:
        # Reuse the strict packed-workspace and alias checks while the SM100
        # path is brought up.  It constructs no raw descriptor storage here;
        # the nested CuTe host creates TensorMaps from the validated views.
        chunk_recurrence_tensor_map_specs(
            descriptor.sm100_recurrence_plan, args, validate_only=True
        )

    if descriptor.gdn_recurrence_plan is not None:
        # The nested CuTe host builds its TMA descriptors from the baked tensor
        # views, so the runtime tensors must match the compiled geometry.
        validate_gdn_recurrence_launch_args(descriptor.gdn_recurrence_plan, args)

    alias_guard = _mma_dag_alias_guard(cute_kernel, args)
    if alias_guard is not None:
        alias_guard.check(args)

    launch_args.extend(grid)
    # The stream is intentionally NOT appended here; it is sampled fresh per
    # launch by the caller so CUDA graph capture sees the capture stream.
    return _CuteLaunchArgCacheEntry(
        schema=tuple(schema),
        launch_args=tuple(launch_args),
        grouped_static_metadata=tuple(grouped_static_metadata),
        owned_tensors=tuple(owned_tensors),
    )


_CUTE_DSL_ARCH_CACHE: dict[int, str] = {}
_CUTE_MIN_CUDA_VERSION = "13"


def _require_cuda13_for_cute() -> None:
    from ..._compat import requires_cuda_version

    if not requires_cuda_version(_CUTE_MIN_CUDA_VERSION):
        raise exc.BackendUnsupported(
            "cute",
            f"requires CUDA >= {_CUTE_MIN_CUDA_VERSION} "
            f"(found torch.version.cuda={torch.version.cuda!r})",
        )


def _ensure_cute_dsl_arch_env(args: tuple[object, ...]) -> None:
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if tensor_args:
        device = tensor_args[0].device
        if device.type != "cuda":
            return
        device_index = device.index if device.index is not None else 0
    elif not torch.cuda.is_available():
        return
    else:
        device_index = torch.cuda.current_device()
    _require_cuda13_for_cute()
    desired = _CUTE_DSL_ARCH_CACHE.get(device_index)
    if desired is None:
        if tensor_args:
            with torch.cuda.device(tensor_args[0].device):
                major, minor = torch.cuda.get_device_capability(tensor_args[0].device)
        else:
            major, minor = torch.cuda.get_device_capability()
        # CUTLASS DSL distinguishes post-Hopper arch variants such as
        # sm_90a/sm_100a, while torch.cuda.get_device_capability() only
        # returns major/minor.
        suffix = "a" if major >= 9 else ""
        desired = f"sm_{major}{minor}{suffix}"
        _CUTE_DSL_ARCH_CACHE[device_index] = desired
    if os.environ.get("CUTE_DSL_ARCH") != desired:
        os.environ["CUTE_DSL_ARCH"] = desired


def _cute_last_launch_arg_guard(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    *,
    signature: ArgSignature | None = None,
) -> _CuteLastLaunchArgGuard:
    if signature is None:
        signature = _cute_arg_signature(cute_kernel, args)
    grouped_mutation_guards: list[_CuteLastGroupedMutationGuard] = []
    for plan in _tcgen05_grouped_static_plans(cute_kernel):
        if _tcgen05_grouped_device_split_sizes(plan):
            continue
        for index in (
            _plan_int_value(plan, "layout_idx"),
            plan.get("n_sizes_idx"),
            plan.get("k_sizes_idx"),
        ):
            if (
                isinstance(index, int)
                and index < len(args)
                and isinstance(args[index], torch.Tensor)
            ):
                tensor = args[index]
                assert isinstance(tensor, torch.Tensor)
                grouped_mutation_guards.append(
                    _CuteLastGroupedMutationGuard(
                        index=index,
                        tensor_id=id(tensor),
                        mutation_key=_tcgen05_grouped_tensor_mutation_key(tensor),
                    )
                )
    return _CuteLastLaunchArgGuard(
        arg_count=len(args),
        grid=grid,
        bake_tensor_shapes=_cute_bake_tensor_shapes_guard(cute_kernel),
        arg_guards=signature.guards(),
        grouped_mutation_guards=tuple(grouped_mutation_guards),
        grouped_launch_contexts=_cute_grouped_launch_contexts(cute_kernel, args),
    )


def _cute_maybe_build_fastpath(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
    launch: _CuteLaunchArgCacheEntry,
    compiled: object,
    *,
    signature: ArgSignature | None = None,
) -> None:
    """Build (once per kernel) the fast relaunch state after a successful
    slow-path launch.  ``False`` marks a permanent probe failure so the
    slow path is not re-probed on every call.

    Not under CUDA-graph capture: a fresh compile's first launch there would
    JIT the C-API twin (``capi_function``, tens of milliseconds) and load its
    library inside the capture.  The graph keeps the marshalled launch just
    issued and the state stays unset; the next eager launch that misses the
    last-launch cache (same-pointer calls keep hitting it) builds it.
    """
    if getattr(cast("Any", cute_kernel), "_helion_cute_fastpath", None) is not None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    state = _cute_build_fast_relaunch(
        cute_kernel,
        args,
        grid,
        block,
        compile_options,
        launch,
        compiled,
        signature=signature,
    )
    cast("Any", cute_kernel)._helion_cute_fastpath = (
        state if state is not None else False
    )


def _cute_build_fast_relaunch(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
    launch: _CuteLaunchArgCacheEntry,
    compiled: object,
    *,
    signature: ArgSignature | None = None,
) -> _CuteFastRelaunch | None:
    if not isinstance(compiled, _CompiledCuteLauncher):
        return None
    if cast("Any", compiled)._compiled is None:
        return None
    # Kernels with grouped-scheduler plans, dynamic tensormaps, or launcher-
    # owned side tensors have per-call state beyond (pointers, stream);
    # leave them on the full path.
    if launch.owned_tensors or launch.grouped_static_metadata:
        return None
    descriptor = cute_launch_descriptor(cute_kernel)
    if descriptor.grouped_static_plans:
        return None
    if _cute_dynamic_tensormap_contexts(cute_kernel, args):
        return None
    # The whole-root plans that pass host-encoded TensorMap descriptors stay
    # on the pointer-keyed launch cache (``FASTPATH_EXCLUDED_KINDS``).
    if descriptor.fastpath_excluded:
        return None
    if signature is None:
        signature = _cute_arg_signature(cute_kernel, args, validate=False)
    device_index: int | None = None
    tensors: list[tuple[int, torch.Tensor, int]] = []
    for fact in signature.tensors:
        if fact.device_type != "cuda":
            return None
        if device_index is None:
            device_index = fact.device_index
        alignment = _cute_schema_pointer_alignment(launch.schema[fact.index])
        if fact.data_ptr % alignment != 0:
            return None
        tensors.append((fact.index, cast("torch.Tensor", args[fact.index]), alignment))
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
        # Resolved after the structural checks: a fresh TVM-FFI compile JITs
        # its C-API twin here, which is wasted on kernels that cannot relaunch.
        dsl_fn = cast("Any", compiled).capi_function()
        if dsl_fn is None:
            return None
        execution_args = getattr(dsl_fn, "execution_args", None)
        executor = getattr(dsl_fn, "_default_executor", None)
        if execution_args is None or not hasattr(
            execution_args, "generate_execution_args"
        ):
            return None
        if executor is None:
            executor = dsl_fn.to(None)
        if not hasattr(executor, "run_compiled_program"):
            return None
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        gmem_space, make_ptr_obj, _current_stream_obj = _get_cute_launcher_imports()
        make_ptr = cast("Any", make_ptr_obj)
        raw0 = int(torch._C._cuda_getCurrentRawStream(device_index))
        # Clone the pointer entries into fastpath-owned objects: pointer
        # args marshal by reference into per-object cached ctypes cells,
        # and the per-call patch pokes those cells — they must not be
        # shared with the slow path's ``launch.launch_args`` objects.
        orig_base = tuple(launch.launch_args)
        base_ptr_positions = [
            i
            for i, entry in enumerate(orig_base)
            if not isinstance(entry, (int, float, bool))
        ]
        if len(base_ptr_positions) != len(tensors):
            return None
        own_base = list(orig_base)
        for k, (_arg_index, tensor, alignment) in enumerate(tensors):
            own_base[base_ptr_positions[k]] = make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                int(tensor.data_ptr()),
                gmem_space,
                assumed_align=alignment,
            )
        base = tuple(own_base)
        stream_a = cuda_driver.CUstream(raw0)
        exe1, adapted1 = execution_args.generate_execution_args((*base, stream_a), {})
        exe2, _adapted2 = execution_args.generate_execution_args((*base, stream_a), {})

        def norm(value: object) -> object:
            return value.value if isinstance(value, ctypes.c_void_p) else value

        n1 = [norm(v) for v in exe1]
        n2 = [norm(v) for v in exe2]
        if len(n2) != len(n1):
            return None
        # Scalar args (eps, grid dims, ...) marshal into FRESH ctypes cells
        # on every call, so their exe_args entries are unstable addresses
        # while their contents are fixed (the metadata guard pins every
        # scalar value).  Restrict slot detection to the slots that are
        # stable across two identical marshals; unstable slots are reused
        # from ``exe1`` (their storage stays alive via ``adapted1``).
        stable = [a == b for a, b in zip(n1, n2, strict=True)]

        def deref(addr: object) -> int | None:
            # Only dereference values that plausibly ARE host heap
            # addresses (ctypes cell storage): 8-aligned and above the
            # low canonical range.  Large by-value integers (e.g. tensor
            # extents) must never be dereferenced — from_address on a
            # non-address segfaults uncatchably.
            if (
                isinstance(addr, int)
                and addr > _CUTE_HOST_CELL_ADDRESS_FLOOR
                and addr % 8 == 0
            ):
                return int(ctypes.c_uint64.from_address(addr).value)
            return None

        # --- Tensor-pointer slots: re-marshal with per-tensor shifted
        # pointer clones and locate each tensor's slot by its shifted
        # value.  Handles both marshalling conventions: by value (the slot
        # holds the device pointer itself) and by reference (the slot holds
        # the address of a per-object ctypes cell containing it).
        alt_base = list(base)
        shifts: list[int] = []
        for k, (_arg_index, tensor, alignment) in enumerate(tensors):
            shift = 512 * (k + 1)
            shifts.append(shift)
            alt_base[base_ptr_positions[k]] = make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                int(tensor.data_ptr()) + shift,
                gmem_space,
                assumed_align=alignment,
            )
        exe4, _adapted4 = execution_args.generate_execution_args(
            (*tuple(alt_base), stream_a), {}
        )
        n4 = [norm(v) for v in exe4]
        if len(n4) != len(n1):
            return None
        tensor_slots: list[tuple[int, int | None, object | None]] = []
        for k, (arg_index, tensor, _alignment) in enumerate(tensors):
            ptr = int(tensor.data_ptr())
            want = ptr + shifts[k]
            by_val = [
                i for i, v in enumerate(n4) if stable[i] and v == want and n1[i] == ptr
            ]
            if len(by_val) == 1:
                tensor_slots.append((arg_index, by_val[0], None))
                continue
            by_ref = [
                i
                for i in range(len(n4))
                if stable[i] and deref(n1[i]) == ptr and deref(n4[i]) == want
            ]
            if len(by_ref) == 1:
                tensor_slots.append(
                    (
                        arg_index,
                        None,
                        ctypes.c_uint64.from_address(cast("int", n1[by_ref[0]])),
                    )
                )
                continue
            return None

        # --- Stream slot(s): re-marshal with a different (never-launched)
        # handle.
        alt_raw = raw0 + 0x40
        stream_b = cuda_driver.CUstream(alt_raw)
        exe3, _adapted3 = execution_args.generate_execution_args((*base, stream_b), {})
        n3 = [norm(v) for v in exe3]
        if len(n3) != len(n1):
            return None
        slots = [
            i
            for i, (a, b) in enumerate(zip(n1, n3, strict=True))
            if stable[i] and a != b
        ]
        # The wrapper takes ONE stream parameter; its handle can surface in
        # at most a couple of exe slots (a by-value copy plus a by-ref
        # cell).  More differing slots means the marshalling isn't the
        # shape we probe-verified — fall back.
        if not slots or len(slots) > 2:
            return None
        by_ref_writers: list[object] = []
        by_val_slots: list[int] = []
        for i in slots:
            a, b = n1[i], n3[i]
            if a == raw0 and b == alt_raw:
                by_val_slots.append(i)
                continue
            # By-reference: the slot holds the address of an 8-byte cell
            # containing the handle.  Verify BOTH probes' cells before
            # trusting the address.
            if deref(a) == raw0 and deref(b) == alt_raw:
                by_ref_writers.append(ctypes.c_uint64.from_address(cast("int", a)))
                continue
            return None

        # --- Metadata guards (no pointer equality).
        constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
        tensor_guards: list[FastpathTensorGuard] = [
            fact.fastpath_guard(
                _cute_schema_pointer_alignment(launch.schema[fact.index])
            )
            for fact in signature.tensors
        ]
        scalar_guards: list[_CuteLastScalarArgGuard] = [
            fact.guard() for fact in signature.scalars
        ]
        exe_args = list(exe1)
        return _CuteFastRelaunch(
            executor=executor,
            exe_args=exe_args,
            tensor_guards=tuple(tensor_guards),
            scalar_guards=tuple(scalar_guards),
            constexpr_flags=tuple(constexpr_flags),
            tensor_slots=tuple(tensor_slots),
            by_ref_writers=by_ref_writers,
            by_val_slots=by_val_slots,
            arg_count=len(args),
            grid=grid,
            block=block,
            compile_options=compile_options,
            device_index=device_index,
            last_raw=raw0,
            keepalive=(base, stream_a, adapted1, exe1),
            packed_launch=cast(
                "_CutePackedLaunch | None", _cute_packed_launch(executor, exe_args)
            ),
            # The metadata guards above pin the spans the alias guard
            # computed from this launch's tensors.
            alias_guard=_mma_dag_alias_guard(cute_kernel, args),
            cxx_guards=_cute_cxx_tensor_guards(args, tensor_guards),
        )
    except Exception:
        return None


def _cute_last_launch_cache_entry(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
) -> _CuteLastLaunchCacheEntry | None:
    entry = getattr(cast("Any", cute_kernel), "_helion_cute_last_launch_cache", None)
    if not isinstance(entry, _CuteLastLaunchCacheEntry):
        return None
    if not entry.arg_guard.matches(cute_kernel, args, grid):
        return None
    discriminator = _cute_compiled_launcher_discriminator(
        entry.launch.schema,
        block,
        compile_options,
        args,
        getattr(cast("Any", cute_kernel), "_helion_cute_cluster_shape", None),
    )[0]
    if discriminator != entry.compiled_discriminator:
        return None
    if not _cute_grouped_static_metadata_matches(
        entry.launch.grouped_static_metadata, cute_kernel, args
    ):
        return None
    return entry


def _set_cute_last_launch_cache_entry(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
    launch: _CuteLaunchArgCacheEntry,
    compiled: object,
    *,
    signature: ArgSignature | None = None,
) -> None:
    arg_guard = _cute_last_launch_arg_guard(
        cute_kernel, args, grid, signature=signature
    )
    compiled_discriminator = _cute_compiled_launcher_discriminator(
        launch.schema,
        block,
        compile_options,
        args,
        getattr(cast("Any", cute_kernel), "_helion_cute_cluster_shape", None),
    )[0]
    cast("Any", cute_kernel)._helion_cute_last_launch_cache = _CuteLastLaunchCacheEntry(
        arg_guard=arg_guard,
        compiled_discriminator=compiled_discriminator,
        launch=launch,
        compiled=compiled,
    )


def default_cute_launcher(
    cute_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> object:
    _ensure_cuda_context()
    pending = prepare_capture_launch(cute_kernel, args, grid)
    result = _default_cute_launcher_untracked(cute_kernel, grid, *args, **kwargs)
    if pending is not None:
        pending.finish()
    return result


def _default_cute_launcher_untracked(
    cute_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> object:
    # The relaunch of a kernel launched before, compared as the host code
    # spells the launch (every tensor and scalar argument still guarded).
    fastpath = getattr(cute_kernel, "_helion_cute_fastpath", None)
    if fastpath.__class__ is _CuteFastRelaunch:
        hit, result = fastpath.try_launch_raw(args, grid, kwargs)
        if hit:
            return result
    return _default_cute_launcher_normalized(
        cute_kernel, grid, args, kwargs, dict(kwargs)
    )


def _cute_record_raw_launch(
    cute_kernel: object,
    grid: tuple[int, ...],
    raw_kwargs: dict[str, object],
    grid_xyz: tuple[int, int, int],
    block_xyz: tuple[int, int, int],
    compile_options: str | None,
) -> None:
    """Record how the host code spells the launch the fast relaunch just
    served or was built for (``try_launch_raw`` compares against it) -- only
    when this call's normalized grid, block and options are the fastpath's:
    a concurrent first launch with another grid must not leave its spelling
    on a fastpath pinned to a different one."""
    fastpath = getattr(cast("Any", cute_kernel), "_helion_cute_fastpath", None)
    if (
        fastpath.__class__ is _CuteFastRelaunch
        and fastpath.raw_grid is None
        and fastpath.grid == grid_xyz
        and fastpath.block == block_xyz
        and fastpath.compile_options == compile_options
    ):
        # kwargs first: a raw check that sees the grid must see them too.
        fastpath.raw_kwargs = raw_kwargs
        fastpath.raw_grid = grid


def _default_cute_launcher_normalized(
    cute_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    raw_kwargs: dict[str, object],
) -> object:
    block = kwargs.pop("block", (256, 1, 1))
    cute_compile_options = kwargs.pop("cute_compile_options", None)
    if cute_compile_options is not None and not isinstance(cute_compile_options, str):
        raise ValueError(f"Invalid CuTe compile options: {cute_compile_options!r}")
    if not isinstance(block, tuple) or len(block) < 1:
        raise ValueError(f"Invalid block specification: {block}")
    if not isinstance(grid, tuple) or len(grid) < 1:
        raise ValueError(f"Invalid grid specification: {grid}")
    if kwargs:
        raise exc.BackendUnsupported("cute", f"launcher kwargs: {sorted(kwargs)}")

    grid_xyz = (
        int(grid[0]),
        int(grid[1]) if len(grid) > 1 else 1,
        int(grid[2]) if len(grid) > 2 else 1,
    )
    block_xyz = (
        int(block[0]),
        int(block[1]) if len(block) > 1 else 1,
        int(block[2]) if len(block) > 2 else 1,
    )

    if grid_xyz[0] <= 0 or grid_xyz[1] <= 0 or grid_xyz[2] <= 0:
        return None

    args_tuple = tuple(args)
    # Metadata-guarded fast relaunch: skips the pointer-keyed caches AND the
    # DSL's per-call marshalling entirely (fresh output allocations change
    # tensor pointers on every call in real workloads, so pointer-keyed
    # caching alone still pays the full marshalling cost each time).
    fastpath = getattr(cast("Any", cute_kernel), "_helion_cute_fastpath", None)
    if isinstance(fastpath, _CuteFastRelaunch):
        hit, result = fastpath.try_launch(
            args_tuple, grid_xyz, block_xyz, cute_compile_options
        )
        if hit:
            _cute_record_raw_launch(
                cute_kernel, grid, raw_kwargs, grid_xyz, block_xyz, cute_compile_options
            )
            return result
    # Under-aligned TMA/vector operands of the staged plan kinds never reach
    # the compiled wrapper: the fast path's alignment guard rejected them
    # above, and the copies below are what the slower paths see and cache.
    staging = _cute_stage_under_aligned_operands(cute_kernel, args_tuple)
    if staging is None:
        result = _launch_cute_marshalled(
            cute_kernel, args_tuple, grid_xyz, block_xyz, cute_compile_options
        )
    else:
        staged_args, copy_backs = staging
        result = _launch_cute_marshalled(
            cute_kernel, staged_args, grid_xyz, block_xyz, cute_compile_options
        )
        for destination, copy in copy_backs:
            destination.copy_(copy)
    # The marshalled launch may have built the fastpath for this launch.
    _cute_record_raw_launch(
        cute_kernel, grid, raw_kwargs, grid_xyz, block_xyz, cute_compile_options
    )
    return result


def _launch_cute_marshalled(
    cute_kernel: object,
    args_tuple: tuple[object, ...],
    grid_xyz: tuple[int, int, int],
    block_xyz: tuple[int, int, int],
    cute_compile_options: str | None,
) -> object:
    """Launch through the last-launch cache or the schema-cached build path."""
    last_launch = _cute_last_launch_cache_entry(
        cute_kernel,
        args_tuple,
        grid_xyz,
        block_xyz,
        cute_compile_options,
    )
    if last_launch is not None:
        _retain_cute_capture_owned_launch_tensors(
            cute_kernel,
            grouped_launch_contexts=last_launch.arg_guard.grouped_launch_contexts,
            owned_tensors=last_launch.launch.owned_tensors,
        )
        _record_cute_owned_launch_tensors(last_launch.launch.owned_tensors)
        return cast("Any", last_launch.compiled)(
            *last_launch.launch.launch_args,
            _cute_current_stream(),
        )

    signature = _cute_arg_signature(cute_kernel, args_tuple)
    launch = _build_cached_cute_schema_and_args(
        cute_kernel, args_tuple, grid_xyz, signature=signature
    )
    compiled = _get_compiled_cute_launcher(
        cute_kernel,
        launch.schema,
        block_xyz,
        compile_options=cute_compile_options,
        arch_args=args_tuple,
    )
    _record_cute_owned_launch_tensors(launch.owned_tensors)
    # Append the CUDA stream fresh on every launch (never cached): under CUDA
    # graph capture the current stream is the capture stream, so the kernel must
    # be issued there and not on a stale stream baked into the cached args.
    result = cast("Any", compiled)(*launch.launch_args, _cute_current_stream())
    _set_cute_last_launch_cache_entry(
        cute_kernel,
        args_tuple,
        grid_xyz,
        block_xyz,
        cute_compile_options,
        launch,
        compiled,
        signature=signature,
    )
    # Build the metadata-guarded fast relaunch once the compiled artifact
    # exists (first successful launch); later calls with fresh output
    # allocations then skip the marshalling path entirely.
    _cute_maybe_build_fastpath(
        cute_kernel,
        args_tuple,
        grid_xyz,
        block_xyz,
        cute_compile_options,
        launch,
        compiled,
        signature=signature,
    )
    return result
