"""CuTe DSL (cutlass) runtime: launcher, wrapper codegen, and launch caching.

Extracted from ``helion/runtime/__init__.py``; the cutedsl-specific launch path
lives here so the package root stays backend-neutral. ``helion.runtime`` re-exports
the public entry points (``default_cute_launcher``, ``cute_cuda_graph``) plus the
handful of helpers that tests import, for backward compatibility. ``cutlass`` is
imported lazily inside the functions that need it, so importing this module never
requires a CuTe/cutlass install.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from contextlib import contextmanager
from contextlib import suppress
import contextvars
import ctypes
from dataclasses import dataclass
from dataclasses import field
import hashlib
import importlib
import inspect
import json
import linecache
import logging
import os
import sys
import threading
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
from ..._compiler.cute.grouped_worklist import GroupedWorklistRows
from ..._compiler.cute.grouped_worklist import Tcgen05GroupedWorklistValidationError
from ..._compiler.cute.grouped_worklist import (
    _tcgen05_grouped_worklist_rows_from_flattened,
)
from ..._compiler.cute.grouped_worklist import (
    tcgen05_grouped_worklist_compatible_source_m_tiles,
)
from ..._compiler.cute.grouped_worklist import validate_tcgen05_grouped_worklist_rows
from ..._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from ..._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from ..._compiler.cute.strategies import tcgen05_smem_layout_expr
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS,
)
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_RUNTIME_TILE_FIELD_COUNT
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from ..._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
)
from ..._compiler.cute.tcgen05_constants import TCGEN05_GROUPED_WORKLIST_STORE_SHAPE
from ..._compiler.cute.tcgen05_constants import Tcgen05GroupedRuntimeTileField
from ..triton.launcher import get_num_sm

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from torch.cuda import _POOL_HANDLE

log: logging.Logger = logging.getLogger(__name__)


_CUTLASS_SHUTDOWN_PATCHED = False


def _patch_cutlass_jit_shutdown_unload() -> None:
    """Avoid CUDA library unload hangs during interpreter shutdown.

    On current CUTLASS DSL builds, ``CudaDialectJitModule.__del__`` unconditionally
    calls ``cudaLibraryUnload``. On B200 this can hang during Python finalization
    after a CuTe kernel has already finished executing. Skipping that unload during
    interpreter teardown lets the process exit cleanly while preserving the normal
    unload path during regular runtime GC.
    """

    global _CUTLASS_SHUTDOWN_PATCHED
    if _CUTLASS_SHUTDOWN_PATCHED:
        return

    try:
        import cutlass.cutlass_dsl.cuda_jit_executor as cuda_jit_executor
    except ImportError:
        return

    module_type = cuda_jit_executor.CudaDialectJitModule
    if getattr(module_type, "_helion_shutdown_patch", False):
        _CUTLASS_SHUTDOWN_PATCHED = True
        return

    original_del = cast("Any", module_type.__del__)

    def _helion_del(self: object) -> None:
        module = cast("Any", self)
        if sys.is_finalizing():
            with suppress(Exception):
                module._unloaded = True
            return
        original_del(module)

    module_type.__del__ = _helion_del
    module_type._helion_shutdown_patch = True
    _CUTLASS_SHUTDOWN_PATCHED = True


_TORCH_DTYPE_TO_CUTLASS: dict[torch.dtype, object] | None = None


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> object:
    global _TORCH_DTYPE_TO_CUTLASS
    mapping: dict[torch.dtype, object] | None = _TORCH_DTYPE_TO_CUTLASS
    if mapping is None:
        _patch_cutlass_jit_shutdown_unload()
        import cutlass

        mapping = {
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
            torch.float64: cutlass.Float64,
            torch.bfloat16: cutlass.BFloat16,
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.float8_e5m2: cutlass.Float8E5M2,
            torch.float4_e2m1fn_x2: cutlass.Uint8,
            # CuTe does not support i1 global-memory tensors; torch.bool is
            # stored as one byte, so pass bool tensor pointers as uint8 and
            # let load lowering convert nonzero bytes back to cutlass.Boolean
            # registers.
            torch.bool: cutlass.Uint8,
            torch.int8: cutlass.Int8,
            torch.int16: cutlass.Int16,
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
            torch.uint8: cutlass.Uint8,
            torch.uint32: cutlass.Uint32,
            torch.uint64: cutlass.Int64,
        }
        _TORCH_DTYPE_TO_CUTLASS = mapping
    cutlass_dtype = mapping.get(dtype)
    if cutlass_dtype is None:
        raise exc.BackendUnsupported("cute", f"dtype: {dtype}")
    return cutlass_dtype


def _normalize_cute_scalar(arg: object) -> tuple[str, object]:
    if isinstance(arg, (bool, torch.SymBool)):
        return ("bool", bool(arg))
    if isinstance(arg, (int, torch.SymInt)):
        return ("int", int(arg))
    if isinstance(arg, (float, torch.SymFloat)):
        return ("float", float(arg))
    raise exc.BackendUnsupported("cute", f"launcher scalar argument type: {type(arg)}")


def _cute_scalar_annotation(kind: str) -> str:
    mapping = {
        "bool": "cutlass.Boolean",
        "int": "cutlass.Int64",
        "float": "cutlass.Float32",
    }
    return mapping[kind]


def _cute_kernel_param_is_constexpr(cute_kernel: object) -> tuple[bool, ...]:
    """Return per-parameter Constexpr flags for a ``@cute.kernel``.

    Cached on the kernel object to avoid repeated signature inspection.
    The newer cutlass DSL (>=4.5) enforces region isolation: a runtime scalar
    passed through the wrapper cannot satisfy a kernel parameter declared as
    ``cutlass.Constexpr``.  When the wrapper sees a Constexpr-typed kernel
    parameter, it must propagate the value as a Constexpr (i.e., baked into
    the compiled wrapper) rather than as a runtime ``cutlass.Int64``.
    """
    cached = getattr(cast("Any", cute_kernel), "_helion_cute_param_constexpr", None)
    if cached is not None:
        return cast("tuple[bool, ...]", cached)
    import cutlass

    try:
        sig = inspect.signature(cute_kernel)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        flags: tuple[bool, ...] = ()
    else:
        from typing import get_origin
        from typing import get_type_hints

        # Helion-emitted kernels use ``from __future__ import annotations`` so
        # ``param.annotation`` is the source string. ``get_type_hints`` resolves
        # those strings against the function's globals (which include
        # ``cutlass``).
        try:
            hints = get_type_hints(cute_kernel)  # type: ignore[arg-type]
        except Exception:
            hints = {}
        flags_list: list[bool] = []
        for name, param in sig.parameters.items():
            ann = hints.get(name, param.annotation)
            is_constexpr = ann is cutlass.Constexpr or get_origin(ann) is (
                cutlass.Constexpr
            )
            flags_list.append(is_constexpr)
        flags = tuple(flags_list)
    with suppress(AttributeError, TypeError):
        cast("Any", cute_kernel)._helion_cute_param_constexpr = flags
    return flags


def _tcgen05_grouped_dynamic_ab_tensormap_rank(plan: dict[str, object]) -> int:
    rank = plan.get("dynamic_ab_tensormap_rank", 3)
    if (
        not isinstance(rank, int)
        or rank not in (2, 3)
        or (
            rank == 2
            and not bool(plan.get("dynamic_ab_tensormaps"))
            and not bool(plan.get("fixed_ab_tensormaps"))
            and not bool(plan.get("fixed_tensormaps"))
        )
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped A/B TensorMap rank must be 2 or 3, and rank 2 "
            "requires dynamic or fixed full-allocation A/B TensorMaps",
        )
    return rank


def _tcgen05_grouped_scheduler_mode(
    plan: dict[str, object],
) -> Tcgen05GroupedSchedulerMode:
    value = plan.get("scheduler_mode", Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH)
    mode: Tcgen05GroupedSchedulerMode | None = None
    if isinstance(value, Tcgen05GroupedSchedulerMode):
        mode = value
    elif isinstance(value, str):
        with suppress(ValueError):
            mode = Tcgen05GroupedSchedulerMode(value)
    if mode is None:
        choices = ", ".join(
            candidate.value for candidate in Tcgen05GroupedSchedulerMode
        )
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped scheduler mode must be one of: {choices}",
        )
    uses_runtime_table = mode in (
        Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
        Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
    )
    has_runtime_table = isinstance(plan.get("runtime_tile_records_arg"), str)
    if uses_runtime_table != has_runtime_table:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped runtime scheduler state requires matching tile-table metadata",
        )
    if uses_runtime_table and _tcgen05_plan_orientation(plan) != "nm":
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped runtime scheduling requires N,M orientation"
        )
    if mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC and not bool(
        plan.get("fixed_tensormaps")
    ):
        raise exc.BackendUnsupported(
            "cute", "grouped runtime CLC requires fixed full-allocation TensorMaps"
        )
    return mode


def _append_cute_wrapper_plan(
    body: list[str],
    call_args: list[str],
    plan: dict[str, object],
    num_sm: int | None = None,
) -> None:
    def plan_int(key: str, default: int | None = None) -> int:
        value = plan.get(key, default) if default is not None else plan[key]
        assert isinstance(value, int)
        return value

    def plan_optional_int(key: str) -> int | None:
        value = plan.get(key)
        assert value is None or isinstance(value, int)
        return value

    def plan_optional_str(key: str) -> str | None:
        value = plan.get(key)
        assert value is None or isinstance(value, str)
        return value

    def plan_optional_order(key: str) -> tuple[int, ...] | None:
        value = plan.get(key)
        if value is None:
            return None
        assert isinstance(value, (list, tuple))
        assert all(type(dim) is int for dim in value)
        order = tuple(int(dim) for dim in value)
        assert sorted(order) == list(range(len(order)))
        return order

    def append_permuted_cute_tensor_view(
        name: str,
        arg_idx: int,
        order: tuple[int, ...],
    ) -> None:
        shape = ", ".join(f"arg{arg_idx}_shape{dim}" for dim in order)
        stride = ", ".join(f"arg{arg_idx}_stride{dim}" for dim in order)
        body.append(
            f"    {name} = cute.make_tensor("
            f"arg{arg_idx}.iterator, "
            f"layout=cute.make_layout(({shape}), stride=({stride})))"
        )

    def require_positive_int(value: int | None, name: str) -> int:
        assert type(value) is int, name
        assert value > 0, name
        return value

    def append_tcgen05_epilogue_tma_wrapper(
        *,
        tensor_idx: int,
        bm: int,
        bn: int,
        stage_count: int,
        dtype: str,
        kernel_args: list[str],
        copy_op: str,
        epi_tile_m: int | None = None,
        epi_tile_n: int | None = None,
        d_store_box_n: int | None = None,
        epi_tile_raw_expr: str | None = None,
        tensor_name: str | None = None,
        rank3_mnl_tensor: bool = False,
        orientation: str = "mn",
        column_major: bool = False,
    ) -> None:
        assert len(kernel_args) == 2
        assert orientation in ("mn", "nm")
        worklist_nm_store = orientation == "nm"
        d_store_layout = (
            "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
            if worklist_nm_store or column_major
            else "cutlass.utils.layout.LayoutEnum.ROW_MAJOR"
        )
        tensor_expr = tensor_name if tensor_name is not None else f"arg{tensor_idx}"
        explicit_epi_tile = any(
            value is not None for value in (epi_tile_m, epi_tile_n, d_store_box_n)
        )
        if epi_tile_raw_expr is not None:
            # The bm=128 CtaGroup.TWO family threads the device-exact (N-mode
            # permuted) epilogue-tile expression verbatim so the host TMA-store
            # atom is built from the same layout the device r2s copy writes
            # through. The plain ``epi_tile_m/n`` integer keys cannot express
            # the permutation. See ``tcgen05_two_cta_m128_epilogue_tile_expr``.
            assert not explicit_epi_tile
            epi_tile_expr = epi_tile_raw_expr
        elif explicit_epi_tile:
            checked_epi_tile_m = require_positive_int(epi_tile_m, "epi_tile_m")
            checked_epi_tile_n = require_positive_int(epi_tile_n, "epi_tile_n")
            checked_d_store_box_n = require_positive_int(d_store_box_n, "d_store_box_n")
            assert checked_epi_tile_n == checked_d_store_box_n
            epi_tile_expr = tcgen05_explicit_d_store_tile_expr(
                checked_epi_tile_m, checked_d_store_box_n
            )
        else:
            epi_tile_expr = tcgen05_default_epilogue_tile_expr(
                bm,
                bn,
                dtype,
                c_layout=d_store_layout,
            )
        tma_atom, tma_tensor = kernel_args
        epi_tile = f"{tma_atom}_epi_tile"
        smem_layout = f"{tma_atom}_smem_layout"
        cta_v_layout = f"{tma_atom}_cta_v_layout"
        gmem_tensor = (
            f"{tma_atom}_d_nm"
            if worklist_nm_store and rank3_mnl_tensor
            else f"{tma_atom}_gmem_mnl"
            if rank3_mnl_tensor
            else tensor_expr
        )
        cta_tiler_expr = (
            epi_tile
            if rank3_mnl_tensor
            else f"cute.composition(cute.make_identity_layout({gmem_tensor}.shape), {epi_tile})"
        )
        rank3_gmem_shape = (
            f"(arg{tensor_idx}_shape1, arg{tensor_idx}_shape0, 1)"
            if worklist_nm_store
            else f"(arg{tensor_idx}_shape0, arg{tensor_idx}_shape1, 1)"
        )
        rank3_gmem_stride = (
            f"(arg{tensor_idx}_stride1, arg{tensor_idx}_stride0, 0)"
            if worklist_nm_store
            else f"(arg{tensor_idx}_stride0, arg{tensor_idx}_stride1, 0)"
        )
        # Keep these layout arguments in sync with the device-side
        # ``make_smem_layout_epi`` calls; the wrapper's TMA atom and the kernel's
        # SMEM staging must slice the same epilogue tile shape.
        body.extend(
            (
                f"    {epi_tile} = {epi_tile_expr}",
                (
                    f"    {smem_layout} = cutlass.utils.blackwell_helpers."
                    "make_smem_layout_epi("
                    f"{dtype}, {d_store_layout}, "
                    f"{epi_tile}, {stage_count})"
                ),
                *(
                    (
                        (
                            f"    {gmem_tensor} = cute.make_tensor("
                            f"arg{tensor_idx}.iterator, "
                            "layout=cute.make_layout("
                            f"{rank3_gmem_shape}, stride={rank3_gmem_stride}))"
                        ),
                    )
                    if rank3_mnl_tensor
                    else ()
                ),
                (f"    {cta_v_layout} = {cta_tiler_expr}"),
                (
                    f"    {tma_atom}, {tma_tensor} = "
                    "cute.nvgpu.cpasync.make_tiled_tma_atom("
                    f"{copy_op}, "
                    f"{gmem_tensor}, cute.slice_({smem_layout}, (None, None, 0)), "
                    f"{cta_v_layout})"
                ),
            )
        )
        call_args.extend(kernel_args)

    kind = plan["kind"]
    if kind == "helion_small_biased_attention":
        batch = plan_int("batch")
        seq = plan_int("seq")
        body.extend(
            [
                f"    grid_x = cutlass.Int32({seq})",
                f"    grid_y = cutlass.Int32({batch})",
                "    grid_z = cutlass.Int32(1)",
            ]
        )
        return

    if kind == "helion_flash":
        # Fused tcgen05 flash-attention host setup: reorder Helion's (B, S, D)
        # tensors to the reference (S, D, B) / (D, S, B) layouts, build the two
        # tiled_mma (QK from SMEM, PV with OperandSource.TMEM) and the three TMA
        # atoms, then append all kernel args. This mirrors the standalone
        # 3D-batched host setup validated for the specialized flash path.
        q_idx = plan_int("q_idx")
        k_idx = plan_int("k_idx")
        v_idx = plan_int("v_idx")
        o_idx = plan_int("o_idx")
        lse_idx = plan_optional_int("lse_idx")
        bias_idx = plan_optional_int("bias_idx")
        alibi_idx = plan_optional_int("alibi_idx")
        document_idx = plan_optional_int("document_idx")
        seq = plan_int("seq")
        head_dim = plan_int("head_dim")
        batch = plan_int("batch")
        scale_log2 = plan["scale_log2"]
        assert isinstance(scale_log2, float)
        score_bias_scale = plan.get("score_bias_scale", 0.0)
        assert isinstance(score_bias_scale, float)
        alibi_count = plan_int("alibi_count", default=batch)
        document_batch = plan_int("document_batch", default=batch)
        document_heads_per_batch = plan_int("document_heads_per_batch", default=1)
        kv_stage = plan_int("kv_stage")
        q_stage = plan_int("q_stage", default=1)
        use_2cta_instrs = bool(plan.get("use_2cta_instrs"))
        use_cga2_local_cta = bool(plan.get("use_cga2_local_cta"))
        use_clc_scheduler = bool(plan.get("use_clc_scheduler"))
        cluster_m = 2 if use_2cta_instrs or use_cga2_local_cta else 1
        num_kv = (seq + 127) // 128
        # Static-persistent scheduler: total_tiles = num_bh * num_m_tiles (the
        # flat tile-id space the device-body strided while loop walks). When
        # persistent, the host clamps grid_x down to min(total_tiles, num_SMs)
        # so each SM gets one CTA that strides over many work tiles.
        persistent = bool(plan.get("persistent"))
        total_tiles = plan_int("total_tiles", default=batch * (seq // 128))
        pass_dynamic_tile_counts = plan.get("topology") != "fa4"
        hd = head_dim
        dtype = str(plan.get("dtype", "cutlass.Float16"))
        assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
        tensor_4d_batch = plan_int("tensor_4d_batch", default=0)
        tensor_4d_heads = plan_int("tensor_4d_heads", default=0)
        use_tensor_4d_tma = (
            tensor_4d_batch > 0
            and tensor_4d_heads > 0
            and tensor_4d_batch * tensor_4d_heads == batch
        )
        # (S, D, B) views over the existing (B, S, D) row-major buffers. The
        # dense FA4 4D-TMA knob instead treats the same flat storage as
        # (S, D, H, Z), matching FA4's tensor-map rank for contiguous q[z,h,s,d].
        bw = "cutlass.utils.blackwell_helpers"
        mma_m = 256 if use_2cta_instrs else 128
        qkd = f"({mma_m}, 128, {hd})"
        pvd = f"({mma_m}, {hd}, 128)"
        if use_tensor_4d_tma:
            bh_stride = seq * hd
            batch_stride = tensor_4d_heads * bh_stride
            sdb = (
                f"cute.make_layout(({seq}, {hd}, {tensor_4d_heads}, "
                f"{tensor_4d_batch}), stride=({hd}, 1, {bh_stride}, "
                f"{batch_stride}))"
            )
            dsb = (
                f"cute.make_layout(({hd}, {seq}, {tensor_4d_heads}, "
                f"{tensor_4d_batch}), stride=(1, {hd}, {bh_stride}, "
                f"{batch_stride}))"
            )
        else:
            sdb = (
                f"cute.make_layout(({seq}, {hd}, {batch}), "
                f"stride=({hd}, 1, {seq * hd}))"
            )
            dsb = (
                f"cute.make_layout(({hd}, {seq}, {batch}), "
                f"stride=(1, {hd}, {seq * hd}))"
            )
        ssb = (
            f"cute.make_layout(({seq}, {seq}, {batch}), stride=({seq}, 1, {seq * seq}))"
        )
        sb = f"cute.make_layout(({seq}, {batch}), stride=(1, {seq}))"
        majk = "cute.nvgpu.OperandMajorMode.K"
        cg1 = "cute.nvgpu.tcgen05.CtaGroup.ONE"
        cg = "cute.nvgpu.tcgen05.CtaGroup.TWO" if use_2cta_instrs else cg1
        sel = "cute.select"
        flash_lines = [
            f"_flash_mQ = cute.make_tensor(arg{q_idx}.iterator, {sdb})",
            f"_flash_mK = cute.make_tensor(arg{k_idx}.iterator, {sdb})",
            # V is MN-major: (D, S, B).
            f"_flash_mV = cute.make_tensor(arg{v_idx}.iterator, {dsb})",
            f"_flash_mO = cute.make_tensor(arg{o_idx}.iterator, {sdb})",
            f"_flash_qk_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg}, ({mma_m}, 128))",
            f"_flash_pv_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, cute.nvgpu.OperandMajorMode.MN, cutlass.Float32, {cg}, ({mma_m}, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
            f"_flash_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(({2 if use_2cta_instrs else 1}, 1, 1)), (_flash_qk_mma.thr_id.shape,))",
            f"_flash_qsl = {bw}.make_smem_layout_a(_flash_qk_mma, {qkd}, {dtype}, {q_stage})",
            # K/V are multi-stage TMA rings (Stage 3); the stage count must match
            # the device-body kv_stage + the SharedStorage MemRange depths.
            f"_flash_ksl = {bw}.make_smem_layout_b(_flash_qk_mma, {qkd}, {dtype}, {kv_stage})",
            f"_flash_vsl = {bw}.make_smem_layout_b(_flash_pv_mma, {pvd}, {dtype}, {kv_stage})",
            f"_flash_ptl = {bw}.make_smem_layout_a(_flash_pv_mma, {pvd}, {dtype}, 1)",
            f"_flash_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg})",
            f"_flash_tma_q, _flash_mQt = cute.nvgpu.make_tiled_tma_atom_A(_flash_op, _flash_mQ, {sel}(_flash_qsl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape)",
            f"_flash_tma_k, _flash_mKt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mK, {sel}(_flash_ksl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape)",
            f"_flash_tma_v, _flash_mVt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mV, {sel}(_flash_vsl, mode=[0, 1, 2]), {pvd}, _flash_pv_mma, _flash_cluster_layout_vmnk.shape)",
            f"_flash_scale_log2 = cutlass.Float32({scale_log2!r})",
            f"_flash_num_kv_tiles = cutlass.Int32({num_kv})",
        ]
        if bias_idx is not None:
            flash_lines.extend(
                [
                    f"_flash_mBias = cute.make_tensor(arg{bias_idx}.iterator, {ssb})",
                    f"_flash_score_bias_scale = cutlass.Float32({score_bias_scale!r})",
                ]
            )
        if alibi_idx is not None:
            flash_lines.extend(
                [
                    (
                        f"_flash_mAlibi = cute.make_tensor(arg{alibi_idx}.iterator, "
                        f"cute.make_layout(({alibi_count},), stride=(1,)))"
                    ),
                    f"_flash_num_alibi = cutlass.Int32({alibi_count})",
                ]
            )
        if document_idx is not None:
            sdoc = f"cute.make_layout(({seq}, {document_batch}), stride=(1, {seq}))"
            flash_lines.extend(
                [
                    f"_flash_mDoc = cute.make_tensor(arg{document_idx}.iterator, {sdoc})",
                    (
                        "_flash_doc_heads_per_batch = "
                        f"cutlass.Int32({document_heads_per_batch})"
                    ),
                ]
            )
        if pass_dynamic_tile_counts:
            flash_lines.extend(
                [
                    f"_flash_num_bh = cutlass.Int32({batch})",
                    f"_flash_total_tiles = cutlass.Int32({total_tiles})",
                ]
            )
        if lse_idx is not None:
            flash_lines.append(
                f"_flash_mLSE = cute.make_tensor(arg{lse_idx}.iterator, {sb})"
            )
        epi_tma = bool(plan.get("epi_tma"))
        epi_stg = bool(plan.get("epi_stg"))
        if epi_tma or epi_stg:
            # Build the O smem layout for epilogue-warp store paths. The TMA
            # variant also builds the O TMA STORE atom; the STG variant reuses
            # the layout but stores with a universal-copy tiled copy in device code.
            otile = f"(128, {hd})"
            flash_lines.extend(
                [
                    (
                        f"_flash_osl = {bw}.make_smem_layout_epi("
                        f"{dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {otile}, 2)"
                    ),
                ]
            )
            if epi_tma:
                flash_lines.extend(
                    [
                        (
                            f"_flash_o_cta_v = cute.composition("
                            f"cute.make_identity_layout(_flash_mO.shape), {otile})"
                        ),
                        (
                            "_flash_tma_o, _flash_mOt = "
                            "cute.nvgpu.cpasync.make_tiled_tma_atom("
                            "cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _flash_mO, "
                            "cute.select(_flash_osl, mode=[0, 1]), _flash_o_cta_v)"
                        ),
                    ]
                )
            else:
                flash_lines.append("_flash_mOt = _flash_mO")
        else:
            # mO stays the (S, D, B) view (no TMA atom; the epilogue uses
            # autovec_copy straight to gmem).
            flash_lines.append("_flash_mOt = _flash_mO")
        body.extend(f"    {line}" for line in flash_lines)
        if use_clc_scheduler:
            # CLC launches the full problem grid; the device starts from blockIdx
            # and uses cluster launch control to dynamically steal remaining work.
            clc_heads = plan_int("clc_heads_per_batch", batch)
            if clc_heads <= 0 or batch % clc_heads != 0:
                clc_heads = batch
            body.extend(
                [
                    f"    grid_x = cutlass.Int32({total_tiles // batch})",
                    f"    grid_y = cutlass.Int32({clc_heads})",
                    f"    grid_z = cutlass.Int32({batch // clc_heads})",
                ]
            )
        elif persistent:
            # Cap the flat grid at num_SMs (computed host-side from the q tensor's
            # device at wrapper-build time and baked as a literal). grid_y/grid_z
            # stay 1 (already true for the flat flash grid). The device-body
            # strided while loop then covers all total_tiles work items.
            assert num_sm is not None and num_sm > 0
            ctas_per_sm = max(1, plan_int("persistent_ctas_per_sm", 1))
            max_ctas = ((num_sm * ctas_per_sm) // cluster_m) * cluster_m
            grid_cap = min(total_tiles * cluster_m, max_ctas)
            body.append(f"    grid_x = cutlass.Int32({grid_cap})")
        elif plan.get("topology") == "fa4":
            # The fa4 topology processes a PAIR of adjacent 128-row Q-tiles per
            # CTA, so it needs exactly total_tiles (= batch * seq // 256) CTAs.
            # The default root grid would launch batch * seq // 128; override it
            # to the halved fa4 tile count.
            body.append(f"    grid_x = cutlass.Int32({total_tiles * cluster_m})")
        call_args.extend(
            [
                "_flash_qk_mma",
                "_flash_pv_mma",
                "_flash_tma_q",
                "_flash_mQt",
                "_flash_tma_k",
                "_flash_mKt",
                "_flash_tma_v",
                "_flash_mVt",
                "_flash_mOt",
                "_flash_qsl",
                "_flash_ksl",
                "_flash_vsl",
                "_flash_ptl",
                "_flash_scale_log2",
                "_flash_num_kv_tiles",
            ]
        )
        if pass_dynamic_tile_counts:
            call_args.extend(["_flash_num_bh", "_flash_total_tiles"])
        if lse_idx is not None:
            call_args.append("_flash_mLSE")
        if bias_idx is not None:
            call_args.extend(["_flash_mBias", "_flash_score_bias_scale"])
        if alibi_idx is not None:
            call_args.extend(["_flash_mAlibi", "_flash_num_alibi"])
        if document_idx is not None:
            call_args.extend(["_flash_mDoc", "_flash_doc_heads_per_batch"])
        if epi_tma:
            call_args.extend(["_flash_tma_o", "_flash_osl"])
        elif epi_stg:
            call_args.append("_flash_osl")
        return
    if kind == "tcgen05_d_tma":
        d_idx = plan_int("d_idx")
        bm = plan_int("bm")
        bn = plan_int("bn")
        c_stage_count = plan_int("c_stage_count")
        output_dtype = str(plan["output_dtype"])
        kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
        d_tensor_name = None
        if bool(plan.get("d_leading_passthrough")):
            d_tensor_name = f"{kernel_args[0]}_d_tma"
            append_permuted_cute_tensor_view(d_tensor_name, d_idx, (1, 2, 0))
        append_tcgen05_epilogue_tma_wrapper(
            tensor_idx=d_idx,
            bm=bm,
            bn=bn,
            stage_count=c_stage_count,
            dtype=output_dtype,
            kernel_args=kernel_args,
            copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()",
            epi_tile_m=plan_optional_int("epi_tile_m"),
            epi_tile_n=plan_optional_int("epi_tile_n"),
            d_store_box_n=plan_optional_int("d_store_box_n"),
            epi_tile_raw_expr=plan_optional_str("epi_tile_raw_expr"),
            tensor_name=d_tensor_name,
            rank3_mnl_tensor=bool(plan.get("rank3_mnl_tensor")),
            orientation=_tcgen05_plan_orientation(plan),
            column_major=bool(plan.get("d_column_major")),
        )
        return
    if kind == "tcgen05_aux_tma":
        c_idx = plan_int("c_idx")
        bm = plan_int("bm")
        bn = plan_int("bn")
        stage_count = plan_int("stage_count")
        input_dtype = str(plan["input_dtype"])
        kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
        append_tcgen05_epilogue_tma_wrapper(
            tensor_idx=c_idx,
            bm=bm,
            bn=bn,
            stage_count=stage_count,
            dtype=input_dtype,
            kernel_args=kernel_args,
            copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()",
        )
        return
    if kind == "tcgen05_grouped_static_persistent":
        assert num_sm is not None and num_sm > 0
        sched_params_arg = str(plan["sched_params_arg"])
        total_clusters_arg = str(plan["total_clusters_arg"])
        cluster_m = plan_int("cluster_m", 1)
        cluster_n = plan_int("cluster_n", 1)
        reserved_sms = plan_int(TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY, 0)
        max_active_clusters = _tcgen05_grouped_static_active_clusters(
            num_sm=num_sm,
            cluster_m=max(1, cluster_m),
            reserved_sms=reserved_sms,
        )
        quota_args = tuple(
            str(arg)
            for arg in cast(
                "tuple[object, ...]", plan.get("static_group_quota_args", ())
            )
        )
        quotas: tuple[int, ...] = ()
        scheduler_mode = _tcgen05_grouped_scheduler_mode(plan)
        runtime_direct_tile_table = scheduler_mode in (
            Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
            Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
        )
        runtime_direct_clc = scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC
        if quota_args:
            problem_shapes = cast(
                "tuple[tuple[int, int, int], ...]", plan["static_problem_shapes"]
            )
            assert len(quota_args) == len(problem_shapes)
            assert _tcgen05_plan_orientation(plan) != "nm"
            assert cluster_m == 1 and cluster_n == 1
            quotas = _tcgen05_grouped_static_quotas(
                problem_shapes,
                bm=plan_int("bm"),
                bn=plan_int("bn"),
                bk=plan_int("bk"),
                max_active_clusters=max_active_clusters,
            )
        if runtime_direct_clc:
            # The host table is exact: one real output record per z cluster, with
            # no padded/phantom tail. Launching the full request grid lets CLC
            # cancel each pending z cluster at most once and makes response bidz
            # the record index consumed identically by every CTA in the cluster.
            body.extend(
                (
                    f"    grid_x = cutlass.Int32({cluster_m})",
                    f"    grid_y = cutlass.Int32({cluster_n})",
                    f"    grid_z = {total_clusters_arg}",
                )
            )
        else:
            body.append(
                f"    {sched_params_arg} = cutlass.utils.PersistentTileSchedulerParams("
                f"({cluster_m}, {cluster_n}, {total_clusters_arg}), "
                f"({cluster_m}, {cluster_n}, 1))"
            )
            body.extend(
                (
                    (
                        "    _tcgen05_grouped_grid = "
                        "cutlass.utils.StaticPersistentGroupTileScheduler.get_grid_shape("
                        f"{sched_params_arg}, cutlass.Int32({max_active_clusters}))"
                    ),
                    "    grid_x = _tcgen05_grouped_grid[0]",
                    "    grid_y = _tcgen05_grouped_grid[1]",
                    "    grid_z = _tcgen05_grouped_grid[2]",
                )
            )
            if not runtime_direct_tile_table:
                call_args.append(sched_params_arg)
        if runtime_direct_tile_table:
            call_args.append(total_clusters_arg)
        call_args.extend(f"cutlass.Int32({quota})" for quota in quotas)
        return
    if kind != "tcgen05_ab_tma":
        raise exc.BackendUnsupported("cute", f"wrapper plan kind: {kind}")

    lhs_idx = plan_int("lhs_idx")
    rhs_idx = plan_int("rhs_idx")
    bm = plan_int("bm")
    bn = plan_int("bn")
    bk = plan_int("bk")
    cluster_m = plan_int("cluster_m", 1)
    cluster_n = plan_int("cluster_n", 1)
    input_dtype = str(plan["input_dtype"])
    acc_dtype = str(plan["acc_dtype"])
    ab_stage_count = plan_int("ab_stage_count", 2)
    # Optional ``smem_swizzle_*`` overrides recorded by the device-side
    # codegen when the user opts into a non-default A/B SMEM atom
    # swizzle. When absent the wrapper emits the legacy
    # ``make_smem_layout_a/b`` calls. The no-override wrapper markers
    # are covered by the focused tcgen05 SMEM-swizzle codegen test.
    smem_swizzle_a_raw = plan.get("smem_swizzle_a")
    smem_swizzle_b_raw = plan.get("smem_swizzle_b")
    smem_swizzle_a: int | None = (
        int(smem_swizzle_a_raw) if isinstance(smem_swizzle_a_raw, int) else None
    )
    smem_swizzle_b: int | None = (
        int(smem_swizzle_b_raw) if isinstance(smem_swizzle_b_raw, int) else None
    )
    a_k_major = bool(plan.get("a_k_major", True))
    # K-major (column-major / K-contiguous) B. Absent on the MN-major
    # (row-major B) default path.
    b_k_major = bool(plan.get("b_k_major"))
    lhs_tma_order = plan_optional_order("lhs_tma_order")
    rhs_tma_order = plan_optional_order("rhs_tma_order")
    rhs_rank3_grouped_nt = bool(plan.get("rhs_rank3_grouped_nt"))
    lhs_rank3_grouped_nt = bool(plan.get("lhs_rank3_grouped_nt"))
    orientation = _tcgen05_plan_orientation(plan)
    swapped_nm = orientation == "nm"
    dynamic_ab_tensormaps = bool(plan.get("dynamic_ab_tensormaps"))
    fixed_ab_tensormaps = bool(plan.get("fixed_ab_tensormaps"))
    fixed_grouped_b_rank3 = bool(plan.get("fixed_grouped_b_rank3"))
    if fixed_grouped_b_rank3 and not fixed_ab_tensormaps:
        raise exc.BackendUnsupported(
            "cute", "rank-3 immutable grouped B requires fixed A/B TensorMaps"
        )
    if fixed_ab_tensormaps and (not swapped_nm or dynamic_ab_tensormaps):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation A/B TensorMaps require the N,M worklist "
            "orientation and cannot also be dynamic",
        )
    if swapped_nm and not (dynamic_ab_tensormaps or fixed_ab_tensormaps):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 N,M-oriented A/B TensorMaps require dynamic per-group or "
            "fixed full-allocation descriptors",
        )
    dynamic_ab_tensormap_rank2 = (
        dynamic_ab_tensormaps and _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) == 2
    )
    kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
    assert len(kernel_args) == 4
    tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = kernel_args

    # CtaGroup.TWO is selected when ``cluster_m == 2 and bm == 256`` —
    # the V=2 path. ``cluster_n`` extends the cluster along the N axis
    # but does not change the V dimension. Cycle 26's
    # ``cluster_m * cluster_n == 2`` test happened to work for
    # cluster_m=2 cluster_n=1 but rejects the canonical Quack-best
    # cluster_m=2 cluster_n=2 4-CTA cluster (product=4). Use
    # ``cluster_m == 2`` directly so cluster_n=2 keeps CtaGroup.TWO.
    #
    # The bm=128 CtaGroup.TWO family (fp8 small-grid) cannot be derived from
    # ``bm == 256`` here, so the device codegen records the resolved decision
    # on the plan as ``use_2cta_instrs``. Honor it when present; fall back to
    # the legacy bm==256 derivation for golden-stable older plans.
    plan_use_2cta = plan.get("use_2cta_instrs")
    if plan_use_2cta is not None:
        assert isinstance(plan_use_2cta, bool)
        use_2cta_instrs = plan_use_2cta
    else:
        use_2cta_instrs = cluster_m == 2 and bm == 256
    cta_group = (
        "cute.nvgpu.tcgen05.CtaGroup.TWO"
        if use_2cta_instrs
        else "cute.nvgpu.tcgen05.CtaGroup.ONE"
    )
    cluster_shape = f"({cluster_m}, {cluster_n}, 1)"
    tiled_mma = f"{tma_atom_a}_tiled_mma"
    cluster_layout_vmnk = f"{tma_atom_a}_cluster_layout_vmnk"
    smem_a_layout = f"{tma_atom_a}_smem_layout"
    smem_b_layout = f"{tma_atom_b}_smem_layout"
    lhs_tma = f"{tma_atom_a}_lhs_tma"
    lhs_tma_arg = (
        lhs_tma
        if (
            dynamic_ab_tensormaps
            or fixed_ab_tensormaps
            or swapped_nm
            or lhs_tma_order is not None
        )
        else f"arg{lhs_idx}"
    )
    rhs_tma = f"{tma_atom_b}_rhs_tma"
    if swapped_nm:
        if not lhs_rank3_grouped_nt:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 N,M-oriented A/B TensorMaps require grouped rank-3 logical A",
            )
        if fixed_ab_tensormaps:
            if fixed_grouped_b_rank3:
                lhs_tma_layout = (
                    f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2, "
                    f"arg{lhs_idx}_shape0), "
                    f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2, "
                    f"arg{lhs_idx}_stride0)"
                )
            else:
                # K-major grouped B can flatten to one immutable [G*N,K]
                # allocation. Device coordinates add the group base.
                lhs_tma_layout = (
                    f"(arg{lhs_idx}_shape0 * arg{lhs_idx}_shape1, "
                    f"arg{lhs_idx}_shape2), stride=(arg{lhs_idx}_stride1, "
                    f"arg{lhs_idx}_stride2)"
                )
        elif dynamic_ab_tensormap_rank2:
            lhs_tma_layout = (
                f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2), "
                f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2)"
            )
        else:
            lhs_tma_layout = (
                f"(arg{lhs_idx}_shape1, arg{lhs_idx}_shape2, "
                f"arg{lhs_idx}_shape0), "
                f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride2, "
                f"arg{lhs_idx}_stride0)"
            )
        if (
            fixed_ab_tensormaps
            or dynamic_ab_tensormap_rank2
            or not dynamic_ab_tensormaps
        ):
            rhs_tma_layout = (
                f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1), "
                f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1)"
            )
        else:
            rhs_tma_layout = (
                f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1, 1), "
                f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1, 0)"
            )
        lhs_tma_setup = (
            (
                f"    {lhs_tma} = cute.make_tensor("
                f"arg{lhs_idx}.iterator, "
                f"layout=cute.make_layout({lhs_tma_layout}))"
            ),
        )
        rhs_tma_setup = (
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                f"layout=cute.make_layout({rhs_tma_layout}))"
            ),
        )
    else:
        lhs_tma_setup = (
            (
                (
                    f"    {lhs_tma} = cute.make_tensor("
                    f"arg{lhs_idx}.iterator, "
                    "layout=cute.make_layout("
                    + (
                        f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1), "
                        f"stride=(arg{lhs_idx}_stride0, arg{lhs_idx}_stride1)))"
                        if dynamic_ab_tensormap_rank2
                        else (
                            f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1, 1), "
                            f"stride=(arg{lhs_idx}_stride0, "
                            f"arg{lhs_idx}_stride1, 0)))"
                        )
                    )
                ),
            )
            if dynamic_ab_tensormaps
            else ()
        )
        rhs_tma_setup = (
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                + (
                    f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2), "
                    f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2)))"
                    if dynamic_ab_tensormap_rank2
                    else (
                        f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2, "
                        f"arg{rhs_idx}_shape0), "
                        f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2, "
                        f"arg{rhs_idx}_stride0)))"
                    )
                )
                if rhs_rank3_grouped_nt
                else f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape0), "
                f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride0)))"
            ),
        )
    smem_a_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="a",
        swizzle_override=smem_swizzle_a,
        k_major=a_k_major,
    )
    smem_b_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="b",
        swizzle_override=smem_swizzle_b,
        k_major=b_k_major,
    )
    if lhs_tma_order is not None:
        append_permuted_cute_tensor_view(lhs_tma, lhs_idx, lhs_tma_order)
    if rhs_tma_order is not None:
        append_permuted_cute_tensor_view(rhs_tma, rhs_idx, rhs_tma_order)
    lhs_tma_setup_lines = () if lhs_tma_order is not None else lhs_tma_setup
    rhs_tma_setup_lines = () if rhs_tma_order is not None else rhs_tma_setup
    body.extend(
        (
            (
                f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
                f"{input_dtype}, "
                f"{input_dtype}, "
                + (
                    "cute.nvgpu.OperandMajorMode.K, "
                    if a_k_major
                    else "cute.nvgpu.OperandMajorMode.MN, "
                )
                + (
                    "cute.nvgpu.OperandMajorMode.K, "
                    if b_k_major
                    else "cute.nvgpu.OperandMajorMode.MN, "
                )
                + f"{acc_dtype}, "
                f"{cta_group}, "
                f"({bm}, {bn}), "
                "cute.nvgpu.tcgen05.OperandSource.SMEM)"
            ),
            (
                f"    {cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout({cluster_shape}), ({tiled_mma}.thr_id.shape,))"
            ),
            f"    {smem_a_layout} = {smem_a_layout_expr}",
            f"    {smem_b_layout} = {smem_b_layout_expr}",
            *lhs_tma_setup_lines,
            *rhs_tma_setup_lines,
            # ``make_tiled_tma_atom_A`` vs ``_B`` asymmetry:
            # - ``_B`` always passes ``cluster_layout_vmnk.shape`` as
            #   its trailing arg (CuTe's signature for B requires the
            #   cluster shape; the cluster_m=1 cluster_n=1 case still
            #   passes the 1x1x1 shape harmlessly).
            # - ``_A`` only adds the same trailing arg when
            #   ``cluster_n > 1``. For the validated cluster_n=1
            #   paths, A's atom is constructed without the cluster
            #   shape while B still receives it. The asymmetry is
            #   intentional: A only needs the cluster shape when N
            #   multicast is active (cluster_n>1). The cluster_n=1
            #   form is pinned by
            #   ``test_tcgen05_role_local_monolithic_codegen_markers``.
            (
                f"    {tma_atom_a}, {tma_tensor_a} = cute.nvgpu.make_tiled_tma_atom_A("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"{lhs_tma_arg}, "
                f"cute.slice_({smem_a_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}"
                + (f", {cluster_layout_vmnk}.shape" if cluster_n > 1 else "")
                + ")"
            ),
            # See the asymmetry comment above ``make_tiled_tma_atom_A``
            # for why ``_B`` always passes the cluster shape and ``_A``
            # only does at cluster_n>1.
            (
                f"    {tma_atom_b}, {tma_tensor_b} = cute.nvgpu.make_tiled_tma_atom_B("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_B("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"{rhs_tma}, "
                f"cute.slice_({smem_b_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}, {cluster_layout_vmnk}.shape)"
            ),
        )
    )
    call_args.extend(kernel_args)


def _cute_cluster_shape_from_wrapper_plans(
    wrapper_plans: list[dict[str, object]],
) -> tuple[int, int, int] | None:
    cluster_m = 1
    cluster_n = 1
    for plan in wrapper_plans:
        if plan.get("kind") != "tcgen05_ab_tma":
            continue
        plan_cluster_m = plan.get("cluster_m", 1)
        plan_cluster_n = plan.get("cluster_n", 1)
        assert isinstance(plan_cluster_m, int)
        assert isinstance(plan_cluster_n, int)
        cluster_m = max(cluster_m, plan_cluster_m)
        cluster_n = max(cluster_n, plan_cluster_n)
    if cluster_m * cluster_n <= 1:
        return None
    return (cluster_m, cluster_n, 1)


def _cute_cluster_shape(
    cute_kernel: object, wrapper_plans: list[dict[str, object]]
) -> tuple[int, int, int] | None:
    explicit_cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    if explicit_cluster_shape is not None:
        if (
            isinstance(explicit_cluster_shape, tuple)
            and len(explicit_cluster_shape) == 3
            and all(isinstance(dim, int) for dim in explicit_cluster_shape)
        ):
            return cast("tuple[int, int, int]", explicit_cluster_shape)
        raise exc.BackendUnsupported(
            "cute",
            f"invalid _helion_cute_cluster_shape: {explicit_cluster_shape!r}",
        )
    return _cute_cluster_shape_from_wrapper_plans(wrapper_plans)


def _create_cute_wrapper(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    num_sm: int | None = None,
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
            if len(entry) == 5:
                # ("tensor", dtype, rank, sizes, strides) — baked layout.
                # Wrapper plans (matmul TMA) also reference
                # ``arg{i}_shape{d}`` / ``arg{i}_stride{d}`` names, so we
                # bind those names to their literal values in the wrapper
                # body before constructing the tensor.
                (_, _dtype, rank, sizes_t, strides_t) = entry
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
            (_, _dtype, rank) = entry
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
            (_, name, _dtype, rank, sizes_t, strides_t) = entry
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
            (_, name, _dtype, rank, tail_sizes, strides) = entry
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
    for plan in wrapper_plans:
        _append_cute_wrapper_plan(body, call_args, plan, num_sm=num_sm)
    launch_suffix = f", block={block!r}"
    cluster_shape = _cute_cluster_shape(cute_kernel, wrapper_plans)
    if cluster_shape is not None:
        launch_suffix += f", cluster={list(cluster_shape)!r}"
    # G2-H (cute_plan.md, see plan: G2-H CLC): CLC kernels need PDL
    # enabled at the host launch so ``nvvm.clusterlaunchcontrol_try_cancel``
    # returns valid responses. ``use_pdl`` is set on the per-matmul
    # wrapper plan in ``cute_mma._codegen_cute_mma`` when
    # ``Tcgen05PersistenceModel.CLC_PERSISTENT`` is active. Reading
    # from the plan rather than a kernel-level side-channel attribute
    # mirrors how ``cluster_m``/``cluster_n`` flow through this layer.
    if any(plan.get("use_pdl") for plan in wrapper_plans):
        launch_suffix += ", use_pdl=True"
    # The fa4 flash topology (16-warp/512-thread) uses ``cute.arch.setmaxregister``
    # for per-warp register reallocation (softmax warps inc to 200; mma/corr/load/empty
    # dec). ptxas only emits the ``EIATTR_REG_RECONFIG`` that HONORS those ``setmaxnreg``
    # ops when the kernel declares ``min_blocks_per_mp`` (>= 1); WITHOUT it ptxas
    # SILENTLY DROPS every setmaxnreg and all warps are stuck at the static uniform
    # split -- so the softmax warp never reaches its 200-reg grant and spills its
    # resident row to local memory. fa4 already pins 1 CTA/SM (512 threads + TMEM = 1
    # tcgen05 unit/SM + smem near the cap), so ``min_blocks_per_mp=1`` matches its real
    # occupancy and enables the reallocation (=1 avoids the smem-carveout path >1 would
    # trigger). NOT applied to ws_overlap (256-thread): forcing 1 CTA/SM there cuts its
    # 2-blocks/SM occupancy and regresses it ~4pp.
    explicit_min_blocks = getattr(
        cast("Any", cute_kernel), "_helion_cute_min_blocks_per_mp", None
    )
    if isinstance(explicit_min_blocks, int) and explicit_min_blocks > 0:
        launch_suffix += f", min_blocks_per_mp={explicit_min_blocks}"
    elif any(plan.get("topology") == "fa4" for plan in wrapper_plans):
        launch_suffix += ", min_blocks_per_mp=1"
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
    return namespace[func_name]


class _CompiledCuteLauncher:
    """Lazily compile a Helion ``@cute.jit`` wrapper via ``cute.compile``.

    The first call uses ``cute.compile(jit_func, *args)`` to produce a compiled
    callable; subsequent calls invoke the compiled callable directly. This
    bypasses the per-launch ``@cute.jit`` argument-handling/dispatch path,
    matching Quack's pattern (see ``gemm_tvm_ffi_utils.py``). On B200 this
    collapses ~200ms of per-launch host overhead into ~0.1ms.

    When ``cache_key`` is provided, the lowered IR module of the compiled
    kernel is persisted under ``CUTE_DSL_CACHE_DIR`` and reloaded on a later
    process, skipping recompilation.  ``cute.compile`` forces the CuTe DSL's
    own ``no_cache=True`` path, so Helion drives the on-disk cache itself: it
    writes the post-pass ``ir_module`` bytecode (plus a small JSON sidecar
    holding the mangled entry symbol) and, on a hit, reconstructs a runnable
    ``CudaDialectJitCompiledFunction`` by JIT-loading the stored module.
    Any failure in the cache layer falls back to a plain ``cute.compile``.
    """

    __slots__ = ("_cache_key", "_compile_options", "_compiled", "_jit_func")

    def __init__(
        self,
        jit_func: object,
        compile_options: str | None,
        cache_key: str | None = None,
    ) -> None:
        self._jit_func = jit_func
        self._compile_options = compile_options
        self._compiled: object = None
        self._cache_key = cache_key

    def __call__(self, *args: object) -> object:
        compiled = self._compiled
        if compiled is None:
            import cutlass.cute as cute

            compiled = None
            if self._cache_key is not None:
                compiled = self._reload_from_disk()
            if compiled is None:
                if self._compile_options is None:
                    compiled = cute.compile(self._jit_func, *args)
                else:
                    compiled = cute.compile(
                        self._jit_func,
                        *args,
                        options=self._compile_options,
                    )
                if self._cache_key is not None:
                    self._persist_to_disk(compiled)
            self._compiled = compiled
        return cast("Any", compiled)(*args)

    def persist_compiled(self) -> None:
        """Persist the already-compiled module into the current on-disk cache dir.

        Used by ``finalize_ephemeral_cache``: the artifact written during
        autotuning died with the ephemeral dir, but the compiled module is
        still in memory and ``_cache_file_paths`` resolves the destination
        from the (now restored) ``CUTE_DSL_CACHE_DIR`` at call time.
        """
        if self._cache_key is not None and self._compiled is not None:
            self._persist_to_disk(self._compiled)

    def _cache_file_paths(self) -> tuple[str, str, str]:
        from cutlass.base_dsl.cache_helpers import get_default_generated_ir_path

        cache_dir = get_default_generated_ir_path("CUTE_DSL")
        mlir = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.mlir")
        meta = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.json")
        return cache_dir, mlir, meta

    def _persist_to_disk(self, compiled: object) -> None:
        try:
            from cutlass.base_dsl.cache_helpers import save_ir
            from cutlass.base_dsl.cache_helpers import write_bytecode_with_crc32

            ir_module = getattr(compiled, "ir_module", None)
            function_name = getattr(compiled, "function_name", None)
            if ir_module is None or function_name is None:
                return
            cache_dir, _mlir, meta = self._cache_file_paths()
            os.makedirs(cache_dir, exist_ok=True)
            save_ir(
                "CUTE_DSL",
                ir_module,
                str(self._cache_key),
                output_dir=cache_dir,
                as_bytecode=True,
                bytecode_writer=lambda f: write_bytecode_with_crc32(f, ir_module),
            )
            # Atomic sidecar with the mangled entry symbol (process-dependent,
            # so it cannot be recomputed and must be stored alongside the IR).
            tmp = f"{meta}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "function_name": function_name,
                        "has_gpu_module": bool(
                            getattr(compiled, "has_gpu_module", True)
                        ),
                    },
                    f,
                )
            os.replace(tmp, meta)
        except (ImportError, OSError):
            # Old cutlass or an unwritable cache dir; just recompile next time.
            log.debug(
                "CuTe disk-cache persist failed for key %s",
                self._cache_key,
                exc_info=True,
            )

    def _reload_from_disk(self) -> object:
        try:
            from cutlass.base_dsl.cache_helpers import load_ir
            from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32
            from cutlass.cutlass_dsl.cuda_jit_executor import (
                CudaDialectJitCompiledFunction,
            )
            from cutlass.cutlass_dsl.cutlass import CuTeDSL

            _cache_dir, mlir, meta = self._cache_file_paths()
            if not (os.path.exists(mlir) and os.path.exists(meta)):
                return None
            with open(meta) as f:
                metadata = json.load(f)
            function_name = metadata["function_name"]
            # The parsed Module holds an internal reference to the ir.Context
            # that load_ir opened, so it stays valid after load_ir returns even
            # though its ``with ir.Context()`` block has already exited.
            _, module = load_ir(
                mlir,
                asBytecode=True,
                bytecode_reader=read_bytecode_and_check_crc32,
            )
            dsl = CuTeDSL._get_dsl()
            engine = dsl.compiler_provider.jit(
                module, shared_libs=dsl.get_shared_libs()
            )
            capi_func = engine.lookup(function_name)
            # The signature is reconstructable from the wrapper, so it does not
            # need to be persisted.
            wrapped = getattr(self._jit_func, "__wrapped__", self._jit_func)
            signature = inspect.signature(cast("Any", wrapped), eval_str=True)
            # Empty kernel_info / default extra-arg state is correct only for the
            # non-experimental ``cute.compile`` path Helion uses here; the
            # experimental DSL would populate these from module attributes.
            return CudaDialectJitCompiledFunction(
                module,
                engine,
                capi_func,
                signature,
                function_name,
                {},
                False,
                None,
                has_gpu_module=bool(metadata.get("has_gpu_module", True)),
            )
        except Exception:
            # Any cutlass-internal change or corrupt artifact -> recompile.
            return None


_TVM_FFI_COMPILE_OPTION = "--enable-tvm-ffi"
_CUTE_NUM_SM_CACHE: dict[tuple[str, int | None], int] = {}


def _merge_tvm_ffi_compile_option(compile_options: str | None) -> str:
    """Ensure ``--enable-tvm-ffi`` is present in *compile_options*.

    The generic launcher always benefits from the FFI bridge (it skips
    CUTLASS-DSL's per-arg cast/pointer work). Other flags such as
    ``--generate-line-info`` may already be present (e.g. when the
    autotuner picks ``tcgen05_cubin_lineinfo=True``), so we splice rather
    than replace.
    """
    if compile_options is None:
        return _TVM_FFI_COMPILE_OPTION
    tokens = compile_options.split()
    if _TVM_FFI_COMPILE_OPTION in tokens:
        return compile_options
    tokens.append(_TVM_FFI_COMPILE_OPTION)
    return " ".join(tokens)


def _cute_num_sm_from_arch_args(arch_args: tuple[object, ...] | None) -> int | None:
    if arch_args is None:
        return None
    for arg in arch_args:
        if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
            cache_key = (arg.device.type, arg.device.index)
            cached = _CUTE_NUM_SM_CACHE.get(cache_key)
            if cached is None:
                cached = get_num_sm(arg.device)
                _CUTE_NUM_SM_CACHE[cache_key] = cached
            return cached
    return None


def _cute_compiled_launcher_discriminator(
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    compile_options: str | None,
    arch_args: tuple[object, ...] | None,
) -> tuple[tuple[object, ...], str, int | None]:
    merged_compile_options = _merge_tvm_ffi_compile_option(compile_options)
    num_sm = _cute_num_sm_from_arch_args(arch_args)
    return (
        (
            schema_key,
            block,
            merged_compile_options,
            num_sm,
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
    # ``compile_options``.
    cache_key, compile_options, num_sm = _cute_compiled_launcher_discriminator(
        schema_key,
        block,
        compile_options,
        arch_args,
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
    cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    jit_func = _create_cute_wrapper(cute_kernel, schema_key, block, num_sm=num_sm)
    disk_cache_key = _cute_disk_cache_key(
        cute_kernel,
        schema_key,
        block,
        wrapper_plans,
        cluster_shape,
        compile_options,
        num_sm,
    )
    launcher = _CompiledCuteLauncher(
        jit_func, compile_options, cache_key=disk_cache_key
    )
    cache[cache_key] = launcher
    return launcher


def _cute_cache_relevant_env() -> tuple[tuple[str, str], ...]:
    """Return CuTe DSL env vars that can change the compiled IR.

    The CuTe DSL folds *every* one of its ``CUTE_DSL_*`` env vars into its own
    module hash (e.g. ``CUTE_DSL_ENABLE_ASSERTIONS``, ``CUTE_DSL_LINEINFO``,
    ``CUTE_DSL_KEEP``, the tvm-ffi flags), so any of them can alter the
    persisted artifact.  We snapshot the whole set (so future flags are covered
    too) and only exclude the cache *location* ``CUTE_DSL_CACHE_DIR`` — that
    selects where artifacts live (autotuning uses an ephemeral dir) and must not
    affect the key.  Including an env var that does not actually affect codegen
    only costs an occasional missed cache hit, never a wrong-kernel reload.
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in os.environ.items()
            if k.startswith("CUTE_DSL_") and k != "CUTE_DSL_CACHE_DIR"
        )
    )


def _cute_disk_cache_key(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    wrapper_plans: tuple[object, ...],
    cluster_shape: object,
    compile_options: str | None,
    num_sm: int | None = None,
) -> str | None:
    """Compute a stable cross-process key for the on-disk CuTe compile cache.

    Returns ``None`` (disabling the on-disk cache) when the generated-source
    hash is unavailable.  The key must be computable *before* the kernel is
    compiled (so a hit can skip recompilation), so it is derived from the
    inputs that determine the lowered IR rather than from the IR itself:
    generated device-kernel source, full input specialization (dtypes, ranks,
    baked shapes/strides, constexpr values), launch shape (block/cluster), CuTe
    compile options, the IR-affecting ``CUTE_DSL_*`` env vars (target SM arch
    among them), and the cutlass version.

    ``num_sm`` is the device SM count the persistent flash wrapper bakes into
    its grid clamp as a literal (``cute.compile`` lowers that literal into the
    persisted ``ir_module``).  The env-var arch capture only distinguishes the
    target *arch*, not the SM *count*, so two same-arch GPUs with different SM
    counts would otherwise collide on one on-disk artifact carrying the wrong
    grid clamp.  It is included unconditionally to match the in-memory cache
    key; for non-persistent kernels num_sm does not affect codegen, so it only
    costs an occasional cross-GPU miss, never a wrong-kernel reload.
    """
    source_hash = getattr(cute_kernel, "_helion_cute_source_hash", None)
    if source_hash is None:
        return None
    try:
        import cutlass

        cutlass_version = getattr(cutlass, "__version__", "")
    except Exception:
        cutlass_version = ""
    payload = repr(
        (
            "helion-cute-cache-v1",
            source_hash,
            schema_key,
            block,
            wrapper_plans,
            repr(cluster_shape),
            compile_options or "",
            _cute_cache_relevant_env(),
            cutlass_version,
            num_sm,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return base64.b32encode(digest).decode().rstrip("=")


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


# Keep the per-kernel launch-argument cache small: production kernels normally
# relaunch one or two stable tensor signatures, while autotune may probe many.
_CUTE_LAUNCH_ARG_CACHE_LIMIT = 8
_TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT = 8
_TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT = 8

_CuteGroupedLaunchContext = tuple[str, int | None, int, int | None]


@dataclass
class _CuteCudaGraphResources:
    cache_entries: dict[tuple[int, str, object], tuple[object, str, object, object]]
    tensors: dict[int, torch.Tensor]
    devices: set[torch.device]
    streams: set[torch.cuda.Stream]

    def retain(
        self,
        owner: object,
        attribute: str,
        key: object,
        value: object,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        self.cache_entries[(id(owner), attribute, key)] = (
            owner,
            attribute,
            key,
            value,
        )
        for tensor in tensors:
            self.tensors[id(tensor)] = tensor
            self.devices.add(tensor.device)
            stream = torch.cuda.current_stream(tensor.device)
            tensor.record_stream(stream)
            self.streams.add(stream)

    def record_replay_streams(self) -> None:
        for device in self.devices:
            stream = torch.cuda.current_stream(device)
            if stream not in self.streams:
                for tensor in self.tensors.values():
                    if tensor.device == device:
                        tensor.record_stream(stream)
                self.streams.add(stream)

    def release(self) -> None:
        for owner, attribute, key, value in self.cache_entries.values():
            cache = getattr(owner, attribute, None)
            if isinstance(cache, dict) and cache.get(key) is value:
                cache.pop(key)
        self.cache_entries.clear()
        self.tensors.clear()
        self.devices.clear()
        self.streams.clear()


class _CuteCUDAGraph(torch.cuda.CUDAGraph):
    def __init__(self) -> None:
        super().__init__()
        self._helion_resources = _CuteCudaGraphResources({}, {}, set(), set())
        self._helion_resource_finalizer = weakref.finalize(
            self, self._helion_resources.release
        )

    def replay(self) -> None:
        self._helion_resources.record_replay_streams()
        super().replay()


_CUTE_ACTIVE_CUDA_GRAPH: contextvars.ContextVar[_CuteCUDAGraph | None] = (
    contextvars.ContextVar("helion_cute_active_cuda_graph", default=None)
)


def _track_cute_cuda_graph_cache_entry(
    owner: object,
    attribute: str,
    key: object,
    value: object,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    graph = _CUTE_ACTIVE_CUDA_GRAPH.get()
    if graph is None:
        # Raw torch.cuda.graph callers are conservatively retained because the
        # CUDA capture ID does not expose the owning Python CUDAGraph lifetime.
        return
    graph._helion_resources.retain(
        owner,
        attribute,
        key,
        value,
        tensors,
    )


@contextmanager
def cute_cuda_graph(
    *,
    pool: _POOL_HANDLE | None = None,
    stream: torch.cuda.Stream | None = None,
    capture_error_mode: str = "global",
) -> Iterator[_CuteCUDAGraph]:
    """Capture CuTe launches while tying raw-pointer owners to ``graph``.

    Raw ``torch.cuda.graph`` captures cannot expose their Python graph owner to
    the launcher, so their resources are conservatively retained without bound.
    """
    graph = _CuteCUDAGraph()
    token = _CUTE_ACTIVE_CUDA_GRAPH.set(graph)
    try:
        with torch.cuda.graph(
            graph,
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
        ):
            yield graph
    finally:
        _CUTE_ACTIVE_CUDA_GRAPH.reset(token)


@dataclass(frozen=True)
class _Tcgen05GroupedStaticMetadataResult:
    problem_sizes: torch.Tensor | None
    starts: torch.Tensor | None
    total_clusters: int
    real_groups: torch.Tensor | None = None
    runtime_tile_records: torch.Tensor | None = None
    direct_pointers: torch.Tensor | None = None
    direct_strides: torch.Tensor | None = None

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *((self.problem_sizes,) if self.problem_sizes is not None else ()),
            *((self.starts,) if self.starts is not None else ()),
            *((self.real_groups,) if self.real_groups is not None else ()),
            *(
                (self.runtime_tile_records,)
                if self.runtime_tile_records is not None
                else ()
            ),
            *((self.direct_pointers,) if self.direct_pointers is not None else ()),
            *((self.direct_strides,) if self.direct_strides is not None else ()),
        )


@dataclass(frozen=True)
class _Tcgen05GroupedStaticMetadataCacheEntry:
    layout_ref: weakref.ReferenceType[torch.Tensor]
    n_sizes_ref: weakref.ReferenceType[torch.Tensor] | None
    k_sizes_ref: weakref.ReferenceType[torch.Tensor] | None
    has_m_tail: bool
    has_n_tail: bool
    result: _Tcgen05GroupedStaticMetadataResult

    def matches(
        self,
        layout: torch.Tensor,
        n_sizes: torch.Tensor | None,
        k_sizes: torch.Tensor | None,
    ) -> bool:
        if self.layout_ref() is not layout:
            return False
        if self.n_sizes_ref is None:
            if n_sizes is not None:
                return False
        elif self.n_sizes_ref() is not n_sizes:
            return False
        if self.k_sizes_ref is None:
            return k_sizes is None
        return self.k_sizes_ref() is k_sizes


@dataclass(frozen=True)
class _CuteLaunchArgCacheEntry:
    schema: tuple[tuple[object, ...], ...]
    launch_args: tuple[object, ...]
    grouped_static_metadata: tuple[_Tcgen05GroupedStaticMetadataCacheEntry, ...]
    owned_tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _CuteLastLaunchCacheEntry:
    arg_guard: _CuteLastLaunchArgGuard
    compiled_discriminator: tuple[object, ...]
    launch: _CuteLaunchArgCacheEntry
    compiled: object


@dataclass(frozen=True)
class _CuteLastTensorArgGuard:
    index: int
    data_ptr: int
    device_type: str
    device_index: int | None
    dtype: torch.dtype
    ndim: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def matches(self, args: tuple[object, ...]) -> bool:
        if self.index >= len(args) or not isinstance(args[self.index], torch.Tensor):
            return False
        tensor = args[self.index]
        assert isinstance(tensor, torch.Tensor)
        return (
            int(tensor.data_ptr()) == self.data_ptr
            and tensor.device.type == self.device_type
            and tensor.device.index == self.device_index
            and tensor.dtype == self.dtype
            and tensor.ndim == self.ndim
            and tensor.size() == self.shape
            and tensor.stride() == self.stride
        )


@dataclass(frozen=True)
class _CuteLastScalarArgGuard:
    index: int
    is_constexpr: bool
    scalar_kind: str
    scalar_value: object

    def matches(
        self,
        args: tuple[object, ...],
        constexpr_flags: tuple[bool, ...],
    ) -> bool:
        if self.index >= len(args):
            return False
        arg = args[self.index]
        if isinstance(arg, torch.Tensor):
            return False
        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        is_constexpr = self.index < len(constexpr_flags) and constexpr_flags[self.index]
        return (
            is_constexpr == self.is_constexpr
            and scalar_kind == self.scalar_kind
            and _cute_scalar_cache_value(scalar_kind, scalar_value) == self.scalar_value
        )


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


def _cute_scalar_cache_value(scalar_kind: str, scalar_value: object) -> object:
    return cast("float", scalar_value).hex() if scalar_kind == "float" else scalar_value


def _validate_cute_launcher_tensor(arg: torch.Tensor) -> None:
    if arg.device.type != "cuda":
        raise exc.BackendUnsupported("cute", "launcher requires CUDA tensors")
    if arg.ndim <= 0:
        raise exc.BackendUnsupported("cute", "launcher requires tensor rank >= 1")


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
    return [
        cast("dict[str, object]", plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
        if cast("dict[str, object]", plan).get("kind")
        == "tcgen05_grouped_static_persistent"
    ]


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


def _tcgen05_grouped_static_active_clusters(
    *,
    num_sm: int,
    cluster_m: int,
    reserved_sms: int,
) -> int:
    if num_sm <= 0:
        raise ValueError("num_sm must be positive")
    if cluster_m <= 0:
        raise ValueError("cluster_m must be positive")
    if reserved_sms < 0:
        raise ValueError("reserved_sms must be non-negative")
    active_sms = max(1, num_sm - reserved_sms)
    return max(1, active_sms // cluster_m)


def _tcgen05_grouped_static_quotas(
    problem_shapes: tuple[tuple[int, int, int], ...],
    *,
    bm: int,
    bn: int,
    bk: int,
    max_active_clusters: int,
) -> tuple[int, ...]:
    """Balance each group's longest CTA without assigning idle CTAs.

    If fewer CTAs than groups are active, the returned ones are unused because
    the device selects the generic scheduler instead of the specialized path.
    """
    tile_counts = [
        ((problem_m + bm - 1) // bm) * ((problem_n + bn - 1) // bn)
        for problem_m, problem_n, _problem_k in problem_shapes
    ]
    k_tile_counts = [
        (problem_k + bk - 1) // bk
        for _problem_m, _problem_n, problem_k in problem_shapes
    ]
    quotas = [1] * len(problem_shapes)
    grid_size = min(sum(tile_counts), max_active_clusters)
    for _ in range(max(0, grid_size - len(quotas))):
        candidates = [
            index
            for index, (quota, tile_count) in enumerate(
                zip(quotas, tile_counts, strict=True)
            )
            if quota < tile_count
        ]
        group = max(
            candidates,
            key=lambda index: (
                (tile_counts[index] + quotas[index] - 1)
                // quotas[index]
                * k_tile_counts[index],
                tile_counts[index] * k_tile_counts[index],
                -index,
            ),
        )
        quotas[group] += 1
    return tuple(quotas)


def _plan_int_value(plan: dict[str, object], key: str) -> int:
    value = plan[key]
    assert isinstance(value, int)
    return value


def _plan_str_value(plan: dict[str, object], key: str) -> str:
    value = plan[key]
    assert isinstance(value, str)
    return value


def _tcgen05_plan_orientation(plan: dict[str, object]) -> str:
    orientation = plan.get("orientation", "mn")
    if orientation not in ("mn", "nm"):
        raise exc.BackendUnsupported(
            "cute", f"unsupported tcgen05 plan orientation {orientation!r}"
        )
    return cast("str", orientation)


def _cuda_stream_capture_context(device: torch.device) -> tuple[int, int | None]:
    """Return the current stream and its active capture ID, if any."""
    cuda_runtime = importlib.import_module("cuda.bindings.runtime")
    with torch.cuda.device(device):
        stream_handle = int(torch.cuda.current_stream().cuda_stream)
        capture_info = cuda_runtime.cudaStreamGetCaptureInfo(
            cuda_runtime.cudaStream_t(stream_handle)
        )
    error = capture_info[0]
    status = capture_info[1]
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        raise exc.BackendUnsupported(
            "cute",
            f"failed to query the CUDA stream capture context: {error}",
        )
    if status == cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone:
        return stream_handle, None
    if status == cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return stream_handle, int(capture_info[2])
    raise exc.BackendUnsupported(
        "cute", "the current CUDA stream capture has been invalidated"
    )


def _tcgen05_grouped_dynamic_tensormap_workspace(
    cute_kernel: object,
    *,
    device: torch.device,
    tensormap_count: int,
) -> torch.Tensor:
    num_sm = get_num_sm(device)
    stream_handle, capture_id = _cuda_stream_capture_context(device)
    cache_key = (
        device.type,
        device.index,
        num_sm,
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
        (num_sm, tensormap_count, 128 // 8),
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
    if _tcgen05_plan_orientation(plan) != "nm":
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
        or source_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
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
        not bool(plan.get("dynamic_ab_tensormaps"))
        or _tcgen05_grouped_dynamic_ab_tensormap_rank(plan) != 2
        or not bool(plan.get("dynamic_d_tensormap"))
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires rank-2 dynamic A/B "
            "TensorMaps and a dynamic D TensorMap",
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
        or source_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped device split_sizes requires positive G, M, and N "
            "and a validated source M tile",
        )
    n_clusters = (n_size + physical_mma_m - 1) // physical_mma_m
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
            bool(plan.get("dynamic_ab_tensormaps"))
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
        lhs.dtype is not torch.bfloat16
        or rhs.dtype is not torch.bfloat16
        or int(lhs.size(1)) != k_total_size
        or int(rhs.size(1)) != n_size
        or int(rhs.size(2)) != k_total_size
        or physical_mma_m not in (128, 256)
        or n_size % physical_mma_m != 0
        or k_total_size % bk != 0
    ):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require contiguous BF16 "
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
        or output.dtype is not torch.bfloat16
        or output.ndim != 2
        or tuple(int(size) for size in output.shape) != (int(lhs.size(0)), n_size)
        or tuple(int(stride) for stride in output.stride()) != (n_size, 1)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "fixed full-allocation TensorMaps require contiguous BF16 "
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


def _cute_bake_tensor_shapes_guard(cute_kernel: object) -> bool:
    any_obj = cast("Any", cute_kernel)
    wrapper_plans = getattr(any_obj, "_helion_cute_wrapper_plans", None)
    wrapper_plans_disable_bake = bool(wrapper_plans) and any(
        not _cute_wrapper_plan_bakes_tensor_shapes(plan) for plan in wrapper_plans
    )
    return not bool(
        getattr(any_obj, "_helion_cute_disable_bake_tensor_shapes", False)
        or wrapper_plans_disable_bake
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


def _cute_launch_arg_cache_key(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    *,
    dynamic_tensormap_contexts: tuple[_CuteGroupedLaunchContext, ...] | None = None,
) -> tuple[object, ...]:
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    key: list[object] = [len(args), grid, _cute_bake_tensor_shapes_guard(cute_kernel)]
    if dynamic_tensormap_contexts is None:
        dynamic_tensormap_contexts = _cute_dynamic_tensormap_contexts(cute_kernel, args)
    if dynamic_tensormap_contexts:
        key.append(("dynamic_tensormap_contexts", dynamic_tensormap_contexts))
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            _validate_cute_launcher_tensor(arg)
            key.append(
                (
                    "tensor",
                    arg.device.type,
                    arg.device.index,
                    str(arg.dtype),
                    arg.ndim,
                    int(arg.data_ptr()),
                    tuple(int(arg.size(d)) for d in range(arg.ndim)),
                    tuple(int(arg.stride(d)) for d in range(arg.ndim)),
                )
            )
            continue

        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        scalar_key_value = _cute_scalar_cache_value(scalar_kind, scalar_value)
        is_constexpr = i < len(constexpr_flags) and constexpr_flags[i]
        key.append(
            (
                "scalar_constexpr" if is_constexpr else "scalar",
                scalar_kind,
                scalar_key_value,
            )
        )
    _append_tcgen05_grouped_static_mutation_guards(key, cute_kernel, args)
    return tuple(key)


def _retain_cute_capture_owned_launch_tensors(
    cute_kernel: object,
    *,
    grouped_launch_contexts: tuple[_CuteGroupedLaunchContext, ...],
    owned_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Keep raw-pointer launch tensors alive for captured graph replays."""
    capture_contexts = tuple(
        context for context in grouped_launch_contexts if context[3] is not None
    )
    if not capture_contexts or not owned_tensors:
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
) -> _CuteLaunchArgCacheEntry:
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

    built = _build_cute_schema_and_args(cute_kernel, args, grid)
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


def _cute_wrapper_plan_bakes_tensor_shapes(plan: dict[str, object]) -> bool:
    kind = str(plan.get("kind", ""))
    if kind == "helion_small_biased_attention":
        return True
    if not kind.startswith("tcgen05"):
        return False
    if kind != "tcgen05_ab_tma":
        return True
    extent_blocks = (
        (("n_size", "bm"), ("m_size", "bn"), ("k_total_size", "bk"))
        if _tcgen05_plan_orientation(plan) == "nm"
        else (("m_size", "bm"), ("n_size", "bn"), ("k_total_size", "bk"))
    )
    for extent_key, block_key in extent_blocks:
        extent = plan.get(extent_key)
        block = plan.get(block_key)
        if type(extent) is not int or type(block) is not int or extent % block:
            return False
    return True


def _build_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    bake_tensor_shapes: bool = True,
) -> _CuteLaunchArgCacheEntry:
    # NOTE: the returned launch args deliberately EXCLUDE the CUDA stream. The
    # stream is the only launch arg that is not a pure function of
    # (grid, tensor metadata, scalars), so it must not be baked into the cached
    # args — the caller appends a freshly sampled ``_cute_current_stream()`` on
    # every launch (see ``default_cute_launcher``). Caching the stream would
    # break CUDA graph capture (empty-graph / no-op replay).
    gmem_space, make_ptr_obj, _current_stream_obj = _get_cute_launcher_imports()
    make_ptr = cast("Any", make_ptr_obj)
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    # Universal MMA needs runtime tensor layouts for its SMEM-load guards.
    # Full-tile tcgen05 wrapper schemas are specialized by problem shape and
    # stride, while partial-tile paths still propagate runtime tensor layouts.
    if bake_tensor_shapes:
        if not _cute_bake_tensor_shapes_guard(cute_kernel):
            bake_tensor_shapes = False
    schema: list[tuple[object, ...]] = []
    launch_args: list[object] = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            _validate_cute_launcher_tensor(arg)
            ndim = arg.ndim
            if ndim <= 0:
                raise exc.BackendUnsupported(
                    "cute", "launcher requires tensor rank >= 1"
                )
            sizes_t = tuple(int(arg.size(d)) for d in range(ndim))
            strides_t = tuple(int(arg.stride(d)) for d in range(ndim))
            launch_args.append(
                make_ptr(
                    cast("Any", _torch_dtype_to_cutlass(arg.dtype)),
                    arg.data_ptr(),
                    gmem_space,
                    assumed_align=16,
                )
            )
            # ``cute.make_layout`` rejects a 0 in any shape dimension, so
            # zero-sized tensors must keep the runtime-shape path.
            if bake_tensor_shapes and all(s > 0 for s in sizes_t):
                # Bake the shape / stride tuple into the schema key.  The
                # generated wrapper substitutes literal Int values for each
                # dimension, so the CuTe DSL sees a fully static tensor
                # layout and the per-load offset arithmetic collapses to
                # constant strides — typically a 2-3x reduction in
                # ``smsp__inst_executed`` for reduction kernels where the
                # inner loop is dominated by stride multiplies.
                schema.append(("tensor", str(arg.dtype), ndim, sizes_t, strides_t))
            else:
                schema.append(("tensor", str(arg.dtype), ndim))
                launch_args.extend(sizes_t)
                launch_args.extend(strides_t)
            continue

        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        is_constexpr = i < len(constexpr_flags) and constexpr_flags[i]
        if is_constexpr:
            # Bake Constexpr values into the wrapper / cache key. cutlass DSL
            # >=4.5 fails IR verification ("value defined outside the region")
            # if a runtime scalar is fed to a kernel parameter declared as
            # ``cutlass.Constexpr``.
            schema.append(
                (
                    "scalar_constexpr",
                    scalar_kind,
                    _cute_scalar_cache_value(scalar_kind, scalar_value),
                    scalar_value,
                )
            )
        else:
            schema.append(("scalar", scalar_kind))
            launch_args.append(scalar_value)

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
        launch_args.append(
            make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                tensor.data_ptr(),
                gmem_space,
                assumed_align=16,
            )
        )
        if runtime_leading_extent:
            schema.append(
                (
                    "wrapper_tensor_runtime_leading_extent",
                    name,
                    str(tensor.dtype),
                    tensor.ndim,
                    sizes[1:],
                    strides,
                )
            )
            launch_args.append(sizes[0])
        else:
            schema.append(
                (
                    "wrapper_tensor",
                    name,
                    str(tensor.dtype),
                    tensor.ndim,
                    sizes,
                    strides,
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
) -> _CuteLastLaunchArgGuard:
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    arg_guards: list[_CuteLastTensorArgGuard | _CuteLastScalarArgGuard] = []
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            _validate_cute_launcher_tensor(arg)
            arg_guards.append(
                _CuteLastTensorArgGuard(
                    index=index,
                    data_ptr=int(arg.data_ptr()),
                    device_type=arg.device.type,
                    device_index=arg.device.index,
                    dtype=arg.dtype,
                    ndim=arg.ndim,
                    shape=tuple(int(arg.size(dim)) for dim in range(arg.ndim)),
                    stride=tuple(int(arg.stride(dim)) for dim in range(arg.ndim)),
                )
            )
            continue
        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        arg_guards.append(
            _CuteLastScalarArgGuard(
                index=index,
                is_constexpr=index < len(constexpr_flags) and constexpr_flags[index],
                scalar_kind=scalar_kind,
                scalar_value=_cute_scalar_cache_value(scalar_kind, scalar_value),
            )
        )
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
        arg_guards=tuple(arg_guards),
        grouped_mutation_guards=tuple(grouped_mutation_guards),
        grouped_launch_contexts=_cute_grouped_launch_contexts(cute_kernel, args),
    )


_CUTE_FASTPATH_MISS: tuple[bool, None] = (False, None)


class _CuteFastRelaunch:
    """Metadata-guarded, zero-marshalling relaunch of a compiled cute kernel.

    The CuTe DSL's per-call path (schema/arg cache key construction plus
    ``generate_execution_args`` marshalling: ctypes storage allocation and
    pointer extraction, ~25-45us of Python) recomputes almost exactly the
    same values on every call — for a given compiled kernel only the tensor
    DATA POINTERS and the CUDA stream can change between calls once the
    tensor metadata (dtype/shape/stride/device), the scalars, the grid, and
    the block are pinned.  Real workloads allocate fresh output tensors on
    every call, so pointer-keyed caches miss constantly; for O(30us)
    memory-bound kernels the marshalling then starves the GPU.

    This caches the marshalled ``exe_args`` once and per call only:

    1. checks the metadata guard (no pointer equality),
    2. writes each tensor arg's ``data_ptr()`` into its probe-verified
       ``exe_args`` slot (tensor pointers marshal by value),
    3. refreshes the CUDA stream slot(s), and
    4. invokes the executor's ``run_compiled_program``.

    Slot discovery is probe-verified at build time: tensor slots by
    re-marshalling with per-tensor shifted pointers, stream slots by
    re-marshalling with a different stream handle (handling both by-value
    slots and by-reference ctypes cells).  Anything unexpected falls back
    to the full DSL call path.

    The current stream is sampled fresh on every call via
    ``torch._C._cuda_getCurrentRawStream`` (same primitive the Triton
    launcher uses), so CUDA-graph capture streams are honored.
    """

    __slots__ = (
        "arg_count",
        "block",
        "by_ref_writers",
        "by_val_slots",
        "compile_options",
        "constexpr_flags",
        "device_index",
        "exe_args",
        "executor",
        "grid",
        "keepalive",
        "last_raw",
        "lock",
        "scalar_guards",
        "tensor_guards",
        "tensor_slots",
    )

    def __init__(
        self,
        *,
        executor: object,
        exe_args: list[object],
        tensor_guards: tuple[
            tuple[int, str, int | None, torch.dtype, tuple[int, ...], tuple[int, ...]],
            ...,
        ],
        scalar_guards: tuple[_CuteLastScalarArgGuard, ...],
        constexpr_flags: tuple[bool, ...],
        tensor_slots: tuple[tuple[int, int | None, object | None], ...],
        by_ref_writers: list[object],
        by_val_slots: list[int],
        arg_count: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        compile_options: str | None,
        device_index: int,
        last_raw: int,
        keepalive: tuple[object, ...],
    ) -> None:
        self.executor = executor
        self.exe_args = exe_args
        self.tensor_guards = tensor_guards
        self.scalar_guards = scalar_guards
        self.constexpr_flags = constexpr_flags
        self.tensor_slots = tensor_slots
        self.by_ref_writers = by_ref_writers
        self.by_val_slots = by_val_slots
        self.arg_count = arg_count
        self.grid = grid
        self.block = block
        self.compile_options = compile_options
        self.device_index = device_index
        self.last_raw = last_raw
        self.keepalive = keepalive
        self.lock = threading.Lock()

    def try_launch(
        self,
        args: tuple[object, ...],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        compile_options: str | None,
    ) -> tuple[bool, object]:
        if (
            len(args) != self.arg_count
            or grid != self.grid
            or block != self.block
            or compile_options != self.compile_options
        ):
            return _CUTE_FASTPATH_MISS
        for (
            index,
            device_type,
            device_index,
            dtype,
            shape,
            stride,
        ) in self.tensor_guards:
            tensor = args[index]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype is not dtype
                or tensor.device.type != device_type
                or tensor.device.index != device_index
                or tensor.size() != shape
                or tensor.stride() != stride
            ):
                return _CUTE_FASTPATH_MISS
        for guard in self.scalar_guards:
            if not guard.matches(args, self.constexpr_flags):
                return _CUTE_FASTPATH_MISS
        raw = torch._C._cuda_getCurrentRawStream(self.device_index)
        with self.lock:
            exe_args = self.exe_args
            # tensor_slots entries: (arg_index, exe_slot_or_None, writer_or_None)
            for index, slot, writer in self.tensor_slots:
                ptr = cast("torch.Tensor", args[index]).data_ptr()
                if writer is not None:
                    cast("Any", writer).value = ptr
                else:
                    exe_args[cast("int", slot)] = ptr
            if raw != self.last_raw:
                for stream_writer in self.by_ref_writers:
                    cast("Any", stream_writer).value = raw
                for stream_slot in self.by_val_slots:
                    exe_args[stream_slot] = raw
                self.last_raw = raw
            return (True, cast("Any", self.executor).run_compiled_program(exe_args))


def _cute_maybe_build_fastpath(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
    launch: _CuteLaunchArgCacheEntry,
    compiled: object,
) -> None:
    """Build (once per kernel) the fast relaunch state after a successful
    slow-path launch.  ``False`` marks a permanent probe failure so the
    slow path is not re-probed on every call."""
    if getattr(cast("Any", cute_kernel), "_helion_cute_fastpath", None) is not None:
        return
    state = _cute_build_fast_relaunch(
        cute_kernel, args, grid, block, compile_options, launch, compiled
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
) -> _CuteFastRelaunch | None:
    if not isinstance(compiled, _CompiledCuteLauncher):
        return None
    dsl_fn = cast("Any", compiled)._compiled
    if dsl_fn is None:
        return None
    execution_args = getattr(dsl_fn, "execution_args", None)
    executor = getattr(dsl_fn, "_default_executor", None)
    if (
        execution_args is None
        or not hasattr(execution_args, "generate_execution_args")
        or executor is None
        or not hasattr(executor, "run_compiled_program")
    ):
        return None
    # Kernels with grouped-scheduler plans, dynamic tensormaps, or launcher-
    # owned side tensors have per-call state beyond (pointers, stream);
    # leave them on the full path.
    if launch.owned_tensors or launch.grouped_static_metadata:
        return None
    if _tcgen05_grouped_static_plans(cute_kernel):
        return None
    if _cute_dynamic_tensormap_contexts(cute_kernel, args):
        return None
    device_index: int | None = None
    tensors: list[tuple[int, torch.Tensor]] = []
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if arg.device.type != "cuda":
                return None
            if device_index is None:
                device_index = arg.device.index
            tensors.append((index, arg))
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
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
        for k, (_arg_index, tensor) in enumerate(tensors):
            own_base[base_ptr_positions[k]] = make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                int(tensor.data_ptr()),
                gmem_space,
                assumed_align=16,
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
            if isinstance(addr, int) and addr > (1 << 40) and addr % 8 == 0:
                return int(ctypes.c_uint64.from_address(addr).value)
            return None

        # --- Tensor-pointer slots: re-marshal with per-tensor shifted
        # pointer clones and locate each tensor's slot by its shifted
        # value.  Handles both marshalling conventions: by value (the slot
        # holds the device pointer itself) and by reference (the slot holds
        # the address of a per-object ctypes cell containing it).
        alt_base = list(base)
        shifts: list[int] = []
        for k, (_arg_index, tensor) in enumerate(tensors):
            shift = 512 * (k + 1)
            shifts.append(shift)
            alt_base[base_ptr_positions[k]] = make_ptr(
                cast("Any", _torch_dtype_to_cutlass(tensor.dtype)),
                int(tensor.data_ptr()) + shift,
                gmem_space,
                assumed_align=16,
            )
        exe4, _adapted4 = execution_args.generate_execution_args(
            (*tuple(alt_base), stream_a), {}
        )
        n4 = [norm(v) for v in exe4]
        if len(n4) != len(n1):
            return None
        tensor_slots: list[tuple[int, int | None, object | None]] = []
        for k, (arg_index, tensor) in enumerate(tensors):
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
        tensor_guards: list[
            tuple[int, str, int | None, torch.dtype, tuple[int, ...], tuple[int, ...]]
        ] = []
        scalar_guards: list[_CuteLastScalarArgGuard] = []
        for index, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_guards.append(
                    (
                        index,
                        arg.device.type,
                        arg.device.index,
                        arg.dtype,
                        tuple(int(arg.size(d)) for d in range(arg.ndim)),
                        tuple(int(arg.stride(d)) for d in range(arg.ndim)),
                    )
                )
                continue
            scalar_kind, scalar_value = _normalize_cute_scalar(arg)
            scalar_guards.append(
                _CuteLastScalarArgGuard(
                    index=index,
                    is_constexpr=index < len(constexpr_flags)
                    and constexpr_flags[index],
                    scalar_kind=scalar_kind,
                    scalar_value=_cute_scalar_cache_value(scalar_kind, scalar_value),
                )
            )
        return _CuteFastRelaunch(
            executor=executor,
            exe_args=list(exe1),
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
) -> None:
    arg_guard = _cute_last_launch_arg_guard(cute_kernel, args, grid)
    compiled_discriminator = _cute_compiled_launcher_discriminator(
        launch.schema,
        block,
        compile_options,
        args,
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

    if any(dim <= 0 for dim in grid_xyz):
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
            return result
    last_launch = _cute_last_launch_cache_entry(
        cute_kernel,
        args_tuple,
        grid_xyz,
        block_xyz,
        cute_compile_options,
    )
    if last_launch is not None:
        _record_cute_owned_launch_tensors(last_launch.launch.owned_tensors)
        return cast("Any", last_launch.compiled)(
            *last_launch.launch.launch_args,
            _cute_current_stream(),
        )

    launch = _build_cached_cute_schema_and_args(cute_kernel, args_tuple, grid_xyz)
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
    )
    return result
