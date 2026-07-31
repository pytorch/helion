from __future__ import annotations

import base64
from contextlib import suppress
import hashlib
import importlib
import inspect
import json
import linecache
import logging
import os
import sys
from typing import Any
from typing import cast

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from .._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from .._compiler.cute.strategies import tcgen05_smem_layout_expr
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .pallas.launcher import default_pallas_launcher as default_pallas_launcher
from .settings import is_pallas_interpret as _module_is_pallas_interpret
from .triton.launcher import default_launcher as _triton_default_launcher
from .triton.launcher import get_num_sm as _triton_get_num_sm
from .triton.launcher import get_num_xcd as get_num_xcd
from .triton.launcher import set_triton_allocator as set_triton_allocator

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


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Thin in-process wrapper over the dependency-free
    :func:`helion.runtime.triton.launcher.default_launcher` that translates
    Triton's opaque "incompatible dimensions" error into
    :class:`helion.exc.ShapeMismatch`.
    """
    try:
        return _triton_default_launcher(
            triton_kernel,
            grid,
            *args,
            num_warps=num_warps,
            num_stages=num_stages,
            ptx_options=ptx_options,
            launch_cooperative_grid=launch_cooperative_grid,
            **kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        raise


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """Number of SMs (persistent-kernel grid size) for any Helion device.

    Adds the CPU (Pallas-interpret) and TPU cases on top of the dependency-free
    GPU helper :func:`helion.runtime.triton.launcher.get_num_sm`. See that
    function for argument/return semantics.
    """
    if device.type == "cpu":
        if not _module_is_pallas_interpret():
            raise AssertionError("TODO: implement for other devices")
        return 1
    if device.type == "tpu":
        return 1
    return _triton_get_num_sm(device, reserved_sms=reserved_sms)


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
        column_major: bool = False,
    ) -> None:
        assert len(kernel_args) == 2
        tensor_expr = tensor_name if tensor_name is not None else f"arg{tensor_idx}"
        c_layout = (
            "cutlass.utils.layout.LayoutEnum.COL_MAJOR"
            if column_major
            else "cutlass.utils.layout.LayoutEnum.ROW_MAJOR"
        )
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
                c_layout=c_layout,
            )
        tma_atom, tma_tensor = kernel_args
        epi_tile = f"{tma_atom}_epi_tile"
        smem_layout = f"{tma_atom}_smem_layout"
        cta_v_layout = f"{tma_atom}_cta_v_layout"
        # Keep these layout arguments in sync with the device-side
        # ``make_smem_layout_epi`` calls; the wrapper's TMA atom and the kernel's
        # SMEM staging must slice the same epilogue tile shape.
        body.extend(
            (
                f"    {epi_tile} = {epi_tile_expr}",
                (
                    f"    {smem_layout} = cutlass.utils.blackwell_helpers."
                    "make_smem_layout_epi("
                    f"{dtype}, {c_layout}, "
                    f"{epi_tile}, {stage_count})"
                ),
                (
                    f"    {cta_v_layout} = cute.composition("
                    f"cute.make_identity_layout({tensor_expr}.shape), {epi_tile})"
                ),
                (
                    f"    {tma_atom}, {tma_tensor} = "
                    "cute.nvgpu.cpasync.make_tiled_tma_atom("
                    f"{copy_op}, "
                    f"{tensor_expr}, cute.slice_({smem_layout}, (None, None, 0)), "
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
    if kind != "tcgen05_ab_tma":
        raise exc.BackendUnsupported("cute", f"wrapper plan kind: {kind}")

    lhs_idx_key = "lhs_idx" if "lhs_idx" in plan else "lhsidx"
    rhs_idx_key = "rhs_idx" if "rhs_idx" in plan else "rhsidx"
    lhs_idx = plan_int(lhs_idx_key)
    rhs_idx = plan_int(rhs_idx_key)
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
    # K-major (column-major / K-contiguous) B. Absent on the MN-major
    # (row-major B) default path.
    b_k_major = bool(plan.get("b_k_major"))
    lhs_leading_passthrough = bool(plan.get("lhs_leading_passthrough"))
    rhs_leading_passthrough = bool(plan.get("rhs_leading_passthrough"))
    lhs_trailing_transpose = bool(plan.get("lhs_trailing_transpose"))
    rhs_trailing_transpose = bool(plan.get("rhs_trailing_transpose"))
    assert not (lhs_leading_passthrough and lhs_trailing_transpose)
    assert not (rhs_leading_passthrough and rhs_trailing_transpose)
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
    rhs_tma = f"{tma_atom_b}_rhs_tma"
    lhs_tma_operand = (
        lhs_tma
        if lhs_leading_passthrough or lhs_trailing_transpose
        else f"arg{lhs_idx}"
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
        b_k_major=b_k_major,
    )
    if lhs_leading_passthrough:
        append_permuted_cute_tensor_view(lhs_tma, lhs_idx, (1, 2, 0))
    elif lhs_trailing_transpose:
        append_permuted_cute_tensor_view(lhs_tma, lhs_idx, (1, 0))
    if rhs_leading_passthrough:
        append_permuted_cute_tensor_view(rhs_tma, rhs_idx, (2, 1, 0))
    ab_tma_lines = [
        (
            f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
            f"{input_dtype}, "
            f"{input_dtype}, "
            "cute.nvgpu.OperandMajorMode.K, "
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
    ]
    if rhs_trailing_transpose:
        ab_tma_lines.append(
            f"    {rhs_tma} = cute.make_tensor("
            f"arg{rhs_idx}.iterator, "
            "layout=cute.make_layout("
            f"(arg{rhs_idx}_shape0, arg{rhs_idx}_shape1), "
            f"stride=(arg{rhs_idx}_stride0, arg{rhs_idx}_stride1)))"
        )
    elif not rhs_leading_passthrough:
        ab_tma_lines.append(
            f"    {rhs_tma} = cute.make_tensor("
            f"arg{rhs_idx}.iterator, "
            "layout=cute.make_layout("
            f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape0), "
            f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride0)))"
        )
    ab_tma_lines.extend(
        [
            # B is viewed as (N, K). For row-major B (MN-major) the N axis
            # (position 0) is contiguous; for column-major B (K-major, native
            # fp8 layout) the K axis (position 1) is contiguous.
            f"    {rhs_tma}.mark_layout_dynamic(leading_dim={1 if b_k_major else 0})",
            # ``make_tiled_tma_atom_A`` vs ``_B`` asymmetry:
            # - ``_B`` always passes ``cluster_layout_vmnk.shape`` as
            #   its trailing arg (CuTe's signature for B requires the
            #   cluster shape; the cluster_m=1 cluster_n=1 case still
            #   passes the 1×1×1 shape harmlessly).
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
                f"{lhs_tma_operand}, "
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
        ]
    )
    body.extend(ab_tma_lines)
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
    if any(plan.get("topology") == "fa4" for plan in wrapper_plans):
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
    compile_options = _merge_tvm_ffi_compile_option(compile_options)
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_compiled_launchers
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_compiled_launchers = cache
    wrapper_plans = tuple(
        repr(plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    )
    cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    # Persistent flash kernels bake the device SM count into the wrapper grid
    # clamp; resolve it from the first tensor arg's device so the cache key (and
    # the baked literal) stay device-correct across GPUs.
    num_sm: int | None = None
    if arch_args is not None:
        for arg in arch_args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
                num_sm = get_num_sm(arg.device)
                break
    cache_key = (
        schema_key,
        block,
        wrapper_plans,
        repr(cluster_shape),
        compile_options,
        num_sm,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if arch_args is not None:
        _ensure_cute_dsl_arch_env(arch_args)
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


def _cute_scalar_cache_value(scalar_kind: str, scalar_value: object) -> object:
    return cast("float", scalar_value).hex() if scalar_kind == "float" else scalar_value


def _validate_cute_launcher_tensor(arg: torch.Tensor) -> None:
    if arg.device.type != "cuda":
        raise exc.BackendUnsupported("cute", "launcher requires CUDA tensors")
    if arg.ndim <= 0:
        raise exc.BackendUnsupported("cute", "launcher requires tensor rank >= 1")


def _cute_launch_arg_cache_key(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
) -> tuple[object, ...]:
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    key: list[object] = [grid]
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
                    arg.data_ptr(),
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
    return tuple(key)


def _build_cached_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
    cache_key = _cute_launch_arg_cache_key(cute_kernel, args, grid)
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_launch_arg_cache
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_launch_arg_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        cache[cache_key] = cache.pop(cache_key)
        return cached

    built = _build_cute_schema_and_args(cute_kernel, args, grid)
    cache[cache_key] = built
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
    for extent_key, block_key in (
        ("m_size", "bm"),
        ("n_size", "bn"),
        ("k_total_size", "bk"),
    ):
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
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
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
        any_obj = cast("Any", cute_kernel)
        wrapper_plans = getattr(any_obj, "_helion_cute_wrapper_plans", None)
        non_bakeable_plan = bool(wrapper_plans) and any(
            not _cute_wrapper_plan_bakes_tensor_shapes(plan) for plan in wrapper_plans
        )
        if (
            getattr(any_obj, "_helion_cute_disable_bake_tensor_shapes", False)
            or non_bakeable_plan
        ):
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

    launch_args.extend(grid)
    # The stream is intentionally NOT appended here; it is sampled fresh per
    # launch by the caller so CUDA graph capture sees the capture stream.
    return tuple(schema), tuple(launch_args)


_CUTE_DSL_ARCH_CACHE: dict[int, str] = {}
_CUTE_MIN_CUDA_VERSION = "13"


def _require_cuda13_for_cute() -> None:
    from .._compat import requires_cuda_version

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
    schema_key, launch_args = _build_cached_cute_schema_and_args(
        cute_kernel, args_tuple, grid_xyz
    )
    compiled = _get_compiled_cute_launcher(
        cute_kernel,
        schema_key,
        block_xyz,
        compile_options=cute_compile_options,
        arch_args=args_tuple,
    )
    # Append the CUDA stream fresh on every launch (never cached): under CUDA
    # graph capture the current stream is the capture stream, so the kernel must
    # be issued there and not on a stale stream baked into the cached args.
    return cast("Any", compiled)(*launch_args, _cute_current_stream())


def default_metal_launcher(
    metal_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _block_dims: tuple[int, int, int] = (256, 1, 1),
    **kwargs: object,
) -> None:
    """Default launcher for Metal kernels on Apple MPS devices.

    The ``metal_kernel`` is a ``@metal_jit`` decorated function that
    translates its Python AST body to MSL and compiles it via
    ``torch.mps.compile_shader`` on each call.
    This launcher dispatches the compiled kernel with the given grid and
    threadgroup dimensions.

    Uses a 3D threadgroup dispatch model: ``_block_dims`` specifies the
    threadgroup size as ``(x, y, z)``.  The grid specifies the number of
    threadgroups per dimension.
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "metal", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    from .._compiler.metal.metal_launcher import set_required_threads_per_threadgroup

    set_required_threads_per_threadgroup(metal_kernel, _block_dims)
    lib, kernel_name = metal_kernel(*args)  # type: ignore[operator]

    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    dispatch_fn = getattr(lib, kernel_name)
    bx, by, bz = _block_dims
    # Pad grid to 3D
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1
    total_threads = (gx * bx, gy * by, gz * bz)
    group_size = (bx, by, bz)
    dispatch_fn(*tensor_args, threads=total_threads, group_size=group_size)
