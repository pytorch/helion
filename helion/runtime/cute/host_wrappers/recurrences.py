"""Host side of the KDA / GDN recurrence schedules and the KDA prepare kernel.

The whole-root recurrence kernels (``chunk_recurrence_sm100``,
``chunk_recurrence_warp_dv4``, ``gdn_recurrence_sm100``) replace the generic
launch with a call into their vendored CuTe host schedule; the KDA prepare
kernel (``chunk_prepare_tma``) and the warp-DV4 recurrence take raw
``cuTensorMapEncodeTiled`` descriptors the launcher encodes over the
validated packed workspace.  This module owns the plan re-validation, the
host-call text, the namespace bindings and the tensor-map specs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from .... import exc
from ..tensor_maps import TensorMapSpec
from ..tensor_maps import build_tensor_maps
from ..tensor_maps import factor_slab_geometry
from ..tensor_maps import spec_fields
from .context import HostWrapperContext
from .context import HostWrapperEmitter

if TYPE_CHECKING:
    from collections.abc import Sequence


def emit_chunk_recurrence_sm100(
    ctx: HostWrapperContext, plan: dict[str, object]
) -> None:
    """``chunk_recurrence_sm100``: re-prove the DV2 schedule ABI; the host call replaces the launch."""
    outputs_scaled = plan.get("outputs_scaled")
    factor_key_xor = plan.get("factor_key_xor")
    if (
        ctx.plan_int("chunk_size") != 16
        or ctx.plan_int("key_size") != 128
        or ctx.plan_int("value_size") != 128
        or ctx.plan_int("threads") != 512
        or ctx.plan_int("workspace_layout_version") != 2
        or outputs_scaled is not False
        or factor_key_xor != 8
        or ctx.plan_int("device_abi") != 2
        or ctx.plan_int("input_stages") != 8
        or ctx.plan_int("tma_stages") != 6
        or ctx.plan_int("factor_tma_value_splits") != 2
        or ctx.plan_int("output_acc_stages") != 2
        or ctx.plan_int("output_smem_stages") != 7
        or ctx.plan_int("output_store_wait_groups") != 6
        or ctx.plan_int("tmem_cols") != 512
    ):
        raise exc.BackendUnsupported(
            "cute", "invalid SM100 chunk-recurrence schedule ABI"
        )


def emit_gdn_recurrence_sm100(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``gdn_recurrence_sm100``: re-prove the schedule ABI; the host call replaces the launch."""
    validate_gdn_recurrence_plan(plan)


def emit_chunk_recurrence_warp_dv4(
    ctx: HostWrapperContext, plan: dict[str, object]
) -> None:
    """``chunk_recurrence_warp_dv4``: re-prove the warp-DV4 schedule ABI and pass its descriptor arguments."""
    outputs_scaled = plan.get("outputs_scaled")
    factor_key_xor = plan.get("factor_key_xor")
    if (
        ctx.plan_int("chunk_size") != 16
        or ctx.plan_int("key_size") != 128
        or ctx.plan_int("value_size") != 128
        or ctx.plan_int("threads") != 192
        or ctx.plan_int("smem_bytes") != 97_536
        or ctx.plan_int("workspace_layout_version") != 2
        or outputs_scaled is not False
        or factor_key_xor != 8
        or ctx.plan_int("device_abi") != 3
        or ctx.plan_int("input_stages") != 6
        or ctx.plan_int("tma_stages") != 6
        or ctx.plan_int("factor_tma_value_splits") != 1
        or ctx.plan_int("output_acc_stages") != 2
        or ctx.plan_int("output_smem_stages") != 3
        or ctx.plan_int("output_store_wait_groups") != 0
        or ctx.plan_int("tmem_cols") != 0
        or ctx.plan_int("dv_partitions") != 4
    ):
        raise exc.BackendUnsupported(
            "cute", "invalid SM100 warp-DV4 chunk-recurrence schedule ABI"
        )
    desc_args = plan.get("desc_args")
    if (
        not isinstance(desc_args, (list, tuple))
        or len(desc_args) != 7
        or not all(isinstance(name, str) for name in desc_args)
    ):
        raise exc.BackendUnsupported("cute", "invalid SM100 warp-DV4 descriptor ABI")
    ctx.body.extend(
        (
            f"    grid_x = cutlass.Int32({4 * ctx.plan_int('sequences')})",
            f"    grid_y = cutlass.Int32({ctx.plan_int('heads')})",
            "    grid_z = cutlass.Int32(1)",
        )
    )
    ctx.call_args.extend(cast("Sequence[str]", desc_args))


def emit_chunk_prepare_tma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``chunk_prepare_tma``: re-prove the prepare schedule ABI and pass its descriptor arguments."""
    outputs_scaled = plan.get("outputs_scaled")
    factor_key_xor = plan.get("factor_key_xor")
    device_abi = ctx.plan_int("device_abi")
    schedule = plan.get("schedule")
    split_alias_schedules = tuple(
        f"split_alias_cpc{chunks_per_cta}" for chunks_per_cta in range(1, 6)
    )
    expected_smem = 21_968 if schedule in split_alias_schedules else None
    if (
        ctx.plan_int("chunk_size") != 16
        or ctx.plan_int("key_size") != 128
        or ctx.plan_int("chunks_per_cta") not in (1, 2, 3, 4, 5)
        or (
            schedule in split_alias_schedules
            and schedule != f"split_alias_cpc{ctx.plan_int('chunks_per_cta')}"
        )
        or expected_smem is None
        or ctx.plan_int("smem_bytes") != expected_smem
        or ctx.plan_int("bf16_mma_count") != 168
        or ctx.plan_int("fp16_mma_count") != 24
        or type(outputs_scaled) is not bool
        or device_abi != (3 if outputs_scaled else 4)
        or factor_key_xor != (0 if outputs_scaled else 8)
    ):
        raise exc.BackendUnsupported("cute", "invalid chunk-prepare schedule ABI")
    desc_args = plan.get("desc_args")
    if (
        not isinstance(desc_args, (list, tuple))
        or len(desc_args) != 4
        or not all(isinstance(name, str) for name in desc_args)
    ):
        raise exc.BackendUnsupported("cute", "invalid chunk-prepare descriptor ABI")
    chunks_per_cta = ctx.plan_int("chunks_per_cta")
    grid_x = (ctx.plan_int("total_chunks") + chunks_per_cta - 1) // chunks_per_cta
    ctx.body.extend(
        (
            f"    grid_x = cutlass.Int32({grid_x})",
            f"    grid_y = cutlass.Int32({ctx.plan_int('heads')})",
            "    grid_z = cutlass.Int32(1)",
        )
    )
    ctx.call_args.extend(cast("Sequence[str]", desc_args))


def append_sm100_chunk_recurrence_host_call(
    body: list[str], plan: dict[str, object]
) -> None:
    """Build the tensor views consumed by the SM100 recurrence host schedule."""

    def plan_int(key: str) -> int:
        value = plan.get(key)
        if type(value) is not int:
            raise exc.BackendUnsupported(
                "cute", f"invalid SM100 chunk-recurrence {key}"
            )
        return value

    def tensor_arg(key: str) -> str:
        return f"arg{plan_int(key)}"

    output_scale = f"arg{plan_int('scale_idx')}"

    heads = plan_int("heads")
    sequences = plan_int("sequences")
    total_tokens = plan_int("total_tokens")
    total_chunks = plan_int("total_chunks")
    rows = total_chunks * 16
    kd = tensor_arg("kd_idx")
    qd = tensor_arg("qd_idx")
    ak = tensor_arg("ak_idx")
    aq = tensor_arg("aq_idx")
    gt = tensor_arg("gt_idx")
    values = tensor_arg("v_idx")
    output = tensor_arg("out_idx")
    state = tensor_arg("state_idx")
    cu_seqlens = tensor_arg("cu_seqlens_idx")
    cu_chunks = tensor_arg("cu_chunks_idx")

    factor_shape = (1, heads, rows, 128)
    factor_stride = (heads * rows * 128, rows * 128, 128, 1)
    aq_shape = (1, heads, total_chunks, 256)
    aq_stride = (heads * total_chunks * 256, total_chunks * 256, 256, 1)
    gt_shape = (1, heads, total_chunks, 128)
    gt_stride = (heads * total_chunks * 128, total_chunks * 128, 128, 1)
    activation_shape = (1, total_tokens, heads, 128)
    activation_stride = (total_tokens * heads * 128, heads * 128, 128, 1)

    def append_view(
        name: str, source: str, shape: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        body.append(
            f"    {name} = cute.make_tensor({source}.iterator, "
            f"layout=cute.make_layout({shape!r}, stride={stride!r}))"
        )

    append_view("_chunk_kd", kd, factor_shape, factor_stride)
    append_view("_chunk_qd", qd, factor_shape, factor_stride)
    append_view("_chunk_ak", ak, factor_shape, factor_stride)
    append_view("_chunk_aq", aq, aq_shape, aq_stride)
    append_view("_chunk_gt", gt, gt_shape, gt_stride)
    append_view("_chunk_v", values, activation_shape, activation_stride)
    append_view("_chunk_out", output, activation_shape, activation_stride)
    host_call = (
        "    _helion_sm100_chain_host("
        "_chunk_v, "
        f"{cu_seqlens}, {cu_chunks}, "
        "_chunk_kd, _chunk_qd, _chunk_ak, _chunk_aq, _chunk_gt, "
        f"{state}, _chunk_out, stream, "
        f"cutlass.Int32(0), cutlass.Int32({heads}), "
        f"cutlass.Float32({output_scale}), "
        "512)"
    )
    body.extend(("    _helion_cute_kernel_tag = 'chunk_recurrence_sm100'", host_call))

    if sequences <= 0:
        raise exc.BackendUnsupported("cute", "empty SM100 recurrence launch")


def gdn_recurrence_plan_int(plan: dict[str, object], key: str) -> int:
    value = plan.get(key)
    if type(value) is not int:
        raise exc.BackendUnsupported("cute", f"invalid gdn recurrence plan {key}")
    return value


GDN_RECURRENCE_HOST_CONSTEXPR_KEYS: tuple[str, ...] = (
    "block_v",
    "chunk",
    "dhead",
    "epilogue_warps",
    "stages",
    "token_groups",
    "mma_m",
    "state_col",
    "state_image_col",
    "acc_col",
    "update_image_col",
    "tmem_cols",
)


def validate_gdn_recurrence_plan(plan: dict[str, object]) -> None:
    """Re-prove the gdn recurrence schedule ABI carried by the wrapper plan."""

    from ...._compiler.cute.gdn_recurrence_geometry import GDN_RECURRENCE_DEVICE_ABI
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_cta_warps
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_epilogue_warp_choices
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_mma_m_choices
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_shape_admitted
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_smem_bytes
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_stage_choices
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_tmem_layout
    from ...._compiler.cute.gdn_recurrence_geometry import gdn_token_group_choices

    values = {
        key: gdn_recurrence_plan_int(plan, key)
        for key in (
            "batch",
            "heads",
            "seqlen",
            "num_chunks",
            "threads",
            "smem_bytes",
            "device_abi",
            *GDN_RECURRENCE_HOST_CONSTEXPR_KEYS,
        )
    }
    tmem = gdn_tmem_layout(values["dhead"], values["chunk"])
    if (
        values["device_abi"] != GDN_RECURRENCE_DEVICE_ABI
        or min(values["batch"], values["heads"], values["seqlen"]) <= 0
        or values["num_chunks"] != -(-values["seqlen"] // values["chunk"])
        or not gdn_shape_admitted(
            dhead=values["dhead"],
            chunk=values["chunk"],
            dstate=gdn_recurrence_plan_int(plan, "dstate"),
            block_v=values["block_v"],
        )
        or values["epilogue_warps"]
        not in gdn_epilogue_warp_choices(values["chunk"], values["dhead"])
        or values["mma_m"]
        not in gdn_mma_m_choices(
            values["chunk"],
            values["dhead"],
            values["block_v"],
            values["epilogue_warps"],
        )
        or values["stages"]
        not in gdn_stage_choices(
            values["chunk"], values["dhead"], values["block_v"], values["mma_m"]
        )
        or values["threads"]
        != 32
        * gdn_cta_warps(values["block_v"], values["epilogue_warps"], values["mma_m"])
        or values["token_groups"]
        not in gdn_token_group_choices(
            values["chunk"],
            values["block_v"],
            values["epilogue_warps"],
            values["mma_m"],
        )
        or values["smem_bytes"]
        != gdn_smem_bytes(
            values["chunk"],
            values["dhead"],
            values["block_v"],
            values["stages"],
            values["mma_m"],
        )
        or tmem is None
        or (
            values["state_col"],
            values["state_image_col"],
            values["acc_col"],
            values["update_image_col"],
            values["tmem_cols"],
        )
        != (
            tmem.state_col,
            tmem.state_image_col,
            tmem.acc_col,
            tmem.update_image_col,
            tmem.alloc_cols,
        )
    ):
        raise exc.BackendUnsupported("cute", "invalid gdn recurrence schedule ABI")


def append_gdn_recurrence_host_call(body: list[str], plan: dict[str, object]) -> None:
    """Call the gdn recurrence host schedule with the wrapper's tensor views."""

    validate_gdn_recurrence_plan(plan)
    tensors = ", ".join(
        f"arg{gdn_recurrence_plan_int(plan, key)}"
        for key in ("k_idx", "w_idx", "u_idx", "g_idx", "h_idx")
    )
    constexprs = ", ".join(
        str(gdn_recurrence_plan_int(plan, key))
        for key in GDN_RECURRENCE_HOST_CONSTEXPR_KEYS
    )
    body.extend(
        (
            "    _helion_cute_kernel_tag = 'gdn_recurrence_sm100'",
            f"    _helion_gdn_recurrence_host({tensors}, stream, {constexprs})",
        )
    )


def validate_gdn_recurrence_launch_args(
    plan: dict[str, object], args: tuple[object, ...]
) -> None:
    """Check the runtime tensors against the geometry baked into the plan."""

    from ...._compiler.cute.gdn_recurrence_geometry import GDN_TMA_ALIGNMENT_BYTES

    validate_gdn_recurrence_plan(plan)
    batch = gdn_recurrence_plan_int(plan, "batch")
    heads = gdn_recurrence_plan_int(plan, "heads")
    seqlen = gdn_recurrence_plan_int(plan, "seqlen")
    dhead = gdn_recurrence_plan_int(plan, "dhead")
    dstate = gdn_recurrence_plan_int(plan, "dstate")
    num_chunks = gdn_recurrence_plan_int(plan, "num_chunks")
    expected = (
        ("k_idx", torch.bfloat16, (batch, seqlen, heads, dhead)),
        ("w_idx", torch.bfloat16, (batch, seqlen, heads, dhead)),
        ("u_idx", torch.bfloat16, (batch, seqlen, heads, dstate)),
        ("g_idx", torch.float32, (batch, seqlen, heads)),
        ("h_idx", torch.bfloat16, (batch, num_chunks, heads, dhead, dstate)),
    )
    tensors: list[torch.Tensor] = []
    for key, dtype, shape in expected:
        index = gdn_recurrence_plan_int(plan, key)
        tensor = args[index] if 0 <= index < len(args) else None
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.device.type != "cuda"
        ):
            raise exc.BackendUnsupported(
                "cute", f"gdn recurrence {key} does not match the compiled geometry"
            )
        tensors.append(tensor)
    if any(tensor.device != tensors[0].device for tensor in tensors):
        raise exc.BackendUnsupported(
            "cute", "gdn recurrence tensors must share one CUDA device"
        )
    output = tensors[-1]
    output_begin = output.data_ptr()
    output_end = output_begin + output.numel() * output.element_size()
    for tensor in tensors[:-1]:
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        if begin < output_end and output_begin < end:
            raise exc.BackendUnsupported(
                "cute", "gdn recurrence output must not alias its inputs"
            )
    # k, w and u are read through TMA tensor maps.
    if any(tensor.data_ptr() % GDN_TMA_ALIGNMENT_BYTES for tensor in tensors[:3]):
        raise exc.BackendUnsupported("cute", "gdn recurrence TMA base is misaligned")


def append_sm100_warp_dv4_host_call(body: list[str], plan: dict[str, object]) -> None:
    """Launch the selected warp-HMMA DV4 schedule through its pinned host entry."""

    def plan_int(key: str) -> int:
        value = plan.get(key)
        if type(value) is not int:
            raise exc.BackendUnsupported(
                "cute", f"invalid SM100 warp-DV4 recurrence {key}"
            )
        return value

    def tensor_arg(key: str) -> str:
        return f"arg{plan_int(key)}"

    desc_args = plan.get("desc_args")
    if not isinstance(desc_args, (tuple, list)) or not all(
        isinstance(name, str) for name in desc_args
    ):
        raise exc.BackendUnsupported("cute", "invalid warp-DV4 descriptor arguments")

    heads = plan_int("heads")
    sequences = plan_int("sequences")
    total_tokens = plan_int("total_tokens")
    output = tensor_arg("out_idx")
    cu_seqlens = tensor_arg("cu_seqlens_idx")
    cu_chunks = tensor_arg("cu_chunks_idx")
    output_scale = f"arg{plan_int('scale_idx')}"
    body.extend(
        (
            (
                "    _chunk_output = cute.make_tensor("
                f"{output}.iterator, cute.make_layout("
                f"({total_tokens * heads * 128},), stride=(1,)))"
            ),
            (
                "    _chunk_cu_seqlens = cute.make_tensor("
                f"{cu_seqlens}.iterator, cute.make_layout("
                f"({sequences + 1},), stride=(1,)))"
            ),
            (
                "    _chunk_cu_chunks = cute.make_tensor("
                f"{cu_chunks}.iterator, cute.make_layout("
                f"({sequences + 1},), stride=(1,)))"
            ),
            "    _helion_cute_kernel_tag = 'chunk_recurrence_warp_dv4'",
            "    _helion_sm100_warp_dv4_host("
            + ", ".join(
                (
                    "_chunk_output",
                    "_chunk_cu_seqlens",
                    "_chunk_cu_chunks",
                    *cast("Sequence[str]", desc_args),
                    f"cutlass.Int32({heads})",
                    f"cutlass.Float32({output_scale})",
                    "grid_x",
                    "grid_y",
                    "stream",
                )
            )
            + ")",
        )
    )


# The prepare kernel's descriptors are all SWIZZLE_128B tiled maps.
ChunkPrepareTensorMapSpec = TensorMapSpec


def chunk_prepare_expected_tensor_map_specs(
    *,
    total_tokens: int,
    total_chunks: int,
    heads: int,
    q_ptr: int,
    k_ptr: int,
    gate_ptr: int,
    factor_ptr: int,
    gate_dtype: torch.dtype,
) -> tuple[ChunkPrepareTensorMapSpec, ...]:
    """Build descriptor geometry from already-validated ABI metadata."""

    def activation_spec(dtype: torch.dtype, pointer: int) -> ChunkPrepareTensorMapSpec:
        element_bytes = dtype.itemsize
        segment = 32 if dtype is torch.float32 else 64
        segments = 128 // segment
        return TensorMapSpec(
            dtype=dtype,
            base_ptr=pointer,
            global_dim=(segment, total_tokens, heads, segments),
            global_stride_bytes=(
                heads * 128 * element_bytes,
                128 * element_bytes,
                segment * element_bytes,
            ),
            box_dim=(segment, 16, 1, segments),
            swizzle="128B",
        )

    return (
        activation_spec(torch.bfloat16, q_ptr),
        activation_spec(torch.bfloat16, k_ptr),
        activation_spec(gate_dtype, gate_ptr),
        # The fused Kd/Qd/Ak slab the recurrence reads back through the same
        # geometry (``tensor_maps.factor_slab_geometry``).
        TensorMapSpec(
            dtype=torch.bfloat16,
            base_ptr=factor_ptr,
            swizzle="128B",
            **spec_fields(factor_slab_geometry(total_chunks * 16, heads)),
        ),
    )


def chunk_prepare_plan_tensor(
    plan: dict[str, object],
    args: tuple[object, ...],
    key: str,
) -> torch.Tensor:
    index = plan.get(key)
    if type(index) is not int or not 0 <= index < len(args):
        raise exc.BackendUnsupported("cute", f"invalid chunk-prepare {key}")
    tensor = args[index]
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"chunk-prepare {key} is not a tensor")
    return tensor


def chunk_prepare_tensor_map_specs(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> tuple[ChunkPrepareTensorMapSpec, ...]:
    """Validate the five-factor ABI and return the four exact TMA maps."""

    def plan_int(key: str) -> int:
        value = plan.get(key)
        if type(value) is not int:
            raise exc.BackendUnsupported("cute", f"invalid chunk-prepare {key}")
        return value

    schedule = plan.get("schedule")
    split_alias_schedules = tuple(
        f"split_alias_cpc{chunks_per_cta}" for chunks_per_cta in range(1, 6)
    )
    if (
        plan_int("device_abi") not in (3, 4)
        or plan_int("chunk_size") != 16
        or plan_int("key_size") != 128
        or plan_int("chunks_per_cta") not in (1, 2, 3, 4, 5)
        or (
            schedule in split_alias_schedules
            and schedule != f"split_alias_cpc{plan_int('chunks_per_cta')}"
        )
        or type(plan.get("outputs_scaled")) is not bool
        or plan_int("device_abi") != (3 if plan["outputs_scaled"] else 4)
        or plan.get("factor_key_xor") != (0 if plan["outputs_scaled"] else 8)
        or schedule not in split_alias_schedules
    ):
        raise exc.BackendUnsupported("cute", "unsupported chunk-prepare ABI")
    total_tokens = plan_int("total_tokens")
    total_chunks = plan_int("total_chunks")
    heads = plan_int("heads")
    if total_tokens <= 0 or total_chunks <= 0 or heads <= 0:
        raise exc.BackendUnsupported("cute", "empty chunk-prepare launch")

    names = (
        "q_idx",
        "k_idx",
        "g_idx",
        "beta_idx",
        "a_log_idx",
        "dt_idx",
        "cu_seqlens_idx",
        "cu_chunks_idx",
        "chunk_to_seq_idx",
        "kd_idx",
        "qd_idx",
        "ak_idx",
        "aq_idx",
        "gt_idx",
    )
    (
        q,
        k,
        gate,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
        kd,
        qd,
        ak,
        aq,
        g_total,
    ) = tuple(chunk_prepare_plan_tensor(plan, args, name) for name in names)
    tensors = (
        q,
        k,
        gate,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
        kd,
        qd,
        ak,
        aq,
        g_total,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise exc.BackendUnsupported("cute", "chunk-prepare tensors span devices")
    if q.device.type != "cuda" or any(not tensor.is_contiguous() for tensor in tensors):
        raise exc.BackendUnsupported(
            "cute", "chunk-prepare tensors must be contiguous CUDA tensors"
        )

    token_rows = total_tokens * heads
    rows = total_chunks * 16
    expected = (
        (q, torch.bfloat16, (token_rows, 128)),
        (k, torch.bfloat16, (token_rows, 128)),
        (
            gate,
            torch.float32 if bool(plan.get("gate_is_fp32")) else torch.bfloat16,
            (token_rows, 128),
        ),
        (beta, torch.bfloat16, (token_rows,)),
        (a_log, torch.float32, (heads,)),
        (dt_bias, torch.float32, (heads, 128)),
        (chunk_to_seq, torch.int32, (total_chunks,)),
        (kd, torch.bfloat16, (heads * rows, 128)),
        (qd, torch.bfloat16, (heads * rows, 128)),
        (ak, torch.bfloat16, (heads * rows, 128)),
        (aq, torch.bfloat16, (heads * total_chunks, 256)),
        (g_total, torch.float32, (heads * total_chunks, 128)),
    )
    for tensor, dtype, shape in expected:
        if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
            raise exc.BackendUnsupported(
                "cute", "chunk-prepare tensor dtype/shape ABI mismatch"
            )
    if (
        cu_seqlens.dtype is not torch.int32
        or cu_chunks.dtype is not torch.int32
        or cu_seqlens.ndim != 1
        or tuple(cu_chunks.shape) != tuple(cu_seqlens.shape)
    ):
        raise exc.BackendUnsupported("cute", "chunk-prepare metadata ABI mismatch")

    vector_bytes = heads * rows * 128 * 2
    aq_bytes = heads * total_chunks * 256 * 2
    kd_ptr = kd.data_ptr()
    expected_ptrs = (
        kd_ptr,
        kd_ptr + vector_bytes,
        kd_ptr + 2 * vector_bytes,
        kd_ptr + 3 * vector_bytes,
        kd_ptr + 3 * vector_bytes + aq_bytes,
    )
    actual_ptrs = tuple(tensor.data_ptr() for tensor in (kd, qd, ak, aq, g_total))
    output_storages = tuple(
        tensor.untyped_storage() for tensor in (kd, qd, ak, aq, g_total)
    )
    storage_bases = tuple(storage.data_ptr() for storage in output_storages)
    workspace_bytes = (
        3 * vector_bytes + aq_bytes + g_total.numel() * g_total.element_size()
    )
    if (
        kd_ptr % 256
        or actual_ptrs != expected_ptrs
        or len(set(storage_bases)) != 1
        or storage_bases[0] != kd_ptr
        or any(storage.nbytes() < workspace_bytes for storage in output_storages)
    ):
        raise exc.BackendUnsupported(
            "cute", "chunk-prepare factors must use the exact packed workspace ABI"
        )

    output_begin = kd_ptr
    output_end = g_total.data_ptr() + g_total.numel() * g_total.element_size()
    for tensor in (
        q,
        k,
        gate,
        beta,
        a_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
    ):
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        if begin < output_end and output_begin < end:
            raise exc.BackendUnsupported("cute", "chunk-prepare input aliases output")
    if any(tensor.data_ptr() % 16 for tensor in (q, k, gate, kd)):
        raise exc.BackendUnsupported("cute", "chunk-prepare TMA base is misaligned")

    return chunk_prepare_expected_tensor_map_specs(
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        heads=heads,
        q_ptr=q.data_ptr(),
        k_ptr=k.data_ptr(),
        gate_ptr=gate.data_ptr(),
        factor_ptr=kd_ptr,
        gate_dtype=gate.dtype,
    )


def encode_chunk_prepare_tensor_map(spec: TensorMapSpec) -> bytes:
    rank = len(spec.global_dim)
    if (
        rank not in (3, 4)
        or len(spec.global_stride_bytes) != rank - 1
        or len(spec.box_dim) != rank
    ):
        raise exc.BackendUnsupported("cute", "invalid chunk-prepare tensor-map rank")
    try:
        return spec.encode()
    except (RuntimeError, ValueError) as error:
        raise exc.BackendUnsupported(
            "cute", f"chunk-prepare tensor-map encoding failed: {error}"
        ) from error


def build_chunk_prepare_tensor_maps(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    specs = chunk_prepare_tensor_map_specs(plan, args)
    payload = bytearray().join(encode_chunk_prepare_tensor_map(spec) for spec in specs)
    q = chunk_prepare_plan_tensor(plan, args, "q_idx")
    storage = torch.frombuffer(payload, dtype=torch.uint8).clone().to(q.device)
    if storage.data_ptr() % 64:
        raise exc.BackendUnsupported(
            "cute", "chunk-prepare tensor-map storage is misaligned"
        )
    base = storage.data_ptr()
    return storage, cast(
        "tuple[int, int, int, int]",
        tuple(base + 128 * index for index in range(4)),
    )


def chunk_recurrence_plan_tensor(
    plan: dict[str, object],
    args: tuple[object, ...],
    key: str,
) -> torch.Tensor:
    index = plan.get(key)
    if type(index) is not int or not 0 <= index < len(args):
        raise exc.BackendUnsupported("cute", f"invalid chunk-recurrence {key}")
    tensor = args[index]
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"chunk-recurrence {key} is not a tensor")
    return tensor


def chunk_recurrence_tensor_map_specs(
    plan: dict[str, object],
    args: tuple[object, ...],
    *,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate v2-workspace recurrence arguments and describe DV4 TensorMaps."""

    def plan_int(key: str) -> int:
        value = plan.get(key)
        if type(value) is not int:
            raise exc.BackendUnsupported("cute", f"invalid chunk-recurrence {key}")
        return value

    kind = plan.get("kind")
    if (
        kind not in ("chunk_recurrence_sm100", "chunk_recurrence_warp_dv4")
        or plan_int("workspace_layout_version") != 2
        or plan.get("outputs_scaled") is not False
        or plan.get("factor_key_xor") != 8
        or plan_int("chunk_size") != 16
        or plan_int("key_size") != 128
        or plan_int("value_size") != 128
    ):
        raise exc.BackendUnsupported("cute", "unsupported chunk-recurrence ABI")
    total_tokens = plan_int("total_tokens")
    total_chunks = plan_int("total_chunks")
    heads = plan_int("heads")
    sequences = plan_int("sequences")
    if min(total_tokens, total_chunks, heads, sequences) <= 0:
        raise exc.BackendUnsupported("cute", "empty chunk-recurrence launch")

    names = (
        "kd_idx",
        "qd_idx",
        "ak_idx",
        "aq_idx",
        "gt_idx",
        "v_idx",
        "out_idx",
        "state_idx",
        "cu_seqlens_idx",
        "cu_chunks_idx",
    )
    (
        kd,
        qd,
        ak,
        aq,
        g_total,
        values,
        output,
        state,
        cu_seqlens,
        cu_chunks,
    ) = tuple(chunk_recurrence_plan_tensor(plan, args, name) for name in names)
    tensors = (kd, qd, ak, aq, g_total, values, output, state, cu_seqlens, cu_chunks)
    if kd.device.type != "cuda" or any(
        tensor.device != kd.device for tensor in tensors
    ):
        raise exc.BackendUnsupported(
            "cute", "chunk-recurrence tensors must share one CUDA device"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise exc.BackendUnsupported(
            "cute", "chunk-recurrence tensors must be contiguous"
        )

    rows = total_chunks * 16
    expected = (
        (kd, torch.bfloat16, (heads * rows, 128)),
        (qd, torch.bfloat16, (heads * rows, 128)),
        (ak, torch.bfloat16, (heads * rows, 128)),
        (aq, torch.bfloat16, (heads * total_chunks, 256)),
        (g_total, torch.float32, (heads * total_chunks, 128)),
        (values, torch.bfloat16, (total_tokens * heads, 128)),
        (output, torch.bfloat16, (total_tokens * heads, 128)),
        (state, torch.bfloat16, (sequences, heads, 128, 128)),
        (cu_seqlens, torch.int32, (sequences + 1,)),
        (cu_chunks, torch.int32, (sequences + 1,)),
    )
    for tensor, dtype, shape in expected:
        if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
            raise exc.BackendUnsupported(
                "cute", "chunk-recurrence tensor dtype/shape ABI mismatch"
            )

    vector_bytes = heads * rows * 128 * 2
    aq_bytes = heads * total_chunks * 256 * 2
    kd_ptr = kd.data_ptr()
    expected_ptrs = (
        kd_ptr,
        kd_ptr + vector_bytes,
        kd_ptr + 2 * vector_bytes,
        kd_ptr + 3 * vector_bytes,
        kd_ptr + 3 * vector_bytes + aq_bytes,
    )
    factor_tensors = (kd, qd, ak, aq, g_total)
    actual_ptrs = tuple(tensor.data_ptr() for tensor in factor_tensors)
    storages = tuple(tensor.untyped_storage() for tensor in factor_tensors)
    storage_bases = tuple(storage.data_ptr() for storage in storages)
    workspace_bytes = (
        3 * vector_bytes + aq_bytes + g_total.numel() * g_total.element_size()
    )
    if (
        kd_ptr % 256
        or actual_ptrs != expected_ptrs
        or len(set(storage_bases)) != 1
        or storage_bases[0] != kd_ptr
        or any(storage.nbytes() < workspace_bytes for storage in storages)
    ):
        raise exc.BackendUnsupported(
            "cute", "chunk-recurrence factors must use the exact packed workspace ABI"
        )

    def byte_range(tensor: torch.Tensor) -> tuple[int, int]:
        begin = tensor.data_ptr()
        return begin, begin + tensor.numel() * tensor.element_size()

    def overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        lhs_begin, lhs_end = byte_range(lhs)
        rhs_begin, rhs_end = byte_range(rhs)
        return lhs_begin < rhs_end and rhs_begin < lhs_end

    value_range = byte_range(values)
    output_range = byte_range(output)
    if value_range != output_range and overlaps(values, output):
        raise exc.BackendUnsupported(
            "cute", "chunk-recurrence value/output must alias exactly or be disjoint"
        )
    for name, tensor in (("cu_seqlens", cu_seqlens), ("cu_chunks", cu_chunks)):
        if overlaps(output, tensor):
            raise exc.BackendUnsupported(
                "cute", f"chunk-recurrence output aliases {name}"
            )
    for tensor in (*factor_tensors, values, output, cu_seqlens, cu_chunks):
        if overlaps(state, tensor):
            raise exc.BackendUnsupported("cute", "chunk-recurrence state aliases input")
    factor_begin = kd_ptr
    factor_end = kd_ptr + workspace_bytes
    for tensor in (values, output, state, cu_seqlens, cu_chunks):
        begin, end = byte_range(tensor)
        if begin < factor_end and factor_begin < end:
            raise exc.BackendUnsupported(
                "cute", "chunk-recurrence factor workspace aliases another argument"
            )
    if any(
        tensor.data_ptr() % 16 for tensor in (kd, aq, g_total, values, output, state)
    ):
        raise exc.BackendUnsupported("cute", "chunk-recurrence TMA base is misaligned")

    if validate_only:
        return {}

    # The DV2 host schedule builds CuTe TensorMaps from the validated tensor
    # views itself.  Only the external warp-DV4 schedule consumes raw encoded
    # descriptors owned by this launcher.
    if kind != "chunk_recurrence_warp_dv4":
        raise exc.BackendUnsupported(
            "cute", "raw chunk-recurrence TensorMaps require the warp-DV4 schedule"
        )

    from ...._compiler.cute.chunk_recurrence_dv4_sm100 import (
        recurrence_tensor_map_specs,
    )

    return recurrence_tensor_map_specs(
        kd_ptr=kd_ptr,
        aq_ptr=aq.data_ptr(),
        gt_ptr=g_total.data_ptr(),
        v_ptr=values.data_ptr(),
        out_ptr=output.data_ptr(),
        heads=heads,
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        sequences=sequences,
        state_ptr=state.data_ptr(),
    )


def build_chunk_recurrence_tensor_maps(
    plan: dict[str, object],
    args: tuple[object, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int, int]]:
    if plan.get("kind") != "chunk_recurrence_warp_dv4":
        raise exc.BackendUnsupported("cute", "unsupported chunk-recurrence plan")
    from ...._compiler.cute.chunk_recurrence_dv4_sm100 import ROLES

    specs = chunk_recurrence_tensor_map_specs(plan, args)
    kd = chunk_recurrence_plan_tensor(plan, args, "kd_idx")
    try:
        tensor_maps = build_tensor_maps(specs, kd.device)
    except (RuntimeError, ValueError) as error:
        raise exc.BackendUnsupported(
            "cute", f"chunk-recurrence tensor-map construction failed: {error}"
        ) from error
    return tensor_maps.storage, cast(
        "tuple[int, int, int, int, int, int, int]",
        tuple(tensor_maps.address(role) for role in ROLES),
    )


def chunk_recurrence_sm100_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute.chunk_recurrence_sm100 import host_chain_dv2

    return {"_helion_sm100_chain_host": host_chain_dv2}


def chunk_recurrence_warp_dv4_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute.chunk_recurrence_dv4_sm100 import _recurrence_entry

    return {"_helion_sm100_warp_dv4_host": _recurrence_entry}


def gdn_recurrence_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute.gdn_recurrence_sm100 import host_gdn_recurrence

    return {"_helion_gdn_recurrence_host": host_gdn_recurrence}


EMITTERS = (
    HostWrapperEmitter(
        "chunk_recurrence_sm100",
        emit_chunk_recurrence_sm100,
        namespace=chunk_recurrence_sm100_namespace,
        host_call=append_sm100_chunk_recurrence_host_call,
        source_dependencies=("_compiler/cute/chunk_recurrence_sm100.py",),
    ),
    HostWrapperEmitter(
        "chunk_recurrence_warp_dv4",
        emit_chunk_recurrence_warp_dv4,
        namespace=chunk_recurrence_warp_dv4_namespace,
        host_call=append_sm100_warp_dv4_host_call,
        source_dependencies=(
            "_compiler/cute/affine_recurrence_primitives.py",
            "_compiler/cute/chunk_recurrence_dv4_sm100.py",
            "_compiler/cute/device/ptx.py",
        ),
    ),
    HostWrapperEmitter(
        "gdn_recurrence_sm100",
        emit_gdn_recurrence_sm100,
        namespace=gdn_recurrence_namespace,
        host_call=append_gdn_recurrence_host_call,
        source_dependencies=(
            "_compiler/cute/gdn_recurrence_sm100.py",
            "_compiler/cute/gdn_recurrence_geometry.py",
            "_compiler/cute/affine_recurrence_primitives.py",
            "_compiler/cute/device/ptx.py",
        ),
    ),
    HostWrapperEmitter("chunk_prepare_tma", emit_chunk_prepare_tma),
)
