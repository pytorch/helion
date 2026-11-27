"""The driver's kernel builds without vLLM: vllm_helion_verify.py's own _Runner and
forward over a synthetic Inkling-2-Small TP4 model (rank 0's shard: the vLLM
Inkling 2 branch's shapes, dtypes and layouts, uninitialized values), with stub
vllm modules for its imports and each launch key's first launch recorded instead
of run (later launches do nothing)."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING
from typing import NamedTuple
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.distributed as dist

H, LAYERS, INTER = 4608, 48, 2304
HEADS, KV_HEADS, HEAD_DIM, D_REL = 12, 2, 128, 16  # a rank's attention heads
WINDOW = 1024  # sliding_window_size, also rel_extent
EXPERTS, ROUTED, TAPS = 96, 384, 4  # a rank's experts of the routed ones; conv taps
MXFP8 = range(40, 48)  # the layers with MXFP8 experts; the others' are NVFP4
PAGE, CONV_HEAD = 128, 8192  # KV cache page; conv-state width a KV head
KV_BLOCKS = 32000  # --num-gpu-blocks-override (keeps the cache views int32-indexable)
MAX_LEN = 16384  # --max-model-len: the block tables' widths
_DRIVER = "pretuned_kernels.megakernels.inkling2_small.vllm_helion_verify"


class Launch(NamedTuple):
    """A kernel build of the driver (``_config.variant_kernel``'s arguments, from
    its first launch), standing in for the kernel: launching it does nothing."""

    fn: Callable[..., object]
    args: tuple[object, ...]
    config: str
    tma: str

    def __call__(self, *args: object) -> None:
        pass


def served_builds(
    group: dist.ProcessGroup, device: torch.device
) -> dict[object, Launch]:
    """The driver's kernel builds by its key ((kernel, the next layer's attention
    inputs) or the attention kernel): those of the served forward, then those of
    INK_HELION_LAYERS=0 (an attention and a MoE kernel a layer)."""

    def empty(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    model = _model(empty)
    metadata = {}
    for layer in model.layers:
        metadata[layer.attn.prefix] = NS(
            block_table=empty(1, MAX_LEN // PAGE, dtype=torch.int32),
            seq_lens=empty(1, dtype=torch.int32),
            slot_mapping=empty(8, dtype=torch.int64),
        )
        metadata[layer.conv_state.prefix] = NS(
            block_table=empty(1, MAX_LEN // TAPS, dtype=torch.int32),
            slot_mapping=empty(8, dtype=torch.int64),
        )
    tp = NS(device_group=group, rank_in_group=group.rank())
    stubs = {
        "vllm": NS(),
        "vllm.forward_context": NS(get_forward_context=None),
        "vllm.distributed": NS(get_tp_group=lambda: tp),
    }
    saved = {name: sys.modules.get(name) for name in [*stubs, _DRIVER]}
    sys.modules.update(stubs)  # pyrefly: ignore[no-matching-overload]
    sys.modules.pop(_DRIVER, None)
    # A fresh import: the driver reads its INK_HELION_* settings at import, the
    # served builds are their defaults'.
    env = {k: v for k, v in os.environ.items() if not k.startswith("INK_HELION")}
    try:
        with patch.dict(os.environ, env, clear=True):
            driver = importlib.import_module(_DRIVER)
        runner = driver._Runner(model, device)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    inputs_embeds = empty(8, H, dtype=torch.bfloat16)
    with (
        patch.object(driver, "_variant", Launch),
        patch.object(driver, "get_forward_context", lambda: NS(attn_metadata=metadata)),
    ):
        runner.forward(inputs_embeds)
        with patch.multiple(driver, LAYER_MOD="", LAYER_MX_MOD="", LAYER_BLOCK_MOD=""):
            runner.forward(inputs_embeds)
    return runner.kernels


class _Norm(NS):
    def __call__(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return x


def _model(empty: Callable[..., torch.Tensor]) -> NS:
    """The model as the driver reads it (vLLM's Inkling2 modules' attributes)."""
    # vLLM's paged caches, slot-major ([block][slot][head][width] in memory).
    kv = empty(KV_BLOCKS, PAGE, KV_HEADS, 2 * HEAD_DIM).transpose(1, 2)
    conv = empty(KV_BLOCKS, TAPS, KV_HEADS, CONV_HEAD).transpose(1, 2)
    width = H // KV_HEADS  # a KV head's attention / MLP conv streams
    streams = ((0, HEAD_DIM), (HEAD_DIM, HEAD_DIM), (2 * HEAD_DIM, width))
    streams += ((2 * HEAD_DIM + width, width),)
    e4m3, u8, f32 = torch.float8_e4m3fn, torch.uint8, torch.float32
    experts = {True: {}, False: {}}  # NVFP4, MXFP8
    for name, n, k in (("w13", 2 * INTER, H), ("w2", H, INTER)):
        # NVFP4: e2m1 pairs, E4M3 scales of 16 values, amax; MXFP8: E4M3, e8m0 of 32
        experts[True][f"{name}_weight"] = empty(EXPERTS, n, k // 2, dtype=u8)
        experts[True][f"{name}_weight_scale"] = empty(EXPERTS, n, k // 16, dtype=e4m3)
        experts[True][f"{name}_weight_amax"] = empty(EXPERTS, dtype=f32)
        experts[True][f"{name}_weight_outer_scale_denominator"] = empty(1, dtype=f32)
        experts[False][f"{name}_weight"] = empty(EXPERTS, n, k, dtype=e4m3)
        experts[False][f"{name}_weight_scale"] = empty(EXPERTS, n, k // 32, dtype=u8)
    gate = NS(
        weight=empty(ROUTED, H), bias=empty(ROUTED, dtype=f32), topk=8, route_scale=8.0
    )
    q, kv_cols = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    split = [q, HEADS * D_REL, q, HEADS, kv_cols, kv_cols, 4]  # q r gate sinks k v pad

    def layer(local: bool, nvfp4: bool) -> NS:
        def sconv() -> NS:  # local: static taps; global: token-dependent ones
            if local:
                return NS(conv_weight=empty(kv_cols, TAPS))
            return NS(
                conv_weight=empty(TAPS, kv_cols), weight_proj=empty(KV_HEADS * TAPS, H)
            )

        prefix = f"{'local' if local else 'global'}_{'nvfp4' if nvfp4 else 'mxfp8'}"
        inputs = 1 if local else 3
        attn = NS(
            prefix=f"{prefix}.attn",
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            rel_extent=WINDOW,
            split_sizes=split,
            qkvr=NS(weight=empty(sum(split), H)),
            wo_ud=NS(weight=empty(H, q)),
            rel_logits_proj=empty(D_REL, WINDOW),
            q_scale=empty(HEADS, HEAD_DIM),
            r_scale=empty(HEADS, D_REL),
            k_sconv=sconv(),
            v_sconv=sconv(),
            kv_cache=kv,
        )
        return NS(
            is_local=local,
            num_attn_inputs=inputs,
            attn=attn,
            conv_state=NS(
                prefix=f"{prefix}.conv", kv_cache=conv, stream_ranges=streams
            ),
            mlp=NS(
                gate=gate,
                experts=NS(nvfp4=nvfp4, gate_cap=4.0, up_cap=25.0, **experts[nvfp4]),
            ),
            attn_norm=_Norm(weight=empty(inputs, H)),
            mlp_norm=NS(weight=empty(1, H)),
            attn_sconv=NS(conv_weight=empty(H, TAPS)),
            mlp_sconv=NS(conv_weight=empty(H, TAPS)),
        )

    # Every sixth layer, from layer 5, is a global one.
    kinds = {(lid % 6 != 5, lid not in MXFP8) for lid in range(LAYERS)}
    layers = {kind: layer(*kind) for kind in kinds}
    return NS(
        config=NS(hidden_size=H, sliding_window_size=WINDOW, intermediate_size=INTER),
        attn_res=NS(
            attn_queries=[
                empty(1 if i % 6 != 5 else 3, H, dtype=f32) for i in range(1, LAYERS)
            ],
            mlp_queries=[empty(1, H, dtype=f32) for _ in range(LAYERS)],
            output_query=empty(1, H, dtype=f32),
        ),
        layers=[layers[lid % 6 != 5, lid not in MXFP8] for lid in range(LAYERS)],
        norm=NS(weight=empty(1, H)),
        aux_hidden_state_layers=(),
    )
