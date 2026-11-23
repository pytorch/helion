"""better-benchmark repro pointwise_fb4ca2518d23 (timm nfnet_l0).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_div_exp_mul_neg_sigmoid_0``: the
end of an NFNet block and the activation that opens the next one. The
squeeze-and-excitation gate (``sigmoid`` of a bf16 ``[N, C, 1, 1]`` tensor)
scales the channels-last bf16 ``[N, C, H, W]`` branch output, which is then
multiplied by ``2.0`` and by ``0.2`` (the block's ``alpha``) and added to the
channels-last bf16 shortcut; that sum is the first output. The second output
is the scaled SiLU of the sum, ``sum / (exp(-sum) + 1) * 0.9622504486493761``.
Both are stored in bf16, channels-last.

Inductor indexes the dense storage as one flat dimension (the branch, the
shortcut and both outputs at the flat index; the gate at
``(index // (H*W*C)) * C + index % C``, i.e. three loads and two stores per
element) and computes everything in fp32: none of the graph's bf16 roundings
is materialized (no ``emulate_precision_casts``), so the SiLU reads the fp32
sum rather than the bf16 value it stores as the first output. The port tiles
the same storage as ``[N, H*W, C]`` (views of the channels-last tensors), loads
the ``[N, C]`` gate once per batch/channel tile and broadcasts it over the
spatial dim, and writes two ``[N, H*W, C]`` outputs the host views back to the
channels-last ``[N, C, H, W]``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def se_scale_add_silu(
    x: torch.Tensor, gate: torch.Tensor, shortcut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # x, shortcut: [N, H*W, C], views of the channels-last tensors; gate: [N, C].
    batch, plane, channels = x.shape
    out_add = torch.empty([batch, plane, channels], dtype=x.dtype, device=x.device)
    out_act = torch.empty([batch, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_s, tile_c in hl.tile([batch, plane, channels]):
        scale = torch.sigmoid(gate[tile_n, tile_c].to(torch.float32))
        y = x[tile_n, tile_s, tile_c].to(torch.float32) * scale[:, None, :]
        y = y * 2.0
        y = y * 0.2
        y = y + shortcut[tile_n, tile_s, tile_c].to(torch.float32)
        out_add[tile_n, tile_s, tile_c] = y.to(out_add.dtype)
        silu = y / (torch.exp(-y) + 1)
        out_act[tile_n, tile_s, tile_c] = (silu * 0.9622504486493761).to(out_act.dtype)
    return out_add, out_act


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    gate, x, shortcut = inputs[:3]
    n, c, h, w = x.shape
    out_add, out_act = se_scale_add_silu(
        x.permute(0, 2, 3, 1).view(n, h * w, c),
        gate.view(n, c),
        shortcut.permute(0, 2, 3, 1).view(n, h * w, c),
    )
    return (
        out_add.view(n, h, w, c).permute(0, 3, 1, 2),
        out_act.view(n, h, w, c).permute(0, 3, 1, 2),
    )
