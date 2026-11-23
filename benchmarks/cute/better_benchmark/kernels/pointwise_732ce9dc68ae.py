"""better-benchmark repro pointwise_732ce9dc68ae (hf/infer YituTechConvBert).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_clone_constant_pad_nd_full_index_iota_permute_unsqueeze_view_0``:
the sliding windows of ConvBert's span-based dynamic convolution (an unfold
with kernel 9 and padding 4 along the sequence), ``out[b, s, c, k] =
x[b, s + k - 4, c]`` for source positions inside the sequence and zero in the
padding, of a contiguous bf16 ``[batch * seq, channels]`` activation. The
graph's views, permutes, the intermediate clone, the zero padding and the
``iota(512) + iota(9)`` index grid all fold into the one kernel. It runs over
two dimensions, the input's flat index ``y`` (``batch * seq * channels``
elements) and the window slot ``x``: the source position is
``(y // channels) % seq + x - 4``, and one bf16 load at flat offset
``y + channels * (x - 4)``, masked to source positions inside the sequence
(the masked lanes read as the padding's zero) and widened to fp32, feeds one
bf16 store at ``9 * y + x`` of the contiguous ``[batch, seq, channels, 9]``
output. The port tiles the same two dimensions with the same index math and
the same masked load; the output is returned through the graph's
``[batch * seq * heads, head_dim, 9]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def sliding_windows(
    x: torch.Tensor, window: hl.constexpr, pad: hl.constexpr
) -> torch.Tensor:
    # x: [batch, seq, channels], contiguous.
    _, seq, channels = x.shape
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n, window], dtype=x.dtype, device=x.device)
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        # constant_pad_nd(..., [0, 0, pad, pad]) indexed with
        # iota(seq)[None, :] + iota(window)[:, None]: slot k of position s
        # reads position s + k - pad, the padding's zero outside the sequence.
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        v = hl.load(flat, [offset], extra_mask=in_seq).to(torch.float32)
        out[tile_n, tile_k] = v.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    seq_shape = inputs[1]  # _shape_param_0 = [batch, seq, channels]
    padding = inputs[2]  # _shape_param_1 = [0, 0, pad, pad]
    window_shape = inputs[4]  # _shape_param_3 = [batch, seq, channels, window]
    out = sliding_windows(arg0_1.view(seq_shape), window_shape[3], padding[2])
    # view_3 and expand: [batch * seq * heads, head_dim, window].
    return (out.view(inputs[5]).expand(inputs[6]),)
