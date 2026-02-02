# Gated Delta Rule - Naive/Reference implementations
# Standalone implementations without fla dependencies

import torch


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Naive recurrent implementation of Gated Delta Rule.

    Args:
        q: queries of shape [B, T, H, K]
        k: keys of shape [B, T, H, K]
        v: values of shape [B, T, H, V]
        beta: update gates of shape [B, T, H]
        g: decay gates (log space) of shape [B, T, H]
        scale: scaling factor (default: 1/sqrt(K))
        initial_state: initial state of shape [B, H, K, V]
        output_final_state: whether to return final state

    Returns:
        o: output of shape [B, T, H, V]
        h: final state of shape [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K = k.shape
    V = v.shape[-1]

    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.clone()

    if scale is None:
        scale = 1 / (K ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()

        # Apply decay gate
        h = h.clone() * g[:, :, i].exp()[..., None, None]

        # Apply delta rule update
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)

        # Memory readout
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None

    o = o.transpose(1, 2).contiguous()
    return o, h
