# === HELION KERNEL REPRO ===
import helion
import helion.language as hl
import torch

@helion.kernel(autotune_effort="none", static_shapes=True)
def layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Minimal kernel that triggers the Triton tensor numel limit."""

    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))

    num_blocks = (x.size(0) + m_block - 1) // m_block
    grad_weight_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)

    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        grad_w_acc = x.new_zeros(n, dtype=torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            x_mb = x[mb, :]
            dy_mb = grad_out[mb, :]
            grad_w_acc += torch.sum(dy_mb * x_mb, dim=0)

        grad_weight_blocks[mb_cta.id, :] = grad_w_acc

    grad_weight = grad_weight_blocks.sum(0)
    return grad_weight

def helion_repro_caller():
    m = 32
    n = 32769
    grad_out = torch.ones((m, n), dtype=torch.float32, device='cuda:0')
    x = torch.ones((m, n), dtype=torch.float32, device='cuda:0')
    return layer_norm_bwd(grad_out, x)

helion_repro_caller()
# === END HELION KERNEL REPRO ===
