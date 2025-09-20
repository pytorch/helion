from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

helion.runtime.set_triton_allocator()

@triton.jit
def _helion_kl_div_forward(y_pred, y_true, kl_loss, loss, log_target, eps, _NUM_SM: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    y_pred_desc = tl.make_tensor_descriptor(y_pred, [4096, 4096], [4096, 1], [_BLOCK_SIZE_1, _BLOCK_SIZE_0])
    y_true_desc = tl.make_tensor_descriptor(y_true, [4096, 4096], [4096, 1], [_BLOCK_SIZE_1, _BLOCK_SIZE_0])
    kl_loss_desc = tl.make_tensor_descriptor(kl_loss, [4096, 4096], [4096, 1], [_BLOCK_SIZE_1, _BLOCK_SIZE_0])
    total_pids = 4096
    for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM, loop_unroll_factor=1, num_stages=2):
        pid_0 = virtual_pid
        offset_1 = pid_0
        indices_1 = offset_1 + tl.zeros([1], tl.int32)
        loss_sum = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0.0, tl.float32)
        for offset_0 in tl.range(0, 4096, _BLOCK_SIZE_0, loop_unroll_factor=2, num_stages=3, disallow_acc_multi_buffer=False, flatten=True):
            loss_sum_copy = loss_sum
            loss_sum_copy_0 = loss_sum_copy
            y_pred_val = y_pred_desc.load([offset_1, offset_0])
            y_true_val = y_true_desc.load([offset_1, offset_0])
            if log_target:
                y_true_val_copy = y_true_val
                y_pred_val_copy = y_pred_val
                y_true_val_copy_0 = y_true_val_copy
                y_pred_val_copy_0 = y_pred_val_copy
                v_0 = libdevice.exp(y_true_val_copy_0)
                v_1 = y_true_val_copy_0 - y_pred_val_copy_0
                v_2 = v_0 * v_1
                kl_loss_desc.store([offset_1, offset_0], v_2)
            _not = not log_target
            if _not:
                y_true_val_copy_1 = y_true_val
                y_pred_val_copy_1 = y_pred_val
                y_true_val_copy_1_0 = y_true_val_copy_1
                y_pred_val_copy_1_0 = y_pred_val_copy_1
                v_3 = triton_helpers.maximum(y_true_val_copy_1_0, eps)
                v_4 = tl_math.log(v_3)
                v_5 = v_4 - y_pred_val_copy_1_0
                v_6 = y_true_val_copy_1_0 * v_5
                kl_loss_desc.store([offset_1, offset_0], v_6)
            load_2 = kl_loss_desc.load([offset_1, offset_0])
            loss_sum = loss_sum_copy_0 + load_2
        sum_1 = tl.cast(tl.sum(loss_sum, 1), tl.float32)
        tl.store(loss + indices_1 * 1, sum_1, None)

def kl_div_forward(y_pred: Tensor, y_true: Tensor, log_target: bool=False, reduction: str='batchmean', eps: float=1e-10, *, _launcher=_default_launcher):
    """
    Compute KL Divergence loss.

    Args:
        y_pred: Input predictions in log-space, shape (BT, V)
        y_true: Target values (probabilities or log-probabilities), shape (BT, V)
        log_target: If True, y_true is in log-space; if False, y_true is probabilities
        reduction: Reduction mode ('none', 'sum', 'mean', 'batchmean')
        eps: Small value to avoid numerical issues

    Returns:
        loss: KL divergence loss
    """
    BT, V = y_pred.shape
    assert y_true.shape == y_pred.shape, f'Shape mismatch: {y_true.shape} != {y_pred.shape}'
    if reduction == 'none':
        loss = torch.zeros_like(y_pred)
    else:
        loss = torch.zeros((BT,), dtype=torch.float32, device=y_pred.device)
    kl_loss = torch.zeros_like(y_pred)
    BT_SIZE = helion.cdiv(BT, BT)
    _NUM_SM = helion.runtime.get_num_sm(y_pred.device)
    _BLOCK_SIZE_0 = 128
    torch.cuda.synchronize()
    _launcher(_helion_kl_div_forward, (_NUM_SM,), y_pred, y_true, kl_loss, loss, log_target, eps, _NUM_SM, 1, _BLOCK_SIZE_0, num_warps=1, num_stages=5)
    torch.cuda.synchronize()
    if reduction == 'batchmean':
        final_loss = torch.sum(loss) / BT
    elif reduction == 'sum':
        final_loss = torch.sum(loss, dim=0)
    elif reduction == 'mean':
        final_loss = torch.sum(loss) / (BT * V)
    else:
        final_loss = loss
    return final_loss

def call():
    from torch._dynamo.testing import rand_strided
    y_pred = rand_strided(size=(4096, 4096), stride=(4096, 1), dtype=torch.float32, device='cuda:0')
    y_true = rand_strided(size=(4096, 4096), stride=(4096, 1), dtype=torch.float32, device='cuda:0')
    log_target = False
    reduction = 'batchmean'
    eps = 1.1
    kl_div_forward(y_pred, y_true, log_target, reduction, eps)

if __name__ == '__main__':
    call()
