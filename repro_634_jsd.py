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
def _helion_jsd_forward(_input, target, loss, dX, beta, _NUM_SM: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    _input_desc = tl.make_tensor_descriptor(_input, [8192, 4096], [4096, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    target_desc = tl.make_tensor_descriptor(target, [8192, 4096], [4096, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    loss_desc = tl.make_tensor_descriptor(loss, [8192, 4096], [4096, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    dX_desc = tl.make_tensor_descriptor(dX, [8192, 4096], [4096, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    total_pids = 8192
    for virtual_pid in tl.range(tl.program_id(0), total_pids, _NUM_SM, loop_unroll_factor=2, num_stages=1):
        pid_0 = virtual_pid
        offset_0 = pid_0
        for offset_1 in tl.range(0, 4096, _BLOCK_SIZE_1, loop_unroll_factor=4, num_stages=4, disallow_acc_multi_buffer=True, flatten=False):
            X = _input_desc.load([offset_0, offset_1])
            Y = target_desc.load([offset_0, offset_1])
            X_max = tl.cast(tl.max(X, 0), tl.float32)
            Y_max = tl.cast(tl.max(Y, 0), tl.float32)
            eq = beta == 0
            if eq:
                Y_copy = Y
                Y_max_copy = Y_max
                X_copy = X
                Y_copy_0 = Y_copy
                Y_max_copy_0 = Y_max_copy
                X_copy_0 = X_copy
                v_0 = Y_max_copy_0[None, :]
                v_1 = Y_copy_0 - v_0
                v_2 = libdevice.exp(v_1)
                v_3 = libdevice.exp(Y_max_copy_0)
                v_4 = v_3[None, :]
                v_5 = v_2 * v_4
                v_6 = Y_copy_0 - X_copy_0
                v_7 = v_5 * v_6
                loss_desc.store([offset_0, offset_1], v_7)
                v_8 = -v_5
                dX_desc.store([offset_0, offset_1], v_8)
            _not = not eq
            if _not:
                X_copy_1 = X
                X_max_copy = X_max
                Y_copy_1 = Y
                Y_max_copy_1 = Y_max
                X_copy_1_0 = X_copy_1
                X_max_copy_0 = X_max_copy
                Y_copy_1_0 = Y_copy_1
                Y_max_copy_1_0 = Y_max_copy_1
                eq_1 = beta == 1.0
                if eq_1:
                    X_copy_1_0_copy = X_copy_1_0
                    X_max_copy_0_copy = X_max_copy_0
                    Y_copy_1_0_copy = Y_copy_1_0
                    X_copy_1_0_copy_0 = X_copy_1_0_copy
                    X_max_copy_0_copy_0 = X_max_copy_0_copy
                    Y_copy_1_0_copy_0 = Y_copy_1_0_copy
                    v_9 = X_max_copy_0_copy_0[None, :]
                    v_10 = X_copy_1_0_copy_0 - v_9
                    v_11 = libdevice.exp(v_10)
                    v_12 = libdevice.exp(X_max_copy_0_copy_0)
                    v_13 = v_12[None, :]
                    v_14 = v_11 * v_13
                    v_15 = X_copy_1_0_copy_0 - Y_copy_1_0_copy_0
                    v_16 = v_14 * v_15
                    loss_desc.store([offset_0, offset_1], v_16)
                    load = loss_desc.load([offset_0, offset_1])
                    v_17 = load + v_14
                    dX_desc.store([offset_0, offset_1], v_17)
                _not_1 = not eq_1
                if _not_1:
                    X_max_copy_0_copy_1 = X_max_copy_0
                    Y_max_copy_1_0_copy = Y_max_copy_1_0
                    X_copy_1_0_copy_1 = X_copy_1_0
                    Y_copy_1_0_copy_1 = Y_copy_1_0
                    X_max_copy_0_copy_1_0 = X_max_copy_0_copy_1
                    Y_max_copy_1_0_copy_0 = Y_max_copy_1_0_copy
                    X_copy_1_0_copy_1_0 = X_copy_1_0_copy_1
                    Y_copy_1_0_copy_1_0 = Y_copy_1_0_copy_1
                    v_18 = triton_helpers.maximum(X_max_copy_0_copy_1_0, Y_max_copy_1_0_copy_0)
                    v_19 = v_18[None, :]
                    v_20 = X_copy_1_0_copy_1_0 - v_19
                    v_21 = v_18[None, :]
                    v_22 = Y_copy_1_0_copy_1_0 - v_21
                    v_23 = libdevice.exp(v_18)
                    v_24 = libdevice.exp(v_20)
                    v_25 = v_23[None, :]
                    v_26 = v_24 * v_25
                    v_27 = libdevice.exp(v_22)
                    v_28 = v_23[None, :]
                    v_29 = v_27 * v_28
                    v_30 = v_29 * beta
                    sub_2 = 1.0 + -1 * beta
                    v_31 = v_26 * sub_2
                    v_32 = v_30 + v_31
                    v_33 = tl_math.log(v_32)
                    v_34 = v_30 * Y_copy_1_0_copy_1_0
                    v_35 = v_31 * X_copy_1_0_copy_1_0
                    v_36 = v_34 + v_35
                    v_37 = v_32 * v_33
                    v_38 = v_36 - v_37
                    loss_desc.store([offset_0, offset_1], v_38)
                    v_39 = X_copy_1_0_copy_1_0 - v_33
                    v_40 = v_31 * v_39
                    dX_desc.store([offset_0, offset_1], v_40)
            load_2 = loss_desc.load([offset_0, offset_1])
            v_41 = 0.0001220703125
            v_42 = load_2 * v_41
            loss_desc.store([offset_0, offset_1], v_42)
            load_3 = dX_desc.load([offset_0, offset_1])
            v_43 = 0.0001220703125
            v_44 = load_3 * v_43
            dX_desc.store([offset_0, offset_1], v_44)

def jsd_forward(_input: Tensor, target: Tensor, shift_labels: Tensor | None=None, beta: float=0.5, ignore_index: int=-100, *, _launcher=_default_launcher):
    """
    Compute Jensen-Shannon Divergence loss.

    Args:
        _input: Student predictions in log-space, shape (BT, V)
        target: Teacher targets in log-space, shape (BT, V)
        shift_labels: Optional labels for masking, shape (BT,)
        beta: Coefficient for generalized JSD in [0, 1]
        ignore_index: Index to ignore in labels

    Returns:
        loss: Scalar JSD loss
        dX: Gradient of loss wrt input
    """
    BT, V = _input.shape
    assert target.shape == _input.shape, f'Shape mismatch: {target.shape} != {_input.shape}'
    n_rows = BT
    loss = torch.zeros(_input.shape, dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(_input)
    n_non_ignore = float(BT)
    if shift_labels is not None:
        n_non_ignore = float((shift_labels != ignore_index).sum().item())
        if n_non_ignore == 0:
            return (torch.zeros([], dtype=_input.dtype, device=_input.device), torch.zeros_like(_input))
    BT_SIZE = helion.cdiv(BT, n_rows)
    _NUM_SM = helion.runtime.get_num_sm(_input.device)
    _BLOCK_SIZE_1 = 4
    torch.cuda.synchronize()
    _launcher(_helion_jsd_forward, (_NUM_SM,), _input, target, loss, dX, beta, _NUM_SM, _BLOCK_SIZE_1, 1, num_warps=16, num_stages=8)
    torch.cuda.synchronize()
    final_loss = torch.sum(loss)
    return (final_loss, dX)

def call():
    from torch._dynamo.testing import rand_strided
    _input = rand_strided(size=(8192, 4096), stride=(4096, 1), dtype=torch.float32, device='cuda:0')
    target = rand_strided(size=(8192, 4096), stride=(4096, 1), dtype=torch.float32, device='cuda:0')
    shift_labels = None
    beta = 1.1
    ignore_index = 8192
    jsd_forward(_input, target, shift_labels, beta, ignore_index)

if __name__ == '__main__':
    call()
