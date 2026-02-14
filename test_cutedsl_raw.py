"""Test raw CuteDSL compilation on this machine."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


@cute.kernel
def simple_add(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor, N: cutlass.Constexpr[int], stream: int):
    pid = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    idx = pid * N + tid
    out[idx] = x[idx] + y[idx]


x = torch.randn(128, device='cuda')
y = torch.randn(128, device='cuda')
out = torch.empty_like(x)

cx = from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
cy = from_dlpack(y.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)
co = from_dlpack(out.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

stream = torch.cuda.current_stream(x.device).cuda_stream
compiled = cute.compile(simple_add, cx, cy, co, 128, stream)
compiled(x.detach(), y.detach(), out.detach(), 128, stream)
print('Expected:', (x + y)[:5])
print('Got:', out[:5])
print('Match:', torch.allclose(x + y, out))
