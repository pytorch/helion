"""CPU/offline compilation of the actual conditional collective arithmetic."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import pytest
import torch

from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_conditional_page import launch_conditional_page

pytestmark = [pytest.mark.cpu_only, pytest.mark.slow]


@pytest.mark.parametrize("columns,fast_math", [(32, False), (40, True)])
def test_conditional_collective_actual_helper_compiles(tmp_path, columns, fast_math):
    initialized = torch.cuda.is_initialized()
    tensors = [
        make_fake_tensor(dtype, shape, stride, assumed_align=16)
        for dtype, shape, stride in (
            (cutlass.Float32, (128, columns), (columns, 1)),
            (cutlass.Int32, (128, columns), (columns, 1)),
            (cutlass.Float32, (columns,), (1,)),
            (cutlass.Float32, (128, columns), (columns, 1)),
            (cutlass.Float32, (columns, 3), (3, 1)),
            (cutlass.Float32, (128, columns), (columns, 1)),
            (cutlass.BFloat16, (columns, 128), (128, 1)),
            (cutlass.Int32, (128,), (1,)),
        )
    ]
    with cpu_target(forbid_native=False):
        compiled = cute.compile(
            launch_conditional_page,
            *tensors,
            cuda.CUstream(0),
            cutlass.Boolean(True),
            cutlass.Float32(8.161115572774316),
            cutlass.Float32(2),
            columns,
            fast_math,
            no_jit_engine=True,
            no_cache=True,
        )
    (tmp_path / "compiled.mlir").write_text(str(compiled.ir_module))
    assert torch.cuda.is_initialized() == initialized
