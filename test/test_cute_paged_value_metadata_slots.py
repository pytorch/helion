from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import pytest

from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_paged_value_metadata import COMMANDS
from helion._testing.cute_paged_value_metadata import HIGH_PAGE_STRIDE
from helion._testing.cute_paged_value_metadata import launch_metadata_slots
from helion._testing.cute_paged_value_metadata import metadata_layout

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("read_overlap", [False, True])
@forbid_cuda_work(forbid_native=False)
def test_metadata_slot_helper_compiles_with_a_wide_word_offset(read_overlap):
    shape, strides, _words = metadata_layout(HIGH_PAGE_STRIDE)
    assert strides[3] < 2**31
    assert (shape[3] - 1) * strides[3] > 2**31
    count = len(COMMANDS) * 2
    tensors = [
        make_fake_tensor(cutlass.Int32, size, stride, assumed_align=16)
        for size, stride in (
            (shape, strides),
            ((count, 2), (2, 1)),
            ((count, 128), (128, 1)),
            ((count,), (1,)),
        )
    ]
    with cpu_target(forbid_native=False):
        cute.compile(
            launch_metadata_slots,
            *tensors,
            cuda.CUstream(0),
            read_overlap=read_overlap,
            no_jit_engine=True,
        )
