from __future__ import annotations

import pytest

from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("read_overlap", [False, True])
@forbid_cuda_work(forbid_native=False)
def test_copied_metadata_and_tma_fallback_fixture_compiles(read_overlap):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor

    from helion._testing.cute_paged_value_metadata import HEADS
    from helion._testing.cute_paged_value_metadata import KEYS
    from helion._testing.cute_paged_value_metadata import PAGES
    from helion._testing.cute_paged_value_metadata import metadata_layout
    from helion._testing.cute_paged_value_metadata_tma import PAGE_STRIDE
    from helion._testing.cute_paged_value_metadata_tma import TMA_COMMANDS
    from helion._testing.cute_paged_value_metadata_tma import launch_metadata_tma_slots

    count = len(TMA_COMMANDS) * 2
    shape, strides, _words = metadata_layout(PAGE_STRIDE)
    raw = make_fake_tensor(
        cutlass.Int8,
        (PAGES, KEYS, HEADS, 128),
        (KEYS * HEADS * 128, HEADS * 128, 128, 1),
        assumed_align=16,
    )
    rest = [
        make_fake_tensor(cutlass.Int32, size, stride, assumed_align=16)
        for size, stride in (
            (shape, strides),
            ((count, 2), (2, 1)),
            ((count, 4096), (4096, 1)),
            ((count, KEYS), (KEYS, 1)),
            ((count,), (1,)),
        )
    ]
    with cpu_target(forbid_native=False):
        cute.compile(
            launch_metadata_tma_slots,
            raw,
            *rest,
            cuda.CUstream(0),
            read_overlap=read_overlap,
            no_jit_engine=True,
        )
