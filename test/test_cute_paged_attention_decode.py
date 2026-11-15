"""How the paged-attention conversion helpers address their tensor-memory stores.

A tensor-memory address holds the lane in bits 16 and up and the column
below. The helpers add the two fields instead of combining them with OR: the
values are equal because no bit is set in both, and only the sum lets the
compiler share one ``(lane << 16) + base`` between a helper's stores and fold
each store's constant column into an immediate.
"""

from __future__ import annotations

from collections import Counter
import importlib
import re
from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import pytest

from helion._compiler.cute._paged_attention_runtime import _tmem_store_address
from helion._compiler.cute.cute_mma_dag import MMA_DAG_TMEM_COLUMNS
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_paged_attention_decode import launch

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only

# ``tcgen05.st.sync.aligned.<shape> [%rN], {...}``: the register that holds
# the address.
_STORE = re.compile(r"tcgen05\.st\.sync\.aligned\.\S+\s+\[(%r\d+)\]")


def test_lane_and_column_fields_share_no_bit():
    # Every lane of a 128-lane tile and every column of an SM's tensor memory:
    # the sum equals the OR the other attention stores use.
    assert MMA_DAG_TMEM_COLUMNS <= 1 << 16
    for lane in range(128):
        for column in range(MMA_DAG_TMEM_COLUMNS):
            assert _tmem_store_address(lane, column) == (lane << 16) | column


def _store_address_opcodes(
    tmp_path: Path,
    transpose: bool,
    columns: int,
    scale_group: int,
    post_scale_log2: int,
    positive_zero: bool,
) -> Counter[str]:
    """The opcodes that define the address registers of the ``tcgen05.st``
    instructions in the PTX of the standalone conversion kernel, compiled
    offline."""
    cuda = importlib.import_module("cuda.bindings.driver")
    groups = columns // scale_group
    tensors = [
        make_fake_tensor(dtype, shape, strides, assumed_align=16)
        for dtype, shape, strides in (
            (cutlass.Int8, (2, 128, columns), (128 * columns, columns, 1)),
            (cutlass.Uint8, (2, 128, groups), (128 * groups, groups, 1)),
            (cutlass.Int32, (2, 128, columns // 2), (64 * columns, columns // 2, 1)),
        )
    ]
    with cpu_target(forbid_native=False):
        cute.compile(
            launch,
            *tensors,
            cuda.CUstream(0),
            groups,
            False,
            transpose,
            columns,
            scale_group,
            post_scale_log2,
            positive_zero,
            options=f"--keep-ptx --dump-dir={tmp_path}",
            no_jit_engine=True,
            no_cache=True,
        )
    (path,) = tmp_path.glob("*.ptx")
    ptx = path.read_text()
    found: Counter[str] = Counter()
    for register in _STORE.findall(ptx):
        definitions = re.findall(
            rf"^\s*(\S+)\s+{re.escape(register)},", ptx, re.MULTILINE
        )
        assert len(definitions) == 1, (register, definitions)
        found[definitions[0]] += 1
    return found


@pytest.mark.parametrize(
    "transpose,columns,scale_group,post_scale_log2,positive_zero",
    [
        # K rows with the packed scale word.
        (False, 128, 32, 0, False),
        # K rows of a generic layout: one store per scale group.
        (False, 96, 32, 0, False),
        # V transpose: native and full-domain arm.
        (True, 128, 32, 0, False),
        # A postscale with a zero domain adds the sign-transport route and
        # the native-only arm.
        (True, 128, 32, -16, False),
        # The source's ``+ 0`` replaces the sign transport by constant stores
        # (``stage_positive_zero_value_tile``).
        (True, 128, 32, -16, True),
    ],
)
@forbid_cuda_work(forbid_native=False)
def test_conversion_stores_add_the_column_to_the_lane_base(
    tmp_path, transpose, columns, scale_group, post_scale_log2, positive_zero
):
    found = _store_address_opcodes(
        tmp_path, transpose, columns, scale_group, post_scale_log2, positive_zero
    )
    # The route stores to tensor memory, and no store address is an OR of the
    # two fields: every one is a sum.
    assert found and set(found) == {"add.s32"}, found
