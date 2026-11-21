"""How the paged-attention conversion helpers address their tensor-memory
stores, and the forms the paired body takes on the packed scale tile.

A tensor-memory address holds the lane in bits 16 and up and the column
below. The helpers add the two fields instead of combining them with OR: the
values are equal because no bit is set in both, and only the sum lets the
compiler share one ``(lane << 16) + base`` between a helper's stores and fold
each store's constant column into an immediate.

On the packed four-scales-per-row tile (``packed_scale_tile_layout``) the
paired body's V conversion stages the whole tile natively and votes on its
scales behind it, and both conversions correct the scale exponents once per
lane or row. The standalone conversion kernel is compiled offline in both
forms and its PTX read for where the vote stands.
"""

from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path
import re
import tempfile

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor
import pytest

from helion._compiler.cute._paged_attention_runtime import _tmem_store_address
from helion._compiler.cute.cute_mma_dag import MMA_DAG_TMEM_COLUMNS
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_paged_attention_decode import launch

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


# The PTX of each configuration this module compiles, by its parameters:
# several tests read the same kernels.
_PTX: dict[tuple[bool, int, int, int, bool, bool], str] = {}


def _conversion_ptx(
    transpose: bool,
    columns: int,
    scale_group: int,
    post_scale_log2: int,
    positive_zero: bool,
    packed_tile_forms: bool = False,
) -> str:
    """The PTX of the standalone conversion kernel, compiled offline, once
    per configuration."""
    key = (
        transpose,
        columns,
        scale_group,
        post_scale_log2,
        positive_zero,
        packed_tile_forms,
    )
    if key not in _PTX:
        with tempfile.TemporaryDirectory() as dump_dir:
            _PTX[key] = _compile_conversion(Path(dump_dir), *key)
    return _PTX[key]


def _compile_conversion(
    dump_dir: Path,
    transpose: bool,
    columns: int,
    scale_group: int,
    post_scale_log2: int,
    positive_zero: bool,
    packed_tile_forms: bool,
) -> str:
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
            packed_tile_forms,
            options=f"--keep-ptx --dump-dir={dump_dir}",
            no_jit_engine=True,
            no_cache=True,
        )
    (path,) = dump_dir.glob("*.ptx")
    ptx = path.read_text()
    # The kernel's symbol spells its compile-time arguments.
    (entry,) = set(re.findall(r"\.entry\s+(\w+)\(", ptx))
    return ptx.replace(entry, "packet_kernel")


def _store_address_opcodes(
    transpose: bool,
    columns: int,
    scale_group: int,
    post_scale_log2: int,
    positive_zero: bool,
    packed_tile_forms: bool = False,
) -> Counter[str]:
    """The opcodes that define the address registers of the ``tcgen05.st``
    instructions in the PTX of the standalone conversion kernel."""
    ptx = _conversion_ptx(
        transpose,
        columns,
        scale_group,
        post_scale_log2,
        positive_zero,
        packed_tile_forms,
    )
    found: Counter[str] = Counter()
    for register in _STORE.findall(ptx):
        definitions = re.findall(
            rf"^\s*(\S+)\s+{re.escape(register)},", ptx, re.MULTILINE
        )
        assert len(definitions) == 1, (register, definitions)
        found[definitions[0]] += 1
    return found


@pytest.mark.parametrize(
    "transpose,columns,scale_group,post_scale_log2,positive_zero,packed_tile_forms",
    [
        # K rows with the packed scale word.
        (False, 128, 32, 0, False, False),
        # K rows of a generic layout: one store per scale group.
        (False, 96, 32, 0, False, False),
        # V transpose: native and full-domain arm.
        (True, 128, 32, 0, False, False),
        # A postscale with a zero domain adds the sign-transport route and
        # the native-only arm.
        (True, 128, 32, -16, False, False),
        # The source's ``+ 0`` replaces the sign transport by constant stores
        # (``stage_positive_zero_value_tile``).
        (True, 128, 32, -16, True, False),
        # The paired body's forms of the packed tile, K and V.
        (False, 128, 32, 0, False, True),
        (True, 128, 32, 0, False, True),
    ],
)
@forbid_cuda_work(forbid_native=False)
def test_conversion_stores_add_the_column_to_the_lane_base(
    transpose,
    columns,
    scale_group,
    post_scale_log2,
    positive_zero,
    packed_tile_forms,
):
    found = _store_address_opcodes(
        transpose,
        columns,
        scale_group,
        post_scale_log2,
        positive_zero,
        packed_tile_forms,
    )
    # The route stores to tensor memory, and no store address is an OR of the
    # two fields: every one is a sum.
    assert found and set(found) == {"add.s32"}, found


# One ``16x256b`` store stages a 16-key block of a V warp's two dimension
# halves; a tile of 128 keys is sixteen of them.
_V_STORE = "tcgen05.st.sync.aligned.16x256b"
_VOTE = re.compile(r"vote\.sync\.all\.pred\s+(%p\d+),")


def _lines_with(lines: list[str], text: str) -> list[int]:
    return [index for index, line in enumerate(lines) if text in line]


def _votes(lines: list[str]) -> list[tuple[int, str]]:
    """Line and result predicate of every warp vote."""
    return [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := _VOTE.search(line))
    ]


def _consumers(lines: list[str], predicate: str) -> list[int]:
    """Lines that read ``predicate``: branches on it and operands."""
    return [
        index
        for index, line in enumerate(lines)
        if re.search(rf"{re.escape(predicate)}\b", line) and not _VOTE.search(line)
    ]


@forbid_cuda_work(forbid_native=False)
def test_a_packed_v_tile_is_staged_ahead_of_its_one_scale_vote():
    """The V conversion of the packed tile, with the paired body's forms:
    sixteen stores of the native arm, then the tile's one vote, whose
    consumer is the branch into the full-domain arm's sixteen stores behind
    a store wait. The guard at the head of the tile, which every other
    caller keeps, votes ahead of its first store."""
    forms = _conversion_ptx(
        True, 128, 32, 0, False, packed_tile_forms=True
    ).splitlines()
    stores = _lines_with(forms, _V_STORE)
    assert len(stores) == 32
    ((vote, predicate),) = _votes(forms)
    # The vote stands behind the tile's sixteenth store and ahead of every
    # store of the arm that replaces the tile.
    assert stores[15] < vote < stores[16]
    consumers = _consumers(forms, predicate)
    assert consumers and all(vote < line < stores[16] for line in consumers)
    assert any(re.search(r"\bbra\b", forms[line]) for line in consumers)
    # The replacement is issued behind the completion of the first pass.
    waits = _lines_with(forms, "tcgen05.wait::st")
    assert [line for line in waits if vote < line < stores[16]]
    # One exponent correction per lane (0x06060606: six fractional bits in
    # each byte), and per 16-key block one byte permute for its upper pair.
    assert len([line for line in forms if "101058054" in line]) == 1
    assert len([line for line in forms if "0x3232U" in line]) == 8

    head = _conversion_ptx(True, 128, 32, 0, False).splitlines()
    head_stores = _lines_with(head, _V_STORE)
    ((head_vote, _predicate),) = _votes(head)
    assert len(head_stores) == 32 and head_vote < head_stores[0]
    assert not [line for line in head if "0x3232U" in line]


@forbid_cuda_work(forbid_native=False)
def test_packed_k_rows_correct_their_scales_once_behind_the_same_vote():
    """The K conversion of the packed tile keeps its unanimous vote ahead of
    both arms; the forms give the unanimous arm one exponent correction per
    row and one byte permute per scale group."""
    forms = _conversion_ptx(
        False, 128, 32, 0, False, packed_tile_forms=True
    ).splitlines()
    plain = _conversion_ptx(False, 128, 32, 0, False).splitlines()
    for lines in (forms, plain):
        ((vote, _predicate),) = _votes(lines)
        assert vote < min(_lines_with(lines, "tcgen05.st.sync.aligned"))
    assert len([line for line in forms if "101058054" in line]) == 1
    assert not [line for line in plain if "101058054" in line]
    # Groups one to three select their corrected byte with one permute each
    # (group zero's bytes are the word's low ones).
    permutes = re.compile(r"prmt\.b32\s+%r\d+, %r\d+, 0, 0x(11|22|33)U;")
    assert len([line for line in forms if permutes.search(line)]) == 3
    assert not [line for line in plain if permutes.search(line)]


@pytest.mark.parametrize(
    "transpose,columns,scale_group,post_scale_log2",
    [
        # A postscale below 2**-6: a corrected exponent can be negative.
        (True, 128, 32, -7),
        (False, 128, 32, -7),
        # Four scales per row that are not the tile's 32-dimension groups.
        (False, 64, 16, 0),
    ],
)
@forbid_cuda_work(forbid_native=False)
def test_another_layout_keeps_its_code_when_asked_for_the_forms(
    transpose, columns, scale_group, post_scale_log2
):
    """The forms are the packed tile's alone: asked for anywhere else, the
    conversions compile to the code they had."""
    asked = _conversion_ptx(
        transpose, columns, scale_group, post_scale_log2, False, packed_tile_forms=True
    )
    plain = _conversion_ptx(transpose, columns, scale_group, post_scale_log2, False)
    assert asked == plain
