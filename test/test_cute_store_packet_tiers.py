"""``cute_store_packet_tiers`` on a kernel: a device loop that stores along
its tile into rows whose alignment changes from row to row stores narrower
packets where a row fails the full packet's guard, the knob is offered by
structure and normalized by the lanes the config forms and the packets the
stored tensor's layout leaves, and absent or off it leaves the code as it
was.  ``test_cute_affine_pointer_prefix.py`` executes the tier chain itself;
``test_cute_store_packet_tiers_cuda.py`` checks the stored values on a GPU.
"""

from __future__ import annotations

import ast
import random
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.store_packet_layout import ANY_STORE_PACKET_LANES
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.stored_row_kernels import centered_slices
from helion._testing.stored_row_kernels import doubled_pairs
from helion._testing.stored_row_kernels import doubled_rows
from helion._testing.stored_row_kernels import doubled_rows_from
from helion._testing.stored_row_kernels import doubled_rows_into
from helion._testing.stored_row_kernels import doubled_rows_into_two_dtypes
from helion._testing.stored_row_kernels import doubled_rows_past_three
from helion._testing.stored_row_kernels import random_rows_into
from helion._testing.stored_row_kernels import row_fill_from
from helion._testing.stored_row_kernels import row_view
from helion._testing.tail_peel_kernels import online_softmax_stats
from helion._testing.tail_peel_kernels import padded_rows
from helion.autotuner.config_spec import _CUTE_VECTOR_WIDTH_CHOICES

if TYPE_CHECKING:
    from helion.autotuner.compiler_coverage import CompilerCoverageGroup
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


def _config(tile: int, width: int, **knobs: object) -> dict[str, object]:
    """Four rows per CTA, one warp per row, ``width`` elements per lane."""
    return {
        "block_sizes": [4, tile],
        "num_threads": [0, 32],
        "cute_vector_widths": [1, width],
        **knobs,
    }


def _bound(columns: int, dtype: torch.dtype = torch.bfloat16):
    """``doubled_rows`` on 64 rows of ``columns`` elements, the input rows
    padded to 1024 (aligned) and the output's ``columns`` apart."""
    x = torch.zeros(64, 1024, dtype=dtype)[:, :columns]
    return cpu_bind(doubled_rows, (x,))


def _code(columns: int, config: dict[str, object]) -> str:
    with mock_cuda_unavailable(), cpu_render_target():
        bound = _bound(columns)
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**config))
        )


def _store_arms(code: str) -> list[tuple[str, str]]:
    """``(head line, body)`` of each arm of the lane loop's store dispatch,
    in order."""
    lines = code.splitlines()
    heads = [
        index
        for index, line in enumerate(lines)
        if line.startswith(
            ("            if ", "            elif ", "            else:")
        )
    ]
    return [
        (lines[head], "\n".join(lines[head + 1 : following]))
        for head, following in zip(heads, [*heads[1:], len(lines)], strict=True)
    ]


def _checked_lanes(head: str, tensor: str) -> str | None:
    """The lanes an arm checks the row offset of ``tensor`` against."""
    match = re.search(rf"{tensor}\.layout\.stride\[0\]\) % (\d+) == 0", head)
    return match.group(1) if match else None


def _store_branches(code: str) -> list[tuple[str | None, int, int]]:
    """``(lanes the row offset is checked against or None, packet stores,
    scalar stores)`` of each arm of the lane loop's store dispatch, in
    order."""
    return [
        (
            _checked_lanes(head, "out"),
            body.count("_cute_store_u16_vec("),
            body.count(".store(cutlass.BFloat16"),
        )
        for head, body in _store_arms(code)
    ]


def _tier_group(spec: ConfigSpec) -> CompilerCoverageGroup:
    """The coverage group that owns ``cute_store_packet_tiers``."""
    (group,) = (
        group
        for group in spec.compiler_coverage_groups
        if group.key == "cute_store_packet_tiers"
    )
    return group


def test_an_odd_row_length_stores_eight_four_and_two_lane_packets() -> None:
    """1021 bf16 columns: a row starts at any 2-byte phase, so the store is
    guarded at eight, four and two lanes (one, two and four packets), with
    the scalar loop last.  The first two arms are the two emitters' whole
    packets, as without the knob."""
    code = _code(1021, _config(256, 8, cute_store_packet_tiers=True))
    assert _store_branches(code) == [
        ("8", 1, 0),
        ("8", 1, 0),
        ("4", 2, 0),
        ("2", 4, 0),
        (None, 0, 1),
    ]
    # Every arm stores what one eight-lane load read.
    assert code.count("cute.arch.load(") == 4
    assert code.count("ir.VectorType.get([8]") == 4


def test_an_even_row_length_ends_at_the_two_lane_tier() -> None:
    """1018 columns: rows are four bytes apart at worst, so the two-lane
    tier checks no row offset and ends the chain (the scalar loop is left
    for a fragment with a masked lane)."""
    code = _code(1018, _config(256, 8, cute_store_packet_tiers=True))
    assert _store_branches(code) == [
        ("8", 1, 0),
        ("8", 1, 0),
        ("4", 2, 0),
        (None, 4, 0),
        (None, 0, 1),
    ]


def test_each_store_of_a_loop_narrows_only_as_far_as_its_own_rows_ask() -> None:
    """Sixteen lanes stored as bf16 into rows 1018 elements apart (aligned
    to two lanes, to more on some rows) and as fp32 into rows 1024 apart
    (aligned to every packet).  The bf16 store goes through its four tiers
    (16, 8, 4 and 2 lanes), the last one unguarded, so a row aligned to two
    bf16 lanes only still stores packets; the fp32 store, whose chain has
    three tiers, keeps its four 16-byte packets in every arm, because no
    guard of its own fails.  The scalar loop is left for a fragment with a
    masked lane."""
    with mock_cuda_unavailable(), cpu_render_target():
        x = torch.zeros(64, 1024, dtype=torch.float32)[:, :1018]
        half = torch.zeros(64, 1018, dtype=torch.bfloat16)
        full = torch.zeros(64, 1024, dtype=torch.float32)[:, :1018]
        bound = cpu_bind(doubled_rows_into_two_dtypes, (x, half, full))
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**_config(512, 16, cute_store_packet_tiers=True))
            )
        )
    # (bf16 lanes checked, bf16 packets, fp32 packets, scalar stores)
    assert [
        (
            _checked_lanes(head, "half"),
            body.count("_cute_store_u16_vec("),
            body.count("_cute_store_u32_vec("),
            body.count(".store(cutlass."),
        )
        for head, body in _store_arms(code)
    ] == [
        ("16", 2, 4, 0),
        ("8", 2, 4, 0),
        ("4", 4, 4, 0),
        (None, 8, 4, 0),
        (None, 0, 0, 2),
    ]


def test_two_stores_into_unaligned_rows_both_reach_their_narrowest_packet() -> None:
    """The same loop with the fp32 rows 1018 elements apart as well: both
    stores are guarded, and in the bf16 store's last tier (4-byte packets)
    the fp32 store, which has none left, stays at its 8-byte packets instead
    of ending the chain one tier early."""
    with mock_cuda_unavailable(), cpu_render_target():
        x = torch.zeros(64, 1024, dtype=torch.float32)[:, :1018]
        half = torch.zeros(64, 1018, dtype=torch.bfloat16)
        full = torch.zeros(64, 1018, dtype=torch.float32)
        bound = cpu_bind(doubled_rows_into_two_dtypes, (x, half, full))
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(**_config(512, 16, cute_store_packet_tiers=True))
            )
        )
    # (bf16 lanes checked, fp32 lanes checked, bf16 packets, fp32 packets,
    # scalar stores)
    assert [
        (
            _checked_lanes(head, "half"),
            _checked_lanes(head, "full"),
            body.count("_cute_store_u16_vec("),
            body.count("_cute_store_u32_vec("),
            body.count(".store(cutlass."),
        )
        for head, body in _store_arms(code)
    ] == [
        ("16", "16", 2, 4, 0),
        ("8", "4", 2, 4, 0),
        ("4", None, 4, 8, 0),
        (None, None, 8, 8, 0),
        (None, None, 0, 0, 2),
    ]


def test_without_the_knob_the_code_is_unchanged() -> None:
    """Absent and False are the same code: the whole packet under its guard,
    the scalar loop behind it."""
    absent = _code(1021, _config(256, 8))
    assert absent == _code(1021, _config(256, 8, cute_store_packet_tiers=False))
    assert _store_branches(absent) == [("8", 1, 0), ("8", 1, 0), (None, 0, 1)]


def test_aligned_rows_take_no_tier() -> None:
    """1024 columns: every store is proven aligned, no row offset is checked
    and the knob changes nothing."""
    on = _code(1024, _config(256, 8, cute_store_packet_tiers=True))
    assert on == _code(1024, _config(256, 8))
    assert "out.layout.stride[0]) %" not in on


def test_the_knob_follows_the_structure_and_the_lane_width() -> None:
    with mock_cuda_unavailable(), cpu_render_target():
        spec = _bound(1018).config_spec
        assert "cute_store_packet_tiers" in spec._flat_fields()
        default = spec.default_config().config
        assert default["cute_store_packet_tiers"] is False

        def normalized(**changes: object) -> dict[str, object]:
            return spec.normalized_config(
                helion.Config.from_dict(default | _config(256, 8) | changes)
            ).config

        assert normalized(cute_store_packet_tiers=True)["cute_store_packet_tiers"]
        # Lanes of two elements are the narrowest packet already, and scalar
        # lanes store none: the knob has nothing to split and reads off, so
        # the search does not time the same code twice.
        for width in (2, 1):
            off = normalized(
                cute_vector_widths=[1, width], cute_store_packet_tiers=True
            )
            assert off["cute_store_packet_tiers"] is False
            assert off == normalized(cute_vector_widths=[1, width])
        # A kernel that stores no row in a device loop: rejected, and an
        # explicit False (a config tuned where the knob was offered) dropped.
        plain = cpu_bind(online_softmax_stats, (padded_rows(64, 1018, "cpu"),))
        spec = plain.config_spec
        assert "cute_store_packet_tiers" not in spec._flat_fields()
        default = spec.default_config().config
        with pytest.raises(exc.InvalidConfig, match="cute_store_packet_tiers requires"):
            spec.normalized_config(
                helion.Config.from_dict(default | {"cute_store_packet_tiers": True})
            )
        off = spec.normalized_config(
            helion.Config.from_dict(default | {"cute_store_packet_tiers": False})
        )
        assert "cute_store_packet_tiers" not in off.config


def _tier_coverage(columns: int, dtype: torch.dtype = torch.bfloat16):
    bound = _bound(columns, dtype)
    return bound, _tier_group(bound.config_spec)


def test_witnesses_ride_on_seeds_whose_rows_the_knob_changes() -> None:
    """1018 columns: the seeds with lanes of four or more elements on the
    storing loop, several tiles and threads carry a witness, and the witness
    stores the tiers its carrier does not."""
    with mock_cuda_unavailable(), cpu_render_target():
        bound, group = _tier_coverage(1018)
        assert group.legacy is False and group.domain == (False, True)
        assert group.witnesses
        spec = bound.config_spec
        for witness in group.witnesses:
            assert witness.value is True
            carrier = witness.carrier.config
            assert "cute_store_packet_tiers" not in carrier or (
                carrier["cute_store_packet_tiers"] is False
            )
            assert carrier["cute_vector_widths"][1] >= 4
            assert carrier["num_threads"][1] > 0
            assert carrier["block_sizes"][1] < 1018
        carrier = group.witnesses[0].carrier
        off = bound.to_code(spec.normalized_config(carrier))
        on = bound.to_code(
            spec.normalized_config(
                helion.Config.from_dict(
                    carrier.config | {"cute_store_packet_tiers": True}
                )
            )
        )
    assert on.count("_cute_store_u16_vec(") > off.count("_cute_store_u16_vec(")


def test_a_whole_row_seed_carries_a_witness_where_no_seed_runs_several_tiles() -> None:
    """250 columns: every seed with lanes to split holds the row in one
    tile.  The seeds that run several tiles come first where there are any
    (1018 columns, above); here the whole-row seeds carry the witnesses, and
    a witness stores the tiers its carrier does not."""
    with mock_cuda_unavailable(), cpu_render_target():
        bound, group = _tier_coverage(250)
        assert group.witnesses
        spec = bound.config_spec
        for witness in group.witnesses:
            assert witness.carrier.config["block_sizes"][1] >= 250
        carrier = group.witnesses[0].carrier
        off = bound.to_code(spec.normalized_config(carrier))
        on = bound.to_code(
            spec.normalized_config(
                helion.Config.from_dict(
                    carrier.config | {"cute_store_packet_tiers": True}
                )
            )
        )
    assert on.count("_cute_store_u16_vec(") > off.count("_cute_store_u16_vec(")


def test_aligned_rows_carry_no_witness_and_keep_the_knob_out_of_the_population() -> (
    None
):
    """1024 columns: no seed's rows can fail a guard, so none carries a
    witness.  The group is registered all the same and owns the knob: the
    random initial population draws it off, as before the knob existed,
    where a random draw would be on in rows whose lanes are too narrow for
    it and normalization would rewrite them."""
    with mock_cuda_unavailable(), cpu_render_target():
        bound, group = _tier_coverage(1024)
        assert group.witnesses == ()
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            random.seed(0)
            rows = generation.initial_population_view().random_population_flat(40)
            for row in rows:
                flat, config = generation.canonicalize_flat(row)
                assert flat == row
                assert config.config["cute_store_packet_tiers"] is False


def test_the_search_surface_does_not_depend_on_the_row_length() -> None:
    """1018 columns leave the output rows unaligned and 1024 do not: the
    knob is offered at both, so configs transfer between the shapes."""
    with mock_cuda_unavailable(), cpu_render_target():
        fingerprints = [
            _bound(columns).config_spec.structural_fingerprint()
            for columns in (1018, 1021, 1024)
        ]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def _knob_off_and_forced(
    bound: BoundKernel, config: dict[str, object]
) -> tuple[bool, str, str]:
    """``(the knob after normalization, the code without the knob, the code
    with the knob kept on whatever normalization says)`` of ``config``."""
    spec = bound.config_spec
    default = spec.default_config().config

    def normalized(tiers: bool) -> helion.Config:
        return spec.normalized_config(
            helion.Config.from_dict(
                default | config | {"cute_store_packet_tiers": tiers}
            )
        )

    kept = normalized(True).config["cute_store_packet_tiers"]
    with patch.object(search_surface, "_stores_a_splittable_packet", lambda *_: True):
        forced = normalized(True)
        assert forced.config["cute_store_packet_tiers"] is True
    return kept, bound.to_code(normalized(False)), bound.to_code(forced)


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [(doubled_pairs, (64, 510)), (centered_slices, (8, 6, 96))],
    ids=["component", "full slice"],
)
def test_a_tile_off_the_stored_tensors_last_dimension_gets_no_knob(
    kernel: helion.Kernel, shape: tuple[int, ...]
) -> None:
    """A device loop whose tile is ahead of a component index
    (``out[tile_m, tile_n, 0]``) or beside a full slice
    (``out[tile_s, tile_r, :]``, the rows of a LayerNorm-backward nest)
    stores no packet along that tile: the knob is not a field of the search
    space, has no coverage group, and is rejected."""
    with mock_cuda_unavailable(), cpu_render_target():
        spec = cpu_bind(kernel, (torch.zeros(shape, dtype=torch.bfloat16),)).config_spec
        assert not spec.cute_store_tier_blocks
        assert "cute_store_packet_tiers" not in spec._flat_fields()
        assert all(
            group.key != "cute_store_packet_tiers"
            for group in spec.compiler_coverage_groups
        )
        default = spec.default_config().config
        assert "cute_store_packet_tiers" not in default
        with pytest.raises(exc.InvalidConfig, match="cute_store_packet_tiers requires"):
            spec.normalized_config(
                helion.Config.from_dict(default | {"cute_store_packet_tiers": True})
            )


def _output_view(view: str, dtype: torch.dtype, columns: int) -> torch.Tensor:
    """A ``[64, columns]`` output whose rows are 2042 elements apart, a
    multiple of no packet: the rows start at changing phases."""
    return row_view(torch.zeros(64, 2042, dtype=dtype), view, columns)


# What the stored tensor's layout leaves: (view, dtype, the most lanes of a
# packet, the vector widths that keep the knob).  Each row is a distinct
# refusal of the lowering: a base 12 bytes and 2 bytes off every packet, a
# base on an 8-byte boundary (four bf16 lanes fit, eight do not; no fp32
# packet has a tier to split into), a last dimension that is not
# unit-stride, and the aligned view as the control.
_OUTPUT_VIEWS = [
    ("3", torch.float32, 0, ()),
    ("1", torch.bfloat16, 0, ()),
    ("4", torch.bfloat16, 4, (4,)),
    ("2", torch.float32, 2, ()),
    ("strided", torch.bfloat16, 0, ()),
    ("0", torch.bfloat16, 16, (4, 8, 16)),
]


@pytest.mark.parametrize(
    ("view", "dtype", "limit", "kept_widths"),
    _OUTPUT_VIEWS,
    ids=[f"{view}-{dtype}".replace("torch.", "") for view, dtype, *_ in _OUTPUT_VIEWS],
)
def test_an_output_view_that_takes_no_packet_reads_the_knob_off(
    view: str, dtype: torch.dtype, limit: int, kept_widths: tuple[int, ...]
) -> None:
    """The knob stays on only at the vector widths whose packets the output
    view's base and last stride admit.  Where it is read off, the code with
    the knob forced on is the code without it (no choice is lost); where it
    is kept the tiers are there."""
    columns = 1018
    with mock_cuda_unavailable(), cpu_render_target():
        x = torch.zeros(64, 1024, dtype=dtype)[:, :columns]
        bound = cpu_bind(doubled_rows_into, (x, _output_view(view, dtype, columns)))
        spec = bound.config_spec
        (block,) = spec.cute_store_tier_blocks
        assert spec.cute_store_tier_packet_lanes == {block: limit}
        assert "cute_store_packet_tiers" in spec._flat_fields()
        # No seed carries a witness that normalization would turn back into
        # the seed; the group still owns the knob.
        group = _tier_group(spec)
        if limit in (0, 16):
            assert bool(group.witnesses) is (limit == 16)
        for width in (4, 8, 16):
            kept, off, forced = _knob_off_and_forced(bound, _config(32 * width, width))
            assert kept is (width in kept_widths), width
            if kept:
                assert forced != off
                assert forced.count("_vec(") > off.count("_vec(")
            else:
                assert forced == off, width


def test_the_any_width_lane_limit_is_the_widest_vector_width() -> None:
    """``ANY_STORE_PACKET_LANES`` stands for "packets at every vector
    width": the normalization compares a config's lanes with it, so it is
    the widest ``cute_vector_widths`` choice."""
    assert max(_CUTE_VECTOR_WIDTH_CHOICES) == ANY_STORE_PACKET_LANES


def test_lanes_the_config_does_not_form_read_the_knob_off() -> None:
    """The knob follows the lanes the tile runs with, not the requested
    ``cute_vector_widths`` entry: a width that does not divide each thread's
    elements, a thread per element and a tile without threads all run
    scalar lanes, with the same code whatever the knob says."""
    with mock_cuda_unavailable(), cpu_render_target():
        bound = _bound(1018)
        kept, off, forced = _knob_off_and_forced(bound, _config(256, 8))
        assert kept is True and forced != off
        for config in (
            # Eight elements per thread, sixteen lanes requested.
            _config(256, 16),
            # Four elements per thread, eight lanes requested.
            _config(128, 8),
            # One element per thread.
            _config(32, 8),
            # No threads on the storing tile: no lane loop.
            {**_config(256, 8), "num_threads": [0, 0]},
        ):
            kept, off, forced = _knob_off_and_forced(bound, config)
            assert kept is False, config
            assert forced == off, config


def test_rows_aligned_to_the_lanes_read_the_knob_off() -> None:
    """A row stride that is a multiple of the lanes proves every store
    aligned: no guard, no tier, and the knob reads off, the forced-on code
    being the knob-off code.  1016 columns are a multiple of eight lanes
    and not of sixteen; 1018 of none."""
    with mock_cuda_unavailable(), cpu_render_target():
        for columns, kept_widths in ((1024, ()), (1016, (16,)), (1018, (4, 8, 16))):
            bound = _bound(columns)
            spec = bound.config_spec
            (block,) = spec.cute_store_tier_blocks
            assert spec.cute_store_tier_row_multiples == {block: columns}
            for width in (4, 8, 16):
                kept, off, forced = _knob_off_and_forced(
                    bound, _config(32 * width, width)
                )
                assert kept is (width in kept_widths), (columns, width)
                assert (forced != off) is kept, (columns, width)


def test_a_loop_that_begins_off_its_lanes_keeps_the_knob_on_aligned_rows() -> None:
    """The rows' alignment says nothing where the tiles begin at a column
    the caller chooses: the lane-zero offset is guarded at run time, the
    tiers hang on that guard, and the knob stays on.  This pins the
    normalization and the generated arms only: this kernel loads along the
    columns it stores, every arm keeps the loads' whole packets, and a
    fragment that is off the store's boundary is off theirs, so its narrower
    arms do not run (``row_fill_from`` is the kernel whose arms do)."""
    with mock_cuda_unavailable(), cpu_render_target():
        x = torch.zeros(64, 1024, dtype=torch.bfloat16)
        bound = cpu_bind(doubled_rows_from, (x, 3))
        spec = bound.config_spec
        (block,) = spec.cute_store_tier_blocks
        assert spec.cute_store_tier_row_multiples == {block: 1}
        kept, off, forced = _knob_off_and_forced(bound, _config(256, 8))
        assert kept is True
        assert forced.count("_vec(") > off.count("_vec(")


def test_witnesses_follow_the_normalization_where_a_loop_begins_off_its_lanes() -> None:
    """A loop that begins at column 3 of rows 1024 elements apart: the rows
    are aligned to every packet, but the tiles are not, so the knob stays on
    (as above) and changes the code.  The witnesses read the same facts as
    the normalization: the seeds whose lanes keep the knob carry one, and it
    renders tiers its carrier does not.  (Like the kernel above this one
    loads along the columns it stores, so those arms do not run and its
    witnesses time their carriers' code: what normalization keeps on without
    need, the witnesses cover without need.)"""
    with mock_cuda_unavailable(), cpu_render_target():
        x = torch.zeros(64, 1024, dtype=torch.bfloat16)
        bound = cpu_bind(doubled_rows_past_three, (x,))
        spec = bound.config_spec
        (block,) = spec.cute_store_tier_blocks
        assert spec.cute_store_tier_row_multiples == {block: 1}
        group = _tier_group(spec)
        assert group.witnesses
        for witness in group.witnesses:
            carrier = witness.carrier.config
            assert carrier["cute_vector_widths"][1] >= 4
            assert carrier["block_sizes"][1] < 1021
        carrier = group.witnesses[0].carrier
        off = bound.to_code(spec.normalized_config(carrier))
        on = bound.to_code(
            spec.normalized_config(
                helion.Config.from_dict(
                    carrier.config | {"cute_store_packet_tiers": True}
                )
            )
        )
    assert on.count("_cute_store_u16_vec(") > off.count("_cute_store_u16_vec(")


def test_a_row_fill_from_a_callers_column_stores_each_packet_at_its_own_offset() -> (
    None
):
    """A loop that begins at a caller's column and stores a value it loads
    per row has no packet load along its tile, so nothing but the store's
    own guards decides which arm runs: the narrower arms are reachable, and
    their later packets take the offset of their own first lane instead of
    a pointer step from the first packet (the whole loop's alignment is not
    proven, so the offset could wrap between them)."""
    with mock_cuda_unavailable(), cpu_render_target():
        scale = torch.zeros(64, dtype=torch.bfloat16)
        out = torch.zeros(64, 1018, dtype=torch.bfloat16)
        bound = cpu_bind(row_fill_from, (scale, out, 2))
        kept, off, forced = _knob_off_and_forced(bound, _config(256, 8))
    assert kept is True
    assert "cute.arch.load(" not in forced
    assert forced.count("_cute_store_u16_vec(") > off.count("_cute_store_u16_vec(")
    assert "cutlass.Int32(lane_base_1 + cutlass.Int32(2))" in forced
    assert "cutlass.Int32(lane_base_1 + cutlass.Int32(2))" not in off


def _guard_holds(test: ast.expr, row: int) -> bool:
    """A generated guard at the first tile of ``row`` of an output whose
    rows are 1018 elements apart."""
    names = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "cute": SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0))),
        "out": SimpleNamespace(layout=SimpleNamespace(stride=(1018, 1))),
        "indices_0": row,
        "lane_base_1": 0,
    }
    return bool(eval(compile(ast.Expression(test), "<guard>", "eval"), names))


def test_a_philox_packet_stores_in_tiers_too() -> None:
    """The stores of a Philox packet (``cute_rng_packet``) are packed by the
    Philox lowering's own call of the affine vectorizer, which takes the knob
    as the lane loops' pass does: four fp32 lanes store one 16-byte packet
    where the row is aligned to it, two 8-byte packets where it is not (each
    at the offset of its own first lane), and scalars last.  Without the
    knob the packet has the one guard and the scalar loop.

    The packet's own guard (its first value is a multiple of four into the
    stream) must not imply the store's: the output is 1016 columns of rows
    1018 elements apart, so the packet is drawn on every row and the odd
    rows, half a packet off the 16-byte boundary, take the 8-byte arm."""
    codes = {}
    with mock_cuda_unavailable(), cpu_render_target():
        bound = cpu_bind(random_rows_into, (torch.zeros(64, 1018)[:, :1016], 123))
        for tiers in (True, False):
            config = _config(
                128, 4, cute_rng_packet=True, cute_store_packet_tiers=tiers
            )
            normalized = bound.config_spec.normalized_config(helion.Config(**config))
            assert normalized.config.get("cute_store_packet_tiers", False) is tiers
            codes[tiers] = bound.to_code(normalized)
    for code in codes.values():
        assert "philox_packet_" in code and "cute.arch.load(" not in code
    assert codes[False].count("_cute_store_u32_vec(") == 1
    assert "% 2 == 0" not in codes[False]
    packets = re.findall(r"_cute_store_u32_vec\((.*), (\S+)\)\n", codes[True])
    assert [values for _address, values in packets] == [
        "_helion_affine_store",
        "_helion_affine_store_[0:2]",
        "_helion_affine_store_[2:4]",
    ]
    # The second packet of the narrower tier is addressed from its own lane.
    assert "lane_base_1 + cutlass.Int32(2)" in packets[2][0]
    # Which arm a row runs: the Philox guard holds on every row, the whole
    # packet's on the even ones and the two half packets' on the odd ones.
    (drawn,) = (
        node
        for node in ast.walk(ast.parse(codes[True]))
        if isinstance(node, ast.If) and "& 3 == 0" in ast.unparse(node.test)
    )
    (whole,) = (node for node in drawn.body if isinstance(node, ast.If))
    (halves,) = whole.orelse
    assert isinstance(halves, ast.If)
    assert "_helion_affine_store_[2:4]" in ast.unparse(halves.body)
    for row in range(4):
        assert _guard_holds(drawn.test, row)
        assert _guard_holds(whole.test, row) is (row % 2 == 0)
        assert _guard_holds(halves.test, row)


def test_the_search_surface_does_not_depend_on_the_output_view() -> None:
    """An output view that takes no packet keeps the knob's field (it is
    normalized, not withdrawn): configs transfer between an aligned and an
    unaligned call of one kernel."""
    with mock_cuda_unavailable(), cpu_render_target():
        fingerprints = []
        for view in ("0", "3", "strided"):
            x = torch.zeros(64, 1024, dtype=torch.float32)[:, :1018]
            out = _output_view(view, torch.float32, 1018)
            spec = cpu_bind(doubled_rows_into, (x, out)).config_spec
            fingerprints.append(spec.structural_fingerprint())
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
