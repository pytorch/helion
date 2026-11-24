"""``cute_accumulate_stores``: a store that reads no reduction result runs in
the lane split's accumulate pass, which computes its value once, instead of
in the consume pass, which computed it again.  Codegen checks on CPU."""

from __future__ import annotations

import ast
import textwrap

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.lane_loops.markers import _is_lane_reduce_marker_assign
from helion._compiler.cute.lane_loops.split import _accumulate_pass_stores
from helion._testing.accumulate_store_kernels import SEED_CONFIG
from helion._testing.accumulate_store_kernels import accumulate_into_stored
from helion._testing.accumulate_store_kernels import column_sums
from helion._testing.accumulate_store_kernels import store_and_column_sums
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable

pytestmark = pytest.mark.cpu_only


def _kernel_source(bound: object, config: dict[str, object]) -> str:
    code = bound.to_code(helion.Config(**config))  # type: ignore[attr-defined]
    return code[code.index("@cute.kernel") :]


@pytest.mark.parametrize(
    "shape",
    [(8, 200, 64), (3, 128, 40)],
    ids=["rows_off_the_tile", "columns_off_the_tile"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_value_is_computed_once(
    shape: tuple[int, int, int], dtype: torch.dtype
) -> None:
    # Off, the consume pass computes each stored value again from the loaded
    # elements: two exps per element, one in each pass,
    #     for lane_2 in ...:                 # accumulate
    #         v_3 = v_1 * cute.math.exp(v_1); sum_1_lane_acc += v_3
    #     for lane_2 in ...:                 # consume
    #         v_3 = v_1 * cute.math.exp(v_1); store(v_3)
    # and on, one: the store follows the add of its lane.
    with mock_cuda_unavailable():
        bound = cpu_bind(store_and_column_sums, (torch.empty(shape, dtype=dtype),))
        off, on = (
            _kernel_source(bound, {**SEED_CONFIG, "cute_accumulate_stores": knob})
            for knob in (False, True)
        )
    assert off.count("cute.math.exp") == 2 * on.count("cute.math.exp") > 0, on


def test_a_sink_that_does_not_apply_takes_the_knob_with_it() -> None:
    # 36 bf16 columns: the seed's 8-element packets leave a ragged tail, the
    # sink does not apply at codegen (VloopSinkNotApplied), and the kernel is
    # generated again without it.  The knob goes too: on the scalar lanes
    # its stores would sit between one lane's loads and the next one's.
    with mock_cuda_unavailable():
        bound = cpu_bind(
            store_and_column_sums, (torch.empty(8, 200, 36, dtype=torch.bfloat16),)
        )
        off, on = (
            _kernel_source(bound, {**SEED_CONFIG, "cute_accumulate_stores": knob})
            for knob in (False, True)
        )
    assert "_vsink" not in on
    assert on == off


def test_the_knob_is_offered_and_covered() -> None:
    # A loop that stores a value and sums it offers the knob; the seeds stay
    # knob-off, and the column reduction's seeds, whose vector packets the
    # sunk lanes load ahead of the stores, carry knob-on witnesses.
    with mock_cuda_unavailable():
        bound = cpu_bind(
            store_and_column_sums, (torch.empty(8, 200, 64, dtype=torch.bfloat16),)
        )
    spec = bound.config_spec
    assert spec.cute_accumulate_store_sites
    seeds = [dict(seed.config) for seed in spec.compiler_seed_configs]
    assert seeds
    assert not any(seed.get("cute_accumulate_stores") for seed in seeds), seeds
    (group,) = [
        g for g in spec.compiler_coverage_groups if g.key == "cute_accumulate_stores"
    ]
    assert group.witnesses
    assert all(
        witness.carrier.config.get("cute_vloop_sink") is True
        for witness in group.witnesses
    )


def test_a_stored_tensor_the_loop_reads_keeps_its_place() -> None:
    # ``y[...] = y[...] + value``: the loop's load of y orders the store, so it
    # stays in the consume pass and the knob changes nothing.
    x = torch.empty(8, 200, 64, dtype=torch.bfloat16)
    with mock_cuda_unavailable():
        bound = cpu_bind(accumulate_into_stored, (x, torch.empty_like(x)))
        assert bound.config_spec.cute_accumulate_store_sites
        off, on = (
            _kernel_source(bound, {**SEED_CONFIG, "cute_accumulate_stores": knob})
            for knob in (False, True)
        )
    assert off == on


def test_a_loop_that_stores_nothing_rejects_the_knob() -> None:
    with mock_cuda_unavailable():
        bound = cpu_bind(column_sums, (torch.empty(8, 200, 64, dtype=torch.bfloat16),))
    spec = bound.config_spec
    assert not spec.cute_accumulate_store_sites
    with pytest.raises(exc.InvalidConfig, match="cute_accumulate_stores requires"):
        spec.normalize(helion.Config(**{**SEED_CONFIG, "cute_accumulate_stores": True}))
    # An explicit False, the knob-off code, is dropped.
    config = helion.Config(**{**SEED_CONFIG, "cute_accumulate_stores": False})
    spec.normalize(config)
    assert "cute_accumulate_stores" not in config.config


def test_the_knob_needs_the_sunk_vector_loop() -> None:
    # Knob-on scalar lanes hold each lane's loads behind the previous lane's
    # store: the search's knob-on rows without the vector-loop sink normalize
    # to the knob-off code, and an explicit request for them is an error.
    with mock_cuda_unavailable():
        bound = cpu_bind(
            store_and_column_sums, (torch.empty(8, 200, 64, dtype=torch.bfloat16),)
        )
    spec = bound.config_spec
    scalar = {**SEED_CONFIG, "cute_vloop_sink": False, "cute_lane_unroll": 1}
    narrow = {**SEED_CONFIG, "cute_vector_widths": [1, 1, 1]}
    for base in (scalar, narrow):
        with pytest.raises(exc.InvalidConfig, match="requires cute_vloop_sink"):
            spec.normalize(helion.Config(**{**base, "cute_accumulate_stores": True}))
        config = helion.Config(**{**base, "cute_accumulate_stores": True})
        spec.normalize(config, _fix_invalid=True)
        assert config.config["cute_accumulate_stores"] is False
    config = helion.Config(**{**SEED_CONFIG, "cute_accumulate_stores": True})
    spec.normalize(config)
    assert config.config["cute_accumulate_stores"] is True


@pytest.mark.parametrize(
    ("produce", "moves"),
    [("w = v * 2.0", True), ("v = v * 2.0", False)],
    ids=["new_name", "rebinds_the_reduced_value"],
)
def test_a_producer_that_rebinds_a_reduced_value_keeps_its_store(
    produce: str, moves: bool
) -> None:
    # The accumulations follow the accumulate pass's other statements: a
    # producer behind the marker that binds its input again would hand the
    # sum its own binding (20 instead of 10 on [1, 2, 3, 4]), so its store
    # stays in the consume pass.
    stored = produce.split(" = ")[0]
    body: list[ast.AST] = list(
        ast.parse(
            textwrap.dedent(
                f"""
                load = (in0.iterator + lane_2).load()
                v = cutlass.Float32(load)
                s = _helion_lane_reduce(v, 'sum', 0.0, threads_in_group=32)
                {produce}
                (out0.iterator + lane_2).store(cutlass.BFloat16({stored}))
                """
            )
        ).body
    )
    marker = _is_lane_reduce_marker_assign(body[2])
    assert marker is not None
    early = _accumulate_pass_stores(
        body,
        "lane_2",
        [(2, marker)],
        [4],
        {0, 1},
        lambda stmt: True,
        {frozenset({"in0", "out0"})},
        {},
        4,
        {},
    )
    assert early == ({4: [3]} if moves else {})
