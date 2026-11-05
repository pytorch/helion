"""CPU checks of the grouped store-commit protocol of a shared-DAG fanout.

``tcgen05_fanout_store_commit='grouped'`` publishes one acquire before any ring
write, streams every output's original R2S into its own ring in output order,
then pays one fence, one barrier and ONE bulk commit group for the outputs' TMA
stores. ``serial`` stays the default and the original per-output protocol; its
own tests (``test_cute_shared_epilogue_dag``) are unchanged.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import re
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test.test_cute_shared_epilogue_dag import _external_four
from test.test_cute_shared_epilogue_dag import _external_three
from test.test_cute_shared_epilogue_dag import _external_two
from test.test_cute_shared_epilogue_dag import bound_original
from test.test_cute_shared_epilogue_dag import configuration

import helion
from helion._compiler.cute import shared_epilogue_dag
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.epilogue_fanout import FANOUT_STORE_COMMIT_CONFIG_KEY
from helion._compiler.cute.epilogue_fanout import FANOUT_STORE_COMMIT_MODES
from helion._compiler.cute.epilogue_fanout import FanoutStore
from helion._compiler.cute.shared_epilogue_dag import SharedDagFanoutPlan
from helion._compiler.cute.shared_epilogue_dag import grouped_store_commit
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

KEY = FANOUT_STORE_COMMIT_CONFIG_KEY
SHAPE = (128, 128)


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _external_args() -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(SHAPE, dtype=torch.bfloat16),
        torch.empty(SHAPE, dtype=torch.bfloat16),
        torch.empty(SHAPE, dtype=torch.float32),
        torch.empty(SHAPE, dtype=torch.float32),
        *(torch.empty(SHAPE, dtype=torch.float32) for _ in range(4)),
    )


def _bound_external(count: int):
    kernel = {2: _external_two, 3: _external_three, 4: _external_four}[count]
    return cpu_bind(kernel, _external_args())


def _config(mode: str | None, *, placement: str = "post_acc_wait") -> helion.Config:
    config = configuration()
    config.config["tcgen05_aux_load_placement"] = placement
    if mode is not None:
        config.config[KEY] = mode
    return config


def _subtile_loop(source: str) -> ast.For:
    loops = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_tcgen05_subtile"
    ]
    assert len(loops) == 1
    return loops[0]


def _loop_lines(source: str) -> list[str]:
    loop = _subtile_loop(source)
    return [line.strip() for line in ast.unparse(loop).splitlines()[1:]]


def _without_loop(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_tcgen05_subtile"
        ):
            node.body = [ast.Pass()]
    return ast.unparse(tree)


def _index(lines: list[str], pattern: str, start: int = 0) -> int:
    for position in range(start, len(lines)):
        if re.search(pattern, lines[position]):
            return position
    raise AssertionError(f"missing {pattern!r} after line {start}")


def check_grouped_protocol(lines: list[str], count: int) -> None:
    """The grouped subtile body, in order, for ``count`` outputs."""
    text = "\n".join(lines)
    assert text.count(".producer_acquire()") == 1
    assert text.count(".producer_commit()") == 1
    assert text.count("tcgen05_epilog_sync_barrier.arrive_and_wait()") == 2
    assert text.count(".arrive_and_wait()") == 2
    assert text.count("cute.arch.fence_view_async_shared()") == 1
    assert len(re.findall(r"cute\.copy\(tcgen05_tiled_copy_r2s", text)) == count
    assert len(re.findall(r"cute\.copy\(tcgen05_tma_store_atom", text)) == count
    assert len(re.findall(r"cute\.copy\(tcgen05_tiled_copy_t2r", text)) == 1
    acquire = _index(lines, r"tcgen05_c_pipeline\.producer_acquire\(\)")
    wait = _index(lines, r"acc_pipeline\.consumer_wait", acquire)
    tmem = _index(lines, r"cute\.copy\(tcgen05_tiled_copy_t2r", wait)
    publish = _index(lines, r"epilog_sync_barrier\.arrive_and_wait\(\)", tmem)
    position = publish
    suffixes = ["" if k == 0 else f"_{k}" for k in range(count)]
    for suffix in suffixes:
        buffer = _index(lines, rf"tcgen05_c_buffer{suffix} = ", position)
        position = _index(
            lines,
            rf"cute\.copy\(tcgen05_tiled_copy_r2s{suffix}, tcgen05_tRS_rD{suffix}, "
            rf"tcgen05_tRS_sD{suffix}\[None, None, None, tcgen05_c_buffer{suffix}\]\)",
            buffer,
        )
    release = _index(lines, r"acc_pipeline\.consumer_release", publish)
    last_r2s = position
    assert release < last_r2s
    fence = _index(lines, r"fence_view_async_shared\(\)", last_r2s)
    barrier = _index(lines, r"epilog_sync_barrier\.arrive_and_wait\(\)", fence)
    position = barrier
    for suffix in suffixes:
        position = _index(
            lines,
            rf"cute\.copy\(tcgen05_tma_store_atom{suffix}, tcgen05_bSG_sD{suffix}"
            rf"\[None, tcgen05_c_buffer{suffix}\]",
            position + 1,
        )
    commit = _index(lines, r"tcgen05_c_pipeline\.producer_commit\(\)", position)
    assert commit == len(lines) - 1 or "advance" in lines[commit + 1]


def _arithmetic(lines: list[str]) -> list[str]:
    return [
        line
        for line in lines
        if re.match(
            r"(tcgen05_dag_step_\d+|tcgen05_acc_vec(_\d+)?|tcgen05_aux_\w+|"
            r"tcgen05_tTR_gAux_subtile_\d+|tcgen05_acc_loaded_\d+|"
            r"tcgen05_c_buffer(_\d+)?) = ",
            line,
        )
        or "autovec_copy" in line
        or ".store(" in line
        or "consumer_release" in line
        or "fence_view_async_tmem_load" in line
    ]


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
@pytest.mark.parametrize("count", (2, 3, 4))
def test_grouped_order_counts_and_retained_rings(count, placement) -> None:
    bound = _bound_external(count)
    serial = bound.to_code(_config(None, placement=placement))
    grouped = bound.to_code(_config("grouped", placement=placement))
    ast.parse(grouped)
    check_grouped_protocol(_loop_lines(grouped), count)
    serial_text = "\n".join(_loop_lines(serial))
    assert serial_text.count(".producer_commit()") == count
    assert serial_text.count(".arrive_and_wait()") == 2 * count
    assert serial_text.count("fence_view_async_shared()") == count
    # Every ring keeps its pipeline, pre-loop acquire, drain and descriptor;
    # one TMEM read, one release and one accumulator advance remain.
    for token in (
        "PipelineTmaStore.create(",
        "producer_tail()",
        "'kind': 'tcgen05_d_tma'",
    ):
        assert grouped.count(token) == serial.count(token) == count, token
    for token in (
        ".consumer_release(tcgen05_acc_consumer_state)",
        "tcgen05_acc_consumer_state.advance()",
        "tcgen05_tmem_allocator.free(",
    ):
        assert grouped.count(token) == serial.count(token) == 1, token
    pre_loop = grouped.count(".producer_acquire()") - 1
    assert pre_loop == serial.count(".producer_acquire()") - count == count


@pytest.mark.parametrize("count", (2, 3, 4))
def test_grouped_keeps_arithmetic_cast_release_and_setup_text(count) -> None:
    bound = _bound_external(count)
    serial = bound.to_code(_config(None))
    grouped = bound.to_code(_config("grouped"))
    assert _arithmetic(_loop_lines(grouped)) == _arithmetic(_loop_lines(serial))
    # Outside the subtile loop the kernel is byte-identical.
    assert _without_loop(grouped) == _without_loop(serial)


def test_original_bool_modular_four_output_kernel_renders_grouped() -> None:
    bound = bound_original()
    plans = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert len(plans) == 1 and isinstance(plans[0], SharedDagFanoutPlan)
    serial = bound.to_code(configuration())
    config = configuration()
    config.config[KEY] = "grouped"
    grouped = bound.to_code(config)
    check_grouped_protocol(_loop_lines(grouped), 4)
    assert _arithmetic(_loop_lines(grouped)) == _arithmetic(_loop_lines(serial))
    assert _without_loop(grouped) == _without_loop(serial)
    assert grouped.count("cutlass.Uint8(0)") == serial.count("cutlass.Uint8(0)") == 1


@pytest.mark.parametrize(
    "mutation",
    ("drop_fence", "drop_publish", "drop_final_barrier", "tma_before_barrier"),
)
def test_structure_check_rejects_a_removed_barrier_or_fence(mutation) -> None:
    lines = _loop_lines(_bound_external(4).to_code(_config("grouped")))
    check_grouped_protocol(lines, 4)
    barriers = [
        i for i, line in enumerate(lines) if "epilog_sync_barrier.arrive" in line
    ]
    fence = next(i for i, line in enumerate(lines) if "fence_view_async_shared" in line)
    mutated = list(lines)
    if mutation == "drop_fence":
        del mutated[fence]
    elif mutation == "drop_publish":
        del mutated[barriers[0]]
    elif mutation == "drop_final_barrier":
        del mutated[barriers[1]]
    else:
        tma = next(
            i
            for i, line in enumerate(lines)
            if "cute.copy(tcgen05_tma_store_atom" in line
        )
        mutated.insert(barriers[1], mutated.pop(tma))
    with pytest.raises(AssertionError):
        check_grouped_protocol(mutated, 4)


# ---------------------------------------------------------------- proof facts


def _captured_stores(count: int = 4) -> list[FanoutStore]:
    bound = _bound_external(count)
    captured: list[list[FanoutStore]] = []
    original = shared_epilogue_dag.combine_dag_stores

    def capture(stores):
        captured.append([replace(store) for store in stores])
        return original(stores)

    # The state's registration imports the combiner at call time.
    with patch.object(shared_epilogue_dag, "combine_dag_stores", side_effect=capture):
        bound.to_code(_config("grouped"))
    assert len(captured) == 1
    return captured[0]


def test_recorded_protocol_facts_are_equal_and_admit_grouping() -> None:
    stores = _captured_stores()
    assert len(stores) == 4
    assert {store.commit_mode for store in stores} == {"grouped"}
    assert len({store.barrier_id for store in stores}) == 1
    assert {store.barrier_threads for store in stores} == {128}
    assert all(store.pipeline_stages == store.c_stages == 2 for store in stores)
    assert len({store.issuer for store in stores}) == 1
    assert grouped_store_commit(stores)
    serial = [replace(store, commit_mode="serial") for store in stores]
    assert not grouped_store_commit(serial)


@pytest.mark.parametrize(
    "fault",
    (
        "barrier_id",
        "barrier_threads",
        "pipeline_stages",
        "issuer",
        "c_stages",
        "missing_barrier_id",
        "missing_stages",
        "missing_issuer",
        "mixed_modes",
        "unknown_mode",
        "tma_without_issuer",
    ),
)
def test_protocol_fact_mismatch_fails_closed(fault) -> None:
    stores = _captured_stores()
    last = stores[-1]
    if fault == "barrier_id":
        assert last.barrier_id is not None
        stores[-1] = replace(last, barrier_id=last.barrier_id + 1)
    elif fault == "barrier_threads":
        stores[-1] = replace(last, barrier_threads=64)
    elif fault == "pipeline_stages":
        stores[-1] = replace(last, pipeline_stages=1)
    elif fault == "issuer":
        stores[-1] = replace(last, issuer="tcgen05_warp_idx == cutlass.Int32(1)")
    elif fault == "c_stages":
        stores[-1] = replace(last, c_stages=4, pipeline_stages=4)
    elif fault == "missing_barrier_id":
        stores[-1] = replace(last, barrier_id=None)
    elif fault == "missing_stages":
        stores[-1] = replace(last, pipeline_stages=None)
    elif fault == "missing_issuer":
        stores[-1] = replace(last, issuer="")
    elif fault == "mixed_modes":
        stores[-1] = replace(last, commit_mode="serial")
    elif fault == "unknown_mode":
        stores = [replace(store, commit_mode="bogus") for store in stores]
    else:
        stores[-1] = replace(last, tma=last.tma.split("\n", 1)[1])
    with pytest.raises(BackendUnsupported):
        grouped_store_commit(stores)


# ---------------------------------------------------------------- schema and coverage


def test_default_seeds_and_serial_stay_implicit() -> None:
    bound = _bound_external(4)
    spec = bound.config_spec
    assert KEY not in spec.default_config().config
    assert spec.compiler_default_config is None or (
        KEY not in spec.compiler_default_config.config
    )
    assert all(KEY not in seed.config for seed in spec.compiler_seed_configs)
    serial = spec.normalized_config(_config("serial"))
    assert KEY not in serial.config
    assert serial == spec.normalized_config(_config(None))
    grouped = spec.normalized_config(_config("grouped"))
    assert grouped.config[KEY] == "grouped"
    with bound.env:
        generation = spec.create_config_generation()
        flat, effective = generation.strict_config_pair(_config("grouped"))
        assert effective.config[KEY] == "grouped"
        assert generation.unflatten(flat) == effective
        assert generation.strict_config_pair(effective) == (flat, effective)


def test_field_is_prepared_before_coverage_and_reachable_by_canonical_coverage() -> (
    None
):
    bound = _bound_external(3)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.shared_dag_store_commit_search_enabled
    with bound.env:
        fields = spec.create_config_generation()._flat_fields()
    assert tuple(fields[KEY].choices) == FANOUT_STORE_COMMIT_MODES
    # Recorded in the coverage field fingerprint taken at the FIRST group.
    assert any(row[0] == KEY for row in spec._compiler_coverage_fields)
    groups = spec.compiler_coverage_groups
    keys = [group.key for group in groups]
    assert keys.index(FANOUT_CONFIG_KEY) < keys.index(KEY)
    (group,) = [g for g in groups if g.key == KEY]
    assert group.mechanism == "cute.shared_dag_store_commit"
    assert tuple(group.domain) == FANOUT_STORE_COMMIT_MODES
    assert group.legacy == "serial"
    assert [(d.key, d.value) for d in group.dependencies] == [
        (FANOUT_CONFIG_KEY, "shared")
    ]
    assert {w.value for w in group.witnesses} == {"serial", "grouped"}
    assert all(w.carrier.config[FANOUT_CONFIG_KEY] == "shared" for w in group.witnesses)
    with bound.env:
        _rows, outcomes = append_compiler_coverage(
            [], spec.create_config_generation(), cached_configs=lambda: (), pin=Mock()
        )
    added = [
        outcome
        for outcome in outcomes
        if outcome.mechanism == group.mechanism and outcome.outcome == "added"
    ]
    assert added
    assert all(
        outcome.effective is not None and outcome.effective.config[KEY] == "grouped"
        for outcome in added
    )


@pytest.mark.parametrize("value", (True, 1, "bogus", None))
def test_mistyped_mode_is_refused(value) -> None:
    bound = _bound_external(2)
    config = _config(None)
    config.config[KEY] = value
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(config)


@pytest.mark.parametrize(
    "field,value",
    (
        (FANOUT_CONFIG_KEY, "off"),
        ("tcgen05_epi_round_subtiles", 2),
        ("tcgen05_epi_warp_groups", 2),
        ("tcgen05_cluster_m", 2),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_aux_prefetch_subtiles", 1),
        ("tcgen05_aux_fragment_hoist", True),
        ("tcgen05_c_store_mode", "direct"),
        ("tcgen05_grouped_mode", "worklist_nm"),
        ("tcgen05_c_stages", 1),
        ("tcgen05_c_acquire_placement", "before_store"),
    ),
)
def test_explicit_grouped_on_an_unsupported_schedule_fails_loudly(field, value) -> None:
    bound = bound_original()
    config = configuration()
    config.config[KEY] = "grouped"
    config.config[field] = value
    # The mode's own gate refuses the schedule, whatever else refuses first.
    state = bound.config_spec._cute_tcgen05_config
    assert state.fanout_store_commit_refusal(config.config) is not None
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(config)


def test_supported_schedule_is_admitted_by_the_gate() -> None:
    bound = bound_original()
    state = bound.config_spec._cute_tcgen05_config
    config = configuration()
    config.config[KEY] = "grouped"
    assert state.fanout_store_commit_refusal(config.config) is None


def test_search_repair_drops_an_unproved_grouped_sample() -> None:
    bound = _bound_external(2)
    config = _config("grouped")
    config.config[FANOUT_CONFIG_KEY] = "off"
    raw = dict(config.config)
    bound.config_spec.normalize(raw, _fix_invalid=True)
    assert KEY not in raw
    assert raw.get(FANOUT_CONFIG_KEY, "off") == "off"


# ---------------------------------------------------------------- kernels without a shared-DAG plan


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _single_output(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        out[row, col] = acc


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _masked_fanout(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        value = acc + residual[row, col]
        out0[row, col] = value
        cols = col.index.reshape([1, col.index.size(0)])
        hl.store(out1, [row, col], value * 2.0, extra_mask=cols < columns - 1)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _mixed_dtype_fanout(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        value = acc + residual[row, col]
        out0[row, col] = value
        out1[row, col] = (value * 2.0).to(torch.bfloat16)
        out2[row, col] = value * 3.0


def _no_dag_cases():
    return {
        "single_output": lambda: cpu_bind(
            _single_output,
            (
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.float32),
            ),
        ),
        "masked_store": lambda: cpu_bind(
            _masked_fanout,
            (
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.float32),
                torch.empty(SHAPE, dtype=torch.float32),
                torch.empty(SHAPE, dtype=torch.float32),
            ),
        ),
        "mixed_dtype": lambda: cpu_bind(
            _mixed_dtype_fanout,
            (
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.float32),
                torch.empty(SHAPE, dtype=torch.float32),
                torch.empty(SHAPE, dtype=torch.bfloat16),
                torch.empty(SHAPE, dtype=torch.float32),
            ),
        ),
        "overlapping_outputs": lambda: bound_original(alias="outputs"),
        "overlapping_residual": lambda: bound_original(alias="residual"),
    }


@pytest.mark.parametrize(
    "case",
    (
        "single_output",
        "masked_store",
        "mixed_dtype",
        "overlapping_outputs",
        "overlapping_residual",
    ),
)
def test_kernels_without_a_shared_dag_plan_never_gain_the_mode(case) -> None:
    bound = _no_dag_cases()[case]()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert not any(
        isinstance(plan, SharedDagFanoutPlan) for plan in state.epilogue_fanout_plans
    )
    assert not state.shared_dag_store_commit_search_enabled
    assert all(group.key != KEY for group in spec.compiler_coverage_groups)
    if state.search_enabled:
        with bound.env:
            assert KEY not in spec.create_config_generation()._flat_fields()
    assert KEY not in spec.default_config().config
    config = (
        _config_from(spec.compiler_seed_configs[0])
        if spec.compiler_seed_configs
        else helion.Config(block_sizes=[128, 32, 64])
    )
    config.config[KEY] = "grouped"
    with pytest.raises(InvalidConfig, match=KEY):
        spec.normalized_config(config)


def _config_from(seed: helion.Config) -> helion.Config:
    return helion.Config.from_dict(dict(seed.config))


# ---------------------------------------------------------------- admitted schedule variants


SCHEDULES = {
    "blocked_scheduler": {},
    "blocked_monolithic": {
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_warp_spec_scheduler_warps": 0,
    },
    "interleaved_scheduler": {"pid_type": "persistent_interleaved"},
    "interleaved_monolithic": {
        "pid_type": "persistent_interleaved",
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_warp_spec_scheduler_warps": 0,
    },
}


@pytest.mark.parametrize("c_stages", (2, 4))
@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
@pytest.mark.parametrize("count", (2, 4))
def test_grouped_follows_every_admitted_schedule_and_its_budget(
    count, schedule, c_stages
) -> None:
    bound = _bound_external(count)
    spec = bound.config_spec
    serial = _config(None)
    grouped = _config("grouped")
    for config in (serial, grouped):
        config.config.update(SCHEDULES[schedule], tcgen05_c_stages=c_stages)
    # Grouped never widens or narrows what the serial shared fanout admits: a
    # ring budget the serial protocol cannot fit is refused for both.
    try:
        serial_n = spec.normalized_config(serial)
    except InvalidConfig:
        with pytest.raises(InvalidConfig):
            spec.normalized_config(grouped)
        return
    grouped_n = spec.normalized_config(grouped)
    assert grouped_n.config[KEY] == "grouped"
    grouped_code = bound.to_code(grouped_n)
    serial_code = bound.to_code(serial_n)
    check_grouped_protocol(_loop_lines(grouped_code), count)
    assert _arithmetic(_loop_lines(grouped_code)) == _arithmetic(
        _loop_lines(serial_code)
    )
    assert _without_loop(grouped_code) == _without_loop(serial_code)
    assert f"% cutlass.Int32({c_stages})" in grouped_code
