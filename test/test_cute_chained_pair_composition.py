from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.chained_completed_register import pair_result_cut
from helion._compiler.cute.chained_completed_register import retire_result_cut
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_parallel_merges import JOIN
from helion._compiler.cute.chained_parallel_merges import _available_barriers
from helion._compiler.cute.chained_parallel_merges import _PairGroup
from helion._compiler.cute.chained_parallel_merges import _parallelize_pairs
from helion._compiler.cute.chained_parallel_merges import _Stage
from helion._compiler.cute.chained_parallel_merges import ready_waves
from helion._compiler.cute.chained_warp_pair import WarpPairLayout
from helion._compiler.cute.chained_warp_pair import WarpPairTeam
from helion._compiler.cute.chained_warp_stage import rebind_warp_execution

if TYPE_CHECKING:
    from helion._compiler.cute.chained_root_warp_stage import WarpOperandEmission
    from helion._compiler.cute.chained_warp_pair import WarpPairAttempt


def test_weighted_frontiers_preserve_dependencies_and_workspace_capacity() -> None:
    nodes = (0, 2, 4, 6, 8)
    deps = {
        0: frozenset(),
        2: frozenset(),
        4: frozenset({0, 2}),
        6: frozenset({0}),
        8: frozenset({4, 6}),
    }
    weights = {0: 1, 2: 2, 4: 2, 6: 1, 8: 3}
    assert ready_waves(nodes, deps, 3, weights=weights) == ((0, 2), (4, 6), (8,))
    assert ready_waves(
        nodes,
        deps,
        3,
        weights=weights,
        workspace_sizes=dict.fromkeys(nodes, (65, 0)),
        workspace_capacities=(192, 0),
    ) == ((0,), (2,), (4,), (6,), (8,))
    assert ready_waves(nodes, deps, 2, weights=weights) is None
    assert ready_waves((0, 2), {0: frozenset({2}), 2: frozenset({0})}, 4) is None


def _team(threads=64, first_thread=32, barrier_id=3):
    shape = (threads // 2, 16, 16)
    layout = WarpPairLayout(shape, shape, torch.float16, torch.float16, threads, 8, 1)
    execution = ChainedExecution(
        threads,
        thread="local_thread",
        warp="local_warp",
        barriers="local_join",
        sync="local_join.arrive_and_wait()",
    )
    return WarpPairTeam(layout, execution, first_thread, 128, barrier_id)


def test_pair_team_and_rebinding_preserve_local_coordinates() -> None:
    team = _team()
    assert team.layout.topology == (2, 1, 1)
    assert team.bindings == (
        "local_thread = chain_thread - 32",
        "local_warp = local_thread // 32",
    )
    body = ("lane = chain_thread % 32", "owner = chain_warp", "fragment[lane] = value")
    changed = rebind_warp_execution(body, ChainedExecution(64), team.execution)
    assert changed == (
        "lane = local_thread % 32",
        "owner = local_warp",
        "fragment[lane] = value",
    )
    with pytest.raises(_UnsupportedChain, match="CTA collective"):
        rebind_warp_execution((*body, JOIN), ChainedExecution(64), team.execution)
    for invalid in (
        lambda: replace(team, first_thread=1),
        lambda: replace(team, first_thread=96),
        lambda: replace(team, barrier_id=0),
        lambda: replace(team, execution=ChainedExecution(64)),
        lambda: replace(team, execution=replace(team.execution, warp="local_thread")),
    ):
        with pytest.raises(_UnsupportedChain, match="local team"):
            invalid()
    assert _available_barriers(
        ("b = pipeline.NamedBarrier(barrier_id=3, num_threads=32)",)
    ) == tuple(i for i in range(1, 16) if i != 3)
    assert (
        _available_barriers(
            ("b = pipeline.NamedBarrier(barrier_id=dynamic, num_threads=32)",)
        )
        == ()
    )


def test_rebinding_deep_expression_preserves_exact_arithmetic_text() -> None:
    expression = " + ".join(["coefficient"] * 600)
    body = (
        "for index in range(4):",
        f"    value = ({expression}) if chain_thread < 64 else chain_warp",
    )
    with patch("ast.unparse", side_effect=AssertionError("recursive unparse")):
        changed = rebind_warp_execution(body, ChainedExecution(64), _team().execution)
    assert changed == (
        body[0],
        f"    value = ({expression}) if local_thread < 64 else local_warp",
    )


def test_rebinding_uses_utf8_name_spans_and_preserves_non_names() -> None:
    body = (
        'label = "chain_thread"; π = chain_thread  # chain_warp',
        "chain_thread = object.chain_thread + chain_warp",
        "call(chain_thread=chain_thread)",
        'text = f"chain_thread={chain_thread}, chain_warp={chain_warp}"',
    )
    assert rebind_warp_execution(body, ChainedExecution(64), _team().execution) == (
        'label = "chain_thread"; π = local_thread  # chain_warp',
        "local_thread = object.chain_thread + local_warp",
        "call(chain_thread=local_thread)",
        'text = f"chain_thread={local_thread}, chain_warp={local_warp}"',
    )
    for collective in ("cute.arch.sync_threads()", "(cute.arch).barrier(barrier_id=2)"):
        with pytest.raises(_UnsupportedChain, match="CTA collective"):
            rebind_warp_execution(
                (*body, collective), ChainedExecution(64), _team().execution
            )


def _group(
    index,
    threads,
    plan,
    *,
    dependencies=frozenset(),
    reads=frozenset(),
    writes=frozenset(),
):
    """Recorded tiny stage bodies exercise scheduling without Helion/SDK emission."""
    shape = (threads // 2, 16, 16)
    pair = SimpleNamespace(
        source=index,
        layout=WarpPairLayout(
            shape, shape, torch.float16, torch.float16, threads, 8, 1
        ),
        plan=plan,
        phase="consumed",
        team=None,
        execution=ChainedExecution(threads),
        check=lambda: None,
        selection=lambda: (index, threads),
    )
    members = []
    for stage in (index, index + 1):
        prefix = f"chain_{stage}"
        prep = (
            f"{prefix}_a_ptr = chain_a_workspace",
            f"{prefix}_b_ptr = chain_b_workspace",
        )
        declaration = (
            f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, 320, alignment=128)",
            f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, cute.make_layout((16, 16), stride=(20, 1)))",
        )
        publication = (
            f"{prefix}_sc = {prefix}_thr.partition_C({prefix}_c)",
            f"cute.autovec_copy({prefix}_acc, {prefix}_sc)",
        )
        if stage == index:
            register = (
                f"chain_pair_{index}_a = cute.make_rmem_tensor(8, cutlass.Float16)",
            )
            compute = (
                f"{prefix}_thr = mma.get_slice(chain_thread)",
                f"{prefix}_acc = {prefix}_b_ptr[chain_thread]",
                f"chain_pair_{index}_a[0] = cutlass.Float16(-{prefix}_acc)",
            )
            pair.producer = SimpleNamespace(_publication=publication)
        else:
            register = ()
            compute = (
                f"{prefix}_thr = mma.get_slice(chain_thread)",
                f"{prefix}_acc = chain_pair_{index}_a[0] * {prefix}_b_ptr[chain_thread]",
            )
        source = (
            *prep,
            *register,
            JOIN,
            *declaration,
            f"if chain_thread < {threads}:",
            *("    " + line for line in (*compute, *publication)),
            JOIN,
        )
        members.append(
            _Stage(
                stage,
                plan.dots[stage],
                source,
                prep,
                declaration,
                publication,
                cast(
                    "WarpOperandEmission",
                    SimpleNamespace(axes=(("a", 1), ("b", 1)), layouts=()),
                ),
                "cutlass.Float16",
                shape,
                threads,
                dependencies if stage == index else frozenset({index}),
                reads if stage == index else frozenset(),
                writes if stage == index + 1 else frozenset(),
                (64, 64) if stage == index else (0, 64),
                compute,
                register,
                stage == index + 1,
            )
        )
    return _PairGroup(
        cast("WarpPairAttempt", pair),
        (members[0], members[1]),
        ((0, 0), (64, 64)),
        (64, 128),
    )


def _fixture(*, hazard=False, late_reader=False, serial=False):
    plan = SimpleNamespace(
        threads=128,
        dots=tuple(object() for _ in range(4)),
        root_native=SimpleNamespace(original_workspace_elements=(256, 512)),
        operand_dtype=lambda stage: torch.float16,
    )
    first = _group(0, 32, plan, reads=frozenset({(0, 4)}), writes=frozenset({(0, 4)}))
    second = _group(
        2,
        64,
        plan,
        dependencies=frozenset({1}) if serial else frozenset(),
        reads=frozenset({(0, 4)}) if hazard else frozenset(),
    )
    lines = ("before = 1", *first.source, *second.source, "after = 2")
    if late_reader:
        lines += ("export[0] = chain_0_c[0]",)
    emission = SimpleNamespace(
        plan=plan, lowering=SimpleNamespace(result=lines), images=()
    )
    return emission, (first, second)


def _lower(emission, groups):
    with (
        patch(
            "helion._compiler.cute.chained_parallel_merges._pair_group",
            side_effect=groups,
        ),
        patch(
            "helion._compiler.cute.chained_direct_image._native_c_coordinates",
            return_value=True,
        ),
    ):
        return _parallelize_pairs(emission, tuple(group.pair for group in groups))


def test_complete_pair_deep_body_keeps_arithmetic_and_retirement_guards() -> None:
    emission, groups = _fixture()
    expression = " + ".join(["coefficient"] * 1200)
    body = (f"if ({expression}) > chain_thread:", "    value = chain_warp")
    first, second = groups[0].members
    first = replace(
        first,
        compute=(*first.compute, *body),
        source=(
            *first.source[:-1],
            *("    " + line for line in body),
            first.source[-1],
        ),
    )
    groups = (replace(groups[0], members=(first, second)), groups[1])
    emission.lowering.result = (
        "before = 1",
        *(line for group in groups for line in group.source),
        "after = 2",
    )
    lowered = _lower(emission, groups)
    assert lowered is not None
    assert f"    if ({expression}) > chain_pair_0_team_thread:" in lowered.result
    assert "        value = chain_pair_0_team_warp" in lowered.result
    assert "chain_0_c" not in {
        node.id
        for node in ast.walk(ast.parse("\n".join(lowered.result)))
        if isinstance(node, ast.Name)
    }
    for receipt in lowered.teams:
        receipt.check()
    emission.lowering.result += ("export[0] = chain_0_c[0]",)
    with pytest.raises(_UnsupportedChain, match="still has an emitted reader"):
        _lower(emission, groups)


@pytest.mark.parametrize("serial", [False, True])
def test_complete_pairs_keep_one_owner_and_retire_only_first_result(serial) -> None:
    emission, groups = _fixture(serial=serial)
    lowered = _lower(emission, groups)
    assert lowered is not None
    assert lowered.waves == (((0,), (2,)) if serial else ((0, 2),))
    assert lowered.result[0] == "before = 1" and lowered.result[-1] == "after = 2"
    assert lowered.retired_pairs == tuple(group.pair for group in groups)
    assert lowered.nodes == emission.plan.dots
    tree = ast.parse("\n".join(lowered.result))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not {"chain_0_c", "chain_0_c_ptr", "chain_2_c", "chain_2_c_ptr"} & names
    assert {"chain_1_c", "chain_3_c"} <= names
    for receipt in lowered.teams:
        receipt.check()
        body = "\n".join(receipt.source)
        assert JOIN not in body
        assert body.index("arrive_and_wait()") < body.index("cute.autovec_copy(")
        assert body.count("cute.autovec_copy(") == 1
        assert "cutlass.Float16(-" in body
    assert "chain_3_b_ptr = (chain_b_workspace + " in "\n".join(lowered.result)


def test_cross_team_in_place_hazard_and_late_export_decline() -> None:
    emission, groups = _fixture(hazard=True)
    assert _lower(emission, groups) is None
    emission, groups = _fixture(late_reader=True)
    with pytest.raises(_UnsupportedChain, match="still has an emitted reader"):
        _lower(emission, groups)
    emission, groups = _fixture()
    first = groups[0]
    cut = pair_result_cut(first.pair, first.members[0].source)
    retired = retire_result_cut(emission.lowering.result, cut)
    with pytest.raises(_UnsupportedChain, match="publication changed"):
        retire_result_cut(retired, cut)
    with pytest.raises(_UnsupportedChain, match="receipt changed"):
        retire_result_cut(emission.lowering.result, replace(cut, owner="chain_2_c"))
