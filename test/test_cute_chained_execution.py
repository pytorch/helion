from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from dataclasses import replace
import hashlib
import json
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _group_config_candidate
from .test_cute_chained_loop_collectives import _inputs as _collective_args
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
from .test_cute_chained_producer_teams import _args
from .test_cute_chained_producer_teams import _cooperative_loop
from .test_cute_chained_residency_search import _loop_residency
from .test_cute_chained_residency_search import _root_residency
import helion
from helion._compiler.cute import chained_collectives
from helion._compiler.cute import chained_matmul
from helion._compiler.cute import chained_pointwise_residency
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute import chained_vector_stage
from helion._compiler.cute import chained_warp_mma
from helion._compiler.cute.chained_execution import ChainedExecution

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx import Node

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST


_CASES = (
    "stage",
    "vector",
    "warp",
    "collective_serial",
    "collective_warp",
    "cache",
    "root",
)

# SHA256 of raw (helper-name, emitted-lines) records from these CPU fixtures.
# The warp entry includes the prepared K helper call; its ordered physical
# schedule is checked independently in test_cute_prepared_warp_contraction.py.
# Collective/cache entries include the shared producer's AST rendering; their
# operator trees and helper boundaries match the pre-extraction fixtures.
# Whole-kernel source is excluded because unrelated arena allocation is owned
# elsewhere.
_LEGACY_DIGESTS = {
    "stage": "52185f20d9b5c5650c3105dce3be9ac369554da5dd33c4368b790da40e0f0bd2",
    "vector": "b8e86ed340b384d787b1fec5ee110a38dae9b435243535e326e94d95eb24a234",
    "warp": "91dd98c20b7a1053170d6240d95b8e5f73a828a9ff6f5483cf7ef1d5ce0b3e69",
    "collective_serial": "f696de39078ac5963d98f2ab5c975c8e87ee138e8ed1ad91657338edae5262a7",
    "collective_warp": "fb32bf2bad6a4eb243026f863d421997165bc3af8826489d89d997676558ec8e",
    "cache": "1a78aa495d425f07fdcb5e519c887ce9a9a07da16d38eaf646ba303f5b8e8017",
    "root": "def62008cde4968f5be0817f009d5f2a00f818eab992aa134561a198d44b05b5",
}


def _emissions(
    case: str,
    execution: ChainedExecution | None = None,
    *,
    check: Callable | None = None,
) -> list[tuple[str, list[str]]]:
    """Record actual helper emissions, excluding unrelated allocation changes."""
    records: list[tuple[str, list[str]]] = []

    def recorder(name: str, original: Callable):
        def record(*args, **kwargs):
            if check is not None:
                check(name, original, args, kwargs)
            if execution is not None:
                kwargs.setdefault("execution", execution)
            result = original(*args, **kwargs)
            if result:
                records.append((name, result))
            return result

        return record

    stage = recorder("stage", chained_tcgen_stage.emit_stage)
    vector = recorder("vector", chained_vector_stage.emit_vector_stage)
    mma = recorder("warp", chained_warp_mma.emit_warp_mma)
    load = recorder("load", chained_tcgen05._load_result)
    collective = recorder("collective", chained_collectives.emit_collectives_before)
    cache = recorder("cache", chained_pointwise_residency.emit_pointwise_cache_before)
    with _cpu_codegen(), ExitStack() as stack:
        for module, name, function in (
            (chained_tcgen_stage, "emit_stage", stage),
            (chained_tcgen_stage, "emit_vector_stage", vector),
            (chained_vector_stage, "emit_vector_stage", vector),
            (chained_tcgen_stage, "emit_warp_mma", mma),
            (chained_warp_mma, "emit_warp_mma", mma),
            (chained_tcgen_stage, "_load_result", load),
            (chained_tcgen05, "_load_result", load),
            (chained_collectives, "emit_collectives_before", collective),
            (chained_tcgen05, "emit_collectives_before", collective),
            (chained_pointwise_residency, "emit_pointwise_cache_before", cache),
            (chained_tcgen05, "emit_pointwise_cache_before", cache),
        ):
            stack.enter_context(patch.object(module, name, function))
        config = helion.Config(num_warps=16, cute_chained_mma_schedule="tcgen05_tmem")
        if case == "wide_warp":
            bound = _group_config_candidate._bind_isolated(
                (
                    torch.empty((3, 64, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((64, 32), dtype=torch.float32),
                )
            )
            config.config["cute_chained_warp_mma_rows"] = 64
        elif case in ("stage", "vector", "warp"):
            bound = _cooperative_loop._bind_isolated(_args("cpu", 3, True))
            if case != "warp":
                config.config["cute_chained_group_contractions"] = True
            if case == "vector":
                config.config["cute_chained_pointwise_vectorize"] = True
            if case == "warp":
                config.config["cute_chained_warp_mma_rows"] = 32
        elif case.startswith("collective_"):
            bound = _prefix_coefficient_recurrence._bind_isolated(
                _collective_args("cpu", 3, 1)
            )
            config.config["cute_chained_scan_schedule"] = case.removeprefix(
                "collective_"
            )
        elif case == "cache":
            bound = _loop_residency._bind_isolated(
                (
                    torch.empty((3, 128, 16), dtype=torch.bfloat16),
                    torch.empty((3, 16, 32), dtype=torch.bfloat16),
                    torch.empty((128, 32), dtype=torch.float32),
                )
            )
            config.config["cute_chained_pointwise_cache_bytes"] = 4096
        else:
            assert case == "root"
            bound = _root_residency._bind_isolated(
                (
                    torch.empty((128, 16), dtype=torch.bfloat16),
                    torch.empty((16, 32), dtype=torch.bfloat16),
                )
            )
            config = helion.Config(
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_pointwise_cache_bytes=4096,
            )
        bound.to_code(config)
    assert records
    return records


def _digest(records: list[tuple[str, list[str]]]) -> str:
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


@pytest.mark.parametrize("threads", [0, 1, 31, 33, 1025, 2048, True, 128.0])
def test_context_rejects_partial_warp_or_invalid_team_sizes(threads) -> None:
    with pytest.raises(ValueError, match="whole warps"):
        ChainedExecution(threads)


def test_context_is_immutable_and_frame_resources_can_be_rebound() -> None:
    original = ChainedExecution(
        256,
        thread="prep_thread",
        warp="prep_warp",
        sync="prep_barrier.arrive_and_wait()",
    )
    frame = replace(
        original,
        a_workspace="prep_a + frame * 4096",
        b_workspace="prep_b + frame * 2048",
    )
    assert frame.thread == original.thread and frame.sync == original.sync
    assert original.a_workspace == "chain_a_workspace"
    assert frame.a_workspace == "prep_a + frame * 4096"
    with pytest.raises(FrozenInstanceError):
        original.threads = 128  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("explicit", [False, True])
def test_default_emissions_match_recorded_bytes(case: str, explicit: bool) -> None:
    execution = ChainedExecution(128 if case == "root" else 512) if explicit else None
    assert _digest(_emissions(case, execution)) == _LEGACY_DIGESTS[case]


def _role(threads: int = 256) -> ChainedExecution:
    return ChainedExecution(
        threads,
        thread="prep_thread",
        warp="prep_warp",
        sync="prep_barrier.arrive_and_wait()",
        a_workspace="prep_a + frame * 4096",
        b_workspace="prep_b + frame * 2048",
        tmem="prep_tmem",
        barriers="prep_mma_bars + frame * 3",
    )


@pytest.mark.parametrize("case", _CASES[:-1])
def test_nonzero_role_uses_only_local_ownership_and_resources(case: str) -> None:
    records = _emissions(case, _role())
    combined = "\n".join(line for _, lines in records for line in lines)
    for forbidden in (
        "chain_thread",
        "chain_warp",
        "cute.arch.sync_threads()",
        "chain_a_workspace",
        "chain_b_workspace",
        "chain_tptr",
        "chain_bars",
    ):
        assert forbidden not in combined
    assert "prep_thread" in combined
    assert "prep_barrier.arrive_and_wait()" in combined
    assert "prep_a + frame * 4096" in combined
    assert "prep_b + frame * 2048" in combined
    assert "prep_tmem" in combined
    assert "if prep_warp == 0:" in combined
    assert "tcgen05.commit(prep_mma_bars + frame * 3 +" in combined
    assert "cute.arch.mbarrier_wait(prep_mma_bars + frame * 3 +" in combined
    assert "if prep_thread < 128:" in combined
    if case != "vector":
        assert "chain_0_a_0_step * 256" in combined
    else:
        vector_source = "\n".join(
            line for kind, lines in records if kind == "vector" for line in lines
        )
        assert "get_slice(prep_thread)" in vector_source
        assert "prep_thread // 2" in vector_source
        assert "prep_thread % 2 * 8" in vector_source
    if case == "cache":
        cache_source = "\n".join(
            line for kind, lines in records if kind == "cache" for line in lines
        )
        assert "_step * 256" in cache_source
    for kind, lines in records:
        if kind != "stage":
            continue
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test).startswith(
                "prep_thread <"
            ):
                assert not any(
                    isinstance(call, ast.Call)
                    and ast.unparse(call.func) == "prep_barrier.arrive_and_wait"
                    for call in ast.walk(node)
                )


@pytest.mark.parametrize("threads", [32, 64, 96])
def test_explicit_small_roles_reject_tcgen_copy_participants(threads: int) -> None:
    observed = []

    def check(name, original, args, kwargs):
        if name == "stage" and args[3] == 0:
            with pytest.raises(ValueError, match="at least 128"):
                original(*args, **kwargs, execution=_role(threads))
            observed.append(True)

    _emissions("stage", check=check)
    assert observed == [True]
    with pytest.raises(ValueError, match="at least 128"):
        chained_tcgen05._load_result("test", (128, 32), execution=_role(threads))


@pytest.mark.parametrize("shape", [(128, 16), (128, 128), (64, 64)])
def test_tmem_load_helper_accepts_context_without_callsite_rewriting(shape) -> None:
    legacy = chained_tcgen05._load_result("test", shape)
    role = chained_tcgen05._load_result("test", shape, execution=_role())
    assert role[1] == "test_thread = test_copy.get_slice(prep_thread)"
    assert role[:1] + role[2:] == legacy[:1] + legacy[2:]


def test_warp_mma_rejects_an_active_team_larger_than_the_role() -> None:
    with pytest.raises(ValueError, match="exceed"):
        chained_warp_mma.emit_warp_mma(
            "test",
            "cutlass.BFloat16",
            (16, 32, 16),
            128,
            {"a": 1, "b": 1},
            [],
            execution=_role(64),
        )


def test_three_warp_role_uses_a_divisor_team_for_warp_mma() -> None:
    observed = []

    def check(name, original, args, kwargs):
        if name != "stage" or args[3] != 0:
            return
        independent = (*args[:2], dict(args[2]), *args[3:])
        lines = original(*independent, **kwargs, execution=_role(96))
        source = "\n".join(lines)
        assert "atom_layout_mnk=(1, 2, 1)" in source
        assert "if prep_thread < 64:" in source
        assert "chain_0_a_0_step * 96" in source
        assert source.endswith("prep_barrier.arrive_and_wait()")
        observed.append(source)

    _emissions("wide_warp", check=check)
    assert len(observed) == 1


def test_vector_stage_rejects_a_fractional_vector_team_before_expression_emission() -> (
    None
):
    # 96 participants cannot form rows of 64 vector producers. The caller
    # must retain the scalar route rather than send excess threads to a copy.
    with patch.object(
        chained_vector_stage.chain,
        "_Expression",
        side_effect=AssertionError("expression must not run"),
    ):
        assert (
            chained_vector_stage.emit_vector_stage(
                cast("GenerateAST", None),
                cast("ChainedMatmulPlan", None),
                {},
                cast("Node", None),
                chained_tcgen_stage.StageGeometry((16, 16, 16), False),
                role="a",
                shape=(128, 512),
                offset=0,
                tag="vector",
                target="operand",
                execution=_role(96),
            )
            is None
        )


@pytest.mark.parametrize("case", ["collective_serial", "collective_warp"])
def test_selected_collective_keeps_complete_plan_name_and_one_publication(
    case: str,
) -> None:
    observed = []

    def check(name, original, args, kwargs):
        if name != "collective" or args[3] != 0:
            return
        cg, plan, boundaries, stage = args
        bindings = chained_collectives.collective_bindings(plan)
        assert len(bindings) == 3
        chosen = tuple(bindings)[1]
        independent = dict(boundaries)
        independent.pop(chosen, None)
        source = "\n".join(
            original(
                cg,
                plan,
                independent,
                stage,
                selected=frozenset((chosen,)),
                execution=_role(),
            )
        )
        assert "chain_collective_1_acc" in source
        assert "chain_collective_0_acc" not in source
        assert "chain_collective_2_acc" not in source
        assert source.count("prep_barrier.arrive_and_wait()") == 1
        assert independent[chosen] == bindings[chosen]
        assert (
            original(
                cg,
                plan,
                dict(boundaries),
                stage,
                selected=frozenset(),
                execution=_role(),
            )
            == []
        )
        with pytest.raises(chained_matmul._UnsupportedChain, match="not eligible"):
            original(
                cg, plan, dict(boundaries), stage, selected=frozenset((plan.dots[0],))
            )
        with pytest.raises(chained_matmul._UnsupportedChain, match="not eligible"):
            original(
                cg, plan, dict(boundaries), stage + 1, selected=frozenset((chosen,))
            )
        observed.append(source)

    _emissions(case, check=check)
    assert len(observed) == 1


def test_selected_cache_preserves_peer_names_readiness_and_single_publication() -> None:
    observed = []

    def check(name, original, args, kwargs):
        if name != "cache" or args[4] != 0:
            return
        cg, plan, boundaries, cache, stage = args
        (entry,) = cache.entries
        peer_node = plan.dots[0].args[1]
        peer = replace(
            entry,
            node=peer_node,
            name="chain_pointwise_cache_peer",
            shape=chained_matmul._shape(peer_node),
            dtype=peer_node.meta["val"].dtype,
            dependencies=(),
        )
        cache = replace(cache, entries=(entry, peer))
        independent = dict(boundaries)
        source = "\n".join(
            original(
                cg,
                plan,
                independent,
                cache,
                stage,
                selected=frozenset((peer_node,)),
                execution=_role(),
            )
        )
        assert peer.name in source and entry.name not in source
        assert source.count("prep_barrier.arrive_and_wait()") == 1
        assert independent[peer_node] == peer.name and entry.node not in independent
        assert (
            original(
                cg,
                plan,
                dict(boundaries),
                cache,
                stage,
                selected=frozenset(),
                execution=_role(),
            )
            == []
        )
        with pytest.raises(chained_matmul._UnsupportedChain, match="not eligible"):
            original(
                cg,
                plan,
                dict(boundaries),
                cache,
                stage + 1,
                selected=frozenset((peer_node,)),
            )
        with pytest.raises(chained_matmul._UnsupportedChain, match="published source"):
            original(cg, plan, {}, cache, stage, selected=frozenset((entry.node,)))
        observed.append(source)

    _emissions("cache", check=check)
    assert len(observed) == 1
