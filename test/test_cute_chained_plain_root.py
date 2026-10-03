from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_tmem_transport import _inputs as _loop_inputs
from .test_cute_chained_loop_tmem_transport import _packed_sequence
from .test_cute_chained_loop_tmem_transport import _source as _loop_source
from .test_cute_chained_preparation_pipeline import _config as _pipeline_config
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as plain
from helion._compiler.cute import chained_stage_operands as producers
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_execution import ChainedExecution
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, planes, rows, groups, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty(
        (batch, planes, groups, rows, columns), device=a.device, dtype=torch.float32
    )
    for bb, pp, gg, rr, cc in hl.tile(
        [batch, planes, groups, rows, columns], block_size=[1, 1, 1, None, None]
    ):
        kk = hl.arange(reduction)
        out[bb.begin, pp.begin, gg.begin, rr, cc] = hl.dot(
            a[bb.begin, pp.begin, rr, gg.begin, kk],
            b[bb.begin, pp.begin, cc, gg.begin, kk].T,
        )
    return out


def _config(columns: int = 32, **overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [128, columns],
            "num_warps": 4,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            **overrides,
        }
    )


def _source(args: tuple[torch.Tensor, ...], config: helion.Config) -> str:
    with _cpu_codegen():
        return _plain._bind_isolated(args).to_code(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("early_release", [False, True])
def test_real_stage_route_retains_original_root_source(dtype, early_release) -> None:
    initialized = torch.cuda.is_initialized()
    args = tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    )
    config = _config(
        cute_chained_tmem_early_release=early_release,
        cute_chained_auxiliary_cache=True,
    )
    with patch.object(plain, "codegen_plain_root", return_value=False):
        reference = _source(args, config)
    with (
        patch.object(
            legacy,
            "codegen_chained_tcgen05",
            side_effect=AssertionError("old root called"),
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        actual = _source(args, config)
    assert emitted.call_count == 1
    assert emitted.call_args.kwargs["terminal_fragment"] is True
    assert emitted.call_args.args[5] == "0"
    assert actual == reference
    assert "chain_0_c =" not in actual
    assert "chain_loop_index" not in actual
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize(
    "columns,reduction", [(64, 16), (64, 64), (256, 128), (128, 256)]
)
def test_plain_stage_shapes_keep_original_publication_and_order(columns, reduction):
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, reduction), (1, 2, columns, 3, reduction))
    )
    config = _config(columns)
    with patch.object(plain, "codegen_plain_root", return_value=False):
        reference = _source(args, config)
    with patch.object(
        legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root called")
    ):
        actual = _source(args, config)
    assert actual == reference
    assert actual.index("cp_async_wait_group(0)") < actual.index("wait_for_alloc()")
    assert actual.index("mbarrier_wait(chain_bars + 0, 0)") < actual.index(
        "chain_epi_values ="
    )


@pytest.mark.parametrize("case", ["mn_major", "fractional_copy_team", "m64"])
def test_physical_transfer_keeps_exact_source_across_stage_admission(case):
    rows, columns, reduction = (
        (64, 32, 128)
        if case == "m64"
        else (128, 32, 16 if case == "fractional_copy_team" else 128)
    )
    a = torch.empty((1, 2, rows, 3, reduction), dtype=torch.bfloat16)
    b = (
        torch.empty((1, 2, reduction, 3, columns), dtype=torch.bfloat16).transpose(2, 4)
        if case == "mn_major"
        else torch.empty((1, 2, columns, 3, reduction), dtype=torch.bfloat16)
    )
    config = _config(columns)
    config.config["block_sizes"] = [rows, columns]
    with patch.object(plain, "codegen_plain_root", return_value=False):
        reference = _source((a, b), config)
    if case == "m64":
        with (
            patch.object(
                legacy,
                "codegen_chained_tcgen05",
                side_effect=AssertionError("old root called"),
            ),
            patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
        ):
            actual = _source((a, b), config)
        assert emitted.call_count == 1
        assert emitted.call_args.args[4].native_rows == 64
    else:
        with patch.object(
            stages, "emit_stage", side_effect=AssertionError("unproved stage called")
        ):
            actual = _source((a, b), config)
    assert actual == reference


def test_lexical_loop_executes_the_same_stage_without_root_orchestrator() -> None:
    with (
        patch.object(
            legacy,
            "codegen_chained_tcgen05",
            side_effect=AssertionError("old root called"),
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        source = _loop_source(
            _packed_sequence,
            _loop_inputs("cpu", torch.bfloat16, steps=3),
            _pipeline_config(16, pipeline=True, consumer_warps=8),
        )
    assert emitted.call_count >= 3
    assert any(call.args[1].loop is not None for call in emitted.call_args_list)
    assert all(
        not call.kwargs.get("terminal_fragment", False)
        for call in emitted.call_args_list
    )
    assert "chain_loop_index" in source or "chain_iteration" in source


def test_declined_complete_proof_does_not_install_partial_code_or_resources() -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    )
    with patch.object(plain, "codegen_plain_root", return_value=False):
        reference = _source(args, _config())
    observed = []

    def decline(cg, plan, stage, geometry):
        observed.append(
            (tuple(cg.device_function.body), tuple(cg.device_function.preamble))
        )
        return None

    with (
        patch.object(plain, "plan_direct_stage_operands", side_effect=decline),
        patch.object(
            stages, "emit_stage", side_effect=AssertionError("partial stage emitted")
        ),
    ):
        actual = _source(args, _config())
    assert len(observed) == 1
    assert actual == reference
    assert actual.count("chain_allocator.allocate(") == 1


def test_b_rejection_after_a_preflight_is_name_and_body_side_effect_free() -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, 16), (1, 2, 32, 3, 16))
    )
    results = []
    original = producers._preflight

    def observe(cg, plan, node, geometry, role):
        df = cg.device_function
        before = (
            dict(df.namespace._base_count),
            set(df.namespace._used_names),
            list(df.arguments),
            dict(plan.tensor_aliases),
            list(df.body),
            list(df.preamble),
        )
        result = original(cg, plan, node, geometry, role)
        after = (
            dict(df.namespace._base_count),
            set(df.namespace._used_names),
            list(df.arguments),
            dict(plan.tensor_aliases),
            list(df.body),
            list(df.preamble),
        )
        assert before == after
        results.append((role, result))
        return result

    with patch.object(producers, "_preflight", side_effect=observe):
        _source(args, _config())
    assert results == [("a", True), ("b", False)]


def test_both_proofs_precede_emission_and_discrepancy_never_installs_a_body() -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    )
    events: list[tuple[str, str, bool | None]] = []
    before_body = []
    original = producers._preflight

    def observe(cg, plan, node, geometry, role):
        before_body.append((cg.device_function, list(cg.device_function.body)))
        result = original(cg, plan, node, geometry, role)
        events.append(("proof", role, result))
        return result

    def decline(cg, plan, stage, role, dtype):
        events.append(("emit", role, None))
        return None

    with (
        patch.object(producers, "_preflight", side_effect=observe),
        patch.object(producers, "direct_stage", side_effect=decline),
        patch.object(
            legacy,
            "codegen_chained_tcgen05",
            side_effect=AssertionError("unsafe fallback"),
        ),
        patch.object(stages, "emit_stage", side_effect=AssertionError("partial stage")),
        pytest.raises(exc.BackendUnsupported, match="proof changed during emission"),
    ):
        _source(args, _config())
    assert events == [
        ("proof", "a", True),
        ("proof", "b", True),
        ("emit", "a", None),
        ("emit", "b", None),
    ]
    assert all(df.body == previous for df, previous in before_body)


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "coupled_geometry",
        "dtype",
        "nodes",
        "reused_stage",
        "participant_count",
        "empty",
        "pending_type",
        "pending_without_copy",
        "terminal_wide",
    ],
)
def test_stage_rejects_incompatible_direct_and_terminal_contracts_before_writes(case):
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    )
    original = stages.emit_stage
    observed = []

    def check(*args, **kwargs):
        rejected = dict(kwargs)
        rejected_args = list(args)
        operands = rejected["direct_operands"]
        if case == "shape":
            rejected["direct_operands"] = replace(operands, shape=(128, 64, 128))
        elif case == "coupled_geometry":
            rejected["direct_operands"] = replace(operands, shape=(128, 64, 128))
            rejected_args[4] = stages.StageGeometry((128, 64, 128), False)
        elif case == "dtype":
            rejected["direct_operands"] = replace(operands, dtype=torch.float32)
        elif case == "nodes":
            rejected["direct_operands"] = replace(operands, nodes=operands.nodes[::-1])
        elif case == "reused_stage":
            plan = args[1]
            rejected_args[1] = replace(
                plan,
                dots=(plan.dots[0], plan.dots[0]),
                shapes=(plan.shapes[0], plan.shapes[0]),
                region=None,
            )
            rejected_args[3] = 1
        elif case == "participant_count":
            rejected_args[1] = replace(args[1], threads=256)
            rejected["execution"] = ChainedExecution(128)
        elif case == "empty":
            rejected["direct_operands"] = replace(operands, b=())
        elif case == "pending_type":
            rejected["pending_allocation"] = 1
        elif case == "pending_without_copy":
            rejected["direct_operands"] = None
        else:
            rejected.update(
                direct_operands=None,
                pending_allocation=None,
                execution=ChainedExecution(256),
            )
        boundaries = dict(args[2])
        body = list(args[0].device_function.body)
        with pytest.raises(chain._UnsupportedChain):
            original(*rejected_args, **rejected)
        assert args[2] == boundaries
        assert args[0].device_function.body == body
        observed.append(case)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=check):
        _source(args, _config())
    assert observed == [case]


@pytest.mark.parametrize("case", ["geometry", "participant_count"])
def test_direct_factory_rejects_inconsistent_contract_before_address_proof(
    case,
) -> None:
    args = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    )
    original = plain.plan_direct_stage_operands
    observed = []

    def check(cg, plan, stage, geometry):
        with patch.object(
            producers,
            "_preflight",
            side_effect=AssertionError("stale contract reached proof"),
        ):
            assert (
                original(
                    cg,
                    replace(plan, threads=256) if case == "participant_count" else plan,
                    stage,
                    stages.StageGeometry((128, 64, 128), False)
                    if case == "geometry"
                    else geometry,
                )
                is None
            )
        observed.append(True)
        return original(cg, plan, stage, geometry)

    with patch.object(plain, "plan_direct_stage_operands", side_effect=check):
        _source(args, _config())
    assert observed == [True]


@pytest.mark.parametrize("early_release", [False, True])
def test_allocation_can_overlap_operand_copies_without_changing_permit_order(
    early_release: bool,
) -> None:
    begin = stages.begin_tmem_resources(32, 1, 128, early_release=early_release)
    finish = stages.finish_tmem_allocation(early_release=early_release)
    release = "chain_allocator.relinquish_alloc_permit()"
    assert (release in begin) is early_release
    assert (release in finish) is not early_release
    assert not any("wait_for_alloc" in line or "retrieve_ptr" in line for line in begin)
    assert finish[:2] == [
        "chain_allocator.wait_for_alloc()",
        "chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)",
    ]
    assert begin.count("chain_allocator.allocate(32)") == 1
    assert sum(line == release for line in begin + finish) == 1


@pytest.mark.parametrize("columns", [32, 64, 160, 512])
@pytest.mark.parametrize("threads", [128, 384, 512])
def test_default_allocation_is_exact_immediate_composition(
    columns: int, threads: int
) -> None:
    allocations = ("example = cute.arch.alloc_smem(cutlass.Float32, 32)",)
    assert stages.allocate_tmem_resources(columns, 3, threads, allocations) == [
        *stages.begin_tmem_resources(columns, 3, threads, allocations),
        *stages.finish_tmem_allocation(early_release=True),
    ]


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_allocation_release_policy_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        stages.begin_tmem_resources(32, 1, 128, early_release=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        stages.finish_tmem_allocation(early_release=value)  # type: ignore[arg-type]
