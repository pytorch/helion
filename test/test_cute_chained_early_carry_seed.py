from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _args as _args128
from .test_cute_chained_load_only_preparation import _config as _config128
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_orientation import _config as _parent_config
from .test_cute_chained_m64_orientation import _square_intermediate
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_early_carry_seed as early
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_result_transport import load_operation
from helion._compiler.cute.chained_seed_tiles import plan_seed_panels


def _config(enabled: bool = True, **options: object) -> helion.Config:
    return helion.Config.from_dict(
        _parent_config().config | {early.KEY: enabled} | options
    )


def _source(enabled: bool = True, **options: object) -> str:
    args = _args()
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_square_intermediate, args)
        ):
            return bound.to_code(_config(enabled, **options))


@pytest.fixture(scope="module")
def emitted() -> tuple[str, str, early.EarlyCarrySeed]:
    original = early.bind_early_carry_seed
    attempts = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        assert result is not None
        attempts.append(result)
        return result

    with patch.object(early, "bind_early_carry_seed", observe):
        off, on = _source(False), _source()
    assert len(attempts) == 1
    return off, on, attempts[0]


def _leaves(source: str) -> Counter[str]:
    tree = ast.parse(source)
    aliases = {
        node.targets[0].id: "chain_value_" + node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("chain_value_")
        and isinstance(node.value, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in aliases:
            node.id = aliases[node.id]
    return Counter(
        ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
    )


def test_original_operations_and_late_fence_are_preserved(
    emitted: tuple[str, str, early.EarlyCarrySeed],
) -> None:
    off, on, attempt = emitted
    assert _leaves(off) == _leaves(on)
    assert attempt.state == "settled"
    assert attempt.candidate.seed_window == (256, 384)
    assert attempt.candidate.snapshot_window == (384, 448)
    assert attempt.candidate.working_windows == ((0, 128), (0, 64))
    assert (
        on.index("tcgen05.commit(chain_bars + 0)")
        < on.index("chain_3_seed_4_load =")
        < on.index("cute.arch.mbarrier_wait(chain_bars + 0")
    )
    late_seed = on.index("cute.copy(chain_3_seed_3_copy")
    late_fence = on.index("cute.arch.fence_view_async_tmem_store()", late_seed)
    late_join = on.index("chain_recurrence_barrier.arrive_and_wait()", late_fence)
    final_mma = on.index("cute.gemm(chain_3_mma")
    assert late_seed < late_fence < late_join < final_mma
    assert on.index("cute.copy(chain_3_seed_4_copy") < late_seed


@pytest.mark.parametrize("value", [1, None, "early"])
def test_invalid_option_rejects(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        _square_intermediate._bind_isolated(_args()).config_spec.normalized_config(
            _config().config | {early.KEY: value}
        )


@pytest.mark.parametrize(
    "options",
    [
        {"cute_chained_preparation_pipeline": False},
        {"cute_chained_pipeline_consumer_warps": 8, "num_warps": 16},
        {"cute_chained_seed_tile_columns": 32},
        {"cute_chained_snapshot_tile_columns": 32},
        {"cute_chained_drain_tile_columns": 32},
        {"cute_chained_mma_schedule": "coalesced"},
    ],
)
def test_unsupported_protocol_rejects(options: dict[str, object]) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig):
        _square_intermediate._bind_isolated(_args()).config_spec.normalized_config(
            _config().config | options
        )


def test_default_roundtrip_and_one_seed_preserve_overrides() -> None:
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(_args())
        assert early.KEY in bound.config_spec._flat_fields()
        config = bound.config_spec.normalized_config(_config())
        assert bound.config_spec.normalized_config(config).config == config.config
        parent = bound.config_spec.normalized_config(_config(False))
        assert early.KEY not in parent.config
        parents = [parent]
        seeded = CuteChainedMatmulHeuristic._with_early_carry_seed(bound.env, parents)
        assert seeded[:-1] == parents and len(seeded) == 2
        assert seeded[-1].config == parent.config | {early.KEY: True}
        assert (
            CuteChainedMatmulHeuristic._with_early_carry_seed(bound.env, seeded)
            == seeded
        )
        with patch.dict(
            bound.env.settings.autotune_config_overrides, {early.KEY: False}
        ):
            assert (
                CuteChainedMatmulHeuristic._with_early_carry_seed(bound.env, parents)
                == parents
            )


def test_reordered_or_repeated_role_declines(
    emitted: tuple[str, str, early.EarlyCarrySeed],
) -> None:
    attempt = emitted[2]
    with pytest.raises(chain._UnsupportedChain, match="action or role"):
        attempt.check_stage(attempt.candidate.predecessor, ChainedExecution(128), "0")
    with pytest.raises(chain._UnsupportedChain, match="consumed out of order"):
        attempt.skip_member(attempt.candidate.member)
    empty = early.EarlyCarrySeed(attempt.body, attempt.candidate)
    with pytest.raises(chain._UnsupportedChain, match="incomplete"):
        empty.finish()
    with pytest.raises(chain._UnsupportedChain, match="missing prior"):
        empty.hoisted_setup(empty.candidate.destination)
    with pytest.raises(chain._UnsupportedChain, match="late aggregate"):
        empty.settle(
            ["cute.arch.fence_view_async_tmem_store()", empty.body.execution.sync]
        )


def test_moved_fence_with_same_instruction_counts_rejects(
    emitted: tuple[str, str, early.EarlyCarrySeed],
) -> None:
    attempt = emitted[2]
    stage = attempt.candidate.destination
    issue, span = attempt.issues[-1]
    valid = [*attempt.late_lines, *span]
    attempt.verify_stage(stage, valid)
    malformed = [
        attempt.late_lines[-2],
        *attempt.late_lines[:-2],
        attempt.late_lines[-1],
        *span,
    ]
    assert Counter(malformed) == Counter(valid)
    with pytest.raises(chain._UnsupportedChain, match="temporal source"):
        attempt.verify_stage(stage, malformed)
    with pytest.raises(chain._UnsupportedChain, match="precedes settlement"):
        attempt.verify_stage(stage, [*span, *attempt.late_lines])
    # Specifically preserve COMMIT after the late fence but move the first
    # MMA before it. Checking only the commit position would accept this.
    split = next(
        i for i, line in enumerate(span) if "with cute.arch.elect_one()" in line
    )
    before_commit = [*span[:split], *attempt.late_lines, *span[split:]]
    fence = before_commit.index("cute.arch.fence_view_async_tmem_store()")
    mma = next(i for i, line in enumerate(before_commit) if "cute.gemm(" in line)
    commit = next(
        i for i, line in enumerate(before_commit) if "tcgen05.commit(" in line
    )
    assert mma < fence < commit and Counter(before_commit) == Counter(valid)
    with pytest.raises(chain._UnsupportedChain, match="temporal source"):
        attempt.verify_stage(stage, before_commit)
    body = replace(attempt.body, _pending=attempt.body.recurrence.stages[-1])
    unready = early.EarlyCarrySeed(body, attempt.candidate)
    with pytest.raises(chain._UnsupportedChain, match="before settlement"):
        unready.before_issue(
            issue.prefix, issue.stage, issue.phase, issue.execution, issue.lines
        )
    with pytest.raises(chain._UnsupportedChain, match="owned native issue"):
        unready.after_commit(issue)
    with pytest.raises(chain._UnsupportedChain, match="unowned issue"):
        unready.complete_issue(replace(issue, phase="0"), list(span))


@pytest.mark.parametrize(
    "option",
    [
        "root_actions",
        "root_body",
        "root_action_body",
        "prepared_edge",
        "direct_operands",
        "operand_producer",
        "prepared_shape",
        "pending_allocation",
        "terminal_fragment",
    ],
)
def test_alternate_issue_protocol_is_explicitly_rejected(
    emitted: tuple[str, str, early.EarlyCarrySeed], option: str
) -> None:
    attempt = emitted[2]
    body = attempt.body
    group = attempt.candidate.predecessor
    options: dict[str, Any] = {option: True}
    with pytest.raises(chain._UnsupportedChain, match="bound recurrence role"):
        chained_tcgen_stage.emit_stage(
            body.codegen,
            body.plan,
            body.boundaries,
            group.stages[0],
            group.geometries[0],
            "chain_iteration & 1",
            group,
            execution=body.execution,
            early_carry=attempt,
            **options,
        )


@pytest.mark.parametrize("fault", ["args", "users", "dtype", "role"])
def test_stale_graph_and_transport_facts_reject(
    emitted: tuple[str, str, early.EarlyCarrySeed], fault: str
) -> None:
    attempt = emitted[2]
    assert attempt.body.carry is not None
    node = attempt.body.carry.candidate.accumulator
    if fault == "role":
        old = attempt.body.execution
        attempt.body.execution = replace(old, thread="other_thread")
        try:
            with pytest.raises(chain._UnsupportedChain, match="changed"):
                attempt.check()
        finally:
            attempt.body.execution = old
        return
    if fault == "args":
        old_args = node.args
        node.args = (*node.args, 1)
        try:
            with pytest.raises(chain._UnsupportedChain, match="changed"):
                attempt.check()
        finally:
            node.args = old_args
    elif fault == "dtype":
        old_val = node.meta["val"]
        node.meta["val"] = old_val.to(dtype=torch.float16)
        try:
            with pytest.raises(chain._UnsupportedChain, match="changed"):
                attempt.check()
        finally:
            node.meta["val"] = old_val
    else:
        with node.graph.inserting_after(node):
            user = node.graph.call_function(torch.ops.aten.neg.default, (node,))
        try:
            with pytest.raises(chain._UnsupportedChain, match="changed"):
                attempt.check()
        finally:
            node.graph.erase_node(user)


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_final_storage_authority_preserved(fault: str) -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source()


def test_no_snapshot_or_live_raw_seed_input_is_not_ready(
    emitted: tuple[str, str, early.EarlyCarrySeed],
) -> None:
    attempt = emitted[2]
    assert early.bind_early_carry_seed(attempt.body, None) is None
    # Remove the exact producer image from the prepared boundaries. Walking
    # through its original raw global load must not grant READY permission.
    assert attempt.body.carry is not None
    candidate = attempt.body.carry.candidate
    frontier = attempt.body.pipeline.frame.frontier_order
    ancestors = set()
    pending = [candidate.accumulator]
    while pending:
        node = pending.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        pending.extend(node.all_input_nodes)
    raw = next(node for node in frontier if node in ancestors)
    bindings = dict(attempt.body.boundaries)
    bindings.pop(raw)
    assert not early.ready_accumulator(replace(attempt.body, boundaries=bindings))


@pytest.mark.parametrize(
    "rows,total,offset,width",
    [(64, 192, 64, 128), (128, 192, 64, 128), (64, 128, 32, 96), (128, 256, 128, 128)],
)
def test_actual_original_seed_read_write_coordinates(
    rows: int, total: int, offset: int, width: int
) -> None:
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    tcgen = importlib.import_module("cutlass.cute.nvgpu.tcgen05")
    helpers = importlib.import_module("cutlass.utils.blackwell_helpers")
    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen.CtaGroup.ONE,
                (rows, total),
                tcgen.OperandSource.SMEM,
            )
            acc = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, 192, cute.AddressSpace.tmem),
                mma.make_fragment_C(mma.partition_shape_C((rows, total))).layout,
            )
            namespace: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "tcgen05": tcgen,
                "group_acc": acc,
                "group_slice": mma.get_slice(0),
            }
            panel = plan_seed_panels((rows, total), offset, width)[0]
            exec("\n".join(panel.store_views("seed", "group", "0")[:3]), namespace)
            store, identity = namespace["seed_copy"], namespace["seed_identity"]
            load = tcgen.make_tmem_copy(
                cute.make_copy_atom(
                    eval(load_operation((rows, width)), namespace), cutlass.Float32
                ),
                namespace["seed_segment"],
            )
            owners = []
            for thread in range(128):
                read = load.get_slice(thread).partition_D(identity)
                write = store.get_slice(thread).partition_S(identity)
                values = [tuple(read[i]) for i in range(cute.size(read))]
                assert values == [tuple(write[i]) for i in range(cute.size(write))]
                physical_read = load.get_slice(thread).partition_S(
                    namespace["seed_segment"]
                )
                physical_write = store.get_slice(thread).partition_D(
                    namespace["seed_segment"]
                )
                read_addresses = [
                    192 + offset + int(physical_read.layout(i))
                    for i in range(cute.size(read))
                ]
                assert read_addresses == [
                    192 + offset + int(physical_write.layout(i))
                    for i in range(cute.size(write))
                ]
                assert all(
                    192 + offset <= address % 65536 < 192 + offset + width
                    for address in read_addresses
                )
                owners.extend(values)
            assert Counter(owners) == Counter(
                (r, c) for r in range(rows) for c in range(offset, offset + width)
            )
        assert module.operation.verify()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_gpu_changing_masked_frames_and_both_seeds(
    dtype: torch.dtype, steps: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell")
    args = list(_args("cuda", dtype=dtype, steps=steps))
    generator = torch.Generator(device="cuda").manual_seed(73)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, device="cuda", generator=generator) * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(steps):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (raw - right[:, step].double() @ entering).to(dtype).double()
        solved = (transform[:, step].double() @ corrected).to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ solved + cross).to(dtype)
        state = (
            update[:, step].double() @ solved
            + state * gate[:, step, -1, None, None].double()
        )
    outputs = []
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        result = _square_intermediate._bind_isolated(tuple(args)).compile_config(
            _config(enabled, cute_chained_fragment_reuse=True)
        )(*args)
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
        outputs.append(tuple(tensor.clone() for tensor in result))
    for original, selected in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(original, selected, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_m128_uses_original_transport_without_new_geometry() -> None:
    args = _args128()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_load_only_recurrence, args)
        ):
            config = helion.Config.from_dict(_config128().config | {early.KEY: True})
            source = bound.to_code(config)
    assert "chain_resident_carry_0_snapshot" in source
    assert (
        source.index("tcgen05.commit(chain_bars + 0)")
        < source.index("chain_2_seed_3_load =")
        < source.index("cute.arch.mbarrier_wait(chain_bars + 0")
    )


def test_gpu_m128_changing_frames_and_state() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires Blackwell")
    dtype, steps = torch.bfloat16, 3
    args = _args128("cuda", steps=steps, dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(7201)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, device="cuda", generator=generator) * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    for step in range(steps):
        entering = state.to(dtype).double()
        corrected = value[:, step].double() - right[:, step].double() @ entering
        narrowed = corrected.to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ narrowed + cross).to(dtype)
        state = (
            (
                update[:, step].double() @ narrowed
                + state * gate[:, step, -1, None, None].double()
            )
            .float()
            .double()
        )
    outputs = []
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        config = helion.Config.from_dict(_config128().config | {early.KEY: enabled})
        result = _load_only_recurrence._bind_isolated(args).compile_config(config)(
            *args
        )
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[0][:, -1], expected[:, -1], atol=0, rtol=0)
        outputs.append(tuple(tensor.clone() for tensor in result))
    for original, selected in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(original, selected, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "carry_overlap",
        "snapshot_overlap",
        "snapshot_seed",
        "unknown_arena",
        "snapshot_source",
    ],
)
def test_admission_rejects_unknown_or_overlapping_transport(
    emitted: tuple[str, str, early.EarlyCarrySeed], fault: str
) -> None:
    attempt = emitted[2]
    body, snapshot = attempt.body, attempt.candidate.ready
    assert (
        body.carry is not None
        and body.selected_transports is not None
        and body.storage is not None
    )
    if fault == "snapshot_source":
        assert (
            early.bind_early_carry_seed(
                body, replace(snapshot, lines=snapshot.lines[1:])
            )
            is None
        )
        return
    carry = body.carry
    if fault == "carry_overlap":
        carry = replace(carry, arena_offset=0)
    elif fault == "snapshot_overlap":
        carry = replace(carry, snapshot_offset=attempt.candidate.seed_window[0])
    elif fault == "snapshot_seed":
        carry = replace(carry, snapshot_seed=((), "0.0"))
    selections = tuple(
        replace(item, tmem_carry=carry, tmem_accumulator=carry.arena)
        if item.group is attempt.candidate.destination
        else replace(item, tmem_input=carry.operand(item.group))
        for item in body.selected_transports
    )
    if fault == "unknown_arena":
        selections = (
            replace(selections[0], tmem_accumulator="unproved_arena"),
            *selections[1:],
        )
    modified = replace(
        body,
        carry=carry,
        selected_transports=selections,
        storage=replace(body.storage, stages=selections),
        early_carry=None,
    )
    ready = replace(
        snapshot,
        carry=carry,
        facts=early._records((carry, body.execution, snapshot.lines)),
    )
    assert early.bind_early_carry_seed(modified, ready) is None
