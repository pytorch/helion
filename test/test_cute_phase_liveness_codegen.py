"""Full static axes survive projection into an ordered consumer phase."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.compile_environment import tensor_input_sources
from helion._compiler.cute.materialized_fission_codegen import _region_graph_ids
from helion._compiler.cute.materialized_fission_codegen import _stage_host
from helion._compiler.cute.ordered_phases import _ordered_phase_bundle
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_fragment_simulator import simulate_independent_fragment
from helion._testing.cute_phase_liveness import explicit_barrier_row_topk
from helion._testing.cute_phase_liveness import explicit_direct_selection
from helion._testing.cute_phase_liveness import grouped_selection_after_producer
from helion._testing.cute_phase_liveness import nested_row_copy
from helion._testing.cute_phase_liveness import padded_factory_phases
from helion._testing.cute_phase_liveness import routed_after_producer
from helion._testing.cute_phase_liveness import routed_phase_config
from helion._testing.cute_phase_liveness import row_sum_after_independent_store
from helion._testing.cute_phase_liveness import row_sum_after_producer
from helion._testing.cute_phase_liveness import row_sum_phase_config
from helion._testing.cute_phase_liveness import row_sum_two_outputs
from helion._testing.cute_phase_liveness import selected_rows_after_producer
from helion._testing.cute_phase_liveness import selected_rows_config
from helion._testing.cute_phase_liveness import slice_sum_phases
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_projected_consumer_retains_static_memory_axes(tensor_indices):
    with cpu_target():
        bound = cpu_bind(
            slice_sum_phases,
            (
                torch.ones(2, 3, 7, 8, dtype=torch.bfloat16),
                torch.ones(2, 3, 8, 11, dtype=torch.bfloat16),
                tensor_indices,
            ),
        )
        config = dict(bound.config_spec.default_config()) | {
            "pid_type": "flat",
        }
        code = bound.to_code(helion.Config.from_dict(config))
    stages = stage_sources(code)
    assert len(stages) == 2
    assert "thread_idx()[0]" in stages[1]
    assert "thread_idx()[1]" in stages[1]
    expected_block = "(7, 11, 1)" if tensor_indices else "(8, 16, 1)"
    assert f"block={expected_block}" in stages[1]


def test_factory_padding_does_not_resurrect_another_phases_axes():
    with cpu_target():
        bound = cpu_bind(padded_factory_phases, (torch.ones(8, 16),))
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            bundle, plan = _ordered_phase_bundle(bound.host_function, ir)
            graph_ids = _region_graph_ids(ir.graphs, ir.root_ids[1])
            stage = _stage_host(
                bundle, ir.phases[1].root_nodes[0], 1, ir.graphs, graph_ids, plan
            )
            unrelated = {bound.env.resolve_block_id(size) for size in (8, 16)}
            assert None not in unrelated
            assert unrelated.isdisjoint(stage.device_ir.codegen_active_block_ids)
            logical = {bound.env.resolve_block_id(size) for size in (7, 11)}
            assert None not in logical
            assert logical.issubset(stage.device_ir.codegen_active_block_ids)
            assert any(
                isinstance(value := node.meta.get("val"), torch.Tensor)
                and tuple(value.shape) == (8, 16)
                for graph in stage.device_ir.graphs
                for node in graph.graph.nodes
                if node.op == "call_function"
            )


def test_warp_routing_does_not_enlarge_a_siblings_partial_warp():
    with cpu_render_target(fake_device_sm_count=148, smem_budget_bytes=232448):
        bound = cpu_bind(
            routed_after_producer,
            (
                torch.zeros((9, 4), dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                2,
            ),
        )
        assert "fragment_block_volume" in bound.to_code(
            routed_phase_config(1, warp=False)
        )
        with pytest.raises(exc.InvalidConfig, match="whole number of warps"):
            bound.to_code(routed_phase_config(1, warp=True))


def test_interleaved_grid_restarts_at_each_dependency_phase():
    with cpu_render_target(fake_device_sm_count=148, smem_budget_bytes=232448):
        bound = cpu_bind(
            routed_after_producer,
            (
                torch.zeros((307, 16), dtype=torch.int32),
                torch.zeros(16, dtype=torch.int32),
                4,
            ),
        )
        # The ordinary blocked schedule stays the default. The interleaved
        # choice assigns even a short phase independently of its global offset.
        assert bound.config_spec.allowed_pid_types == (
            "persistent_blocked",
            "persistent_interleaved",
        )
        spec = bound.config_spec
        key = "cute_fragment_reduction"
        assert spec._flat_fields()[key].choices == ("serial", "warp")
        group = next(g for g in spec.compiler_coverage_groups if g.key == key)
        assert group.domain == ("serial", "warp")
        for invalid in ("cta", "cta_unroll"):
            requested = routed_phase_config(8, warp=False)
            requested.config[key] = invalid
            with pytest.raises(exc.InvalidConfig, match=key):
                spec.normalize(requested)
        config = routed_phase_config(8, warp=True)
        config.config["pid_type"] = "persistent_interleaved"
        source = bound.to_code(config)
        kernel = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)
            and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
        )
        loops = [
            n
            for n in kernel.body
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id.startswith("phase_iteration")
        ]
        barriers = [
            n
            for n in kernel.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "_cute_grid_barrier"
        ]
        assert len(loops) == 2 and len(barriers) == 1
        assert loops[0].lineno < barriers[0].lineno < loops[1].lineno
        # The producer alone reads x and the consumer alone reads payload.
        # Both still share the scores buffer on opposite sides of the barrier.
        operands = {"x", "payload", "scores", "out"}
        assert [
            {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)} & operands
            for loop in loops
        ] == [{"x", "scores"}, {"payload", "scores", "out"}]
        for loop in loops:
            assert isinstance(loop.iter, ast.Call) and len(loop.iter.args) == 1
            assert "block_idx()" in ast.unparse(loop.iter.args[0])
            assert "_NUM_SM" in ast.unparse(loop.iter.args[0])
            assigned = loop.body[0]
            assert isinstance(assigned, ast.Assign)
            assert ast.unparse(assigned.targets[0]) == "virtual_pid"
            assert loop.target.id in ast.unparse(assigned.value)

        # Evaluate the emitted first-phase trip count and owned IDs at the
        # signed index limit. Here each row produces one physical PID.
        class Arch:
            worker = 0

            def block_idx(self):
                return (self.worker, 0, 0)

        arch = Arch()
        context = {
            "cutlass": SimpleNamespace(Int32=int, Int64=int),
            "cute": SimpleNamespace(arch=arch),
            "rows": 2**31 - 1,
            "_BLOCK_SIZE_1": 16,
            "_NUM_SM": 148,
        }
        # Expressions originate in the compiled fixture above.
        first = loops[0]
        extent = compile(
            ast.fix_missing_locations(ast.Expression(first.iter.args[0])),
            "<phase-count>",
            "eval",
        )
        index = compile(
            ast.fix_missing_locations(ast.Expression(first.body[0].value)),
            "<phase-id>",
            "eval",
        )
        for worker in (0, 1, 7, 147):
            arch.worker = worker
            trips = eval(extent, {"__builtins__": {}}, context)
            assert trips == len(range(worker, 2**31 - 1, 148))
            for i in (0, trips - 1):
                context[first.target.id] = i
                assert eval(index, {"__builtins__": {}}, context) == worker + i * 148


@pytest.mark.parametrize("mode", ["direct", "reduced_index", "nested_reduced_index"])
def test_grouped_selection_keeps_refusal_for_an_aliased_payload(mode):
    x = torch.zeros((7, 62), dtype=torch.int32)
    output = torch.zeros((7, 2, 4), dtype=torch.int32)
    # These views share storage even though their shapes and subscripts differ.
    storage = torch.zeros(80, dtype=torch.int32)
    with cpu_render_target(fake_device_sm_count=148, smem_budget_bytes=232448):
        bound = cpu_bind(
            grouped_selection_after_producer,
            (x, storage[:62], storage[: output.numel()].view_as(output), mode),
        )
        if mode == "direct":
            assert not any(
                g.key == "cute_fragment_topk_network"
                for g in bound.config_spec.compiler_coverage_groups
            )
        with pytest.raises(exc.BackendUnsupported, match="topk output"):
            bound.to_code(routed_phase_config(8, warp=False))
        if mode != "direct":
            with pytest.raises(exc.InvalidConfig, match="cute_fragment_reduction"):
                bound.to_code(routed_phase_config(8, warp=True))


@helion.kernel(backend="cute", static_shapes=True)
def _shared_row_topk(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    scores = torch.empty_like(x)
    out = torch.empty((rows, 4), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        scores[row, col] = x[row, col] * 3 - 1
    for row, slot in hl.tile([rows, 4], block_size=[1, 1]):
        values = torch.topk(scores[row.begin, :], 4).values
        out[row.begin, slot.begin] = torch.where(
            hl.arange(4) == slot.begin, values, 0
        ).sum()
    return out


def test_shared_values_only_int64_topk_cannot_own_the_cta() -> None:
    x = torch.empty(2, 32, dtype=torch.int64)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shared_row_topk, (x,))
        spec = bound.config_spec
        assert spec.cute_fragment_topk_network_root_ids
        assert not spec.cute_fragment_cta_topk_root_ids
        assert all(
            group.key != "cute_fragment_cta_topk"
            for group in spec.compiler_coverage_groups
        )
        warp = routed_phase_config(32, warp=True)
        assert "fragment_topk_keys" in bound.to_code(warp)
        with pytest.raises(exc.InvalidConfig, match="cute_fragment_cta_topk"):
            spec.normalized_config(
                helion.Config.from_dict(warp.config | {"cute_fragment_cta_topk": True})
            )


def test_explicit_barrier_values_only_topk_requires_an_owned_launch() -> None:
    x = torch.empty(2, 32, dtype=torch.int64)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(explicit_barrier_row_topk, (x,))
        spec = bound.config_spec
        assert spec.cute_fragment_cta_topk_root_ids
        threads = next(
            g for g in spec.compiler_coverage_groups if g.key == "cute_fragment_threads"
        )
        bigger = next(w for w in threads.witnesses if w.value == 256)
        threaded = spec.normalized_config(bigger.carrier.config | {threads.key: 256})
        assert threaded.pid_type == "flat"
        assert "block=(256, 1, 1)" in stage_sources(bound.to_code(threaded))[-1]
        config = routed_phase_config(32, warp=True).config | {
            "cute_fragment_cta_topk": True
        }
        config.pop("cross_loop_pipeline")
        with pytest.raises(exc.InvalidConfig, match="cute_fragment_cta_topk"):
            spec.normalized_config(helion.Config.from_dict(config))
        with pytest.raises(exc.InvalidConfig, match="cute_fragment_reduction"):
            spec.normalized_config(
                helion.Config.from_dict(
                    config
                    | {
                        "cute_fragment_cta_topk": False,
                        "cute_fragment_reduction": "cta",
                    }
                )
            )
        warp = helion.Config.from_dict(config | {"cute_fragment_cta_topk": False})
        assert "fragment_topk_keys" in bound.to_code(warp)
        flat = helion.Config.from_dict(config | {"pid_type": "flat"})
        assert spec.normalized_config(flat).get("cute_fragment_cta_topk") is True
        assert "fragment_cta_sort_keys" in stage_sources(bound.to_code(flat))[-1]
        reductions = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_reduction"
        )
        warp_witness = next(w for w in reductions.witnesses if w.value == "warp")
        assert warp_witness.carrier.pid_type in (
            "persistent_blocked",
            "persistent_interleaved",
        )
        assert all(
            w.carrier.pid_type == "flat"
            for w in reductions.witnesses
            if w.value != "warp"
        )
        cta = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_cta_topk"
        )
        assert cta.witnesses
        for witness in cta.witnesses:
            candidate = spec.normalized_config(witness.carrier.config | {cta.key: True})
            assert candidate.pid_type == "flat"
            assert (
                "fragment_cta_sort_keys" in stage_sources(bound.to_code(candidate))[-1]
            )


def test_forced_persistent_barrier_offers_only_warp_reductions() -> None:
    kernel = helion.kernel(
        explicit_barrier_row_topk.fn,
        backend="cute",
        static_shapes=True,
        autotune_force_persistent=True,
    )
    x = torch.empty(2, 32, dtype=torch.int64)
    with cpu_render_target(fake_device_sm_count=148):
        spec = cpu_bind(kernel, (x,)).config_spec
        assert "flat" not in spec.allowed_pid_types
        reduction = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_reduction"
        )
        assert reduction.domain == ("serial", "warp")
        assert not any(
            g.key == "cute_fragment_cta_topk" for g in spec.compiler_coverage_groups
        )


@helion.kernel(backend="cute", static_shapes=True)
def _explicit_rank_three_topk(x: torch.Tensor) -> torch.Tensor:
    scores = torch.empty_like(x)
    values = torch.empty((*x.shape[:-1], 4), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        scores[row, :, :] = x[row, :, :] + 3
    hl.barrier()
    for row in hl.tile(x.size(0)):
        values[row, :, :] = torch.topk(scores[row, :, :], 4).values
    return values


@helion.kernel(backend="cute", static_shapes=True)
def _explicit_fp32_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scores = torch.empty_like(x)
    out = torch.empty_like(x)
    for batch in hl.tile(x.size(0), block_size=1):
        scores[batch, :, :] = x[batch, :, :] + 3
    hl.barrier()
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(scores[batch, :, :] * 2, y[batch, :, :])
    return out


@pytest.mark.parametrize("dot", [False, True], ids=["rank-three-selection", "fp32-dot"])
def test_explicit_complete_fragments_get_flat_coverage(dot):
    key, mode = (
        ("cute_fp32_dot", "tf32_mma") if dot else ("cute_fragment_topk_network", True)
    )
    kernel = _explicit_fp32_dot if dot else _explicit_rank_three_topk
    args = (
        (torch.empty(2, 64, 64), torch.empty(2, 64, 64))
        if dot
        else (torch.empty(2, 3, 32, dtype=torch.int64),)
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        group = next(g for g in spec.compiler_coverage_groups if g.key == key)
        witness = next(w for w in group.witnesses if w.value == mode)
        cfg = witness.carrier.config | {key: mode}
        assert spec.normalized_config(cfg).pid_type == "flat"
        with pytest.raises(exc.InvalidConfig, match=key):
            spec.normalized_config(cfg | {"pid_type": "persistent_blocked"})
        source = stage_sources(bound.to_code(helion.Config.from_dict(cfg)))[-1]
        assert (
            "_helion_fragment_mma.mma_tf32(" if dot else "fragment_topk_keys"
        ) in source


@helion.kernel(backend="cute", static_shapes=True)
def _explicit_independent_producers(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    scores, extra = torch.empty_like(x), torch.empty_like(x)
    values = torch.empty((rows, 4), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([rows, cols], block_size=[1, None]):
        scores[row, col] = x[row, col] + 1
    for row, col in hl.tile([rows, cols], block_size=[1, None]):
        extra[row, col] = x[row, col] - 1
    hl.barrier()
    for row, slot in hl.tile([rows, 4], block_size=[1, 1]):
        chosen = torch.topk(scores[row.begin, :], 4).values
        values[row.begin, slot.begin] = torch.where(
            hl.arange(4) == slot.begin, chosen, 0
        ).sum()
    return values, extra


def test_shared_warp_selection_after_two_independent_explicit_roots():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            _explicit_independent_producers, (torch.empty(2, 32, dtype=torch.int64),)
        )
        ir = bound.host_function.device_ir
        assert not ir.implicit_dependency_starts and [
            len(phase.roots) for phase in ir.phases
        ] == [2, 1]
        spec = bound.config_spec
        assert not spec.cute_fragment_cta_topk_root_ids
        warp = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_topk_network"
        )
        witness = warp.witnesses[0]
        config = spec.normalized_config(witness.carrier.config | {warp.key: True})
        assert config.pid_type.startswith("persistent_")
        assert "fragment_topk_keys" in bound.to_code(config)


@pytest.mark.parametrize("explicit", [False, True])
def test_shared_selection_without_reduction_can_use_warp_topk(explicit) -> None:
    x = torch.empty((7, 62), dtype=torch.int32)
    payload = torch.empty(62, dtype=torch.int32)
    out = torch.empty((7, 2, 4), dtype=torch.int32)
    with cpu_render_target(fake_device_sm_count=148):
        bound = (
            cpu_bind(explicit_direct_selection, (x, payload, out, "direct"))
            if explicit
            else cpu_bind(grouped_selection_after_producer, (x, payload, out, "direct"))
        )
        assert bound.config_spec.cute_fragment_topk_network_root_ids
        config = routed_phase_config(32, warp=True)
        config.config.pop("cute_fragment_reduction")
        if explicit:
            config.config.pop("cross_loop_pipeline")
            config.config["pid_type"] = "flat"
            assert "fragment_topk_keys" in stage_sources(bound.to_code(config))[-1]
            config.config["cute_fragment_topk_network"] = False
            assert "fragment_topk_keys" not in stage_sources(bound.to_code(config))[-1]
        else:
            assert "fragment_topk_keys" in bound.to_code(config)


def test_forced_persistent_rank_three_omits_the_unusable_warp_choice():
    kernel = helion.kernel(
        _explicit_rank_three_topk.fn,
        backend="cute",
        static_shapes=True,
        autotune_force_persistent=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        spec = cpu_bind(kernel, (torch.empty(2, 3, 32, dtype=torch.int64),)).config_spec
        assert "flat" not in spec.allowed_pid_types
        assert "cute_fragment_topk_network" not in spec._flat_fields()


def test_flat_selection_rejects_a_payload_alias_before_offering_the_network():
    x = torch.empty((7, 62), dtype=torch.int32)
    storage = torch.zeros(80, dtype=torch.int32)
    payload = storage[:62]
    out = storage[:56].view(7, 2, 4)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(explicit_direct_selection, (x, payload, out, "direct"))
        spec = bound.config_spec
        assert not any(
            g.key == "cute_fragment_topk_network" for g in spec.compiler_coverage_groups
        )
        with pytest.raises(exc.InvalidConfig, match="cute_fragment_topk_network"):
            spec.normalized_config(
                dict(spec.default_config())
                | {"pid_type": "flat", "cute_fragment_topk_network": True}
            )


@helion.kernel(backend="cute", static_shapes=True)
def _standalone_nested_inplace(x: torch.Tensor) -> torch.Tensor:
    for _row in hl.tile(1, block_size=1):
        values = torch.topk(x[:], 8).values
        chosen = torch.topk(values, 4).values
        x[hl.arange(4)] = chosen
    return x


def test_standalone_nested_topk_retains_its_inplace_network():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            _standalone_nested_inplace, (torch.empty(32, dtype=torch.int32),)
        )
        group = next(
            g
            for g in bound.config_spec.compiler_coverage_groups
            if g.key == "cute_fragment_topk_network"
        )
        cfg = helion.Config.from_dict(
            group.witnesses[0].carrier.config | {group.key: True}
        )
        assert "fragment_topk_keys" in bound.to_code(cfg)


@pytest.mark.parametrize("reduction_tile", [None, 32])
def test_flat_serial_selection_with_a_native_iota_reduction(reduction_tile):
    x = torch.empty((7, 62), dtype=torch.int32)
    payload = torch.empty(62, dtype=torch.int32)
    out = torch.empty((7, 2, 4), dtype=torch.int32)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(explicit_direct_selection, (x, payload, out, "reduce"))
        config = routed_phase_config(32, warp=False)
        config.config.pop("cross_loop_pipeline")
        config.config["pid_type"] = "flat"
        config.config["reduction_loops"] = [reduction_tile]
        stage = stage_sources(bound.to_code(config))[-1]
        assert "fragment_topk_candidate" in stage and "fragment_topk_keys" not in stage


def test_flat_topk_witness_owns_a_nested_consumer_loop():
    x = torch.empty((7, 62), dtype=torch.int32)
    payload = torch.empty(62, dtype=torch.int32)
    out = torch.empty((7, 2, 4), dtype=torch.int32)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(explicit_direct_selection, (x, payload, out, "loop"))
        spec = bound.config_spec
        ir = bound.host_function.device_ir
        root = ir.root_ids[-1]
        assert len(_region_graph_ids(ir.graphs, root)) > 1
        assert root in spec.cute_fragment_topk_network_root_ids
        group = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_topk_network"
        )
        witness = next(w for w in group.witnesses if w.value is True)
        cfg = spec.normalized_config(witness.carrier.config | {group.key: True})
        assert cfg.pid_type == "flat"
        assert "fragment_topk_keys" in stage_sources(bound.to_code(cfg))[-1]
        with pytest.raises(exc.InvalidConfig, match=group.key):
            spec.normalized_config(dict(cfg) | {"pid_type": "persistent_blocked"})


def test_shared_row_slots_are_independent_of_aliases_and_k_extent() -> None:
    fingerprints = []
    # K=128 admits a warp reduction, K=127 does not; K=256 also admits
    # packet transport. The scalar parent still runs when output aliases w.
    with cpu_render_target(fake_device_sm_count=148):
        for k, aliased in ((128, False), (128, True), (127, False), (256, False)):
            x = torch.zeros(k, dtype=torch.bfloat16)
            w = torch.zeros(17, k, dtype=torch.bfloat16)
            out = w.flatten()[:34].view(torch.float32) if aliased else torch.zeros(17)
            bound = cpu_bind(row_sum_after_producer, (x, w, out, False))
            fingerprints.append(bound.config_spec.structural_fingerprint())
            if aliased:
                with pytest.raises(exc.InvalidConfig, match="row carry"):
                    bound.to_code(row_sum_phase_config())
        assert all(fields == fingerprints[0] for fields in fingerprints)
        assert {
            "cute_fragment_row_carry",
            "cute_fragment_reduction",
            "cute_fragment_unrolled_loops",
            "cute_fragment_vector_io",
        } <= {slot[0] for slot in fingerprints[0]}


@pytest.mark.parametrize("mode", ["readonly", "aliased", "recurrent"])
def test_row_carry_requires_a_readonly_phase_and_independent_terms(mode):
    x = torch.zeros(128, dtype=torch.bfloat16)
    w = torch.zeros(17, 128, dtype=torch.bfloat16)
    out = w.view(torch.float32).flatten()[:17] if mode == "aliased" else torch.zeros(17)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(row_sum_after_producer, (x, w, out, mode == "recurrent"))
        spec = bound.config_spec
        if mode != "readonly":
            # An alias changes admission; a recurrence changes the program.
            slots = {field[0] for field in spec.structural_fingerprint()}
            assert ("cute_fragment_compact_rows" in slots) == (mode == "aliased")
            assert not spec.cute_fragment_row_carry_root_ids
            with pytest.raises(exc.InvalidConfig, match="row carry"):
                bound.to_code(row_sum_phase_config())
            return
        assert spec.cute_fragment_row_carry_root_ids == {
            bound.host_function.device_ir.root_ids[-1]
        }
        group = next(
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_row_carry"
        )
        witness = next(w for w in group.witnesses if w.value is True)
        declared = helion.Config.from_dict(
            witness.carrier.config | {group.key: witness.value}
        )
        assert "fragment_row_carry = cutlass.Float32(0)" in bound.to_code(declared)
        seeds = [
            candidate
            for _, candidate in ConfigGeneration(spec).seed_flat_config_pairs()
            if candidate.config.get("cute_fragment_row_carry")
        ]
        assert seeds
        assert all(
            candidate.config.get("pid_type") == "persistent_interleaved"
            for candidate in seeds
        )
        assert seeds[0].config.get("cute_fragment_reduction") == "warp"
        assert "fragment_carry_columns" in bound.to_code(seeds[0])
        serial = next(
            cfg
            for cfg in seeds
            if cfg.config.get("cute_fragment_reduction", "serial") == "serial"
        )
        assert "fragment_row_carry = cutlass.Float32(0)" in bound.to_code(serial)

        # A partial warp can still run the serial carry. Strict normalization
        # must never label this schedule as a warp reduction that will not run.
        narrow = row_sum_phase_config().config | {
            "block_sizes": [16, 32, 8],
            "cute_fragment_reduction": "warp",
        }
        with pytest.raises(exc.InvalidConfig, match="warp row carry needs"):
            spec.normalize(dict(narrow))
        spec.normalize(narrow, _fix_invalid=True)
        assert narrow.get("cute_fragment_row_carry")
        assert "cute_fragment_reduction" not in narrow
        spec.normalize(narrow)


def test_unrelated_nested_copy_reuses_its_dynamic_binding() -> None:
    # input_tensor_metadata would specialize both dimensions and strides
    # even though there is no matrix carry followed by a row sum.
    op = helion.kernel(backend="cute", static_shapes=False)(nested_row_copy.fn)
    with cpu_target():
        first = torch.empty((64, 96))
        second = torch.empty((80, 112))
        one = op.bind((first, torch.empty_like(first)))
        two = op.bind((second, torch.empty_like(second)))
        assert not one.config_spec.cute_fragment_row_carry_root_ids
        assert one is two


@pytest.mark.slow
def test_qwen_routed_forward_declared_row_carry_witness() -> None:
    from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import (
        _helion_aot_qwen3_30b_a3b_moe_cuda_sm100 as aot,
    )
    from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import (
        qwen3_30b_a3b_moe as model,
    )

    # Fake CPU operands preserve the original dense strides without allocating
    # the two large expert banks or requiring an Inductor meta scheduler.
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = (
            tuple(
                torch.empty(shape, dtype=dtype)
                for shape, dtype in aot._TENSOR_SIGNATURES
            )
            + aot._STATIC_ARGS
        )
    op = helion.kernel(
        backend="cute", static_shapes=model.qwen3_30b_a3b_moe.settings.static_shapes
    )(model.qwen3_30b_a3b_moe.fn)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(op, args)
        group = next(
            g
            for g in bound.config_spec.compiler_coverage_groups
            if g.key == "cute_fragment_row_carry"
        )
        witness = next(w for w in group.witnesses if w.value is True)
        declared = helion.Config.from_dict(
            witness.carrier.config | {group.key: witness.value}
        )
        code = bound.to_code(declared)
        assert "fragment_row_carry = cutlass.Float32(0)" in code
        assert code.count("_launcher(") == 1
        # The search benchmarks flattened seeds. Flattening an omitted K-axis
        # thread count used to turn three valid raw candidates into impossible
        # two-dimensional launches, including the unrolled packet candidate.
        spec = bound.config_spec
        contract_ids = {axis for axis, _ in spec.cute_fragment_row_carry_contractions}
        thread_ids = list(spec.num_threads.valid_block_ids())
        candidates = [
            cfg
            for _, cfg in ConfigGeneration(spec).seed_flat_config_pairs()
            if cfg.config.get("cute_fragment_row_carry")
        ]
        assert candidates
        for cfg in candidates:
            assert all(
                cfg.config["num_threads"][thread_ids.index(k)] == 1
                for k in contract_ids
            )
        packet = next(
            cfg
            for cfg in candidates
            if cfg.config.get("cute_fragment_vector_io")
            and cfg.config.get("cute_fragment_unrolled_loops")
        )
        packet_code = bound.to_code(packet)
        assert "fragment_carry_packet = cute.make_rmem_tensor" in packet_code
        assert packet.config.get("cute_fragment_unrolled_loops")
        assert "fragment_carry_section in cutlass.range_constexpr(8)" in packet_code
        assert "cutlass.const_expr(norm_weight.iterator.alignment >= 16)" in packet_code
        assert "fragment_carry_pointer.toint()" in packet_code
        assert "fragment_carry_scalar in cutlass.range_constexpr" in packet_code

        # A K tile of 512 cannot cover the 768-wide down projection. The
        # router has an independent warp owner; the packet carry does not.
        partial = dict(packet.config)
        axes = list(bound.config_spec.block_sizes.valid_block_ids())
        blocks = list(partial["block_sizes"])
        for axis, length in bound.config_spec.cute_fragment_row_carry_contractions:
            if length == 768:
                blocks[axes.index(axis)] = 512
        partial["block_sizes"] = blocks
        bound.config_spec.normalize(partial, _fix_invalid=True)
        assert partial.get("cute_fragment_reduction") == "warp"
        assert not partial.get("cute_fragment_row_carry")
        assert not partial.get("cute_fragment_vector_io")
        assert not partial.get("cute_fragment_unrolled_loops")
        bound.config_spec.normalize(partial)

        # A valid warp carry with a half-width K tile retains its warp and
        # unroll, but cannot fetch one full 16-byte packet per lane.
        mismatch = dict(packet.config)
        blocks = list(mismatch["block_sizes"])
        for axis in contract_ids:
            blocks[axes.index(axis)] = 128
        mismatch["block_sizes"] = blocks
        with pytest.raises(exc.InvalidConfig, match="one complete packet"):
            spec.normalize(dict(mismatch))
        spec.normalize(mismatch, _fix_invalid=True)
        assert mismatch.get("cute_fragment_row_carry") and mismatch.get(
            "cute_fragment_unrolled_loops"
        )
        assert mismatch.get("cute_fragment_reduction") == "warp"
        assert not mismatch.get("cute_fragment_vector_io")
        spec.normalize(mismatch)

        # A whole K tile may still have too few columns for a warp or too
        # many K sections to unroll. Another fragment can keep its warp
        # reduction, but only a short warp K loop owns this unroll flag.
        for k_tile in (16, 32):
            narrow = dict(packet.config)
            narrow.pop("cute_fragment_vector_io", None)
            blocks = list(narrow["block_sizes"])
            for axis, _ in bound.config_spec.cute_fragment_row_carry_contractions:
                blocks[axes.index(axis)] = k_tile
            narrow["block_sizes"] = blocks
            with pytest.raises(exc.InvalidConfig, match="unrolling needs a short warp"):
                bound.config_spec.normalize(dict(narrow))
            bound.config_spec.normalize(narrow, _fix_invalid=True)
            assert narrow.get("cute_fragment_reduction") == "warp"
            assert narrow.get("cute_fragment_row_carry")
            assert not narrow.get("cute_fragment_unrolled_loops")
            bound.config_spec.normalize(narrow)


def test_carry_packets_require_guarded_strides_for_each_input() -> None:
    # The Inductor binding can use contiguous placeholders for a graph
    # that admits strided weights. The same constant graph strides may be
    # used for packetization when the graph actually guards them.
    x = torch.empty(512, dtype=torch.bfloat16)
    w = torch.empty((17, 512), dtype=torch.bfloat16)
    out = torch.empty(17)
    cfg = helion.Config.from_dict(
        row_sum_phase_config().config
        | {
            "block_sizes": [256, 32, 8],
            "cute_fragment_reduction": "warp",
            "cute_fragment_vector_io": True,
        }
    )

    def packet_sources(code: str) -> set[str]:
        return {
            ast.unparse(node.value.left)
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("fragment_carry_pointer")
            and isinstance(node.value, ast.BinOp)
        }

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(row_sum_after_producer, (x, w, out, False))
        with bound.env.without_dispatch_guards():
            assert "w.iterator" in packet_sources(bound.to_code(cfg))
        with bound.env.without_dispatch_guards(
            symbolic_strides=tensor_input_sources({"w": w}, [w])
        ):
            assert "w.iterator" not in packet_sources(bound.to_code(cfg))


@pytest.mark.parametrize("unroll", [False, True])
def test_warp_row_dot_guards_both_input_pointers_before_replacing_the_carry(
    unroll: bool,
) -> None:
    rows, k = 64, 768
    args = (
        torch.empty(k, dtype=torch.bfloat16),
        torch.empty(3 * rows, k, dtype=torch.bfloat16),
        torch.empty(1, dtype=torch.int32),
        torch.empty(k, dtype=torch.bfloat16),
        torch.empty(rows),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(selected_rows_after_producer, args)
        config = selected_rows_config(unroll=unroll)
        source = bound.to_code(config)
    assert source.count("_launcher(") == 1
    tree = ast.parse(source)
    mma = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_cute_mma_bf16"
    ]
    assert mma, "the row dot silently fell back to the scalar carry"
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and ast.unparse(node.test.func) == "cutlass.const_expr"
        and len(node.test.args) == 1
        and isinstance(condition := node.test.args[0], ast.BoolOp)
        and isinstance(condition.op, ast.And)
        and {ast.unparse(term) for term in condition.values}
        == {"weight.iterator.alignment >= 16", "published.iterator.alignment >= 16"}
    ]
    assert guards
    # The compiler can duplicate a phase's dispatch body. Every actual MMA
    # must be dominated by both alignment checks and have a non-MMA fallback.
    guarded_calls = set()
    for guard in guards:
        guarded_calls.update(n for stmt in guard.body for n in ast.walk(stmt))
        assert any(
            isinstance(n, ast.For)
            and ast.unparse(n.iter)
            == ("cutlass.range_constexpr(3)" if unroll else "range(0, 768, 256)")
            for stmt in guard.body
            for n in ast.walk(stmt)
        )
        (packets,) = [
            n
            for stmt in guard.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.For)
            and ast.unparse(n.iter) == "cutlass.range_constexpr(8)"
        ]
        # Each matrix half and the broadcast vector resolve their real base
        # and mask before entering the packet loop; the loop only adds K.
        assert all(
            not isinstance(node, ast.Attribute) or node.attr != "iterator"
            for node in ast.walk(packets)
        )
        fallback = ast.unparse(ast.Module(body=guard.orelse, type_ignores=[]))
        assert "fragment_carry_columns" in fallback
        assert "_cute_mma_bf16(" not in fallback
    assert set(mma) <= guarded_calls


def test_row_carry_repairs_partial_tiles_after_block_size_normalization() -> None:
    x = torch.empty(127, dtype=torch.bfloat16)
    w = torch.empty(17, 127, dtype=torch.bfloat16)
    out = torch.empty(17)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(row_sum_after_producer, (x, w, out, False))
        spec = bound.config_spec
        assert spec.cute_fragment_row_carry_root_ids
        for blocks in ([32, 32, 8], [[32, 32], 8]):
            config = row_sum_phase_config().config | {"block_sizes": blocks}
            with pytest.raises(exc.InvalidConfig, match="whole contraction tiles"):
                spec.normalize(dict(config))
            expected = {
                k: v for k, v in config.items() if k != "cute_fragment_row_carry"
            }
            spec.normalize(expected, _fix_invalid=True)
            spec.normalize(config, _fix_invalid=True)
            assert config == expected
            repaired = row_sum_phase_config().config | {
                "block_sizes": blocks,
                "cute_fragment_reduction": "warp",
            }
            spec.normalize(repaired, _fix_invalid=True)
            assert "cute_fragment_row_carry" not in repaired
            assert "cute_fragment_reduction" not in repaired
            spec.normalize(repaired)


def test_independent_shared_roots_do_not_advertise_row_carry() -> None:
    args = (
        torch.empty(128, dtype=torch.bfloat16),
        torch.empty(17, 128, dtype=torch.bfloat16),
        torch.empty(61),
    )
    with cpu_render_target(fake_device_sm_count=148):
        independent = cpu_bind(row_sum_after_independent_store, args)
        spec = independent.config_spec
        assert independent.host_function is not None
        assert len(independent.host_function.device_ir.root_ids) == 2
        assert spec.cross_loop_pipeline is None
        assert not spec.cute_fragment_row_carry_root_ids
        assert all(
            g.key != "cute_fragment_row_carry" for g in spec.compiler_coverage_groups
        )
        assert not any(
            c.config.get("cute_fragment_row_carry") for c in spec.compiler_seed_configs
        )
        # A real local phase orders the two roots. Common normalization owns
        # repairing or rejecting unsupported barrier enum values.
        dependent = cpu_bind(
            row_sum_after_producer, (*args[:2], torch.empty(17), False)
        )
        config = row_sum_phase_config().config | {"cross_loop_pipeline": "static"}
        with pytest.raises(
            exc.InvalidConfig, match="cross_loop_pipeline must be one of"
        ):
            dependent.config_spec.normalize(dict(config))
        dependent.config_spec.normalize(config, _fix_invalid=True)
        assert config.get("cute_fragment_row_carry") is True
        assert config.get("cross_loop_pipeline") == "barrier"


def test_checked_row_carry_uses_its_original_k_domain() -> None:
    rows, k = 8, 16
    x = (torch.arange(k) % 7 - 3).float()
    w = (torch.arange(rows * k).reshape(rows, k) % 11 - 5).float()
    w[:, 0] = 1
    a, b = torch.empty(rows), torch.empty(rows)
    expected = w @ x
    predicates: list[bool] = []

    def observe(values: tuple[int], **kwargs: object) -> None:
        predicates.append(bool(values[0]))

    for carry in (False, True):
        cfg = helion.Config(block_sizes=[2, 4], cute_fragment_row_carry=carry)
        with cpu_render_target(fake_device_sm_count=148):
            bound = cpu_bind(row_sum_two_outputs, (x, w, a, b, True))
            code = bound.to_code(cfg, output_origin_lines=False)
            # A real K tile of w still exceeds the last dimension of short x.
            short = cpu_bind(row_sum_two_outputs, (x[:-1], w, a, b, True))
            with pytest.raises(
                exc.BackendUnsupported, match="unmasked tile not proved"
            ):
                short.to_code(cfg)
        assert ("fragment_row_carry" in code) == carry
        assert code.count("call __assertfail,") == 1
        args = {"w": w, "x": x}
        outputs = {"a": a, "b": b}
        a.fill_(-313)
        b.fill_(211)
        predicates.clear()

        simulate_independent_fragment(code, args, outputs, 2, inline_asm=observe)
        assert predicates == [True, True]
        torch.testing.assert_close(a, expected, rtol=0, atol=0)
        torch.testing.assert_close(b, 2 * expected, rtol=0, atol=0)

        # The same compiled code must still observe a change to the checked
        # weight on the second CTA; the CPU model records the device predicate.
        w[4, 0] = -1
        predicates.clear()
        simulate_independent_fragment(code, args, outputs, 2, inline_asm=observe)
        assert predicates == [True, False]
        w[4, 0] = 1


def test_row_carry_rejects_overlapping_write_only_outputs() -> None:
    x = torch.empty(128, dtype=torch.bfloat16)
    w = torch.empty(17, 128, dtype=torch.bfloat16)
    storage = torch.empty(18)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(row_sum_two_outputs, (x, w, storage[:17], storage[1:], False))
        spec = bound.config_spec
        assert spec.cute_fragment_row_carry_root_ids
        seed = next(
            c
            for c in spec.compiler_seed_configs
            if c.config.get("cute_fragment_row_carry")
        )
        with pytest.raises(exc.InvalidConfig, match="disjoint output storage"):
            bound.to_code(seed)


def test_checked_carry_cannot_rebind_an_active_bank_tile() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def reused(x, w, out, checked: hl.constexpr, which: hl.constexpr):
        banks, rows, k = w.shape
        block = hl.register_block_size(k)
        for bank, row in hl.tile([banks, rows], block_size=[block, None]):
            saved = bank.begin
            if checked:
                torch._assert_async(w[saved, row.begin, 0] > 0)
            acc = hl.zeros([row, block], dtype=torch.float32)
            for col in hl.tile(k, block_size=block):
                vector = x[bank] if which == "tile" else x[col]
                weight = w[saved if which == "origin" else 0, row, col]
                acc = acc + weight * vector[None, :]
            out[saved, row] = torch.sum(acc, dim=-1)

    w = torch.arange(1, 16 * 4 * 16 + 1, dtype=torch.float32).view(16, 4, 16)
    args = (torch.ones(16), w, torch.empty(16, 4))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(reused, (*args, True, "origin"))
        plain = cpu_bind(reused, (*args, False, "origin"))
        assert bound.config_spec.cute_fragment_row_carry_root_ids
        for enabled in (False, True):
            cfg = helion.Config(block_sizes=[2, 4], cute_fragment_row_carry=enabled)
            with pytest.raises(exc.BackendUnsupported, match="reuses an active tile"):
                bound.to_code(cfg)
            with pytest.raises(exc.NestedDeviceLoopsConflict):
                plain.to_code(cfg)

        # Even where the old experimental fold happened to work (fixed
        # bank, complete outer tile), sharing the ID is unsupported. Its
        # block-size symbol cannot distinguish x[bank] from x[col], and
        # a partial outer bank can instead mask the final K reduction.
        for banks, which in ((16, "fixed"), (16, "tile"), (15, "fixed")):
            x, w, out = args
            other = cpu_bind(reused, (x, w[:banks], out[:banks], False, which))
            assert other.config_spec.cute_fragment_row_carry_root_ids
            for enabled in (False, True):
                cfg = helion.Config(block_sizes=[2, 4], cute_fragment_row_carry=enabled)
                with pytest.raises(exc.NestedDeviceLoopsConflict):
                    other.to_code(cfg)
