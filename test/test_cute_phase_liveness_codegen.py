"""Full static axes survive projection into an ordered consumer phase."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.materialized_fission_codegen import _region_graph_ids
from helion._compiler.cute.materialized_fission_codegen import _stage_host
from helion._compiler.cute.ordered_phases import _ordered_phase_bundle
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_phase_liveness import explicit_barrier_row_topk
from helion._testing.cute_phase_liveness import explicit_direct_selection
from helion._testing.cute_phase_liveness import grouped_selection_after_producer
from helion._testing.cute_phase_liveness import padded_factory_phases
from helion._testing.cute_phase_liveness import routed_after_producer
from helion._testing.cute_phase_liveness import routed_phase_config
from helion._testing.cute_phase_liveness import slice_sum_phases
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
