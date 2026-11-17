from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

import helion
from helion._compiler.autotuner_heuristics import triton as triton_heuristics
from helion._compiler.roll_reduction import EffectfulReductionReplay
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.inline_asm_kernels import overwrite_after_reduction
from helion._testing.inline_asm_kernels import overwritten_reduction_input
from helion._testing.inline_asm_kernels import read_after_reduction_effect
from helion._testing.inline_asm_kernels import store_between_reductions
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def _unused_asm(x: torch.Tensor, is_pure: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        value = x[tile]
        hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=r,r", [value], torch.int32, is_pure, 1
        )
        out[tile] = value
    return out


def _asm_calls(kernel, *extra_args):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(32, dtype=torch.int32), *extra_args))
        source = bound.to_code(helion.Config(block_sizes=[32]))
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in ("tl.inline_asm_elementwise", "_cute_inline_asm_elementwise")
    ]


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("is_pure", (False, True))
def test_unused_asm_respects_declared_effects(is_pure):
    # The declaration controls effects, even for identical instruction text.
    # Inspect AST calls so a retained source comment cannot make this pass.
    assert len(_asm_calls(_unused_asm, is_pure)) == (0 if is_pure else 1)


@helion.kernel(autotune_effort="none")
def _multi_output_asm(
    x: torch.Tensor, is_pure: hl.constexpr, use_outputs: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        value = x[tile]
        first, second = hl.inline_asm_elementwise(
            "mov.b32 $0, $2; mov.b32 $1, $2;",
            "=r,=r,r",
            (value,),
            (torch.int32, torch.int32),
            is_pure,
            1,
        )
        out[tile] = first + second if use_outputs else value
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("is_pure", (False, True))
@pytest.mark.parametrize("use_outputs", (False, True))
def test_multi_output_asm_executes_once_unless_pure_and_unused(is_pure, use_outputs):
    assert len(_asm_calls(_multi_output_asm, is_pure, use_outputs)) == (
        0 if is_pure and not use_outputs else 1
    )


@helion.kernel(autotune_effort="none")
def _single_tuple_asm(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        (value,) = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=r,r", [x[tile]], (torch.int32,), True, 1
        )
        out[tile] = value
    return out


@onlyBackends(["triton", "cute"])
def test_single_tuple_asm_retains_tuple_result_type():
    (call,) = _asm_calls(_single_tuple_asm)
    dtype = (
        call.args[3]
        if ast.unparse(call.func) == "tl.inline_asm_elementwise"
        else next(keyword.value for keyword in call.keywords if keyword.arg == "dtype")
    )
    assert isinstance(dtype, ast.Tuple) and len(dtype.elts) == 1


@helion.kernel(autotune_effort="none", static_shapes=True)
def _asm_shared_across_sweeps(x: torch.Tensor, is_pure: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        value = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, is_pure, 1
        )
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("is_pure", [False, True])
@pytest.mark.parametrize("width", [128, 4096])
def test_shared_sweep_producer_only_rolls_when_pure(is_pure, width, caplog):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_asm_shared_across_sweeps, (torch.empty(2, width), is_pure))
        assert len(bound.config_spec.reduction_loops.valid_block_ids()) == 1
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        if not is_pure and bound.env.backend_name == "triton":
            # A complete saved config from before effect-aware rolling. Its
            # indexing entries include the old duplicated producer sweep.
            config = helion.Config.from_dict(
                {
                    "atomic_indexing": [],
                    "block_sizes": [],
                    "indexing": ["pointer"] * 5,
                    "load_eviction_policies": [""] * 3,
                    "num_stages": 1,
                    "num_warps": 4,
                    "pid_type": "flat",
                    "range_flattens": [None],
                    "range_multi_buffers": [None],
                    "range_num_stages": [0],
                    "range_unroll_factors": [0],
                    "range_warp_specializes": [],
                    "reduction_loops": [32],
                }
            )
        normalized = bound._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == ([32] if is_pure else [None])
        assert ("persistent" in caplog.text) == (not is_pure)
        if not is_pure:
            assert bound.config_spec.default_config().config["reduction_loops"] == [
                None
            ]
            bound.to_code(normalized)
            # Retiring unsafe chunks does not forgive malformed old configs.
            for invalid in ([3], [32, 32]):
                config.config["reduction_loops"] = invalid
                with pytest.raises(helion.exc.InvalidConfig):
                    bound._normalized_config_copy(config)


@helion.kernel(autotune_effort="none")
def _loop_carried_reduction(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[1], block_size=1):
        value = x[0, row, :] * 0
        for k in hl.tile(x.shape[0], block_size=1):
            value = value + x[k, row, :].sum(0)
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@onlyBackends(["triton", "cute"])
def test_pure_loop_carried_reduction_keeps_rolling():
    with cpu_render_target():
        bound = cpu_bind(_loop_carried_reduction, (torch.empty(2, 2, 128),))
        assert len(bound.config_spec.reduction_loops.valid_block_ids()) == 1
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        assert bound._normalized_config_copy(config).config["reduction_loops"] == [32]
        bound.to_code(config)


@helion.kernel(autotune_effort="none")
def _nested_effect_reduction(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[1], block_size=1):
        value = x[0, row, :] * 0
        for k in hl.tile(x.shape[0], block_size=1):
            value = hl.inline_asm_elementwise(
                "mov.b32 $0, $1;",
                "=f,f",
                [value + x[k, row, :].sum(0)],
                torch.float32,
                False,
                1,
            )
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@onlyBackends(["triton", "cute"])
def test_nested_control_flow_keeps_opaque_effects_persistent(caplog):
    with cpu_render_target():
        bound = cpu_bind(_nested_effect_reduction, (torch.empty(2, 2, 128),))
        normalized = bound._normalized_config_copy(helion.Config(reduction_loops=[32]))
        assert normalized.config["reduction_loops"] == [None]
        assert "effectful producer" in caplog.text
        bound.to_code(normalized)


@helion.kernel(autotune_effort="none", static_shapes=True)
def _nested_replayed_child(x: torch.Tensor):
    out = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[0], block_size=1):
        value = x[row, :] * 0
        for _j in hl.tile(2, block_size=1):
            current = hl.inline_asm_elementwise(
                "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, False, 1
            )
            value = current - current.sum(-1, keepdim=True)
        out[row] = value.sum(-1)
    return out


@onlyBackends(["triton", "cute"])
def test_mixed_child_does_not_gain_a_persistent_config_slot():
    with cpu_render_target():
        bound = cpu_bind(_nested_replayed_child, (torch.empty(2, 128),))
        # Before effect-aware rolling this child needed two sweeps, so its
        # caller could not roll and there was no reduction_loops slot.
        assert not bound.config_spec.reduction_loops
        ast.parse(bound.to_code(helion.Config()))


@onlyBackends(["triton", "cute"])
def test_load_replay_cannot_cross_discarded_assembly(caplog):
    with cpu_render_target():
        bound = cpu_bind(
            overwritten_reduction_input,
            (
                torch.empty(2, 128, dtype=torch.int32),
                torch.empty(2, 128, dtype=torch.int64),
            ),
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        normalized = bound._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == [None]
        assert "persistent" in caplog.text
        bound.to_code(normalized)


@onlyBackends(["triton", "cute"])
def test_queued_reload_cannot_move_after_a_later_opaque_write():
    with cpu_render_target():
        bound = cpu_bind(
            overwrite_after_reduction,
            (torch.empty(2, 128, dtype=torch.int32), torch.empty(2, dtype=torch.int64)),
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        normalized = bound._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == [None]
        bound.to_code(normalized)


@helion.kernel(autotune_effort="none")
def _two_axes_with_one_opaque_producer(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    second = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        value = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, False, 1
        )
        second[row, :] = value - value.sum(-1, keepdim=True)
    # The other axis belongs to a separate graph. Overlapping axes have only
    # one legacy config slot, covered by the three-input regression below.
    for row in hl.tile(x.shape[0], block_size=1):
        values = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=f,f", [y[row, :]], torch.float32, True, 1
        )
        first[row] = values.sum(-1)
    return first, second


@helion.kernel(autotune_effort="none", static_shapes=True)
def _overlapping_axes_with_independent_tail(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, is_pure: hl.constexpr
):
    first = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    second = torch.empty_like(x)
    tail = torch.empty((z.shape[0],), dtype=z.dtype, device=z.device)
    for row in hl.tile(x.shape[0], block_size=1):
        first[row] = x[row, :].sum(-1) + y[row, :].sum(-1)
    for row in hl.tile(x.shape[0], block_size=1):
        value = hl.inline_asm_elementwise(
            "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, is_pure, 1
        )
        second[row, :] = value - value.sum(-1, keepdim=True)
    for row in hl.tile(z.shape[0], block_size=1):
        tail[row] = z[row, :].sum(-1)
    return first, second, tail


@onlyBackends(["triton"])
def test_persistent_placeholder_preserves_old_multi_axis_config():
    with cpu_render_target(fake_device_sm_count=148):
        args = (torch.empty(2, 256), torch.empty(2, 512), torch.empty(2, 1024))
        pure = cpu_bind(_overlapping_axes_with_independent_tail, (*args, True))
        assert [spec.size_hint for spec in pure.config_spec.reduction_loops] == [
            256,
            1024,
        ]
        saved = helion.Config.from_dict({"reduction_loops": [64, 128]})
        effectful = cpu_bind(_overlapping_axes_with_independent_tail, (*args, False))
        assert [spec.size_hint for spec in effectful.config_spec.reduction_loops] == [
            256,
            1024,
        ]
        normalized = effectful._normalized_config_copy(saved)
        assert normalized.reduction_loops == [None, 128]
        ast.parse(effectful.to_code(normalized))


@onlyBackends(["triton", "cute"])
def test_persistent_axis_does_not_prevent_another_axis_from_rolling():
    with cpu_render_target():
        bound = cpu_bind(
            _two_axes_with_one_opaque_producer,
            # Both roots use one warp's physical lanes on CuTe. The second
            # axis still needs three rolled chunks, independent of the first.
            (torch.empty(2, 32), torch.empty(2, 96)),
        )
        specs = bound.config_spec.reduction_loops
        assert len(specs) == 2
        assert specs[0].persistent_reason is not None
        assert specs[1].persistent_reason is None
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32, 32]
        normalized = bound._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == [None, 32]
        bound.to_code(normalized)


@onlyBackends(["triton"])
@pytest.mark.parametrize("primary_slot", [0, 1])
def test_reduction_seeds_preserve_persistent_slots(primary_slot):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            _two_axes_with_one_opaque_producer,
            (torch.empty(2, 128), torch.empty(2, 96)),
        )
        spec = bound.config_spec
        assert len(spec.reduction_loops) == 2
        assert spec.reduction_loops[0].persistent_reason is not None
        assert spec.reduction_loops[1].persistent_reason is None
        fact = spec.reduction_kernel_fact
        assert fact is not None
        primary = next(
            desc
            for desc in fact.reductions
            if desc.block_id == spec.reduction_loops[primary_slot].block_id
        )
        allocation = SimpleNamespace(
            block_sizes=list(spec.default_config().block_sizes),
            primary_r_block=32,
            rolled_loop_sizes={loop.block_id: 32 for loop in spec.reduction_loops},
            num_warps=4,
        )
        raw = []
        materialize = triton_heuristics._materialize_config

        def record_seed(config, **kwargs):
            raw.append(dict(config))
            return materialize(config, **kwargs)

        with (
            patch.object(
                triton_heuristics, "_materialize_config", side_effect=record_seed
            ),
            patch.object(
                triton_heuristics, "_primary_descriptor_selected", return_value=primary
            ),
            patch.object(
                triton_heuristics.TritonReductionHeuristic,
                "size_reduction_tiles_liveness",
                return_value=allocation,
            ),
        ):
            seed = triton_heuristics.TritonReductionHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
        assert seed is not None
        assert raw[0]["reduction_loops"] == [None, 32]
        assert seed.config["reduction_loops"] == [None, 32]
        normalized = bound._normalized_config_copy(seed)
        assert normalized.config["reduction_loops"] == [None, 32]


@onlyBackends(["triton"])
def test_reduction_resource_search_keeps_effectful_axis_full_width():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            _two_axes_with_one_opaque_producer,
            (torch.empty(2, 128), torch.empty(2, 96)),
        )
        spec = bound.config_spec
        fact = spec.reduction_kernel_fact
        assert fact is not None
        persistent = spec.reduction_loops[0]
        assert persistent.persistent_reason is not None
        primary = next(
            desc for desc in fact.reductions if desc.block_id == persistent.block_id
        )
        seen = []
        original = triton_heuristics.TritonReductionHeuristic._reduction_resources

        def record(env, candidate, *args):
            seen.append(candidate.reduction_width(persistent.block_id))
            return original(env, candidate, *args)

        with (
            bound.env,
            patch.object(
                triton_heuristics.TritonReductionHeuristic,
                "_reduction_resources",
                side_effect=record,
            ),
        ):
            allocation = triton_heuristics.TritonReductionHeuristic.size_reduction_tiles_liveness(
                bound.env, spec, primary
            )
        assert seen and all(width == 128 for width in seen)
        assert allocation.primary_r_block == 128


@helion.kernel(autotune_effort="none")
def _mixed_refusal_axes(x: torch.Tensor, y: torch.Tensor, effect_first: hl.constexpr):
    effect = torch.empty_like(x)
    selected = torch.empty((x.shape[0], 16), dtype=x.dtype, device=x.device)
    other = torch.empty((y.shape[0],), dtype=y.dtype, device=y.device)
    for row in hl.tile(x.shape[0], block_size=1):
        if effect_first:
            value = hl.inline_asm_elementwise(
                "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, False, 1
            )
            effect[row, :] = value - value.sum(-1, keepdim=True)
        else:
            values, indices = torch.topk(x[row, :], 16, dim=-1)
            selected[row, :] = values
    for row in hl.tile(x.shape[0], block_size=1):
        if effect_first:
            values, indices = torch.topk(x[row, :], 16, dim=-1)
            selected[row, :] = values
        else:
            value = hl.inline_asm_elementwise(
                "mov.b32 $0, $1;", "=f,f", [x[row, :]], torch.float32, False, 1
            )
            effect[row, :] = value - value.sum(-1, keepdim=True)
    for row in hl.tile(y.shape[0], block_size=1):
        other[row] = y[row, :].sum(-1)
    return effect, selected, other


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("effect_first", [False, True])
def test_plain_refusal_controls_slot_identity_in_either_graph_order(effect_first):
    with cpu_render_target():
        bound = cpu_bind(
            _mixed_refusal_axes,
            (torch.empty(2, 128), torch.empty(2, 96), effect_first),
        )
        specs = bound.config_spec.reduction_loops
        assert len(specs) == 1
        with bound.env:
            (other_axis,) = [
                block.block_id
                for block in bound.env.block_sizes
                if block.reduction and block.size_hint() == 96
            ]
        assert specs[0].block_id == other_axis
        assert specs[0].persistent_reason is None
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        normalized = bound._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == [32]
        ast.parse(bound.to_code(normalized))


@onlyBackends(["triton"])
def test_outer_load_stays_after_pending_reduction_effect():
    # The last load has no reduction axis: it must follow the rolled assembly
    # loop, even though its pointer has no data dependency on the assembly.
    with cpu_render_target():
        bound = cpu_bind(
            read_after_reduction_effect,
            (
                torch.empty(2, 128, dtype=torch.int32),
                torch.empty(2, 128, dtype=torch.int64),
            ),
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        source = bound.to_code(config)
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    (assembly,) = [
        node for node in calls if ast.unparse(node.func) == "tl.inline_asm_elementwise"
    ]
    loads = [node for node in calls if ast.unparse(node.func) == "tl.load"]
    assert len(loads) == 3
    assert max(node.lineno for node in loads) > assembly.lineno


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("target_kind", ["fresh_view", "input", "input_alias"])
def test_load_replay_distinguishes_fresh_storage_from_input_aliases(target_kind):
    x = torch.empty(3, 128)
    target = x[:, 0] if target_kind == "input_alias" else torch.empty(3)
    with cpu_render_target():
        bound = cpu_bind(
            store_between_reductions, (x, target, target_kind == "fresh_view")
        )
        config = bound.config_spec.default_config()
        config.config.update(reduction_loops=[32], reduction_fusion=True)
        normalized = bound._normalized_config_copy(config)
        # External storage may use the recorded runtime disjointness guard;
        # compiler-owned allocations need no such runtime predicate.
        assert normalized.config["reduction_loops"] == (
            [None] if target_kind == "input_alias" else [32]
        )
        source = bound.to_code(normalized)
        if target_kind == "fresh_view" and bound.env.backend_name == "triton":
            assert source.count("for roffset") == 2


@onlyBackends(["triton", "cute"])
def test_large_reduction_keeps_explicit_chunks_with_guarded_input_stores():
    kernel = helion.kernel(store_between_reductions.fn, autotune_effort="none")
    x = torch.empty(1, 2**21)
    target = torch.empty(1)
    with cpu_render_target():
        initial_alias = kernel.bind((x, x[:, 0], False))
        separate = kernel.bind((x, target, False))
        assert separate is not initial_alias
        assert separate.config_spec.reduction_loops[0].persistent_reason is None
        config = separate.config_spec.default_config()
        config.config["reduction_loops"] = [4096]
        normalized = separate._normalized_config_copy(config)
        assert normalized.config["reduction_loops"] == [4096]
        ast.parse(separate.to_code(normalized))
        assert "tensor_storage_disjoint_matrix_v1" in (
            separate.env.runtime_input_specializations
        )
        # Repeated views are distinct Python objects but share storage. The
        # storage predicate, not object identity or shape, must split binding.
        alias = kernel.bind((x, x[:, 0], False))
        assert alias is initial_alias
        assert alias is not separate
        assert alias.config_spec.reduction_loops[0].persistent_reason is not None
        assert kernel.bind((x, target, False)) is separate


@onlyBackends(["triton", "cute"])
def test_reduction_codegen_requires_recorded_storage_facts(monkeypatch):
    x, target = torch.empty(3, 128), torch.empty(3)
    with cpu_render_target():
        bound = cpu_bind(store_between_reductions, (x, target, False))
        config = bound.config_spec.default_config()
        config.config.update(reduction_loops=[32], reduction_fusion=True)
        config = bound._normalized_config_copy(config)
        assert config.reduction_loops == [32]
        facts = bound.env.bound_runtime_input_specialization_results
        key = "tensor_storage_disjoint_matrix_v1"
        assert key in facts
        with (
            bound.env,
            bound.host_function,
            bound._runtime_arg_values_for_codegen(),
        ):
            bound.host_function.device_ir.build_codegen_graphs(config)
            monkeypatch.delitem(facts, key)
            with pytest.raises(EffectfulReductionReplay):
                bound.host_function.device_ir.build_codegen_graphs(config)


@helion.kernel(autotune_effort="none")
def _distributed_asm(
    x: torch.Tensor, is_pure: hl.constexpr, group: hl.ProcessGroupName
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        stamp = hl.inline_asm_elementwise(
            "mov.u32 $0, %clock;", "=r", [], torch.int32, is_pure, 1
        )
        out[tile] = x[tile] + stamp
    hl.barrier()
    for tile in hl.tile(out.shape):
        out[tile] = out[tile] + 1
    return out


@onlyBackends(["triton"])
@pytest.mark.parametrize("is_pure", [False, True])
def test_distributed_roots_require_visible_assembly_effects(is_pure):
    # Bind with a real fake process group: no transports, collectives or CUDA.
    if dist.is_initialized():
        pytest.skip("needs exclusive control of the process group")
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
    try:
        group = dist.group.WORLD.group_name
        with cpu_render_target():
            args = (torch.empty(32, dtype=torch.int32), is_pure, group)
            if is_pure:
                bound = cpu_bind(_distributed_asm, args)
                assert bound.env.process_group_name == group
            else:
                with pytest.raises(
                    helion.exc.CrossLoopSchedulingError, match="may access another rank"
                ):
                    cpu_bind(_distributed_asm, args)
    finally:
        dist.destroy_process_group()
