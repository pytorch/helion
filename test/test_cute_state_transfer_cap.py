from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib
import inspect
import itertools
import json
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from test._cute_aux import _cpu_codegen
from test._cute_aux import _rank_two_aux
from test._cute_aux import _rank_two_inputs

import helion
from helion._compiler.autotuner_heuristics.cute import CuteChunkPrefillHeuristic
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_prefill_prepared_state_abi as abi
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_SCHEDULE_KEY as SCHEDULE
from helion.autotuner.config_spec import CUTE_CHUNK_PREFILL_TASK_ORDER_KEY as ORDER
from helion.autotuner.config_spec import CUTE_STATE_TRANSFER_MAX_BITS_KEY as WIDTH
from helion.autotuner.config_spec import CUTE_STATE_TRANSFER_TRANSPORT_KEY as TRANSPORT
from helion.exc import InvalidConfig

pytest.importorskip("cutlass.cute")


@contextmanager
def _cpu():
    with (
        _cpu_codegen(),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
    ):
        yield


def _inputs():
    q = torch.empty((1, 64, 8, 128), dtype=torch.bfloat16)
    state = torch.empty((2, 8, 128, 128), dtype=torch.float32)
    return (
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty((1, 64, 8), dtype=torch.bfloat16),
        torch.empty((8,), dtype=torch.float32),
        torch.empty((8, 128), dtype=torch.float32),
        state,
        torch.empty_like(q),
        torch.empty_like(state),
        torch.tensor([0, 32, 64], dtype=torch.int64),
        128**-0.5,
        -5 * 1.4426950408889634,
    )


def _plan(source):
    (plans,) = [
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_helion_cute_wrapper_plans"
            for target in node.targets
        )
    ]
    (plan,) = plans
    return plan


@pytest.fixture(scope="module", params=(16, 32))
def captured(request):
    seen = []
    original = abi.bind_external_state_abi

    def observe(region, root, **kwargs):
        program = original(region, root, **kwargs)
        seen.append((region, root, program))
        return program

    kernel = (
        kda_prefill_native_math if request.param == 16 else kda_prefill_native_math_bt32
    )
    with _cpu(), patch.object(abi, "bind_external_state_abi", observe):
        bound = kernel._bind_isolated(_inputs())
    # Registration itself must have called the real binder, not a shape marker.
    assert seen and bound.config_spec.cute_state_transfer_max_bits is not None
    region, root, program = seen[0]
    return bound, region, root, program


def test_actual_old_seed_cross_product_and_search_roundtrip(captured):
    bound, region, root, program = captured
    spec = bound.config_spec
    with _cpu(), bound.env, bound.host_function:
        orders = spec.cute_chunk_prefill_task_order.choices
        schedules = spec.cute_chunk_prefill_schedule.choices
        old_seeds = [
            {ORDER: order, SCHEDULE: schedule}
            for schedule in schedules
            for order in orders
        ]
        seeds = CuteChunkPrefillHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        register_seeds = [
            {**seed, WIDTH: width} for width in (256, 128) for seed in old_seeds
        ]
        old_prefix = register_seeds.copy()
        transports = (
            ("register", "tma", "tma_pipelined", "tma_planar")
            if region.chunk_size == 32
            else ("register",)
        )
        if region.chunk_size == 32:
            old_prefix += [
                {**seed, TRANSPORT: transport}
                for transport in ("tma", "tma_pipelined")
                for seed in register_seeds
            ]
        expected = old_prefix.copy()
        if region.chunk_size == 32:
            expected += [{**seed, TRANSPORT: "tma_planar"} for seed in register_seeds]
        assert len(old_prefix) == 18
        assert [dict(seed) for seed in seeds[: len(old_prefix)]] == old_prefix
        assert [dict(seed) for seed in seeds] == expected
        assert len(seeds) == (18 if region.chunk_size == 16 else 24)
        fields = ["block_sizes", ORDER, SCHEDULE, WIDTH]
        if region.chunk_size == 32:
            fields.append(TRANSPORT)
        assert list(spec._flat_fields()) == fields
        generation = spec.create_config_generation()
        assert [fragment.cardinality() for fragment in generation.flat_spec] == [
            1,
            len(orders),
            len(schedules),
            2,
        ] + ([4] if region.chunk_size == 32 else [])
        assert generation.default_flat()[3] == 256
        if region.chunk_size == 32:
            assert generation.default_flat()[4] == "register"
        assert TRANSPORT not in spec.default_config()
        # Enumerate every old effective schedule/order combination at both caps.
        for seed in seeds:
            config = spec.normalized_config(
                helion.Config.from_dict({"block_sizes": [64], **seed})
            )
            flat = generation.flatten(config)
            assert generation.unflatten(flat) == config
            assert (
                helion.Config.from_dict(json.loads(json.dumps(dict(config)))) == config
            )
            width_neighbors = [
                neighbor
                for neighbor in generation.coordinate_neighbor_projections(flat)
                if neighbor.key == WIDTH
            ]
            assert len(width_neighbors) == 1
            assert width_neighbors[0].to_value == (128 if config[WIDTH] == 256 else 256)
        population = generation.random_population_flat(32, user_seed_configs=seeds)
        actual = {
            (
                config[ORDER],
                config[SCHEDULE],
                config[WIDTH],
                config.cute_state_transfer_transport,
            )
            for config in map(generation.unflatten, population)
        }
        assert actual == set(
            itertools.product(orders, schedules, (256, 128), transports)
        )
        for flat in population:
            assert generation.flatten(generation.unflatten(flat)) == flat
        mutated = generation.differential_mutation(*population[:4], crossover_rate=0.5)
        assert generation.unflatten(mutated)[WIDTH] in (256, 128)
        assert generation.unflatten(mutated).cute_state_transfer_transport in transports


@pytest.mark.parametrize("width", (256, 128))
def test_actual_public_program_all_schedules(captured, width):
    bound, region, root, program = captured
    with _cpu():
        for schedule in bound.config_spec.cute_chunk_prefill_schedule.choices:
            base = {"block_sizes": [64], SCHEDULE: schedule}
            old_source = bound.to_code(helion.Config.from_dict(base))
            source = bound.to_code(helion.Config.from_dict({**base, WIDTH: width}))
            old = _plan(old_source)
            new = _plan(source)
            assert old["prepared_state_abi_program"] == program
            if width == 256:
                assert source == old_source
            changed = new.pop("prepared_state_abi_program")
            old_program = old.pop("prepared_state_abi_program")
            assert old == new
            assert changed == abi.bind_external_state_abi(region, root, max_bits=width)
            external = 0
            for old_cycle, new_cycle in zip(old_program, changed, strict=True):
                for before, after in zip(old_cycle, new_cycle, strict=True):
                    opcode, mode = before[:2]
                    is_external = (opcode == 0 and mode != 1) or (
                        opcode == 1 and mode != 0
                    )
                    assert len(before) == 7
                    assert (
                        after == (*before, 128)
                        if width == 128 and is_external
                        else after == before
                    )
                    external += is_external
            assert external == 16  # init4 + final4 + zero-trip read4/store4


@pytest.mark.parametrize("invalid", (64, 512, 128.0, "128", True))
def test_invalid_cap_rejected_and_fix_invalid_is_default(captured, invalid):
    bound, region, root, program = captured
    with _cpu(), bound.env, bound.host_function:
        config = helion.Config.from_dict(
            json.loads(json.dumps({"block_sizes": [64], WIDTH: invalid}))
        )
        with pytest.raises(InvalidConfig, match="must be one of"):
            bound.config_spec.normalized_config(config)
        fixed = helion.Config.from_dict(dict(config))
        bound.config_spec.normalize(fixed, _fix_invalid=True)
        assert fixed[WIDTH] == 256
        with pytest.raises(chain._UnsupportedChain, match="transfer cap"):
            abi.bind_external_state_abi(region, root, max_bits=invalid)


@pytest.mark.parametrize("mutation", ("zero_trip", "duplicate_store"))
def test_external_cap_does_not_bypass_original_state_owner(captured, mutation):
    bound, region, root, program = captured
    changed = root.copy()
    if mutation == "zero_trip":
        phi = next(node for node in changed.graph.nodes if node.target is abi._phi)
        phi.args = (phi.args[1], phi.args[1])
    else:
        store = next(
            node
            for node in changed.graph.nodes
            if abi._same_ref(abi._store_ref(node), region.final_state)
        )
        with changed.graph.inserting_after(store):
            changed.graph.call_function(store.target, store.args, store.kwargs)
    for width in (256, 128):
        with pytest.raises(chain._UnsupportedChain, match="original"):
            abi.bind_external_state_abi(region, changed, max_bits=width)


@pytest.mark.parametrize("width", (256, 128))
def test_unrelated_route_does_not_register_or_accept_cap(width):
    with _cpu():
        bound = _rank_two_aux._bind_isolated(_rank_two_inputs())
        spec = bound.config_spec
        assert spec.cute_state_transfer_max_bits is None
        assert WIDTH not in spec._flat_fields()
        config = helion.Config.from_dict({"block_sizes": [64, 64, 64], WIDTH: width})
        with pytest.raises(InvalidConfig, match="bound external-state ABI"):
            spec.normalized_config(config)
        spec.normalize(config, _fix_invalid=True)
        assert WIDTH not in config


@pytest.mark.parametrize("width", (256, 128))
def test_real_executor_forwards_only_external_caps(captured, width):
    bound, region, root, legacy = captured
    module = importlib.import_module("helion._compiler.cute.prepared_tcgen_edge")
    fn = next(
        node
        for node in ast.parse(inspect.getsource(module)).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_prepared_state_abi"
    )
    fn.decorator_list = []
    calls = []

    def record(name, result=None):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return call

    values = list(range(32))
    namespace = {
        "cutlass": SimpleNamespace(
            Constexpr=object, range_constexpr=range, const_expr=bool
        ),
        "cast": lambda type_name, value: value,
        "execute_prepared_read": record("tmem_read", (values, None)),
        "execute_prepared_store": record("tmem_store"),
        "execute_prepared_store_completion": record("completion"),
        "read_linear_state_abi": record("external_read", values),
        "store_linear_state_abi": record("external_store"),
    }
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            "<actual-state-abi-executor>",
            "exec",
        ),
        namespace,
    )
    for cycle in abi.bind_external_state_abi(region, root, max_bits=width):
        namespace[fn.name](cycle, "initial", "final", 4096, (2, 3, 4), False)
    external = [call for call in calls if call[0].startswith("external")]
    assert len(external) == 16
    assert all(
        kwargs == {"MAX_BITS": width} and args[-1] is False
        for name, args, kwargs in external
    )
    assert all(
        not kwargs for name, args, kwargs in calls if not name.startswith("external")
    )
    # The separate BT16 initialization must decode the same instruction field.
    direct = importlib.import_module("helion._compiler.cute.chunk_prefill_tmem")
    tree = ast.parse(inspect.getsource(direct))
    (call,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_linear_state_abi"
    ]
    (keyword,) = call.keywords
    assert keyword.arg == "MAX_BITS"
    expression = compile(ast.Expression(keyword.value), "<actual-bt16-decoder>", "eval")
    for read in abi.bind_external_state_abi(region, root, max_bits=width)[0][:-1:2]:
        assert eval(expression, {"read": read}) == width
