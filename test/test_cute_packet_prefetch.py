from __future__ import annotations

import ast
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from examples.aot_example import vector_scale
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.autotuner_heuristics.cute import CutePointwiseVecHeuristic
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.full_tile_bounds import FullTilePlan
from helion._compiler.cute.full_tile_bounds import specialize_full_tile_bounds
from helion._compiler.cute.tensor_layout_relations import TensorLayoutRelation
from helion._compiler.device_function import DeviceFunction
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
import helion.language as hl


@pytest.fixture(autouse=True)
def _cpu_only():
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _bound(dtype=torch.bfloat16, *, length=8192, function=None, args=None):
    kernel = helion.kernel(
        vector_scale.fn if function is None else function,
        backend="cute",
        static_shapes=False,
        autotune_effort="full",
    )
    return _cpu_bind(
        kernel, (torch.empty(length, dtype=dtype), 2.0) if args is None else args
    )


def _config(batch=0, *, width=8, packets=2, bounds=True):
    return helion.Config(
        block_sizes=[256 * width * packets],
        num_threads=[256],
        cute_vector_widths=[width],
        cute_lane_layouts=["strided"],
        cute_proven_bounds=bounds,
        cute_packet_prefetch=batch,
    )


def _functions(source):
    return [
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    ]


def _dump(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]))


@skipUnlessBackends(["cute"])
def test_default_is_source_identical_and_does_not_collect_new_storage_facts():
    bound = _bound()
    explicit = _config()
    absent = helion.Config.from_dict(
        {
            key: value
            for key, value in explicit.config.items()
            if key != "cute_packet_prefetch"
        }
    )
    assert helion.Config().cute_packet_prefetch == 0
    with patch(
        "helion._compiler.cute.prefetch_pointwise_packets.prefetch_pointwise_packets",
        side_effect=AssertionError("disabled pass called"),
    ):
        assert bound.to_code(explicit) == bound.to_code(absent)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        assert generation.flatten(explicit) == generation.flatten(absent)
        assert generation.unflatten(generation.default_flat()).cute_packet_prefetch == 0


@pytest.mark.parametrize("batch", [2, 4, 8])
@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 8), (torch.float32, 4)])
@skipUnlessBackends(["cute"])
def test_ordinary_allocating_caller_abi_fallback_and_math_are_preserved(
    batch, dtype, width
):
    bound = _bound(dtype)
    off = bound.to_code(_config(width=width, packets=batch))
    on = bound.to_code(_config(batch, width=width, packets=batch))
    before = _functions(off)
    after = _functions(on)
    assert len(before) == len(after)
    assert ast.dump(before[-1]) == ast.dump(after[-1])
    assert ast.dump(before[-2].args) == ast.dump(after[-2].args)
    old_branch, new_branch = before[-2].body[0], after[-2].body[0]
    assert isinstance(old_branch, ast.If) and isinstance(new_branch, ast.If)
    assert ast.dump(old_branch.test) == ast.dump(new_branch.test)
    assert _dump(old_branch.orelse) == _dump(new_branch.orelse)
    assert _dump(old_branch.body) != _dump(new_branch.body)
    assert "_helion_prefetch" in ast.unparse(new_branch)
    for call in (
        node
        for node in ast.walk(old_branch)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).startswith("_cute_store_")
    ):
        assert any(ast.dump(call) == ast.dump(other) for other in ast.walk(new_branch))
    scale_bindings = {"scale"} | {
        target.id
        for node in ast.walk(old_branch)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and any(
            isinstance(arg, ast.Name) and arg.id == "scale" for arg in node.value.args
        )
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert scale_bindings
    floating_math = [
        ast.dump(node)
        for node in ast.walk(old_branch)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and any(
            isinstance(child, ast.Name) and child.id in scale_bindings
            for child in ast.walk(node)
        )
    ]
    assert floating_math
    assert all(
        expression in {ast.dump(node) for node in ast.walk(new_branch)}
        for expression in floating_math
    )


@skipUnlessBackends(["cute"])
def test_original_timed_shape_uses_real_fresh_storage_proof():
    bound = _bound(length=33554432)
    observed = []
    original = DeviceFunction.proven_disjoint_tensor_pairs

    def record(function):
        result = original(function)
        observed.append(result)
        return result

    with patch.object(DeviceFunction, "proven_disjoint_tensor_pairs", record):
        source = bound.to_code(_config(2))
    assert "_helion_prefetch" in source
    assert any(observed)


@pytest.mark.parametrize("pairs", [set(), {frozenset(("unrelated", "out"))}])
@skipUnlessBackends(["cute"])
def test_missing_or_wrong_actual_storage_proof_preserves_source(pairs):
    bound = _bound()
    with patch.object(
        DeviceFunction, "proven_disjoint_tensor_pairs", return_value=pairs
    ):
        assert bound.to_code(_config()) == bound.to_code(_config(2))


def _store(x: torch.Tensor, out: torch.Tensor, scale: float) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


@pytest.mark.parametrize("overlap", [False, True])
@skipUnlessBackends(["cute"])
def test_actual_same_or_overlapping_input_output_declines(overlap):
    storage = torch.empty(8193, dtype=torch.bfloat16)
    x = storage[:8192]
    out = storage[1:] if overlap else x
    bound = _bound(function=_store, args=(x, out, 2.0))
    assert bound.to_code(_config()) == bound.to_code(_config(2))


@pytest.mark.parametrize("batch", [False, True, -2, 1, 3, 16, 2.0, "2"])
def test_exact_integer_admission_and_fix_invalid(batch):
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_proven_bounds()
    spec.enable_cute_packet_prefetch()
    values = {"cute_proven_bounds": True, "cute_packet_prefetch": batch}
    with pytest.raises(InvalidConfig, match="must be 0, 2, 4, or 8"):
        spec._normalize_cute_packet_prefetch(values.copy(), fix_invalid=False)
    spec._normalize_cute_packet_prefetch(values, fix_invalid=True)
    assert "cute_packet_prefetch" not in values


@pytest.mark.parametrize(
    "enabled,bounds", [(False, True), (True, False), (False, False)]
)
def test_positive_choice_requires_pointwise_and_complete_tile_admission(
    enabled, bounds
):
    spec = ConfigSpec(backend=CuteBackend())
    spec.enable_cute_proven_bounds()
    if enabled:
        spec.enable_cute_packet_prefetch()
    values = {"cute_proven_bounds": bounds, "cute_packet_prefetch": 2}
    with pytest.raises(InvalidConfig, match="requires pointwise facts"):
        spec._normalize_cute_packet_prefetch(values.copy(), fix_invalid=False)
    spec._normalize_cute_packet_prefetch(values, fix_invalid=True)
    assert "cute_packet_prefetch" not in values


@pytest.mark.parametrize("batch", [0, 2, 4, 8])
@skipUnlessBackends(["cute"])
def test_search_coordinate_roundtrip_and_config_identity(batch):
    bound = _bound()
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        config = _config(batch, packets=max(batch, 2))
        flat = generation.flatten(config)
        normalized = generation.unflatten(flat)
        assert normalized.cute_packet_prefetch == batch
        assert generation.flatten(normalized) == flat
        assert helion.Config.from_json(config.to_json()) == config


@skipUnlessBackends(["cute"])
def test_all_old_seeds_remain_before_exact_bounded_strided_pairs():
    bound = _bound()
    spec = bound.config_spec
    with bound.env:
        with patch.object(spec, "cute_packet_prefetch_enabled", False):
            old = CutePointwiseVecHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        new = CutePointwiseVecHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert old and new
    assert [item.config for item in new[: len(old)]] == [item.config for item in old]
    suffix = new[len(old) :]
    assert 0 < len(suffix) <= 6 and len(suffix) % 2 == 0
    for control, candidate in zip(suffix[::2], suffix[1::2], strict=True):
        assert control.cute_packet_prefetch == 0
        assert control.config["cute_lane_layouts"] == ["strided"]
        assert candidate.config == control.config | {"cute_packet_prefetch": 2}
        assert candidate.cute_proven_bounds is True
        assert candidate.block_sizes == control.block_sizes
    preserved = deepcopy([item.config for item in new[:-1]])
    new[-1].config["block_sizes"][0] = 1
    assert [item.config for item in new[:-1]] == preserved


def test_no_full_tile_proof_does_not_invoke_packet_matcher():
    body = ast.parse("value = external_call()\n").body
    plan = FullTilePlan(
        TensorLayoutRelation("n", "x", "size", 0), 4096, (256, 1, 1), ("x", "out")
    )
    with patch(
        "helion._compiler.cute.prefetch_pointwise_packets.prefetch_pointwise_packets",
        side_effect=AssertionError("unproved source reached packet matcher"),
    ):
        assert (
            specialize_full_tile_bounds(
                body,
                plan,
                argument_names=("x", "out", "n"),
                constexpr_values={},
                rename_groups={},
                packet_prefetch=2,
                proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
            )
            is body
        )


def _name_collision(x: torch.Tensor, _helion_prefetched_vectors: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * _helion_prefetched_vectors
    return out


@skipUnlessBackends(["cute"])
def test_final_parameter_renaming_cannot_capture_new_packet_lists():
    bound = _bound(function=_name_collision)
    off = _functions(bound.to_code(_config()))[-2]
    on = _functions(bound.to_code(_config(2)))[-2]
    assert ast.dump(off.args) == ast.dump(on.args)
    parameters = {argument.arg for argument in on.args.args}
    private_lists = {
        target.id
        for node in ast.walk(on)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert private_lists.isdisjoint(parameters)
    assert _dump(off.body[0].orelse) == _dump(on.body[0].orelse)


@pytest.mark.parametrize("length,stride", [(0, 1), (1027, 1), (8192, 2)])
@skipUnlessBackends(["cute"])
def test_empty_tail_and_nonunit_stride_choose_exact_original_fallback(length, stride):
    tensor = torch.empty(max(length * stride, 1), dtype=torch.bfloat16)[
        : length * stride : stride
    ]
    bound = _bound(args=(tensor, 2.0))
    before = _functions(bound.to_code(_config()))[-2]
    after = _functions(bound.to_code(_config(2)))[-2]
    if not isinstance(after.body[0], ast.If):
        assert _dump(before.body) == _dump(after.body)
        return
    assert _dump(before.body[0].orelse) == _dump(after.body[0].orelse)
    assert ast.dump(before.body[0].test) == ast.dump(after.body[0].test)
    value = SimpleNamespace(layout=SimpleNamespace(shape=(length,), stride=(stride,)))
    result = eval(
        compile(ast.Expression(after.body[0].test), "<full-tile-guard>", "eval"),
        {
            "cutlass": SimpleNamespace(const_expr=bool),
            "cute": SimpleNamespace(
                is_static=lambda _: True, size=lambda x, mode: x.layout.shape[mode[0]]
            ),
            "x": value,
            "out": value,
        },
    )
    assert result is False


def test_knob_is_not_admitted_by_other_backends():
    spec = ConfigSpec(backend=TritonBackend())
    assert not spec.supports_config_key("cute_packet_prefetch")
    with pytest.raises(InvalidConfig, match="requires CuTe"):
        spec.enable_cute_packet_prefetch()
    with pytest.raises(InvalidConfig, match="Unsupported config keys"):
        spec.normalize(helion.Config(cute_packet_prefetch=2))


NAMESPACE_SOURCE = """
pid = cutlass.Int32(cute.arch.block_idx()[0])
for lane in range(4):
    base = pid * 8192 + (cutlass.Int32(lane) * 256 + cutlass.Int32(cute.arch.thread_idx()[0])) * 8
    packed = cute.arch.load(x.iterator + base, ir.VectorType.get([8], cutlass.Uint16.mlir_type))
    values = []
    for vector_lane in cutlass.range_constexpr(8):
        index = base + cutlass.Int32(vector_lane)
        mask = index < n
        value = cutlass.Uint16(packed[vector_lane]).bitcast(cutlass.BFloat16) if mask else cutlass.BFloat16(0)
        result = value * v_0
        values.append(cutlass.BFloat16(result).bitcast(cutlass.Uint16))
    _cute_store_u16_vec(out.iterator + base, values)
"""


@pytest.mark.parametrize(
    "renames,unused",
    [
        ({"v_0": "_helion_packet"}, ()),
        ({"_helion_packet": "v_0"}, ()),
        ({"v_0": "_helion_prefetched_vectors"}, ()),
        ({}, ("_helion_packet_batch",)),
    ],
)
def test_actual_full_tile_hook_protects_absent_final_rename_namespace(renames, unused):
    from helion._compiler.ast_read_writes import ast_rename

    body = ast.parse(NAMESPACE_SOURCE).body
    arguments = ("x", "out", "n", "v_0", *unused)
    plan = FullTilePlan(
        TensorLayoutRelation("n", "x", "size", 0), 8192, (256, 1, 1), ("x", "out")
    )
    result = specialize_full_tile_bounds(
        body,
        plan,
        argument_names=arguments,
        constexpr_values={},
        rename_groups=renames,
        packet_prefetch=2,
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    assert "_helion_prefetched_vectors" in ast.unparse(
        ast.Module(body=result, type_ignores=[])
    )
    renamed = ast_rename(ast.Module(body=result, type_ignores=[]), renames)
    protected = {renames.get(name, name) for name in arguments}
    written = {
        node.id
        for node in ast.walk(renamed)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert not ((protected - {"n"}) & written)


def test_final_rename_cannot_hide_a_loop_carried_load_address():
    source = NAMESPACE_SOURCE.replace("pid =", "carry = 0\npid =")
    source = source.replace(
        "    packed =", "    source_base = base ^ (carry * 8)\n    packed ="
    )
    source = source.replace("x.iterator + base", "x.iterator + source_base")
    source += "    carry_next = carry + 1\n"
    body = ast.parse(source).body
    plan = FullTilePlan(
        TensorLayoutRelation("n", "x", "size", 0), 8192, (256, 1, 1), ("x", "out")
    )
    result = specialize_full_tile_bounds(
        body,
        plan,
        argument_names=("x", "out", "n", "v_0"),
        constexpr_values={},
        rename_groups={"carry_next": "carry"},
        packet_prefetch=2,
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    assert isinstance(result[0], ast.If)
    disabled = specialize_full_tile_bounds(
        ast.parse(source).body,
        plan,
        argument_names=("x", "out", "n", "v_0"),
        constexpr_values={},
        rename_groups={"carry_next": "carry"},
        packet_prefetch=0,
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == ast.dump(
        ast.Module(body=disabled, type_ignores=[])
    )
    assert "_helion_prefetched_vectors" not in ast.unparse(
        ast.Module(body=result, type_ignores=[])
    )


@pytest.mark.parametrize("bounds", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2, 4, 8])
@skipUnlessBackends(["cute"])
def test_flat_dependency_repair_changes_only_prefetch(bounds, prefetch):
    bound = _bound()
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat = generation.flatten(_config(2))
        p = generation._key_to_flat_indices["cute_packet_prefetch"][0][0]
        b = generation._key_to_flat_indices["cute_proven_bounds"][0][0]
        flat[p], flat[b] = prefetch, bounds
        before = deepcopy(flat)
        generation._repair_cute_num_threads(flat)
        assert flat[p] == (prefetch if bounds else 0)
        assert flat[:p] + flat[p + 1 :] == before[:p] + before[p + 1 :]
        assert generation.flatten(generation.unflatten(flat)) == flat


@skipUnlessBackends(["cute"])
def test_flat_repair_is_inert_when_fields_or_backend_are_absent():
    bound = _bound()
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat = generation.default_flat()
        before = deepcopy(flat)
        with patch.object(generation, "_key_to_flat_indices", {}):
            generation._repair_cute_packet_prefetch(flat)
        assert flat == before
    generation = ConfigGeneration(ConfigSpec(backend=TritonBackend()))
    flat = generation.default_flat()
    before = deepcopy(flat)
    generation._repair_cute_packet_prefetch(flat)
    assert flat == before


@skipUnlessBackends(["cute"])
def test_random_biased_and_coordinate_neighbors_round_trip():
    import random

    saved = random.getstate()
    try:
        random.seed(2026091627)
        bound = _bound()
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            for _ in range(16):
                for flat in (generation.random_flat(), generation.biased_random_flat()):
                    assert generation.flatten(generation.unflatten(flat)) == flat
            for bounds in (False, True):
                base = generation.flatten(_config(0, bounds=bounds))
                neighbors = generation.coordinate_neighbor_projections(base)
                packet_neighbors = [
                    item for item in neighbors if item.key == "cute_packet_prefetch"
                ]
                assert {item.to_value for item in packet_neighbors} == {2, 4, 8}
                for item in packet_neighbors:
                    flat, config = item.flat_values, item.config
                    assert flat is not None and config is not None
                    assert generation.flatten(config) == flat
                    assert config.cute_packet_prefetch == (
                        item.to_value if bounds else 0
                    )
    finally:
        random.setstate(saved)
