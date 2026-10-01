from __future__ import annotations

import ast
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_collective_retention_search import _config as _pipeline_config
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_tcgen05 import _both_operands_chain
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_tcgen05 as roots
from helion._compiler.cute import chained_tmem_drains as drains
from helion._compiler.cute.chained_tmem_drains import DrainTiling
from helion.autotuner.config_spec import CUTE_CHAINED_DRAIN_TILE_COLUMNS_KEY as KEY
from helion.autotuner.config_spec import ConfigSpec
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


def _seed_context(width=128, *, exclusive=False, initialized=False, rows=128):
    graph = torch.fx.Graph()
    operand = graph.placeholder("operand")
    first = graph.call_function(dot, (operand, operand, None))
    first.meta["val"] = torch.empty((rows, width), dtype=torch.float32)
    converted = graph.call_method("to", (first, torch.bfloat16))
    graph.call_function(
        dot,
        (
            converted,
            operand if exclusive else converted,
            operand if initialized else None,
        ),
    )
    return (
        cast(
            "CompileEnvironment",
            SimpleNamespace(
                config_spec=SimpleNamespace(
                    cute_chained_tcgen05_search_enabled=True,
                    cute_chained_loop_search_enabled=False,
                )
            ),
        ),
        cast(
            "DeviceIR",
            SimpleNamespace(
                graphs=[SimpleNamespace(graph=graph)], host_function=nullcontext()
            ),
        ),
    )


def test_seed_siblings_append_without_touching_prefix_or_multiplicity():
    first = helion.Config(
        block_sizes=[], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )
    second = helion.Config.from_dict(
        first.config | {"cute_chained_tmem_early_release": True}
    )
    seeds = [first, first, second]
    env, ir = _seed_context()
    result = CuteChainedMatmulHeuristic._with_complete_drain_seeds(env, ir, seeds)
    assert all(a is b for a, b in zip(result[:3], seeds, strict=True))
    assert [seed.config for seed in result[3:]] == [
        seed.config | {KEY: 32} for seed in (first, second)
    ]
    assert all(KEY not in seed.config for seed in seeds)
    assert (
        CuteChainedMatmulHeuristic._with_complete_drain_seeds(env, ir, result) is result
    )


@pytest.mark.parametrize(
    "options",
    [
        {"width": 32},
        {"width": 48},
        {"width": 288},
        {"rows": 64},
        {"exclusive": True},
        {"initialized": True},
    ],
)
def test_ineligible_graph_does_not_seed_panels(options):
    seeds = [
        helion.Config(
            block_sizes=[], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    ]
    env, ir = _seed_context(**options)
    assert (
        CuteChainedMatmulHeuristic._with_complete_drain_seeds(env, ir, seeds) is seeds
    )


def test_unsupported_root_participant_parent_is_not_repaired():
    seeds = [
        helion.Config(
            block_sizes=[], num_warps=8, cute_chained_mma_schedule="tcgen05_tmem"
        )
    ]
    env, ir = _seed_context()
    assert (
        CuteChainedMatmulHeuristic._with_complete_drain_seeds(env, ir, seeds) is seeds
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_real_root_seed_prefix_and_default_source(dtype):
    args = (torch.empty((2, 128, 128), dtype=dtype),) * 2
    with _cpu_codegen():
        with patch.object(
            CuteChainedMatmulHeuristic,
            "_with_complete_drain_seeds",
            side_effect=lambda env, ir, seeds: seeds,
        ):
            old = _both_operands_chain._bind_isolated(args)
            parents = [
                dict(seed.config) for seed in old.config_spec.compiler_seed_configs
            ]
            default = old.config_spec.default_config()
            source = old.to_code(default)
        current = _both_operands_chain._bind_isolated(args)
        spec = current.config_spec
        seeds = spec.compiler_seed_configs
        assert [seed.config for seed in seeds[: len(parents)]] == parents
        assert spec.default_config() == default
        assert current.to_code(default) == source
        siblings = seeds[len(parents) :]
        assert 0 < len(siblings) <= 2
        for seed in siblings:
            assert {k: v for k, v in seed.config.items() if k != KEY} in parents
            assert "chain_0_drain_panel" in current.to_code(
                spec.normalized_config(seed)
            )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize(
    "value", [None, True, False, 1, 16, 64, 0.0, 32.0, "32", [], {}]
)
def test_strict_selection(retention_bound, repair, value):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(
            _pipeline_config(**{KEY: value}), _fix_invalid=repair
        )


def test_append_only_coordinate_default_and_roundtrip(retention_bound):
    spec = retention_bound.config_spec
    fields = spec._flat_fields()
    assert tuple(fields)[-5:] == (
        "cute_native_matmul_metadata",
        KEY,
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    assert fields[KEY].search_values() == [0, 32]
    assert not ConfigSpec._requests_cute_chained_loop({KEY: 0})
    assert ConfigSpec._requests_cute_chained_loop({KEY: 32})
    assert spec.normalized_config(
        helion.Config.from_dict({KEY: 0})
    ) == spec.normalized_config(helion.Config())
    generation = spec.create_config_generation()
    for columns in (0, 32):
        config = spec.normalized_config(_pipeline_config(**{KEY: columns}))
        flat = generation.flatten(config)
        assert generation.unflatten(flat) == config
        assert (KEY in config.config) == bool(columns)
        with patch.object(
            spec,
            "_flat_fields",
            return_value={k: v for k, v in fields.items() if k != KEY},
        ):
            old = spec.create_config_generation()
            old_config = helion.Config.from_dict(
                {k: v for k, v in config.config.items() if k != KEY}
            )
            # Later island, vector-store and fragment coordinates remain intact.
            assert flat[:-4] + flat[-3:] == old.flatten(old_config)


@pytest.mark.parametrize(
    "missing", ["cute_chained_mma_schedule", "cute_chained_preparation_pipeline"]
)
def test_no_implicit_prerequisite(retention_bound, missing):
    config = _pipeline_config(**{KEY: 32})
    config.config.pop(missing)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["row_major", "xor"])
def test_real_materialized_root_uses_panels_only_for_intermediate(dtype, layout):
    with _cpu_codegen():
        bound = _both_operands_chain._bind_isolated(
            (torch.empty((2, 128, 128), dtype=dtype),) * 2
        )
        config = helion.Config(
            block_sizes=[],
            num_warps=4,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_scratch_layout=layout,
        )
        original = bound.to_code(config)
        explicit = bound.to_code(helion.Config.from_dict(config.config | {KEY: 0}))
        if layout == "xor":
            with pytest.raises(exc.BackendUnsupported, match="existing materialized"):
                bound.to_code(helion.Config.from_dict(config.config | {KEY: 32}))
            assert original == explicit
            return
        selected = bound.to_code(helion.Config.from_dict(config.config | {KEY: 32}))
    assert original == explicit
    assert "for chain_0_drain_panel in cutlass.range(4, unroll=1)" in selected
    assert "chain_0_drain_shared_target" in selected
    assert "chain_1_drain" not in selected
    assert "chain_1_values = cute.make_rmem_tensor" in selected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_real_public_carry_activates(dtype):
    args = (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in ((3, 128, 16), (3, 16, 64), (3, 64, 16), (3, 16, 64))
        ),
        torch.empty((128, 64), dtype=torch.float32),
    )
    config = _config(16, pipeline=True, consumer_warps=8)
    config.config["cute_chained_warp_mma_rows"] = 64
    default = _source(_resident_carry_sequence, args, config)
    assert (
        _source(
            _resident_carry_sequence,
            args,
            helion.Config.from_dict(config.config | {KEY: 0}),
        )
        == default
    )
    selected = _source(
        _resident_carry_sequence,
        args,
        helion.Config.from_dict(config.config | {KEY: 32}),
    )
    assert "_drain_panel in cutlass.range(2, unroll=1)" in selected
    assert "chain_tmem_barrier.arrive_and_wait()" in selected


def test_positive_requires_actual_activation():
    DrainTiling().validate()
    with pytest.raises(exc.BackendUnsupported, match="emitted complete"):
        DrainTiling(32).validate()


def _restore_span(source, candidate, original):
    tree = ast.parse(source)
    new = ast.parse("\n".join(candidate)).body
    old = ast.parse("\n".join(original)).body
    count = 0
    for parent in ast.walk(tree):
        for name, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index in range(len(value) - len(new) + 1):
                if all(
                    isinstance(a, ast.AST) and ast.dump(a) == ast.dump(b)
                    for a, b in zip(value[index : index + len(new)], new, strict=True)
                ):
                    setattr(
                        parent, name, value[:index] + old + value[index + len(new) :]
                    )
                    count += 1
    assert count == 1, candidate[0]
    return ast.unparse(tree)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("client", ["root", "loop"])
def test_complete_source_inverse_only_original_materialization_changes(dtype, client):
    spans = []
    emitter = drains.emit_streamed_fp32_drain
    materialize = roots._materialize_result

    def capture(panels, prefix, source_prefix, publication, **kwargs):
        result = emitter(panels, prefix, source_prefix, publication, **kwargs)
        if isinstance(publication, drains.ScalarPublication):
            spans.append(
                (
                    result,
                    roots._load_result(
                        source_prefix, panels.shape, execution=kwargs["execution"]
                    )
                    + publication.lines(
                        f"{source_prefix}_values", f"{source_prefix}_coords"
                    ),
                )
            )
        return result

    def capture_root(prefix, shape, scratch, **kwargs):
        result = materialize(prefix, shape, scratch, **kwargs)
        if kwargs.get("drain_tiling") is not None:
            spans.append(
                (
                    result,
                    roots._load_result(prefix, shape)
                    + materialize(prefix, shape, scratch),
                )
            )
        return result

    if client == "root":
        kernel = _both_operands_chain
        args = (torch.empty((2, 128, 128), dtype=dtype),) * 2
        config = helion.Config(
            block_sizes=[], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    else:
        kernel = _resident_carry_sequence
        args = (
            *(
                torch.empty(shape, dtype=dtype)
                for shape in ((3, 128, 16), (3, 16, 64), (3, 64, 16), (3, 16, 64))
            ),
            torch.empty((128, 64), dtype=torch.float32),
        )
        config = _config(16, pipeline=True, consumer_warps=8)
        config.config["cute_chained_warp_mma_rows"] = 64
    original = _source(kernel, args, config)
    with (
        patch.object(drains, "emit_streamed_fp32_drain", capture),
        patch.object(roots, "_materialize_result", capture_root),
        patch(
            "helion._compiler.cute.chained_loop_tmem_carry_transport.emit_streamed_fp32_drain",
            capture,
        ),
    ):
        current = _source(
            kernel, args, helion.Config.from_dict(config.config | {KEY: 32})
        )
    assert spans
    for new, old in spans:
        current = _restore_span(current, new, old)
    assert ast.dump(ast.parse(current)) == ast.dump(ast.parse(original))
