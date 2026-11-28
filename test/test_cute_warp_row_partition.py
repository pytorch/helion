"""A shared physical CTA may give a pure implicit row reduction its own warps."""

from __future__ import annotations

import ast

from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import (
    _helion_aot_qwen3_30b_a3b_moe_cuda_sm100 as aot,
)
from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import qwen3_30b_a3b_moe as original
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute.autotune.config_keys import ROW_DOT_KEY
from helion._compiler.cute.computed_fragment import FragmentCompiler
from helion._compiler.cute.warp_row_partition import KEY
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import default_cute_mma_support
import helion.language as hl
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _partitioned(
    x: torch.Tensor, weights: torch.Tensor, peer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(weights.size(0), dtype=torch.float32, device=x.device)
    dst = torch.empty_like(peer)
    for row in hl.tile(weights.size(0)):
        norm = (x[:].float() * x[:].float()).mean()
        out[row] = (weights[row, :].float() * (x[:].float() / (norm + 1))).sum(-1)
    hl.barrier()
    for j in hl.tile(peer.numel()):
        dst[j] = peer[j] + 1
    return out, dst


@helion.kernel(backend="cute", static_shapes=True)
def _with_second_effect(
    x: torch.Tensor, weights: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(weights.size(0), device=x.device)
    for row in hl.tile(weights.size(0)):
        acc = (weights[row, :].float() * x[:].float()).sum(-1)
        out[row] = acc
        scratch[row] = acc + 2
    hl.barrier()
    for j in hl.tile(scratch.numel()):
        scratch[j] += 1
    return out


def _args(n: int, k: int) -> tuple[torch.Tensor, ...]:
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        return (
            torch.empty(k, dtype=torch.bfloat16),
            torch.empty(n, k, dtype=torch.bfloat16),
            torch.empty(256),
        )


def _target():
    return cpu_render_target(
        fake_device_sm_count=148,
        mma_support=default_cute_mma_support(),
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    )


def test_semantic_admission_rejects_effects_and_partial_warps() -> None:
    with _target():
        ok = cpu_bind(_partitioned, _args(16, 128)).config_spec
        assert next(iter(ok.cute_warp_row_reduction_roots.values()))[-2:] == (128, 2)
        assert not cpu_bind(
            _partitioned, _args(16, 129)
        ).config_spec.cute_warp_row_reduction_roots
        side = cpu_bind(_with_second_effect, _args(16, 128))
        assert not side.config_spec.cute_warp_row_reduction_roots
        with pytest.raises(exc.InvalidConfig, match="pure sum"):
            side.to_code(
                helion.Config.from_dict(
                    dict(side.config_spec.default_config()) | {KEY: True}
                )
            )


@pytest.mark.slow
def test_original_seed_partitions_only_the_row_reductions() -> None:
    # The frozen production program also contains two rolled expert projections
    # and the output merge. Register and render the real ordinary mixed seed.
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = (
            tuple(
                torch.empty(shape, dtype=dtype)
                for shape, dtype in aot._TENSOR_SIGNATURES
            )
            + aot._STATIC_ARGS
        )
    op = helion.kernel(
        backend="cute", static_shapes=original.qwen3_30b_a3b_moe.settings.static_shapes
    )(original.qwen3_30b_a3b_moe.fn)
    with _target():
        bound = cpu_bind(op, args)
        spec = bound.config_spec
        # Combined seeds also partition the router, but switch the separate
        # down projection to MMA. Exercise the original scalar companion here.
        (requested,) = [
            c
            for c in spec.compiler_seed_configs
            if c.get(KEY) and not c.get(ROW_DOT_KEY)
        ]
        generation = spec.create_config_generation()
        flat, cfg = generation.strict_config_pair(requested)
        assert any(pair[0] == flat for pair in generation.seed_flat_config_pairs())
        # The normalizer may not reserve these lanes for the other reductions.
        assert max(cfg.config["num_threads"]) == 256
        src = ast.parse(bound.to_code(cfg))
        calls = [n for n in ast.walk(src) if isinstance(n, ast.Call)]
        (launcher,) = [
            n
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "_launcher"
        ]
        assert ast.literal_eval(
            next(kw.value for kw in launcher.keywords if kw.arg == "block")
        ) == (256, 1, 1)
        reductions = [
            n for n in calls if ast.unparse(n.func) == "cute.arch.warp_reduction_sum"
        ]
        # Each reachable phase is now emitted once. The router has the two
        # proved sums (RMS and dot), while the other three phases have none.
        (facts,) = spec.cute_warp_row_reduction_roots.values()
        assert len(reductions) == facts[-1] == 2 and all(
            ast.literal_eval(n.keywords[0].value) == 32 for n in reductions
        )
        leaders = [
            n
            for n in ast.walk(src)
            if isinstance(n, ast.Compare)
            and ast.unparse(n) == "cute.arch.thread_idx()[0] % 32 == 0"
        ]
        assert len(leaders) == len(spec.cute_warp_row_reduction_roots) == 1
        # A local store is valid only if every output has a complete warp.
        row, _, _, _ = next(iter(spec.cute_warp_row_reduction_roots.values()))
        bs = list(cfg.block_sizes)
        bs[spec.block_sizes.valid_block_ids().index(row)] = 32
        with pytest.raises(exc.InvalidConfig, match="linear CTA"):
            bound.to_code(helion.Config.from_dict(cfg.config | {"block_sizes": bs}))


def test_carry_shape_only_edge_does_not_evaluate_its_unselected_value() -> None:
    graph = torch.fx.Graph()
    producer = graph.call_function(_tracing_ops._new_var, (graph.placeholder("x"),))
    size = graph.call_function(torch.ops.aten.sym_size.int, (producer, 0))
    size.meta["val"] = 37
    compiler = object.__new__(FragmentCompiler)
    compiler.pending_barrier = False
    assert compiler.node(size, {}) == 37
