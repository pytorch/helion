from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _input
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute_positional_cache import (
    CutePositionalCacheHeuristic,
)
from helion._compiler.cute.positional_pointwise_cache import plan_contraction_cache
from helion._compiler.cute.positional_root import PositionalEmitter
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_spec import (
    CUTE_POSITIONAL_POINTWISE_CACHE_BYTES_KEY as KEY,
)
from helion.autotuner.config_spec import VALID_CUTE_POSITIONAL_POINTWISE_CACHE_BYTES
import helion.language as hl
from helion.language import view_ops
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from torch.fx import Node


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def weighted_gram(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    batch, rows, _ = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        weight = gate[tb, :].float().exp().to(x.dtype)
        values = x[tb, :, :]
        out[tb, :, :] = hl.dot(values * weight[:, None, :], values.transpose(-1, -2))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def weighted_gram_tail(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    batch, rows, width = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        total = hl.zeros([tb, rows, rows], dtype=torch.float32)
        for tk in hl.tile(width, block_size=16):
            weight = gate[tb, tk].float().exp().to(x.dtype)
            values = x[tb, :, tk]
            total = hl.dot(values * weight[:, None, :], values.transpose(-1, -2), total)
        out[tb, :, :] = total
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def repeated_weight(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # The first dot's temporary cache must not become a persistent producer cut.
    batch, rows, _ = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        weight = gate[tb, :].float().exp()
        values = x[tb, :, :].float()
        lhs = values * weight[:, None, :]
        first = hl.dot(lhs, values.transpose(-1, -2))
        second = hl.dot(lhs, first)
        out[tb, :, :] = second + weight[:, None, :]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def mutated_weight(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    batch, rows, _ = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        weight = gate[tb, :].float().exp()
        gate[tb, :] = gate[tb, :] + 1
        values = x[tb, :, :].float()
        out[tb, :, :] = hl.dot(values * weight[:, None, :], values.transpose(-1, -2))
    return out


def _fixture(
    *, width: int = 32, dtype: torch.dtype = torch.float32, narrow: bool = False
) -> tuple[Graph, Node, Node]:
    graph = Graph()
    raw = _input(graph, "raw", (1, width), dtype)
    computed = _call(graph, torch.ops.aten.exp.default, (raw,), (1, width), dtype)
    if narrow:
        computed = _convert(graph, computed, torch.bfloat16)
        dtype = torch.bfloat16
    expanded = _call(
        graph,
        torch.ops.aten.unsqueeze.default,
        (computed, 1),
        (1, 1, width),
        dtype,
    )
    values = _input(graph, "values", (1, width, width), dtype)
    weighted = _call(
        graph,
        torch.ops.aten.mul.Tensor,
        (values, expanded),
        (1, width, width),
        dtype,
    )
    result = _call(
        graph,
        dot,
        (weighted, values, None, torch.float32),
        (1, width, width),
        torch.float32,
    )
    graph.output(result)
    return graph, result, computed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_small_vector_wins_over_full_operand(dtype: torch.dtype) -> None:
    _, result, computed = _fixture(dtype=dtype)
    plan = plan_contraction_cache(result, set(), 16384)
    assert len(plan.entries) == 1
    assert plan.entries[0].node is computed
    assert plan.entries[0].dtype == dtype
    assert plan.entries[0].shape == (1, 32)


@pytest.mark.parametrize("budget", [0, 1, 127, 128, 255, 256])
def test_alignment_and_budget(budget: int) -> None:
    _, result, _ = _fixture(width=33)
    plan = plan_contraction_cache(result, set(), budget)
    assert plan.shared_bytes <= budget
    assert bool(plan.entries) == (budget >= 256)


def test_cast_is_a_typed_cut_and_existing_boundary_stops_discovery() -> None:
    _, result, computed = _fixture(narrow=True)
    plan = plan_contraction_cache(result, set(), 256)
    assert plan.entries[0].node is computed
    assert plan.entries[0].dtype is torch.bfloat16
    assert not plan_contraction_cache(result, {computed}, 256).entries


def test_internal_gather_does_not_redirect_outside_physical_domain() -> None:
    graph, result, computed = _fixture()
    original = next(iter(computed.users))
    with graph.inserting_before(original):
        gather = _call(
            graph,
            view_ops.subscript,
            (computed, (slice(None), -1)),
            (1,),
            torch.float32,
        )
    # This synthetic graph is intentionally only a coordinate-proof fixture.
    original.replace_input_with(computed, gather)
    assert not plan_contraction_cache(result, set(), 256).entries


def _source(kernel: helion.Kernel, width: int, budget: int) -> str:
    x = torch.empty(2, 32, width, dtype=torch.bfloat16)
    gate = torch.empty(2, width, dtype=torch.float32)
    with _cpu_codegen():
        bound = kernel._bind_isolated((x, gate))
        config = bound.config_spec.default_config()
        config.config[KEY] = budget
        return bound.to_code(config)


def test_default_zero_and_search_serialization() -> None:
    with _cpu_codegen():
        args = (torch.empty(2, 32, 32), torch.empty(2, 32))
        bound = weighted_gram._bind_isolated(args)
        spec = bound.config_spec
        assert spec.cute_positional_pointwise_cache_search_enabled
        assert spec.supports_config_key(KEY)
        default = spec.default_config()
        assert KEY not in default.config
        assert spec.flatten_missing_field_default(KEY, {}) == (True, 0)
        explicit = deepcopy(default)
        explicit.config[KEY] = 0
        assert spec.normalized_config(explicit) == spec.normalized_config(default)
        assert bound.to_code(default) == bound.to_code(explicit)
        field = spec._flat_fields()[KEY]
        assert isinstance(field, EnumFragment)
        assert field.choices == VALID_CUTE_POSITIONAL_POINTWISE_CACHE_BYTES
        assert not CutePositionalCacheHeuristic.promote_seed_to_default
        for budget in (-1, True, 128, "256"):
            with pytest.raises(exc.InvalidConfig, match=KEY):
                spec.normalize(default.config | {KEY: budget})


def test_codegen_keeps_typed_publication_before_scalar_k_loop() -> None:
    source = _source(weighted_gram, 32, 256)
    assert "ptp_cache_" in source
    assert "alloc_smem(cutlass.BFloat16, 32, alignment=128)" in source
    tree = ast.parse(source)
    scalar_k = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node.target).startswith("ptp_k_")
    ]
    assert len(scalar_k) == 1
    assert "ptp_cache_" in ast.unparse(scalar_k[0])
    assert "exp" not in ast.unparse(scalar_k[0])
    assert "cutlass.Float32" in ast.unparse(scalar_k[0])


def test_loop_republishes_and_k_tail_mask_survives() -> None:
    source = _source(weighted_gram_tail, 40, 256)
    loops = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)]
    cache_stores = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and ast.unparse(target.value).startswith("ptp_cache_")
            for target in node.targets
        )
    ]
    assert len(cache_stores) == 1
    assert any(
        "ptp_k_" in ast.unparse(loop)
        and "ptp_cache_" in ast.unparse(loop)
        and "exp" in ast.unparse(loop)
        for loop in loops
    )
    scalar_k = next(
        loop for loop in loops if ast.unparse(loop.target).startswith("ptp_k_")
    )
    body = ast.unparse(scalar_k)
    assert "if" in body and "40" in source
    assert "exp" not in body


def test_cache_bindings_restore_and_total_budget_is_root_wide() -> None:
    original = PositionalEmitter.emit_contraction
    observations: list[tuple[int, int]] = []

    def observe(self: PositionalEmitter, node: Node) -> list[str]:
        before = self.boundaries.copy()
        code = original(self, node)
        assert self.boundaries.keys() == before.keys() | {node}
        observations.append((self.pointwise_cache_bytes, self.allocated_bytes))
        return code

    with patch.object(PositionalEmitter, "emit_contraction", observe):
        source = _source(repeated_weight, 32, 256)
    assert len(observations) == 2
    assert observations[0][0] == 128 and observations[1][0] == 256
    assert source.count("alignment=128), cute.make_layout((1, 32)))") == 2


def test_capacity_limits_optional_cache_without_changing_base_admission() -> None:
    with (
        _cpu_codegen(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=4096
        ),
    ):
        args = (torch.empty(2, 32, 32), torch.empty(2, 32))
        bound = weighted_gram._bind_isolated(args)
        config = bound.config_spec.default_config()
        config.config[KEY] = 256
        source = bound.to_code(config)
    assert "ptp_cache_" not in source
    assert "ptp_thread" in source


def test_mutation_uses_existing_snapshot_and_counts_cache_storage() -> None:
    original = PositionalEmitter.write_tile
    publications: list[list[str]] = []

    def observe(self: PositionalEmitter, node: Node, buffer: str) -> list[str]:
        code = original(self, node, buffer)
        if buffer.startswith("ptp_cache_"):
            # The old-value read and the store's RHS read are separate loads.
            assert self.snapshot_bytes == 256
            publications.append(code)
        return code

    with patch.object(PositionalEmitter, "write_tile", observe):
        source = _source(mutated_weight, 32, 256)
    assert len(publications) == 1
    publication = "\n".join(publications[0])
    assert "ptp_tile_" in publication  # reads the pre-write snapshot
    assert "cute.arch.load" not in publication
    assert "ptp_cache_" in source


def test_unrelated_nonpositional_kernel_rejects_explicit_cache() -> None:
    from .test_cute_positional_root import square_gram

    with _cpu_codegen():
        bound = square_gram._bind_isolated((torch.empty(2, 32, 32),))
        assert not bound.config_spec.cute_positional_pointwise_cache_search_enabled
        config = bound.config_spec.default_config()
        config.config[KEY] = 256
        with pytest.raises(
            exc.InvalidConfig, match="positional contraction arithmetic"
        ):
            bound.to_code(config)
