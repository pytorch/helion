from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute import chained_root_vector_cache as vector
from helion._compiler.cute.chained_matmul import _UnsupportedChain
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import _Expression
    from helion._compiler.cute.chained_root_native import RootNativeStage
    from helion._compiler.cute.chained_root_native_emission import RootNativeEmission


def _native_vector_scope_fixture():
    from torch.fx import Graph

    from helion._compiler.cute.chained_execution import ChainedExecution
    from helion._compiler.cute.chained_tcgen05 import _load_result
    from helion._compiler.cute.chained_tcgen_stage import StageGeometry

    graph = Graph()
    source, node = graph.placeholder("source"), graph.placeholder("published_vector")
    plan = SimpleNamespace()
    cg = SimpleNamespace()
    stage = SimpleNamespace(
        index=0,
        node=source,
        geometry=StageGeometry((64, 64, 128), False, native_rows=64),
    )
    body = SimpleNamespace(
        plan=plan,
        cg=cg,
        execution=ChainedExecution(128),
        check=lambda: None,
        native=SimpleNamespace(selected=lambda plan, index: stage),
    )
    program = tuple(_load_result("chain_0", (64, 64), execution=body.execution))
    lines = ["issue()", "wait()", *program]
    coordinates = ("chain_0_typed_row", "chain_0_typed_column")
    scope = vector.NativeFragmentScope(
        cast("RootNativeEmission", body),
        cast("RootNativeStage", stage),
        lines,
        tuple(lines),
        coordinates,
        "chain_0_typed_index",
        "chain_0_values[chain_0_typed_index]",
        program,
    )
    pub = SimpleNamespace(
        cache=SimpleNamespace(node=node, name="owned_vector", extent=64),
        codegen=cg,
        plan=plan,
        check=lambda: None,
        reads=[],
    )
    binding = vector.PublishedRootVector(cast("vector.RootVectorPublication", pub))
    expression = SimpleNamespace(
        cg=cg,
        plan=plan,
        boundaries={node: binding},
        fragments={source: (coordinates, scope.fragment)},
        coordinate_normalization=None,
        root_vector_scope=None,
    )
    read_scope = vector.RootReadScope(cast("_Expression", expression), scope)
    expression.root_vector_scope = read_scope
    return scope, expression, read_scope, binding, node


def test_native_fragment_vector_read_keeps_actual_coordinates_and_owners():
    scope, expression, read_scope, binding, node = _native_vector_scope_fixture()
    read_scope.read(binding, node, (scope.coordinates[0],))
    read_scope.read(binding, node, (scope.coordinates[1],))
    assert binding.publication.reads == [read_scope]
    for coordinate in ("foreign_row", "chain_0_typed_row + 1", "-1"):
        with pytest.raises(_UnsupportedChain, match="unbounded"):
            read_scope.read(binding, node, (coordinate,))
    object.__setattr__(binding.publication.cache, "extent", 32)
    with pytest.raises(_UnsupportedChain, match="unbounded"):
        read_scope.read(binding, node, (scope.coordinates[0],))
    object.__setattr__(binding.publication.cache, "extent", 64)
    expression.boundaries[node] = str(binding)
    with pytest.raises(_UnsupportedChain, match="foreign owner"):
        read_scope.read(binding, node, (scope.coordinates[0],))


def test_native_fragment_scope_rejects_changed_load_or_fragment():
    scope, expression, read_scope, binding, node = _native_vector_scope_fixture()
    original = scope.lines[-1]
    scope.lines[-1] = "changed_load_fence()"
    with pytest.raises(_UnsupportedChain, match="coordinate scope changed"):
        read_scope.check()
    scope.lines[-1] = original
    expression.fragments[scope.stage.node] = (scope.coordinates, "other_fragment")
    with pytest.raises(_UnsupportedChain, match="fragment context is foreign"):
        read_scope.check()


def test_native_fragment_wrapper_keeps_owned_load_index_and_domain():
    scope, expression, read_scope, binding, node = _native_vector_scope_fixture()
    text = "\n".join(scope.wrap(["typed[row, col] = value"]))
    assert "cute.size(chain_0_values)" in text
    assert "chain_0_coords[chain_0_typed_index]" in text
    assert "0 <= chain_0_typed_row < 64 and 0 <= chain_0_typed_column < 64" in text
    with pytest.raises(_UnsupportedChain, match="source binding changed"):
        scope.check_fragment(
            scope.body.plan,
            scope.stage.node,
            tuple(reversed(scope.coordinates)),
            scope.fragment,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence(a, b, rhs, raw):
    columns = hl.specialize(rhs.size(2))
    output = torch.empty((a.shape[0], 16, columns), device=a.device, dtype=a.dtype)
    side = torch.empty((a.shape[0], 16), device=a.device, dtype=torch.float32)
    for batch in hl.tile(a.shape[0], block_size=1):
        index = hl.arange(16)
        valid = index < raw.shape[1]
        scalar = hl.load(raw, [batch.begin, index], extra_mask=valid).float()
        weights = torch.where(valid, torch.sigmoid(scalar), 0.0)
        base = hl.dot(
            (a[batch.begin, :, :].float() * weights[:, None]).to(a.dtype),
            b[batch.begin, :, :],
        )
        state = base
        for position in hl.static_range(1, 3):
            row = torch.where(index[:, None] == position, state, 0.0).sum(0)
            column = torch.where(index[None, :] == position, base, 0.0).sum(1)
            state = torch.tanh(state * 0.5 + column[:, None] * row[None, :] + 0.0625)
        output[batch.begin, :, :] = hl.dot(
            (state * weights[:, None]).to(a.dtype),
            hl.load(rhs, [batch.begin, index, hl.arange(32)]),
        ).to(a.dtype)
        side[batch.begin, :] = weights
    return output, side


def _args():
    return tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((3, 16, 16), (3, 16, 16), (3, 16, 32), (3, 13))
    )


def _config(budget=4096, **kwargs):
    return helion.Config.from_dict(
        {
            "num_warps": 4,
            "cute_chained_mma_schedule": "coalesced",
            "cute_chained_pointwise_schedule": "warp_rows",
            "cute_chained_operand_coordinates": True,
            "cute_chained_pointwise_cache_bytes": budget,
            **kwargs,
        }
    )


def test_actual_public_emission():
    publications = []
    original = vector.RootVectorPublication.finish

    def observe(self, boundaries):
        original(self, boundaries)
        publications.append(self)

    with _cpu_codegen(), patch.object(vector.RootVectorPublication, "finish", observe):
        bound = _sequence._bind_isolated(_args())
        assert bound.config_spec.cute_chained_root_vector_cache_search_enabled
        cached = bound.to_code(_config())
        ordinary = bound.to_code(_config(0))
        absent = _config(0)
        absent.config.pop("cute_chained_pointwise_cache_bytes")
        assert bound.to_code(absent) == ordinary
    assert len(publications) == 1
    publication = publications[0]
    assert publication.cache.extent == 16
    assert publication.cache.allocated_bytes == 128
    assert cached.count("cute.math.rcp(") == 1
    assert ordinary.count("cute.math.rcp(") > 1
    assert cached.count("sync_threads()") == ordinary.count("sync_threads()") + 1
    assert cached.count("cute.gemm(") == ordinary.count("cute.gemm(")
    assert cached.count("alloc_smem(") == ordinary.count("alloc_smem(") + 1
    ast.parse(cached)


def test_actual_proof_mutations():
    from helion._compiler.cute.chained_coordinates import CoordinateText
    from helion._compiler.cute.chained_coordinates import constant
    from helion._compiler.cute.chained_execution import ChainedExecution
    from helion.language import memory_ops

    checked = []
    original = vector.RootVectorPublication.finish

    def observe(pub, boundaries):
        original(pub, boundaries)
        cache, plan = pub.cache, pub.plan

        def reject(label, check, mutate, restore):
            mutate()
            try:
                with pytest.raises(_UnsupportedChain):
                    check()
            finally:
                restore()
            original(pub, boundaries)
            checked.append(label)

        def field(label, obj, name, replacement, check=None):
            old = getattr(obj, name)
            reject(
                label,
                check or pub.check,
                lambda: object.__setattr__(obj, name, replacement),
                lambda: object.__setattr__(obj, name, old),
            )

        raw = next(n.args[0] for n in cache.closure if n.target is memory_ops.load)
        value = raw.meta["val"]
        for label, replacement in (
            ("foreign allocation", value.clone()),
            (
                "same storage shifted view",
                value.as_strided(value.shape, value.stride(), 1),
            ),
        ):
            reject(
                label,
                pub.check,
                lambda v=replacement: raw.meta.__setitem__("val", v),
                lambda: raw.meta.__setitem__("val", value),
            )
        offset = value.storage_offset()
        storage, shape, stride = value.untyped_storage(), value.shape, value.stride()
        reject(
            "in-place storage offset",
            pub.check,
            lambda: value.set_(storage, offset + 1, shape, stride),
            lambda: value.set_(storage, offset, shape, stride),
        )
        field("host argument", raw, "args", ("foreign",))
        field("producer operator", cache.node, "target", torch.ops.aten.neg.default)
        field("root phase", cache.root, "phase_index", cache.root.phase_index + 1)
        field("extent", cache, "extent", cache.extent - 1)
        field("storage budget", cache, "allocated_bytes", 0)
        field("alias map identity", plan, "tensor_aliases", dict(plan.tensor_aliases))
        assert pub.alias_items
        key, old = pub.alias_items[0]
        reject(
            "existing alias replacement",
            pub.check,
            lambda: pub.aliases.__setitem__(key, "foreign"),
            lambda: pub.aliases.__setitem__(key, old),
        )
        reject(
            "existing alias deletion",
            pub.check,
            lambda: pub.aliases.pop(key),
            lambda: pub.aliases.__setitem__(key, old),
        )
        pub.aliases["extra_name"] = "extra_alias"
        original(pub, boundaries)
        pub.aliases.pop("extra_name")
        field("team", pub, "execution", ChainedExecution(32))
        field("publication identity", pub, "plan", replace(plan))
        old_line = pub.lines[len(pub.prefix) - 1]
        reject(
            "removed publication barrier",
            pub.check,
            lambda: pub.lines.__setitem__(len(pub.prefix) - 1, "pass"),
            lambda: pub.lines.__setitem__(len(pub.prefix) - 1, old_line),
        )
        binding = boundaries[cache.node]
        reject(
            "untyped published name",
            lambda: original(pub, boundaries),
            lambda: boundaries.__setitem__(cache.node, str(binding)),
            lambda: boundaries.__setitem__(cache.node, binding),
        )
        with pytest.raises(_UnsupportedChain):
            vector.RootVectorPublication(
                pub.codegen, plan, pub.lines, cache, pub.program, object()
            )
        for read in pub.reads:
            if read.context is None:
                field("export scope", read.origin, "prefix", "foreign", read.check)
                continue
            ctx = read.context
            field(
                "context identity",
                read.expression,
                "coordinate_normalization",
                type(ctx)(ctx.domain),
                read.check,
            )
            field("domain identity", ctx, "domain", replace(ctx.domain), read.check)
            field("scope identity", read, "origin", replace(read.origin), read.check)
            key, old_term = read.context_seed[0]
            reject(
                "seed coordinate",
                read.check,
                lambda ctx=ctx, key=key: ctx.expressions.__setitem__(key, constant(0)),
                lambda ctx=ctx, key=key, old_term=old_term: ctx.expressions.__setitem__(
                    key, old_term
                ),
            )
            if read.reads:
                coord, original_coord, term, original_term = read.reads[0]
                previous = ctx.expressions.get(original_coord)
                if previous is not None:
                    reject(
                        "consumed original coordinate",
                        read.check,
                        lambda ctx=ctx, original_coord=original_coord: (
                            ctx.expressions.__setitem__(
                                original_coord, constant(cache.extent)
                            )
                        ),
                        lambda ctx=ctx, original_coord=original_coord, previous=previous: (
                            ctx.expressions.__setitem__(original_coord, previous)
                        ),
                    )
                with pytest.raises(_UnsupportedChain):
                    read.read(binding, cache.node, (CoordinateText(coord, "foreign"),))
            with pytest.raises(_UnsupportedChain):
                read.read(binding, cache.node, (str(cache.extent),))
            with pytest.raises(_UnsupportedChain):
                read.read(vector.PublishedRootVector(pub), cache.node, ("0",))
        original(pub, boundaries)

    with _cpu_codegen(), patch.object(vector.RootVectorPublication, "finish", observe):
        _sequence._bind_isolated(_args()).to_code(_config())
    assert len(checked) >= 22


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_chained_operand_coordinates": False},
        {"cute_chained_warp_register_a": True},
        {"cute_chained_pointwise_cache_entries": 2},
        {"cute_chained_pointwise_cache_nested": True},
        {"cute_chained_pointwise_cache_collectives": True},
        {"cute_chained_group_contractions": True},
    ],
)
def test_unproved_combinations_remain_rejected(extra):
    with _cpu_codegen():
        bound = _sequence._bind_isolated(_args())
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(_config(**extra))


def test_probe_boundaries_preserve_other_owners():
    from types import SimpleNamespace

    from torch.fx import Graph

    graph = Graph()
    root, other = graph.placeholder("root"), graph.placeholder("other")
    binding = vector.PublishedRootVector(
        cast(
            "vector.RootVectorPublication",
            SimpleNamespace(cache=SimpleNamespace(name="cache")),
        )
    )
    actual = {root: binding, other: "original_image"}
    probes = vector.original_probe_boundaries(actual)
    assert probes == {other: "original_image"}
    assert vector.executable_boundaries(probes, actual) == actual


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _one_stage_vector(a, b, rhs, raw):
    output = torch.empty((a.shape[0], 16, 32), device=a.device, dtype=a.dtype)
    side = torch.empty((a.shape[0], 16), device=a.device, dtype=torch.float32)
    for batch in hl.tile(a.shape[0], block_size=1):
        index = hl.arange(16)
        scalar = hl.load(raw, [batch.begin, index]).float()
        weights = torch.sigmoid(scalar)
        left = a[batch.begin, :, :].float() * weights[:, None]
        right = b[batch.begin, :, :].float() * weights[None, :]
        base = hl.dot(left.to(a.dtype), right.to(a.dtype))
        state = base
        for position in hl.static_range(1, 3):
            row = torch.where(index[:, None] == position, state, 0.0).sum(0)
            column = torch.where(index[None, :] == position, base, 0.0).sum(1)
            state = state * 0.5 + column[:, None] * row[None, :] + 0.0625
        output[batch.begin, :, :] = hl.dot(
            state[index, :].to(a.dtype),
            hl.load(rhs, [batch.begin, index, hl.arange(32)]),
        ).to(a.dtype)
        side[batch.begin, :] = weights
    return (output, side)


def test_root_only_field_preserves_prefix_and_roundtrip():
    from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
    from helion.autotuner.config_fragment import EnumFragment

    with _cpu_codegen():
        args = tuple(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((3, 16, 16), (3, 16, 16), (3, 16, 32), (3, 16))
        )
        bound = _one_stage_vector._bind_isolated(args)
        spec = bound.config_spec
        host = bound.host_function
        assert host is not None
        assert spec.cute_chained_root_vector_cache_search_enabled
        assert spec.cute_chained_row_register_search_enabled
        assert not spec.cute_chained_pointwise_residency_search_enabled
        assert not spec.cute_chained_pointwise_collective_search_enabled
        with patch.object(spec, "cute_chained_root_vector_cache_search_enabled", False):
            old_fields = spec._flat_fields()
            with bound.env, host:
                old_seeds = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, host.device_ir
                )
        fields = spec._flat_fields()
        key = "cute_chained_pointwise_cache_bytes"
        appended_rows = [
            "cute_chained_row_panel_cache",
            "cute_chained_row_subwarp_packing",
            "cute_chained_row_frontier",
        ]
        old_order = list(old_fields)
        row_start = old_order.index(appended_rows[0])
        assert old_order[row_start : row_start + len(appended_rows)] == appended_rows
        assert list(fields) == [*old_order[:row_start], key, *old_order[row_start:]]
        assert [name for name in fields if name != key] == list(old_fields)
        budget_field = fields[key]
        assert isinstance(budget_field, EnumFragment)
        assert budget_field.choices == (0, 4096, 16384)
        assert budget_field.default() == 0
        with bound.env, host:
            assert (
                CuteChainedMatmulHeuristic.get_seed_configs(bound.env, host.device_ir)
                == old_seeds
            )
        generation = spec.create_config_generation()
        for budget in (0, 4096, 16384):
            config = spec.normalized_config(_config(budget))
            assert generation.unflatten(generation.flatten(config)) == config
        source = bound.to_code(_config())
        assert source.count("cute.math.rcp(") == 1
        assert "chain_root_vector_0" in source
        zero = bound.to_code(_config(0))
        implicit = _config(0)
        implicit.config.pop(key)
        assert bound.to_code(implicit) == zero
        assert "chain_root_vector_0" not in zero
