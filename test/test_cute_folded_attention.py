from __future__ import annotations

import ast
import dataclasses
import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.analysis.separable_index_map import prove_separable_index_map
from helion._compiler.cute.cute_mma_dag import _Matcher
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_paged_attention
from helion._testing.cute_paged_v_lag import without_retirement_seeds
import helion.language as hl
from helion.language import tile_ops
from helion.language.memory_ops import load

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_folded_gather(
    storage: torch.Tensor,
    weight: torch.Tensor,
    row_begin: int,
    item_begin: int,
) -> torch.Tensor:
    row_begin = hl.specialize(row_begin)
    item_begin = hl.specialize(item_begin)
    output = torch.empty((2, 40, 128), dtype=storage.dtype, device=storage.device)
    for item, rows in hl.tile(
        [item_begin, row_begin], [item_begin + 2, row_begin + 40], block_size=[1, None]
    ):
        channel = hl.arange(128)
        offset = (
            item.begin * 16384
            + (rows.index // 8)[:, None] * 2048
            + (rows.index % 8)[:, None] * 128
            + channel[None, :]
        )
        value = storage[offset]
        output[item.id - item_begin, rows.index - row_begin, :] = hl.dot(
            value, weight[:, :]
        )
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _shifted_folded_iota_gather(
    storage: torch.Tensor, weight: torch.Tensor, item_begin: int
) -> torch.Tensor:
    item_begin = hl.specialize(item_begin)
    output = torch.empty((2, 64, 128), dtype=storage.dtype, device=storage.device)
    for item in hl.tile(item_begin, item_begin + 2, block_size=1):
        rows = hl.arange(64)
        channel = hl.arange(128)
        offset = (
            item.begin * 16384
            + (rows // 8)[:, None] * 2048
            + (rows % 8)[:, None] * 128
            + channel[None, :]
        )
        output[item.id - item_begin, :, :] = hl.dot(storage[offset], weight[:, :])
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _nonlead_folded_gather(storage: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty((64, 128), dtype=storage.dtype, device=storage.device)
    for rows in hl.tile(64):
        channel = hl.arange(128)
        offset = (
            rows.begin * 256
            + (rows.index // 8)[:, None] * 2048
            + (rows.index % 8)[:, None] * 128
            + channel[None, :]
        )
        output[rows, :] = hl.dot(storage[offset], weight[:, :])
    return output


@functools.lru_cache(None)
def _bound(query_len: int = 5, variant: str = "plain") -> BoundKernel:
    batch, heads, dim, pages = 3, 16, 128, 7
    column_stride = 2 if variant == "strided_columns" else 1
    head_stride = dim * column_stride
    query_stride = 4608
    query_batch = query_len * query_stride
    offset = 8 if variant == "padded_output" else 0
    q = torch.empty(query_batch * batch + offset, dtype=torch.bfloat16).as_strided(
        (batch, query_len, heads, dim),
        (query_batch, query_stride, head_stride, column_stride),
        offset,
    )
    out_query = 1024 if variant == "overlap" else 2048
    if variant == "padded_output":
        out_query += 128
    out_batch = query_len * out_query + (256 if variant == "padded_output" else 0)
    out = torch.empty(out_batch * batch + 2048 + offset, dtype=q.dtype).as_strided(
        q.shape, (out_batch, out_query, 128, 1), offset
    )
    storage = [torch.empty(pages * 33792, dtype=torch.int8) for _ in range(2)]
    k, v = [x.as_strided((pages, 128, 2, 128), (33792, 256, 128, 1)) for x in storage]
    ks, vs = [
        x.view(torch.uint8).as_strided((pages, 128, 2, 4), (33792, 4, 512, 1), 32768)
        for x in storage
    ]
    args = (
        q,
        k,
        v,
        ks,
        vs,
        torch.empty((batch, 17), dtype=torch.int32),
        torch.empty(batch, dtype=torch.int32),
        31,
        128,
        True,
        out,
    )
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(folded_paged_attention, args)


def _config(
    query_len: int = 5, mode: str = "tcgen05", schedule: str = "attention"
) -> helion.Config:
    return helion.Config(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps=schedule,
        cute_min_blocks_per_mp=1,
        cute_pdl=False,
    )


@pytest.mark.parametrize("query_len,variant", [(5, "plain"), (1, "padded_output")])
def test_folded_attention_emits_one_direct_kernel(query_len: int, variant: str) -> None:
    bound = _bound(query_len, variant)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(_config(query_len))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paged_attention_partial_body"
    ]
    assert len(calls) == 1
    layouts = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "io_layouts"
    )
    query, output, lse = ast.literal_eval(layouts)
    assert query == (query_len * 4608, 4608, 128, 1)
    out_query = 2176 if variant == "padded_output" else 2048
    assert output == (
        query_len * out_query + (256 if variant == "padded_output" else 0),
        out_query,
        128,
        1,
    )
    assert lse == (query_len * 16, 16, 1)
    assert source.count("@cute.kernel") == 1
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.mapped_io


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_folded_attention_rejects_ordinary_mma_routes(mode: str) -> None:
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(
            (helion.exc.InvalidConfig, helion.exc.BackendUnsupported),
            match="folded storage",
        ),
    ):
        _bound(5, "plain").to_code(_config(mode=mode, schedule="off"))


@pytest.mark.parametrize("variant", ["overlap", "strided_columns"])
def test_folded_attention_declines_unimplemented_physical_layouts(variant: str) -> None:
    assert _bound(5, variant).config_spec.cute_mma_dag_facts is None


def test_folded_attention_rejects_shifted_output_request_head_map() -> None:
    bound = _bound(5, "plain")
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None and semantic.output.index_map is not None
        output_map = semantic.output.index_map
        shifted = dataclasses.replace(output_map, base=output_map.base + 128)
        semantic = dataclasses.replace(
            semantic, output=dataclasses.replace(semantic.output, index_map=shifted)
        )
        assert plan_paged_attention_transport(bound.env, semantic) is None


def test_folded_attention_exposes_only_implemented_mma_seeds() -> None:
    bound = _bound(5, "plain")
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.mapped_io
    assert facts.attention is not None
    with bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        seeds = without_retirement_seeds(seeds)
        assert [
            (
                seed.get("cute_mma_dag_pass_warps"),
                seed.get("cute_min_blocks_per_mp"),
                seed.get("cute_mma_dag_v_lag", 0),
                seed.get("cute_mma_dag_sleep_waits", False),
                seed.get("cute_mma_dag_query_tma", False),
                tuple(
                    seed.get("load_eviction_policies")[slot]
                    for slot in facts.attention.tma_load_slots
                ),
                seed.get("cute_pdl", False),
                seed.get("cute_pdl_capture_overlap", False),
                seed.get("cute_pdl_release", "auto"),
            )
            for seed in seeds
        ] == [
            (*schedule, False, False, (policy,) * 4, False, False, "auto")
            for schedule in [
                ("attention", 1, 0),
                ("attention", 2, 0),
                ("attention_scoped", 2, 0),
                *(("attention_paired", 1, lag) for lag in range(4)),
            ]
            for policy in ("", "streaming", "l2_last")
        ] + [
            ("attention_paired", 1, lag, True, False, ("",) * 4, False, False, "auto")
            for lag in range(4)
        ] + [
            ("attention_paired", 1, lag, False, True, ("",) * 4, False, False, "auto")
            for lag in range(4)
        ]
        for seed in seeds:
            config = bound.config_spec.normalized_config(seed)
            assert not seed.get("cute_pdl_capture_overlap", False)
            assert config.get("cute_pdl_capture_overlap", False) is False
            assert config.get("cute_pdl_release", "auto") == seed.get(
                "cute_pdl_release", "auto"
            )
            assert config.get("cute_mma_dag_query_tma", False) == seed.get(
                "cute_mma_dag_query_tma", False
            )
            assert config.get("cute_mma_dag_sleep_waits", False) == seed.get(
                "cute_mma_dag_sleep_waits", False
            )
            assert config.get("cute_mma_dag_pass_warps") == seed.get(
                "cute_mma_dag_pass_warps"
            )
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, config
            )


def test_folded_index_proof_checks_intermediate_integer_overflow() -> None:
    bound = _bound(5, "plain")
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        matcher = _Matcher(bound.host_function.device_ir.graphs)
        program = matcher.match()
        assert program is not None
        semantic = match_paged_attention(program)
        assert semantic is not None
        index = semantic.query.node.args[1][0]

        def rewritten_index(inject_overflow: bool) -> torch.fx.Node:
            graph = torch.fx.Graph()
            copied: dict[torch.fx.Node, torch.fx.Node] = {}

            def clone(node: torch.fx.Node) -> torch.fx.Node:
                if node not in copied:
                    result = graph.node_copy(node, clone)
                    result.meta = dict(node.meta)
                    if inject_overflow and node.target is tile_ops.tile_index:
                        product = graph.call_function(
                            torch.ops.aten.mul.Tensor, (result, 2**30)
                        )
                        product.meta = dict(node.meta)
                        result = graph.call_function(
                            torch.ops.aten.div.Tensor_mode,
                            (product, 2**30),
                            {"rounding_mode": "floor"},
                        )
                        result.meta = dict(node.meta)
                    copied[node] = result
                return copied[node]

            return clone(index)

        assert (
            prove_separable_index_map(
                matcher, rewritten_index(False), semantic.query.fake.numel()
            )
            is not None
        )
        # Algebraically unchanged coordinates still overflow in the original
        # Int32 intermediate. Simplifying the full expression first is unsafe.
        assert (
            prove_separable_index_map(
                matcher, rewritten_index(True), semantic.query.fake.numel()
            )
            is None
        )


@pytest.mark.parametrize("row_begin,item_begin", [(8, 0), (0, 1)])
def test_folded_index_proof_rejects_nonzero_task_origins(
    row_begin: int, item_begin: int
) -> None:
    # These are real shifted source loops, with enough backing storage that
    # every original address remains valid. Extent alone is not a coordinate.
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        row_begin,
        item_begin,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shifted_folded_gather, args)
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            assert ir.noncanonical_task_origin_block_ids
            matcher = _Matcher(ir.graphs)
            # Initialize matcher shape/origin tables even when the complete
            # non-attention GEMM does not match the specialized consumer.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            assert prove_separable_index_map(matcher, indices[0], 65536) is None


def test_folded_index_proof_rejects_shifted_lead_with_canonical_iotas() -> None:
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        1,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_shifted_folded_iota_gather, args)
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            matcher = _Matcher(ir.graphs)
            # The complete matcher initializes the coordinate tables used by
            # the focused address proof; full GEMM admission is not required.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            iotas = [
                node
                for graph in ir.graphs
                for node in graph.graph.nodes
                if node.target is torch.ops.prims.iota.default
            ]
            assert len(iotas) == 2
            for node in iotas:
                (axis,) = matcher.shape_axes(node)
                assert axis is not None
                assert axis.block_id not in ir.noncanonical_task_origin_block_ids
            # Only the uniform leading item has a shifted origin, so this
            # specifically exercises bounds(), not axis_expression().
            assert ir.noncanonical_task_origin_block_ids
            assert prove_separable_index_map(matcher, indices[0], 65536) is None


def test_folded_index_proof_declines_nonleading_uniform_base() -> None:
    args = (
        torch.empty(65536, dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_nonlead_folded_gather, args)
        with bound.env, bound.host_function:
            matcher = _Matcher(bound.host_function.device_ir.graphs)
            # Nonleading tile.begin is intentionally outside the map contract.
            matcher.match()
            indices = [
                node.args[1][0]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is load
                and isinstance(node.args[1], (tuple, list))
                and len(node.args[1]) == 1
                and isinstance(node.args[1][0], torch.fx.Node)
                and node.args[1][0].meta["val"].ndim == 2
            ]
            assert len(indices) == 1
            assert prove_separable_index_map(matcher, indices[0], 65536) is None
