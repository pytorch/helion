"""The BF16 reduction uses its actual page table and complete K fold."""

from __future__ import annotations

import ast
import copy

import pytest
import torch

import helion
from helion._compiler.cute.analysis.gathered_bf16_mma import gathered_phase_region
from helion._compiler.cute.analysis.gathered_bf16_mma import prove_gathered_bf16_mma
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import default_cute_mma_support
from helion._testing.cute_gathered_bf16 import routed_transposed_product
from helion._testing.cute_gathered_bf16 import split_paged_attention

pytestmark = pytest.mark.cpu_only


def _bind(
    seed: float = 0,
    dtype: torch.dtype = torch.bfloat16,
    page_shift: int = 0,
    columns: int = 32,
    depth: int = 80,
):
    return cpu_bind(
        routed_transposed_product,
        (
            torch.empty((32, depth), dtype=dtype),
            torch.empty((5, columns, depth), dtype=dtype),
            torch.empty((7,), dtype=torch.int64),
            seed,
            page_shift,
            False,
        ),
    )


def test_native_gathered_reduction() -> None:
    with cpu_target():
        bound = _bind()
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            proofs = [
                result
                for info in ir.graphs
                for node in info.graph.nodes
                if (result := prove_gathered_bf16_mma(node, ir.graphs, bound.env))
            ]
            assert len(proofs) == 1
            assert (proofs[0].rows, proofs[0].columns, proofs[0].depth) == (32, 32, 80)
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": True,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        source = bound.to_code(helion.Config.from_dict(config))
        assert "gathered_mma_page" in source
        assert "_helion_fragment_mma.mma_bf16(" in source


@pytest.mark.parametrize("enabled", [True, False])
def test_checked_gather_keeps_unmasked_page_bounds(enabled: bool) -> None:
    with cpu_target():
        bound = cpu_bind(
            routed_transposed_product,
            (
                torch.empty((32, 80), dtype=torch.bfloat16),
                torch.empty((5, 32, 80), dtype=torch.bfloat16),
                torch.full((7,), 5, dtype=torch.int64),
                0,
                0,
                True,
            ),
        )
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": enabled,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        with pytest.raises(helion.exc.BackendUnsupported, match="not proved in bounds"):
            bound.to_code(helion.Config.from_dict(config))


def test_wide_page_reduction_rolls_k_instead_of_unrolling_the_full_axis() -> None:
    with cpu_target():
        bound = _bind(depth=65536)
        config = helion.Config(
            block_sizes=bound.config_spec.default_config().block_sizes,
            cute_fragment_half_mma=True,
            cute_mma_dag="off",
        )
        source = bound.to_code(config)
        assert "gathered_mma_page" in source
        assert "for fragment_mma_step in range(4096)" in source


@pytest.mark.parametrize(
    "seed,dtype,page_shift,columns",
    [
        (1.0, torch.bfloat16, 0, 32),
        (0.0, torch.float16, 0, 32),
        (0.0, torch.bfloat16, 1, 32),
        (0.0, torch.bfloat16, 0, 24),
    ],
)
def test_unproved_reduction_falls_back(
    seed: float, dtype: torch.dtype, page_shift: int, columns: int
) -> None:
    with cpu_target():
        bound = _bind(seed, dtype, page_shift, columns)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            assert not any(
                prove_gathered_bf16_mma(node, graphs, bound.env)
                for graph in graphs
                for node in graph.graph.nodes
            )
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": True,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        assert "gathered_mma_page" not in bound.to_code(helion.Config.from_dict(config))


def test_ordered_producer_keeps_the_combines_launch_and_barrier() -> None:
    with torch.device("meta"):
        inputs = (
            torch.empty((1, 1, 16, 576), dtype=torch.bfloat16),
            torch.empty((1024, 1, 64, 576), dtype=torch.bfloat16),
            torch.empty((1, 1024), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            576**-0.5,
        )
    with cpu_render_target(
        fake_device_sm_count=148,
        mma_support=default_cute_mma_support(),
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    ):
        kernel = helion.kernel(backend="cute", static_shapes=False)(
            split_paged_attention
        )
        bound = cpu_bind(kernel, inputs)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            producer = gathered_phase_region(ir, bound.env, ir.graphs)
            assert producer is not None
            assert bound.config_spec.cute_fragment_half_mma_root_ids == {ir.root_ids[0]}
            assert ir.root_ids[1] not in {g.graph_id for g in producer}
            for attribute, value in (
                ("implicit_dependency_starts", frozenset()),
                ("root_ids", list(reversed(ir.root_ids))),
                ("tile_dependency_graph", None),
            ):
                changed = copy.copy(ir)
                setattr(changed, attribute, value)
                assert gathered_phase_region(changed, bound.env, ir.graphs) is None
        group = next(
            g
            for g in bound.config_spec.compiler_coverage_groups
            if g.key == "cute_fragment_half_mma"
        )
        witness = next(w for w in group.witnesses if w.value is True)
        config = witness.carrier.config | {group.key: witness.value}
        assert (
            tuple(config["num_threads"])
            in bound.config_spec.cute_gathered_phase_search_lanes
        )
        source = bound.to_code(helion.Config.from_dict(config))
        assert source.count("_launcher(") == 1
        assert source.count("_cute_grid_barrier(x_grid_sem.iterator)") == 1
        assert "block=(64, 2, 1)" in source
        assert "q2d_stride_1" in source and "kv3_stride_2" in source
        # Every warp leaving a producer iteration (even on a false predicate)
        # rendezvous before the persistent CTA reuses its local storage.
        assert any(
            isinstance(node, ast.If)
            and node.body
            and ast.unparse(node.body[-1]) == "cute.arch.sync_threads()"
            and any(
                isinstance(n, ast.Name) and n.id == "gathered_mma_page"
                for n in ast.walk(node)
            )
            for node in ast.walk(ast.parse(source))
        )
        assert (
            "gathered_mma_page" in source and "_helion_fragment_mma.mma_bf16(" in source
        )
        conflict = config | {"num_threads": [1, 1, 4, 1, 4, 1, 64]}
        with pytest.raises(
            helion.exc.InvalidConfig, match="needs 128 physical threads"
        ):
            bound.to_code(helion.Config.from_dict(conflict))
