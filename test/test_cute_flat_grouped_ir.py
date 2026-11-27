from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.promote_output_axis import _PromotedGridAxisGraphInfo
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import skipUnlessBackends
from helion._testing.cute_flat_grouped import bound_host_function as _host
from helion._testing.cute_flat_grouped import cuda_binding
from helion.language import _tracing_ops

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

META_DEVICE = "meta"


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("scalar_group", [False, True])
def test_actual_cuda_traced_typed_program_is_proved(
    static: bool, bits: int, scalar_group: bool
) -> None:
    with cuda_binding(static=static, bits=bits, scalar_group=scalar_group) as bound:
        proof = prove_flat_grouped_rna(bound.env, _host(bound).device_ir)
        assert proof is not None
        assert proof.offset_bits == bits
        assert (
            proof.offsets_argument,
            proof.a_argument,
            proof.b_argument,
            proof.bias_argument,
        ) == (
            "routing",
            "left",
            "right",
            "addend",
        )
        assert (
            len(
                {
                    proof.group_axis,
                    proof.row_axis,
                    proof.column_axis,
                    proof.reduction_axis,
                }
            )
            == 4
        )
        assert proof.a.device.type == proof.b.device.type == "cuda"
        if not static:
            assert (
                proof.a.requires_grad
                and proof.b.requires_grad
                and proof.bias.requires_grad
            )


@pytest.mark.parametrize("static", [False, True])
def test_original_large_cuda_metadata_static_and_dynamic_are_separate(
    static: bool,
) -> None:
    with cuda_binding(
        static=static, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        ir = _host(bound).device_ir
        proof = prove_flat_grouped_rna(bound.env, ir)
        assert proof is not None and proof.offset_bits == 64
        assert len(ir.grid_block_ids[0]) == (2 if static else 1)
        assert (
            proof.offsets_argument,
            proof.a_argument,
            proof.b_argument,
            proof.bias_argument,
        ) == ("seq_offsets", "jagged", "dense", "bias")


@pytest.mark.parametrize("change", ["extra_axis", "grid_order", "loop_bound"])
def test_promoted_column_requires_exact_empty_scope_and_grid_extent(
    change: str,
) -> None:
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        ir = _host(bound).device_ir
        proof = prove_flat_grouped_rna(bound.env, ir)
        assert proof is not None and len(ir.grid_block_ids[0]) == 2
        if change in ("extra_axis", "grid_order"):
            original = ir.grid_block_ids[0]
            try:
                ir.grid_block_ids[0] = (
                    [*original, proof.row_axis]
                    if change == "extra_axis"
                    else list(reversed(original))
                )
                assert prove_flat_grouped_rna(bound.env, ir) is None
            finally:
                ir.grid_block_ids[0] = original
        else:
            promoted = next(
                info
                for info in ir.graphs
                if isinstance(info, _PromotedGridAxisGraphInfo)
            )
            call = next(
                node
                for info in ir.graphs
                for node in info.graph.find_nodes(
                    op="call_function", target=_tracing_ops._for_loop
                )
                if node.args[0] == promoted.graph_id
            )
            original_args = call.args
            try:
                call.args = (original_args[0], [0], [132], original_args[3])
                assert prove_flat_grouped_rna(bound.env, ir) is None
            finally:
                call.args = original_args
        assert prove_flat_grouped_rna(bound.env, ir) is not None


@pytest.mark.parametrize(
    "mode",
    [
        "narrow_rows",
        "reset_carry",
        "wrong_stride",
        "shift_bias",
        "extra_math",
        "partial_groups",
    ],
)
def test_frontend_counterexamples_retain_original_lowering(mode: str) -> None:
    with cuda_binding(mode=mode) as bound:
        assert prove_flat_grouped_rna(bound.env, _host(bound).device_ir) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_other_numerical_domains_are_not_rna_admission(dtype: torch.dtype) -> None:
    with cuda_binding(dtype=dtype) as bound:
        assert prove_flat_grouped_rna(bound.env, _host(bound).device_ir) is None


@pytest.mark.parametrize(
    "change", ["beta", "literal_seed", "negative_zero", "output_alias", "bad_view"]
)
def test_exact_ir_and_host_dependency_mutations_decline(change: str) -> None:
    with cuda_binding() as bound:
        ir = _host(bound).device_ir
        proof = prove_flat_grouped_rna(bound.env, ir)
        assert proof is not None
        mm = proof.contraction_node
        info = next(info for info in ir.graphs if info.graph is mm.graph)
        assert isinstance(info, ForLoopGraphInfo)
        zero = info.node_args[1]
        if change == "beta":
            original = mm.kwargs
            try:
                mm.kwargs = {"beta": 0.0}
                assert prove_flat_grouped_rna(bound.env, ir) is None
            finally:
                mm.kwargs = original
        elif change == "literal_seed":
            original_args = mm.args
            try:
                mm.args = (zero, *mm.args[1:])
                assert prove_flat_grouped_rna(bound.env, ir) is None
            finally:
                mm.args = original_args
        elif change == "negative_zero":
            original_args = zero.args
            try:
                zero.args = (zero.args[0], -0.0, *zero.args[2:])
                assert prove_flat_grouped_rna(bound.env, ir) is None
            finally:
                zero.args = original_args
        else:
            assignment = next(
                node
                for node in _host(bound).body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == proof.output_flat_expression
            )
            replacement = ast.parse(
                "left.view(-1)"
                if change == "output_alias"
                else "torch.empty((rows, columns)).view(unknown)",
                mode="eval",
            ).body
            with patch.object(assignment, "value", replacement):
                assert prove_flat_grouped_rna(bound.env, ir) is None
        assert prove_flat_grouped_rna(bound.env, ir) is not None
