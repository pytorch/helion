from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.promote_output_axis import _PromotedGridAxisGraphInfo
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops

pytestmark = skipUnlessBackends(["cute"])

META_DEVICE = "meta"

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel


def _host(bound: BoundKernel) -> HostFunction:
    host = bound.host_function
    assert host is not None
    return host


def flat_contract(
    routing: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    addend: torch.Tensor,
    mode: hl.constexpr,
    scalar_group: hl.constexpr,
) -> torch.Tensor:
    rows, reduction = left.shape
    groups, reduction, columns = right.shape
    if mode == "partial_groups":
        groups = groups - 1
    left = left.view(-1)
    result = torch.empty((rows, columns), dtype=right.dtype, device=right.device).view(
        -1
    )
    for group in hl.tile(groups, block_size=1 if scalar_group else None):
        begin = routing[group]
        end = routing[group.index + 1]
        length = end - begin
        for row in hl.jagged_tile(length):
            raw_indices = begin[:, None] + row.index[None, :]
            if mode == "narrow_rows":
                indices = raw_indices.to(torch.int32)
            else:
                indices = raw_indices
            for column in hl.tile(0, columns):
                carry = hl.zeros(
                    [group, row, column], dtype=torch.float32, device=right.device
                )
                for inner in hl.tile(0, reduction):
                    if mode == "reset_carry":
                        carry = hl.zeros(
                            [group, row, column],
                            dtype=torch.float32,
                            device=right.device,
                        )
                    if mode == "wrong_stride":
                        x = hl.load(
                            left,
                            [
                                indices[:, :, None] * (reduction + 4)
                                + inner.index[None, None, :]
                            ],
                        )
                    else:
                        x = hl.load(
                            left,
                            [
                                indices[:, :, None] * reduction
                                + inner.index[None, None, :]
                            ],
                        )
                    y = right[group, inner, column]
                    carry = carry + torch.matmul(x, y)
                if mode == "shift_bias":
                    z = addend[group, column.index + 1]
                else:
                    z = addend[group, column]
                carry = carry + z.unsqueeze(1)
                if mode == "extra_math":
                    carry = carry * 2.0
                hl.store(
                    result,
                    [indices[:, :, None] * columns + column.index[None, None, :]],
                    carry,
                )
    return result.reshape(rows, columns)


def _cuda_fake(backend: object, tensor: torch.Tensor) -> torch.Tensor:
    assert isinstance(tensor, FakeTensor)
    with torch._C._DisableTorchDispatch():
        meta = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=META_DEVICE,
            requires_grad=tensor.requires_grad,
        )
    return FakeTensor(tensor.fake_mode, meta, torch.device("cuda:0"))


@contextmanager
def cuda_binding(
    *,
    static: bool = False,
    bits: int = 64,
    mode: str = "ordinary",
    groups: int = 3,
    rows: int = 273,
    k: int = 64,
    n: int = 132,
    dtype: torch.dtype = torch.float32,
    scalar_group: bool = False,
    original: bool = False,
    misalign: int | None = None,
    offsets_stride: int = 1,
    dot_precision: str = "tf32",
) -> Iterator[BoundKernel]:
    inputs = [
        torch.empty_strided(
            (groups + 1,),
            (offsets_stride,),
            dtype=torch.int32 if bits == 32 else torch.int64,
        ),
        torch.empty((rows, k), dtype=dtype, requires_grad=True),
        torch.empty((groups, k, n), dtype=dtype, requires_grad=True),
        torch.empty((groups, n), dtype=dtype, requires_grad=True),
    ]
    if misalign is not None:
        original_input = inputs[misalign]
        assert isinstance(original_input, torch.Tensor)
        shifted = torch.empty(original_input.numel() + 1, dtype=original_input.dtype)[
            1:
        ].view(original_input.shape)
        shifted.requires_grad_(original_input.requires_grad)
        inputs[misalign] = shifted
    flags = {
        f"HELION_CUTE_{name}": "1"
        for name in (
            "REGION_FISSION",
            "FULL_SLICE_MATMUL_TILING",
            "SEGMENTED_MATMUL_TILING",
            "FLATTEN_NESTED_REDUCTIONS",
            "MATERIALIZE_TRANSFORMED_OPERANDS",
        )
    }
    with (
        _target(),
        patch.dict(os.environ, flags),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        _mock_cuda_unavailable(),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=_cuda_fake,
        ),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda",
                index=0,
                cc=100,
                warp_size=32,
                multi_processor_count=148,
            ),
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        function = flat_contract
        arguments = (*inputs, mode, scalar_group)
        if original:
            from examples.jagged_dense_bmm import jagged_dense_bmm

            function = jagged_dense_bmm.fn
            arguments = tuple(inputs)
        kernel = helion.kernel(
            function,
            backend="cute",
            static_shapes=static,
            autotune_effort="none",
            dot_precision=dot_precision,
        )
        bound = kernel.bind(arguments)
        assert all(tensor.requires_grad for tensor in inputs[1:4])
        assert bound.env.device.type == "cuda"
        assert all(tensor.device.type == "cuda" for tensor in bound.fake_args[:4])
        with bound.env, _host(bound):
            yield bound


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
