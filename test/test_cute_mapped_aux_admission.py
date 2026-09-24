from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test import test_cute_fragment_epilogue as fragment
from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen
from test.test_cute_mapped_aux_codegen import _inputs
from test.test_cute_mapped_aux_codegen import _mapped_batched_aux
from test.test_cute_mapped_aux_index import (
    test_original_intermediate_proofs_and_no_gpu_initialization as check_index_proofs,
)

import helion
from helion._compiler.cute.aux_tensor import (
    host_function_has_tcgen05_aux_kernel_pattern,
)
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True)
def _fresh_mapped_batched_aux(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    residual: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    batches, rows, inner = lhs.shape
    columns = rhs.shape[-1]
    out = torch.empty_like(residual)
    for bi, mi, ni in hl.tile([batches, rows, columns], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(inner):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        address = (
            bi.index[:, None, None] * rows + mi.index[None, :, None]
        ) * columns + ni.index[None, None, :]
        out[bi, mi, ni] = (acc * weights[address] + residual[bi, mi, ni].float()).to(
            out.dtype
        )
    return out


@pytest.mark.parametrize("reassign", [False, True])
def test_existing_computed_alias_rejection_and_ordinary_fallback(
    reassign: bool,
) -> None:
    captured: list[BoundKernel] = []
    original_bind = Kernel.bind

    def capture(kernel: Kernel, args: tuple[object, ...]) -> BoundKernel:
        bound = original_bind(kernel, args)
        captured.append(bound)
        return bound

    with (
        _cpu_codegen(),
        patch.object(fragment, "DEVICE", "cpu"),
        patch.object(Kernel, "bind", capture),
    ):
        # Execute the original regression, including its unchanged assertion.
        fragment.test_fragment_epilogue_rejects_computed_output_alias(reassign)
        bound = captured[-1]
        assert not bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(
            helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
        )
    assert "def " in source
    assert "tcgen05_aux_direct" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_disjoint_external_mapped_output_still_admitted(dtype: torch.dtype) -> None:
    with _cpu_codegen():
        bound = _mapped_batched_aux._bind_isolated(_inputs(dtype))
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(_config())
    assert source.count("'kind': 'tcgen05_aux_direct'") == 3
    assert source.count("'kind': 'tcgen05_aux_tma'") == 1


def test_fresh_mapped_output_still_admitted() -> None:
    args = (
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 64), dtype=torch.bfloat16),
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((2 * 128 * 64,), dtype=torch.float32),
    )
    with _cpu_codegen():
        bound = _fresh_mapped_batched_aux._bind_isolated(args)
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(_config(block_sizes=[128, 64, 64]))
    assert "'kind': 'tcgen05_aux_direct'" in source


def test_existing_fresh_fragment_output_still_admitted() -> None:
    args = (
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((2, 64, 64), dtype=torch.bfloat16),
        torch.empty((2 * 128 * 8,), dtype=torch.float32),
        torch.empty((2, 128, 64), dtype=torch.bfloat16),
        torch.empty((8,), dtype=torch.bfloat16),
        None,
    )
    with _cpu_codegen():
        bound = fragment._indexed_scale_matmul._bind_isolated(args)
        assert bound.config_spec.cute_tcgen05_search_enabled
        source = bound.to_code(_config(block_sizes=[128, 64, 64]))
    assert "cute.gemm(" in source


@pytest.mark.parametrize("source_name", ["diagonal", "weights"])
def test_known_fx_mapped_alias_cannot_widen_aux_search(source_name: str) -> None:
    with _cpu_codegen():
        bound = _mapped_batched_aux._bind_isolated(_inputs(torch.bfloat16))
        assert bound.host_function is not None
        with bound.env:
            nodes = {
                node.name: node
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
            }
            source = nodes[source_name].meta["val"]
            output = nodes["out"].meta["val"]
            # Real input aliases need not survive FakeTensor conversion. This
            # control explicitly exercises a known FX storage alias; per-call
            # runtime alias guards are covered separately and remain required.
            view = output.view(source.dtype).reshape(-1)
            nodes[source_name].meta["val"] = view[: source.numel()].view(source.shape)
            assert not host_function_has_tcgen05_aux_kernel_pattern(bound.host_function)


@pytest.mark.parametrize("initialized", [False, True])
def test_index_proofs_preserve_preexisting_cuda_state(initialized: bool) -> None:
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        check_index_proofs()


def test_index_proof_detects_initialization_state_change() -> None:
    with (
        patch("torch.cuda.is_initialized", side_effect=[False, True]),
        pytest.raises(AssertionError),
    ):
        check_index_proofs()
