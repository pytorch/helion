from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute.chained_root_exports import classify_root_exports
import helion.language as hl

if TYPE_CHECKING:
    from torch.fx import Node


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_exports(
    a: torch.Tensor, b: torch.Tensor, raw: torch.Tensor, mode: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, valid_rows, width = a.shape
    rows = hl.specialize(b.size(2))
    out = torch.empty((batch, rows, rows), dtype=a.dtype, device=a.device)
    side = torch.empty(
        (width, rows) if mode == "omitted" else (batch, width, rows),
        dtype=torch.int32 if mode == "dtype" else a.dtype,
        device=a.device,
    )
    gate = torch.empty((batch, rows), dtype=torch.float32, device=a.device)
    for bt in hl.tile(batch, block_size=2 if mode == "blocked" else 1):
        bi: Any = bt.id
        idx = hl.arange(rows)
        valid = idx < valid_rows
        if mode == "unmasked":
            left = a[bi, :, :]
        else:
            left = hl.load(a, [bi, idx, hl.arange(width)], extra_mask=valid[:, None])
        values = hl.load(raw, [bi, idx], extra_mask=valid).float()
        if mode == "read_output":
            values = values + side[bi, 0, 0].float()
        prefix = hl.cumsum(values, dim=0)
        result = hl.dot(left, b[bi, :, :])
        out[bi, :, :] = (result * torch.exp2(prefix[:, None])).to(a.dtype)
        transformed = (left.float() * torch.exp2(prefix)[:, None]).to(a.dtype).T
        if mode == "omitted":
            side[:, :] = transformed
        elif mode == "offset":
            side[bi + 1, :, :] = transformed
        elif mode == "masked":
            hl.store(
                side, [bi, slice(None), slice(None)], transformed, extra_mask=bi == 0
            )
        else:
            side[bi, :, :] = transformed
        if mode == "independent_scan":
            gate[bi, :] = hl.cumsum(values * 2, dim=0)
        else:
            gate[bi, :] = torch.exp2(prefix)
    return out, side, gate


def _args(
    dtype: torch.dtype = torch.bfloat16, rows: int = 57, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((3, rows, 64), dtype=dtype, device=device),
        torch.empty((3, 64, 64), dtype=dtype, device=device),
        torch.empty((3, rows), dtype=dtype, device=device),
    )


def _config(schedule: str, metadata: bool = False) -> helion.Config:
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule=schedule,
        cute_native_matmul_metadata=metadata,
    )


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mixed_fresh_full_root_outputs_codegen(
    schedule: str, dtype: torch.dtype
) -> None:
    with _cpu_codegen():
        bound = _root_exports._bind_isolated((*_args(dtype), "valid"))
        assert bound.config_spec.cute_chained_matmul_search_enabled
        sources = [bound.to_code(_config(schedule, flag)) for flag in (False, True)]
    assert all("chain_root_export_1" in source for source in sources)
    assert all("chain_root_export_2" in source for source in sources)
    assert sources[0].count("cute.gemm(") == sources[1].count("cute.gemm(") == 1

    def device(source: str) -> list[str]:
        return [
            ast.dump(node)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion")
        ]

    assert device(sources[0]) == device(sources[1])


@pytest.mark.parametrize(
    "mode",
    ["omitted", "dtype", "offset", "masked", "read_output", "independent_scan"],
)
def test_unsafe_fresh_export_ownership_does_not_classify(mode: str) -> None:
    with _cpu_codegen():
        bound = _root_exports._bind_isolated((*_args(), mode))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert all(
                classify_root_exports(tuple(graph.graph.nodes)) is None
                for graph in bound.host_function.device_ir.graphs
            )
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config("tcgen05_tmem"))


def test_nonunit_grid_tile_rejects_complete_export_claim() -> None:
    with _cpu_codegen():
        bound = _root_exports._bind_isolated((*_args(), "blocked"))
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config("tcgen05_tmem"))


@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_unresolved_full_slice_export_shape_keeps_existing_rejection(
    schedule: str,
) -> None:
    with _cpu_codegen():
        bound = _root_exports._bind_isolated((*_args(rows=64), "unmasked"))
        with pytest.raises((exc.InvalidConfig, exc.BackendUnsupported)):
            bound.to_code(_config(schedule))


@pytest.mark.parametrize("mutation", ["alias", "short", "partial"])
def test_fresh_export_proof_rechecks_storage_and_complete_shape(mutation: str) -> None:
    with _cpu_codegen():
        bound = _root_exports._bind_isolated((*_args(), "valid"))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            nodes = next(
                tuple(graph.graph.nodes)
                for graph in graphs
                if classify_root_exports(tuple(graph.graph.nodes)) is not None
            )
            proof = classify_root_exports(nodes)
            assert proof is not None
            target = cast("Node", proof.stores[1].store.args[0])
            if mutation == "partial":
                store = proof.stores[1].store
                indices = list(cast("list[Any]", store.args[1]))
                indices[-1] = slice(0, 32)
                with patch.object(
                    store, "_args", (store.args[0], indices, *store.args[2:])
                ):
                    assert classify_root_exports(nodes) is None
            else:
                tensor = (
                    cast("Node", proof.primary.args[0]).meta["val"]
                    if mutation == "alias"
                    else torch.empty((3, 64, 32), dtype=torch.bfloat16)
                )
                with patch.dict(target.meta, {"val": tensor}):
                    assert classify_root_exports(nodes) is None


def test_existing_single_output_source_is_unchanged() -> None:
    from .test_cute_chained_loop_exports import _root_indexed_chain
    from helion._compiler.cute import chained_root_exports

    args = (
        torch.empty((4, 128, 128), dtype=torch.bfloat16),
        torch.empty((4, 128, 128), dtype=torch.bfloat16),
        False,
    )
    with _cpu_codegen():
        bound = _root_indexed_chain._bind_isolated(args)
        source = bound.to_code(_config("coalesced"))
        with patch.object(
            chained_root_exports, "classify_root_exports", return_value=None
        ):
            reference = bound.to_code(_config("coalesced"))
    assert source == reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [57, 64])
def test_full_root_outputs_preserve_tails_and_transposed_values_gpu(
    schedule: str, dtype: torch.dtype, rows: int
) -> None:
    if schedule == "tcgen05_tmem" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    a, b, raw = _args(dtype, rows, "cuda")
    torch.manual_seed(1123)
    a.copy_(torch.randint(-4, 5, a.shape, device="cuda") * 0.0625)
    b.copy_(torch.randint(-4, 5, b.shape, device="cuda") * 0.0625)
    raw.copy_(torch.randint(-2, 3, raw.shape, device="cuda") * 0.0009765625)
    snapshots = [tensor.clone() for tensor in (a, b, raw)]
    padded = torch.zeros((3, 64, 64), device="cuda", dtype=torch.float64)
    padded[:, :rows] = a.double()
    values = torch.zeros((3, 64), device="cuda", dtype=torch.float32)
    values[:, :rows] = raw.float()
    gate = torch.exp2(values.cumsum(1))
    expected = (
        ((padded @ b.double()).float() * gate[:, :, None]).to(dtype),
        (padded.float() * gate[:, :, None]).to(dtype).transpose(1, 2),
        gate,
    )
    args = (a, b, raw, "valid")
    actual = _root_exports._bind_isolated(args).compile_config(_config(schedule))(*args)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0.002, rtol=0.01)
    for tensor, snapshot in zip((a, b, raw), snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
