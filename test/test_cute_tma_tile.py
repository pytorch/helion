from __future__ import annotations

import ast
import hashlib
import importlib
import json
import textwrap
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tmem_segments import _depends_on
from helion.runtime.cute import chained_leaf_pipeline
from helion.runtime.cute import chained_startup
from helion.runtime.cute.tma_tile import append_tma_tile

# Actual client output captured before extraction, including a preexisting body
# statement and argument. The complete original output is retained in artifacts.
_ORIGINAL_DIGESTS = (
    "8dab4f9042d1af8aa4889ca41d399ea1da2a6e445471f25a24463c93d25f6f50",
    "ae863de39fe6bc428cad929389a525380674cb9049a4eb1c1363603b204a8b3c",
    "cb432547f6b37cf4a519f56e43f78fabbb9bfbefde96c6bd7e59e330b778fffb",
    "b2b20a920f8b1af174769dc95492a3dbf0a9f407050eca83bfa3f8273a9d8d92",
    "5fd31963a74e8a6994f3f61c0204455b024aae5ed550beb985ca3291ad2b62f1",
    "54230294d76ebbb6cd93417bfa2bb849f7477ac36e428bd21d10e602b30f504c",
    "bb8a5678df689a901e92382bcd715b2796ad3dc8c206d7b23b348d99f0cdb3b0",
    "28f319def3ac66ae7e95dea960ff7541d042d4a6482933b8a8a678c0b23e9bc7",
    "124623be95f1830aa956e9881313869ef49807b1e159f6c059bb38848e5ba10f",
    "b2c7ca4f440e30af16c77290ee6d07410dfc11f68b47180025a5ce32825be74c",
    "54a4162aed3ea97920ad69e8cb8dcf31da66a67ad4bac380520ce0c9f9d33b3d",
)
_STARTUP_TILES = ((64, 16), (64, 32), (128, 64), (32, 128), (64, 256))
_DTYPES = {
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
    torch.float32: "Float32",
}


@pytest.mark.parametrize("case", range(11))
def test_existing_clients_delegate_with_byte_identical_wrapper_and_arguments(case):
    paired = case == 10
    index = 0 if paired else case
    dtype = "float32" if paired else "bfloat16" if case < 5 else "float16"
    tile = (128, 32) if paired else _STARTUP_TILES[case % 5]
    client = chained_leaf_pipeline if paired else chained_startup
    plan = {
        "lhs_idx": index + 2,
        "kernel_args": [f"transfer_{index}_atom", f"transfer_{index}_tensor"],
        "rows": 4096,
        "columns": 512,
        "dtype": dtype,
        "tile": tile,
    }
    body, call_args = ["    old_statement = 1"], ["original_arg"]
    with patch.object(client, "append_tma_tile", wraps=append_tma_tile) as shared:
        client.append_wrapper(body, call_args, plan)
    shared.assert_called_once()
    assert (
        hashlib.sha256(json.dumps([body, call_args]).encode()).hexdigest()
        == (_ORIGINAL_DIGESTS[case])
    )
    assert body[0] == "    old_statement = 1"
    assert call_args == ["original_arg", *plan["kernel_args"]]


def _build(dtype, tile, index=3):
    body, args = [], []
    append_tma_tile(
        body,
        args,
        source_index=index,
        atom="leaf_atom",
        tensor="leaf_tensor",
        rows=4096,
        columns=512,
        tile=tile,
        dtype=dtype,
    )
    assert args == ["leaf_atom", "leaf_tensor"]
    return body


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "tile", [(16, 16), (32, 32), (64, 48), (128, 64), (32, 128), (64, 256)]
)
def test_leaf_dtype_and_contiguous_bytewidth_select_existing_native_layout(dtype, tile):
    body = _build(dtype, tile)
    dtype_name = f"cutlass.{_DTYPES[dtype]}"
    byte_width = tile[1] * dtype.itemsize
    swizzle = min(128, byte_width & -byte_width)
    assert f"SmemLayoutAtomKind.K_SW{swizzle}, {dtype_name}" in body[1]
    assert "arg3.iterator.align(16)" in body[0]
    assert "stride=(512, 1)" in body[0]
    assert "CopyBulkTensorTileG2SOp()" in body[2]
    assert f"{tile!r}, order=(0, 1)" in body[1]
    tree = ast.parse("def wrapper():\n" + "\n".join(body))
    assert not any(
        isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(tree)
    )
    assert all("mbarrier" not in line and "cache" not in line for line in body)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.float64])
def test_unsupported_leaf_types_are_not_reinterpreted_or_packed(dtype):
    with pytest.raises(KeyError):
        _build(dtype, (32, 32))


def test_startup_client_does_not_gain_fp32_admission_from_shared_builder():
    with pytest.raises(KeyError):
        chained_startup.append_wrapper(
            [],
            [],
            {
                "lhs_idx": 0,
                "kernel_args": ["atom", "tensor"],
                "rows": 128,
                "columns": 128,
                "dtype": "float32",
                "tile": (128, 32),
            },
        )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("tile", [(32, 16), (64, 32), (32, 64), (32, 128), (64, 256)])
def test_actual_cute_tma_atom_and_layout_construct_without_a_device(dtype, tile):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    native = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    body = _build(dtype, tile)
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            pointer = cute.make_ptr(native, 4096, cute.AddressSpace.gmem)
            original_pointer = pointer.__extract_mlir_values__()[0]
            argument = cute.make_tensor(
                pointer, cute.make_layout((4096, 512), stride=(512, 1))
            )
            namespace: dict[str, Any] = {
                "cute": cute,
                "cutlass": cutlass,
                "arg3": argument,
            }
            exec(textwrap.dedent("\n".join(body)), namespace)
            global_tensor = namespace["leaf_atom_global"]
            assert global_tensor.element_type == native
            assert tuple(map(int, global_tensor.shape)) == (4096, 512)
            assert tuple(map(int, global_tensor.stride)) == (512, 1)
            assert _depends_on(
                global_tensor.iterator.__extract_mlir_values__()[0], original_pointer
            )
            layout = namespace["leaf_atom_layout"]
            assert (
                tuple(int(cute.size(layout, mode=[axis])) for axis in range(2)) == tile
            )
            assert int(cute.cosize(layout)) == tile[0] * tile[1]
            assert tuple(map(int, namespace["leaf_tensor"].shape)) == (4096, 512)
        assert module.operation.verify()
        text = str(module)
        assert "make_non_exec_tiled_tma_load" in text
        assert "cp_async" not in text and "mbarrier" not in text
    assert torch.cuda.is_initialized() == before
