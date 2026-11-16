"""The CuTe backend's torch -> CUTLASS dtype table is declared once."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
import torch

from helion import exc
from helion._compiler.cute import dtypes
from helion._compiler.cute.backend import CuteBackend


def test_pointer_table_differs_from_register_names_on_bool_only() -> None:
    assert set(dtypes.CUTLASS_POINTER_DTYPE_NAME) == set(dtypes.CUTLASS_DTYPE_NAME)
    differing = {
        dtype
        for dtype, name in dtypes.CUTLASS_POINTER_DTYPE_NAME.items()
        if dtypes.CUTLASS_DTYPE_NAME[dtype] != name
    }
    assert differing == {torch.bool}
    assert dtypes.CUTLASS_POINTER_DTYPE_NAME[torch.bool] == "cutlass.Uint8"


def test_register_names_agree_with_the_backend_dtype_str() -> None:
    backend = CuteBackend()
    for dtype, name in dtypes.CUTLASS_DTYPE_NAME.items():
        assert backend.dtype_str(dtype) == name, dtype


def test_standalone_launcher_carries_the_pointer_table() -> None:
    """The Helion-free standalone launcher's frozen copy equals the table
    (read as source: the exporter splices the module text; importing it
    needs cutlass)."""
    source = (
        Path(dtypes.__file__).parents[2] / "runtime" / "cute" / "standalone_launcher.py"
    ).read_text()
    table = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_STANDALONE_DTYPES"
            for target in node.targets
        )
    )
    assert isinstance(table, ast.Dict)
    frozen = {
        ast.literal_eval(cast("ast.expr", key)): ast.unparse(value)
        for key, value in zip(table.keys, table.values, strict=True)
    }
    assert frozen == {
        str(dtype): name for dtype, name in dtypes.CUTLASS_POINTER_DTYPE_NAME.items()
    }


def test_pointer_dtype_resolves_lazily() -> None:
    pytest.importorskip("cutlass")
    import cutlass

    assert dtypes.cutlass_pointer_dtype(torch.bool) is cutlass.Uint8
    assert dtypes.cutlass_pointer_dtype(torch.float8_e8m0fnu) is cutlass.Float8E8M0FNU
    with pytest.raises(exc.BackendUnsupported, match="dtype"):
        dtypes.cutlass_pointer_dtype(torch.complex64)
