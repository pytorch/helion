from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


@pytest.mark.parametrize("changed_address", (False, True))
def test_many_loads_freeze_each_binding_once_per_sweep(
    monkeypatch: pytest.MonkeyPatch, changed_address: bool
) -> None:
    count = 12
    sweeps = []
    for sweep in range(2):
        lines = ["for synthetic_lane_7 in range(8):"]
        for index in range(count):
            offset = index + (
                count if changed_address and sweep == 1 and index == 6 else 0
            )
            lines.extend(
                (
                    f"    address = synthetic_lane_7 + {offset}",
                    f"    value_{sweep}_{index} = (x.iterator + address).load()",
                )
            )
        sweeps.append("\n".join(lines))

    freezes = 0
    original = _CuteFuseTwoPassLoads._freeze_binding

    def freeze(
        self: _CuteFuseTwoPassLoads,
        stmt: ast.stmt,
        value: ast.expr,
        previous: dict[str, ast.expr],
    ) -> ast.expr:
        nonlocal freezes
        freezes += 1
        return original(self, stmt, value, previous)

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_freeze_binding", freeze)
    result = fuse_two_pass_loads(
        ast.parse("\n".join(sweeps)).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # Rebinding the same address name must preserve each load's own offset.
    assert code.count(".load()") == count + int(changed_address)
    assert code.count("cute.make_rmem_tensor") == count - int(changed_address)
    # Count work instead of wall time: a long sweep must not re-expand every
    # preceding binding separately for each new load.
    assert freezes <= 4 * count
