from __future__ import annotations

import ast
import importlib
from textwrap import indent
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._testing import DEVICE
import helion.language as hl


class _Pointer:
    def __init__(self, data: list, offset: int = 0) -> None:
        self.data, self.offset = data, offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.data, self.offset + int(offset))

    def load(self):
        return self.data[self.offset]

    def store(self, value) -> None:
        self.data[self.offset] = value


def _sweeps(middle: str = "", second: str = "base") -> str:
    return f"""
for synthetic_lane_7 in range(8):
    slot = (index.iterator + base).load()
    value = (x.iterator + slot * 8 + synthetic_lane_7).load()
{middle}
for synthetic_lane_7 in range(8):
    slot = (index.iterator + {second}).load()
    value = (x.iterator + slot * 8 + synthetic_lane_7).load()
    (out.iterator + outer * 8 + synthetic_lane_7).store(value)
"""


def _fuse(source: str, *, disjoint: bool = True, mode: str = "register") -> str:
    pairs = {frozenset(("x", "out")), frozenset(("index", "out"))}
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={
            "x": "cutlass.Float32",
            "index": "cutlass.Int32",
            "out": "cutlass.Float32",
        },
        reload_modes={7: mode},
        proven_disjoint_tensor_pairs=pairs if disjoint else set(),
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


def _run(source: str, active: bool = True) -> tuple[list, list, list]:
    x = [float((i % 13) - 6) / 8 for i in range(128)]
    index = [2, 0, 3, 1, 4, 5, 6, 7]
    out = [float("inf")] * 32
    env = {
        "x": SimpleNamespace(iterator=_Pointer(x)),
        "index": SimpleNamespace(iterator=_Pointer(index)),
        "out": SimpleNamespace(iterator=_Pointer(out)),
        "cute": SimpleNamespace(make_rmem_tensor=lambda size, dtype: [None] * size),
        "cutlass": SimpleNamespace(Float32=float, Int32=int),
        "active": active,
    }
    exec(source, env)
    return x, index, out


@pytest.mark.parametrize("kind", ["for", "nested", "branch", "completed", "capture"])
def test_inherited_readonly_indices_execute_with_original_lexical_scope(
    kind: str,
) -> None:
    body = "base = outer\n" + _sweeps()
    if kind == "nested":
        body = "for inner in range(2):\n" + indent(
            "base = outer + inner\n" + _sweeps(), "    "
        )
    elif kind == "branch":
        body = (
            "base = outer\nif active:\n"
            + indent(_sweeps(), "    ")
            + "else:\n"
            + indent("base = 2 - outer\n" + _sweeps(), "    ")
        )
    elif kind == "completed":
        body = "for old in range(2):\n    pass\nbase = old + outer\n" + _sweeps()
    elif kind == "capture":
        body = "base = outer\nsaved = base\nbase = 5\n" + _sweeps().replace(
            "+ base)", "+ saved)"
        )
    source = "for outer in range(3):\n" + indent(body, "    ")
    result = _fuse(source)
    assert result.count("(x.iterator") == (2 if kind == "branch" else 1)
    assert "_fuse_cache_" in result
    for active in (False, True):
        assert _run(result, active) == _run(source, active)
    assert _fuse(source, mode="gmem") == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "kind",
    [
        "coordinate",
        "index-write",
        "conditional",
        "completed-loop",
        "loop-carried",
        "unsupported",
    ],
)
def test_inherited_bindings_do_not_cross_mutation_or_unknown_control(kind: str) -> None:
    middle = {
        "coordinate": "base = outer + 1",
        "index-write": "(index.iterator + base).store(4)",
        "conditional": "if active:\n    base = outer + 1",
        "completed-loop": "for base in range(2):\n    pass",
        "loop-carried": "base += 1",
        "unsupported": "",
    }[kind]
    body = "base = outer\n" + _sweeps(middle)
    if kind == "unsupported":
        # The while body must not assume the outer coordinate is invariant.
        body = "base = outer\nonce = 0\nwhile once < 1:\n" + indent(
            _sweeps() + "once += 1\n", "    "
        )
    source = "for outer in range(3):\n" + indent(body, "    ")
    result = _fuse(source)
    assert result.count("(x.iterator") == 2
    for active in (False, True):
        assert _run(result, active) == _run(source, active)


def test_outer_carry_is_not_replaced_by_initial_value() -> None:
    source = "base = 0\nfor outer in range(3):\n" + indent(_sweeps("base += 1"), "    ")
    result = _fuse(source)
    assert result.count("(x.iterator") == 2
    assert _run(result) == _run(source)


def test_sibling_branch_rebinding_stays_local_to_each_snapshot() -> None:
    source = (
        "for outer in range(3):\n    base = outer\n    if active:\n"
        + indent(_sweeps("base += 1"), "        ")
        + "    else:\n"
        + indent(_sweeps(), "        ")
    )
    result = _fuse(source)
    assert result.count("(x.iterator") == 3
    for active in (False, True):
        assert _run(result, active) == _run(source, active)


def test_inherited_readonly_proof_keeps_unknown_alias_barrier() -> None:
    source = "for outer in range(3):\n" + indent(
        "base = outer\n" + _sweeps("(out.iterator + 0).store(4.0)"), "    "
    )
    result = _fuse(source, disjoint=False)
    assert result.count("(x.iterator") == 2


def test_for_iterator_assignment_does_not_reuse_pre_iterator_binding() -> None:
    source = "base = 0\nfor outer in range(base := 3):\n" + indent(
        _sweeps(second="0"), "    "
    )
    result = _fuse(source)
    assert result.count("(x.iterator") == 2
    assert _run(result) == _run(source)


def test_if_test_assignment_does_not_reuse_pre_test_binding() -> None:
    source = "outer = 0\nbase = 0\nif (base := 2):\n" + indent(
        _sweeps(second="0"), "    "
    )
    result = _fuse(source)
    assert result.count("(x.iterator") == 2
    assert _run(result) == _run(source)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indexed_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    indices: torch.Tensor,
    accepted: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    batch, heads, width = x.shape
    reduction = state.size(3)
    for row, head, column in hl.tile([batch, heads, width], block_size=[1, 1, None]):
        selected = accepted[row.id].long()
        slot = indices[row.id, selected].long()
        rr = hl.arange(reduction)
        current = state[slot, head.id, column.index, rr].float()
        value = x[row.id, head.id, column].float()
        bv = b[row.id, head.id, rr].float()
        cv = c[row.id, head.id, rr].float()
        current = current * 0.5 + value[:, None] * bv[None, :]
        out[row.id, head.id, column] = (current * cv[None, :]).sum(-1)
        state[slot, head.id, column.index, rr] = current
    return out


def _config(mode: str) -> helion.Config:
    return helion.Config(
        block_sizes=[16],
        num_threads=[1, 32],
        loop_orders=[[1, 2, 0]],
        pid_type="flat",
        cute_reduction_reloads=[mode],
    )


def _args(
    dtype: torch.dtype, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, ...]:
    def values(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-4, 5, shape, device=device).to(dtype) / 16

    return (
        values((5, 3, 20, 128))[:, :, :17],
        values((2, 3, 17)),
        values((2, 3, 128)),
        values((2, 3, 128)),
        torch.tensor([[2, 4, 0], [0, 1, 3]], device=device),
        torch.tensor([0, 0], device=device),
        torch.empty((2, 3, 17), device=device),
    )


def test_nested_indexed_reduction_codegen_uses_cache() -> None:
    module = importlib.import_module("helion._compiler.cute.fuse_two_pass_loads")
    original = module._CuteFuseTwoPassLoads._try_fuse

    def without_inherited(self, body, *, allow_smem=True, inherited=None):
        return original(self, body, allow_smem=allow_smem)

    args = _args(torch.bfloat16)
    with _cpu_codegen():
        bound = _indexed_state_update._bind_isolated(args)
        with patch.object(module._CuteFuseTwoPassLoads, "_try_fuse", without_inherited):
            old = bound.to_code(_config("register"))
            old_gmem = bound.to_code(_config("gmem"))
        code = bound.to_code(_config("register"))
        gmem = bound.to_code(_config("gmem"))
    assert "_fuse_cache_" in code
    assert code.count("state.iterator") == 2
    assert old.count("state.iterator") == 3
    assert old_gmem == gmem


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gpu_nested_indexed_reduction_preserves_changing_indices(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(981)
    original = _args(dtype, DEVICE)
    for mode in ("gmem", "register"):
        state, x, b, c, indices, accepted, output = original
        storage = torch.full((5, 3, 20, 128), 3.0, device=DEVICE, dtype=dtype)
        storage[:, :, :17].copy_(state)
        out_storage = torch.full((3, 3, 20), float("nan"), device=DEVICE)
        args = (storage[:, :, :17], x, b, c, indices, accepted, out_storage[:2, :, :17])
        readonly = [t.clone() for t in args[1:5]]
        bound = _indexed_state_update._bind_isolated(args)
        source = bound.to_code(_config(mode))
        assert source.count("state.iterator") == (2 if mode == "register" else 3)
        compiled = bound.compile_config(_config(mode))
        for choices in ([0, 0], [1, 1], [2, 2]):
            accepted.copy_(torch.tensor(choices, device=DEVICE))
            slots = indices[torch.arange(2, device=DEVICE), accepted]
            before = storage.clone()
            current = (
                before[slots, :, :17].double() * 0.5
                + x.double()[..., None] * b.double()[:, :, None, :]
            )
            expected_output = (current * c.double()[:, :, None, :]).sum(-1)
            expected_state = before.clone()
            expected_state[slots, :, :17] = current.to(dtype)
            out_storage.fill_(float("nan"))
            compiled(*args)
            torch.testing.assert_close(
                args[-1].double(), expected_output, atol=2e-5, rtol=2e-5
            )
            torch.testing.assert_close(storage, expected_state, atol=0, rtol=0)
            assert (
                torch.isnan(out_storage[2]).all()
                and torch.isnan(out_storage[:2, :, 17:]).all()
            )
            torch.testing.assert_close(list(args[1:5]), readonly, atol=0, rtol=0)
            assert accepted.tolist() == choices
