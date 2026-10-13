from __future__ import annotations

import ast
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.cute.fuse_two_pass_loads import _own_statement_occurrences
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


def _shared_sweeps(effect: str = "") -> list[ast.stmt]:
    body = ast.parse(
        f"""
total = 0.0
for synthetic_lane_7 in range(8):
    value = (x.iterator + synthetic_lane_7).load()
    total = total + value
{effect}
for synthetic_lane_7 in range(8):
    value = (x.iterator + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(value + total)
"""
    ).body
    first = body[1]
    last = body[-1]
    assert isinstance(first, ast.For) and isinstance(last, ast.For)
    last.body[0] = first.body[0]
    return body


def _text(body: list[ast.stmt]) -> str:
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_lexical_ownership_keeps_annotations_without_copying_metadata() -> None:
    statement = statement_from_string("value = source + 1")
    marker = object()
    vars(statement)["_occurrence_test_marker"] = marker
    body = _shared_sweeps()
    first, last = body[1], body[-1]
    assert isinstance(first, ast.For) and isinstance(last, ast.For)
    first.body[0] = last.body[0] = statement
    before = _text(body)
    result = _own_statement_occurrences(body)
    copies = [
        node
        for node in ast.walk(ast.Module(body=result, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "value"
    ]
    assert len(copies) == 2 and copies[0] is not copies[1]
    assert copies[0].value is not copies[1].value
    for copied in copies:
        assert vars(copied)["_occurrence_test_marker"] is marker
        assert vars(copied)["_location"] is vars(statement)["_location"]
        assert type(copied) is type(statement)
    assert _text(result) == before == _text(body)
    assert _own_statement_occurrences(result) is result


@pytest.mark.parametrize("mode,loads", [("gmem", 2), ("register", 1)])
def test_shared_load_occurrences_match_independent_syntax(
    monkeypatch: pytest.MonkeyPatch, mode: str, loads: int
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    shared = _shared_sweeps()
    independent = ast.parse(_text(shared)).body
    options = {
        "tensor_dtypes": {"x": "cutlass.Float32", "out": "cutlass.Float32"},
        "reload_modes": {7: mode},
        "proven_disjoint_tensor_pairs": {frozenset(("x", "out"))},
    }
    actual = _text(fuse_two_pass_loads(shared, **options))
    expected = _text(fuse_two_pass_loads(independent, **options))
    assert actual == expected
    assert actual.count(".load()") == loads


@pytest.mark.parametrize(
    "effect",
    [
        "(x.iterator + 0).store(42.0)",
        "cute.arch.atomic_add(x.iterator + 0, 42.0)",
        "(opaque_pointer + 0).store(42.0)",
        "(other.iterator + 0).store(42.0)",
    ],
)
def test_shared_occurrences_do_not_bypass_alias_barriers(effect: str) -> None:
    result = fuse_two_pass_loads(
        _shared_sweeps(effect),
        tensor_dtypes=dict.fromkeys(("x", "out", "other"), "cutlass.Float32"),
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    assert "_fuse_cache_" not in _text(result)
    assert _text(result).count(".load()") == 2


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    "case", ["same", "different-index", "index-write", "captured", "captured-different"]
)
def test_memory_derived_address_bindings_are_versioned(shared: bool, case: str) -> None:
    first = "(index.iterator + 0).load()"
    second = (
        "(index.iterator + 1).load()"
        if case in ("different-index", "captured-different")
        else first
    )
    middle = "(index.iterator + 0).store(9)" if case == "index-write" else ""
    captured = (
        "saved = slot\nslot = (index.iterator + 1).load()"
        if case.startswith("captured")
        else "saved = slot"
    )
    body = ast.parse(
        f"""
slot = {first}
{captured}
for synthetic_lane_7 in range(8):
    value = (x.iterator + saved + synthetic_lane_7).load()
{middle}
slot = {second}
saved = slot
for synthetic_lane_7 in range(8):
    value = (x.iterator + saved + synthetic_lane_7).load()
    (out.iterator + synthetic_lane_7).store(value)
"""
    ).body
    loops = [stmt for stmt in body if isinstance(stmt, ast.For)]
    if shared:
        loops[1].body[0] = loops[0].body[0]
    result = fuse_two_pass_loads(
        body,
        tensor_dtypes={
            "index": "cutlass.Int32",
            "x": "cutlass.Float32",
            "out": "cutlass.Float32",
        },
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={
            frozenset(pair) for pair in (("x", "out"), ("index", "out"), ("index", "x"))
        },
    )
    # Captured keeps the old slot even after a distinct load rebinds slot.
    expect_cache = case in ("same", "captured")
    assert ("_fuse_cache_" in _text(result)) is expect_cache
    assert _text(result).count("(x.iterator") == (1 if expect_cache else 2)


def test_expanding_address_definitions_fail_closed_at_key_budget() -> None:
    definitions = "offset = base\n" + "offset = offset + offset\n" * 12
    source = (
        definitions + "for synthetic_lane_7 in range(8):\n"
        "    value = (x.iterator + offset + synthetic_lane_7).load()\n"
        + definitions
        + "for synthetic_lane_7 in range(8):\n"
        "    value = (x.iterator + offset + synthetic_lane_7).load()\n"
    )
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    assert "_fuse_cache_" not in _text(result)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("rebind", [False, True])
def test_completed_unrelated_induction_is_a_versioned_scalar(
    capture: bool, rebind: bool
) -> None:
    definition = "slot = j" if capture else ""
    address = "slot" if capture else "j"
    middle = f"for j in range(3):\n    pass\n{definition}" if rebind else ""
    body = ast.parse(
        f"""
for j in range(2):
    pass
{definition}
for synthetic_lane_7 in range(8):
    value = (x.iterator + {address} + synthetic_lane_7).load()
{middle}
for synthetic_lane_7 in range(8):
    value = (x.iterator + {address} + synthetic_lane_7).load()
"""
    ).body
    result = fuse_two_pass_loads(
        body, tensor_dtypes={"x": "cutlass.Float32"}, reload_modes={7: "register"}
    )
    assert ("_fuse_cache_" in _text(result)) is not rebind


@pytest.mark.parametrize(
    "first,second",
    [
        ("slot += 1", "slot += 2"),
        ("slot = slot + 1", "slot = slot + 2"),
    ],
)
def test_loop_carried_address_recurrences_are_not_initial_values(
    first: str, second: str
) -> None:
    body = ast.parse(
        f"""
slot = 0
for synthetic_lane_7 in range(8):
    value = (x.iterator + slot + synthetic_lane_7).load()
    {first}
slot = 0
for synthetic_lane_7 in range(8):
    value = (x.iterator + slot + synthetic_lane_7).load()
    {second}
"""
    ).body
    result = fuse_two_pass_loads(
        body, tensor_dtypes={"x": "cutlass.Float32"}, reload_modes={7: "register"}
    )
    assert "_fuse_cache_" not in _text(result)


@pytest.mark.parametrize("rebind", [False, True])
def test_matching_induction_captures_require_unchanged_cache_index(
    rebind: bool,
) -> None:
    change = "synthetic_lane_7 += 1" if rebind else ""
    sweep = f"""
for synthetic_lane_7 in range(8):
    saved = synthetic_lane_7
    {change}
    value = (x.iterator + saved).load()
"""
    result = fuse_two_pass_loads(
        ast.parse(sweep + sweep).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    assert ("_fuse_cache_" in _text(result)) is not rebind


@pytest.mark.parametrize("same_domain", [False, True])
def test_nested_sweep_keys_share_only_proven_active_iteration_domains(
    same_domain: bool,
) -> None:
    second_range = "range(4)" if same_domain else "range(1, 5)"
    result = fuse_two_pass_loads(
        ast.parse(
            f"""
for synthetic_lane_7 in range(8):
    row = synthetic_lane_7 * 4
    for vec_lane_0 in range(4):
        value = (x.iterator + row + vec_lane_0).load()
for synthetic_lane_7 in range(8):
    row = synthetic_lane_7 * 4
    for vec_lane_1 in {second_range}:
        value = (x.iterator + row + vec_lane_1).load()
"""
        ).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    assert ("_fuse_cache_" in _text(result)) is same_domain


def test_same_spelled_sweep_bounds_must_keep_their_original_bindings() -> None:
    body = ast.parse(
        """
base = 0
for synthetic_lane_7 in range(base, base + 8):
    value = (x.iterator + synthetic_lane_7).load()
base = 8
for synthetic_lane_7 in range(base, base + 8):
    value = (x.iterator + synthetic_lane_7).load()
"""
    ).body
    result = fuse_two_pass_loads(
        body, tensor_dtypes={"x": "cutlass.Float32"}, reload_modes={7: "register"}
    )
    assert "_fuse_cache_" not in _text(result)


def test_cache_index_bounds_cannot_be_rebound_inside_the_sweep() -> None:
    sweep = """
base = 0
for synthetic_lane_7 in range(base, base + 8):
    base = base + 1
    value = (x.iterator + synthetic_lane_7).load()
"""
    result = fuse_two_pass_loads(
        ast.parse(sweep + sweep).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    assert "_fuse_cache_" not in _text(result)


def test_shared_occurrences_execute_complete_producer_before_cached_consumer() -> None:
    class Pointer:
        def __init__(self, values: list[float], offset: int = 0) -> None:
            self.values, self.offset = values, offset

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.values, self.offset + offset)

        def load(self) -> float:
            return self.values[self.offset]

        def store(self, value: float) -> None:
            self.values[self.offset] = value

    result = fuse_two_pass_loads(
        _shared_sweeps(),
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    for source in ([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], [9.0] * 8):
        output = [float("nan")] * 8
        ns = {
            "x": SimpleNamespace(iterator=Pointer(source)),
            "out": SimpleNamespace(iterator=Pointer(output)),
            "cute": SimpleNamespace(make_rmem_tensor=lambda size, dtype: [None] * size),
            "cutlass": SimpleNamespace(Float32=float),
        }
        exec(_text(result), ns)
        assert output == [value + sum(source) for value in source]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _state_reduction(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    bias: torch.Tensor,
    slots: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    batch, heads, width = x.shape
    state_dim = state.size(3)
    per_group = heads // b.size(1)
    for tb, th, tv in hl.tile([batch, heads, width], block_size=[1, 1, None]):
        ib, ih = tb.id, th.id
        nn = hl.arange(state_dim)
        slot = slots[ib].long()
        current = state[slot, ih, tv.index, nn].float()
        raw = dt[ib, ih, tv].float() + bias[ih, tv].float()
        delta = torch.where(raw <= 20.0, torch.log1p(torch.exp(raw)), raw)
        decay = a[ih, tv.index, nn].float()
        value = x[ib, ih, tv].float()
        group = ih // per_group
        bv = b[ib, group, nn].float()
        cv = c[ib, group, nn].float()
        current = current * torch.exp(delta[:, None] * decay)
        current = current + (delta * value)[:, None] * bv[None, :]
        output[ib, ih, tv] = (current * cv[None, :]).sum(-1) + d[ih, tv].float() * value
        state[slot, ih, tv.index, nn] = current
    return output


def _args(
    dtype: torch.dtype, width: int, state_dim: int, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, ...]:
    def random(shape: tuple[int, ...], dt: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dt) * 0.125

    return (
        random((5, 4, width, state_dim), dtype),
        random((2, 4, width), dtype),
        random((2, 4, 1)).expand(2, 4, width),
        (-random((4, 1, 1)).abs() - 0.5).expand(4, width, state_dim),
        random((2, 2, state_dim), dtype),
        random((2, 2, state_dim), dtype),
        random((4, 1)).expand(4, width),
        (random((4, 1)) - 2.0).expand(4, width),
        torch.tensor([2, 0], dtype=torch.int64, device=device),
        torch.empty((2, 4, width), dtype=dtype, device=device),
    )


def _config(mode: str) -> helion.Config:
    return helion.Config(
        block_sizes=[16],
        num_threads=[0, 16],
        loop_orders=[[1, 2, 0]],
        pid_type="flat",
        cute_reduction_reloads=[mode],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,state_dim", [(32, 128), (17, 128), (17, 73)])
def test_actual_split_codegen_register_reuses_raw_state_and_gmem_stays_exact(
    dtype: torch.dtype, width: int, state_dim: int
) -> None:
    module = importlib.import_module("helion._compiler.cute.fuse_two_pass_loads")
    args = _args(dtype, width, state_dim)

    def without_ownership(body, *, scalar_bindings=None):
        if scalar_bindings is not None:
            scalar_bindings.clear()
            scalar_bindings.update(
                module._scalar_bindings(ast.Module(body=body, type_ignores=[]))
            )
        return body

    with _cpu_codegen():
        bound = _state_reduction._bind_isolated(args)
        with patch.object(
            module, "_own_statement_occurrences", side_effect=without_ownership
        ):
            old_gmem = bound.to_code(_config("gmem"))
            old_register = bound.to_code(_config("register"))
        new_gmem = bound.to_code(_config("gmem"))
        new_register = bound.to_code(_config("register"))
    assert new_gmem == old_gmem == old_register
    if state_dim == 73:
        # This geometry selects a different rolled reduction. Occurrence
        # ownership must not force it through the persistent-sweep matcher.
        assert new_register == old_register
        return
    assert "_fuse_cache_" in new_register
    assert new_register.count("state.iterator") == 2  # One read and one write.
    assert old_gmem.count("state.iterator") == 3
    for spelling in ("cute.math.exp2(", "cute.math.log(", "warp_reduction_sum("):
        assert new_register.count(spelling) == old_gmem.count(spelling)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,state_dim", [(32, 128), (17, 128)])
@skipUnlessBackends(["cute"])
def test_gpu_repeated_snapshot_updates_and_unrounded_output(
    dtype: torch.dtype, width: int, state_dim: int
) -> None:
    torch.manual_seed(8401)
    original = _args(dtype, width, state_dim, DEVICE)
    readonly = [tensor.clone() for tensor in original[1:-1]]
    group_map = torch.tensor([0, 0, 1, 1], device=DEVICE)
    histories = []
    for mode in ("gmem", "register"):
        args = (original[0].clone(), *original[1:-1], torch.empty_like(original[-1]))
        bound = _state_reduction._bind_isolated(args)
        source = bound.to_code(_config(mode))
        cache_declarations = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_fuse_cache_")
        ]
        assert len(cache_declarations) == (3 if mode == "register" else 0)
        assert source.count("state.iterator") == (2 if mode == "register" else 3)
        fn = bound.compile_config(_config(mode))
        history = []
        for _ in range(3):
            state, x, dt, a, b, c, d, bias, slots, output = args
            before = state.clone()
            delta = torch.nn.functional.softplus(dt.double() + bias.double())
            current = before[slots].double() * torch.exp(delta[..., None] * a.double())
            current = (
                current
                + (delta * x.double())[..., None] * b.double()[:, group_map, None, :]
            )
            expected = (current * c.double()[:, group_map, None, :]).sum(
                -1
            ) + d.double() * x.double()
            expected_state = before.clone()
            expected_state[slots] = current.to(dtype)
            output.fill_(float("nan"))
            fn(*args)
            torch.testing.assert_close(output, expected.to(dtype), atol=0.02, rtol=0.01)
            torch.testing.assert_close(state, expected_state, atol=0.02, rtol=0.01)
            torch.testing.assert_close(
                state[[1, 3, 4]], before[[1, 3, 4]], atol=0, rtol=0
            )
            history.append((output.clone(), state.clone()))
        histories.append(history)
    # A sum of two FP32 products may contract either product into the add.
    # Keep the original numeric contract; raw cache identity is tested separately.
    torch.testing.assert_close(histories[0], histories[1], atol=0.02, rtol=0.01)
    torch.testing.assert_close(list(original[1:-1]), readonly, atol=0, rtol=0)
