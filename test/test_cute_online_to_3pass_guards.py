"""CPU admission and source-arithmetic checks for the online softmax rewrite."""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
import math
import textwrap
from types import FunctionType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.aot_example import row_softmax
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.online_to_3pass import _detect_online_softmax
from helion._compiler.cute.online_to_3pass import _OnlineToThreePassTransformer
from helion._compiler.cute.online_to_3pass import rewrite_online_to_3pass
from helion._compiler.host_function import HostFunction
from helion._compiler.host_function import KernelDefinition
from helion._compiler.source_location import UnknownLocation
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


_BARE = """
def kernel(x, block_size_n=2):
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=65536):
        mi = hl.full([tile_m], float('-inf'), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next[:, None]).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out
"""
_TYPED = (
    _BARE.replace("x[tile_m, tile_n]", "x[tile_m, tile_n].to(torch.float32)")
    .replace(
        "torch.exp(values - mi_next[:, None]).sum(dim=1)",
        "torch.sum(torch.exp(values - mi_next[:, None]), dim=1)",
    )
    .replace(
        "torch.exp(values - mi[:, None]) / di[:, None]",
        "(torch.exp(values - mi[:, None]) / di[:, None]).to(out.dtype)",
    )
)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HELION_DISABLE_ONLINE_TO_3PASS", raising=False)
    monkeypatch.setenv("HELION_ONLINE_TO_3PASS_MIN_N", "0")
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _function(source: str) -> ast.FunctionDef:
    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    node.decorator_list = []
    return node


def _outer(function: ast.FunctionDef) -> ast.For:
    return next(node for node in function.body if isinstance(node, ast.For))


def _loop(statement: ast.stmt) -> ast.For:
    assert isinstance(statement, ast.For)
    return statement


def _value(statement: ast.stmt) -> ast.expr:
    assert isinstance(statement, ast.Assign)
    return statement.value


def _rewrite(function: ast.FunctionDef) -> tuple[ast.FunctionDef, int]:
    result = copy.deepcopy(function)
    parameters = frozenset(
        node.arg for node in ast.walk(result.args) if isinstance(node, ast.arg)
    )
    transformer = _OnlineToThreePassTransformer(result.body, parameters)
    transformer.visit(result)
    # ExtendedAST nodes carry source locations; normalize them for Python's
    # compiler after testing the actual rewrite rather than a separate model.
    return _function(ast.unparse(result)), transformer.rewrites


def _assert_declined(function: ast.FunctionDef) -> None:
    result, rewrites = _rewrite(function)
    assert rewrites == 0
    assert ast.dump(result) == ast.dump(function)


def _host(source: str, bindings: dict[str, object] | None = None) -> HostFunction:
    function = _function(source)
    namespace = {"torch": torch, "hl": hl, "math": math, **(bindings or {})}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<online>", "exec"),
        namespace,
    )
    fn = namespace[function.name]
    assert isinstance(fn, FunctionType)
    params = inspect.signature(fn).bind_partial(torch.empty(1, 4))
    params.apply_defaults()
    return HostFunction(
        KernelDefinition(fn, {}, function.name, function.args, function.body, params),
        UnknownLocation(),
    )


def _evaluate(function: ast.FunctionDef, x: torch.Tensor, block: int) -> object:
    class TileBegin(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            self.generic_visit(node)
            if node.attr == "begin":
                node.attr = "start"
            return node

    def tile(extent: int, *, block_size: int) -> Iterator[slice]:
        for start in range(0, extent, block_size):
            yield slice(start, min(start + block_size, extent))

    def shape(dimensions: list[slice]) -> list[int]:
        return [dimension.stop - dimension.start for dimension in dimensions]

    # A CPU slice represents the same half-open logical tile. Only the DSL
    # tile/factory operations are adapted; torch executes the rewritten AST.
    cpu_hl = SimpleNamespace(
        tile=tile,
        full=lambda dimensions, fill, **kwargs: torch.full(
            shape(dimensions), fill, **kwargs
        ),
        zeros=lambda dimensions, **kwargs: torch.zeros(shape(dimensions), **kwargs),
    )
    normalized = TileBegin().visit(copy.deepcopy(function))
    namespace: dict[str, object] = {"torch": torch, "hl": cpu_hl, "math": math}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[normalized], type_ignores=[])),
            "<CPU logical tiles>",
            "exec",
        ),
        namespace,
    )
    fn = namespace[function.name]
    assert isinstance(fn, FunctionType)
    return fn(x, block)


def _execute(function: ast.FunctionDef, x: torch.Tensor, block: int) -> torch.Tensor:
    result = _evaluate(function, x, block)
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("source", [_BARE, _TYPED], ids=["bare", "typed"])
@pytest.mark.parametrize("positional_dim", [False, True])
def test_complete_supported_signatures_match(source: str, positional_dim: bool) -> None:
    if positional_dim:
        source = source.replace("dim=1", "1")
    before = _function(source)
    after, rewrites = _rewrite(before)
    assert rewrites == 1
    assert sum(isinstance(stmt, ast.For) for stmt in _outer(after).body) == 3
    original_load = _value(_loop(_outer(before).body[2]).body[0])
    for loop in (node for node in _outer(after).body if isinstance(node, ast.For)):
        assert ast.dump(_value(loop.body[0])) == ast.dump(original_load)
    assert ast.dump(_outer(after).body[-1]) == ast.dump(_outer(before).body[-1])


@pytest.mark.parametrize("source", [_BARE, _TYPED], ids=["bare", "typed"])
@pytest.mark.parametrize(
    ("site", "keyword", "value"),
    [
        ("mi", "dtype", "torch.float64"),
        ("di", "dtype", "torch.float16"),
        ("mi", "device", "record()"),
        ("di", "device", "record()"),
        ("amax", "keepdim", "True"),
        ("amax", "out", "out"),
        ("maximum", "out", "out"),
        ("rescale", "out", "out"),
        ("sum_exp", "out", "out"),
        ("sum", "dtype", "torch.float64"),
        ("sum", "keepdim", "True"),
        ("sum", "out", "out"),
        ("sum", None, "options"),
    ],
)
def test_unknown_or_semantic_call_keywords_decline(
    source: str, site: str, keyword: str | None, value: str
) -> None:
    function = _function(source)
    body = _outer(function).body
    online = _loop(body[2]).body
    update = _value(online[3])
    assert isinstance(update, ast.BinOp)
    assert isinstance(update.left, ast.BinOp)
    assert isinstance(update.right, ast.Call)
    if source == _TYPED:
        sum_exp = update.right.args[0]
    else:
        assert isinstance(update.right.func, ast.Attribute)
        sum_exp = update.right.func.value
    calls = {
        "mi": _value(body[0]),
        "di": _value(body[1]),
        "amax": _value(online[1]),
        "maximum": _value(online[2]),
        "rescale": update.left.right,
        "sum": update.right,
        "sum_exp": sum_exp,
    }
    call = calls[site]
    assert isinstance(call, ast.Call)
    call.keywords = [kw for kw in call.keywords if kw.arg != keyword]
    call.keywords.append(
        ast.keyword(arg=keyword, value=ast.parse(value, mode="eval").body)
    )
    _assert_declined(function)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("dtype=torch.float32", ""),
        (
            "hl.zeros([tile_m], dtype=torch.float32)",
            "hl.full([tile_m], -0.0, dtype=torch.float32)",
        ),
        (
            "hl.zeros([tile_m], dtype=torch.float32)",
            "hl.full([tile_m], False, dtype=torch.float32)",
        ),
        ("float('-inf')", "float(record())"),
        (
            "hl.tile(n, block_size=block_size_n)",
            "hl.tile(1, n, block_size=block_size_n)",
        ),
        ("hl.tile(n, block_size=block_size_n)", "hl.tile(n, block_size=record())"),
        (
            "hl.tile(n, block_size=block_size_n)",
            "hl.tile(source.size(1), block_size=block_size_n)",
        ),
        (
            "hl.tile(n, block_size=block_size_n)",
            "hl.tile(n, block_size=block_size_n, reverse=True)",
        ),
        ("hl.tile(n, block_size=block_size_n)", "hl.tile(n, block_size=0)"),
        ("hl.tile(n, block_size=block_size_n)", "hl.tile(n, block_size=True)"),
        (
            "hl.tile(n, block_size=block_size_n)",
            "hl.tile(tile_n, block_size=block_size_n)",
        ),
        ("x[tile_m, tile_n]", "x[tile_n, tile_m]"),
        ("values - mi[:, None]", "values - mi_next[:, None]"),
        ("/ di[:, None]", "/ di[:, None] + record()"),
        ("out[tile_m, tile_n] =", "out[tile_n, tile_m] ="),
    ],
)
def test_unsafe_expression_and_iterator_shapes_decline(
    original: str, replacement: str
) -> None:
    assert original in _BARE
    _assert_declined(_function(_BARE.replace(original, replacement)))


@pytest.mark.parametrize(
    "conversion",
    [
        ".to(torch.float64)",
        ".to(dtype)",
        ".to(torch.float32, copy=True)",
        ".to(torch.float32, device=record())",
        ".to(torch.float16).to(torch.float32)",
    ],
)
def test_unsupported_load_conversions_decline(conversion: str) -> None:
    _assert_declined(_function(_TYPED.replace(".to(torch.float32)", conversion)))


@pytest.mark.parametrize(
    "keyword", ["copy=True", "device=record()", "dtype=torch.float64"]
)
def test_output_conversion_keeps_its_exact_contract(keyword: str) -> None:
    _assert_declined(
        _function(_TYPED.replace(".to(out.dtype)", f".to(out.dtype, {keyword})"))
    )


@pytest.mark.parametrize(
    "source",
    [
        _TYPED.replace(".to(torch.float32)", ".to(dtype=torch.float32)"),
        _TYPED.replace(".to(out.dtype)", ".to(dtype=out.dtype)"),
        _BARE.replace("float('-inf')", "-math.inf"),
        _BARE.replace(
            "hl.zeros([tile_m], dtype=torch.float32)",
            "hl.full([tile_m], 0.0, dtype=torch.float32)",
        ),
    ],
    ids=["keyword-load", "keyword-output", "math-inf", "full-zero"],
)
def test_equivalent_pure_spellings_keep_their_original_casts(source: str) -> None:
    original = _function(source)
    rewritten, count = _rewrite(original)
    assert count == 1
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    torch.testing.assert_close(_execute(rewritten, x, 2), _execute(original, x, 2))


@pytest.mark.parametrize("loop", [2, 3])
def test_loop_else_is_not_discarded(loop: int) -> None:
    function = _function(_BARE)
    _loop(_outer(function).body[loop]).orelse = ast.parse("di = di + 1").body
    _assert_declined(function)


@pytest.mark.parametrize(
    ("before", "after"),
    list(itertools.combinations(("mi", "di", "values", "local_amax", "mi_next"), 2)),
)
def test_pairwise_state_and_temporary_aliases_decline(before: str, after: str) -> None:
    function = _function(_TYPED)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == before:
            node.id = after
    _assert_declined(function)


@pytest.mark.parametrize(
    "name", ["mi", "di", "values", "local_amax", "mi_next", "tile_n"]
)
@pytest.mark.parametrize(
    "position", ["prefix", "suffix", "after_outer", "parameter", "global"]
)
def test_whole_function_observations_decline(name: str, position: str) -> None:
    function = _function(_TYPED)
    observation = ast.parse(f"saved = {name}").body[0]
    if position == "prefix":
        # An earlier binding makes the first prefix read valid; later reads
        # observe the preceding outer iteration's locals.
        function.body.insert(0, ast.parse(f"{name} = initial").body[0])
        _outer(function).body.insert(0, observation)
    elif position == "suffix":
        _outer(function).body.append(observation)
    elif position == "after_outer":
        function.body.insert(-1, observation)
    elif position == "parameter":
        function.args.args.append(ast.arg(arg=name))
        function.args.defaults.append(ast.Constant(value=0))
    else:
        function.body.insert(0, ast.Global(names=[name]))
    _assert_declined(function)


def test_extra_consumer_statement_and_changed_load_decline() -> None:
    function = _function(_TYPED)
    _loop(_outer(function).body[3]).body.append(ast.parse("saved = mi_next").body[0])
    _assert_declined(function)
    _assert_declined(
        _function(
            _TYPED.replace(
                "values = x[tile_m, tile_n].to(torch.float32)",
                "values = y[tile_m, tile_n].to(torch.float32)",
                1,
            )
        )
    )


def test_helper_names_are_fresh_across_parameters_and_after_outer_uses() -> None:
    function = _function(
        _TYPED.replace(
            "block_size_n=2):",
            "block_size_n=2, _helion_first_tile_max=17, _helion_first_tile_max_local=19):",
        ).replace(
            "return out",
            "return out, _helion_first_tile_max, _helion_first_tile_max_local",
        )
    )
    rewritten, count = _rewrite(function)
    assert count == 1
    assigned = {
        node.id
        for node in ast.walk(rewritten)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "_helion_first_tile_max" not in assigned
    assert "_helion_first_tile_max_local" not in assigned
    assert "_helion_first_tile_max_" in assigned
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    before = _evaluate(function, x, 2)
    after = _evaluate(rewritten, x, 2)
    assert isinstance(before, tuple) and isinstance(after, tuple)
    expected, *original_sentinels = before
    actual, *rewritten_sentinels = after
    assert original_sentinels == rewritten_sentinels == [17, 19]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["torch", "hl", "math", "float", "slice"])
def test_shadowed_required_bindings_decline(name: str) -> None:
    source = _TYPED
    if name == "math":
        source = source.replace("float('-inf')", "-math.inf")
    elif name == "slice":
        source = source.replace("[:, None]", "[slice(None, None, None), None]")
    host = _host(source, {name: object()})
    before = ast.dump(ast.Module(body=host.body, type_ignores=[]))
    assert not rewrite_online_to_3pass(host)
    assert ast.dump(ast.Module(body=host.body, type_ignores=[])) == before


def test_public_entry_and_opt_out_preserve_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(_TYPED)
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")
    before = ast.dump(ast.Module(body=host.body, type_ignores=[]))
    assert not rewrite_online_to_3pass(host)
    assert ast.dump(ast.Module(body=host.body, type_ignores=[])) == before
    monkeypatch.delenv("HELION_DISABLE_ONLINE_TO_3PASS")
    monkeypatch.setenv("HELION_ONLINE_TO_3PASS_MIN_N", "8")
    assert not rewrite_online_to_3pass(host)
    monkeypatch.setenv("HELION_ONLINE_TO_3PASS_MIN_N", "0")
    assert rewrite_online_to_3pass(host)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block", [1, 2, 3, 4, 8])
def test_actual_rewrite_preserves_ieee_classes(dtype: torch.dtype, block: int) -> None:
    values = [-math.inf, -1000.0, -1.0, -0.0, 0.0, 1.0, 1000.0, math.inf, math.nan]
    x = torch.tensor(list(itertools.product(values, repeat=4)), dtype=dtype)
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1
    expected = _execute(original, x, block)
    actual = _execute(rewritten, x, block)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-6 if dtype == torch.float32 else 2e-3,
        atol=2e-7,
        equal_nan=True,
    )


def test_first_logical_tile_not_a_lane_subgroup_controls_poison() -> None:
    x = torch.zeros((1, 4096), dtype=torch.bfloat16)
    x[:, torch.arange(4096) % 32 < 8] = -math.inf
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1
    expected = _execute(original, x, 4096)
    actual = _execute(rewritten, x, 4096)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leading = torch.tensor([[-math.inf, -math.inf, 0.0, 0.0]])
    assert torch.isnan(_execute(rewritten, leading, 2)).all()
    trailing = leading.flip(1)
    assert torch.isfinite(_execute(rewritten, trailing, 2)).all()


def test_global_max_and_sum_cannot_replace_first_tile_statistic() -> None:
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1

    # An executable ablation of the old two-statistic expectation. Retaining
    # only global max/sum silently loses the online recurrence's NaN history.
    class RemovePoison(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                ast.unparse(node.func) == "torch.where"
                and len(node.args) == 3
                and isinstance(node.args[2], ast.Name)
                and node.args[2].id == "di"
            ):
                return node.args[2]
            return node

    ablation = RemovePoison().visit(copy.deepcopy(rewritten))
    assert isinstance(ablation, ast.FunctionDef)
    leading = torch.tensor([[-math.inf, -math.inf, 0.0, 0.0]])
    trailing = leading.flip(1)
    assert torch.isnan(_execute(original, leading, 2)).all()
    assert torch.isnan(_execute(rewritten, leading, 2)).all()
    assert torch.isfinite(_execute(ablation, leading, 2)).all()
    torch.testing.assert_close(
        _execute(original, trailing, 2),
        _execute(rewritten, trailing, 2),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (3, 0)])
def test_empty_shapes_and_positive_zero_state(shape: tuple[int, int]) -> None:
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1
    x = torch.empty(shape)
    actual = _execute(rewritten, x, 2)
    assert actual.shape == shape and actual.numel() == 0
    if shape[0]:
        # Inspect internal state after matching; live-out user code itself is
        # deliberately rejected by the admission test above.
        for function in (original, rewritten):
            function.body[-1] = ast.parse("return out, mi, di").body[0]
        before = _evaluate(original, x, 2)
        after = _evaluate(rewritten, x, 2)
        assert isinstance(before, tuple) and isinstance(after, tuple)
        _, old_max, old_sum = before
        _, new_max, new_sum = after
        assert torch.equal(old_max, new_max) and torch.isneginf(new_max).all()
        assert torch.equal(old_sum, new_sum) and not torch.signbit(new_sum).any()


def test_narrowing_precedes_reduction_and_signed_zero_stays_positive() -> None:
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1
    x = torch.tensor(
        [[16777216.0, 16777217.0, 16777216.0, 16777217.0]], dtype=torch.float64
    )
    expected = _execute(original, x, 2)
    actual = _execute(rewritten, x, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.full_like(x, 0.25), rtol=0, atol=0)
    assert not torch.equal(actual, _execute(_function(_BARE), x, 2))
    zeros = torch.tensor([[0.0, -0.0, -0.0, 0.0]])
    normalized = _execute(rewritten, zeros, 2)
    assert not torch.signbit(normalized).any()
    torch.testing.assert_close(normalized, torch.full_like(zeros, 0.25), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_finite_reassociation_uses_the_existing_tolerance(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(2026091361)
    x = torch.randn(64, 257, generator=generator)
    x *= torch.tensor([0.05, 1.0, 20.0, 80.0]).repeat(16)[:, None]
    x = x.to(dtype)
    original = _function(_TYPED)
    rewritten, count = _rewrite(original)
    assert count == 1
    for block in (1, 3, 32, 128, 512):
        torch.testing.assert_close(
            _execute(rewritten, x, block),
            _execute(original, x, block),
            rtol=1e-2 if dtype != torch.float32 else 3e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize(
    ("blocks", "threads"),
    [([1, 4096], [0, 128]), ([1, 4096], [0, 256]), ([4, 4096], [4, 32])],
)
def test_original_aot_has_three_complete_logical_reductions(
    blocks: list[int], threads: list[int]
) -> None:
    pytest.importorskip("cutlass")
    source = _function(inspect.getsource(row_softmax.fn))
    assert (
        _detect_online_softmax(_outer(source).body, scope_body=source.body) is not None
    )
    kernel = helion.kernel(row_softmax.fn, backend="cute", static_shapes=False)
    assert not kernel.settings.fast_math
    bound = kernel._bind_isolated((torch.empty(17, 4097, dtype=torch.bfloat16),))
    generated = bound.to_code(
        helion.Config(
            block_sizes=blocks,
            num_threads=threads,
            cute_vector_widths=[8, 8],
            cute_lane_layouts=["blocked", "blocked"],
        )
    )
    assert not kernel.settings.fast_math
    tree = ast.parse(generated)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    operations = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {"_cute_grouped_reduce_shared_two_stage", "_cute_grouped_reduce_warp"}
        ):
            continue
        operations.append(ast.literal_eval(node.args[1]))
        ancestor = node
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.For):
                assert ast.unparse(ancestor.target).startswith("tile_offset_")
    assert sorted(operations) == ["max", "max", "sum"]
    # Positive shape hints let ShapeEnv fold n > 0 in generated code. Check
    # the source guard before that specialization; empty execution is above.
    rewritten, count = _rewrite(source)
    assert count == 1
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.ops[0], ast.Gt)
        and ast.unparse(node.left) == "n"
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
        for node in ast.walk(rewritten)
    )
