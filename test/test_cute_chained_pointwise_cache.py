from __future__ import annotations

import ast

import pytest
import torch

from .test_cute_chained_scan_export import _args
from .test_cute_chained_scan_export import _config
from .test_cute_chained_scan_export import _cpu_codegen
from .test_cute_chained_scan_export import _scan_export
from .test_cute_chained_tcgen05_config import _without_early_release_seed
import helion
from helion import exc
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_pointwise_cache import PointwiseReadCache
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import EnumFragment

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_pointwise_read_cache"


def _code(value: object = False, *, include: bool = True, kind: str = "dense") -> str:
    config = _config().config | {"cute_chained_pointwise_vectorize": True}
    if include:
        config[KEY] = value
    with _cpu_codegen():
        return _scan_export._bind_isolated((*_args(kind=kind), "normal")).to_code(
            helion.Config.from_dict(config)
        )


def test_default_source_is_exact() -> None:
    assert _code() == _code(include=False)


@pytest.mark.parametrize("kind", ["dense", "offset", "stride"])
def test_scan_cache_is_typed_guarded_and_after_publication(kind: str) -> None:
    source = _code(True, kind=kind)
    assert "read_cache_" in source
    tree = ast.parse(source)
    declarations = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and "read_cache_" in n.targets[0].id
    ]
    assert len(declarations) == 1
    assert ast.unparse(declarations[0].value).endswith("cutlass.Float32)")
    cache_loop = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.endswith("_cache_element")
    )
    preload = ast.unparse(cache_loop)
    assert "chain_scan_0_values" in preload and "< 128" in preload
    assert "pointwise_row" not in preload
    branch = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and cache_loop in n.body
    )
    assert "toint() % 16 == 0" in ast.unparse(branch.test)
    assert "_read_cache_" not in ast.unparse(
        ast.Module(body=branch.orelse, type_ignores=[])
    )
    assert source.index("cute.arch.sync_threads()") < source.index("read_cache_")
    old = ast.parse(_code(False, kind=kind))
    assert ast.dump(tree.body[-1]) == ast.dump(old.body[-1])
    for term in ("cute.gemm", "cute.arch.sync_threads", "cute.arch.mbarrier_wait"):

        def calls(module: ast.AST, term: str) -> list[str]:
            return [
                ast.dump(n)
                for n in ast.walk(module)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == term
            ]

        assert calls(tree, term) == calls(old, term)


@pytest.mark.parametrize("value", [0, 1, 2, "true", None])
def test_invalid_cache_values(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _code(value)


def _expression(rhs: str, dtype: torch.dtype = torch.bfloat16) -> _Expression:
    graph = torch.fx.Graph()
    source = graph.placeholder("src")
    source.meta["val"] = torch.empty(1, dtype=dtype)
    load = graph.call_function(torch.clone, (source,))
    # This unit fixture supplies the real typed expression catalog consumed by
    # prepare; it does not evaluate FX nodes or require a GenerateAST instance.
    expression = _Expression.__new__(_Expression)
    expression.plan = ChainedMatmulPlan(
        root_graph_id=0,
        dots=(),
        store=graph.output(load),
        axes=(),
        shapes=(),
        dtype=dtype,
        threads=128,
    )
    expression.definitions = {"loaded": rhs}
    expression.loaded_inputs = [(load, (), [], "loaded")]
    expression.memo = {}
    expression.lines = [f"loaded = {rhs}", "result = cutlass.Float32(loaded) * row"]
    return expression


@pytest.mark.parametrize(
    "rhs",
    [
        "src[col] if row < 8 else cutlass.BFloat16(0)",
        "src[col + row]",
        "src[col] if col < limit else cutlass.BFloat16(0)",
        "src[row]",
        "src[0]",
    ],
)
def test_complete_rhs_and_temporary_dependency_rejected(rhs: str) -> None:
    expression = _expression(rhs)
    expression.definitions["limit"] = "row + 1"
    old = list(expression.lines)
    cache = PointwiseReadCache(True)
    assert cache.prepare(expression, ("row", "col"), "e", "stage", 16, 16) == []
    assert expression.lines == old
    with pytest.raises(exc.BackendUnsupported, match="row-invariant"):
        cache.validate()


@pytest.mark.parametrize(
    "dtype,name",
    [
        (torch.bfloat16, "cutlass.BFloat16"),
        (torch.float16, "cutlass.Float16"),
        (torch.float32, "cutlass.Float32"),
    ],
)
def test_native_load_dtype_and_complete_rhs_preserved(
    dtype: torch.dtype, name: str
) -> None:
    rhs = f"src[col] if 0 <= col < 127 else {name}(0)"
    expression = _expression(rhs, dtype)
    cache = PointwiseReadCache(True)
    with _cpu_codegen():
        # Use a real environment's dtype rendering while keeping this test of
        # planner rejection independent of a particular contraction kernel.
        bound = _scan_export._bind_isolated((*_args(), "normal"))
        with bound.env:
            lines = cache.prepare(expression, ("row", "col"), "e", "stage", 16, 16)
    assert lines[0].endswith(f"{name})")
    assert (
        lines[-1]
        == f"    stage_read_cache_0[stage_cache_element] = src[stage_cache_col] if 0 <= stage_cache_col < 127 else {name}(0)"
    )
    assert expression.lines == [
        "loaded = stage_read_cache_0[e]",
        "result = cutlass.Float32(loaded) * row",
    ]
    cache.validate()


def test_single_trip_and_disabled_leave_every_byte_alone() -> None:
    for enabled, trips in ((False, 16), (True, 1)):
        expression = _expression("src[col]")
        old = list(expression.lines)
        assert (
            PointwiseReadCache(enabled).prepare(
                expression, ("row", "col"), "e", "stage", 16, trips
            )
            == []
        )
        assert expression.lines == old


def test_activation_is_private() -> None:
    cache = PointwiseReadCache(True)
    PointwiseReadCache(False).validate()
    with pytest.raises(exc.BackendUnsupported):
        cache.validate()


def test_search_knob_default_and_seed_siblings() -> None:
    with _cpu_codegen():
        bound = _scan_export._bind_isolated((*_args(), "normal"))
    spec = bound.config_spec
    assert spec.cute_chained_pointwise_read_cache_search_enabled
    fragment = spec._flat_fields()[KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment.search_values() == [False, True]
    seeds = _without_early_release_seed(spec.compiler_seed_configs, spec)
    assert seeds and not seeds[0].config.get(KEY, False)
    cached = [seed for seed in seeds if seed.config.get(KEY)]
    assert cached
    for seed in cached:
        assert seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
        assert seed.config["cute_chained_pointwise_vectorize"]
        parent = {k: v for k, v in seed.config.items() if k != KEY}
        assert any(
            parent == {k: v for k, v in other.config.items() if k != KEY}
            for other in seeds
            if not other.config.get(KEY, False)
        )
