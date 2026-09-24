from __future__ import annotations

import ast
from dataclasses import replace
from itertools import starmap
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_scan_export import _args as scan_args
from .test_cute_chained_scan_export import _config
from .test_cute_chained_scan_export import _cpu_codegen
from .test_cute_chained_scan_export import _scan_export
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute import _with_coefficient_cache_seed
from helion._compiler.cute import chained_coefficient_cache as coefficient
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen05 as emitter
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
KEY = "cute_chained_coefficient_cache"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _weighted(a, b, scale, bias, mode: hl.constexpr):
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=torch.float32, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        factor = torch.exp(scale[bi, kk].float().clamp(max=0)) * bias[bi].float()
        if mode == "add_sub":
            factor = (factor + 0.75) - 0.25
        elif mode == "narrow":
            factor = factor.to(a.dtype).float()
        elif mode == "fp64":
            factor = torch.exp(scale[bi, kk].double()).float()
        left = (a[bi, row, kk].float() * factor[None, :]).to(a.dtype)
        right = b[bi, kk, col]
        if mode == "two":
            other = torch.exp(scale[bi, kk].float() + 0.125)
            right = (right.float() * other[:, None]).to(a.dtype)
        out[bi, row, col] = hl.dot(left, right)
    return out


def _args(dtype=torch.bfloat16, kind="dense", reduction=128):
    a = torch.empty((2, 128, reduction), dtype=dtype)
    b = torch.empty((2, reduction, 64), dtype=dtype)
    if kind == "stride":
        scale = torch.empty((2, reduction * 2), dtype=dtype)[:, ::2]
    elif kind == "offset":
        scale = torch.empty((2 * reduction + 8,), dtype=dtype)[8:].view(2, reduction)
    elif kind == "short":
        scale = torch.empty((2, reduction // 2), dtype=dtype)
    else:
        scale = torch.empty((2, reduction), dtype=dtype)
    return a, b, scale, torch.empty((2,), dtype=dtype)


def _source(args, value: Any = True, *, vector=False, mode="normal", extra=None):
    config = _config().config | {"cute_chained_pointwise_vectorize": vector}
    if value is not None:
        config[KEY] = value
    config.update(extra or {})
    with _cpu_codegen():
        return _weighted._bind_isolated((*args, mode)).to_code(
            helion.Config.from_dict(config)
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "stride", "offset"])
@pytest.mark.parametrize("vector", [False, True])
def test_publication_default_host_allocation_and_lifetime(dtype, kind, vector):
    args = _args(dtype, kind)
    old = _source(args, None, vector=vector)
    assert _source(args, False, vector=vector) == old
    new = _source(args, vector=vector)
    assert "chain_coefficient_0 = cute.make_tensor(chain_output_ptr + 0" in new
    assert (
        "chain_coefficient_0_index = chain_thread + chain_coefficient_0_step * 128"
        in new
    )
    assert "if chain_coefficient_0_index < 128:" in new
    assert (
        new.count("cute.arch.sync_threads()")
        == old.count("cute.arch.sync_threads()") + 1
    )
    assert [line for line in old.splitlines() if "alloc_smem(" in line] == [
        line for line in new.splitlines() if "alloc_smem(" in line
    ]
    assert ast.dump(ast.parse(old).body[-1]) == ast.dump(ast.parse(new).body[-1])
    begin = new.index("chain_coefficient_0 =")
    publication = new.index("cute.arch.sync_threads()", begin)
    consume = new.index("chain_0_a_ptr =", publication)
    pre_mma = new.index("cute.arch.sync_threads()", consume)
    overwrite = new.index("chain_epi_values")
    assert begin < publication < consume < pre_mma < overwrite


def test_published_scan_precedes_coefficient_and_terminal_retained():
    config = _config().config | {KEY: True}
    with _cpu_codegen():
        old = _scan_export._bind_isolated((*scan_args(), "normal")).to_code(_config())
        new = _scan_export._bind_isolated((*scan_args(), "normal")).to_code(
            helion.Config.from_dict(config)
        )
    assert new.index("chain_scan_0_values") < new.index("chain_coefficient_0 =")
    assert "cutlass.Float32(chain_scan_0_values[127])" in new
    assert (
        new.count("chain_scan_0_pointer =") == old.count("chain_scan_0_pointer =") == 1
    )


@pytest.mark.parametrize("value", [0, 1, "true", [], 2])
def test_strict_bool(value):
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _source(_args(), value)


def test_actual_initial100_and_filtered_old_seed_order():
    with _cpu_codegen():
        bound = _weighted._bind_isolated((*_args(), "normal"))
        spec = bound.config_spec
        assert spec.cute_chained_coefficient_cache_search_enabled
        assert bound.host_function is not None
        with bound.env:
            new = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
            spec.cute_chained_coefficient_cache_search_enabled = False
            try:
                old = CuteChainedMatmulHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
            finally:
                spec.cute_chained_coefficient_cache_search_enabled = True
            assert new is not None and old is not None
            assert [seed for seed in new if not seed.config.get(KEY)] == old
            generation = ConfigGeneration(spec)
            population = [
                generation.unflatten(item)
                for item in generation.random_population_flat(100)
            ]
            candidates = [seed for seed in population if seed.config.get(KEY)]
            assert candidates
            assert not population[0].config.get(KEY)
        assert "chain_coefficient_0 =" in bound.to_code(candidates[0])


def test_sibling_preserves_duplicate_objects_and_first_seed():
    first = helion.Config(cute_chained_mma_schedule="coalesced")
    parent = helion.Config(cute_chained_mma_schedule="tcgen05_tmem")
    old = [first, parent, parent, first]
    assert _with_coefficient_cache_seed(old, enabled=False) is old
    new = _with_coefficient_cache_seed(old, enabled=True)
    restored = [seed for seed in new if not seed.config.get(KEY)]
    assert all(a is b for a, b in zip(old, restored, strict=True))
    assert new[0] is first


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ([], True),
        (["0 <= (127) < 128"], True),
        (["0 <= 0 < 1"], True),
        (["0 <= 128 < 128"], False),
        (["0 <= q < 128"], False),
        (["q + origin < 128"], False),
        (["True"], False),
        (["0 <= True < 128"], False),
        (["0 <= (-1) < 128"], False),
        (["-1 <= 0 < 128"], False),
    ],
)
def test_only_literal_domain_tautologies(bounds, expected):
    assert coefficient._constant_true_domain(bounds) is expected


@pytest.mark.parametrize(
    "coordinate,expected",
    [
        ("row", True),
        ("col", True),
        ("row + 0", True),
        ("row + 1", False),
        ("row + col", False),
        ("127 - row", False),
        ("row % 128", False),
        ("cutlass.Int32(row) + 1", False),
    ],
)
def test_exact_consumer_axis_projection(coordinate, expected):
    expression: Any = SimpleNamespace(origins={}, definitions={})
    assert (
        coefficient._axis_projection(
            expression, coordinate, ("row", "col"), (128, 128), 128
        )
        is expected
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"cute_chained_direct_output": True},
        {"cute_chained_mma_schedule": "coalesced"},
    ],
)
def test_unsupported_schedule_or_arena_rejects(extra):
    with pytest.raises(exc.InvalidConfig):
        _source(_args(), extra=extra)


def test_no_fp64_precision_escape():
    with pytest.raises(exc.InvalidConfig, match="coefficient cache"):
        _source(_args(), mode="fp64")


def test_partial_logical_axis_rejects():
    with pytest.raises(exc.BackendUnsupported):
        _source(_args(reduction=49))


def test_output_alias_proof_on_real_fx_leaf():
    args = (*_args()[:2], torch.empty((2, 128)), torch.empty((2,)))
    original = emitter.make_coefficient_cache

    def aliased(cg, plan, boundaries, scans, published):
        root = plan.store.args[0]
        saved = root.meta["val"]
        source = next(
            node
            for node in chain._ancestors(plan.dots[0].args[0])
            if node.target is chain._tracing_ops._host_tensor
            and node.meta["val"].dtype == torch.float32
            and node.meta["val"].ndim == 2
        )
        root.meta["val"] = source.meta["val"]
        try:
            return original(cg, plan, boundaries, scans, published)
        finally:
            root.meta["val"] = saved

    with (
        patch.object(emitter, "make_coefficient_cache", aliased),
        pytest.raises(exc.BackendUnsupported, match="aliases an output"),
    ):
        _source(args)


def test_unpublished_scan_cannot_enter_pure_coefficient():
    original = emitter.make_coefficient_cache
    observed = []

    def inspect(cg, plan, boundaries, scans, published):
        assert published == set(plan.scans) and published
        nodes = [
            node
            for node in chain._ancestors(plan.dots[0].args[0])
            if node.target is torch.ops.aten.exp.default
        ]
        assert nodes
        for node in nodes:
            assert coefficient._pure_computed(node, published)
            assert not coefficient._pure_computed(node, set())
        observed.append(True)
        return original(cg, plan, boundaries, scans, published)

    with _cpu_codegen(), patch.object(emitter, "make_coefficient_cache", inspect):
        _scan_export._bind_isolated((*scan_args(), "normal")).to_code(
            helion.Config.from_dict(_config().config | {KEY: True})
        )
    assert observed == [True]


class _Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset):
        return _Pointer(self.tensor, self.offset + int(offset))

    def load(self):
        storage = self.tensor.as_strided(
            (self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),),
            (1,),
            storage_offset=0,
        )
        return storage[self.tensor.storage_offset() + self.offset]


def _source_environment(source, args):
    def typed(dtype):
        return lambda value: torch.as_tensor(value, dtype=dtype)

    types = SimpleNamespace(
        Float32=typed(torch.float32),
        BFloat16=typed(torch.bfloat16),
        Float16=typed(torch.float16),
        Int32=int,
    )
    math = SimpleNamespace(
        exp2=torch.exp2,
        min=lambda a, b, propagate_nan: torch.minimum(a, b),
        max=lambda a, b, propagate_nan: torch.maximum(a, b),
    )
    env: dict[str, Any] = {
        "cutlass": types,
        "cute": SimpleNamespace(math=math),
        "chain_origin_0": 0,
        "chain_origin_1": 0,
        "chain_origin_2": 0,
        "chain_0_a": {},
        "out": SimpleNamespace(),
    }
    env.update(
        {
            name: SimpleNamespace(
                iterator=_Pointer(value), layout=SimpleNamespace(stride=value.stride())
            )
            for name, value in zip(("a", "b", "scale", "bias"), args, strict=True)
        }
    )
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
    )
    for node in function.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("input_tensor")
        ):
            exec(
                compile(ast.Module([node], type_ignores=[]), "<source-alias>", "exec"),
                env,
            )
    return env


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["normal", "add_sub", "narrow"])
@pytest.mark.parametrize("kind", ["stride", "short"])
def test_actual_emitted_scalar_arithmetic_exact_bits(dtype, mode, kind):
    args = _args(dtype, kind)
    generator = torch.Generator().manual_seed(77)
    for tensor in args:
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
    before_inputs = [value.clone() for value in args]
    sources = [_source(args, value, mode=mode) for value in (False, True)]
    environments = [_source_environment(source, args) for source in sources]
    new = ast.parse(sources[1])
    producer = next(
        node
        for node in ast.walk(new)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_coefficient_0_step"
    )
    env = environments[1]
    env["chain_coefficient_0"] = torch.empty(128, dtype=torch.float32)
    env["chain_coefficient_0_step"] = 0
    producer_code = compile(
        ast.Module(producer.body, type_ignores=[]), "<emitted-producer>", "exec"
    )
    for thread in range(128):
        env["chain_thread"] = thread
        exec(producer_code, env)
    loops = [
        next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "chain_0_a_step"
        )
        for source in sources
    ]
    for thread in (0, 1, 7, 31, 63, 127):
        for step in (0, 1, 31, 63, 127):
            for loop, env in zip(loops, environments, strict=True):
                env.update(chain_thread=thread, chain_0_a_step=step)
                exec(
                    compile(
                        ast.Module(loop.body, type_ignores=[]),
                        "<emitted-operand>",
                        "exec",
                    ),
                    env,
                )
    old, new = [env["chain_0_a"] for env in environments]
    assert old.keys() == new.keys()
    assert all(
        torch.equal(old[key].view(torch.int16), new[key].view(torch.int16))
        for key in old
    )
    assert all(starmap(torch.equal, zip(args, before_inputs, strict=True)))


def test_two_coefficients_have_disjoint_slots_and_one_publication():
    old = _source(_args(), False, mode="two")
    new = _source(_args(), True, mode="two")
    assert "chain_coefficient_0 = cute.make_tensor(chain_output_ptr + 0" in new
    assert "chain_coefficient_1 = cute.make_tensor(chain_output_ptr + 128" in new
    assert (
        new.count("cute.arch.sync_threads()")
        == old.count("cute.arch.sync_threads()") + 1
    )
    assert not set(range(128)) & set(range(128, 256))


@pytest.mark.parametrize(
    "rows,columns", [(64, 32), (64, 96), (128, 32), (128, 96), (128, 256)]
)
def test_shared_arena_geometry_and_exact_loader_coverage(rows, columns):
    a, b, scale, bias = _args()
    a = torch.empty((2, rows, 128), dtype=a.dtype)
    b = torch.empty((2, 128, columns), dtype=b.dtype)
    if columns == 96:
        # This ordinary tunable axis is power-of-two-only before lowering;
        # the cache must not broaden the existing shape/config envelope.
        for enabled in (False, True):
            with pytest.raises(exc.InvalidConfig, match="power of two"):
                _source(
                    (a, b, scale, bias), enabled, extra={"block_sizes": [rows, columns]}
                )
        return
    source = _source((a, b, scale, bias), extra={"block_sizes": [rows, columns]})
    assert f"alloc_smem(cutlass.Float32, {rows * columns}, alignment=128)" in source
    assert "chain_coefficient_0 =" in source
    coverage = [
        thread + step * 128
        for step in range(1)
        for thread in range(128)
        if thread + step * 128 < 128
    ]
    assert coverage == list(range(128))


def test_arena_guards_use_the_actual_single_dot_plan():
    original = emitter.make_coefficient_cache
    observed = []

    def inspect(cg, plan, boundaries, scans, published):
        assert coefficient.separate_output_capacity(plan) == 8192
        assert (
            coefficient.separate_output_capacity(replace(plan, direct_output=True)) == 0
        )
        assert coefficient.separate_output_capacity(replace(plan, threads=256)) == 0
        assert (
            coefficient.separate_output_capacity(replace(plan, dots=plan.dots * 2)) == 0
        )
        axis, extent, block = plan.axes[-1]
        assert (
            coefficient.separate_output_capacity(
                replace(plan, axes=(*plan.axes[:-1], (axis, extent - 1, block)))
            )
            == 0
        )
        observed.append(True)
        return original(cg, plan, boundaries, scans, published)

    with patch.object(emitter, "make_coefficient_cache", inspect):
        _source(_args())
    assert observed == [True]
