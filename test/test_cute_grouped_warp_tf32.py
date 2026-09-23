from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_flat_grouped_public import _native_source
from test.test_cute_flat_grouped_public import _source

import helion
from helion._compiler.cute.grouped_warp_tf32_kernel import grouped_warp_tf32_module
from helion._compiler.cute.grouped_warp_tf32_plan import GroupedWarpTF32Plan
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import (
    normalize_grouped_rna_config,
)
from helion._compiler.cute.tcgen05_tma_rn import CONVERSION_KEY
from helion._compiler.cute.tcgen05_tma_rn import TMA_RN
from helion._compiler.cute.tcgen05_tma_rn import WARP_RAW
from helion.autotuner.config_generation import ConfigGeneration


def _config(bound):
    return helion.Config.from_dict(
        bound.config_spec._base_default_config().config | {CONVERSION_KEY: WARP_RAW}
    )


@pytest.mark.parametrize("k,n", [(64, 64), (128, 128), (192, 68), (64, 4)])
def test_ordinary_binding_emits_the_complete_raw_tf32_family(k, n):
    with cuda_binding(
        static=True, original=True, groups=5, rows=131, k=k, n=n
    ) as bound:
        spec = bound.config_spec
        assert spec.cute_grouped_warp_tf32_available
        config = spec.normalized_config(_config(bound))
        assert config[CONVERSION_KEY] == WARP_RAW
        assert WARPS_KEY not in config.config and K_KEY not in config.config
        assert any(
            seed.get(CONVERSION_KEY) == WARP_RAW for seed in spec.compiler_seed_configs
        )
        generator = ConfigGeneration(spec)
        restored = generator.unflatten(generator.flatten(config))
        assert restored[CONVERSION_KEY] == WARP_RAW
        source = _source(bound, config)
        device = _native_source(source)
        plan = GroupedWarpTF32Plan(5, 131, k, n, 232448)
        assert device == grouped_warp_tf32_module(plan)
        assert f"(5, {(n + 63) // 64}, 1), seq_offsets" in source
        assert "block=(512, 1, 1)" in source
        assert (
            f"jagged.view((131, {k}))" in source and "dense.transpose(1, 2)" in source
        )
        assert "_helion_cute_wrapper_plans = []" in device
        assert "_helion_cute_disable_bake_tensor_shapes = True" in device
        assert "MmaTF32Op((16, 8, 8))" in device
        assert device.count("LdMatrix8x8x16bOp") == 2
        assert "cvt" not in device and "convert" not in device
        assert "tcgen05." not in device


@pytest.mark.parametrize(
    "case",
    [
        "dynamic",
        "ieee",
        "tf32x3",
        "offset32",
        "offset_stride",
        "misalign_a",
        "misalign_b",
        "misalign_bias",
        "wrong_stride",
        "extra_math",
        "k_tail",
        "n_vector_tail",
    ],
)
def test_unsupported_precision_memory_or_ir_never_selects_raw(case):
    with cuda_binding(
        static=case != "dynamic",
        original=case not in ("wrong_stride", "extra_math"),
        k=96 if case == "k_tail" else 64,
        n=66 if case == "n_vector_tail" else 64,
        bits=32 if case == "offset32" else 64,
        offsets_stride=2 if case == "offset_stride" else 1,
        misalign={"misalign_a": 1, "misalign_b": 2, "misalign_bias": 3}.get(case),
        mode=case if case in ("wrong_stride", "extra_math") else "ordinary",
        dot_precision=case if case in ("ieee", "tf32x3") else "tf32",
    ) as bound:
        assert not bound.config_spec.cute_grouped_warp_tf32_available
        with pytest.raises(helion.exc.InvalidConfig, match="raw warp TF32 requires"):
            bound.config_spec.normalized_config(_config(bound))
        config = _config(bound)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert config == bound.config_spec._base_default_config()


def test_raw_and_explicit_rn_are_separate_canonical_precision_choices():
    rn = {CONVERSION_KEY: TMA_RN, K_KEY: 64}
    normalize_grouped_rna_config(
        rn, available_k=(32, 64), warp_raw_available=True, fix_invalid=False
    )
    assert rn == {CONVERSION_KEY: TMA_RN, K_KEY: 64}
    raw = {
        CONVERSION_KEY: WARP_RAW,
        K_KEY: 32,
        WARPS_KEY: 8,
        "cute_grouped_ab_stages": 2,
    }
    normalize_grouped_rna_config(
        raw, available_k=(32, 64), warp_raw_available=True, fix_invalid=False
    )
    assert raw == {CONVERSION_KEY: WARP_RAW}
    default = {"ordinary": 1}
    normalize_grouped_rna_config(
        default, available_k=(32, 64), warp_raw_available=True, fix_invalid=False
    )
    assert default == {"ordinary": 1}


def test_complete_simultaneous_storage_and_grid_limits():
    plan = GroupedWarpTF32Plan(5, 131, 128, 68, 82944)
    assert plan.thread_block == (512, 1, 1) and plan.grid == (5, 2, 1)
    assert plan.shared_bytes == 32768 + 32768 + 16384
    for changes in (
        {"shared_capacity": 82943},
        {"shared_capacity": True},
        {"reduction": 96},
        {"columns": 66},
        {"columns": 65536 * 64},
        {"rows": (1 << 31) - 1},
        {"groups": 0},
        {"groups": True},
    ):
        with pytest.raises(ValueError):
            replace(plan, **changes)


def _kernel(source):
    return next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_helion_native_grouped_rna"
    )


def _calls(statements):
    return [
        ast.unparse(node.func)
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]


@pytest.mark.parametrize("k", [64, 128, 192])
def test_k_refill_preserves_accumulator_and_full_cta_publication(k):
    source = grouped_warp_tf32_module(GroupedWarpTF32Plan(3, 273, k, 68, 232448))
    kernel = _kernel(source)
    rows = next(n for n in kernel.body if isinstance(n, ast.While))
    assert _calls(rows.body).count("acc.fill") == 1
    refill = [
        n
        for n in rows.body
        if isinstance(n, ast.For) and ast.unparse(n.target) == "k_block"
    ]
    assert len(refill) == (k > 64)
    if refill:
        assert ast.unparse(refill[0].iter) == f"cutlass.range(1, {k // 64}, unroll=1)"
        calls = _calls(refill[0].body)
        assert calls.index("cute.arch.sync_threads") < calls.index(
            "_grouped_warp_copy_ab"
        )
        wait = calls.index("cute.arch.cp_async_wait_group")
        assert calls[wait + 1 : wait + 4] == [
            "cute.arch.sync_threads",
            "cute.copy",
            "cute.copy",
        ]
        assert calls.index("cute.gemm") > calls.index("cute.copy")
        assert "acc.fill" not in calls
        bias = next(
            n
            for n in rows.body
            if isinstance(n, ast.For) and ast.unparse(n.target) == "half_n"
        )
        assert rows.body.index(refill[0]) < rows.body.index(bias)
    assert _calls(kernel.body[-2:]) == [
        "cute.arch.cp_async_wait_group",
        "cute.arch.sync_threads",
    ]


@pytest.mark.parametrize("k,n", [(64, 64), (128, 128), (192, 68), (64, 4)])
def test_actual_copy_expressions_cover_each_tail_and_refill(k, n):
    # Evaluate the emitted address/predicate expressions as integers. The
    # unchanged shared swizzle is already covered by the SDK transport tests;
    # this test targets new K/N bases, validity and packet multiplicity.
    source = grouped_warp_tf32_module(GroupedWarpTF32Plan(3, 273, k, n, 232448))
    helper = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_grouped_warp_copy_ab"
    )
    helper.decorator_list = []
    transfers = []
    scope = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int, range_constexpr=range),
        "cute": SimpleNamespace(
            crd2idx=lambda coordinate, layout: coordinate[0] * 64 + coordinate[1],
            arch=SimpleNamespace(
                cp_async_shared_global=lambda *args, **kwargs: transfers.append(
                    (args, kwargs)
                ),
                cp_async_commit_group=lambda: None,
            ),
        ),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "<emitted_copy>", "exec"),
        scope,
    )
    copy = scope[helper.name]
    a = SimpleNamespace(iterator=0, layout=None)
    b = SimpleNamespace(iterator=65536, layout=None)
    lhs = SimpleNamespace(iterator=0, layout=SimpleNamespace(stride=(k, 1)))
    rhs = SimpleNamespace(
        iterator=1 << 30, layout=SimpleNamespace(stride=(k * n, 1, n))
    )
    for length in (0, 1, 127, 128, 129, 273):
        for offset in (0, 128, 256, 384):
            for kb in range(0, k, 64):
                for nb in range(0, n, 64):
                    transfers.clear()
                    for thread in range(512):
                        extra = ([kb] if k > 64 else []) + ([nb] if n != 64 else [])
                        copy(a, b, lhs, rhs, 7, length, 2, offset, thread, *extra)
                    got_a, got_b = Counter(), Counter()
                    for (destination, address, width, policy), options in transfers:
                        size = options["cp_size"]
                        assert size in (0, width)
                        if width == 16:
                            assert destination < 32768 and policy == "cg"
                            got_a.update(
                                address + element for element in range(size // 4)
                            )
                        else:
                            assert (
                                destination >= 65536 and width == 4 and policy == "ca"
                            )
                            if size:
                                got_b[address - (1 << 30)] += 1
                    assert got_a == Counter(
                        (7 + row) * k + inner
                        for row in range(offset, min(offset + 128, length))
                        for inner in range(kb, kb + 64)
                    )
                    expected_b = Counter(
                        2 * k * n + inner * n + column
                        for inner in range(kb, kb + 64)
                        for column in range(nb, min(nb + 64, n))
                    )
                    assert got_b == (expected_b if offset < length else Counter())


@pytest.mark.parametrize("n", [4, 64, 68, 128, 132])
def test_actual_store_and_bias_addresses_preserve_vector_tail_bounds(n):
    kernel = _kernel(
        grouped_warp_tf32_module(GroupedWarpTF32Plan(3, 4674, 128, n, 232448))
    )
    stores = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and "cutlass.Int64(row)" in ast.unparse(node.test)
    ]
    assert len(stores) == 1
    destination = next(
        node.value
        for node in ast.walk(stores[0])
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "destination"
    )
    condition = compile(ast.Expression(stores[0].test), "<actual_store_guard>", "eval")
    address = compile(ast.Expression(destination), "<actual_store_address>", "eval")
    bias_loop = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "half_n"
    )
    bias_if = next((node for node in bias_loop.body if isinstance(node, ast.If)), None)
    bias_guard = (
        compile(ast.Expression(bias_if.test), "<actual_bias_guard>", "eval")
        if bias_if
        else None
    )
    bias_address = next(
        node.value
        for node in ast.walk(bias_loop)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) in ("source", "bias_source")
    )
    bias_address = compile(
        ast.Expression(bias_address), "<actual_bias_address>", "eval"
    )
    scope = {
        "cutlass": SimpleNamespace(Int64=int),
        "start": 7,
        "group": 2,
        "out": SimpleNamespace(iterator=0, layout=SimpleNamespace(stride=(n, 1))),
        "bias": SimpleNamespace(iterator=0, layout=SimpleNamespace(stride=(n, 1))),
    }
    for length in (0, 1, 127, 128, 129, 4674):
        got = Counter()
        for nb in range(0, n, 64):
            scope["n_base"] = nb
            for offset in range(0, length + 128, 128):
                for t in range(512):
                    scope.update(t=t, length=length)
                    for packet in range(4):
                        scope["row"] = offset + t // 16 + packet * 32
                        if eval(condition, scope):
                            first = eval(address, scope)
                            assert first % 4 == 0
                            got.update(range(first, first + 4))
            for column in range(0, 64, 2):
                scope["column"] = column
                if bias_guard is None or eval(bias_guard, scope):
                    first = eval(bias_address, scope)
                    assert 2 * n <= first and first + 1 < 3 * n and first % 2 == 0
        assert got == Counter(range(7 * n, (7 + length) * n))
