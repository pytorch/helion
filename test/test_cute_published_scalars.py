from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_atomic_ops import _simulate_register_load_program

import helion
from helion._compiler.cute.published_scalars import reuse_published_scalars
from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl

KEY = "cute_fragment_published_scalars"


def _transform(source):
    body = ast.parse(source).body
    counter = itertools.count()
    changed = reuse_published_scalars(
        body, {"shared"}, "thread", 32, lambda p: f"{p}_{next(counter)}"
    )
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    ), changed


_SOURCE = """
for owner in range(thread, 1, 32):
    shared[0] = cutlass.Int32(source)
cute.arch.sync_threads()
for i in range(thread, 65, 32):
    a = shared[0]
    b = cutlass.Int32(a)
    c = cutlass.Int32(b + 3)
    out[i] = cutlass.Int32(c + i)
cute.arch.sync_threads()
for j in range(thread, 65, 32):
    d = cutlass.Int32(shared[0])
    e = cutlass.Int32(d + 3)
    out[j] = cutlass.Int32(e + j)
cute.arch.sync_threads()
"""


def test_exact_source_reuse_and_live_typed_definitions():
    transformed, count = _transform(_SOURCE)
    assert count > 0
    assert transformed.count("sync_threads()") == _SOURCE.count("sync_threads()")
    assert transformed.count("shared[0]") <= _SOURCE.count("shared[0]")
    assert all(
        "shared[0]" not in ast.unparse(loop)
        for loop in ast.parse(transformed).body
        if isinstance(loop, ast.For) and ast.unparse(loop.target) in ("i", "j")
    )
    for value in (-2147483648, -17, 0, 2147483644):
        for order in (range(32), reversed(range(32))):
            results = []
            for code in (_SOURCE, transformed):
                out = [None] * 65
                # Execute independently published phases for every lane.
                statements = ast.parse(code).body
                phases = [[]]
                for statement in statements:
                    if (
                        isinstance(statement, ast.Expr)
                        and ast.unparse(statement) == "cute.arch.sync_threads()"
                    ):
                        phases.append([])
                    else:
                        phases[-1].append(statement)
                envs = [
                    {
                        "shared": [None],
                        "source": np.int32(value),
                        "thread": t,
                        "out": out,
                        "cutlass": SimpleNamespace(Int32=np.int32),
                    }
                    for t in range(32)
                ]
                shared = [None]
                for env in envs:
                    env["shared"] = shared
                lanes = list(order)
                if not lanes:
                    lanes = list(reversed(range(32)))
                for phase in phases:
                    for t in lanes:
                        module = ast.fix_missing_locations(
                            ast.Module(body=phase, type_ignores=[])
                        )
                        with np.errstate(over="ignore"):
                            exec(
                                compile(module, "<published-scalar-phase>", "exec"),
                                envs[t],
                            )
                results.append(out)
            assert results[0] == results[1]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("cute.arch.sync_threads()", "pass", 1),
        lambda s: s.replace("range(thread, 1, 32)", "range(thread, 2, 32)"),
        lambda s: s.replace(
            "shared[0] = cutlass.Int32(source)",
            "shared[thread] = cutlass.Int32(source)",
        ),
        lambda s: s + "\nshared[0] = cutlass.Int32(9)\n",
        lambda s: s + "\nalias = shared\n",
        lambda s: s.replace("range(thread, 65, 32)", "range(thread, 0, 32)"),
        lambda s: s.replace("range(thread, 65, 32)", "range(thread, 17, 32)"),
        lambda s: s.replace("a = shared[0]", "a = shared[i]"),
    ],
)
def test_unproved_ownership_publication_epochs_and_loop_domains_decline(change):
    source = change(_SOURCE)
    result, changed = _transform(source)
    if "shared[i]" in source:
        # A nonzero coordinate invalidates the shared-slot proof entirely.
        assert not changed
    else:
        assert not changed
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(source))


def test_masked_first_use_and_zero_trip_do_not_speculate_division():
    source = """
for owner in range(thread, 1, 32):
    shared[0] = cutlass.Int32(source)
cute.arch.sync_threads()
for i in range(thread, 65, 32):
    if i < valid:
        a = shared[0]
        b = cutlass.Int32(7 // a)
        out[i] = b
"""
    result, count = _transform(source)
    assert count == 0
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(source))


def test_current_ssa_loop_carry_and_guarded_redefinition_do_not_escape():
    source = _SOURCE.replace(
        "c = cutlass.Int32(b + 3)", "b = i\n    c = cutlass.Int32(b + 3)"
    )
    result, changed = _transform(source)
    assert changed
    assert "c = cutlass.Int32(b + 3)" in result
    source = _SOURCE.replace(
        "c = cutlass.Int32(b + 3)",
        "if i > 0:\n        b = i\n    c = cutlass.Int32(b + 3)",
    )
    result, changed = _transform(source)
    assert changed and "c = cutlass.Int32(b + 3)" in result


def test_literal_types_signed_zero_and_unknown_calls_keep_exact_keys():
    source = _SOURCE.replace(
        "c = cutlass.Int32(b + 3)",
        """c = cutlass.Float32(b + 0)
    floating = cutlass.Float32(b + 0.0)
    negative = cutlass.Float32(b + -0.0)
    boolean = cutlass.Float32(b + False)
    opaque = unknown(b)""",
    )
    result, changed = _transform(source)
    assert changed
    assert "b + 0.0" not in result  # Names are rebound to immutable uniform registers.
    assert " + 0.0" in result and " + -0.0" in result and " + False" in result
    assert "opaque = unknown(b)" in result


def test_warp_writer_requires_leader_and_published_epoch():
    direct = _SOURCE.replace("range(thread, 1, 32)", "range(thread // 32, 1, 1)")
    assert _transform(direct)[1] == 0
    guarded = direct.replace(
        "    shared[0] = cutlass.Int32(source)",
        "    if thread % 32 == 0:\n        shared[0] = cutlass.Int32(source)",
    )
    assert _transform(guarded)[1] > 0
    assert _transform(guarded.replace("thread % 32 == 0", "thread % 32 == 1"))[1] == 0


@pytest.mark.parametrize("strategy_name", ("FROM_RANDOM", "FROM_BEST_AVAILABLE"))
def test_published_scalar_coverage_preserves_entire_previous_population(strategy_name):
    import random
    from unittest.mock import patch

    from test.test_compiler_coverage import make_search

    from helion.autotuner.pattern_search import InitialPopulationStrategy

    x = torch.ones((3, 65), dtype=torch.int32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        with patch(
            "helion._compiler.autotuner_heuristics.register_fragment_published_scalars_coverage"
        ):
            previous = _cpu_bind(
                helion.kernel(_published_recipe.fn, backend="cute", static_shapes=True),
                (x,),
            )
        current = _cpu_bind(
            helion.kernel(_published_recipe.fn, backend="cute", static_shapes=True),
            (x,),
        )
    assert previous.config_spec.default_config() == current.config_spec.default_config()
    strategy = InitialPopulationStrategy[strategy_name]
    old = make_search(previous.config_spec, count=20, strategy=strategy)
    new = make_search(current.config_spec, count=20, strategy=strategy)
    for seed in (73, 741, 2031):
        random.seed(seed)
        prior = old._generate_initial_population_flat()
        state = random.getstate()
        random.seed(seed)
        actual = new._generate_initial_population_flat()
        assert random.getstate() == state
        expected = [old.config_gen.unflatten(row) for row in prior]
        configs = [new.config_gen.unflatten(row) for row in actual]
        assert configs[: len(expected)] == expected
        assert len(configs) == len(expected) + 1
        assert configs[-1][KEY] is True


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64, torch.float32))
def test_actual_sdk_published_scalar_reuse(dtype, tmp_path):
    import importlib.util

    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    x = torch.ones((3, 65), dtype=dtype)
    _, source = _code(x, True)
    assert "fragment_published_scalar" in source
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    fn.name = "staged"
    fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
    ] + [fn]
    path = tmp_path / "staged.py"
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    spec = importlib.util.spec_from_file_location("published_scalars_staged", path)
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with ir.InsertionPoint(emitted.body):
            entry = func.FuncOp("entry", ([], []))
            with ir.InsertionPoint(entry.add_entry_block()):
                kind = {
                    torch.int32: cutlass.Int32,
                    torch.int64: cutlass.Int64,
                    torch.float32: cutlass.Float32,
                }[dtype]
                tensors = [
                    cute.make_tensor(
                        cute.make_ptr(
                            kind, 0, cute.AddressSpace.gmem, assumed_align=16
                        ),
                        cute.make_layout((512,)),
                    )
                    for _ in fn.args.args
                ]
                sdk.staged(*tensors)
                func.ReturnOp([])
        assert emitted.operation.verify()
        assert "nvvm.barrier" in str(emitted)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _published_recipe(x: torch.Tensor):
    out = torch.empty_like(x)
    width = hl.specialize(x.size(1))
    for row in hl.grid(x.size(0)):
        columns = hl.arange(width)
        original = hl.load(x, [row, columns])
        shifted = original + 1
        values = hl.cumsum(shifted, dim=0)
        low = values.min()
        high = values.max()
        scale = (high - low).to(torch.int32)
        hl.store(out, [row, columns], values + original + shifted + scale)
    return out


def _code(x, flag):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_published_recipe, (x,))
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {"cute_fragment_threads": 32, KEY: flag}
        )
        return bound, bound.to_code(config)


@pytest.mark.parametrize("width", (17, 65, 129))
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_actual_generated_program_owned_values_and_default(width, dtype):
    x = (torch.arange(3 * width) % 31 - 11).reshape(3, width).to(dtype)
    bound, before = _code(x, False)
    _, after = _code(x, True)
    assert KEY not in bound.config_spec.default_config()
    assert (
        "fragment_published_scalar" in after
        if width >= 32
        else "fragment_published_scalar" not in after
    )
    assert before.count("sync_threads()") == after.count("sync_threads()")
    for order in (list(range(32)), list(reversed(range(32)))):
        results = []
        for code in (before, after):
            out = torch.full_like(x, -999)
            _, barriers = _simulate_register_load_program(
                code, x, 32, lane_order=order, host_tensors={"out": out}
            )
            results.append((out, barriers))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        assert results[0][1] == results[1][1]
        values = (x + 1).cumsum(1, dtype=x.dtype)
        expected = (
            values
            + x
            + (x + 1)
            + (
                values.max(1, keepdim=True).values - values.min(1, keepdim=True).values
            ).to(torch.int32)
        )
        torch.testing.assert_close(results[1][0], expected, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64, torch.float32))
def test_published_scalars_native(dtype):
    for width in (17, 65, 129):
        x = (
            (torch.arange(3 * width, device=DEVICE) % 31 - 11)
            .reshape(3, width)
            .to(dtype)
        )
        original = x.clone()
        values = (x + 1).cumsum(1, dtype=dtype)
        expected = (
            values
            + x
            + (x + 1)
            + (
                values.max(1, keepdim=True).values - values.min(1, keepdim=True).values
            ).to(torch.int32)
        )
        bound = _published_recipe.bind((x,))
        for flag in (False, True):
            config = helion.Config.from_dict(
                dict(bound.config_spec.default_config())
                | {"cute_fragment_threads": 32, KEY: flag}
            )
            result = bound.compile_config(config)(x)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(x, original, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _snapshot_published_recipe(x: torch.Tensor):
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    width = hl.specialize(x.size(1))
    for row in hl.grid(x.size(0)):
        columns = hl.arange(helion.next_power_of_2(width))
        valid = columns < width
        original = hl.load(x, [row, columns], extra_mask=valid)
        shifted = original.to(torch.int32) + 1
        prefix = hl.cumsum(columns, dim=0)
        snapshot_maximum = torch.where(valid, shifted, -2147483648).max()
        maximum = hl.load(x, [row, 0]).to(torch.int32)
        scale = maximum + 3
        combined = hl.cumsum(prefix + scale, dim=0)
        hl.store(first, [row, columns], original + combined, extra_mask=valid)
        hl.store(
            second,
            [row, columns],
            original - snapshot_maximum - scale,
            extra_mask=valid,
        )
    return first, second


@pytest.mark.parametrize("threads,width", [(32, 65), (128, 129), (512, 513)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_snapshot_register_loops_reuse_published_scalars(threads, width, dtype):
    x = (torch.arange(2 * width) % 31 - 11).reshape(2, width).to(dtype)
    original = x.clone()
    codes = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_snapshot_published_recipe, (x,))
        for enabled in (False, True):
            requested = dict(bound.config_spec.default_config()) | {
                "cute_fragment_threads": threads,
                "cute_fragment_register_snapshots": True,
                KEY: enabled,
            }
            config = helion.Config.from_dict(requested)
            bound.config_spec.normalize(config, _fix_invalid=False)
            code = bound.to_code(config)
            assert "fragment_snapshot_reduce" in code
            codes.append(code)
    # Every CTA thread executes slot zero inside the physical-domain guard.
    # Later partial slots retain their original masks.
    assert codes[0] != codes[1]
    assert "fragment_published_scalar" in codes[1]
    snapshot_maximum = (x + 1).max(1, keepdim=True).values
    scale = x[:, :1].to(torch.int32) + 3
    for order in (list(range(threads)), list(reversed(range(threads)))):
        outputs = []
        for code in codes:
            first, second = torch.full_like(x, -999), torch.full_like(x, -999)
            _, barriers = _simulate_register_load_program(
                code,
                x,
                threads,
                lane_order=order,
                host_tensors={"first": first, "second": second},
            )
            torch.testing.assert_close(
                first,
                x + (scale + torch.arange(width).cumsum(0)).cumsum(1).to(dtype),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                second, x - snapshot_maximum - scale, rtol=0, atol=0
            )
            outputs.append((first, second, barriers))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
        assert outputs[0][2] == outputs[1][2]
    assert torch.equal(original, x)


_FINAL_EPOCH_SOURCE = """
for scratch in range(thread, 32, 32):
    shared[scratch] = cutlass.Int64(100 + scratch)
cute.arch.sync_threads()
for read in range(thread, 32, 32):
    earlier[read] = shared[read]
cute.arch.sync_threads()
for owner in range(thread // 32, 1, 1):
    if thread % 32 == 0:
        shared[0] = cutlass.Int64(source)
cute.arch.sync_threads()
for slot in cutlass.range_constexpr(3):
    index = thread + slot * 32
    if index < 65:
        a = shared[0]
        b = cutlass.Int64(a + 3)
        out[index] = cutlass.Int64(b + index)
cute.arch.sync_threads()
"""


def _execute_published_phases(source, value, dtype, order):
    statements = ast.parse(source).body
    phases = [[]]
    for statement in statements:
        if (
            isinstance(statement, ast.Expr)
            and ast.unparse(statement) == "cute.arch.sync_threads()"
        ):
            phases.append([])
        else:
            phases[-1].append(statement)
    out, earlier, shared = [None] * 65, [None] * 32, [None] * 32
    envs = [
        {
            "thread": t,
            "source": value,
            "shared": shared,
            "out": out,
            "earlier": earlier,
            "cutlass": SimpleNamespace(
                Int64=np.int64, Float32=np.float32, range_constexpr=range
            ),
        }
        for t in range(32)
    ]
    for phase in phases:
        code = compile(
            ast.fix_missing_locations(ast.Module(body=phase, type_ignores=[])),
            "<final-epoch>",
            "exec",
        )
        for t in order:
            with np.errstate(over="ignore", invalid="ignore"):
                exec(code, envs[t])
    return np.asarray(out, dtype=dtype).tobytes(), np.asarray(
        earlier, dtype=dtype
    ).tobytes()


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.int64, [-(2**63), -7, 0, 2**63 - 2]),
        (np.float32, [-0.0, 0.0, -7.5, np.inf, -np.inf, np.nan]),
    ],
)
def test_final_scalar_epoch_and_full_first_slot_preserve_earlier_observers(
    dtype, values
):
    source = (
        _FINAL_EPOCH_SOURCE.replace("Int64", "Float32")
        if dtype is np.float32
        else _FINAL_EPOCH_SOURCE
    )
    result, count = _transform(source)
    assert count > 0
    before, after = ast.parse(source).body, ast.parse(result).body
    # Scratch stores, old readers, final writer and its publication are exact.
    assert [ast.dump(n) for n in before[:6]] == [ast.dump(n) for n in after[:6]]
    assert result.count("sync_threads()") == source.count("sync_threads()")
    for value in values:
        for order in (list(range(32)), list(reversed(range(32)))):
            original = _execute_published_phases(source, dtype(value), dtype, order)
            actual = _execute_published_phases(result, dtype(value), dtype, order)
            assert original == actual
            assert np.frombuffer(actual[1], dtype=dtype).tolist() == list(
                range(100, 132)
            )


@pytest.mark.parametrize(
    "old,new",
    [
        ("for scratch", "alias = shared\nfor scratch"),
        ("for scratch", "alias = shared.iterator\nfor scratch"),
        (
            "        shared[0] = cutlass.Int64(source)",
            "        shared[0] = cutlass.Int64(source)\n        shared[1] = cutlass.Int64(9)",
        ),
        ("thread % 32 == 0", "thread % 32 == 1"),
        (
            "shared[0] = cutlass.Int64(source)\ncute.arch.sync_threads()",
            "shared[0] = cutlass.Int64(source)\npass",
        ),
        ("a = shared[0]", "a = shared[1]"),
        (
            "out[index] = cutlass.Int64(b + index)",
            "shared[0] = b\n        out[index] = b",
        ),
    ],
)
def test_final_epoch_alias_writer_and_later_mutation_reject(old, new):
    source = _FINAL_EPOCH_SOURCE.replace(old, new)
    result, count = _transform(source)
    assert count == 0
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))


@pytest.mark.parametrize(
    "old,new",
    [
        ("range_constexpr(3)", "range_constexpr(0)"),
        ("range_constexpr(3)", "range_constexpr(1, 3)"),
        ("index < 65", "index < 17"),
        ("index < 65", "index < width"),
        ("index < 65", "index < 65 and enabled"),
        ("thread + slot * 32", "thread + slot * 64"),
        ("thread + slot * 32", "thread + (slot + 1) * 32"),
        ("        a = shared[0]", "        index = 0\n        a = shared[0]"),
        ("        a = shared[0]", "        slot = 0\n        a = shared[0]"),
        ("        a = shared[0]", "        continue\n        a = shared[0]"),
        (
            "        a = shared[0]",
            "        if index < valid:\n            a = shared[0]",
        ),
    ],
)
def test_first_slot_partial_zero_trip_rebinding_and_inner_masks_reject(old, new):
    source = _FINAL_EPOCH_SOURCE.replace(old, new)
    result, count = _transform(source)
    assert count == 0
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("threads,width", [(32, 65), (128, 129), (512, 513)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_snapshot_final_epoch_scalars_native(threads, width, dtype):
    x = (torch.arange(2 * width, device=DEVICE) % 31 - 11).reshape(2, width).to(dtype)
    before = x.clone()
    maximum = (x + 1).max(1, keepdim=True).values
    scale = x[:, :1].to(torch.int32) + 3
    expected = (
        x + (scale + torch.arange(width, device=DEVICE).cumsum(0)).cumsum(1).to(dtype),
        x - maximum - scale,
    )
    bound = _snapshot_published_recipe.bind((x,))
    for flag in (False, True):
        config = helion.Config.from_dict(
            dict(bound.config_spec.default_config())
            | {
                "cute_fragment_threads": threads,
                "cute_fragment_register_snapshots": True,
                KEY: flag,
            }
        )
        actual = bound.compile_config(config)(x)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
