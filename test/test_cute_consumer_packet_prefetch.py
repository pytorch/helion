"""Executed packet-motion proofs; no native compilation or GPU is needed."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_bounded_cache_codegen import _cpu_target
from test.test_cute_full_reduction_loop_encoding import rms_norm

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads
import helion._compiler.cute.prefetch_reduction_packets as prefetch_module
from helion._compiler.cute.prefetch_reduction_packets import prefetch_reduction_packets
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import CuteReductionReloadSpec
import helion.language as hl


def source(*, lanes: int = 1, width: int = 4, carrier: str = "Uint16") -> str:
    return f"""
cache = cute.make_rmem_tensor({lanes * width}, cutlass.{carrier})
cute.arch.griddepcontrol_wait()
total = cutlass.Float32(0)
for roffset_1 in range(cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(64)):
    for lane in range({lanes}):
        before = roffset_1 + cute.arch.thread_idx()[0] * {width} + lane * 32
        px = cute.arch.load(x.iterator + before, ir.VectorType.get([{width}], cutlass.{carrier}.mlir_type))
        for v in cutlass.range_constexpr({width}):
            cache[lane * {width} + v] = px[v]
            total = total + cutlass.Float32(px[v])
total = cute.arch.warp_reduction_sum(total)
cute.arch.sync_threads()
for roffset_1 in range(cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(64)):
    for lane in range({lanes}):
        after = roffset_1 + cute.arch.thread_idx()[0] * {width} + lane * 32
        valid = after < extent
        pw = _cute_load_l2_evict_last(weight.iterator + after if valid else weight.iterator, ir.VectorType.get([{width}], cutlass.{carrier}.mlir_type))
        values = []
        for v in cutlass.range_constexpr({width}):
            result = cutlass.Float32(cache[lane * {width} + v]) * total * cutlass.Float32(pw[v]) if valid else cutlass.Float32(0)
            values.append(cutlass.{carrier}(result))
        if valid:
            _cute_store_u16_vec(out.iterator + after, values)
"""


def run_pass(
    text: str, *, disjoint: bool = True, mode: str = "register_prefetch"
) -> list[ast.stmt]:
    return prefetch_reduction_packets(
        ast.parse(text).body,
        constexpr_values={},
        thread_block_dims=(8, 1, 1),
        tensor_dtypes={
            "x": "cutlass.BFloat16",
            "weight": "cutlass.BFloat16",
            "out": "cutlass.BFloat16",
        },
        reload_modes={1: mode},
        argument_names={"x", "weight", "out", "extent"},
        proven_disjoint_tensor_pairs={frozenset(("weight", "out"))}
        if disjoint
        else set(),
    )


def dumped(body: list[ast.stmt]) -> str:
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def admitted(body: list[ast.stmt]) -> bool:
    return "_consume_prefetch_" in ast.unparse(ast.Module(body=body, type_ignores=[]))


class Scalar:
    mlir_type = "raw"

    def __call__(self, value: object) -> object:
        return value


class Pointer:
    def __init__(self, name: str, data: list[float], offset: int = 0) -> None:
        self.name, self.data, self.offset = name, data, offset

    def __add__(self, offset: int) -> Pointer:
        return Pointer(self.name, self.data, self.offset + offset)


def execute(
    body: list[ast.stmt], *, thread: int, extent: int
) -> tuple[list[float], list[tuple]]:
    events: list[tuple] = []
    output = [-999.0] * 128
    x = [float(i % 13 - 7) for i in range(128)]
    weight = [float(i % 5 + 1) for i in range(128)]

    def load(
        pointer: Pointer, shape: list[int], policy: str = "default"
    ) -> list[float]:
        events.append(("load", pointer.name, pointer.offset, shape[0], policy))
        assert 0 <= pointer.offset <= len(pointer.data) - shape[0]
        return pointer.data[pointer.offset : pointer.offset + shape[0]]

    def store(pointer: Pointer, values: list[float]) -> None:
        events.append(("store", pointer.name, pointer.offset))
        pointer.data[pointer.offset : pointer.offset + len(values)] = values

    scalar = Scalar()
    arch = SimpleNamespace(
        thread_idx=lambda: (thread, 0, 0),
        load=load,
        griddepcontrol_wait=lambda: events.append(("wait",)),
        sync_threads=lambda: events.append(("barrier",)),
        warp_reduction_sum=lambda value: value,
    )
    env = {
        "cute": SimpleNamespace(
            arch=arch, make_rmem_tensor=lambda n, dtype: [None] * n
        ),
        "cutlass": SimpleNamespace(
            Int32=scalar,
            Float32=scalar,
            Uint16=scalar,
            Uint32=scalar,
            range_constexpr=range,
        ),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: shape)
        ),
        "x": SimpleNamespace(iterator=Pointer("x", x)),
        "weight": SimpleNamespace(iterator=Pointer("weight", weight)),
        "out": SimpleNamespace(iterator=Pointer("out", output)),
        "extent": extent,
        "_cute_load_l2_evict_last": lambda p, s: load(p, s, "l2_last"),
        "_cute_store_u16_vec": store,
    }
    exec(
        compile(
            ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[]))),
            "<packet motion>",
            "exec",
        ),
        env,
    )
    return output, events


@pytest.mark.parametrize(
    "lanes,width,carrier",
    [(1, 4, "Uint16"), (2, 4, "Uint16"), (2, 8, "Uint16"), (2, 4, "Uint32")],
)
@pytest.mark.parametrize("extent", [16, 48, 96])
def test_executed_raw_cache_preserves_masks_bits_and_order(
    lanes: int, width: int, carrier: str, extent: int
) -> None:
    text = source(lanes=lanes, width=width, carrier=carrier)
    before, after = ast.parse(text).body, run_pass(text)
    assert admitted(after)
    for thread in range(8):
        old, old_events = execute(before, thread=thread, extent=extent)
        new, new_events = execute(after, thread=thread, extent=extent)
        assert old == new
        # Exactly the same source addresses, widths and cache policies, including
        # redirected masked packets. Only their position relative to the barrier changes.
        assert sorted(e for e in old_events if e[0] == "load") == sorted(
            e for e in new_events if e[0] == "load"
        )
        assert old_events[0] == new_events[0] == ("wait",)
        barrier = new_events.index(("barrier",))
        assert all(
            i < barrier for i, e in enumerate(new_events) if e[:2] == ("load", "weight")
        )
        assert all(i > barrier for i, e in enumerate(new_events) if e[0] == "store")


@pytest.mark.parametrize(
    "old,new",
    [
        ("weight.iterator + after", "out.iterator + after"),
        ("total = cute.arch.warp_reduction_sum(total)", "opaque(total)"),
        (
            "total = cute.arch.warp_reduction_sum(total)",
            "cute.arch.griddepcontrol_wait()",
        ),
        (
            "total = cute.arch.warp_reduction_sum(total)",
            "(weight.iterator + 1).store(total)",
        ),
        (
            "total = cute.arch.warp_reduction_sum(total)",
            "alias = weight\nalias[1] = total",
        ),
        (
            "total = cute.arch.warp_reduction_sum(total)",
            "atomic_add(weight.iterator, total)",
        ),
        ("valid = after < extent", "valid = after < total"),
        ("valid = after < extent", "valid = after < extent\n        after = after + 1"),
        (
            "valid = after < extent",
            "copy = after\n        after = after + 1\n        valid = copy < extent",
        ),
        ("valid = after < extent", "valid = table[after] < extent"),
        ("pw[v]", "pw[v] + inspect(pw)"),
        (
            "values.append(cutlass.Uint16(result))",
            "values.append(cutlass.Uint16(result))\n            lane = lane + 1",
        ),
        (
            "cutlass.Int32(64), cutlass.Int32(64)",
            "cutlass.Int32(128), cutlass.Int32(64)",
        ),
        (
            "cutlass.Int32(64), cutlass.Int32(64)",
            "cutlass.Int32(extent), cutlass.Int32(64)",
        ),
        ("range(1)", "range(1, 2)"),
        ("range(1)", "range(33)"),
        ("range(1)", "range(0)"),
        (
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)",
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)\nextra = cute.make_rmem_tensor(125, cutlass.Uint16)",
        ),
        (
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)",
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)\nextra = cute.make_rmem_tensor(63, cutlass.Float64)",
        ),
        (
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)",
            "cache = cute.make_rmem_tensor(4, cutlass.Uint16)\nif extent:\n    extra = cute.make_rmem_tensor(1, cutlass.Uint16)",
        ),
        (
            "cute.arch.griddepcontrol_wait()",
            "cute.arch.griddepcontrol_wait()\n_cute_load_l2_evict_last = arbitrary",
        ),
        (
            "cute.arch.griddepcontrol_wait()",
            "cute.arch.griddepcontrol_wait()\nweight = out",
        ),
    ],
)
def test_unproved_motion_declines_without_ast_changes(old: str, new: str) -> None:
    text = source().replace(old, new)
    assert text != source()
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_distinct_arguments_are_not_alias_proof() -> None:
    text = source()
    assert not admitted(run_pass(text, disjoint=False))
    assert admitted(run_pass(text, disjoint=True))


@pytest.mark.parametrize("mode", ["auto", "register", "gmem"])
def test_old_modes_unchanged(mode: str) -> None:
    text = source()
    assert dumped(run_pass(text, mode=mode)) == dumped(ast.parse(text).body)


@pytest.mark.parametrize("override", ["disabled", "smem", "register"])
def test_diagnostic_fuser_override_disables_optional_motion(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("HELION_FUSER_MODE", override)
    assert not admitted(run_pass(source()))


def test_reload_enum_preserves_old_prefix_and_default() -> None:
    item = CuteReductionReloadSpec(block_id=1)
    fragment = item._fragment(ConfigSpec(backend=CuteBackend()))
    assert tuple(fragment.choices) == ("auto", "register", "gmem", "register_prefetch")
    assert item._fill_missing() == "auto"
    assert fragment.search_values() == list(fragment.choices)
    assert "register_prefetch" in fragment.pattern_neighbors("register")
    for choice in fragment.choices:
        assert item._normalize("reload", choice) == choice


def inverse_motion(
    control: list[ast.stmt], transformed: list[ast.stmt]
) -> list[ast.stmt]:
    """Undo only raw cache plumbing, then compare every other AST operation."""
    original = ast.parse(ast.unparse(ast.Module(body=control, type_ignores=[]))).body
    result = ast.parse(ast.unparse(ast.Module(body=transformed, type_ignores=[]))).body
    original_last = [s for s in original if isinstance(s, ast.For)][-1]
    original_consumer = original_last.body[0].body
    original_load = next(
        s
        for s in original_consumer
        if isinstance(s, ast.Assign)
        and isinstance(s.value, ast.Call)
        and (
            ast.unparse(s.value.func).startswith("_cute_load_")
            or ast.unparse(s.value.func) == "cute.arch.load"
        )
    )
    name = original_load.targets[0].id
    cache = result[0].targets[0].id

    class Undo(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith(cache):
                return None
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == cache
            ):
                return None
            return self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id == cache:
                assert isinstance(node.slice, ast.BinOp) and isinstance(
                    node.slice.op, ast.Add
                )
                return ast.Subscript(
                    value=ast.Name(id=name, ctx=ast.Load()),
                    slice=node.slice.right,
                    ctx=ast.Load(),
                )
            return self.generic_visit(node)

    restored = Undo().visit(ast.Module(body=result, type_ignores=[])).body
    last = [s for s in restored if isinstance(s, ast.For)][-1]
    last.body[0].body.insert(original_consumer.index(original_load), original_load)
    return restored


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_inverse_ast_keeps_all_arithmetic_collectives_and_masks(lanes: int) -> None:
    original = ast.parse(source(lanes=lanes)).body
    transformed = run_pass(source(lanes=lanes))
    assert admitted(transformed)
    assert dumped(inverse_motion(original, transformed)) == dumped(original)


def test_signed_loop_wrap_not_a_single_trip_proof() -> None:
    text = source().replace(
        "cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(64)",
        "cutlass.Int32(2147483640), cutlass.Int32(2147483647), cutlass.Int32(64)",
    )
    assert not admitted(run_pass(text))


@pytest.mark.parametrize("count,admit", [(120, True), (121, False)])
def test_combined_cache_policy_counts_every_32bit_slot(count: int, admit: bool) -> None:
    text = f"extra = cute.make_rmem_tensor({count}, cutlass.Uint16)\n" + source()
    assert admitted(run_pass(text)) == admit


@pytest.mark.parametrize(
    "width,dtype,lanes",
    [(4096, "bfloat16", 1), (7168, "bfloat16", 1), (7168, "float32", 2)],
)
def test_actual_strict_configs_keep_control_math_and_reach_bounded_lane_domains(
    width: int, dtype: str, lanes: int
) -> None:
    initialized = torch.cuda.is_initialized()
    with (
        _cpu_target(),
        _forbid_native_compile() as calls,
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
    ):
        kernel = helion.kernel(rms_norm, backend="cute", static_shapes=True)
        bound = kernel._bind_isolated(
            (
                torch.empty(
                    (1, width),
                    dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype],
                ),
                torch.empty(
                    width,
                    dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype],
                ),
                1e-6,
            )
        )
        spec = bound.config_spec
        owner = spec.reduction_loops[0].block_id
        vector = 8 if dtype != "float32" and width == 7168 else 4
        config = helion.Config(
            block_sizes=[1],
            num_threads=[
                1024 if block == owner else 1
                for block in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[1 << (width - 1).bit_length()],
            cute_vector_widths=[
                vector if block == owner else 1
                for block in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_reduction_reloads=["register"],
            cute_pdl=True,
        )
        generation = ConfigGeneration(spec)
        with bound.env:
            flat, effective = generation.strict_config_pair(config)
            assert generation.strict_unflatten(flat) == effective
        control = bound.to_code(effective)
        enabled = helion.Config(
            **dict(effective.config, cute_reduction_reloads=["register_prefetch"])
        )
        with bound.env:
            flat, enabled = generation.strict_config_pair(enabled)
            assert generation.strict_unflatten(flat) == enabled
            assert enabled.config["reduction_loops"] == [1 << (width - 1).bit_length()]
        code = bound.to_code(enabled)
        trees = [ast.parse(text) for text in (control, code)]
        bodies = [
            next(
                n.body
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
            )
            for tree in trees
        ]
        assert admitted(bodies[1])
        assert dumped(inverse_motion(*bodies)) == dumped(bodies[0])
        assert f"cute.make_rmem_tensor({lanes * vector}, cutlass.Uint" in code
        assert "cute.arch.griddepcontrol_wait()" in code
        assert all(call.call_count == 0 for call in calls)
    assert torch.cuda.is_initialized() == initialized


def fused_residual_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        value = x[row, :].to(torch.float32) + residual[row, :].to(torch.float32)
        stored = value.to(x.dtype)
        inverse = torch.rsqrt(torch.mean(value * value, dim=-1) + 1e-6)
        output[row, :] = (value * inverse[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
        residual_out[row, :] = stored
    return output, residual_out


@pytest.mark.parametrize(
    "width,rows,threads", [(4096, 2, 128), (4096, 4, 128), (8192, 2, 256)]
)
def test_multirow_launch_dimensions_allow_prefetch_without_changing_math(
    width: int, rows: int, threads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    initialized = torch.cuda.is_initialized()
    with (
        _cpu_target(),
        _forbid_native_compile() as native,
        patch.object(
            BaseSearch, "autotune", side_effect=AssertionError("autotune forbidden")
        ) as tune,
    ):
        kernel = helion.kernel(
            fused_residual_norm,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        bound = kernel._bind_isolated(
            (
                torch.empty((1024, width), dtype=torch.bfloat16),
                torch.empty((1024, width), dtype=torch.bfloat16),
                torch.empty(width, dtype=torch.bfloat16),
            )
        )
        spec = bound.config_spec
        owner = spec.reduction_loops[0].block_id
        config = helion.Config(
            block_sizes=[rows],
            reduction_loops=[width],
            num_threads=[
                threads if block == owner else rows
                for block in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                8 if block == owner else 1
                for block in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_lane_layouts=[
                "strided" if block == owner else "blocked"
                for block in spec.cute_lane_layouts.valid_block_ids()
            ],
            cute_reduction_reloads=["register"],
            cute_independent_reduction=True,
            cute_replicated_reduction=False,
            cute_pdl=True,
            load_eviction_policies=["l2_last"]
            * len(spec.load_eviction_policies.default()),
        )
        generation = ConfigGeneration(spec)
        with bound.env:
            flat, effective = generation.strict_config_pair(config)
            assert generation.strict_unflatten(flat) == effective
            for key, value in config.config.items():
                assert effective.config[key] == value
            requested = helion.Config(
                **dict(effective.config, cute_reduction_reloads=["register_prefetch"])
            )
            flat, enabled = generation.strict_config_pair(requested)
            assert generation.strict_unflatten(flat) == enabled
            assert enabled == requested
        control = bound.to_code(effective)
        code = bound.to_code(enabled)
        # Reproduce the old grammar on this actual generated launch, rather
        # than changing packet indices or masks to introduce the intrinsic.
        with patch.object(
            prefetch_module,
            "_PURE",
            prefetch_module._PURE - {"cute.arch.block_dim"},
        ):
            legacy = bound.to_code(enabled)
        bodies = [
            next(
                node.body
                for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef)
                and any(
                    ast.unparse(decorator) == "cute.kernel"
                    for decorator in node.decorator_list
                )
            )
            for text in (control, code, legacy)
        ]
        assert "cute.arch.block_dim()" in control
        assert f"block=({threads}, {rows}, 1)" in code
        assert not admitted(bodies[2])
        assert dumped(bodies[2]) == dumped(bodies[0])
        assert admitted(bodies[1])
        assert dumped(inverse_motion(*bodies[:2])) == dumped(bodies[0])

        producer, consumer = [
            statement for statement in bodies[1] if isinstance(statement, ast.For)
        ]
        weight_loads = [
            node
            for node in ast.walk(producer)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "_cute_load_l2_evict_last"
            and any(
                isinstance(part, ast.Attribute)
                and isinstance(part.value, ast.Name)
                and part.value.id == "weight"
                and part.attr == "iterator"
                for part in ast.walk(node.value)
            )
        ]
        assert len(weight_loads) == 1
        assert not any(
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_cute_load_l2_evict_last"
            for node in ast.walk(consumer)
        )
        stores = [
            node
            for node in ast.walk(consumer)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_cute_store_u16_vec"
        ]
        assert len(stores) == 2
        assert {ast.unparse(node.args[0]).split(".", 1)[0] for node in stores} == {
            "output",
            "residual_out",
        }
        wait = next(
            index
            for index, statement in enumerate(bodies[1])
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value.func) == "cute.arch.griddepcontrol_wait"
        )
        assert wait < bodies[1].index(producer) < bodies[1].index(consumer)
        assert all(call.call_count == 0 for call in native)
        assert tune.call_count == 0
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("stop", [64, 128])
def test_register_prefetch_retains_existing_register_fuser_semantics(
    stop: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HELION_FUSER_MODE", raising=False)
    text = f"""
for roffset_1 in range(0, {stop}, 64):
    first = (x.iterator + roffset_1).load()
    square = first * first
for roffset_1 in range(0, {stop}, 64):
    second = (x.iterator + roffset_1).load()
    result = second + second
"""
    outputs = []
    for mode in ("register", "register_prefetch"):
        result = fuse_two_pass_loads(
            ast.parse(text).body,
            tensor_dtypes={"x": "cutlass.BFloat16", "out": "cutlass.BFloat16"},
            reload_modes={1: mode},
            proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
        )
        outputs.append(result)
    assert dumped(outputs[0]) == dumped(outputs[1])
    assert "_fuse_cache_" in ast.unparse(ast.Module(body=outputs[0], type_ignores=[]))
    if stop == 128:
        assert not admitted(run_pass(text))


def test_vector_read_before_defining_load_is_not_rewritten() -> None:
    text = source().replace("        pw = ", "        stale = pw[0]\n        pw = ")
    assert not admitted(run_pass(text))


def test_mismatched_consumer_domain_declines() -> None:
    text = source()
    position = text.rindex("for roffset_1 in range")
    text = text[:position] + text[position:].replace(
        "cutlass.Int32(0)", "cutlass.Int32(1)", 1
    )
    assert not admitted(run_pass(text))


def block_dim_source() -> str:
    return source().replace(
        "before = roffset_1 +",
        "before = cutlass.Int32(cute.arch.block_dim()[0]) * 0 + roffset_1 +",
    )


def test_block_dim_prefix_peer_store_keeps_read_only_policy() -> None:
    text = block_dim_source().replace(
        "total = cutlass.Float32(0)",
        "if cute.arch.thread_idx()[0] == 1:\n    _cute_store_u16_vec(weight.iterator, [cutlass.Uint16(17)] * 4)\ntotal = cutlass.Float32(0)",
    )
    assert not admitted(run_pass(text))
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_block_dim_read_only_control_still_prefetches() -> None:
    assert admitted(run_pass(block_dim_source()))
