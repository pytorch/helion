from __future__ import annotations

import ast
from contextlib import contextmanager
import dataclasses
import itertools
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute import cute_mma_dag_codegen as codegen
from helion._compiler.cute.cute_warp_mma_gemm import _swizzled_chunk
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import BackendUnsupported
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_codegen_target(), forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shared_a(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor, recipe: hl.constexpr):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    out1 = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        acc1 = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k):
            left = a[row, inner].to(torch.float32).to(a.dtype)
            if recipe:
                left = (left + 0.25).to(a.dtype)
            acc = torch.addmm(acc, left, b[inner, col])
            acc1 = torch.addmm(acc1, left, b1[inner, col])
        out[row, col] = acc.to(a.dtype).to(torch.float32)
        out1[row, col] = acc1.to(a.dtype)
    return out, out1


def _bind(k=1000, *, dtype=torch.bfloat16, a_trans=False, b_trans=False, recipe=False):
    a = torch.empty((k, 32) if a_trans else (32, k), dtype=dtype, device="cpu")
    b = torch.empty((64, k) if b_trans else (k, 64), dtype=dtype, device="cpu")
    if a_trans:
        a = a.T
    if b_trans:
        b = b.T
    return cpu_bind(
        _shared_a, (a, b, b.clone(memory_format=torch.preserve_format), recipe)
    )


def _program(bound):
    with bound.env, bound.host_function:
        program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
    assert program is not None
    return program


def _config(bk=64, stages=2, *, mode="warp"):
    return helion.Config(
        block_sizes=[16, 32, bk],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4 if mode == "tcgen05" else 2,
        cute_mma_dag_stages=stages,
        cute_mma_dag_loads="tma",
    )


@pytest.mark.parametrize("k", [8, 24, 40, 72, 1000, 1032])
@pytest.mark.parametrize("bk", [16, 32, 64, 128])
def test_packet_aligned_k_has_warp_seeds_and_ceil_trips(k: int, bk: int) -> None:
    bound = _bind(k)
    program = _program(bound)
    facts = bound.config_spec.cute_mma_dag_facts
    axis = program.dots[0].k
    assert facts.axes[axis.index].partial_k
    assert not facts.axes[axis.index].ragged
    assert dag.mma_dag_seed_ladder(facts.axes[axis.index])
    assert dag.mma_dag_seed_configs(facts)
    source = bound.to_code(_config(bk))
    ast.parse(source)
    assert f"cutlass.range(0, {math.ceil(k / bk)})" in source
    assert "_helion_mma_dag.mma_bf16(" in source
    assert "cp_async_bulk_tensor" not in source
    assert "tcgen05_mma(" not in source
    assert "_zfill(" in source
    assert f"< {k}" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "a_trans,b_trans", list(itertools.product([False, True], repeat=2))
)
@pytest.mark.parametrize("stages", [1, 2, 4])
def test_orientations_stages_shared_a_and_output_rounding(
    dtype, a_trans, b_trans, stages
):
    bound = _bind(40, dtype=dtype, a_trans=a_trans, b_trans=b_trans)
    program = _program(bound)
    first, second = program.dots
    assert first.a.chain_key == second.a.chain_key
    assert first.slot != second.slot
    assert program.stores[0].value.args[0].meta["val"].dtype == dtype
    source = bound.to_code(_config(64, stages))
    ast.parse(source)
    assert "cutlass.range(0, 1)" in source
    assert "cp_async_bulk_tensor" not in source
    assert source.count("cp_async_commit_group()") >= 1
    assert "cp_async_wait_group(" in source
    assert "_zfill(" in source


def test_collective_tail_seed_round_trip_keeps_geometry_and_warp_mode():
    bound = _bind()
    seeds = [
        c
        for c in bound.config_spec.compiler_seed_configs
        if c.get("cute_mma_dag") == "warp"
    ]
    assert seeds
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        for seed in seeds[:3]:
            flat, config = generation.canonicalize_flat(generation.flatten(seed))
            flat2, config2 = generation.canonicalize_flat(flat)
            assert flat == flat2 and config == config2
            assert config.get("cute_mma_dag") == "warp"
            assert config.block_sizes == seed.block_sizes
    assert not [
        c
        for c in bound.config_spec.compiler_seed_configs
        if c.get("cute_mma_dag") == "tcgen05"
    ]


@pytest.mark.parametrize("k", [7, 15, 17, 31, 63, 1001])
def test_non_packet_aligned_tail_keeps_divisibility_refusal(k):
    bound = _bind(k)
    facts = bound.config_spec.cute_mma_dag_facts
    if facts is None:
        assert not any(
            c.get("cute_mma_dag") in ("warp", "tcgen05")
            for c in bound.config_spec.compiler_seed_configs
        )
        return
    assert not any(axis.partial_k for axis in facts.axes)
    sizes = {
        axis.block_id: 16 if axis.kind == dag.AXIS_ROOT else 64 for axis in facts.axes
    }
    reason = dag.mma_dag_config_reason(facts, sizes, 2, 2)
    assert reason is not None and "must divide the extent" in reason


def test_tcgen05_partial_k_remains_rejected():
    bound = _bind()
    with pytest.raises(
        (BackendUnsupported, helion.exc.InvalidConfig), match="must divide the extent"
    ):
        bound.to_code(_config(mode="tcgen05"))


class _PacketOracle:
    """Execute the real packet emitter and ownership walk on CPU integers."""

    own_packets = codegen._Emitter.own_packets
    emit_issue_tile = codegen._Emitter.emit_issue_tile

    def __init__(self, load, axis, bk, rows_pos, threads=128):
        self.load, self.axis, self.bk, self.rows_pos = load, axis, bk, rows_pos
        self.plan = SimpleNamespace(
            partial_k_axes=frozenset({axis.index}),
            threads=threads,
            load_hints={},
            size=lambda axis: bk,
        )
        self.df = SimpleNamespace(
            tensor_arg=lambda *args, **kwargs: SimpleNamespace(name="source")
        )
        self.evaluator = SimpleNamespace(push=lambda: None, pop=lambda: None)
        self.lines = []
        self.indent = ""
        self.names = itertools.count()

    def emit(self, line):
        self.lines.append(self.indent + line)

    def bind(self, prefix, expression):
        name = f"{prefix}_{next(self.names)}"
        self.emit(f"{name} = {expression}")
        return name

    @contextmanager
    def nest(self, line):
        self.emit(line)
        self.indent += "    "
        yield
        self.indent = self.indent[:-4]

    def packet_terms(self, cpr):
        row, col = f"(md_tidx // {cpr})", f"(md_tidx % {cpr})"
        return row, col, _swizzled_chunk(row, col, cpr)

    def row_exists(self, *args):
        return None

    def axis_offset(self, axis, value):
        return value

    def iteration_base(self, load, values):
        k_stride = load.tile_dim_strides[load.tile_dims.index(self.axis)] * 2
        return (
            f"(BASE + cutlass.Int64({values[self.axis.loop]}) * {self.bk * k_stride})"
        )


def _oracle_packets(k, bk, *, k_rows, iteration, threads=128):
    axis = SimpleNamespace(index=2, loop=0, extent=k)
    other = SimpleNamespace(index=0)
    # A physical tile's rows are its non-contiguous dimension. A padded
    # pitch prevents the allocation's adjacent row from disguising an OOB
    # logical K packet as valid memory.
    logical_columns = 32 if k_rows else k
    pitch = logical_columns + 16
    load = SimpleNamespace(
        node=object(),
        fake=None,
        name="source",
        dtype=torch.bfloat16,
        tile_dims=(axis, other) if k_rows else (other, axis),
        tile_dim_strides=(pitch, 1),
    )
    tile = codegen._Tile("shared", 0, bk if k_rows else 16, 32 if k_rows else bk, 2, 1)
    oracle = _PacketOracle(load, axis, bk, 0, threads)
    oracle.emit_issue_tile(tile, load, 0, {0: "future"}, "0")
    source = "\n".join(oracle.lines)
    ast.parse(source)
    base = 1 << 20
    packets = []

    def copy(dst, src, size):
        assert size in (0, 16)
        packets.append((dst, src, size))

    for thread in range(threads):
        namespace = {
            "cutlass": SimpleNamespace(Int32=int, Int64=int),
            "_helion_mma_dag": SimpleNamespace(
                cp_async_16_zfill=copy, cp_async_16=copy
            ),
            "source": SimpleNamespace(iterator=SimpleNamespace(toint=lambda: base)),
            "BASE": base,
            "shared": 0,
            "future": iteration,
            "md_tidx": thread,
        }
        exec(source, namespace)
    assert len(packets) == tile.packets
    assert len({dst for dst, _src, _size in packets}) == tile.packets
    padded = [float("nan")] * (tile.rows * tile.cols)
    seen = set()
    for dst, src, size in packets:
        row = dst // tile.row_bytes
        swizzled_chunk = (dst % tile.row_bytes) // 16
        row_mask = row % 8 if tile.cpr >= 8 else (row // (8 // tile.cpr)) % tile.cpr
        chunk = swizzled_chunk ^ row_mask
        for element in range(8):
            col = chunk * 8 + element
            kk = iteration * bk + (row if k_rows else col)
            mn = col if k_rows else row
            valid = kk < k
            assert (size == 16) == valid
            if valid:
                expected = (
                    base + ((kk * pitch + mn) if k_rows else (mn * pitch + kk)) * 2
                )
                assert src + element * 2 == expected
                seen.add((mn, kk))
            else:
                assert src == base
            padded[row * tile.cols + col] = 1.0 if size else 0.0
    assert not any(math.isnan(value) for value in padded)
    return seen, padded


@pytest.mark.parametrize("k", [8, 24, 40, 72, 1000, 1032])
@pytest.mark.parametrize("bk", [16, 32, 64, 128])
@pytest.mark.parametrize("k_rows", [False, True])
def test_real_packet_emitter_covers_every_valid_k_and_zeroes_all_padding(k, bk, k_rows):
    seen = set()
    for iteration in range(-(-k // bk)):
        valid, _padded = _oracle_packets(k, bk, k_rows=k_rows, iteration=iteration)
        assert not (seen & valid)
        seen.update(valid)
    rows = 32 if k_rows else 16
    assert seen == {(row, kk) for row in range(rows) for kk in range(k)}


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("k,bk", [(8, 64), (40, 64), (72, 32), (1000, 64)])
def test_tail_ring_future_refills_overwrite_poisoned_slots(stages, k, bk):
    # Model the emitted prologue/refill order, including empty commit groups
    # after the end. Both physical orientations use the actual packet source.
    trips = -(-k // bk)
    for k_rows in (False, True):
        slots = {}
        pending = []
        for ahead in range(stages - 1):
            if ahead < trips:
                slots[ahead % stages] = _oracle_packets(
                    k, bk, k_rows=k_rows, iteration=ahead
                )[1]
            pending.append(ahead if ahead < trips else None)
        for iteration in range(trips):
            stage = iteration % stages
            if stages == 1:
                slots[stage] = _oracle_packets(
                    k, bk, k_rows=k_rows, iteration=iteration
                )[1]
            expected = _oracle_packets(k, bk, k_rows=k_rows, iteration=iteration)[1]
            assert slots[stage] == expected
            future = iteration + stages - 1
            if stages > 1:
                if future < trips:
                    slots[future % stages] = _oracle_packets(
                        k, bk, k_rows=k_rows, iteration=future
                    )[1]
                pending.append(future if future < trips else None)
        assert len(pending) == stages - 1 + (trips if stages > 1 else 0)


def test_inplace_recipe_masks_both_operands_after_arithmetic():
    bound = _bind(40, recipe=True)
    source = bound.to_code(_config(64, 2))
    ast.parse(source)
    assert "< 40" in source
    assert "cutlass.Float32(0.0)" in source
    assert "pack_bf16x2" in source
    assert "cp_async_bulk_tensor" not in source
    # The zero-filled source's padded values must not turn into the recipe's
    # nonzero 0.25 contribution. Exercise the actual masking expression.
    program = _program(bound)
    dot = program.dots[0]
    emitter = SimpleNamespace(
        plan=SimpleNamespace(partial_k_axes=frozenset({dot.k.index})),
        axis_base=lambda axis: "base",
    )
    with bound.env:
        for operand in (dot.a, dot.b):
            coords = tuple("local" if axis is dot.k else "0" for axis in operand.dims)
            expression = codegen._Emitter.masked(emitter, dot, operand, coords, "value")
            assert "< 40" in expression and "0.0" in expression
            for poison in (0.25, float("inf"), float("nan")):
                namespace = {
                    "base": 32,
                    "local": 8,
                    "value": poison,
                    "cutlass": SimpleNamespace(Float32=float),
                }
                assert eval(expression, namespace) == 0.0
            namespace["local"] = 7
            namespace["value"] = 0.25
            assert eval(expression, namespace) == 0.25


@pytest.mark.parametrize(
    "mutation",
    [
        "nested",
        "multi_axis",
        "varying_bound",
        "scan",
        "effect",
        "fragment",
        "repeated",
        "window",
        "offset",
        "side_load",
        "source_dtype",
        "strided_columns",
    ],
)
def test_structural_partial_k_proof_keeps_unsupported_protocols_out(mutation):
    program = _program(_bind(40))
    loop = program.loops[0]
    first = program.dots[0]
    if mutation == "nested":
        program = dataclasses.replace(
            program, loops=(dataclasses.replace(loop, parent=0),)
        )
    elif mutation == "multi_axis":
        program = dataclasses.replace(
            program, loops=(dataclasses.replace(loop, axes=(first.k, first.rows)),)
        )
    elif mutation == "varying_bound":
        program = dataclasses.replace(
            program, loops=(dataclasses.replace(loop, tile_end=object()),)
        )
    elif mutation == "scan":
        program = dataclasses.replace(program, scans=(object(),))
    elif mutation == "effect":
        program = dataclasses.replace(
            program, loops=(dataclasses.replace(loop, steps=(*loop.steps, object())),)
        )
    else:
        operand = first.a
        load = operand.load
        if mutation == "fragment":
            operand = dataclasses.replace(
                operand, kind="frag", load=None, frag=first.acc
            )
        elif mutation == "repeated":
            load = dataclasses.replace(load, dims=(first.k, first.k))
        elif mutation == "window":
            subscripts = list(load.subscripts)
            subscripts[0] = dataclasses.replace(subscripts[0], window=object())
            load = dataclasses.replace(load, subscripts=tuple(subscripts))
        elif mutation == "offset":
            subscripts = list(load.subscripts)
            subscripts[0] = dataclasses.replace(subscripts[0], offset=object())
            load = dataclasses.replace(load, subscripts=tuple(subscripts))
        elif mutation == "side_load":
            operand = dataclasses.replace(operand, aux_loads=(load,))
        elif mutation == "source_dtype":
            load = dataclasses.replace(
                load, fake=torch.empty((32, 40), dtype=torch.float32, device="cpu")
            )
        elif mutation == "strided_columns":
            load = dataclasses.replace(load, dim_strides=(80, 2))
        if operand.load is not None:
            operand = dataclasses.replace(operand, load=load)
        changed = dataclasses.replace(first, a=operand)
        program = dataclasses.replace(program, dots=(changed, *program.dots[1:]))
    assert not dag.partial_static_k_axes(program)


def test_unaligned_runtime_base_is_rejected_before_plan_seeding():
    a = torch.empty((32 * 40 + 1,), dtype=torch.bfloat16, device="cpu")[1:].view(32, 40)
    b = torch.empty((40, 64), dtype=torch.bfloat16, device="cpu")
    bound = cpu_bind(_shared_a, (a, b, b.clone(), False))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts.axes[facts.k_axes[0]].partial_k
    assert not facts.axes[facts.k_axes[0]].partial_k_packets
    assert not dag.mma_dag_seed_configs(facts)
    assert not any(
        c.get("cute_mma_dag") in ("warp", "tcgen05")
        for c in bound.config_spec.compiler_seed_configs
    )
    assert bound.config_spec.default_config().get("cute_mma_dag") == "off"
    with pytest.raises(helion.exc.InvalidConfig, match="cache-key-proved"):
        bound.to_code(_config())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_k(a: torch.Tensor, b: torch.Tensor, b1: torch.Tensor):
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    out1 = torch.empty_like(out)
    for row, col in hl.tile((m, n)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        acc1 = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(k, block_size=64):
            left = a[row, inner]
            acc = torch.addmm(acc, left, b[inner, col])
            acc1 = torch.addmm(acc1, left, b1[inner, col])
        out[row, col] = acc.to(a.dtype)
        out1[row, col] = acc1.to(a.dtype)
    return out, out1


@pytest.mark.parametrize("k", (40, 72, 1000))
def test_fixed_k_cannot_use_floor_fold_to_admit_a_partial_tail(k):
    bound = cpu_bind(
        _fixed_k,
        (
            torch.empty((32, k), dtype=torch.bfloat16),
            torch.empty((k, 64), dtype=torch.bfloat16),
            torch.empty((k, 64), dtype=torch.bfloat16),
        ),
    )
    facts = bound.config_spec.cute_mma_dag_facts
    if facts is None:
        # Fixed ragged extents may already fail axis admission.  They must
        # not acquire the newly introduced partial-K surface or seeds.
        assert not any(
            c.get("cute_mma_dag") in ("warp", "tcgen05")
            for c in bound.config_spec.compiler_seed_configs
        )
        return
    program = _program(bound)
    axis = program.dots[0].k
    assert axis.size == 64
    assert not dag.partial_static_k_axes(program)
    assert not facts.axes[axis.index].partial_k
    sizes = {a.block_id: 32 for a in facts.axes if a.tunable}
    assert "must divide" in dag.mma_dag_config_reason(facts, sizes, 2, 2)
    with pytest.raises(helion.exc.InvalidConfig, match="must divide"):
        bound.to_code(helion.Config(block_sizes=[16, 32], cute_mma_dag="warp"))


def test_fixed_whole_k_and_automatic_tail_keep_their_existing_geometry():
    k = 128
    bound = cpu_bind(
        _fixed_k,
        (
            torch.empty((32, k), dtype=torch.bfloat16),
            torch.empty((k, 64), dtype=torch.bfloat16),
            torch.empty((k, 64), dtype=torch.bfloat16),
        ),
    )
    facts = bound.config_spec.cute_mma_dag_facts
    assert not facts.axes[facts.k_axes[0]].partial_k
    assert dag.mma_dag_seed_configs(facts)
    automatic = _bind(40)
    axis = automatic.config_spec.cute_mma_dag_facts.axes[
        _program(automatic).dots[0].k.index
    ]
    assert axis.fixed is None and axis.partial_k and axis.partial_k_packets


def test_fixed_loop_min_fold_remains_whole_trip_only():
    import sympy

    index = sympy.Symbol("fixed_index", integer=True, nonnegative=True)
    matcher = SimpleNamespace(
        loop_symbols={0: index},
        axis_list=[dag.Axis(0, 2, dag.AXIS_LOOP, 1000, loop=0, size=64)],
    )
    expression = sympy.Min(64 * index + 64, 1000)
    # This fold is deliberately unchanged.  It ranges over floor trips,
    # and would be wrong at the partial trip if fixed-K admission were widened.
    folded = dag._fold_tile_loop_bounds(expression, matcher, [0])
    assert folded == 64 * index + 64
    assert expression.subs(index, 15) == 1000 and folded.subs(index, 15) == 1024


def test_unknown_fake_alignment_has_no_partial_packet_seeds_or_default():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_fallback_kernels=False):
        a = torch.empty((32, 40), dtype=torch.bfloat16, device="cpu")
        b = torch.empty((40, 64), dtype=torch.bfloat16, device="cpu")
        bound = cpu_bind(_shared_a, (a, b, b.clone(), False))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts.axes[facts.k_axes[0]].partial_k
    assert not facts.axes[facts.k_axes[0]].partial_k_packets
    assert not dag.mma_dag_seed_configs(facts)
    assert bound.config_spec.default_config().get("cute_mma_dag") == "off"


@pytest.mark.parametrize("recipe", (False, True))
def test_partial_packet_seeds_have_the_same_routes_as_the_planner(recipe):
    bound = _bind(40, recipe=recipe)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts.axes[facts.k_axes[0]].partial_k_packets
    captured = []
    original = dag.plan_mma_dag

    def capture(*args, **kwargs):
        plan = original(*args, **kwargs)
        if plan is not None:
            captured.append(plan)
        return plan

    from unittest.mock import patch

    with patch.object(dag, "plan_mma_dag", capture):
        seeds = [
            c
            for c in bound.config_spec.compiler_seed_configs
            if c.get("cute_mma_dag") == "warp"
        ]
        assert seeds
        for seed in seeds[:3]:
            bound.to_code(seed)
            plan = captured[-1]
            assert plan.partial_k_axes
            assert all(
                plan.route(dot, operand) in (dag.ROUTE_DIRECT, dag.ROUTE_INPLACE)
                for dot in plan.program.dots
                for operand in (dot.a, dot.b)
            )


def test_partial_packet_specialization_keeps_the_dynamic_alignment_guard():
    from torch._dynamo.source import LocalSource

    from helion._compiler.cute import memory_ops

    a = torch.empty((32, 40), dtype=torch.bfloat16)
    b = torch.empty((40, 64), dtype=torch.bfloat16)
    args = (a, b, b.clone(), False)
    bound = cpu_bind(_shared_a, args)
    specialization = bound.env.runtime_input_specializations[
        memory_ops._PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    ]
    names = tuple(bound.kernel.signature.parameters)
    extractors = []
    for source in specialization.sources:
        assert isinstance(source, LocalSource)
        at = names.index(source.local_name)
        extractors.append(lambda values, at=at: values[at])
    expected = specialization.classifier(
        tuple(extractor(args) for extractor in extractors)
    )
    guard = specialization.prepared_guard_factory(extractors, expected)
    assert guard(args)
    changed = list(args)
    changed[0] = torch.empty(32 * 40 + 1, dtype=torch.bfloat16)[1:].view(32, 40)
    assert not guard(tuple(changed))
