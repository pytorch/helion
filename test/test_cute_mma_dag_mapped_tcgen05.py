"""Strict CPU proofs/source tests for flat mapped-store tcgen05 regions."""

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_mma_dag import _kernel_body
from test.test_cute_mma_dag import _program
from test.test_cute_mma_dag_split_store import _strides
from test.test_cute_mma_dag_split_store import _three_shared_a_split_stores

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute import cute_mma_dag_codegen as warp
from helion._compiler.cute import cute_mma_dag_tcgen05 as tc
from helion._compiler.cute import cute_mma_dag_tcgen05_regions as regions
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _MmaDagAliasGuard

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _args(dtype=torch.float32, offset=0, *, k=128, period=128, col_period=128):
    m, n = 2 * period, 2 * col_period
    shape = (2, 2, period, col_period)
    base = torch.empty(m * n + offset, dtype=dtype)
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        *(torch.empty((n, k), dtype=torch.bfloat16).T for _ in range(3)),
        torch.empty((m, n), dtype=torch.bfloat16),
        torch.empty((m, n), dtype=torch.bfloat16),
        base.as_strided(shape, _strides(shape), offset),
    )


def _bind(args):
    return cpu_bind(
        helion.kernel(_three_shared_a_split_stores, backend="cute", static_shapes=True),
        args,
    )


def _config(bm=128, bn=128, bk=32, *, store="staged", stages=2, warps=4, **kwargs):
    return helion.Config(
        block_sizes=[bm, bn, bk],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=warps,
        cute_mma_dag_stages=stages,
        cute_mma_dag_store=store,
        **kwargs,
    )


@pytest.mark.parametrize("bm,bn", ((64, 64), (64, 128), (128, 64), (128, 128)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("offset", (0, 7, 8))
@pytest.mark.parametrize("store", ("direct", "staged"))
def test_mapped_region_source_keeps_native_slots_casts_pointer_and_non_tma_output(
    bm, bn, dtype, offset, store
):
    with forbid_cuda_work():
        args = _args(dtype, offset)
        bound = _bind(args)
        program = _program(bound)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts.tcgen05.regions and facts.tcgen05.mapped_store_regions
        assert len(program.carries) == len(program.dots) == 3
        assert len({dot.slot for dot in program.dots}) == 3
        assert len({dot.a.chain_key for dot in program.dots}) == 1
        mapped = program.stores[2].access
        assert mapped.store_map is not None and mapped.dtype == dtype
        with bound._runtime_arg_values_for_codegen():
            assert bound.env.runtime_arg_values_by_name["out2"] is args[-1]
            assert args[-1].storage_offset() == offset
        captured = []
        original = regions.plan_tcgen05_regions

        def capture(*a, **kw):
            plan = original(*a, **kw)
            captured.append(plan)
            return plan

        with patch.object(regions, "plan_tcgen05_regions", capture):
            source = bound.to_code(_config(bm, bn, store=store))
        plan = captured[-1]
        assert plan.regions and not plan.register_carries
        assert len({slot.column for slot in plan.slots}) == 3
        assert mapped.node not in plan.store_tma
        assert not any(spec.access.store_map is not None for spec in plan.tma)
        aligned = (offset * dtype.itemsize) % 16 == 0
        assert any(s.access.name == "out2" for s in plan.staged_stores) == (
            store == "staged" and aligned
        )
        body = _kernel_body(source)
        assert "tcgen05_mma(" in body and ".mma_bf16(" not in body
        assert "virtual_pid" not in body and "cluster_wait" not in body
        base = next(
            line for line in body.splitlines() if "out2.iterator.toint()" in line
        )
        assert all(
            term in base
            for term in mapped.store_map.base_terms(
                f"md_a{mapped.dims[0].index}",
                f"md_a{mapped.dims[1].index}",
                dtype.itemsize,
            )
        )
        if store == "direct":
            assert "cp_async_bulk_tensor_global_shared_cta(" not in body
        if dtype == torch.float32:
            assert program.stores[2].value.args[0].meta["val"].dtype == torch.bfloat16
            assert "BFloat16" in body
        alias_plan = next(
            p for p in wrapper_plans(source) if p["kind"] == "helion_mma_dag_alias"
        )
        assert alias_plan["stored"][alias_plan["tensor_names"].index("out2")]
        _MmaDagAliasGuard(alias_plan, args).check(args)
        with pytest.raises(BackendUnsupported, match="output TMA descriptor proof"):
            tc._tma_dims(mapped)


def test_mapped_region_budget_native_geometry_and_launch_refusals():
    bound = _bind(_args())
    program = _program(bound)
    facts = bound.config_spec.cute_mma_dag_facts
    sizes = {a.block_id: 128 for a in facts.axes}
    assert facts.smem_capacity == 232448
    assert dag.mma_dag_config_reason(facts, sizes, 4, 1, "tcgen05") is None
    assert "capacity" in dag.mma_dag_config_reason(facts, sizes, 4, 2, "tcgen05")
    assert regions.region_tmem_columns(facts, [128] * len(facts.axes)) == 384
    for options in (
        {"persistent": True},
        {"epilogue": "split"},
        {"epilogue": "streamed"},
        {"lookahead": "on"},
        {"pass_warps": "mma"},
    ):
        assert "flat single-CTA" in dag.mma_dag_config_reason(
            facts, sizes, 4, 1, "tcgen05", **options
        )
    for config in (
        _config(cute_mma_dag_persistent=1),
        _config(cute_mma_dag_cluster=2),
        _config(bm=256),
        _config(bn=256),
    ):
        with pytest.raises((BackendUnsupported, InvalidConfig)):
            bound.to_code(config)
    with bound.env, bound.host_function:
        assert regions.region_facts(program) is None
        assert regions.region_facts(
            program, mapped_store_regions=True
        ).mapped_store_regions


@pytest.mark.parametrize("warps", (4, 8))
def test_mapped_mma_warp_is_rejected_or_pruned_before_region_planning(warps):
    with forbid_cuda_work():
        args = _args()
        bound = _bind(args)
        facts = bound.config_spec.cute_mma_dag_facts
        sizes = {a.block_id: 32 if a.kind == dag.AXIS_LOOP else 64 for a in facts.axes}
        assert dag.mma_dag_config_reason(facts, sizes, warps, 2, "tcgen05") is None
        reason = dag.mma_dag_config_reason(
            facts, sizes, warps, 2, "tcgen05", pass_warps="mma"
        )
        # The existing pass_warps != "off" guard includes "mma"; the
        # planner derives its extra-warp flag only from that same value.
        assert reason == "mapped tcgen05 stores require flat single-CTA row epilogues"
        configs = [
            *(
                seed
                for seed in bound.config_spec.compiler_seed_configs
                if seed.get("cute_mma_dag") == "tcgen05"
            ),
            bound.config_spec.default_config(),
            _config(bm=64, bn=64, warps=warps, cute_mma_dag_pass_warps="mma"),
        ]
        captured = []
        original = regions.plan_tcgen05_regions

        def guarded_plan(*a, **kw):
            assert not kw.get("mma_warp")
            plan = original(*a, **kw)
            assert not plan.mma_warp
            captured.append(plan)
            return plan

        with patch.object(regions, "plan_tcgen05_regions", guarded_plan):
            for config in configs:
                normalized = bound._normalized_config_copy(config)
                assert normalized.get("cute_mma_dag_pass_warps") == "off"
            # Normalization may prune the inert knob rather than rejecting
            # the whole config.  Its source must still use only the planned
            # epilogue/store warps, never an extra MMA-issuing warp.
            source = bound.to_code(configs[-1])
        assert captured and "tcgen05_mma(" in source
        assert f"block=({warps * 32}, 1, 1)" in source


@pytest.mark.parametrize("k,bk", ((1000, 64), (72, 32), (96, 64)))
def test_mapped_tcgen05_never_uses_the_merged_partial_k_permission(k, bk):
    bound = _bind(_args(k=k))
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.axes[facts.k_axes[0]].partial_k
    with pytest.raises((BackendUnsupported, InvalidConfig)):
        bound.to_code(_config(bk=bk))
    if k % dag.MMA_DAG_K_STEP:
        assert facts.tcgen05 is None
        assert bound.config_spec.cute_mma_dag.choices == ("off", "warp")
    else:
        assert facts.tcgen05.mapped_store_regions
        assert "must divide" in dag.mma_dag_config_reason(
            facts,
            {a.block_id: bk if a.kind == dag.AXIS_LOOP else 128 for a in facts.axes},
            4,
            2,
            "tcgen05",
        )


def test_mapped_row_period_below_native_height_stays_warp_only():
    bound = _bind(_args(period=32, col_period=64))
    assert bound.config_spec.cute_mma_dag_facts.tcgen05 is None
    assert bound.config_spec.cute_mma_dag.choices == ("off", "warp")


def test_mapped_native_runtime_alias_guard_keeps_original_view_spans():
    args = _args(torch.bfloat16)
    bound = _bind(args)
    source = bound.to_code(_config(store="direct"))
    plan = next(p for p in wrapper_plans(source) if p["kind"] == "helion_mma_dag_alias")
    guard = _MmaDagAliasGuard(plan, args)
    aliased = list(args)
    aliased[-1] = args[-3].view(args[-1].shape)
    with pytest.raises(BackendUnsupported, match="alias"):
        guard.check(tuple(aliased))


def test_mapped_native_fact_rejects_unsupported_regions():
    bound = _bind(_args())
    program = _program(bound)
    first, *rest = program.dots
    load = first.a.load
    subscripts = list(load.subscripts)
    subscripts[0] = dataclasses.replace(subscripts[0], window=SimpleNamespace())
    windowed = dataclasses.replace(
        program,
        dots=(
            dataclasses.replace(
                first,
                a=dataclasses.replace(
                    first.a,
                    load=dataclasses.replace(load, subscripts=tuple(subscripts)),
                ),
            ),
            *rest,
        ),
    )
    extra = dag.Axis(len(program.axes), -1, dag.AXIS_LEAD, 2)
    variants = (
        windowed,
        dataclasses.replace(program, axes=(*program.axes, extra)),
        dataclasses.replace(
            program, derived=(SimpleNamespace(loads=(), aux_loads=()),)
        ),
        dataclasses.replace(program, loops=(*program.loops, program.loops[0])),
        dataclasses.replace(
            program,
            stores=(
                dataclasses.replace(program.stores[0], swap=True),
                *program.stores[1:],
            ),
        ),
        dataclasses.replace(
            program,
            dots=(
                dataclasses.replace(
                    first, a=dataclasses.replace(first.a, pointwise=True)
                ),
                *rest,
            ),
        ),
        dataclasses.replace(
            program, dots=(dataclasses.replace(first, slot=program.dots[1].slot), *rest)
        ),
    )
    with bound.env, bound.host_function:
        for unsupported in variants:
            assert regions.mapped_store_region_reason(unsupported) is not None
            assert regions.region_facts(unsupported, mapped_store_regions=True) is None


def test_mapped_native_noncontiguous_columns_keep_scalar_stores():
    args = list(_args())
    args[-1] = args[-1].as_strided((2, 2, 128, 128), (32768, 16384, 1, 128))
    bound = _bind(tuple(args))
    program = _program(bound)
    assert program.stores[2].access.store_map.inner_strides == (1, 128)
    source = bound.to_code(_config(store="staged"))
    body = _kernel_body(source)
    assert "tcgen05_mma(" in body and ".stg_f32(" in body
    assert "output TMA descriptor" not in body


def test_mapped_native_eight_warp_source_keeps_one_epilogue_partition():
    bound = _bind(_args())
    source = bound.to_code(_config(warps=8, store="staged"))
    assert "tcgen05_mma(" in source
    assert "md_epi = md_warp < 4" in source
    assert "block=(256, 1, 1)" in source


class _StoreOracle:
    """Execute the real native row/packet store emitters on CPU integers."""

    slot = tc._Tcgen05Emitter.slot
    row_var = tc._Tcgen05Emitter.row_var
    half_rows = tc._Tcgen05Emitter.half_rows
    warp_cols = tc._Tcgen05Emitter.warp_cols
    half_cols = tc._Tcgen05Emitter.half_cols
    local_cols = tc._Tcgen05Emitter.local_cols
    has_cb = tc._Tcgen05Emitter.has_cb
    col_expr = tc._Tcgen05Emitter.col_expr
    row_guard = tc._Tcgen05Emitter.row_guard
    element_coords = tc._Tcgen05Emitter.element_coords
    emit_direct_store = tc._Tcgen05Emitter.emit_direct_store
    emit_stage_out = tc._Tcgen05Emitter.emit_stage_out
    staging_address = tc._Tcgen05Emitter.staging_address
    packets = tc._Tcgen05Emitter.packets
    pass_guard = tc._Tcgen05Emitter.pass_guard
    pair_store = warp._Emitter.pair_store
    address = warp._Emitter.address
    axis_base = warp._Emitter.axis_base
    define_bases = warp._Emitter.define_bases

    def __init__(self, store, bm, bn, threads=128):
        self.tc = SimpleNamespace(slots=(tc.SlotGeometry(bm, bn, 0),))
        self.plan = SimpleNamespace(threads=threads, mma_warp=False)
        self.program = SimpleNamespace(
            frag_slots={store.frag: 0}, scalar_loads=(), accesses=(store.access,)
        )
        self.evaluator = SimpleNamespace(
            enter_frag=lambda *a: None,
            leave_frag=lambda: None,
            push=lambda: None,
            pop=lambda: None,
            evaluate=lambda *a: "0.0",
            reg=None,
        )
        self.df = SimpleNamespace(
            tensor_arg=lambda *a, **kw: SimpleNamespace(name="out"),
            placeholder_args=set(),
            new_var=self.new_var,
        )
        self.split = False
        self.pass_warps_branch = False
        self.fold_addresses = False
        self.row_guards = 0
        self.base_vars = {}
        self.iter_vars = {}
        self.lines = []
        self.indent = ""
        self.names = itertools.count()

    def new_var(self, prefix):
        return f"{prefix}_{next(self.names)}"

    def emit(self, line):
        self.lines.append(self.indent + line)

    def bind(self, prefix, expression):
        name = self.new_var(prefix)
        self.emit(f"{name} = {expression}")
        return name

    @contextmanager
    def block(self, header):
        if header:
            self.emit(header)
            self.indent += "    "
        yield
        if header:
            self.indent = self.indent[:-4]

    nest = block

    def _is_lead_dim(self, *args):
        return False

    def pairs_aligned(self, access, *args):
        return access.tile_dim_strides[1] == 1

    def hoist_scalars(self, *args):
        pass

    def iteration_base(self, access, *args):
        return self.base_vars[access.node]

    def staging_base(self, *args):
        return "0"


@pytest.mark.parametrize(
    "bm,bn", ((64, 8), (64, 64), (64, 128), (128, 8), (128, 64), (128, 128))
)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("store_mode", ("direct", "staged"))
@pytest.mark.parametrize("threads", (128, 256))
def test_real_native_row_and_packet_emission_matches_split_map_bijection(
    bm, bn, dtype, store_mode, threads
):
    bound = _bind(_args(dtype))
    program = _program(bound)
    store = program.stores[2]
    oracle = _StoreOracle(store, bm, bn, threads)
    oracle.define_bases()
    oracle.iter_vars[store.access.node] = oracle.base_vars[store.access.node]
    if store_mode == "direct":
        with oracle.block("if md_tidx < 128:"), oracle.row_guard(0):
            oracle.emit_direct_store(store)
    else:
        oracle.emit_stage_out(store, "0", {}, "md_tidx", threads)
    executable = compile("\n".join(oracle.lines), "<native-store-oracle>", "exec")
    proof = store.access.store_map
    es = dtype.itemsize
    for row_base, col_base in ((0, 0), (128, 128)):
        seen = []

        def record(address, size, seen=seen):
            seen.extend(range(address, address + size))

        runtime = SimpleNamespace(
            stg_f32=lambda addr, *v: record(addr, 4),
            stg_bf16=lambda addr, *v: record(addr, 2),
            stg_b32=lambda addr, *v: record(addr, 4),
            stg_v2_f32=lambda addr, *v: record(addr, 8),
            stg_v4_b32=lambda addr, *v: record(addr, 16),
            lds_v4_b32=lambda *a: (0, 0, 0, 0),
            pack_bf16x2=lambda *a: 0,
        )
        for tid in range(threads):
            lane, w = tid % 32, tid // 32
            scope = {
                "cutlass": SimpleNamespace(Int64=int),
                "out": SimpleNamespace(iterator=SimpleNamespace(toint=lambda: 32)),
                "_helion_mma_dag": runtime,
                "md_tidx": tid,
                "md_row64": w * 16 + lane % 16,
                "md_row128": w * 32 + lane,
                "md_rowok": lane < 16,
                "md_cb0": (lane // 16) * (bn // 2),
                f"md_a{store.access.dims[0].index}": row_base,
                f"md_a{store.access.dims[1].index}": col_base,
            }
            exec(executable, scope)
        expected = {
            32 + proof.offset(row_base + row, col_base + col) * es + byte
            for row in range(bm)
            for col in range(bn)
            for byte in range(es)
        }
        assert len(seen) == len(expected)
        assert set(seen) == expected
