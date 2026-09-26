from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
from itertools import pairwise
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass.pipeline import helpers as pipeline_helpers
from cutlass.utils import blackwell_helpers as sm100

from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_flat_grouped_ir import _host
from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_flat_grouped_public import _native_source
from test.test_cute_flat_grouped_public import _source
from test.test_cute_tma_rn import rn_config
from test.test_cute_tma_rn import schedule

import helion
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.cute import tcgen05_flattened_prefix as prefix_module
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_kernel_module,
)
from helion._compiler.cute.tcgen05_flat_grouped_config import K_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import RESIDENT_CTAS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import STAGES_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import WARPS_KEY
from helion._compiler.cute.tcgen05_flat_grouped_config import (
    normalize_grouped_rna_config,
)
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_storage
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._compiler.cute.tcgen05_operand_transform import prove_operand_stage_span
from helion._compiler.cute.tcgen05_tma_rn import CONVERSION_KEY
from helion._compiler.cute.tcgen05_tma_rn import TMA_RN
from helion._compiler.cute.tcgen05_tma_rn import WARP_RAW
from helion.autotuner.config_generation import ConfigGeneration


def low_config(bound, ctas=1, wrapped=True):
    return helion.Config.from_dict(
        rn_config(bound, wrapped=wrapped).config
        | {
            STAGES_KEY: 2,
            RESIDENT_CTAS_KEY: ctas,
        }
    )


def low_schedule(bound, ctas=1, **changes):
    proof = prove_flat_grouped_rna(bound.env, _host(bound).device_ir)
    assert proof is not None
    kwargs = {
        "converter_warps": 0,
        "block_k": 32,
        "ab_stages": 2,
        "shared_capacity": 232448,
        "tma_rn": True,
        "resident_ctas": ctas,
        "descriptor_policy": WRAPPED,
    }
    kwargs.update(changes)
    return flat_grouped_rna_schedule(bound.env, proof, **kwargs)


@pytest.mark.parametrize("operand", ["A", "B"])
def test_actual_SDK_two_stage_layout_and_TMA_bytes(operand):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = sm100.make_trivial_tiled_mma(
                cutlass.TFloat32,
                cutlass.TFloat32,
                cute.nvgpu.OperandMajorMode.MN,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                (128, 128),
            )
            layout_fn = (
                sm100.make_smem_layout_a if operand == "A" else sm100.make_smem_layout_b
            )
            layout = layout_fn(
                mma, (128, 128, 32), cutlass.Float32, 2, is_k_major=operand == "B"
            )
            span = prove_operand_stage_span(layout, stages=2, alignment_bytes=1024)
            assert span is not None
            assert span.words == span.stride_words == 4096
            assert span.allocation_bytes == cute.cosize(layout.outer) * 4 == 32768
            stage = cute.slice_(layout, (None, None, None, 0))
            assert cute.size_in_bytes(cutlass.Float32, stage) == 16384
            assert 2 * cute.size_in_bytes(cutlass.Float32, stage) == 32768
            shape, stride = (
                ((128, 128, 3), (1, 128, 16384))
                if operand == "A"
                else ((273, 128), (128, 1))
            )
            tensor = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout(shape, stride=stride),
            )
            atom_fn = (
                cute.nvgpu.make_tiled_tma_atom_A
                if operand == "A"
                else cute.nvgpu.make_tiled_tma_atom_B
            )
            atom, _ = atom_fn(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                    cute.nvgpu.tcgen05.CtaGroup.ONE
                ),
                tensor,
                stage,
                (128, 128, 32),
                mma,
                (1, 1, 1, 1),
                internal_type=cutlass.TFloat32,
            )
            assert "tma_format = TF32_RN" in str(atom.type)


def test_typed_resources_and_strict_pipeline_domain():
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        for ctas in (1, 2):
            selected = low_schedule(bound, ctas)
            assert selected is not None and selected.storage.bytes + 1024 == 100876
            assert selected.thread_block == (32, 8, 1)
            assert selected.roles.converter_warp_ids == ()
            exact_capacity = ctas * 100876
            assert low_schedule(bound, ctas, shared_capacity=exact_capacity) is not None
            assert low_schedule(bound, ctas, shared_capacity=exact_capacity - 1) is None
        assert flat_grouped_storage(256, 32, 3, tma_rn=True).bytes + 1024 == 133660
        for kwargs in (
            {"ab_stages": 3},
            {"block_k": 64},
            {"tma_rn": False, "converter_warps": 8},
        ):
            assert low_schedule(bound, **kwargs) is None
        for value in (0, 3, True, 2.0):
            assert low_schedule(bound, value) is None
            with pytest.raises(ValueError):
                replace(low_schedule(bound), resident_ctas=value)
        with pytest.raises(ValueError):
            replace(schedule(bound), resident_ctas=2)


@pytest.mark.parametrize(
    "key,value",
    [
        (STAGES_KEY, True),
        (STAGES_KEY, 3),
        (STAGES_KEY, "2"),
        (RESIDENT_CTAS_KEY, True),
        (RESIDENT_CTAS_KEY, 0),
        (RESIDENT_CTAS_KEY, 3),
    ],
)
def test_invalid_or_unavailable_choices_reject_and_repair(key, value):
    values = {CONVERSION_KEY: TMA_RN, K_KEY: 32, STAGES_KEY: 2, key: value}
    with pytest.raises(helion.exc.InvalidConfig):
        normalize_grouped_rna_config(
            values.copy(),
            available_k=(32, 64),
            rn_two_stage_ctas=(1, 2),
            fix_invalid=False,
        )
    normalize_grouped_rna_config(
        values, available_k=(32, 64), rn_two_stage_ctas=(1, 2), fix_invalid=True
    )
    assert values.get(STAGES_KEY) in (None, 2)
    assert values.get(RESIDENT_CTAS_KEY) in (None, 2)
    for unavailable in (
        {CONVERSION_KEY: TMA_RN, K_KEY: 64, STAGES_KEY: 2},
        {WARPS_KEY: 8, K_KEY: 32, STAGES_KEY: 2},
        {CONVERSION_KEY: TMA_RN, K_KEY: 32, RESIDENT_CTAS_KEY: 2},
    ):
        with pytest.raises(helion.exc.InvalidConfig):
            normalize_grouped_rna_config(
                unavailable,
                available_k=(32, 64),
                rn_two_stage_ctas=(1, 2),
                fix_invalid=False,
            )
    with pytest.raises(helion.exc.InvalidConfig):
        normalize_grouped_rna_config(
            {CONVERSION_KEY: TMA_RN, K_KEY: 32, STAGES_KEY: 2, RESIDENT_CTAS_KEY: 2},
            available_k=(32,),
            rn_two_stage_ctas=(1,),
            fix_invalid=False,
        )


def test_old_complete_seed_prefix_and_search_roundtrips():
    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        spec = bound.config_spec
        assert spec.cute_grouped_rn_two_stage_ctas == (1, 2)
        base = spec._base_default_config().config
        seeds = CuteGroupedRnaHeuristic.get_seed_configs(
            bound.env, _host(bound).device_ir
        )
        old = [
            helion.Config.from_dict(deepcopy(base) | {WARPS_KEY: warps, K_KEY: k})
            for k in (32, 64)
            for warps in (4, 8, 2, 1)
        ]
        old += [
            helion.Config.from_dict(deepcopy(s.config) | {DESCRIPTOR_KEY: WRAPPED})
            for s in tuple(old)
        ]
        rn = [
            helion.Config.from_dict(deepcopy(base) | {CONVERSION_KEY: TMA_RN, K_KEY: k})
            for k in (32, 64)
        ]
        rn += [
            helion.Config.from_dict(deepcopy(s.config) | {DESCRIPTOR_KEY: WRAPPED})
            for s in tuple(rn)
        ]
        assert seeds[:20] == old + rn
        assert len(seeds) == 25
        assert seeds[-1].get(CONVERSION_KEY) == WARP_RAW
        generator = ConfigGeneration(spec)
        for seed in seeds[20:24]:
            normalized = spec.normalized_config(seed)
            restored = spec.normalized_config(
                generator.unflatten(generator.flatten(normalized))
            )
            assert restored == with_flat_min_blocks_default(spec, normalized)
            assert restored[STAGES_KEY] == 2
            assert restored.get(RESIDENT_CTAS_KEY, 1) in (1, 2)
        for seed in (old[0], rn[0]):
            explicit = helion.Config.from_dict(
                seed.config | {STAGES_KEY: 0, RESIDENT_CTAS_KEY: 1}
            )
            assert spec.normalized_config(explicit) == spec.normalized_config(seed)
            assert _source(bound, explicit) == _source(bound, seed)


@pytest.mark.parametrize("ctas,wrapped", tuple(product((1, 2), (False, True))))
def test_emitted_one_allocation_publication_ring_and_workspace(ctas, wrapped):
    with cuda_binding(
        static=True, original=True, groups=256, rows=30856, k=128, n=128
    ) as bound:
        source = _source(bound, low_config(bound, ctas, wrapped))
        native = _native_source(source)
        plan = low_schedule(
            bound, ctas, descriptor_policy=WRAPPED if wrapped else "dynamic"
        )
        assert native == flat_grouped_kernel_module(plan)
        kernel = next(
            n
            for n in ast.parse(native).body
            if isinstance(n, ast.FunctionDef) and n.name == "_helion_native_grouped_rna"
        )
        calls = [
            (ast.unparse(n.func), n.lineno, n)
            for n in ast.walk(kernel)
            if isinstance(n, ast.Call)
        ]

        def site(name):
            (item,) = [item for item in calls if item[0] == name]
            return item[1]

        alloc, permit, wait, free = [
            site("tcgen05_tmem_allocator." + method)
            for method in (
                "allocate",
                "relinquish_alloc_permit",
                "wait_for_alloc",
                "free",
            )
        ]
        mma_lines = [line for name, line, _ in calls if name == "cute.gemm"]
        syncs = [line for name, line, _ in calls if name == "cute.arch.sync_threads"]
        assert alloc < wait < min(mma_lines) < free
        assert any(alloc < line < wait for line in syncs)
        assert any(max(mma_lines) < line < free for line in syncs)
        assert (
            (alloc < permit < wait) if ctas == 2 else (max(mma_lines) < permit < free)
        )
        for _, _, call in calls:
            if ast.unparse(call.func) in (
                "cutlass.utils.blackwell_helpers.make_smem_layout_a",
                "cutlass.utils.blackwell_helpers.make_smem_layout_b",
            ):
                assert ast.literal_eval(call.args[3]) == 2
            if ast.unparse(call.func) == "cutlass.pipeline.PipelineTmaUmma.create":
                assert (
                    next(
                        ast.literal_eval(k.value)
                        for k in call.keywords
                        if k.arg == "num_stages"
                    )
                    == 2
                )
        ring_loops = [
            n
            for n in ast.walk(kernel)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "tile_offset_2"
        ]
        assert len(ring_loops) == 2
        producer, consumer = ring_loops
        for loop, expected in (
            (producer, ["producer_acquire", "producer_commit"]),
            (consumer, ["consumer_wait", "consumer_release"]),
        ):
            ring_calls = [
                ast.unparse(n.func).split(".")[-1]
                for n in ast.walk(loop)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and ast.unparse(n.func.value) == "tcgen05_ab_pipeline"
            ]
            assert ring_calls.index(expected[0]) < ring_calls.index(expected[1])
        consumer_text = ast.unparse(consumer)
        assert (
            consumer_text.index("consumer_wait(")
            < consumer_text.index("cute.gemm(")
            < consumer_text.index("consumer_release(")
        )
        assert consumer_text.count("tcgen05_ab_consumer_state.advance()") == 1
        assert ast.unparse(producer).count("tcgen05_ab_producer_state.advance()") == 1
        assert "tcgen05_stride = cutlass.Int64(cute.arch.grid_dim()[2])" in native
        host = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "jagged_dense_bmm"
        )
        grid_assign = next(
            n
            for n in host.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_helion_native_grid_z"
                for t in n.targets
            )
        )
        expr = ast.unparse(grid_assign.value)
        assert expr == "_helion_native_get_num_sm(jagged.device)" + (
            " * 2" if ctas == 2 else ""
        )
        assert "empty((_helion_native_grid_z, 3, 16)" in source
        if not wrapped:
            indices = [
                n.value
                for n in ast.walk(kernel)
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id.endswith("tensormap_workspace_idx")
                    for t in n.targets
                )
            ]
            assert len(indices) == 2
            for grid in (1, 3, 148 * ctas):
                occupied = set()
                for block in range(grid):
                    for index in indices:
                        namespace = {
                            "cute": SimpleNamespace(
                                arch=SimpleNamespace(
                                    block_idx=lambda block=block: (0, 0, block)
                                )
                            ),
                            "tcgen05_grouped_tensormap_grid_dim": (1, 1, grid),
                            "tcgen05_grouped_d_tensormap_grid_dim": (1, 1, grid),
                        }
                        assert (
                            eval(
                                compile(
                                    ast.Expression(index), "workspace-index", "eval"
                                ),
                                namespace,
                            )
                            == block
                        )
                    cells = {
                        block * 48 + role * 16 + word
                        for role in range(3)
                        for word in range(16)
                    }
                    assert not (cells & occupied)
                    occupied |= cells
                assert occupied == set(range(grid * 48))


def _scalar_prefix_functions(bits):
    tree = ast.parse(Path(prefix_module.__file__).read_text())
    names = {
        "project_flat_interval",
        "reload_flat_interval",
        "build_flat_tile_prefix",
        "resolve_flat_nm_work",
    }
    functions = [
        deepcopy(n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for fn in functions:
        fn.decorator_list = []

    def integer(value, width):
        value = int(value) % (1 << width)
        return value - (1 << width) if value >= (1 << (width - 1)) else value

    ns = {
        "cutlass": SimpleNamespace(
            Int32=lambda v: integer(v, 32), Int64=lambda v: integer(v, 64), Boolean=bool
        ),
        "load_layout_value": lambda values, index: values[index],
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "actual-prefix-scalar-model", "exec"
        ),
        ns,
    )

    class Offsets(list):
        element_type = staticmethod(lambda v: integer(v, bits))

    return ns, Offsets


@pytest.mark.parametrize("bits", (32, 64))
def test_actual_resolver_ordinal_partition_tail_empty_and_terminal(bits):
    ns, Offsets = _scalar_prefix_functions(bits)
    for values in ([0, 1, 129, 129, 386, 515], [0] * 6, [0, 129, 17, 146, 146, 515]):
        offsets = Offsets(values)
        prefix = [None] * 5
        sizes = (515 * 128, 128, 128, 515 * 132, 132, 132)
        assert ns["build_flat_tile_prefix"](offsets, prefix, 5, *sizes, 128, 128)
        expected = set()
        for group, (start, end) in enumerate(pairwise(values)):
            for row in range(0, max(0, end - start), 128):
                expected.update(
                    (
                        group,
                        row // 128,
                        col // 128,
                        start * 128,
                        start * 132,
                        end - start,
                    )
                    for col in (0, 128)
                )
        for grid in (1, 2, 3, 148, 296):
            found = []
            for block in range(grid):
                ordinal = block
                while True:
                    record = ns["resolve_flat_nm_work"](
                        offsets, prefix, ordinal, 5, *sizes, 128, 128
                    )
                    if record[2] == 0:
                        break
                    assert record[2] == 1
                    found.append(
                        (
                            record[3],
                            record[0],
                            record[1],
                            record[4],
                            record[8],
                            record[5],
                        )
                    )
                    ordinal += grid
            assert len(found) == len(set(found)) and set(found) == expected
        bad = Offsets([-1, 1, 1, 1, 1, 1])
        assert not ns["build_flat_tile_prefix"](bad, prefix, 5, *sizes, 128, 128)
        assert ns["resolve_flat_nm_work"](bad, prefix, 0, 5, *sizes, 128, 128)[2] == -1


def test_actual_SDK_state_wrap_delayed_UMMA_release_and_multi_tile_tail():
    tree = ast.parse(inspect.getsource(pipeline_helpers.PipelineState))
    klass = tree.body[0]
    names = {"__init__", "index", "count", "stages", "phase", "advance"}
    klass.body = [
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for fn in klass.body:
        fn.decorator_list = [
            n
            for n in fn.decorator_list
            if isinstance(n, ast.Name) and n.id == "property"
        ]
    ns = {
        "Int32": lambda value, **kw: int(value),
        "if_generate": lambda condition, yes, no, values, types, **kw: (
            yes if condition else no
        )(*values),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            klass,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "actual-SDK-state-scalar-model", "exec"
        ),
        ns,
    )
    State = ns["PipelineState"]
    producer, consumer = State(2, 0, 0, 1), State(2, 0, 0, 0)
    slots = [None, None]
    pending = None
    produced = consumed = released = 0
    # Seven output tiles, each with the original four K32 iterations. Delayed
    # UMMA completion keeps a stage unavailable even after its consumer read.
    while released < 7 * 4:
        if produced < 28 and slots[producer.index] is None:
            assert producer.index == produced % 2 and producer.phase == 1 ^ (
                (produced // 2) % 2
            )
            slots[producer.index] = produced
            produced += 1
            producer.advance()
        if pending is not None:
            slot, item = pending
            assert slots[slot] == item
            slots[slot] = None
            released += 1
            pending = None
        elif consumed < produced:
            assert (
                consumer.index == consumed % 2 and consumer.phase == (consumed // 2) % 2
            )
            assert slots[consumer.index] == consumed
            pending = consumer.index, consumed
            consumer.advance()
            consumed += 1
    assert produced == consumed == released == 28
    assert slots == [None, None]
