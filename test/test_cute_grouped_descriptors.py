from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import patch

from cutlass.experimental.primitives import nvvm_wrapper
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion import exc
from helion._compiler.cute import tcgen05_grouped_descriptors
from helion._compiler.cute import tensormap_alloc_class
from helion._compiler.cute.autotune.heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_kernel_module,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_wrapper_plans,
)
from helion._compiler.cute.tcgen05_flat_grouped_ir import prove_flat_grouped_rna
from helion._compiler.cute.tcgen05_flat_grouped_plan import flat_grouped_rna_schedule
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import ELEMENT_BYTES
from helion._compiler.cute.tcgen05_grouped_descriptors import EXTRA_EXTENT
from helion._compiler.cute.tcgen05_grouped_descriptors import LARGE_ALLOC_MIN_BYTES
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAP_EXTENT
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAP_STRIDE
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._compiler.cute.tcgen05_grouped_descriptors import (
    WrappedGroupedDescriptorPlan,
)
from helion._compiler.cute.tcgen05_grouped_descriptors import (
    wrapped_grouped_descriptor_supported,
)
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target
from helion._testing.cute_flat_grouped import bound_host_function as _host
from helion._testing.cute_flat_grouped import compiled_source as _source
from helion._testing.cute_flat_grouped import cuda_binding
from helion._testing.cute_flat_grouped import flat_converter_config as _config
from helion._testing.cute_flat_grouped import flat_rna_schedule as _schedule
from helion._testing.cute_flat_grouped import native_kernel_source as _native_source
from helion._testing.cute_grouped_offsets import grouped_fp32_output
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _create_cute_wrapper

pytestmark = pytest.mark.cpu_only


def signed(value: int, bits: int) -> int:
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("stride", [4, 8, 32, 128, 1024])
def test_modular_element_bases_and_tail_addressing(bits: int, stride: int) -> None:
    # Enumerate valid source products, including large signed offsets that wrap
    # back into the allocation, and compare the original per-element arithmetic.
    period = (1 << bits) // stride
    for raw_start in (0, 1, 37, period, -period, period + 37, -period + 37):
        raw_start = signed(raw_start, bits)
        base = signed(raw_start * stride, bits)
        assert 0 <= base < 4096 * stride and base % stride == 0
        for length in (1, 17, 127, 128, 129):
            for i in (0, length - 1, length, length + 127):
                coord = WRAP_EXTENT - length + i
                in_bounds = 0 <= coord < WRAP_EXTENT
                assert in_bounds == (i < length)
                if in_bounds:
                    row_base = base // stride
                    address = (
                        coord * stride
                        + WRAP_EXTENT * (WRAP_STRIDE - stride)
                        + (row_base + length) * stride
                    ) % (1 << 64)
                    assert address == signed(signed(raw_start + i, bits) * stride, bits)
                    assert 0 <= row_base + length < EXTRA_EXTENT
    # Non-power-of-two source wrap can yield an aligned-to-elements but non-row
    # base. Such a binding must stay dynamic, even if this one interval is valid.
    assert signed((1 << (bits - 2)) * 12 + 4, bits) == 4


@pytest.mark.parametrize(
    "metadata",
    [
        (3, 273, 96, 128, 64),
        (3, 273, 128, 132, 64),
        (True, 273, 128, 128, 64),
        (3, 273, 128, 128, 16),
        (3, 1 << 30, 128, 128, 64),
        (1 << 20, 273, 128, 128, 64),
    ],
)
def test_unsupported_metadata_is_not_admitted(metadata: tuple[int, ...]) -> None:
    assert not wrapped_grouped_descriptor_supported(*metadata)
    with pytest.raises(ValueError):
        WrappedGroupedDescriptorPlan(*metadata)


@pytest.mark.parametrize(
    "rows,k,n,admitted",
    [
        (256, 128, 128, True),
        (255, 128, 128, False),
        # The narrower of the two row-packed buffers decides.
        (1024, 128, 32, True),
        (1023, 128, 32, False),
        (1023, 32, 128, False),
    ],
)
def test_wrapped_buffers_span_the_large_class_window(
    rows: int, k: int, n: int, admitted: bool
) -> None:
    # The driver encodes a wrapped TensorMap in the large allocation class,
    # which reads up to this far from the buffer's first element.
    assert (ELEMENT_BYTES * rows * min(k, n) >= LARGE_ALLOC_MIN_BYTES) is admitted
    assert wrapped_grouped_descriptor_supported(3, rows, k, n, 64) is admitted


def _expression_statements(source: str) -> list[str]:
    return [
        ast.unparse(node)
        for node in sorted(
            (
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Expr)
            ),
            key=lambda node: node.lineno,
        )
    ]


@pytest.mark.parametrize("tma_rn", [False, True])
def test_dynamic_templates_take_the_small_allocation_class(tma_rn: bool) -> None:
    with cuda_binding(
        static=True, original=True, groups=3, rows=273, k=128, n=128
    ) as bound:
        proof = prove_flat_grouped_rna(bound.env, _host(bound).device_ir)
        assert proof is not None
        schedule = flat_grouped_rna_schedule(
            bound.env,
            proof,
            converter_warps=0 if tma_rn else 8,
            block_k=32,
            ab_stages=2 if tma_rn else 6,
            shared_capacity=232448,
            tma_rn=tma_rn,
        )
        assert schedule is not None and schedule.descriptors is None
    statements = _expression_statements(flat_grouped_kernel_module(schedule))
    # The templates of the row-packed operand and of the output are encoded
    # for all rows and rewritten to one group's: the small class, right after
    # the copy and before any rewrite.
    for manager, atom, template, warp, barrier in (
        (
            "tcgen05_grouped_tensormap_manager",
            "tma_atom_b",
            "tcgen05_grouped_tensormap_b_smem_ptr",
            5,
            "cute.arch.sync_warp()",
        ),
        (
            "tcgen05_grouped_d_tensormap_manager",
            "tcgen05_tma_store_atom",
            "tcgen05_grouped_d_tensormap_smem_ptr",
            0,
            "cute.arch.sync_threads()",
        ),
    ):
        copied = statements.index(
            f"{manager}.init_tensormap_from_atom({atom}, {template}, {warp})"
        )
        # Every lane clears it: the flat kernel's form
        # (the general grouped path elects one lane).
        assert statements[copied + 1] == (
            f"select_small_alloc_class_every_lane({template}, {warp})"
        )
        rewrite = next(
            i
            for i, text in enumerate(statements)
            if text.startswith(f"{manager}.update_tensormap(")
        )
        # The lanes' stores are ordered ahead of the rewrite, in which one
        # lane replaces fields of the template: the output's template waits
        # for the block's barrier, and the operand's warp, which rewrites
        # next, synchronizes right behind its clear.
        assert barrier in statements[copied + 2 : rewrite]
        if warp == 5:
            assert statements[copied + 2] == barrier
    # The dense template describes one group's block and only ever another
    # block of that extent: it keeps the class the driver encoded.
    selections = [text for text in statements if "select_small_alloc_class" in text]
    assert len(selections) == 2
    assert not any("_a_smem_ptr" in text for text in selections)


def test_two_ordinary_inputs_clear_every_template_they_rewrite() -> None:
    # The general grouped path: packed rows routed to one weight matrix per
    # group by device offsets, both operands and the output through
    # TensorMaps the kernel rewrites per group. One lane of the owning warp
    # clears each template's class right behind its copy.
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 512, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    config = helion.Config(
        block_sizes=[256, 128, 128],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=3,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_full_allocation_a=False,
    )
    with tcgen05_target(), mock_cuda_unavailable(), forbid_cuda_work():
        source = cpu_bind(grouped_fp32_output, args).to_code(config)
    statements = _expression_statements(source)
    copies = [
        index
        for index, text in enumerate(statements)
        if ".init_tensormap_from_atom(" in text
    ]
    # Both operands' templates and the output's.
    assert len(copies) == 3
    for index in copies:
        _atom, template, warp = (
            statements[index].split("init_tensormap_from_atom(")[1][:-1].split(", ")
        )
        assert statements[index + 1] == (
            f"_cute_select_small_alloc_class({template}, {warp})"
        )
    assert source.count("_cute_select_small_alloc_class(") == 3


def test_the_helper_clears_the_bit_the_dsl_fix_up_clears() -> None:
    # CuTe DSL 4.8.1's experimental ``tensormap_replace`` resets a descriptor
    # copy to the small class by clearing one bit of one 32-bit word
    # (``_apply_tensormap_fixup``). The helper addresses the descriptor by
    # quadwords and clears the same bit. A DSL release that moves the bit, or
    # no longer names it, fails here.
    cleared = ~nvvm_wrapper._TENSORMAP_FIXUP_MASK & 0xFFFFFFFF
    assert cleared.bit_count() == 1
    quadword, half = divmod(nvvm_wrapper._TENSORMAP_FIXUP_WORD, 2)
    assert (quadword, cleared << (32 * half)) == (
        tensormap_alloc_class.ALLOC_CLASS_QWORD,
        tensormap_alloc_class.ALLOC_CLASS_LARGE,
    )


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("warps,bk", [(1, 32), (8, 32), (4, 64)])
def test_typed_plan_wrapper_device_and_cache_agree(
    bits: int, warps: int, bk: int
) -> None:
    with cuda_binding(
        static=True, original=True, bits=bits, groups=3, rows=273, k=128, n=128
    ) as bound:
        old = _schedule(bound, warps, bk)
        assert old is not None
        descriptors = WrappedGroupedDescriptorPlan(3, 273, 128, 128, bits)
        new = dataclasses.replace(old, descriptors=descriptors)
        assert new.storage == old.storage and new.roles == old.roles
        assert new.cache_identity()[:-1] == old.cache_identity()
        assert new.cache_identity()[-1] == descriptors.cache_identity()
        native = flat_grouped_kernel_module(new)
        assert "TensorMapManager" not in native and "tma_desc_ptr=" not in native
        assert "alloc_class" not in native
        assert "convert_and_publish_operand_stage" in native
        assert "ld_volatile_shared_u32" in native
        plans = flat_grouped_wrapper_plans(new)
        body: list[str] = []
        args: list[str] = []
        for plan in plans:
            _append_cute_wrapper_plan(body, args, plan, num_sm=148)
        wrapper = "\n".join(body)
        assert wrapper.count("internal_type=cutlass.Float32") == 2
        assert descriptors.layout("arg1") in wrapper
        assert descriptors.layout("arg4", output=True) in wrapper
        assert "arg2_shape1, arg2_shape2, arg2_shape0" in wrapper
        broken = deepcopy(plans[0])
        broken["wrapped_grouped_descriptors"] = WrappedGroupedDescriptorPlan(
            3, 274, 128, 128, bits
        ).cache_identity()
        with pytest.raises(helion.exc.BackendUnsupported):
            _append_cute_wrapper_plan([], [], broken)


@pytest.mark.parametrize(
    "rows,k,n",
    # 255 rows of 128: one row short of the large-class window.
    [(273, 128, 128), (273, 96, 128), (273, 128, 132), (255, 128, 128)],
)
def test_ordinary_config_policy_normalization_and_seed_identity(
    rows: int, k: int, n: int
) -> None:
    with cuda_binding(
        static=True, original=True, groups=3, rows=rows, k=k, n=n
    ) as bound:
        old = _config(bound, 8, 32)
        dynamic = helion.Config.from_dict(old.config | {DESCRIPTOR_KEY: "dynamic"})
        wrapped = helion.Config.from_dict(old.config | {DESCRIPTOR_KEY: WRAPPED})
        spec = bound.config_spec
        assert spec.normalized_config(dynamic) == spec.normalized_config(old)
        normalized = spec.normalized_config(wrapped)
        supported = k == n == 128 and rows >= 256
        assert (DESCRIPTOR_KEY in normalized.config) == supported
        assert (
            spec.normalized_config(
                helion.Config.from_dict(json.loads(json.dumps(normalized.config)))
            )
            == normalized
        )
        generator = ConfigGeneration(spec)
        restored = generator.unflatten(generator.flatten(normalized))
        assert restored.get(DESCRIPTOR_KEY, "dynamic") == (
            WRAPPED if supported else "dynamic"
        )
        old_source = _source(bound, old)
        new_source = _source(bound, wrapped)
        assert _source(bound, dynamic) == old_source
        if supported:
            assert "TensorMapManager" not in _native_source(new_source)
            assert "TensorMapManager" in _native_source(old_source)
            # Full original allocating host remains identical.
            hosts = [
                next(
                    x
                    for x in ast.parse(s).body
                    if isinstance(x, ast.FunctionDef) and x.name == "jagged_dense_bmm"
                )
                for s in (old_source, new_source)
            ]
            assert ast.dump(hosts[0]) == ast.dump(hosts[1])
        else:
            assert new_source == old_source
        seeds = CuteGroupedRnaHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        # Preserve the exact old RNA descriptor-policy Cartesian product.
        seeds = [seed for seed in seeds if seed.get("cute_grouped_rna_warps", 0)]
        if supported:
            ordinary = [s for s in seeds if DESCRIPTOR_KEY not in s.config]
            selected = [s for s in seeds if s.get(DESCRIPTOR_KEY) == WRAPPED]
            assert len(ordinary) == len(selected) == 8
            assert [s.config for s in ordinary] == [
                {k: v for k, v in s.config.items() if k != DESCRIPTOR_KEY}
                for s in selected
            ]
        else:
            assert all(DESCRIPTOR_KEY not in s.config for s in seeds)


def test_dynamic_binding_does_not_offer_policy() -> None:
    with cuda_binding(static=False, original=True, k=128, n=128) as bound:
        assert not bound.config_spec.cute_grouped_wrapped_descriptors_available
        assert not bound.config_spec.cute_grouped_rna_k_choices


@pytest.mark.parametrize("mutation", ["missing", "different", "wrong_kind"])
def test_wrapper_rejects_disagreeing_input_and_output_proofs(mutation: str) -> None:

    with cuda_binding(static=True, original=True, k=128, n=128) as bound:
        old = _schedule(bound)
        assert old is not None
        plan = WrappedGroupedDescriptorPlan(3, 273, 128, 128, 64)
        plans = flat_grouped_wrapper_plans(dataclasses.replace(old, descriptors=plan))
    if mutation == "missing":
        plans[1].pop("wrapped_grouped_descriptors")
    elif mutation == "different":
        plans[1]["wrapped_grouped_descriptors"] = WrappedGroupedDescriptorPlan(
            3, 274, 128, 128, 64
        ).cache_identity()
    else:
        plans[1]["kind"] = "tcgen05_aux_tma"
    kernel = SimpleNamespace(__name__="proof_only", _helion_cute_wrapper_plans=plans)
    with pytest.raises(helion.exc.BackendUnsupported, match="one shared proof"):
        _create_cute_wrapper(kernel, (), (32, 12, 1))


@pytest.mark.parametrize("bits", [32, 64])
def test_renamed_contract_with_distinct_row_strides_uses_ordinary_policy(
    bits: int,
) -> None:
    # 1027 rows: both row-packed buffers span the large-class window.
    with cuda_binding(static=True, bits=bits, groups=5, rows=1027, k=64, n=32) as bound:
        config = helion.Config.from_dict(
            _config(bound, 2, 32).config | {DESCRIPTOR_KEY: WRAPPED}
        )
        source = _source(bound, config)
        native = _native_source(source)
        assert "TensorMapManager" not in native
        assert "routing, left.view((1027, 64))" in source
        assert "right.transpose(1, 2)" in source
        assert "flat_contract" in source
        assert "jagged_dense_bmm" not in source
        assert "cutlass.Int64(a_packed.layout.stride[0])" in native
        assert "cutlass.Int64(out.layout.stride[0])" in native


@pytest.mark.parametrize("bits", [32, 64])
def test_wrapped_source_for_a_buffer_below_the_window_is_refused(bits: int) -> None:
    """Generated source names its descriptor policy in the kernel's wrapper
    plans, where the launcher reads it, and source that left the process
    (``BoundKernel.to_code``) can have been generated without the large-class
    window: with the wrapped policy for a row-packed buffer below it, here an
    output of 131 rows of 32 columns, 16768 bytes. The launcher refuses that
    kernel and says what to do."""
    with (
        # The admission without the window.
        patch.object(tcgen05_grouped_descriptors, "LARGE_ALLOC_MIN_BYTES", 0),
        cuda_binding(static=True, bits=bits, groups=5, rows=131, k=64, n=32) as bound,
    ):
        config = helion.Config.from_dict(
            _config(bound, 2, 32).config | {DESCRIPTOR_KEY: WRAPPED}
        )
        native = _native_source(_source(bound, config))
    kernel = PyCodeCache.load(native)._helion_native_grouped_rna
    identity = (
        "wrapped_flat_grouped_descriptors_v1",
        *(5, 131, 64, 32, bits),
        *(WRAP_EXTENT, WRAP_STRIDE, EXTRA_EXTENT),
    )
    assert [
        plan["wrapped_grouped_descriptors"]
        for plan in kernel._helion_cute_wrapper_plans
    ] == [identity, identity]
    assert not wrapped_grouped_descriptor_supported(5, 131, 64, 32, bits)
    with pytest.raises(exc.BackendUnsupported) as refused:
        _create_cute_wrapper(kernel, (), (32, 12, 1))
    message = str(refused.value)
    assert f"= (5, 131, 64, 32, {bits})" in message
    assert f"span {LARGE_ALLOC_MIN_BYTES} bytes" in message
    assert "has to be generated again" in message
