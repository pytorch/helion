from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from test.test_cute_flat_grouped_codegen import _schedule
from test.test_cute_flat_grouped_ir import cuda_binding
from test.test_cute_flat_grouped_public import _config
from test.test_cute_flat_grouped_public import _native_source
from test.test_cute_flat_grouped_public import _source

import helion
from helion._compiler.autotuner_heuristics.cute_grouped_rna import (
    CuteGroupedRnaHeuristic,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_kernel_module,
)
from helion._compiler.cute.tcgen05_flat_grouped_codegen import (
    flat_grouped_wrapper_plans,
)
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import EXTRA_EXTENT
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAP_EXTENT
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAP_STRIDE
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._compiler.cute.tcgen05_grouped_descriptors import (
    WrappedGroupedDescriptorPlan,
)
from helion._compiler.cute.tcgen05_grouped_descriptors import (
    wrapped_grouped_descriptor_supported,
)
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _create_cute_wrapper


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


@pytest.mark.parametrize("k,n", [(128, 128), (96, 128), (128, 132)])
def test_ordinary_config_policy_normalization_and_seed_identity(k: int, n: int) -> None:
    with cuda_binding(
        static=True, original=True, groups=3, rows=273, k=k, n=n
    ) as bound:
        old = _config(bound, 8, 32)
        dynamic = helion.Config.from_dict(old.config | {DESCRIPTOR_KEY: "dynamic"})
        wrapped = helion.Config.from_dict(old.config | {DESCRIPTOR_KEY: WRAPPED})
        spec = bound.config_spec
        assert spec.normalized_config(dynamic) == spec.normalized_config(old)
        normalized = spec.normalized_config(wrapped)
        supported = k == n == 128
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
    with cuda_binding(static=True, bits=bits, groups=5, rows=131, k=64, n=32) as bound:
        config = helion.Config.from_dict(
            _config(bound, 2, 32).config | {DESCRIPTOR_KEY: WRAPPED}
        )
        source = _source(bound, config)
        native = _native_source(source)
        assert "TensorMapManager" not in native
        assert "routing, left.view((131, 64))" in source
        assert "right.transpose(1, 2)" in source
        assert "flat_contract" in source
        assert "jagged_dense_bmm" not in source
        assert "cutlass.Int64(a_packed.layout.stride[0])" in native
        assert "cutlass.Int64(out.layout.stride[0])" in native
