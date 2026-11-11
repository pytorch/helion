from __future__ import annotations

import ast
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.source import AttrSource

import helion
from helion._compiler.cute.host_full_sum import prove_host_full_sum
from helion._compiler.cute.standalone import _input_specializations
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.aot_structural_export import _export_classifiers
from helion.autotuner.aot_structural_export import _export_guard_key
from helion.autotuner.aot_structural_export import export_runtime_guards
import helion.language as hl
from helion.runtime.cute.standalone_launcher import _standalone_validate_aliases
from helion.runtime.cute.standalone_launcher import _standalone_validate_alignment

pytestmark = pytest.mark.cpu_only


@dataclass(frozen=True)
class DataclassWeight:
    weight: torch.Tensor


class NamedTupleWeight(NamedTuple):
    weight: torch.Tensor


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _add_field(x, holder):
    weight = holder.weight
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + weight[tile]
    return x


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _add_item(x, holder):
    weight = holder["weight"]
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + weight[tile]
    return x


@pytest.mark.parametrize("holder_type", [DataclassWeight, NamedTupleWeight, dict])
@pytest.mark.parametrize("export", ["standalone", "aot"])
def test_field_storage_guards_export_actual_bound_kernel(holder_type, export):
    x, weight = torch.empty(37), torch.empty(37)
    holder = holder_type(weight=weight)
    args = (x, holder)
    kernel = _add_item if holder_type is dict else _add_field
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        if holder_type is not dict:
            assert any(
                isinstance(source, AttrSource) and source.member == "weight"
                for specialization in bound.env.runtime_input_specializations.values()
                for source in specialization.sources
            )
        changed = (
            (x, holder_type(weight=x)),
            (x, holder_type(weight=torch.empty(38)[1:])),
        )
        if export == "aot":
            guards, observed = export_runtime_guards(bound, args)
            # Field paths retain the original str-step artifact encoding.
            assert (1, ("weight",)) in {path for guard in guards for path in guard[-1]}
            classifiers = _export_classifiers(guards)
            assert _export_guard_key(args, guards, classifiers) == observed
            for values in changed:
                assert _export_guard_key(values, guards, classifiers) != observed
        else:
            _, guards = _input_specializations(bound)
            code = compile(ast.Module(body=guards, type_ignores=[]), "<guards>", "exec")
            namespace = {
                "_standalone_validate_aliases": _standalone_validate_aliases,
                "_standalone_validate_alignment": _standalone_validate_alignment,
                "x": x,
                "holder": holder,
            }
            exec(code, namespace)
            for values in changed:
                namespace["holder"] = values[1]
                with pytest.raises(ValueError, match="aliasing|alignment"):
                    exec(code, namespace)


def test_namedtuple_field_full_standalone_source():
    # Standalone capture reads this device fact while running only fake tensors.
    with (
        cpu_render_target(),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        bound = cpu_bind(
            _add_field, (torch.empty(37), NamedTupleWeight(torch.empty(37)))
        )
        code = bound.to_code(
            helion.Config(block_sizes=[64]),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    compile(code, "<standalone>", "exec")
    assert "holder.weight" in code


@dataclass(frozen=True)
class DataclassMode:
    mode: str


class NamedTupleMode(NamedTuple):
    mode: str


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _sum_by_field(x, holder):
    partial = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        partial[tile] = x[tile, :].sum(-1)
    mode = holder.mode
    if mode == "sum":
        total = partial.sum()
    else:
        total = partial.mean()
    return total


@pytest.mark.parametrize("holder_type", [DataclassMode, NamedTupleMode])
@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_host_sum_folds_keyed_class_field(holder_type, mode):
    with cpu_render_target():
        bound = cpu_bind(_sum_by_field, (torch.empty(32, 8), holder_type(mode)))
        assert bound.host_function is not None
        with bound.env:
            proof = prove_host_full_sum(bound.host_function)
            assert [site.mean for site in proof.sites] == [mode == "mean"]
            config = bound.env.config_spec.default_config()
        config.config["cute_host_paired_sum"] = "narrow"
        code = bound.to_code(config)
    assert "_cute_try_full_sum(partial," in code
    assert "if mode ==" not in code


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _sum_by_nested_tensor_dtype(x, holders):
    partial = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        partial[tile] = x[tile, :].sum(-1)
    weight = holders["nested"][0].weight
    mode = weight.dtype
    if mode == torch.float32:
        total = partial.sum()
    else:
        total = partial.mean()
    return total


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _sum_by_nested_tensor_device(x, holders):
    partial = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        partial[tile] = x[tile, :].sum(-1)
    weight = holders["nested"][0].weight
    mode = weight.device.type
    if mode == "cpu":
        total = partial.sum()
    else:
        total = partial.mean()
    return total


@pytest.mark.parametrize("holder_type", [DataclassWeight, NamedTupleWeight])
@pytest.mark.parametrize("check_dtype", [False, True])
def test_host_sum_field_metadata_respects_tensor_key(holder_type, check_dtype):
    args = (
        torch.empty(32, 8),
        {"nested": [holder_type(torch.empty(8))]},
    )
    kernel = (
        _sum_by_nested_tensor_dtype if check_dtype else _sum_by_nested_tensor_device
    )
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env:
            proof = prove_host_full_sum(bound.host_function)
    # Dtype is keyed; device is deliberately absent from tensor keys.
    assert bool(proof.sites) is check_dtype
