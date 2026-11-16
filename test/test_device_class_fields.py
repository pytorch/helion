from __future__ import annotations

import ast
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.source import AttrSource
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource

import helion
from helion._compiler.compile_environment import _find_tensor_input_source
from helion._compiler.compile_environment import _is_supported_tensor_input_source
from helion._compiler.compile_environment import _replay_tensor_input_source
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tensor_container_guards import TensorField
from helion._testing.tensor_container_guards import TensorTuple
from helion._testing.tensor_container_guards import field_increment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@dataclass(frozen=True)
class DataclassScale:
    scale: float


class NamedTupleScale(NamedTuple):
    scale: float


@helion.kernel(autotune_effort="none")
def _scaled(x, config: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * config.scale
    return out


@pytest.mark.parametrize("holder_type", [DataclassScale, NamedTupleScale])
@onlyBackends(["triton", "cute"])
def test_device_code_reads_class_field(holder_type):
    with cpu_render_target():
        bound = cpu_bind(_scaled, (torch.empty(17), holder_type(2.5)))
        assert bound.host_function is not None
        products = [
            node
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target == torch.ops.aten.mul.Tensor
        ]
        assert len(products) == 1 and products[0].args[1] == 2.5
        ast.parse(bound.to_code(helion.Config(block_sizes=[32])))


@dataclass(frozen=True)
class DataclassFields:
    scale: float
    weight: torch.Tensor


class NamedTupleFields(NamedTuple):
    scale: float
    weight: torch.Tensor


@pytest.mark.parametrize("nonleaf", [False, True])
def test_dataclass_specialization_key_does_not_copy_tensor_fields(nonleaf):
    weight = torch.ones(17, requires_grad=True)
    if nonleaf:
        weight = weight * 2
    args = (torch.empty(17), DataclassFields(2.5, weight))
    with patch.object(
        torch.Tensor, "__deepcopy__", side_effect=AssertionError("copied tensor")
    ):
        key = _runtime_fields.specialization_key(args)
        assert key == _runtime_fields.specialization_key(args)
        assert key != _runtime_fields.specialization_key(
            (args[0], DataclassFields(2.5, weight.to(torch.float64)))
        )
    assert args[1].weight is weight


@dataclass
class DataclassNonInitFields:
    weight: torch.Tensor
    scale: float = field(init=False, default=2.5)


@dataclass(frozen=True, slots=True)
class DataclassPseudoFields:
    weight: torch.Tensor
    initial_scale: InitVar[float]
    scale: float = field(init=False)
    category: ClassVar[object] = object()

    def __post_init__(self, initial_scale):
        object.__setattr__(self, "scale", initial_scale)


@pytest.mark.parametrize("holder_type", [DataclassNonInitFields, DataclassPseudoFields])
@onlyBackends(["triton", "cute"])
def test_fake_dataclass_preserves_non_init_fields_without_reconstruction(holder_type):
    x = torch.empty(17)
    if holder_type is DataclassPseudoFields:
        holder = holder_type(x, 2.5)
    else:
        holder = holder_type(x)
    object.__setattr__(holder, "scale", 7.5)
    with (
        patch.object(
            holder_type, "__init__", side_effect=AssertionError("called init")
        ),
        patch.object(
            holder_type, "__copy__", return_value=holder, create=True
        ) as copy_hook,
        cpu_render_target(),
    ):
        bound = cpu_bind(_runtime_fields, (x, holder))
        code = bound.to_code(helion.Config(block_sizes=[32]))
    assert "config.scale" in code and "config.weight" in code
    copy_hook.assert_not_called()
    assert holder.weight is x and holder.scale == 7.5
    assert bound.host_function is not None
    fake = bound.fake_args[1]
    assert fake is not holder and type(fake) is holder_type
    assert fake.weight is not x
    if holder_type is DataclassPseudoFields:
        assert fake.category is holder.category


@helion.kernel(autotune_effort="none")
def _runtime_fields(x, config):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * config.scale + config.weight[tile]
    return out


@pytest.mark.parametrize("holder_type", [DataclassFields, NamedTupleFields])
@onlyBackends(["triton", "cute"])
def test_host_launcher_reads_runtime_fields_as_attributes(holder_type):
    x = torch.empty(17)
    config = holder_type(2.5, torch.empty_like(x))
    with cpu_render_target():
        bound = cpu_bind(_runtime_fields, (x, config))
        code = bound.to_code(helion.Config(block_sizes=[32]))
    wrapper = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name == _runtime_fields.fn.__name__
    )
    (launcher,) = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    field_arguments = {
        ast.unparse(argument): argument
        for argument in launcher.args
        if any(isinstance(n, ast.Name) and n.id == "config" for n in ast.walk(argument))
    }
    assert "config.scale" in field_arguments and "config.weight" in field_arguments
    assert not any("config[" in value for value in field_arguments)
    # Evaluate the actual emitted launcher arguments without launching a GPU.
    namespace = {"config": config}
    scalar = eval(
        compile(ast.Expression(field_arguments["config.scale"]), "<field>", "eval"),
        namespace,
    )
    tensor = eval(
        compile(ast.Expression(field_arguments["config.weight"]), "<field>", "eval"),
        namespace,
    )
    assert scalar == 2.5 and tensor is config.weight


@helion.kernel(static_shapes=False, autotune_effort="none")
def _specialized_field_stride(groups, device):
    x = groups["configs"][0].weight
    stride = hl.specialize(x.stride(-1))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + stride
    return out


@pytest.mark.parametrize("holder_type", [DataclassFields, NamedTupleFields])
@onlyBackends(["triton", "cute"])
def test_nested_field_stride_specialization_replays_actual_inputs(holder_type):
    x = torch.empty(16, 16)
    original = ({"configs": [holder_type(2.5, x)]}, x.device)
    transposed = ({"configs": [holder_type(2.5, x.T)]}, x.device)
    with cpu_render_target():
        bound = cpu_bind(_specialized_field_stride, original)
        extractors = bound._specialize_extra()
    assert extractors
    assert tuple(fn(original) for fn in extractors) != tuple(
        fn(transposed) for fn in extractors
    )


@pytest.mark.parametrize("holder_type", [DataclassFields, NamedTupleFields])
def test_nested_field_source_is_supported_and_replayable(holder_type):
    x = torch.empty(16, 16)
    groups = {"configs": [holder_type(2.5, x)]}
    root = LocalSource("groups", is_input=True)
    source = AttrSource(GetItemSource(GetItemSource(root, "configs"), 0), "weight")
    assert _is_supported_tensor_input_source(source)
    assert _replay_tensor_input_source(source, {"groups": groups}) is x
    assert _find_tensor_input_source(x, groups, root) == source


@helion.kernel(static_shapes=False, autotune_effort="none")
def _specialized_field_size(groups, device):
    x = groups["configs"][0].weight
    size = hl.specialize(x.size(-1))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + size
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def _specialized_field_scalar(groups, device):
    config = groups["configs"][0]
    scale = hl.specialize(config.scale)
    x = config.weight
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + scale
    return out


@pytest.mark.parametrize("holder_type", [DataclassFields, NamedTupleFields])
@pytest.mark.parametrize("specialization", ["size", "scalar"])
@pytest.mark.parametrize("sequence_type", [list, tuple])
@onlyBackends(["triton", "cute"])
def test_nested_field_specialization_keeps_fake_conversion_path(
    holder_type, specialization, sequence_type
):
    x = torch.empty(16, 16)
    original = ({"configs": sequence_type([holder_type(5, x)])}, x.device)
    if specialization == "size":
        kernel = _specialized_field_size
        changed_holder = holder_type(5, torch.empty(16, 17))
    else:
        kernel = _specialized_field_scalar
        changed_holder = holder_type(7, x)
    changed = ({"configs": sequence_type([changed_holder])}, x.device)
    with cpu_render_target():
        bound = cpu_bind(kernel, original)
        extractors = bound._specialize_extra()
    original_key = tuple(fn(original) for fn in extractors)
    changed_key = tuple(fn(changed) for fn in extractors)
    hash(original_key)
    hash(changed_key)
    assert original_key != changed_key


@helion.kernel(static_shapes=True, autotune_effort="none")
def _captured_tensor(x, callback):
    weight = callback()
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + weight[tile]
    return out


@onlyBackends(["triton", "cute"])
def test_captured_callable_tensor_stays_external():
    x, weight = torch.empty(16), torch.empty(16)

    def callback():
        return weight

    with cpu_render_target():
        bound = cpu_bind(_captured_tensor, (x, callback))
        assert bound.host_function is not None
        captures = [
            (tensor, source)
            for tensor, source in bound.env.input_sources.items()
            if isinstance(source, AttrSource) and source.member == "cell_contents"
        ]
        assert len(captures) == 1
        tensor, source = captures[0]
        assert not _is_supported_tensor_input_source(source)
        with bound.env, bound.host_function:
            assert bound.env.tensor_alignment_owner(tensor) is None
            assert bound.env.tensor_storage_input_source(tensor) is None
        assert len(bound._host_semantic_input_tensors()) == 1
        assert len(bound._host_semantic_external_tensor_keys()) == 1


@pytest.mark.parametrize("holder_type", [TensorField, TensorTuple])
@pytest.mark.parametrize("direct", [False, True])
def test_record_tensor_store_matches_local_alias(holder_type, direct):
    values = torch.arange(38, dtype=torch.float32)[::2]
    holder = holder_type(values)
    with cpu_render_target():
        kernel = helion.kernel(field_increment.fn, static_shapes=False)
        bound = cpu_bind(kernel, (holder, direct, values.device))
        code = bound.to_code(helion.Config(block_sizes=[32]))
        assert "holder.weight" in code
        additions = [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]
        assert additions == [2]
        ast.parse(code)
    assert holder.weight is values
    torch.testing.assert_close(values, torch.arange(38, dtype=torch.float32)[::2])


@helion.kernel(autotune_effort="none")
def _invalid_record_store(holder, device: torch.device):
    for tile in hl.tile(holder.weight.size()):
        holder.weight[tile] = (1, 2)
    return holder.weight


def test_record_tensor_store_still_validates_rhs():
    with cpu_render_target(), pytest.raises(helion.exc.RequiresTensorInAssignment):
        cpu_bind(
            _invalid_record_store, (TensorField(torch.ones(7)), torch.device("cpu"))
        )
