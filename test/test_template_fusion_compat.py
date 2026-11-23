from __future__ import annotations

from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor import ir
from torch._inductor import select_algorithm
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import helion
from helion import _compat
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def clear_version_caches():
    def clear():
        _compat.requires_torch_version.cache_clear()
        _compat.torch_uses_template_producer_fusion.cache_clear()
        _compat.supports_torch_compile_template_lowering.cache_clear()
        _compat.supports_torch_compile_fusion.cache_clear()

    clear()
    yield
    clear()


@pytest.mark.parametrize(
    "torch_version,producer_api",
    [
        ("2.13.0", False),
        ("2.15.1+cu130", False),
        ("2.16.0.dev20261005", False),
        ("2.16.0.dev20261006+cu130", False),
        ("2.16.0.dev20261007+cu130", True),
        ("2.16.0.dev20261008", True),
        ("2.16.0a0+gitabcdef", True),
        ("2.16.0rc1", True),
        ("2.16.0", True),
        ("2.17.0.dev20261008", True),
    ],
)
def test_template_fusion_version_boundary(monkeypatch, torch_version, producer_api):
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(ir, "TemplateBuffer", _ProducerTemplate)
    assert _compat.torch_uses_template_producer_fusion() is producer_api


class _LegacyTemplate:
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        mutated_inputs=None,
        allowed_prologue_inps=None,
        named_inputs=None,
    ):
        self.inputs = inputs
        self.allowed_prologue_inps = allowed_prologue_inps or OrderedSet()
        self.allow_prologue_fusion = None
        self.allow_epilogue_fusion = None

    def get_allowed_prologue_inps(self):
        return self.allowed_prologue_inps

    def has_aliasing_or_mutation_for_prologue_fusion(self, node):
        return node.has_aliasing_or_mutation()


class _ProducerTemplate:
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        mutated_inputs=None,
        load_input_fusion_allowed_inputs=None,
        store_output_fusion_allowed_inputs=None,
        named_inputs=None,
    ):
        self.inputs = inputs
        self.load_input_fusion_allowed_inputs = (
            load_input_fusion_allowed_inputs or OrderedSet()
        )
        self.store_output_fusion_allowed_inputs = (
            store_output_fusion_allowed_inputs or OrderedSet()
        )
        self.allow_prologue_fusion = None
        self.allow_epilogue_fusion = None

    def has_aliasing_or_mutation_for_producer_fusion(self, node):
        return node.has_aliasing_or_mutation()


@pytest.mark.parametrize(
    "missing_api", ["template_class", "constructor_keyword", "mutation_hook"]
)
def test_template_fusion_requires_the_new_api(monkeypatch, missing_api):
    monkeypatch.setattr(torch, "__version__", "2.16.0.dev20261007+cu130")
    monkeypatch.setattr(ir, "TemplateBuffer", _ProducerTemplate)
    if missing_api == "template_class":
        monkeypatch.delattr(ir, "TemplateBuffer")
    elif missing_api == "constructor_keyword":
        monkeypatch.setattr(_ProducerTemplate, "__init__", _LegacyTemplate.__init__)
    else:
        monkeypatch.delattr(
            _ProducerTemplate, "has_aliasing_or_mutation_for_producer_fusion"
        )
    assert not _compat.torch_uses_template_producer_fusion()


@pytest.fixture(params=["legacy", "producer", "new-version-legacy"])
def template_api(request):
    if request.param == "producer":
        return (
            "2.16.0.dev20261007+cu130",
            _ProducerTemplate,
            "has_aliasing_or_mutation_for_producer_fusion",
        )
    return (
        "2.16.0a0+gitabcdef"
        if request.param == "new-version-legacy"
        else "2.16.0.dev20261006+cu130",
        _LegacyTemplate,
        "has_aliasing_or_mutation_for_prologue_fusion",
    )


@pytest.mark.parametrize("matching_version", [True, False])
def test_fusion_support_checks_the_versioned_hook(
    monkeypatch, template_api, matching_version
):
    torch_version, template_cls, hook = template_api
    if not matching_version:
        torch_version = (
            "2.16.0.dev20261006"
            if template_cls is _ProducerTemplate
            else "2.16.0.dev20261007"
        )
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(ir, "TemplateBuffer", template_cls)
    monkeypatch.setattr(
        select_algorithm, "ExternalTritonTemplateKernel", object, raising=False
    )
    # A new version that still exposes the legacy API should remain supported.
    assert _compat.supports_torch_compile_fusion() is (
        matching_version or template_cls is _LegacyTemplate
    )


@pytest.mark.parametrize("allowed", [None, OrderedSet(("producer",))])
def test_template_adapter_preserves_inputs_and_mutation_gate(
    monkeypatch, template_api, allowed
):
    if not _compat.supports_torch_compile_fusion():
        pytest.skip("this PyTorch build lacks Helion template fusion entrypoints")
    from helion._compiler._inductor.template_buffer import HelionTemplateBuffer

    torch_version, template_cls, hook = template_api
    monkeypatch.setattr(torch, "__version__", torch_version)
    _compat.torch_uses_template_producer_fusion.cache_clear()
    monkeypatch.setattr(ir.TemplateBuffer, "__init__", template_cls.__init__)
    monkeypatch.setattr(
        ir.TemplateBuffer, hook, getattr(template_cls, hook), raising=False
    )
    monkeypatch.setattr(
        ir.TemplateBuffer,
        "get_allowed_prologue_inps",
        _LegacyTemplate.get_allowed_prologue_inps,
        raising=False,
    )
    allowed = allowed.copy() if allowed is not None else None
    template = HelionTemplateBuffer(
        layout=None,
        inputs=[],
        kernel=None,
        bound_kernel=None,
        constant_args={},
        allowed_prologue_inps=allowed,
    )
    actual = template.get_allowed_prologue_inps()
    assert actual == (allowed or OrderedSet())
    if allowed:
        assert actual is allowed
    if template_cls is _ProducerTemplate:
        assert not template.store_output_fusion_allowed_inputs

    # Use the scheduler's actual entrypoint for each API. The conservative
    # base-class hook would reject even independent synthetic mutations.
    gate = getattr(template, hook)
    mutation = SimpleNamespace(
        node=object.__new__(ir.MutationOutput),
        get_aliases=lambda: (),
        get_mutations=lambda: ("mutated",),
    )
    node = SimpleNamespace(
        get_outputs=lambda: [mutation],
        has_aliasing_or_mutation=lambda: True,
    )
    assert gate(node) is False
    actual.add("mutated")
    assert gate(node) is True


@helion.kernel(autotune_effort="none")
def _reads_identity_and_device_metadata(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    assert x is not None
    out = torch.empty(y.size())
    for tile_m, tile_n in hl.tile(y.size()):
        loaded = hl.load(y, [tile_m, tile_n])
        out[tile_m, tile_n] = x[tile_m, tile_n].to(x.dtype) + loaded
    return out


@helion.kernel(autotune_effort="none")
def _reads_layout_and_grid_bounds(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def _reads_values(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for tile in hl.tile(out.size()):
        out[tile] = out[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def _reads_identity_and_no_load(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x is not y
    out = torch.empty_like(y)
    for tile in hl.tile(x.size()):
        out[tile] = y[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def _reads_types(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    assert not isinstance(y, torch.nn.Parameter)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def _reads_loads_in_two_loops(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for tile_m in hl.tile(y.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(y.size(1)):
            acc = acc + x[tile_m, tile_n].sum(-1)
        for tile_n in hl.tile(y.size(1)):
            same = hl.load(y, [tile_m, tile_n]) + y[tile_m, tile_n]
            out[tile_m, tile_n] = x[tile_m, tile_n] - acc[:, None] + same
    return out


@helion.kernel(autotune_effort="none")
def _reads_loads_in_two_branches(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(y)
    for tile_m, tile_n in hl.tile(y.size()):
        if tile_m.begin == 0:
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
        else:
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def _reads_nested_reloads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x[tile] again in a branch inside its first load's block, and inside an
    # inner loop: one form, as the inner loads reuse the outer one.
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        a = x[tile]
        if tile.begin == 0:
            a = a + x[tile]
        for _ in hl.tile(y.size(1)):
            a = a + x[tile]
        out[tile] = a
    return out


@helion.kernel(autotune_effort="none")
def _reads_unrolled_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x[tile_m, i] is a load per unrolled iteration, each at another column;
    # y[tile_m, 0] is one load whatever the iteration.
    out = x.new_zeros([x.size(0)])
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for i in hl.static_range(1, 3):
            acc = acc + x[tile_m, i] + y[tile_m, 0]
        out[tile_m] = acc
    return out


@helion.kernel(autotune_effort="none")
def _reads_tuple_loop_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A loop over a tuple unrolls by repeating its body: x[tile_m, c] is a
    # load per element.
    out = x.new_zeros([x.size(0)])
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for c in (1, 2):
            acc = acc + x[tile_m, c]
        out[tile_m] = acc
    return out


@helion.kernel(autotune_effort="none")
def _reads_named_tuple_loop_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A loop over a tuple named by a variable unrolls in the device IR, past
    # the AST the analysis reads: x[tile_m, c] is still a load per element.
    out = x.new_zeros([x.size(0)])
    cols = (1, 2)
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for c in cols:
            acc = acc + x[tile_m, c]
        out[tile_m] = acc
    return out


@helion.kernel(autotune_effort="none")
def _reads_comprehension_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # A comprehension unrolls in the device IR too.
    out = x.new_zeros([x.size(0)])
    for tile_m in hl.tile(x.size(0)):
        parts = [x[tile_m, c] for c in (1, 2)]
        out[tile_m] = parts[0] + parts[1]
    return out


@helion.kernel(autotune_effort="none")
def _reads_derived_unrolled_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # An index computed from an hl.static_range variable: a load per
    # iteration as well.
    out = x.new_zeros([x.size(0)])
    for tile_m in hl.tile(x.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for i in hl.static_range(1, 3):
            j = i + 1
            acc = acc + x[tile_m, j]
        out[tile_m] = acc
    return out


@helion.kernel(autotune_effort="none")
def _reads_masked_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(y)
    for tile_m, tile_n in hl.tile(y.size()):
        mask = (tile_n.index % 2 == 0)[None, :]
        even = hl.load(y, [tile_m, tile_n], extra_mask=mask)
        mask = (tile_n.index % 2 == 1)[None, :]
        odd = hl.load(y, [tile_m, tile_n], extra_mask=mask)
        out[tile_m, tile_n] = even + odd + hl.load(x, [tile_m, tile_n], extra_mask=mask)
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def _reads_dynamic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = x.new_empty(x.shape)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


# Kernels on one axis, each a distinct way the generated code reads a
# parameter; each binds on CPU in well under a second.
@pytest.mark.parametrize(
    "kernel,expected",
    [
        (
            _reads_identity_and_device_metadata,
            {"x": {"referenced", "device"}, "y": {"size"}},
        ),
        (
            _reads_layout_and_grid_bounds,
            {"x": {"device"}, "y": {"size", "stride", "dtype", "device"}},
        ),
        (_reads_values, {"x": {"values", "device", "referenced"}}),
        (
            _reads_identity_and_no_load,
            {
                "x": {"values", "device", "referenced"},
                "y": {"values", "size", "stride", "dtype", "device"},
            },
        ),
        (
            _reads_types,
            {"x": {"size", "stride", "dtype", "device", "referenced"}, "y": {"values"}},
        ),
        (
            _reads_loads_in_two_loops,
            {"x": {"values", "device"}, "y": {"size", "stride", "dtype", "device"}},
        ),
        (
            _reads_loads_in_two_branches,
            {"x": {"values", "device"}, "y": {"size", "stride", "dtype", "device"}},
        ),
        (
            _reads_nested_reloads,
            {"x": {"size", "stride", "dtype", "device"}, "y": {"referenced"}},
        ),
        (
            _reads_unrolled_loads,
            {"x": {"values", "size", "dtype", "device"}},
        ),
        (
            _reads_tuple_loop_loads,
            {"x": {"values", "size", "dtype", "device"}, "y": {"referenced"}},
        ),
        (
            _reads_derived_unrolled_loads,
            {"x": {"values", "size", "dtype", "device"}, "y": {"referenced"}},
        ),
        (
            _reads_named_tuple_loop_loads,
            {"x": {"values", "size", "dtype", "device"}, "y": {"referenced"}},
        ),
        (
            _reads_comprehension_loads,
            {"x": {"values", "size", "dtype", "device"}, "y": {"referenced"}},
        ),
        (
            _reads_masked_loads,
            {"x": {"device"}, "y": {"values", "size", "stride", "dtype", "device"}},
        ),
        (
            _reads_dynamic,
            {"x": {"size", "stride", "dtype", "device"}, "y": {"size", "stride"}},
        ),
    ],
)
@skipIfRefEager("reads compiler state, which ref mode does not build")
def test_reads_besides_loads(kernel, expected):
    """What a fused prologue's stand-in must match: device loads, device-code
    dtypes and static grid bounds read nothing of it; ``x is None`` and a
    missing device load only need it in the call, as does
    ``isinstance(x, torch.Tensor)``; ``x is y``, other type checks and loads in
    two forms (in two loops or branches, with masks, or at each iteration of
    an ``hl.static_range``: ``hl.load(y, [i, j])`` and ``y[i, j]`` in one block
    are one form) read its value; code generation
    may read the device of the first tensor."""
    if not _compat.supports_torch_compile_template_lowering():
        pytest.skip("this PyTorch build lacks Helion's template lowering")
    from helion._compiler._inductor.template_buffer import _reads_besides_loads

    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.randn(8, 8), torch.randn(8, 8)))
    reads = _reads_besides_loads(bound, ("x", "y"))
    assert {name: set(fields) for name, fields in reads.items()} == expected


def test_prologue_stand_in_order_ignores_the_set_order():
    """The source passed for a fused parameter is the first of
    ``_stand_ins_first``: one with the layout the generated code reads, then
    by name, whatever order Inductor's frozenset of sources iterates in."""
    if not _compat.supports_torch_compile_template_lowering():
        pytest.skip("this PyTorch build lacks Helion's template lowering")
    from helion._compiler._inductor.template_buffer import _LAYOUT_FIELDS
    from helion._compiler._inductor.template_buffer import _stand_ins_first

    def buffer(size: list[int], stride: list[int]) -> ir.Buffer:
        layout = ir.FixedLayout(torch.device("cpu"), torch.float32, size, stride)
        return ir.Buffer(name=None, layout=layout)

    # ``flip(x) + b`` into a parameter laid out like x: the stand-in x sorts
    # after the bias by name.
    buffers = {"arg_b": buffer([64], [1]), "buf_x": buffer([64, 64], [64, 1])}
    producer = buffer([64, 64], [64, 1])
    with V.set_graph_handler(SimpleNamespace(try_get_buffer=buffers.get)):
        for sources in (("arg_b", "buf_x"), ("buf_x", "arg_b")):
            assert _stand_ins_first(sources, producer, _LAYOUT_FIELDS) == (
                "buf_x",
                "arg_b",
            )
            # Nothing read: every source qualifies; the whole layout still
            # comes first.
            assert _stand_ins_first(sources, producer, ()) == ("buf_x", "arg_b")


def test_a_matching_flexible_source_is_frozen():
    """An intermediate buffer's layout is decided after lowering, where the
    stand-in check runs: one that matches as it stands can stand in and is
    frozen, so it stays a match; one that does not match stays flexible."""
    if not _compat.supports_torch_compile_template_lowering():
        pytest.skip("this PyTorch build lacks Helion's template lowering")
    from helion._compiler._inductor.template_buffer import _LAYOUT_FIELDS
    from helion._compiler._inductor.template_buffer import _VALUES
    from helion._compiler._inductor.template_buffer import _ensure_stand_in

    cpu = torch.device("cpu")
    laid_out = ir.Buffer(
        name=None, layout=ir.FixedLayout(cpu, torch.float32, [64, 64], [64, 1])
    )

    def producer(*sources: str) -> SimpleNamespace:
        reads = {MemoryDep(name, sympy.Integer(0), (), ()) for name in sources}
        return SimpleNamespace(
            get_read_writes=lambda: SimpleNamespace(reads=reads),
            get_size=laid_out.get_size,
            get_stride=laid_out.get_stride,
            get_dtype=laid_out.get_dtype,
            get_device=laid_out.get_device,
        )

    def flexible(size: list[int]) -> ir.Buffer:
        return ir.Buffer(name=None, layout=ir.FlexibleLayout(cpu, torch.float32, size))

    buffers = {"buf_bias": flexible([64]), "buf_r": flexible([64, 64])}
    with V.set_graph_handler(SimpleNamespace(try_get_buffer=buffers.get)):
        assert not _ensure_stand_in(producer("buf_bias"), _LAYOUT_FIELDS)
        assert not _ensure_stand_in(producer("buf_r"), {*_LAYOUT_FIELDS, _VALUES})
        assert all(
            isinstance(buf.get_layout(), ir.FlexibleLayout) for buf in buffers.values()
        )
        assert _ensure_stand_in(producer("buf_bias", "buf_r"), _LAYOUT_FIELDS)
    assert isinstance(buffers["buf_r"].get_layout(), ir.FixedLayout)
    assert isinstance(buffers["buf_bias"].get_layout(), ir.FlexibleLayout)
