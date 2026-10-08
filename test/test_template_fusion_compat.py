from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch._inductor import ir
from torch._inductor import select_algorithm
from torch.utils._ordered_set import OrderedSet

from helion import _compat


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
