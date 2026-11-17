"""Normal two-phase source with dedicated split and seeded page controllers."""

from __future__ import annotations

import json

import pytest
import torch

from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_compile import compile_conditional_entries
from helion._testing.cute_conditional_phase_binding import phase_fixture
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def composition(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("conditional-composition")
    initialized = torch.cuda.is_initialized()
    inputs = phase_fixture(batch=320, buckets=148)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(written_conditional_attention, inputs)
        assert len(bound.host_function.device_ir.graphs) == 16
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
    (tmp_path / "generated.py").write_text(source)
    (tmp_path / "config.json").write_text(json.dumps(config.config, indent=2) + "\n")
    assert torch.cuda.is_initialized() == initialized
    return tmp_path, inputs, source


def test_complete_conditional_attention_composition(composition):
    tmp_path, inputs, source = composition
    stages = stage_sources(source)
    assert len(stages) == 2
    for index, stage in enumerate(stages):
        (tmp_path / f"stage-{index}.py").write_text(stage)
    assert "_conditional_attention_page(" in stages[0]
    assert "_conditional_attention_page(" in stages[1]
    assert "fragment_arena" not in stages[1]


@pytest.mark.slow
def test_conditional_attention_composition_compiles(composition):
    tmp_path, inputs, source = composition
    initialized = torch.cuda.is_initialized()
    modules = compile_conditional_entries(
        tmp_path / "generated.py", inputs, complete=True
    )
    assert len(modules) == 2
    for index, module in enumerate(modules):
        (tmp_path / f"stage-{index}.mlir").write_text(module)
    assert torch.cuda.is_initialized() == initialized
