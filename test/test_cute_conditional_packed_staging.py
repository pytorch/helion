"""Packed SS staging admission and complete byte/layout correspondence."""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.analysis.conditional_packed_staging import (
    can_pack_native_panel,
)
from helion._compiler.cute.conditional_attention_codegen import (
    conditional_attention_compiler,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_native import validate_conditional_source
from helion._testing.cute_conditional_packed import _launch_packed_panel
from helion._testing.cute_conditional_packed import packed_panel_inputs
from helion._testing.cute_conditional_packed import packed_panel_reference
from helion._testing.cute_conditional_packed import validate_packed_source
from helion._testing.cute_conditional_phase_binding import phase_fixture
from helion._testing.cute_paged_conditional_semantics import native_s2f6_decode
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.slow
@pytest.mark.parametrize("byte_offset", [0, 1])
def test_packed_staging_actual_bound_pointer_and_fallback(byte_offset, tmp_path):
    initialized = torch.cuda.is_initialized()
    inputs = list(phase_fixture())
    if byte_offset:
        for data_index, scale_index in ((1, 3), (2, 4)):
            data, scales = inputs[data_index], inputs[scale_index]
            storage = data.as_strided((data.untyped_storage().nbytes(),), (1,), 0)
            shifted = torch.empty(storage.numel() + byte_offset, dtype=torch.int8)
            shifted[byte_offset:].copy_(storage)
            inputs[data_index] = shifted.as_strided(
                data.shape, data.stride(), data.storage_offset() + byte_offset
            )
            inputs[scale_index] = shifted.view(torch.uint8).as_strided(
                scales.shape, scales.stride(), scales.storage_offset() + byte_offset
            )
            torch.testing.assert_close(inputs[data_index], data, rtol=0, atol=0)
            torch.testing.assert_close(inputs[scale_index], scales, rtol=0, atol=0)
    observed = []

    def observe(cg, graphs):
        compiler = conditional_attention_compiler(cg, graphs)
        assert compiler is not None
        for panel in (compiler.program.accesses.key, compiler.program.accesses.value):
            assert can_pack_native_panel(panel, compiler.env) == (byte_offset == 0)
            assert panel.shifted_row.meta["val"].dtype == torch.int32
            if byte_offset == 0:
                # Each negative changes one required physical/semantic fact.
                for changed in (
                    replace(panel, raw=replace(panel.raw, dtype=torch.uint8)),
                    replace(panel, scale=replace(panel.scale, dtype=torch.int8)),
                    replace(panel, group_size=16),
                    replace(
                        panel, raw=replace(panel.raw, strides=(33793, 256, 128, 1))
                    ),
                    replace(
                        panel, raw=replace(panel.raw, strides=(33792, 256, 128, 2))
                    ),
                    replace(panel, scale=replace(panel.scale, shape=(7, 128, 2, 3))),
                ):
                    assert not can_pack_native_panel(changed, compiler.env)
        observed.append(compiler.packed_staging)
        return compiler

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(written_conditional_attention, tuple(inputs))
        with patch(
            "helion._compiler.cute.conditional_attention_codegen.conditional_attention_compiler",
            observe,
        ):
            source = bound.to_code(bound.config_spec.default_config())
    (tmp_path / "generated.py").write_text(source)
    validate_conditional_source(source)
    if byte_offset == 0:
        validate_packed_source(source)
        with pytest.raises(AssertionError):
            validate_packed_source(
                source.replace("_conditional_attention_stage_packed(", "wrong_staging(")
            )
    else:
        with pytest.raises(AssertionError):
            validate_packed_source(source)
    assert observed == [byte_offset == 0] * 2
    for stage in stage_sources(source):
        assert stage.count("_conditional_attention_stage_packed(") == (
            2 if byte_offset == 0 else 0
        )
        if byte_offset:
            assert "cutlass.range_constexpr(128)" in stage
    assert torch.cuda.is_initialized() == initialized


def test_packed_ss_full_address_map():
    """Execute actual index expressions and compare every byte to the scalar map."""
    path = (
        Path(__file__).parents[1]
        / "helion/_compiler/cute/_conditional_attention_staging.py"
    )
    function = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
    )
    expressions = {
        node.targets[0].id: compile(ast.Expression(node.value), str(path), "eval")
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("linear", "row", "channel", "offset")
    }
    owned = set()
    for iteration in range(64):
        for thread in range(128):
            state = {"iteration": iteration, "thread": thread}
            for name in ("linear", "row", "channel", "offset"):
                state[name] = eval(expressions[name], {"__builtins__": {}}, state)
            row, channel, offset = (state[k] for k in ("row", "channel", "offset"))
            assert 0 <= row < 128 and 0 <= channel < 127 and channel % 2 == 0
            assert channel // 32 == (channel + 1) // 32
            assert offset % 2 == 0  # BF16 offset implies a 4-byte-aligned store.
            for half in range(2):
                col = channel + half
                scalar = row * 64 + ((col % 64) ^ ((row % 8) * 8)) + 8192 * (col // 64)
                assert offset + half == scalar
                assert (row, col) not in owned
                owned.add((row, col))
    assert owned == {(row, col) for row in range(128) for col in range(128)}


def test_packed_packet_oracle_matches_authored_integer_lattice():
    first = None
    for phase in ("original", "changed", "restored"):
        backing, raw, scales, coordinates = packed_panel_inputs(phase)
        before = backing.clone()
        expected = packed_panel_reference(raw, scales, coordinates)
        for task, (page, head, shift) in enumerate(coordinates.tolist()):
            if not 0 <= page < 4 or not 0 <= head < 2:
                assert torch.equal(
                    expected[task].view(torch.int16),
                    torch.zeros((128, 128), dtype=torch.int16),
                )
                continue
            rows = (torch.arange(128, dtype=torch.int64) + shift).to(torch.int32)
            selected = rows.clamp(0, 127).long()
            actual = native_s2f6_decode(
                raw[page, selected, head],
                scales[page, selected, head].repeat_interleave(32, -1),
            )
            actual[(rows < 0) | (rows >= 128)] = 0
            assert torch.equal(
                actual.view(torch.int16), expected[task].view(torch.int16)
            )
        assert torch.equal(backing, before)
        if phase == "original":
            pair_ids = (
                raw[:, :, 0].view(torch.uint8).int()
                + 256 * scales[:, :, 0].repeat_interleave(32, -1).int()
            )
            assert pair_ids.unique().numel() == 65536
            first = expected.view(torch.int16).clone()
        if phase == "restored":
            assert torch.equal(expected.view(torch.int16), first)


@pytest.mark.slow
def test_packed_packet_observer_compiles_offline(tmp_path):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor

    cuda = importlib.import_module("cuda.bindings.driver")
    initialized = torch.cuda.is_initialized()
    _, raw, scales, coordinates = packed_panel_inputs("original")
    output = torch.empty((20, 128, 128), dtype=torch.bfloat16)
    args = tuple(
        make_fake_tensor(
            dtype, tuple(value.shape), tuple(value.stride()), assumed_align=16
        )
        for value, dtype in zip(
            (raw, scales, coordinates, output),
            (cutlass.Int8, cutlass.Uint8, cutlass.Int64, cutlass.BFloat16),
            strict=True,
        )
    )
    with cpu_target(forbid_native=False):
        compiled = cute.compile(
            _launch_packed_panel,
            *args,
            cuda.CUstream(0),
            no_jit_engine=True,
            no_cache=True,
        )
    (tmp_path / "observer.mlir").write_text(str(compiled.ir_module))
    assert torch.cuda.is_initialized() == initialized
