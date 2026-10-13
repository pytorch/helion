from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_vector_expression as fixture
from .test_cute_chained_vector_leaf import _plan
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_vector_expression import emit_vector_expression
from helion._compiler.cute.chained_vector_leaf import DenseVectorSink
from helion._compiler.cute.chained_vector_leaf import emit_vector_leaf
from helion._compiler.cute.chained_vector_stage import VectorStaging


def _capture(dtype, enabled, *, mode="identity"):
    result = {}

    def observe(cg, plan, boundaries, node, **kwargs):
        row, base, element = "frontier_row", "frontier_base", "frontier_element"
        coords = kwargs["coordinates"](row, f"({base} + {element})")
        probe = chain._Expression(cg, plan, boundaries)
        probe.coordinate_names.update((row, base, element))
        probe.value(node, coords)
        assert probe.loaded_inputs
        selected = node if mode == "arithmetic" else probe.loaded_inputs[0][0]
        shape = chain._shape(selected)
        assert len(shape) == 2
        shape = (shape[0], shape[1])
        dtype_name = {
            torch.bfloat16: "cutlass.BFloat16",
            torch.float16: "cutlass.Float16",
            torch.float32: "cutlass.Float32",
        }[selected.meta["val"].dtype]
        sink = DenseVectorSink("prepared_image", shape, dtype_name)
        if mode == "wrong_dtype":
            sink = replace(sink, dtype="cutlass.Float64")
        elif mode == "wrong_shape":
            sink = replace(sink, shape=(shape[0] + 1, shape[1]))
        elif mode == "wrong_target":
            sink = replace(sink, target="another_image")
        kwargs.update(offset=0, shape=shape, shared_sink=sink)
        if mode == "final_mask":
            kwargs["final_value"] = lambda value, dtype, coords: f"{dtype}({value})"
        elif mode == "scalar_target":
            kwargs["vector_store"] = False
        elif mode == "boundary":
            boundaries = {**boundaries, selected: "already_materialized"}
        elif mode == "no_layout":
            kwargs["shared_sink"] = None
        staging = VectorStaging(True, async_enabled=enabled)
        lines = emit_vector_expression(
            cg, plan, boundaries, selected, async_staging=staging, **kwargs
        )
        result.update(lines=lines, activated=staging.async_activated)
        return lines

    with patch.object(fixture, "emit_vector_expression", observe):
        fixture._frontier_capture(host_dtype=dtype)
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_original_expression_client_admits_only_exact_identity_shared_sink(dtype):
    ordinary = _capture(dtype, False)
    selected = _capture(dtype, True)
    assert ordinary["lines"] is not None and selected["lines"] is not None
    assert selected["activated"] and not ordinary["activated"]
    text = "\n".join(selected["lines"])
    ast.parse(text)
    assert text.count("cpasync.CopyG2SOp()") == 1
    assert text.endswith(
        "cute.arch.cp_async_commit_group()\ncute.arch.cp_async_wait_group(0)"
    )
    assert text.count("if not frontier_leaf_0_vectorized:") == 2
    assert "frontier_leaf_0_shared_pointer.toint()) % 16 == 0" in text
    assert "frontier_leaf_0_last_address == frontier_leaf_0_address +" in text
    # The original expression, not a replacement scalar integer renderer,
    # still owns every failed guard and masked value.
    assert "cutlass.Int32" in text and "operator.lt" in text
    assert "cute.math.exp2" not in text
    assert "CopyG2SOp" not in "\n".join(ordinary["lines"])


@pytest.mark.parametrize(
    "mode",
    [
        "arithmetic",
        "wrong_dtype",
        "wrong_shape",
        "wrong_target",
        "final_mask",
        "scalar_target",
        "boundary",
        "no_layout",
    ],
)
def test_unsupported_sink_or_expression_preserves_the_original_source(mode):
    ordinary = _capture(torch.bfloat16, False, mode=mode)
    selected = _capture(torch.bfloat16, True, mode=mode)
    assert selected["lines"] == ordinary["lines"]
    assert not selected["activated"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_leaf_extension_keeps_all_original_source_guards_and_scalar_fallback(dtype):
    plan = _plan(dtype=dtype)
    assert plan is not None

    def emit(shared_pointer: str | None = None):
        return emit_vector_leaf(
            plan,
            tensor="tensor",
            prefix="leaf",
            pointer_for_indices=lambda ix: (
                f"tensor.iterator + cutlass.Int32({ix[0]}) * "
                f"cutlass.Int32(tensor.layout.stride[0]) + cutlass.Int32({ix[1]})"
            ),
            scalar_for_element=lambda element: (
                [f"position = base + {element}"],
                "original_masked_scalar(position)",
            ),
            shared_pointer=shared_pointer,
        )

    before = emit()
    after = emit("shared.iterator + destination")
    old, new = ast.parse("\n".join(before.lines)), ast.parse("\n".join(after.lines))
    assert ast.dump(old.body[-1]) == ast.dump(new.body[-1])
    old_bounds = next(n for n in old.body if isinstance(n, ast.If))
    new_bounds = next(n for n in new.body if isinstance(n, ast.If))
    assert ast.dump(old_bounds.test) == ast.dump(new_bounds.test)
    assert before.values == after.values and before.vectorized == after.vectorized
    assert "\n".join(before.lines) == "\n".join(emit(shared_pointer=None).lines)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_async_choice_is_a_strict_bool(value):
    with pytest.raises(TypeError, match="explicit bool"):
        VectorStaging(True, async_enabled=value)


def test_selected_but_unactivated_transport_rejects():
    with pytest.raises(exc.BackendUnsupported, match="identity shared sink"):
        VectorStaging(True, activated=True, async_enabled=True).validate()
