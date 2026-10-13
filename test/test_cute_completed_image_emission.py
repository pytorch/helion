from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_completed_image import _fixture

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.chained_completed_image_emission import (
    CompletedImageEmission,
)
from helion._compiler.cute.chained_completed_image_emission import image_shared_bytes
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_root_warp_stage import WarpOperandLayout


def _context(width=8):
    plan, *_ = _fixture(width)
    plan = dataclasses.replace(plan, completed_image_bytes=image_shared_bytes(plan))
    counter = itertools.count()
    cg = SimpleNamespace(
        device_function=SimpleNamespace(
            config=SimpleNamespace(config={}),
            new_var=lambda name: f"{name}_{next(counter)}",
        )
    )
    emitter = CompletedImageEmission(cg, plan)
    lines = ["chain_thread = cute.arch.thread_idx()[0]"]
    backend = SimpleNamespace(
        dtype_str=lambda dtype: (
            "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
        )
    )
    return (
        emitter,
        lines,
        patch.object(
            CompileEnvironment, "current", return_value=SimpleNamespace(backend=backend)
        ),
    )


def _stage(emitter, lines, boundaries, stage):
    emitter.before_stage(stage, boundaries, lines, ())
    lines.append("cute.arch.sync_threads()")
    emitter.joined_inputs(stage, lines)
    owner = f"chain_{stage}_c"
    publication = (
        f"chain_{stage}_sc = chain_{stage}_thr.partition_C({owner})",
        f"cute.autovec_copy(chain_{stage}_acc, chain_{stage}_sc)",
    )
    lines.extend([*publication, "cute.arch.sync_threads()"])
    node = emitter.plan.dots[stage]
    boundaries[node] = owner
    emitter.complete_dot(stage, node, owner, lines, publication)


@pytest.mark.parametrize("width", [8, 16])
def test_joined_publication_direct_views_and_exports(width):
    emitter, lines, context = _context(width)
    boundaries = {}
    with context:
        emitter.allocate(lines)
        _stage(emitter, lines, boundaries, 0)
        _stage(emitter, lines, boundaries, 1)
        emitter.before_stage(2, boundaries, lines, ())
        image = emitter.images[0]
        for view in image.operands:
            role = "b" if view.operand_index else "a"
            inner = 0 if view.transpose else 1
            layout = WarpOperandLayout(
                view.operand, role, view.physical_shape, (1, 1), inner, image.dtype
            )
            result = emitter.operand(2, layout, [])
            assert result and view.layout(image.shape[1]) in result[1]
            lines.extend(result)
        lines.append("cute.arch.sync_threads()")
        emitter.joined_inputs(2, lines)
        emitter.bind_exports(boundaries, lines)
    assert boundaries[image.node] == emitter.names[0]
    assert len(emitter.published) == len(emitter.visible) == 2
    assert sum("sync_threads" in line for line in lines) == 5
    # Retain the physical FP32 C source and crop to the logical first half.
    assert any(
        "if chain_completed_publish_" in line and f"< {width * width}:" in line
        for line in lines
    )
    assert not any("alloc_smem(cutlass.Float32" in line for line in lines)
    with pytest.raises(_UnsupportedChain):
        emitter.bind_exports(boundaries, lines)


@pytest.mark.parametrize("mutation", ["prefix", "config", "plan", "graph", "alias"])
def test_owner_context_mutations_decline(mutation):
    emitter, lines, context = _context()
    with context:
        emitter.allocate(lines)
    if mutation == "prefix":
        lines[0] = "tampered = 1"
    elif mutation == "config":
        emitter.cg.device_function.config.config["different"] = True
    elif mutation == "plan":
        object.__setattr__(emitter.plan, "threads", 64)
    elif mutation == "graph":
        emitter.images[0].node.args = (emitter.plan.dots[0], torch.bfloat16)
    else:
        emitter.aliases["existing"] = "old"
        emitter.plan.tensor_aliases["existing"] = "new"
    with pytest.raises(_UnsupportedChain):
        emitter.before_stage(0, {}, lines, ())


@pytest.mark.parametrize(
    "mutation",
    ["missing_join", "out_of_order", "publication", "owner_rebind", "early_export"],
)
def test_completion_sequence_rejects_invalid_evidence(mutation):
    emitter, lines, context = _context()
    boundaries = {}
    with context:
        emitter.allocate(lines)
        if mutation == "out_of_order":
            with pytest.raises(_UnsupportedChain):
                emitter.before_stage(1, boundaries, lines, ())
            return
        emitter.before_stage(0, boundaries, lines, ())
        if mutation == "missing_join":
            with pytest.raises(_UnsupportedChain):
                emitter.joined_inputs(0, lines)
            return
        lines.append("cute.arch.sync_threads()")
        emitter.joined_inputs(0, lines)
        if mutation == "early_export":
            with pytest.raises(_UnsupportedChain):
                emitter.bind_exports(boundaries, lines)
            return
        node = emitter.plan.dots[0]
        owner = "chain_0_c"
        publication = (
            "chain_0_sc = chain_0_thr.partition_C(chain_0_c)",
            "cute.autovec_copy(chain_0_acc, chain_0_sc)",
        )
        lines.extend([*publication, "cute.arch.sync_threads()"])
        if mutation == "publication":
            with pytest.raises(_UnsupportedChain):
                emitter.complete_dot(0, node, owner, lines, publication[:1])
        else:
            emitter.complete_dot(0, node, owner, lines, publication)
            boundaries[node] = "rebound"
            with pytest.raises(_UnsupportedChain):
                emitter.before_stage(1, boundaries, lines, ())


@pytest.mark.parametrize("mutation", ["domain", "axis", "dtype", "shape"])
def test_original_predicates_and_native_layout_not_silently_removed(mutation):
    emitter, lines, context = _context()
    boundaries = {}
    with context:
        emitter.allocate(lines)
        _stage(emitter, lines, boundaries, 0)
        _stage(emitter, lines, boundaries, 1)
        emitter.before_stage(2, boundaries, lines, ())
    image = emitter.images[0]
    view = image.operands[0]
    layout = WarpOperandLayout(
        view.operand, "a", view.physical_shape, (8, 1), 1, image.dtype
    )
    domain = []
    if mutation == "domain":
        domain = ["original_tail_predicate"]
    elif mutation == "axis":
        layout = dataclasses.replace(layout, inner=0)
    elif mutation == "dtype":
        layout = dataclasses.replace(layout, dtype=torch.float16)
    else:
        layout = dataclasses.replace(layout, shape=(32, 8))
    assert emitter.operand(2, layout, domain) is None
    assert len(emitter.rejected) == 1 and not emitter.used


def test_last_stage_only_image_declines_without_inventing_join():
    plan, *_ = _fixture()
    # Remove the following consumer: one publication would occur after the last
    # original operand join, so the image cannot be exported on this schedule.
    plan = dataclasses.replace(plan, dots=plan.dots[:2], shapes=plan.shapes[:2])
    assert image_shared_bytes(plan) == 0
