from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_completed_image import _fixture

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.chained_completed_image import discover_completed_images
from helion._compiler.cute.chained_completed_image_emission import (
    CompletedImageEmission,
)
from helion._compiler.cute.chained_completed_image_emission import image_shared_bytes
from helion._compiler.cute.chained_direct_image import DirectImageEdit
from helion._compiler.cute.chained_direct_image import _apply
from helion._compiler.cute.chained_direct_image import _read_cut
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_root_warp_stage import WarpOperandLayout

if TYPE_CHECKING:
    from helion._compiler.generate_ast import GenerateAST


@pytest.mark.parametrize("width", [8, 16])
def test_exclusive_cut_allows_only_original_image_and_native_readers(width):
    plan, *_ = _fixture(width)
    (image,) = discover_completed_images(plan)
    assert _read_cut(image, plan)


@pytest.mark.parametrize(
    "kind", ["fp32_export", "arithmetic", "second_cast", "unknown"]
)
def test_retained_fp32_reader_prevents_retirement(kind):
    plan, source, _, _, _, _ = _fixture(16)
    (image,) = discover_completed_images(plan)
    graph = source.graph
    if kind == "fp32_export":
        graph.output(source)
    else:
        target, args = {
            "arithmetic": (torch.ops.aten.neg.default, (source,)),
            "second_cast": (
                torch.ops.prims.convert_element_type.default,
                (source, torch.float32),
            ),
            "unknown": (torch.ops.aten.alias.default, (source,)),
        }[kind]
        user = graph.call_function(target, args)
        user.meta["val"] = torch.empty_like(source.meta["val"])
        graph.output(user)
    assert not _read_cut(image, plan)


def _complete(domain=False, mutate=None):
    plan, *_ = _fixture(16)
    plan = dataclasses.replace(plan, completed_image_bytes=image_shared_bytes(plan))
    counter = itertools.count()
    cg = SimpleNamespace(
        cute_matmul_layouts=None,
        device_function=SimpleNamespace(
            config=SimpleNamespace(config={"cute_chained_operand_coordinates": True}),
            new_var=lambda name: f"{name}_{next(counter)}",
        ),
    )
    object.__setattr__(
        plan.region,
        "contractions",
        tuple(
            SimpleNamespace(operand_dtypes=(plan.dtype, plan.dtype)) for _ in plan.dots
        ),
    )
    emitter = CompletedImageEmission(cast("GenerateAST", cg), plan)
    assert len(emitter.direct.owners) == 2
    lines = ["chain_thread = cute.arch.thread_idx()[0]"]
    boundaries = {}
    backend = SimpleNamespace(dtype_str=lambda dtype: "cutlass.BFloat16")
    with patch.object(
        CompileEnvironment, "current", return_value=SimpleNamespace(backend=backend)
    ):
        emitter.allocate(lines)
        for stage, node in enumerate(plan.dots):
            emitter.before_stage(stage, boundaries, lines, ())
            start = len(lines)
            for image in emitter.images:
                for view in image.operands:
                    if view.stage != stage:
                        continue
                    layout = WarpOperandLayout(
                        view.operand,
                        "b" if view.operand_index else "a",
                        view.physical_shape,
                        (1, 1),
                        0 if view.transpose else 1,
                        image.dtype,
                    )
                    operand = emitter.operand(
                        stage, layout, ["original_tail"] if domain else []
                    )
                    if operand is not None:
                        lines.extend(operand)
            lines.append("cute.arch.sync_threads()")
            emitter.joined_inputs(stage, lines)
            rows, columns, _ = plan.shapes[stage]
            name = f"chain_{stage}_c"
            lines.extend(
                [
                    f"{name}_ptr = cute.arch.alloc_smem(cutlass.Float32, {rows * (columns + 4)}, alignment=128)",
                    f"{name} = cute.make_tensor({name}_ptr, cute.make_layout({(rows, columns)!r}, stride=({columns + 4}, 1)))",
                ]
            )
            publication = (
                f"chain_{stage}_sc = chain_{stage}_thr.partition_C({name})",
                f"cute.autovec_copy(chain_{stage}_acc, chain_{stage}_sc)",
            )
            lines.extend([*publication, "cute.arch.sync_threads()"])
            boundaries[node] = name
            emitter.complete_dot(stage, node, name, lines, publication)
            emitter.record_stage(node, lines[start:])
        if domain:
            # An unavailable native image read retains the additive body. The
            # ordinary export binder still requires at least one native reader.
            from helion._compiler.cute.chained_direct_image import lower_direct_images

            original = tuple(lines)
            assert lower_direct_images(emitter, lines) is None
            assert tuple(lines) == original
            return
        emitter.bind_exports(boundaries, lines)
        lines.append(
            "original_export = chain_completed_image_0[original_row % 32, original_column]"
        )
        original = tuple(line for block in lines for line in block.splitlines())
        if mutate == "prefix":
            lines[0] = "foreign_owner = 1"
        elif mutate == "selection":
            object.__setattr__(emitter.direct.owners[0].publication, "extent", (1, 1))
        elif mutate == "native":
            emitter.native_stages.pop()
        if mutate:
            with pytest.raises(_UnsupportedChain):
                emitter.retire_owners(lines)
            return
        result = emitter.retire_owners(lines)
    assert emitter.lowering is not None
    assert emitter.lowering.original == original
    assert len([line for line in original if "alloc_smem(cutlass.Float32" in line]) == 3
    assert len([line for line in result if "alloc_smem(cutlass.Float32" in line]) == 1
    assert sum("sync_threads()" in line for line in result) == sum(
        "sync_threads()" in line for line in original
    )
    assert result[-1] == original[-1]
    assert any("0 <= chain_0_image_row < 16" in line for line in result)
    assert any(
        "cutlass.BFloat16(cutlass.Float32(chain_0_acc[" in line for line in result
    )
    # Uniform enclosing indentation preserves the original relative scope.
    assert _apply(
        ("    if row < 8:", "        a = Float32(x)"),
        (
            DirectImageEdit(
                ("if row < 8:", "    a = Float32(x)"),
                ("if row < 8:", "    a = BFloat16(Float32(x))"),
            ),
        ),
        complete=True,
    ) == ("    if row < 8:", "        a = BFloat16(Float32(x))")


def test_complete_native_cut_retires_storage_and_retains_guards():
    _complete()


def test_original_nonempty_domain_retains_additive_body():
    _complete(domain=True)


@pytest.mark.parametrize("mutation", ["prefix", "selection", "native"])
def test_changed_owner_or_publication_receipt_rejects(mutation):
    _complete(mutate=mutation)


@pytest.mark.parametrize(
    "source",
    [
        ("if row < 7:", "    a = Float32(x)"),
        ("if row < 8:", "    a = Float32(y)"),
        ("if row < 8:", "a = Float32(x)"),
        ("if row < 8:", "    a = Float32(x)", "if row < 8:", "    a = Float32(x)"),
    ],
)
def test_span_guard_cannot_hide_changed_mask_value_scope_or_duplicate(source):
    edit = DirectImageEdit(("if row < 8:", "    a = Float32(x)"), ("replacement",))
    with pytest.raises(_UnsupportedChain):
        _apply(source, (edit,), complete=True)
