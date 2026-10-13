from __future__ import annotations

import ast
import copy
from dataclasses import replace
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_async_groups import _check_masked_frames
from .test_cute_chained_async_groups import _config
from .test_cute_chained_async_groups import _inputs
from .test_cute_chained_async_groups import _many_raw_images
from .test_cute_chained_async_groups import _source
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_rectangular_frontier as rectangle
from helion._compiler.cute.chained_async_groups import EnqueuedAsyncCopy

_KEY = "cute_chained_rectangular_async"


def _capture(
    *,
    args: tuple[Any, ...] | None = None,
    rendered: dict[str, list[str]] | None = None,
    **options: object,
) -> tuple[str, list[rectangle.RectangularFrontier]]:
    captured = []
    bind = rectangle.bind_rectangular_frontier

    def observe(*args: Any, **kwargs: Any) -> rectangle.RectangularFrontier | None:
        result = bind(*args, **kwargs)
        if result is not None:
            assert "partition_D(" + result.sink.target + ")" in "\n".join(
                result.lines()
            )
            if rendered is not None:
                rendered[result.prefix] = result.lines()
            captured.append(result)
        return result

    with patch.object(rectangle, "bind_rectangular_frontier", observe):
        source = _source(_config(8, **({_KEY: True} | options)), args)
    return source, captured


def test_complete_rectangles_keep_original_fallback_and_sync() -> None:
    with patch.object(rectangle, "bind_rectangular_frontier", return_value=None):
        before = _source(_config(8, **{_KEY: True}))
    after, captured = _capture()
    assert captured
    for item in captured:
        assert item.proof.dtype == item.sink.dtype
        assert item.fallback
        assert item.proof.pitch % 8 == 0
        assert item.ownership.matches(item.sink.shape, item.execution.threads)
    for marker in (
        "cp_async_commit_group()",
        "cp_async_wait_group(0)",
        "arrive_and_wait()",
        "arrive_mbarrier(",
        "mbarrier_wait(",
        "cute.gemm(",
    ):
        assert before.count(marker) == after.count(marker)
    ast.parse(after)

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id.endswith(
                "_rectangle_copied"
            ):
                return None
            return node

        def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt] | None:
            if (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id.endswith("_rectangle_copied")
            ):
                return node.body
            if any(
                isinstance(child, ast.Name) and child.id.endswith("_rectangle_copied")
                for child in ast.walk(node)
            ) and any(
                isinstance(child, ast.Name) and child.id.endswith("_rectangle_pointer")
                for statement in node.body[:1]
                for child in ast.walk(statement)
            ):
                return None
            return self.generic_visit(node)

    class Alpha(ast.NodeTransformer):
        def __init__(self) -> None:
            self.names: dict[str, str] = {}

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if re.fullmatch(r"(?:chain_value|v)_\d+", node.id):
                node.id = self.names.setdefault(
                    node.id, f"chain_value_{len(self.names)}"
                )
            return node

    # Proof construction consumes fresh temporary names but changes no
    # original expression. Compare every remaining statement after renaming.
    original = ast.unparse(Alpha().visit(ast.parse(before))).splitlines()
    restored = ast.unparse(
        Alpha().visit(Restore().visit(ast.parse(after)))
    ).splitlines()
    for left, right in zip(original, restored, strict=True):
        assert left == right


def test_default_packet_path_unchanged() -> None:
    before = _source(_config(8))
    with patch.object(
        rectangle, "bind_rectangular_frontier", side_effect=AssertionError
    ):
        assert _source(_config(8)) == before
    assert "_rectangle_copied" not in before


@pytest.mark.parametrize(
    "fault", ["pitch", "shape", "dtype", "role", "fallback", "ownership", "sink"]
)
def test_changed_rectangle_evidence_rejects(fault: str) -> None:
    _, captured = _capture()
    item = captured[0]
    if fault == "pitch":
        item = replace(item, proof=replace(item.proof, pitch=item.proof.pitch + 8))
    elif fault == "shape":
        item = replace(item, proof=replace(item.proof, tile_shape=(1, 8)))
    elif fault == "dtype":
        item = replace(item, proof=replace(item.proof, dtype="cutlass.Float32"))
    elif fault == "role":
        item = replace(item, execution=replace(item.execution, threads=32))
    elif fault == "ownership":
        item = replace(
            item, ownership=replace(item.ownership, thread_order="column_major")
        )
    elif fault == "sink":
        item = replace(item, sink=replace(item.sink, target="foreign_image"))
    else:
        item = replace(item, fallback=("foreign_write()",))
    with pytest.raises(
        chain._UnsupportedChain, match="rectangular frontier evidence changed"
    ):
        item.check()


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("field", ["args", "target", "meta"])
def test_changed_original_load_facts_reject(host: bool, field: str) -> None:
    _, captured = _capture()
    item = captured[0]
    node = item.source if host else item.node
    value = {"args": (), "target": torch.neg, "meta": {}}[field]
    original = getattr(node, field)
    try:
        setattr(node, field, value)
        with pytest.raises(chain._UnsupportedChain):
            item.check()
    finally:
        setattr(node, field, original)


def test_changed_index_expression_rejects() -> None:
    _, captured = _capture()
    item = captured[0]
    index = next(node for node in item.node.all_input_nodes if node is not item.source)
    with (
        patch.object(index, "target", torch.neg),
        pytest.raises(chain._UnsupportedChain),
    ):
        item.check()


def test_async_receipt_keeps_complete_rectangle_evidence() -> None:
    captured = []
    matches = EnqueuedAsyncCopy.matches

    def observe(item: EnqueuedAsyncCopy) -> bool:
        if item.rectangle is not None:
            item.rectangle.check()
            assert tuple(item.rectangle.lines()) == item.body
            captured.append(item)
        return matches(item)

    with patch.object(EnqueuedAsyncCopy, "matches", observe):
        _capture()
    assert captured


def test_rectangle_default_normalization_and_seed_preserve_packet_pool() -> None:
    with _cpu_codegen():
        bound = _many_raw_images._bind_isolated(_inputs())
        spec = bound.config_spec
        assert spec._flat_fields()[_KEY].search_values() == [False, True]
        packet = spec.normalized_config(_config(8))
        assert spec.normalized_config(packet.config | {_KEY: False}) == packet
        generation = spec.create_config_generation()
        rectangle_config = spec.normalized_config(packet.config | {_KEY: True})
        assert (
            generation.unflatten(generation.flatten(rectangle_config))
            == rectangle_config
        )
        with bound.env:
            make = CuteChainedMatmulHeuristic._with_rectangular_async_seed
            parents = [_config(1), _config(8)]
            original = copy.deepcopy(parents)
            seeds = make(bound.env, parents)
            assert seeds[:-1] == original == parents
            assert seeds[-1].config == parents[-1].config | {_KEY: True}
            assert make(bound.env, seeds) == seeds
            with patch.dict(bound.settings.autotune_config_overrides, {_KEY: False}):
                assert make(bound.env, parents) is parents


@pytest.mark.parametrize("value", [1, 0, None, "True"])
def test_non_bool_rectangle_choice_rejects(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        bound = _many_raw_images._bind_isolated(_inputs())
        bound.config_spec.normalized_config(_config(8, **{_KEY: value}))


@pytest.mark.parametrize(
    "missing", ["cute_chained_preparation_pipeline", "cute_chained_async_vector_store"]
)
def test_rectangle_requires_original_async_preparation(missing: str) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig):
        bound = _many_raw_images._bind_isolated(_inputs())
        bound.config_spec.normalized_config(_config(1, **{_KEY: True, missing: False}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("misaligned", [False, True])
def test_complete_and_masked_rectangles_preserve_changing_state_gpu(
    dtype: torch.dtype, misaligned: bool
) -> None:
    _check_masked_frames(
        dtype,
        misaligned,
        (_config(8), _config(8, **{_KEY: True, "cute_chained_scratch_layout": "xor"})),
    )


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_rectangle_cannot_bypass_final_storage(fault: str) -> None:
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def broken(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return finalize(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", broken),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _capture()
