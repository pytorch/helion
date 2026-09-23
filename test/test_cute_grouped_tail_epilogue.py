from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from test import test_cute_rank3_rhs_b_tma as grouped

import helion
from helion._compiler.cute import cute_epilogue
from helion.language import memory_ops

pytestmark = grouped.pytestmark


def _capture_tail() -> tuple:
    grouped._require_cuda("grouped tail proof requires CUDA fake inputs")
    original = cute_epilogue.analyze_tcgen05_grouped_tail_epilogue
    captured = []

    def observe(
        *args: object, **kwargs: object
    ) -> cute_epilogue.Tcgen05GroupedTailEpilogueMatch | None:
        result = original(*args, **kwargs)
        if result is not None and result.store_mask is not None:
            captured.append((args, kwargs, result))
        return result

    kernel = helion.kernel(backend="cute")(
        grouped._rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.fn
    )
    with patch.object(cute_epilogue, "analyze_tcgen05_grouped_tail_epilogue", observe):
        code = grouped._code_for(
            kernel,
            grouped._make_documented_mixed_k_args(),
            grouped._dynamic_bk64_config(direct=True),
        )
    assert "StaticPersistentGroupTileScheduler" in code
    assert captured
    args, kwargs, proof = captured[-1]
    assert proof.has_m_tail_mask and proof.has_n_tail_mask
    return original, args, kwargs, proof


def test_actual_masked_tail_and_legacy_where_match() -> None:
    analyze, args, kwargs, proof = _capture_tail()
    assert analyze(*args, **kwargs) is not None
    store = proof.store_node
    value = args[0]
    old_args = store.args
    graph = store.graph
    with graph.inserting_before(store):
        old = graph.call_function(
            memory_ops.load, (store.args[0], store.args[1], None, None)
        )
        old.meta = dict(value.meta)
        where = graph.call_function(
            torch.ops.aten.where.self, (proof.store_mask, value, old)
        )
        where.meta = dict(value.meta)
    try:
        store.args = (old_args[0], old_args[1], where, None)
        legacy = analyze(where, **kwargs)
        assert legacy is not None and legacy.store_mask is None
        assert legacy.anchor is proof.anchor
        assert legacy.safe_group_node is proof.safe_group_node
        assert legacy.n_sizes_tensor is proof.n_sizes_tensor
        assert legacy.has_m_tail_mask and legacy.has_n_tail_mask
    finally:
        store.args = old_args
        graph.erase_node(where)
        graph.erase_node(old)


@pytest.mark.parametrize(
    "mutation",
    [
        "output_dtype",
        "value_dtype",
        "output_rank",
        "shifted_output_axis",
        "wrong_row_tensor",
        "wrong_column_comparison",
        "extra_mask_user",
        "extra_cast_user",
        "non_bool_mask",
        "missing_mask",
    ],
)
def test_actual_grouped_mask_rejections(mutation: str) -> None:
    analyze, args, kwargs, proof = _capture_tail()
    value = args[0]
    store = proof.store_node
    condition = proof.store_mask
    assert condition is not None
    graph = store.graph
    restore = []
    inserted = []

    def change(obj: torch.fx.Node, name: str, replacement: object) -> None:
        restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    try:
        if mutation in ("output_dtype", "output_rank", "value_dtype", "non_bool_mask"):
            node = (
                store.args[0]
                if mutation.startswith("output")
                else value
                if mutation == "value_dtype"
                else condition
            )
            tensor = node.meta["val"]
            with tensor.fake_mode:
                shape = (1,) if mutation == "output_rank" else tensor.shape
                altered = torch.empty(shape, dtype=torch.float32, device=tensor.device)
            change(node, "meta", {**node.meta, "val": altered})
        elif mutation == "shifted_output_axis":
            change(
                store,
                "args",
                (store.args[0], [store.args[1][1], store.args[1][0]], value, condition),
            )
        elif mutation == "wrong_row_tensor":
            row = next(
                n
                for n in proof.producer_nodes
                if n.target is memory_ops.load and n.meta["val"].ndim == 1
            )
            change(row, "args", (store.args[0], *row.args[1:]))
        elif mutation == "wrong_column_comparison":
            col = next(
                n for n in proof.producer_nodes if n.target is torch.ops.aten.lt.Tensor
            )
            change(col, "target", torch.ops.aten.le.Tensor)
        elif mutation in ("extra_mask_user", "extra_cast_user"):
            node = condition if mutation == "extra_mask_user" else value
            with graph.inserting_before(store):
                inserted.append(
                    graph.call_function(torch.ops.aten.clone.default, (node,))
                )
        elif mutation == "missing_mask":
            change(store, "args", (store.args[0], store.args[1], value, None))
        else:
            raise AssertionError(mutation)
        assert analyze(*args, **kwargs) is None
    finally:
        for obj, name, old in reversed(restore):
            setattr(obj, name, old)
        for node in inserted:
            graph.erase_node(node)
