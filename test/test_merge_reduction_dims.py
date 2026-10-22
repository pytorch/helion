from __future__ import annotations

import copy
import pickle

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compiler.device_ir import _get_custom_decomp_table
from helion._compiler.merge_reduction_dims import _META_KEY


def test_reduction_metadata_survives_copy_and_pickle() -> None:
    op = torch.ops.aten.sum.dim_IntList
    x = torch.randn(2, 3)
    graph = make_fx(
        lambda value: op(value, [0, 1]),
        decomposition_table=_get_custom_decomp_table(),
    )(x)
    tags = [
        node.meta[_META_KEY] for node in graph.graph.nodes if _META_KEY in node.meta
    ]
    assert tags
    for tag in tags:
        assert pickle.loads(pickle.dumps(tag)) == tag
    cloned = copy.deepcopy(graph)
    torch.testing.assert_close(cloned(x), op(x, [0, 1]))
