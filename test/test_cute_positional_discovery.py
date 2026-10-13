from __future__ import annotations

from unittest.mock import patch

import pytest
from torch.fx import Graph

from helion._compiler.cute.positional_root import has_positional_contraction_candidate
from helion._compiler.device_ir import RootGraphInfo


@pytest.mark.parametrize("empty_graphs", [False, True])
def test_discovery_without_contractions_needs_no_compile_context(
    empty_graphs: bool,
) -> None:
    graph = Graph()
    graph.output(None)
    graphs = [] if empty_graphs else [RootGraphInfo(0, graph)]
    with patch(
        "helion._compiler.compile_environment.CompileEnvironment.current",
        side_effect=AssertionError("No geometry exists to resolve"),
    ):
        assert not has_positional_contraction_candidate(graphs)
