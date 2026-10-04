from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import chained_vector_expression
from helion._compiler.cute.chained_execution import ChainedExecution

if TYPE_CHECKING:
    from torch.fx import Node

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
    from helion._compiler.generate_ast import GenerateAST


class _UnreadableNode:
    @property
    def meta(self) -> None:
        raise AssertionError("ineligible geometry must not inspect the node")


@pytest.mark.parametrize(
    ("threads", "width"),
    [(128, 0), (128, 7), (128, 24), (96, 512), (128, 2048)],
)
@pytest.mark.parametrize("explicit_execution", [False, True])
def test_vector_geometry_rejects_before_node_callbacks_or_unroll(
    threads: int, width: int, explicit_execution: bool
) -> None:
    coordinates = Mock(side_effect=AssertionError("coordinates must not run"))
    final_value = Mock(side_effect=AssertionError("final value must not run"))
    unroll = Mock()
    execution = ChainedExecution(threads) if explicit_execution else None
    plan = cast(
        "ChainedMatmulPlan",
        None if explicit_execution else SimpleNamespace(threads=threads),
    )
    with patch.object(
        chained_vector_expression.chain,
        "_Expression",
        side_effect=AssertionError("expression must not run"),
    ):
        assert (
            chained_vector_expression.emit_vector_expression(
                cast("GenerateAST", None),
                plan,
                {},
                cast("Node", _UnreadableNode()),
                shape=(128, width),
                coordinates=coordinates,
                offset=0,
                tag="guard",
                target="target",
                final_value=final_value,
                producer_unroll=cast("BoundedProducerUnroll", unroll),
                execution=execution,
            )
            is None
        )
    coordinates.assert_not_called()
    final_value.assert_not_called()
    unroll.loop_factor.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64, torch.bool])
def test_valid_vector_geometry_still_rejects_unsupported_dtype_early(
    dtype: torch.dtype,
) -> None:
    coordinates = Mock(side_effect=AssertionError("coordinates must not run"))
    node = cast("Node", SimpleNamespace(meta={"val": torch.empty((), dtype=dtype)}))
    with patch.object(
        chained_vector_expression.chain,
        "_Expression",
        side_effect=AssertionError("expression must not run"),
    ):
        assert (
            chained_vector_expression.emit_vector_expression(
                cast("GenerateAST", None),
                cast("ChainedMatmulPlan", None),
                {},
                node,
                shape=(128, 512),
                coordinates=coordinates,
                offset=0,
                tag="guard",
                target="target",
                execution=ChainedExecution(128),
            )
            is None
        )
    coordinates.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_valid_vector_geometry_and_dtype_reach_expression(dtype: torch.dtype) -> None:
    coordinates = Mock(side_effect=lambda row, column: (row, column))
    node = cast("Node", SimpleNamespace(meta={"val": torch.empty((), dtype=dtype)}))
    with (
        patch.object(
            chained_vector_expression.chain,
            "_Expression",
            side_effect=RuntimeError("expression reached"),
        ),
        pytest.raises(RuntimeError, match="expression reached"),
    ):
        chained_vector_expression.emit_vector_expression(
            cast("GenerateAST", None),
            cast("ChainedMatmulPlan", None),
            {},
            node,
            shape=(128, 512),
            coordinates=coordinates,
            offset=0,
            tag="guard",
            target="target",
            execution=ChainedExecution(128),
        )
    coordinates.assert_called_once()
