from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.memory_ops import _cute_resolved_load_mask

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


@dataclass
class _MaskStrategy:
    index: str
    mask: str | None

    def index_var(self, block_id: int) -> str:
        return self.index

    def mask_var(self, block_id: int) -> str | None:
        return self.mask


@pytest.mark.parametrize("k_mask", [None, "mask_k"])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("alias", ["none", "same_mask", "different_mask"])
def test_full_slice_preserves_resolved_axis_mask(
    k_mask: str | None, extra: bool, alias: str
) -> None:
    shape_env = ShapeEnv()
    sizes = [shape_env.create_unbacked_symint() for _ in range(3)]
    env = SimpleNamespace(
        get_block_id=lambda size: next(
            (index for index, candidate in enumerate(sizes) if size is candidate),
            None,
        ),
        known_equal=operator.eq,
        block_sizes=[
            SimpleNamespace(block_id=index, size=64, var=size)
            for index, size in enumerate(sizes)
        ],
    )
    codegen = SimpleNamespace(
        active_device_loops={
            index: [
                SimpleNamespace(
                    strategy=_MaskStrategy(
                        f"index_{2 if index == 0 and alias != 'none' else index}",
                        k_mask if index == 0 and alias == "same_mask" else mask,
                    ),
                )
            ]
            for index, mask in enumerate(("mask_m", "mask_n", k_mask))
        },
        lift=lambda value, **kwargs: value,
    )
    state = cast(
        "CodegenState",
        SimpleNamespace(
            codegen=codegen,
            device_function=SimpleNamespace(
                cute_state=SimpleNamespace(
                    matmul_operand_block_remap={}, matmul_operand_index_override={}
                )
            ),
        ),
    )
    predicate = ast.Name(id="predicate", ctx=ast.Load()) if extra else None
    with patch.object(CompileEnvironment, "current", return_value=env):
        if alias == "different_mask":
            with pytest.raises(
                helion.exc.BackendUnsupported, match="ambiguous coordinate bounds"
            ):
                _cute_resolved_load_mask(
                    state, torch.empty(64), [None, slice(None)], ["index_2"], predicate
                )
            return
        actual = _cute_resolved_load_mask(
            state, torch.empty(64), [None, slice(None)], ["index_2"], predicate
        )
    # Neither an unrelated M/N tail nor an explicit false user mask may be
    # lost by matching equal extents. The K mask may legitimately be elided.
    for m_valid in (False, True):
        for k_valid in (False, True):
            for user_valid in (False, True):
                values = {
                    "mask_m": m_valid,
                    "mask_n": not m_valid,
                    "mask_k": k_valid,
                    "predicate": user_valid,
                }
                assert eval(actual or "True", {}, values) == (
                    (k_mask is None or k_valid) and (not extra or user_valid)
                )
