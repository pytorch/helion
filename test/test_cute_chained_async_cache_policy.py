from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_masked_async_copy import copy_g2s_op
from helion._compiler.cute.chained_masked_async_copy import emit_masked_affine_copy


@pytest.mark.parametrize("config", ({}, {"cute_async_load_cache": "ca"}))
def test_omission_and_ca_retain_sdk_default_source(config):
    assert copy_g2s_op(config) == "cute.nvgpu.cpasync.CopyG2SOp()"


def test_explicit_cg_uses_existing_global_cache_mode():
    assert copy_g2s_op({"cute_async_load_cache": "cg"}) == (
        "cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL)"
    )


@pytest.mark.parametrize("value", (None, True, 0, "", "CG", "streaming", []))
def test_unvalidated_cache_choice_is_not_silently_ignored(value):
    with pytest.raises(ValueError, match="chained async load cache"):
        copy_g2s_op({"cute_async_load_cache": value})


def _masked(config, *, stride=(72, 1), mask="chain_masked_copy_row < 64"):
    graph = Graph()
    host = graph.call_function(chain._tracing_ops._host_tensor, ())
    leaf = graph.call_function(chain.memory_ops.load, (host, (), True))
    host.meta["val"] = torch.empty((64, 64), dtype=torch.bfloat16)
    leaf.meta["val"] = torch.empty((64, 64), dtype=torch.bfloat16)
    plan = SimpleNamespace(threads=128, loop=None)
    cg = SimpleNamespace(
        device_function=SimpleNamespace(
            config=SimpleNamespace(config=config), _expr_args={}
        )
    )

    class Expression:
        def __init__(self, *args):
            self.coordinate_names = set()
            self.definitions = {}
            self.origins = {}
            self.loaded_inputs = []

        def value(self, node, coords):
            assert node is leaf
            self.loaded_inputs = [(leaf, coords, coords, "original_scalar")]
            return "original_scalar"

        def _load_mask(self, node, coords):
            assert node is leaf
            return mask

        def tensor_name(self, source):
            assert source is host
            return "original_input"

    with (
        patch.object(chain, "_Expression", Expression),
        patch.object(chain, "_operand_domain", return_value=[]),
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(index_type=lambda: "cutlass.Int32"),
        ),
    ):
        return emit_masked_affine_copy(
            cast("chain.GenerateAST", cg),
            cast("chain.ChainedMatmulPlan", plan),
            leaf,
            "stage",
            "a",
            (64, 64),
            stride,
            1,
            "cutlass.BFloat16",
            [
                "original_predicate = row < 63",
                "destination[row, col] = original_scalar",
            ],
        )


def test_actual_masked_emitter_changes_only_atom_and_preserves_all_guards_fallback():
    default = _masked({})
    ca = _masked({"cute_async_load_cache": "ca"})
    cg = _masked({"cute_async_load_cache": "cg"})
    assert default is not None and cg is not None and default == ca
    old, new = "\n".join(default), "\n".join(cg)
    assert new.count(copy_g2s_op({"cute_async_load_cache": "cg"})) == 1
    assert (
        new.replace(copy_g2s_op({"cute_async_load_cache": "cg"}), copy_g2s_op({}))
        == old
    )
    assert ast.dump(
        ast.parse(
            new.replace(copy_g2s_op({"cute_async_load_cache": "cg"}), copy_g2s_op({}))
        )
    ) == ast.dump(ast.parse(old))
    assert "original_predicate = row < 63" in new
    assert "destination[row, col] = original_scalar" in new
    assert "num_bits_per_copy=128" in new
    assert "% 16 == 0" in new and ".layout.stride[" in new
    assert "cp_async_commit_group" not in new and "cp_async_wait_group" not in new


def test_cache_setting_cannot_admit_unaligned_or_nonrectangular_copy():
    for config in ({}, {"cute_async_load_cache": "cg"}):
        assert _masked(config, stride=(68, 1)) is None
        # Nonrectangular hole retains the existing scalar fallback behavior.
        result = _masked(
            config, mask="(chain_masked_copy_row < 64) & (chain_masked_copy_row != 31)"
        )
        if result is not None:
            assert "original_predicate = row < 63" in "\n".join(result)
