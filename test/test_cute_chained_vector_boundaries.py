from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_pipeline import _args
from .test_cute_chained_preparation_pipeline import _config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@pytest.mark.parametrize("pipeline", [False, True])
def test_boundary_only_operands_keep_original_scalar_ownership_cpu(pipeline):
    args = _args()
    config = _config(16, pipeline=pipeline)
    config.config["cute_chained_pointwise_vectorize"] = True
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
            source = bound.to_code(config)
    tree = ast.parse(source)
    vector_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.endswith("_vector_step")
    ]
    boundary_prefixes = ("chain_prepared_",) if pipeline else ("chain_",)
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id.startswith(boundary_prefixes)
        and not node.value.id.endswith(("_values", "_target", "_coords"))
        for loop in vector_loops
        for node in ast.walk(loop)
    )
    assert "num_bits_per_copy=128" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_vector_boundary_gpu_retains_typed_carries_and_slot_generations(
    pipeline, steps
):
    args = _args(DEVICE, steps, True)
    saved = tuple(value.clone() for value in args[:4])
    config = _config(16, pipeline=pipeline)
    bound = _typed_sequence._bind_isolated(args)
    with bound.env.use_runtime_arg_values(_runtime_values(_typed_sequence, args)):
        scalar = bound.compile_config(config)
        config.config["cute_chained_pointwise_vectorize"] = True
        vector = bound.compile_config(config)
    expected = scalar(*args)
    actual = vector(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(vector(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args[:4], saved, atol=0, rtol=0)
