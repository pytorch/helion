"""Generated code must not depend on the process's string hash seed: the
source keys the disk caches and the autotuner's records across processes."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest import mock

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def test_an_outlined_helpers_constexprs_are_declared_in_order() -> None:
    # The block sizes an outlined Triton body reads become module constexprs,
    # which were declared in the order of a set of names: one that follows
    # the hash seed, almost never sorted for eight names.
    device_function = object.__new__(DeviceFunction)
    device_function.arguments = []
    device_function.wrapper_only_params = []
    device_function.preamble = []
    cast("Any", device_function).namespace = SimpleNamespace(
        create_name=lambda name, _value: name
    )
    device_function.triton_outlined_helpers = []
    device_function.triton_outlined_helper_constexprs = {}
    device_function._variable_renames = {}
    device_function.dce_vars = []
    device_function.config = helion.Config()
    cast("Any", device_function).codegen = SimpleNamespace(module_statements=[])
    cast("Any", device_function).helper_manager = SimpleNamespace(
        codegen_helper_functions=list
    )
    names = [f"_BLOCK_SIZE_{block_id}" for block_id in (7, 3, 5, 1, 8, 2, 6, 4)]
    body = ast.parse(f"tl.store(output, {' + '.join(names)})\n").body
    environment = SimpleNamespace(
        backend_name="triton",
        block_sizes=[
            SimpleNamespace(from_config_assert=lambda _config: 16) for _ in range(9)
        ],
    )
    with mock.patch.object(CompileEnvironment, "current", return_value=environment):
        device_function.register_triton_outlined_helper("opaque_tile", body)
    declared = [
        statement.targets[0].id
        for statement in device_function.codegen_helper_functions()
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
    ]
    assert declared == sorted(names)


def _softmax_rows_then_double(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    a = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        top = torch.amax(row, -1, keepdim=True)
        total = torch.exp(row - top).sum(-1, keepdim=True)
        for tile_n in hl.tile(n):
            a[tile_m, tile_n] = (torch.exp(x[tile_m, tile_n] - top) / total).to(a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = a[tile_m, tile_n] * 2
    return out


def render_outlined_root() -> str:
    """The dynamic cross-loop pipeline of ``_softmax_rows_then_double``,
    whose second root becomes a helper that reads two block sizes as module
    constexprs."""
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            helion.kernel(
                _softmax_rows_then_double, static_shapes=True, backend="triton"
            ),
            (torch.empty(64, 256),),
        )
        return bound.to_code(
            helion.Config(
                block_sizes=[16, 32, 16, 32],
                reduction_loops=[64],
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
            )
        )


@pytest.mark.slow
@onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_an_outlined_root_renders_the_same_under_two_hash_seeds() -> None:
    # Under hash seeds 0 and 2 the helper declared _BLOCK_SIZE_3 and
    # _BLOCK_SIZE_4 in opposite orders.
    root = Path(__file__).resolve().parents[1]
    renders = [
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from test.test_codegen_hash_seed import render_outlined_root; "
                    "print(render_outlined_root())"
                ),
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": str(root),
                "CUDA_VISIBLE_DEVICES": "",
            },
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for seed in ("0", "2")
    ]
    assert "_BLOCK_SIZE_3 = tl.constexpr(16)" in renders[0]
    assert renders[0] == renders[1]
