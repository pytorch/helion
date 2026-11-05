"""Inductor template renaming of multi-stage CuTe host shells (CPU only).

Inductor embeds a Helion kernel's module under a placeholder kernel name.  A
multi-stage host shell (result fission, split-K workspace) attaches its stage
kernels and metadata to the host function at module level
(``host._helion_cute_kernels = ...``); those references must follow the
renamed host function, or the embedded module fails to load (NameError).
"""

from __future__ import annotations

import ast
import types
from unittest.mock import patch

import torch
from torch._inductor.utils import Placeholder

from test.test_cute_result_fission import _seed
from test.test_cute_result_fission import bind
from test.test_cute_result_fission import four_args
from test.test_cute_result_fission import four_outputs

import helion
from helion._compiler._inductor.template_buffer import HelionTemplateBuffer
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target

matmul_split_k = import_example("matmul_split_k").matmul_split_k

EMBEDDED_NAME = "triton_unk_fused_1"


def _renamed(source: str, host: str) -> str:
    root = ast.parse(source)
    stub = types.SimpleNamespace(_helion_kernel=types.SimpleNamespace(name=host))
    HelionTemplateBuffer._rename_with_placeholders(stub, root)  # type: ignore[arg-type]
    return ast.unparse(root).replace(str(Placeholder.KERNEL_NAME), EMBEDDED_NAME)


def _load(text: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(compile(text, "<inductor-embedded>", "exec"), namespace)
    return namespace


def _top_level_host_references(source: str, host: str) -> list[str]:
    return [
        ast.unparse(statement)
        for statement in ast.parse(source).body
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(node, ast.Name) and node.id == host
            for node in ast.walk(statement)
        )
    ]


def test_result_fission_metadata_follows_the_renamed_host():
    with cpu_codegen_target(), forbid_cuda_work():
        bound = bind(four_outputs, four_args())
        source = bound.to_code(_seed(bound))
    assert _top_level_host_references(source, "four_outputs")
    text = _renamed(source, "four_outputs")
    assert "four_outputs" not in [
        node.id for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Name)
    ]
    namespace = _load(text)
    host = namespace[EMBEDDED_NAME]
    kernels = host._helion_cute_kernels  # type: ignore[attr-defined]
    assert isinstance(kernels, tuple) and len(kernels) == 2
    assert host._helion_cute_result_fission["identity_output"] == "o0"  # type: ignore[attr-defined]


def test_split_k_workspace_metadata_follows_the_renamed_host():
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
    )
    args = (
        torch.empty((128, 32768), dtype=torch.float16),
        torch.empty((32768, 256), dtype=torch.float16),
        lambda acc, tile: acc,
    )
    with (
        cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        bound = kernel._bind_isolated(args)
        source = bound.to_code(
            helion.Config(
                block_sizes=[128, 128, 64],
                split_k=32,
                cute_split_k_workspace=True,
                cute_split_k_stages=12,
            )
        )
    assert _top_level_host_references(source, "matmul_split_k")
    namespace = _load(_renamed(source, "matmul_split_k"))
    kernels = namespace[EMBEDDED_NAME]._helion_cute_kernels  # type: ignore[attr-defined]
    assert isinstance(kernels, tuple) and len(kernels) == 2


def test_names_inside_function_bodies_are_untouched():
    source = (
        "def host(x):\n"
        "    host = x + 1\n"
        "    return host\n"
        "def _helion_host(y):\n"
        "    return y\n"
        "host.meta = (_helion_host,)\n"
    )
    text = _renamed(source, "host")
    tree = ast.parse(text)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert set(functions) == {EMBEDDED_NAME, f"_helion_host_{EMBEDDED_NAME}"}
    body_names = {
        node.id
        for node in ast.walk(functions[EMBEDDED_NAME])
        if isinstance(node, ast.Name)
    }
    assert "host" in body_names and EMBEDDED_NAME not in body_names
    assert ast.unparse(tree.body[-1]) == (
        f"{EMBEDDED_NAME}.meta = (_helion_host_{EMBEDDED_NAME},)"
    )
    assert _load(text)[EMBEDDED_NAME](1) == 2  # type: ignore[operator]


def test_single_stage_modules_have_no_top_level_host_reference():
    # The added renaming touches only top-level references to the host, which
    # an ordinary (fused) module does not contain: its renamed text is
    # unchanged by this fix.
    with cpu_codegen_target(), forbid_cuda_work():
        bound = bind(four_outputs, four_args())
        source = bound.to_code(bound.config_spec.default_config())
    assert _top_level_host_references(source, "four_outputs") == []
