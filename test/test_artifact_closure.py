from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import types

import pytest

from helion._compiler.output_code_utils import close_artifact_source
from helion._compiler.output_code_utils import dependency_free_runtime_source
from helion._compiler.triton.backend import TritonBackend
from helion._utils import cdiv
from helion._utils import next_power_of_2
from helion.autotuner.aot_compile import generate_standalone_file

pytestmark = pytest.mark.cpu_only


def test_unused_source_import_is_removed() -> None:
    source = """
import original_kernel as _source_module
import torch

def kernel(x):
    return torch.empty_like(x)
"""
    closed = close_artifact_source(source)
    assert "original_kernel" not in closed
    assert "import torch" in closed
    assert close_artifact_source(closed) == closed


def test_shadowed_local_does_not_keep_source_import() -> None:
    source = """
import original_kernel as value

def kernel(value):
    def inner():
        return value
    return inner()
"""
    closed = close_artifact_source(source)
    assert "original_kernel" not in closed
    namespace = {}
    exec(closed, namespace)
    assert namespace["kernel"](7) == 7


@pytest.mark.parametrize(
    "body",
    [
        "def kernel(x):\n    return x * source.factor",
        "def kernel(x):\n    return source.helper(x)",
        "def kernel(x):\n    def inner():\n        return source.factor\n    return inner()",
        "@source.decorate\ndef kernel(x):\n    return x",
        "def kernel(x=source.default):\n    return x",
        "def kernel(*, x=source.default):\n    return x",
        "class Wrapper(source.Base):\n    pass",
        "class Wrapper:\n    value = source.factor",
        "def kernel(x):\n    return [v for v in source.values]",
        "def kernel(x: source.Tensor):\n    return x",
        'def kernel(x: "source.Tensor"):\n    return x',
        "def kernel(x) -> source.Tensor:\n    return x",
        "value: source.Tensor = None",
    ],
)
def test_live_external_binding_is_rejected(body: str) -> None:
    with pytest.raises(ValueError, match="external module"):
        close_artifact_source(
            "from __future__ import annotations\n"
            "import original_kernel as source\n" + body
        )


def test_live_from_import_is_rejected() -> None:
    with pytest.raises(ValueError, match="external module"):
        close_artifact_source(
            "from original_kernel import factor as scale\n"
            "def kernel(x):\n    return x * scale\n"
        )


def test_only_unused_aliases_are_dropped() -> None:
    closed = close_artifact_source(
        "import original_kernel as source, torch\n"
        "def kernel(x):\n    return torch.empty_like(x)\n"
    )
    assert "original_kernel" not in closed
    assert "import torch" in closed


@pytest.mark.parametrize(
    "source",
    [
        "from original_kernel import *",
        "from . import original_kernel",
        "def kernel(x):\n    import original_kernel\n    return x",
        "def unused_helper():\n    import original_kernel\n    return 1",
    ],
)
def test_unrelocatable_imports_are_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="imports|external module"):
        close_artifact_source(source)


@pytest.mark.parametrize(
    "source",
    [
        'def kernel():\n    return __import__("original_kernel")',
        'def kernel():\n    loader = __import__\n    return loader("original_kernel")',
        "from torch import __builtins__ as namespace\n",
        'import importlib as loader\ndef kernel():\n    return loader.import_module("original_kernel")',
        'from importlib import import_module as loader\ndef kernel():\n    return loader("original_kernel")',
        'def kernel():\n    return globals()["original_kernel"]',
        'def kernel():\n    return locals()["original_kernel"]',
        'def kernel():\n    return vars()["original_kernel"]',
        'def kernel():\n    return eval("original_kernel.factor")',
    ],
)
def test_dynamic_namespace_access_is_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="dynamic Python namespace"):
        close_artifact_source(source)


def test_unbound_global_is_rejected() -> None:
    with pytest.raises(ValueError, match="unbound global names: factor"):
        close_artifact_source("def kernel(x):\n    return x * factor\n")


def test_native_runtime_shim_remains_closed() -> None:
    source = (
        "from __future__ import annotations\nimport types\n"
        + dependency_free_runtime_source(TritonBackend().dependency_free_launcher_info)
        + "\ndef kernel(x):\n    return _default_launcher\n"
    )
    closed = close_artifact_source(source)
    assert "importlib.import_module('cuda.bindings.driver')" in closed
    assert "def default_launcher" in closed
    assert "def get_num_sm" in closed
    assert "def set_triton_allocator" in closed
    compile(closed, "<artifact>", "exec")
    assert close_artifact_source(closed) == closed


def test_standalone_host_helpers_execute_canonical_implementations() -> None:
    source = (
        "from __future__ import annotations\nimport types\n"
        + dependency_free_runtime_source(TritonBackend().dependency_free_launcher_info)
        + "\ndef block_size(k, split_k):\n"
        "    return helion.next_power_of_2(helion.cdiv(k, split_k))\n"
    )
    namespace = {}
    exec(close_artifact_source(source), namespace)
    embedded = namespace["helion"]
    assert isinstance(embedded, types.SimpleNamespace)
    assert "cdiv" not in namespace  # Helpers cannot collide with kernel names.
    assert "next_power_of_2" not in namespace
    for k in (-1, 0, 1, 3, 16, 64, 65):
        for split_k in (1, 2, 8):
            assert embedded.cdiv(k, split_k) == cdiv(k, split_k)
            assert embedded.next_power_of_2(k) == next_power_of_2(k)
            assert namespace["block_size"](k, split_k) == next_power_of_2(
                cdiv(k, split_k)
            )


def test_aot_variants_share_runtime_and_host_helpers() -> None:
    source = (
        "from helion.runtime import default_launcher as _default_launcher\n"
        "def kernel(n):\n"
        "    return helion.next_power_of_2(helion.cdiv(n, 3))\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = generate_standalone_file(
            "kernel",
            [source, source],
            "def key_kernel(*args):\n    return 0\n",
            Path(directory),
        )
        closed = close_artifact_source(path.read_text())
    assert closed.count("def _make_helion_runtime(") == 1
    assert closed.count("def next_power_of_2(") == 1
    namespace = {}
    exec(closed, namespace)
    assert namespace["_kernel_c0"](65) == 32
    assert namespace["_kernel_c1"](65) == 32
    assert namespace["kernel"](65) == 32


@pytest.mark.parametrize(
    "expression",
    [
        "helion.unknown(x)",
        "helion.runtime.unknown(x)",
        "getattr(helion, 'unknown')(x)",
        "getattr(helion.runtime, 'unknown')(x)",
        "helion",
        "helion.runtime",
    ],
)
def test_unsupported_helion_namespace_is_rejected(expression: str) -> None:
    source = (
        "import types\n"
        + dependency_free_runtime_source(TritonBackend().dependency_free_launcher_info)
        + f"\ndef kernel(x):\n    return {expression}\n"
    )
    with pytest.raises(ValueError, match="unsupported Helion namespace"):
        close_artifact_source(source)


def test_modified_runtime_shim_does_not_get_runtime_exemption() -> None:
    source = "import types\n" + dependency_free_runtime_source(
        TritonBackend().dependency_free_launcher_info
    )
    tree = ast.parse(source)
    runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    runtime.body.insert(1, ast.parse("import original_kernel").body[0])
    with pytest.raises(ValueError, match="external module|dynamic Python namespace"):
        close_artifact_source(ast.unparse(tree))
