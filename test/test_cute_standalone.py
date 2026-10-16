from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple

from examples.attention import attention_output
import pytest
import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _pointwise(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * scale + 1.0
    return out


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _run_without_helion(tmp_path: Path, body: str) -> None:
    blocker = textwrap.dedent("""
        import importlib.abc
        import sys
        import torch

        class NoHelion(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "helion" or fullname.startswith("helion."):
                    raise AssertionError("Standalone module tried to import " + fullname)
        sys.meta_path.insert(0, NoHelion())
        assert not any(name == "helion" or name.startswith("helion.") for name in sys.modules)
    """)
    subprocess.run(
        [sys.executable, "-c", blocker + textwrap.dedent(body)],
        cwd=tmp_path,
        check=True,
        timeout=120,
    )


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("layout", ["contiguous", "offset", "transposed"])
def test_pointwise(tmp_path: Path, layout: str) -> None:
    if layout == "offset":
        x = torch.randn(129, device=DEVICE)[1:]
    elif layout == "transposed":
        x = torch.randn(8, 16, device=DEVICE).T
    else:
        x = torch.randn(128, device=DEVICE)
    bound = _pointwise.bind((x, 2.0))
    code = bound.to_code(
        bound.config_spec.default_config(),
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    path = tmp_path / "native_pointwise.py"
    path.write_text(code)
    exported = _load(path)
    torch.testing.assert_close(exported._pointwise(x, 2.0), x * 2.0 + 1.0)
    torch.testing.assert_close(exported._pointwise(x, 3.0), x * 3.0 + 1.0)
    with pytest.raises(ValueError, match="tensor layout"):
        exported._pointwise(x[:1], 2.0)
    with pytest.raises(ValueError, match="scalar type"):
        exported._pointwise(x, 2)
    if layout == "contiguous":
        with pytest.raises(ValueError, match="tensor layout"):
            exported._pointwise(torch.randn(129, device=DEVICE)[1:], 2.0)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _constant(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * scale
    return out


@pytest.mark.parametrize("scale", [2.0, math.inf, -math.inf, math.nan])
def test_constant_guard(tmp_path: Path, scale: float) -> None:
    x = torch.randn(128, device=DEVICE)
    bound = _constant.bind((x, hl.constexpr(scale)))
    config = bound.config_spec.default_config()
    torch.testing.assert_close(
        bound.compile_config(config)(x, scale), x * scale, equal_nan=True
    )
    path = tmp_path / "native_constant.py"
    path.write_text(
        bound.to_code(
            config,
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    exported = _load(path)
    torch.testing.assert_close(exported._constant(x, scale), x * scale, equal_nan=True)
    with pytest.raises(ValueError, match="constant"):
        exported._constant(x, 3.0)


def test_dynamic_shapes_rejected() -> None:
    kernel = helion.kernel(
        _pointwise.fn, backend="cute", autotune_effort="none", static_shapes=False
    )
    bound = kernel.bind((torch.randn(128, device=DEVICE), 2.0))
    with pytest.raises(NotImplementedError, match="static_shapes=True"):
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _tuple_and_mutation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        value = x[tile] + 1.0
        x[tile] = value
        out[tile] = value * 2.0
    return x, out


def test_tuple_return_and_mutation(tmp_path: Path) -> None:
    x = torch.randn(128, device=DEVICE)
    original = x.clone()
    bound = _tuple_and_mutation.bind((x,))
    path = tmp_path / "native_mutation.py"
    path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    # Export executes the host only on FakeTensors.
    torch.testing.assert_close(x, original)
    exported = _load(path)
    mutated, out = exported._tuple_and_mutation(x)
    assert mutated is x
    torch.testing.assert_close(x, original + 1.0)
    torch.testing.assert_close(out, (original + 1.0) * 2.0)
    mutated, out = exported._tuple_and_mutation(x)
    torch.testing.assert_close(mutated, original + 2.0)
    torch.testing.assert_close(out, (original + 2.0) * 2.0)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x


def test_input_alias_guard(tmp_path: Path) -> None:
    x, y = (torch.randn(128, device=DEVICE) for _ in range(2))
    expected = x + y
    bound = _add.bind((x, y))
    path = tmp_path / "native_add.py"
    path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    exported = _load(path)
    torch.testing.assert_close(exported._add(x, y), expected)
    with pytest.raises(ValueError, match="aliasing"):
        exported._add(x, x)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _nested(bundle: tuple[torch.Tensor, float]) -> torch.Tensor:
    x, scale = bundle
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * scale
    return out


@pytest.mark.parametrize("named_tuple", [False, True])
def test_nested_runtime_scalar(tmp_path: Path, named_tuple: bool) -> None:
    class Pair(NamedTuple):
        x: torch.Tensor
        scale: float

    x = torch.randn(128, device=DEVICE)
    pair = Pair if named_tuple else lambda x, scale: (x, scale)
    bound = _nested.bind((pair(x, 2.0),))
    path = tmp_path / "native_nested.py"
    path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    exported = _load(path)
    torch.testing.assert_close(exported._nested(pair(x, 3.0)), x * 3.0)


def test_native_source_edit_without_helion(tmp_path: Path) -> None:
    x = torch.randn(128, device=DEVICE)
    bound = _pointwise.bind((x, 2.0))
    code = bound.to_code(
        bound.config_spec.default_config(),
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    original = tmp_path / "original.py"
    original.write_text(code)
    tree = ast.parse(code)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion__pointwise"
    )
    for node in ast.walk(device):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and node.value == 1.0
        ):
            node.value = 7.0
    edited = tmp_path / "edited.py"
    edited.write_text(ast.unparse(tree))
    _run_without_helion(
        tmp_path,
        """
        import original
        import edited
        x = torch.randn(128, device="cuda")
        torch.testing.assert_close(original._pointwise(x, 3.0), x * 3.0 + 1.0)
        torch.testing.assert_close(edited._pointwise(x, 3.0), x * 3.0 + 7.0)
    """,
    )


def test_current_stream_and_cuda_graph(tmp_path: Path) -> None:
    x = torch.ones(128, device=DEVICE)
    bound = _pointwise.bind((x, 2.0))
    path = tmp_path / "native_graph.py"
    path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    exported = _load(path)
    exported._pointwise(x, 2.0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = exported._pointwise(x, 3.0)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(output, x * 3.0 + 1.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = exported._pointwise(x, 3.0)
    x.fill_(2.0)
    graph.replay()
    torch.testing.assert_close(captured, x * 3.0 + 1.0)


@pytest.mark.parametrize(("family", "head_dim"), [("fa4", 64), ("fa4_alt", 128)])
def test_flash_attention(tmp_path: Path, family: str, head_dim: int) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("flash attention requires Blackwell")
    kernel = helion.kernel(
        attention_output.fn, backend="cute", autotune_effort="none", static_shapes=True
    )
    args = tuple(
        torch.randn(1, 1, 256, head_dim, device=DEVICE, dtype=torch.float16)
        for _ in range(3)
    )
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    config.config["cute_flash_pipeline_family"] = family
    torch.testing.assert_close(
        bound.compile_config(config)(*args),
        torch.nn.functional.scaled_dot_product_attention(*args),
        atol=5e-2,
        rtol=2e-2,
    )
    code = bound.to_code(
        config,
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    path = tmp_path / "native_attention.py"
    path.write_text(code)
    assert "_make_standalone_flash_runtime" in code
    assert "_make_standalone_flash_gemm_ptx" in code
    assert f"flash_{family}_shared_storage" in code
    _run_without_helion(
        tmp_path,
        f"""
        import native_attention
        args = tuple(torch.randn(1, 1, 256, {head_dim}, device="cuda", dtype=torch.float16) for _ in range(3))
        actual = native_attention.attention_output(*args)
        expected = torch.nn.functional.scaled_dot_product_attention(*args)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)
    """,
    )
