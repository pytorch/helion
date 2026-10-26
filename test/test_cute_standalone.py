from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import noop_mask

import helion
from helion._testing import DEVICE
from helion._testing import import_example
import helion.language as hl

attention = import_example("attention")
attention_output = attention.attention_output
attention_backward = attention.attention_backward
flex_example = import_example("flex_attention")

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
def _host_offset(x: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    assert scale > 0
    offset = shift + 1.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * scale + offset
    return out


def test_only_host_branched_floats_are_hard_coded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By default a host branch on a float (``scale``) fails export with
    GuardOnDataDependentSymNode, as before; with
    ``HELION_CUTE_EXPORT_HARDCODE_SCALARS=1`` the export hard-codes it and
    keeps the float the host code only computes with (``shift``) a runtime
    argument."""
    x = torch.randn(128, device=DEVICE)
    bound = _host_offset.bind((x, 2.0, 2.0))
    monkeypatch.delenv("HELION_CUTE_EXPORT_HARDCODE_SCALARS", raising=False)
    with pytest.raises(GuardOnDataDependentSymNode):
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    monkeypatch.setenv("HELION_CUTE_EXPORT_HARDCODE_SCALARS", "1")
    path = tmp_path / "native_host_offset.py"
    path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    exported = _load(path)
    torch.testing.assert_close(exported._host_offset(x, 2.0, 2.0), x * 2.0 + 3.0)
    torch.testing.assert_close(exported._host_offset(x, 2.0, 3.0), x * 2.0 + 4.0)
    with pytest.raises(ValueError, match="constant"):
        exported._host_offset(x, 3.0, 2.0)
    # The repro caller passes the hard-coded value instead of a placeholder.
    repro_path = tmp_path / "native_host_offset_repro.py"
    repro_path.write_text(
        bound.to_code(
            bound.config_spec.default_config(),
            emit_repro_caller=True,
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
    )
    call = next(
        node
        for node in ast.parse(repro_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "call"
    )
    assert "    scale = 2.0" in ast.unparse(call).splitlines()
    _load(repro_path).call()


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


def test_sass_checked_kernels_are_not_exported() -> None:
    # The exported launcher has no SASS verification step
    # (``helion.runtime.cute.sass_check``), so a config whose render needs
    # one -- hinted mma_dag packets, ``_helion_cute_sass_checks`` -- is
    # refused at export; the plain config of the same kernel exports.
    probes = importlib.import_module("test.test_cute_mma_dag")
    kernel = helion.kernel(
        probes._head_sum_logits_batched,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
    )
    args = tuple(t.to(DEVICE) for t in probes._head_sum_inputs(h=32))
    bound = kernel.bind(args)
    hinted = probes._config(
        1, 64, warps=4, stages=4, load_eviction_policies=["l2_last"] * 2 + [""] * 5
    )
    with bound.env:
        hinted = bound.config_spec.normalized_config(hinted)
        plain = bound.config_spec.normalized_config(
            probes._config(1, 64, warps=4, stages=4)
        )
    options = helion.OutputCodeOptions(allow_helion_deps=False)
    assert "_helion_cute_sass_checks" in bound.to_code(hinted)
    with pytest.raises(NotImplementedError, match="SASS"):
        bound.to_code(hinted, options=options)
    assert "_helion_cute_sass_checks" not in bound.to_code(plain, options=options)


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


# The smoke test of the Helion-free interpreter; the flash attention exports
# below run the same check on far larger kernels and are the slow tier.
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


def test_mma_dag_kernel_exports_the_shared_ptx_rows(tmp_path: Path) -> None:
    # ``_mma_dag_runtime`` re-exports ``stg_b32`` / ``unpack_f16x2`` from
    # ``device/ptx.py``: the export has to carry the table builder with its
    # rows and bind the re-exports to them, and the module must load and run
    # with ``helion`` unimportable.  (The warp_mma GEMM family renders the
    # rows too but its launch plan is not exportable yet.)
    probes = importlib.import_module("test.test_cute_mma_dag")
    kernel = helion.kernel(
        probes._accumulating_addmm,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
        cute_materialize_transformed_operands=False,
    )
    g = torch.rand(1024, 512, device=DEVICE, dtype=torch.float16)
    c = torch.randn(1024, 256, device=DEVICE, dtype=torch.float16)
    bound = kernel.bind((g, g, g, c))
    with bound.env:
        default = bound.config_spec.default_config()
        config = bound.config_spec.normalized_config(
            helion.Config(
                **{
                    **default.config,
                    "cute_mma_dag": "warp",
                    # The TMA ring adds a descriptor-preparation plan the
                    # exporter refuses; per-thread packets leave the alias plan.
                    "cute_mma_dag_loads": "lane",
                }
            )
        )
    code = bound.to_code(config)
    for name in ("stg_b32", "unpack_f16x2"):
        assert f"_helion_mma_dag.{name}(" in code, name
    code = bound.to_code(
        config, options=helion.OutputCodeOptions(allow_helion_deps=False)
    )
    assert "_make_standalone_ptx" in code
    assert "_make_standalone_mma_dag_runtime" in code
    (tmp_path / "native_addmm.py").write_text(code)
    _run_without_helion(
        tmp_path,
        """
        import native_addmm
        rows = native_addmm._standalone_ptx
        runtime = native_addmm._standalone_mma_dag_runtime
        for name in ("stg_b32", "unpack_f16x2"):
            assert getattr(runtime, name) is getattr(rows, name), name
        torch.manual_seed(0)
        g = torch.rand(1024, 512, device="cuda", dtype=torch.float16)
        c = torch.randn(1024, 256, device="cuda", dtype=torch.float16)
        actual = native_addmm._accumulating_addmm(g, g, g, c)
        gf = g.float()
        expected = (c.float().T @ (gf * gf * gf * (1.0 - gf))).to(torch.float16)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)
    """,
    )


@pytest.mark.slow
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


@pytest.mark.slow
def test_flash_attention_backward(tmp_path: Path) -> None:
    """The fused flash backward's ``helion_flash_bwd`` plan only selects its
    launch path, so it exports like the forward."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("flash attention requires Blackwell")
    kernel = helion.kernel(
        attention_backward.fn,
        backend="cute",
        autotune_effort="none",
        autotune_accuracy_check=False,
        static_shapes=True,
    )
    torch.manual_seed(0)
    shape = (1, 2, 256, 64)
    q, k, v, o, do = (
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(5)
    )
    sm_scale = 1.0 / math.sqrt(shape[-1])
    # The example takes K pre-scaled to log2 units and the matching log2 LSE.
    scores = q.float() @ k.float().transpose(-1, -2) * sm_scale
    lse = torch.logsumexp(scores, -1) / math.log(2)
    delta = torch.randn(shape[:-1], device=DEVICE)
    args = (q, k * (sm_scale / math.log(2)), v, o, lse, do, delta, sm_scale)
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    expected = bound.compile_config(config)(*args)
    code = bound.to_code(
        config,
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    assert "'kind': 'helion_flash_bwd'" in code
    (tmp_path / "native_attention_bwd.py").write_text(code)
    torch.save((args, expected), tmp_path / "case.pt")
    _run_without_helion(
        tmp_path,
        """
        import native_attention_bwd
        args, expected = torch.load("case.pt")
        actual = native_attention_bwd.attention_backward(*args)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=2e-2, rtol=2e-2)
    """,
    )


@pytest.mark.slow
def test_flex_attention_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FlexAttention backward exports on the fused backward with its
    score/mask mods compiled in (the entry point omits them), the
    ``enable_gqa`` its host code branches on hard-coded
    (``HELION_CUTE_EXPORT_HARDCODE_SCALARS=1``) and ``scale`` a runtime
    scalar, and runs without Helion."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("flash attention requires Blackwell")
    monkeypatch.setenv("HELION_CUTE_EXPORT_HARDCODE_SCALARS", "1")
    kernel = helion.kernel(
        flex_example.helion_flex_attention_backward_kernel.fn,
        backend="cute",
        autotune_effort="none",
        static_shapes=True,
        fast_math=True,
    )
    torch.manual_seed(0)
    shape = (1, 2, 256, 64)
    q, k, v, out, grad_out = (
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(5)
    )
    scale = 1.0 / math.sqrt(shape[-1])
    lse = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) * scale, -1)
    block_mask = create_block_mask(
        noop_mask, None, None, shape[2], shape[2], device=str(DEVICE), BLOCK_SIZE=128
    )
    args = flex_example.flex_attention_backward_args(
        q, k, v, grad_out, lse, out, None, None, block_mask, scale, False
    )
    # The row terms of ``_flex_attention_bwd_preprocess`` (a Helion kernel of
    # its own, which would autotune here).
    lse2 = lse * 1.44269504
    delta = (out.float() * grad_out.float()).sum(-1)
    args = (*args[:4], lse2, delta, *args[6:])
    bound = kernel.bind(args)
    config = bound.config_spec.default_config()
    compiled = bound.compile_config(config)
    rescaled = (*args[:-2], scale * 0.5, args[-1])
    expected = (compiled(*args), compiled(*rescaled))
    code = bound.to_code(
        config,
        options=helion.OutputCodeOptions(allow_helion_deps=False),
    )
    assert "'kind': 'helion_flash_bwd'" in code
    entry = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "helion_flex_attention_backward_kernel"
    )
    names = [arg.arg for arg in entry.args.args]
    assert not {"score_mod", "score_mod_bwd", "block_mask_mask_mod"} & set(names)
    validate = entry.body[0]
    assert isinstance(validate, ast.Expr) and isinstance(validate.value, ast.Call)
    specs = dict(zip(names, ast.literal_eval(validate.value.args[1]), strict=True))
    assert specs["enable_gqa"] == ("constant", False)
    assert specs["scale"] == ("scalar", "float")
    (tmp_path / "native_flex_bwd.py").write_text(code)
    entry_args = tuple(
        bound.env.backend.standalone_entry_args(bound, case)
        for case in (args, rescaled)
    )
    torch.save((entry_args, expected), tmp_path / "cases.pt")
    _run_without_helion(
        tmp_path,
        """
        import native_flex_bwd
        cases, expected = torch.load("cases.pt")
        for args, references in zip(cases, expected, strict=True):
            actual = native_flex_bwd.helion_flex_attention_backward_kernel(*args)
            for value, reference in zip(actual, references, strict=True):
                torch.testing.assert_close(value, reference, atol=2e-2, rtol=2e-2)
    """,
    )
