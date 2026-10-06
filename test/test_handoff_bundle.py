from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
import stat
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion.autotuner import FiniteSearch
from helion.autotuner import HandoffBundle
from helion.autotuner import HandoffPolicy
from helion.autotuner import build_handoff
from helion.autotuner import find_handoff
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.handoff_bundle import _input_metadata
from helion.autotuner.handoff_bundle import _tolerances
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_input_contract_preserves_storage_metadata() -> None:
    storage = torch.randn(32)
    metadata = _input_metadata((storage[1:9], storage[9:17], storage.clone(), 2.0))
    assert isinstance(metadata, tuple)
    assert metadata[0]["storage_offset"] == 1
    assert metadata[0]["storage_group"] == metadata[1]["storage_group"]
    assert metadata[0]["storage_group"] != metadata[2]["storage_group"]
    assert metadata[3] == {"type": "float", "value": 2.0}


def test_fp8_reference_tolerances() -> None:
    inputs = (torch.ones(8),)
    output = torch.ones(8).to(torch.float8_e4m3fn)
    assert _tolerances(helion.Settings(), output, inputs, inputs) == (0.0, 0.0)
    assert _tolerances(
        helion.Settings(autotune_baseline_atol=0.02), output, inputs, inputs
    ) == (0.02, 0.01)
    assert _tolerances(helion.Settings(), output, inputs, (inputs[0] + 1,)) == (
        0.01,
        0.01,
    )


def test_agent_rejects_non_source_replacements_before_writing(tmp_path: Path) -> None:
    (tmp_path / "kernel.py").write_text("original")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"cases": [{"source": "kernel.py"}]})
    )
    bundle = HandoffBundle(tmp_path)
    with pytest.raises(ValueError, match="existing native source"):
        bundle.run_agent(lambda _: {"kernel.py": "changed", "reference.pt": "bad"})
    assert bundle.sources == {"kernel.py": "original"}


@pytest.mark.parametrize("kind", ["absolute", "parent", "symlink"])
@pytest.mark.parametrize("operation", ["sources", "single_round", "rounds"])
def test_reopened_bundle_rejects_escaping_sources(tmp_path, kind, operation):
    outside = tmp_path / "outside.py"
    outside.write_text("outside source")
    mode = stat.S_IMODE(outside.stat().st_mode)
    directory = tmp_path / "bundle"
    directory.mkdir()
    if kind == "absolute":
        source = str(outside)
    elif kind == "parent":
        source = "../outside.py"
    else:
        source = "kernel.py"
        (directory / source).symlink_to(outside)
    (directory / "manifest.json").write_text(
        json.dumps({"cases": [{"source": source}]})
    )
    (directory / "prompt.md").write_text("prompt")
    bundle = HandoffBundle(directory)

    def agent(workspace):
        pytest.fail("Escaping source paths must be rejected before calling the agent")

    with pytest.raises(ValueError, match="remain inside the bundle"):
        if operation == "sources":
            assert bundle.sources
        elif operation == "single_round":
            bundle.run_agent(agent)
        else:
            bundle.run_agent_rounds(agent, budget_seconds=30)
    assert outside.read_text() == "outside source"
    assert stat.S_IMODE(outside.stat().st_mode) == mode


def test_agent_revalidates_source_path_after_callback(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("outside source")
    directory = tmp_path / "bundle"
    directory.mkdir()
    source = directory / "kernel.py"
    source.write_text("original")
    (directory / "manifest.json").write_text(
        json.dumps({"cases": [{"source": "kernel.py"}]})
    )

    def agent(workspace):
        source.unlink()
        source.symlink_to(outside)
        return {"kernel.py": "changed"}

    with pytest.raises(ValueError, match="remain inside the bundle"):
        HandoffBundle(directory).run_agent(agent)
    assert outside.read_text() == "outside source"


def test_relative_bundle_directory_is_supported(tmp_path, monkeypatch):
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "kernel.py").write_text("original")
    (directory / "manifest.json").write_text(
        json.dumps({"cases": [{"source": "kernel.py"}]})
    )
    monkeypatch.chdir(tmp_path)
    assert HandoffBundle(Path("bundle")).sources == {"kernel.py": "original"}


class _EquivalentSource(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, (int, float))
        ):
            node.op = ast.Sub()
            node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
        return node


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["cute", "triton"])
@pytest.mark.parametrize("multi_shape", [False, True])
def test_native_source_handoff(backend: str, multi_shape: bool, tmp_path: Path) -> None:
    if backend == "cute":
        pytest.importorskip("cutlass.cute")

    @helion.kernel(
        backend=backend,
        static_shapes=True,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
    )
    def scaled_add(x: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            out[tile] = x[tile] * scale + 1.0
        return out

    x = torch.randn(129, device=DEVICE)[1:]
    original = x.clone()
    args = (x, hl.constexpr(2.0))
    bound = scaled_add.bind(args)
    search_args: Sequence[object] = args
    if multi_shape:
        second = (torch.randn(257, device=DEVICE)[1:], hl.constexpr(2.0))
        search_args = cast(
            "Sequence[object]",
            _MultiShapeAutotuneArgs(
                cases=((bound, args), (scaled_add.bind(second), second)),
                aggregation="max",
                relative_to="default",
                cache_tag=None,
                workload_key=("native_handoff",),
            ),
        )
    search = FiniteSearch(
        bound,
        search_args,
        [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])],
    )
    point = find_handoff(search, HandoffPolicy())
    bundle = build_handoff(search, point, tmp_path / "bundle", repetitions=2)
    manifest = bundle.manifest
    assert manifest["objective"]["unit"] == ("ratio" if multi_shape else "ms")
    assert manifest["baseline_evaluation"]["ok"]
    assert manifest["cases"][0]["timing"] == "cuda_graph"
    assert len(manifest["cases"]) == (2 if multi_shape else 1)
    assert manifest["cases"][0]["reference_kind"] == "confirmed_original"
    assert manifest["cases"][0]["input_metadata"][0]["storage_offset"] == 1
    assert manifest["autotune"]["measurements"]
    assert (
        "CuTe DSL" in bundle.prompt if backend == "cute" else "Triton" in bundle.prompt
    )
    assert dataclasses.fields(bundle)[0].name == "directory"
    originals = bundle.sources

    def agent(workspace: HandoffBundle) -> dict[str, str]:
        replacements = {}
        for path, source in workspace.sources.items():
            tree = _EquivalentSource().visit(ast.parse(source))
            replacements[path] = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
            assert replacements[path] != source
        return replacements

    result = bundle.run_agent(agent, repetitions=2)
    assert result.ok, result
    assert all(len(case.samples) == 2 for case in result.cases)
    assert result.perf is not None and result.perf > 0
    assert (tmp_path / "bundle" / "case_0" / "original.py").read_text() == originals[
        "case_0/kernel.py"
    ]
    # A replay needs only the workspace; the original search is no longer used.
    if not multi_shape:
        replay = HandoffBundle(bundle.directory).evaluate(repetitions=1)
        assert replay.ok, replay
        assert replay.cases[0].source_hash == result.cases[0].source_hash

    def invalid_agent(workspace: HandoffBundle) -> dict[str, str]:
        source = workspace.sources["case_0/kernel.py"]
        return {
            "case_0/kernel.py": source
            + "\n_original = scaled_add\ndef scaled_add(x, scale):\n    output = _original(x, scale)\n    x.zero_()\n    return output\n"
        }

    invalid = bundle.run_agent(invalid_agent, repetitions=1)
    assert not invalid.ok and invalid.perf is None
    assert invalid.cases[0].status == "accuracy_error"
    torch.testing.assert_close(x, original)

    if not multi_shape and backend == "triton":
        with pytest.raises(RuntimeError, match="baseline failed"):
            build_handoff(
                search,
                point,
                tmp_path / "bad_reference",
                reference_fn=lambda x, scale: torch.zeros_like(x),
                repetitions=1,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_numerical_reference_preserves_original_mutation_and_aliases(
    tmp_path: Path,
) -> None:
    @helion.kernel(
        backend="triton",
        static_shapes=True,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
    )
    def increment(x: torch.Tensor) -> torch.Tensor:
        for tile in hl.tile(x.size()):
            x[tile] = x[tile] + 1.0
        return x

    args = (torch.randn(128, device=DEVICE),)
    before = args[0].clone()
    bound = increment.bind(args)
    search = FiniteSearch(
        bound, args, [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])]
    )
    point = find_handoff(search)
    bundle = build_handoff(
        search,
        point,
        tmp_path / "bundle",
        reference_fn=lambda x: x + 1.0,
        repetitions=1,
    )
    assert bundle.manifest["cases"][0]["reference_kind"] == "reference_fn"
    # The CLI replays the saved workload without the defining Helion kernel.
    completed = subprocess.run(
        [sys.executable, str(bundle.directory / "evaluate.py"), "--repetitions", "1"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads(completed.stdout)["ok"]
    failed = bundle.run_agent(
        lambda workspace: {
            "case_0/kernel.py": workspace.sources["case_0/kernel.py"]
            + "\n_original = increment\ndef increment(x):\n    return _original(x).clone()\n",
        },
        repetitions=1,
    )
    assert not failed.ok and failed.cases[0].status == "accuracy_error"
    torch.testing.assert_close(args[0], before)
