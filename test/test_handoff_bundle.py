from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner import AccuracyCheck
from helion.autotuner import FiniteSearch
from helion.autotuner import HandoffBundle
from helion.autotuner import HandoffPolicy
from helion.autotuner import build_handoff
from helion.autotuner import find_handoff
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.handoff_bundle import _accuracy
from helion.autotuner.handoff_bundle import _input_metadata
from helion.autotuner.handoff_bundle import _tolerances
from helion.autotuner.handoff_inputs import _clone_inputs
from helion.autotuner.handoff_inputs import _load_inputs
from helion.autotuner.handoff_inputs import _save_inputs
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def test_input_contract_preserves_storage_metadata() -> None:
    storage = torch.randn(32)
    metadata = _input_metadata((storage[1:9], storage[9:17], storage.clone(), 2.0))
    assert isinstance(metadata, tuple)
    assert metadata[0]["storage_offset"] == 1
    assert metadata[0]["storage_group"] == metadata[1]["storage_group"]
    assert metadata[0]["storage_group"] != metadata[2]["storage_group"]
    assert metadata[3] == {"type": "float", "value": 2.0}


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_input_metadata(value: float) -> None:
    metadata = _input_metadata((value,))
    assert json.loads(json.dumps(metadata, allow_nan=False)) == [
        {"type": "float", "value": str(value)}
    ]


def test_saved_inputs_preserve_named_fields_and_storage(tmp_path: Path) -> None:
    class Pair(NamedTuple):
        x: torch.Tensor
        alias: torch.Tensor

    base = torch.arange(33, dtype=torch.float32)
    x = torch.from_dlpack(base[1:])
    args = (Pair(x, x[2:]), {"duplicate": x})
    path = tmp_path / "inputs.pt"
    cloned = _clone_inputs(args)
    _save_inputs(cloned, path)
    loaded = _load_inputs(path)
    for actual in (cloned, loaded):
        pair, duplicate = actual
        assert pair._fields == ("x", "alias")
        assert pair.x is duplicate["duplicate"]
        assert pair.x.data_ptr() % 256 == x.data_ptr() % 256
        assert pair.x.storage_offset() == x.storage_offset() == 0
        assert pair.x.untyped_storage().nbytes() == x.untyped_storage().nbytes()
        assert pair.x.untyped_storage() == pair.alias.untyped_storage()
        assert pair.x.untyped_storage() != x.untyped_storage()
        torch.testing.assert_close(pair.x, x)
        torch.testing.assert_close(pair.alias, x[2:])


def _replay_bundle(bundle: HandoffBundle, cwd: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(bundle.directory / "evaluate.py"), "--repetitions", "1"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads(completed.stdout)["ok"]


def test_fp8_reference_tolerances() -> None:
    """The bundle applies the autotuner's fp8 rule: exact bytes under the
    'bitwise' mode (or explicit zero tolerances), the default tolerances on
    top of one dequantized ulp under the default 'ulp' mode."""
    inputs = (torch.ones(8),)
    output = torch.ones(8).to(torch.float8_e4m3fn)
    bitwise = helion.Settings(autotune_fp8_accuracy="bitwise")
    assert _tolerances(bitwise, output, inputs, inputs) == (0.0, 0.0)
    assert _tolerances(helion.Settings(), output, inputs, inputs) == (0.01, 0.01)
    assert _tolerances(
        helion.Settings(autotune_baseline_atol=0.0, autotune_baseline_rtol=0.0),
        output,
        inputs,
        inputs,
    ) == (0.0, 0.0)
    assert _tolerances(
        helion.Settings(autotune_baseline_atol=0.02), output, inputs, inputs
    ) == (0.02, 0.01)
    assert _tolerances(bitwise, output, inputs, (inputs[0] + 1,)) == (0.01, 0.01)


def test_accuracy_rule_records_an_accuracy_check() -> None:
    """A bundle records the rule autotuning applied: an AccuracyCheck's own
    tolerances and atol scaling, else the autotune_baseline_* settings."""
    inputs = (torch.ones(8),)
    output = torch.ones(8)
    check = AccuracyCheck(atol=0.03, rtol=0.0, atol_scale="peak")
    assert _accuracy(
        helion.Settings(autotune_baseline_accuracy_check_fn=check),
        output,
        inputs,
        inputs,
    ) == {
        "atol": 0.03,
        "rtol": 0.0,
        "atol_scale": "peak",
        "scale_atol": False,
        "fp8_accuracy": "ulp",
    }
    default = _accuracy(helion.Settings(), output, inputs, inputs)
    assert default["atol_scale"] == "rms" and default["scale_atol"]
    explicit = _accuracy(
        helion.Settings(autotune_baseline_atol=0.02), output, inputs, inputs
    )
    assert (explicit["atol"], explicit["atol_scale"], explicit["scale_atol"]) == (
        0.02,
        None,
        False,
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@skipUnlessBackends(["triton"])
def test_handoff_applies_an_accuracy_check(tmp_path: Path) -> None:
    """build_handoff accepts an AccuracyCheck, whose peak-scaled bound then
    decides the agent's rounds, and still refuses arbitrary callbacks."""

    def offset_kernel(check: object) -> Any:
        @helion.kernel(
            backend="triton",
            static_shapes=True,
            autotune_precompile=None,
            autotune_benchmark_subprocess=False,
            autotune_baseline_accuracy_check_fn=check,
        )
        def offset(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0
            return out

        return offset

    # x + 1 peaks at 16: atol 0.03 x peak admits 0.48 absolute.
    args = (torch.arange(16, device=DEVICE, dtype=torch.float32),)
    configs = [helion.Config(block_sizes=[8]), helion.Config(block_sizes=[16])]
    check = AccuracyCheck(atol=0.03, rtol=0.0, atol_scale="peak")
    search = FiniteSearch(offset_kernel(check).bind(args), args, configs)
    point = find_handoff(search)
    bundle = build_handoff(search, point, tmp_path / "bundle", repetitions=1)
    case = bundle.manifest["cases"][0]
    assert (case["atol"], case["rtol"], case["atol_scale"]) == (0.03, 0.0, "peak")
    # Version-1 evaluators would read scale_atol=False as an unscaled atol.
    assert bundle.manifest["schema_version"] == 2

    source = (bundle.directory / "case_0/kernel.py").read_text()

    def shifted(delta: float) -> Any:
        def agent(workspace: Any) -> dict[str, str]:
            return {
                "case_0/kernel.py": source + f"\n_original = offset\ndef offset(x):\n"
                f"    return _original(x) + {delta}\n"
            }

        return agent

    rejected = bundle.run_agent(shifted(0.6), repetitions=1)
    assert rejected.cases[0].status == "accuracy_error", rejected.cases[0].error
    accepted = bundle.run_agent(shifted(0.4), repetitions=1)
    assert accepted.cases[0].status == "ok", accepted.cases[0].error

    # Refused before the point is used, so the first search's point serves.
    custom = FiniteSearch(
        offset_kernel(lambda actual, expected: None).bind(args), args, configs
    )
    with pytest.raises(NotImplementedError, match="AccuracyCheck"):
        build_handoff(custom, point, tmp_path / "custom", repetitions=1)


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


# The bundle replays below each start a fresh Python process that imports torch
# and Helion and compiles the kernel again: 20-33s alone on a B200, past the 60s
# CI default once eight xdist workers share the runner's CPUs. A bundle's kernel
# backend is fixed at definition, so each test runs in the lane of its backend
# (skipUnlessBackends); the replays are the slow tier except one smoke test per
# lane: this one for CuTe and the numerical-reference test below for Triton.
_LANE_BACKENDS = [
    pytest.param("cute", marks=skipUnlessBackends(["cute"])),
    pytest.param("triton", marks=skipUnlessBackends(["triton"])),
]


@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@skipUnlessBackends(["cute"])
@skipUnlessCuteAvailable("Native CuTe handoff requires the supported CuTe DSL")
def test_namedtuple_bundle_replays_in_fresh_process(tmp_path: Path) -> None:
    class Pair(NamedTuple):
        x: torch.Tensor
        scale: float

    @helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
    )
    def scaled_pair(pair):
        x, scale = pair.x, pair.scale
        output = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            output[tile] = x[tile] * scale
        return output

    args = (Pair(torch.randn(128, device=DEVICE), 2.0),)
    bound = scaled_pair.bind(args)
    search = FiniteSearch(
        bound, args, [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])]
    )
    point = find_handoff(search)
    bundle = build_handoff(search, point, tmp_path / "bundle", repetitions=1)
    _replay_bundle(bundle, tmp_path)


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@skipUnlessBackends(["cute"])
@skipUnlessCuteAvailable("Native CuTe handoff requires the supported CuTe DSL")
# The kernel below maps every scale but 2.0 to 1.0, so +-inf replay exactly as
# nan does (a nonfinite constexpr scalar through the metadata and the fresh
# process); test_nonfinite_input_metadata covers the JSON encoding of all three.
@pytest.mark.parametrize("scale", [2.0, math.nan])
def test_dlpack_and_nonfinite_bundle_inputs(scale: float, tmp_path: Path) -> None:
    @helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
    )
    def scaled_input(x: torch.Tensor, scale: float) -> torch.Tensor:
        output = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            # Keep outputs finite so this also checks nonfinite scalar replay.
            output[tile] = x[tile] * (scale if scale == 2.0 else 1.0)
        return output

    base = torch.ones(129, device=DEVICE)
    x = torch.from_dlpack(base[1:])
    args = (x, hl.constexpr(scale))
    bound = scaled_input.bind(args)
    search = FiniteSearch(
        bound, args, [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])]
    )
    point = find_handoff(search)
    torch.testing.assert_close(point.fn(x, scale), x * (scale if scale == 2.0 else 1.0))
    bundle = build_handoff(search, point, tmp_path / "bundle", repetitions=1)
    assert bundle.manifest["baseline_evaluation"]["ok"]
    assert bundle.manifest["cases"][0]["input_metadata"][0]["storage_offset"] == 0
    _replay_bundle(bundle, tmp_path)


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@pytest.mark.parametrize("backend", _LANE_BACKENDS)
@pytest.mark.parametrize("multi_shape", [False, True])
def test_native_source_handoff(backend: str, multi_shape: bool, tmp_path: Path) -> None:
    if backend == "cute" and not _has_cute_dsl():
        pytest.skip("requires the supported CuTe DSL")

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
    expected_timing = "cuda_graph" if torch.version.hip is None else "cuda_event"
    assert manifest["cases"][0]["timing"] == expected_timing
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
    if multi_shape:
        return
    # A replay needs only the workspace; the original search is no longer used.
    replay = HandoffBundle(bundle.directory).evaluate(repetitions=1)
    assert replay.ok, replay
    assert replay.cases[0].source_hash == result.cases[0].source_hash

    # The accuracy-error round below is evaluated per case, the same for the
    # multi-shape bundle (test_handoff_evaluation covers a failed case disabling
    # its aggregate), so it runs once per backend: each round cold-compiles the
    # kernel in a fresh worker.
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

    if backend == "triton":
        with pytest.raises(RuntimeError, match="baseline failed"):
            build_handoff(
                search,
                point,
                tmp_path / "bad_reference",
                reference_fn=lambda x, scale: torch.zeros_like(x),
                repetitions=1,
            )


def _leaky(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, x * 0.5)


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@skipUnlessBackends(["cute"])
@skipUnlessCuteAvailable("Native CuTe handoff requires the supported CuTe DSL")
def test_cute_handoff_with_callable_bool_and_float_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callables (score/mask mods) are compiled into the kernel, so the exported
    entry point omits them; with ``HELION_CUTE_EXPORT_HARDCODE_SCALARS=1`` a
    bool or float the host code branches on is exported as the constant of
    the case, passed (``check``) or left at its default (``scale``), even when
    the bound kernel was bound with another value (bools and floats key bound
    kernels by type), and a float only passed to the launch stays a runtime
    scalar."""
    monkeypatch.setenv("HELION_CUTE_EXPORT_HARDCODE_SCALARS", "1")

    @helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_precompile=None,
        autotune_benchmark_subprocess=False,
    )
    def mapped_add(
        x: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        check: bool = False,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> torch.Tensor:
        if check:
            assert x.size(0) == y.size(0)
        assert scale > 0
        out = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            out[tile] = fn(x[tile]) * scale + y[tile] + shift
        return out

    x = torch.randn(256, device=DEVICE)
    y = torch.randn(256, device=DEVICE)
    unchecked = mapped_add.bind((x, _leaky, y, False, 2.0, 1.0))
    args = (x, _leaky, y, True)
    assert mapped_add.bind(args) is unchecked
    search = FiniteSearch(
        unchecked,
        args,
        [helion.Config(block_sizes=[64]), helion.Config(block_sizes=[128])],
    )
    point = find_handoff(search, HandoffPolicy())
    torch.testing.assert_close(point.fn(*args), torch.where(x > 0, x, x * 0.5) + y)
    bundle = build_handoff(search, point, tmp_path / "bundle", repetitions=1)
    assert bundle.manifest["baseline_evaluation"]["ok"]
    assert len(bundle.manifest["cases"][0]["input_metadata"]) == 3
    source = bundle.sources["case_0/kernel.py"]
    entry = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "mapped_add"
    )
    assert [arg.arg for arg in entry.args.args] == [
        "x",
        "y",
        "check",
        "scale",
        "shift",
    ]
    # The hard-coded scalars default to the case's values.
    assert "check: bool=True, scale: float=1.0, shift: float=0.0" in ast.unparse(
        entry.args
    )
    validate = ast.unparse(entry.body[0])
    assert validate.startswith("_standalone_validate_args((x, y, check, scale, shift),")
    assert validate.endswith(
        "('constant', True), ('constant', 1.0), ('scalar', 'float')))"
    )
    assert "_source_module" not in source
    result = bundle.run_agent(lambda workspace: dict(workspace.sources), repetitions=1)
    assert result.ok, result


# The Triton lane's one unmarked fresh-process replay (see _LANE_BACKENDS).
@pytest.mark.timeout(300)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipIfRefEager("Native handoff requires compilation")
@skipUnlessBackends(["triton"])
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
    _replay_bundle(bundle, tmp_path)
    failed = bundle.run_agent(
        lambda workspace: {
            "case_0/kernel.py": workspace.sources["case_0/kernel.py"]
            + "\n_original = increment\ndef increment(x):\n    return _original(x).clone()\n",
        },
        repetitions=1,
    )
    assert not failed.ok and failed.cases[0].status == "accuracy_error"
    torch.testing.assert_close(args[0], before)
