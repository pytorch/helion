"""CPU checks of helion.flex_attention's FX transforms of score_mod / mask_mod
graphs, on graphs traced with Inductor's decompositions (the form the
lowering receives), of configure(), and of what its FX graph cache key
follows."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Callable
import zipfile

import pytest
import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx

from helion import _compat
from helion._compat import supports_inductor_flex_attention_choices
from helion._compiler.backend_registry import get_backend

if not supports_inductor_flex_attention_choices():
    pytest.skip(
        "helion.flex_attention needs Inductor's flex_attention choice hooks "
        "(PyTorch 2.15 and later)",
        allow_module_level=True,
    )

import helion.flex_attention
from helion.flex_attention import _choices
from helion.flex_attention._lowering import _non_negative

# Importing the package turns the routing on for every compile in this
# process; these tests compile nothing.
helion.flex_attention.disable()

pytestmark = pytest.mark.cpu_only


def _index_graph(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], size: int
) -> tuple[torch.fx.Node, dict[torch.fx.Node, int]]:
    """``fn(q_idx, kv_idx)`` traced as Inductor decomposes it, on int32
    coordinates as flex_attention passes them: its output node and each
    coordinate's largest value (``size - 1``)."""
    gm = make_fx(fn, decomposition_table=select_decomp_table())(
        torch.tensor(3, dtype=torch.int32), torch.tensor(5, dtype=torch.int32)
    )
    coords = gm.graph.find_nodes(op="placeholder")
    return gm.graph.output_node().args[0], dict.fromkeys(coords, size - 1)


# Each case is a distinct op (or op form after decomposition) the analysis
# must classify, or a product near the int32 limit; all trace on CPU in
# milliseconds.
@pytest.mark.parametrize(
    ("fn", "size", "expected"),
    [
        (lambda q, kv: kv // 64, 4096, True),  # div with rounding_mode="floor"
        (lambda q, kv: (q - kv).clamp(0, 255), 4096, True),  # clamp_min, _max
        (lambda q, kv: kv % 64, 4096, True),
        (lambda q, kv: (q * kv).clamp(max=255), 4096, True),
        (lambda q, kv: kv.clamp(max=255), 4096, True),
        (lambda q, kv: kv - q + 4095, 4096, True),  # relative position
        (torch.minimum, 4096, True),
        (lambda q, kv: torch.maximum(q - kv, torch.zeros_like(q)), 4096, True),
        (operator.sub, 4096, False),
        (lambda q, kv: (q - kv).clamp(max=255), 4096, False),
        (lambda q, kv: torch.add(q, kv, alpha=-1), 4096, False),
        (lambda q, kv: torch.add(q + 256, kv, alpha=-1), 256, True),
        (lambda q, kv: kv.to(torch.int8).long(), 4096, False),
        # int32 products that can wrap negative, as eager's do.
        (lambda q, kv: (q * kv).clamp(max=255), 65536, False),
        (lambda q, kv: (kv * 2147483647).clamp(max=255), 4096, False),
    ],
    ids=[
        "floor_division",
        "clamp_both_sides",
        "remainder",
        "product_that_fits",
        "minimum_of_coordinate",
        "relative_position",
        "minimum_of_coordinates",
        "maximum_with_zero",
        "difference",
        "minimum_of_difference",
        "negative_alpha",
        "negative_alpha_offset",
        "narrowing_cast",
        "product_that_overflows",
        "constant_product_that_overflows",
    ],
)
def test_gather_index_non_negative_by_construction(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    size: int,
    expected: bool,
) -> None:
    """A gather index the analysis proves non-negative is read as is; any
    other is wrapped (``where(i < 0, i + n, i)``), as eager reads a negative
    index."""
    index, bounds = _index_graph(fn, size)
    assert _non_negative(index, bounds) is expected


def test_configure_rejects_values_json_cannot_encode() -> None:
    """The FX graph cache key of every compile encodes the configured
    configs, so configure() rejects what JSON cannot encode up front."""
    with pytest.raises(ValueError, match="forward_config must hold JSON values"):
        helion.flex_attention.configure(forward_config={"num_warps": object()})
    helion.flex_attention.configure(forward_config={"block_sizes": (64, 128)})
    try:
        assert _choices.options.forward_config == {"block_sizes": [64, 128]}
    finally:
        helion.flex_attention.configure()


@pytest.mark.parametrize("tolerance", [-1, 0, True, "0.1"])
def test_configure_rejects_invalid_autotune_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="autotune_tolerance must be"):
        helion.flex_attention.configure(autotune_tolerance=tolerance)


def test_cache_key_builds_no_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contributor's uuid() is part of the FX graph cache key of every
    graph compiled after the import, flex_attention or not: it reads the
    environment without building a Helion kernel (a bad
    HELION_AUTOTUNE_EFFORT fails only the flex_attention compiles), and
    follows whether autotuning is disallowed and the HELION_* variables."""
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "bogus")
    choices = _choices.HelionFlexAttentionChoices()
    helion.flex_attention.enable()
    try:
        base = choices.uuid()
        monkeypatch.setenv("HELION_DISALLOW_AUTOTUNING", "1")
        disallowed = choices.uuid()
        monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "quick")
        assert len({base, disallowed, choices.uuid()}) == 3
    finally:
        helion.flex_attention.disable()


def test_non_codegen_env_vars_exist() -> None:
    """Every variable left out of the key is one Helion, its benchmarks or
    its scripts read."""
    root = Path(helion.__file__).parent.parent
    # Not _choices.py, which spells every name in the lists themselves.
    source = "\n".join(
        path.read_text()
        for directory in ("helion", "benchmarks", "scripts")
        for path in (root / directory).rglob("*.py")
        if path != Path(_choices.__file__)
    )
    for name in (*_choices._NON_CODEGEN_ENV_VARS, *_choices._NON_CODEGEN_ENV_PREFIXES):
        assert f'"{name}' in source, name


def test_keyed_env_vars() -> None:
    """HELION_* variables are keyed, but not log and dump paths, cache
    locations, the test harness, the LLM endpoint, or those of settings that
    change no generated code; other variables are not."""
    for name in (
        "HELION_AUTOTUNE_LOG",
        "HELION_CUTE_LAUNCH_SHAPE_LOG",
        "HELION_AUTOTUNE_RANDOM_SEED",
        "HELION_AUTOTUNE_PRECOMPILE",
        "HELION_CACHE_DIR",
        "HELION_REMOTE_CACHE_BACKEND",
        "HELION_TEST_SHARD",
        "HELION_DISALLOW_AUTOTUNING",
        "HELION_LLM_API_KEY",
        "HELION_AUTOREVIEW_SESSION",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert not _choices._keyed_env_var(name), name
    for name in (
        "HELION_CUTE_FLASH_KV_STAGE",
        "HELION_AUTOTUNER",
        "HELION_HEURISTIC_DIR",
        "HELION_LLM_MODEL",
    ):
        assert _choices._keyed_env_var(name), name


def test_cute_names_its_codegen_package() -> None:
    """The DSL package CuTe's generated code calls, which Inductor's key
    leaves out (it keys torch, Triton and CUDA)."""
    assert get_backend("triton").codegen_versions() == ()
    assert [name for name, _ in get_backend("cute").codegen_versions()] == [
        "nvidia-cutlass-dsl"
    ]


def test_source_hash_reads_archives(tmp_path: Path) -> None:
    """The fbcode source key hashes a package the same from a directory and
    from a zip archive, follows each file's contents, and is None without
    source files (the key then stays in this process)."""
    package = tmp_path / "src" / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    (package / "sub" / "mod.py").write_text("y = 2\n")
    (package / "data.txt").write_text("not source\n")
    on_disk = _choices._package_source_hash(package)
    assert on_disk is not None
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in package.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(package.parent).as_posix())
    assert _choices._package_source_hash(zipfile.Path(archive, "pkg/")) == on_disk
    (package / "sub" / "mod.py").write_text("y = 3\n")
    assert _choices._package_source_hash(package) != on_disk
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _choices._package_source_hash(empty) is None


def test_fbcode_source_key_without_source_stays_in_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_choices, "is_fbcode", lambda: True)
    monkeypatch.setattr(_choices.importlib.resources, "files", lambda name: tmp_path)
    _choices._helion_source_key.cache_clear()
    try:
        first = _choices._helion_source_key()
        assert first.startswith("process-")
        _choices._helion_source_key.cache_clear()
        assert _choices._helion_source_key() != first
    finally:
        _choices._helion_source_key.cache_clear()


_UNBOXED = "    tb = TensorBox(mo)\n"
_BOXED = "    tb = TensorBox.create(mo)\n"


# One case per way the check decides: releases and a 2.17 source build by
# version (the source is not read); a 2.16 nightly, release candidate or
# source build by its source, fixed or not; and a 2.16 nightly without
# source by its date, before and after pytorch/pytorch#205348 (None: no
# source).
@pytest.mark.parametrize(
    ("torch_version", "source", "expected"),
    [
        ("2.15.0+cu130", _BOXED, False),
        ("2.16.0", _UNBOXED, True),
        ("2.17.0a0+git0123abc", _UNBOXED, True),
        ("2.16.0.dev20261125+cu130", _UNBOXED, False),
        ("2.16.0.dev20261118+cu130", _BOXED, True),
        ("2.16.0rc1", _BOXED, True),
        ("2.16.0a0+git4e7fb14", _UNBOXED, False),
        ("2.16.0a0+git4e7fb14", _BOXED, True),
        ("2.16.0.dev20261118+cu130", None, False),
        ("2.16.0.dev20261119+cu130", None, True),
    ],
)
def test_inductor_boxes_template_multi_outputs(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
    source: str | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(_compat.torch, "__version__", torch_version)

    def getsource(fn: object) -> str:
        if source is None:
            raise OSError("could not get source code")
        return source

    monkeypatch.setattr(_compat.inspect, "getsource", getsource)
    _compat.inductor_boxes_template_multi_outputs.cache_clear()
    try:
        assert _compat.inductor_boxes_template_multi_outputs() is expected
    finally:
        _compat.inductor_boxes_template_multi_outputs.cache_clear()
