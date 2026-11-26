"""CPU checks of helion.flex_attention's FX transforms of score_mod / mask_mod
graphs, on graphs traced with Inductor's decompositions (the form the
lowering receives), of configure(), and of what its FX graph cache key
follows."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Callable

from packaging.version import Version
import pytest
import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx

from helion._compat import supports_inductor_flex_attention_backends

if not supports_inductor_flex_attention_backends():
    pytest.skip(
        "helion.flex_attention needs a PyTorch whose Inductor lets a "
        "contributor lower a whole flex_attention call "
        "(InductorChoices.lower_flex_attention)",
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
    """The FX graph cache key of flex_attention graphs encodes the configured
    configs, so configure() rejects what JSON cannot encode up front."""
    with pytest.raises(ValueError, match="forward_config must hold JSON values"):
        helion.flex_attention.configure(forward_config={"num_warps": object()})
    helion.flex_attention.configure(forward_config={"block_sizes": (64, 128)})
    try:
        assert _choices.options.forward_config == {"block_sizes": [64, 128]}
    finally:
        helion.flex_attention.configure()


@pytest.mark.parametrize("tolerance", [-1, 0, float("nan"), True, "0.1"])
def test_configure_rejects_invalid_autotune_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="autotune_tolerance must be"):
        helion.flex_attention.configure(autotune_tolerance=tolerance)


def test_cache_key_builds_no_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contributor's key joins the FX graph cache key of every graph that
    calls flex_attention, Helion's or not: it reads the environment without
    building a Helion kernel (a bad HELION_AUTOTUNE_EFFORT fails only the
    flex_attention calls that pass Helion's device and dtype checks, not
    every compile), and follows whether autotuning is disallowed and the
    HELION_* variables. Its uuid(), in every graph's key, is constant."""
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "bogus")
    choices = _choices.HelionFlexAttentionChoices()
    base = choices.flex_attention_cache_key()
    monkeypatch.setenv("HELION_DISALLOW_AUTOTUNING", "1")
    disallowed = choices.flex_attention_cache_key()
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "quick")
    assert len({base, disallowed, choices.flex_attention_cache_key()}) == 3
    assert choices.uuid() == "helion.flex_attention"
    assert choices.flex_attention_backends() == ("HELION",)


def test_backend_options_override_configure() -> None:
    """A BACKEND="HELION" call's BACKEND_OPTIONS replace configure()'s
    settings they name, checked as configure() checks them."""
    helion.flex_attention.configure(backward_config={"num_warps": 4})
    try:
        options = _choices.call_options(
            {"forward_config": {"block_sizes": (64, 128)}, "autotune_tolerance": 1}
        )
        assert options.forward_config == {"block_sizes": [64, 128]}
        assert options.backward_config == {"num_warps": 4}
        assert options.autotune_tolerance == 1.0
        assert _choices.call_options({}) == _choices.options
    finally:
        helion.flex_attention.configure()
    with pytest.raises(ValueError, match=r"takes .*got \['tile'\]"):
        _choices.call_options({"tile": 64})
    with pytest.raises(ValueError, match="autotune_tolerance must be"):
        _choices.call_options({"autotune_tolerance": 0})
    with pytest.raises(ValueError, match="must be a dict of config knobs"):
        _choices.call_options({"forward_config": [1, 2]})


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


def test_cache_key_follows_the_cute_dsl_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CuTe DSL the generated code calls, which Inductor's key leaves out
    (it keys torch, Triton and CUDA)."""
    keys = set()
    for version in ("4.3.0", "4.4.0"):
        monkeypatch.setattr(
            _choices, "installed_cute_dsl_version", lambda v=version: Version(v)
        )
        keys.add(_choices.cache_key())
    assert len(keys) == 2
