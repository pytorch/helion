from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.pdl_release import ZERO_FILL_KERNEL_PREFIX
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import full_tile_launch_plan
from helion._testing.cute_zero_fill import COLUMN_SUM_CONFIG
from helion._testing.cute_zero_fill import atomic_column_sum
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.compile_cache import disk_cache_key

pytestmark = pytest.mark.cpu_only


def _sequential_loops(x: torch.Tensor) -> torch.Tensor:
    intermediate = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        intermediate[tile] = x[tile] + 1
    hl.barrier()
    for tile in hl.tile(x.size(0)):
        out[tile] = intermediate[tile] * 2
    return out


@pytest.fixture(scope="module")
def bound():
    with cpu_target():
        kernel = helion.kernel(_sequential_loops, backend="cute")
        yield cpu_bind(kernel, (torch.empty(33),))


def _config(**kwargs: object) -> helion.Config:
    return helion.Config(block_sizes=[32, 32], num_threads=[32, 32], **kwargs)


@pytest.fixture(scope="module")
def sources(bound):
    return {
        choice: bound.to_code(
            _config(**({} if choice is None else {"cute_compiler_pipeline": choice}))
        )
        for choice in (None, "default", "opt2")
    }


def test_default_and_opt2_change_only_launch_options(sources):
    default = ast.parse(sources[None])
    assert ast.dump(default) == ast.dump(ast.parse(sources["default"]))
    probe = ast.parse(sources["opt2"])
    observed = []
    for node in ast.walk(probe):
        if isinstance(node, ast.Call):
            for keyword in list(node.keywords):
                if keyword.arg == "cute_compile_options":
                    observed.append(ast.literal_eval(keyword.value))
                    node.keywords.remove(keyword)
    assert observed == ["--opt-level 2"]
    assert ast.dump(probe) == ast.dump(default)


def test_search_reaches_opt2_and_preserves_legacy_seed(bound):
    spec = bound.config_spec
    legacy = _config()
    explicit = _config(cute_compiler_pipeline="default")
    with bound.env:
        generation = ConfigGeneration(spec)
        base = generation.flatten(legacy)
        assert base == generation.flatten(explicit)
        neighbors = generation.coordinate_neighbor_projections(base)
        moves = [move for move in neighbors if move.key == "cute_compiler_pipeline"]
        assert len(moves) == 1
        move = moves[0]
        assert move.outcome == "candidate"
        assert move.config.cute_compiler_pipeline == "opt2"
        assert move.flat_values != base
        assert move.config != generation.canonicalize_flat(base)[1]
        with patch.object(spec, "compiler_seed_configs", [legacy]):
            pairs = generation.seed_flat_config_pairs()
            assert len(pairs) == 1
            assert pairs[0][1].cute_compiler_pipeline == "default"
            assert spec.compiler_seed_configs == [legacy]
    assert "cute_compiler_pipeline" not in legacy


@pytest.mark.parametrize("value", ["opt3", "opt2_no_licm", 2, False])
def test_invalid_pipeline_is_rejected_or_repaired(bound, value):
    config = _config(cute_compiler_pipeline=value)
    with bound.env:
        with pytest.raises(InvalidConfig, match="cute_compiler_pipeline"):
            bound.config_spec.normalize(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
    assert config.cute_compiler_pipeline == "default"


def test_old_config_remains_sparse(bound):
    config = _config()
    with bound.env:
        bound.config_spec.normalize(config)
    assert config.cute_compiler_pipeline == "default"
    assert "cute_compiler_pipeline" not in config
    assert helion.Config.from_json(config.to_json()) == config


def test_opt2_composes_with_existing_options(bound):
    # Keep SDK-specific integration imports out of ordinary codegen collection.
    from cutlass.base_dsl.compiler import EnableTVMFFI
    from cutlass.base_dsl.compiler import OptLevel
    from cutlass.base_dsl.compiler import PtxasOptions
    from cutlass.base_dsl.compiler import _parse_compile_options_from_str

    config = _config(
        cute_compiler_pipeline="opt2",
        advanced_controls_file="control.txt",
    )
    code = ast.parse(bound.to_code(config))
    options = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(code)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "cute_compile_options"
    ]
    (option,) = options
    # The runtime adds TVM FFI to this same complete string.
    parsed = _parse_compile_options_from_str(option + " --enable-tvm-ffi")
    assert parsed.options[OptLevel].value == 2
    assert parsed.options[EnableTVMFFI].value is True
    assert "--apply-controls control.txt" in parsed.options[PtxasOptions].value


@pytest.mark.parametrize(
    ("captured_options", "expected_default", "expected_opt2"),
    [("", 3, 2), ("opt-level=1", 1, 1), ("use-loop-invariant-pass=false", 3, 2)],
)
def test_sdk_captured_override_preserves_unrelated_options(
    captured_options, expected_default, expected_opt2
):
    from cutlass.base_dsl.compiler import ExtraCompilerOpts
    from cutlass.base_dsl.compiler import OptLevel
    from cutlass.base_dsl.compiler import _parse_compile_options_from_str
    from cutlass.base_dsl.env_manager import EnvironmentVarManager

    # This pins the installed SDK behavior, including its current default level
    # of 3; Helion does not promise that default across SDK versions.
    # The SDK snapshots environment settings; current os.environ is not that
    # state. Model its captured value and exercise the actual option application.
    environment = EnvironmentVarManager("CUTE_DSL")
    environment.compiler_opt = captured_options
    default = _parse_compile_options_from_str("--enable-tvm-ffi")
    probe = _parse_compile_options_from_str("--enable-tvm-ffi --opt-level 2")
    probe.apply_envar_settings(environment, "test_pipeline")
    default.apply_envar_settings(environment, "test_pipeline")
    assert int(probe.options[OptLevel].value) == expected_opt2
    assert int(default.options[OptLevel].value) == expected_default
    for options in (default, probe):
        assert (
            "use-loop-invariant-pass=false" in options.options[ExtraCompilerOpts].value
        ) == (captured_options == "use-loop-invariant-pass=false")
    # Fresh options do not inherit another compilation's environment override.
    assert (
        _parse_compile_options_from_str("--enable-tvm-ffi").options[OptLevel].value == 3
    )


def test_pipeline_changes_disk_key_without_changing_kernel():
    kernel = SimpleNamespace(_helion_cute_source_hash="same-source")
    keys = [
        disk_cache_key(kernel, (), (32, 1, 1), (), None, option)
        for option in (None, "--opt-level 2")
    ]
    assert None not in keys
    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    ("extra", "accepted"),
    [
        (", cute_compile_options='--opt-level 2'", True),
        (", cute_compile_options=effect(input_tensor)", False),
        (", cute_compile_options=runtime_options", False),
        (", **runtime_keywords", False),
    ],
)
def test_compile_option_keywords_keep_host_effect_proof(extra, accepted):
    assert (full_tile_launch_plan(block="(4, 1, 1)" + extra) is not None) is accepted


def test_zero_fill_launch_preserves_compile_options() -> None:
    with cpu_target():
        bound = cpu_bind(atomic_column_sum, (torch.empty(1024, 64),))
        config = helion.Config(
            **COLUMN_SUM_CONFIG,
            cute_pdl=True,
            advanced_controls_file="compile-controls.txt",
        )
        source = bound.to_code(config)
    launches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert len(launches) == 2
    assert any(
        ast.unparse(node.args[0]).startswith(ZERO_FILL_KERNEL_PREFIX)
        for node in launches
    )
    for launch in launches:
        options = [
            ast.literal_eval(keyword.value)
            for keyword in launch.keywords
            if keyword.arg == "cute_compile_options"
        ]
        assert options == ["--ptxas-options='--apply-controls compile-controls.txt'"]
