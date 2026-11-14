"""The ordinary PDL entry wait dominates paged attention's scalar prelude."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest

import helion
from helion._compiler.cute import paged_attention_codegen
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.pdl_release import generic_release_rendered
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import WAIT_CALL
from helion._testing.cute_pdl import kernel_definition
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@functools.cache
def _bound(per_key: bool, requests: int = 2) -> BoundKernel[object]:
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(
            cast("Kernel[object]", paged_online_attention),
            (
                *paged_attention_args(rows=8 if per_key else 64, requests=requests),
                None,
                per_key,
            ),
        )


@pytest.mark.parametrize(
    "per_key,paired", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("pdl", [False, True])
def test_paged_attention_uses_generic_entry_pdl(per_key, paired, pdl):
    bound = _bound(per_key)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[8 if per_key else 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
                cute_min_blocks_per_mp=1,
                cute_pdl=pdl,
            )
        )
        assert config.get("cute_mma_dag_pass_warps") == (
            "attention_paired" if paired else "attention"
        )
        assert config.get("cute_pdl") is pdl
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    wait = WAIT_CALL
    release = "cute.arch.griddepcontrol_launch_dependents()"
    calls = [
        ast.unparse(node) for node in ast.walk(kernel) if isinstance(node, ast.Call)
    ]
    assert ("._helion_cute_use_pdl = True" in source) is pdl
    assert calls.count(wait) == int(pdl)
    assert calls.count(release) == int(pdl)
    assert any("paged_attention" in call and "_body(" in call for call in calls)
    if pdl:
        # This precedes even the scalar tensor loads generated for lengths
        # and bounds, not just the opaque asynchronous attention body call.
        assert ast.unparse(kernel.body[0]) == wait
        assert "capture_pdl_wait_mode" not in source
        assert "capture_pdl_contract" not in source
        assert ast.unparse(kernel.body[1]) == release
        assert any(call.startswith("_helion_mma_dag.ldg_s32(") for call in calls)


@pytest.mark.parametrize("requests", [75, 148, 150])
@pytest.mark.parametrize("release", [None, "last_wave"])
def test_paired_pdl_counts_physical_ctas(requests, release):
    # Two item/head tiles form each physical CTA. Check below, at and above
    # one resident wave for both automatic and explicit release placement.
    bound = _bound(True, requests)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_release=release,
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    module = ast.parse(source)
    (launch,) = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    ]
    assert eval(
        compile(ast.Expression(launch.args[1]), "<paired-grid>", "eval"),
        {"__builtins__": {}},
    ) == (requests,)
    kernel = kernel_definition(source)
    guards = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("pdl_cta >=")
    ]
    if requests <= 148:
        assert not guards
        assert ast.unparse(kernel.body[1]) == (
            "cute.arch.griddepcontrol_launch_dependents()"
        )
    else:
        (guard,) = guards
        assert ast.unparse(guard.test) == f"pdl_cta >= {requests - 148}"
        assert ast.unparse(guard.body[0]) == (
            "cute.arch.griddepcontrol_launch_dependents()"
        )
    # The wait must still dominate both the release and all input loads.
    assert ast.unparse(kernel.body[0]) == WAIT_CALL
    assert "capture_pdl_wait_mode" not in source


@pytest.mark.parametrize(
    "paired,pdl,overlap,release,active",
    [
        (True, True, True, "auto", True),
        (True, True, True, "entry", True),
        (True, True, False, "auto", False),
        (True, False, True, "auto", False),
        (False, True, True, "auto", False),
        (True, True, True, "exit", False),
        (True, True, True, "loader_tail", True),
        (True, True, False, "loader_tail", False),
        (False, True, True, "loader_tail", False),
        (True, False, True, "loader_tail", False),
    ],
)
def test_capture_overlap_is_an_independent_proved_choice(
    paired, pdl, overlap, release, active
):
    # The ordinary enable/disable gates and the tail's independent overlap,
    # family and launch gates need distinct counterexamples, not a product.
    bound = _bound(True)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        config = helion.Config(
            block_sizes=[8],
            cute_mma_dag="tcgen05",
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
            cute_min_blocks_per_mp=1,
            cute_pdl=pdl,
            cute_pdl_release=release,
            cute_pdl_capture_overlap=overlap,
        )
        if not active and (overlap or release == "loader_tail"):
            # Explicit configs are strict. Only search repair may discard an
            # inactive choice before the rendered-source checks below.
            with pytest.raises(helion.exc.InvalidConfig, match="capture_overlap"):
                bound.config_spec.normalized_config(config)
            assert config.get("cute_pdl") is pdl
            assert config.get("cute_pdl_release") == release
            assert config.get("cute_pdl_capture_overlap") is overlap
            bound.config_spec.normalize(config, _fix_invalid=True)
        else:
            config = bound.config_spec.normalized_config(config)
        if active and release == "loader_tail":
            from helion.autotuner.config_generation import ConfigGeneration

            generation = ConfigGeneration(bound.config_spec)
            _, round_trip = generation.canonicalize_flat(generation.flatten(config))
            assert round_trip.get("cute_pdl_release") == "loader_tail"
            assert round_trip.get("cute_pdl_capture_overlap") is True
    assert config.get("cute_pdl_capture_overlap", False) is active
    if not active and (not paired or not pdl or release == "exit"):
        assert "cute_pdl_capture_overlap" not in config
    else:
        assert "cute_pdl_capture_overlap" in config
    tail = active and release == "loader_tail"
    if release == "loader_tail":
        assert config.get("cute_pdl_release", "auto") == (
            "loader_tail" if tail else "auto"
        )
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    kernel = kernel_definition(source)
    assert ("capture_pdl_contract" in source) is active
    assert ("capture_pdl_parameter_sizes" in source) is active
    assert ("capture_pdl_wait_mode" in source) is active
    assert ("capture_release_tail=True" in source) is tail
    if active:
        assert ast.unparse(kernel.body[0].test) == "capture_pdl_wait_mode == 0"
        assert ast.unparse(kernel.body[0].body[0]) == WAIT_CALL
        # The EARLY anchor finishes waiting before either release location.
        release_call = "cute.arch.griddepcontrol_launch_dependents()"
        if tail:
            assert release_call not in ast.unparse(kernel)
        else:
            assert ast.unparse(kernel.body[1]) == release_call
        assert kernel.args.args[-1].arg == "capture_pdl_wait_mode"
        assert ast.unparse(kernel.args.args[-1].annotation) == "cutlass.Int32"
        (plans,) = [
            ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_helion_cute_wrapper_plans"
        ]
        assert plans[0]["capture_pdl_parameter_sizes"] == (128, 128, 128, 128, 4)
        alias = plans[1]
        assert {
            name
            for name, loaded in zip(
                alias["tensor_names"], alias["capture_loaded"], strict=True
            )
            if loaded
        } == {"q", "k", "v", "k_scale", "v_scale", "pages", "lengths"}
        assert {
            name
            for name, stored in zip(alias["tensor_names"], alias["stored"], strict=True)
            if stored
        } == {"out", "lse"}
        assert generic_release_rendered(source) is not tail
        assert CuteBackend.finalist_preference(config, lambda: source) == (
            2 if tail else 1
        )
        # A conditional wait without this exact audited contract is not ranked
        # as the generic entry release. Nor is its opposite wait-mode branch.
        assert not generic_release_rendered(
            source.replace("paired-output-stores-v1", "unknown-contract")
        )
        assert not generic_release_rendered(
            source.replace("capture_pdl_wait_mode == 0", "capture_pdl_wait_mode != 0")
        )
    elif pdl:
        assert ast.unparse(kernel.body[0]) == WAIT_CALL


def test_capture_overlap_surface_and_unavailable_normalization():
    from helion._compiler.cute.autotune.search_surface import (
        _normalize_cute_pdl_capture_overlap,
    )
    from helion._compiler.cute.autotune.search_surface import pdl_fields
    from helion.autotuner.config_spec import ConfigSpec

    spec = _bound(True).config_spec
    fields = {}
    with cpu_render_target(fake_device_sm_count=148), _bound(True).env:
        pdl_fields(spec, fields)
    field = fields["cute_pdl_capture_overlap"]
    assert field.choices == (False, True)
    assert field.default() is False
    assert field._active_choices() == (False,)
    assert fields["cute_pdl_release"]._active_choices() == ("auto",)
    assert fields["cute_pdl_release"].default() == "auto"
    # Unmatched and multi-root bindings never receive the availability proof.
    unavailable = ConfigSpec(backend=_bound(True).env.backend)
    for selected in (False, True):
        config = {
            "cute_mma_dag": "tcgen05",
            "cute_mma_dag_pass_warps": "attention_paired",
            "cute_pdl": True,
            "cute_pdl_capture_overlap": selected,
        }
        if selected:
            with pytest.raises(helion.exc.InvalidConfig, match="capture_overlap"):
                _normalize_cute_pdl_capture_overlap(
                    unavailable, config, fix_invalid=False
                )
            assert config["cute_pdl_capture_overlap"] is True
            _normalize_cute_pdl_capture_overlap(unavailable, config, fix_invalid=True)
        else:
            _normalize_cute_pdl_capture_overlap(unavailable, config, fix_invalid=False)
        assert "cute_pdl_capture_overlap" not in config
    with pytest.raises(helion.exc.InvalidConfig, match="must be a boolean"):
        _normalize_cute_pdl_capture_overlap(
            spec, {"cute_pdl_capture_overlap": 1}, fix_invalid=False
        )


def test_capture_overlap_spread_is_rejected_before_generic_pad_insertion():
    bound = _bound(True)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        cute_pdl_spread=True,
    )
    # This family already owns a shared-memory arena. Its contract rejects
    # generic padding before the pad insertion pass sees a conditional wait.
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.InvalidConfig, match="no PDL shared-memory padding"),
    ):
        bound.to_code(config)


def test_multiple_roots_do_not_offer_capture_overlap():
    from helion._compiler.cute.autotune.search_surface import pdl_fields
    from helion._testing.cute_paged_attention import ordered_paged_online_attention

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(ordered_paged_online_attention(), paged_attention_args())
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.root_ids) > 1
    assert not bound.config_spec.cute_pdl_capture_overlap_available
    fields = {}
    with cpu_render_target(fake_device_sm_count=148), bound.env:
        pdl_fields(bound.config_spec, fields)
    assert "cute_pdl" in fields
    assert "cute_pdl_capture_overlap" not in fields
    assert fields["cute_pdl_release"]._active_choices() == ("auto",)


def test_declined_final_capture_effect_proof_restores_automatic_pdl(monkeypatch):
    original = paged_attention_codegen._is_call

    def reject_store_effect(node, target):
        # Simulate an effect proof that no longer recognizes the output store
        # after normalization accepted loader-tail release. The body stays valid.
        return target is not memory_ops.store and original(node, target)

    monkeypatch.setattr(paged_attention_codegen, "_is_call", reject_store_effect)
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        cute_pdl=True,
        cute_pdl_capture_overlap=True,
        cute_pdl_release="loader_tail",
    )
    with cpu_render_target(fake_device_sm_count=148):
        source = _bound(True).to_code(config)
    kernel = kernel_definition(source)
    assert ast.unparse(kernel.body[0]) == WAIT_CALL
    assert ast.unparse(kernel.body[1]) == (
        "cute.arch.griddepcontrol_launch_dependents()"
    )
    assert "capture_pdl_wait_mode" not in source
    assert "capture_release_tail=True" not in source
    assert "capture_pdl_contract" not in source
