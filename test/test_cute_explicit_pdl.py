"""Explicit hl.pdl_* markers retain placement independently of cute_pdl tuning."""

from __future__ import annotations

import ast
from typing import cast

import pytest
import torch

import helion
import helion._compiler.cute.autotune.heuristics as cute_heuristics
from helion._compiler.cute.autotune.heuristics.cute import _dependent_launch
from helion._compiler.cute.grouped_row_union import LEGACY_SCHEDULE
from helion._compiler.cute.grouped_row_union import PAIRED_CLC_SCHEDULE
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED_SCHEDULE
from helion._compiler.cute.grouped_row_union import physical_schedule
from helion._compiler.cute.kernel_generators import KERNEL_GENERATORS
from helion._compiler.cute.state import cute_env_state
from helion._compiler.cute.tcgen05_config.surface import optional_fragments
from helion._compiler.cute.tcgen05_config.surface import restrict_cluster_m_search
from helion._compiler.emitter_protocols import KernelTransforms
from helion._testing import cpu_render_target
from helion._testing import default_cute_mma_support
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _entry(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _exit(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    return out


def _render(
    kernel: helion.Kernel,
    capability: tuple[int, int] | None = (10, 0),
    **config: object,
) -> str:
    with cpu_render_target(capability=capability):
        bound = cpu_bind(kernel, (torch.empty(256),), capability=capability)
        return bound.to_code(
            helion.Config.from_dict(
                {"block_sizes": [256], "num_threads": 128, **config}
            )
        )


def _device_body(code: str) -> list[ast.stmt]:
    (definition,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    ]
    return definition.body


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_entry_order_and_programmatic_launch(capability: tuple[int, int]) -> None:
    code = _render(_entry, capability)
    body = _device_body(code)
    calls = [ast.unparse(stmt) for stmt in body]
    release = calls.index("cute.arch.griddepcontrol_launch_dependents()")
    wait = calls.index("cute.arch.griddepcontrol_wait()")
    assert release < wait
    assert code.count("griddepcontrol_") == 2
    assert "._helion_cute_use_pdl = True" in code
    assert "tl.extra.cuda" not in code
    # The explicit wait precedes every generated load/store, even if a pass
    # hoists harmless register or shared allocations above the markers.
    for stmt in body[:wait]:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"load", "store", "copy", "atomic_add"}


@pytest.mark.parametrize("cute_pdl", [False, True])
def test_explicit_exit_overrides_automatic_release(cute_pdl: bool) -> None:
    code = _render(_exit, cute_pdl=cute_pdl, cute_pdl_release="entry")
    body = _device_body(code)
    assert ast.unparse(body[-1]) == "cute.arch.griddepcontrol_launch_dependents()"
    assert code.count("griddepcontrol_wait()") == 1
    assert code.count("griddepcontrol_launch_dependents()") == 1
    assert "._helion_cute_use_pdl = True" in code


@pytest.mark.parametrize("capability", [(8, 0), None])
def test_unsupported_targets_remain_stream_ordered(
    capability: tuple[int, int] | None,
) -> None:
    code = _render(_entry, capability)
    assert "griddepcontrol" not in code
    assert "_helion_cute_use_pdl" not in code
    assert "tl.extra.cuda" not in code


def test_automatic_pdl_normalizes_to_one_search_choice() -> None:
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", _entry), (torch.empty(256),))
        base = helion.Config(block_sizes=[256], num_threads=128)
        requested = helion.Config.from_dict(
            {
                **base.config,
                "cute_pdl": True,
                "cute_pdl_release": "exit",
                "cute_pdl_spread": True,
            }
        )
        assert bound.config_spec.normalized_config(requested) == (
            bound.config_spec.normalized_config(base)
        )
        fields = bound.config_spec._flat_fields()
        for name in ("cute_pdl", "cute_pdl_spread"):
            field = fields[name]
            assert isinstance(field, EnumFragment)
            assert field.search_choices == (False,)
        # The caller's saved configuration remains untouched.
        assert requested.config["cute_pdl"] is True


def test_search_avoids_unsupported_native_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", _entry), (torch.empty(256),))
        spec = bound.config_spec
        # These facts normally come from independent structural proofs. The
        # explicit protocol must prune their native schedules even when proved.
        monkeypatch.setattr(spec, "cute_materialized_schedule_search_enabled", True)
        monkeypatch.setattr(
            spec, "cute_materialized_operand_schedule_search_enabled", True
        )
        monkeypatch.setattr(spec, "cute_sequence_reduction_blocks", {0})
        fields = spec._flat_fields()
        for name, choices in (
            ("cute_materialized_schedule", ("off",)),
            ("cute_materialized_operand_schedule", ("off",)),
            ("cute_reduction_sequence", ("scalar", "reload", "resident")),
        ):
            field = fields[name]
            assert isinstance(field, EnumFragment)
            assert field.search_choices == choices
            assert len(field.choices) > len(choices)  # Pins remain diagnosable.
        assert _dependent_launch(spec) == {}
        monkeypatch.setattr(
            spec.tcgen05, "grouped_prefix_adaptive_n_search_enabled", True
        )
        adaptive = optional_fragments(spec.tcgen05, for_search=True)[
            "tcgen05_grouped_prefix_adaptive_n"
        ]
        assert isinstance(adaptive, EnumFragment)
        assert adaptive.search_choices == (False,)


def test_native_seed_screen_matches_generator_registry() -> None:
    # Every config-selected generator needs a witness, including choices that
    # enter through seeds or coverage rather than random search fragments.
    witnesses = {
        "grouped_prefix_adaptive_n": {"tcgen05_grouped_prefix_adaptive_n": True},
        "packed_operand": {"cute_materialized_operand_schedule": "warp_narrow4"},
        "row_resident": {"cute_materialized_schedule": "warp_rows2"},
        "grouped_warp_tf32": {"cute_mma_f32_conversion": "warp_raw"},
        "flat_grouped_native": {"cute_grouped_rna_warps": 4},
        "bounded_cache": {"cute_reduction_sequence": "bounded"},
        "split_k_cluster": {"cute_split_k_schedule": "cluster8_k4"},
        "split_k_workspace": {"cute_split_k_workspace": True},
    }
    assert set(witnesses) == {g.name for g in KERNEL_GENERATORS} - {
        # These generators require an actual bound source/dependency proof,
        # rather than a Config knob. The dependent-phase module binds its
        # real witness and checks normal ordered, non-PDL emission.
        "dependent_phases",
        "materialized_fission",
        "ordered_phases",
    }
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", _entry), (torch.empty(256),))
        backend = bound.env.backend
        host_function = bound.host_function
        assert host_function is not None
        assert backend.autotune_config_is_viable(bound.config_spec, helion.Config())
        for generator in KERNEL_GENERATORS:
            if generator.name not in witnesses:
                continue
            config = helion.Config.from_dict(witnesses[generator.name])
            assert generator.applies(host_function, config, KernelTransforms())
            assert not backend.autotune_config_is_viable(bound.config_spec, config)


def test_explicit_pdl_excludes_two_cta_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", _entry), (torch.empty(256),))
        state = bound.config_spec.tcgen05
        restrict_cluster_m_search(state, (1, 2))
        assert state.cluster_m_search_choices == (1,)
        # Independent pairing proofs must not widen the explicit protocol.
        monkeypatch.setattr(state, "full_tile_added_pairing", True)
        fields = optional_fragments(state, for_search=True)
        for name, choices in (
            ("tcgen05_cluster_m", (1,)),
            ("tcgen05_cluster_n", (1,)),
            ("tcgen05_cta_group", ("auto",)),
        ):
            field = fields[name]
            assert isinstance(field, EnumFragment)
            assert field.choices == choices
        for config in (
            helion.Config(tcgen05_cluster_m=2),
            helion.Config(tcgen05_cta_group="two"),
        ):
            assert not bound.env.backend.autotune_config_is_viable(
                bound.config_spec, config
            )


@pytest.mark.parametrize("capability", [(8, 0), (10, 0)])
@pytest.mark.parametrize("clustered", [False, True])
def test_promoted_defaults_respect_explicit_pdl(
    capability: tuple[int, int], clustered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with cpu_render_target(capability=capability):
        bound = cpu_bind(
            cast("helion.Kernel[object]", _entry),
            (torch.empty(256),),
            capability=capability,
        )
        spec = bound.config_spec
        promoted = helion.Config(
            block_sizes=[256], num_threads=128, tcgen05_cluster_m=2 if clustered else 1
        )
        spec.compiler_default_config = promoted
        monkeypatch.setattr(
            cute_heuristics,
            "postprocess_seed_configs",
            lambda env, ir, configs: configs,
        )
        assert bound.host_function is not None
        configs = bound.env.backend.postprocess_seed_configs(
            bound.env, bound.host_function.device_ir, [promoted]
        )
        if capability >= (9, 0) and clustered:
            assert configs == []
            assert spec.compiler_default_config is None
            code = bound.to_code(spec.default_config())
            assert code.count("griddepcontrol_wait()") == 1
            assert code.count("griddepcontrol_launch_dependents()") == 1
        else:
            assert configs == [promoted]
            assert spec.compiler_default_config is promoted


@pytest.mark.parametrize(
    "schedule", [LEGACY_SCHEDULE, TRANSPOSED_SCHEDULE, PAIRED_CLC_SCHEDULE]
)
def test_projected_clustered_row_union_seeds_are_screened(schedule: str) -> None:
    config = helion.Config.from_dict(
        {
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_grouped_dense_row_union": True,
            SCHEDULE_KEY: schedule,
        }
    )
    profile = physical_schedule(config)
    if profile is not None:
        # The public carrier stays single-CTA until codegen applies this
        # physical profile. Screening must inspect the schedule itself.
        assert config.get("tcgen05_cluster_m") == 1
        assert profile.codegen_values()["tcgen05_cluster_m"] == 2
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", _entry), (torch.empty(256),))
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, config
        ) == (profile is None)


def _fission_with_markers(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column] + 1
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * 2
    hl.pdl_launch_dependents()
    return out


def _fission_without_markers(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        for column in hl.tile(columns):
            middle[row, column] = x[row, column] + 1
        for column in hl.tile(columns):
            out[row, column] = middle[row, column] * 2
    return out


def test_explicit_pdl_keeps_fission_on_ordinary_route() -> None:
    kernel = helion.kernel(
        _fission_with_markers,
        backend="cute",
        static_shapes=True,
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
    )
    with cpu_render_target():
        bound = cpu_bind(cast("helion.Kernel[object]", kernel), (torch.empty(13, 37),))
        assert cute_env_state(bound.env).fission_plan is None
        code = bound.to_code(
            helion.Config(block_sizes=[16, 64, 64], pid_type="persistent_blocked")
        )
    assert code.count("griddepcontrol_wait()") == 1
    assert code.count("griddepcontrol_launch_dependents()") == 1
    # This exact sibling-loop shape does fission when the explicit markers
    # are absent, so the negative control cannot pass through another gate.
    ordinary = helion.kernel(
        _fission_without_markers,
        backend="cute",
        static_shapes=True,
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
    )
    with cpu_render_target():
        control = cpu_bind(
            cast("helion.Kernel[object]", ordinary), (torch.empty(13, 37),)
        )
        assert cute_env_state(control.env).fission_plan is not None


def test_explicit_pdl_allows_unused_fission_setting() -> None:
    kernel = helion.kernel(
        _entry.fn,
        backend="cute",
        static_shapes=True,
        cute_region_fission=True,
    )
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(256),))
        assert cute_env_state(bound.env).fission_plan is None
        code = bound.to_code(helion.Config(block_sizes=[256], num_threads=128))
    assert "griddepcontrol_wait()" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    hl.pdl_wait()
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    hl.pdl_launch_dependents()
    return out


@pytest.mark.parametrize("persistent", [False, True])
def test_explicit_tcgen05_exit_overrides_automatic_role_releases(
    persistent: bool,
) -> None:
    with cpu_render_target(
        mma_support=default_cute_mma_support(), fake_device_sm_count=148
    ):
        bound = cpu_bind(
            cast("helion.Kernel[object]", _gemm),
            (torch.empty(1024, 1024, dtype=torch.float16),) * 2,
        )
        config = helion.Config(
            block_sizes=[128, 128, 64] if persistent else [64, 16, 128],
            pid_type="persistent_blocked" if persistent else "flat",
            indexing=["tensor_descriptor" if persistent else "pointer"] * 3,
            tcgen05_ab_stages=4 if persistent else 2,
            cute_pdl=True,
            cute_pdl_release="entry",
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
        assert bound.config_spec.default_config().config["cute_pdl"] is False
        adaptive = optional_fragments(bound.config_spec.tcgen05)[
            "tcgen05_grouped_prefix_adaptive_n"
        ]
        assert isinstance(adaptive, BooleanFragment)  # Validation stays strict.
    assert "tcgen05" in code
    assert code.count("griddepcontrol_wait()") == 1
    assert code.count("griddepcontrol_launch_dependents()") == 1
    assert ast.unparse(_device_body(code)[-1]) == (
        "cute.arch.griddepcontrol_launch_dependents()"
    )
    assert "._helion_cute_use_pdl = True" in code


def test_native_whole_kernel_rejected_without_losing_markers() -> None:
    with cpu_render_target(
        mma_support=default_cute_mma_support(), fake_device_sm_count=148
    ):
        bound = cpu_bind(
            cast("helion.Kernel[object]", _gemm),
            (
                torch.empty(128, 16384, dtype=torch.float16),
                torch.empty(16384, 128, dtype=torch.float16),
            ),
        )
        with pytest.raises(helion.exc.InvalidConfig, match="explicit PDL markers"):
            bound.to_code(
                helion.Config(
                    block_sizes=[128, 128, 64],
                    cute_split_k_workspace=True,
                    cute_split_k_partitions=4,
                )
            )
        ordinary = bound.to_code(helion.Config(block_sizes=[64, 16, 128]))
    assert ordinary.count("griddepcontrol_wait()") == 1
    assert ordinary.count("griddepcontrol_launch_dependents()") == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gemv(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m, k = w.size()
    x2 = x.view(k, 1)
    out = torch.empty((m, 1), dtype=x.dtype, device=x.device)
    hl.pdl_wait()
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 1], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_m, tile_k], x2[tile_k, :], acc=acc)
        out[tile_m, :] = acc
    hl.pdl_launch_dependents()
    return out


def test_ring_mma_preserves_explicit_exit_after_epilogue() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            cast("helion.Kernel[object]", _gemv),
            (
                torch.empty(256, 4096, dtype=torch.float16),
                torch.empty(4096, dtype=torch.float16),
            ),
        )
        code = bound.to_code(
            helion.Config(
                block_sizes=[64, 256],
                num_threads=[1, 128],
                cute_vector_widths=[1, 2],
                cute_dot_ring_mma=True,
                cute_dot_ring_stages=3,
                cute_dot_ring_split="k",
                cute_pdl=True,
            )
        )
    assert "_cute_ring_ldmatrix_x4(" in code
    body = _device_body(code)
    assert ast.unparse(body[0]) == "cute.arch.griddepcontrol_wait()"
    assert ast.unparse(body[-1]) == "cute.arch.griddepcontrol_launch_dependents()"
    assert code.count("griddepcontrol_launch_dependents()") == 1
