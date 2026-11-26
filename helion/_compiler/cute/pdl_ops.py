"""Explicit PDL markers: host no-ops, rendered at device entry or exit."""

from __future__ import annotations

import ast
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ...language import _decorators
from ...language.pdl_ops import pdl_launch_dependents
from ...language.pdl_ops import pdl_wait
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from .autotune.config_keys import COMPACT_ROWS_KEY
from .grouped_row_union import physical_schedule
from .state import cute_codegen_state

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ..device_function import DeviceFunction
    from ..emitter_protocols import KernelGenerator
    from ..emitter_protocols import KernelTransforms
    from ..host_function import HostFunction
    from ..inductor_lowering import CodegenState


def has_explicit_pdl() -> bool:
    """Type propagation records markers only for PDL-capable targets."""
    return CompileEnvironment.current().config_spec.has_explicit_pdl


def config_conflicts_with_required_owner(spec: ConfigSpec, config: Config) -> bool:
    """Reject native schedules that cannot issue a required PDL or assertion effect.

    Config-selected generators are screened here; structural fission and
    ordered-phase launches are checked during planning or codegen. Only PDL
    excludes clustered tcgen05 schedules for their intrinsic release protocol.
    """
    from .scalar_fragment_assertion import conflicting_fragment_schedule
    from .scalar_fragment_assertion import requires_assertion_owner

    assertion = requires_assertion_owner(spec)
    if not spec.has_explicit_pdl and not assertion:
        return False
    if (
        config.get("tcgen05_grouped_prefix_adaptive_n", False)
        or config.get("cute_materialized_operand_schedule", "off") != "off"
        or config.get("cute_materialized_schedule", "off") != "off"
        or config.get("cute_mma_f32_conversion") in {"warp_raw", "tma_rn"}
        or config.get("cute_grouped_rna_warps", 0)
        or config.get("cute_reduction_sequence") in {"bounded", "bounded_layout"}
        or config.get("cute_split_k_schedule") == "cluster8_k4"
        or config.get("cute_split_k_workspace", False)
        or config.get(COMPACT_ROWS_KEY, False)
    ):
        return True
    if assertion and (
        conflicting_fragment_schedule(spec, config) is not None
        or (
            spec.tcgen05.warp_mma_admitted
            and config.get("cute_matmul_family") == "warp_mma"
        )
        or (
            spec.cute_mma_dag_facts is not None
            and config.get("cute_mma_dag") in {"warp", "tcgen05"}
        )
    ):
        # These explicit modes require their native matmul root. Other
        # planned owners are validated only when their actual plan is known.
        return True
    # Only explicit PDL conflicts with the intrinsic clustered release protocol.
    # An inert cluster knob does not select a native owner for a fragment.
    return spec.has_explicit_pdl and (
        config.get("tcgen05_cluster_m", 1) == 2
        or config.get("tcgen05_cta_group", "auto") == "two"
        # Physical row-union profiles project a single-CTA search carrier
        # into clustered codegen geometry after normalization.
        or physical_schedule(config) is not None
    )


def _unsupported_generator(
    name: str,
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    transforms: KernelTransforms,
    *,
    scalar_assertion: bool,
) -> ast.Module:
    if scalar_assertion:
        raise exc.InvalidConfig(
            f"torch._assert_async cannot use whole-kernel generator {name}"
        )
    raise exc.InvalidConfig(
        f"explicit PDL markers are not supported by whole-kernel generator {name}; "
        "select an ordinary device-function schedule (for ordered phases, a "
        "supported persistent pid_type)"
    )


def guard_kernel_generators(
    generators: tuple[KernelGenerator, ...], *, scalar_assertion: bool = False
) -> tuple[KernelGenerator, ...]:
    if not scalar_assertion and not has_explicit_pdl():
        return generators
    # These generators replace or split the ordinary device function. Until
    # each carries the explicit protocol across its native launch boundary,
    # reject that configuration instead of dropping/reordering user markers.
    return tuple(
        replace(
            generator,
            generate=partial(
                _unsupported_generator,
                generator.name,
                scalar_assertion=scalar_assertion,
            ),
        )
        for generator in generators
    )


def validate_explicit_launch(
    fn: DeviceFunction, definition: ast.FunctionDef, *, scalar_assertion: bool = False
) -> None:
    explicit_pdl = has_explicit_pdl()
    if not explicit_pdl and not scalar_assertion:
        return
    from ...runtime.cute.host_wrappers import HOST_WRAPPER_EMITTERS

    for plan in cute_codegen_state(fn.codegen).wrapper_plans:
        kind = cast("str", plan["kind"])
        if HOST_WRAPPER_EMITTERS[kind].host_call is not None:
            if scalar_assertion:
                raise exc.InvalidConfig(
                    f"torch._assert_async cannot use native host wrapper {kind}"
                )
            raise exc.InvalidConfig(
                f"explicit PDL markers are not supported by native host wrapper {kind}"
            )
    if not explicit_pdl:
        return
    # Intrinsic dependency protocols (for example a materialized producer)
    # cannot be replaced with the ordinary explicit entry/exit sequence.
    if any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.griddepcontrol_launch_dependents"
        for node in ast.walk(definition)
    ):
        raise exc.InvalidConfig(
            "explicit PDL markers conflict with this schedule's native release protocol"
        )


def _host_noop(state: CodegenState) -> object:
    # DeviceFunction renders the markers through the backend's instruction hook.
    return expr_from_string("None")


_decorators.codegen(pdl_wait, "cute")(_host_noop)
_decorators.codegen(pdl_launch_dependents, "cute")(_host_noop)
