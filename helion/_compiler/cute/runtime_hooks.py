"""Host-side runtime hooks of the CuTe backend (host launcher / planner glue).

``CuteBackend`` forwards the runtime hooks of :class:`helion._compiler.backend.Backend`
here, so :mod:`helion.runtime.kernel` stays backend-neutral:

* :func:`plan_bound_kernel` (``Backend.on_bind``): bind-time planning of the
  structural passes that rewrite a kernel into several CuTe launches
  (materialized operands, region fission).
* :func:`record_host_structural_facts` (``Backend.host_structural_facts``): the
  tcgen05 auxiliary-tensor proofs the compiler seeds and the search surface
  read off the traced host function.
* :func:`begin_codegen` / :func:`finish_codegen` / :func:`register_compiled`
  (``Backend.before_codegen`` / ``after_codegen`` / ``after_compile``): the
  input-value specializations generated code depends on -- the M/N tails of a
  grouped static-persistent GEMM layout and the power-of-2 bucket of every
  hint-sized dynamic reduction extent.
* :func:`decorator_settings`, :func:`structural_policy_type`: the repro
  decorator parts and the structural policy class.

Tests patch ``host_function_matmul_has_non_tcgen05_operand`` on this module and the
grouped static-tail key helpers on :mod:`.grouped_static_tail`; every use below
resolves them through their module at call time.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Any
from typing import Hashable
from typing import NamedTuple

import sympy
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from ..._utils import next_power_of_2
from ...autotuner.benchmark_provider import _SEARCHING_CANDIDATES
from ...runtime.cute.structural_policy import CuteStructuralPolicy
from ...runtime.kernel import PreparedMetadataSpecializationExtractor
from ...runtime.kernel import evaluate_sympy_expr
from ...runtime.kernel import maybe_skip_dtype_check_in_meta_registrations
from ...runtime.kernel import tensor_property_value
from ..inductor_lowering_extra import patch_inductor_lowerings
from . import grouped_static_tail
from .aux_tensor import host_function_has_tcgen05_aux_kernel_pattern
from .aux_tensor import host_function_has_tcgen05_exact_shape_aux_kernel_pattern
from .aux_tensor import host_function_has_tcgen05_exact_shape_aux_operand
from .aux_tensor import host_function_matmul_has_non_tcgen05_operand
from .aux_tensor import host_function_tcgen05_aux_loads_on_every_store
from .aux_tensor import host_function_tcgen05_rowvec_aux_facts
from .split_k_exchange import host_function_has_plain_store_body
from .state import cute_env_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ...runtime.settings import Settings
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction


def plan_bound_kernel(
    bound: BoundKernel[Any], args: tuple[object, ...], *, rewrite_ast: bool
) -> None:
    """Plan the operand-materialization / region-fission rewrite of a bind."""
    settings = bound.settings
    if rewrite_ast and (
        settings.cute_region_fission or settings.cute_materialize_transformed_operands
    ):
        from .materialize_operand import plan_operand_materialization
        from .materialized_fission import plan_materialized_fission

        env = bound.env
        env_state = cute_env_state(env)
        with (
            maybe_skip_dtype_check_in_meta_registrations(),
            patch_inductor_lowerings(),
        ):
            if settings.cute_materialize_transformed_operands:
                env_state.fission_plan = plan_operand_materialization(
                    bound.kernel, args, env
                )
            if settings.cute_region_fission and env_state.fission_plan is None:
                env_state.fission_plan = plan_materialized_fission(
                    bound.kernel, args, env
                )


def record_host_structural_facts(
    host_function: HostFunction, config_spec: ConfigSpec
) -> None:
    """Record the tcgen05 auxiliary-tensor proofs of the traced host function.

    Detect kernels whose tcgen05 matmul is followed by an aux-fused store
    (``out[tile] = (acc + residual[tile]).to(...)`` and variants -- see
    ``host_function_has_tcgen05_aux_kernel_pattern`` for the accepted
    shapes). When detected, the autotune surface widens to admit
    ``tcgen05_strategy=ROLE_LOCAL_WITH_SCHEDULER`` +
    ``tcgen05_warp_spec_c_input_warps=1`` so the productive C-input warp lift
    is reachable from the normal autotune path. For pure-matmul kernels the
    detector returns False and the autotune surface keeps the narrow
    ``MONOLITHIC + c_input_warps=0`` shape so autotune cannot sample the
    strictly-worse inert C-input warp configuration.  The exact-shape detector
    is narrower: it gates the ``tcgen05_aux_load_mode=tma`` seed/search axis.
    The compiler seeds read these facts (the FFI direct-entry gate consults the
    non-tcgen05 operand flag, the cluster_m=2 seeds and the search projection
    the aux facts), so they are recorded before ``compiler_seed_configs``
    runs; they depend only on the traced host function, not on the runtime
    arguments.
    """
    config_spec.tcgen05.aux_kernel_detected = (
        host_function_has_tcgen05_aux_kernel_pattern(host_function)
    )
    # ``tcgen05_split_k`` guards only the tcgen05 tile store with its
    # last-arriver check; the knob needs a body whose only side effect is
    # that store.
    config_spec.tcgen05.split_k_plain_body = host_function_has_plain_store_body(
        host_function
    )
    config_spec.tcgen05.exact_shape_aux_kernel_detected = (
        host_function_has_tcgen05_exact_shape_aux_kernel_pattern(host_function)
    )
    config_spec.tcgen05.exact_shape_aux_operand_detected = (
        host_function_has_tcgen05_exact_shape_aux_operand(host_function)
    )
    # ``pre_acc_wait`` renders only where every store chain has per-subtile
    # auxiliary loads (or the shared fanout epilogue merges the chains); the
    # strict normalizer and the seeds read this proof so a two-output kernel
    # whose first output is aux-free never samples the placement into a
    # codegen refusal.
    config_spec.tcgen05.aux_loads_on_every_store = (
        host_function_tcgen05_aux_loads_on_every_store(host_function)
    )
    config_spec.tcgen05.matmul_has_non_tcgen05_operand = (
        host_function_matmul_has_non_tcgen05_operand(host_function)
    )
    config_spec.tcgen05.rowvec_aux_facts = host_function_tcgen05_rowvec_aux_facts(
        host_function
    )


def begin_codegen(bound: BoundKernel[Any]) -> None:
    cute_env_state(bound.env).hint_sized_reduction_extents.clear()


def finish_codegen(bound: BoundKernel[Any], config: Config) -> None:
    register_grouped_static_tail_specializations(bound)
    bound._cute_reduction_extents[config] = _cute_reduction_extent_descriptors(
        bound.env
    )


def register_compiled(bound: BoundKernel[Any], config: Config) -> None:
    register_reduction_extent_specializations(bound, config)


def register_grouped_static_tail_specializations(bound: BoundKernel[Any]) -> None:
    """Key the M/N tails of a grouped static-persistent GEMM's layout tensors."""
    if not bound._cache_managed:
        return
    bound._extend_descriptor_specializations(
        bound.kernel._cute_grouped_static_tail_extra_descriptors,
        grouped_static_tail.tail_descriptors(
            cute_env_state(bound.env).resolved_wrapper_plans
        ),
        grouped_static_tail.make_tail_extractor,
    )


def register_reduction_extent_specializations(
    bound: BoundKernel[Any], config: Config
) -> None:
    """Key the power-of-2 bucket of each hint-sized dynamic reduction extent.

    A CuTe persistent reduction sizes its thread layout (and the CTA) from
    the extent's size hint, while one dynamic-shape bound kernel serves
    every size: a later call with a larger extent would launch too few
    threads along that axis and skip the rest of each row.  Every config
    compiled to run is keyed, whether by set_config or directly through
    compile_config (whose callable a later bind() would otherwise hand
    back for any size), but not the configs the autotuner only measures
    (its reference baseline and candidates), so a looped winner keeps one
    bound kernel across sizes.
    """
    if not bound._cache_managed or _SEARCHING_CANDIDATES.get():
        return
    param_names = tuple(bound.kernel.signature.parameters)
    bound._extend_descriptor_specializations(
        bound.kernel._cute_reduction_extent_descriptors,
        bound._cute_reduction_extents.get(bound._normalized_config_copy(config), ()),
        lambda descriptor: PreparedMetadataSpecializationExtractor(
            _CuteReductionExtentExtractor(descriptor, param_names)
        ),
    )


def decorator_settings(settings: Settings) -> list[str]:
    """The CuTe parts of the ``@helion.kernel`` repro decorator."""
    # Structural settings determine the config's axis schema at binding.
    # Record opt-outs too, independent of the replay process's environment.
    # The graph CSE pass belongs here as well: the merged and the traced
    # device IR roll different reduction sweeps, so their configs carry a
    # different number of ``load_eviction_policies`` slots; a repro
    # printed under HELION_CUTE_GRAPH_CSE=0 must replay the traced graph.
    parts = [
        "backend='cute'",
        f"cute_region_fission={settings.cute_region_fission}",
        f"cute_full_slice_matmul_tiling={settings.cute_full_slice_matmul_tiling}",
        f"cute_segmented_matmul_tiling={settings.cute_segmented_matmul_tiling}",
        f"cute_flatten_nested_reductions={settings.cute_flatten_nested_reductions}",
        f"cute_materialize_transformed_operands={settings.cute_materialize_transformed_operands}",
        f"cute_graph_cse={settings.cute_graph_cse}",
        f"cute_rng_stream={settings.cute_rng_stream!r}",
    ]
    if not settings.cute_stage_unaligned_operands:
        # Non-default only: the pass changes the host code, not configs.
        parts.append("cute_stage_unaligned_operands=False")
    if not settings.cute_flat_edge_tma:
        parts.append("cute_flat_edge_tma=False")
    return parts


def structural_policy_type() -> type[CuteStructuralPolicy]:
    return CuteStructuralPolicy


class _CuteReductionExtent(NamedTuple):
    """A dynamic reduction extent and the input size behind each of its symbols."""

    extent: sympy.Expr
    symbols: tuple[sympy.Symbol, ...]
    sizes: tuple[TensorPropertySource, ...]


def _input_size_of(
    env: CompileEnvironment, symbol: sympy.Symbol
) -> TensorPropertySource | None:
    for source in env.shape_env.var_to_sources.get(symbol, ()):
        if (
            isinstance(source, TensorPropertySource)
            and source.prop == TensorProperty.SIZE
            and source.idx is not None
        ):
            return source
    return None


def _cute_reduction_extent_descriptors(
    env: CompileEnvironment,
) -> tuple[_CuteReductionExtent, ...]:
    """Describe each recorded reduction extent by the input sizes it reads."""
    descriptors: list[_CuteReductionExtent] = []
    for extent in cute_env_state(env).hint_sized_reduction_extents:
        symbols = tuple(
            sorted(
                (s for s in extent.free_symbols if isinstance(s, sympy.Symbol)),
                key=str,
            )
        )
        sizes = [_input_size_of(env, symbol) for symbol in symbols]
        # A block-size symbol leaves no input size: the config fixes it.
        if all(size is not None for size in sizes):
            descriptors.append(
                _CuteReductionExtent(
                    extent,
                    symbols,
                    tuple(size for size in sizes if size is not None),
                )
            )
    return tuple(sorted(descriptors, key=repr))


@dataclasses.dataclass(frozen=True)
class _CuteReductionExtentExtractor:
    """Power-of-2 bucket of a dynamic reduction extent read from input sizes."""

    descriptor: _CuteReductionExtent
    param_names: tuple[str, ...]

    def __call__(self, args: Sequence[object]) -> Hashable:
        root_values = dict(zip(self.param_names, args, strict=False))
        values: dict[sympy.Symbol, int] = {}
        for symbol, source in zip(
            self.descriptor.symbols, self.descriptor.sizes, strict=True
        ):
            if (value := tensor_property_value(source, root_values)) is None:
                return None
            values[symbol] = value
        return next_power_of_2(evaluate_sympy_expr(self.descriptor.extent, values))
