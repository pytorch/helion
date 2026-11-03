"""CuTe device-body pass pipeline: the ordered pass tables and their context.

Role: device-body pass (the pipeline tables).  This module owns the ORDER in which the CuTe AST passes run
over a rendered device body and the facts each one receives; the pass bodies
live in their own modules (``fuse_two_pass_loads``, ``hoist_warp_reduce``,
``scalar_epilogue``, ...) and are imported at call time, so a test patching a
pass through its module (``patch.object(scalar_epilogue,
"lower_scalar_epilogue", ...)``) still intercepts the render, and the
``device_function`` <-> cute import cycle stays where it is today.

``DeviceFunction.codegen_function_def`` reaches the two stages through
``Backend.finalize_device_body`` (pre-rename, over the kernel body) and
``Backend.finalize_kernel_def`` (post-rename, over the ``FunctionDef``);
``CuteBackend`` forwards them to ``run_device_body_passes`` and
``run_kernel_def_passes``.

Facts.  ``PassContext`` is the frozen record of everything the passes read
besides the body: the kernel's names (arguments, the kernel signature, the
tensor arguments), the static constexpr values, the tensor dtypes, the
proofs (disjoint tensor pairs, strides, sizes), the rename map, the
autotuner's reload modes, the settings and the launch shape.  It is built
ONCE per render (``build_pass_context``) from the exact expressions the
inline code once spelled at each call site; the divergent same-named facts
keep distinct names (``tensor_dtypes_str_skip`` is the dtype map that skips
tensors without a CuTe spelling; ``tensor_arg_names`` / ``argument_names`` /
``signature_names`` / ``boundary_names`` / ``param_uniform_names`` are the
five different name sets the inline code built from ``fn.arguments``, the
signature and the parameter lists).  ``exact_thread_block_dims`` is the proven
static launch shape or ``None`` (never a ``(1, 1, 1)`` placeholder): the
passes that need a shape are gated on it, ``lower_scalar_epilogue`` and
``simplify_proven_bounds`` return early on ``None``, and the register-row
tensor copy declines without a shape.

Stages.  ``DEVICE_BODY_PROLOGUE`` runs over the context built from the
rendered body (the launch shape still the strategies' raw inference); its
entries lower the collective matmuls and decide the launch shape (the
gathered-MMA extension of the signature, the owned block dims, the bounded
cache plan's block).  ``freeze_pass_context`` then seals the launch shape and
the final signature into the context every ``DEVICE_BODY_PASSES`` entry
reads; ``KERNEL_DEF_PASSES`` reads the same context after ``ast_rename``
(``DeviceBodyContext.pass_context`` carries it across).

Every entry is ``(name, gate, run, requires, provides)``: ``gate`` reads the
per-render state and decides whether ``run`` executes; ``run`` rewrites the
body (and the stage scratch: the lane-unroll plan, the private fragments).
``requires`` / ``provides`` are static ordering tags: an entry may only
require a tag an earlier entry provides, checked once at import
(``_check_pass_order``).  They record the data-carrying edges and the
orderings that used to live in comments (state-load pipelining before fp32
packing before BF16 packing, the FMA contraction before packet prefetching,
the register-row lowering before the rolled cluster pair, the sweep matchers
before the exact-division redo, the cluster pairs before the placement
validation, the late memory cuts before the scalar-policy loads).  The order
itself is a compatibility invariant: several passes recognize the shapes
earlier passes leave behind and the autotune cache key covers the rendered
source, so nothing here may be reordered without a render check.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple
from typing import cast

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.symbol import SymT
from torch.utils._sympy.symbol import symbol_is_type

from ...ast_extension import create_arg
from ...compile_environment import CompileEnvironment
from ...device_function import NumericArgument
from ...device_function import TensorArg
from ...device_function import TensorDescriptorArg
from ...device_function import TensorPropertyArg
from ...host_function import HostFunction
from ..autotune.search_surface import cute_fragment_mma_fact
from ..fragment_mma_dot import FRAGMENT_MMA_KEY
from ..ring_mma_dot import RING_MMA_KEY

if TYPE_CHECKING:
    from ....runtime.config import Config
    from ...device_function import DeviceBodyContext
    from ...device_function import DeviceFunction
    from ..bounded_loop_cache import OwnedFragment


@dataclasses.dataclass(frozen=True, eq=False)
class PassContext:
    """The frozen facts of one device-body render (see the module docstring).
    One context per render, compared by identity (``eq=False``).

    Built once by ``build_pass_context``; ``with_launch_shape`` seals the
    launch shape the prologue decided.  The containers are the objects the
    inline code built (``constexpr_values`` is ``DeviceBodyContext``'s dict,
    filled here and read by the post-rename stage); no pass writes to them.
    """

    fn: DeviceFunction
    env: CompileEnvironment
    config: Config
    fast_math: bool
    exact_division: bool
    target_device_capability: tuple[int, int] | None
    # ``env.config_spec.matmul_facts`` is set: the tcgen05 scheduling passes
    # pin exact statement sequences, so the scalar contractions stay off.
    matmul_kernel: bool
    # Float scalar kernel parameters (``SymFloat`` expression arguments and
    # host floats passed as numeric arguments).
    float_scalar_names: frozenset[str]
    # Static integer value of every constexpr name: literal inlined
    # constexpr args and host-side literal integer assignments.
    constexpr_values: dict[str, int]
    # Tensor argument -> CuTe dtype spelling, SKIPPING the tensors the backend
    # has no canonical spelling for (uint32 barrier flags): such tensors do
    # not take part in load fusion.
    tensor_dtypes_str_skip: dict[str, str]
    # The five name sets of the inline code: every tensor argument, every
    # argument, every parameter of the kernel signature (``ast.arg`` names,
    # after the gathered-MMA extension), the sorted arguments ahead of the
    # constexpr split (collective-matmul boundary), and the numeric /
    # tensor-property parameters (uniform across the launch).
    tensor_arg_names: frozenset[str]
    argument_names: frozenset[str]
    signature_names: frozenset[str]
    boundary_names: frozenset[str]
    param_uniform_names: frozenset[str]
    proven_disjoint_tensor_pairs: set[frozenset[str]]
    proven_tensor_stride_values: dict[tuple[str, int], int]
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]]
    # Pre-rename name -> the canonical name ``ast_rename`` collapses it to.
    rename_groups: dict[str, str]
    # Autotuner-selected reload mode per rolled or persistent reduction
    # block ("auto" / "register" / "gmem").
    reload_modes: dict[int, str]
    dynamic_trip_counts: dict[str, tuple[int, str]]
    new_var: Callable[[str], str]
    # Launch shape (final after ``freeze_pass_context``).
    exact_thread_block_dims: tuple[int, int, int] | None
    # ``xyz`` program ids without a cluster: the per-axis block-grid extents
    # proven from exact runtime sizes (``None`` where unproven).
    xyz_grid: bool
    block_grid_dims: tuple[int | None, int | None, int | None]

    def with_launch_shape(
        self,
        exact_thread_block_dims: tuple[int, int, int] | None,
        signature_names: frozenset[str],
    ) -> PassContext:
        return dataclasses.replace(
            self,
            exact_thread_block_dims=exact_thread_block_dims,
            signature_names=signature_names,
        )


def _float_scalar_names(fn: DeviceFunction) -> frozenset[str]:
    return frozenset(
        argument.name
        for expression, argument in fn._expr_args.items()
        if isinstance(expression, sympy.Symbol)
        and (
            symbol_is_type(expression, SymT.FLOAT)
            or symbol_is_type(expression, SymT.UNBACKED_FLOAT)
        )
    ) | frozenset(
        argument.name
        for argument in fn.arguments
        if isinstance(argument, NumericArgument)
        and isinstance(
            HostFunction.current().params.arguments.get(argument.host_str()),
            (float, torch.SymFloat),
        )
    )


def _collect_constexpr_values(ctx: DeviceBodyContext) -> dict[str, int]:
    # Static integer values for constexpr names so the fusion pass can
    # resolve range(..., step=cutlass.Int32(NAME)) trip counts.  Three
    # sources: literal constexpr inlined args, host-side literal
    # assignments to constexpr-named variables, and the inlined
    # module-level constexpr decls.
    constexpr_values = ctx.constexpr_values
    for arg in ctx.constexpr_to_inline:
        try:
            value = int(arg.host_str())
        except (TypeError, ValueError):
            continue
        constexpr_values[arg.name] = value
    for stmt in ctx.device_function.codegen.host_statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, int)
        ):
            constexpr_values[stmt.targets[0].id] = stmt.value.value
    return constexpr_values


def _tensor_dtypes_str_skip(fn: DeviceFunction, env: CompileEnvironment) -> dict:
    # ``dtype_str`` raises for dtypes the cute backend has no canonical
    # spelling for (e.g. uint32 barrier flags, which SIMT loads treat as raw
    # bytes); such tensors simply don't participate in load fusion.
    tensor_dtypes: dict[str, str] = {}
    for arg in fn.arguments:
        if isinstance(arg, TensorArg):
            try:
                tensor_dtypes[arg.name] = env.backend.dtype_str(arg.fake_value.dtype)
            except ValueError:
                continue
    return tensor_dtypes


def _reload_modes(fn: DeviceFunction, env: CompileEnvironment) -> dict[int, str]:
    reload_cfg = cast(
        "list[str]", fn.config.config.get("cute_reduction_reloads", []) or []
    )
    return {
        block_id: env.config_spec.cute_reduction_reloads.config_get(
            reload_cfg, block_id, "auto"
        )
        for block_id in env.config_spec.cute_reduction_reloads.valid_block_ids()
    }


def _block_grid_dims(
    fn: DeviceFunction, env: CompileEnvironment, constexpr_values: dict[str, int]
) -> tuple[bool, tuple[int | None, int | None, int | None]]:
    # For an exact in-place vector state-update pattern the state-load
    # pipeliner may stage future rows through a private-per-thread cp.async
    # ring; its matcher (and the proven-bounds simplifier) consume the exact
    # grid extents proved from cache-keyed runtime size facts.
    from ...program_id import XYZProgramIDs

    xyz_pid = (
        fn.pid
        if isinstance(fn.pid, XYZProgramIDs)
        and fn.cute_state.lanes.simt_cluster_n == 1
        and len(fn.pid.pid_info) <= 3
        else None
    )
    xyz_grid = xyz_pid is not None
    exact_grid_extent_values: dict[sympy.Expr, int] = {}
    ambiguous_grid_extent_expressions: set[sympy.Expr] = set()
    if xyz_grid and "input_tensor_metadata" in env.compiler_fact_specialization_facts:
        for argument in fn.arguments:
            if not isinstance(argument, TensorArg) or isinstance(
                argument, TensorDescriptorArg
            ):
                continue
            fake_tensor = argument.fake_value
            if env.tensor_input_source(fake_tensor) is None:
                continue
            runtime_tensor = env.runtime_value_for_tensor(fake_tensor)
            if not isinstance(runtime_tensor, torch.Tensor) or isinstance(
                runtime_tensor, FakeTensor
            ):
                continue
            for dimension, size in enumerate(fake_tensor.shape):
                if isinstance(size, int):
                    continue
                try:
                    expression = env.shape_env.replace(size._sympy_())
                    value = int(runtime_tensor.size(dimension))
                    if value != int(env.size_hint(size)):
                        continue
                except (RuntimeError, TypeError, ValueError):
                    continue
                previous = exact_grid_extent_values.get(expression)
                if previous is not None and previous != value:
                    ambiguous_grid_extent_expressions.add(expression)
                    exact_grid_extent_values.pop(expression, None)
                elif expression not in ambiguous_grid_extent_expressions:
                    exact_grid_extent_values[expression] = value
    block_grid_dims: list[int | None] = [None, None, None]
    if xyz_pid is not None:
        for axis, pid_info in enumerate(xyz_pid.pid_info):
            numel = pid_info.numel
            if isinstance(numel, (int, sympy.Integer)):
                extent = int(numel)
            elif isinstance(numel, sympy.Expr):
                try:
                    normalized_numel = env.shape_env.replace(numel)
                except (RuntimeError, TypeError, ValueError):
                    continue
                if normalized_numel not in exact_grid_extent_values:
                    continue
                extent = exact_grid_extent_values[normalized_numel]
            else:
                continue
            try:
                block_size = int(pid_info.block_size_var)
            except (TypeError, ValueError):
                block_size = constexpr_values.get(pid_info.block_size_var, 0)
            if block_size > 0 and extent >= 0:
                block_grid_dims[axis] = (extent + block_size - 1) // block_size
        for axis in range(len(xyz_pid.pid_info), 3):
            block_grid_dims[axis] = 1
    return xyz_grid, (block_grid_dims[0], block_grid_dims[1], block_grid_dims[2])


def build_pass_context(ctx: DeviceBodyContext) -> PassContext:
    """The facts of the render handed to ``Backend.finalize_device_body``.

    The launch shape is the strategies' raw inference here; the prologue
    passes decide the final one (``freeze_pass_context``).
    """
    # Looked up through its module at call time: a test patches it there.
    from ...device_function import _exact_thread_block_dims

    fn = ctx.device_function
    env = CompileEnvironment.current()
    constexpr_values = _collect_constexpr_values(ctx)
    xyz_grid, block_grid_dims = _block_grid_dims(fn, env, constexpr_values)
    return PassContext(
        fn=fn,
        env=env,
        config=fn.config,
        fast_math=env.settings.fast_math,
        exact_division=env.settings.cute_exact_division,
        target_device_capability=env.config_spec.target_device_capability,
        matmul_kernel=bool(env.config_spec.matmul_facts),
        float_scalar_names=_float_scalar_names(fn),
        constexpr_values=constexpr_values,
        tensor_dtypes_str_skip=_tensor_dtypes_str_skip(fn, env),
        tensor_arg_names=frozenset(
            arg.name for arg in fn.arguments if isinstance(arg, TensorArg)
        ),
        argument_names=frozenset(arg.name for arg in fn.arguments),
        signature_names=frozenset(arg.arg for arg in ctx.args),
        boundary_names=frozenset(arg.name for arg in ctx.sorted_arguments),
        param_uniform_names=frozenset(
            arg.name
            for arg in ctx.param_args
            if isinstance(arg, (NumericArgument, TensorPropertyArg))
        ),
        proven_disjoint_tensor_pairs=fn.proven_disjoint_tensor_pairs(),
        proven_tensor_stride_values=fn.proven_tensor_stride_values(),
        proven_tensor_size_values=fn.proven_tensor_size_values(),
        rename_groups={k: v[0] for k, v in fn._variable_renames.items()},
        reload_modes=_reload_modes(fn, env),
        dynamic_trip_counts=fn.cute_state.lanes.dynamic_reduction_trips,
        new_var=lambda name: fn.new_var(name, dce=False),
        exact_thread_block_dims=_exact_thread_block_dims(fn.tile_strategy),
        xyz_grid=xyz_grid,
        block_grid_dims=block_grid_dims,
    )


class _DeviceBodyState:
    """Mutable per-render state of the pre-rename stage.

    ``ctx`` is the frozen facts (the prologue's context until
    ``freeze_pass_context`` seals the launch shape); the rest is the body,
    the signature and the scratch the stage steps hand to one another.
    """

    def __init__(self, body: list[ast.stmt], ctx: DeviceBodyContext) -> None:
        self.ctx = build_pass_context(ctx)
        self.fn = self.ctx.fn
        self.kernel_body = body
        # The kernel signature so far; a prologue step may append parameters.
        self.args: list[ast.arg] = list(ctx.args)
        self.param_args = ctx.param_args
        self.wrapper_only_params = ctx.wrapper_only_params
        self.bounded_cache_request = ctx.bounded_cache_request
        # The launch shape the prologue settles on.
        self.launch_shape = self.ctx.exact_thread_block_dims
        # Scratch the stage steps share.
        self.private_fragments: tuple[OwnedFragment, ...] = ()
        self.knob_unroll = 1
        self.sinking = False
        self.fold_sites: dict[str, set[str]] = {}
        self.folded_lanes: frozenset[str] = frozenset()


class _KernelDefState:
    """Mutable per-render state threaded through ``KERNEL_DEF_PASSES``."""

    def __init__(
        self,
        definition: ast.FunctionDef,
        prefix: list[ast.stmt],
        ctx: DeviceBodyContext,
    ) -> None:
        self.ctx = cast("PassContext", ctx.pass_context)
        self.fn = self.ctx.fn
        self.definition = definition
        self.prefix = prefix
        self.param_args = ctx.param_args


class DeviceBodyPass(NamedTuple):
    name: str
    gate: Callable[[_DeviceBodyState], bool]
    run: Callable[[_DeviceBodyState], None]
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()


class KernelDefPass(NamedTuple):
    name: str
    gate: Callable[[_KernelDefState], bool]
    run: Callable[[_KernelDefState], None]
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()


def _always(state: object) -> bool:
    return True


def _config_flag(key: str) -> Callable[[_DeviceBodyState | _KernelDefState], bool]:
    """Gate on a boolean config knob, as the inline ``if self.config.get(key, False):``."""

    def gate(s: _DeviceBodyState | _KernelDefState) -> bool:
        return bool(s.ctx.config.get(key, False))

    return gate


def _fast_math(s: _DeviceBodyState) -> bool:
    return s.ctx.fast_math


def _has_bounded_cache_request(s: _DeviceBodyState) -> bool:
    return s.bounded_cache_request is not None


def _has_exact_thread_block_dims(s: _DeviceBodyState) -> bool:
    return s.ctx.exact_thread_block_dims is not None


# ---------------------------------------------------------------------------
# Prologue: the collective lowerings and the launch shape
# ---------------------------------------------------------------------------


def _lower_uniform_comparisons(s: _DeviceBodyState) -> None:
    from ..uniform_comparison import lower_uniform_comparisons

    s.kernel_body = lower_uniform_comparisons(
        s.kernel_body, s.fn, float_scalar_names=set(s.ctx.float_scalar_names)
    )


def _lower_collective_matmul(s: _DeviceBodyState) -> None:
    from ..collective_matmul import lower_collective_matmul

    s.kernel_body = lower_collective_matmul(
        s.kernel_body,
        s.fn,
        boundary_names=set(s.ctx.boundary_names),
        disjoint_pairs=s.ctx.proven_disjoint_tensor_pairs,
        rename_groups=s.ctx.rename_groups,
    )


def _lower_simt_gemm(s: _DeviceBodyState) -> None:
    from ..simt_gemm import lower_simt_gemm

    s.kernel_body = lower_simt_gemm(
        s.kernel_body, s.fn, rename_groups=s.ctx.rename_groups
    )


def _has_gathered_mma_tma_plan(s: _DeviceBodyState) -> bool:
    return any(
        plan.get("kind") == "gathered_mma_tma"
        for plan in s.fn.codegen.cute_wrapper_plans
    )


def _extend_gathered_mma_tma_launch(s: _DeviceBodyState) -> None:
    # A proved late region may introduce TMA objects supplied by
    # the wrapper and an explicit producer/consumer launch shape.
    s.args.extend(
        create_arg(name)
        for name in s.fn.wrapper_only_params
        if name not in s.wrapper_only_params
    )
    s.launch_shape = (288, 1, 1)


def _owned_block_dims(s: _DeviceBodyState) -> tuple[int, int, int] | None:
    return (
        s.fn.cute_state.lanes.collective_register_chain_block_dims
        or s.fn.cute_state.lanes.owned_root_block_dims
    )


def _has_owned_block_dims(s: _DeviceBodyState) -> bool:
    return _owned_block_dims(s) is not None


def _adopt_owned_block_dims(s: _DeviceBodyState) -> None:
    s.launch_shape = _owned_block_dims(s)


def _bounded_cache_prepare(s: _DeviceBodyState) -> None:
    request = s.bounded_cache_request
    assert request is not None
    ctx = s.ctx
    s.kernel_body = request.prepare(
        s.kernel_body,
        s.fn,
        s.param_args,
        ctx.constexpr_values,
        ctx.tensor_dtypes_str_skip,
        ctx.proven_disjoint_tensor_pairs,
    )
    if request.plan is not None:
        s.launch_shape = request.plan.launch_block


def freeze_pass_context(s: _DeviceBodyState) -> None:
    """Seal the launch shape and the final signature into the context."""
    s.ctx = s.ctx.with_launch_shape(
        s.launch_shape, frozenset(arg.arg for arg in s.args)
    )


# ---------------------------------------------------------------------------
# Pre-rename stage: ``finalize_device_body``
# ---------------------------------------------------------------------------


def _finalize_shared_reduce_groups(s: _DeviceBodyState) -> None:
    # The strategies' launch shape is final here: size the
    # cross-warp shared reductions for the threads that launch
    # instead of the 1024-thread budget they were emitted against.
    # Declined when the body claimed a wider axis (a free
    # ``hl.arange`` thread axis the launcher adds to ``block=``);
    # the launcher checks the recorded shape against its launch.
    from ..finalize_reduce_groups import finalize_shared_reduce_groups_for_launch

    exact_thread_block_dims = s.ctx.exact_thread_block_dims
    assert exact_thread_block_dims is not None
    s.kernel_body, sized_for = finalize_shared_reduce_groups_for_launch(
        s.kernel_body,
        thread_block_dims=exact_thread_block_dims,
        claimed_axis_sizes=s.fn.codegen.launch_thread_axis_sizes(),
    )
    if sized_for is not None:
        s.fn.cute_state.lanes.shared_reduce_launch_block = sized_for


def _fuse_two_pass_loads(s: _DeviceBodyState) -> None:
    from ..fuse_two_pass_loads import fuse_two_pass_loads

    ctx = s.ctx
    exact_thread_block_dims = ctx.exact_thread_block_dims
    assert exact_thread_block_dims is not None
    s.kernel_body = fuse_two_pass_loads(
        s.kernel_body,
        ctx.constexpr_values,
        thread_block_dims=exact_thread_block_dims,
        tensor_dtypes=ctx.tensor_dtypes_str_skip,
        reload_modes=ctx.reload_modes,
        proven_disjoint_tensor_pairs=ctx.proven_disjoint_tensor_pairs,
        proven_tensor_stride_values=ctx.proven_tensor_stride_values,
        dynamic_trip_counts=ctx.dynamic_trip_counts,
    )


def _vectorize_branch_local_persistent_fragments(s: _DeviceBodyState) -> None:
    # Some exact persistent fragments have addresses defined only
    # inside a runtime branch (for example a loaded state-slot index),
    # so the normal root-wrapper hoist cannot legally reference them.
    # Their lowering markers survive the reduction split and load
    # fuser; place one vector transaction beside the earliest surviving
    # constexpr sweep and erase every unhandled marker here.
    from ..cache_policy_loads import hint_wide_packets
    from ..persistent_branch_vec import vectorize_branch_local_persistent_fragments

    s.kernel_body = vectorize_branch_local_persistent_fragments(
        s.kernel_body,
        proven_disjoint_tensor_pairs=s.ctx.proven_disjoint_tensor_pairs,
        proven_tensor_stride_values=s.ctx.proven_tensor_stride_values,
        wide_packet_hints=hint_wide_packets(s.ctx.config.cute_wide_packet_hints),
    )


def _hoist_warp_reduce_from_vloop(s: _DeviceBodyState) -> None:
    # Hoist warp reductions out of constexpr V-loops to collapse
    # 4 per-V-lane warp reductions into 1 V-fold + 1 warp reduce.
    # For online softmax style kernels this drops per-row reductions
    # from ~396 to ~99 (4x fewer SHFL trees).  The collapse stage needs
    # the post-rename canonical map to see through loop-carried alias
    # vars (``v_1`` renaming to ``mi``).
    from ..hoist_warp_reduce import hoist_warp_reduce_from_vloop

    s.kernel_body = hoist_warp_reduce_from_vloop(
        s.kernel_body,
        running_sum_accumulators=s.fn.cute_matmul_running_sums,
        rename_groups=s.ctx.rename_groups,
        unfoldable_vec_lanes=s.fn.cute_state.lanes.unfoldable_vec_lanes,
    )


def _has_simt_cluster(s: _DeviceBodyState) -> bool:
    return s.fn.cute_state.lanes.simt_cluster_n > 1


def _eliminate_duplicate_cluster_maxima(s: _DeviceBodyState) -> None:
    from ..duplicate_reduction_carries import eliminate_duplicate_cluster_maxima

    s.kernel_body = eliminate_duplicate_cluster_maxima(
        s.kernel_body, s.ctx.constexpr_values, s.ctx.rename_groups
    )


def _bounded_cache_cache(s: _DeviceBodyState) -> None:
    request = s.bounded_cache_request
    assert request is not None
    s.kernel_body, s.private_fragments = request.cache(
        s.kernel_body, s.ctx.constexpr_values, s.ctx.rename_groups
    )


def _vectorize_affine_tile_lanes(s: _DeviceBodyState) -> None:
    from ..affine_vector_io import vectorize_affine_tile_lanes

    if s.bounded_cache_request is not None:
        s.kernel_body = vectorize_affine_tile_lanes(
            s.kernel_body,
            s.fn,
            s.ctx.constexpr_values,
            private_fragments=s.private_fragments,
        )
    else:
        s.kernel_body = vectorize_affine_tile_lanes(
            s.kernel_body, s.fn, s.ctx.constexpr_values
        )


def _merge_literal_offset_loads(s: _DeviceBodyState) -> None:
    from ..literal_offset_loads import merge_literal_offset_loads

    s.kernel_body = merge_literal_offset_loads(s.kernel_body, s.fn)


def _fuse_fp4_pair_dot(s: _DeviceBodyState) -> None:
    # Replace a byte-by-byte fp4 x fp4 decode-multiply-add chain over
    # two merged 8-byte vectors with the exact f16x2 group dot
    # (``quantized_helpers.float4_e2m1fn_x16_dot_float32``).
    from ..fuse_fp4_pair_dot import fuse_fp4_pair_dot

    s.kernel_body = fuse_fp4_pair_dot(s.kernel_body)


def _merge_sibling_v_loops(s: _DeviceBodyState) -> None:
    # Merge adjacent constexpr V-loops that share an identical
    # statement prefix.  Caches the last common per-V-lane value
    # into a register fragment so V-loop 2's bitcast/cast chain
    # disappears and the SASS scheduler can issue V-loop 2's
    # arithmetic without waiting for V-loop 1's results.
    from ..merge_sibling_v_loops import merge_sibling_v_loops

    s.kernel_body = merge_sibling_v_loops(s.kernel_body)


def _optimize_vector_reductions(s: _DeviceBodyState) -> None:
    from ..vector_reduction_packets import optimize_vector_reductions

    config = s.ctx.config
    s.kernel_body = optimize_vector_reductions(
        s.kernel_body,
        s.ctx.constexpr_values,
        thread_block_dims=s.ctx.exact_thread_block_dims,
        independent_accumulators=config.get("cute_independent_reduction", False)
        is True,
        replicated_single_use=config.get("cute_replicated_reduction", False) is True,
        unroll_packets=config.get("cute_vector_packet_unroll", False) is True,
    )


def _hoist_lane_invariant_reductions(s: _DeviceBodyState) -> None:
    # A persistent row tile may repeat a row-invariant Q/K norm (and
    # its warp reduction) once for every compile-time row lane.  Wide
    # row tiles are useful only if that setup is shared.  Unswitch a
    # proven-uniform terminal guard and move complete, pure invariant
    # reduction slices before the row loop.  Runtime storage facts are
    # required before a load may cross any loop write.
    from ..hoist_lane_invariant_reductions import hoist_lane_invariant_reductions

    ctx = s.ctx
    s.kernel_body = hoist_lane_invariant_reductions(
        s.kernel_body,
        tensor_names=set(ctx.tensor_dtypes_str_skip),
        tensor_dtypes=ctx.tensor_dtypes_str_skip,
        proven_disjoint_tensor_pairs=ctx.proven_disjoint_tensor_pairs,
        rename_groups=ctx.rename_groups,
        uniform_names=set(ctx.signature_names),
        float_scalar_names=set(ctx.float_scalar_names),
        thread_block_dims=ctx.exact_thread_block_dims,
    )


def _hoist_factored_reductions(s: _DeviceBodyState) -> None:
    from ..factor_affine_reductions import hoist_factored_reductions

    s.kernel_body = hoist_factored_reductions(
        s.kernel_body,
        fast_math=True,
    )


def _fuse_fp8_pair_decode(s: _DeviceBodyState) -> None:
    # Fuse adjacent per-lane fp8 decodes in the SIMT matmul V-loop
    # into one ``cvt.rn.f16x2.e4m3x2`` (decode 2 e4m3 bytes per
    # instruction) — halves the decode instruction count on the
    # skinny-M fp8 GEMV path.
    from ..fuse_fp8_pair_decode import fuse_fp8_pair_decode

    s.kernel_body = fuse_fp8_pair_decode(s.kernel_body)


def _hoist_loop_invariant_recips(s: _DeviceBodyState) -> None:
    # Under fast_math, hoist loop-invariant floating-point
    # divisions out of inner tile loops, replacing each
    # ``x / scalar`` with a hoisted ``inv = 1.0 / scalar`` +
    # ``x * inv`` in the loop body.  B200 div is ~22 cycles vs ~2
    # for multiply, so the softmax consume sweep (~12672 divides
    # per row) sees a measured +20% bench gain on (4096, 12672)
    # fp16.  The rewrite rounds differently and fires only in the
    # loops a config creates, so it needs the setting.  The rename map
    # lets the invariance analysis treat ``v_1_0`` (renamed to ``mi``
    # by ast_rename below) as an assignment to ``mi``; without it the
    # FMA hoist would classify ``mi`` as loop-invariant in the reduce
    # loop and capture its stale initial value.
    from ..hoist_loop_invariant_recip import hoist_loop_invariant_recips

    s.kernel_body = hoist_loop_invariant_recips(
        s.kernel_body,
        rename_groups=s.ctx.rename_groups,
        fast_math=s.ctx.fast_math,
    )


def _not_matmul_kernel(s: _DeviceBodyState) -> bool:
    return not s.ctx.matmul_kernel


def _fuse_fma(s: _DeviceBodyState) -> None:
    # Contract single-use fp32 ``t = a*b; w = t + c`` chains into
    # ``cute.math.fma`` (the DSL's arith ops carry no contract
    # flag, so NVVM won't fuse them itself).  Triton contracts by
    # default, so this brings cute numerics closer to the triton
    # backend.
    from ..fuse_fma import fuse_fma

    s.kernel_body = fuse_fma(s.kernel_body, rename_groups=s.ctx.rename_groups)


def _ring_mma_dot(s: _DeviceBodyState) -> None:
    # ``cute_dot_ring_mma``: the GEMV dot on a ``(T, 1, 1)`` block streams the
    # row tile through a cp.async ring in shared memory and multiplies on
    # ``mma.sync`` fragments (``cute/ring_mma_dot.py``).  The knob-off code is
    # regenerated when no dot takes the rewrite.
    from ..memory_ops import cute_tensor_base_is_aligned
    from ..ring_mma_dot import RING_DEFAULT_SPLIT
    from ..ring_mma_dot import RING_DEFAULT_STAGES
    from ..ring_mma_dot import RING_SPLIT_KEY
    from ..ring_mma_dot import RING_STAGES_KEY
    from ..ring_mma_dot import RingMmaNotApplied
    from ..ring_mma_dot import ring_mma_dot
    from ..ring_mma_dot import ring_mma_smem_budget

    fn = s.fn
    env = s.ctx.env
    config = s.ctx.config.config
    ring_fact = cute_fragment_mma_fact(env.config_spec)
    ring_rows: object = None
    ring_chunk: object = None
    ring_k_var: str | None = None
    if (
        ring_fact is not None
        and ring_fact.m_block_id is not None
        and ring_fact.k_block_id is not None
    ):
        ring_rows = env.block_sizes[ring_fact.m_block_id].from_config(fn.config)
        ring_chunk = env.block_sizes[ring_fact.k_block_id].from_config(fn.config)
        # A cache read only: ``block_size_var`` would lift a new constexpr
        # argument after the host call was emitted.
        ring_k_var = fn.block_size_var_cache.get((ring_fact.k_block_id,))
    s.kernel_body, ring_fired = ring_mma_dot(
        s.kernel_body,
        thread_block_dims=s.ctx.exact_thread_block_dims,
        rows=ring_rows if isinstance(ring_rows, int) else None,
        chunk=ring_chunk if isinstance(ring_chunk, int) else None,
        stages=cast("int", config.get(RING_STAGES_KEY, RING_DEFAULT_STAGES)),
        split=cast("str", config.get(RING_SPLIT_KEY, RING_DEFAULT_SPLIT)),
        k_block_var=ring_k_var,
        tensors={
            arg.name: arg.fake_value
            for arg in fn.arguments
            if isinstance(arg, TensorArg)
        },
        aligned={
            arg.name
            for arg in fn.arguments
            if isinstance(arg, TensorArg)
            and cute_tensor_base_is_aligned(env, arg.fake_value, 16)
        },
        new_var=s.ctx.new_var,
        rename_groups=s.ctx.rename_groups,
        smem_budget=ring_mma_smem_budget(env.device),
    )
    if not ring_fired:
        raise RingMmaNotApplied


def _fragment_mma_dot(s: _DeviceBodyState) -> None:
    # ``cute_dot_fragment_mma``: a GEMV's scalar dot fallback (16-bit
    # packets, fp32 per-element FMAs, a grouped shuffle sum over the K
    # threads) on an (8, 4, 1) thread block runs on ``mma.sync`` fragments
    # instead (``cute/fragment_mma_dot.py``); the lane unroll then hoists the
    # fragment packets like any other packet loads.  The knob-off code is
    # regenerated when no dot takes the rewrite.
    from ..fragment_mma_dot import FragmentMmaNotApplied
    from ..fragment_mma_dot import fragment_mma_dot

    s.kernel_body, fragment_fired = fragment_mma_dot(
        s.kernel_body,
        thread_block_dims=s.ctx.exact_thread_block_dims,
        new_var=s.ctx.new_var,
    )
    if not fragment_fired:
        raise FragmentMmaNotApplied


def _plan_lane_unroll(s: _DeviceBodyState) -> None:
    # ``cute_lane_unroll`` without vector-loop sinking, or with a sunk
    # nest whose lane loops the sinking pass left rolled (their loads
    # were persistent markers, now packets): unroll the lane loops (of
    # the grid or of a device loop) at trace time with every unrolled
    # lane's loads (packets and scalars alike) issued before the first
    # lane's compute.  The knob-off code is regenerated when no loop
    # takes the unroll.
    from ..fold_lane_reductions import lanes_with_sites

    fn = s.fn
    lane_unroll = cast("int", s.ctx.config.config.get("cute_lane_unroll", 1))
    s.sinking = bool(s.ctx.config.config.get("cute_vloop_sink"))
    s.knob_unroll = (
        lane_unroll
        if lane_unroll > 1
        and not (s.sinking and fn.cute_state.lanes.vloop_sink_unrolled)
        else 1
    )
    # The vectorized lane loops of a reduced block whose per-lane
    # cross-thread reductions are still inside them unroll completely
    # whatever the knob says, and those reductions are folded across
    # the trace-time lanes below.  A lane loop an earlier pass already
    # reduces once after (the combine moved out of it) keeps its code.
    s.fold_sites = fn.cute_state.lanes.lane_fold_reduce_sites
    s.folded_lanes = lanes_with_sites(s.kernel_body, s.fold_sites)


def _wants_lane_unroll(s: _DeviceBodyState) -> bool:
    return s.knob_unroll > 1 or bool(s.folded_lanes)


def _unroll_lane_loads(s: _DeviceBodyState) -> None:
    # Issue all of a grid lane loop's vec loads before its first
    # store (ptxas cannot prove the store doesn't alias the next
    # iteration's load, so unsplit loops keep only ONE load in
    # flight per thread; the triton backend and handwritten CuTe
    # kernels both load the whole tile fragment first).
    from ..unroll_lane_loads import LaneUnrollNotApplied
    from ..unroll_lane_loads import unroll_lane_loads

    s.kernel_body, unrolled = unroll_lane_loads(
        s.kernel_body,
        s.knob_unroll,
        s.ctx.new_var,
        forced_full=s.folded_lanes,
    )
    # With sinking the knob belongs to the sunk nest's lane loops
    # (``_normalize_cute_vloop_sink``); a nest that offers none
    # keeps its code rather than regenerating knob-off.
    if s.knob_unroll > 1 and not unrolled and not s.sinking:
        raise LaneUnrollNotApplied


def _has_folded_lanes(s: _DeviceBodyState) -> bool:
    return bool(s.folded_lanes)


def _fold_lane_reductions(s: _DeviceBodyState) -> None:
    from ..fold_lane_reductions import fold_lane_reductions

    s.kernel_body = fold_lane_reductions(
        s.kernel_body,
        {lane: keys for lane, keys in s.fold_sites.items() if lane in s.folded_lanes},
        s.ctx.new_var,
    )


def _split_lane_loads(s: _DeviceBodyState) -> None:
    from ..split_lane_loads import split_lane_loads

    s.kernel_body = split_lane_loads(s.kernel_body)


def _prefetch_reduction_packets(s: _DeviceBodyState) -> None:
    # Preserve the global FMA decisions above. This optional pass
    # moves only raw packet loads/cache writes across a proven
    # single-trip reduction; arithmetic and PDL waits stay in place.
    from ..prefetch_reduction_packets import prefetch_reduction_packets

    ctx = s.ctx
    exact_thread_block_dims = ctx.exact_thread_block_dims
    assert exact_thread_block_dims is not None
    s.kernel_body = prefetch_reduction_packets(
        s.kernel_body,
        constexpr_values=ctx.constexpr_values,
        thread_block_dims=exact_thread_block_dims,
        tensor_dtypes=ctx.tensor_dtypes_str_skip,
        reload_modes=ctx.reload_modes,
        argument_names=set(ctx.argument_names),
        proven_disjoint_tensor_pairs=ctx.proven_disjoint_tensor_pairs,
    )


def _pipeline_inner_loads(s: _DeviceBodyState) -> None:
    # P18: software-pipeline the per-iteration vec load by one
    # stage.  Pre-issue iter 0's load above the loop and, inside
    # the body, issue iter N+1's load BEFORE iter N's compute
    # runs.  The B200 SASS scheduler can then keep multiple
    # ld.global instructions in flight, hiding HBM round-trip
    # latency on softmax/online-reduction inner loops where the
    # ``load -> compute(mi, di) -> next iter`` sequential
    # dependency chain dominates the per-iter stall budget.  The
    # rename map lets the loop-carried-write gate identify writes
    # whose pre-rename target is an alias (``v_1_0 = v_1`` renamed to
    # ``mi = v_1``); without it the softmax reduce sweep would look
    # free of loop-carried writes and skip pipelining.
    from ..pipeline_inner_loads import pipeline_inner_loads

    s.kernel_body = pipeline_inner_loads(
        s.kernel_body,
        s.ctx.constexpr_values,
        rename_groups=s.ctx.rename_groups,
        dynamic_trip_counts=s.ctx.dynamic_trip_counts,
    )


def _async_load_stages(s: _DeviceBodyState) -> int:
    return cast("int", s.ctx.config.config.get("cute_async_load_stages", 0))


def _wants_pipelined_state_loads(s: _DeviceBodyState) -> bool:
    return bool(_async_load_stages(s)) and s.ctx.exact_thread_block_dims is not None


def _pipeline_state_loads(s: _DeviceBodyState) -> None:
    from ..memory_ops import runtime_tensor_has_specialized_alignment
    from ..pipeline_state_loads import TensorMetadata
    from ..pipeline_state_loads import pipeline_state_loads

    ctx = s.ctx
    fn = ctx.fn
    env = ctx.env
    exact_thread_block_dims = ctx.exact_thread_block_dims
    assert exact_thread_block_dims is not None
    tensor_metadata: dict[str, TensorMetadata] = {}
    for arg in fn.arguments:
        if not isinstance(arg, TensorArg):
            continue
        try:
            shape = tuple(env.size_hint(dim) for dim in arg.fake_value.shape)
        except (RuntimeError, TypeError, ValueError):
            continue
        tensor_metadata[arg.name] = TensorMetadata(
            ctx.tensor_dtypes_str_skip.get(arg.name, ""), shape
        )
    config = ctx.config.config
    s.kernel_body = pipeline_state_loads(
        s.kernel_body,
        stages=_async_load_stages(s),
        lookahead=cast("int", config.get("cute_async_load_lookahead", 4)),
        group_rows=cast("int", config.get("cute_async_load_group_rows", 2)),
        cache_policy=cast("str", config.get("cute_async_load_cache", "cg")),
        store_policy=cast("str", config.get("cute_async_store_policy", "default")),
        thread_block_dims=exact_thread_block_dims,
        tensor_metadata=tensor_metadata,
        tensor_names=ctx.tensor_arg_names,
        proven_tensor_base_alignments=frozenset(
            arg.name
            for arg in fn.arguments
            if isinstance(arg, TensorArg)
            and runtime_tensor_has_specialized_alignment(env, arg.fake_value, 4)
        ),
        proven_tensor_size_values=ctx.proven_tensor_size_values,
        proven_disjoint_tensor_pairs=ctx.proven_disjoint_tensor_pairs,
        proven_tensor_stride_values=ctx.proven_tensor_stride_values,
        block_grid_dims=ctx.block_grid_dims,
        constexpr_values=ctx.constexpr_values,
        uniform_names=ctx.param_uniform_names | frozenset(ctx.constexpr_values),
        target_device_capability=ctx.target_device_capability,
    )


def _pack_fp32_constexpr_loops(s: _DeviceBodyState) -> None:
    # Pair eligible fp32 constexpr-loop lanes only after state-load
    # pipelining has recognized and staged the original scalar update
    # loop.  Packing first obscures that matcher and forfeits cp.async.
    from ..factor_affine_reductions import pack_fp32_constexpr_loops

    s.kernel_body = pack_fp32_constexpr_loops(
        s.kernel_body,
        fast_math=s.ctx.fast_math,
        target_device_capability=s.ctx.target_device_capability,
        float_scalar_names=set(s.ctx.float_scalar_names),
    )


def _hoist_packed_factored_reductions(s: _DeviceBodyState) -> None:
    from ..factor_affine_reductions import hoist_packed_factored_reductions

    s.kernel_body = hoist_packed_factored_reductions(
        s.kernel_body,
        fast_math=s.ctx.fast_math,
        rename_groups=s.ctx.rename_groups,
    )


def _pack_bf16_recurrences(s: _DeviceBodyState) -> None:
    # Once the async state pass has proved a private aligned BF16
    # recurrence, an opt-in late transform can retain the state and
    # both reductions as native BF16x2 words.  Running after the
    # generic fp32 pair/LICM passes gives the matcher the complete
    # recurrence graph while keeping all earlier passes reusable.
    from ..pack_bf16_recurrence import pack_bf16_recurrences

    ctx = s.ctx
    s.kernel_body = pack_bf16_recurrences(
        s.kernel_body,
        enabled=cast("bool", ctx.config.config.get("cute_bf16x2_recurrence", False)),
        fast_math=ctx.fast_math,
        target_device_capability=ctx.target_device_capability,
        uniform_names=ctx.signature_names | frozenset(ctx.constexpr_values),
        thread_block_dims=ctx.exact_thread_block_dims,
    )


def _simplify_proven_loop_bounds(s: _DeviceBodyState) -> None:
    from ...program_id import FlatProgramIDs
    from ...program_id import XYZProgramIDs
    from ..proven_loop_bounds import exact_static_grid
    from ..proven_loop_bounds import simplify_proven_loop_bounds

    ctx = s.ctx
    fn = ctx.fn
    # Ordinary collective launches use the PID strategy's grid
    # unchanged. Other launch schedulers retain architectural bounds.
    collective_grid = None
    if (
        fn.pid is not None
        and type(fn.pid) in (FlatProgramIDs, XYZProgramIDs)
        and fn.cute_state.lanes.simt_cluster_n == 1
        and fn.cute_state.lanes.collective_mma_sites
        and not fn.codegen.cute_wrapper_plans
    ):
        collective_grid = exact_static_grid(fn.pid.codegen_grid(), ctx.constexpr_values)
    s.kernel_body = simplify_proven_loop_bounds(
        s.kernel_body,
        enabled=bool(ctx.config.config.get("cute_proven_bounds", False)),
        thread_block_dims=ctx.exact_thread_block_dims,
        constexpr_values=ctx.constexpr_values,
        block_grid_dims=collective_grid,
    )


def _simplify_proven_bounds(s: _DeviceBodyState) -> None:
    from ..simplify_proven_bounds import simplify_proven_bounds

    ctx = s.ctx
    s.kernel_body = simplify_proven_bounds(
        s.kernel_body,
        enabled=cast("bool", ctx.config.config.get("cute_proven_bounds", False)),
        xyz_grid=ctx.xyz_grid,
        thread_block_dims=ctx.exact_thread_block_dims,
        block_grid_dims=ctx.block_grid_dims,
        constexpr_values=ctx.constexpr_values,
        proven_tensor_size_values=ctx.proven_tensor_size_values,
    )


def _fuse_cluster_online_pair(s: _DeviceBodyState) -> None:
    # Fuse an online-softmax (max, sum) pair of cluster reduces into
    # ONE packed DSM exchange: the max reduce relocalizes to a
    # CTA-local block reduce, and the sum site exchanges the
    # ``(local_max, local_sum)`` pair once and folds with the
    # online-softmax rescale.  Under fast_math the write sweep also
    # reuses the sum sweep's cached exp values; otherwise it keeps
    # computing them exactly.  Saves a full cluster round-trip per
    # row (+5..9% at cluster_n 2..16 on B200 softmax).
    from ..cluster_online_pair import fuse_cluster_online_pair

    s.kernel_body = fuse_cluster_online_pair(
        s.kernel_body,
        s.ctx.constexpr_values,
        rename_groups=s.ctx.rename_groups,
        fast_math=s.ctx.fast_math,
    )


def _lower_register_row_reduction(s: _DeviceBodyState) -> None:
    from ..register_row_reduction import lower_register_row_reduction

    ctx = s.ctx
    if ctx.config.get("cute_register_row_tensor_copy", False):
        from ..register_row_tensor_copy import context_from_device_function

        s.kernel_body = lower_register_row_reduction(
            s.kernel_body,
            ctx.constexpr_values,
            tensor_copy_context=context_from_device_function(
                ctx.fn, ctx.exact_thread_block_dims, ctx.tensor_dtypes_str_skip
            ),
        )
    else:
        s.kernel_body = lower_register_row_reduction(
            s.kernel_body, ctx.constexpr_values
        )


def _fuse_rolled_cluster_online_pair(s: _DeviceBodyState) -> None:
    # The rolled (``reduction_loops``) form's online pair: tried after
    # the register-row lowering, which consumes the cluster sites when
    # it fires, so the knob is honoured whatever ``cute_register_row``
    # says.  Under fast_math the CTA-level pair caches sweep B's
    # exponentials for the write sweep; otherwise the exact warp pair.
    from ..rolled_cluster_online_pair import fuse_rolled_cluster_online_pair

    s.kernel_body = fuse_rolled_cluster_online_pair(
        s.kernel_body,
        s.ctx.constexpr_values,
        rename_groups=s.ctx.rename_groups,
        fast_math=s.ctx.fast_math,
    )


def _split_rolled_vector_sums(s: _DeviceBodyState) -> None:
    from ..vector_reduction_packets import split_rolled_vector_sums

    s.kernel_body = split_rolled_vector_sums(s.kernel_body)


def _apply_exp2_fastmath(s: _DeviceBodyState) -> None:
    # ``ex2.approx.ftz`` is a numerics change (denormal exp outputs
    # flush to zero), so it is gated on the fast_math SETTING —
    # tuned configs must never change numerics.
    from ..exp2_fastmath import apply_exp2_fastmath

    s.kernel_body = apply_exp2_fastmath(s.kernel_body)


def _validate_cluster_reduce_placement(s: _DeviceBodyState) -> None:
    # The DSM cluster reduce uses a single-phase mbarrier, so each
    # call site must execute exactly ONCE per kernel.  The
    # lane-reduce collapse normally hoists it out of the (staged,
    # multi-trip) lane loop; when the collapse bails for a config,
    # the call would run once per lane iteration and DEADLOCK in
    # ``mbarrier_wait`` (an unkillable kernel that wedges the
    # autotuner's benchmark worker).  Reject such configs loudly so
    # the autotuner skips them.
    from ..hoist_warp_reduce import validate_cluster_reduce_placement

    validate_cluster_reduce_placement(s.kernel_body, s.ctx.constexpr_values)


def _bounded_cache_finalize(s: _DeviceBodyState) -> None:
    request = s.bounded_cache_request
    assert request is not None
    s.kernel_body = request.finalize(s.kernel_body)


def _lower_full_tile_bounds(s: _DeviceBodyState) -> None:
    from ..full_tile_bounds import lower_full_tile_bounds

    s.kernel_body = lower_full_tile_bounds(
        s.kernel_body,
        s.fn,
        s.param_args,
        s.ctx.constexpr_values,
        s.ctx.rename_groups,
    )


def _enable_fp32_warp_redux(s: _DeviceBodyState) -> None:
    from ..native_warp_reduction import enable_fp32_warp_redux

    s.kernel_body = enable_fp32_warp_redux(s.kernel_body)


def _lower_scalar_epilogue(s: _DeviceBodyState) -> None:
    from ..scalar_epilogue import lower_scalar_epilogue

    s.kernel_body = lower_scalar_epilogue(
        s.kernel_body,
        s.fn,
        s.ctx.constexpr_values,
        s.ctx.exact_thread_block_dims,
        s.ctx.proven_disjoint_tensor_pairs,
    )


def _lower_philox_packets(s: _DeviceBodyState) -> None:
    from ..affine_vector_io import vectorize_affine_tile_lanes
    from ..philox_packets import lower_philox_packets

    fn = s.fn
    constexpr_values = s.ctx.constexpr_values
    s.kernel_body = lower_philox_packets(
        s.kernel_body,
        seed_exprs=fn.cute_state.explicit_rng_seed_exprs,
        integer_names=set(constexpr_values),
        new_name=fn.unique_name,
        vectorize_packet=lambda loop, read, known: vectorize_affine_tile_lanes(
            [loop],
            fn,
            constexpr_values,
            register_accesses=frozenset({ast.dump(read, include_attributes=False)}),
            integer_names=known,
        ),
    )


def _wants_exact_division_redo(s: _DeviceBodyState) -> bool:
    ctx = s.ctx
    return (
        ctx.exact_division
        and not ctx.fast_math
        and ctx.config.cute_exact_div_form == "redo"
        and not ctx.matmul_kernel
    )


def _hoist_exact_invariant_divisions(s: _DeviceBodyState) -> None:
    # Without fast_math, hoist the divisor-only work of an fp32
    # ``x / d`` with a loop-invariant divisor as the Markstein
    # sequence, speculated over the loop and redone with div.rn
    # when a dividend left the admitted window: the IEEE quotient
    # bit for bit (exact_division.py), at three FMA-pipe
    # instructions and two FMNMX per element.  Last among the
    # structural passes: the sweep matchers above (independent
    # partial accumulators, pipelined and packed loads, the cluster
    # pairs, the scalar epilogue) recognize the plain ``v = x / d``
    # sweeps, so the redo render keeps the plain render's structure
    # and summation order.  ``cute_exact_div_form="plain"`` keeps
    # the plain division.  Not for matmul kernels, whose tcgen05
    # scheduling passes pin exact statement sequences (as fuse_fma).
    from ..hoist_loop_invariant_recip import hoist_exact_invariant_divisions

    ctx = s.ctx
    s.kernel_body = hoist_exact_invariant_divisions(
        s.kernel_body,
        ctx.rename_groups,
        float_scalar_names=set(ctx.float_scalar_names),
        tensor_dtypes=ctx.tensor_dtypes_str_skip,
        argument_names=set(ctx.signature_names),
        proven_disjoint_tensor_pairs=ctx.proven_disjoint_tensor_pairs,
    )


# ---------------------------------------------------------------------------
# Post-rename stage: ``finalize_kernel_def``
# ---------------------------------------------------------------------------


def _lower_online_chunk_reduction(s: _KernelDefState) -> None:
    from ..online_chunk_reduction import lower_online_chunk_reduction

    s.definition.body = lower_online_chunk_reduction(
        s.definition.body, s.ctx.constexpr_values
    )


def _reassociate_boolean_guards(s: _KernelDefState) -> None:
    from ..boolean_guards import reassociate_boolean_guards

    # Type facts must see the final binding names, including every
    # loop-carried alias, before changing the SDK's Boolean tree shape.
    s.definition.body = reassociate_boolean_guards(s.definition.body)


def _lower_published_scalars(s: _KernelDefState) -> None:
    for request in s.fn.cute_state.published_scalar_requests:
        request.lower(s.definition.body, s.ctx.rename_groups, s.fn.new_var)


def _lower_register_producers(s: _KernelDefState) -> None:
    for request in s.fn.cute_state.register_producer_requests:
        request.lower(s.definition.body, s.ctx.rename_groups, s.fn.new_var)


def _validate_thread_axis_accesses(s: _KernelDefState) -> None:
    from ..backend import validate_thread_axis_accesses

    validate_thread_axis_accesses([*s.prefix, s.definition])


def _lower_scalar_policy_loads(s: _KernelDefState) -> None:
    from ..scalar_policy_loads import lower_scalar_policy_loads

    # Keep ordinary load/effect recognition intact through all memory
    # scheduling, including the late published/register-producer cuts.
    s.definition.body = lower_scalar_policy_loads(s.definition.body)


def _lower_packet_store_policy(s: _KernelDefState) -> None:
    from ..packet_store_policy import lower_packet_store_policy

    variable_renames = s.ctx.rename_groups
    s.definition.body = lower_packet_store_policy(
        s.definition.body,
        policy=s.ctx.config.cute_store_policy,
        # Shared and register pointers are not global store targets.
        global_tensor_names=frozenset(
            variable_renames.get(argument.name, argument.name)
            for argument in s.param_args
            if isinstance(argument, TensorArg)
            and not isinstance(argument, TensorDescriptorArg)
        ),
        target_device_capability=s.ctx.target_device_capability,
        cuda_version=torch.version.cuda,
    )


# ---------------------------------------------------------------------------
# The ordered tables (the call sequence of ``codegen_function_def``, verbatim)
# ---------------------------------------------------------------------------

# Tags every table starts with: the facts the context carries.
CONTEXT_FACTS = "pass_context"
# Provided by ``freeze_pass_context``: the launch shape and signature are final.
LAUNCH_SHAPE = "launch_shape"
# Provided by the kernel-def stage's entry: ``ast_rename`` has run.
RENAMED = "renamed"

DEVICE_BODY_PROLOGUE: tuple[DeviceBodyPass, ...] = (
    DeviceBodyPass(
        "lower_uniform_comparisons",
        _always,
        _lower_uniform_comparisons,
        requires=(CONTEXT_FACTS,),
    ),
    DeviceBodyPass(
        "lower_collective_matmul",
        _always,
        _lower_collective_matmul,
        requires=(CONTEXT_FACTS,),
        provides=("collective_matmul_lowered",),
    ),
    DeviceBodyPass(
        "lower_simt_gemm", _always, _lower_simt_gemm, requires=(CONTEXT_FACTS,)
    ),
    DeviceBodyPass(
        "extend_gathered_mma_tma_launch",
        _has_gathered_mma_tma_plan,
        _extend_gathered_mma_tma_launch,
        # Reads the wrapper-only parameters the collective lowering added.
        requires=("collective_matmul_lowered",),
        provides=("gathered_mma_launch",),
    ),
    DeviceBodyPass(
        "adopt_owned_block_dims",
        _has_owned_block_dims,
        _adopt_owned_block_dims,
        provides=("owned_block_dims",),
    ),
    DeviceBodyPass(
        "bounded_cache_prepare",
        _has_bounded_cache_request,
        _bounded_cache_prepare,
        requires=(CONTEXT_FACTS,),
        provides=("bounded_cache_prepared",),
    ),
    DeviceBodyPass(
        "freeze_pass_context",
        _always,
        freeze_pass_context,
        requires=("gathered_mma_launch", "owned_block_dims", "bounded_cache_prepared"),
        provides=(LAUNCH_SHAPE,),
    ),
)

DEVICE_BODY_PASSES: tuple[DeviceBodyPass, ...] = (
    DeviceBodyPass(
        "finalize_shared_reduce_groups",
        _has_exact_thread_block_dims,
        _finalize_shared_reduce_groups,
        requires=(LAUNCH_SHAPE,),
    ),
    DeviceBodyPass(
        "fuse_two_pass_loads",
        _has_exact_thread_block_dims,
        _fuse_two_pass_loads,
        requires=(LAUNCH_SHAPE,),
        provides=("two_pass_loads_fused",),
    ),
    DeviceBodyPass(
        "vectorize_branch_local_persistent_fragments",
        _always,
        _vectorize_branch_local_persistent_fragments,
        # Its markers must survive the reduction split and the load fuser.
        requires=("two_pass_loads_fused",),
    ),
    DeviceBodyPass(
        "hoist_warp_reduce_from_vloop",
        _always,
        _hoist_warp_reduce_from_vloop,
        provides=("warp_reduces_hoisted",),
    ),
    DeviceBodyPass(
        "eliminate_duplicate_cluster_maxima",
        _has_simt_cluster,
        _eliminate_duplicate_cluster_maxima,
    ),
    DeviceBodyPass(
        "bounded_cache_cache",
        _has_bounded_cache_request,
        _bounded_cache_cache,
        requires=("bounded_cache_prepared",),
        provides=("bounded_cache_cached", "private_fragments"),
    ),
    DeviceBodyPass(
        "vectorize_affine_tile_lanes",
        _always,
        _vectorize_affine_tile_lanes,
        requires=("private_fragments",),
    ),
    DeviceBodyPass("merge_literal_offset_loads", _always, _merge_literal_offset_loads),
    DeviceBodyPass("fuse_fp4_pair_dot", _always, _fuse_fp4_pair_dot),
    DeviceBodyPass("merge_sibling_v_loops", _always, _merge_sibling_v_loops),
    DeviceBodyPass(
        "optimize_vector_reductions",
        _always,
        _optimize_vector_reductions,
        requires=(LAUNCH_SHAPE,),
    ),
    DeviceBodyPass(
        "hoist_lane_invariant_reductions",
        _always,
        _hoist_lane_invariant_reductions,
        requires=(LAUNCH_SHAPE,),
    ),
    DeviceBodyPass("hoist_factored_reductions", _fast_math, _hoist_factored_reductions),
    DeviceBodyPass("fuse_fp8_pair_decode", _always, _fuse_fp8_pair_decode),
    DeviceBodyPass(
        "hoist_loop_invariant_recips", _always, _hoist_loop_invariant_recips
    ),
    DeviceBodyPass("fuse_fma", _not_matmul_kernel, _fuse_fma, provides=("fma_fused",)),
    DeviceBodyPass(
        "ring_mma_dot",
        _config_flag(RING_MMA_KEY),
        _ring_mma_dot,
        requires=(LAUNCH_SHAPE, "fma_fused"),
        provides=("ring_mma_rewritten",),
    ),
    DeviceBodyPass(
        "fragment_mma_dot",
        _config_flag(FRAGMENT_MMA_KEY),
        _fragment_mma_dot,
        requires=(LAUNCH_SHAPE, "ring_mma_rewritten"),
        provides=("fragment_mma_rewritten",),
    ),
    DeviceBodyPass(
        "plan_lane_unroll",
        _always,
        _plan_lane_unroll,
        requires=("fragment_mma_rewritten",),
        provides=("lane_unroll_plan",),
    ),
    DeviceBodyPass(
        "unroll_lane_loads",
        _wants_lane_unroll,
        _unroll_lane_loads,
        requires=("lane_unroll_plan",),
        provides=("lane_loads_unrolled",),
    ),
    DeviceBodyPass(
        "fold_lane_reductions",
        _has_folded_lanes,
        _fold_lane_reductions,
        requires=("lane_unroll_plan", "lane_loads_unrolled"),
    ),
    DeviceBodyPass(
        "split_lane_loads",
        _always,
        _split_lane_loads,
        requires=("lane_loads_unrolled",),
    ),
    DeviceBodyPass(
        "prefetch_reduction_packets",
        _has_exact_thread_block_dims,
        _prefetch_reduction_packets,
        # Preserves the global FMA decisions: packets move, arithmetic stays.
        requires=(LAUNCH_SHAPE, "fma_fused"),
    ),
    DeviceBodyPass("pipeline_inner_loads", _always, _pipeline_inner_loads),
    DeviceBodyPass(
        "pipeline_state_loads",
        _wants_pipelined_state_loads,
        _pipeline_state_loads,
        requires=(LAUNCH_SHAPE,),
        provides=("state_loads_pipelined",),
    ),
    DeviceBodyPass(
        "pack_fp32_constexpr_loops",
        _always,
        _pack_fp32_constexpr_loops,
        # Packing first would obscure the state-load matcher.
        requires=("state_loads_pipelined",),
        provides=("fp32_loops_packed",),
    ),
    DeviceBodyPass(
        "hoist_packed_factored_reductions",
        _always,
        _hoist_packed_factored_reductions,
        requires=("fp32_loops_packed",),
    ),
    DeviceBodyPass(
        "pack_bf16_recurrences",
        _always,
        _pack_bf16_recurrences,
        # Needs the proved private recurrence and the generic fp32 pairs.
        requires=(LAUNCH_SHAPE, "state_loads_pipelined", "fp32_loops_packed"),
    ),
    DeviceBodyPass(
        "simplify_proven_loop_bounds",
        _always,
        _simplify_proven_loop_bounds,
        requires=(LAUNCH_SHAPE,),
    ),
    DeviceBodyPass(
        "simplify_proven_bounds",
        _always,
        _simplify_proven_bounds,
        requires=(LAUNCH_SHAPE,),
    ),
    DeviceBodyPass(
        "fuse_cluster_online_pair",
        _always,
        _fuse_cluster_online_pair,
        provides=("cluster_pairs_fused",),
    ),
    DeviceBodyPass(
        "lower_register_row_reduction",
        _config_flag("cute_register_row"),
        _lower_register_row_reduction,
        requires=(LAUNCH_SHAPE,),
        provides=("register_rows_lowered",),
    ),
    DeviceBodyPass(
        "fuse_rolled_cluster_online_pair",
        _config_flag("cute_cluster_warp_online"),
        _fuse_rolled_cluster_online_pair,
        # The register-row lowering consumes the cluster sites when it fires.
        requires=("register_rows_lowered",),
        provides=("rolled_cluster_pairs_fused",),
    ),
    DeviceBodyPass(
        "split_rolled_vector_sums",
        _config_flag("cute_independent_reduction"),
        _split_rolled_vector_sums,
        provides=("independent_sums_split",),
    ),
    DeviceBodyPass("apply_exp2_fastmath", _fast_math, _apply_exp2_fastmath),
    DeviceBodyPass(
        "validate_cluster_reduce_placement",
        _always,
        _validate_cluster_reduce_placement,
        # A check row: every pass that moves a cluster reduce ran.
        requires=(
            "warp_reduces_hoisted",
            "cluster_pairs_fused",
            "rolled_cluster_pairs_fused",
        ),
    ),
    DeviceBodyPass(
        "bounded_cache_finalize",
        _has_bounded_cache_request,
        _bounded_cache_finalize,
        requires=("bounded_cache_cached",),
    ),
    DeviceBodyPass("lower_full_tile_bounds", _always, _lower_full_tile_bounds),
    DeviceBodyPass(
        "enable_fp32_warp_redux",
        _config_flag("cute_warp_fp32_redux"),
        _enable_fp32_warp_redux,
    ),
    DeviceBodyPass(
        "lower_scalar_epilogue",
        _config_flag("cute_scalar_epilogue"),
        _lower_scalar_epilogue,
        requires=(LAUNCH_SHAPE,),
        provides=("scalar_epilogue_lowered",),
    ),
    DeviceBodyPass(
        "lower_philox_packets", _config_flag("cute_rng_packet"), _lower_philox_packets
    ),
    DeviceBodyPass(
        "hoist_exact_invariant_divisions",
        _wants_exact_division_redo,
        _hoist_exact_invariant_divisions,
        # Last among the structural passes: the sweep matchers above must
        # see the plain ``v = x / d`` sweeps.
        requires=(
            "independent_sums_split",
            "state_loads_pipelined",
            "fp32_loops_packed",
            "cluster_pairs_fused",
            "rolled_cluster_pairs_fused",
            "scalar_epilogue_lowered",
        ),
    ),
)

KERNEL_DEF_PASSES: tuple[KernelDefPass, ...] = (
    KernelDefPass(
        "lower_online_chunk_reduction",
        _config_flag("cute_register_row"),
        _lower_online_chunk_reduction,
        requires=(RENAMED,),
    ),
    KernelDefPass(
        "reassociate_boolean_guards",
        _always,
        _reassociate_boolean_guards,
        # Type facts must see the final binding names.
        requires=(RENAMED,),
    ),
    KernelDefPass(
        "lower_published_scalars",
        _config_flag("cute_fragment_published_scalars"),
        _lower_published_scalars,
        requires=(RENAMED,),
        provides=("published_scalars_lowered",),
    ),
    KernelDefPass(
        "lower_register_producers",
        _config_flag("cute_fragment_register_producers"),
        _lower_register_producers,
        requires=(RENAMED,),
        provides=("register_producers_lowered",),
    ),
    KernelDefPass(
        "validate_thread_axis_accesses",
        _always,
        _validate_thread_axis_accesses,
        # A check row over the renamed definition after the late cuts.
        requires=(RENAMED, "published_scalars_lowered", "register_producers_lowered"),
    ),
    KernelDefPass(
        "lower_scalar_policy_loads",
        _always,
        _lower_scalar_policy_loads,
        # Load/effect recognition stays intact through the late memory cuts.
        requires=("published_scalars_lowered", "register_producers_lowered"),
    ),
    KernelDefPass(
        "lower_packet_store_policy",
        _always,
        _lower_packet_store_policy,
        requires=(RENAMED,),
    ),
)


def _check_pass_order(
    stages: tuple[tuple[str, tuple[DeviceBodyPass | KernelDefPass, ...]], ...],
    provided: set[str],
) -> None:
    """Every ``requires`` tag names a tag an earlier entry (or the stage's
    inputs) provides, and no tag is provided twice: the tables form a DAG."""
    provided = set(provided)
    for stage, entries in stages:
        for entry in entries:
            missing = set(entry.requires) - provided
            if missing:
                raise AssertionError(
                    f"{stage}: {entry.name} requires {sorted(missing)} before any "
                    "entry provides them"
                )
            repeated = set(entry.provides) & provided
            if repeated:
                raise AssertionError(
                    f"{stage}: {entry.name} provides {sorted(repeated)} twice"
                )
            provided |= set(entry.provides)


_check_pass_order(
    (
        ("DEVICE_BODY_PROLOGUE", DEVICE_BODY_PROLOGUE),
        ("DEVICE_BODY_PASSES", DEVICE_BODY_PASSES),
        ("KERNEL_DEF_PASSES", KERNEL_DEF_PASSES),
    ),
    {CONTEXT_FACTS, RENAMED},
)


def run_device_body_passes(
    body: list[ast.stmt], ctx: DeviceBodyContext
) -> tuple[list[ast.stmt], list[ast.arg]]:
    """``CuteBackend.finalize_device_body``: the pre-rename stage.

    Returns the rewritten body and the kernel parameters the stage appended
    to ``ctx.args``.  ``ctx.constexpr_values`` is filled and
    ``ctx.pass_context`` set for the post-rename stage (the proven launch
    shape is ``ctx.pass_context.exact_thread_block_dims``).
    ``LaneUnrollNotApplied``, ``BackendUnsupported`` and ``InvalidConfig``
    propagate unchanged: ``generate_ast`` and the autotuner act on them.
    """
    state = _DeviceBodyState(body, ctx)
    for entry in DEVICE_BODY_PROLOGUE:
        if entry.gate(state):
            entry.run(state)
    for entry in DEVICE_BODY_PASSES:
        if entry.gate(state):
            entry.run(state)
    ctx.pass_context = state.ctx
    return state.kernel_body, state.args[len(ctx.args) :]


def run_kernel_def_passes(
    definition: ast.FunctionDef, prefix: list[ast.stmt], ctx: DeviceBodyContext
) -> None:
    """``CuteBackend.finalize_kernel_def``: the post-rename stage, in place."""
    state = _KernelDefState(definition, prefix, ctx)
    for entry in KERNEL_DEF_PASSES:
        if entry.gate(state):
            entry.run(state)
