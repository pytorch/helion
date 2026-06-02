"""Device IR lowering pipeline.

Lowers traced FX graphs into a fully-populated ``DeviceIR`` in five stages:
trace, transform, lower, optimize, register.

``DeviceIRLowering`` provides the shared implementation.
Backend subclasses override individual stages to customize the pipeline.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from .._compile_time import measure
from .compile_environment import CompileEnvironment
from .tensor_utils import patch_tensor_factories

if TYPE_CHECKING:
    from .device_ir import DeviceIR
    from .host_function import HostFunction


class DeviceIRLowering:
    """Default compilation pipeline from traced FX graphs to DeviceIR.

    Backends subclass this and override individual stages to customize
    the pipeline.  Shared helper functions are importable from this
    module so backend implementations can compose them freely.
    """

    def run(self, func: HostFunction) -> DeviceIR:
        """Execute the full lowering pipeline."""
        from .device_ir import DeviceIR

        device_ir = DeviceIR()
        env = CompileEnvironment.current()
        factory_padding = (
            patch_tensor_factories()
            if env.backend.pad_factory_tensors_to_power_of_2
            else contextlib.nullcontext()
        )
        from torch._dynamo.convert_frame import compile_lock

        with func, device_ir, compile_lock, factory_padding:
            with measure("DeviceIRLowering.trace"):
                self.trace(device_ir, func)
            with measure("DeviceIRLowering.transform"):
                self.transform(device_ir, func)
            with measure("DeviceIRLowering.lower"):
                self.lower(device_ir)
            with measure("DeviceIRLowering.optimize"):
                self.optimize(device_ir)
            with measure("DeviceIRLowering.register"):
                self.register(device_ir)
            return device_ir

    def trace(self, device_ir: DeviceIR, func: HostFunction) -> None:
        """Walk host AST, produce FX graphs and kernel phases."""
        from .. import exc
        from .device_ir import RootGraphInfo
        from .device_ir import WalkHostAST

        visitor = WalkHostAST(device_ir)
        for stmt in func.body:
            visitor.visit(stmt)
        visitor.flush_phases()
        device_ir.phases = visitor.phases

        # Run dependency checks once per phase so codegen does not
        # redo them per-config.
        for phase in device_ir.phases:
            checker = phase.loop_dependency_checker
            for loop_node in phase.root_nodes:
                checker.register_loop(loop_node)

        for phase_idx, phase in enumerate(device_ir.phases):
            for ridx in phase.roots:
                graph_info = device_ir.graphs[device_ir.root_ids[ridx]]
                assert isinstance(graph_info, RootGraphInfo)
                graph_info.phase_index = phase_idx

        if len(device_ir.root_ids) == 0:
            raise exc.NoDeviceLoopsInKernel

    def transform(self, device_ir: DeviceIR, func: HostFunction) -> None:
        """Rewrite graphs before lowering attachment."""
        rewrite_random_ops(device_ir)

    def lower(self, device_ir: DeviceIR) -> None:
        """Attach structured op representations to FX nodes."""
        prepare_graph_lowerings(device_ir)

    def optimize(self, device_ir: DeviceIR) -> None:
        """Validate, simplify, and annotate the lowered IR."""
        validate_host_tensor_usage(device_ir)
        add_tile_with_offset_metadata(device_ir)
        remove_unnecessary_tile_index(device_ir)
        remove_unnecessary_masking(device_ir)

    def register(self, device_ir: DeviceIR) -> None:
        """Register config spec entries for the autotuner."""
        device_ir.register_rollable_reductions()
        detect_epilogue_subtiling(device_ir)
        register_grid_and_pid_constraints(device_ir)


# ── Shared helpers ──────────────────────────────────────────────────
# Public functions that backend DeviceIRLowering implementations
# import and compose as needed.


def rewrite_random_ops(device_ir: DeviceIR) -> None:
    from ..language.random_ops import rewrite_implicit_random_ops

    for graph in device_ir.graphs:
        rewrite_implicit_random_ops(graph.graph)


def prepare_graph_lowerings(device_ir: DeviceIR) -> None:
    from .inductor_lowering import prepare_graph_lowerings as _prepare

    for graph in device_ir.graphs:
        _prepare(graph.graph)


def validate_host_tensor_usage(device_ir: DeviceIR) -> None:
    from .device_ir import validate_host_tensor_usage as _validate

    for graph in device_ir.graphs:
        _validate(graph.graph)


def add_tile_with_offset_metadata(device_ir: DeviceIR) -> None:
    from .device_ir import add_tile_with_offset_metadata as _add_metadata

    for graph in device_ir.graphs:
        _add_metadata(graph)


def remove_unnecessary_tile_index(device_ir: DeviceIR) -> None:
    from .device_ir import remove_unnecessary_tile_index as _remove_tile_index

    for graph in device_ir.graphs:
        _remove_tile_index(graph.graph)


def remove_unnecessary_masking(device_ir: DeviceIR) -> None:
    from .node_masking import remove_unnecessary_masking as _remove_masking

    for graph in device_ir.graphs:
        _remove_masking(graph.graph)


def detect_epilogue_subtiling(device_ir: DeviceIR) -> None:
    from .epilogue_subtiling import has_epilogue_subtiling_candidate

    has_candidate = False
    for graph_info in device_ir.graphs:
        if has_epilogue_subtiling_candidate(graph_info.graph):
            has_candidate = True
            break
    config_spec = CompileEnvironment.current().config_spec
    config_spec.epilogue_subtile_candidate_enabled = has_candidate
    config_spec.epilogue_subtile_k_hint = 0
    config_spec.epilogue_subtile_autotune_choices = None


def register_grid_and_pid_constraints(device_ir: DeviceIR) -> None:
    from .. import exc

    config_spec = CompileEnvironment.current().config_spec
    config_spec.raise_grid_block_minimums()

    if len(device_ir.root_ids) > 1:
        config_spec.disallow_pid_type("xyz")
        if config_spec.cute_tcgen05_search_enabled:
            non_persistent_pid_types = tuple(
                pid_type
                for pid_type in config_spec.allowed_pid_types
                if pid_type not in ("persistent_blocked", "persistent_interleaved")
            )
            if not non_persistent_pid_types:
                raise exc.InvalidConfig(
                    "CuTe tcgen05 multi-root kernels do not support "
                    "persistent pid types yet, and no non-persistent "
                    "pid type is available. Disable forced/distributed "
                    "persistent-only mode or use a single root loop."
                )
            config_spec.allowed_pid_types = non_persistent_pid_types


def register_load_store_tunables(device_ir: DeviceIR) -> None:
    from ..autotuner.config_fragment import EnumFragment
    from ..autotuner.config_fragment import ListOf
    from ..autotuner.config_spec import get_valid_eviction_policies
    from ..language import memory_ops

    env = CompileEnvironment.current()

    total_load_count = 0
    loads_without_eviction_policy = 0
    memory_op_index = 0
    store_indices: list[int] = []

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function":
                if node.target is memory_ops.load:
                    total_load_count += 1
                    memory_op_index += 1
                    eviction_policy_arg = node.kwargs.get("eviction_policy")
                    if eviction_policy_arg is None:
                        if len(node.args) >= 4:
                            eviction_policy_arg = node.args[3]
                        if eviction_policy_arg is None:
                            loads_without_eviction_policy += 1
                elif node.target is memory_ops.store:
                    store_indices.append(memory_op_index)
                    memory_op_index += 1

    store_count = len(store_indices)
    env.config_spec.store_indices = store_indices
    if total_load_count == 0 and store_count == 0:
        return

    if loads_without_eviction_policy > 0:
        env.config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(env.backend_name)),
            length=loads_without_eviction_policy,
        )

    total_count = total_load_count + store_count
    if total_count > 0:
        env.config_spec.indexing = ListOf(
            EnumFragment(choices=env.config_spec.valid_indexing_types()),
            length=total_count,
        )


def register_atomic_tunables(device_ir: DeviceIR) -> None:
    from ..autotuner.config_fragment import EnumFragment
    from ..autotuner.config_fragment import ListOf
    from ..language import atomic_ops

    atomic_count = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op == "call_function" and node.target in (
                getattr(atomic_ops, name) for name in atomic_ops.__all__
            ):
                atomic_count += 1

    if atomic_count == 0:
        return

    env = CompileEnvironment.current()
    env.config_spec.atomic_indexing = ListOf(
        EnumFragment(choices=env.config_spec.valid_atomic_indexing_types()),
        length=atomic_count,
    )


def register_tensor_descriptor_layout_guards(device_ir: DeviceIR) -> None:
    env = CompileEnvironment.current()
    if env.settings.static_shapes:
        return

    from .._compat import supports_tensor_descriptor
    from ..language import atomic_ops
    from ..language import memory_ops

    if not supports_tensor_descriptor():
        return

    atomic_targets = tuple(getattr(atomic_ops, name) for name in atomic_ops.__all__)

    def tensor_arg_value(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val")
        return arg

    memory_op_index = 0
    atomic_op_index = 0
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in (memory_ops.load, memory_ops.store):
                tensor = tensor_arg_value(node.args[0])
                if isinstance(tensor, torch.Tensor) and 2 <= tensor.ndim <= 5:
                    env.register_tensor_descriptor_layout_guard(
                        tensor, memory_op_index=memory_op_index
                    )
                memory_op_index += 1
                continue
            if node.target in atomic_targets:
                tensor = tensor_arg_value(node.args[0])
                if isinstance(tensor, torch.Tensor) and 2 <= tensor.ndim <= 5:
                    env.register_tensor_descriptor_layout_guard(
                        tensor, atomic_op_index=atomic_op_index
                    )
                atomic_op_index += 1
