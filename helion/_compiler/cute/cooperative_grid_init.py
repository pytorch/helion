"""Bring the private software grid counter into an already cooperative invocation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ...exc import InvalidConfig
from ..ast_extension import statement_from_string
from . import cooperative_grid_ptx
from ._ast_pass_utils import call_path
from ._ast_pass_utils import phase_barrier_call
from .autotune.config_keys import GRID_INIT_KEY
from .cooperative_grid_ptx import grid_sync_rejected
from .cooperative_grid_ptx import grid_sync_requirement
from .cooperative_grid_ptx import grid_toolkit_installed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import Argument
    from ..device_function import DeviceFunction


def grid_init_requested(config: Mapping[str, object]) -> bool:
    value = config.get(GRID_INIT_KEY, False)
    if type(value) is not bool:
        raise InvalidConfig(f"{GRID_INIT_KEY} must be a boolean, got {value!r}")
    return value


def _software_grid_surface(spec: ConfigSpec) -> bool:
    """Binding geometry shared by the software and toolkit rendezvous."""
    pipeline = spec.cross_loop_pipeline
    cap = spec.target_device_capability
    return (
        spec.cute_local_grid_init_available
        and any(
            pid in {"persistent_interleaved", "persistent_blocked"}
            for pid in spec.allowed_pid_types
        )
        and spec.multi_phase
        and not spec.cute_dependent_phase_launches
        and (pipeline is None or "barrier" in pipeline.choices)
        and cap is not None
        and cap >= (7, 0)
    )


def grid_init_surface(spec: ConfigSpec) -> bool:
    cap = spec.target_device_capability
    return (
        not spec.unversioned_aot_bundle
        and _software_grid_surface(spec)
        and cap is not None
        and grid_toolkit_installed(cap)
    )


def grid_software_carrier(spec: ConfigSpec, config: dict[str, object]) -> bool:
    # Raw seeds omit the preferred PID until ConfigSpec normalizes them.
    return (
        (pid := config.get("pid_type", spec.allowed_pid_types[0]))
        in spec.allowed_pid_types
        and pid in ("persistent_interleaved", "persistent_blocked")
        and config.get("cross_loop_pipeline", "barrier") == "barrier"
        and _software_grid_surface(spec)
    )


def grid_init_carrier(spec: ConfigSpec, config: dict[str, object]) -> bool:
    cap = spec.target_device_capability
    return (
        not spec.unversioned_aot_bundle
        and grid_software_carrier(spec, config)
        and cap is not None
        and grid_toolkit_installed(cap)
    )


def grid_init_schedule(spec: ConfigSpec, config: dict[str, object]) -> bool:
    return (
        grid_init_carrier(spec, config)
        and (cap := spec.target_device_capability) is not None
        and not grid_sync_rejected(cap)
    )


def initialize_grid_counter(
    body: list[ast.stmt], fn: DeviceFunction, spec: ConfigSpec
) -> None:
    if not fn.config.get(GRID_INIT_KEY):
        return
    cap = spec.target_device_capability
    asm = None if cap is None else cooperative_grid_ptx.grid_sync_asm(cap)
    if asm is None:
        raise InvalidConfig(
            "in-kernel grid initialization needs " + grid_sync_requirement(cap)
        )
    # At this point the phase scheduler and multiroot optimizations have
    # finished. Only an unconditional local join can publish the initial zero.
    sites = [
        (i, call)
        for i, stmt in enumerate(body)
        if (call := phase_barrier_call(stmt)) is not None
    ]
    if not sites or any(
        call_path(call.func) != ("_cute_grid_barrier",) for _, call in sites
    ):
        raise InvalidConfig(
            "in-kernel grid initialization needs unconditional local grid joins"
        )
    first, start = sites[0]
    operand = start.args[0]
    if not (
        isinstance(operand, ast.Attribute)
        and operand.attr == "iterator"
        and isinstance(operand.value, ast.Name)
    ):
        raise InvalidConfig("the local grid join does not use a private tensor")
    name = operand.value.id
    # This argument must have no observable use other than these local
    # rendezvous. Reject conditional joins and aliases.
    refs = [
        node
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and node.id == name
    ]
    if len(refs) != len(sites) or any(
        call_path(call.args[0]) != (name, "iterator") for _, call in sites
    ):
        raise InvalidConfig("cannot prove exclusive use of the local grid counter")
    if name != fn.cute_state.private_grid_init_arg:
        raise InvalidConfig(
            "the first local join does not use the host's private counter"
        )
    body[first : first + 1] = [
        statement_from_string(
            "if cute.arch.block_idx()[0] == 0 and cute.arch.block_idx()[1] == 0 and cute.arch.block_idx()[2] == 0 "
            "and cute.arch.thread_idx()[0] == 0 and cute.arch.thread_idx()[1] == 0 and cute.arch.thread_idx()[2] == 0:\n"
            f"    {name}[0] = cutlass.Uint32(0)"
        ),
        statement_from_string(f"_cute_toolkit_grid_sync({asm!r})"),
    ]


def empty_private_counter(arg: Argument, host_str: str) -> str:
    from ..device_function import TensorArg

    allocation = ast.parse(host_str, mode="eval").body
    if not (
        isinstance(arg, TensorArg)
        and arg.fake_value.dtype is torch.uint32
        and tuple(arg.fake_value.shape) == (1,)
        and isinstance(allocation, ast.Call)
        and call_path(allocation.func) == ("torch", "zeros")
        and {kw.arg for kw in allocation.keywords} == {"device", "dtype"}
        and len(allocation.keywords) == 2
        and len(allocation.args) == 1
        and isinstance(allocation.args[0], ast.Tuple)
        and len(allocation.args[0].elts) == 1
        and isinstance(extent := allocation.args[0].elts[0], ast.Constant)
        and type(extent.value) is int
        and extent.value == 1
    ):
        raise InvalidConfig(
            "the cooperative grid counter must be a private zeroed word"
        )
    assert isinstance(allocation.func, ast.Attribute)
    allocation.func.attr = "empty"
    return ast.unparse(allocation)
