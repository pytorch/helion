"""Sink a host constant fill of a kernel output's unwritten remainder into the kernel.

``grad = torch.empty_like(logits); <device loops store grad[tile_b, tile_l,
tile_v] for tile_l over L = L_ADD_1 - 1 rows>; grad[:, -1, :] = 0`` launches
ATen's fill kernel after the Helion kernel: one more graph node (about 0.7
us per call at the smallest catalogue shapes) for a few kilobytes. When the
kernel's stores provably cover everything but that remainder and the kernel
never reads the tensor, the fill commutes with the kernel's own work, so the
kernel writes it: every thread of every CTA stores its share of the region
at the end of the body (:func:`_cute_remainder_fill`) and the host statement
is dropped. The value is exact (a constant representable in the dtype), so
numerics are unchanged.

Admission (built on the zero-fill coverage proof): a top-level host
statement after the last device loop ``name[...] = <int or float constant>``
whose slices are the ``_remainder_fill`` shape (full ranges except one
``-r`` / ``-r:`` dimension); every device access to the tensor is a complete
tile store with that remainder (no loads, no masked, conditional, atomic or
view stores); no host statement between the launch and the fill reads or
rebinds the name; the tensor is a kernel argument of rank <= 4; the kernel
body has no early return and no grid barrier.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..device_function import TensorArg
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..type_info import TensorType
from .host_zero_fills import _assigns
from .host_zero_fills import _is_load
from .host_zero_fills import _reads
from .host_zero_fills import _remainder_fill
from .host_zero_fills import _store_coverage
from .host_zero_fills import _top_graph

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import GenerateAST
    from ..host_function import HostFunction
    from ..source_location import SourceLocation

FILL_HELPER = "_cute_remainder_fill"
# The helper (remainder_fill_runtime, imported by the generated module only:
# it needs the DSL) writes its coordinate decomposition out per rank.
MAX_RANK = 4
_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
)


@dataclass(frozen=True)
class RemainderFill:
    location: SourceLocation
    statement: str
    name: str
    fake: torch.Tensor
    dim: int
    remainder: int
    value: int | float


def _constant_fill(statement: ast.AST) -> tuple[str, int | float] | None:
    """``name[...] = <int or float constant>`` (no bool, no expression)."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(target := statement.targets[0], ast.Subscript)
        and isinstance(target.value, ast.Name)
        and isinstance(statement.value, ast.Constant)
    ):
        return None
    value = statement.value.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return target.value.id, value


def _exact_in_dtype(value: float, dtype: torch.dtype) -> bool:
    if dtype not in _DTYPES:
        return False
    if not dtype.is_floating_point and type(value) is not int:
        return False
    try:
        converted = torch.tensor(value, dtype=dtype).item()
    except (RuntimeError, OverflowError):
        return False
    return converted == value and (dtype.is_floating_point or -(2**31) <= value < 2**31)


def prove_host_remainder_fills(
    host: HostFunction, env: CompileEnvironment, config: Config
) -> list[RemainderFill]:
    device_ir = host.device_ir
    graphs = device_ir.graphs
    skewed = device_ir.noncanonical_task_origin_block_ids
    roots = {
        graphs[root_id].graph: set(block_ids) - skewed
        for root_id, block_ids in zip(
            device_ir.root_ids, device_ir.grid_block_ids, strict=True
        )
        if isinstance(graphs[root_id], RootGraphInfo)
    }
    if not roots:
        return []
    nodes = list(ast.walk(ast.Module(body=host.body, type_ignores=[])))
    if any(
        isinstance(node, (ast.Delete, ast.Try, ast.Global, ast.Nonlocal))
        for node in nodes
    ) or any(isinstance(node, ast.Name) and node.id == FILL_HELPER for node in nodes):
        return []
    device_loops = [
        index
        for index, statement in enumerate(host.body)
        if isinstance(statement, ExtendedAST) and statement._loop_type == LoopType.GRID
    ]
    if not device_loops:
        return []
    last_device_loop = device_loops[-1]
    host_tensors: dict[torch.Tensor, list[torch.fx.Node]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _host_tensor_target():
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    host_tensors.setdefault(value, []).append(node)
    parents = control_flow_parent_entries(graphs)
    result: list[RemainderFill] = []
    seen: set[str] = set()
    for position in range(last_device_loop + 1, len(host.body)):
        statement = host.body[position]
        found = _constant_fill(statement)
        if found is None:
            continue
        name, value = found
        if name in seen:
            continue
        seen.add(name)
        assert isinstance(statement, ast.Assign)
        target = statement.targets[0]
        assert isinstance(target, ast.Subscript)
        tensor_type = (
            target.value._type_info if isinstance(target.value, ExtendedAST) else None
        )
        if not isinstance(tensor_type, TensorType) or not isinstance(
            statement, ExtendedAST
        ):
            continue
        fake = tensor_type.fake_value
        if fake.ndim > MAX_RANK or not _exact_in_dtype(value, fake.dtype):
            continue
        tensors = host_tensors.get(fake, [])
        if not tensors:
            continue
        coverages: set[tuple[int, ...] | None] = set()
        admitted = True
        for tensor in tensors:
            for user in tensor.users:
                top = _top_graph(user, graphs, parents)
                if top.graph not in roots:
                    if isinstance(top, ReductionLoopGraphInfo):
                        continue
                    admitted = False
                    break
                if _is_load(user, tensor):
                    admitted = False
                    break
                coverages.add(
                    _store_coverage(
                        user, tensor, fake, env, graphs, parents, roots, host, config
                    )
                )
            if not admitted:
                break
        if not admitted or len(coverages) != 1 or None in coverages:
            continue
        (remainders,) = coverages
        assert remainders is not None
        if sum(1 for remainder in remainders if remainder) != 1:
            continue
        if not _remainder_fill(statement, name, remainders):
            continue
        between = host.body[last_device_loop + 1 : position]
        if any(_reads(other, name) or _assigns(other, name) for other in between):
            continue
        dim = next(index for index, remainder in enumerate(remainders) if remainder)
        result.append(
            RemainderFill(
                statement._location,
                ast.unparse(statement),
                name,
                fake,
                dim,
                remainders[dim],
                value,
            )
        )
    return result


def _host_tensor_target() -> object:
    from ...language import _tracing_ops

    return _tracing_ops._host_tensor


def append_device_remainder_fills(
    codegen: GenerateAST, host: HostFunction, config: Config
) -> list[RemainderFill]:
    """After the kernel body is complete, append one device fill per proven site.

    Returns the sites whose host statements must be dropped; a kernel the
    device side cannot host (early returns, a grid barrier, a tensor that is
    not an argument) keeps every host fill.
    """
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    fn = codegen.device_function
    fills = prove_host_remainder_fills(host, env, config)
    if not fills:
        return []
    if any(argument.name.startswith("x_grid_sem") for argument in fn.arguments):
        return []
    if any(
        isinstance(node, ast.Return)
        for statement in fn.body
        for node in ast.walk(statement)
    ):
        return []
    arguments = {
        id(argument.fake_value): argument
        for argument in fn.arguments
        if type(argument) is TensorArg
    }
    accepted: list[RemainderFill] = []
    for fill in fills:
        argument = arguments.get(id(fill.fake))
        if argument is None:
            continue
        # The device statement carries the host fill's own source location.
        with fill.location:
            fn.body.append(
                statement_from_string(
                    f"{FILL_HELPER}({argument.name}, dim={fill.dim}, "
                    f"remainder={fill.remainder}, value={fill.value!r}, "
                    f"is_float={fill.fake.dtype.is_floating_point!r})"
                )
            )
        accepted.append(fill)
    return accepted


def lower_host_remainder_fills(
    body: list[ast.AST], fills: list[RemainderFill]
) -> list[ast.AST]:
    """Drop the host fills the kernel now performs."""
    if not fills:
        return body
    sites = {fill.location: fill for fill in fills}
    result: list[ast.AST] = []
    for statement in body:
        fill = (
            sites.get(statement._location)
            if isinstance(statement, ExtendedAST)
            else None
        )
        if fill is not None and ast.unparse(statement) == fill.statement:
            continue
        result.append(statement)
    assert len(result) == len(body) - len(fills), "remainder fill statement missing"
    return result
