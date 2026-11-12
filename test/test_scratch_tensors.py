"""CPU tests: which host allocations are kernel-local scratch tensors.

``helion._compiler.scratch_tensors.kernel_local_scratch`` proves that nothing
outside a kernel observes a ``torch.empty``-family host allocation that only
the kernel's loads and stores use; anything it cannot prove declines.  Each
kernel below binds on CPU (no code generation) and checks the analysis.
"""

from __future__ import annotations

import ast
import os
import types
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler import generate_ast
from helion._compiler import scratch_tensors
from helion._compiler.scratch_tensors import kernel_local_scratch
from helion._compiler.triton.backend import TritonBackend
from helion._testing import _get_backend
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.scratch_kernels import returned_unread_workspace
from helion._testing.scratch_kernels import two_sweep_workspace
from helion._testing.scratch_kernels import unread_workspace
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [
    pytest.mark.cpu_only,
    # The analysis reads the compiler's host function, which ref eager mode
    # (``HELION_INTERPRET=1``) never builds.
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="no host function in ref eager mode",
    ),
]


def _scratch(fn: object, *args: torch.Tensor) -> set[str]:
    bound = cpu_bind(helion.kernel(fn, static_shapes=True), args)
    return set(kernel_local_scratch(bound.host_function).names)


def workspace_like_and_new_empty(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty_like(x, dtype=torch.float32)
    ws2 = x.new_empty([rows, n])
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        ws2[tile_r, :] = x[tile_r, :] + 1
        out[tile_r, :] = (ws[tile_r, :] + ws2[tile_r, :].float()).to(x.dtype)
    return out


def test_a_workspace_only_the_kernel_reads_is_scratch() -> None:
    # ``out`` is returned, so it is not.
    assert _scratch(two_sweep_workspace, torch.empty(8, 64)) == {"ws"}
    assert _scratch(unread_workspace, torch.empty(8, 64)) == {"ws"}


def test_empty_like_and_new_empty_allocate_scratch() -> None:
    assert _scratch(workspace_like_and_new_empty, torch.empty(8, 64)) == {
        "ws",
        "ws2",
    }


def workspace_sized_from_metadata(x: torch.Tensor) -> torch.Tensor:
    ws = torch.empty([x.size(0), x.shape[1] * 1], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0)):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def test_a_workspace_sized_from_tensor_metadata_is_scratch() -> None:
    # ``size``, subscripts and arithmetic have no side effects for the
    # allocation's deletion to lose.
    assert _scratch(workspace_sized_from_metadata, torch.empty(8, 64)) == {"ws"}


def test_a_stage_of_a_larger_program_has_no_scratch() -> None:
    # Another stage of the program may read what this one allocates.
    bound = cpu_bind(
        helion.kernel(two_sweep_workspace, static_shapes=True), (torch.empty(8, 64),)
    )
    bound.host_function.partial_program = True
    assert not kernel_local_scratch(bound.host_function).names


def aliased_workspace(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    alias = ws
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        alias[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = alias[tile_r, :].to(x.dtype)
    return out


def viewed_workspace(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    flat = ws.view(-1)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out + flat.sum()


def attribute_of_workspace(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(ws.dtype).to(x.dtype)
    return out


def workspace_passed_to_a_call(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    torch.nn.init.zeros_(ws)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = ws[tile_r, :] + x[tile_r, :].float()
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_of_two_loops(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
    hl.barrier()
    for tile_r in hl.tile(rows):
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_read_by_host_code(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out + ws[0, 0]


def rebound_workspace(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def input_workspace(x: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_in_a_branch(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    if rows > 1:
        ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    else:
        ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def _view_as_itself(x: torch.Tensor) -> torch.Tensor:
    return x.view_as(x)


# Not a tensor: its ``new_empty`` returns a view of its argument.
_view_factory = types.ModuleType("_view_factory")
_view_factory.new_empty = _view_as_itself  # pyrefly: ignore [missing-attribute]


def workspace_from_a_non_tensor_new_empty(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = _view_factory.new_empty(x)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :] * 2
        out[tile_r, :] = ws[tile_r, :]
    return out


def workspace_with_a_call_in_its_arguments(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty(
        [rows, n], dtype=torch.promote_types(x.dtype, torch.float32), device=x.device
    )
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


# A module whose attribute reads may run its ``__getattr__``.
_sizes = types.ModuleType("_sizes")
_sizes.rows = 8  # pyrefly: ignore [missing-attribute]


def workspace_sized_by_a_module_attribute(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([_sizes.rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


# A module that computes its attributes: looking a factory up on it runs its
# ``__getattr__``.
_factories = types.ModuleType("_factories")
_factories.__getattr__ = lambda name: getattr(torch, name)  # pyrefly: ignore [missing-attribute]


def workspace_from_a_computed_module_attribute(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = _factories.empty_like(x)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :] * 2
        out[tile_r, :] = ws[tile_r, :]
    return out


class _IndexMeta(type):
    def __index__(cls) -> int:
        return 1


class _Index(metaclass=_IndexMeta):
    """A class whose use as an index runs its metaclass's ``__index__``."""


def workspace_sized_through_an_index_class(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, x.shape[_Index]], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_sized_through_a_slice_bound(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, *x.shape[_Index:]], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


class _ObservedList(list):
    """A list whose indexing and iteration run user code: type propagation
    shows it as a plain list when a global holds it."""

    def __getitem__(self, index: object) -> object:
        _OBSERVED.append(index)
        return super().__getitem__(index)

    def __iter__(self) -> Iterator[int]:
        _OBSERVED.append("iter")
        return super().__iter__()


_OBSERVED: list[object] = []
_SIZES = _ObservedList([8, 64])


def workspace_sized_by_a_global_list(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([_SIZES[0], n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_shaped_by_a_global_list(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty(_SIZES, dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_sized_by_a_starred_global_list(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([*_SIZES], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


class _EffectInt(int):
    """An ``int`` whose arithmetic runs user code: type propagation shows a
    global instance as a plain number."""

    def __mul__(self, other: object) -> int:
        _OBSERVED.append("mul")
        return int(self) * int(other)  # pyrefly: ignore [bad-argument-type]


_ROWS = _EffectInt(8)


def workspace_sized_by_a_global_int_subclass(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([_ROWS * 1, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_sized_by_a_name_of_a_global_int_subclass(
    x: torch.Tensor,
) -> torch.Tensor:
    rows, n = x.shape
    alias = _ROWS
    ws = torch.empty([alias, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_sized_by_a_number_computed_from_a_global_int_subclass(
    x: torch.Tensor,
) -> torch.Tensor:
    rows, n = x.shape
    m = _ROWS * 1
    ws = torch.empty([m, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_sized_by_a_list_argument(
    x: torch.Tensor, sizes: list[int]
) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([sizes[0], *sizes[1:]], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


def workspace_with_out_argument(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device, out=None)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = ws[tile_r, :].to(x.dtype)
    return out


@pytest.mark.parametrize(
    "fn",
    [
        returned_unread_workspace,
        aliased_workspace,
        viewed_workspace,
        attribute_of_workspace,
        workspace_passed_to_a_call,
        workspace_of_two_loops,
        workspace_read_by_host_code,
        rebound_workspace,
        workspace_in_a_branch,
        workspace_with_out_argument,
        workspace_from_a_non_tensor_new_empty,
        workspace_with_a_call_in_its_arguments,
        workspace_sized_by_a_module_attribute,
        workspace_from_a_computed_module_attribute,
        workspace_sized_through_an_index_class,
        workspace_sized_through_a_slice_bound,
        workspace_sized_by_a_global_list,
        workspace_shaped_by_a_global_list,
        workspace_sized_by_a_starred_global_list,
        workspace_sized_by_a_global_int_subclass,
        workspace_sized_by_a_name_of_a_global_int_subclass,
        workspace_sized_by_a_number_computed_from_a_global_int_subclass,
    ],
    ids=lambda fn: fn.__name__,
)
def test_a_workspace_the_analysis_cannot_confine_is_not_scratch(fn: object) -> None:
    # Each kernel lets ``ws`` escape, or reach it in a way the analysis does
    # not follow: returned, aliased, viewed, read through an attribute,
    # passed to a call, used by two loops or by host code, bound twice,
    # allocated under host control flow, with ``out=``, by a ``new_empty``
    # that is not a tensor's (here a view of the input), by a factory looked
    # up on a module that computes its attributes, or with a call, a
    # module's attribute, an index or slice bound whose ``__index__`` is
    # user code, a list a global holds (a subclass whose indexing and
    # iteration are user code, though type propagation shows a plain list),
    # an ``int`` subclass a global holds (a plain number to type propagation)
    # or a number computed from one in its arguments, whose effects deleting
    # the allocation would drop.  (A grid loop under host control flow does
    # not compile: ``NestedGridLoop``.)
    if _get_backend() == "pallas" and fn in (
        workspace_shaped_by_a_global_list,
        workspace_sized_by_a_starred_global_list,
    ):
        # Not this analysis: Pallas does not bind a workspace whose literal
        # width meets the input's size (ShapeSpecializingAllocation).
        pytest.skip("Pallas refuses a workspace of a literal width")
    assert "ws" not in _scratch(fn, torch.empty(8, 64))


def test_a_workspace_sized_by_a_list_argument_is_scratch() -> None:
    # Bind refuses a list subclass as an argument, so indexing and
    # unpacking a list argument run no user code.
    assert _scratch(
        workspace_sized_by_a_list_argument, torch.empty(8, 64), [8, 64]
    ) == {"ws"}


class _AliasingTensor(torch.Tensor):
    """A tensor subclass whose ``new_empty`` returns a view of itself: type
    propagation shows its instance as a plain fake tensor."""

    def new_empty(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.view_as(self)


def test_a_tensor_subclass_argument_has_no_scratch() -> None:
    # Its factories and metadata run its ``__torch_function__``: here
    # ``new_empty`` returns a view of the input, whose stores the caller sees.
    x = torch.empty(8, 64).as_subclass(_AliasingTensor)
    assert not _scratch(workspace_like_and_new_empty, x)


def test_a_name_sharing_an_inputs_storage_is_not_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backstop behind the factory check: whatever counted as a factory,
    # a fake storage another host tensor shares is not a fresh allocation.
    monkeypatch.setattr(scratch_tensors, "_is_factory", lambda value, provenance: True)
    assert "ws" not in _scratch(
        workspace_from_a_non_tensor_new_empty, torch.empty(8, 64)
    )


class _DropStores(ast.NodeTransformer):
    """Deletes the Triton stores into one tensor, as a backend's store
    elimination would."""

    def __init__(self, name: str) -> None:
        self.name = name

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        call = node.value
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "tl.store"
            and any(
                isinstance(part, ast.Name) and part.id == self.name
                for part in ast.walk(call.args[0])
            )
        ):
            return None
        return node


def _triton_code_after_store_elimination(
    monkeypatch: pytest.MonkeyPatch,
    fn: types.FunctionType,
    *,
    opted_in: bool,
    also_dropped: frozenset[str] = frozenset(),
    static_shapes: bool = True,
) -> tuple[ast.FunctionDef, ast.FunctionDef, str]:
    """``fn``'s Triton kernel and host wrapper, with a stand-in for a
    backend's store elimination deleting the stores into ``ws``, Triton
    opted in to the parameter removal when ``opted_in``, and the host names
    ``also_dropped`` reported to host DCE with those of the parameters that
    went."""
    drop_unused = generate_ast.drop_unused_scratch_arguments

    def after_store_elimination(
        host: object,
        device_function: object,
        kernel_def: list[ast.stmt],
        host_statements: object,
    ) -> set[str]:
        for stmt in kernel_def:
            _DropStores("ws").visit(stmt)
        dropped = drop_unused(host, device_function, kernel_def, host_statements)
        return dropped | also_dropped

    monkeypatch.setattr(
        generate_ast, "drop_unused_scratch_arguments", after_store_elimination
    )
    monkeypatch.setattr(
        TritonBackend,
        "drops_unread_scratch_arguments",
        property(lambda self: opted_in),
    )
    bound = cpu_bind(
        helion.kernel(fn, backend="triton", static_shapes=static_shapes),
        (torch.empty(8, 64),),
    )
    code = bound.to_triton_code(bound.config_spec.default_config())
    functions = {
        node.name: node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
    }
    return functions[f"_helion_{fn.__name__}"], functions[fn.__name__], code


def _launched_and_assigned(host: ast.FunctionDef) -> tuple[set[str], set[str]]:
    (launch,) = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_launcher"
    ]
    launched = {arg.id for arg in launch.args if isinstance(arg, ast.Name)}
    assigned = {
        target.id
        for node in ast.walk(host)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return launched, assigned


def unread_workspace_without_a_device(x: torch.Tensor) -> torch.Tensor:
    # Code generation adds ``device=x.device`` to the allocation.
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].float() * 2
        ws[tile_r, :] = v
        out[tile_r, :] = (v + 1).to(x.dtype)
    return out


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    "fn",
    [unread_workspace, unread_workspace_without_a_device],
    ids=["device", "no_device"],
)
def test_an_unread_workspace_loses_its_parameter_and_allocation(
    monkeypatch: pytest.MonkeyPatch, fn: types.FunctionType
) -> None:
    # The workspace's parameter, launcher argument and allocation go; the
    # output's stay.
    kernel, host, code = _triton_code_after_store_elimination(
        monkeypatch, fn, opted_in=True
    )
    assert [arg.arg for arg in kernel.args.args][:2] == ["x", "out"], code
    assert "ws" not in [arg.arg for arg in kernel.args.args], code
    launched, assigned = _launched_and_assigned(host)
    assert "out" in launched and "ws" not in launched, code
    assert "out" in assigned and "ws" not in assigned, code


@skipUnlessBackends(["triton"])
def test_dynamic_shapes_keep_the_allocation_of_an_unread_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With dynamic shapes only the tensor's own parameter goes: its stride
    # parameter stays, and so do the launch's read of the workspace and the
    # allocation.
    kernel, host, code = _triton_code_after_store_elimination(
        monkeypatch, unread_workspace, opted_in=True, static_shapes=False
    )
    params = [arg.arg for arg in kernel.args.args]
    assert "ws" not in params, code
    assert any(param.startswith("ws_stride") for param in params), code
    _launched, assigned = _launched_and_assigned(host)
    assert "ws" in assigned, code


def unread_workspace_beside_a_second_root(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    doubled = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].float() * 2
        ws[tile_r, :] = v
        out[tile_r, :] = (v + 1).to(x.dtype)
    for tile_r in hl.tile(rows):
        doubled[tile_r, :] = x[tile_r, :] * 2
    return out, doubled


@skipUnlessBackends(["triton"])
def test_a_kernel_of_two_roots_keeps_an_unread_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The removal handles a single-root kernel only.
    kernel, host, code = _triton_code_after_store_elimination(
        monkeypatch, unread_workspace_beside_a_second_root, opted_in=True
    )
    assert "ws" in [arg.arg for arg in kernel.args.args], code
    launched, assigned = _launched_and_assigned(host)
    assert "ws" in launched and "ws" in assigned, code


def unread_workspace_beside_an_update(x: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    flag = x.add_(1)  # noqa: F841
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].float() * 2
        ws[tile_r, :] = v
        out[tile_r, :] = (v + 1).to(x.dtype)
    return out


@skipUnlessBackends(["triton"])
def test_host_dce_drops_only_the_allocation_of_a_dropped_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Of the names reported with dropped parameters, host DCE deletes only
    # allocations: the unread in-place update ``flag`` names stays.
    _kernel, host, code = _triton_code_after_store_elimination(
        monkeypatch,
        unread_workspace_beside_an_update,
        opted_in=True,
        also_dropped=frozenset({"flag"}),
    )
    launched, assigned = _launched_and_assigned(host)
    assert "ws" not in launched and "ws" not in assigned, code
    assert "flag" in assigned, code


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize(
    ("fn", "opted_in"),
    [
        # The caller reads the returned workspace.
        (returned_unread_workspace, True),
        # The backend keeps its launch's positions.
        (unread_workspace, False),
    ],
    ids=["returned", "not_opted_in"],
)
def test_an_unread_workspace_that_must_stay_keeps_its_parameter(
    monkeypatch: pytest.MonkeyPatch, fn: types.FunctionType, opted_in: bool
) -> None:
    kernel, host, code = _triton_code_after_store_elimination(
        monkeypatch, fn, opted_in=opted_in
    )
    assert "ws" in [arg.arg for arg in kernel.args.args], code
    launched, assigned = _launched_and_assigned(host)
    assert "ws" in launched and "ws" in assigned, code


def test_an_input_is_not_scratch() -> None:
    assert _scratch(input_workspace, torch.empty(8, 64), torch.empty(8, 64)) == set()
