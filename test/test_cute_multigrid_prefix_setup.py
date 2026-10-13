"""Root metadata must survive even when shared launch geometry needs no lane loop."""

from __future__ import annotations

import ast
import builtins
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _config as mixed_config
from .test_cute_grid_launch_extents import _mixed_rank_copy
from .test_cute_multigrid_config import _bound
from .test_cute_multigrid_config import _roundtrip
from .test_cute_multigrid_surplus import _config as prefix_config
from .test_cute_multigrid_surplus import _prefix_copy
from .test_cute_multigrid_surplus import _prefix_copy_reverse
import helion
from helion._utils import next_power_of_2


def _assert_defined(code: str) -> None:
    """Check lexical dominance for the generated pointwise fixture kernels."""
    module = ast.parse(code)
    globals_ = set(dir(builtins))
    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            globals_.add(statement.name)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            globals_.update(
                alias.asname or alias.name.split(".")[0] for alias in statement.names
            )
        elif isinstance(statement, ast.Assign):
            globals_.update(
                node.id
                for target in statement.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
            )

    def expression(node: ast.AST | None, defined: set[str]) -> None:
        if node is not None:
            missing = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            } - defined
            assert not missing, (sorted(missing), ast.unparse(node))

    def body(statements: list[ast.stmt], incoming: set[str]) -> set[str]:
        defined = set(incoming)
        for statement in statements:
            if isinstance(statement, ast.Assign):
                expression(statement.value, defined)
                for target in statement.targets:
                    if not isinstance(target, ast.Name):
                        expression(target, defined)
                    defined.update(
                        node.id
                        for node in ast.walk(target)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                    )
            elif isinstance(statement, ast.AugAssign):
                expression(statement.value, defined)
                if isinstance(statement.target, ast.Name):
                    assert statement.target.id in defined
                else:
                    expression(statement.target, defined)
            elif isinstance(statement, ast.If):
                expression(statement.test, defined)
                defined = body(statement.body, defined) & body(
                    statement.orelse, defined
                )
            elif isinstance(statement, ast.For):
                expression(statement.iter, defined)
                inside = defined | {
                    node.id
                    for node in ast.walk(statement.target)
                    if isinstance(node, ast.Name)
                }
                body(statement.body, inside)
                body(statement.orelse, defined)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                defined.update(
                    alias.asname or alias.name.split(".")[0]
                    for alias in statement.names
                )
            else:
                expression(statement, defined)
        return defined

    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    assert kernels
    for kernel in kernels:
        body(kernel.body, globals_ | {arg.arg for arg in kernel.args.args})


@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_mixed_grid_definitions_dominate_second_root(
    vector: int, roundtrip: bool
) -> None:
    args = (torch.empty((2, 4096)), torch.empty(1024), torch.empty(1024))
    config = mixed_config("mixed", vector=vector)
    if roundtrip:
        config = helion.Config(
            block_sizes=[1024, 256],
            num_threads=[256, 128],
            cute_vector_widths=[1, vector, 1],
            cute_lane_layouts=["blocked", "strided", "blocked"],
        )
        with _bound(_mixed_rank_copy, args) as bound:
            code = bound.to_code(_roundtrip(bound, config))
    else:
        code = _code(_mixed_rank_copy, args, config)
    _assert_defined(code)


@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_prefix_surplus_definitions_dominate_each_root(
    vector: int, reverse: bool, tail: bool
) -> None:
    columns, length = (1027, 2051) if tail else (1024, 2048)
    args = (
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty(length, dtype=torch.bfloat16),
        torch.empty((3, columns), dtype=torch.bfloat16),
        torch.empty(length, dtype=torch.bfloat16),
    )
    kernel = _prefix_copy_reverse if reverse else _prefix_copy
    code = _code(
        kernel,
        args,
        prefix_config(
            reverse=reverse,
            vector=vector,
            layout="strided",
            first_threads=0,
            second_threads=512,
        ),
    )
    _assert_defined(code)


class _Pointer:
    def __init__(self, storage: torch.Tensor, offset: int):
        self.storage = storage
        self.offset = offset

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.storage, self.offset + offset)

    def load(self) -> float:
        assert 0 <= self.offset < self.storage.numel()
        return self.storage[self.offset].item()

    def store(self, value: float) -> None:
        assert 0 <= self.offset < self.storage.numel()
        self.storage[self.offset] = value


def _scalar_program(code: str) -> Any:
    """Execute the actual scalar emitted statements on CPU, with physical storage."""
    position = [0, 0, 0, 0]
    arch = SimpleNamespace(
        block_idx=lambda: (position[0], 0, 0),
        thread_idx=lambda: tuple(position[1:]),
    )
    cutlass = SimpleNamespace(
        Int32=int, Float32=float, BFloat16=float, range_constexpr=range
    )

    def launch(kernel, grid, *args, block):
        assert grid[1:] in ((), (1,), (1, 1))
        assert len(block) == 3 and all(size > 0 for size in block)
        wrapped = []
        for tensor in args:
            storage = tensor.as_strided(
                (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0
            )
            wrapped.append(
                SimpleNamespace(
                    iterator=_Pointer(storage, tensor.storage_offset()),
                    layout=SimpleNamespace(stride=tensor.stride()),
                )
            )
        for pid in range(grid[0]):
            for z in range(block[2]):
                for y in range(block[1]):
                    for x in range(block[0]):
                        position[:] = pid, x, y, z
                        kernel(*wrapped)

    module = ast.parse(code)
    module.body = [
        node
        for node in module.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    namespace = {
        "torch": torch,
        "cutlass": cutlass,
        "cute": SimpleNamespace(arch=arch),
        "_next_power_of_2": next_power_of_2,
        "_default_cute_launcher": launch,
    }
    exec(compile(module, "<scalar multigrid CPU evidence>", "exec"), namespace)
    return namespace[functions[-1].name]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_scalar_emitted_prefix_preserves_raw_guards(reverse: bool, tail: bool) -> None:
    columns, length = (1027, 2051) if tail else (1024, 2048)
    x = torch.arange(3 * columns).reshape(3, columns).remainder(17).to(torch.bfloat16)
    y = torch.arange(length).remainder(13).to(torch.bfloat16)
    first_storage = torch.full((3, columns * 2 + 32), -8192, dtype=x.dtype)
    first = first_storage[:, 16 : columns + 16]
    second_storage = torch.full((length + 32,), -8192, dtype=y.dtype)
    second = second_storage[16:-16]
    args = (x, y, first, second)
    kernel = _prefix_copy_reverse if reverse else _prefix_copy
    code = _code(
        kernel,
        args,
        prefix_config(
            reverse=reverse,
            vector=1,
            layout="strided",
            first_threads=0,
            second_threads=512,
        ),
    )
    _assert_defined(code)
    run = _scalar_program(code)
    saved = x.clone(), y.clone()
    for _ in range(2):
        first.fill_(float("nan"))
        second.fill_(float("nan"))
        run(*args)
        torch.testing.assert_close(
            first, x + torch.arange(3)[:, None].to(x.dtype), atol=0, rtol=0
        )
        torch.testing.assert_close(second, y, atol=0, rtol=0)
        torch.testing.assert_close((x, y), saved, atol=0, rtol=0)
        assert torch.all(first_storage[:, :16] == -8192)
        assert torch.all(first_storage[:, columns + 16 :] == -8192)
        assert torch.all(second_storage[:16] == -8192)
        assert torch.all(second_storage[-16:] == -8192)
