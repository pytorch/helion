"""CPU admission and prefetch proof for typed scalar index recipes."""

from __future__ import annotations

import ast
import operator
import re
from types import SimpleNamespace

import pytest
import torch

import helion
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_args
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_config
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_matmul
from helion._testing.cute_mma_dag_scalar_indices import wrap_scalar_index

pytestmark = pytest.mark.cpu_only


def _index_program_and_addresses(
    mode, dtype, variant, comparison="gt", scalar_rhs=False, literal=1
):
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            scalar_index_args(
                dtype,
                variant,
                comparison=comparison,
                scalar_rhs=scalar_rhs,
                literal=literal,
            ),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None and program.index_values
        source = bound.to_code(scalar_index_config(mode, 3))
    table_base = re.search(r"(md_base(?:_\d+)?) = order\.iterator\.toint", source)
    assert table_base is not None
    addresses = [
        call.args[0]
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("ldg_s32", "ldg_s64")
        and call.args
        and ast.unparse(call.args[0]).startswith(f"{table_base[1]} +")
    ]
    return program, source, table_base[1], addresses


def _address_environment(source, table_base, values, dtype):
    word_bytes = torch.iinfo(dtype).bits // 8
    environment = {
        "operator": operator,
        "cutlass": SimpleNamespace(
            Boolean=bool,
            Int32=lambda value: wrap_scalar_index(value, 32),
            Int64=lambda value: wrap_scalar_index(value, 64),
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                min=lambda left, right, **kwargs: min(left, right),
                max=lambda left, right, **kwargs: max(left, right),
            )
        ),
        "_helion_mma_dag": SimpleNamespace(
            ldg_s32=lambda address: values[address // word_bytes],
            ldg_s64=lambda address: values[address // word_bytes],
        ),
        "_cute_python_mod": operator.mod,
        "offsets": SimpleNamespace(iterator=SimpleNamespace(toint=lambda: 0)),
        table_base: 0,
    }
    for statement in ast.walk(ast.parse(source)):
        if isinstance(statement, ast.Assign) and ast.unparse(
            statement.value
        ).startswith("offsets.iterator.toint()"):
            exec(
                compile(ast.Module([statement], []), "<index bases>", "exec"),
                environment,
            )
    return environment


@pytest.mark.parametrize(
    "mode,dtype,variant",
    [
        ("warp", torch.int32, "reverse"),
        ("tcgen05", torch.int64, "reverse"),
        ("warp", torch.int64, "wrap"),
        ("tcgen05", torch.int32, "wrap"),
        ("warp", torch.int32, "nested"),
        ("tcgen05", torch.int64, "nested"),
        ("warp", torch.int32, "floor"),
        ("tcgen05", torch.int64, "floor"),
        ("tcgen05", torch.int64, "clamped_narrow"),
    ],
)
def test_typed_scalar_indices_render_at_prefetch_iteration(mode, dtype, variant):
    # Eight cases pair both owners and widths across four index recipes; the
    # ninth checks admitted clamp and integer-narrowing operators together.
    program, source, table_base, addresses = _index_program_and_addresses(
        mode, dtype, variant
    )
    assert any(
        access.name == "order" and access.loops == frozenset({0})
        for access in program.index_loads.values()
    )
    assert "_helion_mma_dag" in source
    # Every dynamic scalar index must be substituted in the future refill,
    # not cached at the current iteration before the guarded transfer.
    table_addresses = [ast.unparse(address) for address in addresses]
    assert any("md_i + 2" in address for address in table_addresses)
    assert any(
        "md_i" in address and "md_i + 2" not in address for address in table_addresses
    )
    # The table/offset loads occur inside the guarded ring prologue, not
    # when defining its address bases for a possibly empty loop.
    prologue, guard, _body = source.partition("if md_scalar < md_scalar_1:")
    assert guard
    assert f"ldg_s32({table_base}" not in prologue
    assert f"ldg_s64({table_base}" not in prologue
    assert "_mma_dag_index_value_" not in source


@pytest.mark.parametrize("variant", ["float", "power", "dynamic_div"])
def test_unsupported_index_recipe_is_not_admitted(variant):
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            scalar_index_args(torch.int32, variant),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert match_mma_dag(bound.host_function.device_ir.graphs) is None


@pytest.mark.parametrize("divisor", [2**31, 2**32 - 1, 2**32])
def test_oversized_scalar_divisor_is_not_admitted(divisor):
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            scalar_index_args(torch.int32, "floor", divisor=divisor),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert match_mma_dag(bound.host_function.device_ir.graphs) is None


@pytest.mark.parametrize("comparison", ["eq", "ne", "lt", "le", "gt", "ge"])
@pytest.mark.parametrize("scalar_rhs", [False, True])
def test_predicate_is_evaluated_at_each_current_and_prefetched_address(
    comparison, scalar_rhs
):
    # Cover each comparison's Tensor/Scalar overload once; pair integer widths
    # and physical owners with the overload rather than adding two more axes.
    mode, dtype = ("warp", torch.int32) if scalar_rhs else ("tcgen05", torch.int64)
    # Boundary literals exercise wrapping before comparisons, whose Boolean
    # result cannot recover an incorrectly widened integer operand afterward.
    literal = (
        {
            "eq": 2**32,
            "ne": -(2**32),
            "lt": 2**32 + 1,
            "le": 2**32 + 1,
            "gt": -(2**32) + 2,
            "ge": -(2**32) + 2,
        }[comparison]
        if scalar_rhs
        else 1
    )
    program, source, table_base, addresses = _index_program_and_addresses(
        mode, dtype, "compare", comparison, scalar_rhs, literal
    )
    assert any(n.meta["val"].dtype == torch.bool for n in program.index_values.values())
    prologue, guard, _body = source.partition("if md_scalar < md_scalar_1:")
    assert guard
    offset_base = re.search(r"(md_base(?:_\d+)?) = offsets\.iterator\.toint", source)
    assert offset_base is not None
    for base in (table_base, offset_base[1]):
        assert f"ldg_s32({base}" not in prologue
        assert f"ldg_s64({base}" not in prologue
    assert addresses and any("md_i + 2" in ast.unparse(a) for a in addresses)
    values = [0, 0]
    environment = _address_environment(source, table_base, values, dtype)
    bits = torch.iinfo(dtype).bits
    limit = (1 << (bits - 1)) - 1
    order_size = next(
        access.fake.numel()
        for access in program.index_loads.values()
        if access.name == "order"
    )
    saw_prefetch_truth_change = False
    for offset, rhs, step in [(-1, 1, 1), (2, 2, 0), (limit, 1, 1), (-limit - 1, 0, 0)]:
        values[:] = [offset, rhs]
        tensor = torch.tensor(values, dtype=dtype)
        current_predicate = getattr(operator, comparison)(
            tensor[0] + step, literal if scalar_rhs else tensor[1]
        )
        for address in addresses:
            future = re.search(r"\bmd_(?:i|scalar) \+ (\d+)", ast.unparse(address))
            effective_step = step + (int(future[1]) if future is not None else 0)
            environment["md_i"] = environment["md_scalar"] = step
            actual = eval(
                compile(
                    ast.Expression(address), "<generated predicate address>", "eval"
                ),
                environment,
            )
            predicate = getattr(operator, comparison)(
                tensor[0] + effective_step, literal if scalar_rhs else tensor[1]
            )
            saw_prefetch_truth_change |= effective_step != step and bool(
                predicate
            ) != bool(current_predicate)
            expected = (effective_step + predicate).remainder(order_size)
            assert actual == expected.item() * (bits // 8)
    assert saw_prefetch_truth_change


@pytest.mark.parametrize(
    "dtype,variant,scalar_rhs",
    [
        (torch.float32, "compare", False),
        (torch.int32, "compare_float_scalar", True),
    ],
)
def test_comparison_address_requires_integer_dependencies(dtype, variant, scalar_rhs):
    with cpu_target():
        args = list(scalar_index_args(torch.int32, variant, scalar_rhs=scalar_rhs))
        # Keep loop bounds and page IDs integral: only the comparison operand
        # is floating, so refusal cannot come from an unrelated index load.
        args[4] = args[4].to(dtype)
        bound = cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            tuple(args),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert match_mma_dag(bound.host_function.device_ir.graphs) is None


def test_bare_boolean_index_remains_unsupported_mask_indexing():
    # Bare Boolean subscripts are rejected before MMA matching. Supporting a
    # Boolean intermediate in an integer recipe must not change that contract.
    with (
        cpu_target(),
        pytest.raises(
            helion.exc.DataDependentOutputShapeNotSupported,
            match="Boolean mask indexing",
        ),
    ):
        cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            scalar_index_args(torch.int64, "bare_bool"),
        )


def test_bare_boolean_explicit_load_is_not_an_integer_mma_index():
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True),
            scalar_index_args(torch.int32, "bare_bool_load"),
        )
        with bound.env, bound.host_function:
            assert match_mma_dag(bound.host_function.device_ir.graphs) is None


@pytest.mark.parametrize(
    "mode,dtype,variant",
    [
        ("warp", torch.int32, "extrema"),
        ("tcgen05", torch.int64, "extrema"),
        ("warp", torch.int64, "extrema_narrow"),
    ],
)
def test_scalar_extrema_preserve_wrapping_before_current_and_future_indices(
    mode, dtype, variant
):
    program, source, table_base, addresses = _index_program_and_addresses(
        mode, dtype, variant
    )
    assert {
        torch.ops.aten.minimum.default,
        torch.ops.aten.maximum.default,
    } <= {node.target for node in program.index_values.values()}
    assert addresses
    assert any("md_i + 2" in ast.unparse(address) for address in addresses)
    assert any("md_i + 2" not in ast.unparse(address) for address in addresses)

    bits = torch.iinfo(dtype).bits
    word_bytes = bits // 8
    values = [0, 0]
    # Execute the emitted address expression with signed-width casts and memory
    # reads modeled on CPU. This tests the generated placement of every wrap,
    # including the future refill's substitution, without compiling a GPU kernel.
    environment = _address_environment(source, table_base, values, dtype)
    limit = (1 << (bits - 1)) - 1
    # MAX-1 + 2 must wrap before max; MIN+1 - 2 must wrap before min.
    cases = [(limit - 1, -3, 2), (-limit, -3, 2), (-17, 4, 2)]
    if variant == "extrema_narrow":
        cases += [(2**31 - 2, -3, 2), (2**32 + 5, 7, 4)]
    order_size = next(
        access.fake.numel()
        for access in program.index_loads.values()
        if access.name == "order"
    )
    for offset, lower, step in cases:
        values[:] = [offset, lower]
        for address in addresses:
            # Include guarded prologue loads as well as current/future refill.
            text = ast.unparse(address)
            future = re.search(r"\bmd_(?:i|scalar) \+ (\d+)", text)
            effective_step = step + (int(future[1]) if future is not None else 0)
            environment["md_i"] = environment["md_scalar"] = step
            actual = eval(
                compile(ast.Expression(address), "<generated table address>", "eval"),
                environment,
            )
            tensor = torch.tensor(values, dtype=dtype)
            added = tensor[0] + effective_step
            if variant == "extrema_narrow":
                added = added.to(torch.int32)
            expected = torch.minimum(
                torch.maximum(added, tensor[1]), tensor[0] - effective_step
            ).remainder(order_size)
            assert actual == expected.item() * word_bytes
