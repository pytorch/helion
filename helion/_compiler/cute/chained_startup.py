"""Typed, full-tile first-operand TMA scheduling for resident contractions.

The source is a compact physical matrix, not a kernel-name classification.
Only a bijective same-dtype leaf may occupy the operand arena before its
pointwise producer. Every thread reads its own old cell before overwriting it.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy

from . import chained_matmul as chain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


@dataclasses.dataclass(frozen=True)
class StartupInput:
    role: str
    operand: Node
    leaf: Node
    coordinates: tuple[str, ...]
    shape: tuple[int, int]
    inner: int
    row: str
    col: str
    wrapper: dict[str, object]


def has_startup_leaf(graphs: Sequence[GraphInfo]) -> bool:
    """Typed seed prefilter; concrete geometry is still proved at emission."""
    root = chain._root_graph(graphs)
    if root is None:
        return False
    first = next((node for node in root.graph.nodes if node.target is chain.dot), None)
    if first is None:
        return False
    dtype = cast("Node", first.args[0]).meta["val"].dtype
    for operand in first.args[:2]:
        for node in chain._ancestors(cast("Node", operand)):
            if node.target is not chain.memory_ops.load or node.meta["val"].ndim != 2:
                continue
            source = cast("Node", node.args[0])
            if source.target is not chain._tracing_ops._host_tensor:
                continue
            fake = source.meta["val"]
            if fake.dtype != dtype:
                continue
            compact = 1
            for size, stride in sorted(
                zip(fake.shape, fake.stride(), strict=True), key=operator.itemgetter(1)
            ):
                if (
                    type(size) is not int
                    or type(stride) is not int
                    or size <= 0
                    or stride <= 0
                ):
                    break
                if size != 1:
                    if stride != compact:
                        break
                    compact *= size
            else:
                return True
    return False


def _interval(value: sympy.Expr, ranges: dict[sympy.Symbol, int]) -> tuple[int, int]:
    """Bound an affine expression over canonical nonnegative tile origins."""
    base = value.subs(dict.fromkeys(ranges, 0))
    if not isinstance(base, sympy.Integer):
        raise chain._UnsupportedChain("startup transfer needs constant affine bases")
    low = high = int(base)
    remainder = sympy.Add(value, -base)
    for symbol, maximum in ranges.items():
        slope = sympy.diff(value, symbol)
        if not isinstance(slope, sympy.Integer):
            raise chain._UnsupportedChain("startup transfer needs affine indices")
        remainder -= slope * symbol
        low += min(0, int(slope) * maximum)
        high += max(0, int(slope) * maximum)
    if sympy.expand(remainder) != 0:
        raise chain._UnsupportedChain("startup transfer has an unknown index")
    return low, high


def _uniform_quotients(
    values: list[sympy.Expr], ranges: dict[sympy.Symbol, int]
) -> dict[sympy.Expr, sympy.Symbol]:
    """Lift bounded CTA-uniform origin quotients into conservative free axes.

    Matrix coordinates are deliberately absent from ``ranges``. Treating a
    quotient as independent of its origin over-approximates bounds; requiring
    alignment for every integer quotient also covers every actual tile origin.
    No modulo, nested quotient, computed numerator or varying divisor is proved.
    """
    result: dict[sympy.Expr, sympy.Symbol] = {}
    for value in values:
        for quotient in sorted(value.atoms(sympy.floor), key=str):
            numerator, denominator = quotient.args[0].as_numer_denom()
            if (
                numerator not in ranges
                or not isinstance(denominator, sympy.Integer)
                or denominator <= 0
            ):
                raise chain._UnsupportedChain(
                    "startup quotient needs an outer origin and positive constant divisor"
                )
            if quotient not in result:
                result[quotient] = sympy.Dummy("startup_quotient", integer=True)
    return result


def _leaf(
    cg: GenerateAST,
    plan: chain.ChainedMatmulPlan,
    expression: chain._Expression,
    operand: Node,
    role: str,
    shape: tuple[int, int],
    inner: int,
    loaded: tuple[Node, tuple[str, ...], list[str], str],
) -> StartupInput | None:
    leaf, coordinates, indices, _ = loaded
    source = cast("Node", leaf.args[0])
    fake = source.meta["val"]
    if fake.dtype != plan.dtype:
        return None
    sizes, strides = tuple(fake.shape), tuple(fake.stride())
    if any(type(v) is not int or v <= 0 for v in (*sizes, *strides)):
        return None
    # Dense permutations are supported. Holes, overlapping and broadcast
    # storage are not a descriptor contract; legacy keeps their masked path.
    compact = 1
    for size, stride in sorted(
        zip(sizes, strides, strict=True), key=operator.itemgetter(1)
    ):
        if size != 1:
            if stride != compact:
                return None
            compact *= size
    names = ("chain_start_row", "chain_start_col")
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*names, *expression.origins.values())
    }
    row, col = (symbols[name] for name in names)
    ranges = {
        symbols[expression.origins[axis]]: extent - block
        for axis, extent, block in plan.axes
    }
    steps = {symbols[expression.origins[axis]]: block for axis, _, block in plan.axes}
    height, width = shape[1 - inner], shape[inner]
    if width % 8 or width > 256 or height > 256:
        return None
    values = [
        sympy.expand(chain._copy_index(index, expression.definitions, symbols))
        for index in indices
    ]
    quotients = _uniform_quotients(values, ranges)
    for quotient, symbol in quotients.items():
        numerator, denominator = cast("sympy.Expr", quotient.args[0]).as_numer_denom()
        ranges[symbol] = ranges[cast("sympy.Symbol", numerator)] // int(denominator)
        # A quotient that is identically zero cannot change tile alignment.
        steps[symbol] = int(ranges[symbol] != 0)
    bases, coefficients = [], []
    for value, extent in zip(values, sizes, strict=True):
        value = value.xreplace(quotients)
        base = value.subs({row: 0, col: 0})
        slopes = sympy.diff(value, row), sympy.diff(value, col)
        if any(not isinstance(s, sympy.Integer) for s in slopes):
            return None
        pair = int(slopes[0]), int(slopes[1])
        if (
            sympy.expand(
                sympy.Add(
                    value, -base, sympy.Mul(-pair[0], row), sympy.Mul(-pair[1], col)
                )
            )
            != 0
        ):
            return None
        low, high = _interval(value, {**ranges, row: height - 1, col: width - 1})
        if low < 0 or high >= extent:
            return None
        bases.append(base)
        coefficients.append(pair)
    physical = tuple(
        sum(
            int(pair[axis]) * stride
            for pair, stride in zip(coefficients, strides, strict=True)
        )
        for axis in (0, 1)
    )
    pitch = physical[0]
    if physical[1] != 1 or pitch < width or pitch % 8 or compact % pitch:
        return None
    if max(pitch, compact // pitch) >= 2**31:
        return None
    base = sympy.expand(
        sum(value * stride for value, stride in zip(bases, strides, strict=True))
    )
    constant = base.subs(dict.fromkeys(ranges, 0))
    if not isinstance(constant, sympy.Integer) or constant < 0:
        return None
    row_base, col_base = (
        sympy.Integer(int(constant) // pitch),
        sympy.Integer(int(constant) % pitch),
    )
    remainder = base - constant
    for symbol in ranges:
        slope = sympy.diff(base, symbol)
        if not isinstance(slope, sympy.Integer) or slope < 0:
            return None
        row_base = sympy.Add(row_base, sympy.Mul(int(slope) // pitch, symbol))
        col_base = sympy.Add(col_base, sympy.Mul(int(slope) % pitch, symbol))
        remainder -= slope * symbol
    if sympy.expand(remainder) != 0:
        return None
    if _interval(row_base, ranges)[1] + height > compact // pitch:
        return None
    if _interval(col_base, ranges)[1] + width > pitch:
        return None
    for base_coordinate, tile in ((row_base, height), (col_base, width)):
        if int(
            cast("sympy.Integer", base_coordinate.subs(dict.fromkeys(ranges, 0)))
        ) % tile or any(
            int(sympy.diff(base_coordinate, symbol)) * step % tile
            for symbol, step in steps.items()
        ):
            return None
    # TensorMap tile origins along the contiguous dimension are 16B aligned.
    if int(constant) % 8 or any(
        int(sympy.diff(base, symbol)) * step % 8 for symbol, step in steps.items()
    ):
        return None
    source_name = cg.device_function.tensor_arg(
        fake, prefer_name=cast("str", source.args[0])
    ).name
    output = cast("Node", plan.store.args[0])
    output_name = cg.device_function.tensor_arg(
        output.meta["val"], prefer_name=cast("str", output.args[0])
    ).name
    tag = f"chain_start_{role}"
    return StartupInput(
        role,
        operand,
        leaf,
        coordinates,
        shape,
        inner,
        chain._copy_code(row_base.xreplace({v: k for k, v in quotients.items()})),
        chain._copy_code(col_base.xreplace({v: k for k, v in quotients.items()})),
        {
            "kind": "chained_startup_tma",
            "lhs_name": source_name,
            "out_name": output_name,
            "shape": sizes,
            "strides": strides,
            "rows": compact // pitch,
            "columns": pitch,
            "tile": (height, width),
            "operand_shape": shape,
            "inner": inner,
            "dtype": str(fake.dtype).removeprefix("torch."),
            "kernel_args": [f"{tag}_atom", f"{tag}_tensor"],
        },
    )


def plan_startup(
    cg: GenerateAST,
    plan: chain.ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    inner_axes: dict[tuple[int, str], int],
) -> list[StartupInput]:
    result = []
    m, n, k = plan.shapes[0]
    for role, shape, operand in zip(
        ("a", "b"), ((m, k), (n, k)), plan.dots[0].args[:2], strict=True
    ):
        operand = cast("Node", operand)
        inner = inner_axes[0, role]
        expression = chain._Expression(cg, plan, boundaries)
        expression.scan_inputs = scans
        names = ("chain_start_row", "chain_start_col")
        expression.coordinate_names.update(names)
        stored = names if inner else names[::-1]
        coords = stored if role == "a" else stored[::-1]
        expression.value(operand, coords)
        candidates = [
            candidate
            for loaded in expression.loaded_inputs
            if (
                candidate := _leaf(
                    cg, plan, expression, operand, role, shape, inner, loaded
                )
            )
            is not None
        ]
        # A raw leaf used at another coordinate could observe an overwritten
        # cell during in-place conversion. Only one occurrence is admitted.
        if candidates:
            candidate = candidates[0]
            if sum(item[0] is candidate.leaf for item in expression.loaded_inputs) == 1:
                result.append(candidate)
    if not result:
        raise chain._UnsupportedChain(
            "startup TMA has no full bijective same-dtype leaf"
        )
    for transfer in result:
        cg.cute_wrapper_plans.append(transfer.wrapper)
        cg.device_function.wrapper_only_params.extend(
            cast("list[str]", transfer.wrapper["kernel_args"])
        )
    return result


def issue_lines(transfers: list[StartupInput]) -> list[str]:
    lines = [
        "chain_start_bar = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=16)",
        "if chain_thread == 0:",
        "    cute.arch.mbarrier_init(chain_start_bar, 1)",
        "cute.arch.mbarrier_init_fence()",
        "cute.arch.sync_threads()",
        "if chain_thread == 0:",
        f"    cute.arch.mbarrier_arrive_and_expect_tx(chain_start_bar, {sum(math.prod(t.shape) * 2 for t in transfers)})",
    ]
    for transfer in transfers:
        tag, role = f"chain_start_{transfer.role}", transfer.role
        height, width = cast("tuple[int, int]", transfer.wrapper["tile"])
        target = f"chain_0_{role}"
        if not transfer.inner:
            target = f"cute.make_tensor({target}.iterator, cute.select({target}.layout, mode=[1, 0]))"
        lines.extend(
            [
                f"{tag}_global = cute.local_tile({tag}_tensor, ({height}, {width}), (None, None))",
                f"{tag}_source = {tag}_global[None, None, ({transfer.row}) // {height}, ({transfer.col}) // {width}]",
                f"{tag}_shared, {tag}_partition = cute.nvgpu.cpasync.tma_partition({tag}_atom, 0, cute.make_layout(1), cute.group_modes({target}, 0, 2), cute.group_modes({tag}_source, 0, 2))",
                "if chain_warp == 0:",
                # CopyBulkTensorTileG2SOp performs its own lane election. All
                # lanes of the selected warp must reach the collective copy.
                f"    cute.copy({tag}_atom, {tag}_partition, {tag}_shared, tma_bar_ptr=chain_start_bar)",
            ]
        )
    return lines


def finish_lines(
    cg: GenerateAST,
    plan: chain.ChainedMatmulPlan,
    transfer: StartupInput,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    dtype: str,
) -> list[str]:
    if chain._direct_operand(transfer.operand):
        return []
    role, shape, inner = transfer.role, transfer.shape, transfer.inner
    index = f"chain_start_{role}_index"
    row, col = f"{index} // {shape[inner]}", f"{index} % {shape[inner]}"
    names = ("chain_start_row", "chain_start_col")
    stored = names if inner else names[::-1]
    coords = stored if role == "a" else stored[::-1]
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(names)
    expression.memo[transfer.leaf, transfer.coordinates] = (
        f"chain_0_{role}[{stored[0]}, {stored[1]}]"
    )
    value = expression.value(transfer.operand, coords)
    domain = chain._operand_domain(cg, transfer.operand, coords, plan)
    return [
        f"for chain_start_{role}_step in cutlass.range({math.prod(shape) // 128}, unroll=1):",
        f"    {index} = chain_thread + chain_start_{role}_step * 128",
        f"    {names[0]} = {row}",
        f"    {names[1]} = {col}",
        chain._indent(expression.lines),
        f"    chain_0_{role}[{stored[0]}, {stored[1]}] = {chain._masked_operand(value, dtype, domain)}",
    ]
