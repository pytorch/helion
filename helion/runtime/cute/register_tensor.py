"""Stage portable tensor operations over statically owned CuTe registers."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import TYPE_CHECKING
from typing import cast

import cutlass
from cutlass import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects import vector
from cutlass._mlir.extras import types as T
import cutlass.cute as cute
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.register_tensor import RegisterTensorMap
    from helion._compiler.cute.register_tensor import RegisterTensorNode
    from helion._compiler.cute.register_tensor import RegisterTensorPlan
DTYPES = {
    "int32": cutlass.Int32,
    "int64": cutlass.Int64,
    "float32": cutlass.Float32,
    "bool": cutlass.Boolean,
}
NUMPY_DTYPES = {
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "bool": np.bool_,
}


def _parse(source: str) -> RegisterTensorPlan:
    plan = cast("RegisterTensorPlan", json.loads(source))
    assert plan["version"] == 1
    return plan


def _permute(
    left: cute.TensorSSA,
    right: cute.TensorSSA,
    indices: Sequence[int],
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:
    return cute.TensorSSA(
        vector.shuffle(left.ir_value(), right.ir_value(), indices, loc=loc, ip=ip),
        (len(indices),),
        left.dtype,
    )


def _predicate(
    members: set[int],
    groups: int,
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cutlass.Boolean:
    for bit in range(groups.bit_length() - 1):
        mask = 1 << bit
        for polarity in (False, True):
            if members == {
                lane for lane in range(groups) if bool(lane & mask) == polarity
            }:
                return group & mask != 0 if polarity else group & mask == 0
    if len(members) == 1:
        return group == next(iter(members))
    mask = sum(1 << lane for lane in members)
    return cutlass.Uint32(mask) >> group & cutlass.Uint32(1) != 0


def _select_group(
    values: Sequence[cute.TensorSSA],
    owners: Sequence[int],
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:

    def select(indices: list[int]) -> cute.TensorSSA:
        if len(indices) == 1:
            return values[indices[0]]
        middle = len(indices) // 2
        left_indices = indices[:middle]
        left = select(left_indices)
        right = select(indices[middle:])
        members = {lane for lane, owner in enumerate(owners) if owner in left_indices}
        condition = _predicate(members, len(owners), group)
        return cute.where(
            cute.full(left.shape, condition, cutlass.Boolean), left, right
        )

    return select(list(range(len(values))))


def _permute_by_group(
    left: cute.TensorSSA,
    right: cute.TensorSSA,
    mapping: RegisterTensorMap,
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:
    choices = [_permute(left, right, row) for row in mapping["rows"]]
    return _select_group(choices, mapping["owners"], group)


@dsl_user_op
def _cute_gather_registers(
    value: cute.Tensor,
    mapping: RegisterTensorMap,
    lane: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.Tensor:
    """Materialize a static local register gather with lane-dependent SSA selects."""
    source = value.load()
    zero = cute.full(source.shape, source.dtype(0), source.dtype)
    group = lane % cutlass.Int32(len(mapping["owners"]))
    routed = _permute_by_group(source, zero, mapping, group, loc=loc, ip=ip)
    result = cute.make_rmem_tensor(routed.shape, routed.dtype)
    result.store(routed)
    return result


def _lane_terms(
    owners: tuple[int, ...],
) -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    groups = len(owners)
    width = groups.bit_length() - 1
    basis = [
        (1 << groups) - 1,
        *[
            sum(1 << lane for lane in range(groups) if lane & 1 << bit)
            for bit in range(width)
        ],
    ]
    affine = []
    for coefficients in sorted(range(1 << len(basis)), key=int.bit_count):
        truth = 0
        for bit, mask in enumerate(basis):
            if coefficients & 1 << bit:
                truth ^= mask
        affine.append((coefficients, truth))
    constant = 0
    shifted = defaultdict(int)
    lookups = []
    for output_bit in range(width):
        truth = sum(
            (1 << lane for lane, owner in enumerate(owners) if owner & 1 << output_bit)
        )
        for coefficients, candidate in affine:
            if candidate == truth:
                if coefficients & 1:
                    constant |= 1 << output_bit
                for input_bit in range(width):
                    if coefficients & 1 << input_bit + 1:
                        shifted[output_bit - input_bit] |= 1 << input_bit
                break
        else:
            lookups.append((truth, output_bit))
    return (constant, tuple(sorted(shifted.items())), tuple(lookups))


def _lane_index(
    owners: tuple[int, ...],
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cutlass.Int32:
    constant, shifted, lookups = _lane_terms(owners)
    result = cutlass.Int32(constant)
    for shift, mask in shifted:
        value = cast(
            "cutlass.Int32", group if mask == len(owners) - 1 else group & mask
        )
        if shift:
            value = cast(
                "cutlass.Int32", value << shift if shift > 0 else value >> -shift
            )
        result = cast("cutlass.Int32", result ^ value)
    for mask, shift in lookups:
        lookup = cast(
            "cutlass.Uint32", cutlass.Uint32(mask) >> group & cutlass.Uint32(1)
        )
        result = cast("cutlass.Int32", result ^ cutlass.Int32(lookup << shift))
    return result


def _gather_groups(
    value: cute.TensorSSA,
    mapping: RegisterTensorMap,
    groups: int,
    registers: int,
    live_registers: Sequence[int],
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:
    matrix = [mapping["rows"][owner] for owner in mapping["owners"]]
    offset = matrix[0][0]
    butterfly = all(
        (row == [lane ^ offset] * registers for lane, row in enumerate(matrix))
    )
    scalars = []
    lane_maps = {}
    live = set(live_registers)
    for register in range(registers):
        if register not in live:
            scalar = value.dtype(0)
        elif all((row[register] == lane for lane, row in enumerate(matrix))):
            scalar = value[register]
        elif butterfly:
            scalar = cute.arch.shuffle_sync_bfly(value[register], offset=offset)
        else:
            owners = tuple(row[register] for row in matrix)
            if owners not in lane_maps:
                lane_maps[owners] = _lane_index(owners, group)
            source = lane_maps[owners]
            scalar = cute.arch.shuffle_sync(
                value[register], offset=source, mask_and_clamp=32 - groups << 8 | 31
            )
        scalars.append(scalar.ir_value())
    result_type = T.vector(registers, value.dtype.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(result_type, scalars, loc=loc, ip=ip),
        (registers,),
        value.dtype,
    )


def _constant(
    node: RegisterTensorNode,
    group: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:
    dtype = DTYPES[node["dtype"]]
    choices = []
    for row in node["rows"]:
        array = np.frombuffer(
            bytes.fromhex(row), dtype=NUMPY_DTYPES[node["dtype"]]
        ).copy()
        vector_type = T.vector(node["shape"][1], dtype.mlir_type)
        if node["dtype"] == "bool":
            array = np.packbits(array, bitorder="little")
        attr = ir.DenseElementsAttr.get(array, type=vector_type)  # pyrefly: ignore [missing-attribute]
        choices.append(
            cute.TensorSSA(
                arith.constant(vector_type, attr, loc=loc, ip=ip),
                (node["shape"][1],),
                dtype,
            )
        )
    return _select_group(choices, node["owners"], group)


def _minimum_maximum(
    left: cute.TensorSSA,
    right: cute.TensorSSA,
    *,
    minimum: bool,
    propagate_nan: bool = True,
    loc: object | None = None,
    ip: object | None = None,
) -> cute.TensorSSA:
    if left.dtype == cutlass.Float32:
        # Native extrema preserve subnormals; each ATen op controls whether
        # NaNs propagate or the available numeric operand is selected.
        # arith.minimumf/maximumf currently expand into comparisons and selects
        # in the CuTe pipeline. NVVM accepts scalar f32 operands, so retain the
        # tensor interface while spelling each register's intrinsic explicitly.
        operation = nvvm.fmin if minimum else nvvm.fmax
        left_value = left.ir_value()
        right_value = right.ir_value()
        elements = []
        for index in range(cute.size(left.shape)):
            lhs = vector.extract(left_value, [], [index], loc=loc, ip=ip)
            rhs = vector.extract(right_value, [], [index], loc=loc, ip=ip)
            elements.append(
                operation(lhs, rhs, nan=propagate_nan, ftz=False, loc=loc, ip=ip)
            )
        return cute.TensorSSA(
            vector.from_elements(left_value.type, elements, loc=loc, ip=ip),
            left.shape,
            left.dtype,
        )
    operation = cute.math.min if minimum else cute.math.max
    return operation(left, right, propagate_nan=propagate_nan, loc=loc, ip=ip)


@dsl_user_op
def _cute_numeric_extremum(
    left: cutlass.Float32 | cute.TensorSSA,
    right: cutlass.Float32 | cute.TensorSSA,
    *,
    minimum: bool,
    loc: object | None = None,
    ip: object | None = None,
) -> cutlass.Float32 | cute.TensorSSA:
    """Use scalar Float32 extrema directly and retain SDK tensor semantics."""
    if isinstance(left, cute.TensorSSA) or isinstance(right, cute.TensorSSA):
        operation = cute.math.min if minimum else cute.math.max
        return operation(left, right, propagate_nan=False, loc=loc, ip=ip)
    operation = nvvm.fmin if minimum else nvvm.fmax
    return cutlass.Float32(
        operation(
            left.ir_value(loc=loc, ip=ip),
            cast("cutlass.Float32", right).ir_value(loc=loc, ip=ip),
            nan=False,
            ftz=False,
            loc=loc,
            ip=ip,
        )
    )


def _cute_stage_register_plan(
    source: str,
    inputs: Sequence[cute.TensorSSA],
    lane: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> tuple[cute.TensorSSA, ...]:
    plan = _parse(source)
    group = lane % plan["groups"]
    values = {
        item["id"]: value for item, value in zip(plan["inputs"], inputs, strict=True)
    }
    for node in plan["nodes"]:
        op = node["op"]
        args = [values[index] for index in node["inputs"]]
        if op == "constant":
            result = _constant(node, group)
        elif op in (
            "aten.gather.default",
            "aten.index_select.default",
            "aten.slice.Tensor",
        ):
            mapping = plan["maps"][node["map"]]
            result = (
                _permute_by_group(args[0], args[0], mapping, group)
                if node["axis"] == 1
                else _gather_groups(
                    args[0],
                    mapping,
                    plan["groups"],
                    node["shape"][1],
                    node["live_registers"],
                    group,
                )
            )
        elif op == "aten.minimum.default":
            result = _minimum_maximum(*args, minimum=True)
        elif op == "aten.maximum.default":
            result = _minimum_maximum(*args, minimum=False)
        elif op == "aten.fmin.default":
            result = _minimum_maximum(*args, minimum=True, propagate_nan=False)
        elif op == "aten.fmax.default":
            result = _minimum_maximum(*args, minimum=False, propagate_nan=False)
        elif op == "aten.where.self":
            result = (
                _permute_by_group(*args, plan["maps"][node["static_map"]], group)
                if "static_map" in node
                else cute.where(*args)
            )
        elif op == "aten.cat.default":
            result = args[0]
            for value in args[1:]:
                result = _permute(
                    result,
                    value,
                    list(range(cute.size(result.shape) + cute.size(value.shape))),
                )
        elif op == "aten.add.Tensor":
            result = args[0] + args[1]
        elif op == "aten.sub.Tensor":
            result = args[0] - args[1]
        elif op == "aten.mul.Tensor":
            result = args[0] * args[1]
        elif op == "aten.neg.default":
            result = -args[0]
        elif op == "aten.eq.Tensor":
            result = args[0] == args[1]
        elif op == "aten.ne.Tensor":
            result = args[0] != args[1]
        elif op == "aten.lt.Tensor":
            result = args[0] < args[1]
        elif op == "aten.le.Tensor":
            result = args[0] <= args[1]
        elif op == "aten.gt.Tensor":
            result = args[0] > args[1]
        elif op == "aten.ge.Tensor":
            result = args[0] >= args[1]
        elif op == "aten.bitwise_and.Tensor":
            result = args[0] & args[1]
        elif op == "aten.bitwise_or.Tensor":
            result = args[0] | args[1]
        elif op == "aten.bitwise_xor.Tensor":
            result = args[0] ^ args[1]
        else:
            raise ValueError(f"Unsupported plan operation: {op}")
        assert result.dtype == DTYPES[node["dtype"]]
        values[node["id"]] = result
    return tuple(values[index] for index in plan["outputs"])


@dsl_user_op
def _cute_execute_register_plan(
    source: str,
    inputs: Sequence[cute.Tensor],
    lane: cutlass.Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> tuple[cute.Tensor, ...]:
    """Preserve the existing RowFragment rmem Tensor input/output ABI."""
    vectors = _cute_stage_register_plan(
        source, tuple(value.load() for value in inputs), lane
    )
    results = []
    for value in vectors:
        result = cute.make_rmem_tensor(value.shape, value.dtype)
        result.store(value)
        results.append(result)
    return tuple(results)
