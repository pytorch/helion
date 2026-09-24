"""Paired FP32 leaf transport in a retired initialized-chain A arena.

The descriptor covers a guarded compact input, not an example or kernel name.
Each 128x32 FP32 panel occupies 16 KiB of the existing 32 KiB A arena.
Raw panel 1 can alias weighted panel 0: both raw snapshots must precede any
weighted store. Each final K64 MMA retires before the next raw overwrite,
including when the surrounding contraction requested overlap64.
"""

from __future__ import annotations

import ast
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_pointwise_cache import PointwiseReadCache
    from .chained_pointwise_unroll import PointwiseUnroll

KEY = "cute_chained_leaf_pipeline"
KIND = "chained_paired_leaf_tma"


def compact_elements(sizes: tuple[int, ...], strides: tuple[int, ...]) -> int | None:
    if any(type(v) is not int or v <= 0 for v in (*sizes, *strides)):
        return None
    compact = 1
    for size, stride in sorted(
        zip(sizes, strides, strict=True), key=operator.itemgetter(1)
    ):
        if size != 1:
            if stride != compact:
                return None
            compact *= size
    return compact


def has_leaf_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Cheap typed seed filter; emission independently proves full geometry."""
    root = chain._root_graph(graphs)
    if root is None:
        return False
    dots = [node for node in root.graph.nodes if node.target is chain.dot]
    if len(dots) != 2:
        return False
    leaves = [
        node
        for node in chain._ancestors(cast("Node", dots[-1].args[0]))
        if node.target is chain.memory_ops.load
        and node.meta["val"].ndim == 2
        and cast("Node", node.args[0]).target is chain._tracing_ops._host_tensor
    ]
    if len(leaves) != 1:
        return False
    source = cast("Node", leaves[0].args[0])
    fake = source.meta["val"]
    return (
        source.target is chain._tracing_ops._host_tensor
        and fake.dtype == torch.float32
        and compact_elements(tuple(fake.shape), tuple(fake.stride())) is not None
    )


def interval(value: sympy.Expr, ranges: dict[sympy.Symbol, int]) -> tuple[int, int]:
    """Affine interval; only provably constant floor expressions are folded."""
    value = constant_floors(value, ranges)
    base = value.subs(dict.fromkeys(ranges, 0))
    if not isinstance(base, sympy.Integer):
        raise chain._UnsupportedChain("paired leaf needs a constant affine base")
    low = high = int(base)
    remainder = sympy.Add(value, -base)
    for symbol, maximum in ranges.items():
        slope = sympy.diff(value, symbol)
        if not isinstance(slope, sympy.Integer):
            raise chain._UnsupportedChain("paired leaf needs affine indices")
        remainder -= slope * symbol
        low += min(0, int(slope) * maximum)
        high += max(0, int(slope) * maximum)
    if sympy.expand(remainder) != 0:
        raise chain._UnsupportedChain("paired leaf has an unbounded dependency")
    return low, high


def constant_floors(value: sympy.Expr, ranges: dict[sympy.Symbol, int]) -> sympy.Expr:
    # Grouped read-only inputs can contain floor(origin / group_size).
    # A varying quotient remains unsupported; no approximation is used.
    for floor in sorted(value.atoms(sympy.floor), key=lambda item: len(str(item))):
        numerator, denominator = floor.args[0].as_numer_denom()
        if not isinstance(denominator, sympy.Integer) or denominator <= 0:
            raise chain._UnsupportedChain("paired leaf has an unsupported quotient")
        low, high = interval(numerator, ranges)
        if low // int(denominator) != high // int(denominator):
            raise chain._UnsupportedChain("paired leaf quotient varies over the root")
        value = value.xreplace({floor: sympy.Integer(low // int(denominator))})
    return sympy.expand(value)


@dataclasses.dataclass(frozen=True)
class LeafPipeline:
    mode: str
    leaf: Node
    row: str
    col: str
    atom: str
    tensor: str
    wrapper: dict[str, object]

    def issue(self, half: str, panel: str) -> list[str]:
        """One warp issues; all threads later wait and acknowledge the phase."""
        return [
            "if chain_thread == 0:",
            "    cute.arch.mbarrier_arrive_and_expect_tx(chain_leaf_bar, 16384)",
            "if chain_warp == 0:",
            "    chain_leaf_layout = cute.tile_to_shape(tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float32), (128, 32), order=(0, 1))",
            f"    chain_leaf_ptr = cute.recast_ptr(chain_a_workspace + ({panel}) * 8192, dtype=cutlass.Float32)",
            "    chain_leaf_raw = cute.make_tensor(cute.recast_ptr(chain_leaf_ptr, chain_leaf_layout.inner, dtype=cutlass.Float32), chain_leaf_layout.outer)",
            f"    chain_leaf_global = cute.local_tile({self.tensor}, (128, 32), (({self.row}) // 128, ({self.col}) // 32 + ({half}) * 2 + ({panel})))",
            f"    chain_leaf_smem, chain_leaf_gmem = cute.nvgpu.cpasync.tma_partition({self.atom}, 0, cute.make_layout(1), cute.group_modes(chain_leaf_raw, 0, 2), cute.group_modes(chain_leaf_global, 0, 2))",
            f"    cute.copy({self.atom}, chain_leaf_gmem, chain_leaf_smem, tma_bar_ptr=chain_leaf_bar)",
        ]


def plan_leaf_pipeline(
    cg: GenerateAST,
    plan: chain.ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    inner: int,
) -> LeafPipeline | None:
    mode = cast("str", cg.device_function.config.config.get(KEY, "legacy"))
    if mode == "legacy":
        return None
    if (
        mode not in ("paired_tma", "paired_tma_coeff_prefetch")
        or plan.k_schedule is None
        or plan.initialized_accumulator is None
        or plan.late_rhs_reuse is None
        or len(plan.dots) != 2
        or plan.shapes[-1][0] != 128
        or plan.shapes[-1][2] != 128
        or plan.late_rhs_reuse.a_bytes < 32768
        or plan.threads != 128
        or inner != 1
        or plan.scan_exports
        or plan.direct_output
    ):
        raise chain._UnsupportedChain(
            "paired leaf requires a retired initialized M128 K128 A arena"
        )
    operand = cast("Node", plan.dots[-1].args[0])
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    names = ("chain_leaf_row", "chain_leaf_col")
    expression.coordinate_names.update(names)
    expression.value(operand, names)
    leaves = [
        loaded
        for loaded in expression.loaded_inputs
        if loaded[0].meta["val"].ndim == 2
        and cast("Node", loaded[0].args[0]).target is chain._tracing_ops._host_tensor
    ]
    if len(leaves) != 1:
        raise chain._UnsupportedChain("paired leaf requires exactly one matrix leaf")
    leaf, _, indices, _ = leaves[0]
    source = cast("Node", leaf.args[0])
    fake = source.meta["val"]
    sizes, strides = tuple(fake.shape), tuple(fake.stride())
    compact = compact_elements(sizes, strides)
    if (
        source.target is not chain._tracing_ops._host_tensor
        or fake.dtype != torch.float32
        or compact is None
    ):
        raise chain._UnsupportedChain("paired leaf requires compact FP32 host storage")
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*names, *expression.origins.values())
    }
    row, col = (symbols[name] for name in names)
    ranges = {
        symbols[expression.origins[axis]]: extent - block
        for axis, extent, block in plan.axes
    }
    domain = {**ranges, row: 127, col: 127}
    bases, slopes = [], []
    for index, extent in zip(indices, sizes, strict=True):
        value = constant_floors(
            chain._copy_index(index, expression.definitions, symbols), domain
        )
        low, high = interval(value, domain)
        if low < 0 or high >= extent:
            raise chain._UnsupportedChain("paired leaf requires a whole in-bounds tile")
        pair = sympy.diff(value, row), sympy.diff(value, col)
        if any(not isinstance(item, sympy.Integer) for item in pair):
            raise chain._UnsupportedChain(
                "paired leaf requires affine row/column ownership"
            )
        bases.append(value.subs({row: 0, col: 0}))
        slopes.append(pair)
    physical = tuple(
        sum(int(pair[i]) * stride for pair, stride in zip(slopes, strides, strict=True))
        for i in (0, 1)
    )
    pitch = physical[0]
    if physical[1] != 1 or pitch < 128 or pitch % 4 or compact % pitch:
        raise chain._UnsupportedChain("paired leaf requires contiguous aligned columns")
    if max(pitch, compact // pitch) >= 2**31:
        raise chain._UnsupportedChain("paired leaf exceeds TensorMap extent")
    base = sympy.expand(
        sum(value * stride for value, stride in zip(bases, strides, strict=True))
    )
    constant = base.subs(dict.fromkeys(ranges, 0))
    if not isinstance(constant, sympy.Integer) or constant < 0:
        raise chain._UnsupportedChain("paired leaf has an unsupported base")
    row_base, col_base = (
        sympy.Integer(int(constant) // pitch),
        sympy.Integer(int(constant) % pitch),
    )
    for symbol in ranges:
        slope = sympy.diff(base, symbol)
        if not isinstance(slope, sympy.Integer) or slope < 0:
            raise chain._UnsupportedChain(
                "paired leaf requires nonnegative root strides"
            )
        row_base = sympy.Add(row_base, sympy.Mul(int(slope) // pitch, symbol))
        col_base = sympy.Add(col_base, sympy.Mul(int(slope) % pitch, symbol))
    if (
        interval(row_base, ranges)[1] + 128 > compact // pitch
        or interval(col_base, ranges)[1] + 128 > pitch
    ):
        raise chain._UnsupportedChain("paired leaf descriptor tile escapes its matrix")
    for coordinate, tile in ((row_base, 128), (col_base, 32)):
        if int(
            cast("sympy.Integer", coordinate.subs(dict.fromkeys(ranges, 0)))
        ) % tile or any(
            int(sympy.diff(coordinate, symbols[expression.origins[axis]]))
            * block
            % tile
            for axis, _, block in plan.axes
        ):
            raise chain._UnsupportedChain("paired leaf needs aligned atom origins")
    df = cg.device_function
    atom, tensor = df.new_var("leaf_tma_atom"), df.new_var("leaf_tma_tensor")
    wrapper: dict[str, object] = {
        "kind": KIND,
        "lhs_name": df.tensor_arg(fake, prefer_name=cast("str", source.args[0])).name,
        "shape": sizes,
        "strides": strides,
        "rows": compact // pitch,
        "columns": pitch,
        "tile": (128, 32),
        "kernel_args": [atom, tensor],
    }
    df.wrapper_only_params.extend((atom, tensor))
    cg.cute_wrapper_plans.append(wrapper)
    return LeafPipeline(
        mode,
        leaf,
        chain._copy_code(row_base),
        chain._copy_code(col_base),
        atom,
        tensor,
        wrapper,
    )


def finish_wrapper(
    cg: GenerateAST, plan: chain.ChainedMatmulPlan, transfer: LeafPipeline
) -> None:
    # The ordinary epilogue must register its scalar reads and output first.
    # Eagerly requesting output here during planning changes sorted argument
    # order even though the caller values would remain paired correctly.
    output = cast("Node", plan.store.args[0])
    transfer.wrapper["out_name"] = cg.device_function.tensor_arg(
        output.meta["val"], prefer_name=cast("str", output.args[0])
    ).name


def row_read_cache(
    expression: chain._Expression, names: tuple[str, str], step: str, tag: str
) -> list[str]:
    """Snapshot independent complete typed RHSs, not arithmetic subexpressions."""
    reads = [
        (expression.definitions[value], cast("Node", node.args[0]).meta["val"].dtype)
        for node, _, _, value in expression.loaded_inputs
    ]
    reads.extend(
        (value, node.meta["val"].dtype)
        for (node, _), value in expression.memo.items()
        if node in expression.plan.scans
    )
    chosen: dict[str, tuple[str, torch.dtype]] = {}
    for text, dtype in reads:
        tree = ast.parse(text, mode="eval").body
        deps = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if (
            dtype in (torch.float32, torch.bfloat16, torch.float16)
            and names[0] in deps
            and names[1] not in deps
            and not deps & expression.definitions.keys()
        ):
            chosen.setdefault(ast.dump(tree), (text, dtype))
    if not chosen or len(chosen) > 4:
        return []
    replacements: dict[str, ast.expr] = {}
    lines = []
    index = f"{tag}_row_cache_step"
    for i, (key, (text, dtype)) in enumerate(chosen.items()):
        cache = f"{tag}_row_cache_{i}"
        dtype_name = CompileEnvironment.current().backend.dtype_str(dtype)
        lines.append(f"{cache} = cute.make_rmem_tensor((4,), {dtype_name})")
        tree = ast.parse(text, mode="eval").body

        class Row(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == names[0]:
                    return ast.parse(
                        f"chain_thread // 4 + {index} * 32", mode="eval"
                    ).body
                return node

        lines.extend(
            (
                f"for {index} in cutlass.range_constexpr(4):",
                f"    {cache}[{index}] = {ast.unparse(Row().visit(tree))}",
            )
        )
        replacements[key] = ast.parse(f"{cache}[{step}]", mode="eval").body

    class Replace(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if isinstance(node, ast.expr) and ast.dump(node) in replacements:
                return replacements[ast.dump(node)]
            return super().visit(node)

    expression.lines = [
        ast.unparse(Replace().visit(ast.parse(line))) for line in expression.lines
    ]
    return lines


def produce_half(
    cg: GenerateAST,
    plan: chain.ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    transfer: LeafPipeline,
    operand: Node,
    dtype: str,
    target: str,
    fallback: list[str],
    pointwise_cache: PointwiseReadCache,
    pointwise_unroll: PointwiseUnroll,
) -> list[str]:
    from .chained_pointwise_cache import PointwiseReadCache
    from .chained_tcgen05 import _hoist
    from .chained_tcgen05 import _vector_leaf

    cache = pointwise_cache
    unroll = pointwise_unroll
    tag = "chain_leaf"
    names = (f"{tag}_row", f"{tag}_col")
    element, step, atom = f"{tag}_element", f"{tag}_step", f"{tag}_panel"
    probe = chain._Expression(cg, plan, boundaries)
    probe.scan_inputs = scans
    probe.coordinate_names.update(names)
    probe.value(operand, names)
    occurrences = [item for item in probe.loaded_inputs if item[0] is transfer.leaf]
    if len(occurrences) != 1:
        raise chain._UnsupportedChain("paired leaf has a repeated coordinate use")
    node, coords, indices, _ = occurrences[0]
    vector = _vector_leaf(probe, node, coords, indices, (128, 128), names)
    if vector is None or vector.dtype != torch.float32:
        raise chain._UnsupportedChain(
            "paired leaf requires the original aligned vector proof"
        )
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(names)
    expression.memo[node, coords] = f"{tag}_raw_values[{element}, {step}, 0]"
    value = expression.value(operand, names)
    domain = chain._operand_domain(cg, operand, names, plan)
    column_base = f"chain_k_half * 64 + {atom} * 32"
    # A requested ordinary read cache keeps its exact RHSs. The prefetch mode
    # additionally enables these snapshots and independent row snapshots.
    local_cache = (
        cache
        if cache.enabled
        else PointwiseReadCache(transfer.mode.endswith("coeff_prefetch"))
    )
    coeff = local_cache.prepare(
        expression, names, element, tag, 4, 4, column_base=column_base
    )
    if transfer.mode.endswith("coeff_prefetch"):
        coeff.extend(row_read_cache(expression, names, step, tag))
        if not coeff:
            raise chain._UnsupportedChain(
                "coefficient prefetch has no independent typed reads"
            )
    invariant, varying = _hoist(expression, {names[1], element})
    compute = [
        # The original half has eight row trips. Two constexpr atom panels
        # split that group into four trips each without dropping its unrolling.
        f"for {step} in cutlass.range(4, unroll={min(4, unroll.loop_factor(8))}):",
        f"    {names[0]} = chain_thread // 4 + {step} * 32",
        chain._indent(invariant),
        f"    for {element} in cutlass.range_constexpr(8):",
        f"        {names[1]} = {column_base} + chain_thread % 4 * 8 + {element}",
        chain._indent(varying, 8),
        f"        {tag}_values[{element}] = {chain._masked_operand(value, dtype, domain)}",
        f"    if cutlass.const_expr({atom} == 0):",
        f"        for {element} in cutlass.range_constexpr(8):",
        f"            {tag}_retained[{element}, {step}] = {tag}_values[{element}]",
        "    else:",
        f"        cute.copy({tag}_copy, {tag}_values, {tag}_target[None, {step}, 0])",
    ]
    panel = [
        f"{tag}_target = {tag}_thread.partition_D(cute.local_tile({target}, (128, 32), (0, chain_k_half * 2 + {atom})))",
        *(coeff if transfer.mode.endswith("coeff_prefetch") else []),
        f"cute.arch.mbarrier_wait(chain_leaf_bar, {atom})",
        f"{tag}_layout = cute.tile_to_shape(tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float32), (128, 32), order=(0, 1))",
        f"{tag}_ptr = cute.recast_ptr(chain_a_workspace + {atom} * 8192, dtype=cutlass.Float32)",
        f"{tag}_raw = cute.make_tensor(cute.recast_ptr({tag}_ptr, {tag}_layout.inner, dtype=cutlass.Float32), {tag}_layout.outer)",
        f"{tag}_partition = {tag}_raw_thread.partition_S({tag}_raw)",
        f"{tag}_raw_values = cute.make_rmem_tensor({tag}_partition.shape, cutlass.Float32)",
        f"cute.copy({tag}_raw_copy, {tag}_partition, {tag}_raw_values)",
        "cute.arch.fence_view_async_shared()",
        "cute.arch.sync_threads()",
        f"if cutlass.const_expr({atom} == 0):",
        chain._indent(transfer.issue("chain_k_half", "1")),
        "else:",
        f"    {tag}_retained_target = {tag}_thread.partition_D(cute.local_tile({target}, (128, 32), (0, chain_k_half * 2)))",
        f"    for {step} in cutlass.range_constexpr(4):",
        f"        cute.copy({tag}_copy, {tag}_retained[None, {step}], {tag}_retained_target[None, {step}, 0])",
        *(coeff if not transfer.mode.endswith("coeff_prefetch") else []),
        *compute,
    ]
    fast = [
        f"{tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 8)))",
        f"{tag}_thread = {tag}_copy.get_slice(chain_thread)",
        f"{tag}_raw_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128), cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 8)))",
        f"{tag}_raw_thread = {tag}_raw_copy.get_slice(chain_thread)",
        f"{tag}_retained = cute.make_rmem_tensor((8, 4), {dtype})",
        f"{tag}_target = {tag}_thread.partition_D(cute.local_tile({target}, (128, 32), (0, chain_k_half * 2)))",
        f"{tag}_values = cute.make_rmem_tensor({tag}_target[None, 0, 0].shape, {dtype})",
        f"for {atom} in cutlass.range_constexpr(2):",
        chain._indent(panel),
    ]
    # The scalar branch is byte-for-byte the original producer. It still drains
    # both transfers before writing, so parity and lifetime are branch invariant.
    bounds = [*vector.bounds, f"{tag}_source_pointer.toint() % 16 == 0"]
    return [
        "if cutlass.const_expr(chain_k_half != 0):",
        chain._indent(transfer.issue("chain_k_half", "0")),
        f"{tag}_source_pointer = {vector.tensor}.iterator + ({vector.offset})",
        f"if {' & '.join(f'({bound})' for bound in dict.fromkeys(bounds))}:",
        chain._indent(fast),
        "else:",
        "    cute.arch.mbarrier_wait(chain_leaf_bar, 0)",
        "    cute.arch.fence_view_async_shared()",
        "    cute.arch.sync_threads()",
        chain._indent(transfer.issue("chain_k_half", "1")),
        "    cute.arch.mbarrier_wait(chain_leaf_bar, 1)",
        "    cute.arch.fence_view_async_shared()",
        "    cute.arch.sync_threads()",
        chain._indent(fallback),
    ]
