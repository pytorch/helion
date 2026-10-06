"""Compose register selection with a row-fragment graph schedule."""

from __future__ import annotations

import ast
import operator
import textwrap
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language._tracing_ops import _get_symnode
from ...language.memory_ops import load
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from .row_fragment import RowFragment
from .row_fragment import RowFragmentEmitter
from .row_fragment import RowFragmentLayout
from .row_fragment import RowFragmentOrder
from .row_fragment import row_fragment_tensor_inputs
from .row_fragment_io import RowFragmentIO
from .row_value_facts import RowValueRequirement
from .row_value_facts import RowValueUses

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..generate_ast import GenerateAST
    from .fx_matcher import _GeneratedCodeTemplate
    from .topk import CuteTopKPlan


def codegen_row_topk(
    cg: GenerateAST,
    plan: CuteTopKPlan,
    *,
    template: _GeneratedCodeTemplate,
    loads: str,
    encode: str,
    fragment_size: int,
    key_type: str,
    key_padding: str,
    selected_key: str,
    selected_index: Callable[[str], str],
    value_store: Callable[[str, str, str | None, RowValueRequirement | None], str],
    helper_name: str,
    helper_hash: str,
    index_type: str,
    row_guard: str,
    group_guard: str,
    input_name: str,
    use_asm_encoder: bool,
    packed_type: str,
    input_word_type: str,
    selection_word_type: str,
) -> bool:
    graph = plan.fragment_graph
    assert graph is not None
    fn = cg.device_function
    backend = CompileEnvironment.current().backend
    statements: list[ast.AST] = []

    def add(source: str) -> None:
        for statement in ast.parse(source).body:
            cg.add_statement(statement)

    def generated(source: str) -> None:
        add(template.render(source))

    row = template.render("topk_row")
    lane = template.render("topk_lane")
    valid_row = template.render("topk_valid_row")
    input_layout = RowFragmentLayout(plan.lanes_per_row, plan.vector_width, lane)
    padded_k = 1 << (plan.k - 1).bit_length()
    output_vector = plan.output_vector_width
    if plan.selection_layout == "distributed":
        output_vector = min(output_vector, max(1, padded_k // plan.lanes_per_row))
    while output_vector > 1 and plan.k % output_vector:
        output_vector //= 2
    output_layout = RowFragmentLayout(plan.lanes_per_row, output_vector, lane)

    def resolve(node: Node) -> Node:
        while node in graph.aliases:
            node = graph.aliases[node]
        return node

    def index_expression(node: Node) -> str:
        if node.target is _get_symnode:
            return row
        fragment = emit(node)
        assert fragment.replicated
        return ast.unparse(fragment.element(0))

    io = RowFragmentIO(
        cg, row=row, lane=lane, valid_row=valid_row, index_expression=index_expression
    )
    value_uses = RowValueUses(graph.stores, resolve=resolve)
    value_nodes = [
        node
        for node in graph.selection.users
        if node.target is operator.getitem and node.args[1] == 0
    ]
    requirements = {value_uses.requirement(node) for node in value_nodes}
    value_requirement = (
        RowValueRequirement.EXACT
        if RowValueRequirement.EXACT in requirements
        else RowValueRequirement.NUMERIC
        if RowValueRequirement.NUMERIC in requirements
        else RowValueRequirement.UNUSED
    )

    def load_fragment(node: Node) -> RowFragment:
        return io.load(node, emitter.layout)

    emitter = RowFragmentEmitter(
        cg,
        load_fragment,
        layout=input_layout,
        valid_row=valid_row,
        resolve=resolve,
        numeric_uses_only=value_uses.numeric_only,
        row_expr=row,
    )

    def emit(node: Node) -> RowFragment:
        actual = resolve(node)
        for alias, target in graph.aliases.items():
            if resolve(target) in emitter.fragments:
                emitter.bind(alias, emitter.fragments[resolve(target)])
        fragment = emitter.emit(actual)
        emitter.bind(node, fragment)
        return fragment

    def scalar_source(node: Node, column: str) -> ast.AST:
        node = resolve(node)
        cached = emitter.fragments.get(node)
        if cached is not None and cached.replicated:
            # Full-row reductions were evaluated before selection. Their
            # replicated results can be reused at any selected column.
            return cached.element(0)
        if node.target is load:
            return expr_from_string(io.expression(node, column))
        inputs = [
            scalar_source(argument, column)
            for argument in row_fragment_tensor_inputs(node)
        ]
        return emitter.emit_pointwise_scalar(node, inputs, logical_column=column)

    with cg.set_statements(statements):
        generated(f"""
topk_thread = cutlass.Int32(cute.arch.thread_idx()[0])
topk_lane = topk_thread % cutlass.Int32({plan.lanes_per_row})
topk_group = topk_thread // cutlass.Int32({plan.lanes_per_row})
topk_row = ({index_type}(cute.arch.block_idx()[0]) * {index_type}({plan.rows_per_block})
            + {index_type}(topk_group))
topk_valid_row = ({row_guard}) & ({group_guard})
topk_keys = cute.make_rmem_tensor({fragment_size}, {key_type})
topk_keys.fill({key_padding})
""")
        source = resolve(graph.source)
        source_tensor_node = source.args[0]
        if (
            source.target is load
            and isinstance(source_tensor_node, Node)
            and source_tensor_node.meta["val"] is plan.x
        ):
            generated(f"""
topk_input_bits = cute.make_tensor(
    cute.recast_ptr({input_name}.iterator, dtype={input_word_type}), {input_name}.layout
)
{loads}
""")
        else:
            fragment = emit(source)
            bits = f"{fragment.name}[topk_i].bitcast({selection_word_type})"
            if plan.selection_dtype == torch.float32:
                bits = f"({bits}).bitcast(cutlass.Int32)"
            elif not use_asm_encoder:
                bits = f"cutlass.Int32(({bits}).bitcast(cutlass.Int16))"
            generated(f"""
for topk_i in cutlass.range_constexpr({fragment.num_registers}):
    topk_col = ((cutlass.Int32(topk_i // {plan.vector_width}) * {plan.lanes_per_row}
                 + topk_lane) * {plan.vector_width} + topk_i % {plan.vector_width})
    if topk_valid_row & (topk_col < {plan.n}):
        topk_bits = {bits}
{textwrap.indent(encode, "        ")}
""")
        generated(f"""
topk_selected = {helper_name}(topk_keys, {padded_k}, {plan.lanes_per_row},
                             {plan.sort_network!r}, {plan.merge_schedule!r})
""")

        # Give every selected rank exactly one owner, including K < subgroup size.
        registers = output_layout.num_registers(plan.k)
        selected = fn.new_var("row_selected_keys")
        add(
            f"{selected} = cute.make_rmem_tensor({registers}, {packed_type})\n"
            f"{selected}.fill({packed_type}(0))"
        )
        if plan.selection_layout == "distributed":
            key = selected_key.replace("topk_output", "topk_j")
            if output_vector > 1:
                transpose_name = f"_cute_subgroup_vectorize_{helper_hash}"
                output_lane = fn.new_var("row_output_lane")
                cg.module_statements.append(
                    ast.ImportFrom(
                        module="helion.runtime.cute.register_layout",
                        names=[
                            ast.alias(name="subgroup_vectorize", asname=transpose_name)
                        ],
                        level=0,
                    )
                )
                generated(
                    f"topk_transposed, {output_lane} = {transpose_name}(topk_selected, {output_vector}, {plan.lanes_per_row})"
                )
                key = key.replace("topk_selected[", "topk_transposed[")
                output_layout = RowFragmentLayout(
                    plan.lanes_per_row,
                    output_vector,
                    output_lane,
                    owner_lanes=(
                        tuple(
                            logical
                            % (plan.lanes_per_row // output_vector)
                            * output_vector
                            + logical // (plan.lanes_per_row // output_vector)
                            for logical in range(plan.lanes_per_row)
                        )
                        if output_vector <= plan.lanes_per_row
                        else None
                    ),
                )
            key = template.render(key)
            loop = template.render("topk_j")
            add(f"""
for {loop} in cutlass.range_constexpr({registers}):
    {selected}[{loop}] = {key}
""")
        else:
            generated(f"""
for topk_output in cutlass.range_constexpr({plan.k}):
    if topk_lane == (topk_output // {output_vector}) % {plan.lanes_per_row}:
        {selected}[(topk_output // {plan.lanes_per_row * output_vector}) * {output_vector} + topk_output % {output_vector}] = {selected_key}
""")

        values = (
            RowFragment(
                fn.new_var("row_selected_values"),
                plan.selection_dtype,
                plan.k,
                output_layout,
                order=RowFragmentOrder(plan.largest),
            )
            if value_requirement is not RowValueRequirement.UNUSED
            else None
        )
        indices = RowFragment(
            fn.new_var("row_selected_indices"), torch.int64, plan.k, output_layout
        )
        for fragment in (values, indices):
            if fragment is None:
                continue
            dtype = backend.dtype_str(fragment.dtype)
            add(
                f"{fragment.name} = cute.make_rmem_tensor({registers}, {dtype})\n"
                f"{fragment.name}.fill({dtype}(0))"
            )
        loop = fn.new_var("row_selected_register")
        column = output_layout.column(loop)
        key = f"{selected}[{loop}]"
        recovered_index = template.render("topk_selected_index")
        decoded_lines = []
        if values is not None:
            decode = value_store(
                key, f"{values.name}[{loop}]", "ROW_RECOVER_VALUE", value_requirement
            )
            rendered = template.render(decode)
            if "ROW_RECOVER_VALUE" not in rendered:
                decoded_lines = rendered.splitlines()
            else:
                recover_statements: list[ast.AST] = []
                with cg.set_statements(recover_statements):
                    recovery = scalar_source(source, recovered_index)
                recovery_lines = "\n".join(ast.unparse(s) for s in recover_statements)
                for line in rendered.splitlines():
                    if "ROW_RECOVER_VALUE" in line:
                        indent = line[: len(line) - len(line.lstrip())]
                        if recovery_lines:
                            decoded_lines.append(
                                textwrap.indent(recovery_lines, indent)
                            )
                        line = line.replace("ROW_RECOVER_VALUE", ast.unparse(recovery))
                    decoded_lines.append(line)
        index_expr = template.render(selected_index(key))
        add(f"""
for {loop} in cutlass.range_constexpr({registers}):
    if {valid_row} & (({column}) < {plan.k}):
        {recovered_index} = {index_expr}
{textwrap.indent(chr(10).join(decoded_lines), "        ")}
        {indices.name}[{loop}] = cutlass.Int64({recovered_index})
""")
        input_fragments = emitter.fragments
        # Input and selected values can use different vector/lane layouts.
        # Re-evaluate a pure input expression in the consumer's layout when
        # an epilogue also needs it; no shared-memory round trip is required.
        # Independent stores can reuse an already materialized input fragment.
        emitter = RowFragmentEmitter(
            cg,
            load_fragment,
            layout=output_layout,
            valid_row=valid_row,
            resolve=resolve,
            numeric_uses_only=value_uses.numeric_only,
            row_expr=row,
        )
        for node in graph.selection.users:
            if node.target is operator.getitem:
                fragment = values if node.args[1] == 0 else indices
                if fragment is not None:
                    emitter.bind(node, fragment)

        # Evaluate all values before any effect; the planner proves disjoint
        # output spans so neither recomputation nor another output can be clobbered.
        outputs = []
        for effect in graph.stores:
            value_node = effect.args[2]
            assert isinstance(value_node, Node)
            fragment = input_fragments.get(resolve(value_node))
            if fragment is None:
                fragment = emit(value_node)
            outputs.append((effect, fragment))
        for effect, fragment in outputs:
            io.store(effect, fragment)
    fn.preamble = []
    fn.body = statements
    return True
