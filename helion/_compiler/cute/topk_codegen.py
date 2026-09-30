"""Emit register selection for a proven row-wise top-k region."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ...runtime.cute import ordered_key as runtime_ordered_key
from ...runtime.cute import register_layout as runtime_register_layout
from ...runtime.cute import sorting_networks as runtime_sorting_networks
from ...runtime.cute import topk as runtime_topk
from ..program_id import XYZProgramIDs
from ..tile_strategy import DeviceGridState
from .fx_matcher import _GeneratedCodeTemplate
from .row_value_facts import RowValueRequirement

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .topk import CuteTopKPlan


def codegen_topk_root(cg: GenerateAST, plan: CuteTopKPlan) -> bool:
    """Replace the matched root with a subgroup-local selection network.

    A signed rank for the 16-bit value and a reversed column index fit in
    31 bits for rows of at most 32768 elements. Int32 minimum padding cannot
    displace a real value, including infinity or NaN.
    Values can be gathered or decoded after selection. Canonical NaN ranks
    retain the input gather. Signed ranks also gather zeros; ordinal ranks
    distinguish their signs and decode them exactly.
    """
    fn = cg.device_function
    root = cg.current_root_graph_info
    if root is None or root.graph is not plan.root_graph or fn.pid is None:
        return False
    grid = cg.current_grid_state
    if not isinstance(grid, DeviceGridState):
        return False
    row_info = grid.block_id_to_info[plan.row_block_id]
    if row_info.begin_expr != 0 or row_info.end_expr != plan.x.size(0):
        return False
    pid_info = fn.pid.pid_info
    if len(pid_info) != 1 or pid_info[0].block_id != plan.row_block_id:
        return False
    fn.pid = XYZProgramIDs(
        pid_info=[pid_info[0]._replace(block_size_var=str(plan.rows_per_block))]
    )

    x_name = fn.tensor_arg(plan.x).name
    values_name = fn.tensor_arg(plan.values).name if plan.values is not None else ""
    indices_name = fn.tensor_arg(plan.indices).name if plan.indices is not None else ""
    template = _GeneratedCodeTemplate(
        "topk",
        (*tuple(argument.name for argument in fn.arguments), *cg._extra_params),
        fn.new_var,
    )
    x = template.protect(x_name)
    values = template.protect(values_name)
    indices = template.protect(indices_name)
    padded_k = 1 << (plan.k - 1).bit_length()
    distributed = plan.selection_layout == "distributed"
    output_lanes = (
        min(plan.lanes_per_row, padded_k) if distributed else plan.lanes_per_row
    )
    selected_per_lane = max(1, padded_k // plan.lanes_per_row)
    # The persistent CuTe cache keys generated source rather than imported
    # helper code. Include the helper content in the actual imported symbol.
    helper_hash = hashlib.sha256(
        (
            inspect.getsource(runtime_topk)
            + inspect.getsource(runtime_sorting_networks)
            + inspect.getsource(runtime_ordered_key)
            + inspect.getsource(runtime_register_layout)
        ).encode("utf-8")
    ).hexdigest()[:16]
    helper_function = "distributed_topk" if distributed else "local_topk"
    helper_name = f"_cute_{helper_function}_{helper_hash}"
    cg.module_statements.append(
        ast.ImportFrom(
            module="helion.runtime.cute.topk",
            names=[ast.alias(name=helper_function, asname=helper_name)],
            level=0,
        )
    )
    index_bits = (plan.n - 1).bit_length()
    index_mask = (1 << index_bits) - 1
    # NaNs use rank 32767. For either dtype, ordinal negative infinity
    # (-1 - infinity_bits) also fits this conservative signed rank range.
    max_key = (32767 << index_bits) | index_mask
    min_key = -(32767 << index_bits)
    key_type = "cutlass.Int32"
    key_padding = "cutlass.Int32(-2147483648)"
    encoded_key = "topk_packed"
    selected_key = "topk_selected[topk_output]"
    if plan.key_dtype == "float32" and max_key <= 2**24:
        # Numeric conversion preserves every packed integer, including the
        # power-of-two padding sentinel, within Float32's exact integer range.
        key_type = "cutlass.Float32"
        key_padding = "cutlass.Float32(-2147483648)"
        encoded_key = "cutlass.Float32(topk_packed)"
        selected_key = "cutlass.Int32(topk_selected[topk_output])"
    elif (
        plan.key_dtype == "float32_bits"
        and 0x40000000 + min_key >= 0x00800000
        and 0x40000000 + max_key <= 0x7F7FFFFF
    ):
        # Positive normal Float32 bit patterns have the same order as their
        # integers. A fixed bias avoids subnormal/NaN keys; zero pads below all
        # real keys. Wider rows retain Int32 selection.
        key_type = "cutlass.Float32"
        key_padding = "cutlass.Float32(0.0)"
        encoded_key = (
            "(topk_packed + cutlass.Int32(1073741824)).bitcast(cutlass.Float32)"
        )
        selected_key = (
            "topk_selected[topk_output].bitcast(cutlass.Int32)"
            " - cutlass.Int32(1073741824)"
        )
    native_float = plan.key_dtype == "float32_native" and plan.n <= (
        16384 if plan.x.dtype == torch.bfloat16 else 8192
    )
    if native_float:
        key_type = "cutlass.Float32"
        key_padding = "-cutlass.Float32.inf"
        selected_key = "topk_selected[topk_output].bitcast(cutlass.Int32)"
        if not plan.largest:
            selected_key = f"({selected_key} ^ cutlass.Int32(-2147483648))"
    per_lane = (plan.n + plan.lanes_per_row - 1) // plan.lanes_per_row
    fragment_size = max(
        1 << (per_lane - 1).bit_length(),
        1 if distributed else padded_k,
        plan.vector_width,
    )
    infinity_bits = 0x7F80 if plan.x.dtype == torch.bfloat16 else 0x7C00
    reverse = "topk_ordered = -topk_ordered" if not plan.largest else ""
    # Ordinal ranks preserve both zero signs; ordering these otherwise tied
    # values is permitted by top-k's unspecified tie ordering.
    rank_expression = (
        "topk_bits ^ (topk_sign & cutlass.Int32(32767))"
        if plan.rank_mode == "ordinal"
        else "(topk_magnitude ^ topk_sign) - topk_sign"
    )
    encode = f"""
topk_magnitude = topk_bits & cutlass.Int32(32767)
topk_sign = topk_bits >> cutlass.Int32(31)
topk_ordered = {rank_expression}
if topk_magnitude > cutlass.Int32({infinity_bits}):
    topk_ordered = cutlass.Int32(32767)
{reverse}
topk_packed = ((topk_ordered << cutlass.Int32({index_bits}))
               | (cutlass.Int32({index_mask}) - topk_col))
topk_keys[topk_i] = {encoded_key}
"""
    use_asm_encoder = not native_float and plan.key_encoder in ("asm", "paired")
    if use_asm_encoder:
        encode_helper = f"_cute_encode_ordered_topk_{helper_hash}"
        cg.module_statements.append(
            ast.ImportFrom(
                module="helion.runtime.cute.topk",
                names=[ast.alias(name="encode_ordered_topk_key", asname=encode_helper)],
                level=0,
            )
        )
        encode = f"""
topk_packed = {encode_helper}(topk_bits, topk_col, {index_bits}, {plan.largest!r}, {plan.rank_mode!r}, {infinity_bits})
topk_keys[topk_i] = {encoded_key}
"""
    if native_float:
        # Every FP16/BF16 value has at least log2(N) unused Float32 mantissa
        # bits. Keep finite values in their native Float32 order. Exceptional
        # values use finite keys above the largest BF16 value and are gathered
        # on output, preserving NaN payloads exactly.
        input_type = (
            "cutlass.BFloat16" if plan.x.dtype == torch.bfloat16 else "cutlass.Float16"
        )
        # BF16 leaves 16 low Float32 bits free. Reserve bit 14 above the index
        # payload: a quarter-ULP bias separates signed zeros and rounds back to
        # every finite BF16 value once the payload is cleared. FP16 retains
        # its zero-only bias because it has fewer free mantissa bits.
        zero_bias = (
            "topk_native_bits = topk_native_bits | cutlass.Int32(16384)"
            if plan.x.dtype == torch.bfloat16
            else "if topk_magnitude == 0:\n    topk_native_bits = topk_native_bits | cutlass.Int32(32768)"
        )
        flip = "topk_native_key = -topk_native_key" if not plan.largest else ""
        encode = f"""
topk_magnitude = topk_bits & cutlass.Int32(32767)
topk_native_bits = cutlass.Float32(cutlass.Uint16(topk_bits).bitcast({input_type})).bitcast(cutlass.Int32)
{zero_bias}
if topk_magnitude == cutlass.Int32({infinity_bits}):
    topk_native_bits = cutlass.Int32(2139062272) | ((topk_bits & cutlass.Int32(32768)) << cutlass.Int32(16))
if topk_magnitude > cutlass.Int32({infinity_bits}):
    topk_native_bits = cutlass.Int32(2139078656)
topk_encoded_index = cutlass.Int32({index_mask}) - topk_col
if topk_native_bits < 0:
    topk_encoded_index = topk_col
topk_native_key = (topk_native_bits | topk_encoded_index).bitcast(cutlass.Float32)
{flip}
topk_keys[topk_i] = topk_native_key
"""
    vector_word = "cutlass.Uint16(topk_vector[topk_element])"
    scalar_word = "topk_input_bits[topk_row, topk_col]"
    if not use_asm_encoder:
        vector_word = f"cutlass.Int32({vector_word}.bitcast(cutlass.Int16))"
        scalar_word = f"cutlass.Int32({scalar_word}.bitcast(cutlass.Int16))"
    # Vector loads require an aligned logical base and complete aligned rows.
    # Irregular widths and sliced bases retain the scalar masked path.
    row_guard = (
        "True"
        if plan.x.size(0) % plan.rows_per_block == 0
        else f"topk_row < {x}.shape[0]"
    )
    group_guard = (
        "True"
        if plan.lanes_per_row * plan.rows_per_block >= 32
        else f"topk_group < cutlass.Int32({plan.rows_per_block})"
    )
    input_full = fragment_size * plan.lanes_per_row == plan.n
    scalar_col_guard = "True" if input_full else f"topk_col < cutlass.Int32({plan.n})"
    vector_col_guard = (
        "True" if input_full else f"topk_col_base < cutlass.Int32({plan.n})"
    )
    output_col_guard = (
        "True"
        if plan.k % plan.lanes_per_row == 0
        else f"topk_output_col < cutlass.Int32({plan.k})"
    )
    row_stride = plan.x.stride(0)
    max_input_offset = (
        max(
            row_stride,
            (plan.x.size(0) - 1) * row_stride + (plan.n - 1) * plan.x.stride(-1),
        )
        if isinstance(row_stride, int)
        else None
    )
    max_output_offset = plan.x.size(0) * plan.k - 1
    if plan.fragment_graph is not None:
        max_output_offset = max(
            sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
            for tensor in plan.fragment_graph.tensors
        )
    index_type = (
        "cutlass.Int32"
        if max_input_offset is not None
        and max(max_input_offset, max_output_offset) <= 2147483647
        else "cutlass.Int64"
    )
    vectorized = (
        plan.vector_width > 1
        and plan.x.stride(-1) == 1
        and plan.n % plan.vector_width == 0
        and isinstance(row_stride, int)
        and row_stride % plan.vector_width == 0
    )
    vector_loads = f"""
for topk_chunk in cutlass.range({fragment_size // plan.vector_width}, unroll_full=True):
    topk_col_base = ((cutlass.Int32(topk_chunk * {plan.lanes_per_row})
                      + topk_lane) * cutlass.Int32({plan.vector_width}))
    if topk_valid_row & ({vector_col_guard}):
        topk_vector = cute.arch.load(
            topk_input_bits.iterator + topk_row * {index_type}({row_stride}) + {index_type}(topk_col_base),
            ir.VectorType.get([{plan.vector_width}], cutlass.Uint16.mlir_type),
        )
        for topk_element in cutlass.range_constexpr({plan.vector_width}):
            topk_i = topk_chunk * {plan.vector_width} + topk_element
            topk_col = topk_col_base + cutlass.Int32(topk_element)
            topk_bits = {vector_word}
{textwrap.indent(encode, "            ")}
"""
    if (
        vectorized
        and plan.key_encoder == "paired"
        and not native_float
        and plan.rank_mode == "ordinal"
    ):
        pair_helper = f"_cute_encode_ordinal_pair_{helper_hash}"
        cg.module_statements.append(
            ast.ImportFrom(
                module="helion.runtime.cute.topk",
                names=[ast.alias(name="encode_ordinal_topk_pair", asname=pair_helper)],
                level=0,
            )
        )
        # The existing vector proof covers both adjacent words, byte alignment
        # and row/column tails. Recast the already-offset pointer so address
        # arithmetic retains its original element units and selected width.
        # A single packed word is a scalar load. NVVM drops vector<1xi32>
        # load_ext values in this path, leaving an undefined encoder input.
        pair_load_type = (
            "cutlass.Uint32"
            if plan.vector_width == 2
            else f"ir.VectorType.get([{plan.vector_width // 2}], cutlass.Uint32.mlir_type)"
        )
        pair_word = (
            "topk_pairs"
            if plan.vector_width == 2
            else "cutlass.Uint32(topk_pairs[topk_pair])"
        )
        vector_loads = f"""
for topk_chunk in cutlass.range({fragment_size // plan.vector_width}, unroll_full=True):
    topk_col_base = ((cutlass.Int32(topk_chunk * {plan.lanes_per_row})
                      + topk_lane) * cutlass.Int32({plan.vector_width}))
    if topk_valid_row & ({vector_col_guard}):
        topk_pairs = cute.arch.load(
            cute.recast_ptr(topk_input_bits.iterator + topk_row * {index_type}({row_stride}) + {index_type}(topk_col_base), dtype=cutlass.Uint32),
            {pair_load_type},
        )
        for topk_pair in cutlass.range_constexpr({plan.vector_width // 2}):
            topk_i = topk_chunk * {plan.vector_width} + 2 * topk_pair
            topk_col = topk_col_base + cutlass.Int32(2 * topk_pair)
            topk_low, topk_high = {pair_helper}({pair_word}, topk_col, {index_bits}, {plan.largest!r}, {infinity_bits})
            topk_packed = topk_low
            topk_keys[topk_i] = {encoded_key}
            topk_packed = topk_high
            topk_keys[topk_i + 1] = {encoded_key}
"""
    scalar_loads = f"""
for topk_i in cutlass.range({fragment_size}, unroll_full=True):
    topk_col = (
        (cutlass.Int32(topk_i // {plan.vector_width})
         * cutlass.Int32({plan.lanes_per_row}) + topk_lane)
        * cutlass.Int32({plan.vector_width})
        + cutlass.Int32(topk_i % {plan.vector_width})
    )
    if topk_valid_row & ({scalar_col_guard}):
        topk_bits = {scalar_word}
{textwrap.indent(encode, "        ")}
"""
    loads = scalar_loads
    if vectorized:
        loads = (
            f"if cutlass.const_expr(topk_input_bits.iterator.alignment >= {plan.vector_width * 2}):\n"
            + textwrap.indent(vector_loads, "    ")
            + "else:\n"
            + textwrap.indent(scalar_loads, "    ")
        )
    value_dtype = (
        "cutlass.BFloat16" if plan.x.dtype == torch.bfloat16 else "cutlass.Float16"
    )
    output_index_dtype = (
        "cutlass.Int32"
        if plan.indices is not None and plan.indices.dtype == torch.int32
        else "cutlass.Int64"
    )
    output_index_bytes = plan.indices.element_size() if plan.indices is not None else 8

    def selected_index(key: str) -> str:
        if native_float:
            return f"(({key} ^ (({key} >> cutlass.Int32(31)) ^ cutlass.Int32(-1))) & cutlass.Int32({index_mask}))"
        return f"(cutlass.Int32({index_mask}) - ({key} & cutlass.Int32({index_mask})))"

    def value_store(
        key: str,
        destination: str,
        source: str | None = None,
        requirement: RowValueRequirement | None = None,
    ) -> str:
        numeric = (
            requirement is RowValueRequirement.NUMERIC
            if requirement is not None
            else plan.softmax
        )
        if source is None:
            source = f"{x}[topk_row, topk_selected_index]"
        if plan.value_mode == "gather":
            return f"{destination} = {source}\n"
        if native_float:
            if numeric:
                # Both infinity sentinels round to signed infinity in FP16
                # and BF16. NaN keys are positive even after undoing a
                # smallest-k sign flip; softmax permits a canonical payload.
                return f"""
topk_native_clean = {key} & cutlass.Int32({~index_mask})
topk_value = topk_native_clean.bitcast(cutlass.Float32).to({value_dtype})
if topk_native_clean == cutlass.Int32(2139078656):
    topk_value = cutlass.Uint16(32767).bitcast({value_dtype})
{destination} = topk_value
"""
            return f"""
topk_native_clean = {key} & cutlass.Int32({~index_mask})
topk_value = topk_native_clean.bitcast(cutlass.Float32).to({value_dtype})
if (topk_native_clean & cutlass.Int32(2147483647)) >= cutlass.Int32(2139062272):
    topk_value = {source}
{destination} = topk_value
"""
        undo_reverse = "topk_value_rank = -topk_value_rank" if not plan.largest else ""
        decode = (
            """
topk_value_bits = topk_value_rank ^ ((topk_value_rank >> cutlass.Int32(31)) & cutlass.Int32(32767))
topk_value_decodable = topk_value_rank != cutlass.Int32(32767)
"""
            if plan.rank_mode == "ordinal"
            else f"""
topk_value_sign = topk_value_rank >> cutlass.Int32(31)
topk_value_magnitude = (topk_value_rank ^ topk_value_sign) - topk_value_sign
topk_value_bits = topk_value_magnitude | ((topk_value_rank >> cutlass.Int32(16)) & cutlass.Int32(32768))
topk_value_decodable = (topk_value_magnitude != 0) & (topk_value_magnitude <= {infinity_bits})
"""
        )
        if numeric:
            # Rank 32767 decodes to NaN in both dtypes. Signed ranks merge
            # the zero signs, which is also immaterial to softmax, so neither
            # case needs the original input's bit pattern or a reload.
            return f"""
topk_value_rank = {key} >> cutlass.Int32({index_bits})
{undo_reverse}
{decode}
{destination} = cutlass.Uint16(topk_value_bits).bitcast({value_dtype})
"""
        return f"""
topk_value_rank = {key} >> cutlass.Int32({index_bits})
{undo_reverse}
{decode}
topk_value = cutlass.Uint16(topk_value_bits).bitcast({value_dtype})
if not topk_value_decodable:
    topk_value = {source}
{destination} = topk_value
"""

    if plan.fragment_graph is not None:
        from .row_topk_codegen import codegen_row_topk

        return codegen_row_topk(
            cg,
            plan,
            template=template,
            loads=loads,
            encode=encode,
            fragment_size=fragment_size,
            key_type=key_type,
            key_padding=key_padding,
            selected_key=selected_key,
            selected_index=selected_index,
            value_store=value_store,
            helper_name=helper_name,
            helper_hash=helper_hash,
            index_type=index_type,
            row_guard=row_guard,
            group_guard=group_guard,
            input_name=x,
            use_asm_encoder=use_asm_encoder,
        )

    softmax_name = f"_cute_softmax_topk_values_{helper_hash}"
    if plan.softmax:
        cg.module_statements.append(
            ast.ImportFrom(
                module="helion.runtime.cute.topk",
                names=[ast.alias(name="softmax_topk_values", asname=softmax_name)],
                level=0,
            )
        )

    def softmax_stores(
        setup: str,
        key: str,
        groups: int,
        vector: int,
        lane: str,
        guard: str,
        *,
        transposed: bool = False,
    ) -> str:
        """Keep the selected-value epilogue in the chosen output layout."""
        group_width = plan.lanes_per_row * vector
        endpoint = 0 if plan.largest else plan.k - 1
        maximum_offset = endpoint // group_width * vector + endpoint % vector
        maximum_lane = endpoint // vector % plan.lanes_per_row
        if transposed:
            span = plan.lanes_per_row // vector
            maximum_lane = maximum_lane % span * vector + maximum_lane // span
        value_load = value_store(key, "topk_softmax_value")
        output_store = f"""
        {values}[topk_row, topk_output_col] = topk_softmax_values[topk_j].to({value_dtype})
        {indices}[topk_row, topk_output_col] = topk_softmax_indices[topk_j]
"""
        if vector > 1:
            output_store = f"""
        topk_value_fragment = cute.make_rmem_tensor({vector}, {value_dtype})
        topk_index_fragment = cute.make_rmem_tensor({vector}, {output_index_dtype})
        for topk_v in cutlass.range_constexpr({vector}):
            topk_value_fragment[topk_v] = topk_softmax_values[topk_j * {vector} + topk_v].to({value_dtype})
            topk_index_fragment[topk_v] = topk_softmax_indices[topk_j * {vector} + topk_v]
        topk_output_offset = topk_row * {index_type}({plan.k}) + {index_type}(topk_output_col)
        topk_output_offset = cute.assume(topk_output_offset, divby={vector})
        topk_value_destination = cute.make_tensor(
            {values}.iterator + topk_output_offset,
            cute.make_layout({vector}, stride=1),
        )
        topk_index_destination = cute.make_tensor(
            {indices}.iterator + topk_output_offset,
            cute.make_layout({vector}, stride=1),
        )
        cute.autovec_copy(topk_value_fragment, topk_value_destination)
        cute.autovec_copy(topk_index_fragment, topk_index_destination)
"""
        return f"""
{setup}
topk_softmax_values = cute.make_rmem_tensor({groups * vector}, cutlass.Float32)
topk_softmax_values.fill(-cutlass.Float32.inf)
topk_softmax_indices = cute.make_rmem_tensor({groups * vector}, {output_index_dtype})
topk_softmax_indices.fill({output_index_dtype}(0))
for topk_j in cutlass.range_constexpr({groups}):
    topk_output_col = cutlass.Int32(topk_j * {group_width}) + {lane} * cutlass.Int32({vector})
    if topk_valid_row & ({guard}):
        for topk_v in cutlass.range_constexpr({vector}):
            topk_selected_key = {key}
            topk_selected_index = {selected_index("topk_selected_key")}
{textwrap.indent(value_load, "            ")}
            topk_softmax_values[topk_j * {vector} + topk_v] = cutlass.Float32(topk_softmax_value)
            topk_softmax_indices[topk_j * {vector} + topk_v] = {output_index_dtype}(topk_selected_index)
{softmax_name}(topk_softmax_values, {maximum_offset}, {maximum_lane}, {plan.lanes_per_row})
for topk_j in cutlass.range_constexpr({groups}):
    topk_output_col = cutlass.Int32(topk_j * {group_width}) + {lane} * cutlass.Int32({vector})
    if topk_valid_row & ({guard}):
{output_store}
"""

    scalar_value_store = value_store(
        "topk_local[topk_j]", f"{values}[topk_row, topk_output_col]"
    )
    scalar_stores = f"""
# Redistribute registers before storing so each memory instruction serves
# the entire lane subgroup instead of issuing one instruction per output rank.
topk_local = cute.make_rmem_tensor({(plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row}, cutlass.Int32)
topk_local.fill(cutlass.Int32(0))
for topk_output in cutlass.range_constexpr({plan.k}):
    if topk_lane == cutlass.Int32(topk_output % {plan.lanes_per_row}):
        topk_local[topk_output // {plan.lanes_per_row}] = {selected_key}
for topk_j in cutlass.range_constexpr({(plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row}):
    topk_output_col = cutlass.Int32(topk_j * {plan.lanes_per_row}) + topk_lane
    if topk_valid_row & ({output_col_guard}):
        topk_selected_index = {selected_index("topk_local[topk_j]")}
{textwrap.indent(scalar_value_store, "        ")}
        {indices}[topk_row, topk_output_col] = {output_index_dtype}(topk_selected_index)
"""
    if plan.softmax:
        scalar_setup = f"""
topk_local = cute.make_rmem_tensor({(plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row}, cutlass.Int32)
topk_local.fill(cutlass.Int32(0))
for topk_output in cutlass.range_constexpr({plan.k}):
    if topk_lane == cutlass.Int32(topk_output % {plan.lanes_per_row}):
        topk_local[topk_output // {plan.lanes_per_row}] = {selected_key}
"""
        scalar_stores = softmax_stores(
            scalar_setup,
            "topk_local[topk_j]",
            (plan.k + plan.lanes_per_row - 1) // plan.lanes_per_row,
            1,
            "topk_lane",
            output_col_guard,
        )
    stores = scalar_stores
    if distributed:
        distributed_output_guard = (
            "True"
            if plan.k == padded_k and plan.lanes_per_row <= padded_k
            else f"topk_output_col < cutlass.Int32({plan.k})"
        )
        distributed_key = selected_key.replace(
            "topk_selected[topk_output]", "topk_selected[topk_j]"
        )
        distributed_value_store = value_store(
            "topk_selected_key", f"{values}[topk_row, topk_output_col]"
        )
        stores = f"""
for topk_j in cutlass.range_constexpr({selected_per_lane}):
    topk_output_col = cutlass.Int32(topk_j * {output_lanes}) + topk_lane
    if topk_valid_row & ({distributed_output_guard}):
        topk_selected_key = {distributed_key}
        topk_selected_index = {selected_index("topk_selected_key")}
{textwrap.indent(distributed_value_store, "        ")}
        {indices}[topk_row, topk_output_col] = {output_index_dtype}(topk_selected_index)
"""
        if plan.softmax:
            stores = softmax_stores(
                "",
                distributed_key,
                selected_per_lane,
                1,
                "topk_lane",
                distributed_output_guard,
            )
    output_vector = plan.output_vector_width
    if distributed:
        output_vector = min(output_vector, selected_per_lane)
        # Wider vectors need complete groups. Retain the existing narrow
        # transpose when the new width would discard a usable smaller vector.
        while output_vector > output_lanes and plan.k % output_vector != 0:
            output_vector //= 2
    # The matcher proves contiguous outputs. Complete vector groups also
    # ensure every row starts at a compatible alignment; odd k stays scalar.
    if output_vector > 1 and plan.k % output_vector == 0:
        group_width = plan.lanes_per_row * output_vector
        output_groups = (
            padded_k // group_width
            if distributed
            else (plan.k + group_width - 1) // group_width
        )
        vector_output_guard = (
            "True"
            if (plan.k == padded_k if distributed else plan.k % group_width == 0)
            else f"topk_output_col < cutlass.Int32({plan.k})"
        )
        output_key = f"topk_output_keys[topk_j * {output_vector} + topk_v]"
        output_lane = "topk_lane"
        if distributed:
            wide_output = output_vector > plan.lanes_per_row
            transpose_helper = (
                "transpose_topk_output_wide" if wide_output else "transpose_topk_output"
            )
            transpose_name = f"_cute_{transpose_helper}_{helper_hash}"
            cg.module_statements.append(
                ast.ImportFrom(
                    module="helion.runtime.cute.topk",
                    names=[ast.alias(name=transpose_helper, asname=transpose_name)],
                    level=0,
                )
            )
            if wide_output:
                vector_setup = f"""
topk_output_keys = {transpose_name}(topk_selected, {output_vector}, {plan.lanes_per_row})
"""
            else:
                vector_setup = f"""
topk_output_keys = {transpose_name}(topk_selected, {output_vector})
topk_output_lane = ((topk_lane % cutlass.Int32({output_vector}))
                    * cutlass.Int32({plan.lanes_per_row // output_vector})
                    + topk_lane // cutlass.Int32({output_vector}))
"""
                output_lane = "topk_output_lane"
            output_key = selected_key.replace("topk_selected[topk_output]", output_key)
        else:
            vector_setup = f"""
topk_output_keys = cute.make_rmem_tensor({output_groups * output_vector}, cutlass.Int32)
topk_output_keys.fill(cutlass.Int32(0))
for topk_output in cutlass.range_constexpr({plan.k}):
    if topk_lane == cutlass.Int32((topk_output // {output_vector}) % {plan.lanes_per_row}):
        topk_output_keys[(topk_output // {group_width}) * {output_vector} + topk_output % {output_vector}] = {selected_key}
"""
        vector_value_store = value_store(
            "topk_selected_key", "topk_value_fragment[topk_v]"
        )
        vector_stores = f"""
{vector_setup}
for topk_j in cutlass.range_constexpr({output_groups}):
    topk_output_col = cutlass.Int32(topk_j * {group_width}) + {output_lane} * cutlass.Int32({output_vector})
    if topk_valid_row & ({vector_output_guard}):
        topk_value_fragment = cute.make_rmem_tensor({output_vector}, {value_dtype})
        topk_index_fragment = cute.make_rmem_tensor({output_vector}, {output_index_dtype})
        for topk_v in cutlass.range_constexpr({output_vector}):
            topk_selected_key = {output_key}
            topk_selected_index = {selected_index("topk_selected_key")}
{textwrap.indent(vector_value_store, "            ")}
            topk_index_fragment[topk_v] = {output_index_dtype}(topk_selected_index)
        topk_output_offset = topk_row * {index_type}({plan.k}) + {index_type}(topk_output_col)
        # Both the row stride and output column are multiples of this vector.
        topk_output_offset = cute.assume(topk_output_offset, divby={output_vector})
        topk_value_destination = cute.make_tensor(
            {values}.iterator + topk_output_offset,
            cute.make_layout({output_vector}, stride=1),
        )
        topk_index_destination = cute.make_tensor(
            {indices}.iterator + topk_output_offset,
            cute.make_layout({output_vector}, stride=1),
        )
        cute.autovec_copy(topk_value_fragment, topk_value_destination)
        cute.autovec_copy(topk_index_fragment, topk_index_destination)
"""
        if (
            plan.defer_value_gathers
            and not native_float
            and plan.rank_mode == "ordinal"
            and plan.value_mode == "decode"
            and vector_output_guard == "True"
        ):
            undo_output_reverse = (
                "topk_value_rank = -topk_value_rank" if not plan.largest else ""
            )
            # Each lane owns increasing global sorted ranks across j/v.
            # Original ordinal ranks therefore attain their maximum at the
            # first output for largest=True, the last for largest=False.
            # Use the valid output length, which can differ from padded_k/L.
            endpoint = 0 if plan.largest else output_groups * output_vector - 1
            endpoint_key = output_key.replace(
                f"topk_j * {output_vector} + topk_v", str(endpoint)
            )
            endpoint_rank = f"(({endpoint_key}) >> cutlass.Int32({index_bits}))"
            if not plan.largest:
                endpoint_rank = f"-({endpoint_rank})"
            vector_stores = f"""
{vector_setup}
if topk_valid_row:
    topk_value_fragments = cute.make_rmem_tensor(({output_vector}, {output_groups}), {value_dtype})
    topk_index_fragments = cute.make_rmem_tensor(({output_vector}, {output_groups}), {output_index_dtype})
    topk_output_max_rank = {endpoint_rank}
    for topk_j in cutlass.range_constexpr({output_groups}):
        for topk_v in cutlass.range_constexpr({output_vector}):
            topk_selected_key = {output_key}
            topk_value_rank = topk_selected_key >> cutlass.Int32({index_bits})
{textwrap.indent(undo_output_reverse, "            ")}
            topk_value_bits = topk_value_rank ^ ((topk_value_rank >> cutlass.Int32(31)) & cutlass.Int32(32767))
            topk_value_fragments[topk_v, topk_j] = cutlass.Uint16(topk_value_bits).bitcast({value_dtype})
            topk_index_fragments[topk_v, topk_j] = {output_index_dtype}({selected_index("topk_selected_key")})
    if topk_output_max_rank == cutlass.Int32(32767):
        for topk_j in cutlass.range_constexpr({output_groups}):
            for topk_v in cutlass.range_constexpr({output_vector}):
                topk_selected_key = {output_key}
                topk_value_rank = topk_selected_key >> cutlass.Int32({index_bits})
{textwrap.indent(undo_output_reverse, "                ")}
                if topk_value_rank == cutlass.Int32(32767):
                    topk_selected_index = topk_index_fragments[topk_v, topk_j]
                    topk_value_fragments[topk_v, topk_j] = {x}[topk_row, topk_selected_index]
    for topk_j in cutlass.range_constexpr({output_groups}):
        topk_output_col = cutlass.Int32(topk_j * {group_width}) + {output_lane} * cutlass.Int32({output_vector})
        topk_output_offset = topk_row * {index_type}({plan.k}) + {index_type}(topk_output_col)
        topk_output_offset = cute.assume(topk_output_offset, divby={output_vector})
        topk_value_destination = cute.make_tensor(
            {values}.iterator + topk_output_offset,
            cute.make_layout({output_vector}, stride=1),
        )
        topk_index_destination = cute.make_tensor(
            {indices}.iterator + topk_output_offset,
            cute.make_layout({output_vector}, stride=1),
        )
        cute.autovec_copy(topk_value_fragments[None, topk_j], topk_value_destination)
        cute.autovec_copy(topk_index_fragments[None, topk_j], topk_index_destination)
"""
        if plan.softmax:
            vector_stores = softmax_stores(
                vector_setup,
                output_key,
                output_groups,
                output_vector,
                output_lane,
                vector_output_guard,
                transposed=distributed and output_vector <= plan.lanes_per_row,
            )
        # Launcher schemas specialize the actual pointer alignment, including
        # shifted out-parameters. Larger vectors use multiple 128-bit stores.
        stores = (
            f"if cutlass.const_expr({values}.iterator.alignment >= {2 * output_vector} "
            f"and {indices}.iterator.alignment >= {min(16, output_index_bytes * output_vector)}):\n"
            + textwrap.indent(vector_stores, "    ")
            + "else:\n"
            + textwrap.indent(stores, "    ")
        )
    body = f"""
topk_thread = cutlass.Int32(cute.arch.thread_idx()[0])
topk_lane = topk_thread % cutlass.Int32({plan.lanes_per_row})
topk_group = topk_thread // cutlass.Int32({plan.lanes_per_row})
topk_row = ({index_type}(cute.arch.block_idx()[0])
            * {index_type}({plan.rows_per_block}) + {index_type}(topk_group))
topk_valid_row = ({row_guard}) & ({group_guard})
topk_input_bits = cute.make_tensor(
    cute.recast_ptr({x}.iterator, dtype=cutlass.Uint16), {x}.layout
)
topk_keys = cute.make_rmem_tensor({fragment_size}, {key_type})
topk_keys.fill({key_padding})
{loads}

topk_selected = {helper_name}(topk_keys, {padded_k}, {plan.lanes_per_row}, {plan.sort_network!r}, {plan.merge_schedule!r})
{stores}
"""
    statements: list[ast.AST] = []
    with cg.set_statements(statements):
        for statement in ast.parse(template.render(body)).body:
            cg.add_statement(statement)
    fn.preamble = []
    fn.body = statements
    fn.placeholder_args.update((x_name, values_name, indices_name))
    return True
