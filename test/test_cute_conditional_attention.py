"""Normal CPU binding of one unchanged authored conditional attention root."""

from __future__ import annotations

import copy
import json
import traceback

import pytest
import torch

from helion._compiler.cute.analysis.conditional_attention_access import (
    prove_conditional_attention_access,
)
from helion._compiler.cute.analysis.conditional_attention_program import (
    prove_conditional_attention_program,
)
from helion._compiler.cute.analysis.conditional_attention_recurrence import (
    _read_recurrence,
)
from helion._compiler.cute.analysis.conditional_attention_recurrence import (
    match_conditional_attention_recurrence,
)
from helion._compiler.cute.analysis.conditional_index_context import (
    ConditionalIndexContext,
)
from helion._compiler.cute.analysis.conditional_output_ownership import (
    prove_conditional_output_ownership,
)
from helion._compiler.cute.analysis.conditional_regions import prove_conditional_regions
from helion._compiler.cute.analysis.separable_index_map import prove_separable_index_map
from helion._compiler.cute.materialized_fission_codegen import _empty_graph
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_conditional_attention import conditional_attention_args
from helion._testing.cute_conditional_attention import (
    conditional_online_attention_prefix,
)
from helion._testing.cute_conditional_attention import observe_fragment_discovery
from helion._testing.cute_conditional_compile import compile_conditional_entries
from helion._testing.cute_conditional_contract import assert_conditional_page_contract
from helion.language.inline_asm_ops import inline_asm_elementwise

pytestmark = [pytest.mark.cpu_only, pytest.mark.slow]


def test_conditional_attention_prefix_contract_and_capacity(tmp_path):
    initialized = torch.cuda.is_initialized()
    result = {
        "scope": "one unchanged authored first root; no guard replacements or GPU"
    }
    final_graphs = []
    validate_final_graphs = True

    def inspect_codegen_graphs(codegen, graphs):
        if not validate_final_graphs:
            return
        view = copy.copy(codegen.host_function.device_ir)
        view.graphs = graphs
        regions = prove_conditional_regions(graphs, view.root_ids[0])
        assert regions is not None
        recurrence = _read_recurrence(view, regions)
        assert prove_conditional_attention_access(recurrence) is not None
        program = prove_conditional_attention_program(view, bound.env)
        assert program is not None
        output = prove_conditional_output_ownership(
            program, view, bound.env, codegen.device_function.config
        )
        assert output is not None
        final_graphs.append(
            {
                "root": program.recurrence.regions.root.graph_id,
                "regions": len(program.recurrence.regions.regions),
                "reads": len(program.reads),
                "writes": len(program.writes),
                "output_span": output.output_layout.item_span,
                "output_root_stride": output.output_layout.root_stride,
            }
        )

    try:
        with (
            cpu_render_target(fake_device_sm_count=148),
            observe_fragment_discovery(tmp_path, inspect_codegen_graphs) as discovery,
        ):
            inputs = conditional_attention_args()
            assert inputs[0].storage_offset() == 8
            assert inputs[3].storage_offset() == 32768
            bound = cpu_bind(conditional_online_attention_prefix, inputs)
            result["bound"] = True
            ir = bound.host_function.device_ir
            regions = prove_conditional_regions(ir.graphs, ir.root_ids[0])
            assert regions is not None
            assert len(regions.regions) == 8
            assert len(regions.branches) == 6
            assert len(regions.carries) == 3
            assert len(regions.unjoined) == 21
            result["region_certificate"] = {
                "regions": len(regions.regions),
                "branch_joins": len(regions.branches),
                "loop_carries": len(regions.carries),
                "unjoined_local_outputs": len(regions.unjoined),
            }
            assert (
                prove_conditional_regions([*ir.graphs, ir.graphs[0]], ir.root_ids[0])
                is None
            )
            branch = regions.branches[0]
            captures = list(branch.call.args[3])
            assert len(captures) > 1 and captures != list(reversed(captures))
            region_mutations = [
                (branch.phi, (branch.phi.args[1], branch.phi.args[0])),
                (
                    branch.call,
                    (
                        *branch.call.args[:3],
                        list(reversed(captures)),
                        branch.call.args[4],
                    ),
                ),
                (
                    branch.call,
                    (
                        branch.call.args[0],
                        branch.call.args[1],
                        branch.call.args[1],
                        *branch.call.args[3:],
                    ),
                ),
            ]
            for node, replacement in region_mutations:
                original = node.args
                try:
                    node.args = replacement
                    assert prove_conditional_regions(ir.graphs, ir.root_ids[0]) is None
                finally:
                    node.args = original
            assert prove_conditional_regions(ir.graphs, ir.root_ids[0]) == regions
            recurrence = match_conditional_attention_recurrence(ir, bound.env)
            assert recurrence is not None and recurrence.regions == regions
            assert recurrence.threshold == 2.0
            assert len(recurrence.stores) == 3
            result["arithmetic_certificate"] = {
                "threshold": recurrence.threshold,
                "score_factor": recurrence.score_factor,
                "store_count": len(recurrence.stores),
            }
            accesses = prove_conditional_attention_access(recurrence)
            assert accesses is not None
            assert accesses.query.metadata_storage_offset == 0
            assert accesses.key.raw.strides == (33792, 256, 128, 1)
            assert accesses.key.scale.strides == (33792, 4, 512, 1)
            assert accesses.key.scale.metadata_storage_offset == 0
            assert accesses.key.group_size == 32
            assert accesses.key.post_scale == 0 and not accesses.key.positive_zero
            assert accesses.key.safe_row is accesses.value.safe_row
            result["access_certificate"] = {
                "raw_strides": list(accesses.key.raw.strides),
                "scale_strides": list(accesses.key.scale.strides),
                "scale_metadata_offset": accesses.key.scale.metadata_storage_offset,
                "source_input_offsets": [8, 32768],
                "page_size": accesses.key.page_size,
                "depth": accesses.key.depth,
                "masked_after_native_decode": True,
            }
            raw_indices = accesses.key.raw.node.args[1]
            scale_index = accesses.key.scale.node.args[1][-1]
            assert isinstance(scale_index, torch.fx.Node)
            mask = accesses.key.value.args[0]
            access_mutations = [
                (
                    accesses.key.raw.node,
                    (
                        accesses.key.raw.host,
                        [*raw_indices[:1], raw_indices[3], *raw_indices[2:]],
                        None,
                        None,
                    ),
                ),
                (scale_index, (scale_index.args[0], 16)),
                (accesses.key.safe_row, (accesses.key.safe_row.args[0], 128)),
                (
                    accesses.key.value,
                    (mask, accesses.key.value.args[1], accesses.key.value.args[1]),
                ),
            ]
            for node, replacement in access_mutations:
                original = node.args
                try:
                    node.args = replacement
                    assert prove_conditional_attention_access(recurrence) is None
                finally:
                    node.args = original
            assert prove_conditional_attention_access(recurrence) == accesses
            with bound.env, bound.host_function:
                program = prove_conditional_attention_program(ir, bound.env)
                assert program is not None
                assert len(program.reads) == 7
                assert len(program.scalar_reads) == 1
                assert len(program.writes) == 3
                result["effect_certificate"] = {
                    "reads": len(program.reads),
                    "scalar_reads": len(program.scalar_reads),
                    "writes": len(program.writes),
                }
                write = program.writes[0]
                original = write.args
                try:
                    write.args = (accesses.query.host, *write.args[1:])
                    assert prove_conditional_attention_program(ir, bound.env) is None
                finally:
                    write.args = original
                # A user-marked pure ASM is not an exact decoder/FMA proof.
                graph = recurrence.loop.graph
                output = graph.find_nodes(op="output")[0]
                with graph.inserting_before(output):
                    extra = graph.call_function(
                        inline_asm_elementwise,
                        (
                            "mov.u32 $0, %tid.x;",
                            "=r",
                            [],
                            torch.int32,
                            True,
                            1,
                        ),
                    )
                extra.meta["val"] = torch.empty((), dtype=torch.int32)
                try:
                    assert prove_conditional_attention_program(ir, bound.env) is None
                finally:
                    graph.erase_node(extra)
                assert prove_conditional_attention_program(ir, bound.env) == program
                context = ConditionalIndexContext(
                    program, ir, bound.env, bound.config_spec.default_config()
                )
                query_map = prove_separable_index_map(
                    context, accesses.query.indices[0], accesses.query.shape[0]
                )
                assert query_map is not None
                output = program.writes[0]
                output_map = prove_separable_index_map(
                    context, output.args[1][0], output.args[0].meta["val"].numel()
                )
                assert output_map is not None
                assert [
                    (a.axis.extent, a.divisor, a.major_stride, a.minor_stride)
                    for a in query_map.axes
                ] == [(32, 8, 4608, 128), (128, 1, 1, 0)]
                assert [
                    (a.axis.extent, a.divisor, a.major_stride, a.minor_stride)
                    for a in output_map.axes
                ] == [(32, 4, 512, 1), (128, 1, 4, 0)]
                result["mapped_addresses"] = {
                    "query": str(query_map.expression),
                    "output": str(output_map.expression),
                }
                config = bound.config_spec.default_config()
                ownership = prove_conditional_output_ownership(
                    program, ir, bound.env, config
                )
                assert ownership is not None
                assert ownership.output_layout.item_span == 4096
                assert ownership.output_layout.root_stride == 8192
                assert ownership.metadata_layout.item_span == 64
                assert ownership.metadata_layout.root_stride == 128
                result["output_ownership"] = {
                    "output_item_span": ownership.output_layout.item_span,
                    "output_root_stride": ownership.output_layout.root_stride,
                    "metadata_item_span": ownership.metadata_layout.item_span,
                    "metadata_root_stride": ownership.metadata_layout.root_stride,
                }
                phase = regions.root.phase_index
                original_grid = ir.grid_block_ids[phase]
                try:
                    # An otherwise valid extra root coordinate does not occur
                    # in the output map and would duplicate its stores.
                    ir.grid_block_ids[phase] = [
                        *original_grid,
                        recurrence.loop.block_ids[0],
                    ]
                    assert (
                        prove_conditional_output_ownership(
                            program, ir, bound.env, config
                        )
                        is None
                    )
                finally:
                    ir.grid_block_ids[phase] = original_grid
                value = output.args[2]
                original_value = value.meta["val"]
                try:
                    # Bounded index maps alone do not cover extra stored
                    # values: the logical shape bridge must also be exact.
                    value.meta["val"] = torch.empty((32, 129), dtype=torch.float32)
                    assert (
                        prove_conditional_output_ownership(
                            program, ir, bound.env, config
                        )
                        is None
                    )
                finally:
                    value.meta["val"] = original_value
                channel_step = next(
                    node
                    for node in regions.root.graph.nodes
                    if node.target is torch.ops.aten.mul.Tensor and node.args[1] == 4
                )
                original = channel_step.args
                try:
                    # Address remains bounded/separable, but the four folded
                    # row residues now overlap neighboring channel outputs.
                    channel_step.args = (channel_step.args[0], 1)
                    assert (
                        prove_separable_index_map(
                            ConditionalIndexContext(program, ir, bound.env, config),
                            output.args[1][0],
                            output.args[0].meta["val"].numel(),
                        )
                        is not None
                    )
                    assert (
                        prove_conditional_output_ownership(
                            program, ir, bound.env, config
                        )
                        is None
                    )
                finally:
                    channel_step.args = original
                l_write = program.writes[2]
                original = l_write.args
                try:
                    l_write.args = (
                        l_write.args[0],
                        [*l_write.args[1][:2], 0],
                        *l_write.args[2:],
                    )
                    assert (
                        prove_conditional_output_ownership(
                            program, ir, bound.env, config
                        )
                        is None
                    )
                finally:
                    l_write.args = original
                assert (
                    prove_conditional_output_ownership(program, ir, bound.env, config)
                    == ownership
                )
                original_graphs = ir.graphs
                hole = _empty_graph(len(ir.graphs))
                try:
                    # The real phase projector preserves graph ids and fills
                    # excluded stages with exactly output(None) roots.
                    ir.graphs = [*ir.graphs, hole]
                    projected = prove_conditional_attention_program(ir, bound.env)
                    assert projected == program
                    assert (
                        prove_conditional_output_ownership(
                            projected, ir, bound.env, config
                        )
                        == ownership
                    )
                    terminal = next(iter(hole.graph.nodes))
                    with hole.graph.inserting_before(terminal):
                        unexpected = hole.graph.call_function(
                            inline_asm_elementwise,
                            ("bar.sync 0;", "", [], torch.int32, False, 1),
                        )
                    unexpected.meta["val"] = torch.empty((), dtype=torch.int32)
                    assert prove_conditional_attention_program(ir, bound.env) is None
                    hole.graph.erase_node(unexpected)
                    terminal.args = ((),)
                    assert prove_conditional_attention_program(ir, bound.env) is None
                finally:
                    ir.graphs = original_graphs
            result["contract"] = assert_conditional_page_contract(ir)
            config = bound.config_spec.default_config()
            result["config"] = config.config
            code = bound.to_code(config)
            (tmp_path / "generated.py").write_text(code)
            assert "_conditional_attention_page(" in code
            result["rendered"] = True
            assert final_graphs
            result["final_codegen_graph_certificates"] = final_graphs
            nodes = [node for info in ir.graphs for node in info.graph.nodes]
            maximum = next(
                node for node in nodes if node.target is torch.ops.aten.fmax.default
            )
            threshold = next(
                node
                for node in nodes
                if node.target is torch.ops.aten.gt.Scalar and node.args[1] == 2.0
            )
            probability_cast = next(
                node
                for node in nodes
                if node.target is torch.ops.prims.convert_element_type.default
                and node.args[1] is torch.bfloat16
            )
            vote_axis = next(
                node
                for node in nodes
                if node.target is torch.ops.aten.amax.default
                and node.meta["val"].dtype is torch.uint8
                and node.args[1] == [1]
            )
            nan_mask = next(
                node
                for node in nodes
                if node.target is torch.ops.aten.where.self
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target is torch.ops.aten.isnan.default
            )
            fma = next(
                node
                for node in nodes
                if node.target is inline_asm_elementwise
                and node.args[0] == "fma.rn.f32 $0, $1, $2, $3;"
            )
            # Mutate existing dataflow edges, reusing the one real bound graph.
            # These are semantic negatives, not extra bind/config cases.
            mutations = [
                (maximum, "target", torch.ops.aten.maximum.default),
                (threshold, "args", (threshold.args[0], 1.0)),
                (probability_cast, "args", (probability_cast.args[0], torch.float32)),
                (vote_axis, "args", (vote_axis.args[0], [0])),
                (
                    nan_mask,
                    "args",
                    (nan_mask.args[0], nan_mask.args[2], nan_mask.args[2]),
                ),
                (fma, "args", ("fma.rz.f32 $0, $1, $2, $3;", *fma.args[1:])),
            ]
            for node, attribute, value in mutations:
                original = getattr(node, attribute)
                try:
                    setattr(node, attribute, value)
                    with pytest.raises(AssertionError):
                        assert_conditional_page_contract(ir)
                    changed = match_conditional_attention_recurrence(ir, bound.env)
                    if node is threshold:
                        # The general proof retains a source threshold rather
                        # than substituting the fixture's particular value.
                        assert changed is not None and changed.threshold == 1.0
                    else:
                        assert changed is None
                finally:
                    setattr(node, attribute, original)
            # The shared denominator proof must keep both exact grouping and
            # the caller's no-kwargs arithmetic contract.
            denominator = recurrence.final_denominator
            for attribute, replacement in (
                ("args", (denominator.args[1], denominator.args[0])),
                ("kwargs", {"alpha": 2}),
            ):
                original = getattr(denominator, attribute)
                try:
                    setattr(denominator, attribute, replacement)
                    assert match_conditional_attention_recurrence(ir, bound.env) is None
                finally:
                    setattr(denominator, attribute, original)
            assert match_conditional_attention_recurrence(ir, bound.env) == recurrence
            # A changed opaque arithmetic contract cannot select the SS owner.
            # Exercise normal codegen: the independent arena may now admit
            # the ordinary owner, but the changed contract never selects SS.
            old_fma = fma.args
            validate_final_graphs = False
            try:
                fma.args = ("fma.rz.f32 $0, $1, $2, $3;", *fma.args[1:])
                fallback = bound.to_code(config)
                assert "_conditional_attention_page(" not in fallback
                assert "fragment_arena" in fallback
            finally:
                fma.args = old_fma
                validate_final_graphs = True
            threshold_args = threshold.args
            try:
                threshold.args = (threshold_args[0], float("nan"))
                assert match_conditional_attention_recurrence(ir, bound.env) is None
            finally:
                threshold.args = threshold_args
            assert match_conditional_attention_recurrence(ir, bound.env) == recurrence
            assert assert_conditional_page_contract(ir) == result["contract"]
        # Leave the render-only native-compilation prohibition before the
        # separately scoped offline compile; CUDA loading stays disabled.
        (compiled_ir,) = compile_conditional_entries(tmp_path / "generated.py", inputs)
        (tmp_path / "compiled.mlir").write_text(compiled_ir)
        result["entry_compiled_offline"] = True
        result["discovery"] = discovery
        assert discovery and all(row["result"] is True for row in discovery)
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["cuda_initialized"] = torch.cuda.is_initialized()
        (tmp_path / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print("Conditional attention CPU proof:", tmp_path)
        assert torch.cuda.is_initialized() == initialized
