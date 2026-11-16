"""Admission of the unchanged seeded/whole continuation page recurrence."""

from __future__ import annotations

import ast
import copy
import json
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.analysis.conditional_attention_program import (
    prove_conditional_attention_program,
)
from helion._compiler.cute.analysis.conditional_attention_recurrence import (
    match_conditional_attention_recurrence,
)
from helion._compiler.cute.analysis.conditional_continuation import (
    prove_conditional_continuation,
)
from helion._compiler.cute.analysis.conditional_continuation_ownership import (
    prove_conditional_continuation_ownership,
)
from helion._compiler.cute.conditional_attention_codegen import (
    conditional_attention_compiler,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_conditional_phase_binding import phase_fixture
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = pytest.mark.cpu_only


@forbid_cuda_work()
def test_actual_seeded_continuation_envelope(tmp_path):
    observations = []

    def observe(cg, graphs):
        result = conditional_attention_compiler(cg, graphs)
        ir = copy.copy(cg.host_function.device_ir)
        ir.graphs = graphs
        if ir.root_ids == [bound.host_function.device_ir.root_ids[1]]:
            proof = prove_conditional_continuation(ir, bound.env)
            assert proof is not None
            assert len(proof.imported_reads) == 3
            recurrence = match_conditional_attention_recurrence(ir, bound.env)
            assert recurrence is not None
            assert recurrence.continuation == proof
            program = prove_conditional_attention_program(ir, bound.env)
            assert program is not None
            assert program.imported_reads == proof.imported_reads
            ownership = prove_conditional_continuation_ownership(
                program, ir, bound.env, cg.device_function.config
            )
            assert ownership is not None
            nodes = [n for info in graphs for n in info.graph.nodes]
            nonempty = next(
                n
                for n in nodes
                if n.target is torch.ops.aten.ne.Scalar
                and n.args[0] is proof.imported_reads[0]
            )
            lane_zero = next(
                n
                for n in nodes
                if n.target is torch.ops.aten.eq.Scalar and n.graph is nonempty.graph
            )
            scale = next(
                n
                for n in nodes
                if n.target is torch.ops.aten.mul.Tensor
                and len(n.args) == 2
                and type(n.args[1]) is int
                and n.args[1] == 64
            )
            initial_o = next(
                b.when_false
                for b in proof.regions.branches
                if b.phi is proof.initial_states[1]
            )
            lse_scale = proof.lse_store.args[2].args[2]
            imported_l = proof.imported_reads[1]
            changes = [
                (nonempty, (nonempty.args[0], 0.0)),
                (lane_zero, (lane_zero.args[0], 1)),
                (
                    proof.rescale_branch.args[0],
                    (
                        proof.rescale_branch.args[0].args[0],
                        proof.rescale_branch.args[0].args[0],
                    ),
                ),
                (scale, (scale.args[0], 32)),
                (
                    proof.denominator,
                    (proof.denominator.args[1], proof.denominator.args[0]),
                ),
                (initial_o, (initial_o.args[0], -0.0, *initial_o.args[2:])),
                (lse_scale, (lse_scale.args[0], 1.0)),
                (
                    imported_l,
                    (imported_l.args[0], [*imported_l.args[1][:2], 0], None, None),
                ),
            ]
            for changed, replacement in changes:
                saved = changed.args
                try:
                    changed.args = replacement
                    assert prove_conditional_continuation(ir, bound.env) is None, (
                        changed.name
                    )
                    assert (
                        match_conditional_attention_recurrence(ir, bound.env) is None
                    ), changed.name
                finally:
                    changed.args = saved
            # A named arithmetic modifier must not bypass the continuation's
            # existing no-kwargs parser through the shared denominator walk.
            saved_kwargs = proof.denominator.kwargs
            try:
                proof.denominator.kwargs = {"alpha": 2}
                assert prove_conditional_continuation(ir, bound.env) is None
                assert match_conditional_attention_recurrence(ir, bound.env) is None
            finally:
                proof.denominator.kwargs = saved_kwargs
            assert prove_conditional_continuation(ir, bound.env) == proof
            # The call argument is a debug label; the lowering consumes the
            # typed SymBool. Neither a matching label nor a mismatching label
            # can override that semantic value.
            saved_args = proof.rescale_seed.args
            saved_value = proof.rescale_seed.meta["val"]
            try:
                proof.rescale_seed.args = ("different debug label",)
                assert prove_conditional_continuation(ir, bound.env) == proof
                proof.rescale_seed.meta["val"] = False
                assert prove_conditional_continuation(ir, bound.env) is None
            finally:
                proof.rescale_seed.args = saved_args
                proof.rescale_seed.meta["val"] = saved_value
            # Ownership uses every task/query/head/channel coordinate. Dropping
            # any output coordinate or indexing outside a source dimension must
            # refuse without changing the already-proved numerical envelope.
            index_changes = []
            for position in range(4):
                indices = list(proof.output_store.args[1])
                indices[position] = 0
                index_changes.append((proof.output_store, indices))
            index_changes.extend(
                (
                    (proof.lse_store, [0]),
                    (
                        proof.imported_reads[1],
                        [99999, *proof.imported_reads[1].args[1][1:]],
                    ),
                    (proof.imported_reads[2], [0]),
                )
            )
            for node, indices in index_changes:
                saved = node.args
                try:
                    node.args = (saved[0], indices, *saved[2:])
                    assert (
                        prove_conditional_continuation_ownership(
                            program, ir, bound.env, cg.device_function.config
                        )
                        is None
                    )
                finally:
                    node.args = saved
            saved = proof.output_store.args
            try:
                proof.output_store.args = (proof.imported_reads[2].args[0], *saved[1:])
                assert prove_conditional_attention_program(ir, bound.env) is None
            finally:
                proof.output_store.args = saved
            assert (
                prove_conditional_continuation_ownership(
                    program, ir, bound.env, cg.device_function.config
                )
                == ownership
            )
            assert prove_conditional_continuation(ir, bound.env) == proof
            observations.append((ir, result))
            for info in graphs:
                if not [
                    node for node in info.graph.nodes if node.op == "call_function"
                ]:
                    continue
                (tmp_path / f"graph-{info.graph_id}.txt").write_text(str(info.graph))
            (tmp_path / "metadata.json").write_text(
                json.dumps(
                    [
                        {
                            "graph": info.graph_id,
                            "kind": type(info).__name__,
                            "fields": {k: repr(v) for k, v in info.kwargs().items()},
                            "nodes": [
                                {
                                    "name": n.name,
                                    "target": str(n.target),
                                    "value": repr(n.meta.get("val")),
                                    "users": [u.name for u in n.users],
                                }
                                for n in info.graph.nodes
                            ],
                        }
                        for info in graphs
                    ],
                    indent=2,
                )
            )
        return result

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(written_conditional_attention, phase_fixture())
        with patch(
            "helion._compiler.cute.conditional_attention_codegen.conditional_attention_compiler",
            side_effect=observe,
        ):
            source = bound.to_code(bound.config_spec.default_config())
    (tmp_path / "generated.py").write_text(source)
    stages = stage_sources(source)
    assert len(stages) == 2
    for index, stage in enumerate(stages):
        calls = [
            node
            for node in ast.walk(ast.parse(stage))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_conditional_attention_page"
        ]
        assert len(calls) == 1
        rescale = calls[0].args[11]
        if index == 0:
            assert isinstance(rescale, ast.Compare)
        else:
            assert isinstance(rescale, ast.BinOp) and isinstance(rescale.op, ast.BitOr)
            assert isinstance(rescale.left, ast.Compare)
            assert isinstance(rescale.right, ast.Compare)
            assert isinstance(rescale.right.ops[0], ast.NotEq)
    assert len(observations) == 1
    assert observations[0][1] is not None
