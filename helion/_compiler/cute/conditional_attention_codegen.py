"""A serial SS schedule for a completely proved conditional attention root.

Only the page recurrence changes physical ownership. The generic fragment
printer retains the source prelude, loop bounds, address recipes and stores.
"""

from __future__ import annotations

import ast
import copy
import logging
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import _compat
from ...language import _tracing_ops
from ...language import memory_ops
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .analysis.conditional_attention_access import _AccessProof
from .analysis.conditional_attention_program import ConditionalAttentionProgram
from .analysis.conditional_attention_program import prove_conditional_attention_program
from .analysis.conditional_continuation_ownership import (
    prove_conditional_continuation_ownership,
)
from .analysis.conditional_output_ownership import prove_conditional_output_ownership
from .analysis.conditional_packed_staging import can_pack_native_panel
from .computed_fragment import FragmentCompiler
from .fragment_context import HostTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .fragment_context import Fragment


log = logging.getLogger(__name__)


def conditional_attention_compiler(
    cg: GenerateAST, graphs: list[GraphInfo]
) -> FragmentCompiler | None:
    """Recheck actual projected graphs; incomplete proofs keep the old owner."""
    env = CompileEnvironment.current()
    if cg.device_function.config.pid_type != "flat":
        # Each tile acquires TMEM and relinquishes the allocation permit.
        # A persistent CTA cannot repeat that lifetime for a second tile.
        log.debug("Conditional attention fallback: TMEM lifetime requires a flat pid")
        return None
    if _compat.target_device_capability(env.device) != (10, 0):
        log.debug(
            "Conditional attention fallback: the dedicated schedule requires SM100"
        )
        return None
    if cg.device_function.config.get("cute_fragment_threads", 128) != 128:
        log.debug(
            "Conditional attention fallback: the dedicated schedule requires 128 threads"
        )
        return None
    ir = copy.copy(cg.host_function.device_ir)
    ir.graphs = graphs
    program = prove_conditional_attention_program(ir, env)
    if program is None:
        log.debug(
            "Conditional attention fallback: source, recurrence, or memory-effect proof did not match"
        )
        return None
    ownership_proof = (
        prove_conditional_continuation_ownership
        if program.recurrence.continuation is not None
        else prove_conditional_output_ownership
    )
    if ownership_proof(program, ir, env, cg.device_function.config) is None:
        log.debug(
            "Conditional attention fallback: output or continuation ownership proof did not match"
        )
        return None
    recurrence = program.recurrence
    rows, depth = recurrence.pv.meta["val"].shape
    if (
        rows not in (8, 16, 32, 40, 64)
        or depth != 128
        or program.accesses.key.page_size != 128
        or len(recurrence.loop.block_ids) != 1
        or cg.device_function.resolved_block_size(recurrence.loop.block_ids[0]) != 1
    ):
        log.debug(
            "Conditional attention fallback: unsupported panel geometry or loop block "
            "(rows=%s, depth=%s, page_size=%s, loop_blocks=%s)",
            rows,
            depth,
            program.accesses.key.page_size,
            recurrence.loop.block_ids,
        )
        return None
    calls = [
        node
        for info in graphs
        for node in info.graph.nodes
        if _tracing_ops.is_for_loop_target(node.target)
    ]
    if len(calls) != 1 or calls[0].target is not _tracing_ops._for_loop:
        log.debug(
            "Conditional attention fallback: expected exactly one serial page loop"
        )
        return None
    return ConditionalAttentionCompiler(cg, graphs, program)


class ConditionalAttentionCompiler(FragmentCompiler):
    """Preserve all generic effects outside the certified page loop."""

    def __init__(
        self,
        cg: GenerateAST,
        graphs: list[GraphInfo],
        program: ConditionalAttentionProgram,
    ) -> None:
        super().__init__(cg, graphs)
        self.program = program
        self.panel_buffers: list[tuple[str, torch.dtype, tuple[int, ...], int]] = []
        accesses = program.accesses
        self.packed_staging = all(
            can_pack_native_panel(panel, self.env)
            for panel in (accesses.key, accesses.value)
        )
        self.direct_reads = frozenset(
            (
                accesses.key.raw.node,
                accesses.key.scale.node,
                accesses.value.raw.node,
                accesses.value.scale.node,
            )
        )

    def panel(self, dtype: torch.dtype, shape: tuple[int, ...], align: int) -> str:
        name = self.df.new_var("conditional_panel")
        self.panel_buffers.append((name, dtype, shape, align))
        # Generic allocation precedes these panels; charge every alignment
        # boundary conservatively and the helper's static shared reservation.
        self.smem_bytes += math.prod(shape) * dtype.itemsize + align - 1
        return name

    def allocations(self) -> list[ast.AST]:
        return [
            *super().allocations(),
            *[
                statement_from_string(
                    f"{name} = {self.allocator}.allocate_tensor("
                    f"{self.dtype(dtype)}, cute.make_layout({shape!r}), "
                    f"byte_alignment={align})"
                )
                for name, dtype, shape, align in self.panel_buffers
            ],
        ]

    def emit_mma_preamble(self) -> None:
        super().emit_mma_preamble()
        self.cg.module_statements.extend(
            ast.parse(
                "from helion._compiler.cute._conditional_attention_page import "
                "conditional_attention_page as _conditional_attention_page, "
                "conditional_attention_begin as _conditional_attention_begin, "
                "conditional_attention_end as _conditional_attention_end"
            ).body
        )
        if self.packed_staging:
            self.cg.module_statements.extend(
                ast.parse(
                    "from helion._compiler.cute._conditional_attention_staging import "
                    "stage_native_ss_pairs as _conditional_attention_stage_packed"
                ).body
            )

    def node(
        self,
        node: Node,
        values: dict[Node, object],
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
    ) -> object:
        if node.target is memory_ops.load and node in self.direct_reads:
            return self.memory(node, values, False, defer_load=True)
        return super().node(node, values, deferred_updates)

    def constexpr(self, name: str, count: int, emit: Callable[[str], None]) -> None:
        loop = statement_from_string(
            f"for {name} in cutlass.range_constexpr({count}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            emit(name)
        self.cg.add_statement(loop)

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        recurrence = self.program.recurrence
        assert node.args[0] == recurrence.loop.graph_id
        graph = recurrence.loop
        width = cast("int", recurrence.pv.meta["val"].shape[0])
        sources = cast("list[Node]", node.args[3])
        incoming = [values[source] for source in sources]
        inner = dict(
            zip(graph.graph.find_nodes(op="placeholder"), incoming, strict=True)
        )
        # Access facts resolve capture aliases back to their defining node.
        # Reuse the already evaluated captured value under that same key;
        # recursively emitting the outer node here would move its loads or
        # branch-join evaluation into the page loop.
        resolve = _AccessProof(recurrence).resolve
        for source, captured in zip(sources, incoming, strict=True):
            inner.setdefault(resolve(source), captured)
        initial = [
            cast("Fragment", inner[item])
            for item in (
                recurrence.maximum_input,
                recurrence.accumulator_input,
                recurrence.denominator_input,
            )
        ]
        self.held.append(incoming)
        carries: list[Fragment] = []
        self.held.append(carries)
        for value in initial:
            carries.append(self.allocate(value))
        maximum = carries[0].storage
        assert maximum is not None
        key = self.panel(torch.bfloat16, (16384,), 1024)
        value = self.panel(torch.bfloat16, (16384,), 1024)
        query = self.panel(torch.bfloat16, (128 * width,), 1024)
        probability = self.panel(torch.bfloat16, (128 * width,), 128)
        alpha = self.panel(torch.float32, (width,), 16)
        # The helper indexes scratch[warp, query], with explicit row-major layout.
        scratch = self.panel(torch.float32, (4 * width,), 16)
        scratch_view = self.df.new_var("conditional_scratch")
        self.emit(
            f"{scratch_view} = cute.make_tensor({scratch}.iterator, cute.make_layout((4, {width}), stride=({width}, 1)))"
        )
        self.smem_bytes += 2048
        partials, output, visible = [
            self.df.new_var(name)
            for name in (
                "conditional_partials",
                "conditional_output",
                "conditional_visible",
            )
        ]
        for name, dtype in (
            (partials, "Float32"),
            (output, "Float32"),
            (visible, "Boolean"),
        ):
            self.emit(f"{name} = cute.make_rmem_tensor(({width},), cutlass.{dtype})")

        def evaluate(target: Node) -> object:
            if target not in inner:
                for source in target.all_input_nodes:
                    evaluate(source)
                inner[target] = self.node(target, inner)
            return inner[target]

        # Query is an unchanged captured fragment. Its source address map is
        # already proven injective; physical SS swizzling is confined to stores.
        q = cast("Fragment", evaluate(cast("Node", recurrence.qk.args[0])))
        self.scopes.append(inner)
        row = self.df.new_var("conditional_query")

        def initialize(i: str) -> None:
            address = f"({i}) * 64 + (({self.thread} % 64) ^ (({i} % 8) * 8)) + {64 * width} * ({self.thread} // 64)"
            self.emit(f"{query}[{address}] = {q.read((i, self.thread))}")
            self.emit(f"{partials}[{i}] = {initial[2].read((i, self.thread))}")
            self.emit(f"{output}[{i}] = {initial[1].read((i, self.thread))}")

        self.constexpr(row, width, initialize)
        self.emit(
            f"if {self.thread} < {width}:\n    {maximum}[{self.thread}] = {initial[0].read((self.thread,))}"
        )
        self.synchronize()
        bid = graph.block_ids[0]
        old_offsets, old_bounds = dict(self.offsets), dict(self.bounds)
        begin = cast("list[object]", node.args[1])[0]
        end = cast("list[object]", node.args[2])[0]
        begin_text = self.scalar(values[begin] if isinstance(begin, Node) else begin)
        end_text = self.scalar(values[end] if isinstance(end, Node) else end)
        index = self.df.new_var("conditional_page")
        base = self.df.new_var("conditional_tmem")
        done = self.df.new_var("conditional_done")
        rescale_previous = f"{index} > 0"
        if recurrence.continuation is not None:
            # The certificate equates this typed symbolic value to the outer
            # seed branch. Use the ordinary symbolic lowering, just as the
            # source _get_symnode does; its string argument is only a label.
            seeded = self.sym(recurrence.continuation.rescale_seed.meta["val"])
            rescale_previous = f"({rescale_previous}) | ({seeded})"
        self.offsets[bid], self.bounds[bid] = index, end_text
        body: list[ast.AST] = []
        previous_opaque = self.lazy_opaque_producers
        self.lazy_opaque_producers = (
            self.program.accesses.key.decode.nodes
            | self.program.accesses.value.decode.nodes
        )
        with self.cg.set_statements(body):
            if self.packed_staging:
                for panel, destination in (
                    (self.program.accesses.key, key),
                    (self.program.accesses.value, value),
                ):
                    raw = evaluate(panel.raw.host)
                    scales = evaluate(panel.scale.host)
                    assert isinstance(raw, HostTensor)
                    assert isinstance(scales, HostTensor)
                    coordinates = [
                        self.scalar(evaluate(coordinate))
                        for coordinate in (
                            panel.physical_page,
                            panel.head,
                            panel.row_shift,
                        )
                    ]
                    self.emit(
                        f"_conditional_attention_stage_packed({raw.name}, {scales.name}, "
                        f"{destination}, {', '.join(coordinates)}, {self.thread}, "
                        f"{panel.raw.shape!r}, {panel.raw.strides!r}, {panel.scale.strides!r}, "
                        f"{self.dtype(panel.shifted_row.meta['val'].dtype)})"
                    )
            else:
                k = cast("Fragment", evaluate(self.program.accesses.key.value))
                v = cast("Fragment", evaluate(self.program.accesses.value.value))
                channel = self.df.new_var("conditional_channel")

                def stage(c: str) -> None:
                    address = f"{self.thread} * 64 + (({c} % 64) ^ (({self.thread} % 8) * 8)) + 8192 * ({c} // 64)"
                    self.emit(f"{key}[{address}] = {k.read((self.thread, c))}")
                    self.emit(f"{value}[{address}] = {v.read((self.thread, c))}")

                self.constexpr(channel, 128, stage)
            mask = cast("Fragment", evaluate(recurrence.visible))

            def stage_mask(i: str) -> None:
                self.emit(f"{visible}[{i}] = {mask.read((i, self.thread))}")

            self.constexpr(self.df.new_var("conditional_query"), width, stage_mask)
            steps = (
                tuple(32 * (i % 4) + 16384 * (i // 4) for i in range(8)),
                tuple(2048 * i for i in range(8)),
                tuple(32 * (i % 4) + 128 * width * (i // 4) for i in range(8)),
                tuple(256 * i for i in range(8)),
            )
            self.emit(
                f"_conditional_attention_page({key}, {value}, {query}, {probability}, "
                f"{maximum}, {alpha}, {scratch_view}, {partials}, {output}, {visible}, "
                f"{self.thread}, {rescale_previous}, cutlass.Float32({recurrence.score_factor!r}), "
                f"cutlass.Float32({recurrence.threshold!r}), {width}, "
                + ", ".join(repr(item) for item in steps)
                + f", {self.env.settings.fast_math!r}, {base}, {done})"
            )
        self.lazy_opaque_producers = previous_opaque
        loop = statement_from_string(
            f"for {index} in range(cutlass.Int32({begin_text}), cutlass.Int32({end_text}), 1):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body = cast("list[ast.stmt]", body)
        # The proved scalar page bounds are uniform across this complete CTA.
        # Keep both acquisition and retirement inside the same nonempty guard;
        # generic initialization/publication still executes on zero-trip paths.
        lifetime = statement_from_string(
            f"if cutlass.Int32({begin_text}) < cutlass.Int32({end_text}):\n    pass"
        )
        assert isinstance(lifetime, ast.If)
        lifetime.body = [
            cast(
                "ast.stmt",
                statement_from_string(
                    f"{base}, {done} = _conditional_attention_begin({output}, {self.thread}, {width})"
                ),
            ),
            loop,
            cast(
                "ast.stmt",
                statement_from_string(
                    f"_conditional_attention_end({base}, {done}, {output}, {self.thread}, {width})"
                ),
            ),
        ]
        self.cg.add_statement(lifetime)
        self.offsets, self.bounds = old_offsets, old_bounds

        def publish(i: str) -> None:
            self.emit(f"{carries[1].read((i, self.thread))} = {output}[{i}]")
            self.emit(f"{carries[2].read((i, self.thread))} = {partials}[{i}]")

        self.constexpr(self.df.new_var("conditional_query"), width, publish)
        self.synchronize()
        self.scopes.pop()
        self.held.pop()
        self.held.pop()
        return carries
