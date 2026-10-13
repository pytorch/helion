"""Original joined root stages with typed native operands and FP32 publication."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

from .chained_root_native import KEY
from .chained_root_native import native_facts
from .chained_warp_stage import _snapshot

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_completed_image_emission import CompletedImageEmission
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_matmul import _StagedInput
    from .chained_root_native import RootNativeStage
    from .chained_root_typed_publication import RootTypedPublication
    from .chained_root_warp_stage import WarpOperandEmission
    from .chained_scratch_layout import ScratchLayouts


@dataclass(frozen=True)
class RootNativeCompletion:
    body: RootNativeEmission
    stage: RootNativeStage
    operands: WarpOperandEmission
    owner: str
    allocation: tuple[str, ...]
    input_prefix: tuple[str, ...]
    phase: tuple[str, ...]
    publication: tuple[str, ...]
    selection: object = None

    def facts(self) -> object:
        return (
            id(self.body),
            id(self.stage),
            self.operands.facts(),
            self.owner,
            self.allocation,
            self.input_prefix,
            self.phase,
            self.publication,
        )

    def check(
        self,
        plan: ChainedMatmulPlan,
        node: Node,
        owner: str,
        phase: tuple[str, ...],
        publication: tuple[str, ...] | None = None,
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        self.body.check()
        if (
            self.selection != self.facts()
            or plan is not self.body.plan
            or node is not self.stage.node
            or owner != self.owner
            or owner != f"chain_{self.stage.index}_c"
            or self.body.completed.get(self.stage.index) is not self
            or phase != self.phase
            or not phase
            or phase[-1] != self.body.execution.sync
            or not self.operands.matches()
            or publication is not None
            and publication != self.publication
            or self.phase != self.body.phases[self.stage.index]
        ):
            raise _UnsupportedChain("native root completion changed")
        for target, layout in zip(
            self.operands.native_destinations, self.operands.layouts, strict=True
        ):
            target.check(
                self.body.cg,
                plan,
                layout.node,
                f"chain_{self.stage.index}",
                layout.role,
                layout.shape,
                layout.stride,
                layout.inner,
                self.body.dtype(self.stage),
            )


class RootNativeEmission:
    """One real root allocation, original ordered stages, then joined release."""

    def __init__(self, cg: GenerateAST, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain

        native = plan.root_native
        if native is None:
            raise _UnsupportedChain("native root body has no schedule")
        native.check(plan)
        self.cg = cg
        self.plan = plan
        self.native = native
        self.execution = native.execution
        self.selection = native_facts(plan)
        self.config = _snapshot(cg.device_function.config.config)
        self.aliases = dict(plan.tensor_aliases)
        self.cursor = 0
        self.prefix: tuple[str, ...] = ()
        self.resources: tuple[str, ...] = ()
        self.completed: dict[int, RootNativeCompletion] = {}
        self.typed_publications: dict[Node, RootTypedPublication] = {}
        self.phases: dict[int, tuple[str, ...]] = {}
        self.finished = False
        self.check()

    def check(self) -> None:
        from .chained_matmul import _UnsupportedChain

        self.native.check(self.plan)
        if (
            self.cg.device_function.cute_state.chained_matmul_plan is not self.plan
            or self.cg.current_root_graph_info is not self.native.root
            or any(
                self.plan.tensor_aliases.get(k) != v for k, v in self.aliases.items()
            )
            or self.selection != native_facts(self.plan)
            or self.config != _snapshot(self.cg.device_function.config.config)
            or self.cg.device_function.config.config.get(KEY) is not True
            or self.execution != self.native.execution
        ):
            raise _UnsupportedChain("native root emission context changed")
        self.aliases = dict(self.plan.tensor_aliases)

    def dtype(self, stage: RootNativeStage) -> str:
        from ..compile_environment import CompileEnvironment

        return CompileEnvironment.current().backend.dtype_str(stage.dtype)

    def allocate(self, lines: list[str]) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_tcgen_stage import allocate_tmem_resources

        self.check()
        if self.resources or self.cursor or self.finished:
            raise _UnsupportedChain("duplicate root native allocation")
        self.resources = tuple(
            allocate_tmem_resources(
                self.native.tmem_columns, self.native.barrier_count, self.plan.threads
            )
        )
        lines.extend(self.resources)
        self.prefix = tuple(lines)

    def before(self, stage: int, lines: list[str]) -> RootNativeStage | None:
        from .chained_matmul import _UnsupportedChain

        self.check()
        if (
            self.finished
            or not self.resources
            or stage != self.cursor
            or tuple(lines[: len(self.prefix)]) != self.prefix
        ):
            raise _UnsupportedChain("native root stage or prior completion changed")
        self.cursor += 1
        return self.native.selected(self.plan, stage)

    def emit(
        self,
        stage: RootNativeStage,
        boundaries: dict[Node, str],
        scan_inputs: list[_ScanInput],
        staged_inputs: list[_StagedInput],
        lines: list[str],
        scratch: ScratchLayouts,
        padding: int,
        output_padding: int,
        images: CompletedImageEmission | None,
    ) -> RootNativeCompletion:
        from .chained_matmul import _UnsupportedChain
        from .chained_native_destination import native_destinations
        from .chained_root_typed_publication import complete_typed_publication
        from .chained_root_typed_publication import prepare_typed_publication
        from .chained_root_warp_stage import emit_original_warp_operands
        from .chained_tcgen05 import _load_result
        from .chained_tcgen_stage import _native_accumulator_setup
        from .chained_tcgen_stage import _native_mma_definition
        from .chained_tcgen_stage import _publish_result
        from .chained_tcgen_stage import emit_full_k_issue

        self.check()
        if (
            stage.index != self.cursor - 1
            or stage.index in self.completed
            or self.native.selected(self.plan, stage.index) is not stage
            or stage.node in boundaries
            or images is not None
            and stage.node in images.sources
        ):
            raise _UnsupportedChain("native stage conflicts with original result owner")
        start = len(lines)
        prefix = f"chain_{stage.index}"
        dtype = self.dtype(stage)
        operands = emit_original_warp_operands(
            self.cg,
            self.plan,
            boundaries,
            scan_inputs,
            {},
            staged_inputs,
            stage.index,
            "cp_async",
            padding,
            dtype,
            native_destinations=native_destinations(self.cg, self.plan, stage, padding),
        )
        if not operands.matches() or len(operands.native_destinations) != 2:
            raise _UnsupportedChain("native stage lost original operand program")
        lines.extend(operands.lines)
        if operands.asynchronous:
            lines.extend(
                [
                    "cute.arch.cp_async_commit_group()",
                    "cute.arch.cp_async_wait_group(0)",
                ]
            )
        lines.extend(["cute.arch.fence_view_async_shared()", self.execution.sync])
        input_prefix = tuple(lines)
        if images is not None:
            images.joined_inputs(stage.index, lines)
        m, n, _k = stage.geometry.physical
        owner = f"{prefix}_c"
        allocation = (
            f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {m * (n + output_padding)}, alignment=128)",
            f"{owner} = cute.make_tensor({prefix}_c_ptr, {scratch.layout(owner, (m, n), row_stride=n + output_padding)})",
        )
        lines.extend(allocation)
        lines.append(_native_mma_definition(prefix, dtype, m, n, "K", "K", "SMEM"))
        lines.extend(_native_accumulator_setup(prefix, m, n, self.execution.tmem))
        lines.append(
            f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_slice.partition_A({prefix}_a))"
        )
        lines.extend(
            emit_full_k_issue(
                prefix,
                stage.index,
                "0",
                f"{prefix}_ra[None, None, {prefix}_kk]",
                False,
                self.execution,
            )
        )
        lines.extend(_load_result(prefix, (m, n), execution=self.execution))
        typed = prepare_typed_publication(self, stage, boundaries, scan_inputs, lines)
        if typed is not None:
            lines.extend(typed.allocation)
            lines.extend(typed.publication)
        publication = tuple(
            _publish_result(
                self.plan, boundaries, prefix, ((stage.index, stage.geometry, 0),)
            )
        )
        lines.extend(publication)
        lines.append(self.execution.sync)
        phase = tuple(lines[start:])
        self.phases[stage.index] = phase
        receipt = RootNativeCompletion(
            self, stage, operands, owner, allocation, input_prefix, phase, publication
        )
        receipt = replace(receipt, selection=receipt.facts())
        self.completed[stage.index] = receipt
        self.prefix = tuple(lines)
        receipt.check(self.plan, stage.node, owner, phase, publication)
        if typed is not None:
            complete_typed_publication(typed, receipt, boundaries)
        return receipt

    def finish(self, lines: list[str]) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_tcgen_stage import free_stages

        self.check()
        if (
            self.finished
            or self.cursor != len(self.plan.dots)
            or set(self.completed) != {item.index for item in self.native.stages}
            or tuple(lines[: len(self.prefix)]) != self.prefix
        ):
            raise _UnsupportedChain("native root lifetime is incomplete")
        for receipt in self.completed.values():
            receipt.check(self.plan, receipt.stage.node, receipt.owner, receipt.phase)
        for typed in self.typed_publications.values():
            typed.check()
        lines.extend(free_stages())
        self.finished = True
