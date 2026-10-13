"""Completed matrix images on the original joined warp-root schedule.

Original additive publication supplies every native/domain and join receipt.
A complete exclusive-use proof can then retire redundant FP32 owners through
exact typed BF16 publication spans. Unsupported readers retain the additive
body; exports always use the original view/index and store-domain lowering.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING

from .chained_completed_image import _plan_facts
from .chained_completed_image import discover_completed_images
from .chained_root_native import joined_root
from .chained_store_expression import _node_facts
from .chained_warp_stage import _snapshot

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_completed_image import CompletedImage
    from .chained_completed_image import CompletedOperandView
    from .chained_direct_image import DirectImageLowering
    from .chained_matmul import ChainedMatmulPlan
    from .chained_root_native_emission import RootNativeCompletion
    from .chained_root_warp_stage import WarpOperandEmission
    from .chained_root_warp_stage import WarpOperandLayout
    from .chained_row_parallel import RowParallelEmission
    from .chained_row_register_emission import RowRegisterEmission
    from .chained_warp_pair import WarpPairAttempt


def image_shared_bytes(plan: ChainedMatmulPlan) -> int:
    """Additive storage; no existing owner/arena capacity rule is widened."""
    if (
        not joined_root(plan)
        or plan.loop is not None
        or plan.warp_workspace is not None
        or plan.warp_result_transport is not None
        or plan.preparation_pipeline is not None
        or plan.root_typed_operands is not None
        or plan.k_schedule is not None
    ):
        return 0
    images = discover_completed_images(plan)
    if any(
        publication.source.ready_before_stage >= len(plan.dots)
        for image in images
        for publication in image.publications
    ):
        # There must already be a later operand join for every image write.
        return 0
    return sum((image.allocated_bytes + 127) // 128 * 128 for image in images)


class CompletedImageEmission:
    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        *,
        warp_pairs: tuple[WarpPairAttempt, ...] = (),
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        self.cg = cg
        self.plan = plan
        self.warp_pairs = warp_pairs
        for pair in warp_pairs:
            pair.check()
        self.images = discover_completed_images(plan)
        if not self.images or image_shared_bytes(plan) != plan.completed_image_bytes:
            raise _UnsupportedChain("completed image storage proposal changed")
        for image in self.images:
            image.check(plan)
        self.config = _snapshot(cg.device_function.config.config)
        self.aliases = dict(plan.tensor_aliases)
        self.names = tuple(
            cg.device_function.new_var("chain_completed_image") for _ in self.images
        )
        self.pointer_names = tuple(
            cg.device_function.new_var("chain_completed_pointer") for _ in self.images
        )
        self.index_names = tuple(
            cg.device_function.new_var("chain_completed_index") for _ in self.images
        )
        self.step_names = tuple(
            cg.device_function.new_var("chain_completed_step") for _ in self.images
        )
        self.sources = {
            item.source.node: item.source
            for image in self.images
            for item in image.publications
        }
        self.completed: dict[Node, str] = {}
        self.published: set[tuple[int, Node]] = set()
        self.visible: set[tuple[int, Node]] = set()
        self.stage = -1
        self.input_join = -1
        self.prefix: tuple[str, ...] = ()
        self.used: set[tuple[int, int]] = set()
        self.rejected: list[tuple[int, str, str]] = []
        self.finished = False
        from .chained_direct_image import plan_direct_images

        self.direct = plan_direct_images(cg, plan, self.images, warp_pairs=warp_pairs)
        self._direct_selection = self.direct
        self._direct_facts = self.direct.facts(self.images)
        self.allocation_lines: tuple[str, ...] = ()
        self.publication_lines: dict[tuple[int, Node], tuple[str, ...]] = {}
        self.native_stages: list[tuple[Node, tuple[str, ...]]] = []
        self.warp_operands: dict[Node, WarpOperandEmission] = {}
        self.native_completions: dict[Node, RootNativeCompletion] = {}
        self.lowering: DirectImageLowering | None = None

    def _check(self, lines: list[str], *, allow_finished: bool = False) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            (self.finished and not allow_finished)
            or self.direct != self._direct_selection
            or self.direct.facts(self.images) != self._direct_facts
            or self.config != _snapshot(self.cg.device_function.config.config)
            or tuple(lines[: len(self.prefix)]) != self.prefix
            or any(
                self.plan.tensor_aliases.get(k) != v for k, v in self.aliases.items()
            )
        ):
            raise _UnsupportedChain("completed image context or emitted prefix changed")
        self.aliases = dict(self.plan.tensor_aliases)
        for pair in self.warp_pairs:
            pair.check()
        # Host tensor aliases may be added by the original expression lowering.
        # They cannot rebind earlier owners; no other proposal fact may change.
        for image in self.images:
            if (
                image._selection != image.selection()
                or _plan_facts(self.plan)[:-1] != image.plan_facts[:-1]
                or self.plan.region is None
                or image.facts != _node_facts(tuple(self.plan.region.graph.nodes))
            ):
                raise _UnsupportedChain("completed image graph or plan changed")

    def allocate(self, lines: list[str]) -> None:
        from ..compile_environment import CompileEnvironment

        if self.prefix:
            raise ValueError("completed images allocated twice")
        begin = len(lines)
        for image, name, pointer, index, step in zip(
            self.images,
            self.names,
            self.pointer_names,
            self.index_names,
            self.step_names,
            strict=True,
        ):
            dtype = CompileEnvironment.current().backend.dtype_str(image.dtype)
            rows, columns = image.shape
            lines.extend(
                [
                    f"{pointer} = cute.arch.alloc_smem({dtype}, {rows * columns}, alignment=128)",
                    f"{name} = cute.make_tensor({pointer}, cute.make_layout({image.shape!r}, stride=({columns}, 1)))",
                    f"for {step} in cutlass.range({math.ceil(rows * columns / self.plan.threads)}, unroll=1):",
                    f"    {index} = chain_thread + {step} * {self.plan.threads}",
                    f"    if {index} < {rows * columns}:",
                    f"        {name}[{index} // {columns}, {index} % {columns}] = {dtype}(0)",
                ]
            )
        self.allocation_lines = tuple(lines[begin:])
        self.prefix = tuple(lines)

    def before_stage(
        self,
        stage: int,
        boundaries: dict[Node, str],
        lines: list[str],
        collective_lines: tuple[str, ...],
        row_registers: RowRegisterEmission | RowParallelEmission | None = None,
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from .chained_execution import ChainedExecution
        from .chained_matmul import _UnsupportedChain
        from .chained_row_parallel import RowParallelEmission
        from .chained_row_register_emission import RowRegisterEmission

        self._check(lines)
        if stage != self.stage + 1 or self.input_join != self.stage:
            raise _UnsupportedChain("completed image stage order changed")
        self.stage = stage
        # Only the original row emitter can add these FP32 boundaries. Its
        # joined return covers every row's publication in this collective batch.
        for node, source in self.sources.items():
            if (
                source.kind == "row"
                and node in boundaries
                and node not in self.completed
            ):
                if (
                    not isinstance(
                        row_registers, (RowRegisterEmission, RowParallelEmission)
                    )
                    or node not in row_registers.completed
                    or not collective_lines
                    or collective_lines[-1] != "cute.arch.sync_threads()"
                    or self.plan.row_registers is None
                    or not any(
                        item.output is node
                        and item.name == boundaries[node]
                        and item.shape == source.shape
                        for item in self.plan.row_registers.islands
                    )
                ):
                    raise _UnsupportedChain(
                        "completed image row lacks original joined owner"
                    )
                if isinstance(row_registers, RowParallelEmission):
                    completion = row_registers.completed[node]
                    if (
                        completion.owner is not row_registers.owner
                        or completion.node is not node
                        or completion.binding != boundaries[node]
                        or completion.execution != ChainedExecution(self.plan.threads)
                        or not completion.phase
                        or completion.phase[-1] != completion.execution.sync
                        or not any(
                            tuple(
                                collective_lines[start : start + len(completion.phase)]
                            )
                            == completion.phase
                            for start in range(len(collective_lines))
                        )
                    ):
                        raise _UnsupportedChain(
                            "completed image row completion receipt changed"
                        )
                self.completed[node] = boundaries[node]
        if any(boundaries.get(node) != owner for node, owner in self.completed.items()):
            raise _UnsupportedChain("completed image FP32 owner rebound")
        for image_index, (image, name) in enumerate(
            zip(self.images, self.names, strict=True)
        ):
            dtype = CompileEnvironment.current().backend.dtype_str(image.dtype)
            for publication in image.publications:
                node = publication.source.node
                key = (image_index, node)
                if key in self.published or node not in self.completed:
                    continue
                if publication.source.ready_before_stage > stage:
                    # A proved parallel batch may publish this FP32 owner early.
                    # Keep the image's conservative original-graph publication cut.
                    continue
                rows, columns = publication.extent
                row, column = publication.image_origin
                source_row, source_column = publication.source_origin
                index = self.cg.device_function.new_var("chain_completed_publish")
                step = self.cg.device_function.new_var("chain_completed_publish_step")
                begin = len(lines)
                lines.extend(
                    [
                        f"for {step} in cutlass.range({math.ceil(rows * columns / self.plan.threads)}, unroll=1):",
                        f"    {index} = chain_thread + {step} * {self.plan.threads}",
                        f"    if {index} < {rows * columns}:",
                        f"        {name}[{row} + {index} // {columns}, {column} + {index} % {columns}] = {dtype}(cutlass.Float32({self.completed[node]}[{source_row} + {index} // {columns}, {source_column} + {index} % {columns}]))",
                    ]
                )
                self.publication_lines[key] = tuple(lines[begin:])
                self.published.add(key)
        self.prefix = tuple(lines)

    @staticmethod
    def native_layout(
        image: CompletedImage, view: CompletedOperandView, layout: WarpOperandLayout
    ) -> bool:
        """Complete 16-byte rows for the unchanged LdMatrix 8x8 atom."""
        return (
            layout.node is view.operand
            and layout.shape == view.physical_shape
            and layout.dtype == image.dtype
            and layout.inner == (0 if view.transpose else 1)
            and all(size % 8 == 0 for size in (*view.physical_shape, *view.period))
            and image.shape[1] % 8 == 0
            and (view.origin[0] * image.shape[1] + view.origin[1]) % 8 == 0
        )

    def operand(
        self, stage: int, layout: WarpOperandLayout, domain: list[str]
    ) -> list[str] | None:
        from .chained_matmul import _UnsupportedChain

        if stage != self.stage or self.input_join >= stage:
            raise _UnsupportedChain("completed operand outside original input phase")
        index = 0 if layout.role == "a" else 1
        for image_index, image in enumerate(self.images):
            view = next(
                (
                    v
                    for v in image.operands
                    if (v.stage, v.operand_index) == (stage, index)
                ),
                None,
            )
            if view is None:
                continue
            rows, columns, reduction = self.plan.shapes[stage]
            threads = 32 * min(self.plan.threads // 32, 2 ** (columns.bit_length() - 4))
            topology = None
            for pair in self.warp_pairs:
                if stage in (pair.source, pair.source + 1):
                    threads = pair.layout.threads
                    topology = pair.layout.topology
                    break
            if (
                domain
                or not self.native_layout(image, view, layout)
                or not native_copy_layout(
                    image,
                    view,
                    layout,
                    (rows, columns, reduction),
                    threads,
                    topology=topology,
                )
            ):
                self.rejected.append(
                    (stage, layout.role, "original domain or native layout not proved")
                )
                return None
            needed = {image.cells[address].source for address in view.addresses} - {
                None
            }
            if any((image_index, node) not in self.published for node in needed):
                raise _UnsupportedChain(
                    "completed operand reads unpublished image cells"
                )
            key = (stage, index)
            if key in self.used:
                raise _UnsupportedChain("completed operand consumed twice")
            self.used.add(key)
            offset = view.origin[0] * image.shape[1] + view.origin[1]
            prefix = f"chain_{stage}_{layout.role}"
            return [
                f"{prefix}_ptr = {self.pointer_names[image_index]} + {offset}",
                f"{prefix} = cute.make_tensor({prefix}_ptr, {view.layout(image.shape[1])})",
            ]
        return None

    def joined_inputs(self, stage: int, lines: list[str]) -> None:
        from .chained_matmul import _UnsupportedChain

        self._check(lines)
        if (
            stage != self.stage
            or self.input_join != stage - 1
            or lines[-1] != "cute.arch.sync_threads()"
        ):
            raise _UnsupportedChain("completed image lacks original operand-read join")
        self.visible = set(self.published)
        self.input_join = stage
        self.prefix = tuple(lines)

    def complete_dot(
        self,
        stage: int,
        node: Node,
        owner: str,
        lines: list[str],
        publication: tuple[str, ...],
        *,
        native_completion: RootNativeCompletion | None = None,
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        self._check(lines)
        if native_completion is not None:
            native_completion.check(
                self.plan, node, owner, native_completion.phase, publication
            )
            if (
                node in self.sources
                or stage != self.input_join
                or tuple(lines[-len(native_completion.phase) :])
                != native_completion.phase
            ):
                raise _UnsupportedChain(
                    "native completion is not the original image cut"
                )
            self.prefix = tuple(lines)
            return
        if (
            self.plan.root_native is not None
            and self.plan.root_native.selected(self.plan, stage) is not None
        ):
            raise _UnsupportedChain("native image cut lacks its completion receipt")
        if (
            stage != self.input_join
            or node is not self.plan.dots[stage]
            or lines[-1] != "cute.arch.sync_threads()"
            or publication
            != (
                f"chain_{stage}_sc = chain_{stage}_thr.partition_C({owner})",
                f"cute.autovec_copy(chain_{stage}_acc, chain_{stage}_sc)",
            )
        ):
            raise _UnsupportedChain("completed image lacks original FP32 publication")
        if node in self.sources:
            if (
                self.sources[node].ready_before_stage != stage + 1
                or node in self.completed
            ):
                raise _UnsupportedChain("completed image contraction readiness changed")
            self.completed[node] = owner
        self.prefix = tuple(lines)

    def bind_exports(self, boundaries: dict[Node, str], lines: list[str]) -> None:
        from .chained_matmul import _UnsupportedChain

        self._check(lines)
        if (
            self.input_join != len(self.plan.dots) - 1
            or not self.used
            or any(
                boundaries.get(node) != owner for node, owner in self.completed.items()
            )
        ):
            raise _UnsupportedChain(
                "completed image did not consume original joined operands"
            )
        for image_index, (image, name) in enumerate(
            zip(self.images, self.names, strict=True)
        ):
            if any(
                (image_index, p.source.node) not in self.visible
                for p in image.publications
            ):
                raise _UnsupportedChain(
                    "completed image export precedes joined completion"
                )
            if image.node in boundaries:
                raise _UnsupportedChain("completed image export boundary already bound")
            # Bind the exact typed graph node. Existing view lowering retains
            # every RHS modulo, and store-domain lowering remains independent.
            boundaries[image.node] = name
        self.finished = True

    def record_stage(
        self,
        node: Node,
        lines: list[str],
        *,
        native_completion: RootNativeCompletion | None = None,
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        source = tuple(line for block in lines for line in block.splitlines())
        prefix = tuple(line for block in self.prefix for line in block.splitlines())
        if (
            len(self.native_stages) >= len(self.plan.dots)
            or node is not self.plan.dots[len(self.native_stages)]
            or not source
            or prefix[-len(source) :] != source
        ):
            raise _UnsupportedChain("direct image native phase changed")
        selected = (
            self.plan.root_native.selected(self.plan, len(self.native_stages))
            if self.plan.root_native is not None
            else None
        )
        if (selected is None) != (native_completion is None):
            raise _UnsupportedChain("native stage kind lacks its physical receipt")
        if native_completion is not None:
            assert selected is not None
            native_completion.check(
                self.plan, node, f"chain_{selected.index}_c", tuple(lines)
            )
        self.native_stages.append((node, source))
        if native_completion is not None:
            self.native_completions[node] = native_completion

    def retire_owners(
        self,
        lines: list[str],
        *,
        row_registers: RowRegisterEmission | RowParallelEmission | None = None,
        record: bool = True,
    ) -> list[str]:
        from .chained_direct_image import lower_direct_images
        from .chained_matmul import _UnsupportedChain
        from .native_matmul_metadata import record_native_stage

        self._check(lines, allow_finished=True)
        if tuple(node for node, _ in self.native_stages) != self.plan.dots:
            raise _UnsupportedChain("direct image native stage coverage changed")
        self.lowering = lower_direct_images(self, lines, row_registers=row_registers)
        expected = (
            set()
            if self.plan.root_native is None
            else {x.node for x in self.plan.root_native.stages}
        )
        if set(self.native_completions) != expected:
            raise _UnsupportedChain("native image stage coverage changed")
        result = lines if self.lowering is None else list(self.lowering.result)
        sources = tuple(
            (node, source if self.lowering is None else self.lowering.segment(source))
            for node, source in self.native_stages
        )
        grouped = None
        if self.cg.device_function.config.config.get("cute_chained_parallel_merges"):
            from .chained_parallel_merges import parallelize_completed_merges

            grouped = parallelize_completed_merges(self, warp_pairs=self.warp_pairs)
        if grouped is not None:
            if grouped.retired_pairs != self.warp_pairs:
                raise _UnsupportedChain("grouped register retirement coverage changed")
            result = list(grouped.result)
        elif self.warp_pairs:
            from .chained_completed_register import retire_pair_results

            retired, sources = retire_pair_results(
                self, result, sources, self.warp_pairs
            )
            result = list(retired)
        source_map = dict(sources)
        for node, completion in self.native_completions.items():
            from .chained_root_typed_publication import retire_typed_publication

            for typed in completion.body.typed_publications.values():
                if typed.completion is not completion:
                    continue
                retirement = retire_typed_publication(typed, result)
                if retirement is None:
                    continue
                retirement.check(result)
                phase = retire_typed_publication(typed, list(source_map[node]))
                if phase is None:
                    raise _UnsupportedChain("typed publication lost its native phase")
                phase.check(list(source_map[node]))
                result = list(retirement.after)
                source_map[node] = phase.after
        sources = tuple((node, source_map[node]) for node, _ in sources)
        for node, source in sources:
            native_completion = self.native_completions.get(node)
            if native_completion is not None:
                native_completion.check(
                    self.plan, node, native_completion.owner, native_completion.phase
                )
            if grouped is not None and node in grouped.nodes:
                if record and node is grouped.nodes[0]:
                    record_native_stage(
                        self.cg, self.plan, grouped.nodes, "warp", grouped.source
                    )
                continue
            if record:
                record_native_stage(
                    self.cg,
                    self.plan,
                    (node,),
                    "warp" if native_completion is None else "tcgen05",
                    source,
                )
        return result


def native_copy_layout(
    image: CompletedImage,
    view: CompletedOperandView,
    layout: WarpOperandLayout,
    shape: tuple[int, int, int],
    threads: int,
    *,
    topology: tuple[int, int, int] | None = None,
) -> bool:
    """Prove actual SDK partitions against PTX LdMatrix packet ordering.

    This constructs host IR only. It neither compiles nor loads a CUDA module.
    Image-cell provenance already proves repeated logical coordinates equal;
    here every physical 16-byte packet and delivered element is checked.
    """
    import cutlass
    import cutlass.cute as cute
    import torch

    ir = importlib.import_module("cutlass._mlir.ir")
    kind = cutlass.BFloat16 if image.dtype is torch.bfloat16 else cutlass.Float16
    atom_k = 16 if shape[2] % 16 == 0 else 8
    role = layout.role.upper()
    matrices = atom_k // (4 if role == "A" else 8)
    strides = (1, image.shape[1]) if view.transpose else (image.shape[1], 1)
    modes = tuple(
        extent if extent == period else (period, extent // period)
        for extent, period in zip(view.physical_shape, view.period, strict=True)
    )
    steps = tuple(
        stride if extent == period else (stride, 0)
        for extent, period, stride in zip(
            view.physical_shape, view.period, strides, strict=True
        )
    )
    origin = view.origin[0] * image.shape[1] + view.origin[1]
    capacity = math.prod(image.shape)
    topology = topology if topology is not None else (1, threads // 32, 1)
    if threads % 32 or min(topology) <= 0 or math.prod(topology) * 32 != threads:
        return False
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(kind, cutlass.Float32, (16, 8, atom_k))
                ),
                atom_layout_mnk=topology,
            )
            native_layout = cute.make_layout(modes, stride=steps)
            identity = cute.make_identity_tensor(view.physical_shape)
            copy = getattr(cute, f"make_tiled_copy_{role}")(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        transpose=layout.inner == 0, num_matrices=matrices
                    ),
                    kind,
                ),
                mma,
            )
            for warp in range(threads // 32):
                sources = [
                    copy.get_slice(warp * 32 + lane).partition_S(identity)
                    for lane in range(32)
                ]
                fragments = [
                    getattr(mma.get_slice(warp * 32 + lane), f"partition_{role}")(
                        identity
                    )
                    for lane in range(32)
                ]
                destinations = [
                    copy.get_slice(warp * 32 + lane).retile(fragments[lane])
                    for lane in range(32)
                ]
                for tile in range(cute.size(sources[0], mode=[1])):
                    for kk in range(shape[2] // atom_k):
                        packets = []
                        for lane in range(8 * matrices):
                            panel = sources[lane][None, tile, kk]
                            coords = [
                                tuple(map(int, panel[i]))
                                for i in range(cute.size(panel))
                            ]
                            offsets = [
                                origin + int(native_layout(coord)) for coord in coords
                            ]
                            if (
                                len(offsets) != 8
                                or offsets != list(range(offsets[0], offsets[0] + 8))
                                or offsets[0] % 8
                                or offsets[0] < 0
                                or offsets[-1] >= capacity
                                or any(
                                    not (
                                        0 <= r < view.physical_shape[0]
                                        and kk * atom_k <= c < (kk + 1) * atom_k
                                    )
                                    for r, c in coords
                                )
                            ):
                                return False
                            packets.append(coords)
                        for lane in range(32):
                            delivered = []
                            for matrix in range(matrices):
                                for half in range(2):
                                    row, column = lane // 4, 2 * (lane % 4) + half
                                    if layout.inner == 0:
                                        row, column = column, row
                                    delivered.append(packets[matrix * 8 + row][column])
                            panel = destinations[lane][None, tile, kk]
                            if delivered != [
                                tuple(map(int, panel[i]))
                                for i in range(cute.size(panel))
                            ]:
                                return False
        return bool(module.operation.verify())
