"""Retire completed FP32 storage only behind a complete typed image read cut."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import view_ops
from .chained_completed_image import CompletedImage
from .chained_completed_image import ImageProvenance
from .chained_completed_image import ImagePublication
from .chained_root_native import joined_root

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chained_completed_image_emission import CompletedImageEmission
    from .chained_matmul import ChainedMatmulPlan
    from .chained_row_parallel import RowParallelEmission
    from .chained_row_register_emission import RowRegisterEmission
    from .chained_row_registers import RowRegisterIsland
    from .chained_warp_pair import WarpPairAttempt


@dataclass(frozen=True)
class DirectImageOwner:
    image: int
    publication: ImagePublication


@dataclass(frozen=True)
class DirectImagePlan:
    owners: tuple[DirectImageOwner, ...] = ()

    def owner(self, node: Node) -> DirectImageOwner | None:
        return next(
            (item for item in self.owners if item.publication.source.node is node),
            None,
        )

    def facts(self, images: tuple[CompletedImage, ...]) -> tuple[object, ...]:
        return tuple(
            (
                item.image,
                item.publication.source.node,
                id(item.publication.source.node.graph),
                id(item.publication.source.node.meta["val"]),
                item.publication.source.shape,
                item.publication.source.kind,
                item.publication.source.ready_before_stage,
                item.publication.source_origin,
                item.publication.image_origin,
                item.publication.extent,
                images[item.image].shape,
                images[item.image].dtype,
                tuple(
                    (cell.source, cell.index, cell.casts)
                    for cell in images[item.image].cells
                ),
            )
            for item in self.owners
        )


def _read_cut(image: CompletedImage, plan: ChainedMatmulPlan) -> bool:
    """All paths terminate at this exact typed image or covered MMA operands.

    This is deliberately stronger than additive image admission: no scalar
    fallback, side arithmetic, unknown view or unobserved live-out can survive.
    """
    from .chained_matmul import _ancestors

    assert plan.region is not None
    sources = tuple(item.source for item in image.publications)
    analysis = ImageProvenance(sources, maximum_elements=len(image.cells) * 4)
    covered = {(v.stage, v.operand_index): v for v in image.operands}
    owners = {source.node for source in sources}
    for source in sources:
        pending, seen = list(source.node.users), set()
        while pending:
            node = pending.pop()
            if node in seen or node is image.node:
                continue
            seen.add(node)
            if node in plan.dots:
                stage = plan.dots.index(node)
                for index, operand in enumerate(node.args[:3]):
                    if not isinstance(operand, Node) or source.node not in _ancestors(
                        operand
                    ):
                        continue
                    view = covered.get((stage, index))
                    if view is None or view.operand is not operand:
                        return False
                continue
            if node.op != "call_function" or node.is_impure() or not node.users:
                return False
            # Split tuples are checked through their selected tensor children.
            if node.target is not view_ops.split:
                tensor = analysis.image(node)
                if tensor is None:
                    return False
                allowed = set(image.cells)
                if tensor.dtype is not torch.float32 and any(
                    cell not in allowed for cell in tensor.cells
                ):
                    return False
            if node in owners:
                return False
            pending.extend(node.users)
    return True


def _native_c_coordinates(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    threads: int,
    *,
    topology: tuple[int, int, int] | None = None,
) -> bool:
    """Verify every original accumulator slot has one physical C owner."""
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    kind = cutlass.BFloat16 if dtype is torch.bfloat16 else cutlass.Float16
    topology = topology if topology is not None else (1, threads // 32, 1)
    if threads % 32 or min(topology) <= 0 or math.prod(topology) * 32 != threads:
        return False
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        kind, cutlass.Float32, (16, 8, 16 if shape[2] % 16 == 0 else 8)
                    )
                ),
                atom_layout_mnk=topology,
            )
            found = []
            for lane in range(threads):
                coordinates = mma.get_slice(lane).partition_C(
                    cute.make_identity_tensor(shape[:2])
                )
                found.extend(
                    tuple(map(int, coordinates[i]))
                    for i in range(cute.size(coordinates))
                )
            if sorted(found) != [
                (row, column) for row in range(shape[0]) for column in range(shape[1])
            ]:
                return False
        return bool(module.operation.verify())


def plan_direct_images(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    images: tuple[CompletedImage, ...],
    *,
    warp_pairs: tuple[WarpPairAttempt, ...] = (),
) -> DirectImagePlan:
    """Propose retirement; original emitted readers still decide admission.

    This query does not lower expressions, create names, or change capacity
    accounting. Unsupported readers keep the original additive body exactly.
    """
    config = cg.device_function.config.config
    if (
        (config.get("cute_chained_warp_register_a") is True and not warp_pairs)
        or config.get("cute_chained_operand_coordinates") is not True
        or config.get("cute_chained_mma_schedule", "coalesced")
        not in ("coalesced", "coalesced_unrolled", "cp_async")
        or config.get("cute_chained_scratch_layout", "row_major") != "row_major"
        or plan.pointwise_cache is not None
        or plan.loop is not None
        or not joined_root(plan)
    ):
        return DirectImagePlan()
    owners: list[DirectImageOwner] = []
    for index, image in enumerate(images):
        image.check(plan)
        if not _read_cut(image, plan):
            return DirectImagePlan()
        for item in image.publications:
            if item.source.kind == "row":
                if item.source_origin != (0, 0) or item.extent != item.source.shape:
                    return DirectImagePlan()
            elif item.source.kind == "contraction":
                stage = plan.dots.index(item.source.node)
                shape = plan.shapes[stage]
                threads = 32 * min(plan.threads // 32, 2 ** (shape[1].bit_length() - 4))
                topology = None
                for pair in warp_pairs:
                    pair.check()
                    if stage in (pair.source, pair.source + 1):
                        threads = pair.layout.threads
                        topology = pair.layout.topology
                        break
                if not _native_c_coordinates(
                    shape, plan.operand_dtype(stage), threads, topology=topology
                ):
                    return DirectImagePlan()
            else:
                return DirectImagePlan()
            owners.append(DirectImageOwner(index, item))
    if len({item.publication.source.node for item in owners}) != len(owners):
        return DirectImagePlan()
    return DirectImagePlan(tuple(owners))


@dataclass(frozen=True)
class DirectImageEdit:
    """One exact compiler-owned span; only a uniform enclosing indent varies."""

    before: tuple[str, ...]
    after: tuple[str, ...]

    def locate(self, lines: tuple[str, ...]) -> tuple[int, str] | None:
        from .chained_matmul import _UnsupportedChain

        matches = []
        for start, line in enumerate(lines):
            indent = line[: len(line) - len(line.lstrip(" "))]
            if tuple(lines[start : start + len(self.before)]) == tuple(
                indent + part for part in self.before
            ):
                matches.append((start, indent))
        if len(matches) > 1:
            raise _UnsupportedChain("ambiguous direct image publication span")
        return matches[0] if matches else None


@dataclass(frozen=True)
class DirectImageLowering:
    original: tuple[str, ...]
    result: tuple[str, ...]
    edits: tuple[DirectImageEdit, ...]

    def segment(self, source: tuple[str, ...]) -> tuple[str, ...]:
        return _apply(source, self.edits, complete=False)


def _apply(
    source: tuple[str, ...], edits: tuple[DirectImageEdit, ...], *, complete: bool
) -> tuple[str, ...]:
    from .chained_matmul import _UnsupportedChain

    ranges = []
    for edit in edits:
        match = edit.locate(source)
        if match is None:
            if complete:
                raise _UnsupportedChain("direct image original publication changed")
            continue
        start, indent = match
        ranges.append((start, start + len(edit.before), edit, indent))
    ranges.sort(key=operator.itemgetter(0))
    result: list[str] = []
    inverse: list[str] = []
    cursor = 0
    for start, stop, edit, indent in ranges:
        if start < cursor:
            raise _UnsupportedChain("overlapping direct image publication spans")
        result.extend(source[cursor:start])
        inverse.extend(source[cursor:start])
        result.extend(indent + line for line in edit.after)
        inverse.extend(indent + line for line in edit.before)
        cursor = stop
    result.extend(source[cursor:])
    inverse.extend(source[cursor:])
    if tuple(inverse) != source:
        raise _UnsupportedChain("direct image publication inverse changed")
    return tuple(result)


def _row_publication_names(
    plan: ChainedMatmulPlan,
    island: RowRegisterIsland,
    rows: RowRegisterEmission | RowParallelEmission | None,
) -> tuple[str, str, str]:
    from .chained_matmul import _UnsupportedChain
    from .chained_row_parallel import RowParallelEmission

    original = (
        f"{island.name}_row",
        f"{island.name}_column",
        f"{island.name}_values",
    )
    if not plan.row_subwarp_packing:
        return original
    if not isinstance(rows, RowParallelEmission):
        raise _UnsupportedChain("direct packed image lacks row completion")
    rows.validate(plan)
    batches = [
        receipt
        for receipt in rows.batches
        if any(item.island is island for item in receipt.batch.islands)
    ]
    if len(batches) != 1:
        raise _UnsupportedChain("direct packed image row completion changed")
    receipt = batches[0]
    if receipt.packing is None:
        return original
    return receipt.packing.publication_names(island, receipt.execution, receipt.phase)


def lower_direct_images(
    emission: CompletedImageEmission,
    lines: list[str],
    *,
    row_registers: RowRegisterEmission | RowParallelEmission | None = None,
) -> DirectImageLowering | None:
    """Retire storage only after every original native reader/export is bound.

    All scalar lowering, domain checks, row phases and original joins have
    already run. This pass replaces exact typed allocation/publication spans;
    it does not normalize arithmetic, indices, masks, or arbitrary AST nodes.
    """
    from ..compile_environment import CompileEnvironment
    from .chained_matmul import _UnsupportedChain

    plan, direct = emission.plan, emission.direct
    expected = {
        (view.stage, view.operand_index)
        for image in emission.images
        for view in image.operands
    }
    if not direct.owners or emission.rejected or emission.used != expected:
        return None
    if not emission.finished or set(emission.completed) != set(emission.sources):
        raise _UnsupportedChain("direct image lacks complete original owners")
    edits: list[DirectImageEdit] = []
    row_aliases: list[str] = []
    for owner in direct.owners:
        item = owner.publication
        image = emission.images[owner.image]
        node = item.source.node
        name = emission.completed[node]
        dtype = CompileEnvironment.current().backend.dtype_str(image.dtype)
        offset = (
            (item.image_origin[0] - item.source_origin[0]) * image.shape[1]
            + item.image_origin[1]
            - item.source_origin[1]
        )
        if offset < 0:
            # No pointer before the allocation, even if all reads add it back.
            return None
        alias = (
            f"{name}_ptr = {emission.pointer_names[owner.image]} + {offset}",
            f"{name} = cute.make_tensor({name}_ptr, cute.make_layout({item.source.shape!r}, stride=({image.shape[1]}, 1)))",
        )
        if item.source.kind == "row":
            assert plan.row_registers is not None
            island = next(i for i in plan.row_registers.islands if i.output is node)
            if island.name != name or island.shape != item.source.shape:
                raise _UnsupportedChain("direct image row owner changed")
            rows, columns = island.shape
            allocation = f"{name} = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {rows * columns}, alignment=128), cute.make_layout({island.shape!r}, stride=({columns}, 1)))"
            edits.append(DirectImageEdit((allocation,), ()))
            row_aliases.extend(alias)
            row, column, registers = _row_publication_names(plan, island, row_registers)
            target = f"{name}[{row}, {column}]"
            value = f"{registers}[{island.events[-1].offset} + {column}]"
            prefix = (
                f"if {row} < {rows}:",
                f"    for {column} in cutlass.range_constexpr({columns}):",
            )
            edits.append(
                DirectImageEdit(
                    (*prefix, f"        {target} = {value}"),
                    (*prefix, f"        {target} = {dtype}(cutlass.Float32({value}))"),
                )
            )
        else:
            stage = plan.dots.index(node)
            prefix = f"chain_{stage}"
            rows, columns = item.source.shape
            original_allocation = (
                f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {rows * (columns + 4)}, alignment=128)",
                f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, cute.make_layout({item.source.shape!r}, stride=({columns + 4}, 1)))",
            )
            edits.append(DirectImageEdit(original_allocation, alias))
            row, column, index = (
                f"{prefix}_image_row",
                f"{prefix}_image_column",
                f"{prefix}_image_index",
            )
            publication = [
                f"{prefix}_image_coordinates = {prefix}_thr.partition_C(cute.make_identity_tensor({item.source.shape!r}))",
                f"for {index} in cutlass.range_constexpr(cute.size({prefix}_acc)):",
                f"    {row}, {column} = {prefix}_image_coordinates[{index}]",
            ]
            bounds = [
                f"{start} <= {coordinate} < {start + extent}"
                for coordinate, start, extent, size in zip(
                    (row, column),
                    item.source_origin,
                    item.extent,
                    item.source.shape,
                    strict=True,
                )
                if start != 0 or extent != size
            ]
            if bounds:
                publication.append(f"    if {' and '.join(bounds)}:")
            indent = "        " if bounds else "    "
            publication.append(
                f"{indent}{name}[{row}, {column}] = {dtype}(cutlass.Float32({prefix}_acc[{index}]))"
            )
            edits.append(
                DirectImageEdit(
                    (
                        f"{prefix}_sc = {prefix}_thr.partition_C({name})",
                        f"cute.autovec_copy({prefix}_acc, {prefix}_sc)",
                    ),
                    tuple(publication),
                )
            )
        original_copy = emission.publication_lines[(owner.image, node)]
        edits.append(DirectImageEdit(original_copy, ()))
    allocation = emission.allocation_lines
    edits.append(DirectImageEdit(allocation, (*allocation, *row_aliases)))
    source = tuple(line for block in lines for line in block.splitlines())
    result = _apply(source, tuple(edits), complete=True)
    return DirectImageLowering(source, result, tuple(edits))
