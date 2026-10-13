"""Finite-stage horizontal transport of a proved pointwise state plane."""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from .horizontal_state_coefficients import plan_feature_publications
from .horizontal_state_expression import lower_scheduled_scalar_program
from .horizontal_state_update import InapplicableStateUpdate
from .horizontal_state_update import _freeze
from .horizontal_state_update import _proven_int

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .horizontal_state_coefficients import FeaturePublication
    from .horizontal_state_expression import ScheduledScalarProgram
    from .horizontal_state_plan import HorizontalStatePlan


@dataclasses.dataclass(frozen=True)
class InstalledHorizontalState:
    plan: HorizontalStatePlan
    program: ScheduledScalarProgram
    publications: tuple[FeaturePublication, ...]
    body: tuple[str, ...]
    wrapper: object
    parameters: tuple[str, ...]
    selection: object = dataclasses.field(default=None, repr=False)

    def facts(self) -> object:
        return _freeze(
            (
                self.plan,
                self.program,
                self.publications,
                self.body,
                self.wrapper,
                self.parameters,
            )
        )

    def check(self, cg: GenerateAST) -> None:
        if self.selection != self.facts():
            raise InapplicableStateUpdate("horizontal installed receipt changed")
        if self.plan is not cg.device_function.cute_state.horizontal_state_plan:
            raise InapplicableStateUpdate(
                "horizontal live plan was removed or replaced"
            )
        self.plan.check()
        self.program.check(cg)
        for publication in self.publications:
            publication.check(cg, self.program)
        if (
            self.body != tuple(map(ast.dump, cg.device_function.body))
            or self.wrapper != _freeze(cg.cute_wrapper_plans)
            or self.parameters != tuple(cg.device_function.wrapper_only_params)
        ):
            raise InapplicableStateUpdate(
                "horizontal body or descriptor protocol changed"
            )


class _SelectConsumerTransport(ast.NodeTransformer):
    """Specialize only the emitted CTA-uniform transport predicate."""

    def __init__(self, eligible: str, selected: bool) -> None:
        self.eligible = eligible
        self.selected = selected

    def choice(self, test: ast.AST) -> bool | None:
        if isinstance(test, ast.Name) and test.id == self.eligible:
            return self.selected
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name)
            and test.operand.id == self.eligible
        ):
            return not self.selected
        return None

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        selected = self.choice(node.test)
        if selected is None:
            return self.generic_visit(node)
        block = ast.Module(body=node.body if selected else node.orelse, type_ignores=[])
        self.generic_visit(block)
        return block.body

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        selected = self.choice(node.test)
        if selected is None:
            return self.generic_visit(node)
        return self.visit(node.body if selected else node.orelse)


def _split_consumer_transport(loop: ast.For, eligible: str) -> ast.If:
    """Duplicate an original stage loop under its already-proved uniform choice.

    The selected typed domain supplies the predicate from readonly prefix loads
    and CTA coordinates. No expression, load, or barrier crosses a stage boundary.
    Unknown uses or writes of the predicate fail closed instead of specializing.
    """
    if any(
        isinstance(node, ast.Name)
        and node.id == eligible
        and not isinstance(node.ctx, ast.Load)
        for node in ast.walk(loop)
    ):
        raise InapplicableStateUpdate("consumer loop changes transport predicate")
    branches = []
    for selected in (True, False):
        branch = _SelectConsumerTransport(eligible, selected).visit(copy.deepcopy(loop))
        if not isinstance(branch, ast.For) or any(
            isinstance(node, ast.Name) and node.id == eligible
            for node in ast.walk(branch)
        ):
            raise InapplicableStateUpdate("unclassified consumer transport use")
        branches.append(branch)
    return ast.If(
        test=ast.Name(id=eligible, ctx=ast.Load()),
        body=[branches[0]],
        orelse=[branches[1]],
    )


def codegen_horizontal_state(cg: GenerateAST) -> bool:
    from ..program_id import CuteProgramIDs
    from ..program_id import FlatProgramIDs

    df = cg.device_function
    plan = df.cute_state.horizontal_state_plan
    if plan is None:
        return False
    plan.check()
    domain, geometry = plan.domain, plan.geometry
    region = domain.region
    if cg.current_root_graph_info is not region.root:
        raise InapplicableStateUpdate("horizontal root identity changed")
    if type(df.pid) not in (FlatProgramIDs, CuteProgramIDs):
        raise InapplicableStateUpdate(
            "horizontal task mapping requires ordinary flat programs"
        )
    assert isinstance(df.pid, (FlatProgramIDs, CuteProgramIDs))
    counts = (*domain.grid_extents[:-1], -(-domain.grid_extents[-1] // geometry.rows))
    if {info.block_id: info.static_num_pids() for info in df.pid.pid_info} != dict(
        zip(domain.grid_block_ids, counts, strict=True)
    ):
        raise InapplicableStateUpdate(
            "original launch does not cover the complete task domain"
        )
    if cg.cute_wrapper_plans or df.wrapper_only_params:
        raise InapplicableStateUpdate("horizontal root has a foreign wrapper")
    names = {
        name: df.new_var(f"horizontal_{name}")
        for name in (
            "lane",
            "warp",
            "tid",
            "pid",
            "row_base",
            "row",
            "local_row",
            "member",
            "row_group",
            "feature",
            "column",
            "item",
            "stage",
            "eligible",
            "descriptor_row",
            "epi",
            "layout",
            "shared",
            "bytes",
            "byte_layout",
            "byte_offset",
            "pointer",
            "empty",
            "full",
            "load_atom",
            "load_tensor",
            "store_atom",
            "store_tensor",
            "load_tile",
            "load_s",
            "load_g",
            "store_tile",
            "store_s",
            "store_g",
            "acc",
            "sum",
            "neighbor",
            "consumers",
            "coefficient_wave",
            "coefficient_offset",
            "coefficient_values",
            "state_values",
            "new_state_values",
            "component",
        )
    }

    def n(name: str) -> str:
        return names[name]

    outer = tuple(df.new_var(f"horizontal_outer_{i}") for i in range(len(counts) - 1))
    program = lower_scheduled_scalar_program(
        cg,
        domain,
        outer_coordinates=outer,
        coordinates=(n("row"), n("feature")),
        reduced_value=n("sum"),
    )
    original = program.original
    program.check(cg)
    publications = plan_feature_publications(cg, program, outer)
    geometry.check(48 * 1024, sum(p.words * 2 + 16 for p in publications) + 1024)
    pairs = {p.binding: df.new_var("horizontal_coefficient_pair") for p in publications}
    dtype = CompileEnvironment.current().backend.dtype_str(domain.storage_dtype)
    index_type = CompileEnvironment.current().index_type()
    prefix = original.old_state.indices[:-2]
    eligible = [
        f"(0 <= {index_type}({x}) < {extent})"
        for x, extent in zip(prefix, domain.state_shape[:-2], strict=True)
    ]
    eligible.append(f"({n('row_base')} + {geometry.rows} <= {domain.state_shape[-2]})")
    row_address = " + ".join(
        f"({index_type}({x}) * {stride // domain.state_shape[-1]})"
        for x, stride in zip(prefix, domain.state_strides[:-2], strict=True)
    )
    if row_address:
        row_address += " + "
    row_address += n("row_base")
    lines: list[str] = []

    def add(depth: int, text: str) -> None:
        lines.extend("    " * depth + row for row in text.splitlines())

    add(0, f"{n('lane')} = cutlass.Int32(cute.arch.thread_idx()[0])")
    add(0, f"{n('warp')} = cutlass.Int32(cute.arch.thread_idx()[1])")
    add(0, f"{n('tid')} = {n('lane')} + 32 * {n('warp')}")
    add(0, f"{n('pid')} = cutlass.Int32(cute.arch.block_idx()[0])")
    for i, name in enumerate(outer):
        add(0, f"{name} = ({n('pid')} // {math.prod(counts[i + 1 :])}) % {counts[i]}")
    add(0, f"{n('row_base')} = ({n('pid')} % {counts[-1]}) * {geometry.rows}")
    for line in program.setup:
        add(0, line)
    add(0, f"{n('eligible')} = {' & '.join(eligible)}")
    add(0, f"if {n('eligible')}:")
    add(1, f"{n('eligible')} = (cutlass.Int32({row_address}) % {geometry.rows}) == 0")
    add(
        0,
        f"{n('epi')} = cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape(({geometry.rows}, {geometry.stage_features}), False, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {dtype}, layout_c=cutlass.utils.layout.LayoutEnum.ROW_MAJOR, elem_ty_c={dtype})",
    )
    add(
        0,
        f"{n('layout')} = cutlass.utils.blackwell_helpers.make_smem_layout_epi({dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {n('epi')}, {geometry.stages})",
    )
    add(
        0,
        f"{n('shared')} = cute.make_tensor(cute.arch.alloc_smem({dtype}, {geometry.state_bytes // geometry.itemsize}, alignment=1024), {n('layout')})",
    )
    add(
        0,
        f"{n('byte_layout')} = cute.make_composed_layout(cute.get_swizzle_portion({n('layout')}), 0, cute.make_layout({geometry.state_bytes}))",
    )
    add(
        0,
        f"{n('bytes')} = cute.recast_ptr({n('shared')}.iterator, dtype=cutlass.Uint8)",
    )
    for publication in publications:
        add(
            0,
            f"{publication.shared} = cute.arch.alloc_smem(cutlass.Uint16, {publication.words}, alignment=16)",
        )
    if publications:
        add(
            0,
            f"{n('consumers')} = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=16)",
        )
    for barrier in ("empty", "full"):
        add(
            0,
            f"{n(barrier)} = cute.arch.alloc_smem(cutlass.Int64, {geometry.stages}, alignment=16)",
        )
    add(0, f"if {n('tid')} == 0:")
    if publications:
        add(
            1,
            f"cute.arch.mbarrier_init({n('consumers')}, {geometry.consumer_warps * 32})",
        )
    add(1, f"for {n('stage')} in cutlass.range_constexpr({geometry.stages}):")
    for barrier in ("empty", "full"):
        add(
            2,
            f"cute.arch.mbarrier_init({n(barrier)} + {n('stage')}, {geometry.arrivals})",
        )
    add(0, "cute.arch.mbarrier_init_fence()")
    add(0, "cute.arch.sync_threads()")
    add(0, f"if {n('warp')} == {geometry.consumer_warps}:")
    add(1, f"for {n('stage')} in cutlass.range_constexpr({geometry.stages}):")
    add(2, f"if {n('lane')} == 0:")
    add(3, f"cute.arch.mbarrier_arrive({n('empty')} + {n('stage')})")
    add(2, f"cute.arch.mbarrier_wait({n('empty')} + {n('stage')}, 0)")
    add(2, f"if {n('eligible')}:")
    add(3, f"{n('descriptor_row')} = cutlass.Int32({row_address})")
    add(3, f"if {n('lane')} == 0:")
    add(
        4,
        f"cute.arch.mbarrier_arrive_and_expect_tx({n('full')} + {n('stage')}, {geometry.rows * geometry.stage_features * geometry.itemsize})",
    )
    # local_tile's M coordinate is a tile coordinate, not an element offset.
    add(
        3,
        f"{n('load_tile')} = cute.local_tile({n('load_tensor')}, ({geometry.rows}, {geometry.stage_features}), ({n('descriptor_row')} // {geometry.rows}, {n('stage')}))",
    )
    add(
        3,
        f"{n('load_s')}, {n('load_g')} = cute.nvgpu.cpasync.tma_partition({n('load_atom')}, 0, cute.make_layout(1), cute.group_modes({n('shared')}[None, None, {n('stage')}], 0, 2), cute.group_modes({n('load_tile')}, 0, 2))",
    )
    add(
        3,
        f"cute.copy({n('load_atom')}, {n('load_g')}, {n('load_s')}, tma_bar_ptr={n('full')} + {n('stage')})",
    )
    add(2, "else:")
    add(3, f"if {n('lane')} == 0:")
    add(4, f"cute.arch.mbarrier_arrive({n('full')} + {n('stage')})")
    add(1, f"for {n('stage')} in cutlass.range_constexpr({geometry.stages}):")
    add(2, f"if {n('lane')} == 0:")
    add(3, f"cute.arch.mbarrier_arrive({n('empty')} + {n('stage')})")
    add(2, f"cute.arch.mbarrier_wait({n('empty')} + {n('stage')}, 1)")
    add(2, f"if {n('eligible')}:")
    add(3, "cute.arch.fence_proxy(kind='async.shared', space='cta')")
    add(3, f"{n('descriptor_row')} = cutlass.Int32({row_address})")
    add(
        3,
        f"{n('store_tile')} = cute.local_tile({n('store_tensor')}, ({geometry.rows}, {geometry.stage_features}), ({n('descriptor_row')} // {geometry.rows}, {n('stage')}))",
    )
    add(
        3,
        f"{n('store_s')}, {n('store_g')} = cute.nvgpu.cpasync.tma_partition({n('store_atom')}, 0, cute.make_layout(1), cute.group_modes({n('shared')}[None, None, {n('stage')}], 0, 2), cute.group_modes({n('store_tile')}, 0, 2))",
    )
    add(3, f"cute.copy({n('store_atom')}, {n('store_s')}, {n('store_g')})")
    add(3, "cute.arch.cp_async_bulk_commit_group()")
    add(3, "cute.arch.cp_async_bulk_wait_group(0, read=True)")
    add(1, "cute.arch.cp_async_bulk_wait_group(0, read=False)")
    add(0, "else:")
    add(1, f"for {n('stage')} in cutlass.range_constexpr({geometry.stages}):")
    add(2, f"cute.arch.mbarrier_arrive({n('empty')} + {n('stage')})")
    for ordinal, publication in enumerate(publications):
        publication.check(cg, program)
        width = publication.vector_width
        packets = publication.words // width
        add(1, f"if {n('warp')} == {ordinal % geometry.consumer_warps}:")
        add(
            2,
            f"for {n('coefficient_wave')} in cutlass.range_constexpr({-(-packets // 32)}):",
        )
        add(
            3,
            f"{n('coefficient_offset')} = ({n('coefficient_wave')} * 32 + {n('lane')}) * {width}",
        )
        add(3, f"if {n('coefficient_offset')} < {publication.words}:")
        pointer = f"cute.recast_ptr({publication.access.pointer}, dtype=cutlass.Uint16) + {n('coefficient_offset')}"
        value_type = (
            f"ir.VectorType.get([{width}], cutlass.Uint16.mlir_type)"
            if width > 1
            else "cutlass.Uint16"
        )
        add(4, f"{n('coefficient_values')} = cute.arch.load({pointer}, {value_type})")
        add(
            4,
            f"cute.arch.store({publication.shared} + {n('coefficient_offset')}, {n('coefficient_values')})",
        )
    add(1, f"{n('row_group')} = {n('lane')} % 16")
    add(1, f"{n('member')} = {n('lane')} // 16")
    add(1, f"{n('local_row')} = {n('warp')} * 16 + {n('row_group')}")
    add(1, f"{n('row')} = {n('row_base')} + {n('local_row')}")
    add(1, f"{n('acc')} = cutlass.Float32(0)")
    invariant_ast = ast.parse("\n".join(program.invariant))
    later_ast = ast.parse("\n".join((*program.element, *program.output)))
    defined = {
        node.id
        for node in ast.walk(invariant_ast)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    read_later = {
        node.id
        for node in ast.walk(later_ast)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for name in sorted(defined & read_later):
        bindings = [node for node, _, value in original.value_bindings if value == name]
        if (
            not bindings
            or any(
                not isinstance(node.meta.get("val"), torch.Tensor) for node in bindings
            )
            or len({node.meta["val"].dtype for node in bindings}) != 1
        ):
            raise InapplicableStateUpdate(
                f"unproved live invariant scalar dtype: {name}, bindings={[(str(node), type(node.meta.get('val')).__name__) for node in bindings]}"
            )
        scalar_dtype = CompileEnvironment.current().backend.dtype_str(
            bindings[0].meta["val"].dtype
        )
        add(1, f"{name} = {scalar_dtype}(0)")
    add(1, f"if {n('row')} < {domain.grid_extents[-1]}:")
    for line in program.invariant or ("pass",):
        add(2, line)
    if publications:
        add(1, f"cute.arch.mbarrier_arrive({n('consumers')})")
        add(1, f"cute.arch.mbarrier_wait({n('consumers')}, 0)")
    consumer_start = len(lines)
    add(1, f"for {n('stage')} in cutlass.range({geometry.stages}):")
    add(2, f"cute.arch.mbarrier_arrive({n('full')} + {n('stage')})")
    add(2, f"cute.arch.mbarrier_wait({n('full')} + {n('stage')}, 0)")
    add(2, f"if {n('row')} < {domain.grid_extents[-1]}:")
    add(3, f"for {n('item')} in cutlass.range_constexpr(0, 16, 2):")
    add(
        4,
        f"{n('column')} = ({n('member')} * 16 + {n('item')} + 2 * ({n('row_group')} // 2)) % 32",
    )
    add(
        4,
        f"{n('byte_offset')} = {n('byte_layout')}(cute.get_nonswizzle_portion({n('shared')}.layout)(({n('local_row')}, {n('column')}, {n('stage')})) * 2)",
    )
    add(
        4,
        f"{n('pointer')} = cute.recast_ptr({n('bytes')} + {n('byte_offset')}, dtype={dtype})",
    )
    add(4, f"{n('state_values')} = _cute_zero_u16_vec(2)")
    add(4, f"if {n('eligible')}:")
    add(
        5,
        f"{n('state_values')} = cute.arch.load({n('pointer')}, ir.VectorType.get([2], cutlass.Uint16.mlir_type))",
    )
    for publication in publications:
        add(
            4,
            f"{pairs[publication.binding]} = cute.arch.load({publication.shared} + {n('stage')} * 32 + {n('column')}, ir.VectorType.get([2], cutlass.Uint16.mlir_type))",
        )
    add(4, f"{n('new_state_values')} = []")
    add(4, f"for {n('component')} in cutlass.range_constexpr(2):")
    add(5, f"{n('feature')} = {n('stage')} * 32 + {n('column')} + {n('component')}")
    replaced = 0
    replaced_coefficients: set[str] = set()
    for line in program.element:
        statement = ast.parse(line).body[0]
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == original.old_value
        ):
            replaced += 1
            line = f"{original.old_value} = (cutlass.Uint16({n('state_values')}[{n('component')}]).bitcast({dtype}) if {n('eligible')} else {ast.unparse(statement.value)})"
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in pairs
        ):
            binding = statement.targets[0].id
            publication = next(p for p in publications if p.binding == binding)
            load_dtype = CompileEnvironment.current().backend.dtype_str(
                publication.access.dtype
            )
            line = f"{binding} = cutlass.Uint16({pairs[binding]}[{n('component')}]).bitcast({load_dtype})"
            if binding in replaced_coefficients:
                raise InapplicableStateUpdate("coefficient load has multiple bindings")
            replaced_coefficients.add(binding)
        add(5, line)
    if replaced != 1:
        raise InapplicableStateUpdate(
            "old state binding is not one exact original load"
        )
    if replaced_coefficients != set(pairs):
        raise InapplicableStateUpdate("coefficient publication lacks its original read")
    add(5, f"{n('acc')} = {n('acc')} + cutlass.Float32({original.reduction_input})")
    add(
        5,
        f"{n('new_state_values')}.append({dtype}({original.storage_value}).bitcast(cutlass.Uint16))",
    )
    add(5, f"if not {n('eligible')}:")
    add(6, f"if {program.state_access.predicate or 'True'}:")
    add(7, f"{program.state_access.pointer}.store({dtype}({original.storage_value}))")
    add(4, f"if {n('eligible')}:")
    add(5, f"_cute_store_u16_vec({n('pointer')}, {n('new_state_values')})")
    add(2, f"cute.arch.mbarrier_arrive({n('empty')} + {n('stage')})")
    consumer = ast.parse("\n".join(row[4:] for row in lines[consumer_start:])).body
    if len(consumer) != 1 or not isinstance(consumer[0], ast.For):
        raise InapplicableStateUpdate("consumer is not one original stage loop")
    split = _split_consumer_transport(consumer[0], n("eligible"))
    lines[consumer_start:] = [
        "    " + row
        for row in ast.unparse(ast.fix_missing_locations(split)).splitlines()
    ]
    add(1, f"{n('neighbor')} = cute.arch.shuffle_sync_bfly({n('acc')}, 16)")
    add(1, f"{n('sum')} = {n('acc')} + {n('neighbor')}")
    add(1, f"if {n('member')} == 0 and {n('row')} < {domain.grid_extents[-1]}:")
    for line in program.output:
        add(2, line)
    add(2, f"if {program.output_access.predicate or 'True'}:")
    output_dtype = CompileEnvironment.current().backend.dtype_str(
        program.output_access.dtype
    )
    add(
        3,
        f"{program.output_access.pointer}.store({output_dtype}({program.output_value}))",
    )
    add(0, "cute.arch.sync_threads()")
    parameters = [
        n(x) for x in ("load_atom", "load_tensor", "store_atom", "store_tensor")
    ]
    df.wrapper_only_params.extend(parameters)
    from .horizontal_state_expression import StateExpression

    argument_expression = StateExpression(cg, domain, outer)
    hosts = tuple(
        dict.fromkeys(
            [
                region.state_store.args[0],
                region.output_store.args[0],
                *(load.args[0] for load in region.loads),
            ]
        )
    )
    arguments = tuple(
        argument_expression.tensor_name(cast("Node", host)) for host in hosts
    )
    metadata = [
        (
            tuple(_proven_int(x) for x in cast("Node", host).meta["val"].shape),
            tuple(_proven_int(x) for x in cast("Node", host).meta["val"].stride()),
            str(cast("Node", host).meta["val"].dtype),
        )
        for host in hosts
    ]
    if any(x is None for shape, strides, _ in metadata for x in (*shape, *strides)):
        raise InapplicableStateUpdate(
            "horizontal host metadata is not statically proved"
        )
    cg.cute_wrapper_plans.append(
        {
            "kind": "horizontal_state_tma",
            "metadata_names": list(arguments),
            "metadata": metadata,
            "write_names": list(arguments[:2]),
            "readonly_names": list(arguments[2:]),
            "aligned_names": list(
                dict.fromkeys(
                    p.access.tensor for p in publications if p.vector_width > 1
                )
            ),
            "state_name": original.old_state.tensor,
            "state_shape": list(domain.state_shape),
            "state_strides": list(domain.state_strides),
            "rows": geometry.rows,
            "features": geometry.stage_features,
            "stages": geometry.stages,
            "dtype": dtype,
            "kernel_args": parameters,
        }
    )
    df.preamble = []
    df.body = list(ast.parse("\n".join(lines)).body)
    program.check(cg)
    for publication in publications:
        publication.check(cg, program)
    installed = InstalledHorizontalState(
        plan,
        program,
        publications,
        tuple(map(ast.dump, df.body)),
        _freeze(cg.cute_wrapper_plans),
        tuple(df.wrapper_only_params),
    )
    installed = dataclasses.replace(installed, selection=installed.facts())
    df.cute_state.horizontal_state_body = installed
    installed.check(cg)
    return True
