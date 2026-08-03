"""Composable SparseCore target integration for the Pallas backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..program_id import ProgramIDs
from ..tile_strategy import DeviceGridState
from ..tile_strategy import NDTileStrategy
from .access import ACCESS_SITE_META
from .access import AccessSite
from .sc_base import SC_SLICE_MULTIPLE
from .sc_base import _reject
from .sc_base import sc_hardware_info
from .sparsecore_access import AccessLoweringContext
from .sparsecore_access import lower_sparsecore_access
from .sparsecore_program import SparseCoreGeometry
from .sparsecore_program import SparseCoreProgram
from .sparsecore_program import schedule_sparsecore_program

if TYPE_CHECKING:
    import ast

    import torch

    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import Config
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


def _static_block_sizes(config: Config) -> dict[int, int]:
    from ..compile_environment import CompileEnvironment

    result: dict[int, int] = {}
    for block_id, info in enumerate(CompileEnvironment.current().block_sizes):
        value = info.from_config(config)
        if isinstance(value, int):
            result[block_id] = value
    return result


def _validate_compute(graph: torch.fx.Graph, access_nodes: set[torch.fx.Node]) -> None:
    from .sparsecore_compute import supports_sparsecore_compute_node

    for node in graph.nodes:
        if node in access_nodes or supports_sparsecore_compute_node(node):
            continue
        _reject(
            "compute_op",
            (
                f"FX node kind {node.op!r} is not implemented"
                if node.op != "call_function"
                else f"operation {node.target} has no SparseCore implementation"
            ),
            node=node,
            operation=getattr(node.target, "__name__", str(node.target)),
        )


def build_sparsecore_program(
    graphs: list[GraphInfo], config: Config
) -> SparseCoreProgram:
    """Build an SC program for one tiled loop."""
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction

    host = HostFunction.current()
    device_ir = host.device_ir
    if len(device_ir.root_ids) != 1:
        _reject("control_flow", "SparseCore supports one root tiled loop")
    root_graph = graphs[device_ir.root_ids[0]].graph
    root_blocks = device_ir.grid_block_ids[0]
    if not root_blocks:
        _reject("control_flow", "SparseCore root loop has no tiled item axis")
    item_block_id = root_blocks[0]
    block_sizes = _static_block_sizes(config)
    tile_size = block_sizes.get(item_block_id)
    if tile_size is None:
        _reject("dynamic_shape", "SparseCore item block size must be static")

    access_sites = [
        site
        for node in root_graph.nodes
        if isinstance((site := node.meta.get(ACCESS_SITE_META)), AccessSite)
    ]
    if not access_sites:
        _reject("access_pattern", "SparseCore kernel has no memory accesses")
    item_count = CompileEnvironment.current().block_sizes[item_block_id].size_hint()
    mesh = sc_hardware_info()
    if mesh is None:
        _reject("control_flow", "active target has no SparseCore mesh")
    num_cores, num_subcores = mesh
    total_subcores = num_cores * num_subcores
    if tile_size % total_subcores:
        _reject(
            "control_flow",
            f"item tile {tile_size} must be divisible by {total_subcores} subcores",
        )
    items_per_subcore = tile_size // total_subcores
    context = AccessLoweringContext(
        block_sizes=block_sizes,
        item_block_id=item_block_id,
        items_per_subcore=items_per_subcore,
    )
    accesses = tuple(lower_sparsecore_access(site, context) for site in access_sites)
    for access in accesses:
        if (
            access.stream is not None
            and access.stream.elements_per_item * items_per_subcore % SC_SLICE_MULTIPLE
        ):
            _reject(
                "layout",
                f"stream has {access.stream.elements_per_item * items_per_subcore} "
                f"values per subcore; expected a multiple of {SC_SLICE_MULTIPLE}",
                node=access.site.node,
            )
    _validate_compute(root_graph, {site.node for site in access_sites})
    program = SparseCoreProgram(
        root_graph,
        SparseCoreGeometry(
            item_block_id,
            item_count,
            tile_size,
            num_cores,
            num_subcores,
        ),
        accesses,
    )
    schedule_sparsecore_program(program)
    return program


def detect_sparsecore_search(device_ir: DeviceIR, config_spec: ConfigSpec) -> None:
    """Offer SC when one root graph has a tiled memory stream."""
    from ...language import memory_ops
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    if env.settings.pallas_interpret or sc_hardware_info() is None:
        return
    if len(device_ir.root_ids) != 1 or not device_ir.grid_block_ids[0]:
        return
    graph = device_ir.graphs[device_ir.root_ids[0]].graph
    has_stream = any(
        node.op == "call_function" and node.target is memory_ops.load
        for node in graph.nodes
    )
    if has_stream:
        config_spec.sparsecore_search_enabled = True  # type: ignore[attr-defined]


@dataclass
class _SparseCoreProgramIDs(ProgramIDs):
    blocks: int = 1

    def codegen(self, state: CodegenState) -> None:
        return None

    def codegen_grid(self) -> ast.AST:
        from ..ast_extension import expr_from_string

        return expr_from_string(f"({self.blocks},)")


@dataclass
class _SparseCoreGridState(DeviceGridState):
    program: SparseCoreProgram | None = None

    def codegen_graph(self, codegen: GenerateAST, graph: torch.fx.Graph) -> None:
        if self.program is None or graph is not self.program.graph:
            raise AssertionError("SparseCore grid received the wrong root graph")
        from .sparsecore_codegen import codegen_sparsecore_program

        codegen_sparsecore_program(self.program, codegen)


class SparseCoreTileStrategy(NDTileStrategy):
    """Tiled loop strategy for SparseCore."""

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        from ..compile_environment import CompileEnvironment

        program = CompileEnvironment.current().sparsecore_program
        if program is None:
            raise AssertionError("SparseCore config has no program")
        state.device_function.set_pid(
            _SparseCoreProgramIDs(blocks=program.geometry.blocks)
        )
        return _SparseCoreGridState(
            self,
            self._create_block_id_info_dict(state),
            program=program,
        )
