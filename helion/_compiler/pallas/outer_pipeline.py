from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import cast

import torch

from ... import exc
from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import CodegenState


# Keep this local for now to avoid widening shared AST-helper APIs in this PR.
def _clone_ast_value(value: object) -> object:
    """Deep-copy an AST value while preserving Helion ``ExtendedAST`` metadata."""
    if isinstance(value, list):
        return [_clone_ast_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_ast_value(item) for item in value)
    if isinstance(value, ast.AST):
        fields = {
            field: _clone_ast_value(getattr(value, field)) for field in value._fields
        }
        if isinstance(value, ExtendedAST):
            return value.copy(**fields)
        return ast.copy_location(type(value)(**fields), value)
    return value


def _clone_expr(expr: ast.AST) -> ast.AST:
    cloned = _clone_ast_value(expr)
    assert isinstance(cloned, ast.AST)
    return cloned


class PrologueTensorPipelineInput(NamedTuple):
    """A per-outer-step prologue tensor promoted to an emit_pipeline input."""

    fake: torch.Tensor
    block_spec: str
    vmem_name: str
    outer_dim: int
    offset_name: str


@dataclasses.dataclass(frozen=True)
class PipelineExpr:
    """AST-backed expression used by folded Pallas pipeline axes."""

    expr: ast.AST

    @classmethod
    def from_string(cls, expr: str) -> PipelineExpr:
        return cls(expr_from_string(expr))

    def render_body(self) -> str:
        return ast.unparse(self.expr)

    def render_lambda(self, context: PipelineContext) -> str:
        return context.resolve_for_lambda(self)


@dataclasses.dataclass(frozen=True)
class PipelineAxis:
    """One folded dimension of a generated Pallas pipeline grid."""

    block_id: int
    kind: Literal["parallel", "ordered"]
    begin: PipelineExpr
    end: PipelineExpr
    max_tiles: int | PipelineExpr
    step: PipelineExpr
    block_extent: str
    lambda_param: str
    pid_var: str | None = None
    offset_var: str | None = None
    index_var: str | None = None

    @property
    def dimension_semantic(self) -> str:
        return "parallel" if self.kind == "parallel" else "arbitrary"

    def max_tiles_expr(self) -> str:
        if isinstance(self.max_tiles, int):
            return str(self.max_tiles)
        return self.max_tiles.render_body()

    def start_expr(self) -> str:
        return (
            f"({self.begin.render_body()}) + "
            f"({self.lambda_param}) * ({self.step.render_body()})"
        )

    def next_start_expr(self) -> str:
        return f"({self.start_expr()}) + ({self.step.render_body()})"


def _store_names(node: ast.AST) -> set[str]:
    """Names assigned (``Store`` context) anywhere within ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _single_name_assign(stmt: ast.AST) -> tuple[str, ast.AST] | None:
    """Return ``(name, value)`` for ``name = <expr>`` statements.

    Scalar prologue assignments are replayed inside the pipeline body and also
    inlined into BlockSpec lambdas so data-dependent folded bounds like
    ``start = offsets[seq]`` can be expressed in the lambda's scope.
    """
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id, stmt.value
    return None


@dataclasses.dataclass
class PipelineContext:
    """State for folding an outer grid into a Pallas emit_pipeline grid.

    This context is created before grid-body codegen so the body can be emitted
    directly in its final pipeline scope.  It is deliberately structural state,
    not a second plan IR.
    """

    axes: list[PipelineAxis] = dataclasses.field(default_factory=list)
    # Grid-body statements emitted before the folded loop: replayed inside the
    # pipeline body, with their defined names (a leak-check for lambda rewrites)
    # and single-name assigns (inlined into BlockSpec lambdas; see
    # _outer_lambda_expr) tracked alongside.
    prologue_replay_stmts: list[ast.AST] = dataclasses.field(default_factory=list)
    prologue_defined_names: set[str] = dataclasses.field(default_factory=set)
    prologue_assigns: dict[str, ast.AST] = dataclasses.field(default_factory=dict)
    # True while the grid-body prologue is being captured (set False once the
    # folded loop's codegen begins).
    capturing_prologue_replay: bool = False
    # True only while _codegen_emit_pipeline emits the generated function and
    # emit_pipeline call into the outer scope; user statements after the folded
    # loop should reject instead of being appended there.
    emitting_folded_pipeline: bool = False
    pipeline_index_var: str = "_pipeline_indices"

    @property
    def enabled(self) -> bool:
        return bool(self.axes)

    @property
    def grid_parts(self) -> list[str]:
        return [axis.max_tiles_expr() for axis in self.axes]

    @property
    def dimension_semantics(self) -> tuple[str, ...]:
        return tuple(axis.dimension_semantic for axis in self.axes)

    @property
    def axis_by_block_id(self) -> dict[int, PipelineAxis]:
        return {axis.block_id: axis for axis in self.axes}

    @property
    def ordered_axis(self) -> PipelineAxis | None:
        ordered = [axis for axis in self.axes if axis.kind == "ordered"]
        if not ordered:
            return None
        assert len(ordered) == 1
        return ordered[0]

    @property
    def outer_axes(self) -> list[PipelineAxis]:
        return [axis for axis in self.axes if axis.pid_var is not None]

    @property
    def folded_axes(self) -> list[PipelineAxis]:
        return [axis for axis in self.axes if axis.pid_var is None]

    @property
    def outer_block_ids(self) -> list[int]:
        return [axis.block_id for axis in self.outer_axes]

    @property
    def outer_grid_extents(self) -> list[str]:
        return [axis.max_tiles_expr() for axis in self.outer_axes]

    @property
    def outer_pid_vars(self) -> list[str]:
        return [axis.pid_var for axis in self.outer_axes if axis.pid_var is not None]

    @property
    def outer_offset_vars(self) -> list[str]:
        return [
            axis.offset_var for axis in self.outer_axes if axis.offset_var is not None
        ]

    @property
    def outer_index_vars(self) -> list[str]:
        return [
            axis.index_var for axis in self.outer_axes if axis.index_var is not None
        ]

    @property
    def folded_block_ids(self) -> list[int]:
        return [axis.block_id for axis in self.folded_axes]

    @property
    def folded_max_tiles(self) -> list[int | PipelineExpr]:
        return [axis.max_tiles for axis in self.folded_axes]

    def add_folded_axis(self, axis: PipelineAxis) -> None:
        if axis.pid_var is not None:
            raise AssertionError("folded pipeline axes must not carry pid vars")
        if axis.block_id in self.axis_by_block_id:
            raise exc.BackendUnsupported(
                "pallas",
                "outer_pipeline cannot fold the same block id more than once",
            )
        self.axes.append(axis)

    def resolve_for_lambda(self, expr: PipelineExpr | ast.AST | str) -> str:
        """Render *expr* in BlockSpec lambda scope.

        Replaces axis-local pid vars with lambda params, offset vars with the
        precise start expression for that lambda coordinate, and recursively
        inlines captured scalar prologue assignments so expressions like
        ``start = offsets[seq]`` are self-contained in the BlockSpec lambda.
        Axis index vars are intentionally not rewritten here: they can be
        vector/lane expressions and must not silently collapse to a scalar
        pipeline index.
        """
        if isinstance(expr, PipelineExpr):
            expr_ast = _clone_expr(expr.expr)
        elif isinstance(expr, ast.AST):
            expr_ast = _clone_expr(expr)
        else:
            expr_ast = expr_from_string(expr)

        replacements: dict[str, ast.AST] = {}
        for axis in self.axes:
            if axis.pid_var is not None:
                replacements[axis.pid_var] = ast.Name(
                    id=axis.lambda_param, ctx=ast.Load()
                )
            if axis.offset_var is not None:
                replacements[axis.offset_var] = expr_from_string(axis.start_expr())

        assigns = self.prologue_assigns

        class Resolver(ast.NodeTransformer):
            def __init__(self) -> None:
                self.stack: set[str] = set()

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if not isinstance(node.ctx, ast.Load):
                    return node
                if node.id in replacements:
                    return self.visit(_clone_expr(replacements[node.id]))
                if node.id in assigns and node.id not in self.stack:
                    self.stack.add(node.id)
                    try:
                        return self.visit(_clone_expr(assigns[node.id]))
                    finally:
                        self.stack.remove(node.id)
                return node

        resolved = Resolver().visit(expr_ast)
        ast.fix_missing_locations(resolved)
        remaining_names = {
            child.id
            for child in ast.walk(resolved)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        body_local_names = (
            set(self.prologue_defined_names)
            | set(self.outer_pid_vars)
            | set(self.outer_offset_vars)
            | set(self.outer_index_vars)
        )
        unresolved = sorted(remaining_names & body_local_names)
        if unresolved:
            raise exc.BackendUnsupported(
                "pallas",
                "outer_pipeline BlockSpec cannot reference body-local folded "
                f"bound expression(s): {', '.join(unresolved)}",
            )
        return ast.unparse(resolved)

    def capture_prologue_statement(self, stmt: ast.AST) -> None:
        """Record a grid-body statement (emitted before the folded loop) for
        replay in the pipeline body, tracking its defined names and any
        single-name assignment for later BlockSpec-lambda inlining."""
        assignment = _single_name_assign(stmt)
        if assignment is None:
            raise exc.BackendUnsupported(
                "pallas",
                "outer_pipeline prologue capture only supports simple single-name "
                "assignments before the folded hl.tile loop",
            )
        name, value = assignment
        self.prologue_replay_stmts.append(stmt)
        self.prologue_defined_names.update(_store_names(stmt))
        self.prologue_assigns[name] = value


OuterPipelineContext = PipelineContext


def _slice_elements(slice_node: ast.AST) -> list[ast.AST]:
    """The per-dim index nodes of a subscript (unwraps a tuple subscript)."""
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def _is_full_slice(node: ast.AST) -> bool:
    """True if ``node`` is a bare ``[:]`` slice (no lower/upper/step)."""
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _normalize_prologue_tensor_dims(
    slice_node: ast.AST, rank: int
) -> list[ast.AST] | None:
    """Pad a subscript's index list with trailing full slices up to ``rank``."""
    dims = _slice_elements(slice_node)
    if len(dims) > rank:
        return None
    return [
        *dims,
        *[
            ast.Slice(lower=None, upper=None, step=None)
            for _ in range(rank - len(dims))
        ],
    ]


def _dim_uses_any_name(dim: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in names for node in ast.walk(dim)
    )


def _has_outer_axis_dim(
    outer_context: OuterPipelineContext,
    dims: list[ast.AST] | None,
) -> bool:
    """True if any dim uses a folded outer-grid offset or tile-index var."""
    if dims is None:
        return False
    axis_names = set(outer_context.outer_offset_vars) | set(
        outer_context.outer_index_vars
    )
    return any(_dim_uses_any_name(dim, axis_names) for dim in dims)


def _has_outer_index_dim(
    outer_context: OuterPipelineContext,
    dims: list[ast.AST] | None,
) -> bool:
    """True if any dim uses a folded outer-grid tile/vector index var."""
    if dims is None:
        return False
    index_names = set(outer_context.outer_index_vars)
    return any(_dim_uses_any_name(dim, index_names) for dim in dims)


def _raise_unsupported_outer_prologue_access(hbm_name: str) -> None:
    raise exc.BackendUnsupported(
        "pallas",
        "unsupported outer_pipeline prologue tensor access to "
        f"{hbm_name}; only scalar tensor[outer_offset, :, ...] with full "
        "non-outer dimensions can be promoted to an emit_pipeline input. "
        "Tile-vector outer indices are not supported in prologue tensor accesses.",
    )


def _outer_offset_dim(
    outer_context: OuterPipelineContext,
    dims: list[ast.AST],
) -> tuple[int, str] | None:
    """Detect ``tensor[outer_offset, :, ...]`` in captured prologue reads."""
    result: tuple[int, str] | None = None
    for dim, idx in enumerate(dims):
        if isinstance(idx, ast.Name) and idx.id in outer_context.outer_offset_vars:
            if result is not None:
                return None
            result = (dim, idx.id)
        elif not _is_full_slice(idx):
            return None
    return result


def _make_outer_prologue_block_spec(
    fake: torch.Tensor,
    outer_dim: int,
    offset_name: str,
    *,
    outer_lambda_params: list[str],
    outer_lambda_param_by_block: dict[int, str],
    inner_lambda_params: list[str],
    outer_lambda_expr: Callable[[str, dict[int, str]], str],
    buffered_block_spec: Callable[[list[str], list[str], list[str]], str],
) -> str:
    """BlockSpec for a per-outer-step prologue operand such as ``k[seq]``."""
    lambda_params = [*outer_lambda_params, *inner_lambda_params]
    outer_index_expr = outer_lambda_expr(offset_name, outer_lambda_param_by_block)

    block_shape_parts: list[str] = []
    lambda_parts: list[str] = []
    for dim, size in enumerate(fake.shape):
        if dim == outer_dim:
            block_shape_parts.append("1")
            lambda_parts.append(outer_index_expr)
        else:
            if not isinstance(size, int):
                raise exc.BackendUnsupported(
                    "pallas",
                    "outer_pipeline prologue tensor pipelining currently "
                    "requires static non-outer tensor dimensions",
                )
            block_shape_parts.append(str(int(size)))
            lambda_parts.append("0")
    return buffered_block_spec(block_shape_parts, lambda_params, lambda_parts)


class _PrologueTensorPipelineInputCollector(ast.NodeVisitor):
    """Find ``tensor[outer_offset, :, ...]`` prologue reads to pipeline."""

    def __init__(
        self,
        *,
        state: CodegenState,
        outer_context: OuterPipelineContext,
        tensor_by_name: dict[str, torch.Tensor],
        prologue_inputs: dict[str, PrologueTensorPipelineInput],
        outer_lambda_params: list[str],
        outer_lambda_param_by_block: dict[int, str],
        inner_lambda_params: list[str],
        outer_lambda_expr: Callable[[str, dict[int, str]], str],
        buffered_block_spec: Callable[[list[str], list[str], list[str]], str],
    ) -> None:
        self.state = state
        self.outer_context = outer_context
        self.tensor_by_name = tensor_by_name
        self.prologue_inputs = prologue_inputs
        self.outer_lambda_params = outer_lambda_params
        self.outer_lambda_param_by_block = outer_lambda_param_by_block
        self.inner_lambda_params = inner_lambda_params
        self.outer_lambda_expr = outer_lambda_expr
        self.buffered_block_spec = buffered_block_spec

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return
        fake = self.tensor_by_name.get(node.value.id)
        if fake is None:
            return
        hbm_name = node.value.id
        dims = _normalize_prologue_tensor_dims(node.slice, len(fake.shape))
        if dims is None:
            raw_dims = _slice_elements(node.slice)
            if _has_outer_axis_dim(self.outer_context, raw_dims):
                _raise_unsupported_outer_prologue_access(hbm_name)
            return
        all_scalar_dims = all(not _is_full_slice(dim) for dim in dims)
        if all_scalar_dims:
            if _has_outer_index_dim(self.outer_context, dims):
                _raise_unsupported_outer_prologue_access(hbm_name)
            return
        outer_info = _outer_offset_dim(self.outer_context, dims)
        if outer_info is None:
            _raise_unsupported_outer_prologue_access(hbm_name)
        assert outer_info is not None
        outer_dim, offset_name = outer_info
        existing = self.prologue_inputs.get(hbm_name)
        if existing is not None:
            if existing.outer_dim != outer_dim or existing.offset_name != offset_name:
                raise exc.BackendUnsupported(
                    "pallas",
                    "outer_pipeline cannot pipeline conflicting "
                    f"outer-prologue slices of {hbm_name}",
                )
            return
        vmem_name = self.state.device_function.new_var(
            hbm_name.replace("_hbm", "") + "_vmem"
        )
        self.prologue_inputs[hbm_name] = PrologueTensorPipelineInput(
            fake,
            _make_outer_prologue_block_spec(
                fake,
                outer_dim,
                offset_name,
                outer_lambda_params=self.outer_lambda_params,
                outer_lambda_param_by_block=self.outer_lambda_param_by_block,
                inner_lambda_params=self.inner_lambda_params,
                outer_lambda_expr=self.outer_lambda_expr,
                buffered_block_spec=self.buffered_block_spec,
            ),
            vmem_name,
            outer_dim,
            offset_name,
        )


def collect_prologue_tensor_pipeline_inputs(
    *,
    state: CodegenState,
    outer_context: OuterPipelineContext,
    outer_lambda_params: list[str],
    outer_lambda_param_by_block: dict[int, str],
    inner_lambda_params: list[str],
    outer_lambda_expr: Callable[[str, dict[int, str]], str],
    buffered_block_spec: Callable[[list[str], list[str], list[str]], str],
) -> dict[str, PrologueTensorPipelineInput]:
    """Collect prologue tensor reads that should become emit_pipeline inputs."""
    tensor_by_name = {
        arg.name: fake for fake, arg in state.device_function._tensor_args.items()
    }
    prologue_inputs: dict[str, PrologueTensorPipelineInput] = {}
    collector = _PrologueTensorPipelineInputCollector(
        state=state,
        outer_context=outer_context,
        tensor_by_name=tensor_by_name,
        prologue_inputs=prologue_inputs,
        outer_lambda_params=outer_lambda_params,
        outer_lambda_param_by_block=outer_lambda_param_by_block,
        inner_lambda_params=inner_lambda_params,
        outer_lambda_expr=outer_lambda_expr,
        buffered_block_spec=buffered_block_spec,
    )
    for stmt in outer_context.prologue_replay_stmts:
        collector.visit(stmt)
    return prologue_inputs


class _PrologueTensorPipelineInputRemapper(ast.NodeTransformer):
    """Rewrite registered prologue tensor reads to their VMEM refs."""

    def __init__(
        self,
        outer_context: OuterPipelineContext,
        prologue_inputs: dict[str, PrologueTensorPipelineInput],
    ) -> None:
        self.outer_context = outer_context
        self.prologue_inputs = prologue_inputs

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        remap = self.prologue_inputs.get(node.value.id)
        if remap is None:
            return node
        fake = remap.fake
        dims = _normalize_prologue_tensor_dims(node.slice, len(fake.shape))
        if dims is None:
            return node
        outer_info = _outer_offset_dim(self.outer_context, dims)
        if outer_info is None:
            return node
        outer_dim, _offset_name = outer_info
        dims[outer_dim] = ast.Constant(value=0)
        new_slice: ast.expr
        if len(dims) == 1:
            new_slice = cast("ast.expr", dims[0])
        else:
            new_slice = ast.Tuple(elts=cast("list[ast.expr]", dims), ctx=ast.Load())
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=remap.vmem_name, ctx=ast.Load()),
                slice=new_slice,
                ctx=node.ctx,
            ),
            node,
        )


def remap_prologue_tensor_pipeline_inputs(
    outer_context: OuterPipelineContext,
    prologue_inputs: dict[str, PrologueTensorPipelineInput],
) -> list[ast.AST]:
    """Replay captured prologue statements, remapping pipelined tensor reads."""
    remapper = _PrologueTensorPipelineInputRemapper(outer_context, prologue_inputs)
    body_stmts: list[ast.AST] = []
    for stmt in outer_context.prologue_replay_stmts:
        stmt_copy = _clone_ast_value(stmt)
        assert isinstance(stmt_copy, ast.AST)
        remapped = remapper.visit(stmt_copy)
        assert isinstance(remapped, ast.AST)
        ast.fix_missing_locations(remapped)
        body_stmts.append(remapped)
    return body_stmts


def static_cdiv(number: int, denom: int) -> int:
    """``ceil(number / denom)`` for positive ``denom`` (>= 1); the folded
    ``max_tiles`` count for a pipeline grid dim."""
    if denom <= 0:
        raise ValueError(f"outer_pipeline block size must be positive, got {denom}")
    return max(1, -(-number // denom))


def declared_or_static_bound(
    env: CompileEnvironment,
    block_id: int,
) -> int | None:
    """Return the static upper bound for a folded tile, if one exists."""
    info = env.block_sizes[block_id]
    if isinstance(info.size, int):
        return info.size
    if isinstance(info.max_extent, int):
        return info.max_extent
    if isinstance(info.size, torch.SymInt):
        if not env.settings.static_shapes:
            raise exc.BackendUnsupported(
                "pallas",
                "outer_pipeline requires a static folded tile extent when "
                "static_shapes=False; pass a specialized integer max_extent "
                "or use pallas_loop_type='fori_loop'.",
            )
        return int(env.size_hint(info.size))
    if isinstance(info.max_extent, torch.SymInt):
        if not env.settings.static_shapes:
            raise exc.BackendUnsupported(
                "pallas",
                "outer_pipeline requires a static folded max_extent when "
                "static_shapes=False; pass a specialized integer "
                "max_extent or use pallas_loop_type='fori_loop'.",
            )
        return int(env.size_hint(info.max_extent))
    return None


def max_tiles_for_block(env: CompileEnvironment, block_id: int, block_size: int) -> int:
    """Static pipeline-grid extent for a folded tile: ``cdiv(bound, block)``,
    where ``bound`` is the loop's static extent or declared ``max_extent``. A
    data-dependent extent with neither is rejected (no silent tile-dropping)."""
    bound = declared_or_static_bound(env, block_id)
    if bound is None:
        from ... import exc

        raise exc.InvalidConfig(
            "outer_pipeline over a data-dependent hl.tile(begin, end) requires "
            "hl.tile(..., max_extent=<static bound>)."
        )
    return static_cdiv(bound, block_size)
