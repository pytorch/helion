from __future__ import annotations

import abc
import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import sympy
import torch

from .. import exc
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import binding_names
from .compile_environment import CompileEnvironment
from .cross_loop_dsmem import cluster_size
from .cross_loop_dsmem import dsmem_transports
from .cross_loop_dsmem import dsmem_worker_count
from .device_function import DeviceFunction
from .device_function import TensorArg
from .host_function import HostFunction
from .variable_origin import TensorSizeOrigin


def _bounded_l2_group_width(
    group: int, num_m: str, static_m: int | None, static_n: int | None
) -> str | None:
    """Cap a short M group before its width is multiplied by the N count."""
    if static_n is not None and max(static_n, 1) * group < 2**31:
        return None  # The ordinary literal product already fits in Int32.
    # Every legal grid count fits signed Int64. A larger requested group
    # can only be the single, short group covering all live rows.
    group = min(group, 2**63 - 1)
    return f"min({num_m}, {group})" if static_m is None else str(min(static_m, group))


def _program_id_type(env: CompileEnvironment) -> str:
    # PID division uses signed axis counts. A narrow or unsigned storage
    # index must not change that arithmetic or truncate a physical CTA ID.
    dtype = torch.int64 if torch.iinfo(env.index_dtype).bits == 64 else torch.int32
    return env.backend.index_type_str(dtype)


def root_coordinate_pid(pid: str, dtype: str) -> str:
    """Convert the scheduler ID before computing a tile origin."""
    env = CompileEnvironment.current()
    if dtype == env.index_type() == _program_id_type(env):
        return pid
    return env.backend.cast_expr(pid, dtype)


def typed_program_id(dim: int = 0) -> str:
    """Generate backend-specific program ID expression.

    Triton uses tl.program_id(). CuTe uses block_idx() as the virtual program ID.
    """
    env = CompileEnvironment.current()
    return env.backend.program_id_expr(dim, index_dtype=_program_id_type(env))


def _stmt_name_uses(stmt: ast.AST) -> tuple[set[str], set[str]]:
    """Return ``(reads, writes)`` for the names referenced in ``stmt``."""
    reads: set[str] = set()
    writes: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                writes.add(node.id)
            else:
                reads.add(node.id)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            reads.add(node.target.id)
    return reads, writes


def _prune_dead_side_effect_free_generated_assignments(
    statements: list[ast.stmt], live_out: set[str]
) -> tuple[list[ast.stmt], set[str]]:
    """Prune dead pure name assignments and return their live-in names.

    ``statements`` must be straight-line assignments with side-effect-free RHSs.
    Compiler-generated reads from immutable metadata tables satisfy that contract.
    """
    live = set(live_out)
    kept: list[ast.stmt] = []
    for stmt in reversed(statements):
        assert (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ), "dead generated-assignment pruning requires simple name assignments"
        target = stmt.targets[0].id
        reads, writes = _stmt_name_uses(stmt)
        assert writes == {target}
        if target not in live:
            continue
        kept.append(stmt)
        live.remove(target)
        live.update(reads)
    kept.reverse()
    return kept, live


def _clone_ast_value(value: object) -> object:
    if isinstance(value, list):
        return [_clone_ast_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_ast_value(item) for item in value)
    if isinstance(value, ast.AST):
        from .tile_dependency import TILE_DEPENDENCY_SITE_ID_ATTR

        fields = {
            field: _clone_ast_value(getattr(value, field)) for field in value._fields
        }
        if isinstance(value, ExtendedAST):
            cloned = value.copy(**fields)
        else:
            cloned = ast.copy_location(type(value)(**fields), value)
        if (site_id := getattr(value, TILE_DEPENDENCY_SITE_ID_ATTR, None)) is not None:
            setattr(cloned, TILE_DEPENDENCY_SITE_ID_ATTR, site_id)
        return cloned
    return value


def _clone_stmt(stmt: ast.stmt) -> ast.stmt:
    return cast("ast.stmt", _clone_ast_value(stmt))


if TYPE_CHECKING:
    from collections.abc import Callable

    # ``(dividend, divisor, static divisor or None) -> expression``
    PidDivisionExpr = Callable[[str, str, int | None], str]
    PidDivisionExprs = tuple[PidDivisionExpr, PidDivisionExpr]
    from .inductor_lowering import CodegenState

NUM_SM_VAR = "_NUM_SM"
MAX_ACTIVE_CLUSTERS_VAR = "_MAX_ACTIVE_CLUSTERS"
NUM_XCD_VAR = "_NUM_XCDS"


def _nonnegative_cdiv(value: str, divisor: str) -> str:
    """Ceiling division without adding to an already full-width count."""
    return f"(({value}) // ({divisor}) + (({value}) % ({divisor}) != 0))"


class PIDInfo(NamedTuple):
    pid_var: str
    block_size_var: str
    numel: sympy.Expr | str  # Can be a sympy.Expr or a string for data-dependent bounds
    block_id: int
    # ``cute_tiles_per_block``: this axis's tiles are packed ``pack`` per CTA
    # on thread axis ``pack_axis`` (the pid count shrinks by ``pack``, which
    # divides the static tile count).
    pack: int = 1
    pack_axis: int = 0
    host_factors: tuple[sympy.Expr | str, ...] = ()

    def num_pids_expr(self, *, is_device: bool, count_dtype: str | None = None) -> str:
        """Get the number of PIDs expression for device or host."""
        # Handle both sympy.Expr and string numel (for data-dependent bounds)
        if self.host_factors:
            fn = DeviceFunction.current()
            # XYZ launches need the raw limit even without a device PID decoder.
            fn.check_flat_grid_factors(self.host_factors)
            if is_device:
                numel_str = (
                    self.numel
                    if isinstance(self.numel, str)
                    else fn.sympy_expr(self.numel)
                )
            else:
                host = HostFunction.current()
                factors = (
                    axis if isinstance(axis, str) else host.sympy_expr(axis, fn.config)
                    for axis in self.host_factors
                )
                numel_str = " * ".join(f"({factor})" for factor in factors)
        elif isinstance(self.numel, str):
            numel_str = self.numel
        elif is_device:
            numel_str = DeviceFunction.current().sympy_expr(self.numel)
        else:
            numel_str = HostFunction.current().sympy_expr(
                self.numel, DeviceFunction.current().config
            )
        count = (
            self.static_num_pids(include_specializations=True) if is_device else None
        )
        # Triton gives a bare literal in this range unsigned type, but a
        # virtual PID divides and takes its remainder in signed Int64.
        if count is not None and 2**31 <= count < 2**32:
            backend = CompileEnvironment.current().backend
            return backend.cast_expr(str(count), backend.dtype_str(torch.int64))
        if count_dtype is not None:
            # Widen the *input* to ceiling division, including data-dependent
            # scalar bounds. A cast of the result cannot recover wrapped add.
            numel_str = CompileEnvironment.current().backend.pid_extent_expr(
                numel_str, count_dtype
            )
        if self.pack > 1:
            assert self.block_size_var == "1"
            return f"(({numel_str}) // {self.pack})"
        if self.block_size_var == "1":
            return numel_str
        if not is_device:
            # Grid dimensions are always non-negative, so we can use integer
            # arithmetic directly instead of a function call like triton.cdiv.
            return f"(({numel_str}) + ({self.block_size_var}) - 1) // ({self.block_size_var})"
        if count is None:
            # A live extent may already use every bit of its signed type. This
            # also applies to host shape arguments the device receives as Int32.
            return _nonnegative_cdiv(numel_str, self.block_size_var)
        width = DeviceFunction.current().resolved_int_constexpr(self.block_size_var)
        if width is not None and count >= (2**31 - 1) // width:
            # Constant folding in an Int32 language (notably MSL) must also
            # avoid overflowing a known, large numerator.
            return str(count)
        return CompileEnvironment.current().backend.cdiv_expr(
            numel_str, self.block_size_var, is_device=is_device
        )

    def static_num_pids(self, *, include_specializations: bool = False) -> int | None:
        """The PID count when both the extent and the block size are static."""
        extent = self.numel
        if (
            include_specializations
            and isinstance(extent, sympy.Expr)
            and extent.free_symbols
        ):
            extent = CompileEnvironment.current().specialize_expr(extent)
        if not isinstance(extent, (int, sympy.Integer)):
            return None
        numel = int(extent)
        if self.pack > 1:
            return numel // self.pack
        if self.block_size_var == "1":
            return numel
        block_size = DeviceFunction.current().resolved_int_constexpr(
            self.block_size_var
        )
        if block_size is None:
            return None
        return -(-numel // block_size)


def _sum_pids(exprs: list[str]) -> str:
    """Sum PID-count expressions, folding all-static terms into one literal."""
    if exprs and all(e.isdigit() for e in exprs):
        return str(sum(int(e) for e in exprs))
    return " + ".join(exprs)


@dataclasses.dataclass
class ProgramIDs(abc.ABC):
    """Base class for all program ID strategies with common functionality."""

    # The CuTe pid decomposition's division form: shifts / masks for the
    # non-negative pids of the tcgen05 persistent kernels, ``//`` / ``%``
    # elsewhere (see ``_decompose_pid_to_statements``).
    def pid_division_exprs(self, state: CodegenState) -> PidDivisionExprs | None:
        """The ``(floordiv, mod)`` expression builders that decompose the
        virtual pid into the per-axis pids, each taking the dividend, the
        divisor and the divisor's static value when known; ``None`` for the
        plain ``//`` / ``%`` forms the analyses evaluate."""
        return None

    # The actual wrapper on this grid root, after tile planning. A scheduler
    # that linearizes the same tiles must use this root's grouping.
    l2_group_size: int = dataclasses.field(
        default=1, init=False, repr=False, compare=False
    )
    shared_pid_var: str | None = None
    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list)

    def append(self, pid: PIDInfo) -> None:
        self.pid_info.append(pid)

    @abc.abstractmethod
    def codegen(self, state: CodegenState) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def codegen_grid(self) -> ast.AST:
        """Generate grid launch expression for kernel execution."""
        raise NotImplementedError

    def device_product(self, factors: list[str], pids: list[PIDInfo]) -> str:
        """Widen products used to decode persistent or flattened physical PIDs."""
        if len(factors) > 1 and (
            self._is_persistent() or isinstance(self, XYZProgramIDs)
        ):
            counts = [pid.static_num_pids() for pid in pids]
            backend = CompileEnvironment.current().backend
            if None not in counts:
                product = math.prod(cast("list[int]", counts))
                if product >= 2**31:
                    # A bare 2**31..2**32-1 literal is UInt32 in Triton;
                    # a signed flat PID cannot divide by that unsigned value.
                    return backend.cast_expr(
                        str(product), backend.dtype_str(torch.int64)
                    )
            else:
                factors = [
                    backend.cast_expr(factors[0], backend.dtype_str(torch.int64)),
                    *factors[1:],
                ]
        return " * ".join(f"({factor})" for factor in factors)

    def host_count_known(self) -> bool:
        fn = DeviceFunction.current()
        return all(
            not isinstance(axis, str) or fn.has_host_grid_value(axis)
            for pid in self.pid_info
            for axis in (pid.host_factors or (pid.numel,))
        )

    def _check_host_scalar_grid_count(self, count: str) -> None:
        if not self.host_count_known():
            return
        # A single input dimension fits a host index by construction. A
        # product of independent axes (even from two tensors), a shifted
        # host length, or an independent Python scalar has no such proof.
        if len(self.pid_info) == 1:
            pid = self.pid_info[0]
            if not pid.host_factors and isinstance(pid.numel, sympy.Symbol):
                info = HostFunction.current().expr_to_origin.get(pid.numel)
                if info is not None and isinstance(info.origin, TensorSizeOrigin):
                    return
        DeviceFunction.current().check_host_grid_count(count)

    def total_pids_expr(
        self, *, is_device: bool, count_dtype: str | None = None
    ) -> str:
        """Get total PIDs expression for device or host."""
        counts = [
            pid.static_num_pids(include_specializations=True) for pid in self.pid_info
        ]
        if 0 in counts:
            return "0"
        if None not in counts:
            count = math.prod(cast("list[int]", counts))
            if count >= 2**63:
                raise exc.InvalidConfig("root grid count exceeds signed Int64")
            if count_dtype:
                return CompileEnvironment.current().backend.cast_expr(
                    str(count), count_dtype
                )
            return str(count)
        factors = [
            pid.num_pids_expr(is_device=is_device, count_dtype=count_dtype)
            for pid in self.pid_info
        ]
        if not is_device or not self._is_persistent():
            product = " * ".join(f"({factor})" for factor in factors)
            if not is_device and not self._is_persistent():
                self._check_host_scalar_grid_count(product)
            return product
        fn = DeviceFunction.current()
        if len(factors) == 1:
            pid = self.pid_info[0]
            if (
                not pid.host_factors
                and isinstance(pid.numel, sympy.Expr)
                and pid.numel.free_symbols
                and self.host_count_known()
            ):
                fn.check_host_grid_count(self.total_pids_expr(is_device=False))
            return factors[0]
        if self.host_count_known():
            fn.check_host_grid_count(self.total_pids_expr(is_device=False))
        else:
            # Counts are nonnegative. Check before multiplying, and let any
            # empty axis annihilate the whole grid rather than trapping.
            backend = CompileEnvironment.current().backend
            limit = backend.uint64_expr(str(2**63 - 1))
            live = [f"(({factor}) != 0)" for factor in factors]
            for i, factor in enumerate(factors[1:], 1):
                prefix = self.device_product(factors[:i], self.pid_info[:i])
                count = counts[i]
                if count == 1 or all(n == 1 for n in counts[:i]):
                    continue
                den = backend.uint64_expr(f"(({factor}) + (({factor}) == 0))")
                overflow = f"({backend.uint64_expr(prefix)} > ({limit} // {den}))"
                fn.check_grid_overflow(" and ".join([overflow, *live]))
        return self.device_product(factors, self.pid_info)

    def grouped_count_dtype(self, group_size: int) -> str | None:
        """Choose one width for both the consumer and its scheduler tile decoder."""
        env = CompileEnvironment.current()
        total = self.total_pids_expr(is_device=True)
        if not total.isdigit() and self.pid_info:
            counts = [
                p.static_num_pids(include_specializations=True) for p in self.pid_info
            ]
            if all(count is not None for count in counts):
                total = str(math.prod(cast("list[int]", counts)))
        # A partial last group may span more coordinates than the real grid.
        bound = str(int(total) * group_size) if total.isdigit() else total
        if env.backend.wide_cumulative_program_ids([bound], blocked=False):
            return env.backend.dtype_str(torch.int64)
        return None

    def static_grid_size(self) -> int | None:
        """The CTA count of the grid ``codegen_grid`` launches, when every
        extent is static (``None`` otherwise).  The flat and xyz strategies
        launch one CTA per program id; the wrappers and the persistent
        strategies override this with what they actually launch."""
        counts = [pid.static_num_pids() for pid in self.pid_info]
        if None in counts:
            return None
        return math.prod(cast("list[int]", counts))

    def decode_cluster_n(self) -> int:
        """The cluster-N width a grouped decode wrapping this strategy must
        respect: 1 unless the strategy pairs CTAs along a cluster N axis."""
        return 1

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Setup persistent kernel if supported. Returns None if not a persistent kernel."""
        return None

    def _setup_persistent_kernel_and_wrap_body(
        self,
        device_function: DeviceFunction,
        virtual_pid_var: str,
        range_expr: str,
        total_pids_expr: str | None = None,
        *,
        relative_pid: str | None = None,
    ) -> list[ast.stmt]:
        """Complete persistent kernel setup: prepare body, wrap in loop, and return."""
        from .ast_extension import create

        # Prepare body for persistent loop
        wrapped_body = list(device_function.body)
        if isinstance(device_function.pid, ForEachProgramID):
            shared_pid_var = device_function.pid.shared_pid_var
            wrapped_body = [
                statement_from_string(
                    f"{shared_pid_var} = {device_function.pid.persistent_pid_value(virtual_pid_var)}"
                ),
                *wrapped_body,
            ]

        # One-shot teardown registered by the body's lowering (a tensor
        # memory dealloc) runs once, after the tile loop.
        wrapped_body, post_loop = (
            CompileEnvironment.current().backend.split_persistent_post_loop(
                device_function, wrapped_body
            )
        )

        if relative_pid is not None:
            wrapped_body.insert(
                0,
                statement_from_string(
                    f"{virtual_pid_var} = {relative_pid} + ({total_pids_expr})"
                ),
            )
        # Create the persistent loop that wraps the entire body
        persistent_loop = create(
            ast.For,
            target=create(
                ast.Name, id=relative_pid or virtual_pid_var, ctx=ast.Store()
            ),
            iter=expr_from_string(range_expr),
            body=wrapped_body,
            orelse=[],
            type_comment=None,
        )
        backend = CompileEnvironment.current().backend
        return [
            backend.persistent_program_loop(persistent_loop, device_function),
            *post_loop,
        ]

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression for this strategy."""
        return typed_program_id(0)

    def _is_persistent(self) -> bool:
        """Check if this is a persistent strategy. Default False."""
        return False

    def _static_empty_coordinates(self) -> list[ast.stmt] | None:
        if len(self.pid_info) > 1 and any(
            pid.static_num_pids() == 0 for pid in self.pid_info
        ):
            # This root never runs. Its emitted body still has to compile;
            # decomposing a physical ID would divide by a static zero extent.
            return [
                statement_from_string(f"{pid.pid_var} = 0") for pid in self.pid_info
            ]
        return None

    def _decompose_pid_to_statements(
        self, pid_var: str, state: CodegenState
    ) -> list[ast.stmt]:
        """Generate statements to decompose a single PID variable into multiple PID components."""
        if (dead := self._static_empty_coordinates()) is not None:
            return dead
        num_blocks = [
            state.device_function.new_var(f"num_blocks_{i}")
            for i in range(len(self.pid_info[:-1]))
        ]
        dtype = self.grouped_count_dtype(1) if num_blocks else None
        statements = []
        for var, pid in zip(num_blocks, self.pid_info[:-1], strict=True):
            count = pid.num_pids_expr(is_device=True, count_dtype=dtype)
            if dtype and not self._is_persistent():
                # Legal flat launch dimensions fit the original index type;
                # only computing the ceiling division needed a wider type.
                env = CompileEnvironment.current()
                if dtype != env.index_type():
                    if pid.pack == 1 and pid.block_size_var == "1":
                        # A named scalar or representable literal has no
                        # intermediate arithmetic. Keep the final index cast.
                        scalar_count = pid.num_pids_expr(is_device=True)
                        if scalar_count.isidentifier() or (
                            scalar_count.isascii()
                            and scalar_count.isdecimal()
                            and int(scalar_count) <= torch.iinfo(env.index_dtype).max
                        ):
                            count = scalar_count
                    count = env.backend.cast_expr(count, env.index_type())
            statements.append(statement_from_string(f"{var} = {count}"))
        # Pids are never negative: a PID strategy may divide them by a static
        # power-of-two pid count with a shift / mask instead of the DSL's
        # floor-division fix-up chains (``pid_division_exprs``; the tcgen05
        # persistent kernels do, their TMA-load warp's prologue is the
        # critical path).  The other kernels keep the ``//`` / ``%`` forms
        # their analyses evaluate (the proven-bounds prover, the gathered-MMA
        # region proof).
        division = self.pid_division_exprs(state)
        static_counts = [
            pid.static_num_pids() if division is not None else None
            for pid in self.pid_info
        ]
        for i, pid in enumerate(self.pid_info):
            expr = pid_var
            if i > 0:
                divisor = self.device_product(num_blocks[:i], self.pid_info[:i])
                if division is not None:
                    static_divisor: int | None = 1
                    for count in static_counts[:i]:
                        static_divisor = (
                            None
                            if static_divisor is None or count is None
                            else static_divisor * count
                        )
                    expr = division[0](expr, divisor, static_divisor)
                else:
                    expr = f"({expr}) // ({divisor})"
            if i + 1 < len(self.pid_info):
                if division is not None:
                    expr = division[1](expr, num_blocks[i], static_counts[i])
                else:
                    expr = f"({expr}) % ({num_blocks[i]})"
            if pid.pack > 1:
                # Thread sub-group ``pack_axis`` of this CTA owns its own tile.
                index_dtype = _program_id_type(CompileEnvironment.current())
                expr = (
                    f"({expr}) * {pid.pack}"
                    f" + {index_dtype}(cute.arch.thread_idx()[{pid.pack_axis}])"
                )
            statements.append(statement_from_string(f"{pid.pid_var} = {expr}"))
        return statements


_PERSISTENT_PHASE = "_helion_persistent_phase"
_ROOT_DISPATCH_TEST = "_helion_root_dispatch_test"


@dataclasses.dataclass(frozen=True)
class PersistentPhase:
    """The scheduler's original loop header, PID reconstruction and range guard.

    The two PID distributions visit every ID in a static [start, stop) on
    some CTA. Keep the proof on the emitted loop: blocked phases reuse one
    induction name, while interleaved phases allocate a fresh name apiece.
    Neither a similar name nor another phase's nonempty domain is evidence.
    The bound inputs must retain their original setup, and a root selector
    must still read the PID from the original dispatch initialisation.
    """

    header: tuple[str, ...]
    nonempty: bool
    inputs: frozenset[str]
    setup: tuple[str, ...]
    dispatch_init: str | None

    @staticmethod
    def _binding(stmt: ast.AST) -> str:
        # Code generation makes an empty else explicit before finalization.
        if isinstance(stmt, ast.If) and all(
            isinstance(n, ast.Pass) for n in stmt.orelse
        ):
            return ast.dump(ast.If(test=stmt.test, body=stmt.body, orelse=[]))
        return ast.dump(stmt)

    @staticmethod
    def _header(loop: ast.For, guard: ast.If) -> tuple[str, ...]:
        return tuple(
            ast.dump(node)
            for node in (loop.target, loop.iter, *loop.body[:-1], guard.test)
        )

    @classmethod
    def mark(
        cls,
        loop: ast.For,
        start: str,
        stop: str,
        setup: list[ast.stmt],
        dispatch_init: ast.AST | None = None,
    ) -> None:
        guard = loop.body[-1]
        assert isinstance(guard, ast.If)
        # Read the header's external inputs, not its local induction or the
        # virtual ID the interleaved prefix computes. A blocked loop also
        # depends on the definitions of start/end and their transitive inputs.
        reads, writes = _stmt_name_uses(
            ast.Tuple(elts=[loop.target, loop.iter, guard.test], ctx=ast.Load())
        )
        for stmt in loop.body[:-1]:
            used, made = _stmt_name_uses(stmt)
            reads.update(used)
            writes.update(made)
        inputs = reads - writes
        selected = []
        for stmt in reversed(setup):
            used, made = _stmt_name_uses(stmt)
            if not inputs.isdisjoint(made):
                inputs.update(used)
                selected.append(cls._binding(stmt))
        setattr(
            loop,
            _PERSISTENT_PHASE,
            cls(
                cls._header(loop, guard),
                start.isdigit() and stop.isdigit() and int(start) < int(stop),
                frozenset(inputs),
                tuple(reversed(selected)),
                ast.dump(dispatch_init) if dispatch_init is not None else None,
            ),
        )

    @classmethod
    def match(cls, loop: ast.For) -> PhaseMatch | None:
        phase = getattr(loop, _PERSISTENT_PHASE, None)
        if not isinstance(phase, cls) or not loop.body or loop.orelse:
            return None
        guard = loop.body[-1]
        if (
            isinstance(guard, ast.If)
            and all(isinstance(stmt, ast.Pass) for stmt in guard.orelse)
            and phase.header == cls._header(loop, guard)
        ):
            return PhaseMatch(phase, guard)
        return None

    @classmethod
    def entered(cls, statements: list[ast.AST]) -> dict[int, int]:
        """Only top-level phases reached through their original bound definitions."""
        result = {}
        writes: dict[int, set[str]] = {}
        for i, stmt in enumerate(statements):
            if not isinstance(stmt, ast.For) or (proof := cls.match(stmt)) is None:
                continue
            phase, guard = proof
            if not phase.nonempty:
                continue
            bindings = []
            for preceding in statements[:i]:
                if id(preceding) not in writes:
                    writes[id(preceding)] = binding_names(preceding)
                if not phase.inputs.isdisjoint(writes[id(preceding)]):
                    bindings.append(preceding)
            if (
                len(bindings) == len(phase.setup)
                and all(
                    cls._binding(binding) == expected
                    for binding, expected in zip(bindings, phase.setup, strict=True)
                )
                and phase.inputs.isdisjoint(binding_names(stmt))
            ):
                result[id(stmt)] = id(guard)
        return result

    def uniform_guards(self, guard: ast.If) -> frozenset[int]:
        """The phase guard and static root selectors reached by its original PID."""
        ids = {id(guard)}
        if not guard.body or self.dispatch_init is None:
            return frozenset(ids)
        init = guard.body[0]
        if not (
            isinstance(init, ast.Assign)
            and len(init.targets) == 1
            and isinstance(init.targets[0], ast.Name)
            and ast.dump(init) == self.dispatch_init
        ):
            return frozenset(ids)
        pid = init.targets[0].id

        def visit(stmts: list[ast.stmt]) -> None:
            for stmt in stmts:
                if (
                    isinstance(stmt, ast.If)
                    and isinstance(test := stmt.test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == pid
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Lt)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and type(test.comparators[0].value) is int
                    and getattr(stmt, _ROOT_DISPATCH_TEST, None) == ast.dump(test)
                ):
                    ids.add(id(stmt))
                    visit(stmt.body)
                    visit(stmt.orelse)
                if pid in binding_names(stmt):
                    break

        visit(guard.body[1:])
        return frozenset(ids)


class PhaseMatch(NamedTuple):
    phase: PersistentPhase
    guard: ast.If


@dataclasses.dataclass
class ForEachProgramID(ProgramIDs):
    """
    Represent multiple top level for loops in the Helion kernel.  Turns into `if` statements in generated code.
    """

    # pyrefly: ignore [bad-override]
    shared_pid_var: str
    cases: list[ProgramIDs] = dataclasses.field(default_factory=list)
    case_bodies: list[list[ast.AST]] = dataclasses.field(default_factory=list)
    case_branches: list[ast.If] = dataclasses.field(default_factory=list)
    case_phases: list[int] = dataclasses.field(default_factory=list)
    uses_phase_loops: bool = dataclasses.field(default=False, init=False)
    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list, init=False)
    barrier_after_root: set[int] = dataclasses.field(default_factory=set)

    def codegen_pid_init(self) -> list[ast.stmt]:
        # Check if persistent kernels are enabled in config - if so, skip regular initialization
        # as it will be handled by the persistent loop wrapper
        from .device_function import DeviceFunction

        current_device_fn = DeviceFunction.current()
        pid_type = current_device_fn.config.get("pid_type", "flat")
        if isinstance(pid_type, str) and pid_type.startswith("persistent"):
            return []
        return [statement_from_string(f"{self.shared_pid_var} = {typed_program_id(0)}")]

    def _wide_schedule(self, counts: list[str]) -> bool:
        # A flat CUDA grid cannot cover more than 2**31 - 1 CTAs. Keep
        # its initialized pid and the values at its dispatch join identical.
        pid_type = DeviceFunction.current().config.get("pid_type", "flat")
        return pid_type in ("persistent_blocked", "persistent_interleaved") and (
            CompileEnvironment.current().backend.wide_cumulative_program_ids(
                counts, blocked=pid_type == "persistent_blocked"
            )
        )

    def device_pid_type(self, counts: list[str] | None = None) -> str:
        env = CompileEnvironment.current()
        if counts is None:
            counts = [case.total_pids_expr(is_device=True) for case in self.cases]
        if self._wide_schedule(counts) or (
            self.cases
            and self.cases[0]._is_persistent()
            and (
                not all(n.isdecimal() for n in counts) or sum(map(int, counts)) >= 2**31
            )
        ):
            return env.backend.dtype_str(torch.int64)
        return _program_id_type(env)

    def persistent_pid_value(self, pid: str) -> str:
        env = CompileEnvironment.current()
        counts = [case.total_pids_expr(is_device=True) for case in self.cases]
        dtype = env.backend.dtype_str(torch.int64)
        # Most ranges already give the induction variable its scheduler type.
        # Only a backend that requested explicit widening needs a cast here.
        if dtype != env.index_type() and self._wide_schedule(counts):
            return env.backend.cast_expr(pid, dtype)
        return pid

    def _device_counts(self, cases: list[ProgramIDs]) -> list[str]:
        counts = [case.total_pids_expr(is_device=True) for case in cases]
        env = CompileEnvironment.current()
        wide = self._wide_schedule(counts)
        if wide or env.backend.wide_cumulative_program_ids(counts, blocked=False):
            # Widen each extent before cdiv; a cast after summation is too
            # late. A legal flat grid fits in Int32, so keep its dispatch
            # subtraction in the same type as the physical block index.
            dtype = env.backend.dtype_str(torch.int64)
            safe_counts = [
                case.total_pids_expr(is_device=True, count_dtype=dtype)
                for case in cases
            ]
            if not wide and dtype != env.index_type():
                return [env.backend.cast_expr(c, env.index_type()) for c in safe_counts]
            return safe_counts
        return counts

    def _cdiv_sum(self, state: CodegenState, exclude_last: bool = False) -> str:
        """Sum the non-empty cdiv expressions from cases (empty if none)."""
        cases = self.cases[:-1] if exclude_last else self.cases
        return self._case_sum(cases, is_device=True)

    def _case_sum(self, cases: list[ProgramIDs], *, is_device: bool) -> str:
        raw = (
            self._device_counts(cases)
            if is_device
            else [case.total_pids_expr(is_device=False) for case in cases]
        )
        parts = [part for part in raw if part]
        total = _sum_pids(parts)
        if total.isdecimal() and int(total) >= 2**63:
            raise exc.InvalidConfig("root grid count exceeds signed Int64")
        if (
            not is_device
            and len(parts) > 1
            and not total.isdecimal()
            and all(case.host_count_known() for case in cases)
        ):
            DeviceFunction.current().check_host_grid_count(total)
        if not (
            is_device
            and len(parts) > 1
            and cases[0]._is_persistent()
            and not total.isdecimal()
        ):
            return total
        fn = DeviceFunction.current()
        backend = CompileEnvironment.current().backend
        if all(case.host_count_known() for case in cases):
            fn.check_host_grid_count(self._case_sum(cases, is_device=False))
        else:
            limit = backend.uint64_expr(str(2**63 - 1))
            # Unlike a Cartesian product, a reached root contributes even if a
            # separate root is empty.  Compare each addend before the sum.
            for i in range(1, len(parts)):
                prefix = _sum_pids([backend.uint64_expr(parts[0]), *parts[1:i]])
                room = f"({limit} - {backend.uint64_expr(parts[i])})"
                fn.check_grid_overflow(f"(({prefix}) > {room})")
        parts[0] = backend.cast_expr(parts[0], backend.dtype_str(torch.int64))
        return _sum_pids(parts)

    def codegen_test(self, state: CodegenState) -> ast.AST:
        return expr_from_string(f"{self.shared_pid_var} < ({self._cdiv_sum(state)})")

    def codegen_dispatch(self) -> list[ast.AST]:
        """Connect the ordered cases without a linear-depth dispatch chain."""

        def emit_cases(begin: int, end: int) -> list[ast.AST]:
            if end - begin == 1:
                return self.case_bodies[begin]
            middle = (begin + end) // 2
            branch = self.case_branches[middle - 1]
            setattr(branch, _ROOT_DISPATCH_TEST, ast.dump(branch.test))
            branch.body = cast("list[ast.stmt]", emit_cases(begin, middle))
            branch.orelse = cast("list[ast.stmt]", emit_cases(middle, end))
            return [branch]

        return emit_cases(0, len(self.case_bodies))

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        total_expr = self.total_pids_expr(is_device=True)
        # If there is only one phase, fall back to existing behavior.
        has_phases = len(set(self.case_phases)) > 1

        def _base_strategy(pid: ProgramIDs) -> ProgramIDs:
            from .tile_strategy import L2GroupingProgramIDs

            if isinstance(pid, L2GroupingProgramIDs):
                assert pid.parent_strategy is not None, (
                    "L2 grouping strategy is missing its parent"
                )
                return pid.parent_strategy
            return pid

        base_strategy = _base_strategy(self.cases[0])

        if not has_phases:
            return base_strategy.setup_persistent_kernel(device_function, total_expr)

        # Nonpersistent backends can own phases as separate launches.
        if not base_strategy._is_persistent():
            return base_strategy.setup_persistent_kernel(device_function, total_expr)

        assert isinstance(base_strategy, PersistentProgramIDs)
        if not base_strategy.is_blocked:
            if not (
                isinstance(base_strategy, PersistentInterleavedProgramIDs)
                and CompileEnvironment.current().backend.barrier_allows_pid_type(
                    "persistent_interleaved"
                )
                and (
                    not HostFunction.current().device_ir.implicit_dependency_starts
                    or device_function.config.cross_loop_pipeline == "barrier"
                )
            ):
                raise exc.InvalidConfig(
                    "interleaved phases require an ordinary persistent grid with local barriers"
                )

        if HostFunction.current().device_ir.implicit_dependency_starts:
            from .cross_loop_codegen import emit_cross_loop_schedule

            return emit_cross_loop_schedule(
                self, base_strategy, device_function, total_expr
            )

        # Delegate to helper for phase-split persistent loops
        return self._emit_phase_loops(base_strategy, device_function, total_expr)

    def total_pids_expr(
        self, *, is_device: bool, count_dtype: str | None = None
    ) -> str:
        """Get total PIDs expression for ForEachProgramID (sum of all pids)."""
        return self._case_sum(self.cases, is_device=is_device)

    def static_grid_size(self) -> int | None:
        """The launch is the persistent grid when the cases are persistent,
        else one CTA per program id of every root (``codegen_grid``)."""
        if not self.cases:
            return None
        if self.cases[0]._is_persistent():
            return self.cases[0].static_grid_size()
        sizes = [case.static_grid_size() for case in self.cases]
        if None in sizes:
            return None
        return sum(cast("list[int]", sizes))

    def codegen(self, state: CodegenState) -> None:
        blocks = self._cdiv_sum(state, exclude_last=True)
        if blocks:
            env = CompileEnvironment.current()
            block_expr = env.backend.cast_expr(f"({blocks})", self.device_pid_type())
            state.codegen.statements_stack[-1].insert(
                0,
                statement_from_string(
                    f"{self.shared_pid_var} -= {block_expr}"
                    if env.backend.supports_augmented_assign
                    else f"{self.shared_pid_var} = {self.shared_pid_var} - {block_expr}"
                ),
            )

    def codegen_grid(self) -> ast.AST:
        # Check if any of the pids is a persistent strategy
        if self.cases[0]._is_persistent():
            # Use SM count grid for persistent kernels
            return self.cases[0].codegen_grid()

        # When persistent kernels are not active, use the full grid size
        return expr_from_string(f"({self.total_pids_expr(is_device=False)},)")

    def _prepare_persistent_body(
        self,
        body: list[ast.AST],
        device_function: DeviceFunction,
        virtual_pid_var: str,
    ) -> list[ast.AST]:
        """Prepare body for persistent loop - handle ForEachProgramID assignment."""
        # In persistent kernels, replace ForEachProgramID init with virtual_pid assignment
        return [
            statement_from_string(
                f"{self.shared_pid_var} = {self.persistent_pid_value(virtual_pid_var)}"
            ),
            *body,
        ]

    def _phase_boundaries(self) -> list[str]:
        """Compute cumulative PID boundaries at phase transitions."""
        boundaries: list[str] = []
        prev_phase = self.case_phases[0]
        for idx in range(len(self.cases)):
            next_phase = (
                self.case_phases[idx + 1]
                if idx + 1 < len(self.case_phases)
                else prev_phase
            )
            if next_phase != prev_phase or idx == len(self.cases) - 1:
                boundaries.append(self._case_sum(self.cases[: idx + 1], is_device=True))
            prev_phase = next_phase
        return boundaries

    def _emit_phase_loops(
        self,
        strategy: PersistentProgramIDs,
        device_function: DeviceFunction,
        total_expr: str,
    ) -> list[ast.stmt]:
        """Emit persistent loops split by KernelPhase boundaries."""
        from .tile_strategy import TileStrategy

        backend = CompileEnvironment.current().backend
        setup = strategy._persistent_setup_statements(total_expr)
        device_function.preamble.extend(setup)

        self.uses_phase_loops = True
        boundaries = self._phase_boundaries()
        block_ids = [pid.block_id for pid in strategy.pid_info]

        base_body = self._prepare_persistent_body(
            device_function.body, device_function, strategy.virtual_pid_var
        )

        barrier_stmt = None
        if len(boundaries) > 1:
            sem_arg = device_function.new_var("x_grid_sem", dce=False)
            barrier_stmt = backend.grid_barrier_stmt(sem_arg)
            if barrier_stmt is not None:
                barrier_dtype = torch.uint32
                device_function.arguments.append(
                    TensorArg(
                        sem_arg,
                        torch.empty(1, device="meta", dtype=barrier_dtype),
                        f"torch.zeros((1,), device={strategy.get_device_str()}, dtype={barrier_dtype})",
                    )
                )

        loops: list[ast.stmt] = []
        start_expr = "0"
        for boundary in boundaries:
            cond = expr_from_string(
                f"({strategy.virtual_pid_var} >= ({start_expr})) and ({strategy.virtual_pid_var} < ({boundary}))"
            )
            loop_body: list[ast.stmt] = [
                create(ast.If, test=cond, body=list(base_body), orelse=[])
            ]
            target = strategy.virtual_pid_var
            if strategy.is_blocked:
                begin, end = strategy.start_pid_var, strategy.end_pid_var
            else:
                # A global-PID loop with a grid-sized step can overflow on the
                # increment after the last tile. Count local trips with unit
                # stride; reconstruct a global ID only when this worker owns it.
                count = f"(({boundary}) - ({start_expr}))"
                worker = typed_program_id(0)
                grid = strategy.grid_size_expr
                extra = backend.cast_expr(
                    f"({worker}) < (({count}) % ({grid}))",
                    CompileEnvironment.current().index_type(),
                )
                begin, end = "0", f"(({count}) // ({grid})) + ({extra})"
                target = device_function.new_var("phase_iteration")
                loop_body.insert(
                    0,
                    statement_from_string(
                        f"{strategy.virtual_pid_var} = ({start_expr}) + "
                        f"(({worker}) + ({target}) * ({grid}))"
                    ),
                )
            loop = create(
                ast.For,
                target=create(ast.Name, id=target, ctx=ast.Store()),
                iter=expr_from_string(
                    TileStrategy.get_range_call_str(
                        device_function.config,
                        block_ids,
                        begin=begin,
                        end=end,
                        program_ids=True,
                    )
                ),
                body=loop_body,
                orelse=[],
                type_comment=None,
            )
            PersistentPhase.mark(loop, start_expr, boundary, setup, base_body[0])
            loops.append(backend.persistent_program_loop(loop, device_function))
            if boundary != boundaries[-1] and barrier_stmt is not None:
                loops.extend(device_function.cta_barrier(barrier_stmt))
            start_expr = boundary
        return loops


class XYZProgramIDs(ProgramIDs):
    """Use the cuda x/y/z launch grid for PIDs"""

    def codegen(self, state: CodegenState) -> None:
        for i, pid in enumerate(self.pid_info):
            state.codegen.statements_stack[-1].insert(
                i, statement_from_string(f"{pid.pid_var} = {typed_program_id(i)}")
            )

    def codegen_grid(self) -> ast.AST:
        self.total_pids_expr(is_device=False)
        env = CompileEnvironment.current()
        if env.backend.name != "pallas":
            assert len(self.pid_info) <= 3
        return expr_from_string(
            f"({', '.join(pid.num_pids_expr(is_device=False) for pid in self.pid_info)},)"
        )

    @property
    def virtual_program_id(self) -> str:
        """
        XYZProgramIDs uses multi-dimensional program IDs and doesn't have a single
        virtual program ID. Wrappers like L2GroupingProgramIDs must explicitly
        handle XYZProgramIDs by flattening the multi-dimensional IDs themselves.
        """
        raise NotImplementedError(
            "XYZProgramIDs does not support virtual_program_id. "
            "Use explicit flattening of multi-dimensional program IDs instead."
        )


def _compile_device_str() -> str:
    """Device expression for the current compile device (for host-side helpers)."""
    device = CompileEnvironment.current().device
    if (origin := HostFunction.current().first_tensor_origin(device)) is not None:
        return f"{origin.host_str()}.device"
    return f"torch.{device!r}"


def _maybe_emit_xcd_remap(
    device_function: DeviceFunction,
    pid_expr: str,
    total_expr: str,
    active_total_expr: str | None = None,
) -> tuple[list[ast.stmt], str]:
    """Optionally remap a program id into contiguous per-XCD regions.

    No-op (returns ``([], pid_expr)``) unless ``xcd_remap`` is enabled.  When
    enabled, emits the AITER-style contiguous-XCD remap (matching aiter
    ``remap_xcd``) and returns the name of the remapped variable.

    Used for:
    - ``flat`` / ``persistent_interleaved``: the (virtual) program id over the
      logical tile count;
    - ``persistent_blocked``: the worker id (``program_id(0)``) over the grid
      size, remapping which contiguous block each worker owns.
    """
    if not device_function.config.get("xcd_remap", False):
        return [], pid_expr

    # Inject _NUM_XCDS as a host-computed constexpr (mirrors _NUM_SM).
    if device_function.constexpr_arg(NUM_XCD_VAR):
        device_function.codegen.host_statements.append(
            statement_from_string(
                f"{NUM_XCD_VAR} = helion.runtime.get_num_xcd({_compile_device_str()})"
            )
        )

    new_var = device_function.new_var
    pids_per = new_var("xcd_pids_per", dce=True)
    tall = new_var("xcd_tall", dce=True)
    xcd = new_var("xcd_id", dce=True)
    local = new_var("xcd_local_pid", dce=True)
    out = new_var("xcd_pid", dce=True)
    nx = NUM_XCD_VAR
    total = f"({active_total_expr or total_expr})"
    # Matches aiter remap_xcd: the first `tall` XCDs own one extra contiguous PID.
    stmts = [
        statement_from_string(f"{pids_per} = {_nonnegative_cdiv(total, nx)}"),
        statement_from_string(f"{tall} = {total} % {nx}"),
        statement_from_string(f"{tall} = tl.where({tall} == 0, {nx}, {tall})"),
        statement_from_string(f"{xcd} = ({pid_expr}) % {nx}"),
        statement_from_string(f"{local} = ({pid_expr}) // {nx}"),
        statement_from_string(
            f"{out} = tl.where("
            f"{xcd} < {tall}, "
            f"{xcd} * {pids_per} + {local}, "
            f"{tall} * {pids_per} + ({xcd} - {tall}) * ({pids_per} - 1) + {local})"
        ),
    ]
    if active_total_expr is not None:
        guarded = new_var("xcd_pid", dce=True)
        stmts.append(
            statement_from_string(
                f"{guarded} = tl.where(({pid_expr}) < {total}, {out}, {total})"
            )
        )
        out = guarded
    return stmts, out


class FlatProgramIDs(ProgramIDs):
    """Only use the x grid and compute other dimensions"""

    def codegen(self, state: CodegenState) -> None:
        pid_var = self.shared_pid_var or typed_program_id(0)
        remap_stmts, pid_var = _maybe_emit_xcd_remap(
            state.device_function,
            pid_var,
            self.total_pids_expr(is_device=True),
        )
        statements = self._decompose_pid_to_statements(pid_var, state)
        state.codegen.statements_stack[-1][:] = [
            *remap_stmts,
            *statements,
            *state.codegen.statements_stack[-1],
        ]

    def codegen_grid(self) -> ast.AST:
        return expr_from_string(f"({self.total_pids_expr(is_device=False)},)")


@dataclasses.dataclass
class WorklistProgramIDs(ProgramIDs):
    """Compact-worklist grid: one program per work item.

    ``codegen`` emits ``_wid = pl.program_id(0)`` and recovers the owner
    coordinate from the ``owner_ids`` scalar-prefetch ref (``work_<owner>_ref[
    _wid]``) so owner-indexed tensors slice the right owner; ``codegen_grid``
    renders the **static** ``UPPER`` (megablocks bound) as the host grid
    positional, while the compact launcher overrides it with the traced
    ``num_work``.
    """

    upper_expr: str = "1"

    def codegen(self, state: CodegenState) -> None:
        from .pallas.compact_worklist import owner_ref_name

        env = CompileEnvironment.current()
        plan = env.compact_worklist_plan
        assert plan is not None
        stmts: list[ast.stmt] = [statement_from_string(f"_wid = {typed_program_id(0)}")]
        if self.pid_info:
            # owner_ids is always in the metadata (see metadata_arg_names): the
            # owner-grid prologue (q_offsets[seq]) is not DCE'd, so the owner pid
            # must be a valid owner index, NOT the work id (which ranges over
            # work items and would index q_offsets out of bounds).
            owner_pid = self.pid_info[0].pid_var
            ref = owner_ref_name(plan) + "_ref"
            stmts.append(statement_from_string(f"{owner_pid} = {ref}[_wid]"))
        state.codegen.statements_stack[-1][:] = [
            *stmts,
            *state.codegen.statements_stack[-1],
        ]

    def codegen_grid(self) -> ast.AST:
        return expr_from_string(f"({self.upper_expr},)")

    def static_grid_size(self) -> int | None:
        # The compact launcher overrides the grid with the traced work count.
        return None


@dataclasses.dataclass
class L2GroupingProgramIDs(ProgramIDs):
    """Used grouped iteration order to promote L2 cache reuse in matmuls"""

    pid_info: list[PIDInfo] = dataclasses.field(default_factory=list, init=False)
    parent_strategy: ProgramIDs | None = dataclasses.field(default=None)
    group_size: int = 1

    def __post_init__(self) -> None:
        if self.parent_strategy is not None:
            self.parent_strategy.l2_group_size = self.group_size

    def append(self, pid: PIDInfo) -> None:
        """Delegate to parent strategy."""
        assert self.parent_strategy is not None
        self.parent_strategy.append(pid)

    def codegen(self, state: CodegenState) -> None:
        # Generate L2 grouping logic
        # Note: Persistent kernel setup is handled by ForEachProgramID if needed
        assert self.parent_strategy is not None
        parent_pids = self.parent_strategy.pid_info
        assert len(parent_pids) >= 2, "L2 grouping requires at least 2 dimensions"
        if (dead := self.parent_strategy._static_empty_coordinates()) is not None:
            state.codegen.statements_stack[-1][:0] = dead
            return
        new_var = state.device_function.new_var

        # Apply L2 grouping to the 2 fastest varying dimensions (pid_0, pid_1)
        # These are always the first 2 dimensions in the PID decomposition
        num_dims = len(parent_pids)
        assignments = []
        parent = self.parent_strategy
        parent_is_blocked = parent._is_persistent() and getattr(
            parent, "is_blocked", False
        )

        env = CompileEnvironment.current()
        count_dtype = parent.grouped_count_dtype(self.group_size)
        # Generate size variables for all dimensions (except the last which doesn't need one)
        num_blocks: list[str] = []
        for i in range(num_dims - 1):
            num_block_var = new_var(f"num_blocks_{i}", dce=True)
            assignments.append(
                (
                    num_block_var,
                    parent_pids[i].num_pids_expr(
                        is_device=True, count_dtype=count_dtype
                    ),
                )
            )
            num_blocks.append(num_block_var)

        # Determine the base PID to use for L2 grouping.
        # For XYZ strategy, we need to compute a flattened index from the multi-dimensional
        # program IDs since L2 grouping works on a flat 1D PID space.
        if isinstance(self.parent_strategy, XYZProgramIDs):
            # XYZ uses separate program_id(0), program_id(1), etc. for each dimension.
            # We flatten these into a single index using row-major order:
            # flattened_pid = pid_0 + pid_1 * num_blocks_0 + pid_2 * num_blocks_0 * num_blocks_1 + ...
            total = parent.static_grid_size()
            wide = total is None or total > 2**31
            backend = CompileEnvironment.current().backend
            terms = [typed_program_id(0)]
            for i in range(1, num_dims):
                coordinate = typed_program_id(i)
                multiplier = " * ".join(num_blocks[:i])
                if wide:
                    # Each physical dimension fits Int32 even when its flat
                    # ordinal does not. Widen before multiplying the stride.
                    coordinate = backend.cast_expr(
                        coordinate, backend.dtype_str(torch.int64)
                    )
                    multiplier = parent.device_product(num_blocks[:i], parent_pids[:i])
                terms.append(f"{coordinate} * ({multiplier})")
            pid = " + ".join(terms)
        elif isinstance(state.device_function.pid, ForEachProgramID):
            # For ForEachProgramID, use the shared PID variable
            pid = state.device_function.pid.shared_pid_var
        else:
            # For other strategies (Flat, Persistent), use the virtual_program_id
            pid = self.virtual_program_id

        # xcd_remap (if enabled) regroups the PID space into contiguous per-XCD
        # regions *before* L2 grouping orders tiles within it.  For a blocked
        # persistent parent the remap is applied at the worker->block level in
        # the persistent setup instead, so skip it here to avoid double-remap.
        if parent_is_blocked:
            xcd_stmts: list[ast.stmt] = []
        else:
            xcd_stmts, pid = _maybe_emit_xcd_remap(
                state.device_function,
                pid,
                parent.total_pids_expr(is_device=True),
            )

        # Apply L2 grouping to the 2 fastest varying dimensions (pid_0, pid_1)
        fastest_m_idx = 0  # pid_0 (fastest varying)
        fastest_n_idx = 1  # pid_1 (second fastest varying)

        # Extract the 2D portion for the fastest 2 dimensions
        inner_2d_size = new_var("inner_2d_size", dce=True)
        inner_2d_pid = new_var("inner_2d_pid", dce=True)

        num_pid_m = new_var("num_pid_m", dce=True)
        num_pid_n = new_var("num_pid_n", dce=True)
        num_pid_in_group = new_var("num_pid_in_group", dce=True)
        group_id = new_var("group_id", dce=True)
        first_pid_m = new_var("first_pid_m", dce=True)
        group_size_m = new_var("group_size_m", dce=True)

        # Set up L2 grouping for the fastest 2 dimensions
        inner_2d_assignments = [
            (
                num_pid_m,
                parent_pids[fastest_m_idx].num_pids_expr(
                    is_device=True, count_dtype=count_dtype
                ),
            ),
            (
                num_pid_n,
                parent_pids[fastest_n_idx].num_pids_expr(
                    is_device=True, count_dtype=count_dtype
                ),
            ),
        ]

        # Only add modulo for 3D+ cases where we need to extract the 2D portion
        if num_dims > 2:
            inner_2d_assignments.extend(
                [
                    (
                        inner_2d_size,
                        parent.device_product([num_pid_m, num_pid_n], parent_pids[:2]),
                    ),
                    (
                        inner_2d_pid,
                        f"{pid} % {inner_2d_size}",
                    ),  # Extract fastest 2D portion
                ]
            )
        else:
            # For 2D case, the entire PID space is the 2D space
            inner_2d_assignments.append((inner_2d_pid, pid))

        assignments.extend(inner_2d_assignments)
        # ``group_size_m`` is the number of M tiles in this PID's group:
        # ``min(num_pid_m - first_pid_m, group_size)``.  Its two divisions are
        # the only runtime-divisor divisions of the decode (every other divisor
        # is a tile count), and a runtime divisor costs a full reciprocal
        # sequence on the tile-coordinate critical path.  When the M tile count
        # is static the value is known: every group is full when the count is
        # a multiple of the group size (``group_id < num_pid_m / group_size``
        # because the flattened PID stays below ``num_pid_m * num_pid_n``), and
        # a count no larger than the group size means there is a single group
        # (``group_id == 0``) holding every M tile.  Same mapping, constant
        # divisors.
        # The fold is a backend trim (``folds_static_program_counts``); Triton
        # renders keep the runtime ``min`` so their generated code stays
        # byte-identical.
        static_num_pid_m = (
            parent_pids[fastest_m_idx].static_num_pids()
            if CompileEnvironment.current().backend.folds_static_program_counts
            else None
        )
        if static_num_pid_m is not None and static_num_pid_m <= self.group_size:
            group_size_m_expr = num_pid_m
        elif static_num_pid_m is not None and static_num_pid_m % self.group_size == 0:
            group_size_m_expr = f"{self.group_size}"
        else:
            group_size_m_expr = None
        group_limit = min(self.group_size, 2**63 - 1)
        decoded_m, decoded_n = (axis.static_num_pids() for axis in parent_pids[:2])
        bounded_width = _bounded_l2_group_width(
            group_limit, num_pid_m, decoded_m, decoded_n
        )
        group_width = bounded_width or str(group_limit)
        if bounded_width is not None and not isinstance(parent, FlatProgramIDs):
            # A flat physical X launch already bounds the product by its grid size.
            backend = CompileEnvironment.current().backend
            group_width = backend.cast_expr(group_width, backend.dtype_str(torch.int64))
        cluster_n = self._decode_cluster_n()
        if cluster_n > 1:
            # Cluster-aware grouped decode (tcgen05 cluster_n > 1). The
            # CUTLASS persistent scheduler hands the ``cluster_n`` CTAs
            # paired along the cluster's N axis the same tile_m and
            # consecutive tile_n — virtual pids exactly ``num_pid_m``
            # apart — and the A multicast fills every peer's SMEM half
            # from one TMA load issued for that shared tile_m. The plain
            # grouped decode below permutes each CTA's virtual pid
            # independently, so peers can decode DIFFERENT tile_m and
            # silently corrupt the multicast. Decode at cluster-box
            # granularity instead ((1 M tile) x (cluster_n N tiles)):
            # peers collapse to one box pid, the grouped raster permutes
            # whole boxes, and the within-box lane recovers each peer's
            # N tile. Reduces to the plain decode at cluster_n=1.
            lane_n = new_var("cluster_lane_n", dce=True)
            box_pid = new_var("cluster_box_pid", dce=True)
            num_box_n = new_var("num_cluster_box_n", dce=True)
            assignments.extend(
                [
                    (lane_n, f"({inner_2d_pid} // {num_pid_m}) % {cluster_n}"),
                    (
                        box_pid,
                        (
                            f"(({inner_2d_pid} // {num_pid_m}) // {cluster_n})"
                            f" * {num_pid_m} + {inner_2d_pid} % {num_pid_m}"
                        ),
                    ),
                    (num_box_n, f"({num_pid_n} + {cluster_n - 1}) // {cluster_n}"),
                    (num_pid_in_group, f"{group_width} * {num_box_n}"),
                    (group_id, f"{box_pid} // {num_pid_in_group}"),
                    (first_pid_m, f"{group_id} * {group_limit}"),
                    (
                        group_size_m,
                        group_size_m_expr
                        or f"min({num_pid_m} - {first_pid_m}, {group_limit})",
                    ),
                    (
                        parent_pids[fastest_m_idx].pid_var,
                        f"{first_pid_m} + (({box_pid} % {num_pid_in_group}) % {group_size_m})",
                    ),
                    (
                        parent_pids[fastest_n_idx].pid_var,
                        (
                            f"(({box_pid} % {num_pid_in_group}) // {group_size_m})"
                            f" * {cluster_n} + {lane_n}"
                        ),
                    ),
                ]
            )
        else:
            assignments.extend(
                [
                    (num_pid_in_group, f"{group_width} * {num_pid_n}"),
                    (group_id, f"{inner_2d_pid} // {num_pid_in_group}"),
                    (first_pid_m, f"{group_id} * {group_limit}"),
                    (
                        group_size_m,
                        group_size_m_expr
                        or f"min({num_pid_m} - {first_pid_m}, {group_limit})",
                    ),
                    (
                        parent_pids[fastest_m_idx].pid_var,
                        f"{first_pid_m} + (({inner_2d_pid} % {num_pid_in_group}) % {group_size_m})",
                    ),
                    (
                        parent_pids[fastest_n_idx].pid_var,
                        f"({inner_2d_pid} % {num_pid_in_group}) // {group_size_m}",
                    ),
                ]
            )

        # Process remaining dimensions (if any) using standard decomposition
        for i in range(2, num_dims):
            expr = pid
            # Add divisor for all faster dimensions
            if i > 0:
                divisor = parent.device_product(num_blocks[:i], parent_pids[:i])
                expr = f"({expr}) // ({divisor})"
            # Add modulo unless this is the outermost dimension
            if i + 1 < num_dims:  # Not the outermost dimension
                expr = f"({expr}) % {num_blocks[i]}"

            assignments.append((parent_pids[i].pid_var, expr))

        if (
            count_dtype
            and not parent._is_persistent()
            and count_dtype != env.index_type()
        ):
            # The group intermediates can exceed a legal flat grid, but its
            # decoded physical coordinates must retain their historical type.
            outputs = {p.pid_var for p in parent_pids}
            assignments = [
                (
                    var,
                    env.backend.cast_expr(expr, env.index_type())
                    if var in outputs
                    else expr,
                )
                for var, expr in assignments
            ]

        statements = [
            statement_from_string(f"{var} = {expr}") for var, expr in assignments
        ]

        state.codegen.statements_stack[-1][:] = [
            *xcd_stmts,
            *statements,
            *state.codegen.statements_stack[-1],
        ]

    def _decode_cluster_n(self) -> int:
        """The cluster-N width the grouped decode must respect.

        Only the tcgen05 persistent scheduler pairs CTAs along a cluster N
        axis (A-multicast peers must decode the same tile_m). This runs at
        grid-lowering time, before the tcgen05 matmul plan is registered,
        so ``_tcgen05_cluster_n`` reads the config knob. Safe because the
        pid strategy and the mma lowering derive the tcgen05 choice from
        the same ``_kernel_specialized_mma_impl`` probe: when this parent
        is the tcgen05 persistent scheduler and the kernel compiles, the
        launch uses the knob's cluster_n — cute_mma raises
        ``BackendUnsupported`` on the tcgen05 cluster_n demotion paths,
        and a late non-tcgen05 fallback never registers a matmul plan,
        which the tcgen05 persistent layout asserts on before any launch.
        The padding gate in cute_mma (N tile count divisible by
        cluster_n) keeps the box decode a bijection over in-range tiles
        for every launchable grid.
        """
        assert self.parent_strategy is not None
        return self.parent_strategy.decode_cluster_n()

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression using parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.virtual_program_id

    def codegen_grid(self) -> ast.AST:
        assert self.parent_strategy is not None
        return self.parent_strategy.codegen_grid()

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Delegate to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.setup_persistent_kernel(
            device_function, total_pids_expr
        )

    def host_count_known(self) -> bool:
        assert self.parent_strategy is not None
        return self.parent_strategy.host_count_known()

    def _is_persistent(self) -> bool:
        """Forward to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy._is_persistent()

    def total_pids_expr(
        self, *, is_device: bool, count_dtype: str | None = None
    ) -> str:
        """Forward to parent strategy."""
        assert self.parent_strategy is not None
        return self.parent_strategy.total_pids_expr(
            is_device=is_device, count_dtype=count_dtype
        )

    def static_grid_size(self) -> int | None:
        """Forward to parent strategy (the grouping reorders, never resizes)."""
        assert self.parent_strategy is not None
        return self.parent_strategy.static_grid_size()


class PersistentProgramIDs(ProgramIDs):
    """Base class for persistent kernels that use num_sms grid size."""

    def __init__(self, is_blocked: bool = False) -> None:
        super().__init__()
        self.is_blocked: bool = is_blocked
        device_function = DeviceFunction.current()
        self.virtual_pid_var: str = device_function.new_var("virtual_pid")
        self.total_pids_var: str = device_function.new_var("total_pids")
        # Get num_sm_multiplier from config for multi-occupancy support
        # pyrefly: ignore [bad-assignment]
        self.num_sm_multiplier: int = device_function.config.get("num_sm_multiplier", 1)
        # Compute grid size expression based on multiplier
        if self.num_sm_multiplier == 1:
            self.grid_size_expr: str = NUM_SM_VAR
        else:
            self.grid_size_expr = f"({NUM_SM_VAR} * {self.num_sm_multiplier})"
        # Generate variables and range expression based on strategy type
        if self.is_blocked:
            self.block_size_var: str = device_function.new_var("block_size")
            self.start_pid_var: str = device_function.new_var("start_pid")
            self.end_pid_var: str = device_function.new_var("end_pid")
            self.range_kwargs: dict[str, str] = {
                "begin": self.start_pid_var,
                "end": self.end_pid_var,
            }
        else:
            self.range_kwargs: dict[str, str] = {
                "begin": typed_program_id(0),
                "end": self.total_pids_var,
                "step": self.grid_size_expr,
            }
        if device_function.constexpr_arg(NUM_SM_VAR):
            reserved_sms = CompileEnvironment.current().settings.persistent_reserved_sms
            reserved_arg = f", reserved_sms={reserved_sms}" if reserved_sms > 0 else ""
            # Dsmem transports launch as many SMs as whole clusters keep
            # resident, by the rule the code was built for
            # (``dsmem_worker_count``).
            if transports := dsmem_transports(device_function):
                reserved_arg += f", cluster_size={cluster_size(transports)}"
            device_function.codegen.host_statements.append(
                statement_from_string(
                    f"{NUM_SM_VAR} = helion.runtime.get_num_sm({self.get_device_str()}{reserved_arg})"
                )
            )
            if transports:
                # The schedule takes the compile device's count, but another
                # device's GPCs may keep fewer clusters resident.
                workers = dsmem_worker_count(
                    device_function, CompileEnvironment.current().config_spec.num_sm
                )
                device_function.codegen.host_statements.append(
                    statement_from_string(
                        f"if {NUM_SM_VAR} != {workers}:\n"
                        f"    raise RuntimeError('cross_loop_dsmem: built for {workers}"
                        f" SMs in clusters of {cluster_size(transports)}, the device"
                        f" keeps ' + str({NUM_SM_VAR}))"
                    )
                )

    def get_device_str(self) -> str:
        """Get the device string for the current device, reusing the first tensor's origin."""
        return _compile_device_str()

    def codegen_grid(self) -> ast.AST:
        # Use num_sms * multiplier for persistent kernels (multi-occupancy)
        return expr_from_string(f"({self.grid_size_expr},)")

    def static_grid_size(self) -> int | None:
        """``_NUM_SM * num_sm_multiplier`` CTAs: the SM count the config spec
        captured for the compile device (the same reserved-adjusted count the
        host computes), ``None`` when it is unknown."""
        num_sm = CompileEnvironment.current().config_spec.num_sm
        if num_sm <= 1:
            return None
        num_sm = dsmem_worker_count(DeviceFunction.current(), num_sm)
        return num_sm * self.num_sm_multiplier

    def _needs_wide_partition(self, total: str) -> bool:
        # A static small launch cannot overflow even on the final advance by
        # the physical worker count. Device lengths and host root expressions
        # need not share the pointer index type of the tensors in the body.
        if total.isdecimal():
            workers = (
                CompileEnvironment.current().config_spec.num_sm * self.num_sm_multiplier
            )
            return int(total) + workers > torch.iinfo(torch.int32).max
        # A Python scalar range does not inherit a small tensor's index width.
        return True

    def _persistent_setup_statements(self, total_pids_expr: str) -> list[ast.stmt]:
        """Generate the preamble statements for persistent kernel setup."""
        env = CompileEnvironment.current()
        backend = env.backend
        # Cast total_pids to match the index type so all persistent scheduling
        # variables (start_pid, end_pid, etc.) have consistent types.
        wide = self._needs_wide_partition(total_pids_expr)
        if wide or env.index_dtype != torch.int32:
            dtype = backend.dtype_str(torch.int64) if wide else _program_id_type(env)
            total_pids_expr = backend.cast_expr(total_pids_expr, dtype)
        stmts: list[ast.stmt] = [
            statement_from_string(f"{self.total_pids_var} = {total_pids_expr}"),
        ]
        if (
            self.is_blocked
            and self.block_size_var
            and self.start_pid_var
            and self.end_pid_var
        ):
            total = self.total_pids_var
            workers = self.grid_size_expr
            block_size = (
                _nonnegative_cdiv(total, workers)
                if wide
                else backend.cdiv_expr(total, workers, is_device=True)
            )
            stmts.append(statement_from_string(f"{self.block_size_var} = {block_size}"))
            worker = typed_program_id(0)
            if DeviceFunction.current().config.get("xcd_remap", False):
                new_var = DeviceFunction.current().new_var
                safe_block_size = new_var("xcd_safe_block_size", dce=True)
                active_workers = new_var("xcd_active_workers", dce=True)
                # Even when total + workers fits in Int32, total + block_size
                # may not: these count different partitions.
                active_count = _nonnegative_cdiv(total, safe_block_size)
                stmts.extend(
                    [
                        statement_from_string(
                            f"{safe_block_size} = tl.maximum({self.block_size_var}, 1)"
                        ),
                        statement_from_string(f"{active_workers} = {active_count}"),
                    ]
                )
                # Remap only the workers that own at least one block.  Slack
                # workers are sent past the end of the compact worker domain so
                # their persistent range is empty after end clipping.
                wstmts, worker = _maybe_emit_xcd_remap(
                    DeviceFunction.current(),
                    typed_program_id(0),
                    active_workers,
                    active_workers,
                )
                stmts.extend(wstmts)
            if wide:
                # ceil(total/workers) * worker can exceed signed Int64 for a
                # slack worker. Clip in unsigned arithmetic before narrowing.
                start = f"min({backend.uint64_expr(worker)} * {backend.uint64_expr(self.block_size_var)}, {backend.uint64_expr(total)})"
                start = backend.cast_expr(start, backend.dtype_str(torch.int64))
                stmts.extend(
                    (
                        statement_from_string(f"{self.start_pid_var} = {start}"),
                        statement_from_string(
                            f"{self.end_pid_var} = {self.start_pid_var} + min({self.block_size_var}, {total} - {self.start_pid_var})"
                        ),
                    )
                )
                return stmts
            stmts.extend(
                [
                    statement_from_string(
                        f"{self.start_pid_var} = {worker} * {self.block_size_var}"
                    ),
                    statement_from_string(
                        f"{self.end_pid_var} = {self.start_pid_var} + {self.block_size_var}"
                    ),
                    create(
                        ast.If,
                        test=expr_from_string(
                            f"{self.end_pid_var} > {self.total_pids_var}"
                        ),
                        body=[
                            statement_from_string(
                                f"{self.end_pid_var} = {self.total_pids_var}"
                            )
                        ],
                        orelse=[],
                    ),
                ]
            )
        return stmts

    def setup_persistent_kernel(
        self, device_function: DeviceFunction, total_pids_expr: str | None = None
    ) -> list[ast.stmt] | None:
        """Setup persistent kernel and return the wrapped body."""
        # Get total PIDs expression
        if total_pids_expr is None:
            total_pids_expr = self.total_pids_expr(is_device=True)
            env = CompileEnvironment.current()
            if env.backend.wide_cumulative_program_ids(
                [total_pids_expr], blocked=self.is_blocked
            ):
                total_pids_expr = self.total_pids_expr(
                    is_device=True, count_dtype=env.backend.dtype_str(torch.int64)
                )

        backend = CompileEnvironment.current().backend
        device_function.preamble.extend(
            self._persistent_setup_statements(total_pids_expr)
        )
        # Collect all block IDs from PID info for range configuration
        pid_block_ids = []
        for pid_info in self.pid_info:
            pid_block_ids.append(pid_info.block_id)

        from .tile_strategy import TileStrategy

        bounds = self.range_kwargs
        relative_pid = None
        if self._needs_wide_partition(total_pids_expr) and not self.is_blocked:
            # Count from worker-total to zero. With the usual positive stride,
            # the last advance can overshoot zero but can never wrap at INT64_MAX.
            relative_pid = device_function.new_var("relative_pid")
            bounds = dict(
                bounds, begin=f"({bounds['begin']}) - {self.total_pids_var}", end="0"
            )
        range_expr = None
        if not total_pids_expr.isdecimal() or int(total_pids_expr) >= 2**31:
            range_expr = backend.persistent_range_str(
                bounds["begin"],
                bounds["end"],
                bounds.get("step"),
            )
        if range_expr is None:
            range_expr = TileStrategy.get_range_call_str(
                device_function.config, pid_block_ids, program_ids=True, **bounds
            )
        return self._setup_persistent_kernel_and_wrap_body(
            device_function,
            self.virtual_pid_var,
            range_expr,
            self.total_pids_var,
            relative_pid=relative_pid,
        )

    def _is_persistent(self) -> bool:
        """Check if this is a persistent strategy."""
        return True

    def _decompose_virtual_pid(
        self,
        state: CodegenState,
        virtual_pid_var: str,
        setup_statements: list[ast.stmt],
    ) -> None:
        """Decompose virtual PID into individual PID variables."""
        # Use shared_pid_var if available, otherwise virtual_pid_var
        pid_var = self.shared_pid_var or virtual_pid_var
        # Interleaved: remap each virtual pid into per-XCD contiguous regions
        # (matches aiter's per-tile remap_xcd).  Blocked remaps the worker->block
        # assignment in the persistent setup instead, so it is skipped here.
        if not self.is_blocked:
            remap_stmts, pid_var = _maybe_emit_xcd_remap(
                state.device_function,
                pid_var,
                self.total_pids_expr(is_device=True),
            )
            setup_statements.extend(remap_stmts)
        statements = self._decompose_pid_to_statements(pid_var, state)
        setup_statements.extend(statements)

    def _generate_pid_statements(self, state: CodegenState) -> list[ast.stmt]:
        """Generate PID decomposition statements based on setup state."""
        if not self.virtual_pid_var:
            # Generate regular PID decomposition
            return self._decompose_pid_to_statements(
                self.shared_pid_var or typed_program_id(0), state
            )

        # Generate persistent PID decomposition
        statements = []
        self._decompose_virtual_pid(state, self.virtual_pid_var, statements)
        return statements

    def _prepend_statements(
        self, state: CodegenState, statements: list[ast.stmt]
    ) -> None:
        """Prepend statements to current statement stack."""
        current_statements = state.codegen.statements_stack[-1]
        current_statements[:] = [*statements, *current_statements]

    def codegen(self, state: CodegenState) -> None:
        """Common codegen logic for persistent kernels."""
        is_shared_pid = isinstance(state.device_function.pid, ForEachProgramID)

        # Set up persistent loop if needed (non-ForEachProgramID case only)
        if not is_shared_pid and not self.virtual_pid_var:
            self.setup_persistent_kernel(state.device_function)

        # Generate and prepend PID decomposition statements
        statements = self._generate_pid_statements(state)
        self._prepend_statements(state, statements)

    @property
    def virtual_program_id(self) -> str:
        """Get the virtual program ID expression for persistent strategies."""
        return self.virtual_pid_var


class PersistentBlockedProgramIDs(PersistentProgramIDs):
    """Persistent kernels where each SM processes a contiguous block of virtual PIDs."""

    def __init__(self) -> None:
        super().__init__(is_blocked=True)


class PersistentInterleavedProgramIDs(PersistentProgramIDs):
    """Persistent kernels where each SM processes every num_sms-th virtual PID."""

    def __init__(self) -> None:
        super().__init__(is_blocked=False)
