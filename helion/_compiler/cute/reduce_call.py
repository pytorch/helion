"""Grouped reduction helper calls as data.

Role: general-purpose emitter.  The CPU-only vocabulary shared by the
reduction emitters, which render
``_cute_grouped_reduce_*`` calls, and the AST passes, which recognize and
rewrite them.  :class:`ReduceScope` pins the dispatcher names of
``reduce_helpers`` (the device module generated code imports; it imports
cutlass, so it is never imported from here) and the role-based name sets the
passes consume; :class:`ReduceCall` records one call of the scalar family
(``(input, 'op', identity, lane, ...)``) and renders it back to the exact
text the emitters write.  ``test/test_cute_reduce_call.py`` asserts the
table against the dispatchers and ``CuteBackend.library_imports``.  The
routing of the shared two-stage dispatcher (``use_serial_block_reduce``,
``use_interleaved_shared_reduce``) lives here too: ``reduce_helpers``
dispatches on it and ``smem_footprint`` sizes each form's buffer by it.
"""

from __future__ import annotations

import ast
import dataclasses
import enum


class ReduceScope(enum.Enum):
    """One emitted grouped-reduce dispatcher of ``reduce_helpers``; the value
    is the helper's name in generated code."""

    # The scalar family: ``(input, 'op', identity, lane, ...)``.
    WARP = "_cute_grouped_reduce_warp"
    SHARED_TWO_STAGE = "_cute_grouped_reduce_shared_two_stage"
    SHARED_TREE = "_cute_grouped_reduce_shared_tree"
    CLUSTER = "_cute_grouped_reduce_cluster"
    BLOCK = "_cute_grouped_reduce_block"
    # Register-tile forms (a list / fragment of per-thread values).
    SHARED_COLUMNS = "_cute_grouped_reduce_shared_columns"
    SHARED_TWO_STAGE_FRAGMENT = "_cute_grouped_reduce_shared_two_stage_fragment"
    # Online-softmax (max, sum) pair exchanges across a cluster.
    CLUSTER_ONLINE_PAIR = "_cute_grouped_reduce_cluster_online_pair"
    CLUSTER_WARP_ONLINE_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair"

    @property
    def helper(self) -> str:
        return self.value


def _names(*scopes: ReduceScope) -> frozenset[str]:
    return frozenset(scope.value for scope in scopes)


# Every grouped-reduce dispatcher generated code may name.
GROUPED_REDUCE_HELPERS: frozenset[str] = _names(*ReduceScope)
# Scalar reduces of one CTA: op literal at args[1], identity at args[2], the
# lane at args[3].  Their result is CTA-local (relocatable with the lane
# loop, foldable over V-lanes, duplicable across sibling tiles only with
# care).
CTA_REDUCES: frozenset[str] = _names(
    ReduceScope.WARP, ReduceScope.SHARED_TWO_STAGE, ReduceScope.SHARED_TREE
)
# The scalar reduces whose per-lane input the lane-fold passes hoist and
# fold (``hoist_warp_reduce``, ``fold_lane_reductions``).
LANE_FOLD_REDUCES: frozenset[str] = CTA_REDUCES | _names(ReduceScope.CLUSTER)
# Cluster exchanges: a single-phase mbarrier, so each site runs once per
# kernel.
CLUSTER_REDUCES: frozenset[str] = _names(
    ReduceScope.CLUSTER,
    ReduceScope.CLUSTER_ONLINE_PAIR,
    ReduceScope.CLUSTER_WARP_ONLINE_PAIR,
)
# The online-softmax pair exchanges: ``(sum, max, lane, ...)`` positionals.
ONLINE_PAIR_REDUCES: frozenset[str] = _names(
    ReduceScope.CLUSTER_ONLINE_PAIR, ReduceScope.CLUSTER_WARP_ONLINE_PAIR
)
# Whole-CTA collectives the scalar epilogue proof reasons about.
COLLECTIVE_REDUCES: frozenset[str] = CTA_REDUCES | CLUSTER_REDUCES
# Dispatchers whose signature accepts ``fp32_redux`` (native fp32 warp
# extrema).
FP32_REDUX_REDUCES: frozenset[str] = _names(
    ReduceScope.WARP,
    ReduceScope.SHARED_TWO_STAGE,
    ReduceScope.CLUSTER,
    ReduceScope.BLOCK,
    ReduceScope.CLUSTER_ONLINE_PAIR,
    ReduceScope.CLUSTER_WARP_ONLINE_PAIR,
)
# Shared-memory reduces keyed on the flattened CTA thread id with
# ``group_span`` / ``group_count`` (resized once the launch shape is known).
# The fragment form is resized like the scalar reduces it replaces, so both
# pick the same combine order (the serial form takes a single group).
GROUPED_SHARED_REDUCES: frozenset[str] = _names(
    ReduceScope.SHARED_TWO_STAGE,
    ReduceScope.SHARED_COLUMNS,
    ReduceScope.SHARED_TWO_STAGE_FRAGMENT,
)

# Above this many warps the serial fold's per-thread chain outgrows the
# two-stage form's constant shuffle cost.
_SERIAL_MAX_WARPS = 8


def use_serial_block_reduce(pre: int, group_span: int, group_count: int) -> bool:
    """Whether ``_cute_grouped_reduce_shared_two_stage`` takes the serial
    form: one group of 2 to 8 warps spanning the whole CTA."""
    # group_count == 1 with pre == 1 means ONE group spanning the whole
    # CTA at every current emitter (they all size group_count as
    # num_threads // group_span); the serial body's warp indexing relies
    # on that invariant.
    return (
        pre == 1
        and group_count == 1
        and group_span % 32 == 0
        and 2 <= group_span // 32 <= _SERIAL_MAX_WARPS
    )


def use_interleaved_shared_reduce(pre: int, group_span: int) -> bool:
    """Whether ``_cute_grouped_reduce_shared_two_stage`` takes an interleaved
    form (the warp form below 32 siblings, the block form from 32): a
    power-of-two sibling count ``pre`` dividing a whole-warp span."""
    return (
        pre > 1
        and pre & (pre - 1) == 0
        and group_span % 32 == 0
        and group_span % pre == 0
        and (pre < 32 or pre % 32 == 0)
    )


_SCALAR_SCOPES: dict[str, ReduceScope] = {
    scope.value: scope
    for scope in (
        ReduceScope.WARP,
        ReduceScope.SHARED_TWO_STAGE,
        ReduceScope.SHARED_TREE,
        ReduceScope.CLUSTER,
        ReduceScope.BLOCK,
    )
}


def runtime_lane_expr(index_type: str) -> str:
    """The flattened CTA thread id of ``index_type`` (``cutlass.Int32`` or
    ``cutlass.Int64``) from the runtime block dims: the lane a cross-warp
    shared reduce is keyed on before the launch shape is final."""
    tid = [f"{index_type}(cute.arch.thread_idx()[{axis}])" for axis in range(3)]
    bdim = [f"{index_type}(cute.arch.block_dim()[{axis}])" for axis in range(2)]
    return (
        f"{tid[0]} + ({tid[1]}) * ({bdim[0]}) + ({tid[2]}) * ({bdim[0]}) * ({bdim[1]})"
    )


def int_keyword(call: ast.Call, name: str) -> int | None:
    """The literal ``int`` value of keyword ``name`` of ``call``, else None."""
    for keyword in call.keywords:
        if keyword.arg == name:
            value = keyword.value
            if isinstance(value, ast.Constant) and type(value.value) is int:
                return value.value
            return None
    return None


@dataclasses.dataclass(frozen=True)
class ReduceCall:
    """One call of the scalar grouped-reduce family.

    ``render`` writes the exact text the emitters produce (positional input,
    op literal, identity and lane expressions; integer keywords in the
    dispatcher's order; ``fp32_redux=True`` last when set), and ``parse``
    reads it back from an ``ast.Call``.  Expressions are kept as source text
    so a record survives the unparse-and-reparse cloning of the passes.
    """

    scope: ReduceScope
    op: str
    input: str
    identity: str
    lane_expr: str
    # Shared forms: the group-relative lane and the sibling coordinate.
    lane_in_group: str | None = None
    lane_mod_pre: str | None = None
    pre: int | None = None
    group_span: int | None = None
    group_count: int | None = None
    # Shared tree: the launched thread count that sizes its scratch.
    num_threads: int | None = None
    # Cluster: the receive buffer / mbarrier names and the CTA count.
    buffers: tuple[str, str] | None = None
    cluster_n: int | None = None
    fp32_redux: bool = False

    @property
    def helper(self) -> str:
        return self.scope.value

    def render(self) -> str:
        head = f"{self.input}, {self.op!r}, {self.identity}, {self.lane_expr}"
        redux = ", fp32_redux=True" if self.fp32_redux else ""
        if self.scope is ReduceScope.WARP:
            tail = f"pre={self.pre}, group_span={self.group_span}"
        elif self.scope is ReduceScope.SHARED_TWO_STAGE:
            head = f"{head}, {self.lane_in_group}, {self.lane_mod_pre}"
            tail = (
                f"pre={self.pre}, group_span={self.group_span}, "
                f"group_count={self.group_count}"
            )
        elif self.scope is ReduceScope.SHARED_TREE:
            head = f"{head}, {self.lane_in_group}, {self.lane_mod_pre}"
            tail = (
                f"pre={self.pre}, group_span={self.group_span}, "
                f"num_threads={self.num_threads}, group_count={self.group_count}"
            )
        elif self.scope is ReduceScope.CLUSTER:
            assert self.buffers is not None
            head = f"{head}, {self.buffers[0]}, {self.buffers[1]}"
            tail = f"group_span={self.group_span}, cluster_n={self.cluster_n}"
        else:
            assert self.scope is ReduceScope.BLOCK
            tail = f"group_span={self.group_span}"
        return f"{self.helper}({head}, {tail}{redux})"

    @staticmethod
    def parse(node: ast.AST) -> ReduceCall | None:
        """The record of ``node`` when it is a well-formed call of the scalar
        family (literal op, the dispatcher's positional count, integer
        keywords in its order); None for anything else."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return None
        scope = _SCALAR_SCOPES.get(node.func.id)
        if scope is None:
            return None
        args = node.args
        positional = {
            ReduceScope.WARP: 4,
            ReduceScope.SHARED_TWO_STAGE: 6,
            ReduceScope.SHARED_TREE: 6,
            ReduceScope.CLUSTER: 6,
            ReduceScope.BLOCK: 4,
        }[scope]
        if len(args) != positional:
            return None
        op_node = args[1]
        if not (isinstance(op_node, ast.Constant) and isinstance(op_node.value, str)):
            return None
        keywords = [keyword.arg for keyword in node.keywords]
        fp32_redux = False
        if keywords and keywords[-1] == "fp32_redux":
            value = node.keywords[-1].value
            if not (isinstance(value, ast.Constant) and value.value is True):
                return None
            fp32_redux = True
            keywords = keywords[:-1]
        expected = {
            ReduceScope.WARP: ["pre", "group_span"],
            ReduceScope.SHARED_TWO_STAGE: ["pre", "group_span", "group_count"],
            ReduceScope.SHARED_TREE: [
                "pre",
                "group_span",
                "num_threads",
                "group_count",
            ],
            ReduceScope.CLUSTER: ["group_span", "cluster_n"],
            ReduceScope.BLOCK: ["group_span"],
        }[scope]
        if keywords != expected:
            return None
        ints = {name: int_keyword(node, name) for name in expected}
        if any(value is None for value in ints.values()):
            return None
        shared = scope in (ReduceScope.SHARED_TWO_STAGE, ReduceScope.SHARED_TREE)
        return ReduceCall(
            scope=scope,
            op=op_node.value,
            input=ast.unparse(args[0]),
            identity=ast.unparse(args[2]),
            lane_expr=ast.unparse(args[3]),
            lane_in_group=ast.unparse(args[4]) if shared else None,
            lane_mod_pre=ast.unparse(args[5]) if shared else None,
            pre=ints.get("pre"),
            group_span=ints.get("group_span"),
            group_count=ints.get("group_count"),
            num_threads=ints.get("num_threads"),
            buffers=(
                (ast.unparse(args[4]), ast.unparse(args[5]))
                if scope is ReduceScope.CLUSTER
                else None
            ),
            cluster_n=ints.get("cluster_n"),
            fp32_redux=fp32_redux,
        )
