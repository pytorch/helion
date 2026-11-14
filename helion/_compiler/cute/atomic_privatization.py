"""Per-CTA shared-memory privatization of small atomic-add targets.

``cute_privatize_atomics``: a relaxed ``hl.atomic_add`` into a small fp32 or
int32 tensor that the kernel touches only through such adds (a histogram, a
bucketed bias gradient) serializes in L2 when many threads hit few cells.
T5's relative-position-bias backward adds 8M values into 32 x 8 cells, eight
cache lines, for milliseconds on B200 against a ~130 us read of its inputs.
With the knob on, every CTA fills a shared-memory copy of the target with the
identity of the add, the atomics add into that copy, and after the CTA's last
tile each changed cell of the copy is added to global memory once.

Admission (``privatizable_atomic_targets``) reads the device IR: every device
use of the target's storage is the destination of a relaxed ``atomic_add``
whose result is unused; the target is an fp32 or int32 host tensor of static
shape and rank at least 1, and the copies of all targets fit
``PRIVATE_MAX_BYTES``; and the target is proven not to overlap any other
tensor the kernel accesses (one of the two is a fresh host allocation, or the
bound kernel's cache key holds the fact that the storages of the two inputs
are disjoint).  Different arguments are no such proof: a caller can pass one
tensor twice, and a read through the other name would miss the adds the copy
holds back.  Kernels with a matmul (whose families generate their own kernel
bodies), with ``hl.barrier()`` phases, or with an operation that orders memory
for another observer (``_orders_memory``: an acquire, release or acq_rel
atomic on any tensor, asm not declared pure, a remote copy) are not admitted:
a relaxed add that a release publishes would reach global memory only at the
CTA's end, after the release.

With the knob on, the storages of the admitted targets
(``privatized_storages``, computed once per codegen) are the one predicate
that the lowering and the rewrite share: their atomics never take the vector
``red.global`` form (``privatizes``), as the rewrite redirects scalar atomics
only, and the rewrite (``privatize_atomic_targets``, once device codegen is
done) redirects every one of them.  Where the generated kernel defeats it (an
early return, a tensor argument the bound kernel cannot prove disjoint from
the target, a reference to the target other than the pointer of a scalar
``cute.arch.atomic_add``), the config is invalid, as normalization makes it
where nothing is admitted, rather than the knob-off code without its vector
atomics.

The knob changes the summation order: a CTA sums its adds to a cell in shared
memory, in no particular order, and adds that partial sum to global memory
once.  For fp32 that rounds differently, and it can overflow where the
original order does not: from -3e38, one thread adding 2e38 twice reaches
about 1e38, but its partial sum 4e38 is inf.  Like ``cute_independent_reduction``
the knob is off by default and left to the autotuner, whose accuracy check on
the tuning inputs is what catches such cases.  int32 adds wrap and commute,
so int32 results are exact.

The copy starts at the identity of the add, -0.0 for fp32 (``x + -0.0`` is
``x`` for every ``x``, either zero included) and 0 for int32, and the flush
skips a cell that still holds those bits: the CTA never updated it, or its
updates summed to the identity, whose add changes no cell's value.  It is
not quite a no-op on the hardware: ``red.global.add.f32`` flushes a subnormal
cell to zero and canonicalizes a NaN payload (``_cute_where_zero_elision``),
so a cell that holds such a value and gets only adds that sum to -0.0 in
every CTA keeps it with the knob on.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch
from torch.utils import _pytree as pytree

from ... import exc
from ...language import _tracing_ops
from ...language._decorators import is_api_func
from ...language.atomic_ops import ATOMIC_OPS
from ...language.atomic_ops import atomic_add
from ...language.inline_asm_ops import inline_asm_elementwise
from ...language.memory_ops import store
from ..ast_extension import create
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..host_function import HostFunction
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from ._ast_pass_utils import body_end_follows_all_tiles
from .atomic_pointer import cute_coord_pointer_expr
from .atomic_pointer import cute_coord_tuple

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST

# The most shared memory the private copies of one kernel take: 4096 fp32 or
# int32 cells.  That can cost occupancy: on B200 (228 KiB per SM, 1 KiB of it
# reserved per CTA) a CTA holding 16 KiB leaves room for 13 CTAs per SM, fewer
# than the 16 that 128-thread CTAs could otherwise keep resident; the
# autotuner weighs that against the contention the copy removes.
PRIVATE_MAX_BYTES = 16384


class _PrivateDtype(NamedTuple):
    """The spelling of a private copy of one dtype: its cutlass type, the
    identity of the add it starts from, and the test (a format string over
    ``value``) that a cell's partial sum differs from that identity -- for
    fp32 in its bits, as ``-0.0 == 0.0``."""

    cutlass: str
    identity: str
    changed: str


_PRIVATE_DTYPES: dict[torch.dtype, _PrivateDtype] = {
    torch.float32: _PrivateDtype(
        "cutlass.Float32",
        "cutlass.Float32(-0.0)",
        "{value}.bitcast(cutlass.Uint32) != cutlass.Uint32(0x80000000)",
    ),
    torch.int32: _PrivateDtype(
        "cutlass.Int32", "cutlass.Int32(0)", "{value} != cutlass.Int32(0)"
    ),
}


def privatizable_atomic_targets(
    env: CompileEnvironment, device_ir: DeviceIR, *, allow_unbound: bool = False
) -> frozenset[int]:
    """The storages (``id(tensor.untyped_storage())``) whose atomics
    ``cute_privatize_atomics`` privatizes.

    A target is disjoint from another tensor of the kernel when either one's
    storage belongs to no input (a host allocation), or by the storage-span
    fact the bound kernel's cache key holds for two inputs
    (``runtime_tensors_are_proven_disjoint``).  ``allow_unbound`` is for
    binding, before the fact is recorded: it reads the fact off the
    arguments the key is built from; codegen keeps the recorded one.  The
    views of one storage must share a dtype: an int32 add and an fp32 add to
    the same bits do not commute, so copies of the two flushed in either
    order need not give what the global atomics do.  A kernel with an
    operation that orders memory for another observer (``_orders_memory``),
    or an access through a stack tensor's pointers, which may name any
    target (``reaches_through_pointers``), has no targets.
    """
    # Not at module level: the CuTe heuristics import this module while
    # ``helion.language``, which ``device_ir_analysis`` reads, initializes.
    from ...language import _MEMORY_OPS
    from ..device_ir_analysis import TENSOR_ROUTING_OPS
    from ..loop_dependency_checker import reaches_through_pointers

    if env.config_spec.matmul_facts or len(device_ir.phases) > 1:
        return frozenset()
    views: dict[int, dict[int, torch.Tensor]] = {}
    added: set[int] = set()
    rejected: set[int] = set()
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if _orders_memory(node, TENSOR_ROUTING_OPS) or (
                node.op == "call_function"
                and node.target in _MEMORY_OPS
                and reaches_through_pointers(node.args[0])
            ):
                return frozenset()
            if (
                node.op != "call_function"
                or node.target is not _tracing_ops._host_tensor
            ):
                continue
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            key = id(value.untyped_storage())
            views.setdefault(key, {})[id(value)] = value
            if not _privatizable_shape(value) or not all(
                _privatizable_use(user, node) for user in node.users
            ):
                rejected.add(key)
            elif node.users:
                added.add(key)
    inputs = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    targets: set[int] = set()
    budget = PRIVATE_MAX_BYTES
    for key, group in views.items():
        if key not in added or key in rejected:
            continue
        size = sum(view.numel() * view.element_size() for view in group.values())
        target = next(iter(group.values()))
        if (
            size > budget
            or len({view.dtype for view in group.values()}) > 1
            or not all(
                other_key not in inputs
                or key not in inputs
                or runtime_tensors_are_proven_disjoint(
                    env,
                    target,
                    next(iter(other.values())),
                    allow_unbound=allow_unbound,
                )
                for other_key, other in views.items()
                if other_key != key
            )
        ):
            continue
        budget -= size
        targets.add(key)
    return frozenset(targets)


def _orders_memory(node: torch.fx.Node, routing_ops: tuple[object, ...]) -> bool:
    """Whether ``node`` may order the kernel's memory accesses for an
    observer outside the CTA (another kernel, a peer rank): an atomic that is
    not relaxed, on any tensor; asm not declared pure, which may hold a
    fence; or any other side-effecting Helion op but a store and the ops that
    touch no memory (``routing_ops``, ``TENSOR_ROUTING_OPS``), such as a
    remote copy or barrier.  A relaxed add before it would reach global
    memory only when the CTA flushes its copy, after it."""
    if node.op != "call_function":
        return False
    if node.target in ATOMIC_OPS:
        return bool(node.args[-1] != "relaxed")
    if node.target is inline_asm_elementwise:
        return not node.args[4]
    return (
        is_api_func(node.target)
        and node.is_impure(impure_random=False)
        and node.target is not store
        and node.target not in routing_ops
    )


def _privatizable_shape(value: torch.Tensor) -> bool:
    # Not a 0-d target: CuTe cannot yet index one (``crd2idx`` of the empty
    # coordinate into its ``(1):(1)`` layout), knob or not.  The size budget
    # is the caller's, over all the views of a storage.
    return (
        value.dtype in _PRIVATE_DTYPES
        and value.ndim > 0
        and all(isinstance(size, int) for size in value.shape)
        and value.numel() > 0
    )


def _privatizable_use(user: torch.fx.Node, node: torch.fx.Node) -> bool:
    return bool(
        user.op == "call_function"
        and user.target is atomic_add
        and len(user.args) == 4
        and user.args[0] is node
        and user.args[3] == "relaxed"
        and not user.users
        and not any(leaf is node for leaf in pytree.tree_leaves(user.args[1:]))
    )


def privatized_storages(fn: DeviceFunction) -> frozenset[int]:
    """The storages whose atomics ``fn``'s config privatizes:
    ``privatizable_atomic_targets`` with ``cute_privatize_atomics`` on, else
    none.  Computed once per codegen, for the lowering (``privatizes``) and
    the rewrite alike."""
    state = fn.cute_state
    if state.privatized_storages is None:
        state.privatized_storages = (
            privatizable_atomic_targets(
                CompileEnvironment.current(), HostFunction.current().device_ir
            )
            if fn.config.get("cute_privatize_atomics") is True
            else frozenset()
        )
    return state.privatized_storages


def privatizes(fn: DeviceFunction, target: torch.Tensor) -> bool:
    """Whether ``fn``'s config privatizes the atomics into ``target``."""
    return id(target.untyped_storage()) in privatized_storages(fn)


def privatize_atomic_targets(codegen: GenerateAST, host: HostFunction) -> None:
    """Redirect the device body's atomics into the privatized targets
    (``privatized_storages``) to per-CTA copies, or reject the config when
    the generated kernel defeats that."""
    fn = codegen.device_function
    storages = privatized_storages(fn)
    if not storages:
        return
    kernel = [*fn.preamble, *fn.body]
    if not body_end_follows_all_tiles(kernel, host.device_ir):
        raise exc.InvalidConfig(
            "cute_privatize_atomics: the kernel returns before its flush"
        )
    tensor_arguments = [
        argument for argument in fn.arguments if isinstance(argument, TensorArg)
    ]
    disjoint = fn.proven_disjoint_tensor_pairs()
    prologue: list[ast.AST] = []
    epilogue: list[ast.AST] = []
    for argument in tensor_arguments:
        storage = id(argument.fake_value.untyped_storage())
        if storage not in storages:
            continue
        # The copy is filled at the top of the body, after the preamble.
        sites = _atomic_sites(fn.body, argument.name)
        if (
            type(argument) is not TensorArg
            or not all(
                id(other.fake_value.untyped_storage()) == storage
                or frozenset((argument.name, other.name)) in disjoint
                for other in tensor_arguments
            )
            or sites is None
            or any(
                isinstance(node, ast.Name) and node.id == argument.name
                for statement in fn.preamble
                for node in ast.walk(statement)
            )
        ):
            raise exc.InvalidConfig(
                f"cute_privatize_atomics: {argument.name} is reached other than "
                "by scalar atomics, or may overlap another tensor argument"
            )
        private = fn.new_var(f"_private_{argument.name}")
        _point_sites_at(sites, private)
        fake = argument.fake_value
        shape = [int(size) for size in fake.shape]
        numel = math.prod(shape)
        dtype = _PRIVATE_DTYPES[fake.dtype]
        row_major = _row_major_strides(shape)
        layout = (
            f"cute.make_layout({cute_coord_tuple(shape)}, "
            f"stride={cute_coord_tuple(row_major)})"
        )
        flat = fn.new_var(f"{private}_cells")
        cell = fn.new_var(f"{private}_cell")
        value = fn.new_var(f"{private}_value")
        prologue.extend(
            statement_from_string(source)
            for source in (
                f"{private} = cute.make_tensor(cute.arch.alloc_smem({dtype.cutlass}, {numel}), {layout})",
                f"{flat} = cute.make_tensor({private}.iterator, ({numel},))",
                (
                    f"for {cell} in range({_LINEAR_THREAD}, {numel}, {_BLOCK_THREADS}):\n"
                    f"    {flat}[{cell}] = {dtype.identity}"
                ),
            )
        )
        pointer = cute_coord_pointer_expr(
            argument.name,
            [
                f"{cell} // {stride} % {size}"
                for size, stride in zip(shape, row_major, strict=True)
            ],
        )
        epilogue.append(
            statement_from_string(
                f"for {cell} in range({_LINEAR_THREAD}, {numel}, {_BLOCK_THREADS}):\n"
                f"    {value} = {flat}[{cell}]\n"
                f"    if {dtype.changed.format(value=value)}:\n"
                f"        cute.arch.atomic_add({pointer}, {value}, sem='relaxed')"
            )
        )
    if not prologue:
        return
    barrier = "cute.arch.sync_threads()"
    fn.body[:0] = [*prologue, statement_from_string(barrier)]
    fn.body.extend([statement_from_string(barrier), *epilogue])


_LINEAR_THREAD = (
    "cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1])"
    " * cutlass.Int32(cute.arch.block_dim()[0]) + cutlass.Int32(cute.arch.thread_idx()[2])"
    " * cutlass.Int32(cute.arch.block_dim()[0]) * cutlass.Int32(cute.arch.block_dim()[1])"
)
_BLOCK_THREADS = (
    "cutlass.Int32(cute.arch.block_dim()[0]) * cutlass.Int32(cute.arch.block_dim()[1])"
    " * cutlass.Int32(cute.arch.block_dim()[2])"
)


def _row_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return strides


_AtomicSite = tuple[ast.Call, tuple[ast.Name, ast.Name]]


def _atomic_sites(body: list[ast.AST], name: str) -> list[_AtomicSite] | None:
    """The relaxed scalar ``cute.arch.atomic_add`` statements into ``name``
    (``_atomic_site``), or None unless there is one and every reference to
    ``name`` sits in one's pointer, ``(name.iterator + cute.crd2idx(coord,
    name.layout)).llvm_ptr``."""
    sites: list[_AtomicSite] = []
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Expr) and (site := _atomic_site(node, name)):
                sites.append(site)
    owned = {id(reference) for _, references in sites for reference in references}
    if not sites or any(
        isinstance(node, ast.Name) and node.id == name and id(node) not in owned
        for statement in body
        for node in ast.walk(statement)
    ):
        return None
    return sites


def _point_sites_at(sites: list[_AtomicSite], private: str) -> None:
    """Point the pointers of ``sites`` at ``private`` and scope their adds to
    the CTA."""
    for call, references in sites:
        for reference in references:
            reference.id = private
        call.keywords.append(
            create(ast.keyword, arg="scope", value=create(ast.Constant, value="cta"))
        )


def _atomic_site(statement: ast.Expr, name: str) -> _AtomicSite | None:
    """A relaxed scalar ``cute.arch.atomic_add`` into ``name`` and the two
    references to ``name`` in its pointer (``cute_coord_pointer_expr``:
    ``(name.iterator + cute.crd2idx(coord, name.layout)).llvm_ptr``, the
    lowering's spelling), or None.  A reference in ``coord`` is not the
    pointer's: the copy's layout has other strides."""
    call = statement.value
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == "cute.arch.atomic_add"
        and len(call.args) == 1
        and not any(keyword.arg == "scope" for keyword in call.keywords)
        and any(
            keyword.arg == "sem"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "relaxed"
            for keyword in call.keywords
        )
    ):
        return None
    pointer = call.args[0]
    if not (
        isinstance(pointer, ast.Attribute)
        and pointer.attr == "llvm_ptr"
        and isinstance(pointer.value, ast.BinOp)
        and isinstance(pointer.value.op, ast.Add)
        and ast.unparse(pointer.value.left) == f"{name}.iterator"
        and isinstance(pointer.value.right, ast.Call)
        and ast.unparse(pointer.value.right.func) == "cute.crd2idx"
        and len(pointer.value.right.args) == 2
        and ast.unparse(pointer.value.right.args[1]) == f"{name}.layout"
    ):
        return None
    iterator = pointer.value.left
    layout = pointer.value.right.args[1]
    assert isinstance(iterator, ast.Attribute) and isinstance(iterator.value, ast.Name)
    assert isinstance(layout, ast.Attribute) and isinstance(layout.value, ast.Name)
    return call, (iterator.value, layout.value)
