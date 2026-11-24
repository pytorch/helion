"""The N-D tiling HelionScheduling gives a Helion kernel: natural_tiling()."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import sympy
from torch._inductor.codegen.simd_kernel_features import EnableReduction
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.functions import ModularIndexing

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from torch._inductor.codegen.simd_kernel_features import NodeScheduleEntry

# The most dims HelionKernel tiles of each kind: tile_z, tile_y and tile_x, and
# tile_r (HelionKernel.supports_tiling()).
MAX_POINTWISE_DIMS = 3
MAX_REDUCTION_DIMS = 1
# The longest rdim a kernel that names it (rdim, codegen_rdim()) names by a
# factory's dim, which Helion cannot roll, so that it stays one block whatever
# the autotuner tries, as Triton's rule reads an inner reduction's: an rdim of up
# to 1024 elements fits one block, and leaving its loop to the autotuner
# (Config.reduction_loops) widens the search, where a quick search can end on a
# loop over chunks of the rdim: for LayerNorm over Swin's shifted windows,
# rdims of 256, it picked reduction_loops [16], slower than the rdim as one block. A
# kernel that does not name its rdim, as a plain sum's in0[tile_x, :], keeps the
# rdim's reduction_loops slot, at any length (HelionKernel.rdim_code()).
UNROLLED_RDIM_MAX = 1024
# The rdim past which natural_tiling() joins pointwise dims that only
# r-invariant accesses keep apart (an access whose index does not move along r
# reads one element per pointwise index): UNROLLED_RDIM_MAX, past which an rdim
# no longer fits one block. A cross-entropy's backward over (y, x) and a long
# rdim of logits reads each (y, x)'s label from a tensor sliced [:, 1:], whose
# y and x do not join: tiled as one axis, with the label read at
# 1 + 513*(x // 512) + x % 512, the CuTe search finds a far faster config (one
# reduction per CTA, many lanes in flight per thread: cute_lane_unroll opens on
# a kernel of one tile axis). A short persistent rdim behind per-channel affine
# weights (an NCHW BatchNorm's) runs faster tiled, where the weights of a
# tile_x serve every y: short rdims keep their dims.
R_INVARIANT_JOIN_MIN = UNROLLED_RDIM_MAX
# The most runs of an rdim HelionKernel reads as a tile of its outer loop
# (reduction_runs()), whose one block holds every run: each CTA keeps a
# partial per run and reduces them at the end, and a block of the runs takes
# that many threads (or lanes) beside the run's own.
REDUCTION_RUNS_MAX = 8


@dataclasses.dataclass
class Access:
    """A load or store of a node of the kernel. Its loop variables are placed in
    the kernel's flat iteration space, [0, numel * reduction_numel) with the
    reduction innermost: `spans` maps each to the flat step of its unit step and
    its size. `weight` is its element count times its element size, what an
    index of // and % costs it."""

    index: sympy.Expr
    spans: dict[sympy.Symbol, tuple[int, int]]
    weight: int


def static_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, sympy.Integer):
        return int(value)
    return None


def accesses(
    node_schedule: list[NodeScheduleEntry],
    numel: int,
    rnumel: int,
    kinds: Sequence[MemoryUsageType] = (
        MemoryUsageType.LOAD,
        MemoryUsageType.STORE,
        MemoryUsageType.STORE_REDUCTION,
        MemoryUsageType.INDEX_EXPR,
    ),
) -> list[Access] | None:
    """The loads and stores of the kernel's nodes, or None if a size is not a
    constant. Each index is read from the node's loop body, over all its loop
    variables: a node's read_writes leave out the variables an index does not
    use, as in w[c] read beside x[n, c, h, w], which would leave c no place in
    the space."""
    total = numel * rnumel
    result = []
    for node in EnableReduction.filter(node_schedule):
        if not isinstance(node, SchedulerNode):
            continue
        body = node._body
        variables = [*body.iter_vars, *body.reduce_vars]
        sizes = [static_int(body.var_ranges[var]) for var in variables]
        if any(size is None for size in sizes):
            return None
        covered = math.prod(sizes)  # pyrefly: ignore [no-matching-overload]
        # Every node EnableReduction.filter() yields iterates all of the space,
        # the reduction innermost: the pointwise nodes outside the reduction,
        # of ranges (numel, 1), sit in the DisableReduction blocks it skips.
        if covered != total:
            return None
        step = 1
        spans: dict[sympy.Symbol, tuple[int, int]] = {}
        for var, size in reversed([*zip(variables, sizes, strict=True)]):
            spans[var] = (step, size)  # pyrefly: ignore [unsupported-operation]
            step *= size  # pyrefly: ignore [unsupported-operation]
        for kind in kinds:
            for entry in body.memory_usage[kind]:
                # Identity(x), which Inductor writes to keep a term from being
                # simplified, is x.
                index = body.indexing_exprs[entry.index_name].replace(
                    Identity, lambda arg: arg
                )
                assert isinstance(index, sympy.Expr)
                # An index computed as a value, such as the x and y
                # (x % 1000 and x // 1000) of a causal mask, moves no bytes:
                # it weighs one per element.
                itemsize = (
                    1
                    if entry.buffer_name is None
                    else V.graph.get_dtype(entry.buffer_name).itemsize
                )
                result.append(Access(index, spans, covered * itemsize))
    return result


def contiguous_reads(
    node_schedule: list[NodeScheduleEntry], numel: int, rnumel: int
) -> tuple[int, int] | None:
    """The bytes the kernel's loads read with a unit stride along a reduction
    variable, and along a pointwise variable only, of the loads that read along
    the reduction, gathers included: Inductor's tiling scores leave out an
    indirect load, as w[i], the rdim of F.embedding(i, w).sum(-1), read at
    1024*tmp1 + r, which counts along the reduction here. A load's bytes are
    its distinct elements: an operand broadcast over a pointwise dim, w[r]
    beside x[y, r], reads its R elements, not one per element of the space
    it is broadcast over. None if a size is not a constant."""
    found = accesses(node_schedule, numel, rnumel, kinds=(MemoryUsageType.LOAD,))
    if found is None:
        return None
    along_r = along_x = 0
    for access in found:
        index = sympy.expand(access.index)
        variables = [var for var in index.free_symbols if var in access.spans]
        if not any(access.spans[var][0] < rnumel for var in variables):
            continue
        unit = [
            var
            for var in variables
            if access.spans[var][1] > 1 and index.coeff(var) == 1
        ]
        covered = math.prod(size for _, size in access.spans.values())
        weight = access.weight // covered
        weight *= math.prod(access.spans[var][1] for var in variables)
        if any(access.spans[var][0] < rnumel for var in unit):
            along_r += weight
        elif unit:
            along_x += weight
    return along_r, along_x


def boundaries(access: Access) -> Iterable[int]:
    """The flat steps where the access's index breaks the space into dims: the
    span of each of its variables, and the divisors and moduli its index takes
    of a variable. The channels-last mean of [128, 1536, 18, 18] reads
    497664*p0 + p1 + 1536*p2, whose p1 of 1536 elements at step 324 (the
    reduction's p2 inside it) breaks the space at 324 and 324 * 1536; an index
    that takes p // 6, or ModularIndexing(p, 6, 4), breaks p at 6 (and 24)."""
    for var, (step, size) in access.spans.items():
        yield step
        yield step * size
        for term in access.index.atoms(FloorDiv, ModularIndexing):
            if term.args[0] != var:
                continue
            divisors = [static_int(arg) for arg in term.args[1:]]
            if any(d is None for d in divisors):
                continue
            # x // d breaks x at d; ModularIndexing(x, d, m), (x // d) % m, at d
            # and d * m.
            cuts = (
                [divisors[0]]
                if isinstance(term, FloorDiv)
                else [
                    divisors[0],
                    divisors[0] * divisors[1],  # pyrefly: ignore [unsupported-operation]
                ]
            )
            for cut in cuts:
                if 1 < cut < size and size % cut == 0:  # pyrefly: ignore [unsupported-operation]
                    yield step * cut  # pyrefly: ignore [unsupported-operation]


def chain(total: int, rnumel: int, candidates: dict[int, int]) -> list[int]:
    """The flat steps of the dims, ascending: 1, the reduction's size, the
    space's size, and the candidate steps, heaviest first, that keep each step
    a divisor of the next."""
    steps = sorted({1, rnumel, total})
    for cut, _ in sorted(candidates.items(), key=lambda item: -item[1]):
        if cut in steps or not 1 < cut < total:
            continue
        upper = next(step for step in steps if step > cut)
        lower = max(step for step in steps if step < cut)
        if cut % lower == 0 and upper % cut == 0:
            steps = sorted([*steps, cut])
    return steps


def strides(access: Access, steps: Sequence[int]) -> list[int] | None:
    """The access's stride along each dim between `steps` (outermost first), or
    None if its index is not affine in the dims' indices."""
    dims = len(steps) - 1
    symbols = [
        sympy.Symbol(f"dim{d}", integer=True, nonnegative=True) for d in range(dims)
    ]
    # Dim d, outermost first, has the flat step steps[dims - 1 - d].
    flat = {steps[dims - 1 - d]: d for d in range(dims)}
    replacements = {}
    for var, (step, size) in access.spans.items():
        if step not in flat or step * size not in (*flat, steps[-1]):
            return None
        replacements[var] = sum(
            (
                symbols[d] * (cut // step)  # pyrefly: ignore [unsupported-operation]
                for cut, d in flat.items()
                if step <= cut < step * size
            ),
            sympy.S.Zero,
        )
    ranges = {symbols[d]: steps[dims - d] // steps[dims - 1 - d] for d in range(dims)}
    index = sympy.expand(access.index.xreplace(replacements))
    index = V.graph.sizevars.simplify_with_ranges(index, ranges)  # pyrefly: ignore [bad-argument-type]
    coefficients = [index.coeff(symbol) for symbol in symbols]
    if not all(isinstance(c, sympy.Integer) for c in coefficients):
        return None
    rest = index.xreplace(dict.fromkeys(symbols, sympy.S.Zero))
    affine = rest + sum(c * s for c, s in zip(coefficients, symbols, strict=True))  # pyrefly: ignore [unsupported-operation]
    if sympy.expand(index - affine) != 0:
        return None
    return [int(c) for c in coefficients]


def natural_tiling(
    node_schedule: list[NodeScheduleEntry],
    numel: sympy.Expr,
    reduction_numel: sympy.Expr,
    max_reduction_dims: int = MAX_REDUCTION_DIMS,
) -> tuple[list[sympy.Expr], list[sympy.Expr]] | None:
    """The pointwise and reduction dims, outermost first, that split the kernel's
    flat ranges into the dims its tensors are indexed by, so that every load and
    store it can is affine in the tiles' indices; None if a size is not a
    constant. Past MAX_POINTWISE_DIMS pointwise and `max_reduction_dims`
    reduction dims, neighbors are joined (reduction_runs() asks for two).

    The space is first cut wherever an index divides a variable or takes it
    modulo a size, and where its variables meet. Two neighboring dims are then
    joined where every affine access steps from one to the other as along one
    dim (the inner's stride times its size is the outer's stride), and, past the
    dims HelionKernel tiles, where joining makes the fewest bytes of accesses
    non-affine. The mean over H and W of a channels-last [128, 1536, 18, 18]
    tensor reads x at 1536*r + 497664*(x // 1536) + x % 1536 over x of 196608
    and r of 324 elements: the dims [128, 1536] and [324], over which it reads
    in0[tile_y, tile_r, tile_x], an outer reduction (in0[tile_y, :, tile_x] in
    one block)."""
    numel_int, rnumel_int = static_int(numel), static_int(reduction_numel)
    if numel_int is None or rnumel_int is None:
        return None
    found = accesses(node_schedule, numel_int, rnumel_int)
    if found is None:
        return None
    total = numel_int * rnumel_int
    candidates: dict[int, int] = {}
    for access in found:
        for cut in set(boundaries(access)):
            candidates[cut] = candidates.get(cut, 0) + access.weight
    steps = chain(total, rnumel_int, candidates)
    affine = []
    for access in found:
        access_strides = strides(access, steps)
        if access_strides is not None:
            affine.append((access, access_strides))
    # Each dim, outermost first: its flat step and size.
    dims = [
        (steps[d], steps[d + 1] // steps[d]) for d in reversed(range(len(steps) - 1))
    ]
    strides_of = [access_strides for _, access_strides in affine]
    weights = [access.weight for access, _ in affine]

    def joinable(d: int) -> list[bool]:
        """Whether each affine access stays affine with dims d and d + 1 joined."""
        return [s[d] == s[d + 1] * dims[d + 1][1] for s in strides_of]

    def join(d: int) -> None:
        keep = joinable(d)
        (_, outer), (step, inner) = dims[d], dims[d + 1]
        dims[d : d + 2] = [(step, outer * inner)]
        for i in reversed(range(len(strides_of))):
            if keep[i]:
                s = strides_of[i]
                s[d : d + 2] = [s[d + 1]]
            else:
                del strides_of[i], weights[i]

    def same_kind(d: int) -> bool:
        # A pointwise dim and a reduction dim are never joined.
        return (dims[d][0] >= rnumel_int) == (dims[d + 1][0] >= rnumel_int)

    # (the closure below sees rnumel_int as optional: this is its int)
    rdim = rnumel_int

    def r_invariant(i: int) -> bool:
        """Whether affine access i is r-invariant: no stride along a
        reduction dim, so it reads one element per pointwise index, as the
        label a cross-entropy reads for each (y, x)."""
        return all(
            stride == 0
            for stride, (step, _) in zip(strides_of[i], dims, strict=True)
            if step < rdim
        )

    # Over rdims longer than R_INVARIANT_JOIN_MIN, pointwise dims that only
    # r-invariant accesses keep apart are joined too: such an access is then
    # read flat, once per pointwise index, where the kernel's other accesses
    # read each index's rdim through a tile of one axis. See
    # R_INVARIANT_JOIN_MIN.
    join_r_invariant = rnumel_int > R_INVARIANT_JOIN_MIN
    progress = True
    while progress:
        progress = False
        for d in range(len(dims) - 1):
            if same_kind(d) and (
                all(joinable(d))
                or (
                    join_r_invariant
                    and dims[d][0] >= rnumel_int
                    and all(ok or r_invariant(i) for i, ok in enumerate(joinable(d)))
                )
            ):
                join(d)
                progress = True
                break
    for limit, pointwise in ((MAX_POINTWISE_DIMS, True), (max_reduction_dims, False)):
        while sum((step >= rnumel_int) == pointwise for step, _ in dims) > limit:
            pairs = [
                d
                for d in range(len(dims) - 1)
                if same_kind(d) and (dims[d][0] >= rnumel_int) == pointwise
            ]
            costs = {
                d: sum(w for w, ok in zip(weights, joinable(d), strict=True) if not ok)
                for d in pairs
            }
            join(min(pairs, key=lambda d: costs[d]))
    pointwise_dims = [size for step, size in dims if step >= rnumel_int and size > 1]
    reduction_dims = [size for step, size in dims if step < rnumel_int and size > 1]
    # A pointwise kernel's tiling has the reduction dim r0_ of 1, as Inductor's.
    return (
        [sympy.Integer(size) for size in pointwise_dims or [numel_int]],
        [sympy.Integer(size) for size in reduction_dims or [rnumel_int]],
    )


def broadcast_runs(
    node_schedule: list[NodeScheduleEntry],
    numel: sympy.Expr,
    reduction_numel: sympy.Expr,
    runs: tuple[int, int],
) -> bool:
    """Whether a reduction of the kernel reduces a value that does not vary
    along the elements of a run or along the runs (reduction_runs(): Q runs of
    W elements, the flat rdim r = W * q + w), as x[0].expand_as(x).sum([0,
    2]) reads x at w + W*c, whatever q. Read as the runs' tile, such a value
    reaches tile_q, or the rdim, by a broadcast alone. A node's value is
    taken not to vary along w (or q) when every load and index expression of
    its body reads one element along it, a load of a buffer that a node
    before it in the kernel writes counting as that node's value (y =
    x[0].expand_as(x) * 2 stored and summed in one kernel). A gather is read
    by its own terms with its indirect values held fixed: those come from
    loads of the same body, which count too, so table[idx] reads one element
    along q wherever idx does (table[idx][None].expand(4, 64, 128).sum([0,
    2]), idx of [64, 128]). A node whose sizes are not constants counts as
    one whose value does not vary."""
    numel_int, rnumel_int = static_int(numel), static_int(reduction_numel)
    if numel_int is None or rnumel_int is None:
        return True
    _, run = runs
    # The first element of three values of x, of run 0: against element 1 of the
    # run (along w), and against element 0 of run 1 (along q).
    xs = sorted({0, numel_int // 2, numel_int - 1})
    offsets = ([1] if run > 1 else []) + [run]
    # Per buffer a node of the kernel writes: whether its value is constant
    # along each offset.
    written: dict[str, tuple[bool, ...]] = {}
    for node in EnableReduction.filter(node_schedule):
        if not isinstance(node, SchedulerNode):
            continue
        body = node._body
        variables = [*body.iter_vars, *body.reduce_vars]
        sizes = [static_int(body.var_ranges[var]) for var in variables]
        if any(size is None for size in sizes):
            return True
        spans: dict[sympy.Symbol, tuple[int, int]] = {}
        step = 1
        for var, size in reversed([*zip(variables, sizes, strict=True)]):
            spans[var] = (step, size)  # pyrefly: ignore [unsupported-operation]
            step *= size  # pyrefly: ignore [unsupported-operation]
        constant = [True] * len(offsets)
        for kind in (MemoryUsageType.LOAD, MemoryUsageType.INDEX_EXPR):
            for entry in body.memory_usage[kind]:
                if entry.buffer_name in written:
                    flags = written[entry.buffer_name]
                    constant = [a and b for a, b in zip(constant, flags, strict=True)]
                    continue
                index = body.indexing_exprs[entry.index_name].replace(
                    Identity, lambda arg: arg
                )
                assert isinstance(index, sympy.Expr)
                constant = [
                    keep
                    and all(
                        _value_at(index, spans, x * rnumel_int)
                        == _value_at(index, spans, x * rnumel_int + offset)
                        for x in xs
                    )
                    for keep, offset in zip(constant, offsets, strict=True)
                ]
        if node.is_reduction() and any(constant):
            return True
        for entry in (
            *body.memory_usage[MemoryUsageType.STORE],
            *body.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name is not None:
                written[entry.buffer_name] = tuple(constant)
    return False


def _value_at(
    index: sympy.Expr, spans: dict[sympy.Symbol, tuple[int, int]], flat: int
) -> object:
    """`index` at flat position `flat` of the space its variables' `spans`
    place them in (accesses())."""
    return index.subs(
        {var: (flat // step) % size for var, (step, size) in spans.items()}
    )


def reduction_runs(
    node_schedule: list[NodeScheduleEntry],
    numel: sympy.Expr,
    reduction_numel: sympy.Expr,
) -> tuple[int, int] | None:
    """The runs a kernel reduces over, (Q, W): Q runs of W elements, when its
    rdim is two dims (natural_tiling()) whose outer one has 2 to
    REDUCTION_RUNS_MAX elements, as the sum over N and W of a [4, 1024, 364]
    tensor reduces 4 runs of 364 elements for each of the 1024 channels.
    HelionKernel reads the runs as a tile of its outer loop beside one run of
    the rdim (HelionKernel.rdim_runs()), at an index affine in both where the
    flat rdim takes // and % of it; None when the rdim is one dim (a
    contiguous rdim) or has more runs."""
    dims = natural_tiling(node_schedule, numel, reduction_numel, max_reduction_dims=2)
    if dims is None:
        return None
    _, reduction_dims = dims
    if len(reduction_dims) != 2:
        return None
    runs, run = (static_int(size) for size in reduction_dims)
    if runs is None or run is None or not 2 <= runs <= REDUCTION_RUNS_MAX:
        return None
    return runs, run
