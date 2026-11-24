from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
from torch._inductor import config
from torch._inductor import metrics
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.common import WorkspaceZeroMode
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.simd_kernel_features import DisableReduction
from torch._inductor.codegen.simd_kernel_features import EnableReduction
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.ir import Reduction
from torch._inductor.ops_handler import OpsHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.scheduler import FusedMixOrderReductions
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import get_fused_kernel_name
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Where

from ..._compiler._inductor.template_buffer import HelionTemplateBuffer
from ..._compiler.backend_registry import get_backend
from .. import config as helion_config
from ..choices import HelionChoices
from ..choices import fused_reindex
from ..choices import restore_unfused_reindex
from ..support import mix_order_split
from ..support import numerics_modes
from ..support import python_wrapper
from .kernel import ARG_REDUCTIONS
from .kernel import REDUCTIONS
from .kernel import HelionKernel
from .overrides import HelionOverrides
from .sequence import module_key
from .sequence import predecessor
from .tiling import UNROLLED_RDIM_MAX
from .tiling import natural_tiling

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    import torch
    from torch._inductor.codegen.multi_kernel import MultiKernel
    from torch._inductor.codegen.simd import SIMDKernel
    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import ComputedBuffer
    from torch._inductor.scheduler import FusedSchedulerNode
    from torch._inductor.scheduler import Scheduler
    from torch._inductor.tiling_utils import CoalesceVarAnalysis


# Ops HelionOverrides inherits from OpOverrides that mean the same in a Helion
# kernel: Python operators (BasicMathOpsMixin's a + b, a < b, ~a), which mean the
# same on torch tensors, and load_seed, ops.load(seeds, i), which reads seeds[i]
# through a 0-d view of the seed buffer, reinterpret_tensor(buf0, (), (), i):
#     tmp0 = in0[()]
# Not every inherited spelling does: BasicMathOpsMixin.mod writes a % b, which
# floors on torch tensors (-7 % 2 == 1) while Inductor's mod truncates (-1).
INHERITED_OPS = OrderedSet(
    [
        "load_seed",
        "add",
        "sub",
        "mul",
        "truediv",
        "neg",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and_",
        "or_",
        "xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
    ]
)
# Every op a Helion kernel emits: those HelionKernel (load, store, dot,
# check_bounds, ...) or HelionOverrides (exp, where, ...) implements, the
# inherited operators, SIMDKernel.store_reduction (which calls HelionKernel.store),
# indirect_indexing (whose symbol HelionKernel's indices read, see
# indirect_vars()) and the body's output. check_bounds, which the body of
# x.gather(1, arange-based index) holds next to its direct index, passes when
# Inductor proves the index in range (from its value ranges), and fails the
# compile with "Unsupported: check_bounds" otherwise.
HELION_OPS = (
    OrderedSet(
        name
        for cls in (HelionKernel, HelionOverrides)
        for name in vars(cls)
        if not name.startswith("_") and name in vars(OpsHandler)
    )
    | INHERITED_OPS
    | OrderedSet(["store_reduction", "indirect_indexing", "output"])
)


@functools.cache
def backend_ops(backend: str) -> OrderedSet[str]:
    """HELION_OPS less the ops that Helion's `backend` cannot compile.
    HelionOverrides writes an op as a torch call, e.g. torch.lgamma(x), which
    Helion compiles through the backend's Inductor op overrides. CuTe's have no
    lgamma, so a CuTe kernel with it would fail to compile with
    NotImplementedError; unsupported() names it instead: "Unsupported: lgamma".
    index_expr, value_expr and masked are HelionKernel code (hl.load, tile
    indices) rather than torch calls, rand is hl.rand(...), which CuTe
    compiles though its op overrides have no rand, and randint64 a helper of
    torch ops (HelionKernel.randint64())."""
    cls = type(get_backend(backend).inductor_op_overrides())
    torch_calls = OrderedSet(
        name
        for name in vars(HelionOverrides)
        if name in vars(OpsHandler)
        and name not in ("index_expr", "value_expr", "masked", "rand", "randint64")
    )
    return HELION_OPS - OrderedSet(
        name for name in torch_calls if getattr(cls, name) is getattr(OpsHandler, name)
    )


def has_dot(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds a native matmul, which only HelionChoices lowers with
    cuda_backend "helion"."""
    return any(snode.is_native_matmul() for snode in node.get_nodes())


class Unsupported(RuntimeError):
    """A kernel that HelionKernel cannot emit yet. With cuda_backend "helion",
    it fails the compile, as HalideScheduling's Unsupported does."""


@functools.cache
def supported(backend: str) -> OrderedSet[str]:
    """The op and reduction names HelionKernel emits on Helion's `backend`."""
    return backend_ops(backend) | OrderedSet([*REDUCTIONS, "dot"])


def unsupported(node: SchedulerNode) -> list[str]:
    """What `node` uses that HelionKernel does not support: a dynamic shape, its
    op and reduction names that supported() lacks, e.g. randn_like's "randn",
    the numerics modes that are on, a CPU scalar input, a dynamic index, a
    direct index with a Where and an arg reduction over several dims."""
    body = node._body
    buffer = cast("ComputedBuffer", node.node)
    reduction_type = buffer.get_reduction_type()
    # HelionKernel tiles static sizes (supports_tiling()).
    if not all(isinstance(numel, (int, sympy.Integer)) for numel in node.group[1]):
        return ["dynamic_shape"]
    names = OrderedSet([*body.op_counts, *([reduction_type] if reduction_type else [])])
    # An arg reduction over several dims gets the flat index of its reduction
    # ranges, which loop reordering permutes: for torch.argmax(x) of a
    # transposed x, tile_r runs over x's storage, so its index is the storage
    # index of the max, not x's. Inductor passes the logical index along with
    # the value to Triton, MPS and C++ only.
    if isinstance(buffer.data, Reduction) and reduction_type in ARG_REDUCTIONS:
        dims = [size for size in buffer.data.reduction_ranges if size != 1]
        if len(dims) > 1:
            names.add("arg_reduction_over_dims")
    missing = [*sorted(names - supported(helion_config.backend)), *numerics_modes()]
    # x * torch.tensor(2.0): the kernel gets a CPU tensor, "Pointer argument
    # cannot be accessed from Triton (cpu tensor?)".
    if any(V.graph.is_unspec_arg(dep.name) for dep in node.read_writes.reads):
        missing.append("cpu_scalar_input")
    # x[:, :50] * 2 with a dynamic dim 1: the index holds that dim's stride,
    # "Unsupported Helion index: s27*((xindex//50)) + ...". An index may read
    # values (indirect0 in x.gather(1, i)), into a static dim: x[i] with a
    # dynamic dim 0 wraps a negative i by i + s0, "Unsupported Helion index: s0".
    known = OrderedSet([*body.var_ranges, *body.indirect_var_ranges])
    if not all(
        index.free_symbols <= known for index in body.indexing_exprs.values()
    ) or not all(
        isinstance(size, (int, sympy.Integer))
        for size in body.indirect_var_ranges.values()
    ):
        missing.append("dynamic_index")
    # x[torch.arange(10) - 1] wraps -1 at Where(p0 - 1 < 0, p0 + 9, p0 - 1),
    # which HelionKernel's index printer and bounds do not take.
    if any(index.has(Where) for index in body.indexing_exprs.values()):
        missing.append("where_index")
    return missing


def check_supported(nodes: Iterable[BaseSchedulerNode]) -> None:
    """Raises Unsupported for a C++ or FX wrapper, or for what one of `nodes`
    uses that HelionKernel does not support (unsupported())."""
    if not python_wrapper():
        raise Unsupported("cpp_wrapper" if V.graph.cpp_wrapper else "fx_wrapper")
    if missing := OrderedSet(
        name for node in nodes for name in unsupported(cast("SchedulerNode", node))
    ):
        raise Unsupported(", ".join(missing))


class HelionScheduling(SIMDScheduling):
    """Inductor's CUDA scheduling for cuda_backend "helion", as HalideScheduling
    is for "halide": every kernel is a Helion kernel, and a kernel HelionKernel
    cannot emit fails the compile with Unsupported."""

    kernel_type = HelionKernel  # pyrefly: ignore [bad-assignment]

    def __init__(self, scheduler: Scheduler | None) -> None:
        super().__init__(scheduler)
        self._template_scheduling = TritonScheduling(scheduler)

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        # Only what HelionKernel emits: in-place buffers, where an output takes
        # over the storage of an input of the kernel (kernel_params()). Without
        # the others, Inductor lowers scan, sort, bucketize and the like to
        # aten kernels rather than to ops Helion lacks.
        return OrderedSet([BackendFeature.INPLACE_BUFFERS])

    @classmethod
    def get_tiling_and_scores(
        cls,
        node_schedule: list[Any],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
        coalesce_analysis: CoalesceVarAnalysis | None = None,
    ) -> tuple[dict[str, sympy.Expr], dict[str, sympy.Expr] | None]:
        """The N-D tiling that splits the kernel's flat ranges into the dims its
        tensors are indexed by (natural_tiling()), if it has more dims than one
        pointwise and one reduction dim; else Inductor's. Helion tiles N-D
        natively, and the kernel then indexes its tensors by tile instead of by
        // and % of a flat index. For the mean over H and W of a channels-last
        [128, 1536, 18, 18] tensor, the 1-D tiling {"x": 196608, "r0_": 324}
        reads it at

            1536*r + 497664*(tile_x.index // 1536) + tile_x.index % 1536

        and this one, {"y": 128, "x": 1536, "r0_": 324}, at

            1536*r + 497664*tile_y.index + tile_x.index

        a tile of a [128, 324, 1536] view, which the kernel subscripts by its
        tiles. Helion lowers the // and % of a reshaped index with sign
        corrections (it no longer knows the index is non-negative), and the
        backend does not see that consecutive tile_x are consecutive addresses,
        so its loads do not vectorize. A schedule with a matmul keeps Inductor's tiling,
        which follows the dot's dims."""
        inductor_tiling, scores = super().get_tiling_and_scores(
            node_schedule, numel, reduction_numel, coalesce_analysis
        )
        if not any(
            isinstance(node, BaseSchedulerNode) and has_dot(node)
            for node in node_schedule
        ) and (dims := natural_tiling(node_schedule, numel, reduction_numel)):
            tiling = cls.create_tiling(*dims)
            # More than one pointwise dim beside r0_, of Integer sizes.
            if len(tiling) > 2 and cls.tiling_is_compatible(
                node_schedule, numel, reduction_numel, tiling
            ):
                # Inductor's scores, which say whether the reduction reads
                # contiguous memory (SIMDKernelFeatures.get_reduction_hint()):
                # without them an online softmax over rdims of 1001 elements
                # gets the hint DEFAULT, and Triton's rule (persistent up to 64
                # elements, not 1024) loops over them twice.
                return tiling, scores
        return inductor_tiling, scores

    @staticmethod
    def format_kernel_definition(
        name: str, source: str, after: str | None = None, after_key: str = ""
    ) -> str:
        escaped = source.replace("\\", "\\\\").replace('"""', '\\"""')
        # A kernel tuned behind another loads its module under that kernel's
        # source (after_key): PyCodeCache gives graphs with the same source
        # one module and one Kernel, whose config the first graph to call it
        # selects, so a graph that calls the kernel behind other work must
        # not get it. In the same graph's code, behind the same kernel, it is
        # the same module.
        load = (
            f"PyCodeCache.load({name}_source{f', {after_key!r}' if after_key else ''})"
        )
        definition = f'{name}_source = """\\\n{escaped}"""\n{name} = {load}.kernel'
        if helion_config.tune_in_sequence:
            # The kernel's first call is tuned behind the first call of
            # kernel ``after``, the Helion kernel call just before it, or
            # alone (helion.inductor.codegen.sequence).
            definition += (
                f'\n{name} = helion_in_sequence({name}, globals(), "{name}", '
                f"after={after!r})"
            )
        return definition

    def generate_node_schedule(
        self,
        nodes: Sequence[BaseSchedulerNode],
        numel: sympy.Expr,
        rnumel: sympy.Expr,
        *,
        required_post_reduction_index: int | None = None,
    ) -> list[Any]:
        # Dots first, so that they share one K loop. No dot reads another node
        # of its kernel, so this keeps every producer before its use.
        if required_post_reduction_index is None and any(map(has_dot, nodes)):
            nodes = sorted(nodes, key=lambda node: not has_dot(node))
        return super().generate_node_schedule(
            nodes,
            numel,
            rnumel,
            required_post_reduction_index=required_post_reduction_index,
        )

    def codegen_node_schedule(
        self, kernel_features: SIMDKernelFeatures
    ) -> SIMDKernel[Any] | MultiKernel:
        check_supported(kernel_features.scheduler_nodes())
        return super().codegen_node_schedule(kernel_features)

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """SIMDScheduling.can_fuse() without its check of a pointwise pair's
        tilings (config.triton.tiling_prevents_pointwise_fusion), which
        compares each node's tiling, from Inductor's select_tiling()
        (BaseSchedulerNode.get_tiling()), with the pair's, from
        natural_tiling() here (get_tiling_and_scores()): y = x.permute(0, 2,
        3, 1).contiguous() and y * 2, a pair natural_tiling() tiles [8, 1024,
        64], were two kernels, and y * 2 read y again. natural_tiling() cuts a
        fused kernel's space wherever any of its nodes' accesses need it, so
        the pair's tiling serves both nodes."""
        with config.patch({"triton.tiling_prevents_pointwise_fusion": False}):
            return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse

    def _native_matmul_row_epilogue_can_fuse(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # SIMDScheduling fuses a native matmul with reductions over its rows,
        # (a @ b).sum(-1) with N <= 128 and K <= 512, into a kernel only
        # TritonKernel emits: "NotImplementedError: NYI:
        # iteration_ranges_codegen_header". Unfused, the dot and the sum are
        # Helion kernels of their own.
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # SIMDScheduling asks MixOrderReduction, which takes Triton kernels only.
        return HelionChoices.mix_order_reductions(node1, node2) or self.can_fuse(
            node1, node2
        )

    def fuse(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        if HelionChoices.mix_order_reductions(node1, node2):
            return FusedMixOrderReductions(node1, node2)
        fused = super().fuse(node1, node2)
        fused_reindex(fused)
        return fused

    def codegen_node(
        self, node: FusedSchedulerNode | SchedulerNode
    ) -> SIMDKernel[Any] | MultiKernel | None:
        # A pointwise node reindexed onto a reduction that did not take it
        # walks its own loops again (reindex_onto_reduction()).
        restore_unfused_reindex(node)
        return super().codegen_node(node)

    def codegen_mix_order_reduction(self, node: FusedMixOrderReductions) -> None:
        """One Helion kernel for a mix-order pair, reductions over r and over x
        of one [x, r] array (HelionChoices.mix_order_reductions()):
        SIMDScheduling's codegen, with Helion's split of x (mix_order_split(),
        a divisor of x, whose view [splits, split, r] the kernel indexes) in
        place of its search over powers of two. The kernel tiles the splits
        (tile_y) and loops over the x of a split (tile_x), reading each rdim as
        one block; it sums the terms of the reduction over x across its split
        into a partial, which it writes to a workspace that the wrapper reduces
        over the splits. For RMSNorm's
        backward over [1152000, 512]:
            for tile_y in hl.tile(9000):
                rdim = hl.zeros([tile_y, 512]).size(1)
                part0 = hl.zeros([tile_y, rdim], dtype=torch.float32)
                for tile_x in hl.tile(128):
                    tmp3 = in2[tile_y, tile_x, :].to(torch.float32)
                    ...
                    out0[tile_y, tile_x, :] = tmp20.to(torch.bfloat16)
                    part0 = part0 + torch.sum(tmp25, 1)
                ws[0, tile_y, :] = part0
        and then
            buf3 = workspace_0[0 : 4608000].view(9000, 512).sum(dim=0).view([512])"""
        check_supported(node.get_nodes())
        numel, _ = MixOrderReduction.get_numel_rnumel(node.node1)
        device = node.node1.get_device()
        assert device is not None
        split = mix_order_split(int(numel), device)
        if split is None:
            raise AssertionError(f"no split of an x of {numel}")
        with config.patch({"triton.mix_order_reduction_split_size": split}):
            super().codegen_mix_order_reduction(node)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        **kwargs: Any,  # noqa: ANN401
    ) -> str | None:
        # A @helion.kernel called in the compiled function is Helion's own
        # template, which renders its Helion kernel and emits it through
        # TritonScheduling (emit_kernel_override), as with cuda_backend "triton".
        # Emitted as a generated kernel instead, its source fails to load:
        # "SyntaxError: expected '('". The prologues and epilogues fused into it
        # (torch_compile_fusion) need Inductor's dtype propagation for Triton,
        # without which "AssertionError: TritonCSEVariable must have dtype".
        # Other templates (CUTLASS's) are not Helion kernels.
        if not isinstance(template_node.node, HelionTemplateBuffer):
            raise Unsupported("template")
        with config.patch(cuda_backend="triton"):
            return self._template_scheduling.codegen_template(
                template_node, epilogue_nodes, prologue_nodes, **kwargs
            )

    def codegen_combo_kernel(self, combo_kernel_node: Any) -> None:  # noqa: ANN401
        # SIMDScheduling's combo kernels are Triton's: "expected self.kernel_type
        # to subclass TritonKernel".
        raise Unsupported("combo_kernels")

    def codegen_node_schedule_with_kernel(
        self, node_schedule: list[Any], kernel: SIMDKernel[Any]
    ) -> None:
        # The nodes create_kernel_choices() emits over segments of the rdim: all
        # of them for one segment, then the next (HelionKernel.rdim_segment()),
        # where a node reads an earlier one's part of the segment through the
        # CSE store cache. The first runs them; the others run within it.
        helion_kernel = cast("HelionKernel", kernel)
        group = [
            node
            for node in node_schedule
            if isinstance(node, SchedulerNode)
            and node.get_name() in helion_kernel.segment_nodes
        ]
        if group:
            emits = {node: node.codegen for node in group}

            def codegen(index_vars: Any) -> None:  # noqa: ANN401
                ranges = {group[0]: index_vars}
                for node in group[1:]:
                    self._prepare_loop_body(node._body)
                    ranges[node] = helion_kernel.split_and_set_ranges(node.get_ranges())
                # Each segment starts from the store cache the group started
                # with: a node that stores a buffer the group reads (x.add_(...)
                # of a load of x) would otherwise hand the next segment its own
                # stored part as x.
                store_cache = dict(helion_kernel.cse.store_cache)
                for lo, hi in helion_kernel.segments:
                    helion_kernel.cse.store_cache.clear()
                    helion_kernel.cse.store_cache.update(store_cache)
                    with helion_kernel.rdim_segment(lo, hi):
                        for node in group:
                            emits[node](ranges[node])

            setattr(group[0], "codegen", codegen)  # noqa: B010
            for node in group[1:]:
                setattr(node, "codegen", lambda index_vars: None)  # noqa: B010
        try:
            super().codegen_node_schedule_with_kernel(node_schedule, kernel)
        finally:
            for node in group:
                del node.codegen
        # What only emitting the kernel shows: an indirect index that needs a
        # bounds check (HelionKernel.check_bounds()), an op the kernel cannot
        # express (HelionKernel.randint64()).
        if helion_kernel.needs_bounds_check:
            raise Unsupported("check_bounds")
        if helion_kernel.unsupported_ops:
            raise Unsupported(", ".join(helion_kernel.unsupported_ops))

    def _codegen_node_schedule_body(
        self, node_schedule: list[Any], kernel: SIMDKernel[Any]
    ) -> None:
        """SIMDScheduling's emission of the node schedule, once for each
        segment of a segmented pointwise axis (HelionKernel.segment_axis()):
        every node of the kernel, for the first segment and then the next
        (HelionKernel.pointwise_segment()). Each segment starts from the store
        cache the first one started with, as an rdim's segments do, so that a
        load of a buffer the kernel stores reads its own segment's part."""
        helion_kernel = cast("HelionKernel", kernel)
        if helion_kernel.axis_segments is None:
            super()._codegen_node_schedule_body(node_schedule, kernel)
            return
        _, segments = helion_kernel.axis_segments
        store_cache = dict(helion_kernel.cse.store_cache)
        for index, (start, _) in enumerate(segments):
            helion_kernel.cse.store_cache.clear()
            helion_kernel.cse.store_cache.update(store_cache)
            with helion_kernel.pointwise_segment(start):
                if index == 0:
                    super()._codegen_node_schedule_body(node_schedule, kernel)
                    continue
                # The emitting pass of _codegen_node_schedule_body(), whose
                # indexing pass the first segment ran.
                with contextlib.ExitStack() as stack:
                    for node in node_schedule:
                        if node is DisableReduction:
                            stack.enter_context(kernel.disable_reduction())
                        elif node is EnableReduction:
                            stack.close()
                        else:
                            node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

    def create_kernel_choices(  # pyrefly: ignore [bad-override]
        self,
        kernel_features: SIMDKernelFeatures,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> Sequence[SIMDKernel[Any]]:
        (tiling,) = kernel_args
        if not HelionKernel.supports_tiling(tiling):
            raise Unsupported(f"tiling {tiling}")
        kernel = HelionKernel(*kernel_args, **kernel_kwargs)
        if kernel.mix_order_reduction:
            return [kernel]
        # Only a node with a masked part (a cat's or pad's ops.masked()) can
        # cut an axis (HelionKernel.pointwise_segments()) or, trailing a
        # persistent reduction, its rdim (rdim_segments()) into segments.
        masked = any(
            isinstance(node, SchedulerNode) and node._body.subblocks
            for node in kernel_features.node_schedule
        )
        if not kernel.persistent_reduction:
            if not masked or kernel.rolled_reduction():
                return [kernel]
            self.probe(kernel_features, kernel)
            return [self.with_pointwise_segments(kernel, kernel_args, kernel_kwargs)]
        # One block holds one run where the runs are a tile (HelionKernel.runs).
        long_rdim = "override_persistent_reduction" not in kernel_kwargs and (
            V.graph.sizevars.statically_known_gt(
                kernel.extents["r0_"], UNROLLED_RDIM_MAX
            )
        )
        group, earlier = self.trailing_nodes(kernel_features.node_schedule)
        cuttable = any(node._body.subblocks for node in group)
        if not long_rdim and not masked:
            return [kernel]
        self.probe(kernel_features, kernel)
        if long_rdim and kernel.uses_rdim and kernel.rdim_anchor is None:
            # An rdim past UNROLLED_RDIM_MAX that the kernel names (rdim) but no
            # input's full rdim does (HelionKernel.rdim_anchor) has a factory's
            # dim, which Helion cannot roll: x.flip(-1).sum(-1) over rdims of
            # 100000 would read each rdim as one block of 131072 lanes. Such a
            # kernel loops over the rdim, as Triton's rule has it. A gather's
            # rdim, in1[tmp1, :], needs no rdim name and stays one block.
            return [
                HelionKernel(
                    *kernel_args, **kernel_kwargs, override_persistent_reduction=False
                )
            ]
        # An rdim past UNROLLED_RDIM_MAX keeps its tail in one emission over the
        # whole rdim, which Helion can roll: a segment's dim is a literal
        # (HelionKernel.rdim_code()), which it cannot.
        segmented = (
            self.rdim_segments(group, earlier, kernel)
            if cuttable and not long_rdim
            else None
        )
        if segmented is None and not long_rdim:
            return [self.with_pointwise_segments(kernel, kernel_args, kernel_kwargs)]
        kernel = HelionKernel(*kernel_args, **kernel_kwargs)
        if segmented is not None:
            kernel.segment_nodes, kernel.segments = segmented
        return [kernel]

    def probe(self, kernel_features: SIMDKernelFeatures, kernel: HelionKernel) -> None:
        """Emit the body into `kernel`, which tells what the kernel needs of its
        rdim and axes (rdim_anchor, rdim_cuts, axis_cuts)."""
        self.codegen_node_schedule_with_kernel(kernel_features.node_schedule, kernel)
        # The kernel emitted to tell is not the one returned: take back the
        # count its Kernel.__init__ added to metrics.generated_kernel_count,
        # which tests compare with the kernels a graph launches.
        metrics.generated_kernel_count -= 1

    @staticmethod
    def with_pointwise_segments(
        probe: HelionKernel, kernel_args: list[Any], kernel_kwargs: dict[str, Any]
    ) -> HelionKernel:
        """A fresh kernel, emitted over the equal segments of the pointwise
        axis whose cuts the `probe` saw (HelionKernel.pointwise_segments()),
        if it saw such cuts."""
        kernel = HelionKernel(*kernel_args, **kernel_kwargs)
        if (segments := probe.pointwise_segments()) is not None:
            kernel.segment_axis(*segments)
        return kernel

    @staticmethod
    def trailing_nodes(
        node_schedule: list[Any],
    ) -> tuple[list[SchedulerNode], list[BaseSchedulerNode]]:
        """The nodes over the rdim after a persistent reduction's last
        reduction, which HelionScheduling.rdim_segments() may emit over
        segments of the rdim, and the nodes before them."""
        group: list[SchedulerNode] = []
        earlier: list[BaseSchedulerNode] = []
        inside = True
        for node in node_schedule:
            if node is DisableReduction:
                inside = False
            elif node is EnableReduction:
                inside = True
            elif isinstance(node, BaseSchedulerNode):
                if (
                    isinstance(node, SchedulerNode)
                    and inside
                    and not node.is_reduction()
                ):
                    group.append(node)
                else:
                    earlier.extend(group)
                    group = []
                    earlier.append(node)
        return group, earlier

    @staticmethod
    def rdim_segments(
        group: list[SchedulerNode],
        earlier: list[BaseSchedulerNode],
        kernel: HelionKernel,
    ) -> tuple[list[str], list[tuple[int, int]]] | None:
        """The nodes of a persistent reduction to emit over segments of the
        rdim (HelionKernel.rdim_segment()), and the segments: the trailing
        nodes (trailing_nodes()), when their comparisons of the rdim index cut
        the rdim (rdim_cut_segments()) and none reads an rdim that a node before
        them writes, whose value is the whole rdim's."""
        segments = kernel.rdim_cut_segments(node.get_name() for node in group)
        if segments is None:
            return None
        # extents: the tile of the runs, where the rdim is one run (rdim_runs()).
        numel = sympy.Mul(
            kernel.extents["r0_"],
            *(kernel.extents[axis] for axis in kernel.outer_axes),
        )
        earlier_full = OrderedSet(
            name
            for node in earlier
            for name in node.get_buffer_names()
            if V.graph.sizevars.statically_known_equals(
                V.graph.get_buffer(name).get_numel(), numel
            )
        )
        if any(
            dep.name in earlier_full for node in group for dep in node.read_writes.reads
        ):
            return None
        return [node.get_name() for node in group], segments

    def define_kernel(
        self, src_code: str, node_schedule: list[Any], kernel: SIMDKernel[Any]
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]
        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        name = "_".join(
            part
            for part in ("helion", fused_name, wrapper.next_kernel_suffix())
            if part
        )
        wrapper.src_to_kernel[src_code] = name
        wrapper.add_import_once("from torch._inductor.codecache import PyCodeCache")
        if helion_config.tune_in_sequence:
            wrapper.add_import_once(
                "from helion.inductor.kernel_sequence import "
                "in_sequence as helion_in_sequence"
            )
        after = after_key = None
        if helion_config.tune_in_sequence and not any(
            # A workspace the call zeroes first is a memset launched between
            # the two calls (generate_workspace_allocation writes it before
            # the call): the kernel is tuned alone.
            workspace.zero_mode != WorkspaceZeroMode.UNINITIALIZED
            for workspace in kernel.args.workspace_args
        ):
            after = predecessor(wrapper)
        if after is not None:
            (after_source,) = (
                source
                for source, kernel_name in wrapper.src_to_kernel.items()
                if kernel_name == after
            )
            after_key = module_key(
                wrapper,
                name,
                after,
                after_source,
                cast("HelionKernel", kernel).call_args(),
            )
        # gpu=False keeps it out of the compile-time autotune block, which has
        # no PyCodeCache import.
        wrapper.define_kernel(
            name,
            self.format_kernel_definition(name, src_code, after, after_key or ""),
            gpu=False,
            standalone=True,
        )
        return name
