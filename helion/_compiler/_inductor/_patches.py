"""Monkey-patches for torch._inductor classes to support Helion fusion.

Patches added here correspond to extension points introduced in PyTorch >= 2.11
(PR #174148, PR #175186).  When running on an older torch the methods/call-sites
are added at import time so that Helion's epilogue / prologue fusion works
regardless of the PyTorch version.

Every patch is guarded: if the target already exists (i.e. torch >= 2.11),
the patch is skipped.
"""

from __future__ import annotations

import functools

_applied = False


def apply_patches() -> None:
    """Apply all monkey-patches (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True

    _patch_simd_kernel_codegen_template_override()
    _patch_kernel_emit_kernel_override()
    _patch_template_buffer_is_multi_outputs_template()
    _patch_template_buffer_can_fuse_multi_output_epilogue()
    _patch_base_scheduling_can_fuse_multi_outputs_template()
    _patch_simd_scheduling_codegen_single_template()
    _patch_triton_scheduling_define_kernel()
    _patch_is_multi_outputs_template_util()
    _patch_triton_template_buffer_mutated_inputs_allowlist()


# ---------------------------------------------------------------------------
# Patch 1: SIMDKernel.codegen_template_override  (default → returns None)
# ---------------------------------------------------------------------------
def _patch_simd_kernel_codegen_template_override() -> None:
    from torch._inductor.codegen.simd import SIMDKernel

    if hasattr(SIMDKernel, "codegen_template_override"):
        return

    def codegen_template_override(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    SIMDKernel.codegen_template_override = codegen_template_override  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Patch 2: Kernel.emit_kernel_override  (default → returns False)
# ---------------------------------------------------------------------------
def _patch_kernel_emit_kernel_override() -> None:
    from torch._inductor.codegen.common import Kernel

    if hasattr(Kernel, "emit_kernel_override"):
        return

    def emit_kernel_override(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return False

    Kernel.emit_kernel_override = emit_kernel_override  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Patch 3: TemplateBuffer.is_multi_outputs_template
# ---------------------------------------------------------------------------
def _patch_template_buffer_is_multi_outputs_template() -> None:
    from torch._inductor.ir import MultiOutputLayout, TemplateBuffer

    if hasattr(TemplateBuffer, "is_multi_outputs_template"):
        return

    def is_multi_outputs_template(self) -> bool:  # type: ignore[no-untyped-def]
        return isinstance(self.layout, MultiOutputLayout)

    TemplateBuffer.is_multi_outputs_template = is_multi_outputs_template  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Patch 4: TemplateBuffer.can_fuse_multi_output_epilogue  (default → False)
# ---------------------------------------------------------------------------
def _patch_template_buffer_can_fuse_multi_output_epilogue() -> None:
    from torch._inductor.ir import TemplateBuffer

    if hasattr(TemplateBuffer, "can_fuse_multi_output_epilogue"):
        return

    def can_fuse_multi_output_epilogue(self, snode: object) -> bool:  # type: ignore[no-untyped-def]
        return False

    TemplateBuffer.can_fuse_multi_output_epilogue = can_fuse_multi_output_epilogue  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Patch 5: BaseScheduling.can_fuse_multi_outputs_template
#   Replace to delegate to TemplateBuffer methods.
# ---------------------------------------------------------------------------
def _patch_base_scheduling_can_fuse_multi_outputs_template() -> None:
    from torch._inductor.ir import MultiOutput, TemplateBuffer
    from torch._inductor.scheduler import BaseScheduling

    # Only patch if the default implementation returns False unconditionally
    # (i.e. the upstream extension point hasn't landed yet).
    # We detect this by checking the source — if it's a trivial "return False",
    # we replace it; otherwise we leave it alone.
    orig = BaseScheduling.can_fuse_multi_outputs_template
    try:
        import inspect

        src = inspect.getsource(orig)
        if "return False" not in src:
            return
    except (OSError, TypeError):
        pass

    def can_fuse_multi_outputs_template(self, node1, node2):  # type: ignore[no-untyped-def]
        # get_template_node() works for both SchedulerNode and
        # FusedSchedulerNode (which wraps template + MultiOutput children).
        n1 = node1.get_template_node()
        if not isinstance(n1, TemplateBuffer):
            return False
        if not n1.is_multi_outputs_template():
            return False
        n2 = getattr(node2, "node", None)
        if isinstance(n2, MultiOutput):
            return (
                len(n2.inputs) == 1
                and n2.inputs[0].get_name() == n1.get_name()
            )
        return n1.can_fuse_multi_output_epilogue(node2)

    BaseScheduling.can_fuse_multi_outputs_template = can_fuse_multi_outputs_template  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Patch 6: SIMDScheduling._codegen_single_template
#   Wrap the original to call kernel.codegen_template_override() after
#   building buf_name_to_prologue_group.  If it returns non-None, use that.
# ---------------------------------------------------------------------------
def _patch_simd_scheduling_codegen_single_template() -> None:
    from torch._inductor.codegen.simd import SIMDScheduling

    orig = SIMDScheduling._codegen_single_template

    # If upstream already wraps this (detectable by a marker attribute),
    # do not double-patch.
    if getattr(orig, "_helion_patched", False):
        return

    @functools.wraps(orig)
    def _codegen_single_template(  # type: ignore[no-untyped-def]
        self,
        kernel,
        render,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        *,
        only_gen_src_code=False,
    ):
        from torch._inductor.analyze_preserves_zero_mask import (
            prologue_preserves_zero_mask,
        )

        # Build buf_name_to_prologue_group (mirrors the original logic)
        buf_name_to_prologue_group: dict[str, list[object]] = {}
        template_reads = template_node.used_buffer_names()
        prologue_group: list[object] = []
        for prologue in prologue_nodes:
            names = prologue.get_buffer_names()
            prologue_group.append(prologue)
            if names & template_reads:
                assert len(names) == 1
                buf_name_to_prologue_group[next(iter(names))] = prologue_group
                kernel.prologue_fused_inputs.add(next(iter(names)))
                prologue_group = []
        assert len(prologue_group) == 0

        # --- extension point ---
        override = kernel.codegen_template_override(
            self,
            template_node,
            epilogue_nodes,
            prologue_nodes,
            buf_name_to_prologue_group,
            prologue_preserves_zero_mask,
            render,
            only_gen_src_code,
        )
        if override is not None:
            return override

        # Fall through to original implementation.
        # We need to undo the prologue_fused_inputs additions since the
        # original _codegen_single_template rebuilds them from scratch.
        for buf_name in buf_name_to_prologue_group:
            kernel.prologue_fused_inputs.discard(buf_name)
        return orig(
            self,
            kernel,
            render,
            template_node,
            epilogue_nodes,
            prologue_nodes,
            only_gen_src_code=only_gen_src_code,
        )

    _codegen_single_template._helion_patched = True  # type: ignore[attr-defined]
    SIMDScheduling._codegen_single_template = _codegen_single_template  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Patch 7: TritonScheduling.define_kernel
#   Wrap to call kernel.emit_kernel_override() first; if it returns True
#   the kernel name is returned directly without the default async_compile
#   logic.
# ---------------------------------------------------------------------------
def _patch_triton_scheduling_define_kernel() -> None:
    from torch._inductor.codegen.triton import TritonScheduling

    orig = TritonScheduling.define_kernel
    if getattr(orig, "_helion_patched", False):
        return

    @functools.wraps(orig)
    def define_kernel(self, src_code, node_schedule, kernel):  # type: ignore[no-untyped-def]
        from torch._inductor.codecache import code_hash, get_path
        from torch._inductor.codegen.triton import get_kernel_metadata
        from torch._inductor.codegen.triton_utils import config
        from torch._inductor.utils import Placeholder
        from torch._inductor.virtualized import V

        wrapper = V.graph.wrapper_code

        # Check cache first (same as original)
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        # Generate kernel name (same as original)
        from torch._inductor.codegen.triton import (
            get_fused_kernel_name,
            get_kernel_category_by_source_code,
        )

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_category = get_kernel_category_by_source_code(src_code)[:3]
        kernel_name = "_".join(
            ["triton", kernel_category, fused_name, wrapper.next_kernel_suffix()]
        )
        if config.aot_inductor.model_name_for_generated_files:
            kernel_name = f"{config.aot_inductor.model_name_for_generated_files}_{kernel_name}"

        wrapper.src_to_kernel[src_code] = kernel_name
        subs_name = kernel_name if config.triton.unique_kernel_names else "triton_"

        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)
        src_code = src_code.replace("#pragma CMT", "#")

        _basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")

        # --- extension point ---
        if kernel.emit_kernel_override(
            wrapper,
            src_code,
            kernel_name,
            node_schedule,
            kernel_path,
            get_kernel_metadata,
        ):
            return kernel_name

        # Fall through to default behaviour (async_compile.triton)
        from torch._inductor.codegen.common import IndentedBuffer
        from torch._inductor.codegen.triton import async_compile

        compile_wrapper = IndentedBuffer()

        if async_compile.use_process_pool():
            async_compile.triton(subs_name, src_code)

        compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
        compile_wrapper.splice(src_code, strip=True)
        current_device = V.graph.get_current_device_or_throw()
        compile_wrapper.writeline(f"''', device_str='{current_device.type}')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(
            kernel_name, compile_wrapper.getvalue(), metadata_comment
        )

        from torch._inductor.metrics import is_metric_table_enabled, log_kernel_metadata

        if is_metric_table_enabled("kernel_metadata"):
            log_kernel_metadata(kernel_name, kernel_path, src_code)

        return kernel_name

    define_kernel._helion_patched = True  # type: ignore[attr-defined]
    TritonScheduling.define_kernel = define_kernel  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Patch 8: torch._inductor.utils.is_multi_outputs_template
#   Replace to use TemplateBuffer.is_multi_outputs_template() instead of
#   hardcoding CppTemplateBuffer.
# ---------------------------------------------------------------------------
def _patch_is_multi_outputs_template_util() -> None:
    import torch._inductor.utils as inductor_utils
    from torch._inductor.ir import TemplateBuffer

    orig = inductor_utils.is_multi_outputs_template

    # If upstream already delegates to TemplateBuffer.is_multi_outputs_template,
    # skip the patch.
    try:
        import inspect

        src = inspect.getsource(orig)
        if "is_multi_outputs_template()" in src:
            return
    except (OSError, TypeError):
        pass

    def is_multi_outputs_template(input_buf):  # type: ignore[no-untyped-def]
        if isinstance(input_buf, TemplateBuffer):
            return input_buf.is_multi_outputs_template()
        return False

    inductor_utils.is_multi_outputs_template = is_multi_outputs_template


# ---------------------------------------------------------------------------
# Patch 9: TritonTemplateBuffer.__init__ mutated-inputs allowlist
#   Expand the hardcoded flex_attention-only allowed set to include Helion HOPs.
# ---------------------------------------------------------------------------
def _patch_triton_template_buffer_mutated_inputs_allowlist() -> None:
    from torch._inductor.ir import TritonTemplateBuffer

    orig_init = TritonTemplateBuffer.__init__

    # If the original __init__ doesn't contain the restrictive assertion,
    # skip the patch.
    try:
        import inspect

        src = inspect.getsource(orig_init)
        if "Mutated inputs are only allowed for" not in src:
            return
    except (OSError, TypeError):
        pass

    if getattr(orig_init, "_helion_patched", False):
        return

    @functools.wraps(orig_init)
    def __init__(self, layout, inputs, make_kernel_render, mutated_inputs=None, allowed_prologue_inps=None):  # type: ignore[no-untyped-def]
        from torch._inductor.ir import (
            Buffer,
            IRNode,
            MutationOutput,
            NoneLayout,
            TemplateBuffer,
        )
        from torch.utils._ordered_set import OrderedSet

        TemplateBuffer.__init__(self, layout, inputs, make_kernel_render)
        self.mutated_inputs = mutated_inputs
        self.outputs: list[Buffer] = [self]
        if mutated_inputs is not None:
            # Removed the restrictive assertion that only allows
            # flex_attention HOPs; Helion HOPs also need mutated inputs.
            assert isinstance(self.inputs[0], IRNode), type(self.inputs[0])
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

        self.allowed_prologue_inps = (
            allowed_prologue_inps if allowed_prologue_inps else OrderedSet()
        )
        self.subgraph_inps = None
        self.subgraph_outs = None

    __init__._helion_patched = True  # type: ignore[attr-defined]
    TritonTemplateBuffer.__init__ = __init__  # type: ignore[assignment]
