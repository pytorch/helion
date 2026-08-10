"""PallasBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

import ast
import dataclasses
import enum
import hashlib
import inspect
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..backend import Backend
from ..backend import LauncherInfo
from ..backend import _loop_contains_matmul
from ..backend import dedupe_preserve_order
from ..backend import read_launcher_source

if TYPE_CHECKING:
    import sympy
    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_fragment import ConfigSpecFragment
    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from ...runtime.settings import DotPrecision
    from ..device_function import Argument
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..tile_dispatch import TileStrategyDispatch
    from .compact_worklist import CompactWorklistPlan

    InductorOpOverrides = OpsHandler[Any]


def _embedded_helper_source(body: str) -> str:
    """Source of the in-kernel Pallas helpers referenced by ``body`` (module-level
    so both ``PallasBackend.embedded_helper_source`` and the jax standalone builder
    can inline them). Only helpers actually referenced are emitted."""
    blocks: list[str] = []
    if "_helion_divide_filter_topk" in body:
        from . import topk_impl

        blocks.extend(
            [
                _embed_source(inspect.getsource(topk_impl)),
                "_helion_divide_filter_topk = divide_filter_topk",
            ]
        )
    if "flatten_worklist" in body:
        from ...runtime import compact_worklist

        blocks.append(_embed_source(inspect.getsource(compact_worklist)))
    return "\n\n\n".join(blocks)


def _embed_source(source: str) -> str:
    """Return a module's source ready to inline: its module docstring and
    ``from __future__`` lines stripped (leading comments -- e.g. an SPDX header --
    and everything else preserved), so the docstring prose can't leak into the
    generated code and the mid-module ``from __future__`` (a SyntaxError) is gone.

    The docstring span is located via ``ast`` (not a quote scan) so a docstring
    whose prose contains a triple-quote can't corrupt the output, and a module
    that opens with code rather than a docstring is handled correctly.
    """
    lines = source.split("\n")
    doc_lines: set[int] = set()
    tree = ast.parse(source)
    if (
        tree.body
        and isinstance(first := tree.body[0], ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        # ast line numbers are 1-based; end_lineno is the closing-quote line.
        doc_lines = set(range(first.lineno - 1, (first.end_lineno or first.lineno)))
    kept = [
        line
        for idx, line in enumerate(lines)
        if idx not in doc_lines and not line.strip().startswith("from __future__")
    ]
    return "\n".join(kept).strip("\n")


# Mapping from torch dtype to JAX dtype string (e.g., "jnp.float32")
_TORCH_TO_JAX_DTYPE: dict[str, str] = {
    "torch.float16": "jnp.float16",
    "torch.float32": "jnp.float32",
    "torch.float64": "jnp.float64",
    "torch.bfloat16": "jnp.bfloat16",
    "torch.int8": "jnp.int8",
    "torch.int16": "jnp.int16",
    "torch.int32": "jnp.int32",
    "torch.int64": "jnp.int64",
    "torch.uint8": "jnp.uint8",
    "torch.uint32": "jnp.uint32",
    "torch.uint64": "jnp.uint64",
    "torch.bool": "jnp.bool_",
    "torch.complex64": "jnp.complex64",
    "torch.complex128": "jnp.complex128",
    "torch.float8_e4m3fn": "jnp.float8_e4m3fn",
    "torch.float8_e4m3fnuz": "jnp.float8_e4m3fnuz",
    "torch.float8_e5m2": "jnp.float8_e5m2",
    "torch.float8_e5m2fnuz": "jnp.float8_e5m2fnuz",
    "torch.float8_e8m0fnu": "jnp.float8_e8m0fnu",
}


def _automatic_collective_id(fn: Callable[..., object]) -> int:
    """Return a stable best-effort barrier namespace for a Helion kernel.

    The ID must agree across independent TorchTPU host processes, so Python's
    randomized hash and process-local allocation order are unsuitable. Source
    text distinguishes changed or separately defined kernels while the
    qualified name provides a stable fallback when source is unavailable.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = ""
    identity = f"{fn.__module__}\0{fn.__qualname__}\0{source}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") & 0x7FFFFFFF


class SliceAddressing(enum.Enum):
    """How a dynamic-offset slice on a tensor dim must be emitted on TPU."""

    DIRECT = enum.auto()  # offset used as-is -> plain pl.ds
    ALIGNED = enum.auto()  # offset rounded to a sublane tile -> aligned-enclosing


def _slice_addressing(
    tensor: torch.Tensor, dim: int, lane_block: int | None = None
) -> SliceAddressing:
    """Whether a dynamic slice on ``dim`` can take any offset.

    TPU only tiles the last two dims into (8, 128) blocks, so a slice on an
    earlier row-major dim reads any offset (DIRECT).  A sublane-dim slice must
    align to a tile boundary (ALIGNED), except f32 over a single lane tile
    (``lane_block`` <= 128) stays contiguous and reads any offset too (DIRECT).
    ``lane_block`` is the last-dim extent (block size, or full width if untiled);
    None stays conservative (ALIGNED).
    """
    if dim < tensor.ndim - 2:
        return SliceAddressing.DIRECT  # major dim: row-major, any offset
    if dim == tensor.ndim - 2:  # 2nd-minor (sublane) dim
        # f32 fills a lane, so a single lane tile is contiguous and reads any
        # offset; bf16 packs two rows per sublane and always needs alignment.
        if (
            tensor.dtype == torch.float32
            and isinstance(lane_block, int)
            and lane_block <= 128
        ):
            return SliceAddressing.DIRECT
        return SliceAddressing.ALIGNED
    return SliceAddressing.ALIGNED  # TODO(tcombes): align lane dim to 128, not sublane


class PallasBackend(Backend):
    """Pallas (JAX) code generation backend for TPU."""

    @property
    def name(self) -> str:
        return "pallas"

    @staticmethod
    # Overrides Backend.map_dot_precision.
    def map_dot_precision(precision: DotPrecision) -> str:
        """Map Helion dot precision to Pallas-specific precision string.

        Pallas/TPU does not support Triton-style TF32/IEEE controls. On the
        current TPU stack, JAX ``high``/``highest`` fp32 dot precision is less
        compatible with PyTorch eager references than JAX default precision, so
        all Helion aliases intentionally lower to the Pallas default.
        """
        return "default"

    @property
    def max_tensor_numel(self) -> int | None:
        # No compile-time element cap on Pallas; VMEM byte budget is the
        # real constraint and is enforced separately at runtime.
        return None

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        return False

    @property
    def requires_shape_specialized_module(self) -> bool:
        # Pallas modules hold monomorphic, shape-specific cached state
        # (output-meta descriptor, launcher cache, ds-pad decision, signature
        # lock), so each static-shape specialization needs its own module.
        return True

    def max_reduction_threads(self) -> int | None:
        return None

    def dtype_str(self, dtype: torch.dtype) -> str:
        key = str(dtype)
        if key not in _TORCH_TO_JAX_DTYPE:
            raise ValueError(f"Unsupported dtype for Pallas backend: {dtype}")
        return _TORCH_TO_JAX_DTYPE[key]

    def acc_type(self, dtype: torch.dtype) -> str:
        # Promote half-precision types to float32 for numerical stability
        if dtype in (torch.float16, torch.bfloat16):
            return "jnp.float32"
        return self.dtype_str(dtype)

    @property
    def function_decorator(self) -> str:
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_pallas_launcher"

    @property
    def dependency_free_launcher_info(self) -> LauncherInfo:
        # Pallas generated code makes no ``helion.runtime.<fn>`` helper calls
        # beyond the launcher, so the shim need only re-export the launcher itself.
        return LauncherInfo(
            launcher_module="helion.runtime.pallas.launcher",
            launcher_symbol="default_pallas_launcher",
            launcher_alias="_default_pallas_launcher",
            deps="torch + jax",
            runtime_helper_names=(),
        )

    def capture_jax_launch_metadata(
        self, bound: BoundKernel[Any], config: Config | dict[str, object]
    ) -> JaxLaunchMeta:
        """Capture jax_fn launch metadata via a real-tensor run + two-probe (see
        :func:`capture_jax_launch_metadata`). Must run outside the fake-tensor env."""
        return capture_jax_launch_metadata(bound, config)

    def build_jax_fn_code(
        self,
        body_root: ast.Module,
        import_lines: list[str],
        meta: object,
        *,
        allow_helion_deps: bool,
    ) -> ast.Module:
        """Rewrite the generated module AST into the jax-native standalone (see
        :func:`build_jax_fn_ast`). ``meta`` is a :class:`JaxLaunchMeta` from
        :meth:`capture_jax_launch_metadata`; ``allow_helion_deps`` toggles whether the
        launch core is inlined (helion-free) or imported from helion."""
        return build_jax_fn_ast(
            body_root,
            import_lines,
            cast("JaxLaunchMeta", meta),
            inline_launcher=not allow_helion_deps,
        )

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "jax": "import jax",
            "jnp": "import jax.numpy as jnp",
            "pl": "from jax.experimental import pallas as pl",
            "lax": "import jax.lax as lax",
            "pltpu": "from jax.experimental.pallas import tpu as pltpu",
            "_default_pallas_launcher": "from helion.runtime import default_pallas_launcher as _default_pallas_launcher",
            # In-kernel helpers the generated code calls. Regular output imports
            # them from helion (conditionally, only when referenced); the
            # dependency-free path drops these imports and embeds the source instead
            # (see ``embedded_helper_source`` / ``build_dependency_free_code``).
            "_helion_divide_filter_topk": "from helion._compiler.pallas.topk_impl import divide_filter_topk as _helion_divide_filter_topk",
            "flatten_worklist": "from helion.runtime.compact_worklist import flatten_worklist",
        }

    def embedded_helper_source(self, body: str) -> str:
        """Inline the in-kernel Pallas helpers referenced by ``body``.

        ``divide_filter_topk`` (aten.topk lowering) and ``flatten_worklist``
        (compact-worklist builder) are pure-``jax`` helpers the generated kernel
        calls. Regular output imports them from helion (see ``library_imports``);
        this embeds their source instead, so a dependency-free / jax standalone is
        self-contained. Called only by the standalone builders (never for regular
        ``to_code``), which drop the corresponding helion imports.
        """
        return _embedded_helper_source(body)

    # Config keys that Pallas actually uses.  Everything else
    # (pid_type, num_warps, num_stages, maxnreg, indexing, etc.)
    # is GPU-specific and should not be tuned.
    _PALLAS_SUPPORTED_KEYS: frozenset[str] = frozenset(
        {
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "pallas_worklist_grouping",
            "pallas_loop_type",
            "pallas_collective_id",
            "pallas_load_buffer_count",
            "pallas_pre_broadcast",
            "pallas_topk_recall_target",
        }
    )

    def supports_config_key(self, key: str) -> bool:
        return key in self._PALLAS_SUPPORTED_KEYS

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"pl.program_id({dim})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return f"lax.convert_element_type({expr_str}, {dtype_str})"

    @property
    def range_requires_python_int(self) -> bool:
        return True

    def range_str(
        self,
        begin: str | None,
        end: str,
        step: str | None,
    ) -> str | None:
        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)
        return f"range({', '.join(range_args)})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"{offsets_var} = {lid} * {block_size_var} + jnp.arange(0, {block_size_var}, dtype={dtype})"

    def sympy_printer_expr(self, expr: sympy.Expr) -> str:
        from .printer import pallas_texpr

        return pallas_texpr(expr)

    def inductor_op_overrides(self) -> InductorOpOverrides:
        from torch._inductor.codegen.pallas import PallasKernelOverrides

        return PallasKernelOverrides()

    def cast_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        return expr_from_string(
            f"lax.convert_element_type({{x}}, {self.dtype_str(target_dtype)})", x=x
        )

    def transform_host_arg(
        self,
        arg: Argument,
        host_str: str,
        tensor_host_args: list[str],
    ) -> str:
        from ..device_function import SymbolArgument
        from ..device_function import TensorSizeArg
        from ..device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            from ..compile_environment import CompileEnvironment

            if tensor_host_args:
                device_expr = f"{tensor_host_args[0]}.device"
            elif CompileEnvironment.current().settings.pallas_interpret:
                device_expr = "'cpu'"
            else:
                device_expr = "'tpu'"
            # Scalars are passed as 1-dim tensors (shape [1]) rather than
            # 0-dim tensors (shape []) because TPU Pallas Mosaic lowering
            # requires rank >= 1 for all block specs.  A 0-dim input causes:
            #   ValueError: The Pallas TPU lowering currently supports only
            #   blocks of rank >= 1.
            # The kernel dereferences the scalar with ``name[0]`` (see
            # ``scalar_arg_preamble``).
            if isinstance(arg, (TensorSizeArg, TensorStrideArg)):
                from ..compile_environment import CompileEnvironment

                idx_dtype = CompileEnvironment.current().index_dtype
                return f"torch.tensor([{host_str}], dtype={idx_dtype!r}, device={device_expr})"
            return f"torch.tensor([{host_str}], dtype=torch.float32 if isinstance({host_str}, float) else torch.int32, device={device_expr})"
        return host_str

    def scalar_arg_preamble(self, arg: Argument) -> list[ast.AST]:
        from ..ast_extension import statement_from_string
        from ..device_function import SymbolArgument
        from ..device_function import TensorSizeArg
        from ..device_function import TensorStrideArg

        if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
            # TPU: scalars are wrapped as 1-dim tensors, index with [0]
            return [statement_from_string(f"{arg.name} = {arg.name}[0]")]
        return []

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"{offset_var} + jnp.arange(0, ({block_size_var}), dtype={dtype})"

    def scalar_load_expr(self, tensor_name: str, index_expr: str | None = None) -> str:
        if index_expr is None:
            index_expr = "0"
        return f"({tensor_name})[{index_expr}]"

    def full_expr(
        self, shape_dims: list[str], value_expr: str, dtype: torch.dtype
    ) -> str:
        return f"jnp.full([{', '.join(shape_dims)}], {value_expr}, {self.dtype_str(dtype)})"

    def reshape_expr(self, expr: str, shape: str) -> str:
        return f"jnp.reshape({expr}, {shape})"

    def broadcast_to_expr(self, expr: str, shape: str) -> str:
        return f"jnp.broadcast_to({expr}, {shape})"

    def reduction_expr(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        *,
        block_size_var: str | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        if reduction_type in {"sum", "max", "min", "prod"}:
            return f"jnp.{reduction_type}({input_name}, axis={dim})"
        raise exc.BackendUnsupported(self.name, f"reduction {reduction_type!r}")

    def is_indexed_reduction(self, reduction_type: str) -> bool:
        return reduction_type in {"argmin", "argmax"}

    def argreduce_result_expr(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        output_dtype: torch.dtype,
        *,
        block_size_var: str | None = None,
        index_dtype: torch.dtype | None = None,
        threads_in_group: int | None = None,
    ) -> str:
        fn = "jnp.argmax" if reduction_type == "argmax" else "jnp.argmin"
        return (
            f"lax.convert_element_type("
            f"{fn}({input_name}, axis={dim}), {self.dtype_str(output_dtype)})"
        )

    def argreduce_loop_update_statements(
        self,
        *,
        reduction_type: str,
        acc: str,
        acc_index: str,
        value: str,
        index: str,
    ) -> list[str]:
        if reduction_type == "argmin":
            better = (
                f"(({value}) < ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        else:
            better = (
                f"(({value}) > ({acc})) | "
                f"((({value}) == ({acc})) & (({index}) < ({acc_index})))"
            )
        return [
            f"{acc} = jnp.where({better}, {value}, {acc})",
            f"{acc_index} = jnp.where({better}, {index}, {acc_index})",
        ]

    def where_expr(self, mask: str, true_val: str, false_val: str) -> str:
        return f"jnp.where({mask}, {true_val}, {false_val})"

    def minimum_expr(self, a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    def arange_index_expr(self, block_size_var: str, dtype: str) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def zeros_expr(self, shape: str, dtype: str) -> str:
        return f"jnp.zeros({shape}, dtype={dtype})"

    def reduction_index_expr(
        self, block_size_var: str, dtype: str, block_idx: int, *, axis: int
    ) -> str:
        return f"jnp.arange(0, {block_size_var}, dtype={dtype})"

    def reduction_index_zero_expr(self, dtype: str) -> str:
        return f"jnp.zeros([0], dtype={dtype})"

    def static_rdim_size(self, numel: int) -> int:
        # Pallas block refs use exact tensor dimensions, so RDIM_SIZE must
        # match (no power-of-2 rounding that would exceed the block ref).
        return numel

    def dynamic_rdim_size_expr(self, expr: str) -> str:
        return expr

    def _get_pallas_required_alignment(
        self, dim_from_end: int, tensor_ndim: int, bitwidth: int
    ) -> int:
        """Requirements documented in https://docs.jax.dev/en/latest/pallas/grid_blockspec.html

        Args:
            dim_from_end (int): The dimension being queried for alignment requirements, indexed from the end. i.e. [... ,2, 1, 0]
            tensor_ndim (int): Amount of dimensions for the tensor.
            bitwidth (int): Bitwidth of tensor elements
        """
        # Cap to 32: wider dtypes (e.g. float64, int64) would cause
        # ZeroDivisionError in 32 // bitwidth.  64-bit types are rejected
        # at runtime, so block spec computation uses 32-bit alignment.
        bitwidth = min(bitwidth, 32)
        if dim_from_end == 0:  # Last dimension
            if tensor_ndim <= 1:
                return 128 * (32 // bitwidth)
            return 128
        if dim_from_end == 1:  # Second to last dimension
            return 8
        return 1  # No requirements for other dimensions

    def sublane_tiling(self, dtype: torch.dtype) -> int:
        """Native sublane (2nd-minor) tile for ``dtype``: f32->8, bf16->16, i8->32.

        The jagged carry slices its emit_pipeline VMEM refs at this
        granularity, and such a ref must be accessed as a *whole* native tile:
        a smaller slice (e.g. 8 rows of a bf16 ref, whose tile is 16) is
        rejected by Mosaic ("E2003: unproven memory access alignment"),
        independent of offset.
        """
        bitwidth = min(dtype.itemsize * 8, 32)
        return 8 * (32 // bitwidth)

    fake_tensor_loads: list[tuple[torch.Tensor, list[object]]]

    def process_fake_tensor_load(
        self,
        tensor: torch.Tensor,
        index: list[object],
    ) -> None:
        if not hasattr(self, "fake_tensor_loads"):
            self.fake_tensor_loads = []
        self.fake_tensor_loads.append((tensor, index))

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        """Enforce TPU alignment on block sizes.

        TPU Pallas requires:
        - 1D last dim: multiple of ``128 * (32 // dtype_bits)``
          (128 for f32, 256 for bf16)
        - 2D+ last dim: multiple of 128
        - 2D+ second-to-last dim: multiple of 8

        When the tensor dimension is smaller than the alignment requirement,
        we set the minimum block size to ``next_power_of_2(tensor_dim)``
        instead.  At runtime the block shape is capped to
        ``min(block_size, tensor_dim)`` which equals the full array
        dimension -- always valid per TPU rules.
        """
        from ...autotuner.config_spec import BlockSizeSpec
        from ..ast_extension import ExtendedAST
        from ..compile_environment import BlockSizeInfo
        from helion._compiler.compile_environment import _to_sympy
        from helion._compiler.host_function import HostFunction
        from helion._compiler.type_info import SequenceType
        from helion._compiler.type_info import TensorType
        from helion._compiler.type_info import TileIndexType

        host_func = HostFunction.current()

        class TensorTiledAccessAnalyzer(ast.NodeVisitor):
            def __init__(self, backend: PallasBackend) -> None:
                super().__init__()
                self.backend = backend
                self.required_alignments: dict[int, int] = {}
                self.update_requirements_from_fake_tensor_loads()

            def visit_Subscript(self, node: ast.Subscript) -> None:
                assert isinstance(node, ExtendedAST)
                assert isinstance(node.value, ExtendedAST)
                value_type = node.value._type_info
                if not isinstance(value_type, TensorType):
                    return
                tensor = value_type.fake_value
                if isinstance(node.slice, (ast.Tuple, ast.List)):
                    num_squeezed_dimensions = 0
                    for i, subscript in enumerate(node.slice.elts):
                        if (
                            isinstance(subscript, ast.Constant)
                            and subscript.value is None
                        ):
                            num_squeezed_dimensions += 1
                            continue
                        accessed_dim = i - num_squeezed_dimensions
                        self.maybe_update_alignment_requirement(
                            tensor, accessed_dim, subscript
                        )
                else:
                    self.maybe_update_alignment_requirement(tensor, 0, node.slice)
                # Nested subscripts (e.g. idx[tile] in table[idx[tile], :])
                # are themselves tiled accesses and need their own alignment.
                self.generic_visit(node)

            def maybe_update_alignment_requirement(
                self, tensor: torch.Tensor, accessed_dim_start: int, subscript: ast.AST
            ) -> None:
                if not isinstance(subscript, ExtendedAST):
                    return
                subscript_type = subscript._type_info
                tile_index_types: list[TileIndexType] = []
                if isinstance(subscript_type, TileIndexType):
                    tile_index_types.append(subscript_type)
                elif isinstance(subscript_type, SequenceType):
                    for el_type in subscript_type.element_types:
                        if isinstance(el_type, TileIndexType):
                            tile_index_types.append(el_type)

                for i, tile_index_type in enumerate(tile_index_types):
                    bid = tile_index_type.block_id
                    accessed_dim = accessed_dim_start + i
                    dim_from_end = tensor.ndim - accessed_dim - 1
                    bitwidth = tensor.dtype.itemsize * 8

                    required_alignment = self.backend._get_pallas_required_alignment(
                        dim_from_end, tensor.ndim, bitwidth
                    )
                    self.maybe_update_required_alignment(bid, required_alignment)

            def maybe_update_required_alignment(
                self, bid: int, required_alignment: int
            ) -> None:
                if bid not in self.required_alignments:
                    self.required_alignments[bid] = required_alignment
                else:
                    self.required_alignments[bid] = max(
                        self.required_alignments[bid], required_alignment
                    )

            def update_requirements_from_fake_tensor_loads(self) -> None:
                # When tensors are indexed within external lambdas called by the kernel,
                # they generate fake loads, which we don't pickup during AST walk.
                if not hasattr(self.backend, "fake_tensor_loads"):
                    return
                if block_sizes is None:
                    return
                for info in block_sizes:
                    if not isinstance(info, BlockSizeInfo):
                        continue
                    for tensor, subscripts in self.backend.fake_tensor_loads:
                        for dim, subscript in enumerate(subscripts):
                            if isinstance(subscript, torch.SymInt) and info.dim_matches(
                                _to_sympy(subscript)
                            ):
                                dim_from_end = tensor.ndim - 1 - dim
                                bitwidth = tensor.dtype.itemsize * 8
                                required_alignment = (
                                    self.backend._get_pallas_required_alignment(
                                        dim_from_end, tensor.ndim, bitwidth
                                    )
                                )
                                self.maybe_update_required_alignment(
                                    info.block_id, required_alignment
                                )

        analyzer = TensorTiledAccessAnalyzer(self)
        for stmt in host_func.body:
            analyzer.visit(stmt)

        from torch._inductor.runtime.runtime_utils import next_power_of_2

        if block_sizes is not None and kernel_tensor_sizes is not None:
            for shape in kernel_tensor_sizes:
                for bid, info in enumerate(block_sizes):
                    if not isinstance(info, BlockSizeInfo):
                        continue
                    # pyrefly: ignore[no-matching-overload]
                    if math.prod(shape) == info.var:
                        # avoid creating size-1 kernel tensors, which triggers Pallas Mosaic lowering failure:
                        # https://github.com/jax-ml/jax/issues/36970
                        analyzer.maybe_update_required_alignment(bid, 2)

        for spec in block_specs:
            if not isinstance(spec, BlockSizeSpec):
                continue
            bid = spec.block_ids[0]
            if bid not in analyzer.required_alignments:
                continue
            requirement_alignment = analyzer.required_alignments[bid]
            dim_size = next_power_of_2(max(spec.size_hint, 1))
            # Cap the alignment requirement by the tensor lane dim: when
            # the dim is smaller than the requirement, the full-dim access
            # is always aligned at offset 0 so block_size = dim_size is
            # safe.  When the dim is at least as big as the requirement,
            # ``min`` returns ``requirement_alignment`` and the strict
            # floor still applies (used by aot_example.sum_aot, n=256).
            spec.update_min(min(requirement_alignment, dim_size))

        # Propagate alignment minimums from inner tiles to their bounding outer tiles.
        block_specs_by_id = {
            spec.block_ids[0]: spec
            for spec in block_specs
            if isinstance(spec, BlockSizeSpec)
        }
        for spec in block_specs_by_id.values():
            bounded_by = spec.bounded_by_block_id
            if bounded_by is None:
                continue
            outer_spec = block_specs_by_id.get(bounded_by)
            if outer_spec is not None:
                outer_spec.update_min(spec.min_size)

    def tunable_fragments(self) -> dict[str, ConfigSpecFragment]:
        return {}

    def get_do_bench(self) -> Callable[..., float | tuple[float, ...]]:
        from ...autotuner.benchmarking import do_bench_generic

        return do_bench_generic

    def get_interleaved_bench(self) -> Callable[..., list[float]]:
        from ...autotuner.benchmarking import interleaved_bench_generic

        return interleaved_bench_generic

    def get_paired_device_micros_bench(
        self,
    ) -> Callable[..., list[tuple[float, float]]] | None:
        """Pallas ``jax.profiler`` device-µs bench for the final-pick re-rank.

        Returns None (keeping the wall-clock rebench) when the user opts out via
        ``HELION_AUTOTUNE_PALLAS_RANK_BY=wall_time`` or ``jax`` is unavailable.
        """
        from ...autotuner.benchmarking import make_pallas_paired_device_micros_bench

        return make_pallas_paired_device_micros_bench()

    def supports_precompile(self) -> bool:
        return False

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        # Pallas/JAX compilation and runtime errors are generally expected
        # during autotuning when invalid configs are tried.
        # Only truly fatal errors (KeyboardInterrupt, SystemExit, etc.)
        # should propagate; everything else is a config incompatibility.
        if isinstance(err, Exception):
            return "debug"
        return None

    def rng_seed_buffer_expr(self, count: int) -> str:
        # Generate on CPU, then move to the accelerator so the full 64-bit
        # Philox seed survives backend handoff.
        return f"inductor_prims.seeds({count}, torch.device('cpu')).to(torch.accelerator.current_accelerator())"

    def _compute_block_spec_info(
        self,
        sorted_args: list[Argument] | None,
        config: Config,
    ) -> (
        list[
            tuple[
                tuple[int | None, ...],
                tuple[int | tuple[int, int, int] | None, ...],
            ]
            | None
        ]
        | None
    ):
        """Compute per-tensor ``(block_shape, grid_dims)`` from codegen tiling info.

        Uses ``DeviceFunction.pallas_tensor_dim_tilings`` (recorded during
        ``plan_tiling`` from SymInt subscripts) for an unambiguous
        dim → block_id mapping.
        """
        if sorted_args is None:
            return None

        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..device_function import SymbolArgument
        from ..device_function import TensorArg
        from ..device_function import TensorSizeArg
        from ..device_function import TensorStrideArg
        from ..host_function import HostFunction
        from ..program_id import FlatProgramIDs

        env = CompileEnvironment.current()
        device_fn = DeviceFunction.current()

        # Build block_id → grid_dim from the actual PID ordering (which
        # reflects loop_order).  ``pid_info`` is ordered by grid dimension,
        # so pid_info[g].block_id is the block_id assigned to grid dim g.
        if device_fn.pid is None:
            return None
        flat_grid_block_ids = [pid.block_id for pid in device_fn.pid.pid_info]
        block_id_to_grid_dim = {bid: g for g, bid in enumerate(flat_grid_block_ids)}
        known_block_ids = set(block_id_to_grid_dim)

        # FlattenedTileStrategy collapses all block_ids into a single
        # pid_info entry, but the full set lives in device_ir.grid_block_ids.
        # Recover them so we can build flat decomposition and so downstream
        # checks (e.g. 1D tensor validation) see every block_id.
        flat_decomp: dict[int, tuple[int, int, int]] | None = None
        if isinstance(device_fn.pid, FlatProgramIDs):
            device_ir = HostFunction.current().device_ir
            all_grid_block_ids = [
                bid for bids in device_ir.grid_block_ids for bid in bids
            ]
            known_block_ids.update(all_grid_block_ids)

            if len(all_grid_block_ids) > 1:
                import sympy

                stride = 1
                flat_decomp = {}
                for bid in all_grid_block_ids:
                    bs = env.block_sizes[bid].from_config(config)
                    numel = env.block_sizes[bid].numel
                    if not isinstance(bs, int) or isinstance(numel, str):
                        return None
                    try:
                        numel_val = (
                            int(numel) if isinstance(numel, sympy.Expr) else numel
                        )
                    except (TypeError, ValueError):
                        return None
                    num_blocks = -(-numel_val // bs)  # cdiv
                    flat_decomp[bid] = (0, stride, num_blocks)
                    stride *= num_blocks

        result: list[
            tuple[tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]]
            | None
        ] = []

        for arg in sorted_args:
            if isinstance(arg, (SymbolArgument, TensorSizeArg, TensorStrideArg)):
                result.append(None)  # scalars wrapped as 1-D tensors
                continue
            if not isinstance(arg, TensorArg):
                continue
            if arg.fake_value.ndim == 0:
                result.append(None)
                continue
            tensor = arg.fake_value
            dim_tilings = device_fn.pallas_tensor_dim_tilings.get(id(tensor))
            if dim_tilings is None:
                # this means this tensor isn't accessed at all in the kernel
                result.append(None)
                return None
            block_shape: list[int | None] = []
            grid_dims: list[int | tuple[int, int, int] | None] = []
            for d in range(tensor.ndim):
                dim_tiling = dim_tilings[d]
                if not dim_tiling.can_tile or len(dim_tiling.block_ids) == 0:
                    block_shape.append(None)
                    grid_dims.append(None)
                    continue
                assert len(dim_tiling.block_ids) == 1
                bid = dim_tiling.block_ids[0]
                if bid is not None and bid in known_block_ids:
                    bs = env.block_sizes[bid].from_config(config)
                    if isinstance(bs, int):
                        block_shape.append(bs)
                        dim_size = tensor.shape[d]
                        # When the block covers the entire tensor
                        # dimension there is only one tile, so the grid
                        # index must be constant 0 — iterating would
                        # read out-of-bounds (e.g. bias [1, N] with
                        # block_size > 1).
                        if isinstance(dim_size, int) and dim_size <= bs:
                            grid_dims.append(None)
                        elif flat_decomp is not None and bid in flat_decomp:
                            grid_dims.append(flat_decomp[bid])
                        else:
                            grid_dims.append(block_id_to_grid_dim[bid])
                        continue
                block_shape.append(None)
                grid_dims.append(None)
            result.append((tuple(block_shape), tuple(grid_dims)))
        return result

    def _compute_pad_info(
        self,
        sorted_args: list[Argument] | None,
        config: Config,
    ) -> list[tuple[int, int, int, int]] | None:
        """Identify pl.ds() dims that may need padding and their block sizes.

        Uses ``pallas_pad_info`` recorded during codegen to identify which
        tensor dimensions use ``pl.ds()`` slicing, plus the one dummy row an
        empty resident operand needs (see :meth:`_zero_row_resident_pad_info`).

        Returns ``[(arg_index, tensor_dim, block_size, extra_pad), ...]``
        or ``None``.  The launcher computes the actual pad amount at runtime
        as ``(-tensor.shape[dim]) % block_size + extra_pad``.

        ``extra_pad`` is 0 when the tile loop starts at offset 0,
        ``begin % block_size`` for a constant begin offset, or
        ``block_size - 1`` for a data-dependent begin.
        """
        if sorted_args is None:
            return None

        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..device_function import TensorArg

        env = CompileEnvironment.current()
        device_fn = DeviceFunction.current()

        result: list[tuple[int, int, int, int]] = []
        if device_fn.pallas_pad_info:
            for i, arg in enumerate(sorted_args):
                if not isinstance(arg, TensorArg):
                    continue
                dims_info = device_fn.pallas_pad_info.get(id(arg.fake_value))
                if dims_info is not None:
                    for dim, (block_id, extra_pad) in dims_info.items():
                        bsi = env.block_sizes[block_id]
                        bs = bsi.from_config(config)
                        if isinstance(bs, int) and bs > 1:
                            result.append((i, dim, bs, extra_pad))

        result.extend(self._zero_row_resident_pad_info(sorted_args))
        return result or None

    def _zero_row_resident_pad_info(
        self, sorted_args: list[Argument]
    ) -> list[tuple[int, int, int, int]]:
        """One dummy row for each resident operand that has NO rows.

        A resident window is opened on the operand whether or not any ordered
        range is non-empty, so an operand with zero rows leaves the window with
        no in-bounds row to slice.  Giving it a single row makes the window's
        clamped slice ``pl.ds(0, 1)`` -- a legal, one-row transfer.  The row is
        never read: an operand is only empty when every ordered range is empty,
        so the reduction is zero-trip.

        Emitting the pad rather than declining to cache keeps the resident
        decision ACTIVE, which matters because ``pallas_loop_type='unroll'``
        rejects an inactive decision outright and a legal all-empty ordered
        reduction would otherwise stop compiling.

        ``block_size = 1`` makes ``(-rows) % block_size`` vanish, so the pad is
        exactly ``extra_pad`` (one row).  That formula pads unconditionally, so
        this must only ever fire for a CONCRETELY empty operand -- otherwise a
        real tensor would take a full copy.  Every non-degenerate kernel gets an
        empty list from this helper (ordinary ``pallas_pad_info`` entries are
        unaffected and can still request padding of their own).
        """
        from ..compile_environment import CompileEnvironment
        from ..device_function import TensorArg

        decision = CompileEnvironment.current().compact_worklist_resident_cache_decision
        if decision is None or not decision.active:
            return []
        resident = set(decision.resident_operands)
        return [
            (i, 0, 1, 1)
            for i, arg in enumerate(sorted_args)
            if isinstance(arg, TensorArg)
            and arg.host_str() in resident
            and int(arg.fake_value.size(0)) == 0
        ]

    def _detect_matmul_dot_general_lowering(
        self,
        *,
        sorted_args: list[Argument] | None,
        config: Config,
        output_indices: list[int],
        inplace_indices: list[int],
        block_spec_info: object,
    ) -> dict[str, object] | None:
        """Detect a pure-matmul, no-tiling kernel the launcher can lower as
        ``jax.jit(lax.dot_general(...))`` instead of ``pl.pallas_call(...)``.

        Eligible when: 2 input tensors + 1 output-only tensor; all 2D with
        matching M/K/N contiguous layout (BMM not covered yet); the device IR
        has one ``aten.mm``/``addmm`` family op; and the picked block sizes
        cover every dim (single launch, no inner K tile).  Returns the spec
        dict consumed by ``_build_matmul_dot_general_jit_fn``, else ``None``.
        """
        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..device_function import TensorArg
        from ..host_function import HostFunction

        if sorted_args is None or not output_indices:
            return None
        # Pure-output kernels only (no in-place mutation, single output).
        if inplace_indices or len(output_indices) != 1:
            return None

        # Exactly 2 inputs + 1 output, all tensors (a scalar arg means it isn't
        # a pure ``out = matmul(x, y)``).
        tensor_positions = [
            i for i, arg in enumerate(sorted_args) if isinstance(arg, TensorArg)
        ]
        if len(sorted_args) != 3 or len(tensor_positions) != 3:
            return None

        out_pos = output_indices[0]
        input_positions = [p for p in tensor_positions if p != out_pos]
        if len(input_positions) != 2:
            return None

        lhs_arg = sorted_args[input_positions[0]]
        rhs_arg = sorted_args[input_positions[1]]
        out_arg = sorted_args[out_pos]
        assert isinstance(lhs_arg, TensorArg)
        assert isinstance(rhs_arg, TensorArg)
        assert isinstance(out_arg, TensorArg)
        lhs_t = lhs_arg.fake_value
        rhs_t = rhs_arg.fake_value
        out_t = out_arg.fake_value
        # 2D matmul, matching contraction dim, statically-known shapes.
        if lhs_t.ndim != 2 or rhs_t.ndim != 2 or out_t.ndim != 2:
            return None
        try:
            m = int(lhs_t.shape[0])
            k_lhs = int(lhs_t.shape[1])
            k_rhs = int(rhs_t.shape[0])
            n = int(rhs_t.shape[1])
            out_m = int(out_t.shape[0])
            out_n = int(out_t.shape[1])
        except (TypeError, ValueError):
            return None
        if k_lhs != k_rhs or out_m != m or out_n != n:
            return None

        # The device IR must contain an aten.mm/addmm/bmm family op
        # (via the shared ``_loop_contains_matmul`` predicate).
        device_fn = DeviceFunction.current()
        device_ir = HostFunction.current().device_ir
        if not device_ir.grid_block_ids:
            return None
        # Any root-grid loop containing a matmul qualifies.
        matmul_present = any(
            _loop_contains_matmul(device_fn, list(grid_block_ids))
            for grid_block_ids in device_ir.grid_block_ids
        )
        if not matmul_present:
            return None

        # Orient to lhs=(M, K), rhs=(K, N); the user may have written
        # ``f(y, x) -> x @ y``. For all-equal dims either ordering is the same.
        if lhs_t.shape == (m, k_lhs) and rhs_t.shape == (k_lhs, n):
            lhs_arg_pos, rhs_arg_pos = input_positions
            lhs_resolved, rhs_resolved = lhs_t, rhs_t
        elif lhs_t.shape == (k_lhs, n) and rhs_t.shape == (m, k_lhs):
            rhs_arg_pos, lhs_arg_pos = input_positions
            lhs_resolved, rhs_resolved = rhs_t, lhs_t
        else:
            return None

        # Every block size must be >= max(M, N, K): a smaller block means a
        # multi-launch (tiled) kernel, not the no-tiling case.
        env = CompileEnvironment.current()
        max_dim = max(m, k_lhs, n)
        for bsi in env.block_sizes:
            if bsi is None:  # type: ignore[unreachable]
                continue
            try:
                bs = bsi.from_config(config)
            except Exception:
                return None
            if not isinstance(bs, int) or bs < max_dim:
                return None

        # Every tensor must be fully untiled (all grid_dims None); outer-grid
        # BlockSpecs still need pl.pallas_call.
        if block_spec_info is None or not isinstance(block_spec_info, list):
            return None
        for pos in (input_positions[0], input_positions[1], out_pos):
            if pos >= len(block_spec_info):
                return None
            entry = block_spec_info[pos]
            if entry is None:
                return None
            block_shape, grid_dims = entry
            if any(gd is not None for gd in grid_dims):
                return None

        # All checks passed; build the launcher spec. bf16/fp16 output from an
        # f32 accumulator needs preferred f32 + cast-back; f32 is already f32.
        f32_acc = out_t.dtype in (torch.bfloat16, torch.float16)
        # Map positions to the launcher's tensor-arg order (sorted non-output
        # positions; see ``_pallas_prepare_args``).
        non_output_positions = sorted(p for p in tensor_positions if p != out_pos)
        return {
            "lhs_tensor_arg_index": non_output_positions.index(lhs_arg_pos),
            "rhs_tensor_arg_index": non_output_positions.index(rhs_arg_pos),
            "lhs_dtype": self.dtype_str(lhs_resolved.dtype),
            "rhs_dtype": self.dtype_str(rhs_resolved.dtype),
            "out_dtype": self.dtype_str(out_t.dtype),
            "f32_accumulator": bool(f32_acc),
        }

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: list[Argument] | None = None,
    ) -> list[str]:
        # Determine which arg positions are outputs.  A tensor is an output if:
        #   1. It was created inside the function body (not in input_sources), OR
        #   2. It is a function parameter that is mutated in-place (e.g. x[tile] += ...)
        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..device_function import TensorArg
        from ..host_function import HostFunction

        device_fn = DeviceFunction.current()

        def _empty_allocated_vars(body: list[ast.stmt]) -> set[str]:
            """Return names of variables allocated with torch.empty/empty_like/new_empty.

            Only checks top-level assignments; allocations nested inside
            if/with/try are conservatively missed (treated as needing input,
            which is correct but suboptimal).
            """
            result: set[str] = set()
            for stmt in body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr in ("empty", "empty_like", "new_empty")
                ):
                    result.add(stmt.targets[0].id)
            return result

        output_indices: list[int] = []
        # Indices of output tensors that are also read by the kernel
        # (inplace-mutated params or body-created tensors the kernel reads).
        # These must use VMEM BlockSpecs. Output-only tensors (written but
        # never read) get HBM in_specs to avoid VMEM pressure.
        inplace_indices: list[int] = []
        if sorted_args is not None:
            env = CompileEnvironment.current()
            host_fn = HostFunction.current()
            read_names, write_names = device_fn.get_tensor_read_write_names()
            mutated_params = write_names & {a.arg for a in host_fn.args.args}
            input_storages = {id(t.untyped_storage()) for t in env.input_sources}
            # Only tensors allocated with torch.empty/empty_like/new_empty can be
            # output-only — their initial values are undefined, so it's safe
            # to use HBM BlockSpecs.  Tensors allocated with torch.zeros_like,
            # torch.full, etc. have meaningful initial values that must be
            # preserved via VMEM BlockSpecs.
            empty_vars = _empty_allocated_vars(host_fn.body)
            for i, arg in enumerate(sorted_args):
                if not isinstance(arg, TensorArg):
                    continue
                arg_name = arg.host_str()
                if (
                    id(arg.fake_value.untyped_storage()) not in input_storages
                    and arg_name in write_names
                ):
                    # Tensor created inside the function body (output)
                    output_indices.append(i)
                    if arg_name in read_names or arg_name not in empty_vars:
                        # Also read by the kernel (e.g. broadcast result)
                        inplace_indices.append(i)
                elif arg_name in mutated_params:
                    # Input tensor mutated in-place
                    output_indices.append(i)
                    inplace_indices.append(i)

        # Collect output-only tensor names so codegen can retarget their
        # allocations to ``device='meta'`` and capture the launcher return.
        output_only_set = set(output_indices) - set(inplace_indices)
        output_only_names: list[str] = []
        if sorted_args is not None:
            for i in output_indices:
                if i in output_only_set:
                    arg = sorted_args[i]
                    assert isinstance(arg, TensorArg)
                    output_only_names.append(arg.host_str())
        self._output_only_names = output_only_names

        launcher_args = [*args]
        if has_rng_ops:
            launcher_args.append("_rng_seed_buffer")
        launcher_args.extend(
            [f"_output_indices={output_indices}", f"_inplace_indices={inplace_indices}"]
        )

        block_spec_info = self._compute_block_spec_info(sorted_args, config)
        if block_spec_info is not None:
            if has_rng_ops:
                block_spec_info.append(None)  # RNG seed buffer is untiled
            launcher_args.append(f"_block_spec_info={block_spec_info!r}")

        pad_info = self._compute_pad_info(sorted_args, config)
        if pad_info:
            launcher_args.append(f"_ds_pad_dims={pad_info!r}")

        from ..device_function import PallasMemorySpace

        mem_space = device_fn.pallas_memory_space
        if sorted_args is not None:
            smem_arg_indices = [
                i
                for i, arg in enumerate(sorted_args)
                if isinstance(arg, TensorArg)
                and mem_space.get(id(arg.fake_value)) == PallasMemorySpace.SMEM
            ]
            if smem_arg_indices:
                launcher_args.append(f"_smem_arg_indices={smem_arg_indices!r}")

        # Pass scratch shapes for pipeline/fori_loop launcher
        scratch_shapes = [
            (
                s.shape,
                self.dtype_str(s.dtype) if s.dtype is not None else None,
                s.scratch_type,
            )
            for s in device_fn._scratch_args
        ]
        if scratch_shapes:
            from ..host_function import HostFunction

            scratch_shapes_str = HostFunction.current().literal_expr(scratch_shapes)
            launcher_args.append(f"_scratch_shapes={scratch_shapes_str}")

        if device_fn.requires_remote_copy:
            launcher_args.append("_uses_remote_copy=True")

        if device_fn.requires_collective_id:
            collective_id = config.get("pallas_collective_id")
            if collective_id is None:
                from ..host_function import HostFunction

                collective_id = _automatic_collective_id(HostFunction.current().fn)
            if not isinstance(collective_id, int):
                raise TypeError(
                    "pallas_collective_id must be an integer, got "
                    f"{type(collective_id)!r}"
                )
            launcher_args.append(f"_collective_id={collective_id!r}")

        # Identify which launcher arg positions correspond to pipeline-body
        # tensors (need HBM refs); all others get proper BlockSpecs.
        from ..device_function import TensorArg

        if sorted_args is not None:
            hbm_arg_indices = [
                i
                for i, arg in enumerate(sorted_args)
                if isinstance(arg, TensorArg)
                and mem_space.get(id(arg.fake_value)) == PallasMemorySpace.HBM
            ]
            if hbm_arg_indices:
                launcher_args.append(f"_hbm_arg_indices={hbm_arg_indices!r}")

        if CompileEnvironment.current().settings.pallas_interpret:
            launcher_args.append("_pallas_interpret=True")

        # No-tiling pure 2D matmul: emit ``_matmul_dot_general=...`` so the
        # launcher uses ``jax.jit(lax.dot_general(...))`` instead of
        # ``pl.pallas_call(...)``. XLA can then attach cross_program_prefetch,
        # closing the ~12% gap to ``jnp.matmul`` that ``tpu_custom_call``
        # opacity imposes. Falls back silently when ineligible.
        matmul_spec = self._detect_matmul_dot_general_lowering(
            sorted_args=sorted_args,
            config=config,
            output_indices=output_indices,
            inplace_indices=inplace_indices,
            block_spec_info=block_spec_info,
        )
        if matmul_spec is not None:
            launcher_args.append(f"_matmul_dot_general={matmul_spec!r}")

        if (
            CompileEnvironment.current().compact_worklist_plan is not None
            and sorted_args is not None
        ):
            launcher_args.extend(self._compact_worklist_launcher_args(sorted_args))

        return launcher_args

    def _compact_worklist_launcher_args(self, sorted_args: list[Argument]) -> list[str]:
        """Emit the compact-worklist-specific launcher kwargs.

        ``_build_worklist`` is the module-level jnp builder (emitted in
        generate_ast); the offset arg indices map its params to host-call arg
        positions; the metadata fields + owner-ref position drive scalar-prefetch
        selection and the owner-indexed BlockSpec index_maps.
        """
        from ..compile_environment import CompileEnvironment
        from ..device_function import TensorArg
        from .compact_worklist import metadata_field_names

        env = CompileEnvironment.current()
        plan = env.compact_worklist_plan
        assert plan is not None

        name_to_index: dict[str, int] = {}
        for i, arg in enumerate(sorted_args):
            if isinstance(arg, TensorArg):
                name_to_index[arg.host_str()] = i
        offset_indices = [name_to_index[n] for n in env.compact_worklist_offset_params]
        fields = metadata_field_names(plan)
        # Compact-tile tensors (aligned load + exact store) both get a max-sized
        # compact window sliced at tile_start, so Pallas double-buffers BOTH the
        # load prefetch and the store write-back across work items.
        #
        # The store is a masked full-block write.  The two robust EXACT-store
        # alternatives were both worse/unavailable here: (a) staging VMEM +
        # make_async_copy over pl.ds(tile_start, tile_extent) serializes (~1.8x
        # slower: 4.5ms vs 2.5ms) because a straight-line compact tile has no inner
        # loop to overlap; (b) a pl.BoundedSlice store BlockSpec was once rejected by
        # Mosaic ("Unsupported block dimension type: BoundedSlice"), but that is a
        # LOWERING-PATH difference, not a version one: pl.pallas_call and
        # emit_pipeline's compute_slice still reject it while the pl.kernel pipeline
        # this launcher uses lowers it on jax 0.10.1 and 0.11.0 alike, so the window
        # IS a BoundedSlice now (see _compact_window_block_spec).  The
        # full-block write's only hazard is a partial last tile overlapping the
        # next owner's leading rows; "arbitrary" dimension semantics serialize
        # that grid-ordered overlap so the later, correct write wins (verified
        # bitwise == fori_loop across uniform/partial/unaligned/jagged + 5 random
        # seeds).  Robust+fast exact store == the deferred emit_pipeline +
        # pl.BoundedSlice path.
        aligned_indices = [
            name_to_index[p.arg_name]
            for p in plan.tensor_policies
            if p.kind in ("compact_aligned_load", "compact_exact_store")
            and p.arg_name in name_to_index
        ]
        # The cached resident-cache decision drives every resident-window launcher
        # arg: resident-window tensors, the exact physical window integer, and the
        # ordered/compact offset args used by the overflow guard.
        decision = env.compact_worklist_resident_cache_decision
        ordered_indices: list[int] = []
        range_start_ref_pos = -1
        ordered_offset_arg_index = -1
        active_mask_arg_index = -1
        ordered_window = 0
        if decision is not None and decision.active:
            assert decision.range_spec is not None
            if decision.resident_key_fields != ("range_start",):
                raise exc.InvalidConfig(
                    "compact_worklist resident caching: Phase 1 resident windows "
                    "must be keyed by range_start."
                )
            range_start_ref_pos = (
                fields.index("range_start") if "range_start" in fields else -1
            )
            missing_residents = [
                name for name in decision.resident_operands if name not in name_to_index
            ]
            if missing_residents:
                raise exc.InvalidConfig(
                    "compact_worklist resident caching: active resident operands are "
                    f"missing from the kernel argument list: {missing_residents}."
                )
            ordered_indices = [
                name_to_index[name] for name in decision.resident_operands
            ]
            ordered_offset_arg_index = name_to_index.get(
                decision.range_spec.ordered_offset_arg, -1
            )
            active_mask_arg_index = name_to_index.get(
                decision.range_spec.compact_offset_arg, -1
            )
            ordered_window = decision.physical_window
            if (
                range_start_ref_pos < 0
                or ordered_offset_arg_index < 0
                or active_mask_arg_index < 0
            ):
                raise exc.InvalidConfig(
                    "compact_worklist resident caching: active range metadata or "
                    "offset args are missing from the kernel argument list."
                )
        return [
            "_compact_build_worklist=_build_worklist",
            f"_compact_offset_arg_indices={offset_indices!r}",
            f"_compact_metadata_fields={fields!r}",
            "_compact_owner_ref_pos=0",
            f"_compact_num_scalar_prefetch={len(fields)}",
            f"_compact_aligned_arg_indices={aligned_indices!r}",
            f"_compact_tile_start_ref_pos={fields.index('tile_starts')}",
            f"_compact_block={env.compact_worklist_block * plan.grouping}",
            f"_compact_ordered_aligned_arg_indices={ordered_indices!r}",
            f"_compact_range_start_ref_pos={range_start_ref_pos}",
            f"_compact_ordered_offset_arg_index={ordered_offset_arg_index}",
            f"_compact_active_mask_arg_index={active_mask_arg_index}",
            f"_compact_ordered_window={ordered_window}",
        ]

    def build_launcher_name(self, config: Config) -> str:
        """Return the single Pallas launcher name.

        All ``pallas_loop_type`` values route through the same
        ``default_pallas_launcher``; worklist flattening is selected separately
        by its launcher-observable kwargs.
        """
        from ...autotuner.config_spec import VALID_PALLAS_LOOP_TYPES

        pallas_loop_type = config.get("pallas_loop_type", "unroll")
        if pallas_loop_type not in VALID_PALLAS_LOOP_TYPES:
            raise ValueError(
                f"Invalid pallas_loop_type {pallas_loop_type!r}. "
                f"Expected one of {VALID_PALLAS_LOOP_TYPES}."
            )
        return self.default_launcher_name

    def get_launcher_name(self) -> str:
        """Return the launcher name based on the current config."""
        from ..device_function import DeviceFunction
        from ..device_function import NoCurrentFunction

        try:
            device_fn = DeviceFunction.current()
        except NoCurrentFunction:
            return self.default_launcher_name
        return self.build_launcher_name(device_fn.config)

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        from ...autotuner.config_spec import VALID_PALLAS_WORKLIST_GROUPINGS
        from ..compile_environment import CompileEnvironment
        from .plan_tiling import plan_tiling
        from .tensorcore_plan import build_tensorcore_plans

        # Validate pallas_loop_type before any codegen setup.
        self.build_launcher_name(config)
        grouping = config.get("pallas_worklist_grouping", 0)
        if type(grouping) is not int or grouping not in VALID_PALLAS_WORKLIST_GROUPINGS:
            raise exc.InvalidConfig(
                "Invalid pallas_worklist_grouping "
                f"{grouping!r}. Expected one of {VALID_PALLAS_WORKLIST_GROUPINGS}."
            )

        env = CompileEnvironment.current()

        plan_tiling(graphs, config, tile_strategy)
        build_tensorcore_plans(graphs, config)

        # compact_worklist_* is per-CONFIG state, but one CompileEnvironment is
        # reused across all configs of a BoundKernel (see CompileEnvironment's
        # "no config-specific state" contract). Reset before re-detecting so a
        # later non-flattened config never inherits a prior compact config's plan
        # -- many lowering paths gate on ``env.compact_worklist_plan is not None``
        # (PID strategy, loop-bound remap, fori handling, ds slicing), not on
        # config values, so a stale plan would mis-lower a later config.
        env.compact_worklist_plan = None
        env.compact_worklist_resident_cache_decision = None
        env.compact_worklist_resident_prep_hoists = ()
        env.compact_worklist_upper = 1
        env.compact_worklist_block = 1
        env.compact_worklist_ordered_block = 1
        env.compact_worklist_offset_params = []

        if grouping in (1, 2):
            self._setup_compact_worklist(graphs, config)

    def _setup_compact_worklist(self, graphs: list[GraphInfo], config: Config) -> None:
        """Detect + stash the compact-worklist plan before device codegen.

        Runs early (pre_codegen) so ``env.compact_worklist_plan`` is set when the
        grid strategy selects ``WorklistProgramIDs`` and the inner loop remaps its
        begin/end to metadata refs.  Registers the N metadata ref names as
        ``wrapper_only_params`` (kernel-signature-only) and computes the static
        megablocks ``UPPER``.

        ``detect_compact_worklist_plan`` raises ``exc.InvalidConfig`` for
        kernels whose source doesn't match the compact-worklist pattern
        (e.g. no owner ``hl.grid``, or an unsupported nest).  Since
        ``pallas_worklist_grouping`` is a config knob that can be searched
        independent of the kernel's structure, we catch that here and
        silently downgrade to no-op grouping rather than propagating the
        raise past autotune — otherwise a shape whose best-scoring config
        happens to enable grouping will fail the whole sweep step.
        """
        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..host_function import HostFunction
        from .compact_worklist import detect_compact_worklist_plan
        from .compact_worklist import metadata_arg_names

        env = CompileEnvironment.current()
        host_fn = HostFunction.current()
        try:
            detected = detect_compact_worklist_plan(host_fn)
        except exc.InvalidConfig:
            # Kernel doesn't support compact-worklist grouping — leave
            # env.compact_worklist_plan = None so downstream lowering paths
            # gated on `plan is not None` skip cleanly.
            return
        plan = dataclasses.replace(
            detected,
            grouping=cast("int", config.get("pallas_worklist_grouping", 1)),
        )
        env.compact_worklist_plan = plan

        device_fn = DeviceFunction.current()
        for name in metadata_arg_names(plan):
            ref = f"{name}_ref"
            if ref not in device_fn.wrapper_only_params:
                device_fn.wrapper_only_params.append(ref)

        # Compact-axis tile block size (NOT max(block_sizes): a distinct larger
        # ordered block would undersize the worklist metadata).
        compact_block = env.block_sizes[plan.compact_axis.block_id].from_config(config)
        assert compact_block is not None, "compact tile has no block size"
        env.compact_worklist_block = int(compact_block)
        # Ordered (reduction) tile block -- resident caching uses this compile-side to
        # choose a block-aligned physical window (it can differ from the compact
        # block, e.g. compact_block != ordered_block).
        env.compact_worklist_ordered_block = 1
        if plan.ordered_axis is not None:
            ordered_block = env.block_sizes[plan.ordered_axis.block_id].from_config(
                config
            )
            if ordered_block is not None:
                env.compact_worklist_ordered_block = int(ordered_block)
        env.compact_worklist_upper = self._compact_worklist_upper(plan, config, host_fn)

        if config.get("pallas_loop_type", "unroll") == "unroll" and (
            plan.ordered_axis is not None
        ):
            import jax.experimental.pallas.tpu as pltpu

            from ...runtime.pallas.launcher import _get_vmem_limit_bytes
            from .compact_worklist import build_resident_cache_admission

            # Choose C from the conservative device-reported VMEM budget. The
            # higher Mosaic compile ceiling used by the runtime is not an
            # allocation budget.
            admission = build_resident_cache_admission(
                graphs,
                plan,
                host_fn.params.arguments,
                ordered_block=env.compact_worklist_ordered_block,
                vmem_bytes=_get_vmem_limit_bytes(pltpu, env.settings.pallas_interpret),
            )
            if not admission.decision.active:
                raise exc.InvalidConfig(
                    "pallas_loop_type='unroll' requires resident ordered-loop "
                    f"admission: {admission.decision.inactive_reason}."
                )
            env.compact_worklist_resident_prep_hoists = admission.prep_hoists
            env.compact_worklist_resident_cache_decision = admission.decision

    def _compact_worklist_upper(
        self, plan: CompactWorklistPlan, config: Config, host_fn: HostFunction
    ) -> int:
        """Static UPPER: the padded length of the worklist metadata arrays.

        Must be >= the worst-case
        ``num_work = sum_owners cdiv(length, BLOCK * grouping)``,
        else the dynamic Pallas grid indexes past the scalar-prefetch metadata
        (``jnp.repeat(total_repeat_length=UPPER)`` would silently truncate the
        worklist).  Detection only accepts the packed-offsets idiom (store
        safety), so owner ranges are contiguous/non-overlapping. The compact
        input shape is at least their packed total (and may include padding), so
        ``cdiv(total, BLOCK * grouping) + num_owners - 1`` provably holds. All
        terms are concrete ints under ``static_shapes=True``.
        """
        from ...runtime.compact_worklist import packed_upper_bound
        from ..compile_environment import CompileEnvironment

        params = dict(host_fn.params.arguments)
        # Owner count from the captured grid bound (e.g. offsets.shape[0] - 1).
        # num_owners_expr is a codegen-derived host expression; if it references a
        # name not in params (a source shape we failed to inline), surface it as
        # an autotuner-skippable InvalidConfig rather than a bare exception that
        # would abort the whole search.
        try:
            num_owners = int(eval(plan.num_owners_expr, {}, params))
        except Exception as e:
            raise exc.InvalidConfig(
                f"compact_worklist: could not evaluate owner-count expression "
                f"{plan.num_owners_expr!r}: {e}"
            ) from e
        # total_compact = padded leading dim of the compact_aligned_load tensor.
        compact_arg = next(
            p.arg_name for p in plan.tensor_policies if p.kind == "compact_aligned_load"
        )
        total = int(params[compact_arg].shape[0])
        block = CompileEnvironment.current().compact_worklist_block * plan.grouping
        # Single source of the tight megablocks bound (also unit-tested).
        return packed_upper_bound(total, num_owners, block)


# Launcher kwargs that mark a kernel using a Pallas feature the pure-JAX
# module doesn't emit yet (compact-worklist, matmul dot_general).
_JAX_UNSUPPORTED_KWARGS = (
    "_compact_build_worklist",
    "_matmul_dot_general",
)

_JAX_REMOTE_COPY_KWARGS = (
    "_scratch_shapes",
    "_hbm_arg_indices",
    "_smem_arg_indices",
    "_ds_pad_dims",
)

# Dtypes the Pallas launcher rejects and that JAX would mishandle under x32
# (int64/uint64 silently narrow to 32-bit; float64 is unsupported on TPU).
_JAX_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


@dataclasses.dataclass
class JaxLaunchMeta:
    """Launch metadata captured by running the compiled host wrapper on real tensors
    (outside the fake-tensor env), consumed by the AST builder to emit the jax-native
    entrypoint. Grid / output-shape / scalar-arg values are Python-source expressions
    over ``inputs[i].shape[d]`` (derived by the two-probe) so one standalone is correct
    at every dynamic shape; static dims come through as literals."""

    kernel_name: str
    grid_exprs: list[str]
    output_indices: list[int]
    inplace_indices: list[int]
    user_positions: list[int]
    const_slots: dict[int, str]
    block_spec_info: list[Any]
    scratch_shapes: list[object]
    hbm_arg_indices: list[int]
    smem_arg_indices: list[int]
    ds_pad_dims: list[tuple[int, int, int, int]]
    out_shape_exprs: list[list[str]]
    out_dtypes: list[str]
    interpret: bool
    collective_id: int | None
    n_args: int


def _materialize_args(fake_args: list[object]) -> tuple[object, ...]:
    """Real, sample-shaped tensors reconstructed from a bound kernel's fake args.

    ``to_code`` has no access to the original inputs, but the jax_fn capture only
    needs tensors of the right shape/dtype/device (the capturing launcher records
    metadata without executing the kernel). ``int(sym)`` on a fake dim yields the
    bind-time sample size, so static and dynamic kernels both round-trip. Must be
    called outside the fake-tensor env so ``torch.empty`` allocates real tensors.
    """
    out: list[object] = []
    for fake in fake_args:
        if isinstance(fake, torch.Tensor):
            shape = [int(s) for s in fake.shape]
            out.append(torch.empty(shape, dtype=fake.dtype, device=fake.device))
        else:
            out.append(fake)
    return tuple(out)


def _torch_dtype_to_jnp_name(dtype: torch.dtype) -> str:
    """``torch.float32`` -> ``"jnp.float32"`` (``torch.bool`` -> ``"jnp.bool_"``)."""
    name = str(dtype).rsplit(".", 1)[-1]
    if name == "bool":
        name = "bool_"
    return f"jnp.{name}"


# Cap on inlining a host-wrapper-created constant *tensor* launch arg by value.
# Lifted module scalars (``torch.tensor([_NEG])``) are tiny; a large constant
# tensor is unexpected here and would bloat the standalone, so reject it clearly.
_MAX_EMBED_CONST_ELEMS = 256


def _embed_jax_const(value: object) -> str:
    """Python source reconstructing a host-wrapper-created constant launch arg as a
    JAX value: a lifted scalar-constant tensor -> ``jnp.array(...)``; a
    specialization scalar -> its int/float/bool literal. These are baked into the
    standalone, whose entrypoint takes only the user's tensor inputs."""
    if isinstance(value, torch.Tensor):
        if value.numel() > _MAX_EMBED_CONST_ELEMS:
            raise NotImplementedError(
                "to_code(jax_fn=True) cannot inline a constant tensor "
                f"launch arg with {value.numel()} elements (limit "
                f"{_MAX_EMBED_CONST_ELEMS})"
            )
        values = value.detach().cpu().tolist()
        return f"jnp.array({values!r}, dtype={_torch_dtype_to_jnp_name(value.dtype)})"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    raise NotImplementedError(
        "to_code(jax_fn=True) does not support a launch arg of type "
        f"{type(value).__name__!r}"
    )


# Distinct scale factors for the second shape probe (see
# ``capture_jax_launch_metadata``):
# one per symbolic input dim, distinct so each launch value maps unambiguously to
# the dim it tracks.
_PROBE_FACTORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _scaled_probe_args(
    args: tuple[object, ...],
    dim_factor: dict[tuple[int, int], int],
) -> list[object]:
    """Second-probe args: each user tensor with its symbolic dims scaled by that
    dim's factor (``dim_factor[(arg_index, dim)]``; dims sharing a symbol scale
    together); non-tensor args and concrete (specialized) dims are left unchanged.

    ``dim_factor`` is precomputed from the symbolic dims captured *before* the base
    capture run, because that run specializes ``bound.fake_args`` to concrete sizes."""
    probe: list[object] = []
    for i, a in enumerate(args):
        if not isinstance(a, torch.Tensor):
            probe.append(a)
            continue
        new_shape = [
            int(a.shape[d]) * dim_factor.get((i, d), 1) for d in range(a.dim())
        ]
        probe.append(torch.empty(new_shape, dtype=a.dtype, device=a.device))
    return probe


def _match_input_dim(
    v0: int,
    v1: int,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str | None:
    """``inputs[k].shape[d]`` for a value that scaled ``v0 -> v1`` across the two
    probes; ``None`` if unchanged (a constant). Raises if it changed but matches no
    input dim (a value we can't derive, rather than silently baking it wrong)."""
    if v0 == v1:
        return None
    for k, (s0, s1) in enumerate(zip(in_shapes0, in_shapes1, strict=True)):
        for d in range(len(s0)):
            if s0[d] == v0 and s1[d] == v1:
                return f"inputs[{k}].shape[{d}]"
    raise NotImplementedError(
        "to_code(jax_fn=True) cannot derive a dynamic launch value "
        f"({v0} -> {v1}) from the input shapes"
    )


def _grid_axis_expr(
    g0: int,
    g1: int,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str:
    """Python expression for one grid axis: a constant literal if it didn't change,
    else ``cdiv(inputs[k].shape[d], block)`` for the input dim it tracks (the block
    is recovered from the sample: ``block = dim / grid`` when the sample dim is a
    whole number of blocks)."""
    if g0 == g1:
        return repr(g0)
    for k, (s0, s1) in enumerate(zip(in_shapes0, in_shapes1, strict=True)):
        for d in range(len(s0)):
            a0, a1 = s0[d], s1[d]
            if a0 == a1 or g0 <= 0 or a0 % g0 != 0:
                continue
            block = a0 // g0
            if block >= 1 and -(-a1 // block) == g1:
                if block == 1:
                    return f"inputs[{k}].shape[{d}]"
                return f"(inputs[{k}].shape[{d}] + {block - 1}) // {block}"
    raise NotImplementedError(
        f"to_code(jax_fn=True) cannot derive grid axis ({g0} -> {g1}) "
        "from the input shapes"
    )


def _const_slot_expr(
    v0: object,
    v1: object,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str:
    """Expression filling a host-wrapper-created launch slot: baked by value if it
    stayed constant across the two probes, else input-derived. Covers lifted module
    scalars / specialization ints (constant) and shape-derived scalars such as a
    reduction's row-count ``torch.tensor([t])`` (a ``(1,)`` tensor that tracks a
    runtime dim)."""
    if isinstance(v0, torch.Tensor):
        vals0 = v0.detach().cpu().reshape(-1).tolist()
        vals1 = cast("torch.Tensor", v1).detach().cpu().reshape(-1).tolist()
        if vals0 == vals1:
            return _embed_jax_const(v0)
        if len(vals0) != 1:
            raise NotImplementedError(
                "to_code(jax_fn=True) cannot derive a multi-element "
                "dynamic constant tensor launch arg"
            )
        expr = _match_input_dim(vals0[0], vals1[0], in_shapes0, in_shapes1)
        return f"jnp.array([{expr}], dtype={_torch_dtype_to_jnp_name(v0.dtype)})"
    if v0 == v1:
        return _embed_jax_const(v0)
    return cast(
        "str",
        _match_input_dim(cast("int", v0), cast("int", v1), in_shapes0, in_shapes1),
    )


def capture_jax_launch_metadata(
    bound: BoundKernel[Any],
    config: Config | dict[str, object],
) -> JaxLaunchMeta:
    """Capture the jax_fn launch metadata by running the compiled host wrapper on
    real tensors with a capturing launcher, and derive the dynamic-shape expressions.

    This is the one non-AST step of the jax_fn path: it records the grid, per-tensor
    block specs, output shapes/dtypes, and input/output arg positions, and runs a
    second probe (each symbolic input dim scaled by a distinct factor) so grid /
    output-shape / scalar-arg values that track a runtime dim become
    ``inputs[i].shape[d]`` expressions rather than baked sample constants. The AST
    builder (:func:`build_jax_fn_ast`) turns this into the emitted entrypoint.

    Kernels using unsupported advanced Pallas features or int64/uint64/float64
    args raise ``NotImplementedError``. Distributed remote-copy kernels capture
    their scratch descriptors, HBM/SMEM placements, dynamic-shape padding, and
    in-place aliases as static launch metadata.

    Must be called *outside* the fake-tensor env (the capture materializes and runs
    on real tensors).
    """
    kernel = bound.kernel
    compiled = bound.compile_config(config)
    # Record which input dims are symbolic (dynamic) BEFORE the capture run below:
    # running the compiled wrapper once specializes ``bound.fake_args``' symbols to
    # the concrete sample sizes, which would otherwise erase them before the
    # two-probe dynamic-shape derivation (further down) can read them. ``sym_dims``
    # maps each symbol to the ``(arg_index, dim)`` positions that carry it.
    sym_dims: dict[str, list[tuple[int, int]]] = {}
    for i, fake in enumerate(bound.fake_args):
        shape = getattr(fake, "shape", None)
        if shape is None:
            continue
        for d, size in enumerate(shape):
            if isinstance(size, torch.SymInt) and size.node.expr.is_symbol:
                sym_dims.setdefault(str(size.node.expr), []).append((i, d))
    # ``to_code`` has no access to the original inputs; reconstruct real,
    # sample-shaped tensors from the bound kernel's fake args to drive the capture.
    args = _materialize_args(bound.fake_args)

    # Capture the launch metadata by running the host wrapper with a launcher
    # that records its arguments instead of executing the kernel.
    captured: dict[str, Any] = {}

    def _launcher_return(
        launch_args: tuple[object, ...], kw: dict[str, object]
    ) -> object:
        output_indices = cast("list[int]", kw.get("_output_indices") or [])
        inplace_indices = set(cast("list[int]", kw.get("_inplace_indices") or []))
        output_only_indices = [
            index for index in output_indices if index not in inplace_indices
        ]
        if len(output_only_indices) > 1:
            return tuple(launch_args[index] for index in output_only_indices)
        if output_only_indices:
            return launch_args[output_only_indices[0]]
        return None

    def _capture(
        pallas_kernel: object, grid: object, *launch_args: object, **kw: object
    ) -> object:
        captured["grid"] = tuple(int(g) for g in cast("Any", grid))
        captured["args"] = launch_args
        captured["kwargs"] = kw
        # The real launcher returns only output-only tensors. In-place aliases
        # are updated through their caller-owned arguments and the host wrapper
        # reads those arguments directly when they are part of the user return.
        return _launcher_return(launch_args, kw)

    compiled(*args, _launcher=_capture)

    kw = captured["kwargs"]
    unsupported_kwargs = list(_JAX_UNSUPPORTED_KWARGS)
    if not kw.get("_uses_remote_copy"):
        unsupported_kwargs.extend(_JAX_REMOTE_COPY_KWARGS)
    for name in unsupported_kwargs:
        if kw.get(name):
            raise NotImplementedError(
                f"to_code(jax_fn=True) does not support kernels using "
                f"{name!r} yet (kernel {kernel.name!r})"
            )

    launch_args = cast("tuple[object, ...]", captured["args"])
    output_indices = list(cast("list[int]", kw.get("_output_indices") or []))
    block_spec_info = cast("list[Any] | None", kw.get("_block_spec_info"))
    if block_spec_info is None:
        # Emitted only when codegen resolved a grid/tiling; its absence means a
        # no-tiling / degenerate-grid kernel the launch core can't map.
        raise NotImplementedError(
            "to_code(jax_fn=True) does not support kernels without a "
            "resolved block spec (no-tiling / degenerate grid) yet"
        )
    for a in launch_args:
        if isinstance(a, torch.Tensor) and a.dtype in _JAX_UNSUPPORTED_DTYPES:
            raise NotImplementedError(
                f"to_code(jax_fn=True) does not support {a.dtype} tensors "
                "(unsupported on TPU / narrowed by JAX x32)"
            )
    # The host wrapper may interleave its own output buffers with the kernel's user
    # arguments, so recover user launch slots by identity instead of assuming the
    # first ``len(args)`` positions correspond one-for-one. The standalone takes
    # only user tensor inputs; wrapper-created buffers/constants are reconstructed.
    user_positions: list[int] = []
    for user_index, user_arg in enumerate(args):
        if not isinstance(user_arg, torch.Tensor):
            continue
        position = next(
            (p for p, launch_arg in enumerate(launch_args) if launch_arg is user_arg),
            None,
        )
        if position is None:
            raise NotImplementedError(
                "to_code(jax_fn=True) cannot map user tensor input "
                f"{user_index} to a Pallas launch argument"
            )
        user_positions.append(position)
    const_positions = [
        p
        for p in range(len(launch_args))
        if p not in user_positions and p not in output_indices
    ]
    out_dtypes = [
        _torch_dtype_to_jnp_name(cast("torch.Tensor", launch_args[p]).dtype)
        for p in output_indices
    ]
    interpret = bool(kw.get("_pallas_interpret") or False)
    collective_id = cast("int | None", kw.get("_collective_id"))

    # Derive the grid, output shapes, and shape-derived scalar launch args from the
    # RUNTIME input shapes so a single standalone is correct at every dynamic shape.
    # One trace can't tell a value that happens to equal the sample size from one
    # that tracks an input dim -- and a materialized row-count ``torch.tensor([t])``
    # even specializes that dim during tracing -- so probe a SECOND shape (each
    # symbolic input dim scaled by a distinct factor) and compare: a launch value
    # that moved tracks the input dim it moved with (derive it); one that stayed is
    # a genuine constant (bake it). Static kernels have no symbolic dims, so every
    # value stays -> all baked (identical standalone as before).
    grid0 = cast("tuple[int, ...]", captured["grid"])
    in_shapes0 = [
        [int(s) for s in cast("torch.Tensor", launch_args[p]).shape]
        for p in user_positions
    ]

    if sym_dims:
        sym_factor = {sym: _PROBE_FACTORS[k] for k, sym in enumerate(sorted(sym_dims))}
        # (arg_index, dim) -> scale factor, from the pre-run symbolic dims (dims
        # sharing a symbol scale together).
        dim_factor: dict[tuple[int, int], int] = {
            pos: sym_factor[sym]
            for sym, positions in sym_dims.items()
            for pos in positions
        }
        probe_cap: dict[str, Any] = {}

        def _probe(pk: object, grid: object, *pa: object, **pkw: object) -> object:
            probe_cap["grid"] = tuple(int(g) for g in cast("Any", grid))
            probe_cap["args"] = pa
            return _launcher_return(pa, pkw)

        probe_args = _scaled_probe_args(args, dim_factor)
        compiled(*probe_args, _launcher=_probe)
        grid1 = cast("tuple[int, ...]", probe_cap["grid"])
        launch1 = cast("tuple[object, ...]", probe_cap["args"])
        in_shapes1 = [
            [int(s) for s in cast("torch.Tensor", launch1[p]).shape]
            for p in user_positions
        ]
    else:
        grid1, launch1, in_shapes1 = grid0, launch_args, in_shapes0

    grid_exprs = [
        _grid_axis_expr(g0, g1, in_shapes0, in_shapes1)
        for g0, g1 in zip(grid0, grid1, strict=True)
    ]
    out_shape_exprs: list[list[str]] = []
    for p in output_indices:
        sh0 = [int(s) for s in cast("torch.Tensor", launch_args[p]).shape]
        sh1 = [int(s) for s in cast("torch.Tensor", launch1[p]).shape]
        out_shape_exprs.append(
            [
                _match_input_dim(a, b, in_shapes0, in_shapes1) or repr(a)
                for a, b in zip(sh0, sh1, strict=True)
            ]
        )
    const_slots = {
        p: _const_slot_expr(launch_args[p], launch1[p], in_shapes0, in_shapes1)
        for p in const_positions
    }

    inplace_indices = list(cast("list[int] | None", kw.get("_inplace_indices")) or [])
    wrapper_created_inplace = [
        position for position in inplace_indices if position not in user_positions
    ]
    if wrapper_created_inplace:
        raise NotImplementedError(
            "to_code(jax_fn=True) cannot reconstruct wrapper-created in-place "
            f"outputs at launch positions {wrapper_created_inplace}; pass the "
            "initialized destination as a kernel input"
        )

    return JaxLaunchMeta(
        kernel_name=kernel.name,
        grid_exprs=grid_exprs,
        output_indices=output_indices,
        inplace_indices=inplace_indices,
        user_positions=user_positions,
        const_slots=const_slots,
        block_spec_info=cast("list[Any]", block_spec_info),
        scratch_shapes=list(
            cast("list[object] | None", kw.get("_scratch_shapes")) or []
        ),
        hbm_arg_indices=list(
            cast("list[int] | None", kw.get("_hbm_arg_indices")) or []
        ),
        smem_arg_indices=list(
            cast("list[int] | None", kw.get("_smem_arg_indices")) or []
        ),
        ds_pad_dims=list(
            cast(
                "list[tuple[int, int, int, int]] | None",
                kw.get("_ds_pad_dims"),
            )
            or []
        ),
        out_shape_exprs=out_shape_exprs,
        out_dtypes=out_dtypes,
        interpret=interpret,
        collective_id=collective_id,
        n_args=len(launch_args),
    )


def _extract_device_kernel_nodes(
    body_root: ast.Module, kernel_name: str
) -> list[ast.stmt]:
    """The device-kernel statements from the generated module AST: everything except
    the host-wrapper ``def <kernel_name>`` (i.e. the ``_helion_<name>`` device
    kernel(s) and any module-level constants). Raises if that code imports helion (an
    in-kernel helper not inlined yet) or references torch in code (the jax standalone
    is jax-native; torch in *annotations* stays a lazy string and is fine)."""
    nodes = [
        node
        for node in body_root.body
        if not (isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Import)
            and any("helion" in alias.name for alias in node.names)
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and "helion" in node.module
        ):
            raise NotImplementedError(
                f"cannot export {kernel_name!r} for jax_fn: the device kernel "
                "references helion (an in-kernel helper is not inlined yet)"
            )
    if "torch" in _code_name_refs(module):
        raise NotImplementedError(
            f"cannot export {kernel_name!r} for jax_fn: the device kernel "
            "references torch (only jax-native device code is supported)"
        )
    return nodes


def _is_torch_import(imp: str) -> bool:
    """True if ``imp`` is an ``import torch`` / ``from torch ...`` statement."""
    return imp == "import torch" or imp.startswith(
        ("import torch.", "import torch ", "from torch.", "from torch ")
    )


def _stmt_def_names(node: ast.stmt) -> list[str]:
    """Top-level names a statement binds (function/class/assignment targets)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _code_name_refs(node: ast.AST) -> set[str]:
    """Names referenced in a node's *code* (calls, attribute bases, values) --
    ignoring type annotations, which stay lazy strings under
    ``from __future__ import annotations`` and never execute at runtime."""
    refs: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Name):
            refs.add(n.id)
        for field, value in ast.iter_fields(n):
            if field in ("annotation", "returns"):
                continue
            if isinstance(value, ast.AST):
                visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        visit(item)

    visit(node)
    return refs


def _launcher_jax_slice() -> tuple[list[str], list[ast.stmt]]:
    """Return ``(import_lines, def_nodes)`` for the JAX-only slice of the Pallas
    launcher: its jax-relevant import statements (as source lines, matching
    ``to_code``'s ``import_lines`` convention) and the AST nodes of the transitive
    *code* closure of ``_pallas_jax_call`` -- the shared compile core
    (``_pallas_compile_jit_fn`` / block specs / ``pl.kernel`` / the compact variant).
    Drops everything else (the torch launcher, JaxCallable dispatch, torch<->jax
    conversions, ``import torch``). torch names left in kept functions' *type
    annotations* are lazy strings that never execute.
    """
    tree = ast.parse(read_launcher_source("helion.runtime.pallas.launcher"))
    import_nodes: list[ast.stmt] = []
    defs: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
        for name in _stmt_def_names(node):
            defs[name] = node

    keep: set[str] = set()
    queue = ["_pallas_jax_call"]
    while queue:
        name = queue.pop()
        if name in keep or name not in defs:
            continue
        keep.add(name)
        queue.extend(_code_name_refs(defs[name]))

    kept = [node for node in tree.body if any(n in keep for n in _stmt_def_names(node))]
    # Torch in annotations is fine (lazy strings); torch in *code* is a bug.
    if "torch" in _code_name_refs(ast.Module(body=kept, type_ignores=[])):
        raise AssertionError(
            "jax_fn launcher slice unexpectedly references torch in code; the "
            "compile core reachable from _pallas_jax_call must stay torch-free."
        )
    import_lines = [
        line
        for node in import_nodes
        if "helion" not in (line := ast.unparse(node)) and not _is_torch_import(line)
    ]
    return import_lines, kept


def build_jax_fn_ast(
    body_root: ast.Module,
    import_lines: list[str],
    meta: JaxLaunchMeta,
    *,
    inline_launcher: bool,
) -> ast.Module:
    """Rewrite the generated module AST into the jax-native standalone module.

    Takes the generated ``body_root`` (the ``_helion_<name>`` device kernel + the
    host wrapper) and the captured ``meta``; returns a new module AST whose
    entrypoint operates on ``jax.Array`` inputs and drives ``_pallas_jax_call`` (the
    same launch path the jax_fn runtime uses). ``import_lines`` is mutated in place to
    the jax import set; the launch core is inlined as AST nodes (``inline_launcher``,
    pure-jax) or imported from helion. The single ``unparse`` in
    ``BoundKernel.to_code`` renders the returned module.
    """
    device_nodes = _extract_device_kernel_nodes(body_root, meta.kernel_name)
    device_kernel = f"_helion_{meta.kernel_name}"
    jax_header = [
        "import jax",
        "import jax.numpy as jnp",
        "from jax.experimental import pallas as pl",
    ]
    preamble: list[ast.stmt] = []
    if inline_launcher:
        # Pure jax: keep the generated jax imports (drop helion/torch), inline the
        # jax-only launcher slice, and embed any in-kernel helpers the device uses.
        gen = [
            imp
            for imp in import_lines
            if "helion" not in imp and not _is_torch_import(imp)
        ]
        launcher_imports, launcher_nodes = _launcher_jax_slice()
        new_imports = dedupe_preserve_order([*jax_header, *gen, *launcher_imports])
        referenced = _code_name_refs(ast.Module(body=device_nodes, type_ignores=[]))
        embedded = _embedded_helper_source(" ".join(sorted(referenced)))
        helper_nodes = ast.parse(embedded).body if embedded else []
        preamble = [*launcher_nodes, *helper_nodes]
    else:
        # jax + helion deps: import the launch core (keep any in-kernel helper
        # imports); drop torch and the torch launcher import.
        gen = [
            imp
            for imp in import_lines
            if not _is_torch_import(imp) and "default_pallas_launcher" not in imp
        ]
        new_imports = dedupe_preserve_order(
            [
                *jax_header,
                *gen,
                "from helion.runtime.pallas.launcher import _pallas_jax_call",
            ]
        )
    # Launch metadata + the jax entrypoint, built from generated snippets (ast.parse
    # constructs each node; no round-trip of the device code, which stays body_root
    # nodes). block_spec_info's repr is a list of tuples, so it parses cleanly.
    meta_nodes: list[ast.stmt] = [
        ast.parse(f"_BLOCK_SPEC_INFO = {meta.block_spec_info!r}").body[0],
        ast.parse(f"_SCRATCH_SHAPES = {meta.scratch_shapes!r}").body[0],
        ast.parse(f"_HBM_ARG_INDICES = {meta.hbm_arg_indices!r}").body[0],
        ast.parse(f"_SMEM_ARG_INDICES = {meta.smem_arg_indices!r}").body[0],
        ast.parse(f"_DS_PAD_DIMS = {meta.ds_pad_dims!r}").body[0],
        ast.parse(f"_OUTPUT_INDICES = {meta.output_indices!r}").body[0],
        ast.parse(f"_INPLACE_INDICES = {meta.inplace_indices!r}").body[0],
        ast.parse(f"_USER_POSITIONS = {meta.user_positions!r}").body[0],
        ast.parse(f"_INTERPRET = {meta.interpret!r}").body[0],
        ast.parse(f"_COLLECTIVE_ID = {meta.collective_id!r}").body[0],
        ast.parse(f"_N_ARGS = {meta.n_args}").body[0],
    ]
    entrypoint = ast.parse(_jax_entrypoint_source(meta, device_kernel)).body[0]
    import_lines[:] = new_imports
    module = ast.Module(
        body=[*preamble, *device_nodes, *meta_nodes, entrypoint],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return module


def _jax_entrypoint_source(meta: JaxLaunchMeta, device_kernel: str) -> str:
    """Source of the jax-native entrypoint: fills the launch slots from the runtime
    inputs -- grid, output shapes, and shape-derived scalars are all derived from
    ``inputs[i].shape[d]`` (see the two-probe in ``capture_jax_launch_metadata``), so
    a single standalone is correct at every dynamic shape -- then drives
    ``_pallas_jax_call``."""
    user_position_set = set(meta.user_positions)
    out_lines = [
        f"    slots[{pos}] = jnp.empty("
        f"({', '.join(meta.out_shape_exprs[oi])},), {meta.out_dtypes[oi]})"
        for oi, pos in enumerate(meta.output_indices)
        if pos not in user_position_set
    ]
    const_lines = [
        f"    slots[{p}] = {expr}" for p, expr in sorted(meta.const_slots.items())
    ]
    # The explanation goes in the docstring rather than ``#`` comments: this source is
    # round-tripped through ``ast.parse`` (comments are dropped, the docstring node is
    # kept). The const-slots sentence is conditional so it appears only when present.
    doc = (
        "Standalone jax entrypoint over the user inputs. Grid and output shapes "
        "derive from the runtime input shapes (two-probe capture), so one module is "
        "correct at every dynamic input shape."
    )
    if meta.const_slots:
        doc += " Extra slots are constants baked in from the original host wrapper."
    lines = [
        f"def {meta.kernel_name}(*inputs):",
        f'    """{doc}"""',
        f"    _grid = ({', '.join(meta.grid_exprs)},)",
        "    slots = [None] * _N_ARGS",
        "    for pos, inp in zip(_USER_POSITIONS, inputs, strict=True):",
        "        slots[pos] = inp",
        *out_lines,
        *const_lines,
        "    orig_shapes = {pos: tuple(slots[pos].shape) for pos in _OUTPUT_INDICES}",
        "    for arg_idx, dim, block_size, extra_pad in _DS_PAD_DIMS:",
        "        value = slots[arg_idx]",
        "        pad_amount = (-value.shape[dim]) % block_size + extra_pad",
        "        if pad_amount:",
        "            pad_widths = [(0, 0)] * value.ndim",
        "            pad_widths[dim] = (0, pad_amount)",
        "            slots[arg_idx] = jnp.pad(value, pad_widths)",
        "    results = _pallas_jax_call(",
        f"        {device_kernel},",
        "        _grid,",
        "        tuple(slots),",
        "        output_indices=_OUTPUT_INDICES,",
        "        inplace_indices=_INPLACE_INDICES,",
        "        block_spec_info=_BLOCK_SPEC_INFO,",
        "        scratch_shapes=_SCRATCH_SHAPES,",
        "        hbm_arg_indices=_HBM_ARG_INDICES,",
        "        smem_arg_indices=_SMEM_ARG_INDICES,",
        "        collective_id=_COLLECTIVE_ID,",
        "        interpret=_INTERPRET,",
        "        compact=None,",
        "        orig_shapes=orig_shapes,",
        "        ds_pad_dims=_DS_PAD_DIMS,",
        "    )",
        "    return results[0] if len(results) == 1 else tuple(results)",
    ]
    return "\n".join(lines)
