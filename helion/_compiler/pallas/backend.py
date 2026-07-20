"""PallasBackend backend class, moved out of the backend-neutral
helion/_compiler/backend.py."""

from __future__ import annotations

import ast
import enum
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..backend import Backend
from ..backend import _loop_contains_matmul

if TYPE_CHECKING:
    import sympy
    from torch._inductor.ops_handler import OpsHandler

    from ...autotuner.config_fragment import ConfigSpecFragment
    from ...runtime.config import Config
    from ...runtime.settings import DotPrecision
    from ..device_function import Argument
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..tile_dispatch import TileStrategyDispatch
    from .compact_worklist import CompactWorklistPlan

    InductorOpOverrides = OpsHandler[Any]


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

        Pallas/TPU has limited support for different precisions, often
        falling back to the highest available precision.
        """
        pallas_precision_by_dot_precision = {
            "default": "default",
            # "high" is mapped to "highest" because Pallas/Mosaic doesn't yet
            # support it on TPU.
            "high": "highest",
            "highest": "highest",
            "tf32": "highest",
            "tf32x3": "highest",
            "ieee": "highest",
        }
        return pallas_precision_by_dot_precision.get(precision, "default")

    @property
    def max_tensor_numel(self) -> int | None:
        # No compile-time element cap on Pallas; VMEM byte budget is the
        # real constraint and is enforced separately at runtime.
        return None

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        return False

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
        }

    # Config keys that Pallas actually uses.  Everything else
    # (pid_type, num_warps, num_stages, maxnreg, indexing, etc.)
    # is GPU-specific and should not be tuned.
    _PALLAS_SUPPORTED_KEYS: frozenset[str] = frozenset(
        {
            "block_sizes",
            "loop_orders",
            "flatten_loops",
            "pallas_loop_type",
            "pallas_load_buffer_count",
            "pallas_pre_broadcast",
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
        tensor dimensions use ``pl.ds()`` slicing.

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
        if not device_fn.pallas_pad_info:
            return None

        result: list[tuple[int, int, int, int]] = []
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

        return result or None

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
        pallas_loop_type = config.get("pallas_loop_type", "unroll")
        scratch_shapes = [
            (
                s.shape,
                self.dtype_str(s.dtype) if s.dtype is not None else None,
                s.scratch_type,
            )
            for s in device_fn._scratch_args
        ]
        if scratch_shapes:
            launcher_args.append(f"_scratch_shapes={scratch_shapes!r}")

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

        if pallas_loop_type == "compact_worklist" and sorted_args is not None:
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
        # Compact-tile tensors (aligned load + exact store) both get a per-tile
        # pl.Element BlockSpec sliced at tile_start, so Pallas double-buffers BOTH
        # the load prefetch and the store write-back across work items.
        #
        # The store is a masked full-block write.  The two robust EXACT-store
        # alternatives were both worse/unavailable here: (a) staging VMEM +
        # make_async_copy over pl.ds(tile_start, tile_extent) serializes (~1.8x
        # slower: 4.5ms vs 2.5ms) because a straight-line compact tile has no inner
        # loop to overlap; (b) a pl.BoundedSlice store BlockSpec (exact + double-buffered)
        # is rejected by this JAX's Mosaic lowering ("Unsupported block dimension
        # type: BoundedSlice" -- it only works inside pltpu.emit_pipeline).  The
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
            f"_compact_block={env.compact_worklist_block}",
            f"_compact_ordered_aligned_arg_indices={ordered_indices!r}",
            f"_compact_range_start_ref_pos={range_start_ref_pos}",
            f"_compact_ordered_offset_arg_index={ordered_offset_arg_index}",
            f"_compact_active_mask_arg_index={active_mask_arg_index}",
            f"_compact_ordered_window={ordered_window}",
        ]

    def build_launcher_name(self, config: Config) -> str:
        """Return the single Pallas launcher name.

        All ``pallas_loop_type`` values (``unroll``, ``emit_pipeline``,
        ``fori_loop``, ``compact_worklist``) route through the same
        ``default_pallas_launcher``; the loop-shape choice happens
        inside based on the launcher-observable kwargs codegen emits.
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
        from ..compile_environment import CompileEnvironment
        from .plan_tiling import plan_tiling

        plan_tiling(graphs, config, tile_strategy)

        # compact_worklist_* is per-CONFIG state, but one CompileEnvironment is
        # reused across all configs of a BoundKernel (see CompileEnvironment's
        # "no config-specific state" contract).  Reset before re-detecting so a
        # later non-compact config never inherits a prior compact config's plan
        # -- many lowering paths gate on ``env.compact_worklist_plan is not None``
        # (PID strategy, loop-bound remap, fori handling, ds slicing), not on
        # pallas_loop_type, so a stale plan would mis-lower a fori/emit config.
        env = CompileEnvironment.current()
        env.compact_worklist_plan = None
        env.compact_worklist_resident_cache_decision = None
        env.compact_worklist_resident_prep_hoists = ()
        env.compact_worklist_upper = 1
        env.compact_worklist_block = 1
        env.compact_worklist_ordered_block = 1
        env.compact_worklist_offset_params = []

        if config.get("pallas_loop_type") == "compact_worklist":
            self._setup_compact_worklist(graphs, config)

    def _setup_compact_worklist(self, graphs: list[GraphInfo], config: Config) -> None:
        """Detect + stash the compact-worklist plan before device codegen.

        Runs early (pre_codegen) so ``env.compact_worklist_plan`` is set when the
        grid strategy selects ``WorklistProgramIDs`` and the inner loop remaps its
        begin/end to metadata refs.  Registers the N metadata ref names as
        ``wrapper_only_params`` (kernel-signature-only) and computes the static
        megablocks ``UPPER``.  ``detect_*`` raises ``exc.InvalidConfig`` on a
        non-matching kernel (autotuner-skippable).
        """
        from ...runtime import _get_vmem_limit_bytes
        from ..compile_environment import CompileEnvironment
        from ..device_function import DeviceFunction
        from ..host_function import HostFunction
        from .compact_worklist import build_resident_cache_admission
        from .compact_worklist import detect_compact_worklist_plan
        from .compact_worklist import metadata_arg_names

        env = CompileEnvironment.current()
        host_fn = HostFunction.current()
        plan = detect_compact_worklist_plan(host_fn)
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

        import jax.experimental.pallas.tpu as pltpu

        # Choose C from the conservative device-reported VMEM budget.  The runtime
        # may pass a higher Mosaic compile ceiling for resident-window kernels so TPU7x
        # accepts this already-sized allocation, but that ceiling is deliberately
        # not used here as an allocation budget.
        vmem_bytes = _get_vmem_limit_bytes(pltpu)
        admission = build_resident_cache_admission(
            graphs,
            plan,
            host_fn.params.arguments,
            ordered_block=env.compact_worklist_ordered_block,
            vmem_bytes=vmem_bytes,
        )
        env.compact_worklist_resident_prep_hoists = admission.prep_hoists
        env.compact_worklist_resident_cache_decision = admission.decision

    def _compact_worklist_upper(
        self, plan: CompactWorklistPlan, config: Config, host_fn: HostFunction
    ) -> int:
        """Static UPPER: the padded length of the worklist metadata arrays.

        Must be >= the worst-case ``num_work = sum_owners cdiv(length, BLOCK)``,
        else the dynamic Pallas grid indexes past the scalar-prefetch metadata
        (``jnp.repeat(total_repeat_length=UPPER)`` would silently truncate the
        worklist).  Detection only accepts the packed-offsets idiom (store
        safety), so owner ranges are contiguous/non-overlapping
        (``sum(length) == total``) and the tight megablocks bound
        ``cdiv(total, BLOCK) + num_owners - 1`` provably holds.  All terms are
        concrete ints under ``static_shapes=True``.
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
        # total_compact = leading dim of the compact_aligned_load tensor.
        compact_arg = next(
            p.arg_name for p in plan.tensor_policies if p.kind == "compact_aligned_load"
        )
        total = int(params[compact_arg].shape[0])
        block = CompileEnvironment.current().compact_worklist_block
        # Single source of the tight megablocks bound (also unit-tested).
        return packed_upper_bound(total, num_owners, block)
