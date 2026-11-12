"""CPU-only CuTe code generation helpers shared by the test suite."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from contextlib import contextmanager
import dataclasses
import importlib.util
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest import SkipTest
from unittest.mock import patch

import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils._pytree import tree_leaves

from .._compiler.compile_environment import CompileEnvironment
from .._compiler.cute.cutedsl_compat import _cute_backend_requirement_error
from .._compiler.cute.full_tile_bounds import prove_full_tile_launch
from .._compiler.cute.lane_loops.barriers import is_barrier as is_barrier
from .._compiler.cute.target_facts import B200
from .._utils import triton_is_available
from ..runtime.config import Config
from ..runtime.ref_mode import RefMode
from ..runtime.ref_mode import RefModeContext
from ..runtime.ref_mode import RefModeTorchFunctionMode
from . import cpu_render_target
from . import default_cute_mma_support
from . import patch_cute_mma_support
from . import patch_target_facts

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Sequence
    from pathlib import Path
    from types import ModuleType
    from typing import Any
    from unittest.mock import MagicMock

    from torch._dynamo.source import Source

    from .._compiler.cute.full_tile_bounds import FullTilePlan
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import Kernel
    from ..runtime.settings import Settings


def tcgen05_mma_support() -> SimpleNamespace:
    """Fresh legacy tcgen05 support record, without supported_impls or FP8."""
    return SimpleNamespace(universal=True, warp_f16bf16=True, tcgen05_f16bf16=True)


def b200_target() -> ExitStack:
    """The B200 capability, SM count and HIP flag without CUDA work."""
    stack = ExitStack()
    stack.enter_context(
        patch_target_facts(capability=(10, 0), num_sm=148, is_hip=False)
    )
    return stack


def tcgen05_target() -> ExitStack:
    """``b200_target`` plus tcgen05 MMA support, the B200 opt-in SMEM capacity
    (the budget derives from it) and the B200 hardware identity."""
    stack = b200_target()
    stack.enter_context(patch_cute_mma_support(tcgen05_mma_support()))
    stack.enter_context(
        patch_target_facts(
            smem_capacity_bytes=232448, hardware=B200.hardware_info(), is_hip=False
        )
    )
    return stack


def check_config_compile_options(bound: BoundKernel, config: Config) -> None:
    """Exercise a real emitter while keeping its device source and ABI fixed."""
    original = ast.parse(bound.to_code(config))
    requested = Config.from_dict(
        config.config
        | {
            "cute_compiler_pipeline": "opt2",
            "advanced_controls_file": "control.txt",
        }
    )
    configured = ast.parse(bound.to_code(requested))
    observed = []
    for node in ast.walk(configured):
        if isinstance(node, ast.Call):
            for keyword in list(node.keywords):
                if keyword.arg == "cute_compile_options":
                    observed.append(ast.literal_eval(keyword.value))
                    node.keywords.remove(keyword)
    assert observed == ["--opt-level 2 --ptxas-options='--apply-controls control.txt'"]
    assert ast.dump(configured) == ast.dump(original)


def stage_sources(source: str) -> list[str]:
    """Extract ordered stage modules embedded in a generated wrapper."""
    return [
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "PyCodeCache.load"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


def load_generated_module(path: Path) -> ModuleType:
    """Execute a generated module anew, including repeated uses of one stem."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_cute_codegen() -> None:
    # Source generation imports SDK types even when native compilation is
    # forbidden. Keep these tests runnable in Triton-only installations.
    if reason := _cute_backend_requirement_error():
        raise SkipTest(reason)


def codegen_skip_reason(backend: str) -> str | None:
    """Report why ``backend`` cannot generate source, without raising."""
    if backend == "cute":
        return _cute_backend_requirement_error()
    if backend in ("triton", "tileir") and not triton_is_available():
        return "Triton is required for source generation"
    return None


def kernel_body(code: str) -> str:
    """The ``@cute.kernel`` function of rendered module ``code``, up to the
    next top-level function (the host function), without the ``# src[...]``
    comments that quote the kernel's source."""
    start = code.index("@cute.kernel")
    kernel_def = code.index("\ndef ", start)
    end = code.find("\ndef ", kernel_def + 1)
    return "\n".join(
        line
        for line in code[start : end if end >= 0 else len(code)].splitlines()
        if "# src[" not in line
    )


def cpu_bind(
    kernel: Kernel[object],
    args: tuple[object, ...],
    *,
    capability: tuple[int, int] | None = (10, 0),
) -> BoundKernel[object]:
    """Bind ``kernel`` for code generation on a CPU-only process.

    ``capability`` is the compute capability the bound kernel targets; ``None``
    models an unknown target (``target_device_capability`` without CUDA).
    """
    if reason := codegen_skip_reason(kernel.settings.backend):
        raise SkipTest(reason)
    with patch_target_facts(capability=capability, num_sm=148, is_hip=False):
        return kernel._bind_isolated(args)


def roll_loops(code: str, target: str = "roffset_1") -> list[ast.For]:
    """The loops over ``target`` in the kernel of rendered module ``code``,
    in order: a looped reduction's sweeps (reduce, then consume)."""
    kernel = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    return [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For) and ast.unparse(node.target) == target
    ]


def exact_division_redo_branches(tree: ast.AST) -> list[ast.If]:
    """The exact division's redo branches in ``tree``: its
    ``if _cute_exact_div_redo_*(...)`` statements."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "_cute_exact_div_redo_" in ast.unparse(node.test)
    ]


@contextmanager
def mock_cuda_unavailable() -> Generator[None, None, None]:
    """Model CPU code generation without filling process-wide device caches.

    The capability probes are ``functools.cache``'d for the process, so each
    is patched itself rather than computed under ``torch.cuda.is_available``
    False; the CuTe environment check becomes a skip when the DSL is missing
    so source-generation tests stay runnable in Triton-only installations.
    """
    with (
        patch(
            "helion._compiler.cute.backend.CuteBackend.validate_environment",
            side_effect=require_cute_codegen,
        ),
        patch("torch.cuda.is_available", return_value=False),
        patch("helion._compat._supports_maxnreg", return_value=False),
        patch("helion._compat._supports_tensor_descriptor", return_value=False),
        patch("helion._compat._is_hip", return_value=False),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.autotuner.config_spec.num_compute_units", return_value=128),
    ):
        yield


@contextmanager
def forbid_native_compile() -> Generator[list[MagicMock], None, None]:
    """Fail if CuTe DSL compilation runs; a no-op when cutlass is not installed."""
    with ExitStack() as stack:
        mocks: list[MagicMock] = []
        if importlib.util.find_spec("cutlass") is not None:
            mocks.extend(
                stack.enter_context(
                    patch(target, side_effect=AssertionError("native forbidden"))
                )
                for target in ("cutlass.cute.compile", "cutlass.compile")
            )
        yield mocks


@contextmanager
def forbid_cuda_work(*, forbid_native: bool = True) -> Generator[None, None, None]:
    """Reject CUDA work; optionally allow device compilation without a JIT engine."""
    initialized = torch.cuda.is_initialized()
    with ExitStack() as stack:
        stack.enter_context(RejectNativeCuda())
        guards = stack.enter_context(forbid_native_compile()) if forbid_native else []
        guards.extend(
            stack.enter_context(
                patch(target, side_effect=AssertionError("CUDA work forbidden"))
            )
            for target in (
                "torch.cuda._lazy_init",
                "torch.cuda.init",
                "torch.cuda.synchronize",
                "torch.cuda.current_stream",
                "torch.cuda.default_stream",
                "torch.cuda.Stream",
                "torch.cuda.Event",
                "torch.cuda.CUDAGraph",
            )
        )
        yield
        for guard in guards:
            guard.assert_not_called()
        assert torch.cuda.is_initialized() == initialized


class RejectNativeCuda(TorchDispatchMode):
    """Stop direct ATen dispatch before it can reach a live CUDA context."""

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = {} if kwargs is None else kwargs
        inputs = tree_leaves((args, kwargs))
        fake_modes = [
            mode
            for mode in _get_current_dispatch_mode_stack()
            if isinstance(mode, FakeTensorMode)
        ]
        for value in inputs:
            if isinstance(value, FakeTensor) and value.fake_mode not in fake_modes:
                fake_modes.append(value.fake_mode)
        for value in inputs:
            if (
                isinstance(value, torch.device)
                and value.type == "cuda"
                and not fake_modes
            ):
                raise AssertionError("CUDA device forbidden by dispatch guard")
            if (
                isinstance(value, torch.Tensor)
                and not isinstance(value, FakeTensor)
                and value.device.type == "cuda"
            ):
                raise AssertionError("CUDA tensor forbidden by dispatch guard")
        # Fake metadata operations need an installed mode or a typed fake input;
        # native fallback must not escape while this dispatch guard is popped.
        with ExitStack() as stack:
            for mode in fake_modes:
                stack.enter_context(patch.object(mode, "allow_fallback_kernels", False))
            result = func(*args, **kwargs)
        for value in tree_leaves(result):
            if (
                isinstance(value, torch.Tensor)
                and not isinstance(value, FakeTensor)
                and value.device.type == "cuda"
            ):
                raise AssertionError("CUDA result forbidden by dispatch guard")
        return result


@contextmanager
def cpu_target(*, forbid_native: bool = True) -> Generator[None, None, None]:
    """Generate tcgen05 CuTe source for a 148-SM sm100 target without CUDA.

    ``cpu_render_target`` with the default (tcgen05-on) MMA support record and
    the full 227 KiB opt-in SMEM as the AB budget: the B200 model's compute
    capability, SM count and HIP flag, ``mock_cuda_unavailable`` and the
    ``torch.cuda._lazy_init`` guard that fails the test if anything
    initializes CUDA. ``forbid_native=False`` permits offline device compilation
    and selects the SDK's sm100a target without probing the physical device.
    """
    with ExitStack() as stack:
        stack.enter_context(
            cpu_render_target(
                mma_support=default_cute_mma_support(),
                smem_budget_bytes=232448,
                forbid_native=forbid_native,
            )
        )
        if not forbid_native:
            from cutlass.cutlass_dsl import CuTeDSL

            # The SDK caches its environment at import. Setting CUTE_DSL_ARCH
            # later can still probe CUDA through its lazy architecture getter.
            # Its module cache key includes this plain backing field, so change
            # it without reading the getter and restore it on every exit.
            stack.enter_context(
                patch.object(CuTeDSL._get_dsl().envar, "_arch", "sm_100a")
            )
        yield


def cuda_fake(backend: object, tensor: torch.Tensor) -> torch.Tensor:
    """Stand-in for a backend's ``normalize_input_fake_tensor`` on a CPU process.

    Re-homes a CPU fake tensor on ``cuda:0`` so code generation sees device
    inputs without a CUDA context: CuTe's device-type gates (the tcgen05
    planners, the recurrence planners and the TMA routes) then see a CUDA
    target, and so do the Inductor lowerings that differ by device
    (addcdiv's ``div_rn``), which a Triton test patches it in for.
    """
    assert isinstance(tensor, FakeTensor)
    with torch._C._DisableTorchDispatch():
        meta = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=torch.device("meta"),
            requires_grad=tensor.requires_grad,
        )
    return FakeTensor(tensor.fake_mode, meta, torch.device("cuda:0"))


def full_tile_launch_plan(
    prefix: str = "n = input_tensor.numel(); flat = input_tensor.view(-1)",
    *,
    grid: str = "((n + BLOCK - 1) // BLOCK,)",
    block: str = "(4, 1, 1)",
    arguments: str = "flat, out, n",
    keyword: str = "block",
    integer_parameters: tuple[str, ...] = ("n",),
    constants: dict[str, Any] | None = None,
    rank: int = 1,
) -> FullTilePlan | None:
    body = ast.parse(
        prefix
        + "\nout = torch.empty_like(flat)\nBLOCK = 8\n"
        + f"_launcher(kernel, {grid}, {arguments}, {keyword}={block})\n"
        + "return out\n"
    ).body
    return prove_full_tile_launch(
        body,
        kernel_name="kernel",
        parameter_names=("x", "y", "n"),
        tensor_parameters={"x": rank, "y": rank},
        integer_parameters=integer_parameters,
        tensor_inputs={"input_tensor": 2, "other": 2},
        scalar_inputs={"length", "runtime"},
        constexpr_values={"BLOCK": 8} if constants is None else constants,
    )


@contextmanager
def cuda_metadata_trace() -> Generator[None, None, None]:
    """Real static converter, with only the traced device replaced by CUDA."""
    old_init = CompileEnvironment.__init__
    old_convert = CompileEnvironment._to_fake_tensor
    old_threads = torch.get_num_threads()
    initial_cuda_state = torch.cuda.is_initialized()
    torch.set_num_threads(1)

    def initialize(
        env: CompileEnvironment,
        device: torch.device,
        settings: Settings,
        *,
        index_dtype: torch.dtype | None = None,
        is_distributed: bool = False,
    ) -> None:
        assert device.type in ("cpu", "cuda")
        old_init(
            env,
            torch.device("cuda:0"),
            settings,
            index_dtype=index_dtype,
            is_distributed=is_distributed,
        )

    def convert(
        env: CompileEnvironment, tensor: torch.Tensor, origin: Source
    ) -> FakeTensor:
        result = old_convert(env, tensor, origin)
        assert isinstance(result, FakeTensor) and result.fake_mode is env.fake_mode
        result.fake_device = torch.device("cuda:0")
        return result

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch_target_facts(
                    capability=(10, 0),
                    num_sm=148,
                    is_hip=False,
                    smem_capacity_bytes=232448,
                    hardware=B200.hardware_info(),
                )
            )
            stack.enter_context(
                patch_cute_mma_support(default_cute_mma_support(tcgen05_f8=False))
            )
            stack.enter_context(
                patch.object(CompileEnvironment, "__init__", initialize)
            )
            stack.enter_context(
                patch.object(CompileEnvironment, "_to_fake_tensor", convert)
            )
            stack.enter_context(mock_cuda_unavailable())
            stack.enter_context(
                patch(
                    "torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")
                )
            )
            stack.enter_context(
                patch("torch.cuda.get_device_capability", return_value=(10, 0))
            )
            stack.enter_context(
                patch.object(
                    DeviceProperties,
                    "create",
                    return_value=DeviceProperties(
                        type="cuda",
                        index=0,
                        multi_processor_count=148,
                        cc=100,
                        major=10,
                        regs_per_multiprocessor=65536,
                        max_threads_per_multi_processor=2048,
                        max_threads_per_block=1024,
                        warp_size=32,
                    ),
                )
            )
            yield
    finally:
        torch.set_num_threads(old_threads)
        assert torch.cuda.is_initialized() == initial_cuda_state


def _fp32_product(lhs: object, rhs: object) -> torch.Tensor:
    lhs, rhs = cast("torch.Tensor", lhs), cast("torch.Tensor", rhs)
    assert lhs.device.type == rhs.device.type == "cpu"
    return lhs.float() @ rhs.float()


class CpuRefMode(RefModeTorchFunctionMode):
    """The ref-mode torch function mode for a reference run on CPU, where
    ``torch.mm`` has no ``out_dtype``: ``hl.dot``'s ``torch.mm(lhs, rhs,
    out_dtype=torch.float32)`` and an ``addmm`` into an FP32 accumulator
    multiply the operands in FP32 explicitly."""

    def __init__(self) -> None:
        super().__init__()
        self._func_handlers[torch.mm] = self._handle_mm

    def _handle_mm(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> torch.Tensor:
        if "out_dtype" not in kwargs:
            return cast("Callable[..., torch.Tensor]", torch.mm)(*args, **kwargs)
        assert kwargs["out_dtype"] is torch.float32
        return _fp32_product(*args)

    def _handle_mm_with_bias(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        mm_func: object,
        op_name: str,
    ) -> torch.Tensor:
        assert op_name == "addmm"
        bias = cast("torch.Tensor", args[0])
        assert bias.dtype is torch.float32 and bias.device.type == "cpu"
        result = _fp32_product(*args[1:3])
        alpha = cast("float", kwargs.get("alpha", 1))
        if alpha != 1:
            result = result * alpha
        # As torch.addmm: a zero beta ignores the bias, NaN and Inf included.
        beta = cast("float", kwargs.get("beta", 1))
        return result if beta == 0 else result + beta * bias


def reference_host_function(bound: BoundKernel[object]) -> ast.FunctionDef:
    """The host function ``bound`` generates, parsed back from its source for
    a test to edit (to give its ``hl.tile`` loops block sizes, say) before
    ``compile_reference`` builds it."""
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        definition = host.codegen_function_def(list(host.body))
    parsed = ast.parse(ast.unparse(ast.fix_missing_locations(definition))).body[0]
    assert isinstance(parsed, ast.FunctionDef)
    return parsed


def compile_reference(
    bound: BoundKernel[object], definition: ast.FunctionDef, filename: str
) -> Callable[..., object]:
    """``definition``, a host function of ``bound``, compiled into the
    kernel's globals without the CuTe launcher, for ``run_cpu_reference``."""
    namespace = dict(bound.kernel.fn.__globals__)
    namespace["_default_cute_launcher"] = None
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[definition.name]


def run_cpu_reference(
    bound: BoundKernel[object], fn: Callable[..., object], args: Sequence[object]
) -> object:
    """``fn(*args)`` in ref mode EAGER on CPU under ``CpuRefMode``, with the
    settings of ``bound``: the kernel's function, or a host function
    ``compile_reference`` built."""
    with patch_target_facts(capability=(10, 0), num_sm=148, is_hip=False):
        env = CompileEnvironment(
            torch.device("cpu"),
            dataclasses.replace(bound.settings, ref_mode=RefMode.EAGER),
        )
    context = RefModeContext(env, None)
    context.func_mode = CpuRefMode()
    with context:
        return fn(*args)
