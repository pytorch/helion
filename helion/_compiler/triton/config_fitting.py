"""Resource fitting for AOT configs selected from another device."""

from __future__ import annotations

import ast
import functools
import hashlib
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import torch

from ..ast_extension import clone_ast
from ..backend import log

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from triton.runtime.errors import OutOfResources

    from ...runtime.config import Config
    from ...runtime.kernel import BoundKernel
    from .backend import TritonBackend

_MAX_DEVICE_FIT_ATTEMPTS = 16
_CONTRACT = "_helion_launch_contract"
_STATEFUL_LAUNCH_ARGS = (
    "_remote_copy_signal_slots_per_program",
    "_remote_barrier_signal_slots_per_program",
    "_remote_copy_scratch_specs",
    "_persistent_state_specs",
    "_minimum_resident_programs",
    "launch_cooperative_grid",
)


def annotate_launch_contract(root: ast.Module) -> None:
    """Record which generated hosts can share a launch's grid and arguments.

    Device bodies may change, but all host code and module bindings must match.
    Only literal launch stages/warps are ignored. This runs on the compiler's
    AST, before rendering; fitting never needs a source file or source parsing.
    """

    class LaunchParameters(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "_launcher":
                for keyword in node.keywords:
                    if (
                        keyword.arg in ("num_warps", "num_stages")
                        and isinstance(keyword.value, ast.Constant)
                        and type(keyword.value.value) is int
                    ):
                        keyword.value = ast.Constant(value=0)
            return node

    def is_device_function(stmt: ast.stmt) -> bool:
        return isinstance(stmt, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "jit"
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            for original in stmt.decorator_list
            for decorator in [
                original.func if isinstance(original, ast.Call) else original
            ]
        )

    host = ast.Module(
        body=[
            cast("ast.stmt", clone_ast(stmt))
            for stmt in root.body
            if not is_device_function(stmt)
        ],
        type_ignores=[],
    )
    key = hashlib.sha256(ast.dump(LaunchParameters().visit(host)).encode()).hexdigest()
    root.body.append(
        ast.Assign(
            targets=[ast.Name(id=_CONTRACT, ctx=ast.Store())],
            value=ast.Constant(value=key),
        )
    )


class _Candidate(NamedTuple):
    config: Config
    compiled: Callable[..., object]
    same_host: bool = True


def _smaller_config(
    backend: TritonBackend, config: Config, error: OutOfResources
) -> Config | None:
    """Lower pipeline stages or effective warps, preserving tile sizes."""
    from ...runtime.config import Config

    if error.name == "shared memory":
        per_loop = cast("list[int]", config.config.get("range_num_stages", []))
        stages = max([config.num_stages, *per_loop])
        if stages > 1:
            estimate = 1 + (stages - 1) * error.limit // error.required
            stages = max(1, min(stages - 1, estimate))
            updates: dict[str, object] = {"num_stages": min(config.num_stages, stages)}
            if per_loop:
                updates["range_num_stages"] = [min(value, stages) for value in per_loop]
            return Config.from_dict({**config.config, **updates})
    elif error.name == "threads" and config.num_warps > 1:
        smaller = Config.from_dict(
            {**config.config, "num_warps": config.num_warps // 2}
        )
        if backend.effective_num_warps(smaller) < backend.effective_num_warps(config):
            return smaller
    return None


def _next_candidate(
    backend: TritonBackend,
    bound: BoundKernel[Any],
    original: Callable[..., object],
    config: Config,
    error: OutOfResources,
) -> _Candidate | None:
    try:
        smaller = _smaller_config(backend, bound._normalized_config_copy(config), error)
        if smaller is not None:
            smaller = bound._normalized_config_copy(smaller)
            compiled = bound.compile_config(smaller, allow_print=False)
            contract = original.__globals__.get(_CONTRACT)
            return _Candidate(
                smaller,
                compiled,
                contract is not None
                and contract == compiled.__globals__.get(_CONTRACT),
            )
    except Exception:
        log.debug(
            "%s: not fitting the config to this device",
            bound.kernel.name,
            exc_info=True,
        )
    return None


def _publish_fit(
    backend: TritonBackend,
    bound: BoundKernel[Any],
    original: Config,
    fitted: _Candidate,
    error: tuple[str, int, int],
) -> None:
    if bound.settings.print_output_code:
        # The probes compile silently; set_config prints the code that will run.
        bound._compile_cache.pop(fitted.config, None)
        bound._cache_path_map.pop(fitted.config, None)
    bound.set_config(fitted.config)
    bound._config_from_other_device = False
    original_config = bound._normalized_config_copy(original).config
    changes = ", ".join(
        f"{field}={value!r}"
        for field, value in fitted.config.config.items()
        if original_config.get(field) != value
    )
    report = log.debug if bound.kernel in backend._fitted_kernels else log.warning
    backend._fitted_kernels.add(bound.kernel)
    report(
        "%s: the selected config exceeds this device's %s limit (%d > %d), "
        "so it runs with %s instead. Configs for other shapes are adjusted "
        "the same way. Tuning on this device (for aot_kernel, with "
        "`python -m helion.autotuner.aot_runner`) can find faster ones.",
        bound.kernel.name,
        *error,
        changes,
    )


def fit_for_rendering(fit: FirstLaunchFit, args: Sequence[object]) -> None:
    """Probe Inductor's examples without launching a kernel or changing RNG."""
    from triton.runtime.errors import OutOfResources

    from ...runtime.precompile_shim import _KernelLoaded
    from ...runtime.precompile_shim import compile_only_launcher

    bound = fit.bound_kernel
    candidate = _Candidate(fit.config, fit.compiled)
    error_info = None
    loaded = False

    def load_kernel(*args: object, **kwargs: object) -> None:
        nonlocal loaded
        try:
            cast("Callable[..., object]", compile_only_launcher)(*args, **kwargs)
        except _KernelLoaded:
            loaded = True

    device = bound.env.device
    with torch.random.fork_rng(devices=[device], device_type=device.type):
        for attempt in range(_MAX_DEVICE_FIT_ATTEMPTS + 1):
            loaded = False
            try:
                candidate.compiled(*args, _launcher=load_kernel)
            except OutOfResources as error:
                if error_info is None:
                    error_info = (error.name, error.required, error.limit)
                smaller = (
                    _next_candidate(
                        fit.backend, bound, fit.compiled, candidate.config, error
                    )
                    if attempt < _MAX_DEVICE_FIT_ATTEMPTS
                    else None
                )
                if smaller is None:
                    break
                candidate = smaller
            except Exception:
                log.debug(
                    "%s: not fitting the config to this device",
                    bound.kernel.name,
                    exc_info=True,
                )
                break
            else:
                # Validate every native kernel before publishing. Empty
                # examples and interruptions keep the check armed.
                if loaded:
                    if error_info is not None:
                        _publish_fit(
                            fit.backend, bound, fit.config, candidate, error_info
                        )
                    bound._config_from_other_device = False
                return
    bound._config_from_other_device = False


class FirstLaunchFit:
    """Retry only a failed native launch, keeping each caller's host code single.

    Candidate selection is shared under a lock. Publication requires a full
    host call to return after successfully launching that exact candidate.
    A fresh candidate on reset prevents an in-flight caller publishing it.
    """

    def __init__(
        self,
        backend: TritonBackend,
        bound: BoundKernel[Any],
        config: Config,
        compiled: Callable[..., object],
    ) -> None:
        self.backend, self.bound_kernel = backend, bound
        self.config, self.compiled = config, compiled
        self.current = _Candidate(config, compiled)
        self.lock = threading.Lock()
        self.attempts = 0
        self.first_error: tuple[str, int, int] | None = None

    def __call__(self, *args: object) -> object:
        if torch.compiler.is_compiling():
            return self.compiled(*args)
        launched: list[_Candidate] = []
        result = self.compiled(
            *args, _launcher=functools.partial(self.launch, launched)
        )
        with self.lock:
            if (
                self.bound_kernel._run is self
                and launched
                and launched[-1] is self.current
            ):
                if self.current.compiled is self.compiled:
                    self.bound_kernel._run = self.compiled
                else:
                    assert self.first_error is not None
                    _publish_fit(
                        self.backend,
                        self.bound_kernel,
                        self.config,
                        self.current,
                        self.first_error,
                    )
        return result

    def launch(
        self,
        launched: list[_Candidate],
        triton_kernel: Any,  # noqa: ANN401
        grid: tuple[int, ...],
        *args: object,
        num_warps: int,
        num_stages: int,
        **kwargs: object,
    ) -> object:
        from triton.runtime.errors import OutOfResources

        from ...runtime import default_launcher

        def invoke(candidate: _Candidate) -> object:
            kernel, warps, stages = triton_kernel, num_warps, num_stages
            if candidate.compiled is not self.compiled:
                if candidate.same_host:
                    kernel = candidate.compiled.__globals__[kernel.__name__]
                    warps = self.backend.effective_num_warps(candidate.config)
                stages = candidate.config.num_stages
            result = default_launcher(
                kernel,
                grid,
                *args,
                num_warps=warps,
                num_stages=stages,
                **kwargs,  # pyrefly: ignore [bad-argument-type]
            )
            launched[:] = [candidate]
            return result

        tried = self.current
        if tried.compiled is not self.compiled:
            # Establish this caller's original diagnostic before joining a fit.
            tried = _Candidate(self.config, self.compiled)
        try:
            return invoke(tried)
        except OutOfResources as error:
            if any(kwargs.get(arg) for arg in _STATEFUL_LAUNCH_ARGS):
                raise
            first_error = last_error = error
        if self.first_error is None:
            self.first_error = (
                first_error.name,
                first_error.required,
                first_error.limit,
            )
        try:
            while True:
                with self.lock:
                    if self.current is tried:
                        self.attempts += 1
                        smaller = (
                            _next_candidate(
                                self.backend,
                                self.bound_kernel,
                                self.compiled,
                                tried.config,
                                last_error,
                            )
                            if self.attempts <= _MAX_DEVICE_FIT_ATTEMPTS
                            else None
                        )
                        if smaller is None:
                            raise first_error
                        if not smaller.same_host:
                            original = self.bound_kernel._normalized_config_copy(
                                self.config
                            )
                            if {**smaller.config.config, "num_stages": None} != {
                                **original.config,
                                "num_stages": None,
                            }:
                                raise first_error
                        self.current = smaller
                    tried = self.current
                try:
                    return invoke(tried)
                except OutOfResources as error:
                    last_error = error
        except Exception:
            log.debug(
                "%s: not fitting the config to this device",
                self.bound_kernel.kernel.name,
                exc_info=True,
            )
            with self.lock:
                self.current = _Candidate(self.config, self.compiled)
                self.attempts = _MAX_DEVICE_FIT_ATTEMPTS
            raise first_error from None
        except BaseException:
            with self.lock:
                self.current = _Candidate(self.config, self.compiled)
            raise
        finally:
            # Tracebacks retain this frame and its launch arguments.
            first_error = last_error = None
