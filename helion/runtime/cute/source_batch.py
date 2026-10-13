"""Opt-in speculative CuTe source generation; module import stays in the parent.

Workers never compile or execute a generated kernel. A failed or stale speculative
result is discarded, leaving the original serial compile_config/error path intact.
"""

from __future__ import annotations

import ast
from contextlib import ExitStack
import dataclasses
import enum
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ..._compile_time import measure
from .source_target import TargetMetadata
from .source_threads import prepare_source_fork
from .source_threads import source_threads_supported

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..._compiler.cute.eviction_relevance import PolicyReceipt
    from ..config import Config
    from ..kernel import BoundKernel

log = logging.getLogger(__name__)


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def _config_key(config: Config) -> str:
    return json.dumps(config.config, sort_keys=True, separators=(",", ":"))


def _freeze_metadata(
    value: object, bound: BoundKernel, active: frozenset[int] = frozenset()
) -> object:
    """Freeze the admitted metadata graph; unknown/cyclic objects decline batching."""
    if value is bound:
        return ("bound", id(bound))
    if value is None or type(value) in (bool, int, float, str):
        return (type(value).__name__, value)
    if isinstance(value, (torch.dtype, torch.device, type)):
        return (
            type(value).__name__,
            str(value),
            id(value) if isinstance(value, type) else None,
        )
    if id(value) in active:
        raise TypeError("Cyclic source metadata")
    active = active | {id(value)}

    def freeze(item: object) -> object:
        return _freeze_metadata(item, bound, active)

    if isinstance(value, enum.Enum):
        return (
            type(value).__module__,
            type(value).__qualname__,
            value.name,
            freeze(value.value),
        )
    if type(value) in (tuple, list, torch.Size):
        return (
            type(value).__name__,
            tuple(freeze(item) for item in cast("Sequence[object]", value)),
        )
    if type(value) in (set, frozenset):
        return (
            type(value).__name__,
            tuple(
                sorted(
                    (
                        freeze(item)
                        for item in cast("set[object] | frozenset[object]", value)
                    ),
                    key=str,
                )
            ),
        )
    if type(value) is dict:
        return (
            "dict",
            tuple((freeze(key), freeze(item)) for key, item in value.items()),
        )
    if type(value) is partial:
        return (
            "partial",
            freeze(value.func),
            freeze(value.args),
            freeze(value.keywords),
        )
    if isinstance(value, types.FunctionType):
        return (
            "function",
            id(value),
            id(value.__code__),
            freeze(value.__defaults__),
            freeze(value.__kwdefaults__),
            tuple(freeze(cell.cell_contents) for cell in value.__closure__ or ()),
        )
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, freeze(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    raise TypeError(f"Unsupported source metadata: {type(value).__qualname__}")


def _snapshot(bound: BoundKernel, tensors: Sequence[torch.Tensor]) -> str:
    env = bound.env
    kernel = bound.kernel
    host = bound.host_function
    assert host is not None
    facts = {
        "owners": [
            (
                id(t),
                t.untyped_storage()._cdata,
                t.data_ptr(),
                tuple(t.shape),
                tuple(t.stride()),
                t.storage_offset(),
                str(t.dtype),
                str(t.device),
                t._version,
            )
            for t in tensors
        ],
        "runtime_specializations": {
            key: (
                _freeze_metadata(value.sources, bound),
                _freeze_metadata(value.classifier_identity, bound),
                _freeze_metadata(value.classifier, bound),
                sorted(value.reusable_tensor_properties),
            )
            for key, value in env.runtime_input_specializations.items()
        },
        "bound_runtime_facts": _freeze_metadata(
            env.bound_runtime_input_specialization_results, bound
        ),
        "compiler_facts": _freeze_metadata(
            env.compiler_fact_specialization_facts, bound
        ),
        "specialized_vars": sorted(map(str, env.specialized_vars)),
        "wrapper_plans": _freeze_metadata(env.cute_resolved_wrapper_plans, bound),
        "fission_plan": _freeze_metadata(env.cute_fission_plan, bound),
        "register_tile_disabled": env.cute_register_tile_disabled,
        "base_spec_key": _freeze_metadata(bound._base_spec_key, bound),
        "bound_reset": bound._reset_generation,
        "kernel_reset": kernel._reset_generation,
        "specialization_generation": kernel._specialization_generation,
        "grouped_tail_descriptors": _freeze_metadata(
            kernel._cute_grouped_static_tail_extra_descriptors, bound
        ),
        "specialize_extra": _freeze_metadata(kernel._specialize_extra, bound),
        "compiler_seed_extra": _freeze_metadata(
            kernel._compiler_seed_specialize_extra, bound
        ),
        "extractors": _freeze_metadata(
            bound._compiler_seed_specialization_extractors, bound
        ),
        "host_body": ast.dump(
            ast.Module(body=host.body, type_ignores=[]),
            include_attributes=True,
        ),
        "graphs": [str(g.graph) for g in host.device_ir.graphs],
        "host_imports": {
            key: value.codegen() for key, value in host.global_imports.items()
        },
        "rng_slots": host.rng_seed_slot_count,
        "chained_region": (
            tuple(host.device_ir.root_ids),
            tuple(
                (
                    tuple(phase.roots),
                    tuple(
                        ast.dump(node, include_attributes=True)
                        for node in phase.root_nodes
                    ),
                )
                for phase in host.device_ir.phases
            ),
        )
        if bound.config_spec.cute_chained_matmul_search_enabled
        else None,
        # The search factory is not read by to_code. Observers commonly close
        # over the live search/metrics objects, which change between batches.
        "settings": tuple(
            (field.name, _freeze_metadata(getattr(bound.settings, field.name), bound))
            for field in dataclasses.fields(bound.settings)
            if field.name != "autotuner_fn"
        ),
        "autotuner_fn_identity": id(bound.settings.autotuner_fn),
        "target": (
            bound.config_spec.target_device_capability,
            bound.config_spec.num_sm,
        ),
    }
    return json.dumps(facts, sort_keys=True, default=str)


@dataclasses.dataclass(frozen=True)
class PreparedSource:
    bound: BoundKernel
    config_key: str
    source: str
    source_hash: str
    authority: str
    tensors: tuple[torch.Tensor, ...]
    storages: tuple[object, ...]
    output: str
    target_metadata: TargetMetadata | None = None
    policy_receipt: PolicyReceipt | None = None

    def consume(self, bound: BoundKernel, config: Config) -> str | None:
        if bound is not self.bound or _config_key(config) != self.config_key:
            return None
        if hashlib.sha256(self.source.encode()).hexdigest() != self.source_hash:
            return None
        try:
            if self.target_metadata is not None:
                self.target_metadata.check()
            if _snapshot(bound, self.tensors) != self.authority:
                return None
        except Exception:
            return None
        if self.output:
            sys.stderr.write(self.output)
        return self.source

    def publish_policy_receipt(
        self, bound: BoundKernel, config: Config, compiled: object
    ) -> None:
        """Publish only after the parent successfully materialized this source."""
        if self.policy_receipt is None:
            return
        from ..._compiler.cute.eviction_relevance import authority

        try:
            if (
                bound is not self.bound
                or _config_key(config) != self.config_key
                or self.policy_receipt.config_key != self.config_key
                or hashlib.sha256(self.source.encode()).hexdigest() != self.source_hash
                or self.policy_receipt.source_sha256 != self.source_hash
                or bound.env.backend.generated_source_hash(compiled) != self.source_hash
                or authority(bound) != self.policy_receipt.authority
            ):
                return
            if self.target_metadata is not None:
                self.target_metadata.check()
            bound._cute_eviction_policy_receipts[self.config_key] = self.policy_receipt
        except Exception:
            # Optional search evidence must never fail ordinary materialization.
            return


class _RejectCudaOperations(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = {} if kwargs is None else kwargs
        values = tree_flatten((args, kwargs))[0]
        if any(
            isinstance(v, torch.Tensor)
            and not isinstance(v, FakeTensor)
            and v.device.type == "cuda"
            for v in values
        ):
            raise RuntimeError("CUDA tensor operation in source worker")
        device = kwargs.get("device")
        if (
            device is not None
            and torch.device(cast("torch.device | str | int", device)).type == "cuda"
        ):
            raise RuntimeError("CUDA allocation in source worker")
        return func(*args, **kwargs)


def _guard_targets() -> list[tuple[object, str]]:
    # Resolve optional SDK interfaces in the parent, before forking. A missing
    # interface declines batching; it never silently reduces the guard surface.
    from cuda.bindings import driver
    from cuda.bindings import runtime
    import cutlass
    from cutlass import cute
    from cutlass._mlir import execution_engine
    from cutlass._mlir._mlir_libs._cutlass_ir import (
        _aot_support,  # pyrefly: ignore [missing-module-attribute]
    )
    from cutlass._mlir._mlir_libs._cutlass_ir import (
        _execution_engine,  # pyrefly: ignore [missing-module-attribute]
    )
    from cutlass.base_dsl import compiler
    from cutlass.base_dsl import dsl
    from cutlass.base_dsl import jit_executor
    from cutlass.cutlass_dsl import cuda_jit_executor
    from cutlass.cutlass_dsl import tvm_ffi_provider
    from torch._dynamo.device_interface import CudaInterface
    from torch._inductor.codecache import PyCodeCache

    from ..kernel import BoundKernel

    targets = [
        (m, n)
        for m, prefix in ((driver, "cu"), (runtime, "cuda"))
        for n in dir(m)
        if n.startswith(prefix)
        and n[len(prefix) : len(prefix) + 1].isupper()
        and callable(getattr(m, n))
        and not isinstance(getattr(m, n), type)
    ]
    targets.extend(
        (torch._C, n)
        for n in dir(torch._C)
        if (n.startswith(("_cuda_", "_accelerator_")) or n == "_get_accelerator")
        and callable(getattr(torch._C, n))
    )
    targets.extend(
        (torch.cuda, n)
        for n in (
            "_lazy_init",
            "is_available",
            "current_device",
            "get_device_properties",
            "get_device_capability",
            "get_device_name",
            "device_count",
            "current_stream",
            "synchronize",
            "set_stream",
            "_exchange_device",
            "_maybe_exchange_device",
            "cudart",
            "Stream",
            "ExternalStream",
            "Event",
            "CUDAGraph",
            "graph",
            "stream",
            "device",
        )
    )
    targets.extend(
        (CudaInterface, n)
        for n in (
            "device",
            "Event",
            "Stream",
            "set_device",
            "stream",
            "current_stream",
            "set_stream",
            "_set_stream_by_id",
            "synchronize",
            "get_raw_stream",
            "exchange_device",
            "maybe_exchange_device",
            "memory_allocated",
            "is_bf16_supported",
        )
    )
    targets.append((CudaInterface.Worker, "set_device"))
    targets.extend(
        [
            (type(cute.compile), "to_precompiled_mlir"),
            (type(cute.compile), "_compile"),
            (cute, "compile"),
            (compiler.Compiler, "compile"),
            (compiler.Compiler, "jit"),
            (compiler.Compiler, "compile_and_jit"),
            (dsl.BaseDSL, "compile_and_jit"),
            (dsl.BaseDSL, "_lookup_jit_entry"),
            (execution_engine, "ExecutionEngine"),
            (_execution_engine, "BinaryExecutionEngine"),
            (jit_executor, "load_kernels_from_ir_module"),
            (jit_executor.JitCompiledFunction, "to"),
            (jit_executor.JitCompiledFunction, "get_aux_func"),
            (jit_executor.JitCompiledFunction, "run_compiled_program"),
            (jit_executor.JitCompiledFunction, "__call__"),
            (jit_executor.JitExecutor, "__call__"),
            (cuda_jit_executor.CudaDialectJitCompiledFunction, "to"),
            (cuda_jit_executor.CudaDialectJitCompiledFunction, "_deserializer"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionBase, "to"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionBase, "run_compiled_program"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunction, "__call__"),
            (tvm_ffi_provider.TVMFFIJitCompiledFunctionWithKwargs, "__call__"),
            (_aot_support, "export_module_to_bytes"),
            (BoundKernel, "compile_config"),
            (BoundKernel, "_compile_config"),
            (BoundKernel, "autotune"),
            (PyCodeCache, "write"),
            (PyCodeCache, "load"),
            (PyCodeCache, "load_by_key_path"),
            (subprocess, "Popen"),
            (os, "fork"),
            (os, "posix_spawn"),
            (os, "posix_spawnp"),
            (os, "system"),
        ]
    )
    # Some SDK versions expose this extra alias; current SDK exposes only
    # cute.compile. Do not require or manufacture an absent optional entry point.
    if hasattr(cutlass, "compile"):
        targets.append((cutlass, "compile"))
    for owner, name in targets:
        if not callable(getattr(owner, name)):
            raise TypeError(f"Missing source worker guard: {name}")
    return targets


def _install_guards(targets: Sequence[tuple[object, str]], calls: list[str]) -> None:
    for owner, name in targets:

        def reject(*args: object, _label: str = name, **kwargs: object) -> NoReturn:
            calls.append(_label)
            raise RuntimeError("Forbidden source worker operation: " + _label)

        setattr(owner, name, reject)


def _run_workers(
    indices: list[int],
    child: Callable[[int], int],
    *,
    output: Path,
    worker_timeout: float,
    max_workers: int = 2,
    budget_exceeded: Callable[[], bool] = lambda: False,
) -> list[dict[str, int]]:
    """Own, time-bound and reap every fork before returning any source batch."""
    assert worker_timeout > 0 and max_workers > 0
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("Source workers require exclusive child reaping")
    if not prepare_source_fork():
        raise RuntimeError("Source workers require a quiescent main thread")
    pending = list(indices)
    active: dict[int, tuple[int, float]] = {}
    reaps: list[dict[str, int]] = []
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)

    def interrupted(signum: int, frame: object) -> NoReturn:
        raise SystemExit(128 + signum)

    def collect(pid: int, status: int) -> int:
        index, _started = active.pop(pid)
        exitcode = os.waitstatus_to_exitcode(status)
        reaps.append({"index": index, "pid": pid, "exitcode": exitcode})
        return exitcode

    def poll() -> list[int]:
        codes = []
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}
        )
        try:
            for pid in tuple(active):
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    assert waited == pid
                    codes.append(collect(pid, status))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return codes

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or active:
            while pending and len(active) < max_workers:
                if budget_exceeded():
                    pending.clear()
                    break
                index = pending.pop(0)
                sys.stdout.flush()
                sys.stderr.flush()
                # Block termination across fork/ownership registration so a signal
                # cannot leave a successfully forked child absent from active.
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT}
                )
                try:
                    if threading.enumerate() != [threading.main_thread()]:
                        raise RuntimeError("Source parent acquired another thread")
                    pid = os.fork()
                    if pid == 0:
                        signal.signal(signal.SIGTERM, signal.SIG_DFL)
                        signal.signal(signal.SIGINT, signal.SIG_DFL)
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                        exitcode = 1
                        try:
                            exitcode = child(index)
                        except BaseException:
                            (output / f"{index}-worker-error.txt").write_text(
                                traceback.format_exc()
                            )
                        finally:
                            # No inherited CUDA/Python tensor finalization.
                            os._exit(exitcode)
                    active[pid] = (index, time.monotonic())
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            codes = poll()
            _write(output / "reaps.json", reaps)
            if any(code != 0 for code in codes):
                raise RuntimeError("Source worker failed; no module materialization")
            now = time.monotonic()
            overdue = [
                index
                for index, started in active.values()
                if now - started > worker_timeout
            ]
            if overdue:
                raise TimeoutError(f"Source workers exceeded timeout: {overdue}")
            if active:
                time.sleep(0.05)
        assert all(row["exitcode"] == 0 for row in reaps)
        return reaps
    finally:
        # A second ordinary termination signal must not interrupt cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            poll()
            for pid in active:
                os.kill(pid, signal.SIGTERM)
            grace_deadline = time.monotonic() + 1.0
            while active and time.monotonic() < grace_deadline:
                poll()
                if active:
                    time.sleep(0.05)
            for pid in active:
                os.kill(pid, signal.SIGKILL)
            # Only unreaped children are addressed; their PID cannot be reused.
            # After SIGKILL, reap each exact PID (never an unrelated process).
            for pid in tuple(active):
                waited, status = os.waitpid(pid, 0)
                assert waited == pid
                collect(pid, status)
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)
            _write(output / "reaps.json", reaps)


def _plain_chained_region(bound: BoundKernel) -> bool:
    host = bound.host_function
    return (
        host is not None
        and len(host.device_ir.root_ids) == 1
        and len(host.device_ir.phases) == 1
        and host.rng_seed_slot_count == 0
        and not bound.env.cute_resolved_wrapper_plans
        and bound.env.cute_fission_plan is None
    )


class SourceBatch:
    def __init__(
        self,
        bound: BoundKernel,
        tensors: tuple[torch.Tensor, ...],
        targets: list[tuple[object, str]],
        timeout: float,
    ) -> None:
        self.bound = bound
        self.tensors = tensors
        self.storages = tuple(t.untyped_storage() for t in tensors)
        self.targets = targets
        self.timeout = timeout
        self.workers = bound.settings.autotune_cute_source_workers
        # Resolve only original FakeTensor analysis metadata in the parent.
        # Never override the child's genuine CUDA initialized/bad-fork state.
        self.avoid_device_init = bound.env.fake_mode.avoid_device_init

    @classmethod
    def create(cls, kernel: object, args: Sequence[object]) -> SourceBatch | None:
        from ..kernel import BoundKernel

        if type(kernel) is not BoundKernel:
            return None
        if (
            sys.platform != "linux"
            or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
            or threading.current_thread() is not threading.main_thread()
            or not source_threads_supported()
            or kernel.settings.backend != "cute"
            or not kernel.settings.static_shapes
            or kernel.env.process_group_name is not None
            or kernel.env.cute_fission_plan is not None
            or kernel.config_spec.num_sm <= 0
            or kernel.settings.print_output_code
            or kernel.settings.autotune_compile_timeout <= 0
        ):
            return None
        # Content-dependent runtime classifiers and arbitrary subclasses are not
        # covered by source-only metadata inheritance. Existing metadata proofs
        # retain their exact sources/identity/results in the per-attempt snapshot.
        if any(
            not s.reusable_tensor_properties
            or not s.reusable_tensor_properties <= {"data_ptr", "storage_span"}
            for s in kernel.env.runtime_input_specializations.values()
        ):
            return None
        tensors = tuple(t for t in tree_flatten(args)[0] if isinstance(t, torch.Tensor))
        if any(type(t) is not torch.Tensor for t in tensors):
            return None
        live = tuple(ref() for ref in kernel._runtime_tensor_refs_by_name.values())
        if any(
            t is None or not any(t is candidate for candidate in tensors) for t in live
        ):
            return None
        try:
            if (
                kernel.config_spec.cute_chained_matmul_search_enabled
                and not _plain_chained_region(kernel)
            ):
                return None
            _snapshot(kernel, tensors)
            targets = _guard_targets()
            return cls(
                kernel, tensors, targets, kernel.settings.autotune_compile_timeout
            )
        except Exception:
            log.debug(
                "CuTe source batching unavailable; using serial compilation",
                exc_info=True,
            )
            return None

    def prepare(
        self, configs: Sequence[Config], budget_exceeded: Callable[[], bool]
    ) -> dict[int, PreparedSource]:
        # This parent wall category includes fork, IPC, validation and drain.
        # Child to_code/CPU timings are not summed into parent wall time.
        with measure("CuTe.source_batch_wall"):
            try:
                return self._prepare(configs, budget_exceeded)
            except Exception:
                log.debug(
                    "Speculative CuTe source batch failed; using serial compilation",
                    exc_info=True,
                )
                return {}

    def _prepare(
        self, configs: Sequence[Config], budget_exceeded: Callable[[], bool]
    ) -> dict[int, PreparedSource]:
        bound = self.bound
        try:
            normalized = [bound._normalized_config_copy(c) for c in configs]
            # Normalization is the configuration authority. Advanced lowering
            # choices do not bypass the single-region/owner/worker guards, and
            # an unsupported worker path still returns to serial compilation.
            if (
                bound.config_spec.cute_chained_matmul_search_enabled
                and not _plain_chained_region(bound)
            ):
                return {}
            authority = _snapshot(bound, self.tensors)
            target_metadata = (
                TargetMetadata.capture(torch, bound.env.device)
                if bound.env.device.type == "cuda"
                else None
            )
        except Exception:
            # In particular, invalid configs must raise in the original compile
            # loop at their original index, not escape from speculative setup.
            return {}
        indices = [i for i, c in enumerate(normalized) if c not in bound._compile_cache]
        if not indices or budget_exceeded():
            return {}
        with tempfile.TemporaryDirectory(prefix="helion-cute-source-") as temporary:
            output = Path(temporary)

            def child(index: int) -> int:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                calls: list[str] = []
                _install_guards(self.targets, calls)
                # Cached original analysis property, not CUDA process state.
                FakeTensorMode.avoid_device_init = property(  # pyrefly: ignore [bad-assignment]
                    lambda _mode: self.avoid_device_init
                )
                with (output / f"{index}.log").open("w") as stream:
                    os.dup2(stream.fileno(), 1)
                    os.dup2(stream.fileno(), 2)
                    with ExitStack() as metadata_stack, _RejectCudaOperations():
                        if target_metadata is not None:
                            target_metadata.install(metadata_stack)
                        from ..._compiler.cute.eviction_relevance import export_receipt

                        # A fork inherits parent evidence. Only fresh generation
                        # in this child may attach evidence to its returned source.
                        bound._cute_eviction_policy_receipts.pop(
                            _config_key(normalized[index]), None
                        )
                        source = None
                        source_error = None
                        policy_receipt = None
                        try:
                            source = bound.to_triton_code(
                                normalized[index], emit_repro_caller=False
                            )
                        except Exception:
                            # A source-invalid candidate does not invalidate its
                            # siblings, but only after all original authority and
                            # forbidden-operation checks below still succeed.
                            source_error = traceback.format_exc()
                        if source_error is None:
                            assert source is not None
                            try:
                                policy_receipt = export_receipt(
                                    bound, normalized[index], source
                                )
                            except Exception:
                                policy_receipt = None
                        if target_metadata is not None:
                            target_metadata.check()
                        if calls or _snapshot(bound, self.tensors) != authority:
                            raise RuntimeError(
                                "Source worker changed retained authority"
                            )
                        if _config_key(
                            bound._normalized_config_copy(configs[index])
                        ) != _config_key(normalized[index]):
                            raise RuntimeError(
                                "Source worker changed normalized config"
                            )
                    if source_error is not None:
                        _write(
                            output / f"{index}.json",
                            {
                                "config": _config_key(normalized[index]),
                                "status": "source_error",
                                "error": source_error,
                            },
                        )
                    else:
                        assert source is not None
                        (output / f"{index}.py").write_text(source)
                        _write(
                            output / f"{index}.json",
                            {
                                "config": _config_key(normalized[index]),
                                "status": "success",
                                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                                "policy_receipt": policy_receipt,
                            },
                        )
                    sys.stdout.flush()
                    sys.stderr.flush()
                return 0

            try:
                reaps = _run_workers(
                    indices,
                    child,
                    output=output,
                    worker_timeout=self.timeout,
                    max_workers=self.workers,
                    budget_exceeded=budget_exceeded,
                )
            except Exception:
                # Drain completed before returning. Repeat through compile_config
                # so callers see the original exception type/context/logging.
                log.debug(
                    "Speculative CuTe source batch failed; using serial compilation",
                    exc_info=True,
                )
                return {}
            if target_metadata is not None:
                target_metadata.check()
            if _snapshot(bound, self.tensors) != authority:
                return {}
            prepared = {}
            for reap in reaps:
                index = reap["index"]
                record = json.loads((output / f"{index}.json").read_text())
                if record["config"] != _config_key(normalized[index]):
                    continue
                if record.get("status") == "source_error":
                    # All children have drained and parent authority matched.
                    # Leave this exact original index to serial compile_config.
                    continue
                if record.get("status", "success") != "success":
                    raise RuntimeError("Unknown source worker result status")
                source = (output / f"{index}.py").read_text()
                if record["sha256"] != hashlib.sha256(source.encode()).hexdigest():
                    continue
                from ..._compiler.cute.eviction_relevance import import_receipt

                try:
                    policy_receipt = import_receipt(
                        bound, normalized[index], source, record.get("policy_receipt")
                    )
                except Exception:
                    policy_receipt = None
                prepared[index] = PreparedSource(
                    bound,
                    record["config"],
                    source,
                    record["sha256"],
                    authority,
                    self.tensors,
                    self.storages,
                    (output / f"{index}.log").read_text(),
                    target_metadata,
                    policy_receipt,
                )
            return prepared
