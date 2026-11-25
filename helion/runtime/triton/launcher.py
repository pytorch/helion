"""Helion-dependency-free runtime launch helpers for the Triton backend.

This module holds the small set of runtime symbols that Helion's *generated*
Triton code depends on at execution time:

* :func:`default_launcher` -- invokes a compiled ``triton.jit`` kernel.
* :func:`get_num_sm` -- persistent-kernel grid size (host statement).
* :func:`set_triton_allocator` -- installs the scratch allocator used by TMA /
  tensor-descriptor kernels (device-function prefix statement).

It depends only on ``torch`` and ``triton`` -- no other ``helion`` module -- so
the ahead-of-time precompiler can bulk-export this file verbatim into a
standalone kernel with zero Helion runtime dependency.

Helion-specific behavior that is only meaningful in-process (translating
Triton's opaque shape errors into :class:`helion.exc.ShapeMismatch`, and the
CPU/TPU cases of :func:`get_num_sm`) lives in thin wrappers in
:mod:`helion.runtime`, not here.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import math
import threading
from typing import TYPE_CHECKING
from typing import cast
import weakref

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from types import ModuleType

try:
    import triton
except ImportError:
    triton = None  # type: ignore[assignment]


# (Triton driver, device of its scratch allocations): querying the driver's
# target reads device properties, too slow to repeat on every kernel launch.
# Top level, so the standalone runtime shim rewrites its ``global`` to
# ``nonlocal``.
_alloc_device: tuple[object, torch.device] | None = None


if triton is not None:

    def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        global _alloc_device
        driver = triton.runtime.driver.active
        cached = _alloc_device
        if cached is None or cached[0] is not driver:
            # Dynamically get device from Triton backend
            current_target = driver.get_current_target()
            if current_target is None:
                raise RuntimeError("No active Triton target available")
            # No device index: allocate on the current device, where Triton
            # launches.
            cached = _alloc_device = (driver, torch.device(current_target.backend))
        return torch.empty(size, device=cached[1], dtype=torch.int8)

    def set_triton_allocator() -> None:
        try:
            from triton import set_allocator
            from triton.runtime._allocation import NullAllocator
            from triton.runtime._allocation import _allocator
        except ImportError:
            return
        if isinstance(_allocator, contextvars.ContextVar):
            existing = _allocator.get()
        else:  # older versions of Triton
            existing = _allocator
        # if allocator isn't NullAllocator, we assume it is set by the user
        if isinstance(existing, NullAllocator):
            set_allocator(_alloc_fn)

else:

    def set_triton_allocator() -> None:  # type: ignore[misc]
        pass


def get_num_sm(
    device: torch.device, *, reserved_sms: int = 0, cluster_size: int = 1
) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified GPU.

    Args:
        device: Device to query. Must be a GPU device (``cuda``/``xpu``/``mps``/
            ``mtia``); CPU/TPU handling lives in :func:`helion.runtime.get_num_sm`.
        reserved_sms: Number of SMs to keep free for other work (e.g., communication
            kernels). Defaults to 0 meaning all device SMs are available to Helion.
        cluster_size: NVIDIA CUDA only: the SMs whole clusters of this many
            CTAs (one an SM) keep resident (``get_cluster_capacity``) less the
            reserved ones bound the count, rounded down to a multiple of it
            (a kernel launched in clusters).

    Returns:
        Grid size to use for a persistent kernel on the device after accounting
        for any reserved SMs. Always at least 1, and at least one whole
        cluster.
    """
    available_sms: int
    assert device.type in [
        "cuda",
        "xpu",
        "mtia",
        "mps",
    ], "TODO: implement for other devices"
    check_cluster_device(device, cluster_size)
    if device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mps":
        available_sms = torch.backends.mps.get_core_count()
    elif device.type == "mtia":
        device_props = torch.mtia.get_device_properties(device.index)
        if "max_grid_height" in device_props and "max_grid_width" in device_props:
            available_sms = (
                device_props["max_grid_height"] * device_props["max_grid_width"]
            )
        else:
            raise RuntimeError(
                f"Unable to determine SM count for MTIA device. "
                f"Available properties: {list(device_props.keys())}"
            )
    else:
        raise NotImplementedError(
            f"get_num_sm not implemented for device type: {device.type}"
        )

    sms = available_sms if reserved_sms <= 0 else max(available_sms - reserved_sms, 1)
    if cluster_size > 1:
        capacity = get_cluster_capacity(device, cluster_size)
        sms = max(
            cluster_workers(sms, capacity, cluster_size, reserved_sms), cluster_size
        )
    return sms


def cluster_workers(
    sms: int, capacity: int, cluster_size: int, reserved_sms: int
) -> int:
    """The SMs a launch in clusters of ``cluster_size`` takes: at most ``sms``
    and the ``capacity`` such clusters keep resident (``get_cluster_capacity``)
    less the reserved SMs, rounded down to whole clusters (0 when none fits).
    The host's grid (``get_num_sm``) and the compiler's static schedule
    (``dsmem_worker_count``) both take it, so they agree."""
    return min(sms, capacity - max(reserved_sms, 0)) // cluster_size * cluster_size


# CUs per XCD by base CDNA architecture.  Used to derive the live,
# partition-visible XCD count from the observed CU count (see get_num_xcd).
_CUS_PER_XCD: dict[str, int] = {
    "gfx942": 38,  # CDNA3 (MI300)
    "gfx950": 32,  # CDNA4 (MI350)
    "gfx951": 32,  # CDNA4 (MI355)
}


def get_num_xcd(device: torch.device | int | None = None) -> int:
    """Number of XCDs visible for ``device`` on AMD CDNA, else ``1``.

    Derived from the live, partition-visible compute-unit count rather than the
    architecture name, so MI300A (6 XCDs) and compute-partition modes such as CPX
    (which expose a single XCD) are handled correctly.  Returns ``1`` -- which
    disables xcd_remap -- for unknown architectures or a CU count that does not
    look like an integer number of XCDs.
    """
    if not torch.cuda.is_available():
        return 1
    try:
        props = torch.cuda.get_device_properties(
            device if device is not None else torch.cuda.current_device()
        )
    except Exception:
        return 1
    arch = getattr(props, "gcnArchName", None)
    if not arch:
        return 1
    cus_per_xcd = _CUS_PER_XCD.get(arch.split(":")[0])
    if cus_per_xcd is None:
        return 1
    cu_count = props.multi_processor_count
    num_xcd = round(cu_count / cus_per_xcd)
    # Tolerate harvested parts, but bail out (return 1) if the live CU count does
    # not look like an integer number of XCDs.
    if num_xcd < 1 or abs(num_xcd * cus_per_xcd - cu_count) > cus_per_xcd // 4:
        return 1
    return num_xcd


def default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    _remote_copy_signal_dst: torch.Tensor | None = None,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_copy_process_group_name: str | None = None,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_barrier_process_group_name: str | None = None,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    _persistent_state_process_group_name: str | None = None,
    _persistent_state_rank_digest: str | None = None,
    _persistent_state_rank_shares: tuple[int, ...] = (),
    _timeline: tuple[int, object] | None = None,
    _timeline_register: Callable[[torch.Tensor], object] | None = None,
    _minimum_resident_programs: int = 0,
    _cluster_size: int = 1,
    _tensor_memory: bool = True,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    if _remote_copy_signal_slots_per_program:
        if _remote_copy_signal_dst is None or _remote_copy_process_group_name is None:
            raise RuntimeError(
                "remote-copy completion storage requires a symmetric destination "
                "and process group"
            )
        signal = _get_remote_copy_signal(
            triton_kernel,
            _remote_copy_signal_dst,
            _remote_copy_process_group_name,
            math.prod(grid) * _remote_copy_signal_slots_per_program,
        )
        # Allocation zeroes new pads and receive waits reset consumed slots.
        # Clearing here could erase a completion sent before this rank launches.
        args = (*args, signal)
    if _remote_barrier_signal_slots_per_program:
        if _remote_barrier_process_group_name is None:
            raise RuntimeError(
                "remote-barrier completion storage requires a process group"
            )
        signal = _get_remote_barrier_signal(
            triton_kernel,
            _remote_barrier_process_group_name,
            math.prod(grid) * _remote_barrier_signal_slots_per_program,
        )
        args = (*args, signal)
    for slot, (scratch_like, numel_per_program) in enumerate(
        _remote_copy_scratch_specs
    ):
        scratch = _get_remote_copy_scratch(
            triton_kernel,
            scratch_like,
            slot,
            math.prod(grid) * numel_per_program,
        )
        args = (*args, scratch)
    if _persistent_state_specs:
        persistent_state_namespace = (
            tuple(grid),
            num_warps,
            num_stages,
            ptx_options,
            launch_cooperative_grid,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
            tuple(spec[1:] for spec in _persistent_state_specs),
            # Symmetric state exchanges the namespace, so ranks compare digests.
            _persistent_state_rank_digest,
        )
        for slot, (state_like, numel, dtype, symmetric) in enumerate(
            _persistent_state_specs
        ):
            if _persistent_state_process_group_name is None:
                state_args = (
                    _get_persistent_state(
                        triton_kernel,
                        state_like,
                        persistent_state_namespace,
                        slot,
                        numel,
                        dtype,
                    ),
                )
            else:
                state_args = _get_process_group_state(
                    triton_kernel,
                    state_like,
                    persistent_state_namespace,
                    slot,
                    numel,
                    dtype,
                    symmetric=symmetric,
                    process_group_name=_persistent_state_process_group_name,
                    shares=_persistent_state_rank_shares,
                )
            # A megakernel_timeline launch: ``_timeline`` is (its state slot, the
            # compiler's metadata); helion.runtime's launcher also passes
            # ``_timeline_register``, which takes the slot's buffer (standalone
            # code records no timeline).
            if (
                _timeline is not None
                and _timeline_register is not None
                and slot == _timeline[0]
            ):
                _timeline_register(cast("torch.Tensor", state_args[0]))
            args = (*args, *state_args)
    # For both CUDA and MTIA, use the same kernel execution.
    run_kwargs: dict = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    # Launches whose CTAs wait on each other: a resident cohort, clusters, or
    # a cooperative grid (a grid barrier), which only tensor memory patches
    # (``_tensor_memory``: False where codegen finds the kernel takes none).
    if (
        _minimum_resident_programs
        or _cluster_size > 1
        or (launch_cooperative_grid and _tensor_memory and _has_tensor_memory())
    ):
        # Another thread's launch of the same kernel must neither see this
        # launch's metadata (its cluster size, its shared memory) nor restore
        # its own under this launch, nor compile the specialization again
        # (replacing the cached kernel this launch patched) between the patch
        # and the launch.
        with _patched_launch_lock(triton_kernel):
            # ``triton_kernel`` is a JITFunction.  Resource information belongs
            # to its exact compiled specialization, so compile (but do not
            # launch) that specialization before asking CUDA for its occupancy.
            compiled_kernel = triton_kernel.run(  # type: ignore[union-attr]
                *args,
                **{**run_kwargs, "warmup": True},
            )
            if compiled_kernel is None:
                raise RuntimeError("unable to compile the launched kernel")
            # Load the module and its function handle, which the driver
            # queries below take, without launching; a kernel that failed to
            # load raises its error again.
            compiled_kernel._init_handles()  # type: ignore[union-attr]
            metadata = compiled_kernel.packed_metadata
            try:
                if _cluster_size > 1:
                    run_kwargs["grid"] = _cluster_grid(
                        compiled_kernel, grid, _cluster_size
                    )
                # A cohort's, a cluster's or a cooperative grid's CTAs wait on
                # each other.
                _fit_tensor_memory(compiled_kernel)
                if _minimum_resident_programs or _cluster_size > 1:
                    # At least one whole cluster: a cluster the driver cannot
                    # place is refused here, not by Triton's launch.
                    _validate_resident_program_capacity(
                        compiled_kernel,
                        args,
                        required_programs=max(
                            _minimum_resident_programs, _cluster_size
                        ),
                        cluster_size=_cluster_size,
                    )
                return triton_kernel.run(  # type: ignore[union-attr]
                    *args,
                    **run_kwargs,
                )
            finally:
                # Restore what Triton compiled, which the checks above read
                # (Triton reads the metadata as it launches).
                compiled_kernel.packed_metadata = metadata  # type: ignore[union-attr]
    return triton_kernel.run(  # type: ignore[union-attr]
        *args,
        **run_kwargs,
    )


def _patched_launch_lock(triton_kernel: object) -> threading.Lock:
    """The lock a launch of ``triton_kernel`` in clusters, as a resident
    cohort or as a cooperative grid holds from its compile and metadata patch
    until its restore: one a kernel, so a cold compile holds up only launches
    of that kernel.  (The precompiler keeps a kernel already cached:
    ``precompile_shim.make_precompiler``.)  Every launch of a generated
    kernel passes the same launcher arguments, so all of them take it."""
    state = vars(triton_kernel)
    lock = state.get("_helion_patched_launch_lock")
    if lock is None:
        # Two first launches on separate threads keep the one stored first.
        lock = state.setdefault("_helion_patched_launch_lock", threading.Lock())
    return lock


# How a config's launch in clusters that cannot run starts its error (the
# default launcher raises ``helion.exc.InvalidConfig`` for it, which the
# autotuner skips).
CLUSTER_LAUNCH_REFUSED = "A launch in clusters cannot run: "

# The largest portable cluster; larger ones need an opt-in
# (``CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED``) that the occupancy
# queries here do not make.
MAX_CLUSTER_SIZE = 8

# Likewise for a launch that no shared memory padding keeps from placing more
# CTAs on an SM than their tensor memory holds (``_fit_tensor_memory``).
TENSOR_MEMORY_REFUSED = "A launch cannot fit its CTAs' tensor memory: "

# The tensor memory columns of a Blackwell SM.
TENSOR_MEMORY_COLUMNS = 512


def _cluster_grid(
    compiled_kernel: object, grid: tuple[int, ...], cluster_size: int
) -> tuple[int, ...]:
    """The grid of clusters launching ``grid`` programs in clusters of
    ``cluster_size`` along x.  Triton's launcher takes the cluster size from the
    compiled kernel's packed metadata (``num_ctas``); a single-CTA module keeps
    its program ids (``blockIdx.x``) and only gains co-scheduled cluster peers.
    A launch that cannot run raises ``CLUSTER_LAUNCH_REFUSED``."""
    if cluster_size > MAX_CLUSTER_SIZE:
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}clusters of {cluster_size} are not "
            f"portable (at most {MAX_CLUSTER_SIZE})"
        )
    if grid[0] % cluster_size:
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}a grid of {grid} does not divide into "
            f"clusters of {cluster_size}"
        )
    # Builds may pack more after these three (fbtriton: a preferred cluster
    # shape); they pass through.
    num_warps, num_ctas, shared, *rest = compiled_kernel.packed_metadata  # type: ignore[attr-defined]
    if num_ctas != 1:
        # A multi-CTA module's program id is its cluster's, not the CTA's.
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}a {num_ctas}-CTA kernel cannot launch in "
            f"clusters of {cluster_size}"
        )
    metadata = compiled_kernel.metadata  # type: ignore[attr-defined]
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        # Triton sizes per-CTA scratch by the grid it launches (of clusters).
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}a kernel with scratch memory cannot launch "
            "in clusters"
        )
    # Triton opts in to more than the default dynamic shared memory only past
    # 48 KB of it, not counting a static cluster slab beside it.
    error = _opt_in_shared(compiled_kernel, shared)
    if error is not None:
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}{shared} bytes of dynamic shared memory "
            f"beside the static cluster slab exceed the SM's: {error}"
        )
    compiled_kernel.packed_metadata = (num_warps, cluster_size, shared, *rest)  # type: ignore[attr-defined]
    return (grid[0] // cluster_size, *grid[1:])


def _opt_in_shared(compiled_kernel: object, shared: int) -> object:
    """Let launches of ``compiled_kernel`` take ``shared`` bytes of dynamic
    shared memory: the driver's error, None once done.  The attribute is one
    limit, so it only rises: a launch asking for less keeps a larger one (set
    once a specialization: the launcher also runs during graph capture)."""
    if shared > vars(compiled_kernel).get("_helion_shared_optin", -1):
        cuda_driver = _cuda_driver()
        (error,) = cuda_driver.cuFuncSetAttribute(
            cuda_driver.CUfunction(int(compiled_kernel.function)),  # type: ignore[attr-defined]
            cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared,
        )
        if error != cuda_driver.CUresult.CUDA_SUCCESS:
            return error
        vars(compiled_kernel)["_helion_shared_optin"] = shared
    return None


def _fit_tensor_memory(compiled_kernel: object) -> None:
    """Pad a launch of ``compiled_kernel`` (its packed metadata's dynamic
    shared memory) so an SM fits no more of its CTAs than their tensor memory
    holds (``_tensor_memory_ctas``).  The block scheduler may give an SM as
    many CTAs as occupancy allows whatever the grid (other streams' CTAs, or a
    dependent launch's predecessor's, hold some SMs), and one past the tensor
    memory blocks in its allocation: a launch whose CTAs wait on each other's
    work (a resident cohort, a cluster's peers, a cooperative grid) would
    hang.  This holds one launch's CTAs to what their columns allow; CTAs of
    two such launches overlapping on separate streams, launches in clusters
    included, may still share an SM past its columns and deadlock when their
    column counts differ (a known limit: closing it needs a process-wide
    policy)."""
    fit = _tensor_memory_ctas(compiled_kernel.metadata)  # type: ignore[attr-defined]
    if fit is None:
        return
    num_warps, num_ctas, shared, *rest = compiled_kernel.packed_metadata  # type: ignore[attr-defined]
    # Once a specialization: the launcher also runs during graph capture.
    cache = vars(compiled_kernel).setdefault("_helion_tensor_memory_shared", {})
    if (num_warps, shared) not in cache:
        cache[num_warps, shared] = _tensor_memory_shared(
            compiled_kernel, num_warps, shared, fit
        )
    padded = cache[num_warps, shared]
    compiled_kernel.packed_metadata = (num_warps, num_ctas, padded, *rest)  # type: ignore[attr-defined]


def _tensor_memory_shared(
    compiled_kernel: object, num_warps: int, shared: int, fit: int
) -> int:
    """The least dynamic shared memory a CTA of ``compiled_kernel`` takes so
    that an SM holds at most ``fit`` of them (``_fit_tensor_memory``).  The
    kernel's own occupancy cannot tell: CUDA reports one CTA an SM for a
    kernel using tcgen05 (B200, driver 580), while the block scheduler places
    as many as its registers, threads and shared memory allow (9 CTAs an SM
    of a 32-column kernel, measured).  So bisect on the occupancy of an empty
    kernel of as many threads taking the same shared memory (static and
    dynamic): it counts the driver's shared memory model (allocation
    granularity, what the SM reserves a CTA), and its few registers fit at
    least as many CTAs as the kernel's."""
    cuda_driver = _cuda_driver()
    success = cuda_driver.CUresult.CUDA_SUCCESS
    error, static = cuda_driver.cuFuncGetAttribute(
        cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        cuda_driver.CUfunction(int(compiled_kernel.function)),  # type: ignore[attr-defined]
    )
    if error != success:
        raise RuntimeError(f"unable to read the static shared memory: {error}")
    device = torch.device("cuda", torch.cuda.current_device())
    # The most a CTA may take beside its static shared memory.
    properties = torch.cuda.get_device_properties(device)
    optin = int(properties.shared_memory_per_block_optin)  # pyrefly: ignore [missing-attribute]
    most = optin - int(static)
    with _empty_function(device) as function:

        def blocks(dynamic: int) -> int:
            error, count = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                function, num_warps * 32, int(static) + dynamic
            )
            if error != success:
                raise RuntimeError(f"CUDA occupancy query failed: {error}")
            return int(count)

        if blocks(shared) <= fit:
            return shared
        if blocks(most) > fit:
            raise RuntimeError(
                f"{TENSOR_MEMORY_REFUSED}{most} bytes of dynamic shared memory "
                f"still put more than {fit} CTAs on an SM"
            )
        # More than ``fit`` CTAs an SM at ``low``, at most ``fit`` at ``high``.
        low, high = shared, most
        while high - low > 1:
            middle = (low + high) // 2
            if blocks(middle) > fit:
                low = middle
            else:
                high = middle
    error = _opt_in_shared(compiled_kernel, high)
    if error is not None:
        raise RuntimeError(
            f"{TENSOR_MEMORY_REFUSED}unable to allow {high} bytes of dynamic "
            f"shared memory: {error}"
        )
    return high


# A kernel that does nothing, for asking CUDA what fits.
_EMPTY_PTX = b""".version 7.8
.target sm_90
.address_size 64
.visible .entry helion_empty() { ret; }
\0"""


def _cuda_driver() -> ModuleType:
    """``cuda.bindings.driver``, which the cluster, cohort and tensor memory
    queries take."""
    try:
        return importlib.import_module("cuda.bindings.driver")
    except ImportError as error:
        raise RuntimeError(
            "launches whose CTAs wait on each other on this GPU need the "
            "cuda-bindings package (pip install cuda-bindings)"
        ) from error


@contextlib.contextmanager
def _empty_function(device: torch.device) -> Iterator[object]:
    """``_EMPTY_PTX``'s function on ``device``, allowed all the dynamic shared
    memory a CTA may opt in to (unloaded after)."""
    cuda_driver = _cuda_driver()
    success = cuda_driver.CUresult.CUDA_SUCCESS
    with torch.cuda.device(device):
        # The driver calls below need the device's primary context, which
        # entering the device does not create (nor does torch.cuda.init());
        # asking for its current stream does.
        torch.cuda.current_stream(device)
        error, module = cuda_driver.cuModuleLoadData(_EMPTY_PTX)
        if error != success:
            raise RuntimeError(f"unable to load a probe kernel: {error}")
        try:
            error, function = cuda_driver.cuModuleGetFunction(module, b"helion_empty")
            if error != success:
                raise RuntimeError(f"unable to find the probe kernel: {error}")
            properties = torch.cuda.get_device_properties(device)
            (error,) = cuda_driver.cuFuncSetAttribute(
                function,
                cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                properties.shared_memory_per_block_optin,  # pyrefly: ignore [missing-attribute]
            )
            if error != success:
                raise RuntimeError(f"unable to size the probe kernel: {error}")
            yield function
        finally:
            cuda_driver.cuModuleUnload(module)


def check_cluster_device(device: torch.device, cluster_size: int) -> None:
    """Clusters of more than one CTA run on NVIDIA CUDA devices of sm_90 and
    up only."""
    if cluster_size <= 1:
        return
    if device.type != "cuda" or torch.version.hip is not None:
        raise ValueError(f"clusters need an NVIDIA CUDA device, not {device}")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise ValueError(f"clusters need sm_90 or newer, not {device}")


def get_cluster_capacity(device: torch.device, cluster_size: int) -> int:
    """How many CTAs of one an SM fit at once in clusters of ``cluster_size``
    (a GPC's SMs left over by its clusters stay idle).

    CuTe asks the same of the driver through cutlass-dsl, which Triton and
    Gluon launches do not need, for clusters of up to 16 CTAs
    (``runtime.cute.occupancy.get_max_active_clusters``).  The two size
    persistent grids with reserved SMs differently: CuTe keeps the clusters
    within the SMs the reserved ones leave, while ``cluster_workers`` also
    takes the reserved SMs off this capacity, because a megakernel's workers
    wait on one another, so each must be resident, and another kernel on a
    reserved SM may sit in a GPC's cluster slot."""
    if not 1 <= cluster_size <= MAX_CLUSTER_SIZE:
        raise ValueError(f"clusters of {cluster_size} (1 to {MAX_CLUSTER_SIZE})")
    index = torch.cuda.current_device() if device.index is None else device.index
    return _cluster_capacity(index, cluster_size)


@functools.cache
def _cluster_capacity(index: int, cluster_size: int) -> int:
    """``get_cluster_capacity`` of device ``index``."""
    device = torch.device("cuda", index)
    cuda_driver = _cuda_driver()
    with _empty_function(device) as function:
        # All the shared memory a CTA may opt in to: one CTA an SM.
        properties = torch.cuda.get_device_properties(device)
        shared = properties.shared_memory_per_block_optin  # pyrefly: ignore [missing-attribute]
        error, clusters = cuda_driver.cuOccupancyMaxActiveClusters(
            function, _cluster_launch_config(cuda_driver, cluster_size, 128, shared)
        )
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA cluster occupancy query failed: {error}")
    return int(clusters) * cluster_size


def _cluster_launch_config(
    cuda_driver: object, cluster_size: int, threads: int, shared: int
) -> object:
    """A launch configuration of one cluster of ``cluster_size`` CTAs."""
    config = cuda_driver.CUlaunchConfig()  # pyrefly: ignore [missing-attribute]
    config.gridDimX, config.gridDimY, config.gridDimZ = cluster_size, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = threads, 1, 1
    config.sharedMemBytes = shared
    attribute = cuda_driver.CUlaunchAttribute()  # pyrefly: ignore [missing-attribute]
    attribute.id = cuda_driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION  # pyrefly: ignore [missing-attribute]
    attribute.value.clusterDim.x = cluster_size
    attribute.value.clusterDim.y = 1
    attribute.value.clusterDim.z = 1
    config.attrs = [attribute]
    config.numAttrs = 1
    return config


def compile_only_launch_args(
    *args: object,
    _remote_copy_signal_slots_per_program: int = 0,
    _remote_barrier_signal_slots_per_program: int = 0,
    _remote_copy_scratch_specs: tuple[tuple[torch.Tensor, int], ...] = (),
    _persistent_state_specs: tuple[
        tuple[torch.Tensor, int, torch.dtype, bool], ...
    ] = (),
    **kwargs: object,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Return ``triton_kernel.run`` arguments for compiling without launching.

    The arguments ``default_launcher`` appends become empty tensors of the same
    dtypes, which is all compilation reads, and launcher-only options drop.
    """
    dtypes = [torch.int64] * (
        bool(_remote_copy_signal_slots_per_program)
        + bool(_remote_barrier_signal_slots_per_program)
    )
    dtypes += [like.dtype for like, _ in _remote_copy_scratch_specs]
    for _, _, dtype, symmetric in _persistent_state_specs:
        dtypes += [dtype, torch.int64] if symmetric else [dtype]
    kwargs = {name: value for name, value in kwargs.items() if not name.startswith("_")}
    if not dtypes:
        return args, kwargs
    # The specs' tensors name the device when every argument is a descriptor.
    likes = [like for like, _ in _remote_copy_scratch_specs]
    likes += [like for like, *_ in _persistent_state_specs]
    device = next(t.device for t in (*args, *likes) if isinstance(t, torch.Tensor))
    placeholders = [torch.empty(0, dtype=dtype, device=device) for dtype in dtypes]
    return (*args, *placeholders), kwargs


def _get_remote_copy_signal(
    triton_kernel: object,
    dst: torch.Tensor,
    process_group_name: str,
    required_slots: int,
) -> torch.Tensor:
    """Return compiler-owned completion slots from ``dst``'s signal pad."""
    import torch.distributed._symmetric_memory as symm_mem

    cache = vars(triton_kernel).setdefault("_helion_remote_copy_signal_cache", {})

    key = (id(dst), process_group_name)
    entry = cache.get(key)
    if entry is not None and entry[0]() is dst:
        signal_pad = entry[1]
    else:
        handle = symm_mem.rendezvous(
            dst,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
        signal_pad = handle.get_signal_pad(handle.rank, dtype=torch.int64)

        def remove_from_cache(_ref: object) -> None:
            cache.pop(key, None)

        cache[key] = (weakref.ref(dst, remove_from_cache), signal_pad)

    capacity = signal_pad.numel()
    if required_slots > capacity:
        raise RuntimeError(
            "Helion remote copies require "
            f"{required_slots} int64 completion slots, but the symmetric-memory "
            f"signal pad has capacity {capacity}. Increase the signal pad size "
            "before allocating symmetric tensors."
        )
    # Reserve from the end so Helion's slots do not overlap PyTorch's standard
    # low-offset signal-pad protocols.
    return signal_pad.narrow(0, capacity - required_slots, required_slots)


def _get_remote_barrier_signal(
    triton_kernel: object,
    process_group_name: str,
    required_slots: int,
) -> torch.Tensor:
    """Return compiler-owned peer-barrier counters from a group workspace."""
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", torch.cuda.current_device())
    cache = vars(triton_kernel).setdefault("_helion_remote_barrier_signal_cache", {})
    key = (device, process_group_name)
    entry = cache.get(key)
    if entry is None:
        workspace = symm_mem.empty(1, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(
            workspace,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
        cache[key] = (workspace, handle)
    else:
        _, handle = entry
    signal_pad = handle.get_signal_pad(handle.rank, dtype=torch.int64)
    capacity = signal_pad.numel()
    if required_slots > capacity:
        raise RuntimeError(
            "Helion remote barriers require "
            f"{required_slots} int64 completion slots, but the symmetric-memory "
            f"signal pad has capacity {capacity}. Increase the signal pad size "
            "before launching the kernel."
        )
    return signal_pad.narrow(0, capacity - required_slots, required_slots)


def _get_remote_copy_scratch(
    triton_kernel: object,
    like: torch.Tensor,
    slot: int,
    required_numel: int,
) -> torch.Tensor:
    """Return stream-local global scratch for one computed DMA source."""
    if like.device.type != "cuda":
        raise RuntimeError("NVSHMEM remote-copy scratch requires a CUDA tensor")
    stream = torch.cuda.current_stream(like.device)
    cache = vars(triton_kernel).setdefault("_helion_remote_copy_scratch_cache", {})
    key = (like.device, like.dtype, stream.cuda_stream, slot)
    scratch = cache.get(key)
    if scratch is None or scratch.numel() < required_numel:
        scratch = torch.empty(
            required_numel,
            dtype=like.dtype,
            device=like.device,
        )
        cache[key] = scratch
    return scratch


def _get_persistent_state(
    triton_kernel: object,
    like: torch.Tensor,
    namespace: tuple[object, ...],
    slot: int,
    required_numel: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return stream-local compiler state retained across kernel launches."""
    if like.device.type != "cuda":
        raise RuntimeError("persistent Triton state requires a CUDA tensor")
    stream = torch.cuda.current_stream(like.device)
    cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
    key = (like.device, dtype, stream.cuda_stream, namespace, slot)
    state = cache.get(key)
    if state is None or state.numel() < required_numel:
        state = torch.zeros(required_numel, dtype=dtype, device=like.device)
        cache[key] = state
    return state


def _get_process_group_state(
    triton_kernel: object,
    like: torch.Tensor,
    namespace: tuple[object, ...],
    slot: int,
    required_numel: int,
    dtype: torch.dtype,
    *,
    symmetric: bool,
    process_group_name: str,
    shares: tuple[int, ...] = (),
) -> tuple[object, ...]:
    """Return the kernel arguments of retained state shared by every stream.

    One epoch then covers the dispatch ticket and peer counters, so launches of
    one kernel must not overlap across streams. A symmetric state is followed by
    its per-rank base pointer table.
    """
    if like.device.type != "cuda":
        raise RuntimeError("persistent Triton state requires a CUDA tensor")
    cache = vars(triton_kernel).setdefault("_helion_persistent_state_cache", {})
    # The numel is part of the namespace, so a cached entry always fits.
    key = (like.device, dtype, process_group_name, namespace, slot)
    entry = cache.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Helion allocates distributed launch state on first use; run the "
                "kernel once before CUDA graph capture"
            )
        if symmetric:
            entry = _new_symmetric_state(
                like.device,
                (
                    triton_kernel.__name__,  # type: ignore[attr-defined]
                    namespace,
                    slot,
                    required_numel,
                    str(dtype),
                ),
                required_numel,
                dtype,
                process_group_name,
                shares,
            )
        else:
            entry = (torch.zeros(required_numel, dtype=dtype, device=like.device),)
        cache[key] = entry
    return entry[:2]


def _new_symmetric_state(
    device: torch.device,
    fingerprint: tuple[object, ...],
    numel: int,
    dtype: torch.dtype,
    process_group_name: str,
    shares: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Allocate zeroed symmetric state once every rank agrees on the launch.

    ``shares``: the share of each tagged tensor this rank pushes; the ranks'
    shares must cover each tensor once.  Returns the state, its per-rank base
    pointer table followed by this rank, and the owning handle.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from torch.distributed.distributed_c10d import _resolve_process_group

    group = _resolve_process_group(
        process_group_name  # pyrefly: ignore[bad-argument-type]
    )
    with torch.cuda.device(device):
        fingerprint = (*fingerprint, torch.cuda.get_device_capability(device))
        world = dist.get_world_size(group)
        gathered: list[object] = [None] * world
        dist.all_gather_object(gathered, (fingerprint, shares), group=group)
        entries = cast("list[tuple[object, tuple[int, ...]]]", gathered)
        fingerprints = [entry[0] for entry in entries]
        if any(other != fingerprint for other in fingerprints):
            raise RuntimeError(
                "Helion distributed kernels require the same launch on every "
                f"rank; got {fingerprints!r}"
            )
        for index in range(len(shares)):
            pushed = [entry[1][index] for entry in entries]
            if sorted(pushed) != list(range(world)):
                raise RuntimeError(
                    "Helion tagged peer transport needs each rank to push its own "
                    f"share of a tensor; the ranks push shares {pushed}, so some "
                    "ranks write the same part of every rank's copy."
                )
        state = symm_mem.empty(numel, dtype=dtype, device=device)
        state.zero_()
        # Zero before rendezvous, since peers may publish right after it.
        torch.cuda.current_stream(device).synchronize()
        handle = symm_mem.rendezvous(
            state,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
    ptrs = torch.tensor(
        [*handle.buffer_ptrs, handle.rank], dtype=torch.int64, device=device
    )
    return state, ptrs, handle


def _has_tensor_memory() -> bool:
    """Whether the current device has tensor memory (NVIDIA tcgen05 targets):
    elsewhere a cooperative grid's launch patches nothing
    (``_fit_tensor_memory``), so it takes neither the lock nor a compile
    lookup before its launch."""
    return (
        torch.version.hip is None
        and _cuda_available()
        and _device_has_tensor_memory(torch.cuda.current_device())
    )


# Once a process (it runs on every cooperative launch).
_cuda_available = functools.cache(torch.cuda.is_available)


@functools.cache
def _device_has_tensor_memory(device: int) -> bool:
    """Whether CUDA device ``device`` has tensor memory (``_has_tensor_memory``)."""
    # tcgen05: sm_100, sm_103 and sm_110, not sm_120 (consumer Blackwell).
    return 10 <= torch.cuda.get_device_capability(device)[0] < 12


def _tensor_memory_ctas(metadata: object) -> int | None:
    """The CTAs of a compiled kernel (its ``metadata``) an SM's tensor memory
    holds at once (None without tensor memory).  The occupancy queries count
    none, but a CTA allocates its columns (a power of two) at entry and holds
    them to its exit: one more resident CTA blocks in its allocation, while
    CTAs that wait on its work keep theirs."""
    columns = getattr(metadata, "tmem_size", None) or 0
    if not columns:
        return None
    # Triton's budget (``CompiledKernel._init_handles``): more on sm_107.
    budget = 576 if metadata.target.arch == 107 else TENSOR_MEMORY_COLUMNS  # type: ignore[attr-defined]
    return budget // columns


def _validate_resident_program_capacity(
    compiled_kernel: object,
    args: tuple[object, ...],
    *,
    required_programs: int,
    cluster_size: int = 1,
) -> None:
    """Reject a polling schedule whose required CTA cohort cannot be resident
    (in clusters of ``cluster_size``: a GPC's SMs left over by its clusters do
    not count), at the dynamic shared memory the launch takes (its packed
    metadata's: padded for tensor memory, ``_fit_tensor_memory``)."""
    tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if tensor is not None and tensor.device.type != "cuda":
        raise RuntimeError("cross-loop residency checks require a CUDA tensor")
    # Without a bare tensor (all host tensor descriptors), the device Triton
    # launches on.
    device = (
        tensor.device
        if tensor is not None
        else torch.device("cuda", torch.cuda.current_device())
    )

    # The compiled warp count (a warp-specialized kernel's includes its
    # partitions').  Cache the exact driver result on the compiled
    # specialization because this wrapper is also called during graph capture.
    function = compiled_kernel.function  # type: ignore[attr-defined]
    num_warps, _, shared, *_ = compiled_kernel.packed_metadata  # type: ignore[attr-defined]
    cache = vars(compiled_kernel).setdefault(
        "_helion_resident_program_capacity_cache", {}
    )
    key = (device, num_warps, shared, cluster_size)
    capacity = cache.get(key)
    if capacity is None:
        cuda_driver = _cuda_driver()
        if cluster_size > 1:
            with torch.cuda.device(device):
                error, clusters = cuda_driver.cuOccupancyMaxActiveClusters(
                    cuda_driver.CUfunction(int(function)),
                    _cluster_launch_config(
                        cuda_driver, cluster_size, num_warps * 32, shared
                    ),
                )
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                # E.g. a cluster size the device (a MIG slice) cannot launch.
                raise RuntimeError(
                    f"{CLUSTER_LAUNCH_REFUSED}the CUDA cluster occupancy query "
                    f"failed for cross-loop kernel: {error}"
                )
            capacity = int(clusters) * cluster_size
        else:
            with torch.cuda.device(device):
                error, blocks_per_sm = (
                    cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        cuda_driver.CUfunction(int(function)),
                        num_warps * 32,
                        shared,
                    )
                )
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"CUDA occupancy query failed for cross-loop kernel: {error}"
                )
            properties = torch.cuda.get_device_properties(device)
            capacity = int(blocks_per_sm) * int(properties.multi_processor_count)
        cache[key] = capacity
    if cluster_size > 1 and capacity < cluster_size:
        raise RuntimeError(
            f"{CLUSTER_LAUNCH_REFUSED}no whole cluster of {cluster_size} CTAs fits "
            "at this launch's resources"
        )
    if required_programs > capacity:
        raise RuntimeError(
            "Cross-loop scheduling requires "
            f"{required_programs} concurrently resident programs, but this "
            f"kernel/device can residently execute only {capacity}. Choose a "
            "lower-resource configuration, a smaller ready prefix, or a "
            "root barrier."
        )
