"""Leases on the CUDA graph that a stream captures into.

A captured graph embeds the addresses of launcher-owned buffers, which must
outlive it. ``cute_cuda_graph`` ties them to its graph object
(``launch_caches``); a raw ``torch.cuda.graph`` does not expose its owner to
the launcher. A lease needs no owner: it is a CUDA user object that the
capturing graph retains. CUDA copies the reference to every executable graph
instantiated from that graph, which is what PyTorch keeps (it destroys the
captured graph at the end of the capture unless asked to keep it), and drops
it when the last of them is destroyed and its launches have completed.

CUDA runs a user object's destructor on an internal thread that may not
block or call CUDA, and may still run it while the interpreter shuts down.
The destructor is therefore libc's ``sem_post`` on a semaphore that libc
allocated, not Python code: it takes no interpreter lock, and a graph that
outlives the interpreter posts into live memory. The lease's holder polls
(:func:`graph_lease_ended`) and releases what the graph used on a thread of
its own, as CUDA's documentation of user objects asks. ``sem_post`` returns
a status where CUDA expects no result; the caller ignores it.

The driver calls go through the capture guard's driver
(``capture_pdl_cuda.CudaCaptureDriver``): its typed entry points of
``libcuda.so.1`` and its one capture query, so a lease and the guard read a
stream's capture the same way. Importing the module loads no library.
"""

from __future__ import annotations

import ctypes
import functools

import torch

from ... import exc
from .capture_pdl_cuda import CudaCaptureDriver

# CU_USER_OBJECT_NO_DESTRUCTOR_SYNC and CU_GRAPH_USER_OBJECT_MOVE.
_NO_DESTRUCTOR_SYNC = 1
_MOVE = 1
# More than a sem_t of any libc.
_SEMAPHORE_BYTES = 64


@functools.cache
def _libc() -> ctypes.CDLL:
    """libc's allocator and unnamed semaphores."""
    library = ctypes.CDLL(None, use_errno=True)
    library.malloc.argtypes = [ctypes.c_size_t]
    library.malloc.restype = ctypes.c_void_p
    library.free.argtypes = [ctypes.c_void_p]
    library.free.restype = None
    library.sem_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
    library.sem_trywait.argtypes = [ctypes.c_void_p]
    library.sem_destroy.argtypes = [ctypes.c_void_p]
    return library


@functools.cache
def cuda_driver() -> CudaCaptureDriver:
    """The process's typed entry points of ``libcuda.so.1``."""
    return CudaCaptureDriver()


def _semaphore(libc: ctypes.CDLL) -> int:
    """A semaphore at zero in memory that libc allocated.

    The poll reads it as soon as the lease exists, so both calls run, and
    are checked, in every interpreter mode: as an ``assert`` they would be
    dropped by ``python -O``, and a recycled block that is not zero reads as
    a lease that has ended.
    """
    semaphore = libc.malloc(_SEMAPHORE_BYTES)
    if not semaphore:
        raise MemoryError("no memory for the semaphore of a CUDA graph lease")
    if libc.sem_init(semaphore, 0, 0):
        error = ctypes.get_errno()
        libc.free(semaphore)
        raise OSError(error, "sem_init failed for the semaphore of a CUDA graph lease")
    return semaphore


def lease_capturing_graph(device: torch.device) -> int:
    """Take a lease on the graph that ``device``'s current stream captures
    into, and return it for :func:`graph_lease_ended`.

    The lease ends when CUDA has destroyed that graph and every executable
    graph instantiated from it, and their launches have completed.
    """
    libc, driver = _libc(), cuda_driver()
    semaphore = _semaphore(libc)
    holder = ctypes.c_void_p()
    with torch.cuda.device(device):
        capture = driver.stream_capture(torch.cuda.current_stream().cuda_stream)
        graph = None if capture is None else capture[1]
        failed = graph is None or driver.library.cuUserObjectCreate(
            ctypes.byref(holder),
            semaphore,
            ctypes.cast(libc.sem_post, ctypes.c_void_p),
            1,
            _NO_DESTRUCTOR_SYNC,
        )
        if failed:
            libc.sem_destroy(semaphore)
            libc.free(semaphore)
        elif driver.library.cuGraphRetainUserObject(graph, holder, 1, _MOVE):
            # CUDA destroys the object, whose destructor posts: the
            # semaphore has to stay.
            failed = True
            driver.library.cuUserObjectRelease(holder, 1)
    if failed:
        raise exc.BackendUnsupported(
            "cute", "the CUDA graph being captured did not take a lease"
        )
    return semaphore


def graph_lease_ended(lease: int) -> bool:
    """Whether the graphs of ``lease`` are destroyed and their launches
    complete. True once: the lease is gone with that answer."""
    libc = _libc()
    if libc.sem_trywait(lease):
        return False
    # A semaphore may be destroyed as soon as a wait on it has returned.
    libc.sem_destroy(lease)
    libc.free(lease)
    return True
