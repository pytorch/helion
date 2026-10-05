from __future__ import annotations

import contextlib
from datetime import timedelta
import io
import itertools
import os
import re
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch
import warnings

import torch
from torch import Tensor
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase as CommonTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore

import helion
from helion._compiler import tile_dependency
from helion._dist_utils import all_gather_object
from helion._dist_utils import kernel_uses_symm_mem
from helion._dist_utils import sync_object
from helion._dist_utils import sync_seed
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import TestCase
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion.autotuner import search_algorithms
from helion.autotuner.effort_profile import _PROFILES
from helion.autotuner.effort_profile import AutotuneEffortProfile
from helion.autotuner.effort_profile import DifferentialEvolutionConfig
from helion.autotuner.effort_profile import PatternSearchConfig
from helion.autotuner.effort_profile import RandomSearchConfig
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

autotuner_names = ["fixed", *search_algorithms]

# The full kernel x autotuner cross product is too slow for CI. Only the
# cheapest kernel (test_allreduce) runs every algorithm; the other kernels keep
# one fast representative each, since the search algorithms themselves are
# covered in test_autotuner.py and distributed coordination is orthogonal to
# the choice of algorithm.
representative_autotuner_names = ["FiniteSearch"]

# LLM-guided autotuners call a real LLM endpoint and abort on every rank when
# no API key is configured, so skip them instead of failing in key-less envs.
_LLM_AUTOTUNER_NAMES = frozenset(
    name for name in search_algorithms if name.startswith("LLM")
)
_HAS_LLM_API_KEY = any(
    os.environ.get(name)
    for name in ("HELION_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
)

# torch.distributed._symmetric_memory.is_symm_mem_tensor exists only on newer
# PyTorch. The symm-mem-based gating degrades to conservative behavior without it,
# so the tests that assert the fine-grained gating require the API.
_HAS_SYMM_MEM_DETECT = hasattr(symm_mem, "is_symm_mem_tensor")


def custom_get_timeout(test_id: str) -> int:
    """
    Use a larger timeout setting to autotune distributed kernels.
    """
    return int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "600"))


def one_shot_allreduce_kernel(
    a_shared: torch.Tensor,
    my_rank: hl.constexpr,
    group_name: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
) -> torch.Tensor:
    out = torch.empty_like(a_shared)
    N = out.size(0)
    a_shared_tuple = torch.ops.symm_mem.get_remote_tensors(a_shared, group_name)

    for tile_n in hl.tile(N):
        acc = hl.zeros([tile_n], dtype=a_shared.dtype, device=a_shared.device)

        for a in a_shared_tuple:
            acc += a[tile_n]

        out[tile_n] = acc
    return out


def _remote_views(
    t: torch.Tensor, group_name: hl.constexpr
) -> tuple[torch.Tensor, ...]:
    return torch.ops.symm_mem.get_remote_tensors(t, group_name)


def _with_pipeline(
    kernel: helion.Kernel, args: tuple[object, ...], pipeline: str
) -> BoundKernel[object]:
    bound = kernel.bind(args)
    config = bound.config_spec.default_config().config
    bound.set_config(helion.Config(**{**config, "cross_loop_pipeline": pipeline}))
    return bound


@helion.kernel(autotune_effort="none", static_shapes=True)
def two_root_peer_read_kernel(
    symm: torch.Tensor,
    views_group: hl.constexpr,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    n = symm.size(0)
    peers = _remote_views(symm, views_group)
    out = torch.empty_like(symm)
    for tile in hl.tile(n):
        symm[tile] = symm[tile] + 1
    # The barrier orders only this rank's roots.
    hl.barrier()
    for tile in hl.tile(n):
        out[tile] = peers[1][tile] + peers[2][tile]
    return out


def _peer_sum(
    symm: torch.Tensor,
    start: hl.constexpr,
    absolute: hl.constexpr,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    n = symm.numel() - start
    if absolute:
        local = torch.as_strided(symm, (n,), (1,), storage_offset=start)
    else:
        local = symm.view(-1)[start:]
    peers = _remote_views(local, group_name)
    out = torch.empty_like(local)
    for tile in hl.tile(local.size(0)):
        out[tile] = peers[1][tile] + peers[2][tile]
    return out


peer_sum_kernel = helion.kernel(_peer_sum, autotune_effort="none", static_shapes=True)
dynamic_peer_sum_kernel = helion.kernel(
    _peer_sum, autotune_effort="none", static_shapes=False
)


@helion.kernel(autotune_effort="none", static_shapes=True)
def peer_read_through_copy_kernel(
    symm: torch.Tensor, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    m, n = symm.size()
    peers = _remote_views(symm, group_name)
    out = symm.new_empty(m)
    for tile_i, tile_j in hl.tile([m, n]):
        symm[tile_i, tile_j] = symm[tile_i, tile_j] + 1
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=symm.dtype)
        q = peers[1]
        for tile_n in hl.tile(n):
            acc = acc + q[tile_m, tile_n].sum(-1)
            q = peers[1]
        out[tile_m] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def two_root_hidden_access_kernel(
    symm: torch.Tensor,
    offsets: torch.Tensor,
    group_name: hl.ProcessGroupName,
    remote_barrier: hl.constexpr,
    local_source: hl.constexpr,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    source = symm if local_source else peers[1]
    out = symm.new_empty([offsets.size(0), 16], dtype=torch.float16)
    for tile in hl.tile(symm.size(0)):
        symm[tile] = symm[tile] + 1
    for tile in hl.tile(offsets.size(0)):
        if remote_barrier:
            hl.remote_barrier(1)
        lanes = hl.load_bfloat16_x16_to_float16(source, offsets[tile])
        for i in hl.static_range(16):
            out[tile, i] = lanes[i]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def inband_rule_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    broken_rule: hl.constexpr,
    world: hl.constexpr,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty_like(symm)
    # R4: no task polls rank 0. R6: the consumer reads half of each peer's buffer.
    first = 1 if broken_rule == "R4" else 0
    read = symm.size(0) // 2 if broken_rule == "R6" else symm.size(0)
    for tile in hl.tile(symm.size(0)):
        if broken_rule == "R3":
            out[tile] = peers[1][tile]
        else:
            out[tile] = x[tile]
    for tile in hl.tile(symm.size(0)):
        symm[tile] = x[tile]
        if broken_rule == "R1":
            symm[tile] = x[tile] + 1
    for tile in hl.tile(read):
        acc = hl.zeros([tile], dtype=x.dtype)
        for r in hl.static_range(first, world):
            acc = acc + peers[r][tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def diagonal_store_kernel(
    symm: torch.Tensor, x: torch.Tensor, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty_like(symm)
    block = hl.register_block_size(symm.size(0))
    # Both root axes share one block id, so tasks (i, j) and (i, k) overlap.
    for tile_i, tile_j in hl.tile(symm.size(), block_size=[block, block]):
        symm[tile_i, tile_i] = x[tile_i, tile_j]
    for tile in hl.tile(symm.size()):
        out[tile] = peers[0][tile] + peers[1][tile]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def pipelined_allreduce_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    variant: hl.constexpr,
    world: hl.constexpr,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        if variant == "constant":
            symm[tile] = hl.full([tile], 1.0, dtype=x.dtype)
        else:
            symm[tile] = x[tile]
        if variant == "R1":
            symm[tile] = x[tile] + 0
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for r in hl.static_range(world):
            acc = acc + peers[r][tile]
        out[tile] = acc
        if variant == "print":
            print("acc", acc)
        if variant == "loop":
            for inner in hl.tile(tile.begin, tile.end, block_size=64):
                part = hl.zeros([inner], dtype=x.dtype)
                for r in hl.static_range(world):
                    part = part + peers[r][inner]
                out[inner] = part
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def chained_exchange_kernel(
    symm: torch.Tensor,
    moment: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    moment_peers = _remote_views(moment, group_name)
    total = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        symm[tile] = x[tile]
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for peer in peers:
            acc = acc + peer[tile]
        total[tile] = acc
    # A one-element exchange that depends on every tile of the last one.
    for tile in hl.tile(moment.size(0)):
        moment[tile] = hl.zeros([tile], dtype=x.dtype) + torch.sum(total[:]) / x.size(0)
    for tile in hl.tile(x.size(0)):
        acc = total[tile]
        for moment_peer in moment_peers:
            acc = acc + moment_peer[:].to(x.dtype)
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def row_exchange_kernel(
    symm: torch.Tensor, x: torch.Tensor, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        symm[0, tile] = x[tile]
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for peer in peers:
            acc = acc + peer[0, tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def matrix_exchange_kernel(
    symm: torch.Tensor, x: torch.Tensor, group_name: hl.ProcessGroupName
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        symm[tile_m, tile_n] = x[tile_m, tile_n]
    for tile_m, tile_n in hl.tile(x.size()):
        acc = hl.zeros([tile_m, tile_n], dtype=x.dtype)
        for peer in peers:
            acc = acc + peer[tile_m, tile_n]
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def scatter_exchange_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    rows: torch.Tensor,
    count: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty(symm.size(), dtype=torch.float32, device=symm.device)
    half = symm.size(1) // 2
    for tile_h, tile_t in hl.tile([2, x.size(0)], block_size=[1, 1]):
        # Tasks past the count skip; a negative row writes nothing.
        if tile_t.begin < count[0]:
            row = rows[tile_t.begin, :]
            cols = tile_h.begin * half + hl.arange(half)
            hl.store(
                symm,
                [row, cols],
                x[tile_t.begin, :, cols],
                extra_mask=(row >= 0)[:, None],
            )
    for tile_m, tile_c in hl.tile(symm.size(), block_size=[8, 32]):
        acc = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for peer in peers:
            acc = acc + peer[tile_m, tile_c].to(torch.float32)
        out[tile_m, tile_c] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def scatter_matmul_exchange_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rows: torch.Tensor,
    count: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    peers = _remote_views(symm, group_name)
    out = torch.empty(symm.size(), dtype=torch.float32, device=symm.device)
    half = symm.size(1) // 2
    for tile_h, tile_t in hl.tile([2, x.size(0)], block_size=[1, 1]):
        if tile_t.begin < count[0]:
            row = rows[tile_t.begin, :]
            cols = tile_h.begin * half + hl.arange(half)
            acc = hl.zeros([row.size(0), half], dtype=torch.float32)
            for tile_k in hl.tile(x.size(2)):
                acc = hl.dot(x[tile_t.begin, :, tile_k], w[tile_k, cols], acc=acc)
            hl.store(
                symm, [row, cols], acc.to(symm.dtype), extra_mask=(row >= 0)[:, None]
            )
    for tile_m, tile_c in hl.tile(symm.size(), block_size=[128, 32]):
        total = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for peer in peers:
            total = total + peer[tile_m, tile_c].to(torch.float32)
        out[tile_m, tile_c] = total
    return out


# make it easy to use a 'smaller' profile than 'quick' in unit test
pattern_search_config = PatternSearchConfig(
    initial_population=6,
    copies=2,
    max_generations=3,
)

differential_evolution_config = DifferentialEvolutionConfig(
    population_size=10,
    max_generations=5,
)

random_search_config = RandomSearchConfig(
    count=20,
)

profile = AutotuneEffortProfile(
    pattern_search=pattern_search_config,
    lfbo_pattern_search=pattern_search_config,
    differential_evolution=differential_evolution_config,
    random_search=random_search_config,
)


@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestDistributed(TestCase, MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(
            unittest.mock.patch.dict(
                os.environ,
                {
                    "HELION_DIST_CHECK_CONFIG_CONSISTANCY": "1",
                    "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
                    "HELION_CAP_REBENCHMARK_REPEAT": "50",
                },
            )
        )
        cls._class_stack.enter_context(
            unittest.mock.patch.object(
                torch.testing._internal.common_distributed,
                "get_timeout",
                custom_get_timeout,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        # The per-test skip_if_lt_x_gpu(4) guards only fire inside the spawned
        # worker processes, so without this pre-check every skip on a <4-GPU
        # machine still pays ~6s of process spawning. Same check, parent-side.
        if not (
            torch.accelerator.is_available() and torch.accelerator.device_count() >= 4
        ):
            self.skipTest("Need at least 4 accelerator devices")
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return int(os.getenv("TEST_WORLD_SIZE", "4"))

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(
            unittest.mock.patch.dict(
                _PROFILES,
                {"full": profile},
            )
        )
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=self.rank,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _cleanup_process(self):
        self._exit_stack.close()
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_sync_seed(self):
        def _all_eq(xlist: list[Tensor]) -> bool:
            assert len(xlist) > 1
            lhs = xlist[0]
            return all(torch.allclose(lhs.cpu(), rhs.cpu()) for rhs in xlist[1:])

        self._init_process()
        torch.manual_seed(42 + self.rank)
        pg_name = dist.group.WORLD.group_name

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)

        self.assertFalse(_all_eq(xlist))

        with sync_seed(process_group_name=pg_name):
            x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)
        self.assertTrue(_all_eq(xlist))

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x, pg_name)
        self.assertFalse(_all_eq(xlist))

        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("autotuner", autotuner_names)
    def test_allreduce(self, autotuner):
        if autotuner in _LLM_AUTOTUNER_NAMES and not _HAS_LLM_API_KEY:
            self.skipTest("LLM autotuners require an LLM API key")
        self._init_process()
        if autotuner == "fixed":
            fixed_num_warps = 16 if torch.version.hip is not None else 32
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[8192],
                    num_warps=fixed_num_warps,
                ),
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(one_shot_allreduce_kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(block_sizes=[8192], num_warps=16),
                    helion.Config(block_sizes=[4096], num_warps=16),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(one_shot_allreduce_kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                one_shot_allreduce_kernel,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_allreduce(kernel)

        self._cleanup_process()

    def do_test_allreduce(self, kernel):
        group = dist.group.WORLD

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(N, dtype=dtype, device=self.device).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)
        result = kernel(
            a_shared,
            symm_mem_hdl.rank,
            group.group_name,
            symm_mem_hdl.world_size,
        )

        torch.cuda.synchronize()

        expected = torch.empty_like(result).copy_(a_shared)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("variant", ("inband", "R1"))
    @parametrize("pipeline", ("static", "dynamic"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_pipelined_allreduce_replays(
        self, variant: str, pipeline: str, dtype: torch.dtype
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        world, n = self.world_size, 4096
        symm = symm_mem.empty(n, device=self.device, dtype=dtype)
        symm_mem.rendezvous(symm, group=group)
        x = torch.empty(n, device=self.device, dtype=dtype)
        # Distinct small integers per element stay exact in bf16 and catch lane
        # mixups in packed words.
        lane = torch.arange(n, device=self.device, dtype=dtype) % 16
        args = (symm, x, group.group_name, variant, world)
        bound = _with_pipeline(pipelined_allreduce_kernel, args, pipeline)
        outs = []

        def launch(step: int, replay: bool = False) -> None:
            # Skewed ranks overwrite symm while peers may still read the last
            # launch's values, which the done barrier orders.
            torch.cuda._sleep(100_000 * ((self.rank + step) % world))
            x.copy_(lane + 4 * step + self.rank)
            if replay:
                graph.replay()
                outs.append(graph_out.clone())
            else:
                outs.append(bound(*args).clone())

        for step in range(4):
            launch(step)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = bound(*args)
        # Alternate graph replays with eager launches.
        for step in range(4, 12):
            launch(step, replay=step % 2 == 0)
        torch.cuda.synchronize()
        for step, out in enumerate(outs):
            expected = world * (lane + 4 * step) + world * (world - 1) // 2
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self._cleanup_process()

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_matrix_exchange_masked_rows(self) -> None:
        self._init_process()
        group = dist.group.WORLD
        # 20 rows in 8-row tiles: the last tile masks rows, pairs pack columns.
        symm = symm_mem.empty(20, 256, device=self.device, dtype=torch.bfloat16)
        symm_mem.rendezvous(symm, group=group)
        x = torch.empty(20, 256, device=self.device, dtype=torch.bfloat16)
        bound = matrix_exchange_kernel.bind((symm, x, group.group_name))
        config = bound.config_spec.default_config().config
        bound.set_config(helion.Config(**{**config, "block_sizes": [8, 64, 8, 64]}))
        lane = torch.arange(20 * 256, device=self.device).view(20, 256) % 16
        for step in range(4):
            torch.cuda._sleep(100_000 * ((self.rank + step) % self.world_size))
            x.copy_(lane + 4 * step + self.rank)
            out = bound(symm, x, group.group_name)
            world = self.world_size
            expected = world * (lane + 4 * step) + world * (world - 1) // 2
            torch.testing.assert_close(out, expected.to(out.dtype), rtol=0, atol=0)
        self._cleanup_process()

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    def test_inband_scatter_reads_unwritten_rows(self, dtype: torch.dtype) -> None:
        self._init_process()
        group = dist.group.WORLD
        world, tasks, lanes, cols = self.world_size, 4, 8, 64
        symm = symm_mem.empty(tasks * lanes, cols, device=self.device, dtype=dtype)
        symm_mem.rendezvous(symm, group=group)
        # Every rank mirrors each rank's buffer; rows no task writes keep values.
        index = torch.arange(tasks * lanes * cols).view(tasks, lanes, cols)
        init = index.view(tasks * lanes, cols) % 5
        mirror = torch.stack([200 + r + init for r in range(world)]).float()
        symm.copy_(mirror[self.rank])
        torch.cuda.synchronize()
        dist.barrier()
        x = torch.empty(tasks, lanes, cols, device=self.device, dtype=dtype)
        rows = torch.empty(tasks, lanes, device=self.device, dtype=torch.int32)
        count = torch.empty(1, device=self.device, dtype=torch.int32)
        args = (symm, x, rows, count, group.group_name)
        bound = scatter_exchange_kernel.bind(args)
        for step in range(6):
            generator = torch.Generator().manual_seed(step)
            plans = []
            for r in range(world):
                # Odd steps skip the last task and every third lane of the rest.
                row = torch.randperm(tasks * lanes, generator=generator)
                row = row.view(tasks, lanes).to(torch.int32)
                live = tasks - step % 2
                if step % 2:
                    row[:, ::3] = -1
                values = ((index * (r + 1) + 3 * step) % 128).float()
                for t in range(live):
                    written = row[t] >= 0
                    mirror[r, row[t][written].long()] = values[t][written]
                plans.append((values, row, live))
            values, row, live = plans[self.rank]
            torch.cuda._sleep(100_000 * ((self.rank + step) % world))
            x.copy_(values)
            rows.copy_(row)
            count.fill_(live)
            out = bound(*args)
            expected = mirror.sum(0).to(self.device)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self._cleanup_process()

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_inband_scatter_fallback_with_warp_specialized_producer(self) -> None:
        # Reading unwritten rows must not hang while WS worker warps are parked.
        self._init_process()
        group = dist.group.WORLD
        world, tasks, lanes, cols, k = self.world_size, 2, 128, 64, 64
        dtype = torch.bfloat16
        symm = symm_mem.empty(tasks * lanes, cols, device=self.device, dtype=dtype)
        symm_mem.rendezvous(symm, group=group)
        symm.fill_(self.rank + 1)
        torch.cuda.synchronize()
        dist.barrier()
        x = torch.arange(tasks * lanes * k, device=self.device) % 2
        x = x.view(tasks, lanes, k).to(dtype)
        w = torch.ones(k, cols, device=self.device, dtype=dtype)
        # Only task 0 runs, and every third of its lanes writes nothing.
        rows = torch.arange(tasks * lanes, device=self.device, dtype=torch.int32)
        rows = rows.view(tasks, lanes)
        rows[:, ::3] = -1
        count = torch.ones(1, device=self.device, dtype=torch.int32)
        args = (symm, x, w, rows, count, group.group_name)
        bound = scatter_matmul_exchange_kernel.bind(args)
        config = bound.config_spec.default_config().config
        ws = {"block_sizes": [32], "range_warp_specializes": [None, True, None]}
        bound.set_config(helion.Config(**{**config, **ws}))
        code = bound.to_triton_code()
        self.assertIn("warp_specialize=True", code)
        self.assertIn("_scatter_cover(", code)
        written = torch.zeros(tasks * lanes, 1, device=self.device, dtype=torch.bool)
        written[rows[0][rows[0] >= 0].long()] = True
        # Each written element sums k // 2 ones; unwritten rows keep rank + 1.
        expected = torch.where(
            written, world * k // 2, world * (world + 1) // 2
        ).expand(tasks * lanes, cols)
        for step in range(3):
            torch.cuda._sleep(100_000 * ((self.rank + step) % world))
            out = bound(*args)
            torch.testing.assert_close(out, expected.float(), rtol=0, atol=0)
        self._cleanup_process()

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "dtypes",
        (
            (torch.float32, torch.float32),
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
        ),
    )
    @parametrize("pipeline", ("static", "dynamic"))
    def test_chained_exchanges_replay(
        self, dtypes: tuple[torch.dtype, torch.dtype], pipeline: str
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        world, n = self.world_size, 4096
        # An 8-byte buffer mixes in-band and peer-counter transports.
        symm, moment = (
            symm_mem.empty(n, device=self.device, dtype=dtypes[0]),
            symm_mem.empty(1, device=self.device, dtype=dtypes[1]),
        )
        symm_mem.rendezvous(symm, group=group)
        symm_mem.rendezvous(moment, group=group)
        x = torch.empty(n, device=self.device, dtype=dtypes[0])
        args = (symm, moment, x, group.group_name)
        bound = _with_pipeline(chained_exchange_kernel, args, pipeline)
        bound(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = bound(*args)
        for step in range(8):
            torch.cuda._sleep(100_000 * ((self.rank + step) % world))
            x.fill_(step + self.rank)
            if step % 2:
                graph.replay()
                out = graph_out
            else:
                out = bound(*args)
            # total = world * step + 6, plus each rank's copy of its mean.
            expected = (world + 1) * (world * step + world * (world - 1) // 2)
            torch.testing.assert_close(out, torch.full_like(out, expected))
        self._cleanup_process()

    @skipIfNotCUDA()
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_ranks_disagree_at_first_launch(self) -> None:
        self._init_process()
        group = dist.group.WORLD
        symm = symm_mem.empty(4096, device=self.device)
        symm_mem.rendezvous(symm, group=group)
        x = torch.zeros(4096, device=self.device)
        # Rank 0 sees another dependency graph; every rank raises, none hangs.
        with (
            patch.object(
                tile_dependency.TileDependencyGraph, "rank_digest", return_value="0"
            )
            if self.rank == 0
            else contextlib.nullcontext(),
            self.assertRaisesRegex(
                RuntimeError, "require the same launch on every rank"
            ),
        ):
            pipelined_allreduce_kernel(
                symm, x, group.group_name, "inband", self.world_size
            )
        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "kernel_name",
        (
            "one_shot_allreduce_bias_rmsnorm_kernel",
            "two_shot_allreduce_bias_rmsnorm_kernel",
        ),
    )
    @parametrize("autotuner", representative_autotuner_names)
    def test_allreduce_bias_rmsnorm(self, kernel_name, autotuner):
        """
        There is a similar test in test/test_examples_dist.py.
        The current test focus more on autotuning functionality.
        """
        self._init_process()
        mod = import_path(EXAMPLES_DIR / "distributed" / "allreduce_bias_rmsnorm.py")

        kernel = getattr(mod, kernel_name).fn
        if autotuner == "fixed":
            fixed_config = helion.Config(
                block_sizes=[8], num_warps=8, reduction_loops=[1024]
            )

            kernel = helion.kernel(
                config=fixed_config,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(block_sizes=[8], num_warps=8),
                    helion.Config(block_sizes=[8], num_warps=4),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_allreduce_bias_rmsnorm(
                kernel, mod.reference_allreduce_bias_rmsnorm
            )

        self._cleanup_process()

    def do_test_allreduce_bias_rmsnorm(self, kernel, ref_kernel):
        N, D = 128, 4096
        eps = 1e-5
        x = torch.randn(N, D, device=self.device)
        symm_mem_buffer = symm_mem.empty(N, D, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)
        torch.manual_seed(42)
        bias = torch.randn(D, device=self.device)
        weight = torch.randn(D, device=self.device)
        result = kernel(
            symm_mem_buffer,
            x,
            bias,
            weight,
            symm_mem_hdl.signal_pad_ptrs_dev,
            eps,
            symm_mem_hdl.rank,
            symm_mem_hdl.world_size,
            dist.group.WORLD.group_name,
        )

        expected = ref_kernel(x, bias, weight)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    # "fixed" stays alongside the representative: its deliberately small block
    # config is the only coverage for large launch grids on this kernel.
    @parametrize("autotuner", ["fixed", *representative_autotuner_names])
    def test_matmul_reduce_scatter(self, autotuner):
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "matmul_reduce_scatter.py")

        kernel = mod.matmul_reduce_scatter_kernel.fn
        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16
        if autotuner == "fixed":
            # small block on purpose to test large grid
            fixed_config = helion.Config(
                block_sizes=[2, 2, 32],
                num_warps=8,
                num_stages=3,
                indexing="block_ptr",
            )

            kernel = helion.kernel(
                config=fixed_config,
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(
                        block_sizes=[64, 64, 32],
                        num_warps=8,
                        num_stages=3,
                        indexing="block_ptr",
                    ),
                    helion.Config(
                        block_sizes=[64, 64, 32],
                        num_warps=4,
                        num_stages=3,
                        indexing="block_ptr",
                    ),
                ],
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            )(kernel)
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )
        else:
            kernel = helion.kernel(
                kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
            )
            context = unittest.mock.patch.dict(
                os.environ, {"HELION_AUTOTUNER": autotuner}
            )

        with context:
            self.do_test_matmul_reduce_scatter(
                kernel, mod.reference_matmul_reduce_scatter
            )
        self._cleanup_process()

    def do_test_matmul_reduce_scatter(self, kernel, ref_kernel):
        M, N, K = 512, 768, 1024

        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, device=self.device)

        # Weight matrix is the same across all ranks
        torch.manual_seed(42)
        b = torch.randn(K, N, device=self.device)

        symm_mem_buffer = symm_mem.empty(M, N, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)
        result = kernel(
            a,
            b,
            symm_mem_buffer,
            symm_mem_hdl.signal_pad_ptrs_dev,
            symm_mem_hdl.rank,  # RANK constexpr
            symm_mem_hdl.world_size,  # WORLD_SIZE constexpr
            dist.group.WORLD.group_name,  # GROUP_NAME constexpr
        )

        expected = ref_kernel(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_fp8_matmul_reduce_scatter(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 requires CUDA compute capability >= 9.0")
        self._init_process()

        mod = import_path(EXAMPLES_DIR / "distributed" / "fp8_matmul_reduce_scatter.py")

        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16
        M, N, K = 512, 768, 1024

        torch.manual_seed(42 + self.rank)
        a = torch.randn(M, K, device=self.device).to(torch.float8_e4m3fn)

        torch.manual_seed(42)
        b = (
            torch.randn(K, N, device=self.device)
            .to(torch.float8_e4m3fn)
            .t()
            .contiguous()
            .t()
        )

        scale_a = torch.rand(M, 1, device=self.device)
        scale_b = torch.rand(1, N, device=self.device)

        symm_mem_buffer = symm_mem.empty(M, N, dtype=torch.bfloat16, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)

        result = mod.fp8_matmul_reduce_scatter_kernel(
            a,
            b,
            scale_a,
            scale_b,
            symm_mem_buffer,
            symm_mem_hdl.signal_pad_ptrs_dev,
            RANK=symm_mem_hdl.rank,
            WORLD_SIZE=symm_mem_hdl.world_size,
            GROUP_NAME=dist.group.WORLD.group_name,
        )

        expected = mod.reference_fp8_matmul_reduce_scatter(a, b, scale_a, scale_b)

        torch.testing.assert_close(result, expected, rtol=8e-1, atol=8e-1)
        self._cleanup_process()

    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_two_dim_parallel_matmul(self):
        self._init_process()
        mod = import_path(EXAMPLES_DIR / "distributed" / "two_dim_parallel_matmul.py")
        _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16

        tp_size = 2
        sp_size = self.world_size // tp_size
        mesh = init_device_mesh(
            "cuda",
            (sp_size, tp_size),
            mesh_dim_names=("sp", "tp"),
        )
        tp_group = mesh.get_group("tp")
        sp_rank = self.rank // tp_size
        tp_rank = self.rank % tp_size

        kernel = mod.two_dim_parallel_matmul_kernel.fn
        kernel = helion.kernel(
            kernel, ignore_warnings=[helion.exc.TensorOperationInWrapper]
        )
        context = unittest.mock.patch.dict(
            os.environ, {"HELION_AUTOTUNER": "LFBOTreeSearch"}
        )

        with context:
            self.do_test_two_dim_parallel_matmul(
                kernel,
                mod.reference_two_dim_parallel_matmul,
                tp_group,
                sp_rank,
                tp_rank,
                tp_size,
                sp_size,
            )
        self._cleanup_process()

    def do_test_two_dim_parallel_matmul(
        self,
        helion_fn,
        ref_fn,
        tp_group,
        sp_rank,
        tp_rank,
        tp_size,
        sp_size,
    ):
        M, K, N = 512, 256, 512
        dtype = torch.float32

        M_local = M // sp_size
        K_local = K // tp_size

        torch.manual_seed(42 + sp_rank * tp_size + tp_rank)
        a_local = torch.randn(M_local, K_local, dtype=dtype, device=self.device)

        torch.manual_seed(42 + tp_rank)
        b_local = torch.randn(K_local, N, dtype=dtype, device=self.device)

        tp_group_name = tp_group.group_name  # type: ignore[missing-attribute]
        self.assertIsNotNone(tp_group_name)
        self.assertEqual(dist.get_world_size(tp_group), 2)
        self.assertEqual(dist.get_world_size(), 4)

        symm_mem_buf = symm_mem.empty(M_local, N, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(symm_mem_buf, tp_group_name)

        import unittest.mock

        with (
            unittest.mock.patch(
                "helion._dist_utils.sync_seed", wraps=sync_seed
            ) as mock_sync_seed,
            # The autotuner's compile/accuracy/timing gathers live in
            # BenchmarkProvider (moved from BaseSearch in #2029), and mock.patch
            # only intercepts the module-level binding, so spy there.
            unittest.mock.patch(
                "helion.autotuner.benchmark_provider.all_gather_object",
                wraps=all_gather_object,
            ) as mock_all_gather_object,
            unittest.mock.patch(
                "helion.autotuner.base_search.sync_object", wraps=sync_object
            ) as mock_sync_object,
        ):
            result = helion_fn(
                a_local,
                b_local,
                symm_mem_buf,
                hdl.signal_pad_ptrs_dev,
                TP_RANK=hdl.rank,
                TP_SIZE=hdl.world_size,
                GROUP_NAME=tp_group_name,
            )

        def _assert_pgn_group_size_2(mock: unittest.mock.MagicMock) -> None:
            mock.assert_called()
            pg_names = dist.distributed_c10d._world.pg_names  # type: ignore[attr-defined]
            for call in mock.call_args_list:
                pgn = call.kwargs.get("process_group_name")
                self.assertIsNotNone(pgn)
                group = next(g for g, name in pg_names.items() if name == pgn)
                self.assertEqual(dist.get_world_size(group), 2)

        _assert_pgn_group_size_2(mock_sync_seed)
        _assert_pgn_group_size_2(mock_all_gather_object)
        _assert_pgn_group_size_2(mock_sync_object)

        expected = ref_fn(a_local, b_local, tp_group)
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDistributedGating(CommonTestCase):
    """Issue #3024: single-process checks that distributed gating keys off
    symmetric-memory usage, not torch.distributed.is_initialized()."""

    def setUp(self) -> None:
        super().setUp()
        self._owns_pg = not dist.is_initialized()
        if self._owns_pg:
            dist.init_process_group(
                backend="gloo", world_size=1, rank=0, store=dist.HashStore()
            )

    def tearDown(self) -> None:
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    @staticmethod
    def _make_add() -> object:
        @helion.kernel(autotune_effort="none")
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        return add

    def _bind_add(self):
        x = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
        stderr = io.StringIO()
        with (
            warnings.catch_warnings(record=True) as caught,
            contextlib.redirect_stderr(stderr),
        ):
            warnings.simplefilter("always")
            bound = self._make_add().bind((x, y))
        return bound, stderr.getvalue(), caught

    @unittest.skipUnless(_HAS_SYMM_MEM_DETECT, "requires symm_mem.is_symm_mem_tensor")
    def test_non_distributed_kernel_not_constrained(self) -> None:
        x = torch.randn(4, device=DEVICE)
        self.assertFalse(kernel_uses_symm_mem((x, x)))
        bound, stderr, caught = self._bind_add()
        spec = bound.env.config_spec
        self.assertIn("flat", spec.allowed_pid_types)
        self.assertEqual(spec.max_num_sm_multiplier, 128)
        self.assertIsNone(bound.env.process_group_name)
        self.assertNotIn("ProcessGroupNameNotFound", stderr)
        self.assertFalse(any("max_num_sm" in str(w.message) for w in caught))

    @unittest.skipUnless(_HAS_SYMM_MEM_DETECT, "requires symm_mem.is_symm_mem_tensor")
    @skipIfRefEager("process-group resolution only happens in compiled mode")
    def test_symm_mem_kernel_still_distributed(self) -> None:
        with patch.object(symm_mem, "is_symm_mem_tensor", return_value=True):
            self.assertTrue(kernel_uses_symm_mem(([torch.randn(4, device=DEVICE)],)))
            with (
                patch(
                    "helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000
                ),
                patch("helion.runtime.get_num_sm", return_value=200),
            ):
                bound, stderr, _ = self._bind_add()
        spec = bound.env.config_spec
        self.assertNotIn("flat", spec.allowed_pid_types)
        self.assertNotIn("xyz", spec.allowed_pid_types)
        self.assertLess(spec.max_num_sm_multiplier, 128)
        self.assertIsNotNone(bound.env.process_group_name)
        self.assertIn("ProcessGroupNameNotFound", stderr)

    @unittest.skipUnless(_HAS_SYMM_MEM_DETECT, "requires symm_mem.is_symm_mem_tensor")
    @skipIfRefEager("barrier pid_type restriction only materializes in compiled mode")
    @skipIfTileIR("TileIR does not support barrier operations")
    def test_barrier_kernel_restricts_pid_but_keeps_multiplier(self) -> None:
        # A barrier inside a distributed process forces persistent pid_types but
        # is NOT a symm-mem signal, so max_num_sm_multiplier must stay unclamped.
        @helion.kernel(autotune_effort="none")
        def barrier_reduction(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                partial[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] * 2.0
            return out

        x = torch.randn([256, 512], device=DEVICE, dtype=torch.float32)
        self.assertFalse(kernel_uses_symm_mem((x,)))
        with (
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            bound = barrier_reduction.bind((x,))
        spec = bound.env.config_spec
        self.assertTrue(bound.env.has_barrier)
        self.assertNotIn("flat", spec.allowed_pid_types)
        self.assertNotIn("xyz", spec.allowed_pid_types)
        self.assertEqual(spec.max_num_sm_multiplier, 128)

    @unittest.skipUnless(_HAS_SYMM_MEM_DETECT, "requires symm_mem.is_symm_mem_tensor")
    @skipIfRefEager("process-group resolution only happens in compiled mode")
    def test_symm_mem_call_does_not_alias_cached_kernel(self) -> None:
        # Same kernel object and identical shapes: a symmetric-memory call must
        # not reuse the non-distributed kernel cached from an ordinary call, and
        # the ordinary key must still resolve back to it. See issue #3024.
        @helion.kernel(autotune_effort="none")
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)

        bound_normal = add.bind((x, y))
        self.assertIn("flat", bound_normal.env.config_spec.allowed_pid_types)

        with (
            patch.object(symm_mem, "is_symm_mem_tensor", return_value=True),
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            bound_symm = add.bind((x, y))
            symm_fast_key = add._fast_dispatch_key((x, y))

        self.assertIsNot(bound_normal, bound_symm)
        self.assertNotIn("flat", bound_symm.env.config_spec.allowed_pid_types)
        # the fast-dispatch key carries the distributed bit too
        self.assertNotEqual(add._fast_dispatch_key((x, y)), symm_fast_key)
        # the ordinary key still resolves back to the original compiled kernel
        self.assertIs(bound_normal, add.bind((x, y)))

    @unittest.skipUnless(_HAS_SYMM_MEM_DETECT, "requires symm_mem.is_symm_mem_tensor")
    @skipIfRefEager("process-group resolution only happens in compiled mode")
    def test_explicit_distributed_flag(self) -> None:
        @helion.kernel(autotune_effort="none", distributed=True)
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
        self.assertFalse(kernel_uses_symm_mem((x, y)))
        with (
            patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
            patch("helion.runtime.get_num_sm", return_value=200),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            bound = add.bind((x, y))
        spec = bound.env.config_spec
        self.assertNotIn("flat", spec.allowed_pid_types)
        self.assertLess(spec.max_num_sm_multiplier, 128)
        self.assertIsNotNone(bound.env.process_group_name)

    def test_distributed_flag_without_process_group(self) -> None:
        if not self._owns_pg:
            self.skipTest("needs exclusive control of the process group")
        if dist.is_initialized():
            dist.destroy_process_group()  # we created it; next test's setUp re-inits

        @helion.kernel(autotune_effort="none", distributed=True)
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)
        bound = add.bind((x, x))  # must not raise
        self.assertIn("flat", bound.env.config_spec.allowed_pid_types)
        self.assertIsNone(bound.env.process_group_name)

    def test_coordination_helpers_tolerate_none_pg(self) -> None:
        self.assertEqual(all_gather_object("x", process_group_name=None), ["x"])
        self.assertEqual(sync_object("x", process_group_name=None), "x")
        with sync_seed(process_group_name=None):
            pass


@onlyBackends(["triton"])
@skipIfNotCUDA()
@instantiate_parametrized_tests
class TestDistributedTileDependencies(TestCase):
    """Single-process compile checks of cross-rank tile dependencies."""

    def setUp(self) -> None:
        super().setUp()
        dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
        self.addCleanup(dist.destroy_process_group)

    def _world(self, world: int, *kernels: helion.Kernel) -> str:
        """Recreate the fake group with ``world`` ranks and reset ``kernels``."""
        dist.destroy_process_group()
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=0, world_size=world
        )
        # The new group reuses the old name, so a cached bind would be stale.
        for kernel in kernels:
            kernel.reset()
        return dist.group.WORLD.group_name

    def _accesses(
        self, kernel: helion.Kernel, args: tuple[object, ...], error: str | None = None
    ) -> tuple[tile_dependency.TileAccess, ...]:
        """Accesses the tile dependency graph of a bind is built from."""
        with (
            patch.object(
                tile_dependency,
                "build_tile_dependency_graph",
                wraps=tile_dependency.build_tile_dependency_graph,
            ) as build,
            (
                self.assertRaisesRegex(helion.exc.CrossLoopSchedulingError, error)
                if error is not None
                else contextlib.nullcontext()
            ),
        ):
            kernel.bind(args)
        return build.call_args.args[0]

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    def test_peer_views_name_their_owner_rank(self) -> None:
        group = dist.group.WORLD.group_name
        symm = torch.zeros(256, device=DEVICE)
        # The barrier orders only this rank's roots, so the peer edge remains.
        accesses = self._accesses(
            two_root_peer_read_kernel, (symm, group, group), "hl.barrier"
        )
        (store,) = (a for a in accesses if a.kind == "store" and a.root == 0)
        self.assertEqual(
            [
                (a.root, a.allocation_id, a.owner_rank)
                for a in accesses
                if a.owner_rank is not None
            ],
            [(1, store.allocation_id, 1), (1, store.allocation_id, 2)],
        )
        # Views taken in another group have no known owner.
        other_group = dist.new_group(list(range(4))).group_name
        with self.assertRaisesRegex(
            helion.exc.CrossLoopSchedulingError, "allocation identity"
        ):
            two_root_peer_read_kernel.bind((symm, other_group, group))

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    def test_bind_makes_no_collective_call(self) -> None:
        group = dist.group.WORLD.group_name
        pipelined_allreduce_kernel.reset()
        digests = []
        # Ranks may bind differently, so they compare digests at the first launch.
        with patch.object(dist, "all_gather_object") as gather:
            for dtype in (torch.float32, torch.float16):
                symm, x = torch.zeros(2, 256, device=DEVICE, dtype=dtype)
                bound = pipelined_allreduce_kernel.bind((symm, x, group, "inband", 4))
                code = bound.to_triton_code()
                digest = re.search(
                    r"_persistent_state_rank_digest='([0-9a-f]{16})'", code
                )
                assert digest is not None
                digests.append(digest[1])
        gather.assert_not_called()
        self.assertNotEqual(*digests)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_rules(self, world: int) -> None:
        group = self._world(world, inband_rule_kernel, diagonal_store_kernel)
        for broken_rule, dtype, expected in (
            ("", torch.float32, "use inband"),
            ("", torch.float16, "use inband"),
            ("R1", torch.float32, "use peer_counter (R1:"),
            ("", torch.float64, "use peer_counter (R2:"),
            ("R3", torch.float32, "use peer_counter (R3:"),
            ("R4", torch.float32, "use peer_counter (R4:"),
            ("R6", torch.float32, "use peer_counter (R6:"),
            # Exactly 4 MiB per rank stays inband; 2 ranks * 2**19 words does not.
            ("cap", torch.float32, "use inband"),
            ("R7", torch.float32, "use peer_counter (R7:"),
        ):
            n = {"cap": (1 << 22) // (8 * world), "R7": 1 << 19}.get(broken_rule, 256)
            symm, x = torch.zeros(2, n, device=DEVICE, dtype=dtype)
            with (
                self.subTest(broken_rule=broken_rule, dtype=dtype),
                self.assertLogs(tile_dependency.log, "INFO") as logs,
            ):
                inband_rule_kernel.bind((symm, x, group, broken_rule, world))
            (line,) = logs.output
            self.assertIn(f"on peers/symm {expected}", line)
        symm, x = torch.zeros(2, 16, 16, device=DEVICE)
        with self.assertLogs(tile_dependency.log, "INFO") as logs:
            diagonal_store_kernel.bind((symm, x, group))
        (line,) = logs.output
        self.assertIn("on peers/symm use peer_counter (R1:", line)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_peer_counters_order_cross_rank_edges(self, world: int) -> None:
        symm, x = torch.zeros(2, 256, device=DEVICE)
        group = self._world(world, inband_rule_kernel)
        bound = inband_rule_kernel.bind((symm, x, group, "R1", world))
        # Peer counters and inband data order either pipeline.
        self.assertEqual(
            bound.config_spec.cross_loop_pipeline.choices, ("static", "dynamic")
        )
        config = bound.config_spec.default_config().config
        state = "tile_dependency_peer_state"
        for pipeline in ("static", "dynamic"):
            code = bound.to_triton_code(
                helion.Config(
                    **{
                        **config,
                        "block_sizes": [64, 64, 64],
                        "cross_loop_pipeline": pipeline,
                    }
                )
            )
            # Root 1 adds its task's key on every rank and root 2 waits for its key
            # from each rank. Both roots' 4 + 4 workers publish to the done slot.
            keys = "tile_dependency_peer_keys"
            for expected in (
                f"_add_on_every_rank({keys}_ptrs, 0 + (",
                f"{world}, {world}, 'release')",
                f"tl.cast(tile_dependency_peer_epoch * {world}, tl.uint64)",
                f"_add_on_every_rank({state}_ptrs, 16, {world}, {world})",
                f"{state} + 16, tile_dependency_peer_epoch * {8 * world})",
            ):
                self.assertIn(expected, code)
            # A static launch takes its epoch from the slot after the done slot,
            # which worker 0 advances once every rank is done with the launch.
            launch = (
                f"tile_dependency_peer_epoch = tl.load({state} + 17) + 1",
                f"tl.store({state} + 17, tile_dependency_peer_epoch)",
            )
            for expected in launch:
                if pipeline == "static":
                    self.assertIn(expected, code)
                else:
                    self.assertNotIn(expected, code)
        bound = inband_rule_kernel.bind((symm, x, group, "", world))
        self.assertEqual(
            bound.config_spec.cross_loop_pipeline.choices, ("static", "dynamic")
        )

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_pushes_and_polls_data(self, world: int) -> None:
        symm, x = torch.zeros(2, 4000, device=DEVICE, dtype=torch.bfloat16)
        group = self._world(world)
        bound = pipelined_allreduce_kernel.bind((symm, x, group, "inband", world))
        config = bound.config_spec.default_config().config
        # A thread holds 2 words of a 1024 block at 16 warps, and 1 of a 64 block.
        for block_size, num_warps, pack in ((1024, 16, 2), (64, 4, 1)):
            code = bound.to_triton_code(
                helion.Config(
                    **{
                        **config,
                        "block_sizes": [block_size, block_size],
                        "num_warps": num_warps,
                    }
                )
            )
            # Root 0 pushes each word to every rank, root 1 polls all mailboxes
            # in one asm, and the data needs no peer counters or done barrier.
            self.assertEqual(code.count("st.relaxed.sys.global.u64"), world)
            self.assertIn("tl.store(symm + ", code)
            self.assertEqual(code.count("@p bra SPIN"), 1)
            # Outside WS loops, every rank's words reload together on a CTA vote.
            self.assertEqual(code.count("while inband_stale != 0:"), 1)
            self.assertIn(f"bar.red.or.pred q, 0, {32 * num_warps}, p;", code)
            words = ", ".join(["tl.uint64"] * world)
            self.assertIn(f"dtype=({words}), is_pure=False, pack={pack})", code)
            # Two parities of a word per rank and element, the done and launch slots.
            self.assertIn(f"(x, {2 * world * 4000 + 2}, torch.uint64, True)", code)
            self.assertNotIn("_wait_at_least", code)
            self.assertNotIn("_add_on_every_rank", code)
        # Under dynamic tickets the poller waits only in band, yet still warms
        # its code with a dry pass.
        code = bound.to_triton_code({**config, "cross_loop_pipeline": "dynamic"})
        self.assertEqual(code.count("for tile_dependency_dry_pass in"), 1)
        # A push may be the kernel's first tensor access.
        bound = pipelined_allreduce_kernel.bind((symm, x, group, "constant", world))
        code = bound.to_triton_code()
        self.assertIn("st.relaxed.sys.global", code)
        self.assertIn("tl.store(symm + ", code)
        # Debug output touches no memory, so it does not block the analysis.
        bound = pipelined_allreduce_kernel.bind((symm, x, group, "print", world))
        self.assertIn("tl.device_print", bound.to_triton_code())
        # A poll's loop runs one stage: Triton would pipeline the first read
        # into an early, weak cp.async.
        bound = pipelined_allreduce_kernel.bind((symm, x, group, "loop", world))
        config = bound.config_spec.default_config().config
        code = bound.to_triton_code({**config, "range_num_stages": [0, 0, 3]})
        self.assertIn("_BLOCK_SIZE_2, num_stages=1)", code)
        self.assertNotIn("num_stages=3", code)
        # Aligned tiles pack two bf16 per word and push 16-byte word pairs; each
        # reader polls whole words and unpacks both lanes in-thread.
        symm, x = torch.zeros(2, 4096, device=DEVICE, dtype=torch.bfloat16)
        code = pipelined_allreduce_kernel.bind(
            (symm, x, group, "inband", world)
        ).to_triton_code()
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), world)
        self.assertNotIn("st.relaxed.sys.global.u64", code)
        self.assertIn("<< 16 | tile_dependency_peer_epoch << 32", code)
        self.assertIn("tl.reshape(tl.join(tl.cast(tl.cast(inband_word", code)
        self.assertNotIn("% 2, tl.uint64) * 16", code)
        self.assertIn(f"(x, {2 * world * 2048 + 2}, torch.uint64, True)", code)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_chains_exchanges(self, world: int) -> None:
        symm, x = torch.zeros(2, 4096, device=DEVICE)
        moment = torch.zeros(1, device=DEVICE)
        group = self._world(world, chained_exchange_kernel)
        code = chained_exchange_kernel.bind((symm, moment, x, group)).to_triton_code()
        # Both buffers go in band, each with its own push and poll. The even
        # symm tiles push word pairs; the one-element moment pushes single words.
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), world)
        self.assertEqual(code.count("st.relaxed.sys.global.u64"), world)
        self.assertEqual(code.count("@p bra SPIN"), 2)
        self.assertNotIn("_wait_at_least", code)
        self.assertNotIn("_add_on_every_rank", code)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_pair_address_from_axis_vectors(self, world: int) -> None:
        symm = torch.zeros(20, 256, device=DEVICE, dtype=torch.bfloat16)
        x = torch.zeros_like(symm)
        group = self._world(world, matrix_exchange_kernel)
        bound = matrix_exchange_kernel.bind((symm, x, group))
        config = bound.config_spec.default_config().config
        code = bound.to_triton_code({**config, "block_sizes": [8, 64, 8, 64]})
        # Pair addresses and the row mask come from per-axis vectors: only the
        # column vector is split, never the full offset or mask tile.
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), world)
        self.assertIn("tl.reshape(indices_1, [_BLOCK_SIZE_1 // 4, 2, 2])", code)
        self.assertIn("(indices_0[:, None] * 256 + inband_lane_10[None, :] * 1", code)
        self.assertIn("tl.cast(mask_0[:, None], tl.int32), [_BLOCK_SIZE_0, ", code)
        self.assertNotIn("tl.reshape(inband_offset", code)
        # The masked poll reads the same words and unpacks them in-thread.
        self.assertIn("tl.reshape(tl.join(", code)
        self.assertNotIn("% 2, tl.uint64) * 16", code)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_done_barrier_waits_on_fallback_roots(self, world: int) -> None:
        symm, x = torch.zeros(2, 4096, device=DEVICE)
        # R2 rejects the 8-byte moment, so it falls back to peer counters.
        moment = torch.zeros(1, device=DEVICE, dtype=torch.float64)
        group = self._world(world, chained_exchange_kernel)
        bound = chained_exchange_kernel.bind((symm, moment, x, group))
        config = bound.config_spec.default_config().config
        code = bound.to_triton_code({**config, "block_sizes": [1024, 1024, 1, 512]})
        # Only roots 2 and 3 publish to the done slot, and the last ticket waits
        # for their 1 + 4096 / 512 tasks, not also the in-band roots' 4 + 4.
        done = f"tile_dependency_peer_state + {2 * world * 4096 + 16}"
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), world)
        self.assertEqual(code.count("_add_on_every_rank"), 3)
        self.assertIn(f"{done}, tile_dependency_peer_epoch * {9 * world})", code)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_done_barrier_skips_ended_roots(self, world: int) -> None:
        symm, x = torch.zeros(2, 4096, device=DEVICE, dtype=torch.float64)
        moment = torch.zeros(1, device=DEVICE)
        group = self._world(world, chained_exchange_kernel)
        bound = chained_exchange_kernel.bind((symm, moment, x, group))
        config = bound.config_spec.default_config().config
        code = bound.to_triton_code({**config, "block_sizes": [1024, 1024, 1, 512]})
        # R2 rejects the 8-byte symm, so root 1 waits on root 0's keyed counters.
        # A barrier orders root 1 before root 2, whose in-band moment every rank
        # polls in full, so root 1 ends before any launch and no root needs done.
        self.assertEqual(code.count("st.relaxed.sys.global.u64"), world)
        self.assertEqual(code.count("_add_on_every_rank"), 1)
        self.assertIn("_add_on_every_rank(tile_dependency_peer_keys_ptrs", code)
        self.assertNotIn("_wait_at_least", code)
        # Without a done barrier each static worker counts its own launches.
        launch = "tile_dependency_peer_launch + tl.program_id(0)"
        self.assertIn(f"tile_dependency_peer_epoch = tl.load({launch}) + 1", code)
        self.assertIn(f"tl.store({launch}, tile_dependency_peer_epoch)", code)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_static_index_store(self, world: int) -> None:
        # A static index is no tile axis: symm[0, tile] fills a [1, N] buffer.
        symm, x = torch.zeros(1, 256, device=DEVICE), torch.zeros(256, device=DEVICE)
        group = self._world(world, row_exchange_kernel)
        with self.assertLogs(tile_dependency.log, "INFO") as logs:
            bound = row_exchange_kernel.bind((symm, x, group))
        (line,) = logs.output
        self.assertIn("on peers/symm use inband", line)
        code = bound.to_triton_code()
        self.assertEqual(code.count("st.relaxed.sys.global"), world)
        self.assertNotIn("_wait_at_least", code)
        # In a [2, N] buffer the same store fills only row 0, which R2 rejects.
        symm = torch.zeros(2, 256, device=DEVICE)
        with self.assertLogs(tile_dependency.log, "INFO") as logs:
            row_exchange_kernel.bind((symm, x, dist.group.WORLD.group_name))
        (line,) = logs.output
        self.assertIn("on peers/symm use peer_counter (R2:", line)

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    @parametrize("world", (2, 4, 8))
    def test_inband_scatter_records_rows(self, world: int) -> None:
        symm = torch.zeros(32, 64, device=DEVICE, dtype=torch.bfloat16)
        x = torch.zeros(4, 8, 64, device=DEVICE, dtype=torch.bfloat16)
        rows = torch.zeros(4, 8, device=DEVICE, dtype=torch.int32)
        count = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        group = self._world(world, scatter_exchange_kernel)
        with self.assertLogs(tile_dependency.log, "INFO") as logs:
            bound = scatter_exchange_kernel.bind((symm, x, rows, count, group))
        (line,) = logs.output
        self.assertIn("on peers/symm use inband", line)
        # Only a static launch marks the guard-skipped tasks.
        self.assertEqual(bound.config_spec.cross_loop_pipeline.choices, ("static",))
        code = bound.to_triton_code()
        # Each task records one tagged key per row, then pushes a done word to
        # every rank; skipped tasks push all-ones done words instead.
        self.assertEqual(code.count("inband_record = "), 1)
        self.assertEqual(code.count("inband_done = "), 1)
        self.assertIn("tl.cast(tl.sum(inband_record), tl.uint64)", code)
        self.assertIn("tile_dependency_peer_epoch << 32 | 4294967295", code)
        # After a long spin, readers scan each source's records and done words and
        # copy rows no task wrote from the peer's buffer into the mailbox.
        self.assertEqual(code.count("_scatter_cover("), world)
        self.assertIn("tl.pointer_type(tl.uint32)) + inband_offset_", code)
        self.assertIn("tl.store(inband_address_", code)
        self.assertIn("_hold_after_fallback(tile_dependency_peer_state", code)
        # The record key must index the row block the reader computes.
        self.assertIn("% 64 // 32 * 1 + inband_offset_1 // 64 * 2", code)

    @skipIfRefEager("peer views are recorded only in compiled mode")
    def test_peer_views_require_allocation_base(self) -> None:
        group = dist.group.WORLD.group_name
        symm = torch.zeros(16, 16, device=DEVICE)

        def guards(kernel: helion.Kernel, *args: object) -> list[object]:
            env = kernel.bind((symm, *args, group)).env
            return [
                guard
                for key, guard in env.runtime_input_specializations.items()
                if key.startswith("symmetric_allocation_base")
            ]

        for kernel, start, absolute in itertools.product(
            (peer_sum_kernel, dynamic_peer_sum_kernel), (0, 16), (False, True)
        ):
            # A view past the owner's start has no known peer counterpart.
            self.assertEqual(len(guards(kernel, start, absolute)), int(start == 0))
        # Torch's peer views ignore a view's offset, so they would not line up.
        shifted = torch.zeros(257, device=DEVICE)[1:].view(16, 16)
        with self.assertRaisesRegex(helion.exc.InvalidAPIUsage, "symmetric allocation"):
            peer_sum_kernel.bind((shifted, 0, False, group))

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    def test_peer_view_ssa_copies_keep_their_owner(self) -> None:
        symm = torch.zeros(64, 64, device=DEVICE)
        accesses = self._accesses(
            peer_read_through_copy_kernel, (symm, dist.group.WORLD.group_name)
        )
        (store,) = (a for a in accesses if a.kind == "store" and a.root == 0)
        self.assertEqual(
            {(a.allocation_id, a.owner_rank) for a in accesses if a.root == 1},
            {(store.allocation_id, 1), (accesses[-1].allocation_id, None)},
        )

    @skipIfRefEager("tile dependencies are built only in compiled mode")
    def test_hidden_accesses_are_rejected(self) -> None:
        symm = torch.zeros(64, dtype=torch.bfloat16, device=DEVICE)
        offsets = torch.zeros(4, dtype=torch.int64, device=DEVICE)
        # Other ranks write this rank's copy too, so its hidden loads count.
        for remote_barrier, local_source, op in (
            (True, False, "remote_barrier"),
            (False, False, "load_bfloat16_x16_to_float16"),
            (False, True, "load_bfloat16_x16_to_float16"),
        ):
            with self.assertRaisesRegex(
                helion.exc.CrossLoopSchedulingError, f"{op} may access another rank"
            ):
                two_root_hidden_access_kernel.bind(
                    (
                        symm,
                        offsets,
                        dist.group.WORLD.group_name,
                        remote_barrier,
                        local_source,
                    )
                )


if __name__ == "__main__":
    run_tests()
