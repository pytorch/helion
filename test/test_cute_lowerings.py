from __future__ import annotations

import ast
import contextlib
import dataclasses
import difflib
import operator
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph

import helion
from helion import exc
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.aten_lowering import _pallas_argreduce
from helion._compiler.aten_lowering import _should_use_cute_argreduce_lowering
from helion._compiler.aten_lowering import _triton_argreduce
from helion._compiler.aten_lowering import codegen_iota_cute
from helion._compiler.aten_lowering import codegen_mm_cute
from helion._compiler.aten_lowering import codegen_squeeze_cute
from helion._compiler.aten_lowering import codegen_stack_cute
from helion._compiler.aten_lowering import codegen_unsqueeze_cute
from helion._compiler.aten_lowering import codegen_view_cute
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import PallasBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.backend import _detect_mma_loop
from helion._compiler.backend import _loop_contains_matmul
from helion._compiler.backend import _loop_may_use_mma
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.argreduce import codegen_cute_tile_argreduce
from helion._compiler.cute.cute_mma import _TCGEN05_CLUSTER_LEADER_PREDICATE
from helion._compiler.cute.cute_mma import _build_initial_prefetch_if
from helion._compiler.cute.cute_mma import _build_kloop_non_pipeline_producer_if
from helion._compiler.cute.cute_mma import _build_kloop_non_pipeline_release_if
from helion._compiler.cute.cute_mma import _build_kloop_pipeline_consumer_if
from helion._compiler.cute.cute_mma import _build_kloop_pipeline_consumer_prefetch_stmts
from helion._compiler.cute.cute_mma import _build_kloop_pipeline_producer_if
from helion._compiler.cute.cute_mma import _build_kloop_pipeline_release_if
from helion._compiler.cute.cute_mma import _build_tcgen05_mma_accumulate_reset_stmt
from helion._compiler.cute.cute_mma import _build_tcgen05_mma_issue_stmt
from helion._compiler.cute.cute_mma import _choose_mma_impl
from helion._compiler.cute.cute_mma import _emit_sched_pipeline_setup
from helion._compiler.cute.cute_mma import _get_mma_k_loop_info
from helion._compiler.cute.cute_mma import _InitialPrefetchTmaArgs
from helion._compiler.cute.cute_mma import _make_tcgen05_layout_plan_setup
from helion._compiler.cute.cute_mma import _mma_result_can_be_deferred
from helion._compiler.cute.cute_mma import _new_tcgen05_layout_plan
from helion._compiler.cute.cute_mma import _new_tcgen05_sched_pipeline_plan
from helion._compiler.cute.cute_mma import _PerKiterTmaArgs
from helion._compiler.cute.cute_mma import _tcgen05_ab_stage_count
from helion._compiler.cute.cute_mma import _tcgen05_epi_warp_count
from helion._compiler.cute.cute_mma import _tcgen05_root_m_threads
from helion._compiler.cute.cute_mma import _tcgen05_tmem_barrier_thread_count
from helion._compiler.cute.cute_mma import _trace_mma_to_store_dtype
from helion._compiler.cute.cute_mma import can_codegen_cute_mma_aten
from helion._compiler.cute.cute_reshape import _get_dim_local_coord
from helion._compiler.cute.cute_reshape import codegen_cute_permute
from helion._compiler.cute.cute_reshape import codegen_cute_reshape
from helion._compiler.cute.indexing import CutePackedAffineLoad
from helion._compiler.cute.indexing import CutePackedTerms
from helion._compiler.cute.indexing import CuteShapeChainView
from helion._compiler.cute.indexing import is_cute_shape_chain_target
from helion._compiler.cute.indexing import match_cute_affine_range_iota
from helion._compiler.cute.indexing import match_cute_duplicate_stack_reshape_rhs
from helion._compiler.cute.indexing import match_cute_stack_reshape_rhs
from helion._compiler.cute.matmul_fallback import _emit_cute_grouped_sum_reduction
from helion._compiler.cute.matmul_utils import cute_resolve_active_block_id
from helion._compiler.cute.matmul_utils import cute_resolve_active_matmul_k_block_id
from helion._compiler.cute.matmul_utils import cute_static_k_invariant_extent
from helion._compiler.cute.matmul_utils import cute_supports_scalar_matmul_fallback
from helion._compiler.cute.strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from helion._compiler.cute.strategies import Tcgen05WarpSpec
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from helion._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from helion._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import GraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import collect_cute_half_atomic_output_promotions
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import BlockReductionStrategy
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._compiler.tile_strategy import DeviceGridState
from helion._compiler.tile_strategy import DeviceLoopState
from helion._compiler.tile_strategy import _lane_loop_iter
from helion._compiler.variable_origin import NameOrigin
from helion._compiler.variable_origin import TileBeginOrigin
from helion._testing import DEVICE
from helion._testing import default_cute_mma_support
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl
from helion.language import _tracing_ops
from helion.language._tracing_ops import _mask_to
from helion.language._tracing_ops import _new_var
from helion.language.matmul_ops import _cute_dot_outer_accumulates_result
from helion.language.memory_ops import _codegen_cute_store_permute_lane_loops
from helion.language.memory_ops import _cute_combined_mask
from helion.language.memory_ops import _cute_index_exprs
from helion.language.memory_ops import _maybe_codegen_cute_packed_affine_lhs_load
from helion.language.memory_ops import load
from helion.runtime import _append_cute_wrapper_plan

# Golden file pinning the byte-identical generated CuTe for the
# retained ROLE_LOCAL_MONOLITHIC seed (cute_plan.md §6.2 pin test #1
# and §10.1 canonical benchmark seed). Lives at
# ``test/golden/tcgen05_role_local_monolithic_4096_bf16.py.expected``.
# The kernel definition is in
# ``test/golden/_tcgen05_role_local_monolithic_4096_bf16_kernel.py``;
# its file path appears in ``src[<file>:<line>]`` comments embedded
# in the generated kernel, so both files must move together if
# either is renamed.
TCGEN05_ROLE_LOCAL_MONOLITHIC_GOLDEN_PATH = (
    Path(__file__).parent
    / "golden"
    / "tcgen05_role_local_monolithic_4096_bf16.py.expected"
)


def _make_tcgen05_persistent_config(**overrides: object) -> helion.Config:
    """Build a ``helion.Config`` for the tcgen05 + TMA pipeline path."""
    defaults: dict[str, object] = {
        "block_sizes": [128, 256, 16],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "num_stages": 2,
        "num_warps": 8,
        "pid_type": "flat",
        "tcgen05_cluster_m": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
    }
    defaults.update(overrides)
    return helion.Config(**defaults)  # type: ignore[arg-type]


def _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
    **overrides: object,
) -> helion.Config:
    """Build the selected clustered CtaGroup.ONE bridge proof config."""
    defaults: dict[str, object] = {
        "block_sizes": [128, 256, 128],
        "l2_groupings": [4],
        "num_warps": 4,
        "num_sm_multiplier": 1,
        "pid_type": "persistent_interleaved",
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        "tcgen05_cluster_m": 2,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 1,
        "tcgen05_c_stages": 4,
        "tcgen05_num_epi_warps": 4,
        "range_flattens": [None, None],
        "range_multi_buffers": [None, None],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_unroll_factors": [0, 0],
    }
    defaults.update(overrides)
    return _make_tcgen05_persistent_config(**defaults)


def _make_tcgen05_large_bn_proof_config(**overrides: object) -> helion.Config:
    """Build the first G4 larger-BN admission proof config."""
    defaults: dict[str, object] = {
        "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
        "l2_groupings": [1],
        "num_warps": 4,
        "pid_type": TCGEN05_LARGE_BN_PROOF_PID_TYPE,
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        "tcgen05_cluster_m": TCGEN05_LARGE_BN_PROOF_CLUSTER_M,
        **dict(TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS),
        "tcgen05_num_epi_warps": 4,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True,
    }
    defaults.update(overrides)
    return _make_tcgen05_persistent_config(**defaults)


class _FakeBlockSize:
    def __init__(
        self, size: int, *, block_id: int | None = None, reduction: bool = False
    ) -> None:
        self.size = size
        self.block_id = block_id
        self.reduction = reduction

    def from_config(self, config: object) -> int:
        return self.size

    def from_config_assert(self, config: object) -> int:
        return self.size

    def is_flattened(self, config: object) -> bool:
        return False


class _FakeDeviceFunction:
    def __init__(self) -> None:
        self.config = object()
        self._counts: dict[str, int] = {}
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}

    def new_var(self, prefix: str, **kwargs: object) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}_{self._counts[prefix]}"


class _FakeGenerateAST:
    def __init__(
        self,
        active_block_ids: set[int],
        current_grid_state: object | None = None,
    ) -> None:
        self.device_function = _FakeDeviceFunction()
        self.active_device_loops = {
            block_id: [
                SimpleNamespace(
                    strategy=_FakeLoopStrategy([block_id]),
                    block_thread_axes={block_id: block_id},
                )
            ]
            for block_id in active_block_ids
        }
        self.current_grid_state = current_grid_state
        self.statements: list[ast.AST] = []

    def add_statement(self, stmt: ast.AST) -> None:
        self.statements.append(stmt)

    def index_var(self, block_idx: int) -> str:
        return f"indices_{block_idx}"

    def offset_var(self, block_idx: int) -> str:
        return f"offset_{block_idx}"


class _FakeLoopStrategy:
    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def offset_var(self, block_idx: int) -> str:
        return f"offset_{block_idx}"

    def index_var(self, block_idx: int) -> str:
        return f"indices_{block_idx}"

    def mask_var(self, block_idx: int) -> None:
        return None


class _FakeMaskedLoopStrategy(_FakeLoopStrategy):
    def mask_var(self, block_idx: int) -> str:
        return f"mask_{block_idx}"


class _FakeGenerateASTForLaneStore:
    def __init__(self, grid_state: DeviceGridState) -> None:
        self.current_grid_state = grid_state
        self.active_device_loops = {}
        self.statements: list[ast.AST] = []
        self.device_function = SimpleNamespace(
            config=object(),
            new_var=lambda prefix: prefix,
            tensor_arg=lambda tensor: SimpleNamespace(name="out"),
        )

    def add_statement(self, stmt: ast.AST) -> None:
        self.statements.append(stmt)

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "tmp") -> ast.AST:
        return expr


class _FakeMaskCodegen:
    def __init__(self, strategy: _FakeLoopStrategy, block_ids: set[int]) -> None:
        self.active_device_loops = {
            block_id: [SimpleNamespace(strategy=strategy)] for block_id in block_ids
        }

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "tmp") -> ast.AST:
        return SimpleNamespace(id=f"{prefix}_0")


class _FakeCuteReductionCodegen:
    def __init__(self) -> None:
        self.device_function = _FakeDeviceFunction()
        self.active_device_loops = {
            0: [
                SimpleNamespace(
                    thread_axis_sizes={0: 3},
                    block_thread_axes={0: 0},
                )
            ],
            1: [
                SimpleNamespace(
                    thread_axis_sizes={1: 16},
                    block_thread_axes={1: 1},
                )
            ],
        }
        self.current_grid_state = None
        self.max_thread_block_dims = (3, 16, 1)
        self.statements: list[object] = []

    def add_statement(self, stmt: object) -> None:
        self.statements.append(stmt)


class _FakeArgreduceTileStrategy:
    def shape_str(self, shape: list[object]) -> str:
        return f"[{', '.join(map(str, shape))}]"

    def shape_dims(self, shape: list[object]) -> list[str]:
        return [str(dim) for dim in shape]


def _fake_env(
    block_ids_by_size: dict[int, int | None],
) -> object:
    block_sizes = {
        block_id: _FakeBlockSize(size)
        for size, block_id in block_ids_by_size.items()
        if block_id is not None
    }
    backend = SimpleNamespace(
        dtype_str=lambda dtype: (
            "cutlass.Int32" if dtype == torch.int32 else "cutlass.Float16"
        ),
        cast_expr=lambda expr, dtype_str: f"{dtype_str}({expr})",
    )
    return SimpleNamespace(
        backend=backend,
        block_sizes=block_sizes,
        canonical_block_id=lambda block_id: block_id,
        get_block_id=lambda size: block_ids_by_size.get(int(size)),
        known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
        resolve_block_id=lambda size: block_ids_by_size.get(int(size)),
    )


def _fake_device_loop(block_id: int) -> DeviceLoopState:
    return DeviceLoopState(
        strategy=_FakeLoopStrategy([block_id]),
        block_id_to_info={},
        for_node=ast.For(
            target=ast.Name(id=f"i_{block_id}", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=1)],
                keywords=[],
            ),
            body=[],
            orelse=[],
            type_comment=None,
        ),
        inner_statements=[],
        block_thread_axes={block_id: 0},
    )


@onlyBackends(["cute"])
class TestCuteLowerings(unittest.TestCase):
    def _argreduce_ctx(self, inp: torch.fx.Node) -> object:
        return SimpleNamespace(
            env={inp: ast.Name(id="x", ctx=ast.Load())},
            cg=SimpleNamespace(
                device_function=SimpleNamespace(
                    tile_strategy=_FakeArgreduceTileStrategy()
                )
            ),
        )

    def _assert_tma_store_epilogue_order(
        self, code: str, *, require_tail: bool = True
    ) -> None:
        acquire = "tcgen05_c_pipeline.producer_acquire()"
        t2r_copy = "cute.copy(tcgen05_tiled_copy_t2r"
        c_buffer = "tcgen05_c_buffer = "
        r2s_source_store = "tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        r2s_copy = "cute.copy(tcgen05_tiled_copy_r2s"
        tma_copy = "cute.copy(tcgen05_tma_store_atom"
        commit = "tcgen05_c_pipeline.producer_commit()"
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        acc_release = (
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"
        )
        acc_advance = "tcgen05_acc_consumer_state.advance()"
        acc_advance_inline = (
            "tcgen05_acc_consumer_state._count = "
            "tcgen05_acc_consumer_state._count + cutlass.Int32(1)"
        )
        gmem_tile = "tcgen05_gC = cute.local_tile("
        subtile_loop = "for _tcgen05_subtile in cutlass.range("
        later_subtile_guard = "if _tcgen05_subtile != 0"
        barrier = "tcgen05_epilog_sync_barrier.arrive_and_wait()"
        shared_fence = "cute.arch.fence_view_async_shared()"
        tail = "tcgen05_c_pipeline.producer_tail()"

        self.assertEqual(code.count(acquire), 2, code)
        self.assertEqual(code.count(acc_wait), 1, code)
        self.assertEqual(code.count(acc_release), 1, code)
        self.assertEqual(
            code.count(acc_advance) + code.count(acc_advance_inline), 1, code
        )
        first_acquire_pos = code.find(acquire)
        gmem_tile_pos = code.find(gmem_tile, first_acquire_pos)
        subtile_loop_pos = code.find(subtile_loop, gmem_tile_pos)
        later_subtile_guard_pos = code.find(later_subtile_guard, subtile_loop_pos)
        acquire_pos = code.find(acquire, first_acquire_pos + len(acquire))
        acc_wait_pos = code.find(acc_wait, acquire_pos)
        t2r_pos = code.find(t2r_copy)
        acc_release_pos = code.find(acc_release)
        acc_advance_pos = code.find(acc_advance, acc_release_pos)
        if acc_advance_pos < 0:
            acc_advance_pos = code.find(acc_advance_inline, acc_release_pos)
        r2s_source_store_pos = code.find(r2s_source_store, acc_release_pos)
        c_buffer_pos = code.find(c_buffer, acquire_pos)
        r2s_pos = code.find(r2s_copy)
        tma_pos = code.find(tma_copy)
        commit_pos = code.find(commit)
        for needle, pos in (
            (acquire, acquire_pos),
            (acc_wait, acc_wait_pos),
            (t2r_copy, t2r_pos),
            (acc_release, acc_release_pos),
            ("acc consumer state advance", acc_advance_pos),
            (c_buffer, c_buffer_pos),
            (r2s_source_store, r2s_source_store_pos),
            (r2s_copy, r2s_pos),
            (tma_copy, tma_pos),
            (commit, commit_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        for needle, pos in (
            ("first C producer acquire", first_acquire_pos),
            (gmem_tile, gmem_tile_pos),
            (subtile_loop, subtile_loop_pos),
            (later_subtile_guard, later_subtile_guard_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        tail_pos = code.find(tail, commit_pos)
        if require_tail:
            self.assertGreaterEqual(tail_pos, 0, f"Missing {tail!r} in:\n{code}")
        else:
            # Callers pass a bounded role-local while source segment here; a
            # tail in that segment would drain once per scheduler-recycled tile.
            self.assertNotIn(tail, code)
            tail_pos = len(code)
        epilogue_slice = code[acquire_pos:tail_pos]
        self.assertEqual(epilogue_slice.count(shared_fence), 1, code)
        self.assertNotIn("cute.arch.fence_proxy('async.shared'", epilogue_slice)
        first_barrier_pos = code.find(barrier, acquire_pos)
        self.assertGreaterEqual(first_barrier_pos, 0, code)
        second_barrier_pos = code.find(barrier, first_barrier_pos + len(barrier))
        self.assertGreaterEqual(second_barrier_pos, 0, code)
        shared_fence_pos = code.find(shared_fence, r2s_pos)
        self.assertGreaterEqual(shared_fence_pos, 0, code)
        self.assertLess(first_acquire_pos, gmem_tile_pos, code)
        self.assertLess(gmem_tile_pos, subtile_loop_pos, code)
        self.assertLess(subtile_loop_pos, later_subtile_guard_pos, code)
        self.assertLess(later_subtile_guard_pos, acquire_pos, code)
        self.assertLess(acquire_pos, acc_wait_pos, code)
        self.assertLess(acc_wait_pos, t2r_pos, code)
        self.assertLess(t2r_pos, acc_release_pos, code)
        self.assertLess(acquire_pos, first_barrier_pos, code)
        self.assertLess(acc_release_pos, first_barrier_pos, code)
        self.assertLess(acc_release_pos, r2s_source_store_pos, code)
        self.assertLess(r2s_source_store_pos, first_barrier_pos, code)
        self.assertLess(first_barrier_pos, c_buffer_pos, code)
        self.assertLess(c_buffer_pos, r2s_pos, code)
        self.assertLess(r2s_pos, shared_fence_pos, code)
        self.assertLess(shared_fence_pos, second_barrier_pos, code)
        self.assertLess(second_barrier_pos, tma_pos, code)
        self.assertLess(tma_pos, commit_pos, code)
        self.assertLess(commit_pos, acc_advance_pos, code)
        self.assertLess(acc_advance_pos, tail_pos, code)
        loop_tail = code[commit_pos:tail_pos]
        self.assertNotIn(barrier, loop_tail, code)
        self.assertEqual(code[acquire_pos:tail_pos].count(barrier), 2, code)

    def _role_local_while_source_for_predicate(
        self, code: str, tree: ast.AST, role_predicate: str
    ) -> tuple[str, int, int]:
        matches: list[ast.While] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.If) and ast.unparse(node.test) == role_predicate
            ):
                continue
            for role_child in ast.walk(node):
                if isinstance(
                    role_child, ast.While
                ) and "tcgen05_role_local" in ast.unparse(role_child.test):
                    matches.append(role_child)
        self.assertEqual(
            len(matches),
            1,
            "Expected exactly one role-local while for "
            f"{role_predicate!r}. Generated code:\n{code}",
        )
        role_src = ast.get_source_segment(code, matches[0])
        self.assertIsNotNone(
            role_src,
            "Expected parsed role-local while to preserve source extent. "
            "Generated code:\n" + code,
        )
        assert role_src is not None
        role_start = code.index(role_src)
        return role_src, role_start, role_start + len(role_src)

    def _assert_role_local_c_store_pipeline_lifetime(
        self, code: str, tree: ast.AST, role_predicate: str
    ) -> str:
        create = "tcgen05_c_pipeline = cutlass.pipeline.PipelineTmaStore.create("
        tail = "tcgen05_c_pipeline.producer_tail()"
        dealloc = "num_allocated_columns=tcgen05_acc_tmem_cols"
        invariant_setup = [
            "tcgen05_kernel_desc = type('Tcgen05KernelDesc'",
            (
                "tcgen05_store_epi_tile = "
                "cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape("
            ),
            "tcgen05_sD_layout = cutlass.utils.blackwell_helpers.make_smem_layout_epi(",
            "tcgen05_sD_ptr = cute.arch.alloc_smem(",
            "tcgen05_sD = cute.make_tensor(",
            (
                "tcgen05_tAcc = "
                "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
            ),
        ]

        role_src, role_start, role_end = self._role_local_while_source_for_predicate(
            code, tree, role_predicate
        )
        self.assertNotIn("PipelineTmaStore.create", role_src)
        self.assertNotIn(tail, role_src)
        create_pos = code.index(create)
        tail_pos = code.index(tail)
        dealloc_pos = code.index(dealloc)
        self.assertLess(create_pos, role_start)
        for needle in invariant_setup:
            self.assertNotIn(needle, role_src)
            self.assertLess(code.index(needle), role_start)
        self.assertLess(role_end, tail_pos)
        self.assertLess(tail_pos, dealloc_pos)
        return role_src

    def test_mma_k_loop_selection_uses_reduction_block(self) -> None:
        env = _fake_env({32: 0, 64: 1, 16: 2, 7: 3})
        r_loop = _fake_device_loop(3)
        k_loop = _fake_device_loop(1)
        cg = SimpleNamespace(
            active_device_loops={3: [r_loop], 1: [k_loop]},
            device_function=_FakeDeviceFunction(),
        )

        loop_info = _get_mma_k_loop_info(
            cg,
            env,
            torch.empty(32, 64),
            torch.empty(64, 16),
        )

        self.assertIsNotNone(loop_info)
        assert loop_info is not None
        device_loop, k_block_id, k_offset_var, bk = loop_info
        self.assertIs(device_loop, k_loop)
        self.assertEqual(k_block_id, 1)
        self.assertEqual(k_offset_var, "offset_1")
        self.assertEqual(bk, 64)

    def test_mma_result_defer_only_when_node_exits_loop(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(addmm)
        self.assertTrue(_mma_result_can_be_deferred(addmm))

        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        relu = graph.call_function(torch.ops.aten.relu.default, args=(addmm,))
        graph.output(relu)
        self.assertFalse(_mma_result_can_be_deferred(addmm))

    def test_mma_k_loop_selection_prefers_node_graph_when_symbols_do_not_resolve(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        dot = graph.call_function(torch.matmul, args=(lhs, rhs))
        graph.output(dot)

        k_loop = _fake_device_loop(2)
        outer_loop = _fake_device_loop(3)
        env = SimpleNamespace(
            block_sizes={2: _FakeBlockSize(64), 3: _FakeBlockSize(7)},
            known_equal=lambda lhs, rhs: False,
            resolve_block_id=lambda size: None,
        )
        cg = SimpleNamespace(
            active_device_loops={3: [outer_loop], 2: [k_loop]},
            codegen_graphs=[
                ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[2])
            ],
            device_function=_FakeDeviceFunction(),
        )

        loop_info = _get_mma_k_loop_info(
            cg,
            env,
            torch.empty(32, 64),
            torch.empty(64, 16),
            fx_node=dot,
        )

        self.assertIsNotNone(loop_info)
        assert loop_info is not None
        device_loop, k_block_id, k_offset_var, bk = loop_info
        self.assertIs(device_loop, k_loop)
        self.assertEqual(k_block_id, 2)
        self.assertEqual(k_offset_var, "offset_2")
        self.assertEqual(bk, 64)

    def test_detect_mma_loop_allows_serialized_k_reduction_loop(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(acc)

        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=graph, node_args=[acc], block_ids=[2]
                )
            ]
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertTrue(
                _detect_mma_loop(
                    fn,
                    [2],
                    block_sizes=[16],
                    num_threads_config=[1],
                )
            )

    def test_detect_mma_loop_rejects_serialized_root_grid_axes(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(acc)

        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=graph, node_args=[acc], block_ids=[0]
                )
            ]
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertFalse(
                _detect_mma_loop(
                    fn,
                    [0],
                    block_sizes=[64],
                    num_threads_config=[1],
                )
            )

    def test_tcgen05_codegen_ignores_serialized_k_loop_threads(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 8, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(block_sizes=[64, 8, 16])

        with (
            patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch_cute_mma_support(),
        ):
            code = cute_matmul_mma_codegen_only.bind(args).to_triton_code(config)

        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05.OperandMajorMode.MN", code)
        self.assertIn(".make_fragment_B(", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn(
            "tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(4)",
            code,
        )
        # 4 epi + 1 exec + 1 ab_load = 6 warps, no power-of-2 round-up.
        self.assertIn("block=(64, 6, 1)", code)
        self.assertIn("'kind': 'tcgen05_d_tma'", code)
        self.assertIn("cutlass.pipeline.PipelineTmaStore.create", code)
        self.assertIn(
            "tcgen05_gC = cute.local_tile(tcgen05_tma_store_tensor, (64, 8),",
            code,
        )
        self.assertIn("partition_C(tcgen05_gC)", code)
        self.assertNotIn("for _tcgen05_store_i in range(", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_tcgen05_codegen_auto_path_preserves_mma_mode(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
        )

        with patch_cute_mma_support():
            bound = cute_matmul_mma_codegen_only.bind(args)
            # Keep the narrowed cluster_m=1 search. Explicit flat
            # cluster_m=2 configs are rejected until G3 runtime ownership is
            # validated, and this auto-path test only needs to pin tcgen05.
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.env.config_spec.restrict_tcgen05_cluster_m_search((1,))
            config = bound.config_spec.default_config()
            code = bound.to_triton_code(config)

        self.assertEqual(config.config["block_sizes"][2], 16)
        self.assertGreaterEqual(config.config["block_sizes"][0], 128)
        self.assertLessEqual(config.config["block_sizes"][0], 256)
        self.assertGreaterEqual(config.config["block_sizes"][1], 8)
        self.assertLessEqual(config.config["block_sizes"][1], 128)
        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05.OperandMajorMode.MN", code)
        self.assertIn(".make_fragment_B(", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn(
            "tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(4)",
            code,
        )
        self.assertIn(
            "if tcgen05_exec_active:",
            code,
        )
        # Dropped dead exec_leader / epi_leader variables; the codegen
        # paths that needed them inline ``cute.arch.elect_one()`` instead.
        self.assertNotIn("tcgen05_exec_leader", code)
        self.assertNotIn("tcgen05_epi_leader", code)
        self.assertIn(
            "tcgen05_epi_tidx = tcgen05_lane_idx + tcgen05_warp_idx * cutlass.Int32(32) if tcgen05_epi_active else cutlass.Int32(0)",
            code,
        )
        self.assertIn("mma_slice_tidx = cutlass.Int32(0)", code)
        self.assertNotIn("_helion_cute_cluster_shape = (2, 1, 1)", code)
        self.assertIn(
            "if tcgen05_epi_active:",
            code,
        )

    def test_tcgen05_large_bn_proof_admits_minimal_codegen(self) -> None:
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        @helion.kernel(backend="cute")
        def cute_matmul_large_bn_proof(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(64, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 512, device=DEVICE, dtype=torch.float16),
        )
        config = _make_tcgen05_large_bn_proof_config()

        with patch_cute_mma_support():
            bound = cute_matmul_large_bn_proof.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(config)

        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        self.assertIn("(64, 512)", code)
        self.assertIn(
            "tcgen05_gC = cute.local_tile(tcgen05_tma_store_tensor, (64, 512),",
            code,
        )
        self.assertIn("cutlass.pipeline.PipelineTmaStore.create", code)
        self.assertNotIn("_helion_cute_cluster_shape = (2, 1, 1)", code)
        self.assertNotIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)

    def test_tcgen05_large_bn_proof_rejects_non_proof_shapes(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_large_bn_wrong_shape(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 512, device=DEVICE, dtype=torch.float16),
        )
        config = _make_tcgen05_large_bn_proof_config(block_sizes=[128, 512, 16])

        with patch_cute_mma_support():
            bound = cute_matmul_large_bn_wrong_shape.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "tcgen05_ab_stages=2, tcgen05_acc_stages=1, and tcgen05_c_stages=2",
            ):
                bound.to_triton_code(config)

        config = _make_tcgen05_large_bn_proof_config()
        with patch_cute_mma_support():
            bound = cute_matmul_large_bn_wrong_shape.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "M=64,N=512,K=16",
            ):
                bound.to_triton_code(config)

        noncontiguous_args = (
            torch.randn(16, 64, device=DEVICE, dtype=torch.float16).t(),
            torch.randn(16, 512, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_large_bn_wrong_shape.bind(noncontiguous_args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "final tcgen05 CtaGroup.ONE TMA-load and TMA-store lowering",
            ):
                bound.to_triton_code(config)

    def test_tcgen05_dot_codegen_preregisters_collective_loads(self) -> None:
        @helion.kernel(backend="cute")
        def cute_dot_codegen_only(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_dot_codegen_only.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            config = _make_tcgen05_persistent_config(
                l2_groupings=[4],
                num_warps=4,
            )
            code = bound.to_triton_code(config)

        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        self.assertIn("PipelineTmaUmma.create(", code)
        self.assertNotIn("load = x[indices_0, indices_2]", code)
        self.assertNotIn("load_1 = y[indices_2, indices_1]", code)
        self.assertIn(
            "if tcgen05_exec_active:\n"
            "                tcgen05_ab_consumer_try_token = tcgen05_ab_pipeline.consumer_try_wait(tcgen05_ab_consumer_state)\n"
            "                tcgen05_ab_pipeline.consumer_wait(tcgen05_ab_consumer_state, tcgen05_ab_consumer_try_token)",
            code,
        )
        # The legacy K-loop no longer emits a ``fence_view_async_shared()``
        # between AB ``consumer_wait`` and ``cute.gemm`` (cute_plan.md
        # §6.9.2): the AB pipeline's transaction-count ``mbarrier_try_wait``
        # already orders the TMA shared stores ahead of the UMMA load.
        self.assertNotIn(
            "if tcgen05_exec_active:\n            cute.arch.fence_view_async_shared()",
            code,
        )
        self.assertIn(
            "if tcgen05_exec_active:\n            for _tcgen05_kblk_idx in range(",
            code,
        )
        self.assertIn(
            "if tcgen05_exec_active:\n                tcgen05_ab_pipeline.consumer_release(",
            code,
        )
        self.assertIn(
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)",
            code,
        )
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_tcgen05_kloop_tma_producer_split_into_separate_if(self) -> None:
        """Pin the per-K-iter producer/consumer split: producer work is
        emitted as ``if {tma_full_tile} and {tma_warp} and
        {tma_next_full_tile}: ...`` separate from the consumer/scalar-
        fallback ``if {tma_full_tile}: ... else: ...`` block.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_kloop_split(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 256, device=DEVICE, dtype=torch.float16),
            torch.randn(256, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_kloop_split.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            config = _make_tcgen05_persistent_config(num_warps=4)
            code = bound.to_triton_code(config)

        self.assertIn(
            "if tcgen05_tma_full_tile and tcgen05_tma_warp "
            "and tcgen05_tma_next_full_tile:",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline.producer_acquire("
            "tcgen05_ab_producer_state, tcgen05_ab_producer_try_token)",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline.producer_commit(tcgen05_ab_producer_state)",
            code,
        )
        self.assertIn("cute.arch.sync_threads()", code)
        compound_shape = (
            "if tcgen05_tma_full_tile:\n"
            "                if tcgen05_tma_warp and tcgen05_tma_next_full_tile:"
        )
        self.assertNotIn(compound_shape, code)

    def test_tcgen05_codegen_emits_setmaxregister_split(self) -> None:
        """Tcgen05 codegen emits Quack-style register reallocation: consumer
        warps (exec MMA + epilogue) call ``setmaxregister_increase(256)``;
        every other warp (TMA, AB-load, idle padding warps) calls
        ``setmaxregister_decrease(120)``. The "not consumer" framing of the
        decrease branch catches idle warps so they don't sit at the default
        ~168-register budget and steal headroom from real consumers."""

        @helion.kernel(backend="cute")
        def cute_matmul_setmaxregister(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_setmaxregister.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            config = _make_tcgen05_persistent_config(l2_groupings=[4])
            code = bound.to_triton_code(config)

        # Non-consumer warps (TMA, AB-load, idle padding) drop to 120 regs.
        self.assertIn(
            "if not (tcgen05_exec_active or tcgen05_epi_active):",
            code,
        )
        self.assertIn(
            "cute.arch.setmaxregister_decrease(120)",
            code,
        )
        # Consumer / epi warps raise to 256 regs.
        self.assertIn(
            "if tcgen05_exec_active or tcgen05_epi_active:",
            code,
        )
        self.assertIn(
            "cute.arch.setmaxregister_increase(256)",
            code,
        )
        # The setmaxregister calls land between the warp-role boolean
        # assignments and the MMA setup, so they sit in the registered
        # invariant block and do not get re-emitted per work-tile.
        decrease_pos = code.find("cute.arch.setmaxregister_decrease(120)")
        increase_pos = code.find("cute.arch.setmaxregister_increase(256)")
        epi_active_pos = code.find("tcgen05_epi_tidx = ")
        mma_slice_pos = code.find("mma_slice_tidx = ")
        self.assertGreater(decrease_pos, epi_active_pos)
        self.assertGreater(increase_pos, epi_active_pos)
        self.assertLess(decrease_pos, mma_slice_pos)
        self.assertLess(increase_pos, mma_slice_pos)

    def test_tcgen05_codegen_does_not_emit_dead_acc_frag_alias(self) -> None:
        """After the acc_frag persistent-loop fix, the prefix should
        NOT emit the dead ``acc_frag = acc_frag_base`` initialization
        that gets unconditionally overwritten by the per-tile
        ``acc_frag = exec_acc_frag_base[..., index]`` slice. Hoisting
        the dead alias under the persistent kernel splitter caused a
        CuTe DSL "structured different after this while" error."""

        @helion.kernel(backend="cute")
        def cute_matmul_acc_frag(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_acc_frag.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            config = _make_tcgen05_persistent_config(l2_groupings=[4])
            code = bound.to_triton_code(config)

        # The per-tile slice that actually gives acc_frag its working
        # shape SHOULD still be emitted.
        self.assertIn(
            "acc_frag = tcgen05_exec_acc_frag_base[None, None, None, "
            "tcgen05_acc_producer_state.index]",
            code,
        )
        # The dead pre-slice initialization should NOT appear.
        # (``acc_frag_base`` is still referenced via
        # ``tcgen05_epi_acc_frag_base = acc_frag_base`` and via the
        # exec/epi acc_frag tensor construction; we're specifically
        # ruling out the bare ``acc_frag = acc_frag_base`` line that
        # used to land in the hoisted invariant block.)
        self.assertNotIn("acc_frag = acc_frag_base", code)

    def test_tcgen05_mma_stage_uses_pipeline_consumer_state(self) -> None:
        """tcgen05 + TMA picks the AB SMEM stage from
        ``tma_consumer_state.index`` rather than the local
        ``(k_offset // bk) % stage_count`` modular form.

        The two values agree within a single tile (both advance by 1
        per K iteration starting at 0), but across persistent virtual
        tiles ``tma_consumer_state`` retains its end-of-tile value while
        ``k_offset`` resets to zero. Computing ``mma_stage`` from
        ``k_offset`` desyncs from the actual stage the consumer just
        unblocked, so the K-loop reads the wrong SMEM slot. Pin the
        consumer-state form here so future refactors can't silently
        regress to the modular formula.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_mma_stage(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_mma_stage.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            config = _make_tcgen05_persistent_config(l2_groupings=[4])
            code = bound.to_triton_code(config)

        # The new form: mma_stage tracks the pipeline state directly.
        self.assertIn("mma_stage = tcgen05_ab_consumer_state.index", code)
        # The old modular form must NOT be emitted in the TMA path. The
        # K-offset variable name is generated and may shift, so check
        # the modular operator pattern itself rather than the full
        # rendered expression -- if it ever comes back, persistent +
        # tcgen05 desyncs across virtual tile boundaries again.
        self.assertNotIn("// cutlass.Int32(16) % cutlass.Int32(2)", code)
        self.assertNotIn("mma_stage = (", code)

    def test_tcgen05_persistent_post_loop_stmts_appear_after_while(self) -> None:
        """Compiling a persistent_blocked kernel must emit the cleanup
        block (producer_tail / TMEM allocator setup / free) AFTER the
        ``while tcgen05_work_tile_valid`` loop.

        Before the post-loop split landed, those statements stayed inside
        the persistent loop and were yielded back as scf.while carries,
        which crashed CuTe IR verification with
        ``operand #N does not dominate this use``. This test pins the
        structural fix.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_persistent_post_loop(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_post_loop.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            # ``persistent_blocked`` is normally disallowed for tcgen05
            # via ``enforce_dot_requirements`` because autotune can still
            # choose configs that fall back to guarded partial persistent
            # paths. The codegen itself accepts the explicit config, which
            # is what this structural test exercises.
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
            )
            code = bound.to_triton_code(cfg)

        # Locate the persistent while loop and verify post-loop
        # statements live OUTSIDE its body. The cleanest check is to find
        # the line of ``while tcgen05_work_tile_valid`` and the first
        # following dedented statement (= post-loop boundary), then
        # confirm producer_tail / free fall on the post-loop side.
        lines = code.splitlines()
        while_line_idx = next(
            i for i, line in enumerate(lines) if "while tcgen05_work_tile_valid" in line
        )
        while_indent = len(lines[while_line_idx]) - len(
            lines[while_line_idx].lstrip(" ")
        )
        post_loop_line_idx = None
        for i in range(while_line_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent <= while_indent:
                post_loop_line_idx = i
                break
        self.assertIsNotNone(
            post_loop_line_idx, "post-loop statements should follow the while"
        )
        post_loop_block = "\n".join(lines[post_loop_line_idx:])
        in_loop_block = "\n".join(lines[while_line_idx + 1 : post_loop_line_idx])
        # Cleanup statements must be in the post-loop block, not the
        # work-tile body.
        for tag in (
            "tcgen05_acc_pipeline.producer_tail",
            "tcgen05_tmem_allocator.free",
        ):
            self.assertIn(tag, post_loop_block, f"{tag} must follow the while loop")
            self.assertNotIn(
                tag, in_loop_block, f"{tag} must not appear inside the while loop"
            )

    def test_tcgen05_persistent_path_compiles(self) -> None:
        """End-to-end compile check for the persistent + tcgen05 combo.

        The post-loop split fixes the CuTe IR
        ``operand #7 does not dominate this use`` verifier crash. This
        test pins that — if a future change re-introduces a state
        carry that can't be expressed as a scf.while iter arg, the IR
        verifier will reject the kernel and ``to_triton_code(cfg)``
        will raise. Static full-tile multi-tile correctness is covered
        by ``test_tcgen05_persistent_multi_tile_runtime_correctness``."""

        @helion.kernel(backend="cute")
        def cute_matmul_persistent_compile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_compile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
            )
            # The act of generating the Python source already runs the
            # CuTe DSL preprocessor; the IR verifier runs at first
            # execution (cute.compile). The code-string checks below pin
            # the persistent host wrapper shape; runtime coverage for both
            # z-seeded and tile-advance scheduler paths lives in
            # ``test_tcgen05_persistent_multi_tile_runtime_correctness``.
            code = bound.to_triton_code(cfg)
            self.assertIn("while tcgen05_work_tile_valid", code)
            self.assertIn("tcgen05_acc_pipeline.producer_tail", code)
            from helion._compiler.program_id import Tcgen05PersistentProgramIDs

            self.assertNotIn(
                Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR,
                code,
            )
            self.assertRegex(code, r"\(\s*1\s*,\s*1\s*,\s*min\s*\(")

    def test_tcgen05_role_local_monolithic_byte_identical_golden(self) -> None:
        """G2-B byte-identity pin (cute_plan.md §6.2 pin tests #1, #2).

        The retained ``ROLE_LOCAL_MONOLITHIC`` seed (cute_plan.md
        §10.1: 4096³ bf16, ``block_sizes=[256, 256, 128]``,
        ``cluster_m=2``, ``pid_type=persistent_interleaved``,
        ``ab/acc/c_stages=2/2/2``, ``num_epi_warps=4``) must
        generate byte-identical CuTe across G2 refactors. The kernel
        is hosted at a stable file path
        (``test/golden/_tcgen05_role_local_monolithic_4096_bf16_kernel.py``)
        so the embedded ``src[<file>:<line>]`` comments do not drift
        when ``test_cute_lowerings.py`` itself changes.

        Why a separate ``.py.expected`` file rather than
        ``helion._testing.AssertExpectedJournal``: the journal stores
        all expected outputs in one shared
        ``test_cute_lowerings.expected`` file under
        ``--- assertExpectedJournal(...)`` section markers, and
        requires the test class to inherit from
        ``helion._testing.TestCase``. ``TestCuteLowerings`` currently
        inherits from ``unittest.TestCase`` (line 400 of this file) —
        switching the base class would change setUp/tearDown
        semantics for ~200 unrelated tests and is out of scope for
        a byte-identity pin. A standalone 315-line ``.py.expected``
        file is also more inspectable than a section inside the
        shared journal. If a second golden test lands here, factor
        the read/write/diff plumbing into a small helper alongside
        the journal class in ``helion/_testing.py`` rather than
        copy-pasting this block.

        Update protocol: when an intentional codegen change makes
        this test fail, regenerate the golden by setting
        ``EXPECTTEST_ACCEPT=1`` (or by running
        ``EXPECTTEST_ACCEPT=1 pytest -k
        test_tcgen05_role_local_monolithic_byte_identical_golden``).
        Reviewers diff the regenerated golden against the previous
        version; only intentional codegen deltas should land.
        """

        from test.golden._tcgen05_role_local_monolithic_4096_bf16_kernel import (
            cute_matmul_role_local_monolithic_4096_bf16,
        )

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        seed_config = helion.Config(
            block_sizes=[256, 256, 128],
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
        )
        with patch_cute_mma_support():
            bound = cute_matmul_role_local_monolithic_4096_bf16.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            actual = bound.to_triton_code(seed_config)

        # Sanity: confirm the generated kernel is on the
        # retained-seed tcgen05 path (a fallback path would diff
        # against the golden but the ``make_trivial_tiled_mma`` /
        # ``PipelineUmmaAsync`` markers below catch a regression to
        # a non-tcgen05 fallback even before the golden compare
        # runs, so the failure points at a real shape change rather
        # than a host-detect regression).
        self.assertIn("make_trivial_tiled_mma", actual)
        self.assertIn("PipelineUmmaAsync.create", actual)
        self.assertIn("PipelineTmaUmma.create", actual)
        self.assertIn("PipelineTmaStore.create", actual)

        # ``EXPECTTEST_ACCEPT=1`` is a *local-regeneration* hook
        # (mirrors helion's ``AssertExpectedJournal`` machinery in
        # ``helion/_testing.py``); CI runs the test without that
        # env var so the golden file always exists and any drift
        # fails. The "missing golden" path below also writes the
        # file but fails the test so a bare ``pytest`` invocation
        # in a fresh checkout produces a useful error rather than
        # silently passing. Set ``EXPECTTEST_ACCEPT=1`` only when
        # intentionally regenerating after a deliberate codegen
        # change, and review the resulting diff in your PR.
        accept = os.environ.get("EXPECTTEST_ACCEPT") == "1"
        if accept or not TCGEN05_ROLE_LOCAL_MONOLITHIC_GOLDEN_PATH.exists():
            TCGEN05_ROLE_LOCAL_MONOLITHIC_GOLDEN_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )
            TCGEN05_ROLE_LOCAL_MONOLITHIC_GOLDEN_PATH.write_text(actual)
            if not accept:
                self.fail(
                    "golden file was missing; wrote initial version. "
                    "Re-run the test to verify."
                )
            return
        expected = TCGEN05_ROLE_LOCAL_MONOLITHIC_GOLDEN_PATH.read_text()
        if actual != expected:
            diff = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    actual.splitlines(keepends=True),
                    fromfile="golden",
                    tofile="actual",
                    n=3,
                )
            )
            self.fail(
                "Generated CuTe differs from the golden — "
                "byte-identity for ROLE_LOCAL_MONOLITHIC seed broken. "
                "If this is an intentional codegen change, regenerate "
                "the golden via "
                "EXPECTTEST_ACCEPT=1 pytest -k "
                "test_tcgen05_role_local_monolithic_byte_identical_golden, "
                "diff the regenerated file against the prior version, "
                "and confirm every delta in your PR description.\n"
                f"--- diff ---\n{diff}"
            )

    def test_tcgen05_codegen_consumes_warp_spec(self) -> None:
        """G2-B data-flow pin: ``Tcgen05WarpSpec`` is consumed by
        codegen as the source of truth for warp role IDs.

        Two distinct call sites must route through the strategy
        layer: ``_tcgen05_epi_warp_count`` (which now takes a
        ``Tcgen05WarpSpec`` argument) and the ``CuteTcgen05MatmulPlan``
        construction (which reads ``ab_load_warp_count`` from the
        spec). Before G2-B both read ``tcgen05_num_epi_warps``
        directly. The test pins both:

        - ``warp_spec_from_config`` is invoked at least once during
          ``to_triton_code``. Today it is invoked exactly once (item
          4 of the G2-B review hoisted the build inside the
          tcgen05 branch); the lower bound is kept loose so a
          future cycle can route additional codegen sites through
          the helper without a test-only edit.
        - The constructed ``CuteTcgen05MatmulPlan`` carries
          ``ab_load_warp_count`` and ``epi_warp_count`` derived from
          the observed spec. A partial revert that hard-coded
          either field would still pass the call-count check; this
          object-shape pin catches that.

        ``register_split`` is also asserted on the observed spec so
        a drift in the (120, 256) decrease/increase pair is caught
        as part of this test rather than waiting for the byte-
        identity golden to flag it.
        """

        from test.golden._tcgen05_role_local_monolithic_4096_bf16_kernel import (
            cute_matmul_role_local_monolithic_4096_bf16,
        )

        from helion._compiler.cute import cute_mma as cute_mma_module
        from helion._compiler.cute import strategies as strategies_module
        from helion._compiler.device_function import (
            CuteTcgen05MatmulPlan as _CuteTcgen05MatmulPlan,
        )

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        seed_config = helion.Config(
            block_sizes=[256, 256, 128],
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
        )

        original_warp_spec = strategies_module.warp_spec_from_config
        observed_specs: list[Tcgen05WarpSpec] = []

        def sniffing_warp_spec_from_config(config):  # type: ignore[no-untyped-def]
            spec = original_warp_spec(config)
            observed_specs.append(spec)
            return spec

        observed_plans: list[_CuteTcgen05MatmulPlan] = []

        def sniffing_matmul_plan(*args, **kwargs):  # type: ignore[no-untyped-def]
            plan = _CuteTcgen05MatmulPlan(*args, **kwargs)
            observed_plans.append(plan)
            return plan

        with patch_cute_mma_support():
            bound = cute_matmul_role_local_monolithic_4096_bf16.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            with (
                patch.object(
                    cute_mma_module,
                    "warp_spec_from_config",
                    sniffing_warp_spec_from_config,
                ),
                patch.object(
                    cute_mma_module,
                    "CuteTcgen05MatmulPlan",
                    sniffing_matmul_plan,
                ),
            ):
                bound.to_triton_code(seed_config)

        # warp_spec_from_config is reached at least once. The hoist
        # in item 4 of the G2-B review collapsed two reads into one,
        # so today this is exactly 1 — keep the assertion loose so
        # future per-strategy split work can add reads without a
        # test-only edit.
        self.assertGreaterEqual(
            len(observed_specs), 1, msg=f"calls={len(observed_specs)}"
        )
        # Documented G2-A defaults for ROLE_LOCAL_MONOLITHIC:
        # 1 ab_load + 1 mma + 4 epi + 0 epi_load + 0 scheduler = 6
        # warps; register_split = (120, 256).
        for spec in observed_specs:
            self.assertEqual(spec.ab_load_warps, 1)
            self.assertEqual(spec.mma_warps, 1)
            self.assertEqual(spec.epi_warps, 4)
            self.assertEqual(spec.epi_load_warps, 0)
            self.assertEqual(spec.scheduler_warps, 0)
            self.assertEqual(spec.total_warps, 6)
            self.assertEqual(spec.register_split, (120, 256))
        # The constructed matmul plan carries warp-role counts
        # derived from the spec — a partial revert of the G2-B
        # refactor that hard-coded ``ab_load_warp_count=1`` directly
        # in the ``CuteTcgen05MatmulPlan(...)`` call would still
        # pass the call-count check above, so pin the field
        # directly on the observed plan object.
        self.assertEqual(len(observed_plans), 1, msg=f"plans={len(observed_plans)}")
        plan = observed_plans[0]
        spec = observed_specs[0]
        self.assertEqual(plan.ab_load_warp_count, spec.ab_load_warps)
        # ``epi_warp_count`` is the cap-applied value (limited by
        # ``cta_thread_count // 32``), but for the seed config the
        # cap is loose and the spec's epi_warps flows through.
        self.assertEqual(plan.epi_warp_count, spec.epi_warps)

        # Strong dataflow pin: rerun codegen with a forced spec
        # whose ``ab_load_warps`` differs from the autotune-pinned
        # value (``1``). The plan's ``ab_load_warp_count`` must
        # follow the forced spec, not the original config key. A
        # partial revert that hard-coded ``ab_load_warp_count=1``
        # in the ``CuteTcgen05MatmulPlan(...)`` call would fail this
        # assertion (the plan would carry ``1`` while the forced
        # spec carries the substituted value).
        forced_spec = dataclasses.replace(
            ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, ab_load_warps=2
        )
        observed_plans.clear()

        # Sentinel raised by the spy immediately after the plan is
        # captured. Short-circuiting codegen here keeps the test
        # focused on the dataflow check and avoids depending on
        # whatever downstream emission would do with a non-default
        # ``ab_load_warps`` (today: explode in epi-warp accounting).
        # The sentinel is plain ``Exception`` and gets wrapped by
        # ``inductor_lowering.run_node`` into ``InductorLoweringError``;
        # we suppress *only* that specific helion exception type, so
        # an unrelated failure (e.g. a future rename of
        # ``warp_spec_from_config`` causing ``patch.object`` to
        # raise ``AttributeError`` at context entry, or a real
        # ``TypeError`` thrown elsewhere in ``to_triton_code``)
        # still surfaces as a test failure pointing at the actual
        # regression instead of being silently eaten.
        class _CapturedPlan(Exception):
            """Sentinel raised once the plan is captured."""

        def short_circuiting_matmul_plan(*args, **kwargs):  # type: ignore[no-untyped-def]
            plan = _CuteTcgen05MatmulPlan(*args, **kwargs)
            observed_plans.append(plan)
            raise _CapturedPlan

        captured_plan_seen = False
        with patch_cute_mma_support():
            bound = cute_matmul_role_local_monolithic_4096_bf16.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            with (
                patch.object(
                    cute_mma_module,
                    "warp_spec_from_config",
                    lambda _config: forced_spec,
                ),
                patch.object(
                    cute_mma_module,
                    "CuteTcgen05MatmulPlan",
                    short_circuiting_matmul_plan,
                ),
            ):
                try:
                    bound.to_triton_code(seed_config)
                except exc.InductorLoweringError as wrapped:
                    # Confirm the suppression was triggered by *our*
                    # sentinel, not by some other codegen failure
                    # that happens to surface as InductorLoweringError.
                    chain: BaseException | None = wrapped
                    while chain is not None:
                        if isinstance(chain, _CapturedPlan):
                            captured_plan_seen = True
                            break
                        chain = chain.__cause__ or chain.__context__
                    if not captured_plan_seen:
                        raise
        self.assertTrue(
            captured_plan_seen,
            msg="forced-spec rerun did not raise the _CapturedPlan sentinel",
        )
        self.assertEqual(len(observed_plans), 1, msg="plan was never constructed")
        self.assertEqual(observed_plans[0].ab_load_warp_count, 2)

    def test_tcgen05_with_scheduler_cluster_m2_per_cta_topology_codegen(
        self,
    ) -> None:
        """G2-C cluster_m=2 codegen pin (cute_plan.md §6.3.1).

        ``ROLE_LOCAL_WITH_SCHEDULER`` at ``cluster_m=2`` uses a
        per-CTA scheduler topology: every CTA in the cluster runs its
        own scheduler that publishes locally and consumers release
        locally. The sched_pipeline emission therefore must NOT carry
        ``consumer_mask=cutlass.Int32(0)`` (which would route every
        CTA's consumer release to the leader CTA's empty barrier and
        starve non-leader CTAs of arrivals — the cluster_m=2 hang the
        prior cycle reproduced). The consumer arrive count must also
        be per-CTA (``role_warp_count - scheduler_warp_count``)
        without the ``× cluster_size`` Quack-style multiplier.

        Pin both invariants on the captured generated code so a
        future refactor that reintroduces the leader-only mask or
        the cluster-wide arrive count fails this test loudly rather
        than silently regressing to the hang at runtime.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_with_scheduler_c2(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        config = helion.Config(
            block_sizes=[256, 256, 128],
            l2_groupings=[1],
            num_warps=4,
            num_sm_multiplier=1,
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            indexing=[
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
        )
        with patch_cute_mma_support():
            bound = cute_matmul_with_scheduler_c2.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(config)

        # The sched_pipeline.create call must omit consumer_mask. A
        # bare assertNotIn("consumer_mask", code) would also catch
        # ``mcast_mask`` / ``producer_mask`` etc. — narrow to the
        # specific Quack-style literal that the prior implementation
        # emitted.
        self.assertNotIn("consumer_mask=cutlass.Int32(0)", code)
        # Locate the sched_pipeline create call and assert the
        # specific kwargs that should NOT appear on it.
        sched_create_idx = code.find("tcgen05_sched_pipeline = ")
        self.assertGreater(
            sched_create_idx,
            -1,
            msg="WITH_SCHEDULER kernel did not emit a sched_pipeline.create",
        )
        sched_create_end = code.find(")", sched_create_idx)
        sched_create_call = code[sched_create_idx:sched_create_end]
        self.assertNotIn("consumer_mask", sched_create_call)
        # ``defer_sync=True`` must still appear so the sched_pipeline
        # init coordinates with the AB / acc / c pipelines under the
        # cluster-deferred protocol.
        self.assertIn("defer_sync=True", sched_create_call)

        # Per-CTA consumer arrive count = role_warps - scheduler_warps =
        # 1 (ab_load) + 1 (mma) + 4 (epi) + 0 (epi_load) = 6.
        # Multiplying by cluster_size (=2) would re-introduce the hang.
        self.assertIn(
            "tcgen05_sched_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(6))",
            code,
        )
        # Negative pin: the cluster-wide count (6 * 2 = 12) must NOT
        # appear in the sched_pipeline consumer group. Anchor on the
        # full ``CooperativeGroup(... cutlass.Int32(12))`` literal so
        # an unrelated 12 elsewhere in the kernel does not flake the
        # test.
        self.assertNotIn(
            "tcgen05_sched_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(12))",
            code,
        )

    def test_tcgen05_clc_persistent_codegen(self) -> None:
        """G2-H CLC scheduler-warp body codegen pin (cute_plan.md).

        Under ``Tcgen05PersistenceModel.CLC_PERSISTENT +
        ROLE_LOCAL_WITH_SCHEDULER``, the scheduler-warp role must
        emit ``nvvm.clusterlaunchcontrol_try_cancel`` (via the
        ``_cute_issue_clc_query_nomulticast`` wrapper from
        ``helion._compiler.cute.clc_helpers``) exactly once on the
        scheduler-warp path. ``StaticPersistentTileScheduler.create``
        is *also* present in the CLC scheduler-warp body — we reuse
        its ``_get_current_work_for_linear_idx`` decoder for both
        the initial ``block_idx`` and each subsequent CLC ``bidz``
        response so tile-coord arithmetic stays identical to the
        static path. The CLC SMEM response buffer + mbarrier are
        allocated alongside the existing ``sched_pipeline`` mbars.

        cluster_m=2 keeps the leader-CTA-only gating (Quack pattern)
        plus the cluster-broadcast publish via
        ``_cute_store_shared_remote_x4`` so non-leader CTAs receive
        the work-tile via cross-CTA SMEM stores.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_clc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        config = helion.Config(
            block_sizes=[256, 256, 128],
            l2_groupings=[1],
            num_warps=4,
            num_sm_multiplier=1,
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            tcgen05_persistence_model="clc_persistent",
            indexing=[
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
        )
        with patch_cute_mma_support():
            bound = cute_matmul_clc.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(config)

        # CLC issuance helper: appears exactly twice in the generated
        # module (one import line at the top + one call inside the
        # scheduler-warp body). A second call site would suggest the
        # scheduler warp issued CLC twice per iteration, which would
        # double-cancel pending clusters and corrupt scheduling.
        self.assertEqual(
            code.count("_cute_issue_clc_query_nomulticast"),
            2,
            msg=(
                "expected 1 import + 1 kernel-body call of "
                "_cute_issue_clc_query_nomulticast; "
                f"actual occurrences = {code.count('_cute_issue_clc_query_nomulticast')}"
            ),
        )
        # ``StaticPersistentTileScheduler.create`` IS present in the
        # CLC scheduler-warp body: we reuse the static scheduler's
        # ``_get_current_work_for_linear_idx`` to decode both the
        # initial ``block_idx`` and each subsequent CLC ``bidz``
        # response back to per-CTA tile coordinates. The decode
        # arithmetic stays identical to the static path so the
        # consumer's ``virtual_pid = work_tile_smem[0] // cluster_m``
        # collapse formula doesn't need to fork. The CLC-specific
        # markers below (``_cute_issue_clc_query_nomulticast``,
        # ``cute.arch.clc_response``, ``cute.arch.mbarrier_init``)
        # pin the actual CLC emission so a regression to a pure
        # static path would still fail the test loudly.
        self.assertIn("StaticPersistentTileScheduler.create", code)
        # CLC SMEM allocations (response buffer + mbarrier).
        self.assertIn("tcgen05_clc_response_smem_ptr", code)
        self.assertIn("tcgen05_clc_mbar_smem_ptr", code)
        # cute.arch.clc_response decoder reads the response buffer.
        self.assertIn("cute.arch.clc_response", code)
        # mbarrier_init for the CLC mbar (arrival count 1, only
        # the CLC issuer arrives).
        self.assertIn("cute.arch.mbarrier_init(tcgen05_clc_mbar_smem_ptr, 1)", code)
        # cluster_m=2: leader-CTA-only gate on the scheduler body
        # plus the cluster-broadcast publish via _cute_store_shared_remote_x4.
        self.assertIn("cute.arch.block_idx_in_cluster()) == cutlass.Int32(0)", code)
        self.assertIn("_cute_store_shared_remote_x4", code)
        # CLC sched_pipeline uses cluster-routed empty mbar
        # (consumer_mask=Int32(0)) for cluster_m>1, mirroring Quack's
        # make_sched_pipeline pattern. Anchor on the
        # ``PipelineAsync.create(...)`` call structure: we look for
        # the open-paren after ``tcgen05_sched_pipeline = ...`` and
        # match balanced parens to capture the full call. A naive
        # ``find("\n\n")`` reliance silently degrades to
        # ``code[idx:-1]`` if the formatter reflows the call; this
        # explicit balanced-paren scan is robust to that.
        sched_create_match = re.search(
            r"tcgen05_sched_pipeline = "
            r"cutlass\.pipeline\.PipelineAsync\.create\(",
            code,
        )
        self.assertIsNotNone(
            sched_create_match,
            msg="CLC kernel did not emit a sched_pipeline.create call",
        )
        # Walk forward from the open paren matching nested parens
        # so the captured slice ends at the call's closing paren.
        open_idx = sched_create_match.end() - 1
        depth = 0
        close_idx = None
        for i in range(open_idx, len(code)):
            ch = code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        self.assertIsNotNone(
            close_idx, msg="unbalanced parens in sched_pipeline.create call"
        )
        sched_create_call = code[sched_create_match.start() : close_idx + 1]
        self.assertIn("consumer_mask=cutlass.Int32(0)", sched_create_call)

    def test_tcgen05_clc_persistent_does_not_perturb_monolithic(self) -> None:
        """G2-H: ``ROLE_LOCAL_MONOLITHIC`` codegen must be byte-
        identical pre/post G2-H. The byte-identity golden test
        (``test_tcgen05_role_local_monolithic_byte_identical_golden``)
        already pins the MONOLITHIC kernel; this test additionally
        confirms that the MONOLITHIC path emits no CLC markers
        regardless of the new persistence-model field.
        """

        from test.golden._tcgen05_role_local_monolithic_4096_bf16_kernel import (
            cute_matmul_role_local_monolithic_4096_bf16,
        )

        args = (
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
            torch.empty([4096, 4096], device=DEVICE, dtype=torch.bfloat16),
        )
        seed_config = helion.Config(
            block_sizes=[256, 256, 128],
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=2,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
        )
        with patch_cute_mma_support():
            bound = cute_matmul_role_local_monolithic_4096_bf16.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            code = bound.to_triton_code(seed_config)

        # MONOLITHIC must NOT emit any of the CLC markers.
        self.assertNotIn("nvvm.clusterlaunchcontrol_try_cancel", code)
        self.assertNotIn("_cute_issue_clc_query_nomulticast", code)
        self.assertNotIn("tcgen05_clc_response_smem_ptr", code)
        self.assertNotIn("tcgen05_clc_mbar_smem_ptr", code)
        self.assertNotIn("cute.arch.clc_response", code)
        # MONOLITHIC keeps the static-persistent emitter.
        self.assertIn("StaticPersistentTileScheduler.create", code)

    def test_tcgen05_persistent_kloop_producer_lifts_to_role_local_while(
        self,
    ) -> None:
        """Persistent tcgen05 lifts producer, exec, and epi work into
        role-local persistent loops. The TMA producer K-loop is now a
        top-level extracted role block, not an inline wrapper inside the
        shared K-loop, and its predicate drops the inline
        ``tcgen05_tma_warp`` gate because the enclosing role-local loop
        already restricts execution to that warp. The MMA-exec role owns
        AB consumer wait/release, UMMA issue, and acc producer state. The
        epi role owns acc consumer wait/release and the TMA-store epilogue
        with a role-local tile counter for the SMEM ring."""

        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 128, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
            )
            code = bound.to_triton_code(cfg)
        self.assertIn("'kind': 'tcgen05_d_tma'", code)
        self.assertIn(
            "tcgen05_tma_store_role_tile = tcgen05_tma_store_role_tile + cutlass.Int32(1)",
            code,
        )
        # The role predicates the partitioner uses for role gates. The
        # TMA-load warp id is 5 and the exec warp id is 4 for the 6-warp layout
        # (epi=warps 0..3, exec=4, tma=5).
        tma_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
        )
        exec_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4)"
        )
        epi_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
        )
        tree = ast.parse(code)
        found_role_local_producer_loop = False
        found_role_local_exec_loop = False
        found_role_local_epi_loop = False
        shared_loop_has_tma_producer = False
        shared_loop_preserves_barriers = False
        shared_scheduler_retargeted_to_exec = False
        shared_loop_excludes_tma = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == tma_role_predicate
            ):
                continue
            for role_child in ast.walk(node):
                if not (
                    isinstance(role_child, ast.While)
                    and "tcgen05_role_local" in ast.unparse(role_child.test)
                ):
                    continue
                role_src = ast.unparse(role_child)
                self.assertIn("pid_0 = virtual_pid % num_blocks_0", role_src)
                self.assertIn("offset_0 = pid_0 * _BLOCK_SIZE_0", role_src)
                self.assertIn("for offset_2 in range", role_src)
                self.assertIn(
                    "if tcgen05_tma_full_tile and tcgen05_tma_next_full_tile",
                    role_src,
                )
                self.assertNotIn(
                    "tcgen05_tma_full_tile and tcgen05_tma_warp and "
                    "tcgen05_tma_next_full_tile",
                    role_src,
                )
                found_role_local_producer_loop = True

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == exec_role_predicate
            ):
                continue
            for role_child in ast.walk(node):
                if not (
                    isinstance(role_child, ast.While)
                    and "tcgen05_role_local" in ast.unparse(role_child.test)
                ):
                    continue
                role_src = ast.unparse(role_child)
                self.assertIn(
                    "tcgen05_acc_pipeline.producer_acquire(tcgen05_acc_producer_state)",
                    role_src,
                )
                self.assertIn(
                    "consumer_try_wait(tcgen05_ab_consumer_state)",
                    role_src,
                )
                self.assertIn("cute.gemm(", role_src)
                self.assertIn(
                    "consumer_release(tcgen05_ab_consumer_state)",
                    role_src,
                )
                self.assertIn(
                    "tcgen05_acc_pipeline.producer_commit(tcgen05_acc_producer_state)",
                    role_src,
                )
                self.assertIn("tcgen05_acc_producer_state.advance()", role_src)
                found_role_local_exec_loop = True

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == epi_role_predicate
            ):
                continue
            for role_child in ast.walk(node):
                if not (
                    isinstance(role_child, ast.While)
                    and "tcgen05_role_local" in ast.unparse(role_child.test)
                ):
                    continue
                role_src = ast.unparse(role_child)
                self.assertIn(
                    "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)",
                    role_src,
                )
                self.assertNotIn("PipelineTmaStore.create", role_src)
                self.assertIn("cute.nvgpu.cpasync.tma_partition", role_src)
                self.assertIn("cute.copy(tcgen05_tma_store_atom", role_src)
                self.assertIn("tcgen05_tma_store_role_tile", role_src)
                self.assertNotIn("cute.nvgpu.CopyUniversalOp()", role_src)
                self.assertIn("cute.copy(tcgen05_tiled_copy_t2r", role_src)
                self._assert_tma_store_epilogue_order(role_src, require_tail=False)
                self.assertIn(
                    "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)",
                    role_src,
                )
                self.assertNotIn("cute.arch.sync_threads()", role_src)
                found_role_local_epi_loop = True

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.While)
                and ast.unparse(node.test) == "tcgen05_work_tile_valid"
            ):
                continue
            shared_src = ast.unparse(node)
            shared_loop_has_tma_producer = (
                "producer_try_acquire(tcgen05_ab_producer_state)" in shared_src
            )
            self.assertNotIn("consumer_try_wait(tcgen05_ab_consumer_state)", shared_src)
            self.assertNotIn("cute.gemm(", shared_src)
            self.assertNotIn(
                "tcgen05_acc_pipeline.producer_commit(tcgen05_acc_producer_state)",
                shared_src,
            )
            self.assertNotIn(
                "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)",
                shared_src,
            )
            self.assertNotIn(
                "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)",
                shared_src,
            )
            self.assertNotIn("cute.nvgpu.CopyUniversalOp()", shared_src)
            self.assertNotIn("PipelineTmaStore.create", shared_src)
            shared_loop_preserves_barriers = "cute.arch.sync_threads()" in shared_src
            shared_scheduler_retargeted_to_exec = (
                "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4)"
                in shared_src
                and "tcgen05_tile_sched.advance_to_next_work()" in shared_src
            )
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if not (test_src.startswith("not ") and "cutlass.Int32(5)" in test_src):
                continue
            shared_loop_excludes_tma = any(
                isinstance(child, ast.While)
                and ast.unparse(child.test) == "tcgen05_work_tile_valid"
                for child in node.body
            )
        self.assertTrue(
            found_role_local_producer_loop,
            "Expected a role-local TMA producer while containing the "
            "producer-only K-loop. Generated code:\n" + code,
        )
        self.assertTrue(
            found_role_local_exec_loop,
            "Expected a role-local MMA-exec while containing AB consumer, "
            "UMMA issue, and acc producer work. Generated code:\n" + code,
        )
        self.assertTrue(
            found_role_local_epi_loop,
            "Expected a role-local epilogue while containing acc consumer "
            "and SIMT store work. Generated code:\n" + code,
        )
        self._assert_role_local_c_store_pipeline_lifetime(
            code, tree, epi_role_predicate
        )
        self.assertTrue(
            shared_loop_preserves_barriers,
            "Shared persistent while must preserve CTA barriers so the "
            "role-local warps can rejoin as barrier participants. Generated code:\n"
            + code,
        )
        self.assertTrue(
            shared_scheduler_retargeted_to_exec,
            "Shared scheduler advance should be owned by the exec warp in "
            "the role-local mainloop path. Generated code:\n" + code,
        )
        self.assertFalse(
            shared_loop_excludes_tma,
            "The TMA warp must still enter the shared while after its "
            "role-local producer loop so existing sync_threads barriers "
            "remain valid. Generated code:\n" + code,
        )
        self.assertFalse(
            shared_loop_has_tma_producer,
            "Shared persistent while should not contain the TMA producer "
            "K-loop. Generated code:\n" + code,
        )

    def test_tcgen05_flat_static_full_uses_tma_store_epilogue(self) -> None:
        """Static-full flat tcgen05 lowers the first G2 TMA-store epilogue.

        The flat fast path has one output tile per CTA, so its staged SMEM +
        TMA bulk store path can use the subtile index directly as the ring
        stage. Persistent kernels use a role-local tile counter instead.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
            )
            code = bound.to_triton_code(cfg)
            self.assertIn("'kind': 'tcgen05_d_tma'", code)
            self.assertIn("cutlass.pipeline.PipelineTmaStore.create", code)
            self.assertIn(
                "cutlass.utils.gemm.sm100.epilogue_smem_copy_and_partition",
                code,
            )
            self.assertIn("cute.nvgpu.cpasync.tma_partition", code)
            self.assertIn("cute.copy(tcgen05_tma_store_atom", code)
            self.assertNotIn("cute.nvgpu.CopyUniversalOp()", code)
            self._assert_tma_store_epilogue_order(code)
            bound.set_config(cfg)
            out = bound(*args)

        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_diagnostic_first_c_acquire_moves_into_subtile_loop(
        self,
    ) -> None:
        """The first-C-acquire discriminator changes only that acquire site."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_c_acquire_placement="first_in_loop",
            )
            code = bound.to_triton_code(cfg)

        acquire = "tcgen05_c_pipeline.producer_acquire()"
        gmem_tile = "tcgen05_gC = cute.local_tile("
        subtile_loop = "for _tcgen05_subtile in cutlass.range("
        first_subtile_guard = (
            "if _tcgen05_subtile == 0 and tcgen05_warp_idx == cutlass.Int32(0):"
        )
        later_subtile_guard = (
            "if _tcgen05_subtile != 0 and tcgen05_warp_idx == cutlass.Int32(0):"
        )
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"

        self.assertEqual(code.count(acquire), 2, code)
        gmem_tile_pos = code.find(gmem_tile)
        subtile_loop_pos = code.find(subtile_loop, gmem_tile_pos)
        first_subtile_guard_pos = code.find(first_subtile_guard, subtile_loop_pos)
        first_acquire_pos = code.find(acquire, first_subtile_guard_pos)
        later_subtile_guard_pos = code.find(later_subtile_guard, first_acquire_pos)
        later_acquire_pos = code.find(acquire, later_subtile_guard_pos)
        acc_wait_pos = code.find(acc_wait, later_acquire_pos)
        for needle, pos in (
            (gmem_tile, gmem_tile_pos),
            (subtile_loop, subtile_loop_pos),
            (first_subtile_guard, first_subtile_guard_pos),
            ("first in-loop C acquire", first_acquire_pos),
            (later_subtile_guard, later_subtile_guard_pos),
            ("later C acquire", later_acquire_pos),
            (acc_wait, acc_wait_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        self.assertLess(gmem_tile_pos, subtile_loop_pos, code)
        self.assertLess(subtile_loop_pos, first_subtile_guard_pos, code)
        self.assertLess(first_subtile_guard_pos, first_acquire_pos, code)
        self.assertLess(first_acquire_pos, later_subtile_guard_pos, code)
        self.assertLess(later_subtile_guard_pos, later_acquire_pos, code)
        self.assertLess(later_acquire_pos, acc_wait_pos, code)

    def test_tcgen05_diagnostic_later_c_acquire_moves_before_barrier(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_c_acquire_placement="later_before_barrier",
            )
            code = bound.to_triton_code(cfg)

        acquire = "tcgen05_c_pipeline.producer_acquire()"
        gmem_tile = "tcgen05_gC = cute.local_tile("
        subtile_loop = "for _tcgen05_subtile in cutlass.range("
        later_subtile_guard = (
            "if _tcgen05_subtile != 0 and tcgen05_warp_idx == cutlass.Int32(0):"
        )
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        rmem_store = "tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        first_barrier = "tcgen05_epilog_sync_barrier.arrive_and_wait()"

        self.assertEqual(code.count(acquire), 2, code)
        self.assertEqual(code.count(later_subtile_guard), 1, code)
        first_acquire_pos = code.find(acquire)
        gmem_tile_pos = code.find(gmem_tile)
        subtile_loop_pos = code.find(subtile_loop, gmem_tile_pos)
        acc_wait_pos = code.find(acc_wait, subtile_loop_pos)
        rmem_store_pos = code.find(rmem_store, acc_wait_pos)
        later_subtile_guard_pos = code.find(later_subtile_guard, rmem_store_pos)
        later_acquire_pos = code.find(acquire, later_subtile_guard_pos)
        first_barrier_pos = code.find(first_barrier, later_acquire_pos)
        for needle, pos in (
            ("pre-loop C acquire", first_acquire_pos),
            (gmem_tile, gmem_tile_pos),
            (subtile_loop, subtile_loop_pos),
            (acc_wait, acc_wait_pos),
            (rmem_store, rmem_store_pos),
            (later_subtile_guard, later_subtile_guard_pos),
            ("delayed later C acquire", later_acquire_pos),
            (first_barrier, first_barrier_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        self.assertLess(first_acquire_pos, gmem_tile_pos, code)
        self.assertLess(gmem_tile_pos, subtile_loop_pos, code)
        self.assertLess(subtile_loop_pos, acc_wait_pos, code)
        self.assertLess(acc_wait_pos, rmem_store_pos, code)
        self.assertLess(rmem_store_pos, later_subtile_guard_pos, code)
        self.assertLess(later_subtile_guard_pos, later_acquire_pos, code)
        self.assertLess(later_acquire_pos, first_barrier_pos, code)

    def test_tcgen05_diagnostic_acc_wait_moves_before_subtile_loop(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_acc_wait_placement="before_subtile_loop",
            )
            code = bound.to_triton_code(cfg)

        acquire = "tcgen05_c_pipeline.producer_acquire()"
        gmem_tile = "tcgen05_gC = cute.local_tile("
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        subtile_loop = "for _tcgen05_subtile in cutlass.range("
        subtile_acc_wait = (
            "if _tcgen05_subtile == 0:\n"
            "            tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        )
        later_subtile_guard = (
            "if _tcgen05_subtile != 0 and tcgen05_warp_idx == cutlass.Int32(0):"
        )
        t2r_copy = "cute.copy(tcgen05_tiled_copy_t2r"

        self.assertEqual(code.count(acc_wait), 1, code)
        self.assertNotIn(subtile_acc_wait, code)
        first_acquire_pos = code.find(acquire)
        gmem_tile_pos = code.find(gmem_tile)
        acc_wait_pos = code.find(acc_wait, gmem_tile_pos)
        subtile_loop_pos = code.find(subtile_loop, acc_wait_pos)
        later_subtile_guard_pos = code.find(later_subtile_guard, subtile_loop_pos)
        later_acquire_pos = code.find(acquire, later_subtile_guard_pos)
        t2r_copy_pos = code.find(t2r_copy, later_acquire_pos)
        for needle, pos in (
            ("pre-loop C acquire", first_acquire_pos),
            (gmem_tile, gmem_tile_pos),
            (acc_wait, acc_wait_pos),
            (subtile_loop, subtile_loop_pos),
            (later_subtile_guard, later_subtile_guard_pos),
            ("later C acquire", later_acquire_pos),
            (t2r_copy, t2r_copy_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        self.assertLess(first_acquire_pos, gmem_tile_pos, code)
        self.assertLess(gmem_tile_pos, acc_wait_pos, code)
        self.assertLess(acc_wait_pos, subtile_loop_pos, code)
        self.assertLess(subtile_loop_pos, later_subtile_guard_pos, code)
        self.assertLess(later_subtile_guard_pos, later_acquire_pos, code)
        self.assertLess(later_acquire_pos, t2r_copy_pos, code)

    def test_tcgen05_diagnostic_split_first_t2r_layout_splits_subtile_loop(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(512, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_c_acquire_placement="first_in_loop",
                tcgen05_epilogue_layout="split_first_t2r",
            )
            code = bound.to_triton_code(cfg)

        epi_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
        )
        role_src, _, _ = self._role_local_while_source_for_predicate(
            code,
            ast.parse(code),
            epi_role_predicate,
        )
        first_subtile_assign = "_tcgen05_subtile = 0"
        split_loop = (
            "for _tcgen05_split_subtile in cutlass.range("
            "tcgen05_subtile_count - 1, unroll_full=True):"
        )
        tail_subtile_assign = "_tcgen05_subtile = _tcgen05_split_subtile + 1"
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        t2r_copy = "cute.copy(tcgen05_tiled_copy_t2r"
        r2s_source_store = "tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        first_barrier = "tcgen05_epilog_sync_barrier.arrive_and_wait()"
        c_acquire = "tcgen05_c_pipeline.producer_acquire()"
        c_commit = "tcgen05_c_pipeline.producer_commit()"
        acc_release = (
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"
        )

        first_assign_pos = role_src.find(first_subtile_assign)
        acc_wait_pos = role_src.find(acc_wait, first_assign_pos)
        first_t2r_pos = role_src.find(t2r_copy, acc_wait_pos)
        first_r2s_store_pos = role_src.find(r2s_source_store, first_t2r_pos)
        first_barrier_pos = role_src.find(first_barrier, first_r2s_store_pos)
        first_commit_pos = role_src.find(c_commit, first_barrier_pos)
        split_loop_pos = role_src.find(split_loop, first_commit_pos)
        tail_assign_pos = role_src.find(tail_subtile_assign, split_loop_pos)
        tail_t2r_pos = role_src.find(t2r_copy, tail_assign_pos)
        first_acquire_pos = role_src.find(c_acquire)
        first_release_pos = role_src.find(acc_release, first_t2r_pos)

        for needle, pos in (
            (first_subtile_assign, first_assign_pos),
            (acc_wait, acc_wait_pos),
            (t2r_copy, first_t2r_pos),
            (r2s_source_store, first_r2s_store_pos),
            (first_barrier, first_barrier_pos),
            (c_commit, first_commit_pos),
            (split_loop, split_loop_pos),
            (tail_subtile_assign, tail_assign_pos),
            ("tail T2R copy", tail_t2r_pos),
            ("first C acquire", first_acquire_pos),
            (acc_release, first_release_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{role_src}")
        self.assertLess(first_acquire_pos, first_r2s_store_pos, role_src)
        self.assertLess(first_assign_pos, acc_wait_pos, role_src)
        self.assertLess(acc_wait_pos, first_t2r_pos, role_src)
        self.assertLess(first_t2r_pos, first_release_pos, role_src)
        self.assertLess(first_release_pos, first_barrier_pos, role_src)
        self.assertLess(first_barrier_pos, first_commit_pos, role_src)
        self.assertLess(first_commit_pos, split_loop_pos, role_src)
        self.assertLess(split_loop_pos, tail_assign_pos, role_src)
        self.assertLess(tail_assign_pos, tail_t2r_pos, role_src)
        self.assertEqual(role_src.count(acc_wait), 1, role_src)
        self.assertNotIn("if _tcgen05_subtile == 0 and", role_src)

    def test_tcgen05_diagnostic_split_acc_t2r_store_tail_layout_splits_regions(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(512, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="split_acc_t2r_store_tail",
            )
            code = bound.to_triton_code(cfg)

        epi_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
        )
        role_src, _, _ = self._role_local_while_source_for_predicate(
            code,
            ast.parse(code),
            epi_role_predicate,
        )
        subtile_loop = (
            "for _tcgen05_subtile in cutlass.range("
            "tcgen05_subtile_count, unroll_full=True):"
        )
        later_guard = "if _tcgen05_subtile != 0 and"
        c_acquire = "tcgen05_c_pipeline.producer_acquire()"
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        t2r_copy = "cute.copy(tcgen05_tiled_copy_t2r"
        acc_release = (
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"
        )
        r2s_source_store = "tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        first_barrier = "tcgen05_epilog_sync_barrier.arrive_and_wait()"
        r2s_copy = "cute.copy(tcgen05_tiled_copy_r2s"
        shared_fence = "cute.arch.fence_view_async_shared()"
        tma_store_copy = "cute.copy(tcgen05_tma_store_atom"
        c_commit = "tcgen05_c_pipeline.producer_commit()"

        subtile_loop_pos = role_src.find(subtile_loop)
        epi_active_pos = role_src.find("if tcgen05_epi_active:", subtile_loop_pos)
        c_region_pos = role_src.find("if True:", epi_active_pos)
        later_guard_pos = role_src.find(later_guard, c_region_pos)
        later_acquire_pos = role_src.find(c_acquire, later_guard_pos)
        acc_region_pos = role_src.find("if True:", later_acquire_pos)
        acc_wait_pos = role_src.find(acc_wait, acc_region_pos)
        t2r_copy_pos = role_src.find(t2r_copy, acc_wait_pos)
        acc_release_pos = role_src.find(acc_release, t2r_copy_pos)
        r2s_source_store_pos = role_src.find(r2s_source_store, t2r_copy_pos)
        tail_region_pos = role_src.find("if True:", r2s_source_store_pos)
        first_barrier_pos = role_src.find(first_barrier, tail_region_pos)
        r2s_copy_pos = role_src.find(r2s_copy, first_barrier_pos)
        shared_fence_pos = role_src.find(shared_fence, r2s_copy_pos)
        second_barrier_pos = role_src.find(first_barrier, shared_fence_pos)
        tma_store_copy_pos = role_src.find(tma_store_copy, second_barrier_pos)
        c_commit_pos = role_src.find(c_commit, tma_store_copy_pos)

        for needle, pos in (
            (subtile_loop, subtile_loop_pos),
            ("epilogue active guard", epi_active_pos),
            ("C-stage acquire source boundary", c_region_pos),
            (later_guard, later_guard_pos),
            (c_acquire, later_acquire_pos),
            ("accumulator wait / T2R source boundary", acc_region_pos),
            (acc_wait, acc_wait_pos),
            (t2r_copy, t2r_copy_pos),
            (acc_release, acc_release_pos),
            (r2s_source_store, r2s_source_store_pos),
            ("C-store barrier / TMA tail source boundary", tail_region_pos),
            (first_barrier, first_barrier_pos),
            (r2s_copy, r2s_copy_pos),
            (shared_fence, shared_fence_pos),
            ("second epilogue barrier", second_barrier_pos),
            (tma_store_copy, tma_store_copy_pos),
            (c_commit, c_commit_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{role_src}")
        self.assertLess(subtile_loop_pos, c_region_pos, role_src)
        self.assertLess(c_region_pos, later_guard_pos, role_src)
        self.assertLess(later_guard_pos, later_acquire_pos, role_src)
        self.assertLess(later_acquire_pos, acc_region_pos, role_src)
        self.assertLess(acc_region_pos, acc_wait_pos, role_src)
        self.assertLess(acc_wait_pos, t2r_copy_pos, role_src)
        self.assertLess(t2r_copy_pos, acc_release_pos, role_src)
        self.assertLess(acc_release_pos, first_barrier_pos, role_src)
        self.assertLess(t2r_copy_pos, r2s_source_store_pos, role_src)
        self.assertLess(r2s_source_store_pos, tail_region_pos, role_src)
        self.assertLess(tail_region_pos, first_barrier_pos, role_src)
        self.assertLess(first_barrier_pos, r2s_copy_pos, role_src)
        self.assertLess(r2s_copy_pos, shared_fence_pos, role_src)
        self.assertLess(shared_fence_pos, second_barrier_pos, role_src)
        self.assertLess(second_barrier_pos, tma_store_copy_pos, role_src)
        self.assertLess(tma_store_copy_pos, c_commit_pos, role_src)
        self.assertEqual(role_src.count("if True:"), 4, role_src)
        self.assertNotIn("_tcgen05_split_subtile", role_src)

    def test_tcgen05_diagnostic_module_helper_acc_t2r_layout_uses_module_helper(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(512, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="module_helper_acc_t2r",
            )
            code = bound.to_triton_code(cfg)

        tree = ast.parse(code)
        helper_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("tcgen05_acc_t2r_region")
        ]
        self.assertEqual(len(helper_defs), 1, code)
        helper_name = helper_defs[0].name
        helper_src = ast.get_source_segment(code, helper_defs[0])
        self.assertIsNotNone(helper_src)
        assert helper_src is not None
        self.assertIn(f"@cute.jit\ndef {helper_name}(", code)
        helper_def_pos = code.find(f"def {helper_name}(")
        kernel_pos = code.find("@cute.kernel")
        self.assertLess(helper_def_pos, kernel_pos, code)

        epi_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
        )
        role_src, _, _ = self._role_local_while_source_for_predicate(
            code,
            tree,
            epi_role_predicate,
        )
        helper_call_pos = role_src.find(f"{helper_name}(")
        first_barrier = "tcgen05_epilog_sync_barrier.arrive_and_wait()"
        first_barrier_pos = role_src.find(first_barrier, helper_call_pos)
        c_commit_pos = role_src.find(
            "tcgen05_c_pipeline.producer_commit()",
            first_barrier_pos,
        )
        for needle, pos in (
            (f"def {helper_name}(", helper_def_pos),
            (f"{helper_name}(", helper_call_pos),
            (first_barrier, first_barrier_pos),
            ("tcgen05_c_pipeline.producer_commit()", c_commit_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        self.assertIn(
            "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)",
            helper_src,
        )
        self.assertIn("cute.copy(tcgen05_tiled_copy_t2r", helper_src)
        self.assertIn(
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)",
            helper_src,
        )
        self.assertIn("tcgen05_tRS_rD.store(tcgen05_acc_vec)", helper_src)
        self.assertLess(helper_call_pos, first_barrier_pos, role_src)
        self.assertLess(first_barrier_pos, c_commit_pos, role_src)
        self.assertNotIn(
            "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)",
            role_src,
        )
        self.assertNotIn("cute.copy(tcgen05_tiled_copy_t2r", role_src)

    def test_tcgen05_diagnostic_module_helper_store_tail_layout_uses_module_helper(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(512, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="module_helper_store_tail",
            )
            code = bound.to_triton_code(cfg)

        tree = ast.parse(code)
        helper_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("tcgen05_store_tail_region")
        ]
        self.assertEqual(len(helper_defs), 1, code)
        helper_name = helper_defs[0].name
        helper_src = ast.get_source_segment(code, helper_defs[0])
        self.assertIsNotNone(helper_src)
        assert helper_src is not None
        self.assertIn(f"@cute.jit\ndef {helper_name}(", code)
        helper_def_pos = code.find(f"def {helper_name}(")
        kernel_pos = code.find("@cute.kernel")
        self.assertLess(helper_def_pos, kernel_pos, code)

        epi_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
        )
        role_src, _, _ = self._role_local_while_source_for_predicate(
            code,
            tree,
            epi_role_predicate,
        )
        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        t2r_copy = "cute.copy(tcgen05_tiled_copy_t2r"
        r2s_source_store = "tcgen05_tRS_rD.store(tcgen05_acc_vec)"
        helper_call_pos = role_src.find(f"{helper_name}(")
        acc_wait_pos = role_src.find(acc_wait)
        t2r_copy_pos = role_src.find(t2r_copy, acc_wait_pos)
        r2s_source_store_pos = role_src.find(r2s_source_store, t2r_copy_pos)

        for needle, pos in (
            (f"def {helper_name}(", helper_def_pos),
            (acc_wait, acc_wait_pos),
            (t2r_copy, t2r_copy_pos),
            (r2s_source_store, r2s_source_store_pos),
            (f"{helper_name}(", helper_call_pos),
        ):
            self.assertGreaterEqual(pos, 0, f"Missing {needle!r} in:\n{code}")
        self.assertIn("tcgen05_epilog_sync_barrier.arrive_and_wait()", helper_src)
        self.assertIn("cute.copy(tcgen05_tiled_copy_r2s", helper_src)
        self.assertIn("cute.arch.fence_view_async_shared()", helper_src)
        self.assertIn("cute.copy(tcgen05_tma_store_atom", helper_src)
        self.assertIn("tcgen05_c_pipeline.producer_commit()", helper_src)
        self.assertLess(acc_wait_pos, t2r_copy_pos, role_src)
        self.assertLess(t2r_copy_pos, r2s_source_store_pos, role_src)
        self.assertLess(r2s_source_store_pos, helper_call_pos, role_src)
        self.assertNotIn("tcgen05_epilog_sync_barrier.arrive_and_wait()", role_src)
        self.assertNotIn("cute.copy(tcgen05_tiled_copy_r2s", role_src)
        self.assertNotIn("cute.copy(tcgen05_tma_store_atom", role_src)

    def test_tcgen05_diagnostic_split_first_t2r_rejects_flat_epilogue(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_epilogue_layout="split_first_t2r",
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "requires the role-local TMA-store tcgen05 epilogue",
            ):
                bound.to_triton_code(cfg)

    def test_tcgen05_diagnostic_split_first_t2r_rejects_one_subtile_epilogue(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_one_subtile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 128, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_one_subtile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 128, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="split_first_t2r",
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "CtaGroup.TWO block_n >= 256",
            ):
                bound.to_triton_code(cfg)

    def test_tcgen05_diagnostic_split_first_t2r_rejects_one_cta_epilogue(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m1_role_local(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m1_role_local.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=1,
                tcgen05_epilogue_layout="split_first_t2r",
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "requires CtaGroup.TWO",
            ):
                bound.to_triton_code(cfg)

    def test_tcgen05_diagnostic_split_first_t2r_runtime_correctness(
        self,
    ) -> None:
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_split_first_t2r(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_split_first_t2r.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="split_first_t2r",
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn(
                "for _tcgen05_split_subtile in cutlass.range("
                "tcgen05_subtile_count - 1, unroll_full=True):",
                code,
            )
            out = bound(*args)

        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_diagnostic_split_acc_t2r_store_tail_runtime_correctness(
        self,
    ) -> None:
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_split_acc_t2r_store_tail(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_split_acc_t2r_store_tail.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="split_acc_t2r_store_tail",
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn(
                "if True:\n"
                "                            if _tcgen05_subtile == 0:\n"
                "                                tcgen05_acc_pipeline.consumer_wait",
                code,
            )
            out = bound(*args)

        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_diagnostic_module_helper_acc_t2r_runtime_correctness(
        self,
    ) -> None:
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_module_helper_acc_t2r(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_module_helper_acc_t2r.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="module_helper_acc_t2r",
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("def tcgen05_acc_t2r_region", code)
            out = bound(*args)

        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_diagnostic_module_helper_store_tail_runtime_correctness(
        self,
    ) -> None:
        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_module_helper_store_tail(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_module_helper_store_tail.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_epilogue_layout="module_helper_store_tail",
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("def tcgen05_store_tail_region", code)
            out = bound(*args)

        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_diagnostic_skip_epilogue_store_drains_acc_pipeline(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_flat_tma_store(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_tma_store.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            unsafe_cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_c_store_mode="skip_epilogue_store",
            )
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "tcgen05_diagnostic_invalid_output",
            ):
                bound.to_triton_code(unsafe_cfg)
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="flat",
                tcgen05_c_store_mode="skip_epilogue_store",
                tcgen05_diagnostic_invalid_output=True,
            )
            code = bound.to_triton_code(cfg)

        acc_wait = "tcgen05_acc_pipeline.consumer_wait(tcgen05_acc_consumer_state)"
        acc_release = (
            "tcgen05_acc_pipeline.consumer_release(tcgen05_acc_consumer_state)"
        )
        c_acquire = "tcgen05_c_pipeline.producer_acquire()"
        c_commit = "tcgen05_c_pipeline.producer_commit()"
        tma_store = "cute.copy(tcgen05_tma_store_atom"
        r2s_copy = "cute.copy(tcgen05_tiled_copy_r2s"

        self.assertIn(acc_wait, code)
        self.assertIn(acc_release, code)
        self.assertNotIn(c_acquire, code)
        self.assertNotIn(c_commit, code)
        self.assertNotIn(tma_store, code)
        self.assertNotIn(r2s_copy, code)

    def test_tcgen05_diagnostic_skip_umma_keeps_pipeline_handshakes(
        self,
    ) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_persistent_role(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(128, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 128, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_role.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            unsafe_cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
                tcgen05_acc_producer_mode="skip_umma",
            )
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "tcgen05_diagnostic_invalid_output",
            ):
                bound.to_triton_code(unsafe_cfg)
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
                tcgen05_acc_producer_mode="skip_umma",
                tcgen05_diagnostic_invalid_output=True,
            )
            code = bound.to_triton_code(cfg)

        exec_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4)"
        )
        role_src, _, _ = self._role_local_while_source_for_predicate(
            code,
            ast.parse(code),
            exec_role_predicate,
        )
        self.assertIn(
            "tcgen05_acc_pipeline.producer_acquire(tcgen05_acc_producer_state)",
            role_src,
        )
        self.assertIn("consumer_try_wait(tcgen05_ab_consumer_state)", role_src)
        self.assertIn("consumer_release(tcgen05_ab_consumer_state)", role_src)
        self.assertIn(
            "tcgen05_acc_pipeline.producer_commit(tcgen05_acc_producer_state)",
            role_src,
        )
        self.assertIn("tcgen05_acc_producer_state.advance()", role_src)
        self.assertNotIn("cute.gemm(", role_src)
        self.assertNotIn("cute.arch.fence_view_async_shared()", role_src)
        self.assertNotIn("cute.nvgpu.tcgen05.Field.ACCUMULATE, True", role_src)

    def test_tcgen05_persistent_partial_single_tile_keeps_shared_tma_path(
        self,
    ) -> None:
        """Partial-tile TMA fallback keeps the legacy shared-loop shape.

        The role-local mainloop path assumes static full tiles and drops
        scalar fallback from the extracted producer/exec K-loops. Partial
        tiles still need the scalar fallback barriers in the shared loop, so
        static edge shapes must not opt into the role-local path.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_persistent_partial(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(96, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 96, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_persistent_partial.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
            )
            code = bound.to_triton_code(cfg)

        self.assertIn("while tcgen05_work_tile_valid", code)
        self.assertNotIn("tcgen05_role_local", code)
        self.assertNotIn("'kind': 'tcgen05_d_tma'", code)
        self.assertIn("tcgen05_tma_warp", code)
        self.assertIn("cute.arch.sync_threads()", code)
        self.assertIn("if tcgen05_tma_full_tile", code)

    def test_tcgen05_persistent_multi_tile_runtime_correctness(self) -> None:
        """Static full-tile persistent + tcgen05 is correct for multi-tile shapes.

        The tcgen05 static scheduler seeds work from ``block_idx.z``. The
        persistent launch grid must therefore be ``(cluster_m, 1,
        persistent_clusters)`` rather than the generic 1D ``(_NUM_SM,)``
        persistent grid. The chosen shape has more scheduler work clusters
        than SMs and more than one M/N tile, so at least one CTA must
        execute ``advance_to_next_work()`` across a 2D output tile space.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_multi_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        num_sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        m_tiles = 2
        n_tiles = (num_sms + m_tiles - 1) // m_tiles + 8
        self.assertGreater(m_tiles * n_tiles, num_sms)
        m = m_tiles * 128
        n = n_tiles * 128
        k = 32

        for dtype in (torch.float16, torch.bfloat16):
            for pid_type in ("persistent_blocked", "persistent_interleaved"):
                with self.subTest(dtype=str(dtype), pid_type=pid_type):
                    torch.manual_seed(0)
                    args = (
                        torch.randn(m, k, device=DEVICE, dtype=dtype),
                        torch.randn(k, n, device=DEVICE, dtype=dtype),
                    )
                    with patch_cute_mma_support():
                        bound = cute_matmul_multi_tile.bind(args)
                        bound.env.config_spec.cute_tcgen05_search_enabled = True
                        cfg = _make_tcgen05_persistent_config(
                            block_sizes=[128, 128, 16],
                            pid_type=pid_type,
                        )
                        bound.set_config(cfg)
                        code = bound.to_triton_code(cfg)
                        self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
                        self.assertRegex(code, r"\(\s*1\s*,\s*1\s*,\s*min\s*\(")
                        self.assertIn("cutlass.pipeline.PipelineTmaStore.create", code)
                        self.assertNotIn("cute.nvgpu.CopyUniversalOp()", code)
                        out = bound(*args)
                    expected = args[0] @ args[1]
                    # Match the single-tile tcgen05 runtime test tolerance:
                    # this test targets persistent scheduler state, not a
                    # new accumulator-precision contract.
                    torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_ab_stages_three_runtime_correctness(self) -> None:
        """``tcgen05_ab_stages=3`` runs end-to-end without deadlock.

        Pre-cycle-17 the initial-prefetch emission only filled stage 0
        and stage ``ab_stage_count - 1``. For ``ab_stages=3`` that left
        stage 1 empty; the K-loop's consumer ``consumer_wait`` on stage
        1 phase 0 deadlocked on the first iteration. This runtime test
        defends directly against the deadlock regression by binding
        seeded inputs, executing the kernel, and asserting closeness
        against the torch reference. See ``cute_plan.md §6.9.1``.

        Pinned codegen markers (``num_stages=3``, intermediate-stage
        gate variable, per-stage prefetch ``k_offset=0,1,2``) are also
        checked so a producer-side regression that codegens fine but
        re-introduces the prefetch gap fails fast at the codegen layer
        before consuming the runtime budget.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_ab3(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        # K=384 = 3 * bk so the K loop iterates exactly ab_stages times,
        # guaranteeing the consumer_wait reaches every prefetched stage.
        # M=N=256 matches the validated CtaGroup.TWO seed tile.
        m, n, k = 256, 256, 384
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(0)
                args = (
                    torch.randn(m, k, device=DEVICE, dtype=dtype),
                    torch.randn(k, n, device=DEVICE, dtype=dtype),
                )
                with patch_cute_mma_support():
                    bound = cute_matmul_ab3.bind(args)
                    bound.env.config_spec.cute_tcgen05_search_enabled = True
                    cfg = _make_tcgen05_persistent_config(
                        block_sizes=[256, 256, 128],
                        pid_type="persistent_interleaved",
                        tcgen05_cluster_m=2,
                        tcgen05_ab_stages=3,
                    )
                    code = bound.to_triton_code(cfg)
                    # Codegen markers: pipeline depth + per-stage prefetch.
                    self.assertIn("PipelineTmaUmma.create(num_stages=3", code)
                    for k_offset in (0, 1, 2):
                        self.assertIn(f"tma_gA[None, cutlass.Int32({k_offset})]", code)
                        self.assertIn(f"tma_gB[None, cutlass.Int32({k_offset})]", code)
                    self.assertIn(
                        "tma_gA[None, tcgen05_tma_k_tile + cutlass.Int32(3)]",
                        code,
                    )
                    self.assertIn("tcgen05_tma_initial_stage_1_full_tile", code)
                    bound.set_config(cfg)
                    out = bound(*args)
                # If the deadlock regresses, the kernel hangs (CI test
                # timeout fires); if it returns, correctness must hold.
                expected = args[0] @ args[1]
                torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_partial_multi_tile_runtime_guard(self) -> None:
        """Partial legacy persistent + tcgen05 still raises ``RuntimeError``.

        Static full tiles use role-local persistent loops and have multi-tile
        coverage. Partial K/M/N shapes stay on the legacy shared TMA fallback
        path, which still launch-fails for multi-tile shapes; keep the
        host-side guard for that non-role-local path.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_partial_multi_tile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        # K=17 is not a static full tile for bk=16, so role-local extraction
        # stays disabled. M/N are 2x2 tiles, so the legacy multi-tile guard
        # must fire before the CUDA launch.
        args = (
            torch.randn(256, 17, device=DEVICE, dtype=torch.float16),
            torch.randn(17, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_partial_multi_tile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("_helion_tcgen05_persistent_total_tiles", code)
            self.assertNotIn("tcgen05_role_local", code)
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

    def test_tcgen05_persistent_cluster_m2_multi_tile_runtime_guard(self) -> None:
        """CtaGroup.ONE cluster_m=2 fallback remains guarded."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_multi_tile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_multi_tile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[128, 128, 16],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertNotIn("tcgen05_role_local", code)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
            self.assertIn(
                "tcgen05_ab_pipeline_consumer_group = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))",
                code,
            )
            self.assertNotIn(
                "tcgen05_ab_pipeline_consumer_group = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(2))",
                code,
            )
            self.assertIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_shape_codegen(self) -> None:
        """Pin the selected clustered CtaGroup.ONE bridge target.

        The explicit 128x256x128 cluster_m=2 config must stay clustered and
        use CtaGroup.ONE, not demote to cluster_m=1 or switch to CtaGroup.TWO.
        Runtime remains guarded until this clustered shape is validated.
        """

        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_cta_group_one(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_cta_group_one.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config()
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

        self.assertEqual(
            cfg.config["indexing"],
            ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
        )
        self.assertEqual(cfg.config["range_flattens"], [None, None])
        self.assertEqual(cfg.config["range_multi_buffers"], [None, None])
        self.assertEqual(cfg.config["range_warp_specializes"], [None, None])
        self.assertEqual(cfg.config["range_num_stages"], [0, 0])
        self.assertEqual(cfg.config["range_unroll_factors"], [0, 0])
        self.assertIn("_helion_cute_cluster_shape = (2, 1, 1)", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        self.assertNotIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertIn(
            "tcgen05_cluster_layout_vmnk = cute.tiled_divide("
            "cute.make_layout((2, 1, 1))",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))",
            code,
        )
        self.assertNotIn(
            "tcgen05_ab_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(2))",
            code,
        )
        self.assertIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_role_local_codegen(
        self,
    ) -> None:
        """Diagnostic bridge maps role-local PIDs by CTA rank but stays guarded."""

        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_cta_group_one_role_local(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_cta_group_one_role_local.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
                **{TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY: True}
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

        self.assertIn("_helion_cute_cluster_shape = (2, 1, 1)", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        self.assertNotIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertIn("tcgen05_role_local_0_work_tile", code)
        self.assertIn("tcgen05_role_local_1_work_tile", code)
        self.assertIn("tcgen05_role_local_2_work_tile", code)
        cta_rank_expr = "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())"
        for role_idx in range(3):
            self.assertIn(
                "virtual_pid = "
                f"tcgen05_role_local_{role_idx}_work_tile.tile_idx[0] + "
                f"{cta_rank_expr}",
                code,
            )
        tma_role_predicate = (
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
        )
        code_ast = ast.parse(code)
        kernel_def = None
        tma_role_matches: list[str] = []
        for node in ast.walk(code_ast):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name
                == "_helion_cute_matmul_cluster_m2_cta_group_one_role_local"
            ):
                kernel_def = node
            if (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == tma_role_predicate
            ):
                tma_role_matches.append(ast.unparse(node))
        self.assertIsNotNone(kernel_def, code)
        assert kernel_def is not None
        self.assertEqual(len(tma_role_matches), 1, code)
        tma_role_src = tma_role_matches[0]
        self.assertIn("cute.copy(tma_atom_a", tma_role_src)
        self.assertIn("cute.copy(tma_atom_b", tma_role_src)
        self.assertNotIn("mcast_mask=tcgen05_a_mcast_mask", tma_role_src)
        self.assertIn("mcast_mask=tcgen05_b_mcast_mask", tma_role_src)
        self.assertIn(
            "tcgen05_b_mcast_mask = "
            "cute.nvgpu.cpasync.create_tma_multicast_mask("
            "tcgen05_cluster_layout_vmnk, "
            "tcgen05_block_in_cluster_coord_vmnk, mcast_mode=2)",
            code,
        )
        self.assertNotIn(_TCGEN05_CLUSTER_LEADER_PREDICATE, tma_role_src)
        self.assertIn(
            "tcgen05_ab_pipeline_producer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, 1)",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline_tx_count = "
            "cute.size_in_bytes(cutlass.Float16, sA_tma_layout) + "
            "cute.size_in_bytes(cutlass.Float16, sB_tma_layout)",
            code,
        )
        self.assertNotIn("* cute.size(tiled_mma.thr_id.shape)", code)
        self.assertIn(
            "tcgen05_acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create("
            "num_stages=1, producer_group=tcgen05_acc_pipeline_producer_group, "
            "consumer_group=tcgen05_acc_pipeline_consumer_group, "
            "barrier_storage=tcgen05_acc_pipeline_barriers, "
            "cta_layout_vmnk=tcgen05_cluster_layout_vmnk)",
            code,
        )
        self.assertIn(
            "tcgen05_ab_pipeline = cutlass.pipeline.PipelineTmaUmma.create("
            "num_stages=2, producer_group=tcgen05_ab_pipeline_producer_group, "
            "consumer_group=tcgen05_ab_pipeline_consumer_group, "
            "tx_count=tcgen05_ab_pipeline_tx_count, "
            "barrier_storage=tcgen05_ab_pipeline_mbars, "
            "cta_layout_vmnk=tcgen05_cluster_layout_vmnk)",
            code,
        )
        self.assertNotIn("defer_sync=True", code)
        self.assertNotIn("cutlass.pipeline.pipeline_init_arrive(", code)
        self.assertNotIn("cutlass.pipeline.pipeline_init_wait(", code)
        self.assertNotIn("while tcgen05_work_tile_valid", code)
        self.assertIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)

    def _cluster_m2_cta_group_one_bridge_diagnostic_code(
        self, mode_key: str, mode_value: str
    ) -> str:
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_bridge_diagnostic(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_bridge_diagnostic.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            unsafe_cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
                **{
                    TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY: True,
                    mode_key: mode_value,
                }
            )
            with self.assertRaisesRegex(
                exc.InvalidConfig,
                "tcgen05_diagnostic_invalid_output",
            ):
                bound.to_triton_code(unsafe_cfg)
            guarded_cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
                **{
                    mode_key: mode_value,
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "128x256x128 bridge shape",
            ):
                bound.to_triton_code(guarded_cfg)
            cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
                **{
                    TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY: True,
                    mode_key: mode_value,
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

        self.assertIn("_helion_cute_cluster_shape = (2, 1, 1)", code)
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        self.assertIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)
        return code

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_skip_acc_advance(
        self,
    ) -> None:
        """Bridge diagnostic removes only the acc producer advance edge."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
            TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP,
        )
        self.assertIn("tcgen05_role_local_1_work_tile", code)
        self.assertIn(
            "tcgen05_acc_pipeline.producer_commit(tcgen05_acc_producer_state)",
            code,
        )
        self.assertNotIn("tcgen05_acc_producer_state.advance()", code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_skip_ab_acquire(
        self,
    ) -> None:
        """Bridge diagnostic removes only AB producer acquire edges."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
            TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP,
        )
        self.assertNotIn("tcgen05_ab_pipeline.producer_acquire(", code)
        self.assertNotIn("tcgen05_ab_pipeline.producer_try_acquire(", code)
        self.assertIn("tcgen05_ab_pipeline.producer_get_barrier(", code)
        self.assertIn("tcgen05_ab_pipeline.producer_commit(", code)
        self.assertIn("tcgen05_ab_producer_state.advance()", code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_skip_initial_ab_acquire(
        self,
    ) -> None:
        """Bridge diagnostic removes only the first initial AB acquire."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
            TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST,
        )
        tree = ast.parse(code)
        stage0_blocks = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if (
                "tcgen05_tma_initial_full_tile" in test_src
                and "tcgen05_tma_initial_next_full_tile" not in test_src
                and "tcgen05_tma_warp" in test_src
            ):
                stage0_blocks.append(node)
        self.assertEqual(len(stage0_blocks), 1, code)
        stage0_src = ast.unparse(
            ast.Module(body=stage0_blocks[0].body, type_ignores=[])
        )
        self.assertNotIn("tcgen05_ab_pipeline.producer_acquire(", stage0_src)
        self.assertIn("tcgen05_ab_pipeline.producer_get_barrier(", stage0_src)
        self.assertIn("tcgen05_ab_pipeline.producer_commit(", stage0_src)
        self.assertEqual(code.count("tcgen05_ab_pipeline.producer_acquire("), 2)
        self.assertEqual(code.count("tcgen05_ab_pipeline.producer_try_acquire("), 1)
        self.assertIn("tcgen05_ab_producer_state.advance()", code)
        self.assertIn("tcgen05_ab_pipeline.producer_tail(", code)

    def test_tcgen05_large_bn_proof_config_validation(self) -> None:
        """Larger-BN proof flag is explicit and tcgen05-only."""

        config_spec = ConfigSpec(backend=CuteBackend())
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05-enabled CuTe matmul kernels",
        ):
            config_spec.normalize({TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True})

        config_spec.cute_tcgen05_search_enabled = True
        for block_id, size_hint in enumerate(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES):
            config_spec.block_sizes.append(
                BlockSizeSpec(
                    block_id=block_id,
                    size_hint=size_hint,
                    max_size=size_hint,
                )
            )
        with self.assertRaisesRegex(exc.InvalidConfig, "must be a boolean"):
            config_spec.normalize(
                {
                    "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
                    "tcgen05_cluster_m": TCGEN05_LARGE_BN_PROOF_CLUSTER_M,
                    **dict(TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS),
                    TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: "yes",
                }
            )

        config = {
            "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
            "tcgen05_cluster_m": TCGEN05_LARGE_BN_PROOF_CLUSTER_M,
            **dict(TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS),
            TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True,
        }
        config_spec.normalize(config)
        self.assertIs(config[TCGEN05_LARGE_BN_PROOF_CONFIG_KEY], True)
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_ab_stages=2, tcgen05_acc_stages=1, and tcgen05_c_stages=2",
        ):
            config_spec.normalize(
                {
                    "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
                    "tcgen05_cluster_m": 2,
                    **dict(TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS),
                    TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_ab_stages=2, tcgen05_acc_stages=1, and tcgen05_c_stages=2",
        ):
            config_spec.normalize(
                {
                    "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
                    "tcgen05_cluster_m": TCGEN05_LARGE_BN_PROOF_CLUSTER_M,
                    "pid_type": "persistent_blocked",
                    **dict(TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS),
                    TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_ab_stages=2, tcgen05_acc_stages=1, and tcgen05_c_stages=2",
        ):
            config_spec.normalize(
                {
                    "block_sizes": list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES),
                    "tcgen05_cluster_m": TCGEN05_LARGE_BN_PROOF_CLUSTER_M,
                    "tcgen05_ab_stages": 2,
                    "tcgen05_acc_stages": 2,
                    "tcgen05_c_stages": 2,
                    TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True,
                }
            )

    def test_tcgen05_ab_initial_producer_acquire_mode_config_validation(
        self,
    ) -> None:
        """Invalid-output initial AB acquire diagnostic is rejected unless opted in."""

        config_spec = ConfigSpec(backend=CuteBackend())
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05-enabled CuTe matmul kernels",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST
                    ),
                }
            )

        config_spec.cute_tcgen05_search_enabled = True
        with self.assertRaisesRegex(exc.InvalidConfig, "must be one of"):
            config_spec.normalize(
                {
                    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY: "invalid",
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_diagnostic_invalid_output",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST
                    ),
                }
            )

    def test_tcgen05_ab_producer_advance_mode_config_validation(self) -> None:
        """Invalid-output AB advance diagnostic is rejected unless opted in."""

        config_spec = ConfigSpec(backend=CuteBackend())
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05-enabled CuTe matmul kernels",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP
                    ),
                }
            )

        config_spec.cute_tcgen05_search_enabled = True
        with self.assertRaisesRegex(exc.InvalidConfig, "must be one of"):
            config_spec.normalize(
                {
                    TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY: "invalid",
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_diagnostic_invalid_output",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP
                    ),
                }
            )

    def test_tcgen05_ab_consumer_wait_mode_config_validation(self) -> None:
        """Invalid-output AB wait diagnostic is rejected unless opted in."""

        config_spec = ConfigSpec(backend=CuteBackend())
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05-enabled CuTe matmul kernels",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY: (
                        TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP
                    ),
                }
            )

        config_spec.cute_tcgen05_search_enabled = True
        with self.assertRaisesRegex(exc.InvalidConfig, "must be one of"):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY: "invalid",
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_diagnostic_invalid_output",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY: (
                        TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP
                    ),
                }
            )

    def test_tcgen05_ab_consumer_phase_mode_config_validation(self) -> None:
        """Invalid-output AB phase diagnostic is rejected unless opted in."""

        config_spec = ConfigSpec(backend=CuteBackend())
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05-enabled CuTe matmul kernels",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1
                    ),
                }
            )

        config_spec.cute_tcgen05_search_enabled = True
        with self.assertRaisesRegex(exc.InvalidConfig, "must be one of"):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY: "invalid",
                    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY: True,
                }
            )
        with self.assertRaisesRegex(
            exc.InvalidConfig,
            "tcgen05_diagnostic_invalid_output",
        ):
            config_spec.normalize(
                {
                    TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY: (
                        TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1
                    ),
                }
            )

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_skip_ab_advance(
        self,
    ) -> None:
        """Bridge diagnostic removes every AB producer advance edge."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
            TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP,
        )
        self.assertIn("tcgen05_ab_pipeline.producer_acquire(", code)
        self.assertIn("tcgen05_ab_pipeline.producer_get_barrier(", code)
        self.assertIn("tcgen05_ab_pipeline.producer_commit(", code)
        self.assertNotIn("tcgen05_ab_producer_state.advance()", code)
        self.assertNotIn(
            "tcgen05_ab_pipeline.producer_tail(tcgen05_ab_producer_state)",
            code,
        )
        self.assertNotIn("tcgen05_ab_producer_state._count", code)
        self.assertNotIn("tcgen05_ab_producer_state._index", code)
        self.assertNotIn("tcgen05_ab_producer_state._phase", code)
        self.assertIn("tcgen05_acc_producer_state.advance()", code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_skip_ab_wait(
        self,
    ) -> None:
        """Bridge diagnostic removes only AB consumer try-wait/wait edges."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
            TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP,
        )
        self.assertIn("tcgen05_ab_pipeline.producer_commit(", code)
        self.assertNotIn("tcgen05_ab_pipeline.consumer_try_wait(", code)
        self.assertNotIn("tcgen05_ab_pipeline.consumer_wait(", code)
        self.assertIn("tcgen05_ab_pipeline.consumer_release(", code)
        self.assertIn("tcgen05_ab_consumer_state.advance()", code)
        self.assertIn("tcgen05_ab_producer_state.advance()", code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_can_init_ab_phase1(
        self,
    ) -> None:
        """Bridge diagnostic initializes AB consumer state with phase 1."""

        code = self._cluster_m2_cta_group_one_bridge_diagnostic_code(
            TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
            TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1,
        )
        self.assertNotIn(
            "tcgen05_ab_consumer_state = "
            "cutlass.pipeline.make_pipeline_state("
            "cutlass.pipeline.PipelineUserType.Consumer, 2)",
            code,
        )
        self.assertIn(
            "tcgen05_ab_consumer_state = "
            "cutlass.pipeline.PipelineState(2, cutlass.Int32(0), "
            "cutlass.Int32(0), cutlass.Int32(1))",
            code,
        )
        self.assertIn("tcgen05_ab_pipeline.consumer_try_wait(", code)
        self.assertIn("tcgen05_ab_pipeline.consumer_wait(", code)
        self.assertIn("tcgen05_ab_pipeline.consumer_release(", code)
        self.assertIn("tcgen05_ab_consumer_state.advance()", code)
        self.assertIn("tcgen05_ab_pipeline_tx_count = ", code)

    def test_tcgen05_cluster_m2_cta_group_one_bridge_role_local_shape_guard(
        self,
    ) -> None:
        """The bridge diagnostic fails loudly outside its exact shape."""

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_cta_group_one_wrong_shape(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=torch.float16),
            torch.randn(128, 128, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_cta_group_one_wrong_shape.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_cluster_m2_cta_group_one_bridge_config(
                block_sizes=[128, 128, 128],
                **{TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY: True},
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "128x256x128 bridge shape",
            ):
                bound.to_triton_code(cfg)

    def test_tcgen05_persistent_cluster_m2_partial_single_tile_runtime_guard(
        self,
    ) -> None:
        """Partial single-tile cluster_m=2 fallback remains guarded."""

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_partial_single_tile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 17, device=DEVICE, dtype=torch.float16),
            torch.randn(17, 256, device=DEVICE, dtype=torch.float16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_partial_single_tile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertNotIn("tcgen05_role_local", code)
            self.assertIn(Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR, code)
            self.assertIn(
                f"if {Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR} > 0:",
                code,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "supports runtime execution only",
            ):
                bound(*args)

    def test_tcgen05_persistent_cluster_m2_two_cta_pdl_codegen(self) -> None:
        """CtaGroup.TWO emits Quack-aligned PDL markers in validated codegen."""

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_pdl(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta_pdl.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)

        self.assertEqual(code.count("cute.arch.griddepcontrol_wait()"), 1)
        self.assertEqual(code.count("cute.arch.griddepcontrol_launch_dependents()"), 1)
        tma_role_pos = code.index(
            "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5):"
        )
        wait_pos = code.index("cute.arch.griddepcontrol_wait()", tma_role_pos)
        tma_sched_pos = code.index(
            "StaticPersistentTileScheduler.create(", tma_role_pos
        )
        tma_acquire_pos = code.index(
            "tcgen05_ab_pipeline.producer_acquire", tma_role_pos
        )
        pdl_launch_pos = code.index("cute.arch.griddepcontrol_launch_dependents()")
        tmem_arrive_pos = code.index("tcgen05_tmem_alloc_barrier.arrive()")

        self.assertLess(wait_pos, tma_sched_pos)
        self.assertLess(wait_pos, tma_acquire_pos)
        self.assertLess(pdl_launch_pos, tmem_arrive_pos)

    def test_tcgen05_persistent_cluster_m2_two_cta_single_tile_runtime_correctness(
        self,
    ) -> None:
        """Single-output-tile CtaGroup.TWO codegen runs correctly.

        The same role-local scheduler path is used for single-tile and
        scheduler-recycling CtaGroup.TWO; this case covers the AB TMA/UMMA
        ownership and tail-drain sequence with one logical tile.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(256, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                # Grouped PID decomposition leaves pure scalar setup in the
                # omitted shared view; single-tile runtime should compile and
                # run instead of tripping the omit-safety assertion.
                l2_groupings=[4],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertEqual(code.count("cute.arch.griddepcontrol_wait()"), 1)
            self.assertEqual(
                code.count("cute.arch.griddepcontrol_launch_dependents()"), 1
            )
            self.assertIn(
                "tcgen05_role_local_0_tile_sched = "
                "cutlass.utils.StaticPersistentTileScheduler.create(",
                code,
            )
            self.assertIn(
                "virtual_pid = tcgen05_role_local_0_work_tile.tile_idx[0] "
                "// cutlass.Int32(2)",
                code,
            )
            self.assertIn("StaticPersistentTileScheduler.create", code)
            self.assertIn(
                "while tcgen05_role_local_0_work_tile.is_valid_tile",
                code,
            )
            self.assertIn(
                "tcgen05_ab_pipeline_consumer_group = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(1))",
                code,
            )
            self.assertIn(
                "tcgen05_acc_pipeline_consumer_group = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread, cutlass.Int32(8))",
                code,
            )
            self.assertIn(
                "tcgen05_acc_pipeline = "
                "cutlass.pipeline.PipelineUmmaAsync.create("
                "num_stages=2, "
                "producer_group=tcgen05_acc_pipeline_producer_group, "
                "consumer_group=tcgen05_acc_pipeline_consumer_group, "
                "barrier_storage=tcgen05_acc_pipeline_barriers, "
                "cta_layout_vmnk=tcgen05_cluster_layout_vmnk, "
                "defer_sync=True)",
                code,
            )
            self.assertIn(
                "tcgen05_ab_pipeline_tx_count = "
                "(cute.size_in_bytes(cutlass.BFloat16, sA_tma_layout) + "
                "cute.size_in_bytes(cutlass.BFloat16, sB_tma_layout)) * "
                "cute.size(tiled_mma.thr_id.shape)",
                code,
            )
            self.assertIn(
                "tcgen05_ab_pipeline = "
                "cutlass.pipeline.PipelineTmaUmma.create("
                "num_stages=2, "
                "producer_group=tcgen05_ab_pipeline_producer_group, "
                "consumer_group=tcgen05_ab_pipeline_consumer_group, "
                "tx_count=tcgen05_ab_pipeline_tx_count, "
                "barrier_storage=tcgen05_ab_pipeline_mbars, "
                "cta_layout_vmnk=tcgen05_cluster_layout_vmnk, "
                "defer_sync=True)",
                code,
            )
            tmem_arrive_pos = code.index("tcgen05_tmem_alloc_barrier.arrive()")
            pdl_launch_pos = code.index("cute.arch.griddepcontrol_launch_dependents()")
            acc_tail_marker = (
                "tcgen05_acc_pipeline.producer_tail"
                if "tcgen05_acc_pipeline.producer_tail" in code
                else "tcgen05_acc_pipeline.producer_acquire(tcgen05_acc_producer_state)"
            )
            acc_tail_pos = code.index(acc_tail_marker)
            tmem_dealloc_allocator_pos = code.index(
                "num_allocated_columns=tcgen05_acc_tmem_cols"
            )
            relinquish_pos = code.index(
                "tcgen05_tmem_allocator.relinquish_alloc_permit()"
            )
            tmem_wait_pos = code.index("tcgen05_tmem_alloc_barrier.arrive_and_wait()")
            tmem_free_pos = code.index("tcgen05_tmem_allocator.free(")
            self.assertLess(pdl_launch_pos, tmem_arrive_pos)
            self.assertLess(tmem_arrive_pos, acc_tail_pos)
            self.assertLess(acc_tail_pos, tmem_dealloc_allocator_pos)
            self.assertLess(tmem_dealloc_allocator_pos, relinquish_pos)
            self.assertLess(relinquish_pos, tmem_wait_pos)
            self.assertLess(tmem_wait_pos, tmem_free_pos)
            self.assertNotIn(
                "cute.arch.sync_threads()",
                code[tmem_dealloc_allocator_pos:relinquish_pos],
            )
            init_arrive = "cutlass.pipeline.pipeline_init_arrive("
            init_wait = "cutlass.pipeline.pipeline_init_wait("
            self.assertLess(
                code.index(
                    "tcgen05_acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create("
                ),
                code.index(init_arrive),
            )
            self.assertLess(
                code.index(
                    "tcgen05_ab_pipeline = cutlass.pipeline.PipelineTmaUmma.create("
                ),
                code.index(init_arrive),
            )
            self.assertLess(code.index(init_arrive), code.index(init_wait))
            self.assertLess(
                code.index(init_wait),
                code.index("tcgen05_tmem_allocator.allocate("),
            )
            self.assertNotIn("tcgen05_sched_pipeline", code)
            self.assertNotIn("tcgen05_work_tile_smem", code)
            self.assertNotIn("tcgen05_tile_sched.advance_to_next_work()", code)
            tma_role_predicate = (
                "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
            )
            exec_role_predicate = (
                "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(4)"
            )
            epi_role_predicate = (
                "cute.arch.make_warp_uniform(cute.arch.warp_idx()) < cutlass.Int32(4)"
            )
            tree = ast.parse(code)
            self._assert_role_local_c_store_pipeline_lifetime(
                code, tree, epi_role_predicate
            )
            found_role_local_tma = False
            found_role_local_exec = False
            found_role_local_epi = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                test_src = ast.unparse(node.test)
                role_src = ast.unparse(node)
                if test_src == tma_role_predicate:
                    self.assertLess(
                        role_src.index("cute.arch.griddepcontrol_wait()"),
                        role_src.index("StaticPersistentTileScheduler.create("),
                    )
                    self.assertLess(
                        role_src.index("cute.arch.griddepcontrol_wait()"),
                        role_src.index("tcgen05_ab_pipeline.producer_acquire"),
                    )
                    self.assertIn("tcgen05_ab_pipeline.producer_acquire", role_src)
                    self.assertIn(
                        "cute.nvgpu.cpasync.tma_partition("
                        "tma_atom_a, tma_a_cta_coord, tma_a_cta_layout",
                        role_src,
                    )
                    self.assertIn(
                        "cute.nvgpu.cpasync.tma_partition("
                        "tma_atom_b, tma_b_cta_coord, tma_b_cta_layout",
                        role_src,
                    )
                    self.assertIn("cute.copy(tma_atom_a", role_src)
                    self.assertNotIn(
                        f"if {_TCGEN05_CLUSTER_LEADER_PREDICATE}", role_src
                    )
                    found_role_local_tma = True
                elif test_src == exec_role_predicate:
                    first_ab_peek_pos = role_src.index(
                        "tcgen05_ab_pipeline.consumer_try_wait"
                    )
                    acc_acquire_pos = role_src.index(
                        "tcgen05_acc_pipeline.producer_acquire"
                    )
                    accumulate_reset_pos = role_src.index(
                        "tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)"
                    )
                    self.assertLess(first_ab_peek_pos, acc_acquire_pos)
                    self.assertLess(acc_acquire_pos, accumulate_reset_pos)
                    self.assertLess(
                        accumulate_reset_pos,
                        role_src.index("tcgen05_ab_pipeline.consumer_wait"),
                    )
                    self.assertIn("tcgen05_ab_pipeline.consumer_wait", role_src)
                    self.assertIn("cute.gemm(", role_src)
                    self.assertIn(
                        "tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)",
                        role_src,
                    )
                    self.assertIn("tcgen05_ab_pipeline.consumer_release", role_src)
                    self.assertIn("tcgen05_ab_consumer_state.advance()", role_src)
                    release_pos = role_src.index("tcgen05_ab_pipeline.consumer_release")
                    advance_pos = role_src.index("tcgen05_ab_consumer_state.advance()")
                    next_ab_peek_blocks = [
                        ast.unparse(child)
                        for child in ast.walk(node)
                        if isinstance(child, ast.If)
                        and "tcgen05_tma_next_consumer_tile" in ast.unparse(child.test)
                        and _TCGEN05_CLUSTER_LEADER_PREDICATE in ast.unparse(child.test)
                    ]
                    self.assertTrue(
                        next_ab_peek_blocks,
                        "Expected a leader-gated AB consumer next-token peek. "
                        "Generated role code:\n" + role_src,
                    )
                    next_ab_peek_pos = role_src.index(
                        "tcgen05_tma_next_consumer_tile", advance_pos
                    )
                    next_ab_try_pos = role_src.index(
                        "tcgen05_ab_pipeline.consumer_try_wait",
                        next_ab_peek_pos,
                    )
                    self.assertLess(release_pos, advance_pos)
                    self.assertLess(advance_pos, next_ab_peek_pos)
                    self.assertLess(next_ab_peek_pos, next_ab_try_pos)
                    leader_blocks = [
                        ast.unparse(child)
                        for child in ast.walk(node)
                        if isinstance(child, ast.If)
                        and _TCGEN05_CLUSTER_LEADER_PREDICATE in ast.unparse(child.test)
                    ]
                    self.assertTrue(
                        any(
                            "tcgen05_acc_pipeline.producer_acquire("
                            "tcgen05_acc_producer_state)" in block
                            for block in leader_blocks
                        ),
                        "Expected the CtaGroup.TWO exec owner to acquire "
                        "the acc producer state. Generated role code:\n" + role_src,
                    )
                    self.assertTrue(
                        any(
                            "tcgen05_acc_pipeline.producer_commit("
                            "tcgen05_acc_producer_state)" in block
                            for block in leader_blocks
                        ),
                        "Expected the CtaGroup.TWO exec owner to commit "
                        "the acc producer state. Generated role code:\n" + role_src,
                    )
                    self.assertTrue(
                        any(
                            "tcgen05_ab_pipeline.consumer_release("
                            "tcgen05_ab_consumer_state)" in block
                            for block in leader_blocks
                        ),
                        "AB empty release must be leader-owned so the "
                        "PipelineTmaUmma multicast mask supplies the peer "
                        "CTA's empty signal exactly once. Generated "
                        "role code:\n" + role_src,
                    )
                    self.assertIn("tcgen05_acc_producer_state.advance()", role_src)
                    found_role_local_exec = True
                elif test_src == epi_role_predicate:
                    self.assertIn("tcgen05_acc_pipeline.consumer_wait", role_src)
                    self.assertNotIn("PipelineTmaStore.create", role_src)
                    self.assertIn("cute.nvgpu.cpasync.tma_partition", role_src)
                    self.assertIn("cute.copy(tcgen05_tma_store_atom", role_src)
                    self.assertIn("tcgen05_tma_store_role_tile", role_src)
                    self.assertNotIn("cute.nvgpu.CopyUniversalOp()", role_src)
                    self._assert_tma_store_epilogue_order(role_src, require_tail=False)
                    found_role_local_epi = True
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.While)
                    and ast.unparse(node.test) == "tcgen05_work_tile_valid"
                ):
                    continue
                self.fail(
                    "Fully role-local CtaGroup.TWO codegen should not emit "
                    "the residual shared scheduler loop. Generated code:\n" + code
                )
            self.assertTrue(
                found_role_local_tma,
                "Expected role-local TMA-load work in CtaGroup.TWO codegen. "
                "Generated code:\n" + code,
            )
            self.assertTrue(
                found_role_local_exec,
                "Expected role-local MMA-exec work in CtaGroup.TWO codegen. "
                "Generated code:\n" + code,
            )
            self.assertTrue(
                found_role_local_epi,
                "Expected role-local TMA-store epilogue work in CtaGroup.TWO "
                "codegen. Generated code:\n" + code,
            )
            total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
            self.assertNotIn(total_var, code)
            out = bound(*args)
        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_m2_two_cta_multi_tile_runtime_correctness(
        self,
    ) -> None:
        """Small multi-tile CtaGroup.TWO codegen runs correctly."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_multi_tile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            with self.subTest(pid_type=pid_type):
                torch.manual_seed(0)
                args = (
                    torch.randn(512, 32, device=DEVICE, dtype=torch.bfloat16),
                    torch.randn(32, 512, device=DEVICE, dtype=torch.bfloat16),
                )
                with patch_cute_mma_support():
                    bound = cute_matmul_cluster_m2_two_cta_multi_tile.bind(args)
                    bound.env.config_spec.cute_tcgen05_search_enabled = True
                    cfg = _make_tcgen05_persistent_config(
                        block_sizes=[256, 256, 16],
                        l2_groupings=[4],
                        pid_type=pid_type,
                        tcgen05_cluster_m=2,
                    )
                    bound.set_config(cfg)
                    code = bound.to_triton_code(cfg)
                    self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
                    self.assertIn(
                        "tcgen05_role_local_0_tile_sched = "
                        "cutlass.utils.StaticPersistentTileScheduler.create(",
                        code,
                    )
                    self.assertIn(
                        "while tcgen05_role_local_0_work_tile.is_valid_tile",
                        code,
                    )
                    self.assertIn("StaticPersistentTileScheduler.create", code)
                    total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
                    self.assertNotIn(total_var, code)
                    out = bound(*args)
                expected = args[0] @ args[1]
                torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_n2_codegen_emits_4cta_cluster(
        self,
    ) -> None:
        """G2 cluster_n=2 codegen markers (cute_plan.md §6.12.7).

        Pins the codegen surface for the canonical Quack-best 4-CTA
        cluster: ``cluster_m=2 cluster_n=2 use_2cta=True``. Asserts:

        - ``cluster_layout_vmnk`` shape becomes ``(2, 2, 1)``.
        - The launch grid x/y dims become ``(2, 2, ...)`` (not
          ``(2, 1, ...)``).
        - The PersistentTileSchedulerParams cluster shape is
          ``(2, 2, 1)``.
        - The AB consumer arrive count becomes 2 (mcast_size formula at
          V=2: ``num_mcast_ctas_a + num_mcast_ctas_b - 1 = 2 + 1 - 1``).
        - The V-leader gate emits ``% cutlass.Int32(2) == cutlass.Int32(0)``
          for AB consumer-release / MMA issue (instead of the cluster_n=1
          ``== cutlass.Int32(0)`` form).
        - The plan persists ``cluster_n=2``.

        This is a *codegen* test only — the runtime correctness test is
        ``test_tcgen05_persistent_cluster_n2_two_cta_runtime_correctness``.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_n2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        # Use a problem big enough that both M and N divide by
        # cluster_m * bm = 512 and cluster_n * bn = 512 respectively.
        args = (
            torch.randn(1024, 128, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(128, 1024, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_n2.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 128],
                l2_groupings=[1],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=2,
                tcgen05_ab_stages=2,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)

        # CtaGroup.TWO selection still applies (cluster_m=2 + bm=256).
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        # cluster_layout_vmnk shape is (2, 2, 1).
        self.assertIn(
            "tcgen05_cluster_layout_vmnk = cute.tiled_divide("
            "cute.make_layout((2, 2, 1)), (tiled_mma.thr_id.shape,))",
            code,
        )
        # PersistentTileSchedulerParams cluster shape is (2, 2, 1).
        # Anchor the cluster-shape literal on the trailing call closer
        # ``(2, 2, 1))`` so a regression that swaps two of the three
        # (2, 2, 1) literal sites (cluster_layout_vmnk, scheduler params,
        # launch grid) still fails this assertion individually.
        self.assertIn(
            "cutlass.utils.PersistentTileSchedulerParams(",
            code,
        )
        self.assertIn(", (2, 2, 1))", code)
        # AB consumer arrive count is 2 (mcast_size formula at V=2
        # cluster_n=2: 2 + 1 - 1).
        self.assertIn(
            "tcgen05_ab_pipeline_consumer_group = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(2))",
            code,
        )
        # V-leader gate uses ``% Int32(2) == Int32(0)`` (not bare
        # ``== Int32(0)``) for AB-pair / MMA owner predicates.
        self.assertIn(
            "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
            "% cutlass.Int32(2) == cutlass.Int32(0)",
            code,
        )
        # Launch-grid x/y match the cluster shape ``(2, 2, ...)``;
        # anchor on the launcher signature so a regression of the
        # cluster_n axis in ``codegen_grid`` fails loudly.
        self.assertRegex(code, r"_launcher\([^,]+,\s*\(2,\s*2,\s*")
        # Launch-grid persistent-capacity divisor is ``_NUM_SM // 2``
        # (cluster_m), NOT ``_NUM_SM // 4`` (cluster_size = cluster_m
        # * cluster_n). The latter caps cluster_n=2 at one wave of
        # 37 cluster slots × 4 CTAs and starves the GPU of wave-overlap
        # parallelism; the cluster_m divisor matches the cluster_n=1
        # baseline so each cluster slot still consumes one SM. Anchor
        # on the ``max(1, _NUM_SM // 2)`` envelope (the launcher emits
        # ``min(total_clusters, max(1, _NUM_SM // cluster_m))`` for
        # this seed) so a regression of either branch (``// 4``
        # re-introduced or ``// 2`` lost) fails this assertion.
        self.assertIn("max(1, _NUM_SM // 2)", code)
        self.assertNotIn("_NUM_SM // 4", code)
        # cluster_n=2 plumbing is exercised end-to-end by the codegen
        # markers above (no plan-field round-trip is asserted here —
        # the validator-surface coverage lives in
        # ``test_cute_tcgen05_strategy_invariants_cluster_n``).

    def test_tcgen05_persistent_cluster_n2_two_cta_runtime_correctness(
        self,
    ) -> None:
        """G2 cluster_n=2 runtime correctness (cute_plan.md §6.12.7 step 5).

        Smoke test for the canonical Quack-best 4-CTA cluster
        ``cluster_m=2 cluster_n=2 use_2cta=True`` at a small shape so
        the test runs in CI: 1024x1024x128 bf16. The V-leader gate
        (cute_plan.md §6.12.3) is the load-bearing fix; without it the
        kernel hangs (cycle 26 reproducer).

        If this test hangs, the V-leader gate / arrive-count plumbing
        is wrong somewhere. This is the *only* end-to-end gate that
        catches a regression of the hang fix.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_n2_runtime(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        # 1024x1024x128 with bm=256 bn=256 bk=128 → 4x4 logical tiles,
        # which the 4-CTA cluster splits into 2x2 cluster slots × 4 K
        # tiles. Big enough to exercise multiple K iterations and
        # multiple cluster work tiles.
        args = (
            torch.randn(1024, 128, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(128, 1024, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_cluster_n2_runtime.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 128],
                l2_groupings=[1],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=2,
                tcgen05_ab_stages=2,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
            )
            bound.set_config(cfg)
            out = bound(*args)
        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_m2_two_cta_grid_caps_for_recycling(
        self,
    ) -> None:
        """CtaGroup.TWO launch grid caps at persistent work-cluster capacity."""

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_grid(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(512, 32, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(32, 512, device=DEVICE, dtype=torch.bfloat16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta_grid.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                num_sm_multiplier=2,
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertIn(
                "tcgen05_role_local_0_tile_sched = "
                "cutlass.utils.StaticPersistentTileScheduler.create(",
                code,
            )
            self.assertIn("StaticPersistentTileScheduler.create", code)
            self.assertIn(
                "while tcgen05_role_local_0_work_tile.is_valid_tile",
                code,
            )
            total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
            self.assertNotIn(total_var, code)
            launcher_lines = [
                line
                for line in code.splitlines()
                if "_launcher(" in line and "_helion_cute" in line
            ]
            self.assertEqual(len(launcher_lines), 1, code)
            self.assertRegex(launcher_lines[0], r"_launcher\([^,]+,\s*\(2,\s*1,")
            self.assertIn("_NUM_SM", launcher_lines[0])
            self.assertRegex(launcher_lines[0], r"//\s*2")
            self.assertIn("min(", launcher_lines[0])

    def test_tcgen05_persistent_cluster_m2_two_cta_grid_z_limit_uses_recycling(
        self,
    ) -> None:
        """CtaGroup.TWO recycling avoids direct-grid z-limit launches."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_grid_z_limit(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(3072, 32, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(32, 3072, device=DEVICE, dtype=torch.bfloat16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta_grid_z_limit.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertNotIn(total_var, code)
            self.assertNotIn("no more than 65535 output tiles", code)
            self.assertIn(
                "tcgen05_role_local_0_tile_sched = "
                "cutlass.utils.StaticPersistentTileScheduler.create(",
                code,
            )
            self.assertIn("StaticPersistentTileScheduler.create", code)
            launcher_lines = [
                line
                for line in code.splitlines()
                if "_launcher(" in line and "_helion_cute" in line
            ]
            self.assertEqual(len(launcher_lines), 1, code)
            self.assertIn("_NUM_SM // 2", launcher_lines[0])
            self.assertIn("min(", launcher_lines[0])
            out = bound(*args)
        expected = args[0] @ args[1]
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_m2_two_cta_large_runtime_correctness(
        self,
    ) -> None:
        """Large CtaGroup.TWO multi-tile codegen recycles scheduler state."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_large_multi_tile(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        torch.manual_seed(0)
        args = (
            torch.randn(4096, 32, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(32, 2048, device=DEVICE, dtype=torch.bfloat16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta_large_multi_tile.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                l2_groupings=[4],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertIn(
                "tcgen05_role_local_0_tile_sched = "
                "cutlass.utils.StaticPersistentTileScheduler.create(",
                code,
            )
            self.assertIn(
                "while tcgen05_role_local_0_work_tile.is_valid_tile",
                code,
            )
            self.assertIn("StaticPersistentTileScheduler.create", code)
            total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
            self.assertNotIn(total_var, code)
            first = bound(*args)
            second = bound(*args)
        expected = args[0] @ args[1]
        torch.testing.assert_close(first, expected, atol=2e-1, rtol=1e-2)
        torch.testing.assert_close(second, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_m2_two_cta_k_cap_runtime_correctness(
        self,
    ) -> None:
        """CtaGroup.TWO codegen runs correctly at the validated K cap."""

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_long_k(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        for k_size in (64, 256, 4096):
            with self.subTest(k_size=k_size):
                torch.manual_seed(0)
                args = (
                    torch.randn(256, k_size, device=DEVICE, dtype=torch.bfloat16),
                    torch.randn(k_size, 256, device=DEVICE, dtype=torch.bfloat16),
                )
                with patch_cute_mma_support():
                    bound = cute_matmul_cluster_m2_two_cta_long_k.bind(args)
                    bound.env.config_spec.cute_tcgen05_search_enabled = True
                    cfg = _make_tcgen05_persistent_config(
                        block_sizes=[256, 256, 16],
                        pid_type="persistent_blocked",
                        tcgen05_cluster_m=2,
                    )
                    bound.set_config(cfg)
                    code = bound.to_triton_code(cfg)
                    self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
                    self.assertIn(
                        "tcgen05_role_local_0_tile_sched = "
                        "cutlass.utils.StaticPersistentTileScheduler.create(",
                        code,
                    )
                    self.assertIn(
                        "while tcgen05_role_local_0_work_tile.is_valid_tile",
                        code,
                    )
                    self.assertIn("StaticPersistentTileScheduler.create", code)
                    total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
                    self.assertNotIn(total_var, code)
                    self.assertNotIn(
                        f"if {Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR} > 0:",
                        code,
                    )
                    out = bound(*args)
                expected = args[0] @ args[1]
                torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_persistent_cluster_m2_two_cta_k_tile_limit_guard(
        self,
    ) -> None:
        """CtaGroup.TWO shapes above the validated K-tile cap stay guarded."""

        @helion.kernel(backend="cute")
        def cute_matmul_cluster_m2_two_cta_k_tile_limit(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 4112, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(4112, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        with patch_cute_mma_support():
            bound = cute_matmul_cluster_m2_two_cta_k_tile_limit.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                pid_type="persistent_blocked",
                tcgen05_cluster_m=2,
            )
            bound.set_config(cfg)
            code = bound.to_triton_code(cfg)
            self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
            self.assertIn(
                f"if {Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR} > 0:",
                code,
            )
            self.assertIn("above the validated K-tile limit", code)
            with self.assertRaisesRegex(
                RuntimeError,
                "above the validated K-tile limit",
            ):
                bound(*args)

    def test_tcgen05_flat_cluster_m2_two_cta_rejected(self) -> None:
        """Flat CtaGroup.TWO configs do not have the persistent host guard."""

        @helion.kernel(backend="cute")
        def cute_matmul_flat_cluster_m2(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float16),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_flat_cluster_m2.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.env.config_spec.restrict_tcgen05_cluster_m_search((2, 1))
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                pid_type="flat",
                tcgen05_cluster_m=2,
            )
            with self.assertRaisesRegex(
                exc.BackendUnsupported,
                "tcgen05_cluster_m > 1",
            ):
                bound.to_triton_code(cfg)

    def test_non_tcgen05_flat_cluster_m2_fallback_is_allowed(self) -> None:
        """tcgen05_cluster_m is irrelevant after falling back from tcgen05."""

        @helion.kernel(backend="cute")
        def cute_matmul_float32_cluster_m2(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 32, device=DEVICE, dtype=torch.float32),
            torch.randn(32, 256, device=DEVICE, dtype=torch.float32),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_float32_cluster_m2.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.env.config_spec.restrict_tcgen05_cluster_m_search((2, 1))
            cfg = _make_tcgen05_persistent_config(
                block_sizes=[256, 256, 16],
                pid_type="flat",
                tcgen05_cluster_m=2,
            )
            code = bound.to_triton_code(cfg)

        self.assertIn("_launcher(_helion_cute_matmul_float32_cluster_m2", code)
        self.assertNotIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertNotIn("_helion_cute_cluster_shape = (2, 1, 1)", code)

    def test_tcgen05_persistent_single_tile_runtime_correctness(self) -> None:
        """Persistent + tcgen05 produces correct output for single-tile shapes.

        The mma_stage / suffix per-tile tagging fixes (mma_stage =
        consumer_state.index, suffix per-tile tagging) made single-tile-
        per-CTA cases correct on B200: ``128x128xK`` for any K, including
        K=48 with 3 K-iterations. This test runs the kernel for those
        documented shapes and checks against ATen; multi-tile static
        full-shape coverage lives in
        ``test_tcgen05_persistent_multi_tile_runtime_correctness``.
        """

        from helion._compiler.cute.mma_support import get_cute_mma_support

        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        @helion.kernel(backend="cute")
        def cute_matmul_single_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        # 128 // 128 = 1 tile each axis -> 1 total work tile -> single-tile
        # path. The cross product of K, dtype, pid_type matters: prior bugs
        # only manifested with specific combinations of these.
        # - K size (16, 32, 48, 64): 16 hits exactly one K-iter; 32, 48, 64
        #   hit multi-K-iter and previously failed with mma_stage desync.
        # - dtype (fp16, bf16): both go through the same tcgen05 codegen
        #   path, but pinning bf16 catches dtype-specific lowering
        #   regressions.
        # - pid_type ('persistent_blocked', 'persistent_interleaved'):
        #   both share ``Tcgen05PersistentProgramIDs`` so the host-side
        #   guard and per-tile tagging apply to both.
        for k_size in (16, 32, 48, 64):
            for dtype in (torch.float16, torch.bfloat16):
                for pid_type in (
                    "persistent_blocked",
                    "persistent_interleaved",
                ):
                    with self.subTest(K=k_size, dtype=str(dtype), pid_type=pid_type):
                        x = torch.randn(128, k_size, device=DEVICE, dtype=dtype)
                        y = torch.randn(k_size, 128, device=DEVICE, dtype=dtype)

                        bound = cute_matmul_single_tile.bind((x, y))
                        bound.env.config_spec.cute_tcgen05_search_enabled = True
                        cfg = helion.Config(
                            block_sizes=[128, 128, 16],
                            l2_groupings=[1],
                            loop_orders=[[0, 1]],
                            num_stages=2,
                            num_warps=8,
                            pid_type=pid_type,
                            tcgen05_cluster_m=1,
                            tcgen05_ab_stages=2,
                            tcgen05_acc_stages=2,
                            tcgen05_c_stages=2,
                            tcgen05_num_epi_warps=4,
                        )
                        bound.set_config(cfg)
                        out = bound(x, y)
                        expected = x @ y
                        # Single-tile cases should be exact: the bug was
                        # a stage desync that only surfaced across tile
                        # boundaries.
                        torch.testing.assert_close(out, expected, atol=2e-1, rtol=1e-2)

    def test_tcgen05_codegen_registers_post_loop_cleanup(self) -> None:
        """``_codegen_cute_store_tcgen05_tile`` returns a list whose tail
        is the one-shot cleanup (TMA / acc producer_tail, TMEM allocator
        setup + free) tagged via ``register_cute_tcgen05_post_loop_stmts``.

        This is what lets the persistent splitter pull those statements
        out of the work-tile loop. Without the registration the
        persistent path corrupts pipeline state because each virtual tile
        runs the drain after its own subtile loop, and the subsequent
        tile's ``producer_acquire`` then sees an already-drained pipeline.
        """

        @helion.kernel(backend="cute")
        def cute_matmul_post_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 128, device=DEVICE, dtype=torch.float16),
        )
        with patch_cute_mma_support():
            bound = cute_matmul_post_loop.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = _make_tcgen05_persistent_config(block_sizes=[128, 128, 16])
            code = bound.to_triton_code(cfg)

        # The cleanup statements are the post-loop tail. They must be
        # emitted exactly once on the non-persistent path (= here, since
        # this config sets pid_type='flat').
        self.assertEqual(code.count("tcgen05_acc_pipeline.producer_tail"), 1)
        self.assertEqual(code.count("tcgen05_tmem_allocator.free"), 1)
        # The post-loop ``TmemAllocator(...)`` is the second one (the
        # first is the in-prefix instance for ``allocate``); it must be
        # the variant that asks the runtime to skip the dealloc-mbarrier
        # init that the prologue allocator already performed.
        self.assertTrue(
            "dealloc_mbarrier_initialized=True" in code
            or "initialize_mbarrier=False" in code,
            f"expected dealloc-mbarrier skip kwarg in code: {code!r}",
        )

    def test_tcgen05_store_rejects_num_epi_warps_not_four(self) -> None:
        """Codegen backstop fires when ``epi_warp_count != 4`` slips past
        ``Config.normalize()`` validation (rationale in cute_plan.md §2)."""

        @helion.kernel(backend="cute")
        def cute_matmul_epi_check(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(16, 128, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch_cute_mma_support():
            for n_epi in (1, 2):
                bound = cute_matmul_epi_check.bind(args)
                bound.env.config_spec.cute_tcgen05_search_enabled = True
                bound.env.config_spec._tcgen05_num_epi_warps_validation_choices = None
                cfg = helion.Config(
                    block_sizes=[128, 128, 16],
                    l2_groupings=[1],
                    loop_orders=[[0, 1]],
                    num_stages=2,
                    num_warps=8,
                    pid_type="flat",
                    tcgen05_cluster_m=1,
                    tcgen05_ab_stages=2,
                    tcgen05_acc_stages=2,
                    tcgen05_c_stages=2,
                    tcgen05_num_epi_warps=n_epi,
                )
                with self.assertRaises(exc.BackendUnsupported) as cm:
                    bound.to_triton_code(cfg)
                msg = str(cm.exception)
                self.assertIn("tcgen05_num_epi_warps=4", msg)
                self.assertIn(f"got {n_epi}", msg)
                self.assertIn("tmem_warp_shape_mn=(4,1)", msg)
                self.assertIn("cute_plan.md", msg)

        with patch_cute_mma_support():
            bound = cute_matmul_epi_check.bind(args)
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            cfg = helion.Config(
                block_sizes=[128, 128, 16],
                l2_groupings=[1],
                loop_orders=[[0, 1]],
                num_stages=2,
                num_warps=8,
                pid_type="flat",
                tcgen05_cluster_m=1,
                tcgen05_ab_stages=2,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                tcgen05_num_epi_warps=4,
            )
            code = bound.to_triton_code(cfg)
            self.assertIn(
                "'epilogue_warp_id': (cutlass.Int32(0), cutlass.Int32(1), "
                "cutlass.Int32(2), cutlass.Int32(3))",
                code,
            )

    def test_tcgen05_setmaxregister_omitted_without_tcgen05(self) -> None:
        """The non-tcgen05 (warp / universal) MMA paths do not emit
        setmaxregister. The Quack-style register split is specific to the
        warp-specialized tcgen05 producer/consumer roles."""

        @helion.kernel(backend="cute")
        def cute_matmul_warp_only(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 64, device=DEVICE, dtype=torch.float16),
        )
        # Force the warp impl by reporting only warp_f16bf16 support.
        with patch_cute_mma_support(
            default_cute_mma_support(
                supported_impls=("universal", "warp"),
                tcgen05_f16bf16=False,
            )
        ):
            bound = cute_matmul_warp_only.bind(args)
            config = bound.config_spec.default_config()
            code = bound.to_triton_code(config)

        # No setmaxregister either way on the warp path: no producer/consumer
        # split exists.
        self.assertNotIn("setmaxregister_decrease", code)
        self.assertNotIn("setmaxregister_increase", code)

    def test_tcgen05_default_store_arrives_with_exec_warp(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
        )

        with patch_cute_mma_support():
            code = cute_matmul_mma_codegen_only.bind(args).to_triton_code(
                helion.Config(
                    block_sizes=[128, 32, 16],
                    l2_groupings=[4],
                    loop_orders=[[0, 1]],
                    num_stages=2,
                    num_warps=4,
                    pid_type="flat",
                )
            )

        self.assertIn("if tcgen05_exec_active:", code)
        self.assertEqual(code.count("tcgen05_tmem_allocator.free("), 1)
        self.assertTrue(
            "num_allocated_columns=tcgen05_acc_tmem_cols, dealloc_mbarrier_initialized=True)"
            in code
            or "num_allocated_columns=tcgen05_acc_tmem_cols, initialize_mbarrier=False)"
            in code,
            f"expected dealloc-mbarrier skip kwarg in code: {code!r}",
        )
        # tcgen05 epilogue teardown: the exec warp signals
        # tcgen05_tmem_alloc_barrier.arrive() so allocator teardown can wait on
        # TMEM consumers, and epi warps arrive_and_wait before freeing. The
        # previous staged-via-smem_c epilogue used a different sequence; the
        # assertion below locks in the current teardown flow.
        self.assertIn("tcgen05_tmem_alloc_barrier.arrive()", code)
        self.assertIn("tcgen05_tmem_alloc_barrier.arrive_and_wait()", code)

    def test_tcgen05_codegen_supports_serialized_root_n_threads(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(128, 16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, 8, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(
            block_sizes=[128, 8, 16],
            num_threads=[128, 2, 0],
            loop_orders=[[0, 1]],
        )

        with (
            patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch_cute_mma_support(),
        ):
            code = cute_matmul_mma_codegen_only.bind(args).to_triton_code(config)

        self.assertIn(
            "mma_active = cutlass.Int32(cute.arch.thread_idx()[1]) < cutlass.Int32(2)",
            code,
        )
        # The `tcgen05_epi_warp_count = cutlass.Int32(4)` binding is gone:
        # the count is now inlined wherever it is used (as a `cutlass.Int32(4)`
        # literal), which keeps the 4-epi-warp shape pinned without paying for
        # an extra named compile-time constant.
        self.assertNotIn("tcgen05_epi_warp_count = ", code)
        self.assertIn(
            "tcgen05_tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=160)",
            code,
        )
        self.assertIn("tcgen05_epi_tidx", code)
        self.assertIn(
            "cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition",
            code,
        )
        self.assertIn("block=(128, 2, 1)", code)

    def test_tcgen05_wrapper_plan_tracks_ab_stage_count(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(
            block_sizes=[128, 32, 16],
            l2_groupings=[4],
            loop_orders=[[0, 1]],
            num_stages=1,
            num_warps=4,
            pid_type="flat",
        )

        with patch_cute_mma_support():
            code = cute_matmul_mma_codegen_only.bind(args).to_triton_code(config)

        self.assertIn("'kind': 'tcgen05_ab_tma'", code)
        self.assertIn("'ab_stage_count': 2", code)

    def test_tcgen05_ab_tma_wrapper_plan_respects_stage_count(self) -> None:
        body: list[str] = []
        call_args: list[str] = []
        _append_cute_wrapper_plan(
            body,
            call_args,
            {
                "kind": "tcgen05_ab_tma",
                "lhs_idx": 0,
                "rhs_idx": 1,
                "bm": 128,
                "bn": 32,
                "bk": 16,
                "ab_stage_count": 1,
                "input_dtype": "cutlass.Float16",
                "acc_dtype": "cutlass.Float32",
                "kernel_args": [
                    "tma_atom_a",
                    "tma_tensor_a",
                    "tma_atom_b",
                    "tma_tensor_b",
                ],
            },
        )
        emitted = "\n".join(body)
        self.assertIn("make_smem_layout_a(", emitted)
        self.assertIn("make_smem_layout_b(", emitted)
        self.assertIn("cutlass.Float16, 1)", emitted)
        self.assertEqual(
            call_args, ["tma_atom_a", "tma_tensor_a", "tma_atom_b", "tma_tensor_b"]
        )

    def test_tcgen05_wide_codegen_uses_dense_physical_participant_ids(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(256, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(64, 128, device=DEVICE, dtype=torch.float16),
        )
        # cluster_m=2 and num_epi_warps=4 were the implicit autotune
        # defaults before the search space was tightened; this test still
        # exercises the cluster-aware multi-warp epi codegen path so
        # request both explicitly.
        config = helion.Config(
            block_sizes=[128, 128, 16],
            loop_orders=[[0, 1]],
            num_stages=1,
            pid_type="persistent_blocked",
            tcgen05_cluster_m=2,
            tcgen05_num_epi_warps=4,
        )

        with (
            patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch_cute_mma_support(),
        ):
            bound = cute_matmul_mma_codegen_only.bind(args)
            # This CtaGroup.ONE cluster_m=2 fallback remains host-guarded
            # before launch, but the structural codegen path is still useful
            # to inspect.
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.env.config_spec.restrict_tcgen05_cluster_m_search((2, 1))
            code = bound.to_triton_code(config)

        # 4 epi + 1 exec + 1 ab_load = 6 warps; no idle scheduler/epi_load.
        self.assertIn("block=(32, 6, 1)", code)
        self.assertIn(
            "mma_tidx = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)",
            code,
        )
        self.assertIn(
            "mma_slice_tidx = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) % cutlass.Int32(2)",
            code,
        )
        self.assertIn("thr_mma = tiled_mma.get_slice(mma_slice_tidx)", code)
        self.assertNotIn("for lane_0 in cutlass.range_constexpr(", code)
        self.assertNotIn("for lane_1 in cutlass.range_constexpr(", code)
        self.assertNotIn(
            "mma_tidx = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(128)",
            code,
        )
        self.assertNotIn("load = x[indices_0, indices_2]", code)
        self.assertNotIn("load_1 = y[indices_2, indices_1]", code)
        self.assertNotIn("tcgen05_store_tmem_load_atom", code)
        # The wide tcgen05 epilogue is emitted by
        # `_codegen_cute_store_tcgen05_tile` instead of staging through an
        # intermediate generic `smem_c` allocation, so the float32 tile-wide
        # SMEM buffer is no longer materialized.
        self.assertNotIn("smem_c = cute.arch.alloc_smem", code)
        self.assertTrue(
            "num_allocated_columns=tcgen05_acc_tmem_cols, dealloc_mbarrier_initialized=True)"
            in code
            or "num_allocated_columns=tcgen05_acc_tmem_cols, initialize_mbarrier=False)"
            in code,
            f"expected dealloc-mbarrier skip kwarg in code: {code!r}",
        )

    def test_permute_codegen_materializes_non_store_use(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        permute = graph.call_function(
            torch.ops.aten.permute.default, args=(inp, [1, 0])
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(permute, permute))
        inp.meta["val"] = torch.empty(4, 8)
        permute.meta["val"] = torch.empty(8, 4)

        cg = _FakeGenerateAST({0, 1})
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_permute(ctx, permute)

        self.assertNotEqual(ast.unparse(result), "load")
        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("permute_smem", emitted)
        self.assertIn("cute.arch.sync_threads()", emitted)

    def test_reshape_codegen_materializes_nontrivial_view(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(inp, [2, 6]),
        )
        inp.meta["val"] = torch.empty(4, 3)
        reshape.meta["val"] = torch.empty(2, 6)

        cg = _FakeGenerateAST({0, 1, 2, 3})
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 3: 1, 2: 2, 6: 3})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_reshape(ctx, reshape)

        self.assertEqual(ast.unparse(result), "load")
        self.assertEqual(cg.statements, [])

    def test_reshape_codegen_includes_grid_lane_offsets(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(inp, [2, 8]),
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(reshape, reshape))
        inp.meta["val"] = torch.empty(4, 4)
        reshape.meta["val"] = torch.empty(2, 8)

        grid_strategy = SimpleNamespace(
            _lane_var_by_block={0: "lane_0", 1: "lane_1"},
            _elements_per_thread_for_block=lambda block_id: 2,
        )
        grid_state = SimpleNamespace(
            block_thread_axes={0: 0, 1: 1, 2: 0, 3: 1},
            has_lane_loops=lambda: True,
            strategy=grid_strategy,
        )
        cg = _FakeGenerateAST({2, 3}, current_grid_state=grid_state)
        ctx = SimpleNamespace(
            cg=cg,
            env={inp: ast.Name(id="load", ctx=ast.Load())},
        )
        env = _fake_env({4: 0, 2: 2, 8: 3})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            coord = _get_dim_local_coord(cg, inp.meta["val"], 0)
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIn("thread_idx()[0]) * cutlass.Int32(2)", coord)
        self.assertIn("cutlass.Int32(lane_0)", coord)
        self.assertEqual(ast.unparse(result), "load")
        self.assertEqual(cg.statements, [])

    def test_codegen_mm_cute_resolves_constant_k_block_ids(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 8)
        rhs.meta["val"] = torch.empty(8, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[7]),
                active_device_loops={},
            ),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            get_block_id=lambda size: None,
            resolve_block_id=lambda size: 7 if int(size) == 8 else None,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 7)

    def test_codegen_mm_cute_does_not_pack_distinct_rhs_without_packed_lhs(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 16)
        lo.meta["val"] = torch.empty(8, 4)
        hi.meta["val"] = torch.empty(8, 4)
        stack.meta["val"] = torch.empty(8, 2, 4)
        rhs.meta["val"] = torch.empty(16, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=16,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertIsNone(emit.call_args.kwargs["k_block_id"])
        self.assertIsInstance(emit.call_args.args[2], ast.AST)

    def test_codegen_mm_cute_packs_distinct_rhs_for_packed_lhs(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        graph.output(mm)
        lhs.meta["val"] = torch.empty(4, 16)
        lo.meta["val"] = torch.empty(8, 4)
        hi.meta["val"] = torch.empty(8, 4)
        stack.meta["val"] = torch.empty(8, 2, 4)
        rhs.meta["val"] = torch.empty(16, 4)
        mm.meta["val"] = torch.empty(4, 4)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[11]),
                active_device_loops={},
            ),
            env={
                lhs: CutePackedAffineLoad(
                    (
                        ast.Name(id="lhs_lo", ctx=ast.Load()),
                        ast.Name(id="lhs_hi", ctx=ast.Load()),
                    )
                ),
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 11)
        self.assertIsInstance(emit.call_args.args[2], CutePackedTerms)

    def test_codegen_mm_cute_preserves_default_out_dtype_under_outer_add(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, mm))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        mm.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float32)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = codegen_mm_cute(ctx, mm)

        self.assertEqual(ast.unparse(result), "mm_result")
        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_mm_cute_does_not_force_integer_outer_add_dtype(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        mm = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, mm))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.int32)
        mm.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        ctx = SimpleNamespace(
            cg=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lhs: ast.Name(id="lhs_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.aten_lowering.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion._compiler.aten_lowering._emit_cute_matmul",
                return_value=ast.Name(id="mm_result", ctx=ast.Load()),
            ) as emit,
        ):
            codegen_mm_cute(ctx, mm)

        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float16)

    def test_codegen_cute_dot_preserves_default_out_dtype_under_outer_add(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, dot_node))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float32)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 8, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(8, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={},
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_cute_dot_does_not_force_integer_outer_add_dtype(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        acc = graph.placeholder("acc")
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(acc, dot_node))
        graph.output(add)
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        acc.meta["val"] = torch.empty(4, 4, dtype=torch.int32)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)
        add.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 8, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(8, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={},
        )
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=8,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            hl.dot._codegen["cute"](state)

        self.assertEqual(emit.call_args.kwargs["out_dtype"], torch.float32)

    def test_codegen_cute_dot_does_not_pack_distinct_rhs_without_packed_lhs(
        self,
    ) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        graph.output(dot_node)
        lhs.meta["val"] = torch.empty(4, 16, dtype=torch.float16)
        lo.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        hi.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        stack.meta["val"] = torch.empty(8, 2, 4, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 16, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(16, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_args=[
                ast.Name(id="lhs_tile", ctx=ast.Load()),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            ast_arg=lambda idx: (
                ast.Name(id="lhs_tile", ctx=ast.Load())
                if idx == 0
                else ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            fx_node=dot_node,
            codegen=SimpleNamespace(current_grid_state=None, active_device_loops={}),
            env={
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops.cute_static_k_invariant_extent",
                return_value=16,
            ),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertIsNone(emit.call_args.kwargs["k_block_id"])
        self.assertIsInstance(emit.call_args.args[2], ast.AST)

    def test_codegen_cute_dot_packs_distinct_rhs_for_packed_lhs(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 4]))
        dot_node = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        graph.output(dot_node)
        lhs.meta["val"] = torch.empty(4, 16, dtype=torch.float16)
        lo.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        hi.meta["val"] = torch.empty(8, 4, dtype=torch.float16)
        stack.meta["val"] = torch.empty(8, 2, 4, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 4, dtype=torch.float16)
        dot_node.meta["val"] = torch.empty(4, 4, dtype=torch.float16)

        mode = FakeTensorMode()
        fake_lhs = mode.from_tensor(torch.empty(4, 16, dtype=torch.float16))
        fake_rhs = mode.from_tensor(torch.empty(16, 4, dtype=torch.float16))
        state = SimpleNamespace(
            proxy_args=[fake_lhs, fake_rhs, None, None],
            ast_arg=lambda idx: (
                ast.Name(id="rhs_tile", ctx=ast.Load())
                if idx == 1
                else ast.Constant(value=None)
            ),
            ast_args=[
                CutePackedAffineLoad(
                    (
                        ast.Name(id="lhs_lo", ctx=ast.Load()),
                        ast.Name(id="lhs_hi", ctx=ast.Load()),
                    )
                ),
                ast.Name(id="rhs_tile", ctx=ast.Load()),
                ast.Constant(value=None),
                None,
            ],
            fx_node=dot_node,
            codegen=SimpleNamespace(
                current_grid_state=SimpleNamespace(block_ids=[11]),
                active_device_loops={},
            ),
            env={
                lo: ast.Name(id="lo_tile", ctx=ast.Load()),
                hi: ast.Name(id="hi_tile", ctx=ast.Load()),
                rhs: ast.Name(id="rhs_tile", ctx=ast.Load()),
            },
        )
        env = SimpleNamespace(
            resolve_block_id=lambda size: 11 if int(size) == 8 else None
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion.language.matmul_ops._cute_mma_matches_dot_semantics",
                return_value=False,
            ),
            patch(
                "helion.language.matmul_ops._emit_cute_matmul",
                return_value=ast.Name(id="dot_result", ctx=ast.Load()),
            ) as emit,
        ):
            result = hl.dot._codegen["cute"](state)

        self.assertEqual(ast.unparse(result), "dot_result")
        self.assertEqual(emit.call_args.kwargs["k_block_id"], 11)
        self.assertIsInstance(emit.call_args.args[2], CutePackedTerms)

    def test_match_cute_affine_range_iota_and_duplicate_stack_rhs(self) -> None:
        graph = Graph()
        block_size = graph.placeholder("block_size")
        begin = graph.call_function(hl.tile_begin, args=(block_size,))
        start = graph.call_function(operator.mul, args=(begin, 2))
        length = graph.call_function(operator.mul, args=(block_size, 2))
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": start,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default, args=([packed, packed], 1)
        )
        rhs = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [length, 8]),
        )
        graph.output((iota, rhs))

        affine = match_cute_affine_range_iota(iota)
        self.assertIsNotNone(affine)
        assert affine is not None
        self.assertEqual(affine.base, block_size)
        self.assertEqual(affine.factor, 2)

        duplicate_rhs = match_cute_duplicate_stack_reshape_rhs(rhs)
        self.assertEqual(duplicate_rhs, (packed, 2))

    def test_match_cute_stack_reshape_rhs_allows_distinct_stack_inputs(self) -> None:
        graph = Graph()
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 8]))
        graph.output(rhs)
        lo.meta["val"] = torch.empty(8, 8)
        hi.meta["val"] = torch.empty(8, 8)
        stack.meta["val"] = torch.empty(8, 2, 8)
        rhs.meta["val"] = torch.empty(16, 8)

        matched = match_cute_stack_reshape_rhs(rhs)
        assert matched is not None
        tensors, factor = matched
        self.assertEqual(tensors, (lo, hi))
        self.assertEqual(factor, 2)

    def test_match_cute_duplicate_stack_reshape_rhs_rejects_distinct_stack_inputs(
        self,
    ) -> None:
        graph = Graph()
        lo = graph.placeholder("lo")
        hi = graph.placeholder("hi")
        stack = graph.call_function(torch.ops.aten.stack.default, args=([lo, hi], 1))
        rhs = graph.call_function(torch.ops.aten.view.default, args=(stack, [16, 8]))
        graph.output(rhs)
        lo.meta["val"] = torch.empty(8, 8)
        hi.meta["val"] = torch.empty(8, 8)
        stack.meta["val"] = torch.empty(8, 2, 8)
        rhs.meta["val"] = torch.empty(16, 8)

        self.assertIsNone(match_cute_duplicate_stack_reshape_rhs(rhs))

    def test_codegen_stack_cute_returns_shape_view_and_reshape_resolves_it(
        self,
    ) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(stack, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))
        self.assertEqual(cg.statements, [])

    def test_codegen_stack_cute_rejects_direct_consumer(self) -> None:
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        c = graph.placeholder("c")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([a, b, c], 1),
        )
        graph.call_function(torch.ops.aten.sum.dim_IntList, args=(stack, [0], False))
        graph.output(stack)
        a.meta["val"] = torch.empty(4, 4)
        b.meta["val"] = torch.empty(4, 4)
        c.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 3, 4)

        cg = _FakeGenerateAST({0, 2})
        ctx = SimpleNamespace(
            cg=cg,
            env={
                a: ast.Name(id="a_tile", ctx=ast.Load()),
                b: ast.Name(id="b_tile", ctx=ast.Load()),
                c: ast.Name(id="c_tile", ctx=ast.Load()),
            },
        )
        env = _fake_env({4: 0})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            self.assertRaisesRegex(
                exc.BackendUnsupported,
                "virtual shape-chain direct consumers are not yet supported",
            ),
        ):
            codegen_stack_cute(ctx, stack)

    def test_codegen_stack_cute_rejects_transpose_user(self) -> None:
        graph = Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([a, b], 0),
        )
        graph.call_function(torch.ops.aten.transpose.int, args=(stack, 0, 1))
        graph.output(stack)
        a.meta["val"] = torch.empty(4, 4)
        b.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(2, 4, 4)

        cg = _FakeGenerateAST({1, 2})
        ctx = SimpleNamespace(
            cg=cg,
            env={
                a: ast.Name(id="a_tile", ctx=ast.Load()),
                b: ast.Name(id="b_tile", ctx=ast.Load()),
            },
        )
        env = _fake_env({4: 0})

        self.assertFalse(is_cute_shape_chain_target(torch.ops.aten.transpose.int))
        self.assertFalse(is_cute_shape_chain_target(torch.ops.aten.t.default))
        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            self.assertRaisesRegex(
                exc.BackendUnsupported,
                "virtual shape-chain direct consumers are not yet supported",
            ),
        ):
            codegen_stack_cute(ctx, stack)

    def test_codegen_shape_chain_stays_virtual_through_unsqueeze_until_reshape(
        self,
    ) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        unsqueeze = graph.call_function(
            torch.ops.aten.unsqueeze.default,
            args=(stack, 1),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(unsqueeze, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        unsqueeze.meta["val"] = torch.empty(4, 1, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            unsqueeze_view = codegen_unsqueeze_cute(ctx, unsqueeze)
            self.assertIsInstance(unsqueeze_view, CuteShapeChainView)
            ctx.env[unsqueeze] = unsqueeze_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))

    def test_codegen_unsqueeze_materializes_virtual_view_before_compute(self) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        view = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [4, 8]),
        )
        unsqueeze = graph.call_function(
            torch.ops.aten.unsqueeze.default,
            args=(view, 0),
        )
        graph.call_function(torch.ops.aten.add.Tensor, args=(unsqueeze, unsqueeze))
        graph.output(unsqueeze)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        view.meta["val"] = torch.empty(4, 8)
        unsqueeze.meta["val"] = torch.empty(1, 4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            view_value = codegen_view_cute(ctx, view)
            self.assertIsInstance(view_value, CuteShapeChainView)
            ctx.env[view] = view_value
            result = codegen_unsqueeze_cute(ctx, unsqueeze)

        self.assertIsInstance(result, ast.AST)
        self.assertIn("packed_tile", ast.unparse(result))

    def test_codegen_shape_chain_squeeze_dim_noop_preserves_values(self) -> None:
        graph = Graph()
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], -1),
        )
        squeeze = graph.call_function(
            torch.ops.aten.squeeze.dim,
            args=(stack, 0),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(squeeze, [4, 8]),
        )
        graph.output(reshape)
        packed.meta["val"] = torch.empty(4, 4)
        stack.meta["val"] = torch.empty(4, 4, 2)
        squeeze.meta["val"] = torch.empty(4, 4, 2)
        reshape.meta["val"] = torch.empty(4, 8)

        cg = _FakeGenerateAST({0, 1})
        packed_ast = ast.Name(id="packed_tile", ctx=ast.Load())
        ctx = SimpleNamespace(cg=cg, env={packed: packed_ast})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            stack_view = codegen_stack_cute(ctx, stack)
            self.assertIsInstance(stack_view, CuteShapeChainView)
            ctx.env[stack] = stack_view
            squeeze_view = codegen_squeeze_cute(ctx, squeeze)
            self.assertIsInstance(squeeze_view, CuteShapeChainView)
            ctx.env[squeeze] = squeeze_view
            result = codegen_cute_reshape(ctx, reshape)

        self.assertIsInstance(result, ast.AST)
        rendered = ast.unparse(result)
        self.assertIn("packed_tile", rendered)
        self.assertNotIn("cutlass.Int32(0), cutlass.Int32(0)", rendered)

    def test_codegen_cute_argreduce_gates_scan_on_reduced_lane_loops(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(inp, 1))
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        grid_strategy = SimpleNamespace(
            _lane_var_by_block={0: "lane_0", 1: "lane_1"},
            _elements_per_thread_for_block=lambda block_id: 2,
        )
        grid_state = SimpleNamespace(
            block_thread_axes={0: 0, 1: 1},
            has_lane_loops=lambda: True,
            lane_loops=[("lane_0", 2), ("lane_1", 2)],
            strategy=grid_strategy,
        )
        cg = _FakeGenerateAST({0, 1}, current_grid_state=grid_state)
        ctx = SimpleNamespace(cg=cg, env={inp: ast.Name(id="inp_tile", ctx=ast.Load())})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_tile_argreduce(
                ctx,
                argmax,
                "argmax",
                dim=1,
                keepdim=False,
            )

        self.assertIsInstance(result, ast.AST)
        emitted = "\n".join(ast.unparse(stmt) for stmt in cg.statements)
        self.assertIn("if lane_1 == 1:", emitted)
        self.assertNotIn("if lane_0 == 1:", emitted)

    def test_loop_contains_matmul_for_root_grid_phase(self) -> None:
        root_graph = Graph()
        x = root_graph.placeholder("x")
        root_loop = root_graph.call_function(_tracing_ops._for_loop, args=(1, x))
        root_graph.output(root_loop)

        nested_graph = Graph()
        acc = nested_graph.placeholder("acc")
        lhs = nested_graph.placeholder("lhs")
        rhs = nested_graph.placeholder("rhs")
        nested_graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        nested_graph.output(acc)

        fn = SimpleNamespace(
            codegen=SimpleNamespace(
                codegen_graphs=[
                    RootGraphInfo(graph_id=0, graph=root_graph, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=nested_graph,
                        node_args=[],
                        block_ids=[2],
                    ),
                ]
            )
        )

        with patch(
            "helion._compiler.host_function.HostFunction.current",
            return_value=SimpleNamespace(
                device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
            ),
        ):
            self.assertTrue(_loop_contains_matmul(fn, [0, 1]))

    def test_loop_may_use_mma_for_root_grid_phase(self) -> None:
        root_graph = Graph()
        x = root_graph.placeholder("x")
        root_loop = root_graph.call_function(_tracing_ops._for_loop, args=(1, x))
        root_graph.output(root_loop)

        nested_graph = Graph()
        acc = nested_graph.placeholder("acc")
        lhs = nested_graph.placeholder("lhs")
        rhs = nested_graph.placeholder("rhs")
        addmm = nested_graph.call_function(
            torch.ops.aten.addmm.default, args=(acc, lhs, rhs)
        )
        nested_graph.output(addmm)
        acc.meta["val"] = torch.empty(64, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(64, 16, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(16, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(64, 8, dtype=torch.float32)

        fn = SimpleNamespace(
            codegen=SimpleNamespace(
                codegen_graphs=[
                    RootGraphInfo(graph_id=0, graph=root_graph, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=nested_graph,
                        node_args=[],
                        block_ids=[2],
                    ),
                ]
            )
        )

        with (
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[0, 1]])
                ),
            ),
            patch(
                "helion._compiler.cute.cute_mma.can_codegen_cute_mma_aten",
                return_value=True,
            ),
        ):
            self.assertTrue(_loop_may_use_mma(fn, [0, 1]))

    def test_collect_cute_half_atomic_output_promotions_declines_mixed_root_uses(
        self,
    ) -> None:
        from helion.language import atomic_add
        from helion.language._tracing_ops import _host_tensor

        atomic_graph = Graph()
        atomic_out = atomic_graph.call_function(_host_tensor, args=("out",))
        atomic_value = atomic_graph.placeholder("atomic_value")
        atomic_graph.call_function(atomic_add, args=(atomic_out, [0], atomic_value))
        atomic_graph.output(atomic_out)

        plain_graph = Graph()
        plain_out = plain_graph.call_function(_host_tensor, args=("out",))
        plain_graph.output(plain_out)

        fake_out = torch.zeros(8, device=DEVICE, dtype=torch.float16)
        fake_value = torch.zeros(8, device=DEVICE, dtype=torch.float32)
        atomic_out.meta["val"] = fake_out
        plain_out.meta["val"] = fake_out
        atomic_value.meta["val"] = fake_value

        fake_host_fn = SimpleNamespace(
            tensor_to_origin={fake_out: NameOrigin("out")},
        )

        with patch.object(HostFunction, "current", return_value=fake_host_fn):
            promotions = collect_cute_half_atomic_output_promotions(
                [
                    RootGraphInfo(graph_id=0, graph=atomic_graph, phase_index=0),
                    RootGraphInfo(graph_id=1, graph=plain_graph, phase_index=1),
                ]
            )

        self.assertEqual(promotions, {})

    def test_collect_cute_half_atomic_output_promotions_declines_mixed_root_loop_uses(
        self,
    ) -> None:
        from helion.language import atomic_add
        from helion.language._tracing_ops import _host_tensor

        root_graph = Graph()
        root_out = root_graph.call_function(_host_tensor, args=("out",))
        root_value = root_graph.placeholder("root_value")
        root_graph.call_function(atomic_add, args=(root_out, [0], root_value))
        root_graph.output(root_out)

        loop_graph = Graph()
        loop_out = loop_graph.call_function(_host_tensor, args=("out",))
        loop_graph.output(loop_out)

        fake_out = torch.zeros(8, device=DEVICE, dtype=torch.float16)
        fake_value = torch.zeros(8, device=DEVICE, dtype=torch.float32)
        root_out.meta["val"] = fake_out
        loop_out.meta["val"] = fake_out
        root_value.meta["val"] = fake_value

        fake_host_fn = SimpleNamespace(
            tensor_to_origin={fake_out: NameOrigin("out")},
        )

        with patch.object(HostFunction, "current", return_value=fake_host_fn):
            promotions = collect_cute_half_atomic_output_promotions(
                [
                    RootGraphInfo(graph_id=0, graph=root_graph, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1,
                        graph=loop_graph,
                        node_args=[],
                        block_ids=[2],
                    ),
                ]
            )

        self.assertEqual(promotions, {})

    def test_lane_loop_iter_uses_python_range_for_cute_lane_loops(self) -> None:
        env = SimpleNamespace(backend=SimpleNamespace(name="cute"))
        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertEqual(ast.unparse(_lane_loop_iter(8)), "range(8)")
            self.assertEqual(ast.unparse(_lane_loop_iter(9)), "range(9)")

    def test_create_loop_strategy_preserves_auto_threads_for_mma_candidate(
        self,
    ) -> None:
        backend = CuteBackend()
        fn = _FakeDeviceFunction()
        fn.codegen = SimpleNamespace(
            codegen_graphs=[
                ForLoopGraphInfo(
                    graph_id=0, graph=Graph(), node_args=[], block_ids=[0, 1]
                )
            ]
        )
        env = SimpleNamespace(
            block_sizes=[
                _FakeBlockSize(128, block_id=0),
                _FakeBlockSize(8, block_id=1),
                _FakeBlockSize(16, block_id=2, reduction=True),
            ],
            config_spec=SimpleNamespace(
                num_threads=SimpleNamespace(config_get=lambda *args: 0),
                loop_orders=SimpleNamespace(config_get=lambda *args: None),
                l2_groupings=SimpleNamespace(config_get=lambda *args: 1),
            ),
        )
        config = SimpleNamespace(loop_orders=None, l2_groupings=None, num_threads=None)

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=SimpleNamespace(
                    device_ir=SimpleNamespace(grid_block_ids=[[2]])
                ),
            ),
            patch("helion._compiler.backend._loop_contains_matmul", return_value=True),
            patch(
                "helion._compiler.backend._detect_mma_loop",
                return_value=True,
            ) as detect_mma_loop,
            patch(
                "helion._compiler.backend._detect_specialized_mma_loop",
                return_value=True,
            ),
        ):
            strategy = backend.create_loop_strategy(fn, [0, 1], config)

        self.assertEqual(strategy.num_threads, [0, 0])
        self.assertTrue(strategy.mma_mode)
        self.assertEqual(
            detect_mma_loop.call_args_list[0].kwargs["num_threads_config"],
            [128, 8],
        )

    def test_should_use_cute_argreduce_lowering_recurses_nested_for_loops(
        self,
    ) -> None:
        outer_graph = Graph()
        inp = outer_graph.placeholder("inp")
        outer_loop = outer_graph.call_function(_tracing_ops._for_loop, args=(1, inp))
        argmax = outer_graph.call_function(
            torch.ops.aten.argmax.default, args=(outer_loop, 1)
        )
        outer_graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 4)
        outer_loop.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)
        argmax.meta["location"] = contextlib.nullcontext()

        mid_graph = Graph()
        mid_inp = mid_graph.placeholder("mid_inp")
        nested_loop = mid_graph.call_function(_tracing_ops._for_loop, args=(2, mid_inp))
        mid_graph.output(nested_loop)

        inner_graph = Graph()
        acc = inner_graph.placeholder("acc")
        lhs = inner_graph.placeholder("lhs")
        rhs = inner_graph.placeholder("rhs")
        addmm = inner_graph.call_function(
            torch.ops.aten.addmm.default, args=(acc, lhs, rhs)
        )
        inner_graph.output(addmm)

        fake_env = SimpleNamespace(backend_name="cute")
        fake_device_ir = SimpleNamespace(
            graphs=[
                SimpleNamespace(graph=outer_graph),
                SimpleNamespace(graph=mid_graph),
                SimpleNamespace(graph=inner_graph),
            ]
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=fake_env),
            patch(
                "helion._compiler.device_ir.DeviceIR.current",
                return_value=fake_device_ir,
            ),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_dim_none(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(mm,))
        graph.output(argmax)
        acc.meta["val"] = torch.empty(4, 4)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        mm.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty((), dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_torch_matmul(self) -> None:
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        matmul = graph.call_function(torch.matmul, args=(lhs, rhs))
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(matmul, 1))
        graph.output(argmax)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        matmul.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_should_use_cute_argreduce_lowering_for_keepdim(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(mm, 1, True),
        )
        graph.output(argmax)
        acc.meta["val"] = torch.empty(4, 4)
        lhs.meta["val"] = torch.empty(4, 4)
        rhs.meta["val"] = torch.empty(4, 4)
        mm.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend_name="cute"),
        ):
            self.assertTrue(_should_use_cute_argreduce_lowering(argmax))

    def test_codegen_cute_argreduce_tracks_validity_separately_from_value(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(torch.ops.aten.argmax.default, args=(inp, 1))
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, dtype=torch.int64)

        cg = _FakeGenerateAST({0, 1})
        ctx = SimpleNamespace(cg=cg, env={inp: ast.Name(id="inp_tile", ctx=ast.Load())})
        env = _fake_env({4: 0, 8: 1})

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_cute_tile_argreduce(
                ctx,
                argmax,
                "argmax",
                dim=1,
                keepdim=False,
            )

        self.assertIsInstance(result, ast.AST)
        emitted = "\n".join(
            stmt if isinstance(stmt, str) else ast.unparse(stmt)
            for stmt in cg.statements
        )
        self.assertIn("argreduce_valid_smem", emitted)
        self.assertIn("_cute_argreduce_index", emitted)

    def test_triton_argreduce_supports_dim_none_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(inp, None, True),
        )
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 4)
        argmax.meta["val"] = torch.empty(1, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=TritonBackend()),
        ):
            result = _triton_argreduce(self._argreduce_ctx(inp), argmax, "argmax")

        emitted = ast.unparse(result)
        self.assertIn("tl.reshape(x, [16])", emitted)
        self.assertIn("axis=0", emitted)
        self.assertIn("tl.full([1, 1]", emitted)

    def test_triton_argreduce_preserves_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmax = graph.call_function(
            torch.ops.aten.argmax.default,
            args=(inp, 1, True),
        )
        graph.output(argmax)
        inp.meta["val"] = torch.empty(4, 8)
        argmax.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=TritonBackend()),
        ):
            result = _triton_argreduce(self._argreduce_ctx(inp), argmax, "argmax")

        emitted = ast.unparse(result)
        self.assertIn("axis=1", emitted)
        self.assertIn("tl.reshape", emitted)
        self.assertIn("[4, 1]", emitted)

    def test_pallas_argreduce_supports_dim_none_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmin = graph.call_function(
            torch.ops.aten.argmin.default,
            args=(inp, None, True),
        )
        graph.output(argmin)
        inp.meta["val"] = torch.empty(4, 4)
        argmin.meta["val"] = torch.empty(1, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=PallasBackend()),
        ):
            result = _pallas_argreduce(self._argreduce_ctx(inp), argmin, "argmin")

        emitted = ast.unparse(result)
        self.assertIn("jnp.reshape(x, [16])", emitted)
        self.assertIn("axis=0", emitted)
        self.assertIn("jnp.full([1, 1]", emitted)

    def test_pallas_argreduce_preserves_keepdim(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        argmin = graph.call_function(
            torch.ops.aten.argmin.default,
            args=(inp, 1, True),
        )
        graph.output(argmin)
        inp.meta["val"] = torch.empty(4, 8)
        argmin.meta["val"] = torch.empty(4, 1, dtype=torch.int64)

        with patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(backend=PallasBackend()),
        ):
            result = _pallas_argreduce(self._argreduce_ctx(inp), argmin, "argmin")

        emitted = ast.unparse(result)
        self.assertIn("axis=1", emitted)
        self.assertIn("jnp.reshape", emitted)
        self.assertIn("[4, 1]", emitted)

    def test_cute_dot_outer_accumulates_result_requires_augassign_source(self) -> None:
        graph = Graph()
        acc_in = graph.placeholder("acc_in")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        copied_acc = graph.call_function(_new_var, args=(acc_in,))
        dot = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(copied_acc, dot))
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 1, in kernel\n    acc += hl.dot(lhs, rhs)\n'
        )
        graph.output(add)
        self.assertTrue(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

        non_acc_graph = Graph()
        tmp_in = non_acc_graph.placeholder("tmp_in")
        lhs = non_acc_graph.placeholder("lhs")
        rhs = non_acc_graph.placeholder("rhs")
        copied_tmp = non_acc_graph.call_function(_new_var, args=(tmp_in,))
        dot = non_acc_graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = non_acc_graph.call_function(
            torch.ops.aten.add.Tensor, args=(dot, copied_tmp)
        )
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 2, in kernel\n'
            "    out = hl.dot(lhs, rhs) + tmp\n"
        )
        non_acc_graph.output(add)
        self.assertFalse(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

    def test_cute_dot_outer_accumulates_result_accepts_simple_assign_pattern(
        self,
    ) -> None:
        graph = Graph()
        acc_in = graph.placeholder("acc_in")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        copied_acc = graph.call_function(_new_var, args=(acc_in,))
        dot = graph.call_function(hl.dot, args=(lhs, rhs, None, None))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(copied_acc, dot))
        add.meta["stack_trace"] = (
            '  File "/tmp/test.py", line 1, in kernel\n'
            "    acc = acc + hl.dot(lhs, rhs)\n"
        )
        graph.output(add)
        self.assertTrue(
            _cute_dot_outer_accumulates_result(
                SimpleNamespace(fx_node=dot),
                is_acc_none=True,
            )
        )

    def test_cute_static_k_invariant_extent_recovers_specialized_full_matmul(self):
        graph = Graph()
        lhs = graph.call_function(
            hl.full,
            args=([32, 512], 1.0 / 512, torch.float32, None),
        )
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertEqual(cute_static_k_invariant_extent(lhs, rhs), 512)

    def test_cute_static_k_invariant_extent_rejects_nonuniform_inputs(self):
        graph = Graph()
        lhs = graph.placeholder("lhs")
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertIsNone(cute_static_k_invariant_extent(lhs, rhs))

    def test_cute_static_k_invariant_extent_rejects_masked_inputs(self):
        graph = Graph()
        lhs_base = graph.call_function(
            hl.full,
            args=([32, 512], 1.0 / 512, torch.float32, None),
        )
        lhs = graph.call_function(_mask_to, args=(lhs_base, 0.0))
        rhs = graph.call_function(
            hl.full,
            args=([512, 512], 1.0, torch.float32, None),
        )
        graph.output((lhs, rhs))
        lhs.meta["val"] = torch.empty(32, 512)
        rhs.meta["val"] = torch.empty(512, 512)

        self.assertIsNone(cute_static_k_invariant_extent(lhs, rhs))

    def test_cute_scalar_matmul_fallback_rejects_thread_carried_n(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(
                block_ids=[0],
                thread_axis_sizes={0: 32, 1: 4},
            )
        )
        lhs = torch.empty(32, 128, dtype=torch.bfloat16)
        rhs = torch.empty(128, 128, dtype=torch.bfloat16)
        out = torch.empty(32, 128, dtype=torch.bfloat16)
        env = SimpleNamespace(resolve_block_id=lambda size: None)

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertFalse(
                cute_supports_scalar_matmul_fallback(
                    cg,
                    lhs,
                    rhs,
                    out,
                    k_block_id=None,
                )
            )

    def test_cute_resolve_active_block_id_ignores_inactive_size_match(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[3]),
            active_device_loops={},
        )
        env = _fake_env({128: 7, 32: 3})

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_block_id(cg, 128))
            self.assertEqual(cute_resolve_active_block_id(cg, 32), 3)

    def test_cute_resolve_active_matmul_k_block_id_rejects_n_alias(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[7]),
            active_device_loops={},
        )
        env = _fake_env({128: 7})

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_matmul_k_block_id(cg, 128, 128, 128))

    def test_cute_resolve_active_block_id_rejects_ambiguous_aliases(self):
        cg = SimpleNamespace(
            current_grid_state=SimpleNamespace(block_ids=[3, 5]),
            active_device_loops={},
        )
        env = _fake_env({64: 7})
        env.canonical_block_id = lambda block_id: {3: 11, 5: 11, 7: 11}.get(
            block_id, block_id
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertIsNone(cute_resolve_active_block_id(cg, 64))

    def test_resolve_block_id_requires_symbolic_origin(self) -> None:
        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: expr,
            get_block_id=lambda size: None,
        )

        self.assertIsNone(CompileEnvironment.resolve_block_id(fake_env, 128))

    def test_resolve_block_id_ignores_specialized_derived_constant(self) -> None:
        u0 = sympy.Symbol("u0")

        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: (
                sympy.Integer(16)
                if sympy.simplify(expr - u0) == 0
                else sympy.Integer(17)
                if sympy.simplify(expr - (u0 + 1)) == 0
                else expr
            ),
            get_block_id=lambda size: 0 if size == u0 else None,
            canonical_block_id=lambda block_id: block_id,
        )

        self.assertEqual(CompileEnvironment.resolve_block_id(fake_env, u0), 0)
        self.assertIsNone(CompileEnvironment.resolve_block_id(fake_env, u0 + 1))

    def test_resolve_block_id_accepts_tile_origin_subclasses(self) -> None:
        sym = sympy.Symbol("tile_begin_0")
        fake_env = SimpleNamespace(
            specialize_expr=lambda expr: expr,
            get_block_id=lambda size: CompileEnvironment.get_block_id(fake_env, size),
        )
        fake_host_fn = SimpleNamespace(
            expr_to_origin={sym: SimpleNamespace(origin=TileBeginOrigin(block_id=3))}
        )

        with patch.object(HostFunction, "current", return_value=fake_host_fn):
            self.assertEqual(CompileEnvironment.resolve_block_id(fake_env, sym), 3)

    def test_persistent_cute_reduction_marks_synthetic_lane_loop_block(self) -> None:
        fake_env = SimpleNamespace(
            backend=SimpleNamespace(name="cute"),
            block_sizes={0: SimpleNamespace(numel=sympy.Integer(1024))},
            index_type=lambda: "cutlass.Int32",
        )
        current_grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        state = SimpleNamespace(
            codegen=SimpleNamespace(
                current_grid_state=current_grid,
                host_statements=[],
                push_active_loops=lambda loop_state: None,
            ),
            device_function=SimpleNamespace(
                constexpr_arg=lambda name: False,
                deferred_rdim_defs=[],
            ),
            add_statement=lambda stmt: None,
        )
        fake_strategy = SimpleNamespace(
            block_index=0,
            _mask_var=None,
            _thread_count=256,
            _synthetic_cute_lane_var="synthetic_lane_0",
            _synthetic_cute_lane_extent=4,
            block_size_var=lambda block_idx: "_RDIM_SIZE_0",
            index_var=lambda block_idx: "indices_0",
            _get_thread_axis=lambda: 0,
            _index_init_expr=lambda block_size_var, dtype, block_idx: "base_idx",
            fn=SimpleNamespace(sympy_expr=lambda expr: str(expr)),
        )

        with patch.object(CompileEnvironment, "current", return_value=fake_env):
            PersistentReductionStrategy.codegen_preamble(fake_strategy, state)

        self.assertIn(0, current_grid.lane_loop_blocks)
        self.assertIn(("synthetic_lane_0", 4), current_grid.lane_loops)

    def test_baddbmm_packed_affine_lhs_load_fuses_on_cute(self) -> None:
        graph = Graph()
        block_size = graph.placeholder("block_size")
        batch_idx = graph.placeholder("batch_idx")
        row_idx = graph.placeholder("row_idx")
        begin = graph.call_function(hl.tile_begin, args=(block_size,))
        start = graph.call_function(operator.mul, args=(begin, 2))
        length = graph.call_function(operator.mul, args=(block_size, 2))
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": start,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        lhs = graph.call_function(
            hl.load,
            args=(graph.placeholder("A"), [batch_idx, row_idx, iota], None, None),
        )
        packed = graph.placeholder("packed")
        stack = graph.call_function(
            torch.ops.aten.stack.default,
            args=([packed, packed], 2),
        )
        rhs = graph.call_function(
            torch.ops.aten.view.default,
            args=(stack, [2, length, 8]),
        )
        acc = graph.placeholder("acc")
        graph.call_function(torch.ops.aten.baddbmm.default, args=(acc, lhs, rhs))
        graph.output(rhs)

        packed.meta["val"] = torch.empty(2, 4, 8)
        rhs.meta["val"] = torch.empty(2, 8, 8)

        loop_strategy = _FakeMaskedLoopStrategy([0])
        codegen = SimpleNamespace(
            active_device_loops={0: [SimpleNamespace(strategy=loop_strategy)]},
            current_grid_state=None,
            codegen_graphs=[
                ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[0])
            ],
        )
        state = SimpleNamespace(
            fx_node=lhs,
            codegen=codegen,
            device_function=SimpleNamespace(
                tensor_arg=lambda tensor: SimpleNamespace(name="A")
            ),
        )

        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32"),
            block_sizes={0: _FakeBlockSize(4)},
            get_block_id=lambda size: None,
            known_equal=operator.eq,
            resolve_block_id=lambda size: None,
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _maybe_codegen_cute_packed_affine_lhs_load(
                state,
                torch.empty(2, 8, 8),
                [0, 0, torch.empty(8, dtype=torch.int32)],
                None,
            )
        self.assertIsInstance(result, CutePackedAffineLoad)

    def test_codegen_iota_cute_recovers_active_axis_from_matching_block_size(self):
        graph = Graph()
        block_size = graph.placeholder("block_size")
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(block_size,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        graph.output(iota)
        iota.meta["val"] = torch.empty(64, dtype=torch.int32)

        cg = _FakeGenerateAST({1})
        strategy = SimpleNamespace(index_var=lambda block_id: f"idx_{block_id}")
        cg.codegen_graphs = [
            ForLoopGraphInfo(graph_id=0, graph=graph, node_args=[], block_ids=[1])
        ]
        cg.active_device_loops = {1: [SimpleNamespace(strategy=strategy)]}
        cg.current_grid_state = None
        ctx = SimpleNamespace(
            cg=cg,
            to_ast=lambda value: ast.Constant(value=value),
        )
        env = SimpleNamespace(
            index_dtype=torch.int32,
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int32"),
            block_sizes={
                0: _FakeBlockSize(64),
                1: _FakeBlockSize(64),
            },
            get_block_id=lambda size: 1 if int(size) == 64 else None,
            resolve_block_id=lambda size: 0 if size is block_size else None,
            resolve_codegen_block_id=lambda block_id, cg, graph=None: (
                1 if block_id == 0 else block_id
            ),
            size_hint=lambda size: 64 if size is block_size else 1,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
        ):
            result = codegen_iota_cute(ctx, iota)

        self.assertEqual(ast.unparse(result), "indices_1 - offset_1")

    def test_codegen_iota_cute_shared_atomic_user_does_not_collapse_to_zero(self):
        graph = Graph()
        length = graph.placeholder("length")
        out = graph.placeholder("out")
        val = graph.placeholder("val")
        iota = graph.call_function(
            torch.ops.prims.iota.default,
            args=(length,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": torch.int32,
                "device": torch.device("cuda"),
                "requires_grad": False,
            },
        )
        graph.call_function(hl.atomic_add, args=(out, [iota], val))
        ordinary_use = graph.call_function(operator.add, args=(iota, 1))
        graph.output(ordinary_use)
        iota.meta["val"] = torch.empty(64, dtype=torch.int32)

        cg = _FakeGenerateAST(set())
        cg.active_device_loops = {}
        cg.current_grid_state = None
        cg.codegen_graphs = []
        ctx = SimpleNamespace(
            cg=cg,
            to_ast=lambda value: ast.Constant(value=value),
        )
        env = SimpleNamespace(
            index_dtype=torch.int32,
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int32"),
            block_sizes={},
            get_block_id=lambda size: None,
            resolve_block_id=lambda size: None,
            resolve_codegen_block_id=lambda block_id, cg, graph=None: block_id,
            size_hint=lambda size: 64 if size is length else 1,
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch("helion._compiler.generate_ast.GenerateAST", _FakeGenerateAST),
            patch(
                "helion._compiler.cute.cute_reshape._get_dim_local_coord",
                return_value="cutlass.Int32(0)",
            ),
            self.assertRaises(exc.BackendUnsupported),
        ):
            codegen_iota_cute(ctx, iota)

    def test_can_codegen_cute_mma_aten_requires_exclusive_loop_body(self) -> None:
        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.output(addmm)
        acc.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(16, 64, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(64, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        with patch(
            "helion._compiler.cute.cute_mma.is_mma_compatible_aten",
            return_value=True,
        ):
            self.assertTrue(can_codegen_cute_mma_aten(addmm, with_acc=True))

        graph = Graph()
        acc = graph.placeholder("acc")
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        addmm = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
        graph.call_function(torch.ops.aten.neg.default, args=(lhs,))
        graph.output(addmm)
        acc.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        lhs.meta["val"] = torch.empty(16, 64, dtype=torch.float16)
        rhs.meta["val"] = torch.empty(64, 8, dtype=torch.float16)
        addmm.meta["val"] = torch.empty(16, 8, dtype=torch.float32)
        with patch(
            "helion._compiler.cute.cute_mma.is_mma_compatible_aten",
            return_value=True,
        ):
            self.assertFalse(can_codegen_cute_mma_aten(addmm, with_acc=True))

    def test_lane_loop_store_permute_codegen_stays_inline(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        permute = graph.call_function(
            torch.ops.aten.permute.default,
            args=(inp, [1, 0]),
        )
        inp.meta["val"] = torch.empty(2, 2)
        permute.meta["val"] = torch.empty(2, 2)

        grid_state = DeviceGridState(
            strategy=SimpleNamespace(block_ids=[0, 1]),
            block_id_to_info={},
            lane_loops=[("lane_0", 2)],
            lane_setup_statements=[],
        )
        codegen = _FakeGenerateASTForLaneStore(grid_state)
        state = SimpleNamespace(
            codegen=codegen,
            device_function=codegen.device_function,
        )
        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32"),
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.generate_ast.GenerateAST",
                _FakeGenerateASTForLaneStore,
            ),
            patch(
                "helion.language.memory_ops._cute_index_exprs",
                return_value=["i0", "i1"],
            ),
            patch("helion.language.memory_ops._cute_combined_mask", return_value=None),
            patch(
                "helion._compiler.cute.cute_reshape._store_permute_info",
                return_value=(inp, [1, 0]),
            ),
            patch(
                "helion._compiler.cute.cute_reshape._permute_reorders_active_dims",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._shape_op_needs_materialization",
                return_value=False,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_tile_shape",
                return_value=[2, 2],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_dim_local_coord",
                return_value="0",
            ),
            patch(
                "helion._compiler.cute.cute_reshape._flat_index_from_coords",
                side_effect=["0", "1"],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._coords_from_flat_index",
                return_value=["0", "1"],
            ),
        ):
            result = _codegen_cute_store_permute_lane_loops(
                state,
                torch.empty(2, 2),
                [slice(None), slice(None)],
                [slice(None), slice(None)],
                ast.Name(id="value", ctx=ast.Load()),
                None,
                permute,
            )

        assert result is not None
        code = ast.unparse(result)
        self.assertIn("cute.arch.sync_threads()", code)
        self.assertIn("permute_smem", code)
        self.assertIn("out.__setitem__((i0, i1)", code)
        self.assertEqual(grid_state.outer_suffix, [])

    def test_mask_to_cute_casts_then_branch_to_tensor_dtype(self) -> None:
        state = SimpleNamespace(
            proxy_arg=lambda index: (
                torch.empty(16, 8, dtype=torch.float16) if index == 0 else 0
            ),
            ast_arg=lambda index: (
                expr_from_string("load + 1") if index == 0 else expr_from_string("0")
            ),
            codegen=SimpleNamespace(mask_var=lambda block_id: f"mask_{block_id}"),
            tile_strategy=SimpleNamespace(expand_str=lambda sizes, dim: ""),
        )
        env = SimpleNamespace(
            backend=CuteBackend(),
            resolve_block_id=lambda size: {16: 0, 8: 1}.get(int(size)),
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _mask_to._codegen["cute"](state)

        self.assertEqual(
            ast.unparse(result),
            "cutlass.Float16(load + 1) if mask_0 and mask_1 else cutlass.Float16(0)",
        )

    def test_lane_loop_store_permute_masked_load_uses_materialization(self) -> None:
        graph = Graph()
        inp = graph.placeholder("inp")
        mask = graph.placeholder("mask")
        load_node = graph.call_function(
            load,
            args=(inp, [slice(None), slice(None)], mask, ""),
        )
        permute = graph.call_function(
            torch.ops.aten.permute.default,
            args=(load_node, [1, 0]),
        )
        inp.meta["val"] = torch.empty(2, 2)
        mask.meta["val"] = torch.empty(2, 2, dtype=torch.bool)
        load_node.meta["val"] = torch.empty(2, 2)
        permute.meta["val"] = torch.empty(2, 2)

        grid_state = DeviceGridState(
            strategy=SimpleNamespace(block_ids=[0, 1]),
            block_id_to_info={},
            lane_loops=[("lane_0", 2)],
            lane_setup_statements=[],
        )
        codegen = _FakeGenerateASTForLaneStore(grid_state)
        state = SimpleNamespace(
            codegen=codegen,
            device_function=codegen.device_function,
        )
        env = SimpleNamespace(
            backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32"),
        )

        with (
            patch.object(CompileEnvironment, "current", return_value=env),
            patch(
                "helion._compiler.generate_ast.GenerateAST",
                _FakeGenerateASTForLaneStore,
            ),
            patch(
                "helion.language.memory_ops._cute_index_exprs",
                return_value=["i0", "i1"],
            ),
            patch("helion.language.memory_ops._cute_combined_mask", return_value=None),
            patch(
                "helion._compiler.cute.cute_reshape._store_permute_info",
                return_value=(load_node, [1, 0]),
            ),
            patch(
                "helion._compiler.cute.cute_reshape._permute_reorders_active_dims",
                return_value=True,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._shape_op_needs_materialization",
                return_value=False,
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_tile_shape",
                return_value=[2, 2],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._get_dim_local_coord",
                return_value="0",
            ),
            patch(
                "helion._compiler.cute.cute_reshape._flat_index_from_coords",
                side_effect=["0", "1"],
            ),
            patch(
                "helion._compiler.cute.cute_reshape._coords_from_flat_index",
                return_value=["0", "1"],
            ),
        ):
            result = _codegen_cute_store_permute_lane_loops(
                state,
                torch.empty(2, 2),
                [slice(None), slice(None)],
                [slice(None), slice(None)],
                ast.Name(id="value", ctx=ast.Load()),
                None,
                permute,
            )

        assert result is not None
        code = ast.unparse(result)
        self.assertIn("cute.arch.sync_threads()", code)
        self.assertIn("permute_smem", code)

    def test_choose_mma_impl_forced_incompatible_override_falls_back(self) -> None:
        with patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support",
            return_value=default_cute_mma_support(),
        ):
            with patch.dict(
                "os.environ", {"HELION_CUTE_MMA_IMPL": "warp"}, clear=False
            ):
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=64, bn=16, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(torch.float32, bm=16, bn=8, bk=16),
                    "universal",
                )
            with patch.dict(
                "os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False
            ):
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=16, bn=8, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=32, bn=16, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(torch.float16, bm=64, bn=512, bk=16),
                    "universal",
                )
                self.assertEqual(
                    _choose_mma_impl(
                        torch.float16,
                        bm=64,
                        bn=512,
                        bk=16,
                        config=helion.Config(
                            block_sizes=[64, 512, 16],
                            tcgen05_cluster_m=1,
                            **{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True},
                        ),
                    ),
                    "tcgen05",
                )
                self.assertEqual(
                    _choose_mma_impl(
                        torch.float16,
                        bm=128,
                        bn=512,
                        bk=16,
                        config=helion.Config(
                            block_sizes=[128, 512, 16],
                            tcgen05_cluster_m=1,
                            **{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY: True},
                        ),
                    ),
                    "universal",
                )

    def test_tcgen05_thread_counts_match_participants_and_cta(self) -> None:
        # ``_tcgen05_epi_warp_count`` takes a ``Tcgen05WarpSpec`` (G2-B);
        # build one from the documented monolithic defaults and only
        # override ``epi_warps`` to exercise the cap behavior.
        def _spec(epi_warps: int) -> Tcgen05WarpSpec:
            return dataclasses.replace(
                ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC, epi_warps=epi_warps
            )

        self.assertEqual(_tcgen05_ab_stage_count(0), 1)
        self.assertEqual(_tcgen05_ab_stage_count(1), 1)
        self.assertEqual(_tcgen05_ab_stage_count(2), 2)
        self.assertEqual(_tcgen05_ab_stage_count(4), 2)
        self.assertEqual(_tcgen05_epi_warp_count(_spec(4), cta_thread_count=32), 1)
        self.assertEqual(_tcgen05_epi_warp_count(_spec(4), cta_thread_count=128), 4)
        self.assertEqual(_tcgen05_epi_warp_count(_spec(2), cta_thread_count=256), 2)
        self.assertEqual(_tcgen05_epi_warp_count(_spec(8), cta_thread_count=128), 4)
        self.assertEqual(_tcgen05_root_m_threads(64, 8), 64)
        self.assertEqual(_tcgen05_root_m_threads(64, 16), 32)
        self.assertEqual(_tcgen05_root_m_threads(128, 256), 32)
        self.assertEqual(_tcgen05_tmem_barrier_thread_count(1), 64)
        self.assertEqual(_tcgen05_tmem_barrier_thread_count(2), 96)
        self.assertEqual(_tcgen05_tmem_barrier_thread_count(4), 160)

    def test_tcgen05_layout_plan_setup_inlines_constants(self) -> None:
        df = _FakeDeviceFunction()
        plan = _new_tcgen05_layout_plan(df)
        stmts = _make_tcgen05_layout_plan_setup(
            plan,
            "tiled_mma",
            bm=128,
            bn=8,
            bk=16,
            ab_stage_count=1,
            is_two_cta=False,
            input_dtype_str="cutlass.Float16",
            acc_dtype_str="cutlass.Float32",
        )

        emitted = "\n".join(ast.unparse(stmt) for stmt in stmts)
        self.assertEqual(len(stmts), 6)
        # Stage / warp counts are now inlined as cutlass.Int32 literals at the
        # call site; the layout plan no longer materializes named constants for
        # them. Checking for the *absence* of those bindings pins that contract.
        self.assertNotIn("acc_stage_count", emitted)
        self.assertNotIn("ab_stage_count_1 = ", emitted)
        self.assertNotIn("c_stage_count", emitted)
        self.assertNotIn("epi_warp_count_1 = ", emitted)
        self.assertNotIn("epilog_sync_barrier_id", emitted)
        self.assertNotIn("acc_pipeline_arrive_count", emitted)
        self.assertNotIn("ab_pipeline_arrive_count", emitted)
        self.assertNotIn("exec_thread_count", emitted)
        # The dead smem_c / smem_desc_view layouts are also gone.
        self.assertNotIn("smem_c_layout", emitted)
        self.assertNotIn("smem_desc_view_layout", emitted)
        self.assertIn(
            "make_smem_layout_a(tiled_mma, (128, 8, 16), cutlass.Float16, 1)", emitted
        )
        self.assertIn(
            "make_smem_layout_b(tiled_mma, (128, 8, 16), cutlass.Float16, 1)", emitted
        )
        self.assertIn(
            "tcgen05_epilogue_rest_mode_1 = cute.make_layout(1, stride=0)",
            emitted,
        )
        # `compute_epilogue_tile_shape` must receive `elem_ty_d` and
        # `elem_ty_c` matching the D-output element type so the helper
        # takes the with-source branch (e.g. bf16/fp16 → `tile_n=64`)
        # rather than the `disable_source=True` branch (`tile_n=32`).
        # The kernel-side, store-side, and wrapper-side calls must all
        # agree on `tile_n`; a regression that drops the kwargs, swaps
        # them to the accumulator dtype, or mismatches the `elem_ty_d`
        # positional must fail one of these assertions.
        self.assertIn(
            (
                "compute_epilogue_tile_shape((128, 8), False, "
                "tcgen05_c_layout_1, cutlass.Float16, "
                "layout_c=tcgen05_c_layout_1, elem_ty_c=cutlass.Float16)"
            ),
            emitted,
        )
        self.assertNotIn(
            "compute_epilogue_tile_shape((128, 8), False, "
            "tcgen05_c_layout_1, cutlass.Float32",
            emitted,
        )

    def test_tcgen05_layout_plan_setup_threads_mixed_input_output_dtype(
        self,
    ) -> None:
        """`_make_tcgen05_layout_plan_setup` honors
        ``epi_elem_dtype_str`` when the caller supplies an output dtype
        that differs from ``input_dtype_str``.

        For bf16/fp16-input matmuls that store directly to fp32 (no
        ``acc.to(x.dtype)`` cast), the matmul plan's
        ``compute_epilogue_tile_shape`` must use the *output* dtype (not
        the input dtype) so the kernel-side ``tile_n`` matches the
        store-side ``tile_n`` and the SMEM staging stays consistent.
        Pin the kwargs explicitly: ``elem_ty_d`` / ``elem_ty_c`` are
        ``cutlass.Float32`` (output), ``smem_layout_a`` / ``smem_layout_b``
        keep ``cutlass.Float16`` (input).
        """
        df = _FakeDeviceFunction()
        plan = _new_tcgen05_layout_plan(df)
        stmts = _make_tcgen05_layout_plan_setup(
            plan,
            "tiled_mma",
            bm=128,
            bn=8,
            bk=16,
            ab_stage_count=1,
            is_two_cta=False,
            input_dtype_str="cutlass.Float16",
            acc_dtype_str="cutlass.Float32",
            epi_elem_dtype_str="cutlass.Float32",
        )
        emitted = "\n".join(ast.unparse(stmt) for stmt in stmts)
        # SMEM-A/B layouts still use the input dtype.
        self.assertIn(
            "make_smem_layout_a(tiled_mma, (128, 8, 16), cutlass.Float16, 1)",
            emitted,
        )
        self.assertIn(
            "make_smem_layout_b(tiled_mma, (128, 8, 16), cutlass.Float16, 1)",
            emitted,
        )
        # `compute_epilogue_tile_shape` uses the *output* dtype.
        self.assertIn(
            (
                "compute_epilogue_tile_shape((128, 8), False, "
                "tcgen05_c_layout_1, cutlass.Float32, "
                "layout_c=tcgen05_c_layout_1, elem_ty_c=cutlass.Float32)"
            ),
            emitted,
        )
        # And explicitly NOT the input dtype.
        self.assertNotIn(
            "compute_epilogue_tile_shape((128, 8), False, "
            "tcgen05_c_layout_1, cutlass.Float16",
            emitted,
        )

    def _build_matmul_store_graphs(
        self,
        *,
        store_dtypes: list[torch.dtype],
        cast_dtype: torch.dtype | None = None,
    ) -> tuple[torch.fx.Node, list[GraphInfo]]:
        """Build synthetic FX graphs that mimic the
        ``for tile_m, tile_n: ... for tile_k: acc = hl.dot(...)
        out[...] = acc[.to(...)]`` shape.

        Returns the matmul fx_node (in the K-loop body) and a
        ``codegen_graphs`` list pairing one ``ForLoopGraphInfo`` for the
        K-loop body with one ``RootGraphInfo`` for the outer store
        body. ``store_dtypes`` controls how many store ops the outer
        graph emits and what tensor dtype each store targets.
        ``cast_dtype`` optionally inserts a ``convert_element_type``
        between the for-loop output and the store's value (mirroring
        ``acc.to(x.dtype)``).
        """
        from helion.language import memory_ops

        body_graph = Graph()
        body_acc_in = body_graph.placeholder("acc_in")
        body_acc_in.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        mma_node = body_graph.call_function(hl.dot, args=(body_acc_in,))
        mma_node.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        body_graph.output((mma_node,))

        root_graph = Graph()
        acc_init = root_graph.call_function(_tracing_ops._new_var, args=())
        acc_init.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        for_loop_node = root_graph.call_function(
            _tracing_ops._for_loop,
            args=(0, [0], [4], [acc_init]),
        )
        getitem_node = root_graph.call_function(
            operator.getitem, args=(for_loop_node, 0)
        )
        getitem_node.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        phi_node = root_graph.call_function(
            _tracing_ops._phi, args=(acc_init, getitem_node)
        )
        phi_node.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        if cast_dtype is not None:
            cast_node = root_graph.call_function(
                torch.ops.prims.convert_element_type.default,
                args=(phi_node, cast_dtype),
            )
            cast_node.meta["val"] = torch.empty(4, 4, dtype=cast_dtype)
            store_value = cast_node
        else:
            store_value = phi_node
        for store_dtype in store_dtypes:
            out_tensor = root_graph.call_function(_tracing_ops._new_var, args=())
            out_tensor.meta["val"] = torch.empty(4, 4, dtype=store_dtype)
            root_graph.call_function(
                memory_ops.store,
                args=(out_tensor, [0, 0], store_value, None),
            )
        root_graph.output(())

        graphs: list[GraphInfo] = [
            ForLoopGraphInfo(
                graph_id=0, graph=body_graph, node_args=[acc_init], block_ids=[1]
            ),
            RootGraphInfo(graph_id=1, graph=root_graph, phase_index=0),
        ]
        return mma_node, graphs

    def test_trace_mma_to_store_dtype_with_cast_returns_cast_dtype(self) -> None:
        """``acc.to(x.dtype)`` pattern: trace returns the post-cast dtype."""
        mma_node, graphs = self._build_matmul_store_graphs(
            store_dtypes=[torch.bfloat16],
            cast_dtype=torch.bfloat16,
        )
        self.assertEqual(_trace_mma_to_store_dtype(mma_node, graphs), torch.bfloat16)

    def test_trace_mma_to_store_dtype_direct_fp32_store(self) -> None:
        """Direct ``out[...] = acc`` (no cast): trace returns the
        store target dtype, not the matmul accumulator dtype."""
        mma_node, graphs = self._build_matmul_store_graphs(
            store_dtypes=[torch.float32],
            cast_dtype=None,
        )
        self.assertEqual(_trace_mma_to_store_dtype(mma_node, graphs), torch.float32)

    def test_trace_mma_to_store_dtype_multi_store_fan_out_returns_none(
        self,
    ) -> None:
        """The matmul value feeds two stores with different dtypes: the
        trace can't pin a unique target and must return ``None`` so the
        caller falls back and the cross-site assertion is the
        loud-failure backstop."""
        mma_node, graphs = self._build_matmul_store_graphs(
            store_dtypes=[torch.bfloat16, torch.float32],
            cast_dtype=None,
        )
        self.assertIsNone(_trace_mma_to_store_dtype(mma_node, graphs))

    def test_trace_mma_to_store_dtype_signature_collision_uses_correct_subgraph(
        self,
    ) -> None:
        """Two structurally identical K-loop bodies (e.g., a kernel with
        two distinct matmuls feeding different stores) collide on
        ``_graph_signature``. The tracer must disambiguate via codegen-
        graph identity (``mma_node.graph is gi.graph``) so each matmul
        sees the dtype of ITS consuming store, not the other matmul's.
        """
        from helion.language import memory_ops

        # Build two identical-shaped K-loop bodies; each has its own
        # matmul fx_node.
        def _make_body() -> tuple[Graph, torch.fx.Node, torch.fx.Node]:
            g = Graph()
            acc_in = g.placeholder("acc_in")
            acc_in.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
            mma = g.call_function(hl.dot, args=(acc_in,))
            mma.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
            g.output((mma,))
            return g, acc_in, mma

        body_a, acc_in_a, mma_a = _make_body()
        body_b, acc_in_b, mma_b = _make_body()

        root_graph = Graph()
        acc_init_a = root_graph.call_function(_tracing_ops._new_var, args=())
        acc_init_a.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        for_loop_a = root_graph.call_function(
            _tracing_ops._for_loop, args=(0, [0], [4], [acc_init_a])
        )
        getitem_a = root_graph.call_function(operator.getitem, args=(for_loop_a, 0))
        getitem_a.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        phi_a = root_graph.call_function(
            _tracing_ops._phi, args=(acc_init_a, getitem_a)
        )
        phi_a.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        out_a = root_graph.call_function(_tracing_ops._new_var, args=())
        out_a.meta["val"] = torch.empty(4, 4, dtype=torch.bfloat16)
        root_graph.call_function(memory_ops.store, args=(out_a, [0, 0], phi_a, None))

        acc_init_b = root_graph.call_function(_tracing_ops._new_var, args=())
        acc_init_b.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        for_loop_b = root_graph.call_function(
            _tracing_ops._for_loop, args=(1, [0], [4], [acc_init_b])
        )
        getitem_b = root_graph.call_function(operator.getitem, args=(for_loop_b, 0))
        getitem_b.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        phi_b = root_graph.call_function(
            _tracing_ops._phi, args=(acc_init_b, getitem_b)
        )
        phi_b.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        out_b = root_graph.call_function(_tracing_ops._new_var, args=())
        out_b.meta["val"] = torch.empty(4, 4, dtype=torch.float32)
        root_graph.call_function(memory_ops.store, args=(out_b, [0, 0], phi_b, None))
        root_graph.output(())

        graphs: list[GraphInfo] = [
            ForLoopGraphInfo(
                graph_id=0, graph=body_a, node_args=[acc_init_a], block_ids=[2]
            ),
            ForLoopGraphInfo(
                graph_id=1, graph=body_b, node_args=[acc_init_b], block_ids=[3]
            ),
            RootGraphInfo(graph_id=2, graph=root_graph, phase_index=0),
        ]

        # Despite identical body signatures, mma_a's user chain leads
        # to the bf16 store and mma_b's leads to the fp32 store.
        self.assertEqual(_trace_mma_to_store_dtype(mma_a, graphs), torch.bfloat16)
        self.assertEqual(_trace_mma_to_store_dtype(mma_b, graphs), torch.float32)

    def test_trace_mma_to_store_dtype_unknown_graph_returns_none(self) -> None:
        """When ``mma_node.graph`` is not in the supplied codegen graph
        list, the trace cannot proceed and must return ``None``."""
        mma_node, _ = self._build_matmul_store_graphs(
            store_dtypes=[torch.bfloat16], cast_dtype=None
        )
        self.assertIsNone(_trace_mma_to_store_dtype(mma_node, []))

    def test_emit_sched_pipeline_setup_round_trips_pipeline_async(self) -> None:
        """``_emit_sched_pipeline_setup`` emits the
        ``cutlass.pipeline.PipelineAsync.create`` wrapper used to
        broadcast tile coordinates from a scheduler warp to consumer
        warps. Mirrors the shape of Quack's ``make_sched_pipeline``
        and the existing inline scheduler emission in
        ``program_id._build_tcgen05_persistent_layout``.

        Four configurations are exercised:

        - Single-CTA, no defer-sync: ``consumer_mask`` and
          ``defer_sync`` are both omitted from the
          ``PipelineAsync.create`` call.
        - Cluster + defer-sync, ``consumer_mask_to_leader=True``: both
          ``consumer_mask=cutlass.Int32(0)`` and ``defer_sync=True``
          appear, matching the Quack ``make_sched_pipeline`` shape used
          by the cluster_m=2 ONE-CTA bridge.
        - Cluster + defer-sync, ``consumer_mask_to_leader=False``:
          ``defer_sync=True`` still appears (the pipeline still
          participates in the cluster-wide deferred-init protocol) but
          ``consumer_mask=`` is *omitted* — each CTA's empty barrier
          collects its own consumer arrivals. This is the
          ``ROLE_LOCAL_WITH_SCHEDULER`` shape; mismatching the topology
          and the cooperative-group arrive count causes a
          clean-on-cluster_m=1 / hang-on-cluster_m=2 regression.
        - Same ``DeviceFunction`` reused: the suffix on each subsequent
          plan's ``new_var`` outputs advances, confirming the helper
          does not memoize state on the device function.
        """
        df = _FakeDeviceFunction()
        plan = _new_tcgen05_sched_pipeline_plan(df)
        self.assertEqual(plan.barriers, "tcgen05_sched_pipeline_mbars_1")
        self.assertEqual(plan.pipeline, "tcgen05_sched_pipeline_1")

        stmts = _emit_sched_pipeline_setup(
            plan,
            sched_stage_count=2,
            consumer_arrive_count=15,
            cluster_size=1,
            defer_sync=False,
            producer_arrive_count=1,
        )
        emitted = "\n".join(ast.unparse(stmt) for stmt in stmts)

        self.assertEqual(len(stmts), 6)
        # mbar size is `2 * num_stages` Int64 slots, wrapped in
        # cutlass.Int32(...) — matches the existing acc-pipeline
        # barrier-storage shape and program_id.py's scheduler
        # emission.
        self.assertIn(
            "tcgen05_sched_pipeline_mbars_1 = cute.arch.alloc_smem("
            "cutlass.Int64, cutlass.Int32(4))",
            emitted,
        )
        # Producer group: caller-supplied arrive count (default 1
        # mirrors the existing scheduler emission in
        # ``program_id._build_tcgen05_persistent_prelude`` for the
        # cluster_m=2 ONE-CTA bridge — the producer warp leader
        # arrives once per stage). Bare ``Agent.Thread`` with no
        # count differs from that established shape and was a
        # pipeline-init misconfiguration source.
        self.assertIn(
            "tcgen05_sched_pipeline_producer_group_1 = "
            "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)",
            emitted,
        )
        # Consumer group: caller-supplied arrive count wrapped in
        # cutlass.Int32(...).
        self.assertIn(
            "tcgen05_sched_pipeline_consumer_group_1 = "
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(15))",
            emitted,
        )
        # PipelineAsync.create appears exactly once with neither
        # consumer_mask nor defer_sync — single-CTA, eager-init
        # path.
        self.assertEqual(emitted.count("cutlass.pipeline.PipelineAsync.create"), 1)
        self.assertIn(
            "cutlass.pipeline.PipelineAsync.create(num_stages=2, "
            "producer_group=tcgen05_sched_pipeline_producer_group_1, "
            "consumer_group=tcgen05_sched_pipeline_consumer_group_1, "
            "barrier_storage=tcgen05_sched_pipeline_mbars_1)",
            emitted,
        )
        self.assertNotIn("consumer_mask", emitted)
        self.assertNotIn("defer_sync", emitted)
        # Producer / consumer pipeline-state initializers.
        self.assertIn(
            "tcgen05_sched_pipeline_producer_state_1 = "
            "cutlass.pipeline.make_pipeline_state("
            "cutlass.pipeline.PipelineUserType.Producer, 2)",
            emitted,
        )
        self.assertIn(
            "tcgen05_sched_pipeline_consumer_state_1 = "
            "cutlass.pipeline.make_pipeline_state("
            "cutlass.pipeline.PipelineUserType.Consumer, 2)",
            emitted,
        )

        # Cluster + defer-sync path. Reuse the same DeviceFunction so
        # the suffix on the second plan advances to ``_2`` —
        # validates that ``_new_tcgen05_sched_pipeline_plan`` is a
        # pure allocator with no memoization.
        plan_cluster = _new_tcgen05_sched_pipeline_plan(df)
        self.assertEqual(plan_cluster.barriers, "tcgen05_sched_pipeline_mbars_2")
        self.assertEqual(plan_cluster.pipeline, "tcgen05_sched_pipeline_2")
        cluster_stmts = _emit_sched_pipeline_setup(
            plan_cluster,
            sched_stage_count=2,
            consumer_arrive_count=16,
            cluster_size=2,
            defer_sync=True,
            producer_arrive_count=1,
        )
        cluster_emitted = "\n".join(ast.unparse(s) for s in cluster_stmts)
        # PipelineAsync.create now carries both consumer_mask
        # (wrapped via cutlass.Int32 to match the existing scheduler
        # emission in program_id.py) and defer_sync=True so the
        # pipeline participates in the cluster-wide deferred-init
        # protocol.
        self.assertIn(
            "cutlass.pipeline.PipelineAsync.create(num_stages=2, "
            "producer_group=tcgen05_sched_pipeline_producer_group_2, "
            "consumer_group=tcgen05_sched_pipeline_consumer_group_2, "
            "barrier_storage=tcgen05_sched_pipeline_mbars_2, "
            "consumer_mask=cutlass.Int32(0), defer_sync=True)",
            cluster_emitted,
        )
        self.assertIn("cutlass.Int32(16)", cluster_emitted)
        self.assertEqual(len(cluster_stmts), 6)

        # Per-CTA topology (``consumer_mask_to_leader=False``):
        # cluster_size > 1 + defer_sync=True keeps the cluster-wide
        # init protocol but each CTA's empty barrier collects its own
        # arrivals. ``ROLE_LOCAL_WITH_SCHEDULER`` uses this shape.
        plan_per_cta = _new_tcgen05_sched_pipeline_plan(df)
        self.assertEqual(plan_per_cta.barriers, "tcgen05_sched_pipeline_mbars_3")
        per_cta_stmts = _emit_sched_pipeline_setup(
            plan_per_cta,
            sched_stage_count=1,
            consumer_arrive_count=6,
            cluster_size=2,
            defer_sync=True,
            producer_arrive_count=1,
            consumer_mask_to_leader=False,
        )
        per_cta_emitted = "\n".join(ast.unparse(s) for s in per_cta_stmts)
        # ``defer_sync=True`` still appears so the pipeline init
        # coordinates with the AB / acc / c pipelines.
        self.assertIn("defer_sync=True", per_cta_emitted)
        # ``consumer_mask=`` must NOT appear — empty-barrier arrivals
        # stay local to each CTA. Asserting on the literal substring
        # avoids matching the ``mcast_mask`` family used elsewhere.
        self.assertNotIn("consumer_mask=", per_cta_emitted)
        self.assertIn("cutlass.Int32(6)", per_cta_emitted)
        self.assertEqual(len(per_cta_stmts), 6)

    def test_tcgen05_codegen_emits_cluster_and_role_split_knobs(self) -> None:
        @helion.kernel(backend="cute")
        def cute_matmul_mma_codegen_only(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        args = (
            torch.randn(8192, 8192, device=DEVICE, dtype=torch.float16),
            torch.randn(8192, 8192, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(
            block_sizes=[256, 256, 16],
            tcgen05_cluster_m=2,
            tcgen05_num_epi_warps=4,
            pid_type="persistent_blocked",
        )

        with (
            patch.dict("os.environ", {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch_cute_mma_support(),
        ):
            bound = cute_matmul_mma_codegen_only.bind(args)
            # matmul_ops narrows tcgen05_cluster_m to (1,) when binding the
            # bf16/fp16 matmul until the cluster=2 path is benchmarked. This
            # K-over-cap codegen-only test still exercises the guarded
            # cluster=2 structure, so widen the choices back to (2, 1).
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.env.config_spec.restrict_tcgen05_cluster_m_search((2, 1))
            code = bound.to_triton_code(config)

        self.assertIn(
            "'cluster_m': 2",
            code,
        )
        self.assertIn(
            "'cluster_n': 1",
            code,
        )
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertIn("_helion_tcgen05_persistent_total_tiles", code)
        self.assertIn("above the validated K-tile limit", code)
        self.assertIn(
            "PersistentTileSchedulerParams(((8192 + _BLOCK_SIZE_0 - 1) // "
            "_BLOCK_SIZE_0 * 2",
            code,
        )
        self.assertIn(
            "virtual_pid = tcgen05_role_local_0_work_tile.tile_idx[0] "
            "// cutlass.Int32(2)",
            code,
        )
        self.assertIn("tcgen05_role_local_0_tile_sched_params", code)
        self.assertIn("tcgen05_role_local_1_tile_sched_params", code)
        self.assertIn("tcgen05_role_local_2_tile_sched_params", code)
        self.assertIn("tcgen05_ab_pipeline.producer_acquire", code)
        self.assertIn(f"if {_TCGEN05_CLUSTER_LEADER_PREDICATE}:", code)
        self.assertIn("cute.copy(tma_atom_a", code)
        self.assertIn(
            "if tcgen05_exec_active and "
            "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
            "== cutlass.Int32(0):",
            code,
        )
        self.assertIn("'kind': 'tcgen05_d_tma'", code)
        self.assertIn("cutlass.pipeline.PipelineTmaStore.create", code)
        self.assertNotIn("cute.nvgpu.CopyUniversalOp()", code)
        self.assertIn("_BLOCK_SIZE_0 = 256", code)
        self.assertIn("_BLOCK_SIZE_1 = 256", code)
        self.assertIn("cute.arch.block_idx_in_cluster()", code)
        self.assertIn("tcgen05_tma_warp = tcgen05_warp_idx == cutlass.Int32(5)", code)
        # `tcgen05_ab_load_active` was dropped: with a single A/B load warp at
        # `tma_warp_id`, `tma_warp` already covers the same predicate, and the
        # field is reintroducible if role-local persistent loops grow a second
        # A/B load warp.
        self.assertNotIn("tcgen05_ab_load_active", code)
        self.assertIn(
            "tcgen05_exec_active = tcgen05_warp_idx == cutlass.Int32(4)", code
        )
        self.assertIn(
            "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)",
            code,
        )
        # Dropped dead warps 6 (epi_load) and 7 (scheduler) -- the role
        # split now launches a 6-warp CTA with no idle padding.
        self.assertNotIn("tcgen05_epi_load_warp", code)
        self.assertNotIn("tcgen05_scheduler_warp", code)
        self.assertIn("block=(32, 6, 1)", code)
        self.assertIn("cta_layout_vmnk=tcgen05_cluster_layout_vmnk", code)
        self.assertIn("cutlass.utils.StaticPersistentTileScheduler.create(", code)
        self.assertIn(
            "cutlass.pipeline.pipeline_init_arrive(cluster_shape_mn=tcgen05_cluster_layout_vmnk, is_relaxed=True)",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=tcgen05_cluster_layout_vmnk)",
            code,
        )
        self.assertIn("while tcgen05_role_local_0_work_tile.is_valid_tile", code)
        self.assertNotIn("while tcgen05_work_tile_valid", code)
        self.assertNotIn("for virtual_pid in ", code)

    def test_cute_grouped_sum_reduction_uses_tree_for_non_warp_multiple_groups(
        self,
    ) -> None:
        cg = _FakeCuteReductionCodegen()
        env = SimpleNamespace(
            backend=CuteBackend(),
            index_dtype=torch.int32,
        )
        loop_state = SimpleNamespace(block_thread_axes={1: 1})

        with patch.object(CompileEnvironment, "current", return_value=env):
            result = _emit_cute_grouped_sum_reduction(
                cg,
                "dot_input",
                value_dtype=torch.float32,
                loop_state=loop_state,
                k_block_id=1,
            )

        self.assertEqual(result, "dot_reduce_result_1")
        emitted = "\n".join(
            ast.unparse(stmt) if isinstance(stmt, ast.AST) else str(stmt)
            for stmt in cg.statements
        )
        self.assertNotIn("cute.arch.warp_reduction_sum", emitted)
        self.assertIn("_cute_grouped_reduce_shared_tree", emitted)

    def test_cute_index_exprs_skip_none_axes_and_zero_singletons(self) -> None:
        state = SimpleNamespace(
            codegen=SimpleNamespace(
                active_device_loops={
                    1: [SimpleNamespace(strategy=_FakeLoopStrategy([1]))],
                },
                lift=lambda expr, *, dce=False, prefix="tmp": SimpleNamespace(
                    id="lifted_index"
                ),
            ),
            sympy_expr=lambda expr: str(expr),
        )
        env = SimpleNamespace(
            get_block_id=lambda size: 1 if int(size) == 8 else None,
            known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
            resolve_block_id=lambda size: 1 if int(size) == 8 else None,
            block_sizes=[SimpleNamespace(size=8, block_id=1)],
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertEqual(
                _cute_index_exprs(
                    state,
                    [None, slice(None)],
                    tensor=torch.empty(8),
                    inactive_slice_expr="None",
                    inactive_singleton_slice_expr="0",
                ),
                ["indices_1"],
            )
            self.assertEqual(
                _cute_index_exprs(
                    state,
                    [slice(None), slice(None)],
                    tensor=torch.empty(1, 8),
                    inactive_slice_expr="None",
                    inactive_singleton_slice_expr="0",
                ),
                ["0", "indices_1"],
            )

    def test_cute_combined_mask_skips_none_axes(self) -> None:
        state = SimpleNamespace(
            codegen=_FakeMaskCodegen(_FakeMaskedLoopStrategy([1]), {1})
        )
        env = SimpleNamespace(
            get_block_id=lambda size: 1 if int(size) == 8 else None,
            known_equal=lambda lhs, rhs: int(lhs) == int(rhs),
            resolve_block_id=lambda size: 1 if int(size) == 8 else None,
            block_sizes=[SimpleNamespace(size=8, block_id=1)],
        )

        with patch.object(CompileEnvironment, "current", return_value=env):
            self.assertEqual(
                _cute_combined_mask(
                    state,
                    [None, slice(None)],
                    None,
                    tensor=torch.empty(8),
                ),
                "(mask_1)",
            )


@onlyBackends(["cute"])
class TestCuteDslCompat(unittest.TestCase):
    def test_umma_async_tail_workaround_preserves_leader_cta_guard(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with patch.object(
            cutedsl_compat, "cutedsl_has_opresultlist_fix", return_value=False
        ):
            src = cutedsl_compat.emit_producer_tail_umma_async(
                "acc_pipeline", "acc_state", num_stages=3
            )

        self.assertIn("cute.arch.block_idx_in_cluster()", src)
        self.assertIn("cute.arch.make_warp_uniform(_pt_bidx)", src)
        leader_guard = "if _pt_cta_rank % cutlass.Int32(2) == cutlass.Int32(0):"
        self.assertIn(leader_guard, src)
        self.assertLess(
            src.index(leader_guard),
            src.index("acc_state._count = acc_state._count + cutlass.Int32(1)"),
        )
        self.assertLess(
            src.index("acc_state._count = acc_state._count + cutlass.Int32(1)"),
            src.index("acc_pipeline.producer_acquire(acc_state)"),
        )
        self.assertEqual(
            src.count("acc_state._count = acc_state._count + cutlass.Int32(1)"),
            2,
        )
        self.assertNotIn("acc_pipeline.sync_object_empty.wait", src)
        self.assertIn("acc_pipeline.producer_acquire(acc_state)", src)

    def test_tma_umma_tail_uses_upstream_when_advance_and_tail_safe(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with (
            patch.object(
                cutedsl_compat, "cutedsl_has_opresultlist_fix", return_value=True
            ),
            patch.object(
                cutedsl_compat,
                "cutedsl_tma_umma_tail_has_peer_cta_semantics",
                return_value=True,
            ),
        ):
            self.assertEqual(
                cutedsl_compat.emit_producer_tail_tma_umma(
                    "ab_pipeline", "ab_state", num_stages=3
                ),
                "ab_pipeline.producer_tail(ab_state)",
            )

    def test_tma_umma_tail_inlines_when_tail_semantics_unsafe(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with (
            patch.object(
                cutedsl_compat, "cutedsl_has_opresultlist_fix", return_value=True
            ),
            patch.object(
                cutedsl_compat,
                "cutedsl_tma_umma_tail_has_peer_cta_semantics",
                return_value=False,
            ),
        ):
            src = cutedsl_compat.emit_producer_tail_tma_umma(
                "ab_pipeline", "ab_state", num_stages=3
            )

        self.assertNotIn("block_idx_in_cluster", src)
        self.assertNotIn("_pt_cta_rank", src)
        self.assertNotIn("if True", src)
        self.assertLess(
            src.index("ab_state._count = ab_state._count + cutlass.Int32(1)"),
            src.index("ab_pipeline.producer_acquire(ab_state)"),
        )
        self.assertEqual(
            src.count("ab_state._count = ab_state._count + cutlass.Int32(1)"),
            2,
        )
        self.assertIn("ab_pipeline.producer_acquire(ab_state)", src)

    def test_tma_umma_tail_inlines_when_advance_workaround_needed(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with patch.object(
            cutedsl_compat, "cutedsl_has_opresultlist_fix", return_value=False
        ):
            src = cutedsl_compat.emit_producer_tail_tma_umma(
                "ab_pipeline", "ab_state", num_stages=3
            )

        self.assertNotIn("block_idx_in_cluster", src)
        self.assertNotIn("if True", src)
        self.assertIn("ab_pipeline.producer_acquire(ab_state)", src)

    def test_tma_umma_tail_can_skip_state_advances(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with (
            patch.object(
                cutedsl_compat, "cutedsl_has_opresultlist_fix", return_value=True
            ),
            patch.object(
                cutedsl_compat,
                "cutedsl_tma_umma_tail_has_peer_cta_semantics",
                return_value=True,
            ),
        ):
            src = cutedsl_compat.emit_producer_tail_tma_umma(
                "ab_pipeline",
                "ab_state",
                num_stages=3,
                skip_advances=True,
            )

        self.assertEqual(src, "ab_pipeline.producer_acquire(ab_state)")
        self.assertNotIn("producer_tail", src)
        self.assertNotIn("ab_state._count", src)
        self.assertNotIn("ab_state.advance", src)

    def test_tma_umma_tail_detector_allows_state_rename(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        cutedsl_compat.cutedsl_tma_umma_tail_has_peer_cta_semantics.cache_clear()
        src = """
def producer_tail(self, producer_state):
    for i in range(self.num_stages - 1):
        producer_state.advance()
    self.producer_acquire(producer_state)
"""
        try:
            with patch.object(cutedsl_compat.inspect, "getsource", return_value=src):
                self.assertTrue(
                    cutedsl_compat.cutedsl_tma_umma_tail_has_peer_cta_semantics()
                )
        finally:
            cutedsl_compat.cutedsl_tma_umma_tail_has_peer_cta_semantics.cache_clear()


@onlyBackends(["cute"])
class TestPersistentLoopSplitter(unittest.TestCase):
    """Unit tests for the per-tile / post-loop splitters and the
    role-block partitioner on ``Tcgen05PersistentProgramIDs``.

    Codegen tags every statement that depends on per-tile coordinates via
    ``register_cute_tcgen05_per_tile_stmts``. The splitter then hoists the
    rest. Statements that read or write a per-tile name (transitively
    seeded from ``virtual_pid_var``) are also kept in the wrapped body, so
    the PID decomposition emitted by ``_decompose_virtual_pid`` doesn't
    have to plumb tagging through every callsite.

    Statements registered via role-specific tcgen05 registration methods
    are routed through ``_collect_tcgen05_role_blocks`` into role blocks;
    everything else stays in shared blocks. The consumer
    ``_emit_role_block_stmts`` wraps non-shared blocks in
    ``if {role_predicate}: ...``.
    """

    def _make_helper(self) -> tuple[object, object]:
        """Construct a real splitter and a minimal device-function stand-in.

        The splitter only needs the predicate methods from device function
        (``has_cute_tcgen05_per_tile_marks``, ``is_cute_tcgen05_per_tile``,
        and the post-loop equivalents). ``Tcgen05PersistentProgramIDs`` is
        a concrete subclass of ``ProgramIDs`` with the splitter; instantiate
        it without ``__init__`` to skip the device-function plumbing."""

        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        class _MinimalDeviceFunction:
            def __init__(self) -> None:
                self._per_tile_ids: set[int] = set()
                self._post_loop_ids: set[int] = set()
                self._tma_load_role_ids: set[int] = set()
                self._mma_exec_role_ids: set[int] = set()
                self._epi_role_ids: set[int] = set()
                self.cute_tcgen05_epi_role_tile_counter_var: str | None = None

            def register_cute_tcgen05_per_tile_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._per_tile_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_per_tile(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._per_tile_ids

            @property
            def has_cute_tcgen05_per_tile_marks(self) -> bool:
                return bool(self._per_tile_ids)

            def register_cute_tcgen05_post_loop_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._post_loop_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_post_loop(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._post_loop_ids

            @property
            def has_cute_tcgen05_post_loop_marks(self) -> bool:
                return bool(self._post_loop_ids)

            def register_cute_tcgen05_tma_load_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                # The real ``DeviceFunction`` accepts both top-level
                # tagged statements (which must also be per-tile-registered)
                # and tagged children inside a per-tile container (e.g.
                # the K-loop body). The unit suite mirrors that: tag any
                # statement, whether or not it's per-tile-registered.
                self._tma_load_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_tma_load_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._tma_load_role_ids

            @property
            def has_cute_tcgen05_tma_load_role_marks(self) -> bool:
                return bool(self._tma_load_role_ids)

            @property
            def cute_tcgen05_tma_load_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._tma_load_role_ids)

            def register_cute_tcgen05_mma_exec_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._mma_exec_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_mma_exec_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._mma_exec_role_ids

            @property
            def has_cute_tcgen05_mma_exec_role_marks(self) -> bool:
                return bool(self._mma_exec_role_ids)

            @property
            def cute_tcgen05_mma_exec_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._mma_exec_role_ids)

            def register_cute_tcgen05_epi_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._epi_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_epi_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._epi_role_ids

            @property
            def has_cute_tcgen05_epi_role_marks(self) -> bool:
                return bool(self._epi_role_ids)

            @property
            def cute_tcgen05_epi_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._epi_role_ids)

        splitter = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
        # The splitter walks ASTs looking for references to the
        # virtual_pid var. Tests use simple statements that don't mention
        # any pid, so an unused sentinel name is fine.
        splitter.virtual_pid_var = "__test_virtual_pid__"  # type: ignore[attr-defined]
        # ``_collect_tcgen05_role_blocks`` calls
        # ``_tcgen05_tma_load_role_predicate`` which would normally hit
        # ``DeviceFunction.current().cute_tcgen05_matmul_plan``. Replace
        # with a sentinel string the tests can match against, so this
        # unit suite stays decoupled from the real DeviceFunction stack.
        splitter._tcgen05_tma_load_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_tma_load_warp__"
        )
        splitter._tcgen05_mma_exec_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_mma_exec_warp__"
        )
        splitter._tcgen05_epi_role_predicate = (  # type: ignore[attr-defined]
            lambda: "__test_epi_warp__"
        )
        return splitter, _MinimalDeviceFunction()

    def _stmt(self, text: str) -> ast.stmt:
        return ast.parse(text).body[0]

    def _make_role_local_stubs(self, *, num_pid_dims: int = 2) -> tuple[object, object]:
        """Build a richer device-function stub plus per-pid stubs that
        the role-local-while builders need (``new_var`` for variable
        allocation, ``num_pids_expr`` for scheduler tile counts, and
        the per-tile / TMA-load registration methods used by
        ``_partition_tcgen05_role_blocks``).

        Returns ``(stub_device_function, splitter)``. The splitter is
        the same one this class's ``_make_helper`` returns, but with
        ``pid_info`` bound to ``num_pid_dims`` stand-in PIDs."""

        class _StubDeviceFunction:
            def __init__(self) -> None:
                self._counter = 0
                self._per_tile_ids: set[int] = set()
                self._tma_load_role_ids: set[int] = set()
                self._mma_exec_role_ids: set[int] = set()
                self._epi_role_ids: set[int] = set()
                self.cute_tcgen05_epi_role_tile_counter_var: str | None = None

            def new_var(self, name: str) -> str:
                self._counter += 1
                return f"{name}__{self._counter}"

            def register_cute_tcgen05_per_tile_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._per_tile_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_per_tile(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._per_tile_ids

            @property
            def has_cute_tcgen05_per_tile_marks(self) -> bool:
                return bool(self._per_tile_ids)

            def register_cute_tcgen05_tma_load_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._tma_load_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_tma_load_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._tma_load_role_ids

            @property
            def has_cute_tcgen05_tma_load_role_marks(self) -> bool:
                return bool(self._tma_load_role_ids)

            @property
            def cute_tcgen05_tma_load_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._tma_load_role_ids)

            def register_cute_tcgen05_mma_exec_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._mma_exec_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_mma_exec_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._mma_exec_role_ids

            @property
            def has_cute_tcgen05_mma_exec_role_marks(self) -> bool:
                return bool(self._mma_exec_role_ids)

            @property
            def cute_tcgen05_mma_exec_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._mma_exec_role_ids)

            def register_cute_tcgen05_epi_role_stmts(
                self, stmts: list[ast.AST]
            ) -> None:
                self._epi_role_ids.update(id(s) for s in stmts)

            def is_cute_tcgen05_epi_role(self, stmt: ast.AST) -> bool:
                return id(stmt) in self._epi_role_ids

            @property
            def has_cute_tcgen05_epi_role_marks(self) -> bool:
                return bool(self._epi_role_ids)

            @property
            def cute_tcgen05_epi_role_stmt_ids(self) -> frozenset[int]:
                return frozenset(self._epi_role_ids)

        class _PidStub:
            def num_pids_expr(self, *, is_device: bool) -> str:
                return "16"

        splitter, _ = self._make_helper()
        splitter.pid_info = [_PidStub() for _ in range(num_pid_dims)]  # type: ignore[attr-defined]
        return _StubDeviceFunction(), splitter

    def _make_minimal_layout(self, *, cluster_m: int = 1) -> object:
        """Build a fake ``_Tcgen05PersistentLayout`` with the minimum
        fields used by the shared-while tile body builder and the
        role-local while builder. The default ``cluster_m=1`` keeps
        the cluster-only branches inert; tests that exercise cluster
        behavior can override."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        return Tcgen05PersistentProgramIDs._Tcgen05PersistentLayout(
            cluster_m=cluster_m,
            scheduler_owner_warp="owner_warp",
            cluster_scheduler_leader="cluster_leader",
            consumer_leader_var="consumer_leader",
            scheduler_leader_predicate="leader_pred",
            tile_sched_params_var="tile_sched_params",
            tile_sched_var="tile_sched",
            work_tile_var="work_tile",
            work_tile_smem_ptr="work_tile_smem_ptr",
            work_tile_smem="work_tile_smem",
            work_tile_smem_tensor="work_tile_smem_t",
            work_tile_coord_vars=["c_0"],
            work_tile_valid_var="work_tile_valid",
            linear_pid_expr="c_0",
            sched_pipeline_mbars="sm",
            sched_pipeline="sp",
            sched_pipeline_producer_group="pg",
            sched_pipeline_consumer_group="cg",
            sched_producer_state="ps",
            sched_consumer_state="cs",
            sched_barrier_ptr="bp",
            sched_peer_rank="pr",
            sched_peer_m="pm",
            refresh_work_tile_stmts=[self._stmt("c_0 = work_tile_smem[0]")],
            work_tile_publish_stmts=[
                self._stmt("work_tile_smem[0] = work_tile.tile_idx[0]")
            ],
            work_tile_consume_stmts=[],
            work_tile_release_stmts=[],
        )

    def test_tcgen05_persistent_foreach_multi_root_keeps_host_guard(self) -> None:
        """Multi-root tcgen05 role-local codegen is guarded as unvalidated.

        ``ForEachProgramID.codegen_grid()`` delegates to the first case's
        grid today. Until tcgen05 grows a scheduler/grid that spans all root
        cases, the validated guard-lift path must exclude multi-root kernels
        and the host guard must count the combined case space.
        """

        from helion._compiler.program_id import ForEachProgramID
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        class _Case:
            def __init__(self, expr: str) -> None:
                self.expr = expr

            def total_pids_expr(self, *, is_device: bool) -> str:
                return self.expr

        fake_pid = ForEachProgramID("pid_shared")
        fake_pid.cases = [_Case("0"), _Case("1")]  # type: ignore[list-item]
        self_stmt = self._stmt("shared_work = 1")

        class _DeviceFunction:
            def __init__(self) -> None:
                self.body = [self_stmt]
                self.pid = fake_pid
                self.codegen = SimpleNamespace(host_statements=[])

        splitter, _ = self._make_helper()
        splitter.virtual_pid_var = "virtual_pid"  # type: ignore[attr-defined]
        layout = self._make_minimal_layout(cluster_m=1)
        role_stmt = self._stmt("role_work = 1")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="role_warp", stmts=[role_stmt]
        )
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[role_block],
            role_blocks_extracted=[role_block],
            shared_body_extracted=[],
        )
        retargeted: list[object] = []
        device_function = _DeviceFunction()

        splitter._extract_tcgen05_post_loop_stmts = (  # type: ignore[attr-defined]
            lambda device, body: (body, [])
        )
        splitter._split_tcgen05_invariant_setup = (  # type: ignore[attr-defined]
            lambda device, body: ([], body)
        )
        splitter._build_tcgen05_persistent_layout = (  # type: ignore[attr-defined]
            lambda device: layout
        )
        splitter._partition_tcgen05_role_blocks = (  # type: ignore[attr-defined]
            lambda device, body: partition
        )
        splitter._retarget_tcgen05_shared_scheduler_to_exec = (  # type: ignore[attr-defined]
            lambda layout_arg: retargeted.append(layout_arg)
        )
        splitter._build_tcgen05_persistent_prelude = (  # type: ignore[attr-defined]
            lambda layout_arg: []
        )
        splitter._build_tcgen05_persistent_tile_body_role_local = (  # type: ignore[attr-defined]
            lambda device, layout_arg, partition_arg: ([], [])
        )

        splitter._setup_tcgen05_persistent_kernel(device_function)

        self.assertEqual(retargeted, [layout])
        total_var = Tcgen05PersistentProgramIDs._MULTI_TILE_GUARD_TOTAL_VAR
        host_src = "\n".join(
            ast.unparse(stmt) for stmt in device_function.codegen.host_statements
        )
        self.assertIn(f"{total_var} = 0 + 1", host_src)
        self.assertIn(f"if {total_var} > 0", host_src)
        self.assertIn("supports runtime execution only", host_src)

    def test_unmarked_statements_are_hoisted(self) -> None:
        splitter, df = self._make_helper()
        invariant_a = self._stmt("tma_atom_a = make_atom()")
        invariant_b = self._stmt("ab_pipeline = PipelineTmaUmma.create()")
        per_tile_a = self._stmt("gA = cute.local_tile(tA, (128, 16), (m // 128, None))")
        per_tile_b = self._stmt("acc_pipeline.producer_acquire(state)")
        body = [invariant_a, invariant_b, per_tile_a, per_tile_b]
        df.register_cute_tcgen05_per_tile_stmts([per_tile_a, per_tile_b])
        hoisted, wrapped = splitter._split_tcgen05_invariant_setup(df, body)
        self.assertEqual(hoisted, [invariant_a, invariant_b])
        self.assertEqual(wrapped, [per_tile_a, per_tile_b])

    def test_relative_order_is_preserved_in_each_slice(self) -> None:
        """The splitter walks ``body`` in order, so hoisted and wrapped
        slices each preserve the input ordering. This is required:
        pipeline ``.create(...)`` must run before the per-tile
        producer_acquire, and consumer_state must be initialized before
        any tile body uses it."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        per_tile = self._stmt("g = local_tile(t, (128, 16), (m, None))")
        b = self._stmt("b = 2")
        df.register_cute_tcgen05_per_tile_stmts([per_tile])
        hoisted, wrapped = splitter._split_tcgen05_invariant_setup(df, [a, per_tile, b])
        self.assertEqual(hoisted, [a, b])
        self.assertEqual(wrapped, [per_tile])

    def test_no_split_when_no_per_tile_marks(self) -> None:
        """With no per-tile registration, the splitter returns the body
        unchanged (no hoisting). The persistent setup will then wrap the
        whole body each iteration. This is the safe default for kernels
        that do not opt into the per-tile splitter."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        hoisted, wrapped = splitter._split_tcgen05_invariant_setup(df, body)
        self.assertEqual(hoisted, [])
        self.assertEqual(wrapped, body)

    def test_post_loop_extraction_removes_marked_stmts(self) -> None:
        """``_extract_tcgen05_post_loop_stmts`` must move post-loop tagged
        statements out of the body so the splitter never sees them. The
        relative order of the remaining body and the extracted post-loop
        statements is preserved independently."""
        splitter, df = self._make_helper()
        per_tile = self._stmt("g = local_tile(t, (128, 16), (m, None))")
        post_a = self._stmt("acc_pipeline.producer_tail(state)")
        post_b = self._stmt("tmem_alloc.free(ptr)")
        df.register_cute_tcgen05_per_tile_stmts([per_tile])
        df.register_cute_tcgen05_post_loop_stmts([post_a, post_b])
        body = [per_tile, post_a, post_b]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, [per_tile])
        self.assertEqual(post_loop, [post_a, post_b])

    def test_post_loop_extraction_passthrough_when_unmarked(self) -> None:
        """With no post-loop marks, extraction is a no-op so the splitter
        just sees the original body. Important: this is the non-persistent
        kernel path, where moving statements is incorrect."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, body)
        self.assertEqual(post_loop, [])

    def test_post_loop_extraction_preserves_relative_order(self) -> None:
        """Both the remaining body and the extracted post-loop list keep
        the input ordering. The splitter never reorders within either
        slice, which lets codegen rely on emit order."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        post_a = self._stmt("acc_pipeline.producer_tail(state)")
        b = self._stmt("b = 2")
        post_b = self._stmt("tmem_alloc.free(ptr)")
        c = self._stmt("c = 3")
        df.register_cute_tcgen05_post_loop_stmts([post_a, post_b])
        body = [a, post_a, b, post_b, c]
        remaining, post_loop = splitter._extract_tcgen05_post_loop_stmts(df, body)
        self.assertEqual(remaining, [a, b, c])
        self.assertEqual(post_loop, [post_a, post_b])

    def test_role_blocks_single_block_when_no_tma_load_marks(self) -> None:
        """``_collect_tcgen05_role_blocks`` returns a single shared block
        carrying the full body when no TMA-load tags are present. This is
        the safe default for non-tcgen05 / universal-MMA paths and any
        kernel that never registers TMA-load role tags. The downstream
        consumer ``_build_tcgen05_persistent_tile_body`` handles the
        single-block case identically to the pre-split implementation."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, body)

    def test_role_blocks_partition_tagged_runs(self) -> None:
        """Each maximal run of consecutive TMA-load-tagged statements
        becomes its own role block; surrounding shared statements stay
        in shared blocks. Block ordering preserves the input emit order
        of the per-tile body."""
        splitter, df = self._make_helper()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        tma_load_y = self._stmt("cute.copy(t, s)")
        shared_b = self._stmt("b = 2")
        tma_load_z = self._stmt("tma_pipeline.producer_commit(s)")
        df.register_cute_tcgen05_per_tile_stmts([tma_load_x, tma_load_y, tma_load_z])
        df.register_cute_tcgen05_tma_load_role_stmts(
            [tma_load_x, tma_load_y, tma_load_z]
        )
        body = [shared_a, tma_load_x, tma_load_y, shared_b, tma_load_z]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 4)
        # First block: shared_a (before any tma_load run)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, [shared_a])
        # Second block: tma_load_x + tma_load_y (consecutive)
        self.assertEqual(blocks[1].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[1].stmts, [tma_load_x, tma_load_y])
        # Third block: shared_b
        self.assertIsNone(blocks[2].role_predicate)
        self.assertEqual(blocks[2].stmts, [shared_b])
        # Fourth block: tma_load_z (singleton run)
        self.assertEqual(blocks[3].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[3].stmts, [tma_load_z])

    def test_role_blocks_partition_distinguishes_warp_roles(self) -> None:
        splitter, df = self._make_helper()
        shared = self._stmt("tile = pid")
        tma_load = self._stmt("tma_pipeline.producer_acquire(s)")
        mma_exec = self._stmt("acc_pipeline.producer_acquire(s)")
        epi = self._stmt("acc_pipeline.consumer_wait(s)")
        df.register_cute_tcgen05_per_tile_stmts([tma_load, mma_exec, epi])
        df.register_cute_tcgen05_tma_load_role_stmts([tma_load])
        df.register_cute_tcgen05_mma_exec_role_stmts([mma_exec])
        df.register_cute_tcgen05_epi_role_stmts([epi])

        blocks = splitter._collect_tcgen05_role_blocks(
            df, [shared, tma_load, mma_exec, epi]
        )

        self.assertEqual(
            [block.role_predicate for block in blocks],
            [
                None,
                "__test_tma_load_warp__",
                "__test_mma_exec_warp__",
                "__test_epi_warp__",
            ],
        )
        self.assertEqual(blocks[1].stmts, [tma_load])
        self.assertEqual(blocks[2].stmts, [mma_exec])
        self.assertEqual(blocks[3].stmts, [epi])

    def test_role_blocks_preserve_relative_order_in_emit(self) -> None:
        """Tagged statements stay in their original positions relative
        to surrounding shared statements. This matters because the
        ``tma_initial_full_tile`` boolean is set in a shared statement
        BEFORE the TMA-load-tagged prefetch IF reads it; reordering
        would dangle a free reference and CuTe DSL would error with
        'cannot access free variable ...'."""
        splitter, df = self._make_helper()
        define_bool = self._stmt("ready = True")
        prefetch = self._stmt("if ready and tma_warp:\n    tma_pipeline.acquire()")
        df.register_cute_tcgen05_per_tile_stmts([prefetch])
        df.register_cute_tcgen05_tma_load_role_stmts([prefetch])
        body = [define_bool, prefetch]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        # define_bool must appear in the FIRST emitted block; prefetch in
        # the SECOND. Reversing the order would put `prefetch` ahead of
        # `define_bool` and `ready` would be undefined when prefetch reads it.
        self.assertEqual(len(blocks), 2)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertEqual(blocks[0].stmts, [define_bool])
        self.assertEqual(blocks[1].role_predicate, "__test_tma_load_warp__")
        self.assertEqual(blocks[1].stmts, [prefetch])

    def test_role_blocks_emit_consumer_wraps_tma_load_in_role_if(self) -> None:
        """The consumer ``_emit_role_block_stmts`` wraps non-shared
        blocks in ``if {role_predicate}: ...`` and emits shared blocks
        as naked statements. An empty block is a no-op (no degenerate
        ``if {}:`` is emitted)."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        splitter, _df = self._make_helper()
        s = self._stmt("a = 1")

        shared = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate=None, stmts=[s]
        )
        out = splitter._emit_role_block_stmts(shared)
        self.assertEqual(out, [s])

        gated = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="my_warp", stmts=[s]
        )
        out = splitter._emit_role_block_stmts(gated)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], ast.If)
        self.assertEqual(ast.unparse(out[0].test), "my_warp")
        self.assertEqual(out[0].body, [s])

        empty_gated = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="my_warp", stmts=[]
        )
        self.assertEqual(splitter._emit_role_block_stmts(empty_gated), [])

    def test_role_blocks_recurse_into_top_level_for_loop(self) -> None:
        """The role partitioner recurses one level into top-level
        ``for`` / ``while`` loop bodies and wraps tagged children in
        ``if {role_predicate}: <child>`` in place. The containing loop
        stays in the shared block. This is structural prep for the
        TMA-load role's role-local-while lift in ``cute_plan.md`` step
        3b: today the per-K-iter producer ``if`` block lives inside
        the K-loop body and gets wrapped by the recursion; step 3b
        lifts it out as a separate top-level statement.
        """
        splitter, df = self._make_helper()
        producer = self._stmt(
            "if tma_full_tile and tma_warp:\n    tma_pipeline.producer_acquire(state)"
        )
        consumer = self._stmt(
            "if tma_full_tile:\n"
            "    if exec_active:\n"
            "        tma_pipeline.consumer_wait(state)"
        )
        kloop = ast.parse("for k in range(K):\n    pass").body[0]
        # Replace the dummy ``pass`` body with the real two child
        # statements so we can tag the producer.
        kloop.body = [producer, consumer]
        # Tag only the producer; the consumer stays shared inside the
        # loop body. The K-loop itself is not tagged -- it's a shared
        # statement that contains a tagged child.
        df.register_cute_tcgen05_tma_load_role_stmts([producer])
        body = [kloop]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        # Single shared block carrying the (mutated) K-loop.
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertIs(blocks[0].stmts[0], kloop)
        # Producer is now wrapped with the role predicate; consumer
        # stays naked.
        self.assertEqual(len(kloop.body), 2)
        wrapped = kloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")
        self.assertEqual(wrapped.body, [producer])
        self.assertIs(kloop.body[1], consumer)

    def test_role_blocks_recurse_into_top_level_while_loop(self) -> None:
        """Same recursion behavior for ``while`` as for ``for``. This
        future-proofs the partitioner in case codegen places tagged
        statements inside a top-level ``while`` (e.g. a producer
        loop driven by pipeline state rather than a static range)."""
        splitter, df = self._make_helper()
        producer = self._stmt("tma_pipeline.producer_acquire(state)")
        wloop = ast.parse("while True:\n    pass").body[0]
        wloop.body = [producer]
        df.register_cute_tcgen05_tma_load_role_stmts([producer])
        body = [wloop]
        blocks = splitter._collect_tcgen05_role_blocks(df, body)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].role_predicate)
        self.assertIs(blocks[0].stmts[0], wloop)
        self.assertEqual(len(wloop.body), 1)
        wrapped = wloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")
        self.assertEqual(wrapped.body, [producer])

    def test_role_blocks_no_recurse_into_non_loop_top_level(self) -> None:
        """Recursion is one level deep AND only into ``for`` / ``while``
        statements -- NOT into ``if`` / function bodies / etc. The
        partitioner asserts that every registered tag is consumed, so
        registering a tag inside a non-loop container fails loudly
        instead of silently dropping the role gate. This keeps the
        contract narrow (only for/while are recursed into) and surfaces
        bad registrations at compile time."""
        splitter, df = self._make_helper()
        inner = self._stmt("tma_pipeline.producer_acquire(state)")
        outer_if = self._stmt("if some_predicate:\n    pass")
        outer_if.body = [inner]
        df.register_cute_tcgen05_tma_load_role_stmts([inner])
        body = [outer_if]
        with self.assertRaises(AssertionError):
            splitter._collect_tcgen05_role_blocks(df, body)

    def test_virtual_pid_dependencies_are_traced_transitively(self) -> None:
        """The splitter walks each statement's name uses / defines and
        propagates the per-tile property: anything that reads or writes a
        per-tile name is itself per-tile, and anything it assigns is also
        per-tile. Seeded with ``virtual_pid_var``, this captures the PID
        decomposition + downstream offset chain emitted by
        ``_decompose_virtual_pid`` without requiring those sites to
        register themselves."""
        splitter, df = self._make_helper()
        # Pretend cute_mma marked just one statement explicitly per-tile.
        marked = self._stmt("g = cute.local_tile(t, (128, 16), (m, None))")
        df.register_cute_tcgen05_per_tile_stmts([marked])

        # Statements that use virtual_pid_var should fall on the
        # wrapped side; downstream statements that consume their
        # writes (pid_0, pid_1 here) should follow.
        invariant_a = self._stmt("smem_a = cute.arch.alloc_smem(F32, 256)")
        decompose_pid = self._stmt("pid_0 = __test_virtual_pid__ % blocks_0")
        decompose_pid_2 = self._stmt("pid_1 = __test_virtual_pid__ // blocks_0")
        derive_m = self._stmt("m_offset = pid_0 * BLOCK_M")
        derive_n = self._stmt("n_offset = pid_1 * BLOCK_N")
        invariant_b = self._stmt("ab_pipeline = PipelineTmaUmma.create()")

        body = [
            invariant_a,
            decompose_pid,
            decompose_pid_2,
            derive_m,
            derive_n,
            marked,
            invariant_b,
        ]
        hoisted, wrapped = splitter._split_tcgen05_invariant_setup(df, body)
        # invariant_a and invariant_b have no per-tile dep — hoisted.
        # decompose_pid uses virtual_pid → wrapped, defines pid_0.
        # decompose_pid_2 uses virtual_pid → wrapped, defines pid_1.
        # derive_m uses pid_0 → wrapped, defines m_offset.
        # derive_n uses pid_1 → wrapped, defines n_offset.
        # marked is explicitly per-tile.
        self.assertEqual(hoisted, [invariant_a, invariant_b])
        self.assertEqual(
            wrapped,
            [decompose_pid, decompose_pid_2, derive_m, derive_n, marked],
        )

    def test_partition_returns_three_views(self) -> None:
        """``_partition_tcgen05_role_blocks`` returns a structured
        ``_PartitionedRoleBody`` with three independent views of the
        same input body:

        - ``role_blocks_inline`` keeps the role-tagged statements in
          their original positions inside the linear block sequence,
          so the legacy inline-weave consumer
          (``_build_tcgen05_persistent_tile_body``) preserves the
          original emit order.
        - ``role_blocks_extracted`` carries each non-shared role
          block as a standalone unit, ready to be lifted into a
          role-local ``while``.
        - ``shared_body_extracted`` is the input body with every
          top-level tagged statement removed (the extract-and-remove
          view), so the role-local-while consumer can wire it
          directly into the shared ``while``.

        Both extracted views are independent of the inline view, so
        mutating one does not affect the other.
        """
        splitter, df = self._make_helper()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_b = self._stmt("b = 2")
        tma_load_y = self._stmt("cute.copy(t, s)")
        shared_c = self._stmt("c = 3")
        df.register_cute_tcgen05_per_tile_stmts([tma_load_x, tma_load_y])
        df.register_cute_tcgen05_tma_load_role_stmts([tma_load_x, tma_load_y])
        body = [shared_a, tma_load_x, shared_b, tma_load_y, shared_c]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # Inline view: shared_a / tma_load_x / shared_b / tma_load_y /
        # shared_c interleaved -- 5 blocks with predicate alternating.
        self.assertEqual(len(partition.role_blocks_inline), 5)
        self.assertIsNone(partition.role_blocks_inline[0].role_predicate)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [shared_a])
        self.assertEqual(
            partition.role_blocks_inline[1].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_inline[1].stmts, [tma_load_x])
        self.assertIsNone(partition.role_blocks_inline[2].role_predicate)
        self.assertEqual(partition.role_blocks_inline[2].stmts, [shared_b])
        self.assertEqual(
            partition.role_blocks_inline[3].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_inline[3].stmts, [tma_load_y])
        self.assertIsNone(partition.role_blocks_inline[4].role_predicate)
        self.assertEqual(partition.role_blocks_inline[4].stmts, [shared_c])
        # Extracted view: only the non-shared blocks, decoupled from
        # the surrounding shared statements -- two role blocks since
        # the runs are non-adjacent.
        self.assertEqual(len(partition.role_blocks_extracted), 2)
        self.assertEqual(
            partition.role_blocks_extracted[0].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_extracted[0].stmts, [tma_load_x])
        self.assertEqual(
            partition.role_blocks_extracted[1].role_predicate, "__test_tma_load_warp__"
        )
        self.assertEqual(partition.role_blocks_extracted[1].stmts, [tma_load_y])
        # Shared body: tagged statements have been removed; only the
        # surrounding shared statements remain in their original order.
        self.assertEqual(
            partition.shared_body_extracted, [shared_a, shared_b, shared_c]
        )

    def test_partition_extracted_views_are_independent_lists(self) -> None:
        """Mutating ``role_blocks_extracted`` or
        ``shared_body_extracted`` must not affect the inline view.
        This isolation is required because the role-local-while
        consumer rewrites the extracted view (e.g. wraps statements in
        an ``ast.If``); the inline view must remain usable for the
        legacy inline-weave consumer in the same kernel."""
        splitter, df = self._make_helper()
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_a = self._stmt("a = 1")
        df.register_cute_tcgen05_per_tile_stmts([tma_load_x])
        df.register_cute_tcgen05_tma_load_role_stmts([tma_load_x])
        body = [tma_load_x, shared_a]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # Mutate the extracted view's stmts list and confirm the
        # inline view's stmts list is unchanged.
        partition.role_blocks_extracted[0].stmts.append(shared_a)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [tma_load_x])
        # Mutate shared_body_extracted and confirm the inline view's
        # shared block (which carries the same stmt) is unchanged.
        partition.shared_body_extracted.append(tma_load_x)
        self.assertEqual(partition.role_blocks_inline[1].stmts, [shared_a])

    def test_partition_no_marks_returns_full_body_unchanged(self) -> None:
        """Without TMA-load marks, the partition is a degenerate one
        with the full body in a single shared block in the inline
        view, no extracted role blocks, and the full body in
        ``shared_body_extracted``. The role-local-while consumer can
        still call this safely; it just gets zero role-local whiles
        and the full body in the shared while."""
        splitter, df = self._make_helper()
        a = self._stmt("a = 1")
        b = self._stmt("b = 2")
        body = [a, b]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        self.assertEqual(len(partition.role_blocks_inline), 1)
        self.assertIsNone(partition.role_blocks_inline[0].role_predicate)
        self.assertEqual(partition.role_blocks_inline[0].stmts, body)
        self.assertEqual(partition.role_blocks_extracted, [])
        self.assertEqual(partition.shared_body_extracted, body)

    def test_partition_recursion_only_mutates_inline_view(self) -> None:
        """When tagged statements live inside top-level ``for`` /
        ``while`` loop bodies, the partitioner mutates the loop body
        in place (wrapping tagged children in
        ``if {role_predicate}:``). This shape applies ONLY to the
        legacy inline-weave consumer; the extracted view must still
        return zero extracted role blocks for the role-local-while
        consumer because the tagged statements are nested inside a
        per-tile container, not at the top level. The role-local
        producer K-loop lift (``cute_plan.md`` step 3b proper) is
        expected to move the tagged producer block to a top-level
        sibling K-loop so it lands in ``role_blocks_extracted``."""
        splitter, df = self._make_helper()
        producer = self._stmt(
            "if tma_full_tile and tma_warp:\n    tma_pipeline.producer_acquire(state)"
        )
        kloop = ast.parse("for k in range(K):\n    pass").body[0]
        kloop.body = [producer]
        df.register_cute_tcgen05_tma_load_role_stmts([producer])
        body = [kloop]
        partition = splitter._partition_tcgen05_role_blocks(df, body)
        # No top-level tagged stmts -> no extracted role blocks.
        self.assertEqual(partition.role_blocks_extracted, [])
        # Shared body holds the K-loop unchanged in identity.
        self.assertEqual(partition.shared_body_extracted, [kloop])
        # Inline view's single shared block carries the K-loop, and
        # the K-loop body has been mutated to wrap producer in an If.
        self.assertEqual(len(partition.role_blocks_inline), 1)
        self.assertEqual(partition.role_blocks_inline[0].stmts, [kloop])
        self.assertEqual(len(kloop.body), 1)
        wrapped = kloop.body[0]
        self.assertIsInstance(wrapped, ast.If)
        self.assertEqual(ast.unparse(wrapped.test), "__test_tma_load_warp__")

    def test_role_local_while_emits_scheduler_and_loop(self) -> None:
        """``_build_role_local_while`` emits a single top-level
        statement: an ``if {role_predicate}:`` whose body contains
        the role-local scheduler init + a ``while`` loop iterating
        the role's work tiles.

        The structure mirrors Quack's per-role persistent loop in
        ``gemm_sm100.py``: allocate a
        ``StaticPersistentTileScheduler``, peek at the first work
        tile, run the loop body once per tile, then advance and
        refresh. Cross-warp synchronization in the parent kernel is
        through pipeline barriers, not through ``sync_threads`` --
        intentional, because the role-local while runs on a disjoint
        warp set from the shared while."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()

        # Pre-bound role block: a single statement representing the
        # initial-prefetch IF that today lives in the TMA-load role.
        prefetch = self._stmt(
            "if tma_initial_full_tile and tma_warp:\n"
            "    tma_pipeline.producer_acquire(state)\n"
            "    cute.copy(atom, gA, sA)"
        )
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_test"
        )

        # Outer ``if {role_predicate}: ...`` wraps the whole role-
        # local while -- only the predicated warps enter.
        self.assertIsInstance(emitted, ast.If)
        self.assertEqual(ast.unparse(emitted.test), "__test_tma_load_warp__")
        # Inside the ``if``: scheduler params, scheduler create,
        # initial work tile, and the while loop. Anything else is
        # premature scaffolding and would surface as a test failure.
        self.assertGreaterEqual(len(emitted.body), 4)
        params_stmt = emitted.body[0]
        sched_stmt = emitted.body[1]
        init_tile_stmt = emitted.body[2]
        while_stmt = emitted.body[-1]
        self.assertIn("PersistentTileSchedulerParams", ast.unparse(params_stmt))
        self.assertIn("StaticPersistentTileScheduler.create", ast.unparse(sched_stmt))
        self.assertIn("initial_work_tile_info", ast.unparse(init_tile_stmt))
        self.assertIsInstance(while_stmt, ast.While)
        # While guard: ``{work_tile_var}.is_valid_tile``.
        self.assertIn("is_valid_tile", ast.unparse(while_stmt.test))
        # Body must include the role block's statement and a
        # scheduler advance + work-tile refresh.
        body_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("producer_acquire(state)", body_src)
        self.assertIn("advance_to_next_work", body_src)
        self.assertIn("get_current_work", body_src)

    def test_role_local_while_assigns_virtual_pid_var(self) -> None:
        """The role-local ``while`` body MUST bind
        ``self.virtual_pid_var`` from the role-local work-tile
        coordinates before running the role block, because the role
        block's statements may reference ``virtual_pid_var``
        transitively (e.g. through PID decomposition). Without this
        binding, the role-local while would dereference a name that
        only the shared ``while`` defines -- a compile-time error
        in the cute-DSL frontend."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_pid_test"
        )
        # Role-local while body's first statement must bind
        # virtual_pid_var so the role block's downstream references
        # see it.
        while_stmt = emitted.body[-1]
        first_in_loop = while_stmt.body[0]
        first_src = ast.unparse(first_in_loop)
        self.assertIn(splitter.virtual_pid_var, first_src)

    def test_role_local_while_rejects_shared_role_block(self) -> None:
        """A shared role block has no predicate, so a role-local
        ``while`` would have nothing to gate on. The helper asserts
        this rather than emitting a ``while`` that runs on every
        warp -- which would be incorrect because the shared body
        runs in the shared ``while``, not in any role-local one."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        shared_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate=None, stmts=[self._stmt("a = 1")]
        )
        with self.assertRaises(AssertionError):
            splitter._build_role_local_while(
                stub_df, layout, shared_block, scheduler_var_prefix="rl_shared_test"
            )

    def test_role_local_dependency_stmts_pull_pid_offset_chain(self) -> None:
        """Extracted role blocks still need tile-local names that were
        computed by the shared PID-decomposition prefix. The dependency
        helper pulls the nearest shared definitions transitively so the
        role-local loop can compute ``offset_0`` before issuing TMA
        partitions that read it."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        pid_0 = self._stmt("pid_0 = __test_virtual_pid__ % num_blocks_0")
        pid_1 = self._stmt("pid_1 = __test_virtual_pid__ // num_blocks_0")
        offset_0 = self._stmt("offset_0 = pid_0 * 128")
        offset_1 = self._stmt("offset_1 = pid_1 * 256")
        unrelated = self._stmt("acc = 0.0")
        tma_load = self._stmt(
            "gA = cute.local_tile(tensor_a, (128, 16), (offset_0, None))"
        )
        body = [pid_0, pid_1, offset_0, offset_1, unrelated, tma_load]
        stub_df.register_cute_tcgen05_per_tile_stmts([tma_load])
        stub_df.register_cute_tcgen05_tma_load_role_stmts([tma_load])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("pid_0 = __test_virtual_pid__ % num_blocks_0", loop_src)
        self.assertIn("offset_0 = pid_0 * 128", loop_src)
        self.assertNotIn("offset_1 = pid_1 * 256", loop_src)
        self.assertNotIn("acc = 0.0", loop_src)
        self.assertLess(loop_src.find("offset_0 ="), loop_src.find("cute.local_tile"))

    def test_role_local_dependency_stmts_ignore_role_internal_defs(self) -> None:
        """Role-local dependencies should include only free reads.

        The producer K-loop defines and then reads per-iteration names such as
        ``tcgen05_tma_full_tile``. Those reads are local to the extracted role
        body and must not cause the dependency scanner to pull the shared
        consumer K-loop that happens to write the same names.
        """
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=2)
        layout = self._make_minimal_layout()
        pid_0 = self._stmt("pid_0 = __test_virtual_pid__ % num_blocks_0")
        offset_0 = self._stmt("offset_0 = pid_0 * 128")
        shared_kloop = self._stmt(
            "for offset_2 in range(K):\n"
            "    tcgen05_tma_full_tile = offset_2 + 16 <= K\n"
            "    tcgen05_tma_next_full_tile = offset_2 + 32 <= K\n"
            "    tcgen05_ab_producer_try_token = shared_try_token\n"
            "    tcgen05_ab_pipeline.consumer_wait(tcgen05_ab_consumer_state)"
        )
        producer_kloop = self._stmt(
            "for offset_2 in range(K):\n"
            "    tcgen05_tma_k_tile = offset_2 // 16\n"
            "    tcgen05_tma_full_tile = offset_2 + 16 <= K\n"
            "    tcgen05_tma_next_full_tile = offset_2 + 32 <= K\n"
            "    tcgen05_ab_producer_try_token = "
            "tcgen05_ab_pipeline.producer_try_acquire(tcgen05_ab_producer_state)\n"
            "    if tcgen05_tma_full_tile and tcgen05_tma_next_full_tile "
            "and tcgen05_ab_producer_try_token:\n"
            "        tma_tile = offset_0"
        )
        body = [pid_0, offset_0, shared_kloop, producer_kloop]
        stub_df.register_cute_tcgen05_per_tile_stmts([producer_kloop])
        stub_df.register_cute_tcgen05_tma_load_role_stmts([producer_kloop])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertEqual(loop_src.count("for offset_2 in range(K):"), 1)
        self.assertIn("offset_0 = pid_0 * 128", loop_src)
        self.assertLess(loop_src.find("offset_0 ="), loop_src.find("for offset_2"))
        self.assertNotIn("consumer_wait", loop_src)

    def test_omit_shared_loop_rejects_observable_shared_stmt(self) -> None:
        """Fully role-local CtaGroup.TWO codegen omits the residual shared
        loop. If the tagged-removed shared view ever contains an observable
        operation, the omission must fail loudly instead of dropping it."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        splitter, _ = self._make_helper()
        unsafe_stmts = [
            "cute.copy(src, dst)",
            "tmp = cute.copy(src, dst)",
            "tmp = some_pipeline_call(src)",
            "tmp = min(num_pid_m, 4, key=some_callable)",
            "if side_effecting_call():\n    tmp = 1",
            "for idx in side_effecting_call():\n    tmp = idx",
        ]
        for stmt_src in unsafe_stmts:
            with self.subTest(stmt_src=stmt_src):
                partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
                    role_blocks_inline=[],
                    role_blocks_extracted=[],
                    shared_body_extracted=[self._stmt(stmt_src)],
                )

                with self.assertRaisesRegex(
                    AssertionError, "discard observable shared statement"
                ):
                    splitter._assert_tcgen05_omit_shared_loop_safe(partition)

    def test_role_local_tile_body_can_skip_shared_body_build(self) -> None:
        """When the caller omits the residual shared loop, the role-local
        builder should not construct and discard that body. Dependency-only
        scalar setup is still allowed because role-local loops clone it."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        shared_stmt = self._stmt("pid_0 = __test_virtual_pid__")
        role_stmt = self._stmt("role_value = pid_0")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[role_stmt]
        )
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[role_block],
            shared_body_extracted=[shared_stmt],
        )

        def fail_shared_builder(layout_arg: object, role_blocks_arg: object) -> None:
            self.fail("omitted shared loop should not build shared_tile_body")

        splitter._build_tcgen05_persistent_tile_body = fail_shared_builder  # type: ignore[method-assign]

        role_local_whiles, shared_tile_body = (
            splitter._build_tcgen05_persistent_tile_body_role_local(
                stub_df, layout, partition, build_shared_tile_body=False
            )
        )

        self.assertEqual(shared_tile_body, [])
        self.assertEqual(len(role_local_whiles), 1)
        while_stmt = role_local_whiles[0].body[-1]
        loop_src = "\n".join(ast.unparse(s) for s in while_stmt.body)
        self.assertIn("pid_0 = __test_virtual_pid__", loop_src)
        self.assertIn("role_value = pid_0", loop_src)

    def test_omit_shared_loop_allows_grouped_pid_scalar_setup(self) -> None:
        """Grouped PID decomposition and scalar fallback remnants are pure.

        Explicit guarded CtaGroup.TWO configs can still use grouped PID
        decomposition before the host guard raises. The omit-shared-loop
        safety check must allow these dependency-only expressions while
        still rejecting observable calls.
        """
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        splitter, _ = self._make_helper()
        partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
            role_blocks_inline=[],
            role_blocks_extracted=[],
            shared_body_extracted=[
                self._stmt("group_size_m = min(num_pid_m - first_pid_m, 4)"),
                self._stmt(
                    "for offset_2 in range(cutlass.Int32(0), "
                    "cutlass.Int32(16), cutlass.Int32(_BLOCK_SIZE_2)):\n"
                    "    indices_2 = offset_2\n"
                    "    mask_2 = indices_2 < 16\n"
                    "    load = cutlass.BFloat16(0)"
                ),
            ],
        )

        splitter._assert_tcgen05_omit_shared_loop_safe(partition)

    def test_role_local_tile_body_builder_returns_two_lists(self) -> None:
        """``_build_tcgen05_persistent_tile_body_role_local`` returns
        ``(role_local_whiles, shared_tile_body)``. The role-local
        whiles list has one entry per unique role predicate in the
        partition; the shared tile body is the per-tile body for the
        shared ``while`` (without the extracted role blocks). The
        persistent-kernel setup wires this consumer when top-level role
        blocks are extracted."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()

        shared_a = self._stmt("a = 1")
        shared_sync = self._stmt("if needs_barrier:\n    cute.arch.sync_threads()")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s)")
        shared_b = self._stmt("b = 2")
        body = [shared_a, shared_sync, tma_load_x, shared_b]
        stub_df.register_cute_tcgen05_per_tile_stmts([tma_load_x])
        stub_df.register_cute_tcgen05_tma_load_role_stmts([tma_load_x])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)
        role_local_whiles, shared_tile_body = (
            splitter._build_tcgen05_persistent_tile_body_role_local(
                stub_df, layout, partition
            )
        )
        # One role-local while per unique predicate (just one here).
        self.assertEqual(len(role_local_whiles), 1)
        rl_while = role_local_whiles[0]
        self.assertIsInstance(rl_while, ast.If)
        self.assertEqual(ast.unparse(rl_while.test), "__test_tma_load_warp__")
        # The shared tile body must NOT contain the extracted
        # tagged statement.
        shared_src = "\n".join(ast.unparse(s) for s in shared_tile_body)
        self.assertNotIn("producer_acquire(s)", shared_src)
        # The role-local intermediate keeps every warp in the shared while
        # after role-local mainloop work, so existing CTA-wide barriers must
        # be preserved rather than stripped recursively.
        self.assertIn("if needs_barrier:\n    cute.arch.sync_threads()", shared_src)
        self.assertIn("cute.arch.sync_threads()", shared_src)
        # The role-local while body should hold the tagged stmt.
        rl_src = "\n".join(ast.unparse(s) for s in rl_while.body)
        self.assertIn("producer_acquire(s)", rl_src)

    def test_role_local_tile_body_merges_extracted_blocks_by_predicate(
        self,
    ) -> None:
        """Multiple top-level TMA-load-tagged runs separated by shared
        statements partition into multiple ``role_blocks_extracted``
        entries with the same role predicate. The role-local consumer
        MUST merge them into a single role-local ``while`` so per-tile
        ordering of the role's statements is preserved -- emitting one
        loop per extracted block would run all tiles' first chunk
        before the second chunk and break AB-pipeline ordering."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        shared_a = self._stmt("a = 1")
        tma_load_x = self._stmt("tma_pipeline.producer_acquire(s_x)")
        shared_b = self._stmt("b = 2")
        tma_load_y = self._stmt("tma_pipeline.producer_acquire(s_y)")
        shared_c = self._stmt("c = 3")
        body = [shared_a, tma_load_x, shared_b, tma_load_y, shared_c]
        stub_df.register_cute_tcgen05_per_tile_stmts([tma_load_x, tma_load_y])
        stub_df.register_cute_tcgen05_tma_load_role_stmts([tma_load_x, tma_load_y])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)
        # Sanity: partitioner produced two extracted blocks.
        self.assertEqual(len(partition.role_blocks_extracted), 2)
        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )
        # Consumer merges them into one loop per unique predicate.
        self.assertEqual(len(role_local_whiles), 1)
        rl_while = role_local_whiles[0]
        loop_body_src = "\n".join(ast.unparse(s) for s in rl_while.body[-1].body)
        # Both tagged stmts must appear in the same loop body, in
        # source order (x before y).
        x_pos = loop_body_src.find("producer_acquire(s_x)")
        y_pos = loop_body_src.find("producer_acquire(s_y)")
        self.assertNotEqual(x_pos, -1)
        self.assertNotEqual(y_pos, -1)
        self.assertLess(x_pos, y_pos)

    def test_role_local_tile_body_orders_tma_before_exec_before_epi(self) -> None:
        """The exec role may be tagged earlier in source order because
        acc-fragment setup is emitted before TMA partition setup. The
        epi role may be tagged even later from the store path. The role-local
        consumer still emits TMA-load before MMA-exec before epi so producer
        work reaches the AB pipeline first and accumulator stages are published
        before epilogue warps consume them."""
        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout()
        mma_exec = self._stmt("acc_pipeline.producer_acquire(s_exec)")
        tma_load = self._stmt("tma_pipeline.producer_acquire(s_tma)")
        epi = self._stmt("acc_pipeline.consumer_wait(s_epi)")
        body = [epi, mma_exec, tma_load]
        stub_df.register_cute_tcgen05_per_tile_stmts([epi, mma_exec, tma_load])
        stub_df.register_cute_tcgen05_epi_role_stmts([epi])
        stub_df.register_cute_tcgen05_mma_exec_role_stmts([mma_exec])
        stub_df.register_cute_tcgen05_tma_load_role_stmts([tma_load])
        partition = splitter._partition_tcgen05_role_blocks(stub_df, body)

        role_local_whiles, _ = splitter._build_tcgen05_persistent_tile_body_role_local(
            stub_df, layout, partition
        )

        self.assertEqual(len(role_local_whiles), 3)
        self.assertEqual(
            ast.unparse(role_local_whiles[0].test), "__test_tma_load_warp__"
        )
        self.assertEqual(
            ast.unparse(role_local_whiles[1].test), "__test_mma_exec_warp__"
        )
        self.assertEqual(ast.unparse(role_local_whiles[2].test), "__test_epi_warp__")

    def test_role_local_while_uses_layout_cluster_m(self) -> None:
        """The role-local scheduler MUST use the same cluster shape as
        the shared scheduler so it visits tiles in the same order. The
        shared scheduler in ``_build_tcgen05_persistent_prelude`` uses
        ``(layout.cluster_m, 1, 1)``; if ``_build_role_local_while``
        hardcoded ``(1, 1, 1)`` instead, role-local and shared would
        diverge for ``cluster_m > 1`` and break AB-pipeline ordering
        between TMA-load and consumer warps."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        emitted = splitter._build_role_local_while(
            stub_df, layout, role_block, scheduler_var_prefix="rl_cluster_test"
        )
        # The PersistentTileSchedulerParams call site must reference
        # cluster_m=2 in the cluster shape tuple.
        body_srcs = [ast.unparse(stmt) for stmt in emitted.body]
        params_srcs = [
            src for src in body_srcs if "PersistentTileSchedulerParams" in src
        ]
        self.assertEqual(len(params_srcs), 1)
        self.assertIn("(2, 1, 1)", params_srcs[0])

    def test_role_local_while_skips_pdl_wait_without_two_cta(self) -> None:
        """The Quack PDL wait is paired with the two-CTA launch-dependents."""
        from helion._compiler.program_id import Tcgen05PersistentProgramIDs

        stub_df, splitter = self._make_role_local_stubs(num_pid_dims=1)
        layout = self._make_minimal_layout(cluster_m=2)
        prefetch = self._stmt("tma_pipeline.producer_acquire(state)")
        role_block = Tcgen05PersistentProgramIDs._PersistentRoleBlock(
            role_predicate="__test_tma_load_warp__", stmts=[prefetch]
        )
        with (
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_tcgen05_is_two_cta",
                return_value=False,
            ),
            patch.object(
                Tcgen05PersistentProgramIDs,
                "_tcgen05_tma_load_role_predicate",
                return_value="__test_tma_load_warp__",
            ),
        ):
            emitted = splitter._build_role_local_while(
                stub_df,
                layout,
                role_block,
                scheduler_var_prefix="rl_non_two_cta_pdl",
            )

        self.assertNotIn("cute.arch.griddepcontrol_wait()", ast.unparse(emitted))


@onlyBackends(["cute"])
class TestPerKiterTmaBuilders(unittest.TestCase):
    """AST shape tests for the per-K-iter TMA builders in ``cute_mma.py``."""

    def _scalar_load_a(self) -> ast.stmt:
        return ast.parse("if mma_active:\n    smem_a[r, c] = lhs[g]").body[0]

    def _scalar_load_b(self) -> ast.stmt:
        return ast.parse("if mma_active:\n    smem_b[r, c] = rhs[g]").body[0]

    def _make_args(
        self,
        *,
        cluster_m: int = 1,
        is_two_cta: bool = False,
        use_tma_b_mcast_mask: bool | None = None,
        use_tma_a: bool = True,
        use_tma_b: bool = True,
    ) -> _PerKiterTmaArgs:
        if use_tma_b_mcast_mask is None:
            use_tma_b_mcast_mask = cluster_m > 1 or is_two_cta
        return _PerKiterTmaArgs(
            tma_pipeline="ab_pipeline",
            tma_producer_state="ab_producer_state",
            tma_consumer_state="ab_consumer_state",
            tma_producer_try_token="ab_producer_try_token",
            tma_consumer_try_token="ab_consumer_try_token",
            tma_barrier_ptr="tma_barrier_ptr",
            tma_full_tile="full_tile",
            tma_next_full_tile="next_full_tile",
            tma_next_consumer_tile="next_consumer_tile",
            tma_warp="tma_warp",
            tma_atom_a="tma_atom_a",
            tma_atom_b="tma_atom_b",
            tma_gA="gA",
            tma_gB="gB",
            tma_sA="sA",
            tma_sB="sB",
            tma_k_tile="tma_k_tile",
            tma_a_mcast_mask="a_mcast_mask",
            tma_b_mcast_mask="b_mcast_mask",
            ab_stage_count=2,
            is_two_cta=is_two_cta,
            use_tma_b_mcast_mask=use_tma_b_mcast_mask,
            use_tma_a=use_tma_a,
            use_tma_b=use_tma_b,
            skip_producer_acquire=False,
            skip_producer_advance=False,
            skip_consumer_wait=False,
            exec_active="exec_active",
            scalar_load_a=self._scalar_load_a(),
            scalar_load_b=self._scalar_load_b(),
        )

    @staticmethod
    def _stmt_kinds(stmts: list[ast.stmt]) -> list[str]:
        kinds = []
        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute):
                    kinds.append(func.attr)
                elif isinstance(func, ast.Name):
                    kinds.append(func.id)
                else:
                    kinds.append(type(func).__name__)
            elif (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
            ):
                kinds.append(f"={stmt.value.func.attr}")
            else:
                kinds.append(type(stmt).__name__)
        return kinds

    def test_pipeline_producer_if_predicate_and_body(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_producer_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(
            ast.unparse(node.test),
            "full_tile and tma_warp and next_full_tile",
        )
        # Body must follow the prefetch protocol: try_acquire (token
        # assign), acquire, get_barrier (assign), copy A, copy B,
        # commit, advance.
        self.assertEqual(
            self._stmt_kinds(node.body),
            [
                "=producer_try_acquire",
                "producer_acquire",
                "=producer_get_barrier",
                "copy",
                "copy",
                "producer_commit",
                "advance",
            ],
        )
        # Both cute.copy calls index the prefetch slot
        # (tma_k_tile + ab_stage_count == 2).
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("tma_k_tile + cutlass.Int32(2)", body_src)

    def test_pipeline_producer_if_can_drop_inline_tma_warp_gate(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_producer_if(args, gate_tma_warp=False)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile and next_full_tile")
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn("tma_warp", body_src)

    def test_pipeline_producer_two_cta_keeps_per_cta_copies(
        self,
    ) -> None:
        args = self._make_args(cluster_m=2, is_two_cta=True)
        node = _build_kloop_pipeline_producer_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(
            ast.unparse(node.test),
            "full_tile and tma_warp and next_full_tile",
        )
        self.assertEqual(
            self._stmt_kinds(node.body),
            [
                "=producer_try_acquire",
                "producer_acquire",
                "=producer_get_barrier",
                "copy",
                "copy",
                "producer_commit",
                "advance",
            ],
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn(_TCGEN05_CLUSTER_LEADER_PREDICATE, body_src)

    def test_non_pipeline_producer_if_can_drop_inline_tma_warp_gate(self) -> None:
        args = self._make_args()
        node = _build_kloop_non_pipeline_producer_if(args, gate_tma_warp=False)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn("tma_warp", body_src)

    def test_pipeline_consumer_if_predicate_and_body(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_consumer_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        # Full-tile branch is one inner ``if exec_active:`` with
        # consumer_try_wait + consumer_wait.
        self.assertEqual(len(node.body), 1)
        inner = node.body[0]
        self.assertIsInstance(inner, ast.If)
        self.assertEqual(ast.unparse(inner.test), "exec_active")
        self.assertEqual(
            self._stmt_kinds(inner.body),
            ["=consumer_try_wait", "consumer_wait"],
        )
        # Partial-tile fallback is scalar loads + sync_threads.
        orelse_src = ast.unparse(ast.Module(body=node.orelse, type_ignores=[]))
        self.assertIn("smem_a", orelse_src)
        self.assertIn("smem_b", orelse_src)
        self.assertIn("cute.arch.sync_threads", orelse_src)

    def test_pipeline_consumer_if_can_drop_exec_gate_and_fallback(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_consumer_if(
            args, gate_exec_warp=False, include_scalar_fallback=False
        )
        self.assertIsInstance(node, ast.If)
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("consumer_try_wait", body_src)
        self.assertIn("consumer_wait", body_src)
        self.assertNotIn("exec_active", body_src)
        self.assertEqual(node.orelse, [])

    def test_pipeline_consumer_if_can_reuse_prefetched_try_token(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_consumer_if(
            args,
            gate_exec_warp=False,
            include_scalar_fallback=False,
            use_existing_try_token=True,
        )

        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn("consumer_try_wait", body_src)
        self.assertIn(
            "ab_pipeline.consumer_wait(ab_consumer_state, ab_consumer_try_token)",
            body_src,
        )

    def test_pipeline_consumer_prefetch_two_cta_gates_try_wait_to_leader(
        self,
    ) -> None:
        args = self._make_args(cluster_m=2, is_two_cta=True)
        stmts = _build_kloop_pipeline_consumer_prefetch_stmts(
            args, gate_exec_warp=False
        )

        self.assertEqual(self._stmt_kinds(stmts), ["=Boolean", "If"])
        peek = stmts[1]
        self.assertIsInstance(peek, ast.If)
        self.assertEqual(
            ast.unparse(peek.test),
            f"next_consumer_tile and {_TCGEN05_CLUSTER_LEADER_PREDICATE}",
        )
        self.assertEqual(self._stmt_kinds(peek.body), ["=consumer_try_wait"])

    def test_pipeline_consumer_prefetch_requires_two_cta(self) -> None:
        args = self._make_args()
        with self.assertRaisesRegex(AssertionError, "CtaGroup.TWO"):
            _build_kloop_pipeline_consumer_prefetch_stmts(args)

    def test_pipeline_consumer_two_cta_gates_wait_to_leader(self) -> None:
        args = self._make_args(cluster_m=2, is_two_cta=True)
        node = _build_kloop_pipeline_consumer_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        self.assertEqual(len(node.body), 1)
        inner = node.body[0]
        self.assertIsInstance(inner, ast.If)
        self.assertEqual(
            ast.unparse(inner.test),
            f"exec_active and {_TCGEN05_CLUSTER_LEADER_PREDICATE}",
        )
        self.assertEqual(
            self._stmt_kinds(inner.body),
            ["=consumer_try_wait", "consumer_wait"],
        )

    def test_pipeline_release_if_predicate_and_body(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_release_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        self.assertEqual(len(node.body), 1)
        inner = node.body[0]
        self.assertIsInstance(inner, ast.If)
        self.assertEqual(ast.unparse(inner.test), "exec_active")
        # Producer-state advance lives in the producer block, not here.
        self.assertEqual(
            self._stmt_kinds(inner.body),
            ["consumer_release", "advance"],
        )
        body_src = ast.unparse(ast.Module(body=inner.body, type_ignores=[]))
        self.assertIn("ab_consumer_state.advance", body_src)
        self.assertNotIn("ab_producer_state.advance", body_src)

    def test_pipeline_release_if_can_drop_exec_gate_and_fallback(self) -> None:
        args = self._make_args()
        node = _build_kloop_pipeline_release_if(
            args, gate_exec_warp=False, include_scalar_fallback=False
        )
        self.assertIsInstance(node, ast.If)
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("consumer_release", body_src)
        self.assertIn("ab_consumer_state.advance", body_src)
        self.assertNotIn("exec_active", body_src)
        self.assertEqual(node.orelse, [])

    def test_pipeline_release_two_cta_releases_from_leader_and_advances_both(
        self,
    ) -> None:
        args = self._make_args(cluster_m=2, is_two_cta=True)
        node = _build_kloop_pipeline_release_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        self.assertEqual(self._stmt_kinds(node.body), ["If", "If"])
        release_gate, advance_gate = node.body
        self.assertIsInstance(release_gate, ast.If)
        self.assertEqual(
            ast.unparse(release_gate.test),
            f"exec_active and {_TCGEN05_CLUSTER_LEADER_PREDICATE}",
        )
        self.assertEqual(self._stmt_kinds(release_gate.body), ["consumer_release"])
        self.assertIsInstance(advance_gate, ast.If)
        self.assertEqual(ast.unparse(advance_gate.test), "exec_active")
        self.assertEqual(self._stmt_kinds(advance_gate.body), ["advance"])

    def test_tcgen05_mma_issue_two_cta_gates_gemm_to_leader(self) -> None:
        node = _build_tcgen05_mma_issue_stmt(
            exec_active="exec_active",
            tiled_mma="tiled_mma",
            acc_frag="acc_frag",
            tcgen05_frag_a="tCrA",
            tcgen05_frag_b="tCrB",
            mma_stage="mma_stage",
            is_two_cta=True,
        )
        self.assertIsInstance(node, ast.If)
        self.assertEqual(
            ast.unparse(node.test),
            f"exec_active and {_TCGEN05_CLUSTER_LEADER_PREDICATE}",
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("cute.gemm", body_src)
        self.assertNotIn("offset_2 != cutlass.Int32(0)", body_src)
        self.assertNotIn("Field.ACCUMULATE, offset_2", body_src)
        self.assertIn(
            "tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)",
            body_src,
        )

    def test_tcgen05_mma_accumulate_reset_two_cta_gates_to_leader(self) -> None:
        node = _build_tcgen05_mma_accumulate_reset_stmt(
            "exec_active", tiled_mma="tiled_mma", is_two_cta=True
        )

        self.assertIsInstance(node, ast.If)
        self.assertEqual(
            ast.unparse(node.test),
            f"exec_active and {_TCGEN05_CLUSTER_LEADER_PREDICATE}",
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn(
            "tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)",
            body_src,
        )

    def test_non_pipeline_release_advances_both_states(self) -> None:
        args = self._make_args()
        node = _build_kloop_non_pipeline_release_if(args)
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "full_tile")
        # Single-stage: CTA sync, then exec-warp release, then BOTH
        # producer + consumer state advance.
        self.assertEqual(
            self._stmt_kinds(node.body),
            ["sync_threads", "If", "advance", "advance"],
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("ab_producer_state.advance", body_src)
        self.assertIn("ab_consumer_state.advance", body_src)
        self.assertIn("consumer_release", body_src)

    def test_producer_skips_copy_when_operand_not_tma_loaded(self) -> None:
        # The pipelined producer asserts both flags True, so this only
        # exercises the non-pipelined branch.
        args_a_only = self._make_args(use_tma_a=True, use_tma_b=False)
        body_src_a_only = ast.unparse(
            ast.Module(
                body=_build_kloop_non_pipeline_producer_if(args_a_only).body,
                type_ignores=[],
            )
        )
        self.assertIn("cute.copy(tma_atom_a", body_src_a_only)
        self.assertNotIn("cute.copy(tma_atom_b", body_src_a_only)

        args_b_only = self._make_args(use_tma_a=False, use_tma_b=True)
        body_src_b_only = ast.unparse(
            ast.Module(
                body=_build_kloop_non_pipeline_producer_if(args_b_only).body,
                type_ignores=[],
            )
        )
        self.assertNotIn("cute.copy(tma_atom_a", body_src_b_only)
        self.assertIn("cute.copy(tma_atom_b", body_src_b_only)

    def test_mcast_mask_asymmetry_between_a_and_b(self) -> None:
        # A only multicasts in 2-CTA mode; B multicasts on cluster_m>1
        # OR 2-CTA. Pin the asymmetry on both producer builders.
        for builder in (
            _build_kloop_pipeline_producer_if,
            _build_kloop_non_pipeline_producer_if,
        ):
            single = ast.unparse(
                ast.Module(
                    body=builder(self._make_args(cluster_m=1, is_two_cta=False)).body,
                    type_ignores=[],
                )
            )
            self.assertNotIn("a_mcast_mask", single)
            self.assertNotIn("b_mcast_mask", single)

            clustered = ast.unparse(
                ast.Module(
                    body=builder(self._make_args(cluster_m=2, is_two_cta=False)).body,
                    type_ignores=[],
                )
            )
            self.assertNotIn("a_mcast_mask", clustered)
            self.assertIn("b_mcast_mask", clustered)

            two_cta = ast.unparse(
                ast.Module(
                    body=builder(self._make_args(cluster_m=2, is_two_cta=True)).body,
                    type_ignores=[],
                )
            )
            self.assertIn("a_mcast_mask", two_cta)
            self.assertIn("b_mcast_mask", two_cta)


@onlyBackends(["cute"])
class TestInitialPrefetchTmaBuilder(unittest.TestCase):
    """AST shape tests for ``_build_initial_prefetch_if`` in ``cute_mma.py``.

    Two call sites use this builder: the stage-0 prefetch
    (``full_tile_gates=[tma_initial_full_tile]``) and -- only when
    ``ab_stage_count > 1`` -- the stage-(N-1) prefetch which also
    AND-gates on ``tma_initial_next_full_tile``. The builder appends
    ``args.tma_warp`` as the trailing gate. Tests pin the predicate
    shape, the body, and the same A/B mcast asymmetry that the per-K-iter
    builders enforce.
    """

    def _make_args(
        self,
        *,
        cluster_m: int = 1,
        is_two_cta: bool = False,
        use_tma_b_mcast_mask: bool | None = None,
    ) -> _InitialPrefetchTmaArgs:
        if use_tma_b_mcast_mask is None:
            use_tma_b_mcast_mask = cluster_m > 1 or is_two_cta
        return _InitialPrefetchTmaArgs(
            tma_pipeline="ab_pipeline",
            tma_producer_state="ab_producer_state",
            tma_barrier_ptr="tma_barrier_ptr",
            tma_warp="tma_warp",
            tma_atom_a="tma_atom_a",
            tma_atom_b="tma_atom_b",
            tma_gA="gA",
            tma_gB="gB",
            tma_sA="sA",
            tma_sB="sB",
            tma_a_mcast_mask="a_mcast_mask",
            tma_b_mcast_mask="b_mcast_mask",
            is_two_cta=is_two_cta,
            use_tma_b_mcast_mask=use_tma_b_mcast_mask,
            skip_producer_acquire=False,
            skip_producer_advance=False,
        )

    @staticmethod
    def _stmt_kinds(stmts: list[ast.stmt]) -> list[str]:
        kinds = []
        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute):
                    kinds.append(func.attr)
                elif isinstance(func, ast.Name):
                    kinds.append(func.id)
                else:
                    kinds.append(type(func).__name__)
            elif (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
            ):
                kinds.append(f"={stmt.value.func.attr}")
            else:
                kinds.append(type(stmt).__name__)
        return kinds

    def test_stage0_predicate_and_body(self) -> None:
        """Stage-0 prefetch: predicate is ``initial_full_tile and
        tma_warp``; body is acquire / get_barrier / copy A / copy B /
        commit / advance with ``k_offset = cutlass.Int32(0)``. No
        try-token (initial prefetch always takes the slow path).
        """
        args = self._make_args()
        node = _build_initial_prefetch_if(
            args,
            full_tile_gates=["initial_full_tile"],
            k_offset="cutlass.Int32(0)",
        )
        self.assertIsInstance(node, ast.If)
        self.assertEqual(
            ast.unparse(node.test),
            "initial_full_tile and tma_warp",
        )
        self.assertEqual(
            self._stmt_kinds(node.body),
            [
                "producer_acquire",
                "=producer_get_barrier",
                "copy",
                "copy",
                "producer_commit",
                "advance",
            ],
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("cutlass.Int32(0)", body_src)
        # No try-token in the prefetch path.
        self.assertNotIn("producer_try_acquire", body_src)
        self.assertNotIn("ab_producer_try_token", body_src)

    def test_stage_n_minus_one_predicate_extends_with_next_full_tile(self) -> None:
        """Stage-(N-1) prefetch (when ``ab_stage_count > 1``): caller
        AND-gates on ``initial_next_full_tile`` via ``full_tile_gates``
        and supplies ``cutlass.Int32(stage_count - 1)`` as ``k_offset``.
        The builder appends ``args.tma_warp`` as the trailing gate.
        """
        args = self._make_args()
        node = _build_initial_prefetch_if(
            args,
            full_tile_gates=["initial_full_tile", "initial_next_full_tile"],
            k_offset="cutlass.Int32(1)",
        )
        self.assertEqual(
            ast.unparse(node.test),
            "initial_full_tile and initial_next_full_tile and tma_warp",
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertIn("cutlass.Int32(1)", body_src)
        # The non-stage-0 prefetch reads from gA[None, Int32(stage-1)] /
        # gB[None, Int32(stage-1)], not Int32(0).
        self.assertNotIn("None, cutlass.Int32(0)", body_src)

    def test_stage0_can_override_producer_acquire(self) -> None:
        """Diagnostic call sites can skip a single initial acquire."""

        node = _build_initial_prefetch_if(
            self._make_args(),
            full_tile_gates=["initial_full_tile"],
            k_offset="cutlass.Int32(0)",
            skip_producer_acquire=True,
        )
        self.assertEqual(
            self._stmt_kinds(node.body),
            [
                "=producer_get_barrier",
                "copy",
                "copy",
                "producer_commit",
                "advance",
            ],
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn("producer_acquire", body_src)
        self.assertIn("producer_get_barrier", body_src)

    def test_two_cta_prefetch_keeps_per_cta_copies(self) -> None:
        args = self._make_args(cluster_m=2, is_two_cta=True)
        node = _build_initial_prefetch_if(
            args,
            full_tile_gates=["initial_full_tile"],
            k_offset="cutlass.Int32(0)",
        )
        self.assertIsInstance(node, ast.If)
        self.assertEqual(ast.unparse(node.test), "initial_full_tile and tma_warp")
        self.assertEqual(
            self._stmt_kinds(node.body),
            [
                "producer_acquire",
                "=producer_get_barrier",
                "copy",
                "copy",
                "producer_commit",
                "advance",
            ],
        )
        body_src = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        self.assertNotIn(_TCGEN05_CLUSTER_LEADER_PREDICATE, body_src)

    def test_mcast_mask_asymmetry_matches_per_kiter(self) -> None:
        """A only multicasts in 2-CTA mode; B multicasts on
        ``cluster_m > 1`` OR 2-CTA. Same asymmetry as the per-K-iter
        builders -- the prefetch must read from the same gA/gB layouts
        with the same multicast modes, otherwise stages 0 and N-1 of
        the pipeline land in different SMEM slices than the K-loop
        producer expects.
        """
        single = ast.unparse(
            ast.Module(
                body=_build_initial_prefetch_if(
                    self._make_args(cluster_m=1, is_two_cta=False),
                    full_tile_gates=["full_tile"],
                    k_offset="cutlass.Int32(0)",
                ).body,
                type_ignores=[],
            )
        )
        self.assertNotIn("a_mcast_mask", single)
        self.assertNotIn("b_mcast_mask", single)

        clustered = ast.unparse(
            ast.Module(
                body=_build_initial_prefetch_if(
                    self._make_args(cluster_m=2, is_two_cta=False),
                    full_tile_gates=["full_tile"],
                    k_offset="cutlass.Int32(0)",
                ).body,
                type_ignores=[],
            )
        )
        self.assertNotIn("a_mcast_mask", clustered)
        self.assertIn("b_mcast_mask", clustered)

        two_cta = ast.unparse(
            ast.Module(
                body=_build_initial_prefetch_if(
                    self._make_args(cluster_m=2, is_two_cta=True),
                    full_tile_gates=["full_tile"],
                    k_offset="cutlass.Int32(0)",
                ).body,
                type_ignores=[],
            )
        )
        self.assertIn("a_mcast_mask", two_cta)
        self.assertIn("b_mcast_mask", two_cta)


@onlyBackends(["cute"])
class TestReductionLoopCarriedAccumulatorCheck(unittest.TestCase):
    """Unit tests for ``_needs_loop_carried_accumulator``.

    The helper consolidates the three cute-specific "no live thread axis"
    conditions in :meth:`BlockReductionStrategy.codegen_reduction`. Each
    condition individually returns the same outcome -- ``expr =
    input_name`` -- so the helper short-circuits the disjunction.

    Tests instantiate ``BlockReductionStrategy`` via ``__new__`` to skip
    the device-function plumbing (the helper only reads ``self._codegen``
    and ``self.block_ids``).
    """

    def _make_strategy(
        self,
        *,
        block_index: int = 0,
        active_device_loops: dict[int, list[object]] | None = None,
        current_grid_state: object | None = None,
    ) -> BlockReductionStrategy:
        strategy = BlockReductionStrategy.__new__(BlockReductionStrategy)
        strategy.block_ids = [block_index]
        strategy._codegen = SimpleNamespace(
            active_device_loops=active_device_loops or {},
            current_grid_state=current_grid_state,
        )
        return strategy

    def _make_loop_state(
        self,
        block_id: int,
        *,
        threaded: bool,
    ) -> DeviceLoopState:
        block_thread_axes: dict[int, int] = {block_id: 0} if threaded else {}
        return DeviceLoopState(
            strategy=_FakeLoopStrategy([block_id]),
            block_id_to_info={},
            for_node=ast.For(
                target=ast.Name(id=f"i_{block_id}", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=1)],
                    keywords=[],
                ),
                body=[],
                orelse=[],
                type_comment=None,
            ),
            inner_statements=[],
            block_thread_axes=block_thread_axes,
        )

    def _make_grid_state(
        self,
        *,
        block_thread_axes: dict[int, int] | None = None,
        lane_loop_blocks: set[int] | None = None,
    ) -> DeviceGridState:
        grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
            block_thread_axes=block_thread_axes or {},
        )
        if lane_loop_blocks:
            for block_id in lane_loop_blocks:
                grid.add_lane_loop(block_id, f"synthetic_lane_{block_id}", 4)
        return grid

    def _patch_backend(self, name: str) -> contextlib.AbstractContextManager:
        max_threads = 32 if name == "cute" else None
        env = SimpleNamespace(
            backend=SimpleNamespace(
                name=name, max_reduction_threads=lambda: max_threads
            )
        )
        return patch.object(CompileEnvironment, "current", return_value=env)

    def test_returns_false_when_block_has_live_thread_axis_in_grid(self) -> None:
        # The reduction block is mapped to a thread axis in the current
        # grid; no loop-carried accumulator is needed.
        grid = self._make_grid_state(block_thread_axes={0: 0})
        strategy = self._make_strategy(current_grid_state=grid)
        with self._patch_backend("cute"):
            self.assertFalse(strategy._needs_loop_carried_accumulator())

    def test_returns_false_when_block_has_live_thread_axis_in_loop(self) -> None:
        loop = self._make_loop_state(0, threaded=True)
        strategy = self._make_strategy(active_device_loops={0: [loop]})
        with self._patch_backend("cute"):
            self.assertFalse(strategy._needs_loop_carried_accumulator())

    def test_returns_true_when_block_serial_loop_no_thread(self) -> None:
        # A serial DeviceLoopState exists for the reduction block but no
        # thread axis -- the surrounding loop must accumulate.
        loop = self._make_loop_state(0, threaded=False)
        strategy = self._make_strategy(active_device_loops={0: [loop]})
        with self._patch_backend("cute"):
            self.assertTrue(strategy._needs_loop_carried_accumulator())

    def test_returns_true_when_block_has_lane_loops(self) -> None:
        # A lane loop iterates the reduction block; the synthetic per-thread
        # iteration is not backed by real threads.
        grid = self._make_grid_state(lane_loop_blocks={0})
        strategy = self._make_strategy(current_grid_state=grid)
        with self._patch_backend("cute"):
            self.assertTrue(strategy._needs_loop_carried_accumulator())

    def test_returns_true_when_no_active_loop_or_grid(self) -> None:
        # No live thread axis at all -> let the surrounding loop
        # accumulate.
        strategy = self._make_strategy()
        with self._patch_backend("cute"):
            self.assertTrue(strategy._needs_loop_carried_accumulator())

    def test_returns_false_for_non_cute_backend(self) -> None:
        # The helper short-circuits to False on backends other than cute,
        # whose native warp / lane reductions handle the reduction.
        strategy = self._make_strategy()
        for backend_name in ("triton", "pallas", "tileir"):
            with self._patch_backend(backend_name):
                self.assertFalse(
                    strategy._needs_loop_carried_accumulator(),
                    f"backend {backend_name!r} should not need loop-carried accumulator",
                )

    def test_returns_false_when_lane_loop_block_differs(self) -> None:
        # Lane loop iterates a different block than the reduction block;
        # the reduction block still needs a thread axis check.
        grid = self._make_grid_state(
            block_thread_axes={0: 0},
            lane_loop_blocks={1},
        )
        strategy = self._make_strategy(current_grid_state=grid)
        with self._patch_backend("cute"):
            self.assertFalse(strategy._needs_loop_carried_accumulator())

    def test_returns_true_when_lane_loop_overrides_thread_axis(self) -> None:
        # When the reduction block has BOTH a lane loop AND a thread axis
        # (from a different active loop state), the lane-loop check fires
        # first and the helper returns True. Lane-looped blocks are not
        # safe to reduce across via warp lanes regardless of whether
        # another active loop also maps a thread axis to the same block.
        grid = self._make_grid_state(
            block_thread_axes={0: 0},
            lane_loop_blocks={0},
        )
        strategy = self._make_strategy(current_grid_state=grid)
        with self._patch_backend("cute"):
            self.assertTrue(strategy._needs_loop_carried_accumulator())

    def test_returns_true_when_serial_loop_overrides_global_thread_axis(self) -> None:
        # When the reduction block is iterated by a serial DeviceLoopState
        # locally but has a thread axis in some OTHER active device loop
        # (e.g., the same block id is reused across nested scopes), the
        # local serial check still wins and the helper returns True. The
        # surrounding serial loop must own the accumulator.
        #
        # This is an internal-ordering pin: the constructed state
        # (same block id appearing as both a serial loop under its own
        # key and a threaded loop under an unrelated key) is unlikely
        # in real code paths but locks in the disjunction's ordering.
        local_loop = self._make_loop_state(0, threaded=False)
        other_loop = self._make_loop_state(0, threaded=True)
        # The local serial check looks at active_device_loops[block_index]
        # (key 0) and sees only the serial loop, while the global
        # has-live-thread-axis walk iterates every value list and also
        # picks up the threaded loop under key 9.
        strategy = self._make_strategy(
            active_device_loops={0: [local_loop], 9: [other_loop]},
        )
        with self._patch_backend("cute"):
            self.assertTrue(strategy._needs_loop_carried_accumulator())

    def test_block_index_property_drives_check(self) -> None:
        # The helper consults ``self.block_index`` (== block_ids[0])
        # rather than a per-call argument; constructing the strategy
        # with a different block_index flips the answer when only that
        # block has a thread axis.
        grid = self._make_grid_state(block_thread_axes={5: 0})
        with self._patch_backend("cute"):
            # block_index=5: thread axis present -> False
            strategy_with_axis = self._make_strategy(
                block_index=5, current_grid_state=grid
            )
            self.assertFalse(strategy_with_axis._needs_loop_carried_accumulator())
            # block_index=7: thread axis absent for this block -> True
            strategy_without_axis = self._make_strategy(
                block_index=7, current_grid_state=grid
            )
            self.assertTrue(strategy_without_axis._needs_loop_carried_accumulator())


@onlyBackends(["cute"])
class TestReductionBlockClassifiers(unittest.TestCase):
    """Direct unit tests for the three "block classifier" helpers
    (``_reduction_block_is_serial`` / ``_reduction_block_has_lane_loops`` /
    ``_reduction_block_has_live_thread_axis``) on ``ReductionStrategy``.

    The consolidated ``_needs_loop_carried_accumulator``
    helper is OR-combined from these three predicates plus a backend
    guard. These tests pin each predicate individually so future changes
    to the OR composition can be validated against the building blocks.
    """

    def _make_strategy(
        self,
        *,
        block_index: int = 0,
        active_device_loops: dict[int, list[object]] | None = None,
        current_grid_state: object | None = None,
    ) -> BlockReductionStrategy:
        strategy = BlockReductionStrategy.__new__(BlockReductionStrategy)
        strategy.block_ids = [block_index]
        strategy._codegen = SimpleNamespace(
            active_device_loops=active_device_loops or {},
            current_grid_state=current_grid_state,
        )
        return strategy

    def _serial_loop(self, block_id: int) -> DeviceLoopState:
        return DeviceLoopState(
            strategy=_FakeLoopStrategy([block_id]),
            block_id_to_info={},
            for_node=ast.For(
                target=ast.Name(id=f"i_{block_id}", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=1)],
                    keywords=[],
                ),
                body=[],
                orelse=[],
                type_comment=None,
            ),
            inner_statements=[],
            block_thread_axes={},
        )

    def _threaded_loop(self, block_id: int) -> DeviceLoopState:
        loop = self._serial_loop(block_id)
        loop.block_thread_axes[block_id] = 0
        return loop

    def test_block_is_serial_true_for_unthreaded_loop(self) -> None:
        strategy = self._make_strategy(active_device_loops={0: [self._serial_loop(0)]})
        self.assertTrue(strategy._reduction_block_is_serial())

    def test_block_is_serial_false_when_loop_threads_axis(self) -> None:
        strategy = self._make_strategy(
            active_device_loops={0: [self._threaded_loop(0)]}
        )
        self.assertFalse(strategy._reduction_block_is_serial())

    def test_block_is_serial_false_when_no_loop(self) -> None:
        strategy = self._make_strategy()
        self.assertFalse(strategy._reduction_block_is_serial())

    def test_block_is_serial_false_when_codegen_unset(self) -> None:
        strategy = BlockReductionStrategy.__new__(BlockReductionStrategy)
        strategy.block_ids = [0]
        # No ``_codegen`` attribute at all -> defensive False.
        self.assertFalse(strategy._reduction_block_is_serial())

    def test_block_has_lane_loops_in_current_grid(self) -> None:
        grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        grid.add_lane_loop(0, "synthetic_lane_0", 4)
        strategy = self._make_strategy(current_grid_state=grid)
        self.assertTrue(strategy._reduction_block_has_lane_loops())

    def test_block_has_lane_loops_in_other_active_loop(self) -> None:
        # Another active grid state holds a lane loop for the reduction
        # block; the predicate walks all active loops.
        other_grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        other_grid.add_lane_loop(0, "synthetic_lane_0", 4)
        strategy = self._make_strategy(
            active_device_loops={9: [other_grid]},
        )
        self.assertTrue(strategy._reduction_block_has_lane_loops())

    def test_block_has_lane_loops_false_for_other_block(self) -> None:
        grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        grid.add_lane_loop(1, "synthetic_lane_1", 4)
        strategy = self._make_strategy(current_grid_state=grid)
        self.assertFalse(strategy._reduction_block_has_lane_loops())

    def test_block_has_lane_loops_false_when_grid_has_no_lane_loops(self) -> None:
        grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
        )
        strategy = self._make_strategy(current_grid_state=grid)
        self.assertFalse(strategy._reduction_block_has_lane_loops())

    def test_block_has_live_thread_axis_in_current_grid(self) -> None:
        grid = DeviceGridState(
            strategy=_FakeLoopStrategy([0]),
            block_id_to_info={},
            block_thread_axes={0: 0},
        )
        strategy = self._make_strategy(current_grid_state=grid)
        self.assertTrue(strategy._reduction_block_has_live_thread_axis())

    def test_block_has_live_thread_axis_in_active_loop(self) -> None:
        strategy = self._make_strategy(
            active_device_loops={0: [self._threaded_loop(0)]}
        )
        self.assertTrue(strategy._reduction_block_has_live_thread_axis())

    def test_block_has_live_thread_axis_in_other_active_loop(self) -> None:
        # The reduction block id appears as a thread axis in some OTHER
        # active loop entry (not the one keyed by the same block id);
        # the predicate's third walk-step covers this.
        strategy = self._make_strategy(
            active_device_loops={9: [self._threaded_loop(0)]},
        )
        self.assertTrue(strategy._reduction_block_has_live_thread_axis())

    def test_block_has_live_thread_axis_false_when_no_axis(self) -> None:
        strategy = self._make_strategy()
        self.assertFalse(strategy._reduction_block_has_live_thread_axis())

    def test_block_has_live_thread_axis_false_when_only_serial_loop(self) -> None:
        strategy = self._make_strategy(active_device_loops={0: [self._serial_loop(0)]})
        self.assertFalse(strategy._reduction_block_has_live_thread_axis())


if __name__ == "__main__":
    unittest.main()
