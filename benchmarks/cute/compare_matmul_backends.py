"""Compare Helion matmul backends against Quack-direct and ATen on B200+ hardware.

The default '--impl all' mode benchmarks the apples-to-apples gate comparison:
    aten ↔ quack-direct exact baseline ↔ helion-triton ↔ helion-cute

CRITICAL: 'quack-direct' calls ``quack.gemm.gemm`` directly. The 'quack' impl
uses ``quack.gemm_interface.gemm`` which wraps the kernel in
``torch.library.custom_op`` for graph-capture compatibility, adding ~30 µs of
overhead per call (~27 % of measured TFLOP/s at sub-100 µs kernel times).
For gate comparisons always use quack-direct, and ensure it is an exact fused
implementation of the full target operation before treating it as the Quack
baseline. Use quack only to see public-API perf as a torch.compile user would.
See ``perf_gap_findings.md``.

Each impl runs in a fresh subprocess (so HELION_BACKEND env mutation does not
leak), with steady-state methodology (2 s thermal warmup, do_bench
warmup=1 s + rep=500 ms, 5 runs), and reports best plus mom-median ms/TFLOP/s.
The gate metric is mom-median TFLOP/s: the median of the five do_bench medians.
Best-of-N is diagnostic only.

Examples:

    # Default comparison
    python benchmarks/cute/compare_matmul_backends.py --m 4096 --n 4096 --k 4096

    # Helion with a fixed config (skip autotune)
    python benchmarks/cute/compare_matmul_backends.py --impl helion-cute \\
        --skip-correctness 1 --helion-force-config 1 \\
        --helion-tile-m 128 --helion-tile-n 256 --helion-tile-k 128 \\
        --helion-pid-type flat --helion-num-epi-warps 4

    # Helion CtaGroup.TWO seed-family diagnostic with explicit indexing/range flags
    python benchmarks/cute/compare_matmul_backends.py --impl helion-cute \\
        --helion-force-config 1 \\
        --helion-tile-m 256 --helion-tile-n 256 --helion-tile-k 128 \\
        --helion-pid-type persistent_interleaved --helion-cluster-m 2 \\
        --helion-num-sm-multiplier 1 \\
        --helion-indexing tensor_descriptor,tensor_descriptor,tensor_descriptor \\
        --helion-range-flattens none,none \\
        --helion-range-multi-buffers none,none \\
        --helion-range-warp-specializes none,none \\
        --helion-num-epi-warps 4 --helion-require-tcgen05 1

    # Quack-direct with a non-default tile shape
    python benchmarks/cute/compare_matmul_backends.py --impl quack-direct \\
        --quack-tile-m 128 --quack-cluster-m 1

    # Quack-direct with the brief gate tuning candidate set
    python benchmarks/cute/compare_matmul_backends.py --impl quack-direct \\
        --quack-tune brief

    # Source/codegen marker comparison for the validated CtaGroup.TWO row
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_matmul_backends.py \\
        --helion-two-cta-codegen-report

    # Nsight Compute scheduler metrics plus SASS source attribution
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_matmul_backends.py \\
        --helion-two-cta-ncu-report --ncu-source-report
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import json
import math
from operator import itemgetter
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NamedTuple

import torch

import helion
from helion._compiler.cute.strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODES
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODES
from helion._compiler.cute.tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_NORMAL
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from helion._compiler.cute.tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE

REPO_ROOT = Path(__file__).resolve().parents[2]
QUACK_PATH = REPO_ROOT / "quack"
_QUACK_CODEGEN_SOURCE_RELATIVE_PATHS = (
    Path("quack") / "gemm_sm100.py",
    Path("quack") / "gemm_base.py",
    Path("quack") / "pipeline.py",
    Path("quack") / "tile_scheduler.py",
)

DEFAULT_IMPLS = ("aten", "quack-direct", "helion-triton", "helion-cute")
ALL_IMPLS = ("aten", "quack", "quack-direct", "helion-cute", "helion-triton")
MATMUL_EPILOGUES = (
    "none",
    "bias",
    "relu",
    "bias_relu",
    "bias_residual_gelu",
)
QUACK_TUNE_CHOICES = ("off", "brief")
# Brief tuning covers the documented default, larger cluster/swizzle variants,
# and one-CTA / N-skinny tile alternatives without turning the gate sweep into
# a full Quack autotuner.
_QUACK_BRIEF_TUNE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 1,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": True,
        "max_swizzle_size": 8,
    },
    {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 2,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": True,
        "max_swizzle_size": 8,
    },
    {
        "tile_m": 256,
        "tile_n": 256,
        "cluster_m": 2,
        "cluster_n": 1,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": True,
        "max_swizzle_size": 16,
    },
    {
        "tile_m": 128,
        "tile_n": 256,
        "cluster_m": 1,
        "cluster_n": 1,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": True,
        "max_swizzle_size": 8,
    },
    {
        "tile_m": 256,
        "tile_n": 128,
        "cluster_m": 2,
        "cluster_n": 1,
        "pingpong": False,
        "persistent": True,
        "is_dynamic_persistent": True,
        "max_swizzle_size": 8,
    },
)

_TCGEN05_CONFIG_KEYS = (
    "tcgen05_cluster_m",
    "tcgen05_ab_stages",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
    TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
    "tcgen05_num_epi_warps",
)
_HELION_INDEXING_CHOICES = ("pointer", "block_ptr", "tensor_descriptor")
_HELION_DEFAULT_RANGE_FLATTENS = (None, True)
_HELION_DEFAULT_RANGE_MULTI_BUFFERS = (None, False)
_HELION_DEFAULT_RANGE_WARP_SPECIALIZES = (None, True)
_HELION_DEFAULT_RANGE_NUM_STAGES = (0, 0)
_HELION_DEFAULT_RANGE_UNROLL_FACTORS = (0, 0)

_TENSOR_DESCRIPTOR_INDEXING = (
    "tensor_descriptor",
    "tensor_descriptor",
    "tensor_descriptor",
)
_TWO_CTA_DIAGNOSTIC_BASE = {
    "helion_force_config": 1,
    "helion_tile_m": TCGEN05_TWO_CTA_BLOCK_M,
    "helion_tile_n": TCGEN05_TWO_CTA_BLOCK_N,
    "helion_tile_k": 128,
    "helion_loop_order": [0, 1],
    "helion_num_stages": 1,
    "helion_l2_grouping": TCGEN05_TWO_CTA_SEED_L2_GROUPING,
    "helion_num_sm_multiplier": 1,
    "helion_indexing": list(_TENSOR_DESCRIPTOR_INDEXING),
    "helion_pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
    "helion_cluster_m": 2,
    "helion_ab_stages": 2,
    "helion_acc_stages": 2,
    "helion_c_stages": 2,
    "helion_num_epi_warps": 4,
    # ``helion_strategy`` intentionally NOT included here: the diagnostic
    # variants use the user's ``--helion-strategy`` CLI value (if any) so the
    # whole sweep / NCU / codegen report runs against the requested strategy.
    # Add per-variant overrides to ``_TWO_CTA_DIAGNOSTIC_VARIANTS`` if a
    # specific diagnostic ever needs to pin a strategy regardless of the CLI.
    "helion_require_tcgen05": 1,
    "helion_range_flattens": [None, None],
    "helion_range_multi_buffers": [None, None],
    "helion_range_warp_specializes": [None, None],
    "helion_range_num_stages": [0, 0],
    "helion_range_unroll_factors": [0, 0],
    "helion_acc_producer_mode": TCGEN05_ACC_PRODUCER_MODE_NORMAL,
    "helion_acc_wait_placement": TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    "helion_c_acquire_placement": TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    "helion_c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
    "helion_epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
}
_TWO_CTA_DIAGNOSTIC_VARIANTS = (
    ("seed", {"helion_num_warps": 4, "helion_maxnreg": None}),
    ("seed_maxnreg128", {"helion_num_warps": 4, "helion_maxnreg": 128}),
    ("warps16", {"helion_num_warps": 16, "helion_maxnreg": None}),
    (
        "quick_selected_family",
        {
            "helion_num_warps": 16,
            "helion_maxnreg": 128,
            "helion_indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
            ],
        },
    ),
)
_SOURCE_MARKERS = {
    "cta_group_two": ("CtaGroup.TWO",),
    "pipeline_tma_umma": ("PipelineTmaUmma",),
    "pipeline_tma_store": ("PipelineTmaStore",),
    "producer_acquire": (".producer_acquire(",),
    "producer_commit": (".producer_commit(",),
    "producer_tail": (".producer_tail(",),
    "consumer_release": (".consumer_release(",),
    "persistent_scheduler": ("StaticPersistentTileScheduler", "TileSchedulerCls"),
    "work_tile_loop": ("while tcgen05_role_local_", "while work_tile.is_valid_tile"),
    "epilogue_barrier_wait": (
        "tcgen05_epilog_sync_barrier.arrive_and_wait",
        "epilogue_barrier.arrive_and_wait",
    ),
    "async_shared_fence": (
        "fence_view_async_shared",
        "fence_proxy('async.shared'",
        'fence_proxy("async.shared"',
    ),
    "pipeline_init_arrive": ("pipeline_init_arrive",),
    "pipeline_init_wait": ("pipeline_init_wait",),
    "tmem_allocator": ("TmemAllocator(",),
    "tmem_relinquish": (".relinquish_alloc_permit(",),
    "tmem_free": (".free(",),
    "griddepcontrol_wait": ("griddepcontrol_wait(",),
    "griddepcontrol_launch_dependents": ("griddepcontrol_launch_dependents(",),
    "setmaxregister": ("setmaxregister_",),
}
_EPILOGUE_BARRIER_BEFORE_SMEM_MARKER = "epilogue_barrier_before_smem_wait"
_EPILOGUE_BARRIER_BEFORE_TMA_MARKER = "epilogue_barrier_before_tma_wait"
_EPILOGUE_BARRIER_CONTEXT_MARKERS = (
    _EPILOGUE_BARRIER_BEFORE_SMEM_MARKER,
    _EPILOGUE_BARRIER_BEFORE_TMA_MARKER,
)
_SOURCE_MARKER_LABELS = (*_SOURCE_MARKERS, *_EPILOGUE_BARRIER_CONTEXT_MARKERS)
_SOURCE_MARKER_LOCATION_LIMIT = 4
_EPILOGUE_BARRIER_PREVIOUS_CONTEXT_LINES = 5
_EPILOGUE_BARRIER_FOLLOWING_CONTEXT_LINES = 3
_HELION_ROLE_LOCAL_MARKER_TRACE_LIMIT = 8
_PIPELINE_EVENT_TRACE_LIMIT = 16
_PIPELINE_COPY_CONTEXT_LINES = 4
_PIPELINE_COPY_ROLE_EVENTS = (
    "tma_load_copy",
    "tmem_to_reg_copy",
    "reg_to_smem_copy",
    "tma_store_copy",
    "generic_copy",
)
_PIPELINE_COPY_EVENT_NEEDLES = (
    ("tma_store_copy", ("tcgen05_tma_store_atom",)),
    ("reg_to_smem_copy", ("tcgen05_tiled_copy_r2s", "tiled_copy_r2s")),
    ("tmem_to_reg_copy", ("tcgen05_tiled_copy_t2r", "tiled_copy_t2r")),
    (
        "tma_load_copy",
        (
            "tcgen05_tma_atom_a",
            "tcgen05_tma_atom_b",
            "tma_atom_a",
            "tma_atom_b",
            "copy_tensor",
        ),
    ),
)
_PIPELINE_EVENT_PATTERNS = (
    ("pdl_wait", ("griddepcontrol_wait(",)),
    ("pdl_launch_dependents", ("griddepcontrol_launch_dependents(",)),
    ("scheduler_create", ("StaticPersistentTileScheduler.create(", "TileSchedulerCls")),
    ("work_tile_loop", ("while tcgen05_role_local_", "while work_tile.is_valid_tile")),
    (
        "ab_producer_acquire",
        ("tcgen05_ab_pipeline.producer_acquire", "ab_pipeline.producer_acquire"),
    ),
    (
        "ab_producer_commit",
        ("tcgen05_ab_pipeline.producer_commit", "ab_pipeline.producer_commit"),
    ),
    (
        "ab_producer_tail",
        ("tcgen05_ab_pipeline.producer_tail", "ab_pipeline.producer_tail"),
    ),
    (
        "ab_consumer_wait",
        ("tcgen05_ab_pipeline.consumer_wait", "ab_pipeline.consumer_wait"),
    ),
    (
        "ab_consumer_release",
        ("tcgen05_ab_pipeline.consumer_release", "ab_pipeline.consumer_release"),
    ),
    (
        "async_shared_fence",
        (
            "cute.arch.fence_view_async_shared",
            "cute.arch.fence_proxy('async.shared'",
            'cute.arch.fence_proxy("async.shared"',
        ),
    ),
    ("umma_gemm", ("cute.gemm(",)),
    (
        "acc_producer_acquire",
        ("tcgen05_acc_pipeline.producer_acquire", "acc_pipeline.producer_acquire"),
    ),
    (
        "acc_producer_commit",
        ("tcgen05_acc_pipeline.producer_commit", "acc_pipeline.producer_commit"),
    ),
    (
        "acc_producer_tail",
        ("tcgen05_acc_pipeline.producer_tail", "acc_pipeline.producer_tail"),
    ),
    (
        "acc_consumer_wait",
        ("tcgen05_acc_pipeline.consumer_wait", "acc_pipeline.consumer_wait"),
    ),
    (
        "acc_consumer_release",
        ("tcgen05_acc_pipeline.consumer_release", "acc_pipeline.consumer_release"),
    ),
    (
        "c_producer_acquire",
        (
            "tcgen05_c_pipeline.producer_acquire",
            "epi_pipeline.producer_acquire",
            "epi_store_pipeline.producer_acquire",
        ),
    ),
    (
        "c_producer_commit",
        (
            "tcgen05_c_pipeline.producer_commit",
            "epi_pipeline.producer_commit",
            "epi_store_pipeline.producer_commit",
        ),
    ),
    (
        "c_producer_tail",
        (
            "tcgen05_c_pipeline.producer_tail",
            "epi_pipeline.producer_tail",
            "epi_store_pipeline.producer_tail",
        ),
    ),
)
_PIPELINE_GAP_PAIRS = (
    ("ab_acquire_to_tma_load", "ab_producer_acquire", "tma_load_copy"),
    ("tma_load_to_ab_commit", "tma_load_copy", "ab_producer_commit"),
    ("ab_commit_to_ab_wait", "ab_producer_commit", "ab_consumer_wait"),
    ("ab_wait_to_umma", "ab_consumer_wait", "umma_gemm"),
    ("umma_to_acc_commit", "umma_gemm", "acc_producer_commit"),
    ("acc_commit_to_acc_wait", "acc_producer_commit", "acc_consumer_wait"),
    ("c_acquire_to_acc_wait", "c_producer_acquire", "acc_consumer_wait"),
    ("c_acquire_to_tma_store", "c_producer_acquire", "tma_store_copy"),
    ("r2s_to_tma_store", "reg_to_smem_copy", "tma_store_copy"),
)
_PIPELINE_WINDOW_LIMIT = 8
_EPILOGUE_STORE_WINDOW_NAME = "epilogue_store_issue"
_PIPELINE_WINDOW_SPECS = (
    (
        "tma_load_issue",
        ("ab_producer_acquire", "tma_load_copy", "ab_producer_commit"),
    ),
    ("umma_issue", ("ab_consumer_wait", "umma_gemm", "acc_producer_commit")),
    (
        _EPILOGUE_STORE_WINDOW_NAME,
        (
            "c_producer_acquire",
            "acc_consumer_wait",
            "tmem_to_reg_copy",
            "reg_to_smem_copy",
            "async_shared_fence",
            "tma_store_copy",
            "c_producer_commit",
        ),
    ),
)
_PIPELINE_EVENT_ALIASES = {
    # Historical reports used one broad "tma_copy" event for every copy-like
    # line. Keep it as a query alias while the trace now separates load, T2R,
    # R2S, and store issue sites.
    "tma_copy": _PIPELINE_COPY_ROLE_EVENTS,
}


def _pipeline_event_count_labels() -> tuple[str, ...]:
    labels: list[str] = []
    for label, _needles in _PIPELINE_EVENT_PATTERNS:
        labels.append(label)
        if label == "ab_consumer_release":
            labels.extend(_PIPELINE_COPY_ROLE_EVENTS)
    return tuple(labels)


_PIPELINE_EVENT_COUNT_LABELS = _pipeline_event_count_labels()
_WAIT_SITE_EVENT_ALIASES = {
    "producer_acquire": (
        "ab_producer_acquire",
        "acc_producer_acquire",
        "c_producer_acquire",
    ),
    "producer_commit": (
        "ab_producer_commit",
        "acc_producer_commit",
        "c_producer_commit",
    ),
    "producer_tail": ("ab_producer_tail", "acc_producer_tail", "c_producer_tail"),
    "consumer_wait": ("ab_consumer_wait", "acc_consumer_wait"),
    **_PIPELINE_EVENT_ALIASES,
}
_WAIT_SITE_MARKER_ALIASES = {
    "arrive_and_wait": (
        "epilogue_barrier_wait",
        *_EPILOGUE_BARRIER_CONTEXT_MARKERS,
        "pipeline_init_wait",
    ),
    "epilogue_barrier_wait": ("epilogue_barrier_wait",),
    _EPILOGUE_BARRIER_BEFORE_SMEM_MARKER: (_EPILOGUE_BARRIER_BEFORE_SMEM_MARKER,),
    _EPILOGUE_BARRIER_BEFORE_TMA_MARKER: (_EPILOGUE_BARRIER_BEFORE_TMA_MARKER,),
}
_SOURCE_SYMBOL_TRACE_LIMIT = 8
_SOURCE_SYMBOL_MARKER_TRACE_LIMIT = 6
_SOURCE_CONTEXT_TRACE_LIMIT = 64
_SOURCE_CONTEXT_EVENT_TRACE_LIMIT = 6
_SOURCE_CONTEXT_MARKER_TRACE_LIMIT = 4
_SOURCE_CONTEXT_WINDOW_LIMIT = 2
_SOURCE_CONTEXT_PRINT_CANDIDATES = 3
_SOURCE_CONTEXT_PRINT_EVENT_LINES = 3
_SOURCE_CONTEXT_PRINT_MARKER_LINES = 3
_SOURCE_CONTEXT_PRINT_WINDOWS = 2
_SOURCE_CONTEXT_PRINT_EVENT_GROUPS = 4
_SOURCE_CONTEXT_PRINT_EVENT_SITES = 4
_SOURCE_CONTEXT_CANDIDATE_LIMIT = 4
_EPILOGUE_TOKEN_EVENTS = (
    "c_producer_acquire",
    "acc_consumer_wait",
    "tmem_to_reg_copy",
    "reg_to_smem_copy",
    "async_shared_fence",
    "tma_store_copy",
    "c_producer_commit",
)
_EPILOGUE_TOKEN_QUERY_EVENTS = (
    *_EPILOGUE_TOKEN_EVENTS,
    "epilogue_barrier_wait",
    *_EPILOGUE_BARRIER_CONTEXT_MARKERS,
)
_EPILOGUE_TOKEN_CONTEXT_ANCHOR_EVENTS = tuple(
    event for event in _EPILOGUE_TOKEN_EVENTS if event != "async_shared_fence"
)
_EPILOGUE_TOKEN_CONTEXT_LIMIT = 6
_EPILOGUE_TOKEN_PRINT_CONTEXTS = 4
_EPILOGUE_TOKEN_PRINT_SITES = 4
_EPILOGUE_SUBREGION_RULES = (
    (
        "scheduler_work_fetch",
        frozenset({"scheduler_create", "work_tile_loop"}),
        frozenset(),
    ),
    (
        "acc_wait_t2r",
        frozenset({"acc_consumer_wait", "tmem_to_reg_copy", "acc_consumer_release"}),
        frozenset(),
    ),
    (
        "c_store_barrier_tma_tail",
        frozenset(
            {
                "reg_to_smem_copy",
                "async_shared_fence",
                "tma_store_copy",
                "c_producer_commit",
            }
        ),
        frozenset(("epilogue_barrier_wait", *_EPILOGUE_BARRIER_CONTEXT_MARKERS)),
    ),
)
_TWO_CTA_NCU_DEFAULT_METRICS = (
    "sm__cycles_elapsed.avg",
    "smsp__cycles_elapsed.avg",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
)
_TWO_CTA_NCU_SOURCE_METRICS = (
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_membar",
    "smsp__pcsamp_warps_issue_stalled_selected",
)
_TWO_CTA_NCU_SOURCE_PROVENANCE_METRICS = ("inst_executed",)
_TWO_CTA_NCU_SOURCE_TOP_LIMIT = 8
_TWO_CTA_NCU_WAIT_SITE_CONTEXT_ROWS = 2
_TWO_CTA_NCU_WAIT_SITE_FOCUS_ROWS = 8
_TWO_CTA_NCU_WAIT_SITE_BRANCH_ROWS = 32
_TWO_CTA_NCU_WAIT_SITE_BRANCH_PRINT_LIMIT = 2
_TWO_CTA_NCU_WAIT_SITE_SASS_PRINT_LIMIT = 3
_TWO_CTA_NCU_WAIT_SITE_LOOP_SASS_PRINT_LIMIT = 5
_TWO_CTA_NCU_WAIT_SITE_SYNC_PRINT_LIMIT = 5
_TWO_CTA_NCU_WAIT_SITE_RELATED_PRINT_LIMIT = 2
_TWO_CTA_NCU_WAIT_TOPOLOGY_PRINT_LIMIT = 4
_TWO_CTA_NCU_SYNC_KEY_TOPOLOGY_PRINT_LIMIT = 4
_TWO_CTA_NCU_SYNC_KEY_SITE_PRINT_LIMIT = 3
_TWO_CTA_NCU_SASS_TOPOLOGY_PRINT_LIMIT = 4
_TWO_CTA_NCU_SASS_TOPOLOGY_CONTEXT_LIMIT = 3
_TWO_CTA_NCU_SASS_TOPOLOGY_BRANCH_EDGE_LIMIT = 4
_TWO_CTA_NCU_SYNC_PROVENANCE_PRINT_LIMIT = 3
_TWO_CTA_NCU_SYNC_PROVENANCE_ROW_PRINT_LIMIT = 3
_TWO_CTA_NCU_SYNC_PROVENANCE_SCAN_ROWS = 64
_TWO_CTA_CUBIN_SYNC_METRIC = "cubin_sync_occurrences"
_TWO_CTA_CUBIN_SYNC_STATE_PRINT_LIMIT = 12
_PIPELINE_EVENT_SITE_LOOKBACK_LINES = 3
_HELION_FIRST_SUBTILE_C_ACQUIRE_SITE = "first_subtile_c_pre_acquire"
_HELION_FIRST_SUBTILE_C_LOOP_ACQUIRE_SITE = "first_subtile_c_loop_acquire"
_HELION_LATER_SUBTILE_C_ACQUIRE_SITE = "later_subtile_c_acquire"
_HELION_LATER_SUBTILE_C_BEFORE_BARRIER_ACQUIRE_SITE = (
    "later_subtile_c_before_barrier_acquire"
)
_HELION_FIRST_SUBTILE_ACC_WAIT_SITE = "first_subtile_acc_wait"
_HELION_FIRST_SUBTILE_ACC_BEFORE_LOOP_WAIT_SITE = "first_subtile_acc_before_loop_wait"
_TWO_CTA_NCU_WAIT_SITE_BRANCH_OVERRIDABLE_EVIDENCE = frozenset(
    {"pre_wait_async_shared_fence"}
)
# SASS instruction addresses advance by 8 bytes on the profiled targets.
_TWO_CTA_NCU_SASS_INSTRUCTION_SIZE = 0x8
_TWO_CTA_NCU_WAIT_SITE_BRANCH_ADDRESS_GAP = (
    _TWO_CTA_NCU_WAIT_SITE_BRANCH_ROWS * _TWO_CTA_NCU_SASS_INSTRUCTION_SIZE
)
# Keep a wait-site group to a small local SASS window, not an entire loop body.
_TWO_CTA_NCU_WAIT_SITE_ADDRESS_GAP = 0x40
_HELION_TWO_CTA_NCU_TARGET_CONFIGS = (
    "two_cta_seed",
    "two_cta_edge_k_tail_monolithic",
    "two_cta_edge_k_tail_scheduler_split",
    "two_cta_acc_stages1",
    "two_cta_first_c_acquire_in_loop",
    "two_cta_later_c_acquire_before_barrier",
    "two_cta_acc_wait_before_subtile_loop",
    "two_cta_split_first_t2r",
    "two_cta_split_acc_t2r_store_tail",
    "two_cta_module_helper_acc_t2r",
    "two_cta_module_helper_store_tail",
    "two_cta_skip_epilogue_store",
    "two_cta_skip_umma",
    "cluster_m1_flat",
)
_QUACK_TWO_CTA_NCU_TARGET_CONFIGS = ("quack_same_tile", "quack_best")
_QUACK_TWO_CTA_NCU_BASE = {
    "quack_pingpong": 0,
    "quack_persistent": 1,
    "quack_dynamic_persistent": 1,
    "quack_max_swizzle_size": 8,
}
_TWO_CTA_NCU_TARGET_CONFIGS = (
    *_HELION_TWO_CTA_NCU_TARGET_CONFIGS,
    *_QUACK_TWO_CTA_NCU_TARGET_CONFIGS,
)
# Matches SASS integer/uniform registers (R<n>, UR<n>, RZ, URZ); predicates are
# intentionally outside this diagnostic slice.
_SASS_REGISTER_RE = re.compile(r"\b(?:UR|R)(?:Z|\d+)\b")
_SASS_REGISTER_DESTINATION_RE = re.compile(r"((?:UR|R)(?:Z|\d+))(?:\.[A-Za-z0-9_]+)*")
_SASS_ZERO_REGISTERS = frozenset({"RZ", "URZ"})
_HELION_CUTE_KERNEL_SYMBOL = "_helion_matmul"
_NVDISASM_TEXT_SECTION_RE = re.compile(
    r"^//-+\s+(?P<section>\.text(?:\.[^\s]+)?)\s+-+$"
)
_NVDISASM_INSTRUCTION_RE = re.compile(
    r"^\s*/\*(?P<address>[0-9a-fA-F]+)\*/\s+(?P<source>.*?)\s*$"
)
_NVDISASM_LINE_INFO_RE = re.compile(
    r'^\s*//## File "(?P<file>.+)", line (?P<line>\d+)\s*$'
)


@dataclass
class _PreparedHelion:
    backend: str
    dtype: torch.dtype
    kernel_args: tuple[Any, ...]
    expected: torch.Tensor
    bound: Any
    active_config: helion.Config
    code: str
    codegen: dict[str, bool]


@dataclass
class _PreparedQuackDirect:
    dtype: torch.dtype
    expected: torch.Tensor
    run: Callable[[], torch.Tensor]
    config: dict[str, Any]


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1e9)


def _parse_optional_bool_list(value: str) -> list[bool | None]:
    result: list[bool | None] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item in ("none", "null", "-"):
            result.append(None)
        elif item in ("1", "true", "t", "yes", "y"):
            result.append(True)
        elif item in ("0", "false", "f", "no", "n"):
            result.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated booleans/none, got {value!r}"
            )
    if len(result) != 2:
        raise argparse.ArgumentTypeError(
            f"expected exactly two comma-separated values, got {value!r}"
        )
    return result


def _format_optional_bool_list(values: list[bool | None]) -> str:
    return ",".join("none" if value is None else str(value).lower() for value in values)


def _parse_int_list(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {value!r}"
        ) from exc
    if len(result) != 2:
        raise argparse.ArgumentTypeError(
            f"expected exactly two comma-separated values, got {value!r}"
        )
    return result


def _format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _parse_loop_order(value: str) -> list[int]:
    result = _parse_int_list(value)
    if sorted(result) != [0, 1]:
        raise argparse.ArgumentTypeError(
            f"expected loop order permutation 0,1 or 1,0, got {value!r}"
        )
    return result


def _parse_optional_int(value: str) -> int | None:
    item = value.strip().lower()
    if item in ("auto", "none", "null", "-"):
        return None
    try:
        parsed = int(item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected positive integer or auto/none/null/-, got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"expected stage count >= 1 or auto/none/null/-, got {value!r}"
        )
    return parsed


def _format_optional_int(value: int | None) -> str:
    return "auto" if value is None else str(value)


def _parse_indexing_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",")]
    invalid = [item for item in result if item not in _HELION_INDEXING_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "expected comma-separated indexing values from "
            f"{_HELION_INDEXING_CHOICES}, got {value!r}"
        )
    return result


def _parse_csv_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def _make_inputs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype) / math.sqrt(k)
    return a, b


def _make_epilogue_inputs(
    args: argparse.Namespace, dtype: torch.dtype
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    bias = None
    residual = None
    if args.epilogue in ("bias", "bias_relu", "bias_residual_gelu"):
        bias = torch.randn((args.n,), device="cuda", dtype=dtype)
    if args.epilogue == "bias_residual_gelu":
        residual = torch.randn((args.m, args.n), device="cuda", dtype=dtype)
    return bias, residual


def _make_matmul_problem(
    args: argparse.Namespace,
) -> tuple[
    torch.dtype, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
]:
    dtype = _dtype_from_name(args.dtype)
    a, b = _make_inputs(args.m, args.n, args.k, dtype, seed=args.seed)
    bias, residual = _make_epilogue_inputs(args, dtype)
    return dtype, a, b, bias, residual


def _apply_epilogue(
    args: argparse.Namespace,
    acc: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if args.epilogue == "none":
        return acc
    if args.epilogue == "bias":
        assert bias is not None
        return acc + bias
    if args.epilogue == "relu":
        return torch.relu(acc)
    if args.epilogue == "bias_relu":
        assert bias is not None
        return torch.relu(acc + bias)
    if args.epilogue == "bias_residual_gelu":
        assert bias is not None
        assert residual is not None
        val = 1.25 * acc.float() + 0.5 * residual.float() + bias.float()
        return torch.nn.functional.gelu(val).to(dtype)
    raise AssertionError(f"unhandled epilogue {args.epilogue!r}")


def _matmul_expected(
    args: argparse.Namespace,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _apply_epilogue(args, a @ b, bias, residual, dtype)


def _check_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        # bf16/fp16 GEMMs accumulate enough rounding noise that benchmark
        # smoke tests need a looser threshold than unit tests.
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=5e-1, rtol=5e-2
        )


def _gpu_warmup(duration_ms: int = 2000) -> None:
    """Drive the GPU to a stable clock state with sustained matmul work.

    Without this, the first benchmark run after process startup gets
    artificially good numbers because the GPU has not been clock-cycled
    into its sustained-load state yet.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000.0
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(50):
            a = a @ a
        torch.cuda.synchronize()


def _bench_steady(
    fn: Callable[[], object],
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    cache_warmup_calls: int = 5,
    thermal_warmup_ms: int = 2000,
) -> dict[str, Any]:
    """Steady-state benchmark.

    1. Cache warmup: call fn() a few times to populate per-launch caches
       (cute.compile, Triton autotune cache).
    2. Thermal warmup: drive the GPU to stable clock state with continuous
       matmul work (separate from the timed kernel).
    3. Measurement: ``num_runs`` of triton's do_bench(warmup, rep), each run
       reports median over rep window. Returns best/mom-median/mean across
       runs; mom-median is the gate metric.
    """
    from triton.testing import do_bench

    for _ in range(cache_warmup_calls):
        fn()
    torch.cuda.synchronize()

    _gpu_warmup(thermal_warmup_ms)

    runs: list[float] = []
    for _ in range(num_runs):
        ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms)
        assert isinstance(ms, float)
        runs.append(ms)

    return {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "mean_ms": sum(runs) / len(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs_ms": runs,
    }


def _result(
    impl: str,
    args: argparse.Namespace,
    stats: dict[str, Any],
    *,
    config: object,
    codegen: dict[str, bool] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "impl": impl,
        "shape": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "epilogue": args.epilogue,
        },
        "best_ms": stats["best_ms"],
        "median_ms": stats["median_ms"],
        "mom_median_ms": stats["median_ms"],
        "mean_ms": stats["mean_ms"],
        "std_ms": stats["std_ms"],
        "runs_ms": stats["runs_ms"],
        "best_tflops": _tflops(args.m, args.n, args.k, stats["best_ms"]),
        "median_tflops": _tflops(args.m, args.n, args.k, stats["median_ms"]),
        "mom_median_tflops": _tflops(args.m, args.n, args.k, stats["median_ms"]),
    }
    if config is not None:
        payload["config"] = config
    if codegen is not None:
        payload["codegen"] = codegen
    return payload


def _benchmark_aten(args: argparse.Namespace) -> dict[str, Any]:
    dtype, a, b, bias, residual = _make_matmul_problem(args)
    fn = lambda: _apply_epilogue(args, a @ b, bias, residual, dtype)  # noqa: E731
    stats = _bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result("aten", args, stats, config=None)


def _benchmark_quack(args: argparse.Namespace) -> dict[str, Any]:
    """Quack via gemm_interface.gemm — torch.library wrapped.

    Includes ~30 µs of per-call torch.library overhead. For kernel-perf
    comparison use quack-direct instead.
    """
    quack_path = args.quack_path or str(QUACK_PATH)
    sys.path.insert(0, quack_path)
    from quack.gemm_interface import gemm  # pyrefly: ignore [missing-import]

    if args.epilogue not in ("none", "bias"):
        raise SystemExit(
            "quack wrapper impl currently supports --epilogue none,bias only; "
            "use quack-direct for fused activation epilogues."
        )
    dtype, a, b, bias, residual = _make_matmul_problem(args)
    expected = _matmul_expected(args, a, b, bias, residual, dtype)
    if not args.skip_correctness:
        _check_close(gemm(a, b, bias=bias, out_dtype=dtype), expected, dtype)
    fn = lambda: gemm(a, b, bias=bias, out_dtype=dtype)  # noqa: E731
    stats = _bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result("quack", args, stats, config=None)


def _quack_direct_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tile_m": args.quack_tile_m,
        "tile_n": args.quack_tile_n,
        "cluster_m": args.quack_cluster_m,
        "cluster_n": args.quack_cluster_n,
        "pingpong": bool(args.quack_pingpong),
        "persistent": bool(args.quack_persistent),
        "is_dynamic_persistent": bool(args.quack_dynamic_persistent),
        "max_swizzle_size": args.quack_max_swizzle_size,
    }


def _quack_direct_config_candidates(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], ...]:
    requested = _quack_direct_config_from_args(args)
    if args.quack_tune == "off":
        return (requested,)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for config in (requested, *_QUACK_BRIEF_TUNE_CONFIGS):
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dict(config))
    return tuple(candidates)


def _prepare_quack_direct(
    args: argparse.Namespace, config: dict[str, Any]
) -> _PreparedQuackDirect:
    quack_path = args.quack_path or str(QUACK_PATH)
    sys.path.insert(0, quack_path)
    from quack.gemm import gemm as gemm_dispatch  # pyrefly: ignore [missing-import]

    dtype, a, b, bias, residual = _make_matmul_problem(args)
    expected = _matmul_expected(args, a, b, bias, residual, dtype)
    out = torch.empty((1, args.m, args.n), dtype=dtype, device="cuda")
    a_d = a.unsqueeze(0)
    b_d = b.mT.unsqueeze(0)
    bias_d = bias.unsqueeze(0) if bias is not None else None
    residual_d = residual.unsqueeze(0) if residual is not None else None

    if args.epilogue in ("relu", "bias_relu", "bias_residual_gelu"):
        from quack.gemm_act import gemm_act  # pyrefly: ignore [missing-import]

        def fn() -> torch.Tensor:
            c_d = residual_d if args.epilogue == "bias_residual_gelu" else None
            activation = "gelu" if args.epilogue == "bias_residual_gelu" else "relu"
            gemm_act(
                a_d,
                b_d,
                None,
                c_d,
                out,
                None,
                activation,
                config["tile_m"],
                config["tile_n"],
                config["cluster_m"],
                config["cluster_n"],
                pingpong=config["pingpong"],
                persistent=config["persistent"],
                is_dynamic_persistent=config["is_dynamic_persistent"],
                max_swizzle_size=config["max_swizzle_size"],
                rowvec_bias=bias_d,
                alpha=1.25 if args.epilogue == "bias_residual_gelu" else 1.0,
                beta=0.5 if args.epilogue == "bias_residual_gelu" else 1.0,
            )
            return out[0]

    else:

        def fn() -> torch.Tensor:
            gemm_dispatch(
                a_d,
                b_d,
                out,
                None,
                None,
                config["tile_m"],
                config["tile_n"],
                config["cluster_m"],
                config["cluster_n"],
                pingpong=config["pingpong"],
                persistent=config["persistent"],
                is_dynamic_persistent=config["is_dynamic_persistent"],
                max_swizzle_size=config["max_swizzle_size"],
                rowvec_bias=bias_d,
            )
            return out[0]

    return _PreparedQuackDirect(dtype=dtype, expected=expected, run=fn, config=config)


def _benchmark_quack_direct(args: argparse.Namespace) -> dict[str, Any]:
    """Quack via gemm.gemm — direct kernel call, no torch.library overhead."""
    tune_records: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for config in _quack_direct_config_candidates(args):
        prepared = _prepare_quack_direct(args, config)
        if not args.skip_correctness:
            _check_close(prepared.run(), prepared.expected, prepared.dtype)

        stats = _bench_steady(
            prepared.run,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        result = _result("quack-direct", args, stats, config=prepared.config)
        tune_records.append(
            {
                "config": prepared.config,
                "mom_median_ms": result["mom_median_ms"],
                "mom_median_tflops": result["mom_median_tflops"],
            }
        )
        if (
            best_result is None
            or result["mom_median_ms"] < best_result["mom_median_ms"]
        ):
            best_result = result

    assert best_result is not None
    if args.quack_tune != "off":
        best_result["quack_tune"] = args.quack_tune
        best_result["quack_tuning"] = tune_records
    return best_result


def _make_helion_config_from_args(args: argparse.Namespace) -> helion.Config:
    """Build a helion.Config from CLI args.

    Includes the range_* fields with the values the matmul autotune currently
    converges on for B200/tcgen05. Without these, fixed-config runs hit
    correctness issues (the K-loop needs warp specialization for tcgen05).
    """
    config: dict[str, Any] = {
        "block_sizes": [args.helion_tile_m, args.helion_tile_n, args.helion_tile_k],
        "l2_groupings": [args.helion_l2_grouping],
        "loop_orders": [args.helion_loop_order],
        "num_stages": args.helion_num_stages,
        "num_warps": args.helion_num_warps,
        "pid_type": args.helion_pid_type,
        # K-loop hints matching autotune's matmul defaults on B200/tcgen05.
        "range_warp_specializes": args.helion_range_warp_specializes,
        "range_flattens": args.helion_range_flattens,
        "range_multi_buffers": args.helion_range_multi_buffers,
        "range_num_stages": args.helion_range_num_stages,
        "range_unroll_factors": args.helion_range_unroll_factors,
        "tcgen05_cluster_m": args.helion_cluster_m,
        "tcgen05_cluster_n": args.helion_cluster_n,
        "tcgen05_num_epi_warps": args.helion_num_epi_warps,
    }
    if args.helion_ab_stages is not None:
        config["tcgen05_ab_stages"] = args.helion_ab_stages
    if args.helion_acc_stages is not None:
        config["tcgen05_acc_stages"] = args.helion_acc_stages
    if args.helion_c_stages is not None:
        config["tcgen05_c_stages"] = args.helion_c_stages
    if args.helion_l2_swizzle_size is not None:
        config["tcgen05_l2_swizzle_size"] = args.helion_l2_swizzle_size
    if args.helion_indexing is not None:
        config["indexing"] = args.helion_indexing
    if args.helion_maxnreg is not None:
        config["maxnreg"] = args.helion_maxnreg
    if args.helion_num_sm_multiplier != 1:
        config["num_sm_multiplier"] = args.helion_num_sm_multiplier
    if args.helion_strategy is not None:
        config["tcgen05_strategy"] = args.helion_strategy
        if args.helion_strategy == "role_local_with_scheduler":
            config["tcgen05_warp_spec_scheduler_warps"] = 1
    if args.helion_c_input_warps is not None:
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = args.helion_c_input_warps
    if args.helion_persistence_model is not None:
        config["tcgen05_persistence_model"] = args.helion_persistence_model
    if args.helion_acc_producer_mode != TCGEN05_ACC_PRODUCER_MODE_NORMAL:
        config[TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY] = args.helion_acc_producer_mode
        config[TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY] = True
    if args.helion_acc_wait_placement != TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP:
        config[TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY] = args.helion_acc_wait_placement
    if args.helion_c_acquire_placement != TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP:
        config[TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY] = args.helion_c_acquire_placement
    if args.helion_c_store_mode != TCGEN05_C_STORE_MODE_NORMAL:
        config[TCGEN05_C_STORE_MODE_CONFIG_KEY] = args.helion_c_store_mode
        config[TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY] = True
    if args.helion_aux_load_mode != TCGEN05_AUX_LOAD_MODE_SIMT:
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = args.helion_aux_load_mode
    if args.helion_sched_consumer_wait_mode != TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL:
        config[TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY] = (
            args.helion_sched_consumer_wait_mode
        )
    if args.helion_sched_stage_count != 1:
        config[TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY] = args.helion_sched_stage_count
    if args.helion_epilogue_layout != TCGEN05_EPILOGUE_LAYOUT_NORMAL:
        config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY] = args.helion_epilogue_layout
    if args.helion_cubin_lineinfo:
        config[TCGEN05_CUBIN_LINEINFO_CONFIG_KEY] = True
    return helion.Config(**config)


_HELION_DIAGNOSTIC_ARG_DEFAULTS: dict[str, object] = {
    "helion_acc_wait_placement": TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    "helion_acc_producer_mode": TCGEN05_ACC_PRODUCER_MODE_NORMAL,
    "helion_c_acquire_placement": TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    "helion_c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
    "helion_aux_load_mode": TCGEN05_AUX_LOAD_MODE_SIMT,
    "helion_sched_consumer_wait_mode": TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL,
    "helion_sched_stage_count": 1,
    "helion_epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    "helion_cubin_lineinfo": False,
}


def _helion_args_with_diagnostic_defaults(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(args).copy()
    for key, default in _HELION_DIAGNOSTIC_ARG_DEFAULTS.items():
        values.setdefault(key, default)
    return argparse.Namespace(**values)


def _helion_diagnostic_flag_args(args: argparse.Namespace) -> list[str]:
    args = _helion_args_with_diagnostic_defaults(args)
    result: list[str] = []
    if args.helion_acc_wait_placement != TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP:
        result.extend(
            [
                "--helion-acc-wait-placement",
                args.helion_acc_wait_placement,
            ]
        )
    if args.helion_acc_producer_mode != TCGEN05_ACC_PRODUCER_MODE_NORMAL:
        result.extend(["--helion-acc-producer-mode", args.helion_acc_producer_mode])
    if args.helion_c_acquire_placement != TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP:
        result.extend(
            [
                "--helion-c-acquire-placement",
                args.helion_c_acquire_placement,
            ]
        )
    if args.helion_c_store_mode != TCGEN05_C_STORE_MODE_NORMAL:
        result.extend(["--helion-c-store-mode", args.helion_c_store_mode])
    if args.helion_aux_load_mode != TCGEN05_AUX_LOAD_MODE_SIMT:
        result.extend(["--helion-aux-load-mode", args.helion_aux_load_mode])
    if args.helion_sched_consumer_wait_mode != TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL:
        result.extend(
            [
                "--helion-sched-consumer-wait-mode",
                args.helion_sched_consumer_wait_mode,
            ]
        )
    if args.helion_sched_stage_count != 1:
        result.extend(
            ["--helion-sched-stage-count", str(args.helion_sched_stage_count)]
        )
    if args.helion_epilogue_layout != TCGEN05_EPILOGUE_LAYOUT_NORMAL:
        result.extend(["--helion-epilogue-layout", args.helion_epilogue_layout])
    if args.helion_cubin_lineinfo:
        result.append("--helion-cubin-lineinfo")
    return result


def _helion_codegen_markers(code: str) -> dict[str, bool]:
    return {
        "uses_tcgen05": "cute.nvgpu.tcgen05.CtaGroup" in code,
        "uses_tcgen05_two_cta": "cute.nvgpu.tcgen05.CtaGroup.TWO" in code,
        "uses_tma_umma_pipeline": "PipelineTmaUmma.create(" in code,
        "uses_generic_virtual_pid_loop": "for virtual_pid in " in code,
    }


class BiasEpilogue(NamedTuple):
    bias: torch.Tensor

    @property
    def fn(self) -> Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
        # Bind tensors into a real Python closure so Helion closure lifting sees
        # the tensor origins directly instead of treating the wrapper as opaque.
        bias = self.bias

        def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return acc + bias[tile[1]]

        return epilogue

    def __call__(
        self, acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.fn(acc, tile)

    @property
    def __closure__(self) -> tuple[Any, ...] | None:
        return self.fn.__closure__


class ReluEpilogue(NamedTuple):
    @property
    def fn(self) -> Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
        def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return torch.relu(acc)

        return epilogue

    def __call__(
        self, acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.fn(acc, tile)

    @property
    def __closure__(self) -> tuple[Any, ...] | None:
        return self.fn.__closure__


class BiasReluEpilogue(NamedTuple):
    bias: torch.Tensor

    @property
    def fn(self) -> Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
        # Bind tensors into a real Python closure so Helion closure lifting sees
        # the tensor origins directly instead of treating the wrapper as opaque.
        bias = self.bias

        def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return torch.relu(acc + bias[tile[1]])

        return epilogue

    def __call__(
        self, acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.fn(acc, tile)

    @property
    def __closure__(self) -> tuple[Any, ...] | None:
        return self.fn.__closure__


class BiasResidualGeluEpilogue(NamedTuple):
    bias: torch.Tensor
    residual: torch.Tensor

    @property
    def fn(self) -> Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
        # Bind tensors into a real Python closure so Helion closure lifting sees
        # the tensor origins directly instead of treating the wrapper as opaque.
        bias = self.bias
        residual = self.residual

        def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
            val = 1.25 * acc
            val = val + 0.5 * residual[tile[0], tile[1]].to(torch.float32)
            val = val + bias[tile[1]].to(torch.float32)
            return torch.nn.functional.gelu(val)

        return epilogue

    def __call__(
        self, acc: torch.Tensor, tile: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.fn(acc, tile)

    @property
    def __closure__(self) -> tuple[Any, ...] | None:
        return self.fn.__closure__


def _helion_matmul_args(
    args: argparse.Namespace,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> tuple[Any, ...]:
    if args.epilogue == "none":
        return (a, b)
    if args.epilogue == "bias":
        assert bias is not None
        return (a, b, BiasEpilogue(bias))
    if args.epilogue == "relu":
        return (a, b, ReluEpilogue())
    if args.epilogue == "bias_relu":
        assert bias is not None
        return (a, b, BiasReluEpilogue(bias))
    if args.epilogue == "bias_residual_gelu":
        assert bias is not None
        assert residual is not None
        return (a, b, BiasResidualGeluEpilogue(bias, residual))
    raise AssertionError(f"unhandled epilogue {args.epilogue!r}")


def _prepare_helion(args: argparse.Namespace) -> _PreparedHelion:
    backend = args.helion_backend
    os.environ["HELION_BACKEND"] = backend
    from examples.matmul import matmul

    dtype, a, b, bias, residual = _make_matmul_problem(args)
    expected = _matmul_expected(args, a, b, bias, residual, dtype)
    kernel_args = _helion_matmul_args(args, a, b, bias, residual)

    bound = matmul.bind(kernel_args)
    config = _make_helion_config_from_args(args) if args.helion_force_config else None
    if config is not None and any(key in config.config for key in _TCGEN05_CONFIG_KEYS):
        bound.env.config_spec.cute_tcgen05_search_enabled = True

    if config is None:
        active_config = bound.autotune(kernel_args, force=True)
        bound = matmul.bind(kernel_args)
        bound.set_config(active_config)
    else:
        bound.set_config(config)
        active_config = config

    code = bound.to_triton_code(active_config)
    codegen = _helion_codegen_markers(code)
    if args.helion_require_tcgen05 and not codegen["uses_tcgen05"]:
        raise RuntimeError(
            f"helion-{backend}: tcgen05 not selected for {active_config!r}"
        )
    return _PreparedHelion(
        backend=backend,
        dtype=dtype,
        kernel_args=kernel_args,
        expected=expected,
        bound=bound,
        active_config=active_config,
        code=code,
        codegen=codegen,
    )


def _benchmark_helion(args: argparse.Namespace) -> dict[str, Any]:
    """Helion via examples/matmul.py.

    Backend is determined by ``args.helion_backend`` (cute or triton). Pass
    --helion-force-config 1 to skip autotune and use the --helion-* args.
    """
    prepared = _prepare_helion(args)

    impl_label = f"helion-{prepared.backend}"
    if args.print_codegen:
        return {
            "impl": impl_label,
            "shape": {
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "dtype": args.dtype,
                "epilogue": args.epilogue,
            },
            "config": repr(prepared.active_config),
            "codegen": prepared.codegen,
        }

    if not args.skip_correctness:
        _check_close(
            prepared.bound(*prepared.kernel_args), prepared.expected, prepared.dtype
        )

    fn = lambda: prepared.bound(*prepared.kernel_args)  # noqa: E731
    stats = _bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result(
        impl_label,
        args,
        stats,
        config=repr(prepared.active_config),
        codegen=prepared.codegen,
    )


_IMPL_DISPATCH: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "aten": _benchmark_aten,
    "quack": _benchmark_quack,
    "quack-direct": _benchmark_quack_direct,
}


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.impl == "helion-cute":
        args.helion_backend = "cute"
        return _benchmark_helion(args)
    if args.impl == "helion-triton":
        args.helion_backend = "triton"
        return _benchmark_helion(args)
    return _IMPL_DISPATCH[args.impl](args)


def _build_subprocess_cmd(args: argparse.Namespace, impl: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--epilogue",
        args.epilogue,
        "--dtype",
        args.dtype,
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--skip-correctness",
        str(int(args.skip_correctness)),
        "--json",
    ]
    if args.quack_path:
        cmd.extend(["--quack-path", args.quack_path])
    if impl == "quack-direct":
        cmd.extend(
            [
                "--quack-tile-m",
                str(args.quack_tile_m),
                "--quack-tile-n",
                str(args.quack_tile_n),
                "--quack-cluster-m",
                str(args.quack_cluster_m),
                "--quack-cluster-n",
                str(args.quack_cluster_n),
                "--quack-pingpong",
                str(int(args.quack_pingpong)),
                "--quack-persistent",
                str(int(args.quack_persistent)),
                "--quack-dynamic-persistent",
                str(int(args.quack_dynamic_persistent)),
                "--quack-max-swizzle-size",
                str(args.quack_max_swizzle_size),
                "--quack-tune",
                args.quack_tune,
            ]
        )
    if impl in ("helion-cute", "helion-triton"):
        cmd.extend(
            [
                "--helion-force-config",
                str(int(args.helion_force_config)),
                "--helion-tile-m",
                str(args.helion_tile_m),
                "--helion-tile-n",
                str(args.helion_tile_n),
                "--helion-tile-k",
                str(args.helion_tile_k),
                "--helion-loop-order",
                _format_int_list(args.helion_loop_order),
                "--helion-num-warps",
                str(args.helion_num_warps),
                "--helion-num-stages",
                str(args.helion_num_stages),
                "--helion-l2-grouping",
                str(args.helion_l2_grouping),
                "--helion-pid-type",
                args.helion_pid_type,
                "--helion-cluster-m",
                str(args.helion_cluster_m),
                "--helion-cluster-n",
                str(args.helion_cluster_n),
                "--helion-ab-stages",
                _format_optional_int(args.helion_ab_stages),
                "--helion-acc-stages",
                _format_optional_int(args.helion_acc_stages),
                "--helion-c-stages",
                _format_optional_int(args.helion_c_stages),
                "--helion-num-epi-warps",
                str(args.helion_num_epi_warps),
                "--helion-require-tcgen05",
                str(int(args.helion_require_tcgen05 if impl == "helion-cute" else 0)),
                "--helion-range-flattens",
                _format_optional_bool_list(args.helion_range_flattens),
                "--helion-range-multi-buffers",
                _format_optional_bool_list(args.helion_range_multi_buffers),
                "--helion-range-warp-specializes",
                _format_optional_bool_list(args.helion_range_warp_specializes),
                "--helion-range-num-stages",
                _format_int_list(args.helion_range_num_stages),
                "--helion-range-unroll-factors",
                _format_int_list(args.helion_range_unroll_factors),
            ]
        )
        if args.helion_num_sm_multiplier != 1:
            cmd.extend(
                ["--helion-num-sm-multiplier", str(args.helion_num_sm_multiplier)]
            )
        cmd.extend(_helion_diagnostic_flag_args(args))
        if args.helion_indexing is not None:
            cmd.extend(["--helion-indexing", ",".join(args.helion_indexing)])
        if args.helion_maxnreg is not None:
            cmd.extend(["--helion-maxnreg", str(args.helion_maxnreg)])
        if args.helion_strategy is not None:
            cmd.extend(["--helion-strategy", args.helion_strategy])
        if args.helion_c_input_warps is not None:
            cmd.extend(["--helion-c-input-warps", str(args.helion_c_input_warps)])
        if args.helion_persistence_model is not None:
            cmd.extend(["--helion-persistence-model", args.helion_persistence_model])
        if args.helion_l2_swizzle_size is not None:
            cmd.extend(["--helion-l2-swizzle-size", str(args.helion_l2_swizzle_size)])
    return cmd


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    impls = args.impls or list(DEFAULT_IMPLS)
    results: list[dict[str, Any]] = []
    for impl in impls:
        if impl not in ALL_IMPLS:
            print(f"unknown impl {impl!r}, skipping", file=sys.stderr)
            continue
        print(f"\n=== Running {impl} ===", flush=True)
        cmd = _build_subprocess_cmd(args, impl)
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"impl {impl} failed (rc={proc.returncode})\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}",
                file=sys.stderr,
            )
            continue
        stdout_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        try:
            results.append(json.loads(stdout_lines[-1]))
        except (IndexError, json.JSONDecodeError):
            print(
                f"impl {impl} produced no JSON output\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}",
                file=sys.stderr,
            )
    return {
        "shape": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "epilogue": args.epilogue,
        },
        "results": results,
    }


def _two_cta_diagnostic_variant_args(
    args: argparse.Namespace,
) -> list[tuple[str, argparse.Namespace]]:
    variants: list[tuple[str, argparse.Namespace]] = []
    for name, overrides in _TWO_CTA_DIAGNOSTIC_VARIANTS:
        values = vars(args).copy()
        values.update(_TWO_CTA_DIAGNOSTIC_BASE)
        values.update(overrides)
        values["impl"] = "helion-cute"
        values["helion_backend"] = "cute"
        # Avoid sharing mutable list overrides between variants and callers.
        for key, value in list(values.items()):
            if isinstance(value, list):
                values[key] = list(value)
        variants.append((name, argparse.Namespace(**values)))
    return variants


def _two_cta_seed_args(args: argparse.Namespace) -> argparse.Namespace:
    for variant, variant_args in _two_cta_diagnostic_variant_args(args):
        if variant == "seed":
            return variant_args
    raise RuntimeError("missing CtaGroup.TWO diagnostic seed variant")


def _two_cta_edge_k_tail_monolithic_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "impl": "helion-cute",
            "helion_backend": "cute",
            "helion_force_config": 1,
            "helion_tile_m": TCGEN05_TWO_CTA_BLOCK_M,
            "helion_tile_n": TCGEN05_TWO_CTA_BLOCK_N,
            "helion_tile_k": TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
            "helion_loop_order": [0, 1],
            "helion_num_stages": 2,
            "helion_num_warps": 4,
            "helion_maxnreg": None,
            "helion_num_sm_multiplier": 1,
            "helion_persistence_model": None,
            "helion_l2_grouping": TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING,
            "helion_l2_swizzle_size": TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
            "helion_indexing": list(_TENSOR_DESCRIPTOR_INDEXING),
            "helion_pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "helion_cluster_m": 2,
            "helion_cluster_n": 1,
            "helion_ab_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
            "helion_acc_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
            "helion_c_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES,
            "helion_num_epi_warps": 4,
            "helion_strategy": "role_local_monolithic",
            "helion_c_input_warps": 0,
            "helion_require_tcgen05": 1,
            "helion_range_flattens": list(_HELION_DEFAULT_RANGE_FLATTENS),
            "helion_range_multi_buffers": list(_HELION_DEFAULT_RANGE_MULTI_BUFFERS),
            "helion_range_warp_specializes": list(
                _HELION_DEFAULT_RANGE_WARP_SPECIALIZES
            ),
            "helion_range_num_stages": list(_HELION_DEFAULT_RANGE_NUM_STAGES),
            "helion_range_unroll_factors": list(_HELION_DEFAULT_RANGE_UNROLL_FACTORS),
            "helion_acc_producer_mode": TCGEN05_ACC_PRODUCER_MODE_NORMAL,
            "helion_acc_wait_placement": TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
            "helion_c_acquire_placement": TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
            "helion_c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
            "helion_epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
            "helion_cubin_lineinfo": bool(
                getattr(args, "helion_cubin_lineinfo", False)
            ),
        }
    )
    return argparse.Namespace(**values)


def _two_cta_edge_k_tail_scheduler_split_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_edge_k_tail_monolithic_args(args)).copy()
    values["helion_strategy"] = "role_local_with_scheduler"
    values["helion_c_input_warps"] = 1
    values["helion_l2_swizzle_size"] = (
        TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
    )
    return argparse.Namespace(**values)


def _two_cta_acc_stages1_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_acc_stages"] = 1
    return argparse.Namespace(**values)


def _two_cta_first_c_acquire_in_loop_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_c_acquire_placement"] = TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
    return argparse.Namespace(**values)


def _two_cta_later_c_acquire_before_barrier_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_c_acquire_placement"] = (
        TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER
    )
    return argparse.Namespace(**values)


def _two_cta_acc_wait_before_subtile_loop_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_acc_wait_placement"] = TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
    return argparse.Namespace(**values)


def _two_cta_split_first_t2r_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_epilogue_layout"] = TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R
    return argparse.Namespace(**values)


def _two_cta_split_acc_t2r_store_tail_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_epilogue_layout"] = TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL
    return argparse.Namespace(**values)


def _two_cta_module_helper_acc_t2r_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_epilogue_layout"] = TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R
    return argparse.Namespace(**values)


def _two_cta_module_helper_store_tail_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_epilogue_layout"] = TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL
    return argparse.Namespace(**values)


def _two_cta_skip_epilogue_store_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_c_store_mode"] = TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE
    return argparse.Namespace(**values)


def _two_cta_skip_umma_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(_two_cta_seed_args(args)).copy()
    values["helion_acc_producer_mode"] = TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA
    return argparse.Namespace(**values)


def _cluster_m1_flat_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    # Preserve any user-supplied ``--helion-strategy``; the diagnostic
    # baseline does not pin a strategy.
    user_strategy = values.get("helion_strategy")
    values.update(
        {
            "impl": "helion-cute",
            "helion_backend": "cute",
            "helion_force_config": 1,
            "helion_tile_m": 128,
            "helion_tile_n": 256,
            "helion_tile_k": 128,
            "helion_loop_order": [0, 1],
            "helion_num_stages": 2,
            "helion_num_warps": 4,
            "helion_maxnreg": None,
            "helion_num_sm_multiplier": 1,
            "helion_l2_grouping": 4,
            "helion_indexing": None,
            "helion_pid_type": "flat",
            "helion_cluster_m": 1,
            "helion_ab_stages": 2,
            "helion_acc_stages": 1,
            "helion_c_stages": 4,
            "helion_num_epi_warps": 4,
            "helion_strategy": user_strategy,
            "helion_acc_producer_mode": TCGEN05_ACC_PRODUCER_MODE_NORMAL,
            "helion_acc_wait_placement": TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
            "helion_c_acquire_placement": TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
            "helion_c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
            "helion_epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
            "helion_require_tcgen05": 1,
            "helion_range_flattens": list(_HELION_DEFAULT_RANGE_FLATTENS),
            "helion_range_multi_buffers": list(_HELION_DEFAULT_RANGE_MULTI_BUFFERS),
            "helion_range_warp_specializes": list(
                _HELION_DEFAULT_RANGE_WARP_SPECIALIZES
            ),
            "helion_range_num_stages": list(_HELION_DEFAULT_RANGE_NUM_STAGES),
            "helion_range_unroll_factors": list(_HELION_DEFAULT_RANGE_UNROLL_FACTORS),
        }
    )
    return argparse.Namespace(**values)


def _cluster_m2_cta_group_one_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    # Preserve any user-supplied ``--helion-strategy``.
    user_strategy = values.get("helion_strategy")
    values.update(
        {
            "impl": "helion-cute",
            "helion_backend": "cute",
            "helion_force_config": 1,
            "helion_tile_m": 128,
            "helion_tile_n": 256,
            "helion_tile_k": 128,
            "helion_loop_order": [0, 1],
            "helion_num_stages": 2,
            "helion_num_warps": 4,
            "helion_maxnreg": None,
            "helion_num_sm_multiplier": 1,
            "helion_l2_grouping": 4,
            "helion_indexing": list(_TENSOR_DESCRIPTOR_INDEXING),
            "helion_pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "helion_cluster_m": 2,
            "helion_ab_stages": 2,
            "helion_acc_stages": 1,
            "helion_c_stages": 4,
            "helion_num_epi_warps": 4,
            "helion_strategy": user_strategy,
            "helion_acc_producer_mode": TCGEN05_ACC_PRODUCER_MODE_NORMAL,
            "helion_acc_wait_placement": TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
            "helion_c_acquire_placement": TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
            "helion_c_store_mode": TCGEN05_C_STORE_MODE_NORMAL,
            "helion_epilogue_layout": TCGEN05_EPILOGUE_LAYOUT_NORMAL,
            "helion_require_tcgen05": 1,
            "helion_range_flattens": [None, None],
            "helion_range_multi_buffers": [None, None],
            "helion_range_warp_specializes": [None, None],
            "helion_range_num_stages": [0, 0],
            "helion_range_unroll_factors": [0, 0],
        }
    )
    return argparse.Namespace(**values)


def _ncu_target_helion_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.ncu_target_config == "two_cta_seed":
        return _two_cta_seed_args(args)
    if args.ncu_target_config == "two_cta_edge_k_tail_monolithic":
        return _two_cta_edge_k_tail_monolithic_args(args)
    if args.ncu_target_config == "two_cta_edge_k_tail_scheduler_split":
        return _two_cta_edge_k_tail_scheduler_split_args(args)
    if args.ncu_target_config == "two_cta_acc_stages1":
        return _two_cta_acc_stages1_args(args)
    if args.ncu_target_config == "two_cta_first_c_acquire_in_loop":
        return _two_cta_first_c_acquire_in_loop_args(args)
    if args.ncu_target_config == "two_cta_later_c_acquire_before_barrier":
        return _two_cta_later_c_acquire_before_barrier_args(args)
    if args.ncu_target_config == "two_cta_acc_wait_before_subtile_loop":
        return _two_cta_acc_wait_before_subtile_loop_args(args)
    if args.ncu_target_config == "two_cta_split_first_t2r":
        return _two_cta_split_first_t2r_args(args)
    if args.ncu_target_config == "two_cta_split_acc_t2r_store_tail":
        return _two_cta_split_acc_t2r_store_tail_args(args)
    if args.ncu_target_config == "two_cta_module_helper_acc_t2r":
        return _two_cta_module_helper_acc_t2r_args(args)
    if args.ncu_target_config == "two_cta_module_helper_store_tail":
        return _two_cta_module_helper_store_tail_args(args)
    if args.ncu_target_config == "two_cta_skip_epilogue_store":
        return _two_cta_skip_epilogue_store_args(args)
    if args.ncu_target_config == "two_cta_skip_umma":
        return _two_cta_skip_umma_args(args)
    if args.ncu_target_config == "cluster_m1_flat":
        return _cluster_m1_flat_args(args)
    raise RuntimeError(f"unknown NCU target config {args.ncu_target_config!r}")


def _ncu_target_quack_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values["impl"] = "quack-direct"
    values.update(_QUACK_TWO_CTA_NCU_BASE)
    if args.ncu_target_config == "quack_same_tile":
        values.update(
            {
                "quack_tile_m": 128,
                "quack_tile_n": 256,
                "quack_cluster_m": 1,
                "quack_cluster_n": 1,
            }
        )
    elif args.ncu_target_config == "quack_best":
        values.update(
            {
                "quack_tile_m": 256,
                "quack_tile_n": 256,
                "quack_cluster_m": 2,
                "quack_cluster_n": 1,
            }
        )
    else:
        raise RuntimeError(
            f"unknown Quack NCU target config {args.ncu_target_config!r}"
        )
    return argparse.Namespace(**values)


def _is_quack_ncu_target(target_config: str) -> bool:
    return target_config in _QUACK_TWO_CTA_NCU_TARGET_CONFIGS


def _run_helion_two_cta_diagnostic_sweep(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for variant, variant_args in _two_cta_diagnostic_variant_args(args):
        print(f"\n=== Running helion-cute:{variant} ===", flush=True)
        result = _benchmark_helion(variant_args)
        result["variant"] = variant
        result["impl"] = f"helion-cute:{variant}"
        results.append(result)
    return {
        "shape": {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype},
        "results": results,
    }


def _epilogue_barrier_context_marker(
    lines: list[str], barrier_index: int
) -> str | None:
    # Diagnostic-only heuristics for the current Helion generated source and
    # Quack TMA epilogue spelling.
    previous_context = "\n".join(
        lines[
            max(
                0, barrier_index - _EPILOGUE_BARRIER_PREVIOUS_CONTEXT_LINES
            ) : barrier_index
        ]
    )
    following_context = "\n".join(
        lines[
            barrier_index + 1 : barrier_index
            + 1
            + _EPILOGUE_BARRIER_FOLLOWING_CONTEXT_LINES
        ]
    )
    previous_line = lines[barrier_index - 1].strip() if barrier_index else ""
    if previous_line == "else:" and "if const_expr(use_tma_epi)" in previous_context:
        return None
    if any(
        needle in previous_context for needle in _SOURCE_MARKERS["async_shared_fence"]
    ):
        return _EPILOGUE_BARRIER_BEFORE_TMA_MARKER
    if "tcgen05_c_buffer =" in following_context or "epi_buffer =" in following_context:
        return _EPILOGUE_BARRIER_BEFORE_SMEM_MARKER
    return None


def _source_markers_for_line(lines: list[str], index: int) -> list[str]:
    line = lines[index]
    markers = [
        label
        for label, needles in _SOURCE_MARKERS.items()
        if any(needle in line for needle in needles)
    ]
    if "epilogue_barrier_wait" in markers:
        context_marker = _epilogue_barrier_context_marker(lines, index)
        if context_marker is not None:
            markers.insert(markers.index("epilogue_barrier_wait"), context_marker)
    return markers


def _source_marker_counts(source: str) -> dict[str, int]:
    counts = dict.fromkeys(_SOURCE_MARKER_LABELS, 0)
    lines = source.splitlines()
    for index in range(len(lines)):
        # Marker counts are line based: a label contributes at most once per
        # source line even if multiple aliases occur on that line.
        for label in _source_markers_for_line(lines, index):
            counts[label] += 1
    return counts


def _source_marker_lines(
    source: str, *, limit: int = _SOURCE_MARKER_LOCATION_LIMIT
) -> dict[str, list[int]]:
    lines = source.splitlines()
    result: dict[str, list[int]] = {label: [] for label in _SOURCE_MARKER_LABELS}
    for index in range(len(lines)):
        lineno = index + 1
        for label in _source_markers_for_line(lines, index):
            if len(result[label]) < limit:
                result[label].append(lineno)
    return result


def _source_bundle_summary(
    sources: dict[str, str], *, location_limit: int = _SOURCE_MARKER_LOCATION_LIMIT
) -> dict[str, Any]:
    markers = dict.fromkeys(_SOURCE_MARKER_LABELS, 0)
    locations: dict[str, list[str]] = {label: [] for label in _SOURCE_MARKER_LABELS}
    line_count = 0
    for source_name, source in sources.items():
        line_count += len(source.splitlines())
        source_counts = _source_marker_counts(source)
        for label, count in source_counts.items():
            markers[label] += count
        source_lines = _source_marker_lines(source, limit=location_limit)
        for label, lines in source_lines.items():
            remaining = location_limit - len(locations[label])
            if remaining > 0:
                locations[label].extend(
                    f"{source_name}:{line}" for line in lines[:remaining]
                )
    return {
        "line_count": line_count,
        "markers": markers,
        "first_marker_locations": locations,
    }


def _nonzero_markers(markers: dict[str, int]) -> dict[str, int]:
    return {label: count for label, count in markers.items() if count}


def _source_marker_trace(
    source: str,
    *,
    line_offset: int = 0,
    limit: int | None = _SOURCE_MARKER_LOCATION_LIMIT,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        lineno = line_offset + index + 1
        for label in _source_markers_for_line(lines, index):
            trace.append({"line": lineno, "marker": label, "text": line.strip()})
            if limit is not None and len(trace) == limit:
                return trace
    return trace


def _pipeline_copy_event_from_text(text: str) -> str | None:
    for event, needles in _PIPELINE_COPY_EVENT_NEEDLES:
        if any(needle in text for needle in needles):
            return event
    return None


def _pipeline_copy_event_for_line(lines: list[str], index: int) -> str | None:
    line = lines[index]
    stripped = line.lstrip()
    if "copy_utils.cvt_copy" in line or "copy_utils.sr_cvt_copy" in line:
        return "reg_to_smem_copy"
    if stripped.startswith(("copy_D(", "copy_aux_out(")):
        return "tma_store_copy"
    if "cute.copy(" not in line and "copy_tensor" not in line:
        return None

    if event := _pipeline_copy_event_from_text(line):
        return event
    # Multi-line CuTe/Quack copy calls often put the atom or source tensor on
    # following lines; the first diagnostic role match in this window wins.
    context = "\n".join(lines[index : index + _PIPELINE_COPY_CONTEXT_LINES])
    return _pipeline_copy_event_from_text(context) or "generic_copy"


def _pipeline_event_site_for_line(lines: list[str], index: int, event: str) -> str:
    # Diagnostic-only labels depend on current generated guard spelling.
    context = "\n".join(
        lines[max(0, index - _PIPELINE_EVENT_SITE_LOOKBACK_LINES) : index + 1]
    )
    if event == "c_producer_acquire":
        if "_tcgen05_subtile != 0" in context:
            if "tcgen05_tRS_rD.store(" in context:
                return _HELION_LATER_SUBTILE_C_BEFORE_BARRIER_ACQUIRE_SITE
            return _HELION_LATER_SUBTILE_C_ACQUIRE_SITE
        if "_tcgen05_subtile == 0" in context:
            return _HELION_FIRST_SUBTILE_C_LOOP_ACQUIRE_SITE
        if (
            "_tcgen05_subtile" not in context
            and "tcgen05_epi_active" in context
            and "tcgen05_warp_idx" in context
        ):
            return _HELION_FIRST_SUBTILE_C_ACQUIRE_SITE
    if event == "acc_consumer_wait":
        if "_tcgen05_subtile == 0" in context:
            return _HELION_FIRST_SUBTILE_ACC_WAIT_SITE
        if "_tcgen05_subtile" not in context and "tcgen05_epi_active" in context:
            return _HELION_FIRST_SUBTILE_ACC_BEFORE_LOOP_WAIT_SITE
    return ""


def _pipeline_event_trace(
    source: str,
    *,
    line_offset: int = 0,
    limit: int | None = _PIPELINE_EVENT_TRACE_LIMIT,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        lineno = line_offset + index + 1
        copy_event = _pipeline_copy_event_for_line(lines, index)
        if copy_event is not None:
            trace.append(
                {
                    "line": lineno,
                    "event": copy_event,
                    "site": _pipeline_event_site_for_line(lines, index, copy_event),
                    "text": line.strip(),
                }
            )
            if limit is not None and len(trace) == limit:
                return trace
        for label, needles in _PIPELINE_EVENT_PATTERNS:
            if any(needle in line for needle in needles):
                trace.append(
                    {
                        "line": lineno,
                        "event": label,
                        "site": _pipeline_event_site_for_line(lines, index, label),
                        "text": line.strip(),
                    }
                )
                if limit is not None and len(trace) == limit:
                    return trace
    return trace


def _pipeline_event_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(_PIPELINE_EVENT_COUNT_LABELS, 0)
    for event in trace:
        counts[event["event"]] += 1
    return {label: count for label, count in counts.items() if count}


def _pipeline_event_gaps(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for name, start_event, end_event in _PIPELINE_GAP_PAIRS:
        for start_idx, start in enumerate(trace):
            if start["event"] != start_event:
                continue
            for end_idx in range(start_idx + 1, len(trace)):
                end = trace[end_idx]
                if end["event"] != end_event:
                    continue
                gaps.append(
                    {
                        "name": name,
                        "start_event": start_event,
                        "end_event": end_event,
                        "start_line": start["line"],
                        "end_line": end["line"],
                        "line_delta": end["line"] - start["line"],
                        "event_delta": end_idx - start_idx,
                    }
                )
                break
            break
    return gaps


def _pipeline_event_windows(
    trace: list[dict[str, Any]],
    *,
    limit: int | None = _PIPELINE_WINDOW_LIMIT,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for name, sequence in _PIPELINE_WINDOW_SPECS:
        anchor_event = sequence[0]
        for start_idx, start in enumerate(trace):
            if start["event"] != anchor_event:
                continue
            next_anchor_idx = next(
                (
                    idx
                    for idx in range(start_idx + 1, len(trace))
                    if trace[idx]["event"] == anchor_event
                ),
                len(trace),
            )
            event_indices = [start_idx]
            missing_events: list[str] = []
            search_idx = start_idx + 1
            for expected_event in sequence[1:]:
                found_idx = next(
                    (
                        idx
                        for idx in range(search_idx, next_anchor_idx)
                        if trace[idx]["event"] == expected_event
                    ),
                    None,
                )
                if found_idx is None:
                    missing_events.append(expected_event)
                    break
                event_indices.append(found_idx)
                search_idx = found_idx + 1

            event_entries = [
                {
                    "event_index": idx,
                    "line": trace[idx]["line"],
                    "event": trace[idx]["event"],
                    "site": trace[idx]["site"],
                    "text": trace[idx]["text"],
                }
                for idx in event_indices
            ]
            last_idx = event_indices[-1]
            windows.append(
                {
                    "name": name,
                    "complete": len(event_indices) == len(sequence),
                    "missing_events": missing_events,
                    "events": event_entries,
                    "start_line": start["line"],
                    "end_line": trace[last_idx]["line"],
                    "line_span": trace[last_idx]["line"] - start["line"],
                    "event_span": last_idx - start_idx,
                }
            )
    windows.sort(key=itemgetter("start_line", "name"))
    if limit is None:
        return windows
    return windows[:limit]


def _pipeline_event_summary(
    source: str,
    *,
    line_offset: int = 0,
    trace_limit: int | None = _PIPELINE_EVENT_TRACE_LIMIT,
    include_windows: bool = True,
    window_limit: int | None = _PIPELINE_WINDOW_LIMIT,
) -> dict[str, Any]:
    full_trace = _pipeline_event_trace(source, line_offset=line_offset, limit=None)
    trace = full_trace if trace_limit is None else full_trace[:trace_limit]
    result = {
        "counts": _pipeline_event_counts(full_trace),
        "trace": trace,
        "trace_shown": len(trace),
        "trace_total": len(full_trace),
        "gaps": _pipeline_event_gaps(full_trace),
    }
    if include_windows:
        result["windows"] = _pipeline_event_windows(full_trace, limit=window_limit)
    return result


def _source_bundle_pipeline_summary(sources: dict[str, str]) -> dict[str, Any]:
    counts = dict.fromkeys(_PIPELINE_EVENT_COUNT_LABELS, 0)
    trace: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    total_events = 0
    for source_name, source in sources.items():
        summary = _pipeline_event_summary(source, include_windows=False)
        total_events += summary["trace_total"]
        for label, count in summary["counts"].items():
            counts[label] += count
        for event in summary["trace"]:
            trace.append({"source": source_name, **event})
        for gap in summary["gaps"]:
            gaps.append({"source": source_name, **gap})
    return {
        "counts": {label: count for label, count in counts.items() if count},
        "trace": trace[:_PIPELINE_EVENT_TRACE_LIMIT],
        "trace_shown": min(len(trace), _PIPELINE_EVENT_TRACE_LIMIT),
        "trace_total": total_events,
        "gaps": gaps,
    }


def _classify_helion_role_local_loop(source: str) -> str:
    # In the current role-local lowering only the TMA-load role acquires
    # the AB producer pipeline; MMA-exec releases AB after consuming it.
    if "tcgen05_ab_pipeline.producer_acquire" in source:
        return "tma_load"
    if "tcgen05_acc_pipeline.producer_acquire" in source:
        return "mma_exec"
    if "tcgen05_c_pipeline.producer_acquire" in source:
        return "epilogue"
    return "unknown"


def _helion_role_local_loop_summaries(
    code: str,
    *,
    trace_limit: int | None = _PIPELINE_EVENT_TRACE_LIMIT,
    marker_trace_limit: int | None = _HELION_ROLE_LOCAL_MARKER_TRACE_LIMIT,
    window_limit: int | None = _PIPELINE_WINDOW_LIMIT,
) -> list[dict[str, Any]]:
    """Summarize each generated role-local persistent loop.

    The CtaGroup.TWO performance investigation needs to know whether markers
    such as PDL waits, scheduler construction, and pipeline operations sit in
    the TMA, MMA, or epilogue role. Plain whole-file counts hide that shape.
    """
    result: list[dict[str, Any]] = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        predicate = ast.unparse(node.test)
        if "cute.arch.make_warp_uniform(cute.arch.warp_idx())" not in predicate:
            continue
        source = ast.get_source_segment(code, node)
        if source is None:
            continue
        if "tcgen05_role_local_" not in source:
            continue
        if "StaticPersistentTileScheduler.create(" not in source:
            continue
        markers = _source_marker_counts(source)
        pipeline_events = _pipeline_event_summary(
            source,
            line_offset=node.lineno - 1,
            trace_limit=trace_limit,
            window_limit=window_limit,
        )
        result.append(
            {
                "role": _classify_helion_role_local_loop(source),
                "predicate": predicate,
                "line_start": node.lineno,
                "line_end": node.end_lineno,
                "markers": markers,
                "marker_trace": _source_marker_trace(
                    source,
                    line_offset=node.lineno - 1,
                    limit=marker_trace_limit,
                ),
                "pipeline_events": pipeline_events,
            }
        )
    return sorted(result, key=itemgetter("line_start"))


def _source_symbol_marker_summaries(
    sources: dict[str, str],
    *,
    trace_limit: int | None = _SOURCE_SYMBOL_TRACE_LIMIT,
    marker_trace_limit: int | None = _SOURCE_SYMBOL_MARKER_TRACE_LIMIT,
    window_limit: int | None = _PIPELINE_WINDOW_LIMIT,
) -> list[dict[str, Any]]:
    """Return marker/pipeline-bearing Quack functions and methods in source order."""
    result: list[dict[str, Any]] = []

    def visit_body(
        *,
        source_name: str,
        source: str,
        body: list[ast.stmt],
        parents: tuple[str, ...],
    ) -> None:
        for child in body:
            if isinstance(child, ast.ClassDef):
                visit_body(
                    source_name=source_name,
                    source=source,
                    body=child.body,
                    parents=(*parents, child.name),
                )
                continue
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified_name = ".".join((*parents, child.name))
            segment = ast.get_source_segment(source, child)
            if segment is None:
                continue
            markers = _source_marker_counts(segment)
            nonzero = _nonzero_markers(markers)
            pipeline_events = _pipeline_event_summary(
                segment,
                line_offset=child.lineno - 1,
                trace_limit=trace_limit,
                window_limit=window_limit,
            )
            if nonzero or pipeline_events["counts"]:
                result.append(
                    {
                        "source": source_name,
                        "symbol": qualified_name,
                        "line_start": child.lineno,
                        "line_end": child.end_lineno,
                        "markers": nonzero,
                        "pipeline_events": pipeline_events,
                        "marker_trace": _source_marker_trace(
                            segment,
                            line_offset=child.lineno - 1,
                            limit=marker_trace_limit,
                        ),
                    }
                )
            visit_body(
                source_name=source_name,
                source=source,
                body=child.body,
                parents=(*parents, child.name),
            )

    for source_name, source in sources.items():
        tree = ast.parse(source)
        visit_body(source_name=source_name, source=source, body=tree.body, parents=())
    return result


def _expand_wait_site_event_labels(candidate_events: list[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for event in candidate_events:
        labels.extend(_WAIT_SITE_EVENT_ALIASES.get(event, (event,)))
    return tuple(dict.fromkeys(labels))


def _expand_wait_site_marker_labels(candidate_events: list[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for event in candidate_events:
        labels.extend(_WAIT_SITE_MARKER_ALIASES.get(event, ()))
    return tuple(dict.fromkeys(labels))


def _source_context_matched_windows(
    windows: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    matched_events: dict[str, int],
    *,
    limit: int | None = _SOURCE_CONTEXT_WINDOW_LIMIT,
) -> list[dict[str, Any]]:
    ranked_windows: list[
        tuple[tuple[int, int, int, int, int, int, int], dict[str, Any]]
    ] = []
    for index, window in enumerate(
        [*windows, *_source_context_window_fragments(trace, matched_events)]
    ):
        window_matches = [
            event["event"]
            for event in window["events"]
            if event["event"] in matched_events
        ]
        if not window_matches:
            continue
        ranked_windows.append(
            (
                (
                    -len(set(window_matches)),
                    -len(window_matches),
                    0 if window["complete"] else 1,
                    window["event_span"],
                    window["line_span"],
                    window["start_line"],
                    index,
                ),
                window,
            )
        )
    ranked_windows.sort(key=itemgetter(0))
    selected: list[dict[str, Any]] = []
    selected_event_indices: list[set[int]] = []
    for _rank, window in ranked_windows:
        event_indices = {event["event_index"] for event in window["events"]}
        if any(
            window["name"] == selected_window["name"]
            and event_indices < selected_indices
            for selected_window, selected_indices in zip(
                selected, selected_event_indices, strict=True
            )
        ):
            continue
        selected.append(window)
        selected_event_indices.append(event_indices)
        if limit is not None and len(selected) == limit:
            break
    return selected


def _source_context_window_fragments(
    trace: list[dict[str, Any]], matched_events: dict[str, int]
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for name, sequence in _PIPELINE_WINDOW_SPECS:
        if not any(event in matched_events for event in sequence):
            continue
        for start_idx, start in enumerate(trace):
            if start["event"] not in sequence:
                continue
            sequence_start = sequence.index(start["event"])
            event_indices: list[int] = []
            missing_events = list(sequence[:sequence_start])
            search_idx = start_idx
            for expected_event in sequence[sequence_start:]:
                found_idx = next(
                    (
                        idx
                        for idx in range(search_idx, len(trace))
                        if trace[idx]["event"] == expected_event
                    ),
                    None,
                )
                if found_idx is None:
                    missing_events.append(expected_event)
                    continue
                event_indices.append(found_idx)
                search_idx = found_idx + 1
            if not any(trace[idx]["event"] in matched_events for idx in event_indices):
                continue
            if not missing_events and len(event_indices) == len(sequence):
                continue
            event_entries = [
                {
                    "event_index": idx,
                    "line": trace[idx]["line"],
                    "event": trace[idx]["event"],
                    "site": trace[idx]["site"],
                    "text": trace[idx]["text"],
                }
                for idx in event_indices
            ]
            last_idx = event_indices[-1]
            fragments.append(
                {
                    "name": name,
                    "complete": False,
                    "missing_events": missing_events,
                    "events": event_entries,
                    "start_line": start["line"],
                    "end_line": trace[last_idx]["line"],
                    "line_span": trace[last_idx]["line"] - start["line"],
                    "event_span": last_idx - start_idx,
                }
            )
    return fragments


def _source_context_event_sites(
    trace: list[dict[str, Any]], matched_events: dict[str, int]
) -> list[dict[str, Any]]:
    site_counts: dict[tuple[str, str], int] = {}
    for trace_entry in trace:
        event = trace_entry["event"]
        site = trace_entry["site"]
        if event in matched_events and site:
            site_counts[(event, site)] = site_counts.get((event, site), 0) + 1
    return [
        {"event": event, "site": site, "count": count}
        for (event, site), count in site_counts.items()
    ]


def _source_context_epilogue_subregions(
    matched_events: dict[str, int],
    matched_markers: dict[str, int],
    windows: list[dict[str, Any]],
) -> tuple[str, ...]:
    events = set(matched_events)
    for window in windows:
        events.update(event["event"] for event in window["events"])
    markers = set(matched_markers)
    return tuple(
        label
        for label, event_labels, marker_labels in _EPILOGUE_SUBREGION_RULES
        if events & event_labels or markers & marker_labels
    )


def _source_context_entry_match(
    entry: dict[str, Any],
    candidate_events: list[str],
    *,
    event_trace_limit: int = _SOURCE_CONTEXT_EVENT_TRACE_LIMIT,
    marker_trace_limit: int = _SOURCE_CONTEXT_MARKER_TRACE_LIMIT,
) -> dict[str, Any] | None:
    event_labels = _expand_wait_site_event_labels(candidate_events)
    marker_labels = _expand_wait_site_marker_labels(candidate_events)
    pipeline_events = entry["pipeline_events"]
    event_counts = pipeline_events["counts"]
    matched_events = {
        label: event_counts[label] for label in event_labels if event_counts.get(label)
    }
    event_sites = _source_context_event_sites(pipeline_events["trace"], matched_events)
    event_trace = [
        trace_entry
        for trace_entry in pipeline_events["trace"]
        if trace_entry["event"] in matched_events
    ][:event_trace_limit]
    matched_windows = _source_context_matched_windows(
        pipeline_events.get("windows", []), pipeline_events["trace"], matched_events
    )
    markers = entry["markers"]
    matched_markers = {
        label: markers[label] for label in marker_labels if markers.get(label)
    }
    contextual_barrier_count = sum(
        matched_markers.get(label, 0) for label in _EPILOGUE_BARRIER_CONTEXT_MARKERS
    )
    generic_barrier_count = matched_markers.get("epilogue_barrier_wait", 0)
    if contextual_barrier_count and generic_barrier_count:
        # Context markers are only emitted alongside a generic barrier marker.
        generic_only_count = generic_barrier_count - contextual_barrier_count
        if generic_only_count > 0:
            matched_markers["epilogue_barrier_wait"] = generic_only_count
        else:
            matched_markers.pop("epilogue_barrier_wait", None)
    contextual_barrier_lines = {
        trace_entry["line"]
        for trace_entry in entry["marker_trace"]
        if trace_entry["marker"] in _EPILOGUE_BARRIER_CONTEXT_MARKERS
    }
    marker_trace = [
        trace_entry
        for trace_entry in entry["marker_trace"]
        if trace_entry["marker"] in matched_markers
        and not (
            trace_entry["marker"] == "epilogue_barrier_wait"
            and trace_entry["line"] in contextual_barrier_lines
        )
    ]
    if contextual_barrier_lines:
        marker_trace.sort(
            key=lambda trace_entry: (
                trace_entry["marker"] not in _EPILOGUE_BARRIER_CONTEXT_MARKERS,
                trace_entry["line"],
            )
        )
    marker_trace = marker_trace[:marker_trace_limit]
    score = sum(matched_events.values()) + sum(matched_markers.values())
    if score == 0:
        return None
    coverage = len(matched_events) + len(matched_markers)
    event_groups = _source_context_event_groups(
        candidate_events,
        matched_events,
        matched_markers,
    )

    return {
        "target": entry["target"],
        "source": entry["source"],
        "line_start": entry["line_start"],
        "line_end": entry["line_end"],
        "score": score,
        "coverage": coverage,
        "matched_events": matched_events,
        "event_groups": event_groups,
        "event_sites": event_sites,
        "epilogue_subregions": _source_context_epilogue_subregions(
            matched_events,
            matched_markers,
            matched_windows,
        ),
        "event_trace": event_trace,
        "windows": matched_windows,
        "matched_markers": matched_markers,
        "marker_trace": marker_trace,
        "role": entry["role"],
        "symbol": entry["symbol"],
    }


def _source_context_event_groups(
    candidate_events: list[str],
    matched_events: dict[str, int],
    matched_markers: dict[str, int],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for event in candidate_events:
        event_labels = _WAIT_SITE_EVENT_ALIASES.get(event, (event,))
        marker_labels = _WAIT_SITE_MARKER_ALIASES.get(event, ())
        event_hits = {
            label: matched_events[label]
            for label in event_labels
            if label in matched_events
        }
        marker_hits = {
            label: matched_markers[label]
            for label in marker_labels
            if label in matched_markers
        }
        if not event_hits and not marker_hits:
            continue
        groups.append(
            {
                "label": event,
                "events": event_hits,
                "markers": marker_hits,
                "score": sum(event_hits.values()) + sum(marker_hits.values()),
                "coverage": len(event_hits) + len(marker_hits),
            }
        )
    groups.sort(key=lambda row: (-row["coverage"], -row["score"], row["label"]))
    return groups


def _wait_site_source_context_candidates(
    entries: list[dict[str, Any]],
    candidate_events: list[str],
    *,
    limit: int = _SOURCE_CONTEXT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    matches = [
        match
        for entry in entries
        if (match := _source_context_entry_match(entry, candidate_events)) is not None
    ]
    matches.sort(
        key=lambda row: (
            -row["coverage"],
            -row["score"],
            row["source"],
            row["line_start"],
        )
    )
    return matches[:limit]


def _is_epilogue_token_context_match(match: dict[str, Any]) -> bool:
    return any(
        match["matched_events"].get(event)
        for event in _EPILOGUE_TOKEN_CONTEXT_ANCHOR_EVENTS
    ) or any(
        match["matched_markers"].get(marker)
        for marker in _EPILOGUE_BARRIER_CONTEXT_MARKERS
    )


def _epilogue_token_topology(entries: list[dict[str, Any]]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    complete_windows = 0
    partial_windows = 0
    for entry in entries:
        match = _source_context_entry_match(entry, list(_EPILOGUE_TOKEN_QUERY_EVENTS))
        if match is not None and _is_epilogue_token_context_match(match):
            contexts.append(match)
            all_windows = _source_context_matched_windows(
                entry["pipeline_events"].get("windows", []),
                entry["pipeline_events"]["trace"],
                match["matched_events"],
                limit=None,
            )
            seen_window_keys: set[tuple[str, frozenset[int]]] = set()
            for window in all_windows:
                if window["name"] != _EPILOGUE_STORE_WINDOW_NAME:
                    continue
                window_key = (
                    window["name"],
                    frozenset(event["event_index"] for event in window["events"]),
                )
                if window_key in seen_window_keys:
                    continue
                seen_window_keys.add(window_key)
                if window["complete"]:
                    complete_windows += 1
                else:
                    partial_windows += 1
    contexts.sort(
        key=lambda row: (
            -row["coverage"],
            -row["score"],
            row["source"],
            row["line_start"],
        )
    )
    event_totals = dict.fromkeys(_EPILOGUE_TOKEN_EVENTS, 0)
    marker_totals = dict.fromkeys(
        (
            "epilogue_barrier_wait",
            *_EPILOGUE_BARRIER_CONTEXT_MARKERS,
        ),
        0,
    )
    event_site_totals: dict[str, int] = {}
    for context in contexts:
        for label in _EPILOGUE_TOKEN_EVENTS:
            event_totals[label] += context["matched_events"].get(label, 0)
        for label in marker_totals:
            marker_totals[label] += context["matched_markers"].get(label, 0)
        for event_site in context["event_sites"]:
            event = event_site["event"]
            site = event_site["site"]
            if event not in _EPILOGUE_TOKEN_EVENTS or not site:
                continue
            label = f"{event}@{site}"
            event_site_totals[label] = (
                event_site_totals.get(label, 0) + event_site["count"]
            )

    event_site_rows = [
        {"label": label, "count": count} for label, count in event_site_totals.items()
    ]
    event_site_rows.sort(key=lambda row: (-row["count"], row["label"]))
    return {
        "events": {label: count for label, count in event_totals.items() if count},
        "markers": {label: count for label, count in marker_totals.items() if count},
        "event_sites": event_site_rows,
        "window_counts": {"complete": complete_windows, "partial": partial_windows},
        "contexts": contexts[:_EPILOGUE_TOKEN_CONTEXT_LIMIT],
    }


def _helion_source_context_entries(code: str) -> list[dict[str, Any]]:
    return [
        {
            "target": "helion",
            "source": "generated_helion.py",
            "role": role["role"],
            "symbol": None,
            "line_start": role["line_start"],
            "line_end": role["line_end"],
            "markers": role["markers"],
            "marker_trace": role["marker_trace"],
            "pipeline_events": role["pipeline_events"],
        }
        for role in _helion_role_local_loop_summaries(
            code,
            trace_limit=None,
            marker_trace_limit=None,
            window_limit=None,
        )
    ]


def _quack_source_context_entries(sources: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "target": "quack",
            "source": symbol["source"],
            "role": None,
            "symbol": symbol["symbol"],
            "line_start": symbol["line_start"],
            "line_end": symbol["line_end"],
            "markers": symbol["markers"],
            "marker_trace": symbol["marker_trace"],
            "pipeline_events": symbol["pipeline_events"],
        }
        for symbol in _source_symbol_marker_summaries(
            sources,
            trace_limit=None,
            marker_trace_limit=None,
            window_limit=None,
        )
    ]


def _ncu_source_context_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if _is_quack_ncu_target(args.ncu_target_config):
        quack_sources = _quack_codegen_sources(
            Path(args.quack_path or QUACK_PATH).expanduser()
        )
        return _quack_source_context_entries(quack_sources)
    prepared = _prepare_helion(_ncu_target_helion_args(args))
    return _helion_source_context_entries(prepared.code)


def _has_ncu_wait_sites(parsed_source: dict[str, Any]) -> bool:
    summaries = [*parsed_source["kernels"], parsed_source["aggregate"]]
    return any(
        bool(wait_sites)
        for summary in summaries
        for wait_sites in summary["wait_sites"].values()
    )


def _annotate_ncu_wait_sites_with_source_context(
    parsed_source: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    summaries = [*parsed_source["kernels"], parsed_source["aggregate"]]
    for summary in summaries:
        for wait_sites in summary["wait_sites"].values():
            for site in wait_sites:
                site["source_context"] = _wait_site_source_context_candidates(
                    entries,
                    site["candidate_events"],
                )
            _annotate_ncu_sync_event_site_candidates(wait_sites)


def _format_pipeline_event_ref(entry: dict[str, Any]) -> str:
    site = entry["site"]
    suffix = f"@{site}" if site else ""
    return f"{entry['line']}:{entry['event']}{suffix}"


def _format_pipeline_windows(windows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for window in windows[:4]:
        events = ">".join(
            _format_pipeline_event_ref(entry) for entry in window["events"]
        )
        status = "ok" if window["complete"] else f"missing={window['missing_events']}"
        parts.append(
            f"{window['name']}({events}; span="
            f"{window['line_span']}/{window['event_span']}; {status})"
        )
    return "; ".join(parts)


def _format_source_context_event_sites(sites: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{site['event']}@{site['site']}={site['count']}"
        for site in sites[:_SOURCE_CONTEXT_PRINT_EVENT_SITES]
    )


def _format_source_context_event_groups(groups: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for group in groups[:_SOURCE_CONTEXT_PRINT_EVENT_GROUPS]:
        hits = [
            *(f"{label}:{value:g}" for label, value in group["events"].items()),
            *(f"{label}:{value:g}" for label, value in group["markers"].items()),
        ]
        parts.append(f"{group['label']}=>{','.join(hits)}")
    return "; ".join(parts)


def _format_epilogue_token_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in counts.items())


def _format_epilogue_token_sites(sites: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{site['label']}={site['count']}"
        for site in sites[:_EPILOGUE_TOKEN_PRINT_SITES]
    )


def _format_source_context_location(context: dict[str, Any]) -> str:
    location = f"{context['source']}:{context['line_start']}-{context['line_end']}"
    if context["role"] is not None:
        return f"{location} role={context['role']}"
    return f"{location} {context['symbol']}"


def _format_ncu_branch_paths(branch_paths: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for path in branch_paths[:_TWO_CTA_NCU_WAIT_SITE_BRANCH_PRINT_LIMIT]:
        target = path["target_address"] or "?"
        fallthrough = (
            f"{path['fallthrough_address']}:{path['fallthrough_opcode']}"
            if path["fallthrough_address"]
            else "?"
        )
        parts.append(
            f"{path['branch_address']}->{target}:{path['target_relation']} "
            f"fallthrough={fallthrough}"
        )
    return "; ".join(parts)


def _ncu_source_row_address_sort_key(row: dict[str, Any]) -> tuple[bool, int, str]:
    address_value = _sass_address_value(row["address"])
    return (address_value is None, address_value or 0, row["address"])


def _format_ncu_wait_site_sass_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> str:
    parts: list[str] = []
    for row in rows[:limit]:
        source = " ".join(row["source"].split())
        parts.append(f"{row['address']}:{row['value']:g}:{source}")
    return "; ".join(parts)


def _format_ncu_wait_site_hot_rows(rows: list[dict[str, Any]]) -> str:
    ranked_rows = sorted(
        rows,
        key=lambda row: (-row["value"], *_ncu_source_row_address_sort_key(row)),
    )
    return _format_ncu_wait_site_sass_rows(
        ranked_rows,
        limit=_TWO_CTA_NCU_WAIT_SITE_SASS_PRINT_LIMIT,
    )


def _format_ncu_wait_site_loop_rows(rows: list[dict[str, Any]]) -> str:
    wait_rows = [row for row in rows if _is_sass_wait_site_row(row)]
    wait_rows.sort(key=_ncu_source_row_address_sort_key)
    return _format_ncu_wait_site_sass_rows(
        wait_rows,
        limit=_TWO_CTA_NCU_WAIT_SITE_LOOP_SASS_PRINT_LIMIT,
    )


def _format_ncu_sync_states(states: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for state in states[:_TWO_CTA_NCU_WAIT_SITE_SYNC_PRINT_LIMIT]:
        parts.append(
            f"{state['address']}:{state['barrier']},{state['phase']}:"
            f"{state['value']:g}:{state['opcode']}"
        )
    return "; ".join(parts)


def _format_ncu_sync_register_provenance(
    provenance: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for state in provenance[:_TWO_CTA_NCU_SYNC_PROVENANCE_PRINT_LIMIT]:
        rows = []
        for row in state["rows"]:
            writes = "/".join(row["writes"])
            reads = "/".join(row["reads"]) or "-"
            rows.append(f"{row['address']}:{writes}<-{row['opcode']}({reads})")
        hidden_rows = state["definition_count"] - len(state["rows"])
        if hidden_rows > 0:
            rows.append(f"+{hidden_rows} more")
        missing = "/".join(state["unresolved_registers"])
        status = f"{state['status']}(missing={missing})" if missing else state["status"]
        parts.append(
            f"{state['key']}@{state['address']}:{status}:"
            f"{','.join(rows) if rows else '-'}"
        )
    return "; ".join(parts)


def _format_ncu_sync_related_wait_sites(
    related_sites: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for site in related_sites[:_TWO_CTA_NCU_WAIT_SITE_RELATED_PRINT_LIMIT]:
        focus = f"/{site['focus_wait_site']}" if site["focus_wait_site"] else ""
        events = ",".join(site["candidate_events"][:3])
        parts.append(
            f"{site['key']}->{site['address_start']}:{site['inferred_wait_site']}"
            f"{focus}:{site['value']:g}:events={events}"
        )
    return "; ".join(parts)


def _format_ncu_sync_event_sites(sync_event_sites: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for sync_site in sync_event_sites[:_TWO_CTA_NCU_WAIT_SITE_SYNC_PRINT_LIMIT]:
        event_sites = ",".join(
            f"{site['event']}@{site['site']}:{site['count']:g}"
            for site in sync_site["event_sites"]
        )
        parts.append(f"{sync_site['key']}:{sync_site['status']}:{event_sites or '-'}")
    return "; ".join(parts)


def _format_sass_line_info(
    line_info_file: str | None,
    line_info_line: int | None,
) -> str | None:
    if line_info_file is None or line_info_line is None:
        return None
    return f"{line_info_file}:{line_info_line}"


def _rank_ncu_wait_topology_totals(
    totals: dict[str, dict[str, float]],
    *,
    total: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "label": label,
            "value": values["value"],
            "pct": values["value"] / total * 100 if total else 0.0,
            "site_count": int(values["site_count"]),
        }
        for label, values in totals.items()
        if values["value"] > 0
    ]
    rows.sort(key=lambda row: (-row["value"], -row["site_count"], row["label"]))
    return rows


def _add_ncu_wait_topology_value(
    totals: dict[str, dict[str, float]],
    label: str,
    value: float,
) -> None:
    entry = totals.setdefault(label, {"value": 0.0, "site_count": 0.0})
    entry["value"] += value
    entry["site_count"] += 1


def _ncu_wait_site_topology(
    wait_sites: list[dict[str, Any]],
    *,
    total: float,
) -> dict[str, list[dict[str, Any]]]:
    wait_totals: dict[str, dict[str, float]] = {}
    window_totals: dict[str, dict[str, float]] = {}
    event_site_totals: dict[str, dict[str, float]] = {}
    sync_status_totals: dict[str, dict[str, float]] = {}
    for site in wait_sites:
        value = site["value"]
        wait_label = site["focus_wait_site"] or site["inferred_wait_site"]
        _add_ncu_wait_topology_value(wait_totals, wait_label, value)

        context = site["source_context"][0] if site["source_context"] else None
        if context is not None:
            if context["windows"]:
                window = context["windows"][0]
                status = "ok" if window["complete"] else "partial"
                window_label = f"{context['target']}:{window['name']}:{status}"
                _add_ncu_wait_topology_value(window_totals, window_label, value)
            seen_event_sites: set[str] = set()
            for event_site in context["event_sites"]:
                if not event_site["site"]:
                    continue
                event_label = f"{event_site['event']}@{event_site['site']}"
                if event_label in seen_event_sites:
                    continue
                seen_event_sites.add(event_label)
                _add_ncu_wait_topology_value(event_site_totals, event_label, value)

        seen_sync_statuses = {
            sync_site["status"] for sync_site in site["sync_event_sites"]
        }
        for status in seen_sync_statuses:
            _add_ncu_wait_topology_value(sync_status_totals, status, value)

    return {
        "wait_sites": _rank_ncu_wait_topology_totals(wait_totals, total=total),
        "source_windows": _rank_ncu_wait_topology_totals(
            window_totals,
            total=total,
        ),
        "event_sites": _rank_ncu_wait_topology_totals(
            event_site_totals,
            total=total,
        ),
        "sync_statuses": _rank_ncu_wait_topology_totals(
            sync_status_totals,
            total=total,
        ),
    }


def _label_counts(labels: Iterable[str]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for label in labels:
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return tuple(f"{label}:{counts[label]}" for label in sorted(counts))


def _ranked_label_counts(labels: Iterable[str]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for label in labels:
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return tuple(
        f"{label}:{count}"
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _ncu_sass_topology_source_label(context: dict[str, Any]) -> str:
    if context["role"] is not None:
        source_label = context["role"]
    else:
        source_label = context["symbol"]
    if context["windows"]:
        window = context["windows"][0]
        status = "ok" if window["complete"] else "partial"
        return f"{context['target']}:{source_label}:{window['name']}:{status}"
    return f"{context['target']}:{source_label}:no_window"


def _ncu_sass_topology_event_site_labels(site: dict[str, Any]) -> tuple[str, ...]:
    event_sites = _sync_event_site_candidates([site])
    return tuple(
        f"{event_site['event']}@{event_site['site']}"
        for event_site in event_sites[:_SOURCE_CONTEXT_PRINT_EVENT_SITES]
        if event_site["site"]
    )


def _ncu_sass_topology_wait_opcodes(site: dict[str, Any]) -> tuple[str, ...]:
    rows = [row for row in site["context_rows"] if _is_sass_wait_site_row(row)]
    rows.sort(key=_ncu_source_row_address_sort_key)
    opcodes: list[str] = []
    for row in rows:
        opcode = row["opcode"]
        if opcode and (not opcodes or opcode != opcodes[-1]):
            opcodes.append(opcode)
    return tuple(opcodes)


def _ncu_sass_topology_sync_opcodes(site: dict[str, Any]) -> tuple[str, ...]:
    sync_opcodes = [state["opcode"] for state in site["sync_states"]]
    return _label_counts(sync_opcodes)


def _ncu_sass_topology_branch_shape(site: dict[str, Any]) -> tuple[str, ...]:
    branch_paths = site["branch_paths"]
    if not branch_paths:
        return ()
    relation_counts = _label_counts(path["target_relation"] for path in branch_paths)
    current_targets = sum(
        1 for path in branch_paths if path["targets_current_wait_site"]
    )
    reachable_fallthroughs = sum(
        1 for path in branch_paths if path["fallthrough_reachable"]
    )
    extras = [
        f"targets_current:{current_targets}",
        f"reachable_fallthrough:{reachable_fallthroughs}",
    ]
    return (*relation_counts, *extras)


def _ncu_sass_topology_branch_edges(site: dict[str, Any]) -> tuple[str, ...]:
    return _ranked_label_counts(
        f"{path['branch_opcode']}->{path['target_relation']}:"
        f"{path['target_opcode'] or '?'}|{path['fallthrough_opcode'] or '?'}"
        for path in site["branch_paths"]
    )


def _ncu_sass_topology_source_windows(site: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _ncu_sass_topology_source_label(context)
        for context in site["source_context"][:_TWO_CTA_NCU_SASS_TOPOLOGY_CONTEXT_LIMIT]
    )


def _ncu_sass_topology_epilogue_subregions(site: dict[str, Any]) -> tuple[str, ...]:
    subregions: list[str] = []
    for context in site["source_context"]:
        subregions.extend(context["epilogue_subregions"])
    return tuple(dict.fromkeys(subregions))


def _ncu_sass_wait_topology(
    wait_sites: list[dict[str, Any]],
    *,
    total: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for site in wait_sites:
        wait_label = site["focus_wait_site"] or site["inferred_wait_site"]
        loop_ops = _ncu_sass_topology_wait_opcodes(site)
        sync_ops = _ncu_sass_topology_sync_opcodes(site)
        branch_shape = _ncu_sass_topology_branch_shape(site)
        branch_edges = _ncu_sass_topology_branch_edges(site)
        source_windows = _ncu_sass_topology_source_windows(site)
        event_sites = _ncu_sass_topology_event_site_labels(site)
        epilogue_subregions = _ncu_sass_topology_epilogue_subregions(site)
        key = (
            wait_label,
            loop_ops,
            sync_ops,
            branch_shape,
            branch_edges,
            source_windows,
            event_sites,
            epilogue_subregions,
        )
        group = groups.setdefault(
            key,
            {
                "wait": wait_label,
                "value": 0.0,
                "site_count": 0,
                "loop_ops": loop_ops,
                "sync_ops": sync_ops,
                "branch_shape": branch_shape,
                "branch_edges": branch_edges,
                "source_windows": source_windows,
                "event_sites": event_sites,
                "epilogue_subregions": epilogue_subregions,
            },
        )
        group["value"] += site["value"]
        group["site_count"] += 1

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        row = dict(group)
        row["pct"] = row["value"] / total * 100 if total else 0.0
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["value"],
            -row["site_count"],
            row["wait"],
            row["loop_ops"],
            row["sync_ops"],
            row["branch_shape"],
            row["branch_edges"],
            row["source_windows"],
            row["epilogue_subregions"],
        )
    )
    return rows


def _ncu_sync_key_topology(
    wait_sites: list[dict[str, Any]],
    *,
    total: float,
) -> list[dict[str, Any]]:
    # A wait-site group can mention more than one sync operand; this reports
    # per-key topology, so per-key values are not exclusive totals.
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for site in wait_sites:
        value = site["value"]
        wait_label = site["focus_wait_site"] or site["inferred_wait_site"]
        scoped_keys = list(
            dict.fromkeys(
                (site["source_block"], state["key"]) for state in site["sync_states"]
            )
        )
        for source_block, key in scoped_keys:
            group = groups.setdefault(
                (source_block, key),
                {
                    "source_block": source_block,
                    "key": key,
                    "value": 0.0,
                    "site_count": 0,
                    "waits": {},
                    "statuses": {},
                    "event_sites": {},
                    "sites": [],
                },
            )
            group["value"] += value
            group["site_count"] += 1
            _add_ncu_wait_topology_value(group["waits"], wait_label, value)

            seen_statuses: set[str] = set()
            seen_event_sites: set[str] = set()
            for sync_site in site["sync_event_sites"]:
                if sync_site["source_block"] != source_block or sync_site["key"] != key:
                    continue
                seen_statuses.add(sync_site["status"])
                for event_site in sync_site["event_sites"]:
                    if not event_site["site"]:
                        continue
                    event_label = f"{event_site['event']}@{event_site['site']}"
                    seen_event_sites.add(event_label)
            for status in seen_statuses:
                _add_ncu_wait_topology_value(group["statuses"], status, value)
            for event_label in seen_event_sites:
                _add_ncu_wait_topology_value(group["event_sites"], event_label, value)
            site_event_sites = _sync_event_site_candidates([site])
            event_site_labels = [
                f"{event_site['event']}@{event_site['site']}"
                for event_site in site_event_sites
                if event_site["site"]
            ]
            group["sites"].append(
                {
                    "address_start": site["address_start"],
                    "address_end": site["address_end"],
                    "value": value,
                    "wait": wait_label,
                    "focus_wait_site": site["focus_wait_site"],
                    "status": _sync_event_site_status(site_event_sites),
                    "event_sites": event_site_labels,
                }
            )

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        value = group["value"]
        site_rows = []
        for site in group["sites"]:
            site_row = dict(site)
            site_row["pct"] = site["value"] / value * 100 if value else 0.0
            site_rows.append(site_row)
        site_rows.sort(key=_ncu_source_wait_site_sort_key)
        rows.append(
            {
                "source_block": group["source_block"],
                "key": group["key"],
                "value": value,
                "pct": value / total * 100 if total else 0.0,
                "site_count": group["site_count"],
                "waits": _rank_ncu_wait_topology_totals(
                    group["waits"],
                    total=value,
                ),
                "statuses": _rank_ncu_wait_topology_totals(
                    group["statuses"],
                    total=value,
                ),
                "event_sites": _rank_ncu_wait_topology_totals(
                    group["event_sites"],
                    total=value,
                ),
                "sites": site_rows,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["value"],
            -row["site_count"],
            row["source_block"],
            row["key"],
        )
    )
    return rows


def _format_ncu_wait_topology_rows(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows[:_TWO_CTA_NCU_WAIT_TOPOLOGY_PRINT_LIMIT]:
        site_count = row["site_count"]
        site_label = "site" if site_count == 1 else "sites"
        parts.append(
            f"{row['label']}={row['value']:g} ({row['pct']:.1f}%, "
            f"{site_count} {site_label})"
        )
    return "; ".join(parts)


def _format_ncu_sass_topology_rows(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows[:_TWO_CTA_NCU_SASS_TOPOLOGY_PRINT_LIMIT]:
        site_count = row["site_count"]
        site_label = "site" if site_count == 1 else "sites"
        loop_ops = ">".join(row["loop_ops"]) or "-"
        sync_ops = ",".join(row["sync_ops"]) or "-"
        branches = ",".join(row["branch_shape"]) or "-"
        branch_edges = (
            ",".join(row["branch_edges"][:_TWO_CTA_NCU_SASS_TOPOLOGY_BRANCH_EDGE_LIMIT])
            or "-"
        )
        sources = ",".join(row["source_windows"]) or "-"
        events = ",".join(row["event_sites"]) or "-"
        subregions = ",".join(row["epilogue_subregions"]) or "-"
        parts.append(
            f"{row['wait']}={row['value']:g} ({row['pct']:.1f}%, "
            f"{site_count} {site_label}; loop={loop_ops}; sync={sync_ops}; "
            f"branches={branches}; edges={branch_edges}; sources={sources}; "
            f"events={events}; subregions={subregions})"
        )
    return "; ".join(parts)


def _format_ncu_sync_key_site_rows(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows[:_TWO_CTA_NCU_SYNC_KEY_SITE_PRINT_LIMIT]:
        focus = f"/{row['focus_wait_site']}" if row["focus_wait_site"] else ""
        event_sites = ",".join(row["event_sites"][:_SOURCE_CONTEXT_PRINT_EVENT_SITES])
        parts.append(
            f"{row['address_start']}..{row['address_end']}:{row['wait']}{focus}="
            f"{row['value']:g} ({row['pct']:.1f}%, {row['status']}, "
            f"events={event_sites or '-'})"
        )
    return "; ".join(parts)


def _format_ncu_sync_key_topology_rows(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows[:_TWO_CTA_NCU_SYNC_KEY_TOPOLOGY_PRINT_LIMIT]:
        site_count = row["site_count"]
        site_label = "site" if site_count == 1 else "sites"
        waits = _format_ncu_wait_topology_rows(row["waits"])
        statuses = _format_ncu_wait_topology_rows(row["statuses"])
        event_sites = _format_ncu_wait_topology_rows(row["event_sites"])
        site_rows = _format_ncu_sync_key_site_rows(row["sites"])
        details = [
            f"waits={waits or '-'}",
            f"status={statuses or '-'}",
            f"events={event_sites or '-'}",
            f"sites={site_rows or '-'}",
        ]
        parts.append(
            f"block{row['source_block']}:{row['key']}={row['value']:g} "
            f"({row['pct']:.1f}%, {site_count} {site_label}; "
            f"{'; '.join(details)})"
        )
    return "; ".join(parts)


def _quack_codegen_sources(quack_root: Path) -> dict[str, str]:
    source_paths = [
        quack_root / relative_path
        for relative_path in _QUACK_CODEGEN_SOURCE_RELATIVE_PATHS
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        expected = ", ".join(
            str(relative_path) for relative_path in _QUACK_CODEGEN_SOURCE_RELATIVE_PATHS
        )
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(
            "--helion-two-cta-codegen-report needs Quack sources under "
            f"{quack_root}. Missing: {missing_text}. Set --quack-path to a "
            f"Quack checkout containing {expected}."
        )
    return {
        str(Path(quack_root.name) / path.relative_to(quack_root)): path.read_text(
            encoding="utf-8"
        )
        for path in source_paths
    }


def _run_helion_two_cta_codegen_report(args: argparse.Namespace) -> dict[str, Any]:
    helion_args = _two_cta_seed_args(args)
    quack_sources = _quack_codegen_sources(
        Path(args.quack_path or QUACK_PATH).expanduser()
    )
    prepared = _prepare_helion(helion_args)
    helion_context_entries = _helion_source_context_entries(prepared.code)
    helion_summary = _source_bundle_summary({"generated_helion.py": prepared.code})
    helion_pipeline_summary = _source_bundle_pipeline_summary(
        {"generated_helion.py": prepared.code}
    )
    helion_role_summaries = _helion_role_local_loop_summaries(prepared.code)
    quack_context_entries = _quack_source_context_entries(quack_sources)
    quack_summary = _source_bundle_summary(quack_sources)
    quack_pipeline_summary = _source_bundle_pipeline_summary(quack_sources)
    quack_symbol_summaries = _source_symbol_marker_summaries(quack_sources)
    marker_delta = {
        label: helion_summary["markers"][label] - quack_summary["markers"][label]
        for label in _SOURCE_MARKER_LABELS
    }
    return {
        "impl": "helion-quack-codegen-report",
        "shape": {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype},
        "helion": {
            "config": repr(prepared.active_config),
            "codegen": prepared.codegen,
            "role_local_loops": helion_role_summaries,
            "pipeline_events": helion_pipeline_summary,
            "epilogue_token_topology": _epilogue_token_topology(helion_context_entries),
            **helion_summary,
        },
        "quack": {
            "sources": list(quack_sources),
            "config": {
                "tile_m": args.quack_tile_m,
                "tile_n": args.quack_tile_n,
                "cluster_m": args.quack_cluster_m,
                "cluster_n": args.quack_cluster_n,
                "persistent": bool(args.quack_persistent),
                "dynamic_persistent": bool(args.quack_dynamic_persistent),
            },
            "marker_symbols": quack_symbol_summaries,
            "pipeline_events": quack_pipeline_summary,
            "epilogue_token_topology": _epilogue_token_topology(quack_context_entries),
            **quack_summary,
        },
        "marker_delta_helion_minus_quack": marker_delta,
    }


def _parse_nvdisasm_sass(disassembly: str) -> list[dict[str, Any]]:
    """Parse ``nvdisasm`` SASS instructions into the NCU-source row shape."""
    rows: list[dict[str, Any]] = []
    source_block = 0
    source_block_name = "cubin"
    saw_text_section = False
    line_info_file: str | None = None
    line_info_line: int | None = None
    for line in disassembly.splitlines():
        section_match = _NVDISASM_TEXT_SECTION_RE.match(line)
        if section_match is not None:
            if saw_text_section:
                source_block += 1
            else:
                saw_text_section = True
            source_block_name = section_match.group("section")
            continue

        line_info_match = _NVDISASM_LINE_INFO_RE.match(line)
        if line_info_match is not None:
            line_info_file = line_info_match.group("file")
            line_info_line = int(line_info_match.group("line"))
            continue

        instruction_match = _NVDISASM_INSTRUCTION_RE.match(line)
        if instruction_match is None:
            continue
        raw_source = instruction_match.group("source").strip()
        opcode = _sass_opcode(raw_source)
        if not opcode or opcode.startswith("."):
            continue
        address = f"0x{int(instruction_match.group('address'), 16):x}"
        sync_occurrences = 1.0 if _sass_sync_state(raw_source) else 0.0
        rows.append(
            {
                "address": address,
                "address_value": _sass_address_value(address),
                "source": raw_source,
                "opcode": opcode,
                "source_kind": _sass_source_kind(opcode),
                "source_block": source_block,
                "source_block_name": source_block_name,
                "line_info_file": line_info_file,
                "line_info_line": line_info_line,
                "line_info": _format_sass_line_info(line_info_file, line_info_line),
                "metrics": {_TWO_CTA_CUBIN_SYNC_METRIC: sync_occurrences},
            }
        )
    return rows


def _cubin_sync_register_provenance(
    rows: list[dict[str, Any]],
    *,
    sync_key: str | None,
) -> list[dict[str, Any]]:
    source_blocks = sorted(
        {row["source_block"] for row in rows if row["address_value"] is not None}
    )
    provenance: list[dict[str, Any]] = []
    for source_block in source_blocks:
        block_rows = [row for row in rows if row["source_block"] == source_block]
        sync_states = _ncu_sync_state_trace(block_rows, _TWO_CTA_CUBIN_SYNC_METRIC)
        if sync_key is not None:
            sync_states = [state for state in sync_states if state["key"] == sync_key]
        block_provenance = _ncu_sync_register_provenance(
            rows,
            sync_states,
            _TWO_CTA_CUBIN_SYNC_METRIC,
            source_block=source_block,
            scan_limit=None,
        )
        states_by_address = {
            (state["key"], state["address"]): state for state in sync_states
        }
        source_block_name = next(
            (row.get("source_block_name", "cubin") for row in block_rows),
            "cubin",
        )
        for state_provenance in block_provenance:
            state = states_by_address[
                (state_provenance["key"], state_provenance["address"])
            ]
            provenance.append(
                {
                    **state_provenance,
                    "source_block": source_block,
                    "source_block_name": source_block_name,
                    "source": state["source"],
                    "opcode": state["opcode"],
                    "line_info": state.get("line_info"),
                }
            )
    provenance.sort(
        key=lambda state: (
            state["source_block"],
            _sass_address_value(state["address"]) or 0,
            state["key"],
        )
    )
    return provenance


def _cubin_sync_state_groups(
    provenance: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for state in provenance:
        group = groups.setdefault(
            state["key"],
            {
                "key": state["key"],
                "count": 0,
                "statuses": {},
                "first_address": state["address"],
                "first_source_block": state["source_block"],
            },
        )
        group["count"] += 1
        group["statuses"][state["status"]] = (
            group["statuses"].get(state["status"], 0) + 1
        )
    result = list(groups.values())
    result.sort(key=lambda group: (-group["count"], group["key"]))
    return result


def _helion_cute_compiled_cubins(prepared: _PreparedHelion) -> list[dict[str, Any]]:
    try:
        cute_kernel = prepared.bound._run.__globals__[_HELION_CUTE_KERNEL_SYMBOL]
    except KeyError as exc:
        raise RuntimeError(
            "--helion-two-cta-cubin-sass-report expected the generated matmul "
            f"kernel global {_HELION_CUTE_KERNEL_SYMBOL!r}; Helion's generated "
            "benchmark wrapper layout may have changed."
        ) from exc
    try:
        launchers = cute_kernel._helion_cute_compiled_launchers
    except AttributeError as exc:
        raise RuntimeError(
            "--helion-two-cta-cubin-sass-report expected the generated CuTe "
            "kernel to expose _helion_cute_compiled_launchers after launch."
        ) from exc
    cubins: list[dict[str, Any]] = []
    for launcher in launchers.values():
        try:
            cubin = launcher._compiled.__cubin__
        except AttributeError as exc:
            raise RuntimeError(
                "--helion-two-cta-cubin-sass-report expected launcher._compiled."
                "__cubin__ on each compiled CuTe launcher."
            ) from exc
        if isinstance(cubin, bytes):
            # Normal CUTE_DSL_KEEP_CUBIN path: the compiled object retains bytes.
            cubin_bytes = cubin
        elif isinstance(cubin, str):
            # Fallback for Cutlass builds that retain a dump path instead.
            cubin_path = Path(cubin)
            if not cubin_path.is_file():
                raise RuntimeError(f"compiled CuTe cubin path does not exist: {cubin}")
            cubin_bytes = cubin_path.read_bytes()
        else:
            raise RuntimeError(
                "compiled CuTe launcher did not retain cubin bytes; "
                "CUTE_DSL_KEEP_CUBIN may have been ignored"
            )
        cubins.append(
            {
                "bytes": cubin_bytes,
                "byte_count": len(cubin_bytes),
            }
        )
    if not cubins:
        raise RuntimeError("no compiled CuTe launchers were found after Helion launch")
    return cubins


def _run_nvdisasm(
    nvdisasm_binary: str,
    cubin_bytes: bytes,
    *,
    cubin_name: str,
    workdir: Path,
    print_line_info: bool = False,
) -> dict[str, Any]:
    cubin_path = workdir / cubin_name
    cubin_path.write_bytes(cubin_bytes)
    command = [nvdisasm_binary]
    if print_line_info:
        command.append("--print-line-info")
    command.append(str(cubin_path))
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "--helion-two-cta-cubin-sass-report requires nvdisasm. "
            f"Could not execute {nvdisasm_binary!r}; pass --nvdisasm-binary."
        ) from exc
    if proc.returncode != 0:
        raise SystemExit(
            "nvdisasm failed for --helion-two-cta-cubin-sass-report "
            f"(rc={proc.returncode}).\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return {"stdout": proc.stdout, "stderr": proc.stderr, "command": command}


def _run_helion_two_cta_cubin_sass_report(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Compile the validated Helion row and analyze full local cubin SASS."""
    helion_args = _ncu_target_helion_args(args)
    with tempfile.TemporaryDirectory(prefix="helion_cute_cubin_") as tmp:
        dump_dir = Path(tmp)
        disasm_dir = dump_dir / "nvdisasm-inputs"
        disasm_dir.mkdir()
        previous_keep_cubin = os.environ.get("CUTE_DSL_KEEP_CUBIN")
        previous_dump_dir = os.environ.get("CUTE_DSL_DUMP_DIR")
        try:
            os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
            os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
            prepared = _prepare_helion(helion_args)
            if args.skip_correctness:
                prepared.bound(*prepared.kernel_args)
            else:
                result = prepared.bound(*prepared.kernel_args)
                _check_close(result, prepared.expected, prepared.dtype)
            torch.cuda.synchronize()
            cubins = _helion_cute_compiled_cubins(prepared)
            disassemblies = [
                _run_nvdisasm(
                    args.nvdisasm_binary,
                    cubin["bytes"],
                    cubin_name=f"helion_sass_report_launcher_{index}.cubin",
                    workdir=disasm_dir,
                    print_line_info=args.helion_cubin_lineinfo,
                )
                for index, cubin in enumerate(cubins)
            ]
        finally:
            if previous_keep_cubin is None:
                os.environ.pop("CUTE_DSL_KEEP_CUBIN", None)
            else:
                os.environ["CUTE_DSL_KEEP_CUBIN"] = previous_keep_cubin
            if previous_dump_dir is None:
                os.environ.pop("CUTE_DSL_DUMP_DIR", None)
            else:
                os.environ["CUTE_DSL_DUMP_DIR"] = previous_dump_dir

    rows: list[dict[str, Any]] = []
    next_source_block = 0
    for cubin_index, disassembly in enumerate(disassemblies):
        parsed_rows = _parse_nvdisasm_sass(disassembly["stdout"])
        source_block_offset = next_source_block
        for row in parsed_rows:
            row["source_block"] += source_block_offset
            row["source_block_name"] = (
                f"launcher_{cubin_index}:{row['source_block_name']}"
            )
        if parsed_rows:
            next_source_block = max(row["source_block"] for row in parsed_rows) + 1
        rows.extend(parsed_rows)
    line_info_sources = sorted(
        {row["line_info_file"] for row in rows if row.get("line_info_file") is not None}
    )
    provenance = _cubin_sync_register_provenance(rows, sync_key=args.cubin_sync_key)
    if not provenance:
        focus_text = (
            f" for sync key {args.cubin_sync_key!r}" if args.cubin_sync_key else ""
        )
        raise SystemExit(
            "nvdisasm completed but no SYNCS.PHASECHK sync states were parsed"
            f"{focus_text}."
        )
    return {
        "impl": "helion-cute:two-cta-cubin-sass-report",
        "shape": {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype},
        "target_config": args.ncu_target_config,
        "config": repr(prepared.active_config),
        "codegen": prepared.codegen,
        "nvdisasm_binary": args.nvdisasm_binary,
        "nvdisasm_line_info": args.helion_cubin_lineinfo,
        "sync_key": args.cubin_sync_key,
        "cubin_count": len(cubins),
        "cubin_byte_counts": [cubin["byte_count"] for cubin in cubins],
        "sass_row_count": len(rows),
        "line_info_row_count": sum(
            1 for row in rows if row.get("line_info_file") is not None
        ),
        "line_info_sources": line_info_sources,
        "sync_state_count": len(provenance),
        "sync_state_groups": _cubin_sync_state_groups(provenance),
        "sync_register_provenance": provenance,
    }


def _run_profiled_ncu_launches(
    fn: Callable[[], object], args: argparse.Namespace
) -> None:
    cudart: Any = torch.cuda.cudart()
    torch.cuda.synchronize()
    start_status = cudart.cudaProfilerStart()
    if start_status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {start_status}")
    try:
        for _ in range(args.ncu_warmup_launches):
            fn()
        torch.cuda.synchronize()
        for _ in range(args.ncu_profile_launches):
            fn()
        torch.cuda.synchronize()
    finally:
        stop_status = cudart.cudaProfilerStop()
        if stop_status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")


def _run_helion_two_cta_ncu_target(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fixed diagnostic row for Nsight Compute collection."""
    if _is_quack_ncu_target(args.ncu_target_config):
        quack_args = _ncu_target_quack_args(args)
        prepared_quack = _prepare_quack_direct(
            quack_args, _quack_direct_config_from_args(quack_args)
        )
        if not args.skip_correctness:
            _check_close(
                prepared_quack.run(), prepared_quack.expected, prepared_quack.dtype
            )

        _run_profiled_ncu_launches(prepared_quack.run, args)
        return {
            "impl": f"quack-direct:{args.ncu_target_config}:ncu-target",
            "target_config": args.ncu_target_config,
            "shape": {
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "dtype": args.dtype,
                "epilogue": args.epilogue,
            },
            "config": prepared_quack.config,
            "warmup_launches": args.ncu_warmup_launches,
            "profile_launches": args.ncu_profile_launches,
        }

    helion_args = _ncu_target_helion_args(args)
    prepared = _prepare_helion(helion_args)
    if not args.skip_correctness:
        _check_close(
            prepared.bound(*prepared.kernel_args), prepared.expected, prepared.dtype
        )

    _run_profiled_ncu_launches(lambda: prepared.bound(*prepared.kernel_args), args)
    return {
        "impl": f"helion-cute:{args.ncu_target_config}:ncu-target",
        "target_config": args.ncu_target_config,
        "shape": {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype},
        "config": repr(prepared.active_config),
        "codegen": prepared.codegen,
        "warmup_launches": args.ncu_warmup_launches,
        "profile_launches": args.ncu_profile_launches,
    }


def _build_two_cta_ncu_profile_command(
    args: argparse.Namespace,
    *,
    metrics: list[str],
    page: str,
    print_source: str | None = None,
) -> list[str]:
    quack_args = (
        _ncu_target_quack_args(args)
        if _is_quack_ncu_target(args.ncu_target_config)
        else args
    )
    cmd = [
        args.ncu_binary,
        "--csv",
        "--page",
        page,
    ]
    if print_source is not None:
        cmd.extend(["--print-source", print_source])
    cmd.extend(
        [
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--launch-skip",
            str(args.ncu_warmup_launches),
            "--launch-count",
            str(args.ncu_profile_launches),
            "--metrics",
            ",".join(metrics),
            sys.executable,
            str(Path(__file__).resolve()),
            "--helion-two-cta-ncu-target",
            "--helion-backend",
            "cute",
            "--ncu-target-config",
            args.ncu_target_config,
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--epilogue",
            args.epilogue,
            "--dtype",
            args.dtype,
            "--seed",
            str(args.seed),
            "--skip-correctness",
            "1",
            "--ncu-warmup-launches",
            str(args.ncu_warmup_launches),
            "--ncu-profile-launches",
            str(args.ncu_profile_launches),
            "--json",
        ]
    )
    if quack_args.quack_path:
        cmd.extend(["--quack-path", quack_args.quack_path])
    cmd.extend(
        [
            "--quack-pingpong",
            str(int(quack_args.quack_pingpong)),
            "--quack-persistent",
            str(int(quack_args.quack_persistent)),
            "--quack-dynamic-persistent",
            str(int(quack_args.quack_dynamic_persistent)),
            "--quack-max-swizzle-size",
            str(quack_args.quack_max_swizzle_size),
        ]
    )
    # Forward Helion strategy / persistence-model flags. Without these the
    # NCU subprocess silently regresses to the default static-path baseline.
    if not _is_quack_ncu_target(args.ncu_target_config):
        if args.helion_strategy is not None:
            cmd.extend(["--helion-strategy", args.helion_strategy])
        if args.helion_c_input_warps is not None:
            cmd.extend(["--helion-c-input-warps", str(args.helion_c_input_warps)])
        if args.helion_persistence_model is not None:
            cmd.extend(["--helion-persistence-model", args.helion_persistence_model])
        cmd.extend(_helion_diagnostic_flag_args(args))
    return cmd


def _build_two_cta_ncu_command(args: argparse.Namespace) -> list[str]:
    return _build_two_cta_ncu_profile_command(
        args,
        metrics=args.ncu_metrics,
        page="raw",
    )


def _ncu_source_collection_metrics(args: argparse.Namespace) -> list[str]:
    metrics: list[str] = []
    for metric in [*args.ncu_source_metrics, *args.ncu_source_provenance_metrics]:
        if metric not in metrics:
            metrics.append(metric)
    return metrics


def _build_two_cta_ncu_source_command(args: argparse.Namespace) -> list[str]:
    return _build_two_cta_ncu_profile_command(
        args,
        metrics=_ncu_source_collection_metrics(args),
        page="source",
        print_source="sass",
    )


def _parse_ncu_counter(value: str) -> float:
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"nan", "n/a"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _sass_opcode(source: str) -> str:
    tokens = source.strip().split()
    while tokens and tokens[0].startswith("@"):
        tokens.pop(0)
    if not tokens:
        return ""
    return tokens[0].rstrip(",")


def _sass_address_value(address: str) -> int | None:
    try:
        return int(address.strip(), 0)
    except ValueError:
        return None


def _sass_branch_target_address(source: str) -> str:
    # This diagnostic parser recognizes absolute hex targets, not relative BRA forms.
    if not _sass_opcode(source).startswith("BRA"):
        return ""
    for token in source.replace(",", " ").replace(";", " ").split():
        cleaned = token.strip("`()[]")
        if cleaned.startswith(("0x", "0X")):
            return cleaned
    return ""


def _sass_sync_state(source: str) -> dict[str, str]:
    tokens = source.replace(",", " ").replace(";", " ").split()
    if tokens and tokens[0].startswith("@"):
        tokens = tokens[1:]
    if len(tokens) < 4 or not tokens[0].startswith("SYNCS.PHASECHK"):
        return {}
    barrier = tokens[2]
    phase = tokens[3]
    if not (barrier.startswith("[") and barrier.endswith("]")):
        return {}
    return {
        "opcode": tokens[0],
        "predicate": tokens[1],
        "barrier": barrier,
        "phase": phase,
        "key": f"{barrier},{phase}",
    }


def _sass_operands(source: str) -> tuple[str, ...]:
    tokens = source.strip().rstrip(";")
    while tokens.startswith("@"):
        _predicate, _separator, tokens = tokens.partition(" ")
        tokens = tokens.lstrip()
    _opcode, separator, operand_text = tokens.partition(" ")
    if not separator:
        return ()
    return tuple(operand.strip() for operand in operand_text.split(",") if operand)


def _sass_registers(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            register
            for register in _SASS_REGISTER_RE.findall(text)
            if register not in _SASS_ZERO_REGISTERS
        )
    )


def _sass_write_registers(source: str) -> tuple[str, ...]:
    operands = _sass_operands(source)
    if not operands:
        return ()
    destination = operands[0].strip()
    # This tracks the common one-destination form, including suffixes such as
    # R1.cc or R1.reuse. Pair destinations are intentionally not modeled.
    match = _SASS_REGISTER_DESTINATION_RE.fullmatch(destination)
    if match is not None and match.group(1) not in _SASS_ZERO_REGISTERS:
        return (match.group(1),)
    return ()


def _sass_read_registers(source: str) -> tuple[str, ...]:
    operands = list(_sass_operands(source))
    if _sass_write_registers(source):
        operands = operands[1:]
    return tuple(
        dict.fromkeys(
            register for operand in operands for register in _sass_registers(operand)
        )
    )


def _sass_source_is_predicated(source: str) -> bool:
    tokens = source.strip().split()
    return bool(tokens) and tokens[0].startswith("@")


def _sass_branch_has_reachable_fallthrough(source: str) -> bool:
    # Any SASS predicate spelling, including @!P0, means the next PC can execute.
    return _sass_source_is_predicated(source) and _sass_opcode(source).startswith("BRA")


def _sass_source_kind(opcode: str) -> str:
    if not opcode:
        return "unknown"
    if opcode.startswith("UTCHMMA"):
        return "umma_issue"
    if opcode.startswith("UTMASTG"):
        return "tma_store_issue"
    if opcode.startswith("UTMACMDFLUSH"):
        return "tma_store_flush"
    if opcode.startswith("UTMA"):
        return "tma_load_issue"
    if opcode.startswith("LDTM"):
        return "tmem_load"
    if opcode.startswith(
        (
            "NANOSLEEP",
            "SYNCS",
            "BAR.",
            "MEMBAR",
            "DEPBAR",
            "FENCE.",
            "UCGABAR",
            "UTCBAR",
        )
    ):
        return "barrier_or_wait"
    if opcode.startswith(("LDS", "STS", "ATOMS", "UTCATOM")):
        return "shared_or_atomic"
    if opcode.startswith(("LD", "ST")):
        return "global_or_const_memory"
    if opcode.startswith(
        ("BRA", "CALL", "RET", "EXIT", "PREEXIT", "BSSY", "BSYNC", "WARPSYNC")
    ):
        return "control_flow"
    return "other"


def _is_sass_wait_site_row(row: dict[str, Any]) -> bool:
    opcode = row["opcode"]
    return row["source_kind"] == "barrier_or_wait" or opcode.startswith("BRA")


def _is_sass_wait_loop_opcode(opcode: str) -> bool:
    return opcode.startswith(("NANOSLEEP", "SYNCS", "BAR.", "UCGABAR", "UTCBAR"))


def _is_sass_unconditional_control_flow_terminator(row: dict[str, Any]) -> bool:
    if _sass_source_is_predicated(row["source"]):
        return False
    return row["opcode"].startswith(("BRA", "BRX", "JMX", "JMP", "EXIT", "RET"))


def _ncu_wait_site_focus_result(wait_site: str, evidence: str) -> dict[str, Any]:
    if wait_site == _EPILOGUE_BARRIER_BEFORE_TMA_MARKER:
        return {
            "wait_site": wait_site,
            "candidate_events": [
                _EPILOGUE_BARRIER_BEFORE_TMA_MARKER,
                "async_shared_fence",
                "tma_store_copy",
                "c_producer_commit",
            ],
            "evidence": evidence,
        }
    return {
        "wait_site": wait_site,
        "candidate_events": [
            _EPILOGUE_BARRIER_BEFORE_SMEM_MARKER,
            "c_producer_acquire",
            "reg_to_smem_copy",
            "async_shared_fence",
        ],
        "evidence": evidence,
    }


def _ncu_wait_site_focus_from_resumed_rows(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    smem_evidence: str,
    tma_evidence: str,
    skip_leading_zero_sample_waits: bool,
) -> dict[str, Any] | None:
    resumed = False
    for row in rows:
        opcode = row["opcode"]
        if not opcode:
            continue
        if _is_sass_unconditional_control_flow_terminator(row):
            break
        if _is_sass_wait_loop_opcode(opcode):
            if (
                skip_leading_zero_sample_waits
                and not resumed
                and row["metrics"].get(metric, 0.0) <= 0.0
            ):
                continue
            break
        resumed = True
        if opcode.startswith(("STS", "ATOMS", "UTCATOM", "FENCE.VIEW.ASYNC.S")):
            return _ncu_wait_site_focus_result(
                _EPILOGUE_BARRIER_BEFORE_SMEM_MARKER,
                smem_evidence,
            )
        if opcode.startswith(("UTMASTG", "UTMACMDFLUSH")):
            return _ncu_wait_site_focus_result(
                _EPILOGUE_BARRIER_BEFORE_TMA_MARKER,
                tma_evidence,
            )
    return None


def _infer_sass_wait_site(opcodes: set[str]) -> tuple[str, list[str]]:
    """Map a hot SASS wait loop to the generated events worth inspecting."""
    if any(opcode.startswith("UTCHMMA") for opcode in opcodes):
        return (
            "umma_wait_or_issue",
            ["ab_consumer_wait", "umma_gemm", "acc_producer_commit"],
        )
    # Store-side UTMA variants must stay before the generic UTMA load case.
    if any(
        opcode.startswith(("UTMASTG", "UTMACMDFLUSH", "LDTM")) for opcode in opcodes
    ):
        return (
            "epilogue_store_wait",
            [
                "acc_consumer_wait",
                "c_producer_acquire",
                "tmem_to_reg_copy",
                "reg_to_smem_copy",
                "tma_store_copy",
                "c_producer_commit",
            ],
        )
    if any(opcode.startswith("UTMA") for opcode in opcodes):
        return (
            "tma_load_wait",
            ["ab_producer_acquire", "tma_load_copy", "ab_producer_commit"],
        )
    if any(opcode.startswith(("FENCE.", "MEMBAR", "DEPBAR")) for opcode in opcodes):
        return (
            "async_fence_wait",
            ["async_shared_fence", "producer_commit", "producer_tail", "consumer_wait"],
        )
    if any(
        opcode.startswith(("NANOSLEEP", "SYNCS", "BAR.", "UCGABAR", "UTCBAR"))
        for opcode in opcodes
    ):
        return (
            "pipeline_barrier_wait",
            ["producer_acquire", "consumer_wait", "arrive_and_wait"],
        )
    return ("branch_wait_loop", ["work_tile_loop", "consumer_wait"])


def _ncu_wait_site_focus_from_sass(
    inferred_site: str,
    metric: str,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Use local post-wait SASS to split epilogue barrier wait loops."""
    if inferred_site not in {
        "pipeline_barrier_wait",
        "async_fence_wait",
        "epilogue_store_wait",
    }:
        return None

    before_opcodes = [row["opcode"] for row in before_rows if row["opcode"]]
    # Attribute to the first resumed epilogue action; a later wait loop starts
    # another barrier site and should not lend its TMA store to this one.
    focus = _ncu_wait_site_focus_from_resumed_rows(
        after_rows,
        metric,
        smem_evidence="post_wait_smem_or_fence",
        tma_evidence="post_wait_tma_store",
        skip_leading_zero_sample_waits=True,
    )
    if focus is not None:
        return focus

    if any(opcode.startswith("FENCE.VIEW.ASYNC.S") for opcode in before_opcodes):
        return _ncu_wait_site_focus_result(
            _EPILOGUE_BARRIER_BEFORE_TMA_MARKER,
            "pre_wait_async_shared_fence",
        )

    return None


def _ncu_branch_target_relation(
    *,
    target_value: int | None,
    target_index: int | None,
    branch_index: int,
    group_indices: set[int],
    group_addresses: set[int],
) -> str:
    if target_value is None:
        return "unknown"
    if target_index in group_indices or target_value in group_addresses:
        return "hot_group"
    if target_index is None:
        return "unresolved"
    if target_index < branch_index:
        return "local_backward"
    if target_index > branch_index:
        return "local_forward"
    return "self"


def _ncu_branch_targets_current_wait_site(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    target_value: int | None,
    target_index: int | None,
    group_indices: set[int],
    group_addresses: set[int],
) -> bool:
    if target_index in group_indices or target_value in group_addresses:
        return True
    if target_index is None:
        return False
    row = rows[target_index]
    adjacent_by_address = target_value is not None and any(
        abs(target_value - address) <= _TWO_CTA_NCU_SASS_INSTRUCTION_SIZE
        for address in group_addresses
    )
    if not adjacent_by_address:
        return False
    return row["metrics"].get(metric, 0.0) <= 0.0 and _is_sass_wait_loop_opcode(
        row["opcode"]
    )


def _ncu_source_wait_site_branch_paths(
    rows: list[dict[str, Any]],
    group_indices: list[int],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    group_source_block = rows[group_indices[0]]["source_block"]
    address_to_indices: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if row["source_block"] != group_source_block:
            continue
        if row["address_value"] is None:
            continue
        address_to_indices.setdefault(row["address_value"], []).append(index)
    group_index_set = set(group_indices)
    group_addresses = {
        rows[index]["address_value"]
        for index in group_indices
        if rows[index]["address_value"] is not None
    }
    group_address_values = list(group_addresses)
    group_min_address = min(group_address_values) if group_address_values else None
    group_max_address = max(group_address_values) if group_address_values else None
    relation_priority = {
        "hot_group": 0,
        "local_backward": 1,
        "local_forward": 2,
        "self": 3,
        "unresolved": 4,
        "unknown": 5,
    }
    ranked_paths: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if row["source_block"] != group_source_block:
            continue
        if not row["opcode"].startswith("BRA"):
            continue
        branch_address_value = row["address_value"]
        if (
            branch_address_value is None
            or group_min_address is None
            or group_max_address is None
        ):
            continue
        address_distance = (
            0
            if group_min_address <= branch_address_value <= group_max_address
            else min(
                abs(branch_address_value - group_min_address),
                abs(branch_address_value - group_max_address),
            )
        )
        if address_distance > _TWO_CTA_NCU_WAIT_SITE_BRANCH_ADDRESS_GAP:
            continue
        target_address = _sass_branch_target_address(row["source"])
        target_value = _sass_address_value(target_address) if target_address else None
        target_indices = (
            address_to_indices.get(target_value, []) if target_value is not None else []
        )
        target_index = target_indices[0] if len(target_indices) == 1 else None
        fallthrough_value = branch_address_value + _TWO_CTA_NCU_SASS_INSTRUCTION_SIZE
        fallthrough_indices = address_to_indices.get(fallthrough_value, [])
        fallthrough_index = (
            fallthrough_indices[0] if len(fallthrough_indices) == 1 else None
        )
        fallthrough = rows[fallthrough_index] if fallthrough_index is not None else None
        relation = _ncu_branch_target_relation(
            target_value=target_value,
            target_index=target_index,
            branch_index=index,
            group_indices=group_index_set,
            group_addresses=group_addresses,
        )
        ranked_paths.append(
            (
                (relation_priority[relation], address_distance, branch_address_value),
                {
                    "branch_address": row["address"],
                    "branch_opcode": row["opcode"],
                    "branch_source": row["source"],
                    "source_block": group_source_block,
                    "target_address": target_address,
                    "target_opcode": (
                        rows[target_index]["opcode"] if target_index is not None else ""
                    ),
                    "target_relation": relation,
                    "targets_current_wait_site": _ncu_branch_targets_current_wait_site(
                        rows=rows,
                        metric=metric,
                        target_value=target_value,
                        target_index=target_index,
                        group_indices=group_index_set,
                        group_addresses=group_addresses,
                    ),
                    "fallthrough_reachable": _sass_branch_has_reachable_fallthrough(
                        row["source"]
                    ),
                    "fallthrough_index": fallthrough_index,
                    "fallthrough_address": fallthrough["address"]
                    if fallthrough
                    else "",
                    "fallthrough_opcode": fallthrough["opcode"] if fallthrough else "",
                    "fallthrough_source_kind": (
                        fallthrough["source_kind"] if fallthrough else ""
                    ),
                },
            )
        )
    return [path for _key, path in sorted(ranked_paths, key=itemgetter(0))]


def _ncu_wait_site_focus_from_branch_paths(
    inferred_site: str,
    branch_paths: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    if inferred_site not in {
        "pipeline_barrier_wait",
        "async_fence_wait",
        "epilogue_store_wait",
    }:
        return None
    for path in branch_paths:
        if not path["targets_current_wait_site"] or not _is_sass_wait_loop_opcode(
            path["target_opcode"]
        ):
            continue
        if not path["fallthrough_reachable"]:
            continue
        fallthrough_index = path["fallthrough_index"]
        if fallthrough_index is None:
            continue
        source_block = path["source_block"]
        fallthrough_address = rows[fallthrough_index]["address_value"]
        if fallthrough_address is None:
            continue
        source_block_rows = [
            row
            for row in rows
            if row["source_block"] == source_block
            and row["address_value"] is not None
            and row["address_value"] >= fallthrough_address
        ]
        source_block_rows.sort(key=itemgetter("address_value", "address"))
        fallthrough_rows = source_block_rows[:_TWO_CTA_NCU_WAIT_SITE_FOCUS_ROWS]
        focus = _ncu_wait_site_focus_from_resumed_rows(
            fallthrough_rows,
            metric,
            smem_evidence="branch_fallthrough_smem_or_fence",
            tma_evidence="branch_fallthrough_tma_store",
            skip_leading_zero_sample_waits=False,
        )
        if focus is not None:
            return focus
    return None


def _rank_ncu_source_totals(
    totals: dict[tuple[str, str], float],
    *,
    total: float,
    label_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for (label, source_kind), value in totals.items():
        if value <= 0:
            continue
        row = {
            label_key: label,
            "value": value,
            "pct": value / total * 100 if total else 0.0,
        }
        if label_key != "source_kind":
            row["source_kind"] = source_kind
        ranked.append(row)
    ranked.sort(key=lambda row: (-row["value"], row[label_key]))
    return ranked[:limit]


def _ncu_source_context_row(row: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "address": row["address"],
        "source": row["source"],
        "opcode": row["opcode"],
        "source_kind": row["source_kind"],
        "value": row["metrics"].get(metric, 0.0),
    }


def _ncu_sync_state_trace(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row in sorted(rows, key=_ncu_source_row_address_sort_key):
        state = _sass_sync_state(row["source"])
        if not state:
            continue
        entry = {
            **state,
            "address": row["address"],
            "value": row["metrics"].get(metric, 0.0),
            "source": row["source"],
        }
        if row.get("line_info") is not None:
            entry["line_info"] = row["line_info"]
            entry["line_info_file"] = row["line_info_file"]
            entry["line_info_line"] = row["line_info_line"]
        states.append(entry)
    return states


def _ncu_sync_state_registers(state: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *_sass_registers(state["barrier"]),
                *_sass_registers(state["phase"]),
            ]
        )
    )


def _ncu_sync_register_provenance(
    rows: list[dict[str, Any]],
    sync_states: list[dict[str, Any]],
    metric: str,
    *,
    source_block: int,
    scan_limit: int | None = _TWO_CTA_NCU_SYNC_PROVENANCE_SCAN_ROWS,
) -> list[dict[str, Any]]:
    source_rows = [
        row
        for row in rows
        if row["source_block"] == source_block and row["address_value"] is not None
    ]
    source_rows.sort(key=itemgetter("address_value", "address"))
    provenance: list[dict[str, Any]] = []
    for state in sync_states:
        registers = _ncu_sync_state_registers(state)
        state_address = _sass_address_value(state["address"])
        if not registers or state_address is None:
            continue
        tracked_registers = set(registers)
        definition_rows: list[dict[str, Any]] = []
        definition_count = 0
        preceding_rows = [
            row for row in source_rows if row["address_value"] < state_address
        ]
        # NCU source-page rows stay local by default; cubin mode passes None to
        # scan every parsed SASS row in the source block.
        if scan_limit is not None:
            preceding_rows = preceding_rows[-scan_limit:]
        for row in reversed(preceding_rows):
            writes = set(_sass_write_registers(row["source"]))
            matched_writes = sorted(writes & tracked_registers)
            if not matched_writes:
                continue
            reads = _sass_read_registers(row["source"])
            definition_count += 1
            if len(definition_rows) < _TWO_CTA_NCU_SYNC_PROVENANCE_ROW_PRINT_LIMIT:
                definition_rows.append(
                    {
                        "address": row["address"],
                        "opcode": row["opcode"],
                        "source": row["source"],
                        "writes": matched_writes,
                        "reads": reads,
                        "value": row["metrics"].get(metric, 0.0),
                        "line_info": row.get("line_info"),
                    }
                )
            # Invariant: tracked_registers is the unresolved frontier of the
            # full backward slice, not only the original sync operands.
            tracked_registers.difference_update(matched_writes)
            tracked_registers.update(reads)
            tracked_registers.difference_update(_SASS_ZERO_REGISTERS)
            if not tracked_registers:
                break
        if not definition_count:
            status = "unresolved"
        elif tracked_registers:
            status = "partial"
        else:
            status = "resolved"
        provenance.append(
            {
                "key": state["key"],
                "address": state["address"],
                "registers": registers,
                "status": status,
                "unresolved_registers": sorted(tracked_registers),
                "definition_count": definition_count,
                "rows": definition_rows,
            }
        )
    return provenance


def _annotate_ncu_sync_related_wait_sites(wait_sites: list[dict[str, Any]]) -> None:
    sync_keys_by_site: dict[int, set[tuple[int, str]]] = {}
    sites_by_sync_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for site in wait_sites:
        scoped_keys = {
            (site["source_block"], state["key"]) for state in site["sync_states"]
        }
        sync_keys_by_site[id(site)] = scoped_keys
        for scoped_key in scoped_keys:
            sites_by_sync_key.setdefault(scoped_key, []).append(site)

    def related_sort_key(row: dict[str, Any]) -> tuple[float, bool, int, str, str]:
        address_value = _sass_address_value(row["address_start"])
        return (
            -row["value"],
            address_value is None,
            address_value or 0,
            row["key"],
            row["inferred_wait_site"],
        )

    for site in wait_sites:
        related: list[dict[str, Any]] = []
        for source_block, key in sync_keys_by_site[id(site)]:
            for other in sites_by_sync_key[(source_block, key)]:
                if other is site:
                    continue
                related.append(
                    {
                        "source_block": other["source_block"],
                        "key": key,
                        "address_start": other["address_start"],
                        "address_end": other["address_end"],
                        "inferred_wait_site": other["inferred_wait_site"],
                        "focus_wait_site": other["focus_wait_site"],
                        "focus_evidence": other["focus_evidence"],
                        "candidate_events": other["candidate_events"],
                        "value": other["value"],
                        "pct": other["pct"],
                    }
                )
        related.sort(key=related_sort_key)
        site["sync_related_wait_sites"] = related


def _sync_event_site_candidates(
    wait_sites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, str, int, int, str, str]] = set()
    for site in wait_sites:
        for context in site["source_context"]:
            context_key = (
                context["target"],
                context["source"],
                context["line_start"],
                context["line_end"],
            )
            for event_site in context["event_sites"]:
                site_label = event_site["site"]
                if not site_label:
                    continue
                key = (event_site["event"], site_label)
                seen_key = (*context_key, *key)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                counts[key] = counts.get(key, 0) + event_site["count"]
    result = [
        {"event": event, "site": site, "count": count}
        for (event, site), count in counts.items()
    ]
    result.sort(key=lambda row: (-row["count"], row["event"], row["site"]))
    return result


def _sync_event_site_status(event_sites: list[dict[str, Any]]) -> str:
    if len(event_sites) == 1:
        return "mapped"
    if len(event_sites) > 1:
        return "ambiguous"
    return "unmapped"


def _annotate_ncu_sync_event_site_candidates(
    wait_sites: list[dict[str, Any]],
) -> None:
    sites_by_sync_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sync_keys_by_site: dict[int, list[tuple[int, str]]] = {}
    for site in wait_sites:
        scoped_keys = list(
            dict.fromkeys(
                (site["source_block"], state["key"]) for state in site["sync_states"]
            )
        )
        sync_keys_by_site[id(site)] = scoped_keys
        for scoped_key in scoped_keys:
            sites_by_sync_key.setdefault(scoped_key, []).append(site)

    for site in wait_sites:
        sync_event_sites: list[dict[str, Any]] = []
        for source_block, key in sync_keys_by_site[id(site)]:
            event_sites = _sync_event_site_candidates(
                sites_by_sync_key[(source_block, key)]
            )
            sync_event_sites.append(
                {
                    "source_block": source_block,
                    "key": key,
                    "status": _sync_event_site_status(event_sites),
                    "event_sites": event_sites,
                    "wait_site_count": len(sites_by_sync_key[(source_block, key)]),
                }
            )
        site["sync_event_sites"] = sync_event_sites


def _ncu_source_wait_site_sort_key(row: dict[str, Any]) -> tuple[float, bool, int, str]:
    address_value = _sass_address_value(row["address_start"])
    return (
        -row["value"],
        address_value is None,
        address_value or 0,
        row["address_start"],
    )


def _ncu_source_wait_site_groups(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    total: float,
    limit: int,
) -> list[dict[str, Any]]:
    indexed_rows = list(enumerate(rows))
    wait_rows = [
        (index, row)
        for index, row in indexed_rows
        if row["metrics"].get(metric, 0.0) > 0.0 and _is_sass_wait_site_row(row)
    ]
    if not wait_rows:
        return []

    groups: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    previous_index = -1
    # Anchor address grouping so nearby rows do not drift into one long chain.
    group_anchor_address: int | None = None
    for index, row in wait_rows:
        address = row["address_value"]
        same_source_block = (
            not current or row["source_block"] == current[-1][1]["source_block"]
        )
        close_by_address = (
            group_anchor_address is not None
            and address is not None
            and abs(address - group_anchor_address)
            <= _TWO_CTA_NCU_WAIT_SITE_ADDRESS_GAP
        )
        close_by_source = previous_index >= 0 and (
            index - previous_index <= _TWO_CTA_NCU_WAIT_SITE_CONTEXT_ROWS + 1
        )
        if current and not (
            same_source_block and (close_by_address or close_by_source)
        ):
            groups.append(current)
            current = []
            group_anchor_address = None
        current.append((index, row))
        if group_anchor_address is None and address is not None:
            group_anchor_address = address
        previous_index = index
    if current:
        groups.append(current)

    wait_sites: list[dict[str, Any]] = []
    for group in groups:
        group_indices = [index for index, _row in group]
        group_source_block = rows[group_indices[0]]["source_block"]
        start_index = max(min(group_indices) - _TWO_CTA_NCU_WAIT_SITE_CONTEXT_ROWS, 0)
        end_index = min(
            max(group_indices) + _TWO_CTA_NCU_WAIT_SITE_CONTEXT_ROWS + 1,
            len(rows),
        )
        context_rows = [
            row
            for row in rows[start_index:end_index]
            if row["source_block"] == group_source_block
        ]
        context_opcodes = {row["opcode"] for row in context_rows if row["opcode"]} | {
            row["opcode"] for _index, row in group if row["opcode"]
        }
        inferred_site, candidate_events = _infer_sass_wait_site(context_opcodes)
        focus_start_index = max(
            min(group_indices) - _TWO_CTA_NCU_WAIT_SITE_FOCUS_ROWS,
            0,
        )
        focus_end_index = min(
            max(group_indices) + _TWO_CTA_NCU_WAIT_SITE_FOCUS_ROWS + 1,
            len(rows),
        )
        focus_before_rows = [
            row
            for row in rows[focus_start_index : min(group_indices)]
            if row["source_block"] == group_source_block
        ]
        focus_after_rows = [
            row
            for row in rows[max(group_indices) + 1 : focus_end_index]
            if row["source_block"] == group_source_block
        ]
        focus = _ncu_wait_site_focus_from_sass(
            inferred_site,
            metric,
            focus_before_rows,
            focus_after_rows,
        )
        branch_paths = _ncu_source_wait_site_branch_paths(
            rows,
            group_indices,
            metric=metric,
        )
        branch_focus = _ncu_wait_site_focus_from_branch_paths(
            inferred_site,
            branch_paths,
            rows,
            metric,
        )
        # Branch fallthrough is only allowed to replace pre-wait fallback
        # evidence; direct post-wait evidence stays closest to the hot stall.
        if branch_focus is not None and (
            focus is None
            or focus["evidence"] in _TWO_CTA_NCU_WAIT_SITE_BRANCH_OVERRIDABLE_EVIDENCE
        ):
            focus = branch_focus
        if focus is not None:
            candidate_events = focus["candidate_events"]
        group_addresses = [
            row["address_value"]
            for _index, row in group
            if row["address_value"] is not None
        ]
        value = sum(row["metrics"].get(metric, 0.0) for _index, row in group)
        sync_states = _ncu_sync_state_trace(context_rows, metric)
        wait_sites.append(
            {
                "source_block": group_source_block,
                "address_start": group[0][1]["address"],
                "address_end": group[-1][1]["address"],
                "address_span": (
                    max(group_addresses) - min(group_addresses)
                    if group_addresses
                    else None
                ),
                "value": value,
                "pct": value / total * 100 if total else 0.0,
                "row_count": len(group),
                "inferred_wait_site": inferred_site,
                "focus_wait_site": focus["wait_site"] if focus else "",
                "focus_evidence": focus["evidence"] if focus else "",
                "candidate_events": candidate_events,
                "opcodes": sorted(context_opcodes),
                "hot_rows": [
                    _ncu_source_context_row(row, metric) for _index, row in group
                ],
                "context_rows": [
                    _ncu_source_context_row(row, metric) for row in context_rows
                ],
                "focus_before_rows": [
                    _ncu_source_context_row(row, metric) for row in focus_before_rows
                ],
                "focus_after_rows": [
                    _ncu_source_context_row(row, metric) for row in focus_after_rows
                ],
                "branch_paths": branch_paths,
                "source_context": [],
                "sync_states": sync_states,
                "sync_register_provenance": _ncu_sync_register_provenance(
                    rows,
                    sync_states,
                    metric,
                    source_block=group_source_block,
                ),
                "sync_related_wait_sites": [],
                "sync_event_sites": [],
            }
        )

    _annotate_ncu_sync_related_wait_sites(wait_sites)
    wait_sites.sort(key=_ncu_source_wait_site_sort_key)
    return wait_sites[:limit]


def _summarize_ncu_source_rows(
    *,
    kernel: str,
    rows: list[dict[str, Any]],
    metric_names: list[str],
    top_limit: int,
) -> dict[str, Any]:
    metric_totals = dict.fromkeys(metric_names, 0.0)
    opcode_totals: dict[str, dict[tuple[str, str], float]] = {
        metric: {} for metric in metric_names
    }
    source_kind_totals: dict[str, dict[tuple[str, str], float]] = {
        metric: {} for metric in metric_names
    }
    top_rows: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metric_names}
    wait_sites: dict[str, list[dict[str, Any]]] = {
        metric: [] for metric in metric_names
    }
    for row in rows:
        for metric in metric_names:
            value = row["metrics"].get(metric, 0.0)
            if value <= 0:
                continue
            metric_totals[metric] += value
            opcode_key = (row["opcode"], row["source_kind"])
            kind_key = (row["source_kind"], row["source_kind"])
            opcode_totals[metric][opcode_key] = (
                opcode_totals[metric].get(opcode_key, 0.0) + value
            )
            source_kind_totals[metric][kind_key] = (
                source_kind_totals[metric].get(kind_key, 0.0) + value
            )
            top_rows[metric].append(
                {
                    "address": row["address"],
                    "source": row["source"],
                    "opcode": row["opcode"],
                    "source_kind": row["source_kind"],
                    "value": value,
                }
            )

    for metric, metric_rows in top_rows.items():
        total = metric_totals[metric]
        metric_rows.sort(key=lambda row: (-row["value"], row["address"]))
        for metric_row in metric_rows[:top_limit]:
            metric_row["pct"] = metric_row["value"] / total * 100 if total else 0.0
        top_rows[metric] = metric_rows[:top_limit]
        wait_sites[metric] = _ncu_source_wait_site_groups(
            rows,
            metric,
            total=total,
            limit=top_limit,
        )

    return {
        "kernel": kernel,
        "row_count": len(rows),
        "metric_totals": {k: v for k, v in metric_totals.items() if v},
        "opcode_totals": {
            metric: _rank_ncu_source_totals(
                totals,
                total=metric_totals[metric],
                label_key="opcode",
                limit=top_limit,
            )
            for metric, totals in opcode_totals.items()
        },
        "source_kind_totals": {
            metric: _rank_ncu_source_totals(
                totals,
                total=metric_totals[metric],
                label_key="source_kind",
                limit=top_limit,
            )
            for metric, totals in source_kind_totals.items()
        },
        "top_rows": {metric: rows for metric, rows in top_rows.items() if rows},
        "wait_sites": {metric: rows for metric, rows in wait_sites.items() if rows},
    }


def _parse_ncu_source_csv(
    csv_text: str,
    metric_names: list[str],
    *,
    top_limit: int = _TWO_CTA_NCU_SOURCE_TOP_LIMIT,
    summary_metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """Parse Nsight Compute ``--page source --print-source sass`` CSV output."""
    requested = set(metric_names)
    summary_metrics = (
        metric_names if summary_metric_names is None else summary_metric_names
    )
    kernels: dict[str, list[dict[str, Any]]] = {}
    kernel_order: list[str] = []
    current_kernel: str | None = None
    current_source_block = -1
    address_idx = -1
    source_idx = -1
    metric_columns: list[tuple[int, str]] = []
    reader = csv.reader(
        line
        for line in csv_text.splitlines()
        if line.strip() and not line.lstrip().startswith("==")
    )
    for row in reader:
        if len(row) >= 2 and row[0] == "Kernel Name":
            current_kernel = row[1]
            current_source_block += 1
            if current_kernel not in kernels:
                kernels[current_kernel] = []
                kernel_order.append(current_kernel)
            address_idx = -1
            source_idx = -1
            metric_columns = []
            continue
        if current_kernel is None:
            continue
        if "Address" in row and "Source" in row:
            address_idx = row.index("Address")
            source_idx = row.index("Source")
            metric_columns = [
                (idx, column) for idx, column in enumerate(row) if column in requested
            ]
            continue
        if not metric_columns or address_idx < 0 or source_idx < 0:
            continue
        if len(row) <= max(address_idx, source_idx):
            continue
        address = row[address_idx]
        source = row[source_idx]
        if not address or not source:
            continue
        metrics = {
            metric: _parse_ncu_counter(row[idx]) if idx < len(row) else 0.0
            for idx, metric in metric_columns
        }
        opcode = _sass_opcode(source)
        kernels[current_kernel].append(
            {
                "address": address,
                "address_value": _sass_address_value(address),
                "source": source.strip(),
                "opcode": opcode,
                "source_kind": _sass_source_kind(opcode),
                "source_block": current_source_block,
                "metrics": metrics,
            }
        )

    kernel_summaries = [
        _summarize_ncu_source_rows(
            kernel=kernel,
            rows=kernels[kernel],
            metric_names=summary_metrics,
            top_limit=top_limit,
        )
        for kernel in kernel_order
    ]
    all_rows = [row for kernel in kernel_order for row in kernels[kernel]]
    return {
        "metrics": summary_metrics,
        "collected_metrics": metric_names,
        "top_limit": top_limit,
        "kernels": kernel_summaries,
        "aggregate": _summarize_ncu_source_rows(
            kernel="all",
            rows=all_rows,
            metric_names=summary_metrics,
            top_limit=top_limit,
        ),
    }


def _has_ncu_source_metric_samples(parsed_source: dict[str, Any]) -> bool:
    aggregate = parsed_source["aggregate"]
    if aggregate["row_count"] <= 0:
        return False
    metric_totals = aggregate["metric_totals"]
    top_rows = aggregate["top_rows"]
    return any(
        metric_totals.get(metric, 0.0) > 0.0 or bool(top_rows.get(metric))
        for metric in parsed_source["metrics"]
    )


def _parse_ncu_csv_metrics(
    csv_text: str, metric_names: list[str] | None = None
) -> dict[str, list[dict[str, str]]]:
    """Parse Nsight Compute raw CSV output into per-metric rows."""
    data_lines = [
        line
        for line in csv_text.splitlines()
        if line.strip() and not line.lstrip().startswith("==")
    ]
    metrics: dict[str, list[dict[str, str]]] = {}
    for offset, line in enumerate(data_lines):
        if "Metric Name" not in line or "Metric Value" not in line:
            continue
        reader = csv.DictReader(data_lines[offset:])
        for row in reader:
            metric = row.get("Metric Name")
            if not metric or metric == "Metric Name":
                continue
            value = row.get("Metric Value")
            if value is None:
                continue
            metrics.setdefault(metric, []).append(
                {
                    "value": value,
                    "unit": row.get("Metric Unit") or "",
                    "kernel": row.get("Kernel Name") or "",
                    "section": row.get("Section Name") or "",
                }
            )
        break
    requested = set(metric_names or _TWO_CTA_NCU_DEFAULT_METRICS)
    for offset, line in enumerate(data_lines):
        if not any(metric in line for metric in requested):
            continue
        reader = csv.reader(data_lines[offset:])
        try:
            header = next(reader)
        except StopIteration:
            continue
        if "Kernel Name" not in header:
            continue
        metric_columns = [
            (idx, column) for idx, column in enumerate(header) if column in requested
        ]
        if not metric_columns:
            continue
        units = next(reader, [])
        kernel_idx = header.index("Kernel Name")
        section_idx = header.index("Section Name") if "Section Name" in header else -1
        for row in reader:
            if len(row) <= kernel_idx or not row[kernel_idx]:
                continue
            for idx, metric in metric_columns:
                if len(row) <= idx or not row[idx]:
                    continue
                unit = units[idx] if idx < len(units) else ""
                section = row[section_idx] if 0 <= section_idx < len(row) else ""
                metrics.setdefault(metric, []).append(
                    {
                        "value": row[idx],
                        "unit": unit,
                        "kernel": row[kernel_idx],
                        "section": section,
                    }
                )
        break
    return metrics


def _run_helion_two_cta_ncu_report(args: argparse.Namespace) -> dict[str, Any]:
    command = _build_two_cta_ncu_command(args)
    payload: dict[str, Any] = {
        "impl": "helion-cute:two-cta-ncu-report",
        "shape": {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype},
        "target_config": args.ncu_target_config,
        "command": command,
        "metrics": args.ncu_metrics,
        "warmup_launches": args.ncu_warmup_launches,
        "profile_launches": args.ncu_profile_launches,
        "command_only": bool(args.ncu_command_only),
        "source_report": bool(args.ncu_source_report),
        "source_metrics": args.ncu_source_metrics,
        "source_provenance_metrics": args.ncu_source_provenance_metrics,
        "source_collection_metrics": _ncu_source_collection_metrics(args),
        "source_top": args.ncu_source_top,
    }
    if args.ncu_source_report:
        payload["source_command"] = _build_two_cta_ncu_source_command(args)
    if args.ncu_command_only:
        return payload

    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "--helion-two-cta-ncu-report requires Nsight Compute. "
            f"Could not execute {args.ncu_binary!r}; pass --ncu-binary or "
            "use --ncu-command-only to inspect the command."
        ) from exc
    payload["returncode"] = proc.returncode
    payload["stdout"] = proc.stdout
    payload["stderr"] = proc.stderr
    payload["parsed_metrics"] = _parse_ncu_csv_metrics(proc.stdout, args.ncu_metrics)
    if proc.returncode != 0:
        raise SystemExit(
            "Nsight Compute failed for --helion-two-cta-ncu-report "
            f"(rc={proc.returncode}).\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    if not payload["parsed_metrics"]:
        raise SystemExit(
            "Nsight Compute completed but no requested metrics were parsed. "
            "Use --ncu-command-only to inspect the command or adjust "
            "--ncu-metrics."
        )
    if args.ncu_source_report:
        try:
            source_proc = subprocess.run(
                payload["source_command"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "--ncu-source-report requires Nsight Compute. "
                f"Could not execute {args.ncu_binary!r}; pass --ncu-binary or "
                "use --ncu-command-only to inspect the command."
            ) from exc
        payload["source_returncode"] = source_proc.returncode
        payload["source_stderr"] = source_proc.stderr
        parsed_source = _parse_ncu_source_csv(
            source_proc.stdout,
            payload["source_collection_metrics"],
            top_limit=args.ncu_source_top,
            summary_metric_names=args.ncu_source_metrics,
        )
        payload["parsed_source"] = parsed_source
        if source_proc.returncode != 0:
            raise SystemExit(
                "Nsight Compute failed for --ncu-source-report "
                f"(rc={source_proc.returncode}).\n--- stdout ---\n"
                f"{source_proc.stdout}\n--- stderr ---\n{source_proc.stderr}"
            )
        if not _has_ncu_source_metric_samples(parsed_source):
            raise SystemExit(
                "Nsight Compute completed but no SASS source rows or requested "
                "source metric samples were parsed. "
                "Use --ncu-command-only to inspect the command or adjust "
                "--ncu-source-metrics."
            )
        if _has_ncu_wait_sites(parsed_source):
            source_context_entries = _ncu_source_context_entries(args)
            _annotate_ncu_wait_sites_with_source_context(
                parsed_source,
                source_context_entries,
            )
    return payload


def _print_cubin_sass_report(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    print(f"\nshape={shape['m']}x{shape['n']}x{shape['k']} dtype={shape['dtype']}")
    print(f"Target config: {payload['target_config']}")
    print(payload["config"])
    print(
        f"CUBINs: {payload['cubin_count']} byte_counts={payload['cubin_byte_counts']}"
    )
    print(f"SASS rows parsed: {payload['sass_row_count']}")
    if payload["nvdisasm_line_info"]:
        print(
            f"Line-info rows parsed: {payload['line_info_row_count']} "
            f"sources={payload['line_info_sources'][:3]}"
        )
    print(f"SYNCS.PHASECHK states: {payload['sync_state_count']}")
    if payload["sync_key"]:
        print(f"Focused sync key: {payload['sync_key']}")
    print(f"nvdisasm binary: {payload['nvdisasm_binary']}")
    print()
    print("Sync-state groups:")
    for group in payload["sync_state_groups"][:_TWO_CTA_CUBIN_SYNC_STATE_PRINT_LIMIT]:
        statuses = ",".join(
            f"{status}:{count}" for status, count in sorted(group["statuses"].items())
        )
        print(
            f"  {group['key']} count={group['count']} statuses=[{statuses}] "
            f"first=block{group['first_source_block']}@{group['first_address']}"
        )
    print()
    print("Sync register provenance:")
    for state in payload["sync_register_provenance"][
        :_TWO_CTA_CUBIN_SYNC_STATE_PRINT_LIMIT
    ]:
        formatted = _format_ncu_sync_register_provenance([state])
        line_info = f" line={state['line_info']}" if state.get("line_info") else ""
        print(
            f"  {state['source_block_name']}{line_info} {formatted} "
            f"source={state['source']}"
        )
    hidden = payload["sync_state_count"] - _TWO_CTA_CUBIN_SYNC_STATE_PRINT_LIMIT
    if hidden > 0:
        print(f"  +{hidden} more sync states")
    print()


def _print_codegen_report(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    print(f"\nshape={shape['m']}x{shape['n']}x{shape['k']} dtype={shape['dtype']}")
    print("Helion validated CtaGroup.TWO config:")
    print(payload["helion"]["config"])
    print(f"Quack sources: {', '.join(payload['quack']['sources'])}")
    print()
    print(f"{'marker':>28}  {'helion':>6}  {'quack':>6}  {'delta':>6}")
    print("-" * 52)
    helion_markers = payload["helion"]["markers"]
    quack_markers = payload["quack"]["markers"]
    deltas = payload["marker_delta_helion_minus_quack"]
    for label in _SOURCE_MARKER_LABELS:
        print(
            f"{label:>28}  {helion_markers[label]:>6}  "
            f"{quack_markers[label]:>6}  {deltas[label]:>6}"
        )
    print()
    print("First marker lines:")
    for label in _SOURCE_MARKER_LABELS:
        print(
            f"{label:>28}  "
            f"helion={payload['helion']['first_marker_locations'][label]}  "
            f"quack={payload['quack']['first_marker_locations'][label]}"
        )
    print()
    print("Helion role-local loops:")
    for role in payload["helion"]["role_local_loops"]:
        markers = _nonzero_markers(role["markers"])
        event_counts = role["pipeline_events"]["counts"]
        event_trace = ", ".join(
            _format_pipeline_event_ref(entry)
            for entry in role["pipeline_events"]["trace"]
        )
        trace = ", ".join(
            f"{entry['line']}:{entry['marker']}" for entry in role["marker_trace"]
        )
        print(
            f"  {role['role']:>8} lines {role['line_start']}-{role['line_end']} "
            f"markers={markers} trace=[{trace}]"
        )
        print(f"           events={event_counts} event_trace=[{event_trace}]")
        print(
            "           windows=["
            f"{_format_pipeline_windows(role['pipeline_events']['windows'])}]"
        )
    print()
    print("Pipeline event counts:")
    print(f"  helion={payload['helion']['pipeline_events']['counts']}")
    print(f"  quack ={payload['quack']['pipeline_events']['counts']}")
    print()
    print("Pipeline source-order gaps:")
    for side in ("helion", "quack"):
        gaps = payload[side]["pipeline_events"]["gaps"]
        print(f"  {side}:")
        for gap in gaps[:8]:
            location = f"{gap.get('source', 'generated_helion.py')}:"
            print(
                f"    {gap['name']} {location}{gap['start_line']}->{gap['end_line']} "
                f"line_delta={gap['line_delta']} event_delta={gap['event_delta']}"
            )
    print()
    print("Epilogue token topology:")
    for side in ("helion", "quack"):
        topology = payload[side]["epilogue_token_topology"]
        windows = topology["window_counts"]
        print(
            f"  {side}: events=[{_format_epilogue_token_counts(topology['events'])}] "
            f"markers=[{_format_epilogue_token_counts(topology['markers'])}] "
            f"sites=[{_format_epilogue_token_sites(topology['event_sites'])}] "
            f"windows=complete:{windows['complete']} partial:{windows['partial']}"
        )
        for context in topology["contexts"][:_EPILOGUE_TOKEN_PRINT_CONTEXTS]:
            event_groups = _format_source_context_event_groups(context["event_groups"])
            event_sites = _format_source_context_event_sites(context["event_sites"])
            windows_text = _format_pipeline_windows(context["windows"])
            markers = context["matched_markers"]
            print(
                f"    {_format_source_context_location(context)} "
                f"score={context['score']} coverage={context['coverage']} "
                f"groups=[{event_groups}] sites=[{event_sites}] "
                f"markers={markers} windows=[{windows_text}]"
            )
    print()
    marker_symbols = payload["quack"]["marker_symbols"]
    print(
        "Quack marker/pipeline-bearing symbols "
        f"(shown {len(marker_symbols)} of {len(marker_symbols)}):"
    )
    for symbol in marker_symbols:
        trace = ", ".join(
            f"{entry['line']}:{entry['marker']}" for entry in symbol["marker_trace"]
        )
        event_trace = ", ".join(
            _format_pipeline_event_ref(entry)
            for entry in symbol["pipeline_events"]["trace"]
        )
        print(
            f"  {symbol['source']}:{symbol['line_start']}-{symbol['line_end']} "
            f"{symbol['symbol']} markers={symbol['markers']} trace=[{trace}] "
            f"events={symbol['pipeline_events']['counts']} "
            f"event_trace=[{event_trace}] "
            f"windows=[{_format_pipeline_windows(symbol['pipeline_events']['windows'])}]"
        )
    print()


def _print_ncu_report(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    print(f"\nshape={shape['m']}x{shape['n']}x{shape['k']} dtype={shape['dtype']}")
    print(f"Target config: {payload['target_config']}")
    print("Nsight Compute command:")
    print(" ".join(str(part) for part in payload["command"]))
    if payload.get("source_command"):
        print("Nsight Compute SASS source command:")
        print(" ".join(str(part) for part in payload["source_command"]))
    print()
    if payload["command_only"]:
        print("command-only mode; no profiler counters collected.")
        return

    print("Parsed Nsight Compute metrics:")
    parsed_metrics = payload.get("parsed_metrics", {})
    if not parsed_metrics:
        print("  no metrics parsed from profiler output")
        return
    for metric in payload["metrics"]:
        rows = parsed_metrics.get(metric, [])
        if not rows:
            print(f"  {metric}: <missing>")
            continue
        values = ", ".join(f"{row['value']} {row['unit']}".strip() for row in rows[:4])
        print(f"  {metric}: {values}")

    parsed_source = payload.get("parsed_source")
    if not parsed_source:
        return
    aggregate = parsed_source["aggregate"]
    print()
    print("SASS source metric attribution:")
    for metric in parsed_source["metrics"]:
        total = aggregate["metric_totals"].get(metric, 0.0)
        print(f"  {metric}: total={total:g}")
        kind_totals = aggregate["source_kind_totals"].get(metric, [])
        if kind_totals:
            formatted_kinds = ", ".join(
                f"{row['source_kind']}={row['value']:g} ({row['pct']:.1f}%)"
                for row in kind_totals[:4]
            )
            print(f"    source kinds: {formatted_kinds}")
        opcode_totals = aggregate["opcode_totals"].get(metric, [])
        if opcode_totals:
            formatted_opcodes = ", ".join(
                f"{row['opcode']}={row['value']:g} ({row['pct']:.1f}%)"
                for row in opcode_totals[:4]
            )
            print(f"    opcodes: {formatted_opcodes}")
        wait_sites = aggregate["wait_sites"].get(metric, [])
        if wait_sites:
            topology = _ncu_wait_site_topology(wait_sites, total=total)
            for label, key in (
                ("displayed wait topology", "wait_sites"),
                ("displayed source windows", "source_windows"),
                ("displayed event sites", "event_sites"),
                ("displayed sync-site status", "sync_statuses"),
            ):
                formatted_topology = _format_ncu_wait_topology_rows(topology[key])
                if formatted_topology:
                    print(f"    {label}: {formatted_topology}")
            sync_key_topology = _format_ncu_sync_key_topology_rows(
                _ncu_sync_key_topology(wait_sites, total=total)
            )
            if sync_key_topology:
                print(f"    displayed sync keys: {sync_key_topology}")
            sass_topology = _format_ncu_sass_topology_rows(
                _ncu_sass_wait_topology(wait_sites, total=total)
            )
            if sass_topology:
                print(f"    displayed SASS topology: {sass_topology}")
            print("    likely wait sites:")
            for site in wait_sites[:4]:
                opcodes = ", ".join(site["opcodes"][:4])
                events = ", ".join(site["candidate_events"])
                focus = (
                    f" focus={site['focus_wait_site']}:{site['focus_evidence']}"
                    if site.get("focus_wait_site")
                    else ""
                )
                branches = (
                    f" branches=[{_format_ncu_branch_paths(site['branch_paths'])}]"
                    if site["branch_paths"]
                    else ""
                )
                hot_sass = _format_ncu_wait_site_hot_rows(site["hot_rows"])
                sass = f" sass=[{hot_sass}]" if hot_sass else ""
                loop_sass = _format_ncu_wait_site_loop_rows(site["context_rows"])
                loop = f" loop=[{loop_sass}]" if loop_sass else ""
                sync_states = _format_ncu_sync_states(site["sync_states"])
                sync = f" sync=[{sync_states}]" if sync_states else ""
                sync_registers = _format_ncu_sync_register_provenance(
                    site["sync_register_provenance"]
                )
                sync_regs = f" sync_regs=[{sync_registers}]" if sync_registers else ""
                related_sites = _format_ncu_sync_related_wait_sites(
                    site["sync_related_wait_sites"]
                )
                related = f" related=[{related_sites}]" if related_sites else ""
                sync_event_sites = _format_ncu_sync_event_sites(
                    site["sync_event_sites"]
                )
                sync_sites = (
                    f" sync_sites=[{sync_event_sites}]" if sync_event_sites else ""
                )
                print(
                    f"      {site['address_start']}..{site['address_end']} "
                    f"{site['inferred_wait_site']}{focus} "
                    f"{site['value']:g} ({site['pct']:.1f}%) "
                    f"events=[{events}] opcodes=[{opcodes}]{branches}"
                    f"{sync}{sync_regs}{related}{sync_sites}{sass}{loop}"
                )
                for context in site["source_context"][
                    :_SOURCE_CONTEXT_PRINT_CANDIDATES
                ]:
                    label = context.get("role") or context.get("symbol") or "source"
                    matched_events = ", ".join(context["matched_events"])
                    event_groups = _format_source_context_event_groups(
                        context["event_groups"]
                    )
                    event_sites = _format_source_context_event_sites(
                        context["event_sites"]
                    )
                    event_lines = ", ".join(
                        _format_pipeline_event_ref(entry)
                        for entry in context["event_trace"][
                            :_SOURCE_CONTEXT_PRINT_EVENT_LINES
                        ]
                    )
                    matched_markers = ", ".join(context["matched_markers"])
                    marker_lines = ", ".join(
                        f"{entry['line']}:{entry['marker']}"
                        for entry in context["marker_trace"][
                            :_SOURCE_CONTEXT_PRINT_MARKER_LINES
                        ]
                    )
                    windows = _format_pipeline_windows(
                        context["windows"][:_SOURCE_CONTEXT_PRINT_WINDOWS]
                    )
                    print(
                        f"        source {context['target']} "
                        f"{context['source']}:{context['line_start']}-"
                        f"{context['line_end']} {label} "
                        f"coverage={context['coverage']} score={context['score']} "
                        f"events=[{matched_events}] "
                        f"groups=[{event_groups}] "
                        f"sites=[{event_sites}] "
                        f"event_lines=[{event_lines}] "
                        f"windows=[{windows}] "
                        f"markers=[{matched_markers}] marker_lines=[{marker_lines}]"
                    )
        for row in aggregate["top_rows"].get(metric, [])[:4]:
            print(
                f"    {row['address']} {row['opcode']} "
                f"{row['value']:g} ({row['pct']:.1f}%) {row['source']}"
            )


def _print_summary(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    epilogue = shape.get("epilogue", "none")
    print(
        f"\nshape={shape['m']}x{shape['n']}x{shape['k']} "
        f"dtype={shape['dtype']} epilogue={epilogue}"
    )
    impl_width = max(
        16,
        max(
            (len(str(result.get("impl", ""))) for result in payload["results"]),
            default=0,
        ),
    )
    separator_width = impl_width + 78
    print("=" * separator_width)
    header = (
        f"{'impl':>{impl_width}}  {'best ms':>10}  {'mom-med ms':>10}  "
        f"{'best TFLOP/s':>14}  {'mom-med TFLOP/s':>18}"
    )
    print(header)
    print("-" * separator_width)
    aten_mom: float | None = None
    quack_direct_mom: float | None = None
    for r in payload["results"]:
        if r.get("impl") == "aten":
            aten_mom = r.get("mom_median_tflops", r.get("median_tflops"))
        elif r.get("impl") == "quack-direct":
            quack_direct_mom = r.get("mom_median_tflops", r.get("median_tflops"))
    for r in payload["results"]:
        if "best_ms" not in r:
            continue
        mom_ms = r.get("mom_median_ms", r["median_ms"])
        mom_tflops = r.get("mom_median_tflops", r["median_tflops"])
        line = (
            f"{r['impl']:>{impl_width}}  "
            f"{r['best_ms']:>10.4f}  {mom_ms:>10.4f}  "
            f"{r['best_tflops']:>14.1f}  {mom_tflops:>18.1f}"
        )
        if aten_mom is not None and r["impl"] != "aten":
            line += f"   {mom_tflops / aten_mom * 100:>5.1f}% ATen"
        if quack_direct_mom is not None and r["impl"] not in (
            "aten",
            "quack-direct",
        ):
            line += f"   {mom_tflops / quack_direct_mom * 100:>5.1f}% Quack-direct"
        print(line)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--impl", choices=("all", *ALL_IMPLS), default="all")
    parser.add_argument(
        "--impls",
        nargs="*",
        default=None,
        help=(
            "Override DEFAULT_IMPLS for --impl all "
            f"(default: {' '.join(DEFAULT_IMPLS)})"
        ),
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument(
        "--epilogue",
        choices=MATMUL_EPILOGUES,
        default="none",
        help="Matmul-family epilogue fused into correctness and timing.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--print-codegen",
        action="store_true",
        help=("(helion-* only) Compile, print codegen markers, exit without timing."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (used by --impl all subprocess collection).",
    )
    parser.add_argument(
        "--helion-two-cta-diagnostic-sweep",
        action="store_true",
        help=(
            "Run a curated fixed-config Helion-cute CtaGroup.TWO sweep for "
            "the validated 256x256x128 seed family."
        ),
    )
    parser.add_argument(
        "--helion-two-cta-codegen-report",
        action="store_true",
        help=(
            "Compile the validated Helion-cute CtaGroup.TWO row and compare "
            "source/codegen marker counts with Quack's SM100 GEMM source."
        ),
    )
    parser.add_argument(
        "--helion-two-cta-ncu-report",
        action="store_true",
        help=(
            "Profile a fixed Helion/Quack two-CTA diagnostic row with Nsight "
            "Compute scheduler/stall metrics."
        ),
    )
    parser.add_argument(
        "--helion-two-cta-cubin-sass-report",
        action="store_true",
        help=(
            "Compile the validated Helion-cute CtaGroup.TWO row, disassemble "
            "the retained cubin with nvdisasm, and trace full-SASS sync-register "
            "provenance."
        ),
    )
    parser.add_argument(
        "--helion-two-cta-ncu-target",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ncu-binary", default="ncu", help=argparse.SUPPRESS)
    parser.add_argument(
        "--nvdisasm-binary",
        default="nvdisasm",
        help="nvdisasm executable used by --helion-two-cta-cubin-sass-report.",
    )
    parser.add_argument(
        "--ncu-metrics",
        type=_parse_csv_list,
        default=list(_TWO_CTA_NCU_DEFAULT_METRICS),
        help=(
            "Comma-separated Nsight Compute metric names for "
            "--helion-two-cta-ncu-report."
        ),
    )
    parser.add_argument(
        "--ncu-target-config",
        choices=_TWO_CTA_NCU_TARGET_CONFIGS,
        default="two_cta_seed",
        help=(
            "Fixed Helion/Quack config profiled by --helion-two-cta-ncu-report "
            f"(choices: {', '.join(_TWO_CTA_NCU_TARGET_CONFIGS)})."
        ),
    )
    parser.add_argument(
        "--ncu-warmup-launches",
        type=int,
        default=1,
        help="Kernel launches skipped before Nsight Compute collects metrics.",
    )
    parser.add_argument(
        "--ncu-profile-launches",
        type=int,
        default=1,
        help="Kernel launches collected by Nsight Compute.",
    )
    parser.add_argument(
        "--ncu-command-only",
        action="store_true",
        help=(
            "Print the Nsight Compute command for --helion-two-cta-ncu-report "
            "without running it."
        ),
    )
    parser.add_argument(
        "--ncu-source-report",
        action="store_true",
        help=(
            "With --helion-two-cta-ncu-report, run an additional Nsight Compute "
            "SASS source-page pass and summarize source-level stall attribution."
        ),
    )
    parser.add_argument(
        "--ncu-source-metrics",
        type=_parse_csv_list,
        default=list(_TWO_CTA_NCU_SOURCE_METRICS),
        help=(
            "Comma-separated source-correlated Nsight Compute metric names for "
            "--ncu-source-report."
        ),
    )
    parser.add_argument(
        "--ncu-source-provenance-metrics",
        type=_parse_csv_list,
        default=list(_TWO_CTA_NCU_SOURCE_PROVENANCE_METRICS),
        help=(
            "Additional source-correlated Nsight Compute metrics collected by "
            "--ncu-source-report to enrich SASS rows for register provenance "
            "without summarizing them as source attribution metrics."
        ),
    )
    parser.add_argument(
        "--ncu-source-top",
        type=int,
        default=_TWO_CTA_NCU_SOURCE_TOP_LIMIT,
        help="Number of top SASS rows/opcodes shown by --ncu-source-report.",
    )
    parser.add_argument(
        "--cubin-sync-key",
        default=None,
        help=(
            "Optional sync key, such as '[R0+URZ],R5', used to focus "
            "--helion-two-cta-cubin-sass-report."
        ),
    )

    parser.add_argument(
        "--quack-path",
        type=str,
        default=None,
        help=f"Path to quack package (default: {QUACK_PATH})",
    )
    # Quack-direct config defaults to B200 best for compute-bound shapes.
    parser.add_argument("--quack-tile-m", type=int, default=256)
    parser.add_argument("--quack-tile-n", type=int, default=256)
    parser.add_argument("--quack-cluster-m", type=int, default=2)
    parser.add_argument("--quack-cluster-n", type=int, default=1)
    parser.add_argument("--quack-pingpong", type=int, choices=(0, 1), default=0)
    parser.add_argument("--quack-persistent", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--quack-dynamic-persistent", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--quack-max-swizzle-size", type=int, default=8)
    parser.add_argument(
        "--quack-tune",
        choices=QUACK_TUNE_CHOICES,
        default="off",
        help=(
            "Quack-direct config tuning mode. 'brief' benchmarks a small "
            "candidate set and reports the selected config."
        ),
    )

    parser.add_argument(
        "--helion-backend",
        choices=("cute", "triton"),
        default="cute",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--helion-force-config",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use the --helion-* args as a fixed config (skip autotune).",
    )
    parser.add_argument("--helion-tile-m", type=int, default=128)
    parser.add_argument("--helion-tile-n", type=int, default=256)
    parser.add_argument("--helion-tile-k", type=int, default=128)
    parser.add_argument(
        "--helion-loop-order",
        type=_parse_loop_order,
        default=[0, 1],
        help="Comma-separated M/N tile loop-order permutation: 0,1 or 1,0.",
    )
    parser.add_argument("--helion-num-stages", type=int, default=2)
    parser.add_argument("--helion-num-warps", type=int, default=4)
    parser.add_argument("--helion-maxnreg", type=int, default=None)
    parser.add_argument("--helion-num-sm-multiplier", type=int, default=1)
    parser.add_argument(
        "--helion-acc-wait-placement",
        choices=TCGEN05_ACC_WAIT_PLACEMENTS,
        default=TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
        help=(
            "CtaGroup.TWO TMA-store epilogue placement for the accumulator "
            "consumer_wait. Mostly diagnostic; the edge+K-tail production "
            "seed uses the measured before-subtile-loop value."
        ),
    )
    parser.add_argument(
        "--helion-acc-producer-mode",
        choices=TCGEN05_ACC_PRODUCER_MODES,
        default=TCGEN05_ACC_PRODUCER_MODE_NORMAL,
        help=(
            "Diagnostic-only invalid-output tcgen05 accumulator producer mode. "
            "Non-normal values require --skip-correctness 1."
        ),
    )
    parser.add_argument(
        "--helion-c-acquire-placement",
        choices=TCGEN05_C_ACQUIRE_PLACEMENTS,
        default=TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
        help=(
            "CtaGroup.TWO TMA-store epilogue placement for C producer_acquire "
            "calls. Mostly diagnostic; the edge+K-tail production seed uses "
            "the measured first-in-loop value."
        ),
    )
    parser.add_argument(
        "--helion-c-store-mode",
        choices=TCGEN05_C_STORE_MODES,
        default=TCGEN05_C_STORE_MODE_NORMAL,
        help=(
            "Diagnostic-only invalid-output CtaGroup.TWO epilogue mode. "
            "Non-normal values require --skip-correctness 1."
        ),
    )
    parser.add_argument(
        "--helion-aux-load-mode",
        choices=TCGEN05_AUX_LOAD_MODES,
        default=TCGEN05_AUX_LOAD_MODE_SIMT,
        help=(
            "Correctness-preserving tcgen05 aux/residual staging mode. "
            "'tma' uses TMA bulk-tensor loads to stage exact-shape "
            "aux/residual tiles into the aux SMEM ring."
        ),
    )
    parser.add_argument(
        "--helion-sched-consumer-wait-mode",
        choices=TCGEN05_SCHED_CONSUMER_WAIT_MODES,
        default=TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL,
        help=(
            "Diagnostic-only scheduler-broadcast consumer_wait topology for "
            "ROLE_LOCAL_WITH_SCHEDULER. 'warp_leader' gates the wait to lane "
            "0 and reconverges before reading the scheduler mailbox."
        ),
    )
    parser.add_argument(
        "--helion-sched-stage-count",
        type=int,
        choices=TCGEN05_SCHED_STAGE_COUNTS,
        default=1,
        help=(
            "Diagnostic CLC scheduler pipeline depth. A value of 2 uses a "
            "stage-indexed scheduler mailbox for Quack-parity handoff probes."
        ),
    )
    parser.add_argument(
        "--helion-epilogue-layout",
        choices=TCGEN05_EPILOGUE_LAYOUTS,
        default=TCGEN05_EPILOGUE_LAYOUT_NORMAL,
        help=(
            "Diagnostic-only CtaGroup.TWO TMA-store epilogue source layout. "
            "Non-normal values are cache-keyed and preserve correctness."
        ),
    )
    parser.add_argument(
        "--helion-cubin-lineinfo",
        action="store_true",
        help=(
            "Diagnostic-only CtaGroup.TWO CuTe compile option that asks "
            "nvdisasm cubin reports to emit source line information."
        ),
    )
    parser.add_argument("--helion-l2-grouping", type=int, default=4)
    parser.add_argument(
        "--helion-indexing",
        type=_parse_indexing_list,
        default=None,
        help=(
            "Comma-separated indexing list for fixed Helion configs, e.g. "
            "tensor_descriptor,tensor_descriptor,tensor_descriptor. Omit to use "
            "the ConfigSpec default length."
        ),
    )
    parser.add_argument(
        "--helion-pid-type",
        default="flat",
        choices=("flat", "persistent_blocked", "persistent_interleaved", "xyz"),
    )
    parser.add_argument("--helion-cluster-m", type=int, default=1)
    parser.add_argument(
        "--helion-cluster-n",
        type=int,
        default=1,
        help=(
            "tcgen05 cluster_n multicast factor along the N axis. "
            "Default 1 keeps the validated cluster_m∈{1,2} cluster_n=1 "
            "paths. ``--helion-cluster-n 2`` requires "
            "``--helion-cluster-m 2`` and bm=256 (V=2); selects the "
            "Quack-canonical 4-CTA cluster (cute_plan.md §6.12)."
        ),
    )
    parser.add_argument(
        "--helion-ab-stages",
        type=_parse_optional_int,
        default=2,
        help="tcgen05 AB pipeline stages, or auto/none/null/- to omit the config key.",
    )
    parser.add_argument(
        "--helion-acc-stages",
        type=_parse_optional_int,
        default=1,
        help=(
            "tcgen05 accumulator pipeline stages, or auto/none/null/- to omit "
            "the config key."
        ),
    )
    parser.add_argument(
        "--helion-c-stages",
        type=_parse_optional_int,
        default=4,
        help=(
            "tcgen05 C-store pipeline stages, or auto/none/null/- to omit "
            "the config key."
        ),
    )
    parser.add_argument("--helion-num-epi-warps", type=int, default=4)
    parser.add_argument(
        "--helion-l2-swizzle-size",
        type=int,
        default=None,
        choices=(1, 2, 4, 8, 16, 32),
        help=(
            "tcgen05 L2 tile-scheduler swizzle size (Quack "
            "``max_swizzle_size`` equivalent). Default ``None`` omits "
            "the config key (CuTe scheduler defaults to swizzle_size=1). "
            "Legal values: 1/2/4/8/16/32. ``1`` = no swizzle "
            "(byte-identity-preserved); larger values group consecutive "
            "cluster linear-IDs along the slow raster axis to improve "
            "L2 reuse on bandwidth-bound shapes (cute_plan.md §7.6.7)."
        ),
    )
    parser.add_argument(
        "--helion-strategy",
        choices=("role_local_monolithic", "role_local_with_scheduler"),
        default=None,
        help=(
            "Optional tcgen05 lowering strategy override. "
            "'role_local_with_scheduler' implicitly sets "
            "tcgen05_warp_spec_scheduler_warps=1."
        ),
    )
    parser.add_argument(
        "--helion-c-input-warps",
        type=int,
        choices=(0, 1),
        default=None,
        help="Optional tcgen05 C-input warp override for fixed Helion configs.",
    )
    parser.add_argument(
        "--helion-persistence-model",
        choices=("static_persistent", "clc_persistent"),
        default=None,
        help=(
            "Optional tcgen05 persistence-model override. G2-H "
            "(cute_plan.md) introduces 'clc_persistent' which "
            "uses 'nvvm.clusterlaunchcontrol_try_cancel' for dynamic "
            "persistent scheduling on sm_100+. Requires "
            "--helion-strategy=role_local_with_scheduler."
        ),
    )
    parser.add_argument("--helion-require-tcgen05", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--helion-range-flattens",
        type=_parse_optional_bool_list,
        default=list(_HELION_DEFAULT_RANGE_FLATTENS),
        help="Comma-separated outer,inner range flatten flags: true,false,none.",
    )
    parser.add_argument(
        "--helion-range-multi-buffers",
        type=_parse_optional_bool_list,
        default=list(_HELION_DEFAULT_RANGE_MULTI_BUFFERS),
        help="Comma-separated outer,inner range multi-buffer flags: true,false,none.",
    )
    parser.add_argument(
        "--helion-range-warp-specializes",
        type=_parse_optional_bool_list,
        default=list(_HELION_DEFAULT_RANGE_WARP_SPECIALIZES),
        help=(
            "Comma-separated outer,inner range warp-specialize flags: true,false,none."
        ),
    )
    parser.add_argument(
        "--helion-range-num-stages",
        type=_parse_int_list,
        default=list(_HELION_DEFAULT_RANGE_NUM_STAGES),
        help="Comma-separated outer,inner range stage counts.",
    )
    parser.add_argument(
        "--helion-range-unroll-factors",
        type=_parse_int_list,
        default=list(_HELION_DEFAULT_RANGE_UNROLL_FACTORS),
        help="Comma-separated outer,inner range unroll factors.",
    )

    return parser.parse_args()


def _uses_invalid_output_diagnostic_mode(args: argparse.Namespace) -> bool:
    if (
        args.helion_c_store_mode != TCGEN05_C_STORE_MODE_NORMAL
        or args.helion_acc_producer_mode != TCGEN05_ACC_PRODUCER_MODE_NORMAL
    ):
        return True
    return args.ncu_target_config in {
        "two_cta_skip_epilogue_store",
        "two_cta_skip_umma",
    } and (
        args.helion_two_cta_ncu_report
        or args.helion_two_cta_ncu_target
        or args.helion_two_cta_cubin_sass_report
    )


def _validate_args(args: argparse.Namespace) -> None:
    special_modes = (
        args.helion_two_cta_diagnostic_sweep,
        args.helion_two_cta_codegen_report,
        args.helion_two_cta_ncu_report,
        args.helion_two_cta_cubin_sass_report,
        args.helion_two_cta_ncu_target,
    )
    if sum(bool(mode) for mode in special_modes) > 1:
        raise SystemExit(
            "choose only one of --helion-two-cta-diagnostic-sweep, "
            "--helion-two-cta-codegen-report, --helion-two-cta-ncu-report, "
            "or --helion-two-cta-cubin-sass-report"
        )
    if (args.helion_two_cta_ncu_report or args.helion_two_cta_ncu_target) and (
        args.ncu_warmup_launches < 0 or args.ncu_profile_launches < 1
    ):
        raise SystemExit(
            "--ncu-warmup-launches must be >= 0 and --ncu-profile-launches must be >= 1"
        )
    if args.ncu_source_top < 1:
        raise SystemExit("--ncu-source-top must be >= 1")
    if args.ncu_source_report and not args.helion_two_cta_ncu_report:
        raise SystemExit("--ncu-source-report requires --helion-two-cta-ncu-report")
    if _uses_invalid_output_diagnostic_mode(args):
        if not args.skip_correctness:
            raise SystemExit(
                "Invalid-output CtaGroup.TWO diagnostic modes change output "
                "correctness; pass --skip-correctness 1 for diagnostic-only runs."
            )
    if args.helion_two_cta_ncu_target:
        if args.helion_backend != "cute":
            raise SystemExit(
                "--helion-two-cta-ncu-target requires --helion-backend cute"
            )
        return
    if not args.helion_two_cta_diagnostic_sweep:
        if args.helion_two_cta_codegen_report:
            if args.print_codegen:
                raise SystemExit(
                    "--helion-two-cta-codegen-report already prints codegen "
                    "markers; omit --print-codegen."
                )
            if args.helion_backend != "cute":
                raise SystemExit(
                    "--helion-two-cta-codegen-report requires --helion-backend cute"
                )
            if args.impl not in ("all", "helion-cute"):
                raise SystemExit(
                    "--helion-two-cta-codegen-report only supports "
                    "--impl all or --impl helion-cute"
                )
        if args.helion_two_cta_ncu_report:
            if args.print_codegen:
                raise SystemExit(
                    "--helion-two-cta-ncu-report does not support "
                    "--print-codegen; use --helion-two-cta-codegen-report "
                    "for generated source diagnostics."
                )
            if args.helion_backend != "cute":
                raise SystemExit(
                    "--helion-two-cta-ncu-report requires --helion-backend cute"
                )
            if args.impl not in ("all", "helion-cute"):
                raise SystemExit(
                    "--helion-two-cta-ncu-report only supports "
                    "--impl all or --impl helion-cute"
                )
        if args.helion_two_cta_cubin_sass_report:
            if args.helion_backend != "cute":
                raise SystemExit(
                    "--helion-two-cta-cubin-sass-report requires --helion-backend cute"
                )
            if args.print_codegen:
                raise SystemExit(
                    "--helion-two-cta-cubin-sass-report does not support "
                    "--print-codegen; use --helion-two-cta-codegen-report "
                    "for generated source diagnostics."
                )
            if args.impl not in ("all", "helion-cute"):
                raise SystemExit(
                    "--helion-two-cta-cubin-sass-report only supports "
                    "--impl all or --impl helion-cute"
                )
            if args.ncu_target_config not in _HELION_TWO_CTA_NCU_TARGET_CONFIGS:
                raise SystemExit(
                    "--helion-two-cta-cubin-sass-report only supports Helion "
                    f"target configs: {', '.join(_HELION_TWO_CTA_NCU_TARGET_CONFIGS)}"
                )
        return
    if args.print_codegen:
        raise SystemExit(
            "--helion-two-cta-diagnostic-sweep does not support "
            "--print-codegen; run a single fixed Helion config with "
            "--print-codegen instead."
        )
    if args.impl not in ("all", "helion-cute"):
        raise SystemExit(
            "--helion-two-cta-diagnostic-sweep only supports "
            "--impl all or --impl helion-cute"
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.helion_two_cta_ncu_target:
        payload = _run_helion_two_cta_ncu_target(args)
        if args.json:
            print(json.dumps(payload))
        else:
            print(json.dumps(payload, indent=2))
        return

    if args.helion_two_cta_codegen_report:
        payload = _run_helion_two_cta_codegen_report(args)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_codegen_report(payload)
        return

    if args.helion_two_cta_ncu_report:
        payload = _run_helion_two_cta_ncu_report(args)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_ncu_report(payload)
        return

    if args.helion_two_cta_cubin_sass_report:
        payload = _run_helion_two_cta_cubin_sass_report(args)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_cubin_sass_report(payload)
        return

    if args.helion_two_cta_diagnostic_sweep:
        payload = _run_helion_two_cta_diagnostic_sweep(args)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_summary(payload)
        return

    if args.impl == "all":
        payload = _run_all(args)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_summary(payload)
        return

    result = _run_impl(args)
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
