"""Pin the semantics of the shared AST helpers in ``_ast_pass_utils``.

The expected values were produced by running every legacy private copy of
each helper (before the copies were replaced) over the same synthetic corpus;
they are literals here on purpose so the tests stay an oracle after the
copies are gone.
"""

from __future__ import annotations

import ast

import pytest

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.cute._ast_pass_utils import assigned_name
from helion._compiler.cute._ast_pass_utils import call_path
from helion._compiler.cute._ast_pass_utils import call_path_str
from helion._compiler.cute._ast_pass_utils import constexpr_v_loop
from helion._compiler.cute._ast_pass_utils import expand_lane
from helion._compiler.cute._ast_pass_utils import exposed_reads
from helion._compiler.cute._ast_pass_utils import is_vloop
from helion._compiler.cute._ast_pass_utils import live_in
from helion._compiler.cute._ast_pass_utils import loop_extent
from helion._compiler.cute._ast_pass_utils import names_read
from helion._compiler.cute._ast_pass_utils import names_stored_or_deleted
from helion._compiler.cute._ast_pass_utils import names_written
from helion._compiler.cute._ast_pass_utils import parse_expr
from helion._compiler.cute._ast_pass_utils import rename_names
from helion._compiler.cute._ast_pass_utils import reparse_expr
from helion._compiler.cute._ast_pass_utils import replace_call
from helion._compiler.cute._ast_pass_utils import replace_node
from helion._compiler.cute._ast_pass_utils import same_expr
from helion._compiler.cute._ast_pass_utils import substitute_names

LOOPS = [
    "for v in range(4): pass",
    "for v in range(1): pass",
    "for v in range(0): pass",
    "for v in range(2): pass",
    "for v in range(8): pass",
    "for v in range(True): pass",
    "for v in range(-1): pass",
    "for v in cutlass.range_constexpr(4): pass",
    "for v in cutlass.range_constexpr(1): pass",
    "for v in cutlass.range_constexpr(0): pass",
    "for v in cutlass.range_constexpr(2): pass",
    "for v in cutlass.range_constexpr(True): pass",
    "for v in cutlass.range_constexpr(N): pass",
    "for v in cutlass.range_constexpr(cutlass.Int32(4)): pass",
    "for v in cutlass.range_constexpr(4, 8): pass",
    "for v in cutlass.range_constexpr(7, -1, -1): pass",
    "for v in range(4, unroll=1): pass",
    "for v, w in range(4): pass",
    "for v in range(4):\n    pass\nelse:\n    pass",
    "for v in other.range_constexpr(4): pass",
    "for v in range(4.0): pass",
    "for v in cutlass.range(4): pass",
    "for v in range_constexpr(4): pass",
    "for v in cutlass.range_constexpr(4):\n    pass\nelse:\n    pass",
    "for v, w in cutlass.range_constexpr(4): pass",
    "for v in cutlass.range_constexpr(4, unroll=1): pass",
]

# Rows whose spelling never occurs in generated lane/V loops (bool extents,
# keywords, tuple targets, else clauses).  The canonical predicate rejects
# them; the legacy parsers disagreed with each other there.
OUTSIDE_GENERATED_GRAMMAR = [5, 11, 16, 17, 18, 23, 24, 25]

# legacy parser -> its result per LOOPS row before the copies were replaced
LEGACY_LOOP_TABLE = {
    "factor_affine_reductions._constexpr_loop": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        None,
        2,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "fold_lane_reductions._constexpr_extent": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        4,
    ],
    "fold_lane_reductions._rolled_extent": [
        4,
        1,
        0,
        2,
        8,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "fuse_fp8_pair_decode._is_constexpr_v_loop": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        4,
    ],
    "hoist_lane_invariant_reductions._is_positive_constexpr_loop": [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ],
    "hoist_lane_invariant_reductions._static_lane_extent": [
        4,
        1,
        0,
        2,
        8,
        True,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        None,
        None,
        4,
        4,
        4,
        None,
        None,
        None,
        None,
        4,
        4,
        4,
    ],
    "hoist_warp_reduce._is_constexpr_v_loop": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        4,
        7,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        4,
    ],
    "hoist_warp_reduce._is_static_range_loop": [
        4,
        1,
        0,
        2,
        8,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "lane_loop_distribution._is_constexpr_loop": [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ],
    "licm_profitability._constexpr_extent": [
        4,
        1,
        None,
        2,
        8,
        None,
        None,
        4,
        1,
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
    ],
    "merge_sibling_v_loops._is_constexpr_v_loop": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        4,
        7,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        4,
    ],
    "pack_bf16_recurrence._constexpr_trip_count": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        None,
    ],
    "persistent_branch_vec._range_extent": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
    ],
    "prefetch_pointwise_packets._literal_range_extent": [
        4,
        1,
        0,
        2,
        8,
        None,
        None,
        4,
        1,
        0,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
    ],
    "register_tile_reductions._constexpr_extent": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        None,
        2,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "resident_sequence._literal_loop_extent": [
        4,
        1,
        None,
        2,
        8,
        None,
        None,
        4,
        1,
        None,
        2,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
    ],
    "sink_vector_loops._is_constexpr_loop": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        1,
        0,
        2,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "sink_vector_loops._lane_loop_extent": [
        4,
        1,
        0,
        2,
        8,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "split_lane_loads._is_const_range_loop": [
        4,
        None,
        None,
        2,
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    "unroll_lane_loads._ascending_range_extent": [
        4,
        1,
        None,
        2,
        8,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
}

# legacy parser -> (kinds, minimum) of the canonical call that replaced it
MIGRATED_LOOP_PARSERS = {
    "factor_affine_reductions._constexpr_loop": (("constexpr",), 1),
    "fold_lane_reductions._constexpr_extent": (("constexpr",), None),
    "fold_lane_reductions._rolled_extent": (("range",), None),
    "fuse_fp8_pair_decode._is_constexpr_v_loop": (("constexpr",), None),
    "hoist_lane_invariant_reductions._is_positive_constexpr_loop": (("constexpr",), 1),
    "hoist_lane_invariant_reductions._static_lane_extent": (
        ("range", "constexpr"),
        None,
    ),
    "hoist_warp_reduce._is_static_range_loop": (("range",), None),
    "lane_loop_distribution._is_constexpr_loop": (("constexpr",), None),
    "licm_profitability._constexpr_extent": (("range", "constexpr"), 1),
    "pack_bf16_recurrence._constexpr_trip_count": (("constexpr",), None),
    "persistent_branch_vec._range_extent": (("constexpr",), 2),
    "prefetch_pointwise_packets._literal_range_extent": (("range", "constexpr"), None),
    "register_tile_reductions._constexpr_extent": (("constexpr",), 1),
    "resident_sequence._literal_loop_extent": (("range", "constexpr"), 1),
    "sink_vector_loops._is_constexpr_loop": (("constexpr",), None),
    "sink_vector_loops._lane_loop_extent": (("range",), None),
    "split_lane_loads._is_const_range_loop": (("range",), 2),
    "unroll_lane_loads._ascending_range_extent": (("range",), 1),
}

BOOL_LOOP_PARSERS = [
    "hoist_lane_invariant_reductions._is_positive_constexpr_loop",
    "lane_loop_distribution._is_constexpr_loop",
]

NAME_CASES = [
    "a = b + c",
    "a[i] = b",
    "a.b = c",
    "a, b = c",
    "a += b",
    "del a",
    "a: int = b",
    "f(a, b=c)",
    "a = cute.arch.load(p, q)",
    "(a := b) + c",
    "[x for x in y]",
    "lambda z: z + w",
    "for i in r:\n    s = s + i",
    "import m",
    "with k as v:\n    pass",
    "a = b if c else d",
]

LEGACY_NAME_TABLE = {
    "(a := b) + c": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["b", "c"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": None,
        "far._names_read": ["b", "c"],
        "far._names_written": ["a"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["b", "c"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["b", "c"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["b", "c"],
        "pbr._name_target": None,
        "prp._assigned": ["a"],
        "prp._assignment": None,
        "prp._names": ["b", "c"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["b", "c"],
        "sr._read_names": ["b", "c"],
    },
    "[x for x in y]": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["x", "y"],
        "dal._stored_names": ["x"],
        "darc._bound_names": ["x"],
        "far._assigned_name": None,
        "far._names_read": ["x", "y"],
        "far._names_written": ["x"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["x", "y"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["x", "y"],
        "pbr._assigned_names": ["x"],
        "pbr._loaded_names": ["x", "y"],
        "pbr._name_target": None,
        "prp._assigned": ["x"],
        "prp._assignment": None,
        "prp._names": ["x", "y"],
        "psl._assigned_names": ["x"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["x"],
        "spb._loaded_names": ["x", "y"],
        "sr._read_names": ["x", "y"],
    },
    "a += b": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["b"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": None,
        "far._names_read": ["b"],
        "far._names_written": ["a"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["b"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["b"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["b"],
        "pbr._name_target": None,
        "prp._assigned": ["a"],
        "prp._assignment": None,
        "prp._names": ["b"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["b"],
        "sr._read_names": ["b"],
    },
    "a = b + c": {
        "apu._assignment_lhs_name": "a",
        "apu._names_read": ["b", "c"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": "a",
        "far._names_read": ["b", "c"],
        "far._names_written": ["a"],
        "ffp._assigned_name": "a",
        "hlir._names_read": ["b", "c"],
        "hwr._names_written": ["a"],
        "mo._cute_names_read": ["b", "c"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["b", "c"],
        "pbr._name_target": "a",
        "prp._assigned": ["a"],
        "prp._assignment": "a",
        "prp._names": ["b", "c"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": "a",
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["b", "c"],
        "sr._read_names": ["b", "c"],
    },
    "a = b if c else d": {
        "apu._assignment_lhs_name": "a",
        "apu._names_read": ["b", "c", "d"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": "a",
        "far._names_read": ["b", "c", "d"],
        "far._names_written": ["a"],
        "ffp._assigned_name": "a",
        "hlir._names_read": ["b", "c", "d"],
        "hwr._names_written": ["a"],
        "mo._cute_names_read": ["b", "c", "d"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["b", "c", "d"],
        "pbr._name_target": "a",
        "prp._assigned": ["a"],
        "prp._assignment": "a",
        "prp._names": ["b", "c", "d"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": "a",
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["b", "c", "d"],
        "sr._read_names": ["b", "c", "d"],
    },
    "a = cute.arch.load(p, q)": {
        "apu._assignment_lhs_name": "a",
        "apu._names_read": ["cute", "p", "q"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": "a",
        "far._names_read": ["cute", "p", "q"],
        "far._names_written": ["a"],
        "ffp._assigned_name": "a",
        "hlir._names_read": ["cute", "p", "q"],
        "hwr._names_written": ["a"],
        "mo._cute_names_read": ["cute", "p", "q"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["cute", "p", "q"],
        "pbr._name_target": "a",
        "prp._assigned": ["a"],
        "prp._assignment": "a",
        "prp._names": ["cute", "p", "q"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": "a",
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["cute", "p", "q"],
        "sr._read_names": ["cute", "p", "q"],
    },
    "a, b = c": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["c"],
        "dal._stored_names": ["a", "b"],
        "darc._bound_names": ["a", "b"],
        "far._assigned_name": None,
        "far._names_read": ["c"],
        "far._names_written": ["a", "b"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["c"],
        "hwr._names_written": ["a", "b"],
        "mo._cute_names_read": ["c"],
        "pbr._assigned_names": ["a", "b"],
        "pbr._loaded_names": ["c"],
        "pbr._name_target": None,
        "prp._assigned": ["a", "b"],
        "prp._assignment": None,
        "prp._names": ["c"],
        "psl._assigned_names": ["a", "b"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["a", "b"],
        "spb._loaded_names": ["c"],
        "sr._read_names": ["c"],
    },
    "a.b = c": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["a", "c"],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": ["a", "c"],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": ["a", "c"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["a", "c"],
        "pbr._assigned_names": [],
        "pbr._loaded_names": ["a", "c"],
        "pbr._name_target": None,
        "prp._assigned": [],
        "prp._assignment": None,
        "prp._names": ["a", "c"],
        "psl._assigned_names": [],
        "sll._assigned_name": None,
        "spb._assigned_names": [],
        "spb._loaded_names": ["a", "c"],
        "sr._read_names": ["a", "c"],
    },
    "a: int = b": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["b", "int"],
        "dal._stored_names": ["a"],
        "darc._bound_names": ["a"],
        "far._assigned_name": None,
        "far._names_read": ["b", "int"],
        "far._names_written": ["a"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["b", "int"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["b", "int"],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": ["b", "int"],
        "pbr._name_target": None,
        "prp._assigned": ["a"],
        "prp._assignment": None,
        "prp._names": ["b", "int"],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["a"],
        "spb._loaded_names": ["b", "int"],
        "sr._read_names": ["b", "int"],
    },
    "a[i] = b": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["a", "b", "i"],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": ["a", "b", "i"],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": ["a", "b", "i"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["a", "b", "i"],
        "pbr._assigned_names": [],
        "pbr._loaded_names": ["a", "b", "i"],
        "pbr._name_target": None,
        "prp._assigned": [],
        "prp._assignment": None,
        "prp._names": ["a", "b", "i"],
        "psl._assigned_names": [],
        "sll._assigned_name": None,
        "spb._assigned_names": [],
        "spb._loaded_names": ["a", "b", "i"],
        "sr._read_names": ["a", "b", "i"],
    },
    "del a": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": [],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": [],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": [],
        "hwr._names_written": [],
        "mo._cute_names_read": [],
        "pbr._assigned_names": ["a"],
        "pbr._loaded_names": [],
        "pbr._name_target": None,
        "prp._assigned": ["a"],
        "prp._assignment": None,
        "prp._names": [],
        "psl._assigned_names": ["a"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["a"],
        "spb._loaded_names": [],
        "sr._read_names": [],
    },
    "f(a, b=c)": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["a", "c", "f"],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": ["a", "c", "f"],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": ["a", "c", "f"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["a", "c", "f"],
        "pbr._assigned_names": [],
        "pbr._loaded_names": ["a", "c", "f"],
        "pbr._name_target": None,
        "prp._assigned": [],
        "prp._assignment": None,
        "prp._names": ["a", "c", "f"],
        "psl._assigned_names": [],
        "sll._assigned_name": None,
        "spb._assigned_names": [],
        "spb._loaded_names": ["a", "c", "f"],
        "sr._read_names": ["a", "c", "f"],
    },
    "for i in r:\n    s = s + i": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["i", "r", "s"],
        "dal._stored_names": ["i", "s"],
        "darc._bound_names": ["i", "s"],
        "far._assigned_name": None,
        "far._names_read": ["i", "r", "s"],
        "far._names_written": ["i", "s"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["i", "r", "s"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["i", "r", "s"],
        "pbr._assigned_names": ["i", "s"],
        "pbr._loaded_names": ["i", "r", "s"],
        "pbr._name_target": None,
        "prp._assigned": ["i", "s"],
        "prp._assignment": None,
        "prp._names": ["i", "r", "s"],
        "psl._assigned_names": ["i", "s"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["i", "s"],
        "spb._loaded_names": ["i", "r", "s"],
        "sr._read_names": ["i", "r", "s"],
    },
    "import m": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": [],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": [],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": [],
        "hwr._names_written": [],
        "mo._cute_names_read": [],
        "pbr._assigned_names": [],
        "pbr._loaded_names": [],
        "pbr._name_target": None,
        "prp._assigned": [],
        "prp._assignment": None,
        "prp._names": [],
        "psl._assigned_names": [],
        "sll._assigned_name": None,
        "spb._assigned_names": [],
        "spb._loaded_names": [],
        "sr._read_names": [],
    },
    "lambda z: z + w": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["w", "z"],
        "dal._stored_names": [],
        "darc._bound_names": [],
        "far._assigned_name": None,
        "far._names_read": ["w", "z"],
        "far._names_written": [],
        "ffp._assigned_name": None,
        "hlir._names_read": ["w", "z"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["w", "z"],
        "pbr._assigned_names": [],
        "pbr._loaded_names": ["w", "z"],
        "pbr._name_target": None,
        "prp._assigned": [],
        "prp._assignment": None,
        "prp._names": ["w", "z"],
        "psl._assigned_names": [],
        "sll._assigned_name": None,
        "spb._assigned_names": [],
        "spb._loaded_names": ["w", "z"],
        "sr._read_names": ["w", "z"],
    },
    "with k as v:\n    pass": {
        "apu._assignment_lhs_name": None,
        "apu._names_read": ["k"],
        "dal._stored_names": ["v"],
        "darc._bound_names": ["v"],
        "far._assigned_name": None,
        "far._names_read": ["k"],
        "far._names_written": ["v"],
        "ffp._assigned_name": None,
        "hlir._names_read": ["k"],
        "hwr._names_written": [],
        "mo._cute_names_read": ["k"],
        "pbr._assigned_names": ["v"],
        "pbr._loaded_names": ["k"],
        "pbr._name_target": None,
        "prp._assigned": ["v"],
        "prp._assignment": None,
        "prp._names": ["k"],
        "psl._assigned_names": ["v"],
        "sll._assigned_name": None,
        "spb._assigned_names": ["v"],
        "spb._loaded_names": ["k"],
        "sr._read_names": ["k"],
    },
}

CALL_CASES = [
    "f(x)",
    "a.b(x)",
    "a.b.c(x)",
    "a[0].b(x)",
    "f()(x)",
    "(a).b(x)",
    "a.b",
    "a",
]

LEGACY_CALL_TABLE = {
    "(a).b(x)": {
        "dal._qualified_name": "a.b",
        "darc._qualified_name": "a.b",
        "far._ast_call_path": "a.b",
        "hlir._call_path": "a.b",
        "licm._call_path": "a.b",
        "pbr._call_path": ["'a'", "'b'"],
        "psl._call_path": ["'a'", "'b'"],
        "spb._call_path": ["'a'", "'b'"],
    },
    "a": {
        "dal._qualified_name": "a",
        "darc._qualified_name": "a",
        "far._ast_call_path": "a",
        "hlir._call_path": "a",
        "licm._call_path": "a",
        "pbr._call_path": ["'a'"],
        "psl._call_path": ["'a'"],
        "spb._call_path": ["'a'"],
    },
    "a.b": {
        "dal._qualified_name": "a.b",
        "darc._qualified_name": "a.b",
        "far._ast_call_path": "a.b",
        "hlir._call_path": "a.b",
        "licm._call_path": "a.b",
        "pbr._call_path": ["'a'", "'b'"],
        "psl._call_path": ["'a'", "'b'"],
        "spb._call_path": ["'a'", "'b'"],
    },
    "a.b(x)": {
        "dal._qualified_name": "a.b",
        "darc._qualified_name": "a.b",
        "far._ast_call_path": "a.b",
        "hlir._call_path": "a.b",
        "licm._call_path": "a.b",
        "pbr._call_path": ["'a'", "'b'"],
        "psl._call_path": ["'a'", "'b'"],
        "spb._call_path": ["'a'", "'b'"],
    },
    "a.b.c(x)": {
        "dal._qualified_name": "a.b.c",
        "darc._qualified_name": "a.b.c",
        "far._ast_call_path": "a.b.c",
        "hlir._call_path": "a.b.c",
        "licm._call_path": "a.b.c",
        "pbr._call_path": ["'a'", "'b'", "'c'"],
        "psl._call_path": ["'a'", "'b'", "'c'"],
        "spb._call_path": ["'a'", "'b'", "'c'"],
    },
    "a[0].b(x)": {
        "dal._qualified_name": None,
        "darc._qualified_name": None,
        "far._ast_call_path": None,
        "hlir._call_path": None,
        "licm._call_path": None,
        "pbr._call_path": None,
        "psl._call_path": None,
        "spb._call_path": None,
    },
    "f()(x)": {
        "dal._qualified_name": None,
        "darc._qualified_name": None,
        "far._ast_call_path": None,
        "hlir._call_path": None,
        "licm._call_path": None,
        "pbr._call_path": None,
        "psl._call_path": None,
        "spb._call_path": None,
    },
    "f(x)": {
        "dal._qualified_name": "f",
        "darc._qualified_name": "f",
        "far._ast_call_path": "f",
        "hlir._call_path": "f",
        "licm._call_path": "f",
        "pbr._call_path": ["'f'"],
        "psl._call_path": ["'f'"],
        "spb._call_path": ["'f'"],
    },
}

STATEMENT_CASES = {
    "ann": "a: int = x\nb = a",
    "augassign": "acc += x\ny = acc",
    "del": "a = x\ndel a\nb = a",
    "for_carried": "for i in cutlass.range_constexpr(4):\n"
    "    acc = acc + p[i]\n"
    "out = acc",
    "for_defines": "for i in range(4):\n    w = i * 2\nq = w",
    "for_else": "for i in range(4):\n    a = i\nelse:\n    b = a\nc = b",
    "for_target_read": "for i in range(n):\n    z = i\nu = i",
    "if_both": "if t:\n    a = x\nelse:\n    a = y\nb = a",
    "if_elif": "if t:\n    a = x\nelif u:\n    a = y\nelse:\n    a = z\nb = a",
    "if_nested_for": "if t:\n    for i in range(4):\n        s = s + q[i]\nr = s",
    "if_one": "if t:\n    a = x\nb = a + z",
    "if_reads_defined": "a = x\nif a:\n    b = a\nelse:\n    b = y\nc = b",
    "loop_reads_later_def": "for i in range(4):\n    a = b\n    b = i\nc = b",
    "method_store": "ptr.store(v)\nk = v + m",
    "nested_if_in_for": "for i in range(4):\n"
    "    if i > 2:\n"
    "        m = i\n"
    "    n = m\n"
    "o = n",
    "reads_own": "a = a + 1\nb = a",
    "straight": "a = x + 1\nb = a * y\nc = b",
    "subscript_store": "t[i] = v\nk = t[j]",
    "tuple_assign": "a, b = f(x)\nc = a + b",
    "while": "while c:\n    a = a + 1\nb = a",
}

LEGACY_EXPOSED_TABLE = {
    "ann": {
        "isd": ["a", "int", "x"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "int"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A", "int"],
        "lld|['b', 'x']|[]|False": ["a", "int"],
        "lld|['b', 'x']|[]|True": ["a", "int"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "int", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A", "int", "x"],
        "lld|[]|[]|False": ["a", "int", "x"],
        "lld|[]|[]|True": ["a", "int", "x"],
        "rs|['b', 'c', 'out']": ["a", "c", "int", "out", "x"],
        "rs|[]": ["a", "int", "x"],
        "sink|['b', 'x']": ["int"],
        "sink|[]": ["int", "x"],
    },
    "augassign": {
        "isd": ["acc", "x"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["acc"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["acc"],
        "lld|['b', 'x']|[]|False": ["acc"],
        "lld|['b', 'x']|[]|True": ["acc"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["acc", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["acc", "x"],
        "lld|[]|[]|False": ["acc", "x"],
        "lld|[]|[]|True": ["acc", "x"],
        "rs|['b', 'c', 'out']": ["acc", "b", "c", "out", "x"],
        "rs|[]": ["acc", "x"],
        "sink|['b', 'x']": ["acc"],
        "sink|[]": ["acc", "x"],
    },
    "del": {
        "isd": ["x"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": [],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": [],
        "lld|['b', 'x']|[]|False": [],
        "lld|['b', 'x']|[]|True": [],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["x"],
        "lld|[]|[]|False": ["x"],
        "lld|[]|[]|True": ["x"],
        "rs|['b', 'c', 'out']": ["c", "out", "x"],
        "rs|[]": ["x"],
        "sink|['b', 'x']": [],
        "sink|[]": ["x"],
    },
    "for_carried": {
        "isd": ["acc", "cutlass", "i", "p"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["acc", "cutlass", "i", "p"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["acc", "cutlass", "p"],
        "lld|['b', 'x']|[]|False": ["acc", "cutlass", "i", "p"],
        "lld|['b', 'x']|[]|True": ["acc", "cutlass", "p"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["acc", "cutlass", "i", "p"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["acc", "cutlass", "p"],
        "lld|[]|[]|False": ["acc", "cutlass", "i", "p"],
        "lld|[]|[]|True": ["acc", "cutlass", "p"],
        "rs|['b', 'c', 'out']": ["acc", "b", "c", "cutlass", "p"],
        "rs|[]": ["acc", "cutlass", "p"],
        "sink|['b', 'x']": ["acc", "cutlass", "p"],
        "sink|[]": ["acc", "cutlass", "p"],
    },
    "for_defines": {
        "isd": ["i", "range", "w"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "range", "w"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["range", "w"],
        "lld|['b', 'x']|[]|False": ["i", "range", "w"],
        "lld|['b', 'x']|[]|True": ["range", "w"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["i", "range", "w"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["range", "w"],
        "lld|[]|[]|False": ["i", "range", "w"],
        "lld|[]|[]|True": ["range", "w"],
        "rs|['b', 'c', 'out']": ["b", "c", "out", "range", "w"],
        "rs|[]": ["range", "w"],
        "sink|['b', 'x']": ["range"],
        "sink|[]": ["range"],
    },
    "for_else": {
        "isd": ["a", "b", "i", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "i", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["range"],
        "lld|['b', 'x']|[]|False": ["a", "i", "range"],
        "lld|['b', 'x']|[]|True": ["range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "b", "i", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["b", "range"],
        "lld|[]|[]|False": ["a", "b", "i", "range"],
        "lld|[]|[]|True": ["b", "range"],
        "rs|['b', 'c', 'out']": ["a", "out", "range"],
        "rs|[]": ["a", "range"],
        "sink|['b', 'x']": ["range"],
        "sink|[]": ["range"],
    },
    "for_target_read": {
        "isd": ["i", "n", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "n", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["i", "n", "range"],
        "lld|['b', 'x']|[]|False": ["i", "n", "range"],
        "lld|['b', 'x']|[]|True": ["i", "n", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["i", "n", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["i", "n", "range"],
        "lld|[]|[]|False": ["i", "n", "range"],
        "lld|[]|[]|True": ["i", "n", "range"],
        "rs|['b', 'c', 'out']": ["b", "c", "i", "n", "out", "range"],
        "rs|[]": ["i", "n", "range"],
        "sink|['b', 'x']": ["i", "n", "range"],
        "sink|[]": ["i", "n", "range"],
    },
    "if_both": {
        "isd": ["a", "t", "x", "y"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["Y", "t"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["Y", "t"],
        "lld|['b', 'x']|[]|False": ["t", "y"],
        "lld|['b', 'x']|[]|True": ["t", "y"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["Y", "t", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["Y", "t", "x"],
        "lld|[]|[]|False": ["t", "x", "y"],
        "lld|[]|[]|True": ["t", "x", "y"],
        "rs|['b', 'c', 'out']": ["c", "out", "t", "x", "y"],
        "rs|[]": ["t", "x", "y"],
        "sink|['b', 'x']": ["t", "y"],
        "sink|[]": ["t", "x", "y"],
    },
    "if_elif": {
        "isd": ["a", "t", "u", "x", "y", "z"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "Y", "t", "u", "z"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A", "Y", "t", "u", "z"],
        "lld|['b', 'x']|[]|False": ["a", "t", "u", "y", "z"],
        "lld|['b', 'x']|[]|True": ["a", "t", "u", "y", "z"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "Y", "t", "u", "x", "z"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A", "Y", "t", "u", "x", "z"],
        "lld|[]|[]|False": ["a", "t", "u", "x", "y", "z"],
        "lld|[]|[]|True": ["a", "t", "u", "x", "y", "z"],
        "rs|['b', 'c', 'out']": ["c", "out", "t", "u", "x", "y", "z"],
        "rs|[]": ["t", "u", "x", "y", "z"],
        "sink|['b', 'x']": ["t", "u", "y", "z"],
        "sink|[]": ["t", "u", "x", "y", "z"],
    },
    "if_nested_for": {
        "isd": ["i", "q", "range", "s", "t"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "q", "range", "s", "t"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["q", "range", "s", "t"],
        "lld|['b', 'x']|[]|False": ["i", "q", "range", "s", "t"],
        "lld|['b', 'x']|[]|True": ["q", "range", "s", "t"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["i", "q", "range", "s", "t"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["q", "range", "s", "t"],
        "lld|[]|[]|False": ["i", "q", "range", "s", "t"],
        "lld|[]|[]|True": ["q", "range", "s", "t"],
        "rs|['b', 'c', 'out']": ["b", "c", "out", "q", "range", "s", "t"],
        "rs|[]": ["q", "range", "s", "t"],
        "sink|['b', 'x']": ["q", "range", "s", "t"],
        "sink|[]": ["q", "range", "s", "t"],
    },
    "if_one": {
        "isd": ["a", "t", "x", "z"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "t", "z"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A", "t", "z"],
        "lld|['b', 'x']|[]|False": ["a", "t", "z"],
        "lld|['b', 'x']|[]|True": ["a", "t", "z"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "t", "x", "z"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A", "t", "x", "z"],
        "lld|[]|[]|False": ["a", "t", "x", "z"],
        "lld|[]|[]|True": ["a", "t", "x", "z"],
        "rs|['b', 'c', 'out']": ["a", "c", "out", "t", "x", "z"],
        "rs|[]": ["a", "t", "x", "z"],
        "sink|['b', 'x']": ["a", "t", "z"],
        "sink|[]": ["a", "t", "x", "z"],
    },
    "if_reads_defined": {
        "isd": ["b", "x", "y"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["Y"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["Y"],
        "lld|['b', 'x']|[]|False": ["y"],
        "lld|['b', 'x']|[]|True": ["y"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["Y", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["Y", "x"],
        "lld|[]|[]|False": ["x", "y"],
        "lld|[]|[]|True": ["x", "y"],
        "rs|['b', 'c', 'out']": ["out", "x", "y"],
        "rs|[]": ["x", "y"],
        "sink|['b', 'x']": ["y"],
        "sink|[]": ["x", "y"],
    },
    "loop_reads_later_def": {
        "isd": ["b", "i", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["range"],
        "lld|['b', 'x']|[]|False": ["i", "range"],
        "lld|['b', 'x']|[]|True": ["range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["b", "i", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["b", "range"],
        "lld|[]|[]|False": ["b", "i", "range"],
        "lld|[]|[]|True": ["b", "range"],
        "rs|['b', 'c', 'out']": ["b", "out", "range"],
        "rs|[]": ["b", "range"],
        "sink|['b', 'x']": ["range"],
        "sink|[]": ["b", "range"],
    },
    "method_store": {
        "isd": ["m", "ptr", "v"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["m", "ptr", "v"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["m", "ptr", "v"],
        "lld|['b', 'x']|[]|False": ["m", "ptr", "v"],
        "lld|['b', 'x']|[]|True": ["m", "ptr", "v"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["m", "ptr", "v"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["m", "ptr", "v"],
        "lld|[]|[]|False": ["m", "ptr", "v"],
        "lld|[]|[]|True": ["m", "ptr", "v"],
        "rs|['b', 'c', 'out']": ["b", "c", "m", "out", "ptr", "v"],
        "rs|[]": ["m", "ptr", "v"],
        "sink|['b', 'x']": ["m", "ptr", "v"],
        "sink|[]": ["m", "ptr", "v"],
    },
    "nested_if_in_for": {
        "isd": ["i", "m", "n", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "m", "n", "range"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["m", "n", "range"],
        "lld|['b', 'x']|[]|False": ["i", "m", "n", "range"],
        "lld|['b', 'x']|[]|True": ["m", "n", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["i", "m", "n", "range"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["m", "n", "range"],
        "lld|[]|[]|False": ["i", "m", "n", "range"],
        "lld|[]|[]|True": ["m", "n", "range"],
        "rs|['b', 'c', 'out']": ["b", "c", "m", "n", "out", "range"],
        "rs|[]": ["m", "n", "range"],
        "sink|['b', 'x']": ["m", "range"],
        "sink|[]": ["m", "range"],
    },
    "reads_own": {
        "isd": ["a"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A"],
        "lld|['b', 'x']|[]|False": ["a"],
        "lld|['b', 'x']|[]|True": ["a"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A"],
        "lld|[]|[]|False": ["a"],
        "lld|[]|[]|True": ["a"],
        "rs|['b', 'c', 'out']": ["a", "c", "out"],
        "rs|[]": ["a"],
        "sink|['b', 'x']": ["a"],
        "sink|[]": ["a"],
    },
    "straight": {
        "isd": ["x", "y"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["Y"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["Y"],
        "lld|['b', 'x']|[]|False": ["y"],
        "lld|['b', 'x']|[]|True": ["y"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["Y", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["Y", "x"],
        "lld|[]|[]|False": ["x", "y"],
        "lld|[]|[]|True": ["x", "y"],
        "rs|['b', 'c', 'out']": ["out", "x", "y"],
        "rs|[]": ["x", "y"],
        "sink|['b', 'x']": ["y"],
        "sink|[]": ["x", "y"],
    },
    "subscript_store": {
        "isd": ["i", "j", "t", "v"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["i", "j", "t", "v"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["i", "j", "t", "v"],
        "lld|['b', 'x']|[]|False": ["i", "j", "t", "v"],
        "lld|['b', 'x']|[]|True": ["i", "j", "t", "v"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["i", "j", "t", "v"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["i", "j", "t", "v"],
        "lld|[]|[]|False": ["i", "j", "t", "v"],
        "lld|[]|[]|True": ["i", "j", "t", "v"],
        "rs|['b', 'c', 'out']": ["b", "c", "i", "j", "out", "t", "v"],
        "rs|[]": ["i", "j", "t", "v"],
        "sink|['b', 'x']": ["i", "j", "t", "v"],
        "sink|[]": ["i", "j", "t", "v"],
    },
    "tuple_assign": {
        "isd": ["a", "b", "f", "x"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "f"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A", "f"],
        "lld|['b', 'x']|[]|False": ["a", "f"],
        "lld|['b', 'x']|[]|True": ["a", "f"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "b", "f", "x"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A", "b", "f", "x"],
        "lld|[]|[]|False": ["a", "b", "f", "x"],
        "lld|[]|[]|True": ["a", "b", "f", "x"],
        "rs|['b', 'c', 'out']": ["a", "b", "f", "out", "x"],
        "rs|[]": ["a", "b", "f", "x"],
        "sink|['b', 'x']": ["f"],
        "sink|[]": ["f", "x"],
    },
    "while": {
        "isd": ["a", "c"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|False": ["A", "c"],
        "lld|['b', 'x']|[('a', 'A'), ('y', 'Y')]|True": ["A", "c"],
        "lld|['b', 'x']|[]|False": ["a", "c"],
        "lld|['b', 'x']|[]|True": ["a", "c"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|False": ["A", "c"],
        "lld|[]|[('a', 'A'), ('y', 'Y')]|True": ["A", "c"],
        "lld|[]|[]|False": ["a", "c"],
        "lld|[]|[]|True": ["a", "c"],
        "rs|['b', 'c', 'out']": ["a", "c", "out"],
        "rs|[]": ["a", "c"],
        "sink|['b', 'x']": ["a", "c"],
        "sink|[]": ["a", "c"],
    },
}


def _loops() -> list[ast.For]:
    return [ast.parse(source).body[0] for source in LOOPS]  # type: ignore[misc]


@pytest.mark.parametrize("site", sorted(MIGRATED_LOOP_PARSERS))
def test_loop_extent_reproduces_each_migrated_parser(site: str) -> None:
    kinds, minimum = MIGRATED_LOOP_PARSERS[site]
    legacy = LEGACY_LOOP_TABLE[site]
    for index, loop in enumerate(_loops()):
        got = loop_extent(loop, kinds=kinds, minimum=minimum)
        if site in BOOL_LOOP_PARSERS:
            got = got is not None
        if index in OUTSIDE_GENERATED_GRAMMAR:
            assert got in (None, False), (site, LOOPS[index], got)
        else:
            assert got == legacy[index], (site, LOOPS[index], got, legacy[index])


def test_is_vloop_is_the_lane_loop_distribution_predicate() -> None:
    legacy = LEGACY_LOOP_TABLE["lane_loop_distribution._is_constexpr_loop"]
    for index, loop in enumerate(_loops()):
        if index in OUTSIDE_GENERATED_GRAMMAR:
            assert not is_vloop(loop)
        else:
            assert is_vloop(loop) == legacy[index], LOOPS[index]
    assert not is_vloop(ast.parse("x = 1").body[0])


def test_constexpr_v_loop_keeps_the_legacy_grammar() -> None:
    # The warp-reduce hoist's parser (any target, keywords, else clauses, bool
    # literals, multi-argument spellings) and the sibling V-loop merge's, which
    # additionally requires a plain Name target.
    hoist = LEGACY_LOOP_TABLE["hoist_warp_reduce._is_constexpr_v_loop"]
    merge = LEGACY_LOOP_TABLE["merge_sibling_v_loops._is_constexpr_v_loop"]
    for index, loop in enumerate(_loops()):
        match = constexpr_v_loop(loop)
        got = None if match is None else match[1]
        assert got == hoist[index], LOOPS[index]
        if match is not None and not isinstance(match[0].target, ast.Name):
            got = None
        assert got == merge[index], LOOPS[index]
    assert constexpr_v_loop(ast.parse("x = 1").body[0]) is None


def test_loop_extent_rejects_multi_argument_and_wrapped_extents() -> None:
    for source in (
        "for v in cutlass.range_constexpr(4, 8): pass",
        "for v in cutlass.range_constexpr(7, -1, -1): pass",
        "for v in cutlass.range_constexpr(cutlass.Int32(4)): pass",
        "for v in cutlass.range_constexpr(N): pass",
        "for v in range(4.0): pass",
        "for v in other.range_constexpr(4): pass",
    ):
        assert loop_extent(ast.parse(source).body[0], minimum=None) is None


@pytest.mark.parametrize("source", NAME_CASES)
def test_name_collectors_pin_legacy_copies(source: str) -> None:
    tree = ast.parse(source)
    expected = LEGACY_NAME_TABLE[source]
    reads = sorted(names_read(tree))
    for legacy in (
        "far._names_read",
        "mo._cute_names_read",
        "pbr._loaded_names",
        "prp._names",
        "sr._read_names",
        "spb._loaded_names",
        "apu._names_read",
        "hlir._names_read",
    ):
        assert reads == expected[legacy], legacy
    written = sorted(names_written(tree))
    for legacy in ("far._names_written", "dal._stored_names", "darc._bound_names"):
        assert written == expected[legacy], legacy
    stored_or_deleted = sorted(names_stored_or_deleted(tree))
    for legacy in (
        "pbr._assigned_names",
        "prp._assigned",
        "spb._assigned_names",
        "psl._assigned_names",
    ):
        assert stored_or_deleted == expected[legacy], legacy
    assigned = assigned_name(tree.body[0])
    for legacy in (
        "far._assigned_name",
        "ffp._assigned_name",
        "pbr._name_target",
        "prp._assignment",
        "sll._assigned_name",
        "apu._assignment_lhs_name",
    ):
        assert assigned == expected[legacy], legacy


@pytest.mark.parametrize("source", CALL_CASES)
def test_call_path_pins_legacy_copies(source: str) -> None:
    expression = ast.parse(source, mode="eval").body
    node = expression.func if isinstance(expression, ast.Call) else expression
    expected = LEGACY_CALL_TABLE[source]
    path = call_path(node)
    tuple_form = None if path is None else [repr(part) for part in path]
    for legacy in ("pbr._call_path", "spb._call_path", "psl._call_path"):
        assert tuple_form == expected[legacy], legacy
    for legacy in (
        "hlir._call_path",
        "licm._call_path",
        "far._ast_call_path",
        "dal._qualified_name",
        "darc._qualified_name",
    ):
        assert call_path_str(node) == expected[legacy], legacy


def _single_assigned(statement: ast.AST) -> tuple[str, ...]:
    name = assigned_name(statement)
    return () if name is None else (name,)


@pytest.mark.parametrize("case", sorted(STATEMENT_CASES))
def test_exposed_reads_pins_lane_loop_distribution(case: str) -> None:
    statements = ast.parse(STATEMENT_CASES[case]).body
    expected = LEGACY_EXPOSED_TABLE[case]
    for defined in (frozenset(), frozenset({"x", "b"})):
        for renames in ({}, {"y": "Y", "a": "A"}):
            for loop_bodies in (False, True):
                key = f"lld|{sorted(defined)}|{sorted(renames.items())}|{loop_bodies}"
                got = exposed_reads(
                    statements,
                    defined=defined,
                    renames=renames,
                    defines="assign",
                    loops="may_skip" if loop_bodies else "opaque",
                    branches="analyze",
                )
                assert sorted(got) == expected[key], key


@pytest.mark.parametrize("case", sorted(STATEMENT_CASES))
def test_exposed_reads_pins_sink_vector_loops(case: str) -> None:
    statements = ast.parse(STATEMENT_CASES[case]).body
    expected = LEGACY_EXPOSED_TABLE[case]
    for defined in (frozenset(), frozenset({"x", "b"})):
        got = exposed_reads(
            statements, defined=defined, defines="writes", loops="runs_once"
        )
        assert sorted(got) == expected[f"sink|{sorted(defined)}"]


@pytest.mark.parametrize("case", sorted(STATEMENT_CASES))
def test_exposed_reads_pins_interchanged_store_dce(case: str) -> None:
    statements = ast.parse(STATEMENT_CASES[case]).body
    got = exposed_reads(
        statements, defines=_single_assigned, loops="opaque", branches="opaque"
    )
    assert sorted(got) == LEGACY_EXPOSED_TABLE[case]["isd"]


@pytest.mark.parametrize("case", sorted(STATEMENT_CASES))
def test_live_in_pins_resident_sequence(case: str) -> None:
    statements = ast.parse(STATEMENT_CASES[case]).body
    expected = LEGACY_EXPOSED_TABLE[case]
    for live_out in (set(), {"c", "out", "b"}):
        got = live_in(statements, live_out)
        assert sorted(got) == expected[f"rs|{sorted(live_out)}"]


def test_same_expr_and_parse_helpers() -> None:
    left = parse_expr("a.b(c + 1)")
    right = ast.parse("a.b(c + 1)", mode="eval").body
    assert same_expr(left, right)
    assert not same_expr(left, parse_expr("a.b(c + 2)"))
    clone = reparse_expr(left)
    assert clone is not left and same_expr(clone, left)
    assert ast.unparse(clone) == "a.b(c + 1)"


def test_replace_node_matches_by_identity_only() -> None:
    tree = ast.parse("y = f(x) + f(x)").body[0]
    assert isinstance(tree, ast.Assign)
    binop = tree.value
    assert isinstance(binop, ast.BinOp)
    replace_node(tree, binop.left, ast.Name(id="z", ctx=ast.Load()))
    assert ast.unparse(tree) == "y = z + f(x)"


def test_replace_call_structural_text_and_first_only() -> None:
    def source() -> ast.stmt:
        return ast.parse("y = g(f(x)) + f(x) + f( x )").body[0]

    call = ast.parse("f(x)", mode="eval").body
    assert isinstance(call, ast.Call)
    tree, count = replace_call(source(), call, lambda: parse_expr("r"))
    assert (ast.unparse(tree), count) == ("y = g(r) + r + r", 3)
    tree, count = replace_call(
        source(), call, lambda: parse_expr("r"), match="text", first_only=True
    )
    assert (ast.unparse(tree), count) == ("y = g(r) + f(x) + f(x)", 1)
    fresh: list[ast.expr] = []

    def make() -> ast.expr:
        fresh.append(parse_expr("r"))
        return fresh[-1]

    tree, count = replace_call(source(), call, make)
    assert count == 3 and len({id(node) for node in fresh}) == 3


def test_substitute_names_contexts_and_clone_callable() -> None:
    cloned: list[ast.expr] = []

    def clone(node: ast.expr) -> ast.expr:
        cloned.append(node)
        return reparse_expr(node)

    mapping = {"a": parse_expr("p + 1"), "b": parse_expr("q")}
    tree = substitute_names(ast.parse("a = a + b"), mapping, clone=clone)
    assert ast.unparse(tree) == "a = p + 1 + q"
    assert len(cloned) == 2
    tree = substitute_names(
        ast.parse("a = a + b"), mapping, clone=clone, loads_only=False
    )
    assert ast.unparse(tree) == "p + 1 = p + 1 + q"
    tree = substitute_names(
        ast.parse("b = 1"), mapping, clone=clone, loads_only=False, keep_ctx=True
    )
    target = tree.body[0].targets[0]  # type: ignore[attr-defined]
    assert isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)


def test_rename_names_is_in_place() -> None:
    tree = ast.parse("a = a + b")
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    rename_names(tree, {"a": "x", "b": "y"}, loads_only=True)
    assert ast.unparse(tree) == "a = x + y"
    rename_names(tree, {"a": "x"})
    assert ast.unparse(tree) == "x = x + y"
    assert [node for node in ast.walk(tree) if isinstance(node, ast.Name)] == names


def test_expand_lane_rewrites_loads_only() -> None:
    tree = ast.parse("lane = ptr[lane] + lane")
    expand_lane(tree, "lane", 4, 3)
    assert ast.unparse(tree) == "lane = ptr[lane * 4 + 3] + (lane * 4 + 3)"
    assert isinstance(expr_from_string("lane * 4 + 3"), ast.expr)
