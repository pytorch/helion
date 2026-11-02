"""CPU execution and refusal proofs for affine rank-relative packet prefetch."""

from __future__ import annotations

import ast

import pytest

from test.test_cute_consumer_packet_prefetch import admitted
from test.test_cute_consumer_packet_prefetch import dumped
from test.test_cute_consumer_packet_prefetch import execute
from test.test_cute_consumer_packet_prefetch import inverse_motion
from test.test_cute_consumer_packet_prefetch import run_pass
from test.test_cute_consumer_packet_prefetch import source


def affine_source(
    *,
    base: str = "cutlass.Int32(cute.arch.block_idx()[1]) * 64",
    lanes: int = 1,
    width: int = 4,
    carrier: str = "Uint16",
    cluster: bool = True,
) -> str:
    text = source(lanes=lanes, width=width, carrier=carrier)
    text = "rank_base = " + base + "\n" + text
    text = text.replace(
        "cutlass.Int32(0), cutlass.Int32(64), cutlass.Int32(64)",
        "cutlass.Int32(rank_base), cutlass.Int32(rank_base + 64), cutlass.Int32(64)",
    )
    # Match the emitted global-bound pointer redirection, including an entirely
    # inactive CTA; the scalar mask and the load's cache policy remain intact.
    text = text.replace(
        "x.iterator + before,",
        "x.iterator + before if before < extent else x.iterator,",
    )
    if cluster:
        text = text.replace(
            "total = cute.arch.warp_reduction_sum(total)",
            "total = _cute_grouped_reduce_cluster(total, 'sum', cutlass.Float32(0), "
            "cute.arch.thread_idx()[0], cluster_buf, cluster_mbar, group_span=32, cluster_n=2)",
        )
    return text


def interpreted(body: list[ast.stmt], rank: int) -> list[ast.stmt]:
    """Give the existing CPU interpreter one CTA coordinate/collective value."""
    copied = ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[])))

    class Bind(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.arch.block_idx"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 1
            ):
                return ast.Constant(value=rank)
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_cute_grouped_reduce_cluster"
            ):
                return ast.parse(
                    "cute.arch.warp_reduction_sum(" + ast.unparse(node.args[0]) + ")",
                    mode="eval",
                ).body
            return self.generic_visit(node)

    return Bind().visit(copied).body


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("extent", [16, 48, 96])
@pytest.mark.parametrize(
    "lanes,width,carrier", [(1, 4, "Uint16"), (2, 8, "Uint16"), (2, 4, "Uint32")]
)
def test_affine_raw_cache_preserves_masks_addresses_bits_and_order(
    rank: int,
    extent: int,
    lanes: int,
    width: int,
    carrier: str,
) -> None:
    text = affine_source(lanes=lanes, width=width, carrier=carrier)
    before = ast.parse(text).body
    after = run_pass(text)
    assert admitted(after)
    for thread in range(8):
        old, old_events = execute(
            interpreted(before, rank), thread=thread, extent=extent
        )
        new, new_events = execute(
            interpreted(after, rank), thread=thread, extent=extent
        )
        assert old == new
        assert sorted(event for event in old_events if event[0] == "load") == sorted(
            event for event in new_events if event[0] == "load"
        )
        assert old_events[0] == new_events[0] == ("wait",)
        barrier = new_events.index(("barrier",))
        assert all(
            index < barrier
            for index, event in enumerate(new_events)
            if event[:2] == ("load", "weight")
        )
        assert all(
            index > barrier
            for index, event in enumerate(new_events)
            if event[0] == "store"
        )
    assert dumped(inverse_motion(before, after)) == dumped(before)


@pytest.mark.parametrize(
    "base",
    [
        "cutlass.Int32(cute.arch.block_idx()[1]) * 64 + 16",
        "(cutlass.Int64(cute.arch.block_idx()[1]) * 2 + 3) * 16",
        "cutlass.Int32(cute.arch.block_idx()[1]) // 2 * 64",
    ],
)
def test_pure_rank_expressions_have_constant_relative_span(base: str) -> None:
    assert admitted(run_pass(affine_source(base=base)))


def test_prefix_value_snapshots_survive_later_name_rebinding() -> None:
    text = affine_source().replace(
        "rank_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64",
        "original_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64\n"
        "rank_base = original_base\noriginal_base = original_base + 16",
    )
    assert admitted(run_pass(text))


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "rank_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64",
            "rank_base = arbitrary()",
        ),
        (
            "rank_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64",
            "rank_base = runtime.load()",
        ),
        (
            "rank_base = cutlass.Int32(cute.arch.block_idx()[1]) * 64",
            "rank_base = cutlass.Int32(cute.arch.block_idx()[1]) * 65536",
        ),
        ("rank_base + 64", "rank_base + extent"),
        ("rank_base + 64", "rank_base + 32"),
        ("rank_base + 64", "rank_base + 128"),
        ("cutlass.Int32(64)", "cutlass.Int32(extent)"),
        (
            "cute.arch.sync_threads()",
            "rank_base = rank_base + 64\ncute.arch.sync_threads()",
        ),
        (
            "cute.arch.sync_threads()",
            "if extent:\n    rank_base = rank_base + 64\ncute.arch.sync_threads()",
        ),
        (
            "cute.arch.sync_threads()",
            "(weight.iterator + 1).store(total)\ncute.arch.sync_threads()",
        ),
        (
            "cute.arch.sync_threads()",
            "cute.arch.griddepcontrol_wait()\ncute.arch.sync_threads()",
        ),
        ("cute.arch.sync_threads()", "opaque(total)\ncute.arch.sync_threads()"),
        ("valid = after < extent", "valid = after < total"),
        ("weight.iterator + after", "out.iterator + after"),
        (
            "cute.arch.griddepcontrol_wait()",
            "cute.arch.griddepcontrol_wait()\n_cute_grouped_reduce_cluster = arbitrary",
        ),
    ],
)
def test_unproved_affine_motion_keeps_original_ast(old: str, new: str) -> None:
    text = affine_source().replace(old, new)
    assert text != affine_source()
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_nonmatching_consumer_domain_is_not_cached() -> None:
    text = affine_source()
    position = text.rfind("rank_base + 64")
    text = text[:position] + text[position:].replace(
        "rank_base + 64", "rank_base + 128", 1
    )
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_affine_cache_keeps_alias_and_total_register_budget_guards() -> None:
    text = affine_source()
    assert not admitted(run_pass(text, disjoint=False))
    assert not admitted(
        run_pass("extra = cute.make_rmem_tensor(125, cutlass.Uint16)\n" + text)
    )
    assert admitted(
        run_pass("extra = cute.make_rmem_tensor(120, cutlass.Uint16)\n" + text)
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "(weight.iterator + 0).store(cutlass.Float32(2))",
        "if cute.arch.thread_idx()[0] == 1:\n    (weight.iterator + 0).store(cutlass.Float32(2))",
        "alias = weight\nif cute.arch.thread_idx()[0] == 1:\n    alias[0] = cutlass.Float32(2)",
        "alias = weight.iterator\nif cute.arch.thread_idx()[0] == 1:\n    alias.store(cutlass.Float32(2))",
        "opaque(weight)",
    ],
)
@pytest.mark.parametrize("affine", [False, True])
def test_whole_body_prefix_writes_cannot_cross_collective(
    prefix: str,
    affine: bool,
) -> None:
    text = affine_source() if affine else source()
    text = prefix + "\n" + text
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_whole_body_suffix_weight_write_keeps_read_only_policy() -> None:
    text = affine_source() + "\n(weight.iterator + 0).store(cutlass.Float32(2))\n"
    assert dumped(run_pass(text)) == dumped(ast.parse(text).body)


def test_whole_body_disjoint_output_write_is_not_a_weight_barrier() -> None:
    text = "(out.iterator + 0).store(cutlass.Float32(0))\n" + affine_source()
    assert admitted(run_pass(text))
