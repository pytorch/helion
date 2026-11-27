"""Generated code of cross-rank in-band transports: gated stores, stores through
another view, Lamport words and tagged credit.

The kernels bind on CPU as rank 0 of a fake process group (``fake_group``,
``fake_world``), tracing on ``cuda:0`` (``cuda_metadata_trace``): kernels with
cross-loop dependencies bind only for a CUDA device.  The transports run on
several GPUs in ``test/test_distributed.py``.
"""

from __future__ import annotations

import contextlib
import io
import re
import types
import unittest

import pytest
import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._compat import gluon_api_available
from helion._compiler import tile_dependency
from helion._testing import TestCase
from helion._testing import skipIfRefEagerMark
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cuda_metadata_trace
from helion._testing.distributed_kernels import fake_group as fake_group
from helion._testing.distributed_kernels import fake_world
from helion._testing.distributed_kernels import inband_gated_kernel
from helion._testing.distributed_kernels import pipelined_allreduce_kernel
from helion._testing.distributed_kernels import remote_views
from helion._testing.triton_codegen import gluon_cpu_codegen
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    # The static/dynamic cross_loop_pipelines' peer transports are Triton-only.
    skipUnlessBackends(["triton"]),
    skipIfRefEagerMark("tile dependencies are built only in compiled mode"),
]


@helion.kernel(autotune_effort="none", static_shapes=True)
def inband_two_gates_kernel(
    symm: torch.Tensor,
    credit: torch.Tensor,
    x: torch.Tensor,
    gates: list[torch.Tensor],
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """``inband_gated_kernel`` whose peer loads take another gate of the same
    list than the store."""
    peers = remote_views(symm, group_name)
    credit_peers = remote_views(credit, group_name)
    out = torch.zeros_like(symm)
    for tile in hl.tile(symm.size(0)):
        hl.store(symm, [tile], x[tile], extra_mask=gates[0][hl.arange(1)] != 0)
        credit[tile] = x[tile]
    for tile in hl.tile(symm.size(0)):
        keep = (tile.index < symm.size(0)) & (gates[1][hl.arange(1)] != 0)
        acc = hl.zeros([tile], dtype=x.dtype)
        for r in hl.static_range(len(peers)):
            acc = acc + hl.load(peers[r], [tile], extra_mask=keep)
            acc = acc + credit_peers[r][tile]
        hl.store(out, [tile], acc, extra_mask=keep)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def inband_view_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    view: hl.constexpr,
) -> torch.Tensor:
    """The store fills the buffer through another view than the peer loads."""
    peers = remote_views(symm, group_name)
    out = torch.empty_like(symm)
    # The store fills the buffer through a [16, 16] view, in either order.
    symm2 = symm.view(16, -1).t() if view == "cols" else symm.view(-1, 16)
    x2 = x.view(16, -1).t() if view == "cols" else x.view(-1, 16)
    for rows, cols in hl.tile(symm2.size()):
        symm2[rows, cols] = x2[rows, cols]
    for tile in hl.tile(symm.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for r in hl.static_range(len(peers)):
            acc = acc + peers[r][tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def inband_partial_polls_kernel(
    symm: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    rows: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    peers = remote_views(symm, group_name)
    b_peers = remote_views(b, group_name)
    c_peers = remote_views(c, group_name)
    out = torch.empty_like(symm)
    for tile in hl.tile(symm.size(0)):
        symm[tile] = x[tile]
        b[tile] = x[tile]
        c[tile] = x[tile]
    # Every rank polls symm in full, b only from rank 0, c at data-dependent rows.
    for tile in hl.tile(symm.size(0)):
        acc = b_peers[0][tile]
        for r, peer in enumerate(peers):
            acc = acc + peer[tile] + c_peers[r][rows[tile]]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def keyed_then_inband_kernel(
    symm: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
    rows: torch.Tensor,
    group_name: hl.ProcessGroupName,
    polls: hl.constexpr,
) -> torch.Tensor:
    """Root 1 reads root 0's keyed symm; root 2, after a barrier on root 1,
    stores the inband m, which root 3 polls as ``polls`` says."""
    peers = remote_views(symm, group_name)
    m_peers = remote_views(m, group_name)
    flat = m.view(-1)
    total = torch.empty_like(x)
    out = torch.zeros_like(m)
    for tile in hl.tile(x.size(0)):
        symm[tile] = x[tile]
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for peer in peers:
            acc = acc + peer[tile]
        total[tile] = acc
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype) + total[:].sum()
        if polls == "keyed":  # root 2 reads keyed peer data too
            acc = acc + peers[0][tile]
        flat[tile] = acc.to(m.dtype)
    for tile in hl.tile(m.size(0)):
        for peer in m_peers:
            if polls == "diagonal":
                out[tile, tile] += peer[tile, tile]
            elif polls == "partial":
                out[tile, :] += peer[rows[tile], :]
            else:
                out[tile, :] += peer[tile, :]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def peer_sum_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    publish: hl.constexpr,
) -> torch.Tensor:
    """Every rank's ``symm`` summed: stored by the first root with
    ``publish``, else by whatever ran before the launch."""
    peers = remote_views(symm, group_name)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        if publish:
            symm[tile] = x[tile]
        else:
            out[tile] = x[tile]
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for peer in peers:
            acc = acc + peer[tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def unit_row_polls_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    begin: hl.constexpr,
) -> torch.Tensor:
    """The ranks' rows summed per 128 columns, the row indexed by its block-1
    tile or, with ``begin``, by the tile's start."""
    peers = remote_views(symm, group_name)
    out = torch.empty_like(x)
    for rows, cols in hl.tile(x.size(), block_size=[1, 512]):
        symm[rows, cols] = x[rows, cols]
    for rows, cols in hl.tile(x.size(), block_size=[1, 128]):
        if begin:
            acc = hl.zeros([cols], dtype=torch.float32)
            for peer in peers:
                acc = acc + peer[rows.begin, cols].to(torch.float32)
            out[rows.begin, cols] = acc.to(x.dtype)
        else:
            acc2 = hl.zeros([rows, cols], dtype=torch.float32)
            for peer in peers:
                acc2 = acc2 + peer[rows, cols].to(torch.float32)
            out[rows, cols] = acc2.to(x.dtype)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True, backend="gluon")
def tagged_credit_in_push_kernel(
    z: torch.Tensor,
    symm: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``tagged_ring_exchange_kernel`` with the credit stored in the push's root,
    which the push's keyed counters end."""
    z_peers = remote_views(z, group_name)
    peers = remote_views(symm, group_name)
    out = torch.empty([z.size(0), w.size(1)], dtype=torch.float32, device=z.device)
    total = torch.empty_like(s)
    k = x.size(1)
    for tile_m in hl.tile(x.size(0)):
        rows = x[tile_m, :]
        for z_peer in z_peers:
            z_peer[tile_m, rank * k + hl.arange(k)] = rows
        symm[tile_m] = s[tile_m]
    for tile in hl.tile(s.size(0)):
        acc = hl.zeros([tile], dtype=s.dtype)
        for peer in peers:
            acc = acc + peer[tile]
        total[tile] = acc
    for tile_m, tile_n in hl.tile([z.size(0), w.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(z.size(1)):
            acc = hl.dot(z[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out, total


@skipIfTileIR("TileIR does not support barrier operations")
@pytest.mark.usefixtures("fake_group")
@instantiate_parametrized_tests
class TestInbandCodegen(TestCase):
    @parametrize("world", (2, 4, 8))
    def test_inband_gated_store(self, world: int) -> None:
        """A store masked by a launch-uniform gate travels inband when every peer
        load carries the gate (its push is predicated on it); else peer counters."""
        group = fake_world(world)
        inband = "use inband (epoch words: a gated store)"
        for variant, dtype, expected in (
            ("", torch.float32, inband),
            # Two bf16 lanes a word: the masked polls take the load's mask.
            ("", torch.bfloat16, inband),
            # A gate built with arithmetic (any pointwise op).
            ("sum", torch.float32, inband),
            ("R9", torch.float32, "use peer_counter (R9:"),
            ("tile-mask", torch.float32, "use peer_counter (R1:"),
            ("written", torch.float32, "use peer_counter (R1:"),
        ):
            symm, x = torch.zeros(2, 256, dtype=dtype)
            credit = torch.zeros(256, dtype=dtype)
            gate = torch.ones(1, dtype=torch.int32)
            with (
                self.subTest(variant=variant, dtype=dtype),
                cuda_metadata_trace(),
            ):
                with self.assertLogs(tile_dependency.log, "INFO") as logs:
                    bound = cpu_bind(
                        inband_gated_kernel, (symm, credit, x, gate, group, variant)
                    )
                (line,) = (line for line in logs.output if "peers/symm " in line)
                self.assertIn(f"on peers/symm {expected}", line)
                if variant in ("", "sum"):
                    code = bound.to_triton_code()
                    # The gated pushes (one asm per rank) are predicated on the
                    # gate, the credit's are not.
                    pushes = code.count("st.relaxed.sys.global.")
                    self.assertEqual(code.count("@p st.relaxed.sys.global."), world)
                    self.assertEqual(pushes, 2 * world)

    def test_inband_gates_differ_by_their_whole_path(self) -> None:
        """Gates are keyed by their host path: a store gated on ``gates[0]``
        whose peer loads take ``gates[1]`` keeps peer counters (R9)."""
        group = fake_world(2)
        symm, x = torch.zeros(2, 256)
        credit = torch.zeros(256)
        gates = [torch.ones(1, dtype=torch.int32) for _ in range(2)]
        with (
            cuda_metadata_trace(),
            self.assertLogs(tile_dependency.log, "INFO") as logs,
        ):
            cpu_bind(inband_two_gates_kernel, (symm, credit, x, gates, group))
        (line,) = (line for line in logs.output if "peers/symm " in line)
        self.assertIn("on peers/symm use peer_counter (R9:", line)

    def test_rank_digest_ignores_other_masks(self) -> None:
        """Ranks agree on the gates of inband pushes, not on the other terms of
        peer loads' or local stores' masks (here built from a rank constant)."""
        group = fake_world(2)
        digests = []
        for variant in ("rank0", "rank1"):
            symm, x = torch.zeros(2, 256)
            credit = torch.zeros(256)
            gate = torch.ones(1, dtype=torch.int32)
            with cuda_metadata_trace():
                code = cpu_bind(
                    inband_gated_kernel, (symm, credit, x, gate, group, variant)
                ).to_triton_code()
            self.assertIn("@p st.relaxed.sys.global.", code)
            digest = re.search(r"_persistent_state_rank_digest='([0-9a-f]{16})'", code)
            assert digest is not None
            digests.append(digest[1])
        self.assertEqual(*digests)

    def test_inband_store_through_another_view(self) -> None:
        """Pushes and polls address words by storage offset, so the store's
        view may differ from the peer loads' in shape and order (R2)."""
        group = fake_world(2)
        # bf16 rows pack two adjacent elements a word: word pairs pushed at
        # (row * 16 + col) // 2, polled at e // 2 and unpacked lane by lane.
        # Column neighbors lie 16 elements apart: one element a word.
        expected = {
            "rows": (
                2,
                "v2.u64 [$1], {$2, $3};",
                (
                    r"tl.broadcast_to\(indices_0\[:, None\] \* 16 "
                    r"\+ inband_lane_\d+\[None, :\] \* 1, \[.*\]\) // 2"
                ),
                r"inband_lane_\d+ \* 1 // 2",
            ),
            "cols": (
                1,
                "u64 [$1], $2;",
                r"indices_0\[:, None\] \* 1 \+ indices_1\[None, :\] \* 16",
                r"indices_2 \* 1",
            ),
        }
        for view, (lanes, push, push_word, poll_word) in expected.items():
            symm, x = torch.zeros(2, 256, dtype=torch.bfloat16)
            with self.subTest(view=view), cuda_metadata_trace():
                with self.assertLogs(tile_dependency.log, "INFO") as logs:
                    bound = cpu_bind(inband_view_kernel, (symm, x, group, view))
                code = bound.to_triton_code()
            (line,) = logs.output
            self.assertIn("on peers/symm2 use inband", line)
            # Each rank's slot holds the 256 elements in words of `lanes`; one
            # push a rank, at the word offset of the store's view.
            self.assertIn(f"tile_dependency_peer_rank * {256 // lanes})", code)
            self.assertEqual(code.count(f"st.relaxed.sys.global.{push}"), 2)
            word = re.search(r"tile_dependency_peer_base_0 \+ \(.*?\) \+ (\w+),", code)
            assert word is not None
            self.assertRegex(code, rf"\n    {word[1]} = {push_word}\n")
            self.assertRegex(
                code,
                r"inband_address_0 = tile_dependency_peer_state \+ \(.*\) "
                rf"\+ {poll_word}\n",
            )
            # The poll unpacks every lane of the words it reads.
            self.assertEqual("tl.cast(inband_word >> 16, tl.uint16)" in code, lanes > 1)

    @parametrize(
        "dtype", (torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn)
    )
    def test_inband_lamport_words(self, dtype: torch.dtype) -> None:
        world, n = 4, 4096
        symm, x = torch.zeros(2, n, dtype=dtype)
        group = fake_world(world)
        with cuda_metadata_trace():
            bound = cpu_bind(
                pipelined_allreduce_kernel, (symm, x, group, "inband", world)
            )
            config = bound.config_spec.default_config().config
            # One candidate, off by default; a config saved before the key
            # existed keeps the epoch words too.
            self.assertEqual(config["cross_loop_inband_protocol"], ["tagged"])
            old = {k: v for k, v in config.items() if k != "cross_loop_inband_protocol"}
            tagged = bound.to_triton_code(helion.Config(**old))
            # Rows of 32 elements fill 16-byte word pairs at every element size
            # (a device's default may tile narrower: 8 float8 elements fill one
            # word).
            lamport = {**config, "block_sizes": [32, 32]}
            code = bound.to_triton_code(
                helion.Config(**{**lamport, "cross_loop_inband_protocol": ["lamport"]})
            )
        self.assertIn("tile_dependency_peer_epoch << 32", tagged)
        self.assertNotIn("fence.acq_rel.gpu", tagged)
        lanes, bits = 8 // dtype.itemsize, 8 * dtype.itemsize
        ones = sum(1 << (bits * i) for i in range(lanes))
        # A word packs 8 bytes of elements and no epoch: three launch slots of
        # half the tagged words each.
        self.assertIn(f"(x, {3 * world * n // lanes + 1}, torch.uint64, True)", code)
        self.assertNotIn("tile_dependency_peer_epoch << 32", code)
        # Pushes are the tagged words' 16-byte stores, one asm per rank, into
        # this launch's slot; no zeroing, fence or atomics on the push side.
        self.assertEqual(code.count("st.relaxed.sys.global.v2.u64"), world)
        self.assertIn("tile_dependency_peer_epoch % 3", code)
        self.assertNotIn("tl.atomic_", code)
        self.assertNotIn("fence.acq_rel", code)
        # The poll re-arms what it read in the previous launch's slot.
        self.assertIn("tl.where(tile_dependency_peer_epoch % 3 == 0, 2, -1)", code)
        self.assertEqual(code.count("& 0)"), world)  # one poll a rank
        # Lanes encode x ^ key (the key's own NaN as key ^ sign) and decode by
        # xor with the key; polls wait while a lane is zero.
        key = tile_dependency.LAMPORT_KEYS[dtype]
        k = f"tl.full([], {key}, tl.uint{bits})"
        spare = f"tl.full([], {key ^ 1 << bits - 1}, tl.uint{bits})"
        self.assertIn(f" == {k}, {spare}, ", code)
        self.assertIn(f"tl.uint{bits}) ^ {k}, ", code)
        self.assertIn(f"sub.u64 t, $0, 0x{ones:016x};", code)
        # (The unparsed code drops the redundant parentheses around the mask.)
        self.assertIn(f">> {bits - 1} & {ones} != 0", code)
        self.assertNotIn("setp.ne.b32 p, hi", code)

    @parametrize("dtype", (torch.bfloat16, torch.float32))
    def test_inband_lamport_masked_polls(self, dtype: torch.dtype) -> None:
        """A tile past the end masks its polls: a masked lane reads every lane
        as the key (a Lamport word just written with zeros), so it neither
        waits nor decodes to garbage."""
        world, n = 4, 4000
        symm, x = torch.zeros(2, n, dtype=dtype)
        group = fake_world(world)
        with cuda_metadata_trace():
            bound = cpu_bind(
                pipelined_allreduce_kernel, (symm, x, group, "inband", world)
            )
            config = bound.config_spec.default_config().config
            # Stores in 16-element tiles, which fill whole words; polls in
            # 1024-element tiles, the fourth of which holds the last 928 elements.
            code = bound.to_triton_code(
                helion.Config(
                    **{
                        **config,
                        "block_sizes": [16, 1024],
                        "cross_loop_inband_protocol": ["lamport"],
                    }
                )
            )
        self.assertNotIn("tile_dependency_peer_epoch << 32", code)
        bits = 8 * dtype.itemsize
        words = sum(
            tile_dependency.LAMPORT_KEYS[dtype] << (bits * i) for i in range(64 // bits)
        )
        zeros = words - (1 << 64) if words >> 63 else words
        self.assertIn(f", {zeros}, volatile=True)", code)

    def test_inband_lamport_needs_full_polls(self) -> None:
        world, n = 4, 4096
        # Separate allocations: a peer view starts at its symmetric allocation.
        symm, b, c, x = (torch.zeros(n) for _ in range(4))
        rows = torch.arange(n).flip(0)
        group = fake_world(world)
        with cuda_metadata_trace():
            with self.assertLogs(tile_dependency.log, "INFO") as logs:
                bound = cpu_bind(
                    inband_partial_polls_kernel, (symm, b, c, rows, x, group)
                )
            config = bound.config_spec.default_config().config
            code = bound.to_triton_code(
                helion.Config(**{**config, "cross_loop_inband_protocol": ["lamport"]})
            )
        # Only rank 0's b is read, and c at rows only known at run time, so a
        # word a rank never reads would never be re-armed: only symm is a
        # candidate, and b and c keep their epochs.
        lines = sorted(logs.output)
        self.assertEqual(len(lines), 3)
        never = "use inband (epoch words: a rank may not poll every word)"
        self.assertIn(f"on b/b_peers {never}", lines[0])
        self.assertIn(f"on c/c_peers {never}", lines[1])
        self.assertIn("on peers/symm use inband (Lamport candidate)", lines[2])
        # symm travels as Lamport words (2 fp32 a word, three slots).
        words = 3 * world * n // 2 + 2 * (2 * world * n) + 1
        self.assertIn(f"(x, {words}, torch.uint64, True)", code)
        self.assertIn("tile_dependency_peer_epoch << 32", code)
        self.assertIn("tile_dependency_peer_epoch % 3", code)

    def test_unit_tile_on_a_size_one_dim_polls_like_a_scalar_index(self) -> None:
        """A block-1 tile on a size-1 dim broadcasts nothing: its polls take
        whole words, one asm for every rank and one CTA vote, as the tile's
        start does (not a spin per element)."""
        group = fake_world(4)
        symm, x = torch.zeros(2, 1, 4096, dtype=torch.bfloat16)
        for begin in (False, True):
            with self.subTest(begin=begin), cuda_metadata_trace():
                bound = cpu_bind(unit_row_polls_kernel, (symm, x, group, begin))
                config = bound.config_spec.default_config().config
                code = bound.to_triton_code(
                    helion.Config(
                        **{**config, "cross_loop_inband_protocol": ["lamport"]}
                    )
                )
            self.assertEqual(code.count("bar.red.or.pred"), 1)
            self.assertEqual(code.count("SPIN${:uid}:"), 1)
            self.assertNotIn(">> tl.cast(inband_offset", code)

    @unittest.skipUnless(gluon_api_available(), "the gluon ring needs Triton >= 3.8")
    def test_tagged_credit_in_a_keyed_root(self) -> None:
        """The credit's polls give the mirrors their parity credit (T5) even
        when the credit is stored in a root keyed counters end."""
        world, m, k, n = 4, 128, 64, 128
        group = fake_world(world)
        z = torch.zeros(m, world * k, dtype=torch.bfloat16)
        symm, s = torch.zeros(2, m)
        x = torch.ones(m, k, dtype=torch.bfloat16)
        w = torch.ones(world * k, n, dtype=torch.bfloat16)
        with (
            gluon_cpu_codegen(),
            self.assertLogs(tile_dependency.log, "INFO") as logs,
        ):
            cpu_bind(tagged_credit_in_push_kernel, (z, symm, x, w, s, group, 0))
        (line,) = [line for line in logs.output if "on z/z_peers" in line]
        self.assertTrue(line.endswith("; tagged candidate"), line)

    def test_done_barrier_needs_full_polls(self) -> None:
        """Every rank's launch ends root 1, through the barrier into root 2, so
        no root needs the done barrier, but only if root 3 surely polls every
        word of m and root 2 reads no keyed peer data (which may arrive after
        root 2's pushes)."""
        group = fake_world(2)
        symm, x = torch.zeros(2, 256, dtype=torch.float64)  # R2 keys symm
        m = torch.zeros(16, 16)
        rows = torch.arange(16).flip(0)
        for polls in ("full", "partial", "diagonal", "keyed"):
            with self.subTest(polls=polls), cuda_metadata_trace():
                with self.assertLogs(tile_dependency.log, "INFO") as logs:
                    bound = cpu_bind(
                        keyed_then_inband_kernel, (symm, m, x, rows, group, polls)
                    )
                code = bound.to_triton_code()
                self.assertIn("on flat/m_peers use inband", logs.output[-1])
                self.assertEqual("_wait_at_least" in code, polls != "full")

    def test_remote_read_without_producer_warns(self) -> None:
        """A peer load of an allocation no root of the kernel stores has no
        cross-rank order: it warns once per root, naming the root and the load,
        unless an earlier root of the kernel publishes the data."""
        group = fake_world(4, peer_sum_kernel)
        symm, x = torch.zeros(2, 256)
        for publish in (True, False):
            with (
                self.subTest(publish=publish),
                cuda_metadata_trace(),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                cpu_bind(peer_sum_kernel, (symm, x, group, publish))
            warnings = stderr.getvalue().count("WARNING[RemoteReadWithoutProducer]")
            self.assertEqual(warnings, 0 if publish else 1)
        self.assertRegex(
            stderr.getvalue(),
            r"root 1 \[lines:\d+-\d+\] reads another rank's copy of an allocation "
            r"that no root of this kernel stores \(peers loaded at line \d+: "
            r"acc = acc \+ peer\[tile\]\)",
        )

    def test_process_group_name_class_annotation(self) -> None:
        """An ``hl.ProcessGroupName`` annotation that is the class (a kernel
        without ``from __future__ import annotations``) declares the group as
        the string form does: the kernel is distributed and keeps its
        cross-rank edge."""
        group = fake_world(4)
        fn = peer_sum_kernel.fn
        plain = types.FunctionType(
            fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
        )
        plain.__annotations__ = {
            **fn.__annotations__,
            "group_name": hl.ProcessGroupName,
        }
        kernel = helion.kernel(plain, autotune_effort="none", static_shapes=True)
        symm, x = torch.zeros(2, 256)
        with cuda_metadata_trace():
            bound = cpu_bind(kernel, (symm, x, group, True))
        self.assertEqual(bound.env.process_group_name, group)
        graph = bound.host_function.device_ir.tile_dependency_graph
        self.assertTrue(graph is not None and graph.crosses_ranks())
