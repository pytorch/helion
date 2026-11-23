from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
import unittest
from unittest import mock
import weakref

import pytest
import torch

import helion
from helion import _compat
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import prefetch_kernels
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing.standalone import run_add_without_helion
import helion.language as hl
from helion.runtime.triton import tensor_maps

if TYPE_CHECKING:
    from collections.abc import Iterator

_BULK = "cp.async.bulk.prefetch.L2.global"
_PER_ELEMENT = "prefetch.global.L2 [$1]"
_BOX = "cp.async.bulk.prefetch.tensor"


def _prefetch_lines(code: str) -> list[str]:
    return [line for line in code.splitlines() if "tl.inline_asm_elementwise" in line]


@helion.kernel(autotune_effort="none")
def add_prefetched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        hl.prefetch(y, [tile_m, tile_n])
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def add_plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(autotune_effort="none")
def row_sum_prefetched(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], device=x.device, dtype=torch.float32)
    for tile_m in hl.tile(x.size(0)):
        hl.prefetch(x, [tile_m, slice(None)])
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    return out


@helion.kernel(autotune_effort="none")
def scale_prefetched(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.prefetch(x, [tile])
        out[tile] = x[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def gather_prefetched(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = torch.empty([rows.size(0), x.size(1)], device=x.device, dtype=x.dtype)
    for tile_r, tile_n in hl.tile([rows.size(0), x.size(1)]):
        row = rows[tile_r]
        hl.prefetch(x, [row[:, None], tile_n.index[None, :]])
        hl.prefetch(x, [row, tile_n], extra_mask=row[:, None] > 0)
        out[tile_r, tile_n] = x[row, tile_n]
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def independent_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0), block_size=16):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0), block_size=16):
        hl.prefetch(tmp, [tile])
        out[tile] = y[tile] * 2
    return tmp, out


hint_read_roots = helion.kernel(autotune_effort="none", static_shapes=True)(
    prefetch_kernels.hint_read_roots
)


@helion.kernel(autotune_effort="none", static_shapes=True)
def dependent_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.empty_like(x)
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0), block_size=16):
        tmp[tile] = x[tile] + 1
    for tile in hl.tile(y.size(0), block_size=16):
        hl.prefetch(tmp, [tile])
        out[tile] = y[tile] * 2 + tmp[tile]
    return tmp, out


@helion.kernel(autotune_effort="none", static_shapes=True)
def box_prefetch_copy(w: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    """Prefetch ``wp`` (a view of ``w``) one leading index at a time, then copy
    ``w``."""
    out = torch.empty_like(w)
    for tile_p in hl.tile(wp.size(0), block_size=1):
        hl.prefetch(wp, [tile_p.begin, slice(None), slice(None)])
    for tile_k, tile_n in hl.tile(w.size()):
        out[tile_k, tile_n] = w[tile_k, tile_n]
    return out


@contextlib.contextmanager
def _empty_prefetch_maps() -> Iterator[None]:
    """An empty tensor-map cache (``tensor_maps.prefetch_map``) for the block."""
    with (
        mock.patch.dict(tensor_maps._PREFETCH_MAPS, clear=True),
        mock.patch.dict(tensor_maps._UNCAPTURED_PREFETCH_MAPS, clear=True),
        mock.patch.object(tensor_maps, "_CAPTURED_PREFETCH_MAPS", []),
    ):
        yield


@onlyBackends(["triton"])
@skipIfNotCUDA()
@skipIfRocm("prefetch lowers to PTX")
@skipIfTileIR("prefetch lowers to PTX")
class TestPrefetch(RefEagerTestDisabled, TestCase):
    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_tile_bulk_partial_last_tile(self) -> None:
        x = torch.randn(100, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        code, out = code_and_output(add_prefetched, (x, y), block_sizes=[8, 128])
        torch.testing.assert_close(out, x + y)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        # 1000 % 128 != 0: the last tile prefetches only its valid bytes.
        self.assertIn("tl.minimum(_BLOCK_SIZE_1, 1000 - offset_1) * 2", line)
        self.assertIn("tl.where(", line)  # rows past the 100th are skipped

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_full_row_bulk(self) -> None:
        x = torch.randn(64, 1000, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(row_sum_prefetched, (x,), block_sizes=[4])
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        self.assertIn("1000 * 2", line)

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_rolled_reduction_placement(self) -> None:
        x = torch.randn(64, 1000, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(
            row_sum_prefetched, (x,), block_sizes=[4], reduction_loops=[128]
        )
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        # The bulk prefetch covers the whole row once, ahead of the rolled loop ...
        self.assertIn(_BULK, line)
        self.assertLess(code.index(line), code.index("for roffset_1 in"))

        capability = mock.PropertyMock(return_value=(8, 0))
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(
                row_sum_prefetched, (x,), block_sizes=[4], reduction_loops=[128]
            )
        torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
        (line,) = _prefetch_lines(code)
        # ... a per-element one covers each rolled chunk inside it.
        self.assertIn(_PER_ELEMENT, line)
        self.assertGreater(code.index(line), code.index("for roffset_1 in"))
        self.assertIn("rindex_1", line)

    @skipIfCudaCapabilityLessThan((9, 0), reason="cp.async.bulk needs sm90")
    def test_scalar_prefetch_issues_from_one_lane_per_warp(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        code, out = code_and_output(scale_prefetched, (x,), block_sizes=[256])
        torch.testing.assert_close(out, x * 2)
        (line,) = _prefetch_lines(code)
        self.assertIn(_BULK, line)
        # An elected lane, not %tid.x (which costs ptxas its uniformity analysis).
        self.assertIn("elect.sync", line)
        self.assertNotIn("%tid.x", line)

    def test_gather_and_extra_mask_prefetch_per_element(self) -> None:
        x = torch.randn(50, 300, device=DEVICE)
        rows = torch.tensor([3, 7, 49, 0, 5], device=DEVICE)
        code, out = code_and_output(gather_prefetched, (x, rows), block_sizes=[4, 128])
        torch.testing.assert_close(out, x[rows])
        lines = _prefetch_lines(code)
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn(_PER_ELEMENT, line)
            self.assertNotIn(_BULK, line)

    def test_pre_sm90_and_non_ptx_targets(self) -> None:
        x = torch.randn(100, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        capability = mock.PropertyMock(return_value=(8, 0))
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(add_prefetched, (x, y), block_sizes=[8, 128])
        torch.testing.assert_close(out, x + y)
        (line,) = _prefetch_lines(code)
        self.assertIn(_PER_ELEMENT, line)
        self.assertNotIn(_BULK, line)

        capability = mock.PropertyMock(return_value=None)
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            code, out = code_and_output(add_prefetched, (x, y), block_sizes=[16, 64])
        torch.testing.assert_close(out, x + y)
        self.assertEqual(_prefetch_lines(code), [])

        # Before sm90 no prefetch can take the box form, so a prefetch adds
        # no key. Shapes no other test binds (the mocked binding is cached).
        x80 = torch.randn(104, 1000, device=DEVICE, dtype=torch.bfloat16)
        y80 = torch.randn_like(x80)
        capability = mock.PropertyMock(return_value=(8, 0))
        with mock.patch.object(CompileEnvironment, "ptx_capability", capability):
            with_prefetch = add_prefetched.bind((x80, y80)).config_spec
            without = add_plain.bind((x80, y80)).config_spec
            self.assertEqual(
                dict(with_prefetch.default_config()), dict(without.default_config())
            )
            self.assertEqual(with_prefetch.prefetch_indexing.length, 0)

        # A target that drops prefetches (no PTX: HIP, TileIR, XPU) has no
        # prefetch_indexing key.
        spec = add_prefetched.bind((x, y)).config_spec
        self.assertTrue(spec.supports_config_key("prefetch_indexing"))
        with mock.patch.object(_compat, "is_hip", return_value=True):
            self.assertFalse(spec.backend.supports_config_key("prefetch_indexing"))
        with mock.patch.object(spec, "target_device_capability", None):
            self.assertFalse(spec.supports_config_key("prefetch_indexing"))

    @skipIfCudaCapabilityLessThan((9, 0), reason="the box form needs sm90")
    def test_prefetch_indexing_key(self) -> None:
        """A prefetch takes a ``prefetch_indexing`` entry, and no ``indexing``,
        eviction-policy or cache-modifier slot: the loads' and stores' slots
        stay where they were, a kernel without prefetches has no such key, and
        a config written before keeps its meaning (every prefetch in the
        pointer form)."""
        # Shapes no other test binds the kernel with (a binding under a mocked
        # capability would be reused).
        x = torch.randn(112, 1000, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        spec = add_prefetched.bind((x, y)).config_spec
        plain = add_plain.bind((x, y)).config_spec
        self.assertEqual(spec.memory_op_facts, plain.memory_op_facts)
        self.assertEqual(spec.prefetch_indexing.length, 1)
        # One entry per prefetch, also one whose slice a config can roll.
        w = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        rollable = box_prefetch_copy.bind((w, w.view(16, 64, 256))).config_spec
        self.assertEqual(len(rollable.reduction_loops), 1)
        self.assertEqual(rollable.prefetch_indexing.length, 1)
        # No key when no prefetch can take the box form (gathers, a mask).
        rows = torch.tensor([3, 7, 49, 0, 5], device=DEVICE)
        gather = gather_prefetched.bind((torch.randn(50, 300, device=DEVICE), rows))
        self.assertNotIn("prefetch_indexing", gather.config_spec.default_config())
        with_prefetch = dict(spec.default_config())
        without = dict(plain.default_config())
        self.assertEqual(with_prefetch.pop("prefetch_indexing"), ["pointer"])
        self.assertEqual(with_prefetch, without)
        indexing = ["tensor_descriptor", "pointer", "tensor_descriptor"]
        config = {"block_sizes": [16, 64], "indexing": [*indexing]}
        spec.normalize(config)
        self.assertEqual(config["indexing"], indexing)
        self.assertEqual(config["prefetch_indexing"], ["pointer"])

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_tensor_descriptor_prefetch_is_a_tma_box(self) -> None:
        """In the tensor_descriptor form, a prefetch is one TMA box of the tile
        its subscript describes, from a tensor map the host builds; the loads'
        and stores' ``indexing`` does not choose it."""
        # Shapes no other test binds the kernel with (a binding under a mocked
        # capability would be reused).
        x = torch.randn(96, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        code, out = code_and_output(
            add_prefetched,
            (x, y),
            block_sizes=[16, 64],
            prefetch_indexing=["tensor_descriptor"],
        )
        torch.testing.assert_close(out, x + y)
        self.assertEqual(code.count(f"{_BOX}.2d.L2.global.tile"), 1)
        self.assertNotIn(_BULK, code)
        self.assertIn("helion_tensor_maps.prefetch_map(y, [16, 64])", code)
        # The box starts at the tile: offsets innermost first.
        self.assertIn("tl.cast(offset_1, tl.int32), tl.cast(offset_0, tl.int32)]", code)
        # The map is acquired once, at the kernel's entry.
        self.assertEqual(code.count("fence.proxy.tensormap"), 1)
        # The pointer form keeps the bulk prefetch whatever the loads use.
        code, out = code_and_output(
            add_prefetched,
            (x, y),
            block_sizes=[16, 64],
            indexing="tensor_descriptor",
            prefetch_indexing="pointer",
        )
        torch.testing.assert_close(out, x + y)
        self.assertIn(_BULK, code)
        self.assertNotIn(_BOX, code)
        # A misspelled form is an error, as in ``indexing``.
        with self.assertRaisesRegex(Exception, "Invalid prefetch_indexing"):
            code_and_output(
                add_prefetched,
                (x, y),
                block_sizes=[16, 64],
                prefetch_indexing="tensor_descriptors",
            )

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    @pytest.mark.slow  # spawns an interpreter
    def test_tensor_descriptor_prefetch_without_helion(self) -> None:
        """Helion-free output inlines the tensor-map builder."""
        x = torch.randn(80, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        code = add_prefetched.bind((x, y)).to_code(
            helion.Config(
                block_sizes=[16, 64], prefetch_indexing=["tensor_descriptor"]
            ),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
        self.assertIn(_BOX, code)
        self.assertNotIn("import helion", code)
        run_add_without_helion(code, "add_prefetched", (80, 1024))

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_tensor_descriptor_prefetch_boxes(self) -> None:
        """A full slice is its whole extent, as consecutive boxes past a box
        dim of 256; a rolled reduction's slice its chunk; a scalar index an
        extent of 1."""
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        for reduction_loops in (None, [128]):
            code, out = code_and_output(
                row_sum_prefetched,
                (x,),
                block_sizes=[4],
                prefetch_indexing="tensor_descriptor",
                **(
                    {}
                    if reduction_loops is None
                    else {"reduction_loops": reduction_loops}
                ),
            )
            torch.testing.assert_close(out, x.float().sum(-1), rtol=1e-3, atol=1e-2)
            # The last dim is the bulk form's row segment: no reduction dim.
            self.assertEqual(code.count(f"{_BOX}.2d.L2.global.tile"), 4)
            self.assertIn("helion_tensor_maps.prefetch_map(x, [4, 256])", code)
        w = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        for reduction_loops, box, rows, block in (
            (None, [1, 64, 256], "0", "offset_0"),
            ([16], [1, 16, 256], "tl.cast(roffset_1, tl.int32)", "offset_0_copy"),
        ):
            # One slot whether or not the config rolls the prefetch's slice.
            code, out = code_and_output(
                box_prefetch_copy,
                (w, w.view(16, 64, 256)),
                block_sizes=[64, 64],
                prefetch_indexing=["tensor_descriptor"],
                **(
                    {}
                    if reduction_loops is None
                    else {"reduction_loops": reduction_loops}
                ),
            )
            torch.testing.assert_close(out, w, atol=0, rtol=0)
            self.assertEqual(code.count(f"{_BOX}.3d.L2.global.tile"), 1)
            self.assertIn(f"helion_tensor_maps.prefetch_map(wp, {box})", code)
            # Innermost first: the full last dim, the rows (the chunk's offset
            # when rolled), the scalar index's value.
            self.assertIn(
                f"wp_pf_map_acquired, 0, {rows}, tl.cast({block}, tl.int32)]", code
            )
        # A unit dim's full slice is an extent of 1: no reduction dim.
        args = (w, w.view(1024, 1, 256))
        bound = box_prefetch_copy.bind(args)
        block_sizes = len(bound.env.block_sizes)
        code, out = code_and_output(
            box_prefetch_copy,
            args,
            block_sizes=[64, 64],
            prefetch_indexing="tensor_descriptor",
        )
        torch.testing.assert_close(out, w, atol=0, rtol=0)
        self.assertIn("helion_tensor_maps.prefetch_map(wp, [1, 1, 256])", code)
        self.assertEqual(len(bound.env.block_sizes), block_sizes)

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_tensor_descriptor_prefetch_falls_back_to_pointer(self) -> None:
        """Subscripts a TMA box cannot describe (gathers, an extra mask) keep
        the pointer form in the tensor_descriptor form; such a kernel has no
        key, so an entry (of any length) is unused."""
        x = torch.randn(50, 304, device=DEVICE)
        rows = torch.tensor([3, 7, 49, 0, 5], device=DEVICE)
        for prefetch_indexing in ("tensor_descriptor", ["tensor_descriptor"]):
            code, out = code_and_output(
                gather_prefetched,
                (x, rows),
                block_sizes=[4, 128],
                prefetch_indexing=prefetch_indexing,
            )
            torch.testing.assert_close(out, x[rows])
            self.assertNotIn(_BOX, code)
            self.assertEqual(code.count(_PER_ELEMENT), 2)

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_tensor_descriptor_prefetch_misaligned_base(self) -> None:
        """A kernel whose configs give a prefetch the tensor form is keyed on
        the prefetched input's base alignment: a misaligned call after an
        aligned one keeps the bulk form instead of building a tensor map the
        encoder rejects.  One whose configs keep the pointer form is not."""
        storage = torch.randn([1024 * 256 + 8], device=DEVICE, dtype=torch.bfloat16)
        calls = [
            (w, w.view(16, 64, 256))
            for offset in (0, 1)
            for w in [storage[offset : offset + 1024 * 256].view(1024, 256)]
        ]
        boxed, pinned = (
            helion.kernel(
                box_prefetch_copy.fn,
                static_shapes=True,
                config=helion.Config(block_sizes=[64, 64], prefetch_indexing=form),
            )
            for form in (["tensor_descriptor"], "pointer")
        )
        codes = []
        for w, wp in calls:
            torch.testing.assert_close(boxed(w, wp), w, atol=0, rtol=0)
            codes.append(boxed.bind((w, wp)).to_code())
        self.assertIn(_BOX, codes[0])
        self.assertIn(_BULK, codes[1])
        self.assertNotIn(_BOX, codes[1])
        self.assertNotIn("prefetch_map(", codes[1])
        self.assertIs(pinned.bind(calls[0]), pinned.bind(calls[1]))

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_prefetch_map_first_built_in_capture(self) -> None:
        """A map first built under CUDA graph capture is copied by every graph
        captured before an eager launch (any of them may replay first), and the
        first eager launch writes it for every stream."""
        w = torch.randn(1024, 256, device=DEVICE, dtype=torch.bfloat16)
        w2 = w.clone()
        config = helion.Config(
            block_sizes=[64, 64], prefetch_indexing="tensor_descriptor"
        )
        run = box_prefetch_copy.bind((w, w.view(16, 64, 256))).compile_config(config)
        with _empty_prefetch_maps():
            run(w, w.view(16, 64, 256))
            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            outs = []
            for graph in graphs:
                with torch.cuda.graph(graph):
                    outs.append(run(w2, w2.view(16, 64, 256)))
            (entry,) = [
                e
                for key, e in tensor_maps._PREFETCH_MAPS.items()
                if key[0] == w2.data_ptr()
            ]
            self.assertFalse(entry.written)
            # Either graph may replay first (the second here, then the one that
            # built the map): each one's own copy writes the map.
            for graph, out in ((graphs[1], outs[1]), (graphs[0], outs[0])):
                entry.device.zero_()
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    entry.device.cpu(), entry.host, atol=0, rtol=0
                )
                torch.testing.assert_close(out, w2, atol=0, rtol=0)
            # The first eager launch copies on its stream without waiting; once
            # that copy completed, no launch copies or waits again.
            run(w2, w2.view(16, 64, 256))
            self.assertIsNotNone(entry.copied)
            torch.cuda.synchronize()
            run(w2, w2.view(16, 64, 256))
            self.assertTrue(entry.written)
            del graphs

    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    def test_prefetch_map_cache(self) -> None:
        """The copy's event is recorded before another thread can see it; maps
        live in a pool of their own, also under capture, with the streams that
        read them recorded; maps no graph captured are dropped past the cap on
        an eager miss and freed once no launch holds them; captured ones
        stay."""
        published: list[bool] = []

        class Event(torch.cuda.Event):
            def record(self, stream: torch.cuda.Stream | None = None) -> None:
                published.extend(
                    e.copied is self for e in tensor_maps._PREFETCH_MAPS.values()
                )
                super().record(stream)

        def pool(address: int) -> tuple[int, int]:
            (segment,) = [
                s
                for s in torch.cuda.memory_snapshot()
                if s["address"] <= address < s["address"] + s["total_size"]
            ]
            return segment["segment_pool_id"]

        tensors = [torch.zeros([64, 64], device=DEVICE) for _ in range(4)]
        with (
            _empty_prefetch_maps(),
            mock.patch.object(tensor_maps, "_MAX_PREFETCH_MAPS", 2),
            mock.patch.object(tensor_maps.torch.cuda, "Event", Event),
        ):
            graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            with torch.cuda.graph(graphs[0]):
                captured = tensor_maps.prefetch_map(tensors[0], [64, 64]).address
            self.assertEqual(
                pool(captured), tensor_maps._PREFETCH_MAP_POOLS[tensors[0].device].id
            )
            # Maps launches hold (their host code keeps them until submitted).
            held = [tensor_maps.prefetch_map(t, [64, 64]) for t in tensors[1:3]]
            self.assertTrue(published)
            self.assertFalse(any(published))
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                tensor_maps.prefetch_map(tensors[1], [64, 64])
            self.assertEqual(
                held[0].streams,
                {torch.cuda.current_stream().stream_id, side.stream_id},
            )
            self.assertEqual(len(tensor_maps._PREFETCH_MAPS), 3)
            # A miss under capture drops nothing.
            with torch.cuda.graph(graphs[1]):
                tensor_maps.prefetch_map(tensors[2].view(128, 32), [128, 32])
            self.assertEqual(len(tensor_maps._PREFETCH_MAPS), 4)
            # An eager miss drops the two uncaptured maps: the one a launch
            # holds lives on, the other is freed.
            unheld = weakref.ref(held.pop())
            tensor_maps.prefetch_map(tensors[1].view(32, 128), [32, 128])
            self.assertEqual(len(tensor_maps._PREFETCH_MAPS), 3)
            self.assertIsNone(unheld())
            self.assertNotIn(held[0], tensor_maps._PREFETCH_MAPS.values())
            self.assertEqual(
                tensor_maps.prefetch_map(tensors[0], [64, 64]).address, captured
            )
            del graphs

    def test_prefetch_is_not_a_cross_loop_read(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        # Prefetching what an earlier loop writes is no dependency on it ...
        spec = independent_roots.bind((x, y)).config_spec
        self.assertIsNone(spec.cross_loop_pipeline)
        # ... and a megakernel whose loop also reads it keeps the prefetch.
        code, (tmp, out) = code_and_output(
            dependent_roots,
            (x, y),
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
            num_sm_multiplier=1,
        )
        torch.testing.assert_close(tmp, x + 1)
        torch.testing.assert_close(out, y * 2 + x + 1)
        self.assertEqual(len(_prefetch_lines(code)), 1)
        self.assertIn("tile_dependency", code)

    def test_prefetch_only_root_runs_without_pipeline(self) -> None:
        # The later root rewrites ``ids`` without waiting for the prefetch-only
        # root's loads (test_prefetch_hints), so nothing is scheduled.
        x = torch.randn(64, 256, device=DEVICE)
        ids = torch.arange(32, device=DEVICE, dtype=torch.int32)
        z = torch.full_like(ids, 7)
        # A config saved while the rewrite still waited still loads.
        config = helion.Config(cross_loop_pipeline="static")
        spec = hint_read_roots.bind((x, ids, z)).config_spec
        self.assertTrue(spec.cross_loop_pipeline_retired)
        spec.normalize(config)
        self.assertNotIn("cross_loop_pipeline", config)
        code, out = code_and_output(
            hint_read_roots,
            (x, ids, z),
            pid_type="persistent_blocked",
            num_sm_multiplier=1,
        )
        torch.testing.assert_close(out, z)
        self.assertNotIn("tile_dependency", code)


if __name__ == "__main__":
    unittest.main()
