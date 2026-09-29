from __future__ import annotations

import ast
import contextlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from helion._compiler.distributed_ll import LLAllocation
from helion._compiler.distributed_ll import _access_is_pair_aligned
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.triton.distributed_ll import _ll_push_transport
from helion._compiler.triton.distributed_ll import _pair_shapes
from helion._compiler.triton.distributed_ll import _split_pairs
from helion.runtime.settings import Settings
from helion.runtime.triton.launcher import _get_distributed_ll_mailbox
from helion.runtime.triton.launcher import default_launcher


class _Names:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def new_var(self, hint: str, *, dce: bool) -> str:
        count = self.counts.get(hint, 0)
        self.counts[hint] = count + 1
        return f"{hint}_{count}"


def _transport_source(*, multicast: bool, masked: bool) -> str:
    statements = _ll_push_transport(
        _Names(),  # type: ignore[arg-type]
        "mailbox_ptrs",
        "multicast_ptr" if multicast else None,
        "slot",
        "offset",
        "word",
        "mask" if masked else None,
        4,
    )
    return ast.unparse(ast.Module(body=statements, type_ignores=[]))


class TestDistributedLLMulticastCodegen(unittest.TestCase):
    @staticmethod
    def _access(*, offset: int, extent: int) -> TileAccess:
        return TileAccess(
            access_id=0,
            memory_op_index=0,
            graph_id=0,
            root=0,
            allocation_id=0,
            kind="store",
            tensor_name="buffer",
            tensor_shape=(4096,),
            tensor_strides=(1,),
            storage_offset=0,
            subscript_dims=(0,),
            subscript_affine_block_ids=(None,),
            subscript_index_scales=(1,),
            subscript_offsets=(offset,),
            subscript_is_scalar=(False,),
            has_explicit_mask=False,
            layout_is_symbolically_exact=True,
            subscript_is_full_slice=(False,),
            subscript_static_extents=(extent,),
        )

    def test_setting_is_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Settings().distributed_ll_multicast)
        self.assertTrue(
            Settings(distributed_ll_multicast=True).distributed_ll_multicast
        )

    def test_default_transport_is_unchanged_unicast(self) -> None:
        source = _transport_source(multicast=False, masked=False)
        self.assertNotIn("multimem.st", source)
        self.assertNotIn("if ", source)
        self.assertEqual(source.count("st.relaxed.sys.global.u64"), 4)
        self.assertEqual(source.count("tl.load(mailbox_ptrs"), 4)

    def test_multicast_has_one_b64_push_and_exact_unicast_fallback(self) -> None:
        source = _transport_source(multicast=True, masked=False)
        self.assertIn("if multicast_ptr != 0:", source)
        self.assertEqual(
            source.count("multimem.st.relaxed.sys.global.b64"),
            1,
        )
        self.assertEqual(source.count("st.relaxed.sys.global.u64"), 4)
        self.assertEqual(source.count("tl.load(mailbox_ptrs"), 4)
        self.assertIn("tl.cast(multicast_ptr, tl.int64)", source)

    def test_multicast_and_fallback_preserve_store_mask(self) -> None:
        source = _transport_source(multicast=True, masked=True)
        self.assertEqual(source.count("setp.ne.b32 p, $3, 0"), 5)
        self.assertEqual(source.count("args=["), 5)
        self.assertEqual(source.count("word, mask"), 5)

    def test_bf16_pairing_halves_mailbox_words(self) -> None:
        allocation = LLAllocation(
            allocation_id=0,
            tensor_name="buffer",
            dtype=torch.bfloat16,
            numel=4096,
            world_size=2,
            mailbox_offset=0,
            producer_root=0,
            elements_per_word=2,
        )
        self.assertEqual(allocation.words_per_source(), 2048)
        self.assertEqual(allocation.slot_words(), 4096)

    def test_pair_shapes_use_selected_codegen_extents(self) -> None:
        for dimensions, expected_numel in ((["32"], "(32)"), (["8"], "(8)")):
            with self.subTest(dimensions=dimensions):
                state = SimpleNamespace(
                    tile_strategy=SimpleNamespace(
                        shape_dims=lambda _shape, dims=dimensions: dims
                    )
                )
                numel, pair_shape, packed_shape = _pair_shapes(
                    state,  # type: ignore[arg-type]
                    [1],
                )
                self.assertEqual(numel, expected_numel)
                self.assertEqual(pair_shape, f"[({expected_numel}) // 2, 2]")
                self.assertEqual(packed_shape, f"[({expected_numel}) // 2]")

    def test_pair_components_use_triton_split(self) -> None:
        statements: list[ast.stmt] = []
        state = SimpleNamespace(
            device_function=_Names(),
            add_statement=statements.append,
        )
        low, high = _split_pairs(
            state,  # type: ignore[arg-type]
            ast.Name(id="pairs", ctx=ast.Load()),
            "part",
        )
        source = ast.unparse(ast.Module(body=statements, type_ignores=[]))
        self.assertEqual((low, high), ("part_low_0", "part_high_0"))
        self.assertEqual(source, "part_low_0, part_high_0 = tl.split(pairs)")
        self.assertNotIn("[:,", source)

    def test_pairing_requires_even_base_extent_and_allocation(self) -> None:
        self.assertTrue(
            _access_is_pair_aligned(self._access(offset=0, extent=32), (4096,))
        )
        self.assertFalse(
            _access_is_pair_aligned(self._access(offset=1, extent=32), (4096,))
        )
        self.assertFalse(
            _access_is_pair_aligned(self._access(offset=0, extent=31), (4096,))
        )
        self.assertFalse(
            _access_is_pair_aligned(self._access(offset=0, extent=32), (4099,))
        )


class TestDistributedLLMulticastLauncher(unittest.TestCase):
    def test_launcher_appends_runtime_multicast_pointer(self) -> None:
        calls: list[tuple[object, ...]] = []

        class FakeJITFunction:
            params: tuple[object, ...] = ()
            cache_key = "kernel"

            def run(self, *args: object, **kwargs: object) -> str:
                calls.append(args)
                return "launched"

        anchor = torch.empty(1)
        with patch(
            "helion.runtime.triton.launcher._get_distributed_ll_mailbox",
            return_value=("mailbox", "peer_ptrs", 1, 0x400000000),
        ) as get_mailbox:
            result = default_launcher(
                FakeJITFunction(),
                (8,),
                anchor,
                num_warps=4,
                num_stages=2,
                _distributed_readiness_device_anchor=anchor,
                _distributed_readiness_world_size=2,
                _distributed_readiness_process_group_name="tp",
                _distributed_ll_mailbox_words=128,
                _distributed_ll_multicast=True,
            )

        self.assertEqual(result, "launched")
        self.assertEqual(
            calls,
            [(anchor, "mailbox", "peer_ptrs", 1, 0x400000000)],
        )
        get_mailbox.assert_called_once_with(
            unittest.mock.ANY,
            anchor,
            "tp",
            128,
            multicast=True,
            expected_world_size=2,
            launch_fingerprint=unittest.mock.ANY,
        )

    def test_multicast_mailbox_uses_group_scoped_allocation(self) -> None:
        kernel = SimpleNamespace()
        destination = SimpleNamespace(device=torch.device("cuda:0"))
        mailbox = MagicMock()
        mailbox.data_ptr.return_value = 1234
        handle = SimpleNamespace(
            rank=0,
            buffer_ptrs=[1234, 5678],
            buffer_ptrs_dev="peer_ptrs",
            multicast_ptr=0x400000000,
        )
        stream = SimpleNamespace(cuda_stream=7, synchronize=MagicMock())

        def gather_fingerprints(output, value, *, group):
            self.assertEqual(group, "resolved-group")
            output[:] = [value, value]

        with (
            patch("torch.cuda.device", return_value=contextlib.nullcontext()),
            patch("torch.cuda.current_stream", return_value=stream),
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value="resolved-group",
            ),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=gather_fingerprints,
            ),
            patch(
                "torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p",
                return_value=mailbox,
            ) as allocate,
            patch(
                "torch._C._distributed_c10d._SymmetricMemory.rendezvous",
                return_value=handle,
            ),
            patch("torch.distributed._symmetric_memory.empty") as legacy_allocate,
        ):
            result = _get_distributed_ll_mailbox(
                kernel,
                destination,  # type: ignore[arg-type]
                "tp",
                128,
                multicast=True,
                expected_world_size=2,
                launch_fingerprint="fingerprint",
            )

        self.assertEqual(result, (mailbox, "peer_ptrs", 0, 0x400000000))
        allocate.assert_called_once_with(
            (128,), (1,), torch.uint64, torch.device("cuda:0"), "tp"
        )
        legacy_allocate.assert_not_called()
        mailbox.zero_.assert_called_once_with()
        stream.synchronize.assert_called_once_with()
        cache = vars(kernel)["_helion_distributed_ll_mailbox_cache"]
        self.assertIs(next(iter(cache.values()))[2], handle)


if __name__ == "__main__":
    unittest.main()
