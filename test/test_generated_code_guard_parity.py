from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import cast

import sympy
import torch
from torch._dynamo.source import GetItemSource
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from helion._compiler.compile_environment import TensorDescriptorAlignmentGuard
from helion._compiler.compile_environment import TensorDescriptorLayoutGuard
from helion._testing import TestCase
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _guard_extractor
from helion.runtime.kernel import BoundKernel


class TestGeneratedCodeGuardParity(TestCase):
    def _bound(self, sources=(), *, layouts=None, alignments=None, strides=()):
        symbols = tuple(sympy.Symbol(f"s{index}") for index in range(len(sources)))
        env = SimpleNamespace(
            specialized_vars=set(symbols),
            specialized_strides=set(strides),
            shape_env=SimpleNamespace(
                var_to_sources=dict(
                    zip(symbols, ([source] for source in sources), strict=True)
                )
            ),
            tensor_descriptor_layout_guards=layouts or {},
            tensor_descriptor_alignment_guards=alignments or {},
            runtime_input_specializations={},
            device=torch.device("cuda"),
            backend_name="triton",
        )
        return SimpleNamespace(
            env=env,
            kernel=SimpleNamespace(
                signature=inspect.signature(lambda value, items: None), configs=[]
            ),
            settings=SimpleNamespace(force_autotune=False),
            _fixed_config_for_td_layout_guards=lambda: None,
        )

    def _check_roundtrip(self, bound, inputs):
        guard_sources = []
        extractors = BoundKernel._specialize_extra(
            cast("BoundKernel", bound), guard_sources
        )
        self.assertTrue(extractors)
        self.assertEqual(len(extractors), len(guard_sources))
        for extractor, source in zip(extractors, guard_sources, strict=True):
            guard = cache._serialized_guard(source, {"value": 0, "items": 1}, "triton")
            self.assertIsNotNone(guard)
            # The real persistence boundary uses JSON, including for nested
            # argument projections and tensor-descriptor guard operands.
            guard = json.loads(json.dumps(guard))
            self.assertTrue(cache._valid_guard(guard))
            rebuilt = _guard_extractor(guard, ("value", "items"))
            for args in inputs:
                with self.subTest(guard=guard, args=args):
                    self.assertEqual(rebuilt(args), extractor(args))

    def test_argument_and_container_projection_roundtrip(self):
        value = LocalSource("value", is_input=True)
        items = LocalSource("items", is_input=True)
        for source, inputs in (
            (value, ((2, []), (7, []))),
            (GetItemSource(items, 1), ((0, [2, 3]), (0, (7, 11)))),
            (GetItemSource(items, "scale"), ((0, {"scale": 2}), (0, {"scale": 3}))),
            (
                GetItemSource(items, "scale"),
                ((0, SimpleNamespace(scale=2)), (0, SimpleNamespace(scale=3))),
            ),
        ):
            self._check_roundtrip(self._bound((source,)), inputs)

    def test_tensor_property_and_tensor_list_roundtrip(self):
        value = LocalSource("value", is_input=True)
        items = LocalSource("items", is_input=True)
        stride = TensorPropertySource(value, TensorProperty.STRIDE, 0)
        sources = (
            TensorPropertySource(value, TensorProperty.SIZE, 1),
            stride,
            TensorPropertySource(items, TensorProperty.SIZE, 0),
            TensorPropertySource(items, TensorProperty.STRIDE, 1),
            TensorPropertySource(GetItemSource(items, 0), TensorProperty.SIZE, 1),
        )
        self._check_roundtrip(
            self._bound(sources, strides=(stride,)),
            (
                (torch.empty(8, 16), [torch.empty(3, 4), torch.empty(5, 6)]),
                (torch.empty(16, 8).T, (torch.empty(4, 3).T, torch.empty(6, 5).T)),
            ),
        )

    def test_descriptor_layout_rank_extent_and_stride_roundtrip(self):
        value = LocalSource("value", is_input=True)
        layout = TensorDescriptorLayoutGuard(ndim=2, element_size=4)
        meta = torch.device("meta")
        self._check_roundtrip(
            self._bound(layouts={value: layout}),
            tuple(
                (tensor, ())
                for tensor in (
                    torch.empty(8, 16),
                    torch.empty(8, 16).T,
                    torch.empty(8, 32)[:, ::2],
                    torch.empty(1, 16),
                    torch.empty(0, 16),
                    torch.empty(16),
                    torch.empty_strided((257, 16), (16, 1), device=meta),
                    torch.empty_strided((2**31, 16), (16, 1), device=meta),
                )
            ),
        )
        # CuTe's layout-only extractor is intentionally not a disk-cache
        # schema: generated source caching is currently Triton-only.
        self.assertIsNone(
            cache._serialized_guard((value, layout, 256), {"value": 0}, "cute")
        )

    def test_descriptor_alignment_and_storage_offset_roundtrip(self):
        value = LocalSource("value", is_input=True)
        storage = torch.empty(129)
        inputs = (
            (storage[:128].view(8, 16), ()),
            (storage[1:].view(8, 16), ()),
            (storage[16:128].view(7, 16), ()),
        )
        for requires_zero_storage_offset in (False, True):
            guard = TensorDescriptorAlignmentGuard(
                requires_zero_storage_offset=requires_zero_storage_offset
            )
            self._check_roundtrip(self._bound(alignments={value: guard}), inputs)
