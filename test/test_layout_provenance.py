from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_function import StaticShape
from helion._compiler.type_info import CallableType
from helion._compiler.type_info import TensorType
from helion._compiler.type_info import TypeInfo
from helion._compiler.variable_origin import ArgumentOrigin
from helion._compiler.variable_origin import NameOrigin
import helion.language as hl


class _StrideHarness:
    def __init__(self) -> None:
        self.runtime_stride = object()

    def _tensor_property(self, *args: object) -> object:
        return self.runtime_stride


class TestLayoutProvenance(unittest.TestCase):
    def test_kernel_trace_does_not_mark_torch_empty_out_as_fresh(self) -> None:
        @helion.kernel(
            autotune_effort="none",
            backend="triton",
            static_shapes=True,
        )
        def empty_out(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.size(), out=x)
            for tile_m, tile_n in hl.tile(x.size(), block_size=[1, 4]):
                out[tile_m, tile_n] = x[tile_m, tile_n]
            return out

        x = torch.empty_strided((2, 4), (1, 2))
        bound = empty_out.bind((x,))
        self.assertFalse(
            bound.env.tensor_layout_is_symbolically_exact(bound.fake_args[0])
        )

    def test_exact_factory_layout_provenance_is_positive_and_storage_scoped(
        self,
    ) -> None:
        env = CompileEnvironment(
            torch.device("cpu"),
            helion.Settings(backend="triton", static_shapes=False),
        )
        real_input = torch.empty_strided((2, 3), (1, 2))
        origin = NameOrigin("allocation")

        with env:
            fake_input = env.to_fake(real_input, ArgumentOrigin("x"))
            assert isinstance(fake_input, torch.Tensor)
            input_type = TensorType(ArgumentOrigin("x"), fake_input)

            shape = TypeInfo.from_example((fake_input.size(0), 32), origin)
            device = TypeInfo.from_example(torch.device("cpu"), origin)
            allocation = CallableType(origin, torch.empty).propagate_call(
                (shape,), {"device": device}, origin
            )
            assert isinstance(allocation, TensorType)

            input_like = CallableType(origin, torch.empty_like).propagate_call(
                (input_type,), {}, origin
            )
            out_target = torch.empty_strided((2, 3), (1, 2))
            out_target_type = TensorType(ArgumentOrigin("out"), out_target)
            input_shape = TypeInfo.from_example((2, 3), origin)
            out_alias = CallableType(origin, torch.empty).propagate_call(
                (input_shape,), {"out": out_target_type}, origin
            )
            exact_like = CallableType(origin, torch.empty_like).propagate_call(
                (allocation,), {}, origin
            )
            assert isinstance(input_like, TensorType)
            assert isinstance(out_alias, TensorType)
            assert isinstance(exact_like, TensorType)

            self.assertFalse(env.tensor_layout_is_symbolically_exact(fake_input))
            self.assertFalse(
                env.tensor_layout_is_symbolically_exact(fake_input.transpose(0, 1))
            )
            self.assertFalse(
                env.tensor_layout_is_symbolically_exact(input_like.fake_value)
            )
            self.assertIs(out_alias.fake_value, out_target)
            self.assertFalse(
                env.tensor_layout_is_symbolically_exact(out_alias.fake_value)
            )
            self.assertTrue(
                env.tensor_layout_is_symbolically_exact(allocation.fake_value)
            )
            self.assertTrue(
                env.tensor_layout_is_symbolically_exact(
                    allocation.fake_value.transpose(0, 1)
                )
            )
            self.assertTrue(
                env.tensor_layout_is_symbolically_exact(exact_like.fake_value)
            )

    def test_factory_layout_registration_requires_fresh_storage(self) -> None:
        env = CompileEnvironment(
            torch.device("cpu"),
            helion.Settings(backend="triton", static_shapes=False),
        )

        with env:
            aliased = torch.empty((2, 3))
            env.register_tensor_factory_layout(
                torch.empty,
                ([{"tensor": aliased}],),
                {},
                aliased,
            )
            self.assertFalse(env.tensor_layout_is_symbolically_exact(aliased))

            out_target = torch.empty((2, 3))
            fresh_result = torch.empty((2, 3))
            env.register_tensor_factory_layout(
                torch.empty,
                ((2, 3),),
                {"out": out_target},
                fresh_result,
            )
            self.assertFalse(env.tensor_layout_is_symbolically_exact(fresh_result))

    def test_dynamic_stride_literal_requires_positive_layout_provenance(self) -> None:
        env = CompileEnvironment(
            torch.device("cpu"),
            helion.Settings(backend="triton", static_shapes=False),
        )
        real_input = torch.empty_strided((2, 3), (1, 2))
        origin = NameOrigin("allocation")

        with env:
            fake_input = env.to_fake(real_input, ArgumentOrigin("x"))
            assert isinstance(fake_input, torch.Tensor)
            input_type = TensorType(ArgumentOrigin("x"), fake_input)
            shape = TypeInfo.from_example((fake_input.size(0), 32), origin)
            device = TypeInfo.from_example(torch.device("cpu"), origin)
            allocation = CallableType(origin, torch.empty).propagate_call(
                (shape,), {"device": device}, origin
            )
            input_like = CallableType(origin, torch.empty_like).propagate_call(
                (input_type,), {}, origin
            )
            assert isinstance(allocation, TensorType)
            assert isinstance(input_like, TensorType)

            harness = _StrideHarness()
            device_function = cast("DeviceFunction", harness)
            literal = DeviceFunction.tensor_stride(
                device_function,
                allocation.fake_value,
                1,
            )
            self.assertIsInstance(literal, StaticShape)
            self.assertEqual(literal.name, "1")

            host = SimpleNamespace(params=SimpleNamespace(arguments={"x": fake_input}))
            with patch(
                "helion._compiler.host_function.HostFunction.current",
                return_value=host,
            ):
                for tensor in (
                    fake_input,
                    fake_input.transpose(0, 1),
                    input_like.fake_value,
                ):
                    stride = DeviceFunction.tensor_stride(
                        device_function,
                        tensor,
                        0,
                    )
                    self.assertIs(stride, harness.runtime_stride)


if __name__ == "__main__":
    unittest.main()
