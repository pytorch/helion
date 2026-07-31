from __future__ import annotations

import inspect

from examples.linear.kda_packed_decode import _helion_fused_recurrent_kda_packed_decode
from examples.linear.kda_packed_decode import helion_fused_recurrent_kda_packed_decode
from examples.linear.kda_packed_decode import make_kda_inputs
from examples.linear.kda_packed_decode import torch_fused_recurrent_kda_packed_decode
import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends


@onlyBackends(["triton"])
class TestKdaPackedDecode(RefEagerTestDisabled, TestCase):
    def test_uses_one_config_for_all_shapes(self) -> None:
        self.assertEqual(len(_helion_fused_recurrent_kda_packed_decode.configs), 1)
        self.assertFalse(
            _helion_fused_recurrent_kda_packed_decode.settings.static_shapes
        )
        config = _helion_fused_recurrent_kda_packed_decode.configs[0]
        self.assertEqual(config.block_sizes, [8])
        self.assertEqual(config.loop_orders, [[2, 1, 0]])
        self.assertEqual(config.pid_type, "xyz")

    def test_public_signature(self) -> None:
        signature = inspect.signature(helion_fused_recurrent_kda_packed_decode)
        self.assertEqual(
            list(signature.parameters),
            [
                "mixed_qkv",
                "a",
                "b",
                "A_log",
                "dt_bias",
                "scale",
                "initial_state",
                "out",
                "ssm_state_indices",
                "use_qk_l2norm_in_kernel",
            ],
        )
        self.assertIs(signature.parameters["use_qk_l2norm_in_kernel"].default, False)

    def test_values_mutations_aliases_and_padding(self) -> None:
        for use_qk_l2norm_in_kernel in (False, True):
            with self.subTest(use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel):
                inputs = make_kda_inputs(
                    3,
                    2,
                    4,
                    128,
                    128,
                    device=DEVICE,
                    pool_size=7,
                    seed=123,
                )
                inputs.ssm_state_indices.copy_(
                    torch.tensor([5, -1, 2], device=DEVICE, dtype=torch.int32)
                )
                inputs.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
                original_state = inputs.initial_state.clone()

                reference_inputs = inputs.clone_mutable()
                actual_inputs = inputs.clone_mutable()
                torch_fused_recurrent_kda_packed_decode(*reference_inputs.args())
                result = helion_fused_recurrent_kda_packed_decode(*actual_inputs.args())

                self.assertEqual(result[0].data_ptr(), actual_inputs.out.data_ptr())
                self.assertEqual(
                    result[1].data_ptr(), actual_inputs.initial_state.data_ptr()
                )
                torch.testing.assert_close(
                    actual_inputs.out, reference_inputs.out, atol=2e-2, rtol=1e-2
                )
                torch.testing.assert_close(
                    actual_inputs.initial_state,
                    reference_inputs.initial_state,
                    atol=2e-2,
                    rtol=1e-2,
                )
                self.assertEqual(torch.count_nonzero(actual_inputs.out[1]).item(), 0)

                untouched = torch.tensor(
                    [0, 1, 3, 4, 6], device=DEVICE, dtype=torch.long
                )
                self.assertTrue(
                    torch.equal(
                        actual_inputs.initial_state[untouched],
                        original_state[untouched],
                    )
                )
