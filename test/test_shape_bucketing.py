from __future__ import annotations

import unittest

import torch

from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


def _dummy(x: torch.Tensor) -> torch.Tensor:
    return x


class TestShapeBucketing(unittest.TestCase):
    def test_min2_bucketing_default(self) -> None:
        k = kernel(_dummy, settings=Settings(static_shapes=False))

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t7 = torch.empty(7, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])
        key_7 = k.specialization_key([t7])

        # min2: 0,1,>=2 (as 2)
        self.assertNotEqual(key_0, key_2)
        self.assertNotEqual(key_1, key_2)
        self.assertEqual(key_2, key_7)

    def test_zero_nonzero_bucketing(self) -> None:
        k = kernel(
            _dummy,
            settings=Settings(static_shapes=False, shape_bucketing="zero_nonzero"),
        )

        t0 = torch.empty(0, 3)
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)

        key_0 = k.specialization_key([t0])
        key_1 = k.specialization_key([t1])
        key_2 = k.specialization_key([t2])

        # zero_nonzero: keep 0 distinct; unify 1 with >=2
        self.assertNotEqual(key_0, key_2)
        self.assertEqual(key_1, key_2)


if __name__ == "__main__":
    unittest.main()
