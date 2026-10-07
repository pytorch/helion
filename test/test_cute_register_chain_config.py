from __future__ import annotations

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion import exc
from helion._compiler.backend import CuteBackend
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import MatmulFact

_KEY = "cute_register_chain"


def _spec(capability: tuple[int, int] | None, *, matmul: bool = True) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=capability,
        device=torch.device("cpu"),
        num_sm=1,
    )
    if matmul:
        spec.matmul_facts.append(
            MatmulFact(2, 2, None, None, None, 64, 64, 64, torch.float32, torch.float32)
        )
    return spec


def _fragment(spec: ConfigSpec) -> EnumFragment:
    fragment = spec._flat_fields()[_KEY]
    assert isinstance(fragment, EnumFragment)
    return fragment


class TestCuteRegisterChainConfig(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_mock_cuda_unavailable())
        for name in (
            "_lazy_init",
            "current_device",
            "get_device_capability",
            "get_device_properties",
        ):
            stack.enter_context(
                patch.object(
                    torch.cuda, name, side_effect=AssertionError("GPU forbidden")
                )
            )

    def test_manual_choice_round_trips_but_does_not_create_search_coverage(
        self,
    ) -> None:
        for capability in ((8, 0), (9, 0), (10, 0), (12, 0)):
            with self.subTest(capability=capability):
                spec = _spec(capability)
                config = spec.default_config()
                config.config[_KEY] = True
                spec.normalize(config)
                generation = ConfigGeneration(spec)
                restored = generation.unflatten(generation.flatten(config))
                self.assertEqual(restored, config)
                self.assertIs(restored[_KEY], True)
                self.assertEqual(_fragment(spec).search_values(), [False])

    def test_structural_seed_adds_only_one_search_coordinate_choice(self) -> None:
        spec = _spec((10, 0))
        defaults = spec.default_config()
        fields = {
            key: fragment.fingerprint()
            for key, fragment in spec._flat_fields().items()
            if key != _KEY
        }
        spec.compiler_seed_configs = [helion.Config(**{_KEY: True})]
        self.assertEqual(_fragment(spec).choices, (False, True))
        self.assertEqual(_fragment(spec).search_values(), [False, True])
        self.assertEqual(_fragment(spec).pattern_neighbors(False), [True])
        self.assertEqual(spec.default_config(), defaults)
        self.assertEqual(
            {
                key: fragment.fingerprint()
                for key, fragment in spec._flat_fields().items()
                if key != _KEY
            },
            fields,
        )
        dimension = next(
            row for row in spec.iter_search_dimensions() if row.name == _KEY
        )
        self.assertEqual(dimension.values, [False, True])

    def test_invalid_types_targets_and_no_matmul_reject(self) -> None:
        for value in (None, 1, "true", []):
            with self.subTest(value=value):
                spec = _spec((9, 0))
                config = helion.Config.from_dict({_KEY: value})
                with self.assertRaisesRegex(exc.InvalidConfig, "cute_register_chain"):
                    spec.normalize(config)
                spec.normalize(config, _fix_invalid=True)
                self.assertIs(config[_KEY], False)
        for capability, matmul in ((None, True), ((7, 5), True), ((10, 0), False)):
            with self.subTest(capability=capability, matmul=matmul):
                spec = _spec(capability, matmul=matmul)
                config = helion.Config(**{_KEY: True})
                with self.assertRaisesRegex(exc.InvalidConfig, "cute_register_chain"):
                    spec.normalize(config)
                spec.normalize(config, _fix_invalid=True)
                self.assertIs(config[_KEY], False)
                self.assertNotIn(_KEY, spec._flat_fields())

    def test_default_is_off_and_manual_chain_does_not_enable_collectives(self) -> None:
        spec = _spec((10, 0))
        self.assertIs(_fragment(spec).default(), False)
        defaults = spec.default_config()
        self.assertIs(defaults.get(_KEY), False)
        self.assertIs(defaults.get("cute_collective_mma"), False)
        config = spec.default_config()
        config.config[_KEY] = True
        spec.normalize(config)
        self.assertIs(config[_KEY], True)
        self.assertEqual(
            {key: value for key, value in config.config.items() if key != _KEY},
            {key: value for key, value in defaults.config.items() if key != _KEY},
        )


if __name__ == "__main__":
    unittest.main()
