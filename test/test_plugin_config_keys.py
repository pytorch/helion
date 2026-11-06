from __future__ import annotations

import contextlib
from typing import Iterator
import unittest
import warnings

import torch

import helion
from helion import backend_api
from helion import exc
from helion._compiler.backend_registry import _REGISTRY
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BACKEND_SPECIFIC_KEYS
from helion.autotuner.config_spec import VALID_KEYS
from helion.autotuner.config_spec import backend_specific_keys
from helion.autotuner.config_spec import valid_keys
import helion.language as hl

_KNOB = "_test_knob"


class _KnobBackend(backend_api.TritonBackend):
    @property
    def name(self) -> str:
        return "_test_knob_backend"

    @property
    def codegen_name(self) -> str:
        return "triton"

    @classmethod
    def plugin_config_keys(cls) -> frozenset[str]:
        return frozenset({_KNOB})

    def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
        return {**super().tunable_fragments(), _KNOB: EnumFragment(choices=(1, 2))}


@contextlib.contextmanager
def _registered(*classes: type[backend_api.Backend]) -> Iterator[None]:
    from helion._compiler import backend_registry

    before = dict(_REGISTRY)
    repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
    try:
        for cls in classes:
            backend_api.register_compiler_backend(cls)
        yield
    finally:
        # Change-only restore (never blank the registry) and undo the repair
        # invalidation that register_compiler_backend performed.
        for name in [k for k in _REGISTRY if k not in before]:
            _REGISTRY.pop(name, None)
        for name, value in before.items():
            if _REGISTRY.get(name) is not value:
                _REGISTRY[name] = value
        backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _user_knob_kernel(x: torch.Tensor) -> torch.Tensor:
    hl.register_tunable(_KNOB, EnumFragment(choices=(1, 2)))
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile]
    return out


class TestPluginConfigKeySets(unittest.TestCase):
    def test_builtin_sets_unchanged_without_plugin_keys(self) -> None:
        self.assertEqual(backend_specific_keys(), BACKEND_SPECIFIC_KEYS)
        self.assertEqual(valid_keys(), VALID_KEYS)

    def test_registered_keys_are_backend_specific(self) -> None:
        with _registered(_KnobBackend):
            self.assertIn(_KNOB, valid_keys())
            self.assertIn(_KNOB, backend_specific_keys())
            self.assertTrue(_KnobBackend().supports_config_key(_KNOB))
            self.assertFalse(backend_api.TritonBackend().supports_config_key(_KNOB))
        self.assertNotIn(_KNOB, valid_keys())

    def test_clash_with_builtin_key_rejected(self) -> None:
        class _Clash(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_clash_builtin"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return frozenset({"num_warps"})

        with _registered(), self.assertRaisesRegex(ValueError, "num_warps"):
            backend_api.register_compiler_backend(_Clash)
        self.assertNotIn("_test_clash_builtin", _REGISTRY)

    def test_clash_between_unrelated_backends_rejected(self) -> None:
        class _Unrelated(backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_clash_other"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return frozenset({_KNOB})

        with _registered(_KnobBackend), self.assertRaisesRegex(ValueError, _KNOB):
            backend_api.register_compiler_backend(_Unrelated)
        self.assertNotIn("_test_clash_other", _REGISTRY)

    def test_backend_family_shares_keys(self) -> None:
        # Like TileIR subclassing Triton: a variant inherits its base's keys.
        class _Variant(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_variant"

        with _registered(_KnobBackend, _Variant):
            self.assertTrue(_Variant().supports_config_key(_KNOB))

    def test_family_variant_can_extend_keys(self) -> None:
        # The natural extension pattern: a variant re-lists the family keys
        # via super() and adds its own. It must register in either order and
        # the base must stay disjoint from the new key.
        knob2 = "_test_knob2"

        class _Extending(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_extending"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return super().plugin_config_keys() | {knob2}

            def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
                return {
                    **super().tunable_fragments(),
                    knob2: EnumFragment(choices=(1, 2)),
                }

        for order in ((_KnobBackend, _Extending), (_Extending, _KnobBackend)):
            with _registered(*order):
                variant = _Extending()
                self.assertTrue(variant.supports_config_key(_KNOB))
                self.assertTrue(variant.supports_config_key(knob2))
                self.assertFalse(_KnobBackend().supports_config_key(knob2))

    def test_sibling_backends_clash(self) -> None:
        # Two subclasses of one base inherit the same keys but are not a
        # family; whether the base is registered does not matter.
        class _Sib1(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_sib1"

        class _Sib2(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_sib2"

        for base in ((), (_KnobBackend,)):
            with (
                _registered(*base, _Sib1),
                self.assertRaisesRegex(ValueError, _KNOB),
            ):
                backend_api.register_compiler_backend(_Sib2)
            self.assertNotIn("_test_knob_sib2", _REGISTRY)

    def test_unrelated_extension_still_rejected(self) -> None:
        # Allowing family overlap must not open unrelated backends to sharing
        # a key: a non-subclass redeclaring _KNOB still clashes.
        class _Unrelated2(backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_knob_unrelated2"

            @property
            def codegen_name(self) -> str:
                return "triton"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return frozenset({_KNOB})

        with _registered(_KnobBackend), self.assertRaisesRegex(ValueError, _KNOB):
            backend_api.register_compiler_backend(_Unrelated2)
        self.assertNotIn("_test_knob_unrelated2", _REGISTRY)

    def test_shared_mixin_keys_still_clash(self) -> None:
        # Two *unrelated* backends that both inherit plugin_config_keys from a
        # common mixin are not a family (neither is a subclass of the other);
        # the mixin must not be treated as a shared owner that licenses the
        # collision.
        class _KnobMixin:
            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return frozenset({_KNOB})

        class _MixA(_KnobMixin, backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_knob_mixa"

        class _MixB(_KnobMixin, backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_knob_mixb"

        with _registered(_MixA), self.assertRaisesRegex(ValueError, _KNOB):
            backend_api.register_compiler_backend(_MixB)
        self.assertNotIn("_test_knob_mixb", _REGISTRY)


class _AllowlistBackend(_KnobBackend):
    # Mimics FlyDSL/Pallas/Metal: supports_config_key never calls the base.
    @property
    def name(self) -> str:
        return "_test_allowlist"

    @classmethod
    def plugin_config_keys(cls) -> frozenset[str]:
        return frozenset({"_test_allowlist_knob"})

    def supports_config_key(self, key: str) -> bool:
        return key == "block_sizes"

    def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
        return {"_test_allowlist_knob": EnumFragment(choices=(1, 2))}


@onlyBackends(["triton"])
class TestPluginConfigKeysInConfigSpec(TestCase):
    def test_key_is_searched_and_validated(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        with _registered(_KnobBackend):
            plugin = helion.kernel(backend="_test_knob_backend")(_add).bind((x, x))
            spec = plugin.config_spec
            self.assertIn(_KNOB, spec._flat_fields())
            self.assertIn(spec.default_config().config[_KNOB], (1, 2))
            config = spec.default_config()
            config.config[_KNOB] = 2
            spec.normalize(config)
            self.assertEqual(config.config[_KNOB], 2)

            # Another backend rejects the plugin's key instead of dropping it.
            triton = helion.kernel(backend="triton")(_add).bind((x, x))
            config = triton.config_spec.default_config()
            config.config[_KNOB] = 1
            with self.assertRaisesRegex(exc.InvalidConfig, _KNOB):
                triton.config_spec.normalize(config)

    def test_random_configs_and_json_round_trip(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        with _registered(_KnobBackend):
            spec = helion.kernel(backend="_test_knob_backend")(_add).bind((x, x))
            spec = spec.config_spec
            generation = ConfigGeneration(spec)
            seen = {generation.random_config().config[_KNOB] for _ in range(32)}
            self.assertEqual(seen, {1, 2})

            config = spec.default_config()
            config.config[_KNOB] = 2
            restored = helion.Config.from_json(config.to_json())
            spec.normalize(restored)
            self.assertEqual(restored.config[_KNOB], 2)

    def test_allowlist_backend_accepts_own_key(self) -> None:
        x = torch.randn(64, 64, device=DEVICE)
        with _registered(_AllowlistBackend):
            spec = helion.kernel(backend="_test_allowlist")(_add).bind((x, x))
            config = spec.config_spec.default_config()
            self.assertIn(config.config["_test_allowlist_knob"], (1, 2))

    def test_plugin_registered_after_bind_stays_rejected_by_old_spec(self) -> None:
        # A ConfigSpec snapshots its backend's plugin keys at bind time: a
        # plugin registered later must not retroactively make foreign keys
        # acceptable to an already-bound spec.
        x = torch.randn(64, 64, device=DEVICE)
        spec = helion.kernel(backend="triton")(_add).bind((x, x)).config_spec
        config = spec.default_config()
        config.config[_KNOB] = 1
        with _registered(_KnobBackend):
            self.assertFalse(spec.supports_config_key(_KNOB))
            with self.assertRaisesRegex(exc.InvalidConfig, _KNOB):
                spec.normalize(config)

    def test_family_variant_extended_key_is_searched_end_to_end(self) -> None:
        knob2 = "_test_knob2"

        class _Extending(_KnobBackend):
            @property
            def name(self) -> str:
                return "_test_knob_extending"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return super().plugin_config_keys() | {knob2}

            def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
                return {
                    **super().tunable_fragments(),
                    knob2: EnumFragment(choices=(1, 2)),
                }

        x = torch.randn(64, 64, device=DEVICE)
        with _registered(_KnobBackend, _Extending):
            variant = (
                helion.kernel(backend="_test_knob_extending")(_add)
                .bind((x, x))
                .config_spec
            )
            self.assertIn(knob2, variant._flat_fields())
            self.assertIn(_KNOB, variant._flat_fields())
            config = variant.default_config()
            self.assertIn(config.config[knob2], (1, 2))
            config.config[_KNOB] = 2
            config.config[knob2] = 1
            variant.normalize(config)
            self.assertEqual(config.config[_KNOB], 2)
            self.assertEqual(config.config[knob2], 1)
            # The base's own spec never sees the variant's key.
            base = (
                helion.kernel(backend="_test_knob_backend")(_add)
                .bind((x, x))
                .config_spec
            )
            self.assertNotIn(knob2, base._flat_fields())
            base_config = base.default_config()
            base_config.config[knob2] = 1
            with self.assertRaisesRegex(exc.InvalidConfig, knob2):
                base.normalize(base_config)

    def test_bound_spec_stays_consistent_after_backend_unregistered(self) -> None:
        # Unregistering (failed-plugin rollback, tests) must leave a bound
        # spec self-consistent: supports_config_key, is_supported_config and
        # normalize all keep accepting the backend's own keys.
        x = torch.randn(64, 64, device=DEVICE)
        with _registered(_KnobBackend):
            spec = (
                helion.kernel(backend="_test_knob_backend")(_add)
                .bind((x, x))
                .config_spec
            )
            _REGISTRY.pop("_test_knob_backend")
            config = spec.default_config()
            config.config[_KNOB] = 2
            self.assertTrue(spec.supports_config_key(_KNOB))
            self.assertTrue(spec.is_supported_config(config.config))
            self.assertIn(_KNOB, spec.supported_config_keys())
            spec.normalize(config)  # used to raise InvalidConfig
            self.assertEqual(config.config[_KNOB], 2)
            self.assertIn(_KNOB, spec.default_config().config)

    def test_declared_key_without_fragment_warns(self) -> None:
        # A key in plugin_config_keys() with no matching tunable_fragments()
        # entry is never searched; warned once per class at bind, not at
        # registration (tunable_fragments() may initialize the GPU runtime).
        calls = []

        class _NoFragment(backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_knob_nofrag"

            @property
            def codegen_name(self) -> str:
                return "triton"

            @classmethod
            def plugin_config_keys(cls) -> frozenset[str]:
                return frozenset({_KNOB})

            def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
                calls.append(None)
                return super().tunable_fragments()

        x = torch.randn(64, device=DEVICE)
        with _registered():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                backend_api.register_compiler_backend(_NoFragment)
            self.assertEqual(calls, [])
            self.assertEqual(caught, [])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                helion.kernel(backend="_test_knob_nofrag")(_add).bind((x, x))
                helion.kernel(backend="_test_knob_nofrag")(_add).bind((x, x))
            msgs = [
                str(w.message) for w in caught if "tunable_fragments" in str(w.message)
            ]
            self.assertEqual(len(msgs), 1)
            self.assertIn(_KNOB, msgs[0])
            self.assertIn(_NoFragment.__qualname__, msgs[0])

    def test_stray_fragment_key_fails_even_with_unrelated_plugin(self) -> None:
        # The unknown-tunables check validates against the backend's OWN key
        # space: a backend returning a foreign plugin's key must fail
        # deterministically regardless of which plugins happen to be loaded.
        class _Stray(backend_api.TritonBackend):
            @property
            def name(self) -> str:
                return "_test_knob_stray"

            @property
            def codegen_name(self) -> str:
                return "triton"

            def tunable_fragments(self) -> dict[str, backend_api.ConfigSpecFragment]:
                return {
                    **super().tunable_fragments(),
                    _KNOB: EnumFragment(choices=(1, 2)),
                }

        x = torch.randn(64, 64, device=DEVICE)
        with (
            _registered(_KnobBackend, _Stray),
            self.assertRaisesRegex(RuntimeError, "unknown tunables"),
        ):
            helion.kernel(backend="_test_knob_stray")(_add).bind((x, x))

    def test_register_tunable_name_conflict(self) -> None:
        # A user tunable may not reuse its own backend's plugin key.
        x = torch.randn(64, device=DEVICE)
        with (
            _registered(_KnobBackend),
            self.assertRaisesRegex(exc.TunableNameConflict, _KNOB),
        ):
            helion.kernel(backend="_test_knob_backend")(_user_knob_kernel).bind((x,))

    def test_user_tunable_survives_later_plugin_with_same_key(self) -> None:
        # Another backend's plugin key is not this kernel's key space: a
        # same-named user tunable keeps working after that plugin registers.
        x = torch.randn(64, device=DEVICE)
        kernel = helion.kernel(backend="triton")(_user_knob_kernel)
        spec = kernel.bind((x,)).config_spec
        config = spec.default_config()
        self.assertIn(config.config[_KNOB], (1, 2))
        with _registered(_KnobBackend):
            self.assertTrue(spec.is_supported_config(config.config))
            spec.normalize(config)
            self.assertIn(config.config[_KNOB], (1, 2))
            # Rebinding resolves against the same per-kernel key space.
            rebound = kernel.bind((x.to(torch.float16),))
            self.assertIn(_KNOB, rebound.config_spec.user_defined_tunables)


if __name__ == "__main__":
    unittest.main()
