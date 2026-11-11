from __future__ import annotations

import pytest
import torch

from helion._compiler.backend_registry import get_backend
from helion.autotuner import BooleanFragment
from helion.autotuner import ConfigSpec
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.config_fragment import NumThreadsFragment

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "fragment",
    [
        BooleanFragment(),
        EnumFragment((2, 16)),
        EnumFragment((2, 16, 32), search_choices=(2, 16), coverage_choices=(16,)),
    ],
)
def test_complete_structural_domains_keep_legacy_persistent_fingerprint(fragment):
    # These domains were already fully encoded by the old structural key.
    # Only incomplete integer/thread-count domains need the new cache salt.
    spec = ConfigSpec(
        backend=get_backend("triton"),
        user_defined_tunables={"factor": fragment},
        device=torch.device("cpu"),
    )
    legacy = spec.structural_fingerprint_hash()
    assert spec.cache_fingerprint_hash() == legacy
    assert spec.projected_cache_fingerprint_hash() == legacy


@pytest.mark.parametrize(
    "first,second",
    [
        (IntegerFragment(2, 16, 2), IntegerFragment(4, 8, 4)),
        (PowerOfTwoFragment(2, 16, 2), PowerOfTwoFragment(4, 8, 4)),
        (NumThreadsFragment(128), NumThreadsFragment(256)),
    ],
)
def test_declared_domains_change_only_persistent_identity(first, second):
    # Construct ConfigSpec directly: this cache contract precedes admission of
    # captured fragment objects by the compiler frontend.
    specs = [
        ConfigSpec(
            backend=get_backend("triton"),
            user_defined_tunables={"factor": fragment},
            device=torch.device("cpu"),
        )
        for fragment in (first, second)
    ]
    assert specs[0].structural_fingerprint() == specs[1].structural_fingerprint()
    assert specs[0].cache_fingerprint_hash() != specs[1].cache_fingerprint_hash()
    assert (
        specs[0].projected_cache_fingerprint_hash()
        != specs[1].projected_cache_fingerprint_hash()
    )
