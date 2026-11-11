from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.tensor_container_guards import container_bias_case

pytestmark = pytest.mark.cpu_only


# Each container projection must replay both a folded broadcast (size == 1)
# and a retained row index (size != 1), including a composed item/field path.
@pytest.mark.parametrize(
    "kind", ["dataclass", "namedtuple", "dict", "list", "tuple", "nested"]
)
@pytest.mark.parametrize("first_size", [1, 4])
@onlyBackends(["triton", "cute"])
def test_container_shape_guards_rebind_broadcast_transitions(kind, first_size):
    x = torch.empty(4, 8)
    kernel, original = container_bias_case(kind, x, torch.empty(first_size, 8))
    _, changed = container_bias_case(kind, x, torch.empty(5 - first_size, 8))
    kernel.reset()
    config = helion.Config(block_sizes=[4, 8])
    with cpu_render_target():
        bound = kernel.bind(original)
        bound.to_code(config)
        facts = bound._guarded_input_facts()
        assert facts
        assert any(fact(original) != fact(changed) for fact in facts)
        assert kernel.specialization_key(original) != kernel.specialization_key(changed)
        replacement = kernel.bind(changed)
        assert replacement is not bound
        replacement.to_code(config)
        assert kernel.bind(original) is bound
