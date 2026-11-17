from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_register_quotas import check_partition_register_quotas
from helion._testing.cute_paged_register_quotas import check_register_quotas_replay


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "quotas,form,query_len,base2,fast_math",
    # Every admitted split other than the default once. The kernels vary 32
    # and 40 query columns, the per-key and the mapped layout and ordinary and
    # base-2 reference scores instead of multiplying those axes.
    [
        ("136/80/80/48", "per_key", 4, False, True),
        ("136/72/72/48", "mapped", 5, True, False),
        ("120/96/96/48", "mapped", 4, False, False),
    ],
)
def test_register_quotas_keep_the_default_splits_results(
    quotas, form, query_len, base2, fast_math
):
    # A split that asks for more registers than the CTA's pool holds would
    # hang here; one that changed a result would differ from the default's
    # bits or fail the reference and replay checks.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_register_quotas_replay(quotas, form, query_len, base2, fast_math, DEVICE)


# Both persistent partitions, each with every other admitted split: the
# roles of a persistent CTA differ from a one-unit CTA's (a loader that plans
# first, walking compute groups, and under ``balanced`` the exchange and the
# spare warp), and each must run inside its quota with every split. One bind
# and one compile per split serve a partition's case.
@skipIfNotCUDA()
@pytest.mark.parametrize("partition", ["whole", "balanced"])
def test_register_quotas_keep_a_persistent_partitions_results(partition):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_partition_register_quotas(partition, DEVICE)
