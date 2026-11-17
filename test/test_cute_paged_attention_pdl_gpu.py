"""A delayed producer exercises the paged native owner's PDL dependency."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import CONTROL_DELAY
from helion._testing.cute_pdl import attribute_only_control
from helion._testing.cute_pdl import produce


@skipIfNotCUDA()
@skipIfRefEager("PDL requires native launch attributes")
@pytest.mark.parametrize(
    "per_key,paired", [(False, False), (True, False), (True, True)]
)
def test_paged_attention_waits_for_delayed_lengths(per_key, paired):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the paged pipeline uses SM100 tensor memory")
    rows = 8 if per_key else 64
    # A paired launch has 300 logical items but only 150 physical CTAs:
    # on a 148-SM target its last-wave release starts at CTA 2, not 152.
    requests = 150 if paired else 2
    args = list(paged_attention_args(device=DEVICE, rows=rows, requests=requests))
    args[-1] = -1
    args.extend((None, per_key))
    q, k, v, sk, sv, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    q.copy_(((torch.arange(q.numel(), device=DEVICE) % 19 - 9) / 32).reshape(q.shape))
    k.copy_((torch.arange(k.numel(), device=DEVICE) % 15 - 7).reshape(k.shape))
    v.copy_((torch.arange(v.numel(), device=DEVICE) % 17 - 8).reshape(v.shape))
    sk.fill_(126)
    sv.fill_(126)
    pages.copy_(
        (torch.arange(pages.numel(), device=DEVICE) * 7 % k.size(0)).reshape(
            pages.shape
        )
    )
    sources = [
        torch.tensor(values * (requests // 2), dtype=torch.int32, device=DEVICE)
        for values in ([0, 257], [129, 0])
    ]
    lengths.copy_(sources[0])
    bound = paged_online_attention.bind(tuple(args))
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[rows],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention_paired" if paired else "attention",
                cute_min_blocks_per_mp=1,
                cute_pdl=True,
            )
        )
    assert config.get("cute_pdl") is True
    assert config.get("cute_mma_dag_pass_warps") == (
        "attention_paired" if paired else "attention"
    )
    source = bound.to_code(config)
    assert "._helion_cute_use_pdl = True" in source
    compiled = bound.compile_config(config)
    compiled(*args)

    # Remove only the entry wait, retaining the real launch attribute. The
    # resulting intentionally unsafe control must consume stale scalar lengths.
    control = attribute_only_control(bound, source)
    control(*args)

    def produce_lengths(values):
        produce(values, lengths, delay=CONTROL_DELAY, grid=1, block=32)

    with bound.env:
        plain_config = bound.config_spec.normalized_config(
            helion.Config.from_dict({**config.config, "cute_pdl": False})
        )
    plain = bound.compile_config(plain_config)
    # The existing arithmetic tests validate the native body. Here the
    # ordinary stream-ordered version isolates the dependent-launch protocol.
    references = []
    for values in sources:
        lengths.copy_(values)
        references.append(plain(*args))
    # Warm the producer and both consumers before measuring the dependency,
    # so first-time host compilation cannot hide the overlap.
    produce_lengths(sources[0])
    compiled(*args)
    torch.cuda.synchronize()
    for iteration in range(4):
        index = iteration % 2
        lengths.copy_(sources[1 - index])
        produce_lengths(sources[index])
        actual = compiled(*args)
        for value, reference in zip(actual, references[index], strict=True):
            torch.testing.assert_close(value, reference, rtol=0.002, atol=0.002)

    stale = 0
    for iteration in range(4):
        index = iteration % 2
        lengths.copy_(sources[1 - index])
        produce_lengths(sources[index])
        actual = control(*args)
        # Zero-trip and nonzero requests swap, so finite LSE distinguishes
        # stale lengths without conflating attention rounding differences.
        stale += int(
            (torch.isfinite(actual[1]) != torch.isfinite(references[index][1])).any()
        )
    assert stale > 0

    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            produce_lengths(sources[0])
            output0 = compiled(*args)
            produce_lengths(sources[1])
            output1 = compiled(*args)
        for _ in range(2):
            graph.replay()
            for actual, reference in zip((output0, output1), references, strict=True):
                for value, expected in zip(actual, reference, strict=True):
                    torch.testing.assert_close(value, expected, rtol=0.002, atol=0.002)
