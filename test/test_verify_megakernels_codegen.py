"""The pretuned speculative-decoding verify megakernels generate code on CPU,
bound with their pinned or measured configs on a modelled B200 (under a fake
process group for the tensor-parallel ones): the Llama-3.2-1B and Qwen3-8B
verify layers, the DeepSeek-V3 verify MoE on 1 GPU and at TP4 (which tensors
stream through the ring), and Muse Spark's served builds (every image of the
decoder-layer megakernel at TP4 and TP8, the LM head, the expert-TP MoE). Their
checks run on GPUs in ``test_pretuned_kernels``."""

from __future__ import annotations

import contextlib
import importlib
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.distributed.fake_pg import FakeStore

from helion._compat import gluon_api_available
from helion._testing.triton_codegen import gluon_cpu_codegen
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Generator

    import helion

if _get_backend() != "triton":
    raise unittest.SkipTest("the verify megakernels run in the triton lane")

pytestmark = pytest.mark.cpu_only

_needs_gluon = pytest.mark.skipif(
    not gluon_api_available(), reason="the kernel is a native Gluon kernel"
)

# A B200: the Muse layer megakernel's 144 persistent workers and 4 reserved SMs.
_SMS = 148
_CPU = torch.device("cpu")


class _LocalSymmetricHandle:
    """A rendezvous stand-in: every peer's buffer is the local one (nothing
    launches)."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor, self.rank = tensor, 0
        self.world_size = dist.get_world_size()
        self.buffer_ptrs = [tensor.data_ptr()] * self.world_size

    def get_remote_tensor(
        self, peer: int, size: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return self.tensor.view(dtype).view(size)


def _symmetric_empty(
    *size: int, dtype: torch.dtype, device: torch.device | None = None
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype)


@contextlib.contextmanager
def _modelled_b200() -> Generator[None, None, None]:
    """CPU tensors trace as ``cuda:0`` ones of a B200; symmetric buffers are
    plain ones."""
    device = SimpleNamespace(
        max_threads_per_multi_processor=2048,
        multi_processor_count=_SMS,
        # The Gluon ring sizes its depths by the shared memory budget.
        shared_memory_per_block_optin=232448,
        shared_memory_per_multiprocessor=233472,
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(gluon_cpu_codegen())
        stack.enter_context(
            patch("torch.cuda.get_device_properties", return_value=device)
        )
        # The driver's cluster occupancy query: every SM, in clusters of any size.
        stack.enter_context(
            patch(
                "helion._compiler.cross_loop_dsmem.get_cluster_capacity",
                new=lambda device, size: _SMS - _SMS % size,
            )
        )
        stack.enter_context(patch.object(symm_mem, "empty", _symmetric_empty))
        stack.enter_context(
            patch.object(
                symm_mem,
                "rendezvous",
                lambda tensor, group=None: _LocalSymmetricHandle(tensor),
            )
        )
        yield


@contextlib.contextmanager
def _modelled_ranks(world: int) -> Generator[dist.ProcessGroup, None, None]:
    """Rank 0 of a fake ``world``-rank group on a modelled B200."""
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=world)
    try:
        with _modelled_b200():
            yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _generate(kernel: helion.Kernel, args: tuple[object, ...]) -> str:
    return kernel.bind(args).to_triton_code(kernel.configs[0])


def _ring_tensors(code: str) -> list[str]:
    """The tensors generated code streams through the Gluon ring by TMA."""
    return sorted(re.findall(r"ring_descriptor\((\w+),", code))


# The verify layers' ring tensors: every GEMM and attention operand, the
# weights' K-partition views and the KV cache's halves view included (views
# the kernel builds, whose strides are known only with static shapes).
_VERIFY_LAYER_COMMON_RING_TENSORS = [
    "activation_chunked",
    "attention_chunked",
    "attention_split_kv_halves",
    "hidden_states",
    "o_partials",
    "o_weight_chunked",
    "partial_out",
    "post_rms_partials",
    "pre_normed",
    "pre_rms_partials",
    "qkv_weight",
    "query_halves",
    "w2_chunked",
]
_VERIFY_LAYER_RING_TENSORS = {
    "llama32_1b_decode_layer_bf16.llama32_1b_verify_layer_bf16": sorted(
        [*_VERIFY_LAYER_COMMON_RING_TENSORS, "ffn_normed", "w13_chunked", "w2_partials"]
    ),
    "qwen3_8b.qwen3_verify_layer_bf16": sorted(
        [*_VERIFY_LAYER_COMMON_RING_TENSORS, "ffn_chunked", "w13_kchunked"]
    ),
}


@_needs_gluon
@pytest.mark.parametrize("name", list(_VERIFY_LAYER_RING_TENSORS))
def test_verify_layer_generates_code(name: str) -> None:
    # Their checks are slow-tier; this keeps code generation covered.
    module = importlib.import_module(f"pretuned_kernels.megakernels.{name}")
    kernel = getattr(module, name.rpartition(".")[2])
    # Uninitialized inputs: binding and code generation need only shapes.
    with _modelled_b200():
        inputs = module._make_inputs(tokens=8, init=False, device=_CPU)
        code = _generate(kernel, module._kernel_args(inputs))
    assert _ring_tensors(code) == _VERIFY_LAYER_RING_TENSORS[name]


def _deepseek_v3_verify_args(tokens: int, intermediate: int) -> tuple[object, ...]:
    """The 1-GPU DeepSeek verify kernel's arguments, uninitialized: binding and
    code generation need only their shapes, strides and dtypes."""
    from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4 import (
        deepseek_v3_moe_nvfp4_verify as single,
    )

    shape = single.Shape(intermediate=intermediate)
    tensors = single._allocate_verify(shape, tokens, init=False, device=_CPU)
    return single._verify_args(tensors, shape)


# The tensors the DeepSeek verify megakernels stream through the Gluon ring by
# TMA: the GEMM operands.  Their configs pick TMA by access position
# (_TMA_SLOTS), so a kernel or compiler change that renumbers the accesses would
# silently move TMA to other tensors.
_DEEPSEEK_VERIFY_RING_TENSORS = [
    "activation_q_steps",
    "activation_scale_steps",
    "hidden_chunked",
    "input_q_steps",
    "input_q_steps",
    "input_scale_steps",
    "input_scale_steps",
    "router_weight_chunked",
    "shared_activation_q_steps",
    "shared_activation_scale_steps",
    "shared_w13_scale",
    "shared_w13_view",
    "shared_w2_scale",
    "shared_w2_view",
    "w13_scale",
    "w13_view",
    "w2_scale",
    "w2_view",
]
_DEEPSEEK_TP_VERIFY_RING_TENSORS = [
    "activation_q_steps",
    "activation_scale_steps",
    "hidden_chunked",
    "input_q_steps",
    "input_q_steps",
    "input_q_steps",
    "input_scale_steps",
    "input_scale_steps",
    "input_scale_steps",
    "router_weight_chunked",
    "shared_activation_q_steps",
    "shared_activation_scale_steps",
    "shared_w13_scale",
    "shared_w13_view",
    "shared_w2_scale",
    "shared_w2_view",
    "w13_scale",
    "w13_view",
    "w13_view",
    "w2_scale",
    "w2_view",
]


@_needs_gluon
@pytest.mark.parametrize("world", [1, 4])
def test_deepseek_v3_verify_ring_operands(world: int) -> None:
    from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4 import (
        deepseek_v3_moe_nvfp4_verify as single,
    )
    from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp import (
        deepseek_v3_moe_nvfp4_tp_verify as tp,
    )

    def ring_tensors(kernel: helion.Kernel, args: tuple[object, ...]) -> list[str]:
        return _ring_tensors(_generate(kernel, args))

    # The TP4 shard's intermediate size for both: the access order does not
    # depend on it, and the weights take a quarter of the memory.
    tokens, intermediate = 8, 2048 // tp.WORLD_SIZE
    args = _deepseek_v3_verify_args(tokens, intermediate)
    if world == 1:
        with _modelled_b200():
            kernel = single.deepseek_v3_moe_nvfp4_verify
            assert ring_tensors(kernel, args) == _DEEPSEEK_VERIFY_RING_TENSORS
        return
    with _modelled_ranks(tp.WORLD_SIZE) as group:
        hidden = single.Shape().hidden
        args = (
            *args,
            torch.empty(tokens, hidden, dtype=torch.bfloat16),
            torch.empty(hidden, dtype=torch.bfloat16),
            torch.empty(tokens, hidden, dtype=torch.bfloat16),
            group.group_name,
        )
        kernel = tp.deepseek_v3_moe_nvfp4_tp_verify
        assert ring_tensors(kernel, args) == _DEEPSEEK_TP_VERIFY_RING_TENSORS


# The served images: the checkpoint's layer kinds (sliding-window or global
# attention, NVFP4 or bf16 experts, the last layer's final norm) and the model's
# prelude. Each (world, image) pair is its own build: TP4 and TP8 run different
# kernels, and each image has its own measured config. The decoder-layer images
# bind in 25-34 s each, so only the prelude (~12 s) runs on pull requests.
_LAYER_IMAGES = {
    "sliding_window": {},
    "global": {"global_attn": True},
    "bf16_experts": {"expert_bf16": True},
    "final_norm": {"global_attn": True, "expert_bf16": True, "final_norm": True},
    "entry": {"entry": True},
}


def _layer_ring_tensors(world: int, image: str) -> list[str]:
    """The tensors an image streams through the ring by TMA: the GEMM operands,
    the expert banks and the MoE's activations (with their block scales on
    NVFP4 experts; TP4's router splits its weight into two planes)."""
    if image == "entry":  # the prelude's QKV GEMM
        return ["act3", "wq4"]
    tensors = ["act", "act", "act3", "out4", "weights2", "weights_g", "wo", "wp"]
    tensors += ["wq4", "wr_hi", "wu4", "x3", "yu3", "z"]
    if "expert_bf16" not in _LAYER_IMAGES[image]:
        tensors += ["scales", "xs3", "z_scale"]
    if world == 4:
        tensors.append("wr_lo")
    if "final_norm" in _LAYER_IMAGES[image]:  # no next-layer QKV
        tensors.remove("act3")
        tensors.remove("wq4")
    return sorted(tensors)


@_needs_gluon
@pytest.mark.parametrize(
    "image",
    [
        pytest.param(name, marks=() if name == "entry" else pytest.mark.slow)
        for name in _LAYER_IMAGES
    ],
)
@pytest.mark.parametrize("world", [4, 8])
def test_layer_megakernel_generates_code(world: int, image: str) -> None:
    from pretuned_kernels.megakernels.muse_spark import layer

    with _modelled_ranks(world) as group:
        case = layer._make_case(group, _CPU, init=False, **_LAYER_IMAGES[image])
        attention, qkv = layer._attention_inputs(
            layer.LayerDims.of(world), 640, 0, torch.empty(1 << 17, 64)
        )
        # The entry image reads layer 0's bf16 embedding rows.
        dtype = torch.bfloat16 if image == "entry" else torch.float32
        stream = torch.empty(layer.TOKENS, layer.HIDDEN, dtype=dtype)
        kernel = case.host._kernel
        code = _generate(kernel, layer._launch_args(case, attention, qkv, stream))
    assert _ring_tensors(code) == _layer_ring_tensors(world, image)


@_needs_gluon
@pytest.mark.parametrize("world", [4, 8])
def test_lm_head_generates_code(world: int) -> None:
    from pretuned_kernels.megakernels.muse_spark import lm_head

    tokens, shard = 8, 4 * lm_head.VOCAB_SHARD // world
    blocks = -(-shard // lm_head.G_BLOCK)
    f32, i32 = torch.float32, torch.int32
    with _modelled_ranks(world) as group:
        args = (
            torch.empty(tokens, lm_head.HIDDEN, dtype=torch.bfloat16),
            torch.empty(shard, lm_head.HIDDEN, dtype=torch.bfloat16),
            torch.empty(tokens, blocks * lm_head.G_BLOCK, dtype=f32),
            torch.empty(tokens, world, shard, dtype=f32),
            torch.empty(tokens, blocks, dtype=f32),
            torch.empty(tokens, blocks, dtype=i32),
            torch.empty(tokens, dtype=f32),
            torch.empty(tokens, dtype=i32),
            torch.ones(1, dtype=i32),
            torch.empty(tokens, dtype=i32),
            group.group_name,
            0,
            world,
            lm_head.LOGIT_SCALE,
            lm_head.LOGIT_CAP,
        )
        kernel = lm_head.muse_spark_lm_head
        code = _generate(kernel, args)
    # Pointer indexing throughout: no TMA ring and no tensor descriptors.
    assert _ring_tensors(code) == []
    assert not re.findall(r"\b\w+_desc\b", code)
