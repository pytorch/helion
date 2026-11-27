"""The pretuned speculative-decoding verify megakernels generate code on CPU,
bound with their pinned or measured configs on a modelled B200 (under a fake
process group for the tensor-parallel ones): the Llama-3.2-1B and Qwen3-8B
verify layers, the Kimi-K3 KDA verify layer, the DeepSeek-V3 verify MoE on 1
GPU and at TP4 (which tensors stream through the ring), Muse Spark's served
builds (every image of the decoder-layer megakernel at TP4 and TP8, and the LM
head), and every Inkling-2-Small kernel build its vLLM driver makes (over a
synthetic model). The GPT-OSS decode and verify layers' pinned ring configs are
checked against ``ring_config_fields`` of the bound kernels, the Qwen3.5 TP4
kernels' pinned pointer slots and enabled DSMEM transports against theirs, the
GLM-5.3 TP4 verify and MTP draft kernels' pinned configs at every served window,
and the GLM-5.3 Flash TP4 kernels' ring tensors (their configs derived at bind
time). Their checks run on GPUs in ``test_pretuned_kernels``."""

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
from helion._testing import skipIfRefEagerMark
from helion._testing.triton_codegen import SM100_DEVICE_PROPERTIES
from helion._testing.triton_codegen import gluon_cpu_codegen
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    import helion

if _get_backend() != "triton":
    raise unittest.SkipTest("the verify megakernels run in the triton lane")

pytestmark = [
    pytest.mark.cpu_only,
    skipIfRefEagerMark("inspects generated code, which ref mode does not build"),
]

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
    return torch.empty(*size, dtype=dtype)


@contextlib.contextmanager
def _modelled_b200() -> Generator[None, None, None]:
    """CPU tensors trace as ``cuda:0`` ones of a B200; symmetric buffers are
    plain ones."""
    # ``gluon_cpu_codegen``'s sm100 properties, with the B200's SM count.
    device = SimpleNamespace(**SM100_DEVICE_PROPERTIES, multi_processor_count=_SMS)
    with contextlib.ExitStack() as stack:
        stack.enter_context(gluon_cpu_codegen())
        stack.enter_context(
            patch("torch.cuda.get_device_properties", return_value=device)
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
@pytest.mark.slow  # binds in 20-23 s per layer
@pytest.mark.parametrize("name", list(_VERIFY_LAYER_RING_TENSORS))
def test_verify_layer_generates_code(name: str) -> None:
    module = importlib.import_module(f"pretuned_kernels.megakernels.{name}")
    kernel = getattr(module, name.rpartition(".")[2])
    # Uninitialized inputs: binding and code generation need only shapes.
    with _modelled_b200():
        inputs = module._make_inputs(tokens=8, init=False, device=_CPU)
        code = _generate(kernel, module._kernel_args(inputs))
    assert _ring_tensors(code) == _VERIFY_LAYER_RING_TENSORS[name]


@_needs_gluon
def test_kda_mtp_verify_generates_code() -> None:
    # Its check is slow-tier; this keeps code generation covered.
    from pretuned_kernels.megakernels.kda_mtp import _reference as ref
    from pretuned_kernels.megakernels.kda_mtp import kda_mtp_verify as kmv

    with _modelled_b200():
        inputs = ref.make_inputs(1, kmv.H, kmv.T, device=_CPU)
        out = torch.empty(kmv.T, ref.HIDDEN, dtype=torch.bfloat16)
        bound = kmv.kda_mtp.bind(kmv.kernel_args(inputs, out))
        code = bound.to_triton_code(kmv.mk_config(bound))
    # The projections' operands: one ring load each.
    assert _ring_tensors(code) == sorted(kmv.RING_LOADS)


def _gpt_oss_served() -> dict[str, torch.Tensor]:
    """The GPT-OSS layers' served weights (``_common.make_layer``'s layout),
    uninitialized: binding needs only their shapes, strides and dtypes."""
    from pretuned_kernels.megakernels.gpt_oss_120b import _common as c

    return c.make_layer(0, _CPU, init=False)[1]


@_needs_gluon
@pytest.mark.parametrize(
    ("name", "case"),
    [("gpt_oss_decode_layer", (1000, -1)), ("gpt_oss_verify_layer", (400, -1, 0.0))],
)
def test_gpt_oss_ring_config(name: str, case: tuple[object, ...]) -> None:
    # The layers pin their ring indexing by access position and their warp
    # specialization by loop position: a kernel or compiler change that moves
    # either must update the pinned config.
    from pretuned_kernels.megakernels._ring_config import ring_config_fields
    from pretuned_kernels.megakernels.gpt_oss_120b import _common as c

    module = importlib.import_module(
        f"pretuned_kernels.megakernels.gpt_oss_120b.{name}"
    )
    with _modelled_b200():
        args = module._make_case(_gpt_oss_served(), c.rope_cos_sin(_CPU), *case, _CPU)
        bound = getattr(module, name).bind(args[-1])
        fields = ring_config_fields(bound, c.RING_LOADS, warp_specialize="k_loops")
    assert fields == {key: module.CONFIG[key] for key in fields}


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
        args, _ = case.host._args(
            qkv=qkv, stream=stream, weights=case.weights, attention=attention
        )
        code = _generate(case.host._kernel, args)
    assert _ring_tensors(code) == _layer_ring_tensors(world, image)


@_needs_gluon
@pytest.mark.parametrize("world", [4, 8])
def test_lm_head_generates_code(world: int) -> None:
    from pretuned_kernels.megakernels.muse_spark import lm_head

    tokens, bf16 = 8, torch.bfloat16
    kernel = lm_head.muse_spark_lm_head
    # launch()'s arguments, recorded instead of launched.
    with (
        _modelled_ranks(world) as group,
        patch.object(lm_head, "muse_spark_lm_head") as call,
    ):
        lm_head.launch(
            torch.empty(tokens, lm_head.HIDDEN, dtype=bf16),
            torch.empty(lm_head.VOCAB // world, lm_head.HIDDEN, dtype=bf16),
            lm_head.make_workspaces(tokens, group, _CPU),
            torch.empty(tokens, lm_head.VOCAB),
            group,
        )
        code = _generate(kernel, call.call_args.args)
    # Pointer indexing throughout: no TMA ring and no tensor descriptors.
    assert _ring_tensors(code) == []
    assert not re.findall(r"\b\w+_desc\b", code)


# The Inkling-2-Small TP4 verify megakernels: every kernel build of the served
# driver (vllm_helion_verify.py), with its arguments, config and TMA lists, by
# its key: (kernel, the next layer's attention inputs) or the attention kernel.
# Each is its own build (the served forward's nine, then INK_HELION_LAYERS=0's
# three: the local attention kernel and the MoE kernels before a global layer).
# The layer kernels take 30-50 s each, the others 14-30 s (a parallel CPU lane):
# only the MXFP8 MoE runs on pull requests.
_INKLING_BUILDS = [
    ("layer_block", 1),
    ("layer", 1),
    ("layer", 3),
    "global",
    ("moe", 1),
    ("layer_mx", 3),
    ("moe_mx", 1),
    "block",
    ("layer_mx", 1),
    "local",
    ("moe", 3),
    ("moe_mx", 3),
]
# The tensors each kernel streams through the ring by TMA: the driver names
# them (TMA_*), so a renamed access would fall back to pointers unnoticed.
_INKLING_ATTN = ["o_in", "w_qkvr", "wo", "x3"]
_INKLING_MOE = ["gate_w", "w13", "w13_s", "w2f", "w2sf", "xq", "xs", "zqf", "zsf"]
_INKLING_MOE_MX = ["gate_w", "w13", "w13_s", "w2", "w2_s", "xq", "xsf", "zq", "zsf"]
# The local layer kernels' attention half (its split qkvr weight and K-split
# input views), and x, the MoE half's input, as the router's operand. The NVFP4
# layers' quantize root (Q) loads x through a second ring descriptor, the MXFP8
# layer's through the descriptor of B's store of x (x_desc, not a ring one).
_INKLING_LAYER_ATTN = ["a_o_in", "a_w_g", "a_w_kv", "a_w_qr", "a_wo"]
_INKLING_LAYER_ATTN += ["a_x6", "a_x6", "a_x8", "x"]
_INKLING_RING_TENSORS = {
    "local": _INKLING_ATTN,
    "block": _INKLING_ATTN,
    "global": ["w_qkvr", "wo"],
    "moe": _INKLING_MOE,
    "moe_mx": _INKLING_MOE_MX,
    "layer": sorted([*_INKLING_LAYER_ATTN, "x", *_INKLING_MOE]),
    "layer_block": sorted(
        ["a_o_in", "a_w_qkvr", "a_wo", "a_x3", "x", "x", *_INKLING_MOE]
    ),
    "layer_mx": sorted([*_INKLING_LAYER_ATTN, *_INKLING_MOE_MX]),
}


@_needs_gluon
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            build,
            marks=() if build == ("moe_mx", 1) else pytest.mark.slow,
            id=build if isinstance(build, str) else f"{build[0]}_a{build[1]}",
        )
        for build in _INKLING_BUILDS
    ],
)
def test_inkling_generates_code(build: str | tuple[str, int]) -> None:
    from pretuned_kernels.megakernels.inkling2_small._config import variant_kernel
    from pretuned_kernels.megakernels.inkling2_small._synthetic import served_builds

    with _modelled_ranks(4) as group:
        builds = served_builds(group, _CPU)
        assert list(builds) == _INKLING_BUILDS
        kern_fn, args, *rest = builds[build]
        code = variant_kernel(kern_fn, args, *rest).bind(args).to_triton_code()
    name = build if isinstance(build, str) else build[0]
    assert _ring_tensors(code) == _INKLING_RING_TENSORS[name]


# The Qwen3.5 TP4 kernels (``_qwen3_5_tp4``): package and the tensors whose
# loads their configs carry over DSMEM. Each pins its pointer accesses by
# indexing slot (``_POINTER_OPS``: each slot's load or store and tensor; the GDN
# mixer has none) and its DSMEM transports by candidate position, as the tensor
# and root of each enabled candidate's load (a tensor can have candidates in
# several roots: token_experts in the MoE's W13 and W2 roots), so a kernel or
# compiler change that moves either must update its config. They bind in 7-12
# s, the merged layer (whose MoE half's slots are the standalone MoE's) in 17 s:
# it runs in the slow tier.
_QWEN3_5_MOE_DSMEM = [("token_experts", 6), ("norm_partial", 11), ("norm_input", 11)]
_QWEN3_5_KERNELS = {
    "qwen3_5_moe_tp4": ("qwen3_5_moe_tp4", _QWEN3_5_MOE_DSMEM),
    "qwen3_5_gdn_mixer_tp4": ("qwen3_5_gdn_tp4", [("o_part4", 6)]),
    "qwen3_5_attn_mixer_tp4": ("qwen3_5_attn_tp4", []),
    "qwen3_5_layer_tp4": ("qwen3_5_moe_tp4", _QWEN3_5_MOE_DSMEM),
}


def _qwen3_5_args(name: str, group_name: str) -> tuple[object, ...]:
    """Kernel ``name``'s arguments from its setup, the attention mixer's
    SGLang-made weights and rope tables and the MoE's weights uninitialized
    (binding needs only their shapes, strides and dtypes)."""
    from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
    from pretuned_kernels.megakernels.qwen3_5_attn_tp4 import _setup as attn
    from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as gdn
    from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.qwen3_5_gdn_mixer_tp4 import (
        qwen3_5_gdn_mixer_tp4,
    )
    from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup as moe
    from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_layer_tp4 import (
        merged_args,
    )
    from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import (
        qwen3_5_moe_tp4,
    )

    bf16, e4m3 = torch.bfloat16, torch.float8_e4m3fn
    partials = symmetric_partials(gdn.T, gdn.HIDDEN, group_name, _CPU)
    if name == "qwen3_5_attn_mixer_tp4":
        rows = attn.QG_SIZE + 2 * attn.KV_SIZE
        layer = {
            "qkv_stored": torch.empty(rows, attn.HIDDEN, dtype=e4m3).t(),
            "o_proj_stored": torch.empty(attn.HIDDEN, attn.Q_SIZE, dtype=e4m3).t(),
            "q_norm_weight": torch.empty(attn.HEAD_DIM, dtype=bf16),
            "k_norm_weight": torch.empty(attn.HEAD_DIM, dtype=bf16),
            "norm_weight": torch.empty(attn.HIDDEN, dtype=bf16),
            "qkv_scales": torch.empty(2),
            "o_scales": torch.empty(2),
            "kv_scales": torch.empty(4),
        }
        rope = (
            torch.empty(attn.MAX_POSITION, attn.ROTARY_DIM),
            torch.empty(attn.ROTARY_DIM // 2, dtype=torch.int64),
        )
        state, x = attn.make_state(0, 100, _CPU), attn.make_inputs(0, _CPU)
        return attn.kernel_args(layer, state, x, rope, partials[0], group_name)
    layer, state = gdn.make_layer(0, _CPU), gdn.make_state(0, device=_CPU)
    x = gdn.make_inputs(0, _CPU)
    mixer = gdn.kernel_args(layer, state, x, partials[0], group_name)
    if name == "qwen3_5_gdn_mixer_tp4":
        return mixer
    E, H, N, SHI = (
        moe.SHAPE.num_experts,
        moe.SHAPE.hidden,
        moe.SHAPE.intermediate,
        moe.SHAPE.shared_intermediate,
    )
    u8 = torch.uint8
    weights = {
        "router": torch.empty(E, H, dtype=bf16),
        "w13": torch.empty(E, 2 * N, H // 2, dtype=u8),
        "w13s": torch.empty(E, 2 * N, H // 16, dtype=e4m3),
        "w2": torch.empty(E, H, N // 2, dtype=u8),
        "w2s": torch.empty(E, H, N // 16, dtype=e4m3),
        "alpha1": torch.empty(E),
        "alpha2": torch.empty(E),
        "input_global": torch.empty(1),
        "act_global": torch.empty(1),
        "sw13": torch.empty(2 * SHI, H, dtype=e4m3),
        "sw2": torch.empty(H, SHI, dtype=e4m3),
        "shared_scales": torch.empty(4),
        "gate_w": torch.empty(H, dtype=bf16),
    }
    moe_args = (
        x["hidden"],
        *moe.kernel_weight_args(weights),
        x["residual"],
        torch.empty(gdn.HIDDEN, dtype=bf16),
        partials[1],
        group_name,
    )
    if name == "qwen3_5_moe_tp4":
        return moe_args
    moe_named = dict(zip(qwen3_5_moe_tp4.signature.parameters, moe_args, strict=True))
    mixer_named = dict(
        zip(qwen3_5_gdn_mixer_tp4.signature.parameters, mixer, strict=True)
    )
    return tuple(merged_args(moe_named, mixer_named))


@_needs_gluon
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name, marks=pytest.mark.slow if name == "qwen3_5_layer_tp4" else ()
        )
        for name in _QWEN3_5_KERNELS
    ],
)
def test_qwen3_5_pinned_slots(name: str) -> None:
    package, dsmem = _QWEN3_5_KERNELS[name]
    module = importlib.import_module(f"pretuned_kernels.megakernels.{package}.{name}")
    kernel = getattr(module, name)
    with _modelled_ranks(4) as group:
        bound = kernel.bind(_qwen3_5_args(name, group.group_name))
    facts = {
        f.indexing_index: f"{f.kind} {f.tensor_name}"
        for f in bound.config_spec.memory_op_facts
    }
    pointer_ops = {} if name == "qwen3_5_gdn_mixer_tp4" else module._POINTER_OPS
    assert {slot: facts[slot] for slot in pointer_ops} == pointer_ops
    graph = bound.host_function.device_ir.tile_dependency_graph
    candidates = [(load.tensor_name, load.root) for _, load in graph.dsmem_candidates]
    flags = kernel.configs[0].config.get("cross_loop_dsmem", [False] * len(candidates))
    assert [n for n, on in zip(candidates, flags, strict=True) if on] == dsmem


# The GLM-5.3 TP4 kernels (``glm5_3_tp4/``) bound at their served windows: the
# target's verify window (EAGLE 5/1/6), the MTP draft's decode step (one row:
# the attention, eh and LM head kernels) and its extend (6 rows). The MoE pins
# its pointer accesses by indexing slot (``_POINTER_OPS``) and its DSMEM
# transports (tensor and root) like the Qwen3.5 MoE; the others pin their ring
# indexing and warp specialization (``ring_config_fields`` of ``RING_LOADS``).
# One table serves both windows: a window that binds fewer memory ops (the
# attention output's longer splits past 4 rows) leaves the table's trailing
# entries, pointer accesses, unread.
_GLM5_3_HIDDEN = 6144
_GLM5_3_KERNELS = {
    "glm5_3_moe_tp4": ((6,), [("norm_partial", 10), ("norm_input", 10)]),
    "glm5_3_attn_front_tp4": ((6, 1), []),
    "glm5_3_attn_out_tp4": ((6, 1), []),
    "glm5_3_indexer_tp4": ((6,), []),
    "glm5_3_draft_eh_tp4": ((1, 6), []),
    "glm5_3_lm_head_tp4": ((1,), []),
}


def _glm5_3_args(name: str, group_name: str, tokens: int) -> tuple[object, ...]:
    """Kernel ``name``'s arguments for a ``tokens``-row window, uninitialized:
    binding needs only shapes, strides and dtypes."""
    from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
    from pretuned_kernels.megakernels.glm5_3_tp4 import _setup
    from pretuned_kernels.megakernels.glm5_3_tp4 import glm5_3_lm_head_tp4 as lm_head
    from pretuned_kernels.megakernels.glm5_3_tp4.glm5_3_attn_out_tp4 import SCALE_LOG2

    bf16, e4m3, H = torch.bfloat16, torch.float8_e4m3fn, _GLM5_3_HIDDEN

    def empty(*size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype)

    def partial(cols: int) -> torch.Tensor:
        return symmetric_partials(tokens, cols, group_name, _CPU)[0]

    a, i = _setup.ATTN_SHAPE, _setup.INDEXER_SHAPE
    if name == "glm5_3_moe_tp4":
        m = _setup.MOE_SHAPE
        E, N, SHI = m.num_experts, m.intermediate, m.shared_intermediate
        weights = {
            "router": empty(E, H, dtype=bf16),
            "bias": empty(E),
            "w13": empty(E, 2 * N, H // 2, dtype=torch.uint8),
            "w13s": empty(E, 2 * N, H // 16, dtype=e4m3),
            "w2": empty(E, H, N // 2, dtype=torch.uint8),
            "w2s": empty(E, H, N // 16, dtype=e4m3),
            "alpha1": empty(E),
            "alpha2": empty(E),
            "input_global": empty(1),
            "act_global": empty(1),
            "sw13": empty(2 * SHI, H, dtype=bf16),
            "sw2": empty(H, SHI, dtype=bf16),
        }
        return (
            empty(tokens, H, dtype=bf16),
            *_setup.moe_weight_args(weights),
            empty(tokens, H, dtype=bf16),
            empty(H, dtype=bf16),
            partial(H),
            group_name,
        )
    if name == "glm5_3_attn_front_tp4":
        return (
            empty(tokens, H, dtype=bf16),
            empty(a.q_lora // 4, H, dtype=bf16),
            empty(a.kv_lora + a.rope, H, dtype=bf16),
            empty(a.q_lora, dtype=bf16),
            empty(a.kv_lora, dtype=bf16),
            empty(a.heads * (a.nope + a.rope), a.q_lora, dtype=bf16),
            empty(a.heads, a.kv_lora, a.nope, dtype=bf16),
            empty(32768, a.rope),
            empty(tokens, dtype=torch.int64),
            empty(tokens, dtype=torch.int64),
            empty(4096, a.kv_lora + a.rope, dtype=e4m3),
            partial(a.q_lora // 4),
            group_name,
        )
    if name == "glm5_3_attn_out_tp4":
        return (
            empty(tokens, a.heads, a.kv_lora + a.rope, dtype=e4m3),
            empty(4096, a.kv_lora + a.rope, dtype=e4m3),
            empty(tokens, a.top_k, dtype=torch.int32),
            empty(a.heads, a.v_head, a.kv_lora, dtype=bf16),
            empty(H, a.heads * a.v_head, dtype=bf16),
            SCALE_LOG2,
            empty(tokens, H, dtype=bf16),
            empty(H, dtype=bf16),
            partial(H),
            group_name,
        )
    if name == "glm5_3_indexer_tp4":
        # A served CUDA graph's page table (the max context's pages).
        return (
            empty(tokens, H, dtype=bf16),
            empty(tokens, i.q_lora, dtype=bf16),
            empty(192, H, dtype=bf16),
            empty(i.heads * i.head_dim, i.q_lora, dtype=bf16),
            empty(i.head_dim),
            empty(i.head_dim),
            empty(32768, i.rope),
            empty(tokens, dtype=torch.int64),
            *_setup.cache_views(empty(4096, 132 * i.page, dtype=torch.uint8)),
            empty(3170, dtype=torch.int32),
            empty(tokens, dtype=torch.int32),
            _setup.INDEXER_WEIGHT_SCALE,
        )
    if name == "glm5_3_draft_eh_tp4":
        return (
            empty(tokens, dtype=torch.int64),
            empty(_setup.VOCAB, H, dtype=bf16),
            empty(tokens, H, dtype=bf16),
            empty(H, dtype=bf16),
            empty(H, dtype=bf16),
            empty(H // 4, 2 * H, dtype=bf16),
            empty(H, dtype=bf16),
            partial(H // 4),
            group_name,
        )
    ws = lm_head.make_workspaces(tokens, group_name, 4, device=_CPU)
    return (
        empty(tokens, H, dtype=bf16),
        empty(_setup.VOCAB // 4, H, dtype=bf16),
        ws.shard,
        empty(tokens, 4, _setup.VOCAB // 4),
        ws.block_max,
        ws.block_arg,
        ws.rank_max,
        ws.rank_arg,
        ws.need_logits,
        ws.greedy_ids,
        group_name,
        0,
        4,
    )


@_needs_gluon
@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        (name, tokens)
        for name, (windows, _) in _GLM5_3_KERNELS.items()
        for tokens in windows
    ],
)
def test_glm5_3_pinned_config(name: str, tokens: int) -> None:
    from pretuned_kernels.megakernels._ring_config import ring_config_fields

    module = importlib.import_module(f"pretuned_kernels.megakernels.glm5_3_tp4.{name}")
    kernel = getattr(module, name)
    config = kernel.configs[0].config
    with _modelled_ranks(4) as group:
        bound = kernel.bind(_glm5_3_args(name, group.group_name, tokens))
    if hasattr(module, "_POINTER_OPS"):
        facts = {
            f.indexing_index: f"{f.kind} {f.tensor_name}"
            for f in bound.config_spec.memory_op_facts
        }
        assert {
            slot: facts[slot] for slot in module._POINTER_OPS
        } == module._POINTER_OPS
        assert len(facts) == module._MEMORY_OPS
    else:
        for key, derived in ring_config_fields(bound, module.RING_LOADS).items():
            pinned = config[key]
            trailing = len(pinned) - len(derived) if key == "indexing" else 0
            assert pinned == derived + ["pointer"] * trailing, key
    graph = bound.host_function.device_ir.tile_dependency_graph
    candidates = [(load.tensor_name, load.root) for _, load in graph.dsmem_candidates]
    flags = config.get("cross_loop_dsmem", [False] * len(candidates))
    dsmem = _GLM5_3_KERNELS[name][1]
    assert [n for n, on in zip(candidates, flags, strict=True) if on] == dsmem


# The GLM-5.3 Flash TP4 kernels (``glm5_3_flash_tp4/``) bound at their served
# windows (the target's verify window, the MTP draft's decode step and, for the
# eh, its extend), with final and raw mHC combs where the kernel folds the layer
# boundary. Their configs follow from the bound kernel (``mk_config``:
# ``ring_config_fields`` of ``RING_LOADS``): each ring tensor must still be a
# load the kernel binds, or its stream would fall to pointer accesses unnoticed.
_GLM5_3_FLASH_KERNELS = {
    "kda_tp4": ("glmf_kda_tp4", ((6, False), (6, True))),
    "moe_tp4": ("glmf_moe_tp4", ((6, False), (6, True))),
    "dense_tp4": ("glmf_dense_tp4", ((6, False), (6, True))),
    "mla_frontidx_tp4": ("glmf_mla_frontidx_tp4", ((6, False),)),
    "mla_out_tp4": ("glmf_mla_out_tp4", ((6, False), (6, True))),
    "mla_front_tp4": ("glmf_mla_front_tp4", ((1, False),)),
    "mtp_eh_tp4": ("glmf_mtp_eh_tp4", ((1, False), (6, False))),
    "mtp_attn_out_tp4": ("glmf_mtp_attn_out_tp4", ((1, False),)),
    "mtp_moe_tp4": ("glmf_mtp_moe_tp4", ((1, False),)),
    "mtp_lm_head_tp4": ("glmf_mtp_lm_head_tp4", ((1, False),)),
}


def _glm5_3_flash_args(
    module: ModuleType, group_name: str, tokens: int, raw: bool
) -> tuple[object, ...]:
    """The module's ``kernel_args`` for a ``tokens``-row window over
    uninitialized operands: binding needs only shapes, strides and dtypes."""
    from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
    from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup

    bf16, e4m3, u8 = torch.bfloat16, torch.float8_e4m3fn, torch.uint8
    i32, i64, H, T = torch.int32, torch.int64, _setup.HIDDEN, tokens

    def empty(*size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype)

    def partial() -> torch.Tensor:
        shape, dtype = module.symmetric_partial_spec(T)
        return symmetric_buffer(shape, dtype, group_name, _CPU)

    mhc = (
        empty(T, 4, H, dtype=bf16),
        empty(T, 4),
        empty(T, 4, 4),
        empty(32, 4, H, dtype=bf16),
        empty(3),
        empty(24),
        empty(H, dtype=bf16),
    )
    out = (empty(T, H, dtype=bf16), empty(T, 4, H, dtype=bf16), empty(T, 4))
    out = (*out, empty(T, 4, 4))
    name = module.__name__.rsplit(".", 1)[1]
    if name == "kda_tp4":
        pw = module.prepared_weights(
            empty(6416, H, dtype=bf16),
            empty(2, 2048, 128, dtype=bf16),
            empty(6144, 4),
            empty(128, dtype=bf16),
            empty(H, 2048, dtype=bf16),
        )
        state = (empty(8, 3, 6144, dtype=bf16), empty(8, 16, 128, 128))
        return module.kernel_args(
            empty(T, H, dtype=bf16), pw, empty(16), empty(2048), *state,
            empty(1, dtype=i32), empty(3, T, 16, 128, 128),
            empty(3, T, 3, 6144, dtype=bf16), empty(1, dtype=i32), partial(), mhc, out,
            group_name, raw, raw,
        )  # fmt: skip
    if name == "moe_tp4":
        E = _setup.MOE_SHAPE.num_experts
        pw = {
            "router": empty(E, 16, 256, dtype=bf16),
            "bias": empty(E),
            "w13": empty(E, 1024, 2048, dtype=u8),
            "w13s": empty(E, 64, 16, 256, dtype=e4m3),
            "w2": empty(E, H, 256, dtype=u8),
            "w2s": empty(E, 32, 16, 256, dtype=e4m3),
            "alpha1": empty(E),
            "alpha2": empty(E),
            "clamp1": empty(E),
            "sw13": empty(1024, 2048, dtype=u8),
            "sw13s": empty(64, 16, 256, dtype=e4m3),
            "sw2": empty(H, 256, dtype=u8),
            "sw2s": empty(32, 16, 256, dtype=e4m3),
            "scales": empty(6),
        }
        x = empty(T, H, dtype=bf16)
        return module.kernel_args(x, pw, partial(), mhc, out, group_name, raw, raw)
    if name == "dense_tp4":
        pw = {
            "w13": empty(6144, 8, 256, dtype=u8),
            "w13s": empty(48 * 8, 16, 256, dtype=e4m3),
            "w2": empty(H, 6, 256, dtype=u8),
            "w2s": empty(32, 6, 16, 256, dtype=e4m3),
            "scales": empty(4),
        }
        x = empty(T, H, dtype=bf16)
        return module.kernel_args(x, pw, partial(), mhc, out, group_name, raw, raw)
    cache = empty(16384, 512, dtype=e4m3)
    if name in ("mla_front_tp4", "mla_frontidx_tp4"):
        pw = {
            "q_a_norm": empty(1536, dtype=bf16),
            "kv_a_norm": empty(512, dtype=bf16),
            "w_uk": empty(16, 512, 256, dtype=bf16),
        }
        x, loc = empty(T, H, dtype=bf16), empty(T, dtype=i64)
        if name == "mla_front_tp4":
            pw["w_a"] = empty(2048, 8, 512, dtype=bf16)
            pw["w_qb"] = empty(4096, 4, 384, dtype=bf16)
            return module.kernel_args(x, pw, loc, cache)
        pw["w_ak"] = empty(2432, 8, 512, dtype=bf16)
        pw["w_q"] = empty(8192, 2, 768, dtype=bf16)
        pw.update(k_norm_w=empty(128), k_norm_b=empty(128), ape=empty(4, 128))
        pw["hmat"] = empty(128, 128, dtype=bf16)
        # A served CUDA graph's padded page table.
        state = {
            "cache": empty(1600, 132 * 64, dtype=u8),
            "page_table": empty(module.WIDE_PAGES, dtype=i32),
            "tail_k": empty(4, 10, 128, dtype=bf16),
            "tail_s": empty(4, 10, 128, dtype=bf16),
            "req": empty(1, dtype=i64),
            "write_start": empty(1, dtype=i32),
            "tail_start": empty(1, dtype=i32),
            "write_loc": empty(1, 2, dtype=i64),
            "pool_lens": empty(T, dtype=i32),
        }
        return module.kernel_args(x, pw, loc, cache, state)
    if name in ("mla_out_tp4", "mtp_attn_out_tp4"):
        attention = (
            empty(T, 16, 512, dtype=e4m3),
            cache,
            empty(T, 2051, dtype=i32),
            empty(16, 256, 512, dtype=bf16),
            empty(H, H, dtype=bf16),
            partial(),
        )
        if name == "mla_out_tp4":
            return module.kernel_args(*attention, mhc, out, group_name, raw, raw)
        residual, norm = empty(T, H, dtype=bf16), empty(H, dtype=bf16)
        rows = (empty(T, H, dtype=bf16), empty(T, H, dtype=bf16))
        return module.kernel_args(*attention, residual, norm, rows, group_name)
    if name == "mtp_eh_tp4":
        sym = module.symmetric_buffers(T, H // 4, group_name, _CPU)[0]
        return module.kernel_args(
            empty(T, dtype=i64), empty(154880 // 4, H, dtype=bf16),
            empty(T, H, dtype=bf16), empty(H, dtype=bf16), empty(H, dtype=bf16),
            empty(H // 4, 2 * H, dtype=bf16), empty(H, dtype=bf16), sym,
            (empty(T, H, dtype=bf16), empty(T, H, dtype=bf16)), group_name, 0,
        )  # fmt: skip
    if name == "mtp_moe_tp4":
        w = {
            "router": empty(288, H, dtype=bf16),
            "bias": empty(288),
            "w13": empty(288, 64, 1024, 64, dtype=bf16),
            "w2": empty(288, 8, H, 64, dtype=bf16),
            "sw13": empty(1024, H, dtype=bf16),
            "sw2": empty(H, 512, dtype=bf16),
        }
        x, residual, norm = empty(T, H, dtype=bf16), empty(T, H, dtype=bf16), empty(H)
        rows = empty(T, H, dtype=bf16)
        return module.kernel_args(
            x, w, residual, norm.bfloat16(), partial(), rows, group_name
        )
    ws = module.make_workspaces(T, group_name, 4, device=_CPU)
    weight = empty(154880 // 4, H, dtype=bf16)
    return module.kernel_args(
        empty(T, H, dtype=bf16), weight, ws, empty(T, 154880), group_name, 0, 4
    )


@_needs_gluon
@pytest.mark.parametrize(
    ("name", "tokens", "raw"),
    [
        (name, tokens, raw)
        for name, (_, cases) in _GLM5_3_FLASH_KERNELS.items()
        for tokens, raw in cases
    ],
)
def test_glm5_3_flash_ring_loads(name: str, tokens: int, raw: bool) -> None:
    module = importlib.import_module(
        f"pretuned_kernels.megakernels.glm5_3_flash_tp4.{name}"
    )
    kernel = getattr(module, _GLM5_3_FLASH_KERNELS[name][0])
    with _modelled_ranks(4) as group:
        bound = kernel.bind(_glm5_3_flash_args(module, group.group_name, tokens, raw))
        config = module.mk_config(bound)
    ring = {
        fact.tensor_name
        for fact in bound.config_spec.memory_op_facts
        if fact.kind == "load"
        and config["indexing"][fact.indexing_index] == "tensor_descriptor"
    }
    assert ring == set(module.RING_LOADS), ring ^ set(module.RING_LOADS)
