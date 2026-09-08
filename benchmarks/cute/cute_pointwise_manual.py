# pyrefly: ignore-errors
"""Handwritten CuTe-DSL pointwise kernels: the structural baseline for the
Helion cute-backend pointwise comparison (quack has no standalone pointwise
kernels; its activations are epilogue device functions, which we reuse here).

Structure (per config = threads x vec x unroll):

- the flat tensor is split into CTA tiles of ``threads * vec * unroll``
  elements; grid = numel / tile (configs where the tile does not divide numel
  are skipped by the sweep, so there is no predication anywhere),
- each thread copies ``unroll`` vectors of ``vec`` contiguous elements
  (vectors of consecutive threads are adjacent -> perfectly coalesced;
  ``cute.autovec_copy`` emits 128-bit accesses when vec*esize >= 16B),
- math is scalarized in fp32 (fastmath, quack's activation device functions),
  matching aten's opmath for 16-bit dtypes.

``make_manual_fn`` compiles a small config sweep, times each with do_bench,
and returns the best callable (mirrors the "quack-tuned" pattern of the
cross_entropy harness).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
QUACK_PATH = REPO_ROOT / "quack"
if str(QUACK_PATH) not in sys.path:
    sys.path.insert(0, str(QUACK_PATH))

import cutlass  # noqa: E402
from cutlass import Float32  # noqa: E402
from cutlass import const_expr  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from quack import compile_utils  # noqa: E402
from quack import cute_dsl_utils  # noqa: E402
import quack.activation as qact  # noqa: E402

make_fake_tensor = compile_utils.make_fake_tensor
torch2cute_dtype_map = cute_dsl_utils.torch2cute_dtype_map

# op name -> number of flat tensor inputs (bias_add additionally reads mBias)
OP_ARITY = {
    "add": 2,
    "mul": 2,
    "copy": 1,
    "cast": 1,
    "relu": 1,
    "gelu_tanh": 1,
    "silu": 1,
    "sigmoid": 1,
    "tanh": 1,
    "exp": 1,
    "rsqrt": 1,
    "addcmul": 3,
    "saxpy": 2,
    "bias_add": 1,
    "leaky_relu": 1,
    "clamp": 1,
}


class PointwiseManual:
    def __init__(
        self,
        op: str,
        numel: int,
        threads: int,
        vec: int,
        unroll: int,
        bias_n: int = 0,
    ) -> None:
        self.op = op
        self.numel = numel
        self.threads = threads
        self.vec = vec
        self.unroll = unroll
        self.bias_n = bias_n
        self.tile = threads * vec * unroll
        assert numel % self.tile == 0, (numel, self.tile)

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor | None,
        mC: cute.Tensor | None,
        mBias: cute.Tensor | None,
        stream,  # noqa: ANN001
    ) -> None:
        self.kernel(mOut, mA, mB, mC, mBias).launch(
            grid=[self.numel // self.tile, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    def _apply(self, a, b, c, bias):  # noqa: ANN001, ANN202
        # Trace-time dispatch (self.op is a Python constant). All values fp32.
        op = self.op
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op in ("copy", "cast"):
            return a
        if op == "relu":
            return qact.relu(a)
        if op == "gelu_tanh":
            return qact.gelu_tanh_approx(a)
        if op == "silu":
            return qact.silu(a)
        if op == "sigmoid":
            return qact.sigmoid(a)
        if op == "tanh":
            return qact.tanh(a)
        if op == "exp":
            return cute.math.exp(a, fastmath=True)
        if op == "rsqrt":
            return cute.math.rsqrt(a, fastmath=True)
        if op == "addcmul":
            return a + b * c
        if op == "saxpy":
            return 2.5 * a + b
        if op == "bias_add":
            return a + bias
        if op == "leaky_relu":
            # x>0 ? x : 0.01x == max(x,0) + 0.01*min(x,0), branch-free/exact
            return qact._maxnumf(a, 0.0) + 0.01 * qact._minnumf(a, 0.0)
        if op == "clamp":
            return qact._minnumf(qact._maxnumf(a, -2.0), 2.0)
        raise ValueError(op)

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor | None,
        mC: cute.Tensor | None,
        mBias: cute.Tensor | None,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv = cute.make_layout(
            ((self.threads,), (self.vec, self.unroll)),
            stride=((self.vec,), (1, self.threads * self.vec)),
        )
        frgs = []
        for mIn in (mA, mB, mC):
            if const_expr(mIn is not None):
                gI = cute.local_tile(mIn, (self.tile,), (bidx,))
                thrI = cute.composition(gI, tv)[(tidx, None)]
                frg = cute.make_fragment_like(thrI)
                cute.autovec_copy(thrI, frg)
                frgs.append(frg)
        gO = cute.local_tile(mOut, (self.tile,), (bidx,))
        thrO = cute.composition(gO, tv)[(tidx, None)]
        frgO = cute.make_fragment_like(thrO)
        out_t = mOut.element_type
        base = bidx * self.tile + tidx * self.vec
        for u in range(self.unroll):  # Python loops unroll at trace time
            for v in range(self.vec):
                i = u * self.vec + v  # linear index into ((vec, unroll))
                a = frgs[0][i].to(Float32)
                b = frgs[1][i].to(Float32) if len(frgs) > 1 else None
                c = frgs[2][i].to(Float32) if len(frgs) > 2 else None
                bias = None
                if const_expr(self.bias_n > 0):
                    col = (base + u * self.threads * self.vec + v) % self.bias_n
                    bias = mBias[col].to(Float32)
                frgO[i] = self._apply(a, b, c, bias).to(out_t)
        cute.autovec_copy(frgO, thrO)


# (threads, vec_bytes, unroll); vec elements = vec_bytes // widest_esize
SWEEP_CONFIGS = (
    (256, 16, 1),
    (256, 16, 2),
    (256, 16, 4),
    (128, 16, 4),
    (512, 16, 1),
    (512, 16, 2),
    (256, 32, 1),
    (256, 32, 2),
    (512, 32, 1),
    (128, 32, 2),
)


def _compile_manual(
    op: str,
    numel: int,
    in_dtypes: tuple[torch.dtype, ...],
    out_dtype: torch.dtype,
    threads: int,
    vec: int,
    unroll: int,
    bias_n: int,
    bias_dtype: torch.dtype | None,
) -> object:
    cutlass.cuda.initialize_cuda_context()
    fakes: list[Any] = []
    for dt in in_dtypes:
        cdt = torch2cute_dtype_map[dt]
        fakes.append(make_fake_tensor(cdt, (numel,), divisibility=128 // cdt.width))
    while len(fakes) < 3:
        fakes.append(None)
    bias_fake = None
    if bias_n:
        cbt = torch2cute_dtype_map[bias_dtype]
        bias_fake = make_fake_tensor(cbt, (bias_n,), divisibility=1)
    cot = torch2cute_dtype_map[out_dtype]
    out_fake = make_fake_tensor(cot, (numel,), divisibility=128 // cot.width)
    kernel = PointwiseManual(op, numel, threads, vec, unroll, bias_n=bias_n)
    return cute.compile(
        kernel,
        out_fake,
        fakes[0],
        fakes[1],
        fakes[2],
        bias_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def make_manual_fn(
    op: str, inputs: tuple[torch.Tensor, ...], sweep: bool = True
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, str, float]:
    """Returns (fn, out_tensor, config_desc, tune_seconds)."""
    bias = None
    tensor_inputs = inputs
    bias_n = 0
    if op == "bias_add":
        tensor_inputs, bias = (inputs[0],), inputs[1]
        bias_n = bias.numel()
    arity = OP_ARITY[op]
    assert len(tensor_inputs) == arity, (op, len(tensor_inputs))
    orig_shape = tensor_inputs[0].shape
    flats = [t.reshape(-1) for t in tensor_inputs]
    numel = flats[0].numel()
    in_dtypes = tuple(t.dtype for t in flats)
    out_dtype = torch.bfloat16 if op == "cast" else tensor_inputs[0].dtype
    out = torch.empty(numel, dtype=out_dtype, device=tensor_inputs[0].device)
    widest = max(t.element_size() for t in [*flats, out])

    configs = list(SWEEP_CONFIGS) if sweep else [SWEEP_CONFIGS[0]]
    t0 = time.time()
    best: tuple[float, Any, str] | None = None
    from triton.testing import do_bench

    for threads, vec_bytes, unroll in configs:
        vec = vec_bytes // widest
        if vec < 1 or numel % (threads * vec * unroll):
            continue
        try:
            compiled = _compile_manual(
                op,
                numel,
                in_dtypes,
                out_dtype,
                threads,
                vec,
                unroll,
                bias_n,
                bias.dtype if bias is not None else None,
            )
        except Exception as e:
            print(f"manual config ({threads},{vec_bytes},{unroll}) failed: {e}")
            continue
        args = [out, *flats]
        while len(args) < 4:
            args.append(None)
        args.append(bias)
        fn = lambda a=tuple(args), c=compiled: c(*a)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        ms = float(do_bench(fn, warmup=10, rep=50, return_mode="median"))  # pyrefly: ignore [bad-argument-type]
        desc = f"threads={threads} vec={vec} unroll={unroll}"
        if best is None or ms < best[0]:
            best = (ms, fn, desc)
    assert best is not None, f"no valid manual config for {op} numel={numel}"
    _, fn, desc = best
    fn()  # repopulate `out` with the winning config's result
    return fn, out.view(orig_shape), desc, round(time.time() - t0, 1)
