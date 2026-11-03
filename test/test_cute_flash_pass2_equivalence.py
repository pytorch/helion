"""The PASS2 twin helpers of the fa4 body are one helper each, under a flag.

``fa4_disc_exp_convert_store{,_pipe,_rep32,_rep32_pipe,_rep32_split}`` take
``causal`` (with the query tile ``m_block`` and the key tile ``n_block``)
instead of a ``_causal`` twin, ``fa4_sp_exp_convert_store{,_rep32_split}``
take ``whole_row_sum`` and ``fa4_correction_epilogue_{,handoff_}to_smem_scoped``
take ``use_2cta_instrs``.  The emitter half pins the call forms the fa4 body
renders over the admitted PASS2 surface: the helper follows the P-store /
S-load repetitions, the pipeline depth and the staged-P split, the masked
segment adds exactly the three mask keywords, and every call binds to the
runtime signature.  The runtime half traces each helper with the DSL mocked
out: the flag adds exactly its step (the mask of every loaded chunk between
its load and its exp, the one packed row sum, the pair's 256-row TMEM atom)
and the plain trace is the flagged trace without those events.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion._testing.cute_aux import flash_env

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

# ``_flash_runtime`` imports cutlass at module scope, so it is loaded after the
# skip gate above.
_rt = importlib.import_module("helion._compiler.cute._flash_runtime")

_MASK_KEYWORDS = ("causal", "m_block", "n_block")
_SM103 = (10, 3)


# --------------------------------------------------------------------------
# the emitter
# --------------------------------------------------------------------------


def _emit(
    *,
    is_causal: bool,
    head_dim: int = 64,
    family: str = "fa4",
    num_kv: int = 512,
    dtype: torch.dtype = torch.float16,
    capability: tuple[int, int] | None = None,
    fast_math: bool = False,
    env: dict[str, str] | None = None,
    **values: object,
) -> tuple[cute_flash.FlashAttentionConfig, ast.Module]:
    """The fa4 body (resolved config, AST) for ``values`` on top of ``family``."""
    with flash_env(env):
        config = cute_flash.resolve_flash_config(
            head_dim,
            num_kv,
            {cute_flash.FLASH_PIPELINE_FAMILY_KEY: family, **values},
            dtype=dtype,
            is_causal=is_causal,
            standard_dense_output=not is_causal,
            standard_causal_output=is_causal,
            target_device_capability=capability,
            fast_math=fast_math,
        )
        body = cute_flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=head_dim,
            num_kv=num_kv,
            sequence_extent=num_kv * 128,
            num_bh=64,
            total_tiles=num_kv * 16,
            cfg=config,
            has_lse=False,
            io_dtype="cutlass.BFloat16"
            if dtype is torch.bfloat16
            else "cutlass.Float16",
            score_plan=causal_score_plan(head_dim)
            if is_causal
            else dense_score_plan(head_dim),
            target_device_capability=capability,
        )
    return config, ast.Module(body=body, type_ignores=[])


def _runtime_calls(module: ast.Module, prefix: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_helion_flash_rt"
        and node.func.attr.startswith(prefix)
    ]


def _keywords(call: ast.Call) -> dict[str, str]:
    return {cast("str", kw.arg): ast.unparse(kw.value) for kw in call.keywords}


def _binds_to_runtime(call: ast.Call) -> None:
    """The rendered call binds to the runtime helper's signature."""
    name = cast("ast.Attribute", call.func).attr
    inspect.signature(getattr(_rt, name)).bind(
        *([None] * len(call.args)), **dict.fromkeys(_keywords(call))
    )


# (P-store rep, S-load rep, pipeline depth, staged-P split, per-chunk
# release) -> the chunked helper the dense body and the causal body name.
# The resolver keeps Rep32 with one P arrival only (the causal split store
# falls back to Rep16), so the causal column differs where it says so.
_DISC_CASES: dict[str, tuple[dict[str, object], str, str]] = {
    "serial": (
        {"ps": 16, "sl": 32, "pipe": 1, "split": True},
        "fa4_disc_exp_convert_store",
        "fa4_disc_exp_convert_store",
    ),
    "pipe": (
        {"ps": 16, "sl": 32, "pipe": 2, "split": True},
        "fa4_disc_exp_convert_store_pipe",
        "fa4_disc_exp_convert_store_pipe",
    ),
    "pipe_full_prologue": (
        {"ps": 16, "sl": 32, "pipe": 4, "split": False},
        "fa4_disc_exp_convert_store_pipe",
        "fa4_disc_exp_convert_store_pipe",
    ),
    "p_chunk_arrive": (
        {"ps": 16, "sl": 32, "pipe": 2, "split": True, "pchunk": True},
        "fa4_disc_exp_convert_store_pipe",
        "fa4_disc_exp_convert_store_pipe",
    ),
    "rep32": (
        {"ps": 32, "sl": 32, "pipe": 1, "split": False},
        "fa4_disc_exp_convert_store_rep32",
        "fa4_disc_exp_convert_store_rep32",
    ),
    "rep32_pipe": (
        {"ps": 32, "sl": 32, "pipe": 2, "split": False},
        "fa4_disc_exp_convert_store_rep32_pipe",
        "fa4_disc_exp_convert_store_rep32_pipe",
    ),
    "rep32_split": (
        {"ps": 32, "sl": 32, "pipe": 1, "split": True},
        "fa4_disc_exp_convert_store_rep32_split",
        "fa4_disc_exp_convert_store",
    ),
    "sload16_paired": (
        {"ps": 16, "sl": 16, "pipe": 1, "split": True},
        "fa4_disc_exp_convert_store_rep32_pipe",
        "fa4_disc_exp_convert_store_rep32_pipe",
    ),
    "sload16_paired_pipe": (
        {"ps": 16, "sl": 16, "pipe": 2, "split": False},
        "fa4_disc_exp_convert_store_rep32_pipe",
        "fa4_disc_exp_convert_store_rep32_pipe",
    ),
}


def _disc_values(knobs: dict[str, object]) -> dict[str, object]:
    return {
        cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
        cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "standard",
        cute_flash.FLASH_P_STORE_REP_KEY: knobs["ps"],
        cute_flash.FLASH_S_LOAD_REP_KEY: knobs["sl"],
        cute_flash.FLASH_DISC_PIPE_KEY: knobs["pipe"],
        cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: knobs["split"],
        cute_flash.FLASH_P_CHUNK_ARRIVE_KEY: knobs.get("pchunk", False),
        cute_flash.FLASH_CAUSAL_MASK_STOREBACK_KEY: False,
    }


@pytest.mark.parametrize("two_cta", (False, True), ids=("1cta", "2cta"))
@pytest.mark.parametrize("case", sorted(_DISC_CASES))
def test_disc_pass2_is_one_helper_under_the_causal_keyword(
    case: str, two_cta: bool
) -> None:
    knobs, dense_helper, causal_helper = _DISC_CASES[case]
    dense_config, dense = _emit(
        is_causal=False,
        family="fa4_2cta" if two_cta else "fa4",
        **_disc_values(knobs),
    )
    causal_config, causal = _emit(
        is_causal=True,
        family="fa4_2cta_causal" if two_cta else "fa4",
        **_disc_values(knobs),
    )
    assert (dense_config.use_2cta_instrs, causal_config.use_2cta_instrs) == (
        two_cta,
        two_cta,
    )
    dense_calls = _runtime_calls(dense, "fa4_disc_exp_convert_store")
    causal_calls = _runtime_calls(causal, "fa4_disc_exp_convert_store")
    assert dense_calls and causal_calls
    assert {cast("ast.Attribute", c.func).attr for c in dense_calls} == {dense_helper}
    assert {cast("ast.Attribute", c.func).attr for c in causal_calls} == {causal_helper}
    for call in dense_calls + causal_calls:
        _binds_to_runtime(call)
    # The dense body never masks; the per-chunk release is its own.
    for call in dense_calls:
        keywords = _keywords(call)
        assert not set(keywords) & set(_MASK_KEYWORDS)
        assert ("pforc_ptr_stage" in keywords) == dense_config.p_chunk_arrive
    # The causal body's masked segment is the same helper with the mask
    # keywords (the query tile coordinate in the pair's MMA-tile units
    # under 2-CTA), its unmasked segment the plain call; the per-chunk
    # release never joins a masked call.
    masked = [c for c in causal_calls if "causal" in _keywords(c)]
    plain = [c for c in causal_calls if "causal" not in _keywords(c)]
    assert masked and plain
    for call in masked:
        keywords = _keywords(call)
        assert keywords["causal"] == "True"
        assert keywords["n_block"] == "flash_kv"
        assert keywords["m_block"] in {
            f"flash_q_mma_tile{stage} * cutlass.Int32(2)"
            if two_cta
            else f"flash_m_tile{stage}"
            for stage in "01"
        }
        assert "pforc_ptr_stage" not in keywords
    for call in plain:
        assert not set(_keywords(call)) & set(_MASK_KEYWORDS)
    # The mask keywords are keyword-only on every chunked helper, so a
    # positional render can never reach them.
    for name in {dense_helper, causal_helper}:
        parameters = inspect.signature(getattr(_rt, name)).parameters
        assert all(
            parameters[key].kind is inspect.Parameter.KEYWORD_ONLY
            for key in _MASK_KEYWORDS
        )
        assert parameters["causal"].default is False


def test_sp_pass2_row_sum_form_is_a_keyword() -> None:
    """The whole-row PASS2 (the sm_103 dense policy's body) names the same
    helper for both row-sum forms; ``whole_row_sum=True`` is the only
    difference between the two renders."""
    policy = get_flash_target_policy(_SM103).tuning.dense_policy(256)
    assert policy is not None
    renders = {}
    for row_sum in ("whole", "fragment"):
        config, module = _emit(
            is_causal=False,
            num_kv=256,
            capability=_SM103,
            fast_math=True,
            **{**policy.knob_values(), cute_flash.FLASH_SP_ROW_SUM_KEY: row_sum},
        )
        assert (config.softmax_disc, config.sp_row_sum) == (False, row_sum)
        calls = _runtime_calls(module, "fa4_sp_exp_convert_store")
        assert calls
        for call in calls:
            assert cast("ast.Attribute", call.func).attr == "fa4_sp_exp_convert_store"
            assert ("whole_row_sum" in _keywords(call)) == (row_sum == "whole")
            _binds_to_runtime(call)
        renders[row_sum] = ast.unparse(module)
    assert renders["whole"] != renders["fragment"]
    assert renders["whole"].replace(", whole_row_sum=True)", ")") == renders["fragment"]


@pytest.mark.parametrize(
    ("env", "helper"),
    (
        ({}, "fa4_correction_epilogue_to_smem_scoped"),
        (
            {"HELION_CUTE_FLASH_SPLIT_CORR_EPILOGUE_HANDOFF": "0"},
            "fa4_correction_epilogue_handoff_to_smem_scoped",
        ),
    ),
    ids=("split_handoff", "handoff"),
)
def test_scoped_correction_epilogue_pair_is_a_keyword(
    env: dict[str, str], helper: str
) -> None:
    for family, two_cta in (("fa4", False), ("fa4_2cta", True)):
        config, module = _emit(
            is_causal=False,
            family=family,
            num_kv=256,
            env=env,
            **{cute_flash.FLASH_EPI_TMA_KEY: True},
        )
        assert config.use_2cta_instrs is two_cta
        calls = _runtime_calls(module, "fa4_correction_epilogue_")
        assert {cast("ast.Attribute", c.func).attr for c in calls} == {helper}
        for call in calls:
            assert _keywords(call) == ({"use_2cta_instrs": "True"} if two_cta else {})
            _binds_to_runtime(call)


# --------------------------------------------------------------------------
# the runtime
# --------------------------------------------------------------------------


class _Tracer:
    """The DSL calls a helper body makes, with cute mocked out.

    Tensors made by ``indexed`` tag their ``[...]`` views by name and key;
    every other object is named in order of first appearance, so two traces
    of the same body compare by value."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self._keep: list[object] = []
        self._views: dict[int, tuple[str, object]] = {}
        self._names: dict[int, str] = {}

    def fresh(self, *args: object, **kwargs: object) -> MagicMock:
        return self.keep(MagicMock())

    def keep(self, obj: MagicMock) -> MagicMock:
        self._keep.append(obj)
        return obj

    def indexed(self, name: str) -> MagicMock:
        tensor = self.keep(MagicMock(name=name))
        self._views[id(tensor)] = (name, "all")

        def item(key: object) -> MagicMock:
            view = self.keep(MagicMock(name=f"{name}[{key}]"))
            self._views[id(view)] = (name, key)
            return view

        tensor.__getitem__.side_effect = item
        return tensor

    def tag(self, obj: object) -> object:
        if obj is None or isinstance(obj, (bool, int, float, str, tuple)):
            return obj
        if id(obj) in self._views:
            return self._views[id(obj)]
        self._keep.append(obj)
        return self._names.setdefault(id(obj), f"obj{len(self._names)}")

    def event(
        self, name: str, *positions: int, result: object = None
    ) -> Callable[..., object]:
        def record(*args: object, **kwargs: object) -> object:
            self.events.append((name, *(self.tag(args[i]) for i in positions)))
            return result

        return record

    def patches(self) -> list[Any]:
        return [
            patch.object(_rt.cutlass, "Float32", side_effect=float),
            patch.object(_rt.cutlass, "const_expr", side_effect=lambda x: x),
            patch.object(_rt.cute, "make_rmem_tensor", side_effect=self.fresh),
            patch.object(_rt.cute, "make_tensor", side_effect=self.fresh),
            patch.object(_rt.cute, "make_layout", side_effect=self.fresh),
            patch.object(_rt.cute, "recast_ptr", side_effect=self.fresh),
            patch.object(_rt.cute, "copy", side_effect=self.event("copy", 1, 2)),
            patch.object(
                _rt.cute.arch,
                "fence_view_async_tmem_store",
                side_effect=self.event("fence_tmem_store"),
            ),
            patch.object(
                _rt.cute.arch,
                "fence_view_async_tmem_load",
                side_effect=self.event("fence_tmem_load"),
            ),
            patch.object(
                _rt,
                "tcgen05_fence_before_thread_sync",
                side_effect=self.event("tcgen05_fence"),
            ),
            patch.object(_rt, "mbarrier_arrive", side_effect=self.event("arrive", 0)),
            patch.object(_rt, "_disc_chunk_exp", side_effect=self.event("exp", 0, 6)),
            patch.object(
                _rt,
                "_disc_chunk_convert_store",
                side_effect=self.event("convert_store", 0, 4),
            ),
            patch.object(
                _rt,
                "_disc_chunk_pair_convert_store",
                side_effect=self.event("pair_convert_store", 0, 1, 5),
            ),
            patch.object(
                _rt,
                "_disc_chunk_rowsum",
                side_effect=self.event("rowsum", 0, result=1.0),
            ),
            patch.object(
                _rt,
                "fadd_reduce_packed",
                side_effect=self.event("reduce", 0, result=4.0),
            ),
            patch.object(_rt, "_disc_pin_frag", side_effect=self.event("pin", 0)),
            patch.object(
                _rt,
                "causal_mask_t2r_chunk",
                side_effect=self.event("mask", 0, 1, 2, 3, 4),
            ),
        ]


def _traced(run: Callable[[_Tracer], object]) -> list[tuple[object, ...]]:
    tracer = _Tracer()
    with patch.multiple(_rt.cute, size=MagicMock(return_value=128)):
        patches = tracer.patches()
        for p in patches:
            p.start()
        try:
            run(tracer)
        finally:
            for p in reversed(patches):
                p.stop()
    return tracer.events


# helper name -> (positional arguments after the six tensors, loaded chunks)
_DISC_RUNTIME: dict[str, tuple[tuple[object, ...], int]] = {
    # scale, minus_max_scale, e2e_freq, e2e_res, e2e_offset, pfor, pfor2,
    # p_store_split, p_store_chunks[, pipe_depth]
    "fa4_disc_exp_convert_store": ((1.0, 0.0, 16, 6, 0, "pfor", "pfor2", 3, 4), 4),
    "fa4_disc_exp_convert_store_pipe": (
        (1.0, 0.0, 16, 6, 0, "pfor", "pfor2", 3, 4, 2),
        4,
    ),
    "fa4_disc_exp_convert_store_rep32": (
        (1.0, 0.0, 16, 6, 0, "pfor", "pfor2", 2, 2),
        4,
    ),
    "fa4_disc_exp_convert_store_rep32_pipe": (
        (1.0, 0.0, 16, 6, 0, "pfor", "pfor2", 2, 2, 2),
        4,
    ),
    # scale, ..., pfor, pfor2, ld_chunks (after nine tensors)
    "fa4_disc_exp_convert_store_rep32_split": (
        (1.0, 0.0, 16, 6, 0, "pfor", "pfor2", 4),
        4,
    ),
}


def _run_disc(name: str, **flags: object) -> list[tuple[object, ...]]:
    arguments, _ = _DISC_RUNTIME[name]

    def run(tracer: _Tracer) -> None:
        tensors: list[object] = [None, tracer.indexed("tLDtS"), tracer.indexed("tLDcS")]
        stores = 2 if name.endswith("_split") else 1
        for index in range(stores):
            tensors += [
                None,
                tracer.indexed(f"tSTtS{index}"),
                tracer.indexed(f"tSTcS{index}"),
            ]
        getattr(_rt, name)(*tensors, *arguments, **flags)

    return _traced(run)


@pytest.mark.parametrize("name", sorted(_DISC_RUNTIME))
def test_disc_helper_causal_flag_adds_exactly_the_chunk_masks(name: str) -> None:
    _, loaded_chunks = _DISC_RUNTIME[name]
    plain = _run_disc(name)
    masked = _run_disc(name, causal=True, m_block="m", n_block="n")
    masks = [event for event in masked if event[0] == "mask"]
    assert "mask" not in {event[0] for event in plain}
    assert [event for event in masked if event[0] != "mask"] == plain
    # One mask per loaded chunk, in chunk order, on the chunk's own
    # coordinates and the given tile indices ...
    assert [event[5] for event in masks] == list(range(loaded_chunks))
    for event in masks:
        _, frg, coords, m_block, n_block, chunk = event
        assert coords == ("tLDcS", (None, chunk, None, None))
        assert (m_block, n_block) == ("m", "n")
        # ... placed between the chunk's load and its exp.
        load = masked.index(("copy", ("tLDtS", (None, chunk, None, None)), frg))
        mask_at = masked.index(event)
        exp_at = next(
            at
            for at, other in enumerate(masked)
            if at > load and other[:2] == ("exp", frg)
        )
        assert load < mask_at < exp_at


# The sp helpers' bodies by name: ``fa4_sp_exp_convert_store`` is
# ``@dsl_user_op``-wrapped (traced through its body, as the scoped correction
# epilogue is below); the Rep32 split helper is a plain function.
_SP_BODIES: dict[str, Callable[..., object]] = {
    "fa4_sp_exp_convert_store": _rt.fa4_sp_exp_convert_store.__wrapped__,
    "fa4_sp_exp_convert_store_rep32_split": _rt.fa4_sp_exp_convert_store_rep32_split,
}


def _run_sp(name: str, **flags: object) -> list[tuple[object, ...]]:
    def run(tracer: _Tracer) -> None:
        tLDrS = tracer.indexed("tLDrS")
        divided = iter(("src", "dst"))

        def divide(tensor: object, layout: object) -> MagicMock:
            return tracer.indexed(next(divided))

        with patch.object(_rt.cute, "logical_divide", side_effect=divide):
            helper = _SP_BODIES[name]
            if name.endswith("_split"):
                helper(
                    tLDrS,
                    None,
                    tracer.indexed("tST32tS"),
                    tracer.indexed("tST32cS"),
                    None,
                    tracer.indexed("tST16tS"),
                    tracer.indexed("tST16cS"),
                    1.0,
                    0.0,
                    16,
                    6,
                    0,
                    "pfor",
                    "pfor2",
                    **flags,
                )
            else:
                helper(
                    tLDrS,
                    None,
                    tracer.indexed("tSTtS"),
                    tracer.indexed("tSTcS"),
                    1.0,
                    0.0,
                    16,
                    6,
                    0,
                    "pfor",
                    "pfor2",
                    3,
                    4,
                    **flags,
                )

    return _traced(run)


@pytest.mark.parametrize(
    "name", ("fa4_sp_exp_convert_store", "fa4_sp_exp_convert_store_rep32_split")
)
def test_sp_helper_whole_row_sum_flag_changes_only_the_reduction(name: str) -> None:
    fragment = _run_sp(name)
    whole = _run_sp(name, whole_row_sum=True)
    published = max(
        at for at, event in enumerate(whole) if event == ("arrive", "pfor2")
    )
    assert whole[: published + 1] == fragment[: published + 1]
    assert whole[published + 1 :] == [("reduce", ("tLDrS", "all"))]
    assert fragment[published + 1 :] == [
        ("rowsum", ("src", (None, chunk))) for chunk in range(4)
    ]


def _run_correction_epilogue(
    handoff: bool, **flags: object
) -> list[tuple[object, ...]]:
    def run(tracer: _Tracer) -> None:
        with (
            patch.object(
                _rt.sm100_utils_flash,
                "get_tmem_load_op",
                side_effect=lambda *args, **kwargs: tracer.events.append(
                    ("tmem_atom", args[0], kwargs["use_2cta_instrs"])
                ),
            ),
            patch.object(
                _rt.cute,
                "make_identity_tensor",
                side_effect=tracer.event("identity", 0),
            ),
            patch.object(_rt.cute, "logical_divide", side_effect=tracer.fresh),
            patch.object(_rt.tcgen05, "make_tmem_copy", side_effect=tracer.fresh),
            patch.object(
                _rt.sm100_utils_flash, "get_smem_store_op", side_effect=tracer.fresh
            ),
            patch.object(_rt.cute, "make_tiled_copy_D", side_effect=tracer.fresh),
            patch.object(
                _rt, "partition_D_position_independent", side_effect=tracer.fresh
            ),
            patch.object(_rt, "cvt_copy", side_effect=tracer.event("cvt_copy", 1, 2)),
            patch.object(_rt, "mbar_spin_wait", side_effect=tracer.event("wait", 0, 1)),
            patch.object(
                _rt.cute.arch,
                "fence_view_async_shared",
                side_effect=tracer.event("fence_shared"),
            ),
            patch.object(
                _rt.cute.arch, "mbarrier_arrive", side_effect=tracer.event("arrive", 0)
            ),
        ):
            if handoff:
                _rt.fa4_correction_epilogue_handoff_to_smem_scoped(
                    "o_full",
                    "o_phase",
                    "epi_empty",
                    "epi_phase",
                    "epi_full",
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    0,
                    0.5,
                    64,
                    16,
                    None,
                    **flags,
                )
            else:
                _rt.fa4_correction_epilogue_to_smem_scoped.__wrapped__(
                    MagicMock(), MagicMock(), MagicMock(), 0, 0.5, 64, 16, None, **flags
                )

    return _traced(run)


@pytest.mark.parametrize("handoff", (False, True), ids=("scoped", "handoff"))
def test_correction_epilogue_pair_flag_changes_only_the_tmem_atom(
    handoff: bool,
) -> None:
    one = _run_correction_epilogue(handoff)
    two = _run_correction_epilogue(handoff, use_2cta_instrs=True)
    head = 2 if handoff else 0  # the two handoff waits precede the epilogue
    assert one[head : head + 2] == [
        ("tmem_atom", (128, 64), False),
        ("identity", (128, 64)),
    ]
    assert two[head : head + 2] == [
        ("tmem_atom", (256, 64), True),
        ("identity", (256, 64)),
    ]
    assert one[:head] == two[:head]
    assert one[head + 2 :] == two[head + 2 :]
    assert [event[0] for event in one].count("cvt_copy") == 4
    if handoff:
        assert one[:2] == [
            ("wait", "o_full", "o_phase"),
            ("wait", "epi_empty", "epi_phase"),
        ]
        assert one[-2:] == [("fence_shared",), ("arrive", "epi_full")]
