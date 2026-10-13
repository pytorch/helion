from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_loop import _config as _m64_config
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_tcgen_stage
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

_KEY = "cute_chained_m64_transpose"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _square_intermediate(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.grid(steps):
            rows = hl.arange(value.size(2))
            columns = hl.arange(value.size(3))
            raw = hl.load(
                value,
                [batch.id, step, rows, columns],
                extra_mask=(rows < step * 17 + 3)[:, None],
            ).float()
            entering = state.to(left.dtype)
            corrected = raw - hl.dot(right[batch.id, step, :, :], entering)
            cross = hl.dot(left[batch.id, step, :, :], entering)
            solved = hl.dot(
                transform[batch.id, step, :, :], corrected.to(left.dtype)
            ).to(left.dtype)
            cross = cross * gate[batch.id, step, :][:, None]
            result = hl.dot(transform[batch.id, step, :, :], solved, acc=cross)
            output[batch.id, step, :, :] = result.to(output.dtype)
            state = hl.dot(
                update[batch.id, step, :, :],
                solved,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


def _config(enabled: bool = True, *, pipeline: bool = True) -> helion.Config:
    return _m64_config(
        **{
            _KEY: enabled,
            "cute_chained_preparation_pipeline": pipeline,
            "cute_chained_late_widening": pipeline,
            "cute_chained_prepared_group_multi_atom": pipeline,
        }
    )


def _source(enabled: bool = True, *, pipeline: bool = True) -> str:
    args = _args()
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_square_intermediate, args)
        ):
            return bound.to_code(_config(enabled, pipeline=pipeline))


@pytest.mark.parametrize("pipeline", [False, True])
def test_original_coordinate_emission_selects_square_orientation(
    pipeline: bool,
) -> None:
    original = chained_tcgen_stage.stage_geometry
    seen = []

    def record(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if result is not None and kwargs.get("transpose_square"):
            seen.append(result)
        return result

    with patch.object(chained_tcgen_stage, "stage_geometry", record):
        selected = _source(pipeline=pipeline)
    assert any(g.logical[:2] == (64, 64) and g.transpose for g in seen)
    assert "tcgen05" in selected and selected != _source(False, pipeline=pipeline)


@pytest.mark.parametrize("value", [1, "transpose", None])
def test_orientation_requires_boolean(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        bound = _square_intermediate._bind_isolated(_args())
        bound.config_spec.normalized_config(_config().config | {_KEY: value})


def test_orientation_requires_m64_and_default_normalizes_away() -> None:
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(_args())
        assert _KEY not in bound.config_spec.normalized_config(_config(False)).config
        with pytest.raises(exc.InvalidConfig, match="grouped M64"):
            bound.config_spec.normalized_config(
                _config().config | {"cute_chained_loop_mma_rows": 128}
            )


def test_orientation_seed_preserves_parents_and_overrides() -> None:
    with _cpu_codegen():
        bound = _square_intermediate._bind_isolated(_args())
        assert bound.host_function is not None
        with bound.env:
            parents = [_config(False, pipeline=False), _config(False)]
            seeded = CuteChainedMatmulHeuristic._with_m64_transpose_seed(
                bound.env, bound.host_function.device_ir, parents
            )
            assert seeded[:-1] == parents and len(seeded) == len(parents) + 1
            assert seeded[-1].config == parents[-1].config | {_KEY: True}
            with patch.dict(
                bound.env.settings.autotune_config_overrides, {_KEY: False}
            ):
                assert (
                    CuteChainedMatmulHeuristic._with_m64_transpose_seed(
                        bound.env, bound.host_function.device_ir, parents
                    )
                    == parents
                )


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_final_storage_veto_remains_active(fault: str) -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pipeline", [False, True])
def test_gpu_orientation_preserves_recurrence(
    dtype: torch.dtype, pipeline: bool
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = list(_args("cuda", dtype=dtype, steps=3))
    generator = torch.Generator(device="cuda").manual_seed(3105)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, device="cuda", generator=generator) * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(3):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (raw - right[:, step].double() @ entering).to(dtype).double()
        solved = (transform[:, step].double() @ corrected).to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ solved + cross).to(dtype)
        state = (
            update[:, step].double() @ solved
            + state * gate[:, step, -1, None, None].double()
        )
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        result = _square_intermediate._bind_isolated(tuple(args)).compile_config(
            _config(enabled, pipeline=pipeline)
        )(*args)
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
