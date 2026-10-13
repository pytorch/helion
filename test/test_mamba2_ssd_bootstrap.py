from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from benchmarks.cute.mamba2_ssd_bootstrap import NoLegalPartner
from benchmarks.cute.mamba2_ssd_bootstrap import bootstrap_partner
from benchmarks.cute.mamba2_ssd_inputs import sha
import pytest
import torch
from torch._inductor.runtime.triton_compat import OutOfResources

from helion import exc
from helion._compiler.cute.backend import CuteBackend
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.runtime.config import Config


class Bound:
    def __init__(self, behavior=None, seeds=(2, 3, 4, 5)):
        self.behavior = behavior or {}
        self.seeds = seeds
        self.calls = []
        self.seed_queries = 0
        self.env = SimpleNamespace(process_group_name=None)
        self.config_spec = SimpleNamespace(
            default_config=lambda: Config(block_sizes=[1]),
            create_config_generation=self.generator,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: sha(fn.__code__.co_filename),
                classify_autotune_exception=lambda e: (
                    CuteBackend.classify_autotune_exception(None, e)
                ),
            ),
        )

    def generator(self):
        self.seed_queries += 1

        def pairs(log):
            log("original transfer log")
            return [(None, Config(block_sizes=[i])) for i in self.seeds]

        return SimpleNamespace(seed_flat_config_pairs=pairs)

    def config_envelope(self, config):
        return SimpleNamespace(
            to_dict=lambda: {"config": config.config, "CPU_MOCK": True}
        )

    def compile_config(self, config):
        value = config.block_sizes[0]
        self.calls.append(("compile", value))
        behavior = self.behavior.get(value, "ok")
        if isinstance(behavior, BaseException):
            raise behavior

        def compiled(*args):
            self.calls.append(("execute", value))
            if behavior == "resource":
                raise OutOfResources(2, 1, "registers")
            if behavior == "bad":
                return args[0] + 100
            if behavior == "write":
                args[0].add_(1)
            if behavior == "assert":
                raise AssertionError("unexpected executor assertion")
            if callable(behavior):
                return behavior(*args)
            return args[0] * 2

        return compiled


def validate(args, result):
    torch.testing.assert_close(result, args[0] * 2, atol=0, rtol=0)


def run(bound, tmp_path, **kwargs):
    return bootstrap_partner(
        bound,
        (torch.arange(4, dtype=torch.float32),),
        validate_result=validate,
        synchronize=lambda: None,
        output=tmp_path / "bootstrap",
        **kwargs,
    )


def test_default_first_no_seed_query_and_retained_real_output(tmp_path):
    bound = Bound()
    member = run(bound, tmp_path)
    assert bound.seed_queries == 0 and bound.calls == [("compile", 1), ("execute", 1)]
    assert member.result is not member.arguments[0]
    assert member.receipt["native_validation"] == "pending_cpu_standin_only"
    assert member.request.source_hash == sha(__file__)
    assert not (tmp_path / "bootstrap/SEED_ORDER.json").exists()


def test_original_order_known_compile_native_oracle_and_raw_failures(tmp_path):
    bound = Bound(
        {
            1: exc.BackendUnsupported("cute", "toy config"),
            2: "resource",
            3: "bad",
            4: "write",
        }
    )
    member = run(bound, tmp_path)
    assert [v for phase, v in bound.calls if phase == "compile"] == [1, 2, 3, 4, 5]
    record = json.loads((tmp_path / "bootstrap/RESULT.json").read_text())
    assert [r["phase"] for r in record["attempts"][:-1]] == [
        "compile",
        "execute",
        "validate",
        "validate",
    ]
    assert all(
        (tmp_path / "bootstrap" / f"attempt-{i:04d}" / "ERROR.txt").exists()
        for i in range(4)
    )
    assert member.request.config.block_sizes == [5]
    assert torch.equal(member.arguments[0], torch.arange(4, dtype=torch.float32))
    assert [
        c["block_sizes"][0]
        for c in json.loads((tmp_path / "bootstrap/SEED_ORDER.json").read_text())[
            "configs"
        ]
    ] == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "error",
    [
        TypeError("compiler bug"),
        AssertionError("compiler assertion"),
        KeyboardInterrupt(),
        asyncio.CancelledError(),
    ],
)
def test_unexpected_or_cancelled_compile_propagates(tmp_path, error):
    bound = Bound({1: error})
    with pytest.raises(type(error)):
        run(bound, tmp_path)
    assert bound.seed_queries == 0
    assert (
        json.loads((tmp_path / "bootstrap/RESULT.json").read_text())["status"]
        == "aborted"
    )


def test_executor_assert_is_not_numerical_rejection(tmp_path):
    bound = Bound({1: "assert"})
    with pytest.raises(AssertionError, match="executor"):
        run(bound, tmp_path)
    assert bound.seed_queries == 0


@pytest.mark.parametrize(
    "message",
    [
        "ordinary DSL config failure",
        "CUDA error: an illegal memory access was encountered",
    ],
)
def test_existing_backend_classification_and_sticky_error(tmp_path, message):
    NativeError = type(
        "NativeError", (RuntimeError,), {"__module__": "cutlass.testing"}
    )
    bound = Bound({1: NativeError(message)})
    if "illegal" in message:
        with pytest.raises(NativeError):
            run(bound, tmp_path)
        assert bound.seed_queries == 0
    else:
        assert run(bound, tmp_path).request.config.block_sizes == [2]


def test_validation_cancellation_assertion_propagates(tmp_path):
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 4:
            raise AssertionError("cancel now")

    bound = Bound()
    with pytest.raises(AssertionError, match="cancel"):
        run(bound, tmp_path, check_cancelled=cancel)
    assert bound.seed_queries == 0


def test_empty_pool_exhausts_without_random_or_timing(tmp_path):
    bound = Bound({1: exc.InvalidConfig("no legal default")}, seeds=())
    with pytest.raises(NoLegalPartner):
        run(bound, tmp_path)
    result = json.loads((tmp_path / "bootstrap/RESULT.json").read_text())
    assert (
        result["status"] == "exhausted"
        and len(result["attempts"]) == 1
        and not result["timed_or_ranked"]
    )


def test_readonly_aliases_and_contiguous_writable_guards(tmp_path):
    owner = torch.full((16,), 9, dtype=torch.float32)
    output = owner[4:8]
    guard = owner[:4]

    def operation(out, read):
        assert out.untyped_storage()._cdata == read.untyped_storage()._cdata
        out.fill_(3)
        return out

    def check(args, result):
        assert result is args[0] and torch.equal(args[0], torch.full((4,), 3.0))
        assert torch.equal(args[1], torch.full((4,), 9.0))

    bound = Bound({1: operation})
    member = bootstrap_partner(
        bound,
        (output, guard),
        validate_result=check,
        synchronize=lambda: None,
        output=tmp_path / "bootstrap",
        writable_arg_indices=(0,),
    )
    assert torch.equal(owner, torch.full((16,), 9.0))
    assert member.arguments[0].storage_offset() == 4
    assert (
        member.arguments[0].untyped_storage()._cdata != owner.untyped_storage()._cdata
    )


def test_writable_overrun_is_rejected_and_next_attempt_starts_pristine(tmp_path):
    owner = torch.full((16,), 9.0)
    view = owner[4:8]

    def bad(out):
        allwords = torch.empty(0).set_(out.untyped_storage(), 0, (16,), (1,))
        allwords[0] = 0
        return out

    def good(out):
        out.fill_(3)
        return out

    def check(args, result):
        assert result is args[0] and torch.equal(result, torch.full((4,), 3.0))

    bound = Bound({1: bad, 2: good})
    member = bootstrap_partner(
        bound,
        (view,),
        validate_result=check,
        synchronize=lambda: None,
        output=tmp_path / "bootstrap",
        writable_arg_indices=(0,),
    )
    assert member.request.config.block_sizes == [2] and torch.equal(
        owner, torch.full((16,), 9.0)
    )


def test_source_mutation_aborts_instead_of_retry(tmp_path):
    raw = torch.arange(4, dtype=torch.float32)

    def corrupt(*args):
        raw.add_(1)
        return args[0] * 2

    bound = Bound({1: corrupt})
    with pytest.raises(BenchmarkObjectiveError, match="Readonly argument bytes"):
        bootstrap_partner(
            bound,
            (raw,),
            validate_result=validate,
            synchronize=lambda: None,
            output=tmp_path / "bootstrap",
        )
    assert bound.seed_queries == 0


def test_distributed_retry_is_explicitly_unsupported(tmp_path):
    bound = Bound()
    bound.env.process_group_name = "group"
    with pytest.raises(BenchmarkObjectiveError, match="single rank"):
        run(bound, tmp_path)
    assert bound.calls == []


def test_noncontiguous_writable_view_is_rejected(tmp_path):
    bound = Bound()
    with pytest.raises(BenchmarkObjectiveError, match="contiguous"):
        bootstrap_partner(
            bound,
            (torch.arange(8.0)[::2],),
            validate_result=validate,
            synchronize=lambda: None,
            output=tmp_path / "bootstrap",
            writable_arg_indices=(0,),
        )
    assert bound.calls == []


@pytest.mark.parametrize("where", ["source", "private"])
def test_same_abi_storage_replacement_is_not_hidden(tmp_path, where):
    raw = torch.arange(4.0)

    def replace(value):
        target = raw if where == "source" else value
        target.set_(target.clone())
        return value * 2

    bound = Bound({1: replace})
    if where == "source":
        with pytest.raises(BenchmarkObjectiveError, match="owner"):
            bootstrap_partner(
                bound,
                (raw,),
                validate_result=validate,
                synchronize=lambda: None,
                output=tmp_path / "bootstrap",
            )
        assert bound.seed_queries == 0
    else:
        member = bootstrap_partner(
            bound,
            (raw,),
            validate_result=validate,
            synchronize=lambda: None,
            output=tmp_path / "bootstrap",
        )
        assert member.request.config.block_sizes == [2]
        assert (
            json.loads((tmp_path / "bootstrap/attempt-0000/RESULT.json").read_text())[
                "phase"
            ]
            == "validate"
        )


def test_unexpected_seed_generation_error_is_not_rejection(tmp_path):
    bound = Bound({1: exc.InvalidConfig("default invalid")})

    def bad():
        raise KeyError("seed generator defect")

    bound.config_spec.create_config_generation = bad
    with pytest.raises(KeyError):
        run(bound, tmp_path)
    result = json.loads((tmp_path / "bootstrap/RESULT.json").read_text())
    assert result["status"] == "aborted" and result["phase"] == "candidate_generation"


def test_callback_bug_is_not_a_config_decline(tmp_path):
    def broken(args, result):
        raise TypeError("callback implementation bug")

    bound = Bound()
    with pytest.raises(TypeError):
        bootstrap_partner(
            bound,
            (torch.arange(4.0),),
            validate_result=broken,
            synchronize=lambda: None,
            output=tmp_path / "bootstrap",
        )
    assert bound.seed_queries == 0
