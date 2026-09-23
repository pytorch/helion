"""CPU semantics for the dormant raw-full to converted-full operand helper."""

from __future__ import annotations

import ast
from collections import Counter
import ctypes
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from typing_extensions import Self

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from helion._compiler.cute import tcgen05_operand_pipeline
from helion._compiler.cute.tcgen05_pipeline import make_pipeline_state

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
pipeline = pytest.importorskip("cutlass.pipeline")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with _mock_cuda_unavailable():

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("GPU forbidden in the operand pipeline proof")

        monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
        yield


class _F32:
    def __init__(self, bits: int) -> None:
        self.bits = bits & 0xFFFFFFFF


class _U32(int):
    def __new__(cls, value: int) -> Self:
        return int.__new__(cls, int(value) & 0xFFFFFFFF)

    def __and__(self, value: int) -> _U32:
        return _U32(int(self) & int(value))

    def bitcast(self, dtype: type[_F32]) -> _F32:
        assert dtype is _F32
        return _F32(self)


def _raw_cvt_result(value: _F32) -> _U32:
    # The captured SM100 cvt.rna expansion adds 0x1000 to finite values.
    # Its low bits are not a Float32 storage contract. Deliberately keep them
    # here so omitting the helper's final mask is observable in this oracle.
    bits = value.bits
    return _U32(bits + (0x1000 if (bits & 0x7FFFFFFF) < 0x7F800000 else 0))


def _quantize(bits: int) -> int:
    """Independent exact binary64 rounding of the finite binary32 value."""
    sign = bits & 0x80000000
    value = struct.unpack("!f", struct.pack("!I", bits))[0]
    if math.isnan(value):
        return bits & 0xFFFFE000
    if math.isinf(value) or value == 0:
        return bits
    magnitude = abs(value)
    if magnitude < math.ldexp(1.0, -126):
        rounded = math.floor(math.ldexp(magnitude, 136) + 0.5)
        result = math.ldexp(float(rounded), -136)
    else:
        fraction, exponent = math.frexp(magnitude)
        rounded = math.floor(fraction * 2048 + 0.5)
        result = math.ldexp(float(rounded), exponent - 11)
    if result >= math.ldexp(1.0, 128):
        return sign | 0x7F800000
    return sign | struct.unpack("!I", struct.pack("!f", result))[0]


def _actual_functions(
    cute_namespace: object,
    cutlass_namespace: object,
    pipeline_namespace: object = None,
    mutate: Callable[[ast.Module], None] | None = None,
) -> dict[str, Any]:
    tree = ast.parse(Path(tcgen05_operand_pipeline.__file__).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in tree.body:
        assert isinstance(node, ast.FunctionDef)
        node.decorator_list = []
        node.returns = None
        for arg in (*node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
    if mutate is not None:
        mutate(tree)
    namespace = {
        "cute": cute_namespace,
        "cutlass": cutlass_namespace,
        "pipeline": pipeline_namespace,
        "cast": cast,
    }
    exec(
        compile(ast.fix_missing_locations(tree), "<actual-operand-helper-AST>", "exec"),
        namespace,
    )
    return namespace


def _word_function() -> Callable[[_F32], _F32]:
    return _actual_functions(
        SimpleNamespace(arch=SimpleNamespace(cvt_f32_tf32=_raw_cvt_result)),
        SimpleNamespace(Float32=_F32, Uint32=_U32),
    )["exact_tf32_rna_word"]


@pytest.mark.parametrize("sign", (0, 0x80000000))
@pytest.mark.parametrize(
    "prefix",
    (
        0x0000,
        0x0001,
        0x007F,
        0x0080,
        0x3F00,
        0x3F80,
        0x3FFF,
        0x4000,
        0x7F7F,
        0x7F80,
        0x7FBF,
        0x7FC0,
        0x7FFF,
    ),
)
def test_actual_word_helper_matches_exact_finite_and_nan_contract(
    sign: int, prefix: int
) -> None:
    convert = _word_function()
    for suffix in range(65536):
        bits = sign | (prefix << 16) | suffix
        actual = convert(_F32(bits)).bits
        assert actual == _quantize(bits), hex(bits)
        assert actual & 0x1FFF == 0
        assert convert(_F32(actual)).bits == actual


def test_all_exponent_ties_and_incomplete_word_negative_control() -> None:
    convert = _word_function()
    for sign in (0, 0x80000000):
        for exponent in range(255):
            for mantissa in (0, 0x2000, 0x7FC000, 0x7FE000):
                for residue in (0, 0xFFF, 0x1000, 0x1001, 0x1FFF):
                    bits = sign | (exponent << 23) | (mantissa + residue)
                    assert convert(_F32(bits)).bits == _quantize(bits)
    # An already representable word must stay unchanged. The raw cvt result
    # intentionally exposes unspecified low bits, which a later TMA RN pass
    # could round again if they were stored without the explicit final mask.
    assert int(_raw_cvt_result(_F32(0x3F802000))) == 0x3F803000
    assert convert(_F32(0x3F802000)).bits == 0x3F802000
    assert convert(_F32(0x7F800001)).bits == 0x7F800000
    assert convert(_F32(0xFF800001)).bits == 0xFF800000


class _Memory:
    def __init__(self) -> None:
        self.values: dict[int, int] = {}
        self.active: set[int] = set()
        self.reads: Counter[int] = Counter()
        self.writes: Counter[int] = Counter()


class _Pointer:
    memspace = "shared"
    alignment = 1024

    def __init__(self, memory: _Memory, address: int) -> None:
        self.memory = memory
        self.address = address

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.memory, self.address + 4 * offset)

    def load(self) -> _F32:
        assert self.address in self.memory.active, ("inactive load", self.address)
        self.memory.reads[self.address] += 1
        return _F32(self.memory.values[self.address])

    def store(self, value: _F32) -> None:
        assert self.address in self.memory.active, ("inactive store", self.address)
        self.memory.writes[self.address] += 1
        self.memory.values[self.address] = value.bits


@dataclass
class _Tensor:
    element_type = _F32
    iterator: _Pointer


def _run_stage(
    *,
    a_words: int,
    b_words: int,
    a_stride: int,
    b_stride: int,
    stages: int,
    stage: int,
    converter_warps: int = 1,
    mutate: Callable[[ast.Module], None] | None = None,
) -> None:
    memory = _Memory()
    current = SimpleNamespace(warp=0, lane=0, waited=False, fenced=False)
    publications: list[int] = []
    bases = (4096, 131072)
    footprints = tuple(
        {(base + index * 4) for index in range((stages - 1) * stride + words)}
        for base, words, stride in zip(
            bases, (a_words, b_words), (a_stride, b_stride), strict=True
        )
    )
    assert footprints[0].isdisjoint(footprints[1])
    all_addresses = footprints[0] | footprints[1]
    values = (0, 0x80000000, 0x3F801000, 0xBF801000, 0x7F800001, 0x7FFFFFFF)
    memory.values = {
        address: values[index % len(values)]
        for index, address in enumerate(sorted(all_addresses))
    }
    original = dict(memory.values)
    memory.active = {
        base + 4 * (stage * stride + index)
        for base, words, stride in zip(
            bases, (a_words, b_words), (a_stride, b_stride), strict=True
        )
        for index in range(words)
    }

    def copy(atom: object, source: object, destination: object) -> None:
        assert current.waited and not current.fenced
        assert atom == ("universal", _F32, 128)
        if isinstance(source, _Tensor):
            assert isinstance(destination, list)
            assert source.iterator.address % 16 == 0
            for index in range(4):
                destination[index] = (source.iterator + index).load()
        else:
            assert isinstance(source, list) and isinstance(destination, _Tensor)
            assert destination.iterator.address % 16 == 0
            for index in range(4):
                (destination.iterator + index).store(source[index])

    def assume(value: int, *, divby: int) -> int:
        assert value % divby == 0
        return value

    def wait(state: SimpleNamespace) -> None:
        assert state.index == stage
        current.waited = True

    def fence(*, kind: str, space: str) -> None:
        assert (kind, space) == ("async.shared", "cta")
        assert current.waited
        current.fenced = True

    def commit(state: SimpleNamespace) -> None:
        assert state.index == stage and current.fenced
        publications.append(current.warp * 32 + current.lane)

    def forbidden_release(state: object) -> None:
        raise AssertionError("conversion cannot recycle the UMMA-empty stage")

    cutlass_mock = SimpleNamespace(
        Float32=_F32,
        Uint32=_U32,
        range=lambda stop, unroll: range(stop),
        range_constexpr=range,
    )
    cute_mock = SimpleNamespace(
        Pointer=_Pointer,
        AddressSpace=SimpleNamespace(smem=_Pointer.memspace),
        arch=SimpleNamespace(
            cvt_f32_tf32=_raw_cvt_result,
            lane_idx=lambda: current.lane,
            fence_proxy=fence,
        ),
        recast_ptr=lambda pointer, dtype: pointer,
        make_tensor=lambda pointer, layout: _Tensor(pointer),
        make_layout=lambda size: size,
        make_rmem_tensor=lambda size, dtype: [None] * size,
        make_copy_atom=lambda op, dtype, num_bits_per_copy: (
            op,
            dtype,
            num_bits_per_copy,
        ),
        nvgpu=SimpleNamespace(CopyUniversalOp=lambda: "universal"),
        copy=copy,
        assume=assume,
    )
    functions = _actual_functions(cute_mock, cutlass_mock, mutate=mutate)
    raw = SimpleNamespace(consumer_wait=wait, consumer_release=forbidden_release)
    converted = SimpleNamespace(
        producer_commit=commit,
        consumer_release=forbidden_release,
        producer_acquire=forbidden_release,
    )
    tensors = tuple(_Tensor(_Pointer(memory, base)) for base in bases)
    for lane in range(32 * converter_warps):
        current.warp, current.lane = divmod(lane, 32)
        current.waited, current.fenced = False, False
        functions["convert_and_publish_operand_stage"](
            raw,
            converted,
            SimpleNamespace(index=stage),
            *tensors,
            a_words,
            a_stride,
            16,
            b_words,
            b_stride,
            16,
            converter_warps,
            current.warp,
        )
    assert publications == list(range(32 * converter_warps))
    assert set(memory.reads) == set(memory.writes) == memory.active
    assert set(memory.reads.values()) == set(memory.writes.values()) == {1}
    for address in all_addresses:
        expected = (
            _quantize(original[address])
            if address in memory.active
            else original[address]
        )
        assert memory.values[address] == expected


@pytest.mark.parametrize(
    "a_words,b_words,a_stride,b_stride,stages",
    ((4096, 2048, 4096, 2048, 6), (1536, 768, 1664, 896, 3), (128, 256, 256, 512, 1)),
)
def test_actual_stage_ast_covers_each_operand_once_and_preserves_gaps(
    a_words: int, b_words: int, a_stride: int, b_stride: int, stages: int
) -> None:
    for stage in range(stages):
        _run_stage(
            a_words=a_words,
            b_words=b_words,
            a_stride=a_stride,
            b_stride=b_stride,
            stages=stages,
            stage=stage,
        )


@pytest.mark.parametrize("fault", ("missing_fence", "early_release"))
def test_publication_or_recycling_fault_is_observable(fault: str) -> None:
    def mutate(tree: ast.Module) -> None:
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "convert_and_publish_operand_stage"
        )
        if fault == "missing_fence":
            function.body.pop(-2)
        else:
            function.body.append(ast.parse("converted.consumer_release(state)").body[0])

    with pytest.raises(AssertionError):
        _run_stage(
            a_words=128,
            b_words=256,
            a_stride=128,
            b_stride=256,
            stages=3,
            stage=1,
            mutate=mutate,
        )


@pytest.mark.parametrize("stages", (1, 2, 3, 6))
@pytest.mark.parametrize("defer_sync", (False, True))
@pytest.mark.parametrize("converter_warps", (1, 2, 4, 8))
def test_factory_allocates_only_full_barriers_and_retains_the_old_empty_object(
    stages: int, defer_sync: bool, converter_warps: int
) -> None:
    one = object()
    allocations: list[tuple[object, int, int]] = []
    syncs: list[str] = []
    empty = object()
    raw = SimpleNamespace(
        num_stages=stages,
        cta_group=one,
        producer_mask=None,
        consumer_mask=None,
        sync_object_empty=empty,
    )

    def allocate(dtype: object, count: int, *, alignment: int) -> object:
        allocations.append((dtype, count, alignment))
        return object()

    def make_full(
        storage: object,
        count: int,
        agent: tuple[object, object],
        *,
        name: str,
        phase: str,
    ) -> SimpleNamespace:
        assert phase == "full"
        return SimpleNamespace(storage=storage, count=count, agent=agent)

    fake_pipeline = SimpleNamespace(
        PipelineAsync=SimpleNamespace(_make_sync_object=make_full),
        PipelineOp=SimpleNamespace(AsyncThread="async_thread"),
        CooperativeGroup=lambda agent, count: (agent, count),
        Agent=SimpleNamespace(Thread="thread"),
        PipelineAsyncUmma=lambda *args: args,
    )
    fake_cute = SimpleNamespace(
        arch=SimpleNamespace(
            alloc_smem=allocate,
            mbarrier_init_fence=lambda: syncs.append("init_fence"),
            sync_threads=lambda: syncs.append("cta_sync"),
        ),
        nvgpu=SimpleNamespace(
            tcgen05=SimpleNamespace(CtaGroup=SimpleNamespace(ONE=one))
        ),
    )
    namespace = _actual_functions(
        fake_cute, SimpleNamespace(Int64="i64", const_expr=bool), fake_pipeline
    )
    result = namespace["make_converted_operand_pipeline"](
        raw, stages, converter_warps, defer_sync
    )
    assert result[1] is empty
    assert result[0].agent == ("async_thread", ("thread", 32 * converter_warps))
    assert result[0].count == result[2] == stages
    assert allocations == [("i64", stages, 8)]
    assert syncs == ([] if defer_sync else ["init_fence", "cta_sync"])
    for unproved_count in (0, True, 3, 6, 16):
        with pytest.raises(AssertionError):
            namespace["make_converted_operand_pipeline"](
                raw, stages, unproved_count, defer_sync
            )


def _sdk_method(cls: type[object], name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(inspect.getfile(cls)).read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )
    function = next(
        node
        for node in definition.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    function.returns = None
    for arg in (*function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    return function


def test_actual_sdk_release_keeps_hardware_umma_completion_on_original_empty_ring() -> (
    None
):
    old = _sdk_method(pipeline.PipelineTmaUmma, "consumer_release")
    new = _sdk_method(pipeline.PipelineAsyncUmma, "consumer_release")
    assert ast.dump(old) == ast.dump(new)
    tree = ast.Module(
        body=[
            new,
            _sdk_method(pipeline.PipelineAsync, "producer_commit"),
            _sdk_method(pipeline.PipelineAsync, "consumer_wait"),
            _sdk_method(pipeline.PipelineAsync, "producer_acquire"),
        ],
        type_ignores=[],
    )

    def if_generate(
        condition: bool, body: Callable[[], None], **kwargs: object
    ) -> None:
        if condition:
            body()

    namespace: dict[str, Any] = {"if_generate": if_generate}
    exec(
        compile(
            ast.fix_missing_locations(tree), "<actual-SDK-pipeline-methods>", "exec"
        ),
        namespace,
    )
    calls: list[tuple[object, ...]] = []
    empty = SimpleNamespace(
        wait=lambda index, phase, **kwargs: calls.append(
            ("old_empty_wait", index, phase)
        ),
        arrive=lambda index, mask, group, **kwargs: calls.append(
            ("old_empty_umma_arrive", index, mask, group)
        ),
    )
    full = SimpleNamespace(
        wait=lambda index, phase, **kwargs: calls.append(
            ("converted_full_wait", index, phase)
        ),
        arrive=lambda index, mask, **kwargs: calls.append(
            ("converted_full_arrive", index, mask)
        ),
    )
    converted = SimpleNamespace(
        sync_object_empty=empty,
        sync_object_full=full,
        producer_mask=None,
        consumer_mask=None,
        cta_group=cute.nvgpu.tcgen05.CtaGroup.ONE,
    )
    state = SimpleNamespace(index=5, phase=1)
    for operation in (
        "producer_commit",
        "consumer_wait",
        "consumer_release",
        "producer_acquire",
    ):
        namespace[operation](converted, state)
    assert calls == [
        ("converted_full_arrive", 5, None),
        ("converted_full_wait", 5, 1),
        ("old_empty_umma_arrive", 5, None, cute.nvgpu.tcgen05.CtaGroup.ONE),
        ("old_empty_wait", 5, 1),
    ]


@dataclass
class _RingSlot:
    status: str = "empty"
    ordinal: int = -1
    published: int = 0


def _ring_schedule(
    stages: int, counts: tuple[int, ...], seed: int, fault: str = ""
) -> int:
    """Finite ordering model, linked above to actual helper/SDK method bodies.

    DMA and MMA completion may lag independently. It covers only A/B lifetime;
    persistent scheduling and accumulator/epilogue composition remain deferred.
    """
    rng = random.Random(seed)
    total = sum(counts)
    slots = [_RingSlot() for _ in range(stages)]
    producer = converter = mma = complete = events = 0
    pending: list[int] = []
    while complete < total:
        options: list[tuple[str, int]] = []
        if producer < total and (
            slots[producer % stages].status == "empty"
            or fault == "recycle_on_publication"
            and slots[producer % stages].status == "converted"
        ):
            options.append(("produce", producer))
        options.extend(
            ("dma", i) for i, slot in enumerate(slots) if slot.status == "raw_pending"
        )
        if converter < total and slots[converter % stages].status in (
            "raw_ready",
            "converting",
        ):
            options.append(("publish_lane", converter))
        if mma < total and slots[mma % stages].status == "converted":
            options.append(("mma_issue", mma))
        options.extend(("mma_complete", ordinal) for ordinal in pending)
        assert options, "unreachable phase or lost publication"
        event, ordinal = rng.choice(options)
        slot = slots[ordinal % stages]
        events += 1
        assert events <= total * 40 + 10
        if event == "produce":
            assert slot.status == "empty", "storage recycled before UMMA completion"
            slot.status, slot.ordinal, slot.published = "raw_pending", ordinal, 0
            producer += 1
        elif event == "dma":
            slot.status = "raw_ready"
        elif event == "publish_lane":
            assert slot.ordinal == ordinal
            slot.status = "converting"
            slot.published += 1
            if slot.published == (1 if fault == "warp_count_for_lane_barrier" else 32):
                slot.status = "converted"
                converter += 1
        elif event == "mma_issue":
            assert slot.ordinal == ordinal and slot.published == 32
            slot.status = "mma_reading"
            pending.append(ordinal)
            mma += 1
        else:
            assert event == "mma_complete"
            assert slot.status == "mma_reading" and slot.ordinal == ordinal
            slot.status = "empty"
            pending.remove(ordinal)
            complete += 1
        assert 0 <= complete <= mma <= converter <= producer <= total
    assert producer == converter == mma == complete == total
    assert all(slot.status == "empty" for slot in slots)
    return events


@pytest.mark.parametrize("stages", (1, 2, 3, 6))
@pytest.mark.parametrize("counts", ((), (0, 0), (0, 1, 0, 7, 2, 0, 19), (4,) * 31))
def test_delayed_dma_and_umma_completion_preserve_the_same_ring_lifetime(
    stages: int, counts: tuple[int, ...]
) -> None:
    for seed in range(8):
        _ring_schedule(stages, counts, seed)


@pytest.mark.parametrize(
    "fault", ("warp_count_for_lane_barrier", "recycle_on_publication")
)
def test_wrong_arrival_units_and_early_recycling_are_detected(fault: str) -> None:
    with pytest.raises(AssertionError):
        _ring_schedule(1, (0, 3, 0, 3), 0, fault)


@cute.jit
def _record_phases(
    stages: cutlass.Constexpr[int],
    k_counts: cutlass.Constexpr[tuple[int, ...]],
    address: cutlass.Int64,
) -> None:
    producer = make_pipeline_state(pipeline.PipelineUserType.Producer, stages)
    converter = make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
    consumer = make_pipeline_state(pipeline.PipelineUserType.Consumer, stages)
    output = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((len(k_counts) * 9,), stride=(1,)),
    )
    for tile in cutlass.range_constexpr(len(k_counts)):
        for _step in cutlass.range_constexpr(k_counts[tile]):
            producer.advance()
            converter.advance()
            consumer.advance()
        output[tile * 9] = producer.index
        output[tile * 9 + 1] = producer.phase
        output[tile * 9 + 2] = producer.count
        output[tile * 9 + 3] = converter.index
        output[tile * 9 + 4] = converter.phase
        output[tile * 9 + 5] = converter.count
        output[tile * 9 + 6] = consumer.index
        output[tile * 9 + 7] = consumer.phase
        output[tile * 9 + 8] = consumer.count


@pytest.mark.parametrize("stages", (1, 2, 3, 6))
def test_actual_helion_sdk_states_keep_phase_across_zero_k_and_wrap(
    stages: int,
) -> None:
    counts = (0, 1, 0, 2, 8, 0, 1, 19, 0)
    storage = (ctypes.c_int32 * (len(counts) * 9))()
    address = ctypes.addressof(storage)
    # This is a CPU host JIT over integer states and ordinary CPU memory. It
    # contains no GPU kernel or launch, and CUDA initialization is forbidden.
    compiled = cute.compile(
        _record_phases,
        stages,
        counts,
        cutlass.Int64(address),
        options="--gpu-arch sm_100a",
    )
    compiled(address)
    total = 0
    for tile, count in enumerate(counts):
        total += count
        index, phase = total % stages, (total // stages) % 2
        assert list(storage[tile * 9 : (tile + 1) * 9]) == [
            index,
            phase ^ 1,
            total,
            index,
            phase,
            total,
            index,
            phase,
            total,
        ]
