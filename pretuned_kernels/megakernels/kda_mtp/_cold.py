"""Cold-L2 timing with a *clean* flush: a 256 MB read before each one-call
CUDA-graph replay (no dirty lines to write back), systems interleaved,
median of event-timed replays."""
import statistics
import torch

_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = (torch.empty(64 * 1024 * 1024, device="cuda"), torch.empty(1, device="cuda"))
    torch.sum(_FLUSH[0], dim=0, out=_FLUSH[1])


def bench_clean(replays, rep=200):
    for _ in range(3):
        for r in replays:
            _flush(); r()
    torch.cuda.synchronize()
    starts = [[torch.cuda.Event(enable_timing=True) for _ in range(rep)] for _ in replays]
    ends = [[torch.cuda.Event(enable_timing=True) for _ in range(rep)] for _ in replays]
    n = len(replays)
    for i in range(rep):
        order = list(range(n))
        order = order[i % n:] + order[:i % n]
        if (i // n) % 2:
            order = order[::-1]
        for j in order:
            _flush()
            starts[j][i].record()
            replays[j]()
            ends[j][i].record()
    torch.cuda.synchronize()
    # event timestamps are quantized (~0.5 us): a trimmed mean resolves finer
    def tmean(v):
        v = sorted(v)
        k = len(v) // 10
        v = v[k:len(v) - k]
        return sum(v) / len(v)
    return [tmean([s.elapsed_time(e) * 1e3 for s, e in zip(starts[j], ends[j])]) for j in range(n)]


def bench_steady(calls, reps_in_graph=20, rep=50):
    """Per-call time in steady state: one CUDA graph runs ``reps_in_graph`` x
    (clean flush, call); minus a graph of the flushes alone; divided by the
    count (the graph launch amortizes away, as in a whole-model graph)."""
    import torch
    _flush()
    graphs = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            for _ in range(2):
                _flush(); c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for c in list(calls) + [None]:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps_in_graph):
                _flush()
                if c is not None:
                    c()
        graphs.append(g)
    torch.cuda.synchronize()
    times = bench_clean_nf([g.replay for g in graphs], rep)
    base = times[-1]
    return [(t - base) / reps_in_graph for t in times[:-1]], base / reps_in_graph


def bench_clean_nf(replays, rep):
    import torch
    for r in replays:
        r()
    torch.cuda.synchronize()
    starts = [[torch.cuda.Event(enable_timing=True) for _ in range(rep)] for _ in replays]
    ends = [[torch.cuda.Event(enable_timing=True) for _ in range(rep)] for _ in replays]
    for i in range(rep):
        for j in range(len(replays)):
            starts[j][i].record()
            replays[j]()
            ends[j][i].record()
    torch.cuda.synchronize()
    return [statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts[j], ends[j])) for j in range(len(replays))]
