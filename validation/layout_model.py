"""Address/ownership model matching EXP-016 Init/Process, not a DMA simulator."""
import numpy as np


def partitions(n, cores):
    q, tail = divmod(n, cores)
    for core in range(cores):
        count = q + (core < tail)
        begin = core * (q + 1) if core < tail else tail * (q + 1) + (core - tail) * q
        yield core, begin, count


def check_layout(batch, n, tile=704, cores=8):
    if tile <= 0 or cores <= 0:
        raise ValueError("tile/cores must be positive")
    size, guard = batch * n * 5, 32
    sentinel = np.int64(-987654321)
    memory = np.full(size + 2 * guard, sentinel, dtype=np.int64)
    output = memory[guard:-guard]
    writes = np.zeros((batch, n), dtype=np.uint8)
    segments = 0
    raw_reads, prior_reads = 0, 0
    for core, begin, count in partitions(n, cores):
        assert 0 <= begin <= begin + count <= n
        for base in range(0, count, tile):
            cur = min(tile, count - base)
            n0 = begin + base
            assert n0 * 5 + cur * 5 <= n * 5
            prior_reads += cur * 5  # one priors tile reused across all batches
            for b in range(batch):
                raw_off = (b * n + n0) * 6
                assert raw_off + cur * 6 <= batch * n * 6
                raw_reads += cur * 6
                off, length = (b * n + n0) * 5, cur * 5
                # Every output carries its exact b,n,field identity. No oracle
                # data is reused here; this checks addresses/layout, not math.
                output[off:off + length] = np.arange(off, off + length, dtype=np.int64)
                writes[b, n0:n0 + cur] += 1
                segments += 1
    assert np.array_equal(output, np.arange(size, dtype=np.int64))
    assert (writes == 1).all(), "missing/duplicate output candidate"
    assert (memory[:guard] == sentinel).all() and (memory[-guard:] == sentinel).all()
    assert raw_reads == batch * n * 6 and prior_reads == n * 5
    return {"passed": True, "tile": tile, "cores": cores, "segments": segments,
            "rawElements": raw_reads, "priorElements": prior_reads,
            "outputElements": size, "guardElements": guard,
            "scope": "CPU addresses/ownership only; no physical DMA/event verification"}
