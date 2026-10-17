"""Deterministic public-spec cases; Mock shapes are hypotheses, not Judge cases."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    batch: int
    n: int
    dtype: str
    seed: int
    distribution: str = "normal"
    hypothesis: bool = False

    def metadata(self):
        return asdict(self)


def cases(group, tier="quick"):
    if group == "coverage":
        # 704 and 8*704 bracket current tile and balanced-core tile transitions.
        ns = [1, 33, 703, 704, 705, 5631, 5632, 5633, 32768]
        bs = [1, 3, 8]
        if tier == "full":
            ns = [1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
                  127, 128, 129, 511, 512, 513, 703, 704, 705,
                  1023, 1024, 1025, 4095, 4096, 4097, 5631, 5632,
                  5633, 16383, 16384, 16385, 32767, 32768]
            bs = list(range(1, 9))
        return [Case(f"coverage-b{b}-n{n}-{d}", group, b, n, d, 1000 + b * 32769 + n)
                for b in bs for n in ns for d in ("float16", "float32")]
    if group == "correctness":
        patterns = ["zeros", "field_tags", "saturation", "near_zero",
                    "cancellation", "half_rounding", "tiny_priors", "mixed_scale"]
        return [Case(f"correctness-{p}-{d}", group, 3, 65, d, 20261017 + i, p)
                for i, p in enumerate(patterns) for d in ("float16", "float32")]
    if group == "mock":
        # Public problem 4.2 names B=1/4/8,N=4096/16384/32768. Additional
        # sizes are an explicit local scaling hypothesis, NOT inferred metadata.
        ns = [512, 704, 1408, 4096, 16384, 32768]
        return [Case(f"mock-b{b}-n{n}-{d}", group, b, n, d, 3000 + b * 32769 + n,
                     hypothesis=True)
                for b in (1, 4, 8) for n in ns for d in ("float16", "float32")]
    raise ValueError(group)


def make_inputs(case):
    rng = np.random.default_rng(case.seed)
    shape = (case.batch, case.n, 6)
    raw = rng.normal(0, 3, shape).astype(case.dtype)
    p = np.empty((case.n, 5), dtype=np.float32)
    p[:, :2] = rng.uniform(0, 32, (case.n, 2))
    p[:, 2] = rng.choice([1, 8, 16, 32, 64], case.n)
    p[:, 3:] = rng.uniform(0.125, 8, (case.n, 2))
    pattern = case.distribution
    if pattern == "zeros":
        raw.fill(0)
        p[:] = [0, 0, 32, 2, 2]
    elif pattern == "field_tags":
        n = np.arange(case.n)[None, :, None]
        b = np.arange(case.batch)[:, None, None]
        raw[:] = ((n % 13 - 6) * .2 + b * .3 + np.arange(6) * .11)
        p[:, 0] = np.arange(case.n)
        p[:, 1] = np.arange(case.n)[::-1]
    elif pattern == "saturation":
        bound = np.finfo(np.dtype(case.dtype)).max
        values = np.array([-bound, -1000, -100, -90, -88, -20, 0, 20,
                           88, 90, 100, 1000, bound], dtype=case.dtype)
        raw[:] = np.resize(values, shape)
    elif pattern == "near_zero":
        values = np.array([-1e-5, -1e-7, -0.0, 0.0, 1e-7, 1e-5], dtype=case.dtype)
        raw[:] = np.resize(values, shape)
    elif pattern == "cancellation":
        z = np.float32(-np.log(3))  # 2*sigmoid(z)-.5 approximately zero
        raw[..., :2] = np.resize(np.array([z, np.nextafter(z, -np.inf, dtype=np.float32),
                                         np.nextafter(z, np.inf, dtype=np.float32)]),
                                (case.batch, case.n, 2)).astype(case.dtype)
        p[:, :2] = 0
    elif pattern == "half_rounding":
        vals = np.array([-8, -1, -.1, 0, .1, 1, 8], dtype=np.float16)
        vals = np.concatenate((vals, np.nextafter(vals, np.float16(np.inf)),
                               np.nextafter(vals, np.float16(-np.inf))))
        raw[:] = np.resize(vals, shape).astype(case.dtype)
    elif pattern == "tiny_priors":
        p[:, :2] = 0
        p[:, 2:] = np.float32(1e-5)
    elif pattern == "mixed_scale":
        p[:, 2] = rng.choice([.0001, .5, 1, 64], case.n)
        p[:, 3:] = rng.choice([.0001, .5, 1, 128], (case.n, 2))
    return raw, p
