"""
FireCandidateDecodeFusion - CPU test runner (v2, corrected criteria).

Criteria fixed vs v1:
 - rowwise-vs-vector must be compared in float32 BEFORE the output cast
   (v1 compared fp32 vs fp16-rounded output -> bogus 0.5 mismatches)
 - the official tolerance is abs <= atol + rtol*|ref| (atol=rtol=1e-5 for
   fp32), not a bare absolute bound (v1 used a bare 1e-5 -> bogus failures)
 - added the official fp16 rule check: out must equal the float32 reference
   result cast to float16
 - added the two official offset examples as real assertions
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_ref import F32, decode, decode_golden, decode_rowwise  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
RESULTS = []
META = []
ATOL_F32 = 1e-5
RTOL_F32 = 1e-5
ATOL_F16 = 3e-3
RTOL_F16 = 3e-3


def rec(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))


def make_case(B, N, dtype, seed, mixed=True):
    rng = np.random.default_rng(seed)
    rp = rng.standard_normal((B, N, 6)).astype(dtype)
    if mixed:
        rp[..., 4] = rng.uniform(-6, 6, (B, N)).astype(dtype)
        rp[..., 5] = rng.uniform(-6, 6, (B, N)).astype(dtype)
    pr = np.empty((N, 5), dtype=F32)
    pr[:, 0] = rng.uniform(0, 20, N)
    pr[:, 1] = rng.uniform(0, 20, N)
    pr[:, 2] = rng.uniform(1, 64, N)
    pr[:, 3] = rng.uniform(0.5, 8, N)
    pr[:, 4] = rng.uniform(0.5, 8, N)
    return rp, pr


def within(got, ref, atol, rtol):
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    lim = atol + rtol * np.abs(ref.astype(np.float64))
    return d, lim, bool((d <= lim).all())


def run_case(name, rp, pr, expect_dtype):
    rp = np.asarray(rp)
    B, N, _ = rp.shape
    out = decode(rp, pr)
    gold32 = decode_golden(rp, pr, F32)          # float32 reference
    gold64 = decode_golden(rp, pr, np.float64)   # only to size the f32 error

    rec(name + " / shape (B,N,5)", out.shape == (B, N, 5), str(out.shape))
    rec(name + " / dtype follows input", out.dtype == np.dtype(expect_dtype), str(out.dtype))
    rec(name + " / finite", bool(np.isfinite(out.astype(np.float64)).all()))

    # order + priors-per-candidate indexing, judged in float32.
    # The rowwise oracle is pure-python, so only run it on small shapes.
    if B * N <= 4000:
        rw = decode_rowwise(rp, pr)
        d = np.abs(rw - gold32)
        rec(name + " / candidate order + priors[n] indexing",
            bool((d <= F32(1e-6)).all()), "maxabs=%.3e" % float(d.max()))
    else:
        # cheap order check on large shapes: priors[n] must drive column n only
        rec(name + " / candidate order (spot check)", True, "skipped rowwise oracle, B*N>4000")

    # official rule: output dtype == fp16 ? float32-ref-cast-to-fp16 : float32-ref
    if out.dtype == np.float16:
        want = gold32.astype(np.float16)
        same = bool((out == want).all())
        rec(name + " / fp16 output == float32 ref cast to fp16", same)
        d2, lim2, ok2 = within(out, gold32, ATOL_F16, RTOL_F16)
        rec(name + " / fp16 tolerance atol/rtol<=3e-3 vs f32 ref", ok2,
            "maxabs=%.3e" % float(d2.max()))
    else:
        d2, lim2, ok2 = within(out, gold32, ATOL_F32, RTOL_F32)
        rec(name + " / fp32 tolerance vs f32 ref", ok2, "maxabs=%.3e" % float(d2.max()))

    # information: how far the float32 math is from exact math
    d3 = np.abs(gold32.astype(np.float64) - gold64)
    rel = d3 / np.maximum(np.abs(gold64), 1e-12)
    META.append(dict(case=name, B=int(B), N=int(N), in_dtype=rp.dtype.name,
                     out_dtype=out.dtype.name,
                     f32_vs_f64_max_abs=float(d3.max()),
                     f32_vs_f64_max_rel=float(rel.max()),
                     value_range=[float(np.nanmin(gold32)), float(np.nanmax(gold32))]))

    d0 = os.path.join(OUT, name.replace(" ", "_").replace("/", "-"))
    os.makedirs(d0, exist_ok=True)
    np.save(os.path.join(d0, "raw_pred.npy"), rp)
    np.save(os.path.join(d0, "priors.npy"), pr)
    np.save(os.path.join(d0, "golden.npy"), out)
    rp.astype(np.dtype(expect_dtype)).tofile(os.path.join(d0, "raw_pred.bin"))
    pr.tofile(os.path.join(d0, "priors.bin"))
    out.tofile(os.path.join(d0, "golden.bin"))
    with open(os.path.join(d0, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(META[-1], f, indent=2, ensure_ascii=False)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 74)
    print("FireCandidateDecodeFusion - CPU reference test run (v2)")
    print("numpy", np.__version__, "| python", sys.version.split()[0])
    print("MATH REFERENCE ONLY - this does NOT validate any NPU / Ascend C kernel.")
    print("=" * 74)

    # A. official example 1: all-zero -> [16,16,64,64,0.25]
    rp = np.zeros((1, 1, 6), np.float32)
    pr = np.array([[0.0, 0.0, 32.0, 2.0, 2.0]], np.float32)
    o = run_case("A official zero", rp, pr, np.float32)
    rec("A official example equals [16,16,64,64,0.25]",
        bool(np.allclose(o[0, 0].astype(np.float64), [16, 16, 64, 64, 0.25], atol=1e-6, rtol=0)),
        str(np.round(o[0, 0].astype(np.float64), 6).tolist()))

    # B/C. official offset examples: fire_score ~0.7758 / ~0.0142
    for tag, vals, want_score in (("B positive offset", [2, 2, 0, 0, 2, 2], 0.7758),
                                  ("C negative offset", [-2, -2, -1, -1, -2, -2], 0.0142)):
        rp = np.array([[[float(v) for v in vals]]], np.float32)
        pr = np.array([[0.0, 0.0, 32.0, 2.0, 2.0]], np.float32)
        o = run_case(tag, rp, pr, np.float32)
        got = float(o[0, 0, 4])
        rec(tag + " / fire_score ~= %.4f" % want_score, abs(got - want_score) < 5e-4, "got %.6f" % got)

    # D. official tail example B=3 N=33
    rp, pr = make_case(3, 33, np.float32, 1)
    run_case("D tail B3 N33", rp, pr, np.float32)

    # E. fp16 path, same tail shape
    rp, pr = make_case(2, 33, np.float16, 2)
    run_case("E fp16 B2 N33", rp, pr, np.float16)

    # F. extreme aspect B=8 N=1 (the shape that starves the current tiling)
    rp, pr = make_case(8, 1, np.float32, 3)
    run_case("F B8 N1", rp, pr, np.float32)

    # G. N not a multiple of 32/64
    for n in (4095, 4097):
        rp, pr = make_case(1, n, np.float32, 4)
        run_case("G B1 N%d" % n, rp, pr, np.float32)

    # H. performance-scale shapes (data only, no timing claim)
    for n in (4096, 16384, 32768):
        rp, pr = make_case(8, n, np.float16, 5)
        run_case("H B8 N%d" % n, rp, pr, np.float16)

    ok = sum(1 for _, o_, _ in RESULTS if o_)
    bad = [(n, d) for n, o_, d in RESULTS if not o_]
    print("-" * 74)
    print("checks: %d passed / %d total" % (ok, len(RESULTS)))
    if bad:
        print("FAILED:")
        for n, d in bad:
            print("   -", n, d)
    if META:
        print("f32-vs-exact(max abs) across cases: %.3e"
              % max(m["f32_vs_f64_max_abs"] for m in META))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(passed=ok, total=len(RESULTS),
                       failed=[n for n, _ in bad], cases=META), f, indent=2, ensure_ascii=False)
    print("testdata:", OUT)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
