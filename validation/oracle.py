"""CPU formula and comparison only; never an Ascend kernel implementation."""
import numpy as np


def validate_inputs(raw, priors):
    if raw.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise TypeError("raw_pred must be float16/float32")
    if raw.ndim != 3 or raw.shape[2] != 6:
        raise ValueError("raw_pred must be [B,N,6]")
    b, n, _ = raw.shape
    if not 1 <= b <= 8 or not 1 <= n <= 32768:
        raise ValueError("B or N outside problem bounds")
    if priors.dtype != np.dtype("float32") or priors.shape != (n, 5):
        raise ValueError("priors must be float32 [N,5]")
    if not raw.flags.c_contiguous or not priors.flags.c_contiguous:
        raise ValueError("inputs must be contiguous ND")
    if not np.isfinite(raw).all() or not np.isfinite(priors).all():
        raise ValueError("inputs must be finite")
    if (priors[:, :2] < 0).any() or (priors[:, 2:] <= 0).any():
        raise ValueError("grid >= 0; stride/anchors > 0")


def _decode(raw, priors, dtype, stable):
    r, p = raw.astype(dtype), priors.astype(dtype)
    one, two, half = dtype(1), dtype(2), dtype(0.5)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if stable:
            # No exp(+large): independent float64 diagnostic of saturation.
            e = np.exp(-np.abs(r))
            s = np.where(r >= 0, one / (one + e), e / (one + e))
        else:
            # Preserve problem's float32 operation order, including overflow
            # of exp(-z) to +inf and hence a zero sigmoid for very negative z.
            s = one / (one + np.exp(-r))
        stride = p[None, :, 2]
        x = (two * s[..., 0] - half + p[None, :, 0]) * stride
        y = (two * s[..., 1] - half + p[None, :, 1]) * stride
        w0, h0 = two * s[..., 2], two * s[..., 3]
        w = (w0 * w0) * p[None, :, 3] * stride
        h = (h0 * h0) * p[None, :, 4] * stride
        score = s[..., 4] * s[..., 5]
    return np.stack((x, y, w, h, score), axis=-1)


def reference_f32(raw, priors):
    validate_inputs(raw, priors)
    return _decode(raw, priors, np.float32, False)


def reference(raw, priors):
    with np.errstate(over="ignore", under="ignore"):
        return reference_f32(raw, priors).astype(raw.dtype)


def stable_reference_f64(raw, priors):
    validate_inputs(raw, priors)
    return _decode(raw, priors, np.float64, True)


def scalar_reference(raw, priors):
    """Independent scalar indexing/order oracle for small correctness cases."""
    validate_inputs(raw, priors)
    f = np.float32
    b_count, n_count, _ = raw.shape
    result = np.empty((b_count, n_count, 5), np.float32)
    with np.errstate(over="ignore", under="ignore"):
        for b in range(b_count):
            for n in range(n_count):
                tx, ty, tw, th, obj, fire = [f(v) for v in raw[b, n]]
                gx, gy, stride, aw, ah = priors[n]
                def sig(x):
                    return f(1) / (f(1) + np.exp(-x))
                sx, sy, sw, sh = sig(tx), sig(ty), sig(tw), sig(th)
                ww, hh = f(2) * sw, f(2) * sh
                result[b, n] = [(f(2) * sx - f(.5) + gx) * stride,
                                (f(2) * sy - f(.5) + gy) * stride,
                                ww * ww * aw * stride, hh * hh * ah * stride,
                                sig(obj) * sig(fire)]
    return result.astype(raw.dtype)


def compare_output(got, raw, priors):
    """fp16 tolerance compares against fp32 calculation CAST TO fp16."""
    want = reference(raw, priors)
    if not isinstance(got, np.ndarray) or got.shape != want.shape:
        return {"passed": False, "reason": "output shape/type mismatch"}
    if got.dtype != raw.dtype or not got.flags.c_contiguous:
        return {"passed": False, "reason": "output dtype/layout mismatch"}
    tol = 3e-3 if raw.dtype == np.float16 else 1e-5
    g, w = got.astype(np.float64), want.astype(np.float64)
    finite = np.isfinite(w)
    with np.errstate(invalid="ignore"):
        delta = np.abs(g - w)
        limit = tol + tol * np.abs(w)
        ok = np.where(finite, np.isfinite(g) & (delta <= limit), g == w)
    # NaN is never accepted. Inputs are finite, but output representability
    # is not bounded by the problem. Matching signed infinities are reported.
    ok &= ~np.isnan(g) & ~np.isnan(w)
    # 学习｜NaN 预填用来发现漏写，但诊断统计不能再产生 NaN/Inf；否则
    # allow_nan=False 的报告保存会二次失败，反而丢失最需要保留的失败现场。
    # maxAbsFinite 只描述双方均有限的元素；非有限候选另计数，绝不忽略其失败。
    finite_pairs = finite & np.isfinite(g)
    finite_delta = delta[finite_pairs]
    return {"passed": bool(ok.all()), "atol": tol, "rtol": tol,
            "mismatches": int((~ok).sum()),
            "maxAbsFinite": float(finite_delta.max()) if finite_delta.size else None,
            "outputNonfinite": int((~np.isfinite(g)).sum()),
            "nonfiniteActualAtFiniteReference": int((finite & ~np.isfinite(g)).sum()),
            "referenceNonfinite": int((~finite).sum())}
