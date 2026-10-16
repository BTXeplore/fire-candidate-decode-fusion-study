"""
FireCandidateDecodeFusion - CPU mathematical reference implementation.

Purpose: a *mathematical* cross-check for the Ascend C operator
(fire_candidate_decode_fusion). All formula steps are executed in
float32 unless explicitly stated.

!!! DISCLAIMER !!!
This program is a MATHEMATICAL reference only. It does NOT verify any NPU /
Ascend C kernel. Passing every test here proves nothing about the correctness
or the precision of the device-side implementation. It only tells you what the
official formula produces in CPU float32.

Source of the formula: CANN Judge problem page "目标候选框解码与置信度融合算子
FireCandidateDecodeFusion" (section 3.2/3.4/3.5).

  sigmoid(z) = 1 / (1 + exp(-z))
  sx = sigmoid(tx), sy = sigmoid(ty), sw = sigmoid(tw), sh = sigmoid(th)
  x = (2*sx - 0.5 + gx) * stride
  y = (2*sy - 0.5 + gy) * stride
  w = (2*sw)^2 * anchor_w * stride
  h = (2*sh)^2 * anchor_h * stride
  fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)

  raw_pred : [B, N, 6]  fp16|fp32   [tx, ty, tw, th, obj_logit, fire_logit]
  priors   : [N, 5]     fp32        [gx, gy, stride, anchor_w, anchor_h]
  decoded  : [B, N, 5]  same as raw_pred
"""
import numpy as np

F32 = np.float32


def sigmoid_f32(z):
    """sigmoid in float32: 1/(1+exp(-z))."""
    return F32(1.0) / (F32(1.0) + np.exp(np.negative(z, dtype=F32), dtype=F32))


def _as_f32(a):
    return np.asarray(a, dtype=F32)


def decode_golden(raw_pred, priors, compute_dtype=F32):
    """Reference decode. compute_dtype=F32 is the official behaviour;
    F64 is only used to measure float32 rounding error."""
    d = compute_dtype
    rp = np.asarray(raw_pred).astype(d)
    pr = np.asarray(priors).astype(d)

    tx = rp[..., 0]; ty = rp[..., 1]; tw = rp[..., 2]
    th = rp[..., 3]; obj = rp[..., 4]; fire = rp[..., 5]

    def sig(z):
        return d(1.0) / (d(1.0) + np.exp(np.negative(z, dtype=d), dtype=d))

    sx = sig(tx); sy = sig(ty); sw = sig(tw); sh = sig(th)

    gx = pr[None, :, 0]; gy = pr[None, :, 1]
    stride = pr[None, :, 2]; aw = pr[None, :, 3]; ah = pr[None, :, 4]

    two = d(2.0); half = d(0.5)
    x = (two * sx - half + gx) * stride
    y = (two * sy - half + gy) * stride
    two_sw = two * sw
    two_sh = two * sh
    w = (two_sw * two_sw) * aw * stride
    h = (two_sh * two_sh) * ah * stride
    fs = sig(obj) * sig(fire)

    out = np.stack([x, y, w, h, fs], axis=-1)
    return out


def decode(raw_pred, priors):
    """Public entry point. Output dtype always equals input dtype (official rule)."""
    rp = np.asarray(raw_pred)
    out_dtype = rp.dtype
    if out_dtype not in (np.float16, F32):
        raise TypeError("raw_pred must be float16 or float32, got %s" % out_dtype)
    golden = decode_golden(rp, priors, F32)
    return golden.astype(out_dtype, copy=False)


def decode_rowwise(raw_pred, priors):
    """Same math but as an explicit n-by-n python loop - used to prove the
    vectorised version keeps candidate order and priors-per-n indexing."""
    rp = np.asarray(raw_pred).astype(F32)
    pr = np.asarray(priors).astype(F32)
    B, N, _ = rp.shape
    out = np.empty((B, N, 5), dtype=F32)
    for b in range(B):
        for n in range(N):
            tx, ty, tw, th, obj, fire = [F32(v) for v in rp[b, n]]
            gx, gy, stride, aw, ah = [F32(v) for v in pr[n]]
            sx = sigmoid_f32(tx); sy = sigmoid_f32(ty)
            sw = sigmoid_f32(tw); sh = sigmoid_f32(th)
            x = (F32(2) * sx - F32(0.5) + gx) * stride
            y = (F32(2) * sy - F32(0.5) + gy) * stride
            w = (F32(2) * sw) * (F32(2) * sw) * aw * stride
            h = (F32(2) * sh) * (F32(2) * sh) * ah * stride
            out[b, n] = [x, y, w, h, sigmoid_f32(obj) * sigmoid_f32(fire)]
    return out
