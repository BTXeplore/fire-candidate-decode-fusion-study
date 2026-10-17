#include "kernel_operator.h"

#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

using namespace AscendC;

constexpr uint32_t TILE_N = 704;
static_assert(TILE_N % 64 == 0, "Tile must preserve vector/block alignment");

constexpr uint32_t RAW_F = 6;
constexpr uint32_t PRI_F = 5;
constexpr uint32_t OUT_F = 5;
static_assert(TILE_N * OUT_F < 16384, "Exact divide-by-five range");

constexpr uint32_t ELEM_RAW_TILE = TILE_N * RAW_F;
constexpr uint32_t ELEM_PRI_TILE = TILE_N * PRI_F;
constexpr uint32_t ELEM_OUT_TILE = TILE_N * OUT_F;

constexpr uint32_t FIELD_VEC_COUNT = 14;
constexpr uint32_t RES_VEC_COUNT = 5;

template <class DT_RAW_PRED>
class KernelFireCandidateDecodeFusion {
public:
    __aicore__ inline KernelFireCandidateDecodeFusion() {}

    __aicore__ inline void Init(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
                                GM_ADDR tiling) {
        GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, tilingData, tiling);
        this->batch = tilingData.batch;
        this->nPerBatch = tilingData.nPerBatch;
        this->coreNum = tilingData.coreNum;
        this->nPerCore = tilingData.nPerCore;
        this->nTail = tilingData.nTail;

        uint32_t coreId = GetBlockIdx();
        uint32_t myN = (coreId < nTail) ? (nPerCore + 1) : nPerCore;
        uint32_t nBegin = (coreId < nTail)
                              ? coreId * (nPerCore + 1)
                              : nTail * (nPerCore + 1) + (coreId - nTail) * nPerCore;
        this->nBegin = nBegin;
        this->myN = myN;

        rawPredGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_RAW_PRED *>(raw_pred),
                                  (uint64_t)batch * nPerBatch * RAW_F);
        priorsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(priors),
                                 (uint64_t)nPerBatch * PRI_F);
        decodedGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_RAW_PRED *>(decoded),
                                  (uint64_t)batch * nPerBatch * OUT_F);

        pipe.InitBuffer(inRawQue, 2, ELEM_RAW_TILE * sizeof(DT_RAW_PRED));
        pipe.InitBuffer(inPriQue, 1, ELEM_PRI_TILE * sizeof(float));
        pipe.InitBuffer(outQue, 2, ELEM_PRI_TILE * sizeof(DT_RAW_PRED));
        pipe.InitBuffer(sigABuf, ELEM_RAW_TILE * sizeof(float));
        pipe.InitBuffer(sigBBuf, ELEM_RAW_TILE * sizeof(float));
        pipe.InitBuffer(onesBuf, ELEM_RAW_TILE * sizeof(float));
        pipe.InitBuffer(fieldBuf, FIELD_VEC_COUNT * TILE_N * sizeof(float));
        if constexpr (std::is_same<DT_RAW_PRED, half>::value) {
            pipe.InitBuffer(outF32Buf, ELEM_PRI_TILE * sizeof(float));
        }
        pipe.InitBuffer(idxBuf, (2 * TILE_N + ELEM_OUT_TILE) * sizeof(uint32_t));

    }

    __aicore__ inline void EnsureScratch(uint32_t count) {
        // Index construction runs before sigmoid; reuse vector-only scratch.
        LocalTensor<int32_t> rampI = sigBBuf.Get<int32_t>();
        if (count > builtRaw) {
            CreateVecIndex(rampI, (int32_t)0, count);
            LocalTensor<int32_t> rawIdxI = idxBuf.Get<int32_t>();
            Muls(rawIdxI, rampI, (int32_t)(4 * RAW_F), (int32_t)count);
            builtRaw = count;
        }
        if (count > builtPri) {
            CreateVecIndex(rampI, (int32_t)0, count);
            LocalTensor<int32_t> idxI = idxBuf.Get<int32_t>();
            LocalTensor<int32_t> priIdxI = idxI[TILE_N];
            Muls(priIdxI, rampI, (int32_t)(4 * PRI_F), (int32_t)count);
            builtPri = count;
        }
        const uint32_t needOut = count * OUT_F;
        if (needOut > builtOut) {
            LocalTensor<int32_t> idxI = idxBuf.Get<int32_t>();
            LocalTensor<int32_t> outIdxI = idxI[2 * TILE_N];
            LocalTensor<int32_t> ramp = sigBBuf.Get<int32_t>();
            LocalTensor<int32_t> oT = sigABuf.Get<int32_t>();
            LocalTensor<int32_t> t1 = oT;
            LocalTensor<int32_t> q5 = oT[TILE_N];
            LocalTensor<int32_t> q2 = oT[2 * TILE_N];
            for (uint32_t off = 0; off < needOut; off += TILE_N) {
                const uint32_t cur = ((needOut - off) < TILE_N) ? (needOut - off) : TILE_N;
                CreateVecIndex(ramp, (int32_t)off, cur);
                Muls(t1, ramp, (int32_t)13108, (int32_t)cur);
                ShiftRight(q5, t1, (int32_t)16, (int32_t)cur);
                Muls(t1, ramp, (int32_t)(4 * TILE_N), (int32_t)cur);
                Muls(q2, q5, (int32_t)(4 * (OUT_F * TILE_N - 1)), (int32_t)cur);
                Sub(outIdxI[off], t1, q2, (int32_t)cur);
            }
            builtOut = needOut;
        }
        const uint32_t needOnes = count * RAW_F;
        if (needOnes > builtOnes) {
            LocalTensor<float> ones = onesBuf.Get<float>();
            Duplicate(ones[builtOnes], (float)1.0, (int32_t)(needOnes - builtOnes));
            builtOnes = needOnes;
        }
    }

    __aicore__ inline void Process() {
        if (myN == 0) {
            return;
        }
        for (uint32_t base = 0; base < myN; base += TILE_N) {
            uint32_t cur = (myN - base < TILE_N) ? (myN - base) : TILE_N;
            uint32_t n0 = nBegin + base;
            LoadPriors(n0, cur);
            if (batch > 0) {
                LoadRaw(0, n0, cur);
            }
            for (uint32_t b = 0; b < batch; ++b) {
                if (b + 1 < batch) {
                    LoadRaw(b + 1, n0, cur);
                }
                ComputeTile(b, n0, cur);
            }
            FlushOut();
        }
    }

    __aicore__ inline void LoadPriors(uint32_t n0, uint32_t count) {
        EnsureScratch(count);
        const uint32_t n5 = count * PRI_F;
        LocalTensor<float> priLoc = inPriQue.AllocTensor<float>();
        {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = n5 * sizeof(float);
            p.srcStride = 0;
            p.dstStride = 0;
            DataCopyPadExtParams<float> pad;
            pad.isPad = false;
            DataCopyPad(priLoc, priorsGm[(uint64_t)n0 * PRI_F], p, pad);
        }
        inPriQue.EnQue(priLoc);
        LocalTensor<float> priIn = inPriQue.DeQue<float>();
        LocalTensor<uint32_t> idxP = idxBuf.Get<uint32_t>();
        LocalTensor<uint32_t> priIdxP = idxP[TILE_N];
        LocalTensor<float> fvP = fieldBuf.Get<float>();
        LocalTensor<float> gxP = fvP[6 * TILE_N];
        LocalTensor<float> gyP = fvP[7 * TILE_N];
        LocalTensor<float> strideP = fvP[8 * TILE_N];
        LocalTensor<float> awP = fvP[9 * TILE_N];
        LocalTensor<float> ahP = fvP[10 * TILE_N];
        Gather(gxP, priIn, priIdxP, (uint32_t)(4 * 0), count);
        Gather(gyP, priIn, priIdxP, (uint32_t)(4 * 1), count);
        Gather(strideP, priIn, priIdxP, (uint32_t)(4 * 2), count);
        Gather(awP, priIn, priIdxP, (uint32_t)(4 * 3), count);
        Gather(ahP, priIn, priIdxP, (uint32_t)(4 * 4), count);
        inPriQue.FreeTensor(priIn);

        LocalTensor<float> tmpT = fvP;
        LocalTensor<float> twoSP = fvP[12 * TILE_N];
        Adds(tmpT, gxP, (float)(-0.5), (int32_t)count);
        Mul(gxP, tmpT, strideP, (int32_t)count);
        Adds(tmpT, gyP, (float)(-0.5), (int32_t)count);
        Mul(gyP, tmpT, strideP, (int32_t)count);
        Muls(tmpT, awP, (float)4.0, (int32_t)count);
        Mul(awP, tmpT, strideP, (int32_t)count);
        Muls(tmpT, ahP, (float)4.0, (int32_t)count);
        Mul(ahP, tmpT, strideP, (int32_t)count);
        Muls(twoSP, strideP, (float)2.0, (int32_t)count);
    }

private:
    __aicore__ inline void FlushOut() {
        if (pendValid) {
            LocalTensor<DT_RAW_PRED> prev = outQue.DeQue<DT_RAW_PRED>();
            DataCopyExtParams pp;
            pp.blockCount = 1;
            pp.blockLen = pendCount * sizeof(DT_RAW_PRED);
            pp.srcStride = 0;
            pp.dstStride = 0;
            DataCopyPad(decodedGm[pendOff], prev, pp);
            outQue.FreeTensor(prev);
            pendValid = 0;
        }
    }

    __aicore__ inline void LoadRaw(uint32_t b, uint32_t n0, uint32_t count) {
        const uint32_t n6 = count * RAW_F;
        LocalTensor<DT_RAW_PRED> rawLoc = inRawQue.AllocTensor<DT_RAW_PRED>();
        {
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = n6 * sizeof(DT_RAW_PRED);
            p.srcStride = 0;
            p.dstStride = 0;
            DataCopyPadExtParams<DT_RAW_PRED> pad;
            pad.isPad = false;
            DataCopyPad(rawLoc, rawPredGm[(uint64_t)(b * nPerBatch + n0) * RAW_F], p, pad);
        }
        inRawQue.EnQue(rawLoc);
    }

    __aicore__ inline void ComputeTile(uint32_t b, uint32_t n0, uint32_t count) {
        const uint32_t n6 = count * RAW_F;
        const uint32_t n5 = count * PRI_F;
        EnsureScratch(count);
        FlushOut();
        LocalTensor<DT_RAW_PRED> rawIn = inRawQue.DeQue<DT_RAW_PRED>();

        LocalTensor<float> sigA = sigABuf.Get<float>();
        LocalTensor<float> sigB = sigBBuf.Get<float>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        if constexpr (std::is_same<DT_RAW_PRED, half>::value) {
            Cast(sigA, rawIn, RoundMode::CAST_NONE, (int32_t)n6);
            Muls(sigB, sigA, (float)(-1.0), (int32_t)n6);
        } else {
            Muls(sigB, rawIn, (float)(-1.0), (int32_t)n6);
        }
        inRawQue.FreeTensor(rawIn);
        Exp(sigA, sigB, (int32_t)n6);
        Adds(sigB, sigA, (float)1.0, (int32_t)n6);
        // Full tiles: gather denominators before reducing six reciprocals
        // to four geometry reciprocals and one combined score reciprocal.
        // Tail tiles keep the original path and its exact valid-count rules.
        const bool fuseScoreDiv = (count == TILE_N);
        if (!fuseScoreDiv) {
            Div(sigA, ones, sigB, (int32_t)n6);
        }

        LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
        LocalTensor<uint32_t> rawIdx = idx;
        LocalTensor<uint32_t> priIdx = idx[TILE_N];
        LocalTensor<uint32_t> outIdx = idx[2 * TILE_N];

        LocalTensor<float> fv = fieldBuf.Get<float>();
        LocalTensor<float> result = fuseScoreDiv ? sigA : fv;
        LocalTensor<float> gatherSrc = fuseScoreDiv ? sigB : sigA;
        LocalTensor<float> sx = result;
        LocalTensor<float> sy = result[TILE_N];
        LocalTensor<float> sw = result[2 * TILE_N];
        LocalTensor<float> sh = result[3 * TILE_N];
        LocalTensor<float> obj = result[4 * TILE_N];
        LocalTensor<float> fire = fv[5 * TILE_N];
        LocalTensor<float> gx = fv[6 * TILE_N];
        LocalTensor<float> gy = fv[7 * TILE_N];
        LocalTensor<float> stride = fv[8 * TILE_N];
        LocalTensor<float> aw = fv[9 * TILE_N];
        LocalTensor<float> ah = fv[10 * TILE_N];
        LocalTensor<float> t0 = fv[11 * TILE_N];
        LocalTensor<float> t1 = fv[12 * TILE_N];
        LocalTensor<float> t2 = fv[13 * TILE_N];

        Gather(fv, gatherSrc, rawIdx, (uint32_t)(4 * 0), count);
        Gather(fv[TILE_N], gatherSrc, rawIdx, (uint32_t)(4 * 1), count);
        Gather(fv[2 * TILE_N], gatherSrc, rawIdx, (uint32_t)(4 * 2), count);
        Gather(fv[3 * TILE_N], gatherSrc, rawIdx, (uint32_t)(4 * 3), count);
        Gather(fv[4 * TILE_N], gatherSrc, rawIdx, (uint32_t)(4 * 4), count);
        Gather(fire, gatherSrc, rawIdx, (uint32_t)(4 * 5), count);
        if (fuseScoreDiv) {
            Mul(fv[4 * TILE_N], fv[4 * TILE_N], fire, (int32_t)count);
            // All five fields are fully populated; dst is a separate buffer.
            Div(sigA, ones, fv, (int32_t)n5);
        }

        FusedMulAdd(sx, t1, gx, (int32_t)count);
        FusedMulAdd(sy, t1, gy, (int32_t)count);
        Mul(t0, sw, sw, (int32_t)count);
        Mul(sw, t0, aw, (int32_t)count);
        Mul(t0, sh, sh, (int32_t)count);
        Mul(sh, t0, ah, (int32_t)count);
        if (!fuseScoreDiv) {
            Mul(obj, obj, fire, (int32_t)count);
        }

        LocalTensor<DT_RAW_PRED> outLoc = outQue.AllocTensor<DT_RAW_PRED>();
        if constexpr (std::is_same<DT_RAW_PRED, half>::value) {
            LocalTensor<float> outF = outF32Buf.Get<float>();
            Gather(outF, result, outIdx, (uint32_t)0, (uint32_t)n5);
            Cast(outLoc, outF, RoundMode::CAST_NONE, n5);
        } else {
            Gather(outLoc, result, outIdx, (uint32_t)0, (uint32_t)n5);
        }
        outQue.EnQue(outLoc);
        pendOff = (uint64_t)(b * nPerBatch + n0) * OUT_F;
        pendCount = n5;
        pendValid = 1;
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inRawQue;
    TQue<QuePosition::VECIN, 1> inPriQue;
    TQue<QuePosition::VECOUT, 2> outQue;
    uint64_t pendOff = 0;
    uint32_t pendCount = 0;
    uint32_t pendValid = 0;
    TBuf<QuePosition::VECCALC> sigABuf;
    TBuf<QuePosition::VECCALC> sigBBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> fieldBuf;
    TBuf<QuePosition::VECCALC> outF32Buf;
    TBuf<QuePosition::VECCALC> idxBuf;

    GlobalTensor<DT_RAW_PRED> rawPredGm;
    GlobalTensor<float> priorsGm;
    GlobalTensor<DT_RAW_PRED> decodedGm;

    uint32_t batch = 0;
    uint32_t nPerBatch = 0;
    uint32_t coreNum = 0;
    uint32_t nPerCore = 0;
    uint32_t nTail = 0;
    uint32_t nBegin = 0;
    uint32_t myN = 0;
    uint32_t builtRaw = 0;
    uint32_t builtPri = 0;
    uint32_t builtOut = 0;
    uint32_t builtOnes = 0;
};

template <typename DT_RAW_PRED>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors,
                                                        GM_ADDR decoded, GM_ADDR workspace,
                                                        GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    KernelFireCandidateDecodeFusion<DT_RAW_PRED> op;
    op.Init(raw_pred, priors, decoded, tiling);
    op.Process();
}
