// 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
// 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
// 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
#include "kernel_operator.h"

#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

using namespace AscendC;

// 学习｜阅读主线：Init 划分候选与申请 UB → Process 调度 → LoadPriors 预处理先验 →
// 学习｜LoadRaw 搬入预测 → ComputeTile 激活/解码/重排 → FlushOut 写回。
// 学习｜raw_pred 是 [B,N,6]，priors 是 [N,5]，decoded 是 [B,N,5]；后者顺序严格固定。
// 学习｜704 是当前版本的 Tile 容量，不代表每核一定获得 704 个候选，也不代表真实硬件核数。
constexpr uint32_t TILE_N = 704;
static_assert(TILE_N % 64 == 0, "Tile must preserve vector/block alignment");

// 学习｜每候选 6 个预测字段：[tx,ty,tw,th,obj_logit,fire_logit]。
// 学习｜每候选 5 个先验字段：[gx,gy,stride,anchor_w,anchor_h]；5 个输出：[x,y,w,h,score]。
// 学习｜TILE_N*5=3520，是本版本输出索引 ramp 的最大有效长度；实际尾块用 count。
constexpr uint32_t RAW_F = 6;
constexpr uint32_t PRI_F = 5;
constexpr uint32_t OUT_F = 5;
static_assert(TILE_N * OUT_F < 16384, "Exact divide-by-five range");

constexpr uint32_t ELEM_RAW_TILE = TILE_N * RAW_F;
constexpr uint32_t ELEM_PRI_TILE = TILE_N * PRI_F;
constexpr uint32_t ELEM_OUT_TILE = TILE_N * OUT_F;

// 学习｜fieldBuf 以 T=704 为槽位跨度，共 14*T 个 FP32 元素。槽位含义在 ComputeTile 展开。
// 学习｜RES_VEC_COUNT 是保留的未使用常量；注释副本不顺手清理源码，便于逐字追溯。
constexpr uint32_t FIELD_VEC_COUNT = 14;
constexpr uint32_t RES_VEC_COUNT = 5;

template <class DT_RAW_PRED>
class KernelFireCandidateDecodeFusion {
public:
    __aicore__ inline KernelFireCandidateDecodeFusion() {}

    __aicore__ inline void Init(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
                                GM_ADDR tiling) {
        // 学习｜Host 通过 tiling 传入 B、N 与均衡分核参数；此处读取，不在设备上重新分析形状。
        GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, tilingData, tiling);
        this->batch = tilingData.batch;
        this->nPerBatch = tilingData.nPerBatch;
        this->coreNum = tilingData.coreNum;
        this->nPerCore = tilingData.nPerCore;
        this->nTail = tilingData.nTail;

        // 学习｜设 q=N/coreNum、r=N%coreNum。前 r 核各取 q+1，余核各取 q。
        // 学习｜例如 N=10、coreNum=3：范围依次 [0,4)、[4,7)、[7,10)，无遗漏或重叠。
        // 学习｜核只划分候选 N 维，每核内部遍历全部 B；先验不含 batch 维，便于复用。
        uint32_t coreId = GetBlockIdx();
        uint32_t myN = (coreId < nTail) ? (nPerCore + 1) : nPerCore;
        uint32_t nBegin = (coreId < nTail)
                              ? coreId * (nPerCore + 1)
                              : nTail * (nPerCore + 1) + (coreId - nTail) * nPerCore;
        this->nBegin = nBegin;
        this->myN = myN;

        // 学习｜GM 表示全局内存。SetGlobalBuffer 建立访问视图；真正搬运发生在 DataCopyPad。
        // 学习｜raw/decoded 的元素类型随模板为 FP16 或 FP32，priors 始终是 FP32。
        rawPredGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_RAW_PRED *>(raw_pred),
                                  (uint64_t)batch * nPerBatch * RAW_F);
        priorsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(priors),
                                 (uint64_t)nPerBatch * PRI_F);
        decodedGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_RAW_PRED *>(decoded),
                                  (uint64_t)batch * nPerBatch * OUT_F);

        // 学习｜UB 分配：raw 队列 2 槽，priors 队列 1 槽，输出队列 2 槽。
        // 学习｜队列深度给并行留下空间，不能单凭深度断言 DMA 与 Vector 已实际重叠。
        // 学习｜sigA、sigB 各 6*T 个 FP32；ones 也是 6*T，供 Div 的分子读取。
        // 学习｜fieldBuf 保存拆分字段与跨 batch 复用的先验系数；FP16 另需 5*T 的 FP32 输出暂存。
        // 学习｜显式申请合计：FP16 168960B、FP32 185856B；不含编译器隐含资源。
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
        // 学习｜索引区布局：[rawIdx:T][priIdx:T][outIdx:5T]，均以 32 位整数存储。
        // 学习｜Gather 使用的是相对 FP32 源张量的字节偏移，所以索引公式中乘 4。
        pipe.InitBuffer(idxBuf, (2 * TILE_N + ELEM_OUT_TILE) * sizeof(uint32_t));

    }

    // 学习｜本轮核心改进：借用尚未用于 sigmoid 的 sigA/sigB，一次构造全部索引。
    // 学习｜在当前 Process 中，首块 count 最大，后续 count 不增长，所以只需首次真正构建。
    // 学习｜缓存的是索引和 ones；sigA/sigB 里的 ramp 是一次性临时值，随后可被浮点计算覆盖。
    // 学习｜该函数不是任意调用顺序都安全的通用工具，不能在 ComputeTile 中途重入。
    __aicore__ inline void EnsureScratch(uint32_t count) {
        // 学习｜水位达到当前有效候选数时直接返回：短尾块沿用大表前缀，无需重建。
        if (count <= builtCount) {
            return;
        }
        const uint32_t needOut = count * OUT_F;
        // 学习｜Get<int32_t> 以整数视图使用同一片 UB，不是把现存浮点值逐元素转换成整数。
        // 学习｜sigA/sigB 各有 6*T 容量，能放下 5*count；outIdx 有独立的 5*T 槽。
        LocalTensor<int32_t> ramp = sigBBuf.Get<int32_t>();
        LocalTensor<int32_t> t1 = sigABuf.Get<int32_t>();
        LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
        LocalTensor<int32_t> rawIdx = idx;
        LocalTensor<int32_t> priIdx = idx[TILE_N];
        LocalTensor<int32_t> outIdx = idx[2 * TILE_N];
        // One shared ramp; every instruction uses separate source/destination ranges.
        // 学习｜ramp[i]=i，范围 0..5*count-1。raw/pri 只取它的前 count 项。
        // 学习｜rawIdx[k]=24*k、priIdx[k]=20*k，分别跨过 6 个或 5 个 FP32 字段。
        // 学习｜raw 输入虽可能 FP16，但 rawIdx 用于 Cast 之后的 FP32 sigmoid 张量，故仍乘 4。
        CreateVecIndex(ramp, (int32_t)0, needOut);
        Muls(rawIdx, ramp, (int32_t)(4 * RAW_F), (int32_t)count);
        Muls(priIdx, ramp, (int32_t)(4 * PRI_F), (int32_t)count);
        // 学习｜精确除 5 的有限范围技巧，而非普通浮点除法：
        // 学习｜写 i=5q+r，0<=r<=4；13108=(65536+4)/5。
        // 学习｜因此 13108*i/65536=q+r/5+4*i/(5*65536)。
        // 学习｜对 0<=i<16384，最后两项之和 <1 且 >=0，右移 16 得到 floor(i/5)。
        // 学习｜当前最大 i=3519，int32 乘法最大 46127052，不会溢出。
        // 学习｜outIdx 此时临时保存 q；t1、ramp、outIdx 位于不同缓冲区，无同指令源/目标别名。
        Muls(t1, ramp, (int32_t)13108, (int32_t)needOut);
        ShiftRight(outIdx, t1, (int32_t)16, (int32_t)needOut);
        // 学习｜把字段连续的 SoA 结果重新排列成候选连续的 AoS 输出：
        // 学习｜令 i 为输出展平位置，候选 k=floor(i/5)，字段 f=i%5。
        // 学习｜源位置是 fv[f*T+k]，所以字节索引 =4*(f*T+k)
        // 学习｜=4*T*i-4*(5*T-1)*floor(i/5)，正是后续两次 Muls 和一次 Sub。
        // 学习｜例：i=0..5 对应源元素 0,T,2T,3T,4T,1；即先写候选0的五字段，再写候选1。
        // 学习｜这里减少索引构建循环，仍保留最终输出 Gather，没有声称消除了重排。
        Muls(t1, ramp, (int32_t)(4 * TILE_N), (int32_t)needOut);
        Muls(ramp, outIdx, (int32_t)(4 * (OUT_F * TILE_N - 1)), (int32_t)needOut);
        Sub(outIdx, t1, ramp, (int32_t)needOut);
        LocalTensor<float> ones = onesBuf.Get<float>();
        // 学习｜Div 的分子全为 1。当前真实构建只发生在 builtCount=0 的首块，起点对齐。
        // 学习｜后续小块不补写；若未来改成 count 从小到大增长，须重新审核这个偏移的对齐条件。
        Duplicate(ones[builtCount * RAW_F], (float)1.0,
                  (int32_t)((count - builtCount) * RAW_F));
        builtCount = count;
    }

    // 学习｜循环次序是 N-tile 在外、batch 在内：同一段先验只搬入和预计算一次，再服务全部 B。
    // 学习｜代价是同一 batch 的完整输出不一定按时间连续写出，但 GM 地址保证最终顺序不变。
    __aicore__ inline void Process() {
        if (myN == 0) {
            return;
        }
        for (uint32_t base = 0; base < myN; base += TILE_N) {
            // 学习｜cur 是真实有效候选数，最后一块可以小于 T；所有搬运/计算只处理这些有效项。
            uint32_t cur = (myN - base < TILE_N) ? (myN - base) : TILE_N;
            uint32_t n0 = nBegin + base;
            LoadPriors(n0, cur);
            if (batch > 0) {
                LoadRaw(0, n0, cur);
            }
            for (uint32_t b = 0; b < batch; ++b) {
                // 学习｜先发射下一 batch 的输入，再计算当前 batch。深度 2 使两份输入可以同时在队列中。
                // 学习｜这只是调度上的预取机会；B=1 时没有下一 batch，不能指望此处提供重叠。
                if (b + 1 < batch) {
                    LoadRaw(b + 1, n0, cur);
                }
                ComputeTile(b, n0, cur);
            }
            // 学习｜每个 N-tile 结束必须排空最后一个待写结果，否则最后 batch 的输出会丢失。
            FlushOut();
        }
    }

    // 学习｜priors[n0:n0+count,:] 对所有 batch 共享。先从 GM 连续搬入 AoS 五字段，再拆成向量。
    __aicore__ inline void LoadPriors(uint32_t n0, uint32_t count) {
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
        // 学习｜DataCopyPad 按有效字节数请求搬入，代码不假设 count*5 已按块大小对齐。
        // 学习｜EnQue/DeQue/FreeTensor 表达搬运、消费和复用关系；不要把调用顺序等同于实测时间线。
        inPriQue.EnQue(priLoc);
        // 学习｜EXP-022 将初始化放在 priors 入队之后、取队之前：它只写独立 scratch/索引/ones。
        // 学习｜源码上有机会与 priors 搬入重叠；是否实际重叠、隐藏多少时间，尚无 profiling 证据。
        EnsureScratch(count);
        LocalTensor<float> priIn = inPriQue.DeQue<float>();
        LocalTensor<uint32_t> idxP = idxBuf.Get<uint32_t>();
        LocalTensor<uint32_t> priIdxP = idxP[TILE_N];
        LocalTensor<float> fvP = fieldBuf.Get<float>();
        LocalTensor<float> gxP = fvP[6 * TILE_N];
        LocalTensor<float> gyP = fvP[7 * TILE_N];
        LocalTensor<float> strideP = fvP[8 * TILE_N];
        LocalTensor<float> awP = fvP[9 * TILE_N];
        LocalTensor<float> ahP = fvP[10 * TILE_N];
        // 学习｜priIdxP[k]=20*k，常量偏移 0/4/8/12/16 字节选中五个字段。
        // 学习｜索引只含候选步长，字段偏移由每次 Gather 单独给出。
        Gather(gxP, priIn, priIdxP, (uint32_t)(4 * 0), count);
        Gather(gyP, priIn, priIdxP, (uint32_t)(4 * 1), count);
        Gather(strideP, priIn, priIdxP, (uint32_t)(4 * 2), count);
        Gather(awP, priIn, priIdxP, (uint32_t)(4 * 3), count);
        Gather(ahP, priIn, priIdxP, (uint32_t)(4 * 4), count);
        inPriQue.FreeTensor(priIn);

        // 学习｜预计算并缓存到 fieldBuf：gx=(gx-0.5)*stride，gy=(gy-0.5)*stride，
        // 学习｜aw=4*anchor_w*stride，ah=4*anchor_h*stride，twoSP=2*stride。
        // 学习｜这些槽位在所有 batch 期间保持；tmpT 借用第0槽，随后会被当前 batch 的 sx 覆盖。
        // 学习｜乘加顺序沿用已测版本，不把代数等价解释为浮点逐位等价。
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
    // 学习｜延后写回策略：保存一个已完成结果及其 GM 地址，下一次 ComputeTile 开始时搬出。
    // 学习｜末块由 Process 显式收尾。FreeTensor 在发出搬运后交回队列，复用同步交由队列机制表达；
    // 学习｜不能把 FreeTensor 理解成绕过搬运依赖的立即任意覆写许可。
    __aicore__ inline void FlushOut() {
        if (pendValid) {
            LocalTensor<DT_RAW_PRED> prev = outQue.DeQue<DT_RAW_PRED>();
            DataCopyExtParams pp;
            pp.blockCount = 1;
            // 学习｜pendCount 的单位是输出元素数，即 5*count；blockLen 的单位才是字节。
            pp.blockLen = pendCount * sizeof(DT_RAW_PRED);
            pp.srcStride = 0;
            pp.dstStride = 0;
            DataCopyPad(decodedGm[pendOff], prev, pp);
            outQue.FreeTensor(prev);
            pendValid = 0;
        }
    }

    // 学习｜连续搬入某 batch 的当前候选片段，不在 GM 中逐字段读取。
    // 学习｜地址 ((b*N+n0)*6) 对应 raw_pred[b,n0,0]；不跨候选错配 priors。
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

    // 学习｜计算主线：FP32 sigmoid → 六字段 Gather → 解码/分数乘法 → 五字段输出 Gather。
    // 学习｜不做阈值过滤、排序或 NMS；每个输入候选始终对应一个输出候选。
    __aicore__ inline void ComputeTile(uint32_t b, uint32_t n0, uint32_t count) {
        const uint32_t n6 = count * RAW_F;
        const uint32_t n5 = count * PRI_F;
        // 学习｜此时先发射上一份结果的写回，再开始当前 Vector 工作，提供写回重叠的机会。
        FlushOut();
        LocalTensor<DT_RAW_PRED> rawIn = inRawQue.DeQue<DT_RAW_PRED>();

        // 学习｜精度路径：FP16 raw 先 Cast 到 FP32；FP32 raw 直接参与 FP32 运算。
        // 学习｜统一计算 sigmoid(z)=1/(1+exp(-z))，6*count 项仍保留交错布局。
        LocalTensor<float> sigA = sigABuf.Get<float>();
        LocalTensor<float> sigB = sigBBuf.Get<float>();
        LocalTensor<float> ones = onesBuf.Get<float>();
        if constexpr (std::is_same<DT_RAW_PRED, half>::value) {
            Cast(sigA, rawIn, RoundMode::CAST_NONE, (int32_t)n6);
            Muls(sigB, sigA, (float)(-1.0), (int32_t)n6);
        } else {
            Muls(sigB, rawIn, (float)(-1.0), (int32_t)n6);
        }
        // 学习｜rawIn 的最后一次源码读取已在上面的 Muls 完成，后续 Exp/Div 只用独立 scratch。
        // 学习｜提前归还输入槽，允许后续 batch 预取复用；具体依赖仍由编译/队列实现管理。
        inRawQue.FreeTensor(rawIn);
        Exp(sigA, sigB, (int32_t)n6);
        Adds(sigB, sigA, (float)1.0, (int32_t)n6);
        Div(sigA, ones, sigB, (int32_t)n6);

        LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
        LocalTensor<uint32_t> rawIdx = idx;
        LocalTensor<uint32_t> priIdx = idx[TILE_N];
        LocalTensor<uint32_t> outIdx = idx[2 * TILE_N];

        // 学习｜fieldBuf 槽位图（每槽 T 个 FP32 元素，只有前 count 项有效）：
        // 学习｜0:sx 1:sy 2:sw 3:sh 4:obj 5:fire；6:gx系数 7:gy系数 8:stride
        // 学习｜9:aw系数 10:ah系数 11:t0平方临时量 12:t1=2*stride 13:t2保留。
        // 学习｜priIdx、stride、t2 有些句柄在本函数不再直接使用；保留原码，不借注释改动做清理。
        LocalTensor<float> fv = fieldBuf.Get<float>();
        LocalTensor<float> sx = fv;
        LocalTensor<float> sy = fv[TILE_N];
        LocalTensor<float> sw = fv[2 * TILE_N];
        LocalTensor<float> sh = fv[3 * TILE_N];
        LocalTensor<float> obj = fv[4 * TILE_N];
        LocalTensor<float> fire = fv[5 * TILE_N];
        LocalTensor<float> gx = fv[6 * TILE_N];
        LocalTensor<float> gy = fv[7 * TILE_N];
        LocalTensor<float> stride = fv[8 * TILE_N];
        LocalTensor<float> aw = fv[9 * TILE_N];
        LocalTensor<float> ah = fv[10 * TILE_N];
        LocalTensor<float> t0 = fv[11 * TILE_N];
        LocalTensor<float> t1 = fv[12 * TILE_N];
        LocalTensor<float> t2 = fv[13 * TILE_N];

        // 学习｜输入 AoS → 计算 SoA：从每六个 sigmoid 中取同一字段，得到六条连续结果向量。
        // 学习｜这样各字段能对不同的先验系数做向量运算，代价是六次 Gather。
        Gather(sx, sigA, rawIdx, (uint32_t)(4 * 0), count);
        Gather(sy, sigA, rawIdx, (uint32_t)(4 * 1), count);
        Gather(sw, sigA, rawIdx, (uint32_t)(4 * 2), count);
        Gather(sh, sigA, rawIdx, (uint32_t)(4 * 3), count);
        Gather(obj, sigA, rawIdx, (uint32_t)(4 * 4), count);
        Gather(fire, sigA, rawIdx, (uint32_t)(4 * 5), count);

        // 学习｜中心 x=sx*(2*stride)+(gx_original-0.5)*stride，y 同理。
        // 学习｜宽高 w=(sw*sw)*(4*anchor_w*stride)，h 同理；复用 t0 顺序计算平方。
        // 学习｜score=obj*fire，仍使用两路独立 sigmoid 后相乘，没有叠加 EXP-020 的倒数合并。
        FusedMulAdd(sx, t1, gx, (int32_t)count);
        FusedMulAdd(sy, t1, gy, (int32_t)count);
        Mul(t0, sw, sw, (int32_t)count);
        Mul(sw, t0, aw, (int32_t)count);
        Mul(t0, sh, sh, (int32_t)count);
        Mul(sh, t0, ah, (int32_t)count);
        Mul(obj, obj, fire, (int32_t)count);

        // 学习｜输出 SoA → AoS：outIdx 从 0..4 槽依次取各候选的 x/y/w/h/score。
        // 学习｜count<T 时槽间仍按 T 跨度寻址，而不是按 count；索引公式已体现这一点。
        // 学习｜FP16 分支先 Gather 到 FP32 的 outF，再最终 Cast；不得提前半精度计算中间值。
        LocalTensor<DT_RAW_PRED> outLoc = outQue.AllocTensor<DT_RAW_PRED>();
        if constexpr (std::is_same<DT_RAW_PRED, half>::value) {
            LocalTensor<float> outF = outF32Buf.Get<float>();
            Gather(outF, fv, outIdx, (uint32_t)0, (uint32_t)n5);
            Cast(outLoc, outF, RoundMode::CAST_NONE, n5);
        } else {
            Gather(outLoc, fv, outIdx, (uint32_t)0, (uint32_t)n5);
        }
        // 学习｜结果入队并登记延后写回元数据。GM 地址 ((b*N+n0)*5) 恢复 [B,N,5] 顺序。
        // 学习｜此处只记录一个 pending 结果；下次 ComputeTile 开始先 FlushOut，避免覆盖其元数据。
        outQue.EnQue(outLoc);
        pendOff = (uint64_t)(b * nPerBatch + n0) * OUT_F;
        pendCount = n5;
        pendValid = 1;
    }

    // 学习｜资源成员：TPipe 管理缓冲；TQue 表达输入/输出生产消费；TBuf 保存计算暂存。
    // 学习｜pend* 是待写结果描述；builtCount 只表示索引和 ones 的已建前缀，不表示 sigmoid 数据缓存。
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
    uint32_t builtCount = 0;
};

// 学习｜全局入口由模板类型选择 FP16/FP32。workspace 参数保留框架 ABI，本实现不申请其空间。
// 学习｜设备计算由 Init→Process 两步启动，Host 的 SetBlockDim 决定参与的逻辑块数。
template <typename DT_RAW_PRED>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors,
                                                        GM_ADDR decoded, GM_ADDR workspace,
                                                        GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    KernelFireCandidateDecodeFusion<DT_RAW_PRED> op;
    op.Init(raw_pred, priors, decoded, tiling);
    op.Process();
}
