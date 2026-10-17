// 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
// 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
// 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

// 学习｜Host 负责读取元数据、确定分核和模板参数，不在 CPU 上计算 sigmoid 或候选框结果。
namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // 学习｜可用 AIV 核数从运行平台取得；不要把 8/40/48 写成当前 Judge 已确认的硬件事实。
        int32_t numCoresAiv = platform.GetCoreNumAiv();
        if (numCoresAiv <= 0) {
            numCoresAiv = 1;
        }

        // 输入张量信息
        const gert::Tensor *tensorRawPred = context->GetRequiredInputTensor(0);
        ge::DataType dtypeRawPred = tensorRawPred->GetDataType();

        // raw_pred 形状 [B, N, 6]
        const gert::Shape &shapeRawPred = tensorRawPred->GetShape().GetStorageShape();
        uint32_t batch = 0;
        uint32_t nPerBatch = 0;
        if (shapeRawPred.GetDimNum() >= 2) {
            batch = static_cast<uint32_t>(shapeRawPred.GetDim(0));
            nPerBatch = static_cast<uint32_t>(shapeRawPred.GetDim(1));
        }
        // 学习｜这里保留原版兜底逻辑；它不是完整输入校验，不代表此算子支持题目范围外的 N=0。
        if (nPerBatch == 0) {
            nPerBatch = 1;
        }

        // tiling key：区分 kernel 侧数据类型
        uint32_t dtRawPred = static_cast<uint32_t>(dtypeRawPred);
        // 学习｜该 dtype 与 kernel 模板头的 DT_RAW_PRED 参数对应；没有根据输入内容选择不同答案。
        ASCENDC_TPL_SEL_PARAM(context, dtRawPred);

        // 配置 tiling 结构体
        FireCandidateDecodeFusionTilingData *tiling =
            context->GetTilingData<FireCandidateDecodeFusionTilingData>();
        tiling->batch = batch;
        tiling->nPerBatch = nPerBatch;

        // 核数不超过 nPerBatch（避免空核），至少 1
        // 学习｜对合法 N>=1，使用 min(可用AIV核数,N)。Host 未固定8核，也未按未知隐藏case定制。
        // 学习｜B 不参与分核：同核处理某段 N 的全部 batch，priors 可复用。
        uint32_t cores = static_cast<uint32_t>(numCoresAiv);
        if (cores > nPerBatch) {
            cores = nPerBatch;
        }
        if (cores == 0) {
            cores = 1;
        }
        tiling->coreNum = cores;

        // 经典 n 维分块：前 nTail 个核多分 1 个 n
        // 学习｜例如 N=10，cores=3，则 nPerCore=3、nTail=1；Kernel 前1核多处理1个候选。
        // 学习｜TILE_N 是 Kernel 内部切片大小，与这里的核间划分不是同一个概念。
        tiling->nPerCore = nPerBatch / cores;
        tiling->nTail = nPerBatch % cores;

        // 启动核数
        // 学习｜逻辑启动块数与 tiling->coreNum 一致，避免地址计算与实际启动数不匹配。
        context->SetBlockDim(cores);

        // 无需 workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        // 学习｜workspace 是额外 GM 临时区；置0不表示不使用片上 UB。UB 在 Kernel 的 Init 申请。
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

// 学习｜形状/类型推导告诉图框架如何分配输出，不会执行 NPU 的逐元素计算。
namespace ge {
    // 输出 decoded 形状固定 [B, N, 5]
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);  // [B, N, 6]
        if (inputShape == nullptr) {
            return GRAPH_FAILED;
        }
        size_t dimNum = inputShape->GetDimNum();
        if (dimNum < 2) {
            return GRAPH_FAILED;
        }
        int64_t b = inputShape->GetDim(0);
        int64_t n = inputShape->GetDim(1);

        gert::Shape *outShape = context->GetOutputShape(0);
        if (outShape == nullptr) {
            return GRAPH_FAILED;
        }
        // 学习｜保留 B 和 N，仅把字段数由6改为5。priors 的配对及输入约束仍须遵守题面。
        outShape->SetDimNum(3);
        outShape->SetDim(0, b);
        outShape->SetDim(1, n);
        outShape->SetDim(2, 5);
        return GRAPH_SUCCESS;
    }

    // 输出 dtype 与 raw_pred 一致
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtype);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

// 学习｜算子注册：raw FP32→输出FP32，raw FP16→输出FP16；两种情况下 priors 均为FP32。
// 学习｜这里声明 ND 格式和 ascend910b 目标，不证明其它 CANN 版本/芯片的接口兼容性。
namespace ops {
    class FireCandidateDecodeFusion : public OpDef {
    public:
        explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
            this->Input("raw_pred")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("priors")
                .ParamType(REQUIRED)
                // 题面 3.2/3.5：priors 固定 float32（仅 raw_pred/decoded 支持 float16）。
                // 索引配对后 combo0=(fp32,fp32,fp32)、combo1=(fp16,float32,fp16)，与题面一致。
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("decoded")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FireCandidateDecodeFusion);
}  // namespace ops
