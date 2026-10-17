// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
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
        if (nPerBatch == 0) {
            nPerBatch = 1;
        }

        // tiling key：区分 kernel 侧数据类型
        uint32_t dtRawPred = static_cast<uint32_t>(dtypeRawPred);
        ASCENDC_TPL_SEL_PARAM(context, dtRawPred);

        // 配置 tiling 结构体
        FireCandidateDecodeFusionTilingData *tiling =
            context->GetTilingData<FireCandidateDecodeFusionTilingData>();
        tiling->batch = batch;
        tiling->nPerBatch = nPerBatch;

        // 核数不超过 nPerBatch（避免空核），至少 1
        uint32_t cores = static_cast<uint32_t>(numCoresAiv);
        if (cores > nPerBatch) {
            cores = nPerBatch;
        }
        if (cores == 0) {
            cores = 1;
        }
        tiling->coreNum = cores;

        // 经典 n 维分块：前 nTail 个核多分 1 个 n
        tiling->nPerCore = nPerBatch / cores;
        tiling->nTail = nPerBatch % cores;

        // 启动核数
        context->SetBlockDim(cores);

        // 无需 workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

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
