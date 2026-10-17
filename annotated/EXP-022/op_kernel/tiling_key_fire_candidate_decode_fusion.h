// 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
// 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
// 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 学习｜声明 DT_RAW_PRED 可选择 float 或 float16；Host 的 ASCENDC_TPL_SEL_PARAM 选择对应实例。
// 学习｜这是当前工程已有模板接口的用途解释，不外推到未核实 SDK 版本。
ASCENDC_TPL_ARGS_DECL(FireCandidateDecodeFusion,
    ASCENDC_TPL_DATATYPE_DECL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);
