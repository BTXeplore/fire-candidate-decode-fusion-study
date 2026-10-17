// 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
// 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
// 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 该结构体在 Host 侧由 TilingFunc 填充，并通过 tiling GM 传给 Kernel 侧。
// 字段含义：
//   batch     : B，batch 数
//   nPerBatch : N，每 batch 的候选数（也是 priors 的行数）
//   coreNum   : 实际参与计算的 AIV 核数（blockDim）
//   nPerCore  : 每个核分到的基准 n 个数 = N / coreNum
//   nTail     : 前 nTail 个核额外多分 1 个 n = N % coreNum
//               （经典分块：前 nTail 核处理 nPerCore+1 个 n，其余 nPerCore 个，尾部自然处理）
// 学习｜Host 和 Kernel 共用同一结构布局，所以字段顺序和类型属于双方协议，不可随意单边改变。
// 学习｜传递的都是维度/划分元数据；不携带输入预测值，也不携带预计算的输出。
// 学习｜coreNum 是逻辑启动块数；nTail 是核间余数，区别于单核内部最后 Tile 的有效 count。
struct FireCandidateDecodeFusionTilingData {
    uint32_t batch;
    uint32_t nPerBatch;
    uint32_t coreNum;
    uint32_t nPerCore;
    uint32_t nTail;
};
