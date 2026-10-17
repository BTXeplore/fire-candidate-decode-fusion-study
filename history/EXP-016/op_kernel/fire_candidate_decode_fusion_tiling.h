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
struct FireCandidateDecodeFusionTilingData {
    uint32_t batch;
    uint32_t nPerBatch;
    uint32_t coreNum;
    uint32_t nPerCore;
    uint32_t nTail;
};
