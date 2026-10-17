# EXP-022 中文注释版

这是为逐段学习准备的七份完整源码副本。只增加带“学习｜”前缀的整行中文注释；删除新增注释并统一换行后，七份文件均与 src/ 中的 EXP-022 原始源码逐字相同。原始冻结版及 Judge 成绩仍以 [src/](../../src/) 和[本轮记录](../../docs/06-exp020-exp022-cross-review.md)为准。注释版未独立编译或提交 Judge，不能把原始版本的12.94µs称作本副本实测。

## 建议阅读顺序

1. [设备主程序](op_kernel/fire_candidate_decode_fusion.cpp)：先读文件开头和 EnsureScratch，理解共享 ramp、精确除5与 SoA→AoS 索引公式。
2. 在同一文件读 Process、LoadPriors、ComputeTile、FlushOut，理解先验跨 batch 复用、输入预取、延后写回和最后一块收尾。
3. [Host 分核与注册](op_host/fire_candidate_decode_fusion.cpp)：区分 Host 元数据计算和设备向量计算。
4. [tiling 数据协议](op_kernel/fire_candidate_decode_fusion_tiling.h)与[模板类型选择](op_kernel/tiling_key_fire_candidate_decode_fusion.h)。
5. 三个 CMakeLists.txt：理解工程入口、Host 库与 Kernel 库关系。

## 读代码时记住

- count 是有效候选数，T=704 是字段槽位跨度；短块不能把跨度改成 count。
- rawIdx/priIdx/outIdx 是 FP32 Gather 的字节偏移，元素偏移与字节偏移不要混淆。
- builtCount 缓存的是索引和 ones，不缓存 sigmoid 值；临时整数视图不代表数值 Cast。
- raw/decoded 可以 FP16，激活、坐标、平方和分数中间值都保持 FP32。
- 队列深度、预取位置和源码上独立的数据流只说明重叠机会，不能替代真实 profiling。
- 本注释不认定 Judge 的实际核数、UB物理Bank映射或未知微码行为。

原 EXP-016、EXP-020 与 EXP-022 的冻结源码及全部对照继续保留，注释不引入新实验候选。
