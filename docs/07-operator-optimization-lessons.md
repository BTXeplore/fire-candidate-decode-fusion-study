# 从 FireCandidateDecodeFusion 学习算子优化

日期：2026-10-18。先读[中文注释版022](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp)：按 `Process → LoadPriors → ComputeTile → EnsureScratch → Init` 读，先理解数据，再算内存。本文解释源码能证明的设计；实验结果见[逐点记录](06-exp020-exp022-cross-review.md)，不把性能猜想写成硬件事实。

## 1. 融合：把中间值留在片上

每个候选有六个预测值，先算 `sᵢ=1/(1+exp(-zᵢ))`，再生成中心、宽高和分数。`ComputeTile` 在一个内核内完成这些步骤，结果经一次输出队列写回。融合的意义是减少需要暴露到GM的中间张量；它仍需执行Exp、Div、Gather等，不能把“融合”理解成运算免费。先画每个数组从哪里来、最后被谁读，再讨论删哪一步。[源码：ComputeTile](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L262)

## 2. AoS/SoA：元素位置与字节地址分开算

输入按候选交错排列，六字段AoS中第j项第f字段是 `6j+f`；FP32的Gather索引写成 `24j`，另加字段字节偏移 `4f`。拆成连续字段后，向量数学可以逐段进行。

输出要恢复五字段交错。令字段跨度为S，输出位置i应读 `4*((i%5)*S+floor(i/5))` 字节。022中S=704。将商记为q，可改写为 `4*S*i-4*(5*S-1)*q`。源码在限定范围内用整数乘与移位精确求q，并共享一次ramp；这减少索引构造循环，**没有消除输出Gather**。完整首块源码调用35→9，不是机器指令或时延同比下降。[索引构造](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L95)；[旧016循环](../history/EXP-016/op_kernel/fire_candidate_decode_fusion.cpp#L66)

## 3. 临时区：同一地址可以有两个不同时段的用途

022构建索引时借用sigA/sigB作整数scratch；随后它们才保存浮点中间值。成立条件是两段用途不重叠，而且索引已复制到独立idxBuf。`builtCount` 让后续Tile使用缓存前缀：当前Process首块最大，之后相同或变小。若以后允许小块之后再来大块，或在解算中重建索引，必须重做生命周期证明，不能只看空间够不够。[EnsureScratch与Process](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L95)

算UB时，用“元素数×元素字节数×队列块数”逐项相加。022显式申请在fp16路径为168960B、fp32路径为185856B，前者还单独申请FP32输出临时区。这不是编译器全部资源的实测值；减少显式容量也不自动意味着更快，读写位置和同步可能同时变化。[Init中的分配](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L75)

## 4. 循环顺序也是数据复用

Process把同一空间Tile的所有batch放在内层。`LoadPriors` 只执行一次，把共享先验整理成系数；每个batch只搬自己的raw并重复读取系数。代价是这些系数在整个batch循环结束前必须保留，不能随意借作临时输出。优化时应标出“最后一次使用”，再决定何时覆盖。[先验预处理](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L169)

## 5. 队列深度不等于真实重叠

`Alloc/EnQue/DeQue/Free` 表达缓冲取得、就绪与归还。Process先发下一batch的LoadRaw，再计算当前batch，提供了重叠窗口；022也把索引构建放在priors的EnQue/DeQue之间。源码能证明操作数区域独立，不能证明DMA已经与向量计算同时运行。队列深度2只提供第二块空间；真实收益还依赖事件、调度、工作量及硬件时间线。[Process](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L143)；[FlushOut/LoadRaw](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L227)

## 6. 精度、尾块与实验解释

fp16输入先转FP32，最终才转回；数学等价改写仍可能改变舍入或中间溢出。尾块按count运算与搬运，字段间距仍是完整S；输出索引必须读每段有效前缀，不能误用count作为跨度。[ComputeTile](../annotated/EXP-022/op_kernel/fire_candidate_decode_fusion.cpp#L262)

[验证框架](../validation/README.md)把问题分开：Coverage查规格与寻址；Correctness查字段、极值和容差；Mock用固定本地负载的真实设备曲线检验代表性。目前只有Judge，本地CPU模型不证明DMA安全、事件正确或NPU时延；Mock设备性能缺测应保留缺测。

同源码前后也会波动。应保留原版对照、全部逐点结果及源码身份，避免只选最慢对照或最快候选。022两次均值13.375与12.94µs都是观测，不能自动组成稳定提速结论。[真实测量与证据限制](06-exp020-exp022-cross-review.md#1-两个变量与真实结果)

阅读练习：手算输出前十个索引，应该依次取五段的第零项，再取五段的第一项；再用只有一个候选的尾块检查它。接着故意交换两个字段、破坏一个输出哨兵，确认自己的测试确实报错。测试能抓住错误，才有资格替改动提供信心。

今晚023只检验字段跨度704→712，浮点算式和Tile不改；12点通过，但表现有负载取舍，已保留22。地址模数改变不证明物理Bank映射；详见[023结果与解释](08-exp023-layout.md)。
