# 2026-10-17：HyosungSink案例源码学习与三类Case方法

研究目标：为 FireCandidateDecodeFusion 的测试框架与两轮迭代提供可复核的方法，不把他人算子的数字当作我们的收益承诺。

固定阅读版本：[f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/tree/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7)。本篇基于选择性源码和文档阅读。这里引用的是固定版本的代码、项目环境声明与原始作者报告，不代表我们已在本机独立复现上游算子。

## 三类Case各自解决什么
- **Coverage**：输入契约各维度、dtype、边界、分支、尾块和缓冲周转都被覆盖；它回答“实现是否适用于全部受支持输入”。
- **Correctness**：用独立参考、稳定数学形式和可解释构造验证误差、语义、布局与确定性；不能只测随机正常值。
- **Mock**：构造具有相似规模/分支压力的公开本地性能工作负载；比较同一组源码版本在本地与官方的耗时变化趋势，回答“这个本地负载是否能指导优化”。它不等于已还原隐藏形状，更不能凭隐藏案例输出作条件分支。
- “趋势一致”只能在多版本、多形状与计时噪声被控制后作为证据。单次本地快、单次平台快，不足以证明映射关系；CPU 时间不能替代 NPU 时间。

**重要缺口**：本仓 README“收录范围”明确排除每个案例的构建、复现、测试、校验脚本。树中没有可直接复用的 Coverage/Correctness/Mock 成套 tests。文档会引用外部 harness，但本次未取得这些 harness 的源码，不能声称已复用或跑通。SwiGLU 文档的 examples 链接也不在本快照中。因此本项目独立实现测试分类，并明确 CPU 参考验证与真实 NPU 验证的边界。当前入口见[validation/README.md](../validation/README.md)。

## 知识点一：减少转换应从数据生命周期出发；流水需要真实的槽位归属

深入案例：05 Causal Conv1d Cast-once。

**已核实的源码事实**
- [causal_conv1d.h L346-L387](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/05-community-causal-conv1d-cast-once-a2/code/op_kernel/causal_conv1d.h#L346)：状态载入后转成 FP32 ring，初始输入先发出 MTE2。
- [RunSeq L415-L424](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/05-community-causal-conv1d-cast-once-a2/code/op_kernel/causal_conv1d.h#L415)：等待当前输入完成，在消费当前数据后，等下一 slot 可写，再发下一输入的 DMA，随后计算当前 token。MTE2→V 是数据就绪；V→MTE2 是源槽位可复用。
- [L451-L477](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/05-community-causal-conv1d-cast-once-a2/code/op_kernel/causal_conv1d.h#L451)：输出按 t&1 交替，第二轮复用前等 MTE3→V；先发 V→MTE3、等输出就绪，再 DMA 写回。双槽不是同步的替代品。
- [installation L9-L15](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/05-community-causal-conv1d-cast-once-a2/docs/installation.md#L9)：项目环境声明 Atlas A2、CANN ==9.0.0。PR 的测试硬件是910B2。这是环境声明和作者实测报告，不是我们逐 API 编译验证。
- [PR L20-L30、L166-L173](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/05-community-causal-conv1d-cast-once-a2/docs/PR-10507.md#L20)：48配置是 dim×seqlen×batch×bias×silu 的完整笛卡尔积；device 使用 msprof 20次中位数，端到端50次中位数；作者报告输出及持久状态均48/48 bit-exact。没有把两种耗时混在一起。

**原理与本项目关联**
重复被使用的低精度值提升一次保留在FP32，可以省循环内反复 Cast；但若算法原本每轮就有量化，取消中间舍入会改变语义，不能一概宣称 bit-exact。我们 EXP-016 已在每个 raw 输入块只 Cast 一次、提前释放原始队列，并在 batch 内提前搬入下一个输入，不能把同一技巧再次作为全新优化候选。此案例最有用的是审查队列和事件状态机：首块、B=1、B=2、B≥3、最后一个 pending 输出、跨N尾块都要有测试。跨batch共享 priors 的收益必须保留。

**未证实**
没有取得05的测试脚本；没有在本机9.0.0头文件逐 API 取证；不能迁移其卷积加速比例、arch35寄存器路径或3072分块参数。

## 知识点二：规则布局可以把大量小DMA合为少量步进DMA，但地址单位由存储空间决定

深入案例：14 mHC Sinkhorn（源码自述910B3/CANN9.0.0，文档保留多次失败与修正）。

**已核实的源码事实**
- [mhc_sinkhorn.cpp L159-L184](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/14-contest-mhc-sinkhorn-a2/code/op_kernel/mhc_sinkhorn.cpp#L159) 的 LoadRows 让同一行号跨 g 个矩阵进行一次 blockCount=g 的步进搬运；StoreRows 在[L206-L248](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/14-contest-mhc-sinkhorn-a2/code/op_kernel/mhc_sinkhorn.cpp#L206) 对称执行。
- fp32 n<8：GM侧 gap 用字节，UB侧 gap 用32B块。blockLen小于32B仍占一块，UB gap在此是 N_MAX-1。这个“减一”来自本例的具体窗口几何，不能机械套用。
- [优化记录 L1163-L1186](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/14-contest-mhc-sinkhorn-a2/docs/mhc_sinkhorn_optimization.md#L1163) 记录两侧都按字节的版本失败，以及只改存方向的隔离实验；修正后做常量、随机、大g、fp16历史分支、仿真器等分层回归。
- [L1190-L1208](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/14-contest-mhc-sinkhorn-a2/docs/mhc_sinkhorn_optimization.md#L1190) 的A/B/A表把未修改的 V 计算时间作为对照量，与 scalar/MTE2/MTE3 分开看。收益数字仅属于该报告输入和设备。

**数学/硬件原理**
地址若可写成 base+i*rowPitch+w*matrixPitch，改变循环顺序让 w 成为 DMA blockCount 轴，能减少 descriptor 和标量发射次数。这里UB行起点按32B对齐，故能用块间隙表达。减少的是搬运指令条数，而不只是字节量。

**与本项目关联和边界**
我们的raw行6、priors行5、输出行5，行宽为12/24B、20B、10/20B；单个字段只有2B/4B。步进DMA不等于任意细粒度scatter：若把每个字段直接搬成32B UB槽，会引入padding和额外布局成本，而且无法直接证明连续5字段AoS输出更便宜。当前EXP-016本来按整块连续搬运，不能把这个例子的 g×n→n 条DMA改善照搬过来。可以借其“规则地址变换先写公式，再验证每个地址”的方法，不能据此推荐已证实不支持的Scatter。

**未证实**
其官方代码引用有9.2.0背景；本次依据的是作者实际910B3/9.0.0代码与失败/通过记录，未独立编译。源码仍有描述旧路径的注释，判断以实际分支为准。没有源harness，也没有自己的实机复现。

## 知识点三：SwiGLU显示数学融合和scratch复用要按dtype分支审查；测试与统计比“技巧名字”重要

深入案例：59 SwiGLU。

**已核实的源码事实**
- [swi_glu.cpp L30-L60](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/59-swiglu/code/op_kernel/swi_glu.cpp#L30) 分别选择fp32与half/bf16路径，double buffer为1/2的显式模板分派。
- [swi_glu_impl.hpp L40-L78](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/59-swiglu/code/op_kernel/swi_glu_impl.hpp#L40)：fp32初始化ones，计算 -a→Exp→+1→Div(1,denom)→乘a→乘b；仍是基础Exp/Div，不提供“更快指数”或被支持的ExpSub证据。
- [swi_glu_bf16.hpp L55-L99](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/59-swiglu/code/op_kernel/swi_glu_bf16.hpp#L55)：half/bf16先升FP32，用 a/(1+exp(-a)) 直接得到Swish，再将已死的inputTempBuffer用于b转换；最后按CAST_RINT降精度。a/(...)与先倒数后乘a数学相同，浮点舍入不同。
- [swi_glu_single.hpp L81-L109](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases/blob/f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7/59-swiglu/code/op_kernel/swi_glu_single.hpp#L81) 单独处理非32B对齐CopyIn/Out。macro仍是CopyIn→Compute→CopyOut；不能仅凭模板深度2证明所有阶段实机重叠。

**适用性**
数学融合可少一个ones向量/乘法，scratch可复用，但不能把Swish的 a/(...) 直接替换我们的Sigmoid，后者没有a这个分子。EXP-017已经试过部分scratch复用但性能判负；“少UB”不自动等于“更快”。本次不重复推荐。

**版本证据限度**
聚合仓README声明官方案例取自正式v9.0.0；然而59随附API文档包含950/Kirin并链接master，不能作为9.0.0逐API能力证明。这里只把已存在基础Exp/Div/Cast/队列模式作为源码学习，不把更新文档里的产品能力迁移到910B。

## 三类Case落地与后续验证边界

以下构造依据本项目公开输入契约和上面的源码方法；已落地框架与本机运行范围以[validation/README.md](../validation/README.md)为准。2026-10-17已完成612/612 CPU case与10/10框架自测，设备adapter与NPU Mock计时尚未执行；实际两个Judge候选结果见[迭代记录](05-2026-10-17-iterations.md)。

- Coverage：B覆盖1/2/3/8，N覆盖1、标量/向量/32B/分核切分边界、每核704块边界及±1、32768；fp16/fp32 raw、priors始终fp32；确认输出[B,N,5]而非SoA、prior沿B共享、不越界写尾块。具体N由实际host分核公式生成，不只测N=704±1。
- Correctness：固定种子随机；逐行/逐字段签名；0 logits有可手算输出；分别只改某一logit定位字段错位；large positive/negative有限logit测饱和与Exp溢出路径；两个batch给相同输入验证共享priors、给不同输入验证未覆盖；sigmoid对称关系与score乘法关系；非有限值仅在原始题面明确定义时设通过标准。高精度reference与算法顺序reference分开，禁止为了通过新优化而放宽原始容差。
- Mock：记录B/N/dtype/seed/源码SHA/设备/CANN/计时模式/预热/重复次数/统计量；固定集合比较多个旧版本及两候选，观察逐case变化符号和排序，不只看一个平均数。CPU回归只能证明reference/框架自身性质，性能字段必须为null/未运行，不能伪装NPU计时。
- A/B/A：同源A→B→A，若A漂移超过噪声门限，应标记“不确定”，不能由单次回归断定候选失败。Judge日志保留本来精度、时间与失败信息，提交源码SHA与日志绑定；若平台按latest上榜，结束时明确恢复已采纳版本。
- Profiling：Task Duration、device event、host wall、Graph Replay分别统计。MTE/V可重叠，忙时不可简单相加作为总时长分解。作者案例“vec读数稳定”是该负载观察，不是所有优化都只看vec的理由；若改DMA，则V应作为控制而总kernel与MTE才是结果。

