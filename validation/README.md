# FireCandidateDecodeFusion：Coverage / Correctness / Mock

这是独立验证目录，不改动参赛源码、冻结基线或提交工作流。当前主机运行 CPU；**612 个 case 通过只证明本框架的数学与地址模型通过，绝不等于 Ascend C 候选通过**。

## 使用

依赖 Python 3、NumPy（本机已有）。在本目录执行：

```powershell
python -m unittest -v test_framework.py
python run.py --suite all --tier full --report cpu-validation-2026-10-17.json
python run.py --suite coverage --tier full --tile 768 --cores 8
```

不传 `--report` 不生成报告。不导入任何原项目脚本；避免旧 runner 在 tools/testdata 写入大批文件。所有随机 case 固定 seed。`full` 为 560 个 Coverage + 16 个 Correctness + 36 个 Mock；`quick` 为 54 + 16 + 36。

## 三类 case 各自解决什么

| 类别 | 构造与检查 | 不能证明什么 |
|---|---|---|
| Coverage | full 覆盖 B=1..8、两种 dtype；35 种 N 包含 32/64 向量边界、512/704/1024 Tile 邻点、8×704 分核邻点与最大 N。地址模型验证候选唯一写入、Batch/Candidate/Field 顺序、priors 跨 Batch 共享与输出两端哨兵。 | 不是物理 DMA 模拟，不能检查真实 DataCopyPad 舍入、越界访存、UB 容量、事件或流水竞争。`--tile/--cores` 不自动改变真实内核。 |
| Correctness | 零输入官方例、不同字段/Batch 标签、±最大有限 logit/±88/±90 饱和、近零、中心坐标抵消、fp16 相邻值、微小与混合尺度 priors。独立标量索引 oracle 对照向量 oracle；另用稳定 float64 logistic 诊断舍入。 | 数学 oracle 不是设备执行；float64 仅诊断，不能替换题目 fp32 参考。 |
| Mock | 公开题面 B=1/4/8、N=4096/16384/32768，加 N=512/704/1408 的本地规模假设，两种 dtype，共36点。明确 `hypothesis=true`。 | 不是隐藏用例，也没有已知官方 shape/dtype 映射。无 NPU 时性能标记 SKIPPED，绝不使用 CPU 耗时填补。 |

所有结果先检查 `[B,N,5]` 连续输出及 dtype。fp32 使用 atol=rtol=1e-5；**fp16 使用 fp32 全程运算后 cast 到 fp16 的参考，再用 atol=rtol=3e-3 比较**，不要求位级相等，也不把 fp16 结果直接对照未 cast 的 fp32 参考。`oracleContract` 自比仅验证框架契约，不能解释成独立正确性证据。真正的 CPU 交叉检查在 `independentScalarOracle`；NPU 正确性只看实际 adapter 的 `candidateComparison`。

题面只约束输入有限，没有给 priors 最大值及输出溢出处理规范。本套所选 Correctness 输出可表示；接口可报告匹配的有符号 Inf，但**不声称 Judge 接受溢出结果**。非法输入只用于验证框架拒绝输入，不提交给比赛算子。

## NPU adapter：明确设备边界

只有显式传 `--adapter` 才导入执行该文件，框架本身不编译、不上传、不提交：

```powershell
python run.py --suite all --tier full --adapter C:\path\actual_npu_adapter.py --revision EXP-018 --report npu-EXP-018.json
```

adapter 必须提供 `run(raw, priors, case_metadata)`，返回：

```python
{
    "output": actual_device_output_numpy,  # 已完成设备执行并同步
    "kernel_us": measured_samples,         # 可省略；一旦提供至少5个正有限样本
    "timing_metadata": {
        "kind": "NPU_KERNEL", "synchronized": True,
        "device": "Ascend910B", "cann": "9.0.0"
    }
}
```

adapter 负责实际分配设备输入/输出、调用候选、同步、读回、设备哨兵或 sanitizer、预热、设备计时事件与稳定重复；计时排除 Python、H2D/D2H、编译及初始化。必须保留硬件、驱动、CANN、源码 SHA、预热次数等原始证据。字段只是接口声明，框架无法远程认证计时来源。设备临时环境未提供，因此本次**没有实现虚假的 NPU adapter，也没有 NPU 时延**。有 adapter 后每个 case 检查输入未被修改；Mock 全部比较通过且有时延时才导出完整 `mockRecord`。

## 用性能曲线评价 Mock，而不是猜隐藏输入

默认优先只读 `../results/measurements.json`（公开仓库布局），若不存在再读取 `../publish/fire-candidate-decode-fusion-study/results/measurements.json`（原研究目录布局），也可用 `--official` 指定：保留同一官方 case 序号跨版本身份，读取 EXP-014/015/016/017/恢复016 的真实12点。输出逐版本均值、相对首版本的几何平均比及最差单点比；不把任何序号标成某个 B/N/dtype。

本地有至少三个**同版本、同设备**的 NPU Mock 测量后，可将各报告的 `mockRecord` 按官方版本顺序放入如下 JSON，再运行 `--mock-history`：

```json
{"measurementKind":"NPU_KERNEL","records":["此处应放实际 mockRecord 对象，不能放示例字符串"]}
```

`records` 每个对象需包含 `record`、`valid:true`、`status:"Pass"`、`caseResults:[{"case":"固定本地case身份","timeUs":真实设备中位数}]`。各版本必须具有相同 case 集合，首版本与官方曲线的首版本一致。**不提供任何伪造时间的可运行样本。**分析比较同修订序列的归一化 log 时延趋势、变化方向一致率与平均绝对趋势差；样本不足或 CPU 计时会拒绝。不同数量的官方与 Mock 点只在各自固定点集内汇总，不强行一一配对。

少数修订趋势相近只能说明这个 Mock 可能有代表性，不能证明隐藏分布相同。恢复同一 EXP-016 的两次实测也用于观察测量噪声，不能算成算法提升。即使 Mock 看起来合适，仍须完整 Coverage/Correctness 和真实平台验证，不得预计算答案或按评测结果硬编码路径。

官方结果中的新增 `JUDGE_UI` 记录只允许在 `valid:null`、`comparisonEligible:true`、整体 `status:"Pass"` 且12个唯一序号1～12逐点 `status:"Pass"` 时参与**描述性序号对照**。报告保留 `apiValid:null`，不把界面 Pass 伪造为 API `valid=true`，也不声称已验证同一隐藏 testcase 身份。缺点、重复、非Pass或缺少明确资格会拒绝。此例外只适用于官方展示曲线；真实本地 Mock 历史仍必须 `measurementKind:"NPU_KERNEL"` 且每个记录 `valid:true/status:"Pass"`，不能借用UI例外。

## 只读比较 Judge 结果

`compare_judge.py` 接收两个已有结果文件，不上传、不提交、不修改文件：

```powershell
python -B compare_judge.py baseline-result.json candidate-result.json
```

支持原项目 `iteration-summary.json` 的 `tc/id/t` 格式及网页观测 `ui-result.json` 的 `cases/ordinal/timeUs` 格式。优先按真实 testcase ID 对齐；缺少 ID 时只能按显示序号对照，并在结果中明确身份未经证实。界面 Pass/0.00% 误差不伪装成 API `valid=true/precision_ratio=1`。CPU测试通过、界面观察和原始平台结果保持独立。

## 来源与已验证范围

- 真实题面：原研究包 `Astra-001/problem-statement.txt` 的 §3.2～3.5、§4.1、§4.2（原文未随此学习仓库发布）；旧数学参考：[../tests/fire_ref.py](../tests/fire_ref.py)。公式、范围、容差及公开性能规模均由原题面核对，本文件保留了验证所需规格。
- 地址模型依据：当前推荐 EXP-016 的 [../src/op_kernel/fire_candidate_decode_fusion.cpp](../src/op_kernel/fire_candidate_decode_fusion.cpp) 中 `Init`、`Process`、`LoadRaw`、`FlushOut`。EXP-016 在本仓库的 `src/`，不位于 `history/`。模型只复现候选分区和寻址，不解释实际并发性能。
- 历史真实结果：上面的 `measurements.json`；EX016 两次结果是同源码，不把重新测量当新算法。
- 三类测试的设计动机来自用户提供的 HyosungSink 聊天截图：Coverage 覆盖规格，Correctness 保证正确性精度，Mock 用跨修订耗时曲线验证性能代表性。截图是方法经验，不作为设备接口支持证据。仓库学习由主任务独立核对，不从聊天文字推导硬件结论。

2026-10-17 本机运行：全量 **612/612 CPU case**、**10/10 框架自测**通过。自测故意调换字段、调换 Batch、改 dtype、改分数、引入 NaN、缺失曲线点、冒用 CPU 时延，确认检查会拒绝；新增实际EXP-018的UI格式兼容及不完整/非Pass/UI冒充Mock的负控。设备执行、内存安全和真实性能尚需实际 NPU 环境。
