# 从试代码到可复现的探索：Case框架的两步改进

2026-10-22。**这是验证框架迭代，没有新增或替换算子候选。** 推荐的EXP-022及77份冻结源码继续保持原样。本轮尝试了CANNLab的910B3/CANN9.0.0镜像，但两次启动被平台返回HTTP400，最终已关机；未进入终端、未编译运行、没有新设备时间或Judge提交。400的具体业务原因未知，不能直接写成“资源不足”。

## 为什么先完善case

Coverage覆盖规格与地址边界，Correctness用独立数学参考检查精度，Mock检查固定本地规模的性能趋势。它们回答不同问题。CPU参考通过不等于Ascend C通过；本地Mock的B/N/dtype是公开规格内的假设，不是隐藏用例身份。先把这些边界记录清楚，失败才有学习价值。

### 第一步：小步接入，让错误可见

[device_contract.py](../validation/device_contract.py)新增六个smoke：B1/N1零输入、B3/N33字段标签、B8/N705普通输入，均覆盖fp16/fp32。先确认调用与布局，再扩展全量矩阵。

`--fail-fast`在首个失败后停止，保留计划数、实跑数与早停状态。动态库/adapter导入错误单列为infrastructureErrors，零case执行，不伪装成所有输入都算错。HTTP错误字典不满足输出契约，不能记作设备运行。`npuExecuted`仅是可信adapter返回输出后的声明，报告明确标记 `ADAPTER_REPORTED_UNVERIFIED`，不是远程真实性认证。

内存检查分三层：host输入逐字节比较，设备输入同步D2H读回，输出前后哨兵。host输入没变不能证明GM输入没变；NaN预填逻辑输出可发现漏写；哨兵可以发现部分邻接写越界。缺少设备证据unknown，已知损坏fail；严格模式要求证据完整。哨兵不能发现越界读取、UB越界、写后恢复或跨过哨兵的写入，还需真实sanitizer。

### 第二步：根据分核数学构造边界

本内核沿N均匀分核。令实际核数C、TILE_N=T=704：

- N=C*T：每核恰好T个候选。
- N=C*T+1：首核T+1项，触发第二Tile与缓存尾块路径。
- N=C*T-1：最后一个核比整块少一项。

因此全局N=705通常不等于每核越过704。新 `device-boundary` 套件按C及C*T邻点、最大N生成合法输入；C*T超过32768时不生成非法case，也不声称覆盖了不可达路径。C必须从实际平台/tiling核对，不能从Judge时间猜。`--cores`只控制case和CPU模型，不改变真实kernel。

负控试错还发现一个报告缺陷：漏写产生NaN后，旧maxAbsFinite统计也是NaN，严格JSON保存会再次失败。现在仅统计双方均有限的差值，非有限候选独立计数；没有有限对则null。错误仍判失败，容差不变，JSON仍严格。这让最需要排查的失败现场可以保存。

## 怎样运行和学习

从仓库validation目录：

```bash
python -B -m unittest -v test_framework.py test_device_contract.py test_runner_device_contract.py
python -B run.py --suite smoke --fail-fast
python -B run.py --suite all --tier full --report cpu-reference-only.json
```

核对真实核数后，运行独立边界套件；以下40仅是生成器示例，不是实机事实：

```bash
python -B run.py --suite device-boundary --tile 704 --cores 40 --report cpu-boundary-example.json
```

真正的adapter实现并验证后再加 `--adapter /actual/adapter.py --revision EXP-022 --require-device-memory-checks`。目前没有声称已完成的设备adapter。原all/full仍为560 Coverage+16 Correctness+36 Mock，共612例；smoke和设备边界单独运行，不改变历史Mock点集。

本轮CPU侧验证：26/26框架自测、612/612原full、6/6 smoke、48/48 C40边界示例。合成回包仅是检查器负控，临时报告不会进入真实measurements。摘要见[验证记录](../results/validation-2026-10-22.json)。所有关键新增源码都有中文解释，可先读 `smoke_cases/device_boundary_cases`，再读 `validate_memory_evidence` 与runner的异常路径。

## 设备恢复后接什么

进入实例后先运行[只读探针](../validation/device/probe_environment.sh)，核对芯片、驱动、CANN、编译工具和NumPy。网页镜像标题不代替实际环境证据。探针本轮仅做Bash静态语法检查，没有云端执行。

现有 `src/` 是完整ASC SHARED工程；无需先假定torch-npu已安装。可沿“ASC构建→生成ACLNN→C++ C ABI→Python ctypes”接NumPy，但必须现场读取实际生成头文件、包config、张量接口和executor生命周期。顶层INSTALL_PATH硬编码build目录，不能认为CMAKE_INSTALL_PREFIX一定改变产物位置。构建、链接与ABI核对未完成前，不把命令模板写成已验证runner。

官方[900工程构建说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00035.html)提供ASC工程、SHARED二进制目标及SDK路径依据；[900 Event计时说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0023.html)提供记录、同步、elapsed流程。它们不能证明本项目已在实例构建成功，也不补全尚未读取的具体ABI。

测量时区分profiler的kernel duration与event包围的stream时间窗；后者可能含主机提交空隙。构建、分配、初始化、H2D/D2H不计入kernel时间，保留暖机数、所有原始样本、设备版本、源码SHA和时间窗定义。环境可运行后先让22通过，再用冻结23/24研究布局与调度；是否更快仍是假设，不提前承诺收益。
