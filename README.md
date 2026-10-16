# FireCandidateDecodeFusion 源码学习

Ascend 910B / CANN 9.0.0 上的候选框解码与置信度融合算子。这里保存实际提交版本、中文导读和实验对照，方便用手机阅读。

**当前推荐源码：[EXP-016 kernel](src/op_kernel/fire_candidate_decode_fusion.cpp)。** 原测均值13.6133µs，同源码恢复复测13.5008µs；12/12 Pass、precision_ratio全1。2026-10-12最后核对榜单为第13名、47.11分，这是历史记录，不代表今天的实时排名。

## 今晚从这三处开始

1. [中文入门导读](docs/01-source-walkthrough.md)：输入、Sigmoid、解码、AoS/SoA、队列。
2. [推荐版本与读代码顺序](docs/02-exp016-reading.md)：理解为什么先回收临时空间，再把tile从512扩到704。
3. [EXP-016 kernel](src/op_kernel/fire_candidate_decode_fusion.cpp)：先找`ComputeTile`，再读`Process`和`LoadPriors`。

手机浏览器登录GitHub后可直接阅读。给网页版GPT讲代码时，请上传相应文件或粘贴函数；聊天页面不一定能读取仓库中的每个文件。

## 源码与历史

| 目录 | 含义 |
| --- | --- |
| [src/](src/) | 推荐的EXP-016完整七文件，与冻结候选逐字一致 |
| [history/EXP-014/](history/EXP-014/) | 输出双缓冲版本 |
| [history/EXP-015/](history/EXP-015/) | raw输入最后读取后提前释放 |
| [history/EXP-017/](history/EXP-017/) | 精度通过、性能退步的未采用方案 |
| [results/measurements.json](results/measurements.json) | 精简后的真实逐点结果与来源 |
| [tests/](tests/) | CPU参考与测试脚本，不代表NPU执行或性能 |
| [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json) | 版本来源、文件SHA-256及Git blob SHA |

## 真实实验记录

| 记录 | Judge | 平均时延µs | 结果 |
| --- | ---: | ---: | --- |
| EXP-014 | 708802 | 14.4792 | 12/12 Pass |
| EXP-015 | 748848 | 14.3842 | 12/12 Pass |
| EXP-016 | 759655 | 13.6133 | 12/12 Pass，采用 |
| EXP-017 | 760514 | 14.4683 | 12/12 Pass，较016慢6.28%，未采用 |
| 恢复EXP-016原源码 | 760625 | 13.5008 | 12/12 Pass，与原016源码完全一致 |

同代码复测的变化不是新算法收益。历史比较没有受控重复A/B或完整设备profiling，不能将差异充分归因为某一条源码改动。

## 算子做什么

输入raw_pred为`[B,N,6]`，每行`[tx,ty,tw,th,obj_logit,fire_logit]`；priors为FP32的`[N,5]`，每行`[gx,gy,stride,anchor_w,anchor_h]`。输出为`[B,N,5]`，顺序`[x,y,w,h,fire_score]`。

```text
s(z) = 1 / (1 + exp(-z))
x = (2*s(tx) - 0.5 + gx) * stride
y = (2*s(ty) - 0.5 + gy) * stride
w = (2*s(tw))^2 * anchor_w * stride
h = (2*s(th))^2 * anchor_h * stride
fire_score = s(obj_logit) * s(fire_logit)
```

FP16输入先转FP32计算，最后输出才转回FP16；FP32输入保持FP32。代码中的代数重排、FusedMulAdd与Cast方式请以对应版本源码为准。

## 运行边界

这七份文件来自CANN Judge自定义工程。直接浏览不需要设备；目标编译/调试需要配套CANN SDK、编译工程和Ascend设备。这里不包含SDK、设备账号、浏览器连接代码或自动评测任务。CPU参考脚本需要Python与NumPy；它只能帮助理解公式，不能测得910B kernel时延。

## 明天的阅读书签

[HyosungSink/CANN-AscendC-optimization-Cases](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases)，本次发现的固定提交为`f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7`。用户要求今晚先完成上传，进一步阅读留到明天。截图里的建议包括Coverage、Correctness、Mock三类测试；这是阅读线索，不表示该仓库已经提供了我们任务的隐藏测试形状或完整测试脚本。

## 公开范围与归属

本仓公开算子源码、整理后的中文导读和精简结果。原始评测响应、请求载荷、用户身份字段、浏览器令牌、本机环境备份和私人聊天图片未收录。第三方代码只链接到原仓，不整库重新发布；CANN模板、外部接口及第三方资料保留其原有权利和许可，本次未额外声明统一开源许可证。
