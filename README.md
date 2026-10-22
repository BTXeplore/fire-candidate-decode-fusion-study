# FireCandidateDecodeFusion 源码学习

Ascend 910B / CANN 9.0.0上的候选框解码与置信度融合算子。这里保存实际候选源码、中文导读和真实实验对照，方便用手机阅读。

**当前推荐源码：[EXP-022 kernel](src/op_kernel/fire_candidate_decode_fusion.cpp)，状态为best_observed，稳定提速尚未证实。** 首次Judge892657显示12/12 Pass、均值13.3750µs；同源码确认Judge893308显示12/12 Pass、均值12.9400µs，逐点显示错误均0.00%。这是界面观测，API valid和precision_ratio未知。同轮原016前后对照漂移约+2.92%，不能把全部差异归因于代码。

原EXP-016七文件完整保存在[history/EXP-016](history/EXP-016/)，继续作为回归参照。src与冻结022逐字节相同，没有叠加020的数学改动。[本轮交叉审阅与测量记录](docs/06-exp020-exp022-cross-review.md)说明源码差异、CPU边界、Gemini建议核验及采用理由。五轮平台归档均已逐份核验7/7。

本次公开榜单核对为**第14名、47.34分**，对应提交时间2026-10-17 17:59:21，榜单12个时延逐一等于022确认轮。核对发生于2026-10-17 18:06（Asia/Shanghai）；这是该时刻的真实观测，非持续排名保证。

**想躺着学代码，可以直接打开[EXP-022 逐段中文注释版](annotated/EXP-022/README.md)。** 七份完整文件解释索引数学、分核、缓冲生命周期、队列、精度和尾块；原始已评测源码继续保存在 `src/`。去掉新增注释后与原码逐行一致，注释版未单独提测。

## 2026-10-18：今晚的一版迭代与学习

当前继续推荐EXP-022。新候选EXP-023字段间距由704改为712，12/12 Pass，均值12.8092µs；原22前/后控为13.0292/13.1383µs。23对前控仅5点快、7点慢，均值与几何指标给出不同取舍，暂不采用，平台已恢复原22。

- [这段优化经历真正学到的七个主题](docs/07-operator-optimization-lessons.md)：融合、布局、生命周期、复用、流水、精度与测量。
- [EXP-023完整逐点结果与未采用原因](docs/08-exp023-layout.md)。
- [023候选源码](history/EXP-023/)：仅布局改动，不替换推荐源码。

## 2026-10-22：独立临时槽调度实验

新候选EXP-024通过12/12，显示均值13.1383µs；原22前/后控为13.2625/12.5650µs。本次只调整四Mul的顺序和临时地址，浮点乘法树、T、UB及队列不变；改善尚不足以升级，平台和推荐源码继续保留EXP-022。

原22的新低观测更新为12.565µs（Judge920494）。这是同源码复测，不是24的代码收益；历史12.94µs等测量仍保留。

[完整源码差异、逐点结果与未采用理由](docs/09-exp024-square-schedule.md)。

## 2026-10-22：先完善探索与学习框架

新增6个smoke、按实际核数生成的Tile边界、设备内存证据门槛、失败即停，以及NaN/Inf失败报告保存修复；关键理由已写入中文源码注释。原612个CPU数学/地址用例、6个smoke、48个C40边界示例和26项框架自测通过，**均不代表NPU内核通过**。

CANNLab已选择910B3/CANN9.0.0镜像，但两次启动HTTP400，最终关机；未进入终端、没有新设备测量或Judge提交。推荐源码EXP-022保持不变。[本轮为什么这样构造case，以及后续设备接入边界](docs/10-cannlab-case-framework.md)。

## 从这三处开始

1. [中文入门导读](docs/01-source-walkthrough.md)：输入、Sigmoid、解码、AoS/SoA和队列；以历史015解释基础。
2. [016历史导读与022阅读差异](docs/02-exp016-reading.md)：先理解704与scratch生命周期，再看一次构造完整索引。
3. [EXP-022 kernel](src/op_kernel/fire_candidate_decode_fusion.cpp)：先读EnsureScratch，再读LoadPriors、ComputeTile和Process。

手机浏览器可直接阅读公开仓库。给网页版GPT讲代码时，请上传相应文件或粘贴函数；聊天页面不一定能读取仓库中的每个文件。本地快捷键与历史行号仅供参考，优先按函数名定位。

## 源码与历史

|目录|含义|
|---|---|
|[src/](src/)|当前推荐EXP-022完整七文件，冻结源逐字节副本|
|[history/EXP-014/](history/EXP-014/)|输出双缓冲版本|
|[history/EXP-015/](history/EXP-015/)|raw输入最后读取后提前释放|
|[history/EXP-016/](history/EXP-016/)|保留的原版704回归参照|
|[history/EXP-017/](history/EXP-017/)|精度通过、布局与Tile组合性能退步，未采用|
|[history/EXP-018/](history/EXP-018/)|跨N Tile预取；收益尚未证实稳定|
|[history/EXP-019/](history/EXP-019/)|priors/raw搬入重排；无稳健收益暂不采用|
|[history/EXP-020/](history/EXP-020/)|完整Tile合并score倒数；12/12 Pass，未推荐|
|[history/EXP-022/](history/EXP-022/)|一次构造索引与首次搬运窗口；当前best_observed|
|[results/measurements.json](results/measurements.json)|19条精简真实测量，包含2026-10-18与2026-10-22记录|
|[validation/](validation/)|Coverage / Correctness / Mock框架，区分CPU与NPU证据|
|[tests/](tests/)|旧版CPU公式参考与测试脚本，保留学习用途|
|[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)|77份冻结源码来源、字节数、SHA256及Git blob SHA|

未提交设备的EXP-021生命周期解耦草稿没有评测结论，不计入本轮两个实测版本。

## 本轮五次真实测量（2026-10-17）

实际顺序为原16→020→022→原16→同源022确认；确认是复测，不是第三个新候选。均值从Judge显示到0.01µs的逐点时延计算。

|记录|Judge|平均时延µs|界面结果|
|---|---:|---:|---|
|前置原EXP-016对照|891118|13.7492|12/12 Pass|
|EXP-020|891999|13.6175|12/12 Pass|
|EXP-022首次|892657|13.3750|12/12 Pass|
|后置原EXP-016对照|893009|14.1508|12/12 Pass|
|EXP-022同源码确认|893308|12.9400|12/12 Pass|

五轮均显示逐点错误0.00%；没有取得API valid、precision_ratio或隐藏case身份。022两次均值均较前置原版低，但原版前后约+2.92%的漂移、022自身两次约-3.25%的差异表明测量状态变化明显。暂将022选为**当前最佳观测源码**，不是已证明的稳定速度提升；没有真实设备profiling、随机交错或多轮受控重复。

022完整首次索引构建的源码API调用由35变9，全部浮点解码数学、T=704、UB和队列保持016。源码调用计数不是机器指令数或具体收益预测。完整逐点表、尾块和数值验证见[交叉审阅记录](docs/06-exp020-exp022-cross-review.md)。

## 历史实验记录

|记录|Judge|平均时延µs|当时结论|
|---|---:|---:|---|
|EXP-014|708802|14.4792|12/12 Pass|
|EXP-015|748848|14.3842|12/12 Pass|
|EXP-016|759655|13.6133|12/12 Pass，当时采用|
|EXP-017|760514|14.4683|12/12 Pass，组合较016慢6.28%，未采用|
|恢复原EXP-016|760625|13.5008|12/12 Pass，同原版源码|
|EXP-018，2026-10-17界面|877556|13.4742|12/12 Pass，收益未定|
|EXP-019，2026-10-17界面|877809|13.8450|12/12 Pass，暂不采用|
|恢复原EXP-016，2026-10-17界面|878046|13.8108|12/12 Pass，上一轮参照|

前五条来自历史API记录；之后的界面记录保留其API字段未知的区别。同代码复测变化不是新算法收益；18→19→16并非受控A/B/A，详见[上一轮数据搬运迭代](docs/05-2026-10-17-iterations.md)。2026-10-12最后核对榜单为第13名、47.11分，这是历史信息，不代表当前实时排名。

## 三类Case框架

[validation/README.md](validation/README.md)给出独立数学oracle、地址/布局模型、设备adapter协议和跨修订曲线比较。2026-10-17本机完成**612/612 CPU case**与**10/10框架负控/自测**；负控故意注入字段、Batch、dtype、分数和NaN错误，确认框架能拒绝。

这些CPU结果不能证明真实Ascend编译、DMA边界、异步事件或NPU精度。当前只有CANN Judge，Mock设备性能仍未测，不能用CPU计时补位；两项新候选的额外整数/数值扫描与真实Judge结果分别记录。

## 算子做什么

输入raw_pred为[B,N,6]，每行[tx,ty,tw,th,obj_logit,fire_logit]；priors为FP32的[N,5]，每行[gx,gy,stride,anchor_w,anchor_h]。输出为[B,N,5]，顺序[x,y,w,h,fire_score]。

~~~text
s(z) = 1 / (1 + exp(-z))
x = (2*s(tx) - 0.5 + gx) * stride
y = (2*s(ty) - 0.5 + gy) * stride
w = (2*s(tw))^2 * anchor_w * stride
h = (2*s(th))^2 * anchor_h * stride
fire_score = s(obj_logit) * s(fire_logit)
~~~

FP16输入先转FP32计算，最后输出才转回FP16；FP32输入保持FP32。代数重排、FusedMulAdd与Cast方式以对应版本源码为准。

## 运行与版本边界

七文件来自CANN Judge自定义工程。浏览不需要设备；目标编译/调试需要配套CANN SDK、工程与Ascend设备。本仓不包含SDK、设备账号、浏览器连接代码或自动评测任务。CPU脚本需要Python与NumPy，不能测得910B kernel时延。

本仓源码和Judge目标为910B/CANN9.0.0。新版本文档、master、A3/950能力不能直接作为该环境的接口支持证据；无版本或实机取证的Bank、微码、核数结论保持未证实。

## 已完成的开源学习与交叉审阅

已选择性深读[HyosungSink/CANN-AscendC-optimization-Cases](https://github.com/HyosungSink/CANN-AscendC-optimization-Cases)固定提交f5ae98458b8989ac79eb8d0ff09e41bfdeb82ee7的05 Cast-once、14 Sinkhorn和59 SwiGLU，重点是缓冲生命周期、真实流水依赖、步进DMA、数学融合及受控测量，见[学习笔记](docs/04-hyosungsink-study.md)。三类测试框架是本项目独立实现，没有把聚合仓误当含完整测试集。

本轮在用户现有Gemini Argon网页交叉审阅公开016 kernel，并采纳经容量与整数验证的完整ramp建议。未采纳缺少版本/实机证据的48核、256B Bank及Inf微码断言；最终三项风险与核验边界见[本轮记录](docs/06-exp020-exp022-cross-review.md)。

## 公开范围与归属

只公开算子源码、中文笔记及精简结果。原始评测响应、ObjectID、用户身份、浏览器令牌、本机环境备份、Gemini原始会话及私人聊天图片未收录。第三方代码只链接原仓，不整库重发；CANN模板、接口和第三方资料保留原有权利与许可，本次未额外声明统一开源许可证。
