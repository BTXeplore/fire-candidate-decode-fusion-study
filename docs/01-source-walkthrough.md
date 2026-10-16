> 这是2026-10-12以EXP-015讲解基础的导读，对应源码在[history/EXP-015](../history/EXP-015/)。推荐基线为[src/](../src/)中的EXP-016，TILE_N=704；手机阅读时以函数名定位，历史行号与本地快捷键仅作参考。

# FireCandidateDecodeFusion 源码导读

本导读依据 `history/EXP-015` 的源码和原实验归档。行号对应刚复制的源码；你添加注释后，使用函数名定位会更可靠。

## 1. 先理解它接收与产生什么

| 对象 | 形状 | 代码里的类型或字段 |
|---|---|---|
| `raw_pred` | `[B,N,6]` | 原始预测，fp16 或 fp32 |
| `priors` | `[N,5]` | 固定 fp32；grid x、grid y、stride、anchor w、anchor h |
| `decoded` | `[B,N,5]` | 解码后的五个字段，类型与 raw_pred 一致 |

`B` 是 batch 数，`N` 是每个 batch 的候选数。可以先想成：有 B 张表，每张表 N 行，每行原始预测有 6 个数字，最终输出 5 个数字。

来源：Host 文件 `TilingFunc` / `InferShape` / `InferDataType`，kernel 的 `RAW_F`、`PRI_F`、`OUT_F` 与 `LoadPriors` 中的五次 Gather。

## 2. 七个文件各管什么

| 文件 | 先理解什么 |
|---|---|
| `op_kernel/fire_candidate_decode_fusion.cpp` | NPU 上真正的数据搬运与计算 |
| `op_host/fire_candidate_decode_fusion.cpp` | 读取形状、选择类型、分配核的工作量、定义算子输入输出 |
| `op_kernel/fire_candidate_decode_fusion_tiling.h` | Host 与 kernel 共同理解的参数结构体 |
| `op_kernel/tiling_key_fire_candidate_decode_fusion.h` | 数据类型与模板选择的关系 |
| 三个 `CMakeLists.txt` | 构建工程；初学阶段可以最后读 |

Host 可以理解为准备工作的人；kernel 是设备上执行工作的程序。`tiling` 是交给 kernel 的任务单。

## 3. 先看 ComputeTile：一次处理一个 Tile

位置：kernel **203 行**。

`TILE_N=512` 是本源码的常量。`count` 是这一次实际要处理的候选数；最后一块不足 512 时，它可能更小。不要把 Tile 数量和 batch 数混在一起。

### 第一步：拿到输入，转成计算使用的形式

`rawIn = inRawQue.DeQue<DT_RAW_PRED>()` 取出已经搬入的局部输入。

如果原始数据是 half，先用 `Cast` 转成 float，再取负；如果已经是 float，直接取负。`if constexpr` 根据模板类型在编译时选择分支，不是每个元素运行一次普通 if。

### 第二步：计算 Sigmoid

数学公式：

```text
σ(x) = 1 / (1 + exp(-x))
```

这段代码的对应关系：

```text
Muls(..., -1) → -x
Exp          → exp(-x)
Adds(..., 1) → 1 + exp(-x)
Div          → 1 / (1 + exp(-x))
```

用 `x=0` 验证：`exp(0)=1`，最终结果为 `0.5`。这个例子只说明数学流程，不是完整的 fp16/fp32 精度测试。

### 第三步：从每个候选的六个字段取出六条向量

六次 `Gather` 把 Sigmoid 结果拆成 `sx`、`sy`、`sw`、`sh`、`obj`、`fire`。

每一条向量容纳一批候选的同一个字段。这样后续的向量指令就能一次处理多个候选。

### 第四步：解码与打分

`LoadPriors` 已经把部分 prior 变成可复用的偏移和系数。`ComputeTile` 使用这些数据计算位置和宽高，并用 `Mul(obj, obj, fire, count)` 得到联合分数。

注意：`gx`、`aw` 等局部变量在 `LoadPriors` 中被改写为计算系数，后面不再简单等同于输入里的原始字段。阅读优化代码时，要追踪变量被赋值后的含义。

### 第五步：输出重排与排队

中间数据按字段分开，输出必须按候选交错。`outIdx` 和最后的 `Gather` 完成这种重排。fp16 输出还需要最后的 `Cast`。

`outQue.EnQue(outLoc)` 后记录待写回的位置，实际写回由 `FlushOut` 处理。进入下一次 `ComputeTile` 时先调用 `FlushOut`；每个 N Tile 的所有 batch 结束后，`Process` 还会调用一次，写回最后的结果。

## 4. 再看 Process：把一次计算变成完整任务

位置：kernel **107 行**。

```text
对本核负责的 N 范围逐 Tile 处理：
  LoadPriors：准备当前 Tile 的 priors
  LoadRaw(0)：先准备 batch 0
  对每个 batch b：
    如果还有 b+1，先 LoadRaw(b+1)
    ComputeTile(b)
  FlushOut：写回最后一份输出
```

这段顺序说明程序有预先搬入下一 batch 的安排。它本身不能证明 DMA 与向量计算在实机上重叠了多少；那需要 profiling 和事件依赖证据。

来源：`Process`、`LoadRaw`、`ComputeTile`、`FlushOut` 的真实调用顺序。

## 5. 认识局部内存与队列

| 名称 | 在这份源码里的作用 |
|---|---|
| `GlobalTensor` | 指向 GM 中的完整输入或输出 |
| `LocalTensor` | 访问局部缓冲中的数据；切片常表示缓冲视图 |
| `TBuf` | 中间计算、索引等临时缓冲 |
| `TQue` | 在数据搬入、计算、写回阶段之间传递缓冲 |
| `AllocTensor` | 从队列申请可用槽 |
| `EnQue` | 把准备好的张量入队 |
| `DeQue` | 在消费阶段取出张量 |
| `FreeTensor` | 完成使用后释放队列槽，便于后续复用 |

不要把 `FreeTensor` 当作“擦掉数组”。关键问题是这个槽何时能再使用，以及最后一次读取与槽复用之间的依赖是否正确。

在源码中沿着同一个变量找 `AllocTensor → EnQue → DeQue → FreeTensor`，比只记接口名称更有用。

## 6. AoS、SoA 与固定索引

AoS 的直观样子：每个候选的一组字段放在一起。

```text
候选0：x0 y0 w0 h0 s0
候选1：x1 y1 w1 h1 s1
```

SoA 的直观样子：同一个字段放在一起。

```text
x 向量：x0 x1 ...
y 向量：y0 y1 ...
w 向量：w0 w1 ...
```

代码实际字段间距固定为 `TILE_N=512`。即便 `count=3`，`sy=fv[TILE_N]` 也不是接在三个 x 值之后，而是在固定的第 512 个 float 位置开始。

输出第 `m` 个标量对应：

```text
j = floor(m / 5)       // 第几个候选
k = m % 5             // 第几个输出字段
源元素位置 = k * 512 + j
源码里的 float Gather 字节偏移 = 4 * (k * 512 + j)
```

来源：`EnsureScratch` 中生成的 `outIdx` 和 `fieldBuf` 的分段位置。复杂的整数向量表达式可以最后研究；先用 `m=0、1、5、6` 验证上面的布局关系。

## 7. 这一轮到底改了什么

双击 `打开014与015差异.cmd`。两份源码的计算公式和接口不变，主要差别是这行的位置：

```cpp
inRawQue.FreeTensor(rawIn);
```

EXP-014 把它放在 ComputeTile 末尾；EXP-015 放在读取 rawIn 的类型分支结束之后、Exp 之前。

**源码事实**：之后不再读取 rawIn，后续使用独立的计算缓冲。**待验证的性能解释**：提前释放槽可能减少后续槽复用等待；本次单次时延差不能证明这一解释。

原始结果：EXP-015 平均 14.384166666666667µs，EXP-014 平均 14.47916666666667µs。两者都是 12/12 Pass；报告保留了一个增加 0.02µs 的测试点。

## 8. 四道阅读练习

1. 若输入是 `[B=2,N=3,6]`，raw_pred、priors 和 decoded 各有多少个标量？
2. `LoadRaw` 的 `(b * nPerBatch + n0) * RAW_F`，每个乘法分别把哪一级坐标变成哪一级偏移？
3. 为什么 `LoadPriors` 在 batch 循环外面？从输入形状 `[N,5]` 和源码调用顺序回答。
4. `myN=600`、`TILE_N=512` 时，一次核任务会分成几个 Tile，每个 Tile 的 count 是多少？

参考答案：第一题分别是 36、15、30 个标量；第四题是两个 Tile，count 分别为 512 和 88。第二、三题先写下你的解释，我们再一起核对。

## 9. 下一步再学 Host

打开 `history/EXP-015/op_host/fire_candidate_decode_fusion.cpp`：先读 `TilingFunc`，再读 `InferShape`、`InferDataType`。Host 把 batch、候选数、核数与分配结果写到 tiling 结构体；kernel 的 `Init` 再读这些字段。

`nPerCore=N/coreNum`、`nTail=N%coreNum` 表示前 nTail 个核多分一个候选。实际核数来自平台并被 N 限制；这份源码没有把参与核数写死成 8。不要把某次测试的硬件观察当作所有运行的代码常量。
