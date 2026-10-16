# CPU参考与测试

`fire_ref.py`和`run_tests.py`是原项目的CPU参考/测试脚本，使用Python和NumPy。运行`python run_tests.py`会在tests/testdata目录生成测试资料。

这些脚本不编译或执行Ascend C kernel，不代表NPU精度或时延验证。真实平台结果见../results/measurements.json。

后续可按Coverage（输入域与尾块覆盖）、Correctness（最终输出精度）、Mock（有设备时的代表性性能形状）组织测试。今晚只保存现有代码，未新增Mock性能脚本，也未推断隐藏测试输入。
