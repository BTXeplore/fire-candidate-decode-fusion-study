# 旧版CPU公式参考

`fire_ref.py`和`run_tests.py`是原项目的CPU参考/测试脚本，使用Python和NumPy。运行`python run_tests.py`会在tests/testdata目录生成测试资料。

这些脚本保留原样，适合阅读输入和解码公式。它们不编译或执行Ascend C kernel，不代表NPU精度或时延验证。真实平台结果见[measurements.json](../results/measurements.json)。

当前验证入口为[validation/README.md](../validation/README.md)：Coverage覆盖契约、布局和尾块；Correctness使用独立数学参考；Mock定义公开的代表规模并比较跨修订耗时趋势。该框架不导入这里的旧runner，避免自动生成旧testdata。

2026-10-17完成612/612 CPU case及10/10框架负控/自测；这只验证CPU数学与地址模型，不代表候选NPU执行。无真实NPU adapter时，设备测试和Mock性能保持SKIPPED。两个候选的实际Judge界面结果见[本轮迭代记录](../docs/05-2026-10-17-iterations.md)，不能混同为CPU框架已验证设备。
