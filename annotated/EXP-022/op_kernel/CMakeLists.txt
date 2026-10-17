# 学习｜Kernel 构建：收集本目录 Ascend C 源码，关联 Host tiling 库，加入根目录声明的算子包。
# 学习｜构建成功、设备精度通过与性能改善是三个不同结论；学习副本未重新提交评测。
# 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
# 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
# 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
# kernel源码
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

# kernel动态库
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# 添加到package中
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)