# 学习｜Host 构建：由注册定义生成接口，构建 tiling 库和 aclnn 调用库，最后加入算子包。
# 学习｜这里是工程生成步骤；设备实际运算在 op_kernel 的 C++ 中。
# 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
# 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
# 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
file(GLOB host_src ./*.cpp)

# host侧源码
npu_op_code_gen(
    SRC ${host_src}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

# tiling动态库
npu_op_library(cust_optiling TILING ${host_src})

# aclnn单算子调用的动态库
file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
)