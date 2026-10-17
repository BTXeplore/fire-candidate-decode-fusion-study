# 学习｜根构建文件：查找配套 ASC 工具链，默认目标 ascend910b，创建 custom 包并加入 Host/Kernel 子目录。
# 学习｜本仓不包含 SDK；这些 CMake 函数由配套 ASC/CANN 工程提供，不是 Python 测试入口。
# 学习｜EXP-022 中文学习副本：只增加本前缀的整行注释，原有代码逐行保留。
# 学习｜真实评测与冻结源码见仓库 src/、history/EXP-022/；本目录未单独提交 Judge。
# 学习｜建议边读函数边对照 annotated/EXP-022/README.md；所有性能因果仍以实测证据为准。
cmake_minimum_required(VERSION 3.16.0)

project(custom_project)

find_package(ASC REQUIRED)

if(NOT DEFINED ASCEND_COMPUTE_UNIT)
    set(ASCEND_COMPUTE_UNIT ascend910b)
endif()

set(package_name custom)

npu_op_package(${package_name}
    TYPE SHARED
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
