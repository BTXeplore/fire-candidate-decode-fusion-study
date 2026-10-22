#!/usr/bin/env bash
# 学习｜只读环境探针。请用 bash probe_environment.sh 执行，不要 source。
# 学习｜它不安装软件、不申请设备内存、不编译/运行算子，也不打印整份环境变量。
# 学习｜仅供真正进入实例后归档；本地静态审阅不代表云端已执行成功。
set -u

section() {
    printf '\n[%s]\n' "$1"
}

report_command() {
    local tool_path
    tool_path=$(command -v "$1" 2>/dev/null || true)
    if [[ -n "$tool_path" ]]; then
        printf '%s: %s\n' "$1" "$tool_path"
    else
        printf '%s: NOT_FOUND\n' "$1"
    fi
}

run_readonly() {
    local output status
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    # 学习｜只调用脚本内写死的探测命令；不用 eval，也不执行版本文件内容。
    # 学习｜设备/驱动探针也可能卡住；有GNU timeout时每条最多等待15秒。
    if command -v timeout >/dev/null 2>&1; then
        output=$(timeout 15s "$@" 2>&1)
    else
        output=$("$@" 2>&1)
    fi
    status=$?
    printf '%s\n' "$output" | sed -n '1,100p'
    printf 'exit_status=%s\n' "$status"
}

section 'probe_contract'
printf 'schema=fire-candidate-environment-probe-v1\n'
printf 'mode=READ_ONLY_NO_KERNEL_EXECUTION\n'
printf 'utc='
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'architecture='
uname -m

section 'tools'
for tool in npu-smi cmake ccec bisheng g++ python3 python; do
    report_command "$tool"
done
if command -v cmake >/dev/null 2>&1; then
    run_readonly cmake --version
fi
if command -v ccec >/dev/null 2>&1; then
    run_readonly ccec --version
fi
if command -v bisheng >/dev/null 2>&1; then
    run_readonly bisheng --version
fi
if command -v g++ >/dev/null 2>&1; then
    run_readonly g++ --version
fi

section 'device_inventory'
if command -v npu-smi >/dev/null 2>&1; then
    # 学习｜保留真实芯片 Name 与驱动报告；不把网页镜像标题当成设备证据。
    run_readonly npu-smi info
else
    printf 'npu-smi unavailable; actual device identity is UNVERIFIED\n'
fi

section 'python_numpy'
python_bin=''
if command -v python3 >/dev/null 2>&1; then
    python_bin=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    python_bin=$(command -v python)
fi
if [[ -n "$python_bin" ]]; then
    # 学习｜-B 禁止 Python 为导入模块写入 pyc；不导入 torch/torch_npu。
    PYTHONDONTWRITEBYTECODE=1 "$python_bin" -B - <<'PY'
import platform
import sys
print('python_version=' + sys.version.split()[0])
print('python_executable=' + sys.executable)
print('python_machine=' + platform.machine())
try:
    import numpy as np
except Exception as exc:
    print('numpy=UNAVAILABLE:' + type(exc).__name__)
else:
    print('numpy_version=' + np.__version__)
PY
else
    printf 'Python unavailable\n'
fi

section 'public_sdk_paths_and_versions'
# 学习｜只探测约定的 SDK 根目录及公开的 SDK 根路径变量，不输出整份 env 或密钥。
# 不扫描用户目录、SSH 配置、浏览器状态或认证文件，不递归扫描整个磁盘。
declare -A seen_roots=()
sdk_roots=(
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend/ascend-toolkit/9.0.0
    /usr/local/Ascend/cann
    /usr/local/Ascend/cann-9.0.0
    /home/developer/Ascend/ascend-toolkit/latest
    /home/developer/Ascend/ascend-toolkit/9.0.0
    "${ASCEND_HOME_PATH:-}"
    "${ASCEND_CANN_PACKAGE_PATH:-}"
)
for sdk_root in "${sdk_roots[@]}"; do
    [[ -n "$sdk_root" && -d "$sdk_root" ]] || continue
    [[ -z "${seen_roots[$sdk_root]+present}" ]] || continue
    seen_roots["$sdk_root"]=1
    printf 'sdk_root=%s\n' "$sdk_root"
    for suffix in \
        compiler/tikcpp/ascendc_kernel_cmake/ASCConfig.cmake \
        compiler/tikcpp/ascendc_kernel_cmake/asc-config.cmake \
        include/acl/acl.h \
        include/aclnn/acl_meta.h \
        lib64/libascendcl.so \
        lib64/libnnopbase.so; do
        if [[ -r "$sdk_root/$suffix" ]]; then
            printf 'readable=%s\n' "$sdk_root/$suffix"
        else
            printf 'missing_or_unreadable=%s\n' "$sdk_root/$suffix"
        fi
    done
    for suffix in version.cfg version.info aarch64-linux/ascend_toolkit_install.info \
        x86_64-linux/ascend_toolkit_install.info; do
        version_file="$sdk_root/$suffix"
        [[ -f "$version_file" && -r "$version_file" ]] || continue
        printf 'public_version_file=%s\n' "$version_file"
        # 学习｜安装信息文件可能还有用户名等字段；仅输出版本/架构白名单键。
        sed -n -E '/^[[:space:]]*(version|Version|CANN_VERSION|version_dir|arch|Architecture)[[:space:]]*[:=]/p' \
            "$version_file" | sed -n '1,12p'
    done
done
if [[ -r /usr/local/Ascend/driver/version.info ]]; then
    printf 'public_version_file=/usr/local/Ascend/driver/version.info\n'
    sed -n -E '/^[[:space:]]*(Version|version|package_version)[[:space:]]*[:=]/p' \
        /usr/local/Ascend/driver/version.info | sed -n '1,12p'
fi

section 'interpretation'
printf '%s\n' \
    'An available path is not a compile/link/runtime success.' \
    'No kernel ran; no correctness, memory-safety, or performance result is claimed.' \
    'Verify the actual device is 910B and installed CANN is exactly 9.0.0 before labeling Judge-compatible timings.'
