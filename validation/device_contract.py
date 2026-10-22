"""设备接入的独立辅助检查；校验收到的证据，不把 CPU 检查冒充 NPU 运行。"""
from collections.abc import Mapping

import numpy as np

from cases import Case


def smoke_cases():
    """六个有顺序的首轮探针：启动/类型、字段/Batch、队列复用与非整齐 N。

    N=705 只是全局边界；实际每核 count 由真实 coreNum 决定，所以这些
    探针不声称覆盖了每核 704 的满 Tile。全部通过后才能扩展原全量套件。
    """
    shapes = [(1, 1, "zeros"), (3, 33, "field_tags"), (8, 705, "normal")]
    return [Case(f"smoke-b{b}-n{n}-{dtype}", "smoke", b, n, dtype,
                 20261022 + i, distribution)
            for i, (b, n, distribution) in enumerate(shapes)
            for dtype in ("float16", "float32")]


def device_boundary_cases(tile, cores):
    """按调用者核对的核数构造每核 Tile 邻点；不改变 kernel 的实际分核。

    学习｜本内核先沿 N 分核，所以全局 N=705 通常并不产生每核704的
    满块/尾块转换。对 C 个核，N=C*T 时每核恰好 T 项；N=C*T+1
    使首核多一项，能触发第二个 Tile 与索引缓存的尾块路径。
    C 必须由真实平台/tiling 核对，不能从 Judge 时间猜出来。
    若 C*T 超过题面 N 上限32768，就不生成非法输入，也不声称覆盖该路径。
    """
    if not isinstance(tile, int) or not isinstance(cores, int) or tile < 1 or cores < 1:
        raise ValueError("tile and cores must be positive integers")
    ns = sorted({n for n in (1, cores - 1, cores, cores + 1,
                             cores * tile - 1, cores * tile, cores * tile + 1, 32768)
                 if 1 <= n <= 32768})
    return [Case(f"device-boundary-c{cores}-t{tile}-b{b}-n{n}-{dtype}",
                 "device-boundary", b, n, dtype, 20261022 + b * 32769 + n,
                 "field_tags" if n <= 129 else "normal")
            for b in (1, 3, 8) for n in ns for dtype in ("float16", "float32")]


def _snapshot(array):
    # 按位比较可发现 +0 被改成 -0；np.array_equal 不能发现这种修改。
    # 这些拷贝必须放在设备计时区间外，不能计入算子时间。
    return (array.dtype.str, array.shape, array.strides,
            array.flags.c_contiguous, array.tobytes(order="C"))


def host_input_snapshot(raw, priors):
    """保存 host 输入字节；不代表检查过设备 GM 输入。"""
    return {"raw_pred": _snapshot(raw), "priors": _snapshot(priors)}


def assert_host_inputs_unchanged(raw, priors, before):
    """检查 adapter 是否修改了调用者数组，包括类型/形状/步长/符号零。"""
    for name, actual in (("raw_pred", raw), ("priors", priors)):
        if _snapshot(actual) != before[name]:
            raise AssertionError(f"adapter modified host input: {name}")


def validate_memory_evidence(evidence, raw, priors):
    """检查 adapter 显式返回的 D2H 输入与输出前后哨兵，缺证据保留 unknown。

    evidence 约定：
      synchronized: True（完成 kernel 和 D2H 后再提供这些数组）
      device_inputs: {raw_pred: ndarray, priors: ndarray}
      output_guards: {expected_prefix, actual_prefix, expected_suffix, actual_suffix}
      output_initialized_to_nan: True（正确性调用之前填满逻辑输出区）
    哨兵为一维连续 uint8 数组，前后各至少 32 字节。expected 来自提交前
    保存的模式，actual 必须从同一设备分配读回；不能拿 host 数组充当 D2H。

    函数只能核对调用者提供的数据，不能认证其设备来源。哨兵发现邻接写越界，
    不能发现越界读取、UB 越界、写出哨兵范围或随后恢复原值的写入；仍需设备
    sanitizer。NaN 预填配合 compare_output 能暴露漏写，不能替代哨兵。
    """
    if evidence is None:
        return {"status": "unknown", "missing": ["device memory evidence"], "errors": []}
    if not isinstance(evidence, Mapping):
        return {"status": "fail", "missing": [], "errors": ["evidence must be a mapping"]}
    missing, errors = [], []
    if evidence.get("synchronized") is not True:
        missing.append("synchronized device/D2H completion")
    device_inputs = evidence.get("device_inputs", {})
    if not isinstance(device_inputs, Mapping):
        errors.append("device_inputs must be a mapping")
        device_inputs = {}
    for name, original in (("raw_pred", raw), ("priors", priors)):
        actual = device_inputs.get(name)
        if actual is None:
            missing.append(f"device input readback: {name}")
        elif (not isinstance(actual, np.ndarray) or actual.dtype != original.dtype
              or actual.shape != original.shape or not actual.flags.c_contiguous
              or actual.tobytes(order="C") != original.tobytes(order="C")):
            errors.append(f"device input changed or malformed readback: {name}")
    guards = evidence.get("output_guards", {})
    if not isinstance(guards, Mapping):
        errors.append("output_guards must be a mapping")
        guards = {}
    for side in ("prefix", "suffix"):
        expected, actual = guards.get(f"expected_{side}"), guards.get(f"actual_{side}")
        if expected is None or actual is None:
            missing.append(f"device output {side} guard")
            continue
        arrays_valid = all(isinstance(a, np.ndarray) and a.dtype == np.uint8
                           and a.ndim == 1 and a.flags.c_contiguous and a.size >= 32
                           for a in (expected, actual))
        if not arrays_valid:
            errors.append(f"malformed device output {side} guard")
        elif not np.array_equal(expected, actual):
            errors.append(f"device output {side} guard changed")
    if evidence.get("output_initialized_to_nan") is not True:
        missing.append("logical output initialized to NaN before correctness launch")
    return {"status": "fail" if errors else "unknown" if missing else "pass",
            "missing": missing, "errors": errors,
            "scope": "provided D2H bytes and declared initialization; not source authentication or sanitizer"}
