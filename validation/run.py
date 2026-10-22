"""Deterministic Coverage / Correctness / Mock runner; default CPU only."""
import argparse
from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from cases import cases, make_inputs
from oracle import reference, scalar_reference, stable_reference_f64, compare_output
from layout_model import check_layout
from curve_analysis import analyze
from device_contract import (smoke_cases, device_boundary_cases, host_input_snapshot,
                             assert_host_inputs_unchanged, validate_memory_evidence)


HERE = Path(__file__).resolve().parent


def default_official_path():
    # Published checkout: ../results. Original research workspace: ../publish/... .
    local = HERE.parent / "results/measurements.json"
    fallback = HERE.parent / "publish/fire-candidate-decode-fusion-study/results/measurements.json"
    return local if local.is_file() else fallback


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("user_npu_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise ValueError("adapter must define run(raw, priors, case_metadata)")
    return module.run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "smoke", "device-boundary", "coverage", "correctness", "mock"], default="all")
    parser.add_argument("--tier", choices=["quick", "full"], default="quick")
    parser.add_argument("--tile", type=int, default=704)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--adapter", type=Path, help="explicit opt-in: imports and executes this NPU adapter")
    parser.add_argument("--official", type=Path, default=default_official_path())
    parser.add_argument("--mock-history", type=Path)
    parser.add_argument("--revision", help="measured NPU candidate name; enables a Mock record for history")
    parser.add_argument("--report", type=Path, help="only writes a report if explicitly requested")
    # 学习｜设备首次接入先跑六个 smoke，失败即停；不要因一个构建/运行错误
    # 连续发起612次调用。没有这些开关时保留原完整套件的行为。
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed case")
    parser.add_argument("--require-device-memory-checks", action="store_true",
                        help="require synchronized input readbacks and output guards from adapter")
    args = parser.parse_args(argv)
    if args.require_device_memory_checks and not args.adapter:
        parser.error("--require-device-memory-checks requires an actual --adapter")
    if args.tile < 1 or args.cores < 1:
        parser.error("--tile and --cores must be positive")
    adapter, load_error = None, None
    try:
        adapter = load_adapter(args.adapter) if args.adapter else None
    except Exception as exc:
        # 学习｜导入/动态库加载失败属于设备接入故障，连kernel都未调用，
        # 应留下零执行报告；不能给每个case填同一失败或丢失整份记录。
        load_error = f"{type(exc).__name__}: {exc}"
    groups = ["coverage", "correctness", "mock"] if args.suite == "all" else [args.suite]
    report = {"schemaVersion": 1, "mode": "NPU_ADAPTER" if args.adapter else "CPU_REFERENCE_ONLY",
              "npuExecuted": False, "officialShapeMetadataKnown": False,
              "adapterInvoked": False,
              "executionEvidence": "ADAPTER_REPORTED_UNVERIFIED" if args.adapter else "NO_DEVICE",
              "infrastructureErrors": [load_error] if load_error else [],
              "numpy": np.__version__, "python": sys.version.split()[0],
              "deviceMemoryChecksRequired": args.require_device_memory_checks,
              "layoutModelParameters": {"tile": args.tile, "cores": args.cores,
                                        "changesActualKernel": False},
              "stoppedEarly": False, "cases": [], "failed": []}
    # 学习｜plannedCaseCount 与实际 caseCount 分开，早停的一例通过不能冒充
    # 完整套件通过；后续 Mock 的有效性也必须要求执行完计划的全部用例。
    def selected_cases(group):
        if group == "smoke":
            return smoke_cases()
        if group == "device-boundary":
            return device_boundary_cases(args.tile, args.cores)
        return cases(group, args.tier)

    selected = [(group, case) for group in groups for case in selected_cases(group)]
    report["plannedCaseCount"] = len(selected)
    if load_error:
        report["stoppedEarly"] = True
    for group, case in ([] if load_error else selected):
        raw, priors = make_inputs(case)
        expected = reference(raw, priors)
        r = {"case": case.metadata()}
        try:
            # Identity comparison tests framework contracts only; it must
            # never appear as evidence that an NPU candidate is correct.
            r["oracleContract"] = compare_output(expected, raw, priors)
            if not r["oracleContract"]["passed"]:
                raise AssertionError("oracle contract failed")
            if group in ("coverage", "device-boundary"):
                r["layoutModel"] = check_layout(case.batch, case.n, args.tile, args.cores)
            if group == "correctness":
                r["independentScalarOracle"] = compare_output(scalar_reference(raw, priors), raw, priors)
                if not r["independentScalarOracle"]["passed"]:
                    raise AssertionError("independent scalar/vector oracle mismatch")
                wide = stable_reference_f64(raw, priors)
                r["diagnosticMaxAbsVsStableF64"] = float(np.max(np.abs(expected.astype(np.float64) - wide)))
                if not np.isfinite(expected).all():
                    raise AssertionError("selected correctness inputs should have representable output")
                if not ((expected[..., 4] >= 0) & (expected[..., 4] <= 1)).all():
                    raise AssertionError("score outside [0,1]")
            if adapter:
                before = host_input_snapshot(raw, priors)
                report["adapterInvoked"] = True
                result = adapter(raw, priors, case.metadata())
                # 学习｜HTTP错误字典不是设备输出；先核对返回契约。即使
                # 契约正确，这仍是可信 adapter 的执行声明，框架不能认证来源。
                if not isinstance(result, Mapping) or not isinstance(result.get("output"), np.ndarray):
                    raise ValueError("adapter must return a mapping with a NumPy device output")
                report["npuExecuted"] = True
                assert_host_inputs_unchanged(raw, priors, before)
                # 学习｜host 数组没变不能证明 GM 输入没变。设备证据必须由
                # adapter 在同步后 D2H 读回；检查器只验证收到的数据与声明。
                memory = validate_memory_evidence(result.get("memory_checks"), raw, priors)
                r["deviceMemory"] = memory
                if memory["status"] == "fail":
                    raise AssertionError("device memory evidence failed: " + "; ".join(memory["errors"]))
                if args.require_device_memory_checks and memory["status"] != "pass":
                    raise AssertionError("device memory evidence incomplete: " + "; ".join(memory["missing"]))
                r["candidateComparison"] = compare_output(result["output"], raw, priors)
                if not r["candidateComparison"]["passed"]:
                    raise AssertionError("NPU output comparison failed")
                samples = result.get("kernel_us")
                timing = result.get("timing_metadata", {})
                if samples is not None:
                    samples = [float(x) for x in samples]
                    if (len(samples) < 5 or not all(np.isfinite(samples)) or min(samples) <= 0
                            or timing.get("kind") != "NPU_KERNEL"
                            or timing.get("synchronized") is not True
                            or timing.get("device") != "Ascend910B"
                            or timing.get("cann") != "9.0.0"):
                        raise ValueError("timing needs >=5 real synchronized 910B/CANN9.0.0 kernel samples")
                    r["npuTiming"] = {"samplesUs": samples, "medianUs": float(np.median(samples)),
                                      "minUs": min(samples), "metadata": timing}
            elif group == "mock":
                r["performance"] = "SKIPPED: no NPU adapter; shape hypothesis only"
        except Exception as exc:
            r["error"] = str(exc)
            report["failed"].append(case.name)
        report["cases"].append(r)
        if args.fail_fast and report["failed"]:
            report["stoppedEarly"] = len(report["cases"]) < len(selected)
            break
    if "mock" in groups and args.official.is_file():
        with args.official.open(encoding="utf-8-sig") as f:
            official = json.load(f)
        mock = None
        if args.mock_history:
            with args.mock_history.open(encoding="utf-8-sig") as f:
                mock = json.load(f)
        report["curveAnalysis"] = analyze(official, mock)
    report["caseCount"] = len(report["cases"])
    report["passedCases"] = report["caseCount"] - len(report["failed"])
    mock_results = [r for r in report["cases"] if r["case"]["group"] == "mock"]
    if adapter and args.revision and mock_results:
        complete = (not report["failed"] and not report["stoppedEarly"]
                    and len(mock_results) == sum(group == "mock" for group, _ in selected)
                    and all("npuTiming" in r for r in mock_results))
        report["mockRecord"] = {"record": args.revision, "valid": complete,
                                "status": "Pass" if complete else "Incomplete",
                                "measurementKind": "NPU_KERNEL",
                                "caseResults": [{"case": r["case"]["name"],
                                                 "timeUs": r["npuTiming"]["medianUs"]}
                                                for r in mock_results if "npuTiming" in r]}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["mode", "npuExecuted", "caseCount", "passedCases", "failed", "infrastructureErrors"]}, ensure_ascii=False))
    if not args.adapter:
        print("CPU数学/地址模型通过不等于NPU内核通过；Mock没有设备时延。")
    return 1 if report["failed"] or report["infrastructureErrors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
