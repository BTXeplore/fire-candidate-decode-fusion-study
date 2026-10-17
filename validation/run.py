"""Deterministic Coverage / Correctness / Mock runner; default CPU only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from cases import cases, make_inputs
from oracle import reference, scalar_reference, stable_reference_f64, compare_output
from layout_model import check_layout
from curve_analysis import analyze


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
    parser.add_argument("--suite", choices=["all", "coverage", "correctness", "mock"], default="all")
    parser.add_argument("--tier", choices=["quick", "full"], default="quick")
    parser.add_argument("--tile", type=int, default=704)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--adapter", type=Path, help="explicit opt-in: imports and executes this NPU adapter")
    parser.add_argument("--official", type=Path, default=default_official_path())
    parser.add_argument("--mock-history", type=Path)
    parser.add_argument("--revision", help="measured NPU candidate name; enables a Mock record for history")
    parser.add_argument("--report", type=Path, help="only writes a report if explicitly requested")
    args = parser.parse_args(argv)
    adapter = load_adapter(args.adapter) if args.adapter else None
    groups = ["coverage", "correctness", "mock"] if args.suite == "all" else [args.suite]
    report = {"schemaVersion": 1, "mode": "NPU_ADAPTER" if adapter else "CPU_REFERENCE_ONLY",
              "npuExecuted": False, "officialShapeMetadataKnown": False,
              "numpy": np.__version__, "python": sys.version.split()[0],
              "cases": [], "failed": []}
    for group in groups:
        for case in cases(group, args.tier):
            raw, priors = make_inputs(case)
            expected = reference(raw, priors)
            r = {"case": case.metadata()}
            try:
                # Identity comparison tests framework contracts only; it must
                # never appear as evidence that an NPU candidate is correct.
                r["oracleContract"] = compare_output(expected, raw, priors)
                if not r["oracleContract"]["passed"]:
                    raise AssertionError("oracle contract failed")
                if group == "coverage":
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
                    raw_before, priors_before = raw.copy(), priors.copy()
                    result = adapter(raw, priors, case.metadata())
                    report["npuExecuted"] = True
                    if not np.array_equal(raw, raw_before) or not np.array_equal(priors, priors_before):
                        raise AssertionError("adapter modified an input")
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
        complete = (not report["failed"] and all("npuTiming" in r for r in mock_results))
        report["mockRecord"] = {"record": args.revision, "valid": complete,
                                "status": "Pass" if complete else "Incomplete",
                                "measurementKind": "NPU_KERNEL",
                                "caseResults": [{"case": r["case"]["name"],
                                                 "timeUs": r["npuTiming"]["medianUs"]}
                                                for r in mock_results if "npuTiming" in r]}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["mode", "npuExecuted", "caseCount", "passedCases", "failed"]}, ensure_ascii=False))
    if not adapter:
        print("CPU数学/地址模型通过不等于NPU内核通过；Mock没有设备时延。")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
