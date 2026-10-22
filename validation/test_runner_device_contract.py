"""CPU负控：模拟故障返回以验证报告链路；不是可用的NPU adapter或设备成绩。"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from oracle import reference
import run as runner


class RunnerDeviceContractTests(unittest.TestCase):
    def test_adapter_import_failure_is_infrastructure_error_with_zero_cases(self):
        with tempfile.TemporaryDirectory(prefix="fire-framework-negative-") as directory:
            report_path = Path(directory) / "report.json"
            with patch.object(runner, "load_adapter", side_effect=OSError("synthetic missing device library")), redirect_stdout(io.StringIO()):
                status = runner.main(["--suite", "smoke", "--adapter", "CONTRACT_TEST_ONLY",
                                      "--report", str(report_path)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 1)
            self.assertEqual((report["caseCount"], report["plannedCaseCount"]), (0, 6))
            self.assertEqual(report["failed"], [])
            self.assertTrue(report["infrastructureErrors"])
            self.assertFalse(report["adapterInvoked"])
            self.assertFalse(report["npuExecuted"])

    def invoke(self, behavior, *options, suite="smoke"):
        # 学习｜测试替换外部调用，只检查框架错误处理。临时报告不进入真实
        # measurements，不能把这些合成回包当成设备执行、时延或正确性证据。
        with tempfile.TemporaryDirectory(prefix="fire-framework-negative-") as directory:
            report_path = Path(directory) / "report.json"
            args = ["--suite", suite, "--adapter", "CONTRACT_TEST_ONLY",
                    "--official", str(Path(directory) / "missing.json"),
                    "--report", str(report_path), *options]
            with patch.object(runner, "load_adapter", return_value=behavior), redirect_stdout(io.StringIO()):
                status = runner.main(args)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            json.dumps(report, allow_nan=False)
            return status, report

    def test_first_launch_exception_stops_and_preserves_failure_report(self):
        calls = []

        def launch_failure(raw, priors, metadata):
            calls.append(metadata["name"])
            raise RuntimeError("synthetic launch failure; no device execution")

        status, report = self.invoke(launch_failure, "--fail-fast")
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual((report["caseCount"], report["plannedCaseCount"]), (1, 6))
        self.assertTrue(report["stoppedEarly"])
        self.assertTrue(report["adapterInvoked"])
        self.assertFalse(report["npuExecuted"])

    def test_http_error_mapping_is_not_device_execution(self):
        status, report = self.invoke(lambda *args: {"error": "HTTP 400"}, "--fail-fast")
        self.assertEqual(status, 1)
        self.assertFalse(report["npuExecuted"])
        self.assertIn("NumPy device output", report["cases"][0]["error"])

    def test_nan_poison_failure_is_saved_as_strict_json(self):
        def poison(raw, priors, metadata):
            output = reference(raw, priors)
            output.flat[-1] = np.nan
            return {"output": output}

        status, report = self.invoke(poison, "--fail-fast")
        self.assertEqual(status, 1)
        comparison = report["cases"][0]["candidateComparison"]
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["nonfiniteActualAtFiniteReference"], 1)
        self.assertEqual(report["executionEvidence"], "ADAPTER_REPORTED_UNVERIFIED")

    def test_missing_device_memory_evidence_cannot_pass_strict_mode(self):
        # 合成输出只用于证明严格门槛先拒绝缺证，不能认证真实设备来源。
        status, report = self.invoke(lambda raw, priors, metadata: {"output": reference(raw, priors)},
                                     "--fail-fast", "--require-device-memory-checks")
        self.assertEqual(status, 1)
        self.assertEqual(report["cases"][0]["deviceMemory"]["status"], "unknown")
        self.assertIn("incomplete", report["cases"][0]["error"])

    def test_partial_mock_failure_cannot_create_valid_mock_history(self):
        def failure(*args):
            raise RuntimeError("synthetic failure")

        status, report = self.invoke(failure, "--fail-fast", "--revision", "TEST_ONLY", suite="mock")
        self.assertEqual(status, 1)
        self.assertFalse(report["mockRecord"]["valid"])
        self.assertEqual(report["mockRecord"]["status"], "Incomplete")
        self.assertEqual(report["mockRecord"]["caseResults"], [])

    def test_strict_memory_mode_rejects_missing_adapter(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            runner.main(["--suite", "smoke", "--require-device-memory-checks"])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
