"""设备接入负控在 CPU 上执行；不生成伪造的 NPU 测量或通过记录。"""
import unittest
import json

import numpy as np

from cases import make_inputs
from device_contract import (smoke_cases, device_boundary_cases, host_input_snapshot,
                             assert_host_inputs_unchanged, validate_memory_evidence)
from oracle import reference, compare_output


class DeviceContractTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.priors = make_inputs(smoke_cases()[0])

    def evidence(self):
        # 仅为验证检查器的 CPU 合成证据，不写入设备报告。
        prefix = np.full(64, 0xA5, dtype=np.uint8)
        suffix = np.full(64, 0x5A, dtype=np.uint8)
        return {"synchronized": True,
                "device_inputs": {"raw_pred": self.raw.copy(), "priors": self.priors.copy()},
                "output_guards": {"expected_prefix": prefix, "actual_prefix": prefix.copy(),
                                  "expected_suffix": suffix, "actual_suffix": suffix.copy()},
                "output_initialized_to_nan": True}

    def test_smoke_order_is_finite_and_keeps_both_dtypes(self):
        probes = smoke_cases()
        self.assertEqual(len(probes), 6)
        self.assertEqual([(c.batch, c.n) for c in probes], [(1, 1)] * 2 + [(3, 33)] * 2 + [(8, 705)] * 2)
        for c in probes:
            raw, priors = make_inputs(c)
            self.assertTrue(compare_output(reference(raw, priors), raw, priors)["passed"])

    def test_missing_evidence_is_unknown(self):
        for evidence in (None, {}, {"synchronized": True}):
            result = validate_memory_evidence(evidence, self.raw, self.priors)
            self.assertEqual(result["status"], "unknown")
            self.assertTrue(result["missing"])

    def test_per_core_boundaries_use_observed_cores_and_public_n_limit(self):
        # 40仅为检查生成器数学的输入，不是声称实机或Judge一定有40核。
        probes = device_boundary_cases(704, 40)
        ns = {c.n for c in probes}
        self.assertTrue({28159, 28160, 28161}.issubset(ns))
        self.assertEqual(len(probes), len(ns) * 3 * 2)
        for cores in (1, 8, 40, 48, 128):
            probes = device_boundary_cases(704, cores)
            self.assertTrue(all(1 <= c.n <= 32768 for c in probes))
            self.assertEqual(len({c.name for c in probes}), len(probes))
        for tile, cores in ((0, 8), (704, 0), (-1, 8)):
            with self.assertRaises(ValueError):
                device_boundary_cases(tile, cores)

    def test_complete_synthetic_evidence_only_checks_contract(self):
        self.assertEqual(validate_memory_evidence(self.evidence(), self.raw, self.priors)["status"], "pass")

    def test_signed_zero_input_modification_is_caught(self):
        before = host_input_snapshot(self.raw, self.priors)
        self.raw.flat[0] = -0.0
        with self.assertRaises(AssertionError):
            assert_host_inputs_unchanged(self.raw, self.priors, before)
        evidence = self.evidence()
        evidence["device_inputs"]["raw_pred"].flat[0] = 0.0
        self.assertEqual(validate_memory_evidence(evidence, self.raw, self.priors)["status"], "fail")

    def test_corrupted_or_short_guard_is_rejected(self):
        for side in ("prefix", "suffix"):
            evidence = self.evidence()
            evidence["output_guards"][f"actual_{side}"][0] ^= 1
            self.assertEqual(validate_memory_evidence(evidence, self.raw, self.priors)["status"], "fail")
        evidence = self.evidence()
        evidence["output_guards"]["actual_prefix"] = np.zeros(8, dtype=np.uint8)
        self.assertEqual(validate_memory_evidence(evidence, self.raw, self.priors)["status"], "fail")

    def test_no_synchronization_or_initialization_does_not_pass(self):
        for missing in ("synchronized", "output_initialized_to_nan"):
            evidence = self.evidence()
            del evidence[missing]
            self.assertEqual(validate_memory_evidence(evidence, self.raw, self.priors)["status"], "unknown")

    def test_poisoned_unwritten_output_is_rejected_by_existing_oracle(self):
        output = reference(self.raw, self.priors)
        output.flat[-1] = np.nan
        self.assertFalse(compare_output(output, self.raw, self.priors)["passed"])

    def test_nonfinite_failure_diagnostics_remain_strict_json_serializable(self):
        for value in (np.nan, np.inf, -np.inf):
            output = reference(self.raw, self.priors)
            output.flat[-1] = value
            result = compare_output(output, self.raw, self.priors)
            self.assertFalse(result["passed"])
            self.assertEqual(result["nonfiniteActualAtFiniteReference"], 1)
            json.dumps(result, allow_nan=False)
        # 所有输出都坏时没有可用的有限差值；用null而非0或NaN说明缺少统计。
        output = np.full_like(reference(self.raw, self.priors), np.nan)
        result = compare_output(output, self.raw, self.priors)
        self.assertIsNone(result["maxAbsFinite"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
