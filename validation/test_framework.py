"""Negative controls ensure framework catches layout, tolerance and provenance errors."""
import unittest
import copy
import numpy as np
from cases import Case, make_inputs, cases
from oracle import reference, compare_output, stable_reference_f64
from layout_model import check_layout, partitions
from curve_analysis import analyze, revision_curve


def actual_ui_exp018():
    # Actual displayed times: Judge 877556, 2026-10-17, ui-result.json.
    # Only public comparison facts are retained; API validity stays UNKNOWN.
    times = [4.71, 5.14, 7.84, 7.85, 27.24, 27.21,
             4.86, 5.52, 7.95, 8.02, 27.65, 27.70]
    return {"record": "EXP-018", "valid": None, "comparisonEligible": True,
            "evidenceType": "JUDGE_UI", "status": "Pass",
            "caseResults": [{"case": i, "status": "Pass", "timeUs": t}
                            for i, t in enumerate(times, 1)]}


class FrameworkTests(unittest.TestCase):
    def test_official_zero(self):
        raw = np.zeros((1, 1, 6), np.float32)
        p = np.array([[0, 0, 32, 2, 2]], np.float32)
        np.testing.assert_array_equal(reference(raw, p), [[[16, 16, 64, 64, .25]]])

    def test_reject_wrong_field_batch_dtype(self):
        c = Case("tags", "correctness", 3, 65, "float32", 5, "field_tags")
        raw, p = make_inputs(c)
        out = reference(raw, p)
        for bad in [out[..., [1, 0, 2, 3, 4]].copy(), out[::-1].copy(), out.astype(np.float16)]:
            self.assertFalse(compare_output(bad, raw, p)["passed"])

    def test_fp16_cast_reference_and_tolerance(self):
        raw, p = make_inputs(Case("round", "correctness", 1, 33, "float16", 9, "half_rounding"))
        out = reference(raw, p)
        self.assertTrue(compare_output(out, raw, p)["passed"])
        bad = out.copy()
        bad[..., 4] = 0.9
        self.assertFalse(compare_output(bad, raw, p)["passed"])

    def test_nan_rejected(self):
        raw, p = make_inputs(Case("nan", "correctness", 1, 1, "float32", 9))
        out = reference(raw, p)
        out[0, 0, 0] = np.nan
        self.assertFalse(compare_output(out, raw, p)["passed"])

    def test_partition_and_guards(self):
        for n in [1, 7, 8, 9, 33, 703, 704, 705, 5631, 5632, 5633, 32768]:
            for tile in [64, 704, 768, 1024]:
                self.assertTrue(check_layout(3, n, tile, 8)["passed"])
            self.assertEqual(sum(c for _, _, c in partitions(n, 8)), n)

    def test_determinism(self):
        c = cases("mock")[0]
        a, b = make_inputs(c)
        aa, bb = make_inputs(c)
        np.testing.assert_array_equal(a, aa)
        np.testing.assert_array_equal(b, bb)
        self.assertTrue(c.hypothesis)

    def test_extreme_logit_stable(self):
        for dtype in ["float16", "float32"]:
            r, p = make_inputs(Case("extreme", "correctness", 3, 65, dtype, 3, "saturation"))
            self.assertTrue(np.isfinite(reference(r, p)).all())
            self.assertTrue(np.isfinite(stable_reference_f64(r, p)).all())

    def test_curve_provenance_and_identity(self):
        def rec(name, ratio):
            return {"record": name, "valid": True, "status": "Pass",
                    "caseResults": [{"case": i, "timeUs": ratio * (i + 1)} for i in range(2)]}
        official = {"records": [rec("v1", 1), rec("v2", .9), rec("v3", 1.1)]}
        out = analyze(official)
        self.assertEqual(out["mockAlignment"]["status"], "not_measured")
        mock = {"records": official["records"], "measurementKind": "CPU"}
        with self.assertRaises(ValueError):
            analyze(official, mock)
        mock["measurementKind"] = "NPU_KERNEL"
        self.assertEqual(analyze(official, mock)["mockAlignment"]["successiveDirectionAgreement"], 1)
        official["records"][1]["caseResults"].pop()
        with self.assertRaises(ValueError):
            revision_curve(official)

    def test_actual_ui_schema_descriptive_compatibility(self):
        out = analyze({"records": [actual_ui_exp018()]})
        row = out["officialRevisionCurve"][0]
        self.assertEqual(row["pointCount"], 12)
        self.assertEqual(row["evidenceType"], "JUDGE_UI")
        self.assertIsNone(row["apiValid"])
        self.assertAlmostEqual(row["meanUs"], 13.474166666666667)
        self.assertIn("unknown", row["comparisonBasis"])

    def test_incomplete_or_nonpass_ui_and_mock_rejected(self):
        valid_ui = actual_ui_exp018()
        bad_records = []
        missing = copy.deepcopy(valid_ui)
        missing["caseResults"].pop()
        bad_records.append(missing)
        failed = copy.deepcopy(valid_ui)
        failed["caseResults"][3]["status"] = "Fail"
        bad_records.append(failed)
        duplicate = copy.deepcopy(valid_ui)
        duplicate["caseResults"][3]["case"] = 1
        bad_records.append(duplicate)
        forged = copy.deepcopy(valid_ui)
        forged["valid"] = True
        bad_records.append(forged)
        ineligible = copy.deepcopy(valid_ui)
        ineligible["comparisonEligible"] = False
        bad_records.append(ineligible)
        for record in bad_records:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    analyze({"records": [record]})
        # The UI exception is allowed only for official descriptions. Actual
        # local Mock histories still require valid=true NPU records.
        with self.assertRaises(ValueError):
            analyze({"records": [valid_ui]}, {"measurementKind": "NPU_KERNEL", "records": [valid_ui]})
        with self.assertRaises(ValueError):
            revision_curve({"records": [valid_ui]})


if __name__ == "__main__":
    unittest.main()
