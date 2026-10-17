"""Read-only Judge comparison. No submission, file mutation, or inferred hidden shape.

Usage: python compare_judge.py BASELINE_JSON CANDIDATE_JSON [--pairing auto|id|ordinal]
Supports historical iteration-summary.json (tc/id/t) and UI ui-result.json (cases/ordinal/timeUs).
Prints JSON to stdout; callers choose whether and where to persist it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def load_result(filename):
    source = Path(filename).resolve()
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected an object")
    if isinstance(data.get("tc"), list):
        kind, raw_cases = "historical_api_summary", data["tc"]
    elif isinstance(data.get("cases"), list):
        kind, raw_cases = "ui_ordinal_observation", data["cases"]
    else:
        raise ValueError(f"{source}: unsupported result schema")
    if not raw_cases:
        raise ValueError(f"{source}: empty result cases")
    cases = []
    seen_ordinals, seen_ids = set(), set()
    for index, row in enumerate(raw_cases, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: non-object case")
        ordinal = row.get("ordinal", index)
        if type(ordinal) is not int or ordinal <= 0 or ordinal in seen_ordinals:
            raise ValueError(f"{source}: invalid or duplicate ordinal")
        seen_ordinals.add(ordinal)
        testcase_id = row.get("id") if kind == "historical_api_summary" else row.get("testcaseId")
        if testcase_id is not None:
            if not isinstance(testcase_id, str) or not testcase_id or testcase_id in seen_ids:
                raise ValueError(f"{source}: invalid or duplicate testcase ID")
            seen_ids.add(testcase_id)
        cases.append({"ordinal": ordinal, "testcaseId": testcase_id,
                      "timeUs": number(row.get("t") if kind == "historical_api_summary" else row.get("timeUs"),
                                       f"{source}: case {ordinal} time"),
                      "status": row.get("st") if kind == "historical_api_summary" else row.get("statusDisplayed"),
                      "precisionRatio": row.get("pr") if kind == "historical_api_summary" else None,
                      "outputErrorPercentDisplayed": row.get("outputErrorPercentDisplayed")})
    return {"path": str(source), "format": kind,
            "experiment": data.get("experiment"), "judgeId": data.get("judgeId", data.get("ID")),
            "objectId": data.get("objectId"),
            "apiValid": data.get("valid") if kind == "historical_api_summary" else data.get("apiValid"),
            "status": data.get("status") if kind == "historical_api_summary" else data.get("statusDisplayed"),
            "cases": cases}


def compare_results(baseline, candidate, requested="auto"):
    old, new = baseline["cases"], candidate["cases"]
    all_have_ids = all(row["testcaseId"] is not None for row in old + new)
    pairing = "id" if requested == "auto" and all_have_ids else ("ordinal" if requested == "auto" else requested)
    warnings = []
    if pairing == "id":
        if not all_have_ids:
            raise ValueError("ID pairing requested but one or more testcase IDs are missing")
        old_by_key = {row["testcaseId"]: row for row in old}
        new_by_key = {row["testcaseId"]: row for row in new}
    else:
        old_by_key = {row["ordinal"]: row for row in old}
        new_by_key = {row["ordinal"]: row for row in new}
        warnings.append("Paired only by displayed ordinal; matching underlying testcase identity is unverified. "
                        "No candidate testcase_id, dtype or hidden shape is inferred from the baseline.")
    if old_by_key.keys() != new_by_key.keys():
        raise ValueError(f"{pairing} case sets differ; refusing partial or fabricated alignment")
    rows = []
    for key, old_row in old_by_key.items():
        new_row = new_by_key[key]
        a, b = old_row["timeUs"], new_row["timeUs"]
        delta = b - a
        rows.append({"ordinal": old_row["ordinal"],
                     "testcaseId": key if pairing == "id" else None,
                     "baselineTimeUs": a, "candidateTimeUs": b,
                     "deltaUs": round(delta, 9), "deltaPercent": (b / a - 1) * 100,
                     "direction": "faster" if delta < -1e-9 else "slower" if delta > 1e-9 else "tie",
                     "baselineStatus": old_row["status"], "candidateStatus": new_row["status"]})
    a_mean = statistics.fmean(row["baselineTimeUs"] for row in rows)
    b_mean = statistics.fmean(row["candidateTimeUs"] for row in rows)
    warnings.extend(["Time comparisons alone do not establish correctness, archive integrity or an adoption decision.",
                     "Single historical comparisons are not repeated A/B evidence or proof of causality."])
    if any(x["format"] == "ui_ordinal_observation" for x in (baseline, candidate)):
        warnings.append("UI times are rounded displayed values; UI Pass/0.00% error is not API valid=true or precision_ratio=1.")
    summary = {"caseCount": len(rows), "baselineMeanUs": a_mean, "candidateMeanUs": b_mean,
               "deltaMeanUs": b_mean - a_mean, "deltaPercent": (b_mean / a_mean - 1) * 100,
               "fasterCount": sum(row["direction"] == "faster" for row in rows),
               "slowerCount": sum(row["direction"] == "slower" for row in rows),
               "tieCount": sum(row["direction"] == "tie" for row in rows)}
    historical_groups = []
    if len(rows) == 12 and {row["ordinal"] for row in rows} == set(range(1, 13)):
        for label, ordinals in [("historical ordinal small", [1, 2, 7, 8]),
                                ("historical ordinal medium", [3, 4, 9, 10]),
                                ("historical ordinal large", [5, 6, 11, 12])]:
            selected = [row for row in rows if row["ordinal"] in ordinals]
            a = statistics.fmean(row["baselineTimeUs"] for row in selected)
            b = statistics.fmean(row["candidateTimeUs"] for row in selected)
            historical_groups.append({"label": label, "ordinals": ordinals,
                                      "knownShapeOrDtype": False,
                                      "baselineMeanUs": a, "candidateMeanUs": b,
                                      "deltaPercent": (b / a - 1) * 100})
        warnings.append("Small/medium/large are inherited ordinal labels, not verified N/B/dtype categories.")
    compact = lambda result: {k: v for k, v in result.items() if k != "cases"}
    return {"schemaVersion": 1, "baseline": compact(baseline), "candidate": compact(candidate),
            "pairing": pairing, "caseIdentityVerified": pairing == "id", "summary": summary,
            "rows": rows, "historicalOrdinalGroups": historical_groups, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--pairing", choices=["auto", "id", "ordinal"], default="auto")
    args = parser.parse_args(argv)
    try:
        result = compare_results(load_result(args.baseline), load_result(args.candidate), args.pairing)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
