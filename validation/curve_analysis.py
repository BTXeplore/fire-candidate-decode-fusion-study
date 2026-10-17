"""Compare revision trends without assigning hidden shapes to Judge point IDs."""
import math
import numpy as np


def revision_curve(document, allow_ui=False):
    records = document["records"]
    base = records[0]
    base_points = {str(p["case"]): float(p["timeUs"]) for p in base["caseResults"]}
    if not base_points or not all(math.isfinite(t) and t > 0 for t in base_points.values()):
        raise ValueError("timings must be finite and positive")
    curve = []
    for r in records:
        ps = r["caseResults"]
        is_ui = r.get("evidenceType") == "JUDGE_UI"
        if is_ui:
            ordinals = [p.get("case") for p in ps]
            if (not allow_ui or "valid" not in r or r["valid"] is not None
                    or r.get("comparisonEligible") is not True or r.get("status") != "Pass"
                    or len(ps) != 12 or any(type(k) is not int for k in ordinals)
                    or set(ordinals) != set(range(1, 13))
                    or any(p.get("status") != "Pass" for p in ps)):
                raise ValueError("UI description requires valid=null, explicit eligibility and all 12 unique Pass ordinals; it is not API valid")
        elif r.get("valid") is not True or r.get("status") != "Pass":
            raise ValueError("only complete valid Pass records are comparable")
        points = {str(p["case"]): float(p["timeUs"]) for p in ps}
        if len(points) != len(ps) or set(points) != set(base_points):
            raise ValueError("every revision needs identical unique case identities")
        if not all(math.isfinite(t) and t > 0 for t in points.values()):
            raise ValueError("timings must be finite and positive")
        logs = np.log([points[k] / base_points[k] for k in base_points])
        curve.append({"record": r["record"], "pointCount": len(points),
                      "evidenceType": "JUDGE_UI" if is_ui else r.get("evidenceType", "VALID_RECORD"),
                      "apiValid": None if is_ui else r["valid"],
                      "comparisonBasis": "UI displayed ordinals only; actual testcase identity and API valid are unknown" if is_ui else "valid Pass record with stable case identities",
                      "meanUs": float(np.mean(list(points.values()))),
                      "geomeanRatioToFirst": float(np.exp(logs.mean())),
                      "worstPointRatioToFirst": float(np.exp(logs.max())),
                      "logRatioToFirst": float(logs.mean())})
    return curve


def analyze(official, mock_history=None):
    real = revision_curve(official, allow_ui=True)
    out = {"officialRevisionCurve": real,
           "shapeMetadataKnown": False,
           "note": "No hidden shape/dtype inference. JUDGE_UI rows are descriptive ordinal comparisons, not verified testcase identities or API valid=true.",
           "mockAlignment": {"status": "not_measured", "npuTimingInvented": False}}
    if mock_history is None:
        return out
    if mock_history.get("measurementKind") != "NPU_KERNEL":
        raise ValueError("Mock history requires actual NPU_KERNEL measurements, not CPU timings")
    local = revision_curve(mock_history)
    by_name = {r["record"]: r for r in local}
    shared = [r for r in real if r["record"] in by_name]
    if len(shared) < 3 or real[0]["record"] != local[0]["record"]:
        raise ValueError("need >=3 shared measured revisions with identical first baseline")
    x = np.array([r["logRatioToFirst"] for r in shared])
    y = np.array([by_name[r["record"]]["logRatioToFirst"] for r in shared])
    dx, dy = np.diff(x), np.diff(y)
    corr = None if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    out["mockAlignment"] = {
        "status": "descriptive_only", "sharedRevisions": [r["record"] for r in shared],
        "meanAbsoluteLogTrendGap": float(np.mean(np.abs(x - y))),
        "successiveDirectionAgreement": float(np.mean(np.sign(dx) == np.sign(dy))),
        "correlation": corr,
        "limitation": "Small revision count, hardware noise and different case distributions; no proof of hidden-case matching."}
    return out
