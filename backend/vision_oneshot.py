"""AutoNerve — one-shot visual fault detection.

Compares a test part image against a single 'golden' reference (one-shot, no
training) using pixel-level difference. Returns a defect score, pass/fail against a
threshold, and the bounding box of the largest changed region so the UI can draw it.

Classical CV (PIL + numpy). This is genuine reference-vs-test anomaly detection; a
trained PatchCore/YOLOv8 model is the production target for un-referenced defects.
"""
import json as _json
from pathlib import Path

HERE = Path(__file__).parent
PARTS = HERE / "parts"

SAMPLES = [
    {"id": "good",           "name": "Unit 1182 — nominal"},
    {"id": "defect_burr",    "name": "Unit 1183 — coating/burr"},
    {"id": "defect_missing", "name": "Unit 1184 — missing block"},
]
THRESHOLD = 2.0   # % of pixels changed beyond which the unit FAILS


def available():
    try:
        from PIL import Image  # noqa
        import numpy  # noqa
        return (PARTS / "ref.png").exists()
    except Exception:
        return False


def inspect(sample_id="defect_burr"):
    """Diff a test image vs the golden reference; locate + score the defect."""
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return {"error": "imaging libs unavailable"}

    ref_p, test_p = PARTS / "ref.png", PARTS / f"{sample_id}.png"
    if not ref_p.exists() or not test_p.exists():
        return {"error": "image not found"}

    ref = np.asarray(Image.open(ref_p).convert("L"), dtype=np.int16)
    test = np.asarray(Image.open(test_p).convert("L"), dtype=np.int16)
    diff = np.abs(ref - test)
    mask = diff > 38                                  # changed pixels
    changed_pct = round(100 * mask.mean(), 2)

    box = None
    if mask.any():
        ys, xs = np.where(mask)
        H, W = mask.shape
        # bounding box as % of image (so the UI scales it to any rendered size)
        box = {"x": round(100 * float(xs.min()) / W, 1), "y": round(100 * float(ys.min()) / H, 1),
               "w": round(100 * float(xs.max() - xs.min()) / W, 1),
               "h": round(100 * float(ys.max() - ys.min()) / H, 1)}

    verdict = "FAIL" if changed_pct >= THRESHOLD else "PASS"
    return {"sample": sample_id, "verdict": verdict, "score": float(changed_pct),
            "threshold": THRESHOLD, "box": box,
            "name": next((s["name"] for s in SAMPLES if s["id"] == sample_id), sample_id)}


def samples():
    return SAMPLES
