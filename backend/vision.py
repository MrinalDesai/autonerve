"""
Vision for AutoNerve. Two distinct jobs -- do not conflate them.

(1) DEFECT DETECTION  -> YOLOv8-n / PatchCore. Object detection / anomaly, NOT a
    VLM. Fast, runs on the Jetson, exactly the deck's pitch. For the video, run a
    pretrained YOLOv8-n on a couple of sample part images; a 50-shot fine-tune is
    ~1 hour if you want real per-station numbers.

(2) VISION-LANGUAGE REASONING (optional) -> local VLM via Ollama, if you want a
    model to look at an image and explain it (read a defect, analyse a chart).
        qwen2.5vl:7b   best chart/document understanding (8-16GB VRAM)
        moondream      tiny ~1.8B, CPU-capable, JSON out -- best edge story
        llava:7b       general, runs on 8GB VRAM

Both functions fall back to a cached result so the demo never breaks.

Deps (install only what you use):
    pip install ultralytics      # YOLOv8
    pip install ollama           # VLM
"""

from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent

# Seeded defect result mirrors the Vision QC screen (scenario / S30).
CACHED_DETECTION = {
    "image": "sample",
    "detections": [
        {"label": "burr", "confidence": 0.91, "bbox": [120, 88, 210, 176]},
    ],
    "verdict": "DEFECT",
    "live": False,
}

VLM_MODEL = "moondream"  # swap to qwen2.5vl:7b for higher quality


def detect_defects(image_path: str, model_path: str = "yolov8n.pt") -> dict:
    """YOLOv8-n inference on one image. Cached fallback if ultralytics/model absent."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)  # your fine-tuned weights, or yolov8n.pt to start
        res = model(image_path)[0]
        dets = [
            {
                "label": res.names[int(b.cls)],
                "confidence": round(float(b.conf), 3),
                "bbox": [round(float(x)) for x in b.xyxy[0].tolist()],
            }
            for b in res.boxes
        ]
        return {
            "image": Path(image_path).name,
            "detections": dets,
            "verdict": "DEFECT" if dets else "OK",
            "live": True,
        }
    except Exception as e:  # noqa: BLE001
        return {**CACHED_DETECTION, "error": str(e)}


def describe_image(image_path: str, question: str = "Describe any visible defect.") -> dict:
    """Local VLM reasoning over an image via Ollama. Cached fallback on failure."""
    try:
        import ollama
        resp = ollama.chat(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": question, "images": [image_path]}],
            options={"temperature": 0},
        )
        return {"text": resp["message"]["content"], "model": VLM_MODEL, "live": True}
    except Exception as e:  # noqa: BLE001
        return {
            "text": "Surface burr along the lower-left edge of the bracket; "
                    "recommend rework before downstream assembly.",
            "model": VLM_MODEL, "live": False, "error": str(e),
        }


if __name__ == "__main__":
    print(json.dumps(detect_defects("sample.jpg"), indent=2))
    print(json.dumps(describe_image("sample.jpg"), indent=2))
