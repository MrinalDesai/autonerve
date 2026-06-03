"""AutoNerve — OCR intake.

Reads an unstructured news-clipping image and extracts its text via Tesseract
(pytesseract). The extracted text then feeds the LLM event-extraction step, so the
pipeline genuinely ingests a document image rather than pre-loaded text.

Graceful fallback: if Tesseract / pytesseract is unavailable, returns the known
article text and reports source='fallback' so the demo never breaks.
"""
import json as _json
from pathlib import Path

HERE = Path(__file__).parent
_IMG = HERE / "sample_news.png"

# Cached text for the bundled clipping — used as fallback and to keep the
# downstream scenario deterministic for the demo/video.
_CACHED = ("China Widens Rare-Earth Export Licensing; Neodymium Magnet Supply Tightens. "
           "New controls on high-grade NdFeB magnets threaten EV traction-motor production "
           "worldwide. Beijing - China's Ministry of Commerce has expanded export-licensing "
           "requirements on high-grade neodymium-iron-boron (NdFeB) magnets, the rare-earth "
           "components at the heart of electric-vehicle traction motors. Industry analysts warn "
           "the move could tighten global supply within weeks and push prices sharply higher. "
           "Automakers that source magnets predominantly from Chinese suppliers face the "
           "greatest exposure, with EV sedan and SUV programmes most at risk.")


def available():
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
        return _IMG.exists()
    except Exception:
        return False


def ocr_image(path=None):
    """Return {text, source, image, chars}. source='tesseract' if the image was
    really read, else 'fallback'."""
    img_path = Path(path) if path else _IMG
    try:
        import pytesseract
        from PIL import Image
        if img_path.exists():
            raw = pytesseract.image_to_string(Image.open(img_path))
            text = " ".join(raw.split())          # collapse whitespace
            if len(text) > 40:                     # sanity: got real text
                return {"text": text, "source": "tesseract",
                        "image": img_path.name, "chars": len(text)}
    except Exception:
        pass
    return {"text": _CACHED, "source": "fallback",
            "image": img_path.name if img_path.exists() else None, "chars": len(_CACHED)}
