# AutoNerve — Setup Guide

> **Reviewers / judges, fastest path:** the full decision thread runs in ~30 seconds with
> **no model, no GPU, no API key, no internet** — deterministic fallbacks cover every AI path.
> ```
> cd backend
> pip install fastapi uvicorn pulp pydantic
> python demo.py
> ```
> That prints the end-to-end pipeline (news → extract → propagate → exposure → optimize → recommend).
> For the full dashboard, see "Run the app" below.

---

## 1. Prerequisites

- **Python 3.11** (3.10+ works). Check: `python --version`
- **pip** (bundled with Python)
- A modern browser (Chrome/Edge/Firefox)
- *(Optional)* internet — only for the 3D globe tab's CDN; everything else is offline.

No database, no Node, no build step. The dashboard is a single static HTML file served by the API.

---

## 2. Get the code

```bash
git clone https://github.com/<your-username>/autonerve.git
cd autonerve
```
Or download the ZIP and extract it.

---

## 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> If activation is blocked, run once:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**macOS / Linux:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

The prompt should now show `(.venv)`. **All commands below run from the `backend/` folder.**

---

## 4. Install dependencies

**Minimal (runs the full system via fallbacks — recommended for a quick review):**
```bash
pip install fastapi uvicorn pulp pydantic
```

**Full (everything in `requirements.txt`, including the optional LLM + vision libs):**
```bash
pip install -r requirements.txt
```

| Feature | Extra install | Without it |
|---------|---------------|-----------|
| Core thread, optimizer, dashboard | nothing beyond minimal | — |
| One-shot vision QC | `pip install pillow numpy` | panel shows a notice |
| OCR news intake | `pip install pytesseract pillow` **+ Tesseract binary** (below) | falls back to cached text (labelled) |
| Live local LLM | `pip install transformers torch accelerate safetensors` + model (below) | deterministic fallback extraction |

---

## 5. Run the app

```bash
uvicorn main:app --reload
```
Open **http://localhost:8000** in your browser.

You should see the AutoNerve command center with its tabs (Control Room, Plant Floor,
Agentic Run, Procurement, Analytics, etc.). Interactive API docs are at
**http://localhost:8000/docs**.

> **After changing any backend file**, restart uvicorn (data/code load at startup).
> Frontend-only changes just need a hard browser refresh (Ctrl/Cmd + F5).

---

## 6. Optional extras

### 6a. Enable OCR (news-clipping intake)
The Agentic Run reads `backend/sample_news.png` via Tesseract.
- **Windows:** `winget install tesseract` (or the UB-Mannheim Tesseract installer), then `pip install pytesseract pillow`
- **macOS:** `brew install tesseract` then `pip install pytesseract pillow`
- **Linux:** `sudo apt install tesseract-ocr` then `pip install pytesseract pillow`

Without it, the Signal agent labels the step "cached text" and still runs.

### 6b. Enable the live local LLM
```bash
pip install transformers torch accelerate safetensors
```
Then place **Qwen2.5-1.5B-Instruct** into `backend/models/Qwen2.5-1.5B-Instruct/`:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-1.5B-Instruct'; AutoTokenizer.from_pretrained(m).save_pretrained('models/Qwen2.5-1.5B-Instruct'); AutoModelForCausalLM.from_pretrained(m).save_pretrained('models/Qwen2.5-1.5B-Instruct')"
```
`python demo.py` then reports `LLM model present: True`, and extraction runs `via llm+validated`.
(Or point `AUTONERVE_MODEL` at an existing copy of the weights to reuse them.)

> CPU is fine. The LLM is a few seconds per call; for the Agentic Run, use the **Replay** button for an instant cached run.

---

## 7. Quick tour (what to click)

1. **Control Room** → click a news item in the feed → watch the alert + AI sourcing decision appear.
2. **Agentic Run** → press **Run agents** (or **Replay** for instant) → the full story: OCR → risk → rejection → defect loop → bottleneck → decision → impact.
3. **Plant Floor** → open the **One-shot visual fault detection** expander → pick a sample → see the defect boxed with a PASS/FAIL verdict.
4. **Procurement** → price forecast, scarcity, and the pre-stock buy plan.
5. **Analytics** → Cpk forecast, supplier scorecard, yield waterfall, PO slip.

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Activate.ps1 is not recognized` | You're in the wrong folder — the venv is inside `backend/`. `cd backend` first. |
| `uvicorn: command not found` | The venv isn't activated, or deps aren't installed. Re-activate and `pip install`. |
| Globe tab is blank | It needs internet for its CDN. The rest works offline. |
| OCR shows "cached text" | Tesseract binary isn't installed (see 6a). The demo still runs. |
| LLM says "model present: False" | Expected without the weights — fallbacks run. Add the model (6b) to enable it. |
| Agentic Run is slow | Use the **Replay** button (cached, instant). Live runs make real LLM calls. |

---

## 9. What's real vs modeled (honest note)

- **Real, running:** OCR (Tesseract), local LLM extraction (Qwen), MILP sourcing optimizer (PuLP),
  knowledge-graph propagation, demand forecast, one-shot reference-vs-test vision diff, the
  defect → re-sourcing closed loop.
- **Modeled / illustrative:** all demo data is synthetic; savings and risk-score figures are
  modeled on stated assumptions; some breadth panels (energy, ERP push) are seeded displays.
- **Roadmap (named, not implemented):** GNN, temporal transformer, YOLOv8/PatchCore, Whisper,
  Altman-Z financial-health ML.
