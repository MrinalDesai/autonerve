# AutoNerve

**A unified AI nervous system for automotive supply chains + smart manufacturing.**
ET AutoTech Hackathon 2026 · Theme 1.

AutoNerve ingests a disruption signal (e.g. a news event), runs a **local LLM** to
extract it, propagates it through a **knowledge-graph BOM**, quantifies the rupee
exposure, and runs a **MILP optimizer** to produce a de-risked sourcing decision —
end to end, on one machine, no cloud. It then closes the loop the other way: a
defect detected on the shop floor re-prices a supplier and re-runs the optimizer.

> **Design principle:** *the LLM decides and explains; deterministic tools compute.*
> The language model handles extraction and orchestration only. A deterministic
> engine produces every number (risk, exposure, sourcing mix), grounded against the
> knowledge graph so the model cannot hallucinate a decision.

---

## Quick start (Windows / PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000
```

macOS / Linux:
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**The full system runs with the core install above — no model, no API key, no cloud.**
When the LLM weights are absent, deterministic fallbacks run the same thread, so a
reviewer can verify correctness immediately. To enable the real LLM, see below.

### One-command sanity check (no server)
```bash
cd backend
python demo.py
```
Prints the full decision thread over several news articles
(extract → propagate → exposure → optimize → recommend).

---

## Enable the local LLM (optional)

The LLM layer uses **Qwen2.5-1.5B-Instruct** in-process via HuggingFace `transformers`
(no Ollama, no API). With it loaded, news extraction is model-driven and validated
against the BOM; without it, a deterministic keyword extractor covers the same path.

```powershell
pip install transformers torch accelerate safetensors
# download weights into backend/models/Qwen2.5-1.5B-Instruct/
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-1.5B-Instruct'; AutoTokenizer.from_pretrained(m).save_pretrained('models/Qwen2.5-1.5B-Instruct'); AutoModelForCausalLM.from_pretrained(m).save_pretrained('models/Qwen2.5-1.5B-Instruct')"
```
`demo.py` then reports `LLM model present: True` and extraction runs `via llm+validated`.
(Weights are not committed — see `.gitignore`. Point `AUTONERVE_MODEL` at an existing copy to reuse it.)

---


### Optional: OCR news intake
The Agentic Run reads a news-clipping image (`backend/sample_news.png`) via Tesseract.
Install the engine to enable it: Windows `winget install tesseract` (or the UB-Mannheim build);
macOS `brew install tesseract`; Linux `apt install tesseract-ocr`. Plus `pip install pytesseract pillow`.
Without it, the step falls back to cached text and labels itself accordingly.

---

## What it does (the live decision thread)

```
news article  →  LLM extraction (validated vs BOM)  →  graph propagation
              →  MRP exposure  →  MILP optimizer  →  recommended sourcing action
                                          ↑
   shop-floor defect  →  supplier re-priced  →  re-optimize   (the closed loop)
```

Worked example (neodymium export control): hits 2 EV models, ₹4.2 Cr exposure over
12 weeks, optimizer shifts the magnet mix from 100% → 18.6% China dependency at +2%
cost — and reports honestly when zero-China is infeasible at full volume.

---

## Dashboard (open `/`)

Eight tabs, all reading the live API:
**Control Room** (fire a signal → decision card) · **Supply Globe** · **Plant Floor**
(lines, vision QC, defect→sourcing loop, energy, operator copilot) · **BOM Explorer**
(selectable supplier paths, live cost/lead rollup) · **Planning & Actions** (forecast
scenarios, meeting→actions) · **Procurement** (price forecast, scarcity, buy plan) ·
**Analytics** (Cpk forecast, yield waterfall, supplier scorecard, PO slip) ·
**Alternate Sourcing** (cheapest vs fastest).

---

## Architecture

| Layer | File | Role |
|-------|------|------|
| Data / knowledge graph | `bom.csv`, `demand_series.csv`, `commodity_prices.csv`, `articles.json`, `sops.json`, `meetings.json` | single source of truth: typed entities + relationships |
| LLM | `llm.py` | Qwen2.5-1.5B in-process (transformers); extraction + grounding only |
| Extraction | `extraction.py` | news → event; LLM + BOM-validation + deterministic fallback |
| Deterministic engine | `engine.py` | propagation, MRP exposure, MILP optimizer, multi-objective sourcing, BOM rollup, forecast, defect loop, analytics |
| API + UI | `main.py`, `frontend/index.html` | FastAPI serves the dashboard at `/` and ~24 endpoints |

**Stack:** Python 3.11 · FastAPI · PuLP (CBC MILP solver) · transformers + PyTorch
(Qwen2.5-1.5B, local) · single-file HTML/JS dashboard (Tailwind + globe.gl, no build step).

Key endpoints: `/scenario`, `/event`, `/article/{id}`, `/bom/{product}`,
`/strategies/{part}`, `/forecast/{id}`, `/procurement`, `/plant`, `/defect`,
`/cpk/{line}`, `/scorecard/{supplier}`, `/po-forecast`. Full list at `/docs`.

---

## Capability fidelity (honest scope)

This prototype implements a **real decision core** and a breadth of supporting views.
We mark fidelity rather than overclaim:

- **Live (real logic):** news extraction (LLM), graph propagation, MRP exposure,
  MILP sourcing optimizer, cheapest/fastest multi-objective sourcing, BOM cost/lead
  rollup, demand forecast (least-squares + bands), defect → re-sourcing closed loop,
  operator SOP retrieval, meeting action extraction.
- **Prototyped / seeded (interface real, model simplified):** vision QC feed, Cpk
  display + trend forecast, energy anomaly, supplier scorecard, PO slip, quality
  monitoring.
- **Roadmap (named target methods, not implemented):** GNN risk, temporal
  transformer / NHITS demand models, YOLOv8 + PatchCore vision, Whisper audio,
  Altman-Z financial-health ML, causal discovery.

All data is **synthetic/illustrative** demo data (not live feeds or real contracts).

---

## Repo layout

```
autonerve/
├── backend/
│   ├── engine.py · llm.py · extraction.py · main.py · demo.py · vision.py
│   ├── bom.csv · demand_series.csv · commodity_prices.csv
│   ├── articles.json · sops.json · meetings.json
│   ├── models/           # Qwen weights go here (gitignored)
│   └── requirements.txt
├── frontend/
│   └── index.html        # single-file dashboard, served at /
├── README.md · RUN.md · AutoNerve_Design_Document.pdf
└── .gitignore
```

---

## License

Prototype for ET AutoTech Hackathon 2026. Demo data is synthetic.
