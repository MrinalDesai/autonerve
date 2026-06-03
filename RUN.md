# AutoNerve — how to run

Windows / PowerShell (Python 3.11). Adjust paths as needed.

## 1. Install core deps (no model needed)
```powershell
cd autonerve\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Try it — one command
```powershell
python demo.py
```
Runs the full thread (article -> extract -> propagate -> MRP exposure -> MILP optimizer)
over several articles. With no model present it uses the deterministic keyword
fallback, so it works immediately.

Expected highlight: the neodymium article hits EVSED-P + EVSUV, ~Rs 15 Cr exposure,
and the optimizer returns POSCO 44% / Sundram 37% / China 19% -> China dependency
100% -> 18.6% at +2% cost.

## 3. Run the API
```powershell
uvicorn main:app --reload
```
Then (new terminal or browser at http://127.0.0.1:8000/docs):
```
GET  /scenario               full knowledge graph (products, parts, suppliers)
GET  /articles               the simulated news corpus
POST /article/ART-001        full live thread from one article
POST /event  {"scenario":"neodymium"}    fire a curated scenario directly
GET  /demand/NM-005          history + forecast series for a part
```

## 4. (Optional) Enable the real Qwen LLM
1. Put the weights here:  backend\models\Qwen2.5-1.5B-Instruct\
   (reuse the folder from your UPS app, or `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`)
2. Uncomment transformers/torch in requirements.txt and `pip install -r requirements.txt`
3. Re-run. `llm.available()` flips to True and extraction uses Qwen; the keyword
   extractor stays as the safety net.

Or point at an existing copy without moving it:
```powershell
$env:AUTONERVE_MODEL = "C:\path\to\Qwen2.5-1.5B-Instruct"
```

## What's where
- `engine.py`   deterministic core: propagation, MRP, MILP optimizer (the tools)
- `llm.py`      Qwen2.5 in-process via transformers (no Ollama)
- `extraction.py` news -> event (LLM + keyword fallback)
- `main.py`     FastAPI
- `demo.py`     one-command demo
- `bom.csv` / `demand_series.csv` / `commodity_prices.csv` / `articles.json`  the data
- `vision.py`   defect detection (YOLOv8/VLM, seeded) — optional
- frontend/     Control Room (next build step; current .tsx is a stub)
