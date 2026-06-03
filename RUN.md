# Running AutoNerve — detailed notes

See README.md for the quick start. Extra notes below.

## Two run modes
1. **Core only** (`pip install -r requirements.txt` without transformers/torch, or
   simply no model in `backend/models/`): the full thread runs via deterministic
   fallbacks. Good enough to verify correctness and explore the dashboard.
2. **With the local LLM**: place Qwen2.5-1.5B-Instruct in `backend/models/` (see
   README). Extraction then runs through the model and reports `via llm+validated`.

## Verify quickly
```bash
cd backend
python demo.py          # prints the full decision thread, no server needed
uvicorn main:app --reload   # then open http://localhost:8000  and  /docs
```

## Reuse an existing model copy (skip re-download)
```powershell
$env:AUTONERVE_MODEL = "C:\path\to\Qwen2.5-1.5B-Instruct"
```

## Notes
- All data under `backend/*.csv` and `*.json` is synthetic demo data.
- CPU is fine for the 1.5B model (a few calls per event). GPU is faster but optional.
- The dashboard is one static file served by FastAPI — no Node/npm build step.
