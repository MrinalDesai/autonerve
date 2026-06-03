# AutoNerve — prototype build

A unified AI nervous system connecting the supply chain to the shop floor.
ET AutoTech Hackathon 2026 · Theme 1. Prototype-round build.

**The thesis to demo:** one closed loop nobody else shows — a news event →
risk → alternate supplier → *predicted Cpk impact of switching* → PO push.
A sourcing decision that knows its own quality consequence.

---

## Run it

**Backend**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # optional; falls back to cached event if unset
uvicorn main:app --reload --port 8000
```
- `GET  /scenario`      → the whole seeded world (every screen reads this)
- `POST /extract`       → article → structured event linked to BOM rows (Day 2 live moment)
- `POST /loop`          → fire the closed loop, returns full reasoning trace (Day 3 live moment)

Quick check the loop with no frontend:
```bash
python graph.py        # prints: Switch 40% to Sundram, +6% cost, −100% China dep, Cpk ≤ 0.03
```

**Frontend**
```bash
npx create-next-app@latest frontend --ts --tailwind --app --eslint
# then copy ControlRoom.tsx into app/ and render it from app/page.tsx
cd frontend && npm run dev      # http://localhost:3000
```

---

## Repo layout
```
autonerve/
  backend/
    scenario.json     ← SINGLE SOURCE OF TRUTH. Get this right, screens become mapping.
    main.py           ← FastAPI: /scenario, /extract, /loop
    graph.py          ← LangGraph 3-node closed loop (the moat)
    extraction.py     ← news → BOM linkage, cached fallback for demo-day safety
    requirements.txt
  frontend/
    ControlRoom.tsx   ← pattern-setter: fetch /scenario, render, fire /loop live
```

---

## 5-day checklist (deadline Thu Jun 4)

### Day 1 — Sat 30 · scaffold + the world
- [ ] `create-next-app`, get `ControlRoom.tsx` rendering against `/scenario`
- [ ] Backend running, `/scenario` returns the seed
- [ ] Control Room (S6) looks like the slide, fully seeded
- [ ] **Cut line:** stop when the hero screen renders. No styling rabbit holes.

### Day 2 — Sun 31 · news → BOM (real AI #1)
- [ ] Pre-fetch 20–40 real articles (GDELT / Reuters RSS) → `articles.json`. Do NOT build live polling.
- [ ] `/extract` returns structured event linked to the 4 magnet BOM rows, live on screen
- [ ] Cache the LLM output so a flaky API can't break the demo
- [ ] **Cut line:** extraction returns the linkage. Don't expand the schema.

### Day 3 — Mon 1 · the loop (the moat, real AI #2)
- [ ] `graph.py` runs end to end → Sundram recommendation with Cpk impact
- [ ] Wire `/loop` to the AI-suggested-action card + Alternate Sourcing screen (S25)
- [ ] Animate each node firing — it reads beautifully on video
- [ ] **Cut line:** loop produces the recommendation. Cpk stays a lookup, not a model.

### Day 4 — Tue 2 · breadth + two cheap live add-ons
- [ ] 3–4 more seeded screens from the JSON: Supply Risk Map (S23), PO Forecast (S24),
      Material Substitution (S26), Real-Time QM (S38) — Recharts + layout, no new logic
- [ ] Meeting extraction (S35): one transcript → LLM → actions JSON (single call)
- [ ] Operator guidance (S37): tiny RAG over `scenario.sops` (single call)
- [ ] **Cut line:** these two work. Do not gold-plate.

### Day 5 — Wed 3 / Thu 4 · freeze, polish, record, submit
- [ ] Wed eve: **feature freeze.** Polish ONLY the click-path the video follows.
- [ ] Make live calls reliable; cached fallbacks verified
- [ ] Record 2–4 min video: alert fires → news→BOM → loop → Cpk impact → PO push
- [ ] Submit prototype + video + architecture

---

## Hard noes (this is where solo builds die)
- No real GNN — precomputed risk scores.
- No trained Cpk / forecasting / energy models — lookups + synthetic series.
- No live vision QC — seeded panels (one webcam clip only if Day 4 finishes early).
- No real SAP/ERP connector — "Push to SAP" writes a local row + toast.
- No live external feeds at judging time — everything shown live is also cached.

## Coverage story for the judges
Live spine touches 8 of 15 focus areas (risk, predictive planning, alternate
sourcing, material substitution, commodity intel, supplier analytics, ERP
decisions, process capability). The two cheap add-ons add 2 more. The rest are
seeded panels. Pitch: "architecture addresses all 15; these 10 run; this loop is
unique." True, and stronger than 15 fragile stubs.
