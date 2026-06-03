# AutoNerve — build TODO

## Data to generate
- [ ] **Historical + forecasted data per part** — for each part/SKU: past actuals
      (demand/consumption history; price history for commodity-linked parts) PLUS a
      forward forecast with uncertainty bands. Feeds MRP explosion, Inventory
      Forecasting (S15), and the demand-series charts (S31). Covers the earlier
      "demand series (5 products × week × qty ± band)" item.

## Build — the live spine (priority order)
- [ ] Postgres + pgvector: load bom.csv; embeddings for article corpus + supplier-capability index
- [ ] Risk propagation (graph traversal + used_in_assemblies → affected products/POs + "why")
- [ ] MRP explosion (qty_per_unit × forecast → time-phased material requirements)
- [ ] Cost-vs-procurement optimizer — MILP via PuLP/OR-Tools (build for real)
- [ ] Scenario simulator (event injection → disruption prob + price → alternates + inventory boost)
- [ ] News→BOM extraction live (extraction.py wired to a pre-fetched article corpus)
- [ ] SSE live-sync (dashboard reacts the instant a signal lands)
- [ ] 3D globe (react-globe.gl / deck.gl arcs) — hero video shot
- [ ] Control Room + ~5 hero screens (Next.js)

## Cheap live adds (only after the spine works)
- [ ] Meeting & action extraction — single Mistral call on the seeded transcript
- [ ] Operator guidance — RAG over scenario.sops (BGE-M3 + Mistral)
- [ ] "Explain this PO" — single Mistral call

## Seeded panels
- [ ] Plant-quality screens (production analysis S12, Cpk S13, vision S14/S20, real-time QM S22)
- [ ] Energy (S16) — seeded panel or drop from demo; decide

## Deck / deliverables
- [ ] Update S39 → local Ollama stack (Mistral 7B + VLM + BGE-M3); on-prem framing
- [ ] Architecture-clarity slide (central BOM as the spine; which columns each layer reads)
- [ ] Business-impact / scalability model (₹ exposure avoided; ₹17.5K MSME economics)
- [ ] Demo video (2–4 min, built around the optimizer re-solve climax)

## Open decisions
- [ ] Live-sync: SSE (leaning yes)
- [ ] Seed accuracy: grounded hero families + AI filler vs fully AI-generated
- [ ] Optimizer objective: cost-min with geopolitical-dependency constraint vs weighted blend
