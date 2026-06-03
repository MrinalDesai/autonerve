"""AutoNerve API — serves the knowledge graph + runs the live decision thread."""
from __future__ import annotations
import csv
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import engine
import agents
import vision_oneshot
import extraction

HERE = Path(__file__).parent
app = FastAPI(title="AutoNerve API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# curated demo scenarios (the scenario simulator — what-if disruption injector)
SCENARIOS = {
    "neodymium": {"label": "China export controls on neodymium magnets",
                  "material": "neodymium", "china_cap_pct": 0.0, "severity": "critical"},
    "taiwan_chip": {"label": "Taiwan wafer-fab disruption (silicon)",
                    "material": "silicon", "china_cap_pct": 0.5, "severity": "high"},
    "copper": {"label": "Chilean strike — copper price shock",
               "commodity": "COPPER", "china_cap_pct": 1.0, "severity": "medium"},
}


@app.get("/health")
def health():
    return {"ok": True, "bom_rows": len(engine.BOM)}


@app.get("/scenario")
def scenario():
    """The whole world: products, parts, suppliers — the dashboard reads this."""
    products = [r for r in engine.BOM if r["part_type"] == "main_assembly"]
    return {"plant": "Mumbai Plant 1", "products": products,
            "parts": engine.BOM, "scenarios": SCENARIOS}


@app.get("/scenarios")
def list_scenarios():
    return SCENARIOS


class EventReq(BaseModel):
    scenario: str | None = None
    material: str | None = None
    region: str | None = None
    commodity: str | None = None
    china_cap_pct: float = 0.0


@app.post("/event")
def event(req: EventReq):
    """Fire a disruption event → propagation → exposure → optimized mix."""
    if req.scenario and req.scenario in SCENARIOS:
        s = SCENARIOS[req.scenario]
        return {"scenario": s["label"], **engine.run_event(
            material=s.get("material"), region=s.get("region"),
            commodity=s.get("commodity"), china_cap_pct=s.get("china_cap_pct", 0.0))}
    return engine.run_event(material=req.material, region=req.region,
                            commodity=req.commodity, china_cap_pct=req.china_cap_pct)


@app.get("/demand/{entity_id}")
def demand(entity_id: str):
    rows = [r for r in engine.DEMAND if r["entity_id"] == entity_id]
    return {"entity_id": entity_id, "series": rows}


@app.get("/articles")
def articles():
    """The simulated news corpus the scenario simulator replays."""
    return [{"id": a["id"], "date": a["date"], "source": a["source"],
             "region": a["region"], "headline": a["headline"]} for a in extraction.ARTICLES]


@app.post("/article/{article_id}")
def article(article_id: str):
    """Full live thread from one article: extract -> propagate -> exposure -> optimize."""
    return extraction.run_article(article_id)


FRONTEND = HERE.parent / "frontend" / "index.html"


@app.get("/")
def home():
    """Serve the Control Room dashboard."""
    return FileResponse(FRONTEND)


@app.get("/plant")
def plant():
    """Shop-floor status: lines, work queue, OEE."""
    return engine.plant_status()


@app.get("/plant/defects")
def plant_defects():
    """Supplier-wise defect summary + a live vision-QC scan feed."""
    return {"suppliers": engine.supplier_defects(), "scans": engine.recent_scans(14)}


class DefectReq(BaseModel):
    supplier_id: str
    part_id: str = "B-4471"


@app.post("/defect")
def defect(req: DefectReq):
    """Plant->Supply loop: flag a defect cluster -> re-optimize sourcing."""
    return engine.flag_defect(req.supplier_id, req.part_id)


class OperatorReq(BaseModel):
    question: str


@app.post("/operator")
def operator(req: OperatorReq):
    """Operator copilot: answer a shop-floor question grounded in the SOP base."""
    return engine.operator_answer(req.question)


@app.get("/plant/energy")
def plant_energy():
    """Line-2 energy trend + anomaly flag (Plant Cortex)."""
    return engine.plant_energy()


@app.get("/parts/sourceable")
def parts_sourceable():
    """BOM parts that have more than one supplier (a sourcing choice exists)."""
    return engine.sourceable_parts()


@app.get("/strategies/{part_id}")
def part_strategies(part_id: str):
    """Cheapest vs fastest sourcing strategy for a part, with the trade-off."""
    return engine.strategies(part_id)


@app.get("/bom/{product}")
def bom(product: str):
    """BOM node graph for a product, with per-node supplier paths (cost + lead)."""
    return engine.bom_tree(product)


@app.get("/forecast/{entity_id}")
def forecast(entity_id: str, scenario: str = "base"):
    """Predictive planning: demand forecast under base / conservative / aggressive."""
    return engine.forecast_series(entity_id, scenario=scenario)


@app.get("/meetings")
def meetings():
    return [{"id": m["id"], "title": m["title"]} for m in engine.MEETINGS]


class MeetingReq(BaseModel):
    meeting_id: str | None = None
    transcript: str | None = None


@app.post("/actions")
def actions(req: MeetingReq):
    """Schema-bound action extraction from a meeting transcript."""
    text = req.transcript
    if req.meeting_id:
        m = next((x for x in engine.MEETINGS if x["id"] == req.meeting_id), None)
        if m: text = m["transcript"]
    return {"actions": engine.extract_actions(text or "")}


@app.get("/procurement")
def procurement():
    """Price forecast + scarcity index + pre-stock buy plan with explanations."""
    return engine.procurement_plan()


@app.get("/cpk/{line_id}")
def cpk(line_id: str):
    return engine.cpk_forecast(line_id)


@app.get("/yield/{line_id}")
def yield_wf(line_id: str):
    return engine.yield_waterfall(line_id)


@app.get("/scorecard/{supplier_id}")
def scorecard(supplier_id: str):
    return engine.supplier_scorecard(supplier_id)


@app.get("/po-forecast")
def po_forecast():
    return engine.po_slip_forecast()


# ---- Agentic run (separate small dataset; real LLM per agent) ----
@app.get("/agent/signal")
def agent_signal():
    return agents.agent_signal()

@app.get("/agent/risk")
def agent_risk():
    return agents.agent_risk()

@app.get("/agent/defect")
def agent_defect():
    return agents.agent_defect()

@app.get("/agent/options")
def agent_options():
    return agents.agent_options()

@app.get("/agent/decision")
def agent_decision():
    return agents.agent_decision()

@app.get("/agent/impact")
def agent_impact():
    return agents.agent_impact()


@app.get("/sample-news.png")
def sample_news():
    from pathlib import Path
    img = Path(__file__).parent / "sample_news.png"
    return FileResponse(img) if img.exists() else {"error": "no image"}


@app.get("/vision/samples")
def vision_samples():
    return vision_oneshot.samples()

@app.get("/vision/inspect/{sample_id}")
def vision_inspect(sample_id: str):
    return vision_oneshot.inspect(sample_id)

@app.get("/parts/{name}.png")
def part_image(name: str):
    from pathlib import Path
    img = Path(__file__).parent / "parts" / f"{name}.png"
    return FileResponse(img) if img.exists() else {"error": "no image"}


@app.get("/agent/rejection")
def agent_rejection():
    return agents.agent_rejection()

@app.get("/agent/bottleneck")
def agent_bottleneck():
    return agents.agent_bottleneck()

@app.get("/agent/stock")
def agent_stock():
    return agents.agent_stock()
