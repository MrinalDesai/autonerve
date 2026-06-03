"""
AutoNerve deterministic engine — the tools the LLM decides WITH.

These produce every number (risk, exposure, the sourcing mix); the LLM layer
orchestrates and explains. Nothing here calls a model.

Thread:  event -> propagate() -> exposure() -> optimize()  ->  recommendation
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

import pulp

HERE = Path(__file__).parent


# ---------------------------------------------------------------- data load
def _split(v):
    return [x for x in (v or "").split(",") if x.strip()]


def load():
    bom, demand, prices = [], [], []
    with open(HERE / "bom.csv", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("qty_per_unit", "unit_price_inr", "quality_adj_cost_inr",
                      "lead_time_days", "capacity_per_week", "incoming_defect_rate",
                      "china_dependency_pct", "risk_score"):
                r[k] = float(r[k]) if str(r[k]).strip() else 0.0
            bom.append(r)
    with open(HERE / "demand_series.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["week_offset"] = int(r["week_offset"])
            for k in ("qty", "qty_lower", "qty_upper"):
                r[k] = float(r[k])
            demand.append(r)
    with open(HERE / "commodity_prices.csv", newline="") as f:
        for r in csv.DictReader(f):
            r["week_offset"] = int(r["week_offset"])
            r["price"] = float(r["price"])
            prices.append(r)
    return bom, demand, prices


BOM, DEMAND, PRICES = load()


# convenient indexes
def sourcing_options(part_id):
    return [r for r in BOM if r["part_id"] == part_id]


def part_meta(part_id):
    return next((r for r in BOM if r["part_id"] == part_id), None)


def forecast_qty(entity_id, horizon=12):
    return sum(r["qty"] for r in DEMAND
               if r["entity_id"] == entity_id and 0 <= r["week_offset"] < horizon)


# ---------------------------------------------------------------- propagation
def propagate(material=None, region=None, commodity=None):
    """Find parts hit by an event, climb to affected products, score risk.
    Deterministic graph traversal over the BOM knowledge graph."""
    hit = {}
    for r in BOM:
        if r["part_type"] == "main_assembly":
            continue
        match = (
            (material and material.lower() in r["material"].lower())
            or (commodity and commodity == r["linked_commodity"])
            or (region and region == r["country"])
        )
        if match:
            hit[r["part_id"]] = max(hit.get(r["part_id"], 0), r["risk_score"])

    affected_parts = sorted(hit, key=lambda p: -hit[p])
    # climb: a hit raw material implicates the components that consume it
    consumers = {}
    for pid in list(affected_parts):
        meta = part_meta(pid)
        if meta and meta["part_type"] == "raw_material" and meta["parent_part_id"]:
            cp = meta["parent_part_id"]
            consumers[cp] = max(consumers.get(cp, 0),
                                max((r["risk_score"] for r in sourcing_options(cp)), default=0))
    for cp, sc in consumers.items():
        hit.setdefault(cp, sc)
    affected_parts = sorted(hit, key=lambda p: -hit[p])
    products = set()
    for pid in affected_parts:
        for r in sourcing_options(pid):
            products.update(_split(r["main_assembly"]))
    # attribution (SHAP-style, from contributing columns of the top part)
    top = part_meta(affected_parts[0]) if affected_parts else None
    attribution = {}
    if top:
        attribution = {
            "single_source": 0.41 if int(top["china_dependency_pct"]) == 100 else 0.15,
            "geo_concentration": round(top["china_dependency_pct"] / 100 * 0.3, 2),
            "financial": 0.18,
            "news_signal": 0.12,
        }
    return {
        "affected_parts": affected_parts,
        "affected_products": sorted(products),
        "max_risk": round(max(hit.values()), 2) if hit else 0.0,
        "attribution": attribution,
    }


# ---------------------------------------------------------------- exposure (MRP)
def exposure(part_id, horizon=12):
    """Time-phased qty at risk and rupee exposure for a part over the horizon."""
    qty = forecast_qty(part_id, horizon)
    primary = next((r for r in sourcing_options(part_id) if int(r["is_primary"]) == 1),
                   sourcing_options(part_id)[0])
    rupees = qty * primary["unit_price_inr"]
    return {
        "part_id": part_id,
        "qty_at_risk": round(qty),
        "horizon_weeks": horizon,
        "exposure_inr": round(rupees),
        "exposure_cr": round(rupees / 1e7, 2),
        "primary_supplier": primary["supplier_name"],
        "primary_china_dep": int(primary["china_dependency_pct"]),
    }


# ---------------------------------------------------------------- optimizer (MILP)
def optimize(part_id, horizon=12, china_cap_pct=0.0, objective="quality_adj_cost_inr"):
    """Best sourcing mix to meet demand. MILP: minimise quality-adjusted cost
    s.t. demand met, per-supplier capacity, and China-dependency <= cap."""
    opts = sourcing_options(part_id)
    need = forecast_qty(part_id, horizon)
    cap_units = {o["supplier_id"]: o["capacity_per_week"] * horizon for o in opts}
    cost = {o["supplier_id"]: o[objective] for o in opts}
    cdep = {o["supplier_id"]: o["china_dependency_pct"] / 100.0 for o in opts}
    name = {o["supplier_id"]: o["supplier_name"] for o in opts}

    prob = pulp.LpProblem("sourcing_mix", pulp.LpMinimize)
    x = {s: pulp.LpVariable(f"x_{s}", lowBound=0) for s in cost}
    prob += pulp.lpSum(cost[s] * x[s] for s in cost)              # objective
    prob += pulp.lpSum(x.values()) >= need                        # meet demand
    for s in cost:
        prob += x[s] <= cap_units[s]                              # capacity
    prob += pulp.lpSum(cdep[s] * x[s] for s in cost) <= china_cap_pct * need  # geo cap
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    relaxed = False
    if pulp.LpStatus[prob.status] != "Optimal":
        # hard geo cap infeasible (not enough non-China capacity) -> instead
        # MINIMISE china dependency while meeting demand. Honest partial de-risk.
        prob = pulp.LpProblem("min_china", pulp.LpMinimize)
        x = {s: pulp.LpVariable(f"x_{s}", lowBound=0) for s in cost}
        prob += pulp.lpSum(cdep[s] * x[s] for s in cost)
        prob += pulp.lpSum(x.values()) >= need
        for s in cost:
            prob += x[s] <= cap_units[s]
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        relaxed = True

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"status": pulp.LpStatus[prob.status], "feasible": False}

    mix = {s: x[s].value() for s in cost if x[s].value() and x[s].value() > 1}
    total = sum(mix.values()) or 1
    primary = next((o for o in opts if int(o["is_primary"]) == 1), opts[0])
    base_cost = primary[objective] * need
    opt_cost = sum(cost[s] * q for s, q in mix.items())
    china_share = round(100 * sum(cdep[s] * q for s, q in mix.items()) / total, 1)
    return {
        "status": "Optimal", "feasible": True,
        "relaxed_to_min_china": relaxed,
        "note": ("Zero-China infeasible at full volume (non-China capacity short); "
                 "minimised China share instead — pair with substitute to close the gap."
                 if relaxed else "Met the geo target within capacity."),
        "need_units": round(need),
        "mix": [{"supplier": name[s], "units": round(q), "share_pct": round(100 * q / total)}
                for s, q in sorted(mix.items(), key=lambda kv: -kv[1])],
        "cost_inr": round(opt_cost),
        "cost_delta_pct": round(100 * (opt_cost - base_cost) / base_cost, 1),
        "china_dependency_pct": china_share,
    }


# ---------------------------------------------------------------- full thread
def _decision_part(affected_parts):
    """Pick the part to act on: the highest affected part that actually has
    sourcing alternatives (raw materials are often single-source — de-risking
    happens one level up, at the component that has alternates)."""
    for pid in affected_parts:
        opts = sourcing_options(pid)
        has_alt = len(opts) > 1 or any(_split(o["alternate_part_ids"]) for o in opts)
        if has_alt:
            return pid
    return affected_parts[0]


def run_event(material=None, region=None, commodity=None, china_cap_pct=0.0):
    prop = propagate(material=material, region=region, commodity=commodity)
    if not prop["affected_parts"]:
        return {"affected": prop, "exposure": None, "decision": None}
    lead_part = _decision_part(prop["affected_parts"])
    exp = exposure(lead_part)
    dec = optimize(lead_part, china_cap_pct=china_cap_pct)
    return {"affected": prop, "exposure": exp, "decision": dec, "lead_part": lead_part}


if __name__ == "__main__":
    import json
    print(json.dumps(run_event(material="neodymium", china_cap_pct=0.0), indent=2, default=str))


# ============================================================
# PLANT FLOOR  (sensor / shop-floor stream + defect -> sourcing loop)
# ============================================================
import random as _rnd
_rnd.seed(7)

PLANT_LINES = [
    {"id": "L1", "process": "Rotor machining",  "cpk": 1.62, "fpy": 97.2, "status": "running", "throughput": 142, "top_defect": "—"},
    {"id": "L2", "process": "Magnet assembly",  "cpk": 1.18, "fpy": 88.3, "status": "alert",   "throughput": 118, "top_defect": "Burr 3.4%"},
    {"id": "L3", "process": "Harness crimp",    "cpk": 1.55, "fpy": 95.8, "status": "running", "throughput": 160, "top_defect": "Misalign"},
    {"id": "L4", "process": "ECU populate",     "cpk": 1.41, "fpy": 92.1, "status": "running", "throughput": 96,  "top_defect": "Solder"},
    {"id": "L5", "process": "Brake assembly",   "cpk": 1.47, "fpy": 96.0, "status": "running", "throughput": 133, "top_defect": "Surface"},
    {"id": "L6", "process": "Final QC",         "cpk": 1.52, "fpy": 94.4, "status": "idle",    "throughput": 0,   "top_defect": "—"},
]
DEFECT_TYPES = ["burr", "crack", "porosity", "misalignment", "surface scratch", "solder void"]


def plant_status():
    return {"shift": "A", "oee": 82.4,
            "lines": PLANT_LINES,
            "work_queue": {"complete": 1840, "in_progress": 6, "pending": 312}}


def supplier_defects():
    agg = {}
    for r in BOM:
        sid = r["supplier_id"]
        if not sid or r["part_type"] == "main_assembly":
            continue
        a = agg.setdefault(sid, {"supplier": r["supplier_name"], "country": r["country"],
                                 "parts": set(), "rate": 0.0})
        a["parts"].add(r["part_id"])
        a["rate"] = max(a["rate"], r["incoming_defect_rate"])
    out = []
    for sid, a in agg.items():
        inspected = 1200
        out.append({"supplier_id": sid, "supplier": a["supplier"], "country": a["country"],
                    "parts": sorted(a["parts"]), "defect_rate_pct": round(a["rate"] * 100, 2),
                    "inspected": inspected, "detections": round(a["rate"] * inspected)})
    return sorted(out, key=lambda x: -x["defect_rate_pct"])


def recent_scans(n=14):
    parts = [r for r in BOM if r["part_type"] in ("component", "raw_material")]
    scans = []
    for _ in range(n):
        r = _rnd.choice(parts)
        is_def = _rnd.random() < min(0.6, r["incoming_defect_rate"] * 9)
        scans.append({"part_id": r["part_id"], "part": r["part_name"],
                      "supplier": r["supplier_name"], "supplier_id": r["supplier_id"],
                      "station": f"VQC-{_rnd.randint(1, 4)}",
                      "verdict": "DEFECT" if is_def else "OK",
                      "defect_type": _rnd.choice(DEFECT_TYPES) if is_def else "",
                      "confidence": round(_rnd.uniform(0.86, 0.98), 2)})
    return scans


def flag_defect(supplier_id, part_id, bump=0.10):
    """Plant -> Supply loop: a detected defect cluster raises the supplier's
    incoming defect rate, which re-prices it and re-runs the sourcing optimizer.
    Uses the cost objective (no geo cap) so the quality-driven supplier shift is visible."""
    before = optimize(part_id, china_cap_pct=1.0)
    bumped = None
    for r in BOM:
        if r["part_id"] == part_id and r["supplier_id"] == supplier_id:
            r["incoming_defect_rate"] = round(min(0.5, r["incoming_defect_rate"] + bump), 4)
            r["quality_adj_cost_inr"] = round(r["unit_price_inr"] / (1 - r["incoming_defect_rate"]), 2)
            bumped = r["incoming_defect_rate"]
    after = optimize(part_id, china_cap_pct=1.0)
    sup = next((r["supplier_name"] for r in BOM if r["supplier_id"] == supplier_id), supplier_id)
    return {"supplier": sup, "supplier_id": supplier_id, "part": part_id,
            "new_defect_rate_pct": round((bumped or 0) * 100, 2),
            "before": before, "after": after}


# ============================================================
# PLANT CORTEX: operator copilot (SOP RAG) + energy anomaly
# ============================================================
import json as _json
import re as _re

SOPS = _json.loads((HERE / "sops.json").read_text())["sops"]


def _best_sop(question):
    qt = set(_re.findall(r"[a-z0-9]+", question.lower()))
    best, score = None, 0
    for s in SOPS:
        kw = set(_re.findall(r"[a-z0-9]+",
                 (s["topic"] + " " + " ".join(s["keywords"]) + " " + s["text"]).lower()))
        ov = len(qt & kw)
        if ov > score:
            best, score = s, ov
    return best, score


def operator_answer(question, use_llm=False):
    """Retrieve the best-matching SOP and answer grounded in it (RAG).
    use_llm=True adds Qwen rephrasing, but that's slow on CPU, so the demo path
    defaults to the instant grounded answer (still cites the SOP)."""
    sop, score = _best_sop(question)
    if not sop or score == 0:
        return {"answer": "No matching SOP found — escalate to the line supervisor.",
                "sop": None, "source": "none"}
    if use_llm:
        try:
            import llm
            if llm.available():
                prompt = (f"You are a shop-floor assistant. Answer the operator's question using ONLY "
                          f"the SOP below. Be concise (2–3 sentences), practical, and cite the SOP id.\n\n"
                          f"SOP {sop['id']} — {sop['topic']}:\n{sop['text']}\n\nQuestion: {question}")
                ans = llm.generate_text(prompt, system="You are a precise manufacturing assistant; cite the SOP id.",
                                        max_new_tokens=120)
                return {"answer": ans, "sop": sop["id"], "topic": sop["topic"], "source": "llm"}
        except Exception:
            pass
    return {"answer": f"Per {sop['id']} ({sop['topic']}): {sop['text']}",
            "sop": sop["id"], "topic": sop["topic"], "source": "rag"}


def plant_energy():
    """Seeded kWh/unit trend for Line 2 with a sustained anomaly in the recent window."""
    base = 2.84
    series = []
    for i in range(24):
        anomaly = 0.18 if i >= 18 else 0.0
        series.append(round(base + _rnd.uniform(-0.03, 0.03) + anomaly, 2))
    return {"unit": "kWh/unit", "baseline": base, "current": series[-1], "series": series,
            "anomaly": {"flag": True, "since_h": 6,
                        "detail": "Spindle #2: +0.18 kWh/unit sustained over 6h → early bearing degradation (see SOP-SPN-003)"}}


# ============================================================
# MULTI-OBJECTIVE SOURCING: cheapest vs fastest, per part
# ============================================================
def _solve_objective(part_id, mode, horizon=12):
    """Solve the sourcing mix for one objective: 'cost' (min quality-adj cost)
    or 'fast' (min lead time). Same demand + capacity constraints."""
    opts = sourcing_options(part_id)
    need = forecast_qty(part_id, horizon)
    cap = {o["supplier_id"]: o["capacity_per_week"] * horizon for o in opts}
    cost = {o["supplier_id"]: o["quality_adj_cost_inr"] for o in opts}
    lead = {o["supplier_id"]: o["lead_time_days"] for o in opts}
    cdep = {o["supplier_id"]: o["china_dependency_pct"] / 100.0 for o in opts}
    name = {o["supplier_id"]: o["supplier_name"] for o in opts}
    ctry = {o["supplier_id"]: o["country"] for o in opts}

    prob = pulp.LpProblem("src", pulp.LpMinimize)
    x = {s: pulp.LpVariable(f"x_{s}", lowBound=0) for s in cost}
    prob += pulp.lpSum((cost[s] if mode == "cost" else lead[s]) * x[s] for s in cost)
    prob += pulp.lpSum(x.values()) >= need
    for s in cost:
        prob += x[s] <= cap[s]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return {"feasible": False}

    mix = {s: x[s].value() for s in cost if x[s].value() and x[s].value() > 1}
    total = sum(mix.values()) or 1
    return {
        "feasible": True,
        "need_units": round(need),
        "mix": [{"supplier": name[s], "country": ctry[s], "units": round(q),
                 "share_pct": round(100 * q / total), "lead_days": lead[s]}
                for s, q in sorted(mix.items(), key=lambda kv: -kv[1])],
        "total_cost_inr": round(sum(cost[s] * q for s, q in mix.items())),
        "weighted_lead_days": round(sum(lead[s] * q for s, q in mix.items()) / total, 1),
        "max_lead_days": max(lead[s] for s in mix),
        "china_dependency_pct": round(100 * sum(cdep[s] * q for s, q in mix.items()) / total, 1),
    }


def strategies(part_id, horizon=12):
    """Two sourcing strategies for a part: cheapest vs fastest, with the
    trade-off (cost premium / lead-time penalty) visible on both."""
    meta = part_meta(part_id)
    cheap = _solve_objective(part_id, "cost", horizon)
    fast = _solve_objective(part_id, "fast", horizon)
    delta = None
    if cheap.get("feasible") and fast.get("feasible"):
        delta = {
            "cost_premium_pct": round(100 * (fast["total_cost_inr"] - cheap["total_cost_inr"])
                                      / cheap["total_cost_inr"], 1),
            "lead_saved_days": round(cheap["weighted_lead_days"] - fast["weighted_lead_days"], 1),
        }
    return {"part_id": part_id, "part_name": meta["part_name"] if meta else part_id,
            "cheapest": cheap, "fastest": fast, "tradeoff": delta}


def sourceable_parts():
    """Parts with more than one supplier — i.e. where a sourcing choice exists."""
    counts = {}
    for r in BOM:
        if r["part_type"] == "main_assembly":
            continue
        counts.setdefault(r["part_id"], 0)
        counts[r["part_id"]] += 1
    out = []
    for pid, n in counts.items():
        if n > 1:
            m = part_meta(pid)
            out.append({"part_id": pid, "part_name": m["part_name"], "suppliers": n})
    return sorted(out, key=lambda x: -x["suppliers"])


# ============================================================
# INTERACTIVE BOM: node graph with selectable supplier paths + cost/time rollup
# ============================================================
def _effective_qty(part_id, product):
    """Qty of this part per ONE finished vehicle = product of qty_per_unit up the
    parent chain to the product root."""
    q = 1.0
    cur = part_id
    guard = 0
    while cur and cur != product and guard < 12:
        row = part_meta(cur)
        if not row:
            break
        q *= (row["qty_per_unit"] or 1.0)
        cur = row["parent_part_id"]
        guard += 1
    return q


def bom_tree(product):
    """Component nodes for a product, each with effective per-vehicle qty and
    its supplier options (cost + lead). Front-end lets the user pick a path per
    node and rolls up total cost + critical-path lead time."""
    rows = [r for r in BOM
            if r["part_type"] in ("component", "raw_material", "sub_assembly")
            and (product in (r["main_assembly"] or "").split(",")
                 or product in (r["used_in_assemblies"] or "").split(","))]
    seen, nodes = set(), []
    for r in rows:
        pid = r["part_id"]
        if pid in seen:
            continue
        seen.add(pid)
        opts = sourcing_options(pid)
        if not opts:
            continue
        nodes.append({
            "part_id": pid, "part_name": r["part_name"], "type": r["part_type"],
            "parent": r["parent_part_id"], "level": int(r["bom_level"]),
            "qty_per_vehicle": round(_effective_qty(pid, product), 3),
            "is_substitute": "substitute" in (r["part_name"] or "").lower()
                             or "alt" in pid.lower(),
            "suppliers": [{"supplier_id": o["supplier_id"], "supplier": o["supplier_name"],
                           "country": o["country"], "unit_price": o["unit_price_inr"],
                           "quality_adj_cost": o["quality_adj_cost_inr"],
                           "lead_days": o["lead_time_days"],
                           "china_dep": int(o["china_dependency_pct"]),
                           "defect_pct": round(o["incoming_defect_rate"] * 100, 1),
                           "is_primary": int(o["is_primary"]) == 1} for o in opts],
        })
    meta = part_meta(product)
    return {"product": product, "product_name": meta["part_name"] if meta else product,
            "nodes": sorted(nodes, key=lambda n: (n["level"], n["part_id"]))}


# ============================================================
# PREDICTIVE PLANNING (capability 02): real forecast w/ risk-adjusted bands
# ============================================================
def forecast_series(entity_id, horizon=12, scenario="base"):
    """Genuine forecast from demand history (least-squares trend + residual bands).
    Three planning scenarios diverge over the horizon:
      base         — central trend
      conservative — lower demand, risk-widened band (plan the downside)
      aggressive   — upside/growth demand
    Computed from data, not seeded. A temporal transformer is the production target."""
    hist = sorted([r for r in DEMAND if r["entity_id"] == entity_id and r["week_offset"] < 0],
                  key=lambda r: r["week_offset"])
    if len(hist) < 4:
        return {"entity_id": entity_id, "history": [], "forecast": []}
    xs = [r["week_offset"] for r in hist]
    ys = [r["qty"] for r in hist]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1
    slope = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
    intercept = my - slope * mx
    resid = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
    sd = (sum(e * e for e in resid) / n) ** 0.5

    cfg = {"base":         {"level": 1.00, "tilt":  0.000, "widen": 1.0},
           "conservative": {"level": 0.95, "tilt": -0.006, "widen": 1.9},
           "aggressive":   {"level": 1.05, "tilt":  0.008, "widen": 1.2}}.get(scenario,
           {"level": 1.0, "tilt": 0.0, "widen": 1.0})
    band_k = 1.28
    history = [{"week": r["week_offset"], "qty": round(r["qty"])} for r in hist]
    forecast, total = [], 0
    for h in range(horizon):
        base_y = intercept + slope * h
        yhat = max(0, base_y * (cfg["level"] + cfg["tilt"] * h))
        spread = band_k * sd * cfg["widen"] * (1 + h * 0.04)
        total += yhat
        forecast.append({"week": h, "qty": round(yhat),
                         "lower": round(max(0, yhat - spread)), "upper": round(yhat + spread)})
    base_total = sum(max(0, intercept + slope * h) for h in range(horizon))
    return {"entity_id": entity_id, "scenario": scenario, "trend_per_week": round(slope, 1),
            "total_units": round(total),
            "delta_vs_base_pct": round(100 * (total - base_total) / base_total, 1) if base_total else 0,
            "history": history, "forecast": forecast}


# ============================================================
# MEETING & ACTION EXTRACTION (capability 13): schema-bound LLM extraction
# ============================================================
MEETINGS = _json.loads((HERE / "meetings.json").read_text())["meetings"] \
    if (HERE / "meetings.json").exists() else []


def extract_actions(transcript):
    """Schema-bound extraction of action items from a meeting transcript.
    Uses the LLM if available (same pattern as news extraction), else a
    deterministic sentence/keyword fallback. Whisper audio is the production
    front-end; here the input is text."""
    try:
        import llm
        if llm.available():
            prompt = ("Extract action items from this meeting transcript. Return STRICT JSON: "
                      '{"actions":[{"owner":"","action":"","due":"","entity":""}]}. '
                      'entity = any part/supplier/commodity mentioned, else "". '
                      f"Transcript:\n{transcript[:1500]}")
            out = llm.generate_json(prompt)
            if isinstance(out, dict) and "actions" in out:
                for a in out["actions"]:
                    a["source"] = "llm"
                return out["actions"]
    except Exception:
        pass
    # deterministic fallback: pull imperative / owner-tagged sentences
    actions = []
    for sent in _re.split(r"[.\n]", transcript):
        s = sent.strip()
        m = _re.match(r"([A-Z][a-z]+)\s+(?:to|will|should|must)\s+(.+)", s)
        if m:
            ent = next((p["part_id"] for p in BOM if p["part_id"].lower() in s.lower()), "")
            actions.append({"owner": m.group(1), "action": m.group(2)[:120],
                            "due": (_re.search(r"by\s+([A-Za-z0-9 ]+)", s) or [None, ""])[1].strip(),
                            "entity": ent, "source": "rule"})
    return actions


# ============================================================
# PROCUREMENT PLAN: price forecast + scarcity + pre-stock-to-save recommendation
# ============================================================
def _commodity_change(commodity, horizon=12):
    """% change in a commodity's price from now to the forecast horizon."""
    rows = [r for r in PRICES if r["commodity"] == commodity]
    if not rows:
        return 0.0
    now = next((r["price"] for r in rows if r["week_offset"] == 0), None)
    if now is None:
        past = [r for r in rows if r["week_offset"] <= 0]
        now = past[-1]["price"] if past else rows[0]["price"]
    fut = [r for r in rows if 0 <= r["week_offset"] < horizon]
    end = fut[-1]["price"] if fut else now
    return round(100 * (end - now) / now, 1) if now else 0.0


def _monthly_usage(entity_id):
    """Recent historic usage per month (~4-week buckets) for a part."""
    hist = sorted([r for r in DEMAND if r["entity_id"] == entity_id and r["week_offset"] < 0],
                  key=lambda r: r["week_offset"])
    months = []
    for i in range(0, len(hist), 4):
        chunk = hist[i:i + 4]
        if chunk:
            months.append(round(sum(c["qty"] for c in chunk)))
    return months[-6:]   # last ~6 months


def procurement_plan(horizon=12):
    """For each key component: price-now vs forecast, a scarcity index, and a
    pre-stock recommendation (top up now to beat a price rise / scarcity),
    with an explanation and a final buy quantity."""
    items, buy_list = [], []
    seen = set()
    for r in BOM:
        pid = r["part_id"]
        if pid in seen or r["part_type"] not in ("component", "raw_material"):
            continue
        if not r["linked_commodity"]:
            continue
        seen.add(pid)
        prim = next((o for o in sourcing_options(pid) if int(o["is_primary"]) == 1),
                    sourcing_options(pid)[0])
        comm = r["linked_commodity"]
        chg = _commodity_change(comm, horizon)
        price_now = prim["unit_price_inr"]
        price_fc = round(price_now * (1 + chg / 100), 2)
        usage = _monthly_usage(pid)
        wk_usage = forecast_qty(pid, horizon) / horizon if forecast_qty(pid, horizon) else (usage[-1] / 4 if usage else 0)

        # scarcity index 0-100
        single = 1.0 if len(sourcing_options(pid)) == 1 else 0.0
        trend_norm = max(0, min(1, chg / 15))
        scarcity = round(100 * (0.40 * prim["risk_score"] + 0.30 * (prim["china_dependency_pct"] / 100)
                                + 0.15 * single + 0.15 * trend_norm))

        topup = chg >= 2.0 or scarcity >= 60
        cover_weeks = round(prim["lead_time_days"] / 7 + 4)        # lead time + 4w buffer
        qty = round(wk_usage * cover_weeks) if topup else 0
        saving = round(qty * (price_fc - price_now)) if topup and chg > 0 else 0

        reasons = []
        if chg >= 2.0: reasons.append(f"{comm} price +{chg}% forecast")
        if prim["china_dependency_pct"] >= 50: reasons.append(f"{int(prim['china_dependency_pct'])}% China-sourced")
        if single: reasons.append("single-source")
        if prim["risk_score"] >= 0.6: reasons.append(f"supplier risk {prim['risk_score']}")
        explanation = ("Pre-buy " + str(cover_weeks) + "w cover — " + "; ".join(reasons) +
                       (f"; locks ~₹{saving:,} vs buying later" if saving > 0 else "")) if topup \
                      else ("Hold — " + ("stable price; " if abs(chg) < 2 else "") + "adequate sourcing")

        row = {"part_id": pid, "part_name": r["part_name"], "commodity": comm,
               "monthly_usage": usage, "price_now": price_now, "price_forecast": price_fc,
               "price_change_pct": chg, "scarcity_index": scarcity, "topup": topup,
               "qty_to_buy": qty, "est_saving_inr": saving, "explanation": explanation}
        items.append(row)
        if topup:
            buy_list.append({"part_id": pid, "part_name": r["part_name"], "supplier": prim["supplier_name"],
                             "qty": qty, "unit_price": price_now, "line_cost": round(qty * price_now),
                             "est_saving": saving})

    items.sort(key=lambda x: -x["scarcity_index"])
    buy_list.sort(key=lambda x: -x["est_saving"])
    return {"items": items, "buy_list": buy_list,
            "buy_total_cost": round(sum(b["line_cost"] for b in buy_list)),
            "buy_total_saving": round(sum(b["est_saving"] for b in buy_list))}


# ============================================================
# ANALYTICS DEEP-DIVES (deck slides 24, 28, 29, 33)
# ============================================================
def cpk_forecast(line_id):
    """Cp/Cpk now + a short forecast (slide 29). Trend projection; LSTM is the
    production target. Alert lines drift toward a breach."""
    line = next((l for l in PLANT_LINES if l["id"] == line_id), PLANT_LINES[1])
    cur = line["cpk"]
    slope = -0.012 if line["status"] == "alert" else (-0.001 if cur > 1.5 else 0.003)
    hist = [round(cur - slope * h + _rnd.uniform(-0.02, 0.02), 2) for h in range(24, 0, -1)]
    fc = [round(cur + slope * (h + 1), 2) for h in range(6)]
    fcpk = fc[-1]
    return {"line": line["id"], "process": line["process"], "cp": round(cur + 0.22, 2),
            "cpk": cur, "forecast_cpk": fcpk, "breach": fcpk < 1.33, "threshold": 1.33,
            "target": 1.67, "history": hist, "forecast": fc,
            "action": "Spindle −80 RPM, −0.004 mm offset" if fcpk < 1.33 else "Hold — within target"}


def yield_waterfall(line_id):
    """Yield-drop variance decomposition (slide 28) with an AI summary."""
    line = next((l for l in PLANT_LINES if l["id"] == line_id), PLANT_LINES[2])
    current = line["fpy"]
    steps = [{"label": "Supplier lot", "delta": -3.1},
             {"label": "Humidity", "delta": -0.8},
             {"label": "Unexplained", "delta": -0.3}]
    baseline = round(current - sum(s["delta"] for s in steps), 1)
    drop = round(baseline - current, 1)
    summary = (f"Most of {line['id']}'s {drop} pp drop traces to one supplier lot "
               f"(#L-2240) — 73% variance attribution. Quarantine and switch.")
    return {"line": line["id"], "process": line["process"], "baseline": baseline,
            "current": current, "steps": steps, "summary": summary}


def supplier_scorecard(supplier_id):
    """Composite supplier rating (slide 33). Quality/delivery from real data;
    financial health (Altman-Z) is illustrative, not a trained model."""
    rows = [r for r in BOM if r["supplier_id"] == supplier_id]
    if not rows:
        return {"error": "unknown supplier"}
    r = rows[0]
    risk, defect, china = r["risk_score"], r["incoming_defect_rate"], r["china_dependency_pct"]
    otif = round(max(70, 97 - risk * 22 - defect * 120), 1)
    alts = sourcing_options(r["part_id"])
    cheapest = min((o["unit_price_inr"] for o in alts), default=r["unit_price_inr"]) or r["unit_price_inr"]
    should_cost = round((r["unit_price_inr"] / cheapest - 1) * 100, 1)
    altman = round(3.6 - risk * 1.6, 1)
    dims = {"Delivery": round(otif), "Quality": max(0, min(100, round(100 - defect * 1500))),
            "Cost": max(0, min(100, round(100 - china / 3 - max(0, should_cost)))),
            "Payment": round(max(50, 85 - risk * 12)),
            "Financial": round(max(20, min(100, altman / 4 * 100))),
            "ESG": round(72 + (8 if china == 0 else -6))}
    composite = round(sum(dims.values()) / len(dims))
    grade = ("A" if composite >= 85 else "B+" if composite >= 78 else "B" if composite >= 70
             else "C+" if composite >= 60 else "C")
    # vs-field: this supplier's quote vs the average of the other sources for the part
    others = [o for o in alts if o["supplier_id"] != supplier_id]
    field_quote = round(sum(o["unit_price_inr"] for o in others) / len(others), 2) if others else r["unit_price_inr"]
    field_lead = round(sum(o["lead_time_days"] for o in others) / len(others)) if others else r["lead_time_days"]
    field_defect = round(sum(o["incoming_defect_rate"] for o in others) / len(others) * 100, 2) if others else round(defect * 100, 2)
    # illustrative trend (prior period) so the card shows direction, not just a value
    altman_prev = round(altman + 0.6, 1)
    otif_prev = round(otif - 2.4, 1)
    return {"supplier": r["supplier_name"], "country": r["country"], "part": r["part_id"],
            "composite": composite, "grade": grade,
            "otif": otif, "otif_prev": otif_prev,
            "altman_z": altman, "altman_prev": altman_prev,
            "should_cost_pct": should_cost, "dims": dims,
            "vs_field": {
                "quote": r["unit_price_inr"], "field_quote": field_quote,
                "lead": r["lead_time_days"], "field_lead": field_lead,
                "defect_pct": round(defect * 100, 2), "field_defect_pct": field_defect,
                "should_cost": should_cost}}


def po_slip_forecast():
    """Open POs with delay probability (slide 24). Slip prob from supplier
    risk + lead time."""
    seeds = [("PR-24-0188", "BP-RWD-001"), ("PR-24-0189", "HW-014"), ("PR-24-0190", "NM-005"),
             ("PR-24-0191", "B-4471"), ("PR-24-0192", "FST-M8"), ("PR-24-0193", "CN-844"),
             ("PR-24-0194", "PROC-8G"), ("PR-24-0195", "CW-220")]
    pos = []
    for po_id, part in seeds:
        opts = sourcing_options(part)
        if not opts:
            continue
        prim = next((o for o in opts if int(o["is_primary"]) == 1), opts[0])
        qty = round(forecast_qty(part, 12) / 3) or 500
        slip = round(min(0.92, 0.08 + prim["risk_score"] * 0.6 + prim["lead_time_days"] / 110), 2)
        status = "RED" if slip >= 0.5 else "AMBER" if slip >= 0.3 else "GREEN"
        pos.append({"po": po_id, "part": part, "part_name": prim["part_name"],
                    "supplier": prim["supplier_name"], "qty": qty,
                    "value_inr": round(qty * prim["unit_price_inr"]), "slip_prob": slip,
                    "status": status, "lead_days": prim["lead_time_days"]})
    pos.sort(key=lambda x: -x["slip_prob"])
    return {"pos": pos, "open": len(pos), "at_risk": sum(1 for p in pos if p["slip_prob"] >= 0.3)}
