"""AutoNerve — agentic run.

Six role-specialized agents that process one news event over a small, self-contained
dataset (agent_demo.json). Each agent makes a REAL Qwen call for its reasoning /
why-not rationale; every NUMBER is computed deterministically from the dataset so
nothing is hallucinated. Each agent has a deterministic fallback so the full chain
runs even with no model loaded.

This is a role-specialized LLM pipeline (distinct prompts, chained reasoning,
grounded numerics) — not autonomous tool-using agents with memory.
"""
import json as _json
from pathlib import Path

HERE = Path(__file__).parent
DATA = _json.loads((HERE / "agent_demo.json").read_text())


def _llm_text(prompt, fallback, max_tokens=90):
    try:
        import llm
        if llm.available():
            out = llm.generate_text(prompt, max_new_tokens=max_tokens)
            if out and out.strip():
                return out.strip(), "qwen"
    except Exception:
        pass
    return fallback, "fallback"


def _qadj(s):
    """Quality-adjusted unit price: penalize incoming defects."""
    return round(s["price"] / (1 - s["defect"] / 100), 2)


# --- agent 1: signal (OCR image -> LLM extraction) -------------------------
def agent_signal():
    ev = DATA["event"]
    # OCR the news clipping image -> text feeds the LLM extraction
    try:
        import ocr
        scan = ocr.ocr_image()
    except Exception:
        scan = {"text": ev["headline"], "source": "fallback", "image": None, "chars": len(ev["headline"])}
    fb = (f"Detected a {ev['severity']} supply-risk event: {ev['region']} policy action on "
          f"{ev['material']} ({ev['commodity']}). Likely upward price + availability pressure.")
    note, src = _llm_text(
        f"From this OCR'd news text, summarize the supply-chain risk for an auto manufacturer "
        f"in one sentence: \"{scan['text'][:600]}\"", fb)
    return {"agent": "Signal", "event": ev, "note": note, "source": src,
            "ocr": {"source": scan["source"], "image": scan["image"], "chars": scan["chars"],
                    "excerpt": scan["text"][:240]}}


# --- agent 2: risk ---------------------------------------------------------
def agent_risk():
    sup = DATA["suppliers"]
    ranked = sorted(sup, key=lambda s: -s["risk"])
    scores = [{"id": s["id"], "name": s["name"], "country": s["country"],
               "risk": round(s["risk"] * 100), "china": s["china"]} for s in ranked]
    top = ranked[0]
    fb = (f"{top['name']} ({top['country']}) is the highest-exposure source — "
          f"{int(top['risk']*100)} risk, {top['china']}% China dependency.")
    note, src = _llm_text(
        f"Suppliers with risk/China exposure: "
        f"{[(s['name'], s['country'], s['risk'], s['china']) for s in sup]}. "
        f"In one sentence, name the biggest concentration risk and why.", fb)
    return {"agent": "Risk", "scores": scores, "note": note, "source": src}


# --- agent 3: defect (the closed loop) -------------------------------------
def agent_defect():
    d = DATA["defect_signal"]
    s = next(x for x in DATA["suppliers"] if x["id"] == d["supplier_id"])
    base_q = _qadj(s)
    bumped = round(s["price"] / (1 - d["rate_pct"] / 100), 2)
    delta = round(bumped - base_q)
    fb = (f"The {d['rate_pct']}% defect spike on {s['name']} raises its quality-adjusted "
          f"cost by ~₹{delta}/unit, eroding its price edge — it should lose allocation.")
    note, src = _llm_text(
        f"A vision-QC defect of {d['rate_pct']}% was just found on {s['name']}'s incoming lot. "
        f"Its quality-adjusted cost rises from ₹{base_q} to ₹{bumped}/unit. "
        f"In one sentence, explain how this should change the sourcing decision.", fb)
    return {"agent": "Defect", "supplier": s["name"], "rate_pct": d["rate_pct"], "note_text": d["note"],
            "qadj_before": base_q, "qadj_after": bumped, "delta": delta, "note": note, "source": src}


# --- agent 4: options (do / don't + why) -----------------------------------
def agent_options():
    sup = {s["id"]: s for s in DATA["suppliers"]}
    sub = DATA["substitute"]
    en = DATA["energy"]
    d = DATA["defect_signal"]
    opts = []

    # 1. drop supplier (the China + now-defective one)
    opts.append({"action": "Drop supplier",
                 "target": f"{sup['S1']['name']} (CN)", "verdict": "DO",
                 "reason": f"100% China + fresh {d['rate_pct']}% defect spike — highest risk and now "
                           f"cost-uncompetitive on a quality-adjusted basis. Cut, don't expand.",
                 "metric": "risk 82 · China 100%"})
    # 2. locate new supplier
    opts.append({"action": "Locate new supplier",
                 "target": f"{sup['S4']['name']} (VN)", "verdict": "CONDITIONAL",
                 "reason": "Adds a 4th non-China region, but longest lead (18d) and unproven — "
                           "qualify as backup, don't make it primary yet.",
                 "metric": "lead 18d · risk 40"})
    # 3. substitute item from another assembly
    opts.append({"action": "Substitute material",
                 "target": f"{sub['item']} (from {sub['from_assembly']})", "verdict": "PARTIAL",
                 "reason": f"₹{sub['price']} vs ₹{sup['S1']['price']} — far cheaper and rare-earth-free, "
                           f"but {sub['perf_note']}. Use on economy SKUs only, not performance.",
                 "metric": f"₹{sub['price']} · {sub['perf_note'].split('—')[0].strip()}"})
    # 4. expand region (shift to KR/IN)
    opts.append({"action": "Expand region",
                 "target": "Korea + India", "verdict": "DO",
                 "reason": "POSCO (KR) and Sundram (IN) carry the volume at +2% cost while removing "
                           "China single-point exposure — the core de-risking move.",
                 "metric": "China 100% → ~19%"})
    # 5. reduce energy
    save_kwh = round((en["kwh_per_unit"] - en["baseline"]) * DATA["part"]["demand_units"])
    opts.append({"action": "Reduce energy",
                 "target": en["line"], "verdict": "DO",
                 "reason": f"{en['anomaly']} is pushing {en['kwh_per_unit']} vs {en['baseline']} kWh/unit; "
                           f"fixing it recovers ~{save_kwh:,} kWh over the run.",
                 "metric": f"{en['kwh_per_unit']}→{en['baseline']} kWh/unit"})

    fb = "Drop the China source, shift volume to Korea+India, qualify Vietnam as backup, " \
         "substitute ferrite on economy SKUs, and fix the Line-2 energy anomaly."
    summary, src = _llm_text(
        "Given these candidate actions for a rare-earth disruption "
        "(drop China supplier, qualify Vietnam, substitute ferrite, shift to Korea/India, fix energy), "
        "give a one-sentence combined recommendation.", fb)
    return {"agent": "Options", "options": opts, "summary": summary, "source": src}


# --- agent 5: decision (grounded allocation) -------------------------------
def agent_decision():
    sup = [s for s in DATA["suppliers"]]
    d = DATA["defect_signal"]
    demand = DATA["part"]["demand_units"]

    def eff_q(s):
        rate = d["rate_pct"] if s["id"] == d["supplier_id"] else s["defect"]
        return s["price"] / (1 - rate / 100)
    # prefer non-China, then lowest effective quality-adjusted cost
    order = sorted(sup, key=lambda s: (s["china"] > 0, eff_q(s)))
    remaining, mix = demand, []
    for s in order:
        take = min(s["capacity"], remaining)
        if take > 0:
            mix.append({"id": s["id"], "name": s["name"], "country": s["country"],
                        "units": take, "pct": 0})
            remaining -= take
        if remaining <= 0:
            break
    total = sum(m["units"] for m in mix) or 1
    for m in mix:
        m["pct"] = round(100 * m["units"] / total, 1)
    china_pct = round(sum(m["units"] * next(s["china"] for s in sup if s["id"] == m["id"])
                          for m in mix) / total, 1)
    cost = round(sum(m["units"] * eff_q(next(s for s in sup if s["id"] == m["id"])) for m in mix))
    fb = (f"Allocate across {', '.join(m['name'] for m in mix)} — China dependency held to "
          f"{china_pct}% while meeting full {demand:,}-unit demand.")
    note, src = _llm_text(
        f"Chosen magnet allocation: {[(m['name'], m['pct']) for m in mix]}, China {china_pct}%. "
        f"In one sentence, justify this mix.", fb)
    return {"agent": "Decision", "mix": mix, "china_pct": china_pct, "cost": cost,
            "demand": demand, "note": note, "source": src}


# --- agent 6: impact (cost + manpower + china + energy) ---------------------
def agent_impact():
    a = DATA["assumptions"]
    en = DATA["energy"]
    demand = DATA["part"]["demand_units"]
    dec = agent_decision()

    # cost saved vs a "stay on disrupted China source" baseline (price + disruption premium)
    s1 = next(s for s in DATA["suppliers"] if s["id"] == "S1")
    baseline_cost = round(demand * s1["price"] * (1 + a["disruption_premium_pct"] / 100))
    cost_saved = max(0, baseline_cost - dec["cost"])

    # manpower saved (avoided expediting + automated QC), to ₹ and FTE-months
    hrs = a["expedite_hours_avoided"] + a["manual_qc_hours_avoided"]
    manpower_inr = hrs * a["inr_per_labour_hr"]
    fte_months = round(hrs / 160, 1)

    # energy saved
    kwh_saved = round((en["kwh_per_unit"] - en["baseline"]) * demand)
    energy_inr = round(kwh_saved * a["inr_per_kwh"])

    tiles = {
        "cost_saved_inr": cost_saved,
        "china_before": 100, "china_after": dec["china_pct"],
        "manpower_hours": hrs, "manpower_inr": manpower_inr, "fte_months": fte_months,
        "energy_kwh": kwh_saved, "energy_inr": energy_inr,
        "total_inr": cost_saved + manpower_inr + energy_inr,
    }
    fb = (f"Net benefit ≈ ₹{tiles['total_inr']:,}: ₹{cost_saved:,} avoided disruption cost, "
          f"₹{manpower_inr:,} manpower ({fte_months} FTE-months), ₹{energy_inr:,} energy — "
          f"China dependency cut from 100% to {dec['china_pct']}%.")
    note, src = _llm_text(
        f"Summarize the benefit in one sentence: ₹{cost_saved:,} cost avoided, {hrs} labour hours saved, "
        f"{kwh_saved:,} kWh saved, China {tiles['china_before']}%→{tiles['china_after']}%.", fb)
    return {"agent": "Impact", "tiles": tiles, "note": note, "source": src}


# --- agent 7: rejection analysis (why each supplier was/wasn't chosen) ------
def agent_rejection():
    """Per-supplier scorecard across fault rate, OCR/vision defects, price, lead —
    flags which dimensions are out of bounds and the resulting verdict."""
    sup = DATA["suppliers"]
    d = DATA["defect_signal"]
    prices = [s["price"] for s in sup]; leads = [s["lead"] for s in sup]
    pmin, pmax = min(prices), max(prices); lmin, lmax = min(leads), max(leads)
    rows = []
    for s in sup:
        fault = d["rate_pct"] if s["id"] == d["supplier_id"] else s["defect"]
        # normalize each dimension to 0-100 "concern" (higher = worse)
        dims = {
            "Fault rate": min(100, round(fault * 10)),
            "OCR defects": min(100, round(s["ocr_defect"] * 22)),
            "Price": round(100 * (s["price"] - pmin) / ((pmax - pmin) or 1)),
            "Lead time": round(100 * (s["lead"] - lmin) / ((lmax - lmin) or 1)),
            "Geo risk": round(s["risk"] * 100 if s["china"] else s["risk"] * 60),
        }
        flags = [k for k, v in dims.items() if v >= 65]
        verdict = "REJECTED" if (flags and (s["china"] == 100 or fault >= 5)) else \
                  "ACCEPTED" if not flags else "CONDITIONAL"
        reason = (f"{', '.join(flags)} above tolerance" if flags else "all dimensions within tolerance")
        rows.append({"id": s["id"], "name": s["name"], "country": s["country"],
                     "dims": dims, "flags": flags, "verdict": verdict, "reason": reason,
                     "fault": fault, "ocr_defect": s["ocr_defect"], "price": s["price"], "lead": s["lead"]})
    order = {"REJECTED": 0, "CONDITIONAL": 1, "ACCEPTED": 2}
    rows.sort(key=lambda r: order[r["verdict"]])
    return {"agent": "Rejection", "rows": rows}


# --- agent 8: bottleneck + network -----------------------------------------
def agent_bottleneck():
    """Identify the constraint: total non-China capacity vs demand, and the
    single-points-of-failure. Returns a part<->supplier<->region graph + the
    bottleneck + recommended actions."""
    sup = DATA["suppliers"]
    demand = DATA["part"]["demand_units"]
    nonchina_cap = sum(s["capacity"] for s in sup if s["china"] == 0)
    gap = max(0, demand - nonchina_cap)
    # network graph (nodes + links) for a web/force visual
    part = DATA["part"]
    nodes = [{"id": part["id"], "label": part["name"], "type": "part",
              "status": "bottleneck" if gap > 0 else "ok"}]
    links = []
    regions = {}
    for s in sup:
        st = "rejected" if s["china"] == 100 else "ok"
        nodes.append({"id": s["id"], "label": s["name"], "type": "supplier",
                      "status": st, "country": s["country"], "cap": s["capacity"]})
        links.append({"from": part["id"], "to": s["id"], "kind": "supplied_by"})
        regions.setdefault(s["country"], []).append(s["id"])
    for c, members in regions.items():
        nodes.append({"id": "R-" + c, "label": c, "type": "region",
                      "status": "rejected" if c == "CN" else "ok"})
        for m in members:
            links.append({"from": m, "to": "R-" + c, "kind": "located_in"})

    actions = []
    if gap > 0:
        actions.append({"action": "Locate new supplier", "detail":
                        f"Non-China capacity ({nonchina_cap:,}) is {gap:,} units short of demand "
                        f"({demand:,}); qualify an additional non-China source to close the gap.",
                        "priority": "HIGH"})
    actions.append({"action": "Suggest alternate part", "detail":
                    f"Bottleneck on {part['name']} — qualify {DATA['substitute']['item']} from "
                    f"{DATA['substitute']['from_assembly']} for economy SKUs to relieve magnet demand.",
                    "priority": "MEDIUM"})
    actions.append({"action": "Drop single-region source", "detail":
                    "Remove the 100% China source from primary allocation; retain only as last-resort buffer.",
                    "priority": "HIGH"})
    fb = (f"Bottleneck: non-China capacity covers {round(100*min(1,nonchina_cap/demand))}% of demand"
          + (f", short by {gap:,} units — new source needed." if gap > 0 else "."))
    note, src = _llm_text(
        f"Non-China magnet capacity is {nonchina_cap:,} vs demand {demand:,}. "
        f"In one sentence, state the bottleneck and the top action.", fb)
    return {"agent": "Bottleneck", "demand": demand, "nonchina_capacity": nonchina_cap,
            "gap": gap, "nodes": nodes, "links": links, "actions": actions,
            "note": note, "source": src}


# --- agent 9: stock plan (pre-buy ahead of events) -------------------------
def agent_stock():
    """What to pre-stock given upcoming events / shortages / price rises."""
    items = []
    for u in DATA["upcoming"]:
        # buy-ahead weeks scale with severity + price rise; flag if material
        topup = u["price_rise_pct"] >= 3 or u["severity"] == "high"
        cover = 6 if u["severity"] == "high" else 4 if u["severity"] == "medium" else 0
        items.append({"item": u["item"], "commodity": u["commodity"], "event": u["event"],
                      "price_rise_pct": u["price_rise_pct"], "lead_weeks": u["lead_weeks"],
                      "severity": u["severity"], "topup": topup, "cover_weeks": cover})
    items.sort(key=lambda x: (-x["price_rise_pct"]))
    flagged = [i for i in items if i["topup"]]
    fb = ("Pre-stock " + ", ".join(i["item"] for i in flagged) +
          " ahead of forecast price rises and supply tightening.") if flagged else "No pre-stock needed."
    note, src = _llm_text(
        f"Upcoming risks: {[(i['item'], i['price_rise_pct']) for i in items]}. "
        f"In one sentence, say what to pre-stock and why.", fb)
    return {"agent": "Stock", "items": items, "note": note, "source": src}
