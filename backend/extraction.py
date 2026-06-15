"""
News -> structured event extraction. Tries the local Qwen LLM (llm.py); if the
model is absent, falls back to a deterministic keyword extractor over the BOM's
own vocabulary, so the article->event->engine thread always runs.

The extracted event feeds engine.run_event() (propagation -> exposure -> optimize).
"""
from __future__ import annotations
import json
from pathlib import Path

import engine
import llm

HERE = Path(__file__).parent
ARTICLES = json.loads((HERE / "articles.json").read_text())["articles"]

# BOM vocabulary, derived from the data (so extractor stays in sync with the graph)
MATERIALS = sorted({r["material"].lower() for r in engine.BOM
                    if r["material"] and r["material"] not in ("assembly", "vehicle")})
COMMODITIES = sorted({r["linked_commodity"] for r in engine.BOM if r["linked_commodity"]})
COUNTRIES = sorted({r["country"] for r in engine.BOM if r["country"] and r["country"] != "-"})

REGION_NAMES = {"china": "CN", "taiwan": "TW", "korea": "KR", "india": "IN",
                "germany": "DE", "malaysia": "MY", "italy": "IT", "chile": "CL"}
MATERIAL_HINTS = {"neodymium": "neodymium", "ndfeb": "neodymium", "rare-earth": "neodymium",
                  "rare earth": "neodymium", "copper": "copper", "silicon": "silicon",
                  "wafer": "silicon", "steel": "steel", "aluminium": "aluminium",
                  "aluminum": "aluminium", "ferrite": "ferrite"}
COMMODITY_HINTS = {"copper": "COPPER", "neodymium": "RARE-EARTH", "rare-earth": "RARE-EARTH",
                   "rare earth": "RARE-EARTH", "silicon": "SILICON", "wafer": "SILICON",
                   "steel": "STEEL", "aluminium": "ALUMINIUM", "aluminum": "ALUMINIUM"}

# canonical material -> commodity, derived straight from the BOM graph
MAT_TO_COMM = {}
for _r in engine.BOM:
    m = (_r["material"] or "").lower()
    c = _r["linked_commodity"]
    if m and c and m not in ("assembly", "vehicle"):
        MAT_TO_COMM.setdefault(m, c)
# fold in the obvious synonyms the LLM tends to use
MAT_TO_COMM.update({"neodymium": "RARE-EARTH", "ndfeb": "RARE-EARTH",
                    "silicon": "SILICON", "copper": "COPPER", "steel": "STEEL",
                    "aluminium": "ALUMINIUM"})
VALID_TYPES = {"export-control", "tariff", "strike", "port-congestion",
               "price", "disruption", "none"}


def _validate(ev: dict, article_text: str) -> dict:
    """Ground a raw LLM event against the BOM vocabulary. The LLM proposes;
    the knowledge graph corrects. Material is trusted over commodity (it maps
    cleanly); a commodity that contradicts the material is overwritten."""
    t = article_text.lower()
    mat = (ev.get("material") or "").strip().lower()
    com = (ev.get("commodity") or "").strip().upper()
    ctype = (ev.get("controlType") or "none").strip().lower()

    # snap material to a known one (drop hallucinations)
    if mat and mat not in MAT_TO_COMM:
        mat = next((v for k, v in MATERIAL_HINTS.items() if k in t), "")
    # derive/repair commodity from the material; never trust a contradictory commodity
    if mat in MAT_TO_COMM:
        com = MAT_TO_COMM[mat]
    elif com not in COMMODITIES:
        com = next((v for k, v in COMMODITY_HINTS.items() if k in t), "")

    if ctype not in VALID_TYPES:
        ctype = "disruption"
    # a disruption with no usable signal is noise; a real signal with type=none
    # but disruption words present is a missed event -> re-check the text
    has_signal = bool(mat or com)
    strong_words = any(w in t for w in ("export control", "strike", "tariff",
                       "congestion", "halt", "shutdown", "tightens", "quota",
                       "firm on", "edged higher", "rally", "sharply higher"))
    benign = any(w in t for w in ("record", "sales rise", "announce", "new plant",
                 "guidance", "slip", "eased", "softer"))
    if ctype == "none" and has_signal and strong_words and not benign:
        ctype = "price"
    if ctype != "none" and not has_signal:
        ctype = "none"

    ev["material"], ev["commodity"], ev["controlType"] = mat, com, ctype
    return ev

EXTRACT_PROMPT = """Extract ONE supply-chain disruption event from the article.
Return STRICT JSON ONLY (no prose, no markdown) with EXACTLY these keys:
  "material": one of {mats} or ""
  "commodity": one of {comms} or ""   (must match the material; leave "" if unsure)
  "region": country name or ""
  "controlType": one of export-control|tariff|strike|port-congestion|price|disruption|none
  "confidence": number 0..1
If the article describes no supply disruption (e.g. sales figures, a new plant,
a company result), set controlType to "none" and leave material/commodity "".

Example:
Article: "China tightens export controls on neodymium magnets."
{{"material":"neodymium","commodity":"RARE-EARTH","region":"China","controlType":"export-control","confidence":0.9}}

ARTICLE:
{article}
"""


def _keyword_extract(text: str) -> dict:
    t = text.lower()
    material = next((v for k, v in MATERIAL_HINTS.items() if k in t), "")
    commodity = next((v for k, v in COMMODITY_HINTS.items() if k in t), "")
    region = next((code for name, code in REGION_NAMES.items() if name in t), "")
    # strong disruption signals override any benign-sounding words in the same article
    strong = {"export control": "export-control", "export controls": "export-control",
              "tariff": "tariff", "strike": "strike", "congestion": "port-congestion",
              "halt": "disruption", "shutdown": "disruption", "incident": "disruption",
              "quota": "export-control", "tightens": "export-control",
              "tightness": "disruption", "disrupt": "disruption"}
    ctype = next((v for k, v in strong.items() if k in t), None)
    if ctype is None:
        benign = any(w in t for w in ("record", "sales rise", "rise ", "announce",
                     "new plant", "new ", "guidance", "slip", "eased", "softer demand",
                     "expand", "commission"))
        rally = any(w in t for w in ("rally", "extends gains", "drives copper", "higher",
                    "firm on", "edged higher", "edged up", "ticked higher"))
        ctype = "price" if (rally and not benign) else "none"
    # a disruption with no material/commodity/region signal is noise -> none
    if ctype != "none" and not (material or commodity or region):
        ctype = "none"
    return {"material": material, "commodity": commodity, "region": region,
            "controlType": ctype, "confidence": 0.7 if ctype != "none" else 0.2,
            "source": "keyword-fallback"}


import os
# Demo speed switch: by default the deterministic keyword path is used so clicks are
# instant on screen. Set AUTONERVE_LIVE_LLM=1 to use the local Qwen model live
# (slower on CPU) — useful to prove the model is real during Q&A.
_USE_LIVE_LLM = os.environ.get("AUTONERVE_LIVE_LLM", "0") == "1"


def extract_event(article_text: str) -> dict:
    """Structured event from article text. LLM only if explicitly enabled, else fast keyword path."""
    if _USE_LIVE_LLM and llm.available():
        try:
            ev = llm.generate_json(EXTRACT_PROMPT.format(
                mats=MATERIALS, comms=COMMODITIES, article=article_text))
            ev = _validate(ev, article_text)   # ground raw LLM output against the BOM
            ev["source"] = "llm+validated"
            return ev
        except Exception as e:  # noqa: BLE001 — never break the demo on extraction
            fb = _keyword_extract(article_text)
            fb["source"] = f"keyword-fallback (llm error: {type(e).__name__})"
            return fb
    return _keyword_extract(article_text)


def run_article(article_id: str) -> dict:
    """Full thread from an article: extract -> engine.run_event -> decision."""
    art = next((a for a in ARTICLES if a["id"] == article_id), None)
    if not art:
        return {"error": f"unknown article {article_id}"}
    ev = extract_event(f"{art['headline']}. {art['body']}")
    if ev["controlType"] == "none":
        return {"article": art["id"], "headline": art["headline"], "event": ev,
                "result": "no disruption detected"}
    # precision: a material/commodity event propagates through those parts only;
    # region is used only when there is no material/commodity signal (e.g. port congestion)
    mat = ev.get("material") or None
    com = ev.get("commodity") or None
    reg = (ev.get("region") or None) if not (mat or com) else None
    result = engine.run_event(material=mat, region=reg, commodity=com, china_cap_pct=0.0)
    return {"article": art["id"], "headline": art["headline"], "event": ev, "result": result}


if __name__ == "__main__":
    for aid in ("ART-001", "ART-003", "ART-002", "ART-008"):
        r = run_article(aid)
        ev = r["event"]
        print(f"\n{aid}: {r['headline']}")
        print(f"  extracted: material={ev['material']!r} commodity={ev['commodity']!r} "
              f"region={ev['region']!r} type={ev['controlType']} via {ev['source']}")
        res = r.get("result")
        if isinstance(res, dict):
            print(f"  products hit: {res['affected']['affected_products']}")
            d = res["decision"]
            if d and d.get("feasible"):
                print(f"  mix: {[(m['supplier'], str(m['share_pct'])+'%') for m in d['mix']]} "
                      f"-> China {d['china_dependency_pct']}% at +{d['cost_delta_pct']}%")
        else:
            print(f"  {res}")
