"""
AutoNerve demo — run the full live thread on the article corpus.

    python demo.py

No model or server needed: if Qwen weights are absent it uses the deterministic
keyword fallback, so this runs out of the box after `pip install -r requirements.txt`.
"""
import extraction, llm


def line(c="-"):
    print(c * 78)


def show(article_id):
    r = extraction.run_article(article_id)
    ev = r["event"]
    print(f"\n[{r['article']}] {r['headline']}")
    print(f"  extracted : material={ev['material']!r}  commodity={ev['commodity']!r}  "
          f"region={ev['region']!r}  type={ev['controlType']}  (via {ev['source']})")
    res = r.get("result")
    if not isinstance(res, dict):
        print(f"  result    : {res}")
        return
    a = res["affected"]
    print(f"  affected  : products {a['affected_products']}  | max risk {a['max_risk']}")
    if res.get("exposure"):
        e = res["exposure"]
        print(f"  exposure  : Rs {e['exposure_cr']} Cr on {e['qty_at_risk']:,} units "
              f"(decision part: {res['lead_part']})")
    d = res.get("decision")
    if d and d.get("feasible"):
        mix = ", ".join(f"{m['supplier']} {m['share_pct']}%" for m in d["mix"])
        print(f"  decision  : {mix}")
        print(f"              China dependency -> {d['china_dependency_pct']}%  "
              f"at +{d['cost_delta_pct']}% cost")
        if d.get("relaxed_to_min_china"):
            print(f"              note: {d['note']}")


if __name__ == "__main__":
    line("=")
    print(" AutoNerve — live decision thread  (article -> extract -> propagate -> "
          "exposure -> optimize)")
    print(f" LLM model present: {llm.available()}  "
          f"({'Qwen path' if llm.available() else 'keyword fallback'})")
    line("=")
    for aid in ("ART-001", "ART-003", "ART-002", "ART-005", "ART-008"):
        show(aid)
    line()
    print("Hero events fire (neodymium precise to its 2 EVs), neutral articles stay silent.")
    print("Start the API instead with:  uvicorn main:app --reload   then POST /article/ART-001")
