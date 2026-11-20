import json, os, re, sys, time
from dotenv import load_dotenv
from groq import Groq
from ingestion.build_index import ROOT
from retrieval.hybrid import HybridRetriever
from evals.baseline import SYSTEM, MODEL

load_dotenv()
CACHE = ROOT / "evals" / "llm_cache_v2.json"
SYS2 = (SYSTEM.replace("incident_date", "incident date")
        .replace("invoice_total", "bill TOTAL")
        .replace("claimed_amount", "amount requested in the email"))


def ask(client, raw, chunks):
    ctx = "\n".join(c["text"] for c in chunks)
    user = f"POLICY CLAUSES:\n{ctx}\n\nCLAIM TEXT:\n{raw}"
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[{"role": "system", "content": SYS2}, {"role": "user", "content": user}])
            return json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
        except Exception as e:
            print("  retrying after error:", str(e)[:80])
            time.sleep(30)
    return {"decision": "ERROR", "clause": "", "payable_amount": -1}


def main(n):
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden_v2.jsonl").read_text().splitlines()]
    ret = HybridRetriever()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    subset = golden[:: max(1, len(golden) // n)][:n]
    ok_all, by_slice = [], {}
    for g in subset:
        cid = g["claim_id"]
        raw = (ROOT / "data" / "claims_v2" / f"{cid}.txt").read_text()
        if cid not in cache:
            m = re.search(r"POL-\d{3}", raw)
            chunks = ret.search(raw, k=5, policy_id=m.group(0) if m else None)
            cache[cid] = ask(client, raw, chunks)
            CACHE.write_text(json.dumps(cache, indent=2))
            time.sleep(10)
        p = cache[cid]
        try:
            ok_p = int(float(p.get("payable_amount", -1))) == g["expected_payable"]
        except Exception:
            ok_p = False
        ok = p.get("decision") == g["expected_decision"] and ok_p
        ok_all.append(ok)
        for key in (f"scenario={g['scenario']}", f"ref={g['ref_style']}", f"injection={g['injection']}"):
            by_slice.setdefault(key, []).append(ok)
        if not ok:
            print(f"  MISS {cid} [{g['scenario']}]: expected {g['expected_decision']}/{g['expected_payable']}, "
                  f"got {p.get('decision')}/{p.get('payable_amount')}")
    res = {"n": len(subset), "decision_and_payable_acc": round(sum(ok_all) / len(ok_all), 3),
           "by_slice": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_slice.items())}}
    print("\nBASELINE V2:", json.dumps(res, indent=2))
    (ROOT / "evals" / "results_v2_baseline.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)