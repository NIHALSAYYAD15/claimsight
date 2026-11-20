import json, os, re, sys, time
from dotenv import load_dotenv
from groq import Groq
from ingestion.build_index import ROOT
from retrieval.hybrid import HybridRetriever

load_dotenv()
MODEL = "openai/gpt-oss-120b"
CACHE = ROOT / "evals" / "llm_cache.json"

SYSTEM = """You are an insurance claims analyst. Use ONLY the policy clauses provided.
Check in this order:
1. incident_date outside the policy period (clause 2.1) -> DENIED, clause 2.1
2. incident within the waiting period days of the policy start date (clause 3.1) -> DENIED, clause 3.1
3. cause listed in exclusions (clause 5.1) -> DENIED, clause 5.1
4. invoice_total above the coverage limit -> PARTIAL, clause 4.1, payable = limit - deductible
5. otherwise APPROVED, payable = invoice_total - deductible (minimum 0), clause 4.2 if deductible > 0 else 4.1
Denied claims pay 0. Always use invoice_total for amounts, never claimed_amount.
Return ONLY a JSON object: {"decision": "APPROVED|PARTIAL|DENIED", "clause": "x.y", "payable_amount": 0, "reasoning": "one sentence"}"""


def load_claim(cid):
    return json.loads((ROOT / "data" / "claims" / f"{cid}.json").read_text())


def make_query(claim):
    return claim["email"] + " policy period waiting period coverage limit deductible exclusions"


def retrieval_eval(ret, golden):
    out = {}
    for mode in ("bm25", "vector", "hybrid"):
        hits = 0
        for g in golden:
            claim = load_claim(g["claim_id"])
            res = ret.search(make_query(claim), k=3, mode=mode, policy_id=claim["policy_id"])
            hits += any(h["clause"] == g["expected_clause"] for h in res)
        out[mode] = round(hits / len(golden), 3)
    return out


def ask_llm(client, claim, chunks):
    ctx = "\n".join(c["text"] for c in chunks)
    user = f"POLICY CLAUSES:\n{ctx}\n\nCLAIM:\n{json.dumps(claim, indent=2)}"
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            return json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
        except Exception as e:
            print("  retrying after error:", str(e)[:80])
            time.sleep(30)
    return {"decision": "ERROR", "clause": "", "payable_amount": -1}


def main(n):
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden_set.jsonl").read_text().splitlines()]
    ret = HybridRetriever()

    print(f"\nPolicy-context hit rate @5 over all {len(golden)} claims:")
    retrieval = retrieval_eval(ret, golden)
    print(retrieval)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    subset = golden[:: max(1, len(golden) // n)][:n]
    dec = clause = pay = 0
    for g in subset:
        cid = g["claim_id"]
        if cid not in cache:
            claim = load_claim(cid)
            #cache[cid] = ask_llm(client, claim, ret.search(make_query(claim), k=5, policy_id=claim["policy_id"]))
            cache[cid] = ask_llm(client, claim, ret.search(make_query(claim), k=5, policy_id=claim["policy_id"]))
            CACHE.write_text(json.dumps(cache, indent=2))
            time.sleep(10)  # stay inside the free-tier token limit
        p = cache[cid]
        ok_d = p.get("decision") == g["expected_decision"]
        ok_c = str(p.get("clause")) == g["expected_clause"]
        try:
            ok_p = int(float(p.get("payable_amount", -1))) == g["expected_payable"]
        except Exception:
            ok_p = False
        dec += ok_d; clause += ok_c; pay += ok_p
        if not ok_d:
            print(f"  MISS {cid}: expected {g['expected_decision']}/{g['expected_clause']}, got {p.get('decision')}/{p.get('clause')}")

    res = {"retrieval_hit_rate": retrieval, "n_llm_claims": len(subset),
           "decision_acc": round(dec / len(subset), 3),
           "clause_acc": round(clause / len(subset), 3), 
           "payable_acc": round(pay / len(subset), 3)}
    print("\nBASELINE:", json.dumps(res, indent=2))
    #(ROOT / "evals" / "results_baseline.json").write_text(json.dumps(res, indent=2))
    (ROOT / "evals" / "results_v1_metadata_filter.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)