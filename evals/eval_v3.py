import json, os, sys
from dotenv import load_dotenv
from groq import Groq
from agents.graph import make_app
from agents.rules import ROOT


def main(n):
    load_dotenv()
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden_v2.jsonl").read_text().splitlines()]
    cache_path = ROOT / "evals" / "extract_cache_v3.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    save = lambda: cache_path.write_text(json.dumps(cache, indent=2))
    app = make_app(Groq(api_key=os.environ["GROQ_API_KEY"]), cache, save)

    subset = golden[:: max(1, len(golden) // n)][:n]
    oks, fraud_oks, reviews, by_slice = [], [], 0, {}
    for g in subset:
        cid = g["claim_id"]
        raw = (ROOT / "data" / "claims_v2" / f"{cid}.txt").read_text()
        out = app.invoke({"claim_id": cid, "raw": raw})
        r, flags = out["result"], out.get("flags", [])
        ok = r["decision"] == g["expected_decision"] and int(r["payable_amount"]) == g["expected_payable"]
        oks.append(ok)
        fraud_oks.append(("amount_mismatch" in flags) == g["fraud_flag"])
        reviews += bool(flags)
        for key in (f"scenario={g['scenario']}", f"ref={g['ref_style']}", f"injection={g['injection']}"):
            by_slice.setdefault(key, []).append(ok)
        if not ok:
            print(f"  MISS {cid} [{g['scenario']}]: expected {g['expected_decision']}/{g['expected_payable']}, "
                  f"got {r['decision']}/{r['payable_amount']} ({r.get('reason')})")

    res = {"n": len(subset),
           "decision_and_payable_acc": round(sum(oks) / len(oks), 3),
           "fraud_flag_acc": round(sum(fraud_oks) / len(fraud_oks), 3),
           "human_review_rate": round(reviews / len(subset), 3),
           "by_slice": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_slice.items())}}
    before = json.loads((ROOT / "evals" / "results_v2_baseline.json").read_text())
    print("\nV3 AGENTS:", json.dumps(res, indent=2))
    print(f"\nBEFORE {before['decision_and_payable_acc']} -> AFTER {res['decision_and_payable_acc']}")
    (ROOT / "evals" / "results_v3_agents.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)