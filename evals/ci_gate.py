import json, sys
from agents.rules import ROOT, load_policies, resolve_policy, apply_rules

THRESHOLD = 0.95


def main():
    policies = load_policies()
    cache = json.loads((ROOT / "evals" / "extract_cache_v3.json").read_text())
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden_v2.jsonl").read_text().splitlines()]
    ok, missing = 0, 0
    for g in golden:
        ex = cache.get(g["claim_id"])
        if ex is None:
            missing += 1
            continue
        pid = resolve_policy(policies, ex)
        res = apply_rules(policies[pid], ex) if pid else {"decision": "NEEDS_REVIEW", "payable_amount": 0}
        hit = res["decision"] == g["expected_decision"] and int(res["payable_amount"]) == g["expected_payable"]
        ok += hit
        if not hit:
            print(f"MISS {g['claim_id']}: expected {g['expected_decision']}/{g['expected_payable']}, "
                  f"got {res['decision']}/{res['payable_amount']}")
    acc = ok / len(golden)
    print(f"accuracy={acc:.3f} ({ok}/{len(golden)}), missing cache entries={missing}, threshold={THRESHOLD}")
    if missing or acc < THRESHOLD:
        print("EVAL GATE FAILED")
        sys.exit(1)
    print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()