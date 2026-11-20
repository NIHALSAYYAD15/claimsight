import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_policies():
    out = {}
    for f in sorted((ROOT / "data" / "policies").glob("POL-*.txt")):
        t = f.read_text()
        g = lambda pat: re.search(pat, t).group(1)
        out[f.stem] = {
            "id": f.stem,
            "type": g(r"- (\w+) INSURANCE").lower(),
            "holder": g(r"Policyholder: (.+)"),
            "start": date.fromisoformat(g(r"period: (\d{4}-\d{2}-\d{2}) to")),
            "end": date.fromisoformat(g(r"to (\d{4}-\d{2}-\d{2})\.")),
            "waiting": int(g(r"within (\d+) days")),
            "limit": int(g(r"claim is INR (\d+)")),
            "deductible": int(g(r"Deductible: INR (\d+)")),
            "excl": g(r"not covered: (.+)\.").split(", "),
        }
    return out


def resolve_policy(policies, ex):
    try:
        pid = f"POL-{int(ex.get('policy_number')):03d}"
        if pid in policies:
            return pid
    except (TypeError, ValueError):
        pass
    holder = (ex.get("holder_name") or "").strip().lower()
    if not holder:
        return None
    cands = [p for p in policies.values() if p["holder"].lower() == holder]
    t = (ex.get("policy_type") or "").lower()
    if t:
        cands = [p for p in cands if p["type"] == t]
    s = ex.get("policy_start_date")
    if s:
        cands = [p for p in cands if p["start"].isoformat() == s]
    return cands[0]["id"] if len(cands) == 1 else None


def is_excluded(cause, exclusions, cutoff=0.85):
    c = (cause or "").lower().strip()
    return any(SequenceMatcher(None, c, e.lower()).ratio() >= cutoff for e in exclusions)


def apply_rules(p, ex):
    try:
        d = date.fromisoformat(ex["incident_date"])
        total = int(ex["bill_total"])
    except Exception:
        return {"decision": "NEEDS_REVIEW", "clause": "", "payable_amount": 0,
                "reason": "missing or invalid incident date or bill total"}
    if not p["start"] <= d <= p["end"]:
        return {"decision": "DENIED", "clause": "2.1", "payable_amount": 0, "reason": "outside policy period"}
    if (d - p["start"]).days < p["waiting"]:
        return {"decision": "DENIED", "clause": "3.1", "payable_amount": 0, "reason": "within waiting period"}
    if is_excluded(ex.get("cause"), p["excl"]):
        return {"decision": "DENIED", "clause": "5.1", "payable_amount": 0, "reason": "excluded cause"}
    if total > p["limit"]:
        return {"decision": "PARTIAL", "clause": "4.1", "payable_amount": p["limit"],
                "reason": "bill exceeds coverage limit"}
    return {"decision": "APPROVED", "clause": "4.2" if p["deductible"] else "4.1",
            "payable_amount": max(total - p["deductible"], 0), "reason": "covered"}