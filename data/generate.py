import json, random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).parent
for d in (ROOT / "policies", ROOT / "claims", ROOT.parent / "evals"):
    d.mkdir(exist_ok=True)

NAMES = ["Aarav Patil", "Diya Kulkarni", "Rohan Deshmukh", "Sneha Joshi", "Vikram Shinde",
         "Ananya More", "Kabir Jadhav", "Isha Pawar", "Aditya Kale", "Meera Gaikwad"]
TYPES = {
    "health": {"ok": ["hospitalization for fever", "appendectomy", "fracture treatment", "dengue treatment"],
               "excl": ["cosmetic surgery", "dental whitening"], "waiting": 30},
    "motor": {"ok": ["collision damage", "windshield breakage", "flood damage", "theft"],
              "excl": ["racing accident", "driving without license"], "waiting": 0},
    "home": {"ok": ["fire damage", "water pipe burst", "burglary", "storm damage"],
             "excl": ["war damage", "wear and tear"], "waiting": 0},
}

def policy_text(p):
    excl = ", ".join(TYPES[p["type"]]["excl"])
    return (f"POLICY {p['id']} - {p['type'].upper()} INSURANCE\n"
            f"Policyholder: {p['holder']}\n"
            f"Clause 2.1 Policy period: {p['start']} to {p['end']}. Incidents outside this period are not covered.\n"
            f"Clause 3.1 Waiting period: Incidents within {p['waiting']} days of the start date are not payable.\n"
            f"Clause 4.1 Coverage limit: The maximum payable per claim is INR {p['limit']}.\n"
            f"Clause 4.2 Deductible: INR {p['deductible']} is deducted from every approved claim.\n"
            f"Clause 5.1 Exclusions: The following are not covered: {excl}.\n")

def decide(p, c):
    d = date.fromisoformat(c["incident_date"])
    if not p["start"] <= d <= p["end"]:
        return "DENIED", "2.1", 0
    if (d - p["start"]).days < p["waiting"]:
        return "DENIED", "3.1", 0
    if c["cause"] in TYPES[p["type"]]["excl"]:
        return "DENIED", "5.1", 0
    if c["invoice_total"] > p["limit"]:
        return "PARTIAL", "4.1", p["limit"] - p["deductible"]
    return "APPROVED", ("4.2" if p["deductible"] else "4.1"), max(c["invoice_total"] - p["deductible"], 0)

policies = []
for i in range(1, 31):
    t = random.choice(list(TYPES))
    start = date(2025, 1, 1) + timedelta(days=random.randint(0, 300))
    p = {"id": f"POL-{i:03d}", "holder": random.choice(NAMES), "type": t, "start": start,
         "end": start + timedelta(days=365), "limit": random.choice([200000, 500000, 1000000]),
         "deductible": random.choice([0, 5000, 10000]), "waiting": TYPES[t]["waiting"]}
    policies.append(p)
    (ROOT / "policies" / f"{p['id']}.txt").write_text(policy_text(p))

golden, n = [], 0
for p in policies:
    cfg = TYPES[p["type"]]
    for _ in range(5):
        n += 1
        scen = random.choice(["ok", "ok", "ok", "outside", "waiting", "excluded", "overlimit", "fraud"])
        cause = random.choice(cfg["ok"])
        incident = p["start"] + timedelta(days=random.randint(cfg["waiting"] + 1, 360))
        invoice = random.randint(10, 90) * 1000
        if scen == "outside":
            incident = p["end"] + timedelta(days=random.randint(5, 60))
        elif scen == "waiting" and cfg["waiting"]:
            incident = p["start"] + timedelta(days=random.randint(1, cfg["waiting"] - 1))
        elif scen == "excluded":
            cause = random.choice(cfg["excl"])
        elif scen == "overlimit":
            invoice = p["limit"] + random.randint(50, 300) * 1000
        claimed = invoice * random.randint(2, 4) if scen == "fraud" else invoice
        c = {"claim_id": f"CLM-{n:03d}", "policy_id": p["id"], "claimant": p["holder"],
             "incident_date": incident.isoformat(), "cause": cause,
             "invoice_total": invoice, "claimed_amount": claimed,
             "email": f"Hello, I am claiming INR {claimed} for {cause} on {incident}. Please process under policy {p['id']}."}
        (ROOT / "claims" / f"{c['claim_id']}.json").write_text(json.dumps(c, indent=2))
        decision, clause, payable = decide(p, c)
        golden.append({"claim_id": c["claim_id"], "policy_id": p["id"],
                       "question": f"Is claim {c['claim_id']} payable under {p['id']}, and how much?",
                       "expected_decision": decision, "expected_clause": clause,
                       "expected_payable": payable, "fraud_flag": claimed != invoice})

with open(ROOT.parent / "evals" / "golden_set.jsonl", "w") as f:
    for g in golden:
        f.write(json.dumps(g) + "\n")
print(f"Created {len(policies)} policies, {len(golden)} claims, golden set ready")