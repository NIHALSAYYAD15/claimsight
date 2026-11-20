import json, time
from agents.rules import ROOT

TRACE_FILE = ROOT / "data" / "traces.jsonl"
# Equivalent paid-API prices in USD per 1M tokens. The free tier costs $0, so check
# current rates on the Groq pricing page and update these.
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60


def log(event):
    event["ts"] = round(time.time(), 3)
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * PRICE_IN_PER_M + completion_tokens * PRICE_OUT_PER_M) / 1_000_000


def _pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def summarize():
    if not TRACE_FILE.exists():
        return {"requests": 0}
    ev = [json.loads(l) for l in TRACE_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    calls = [e for e in ev if e["type"] == "llm_call"]
    ok = [e for e in calls if e.get("ok")]
    reqs = [e for e in ev if e["type"] == "request"]
    hits = sum(e["type"] == "cache_hit" for e in ev)
    lat = [e["latency_ms"] for e in reqs]
    cold = [e["latency_ms"] for e in reqs if e.get("cached") is False]
    warm = [e["latency_ms"] for e in reqs if e.get("cached") is True]
    return {
        "requests": len(reqs),
        "llm_calls": len(calls),
        "llm_failure_rate": round(1 - len(ok) / len(calls), 3) if calls else 0,
        "cache_hit_rate": round(hits / (hits + len(calls)), 3) if (hits + len(calls)) else 0,
        "p50_latency_ms": _pct(lat, 0.5),
        "p95_latency_ms": _pct(lat, 0.95),
        "cold_p50_ms": _pct(cold, 0.5),
        "cold_p95_ms": _pct(cold, 0.95),
        "warm_p50_ms": _pct(warm, 0.5),
        "cost_per_llm_call_usd": round(sum(e["cost_usd"] for e in ok) / len(ok), 6) if ok else 0,
        "avg_prompt_tokens": round(sum(e["prompt_tokens"] for e in ok) / len(ok)) if ok else 0,
        "avg_completion_tokens": round(sum(e["completion_tokens"] for e in ok) / len(ok)) if ok else 0,
        "cost_per_request_usd": round(sum(e["cost_usd"] for e in ok) / len(reqs), 6) if reqs else 0,
        "human_review_rate": round(sum(e["human_review"] for e in reqs) / len(reqs), 3) if reqs else 0,
    }