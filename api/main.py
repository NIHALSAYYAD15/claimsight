import hashlib, os, time
from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
from agents.graph import make_app

load_dotenv()
cache = {}  # in-memory extraction cache (identical claims are not re-sent to the LLM)
graph = make_app(Groq(api_key=os.environ["GROQ_API_KEY"]), cache, lambda: None, pause=0)
api = FastAPI(title="ClaimSight", version="0.1.0")


class ClaimIn(BaseModel):
    text: str


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/claims/analyze")
def analyze(body: ClaimIn):
    t0 = time.perf_counter()
    cid = hashlib.sha256(body.text.encode()).hexdigest()[:12]
    out = graph.invoke({"claim_id": cid, "raw": body.text})
    flags = out.get("flags", [])
    return {
        "claim_id": cid,
        "policy_id": out.get("policy_id"),
        "decision": out["result"]["decision"],
        "clause": out["result"]["clause"],
        "payable_amount": out["result"]["payable_amount"],
        "reason": out["result"].get("reason"),
        "flags": flags,
        "human_review_required": bool(flags),
        "extracted": out.get("extracted", {}),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }