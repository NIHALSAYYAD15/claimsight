import hashlib, os, time
from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
from agents import trace
from agents.graph import make_app
from pathlib import Path
from fastapi.responses import FileResponse

load_dotenv()
cache = {}  # in-memory extraction cache (identical claims are not re-sent to the LLM)
graph = make_app(Groq(api_key=os.environ["GROQ_API_KEY"]), cache, lambda: None, pause=0)
api = FastAPI(title="ClaimSight", version="0.1.0")

@api.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html")

class ClaimIn(BaseModel):
    text: str


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/claims/analyze")
def analyze(body: ClaimIn):
    t0 = time.perf_counter()
    cid = hashlib.sha256(body.text.encode()).hexdigest()[:12]
    was_cached = cid in cache
    out = graph.invoke({"claim_id": cid, "raw": body.text})
    flags = out.get("flags", [])
    latency = round((time.perf_counter() - t0) * 1000)
    trace.log({"type": "request", "claim_id": cid, "decision": out["result"]["decision"],
               "latency_ms": latency, "human_review": bool(flags), "cached": was_cached})
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
        "latency_ms": latency,
    }


@api.get("/metrics")
def metrics():
    return trace.summarize()