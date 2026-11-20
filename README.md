# ClaimSight

Multi-agent insurance claims analysis: reads a messy claim email plus bill, finds the policy, applies the policy rules, and flags fraud, prompt injection and incomplete data for human review.

## Architecture

```mermaid
flowchart LR
    A[Claim email + bill] --> B[Extract fields<br/>LLM via Groq]
    B --> C[Resolve policy]
    C -->|found| D[Rules engine<br/>deterministic Python]
    C -->|not found| H[NEEDS_REVIEW]
    D --> E[Verifier<br/>fraud / injection / incomplete flags]
    E --> F[Decision + clause + payable]
```

Design choice: the LLM only **extracts** structured fields from untrusted text. All decisions (dates, waiting periods, exclusions, limits, deductibles) come from deterministic code, so every outcome is auditable and an injected "approve this claim" instruction cannot change a decision. Built with LangGraph, FastAPI and Pydantic.

## Results (synthetic benchmark)

| Metric | Baseline (single LLM prompt + retrieval) | ClaimSight (extract then rules) |
|---|---|---|
| Decision + payable accuracy, 40 messy claims | 45% | 100% |
| Full 150-claim set | not measured | 150/150 |
| Prompt-injection claims handled correctly | 2/3 | 11/11 |
| Fraud flag accuracy | not measured | 100% |

| Serving metric | Value |
|---|---|
| Cold request latency p50 / p95 | 1,163 ms / 2,248 ms |
| Repeated claim latency p50 (extraction cache) | 2 ms |
| LLM failure rate | 0% over 15 calls |
| Tokens per call (in / out) | ~438 / ~317 |

Cost is $0 on Groq's free tier.

### Baseline lessons
Retrieval over near-identical policies was the first failure: policy-level hit rate was 5-19% until I filtered by policy ID. Even then, the LLM failed at date math, limit math and typo'd exclusions on messy inputs, which is why decisions moved to code.

## Evaluation and CI

A 150-claim golden set covers approved, denied, partial, boundary dates, fraud mismatches, injection text, and policy references by ID or by name. Every push runs the eval gate in GitHub Actions and fails the build if accuracy drops below 95%.

![Gate passing](docs/images/gate-passed.png)
![Gate catching a regression](docs/images/gate-failed.png)

## Demo

![Flagged claim](docs/images/ui-flagged.png)
![Unresolved policy](docs/images/ui-unresolved.png)

## Observability

`GET /metrics` reports request count, cold and warm latency, cache hit rate, LLM failure rate, token averages and human-review rate, from a JSONL trace log.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install -r requirements-api.txt
echo GROQ_API_KEY=your_key > .env
python data/generate.py
python data/generate_v2.py
python -m uvicorn api.main:api --port 8000   # UI at http://127.0.0.1:8000
python -m evals.ci_gate                      # offline eval gate, no API key needed
```

Or with Docker: `docker build -t claimsight .`

## Limitations

- **Synthetic data.** Labels come from the same rules the engine applies, so 100% shows that extraction and routing work, not that the rules are right for real policies.
- Clause text is templated; real policies are longer and less regular.
- Exclusion matching uses fuzzy string matching; a production version would use an LLM classifier.
- The cache is exact-match and in memory.
- Input is pasted text. PDF and scan parsing with OCR is not built yet.

## Repo layout

`agents/` LangGraph flow, rules, tracing · `api/` FastAPI and demo UI · `evals/` golden sets, baselines, CI gate · `data/` synthetic policy and claim generators · `ingestion/`, `retrieval/` baseline hybrid RAG (BM25 + vectors) used in the v1/v2 benchmarks