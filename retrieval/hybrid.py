import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from ingestion.build_index import load_chunks, MODEL_NAME, CHROMA_DIR


def tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.model = SentenceTransformer(MODEL_NAME)
        self.col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection("policies")
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])

    def _bm25(self, q, n, policy_id=None):
        scores = self.bm25.get_scores(tok(q))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        ids = [self.chunks[i]["id"] for i in order
               if policy_id is None or self.chunks[i]["policy_id"] == policy_id]
        return ids[:n]

    def _vector(self, q, n, policy_id=None):
        prefix = "Represent this sentence for searching relevant passages: "
        emb = self.model.encode([prefix + q], normalize_embeddings=True).tolist()
        kwargs = {"where": {"policy_id": policy_id}} if policy_id else {}
        return self.col.query(query_embeddings=emb, n_results=n, **kwargs)["ids"][0]

    def search(self, q, k=5, mode="hybrid", policy_id=None):
        if mode == "bm25":
            ids = self._bm25(q, k, policy_id)
        elif mode == "vector":
            ids = self._vector(q, k, policy_id)
        else:  # reciprocal rank fusion
            fused = {}
            for ranked in (self._bm25(q, 30, policy_id), self._vector(q, 30, policy_id)):
                for rank, cid in enumerate(ranked):
                    fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
            ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.by_id[i] for i in ids]