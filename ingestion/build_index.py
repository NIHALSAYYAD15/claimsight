import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = str(ROOT / "data" / "chroma")


def load_chunks():
    """One chunk per policy clause, with the policy header for context."""
    chunks = []
    for f in sorted((ROOT / "data" / "policies").glob("*.txt")):
        lines = f.read_text().splitlines()
        header = lines[0]
        for line in lines:
            m = re.match(r"Clause (\d+\.\d+)", line)
            if m:
                chunks.append({"id": f"{f.stem}-{m.group(1)}", "policy_id": f.stem,
                               "clause": m.group(1), "text": f"{header} | {line}"})
    return chunks


def build():
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("policies")
    except Exception:
        pass
    col = client.create_collection("policies", metadata={"hnsw:space": "cosine"})
    embs = model.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
    col.add(ids=[c["id"] for c in chunks], embeddings=embs,
            documents=[c["text"] for c in chunks],
            metadatas=[{"policy_id": c["policy_id"], "clause": c["clause"]} for c in chunks])
    print(f"Indexed {len(chunks)} clause chunks")


if __name__ == "__main__":
    build()