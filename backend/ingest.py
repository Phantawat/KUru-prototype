"""
ingest.py  —  Phase 2: Data pipeline for KUru prototype.

Steps:
  1. Chunk programs.json + tcas.json  (chunker.py)
  2. Embed each chunk via Gemini text-embedding-004
  3. Store vectors + metadata in ChromaDB  (local, no server)
  4. Build knowledge graph  (graph_builder.py)

Run:
  python ingest.py

Re-running is safe: drops and recreates the ChromaDB collection,
rebuilds the graph from scratch.
"""

import json
import time
import logging
from pathlib import Path

import chromadb
from google import genai
from google.genai import types as genai_types

import config
import chunker
import graph_builder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "ingest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Gemini embedder ───────────────────────────────────────────────────────────

class Embedder:
    """Thin wrapper around Gemini text-embedding-004 with retry + rate-limit."""

    RETRY_LIMIT = 3
    RETRY_SLEEP = 5     # seconds between retries

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        log.info(f"Embedder ready  model={config.EMBEDDING_MODEL}")

    def embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        for attempt in range(self.RETRY_LIMIT):
            try:
                result = []
                for text in texts:
                    response = self.client.models.embed_content(
                        model   = config.EMBEDDING_MODEL,
                        contents= text,
                        config  = genai_types.EmbedContentConfig(task_type=task_type),
                    )
                    result.append(response.embeddings[0].values)
                return result
            except Exception as exc:
                if attempt < self.RETRY_LIMIT - 1:
                    log.warning(f"  Embed retry {attempt+1}/{self.RETRY_LIMIT}: {exc}")
                    time.sleep(self.RETRY_SLEEP)
                else:
                    raise

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """Embed all chunks, returning vectors in same order."""
        all_vectors = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                log.info(f"  Embedding chunk {i+1}/{total}...")
            vec = self.embed_batch([chunk["text"]], config.EMBEDDING_TASK_TYPE)[0]
            all_vectors.append(vec)
            time.sleep(0.1)   # stay within free-tier rate limit (10 RPM)
        return all_vectors


# ── ChromaDB store ────────────────────────────────────────────────────────────

def setup_chroma() -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """Create (or recreate) the ChromaDB collection."""
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))

    # Drop existing collection so re-runs are idempotent
    existing = [c.name for c in client.list_collections()]
    if config.CHROMA_PLO_COLLECTION in existing:
        client.delete_collection(config.CHROMA_PLO_COLLECTION)
        log.info(f"  Dropped existing collection '{config.CHROMA_PLO_COLLECTION}'")

    collection = client.create_collection(
        name     = config.CHROMA_PLO_COLLECTION,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity
    )
    log.info(f"  Created collection '{config.CHROMA_PLO_COLLECTION}'")
    return client, collection


def store_chunks(
    collection: chromadb.Collection,
    chunks:     list[dict],
    vectors:    list[list[float]],
) -> None:
    """Insert chunks into ChromaDB in batches."""
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch_c = chunks[i : i + BATCH]
        batch_v = vectors[i : i + BATCH]

        collection.add(
            ids        = [c["chunk_id"]  for c in batch_c],
            embeddings = batch_v,
            documents  = [c["text"]      for c in batch_c],
            metadatas  = [
                {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
                for c in batch_c
            ],
        )
    log.info(f"  Stored {len(chunks)} chunks in ChromaDB")


# ── Smoke test after ingestion ────────────────────────────────────────────────

def smoke_test(embedder: Embedder, collection: chromadb.Collection) -> None:
    """
    Run 4 test queries (Thai + English) and verify sensible results.
    This is the core Phase 2 validation.
    """
    test_queries = [
        ("Thai PLO query",     "CPE สอนทักษะอะไรบ้าง",                 "CPE"),
        ("English PLO query",  "What skills does Computer Engineering teach?", "CPE"),
        ("Thai TCAS query",    "CPE รอบ Portfolio ต้องเตรียมอะไร",      "CPE"),
        ("Cross-lingual",      "machine learning artificial intelligence program", "CS"),
    ]

    log.info("\n── Smoke Test ──────────────────────────────────────")
    all_pass = True
    for label, query, expected_program in test_queries:
        vec = embedder.embed_batch([query], config.QUERY_TASK_TYPE)[0]
        results = collection.query(
            query_embeddings = [vec],
            n_results        = 3,
            include          = ["documents", "metadatas", "distances"],
        )
        top_docs  = results["documents"][0]
        top_metas = results["metadatas"][0]
        top_dists = results["distances"][0]

        hit = any(m.get("program_id") == expected_program for m in top_metas)
        status = "PASS ✓" if hit else "FAIL ✗"
        if not hit:
            all_pass = False

        log.info(f"\n  [{status}]  {label}")
        log.info(f"  Query: \"{query}\"")
        log.info(f"  Expected program: {expected_program}")
        for j, (doc, meta, dist) in enumerate(zip(top_docs, top_metas, top_dists)):
            sim = 1 - dist  # cosine distance → similarity
            prog = meta.get("program_id", "?")
            ctype = meta.get("chunk_type", "?")
            log.info(f"    #{j+1}  sim={sim:.3f}  program={prog}  type={ctype}")
            log.info(f"         \"{doc[:120]}...\"")

    log.info(f"\n── Smoke Test Result: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'} ──")
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    config.validate()
    log.info("═" * 60)
    log.info("KUru Prototype — Phase 2: Data Pipeline")
    log.info("═" * 60)

    # ── Step 1: Chunk ─────────────────────────────────────────────────────────
    log.info("\n[Step 1/4]  Chunking documents...")
    plo_chunks  = chunker.chunk_programs(config.DATA_DIR / "programs.json")
    tcas_chunks = chunker.chunk_tcas(config.DATA_DIR / "tcas.json")
    all_chunks  = plo_chunks + tcas_chunks
    log.info(f"  PLO chunks:   {len(plo_chunks)}")
    log.info(f"  TCAS chunks:  {len(tcas_chunks)}")
    log.info(f"  Total chunks: {len(all_chunks)}")

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    log.info("\n[Step 2/4]  Embedding chunks via Gemini text-embedding-004...")
    embedder = Embedder(config.GEMINI_API_KEY)
    t0 = time.time()
    vectors = embedder.embed_chunks(all_chunks)
    elapsed = time.time() - t0
    log.info(f"  Done in {elapsed:.1f}s  |  vector dim={len(vectors[0])}")

    # ── Step 3: Store in ChromaDB ─────────────────────────────────────────────
    log.info("\n[Step 3/4]  Storing in ChromaDB...")
    _, collection = setup_chroma()
    store_chunks(collection, all_chunks, vectors)
    log.info(f"  Collection count: {collection.count()} documents")

    # ── Step 4: Knowledge graph ───────────────────────────────────────────────
    log.info("\n[Step 4/4]  Building knowledge graph...")
    G = graph_builder.build_graph(
        config.DATA_DIR / "programs.json",
        config.DATA_DIR / "careers.json",
    )
    graph_builder.save_graph(G)
    log.info(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    log.info(f"  Saved → {config.GRAPH_PATH}")

    # ── Smoke test ────────────────────────────────────────────────────────────
    log.info("\n[Smoke Test]  Validating retrieval quality...")
    smoke_test(embedder, collection)

    log.info("\n═" * 60)
    log.info("Phase 2 complete. Ready for Phase 3 (RAG pipeline).")
    log.info("═" * 60)


if __name__ == "__main__":
    run()