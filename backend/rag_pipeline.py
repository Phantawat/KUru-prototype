"""
rag_pipeline.py  —  Phase 3a: RAG pipeline for KUru prototype.

Given a student question (Thai or English), this module:
  1. Embeds the query with gemini-embedding-001
  2. Retrieves top-k semantically similar chunks from ChromaDB
  3. Assembles a grounded prompt with strict citation instructions
  4. Calls Gemini 2.0 Flash to generate the answer
  5. Returns a structured response with the answer + source citations

The pipeline never answers from general knowledge —
every factual claim must be grounded in retrieved chunks.
"""

import logging
import time
from dataclasses import dataclass, field

import chromadb
from google import genai
from google.genai import types as genai_types

import config

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:       str
    text:           str
    program_id:     str
    program_name_th: str
    chunk_type:     str
    plo_id:         str
    plo_number:     int
    domain:         str
    language:       str
    similarity:     float   # 0–1, higher = more similar


@dataclass
class RAGResponse:
    question:       str
    answer:         str
    citations:      list[dict]          # [{chunk_id, program_id, plo_id, similarity}]
    retrieved:      list[RetrievedChunk]
    model_used:     str
    retrieval_ms:   int
    generation_ms:  int

    def pretty(self) -> str:
        """Human-readable format for the chat interface."""
        lines = [
            f"คำตอบ / Answer",
            "─" * 60,
            self.answer,
            "",
            "แหล่งอ้างอิง / Sources",
            "─" * 60,
        ]
        seen = set()
        for c in self.citations:
            key = (c["program_id"], c.get("plo_id", ""))
            if key in seen:
                continue
            seen.add(key)
            prog = c["program_id"]
            plo  = c.get("plo_id", "")
            sim  = c.get("similarity", 0)
            ctype = c.get("chunk_type", "")
            if plo:
                lines.append(f"  • {prog} — {plo}  (similarity: {sim:.2f})")
            else:
                lines.append(f"  • {prog} — {ctype}  (similarity: {sim:.2f})")
        lines.append(f"\n⏱  Retrieval {self.retrieval_ms}ms  |  Generation {self.generation_ms}ms")
        return "\n".join(lines)


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """คุณคือ KUru ผู้ช่วยแนะนำหลักสูตรของมหาวิทยาลัยเกษตรศาสตร์
You are KUru, a curriculum advisor for Kasetsart University.

CRITICAL RULES — follow these without exception:
1. Answer ONLY using information from the provided context sections below.
2. If the context does not contain enough information to answer, say so clearly.
   Do NOT invent PLO details, score thresholds, or program information.
3. Always cite the source by mentioning the program ID and PLO number, e.g. "(CPE-PLO3)"
4. You may answer in Thai or English — match the language of the question.
5. Be specific and concrete. Students need actionable information, not vague summaries.
6. For TCAS questions: always state exact round numbers, score weights, and deadlines
   if the context contains them.
"""

def _build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the grounded prompt from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Context {i}] source={chunk.chunk_id}  program={chunk.program_id}"
        if chunk.plo_id:
            header += f"  plo={chunk.plo_id}"
        header += f"  type={chunk.chunk_type}  similarity={chunk.similarity:.3f}"
        context_parts.append(f"{header}\n{chunk.text}")

    context_block = "\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

━━━ RETRIEVED CONTEXT ━━━
{context_block}
━━━ END CONTEXT ━━━

Student question: {question}

Answer (cite sources using their chunk IDs or PLO numbers):"""


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):
        # Gemini client
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

        # ChromaDB
        chroma = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        self.collection = chroma.get_collection(config.CHROMA_PLO_COLLECTION)
        log.info(f"RAGPipeline ready  collection={config.CHROMA_PLO_COLLECTION} "
                 f"({self.collection.count()} chunks)")

    # ── Step 1: embed query ───────────────────────────────────────────────────

    def _embed_query(self, question: str) -> list[float]:
        response = self.client.models.embed_content(
            model    = config.EMBEDDING_MODEL,
            contents = question,
            config   = genai_types.EmbedContentConfig(
                task_type = config.QUERY_TASK_TYPE,
            ),
        )
        return response.embeddings[0].values

    # ── Step 2: retrieve chunks ───────────────────────────────────────────────

    def _retrieve(
        self,
        question:    str,
        query_vec:   list[float],
        program_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Query ChromaDB for top-k chunks.
        Optional program_filter restricts to a single program (e.g. "CPE").
        """
        where = {"program_id": program_filter} if program_filter else None

        results = self.collection.query(
            query_embeddings = [query_vec],
            n_results        = config.TOP_K_RETRIEVE,
            include          = ["documents", "metadatas", "distances"],
            where            = where,
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist   # cosine distance → similarity
            if similarity < config.MIN_SIMILARITY_SCORE:
                continue
            chunks.append(RetrievedChunk(
                chunk_id        = meta.get("chunk_id", ""),
                text            = doc,
                program_id      = meta.get("program_id", ""),
                program_name_th = meta.get("program_name_th", ""),
                chunk_type      = meta.get("chunk_type", ""),
                plo_id          = meta.get("plo_id", ""),
                plo_number      = int(meta.get("plo_number", 0)),
                domain          = meta.get("domain", ""),
                language        = meta.get("language", ""),
                similarity      = round(similarity, 4),
            ))

        log.debug(f"  Retrieved {len(chunks)} chunks above threshold "
                  f"(min_sim={config.MIN_SIMILARITY_SCORE})")
        return chunks

    # ── Step 3: generate answer ───────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model    = config.GENERATION_MODEL,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature      = 0.2,   # low = more faithful to context
                max_output_tokens= 1024,
            ),
        )
        return response.text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        question:       str,
        program_filter: str | None = None,
    ) -> RAGResponse:
        """
        Full RAG loop. Returns a RAGResponse with answer + citations.

        Args:
            question:       The student's question (Thai or English).
            program_filter: If set, restrict retrieval to one program ID.
        """
        # Step 1 — embed
        t0 = time.monotonic()
        query_vec = self._embed_query(question)
        retrieval_start = time.monotonic()

        # Step 2 — retrieve
        chunks = self._retrieve(question, query_vec, program_filter)
        retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)

        if not chunks:
            return RAGResponse(
                question      = question,
                answer        = (
                    "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลหลักสูตร\n"
                    "Sorry, no relevant curriculum information was found for this query."
                ),
                citations     = [],
                retrieved     = [],
                model_used    = config.GENERATION_MODEL,
                retrieval_ms  = retrieval_ms,
                generation_ms = 0,
            )

        # Step 3 — build prompt and generate
        prompt = _build_rag_prompt(question, chunks)
        gen_start = time.monotonic()
        answer = self._generate(prompt)
        generation_ms = int((time.monotonic() - gen_start) * 1000)

        citations = [
            {
                "chunk_id":   c.chunk_id,
                "program_id": c.program_id,
                "plo_id":     c.plo_id,
                "chunk_type": c.chunk_type,
                "similarity": c.similarity,
            }
            for c in chunks
        ]

        log.info(f"RAG  retrieval={retrieval_ms}ms  generation={generation_ms}ms  "
                 f"chunks={len(chunks)}  q=\"{question[:60]}\"")

        return RAGResponse(
            question      = question,
            answer        = answer,
            citations     = citations,
            retrieved     = chunks,
            model_used    = config.GENERATION_MODEL,
            retrieval_ms  = retrieval_ms,
            generation_ms = generation_ms,
        )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    config.validate()

    pipeline = RAGPipeline()

    test_questions = [
        ("Thai  — PLO skills",    "หลักสูตร CPE สอนทักษะอะไรบ้าง"),
        ("English — PLO skills",  "What programming skills does Computer Engineering teach?"),
        ("Thai  — TCAS round 1",  "CPE รอบ Portfolio ต้องเตรียม portfolio อะไรบ้าง"),
        ("Thai  — TCAS scores",   "คะแนนที่ใช้สมัคร CS รอบ 3 มีอะไรบ้าง"),
        ("Cross-lingual",         "What careers can I pursue after studying CS at KU?"),
        ("Out-of-scope hallucination test",
         "CPE มีวิชา React และ Next.js ในหลักสูตรไหม"),  # should say: not in context
    ]

    print("\n" + "═" * 70)
    print("KUru RAG Pipeline — Phase 3 Test")
    print("═" * 70)

    for label, question in test_questions:
        print(f"\n[{label}]")
        print(f"Q: {question}")
        resp = pipeline.ask(question)
        print(resp.pretty())
        print()
