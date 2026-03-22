"""
recommendation_pipeline.py  —  Phase 3b: Recommendation pipeline for KUru.

Given a student's interest profile (topic → weight dict), this module:
  1. Maps interests → weighted skill cluster vector (via INTEREST_SKILL_MATRIX)
  2. Queries ChromaDB with that vector to find semantically matching program chunks
  3. Traverses the knowledge graph to get PLO → Skill → Career chains
  4. Computes a composite fit score for each program
  5. Calls Gemini to generate a plain-language explanation of the top matches

The output is a ranked list of programs with explanations the student can
actually understand — not just scores.
"""

import json
import logging
import time
from dataclasses import dataclass, field

import chromadb
import networkx as nx
from google import genai
from google.genai import types as genai_types

import config
import graph_builder

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ProgramMatch:
    program_id:       str
    program_name_th:  str
    program_name_en:  str
    fit_score:        float          # composite 0–1
    vector_score:     float          # pgvector similarity component
    graph_score:      float          # knowledge graph coverage component
    matched_skills:   list[str]      # skill clusters student profile matches
    top_careers:      list[dict]     # [{title_th, coverage_score, salary_entry}]
    plo_highlights:   list[str]      # top relevant PLO text snippets (Thai)
    explanation:      str            # Gemini-generated plain-language explanation


@dataclass
class RecommendationResponse:
    interests:        dict[str, float]   # raw interest input
    skill_vector:     dict[str, float]   # computed skill cluster weights
    programs:         list[ProgramMatch]
    generation_ms:    int

    def pretty(self) -> str:
        lines = [
            "ผลการแนะนำหลักสูตร / Program Recommendations",
            "─" * 60,
            f"Interest profile: {self.interests}",
            f"Mapped skill clusters: {list(self.skill_vector.keys())[:6]}...",
            "",
        ]
        for i, p in enumerate(self.programs, 1):
            lines += [
                f"#{i}  {p.program_name_th} ({p.program_id})",
                f"    Fit score:  {p.fit_score:.2f}  "
                f"(vector={p.vector_score:.2f}, graph={p.graph_score:.2f})",
                f"    Matched skills: {', '.join(p.matched_skills[:4])}",
                f"    Top careers: "
                + ", ".join(c["title_th"] for c in p.top_careers[:3]),
                "",
                f"    {p.explanation}",
                "",
                "─" * 60,
            ]
        lines.append(f"⏱  Generation {self.generation_ms}ms")
        return "\n".join(lines)


# ── Interest → Skill vector ───────────────────────────────────────────────────

def compute_skill_vector(interests: dict[str, float]) -> dict[str, float]:
    """
    Map a student's interest profile to a weighted skill cluster vector.

    interests: {topic: weight}  e.g. {"programming": 0.9, "ai_data": 0.8}
    returns:   {skill_cluster: weight}  e.g. {"programming": 0.9, "machine_learning": 0.72}

    The matrix in config.py defines how strongly each interest maps to each skill.
    If a topic is not in the matrix it is ignored (unknown interest).
    """
    skill_weights: dict[str, float] = {}

    for topic, interest_weight in interests.items():
        if topic not in config.INTEREST_SKILL_MATRIX:
            log.debug(f"  Unknown interest topic '{topic}' — skipping")
            continue
        for skill, matrix_weight in config.INTEREST_SKILL_MATRIX[topic].items():
            combined = interest_weight * matrix_weight
            # Take max across all topic mappings for the same skill
            skill_weights[skill] = max(skill_weights.get(skill, 0), combined)

    # Normalise so max weight = 1.0
    if skill_weights:
        max_w = max(skill_weights.values())
        if max_w > 0:
            skill_weights = {k: round(v / max_w, 3) for k, v in skill_weights.items()}

    return skill_weights


def skill_vector_to_query_text(skill_vector: dict[str, float]) -> str:
    """
    Convert a skill vector into a natural-language query string for ChromaDB.
    We weight the terms by repeating high-weight skills more — a simple but
    effective way to bias the embedding toward stronger interests.
    """
    terms = []
    for skill, weight in sorted(skill_vector.items(), key=lambda x: -x[1]):
        readable = skill.replace("_", " ")
        if weight >= 0.8:
            terms.extend([readable] * 3)    # repeat 3× for strong interests
        elif weight >= 0.5:
            terms.extend([readable] * 2)
        else:
            terms.append(readable)
    return " ".join(terms)


# ── Vector-based program retrieval ────────────────────────────────────────────

def retrieve_program_candidates(
    skill_vector: dict[str, float],
    collection:   chromadb.Collection,
) -> dict[str, float]:
    """
    Query ChromaDB with the skill query text to find matching program chunks.
    Returns {program_id: max_similarity} across all retrieved chunks per program.
    """
    query_text = skill_vector_to_query_text(skill_vector)
    log.debug(f"  Vector query: \"{query_text[:80]}...\"")

    # Retrieve more than TOP_N so we have enough to rank
    n_results = min(config.TOP_K_RETRIEVE * 4, collection.count())
    results = collection.query(
        query_texts = [query_text],
        n_results   = n_results,
        include     = ["metadatas", "distances"],
    )

    # Aggregate: keep highest similarity per program
    program_scores: dict[str, float] = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        pid  = meta.get("program_id", "")
        sim  = 1.0 - dist
        if sim < config.MIN_SIMILARITY_SCORE:
            continue
        if pid not in program_scores or sim > program_scores[pid]:
            program_scores[pid] = round(sim, 4)

    return program_scores


# ── Graph-based program scoring ───────────────────────────────────────────────

def score_program_by_graph(
    G:            nx.DiGraph,
    program_id:   str,
    skill_vector: dict[str, float],
) -> tuple[float, list[str]]:
    """
    Score a program using graph traversal:
      - Walk program → PLO → skill_cluster
      - For each skill the program develops, look up the student's interest weight
      - Score = weighted sum of matched skills / total possible weight

    Returns (graph_score 0–1, list of matched skill names).
    """
    program_skills = graph_builder.get_program_skills(G, program_id)
    # program_skills = {skill_cluster: plo_count}

    total_interest = sum(skill_vector.values())
    if total_interest == 0:
        return 0.0, []

    matched_weight = 0.0
    matched_skills = []

    for skill, plo_count in program_skills.items():
        if skill in skill_vector:
            interest_w = skill_vector[skill]
            depth_bonus = min(plo_count / 3, 1.0)  # more PLOs = deeper coverage
            matched_weight += interest_w * (1 + depth_bonus * 0.2)
            matched_skills.append(skill)

    graph_score = min(matched_weight / total_interest, 1.0)
    return round(graph_score, 4), matched_skills


# ── PLO highlights ────────────────────────────────────────────────────────────

def get_plo_highlights(
    G:           nx.DiGraph,
    program_id:  str,
    skill_vector: dict[str, float],
    top_n: int = 3,
) -> list[str]:
    """
    Return the top PLO text snippets most relevant to the student's interests.
    Used to give the explanation generator concrete curriculum evidence.
    """
    highlights = []
    for plo_id in G.successors(program_id):
        if G.nodes[plo_id].get("type") != "plo":
            continue
        plo_data = G.nodes[plo_id]
        plo_skills = [
            sc for sc in G.successors(plo_id)
            if G.nodes[sc].get("type") == "skill_cluster"
        ]
        # Score this PLO by how many of its skills match the student's interests
        relevance = sum(skill_vector.get(sc, 0) for sc in plo_skills)
        if relevance > 0:
            text_th = plo_data.get("text_th", "")[:150]
            highlights.append((relevance, f"PLO{plo_data.get('plo_number')}: {text_th}..."))

    highlights.sort(reverse=True)
    return [h[1] for h in highlights[:top_n]]


# ── Explanation generation ────────────────────────────────────────────────────

EXPLANATION_SYSTEM = """คุณคือ KUru ผู้ช่วยแนะนำหลักสูตรของมหาวิทยาลัยเกษตรศาสตร์
Generate a SHORT (3-4 sentences), plain-language explanation of why a program
matches a student's interests. Write in Thai. Be specific — mention actual skills
from the PLOs and actual careers from the graph data. Do NOT be generic."""


def generate_explanation(
    client:         genai.Client,
    program_match:  dict,
    interests:      dict[str, float],
) -> str:
    """Generate a plain-language explanation for one program match."""
    prompt = f"""{EXPLANATION_SYSTEM}

Student interests: {interests}
Program: {program_match['name_th']} ({program_match['id']})
Fit score: {program_match['fit_score']:.2f}
Skills this program develops that match the student: {program_match['matched_skills']}
Top careers this program leads to: {program_match['top_careers']}
Relevant PLO excerpts:
{chr(10).join('  - ' + h for h in program_match['plo_highlights'])}

Write a 3-4 sentence explanation in Thai for why this program suits this student.
Be concrete and mention specific skills and careers."""

    response = client.models.generate_content(
        model    = config.GENERATION_MODEL,
        contents = prompt,
        config   = genai_types.GenerateContentConfig(
            temperature       = 0.4,
            max_output_tokens = 256,
        ),
    )
    return response.text.strip()


# ── Recommendation Pipeline ───────────────────────────────────────────────────

class RecommendationPipeline:

    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

        chroma = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        self.collection = chroma.get_collection(config.CHROMA_PLO_COLLECTION)

        self.G = graph_builder.load_graph(config.GRAPH_PATH)

        log.info(f"RecommendationPipeline ready  "
                 f"graph=({self.G.number_of_nodes()}N, {self.G.number_of_edges()}E)  "
                 f"collection={self.collection.count()} chunks")

    def recommend(
        self,
        interests:  dict[str, float],
        top_n:      int = config.TOP_N_PROGRAMS,
    ) -> RecommendationResponse:
        """
        Return top-N program recommendations for a student's interest profile.

        interests: {topic: weight_0_to_1}
          Valid topics: programming, ai_data, design_ux, systems_hardware,
                        math_theory, business_product, security_networks,
                        research_science
        """
        t_start = time.monotonic()

        # ── Step 1: map interests → skill vector ──────────────────────────────
        skill_vector = compute_skill_vector(interests)
        log.info(f"Skill vector: {skill_vector}")

        # ── Step 2: vector retrieval → per-program similarity scores ─────────
        vector_scores = retrieve_program_candidates(skill_vector, self.collection)
        log.info(f"Vector scores: {vector_scores}")

        # ── Step 3: graph traversal → per-program fit scores ─────────────────
        program_ids = [
            n for n, d in self.G.nodes(data=True)
            if d.get("type") == "program"
        ]

        matches = []
        for pid in program_ids:
            prog_data = self.G.nodes[pid]

            # Graph score
            graph_score, matched_skills = score_program_by_graph(
                self.G, pid, skill_vector
            )

            # Vector score (0 if program didn't appear in vector retrieval)
            vector_score = vector_scores.get(pid, 0.0)

            # Composite: 60% graph (structured knowledge) + 40% vector (semantic)
            fit_score = round(0.6 * graph_score + 0.4 * vector_score, 4)

            if fit_score < 0.1:
                continue

            # Get career paths from graph
            top_careers = graph_builder.get_careers_for_program(self.G, pid)[:3]

            # Get PLO highlights relevant to student's interests
            plo_highlights = get_plo_highlights(self.G, pid, skill_vector)

            matches.append({
                "id":             pid,
                "name_th":        prog_data.get("name_th", ""),
                "name_en":        prog_data.get("name_en", ""),
                "fit_score":      fit_score,
                "vector_score":   vector_score,
                "graph_score":    graph_score,
                "matched_skills": matched_skills,
                "top_careers":    [
                    {"title_th": c["title_th"], "coverage": c["coverage_score"],
                     "salary_entry": c["salary_entry"]}
                    for c in top_careers
                ],
                "plo_highlights": plo_highlights,
            })

        # ── Step 4: rank and take top-N ───────────────────────────────────────
        matches.sort(key=lambda x: x["fit_score"], reverse=True)
        top_matches = matches[:top_n]

        # ── Step 5: generate explanations ────────────────────────────────────
        program_results = []
        for m in top_matches:
            log.info(f"  Generating explanation for {m['id']} (fit={m['fit_score']:.2f})")
            explanation = generate_explanation(self.client, m, interests)
            program_results.append(ProgramMatch(
                program_id      = m["id"],
                program_name_th = m["name_th"],
                program_name_en = m["name_en"],
                fit_score       = m["fit_score"],
                vector_score    = m["vector_score"],
                graph_score     = m["graph_score"],
                matched_skills  = m["matched_skills"],
                top_careers     = m["top_careers"],
                plo_highlights  = m["plo_highlights"],
                explanation     = explanation,
            ))

        gen_ms = int((time.monotonic() - t_start) * 1000)
        log.info(f"Recommendation complete  top={top_n}  total_ms={gen_ms}")

        return RecommendationResponse(
            interests     = interests,
            skill_vector  = skill_vector,
            programs      = program_results,
            generation_ms = gen_ms,
        )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    config.validate()

    pipeline = RecommendationPipeline()

    test_profiles = [
        {
            "label": "AI & programming heavy",
            "interests": {"ai_data": 0.9, "programming": 0.8, "math_theory": 0.6},
        },
        {
            "label": "Design & product focused",
            "interests": {"design_ux": 0.9, "business_product": 0.7, "programming": 0.5},
        },
        {
            "label": "Balanced — software + systems",
            "interests": {"programming": 0.8, "systems_hardware": 0.7, "security_networks": 0.5},
        },
    ]

    print("\n" + "═" * 70)
    print("KUru Recommendation Pipeline — Phase 3 Test")
    print("═" * 70)

    for profile in test_profiles:
        print(f"\n[{profile['label']}]")
        result = pipeline.recommend(profile["interests"])
        print(result.pretty())
