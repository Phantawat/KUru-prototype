"""
config.py  —  Central configuration for KUru prototype
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "mock-data"
DB_DIR      = ROOT / "db"
LOG_DIR     = ROOT / "logs"
ENV_FILE    = ROOT / ".env"

# ── Load .env (silently if missing) ───────────────────────────────────────────
load_dotenv(ENV_FILE)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL       = "gemini-embedding-001"   # replaces text-embedding-004 (retired Jan 14 2026)
GENERATION_MODEL      = "gemini-2.5-flash"
EMBEDDING_TASK_TYPE   = "RETRIEVAL_DOCUMENT"   # for ingestion
QUERY_TASK_TYPE       = "RETRIEVAL_QUERY"       # for query-time embed

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH           = DB_DIR / "chroma"
CHROMA_PLO_COLLECTION = "kuru_plo_chunks"

# ── Graph ─────────────────────────────────────────────────────────────────────
GRAPH_PATH            = DB_DIR / "knowledge_graph.json"

# ── RAG ───────────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE        = 5        # chunks to retrieve per query
MAX_CHUNK_CHARS       = 600      # max characters per PLO chunk

# ── Recommendation ────────────────────────────────────────────────────────────
TOP_N_PROGRAMS        = 3        # programs to return
MIN_SIMILARITY_SCORE  = 0.3      # cosine sim threshold

# ── Interest → Skill cluster mapping matrix ───────────────────────────────────
# Maps a student's interest topic to weighted skill clusters.
# These weights define how strongly each interest maps to each skill.
# Row = interest topic, Col = skill cluster, Value = weight 0.0–1.0
INTEREST_SKILL_MATRIX = {
    "programming":      {"programming": 1.0, "software_engineering": 0.8, "algorithms_data_structures": 0.7, "testing_qa": 0.5},
    "ai_data":          {"machine_learning": 1.0, "data_engineering": 0.9, "mathematics_computation": 0.8, "programming": 0.7, "research_skills": 0.6},
    "design_ux":        {"ux_design": 1.0, "communication": 0.7, "critical_thinking": 0.6, "software_engineering": 0.4},
    "systems_hardware": {"hardware_design": 1.0, "embedded_systems": 0.9, "networking": 0.7, "systems_design": 0.6},
    "math_theory":      {"mathematics_computation": 1.0, "algorithms_data_structures": 0.9, "theoretical_cs": 0.9, "research_skills": 0.7},
    "business_product": {"project_management": 1.0, "communication": 0.9, "systems_design": 0.7, "critical_thinking": 0.8},
    "security_networks":{"cybersecurity": 1.0, "networking": 0.9, "systems_design": 0.7, "software_engineering": 0.5},
    "research_science": {"research_skills": 1.0, "mathematics_computation": 0.8, "machine_learning": 0.7, "technical_writing": 0.6},
}

def validate():
    """Raise if required config is missing."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to kuru_prototype/.env:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)