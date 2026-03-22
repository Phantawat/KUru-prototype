# KUru — Intelligent PLO-to-Career Navigator

**Proof-of-concept prototype · Kasetsart University · Senior Project 2568**

KUru is an AI-powered academic pathway advisor that helps pre-admission students explore programs at Kasetsart University. It connects student interests to real program learning outcomes (PLOs from มคอ.2), TCAS admission pathways, and graduate career data — all through a conversational interface in Thai and English.

---

## What this prototype validates

This is a **Phase 1–5 proof-of-concept** built before the full production system. It proves that:

- A RAG pipeline over Thai academic PDF text (มคอ.2) can answer curriculum questions accurately without hallucinating
- A knowledge graph (`PLO → Skill → Career`) combined with vector similarity can rank KU programs against a student's interest profile
- Cross-lingual retrieval works — Thai queries correctly retrieve English PLO chunks and vice versa
- All three data sources (มคอ.2, TCAS data, graduation outcomes) can be ingested, indexed, and queried end-to-end

---

## Repository structure

```
kuru-prototype/
│
├── backend/                     # Python backend
│   ├── data/
│   │   ├── programs.json        # 3 KU programs with full Thai PLOs (มคอ.2 mock)
│   │   ├── careers.json         # 5 O*NET-based careers with Thai descriptions
│   │   └── tcas.json            # TCAS admission data — all rounds, all programs
│   │
│   ├── db/                      # Generated — created by ingest.py (git-ignored)
│   │   ├── chroma/              # ChromaDB vector store (87 embedded chunks)
│   │   └── knowledge_graph.json # NetworkX graph: 54 nodes, 145 edges
│   │
│   ├── logs/                    # Generated — runtime logs (git-ignored)
│   │
│   ├── config.py                # Central config: models, paths, interest matrix
│   ├── chunker.py               # Splits programs.json + tcas.json into chunks
│   ├── graph_builder.py         # Builds PLO → Skill → Career knowledge graph
│   ├── ingest.py                # Phase 2: embed chunks → ChromaDB + build graph
│   ├── rag_pipeline.py          # Phase 3a: RAG pipeline (embed → retrieve → generate)
│   ├── recommendation_pipeline.py  # Phase 3b: interest → skill vector → ranked programs
│   ├── chat.py                  # Phase 5: CLI conversational interface
│   ├── evaluate.py              # Phase 4: RAGAS + MRR/NDCG evaluation suite
│   ├── main.py                  # FastAPI server (POST /ask, POST /recommend)
│   └── .env                     # GEMINI_API_KEY — never commit this
│
├── frontend/                    # Next.js 14 frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   └── chat/page.tsx    # Main chat interface
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── RAGResponse.tsx
│   │   │   ├── RecommendationResponse.tsx
│   │   │   ├── InterestChips.tsx
│   │   │   └── MessageInput.tsx
│   │   └── lib/
│   │       ├── api.ts           # Typed fetch wrappers → FastAPI
│   │       └── types.ts         # TypeScript types matching Python dataclasses
│   ├── package.json
│   └── tailwind.config.ts
│
└── README.md
```

---

## Architecture

```
Student query (Thai / English)
        │
        ▼
   Intent detection
   (RAG or Recommend)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
RAG          Recommendation
Pipeline     Pipeline
   │         │
   │  embed query (gemini-embedding-001)
   │         │
   ▼         ▼
ChromaDB     ChromaDB + Neo4j-style graph
(pgvector    (interest vector → program
 similarity) → skill → career traversal)
   │         │
   └────┬────┘
        │
        ▼
  Gemini 2.5 Flash-Lite
  (grounded generation)
        │
        ▼
  Answer + citations / Ranked programs + explanations
```

**Knowledge graph stats (mock data):**
- 3 programs (CPE, CS, SKE) · 22 PLOs · 24 skill clusters · 5 careers
- 145 edges: `HAS_PLO` · `DEVELOPS` · `REQUIRED_FOR` · `LEADS_TO`
- 87 embedded chunks across 4 types: `plo_full`, `plo_th`, `plo_en`, `tcas_round`

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM generation | Gemini 2.5 Flash-Lite (`gemini-2.5-flash-lite-preview-06-17`) |
| Embeddings | Gemini Embedding 001 (`gemini-embedding-001`) |
| Vector store | ChromaDB (local, persistent, cosine similarity) |
| Knowledge graph | NetworkX (in-memory, persisted to JSON) |
| Backend API | Python 3.12 · FastAPI · Uvicorn |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS · shadcn/ui |
| Evaluation | RAGAS 0.4 · LangChain Google GenAI wrapper |

---

## Prerequisites

- Python 3.11 or 3.12
- Node.js 18+
- A Gemini API key — free tier is sufficient
  → Get one at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
  → Free tier: **1,000 requests/day, 15 RPM** for Flash-Lite

---

## Setup and run

### 1. Clone and set up the backend

```bash
git clone https://github.com/your-username/kuru-prototype.git
cd kuru-prototype/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install chromadb google-genai networkx python-dotenv \
            fastapi uvicorn ragas datasets langchain-google-genai
```

### 2. Add your API key

```bash
cp .env.example .env
# Open .env and set:
# GEMINI_API_KEY=your_key_here
```

### 3. Run the data pipeline (Phase 2)

This embeds all 87 chunks into ChromaDB and builds the knowledge graph.
Takes ~2 minutes on the free tier.

```bash
python ingest.py
```

Expected output:
```
[Step 1/4]  Chunking documents...   PLO: 72  TCAS: 15  Total: 87
[Step 2/4]  Embedding chunks...     Done in ~90s  dim=3072
[Step 3/4]  Storing in ChromaDB...  87 documents stored
[Step 4/4]  Building graph...       54 nodes  145 edges
[Smoke Test]  ALL PASS ✓
```

### 4. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start the Next.js frontend

```bash
cd ../frontend
npm install
npm run dev
# Open: http://localhost:3000
```

### 6. (Optional) CLI chat — no frontend needed

```bash
cd backend
python chat.py
```

### 7. (Optional) Run the evaluation suite

```bash
python evaluate.py
# Results saved to logs/eval_report.json
```

---

## API reference

### `POST /ask` — RAG pipeline

```json
// Request
{
  "question": "CPE สอนทักษะอะไรบ้าง",
  "program_filter": "CPE"   // optional: "CPE" | "CS" | "SKE" | null
}

// Response
{
  "question": "CPE สอนทักษะอะไรบ้าง",
  "answer": "หลักสูตร CPE มุ่งพัฒนาทักษะ...",
  "citations": [
    { "chunk_id": "CPE-PLO3_full", "program_id": "CPE",
      "plo_id": "CPE-PLO3", "chunk_type": "plo_full", "similarity": 0.91 }
  ],
  "retrieved": [...],
  "model_used": "gemini-2.5-flash-lite-preview-06-17",
  "retrieval_ms": 142,
  "generation_ms": 1840
}
```

### `POST /recommend` — Recommendation pipeline

```json
// Request
{
  "interests": {
    "ai_data": 0.9,
    "programming": 0.8,
    "math_theory": 0.6
  }
}

// Response
{
  "interests": { "ai_data": 0.9, "programming": 0.8, "math_theory": 0.6 },
  "skill_vector": { "machine_learning": 1.0, "data_engineering": 0.9, ... },
  "programs": [
    {
      "program_id": "CS",
      "program_name_th": "วิทยาการคอมพิวเตอร์",
      "program_name_en": "Computer Science",
      "fit_score": 0.91,
      "vector_score": 0.88,
      "graph_score": 0.93,
      "matched_skills": ["machine_learning", "data_engineering", ...],
      "top_careers": [
        { "title_th": "วิศวกรปัญญาประดิษฐ์", "coverage": 0.82,
          "salary_entry": "35,000–65,000 บาท/เดือน" }
      ],
      "plo_highlights": ["PLO3: สามารถออกแบบและพัฒนาระบบ AI..."],
      "explanation": "CS เน้นทฤษฎีและรากฐาน AI อย่างลึกซึ้ง..."
    }
  ],
  "generation_ms": 3200
}
```

---

## Valid interest topics

Use these exact keys in the `/recommend` request body:

| Key | Description |
|---|---|
| `programming` | Coding, software development, app building |
| `ai_data` | AI, machine learning, data science, data engineering |
| `design_ux` | UX/UI design, human-computer interaction |
| `systems_hardware` | Hardware, embedded systems, IoT, robotics |
| `math_theory` | Mathematics, algorithms, theoretical CS |
| `business_product` | Product management, project management, startup |
| `security_networks` | Cybersecurity, networking |
| `research_science` | Academic research, scientific computing |

Values are floats from 0.0 to 1.0 (strength of interest).

---

## Evaluation targets

| Metric | Target | Method |
|---|---|---|
| RAGAS Faithfulness | > 0.80 | LLM-judged NLI on 12 RAG test cases |
| RAGAS Answer Relevancy | > 0.75 | LLM-judged relevance score |
| Recommendation MRR | > 0.60 | 6 interest profiles vs ground-truth rankings |
| Recommendation NDCG@3 | > 0.60 | Discounted cumulative gain on program ordering |
| Task Completion Rate | > 80% | 5 end-to-end tasks with keyword validation |
| Hallucination resistance | 100% | 2 out-of-scope questions must be declined |

---

## Rate limits (free tier)

| Model | RPM | RPD | Used for |
|---|---|---|---|
| `gemini-embedding-001` | 1,500 | unlimited | Ingest + query embedding |
| `gemini-2.5-flash-lite-preview-06-17` | 15 | 1,000 | Generation, explanations |

The ingest pipeline uses ~90 embedding calls. Each chat turn uses 1–3 generation calls depending on intent. The evaluation suite uses ~30 generation calls. All fit comfortably within the free daily quota.

To upgrade for production, switch `GENERATION_MODEL` in `config.py`:
- `gemini-2.5-flash` — better reasoning, 10 RPM / 250 RPD free
- `gemini-2.5-pro` — best quality, 5 RPM / 100 RPD free

---

## Mock data scope

This prototype uses representative mock data covering:

- **3 KU programs** — CPE, CS, SKE — with full Thai PLO text written in มคอ.2 academic style (7–8 PLOs each)
- **5 careers** — Software Developer, Data Engineer, AI/ML Engineer, UX Designer, Systems Analyst — with O*NET SOC codes, Thai descriptions, and realistic Thai salary ranges
- **TCAS data** — All 4 rounds for all 3 programs with actual exam names (TGAT/TPAT3/A-Level), score weights, minimum thresholds, required documents, and deadlines

**Production** will replace the JSON files with real มคอ.2 PDFs processed through PyMuPDF, real TCAS data provided by the faculty advisor, and real graduation outcome data.

---

## Roadmap

**Phase 2 (production)** — extends to enrolled KU students:
- Skill progress dashboard mapped to completed courses
- Elective recommender for gap-closing before graduation
- Career readiness report for job applications

**Pending for production migration:**
- Replace ChromaDB with Supabase pgvector
- Replace NetworkX dict graph with Neo4j AuraDB
- Replace JSON mock data with real มคอ.2 PDFs (PyMuPDF extraction)
- Add Supabase Auth for saved profiles and session persistence
- Deploy backend to Railway, frontend to Vercel

---

## Team

| Name | Student ID | Responsibility |
|---|---|---|
| Thanawat Tantijaroensin | 6610545294 | AI pipeline, RAG system, FastAPI backend, knowledge graph |
| Phantawat Luengsiriwattana | 6610545871 | Next.js frontend, UI/UX design, evaluation, user testing |

Department of Computer Engineering · Faculty of Engineering  
Kasetsart University · Academic Year 2568

---

## .gitignore

Add this to your root `.gitignore` before pushing:

```
# Environment
backend/.env

# Generated databases (rebuild with ingest.py)
backend/db/

# Logs
backend/logs/

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# Node
frontend/node_modules/
frontend/.next/

# OS
.DS_Store
Thumbs.db
```

---

*KUru — "KU รู้" (KU knows) · The only system that connects student interests → real KU PLOs → TCAS admission pathways → actual KU graduate outcomes, specific to Kasetsart University.*