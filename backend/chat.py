"""
chat.py  —  Phase 5: Conversational chat interface for KUru prototype.

Run:
  python chat.py

The chat loop:
  - Detects intent from the student's message
  - Routes to RAG pipeline (curriculum/TCAS questions)
    or Recommendation pipeline (interest/program matching)
  - Displays the answer with citations or ranked programs
  - Maintains a short conversation history for context

Intent detection uses a simple keyword approach + a Gemini classifier
for ambiguous cases — fast and transparent.
"""

import logging
import os
import sys
import textwrap
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

import config
from rag_pipeline import RAGPipeline
from recommendation_pipeline import RecommendationPipeline, compute_skill_vector

# ── Logging: file only — keep terminal clean ──────────────────────────────────
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(levelname)-7s  %(message)s",
    handlers = [logging.FileHandler(config.LOG_DIR / "chat.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


# ── Intent detection ──────────────────────────────────────────────────────────

RAG_KEYWORDS = [
    # Thai
    "สอน", "ทักษะ", "plo", "ผลลัพธ์การเรียนรู้", "หลักสูตร", "วิชา",
    "tcas", "รอบ", "สมัคร", "คะแนน", "portfolio", "เกณฑ์", "คุณสมบัติ",
    "เอกสาร", "ค่าเล่าเรียน", "ทุน", "สัมภาษณ์", "a-level", "tgat", "tpat",
    "รับสมัคร", "โควตา", "ประกาศผล",
    # English
    "teach", "skill", "learning outcome", "curriculum", "subject",
    "admission", "score", "apply", "round", "requirement", "document",
    "tuition", "scholarship", "interview",
]

RECOMMEND_KEYWORDS = [
    # Thai
    "อยากเรียน", "เหมาะกับ", "ควรเรียน", "แนะนำ", "ชอบ", "สนใจ",
    "ถนัด", "เส้นทาง", "อยากเป็น", "อาชีพ", "อยากทำงาน",
    "เลือก", "ตัดสินใจ", "ไม่รู้จะเรียน",
    # English
    "recommend", "suggest", "which program", "what should i study",
    "interested in", "good at", "career path", "become a", "want to be",
    "i like", "i enjoy", "not sure what to study",
]

INTEREST_TOPIC_KEYWORDS = {
    "programming":        ["โปรแกรม", "โค้ด", "coding", "programming", "เขียนโปรแกรม", "developer", "พัฒนา app"],
    "ai_data":            ["ai", "ปัญญาประดิษฐ์", "machine learning", "data science", "data engineer", "ข้อมูล", "ml"],
    "design_ux":          ["design", "ออกแบบ", "ux", "ui", "กราฟิก", "สวยงาม", "สร้างสรรค์"],
    "systems_hardware":   ["hardware", "embedded", "ฮาร์ดแวร์", "วงจร", "iot", "robot", "หุ่นยนต์"],
    "math_theory":        ["คณิต", "math", "ทฤษฎี", "algorithm", "theory", "พีชคณิต"],
    "business_product":   ["ธุรกิจ", "product", "startup", "management", "project", "บริหาร", "จัดการ"],
    "security_networks":  ["security", "ความปลอดภัย", "network", "เครือข่าย", "cyber", "hacker"],
    "research_science":   ["วิจัย", "research", "science", "วิทยาศาสตร์", "ทดลอง", "paper"],
}

PROGRAM_KEYWORDS = {
    "CPE": ["cpe", "วิศวกรรมคอมพิวเตอร์", "computer engineering"],
    "CS":  ["cs", "วิทยาการคอมพิวเตอร์", "computer science"],
    "SKE": ["ske", "วิศวกรรมซอฟต์แวร์", "software engineering", "software and knowledge"],
}


@dataclass
class Intent:
    type:           str     # "rag" | "recommend" | "clarify" | "exit" | "help"
    program_filter: str | None = None   # for RAG: restrict to one program
    interests:      dict[str, float] | None = None  # for recommend


def detect_intent(message: str, client: genai.Client) -> Intent:
    """
    Classify the message intent.
    First tries fast keyword matching, then falls back to Gemini classifier
    for ambiguous messages.
    """
    lower = message.lower()

    # ── Exit / help shortcuts ─────────────────────────────────────────────────
    if lower.strip() in ("exit", "quit", "bye", "q", "ออก"):
        return Intent(type="exit")
    if lower.strip() in ("help", "?", "ช่วยด้วย"):
        return Intent(type="help")

    # ── Detect program filter ─────────────────────────────────────────────────
    program_filter = None
    for pid, keywords in PROGRAM_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            program_filter = pid
            break

    # ── Detect recommendation intent ─────────────────────────────────────────
    rec_score = sum(1 for kw in RECOMMEND_KEYWORDS if kw in lower)

    # Also detect interests mentioned in the message
    detected_interests: dict[str, float] = {}
    for topic, keywords in INTEREST_TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            detected_interests[topic] = min(hits * 0.4 + 0.5, 1.0)

    # ── Detect RAG intent ─────────────────────────────────────────────────────
    rag_score = sum(1 for kw in RAG_KEYWORDS if kw in lower)

    # ── Decide ────────────────────────────────────────────────────────────────
    if rec_score > rag_score or (detected_interests and rec_score >= rag_score):
        # Recommendation intent — use detected interests or ask Gemini to extract them
        if not detected_interests:
            detected_interests = _extract_interests_via_llm(message, client)
        return Intent(type="recommend", interests=detected_interests)

    if rag_score > 0 or program_filter:
        return Intent(type="rag", program_filter=program_filter)

    # ── Ambiguous — use Gemini as fallback classifier ─────────────────────────
    return _classify_via_llm(message, client, program_filter, detected_interests)


def _classify_via_llm(
    message:          str,
    client:           genai.Client,
    program_filter:   str | None,
    detected_interests: dict,
) -> Intent:
    """Use Gemini to classify ambiguous messages."""
    prompt = f"""Classify this student message into one of these intents:
- "rag": asking about curriculum content, PLOs, skills taught, TCAS admission, scores, deadlines
- "recommend": asking which program to choose, expressing interests, seeking guidance on what to study

Message: "{message}"

Reply with ONLY one word: rag OR recommend"""

    try:
        resp = client.models.generate_content(
            model    = config.GENERATION_MODEL,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature=0, max_output_tokens=5
            ),
        )
        intent_type = resp.text.strip().lower()
        if intent_type not in ("rag", "recommend"):
            intent_type = "rag"   # default to RAG for unknown
    except Exception:
        intent_type = "rag"

    if intent_type == "recommend" and not detected_interests:
        detected_interests = _extract_interests_via_llm(message, client)

    return Intent(
        type           = intent_type,
        program_filter = program_filter,
        interests      = detected_interests if intent_type == "recommend" else None,
    )


def _extract_interests_via_llm(message: str, client: genai.Client) -> dict[str, float]:
    """
    Use Gemini to extract interest topics from a free-form student message.
    Returns {topic: weight} for any topics mentioned.
    """
    valid_topics = list(config.INTEREST_SKILL_MATRIX.keys())
    prompt = f"""Extract the student's interests from this message.
Valid topics: {valid_topics}

Message: "{message}"

Return ONLY a JSON object mapping topic names to weights (0.5–1.0).
Example: {{"programming": 0.9, "ai_data": 0.7}}
If no specific interest is clear, return {{"programming": 0.6}} as a default.
Return only valid JSON, nothing else."""

    try:
        resp = client.models.generate_content(
            model    = config.GENERATION_MODEL,
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature=0, max_output_tokens=100
            ),
        )
        import json, re
        text = resp.text.strip()
        # Extract JSON even if wrapped in ```
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            interests = json.loads(match.group())
            # Validate keys
            return {k: float(v) for k, v in interests.items() if k in valid_topics}
    except Exception as e:
        log.warning(f"Interest extraction failed: {e}")

    return {"programming": 0.6}   # safe fallback


# ── Chat UI helpers ───────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
GOLD  = "\033[93m"
GRAY  = "\033[90m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    """Word-wrap text with consistent indentation."""
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=width,
                                     subsequent_indent=indent)
            lines.append(wrapped)
    return "\n".join(lines)


def print_banner():
    print(f"""
{BOLD}{GREEN}╔══════════════════════════════════════════════════════════╗
║           KUru — Intelligent PLO-to-Career Navigator        ║
║           Kasetsart University · Prototype v0.1             ║
╚══════════════════════════════════════════════════════════════╝{RESET}

{CYAN}ยินดีต้อนรับ! / Welcome!{RESET}
ถามเกี่ยวกับหลักสูตรหรือ TCAS ของมหาวิทยาลัยเกษตรศาสตร์
หรือบอกความสนใจของคุณ แล้ว KUru จะแนะนำหลักสูตรที่เหมาะสม

Ask about KU programs or TCAS, or describe your interests
and KUru will recommend the best program for you.

{GRAY}Available programs: CPE (Computer Engineering), 
CS (Computer Science), SKE (Software & Knowledge Engineering)

Type 'help' for example questions. Type 'exit' to quit.{RESET}
""")


def print_help():
    print(f"""
{GOLD}─── Example questions ───────────────────────────────────────{RESET}

{CYAN}Curriculum / PLO questions:{RESET}
  • CPE สอนทักษะอะไรบ้าง
  • What skills does Computer Science teach?
  • SKE มี PLO เกี่ยวกับ UX Design ไหม
  • หลักสูตรไหนเรียนเกี่ยวกับ machine learning

{CYAN}TCAS questions:{RESET}
  • CPE รอบ Portfolio ต้องเตรียม Portfolio อะไรบ้าง
  • คะแนนที่ใช้สมัคร CS รอบ 3 มีอะไรบ้าง
  • SKE รอบ 2 GPAX ขั้นต่ำเท่าไหร่
  • TCAS รอบ 1 ของ CPE ปิดรับเมื่อไหร่

{CYAN}Program recommendation:{RESET}
  • ฉันชอบ programming และ AI อยากเรียนอะไร
  • I enjoy designing apps and working with users
  • ถนัดคณิต อยากทำ data science
  • ไม่รู้จะเรียนอะไร ชอบสร้างของ ชอบแก้ปัญหา

{GRAY}Tip: You can ask in Thai or English, or mix both.{RESET}
""")


# ── Main chat loop ────────────────────────────────────────────────────────────

def run():
    config.validate()

    print_banner()

    # Initialise pipelines (loads ChromaDB + graph)
    print(f"{GRAY}Initialising pipelines...{RESET}", end="", flush=True)
    try:
        rag   = RAGPipeline()
        rec   = RecommendationPipeline()
        gemini = genai.Client(api_key=config.GEMINI_API_KEY)
    except Exception as e:
        print(f"\n{BOLD}Error initialising: {e}{RESET}")
        print("Make sure you have run ingest.py first.")
        sys.exit(1)
    print(f" {GREEN}ready{RESET}\n")

    turn = 0

    while True:
        try:
            user_input = input(f"{CYAN}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}Bye! / ลาก่อน!{RESET}")
            break

        if not user_input:
            continue

        turn += 1
        log.info(f"Turn {turn}: \"{user_input}\"")

        intent = detect_intent(user_input, gemini)

        # ── Exit ──────────────────────────────────────────────────────────────
        if intent.type == "exit":
            print(f"\n{GRAY}Bye! ลาก่อน! 👋{RESET}\n")
            break

        # ── Help ──────────────────────────────────────────────────────────────
        if intent.type == "help":
            print_help()
            continue

        print(f"\n{GOLD}KUru:{RESET} ", end="", flush=True)

        # ── RAG pipeline ──────────────────────────────────────────────────────
        if intent.type == "rag":
            if intent.program_filter:
                print(f"{GRAY}[searching {intent.program_filter} curriculum...]{RESET}")
            else:
                print(f"{GRAY}[searching curriculum & TCAS database...]{RESET}")

            try:
                resp = rag.ask(user_input, program_filter=intent.program_filter)

                print()
                print(wrap(resp.answer))
                print()

                # Citations
                if resp.citations:
                    print(f"{GRAY}Sources:{RESET}")
                    seen = set()
                    for c in resp.citations:
                        key = (c["program_id"], c.get("plo_id", c["chunk_type"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        label = c.get("plo_id") or c["chunk_type"]
                        print(f"  {GRAY}• {c['program_id']} / {label}  "
                              f"(sim={c['similarity']:.2f}){RESET}")

                print(f"\n  {GRAY}⏱ retrieval {resp.retrieval_ms}ms  "
                      f"generation {resp.generation_ms}ms{RESET}")

            except Exception as e:
                print(f"\n{BOLD}Error: {e}{RESET}")
                log.error(f"RAG error: {e}", exc_info=True)

        # ── Recommendation pipeline ───────────────────────────────────────────
        elif intent.type == "recommend":
            interests = intent.interests or {"programming": 0.7}
            print(f"{GRAY}[analysing interests & ranking programs...]{RESET}")
            log.info(f"  Interests detected: {interests}")

            try:
                result = rec.recommend(interests)

                print()
                print(f"จากความสนใจของคุณ / Based on your interests:")
                shown = [f"{k.replace('_',' ')} ({v:.0%})"
                         for k, v in sorted(interests.items(),
                                            key=lambda x: -x[1])[:4]]
                print(f"  {', '.join(shown)}")
                print()

                for i, prog in enumerate(result.programs, 1):
                    bar_len = int(prog.fit_score * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)

                    print(f"{BOLD}#{i}  {prog.program_name_th} ({prog.program_id}){RESET}")
                    print(f"    {GREEN}{bar}{RESET}  {prog.fit_score:.0%} fit")
                    print()
                    print(wrap(prog.explanation, indent="    "))
                    print()
                    print(f"  {GRAY}Top careers: "
                          + ", ".join(c["title_th"] for c in prog.top_careers[:3])
                          + f"{RESET}")
                    if prog.top_careers:
                        sal = prog.top_careers[0].get("salary_entry", "")
                        if sal:
                            print(f"  {GRAY}Entry salary (est.): {sal}{RESET}")
                    print()

                print(f"  {GRAY}⏱ {result.generation_ms}ms total{RESET}")

            except Exception as e:
                print(f"\n{BOLD}Error: {e}{RESET}")
                log.error(f"Recommendation error: {e}", exc_info=True)

        print()


if __name__ == "__main__":
    run()
