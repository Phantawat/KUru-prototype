"""
chunker.py  —  Convert programs.json into retrievable text chunks.

Each chunk is a self-contained unit of curriculum knowledge that can be
retrieved independently. We produce multiple chunk types per PLO so that
different kinds of student questions hit the right content.

Chunk types produced:
  - plo_full      : complete PLO text (Thai + English, context-rich)
  - plo_th        : Thai-only (for Thai queries)
  - plo_en        : English-only (for English queries)
  - program_intro : program description (Thai + English)
  - career_bridge : what careers this program leads to
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import config


def _clean(text: str) -> str:
    """Normalise whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _make_chunk(
    chunk_id: str,
    text: str,
    chunk_type: str,
    program_id: str,
    program_name_th: str,
    plo_id: str | None = None,
    plo_number: int | None = None,
    domain: str | None = None,
    skill_clusters: list | None = None,
    language: str = "bilingual",
) -> Dict[str, Any]:
    return {
        "chunk_id":       chunk_id,
        "text":           _clean(text),
        "chunk_type":     chunk_type,
        "program_id":     program_id,
        "program_name_th": program_name_th,
        "plo_id":         plo_id or "",
        "plo_number":     plo_number or 0,
        "domain":         domain or "",
        "skill_clusters": json.dumps(skill_clusters or [], ensure_ascii=False),
        "language":       language,
    }


def chunk_programs(programs_path: Path) -> List[Dict[str, Any]]:
    """
    Load programs.json and return a flat list of chunks ready for embedding.
    """
    with open(programs_path, encoding="utf-8") as f:
        programs = json.load(f)

    chunks: List[Dict[str, Any]] = []

    for prog in programs:
        pid   = prog["program_id"]
        pname = prog["program_name_th"]
        pen   = prog["program_name_en"]
        fac   = prog["faculty_th"]
        desc_th = prog["program_description_th"]
        desc_en = prog["program_description_en"]
        careers = prog.get("career_outcomes", [])

        # ── 1. Program intro chunk (bilingual) ────────────────────────────────
        intro_text = (
            f"หลักสูตร: {pname} ({pen})\n"
            f"คณะ: {fac}\n"
            f"ปริญญา: {prog['degree']}\n"
            f"ระยะเวลา: {prog['duration_years']} ปี  หน่วยกิต: {prog['total_credits']}\n\n"
            f"คำอธิบายหลักสูตร (ภาษาไทย):\n{desc_th}\n\n"
            f"Program description (English):\n{desc_en}"
        )
        chunks.append(_make_chunk(
            chunk_id      = f"{pid}_intro",
            text          = intro_text,
            chunk_type    = "program_intro",
            program_id    = pid,
            program_name_th = pname,
            language      = "bilingual",
        ))

        # ── 2. Career bridge chunk ────────────────────────────────────────────
        career_text = (
            f"หลักสูตร {pname} ({pid}) เปิดโอกาสให้บัณฑิตประกอบอาชีพ:\n"
            + "\n".join(f"- {c}" for c in careers)
            + f"\n\nProgram {pid} ({pen}) graduates can pursue careers including: "
            + ", ".join(careers)
        )
        chunks.append(_make_chunk(
            chunk_id        = f"{pid}_careers",
            text            = career_text,
            chunk_type      = "career_bridge",
            program_id      = pid,
            program_name_th = pname,
            skill_clusters  = careers,
            language        = "bilingual",
        ))

        # ── 3. Per-PLO chunks ─────────────────────────────────────────────────
        for plo in prog["plos"]:
            plo_id  = plo["plo_id"]
            plo_num = plo["plo_number"]
            domain  = plo["domain"]
            th      = plo["text_th"]
            en      = plo["text_en"]
            sc      = plo.get("skill_clusters", [])
            bloom   = plo.get("bloom_level", "")
            method  = plo.get("assessment_method", "")

            # 3a. Full bilingual chunk — richest context, best for complex questions
            full_text = (
                f"หลักสูตร: {pname} ({pid})\n"
                f"ผลลัพธ์การเรียนรู้ที่คาดหวัง (PLO) ที่ {plo_num}: ด้าน{domain}\n\n"
                f"[ภาษาไทย]\n{th}\n\n"
                f"[English]\n{en}\n\n"
                f"ทักษะที่พัฒนา: {', '.join(sc)}\n"
                f"ระดับการเรียนรู้ (Bloom): {bloom}\n"
                f"วิธีการประเมิน: {method}"
            )
            chunks.append(_make_chunk(
                chunk_id        = f"{plo_id}_full",
                text            = full_text,
                chunk_type      = "plo_full",
                program_id      = pid,
                program_name_th = pname,
                plo_id          = plo_id,
                plo_number      = plo_num,
                domain          = domain,
                skill_clusters  = sc,
                language        = "bilingual",
            ))

            # 3b. Thai-only chunk — for Thai student queries
            th_text = (
                f"หลักสูตร {pname} ({pid}) — PLO {plo_num} (ด้าน{domain})\n\n"
                f"{th}\n\n"
                f"ทักษะที่พัฒนา: {', '.join(sc)}"
            )
            chunks.append(_make_chunk(
                chunk_id        = f"{plo_id}_th",
                text            = th_text,
                chunk_type      = "plo_th",
                program_id      = pid,
                program_name_th = pname,
                plo_id          = plo_id,
                plo_number      = plo_num,
                domain          = domain,
                skill_clusters  = sc,
                language        = "thai",
            ))

            # 3c. English-only chunk — for English queries / cross-lingual retrieval
            en_text = (
                f"Program: {pen} ({pid}) — PLO {plo_num} ({domain})\n\n"
                f"{en}\n\n"
                f"Skills developed: {', '.join(sc)}"
            )
            chunks.append(_make_chunk(
                chunk_id        = f"{plo_id}_en",
                text            = en_text,
                chunk_type      = "plo_en",
                program_id      = pid,
                program_name_th = pname,
                plo_id          = plo_id,
                plo_number      = plo_num,
                domain          = domain,
                skill_clusters  = sc,
                language        = "english",
            ))

    return chunks


def chunk_tcas(tcas_path: Path) -> List[Dict[str, Any]]:
    """
    Load tcas.json and return TCAS admission chunks.
    Separate chunk per program × round so retrieval is precise.
    """
    with open(tcas_path, encoding="utf-8") as f:
        tcas = json.load(f)

    chunks: List[Dict[str, Any]] = []

    for prog in tcas["programs"]:
        pid   = prog["program_id"]
        pname = prog["program_name_th"]

        # ── Per-round chunks ──────────────────────────────────────────────────
        for rnd in prog["admission_rounds"]:
            rnum  = rnd["round"]
            rname = rnd["round_name_th"]
            rdesc = rnd["round_description_th"]
            quota = rnd["quota"]
            elig  = rnd.get("eligibility_th", "")
            notes = rnd.get("notes_th", "")

            # Build score requirements text
            score_lines = []
            if "required_scores" in rnd:
                for exam, details in rnd["required_scores"].items():
                    if isinstance(details, dict):
                        w = details.get("weight_in_selection") or details.get("weight", "")
                        name = details.get("full_name_th") or details.get("subject_th") or exam
                        mn   = details.get("min_score", "")
                        score_lines.append(
                            f"  • {name}: คะแนนขั้นต่ำ {mn if mn else '-'}  น้ำหนัก {w}"
                        )
                        # Sub-components
                        if "components" in details:
                            for comp_key, comp in details["components"].items():
                                cn = comp.get("name_th") or comp_key
                                cm = comp.get("min_score", "")
                                cw = comp.get("weight", "")
                                score_lines.append(f"      - {cn}: ขั้นต่ำ {cm}  น้ำหนัก {cw}")

            # Build selection criteria text
            criteria = rnd.get("selection_criteria_th", {})
            criteria_lines = [
                f"  • {k}: {v}" for k, v in criteria.items() if k != "detail"
            ]
            if "detail" in criteria:
                criteria_lines.append(f"  หมายเหตุ: {criteria['detail']}")

            # Documents
            docs = rnd.get("required_documents_th", [])
            doc_text = "\n".join(f"  • {d}" for d in docs) if docs else "  -"

            # Preferred portfolio (round 1)
            portfolio = rnd.get("preferred_portfolio_content_th", [])
            port_text = "\n".join(f"  • {p}" for p in portfolio) if portfolio else ""

            # Estimated scores
            est = rnd.get("estimated_min_scores_th", {})
            est_lines = []
            for k, v in est.items():
                if k != "note":
                    est_lines.append(f"  • {k}: {v}")
            if est.get("note"):
                est_lines.insert(0, f"  หมายเหตุ: {est['note']}")

            body = (
                f"หลักสูตร: {pname} ({pid})\n"
                f"การรับเข้า TCAS: {rname}\n"
                f"รายละเอียด: {rdesc}\n\n"
                f"จำนวนรับ: {quota} คน\n"
                f"คุณสมบัติผู้สมัคร: {elig}\n"
            )

            if score_lines:
                body += "\nคะแนนที่ใช้:\n" + "\n".join(score_lines) + "\n"

            if criteria_lines:
                body += "\nเกณฑ์การคัดเลือก:\n" + "\n".join(criteria_lines) + "\n"

            if docs:
                body += "\nเอกสารที่ต้องใช้:\n" + doc_text + "\n"

            if port_text:
                body += "\nผลงานที่ควรมีใน Portfolio:\n" + port_text + "\n"

            if est_lines:
                body += "\nคะแนนขั้นต่ำโดยประมาณ (อ้างอิงจากปีก่อน):\n" + "\n".join(est_lines) + "\n"

            if notes:
                body += f"\nคำแนะนำ: {notes}\n"

            chunks.append({
                "chunk_id":         f"tcas_{pid}_round{rnum}",
                "text":             _clean(body),
                "chunk_type":       "tcas_round",
                "program_id":       pid,
                "program_name_th":  pname,
                "plo_id":           "",
                "plo_number":       0,
                "domain":           f"tcas_round_{rnum}",
                "skill_clusters":   "[]",
                "language":         "thai",
            })

        # ── Program-level TCAS overview chunk ─────────────────────────────────
        total   = prog["total_quota"]
        tuition = prog.get("tuition_fee_th", "")
        scholar = prog.get("scholarship_info_th", "")
        overview = (
            f"สรุปการรับเข้า TCAS หลักสูตร {pname} ({pid})\n"
            f"มหาวิทยาลัยเกษตรศาสตร์ ปีการศึกษา 2568\n\n"
            f"จำนวนรับรวมทุกรอบ: {total} คน\n"
            f"ค่าธรรมเนียม: {tuition}\n"
            f"ทุนการศึกษา: {scholar}\n\n"
            "รอบที่เปิดรับ:\n"
        )
        for rnd in prog["admission_rounds"]:
            overview += f"  • รอบ {rnd['round']} ({rnd['round_name_th']}): {rnd['quota']} คน\n"

        chunks.append({
            "chunk_id":         f"tcas_{pid}_overview",
            "text":             _clean(overview),
            "chunk_type":       "tcas_overview",
            "program_id":       pid,
            "program_name_th":  pname,
            "plo_id":           "",
            "plo_number":       0,
            "domain":           "tcas_overview",
            "skill_clusters":   "[]",
            "language":         "thai",
        })

    return chunks


if __name__ == "__main__":
    plo_chunks  = chunk_programs(config.DATA_DIR / "programs.json")
    tcas_chunks = chunk_tcas(config.DATA_DIR / "tcas.json")
    all_chunks  = plo_chunks + tcas_chunks

    print(f"PLO chunks:  {len(plo_chunks)}")
    print(f"TCAS chunks: {len(tcas_chunks)}")
    print(f"Total:       {len(all_chunks)}")

    # Show sample chunks
    print("\n── Sample PLO chunk (full bilingual) ──")
    sample = next(c for c in all_chunks if c["chunk_type"] == "plo_full")
    print(sample["text"][:400], "...")

    print("\n── Sample TCAS chunk ──")
    sample_tcas = next(c for c in all_chunks if c["chunk_type"] == "tcas_round")
    print(sample_tcas["text"][:400], "...")