"""
graph_builder.py  —  Build the KUru knowledge graph.

Node types:
  program       : a KU degree program (CPE, CS, SKE)
  plo           : a Program Learning Outcome
  skill_cluster : an abstract skill grouping (e.g. "machine_learning")
  career        : an O*NET-based career (e.g. "data_engineer")

Edge types:
  HAS_PLO           : program → plo
  DEVELOPS          : plo → skill_cluster
  REQUIRED_FOR      : skill_cluster → career
  LEADS_TO          : program → career  (shortcut, derived)

The graph is stored as a NetworkX DiGraph and also serialised to JSON
so the pipeline can load it without rebuilding every run.
"""

import json
from pathlib import Path
import networkx as nx

import config


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_node(G: nx.DiGraph, node_id: str, **attrs):
    if node_id not in G:
        G.add_node(node_id, **attrs)


def _add_edge(G: nx.DiGraph, src: str, dst: str, **attrs):
    G.add_edge(src, dst, **attrs)


# ── builders ──────────────────────────────────────────────────────────────────

def build_program_graph(G: nx.DiGraph, programs: list) -> None:
    """Add program and PLO nodes + HAS_PLO / DEVELOPS edges."""
    for prog in programs:
        pid   = prog["program_id"]
        pname = prog["program_name_th"]
        pen   = prog["program_name_en"]

        _add_node(G, pid,
            type            = "program",
            name_th         = pname,
            name_en         = pen,
            degree          = prog["degree"],
            faculty_th      = prog["faculty_th"],
            total_credits   = prog["total_credits"],
            duration_years  = prog["duration_years"],
            tuition         = prog["tuition_per_semester_thb"],
        )

        for plo in prog["plos"]:
            plo_id = plo["plo_id"]
            _add_node(G, plo_id,
                type        = "plo",
                program_id  = pid,
                plo_number  = plo["plo_number"],
                domain      = plo["domain"],
                text_th     = plo["text_th"],
                text_en     = plo["text_en"],
                bloom_level = plo.get("bloom_level", ""),
            )
            _add_edge(G, pid, plo_id, relation="HAS_PLO")

            # PLO → skill clusters
            for sc in plo.get("skill_clusters", []):
                _add_node(G, sc, type="skill_cluster", name=sc)
                _add_edge(G, plo_id, sc, relation="DEVELOPS")


def build_career_graph(G: nx.DiGraph, careers: list) -> None:
    """Add career nodes + REQUIRED_FOR edges from skill clusters."""
    for career in careers:
        cid = career["career_id"]
        _add_node(G, cid,
            type            = "career",
            title_th        = career["title_th"],
            title_en        = career["title_en"],
            description_th  = career["description_th"],
            onet_soc_code   = career["onet_soc_code"],
            salary_entry_min= career["salary_range_thb"]["entry_level"]["min"],
            salary_entry_max= career["salary_range_thb"]["entry_level"]["max"],
            salary_mid_min  = career["salary_range_thb"]["mid_level"]["min"],
            salary_mid_max  = career["salary_range_thb"]["mid_level"]["max"],
            outlook_th      = career["job_market_outlook_th"],
            related_programs= career.get("related_programs", []),
        )

        for skill in career["required_skills"]:
            sc = skill["skill_cluster"]
            # Ensure skill_cluster node exists
            _add_node(G, sc, type="skill_cluster", name=sc)
            _add_edge(G, sc, cid,
                relation        = "REQUIRED_FOR",
                importance      = skill["importance_score"],
                level_required  = skill["level_required"],
            )


def add_shortcut_edges(G: nx.DiGraph) -> None:
    """
    Add LEADS_TO edges: program → career.
    Derived by traversing program → plo → skill_cluster → career.
    Weight = average importance of the skill_cluster→career edges
    encountered on all paths from that program.
    """
    programs = [n for n, d in G.nodes(data=True) if d.get("type") == "program"]
    careers  = [n for n, d in G.nodes(data=True) if d.get("type") == "career"]

    for prog in programs:
        # Collect all skill clusters reachable from this program
        prog_skills: dict[str, float] = {}   # skill → max importance weight
        for plo in G.successors(prog):
            if G.nodes[plo].get("type") != "plo":
                continue
            for sc in G.successors(plo):
                if G.nodes[sc].get("type") != "skill_cluster":
                    continue
                prog_skills[sc] = prog_skills.get(sc, 0) + 1  # count coverage

        # Score each career by how many of its required skills this program covers
        for career in careers:
            career_skills = {}
            for sc in G.predecessors(career):
                if G.nodes[sc].get("type") != "skill_cluster":
                    continue
                imp = G[sc][career].get("importance", 50)
                career_skills[sc] = imp

            if not career_skills:
                continue

            # Coverage score: sum of importances for skills the program develops
            covered = {sc: imp for sc, imp in career_skills.items() if sc in prog_skills}
            if not covered:
                continue

            coverage_score = sum(covered.values()) / sum(career_skills.values())  # 0–1
            _add_edge(G, prog, career,
                relation       = "LEADS_TO",
                coverage_score = round(coverage_score, 3),
                covered_skills = list(covered.keys()),
            )


# ── serialisation ─────────────────────────────────────────────────────────────

def graph_to_dict(G: nx.DiGraph) -> dict:
    """Convert to a plain dict for JSON persistence."""
    return {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items() if isinstance(v, (str, int, float, bool, list))}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"src": u, "dst": v, **{k: v2 for k, v2 in d.items() if isinstance(v2, (str, int, float, bool, list))}}
            for u, v, d in G.edges(data=True)
        ],
    }


def dict_to_graph(data: dict) -> nx.DiGraph:
    """Reconstruct graph from persisted dict."""
    G = nx.DiGraph()
    for node in data["nodes"]:
        nid = node.pop("id")
        G.add_node(nid, **node)
    for edge in data["edges"]:
        src = edge.pop("src")
        dst = edge.pop("dst")
        G.add_edge(src, dst, **edge)
    return G


def load_graph(path: Path | None = None) -> nx.DiGraph:
    """Load graph from JSON if it exists, else return empty DiGraph."""
    path = path or config.GRAPH_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return dict_to_graph(json.load(f))
    return nx.DiGraph()


def save_graph(G: nx.DiGraph, path: Path | None = None) -> None:
    path = path or config.GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_to_dict(G), f, ensure_ascii=False, indent=2)


# ── query helpers (used by recommendation pipeline) ───────────────────────────

def get_program_skills(G: nx.DiGraph, program_id: str) -> dict[str, int]:
    """
    Return {skill_cluster: plo_count} for a program.
    plo_count = how many PLOs develop this skill (proxy for depth).
    """
    skills: dict[str, int] = {}
    for plo in G.successors(program_id):
        if G.nodes[plo].get("type") != "plo":
            continue
        for sc in G.successors(plo):
            if G.nodes[sc].get("type") != "skill_cluster":
                continue
            skills[sc] = skills.get(sc, 0) + 1
    return skills


def get_career_path(G: nx.DiGraph, program_id: str, career_id: str) -> dict:
    """
    Return the full PLO→Skill→Career chain for one program→career pair.
    Used to generate the 'why this program' explanation.
    """
    chain = {"program": program_id, "career": career_id, "paths": []}
    if not G.has_node(program_id) or not G.has_node(career_id):
        return chain

    career_skills = {
        sc: G[sc][career_id].get("importance", 50)
        for sc in G.predecessors(career_id)
        if G.nodes[sc].get("type") == "skill_cluster"
    }

    for plo in G.successors(program_id):
        if G.nodes[plo].get("type") != "plo":
            continue
        plo_data = G.nodes[plo]
        for sc in G.successors(plo):
            if sc not in career_skills:
                continue
            chain["paths"].append({
                "plo_id":     plo,
                "plo_number": plo_data.get("plo_number"),
                "plo_text_th": plo_data.get("text_th", "")[:120] + "...",
                "skill":      sc,
                "importance": career_skills[sc],
            })

    chain["paths"].sort(key=lambda x: x["importance"], reverse=True)
    return chain


def get_careers_for_program(G: nx.DiGraph, program_id: str) -> list[dict]:
    """Return all careers reachable from a program with coverage scores."""
    result = []
    for career in G.successors(program_id):
        if G.nodes[career].get("type") != "career":
            continue
        edge = G[program_id][career]
        career_data = G.nodes[career]
        result.append({
            "career_id":      career,
            "title_th":       career_data.get("title_th", ""),
            "title_en":       career_data.get("title_en", ""),
            "coverage_score": edge.get("coverage_score", 0),
            "covered_skills": edge.get("covered_skills", []),
            "salary_entry":   f"{career_data.get('salary_entry_min',0):,}–{career_data.get('salary_entry_max',0):,} บาท/เดือน",
            "outlook_th":     career_data.get("outlook_th", ""),
        })
    result.sort(key=lambda x: x["coverage_score"], reverse=True)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def build_graph(programs_path: Path, careers_path: Path) -> nx.DiGraph:
    with open(programs_path, encoding="utf-8") as f:
        programs = json.load(f)
    with open(careers_path, encoding="utf-8") as f:
        careers = json.load(f)

    G = nx.DiGraph()
    build_program_graph(G, programs)
    build_career_graph(G, careers)
    add_shortcut_edges(G)
    return G


if __name__ == "__main__":
    G = build_graph(
        config.DATA_DIR / "programs.json",
        config.DATA_DIR / "careers.json",
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
    node_types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, d in G.edges(data=True):
        r = d.get("relation", "unknown")
        edge_types[r] = edge_types.get(r, 0) + 1

    print("── Knowledge Graph ─────────────────────────────────")
    print(f"  Nodes: {G.number_of_nodes()}")
    for t, cnt in sorted(node_types.items()):
        print(f"    {t:<20} {cnt}")
    print(f"  Edges: {G.number_of_edges()}")
    for r, cnt in sorted(edge_types.items()):
        print(f"    {r:<20} {cnt}")

    # ── Sample traversal: CPE → careers ──────────────────────────────────────
    print("\n── CPE → careers (LEADS_TO edges) ─────────────────")
    for c in get_careers_for_program(G, "CPE"):
        print(f"  {c['coverage_score']:.2f}  {c['title_th']:<30}  {c['salary_entry']}")

    # ── Sample career path ────────────────────────────────────────────────────
    print("\n── CPE → data_engineer path (top 3 PLO→Skill links) ─")
    path = get_career_path(G, "CPE", "data_engineer")
    for p in path["paths"][:3]:
        print(f"  PLO{p['plo_number']}  →  {p['skill']:<25}  importance={p['importance']}")
        print(f"    \"{p['plo_text_th']}\"")

    # ── Persist ───────────────────────────────────────────────────────────────
    save_graph(G)
    print(f"\n✓  Graph saved → {config.GRAPH_PATH}")