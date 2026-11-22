"""Small offline FTS-vs-hybrid retrieval comparison; never calls Gemini."""
from __future__ import annotations

import json
from pathlib import Path

from yiti_ai.api import _topic_queries, verify_requested_detail
from yiti_ai.config import ROOT
from yiti_ai.local_retrieval import LocalFTSIndex, normalize_policy_query
from yiti_ai.semantic_retrieval import DEFAULT_VECTOR_PATH, SemanticIndex, reciprocal_rank_fusion


DB = ROOT / "data" / "processed" / "search.sqlite3"
REPORT = ROOT / ".cache" / "semantic" / "retrieval_comparison.json"
CASES = [
    ("Hostel curfew?", "EN", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("Hostel kab tak wapas aana hai?", "HI", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("Can I come back late to my hostel?", "EN", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("Gate closing time for hostel?", "EN", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("What's the hostel in-out timing?", "EN", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("Hostel curfue timing?", "EN", ("STUDENTS-DISCIPLINE-AND-CONDUCT-RULES", "Standing-Orders-for-Hostel")),
    ("If I leave university, do I get my tuition money back?", "EN", ("Fee-Refund-Policy",)),
    ("fees wapas kab milegi?", "HI", ("Fee-Refund-Policy",)),
    ("Is any interest paid on a fee refund?", "EN", ("Fee-Refund-Policy",)),
    ("What happens if I skip too many lectures?", "EN", ("Academic-Rules-for-SCSET_UG", "Academic-Hand-Book-SOL")),
    ("Meri attendance kam hai, exam de sakta hu?", "HI", (
        "Academic-Rules-for-SCSET_UG", "Academic-Hand-Book-SOL", "Academic-Rules-for-SCSET_PGDual",
        "BAJMC-Rules", "BA_Liberal_Arts", "BBA-rules", "MBA-Rules", "SEAS-Ord_Rules",
        "MTech2018", "studentHandbookBennett.pdf", "PH.D.-PROGRAMS",
    )),
    ("How do I join the university wireless network?", "EN", ("IT_Policy",)),
    ("wifi kaise connect karu?", "HI", ("IT_Policy",)),
    ("Where can I report sexual harassment?", "EN", ("Posh_",)),
    ("koi harass kar raha hai, complaint kaha karu?", "HI", ("Posh_",)),
    ("What should I do if someone forces me to pay for ragging?", "EN", ("Anti-Ragging",)),
    ("How do I bake a chocolate cake?", "OOD", ()),
    ("Does Bennett have a policy for studying on the moon?", "OOD", ()),
]
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def _fts_results(index: LocalFTSIndex, question: str) -> list[dict]:
    results = []
    for topic, department in _topic_queries(question):
        try:
            results.extend(index.search(topic, department=department, limit=10)["results"])
        except ValueError as exc:
            if "policy topic" not in str(exc) and "policy-domain topic" not in str(exc):
                raise
    return results


def _hybrid_results(index: LocalFTSIndex, semantic: SemanticIndex, question: str,
                    threshold: float) -> list[dict]:
    results = []
    for topic, department in _topic_queries(question):
        search_topic = normalize_policy_query(topic)
        keyword = []
        dense = []
        departments = [department]
        if department == "ACADEMICS":
            departments.append("COURSE_SPECIFIC")
        elif department == "COURSE_SPECIFIC":
            departments.append("ACADEMICS")
        for candidate in departments:
            try:
                keyword.extend(index.search(search_topic, department=candidate, limit=10)["results"])
            except ValueError as exc:
                if "policy topic" not in str(exc) and "policy-domain topic" not in str(exc):
                    raise
            if candidate:
                dense.extend(semantic.search(search_topic, department=candidate, limit=20))
        if not keyword and department:
            try:
                keyword = index.search(search_topic, department=None, limit=10)["results"]
            except ValueError as exc:
                if "policy topic" not in str(exc) and "policy-domain topic" not in str(exc):
                    raise
        results.extend(reciprocal_rank_fusion(keyword, dense, limit=8, min_similarity=threshold))
    unique = {}
    for item in results:
        citation = item["citation"]
        unique[(citation["document_id"], citation["chunk_id"])] = item
    return list(unique.values())


def _expected_hit(results: list[dict], expected: tuple[str, ...], rank_limit: int = 5) -> int:
    if not expected:
        return int(not results[:rank_limit])
    for rank, item in enumerate(results[:rank_limit], 1):
        path = item["citation"]["file_path"].casefold()
        if any(token.casefold() in path for token in expected):
            return rank
    return 0


def _metrics(rows: list[dict], name: str) -> dict:
    positive = [row for row in rows if row["expected"]]
    out_of_domain = [row for row in rows if not row["expected"]]
    hits = sum(bool(row[name]) for row in positive)
    hinglish = [row for row in positive if row["language"] == "HI"]
    return {
        "positive_hit_at_5": f"{hits}/{len(positive)}",
        "positive_recall_at_5": round(hits / max(len(positive), 1), 3),
        "hinglish_hit_at_5": f"{sum(bool(row[name]) for row in hinglish)}/{len(hinglish)}",
        "out_of_domain_false_positives": sum(bool(row.get("hybrid_results" if name == "hybrid_rank" else "fts_results"))
                                              for row in out_of_domain),
    }


def main() -> None:
    index = LocalFTSIndex(DB)
    semantic = SemanticIndex(DEFAULT_VECTOR_PATH, expected_db_path=DB)
    base = []
    for question, language, expected in CASES:
        results = _fts_results(index, question)
        base.append({"question": question, "language": language, "expected": expected,
                     "fts_rank": _expected_hit(results, expected), "fts_results": results})

    sweeps = []
    for threshold in THRESHOLDS:
        rows = []
        for item in base:
            fused = _hybrid_results(index, semantic, item["question"], threshold)
            rows.append({**item, "hybrid_rank": _expected_hit(fused, item["expected"]),
                         "hybrid_results": fused})
        sweeps.append({"threshold": threshold, "metrics": _metrics(rows, "hybrid_rank"), "rows": rows})

    eligible = [item for item in sweeps if item["metrics"]["out_of_domain_false_positives"] == 0]
    chosen = max(eligible or sweeps,
                 key=lambda item: (item["metrics"]["positive_recall_at_5"],
                                   -item["metrics"]["out_of_domain_false_positives"],
                                   item["threshold"]))
    chosen_rows = chosen["rows"]
    for row in chosen_rows:
        row["fts_top"] = [x["citation"]["file_path"] for x in row["fts_results"][:3]]
        row["hybrid_top"] = [x["citation"]["file_path"] for x in row["hybrid_results"][:3]]
        for key in ("fts_results", "hybrid_results"):
            row.pop(key, None)
    evidence_status, detail = verify_requested_detail(
        "Hostel curfew time", _hybrid_results(index, semantic, "Hostel curfew time", chosen["threshold"])
    )
    report = {
        "method": "SQLite FTS5 vs FTS5 + multilingual-e5-small + reciprocal-rank fusion",
        "model_id": semantic.metadata["model_id"],
        "eligible_chunks": semantic.metadata["record_count"],
        "case_count": len(CASES),
        "fts_only_metrics": _metrics(base, "fts_rank"),
        "chosen_semantic_threshold": chosen["threshold"],
        "hybrid_metrics": chosen["metrics"],
        "hostel_curfew_evidence_status": evidence_status,
        "hostel_curfew_requested_detail": detail,
        "cases": chosen_rows,
        "threshold_sweep": [{"threshold": row["threshold"], "metrics": row["metrics"]} for row in sweeps],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "eligible_chunks", "case_count", "fts_only_metrics", "chosen_semantic_threshold",
        "hybrid_metrics", "hostel_curfew_evidence_status", "hostel_curfew_requested_detail",
    )}, ensure_ascii=False, indent=2))
    for row in chosen_rows:
        print(f"{row['language']} | FTS={row['fts_rank'] or '-'} | Hybrid={row['hybrid_rank'] or '-'} | {row['question']}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
