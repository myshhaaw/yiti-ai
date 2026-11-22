"""Read-only search over the preprocessed local SQLite FTS5 index.

This module never creates or modifies the index and has no model/cloud dependencies.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

DEPARTMENTS = {
    "IT", "FINANCE", "ACADEMICS", "COURSE_SPECIFIC", "HOSTEL", "CAMPUS_SERVICES",
    "EMPLOYEES", "STUDENT_SUPPORT",
}
ALIASES = {"FEES": "FINANCE", "CAMPUS-SERVICES": "CAMPUS_SERVICES", "STUDENT-SUPPORT": "STUDENT_SUPPORT"}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "can", "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "the", "to", "what", "when", "where", "which", "who",
    "why", "with", "would", "you", "your", "get", "find", "tell", "show", "explain", "know",
    "check", "look", "search", "give", "provide", "need", "want", "about", "tell",
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "mein", "me", "mujhe", "mera", "meri",
    "kaise", "kese", "karu", "kare", "karna", "please", "kab", "tak", "kaha", "kahan",
}
BROAD_TERMS = {"policy", "policies", "rule", "rules", "campus", "university", "bennett", "hostel",
               "time", "times", "timing", "timings", "hour", "hours"}
DOMAIN_TERMS = {
    "fee", "fees", "refund", "reimbursement", "tuition", "scholarship", "hostel", "warden", "curfew",
    "travel", "wifi", "wi", "wireless", "internet", "it", "academic", "attendance", "exam", "examination",
    "ragging", "grievance", "grievances", "complaint", "complaints", "support", "counselling", "counseling",
    "harassment", "library", "transport", "bus", "shuttle", "mess", "security", "password", "lms", "erp",
    "portal", "withdrawal", "withdraw", "leave", "in.time", "phd", "program", "regulations", "queries",
    "query", "online", "professor", "practice", "faculty", "conduct", "standing", "orders",
    "lecture", "lectures", "class", "classes", "absence", "absent", "eligibility", "deadline",
}

# High-confidence wording used in the policy corpus. These alternatives affect
# retrieval only; answers still have to be supported by the returned passages.
TERM_ALIASES = {
    "curfew": ("in-time", "in time", "return time", "night return", "stipulated time"),
    "refund": ("reimbursement", "return of fees"),
    "reimbursement": ("refund", "fee return"),
    "scholarship": ("fellowship", "financial aid"),
    "wifi": ("wi-fi", "wireless", "internet connection"),
    "internet": ("wifi", "wi-fi", "wireless"),
    "grievance": ("complaint", "complaints"),
    "complaint": ("grievance", "grievances"),
    "transport": ("bus", "shuttle"),
    "bus": ("transport", "shuttle"),
    "shuttle": ("transport", "bus"),
    "wireless": ("wifi", "wi-fi", "internet"),
}

PHRASE_NORMALIZATIONS = (
    (re.compile(r"\bin[\s-]+time\b", re.I), "curfew"),
    (re.compile(r"\bfinancial aid\b", re.I), "scholarship"),
    (re.compile(r"\bfee reimbursement\b", re.I), "fee refund"),
    (re.compile(r"\bstudent complaints?\b", re.I), "student grievance"),
)

# A compact query normalizer for common Romanized Hindi and paraphrase forms.
# Dense retrieval remains the primary language bridge; these rules only give the
# existing lexical retriever useful policy vocabulary when transliteration is
# outside the embedding model's strongest coverage.
QUERY_NORMALIZATIONS = (
    (re.compile(r"\b(?:tuition|fees?|money)\b.*\bback\b.*", re.I), "tuition refund"),
    (re.compile(r"\bhostel\b.*\b(?:kab tak|wa?pas|va?apas)\b.*", re.I), "hostel curfew return in-time"),
    (re.compile(r"\bfees?\s+(?:wa?pas|va?apas)\b.*", re.I), "fee refund"),
    (re.compile(r"\b(?:wa?pas|va?apas)\s+fees?\b.*", re.I), "fee refund"),
    (re.compile(r"\b(?:attendance\s+(?:kam|low|less)|miss(?:ed|ing)?\s+(?:too\s+many\s+)?(?:lectures?|classes?)|skip(?:ped|ping)?\s+(?:too\s+many\s+)?(?:lectures?|classes?))\b.*", re.I), "attendance exam eligibility"),
    (re.compile(r"\b(?:come back late|return late|gate closing time)\b", re.I), "hostel curfew return time"),
)


def normalize_policy_query(query: str) -> str:
    """Normalize a few broad, common phrase patterns without replacing the original text."""
    normalized = query.strip()
    for pattern, replacement in QUERY_NORMALIZATIONS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


class LocalIndexError(RuntimeError):
    """The configured local index is missing, unreadable, or not the expected FTS5 database."""


def department_name(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    name = re.sub(r"\s+", "_", value.strip().upper())
    name = ALIASES.get(name, name)
    if name not in DEPARTMENTS:
        raise ValueError("unknown department; choose IT, FINANCE, ACADEMICS, COURSE_SPECIFIC, HOSTEL, "
                         "CAMPUS_SERVICES, EMPLOYEES, or STUDENT_SUPPORT")
    return name


def _variants(term: str) -> list[str]:
    """Small plural normalization, keeping the original token so inflections remain searchable."""
    variants = [term]
    if len(term) > 4 and term.endswith("ies"):
        variants.append(term[:-3] + "y")
    elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        variants.append(term[:-1])
        if term.endswith("es"):
            variants.append(term[:-2])
    return list(dict.fromkeys(variants))


def query_terms(query: str) -> list[str]:
    for pattern, replacement in PHRASE_NORMALIZATIONS:
        query = pattern.sub(replacement, query)
    query = normalize_policy_query(query)
    out = []
    for term in TOKEN_RE.findall(query.casefold()):
        if term not in STOP_WORDS and (len(term) > 1 or not term.isascii()):
            out.append(term)
    return list(dict.fromkeys(out))


def _group(term: str) -> str:
    values = [*_variants(term), *TERM_ALIASES.get(term, ())]
    return "(" + " OR ".join('"' + v.replace('"', '""') + '"' for v in dict.fromkeys(values)) + ")"


def _term_matches(term: str, searchable: set[str]) -> bool:
    """Match a query concept against its inflections and policy-language aliases."""
    values = [*_variants(term), *TERM_ALIASES.get(term, ())]
    for value in values:
        words = TOKEN_RE.findall(value.casefold())
        if words and all(word in searchable for word in words):
            return True
    return False


class LocalFTSIndex:
    """Search a processed-data/search.sqlite3 file through read-only SQLite connections."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise LocalIndexError("local search index is unavailable")
        uri = "file:" + quote(self.path.as_posix(), safe="/:_") + "?mode=ro"
        try:
            con = sqlite3.connect(uri, uri=True, timeout=2.0)
            con.row_factory = sqlite3.Row
            con.execute("SELECT count(*) FROM chunks").fetchone()
            con.execute("SELECT count(*) FROM documents").fetchone()
            return con
        except sqlite3.Error as exc:
            raise LocalIndexError("local search index could not be opened as an FTS5 database") from exc

    def health(self) -> dict:
        with closing(self._connect()) as con:
            docs = con.execute("SELECT count(*) FROM documents WHERE processing_status LIKE 'indexed%'").fetchone()[0]
            chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"available": True, "documents": docs, "chunks": chunks}

    @staticmethod
    def _search_sql(expression: str, department: str | None, limit: int, has_quality: bool = False) -> str:
        dept_clause = " AND c.department = ?" if department else ""
        quality_select = (", ep.method AS extraction_method, ep.ocr_confidence AS ocr_confidence, "
                          "ep.suspicious_output AS ocr_suspicious" if has_quality else
                          ", NULL AS extraction_method, NULL AS ocr_confidence, NULL AS ocr_suspicious")
        quality_join = (" LEFT JOIN extraction_pages AS ep ON ep.document_id = c.document_id AND ep.page = c.page"
                        if has_quality else "")
        return (
            "SELECT c.rowid AS chunk_id, c.content, c.title, c.department, c.heading, c.document_id, c.file_path, "
            "c.source_url, c.page, c.policy_date, c.course" + quality_select + ", bm25(chunks, 1.0, 4.0) AS rank "
            "FROM chunks AS c JOIN documents AS d ON d.document_id = c.document_id "
            + quality_join + " "
            "WHERE chunks MATCH ? AND d.privacy_flags_json = '[]' AND d.processing_status LIKE 'indexed%'"
            + dept_clause + " ORDER BY rank ASC, c.title ASC, c.page ASC LIMIT ?"
        )

    def search(self, query: str, *, department: str | None = None, limit: int = 5) -> dict:
        clean = query.strip()
        terms = query_terms(clean)
        dept = department_name(department)
        # "IT" is otherwise treated as the English pronoun stop word. With an
        # explicit IT department filter it is a meaningful domain token.
        if dept == "IT" and re.search(r"\bit\b", clean, re.I) and "it" not in terms:
            terms.append("it")
        if not terms:
            raise ValueError("query must contain at least one searchable word")
        relevance_terms = [term for term in terms if term not in BROAD_TERMS]
        if not relevance_terms:
            relevance_terms = [term for term in terms if term in {"hostel", "wifi", "policy", "policies", "rule", "rules"}]
        if not relevance_terms:
            raise ValueError("query needs a policy topic, not only general campus terms")
        # Do not turn generic overlaps (e.g. "policy" + "studying") into
        # policy evidence. Require at least one known policy/domain concept.
        if not any(term in DOMAIN_TERMS for term in terms):
            return {"query": clean, "department": dept, "match_mode": "none",
                    "answer": None, "answer_status": "retrieval_only_no_generated_answer",
                    "results": [], "message": "No policy-domain topic was identified in this query."}
        domain_terms = [term for term in terms if term in DOMAIN_TERMS]
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")

        variants = [_group(term) for term in terms]
        expressions: list[tuple[str, str]] = []
        if len(terms) >= 2:
            phrase = " ".join(terms)
            expressions.append(('"' + phrase.replace('"', '""') + '"', "phrase"))
        expressions.append((" AND ".join(variants), "all_terms"))
        if len(terms) >= 2:
            expressions.append((" OR ".join(variants), "partial_terms"))

        with closing(self._connect()) as con:
            has_quality = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='extraction_pages'"
            ).fetchone() is not None
            rows = []
            match_mode = "none"
            for expression, mode in expressions:
                try:
                    params = [expression]
                    if dept:
                        params.append(dept)
                    params.append(max(limit * 8, 24) if mode == "partial_terms" else max(limit * 4, 12))
                    rows = con.execute(self._search_sql(expression, dept, params[-1], has_quality), params).fetchall()
                except sqlite3.OperationalError as exc:
                    raise LocalIndexError("local search query failed") from exc
                if rows:
                    match_mode = mode
                    break

        ranked = []
        for row in rows:
            searchable = set(TOKEN_RE.findall((row["title"] + " " + row["content"]).casefold()))
            matched = [term for term in terms if _term_matches(term, searchable)]
            relevant_matched = [term for term in relevance_terms if _term_matches(term, searchable)]
            coverage = len(relevant_matched) / len(relevance_terms)
            if not any(_term_matches(term, searchable) for term in domain_terms):
                continue
            # A loose OR fallback is only useful when it matches most of the query;
            # suppressing weak overlaps prevents unrelated fee/handbook passages
            # from appearing as evidence for an unavailable policy.
            if match_mode == "partial_terms" and coverage < 0.60:
                continue
            page = row["page"]
            heading = row["heading"] or None
            label = row["title"] + (f", p. {page}" if page else (f" — {heading}" if heading else ""))
            citation = {
                "title": row["title"], "source_url": row["source_url"] or None,
                "department": row["department"], "file_path": row["file_path"],
                "page": page, "heading": heading, "course": row["course"],
                "policy_date": row["policy_date"], "label": label,
                "extraction_method": row["extraction_method"], "ocr_confidence": row["ocr_confidence"],
                "ocr_suspicious": bool(row["ocr_suspicious"]) if row["ocr_suspicious"] is not None else None,
                "complete": bool(row["source_url"] and (page is not None or heading)),
                "document_id": row["document_id"], "chunk_id": row["chunk_id"],
            }
            ranked.append({"evidence": row["content"], "citation": citation,
                           "matched_terms": matched, "term_coverage": round(coverage, 3),
                           "semantic_score": None, "retrieval_source": "keyword",
                           "_rank": row["rank"]})

        # Prefer broader query coverage first, then the FTS5 BM25 score (title weighted above body text).
        ranked.sort(key=lambda item: (-item["term_coverage"], item["_rank"], item["citation"]["title"]))
        for item in ranked:
            item.pop("_rank", None)
        ranked = ranked[:limit]
        return {"query": clean, "department": dept, "match_mode": match_mode,
                "answer": None, "answer_status": "retrieval_only_no_generated_answer",
                "results": ranked,
                "message": "These are retrieved source passages only. Yiti AI has not generated or verified an answer."}
