"""Demo web API on top of the router. Routing logic lives in router.py; this file only serves it.

Run: uvicorn yiti_ai.api:app --reload    then open http://127.0.0.1:8000
"""
import csv
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import dotenv_values, load_dotenv

from .config import DATA, ROOT, load_config
from .local_retrieval import (LocalFTSIndex, LocalIndexError, department_name,
                              normalize_policy_query, query_terms)
from .providers import ProviderError, QuotaError, provider_error_category, provider_error_types
from .router import Router
from .semantic_retrieval import (DEFAULT_VECTOR_PATH, SemanticIndex, SemanticIndexError,
                                 reciprocal_rank_fusion)

logger = logging.getLogger(__name__)

STATIC = ROOT / "static"
# Sample messages shown in the demo: dev set only, never test data (CLAUDE.md).
SAMPLE_IDS = ["D026", "D031", "D017", "D024"]


class RouteRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    department: str | None = Field(default=None, max_length=40)
    limit: int = Field(default=5, ge=1, le=10)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    department: str | None = Field(default=None, max_length=40)
    history: list[dict] = Field(default_factory=list, max_length=8)


DEPARTMENT_TERMS = {
    "FINANCE": r"\b(fee|fees|refund|reimburse(?:ment|ments)?|scholarship|payment|tuition|receipt|installment|installments|"
               r"(?:wa?pas|va?apas)\s+fees?|fees?\s+(?:wa?pas|va?apas|back)|money back)\b",
    "HOSTEL": r"\b(hostel|warden|curfew|in.time|roommate|room partner|leave from hostel|come back late|return late|"
              r"gate.*(?:closing|time)|kab tak wapas)\b",
    "IT": r"\b(it\s+(?:policy|security|support|services?)|information technology|wifi|wi.fi|wireless|internet|password|login|email|lms|erp|portal|account)\b",
    "COURSE_SPECIFIC": r"\b(scset|seas|sol|mba|bajmc|bba|course.rules|academic rules)\b",
    "ACADEMICS": r"\b(attendance|exam|examination|semester|academic calendar|registrar|course registration|"
                 r"lectures?|classes?|miss(?:ed|ing)?|skip(?:ped|ping)?|eligibility)\b",
    "STUDENT_SUPPORT": r"\b(anti.ragging|ragging|posh|harass(?:ment|ing)?|wellbeing|counsell?ing|grievance|complaints?)\b",
    "CAMPUS_SERVICES": r"\b(mess|food|transport|bus|shuttle|library|sports|maintenance|cleaning|lost and found)\b",
    "EMPLOYEES": r"\b(employee|staff|faculty|payroll|hr manual)\b",
}

SEMANTIC_MIN_SIMILARITY = 0.75


def _department_family(department: str | None) -> list[str | None]:
    """Allow broad academic intent to reach course-specific rules in the corpus."""
    if department == "ACADEMICS":
        return ["ACADEMICS", "COURSE_SPECIFIC"]
    if department == "COURSE_SPECIFIC":
        return ["COURSE_SPECIFIC", "ACADEMICS"]
    return [department]


def _topic_queries(message: str) -> list[tuple[str, str | None]]:
    clauses = [part.strip() for part in re.split(r"\b(?:and|aur|also)\b|[;]", message, flags=re.I) if part.strip()]
    expanded_clauses = []
    for clause in clauses:
        comma_parts = [part.strip() for part in clause.split(",") if part.strip()]
        part_domains = [
            {dept for dept, pattern in DEPARTMENT_TERMS.items() if re.search(pattern, part, re.I)}
            for part in comma_parts
        ]
        # A comma usually joins a single question ("attendance low, can I sit?").
        # Split it only when the parts independently name different domains.
        if len(comma_parts) > 1 and all(part_domains) and len(set.union(*part_domains)) > 1:
            expanded_clauses.extend(comma_parts)
        else:
            expanded_clauses.append(clause)
    clauses = expanded_clauses
    mapped = []
    for clause in clauses:
        departments = [dept for dept, pattern in DEPARTMENT_TERMS.items() if re.search(pattern, clause, re.I)]
        if departments:
            mapped.extend((clause, dept) for dept in departments)
    if mapped:
        return mapped
    all_departments = [dept for dept, pattern in DEPARTMENT_TERMS.items() if re.search(pattern, message, re.I)]
    return [(message, dept) for dept in all_departments] or [(message, None)]


DETAIL_REQUESTS = {
    "clock_time": re.compile(r"\b(curfew|in[ -]?time|closing time|what time|at what time|gate.*time|"
                             r"timings?|kitne baje|kab tak|come back late|return late)\b", re.I),
    "amount": re.compile(r"\b(how much|what amount|how many rupees|cost|price|amount|kitna|kitne paise)\b", re.I),
    "deadline": re.compile(r"\b(deadline|due date|last date|by what date|submit by|kab tak)\b", re.I),
    "duration": re.compile(r"\b(how long|how many days|within how many|kitne din|kitne time tak)\b", re.I),
}
EXPLICIT_VALUES = {
    "clock_time": re.compile(r"\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)\b|"
                              r"\b\d{1,2}:[0-5]\d\b|\b(?:midnight|noon)\b", re.I),
    "amount": re.compile(r"(?:₹|\b(?:rs\.?|inr|rupees?)\s*)[\d,]+(?:\.\d+)?|"
                         r"\b[\d,]+(?:\.\d+)?\s*(?:rupees?|inr|rs\.?|percent|%)\b", re.I),
    "deadline": re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
                           r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
                           r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|"
                           r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", re.I),
    "duration": re.compile(r"\b\d+\s+(?:working\s+)?(?:days?|weeks?|months?|years?|hours?)\b", re.I),
}
DETAIL_CONTEXT = {
    "clock_time": re.compile(r"\b(hostel|curfew|return|in[ -]?time|gate|late|stipulated)\b", re.I),
    "amount": re.compile(r"\b(fee|refund|tuition|scholarship|payment|fine|charge|deposit|cost)\b", re.I),
    "deadline": re.compile(r"\b(application|fee|refund|registration|submit|deadline|due|admission|withdrawal)\b", re.I),
    "duration": re.compile(r"\b(refund|registration|application|leave|hostel|processing|submit|withdrawal)\b", re.I),
}


def verify_requested_detail(query: str, results: list[dict]) -> tuple[str, str | None]:
    """Check that the requested concrete value type occurs in retrieved passages."""
    detail = next((name for name, pattern in DETAIL_REQUESTS.items() if pattern.search(query)), None)
    if detail is None:
        return "relevant_document_found", None
    for item in results:
        evidence = item.get("evidence", "")
        if EXPLICIT_VALUES[detail].search(evidence) and DETAIL_CONTEXT[detail].search(evidence):
            return "exact_answer_supported", detail
    return "related_policy_found_detail_missing", detail


def create_app(router: Router | None = None, *, local_index_path: str | Path | None = None,
               retrieval_only: bool | None = None, answer_provider=None) -> FastAPI:
    """Build the app. Retrieval-only mode does not construct the model-backed Router."""
    # Prefer the standalone project's own credentials/settings over inherited
    # machine variables. Never consult another project's environment.
    project_env = dotenv_values(ROOT / ".env")
    load_dotenv(ROOT / ".env", override=True)
    state = {"router": router, "answer_provider": answer_provider, "semantic_index": None}
    if retrieval_only is None:
        retrieval_only = router is None and os.environ.get("YITI_MODE", "offline").strip().lower() in {
            "offline", "retrieval_only", "retrieval-only",
        }
    index_path = local_index_path or os.environ.get("YITI_LOCAL_INDEX") or (ROOT / "data" / "processed" / "search.sqlite3")
    local_index = LocalFTSIndex(index_path) if index_path else None
    if local_index:
        try:
            state["semantic_index"] = SemanticIndex(DEFAULT_VECTOR_PATH, expected_db_path=index_path)
        except SemanticIndexError as exc:
            logger.info("Local semantic retrieval disabled: %s", str(exc))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["router"] is None and not retrieval_only:
            state["router"] = Router()
        yield

    app = FastAPI(title="Yiti AI demo", lifespan=lifespan)
    origins = [origin.strip() for origin in os.environ.get(
        "WEB_ORIGIN", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    ).split(",") if origin.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        configured_answer_mode = str(project_env.get("YITI_ANSWER_MODE") or "retrieval").strip().lower()
        gemini_enabled = str(project_env.get("YITI_GEMINI_ENABLED") or "false").strip().lower() == "true"
        grounded_answers_enabled = str(project_env.get("YITI_GEMINI_GROUNDED_ANSWERS_ENABLED") or "false").strip().lower() == "true"
        gemini_configured = bool(str(project_env.get("GEMINI_API_KEY") or "").strip())
        if retrieval_only:
            index_error = None
            try:
                index_health = local_index.health() if local_index else {"available": False, "documents": 0, "chunks": 0}
            except LocalIndexError as exc:
                index_health = {"available": False, "documents": 0, "chunks": 0}
                index_error = str(exc)
            semantic = state["semantic_index"]
            semantic_health = {
                "available": semantic is not None,
                "model": semantic.metadata.get("model_id") if semantic is not None else None,
                "chunks": semantic.metadata.get("record_count", 0) if semantic is not None else 0,
            }
            return {"status": "ok" if index_health["available"] else "degraded",
                    "mode": "retrieval_only", "model_provider": "none", "local_index": index_health,
                    "semantic_index": semantic_health,
                    "answer_mode": "gemini" if configured_answer_mode == "gemini" and gemini_enabled and gemini_configured else "retrieval",
                    "gemini_configured": gemini_configured,
                    "gemini_enabled": gemini_enabled,
                    "gemini_grounded_answers_enabled": grounded_answers_enabled,
                    "error": index_error or ("Set YITI_LOCAL_INDEX to a readable search.sqlite3" if not index_health["available"] else None)}
        r = state["router"]
        th = r.cfg["thresholds"]
        return {"status": "ok", "model": r.provider.chat_model, "embed_model": r.provider.embed_model,
                "thresholds": {k: th.get(k) for k in ("t_high", "t_low", "frozen", "tuned_on")},
                "confidence_weights": r.cfg["confidence"], "knn_k": r.knn.k}

    @app.get("/samples")
    def samples() -> list[dict]:
        with open(DATA / "dev_set.csv", encoding="utf-8-sig", newline="") as f:
            dev = {row["id"]: row["question"] for row in csv.DictReader(f)}
        return [{"id": i, "question": dev[i]} for i in SAMPLE_IDS]

    @app.post("/route")
    def route(req: RouteRequest) -> dict:
        """RouteResult JSON, plus example_texts: the text of every nearest example shown in the result."""
        if retrieval_only:
            raise HTTPException(409, "LLM routing is disabled in retrieval-only mode; use POST /retrieve for source evidence.")
        message = req.message.strip()
        if not message:
            raise HTTPException(422, "message is empty")
        r = state["router"]
        try:
            result = r.route(message)
        except QuotaError as e:
            raise HTTPException(429, str(e))
        except ProviderError as e:
            raise HTTPException(502, str(e))
        texts = {e["id"]: e["text"] for e in r.knn.examples}
        neighbour_ids = {n.id for t in result.topics for n in t.neighbours}
        return {**result.model_dump(), "example_texts": {i: texts[i] for i in sorted(neighbour_ids)}}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        """Search the local preprocessed index and return evidence plus source citations, without generating an answer."""
        if local_index is None:
            raise HTTPException(503, "Local index is not configured. Set YITI_LOCAL_INDEX to processed-data/search.sqlite3.")
        try:
            return local_index.search(req.query, department=req.department, limit=req.limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LocalIndexError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        """Local-first student flow. Returns retrieved evidence and citations, not a generated policy answer."""
        if local_index is None:
            raise HTTPException(503, "Local index is not configured. Set YITI_LOCAL_INDEX to processed-data/search.sqlite3.")
        message = req.message.strip()
        if not message:
            raise HTTPException(422, "message is empty")
        try:
            prior_user = next((turn.get("content", "") for turn in reversed(req.history)
                               if turn.get("role") == "user" and isinstance(turn.get("content"), str)), "")
            retrieval_query = message
            if prior_user and re.search(r"\b(what about|how about|and the|when is it|those|that policy)\b", message, re.I):
                retrieval_query = f"{prior_user} {message}"
            topics = [(retrieval_query, req.department)] if req.department else _topic_queries(retrieval_query)
            results, departments = [], []
            match_modes = []
            for topic, department in topics:
                search_topic = normalize_policy_query(topic)
                keyword_results = []
                semantic_results = []
                semantic_index = state["semantic_index"]
                topic_matched = False
                for candidate_department in _department_family(department):
                    try:
                        found = local_index.search(search_topic, department=candidate_department, limit=10)
                        keyword_results.extend(found["results"])
                        if found["results"]:
                            topic_matched = True
                            if found["match_mode"] != "none":
                                match_modes.append("keyword")
                    except ValueError as exc:
                        if "policy topic" not in str(exc) and "policy-domain topic" not in str(exc):
                            raise
                    if semantic_index is not None and candidate_department:
                        try:
                            semantic_results.extend(semantic_index.search(
                                search_topic, department=department_name(candidate_department), limit=20,
                                min_similarity=SEMANTIC_MIN_SIMILARITY,
                            ))
                        except SemanticIndexError as exc:
                            logger.warning("Semantic retrieval disabled for this request: %s", str(exc))
                            state["semantic_index"] = None
                # Do not run unconstrained dense search for text with no identifiable
                # policy domain. E5 cosine scores are not a reliable out-of-domain detector.
                if semantic_index is not None and department is None and topic_matched:
                    try:
                        semantic_results.extend(semantic_index.search(
                            search_topic, limit=20, min_similarity=SEMANTIC_MIN_SIMILARITY,
                        ))
                    except SemanticIndexError as exc:
                        logger.warning("Semantic retrieval disabled for this request: %s", str(exc))
                        state["semantic_index"] = None
                if semantic_results:
                    match_modes.append("semantic")
                fused = reciprocal_rank_fusion(
                    keyword_results, semantic_results, limit=8,
                    min_similarity=SEMANTIC_MIN_SIMILARITY,
                )
                for item in fused:
                    key = (item["citation"]["document_id"], item["citation"]["chunk_id"])
                    if not any((old["citation"]["document_id"], old["citation"]["chunk_id"]) == key
                               for old in results):
                        results.append(item)
                        departments.append(item["citation"]["department"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LocalIndexError as exc:
            raise HTTPException(503, str(exc)) from exc
        if not results:
            clarification = len(query_terms(message)) < 2
            return {"message": message, "status": "clarification_needed" if clarification else "no_policy_match", "answer":
                    ("Could you add a little more detail, such as the department or policy you mean?" if clarification else
                     "I couldn't find a matching public policy passage. Please rephrase or contact the relevant campus office."),
                    "departments": [], "citations": [], "evidence": [],
                    "evidence_status": "no_reliable_evidence", "retrieval_method": "none"}
        departments = list(dict.fromkeys(departments))
        evidence_status, requested_detail = verify_requested_detail(message, results)
        retrieval_method = "+".join(dict.fromkeys(
            source for item in results for source in item.get("retrieval_source", "").split("+") if source))
        answer = "Retrieved policy evidence is shown below. Please check the cited source for the complete and current policy."
        answer_mode, generation_warning = "retrieval", None
        gemini_requested = str(project_env.get("YITI_ANSWER_MODE") or "retrieval").strip().lower() == "gemini"
        gemini_enabled = str(project_env.get("YITI_GEMINI_ENABLED") or "false").strip().lower() == "true"
        grounded_answers_enabled = str(project_env.get("YITI_GEMINI_GROUNDED_ANSWERS_ENABLED") or "false").strip().lower() == "true"
        gemini_active = gemini_requested and gemini_enabled and grounded_answers_enabled
        if gemini_requested and not gemini_enabled:
            generation_warning = "Gemini generation is disabled in YITI_GEMINI_ENABLED"
        elif gemini_requested and not grounded_answers_enabled:
            generation_warning = "Grounded Gemini generation is disabled in YITI_GEMINI_GROUNDED_ANSWERS_ENABLED"
        if gemini_active:
            try:
                provider = state["answer_provider"]
                if provider is None:
                    cfg = load_config()
                    from .providers import Provider
                    # Pin this chat flow to the project's own Gemini key and config;
                    # never select Azure or inherited machine/other-project credentials.
                    provider_env = {**project_env, "GEMINI_API_KEY": project_env.get("GEMINI_API_KEY"),
                                    "LLM_PROVIDER": "gemini"}
                    provider = state["answer_provider"] = Provider(cfg, env=provider_env)
                evidence_text = "\n\n".join(
                    f"[{i}] Department: {item['citation']['department']}\nSource: {item['citation']['label']}\n"
                    f"Date: {item['citation']['policy_date'] or 'not supplied'}\nPassage: {item['evidence'][:1800]}"
                    for i, item in enumerate(results[:5], 1))
                history = [{"role": turn["role"], "content": turn["content"][:1000]}
                           for turn in req.history[-6:] if turn.get("role") in {"user", "assistant"}
                           and isinstance(turn.get("content"), str)]
                answer = provider.chat_text([
                    {"role": "system", "content": (
                        "You are Yiti, a helpful Bennett University policy assistant. Answer only from the supplied "
                        "retrieved passages. Do not infer or invent university rules. If the passages do not answer "
                        "the question, say the policy detail was not found. A semantically related passage is not "
                        "proof that a requested detail exists. Before giving any requested clock time, amount, date, "
                        "deadline, or duration, confirm that the value appears explicitly in the supplied passages; "
                        "if absent, state that it is not specified and give only the related supported policy. "
                        "For multiple topics, answer each separately. "
                        "Use simple English; understand Hinglish. Add inline [1], [2] references matching the supplied "
                        "passage numbers. Never claim a source says more than its passage." )},
                    *history,
                    {"role": "user", "content": f"Question: {message}\n\nRetrieved evidence:\n{evidence_text}"},
                ])
                answer_mode = "gemini"
            except Exception as exc:
                generation_warning = provider_error_category(exc)
                types = ">".join(provider_error_types(exc)) or "unknown"
                logger.warning("Gemini answer request failed category=%s exception_chain=%s",
                               generation_warning, types)
                answer = "I couldn't generate an answer from Gemini right now. The retrieved policy passages and source links are shown below."
        return {"message": message, "status": "evidence_found", "answer": answer,
                "departments": departments,
                "citations": [item["citation"] for item in results[:8]],
                "evidence": results[:8], "match_mode": match_modes,
                "answer_mode": answer_mode, "generation_warning": generation_warning,
                "evidence_status": evidence_status, "requested_detail": requested_detail,
                "retrieval_method": retrieval_method or "keyword",
                "generation_error": generation_warning if answer_mode != "gemini" and gemini_active else None}

    return app


app = create_app()
