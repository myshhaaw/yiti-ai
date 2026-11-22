"""Local multilingual semantic index over public, indexed policy chunks.

The source SQLite database remains read-only. Vector IDs are SQLite chunk rowids,
and every vector carries the citation fields needed to verify and display evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .config import ROOT

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_DIMENSION = 384
SCHEMA_VERSION = 1
SEMANTIC_DIR = ROOT / ".cache" / "semantic"
MODEL_PATH_FILE = SEMANTIC_DIR / "model_snapshot.txt"
DEFAULT_VECTOR_PATH = Path(os.environ.get("YITI_SEMANTIC_INDEX", str(SEMANTIC_DIR / "policy-vectors.npz")))
BOILERPLATE_LINE = re.compile(
    r"^(?:BENNETT\s+UNIVERSITY|THE\s+TIMES\s+GROUP|A\s+TIMES\s+GROUP\s+INITIATIVE|"
    r"PAGE\s+\d+(?:\s+OF\s+\d+)?|\d+\s*/\s*\d+)\s*$",
    re.I,
)


class SemanticIndexError(RuntimeError):
    """The local embedding model or semantic index is unavailable or stale."""


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser().resolve()
    if not db_path.is_file():
        raise SemanticIndexError("policy SQLite index is unavailable")
    uri = "file:" + quote(db_path.as_posix(), safe="/:_") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error as exc:
        raise SemanticIndexError("policy SQLite index could not be opened read-only") from exc


def _embedding_text(row: sqlite3.Row) -> str:
    # Remove repeated document branding from the embedding input only. The original
    # chunk remains untouched in SQLite and is returned verbatim as evidence.
    body = "\n".join(line.strip() for line in row["content"].splitlines()
                      if line.strip() and not BOILERPLATE_LINE.fullmatch(line.strip()))
    parts = [f"Department: {row['department']}.", f"Policy: {row['title']}."]
    if row["heading"]:
        parts.append(f"Section: {row['heading']}.")
    parts.append(body)
    return " ".join(part for part in parts if part)


def eligible_records(db_path: str | Path) -> list[dict[str, Any]]:
    """Read only public indexed chunks; restricted documents never enter this query."""
    with closing(_connect_readonly(db_path)) as con:
        rows = con.execute(
            "SELECT c.rowid AS chunk_id, c.document_id, c.department, c.title, c.heading, "
            "c.file_path, c.source_url, c.page, c.policy_date, c.course, c.content "
            "FROM chunks AS c JOIN documents AS d ON d.document_id = c.document_id "
            "WHERE d.privacy_flags_json = '[]' AND d.processing_status LIKE 'indexed%' "
            "ORDER BY c.rowid"
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        content_hash = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
        record["content_hash"] = content_hash
        record["embedding_text"] = _embedding_text(row)
        records.append(record)
    return records


def records_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['chunk_id']}\0{record['document_id']}\0{record['content_hash']}\n".encode())
    return digest.hexdigest()


def load_model_snapshot() -> Path:
    try:
        snapshot = Path(MODEL_PATH_FILE.read_text(encoding="utf-8").strip()).resolve()
    except OSError as exc:
        raise SemanticIndexError("the local embedding model is not downloaded") from exc
    if not snapshot.is_dir():
        raise SemanticIndexError("the local embedding model snapshot is missing")
    return snapshot


def load_embedding_model(*, local_files_only: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SemanticIndexError("sentence-transformers is not installed") from exc
    try:
        model = SentenceTransformer(str(load_model_snapshot()), device="cpu", local_files_only=local_files_only)
    except Exception as exc:
        raise SemanticIndexError("the local multilingual embedding model could not be loaded") from exc
    return model


def build_semantic_index(db_path: str | Path, output_path: str | Path = DEFAULT_VECTOR_PATH,
                         *, model=None, batch_size: int = 24) -> dict[str, Any]:
    """Embed eligible chunks into a compact NumPy matrix and save atomically."""
    records = eligible_records(db_path)
    if not records:
        raise SemanticIndexError("no public indexed policy chunks were found")
    model = model or load_embedding_model()
    passage_texts = ["passage: " + record["embedding_text"] for record in records]
    vectors = model.encode(
        passage_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(records), MODEL_DIMENSION):
        raise SemanticIndexError("embedding model returned an unexpected vector shape")
    if not np.isfinite(vectors).all():
        raise SemanticIndexError("embedding model returned non-finite vectors")

    metadata_records = [{key: value for key, value in record.items() if key != "embedding_text"}
                        for record in records]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "dimension": MODEL_DIMENSION,
        # Store a portable basename, not a machine-specific absolute path.
        "source_index": Path(db_path).name,
        "source_fingerprint": records_fingerprint(records),
        "record_count": len(records),
        "records": metadata_records,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, vectors=vectors, metadata=np.asarray(json.dumps(metadata)))
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": str(output), "records": len(records), "dimension": MODEL_DIMENSION,
            "source_fingerprint": metadata["source_fingerprint"]}


class SemanticIndex:
    """Read-only cosine search over the locally cached embedding matrix."""

    def __init__(self, vector_path: str | Path = DEFAULT_VECTOR_PATH, *, model=None,
                 expected_db_path: str | Path | None = None):
        self.path = Path(vector_path).expanduser().resolve()
        self.model = model
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                self.vectors = np.asarray(archive["vectors"], dtype=np.float32)
                self.metadata = json.loads(str(archive["metadata"].item()))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise SemanticIndexError("local semantic vector index is missing or corrupt") from exc
        if self.metadata.get("schema_version") != SCHEMA_VERSION or self.metadata.get("model_id") != MODEL_ID:
            raise SemanticIndexError("local semantic vector index has an unsupported schema or model")
        self.records = self.metadata.get("records") or []
        if self.vectors.shape != (len(self.records), MODEL_DIMENSION):
            raise SemanticIndexError("local semantic vector index metadata does not match its vectors")
        if expected_db_path and Path(self.metadata.get("source_index", "")).name != Path(expected_db_path).name:
            raise SemanticIndexError("semantic vectors were built from a different SQLite index")
        if expected_db_path:
            current_records = eligible_records(expected_db_path)
            if records_fingerprint(current_records) != self.metadata.get("source_fingerprint"):
                raise SemanticIndexError("semantic vectors are stale for the current SQLite policy index")

    def _model(self):
        if self.model is None:
            self.model = load_embedding_model()
        return self.model

    def search(self, query: str, *, department: str | None = None, limit: int = 10,
               min_similarity: float = 0.0) -> list[dict[str, Any]]:
        if not 1 <= limit <= 20:
            raise ValueError("semantic search limit must be between 1 and 20")
        department_filter = department.upper().replace("-", "_") if department else None
        query_vector = self._model().encode(
            ["query: " + query.strip()], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        if department_filter:
            eligible = np.asarray([record["department"] == department_filter for record in self.records])
            scores = np.where(eligible, scores, -np.inf)
        ranked_indices = np.argsort(-scores)[:limit]
        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if not np.isfinite(score) or score < min_similarity:
                continue
            record = self.records[int(index)]
            citation = {
                "title": record["title"], "source_url": record["source_url"] or None,
                "department": record["department"], "file_path": record["file_path"],
                "page": record["page"], "heading": record["heading"] or None,
                "course": record["course"], "policy_date": record["policy_date"],
                "label": record["title"] + (f", p. {record['page']}" if record["page"] else ""),
                "extraction_method": None, "ocr_confidence": None, "ocr_suspicious": None,
                "complete": bool(record["source_url"] and (record["page"] is not None or record["heading"])),
                "document_id": record["document_id"], "chunk_id": record["chunk_id"],
            }
            results.append({
                "evidence": record["content"], "citation": citation,
                "matched_terms": [], "term_coverage": 0.0, "semantic_score": score,
                "retrieval_source": "semantic",
            })
        return results


def reciprocal_rank_fusion(keyword_results: list[dict[str, Any]],
                           semantic_results: list[dict[str, Any]], *,
                           limit: int = 8, rank_constant: int = 60,
                           min_similarity: float = 0.0) -> list[dict[str, Any]]:
    """Fuse lexical and dense rankings while keeping their evidence metadata intact."""
    fused: dict[tuple[str, int], dict[str, Any]] = {}

    def add(items: list[dict[str, Any]], source: str) -> None:
        for rank, item in enumerate(items, 1):
            citation = item["citation"]
            if source == "semantic" and (item.get("semantic_score") or 0.0) < min_similarity:
                continue
            key = (str(citation["document_id"]), int(citation["chunk_id"]))
            existing = fused.get(key)
            contribution = 1.0 / (rank_constant + rank)
            if existing is None:
                existing = dict(item)
                existing["retrieval_sources"] = []
                existing["_rrf_score"] = 0.0
                fused[key] = existing
            existing["_rrf_score"] += contribution
            if source not in existing["retrieval_sources"]:
                existing["retrieval_sources"].append(source)
            if source == "keyword" and item.get("term_coverage", 0) > existing.get("term_coverage", 0):
                existing["matched_terms"] = item.get("matched_terms", [])
                existing["term_coverage"] = item.get("term_coverage", 0)
            if item.get("semantic_score") is not None:
                existing["semantic_score"] = item["semantic_score"]
            if source == "keyword":
                # Preserve the SQLite passage and citation representation when both
                # retrievers found the same row.
                for field in ("evidence", "citation"):
                    existing[field] = item[field]

    add(keyword_results, "keyword")
    add(semantic_results, "semantic")
    ranked = sorted(
        fused.values(),
        key=lambda item: (-item["_rrf_score"], -item.get("term_coverage", 0),
                          -(item.get("semantic_score") or 0.0), item["citation"]["file_path"],
                          item["citation"].get("page") or 0),
    )[:limit]
    for item in ranked:
        item["retrieval_source"] = "+".join(item.pop("retrieval_sources"))
        item.pop("_rrf_score", None)
    return ranked

