"""Export the reviewed SQLite policy index to a compact BM25 runtime artifact.

This reads SQLite in read-only mode and never changes the source database. The
output contains extracted policy text, so keep it in an ignored local directory
or private deployment storage; never commit it to the public app repository.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, default=Path("deploy_artifacts/production/search.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("web/.data/policy-index.json.gz"))
    parser.add_argument("--expected-documents", type=int, default=24)
    parser.add_argument("--expected-chunks", type=int, default=1160)
    args = parser.parse_args()

    source = args.sqlite.expanduser().resolve()
    if not source.is_file():
        parser.error(f"SQLite index does not exist: {source}")
    uri = "file:" + quote(source.as_posix(), safe="/:_") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise SystemExit(f"SQLite integrity check failed: {integrity}")
        rows = con.execute(
            "SELECT c.rowid AS chunk_id, c.document_id, c.department, c.title, c.heading, "
            "c.file_path, c.source_url, c.page, c.policy_date, c.course, c.content "
            "FROM chunks AS c JOIN documents AS d ON d.document_id = c.document_id "
            "WHERE d.privacy_flags_json = '[]' AND d.processing_status LIKE 'indexed%' "
            "ORDER BY c.rowid"
        ).fetchall()
        document_count = con.execute(
            "SELECT count(*) FROM documents "
            "WHERE privacy_flags_json = '[]' AND processing_status LIKE 'indexed%'"
        ).fetchone()[0]
    finally:
        con.close()

    if document_count != args.expected_documents or len(rows) != args.expected_chunks:
        raise SystemExit(
            f"Refusing export: found {document_count} eligible documents and {len(rows)} chunks; "
            f"expected {args.expected_documents} and {args.expected_chunks}."
        )

    records = [
        {
            "chunk_id": int(row["chunk_id"]),
            "document_id": row["document_id"],
            "department": row["department"],
            "title": row["title"],
            "heading": row["heading"],
            "file_path": row["file_path"],
            "source_url": row["source_url"],
            "page": row["page"],
            "policy_date": row["policy_date"],
            "course": row["course"],
            "content": row["content"],
        }
        for row in rows
    ]
    payload = {
        "schema_version": 1,
        "retrieval": "bm25",
        "source_index_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "document_count": document_count,
        "chunk_count": len(records),
        "records": records,
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as compressed:
                compressed.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)

    print(f"documents={document_count} chunks={len(records)} bytes={output.stat().st_size}")
    print(f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    print(f"artifact={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
