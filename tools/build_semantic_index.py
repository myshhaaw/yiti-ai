"""Build a local semantic vector index from eligible SQLite policy chunks."""
from __future__ import annotations

from yiti_ai.config import ROOT
from yiti_ai.semantic_retrieval import DEFAULT_VECTOR_PATH, build_semantic_index


if __name__ == "__main__":
    result = build_semantic_index(ROOT / "data" / "processed" / "search.sqlite3", DEFAULT_VECTOR_PATH)
    print(f"Indexed {result['records']} public chunks at {result['path']} ({result['dimension']} dimensions).")
