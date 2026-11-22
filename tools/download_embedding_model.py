"""Download only the approved safe-tensor and tokenizer files for multilingual-e5-small."""
from __future__ import annotations

from huggingface_hub import snapshot_download

from yiti_ai.semantic_retrieval import MODEL_ID, MODEL_PATH_FILE, ROOT


def main() -> None:
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=ROOT / ".cache" / "models",
        allow_patterns=[
            "config.json", "config_sentence_transformers.json", "modules.json",
            "sentence_bert_config.json", "1_Pooling/config.json", "model.safetensors",
            "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.txt", "merges.txt", "added_tokens.json",
        ],
    )
    MODEL_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH_FILE.with_suffix(".tmp")
    temporary.write_text(str(snapshot), encoding="utf-8")
    temporary.replace(MODEL_PATH_FILE)
    print(f"Downloaded {MODEL_ID} model files to project cache.")


if __name__ == "__main__":
    main()
