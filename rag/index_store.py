"""Persist the policy index on disk, with a manifest, so documents are not re-embedded on every start.

    data/index/policy/
        manifest.json   - embedding backend + model, chunker version, SHA-256 of every document and catalog.json
        chunks.jsonl    - one chunk per line (text + metadata)
        vectors.npy     - the chunk embeddings (float32, L2-normalised)

On start-up the manifest is compared with the current documents and settings. When everything matches, the
chunks and vectors are loaded (no embedding API call); when a document was added, edited or removed - or the
embedding model / chunker changed - the index is rebuilt and saved again. Files are written to temporary names
and swapped in with ``os.replace``, manifest last, so a crash mid-save can never leave a half-written index
that looks valid.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rag.text import CHUNKER_VERSION, Chunk, file_sha256

log = logging.getLogger(__name__)
FORMAT_VERSION = 1


def fingerprint(kb_files: List[Path], kb_dir: Path, backend: str, model: str) -> Dict:
    catalog = kb_dir / "catalog.json"
    return {
        "format": FORMAT_VERSION,
        "embedding_backend": backend,
        "embedding_model": model or "",
        "chunker": CHUNKER_VERSION,
        "files": {p.relative_to(kb_dir).as_posix(): file_sha256(p) for p in kb_files},
        "catalog_sha256": file_sha256(catalog) if catalog.exists() else "",
    }


def _same(a: Dict, b: Dict) -> bool:
    keys = ("format", "embedding_backend", "embedding_model", "chunker", "files", "catalog_sha256")
    return all(a.get(k) == b.get(k) for k in keys)


class IndexStore:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.manifest_file = self.folder / "manifest.json"

    def manifest(self) -> Dict:
        try:
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def load(self, expected: Dict) -> Optional[Tuple[List[Chunk], np.ndarray]]:
        """Chunks + vectors when the saved index matches ``expected``; None when it must be rebuilt."""
        saved = self.manifest()
        if not saved or not _same(saved, expected):
            if saved:
                log.info("Knowledge index on disk is out of date - rebuilding")
            return None
        try:
            chunks = [Chunk(**json.loads(line)) for line in
                      (self.folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            vectors = np.load(self.folder / "vectors.npy")
        except Exception as exc:  # damaged files -> rebuild
            log.warning("Could not load the saved knowledge index (%s) - rebuilding", exc)
            return None
        if len(chunks) != saved.get("chunks") or vectors.shape[0] != len(chunks):
            return None
        return chunks, vectors

    def save(self, chunks: List[Chunk], vectors: Optional[np.ndarray], expected: Dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

        def swap(name: str, write) -> None:
            tmp = self.folder / f"{name}.tmp"
            write(tmp)
            os.replace(tmp, self.folder / name)

        swap("chunks.jsonl", lambda p: p.write_text(
            "\n".join(json.dumps({"id": c.id, "source": c.source, "section": c.section, "text": c.text,
                                  "collection": c.collection, "meta": c.meta}, ensure_ascii=False)
                      for c in chunks) + "\n", encoding="utf-8"))
        if vectors is not None:
            def write_vectors(p: Path) -> None:
                with p.open("wb") as fh:
                    np.save(fh, vectors.astype("float32"))
            swap("vectors.npy", write_vectors)
        manifest = {**expected, "chunks": len(chunks),
                    "dimensions": int(vectors.shape[1]) if vectors is not None and vectors.ndim == 2 else 0,
                    "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        swap("manifest.json", lambda p: p.write_text(json.dumps(manifest, indent=1), encoding="utf-8"))

    def clear(self) -> None:
        """Force a rebuild on the next search (used by the Knowledge Base tab's 'Rebuild index' button)."""
        try:
            self.manifest_file.unlink()
        except OSError:
            pass
