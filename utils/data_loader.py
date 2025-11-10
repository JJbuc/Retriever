from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pypdf import PdfReader

from .text_processing import chunk_text
from .types import CorpusArtifacts, Document, DocumentChunk

LOGGER = logging.getLogger(__name__)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted)
    return "\n".join(text_parts)


def compute_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha256.update(block)
    return sha256.hexdigest()


class CorpusBuilder:
    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def discover_documents(self) -> List[Path]:
        pdfs = sorted(self.data_dir.glob("*.pdf"))
        LOGGER.info("Discovered %d pdf files", len(pdfs))
        return pdfs

    def load_documents(self) -> Tuple[List[Document], float]:
        start = time.perf_counter()
        documents: List[Document] = []
        for path in self.discover_documents():
            try:
                text = _read_pdf(path)
            except Exception as exc:  # pragma: no cover - protective logging
                LOGGER.exception("Failed to parse %s: %s", path.name, exc)
                continue
            doc_id = compute_sha256(path)[:12]
            documents.append(
                Document(
                    doc_id=doc_id,
                    source_path=path,
                    text=text,
                    metadata={
                        "filename": path.name,
                        "sha256": compute_sha256(path),
                    },
                )
            )
        elapsed = time.perf_counter() - start
        return documents, elapsed

    def chunk_documents(self, documents: Iterable[Document]) -> Tuple[List[DocumentChunk], float]:
        start = time.perf_counter()
        chunks: List[DocumentChunk] = []
        for doc in documents:
            spans = chunk_text(doc.text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            for idx, span in enumerate(spans):
                chunk_id = f"{doc.doc_id}_{idx:04d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc.doc_id,
                        text=span,
                        metadata={
                            **doc.metadata,
                            "chunk_index": idx,
                            "chunk_size": len(span),
                        },
                    )
                )
        elapsed = time.perf_counter() - start
        return chunks, elapsed

    def build(self) -> CorpusArtifacts:
        documents, load_seconds = self.load_documents()
        chunks, chunk_seconds = self.chunk_documents(documents)
        LOGGER.info(
            "Loaded %d documents and produced %d chunks in %.2fs (chunking %.2fs)",
            len(documents),
            len(chunks),
            load_seconds + chunk_seconds,
            chunk_seconds,
        )
        return CorpusArtifacts(
            documents=documents,
            chunks=chunks,
            load_seconds=load_seconds,
            chunk_seconds=chunk_seconds,
        )


def build_sha_index(paths: Iterable[Path]) -> Dict[str, str]:
    return {path.name: compute_sha256(path) for path in paths}

