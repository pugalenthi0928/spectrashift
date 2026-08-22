from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvidenceDocument:
    document_id: str
    source: str
    anchor: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.anchor}" if self.anchor else self.source


@dataclass(frozen=True)
class EvidenceHit:
    score: float
    document: EvidenceDocument

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "citation": self.document.citation,
            **asdict(self.document),
        }


def _anchor(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:80]


def _markdown_documents(path: Path, root: Path) -> list[EvidenceDocument]:
    source = path.relative_to(root).as_posix()
    documents: list[EvidenceDocument] = []
    title = path.stem
    section = "overview"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        citation_anchor = _anchor(section)
        documents.append(
            EvidenceDocument(
                document_id=f"{source}#{citation_anchor}",
                source=source,
                anchor=citation_anchor,
                title=f"{title}: {section}",
                text=text,
            )
        )
        buffer.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            flush()
            section = match.group(1)
            if title == path.stem:
                title = section
        else:
            buffer.append(line)
    flush()
    return documents


def _json_pointer(parts: tuple[str, ...]) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def _scalar_lines(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            output.extend(_scalar_lines(item, (*prefix, str(key))))
        return output
    if isinstance(value, list):
        output = []
        for index, item in enumerate(value):
            output.extend(_scalar_lines(item, (*prefix, str(index))))
        return output
    label = ".".join(prefix) if prefix else "value"
    return [f"{label}: {value}"]


def _json_documents(path: Path, root: Path) -> list[EvidenceDocument]:
    source = path.relative_to(root).as_posix()
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents: list[EvidenceDocument] = []

    def visit(value: Any, parts: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            direct_scalars = {
                key: item for key, item in value.items() if not isinstance(item, (dict, list))
            }
            if direct_scalars:
                pointer = _json_pointer(parts) if parts else "/"
                text = "\n".join(_scalar_lines(direct_scalars))
                documents.append(
                    EvidenceDocument(
                        document_id=f"{source}#{pointer}",
                        source=source,
                        anchor=pointer,
                        title=" / ".join((path.stem, *parts)) if parts else path.stem,
                        text=text,
                    )
                )
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    visit(item, (*parts, str(key)))
        elif isinstance(value, list):
            scalar_items = [item for item in value if not isinstance(item, (dict, list))]
            if scalar_items:
                pointer = _json_pointer(parts) if parts else "/"
                documents.append(
                    EvidenceDocument(
                        document_id=f"{source}#{pointer}",
                        source=source,
                        anchor=pointer,
                        title=" / ".join((path.stem, *parts)) if parts else path.stem,
                        text="\n".join(str(item) for item in scalar_items),
                    )
                )
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    visit(item, (*parts, str(index)))
        else:
            pointer = _json_pointer(parts) if parts else "/"
            documents.append(
                EvidenceDocument(
                    document_id=f"{source}#{pointer}",
                    source=source,
                    anchor=pointer,
                    title=" / ".join((path.stem, *parts)) if parts else path.stem,
                    text=str(value),
                )
            )

    visit(payload, ())
    return documents


def load_evidence_documents(root: str | Path) -> list[EvidenceDocument]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"evidence root is not a directory: {root}")
    documents: list[EvidenceDocument] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".md":
            documents.extend(_markdown_documents(path, root))
        elif path.suffix.lower() == ".json":
            documents.extend(_json_documents(path, root))
    documents = [document for document in documents if document.text.strip()]
    if not documents:
        raise ValueError(f"no Markdown or JSON evidence found under {root}")
    return documents


class TfidfEvidenceIndex:
    """Local vector retrieval over immutable experiment evidence."""

    def __init__(self, documents: Iterable[EvidenceDocument]) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("scikit-learn is required; install spectrashift[benchmark]") from exc
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one evidence document is required")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        corpus = [f"{document.title}\n{document.text}" for document in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, *, top_k: int = 5) -> list[EvidenceHit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().reshape(-1)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self.documents))]
        return [
            EvidenceHit(score=float(scores[index]), document=self.documents[int(index)])
            for index in order
            if scores[index] > 0.0
        ]


def query_evidence(root: str | Path, question: str, *, top_k: int = 5) -> dict[str, Any]:
    """Return only retrieved evidence; no language model may alter model outputs."""

    documents = load_evidence_documents(root)
    hits = TfidfEvidenceIndex(documents).search(question, top_k=top_k)
    return {
        "question": question,
        "retrieval_backend": "tfidf-cosine",
        "documents_indexed": len(documents),
        "hits": [hit.to_dict() for hit in hits],
        "answer_policy": (
            "Extractive evidence only. Retrieved records may explain but cannot change predictions, "
            "thresholds, ranks, or evidence grades."
        ),
    }
