"""Markdown corpus -> chunks with stable ids and provenance metadata.

Each source file starts with YAML front matter (source_title, source_type, game_system, visibility, ...).
Chunking follows the thesis method: structure-aware split by headings, then fixed-size chunks with overlap.
Optional `<!-- page: N -->` markers (emitted by an offline PDF converter) give chunks page numbers.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

NAMESPACE = uuid.UUID("5d7c0f3e-4a51-4a0f-9f1e-2b8e6c1d9a11")
REQUIRED_META = ("source_title", "source_type", "game_system")
SOURCE_TYPES = {"rules", "lore", "scenario", "narrative_example"}
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
PAGE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def payload(self) -> dict:
        # Same layout as langchain-qdrant, so the collection stays usable from LangChain tooling.
        return {"page_content": self.text, "metadata": self.metadata}


def normalize(text: str) -> str:
    """NFC + collapsed whitespace, so ids are identical across OSes and re-conversions."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def chunk_id(source_title: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_title}\x1f{normalize(text)}".encode()).hexdigest()
    return str(uuid.uuid5(NAMESPACE, digest))


def split_front_matter(md: str) -> tuple[dict, str]:
    md = md.replace("\r\n", "\n").lstrip("\ufeff")
    m = FRONT_MATTER.match(md)
    if not m:
        raise ValueError("missing YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise ValueError(f"front matter misses {missing}")
    if meta["source_type"] not in SOURCE_TYPES:
        raise ValueError(f"unknown source_type {meta['source_type']!r}")
    meta.setdefault("visibility", "public")
    return meta, md[m.end():]


def chunk_markdown(md: str, *, chunk_size: int = 1000, overlap: int = 200, source_path: str = "") -> list[Chunk]:
    meta, body = split_front_matter(md)
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(body)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    page: int | None = None
    for section in sections:
        heads = [section.metadata[k] for _, k in HEADERS if section.metadata.get(k)]
        if heads and section.metadata.get("h1") and meta["source_title"].startswith(section.metadata["h1"]):
            heads = heads[1:] or heads  # a top heading that repeats the document title adds nothing
        section_path = " > ".join(heads)
        for piece in splitter.split_text(section.page_content):
            markers = [int(n) for n in PAGE.findall(piece)]
            # page in effect where the chunk starts: a leading marker opens a new page
            start_page = markers[0] if markers and PAGE.match(piece.lstrip()) else page
            if markers:
                page = markers[-1]
            text = PAGE.sub("", piece).strip()
            if not text:
                continue
            cid = chunk_id(meta["source_title"], text)
            if cid in seen:  # identical text in the same source: keep the first occurrence
                continue
            seen.add(cid)
            chunks.append(Chunk(cid, text, {
                "source_title": meta["source_title"],
                "source_type": meta["source_type"],
                "game_system": meta["game_system"],
                "visibility": meta["visibility"],
                "language": meta.get("language", "en"),
                "license": meta.get("license", ""),
                "section_path": section_path,
                "page": start_page,
                "source_path": source_path,
                "chunk_index": len(chunks),
            }))
    return chunks


def load_corpus(paths: list[Path], *, chunk_size: int, overlap: int, root: Path | None = None) -> list[Chunk]:
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])
    chunks: list[Chunk] = []
    for f in files:
        rel = f.relative_to(root).as_posix() if root and f.is_relative_to(root) else f.name
        chunks.extend(chunk_markdown(f.read_text(encoding="utf-8"), chunk_size=chunk_size, overlap=overlap,
                                     source_path=rel))
    return chunks
