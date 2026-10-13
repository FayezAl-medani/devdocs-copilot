"""Document corpus loading and chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """Represents a discrete semantic text chunk from a documentation document."""

    id: str
    source: str
    heading: str
    text: str


def _slugify(text: str) -> str:
    """Convert heading text to a URL/ID friendly slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", slug) or "section"


def split_long_text(text: str, max_words: int = 300) -> list[str]:
    """Split a section into sub-chunks of approximately max_words preserving paragraphs."""
    paragraphs = text.strip().split("\n\n")
    chunks: list[str] = []
    current_paras: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        words_in_para = len(para_clean.split())

        if current_word_count + words_in_para > max_words and current_paras:
            chunks.append("\n\n".join(current_paras))
            current_paras = [para_clean]
            current_word_count = words_in_para
        else:
            current_paras.append(para_clean)
            current_word_count += words_in_para

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks or [text.strip()]


def parse_markdown_file(file_path: Path, max_words: int = 300) -> list[Chunk]:
    """Parse a single markdown file into section-based Chunks."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    source = file_path.name
    doc_stem = file_path.stem

    sections: list[tuple[str, list[str]]] = []
    current_heading = doc_stem.replace("__", " / ").replace("-", " ").title()
    current_lines: list[str] = []

    heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_regex.match(line.strip())
        if match:
            # Save accumulated content of previous heading
            if current_lines:
                text_accum = "\n".join(current_lines).strip()
                if text_accum:
                    sections.append((current_heading, [text_accum]))
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        text_accum = "\n".join(current_lines).strip()
        if text_accum:
            sections.append((current_heading, [text_accum]))

    result: list[Chunk] = []
    for heading, text_parts in sections:
        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            continue

        sub_texts = split_long_text(full_text, max_words=max_words)
        heading_slug = _slugify(heading)

        for idx, sub_text in enumerate(sub_texts):
            chunk_id = (
                f"{doc_stem}#{heading_slug}"
                if len(sub_texts) == 1
                else f"{doc_stem}#{heading_slug}#part{idx + 1}"
            )
            result.append(
                Chunk(
                    id=chunk_id,
                    source=source,
                    heading=heading,
                    text=sub_text,
                )
            )

    return result


def load_markdown(directory: Path | str, max_words: int = 300) -> list[Chunk]:
    """Load all markdown documents in a directory and return chunked corpus.

    Parameters
    ----------
    directory : Path or str
        Directory containing .md documentation files.
    max_words : int
        Approximate maximum word threshold before splitting long sections.

    Returns
    -------
    list[Chunk]
        Collection of parsed frozen Chunk instances.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    chunks: list[Chunk] = []
    for file_path in sorted(dir_path.glob("*.md")):
        file_chunks = parse_markdown_file(file_path, max_words=max_words)
        chunks.extend(file_chunks)

    return chunks
