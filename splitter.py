from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SECTION_NAMES = {
    "hook",
    "setup",
    "segment",
    "payoff",
    "cta",
    "intro",
    "opening",
    "conclusion",
    "close",
}


METADATA_PREFIXES = (
    "target length:",
    "word count:",
    "estimated runtime",
)


@dataclass(frozen=True)
class SceneChunk:
    index: int
    title: str
    text: str

    @property
    def slug(self) -> str:
        title = re.sub(r"\([^)]*\)", "", self.title)
        title = title.replace("+", " ")
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return slug or "scene"

    @property
    def filename_stem(self) -> str:
        return f"{self.index:02d}_{self.slug}"


def read_script(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_script(text: str, max_words: int = 300) -> list[SceneChunk]:
    sections = _split_by_markdown_headings(text)
    if not sections:
        sections = _split_by_loose_labels(text)
    if not sections:
        sections = [("script", text.strip())]

    chunks: list[SceneChunk] = []
    for title, body in sections:
        cleaned_body = _clean_narration_text(body)
        if not cleaned_body:
            continue
        for part in _chunk_text(cleaned_body, max_words=max_words):
            chunks.append(SceneChunk(index=len(chunks) + 1, title=title, text=part))
    return chunks


def write_chunks(chunks: list[SceneChunk], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for chunk in chunks:
        path = output_dir / f"{chunk.filename_stem}.txt"
        path.write_text(chunk.text + "\n", encoding="utf-8")
        written.append(path)
    return written


def title_from_path(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "video"


def _split_by_markdown_headings(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections: list[tuple[str, str]] = []

    for position, match in enumerate(matches):
        title = _clean_title(match.group(2))
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = _clean_body(text[start:end])
        if body and _looks_like_scene_title(title):
            sections.append((title, body))

    return sections


def _split_by_loose_labels(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"^(?P<title>(?:hook|setup|segment\s*\d+|payoff|cta|intro|opening|conclusion|close)\b.*?)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    sections: list[tuple[str, str]] = []

    for position, match in enumerate(matches):
        title = _clean_title(match.group("title"))
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = _clean_body(text[start:end])
        if body:
            sections.append((title, body))

    return sections


def _chunk_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if len(paragraph_words) > max_words:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_words = 0
            chunks.extend(_split_long_paragraph(paragraph, max_words))
            continue

        if current and current_words + len(paragraph_words) > max_words:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0

        current.append(paragraph)
        current_words += len(paragraph_words)

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > max_words:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
            for index in range(0, len(sentence_words), max_words):
                chunks.append(" ".join(sentence_words[index : index + max_words]).strip())
            continue

        if current and current_words + len(sentence_words) > max_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _looks_like_scene_title(title: str) -> bool:
    normalized = title.lower()
    return any(name in normalized for name in DEFAULT_SECTION_NAMES) or bool(
        re.search(r"\bsegment\s*\d+\b", normalized)
    )


def _clean_title(title: str) -> str:
    title = re.sub(r"^\d+[\).\s-]+", "", title.strip())
    return re.sub(r"\s+", " ", title).strip(" :-") or "scene"


def _clean_body(body: str) -> str:
    lines = [line.rstrip() for line in body.strip().splitlines()]
    return "\n".join(lines).strip()


def _clean_narration_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        normalized = _strip_markdown_emphasis(stripped).lower()

        if not stripped:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"-{3,}", stripped):
            continue
        if stripped.startswith("#"):
            continue
        if any(normalized.startswith(prefix) for prefix in METADATA_PREFIXES):
            continue

        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_markdown_emphasis(text: str) -> str:
    return text.strip().strip("*_").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a narration script into numbered scene text chunks.")
    parser.add_argument("script", type=Path, help="Path to a .md or .txt script.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for chunk .txt files.")
    parser.add_argument("--max-words", type=int, default=300, help="Maximum words per generated chunk.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("output") / title_from_path(args.script) / "chunks"
    chunks = split_script(read_script(args.script), max_words=args.max_words)
    written = write_chunks(chunks, output_dir)

    print(f"Wrote {len(written)} chunks to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()