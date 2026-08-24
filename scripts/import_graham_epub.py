from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup


CHAPTER_RE = re.compile(r"^EPUB/text/ch(\d+)\.xhtml$")
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "blockquote", "pre", "li"}


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _slug(title: str) -> str:
    title = re.sub(r"^\d+\s*", "", title).lower()
    title = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    return title or "untitled"


def _chapter_markdown(raw: bytes, number: int) -> tuple[str, str]:
    soup = BeautifulSoup(raw, "html.parser")
    body = soup.body or soup
    heading = body.find("h1") or soup.find("title")
    title = _clean_text(heading.get_text(" ", strip=True)) if heading else f"Chapter {number:03d}"
    lines = [f"# {title}", ""]
    for element in body.find_all(BLOCK_TAGS):
        if element.find_parent(BLOCK_TAGS):
            continue
        text = _clean_text(element.get_text(" ", strip=True))
        if not text or text == title:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            level = int(element.name[1])
            lines.extend([f"{'#' * level} {text}", ""])
        elif element.name == "li":
            lines.extend([f"- {text}", ""])
        elif element.name == "pre":
            lines.extend(["```", text, "```", ""])
        elif element.name == "blockquote":
            lines.extend([f"> {text}", ""])
        else:
            lines.extend([text, ""])
    return title, "\n".join(lines).strip() + "\n"


def import_epub(epub_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(epub_path) as archive:
        entries = sorted(
            (entry for entry in archive.infolist() if CHAPTER_RE.match(entry.filename)),
            key=lambda entry: int(CHAPTER_RE.match(entry.filename).group(1)),
        )
        for entry in entries:
            number = int(CHAPTER_RE.match(entry.filename).group(1))
            title, markdown = _chapter_markdown(archive.read(entry), number)
            target = output_dir / f"{number:03d}_{_slug(title)}.md"
            target.write_text(markdown, encoding="utf-8")
            written.append(target)
    if not written:
        raise ValueError(f"EPUB 中没有找到 EPUB/text/chNNN.xhtml: {epub_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="将 graham-essays release EPUB 转为 data Markdown")
    parser.add_argument("epub", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/graham-essays"))
    args = parser.parse_args()
    written = import_epub(args.epub, args.output)
    print(f"已导入 {len(written)} 篇文章到 {args.output}")
    print(f"首篇: {written[0].name}")
    print(f"末篇: {written[-1].name}")


if __name__ == "__main__":
    main()
